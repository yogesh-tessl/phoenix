"""
Vercel sandbox backend.

Session-capable — ``find_or_create_session(session_key)`` returns an
``AsyncSandbox`` handle that is reused across calls keyed by ``session_key``.
``execute()`` remains the direct, no-manager one-shot path (create →
run_command → stop) for callers that do not go through ``SandboxSessionManager``.

Vercel's ``AsyncSandbox.create`` accepts no client-supplied id / name /
metadata / tags, and ``AsyncSandbox.list`` filters only by time range — there
is no provider-native lookup primitive that lets two replicas converge on the
same remote sandbox for a given ``session_key``. So Vercel uses a
**module-level** ``_session_id_map: dict[session_key, sandbox_id]`` shared
across ephemeral ``VercelSandboxBackend`` wrapper instances in the same
process. Cross-replica Vercel binding is deferred (see notes.md in
``_work/cross-replica-deterministic-sandbox-session-reuse``): experiments
are jobs-bound to a single replica today, so process-local binding is
correctness-equivalent to a DB table for the currently-shipping consumer.
A DB-backed mapping can be added later without changing this adapter's
contract — ``find_or_create_session`` is identical.

Requires the ``vercel`` extra (``vercel>=0.5.8``). The SDK import is lazy (in
``VercelSandboxBackend._create_sandbox``) so the module remains importable
when the extra is absent. Adapter availability is gated by
``VercelPythonAdapter.probe_dependencies`` /
``VercelTypescriptAdapter.probe_dependencies`` at registration time, which
surfaces a missing extra as ``status=NOT_INSTALLED`` instead of a runtime
error during evaluation.

Authentication uses the Vercel Sandbox access-token triple, forwarded as
explicit kwargs to ``AsyncSandbox.create``. Phoenix resolves those credentials
from the SDK-native ``VERCEL_TOKEN`` / ``VERCEL_PROJECT_ID`` / ``VERCEL_TEAM_ID``
keys. The SDK's alternative OIDC path is not supported — it relied on
``os.environ`` mutation since the SDK has no ``oidc_token=`` kwarg, and
Phoenix's deployment model (self-hosted server, not a Vercel runtime context)
has no documented OIDC workflow. See
https://vercel.com/docs/vercel-sandbox/concepts/authentication

Language routing
----------------
- PYTHON  → runtime="python3.13", run_command("python3", ["-c", code])
- TYPESCRIPT → runtime="node24", run_command("node", ["--input-type=module-typescript", "-e", code])
"""

from __future__ import annotations

import asyncio
import logging
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence, TypedDict

from starlette.datastructures import Secret

from .types import (
    ExecutionResult,
    ProviderCredentialSpec,
    SandboxAdapter,
    SandboxBackend,
    VercelPythonConfig,
    VercelTypescriptConfig,
    compose_secret_values,
    compute_config_fingerprint,
)

if TYPE_CHECKING:
    from vercel.sandbox import AsyncSandbox


class _LanguageConfig(TypedDict):
    runtime: str
    cmd: str
    args_prefix: list[str]


# Canonical Phoenix credential keys.
ENV_VERCEL_TOKEN = "VERCEL_TOKEN"
ENV_VERCEL_PROJECT_ID = "VERCEL_PROJECT_ID"
ENV_VERCEL_TEAM_ID = "VERCEL_TEAM_ID"

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Language → runtime + command mapping
# ---------------------------------------------------------------------------

_LANGUAGE_CONFIGS: dict[str, _LanguageConfig] = {
    "PYTHON": _LanguageConfig(
        runtime="python3.13",
        cmd="python3",
        args_prefix=["-c"],
    ),
    "TYPESCRIPT": _LanguageConfig(
        runtime="node24",
        cmd="node",
        args_prefix=["--input-type=module-typescript", "-e"],
    ),
}
_DEFAULT_LANGUAGE = "TYPESCRIPT"

# Max-lifetime ceiling for Vercel sandboxes. Vercel's SDK accepts an int
# (interpreted as **milliseconds**) or a ``timedelta``; we pass a ``timedelta``
# to keep the unit unambiguous at the call site. 600 s == 10 min, well under
# Vercel's 45 min hard cap and above the SDK's 5 min default.
_VERCEL_CREATE_TIMEOUT = timedelta(seconds=600)

# ---------------------------------------------------------------------------
# Process-local Vercel session binding.
#
# Vercel's ``CreateSandboxRequest`` accepts no client-supplied id / name /
# metadata / tags, and ``AsyncSandbox.list`` filters only by time range — so
# unlike Modal (name namespace), E2B (metadata list), or Daytona (label list),
# Vercel has no provider-native primitive that lets two Phoenix replicas
# converge on a single remote sandbox for a given ``session_key``. We bind
# ``session_key -> sandbox_id`` in a **module-level** dict that is shared
# across every ephemeral ``VercelSandboxBackend`` wrapper in the same process,
# guarded by Phoenix's canonical per-key asyncio.Lock pattern (lazy-insert +
# double-checked locking) so concurrent same-key callers serialize on the
# slot.
#
# Cross-replica Vercel binding is deferred per
# ``_work/cross-replica-deterministic-sandbox-session-reuse/notes.md``:
# experiments are jobs-bound to a single replica today, so this is
# correctness-equivalent to a DB table for the currently-shipping consumer.
# When a DB-backed mapping ships, it slots in behind ``find_or_create_session``
# without changing the adapter contract.
# ---------------------------------------------------------------------------

_session_id_map: dict[str, str] = {}
_session_id_locks: dict[str, asyncio.Lock] = {}
_global_lock = asyncio.Lock()


async def _get_key_lock(session_key: str) -> asyncio.Lock:
    """Return the per-key asyncio.Lock for ``session_key``, creating on miss.

    Lazy-insert under the module-level ``_global_lock`` so two concurrent
    first-time callers cannot each install a different Lock and end up
    holding non-equivalent slots. Pairs with ``_session_id_map``: callers
    that pop the map entry on close also pop the lock entry to bound
    ``_session_id_locks`` growth.
    """
    async with _global_lock:
        return _session_id_locks.setdefault(session_key, asyncio.Lock())


# Vercel statuses that mean the sandbox is no longer usable. ``stop`` is a
# user-visible terminal state; ``failed``/``aborted`` are SDK-emitted error
# terminals; ``snapshotting`` is a transitional state where the sandbox is
# being stopped as a side effect of snapshot creation. ``pending``/``running``
# are the live states we will accept on reconnect. ``stopping`` is treated as
# stale: the sandbox is on its way down and reusing it races the teardown.
_VERCEL_DEAD_STATUSES: frozenset[str] = frozenset(
    {"stopped", "stopping", "failed", "aborted", "snapshotting"}
)


class VercelSandboxBackend(SandboxBackend):
    """Sandbox backend executing code via Vercel Sandbox (vercel >= 0.5.8).

    Session reuse is bound in a **process-local** ``_session_id_map`` keyed by
    the opaque ``session_key`` (see module-level comment for the cross-replica
    caveat). ``find_or_create_session`` reconnects via
    ``AsyncSandbox.get(sandbox_id=...)`` when a mapping exists and the remote
    sandbox is alive; on miss or stale, it calls ``AsyncSandbox.create(...)``
    and records ``sandbox.sandbox_id`` under the same per-key lock.

    Credentials: the access-token triple ``token``/``project_id``/``team_id``
    is forwarded directly to ``AsyncSandbox.create`` / ``AsyncSandbox.get`` as
    explicit kwargs. No ``os.environ`` mutation.

    Network policy: pass ``internet_access`` as ``True`` (allow-all),
    ``False`` (deny-all), or ``None`` (omit — let the SDK default apply).
    The string form is forwarded to ``AsyncSandbox.create(network_policy=)``;
    the SDK accepts ``"allow-all"`` / ``"deny-all"`` and converts internally.
    """

    def __init__(
        self,
        *,
        token: Secret,
        project_id: Secret,
        team_id: Secret,
        language: str = _DEFAULT_LANGUAGE,
        user_env: Optional[Mapping[str, str]] = None,
        packages: Optional[Sequence[str]] = None,
        internet_access: Optional[bool] = None,
    ) -> None:
        if not token or not project_id or not team_id:
            raise ValueError("VercelSandboxBackend requires token, project_id, and team_id.")
        self._token = token
        self._project_id = project_id
        self._team_id = team_id
        self._language = language.upper() if language else _DEFAULT_LANGUAGE
        self._user_env: dict[str, str] = dict(user_env or {})
        self._packages: list[str] = list(packages) if packages else []
        self._internet_access = internet_access
        # Session bindings live in the module-level ``_session_id_map`` so they
        # survive across ephemeral ``VercelSandboxBackend`` wrappers; the per-
        # backend ``_sessions`` map of older versions is gone.
        self.secret_values = compose_secret_values(
            user_env,
            self._token,
            self._project_id,
            self._team_id,
        )

    def config_fingerprint(self) -> str:
        if self._internet_access is None:
            ia_mode: Optional[str] = None
        else:
            ia_mode = "allow" if self._internet_access else "deny"
        return compute_config_fingerprint(
            family="VERCEL",
            packages=self._packages,
            internet_access_mode=ia_mode,
            env_var_keys=list(self._user_env.keys()),
            extra={"language": self._language},
        )

    def _lang_cfg(self) -> _LanguageConfig:
        return _LANGUAGE_CONFIGS.get(self._language, _LANGUAGE_CONFIGS[_DEFAULT_LANGUAGE])

    def _network_policy(self) -> Optional[str]:
        """Map ``internet_access`` (True/False/None) to the SDK string form.

        Returned values: ``"allow-all"``, ``"deny-all"``, or ``None`` to omit
        the kwarg entirely (SDK default applies). The Vercel SDK accepts these
        string aliases on ``AsyncSandbox.create(network_policy=)`` as of 0.5.8.
        """
        if self._internet_access is None:
            return None
        return "allow-all" if self._internet_access else "deny-all"

    async def _create_sandbox(self) -> AsyncSandbox:
        from vercel.sandbox import AsyncSandbox

        runtime: str = self._lang_cfg()["runtime"]
        create_kwargs: dict[str, Any] = {
            "runtime": runtime,
            "token": str(self._token),
            "project_id": str(self._project_id),
            "team_id": str(self._team_id),
            # Hard-coded max-lifetime ceiling so a hard-crashed Phoenix
            # process cannot leak provider-side sandboxes indefinitely. The
            # Vercel SDK interprets int ``timeout`` as milliseconds; we pass
            # a ``timedelta`` to make the unit unambiguous at the call site.
            "timeout": _VERCEL_CREATE_TIMEOUT,
        }
        network_policy = self._network_policy()
        if network_policy is not None:
            create_kwargs["network_policy"] = network_policy
        return await AsyncSandbox.create(**create_kwargs)

    async def _get_sandbox(self, sandbox_id: str) -> AsyncSandbox:
        """Reconnect to an existing Vercel sandbox by id, honoring this
        backend's resolved credentials.

        The SDK's ``AsyncSandbox.get`` constructs a fresh ops client from the
        provided credentials, so passing them explicitly mirrors the
        no-os.environ-mutation invariant of ``_create_sandbox``.
        """
        from vercel.sandbox import AsyncSandbox

        return await AsyncSandbox.get(
            sandbox_id=sandbox_id,
            token=str(self._token),
            project_id=str(self._project_id),
            team_id=str(self._team_id),
        )

    @staticmethod
    def _is_alive(sandbox: AsyncSandbox) -> bool:
        """Return True if ``sandbox`` is in a usable (live) state.

        Vercel's ``SandboxStatus`` enum values are exposed as lowercase
        strings (``"running"``, ``"stopped"``, ...). We compare on the string
        form so the check is robust to the StrEnum representation. Anything
        in ``_VERCEL_DEAD_STATUSES`` is treated as stale.
        """
        try:
            status = str(sandbox.status)
        except Exception:
            return False
        return status not in _VERCEL_DEAD_STATUSES

    async def _install_packages(self, sandbox: AsyncSandbox) -> None:
        """Run language-routed install for configured packages before user code.

        PYTHON → `python3 -m pip install --user <pkgs>` (invoked through the
        same `python3` binary the exec path uses, so install and execute target
        the same interpreter; `--user` avoids needing sudo and writes to the
        sandbox user's ~/.local).
        TYPESCRIPT → `npm install <pkgs>` from the default cwd.

        Raises RuntimeError(stderr) on non-zero exit so callers (start_session
        and ephemeral execute) can either propagate as a fail-fast session
        startup error or surface as an ExecutionResult.error.
        """
        if not self._packages:
            return
        if self._language == "PYTHON":
            cmd = "python3"
            args = ["-m", "pip", "install", "--user", *self._packages]
        else:
            cmd = "npm"
            args = ["install", *self._packages]
        result = await sandbox.run_command(cmd, args)
        if result.exit_code != 0:
            stderr = await result.stderr()
            raise RuntimeError(
                f"{cmd} install {self._packages!r} failed (exit {result.exit_code}): {stderr}"
            )

    async def find_or_create_session(self, session_key: str) -> AsyncSandbox:
        """Return an ``AsyncSandbox`` bound to ``session_key`` (process-local).

        Idempotent under the same ``session_key`` within a single Phoenix
        process: a second call returns the same remote sandbox via
        ``AsyncSandbox.get`` reconnect. On stale (dead status, or any error
        from ``get``) the binding is dropped and a fresh sandbox is created.
        Cross-process / cross-replica convergence is **not** provided — see
        the module-level docstring.
        """
        key_lock = await _get_key_lock(session_key)
        async with key_lock:
            existing_id = _session_id_map.get(session_key)
            if existing_id is not None:
                try:
                    sandbox = await self._get_sandbox(existing_id)
                except Exception:
                    logger.debug(
                        "Vercel find_or_create_session: get(sandbox_id=%s) failed "
                        "for key=%r; treating as stale and recreating",
                        existing_id,
                        session_key,
                        exc_info=True,
                    )
                    sandbox = None
                if sandbox is not None and self._is_alive(sandbox):
                    logger.debug(f"Vercel session '{session_key}' reused")
                    return sandbox
                # Stale binding — drop it so the create-below path can claim
                # a fresh sandbox_id atomically under the same per-key lock.
                _session_id_map.pop(session_key, None)

            sandbox = await self._create_sandbox()
            try:
                await self._install_packages(sandbox)
            except Exception:
                # Install failed — the sandbox is live but unusable. Stop and
                # close it before re-raising so we don't leak a billable
                # Vercel resource that lingers until the SDK's idle timeout.
                try:
                    await sandbox.stop()
                except Exception:
                    logger.debug(
                        f"Error stopping Vercel sandbox after install failure for "
                        f"session '{session_key}'",
                        exc_info=True,
                    )
                try:
                    await sandbox.client.aclose()
                except Exception:
                    logger.debug(
                        f"Error closing Vercel client after install failure for "
                        f"session '{session_key}'",
                        exc_info=True,
                    )
                raise
            _session_id_map[session_key] = sandbox.sandbox_id
            logger.debug(
                "Vercel session '%s' created (sandbox_id=%s)",
                session_key,
                sandbox.sandbox_id,
            )
            return sandbox

    async def execute_in_session(
        self,
        handle: object,
        code: str,
        timeout: Optional[int] = None,
    ) -> ExecutionResult:
        # ``handle`` is the AsyncSandbox returned by find_or_create_session;
        # the type is opaque at the ABC level but always a vercel AsyncSandbox
        # here.
        sandbox: AsyncSandbox = handle  # type: ignore[assignment]
        try:
            session_env: Optional[dict[str, str]] = self._user_env or None
            return await self._exec_code(sandbox, code, env=session_env)
        except Exception as exc:
            return ExecutionResult(stdout="", stderr=str(exc), error=str(exc))

    async def close_session(self, session_key: str) -> None:
        """Release the binding for ``session_key`` and best-effort stop the
        remote sandbox.

        Pop-before-await invariant: the ``_session_id_map`` entry and the
        per-key lock entry are popped synchronously under the per-key lock,
        before the first ``await`` on a remote operation, so a concurrent
        same-key ``find_or_create_session`` cannot overlap a teardown in
        progress on the same slot.
        """
        key_lock = await _get_key_lock(session_key)
        async with key_lock:
            sandbox_id = _session_id_map.pop(session_key, None)
            # Pop the per-key lock under the same critical section so
            # ``_session_id_locks`` does not grow unboundedly across the
            # frontend's per-mount UUID churn.
            _session_id_locks.pop(session_key, None)
            if sandbox_id is None:
                return
            try:
                sandbox = await self._get_sandbox(sandbox_id)
            except Exception:
                logger.debug(
                    "Vercel close_session: get(sandbox_id=%s) failed for key=%r; "
                    "treating as already-absent",
                    sandbox_id,
                    session_key,
                    exc_info=True,
                )
                return
            try:
                await sandbox.stop()
            except Exception:
                logger.debug(
                    "Vercel close_session: stop failed for sandbox_id=%s key=%r",
                    sandbox_id,
                    session_key,
                    exc_info=True,
                )
            try:
                await sandbox.client.aclose()
            except Exception:
                logger.debug(
                    "Vercel close_session: client.aclose failed for sandbox_id=%s key=%r",
                    sandbox_id,
                    session_key,
                    exc_info=True,
                )
            logger.debug(f"Stopped Vercel session '{session_key}'")

    async def _exec_code(
        self,
        sandbox: AsyncSandbox,
        code: str,
        env: Optional[dict[str, str]] = None,
    ) -> ExecutionResult:
        """Run code in a sandbox and collect stdout/stderr."""
        lang_cfg = self._lang_cfg()
        cmd: str = lang_cfg["cmd"]
        args: list[str] = lang_cfg["args_prefix"] + [code]
        result = await sandbox.run_command(cmd, args, env=env)
        stdout, stderr = await asyncio.gather(result.stdout(), result.stderr())
        exit_code = result.exit_code
        error: Optional[str] = stderr if exit_code != 0 else None
        return ExecutionResult(
            stdout=stdout or "",
            stderr=stderr or "",
            error=error,
        )

    async def execute(
        self,
        code: str,
        session_key: str,
        timeout: Optional[int] = None,
    ) -> ExecutionResult:
        # Direct one-shot path (no manager mediation). The session_key is
        # accepted for ABC parity but not consulted: this path always runs
        # ephemeral (create → install → exec → stop) so two direct callers
        # holding distinct wrapper instances don't accidentally share a
        # remote sandbox via the module-level map. Manager-mediated reuse
        # goes through ``find_or_create_session`` + ``execute_in_session``.
        try:
            session_env: Optional[dict[str, str]] = self._user_env or None
            sandbox = await self._create_sandbox()
            try:
                await self._install_packages(sandbox)
                return await self._exec_code(sandbox, code, env=session_env)
            finally:
                try:
                    await sandbox.stop()
                    await sandbox.client.aclose()
                except Exception:
                    pass
        except Exception as exc:
            return ExecutionResult(stdout="", stderr=str(exc), error=str(exc))

    async def close(self) -> None:
        # Session bindings live in the module-level ``_session_id_map``, which
        # outlives any single wrapper instance — the manager calls
        # ``close_session`` per tracked key during shutdown. Nothing to drain
        # at the wrapper level.
        return None


def _resolve_internet_access(config: Mapping[str, Any]) -> Optional[bool]:
    """Project an InternetAccessConfig mode onto a tri-state bool/None.

    ``None`` means the config did not specify internet_access at all — let
    the Vercel SDK default apply (i.e., omit network_policy=). ``True`` →
    "allow-all", ``False`` → "deny-all"; the string mapping lives in
    ``VercelSandboxBackend._network_policy``.
    """
    internet_access = config.get("internet_access")
    if internet_access is None:
        return None
    mode = (
        internet_access.get("mode")
        if isinstance(internet_access, dict)
        else getattr(internet_access, "mode", None)
    )
    if mode == "deny":
        return False
    if mode == "allow":
        return True
    return None


# Linked from credential descriptions and authentication error messages so
# users hitting either surface have a direct pointer to provisioning steps.
_VERCEL_AUTH_DOCS_URL = "https://vercel.com/docs/vercel-sandbox/concepts/authentication"

_VERCEL_ENV_VAR_SPECS = [
    ProviderCredentialSpec(
        key=ENV_VERCEL_TOKEN,
        display_name="Vercel Access Token",
        description=f"Vercel personal access token. See {_VERCEL_AUTH_DOCS_URL}",
    ),
    ProviderCredentialSpec(
        key=ENV_VERCEL_PROJECT_ID,
        display_name="Vercel Project ID",
        description=f"Vercel project ID. See {_VERCEL_AUTH_DOCS_URL}",
    ),
    ProviderCredentialSpec(
        key=ENV_VERCEL_TEAM_ID,
        display_name="Vercel Team ID",
        description=(
            "Vercel team ID — find it under Team Settings → General "
            "(https://vercel.com/teams/your_team_name_here/settings#team-id). "
            f"See {_VERCEL_AUTH_DOCS_URL}"
        ),
    ),
]


def _probe_vercel_sdk() -> None:
    """Verify ``vercel.sandbox`` is installed; ImportError → NOT_INSTALLED."""
    import vercel.sandbox  # noqa: F401


def _build_vercel_backend(
    config: Mapping[str, Any],
    *,
    language: str,
    user_env: Optional[Mapping[str, str]] = None,
) -> SandboxBackend:
    """Construct a VercelSandboxBackend from a resolved config + user_env.

    All three access-token fields must be populated. Raises ``ValueError``
    when any is missing so the executor surfaces an actionable error.
    """
    token = str(config.get(ENV_VERCEL_TOKEN) or "")
    project_id = str(config.get(ENV_VERCEL_PROJECT_ID) or "")
    team_id = str(config.get(ENV_VERCEL_TEAM_ID) or "")
    if not (token and project_id and team_id):
        raise ValueError(
            "Vercel sandbox authentication is not configured. Set "
            "VERCEL_TOKEN, VERCEL_PROJECT_ID, "
            "and VERCEL_TEAM_ID via setSandboxCredential. "
            f"See {_VERCEL_AUTH_DOCS_URL}"
        )
    deps = config.get("dependencies") or {}
    packages: list[str] = deps.get("packages", []) if isinstance(deps, dict) else []
    internet_access = _resolve_internet_access(config)
    return VercelSandboxBackend(
        token=Secret(token),
        project_id=Secret(project_id),
        team_id=Secret(team_id),
        language=language,
        user_env=user_env,
        packages=packages,
        internet_access=internet_access,
    )


class VercelPythonAdapter(SandboxAdapter):
    key = "VERCEL_PYTHON"
    family = "VERCEL"
    display_name = "Vercel"
    language = "PYTHON"
    config_model = VercelPythonConfig
    credential_specs = _VERCEL_ENV_VAR_SPECS

    @classmethod
    def probe_dependencies(cls) -> None:
        _probe_vercel_sdk()

    def build_backend(
        self,
        config: Mapping[str, Any],
        user_env: Optional[Mapping[str, str]] = None,
    ) -> SandboxBackend:
        self._enforce_capabilities(config, user_env)
        return _build_vercel_backend(config, language="PYTHON", user_env=user_env)


class VercelTypescriptAdapter(SandboxAdapter):
    key = "VERCEL_TYPESCRIPT"
    family = "VERCEL"
    display_name = "Vercel"
    language = "TYPESCRIPT"
    config_model = VercelTypescriptConfig
    credential_specs = _VERCEL_ENV_VAR_SPECS

    @classmethod
    def probe_dependencies(cls) -> None:
        _probe_vercel_sdk()

    def build_backend(
        self,
        config: Mapping[str, Any],
        user_env: Optional[Mapping[str, str]] = None,
    ) -> SandboxBackend:
        self._enforce_capabilities(config, user_env)
        return _build_vercel_backend(config, language="TYPESCRIPT", user_env=user_env)
