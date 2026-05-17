"""
Daytona sandbox backend.

Requires the ``daytona_sdk`` package (optional extra). Imports of the SDK are
lazy (in ``DaytonaSandboxBackend._get_client`` and ``execute``) so the module
remains importable when the extra is absent. Adapter availability is gated by
``DaytonaPythonAdapter.probe_dependencies`` /
``DaytonaTypescriptAdapter.probe_dependencies`` at registration time, which
surfaces a missing extra as ``status=NOT_INSTALLED`` instead of a runtime
error during evaluation.

Language routing
----------------
- PYTHON     → CreateSandboxFromSnapshotParams(language=CodeLanguage.PYTHON),
               install via subprocess.run([sys.executable, "-m", "pip", "install", ...])
- TYPESCRIPT → CreateSandboxFromSnapshotParams(language=CodeLanguage.TYPESCRIPT),
               install via node:child_process spawnSync("npm", ["install", ...])

Session reuse is bound provider-side via the sandbox label
``phoenix_session_key``. ``find_or_create_session`` lists sandboxes filtered
on that label, ``client.get(...)``s the oldest alive match, or creates a new
sandbox tagged with the label when none is found. Two Phoenix replicas
asking for the same ``session_key`` against Daytona therefore converge on a
single remote sandbox.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence

from starlette.datastructures import Secret

from .types import (
    DaytonaPythonConfig,
    DaytonaTypescriptConfig,
    ExecutionResult,
    ProviderCredentialSpec,
    SandboxAdapter,
    SandboxBackend,
    compose_secret_values,
    compute_config_fingerprint,
)

if TYPE_CHECKING:
    from daytona_sdk import (
        AsyncDaytona,
        AsyncSandbox,
        CreateSandboxFromSnapshotParams,
        ExecuteResponse,
    )

logger = logging.getLogger(__name__)


def _to_execution_result(response: ExecuteResponse) -> ExecutionResult:
    """Map daytona ExecuteResponse (combined stdout/stderr in `result`) to ExecutionResult."""
    output = response.result or ""
    failed = response.exit_code != 0
    return ExecutionResult(
        stdout="" if failed else output,
        stderr=output if failed else "",
        error=output or f"exit code {response.exit_code}" if failed else None,
    )


_DEFAULT_LANGUAGE = "PYTHON"

# Label key used to bind a Phoenix ``session_key`` to its Daytona sandbox.
# Daytona accepts arbitrary string values in labels, so the opaque session_key
# is stored as-is (no transform).
_LABEL_SESSION_KEY = "phoenix_session_key"

# Hard-coded provider-side TTL kwargs. Units are minutes (per Daytona SDK
# docstring on ``CreateSandboxBaseParams``). ``auto_stop_interval=5`` stops
# the sandbox after 5 idle minutes; ``auto_archive_interval=15`` archives a
# stopped sandbox after 15 minutes. The pair caps how long a leaked sandbox
# can keep accruing billed minutes after a hard Phoenix crash.
_AUTO_STOP_INTERVAL_MIN = 5
_AUTO_ARCHIVE_INTERVAL_MIN = 15


class DaytonaSandboxBackend(SandboxBackend):
    """Sandbox backend executing code in Daytona workspaces.

    Language routing is driven by ``language`` (PYTHON | TYPESCRIPT). The default
    preserves binary compatibility with every existing call site that does not
    pass ``language=`` explicitly. PYTHON routes ``CreateSandboxFromSnapshotParams``
    to ``CodeLanguage.PYTHON`` and installs packages via ``pip``; TYPESCRIPT
    routes to ``CodeLanguage.TYPESCRIPT`` and installs via ``npm``.

    Session reuse is bound provider-side on the ``phoenix_session_key`` label —
    see module docstring for the convergence guarantee. Ephemeral one-shot
    execution via ``execute()`` spins up a fresh sandbox per call and deletes
    it on exit.
    """

    def __init__(
        self,
        api_key: Secret,
        server_url: str = "",
        user_env: Optional[Mapping[str, str]] = None,
        packages: Optional[Sequence[str]] = None,
        network_block_all: bool = False,
        language: str = _DEFAULT_LANGUAGE,
    ) -> None:
        self._api_key = api_key
        self._server_url = server_url
        self._user_env: dict[str, str] = dict(user_env or {})
        self._packages: list[str] = list(packages) if packages else []
        self._network_block_all = network_block_all
        self._language = language.upper() if language else _DEFAULT_LANGUAGE
        self._client: Optional[AsyncDaytona] = None
        self.secret_values = compose_secret_values(user_env, self._api_key)

    def config_fingerprint(self) -> str:
        return compute_config_fingerprint(
            family="DAYTONA",
            packages=self._packages,
            internet_access_mode="deny" if self._network_block_all else "allow",
            env_var_keys=list(self._user_env.keys()),
            extra={"language": self._language},
        )

    def _get_client(self) -> AsyncDaytona:
        if self._client is not None:
            return self._client
        from daytona_sdk import AsyncDaytona, DaytonaConfig

        self._client = AsyncDaytona(
            DaytonaConfig(
                api_key=str(self._api_key),
                api_url=self._server_url or None,
            )
        )
        return self._client

    async def _install_packages(self, workspace: AsyncSandbox) -> None:
        """Run language-routed install for configured packages before first user execute.

        PYTHON     → ``subprocess.run([sys.executable, '-m', 'pip', 'install', *pkgs])``
                     argv-style call from a small generated Python snippet.
        TYPESCRIPT → ``spawnSync('npm', ['install', ...pkgs])`` from
                     ``node:child_process`` — package names embedded as a JSON
                     array literal, NOT shell-string interpolation.

        Raises ``RuntimeError`` on non-zero exit so callers (find_or_create_session
        create path and ephemeral execute) propagate the failure.
        """
        if not self._packages:
            return
        if self._language == "TYPESCRIPT":
            packages_json = json.dumps(self._packages)
            # Install into /tmp/node_modules so the package is resolvable
            # from subsequent code_run invocations. Daytona's TS workspace
            # writes each code_run snippet to /tmp/dtn_*.ts and executes it
            # there; Node's require resolver walks /tmp/node_modules first
            # (verified via require.resolve.paths from a live workspace).
            # Default cwd is /home/daytona, and `npm install -g` lands in
            # nvm's lib/node_modules which is NOT on the legacy GLOBAL_FOLDERS
            # lookup path (Node searches lib/node, singular). Pinning cwd
            # to /tmp puts the package exactly where the resolver looks.
            install_code = (
                f"const {{ spawnSync }} = require('node:child_process');\n"
                f"const pkgs = {packages_json};\n"
                f"const r = spawnSync('npm', "
                f"['install', '--no-audit', '--no-fund', '--silent', ...pkgs], "
                f"{{ cwd: '/tmp', encoding: 'utf8' }});\n"
                f"if (r.status !== 0) {{\n"
                f"  throw new Error("
                f"`npm install failed (status=${{r.status}}): "
                f"${{r.stderr || r.stdout}}`);\n"
                f"}}\n"
            )
        else:
            install_code = (
                f"import subprocess, sys\n"
                f"r = subprocess.run(\n"
                f"    [sys.executable, '-m', 'pip', 'install', *{self._packages!r}],\n"
                f"    capture_output=True, text=True\n"
                f")\n"
                f"if r.returncode != 0:\n"
                f"    raise RuntimeError(r.stderr)\n"
            )
        result = await workspace.process.code_run(install_code)
        if result.exit_code != 0:
            tool = "npm" if self._language == "TYPESCRIPT" else "pip"
            raise RuntimeError(
                f"{tool} install {self._packages!r} failed "
                f"(exit {result.exit_code}): {result.result}"
            )

    def _create_params(self, session_key: Optional[str] = None) -> CreateSandboxFromSnapshotParams:
        """Build params for ``client.create()``.

        When ``session_key`` is supplied (the session-reuse path), it is added
        under ``_LABEL_SESSION_KEY`` so the sandbox can be rediscovered via
        ``client.list(labels=...)``. TTL kwargs (``auto_stop_interval``,
        ``auto_archive_interval``) are always set so provider-side reclamation
        bounds leaked sandboxes after a hard Phoenix crash.
        """
        from daytona_sdk import CodeLanguage, CreateSandboxFromSnapshotParams

        code_language = (
            CodeLanguage.TYPESCRIPT if self._language == "TYPESCRIPT" else CodeLanguage.PYTHON
        )
        labels: Optional[dict[str, str]] = None
        if session_key is not None:
            labels = {_LABEL_SESSION_KEY: session_key}
        return CreateSandboxFromSnapshotParams(
            language=code_language,
            network_block_all=True if self._network_block_all else None,
            labels=labels,
            auto_stop_interval=_AUTO_STOP_INTERVAL_MIN,
            auto_archive_interval=_AUTO_ARCHIVE_INTERVAL_MIN,
        )

    async def _list_sandboxes_for_key(self, session_key: str) -> list[AsyncSandbox]:
        """Return all Daytona sandboxes tagged with this ``session_key``.

        The label filter is server-side; ``client.list`` returns a paginated
        result whose ``items`` we return directly. In steady state at most one
        sandbox carries a given key.
        """
        client = self._get_client()
        response = await client.list(labels={_LABEL_SESSION_KEY: session_key})
        return list(response.items)

    async def _is_alive(self, sandbox: AsyncSandbox) -> bool:
        """Best-effort alive probe for an attached sandbox handle.

        Treats only the ``STARTED`` state as alive: ``STOPPED``/``ARCHIVED``
        require a restart and ``ERROR``/``DESTROYED`` are unrecoverable. Any
        unexpected exception is treated as not-alive so the caller falls
        through to the create path rather than surfacing transient SDK errors
        to evaluator callers.
        """
        try:
            from daytona_api_client_async.models.sandbox_state import SandboxState

            return sandbox.state == SandboxState.STARTED
        except Exception as exc:
            logger.debug(
                "Daytona alive probe failed for sandbox_id=%s: %s",
                getattr(sandbox, "id", "<unknown>"),
                exc,
            )
            return False

    async def find_or_create_session(self, session_key: str) -> object:
        """Bind ``session_key`` to a single remote Daytona sandbox.

        Convergence path: list by ``phoenix_session_key`` label; if an alive
        sandbox already exists, ``client.get`` it and return. On miss (or if
        the candidate fails the alive probe / the get raises), ``client.create``
        a fresh sandbox tagged with the label, install configured packages,
        and return. The returned ``AsyncSandbox`` is the opaque handle passed
        back to ``execute_in_session``.

        Install runs ONLY on the create path. Reusing an existing sandbox MUST
        NOT re-install packages — the existing sandbox already has its
        packages from its own create call.
        """
        client = self._get_client()

        try:
            existing = await self._list_sandboxes_for_key(session_key)
        except Exception as exc:
            # List failures fall through to create so a transient API hiccup
            # does not block the evaluator. The provider's TTL kwargs reap
            # any duplicate that this branch leaves behind.
            logger.debug(
                "Daytona list failed for session_key=%r; falling through to create: %s",
                session_key,
                exc,
            )
            existing = []

        if existing:
            # Pick the oldest by id as a deterministic choice across replicas.
            # Daytona ids are sortable strings; this gives both replicas the
            # same survivor when more than one candidate is returned.
            existing.sort(key=lambda sb: getattr(sb, "id", ""))
            oldest = existing[0]
            sandbox_id = getattr(oldest, "id", None)
            if sandbox_id:
                try:
                    sandbox = await client.get(sandbox_id)
                except Exception as exc:
                    # Stale handle (sandbox died between list and get, or the
                    # get raised DaytonaNotFoundError because the sandbox was
                    # reaped). Fall through to create.
                    logger.debug(
                        "Daytona get failed for session_key=%r sandbox_id=%s; "
                        "falling through to create: %s",
                        session_key,
                        sandbox_id,
                        exc,
                    )
                else:
                    if await self._is_alive(sandbox):
                        logger.debug(
                            "Daytona session reuse: attached to sandbox_id=%s for key=%r",
                            sandbox_id,
                            session_key,
                        )
                        return sandbox
                    logger.debug(
                        "Daytona sandbox_id=%s for key=%r failed alive probe; creating a fresh one",
                        sandbox_id,
                        session_key,
                    )

        sandbox = await client.create(self._create_params(session_key=session_key))
        # Install-on-create only — never on attach.
        await self._install_packages(sandbox)
        logger.debug(
            "Daytona session create: sandbox_id=%s for key=%r",
            getattr(sandbox, "id", "<unknown>"),
            session_key,
        )

        # Post-create dedup: another replica may have raced us. Re-list, pick
        # the oldest survivor by ``id``, delete the rest. Sorting by ``id`` is
        # deterministic across replicas, so both sides agree on the survivor
        # without coordination.
        return await self._dedupe_after_create(session_key, sandbox)

    async def _dedupe_after_create(
        self,
        session_key: str,
        just_created: AsyncSandbox,
    ) -> AsyncSandbox:
        """Resolve post-create races for ``session_key``.

        If only one sandbox is tagged with the key, return the one we just
        created. If multiple exist, keep the oldest by ``id`` and delete the
        rest — including ``just_created`` if a competing replica created an
        earlier one. Failures to delete a loser are logged and swallowed so a
        transient API hiccup does not block the evaluator return; the
        provider's TTL kwargs reap any duplicate this branch leaves behind.
        """
        client = self._get_client()
        try:
            candidates = await self._list_sandboxes_for_key(session_key)
        except Exception as exc:
            logger.warning(
                "Daytona dedup: re-list failed for key=%r; keeping just-created sandbox_id=%s: %s",
                session_key,
                getattr(just_created, "id", "<unknown>"),
                exc,
            )
            return just_created
        if len(candidates) <= 1:
            return just_created
        candidates.sort(key=lambda sb: getattr(sb, "id", ""))
        survivor = candidates[0]
        just_created_id = getattr(just_created, "id", None)
        if getattr(survivor, "id", None) == just_created_id:
            survivor = just_created
        for loser in candidates[1:]:
            try:
                await client.delete(loser)
            except Exception as exc:
                logger.warning(
                    "Daytona dedup: failed to delete loser sandbox_id=%s for key=%r: %s",
                    getattr(loser, "id", "<unknown>"),
                    session_key,
                    exc,
                )
        return survivor

    async def execute_in_session(
        self,
        handle: object,
        code: str,
        timeout: Optional[int] = None,
    ) -> ExecutionResult:
        """Execute ``code`` against the ``AsyncSandbox`` returned by
        ``find_or_create_session``.
        """
        try:
            from daytona_sdk import CodeRunParams

            sandbox: AsyncSandbox = handle  # type: ignore[assignment]
            result = await sandbox.process.code_run(
                code, params=CodeRunParams(env=self._user_env or None)
            )
            return _to_execution_result(result)
        except Exception as exc:
            return ExecutionResult(stdout="", stderr=str(exc), error=str(exc))

    async def execute(
        self,
        code: str,
        session_key: str,
        timeout: Optional[int] = None,
    ) -> ExecutionResult:
        """Direct one-shot execution: spin up an ephemeral sandbox, run, delete.

        The evaluator path enters via ``execute()`` without ever calling
        ``find_or_create_session``, so configured dependencies.packages must
        be installed here too — otherwise they're silently dropped.
        ``session_key`` is unused in the direct path because the ephemeral
        sandbox is single-use; no label is set on the create params so the
        ephemeral sandbox cannot be accidentally picked up by a later
        ``find_or_create_session`` list.
        """
        try:
            from daytona_sdk import CodeRunParams

            client = self._get_client()
            workspace = await client.create(self._create_params(session_key=None))
            try:
                await self._install_packages(workspace)
                result = await workspace.process.code_run(
                    code, params=CodeRunParams(env=self._user_env or None)
                )
                return _to_execution_result(result)
            finally:
                try:
                    await client.delete(workspace)
                except Exception:
                    logger.warning("Failed to delete ephemeral Daytona workspace", exc_info=True)
        except Exception as exc:
            return ExecutionResult(stdout="", stderr=str(exc), error=str(exc))

    async def close_session(self, session_key: str) -> None:
        """Tear down every Daytona sandbox tagged with ``session_key``.

        Idempotent: an empty list (no matching sandboxes) is a no-op. There
        is no backend-local bookkeeping, so the pop-before-await invariant is
        vacuous — the binding lives at the provider and is released by
        ``client.delete``.
        """
        client = self._get_client()
        try:
            matches = await self._list_sandboxes_for_key(session_key)
        except Exception as exc:
            logger.warning(
                "Daytona close_session: list failed for key=%r: %s",
                session_key,
                exc,
            )
            return
        for sandbox in matches:
            try:
                await client.delete(sandbox)
            except Exception as exc:
                logger.warning(
                    "Daytona close_session: delete failed for sandbox_id=%s key=%r: %s",
                    getattr(sandbox, "id", "<unknown>"),
                    session_key,
                    exc,
                )

    async def close(self) -> None:
        """Wrapper-level close.

        Backend wrappers are ephemeral (one per request); there is no
        per-backend session state to release here. Per-session cleanup is
        the manager's job via ``close_session``. The lazy SDK client is
        closed so the underlying httpx pool does not leak.
        """
        if self._client is not None:
            try:
                await self._client.close()  # type: ignore[no-untyped-call]  # SDK 0.140 lacks return annotation
            except Exception:
                logger.warning("Failed to close Daytona client", exc_info=True)
            self._client = None


_DAYTONA_CREDENTIAL_SPECS = [
    ProviderCredentialSpec(
        key="PHOENIX_SANDBOX_DAYTONA_API_KEY",
        display_name="Daytona API Key",
        description="API key for the Daytona sandbox service.",
    ),
]


def _build_daytona_backend(
    config: Mapping[str, Any],
    *,
    language: str,
    user_env: Optional[Mapping[str, str]] = None,
) -> SandboxBackend:
    """Construct a DaytonaSandboxBackend for either language adapter.

    Fail-closed on missing credential. Passing an empty api_key would let the
    Daytona SDK silently fall back to ``DAYTONA_API_KEY`` from the process env
    (daytona_sdk/_async/daytona.py:168). The SDK's autodiscovery name differs
    from Phoenix's declared name (``PHOENIX_SANDBOX_DAYTONA_API_KEY``) so that
    fallback would bypass Phoenix's credential resolution entirely.
    """
    api_key: str = config.get("PHOENIX_SANDBOX_DAYTONA_API_KEY") or ""
    if not api_key:
        raise ValueError(
            "Daytona sandbox authentication is not configured. Set "
            "PHOENIX_SANDBOX_DAYTONA_API_KEY via setSandboxCredential or as "
            "a process environment variable."
        )
    deps = config.get("dependencies") or {}
    packages: list[str] = deps.get("packages", []) if isinstance(deps, dict) else []
    internet_access = config.get("internet_access") or {}
    mode: str = internet_access.get("mode", "") if isinstance(internet_access, dict) else ""
    network_block_all = mode == "deny"
    return DaytonaSandboxBackend(
        api_key=Secret(api_key),
        server_url="",
        user_env=user_env,
        packages=packages,
        network_block_all=network_block_all,
        language=language,
    )


def _probe_daytona_sdk() -> None:
    """Verify ``daytona_sdk`` is installed; ImportError → NOT_INSTALLED."""
    import daytona_sdk  # noqa: F401


class DaytonaPythonAdapter(SandboxAdapter):
    key = "DAYTONA_PYTHON"
    family = "DAYTONA"
    display_name = "Daytona"
    language = "PYTHON"
    config_model = DaytonaPythonConfig
    credential_specs = _DAYTONA_CREDENTIAL_SPECS

    @classmethod
    def probe_dependencies(cls) -> None:
        _probe_daytona_sdk()

    def build_backend(
        self, config: Mapping[str, Any], user_env: Optional[Mapping[str, str]] = None
    ) -> SandboxBackend:
        self._enforce_capabilities(config, user_env)
        return _build_daytona_backend(config, language="PYTHON", user_env=user_env)


class DaytonaTypescriptAdapter(SandboxAdapter):
    key = "DAYTONA_TYPESCRIPT"
    family = "DAYTONA"
    display_name = "Daytona"
    language = "TYPESCRIPT"
    config_model = DaytonaTypescriptConfig
    credential_specs = _DAYTONA_CREDENTIAL_SPECS

    @classmethod
    def probe_dependencies(cls) -> None:
        _probe_daytona_sdk()

    def build_backend(
        self, config: Mapping[str, Any], user_env: Optional[Mapping[str, str]] = None
    ) -> SandboxBackend:
        self._enforce_capabilities(config, user_env)
        return _build_daytona_backend(config, language="TYPESCRIPT", user_env=user_env)
