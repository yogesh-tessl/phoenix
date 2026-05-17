"""
E2B sandbox backend.

Requires the ``e2b_code_interpreter`` package (optional extra). Imports of the
SDK are lazy (in ``E2BSandboxBackend._get_sandbox_cls``) so the module remains
importable when the extra is absent. Adapter availability is gated by
``E2BAdapter.probe_dependencies`` at registration time, which surfaces a
missing extra as ``status=NOT_INSTALLED`` instead of a runtime error.

Session reuse is bound provider-side via the sandbox metadata key
``phoenix_session_key``. ``find_or_create_session`` lists running sandboxes
filtered on that key, connects to the oldest alive match, or creates a new
sandbox tagged with the key when none is found. Two Phoenix replicas asking
for the same ``session_key`` against E2B therefore converge on a single
remote sandbox.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence, cast

from starlette.datastructures import Secret

from .types import (
    E2BConfig,
    ExecutionResult,
    ProviderCredentialSpec,
    SandboxAdapter,
    SandboxBackend,
    compose_secret_values,
    compute_config_fingerprint,
)

if TYPE_CHECKING:
    from e2b.sandbox.sandbox_api import SandboxInfo
    from e2b_code_interpreter.code_interpreter_async import AsyncSandbox
    from e2b_code_interpreter.models import Execution

logger = logging.getLogger(__name__)

ENV_E2B_API_KEY = "E2B_API_KEY"

# Metadata key used to bind a Phoenix ``session_key`` to its E2B sandbox.
# E2B accepts arbitrary string values in metadata, so the opaque session_key
# is stored as-is (no transform).
_METADATA_SESSION_KEY = "phoenix_session_key"

# Hard-coded max-lifetime ceiling. The E2B SDK default is 300s (5 min);
# Phoenix passes 600s explicitly so the provider reaps orphans after a hard
# Phoenix crash without depending on the SDK default.
_SESSION_TIMEOUT_SECONDS = 600


class E2BSandboxBackend(SandboxBackend):
    """Sandbox backend executing code in E2B cloud sandboxes.

    Session reuse is bound provider-side on the ``phoenix_session_key``
    metadata key — see module docstring for the convergence guarantee.
    Ephemeral one-shot execution via ``execute()`` (no session_key match in
    metadata) spins up a fresh sandbox per call and tears it down with
    ``async with``.
    """

    def __init__(
        self,
        api_key: Secret,
        template: Optional[str] = None,
        metadata: Optional[str] = None,
        user_env: Optional[Mapping[str, str]] = None,
        allow_internet_access: bool = True,
        packages: Optional[Sequence[str]] = None,
    ) -> None:
        self._api_key = api_key
        # ``template=None`` lets ``AsyncSandbox.create()`` fall back to its
        # ``default_template`` (``code-interpreter-v1``), which is the only
        # image that runs the Jupyter server ``run_code()`` POSTs to on
        # ``JUPYTER_PORT`` (49999). The previously hard-coded ``"base"`` template
        # is the generic E2B image and does NOT run Jupyter, so every call
        # surfaced as ``502 The sandbox is running but port is not open``.
        self._template = template
        self._metadata = metadata
        self._user_env: dict[str, str] = dict(user_env or {})
        self._allow_internet_access = allow_internet_access
        self._packages: list[str] = list(packages) if packages else []
        self.secret_values = compose_secret_values(user_env, self._api_key)

    def config_fingerprint(self) -> str:
        return compute_config_fingerprint(
            family="E2B",
            packages=self._packages,
            internet_access_mode="allow" if self._allow_internet_access else "deny",
            env_var_keys=list(self._user_env.keys()),
            extra={"template": self._template} if self._template is not None else None,
        )

    def _get_sandbox_cls(self) -> type[AsyncSandbox]:
        from e2b_code_interpreter.code_interpreter_async import AsyncSandbox

        return AsyncSandbox

    def _api_opts(self) -> dict[str, Any]:
        """Return the ``ApiParams`` kwargs forwarded to every SDK call.

        ``api_key`` is the single Phoenix-resolved credential the SDK needs;
        forwarding it on every call (rather than relying on env-var
        autodiscovery) keeps the credential path explicit.
        """
        return {"api_key": str(self._api_key)}

    def _create_kwargs(self, session_key: Optional[str]) -> dict[str, Any]:
        """Build kwargs for ``AsyncSandbox.create()``.

        The E2B SDK expects metadata as ``Dict[str, str]``. The optional
        ``self._metadata`` info-string is tagged under ``"info"``. When a
        ``session_key`` is supplied (the session-reuse path), it is added
        under ``_METADATA_SESSION_KEY`` so the sandbox can be rediscovered
        via ``AsyncSandbox.list(query=SandboxQuery(metadata=...))``.
        """
        kwargs: dict[str, Any] = {
            "allow_internet_access": self._allow_internet_access,
            "timeout": _SESSION_TIMEOUT_SECONDS,
            **self._api_opts(),
        }
        if self._template is not None:
            kwargs["template"] = self._template
        metadata: dict[str, str] = {}
        if self._metadata is not None:
            metadata["info"] = self._metadata
        if session_key is not None:
            metadata[_METADATA_SESSION_KEY] = session_key
        if metadata:
            kwargs["metadata"] = metadata
        return kwargs

    async def _install_packages(self, sandbox: AsyncSandbox) -> None:
        """pip-install configured packages via run_code.

        ``{self._packages!r}`` serializes the list as a Python list literal
        with each spec wrapped in correctly escaped string quotes. ``shlex.quote``
        must NOT be used here: the generated code calls ``subprocess.run`` with
        a list (no shell), so any shell-style quoting becomes part of the argv
        element and pip rejects e.g. ``'numpy>=1.0'`` as an invalid name.
        """
        if not self._packages:
            return
        install_code = (
            "import subprocess, sys\n"
            "r = subprocess.run(\n"
            f"    [sys.executable, '-m', 'pip', 'install', *{self._packages!r}],\n"
            "    capture_output=True, text=True\n"
            ")\n"
            "if r.returncode != 0:\n"
            "    raise RuntimeError(r.stderr)\n"
        )
        execution = await sandbox.run_code(install_code)
        if execution.error:
            raise RuntimeError(f"pip install failed for {self._packages!r}: {execution.error}")

    async def _list_sandboxes_for_key(self, session_key: str) -> list[SandboxInfo]:
        """Return all running E2B sandboxes tagged with this ``session_key``.

        Walks the paginator to exhaustion. The metadata filter is server-side
        so the page count is bounded by how many duplicate sandboxes ever
        leaked under one key (post-create dedup keeps that at 1 in steady
        state).
        """
        from e2b.sandbox.sandbox_api import SandboxQuery

        sandbox_cls = self._get_sandbox_cls()
        paginator = sandbox_cls.list(
            query=SandboxQuery(metadata={_METADATA_SESSION_KEY: session_key}),
            **self._api_opts(),
        )
        results: list[SandboxInfo] = []
        while paginator.has_next:
            results.extend(await paginator.next_items())
        return results

    async def find_or_create_session(self, session_key: str) -> object:
        """Bind ``session_key`` to a single remote E2B sandbox.

        Convergence path: list by ``phoenix_session_key`` metadata; if an
        alive sandbox already exists, ``connect`` to the oldest one. On miss
        (or if the only candidate fails its alive probe), ``create`` a fresh
        sandbox tagged with the key, install configured packages, and
        re-list to dedupe any concurrent creates by competing replicas —
        keep the oldest by ``started_at``, kill the rest. The returned
        ``AsyncSandbox`` is the opaque handle passed back to
        ``execute_in_session``.
        """
        sandbox_cls = self._get_sandbox_cls()

        existing = await self._list_sandboxes_for_key(session_key)
        if existing:
            # Prefer the oldest alive sandbox: if another replica created
            # one earlier, both replicas converge on the same survivor in
            # the dedup step below.
            existing.sort(key=lambda info: info.started_at)
            oldest = existing[0]
            try:
                # SDK's connect() is typed as returning the base AsyncSandbox
                # class; sandbox_cls is the e2b_code_interpreter subclass, so
                # the cast restores the actual runtime type.
                sandbox = cast(
                    "AsyncSandbox",
                    await sandbox_cls.connect(
                        oldest.sandbox_id,
                        **self._api_opts(),
                    ),
                )
            except Exception as exc:
                # Stale handle (e.g. the sandbox died between list and
                # connect, or the provider returned a state that connect
                # can't recover). Fall through to create — surfacing this
                # to the caller would force every transient reaper-race to
                # bubble out as an evaluator failure.
                logger.debug(
                    "E2B connect failed for session_key=%r sandbox_id=%s; "
                    "falling through to create: %s",
                    session_key,
                    oldest.sandbox_id,
                    exc,
                )
            else:
                if await self._is_alive(sandbox):
                    logger.debug(
                        "E2B session reuse: connected to sandbox_id=%s for key=%r",
                        oldest.sandbox_id,
                        session_key,
                    )
                    return sandbox
                logger.debug(
                    "E2B sandbox_id=%s for key=%r failed alive probe; creating a fresh one",
                    oldest.sandbox_id,
                    session_key,
                )

        sandbox = await sandbox_cls.create(**self._create_kwargs(session_key))
        # Install-on-create only — never on connect-to-existing. The reused
        # sandbox already has its packages from its own create call.
        await self._install_packages(sandbox)

        # Post-create dedup: another replica may have raced us. Re-list,
        # pick the oldest survivor, kill the rest. ``started_at`` ordering
        # is deterministic across replicas, so both sides agree on the
        # survivor without coordination.
        deduped = await self._dedupe_after_create(session_key, sandbox)
        return deduped

    async def _dedupe_after_create(
        self,
        session_key: str,
        just_created: AsyncSandbox,
    ) -> AsyncSandbox:
        """Resolve post-create races for ``session_key``.

        If only one sandbox is tagged with the key, return the one we just
        created. If multiple exist, keep the oldest by ``started_at`` and
        kill the rest — including ``just_created`` if a competing replica
        created an earlier one.
        """
        sandbox_cls = self._get_sandbox_cls()
        candidates = await self._list_sandboxes_for_key(session_key)
        if len(candidates) <= 1:
            return just_created
        candidates.sort(key=lambda info: info.started_at)
        survivor_info = candidates[0]
        if survivor_info.sandbox_id == just_created.sandbox_id:
            survivor: AsyncSandbox = just_created
        else:
            try:
                survivor = cast(
                    "AsyncSandbox",
                    await sandbox_cls.connect(
                        survivor_info.sandbox_id,
                        **self._api_opts(),
                    ),
                )
            except Exception as exc:
                # If the older survivor can't be connected, fall back to
                # our own create — keeping the system live takes priority
                # over enforcing the oldest-wins rule on a dead handle.
                logger.warning(
                    "E2B dedup: connect to older survivor sandbox_id=%s "
                    "failed (%s); keeping just-created sandbox_id=%s",
                    survivor_info.sandbox_id,
                    exc,
                    just_created.sandbox_id,
                )
                return just_created
        for loser in candidates[1:]:
            try:
                await sandbox_cls.kill(loser.sandbox_id, **self._api_opts())
            except Exception as exc:
                logger.warning(
                    "E2B dedup: failed to kill loser sandbox_id=%s for key=%r: %s",
                    loser.sandbox_id,
                    session_key,
                    exc,
                )
        return survivor

    async def _is_alive(self, sandbox: AsyncSandbox) -> bool:
        """Best-effort alive probe; treat any SDK error as not-alive."""
        try:
            return await sandbox.is_running()
        except Exception as exc:
            logger.debug(
                "E2B is_running probe failed for sandbox_id=%s: %s",
                getattr(sandbox, "sandbox_id", "<unknown>"),
                exc,
            )
            return False

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
            sandbox: AsyncSandbox = handle  # type: ignore[assignment]
            execution: Execution = await sandbox.run_code(
                code,
                envs=self._user_env or None,
                timeout=timeout,
            )
            return _execution_to_result(execution)
        except Exception as exc:
            return ExecutionResult(
                stdout="",
                stderr=str(exc),
                error=str(exc),
            )

    async def execute(
        self,
        code: str,
        session_key: str,
        timeout: Optional[int] = None,
    ) -> ExecutionResult:
        """Direct one-shot execution: spin up an ephemeral sandbox, run, kill.

        The evaluator path enters via ``execute()`` without ever calling
        ``find_or_create_session``, so configured dependencies.packages must
        be installed here too — otherwise they're silently dropped. Mirrors
        the Daytona ephemeral branch. ``session_key`` is unused in the
        direct path because the ephemeral sandbox is single-use.
        """
        try:
            sandbox_cls = self._get_sandbox_cls()
            async with await sandbox_cls.create(**self._create_kwargs(session_key=None)) as sb:
                await self._install_packages(sb)
                execution: Execution = await sb.run_code(
                    code,
                    envs=self._user_env or None,
                    timeout=timeout,
                )
                return _execution_to_result(execution)
        except Exception as exc:
            return ExecutionResult(
                stdout="",
                stderr=str(exc),
                error=str(exc),
            )

    async def close_session(self, session_key: str) -> None:
        """Tear down every E2B sandbox tagged with ``session_key``.

        Idempotent: an empty list (no matching sandboxes) is a no-op. We do
        not maintain backend-local bookkeeping, so the pop-before-await
        invariant is vacuous here — the binding lives at the provider and
        is released by ``kill``.
        """
        sandbox_cls = self._get_sandbox_cls()
        try:
            matches = await self._list_sandboxes_for_key(session_key)
        except Exception as exc:
            logger.warning(
                "E2B close_session: list failed for key=%r: %s",
                session_key,
                exc,
            )
            return
        for info in matches:
            try:
                await sandbox_cls.kill(info.sandbox_id, **self._api_opts())
            except Exception as exc:
                logger.warning(
                    "E2B close_session: kill failed for sandbox_id=%s key=%r: %s",
                    info.sandbox_id,
                    session_key,
                    exc,
                )

    async def close(self) -> None:
        """Wrapper-level close.

        Backend wrappers are ephemeral (one per request); there is no
        per-backend state to release here. Per-session cleanup is the
        manager's job via ``close_session``.
        """
        return None


def _execution_to_result(execution: Execution) -> ExecutionResult:
    stdout = "\n".join(execution.logs.stdout) if execution.logs.stdout else ""
    stderr = "\n".join(execution.logs.stderr) if execution.logs.stderr else ""
    error_str: Optional[str] = str(execution.error) if execution.error else None
    return ExecutionResult(stdout=stdout, stderr=stderr, error=error_str)


class E2BAdapter(SandboxAdapter):
    key = "E2B"
    family = "E2B"
    display_name = "E2B"
    language = "PYTHON"
    config_model = E2BConfig
    credential_specs = [
        ProviderCredentialSpec(
            key=ENV_E2B_API_KEY,
            display_name="E2B API Key",
            description="API key for the E2B sandbox service.",
        ),
    ]

    @classmethod
    def probe_dependencies(cls) -> None:
        """Verify ``e2b_code_interpreter`` is installed; ImportError → NOT_INSTALLED."""
        import e2b_code_interpreter  # noqa: F401

    def build_backend(
        self,
        config: Mapping[str, Any],
        user_env: Optional[Mapping[str, str]] = None,
    ) -> SandboxBackend:
        self._enforce_capabilities(config, user_env)
        # Fail-closed on missing credential. Passing an empty api_key would let
        # the E2B SDK silently fall back to ``os.getenv("E2B_API_KEY")``
        # (e2b.connection_config:94). Phoenix's resolver already consults that
        # env var, so reaching this branch with an empty key means Phoenix
        # decided "no credential available"; raise rather than let the SDK
        # auto-discover and bypass that decision.
        api_key: str = config.get(ENV_E2B_API_KEY) or ""
        if not api_key:
            raise ValueError(
                "E2B sandbox authentication is not configured. Set "
                "E2B_API_KEY via setSandboxCredential or as a "
                "process environment variable."
            )
        internet_access = config.get("internet_access")
        if internet_access is not None:
            mode = (
                internet_access.get("mode")
                if isinstance(internet_access, dict)
                else getattr(internet_access, "mode", None)
            )
            allow_internet_access = mode != "deny"
        else:
            allow_internet_access = True
        deps = config.get("dependencies") or {}
        packages: list[str] = deps.get("packages", []) if isinstance(deps, dict) else []
        return E2BSandboxBackend(
            api_key=Secret(api_key),
            template=None,
            metadata=None,
            user_env=user_env,
            allow_internet_access=allow_internet_access,
            packages=packages or None,
        )
