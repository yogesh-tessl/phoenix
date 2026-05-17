"""
Modal sandbox backend.

Requires the ``modal`` package (optional extra). The SDK import is lazy (in
``ModalSandboxBackend.__init__`` and the ``_ensure_*`` helpers) so the module
remains importable when the extra is absent. Adapter availability is gated by
``ModalAdapter.probe_dependencies`` at registration time, which surfaces a
missing extra as ``status=NOT_INSTALLED`` instead of a runtime error during
evaluation.

Authentication: credentials are passed explicitly to the Modal SDK via
``modal.Client.from_credentials(token_id, token_secret)`` and threaded through
``modal.App.lookup`` and ``modal.Sandbox.create`` as a ``client=`` kwarg. The
backend never mutates ``os.environ`` — DB-resolved tokens stay scoped to the
adapter instance and cannot leak into the Phoenix process env, subprocesses,
logs, or crash dumps.

Session lifecycle
-----------------
- find_or_create_session(): looks up an existing Modal Sandbox by
  ``provider_session_id(session_key)`` (sha256-derived deterministic name) via
  ``Sandbox.from_name.aio``; on miss or stale handle, creates one with
  ``Sandbox.create.aio(name=..., timeout=600, idle_timeout=300, ...)``. The
  Modal sandbox-name namespace per-app guarantees cross-replica convergence:
  a concurrent winner from a sibling replica surfaces as
  ``AlreadyExistsError`` on ``create``, which is handled by re-attaching via
  ``from_name``.
- execute_in_session(): runs code via ``sandbox.exec.aio("python", "-c", code)``
  against the handle previously returned by ``find_or_create_session``.
- close_session(): looks up the sandbox by name and terminates it; a missing
  name is treated as no-op (idempotent).
- execute(): runs code in an ephemeral sandbox (create → exec → terminate).
  Manager-mediated session reuse lives on ``find_or_create_session`` +
  ``execute_in_session``; ``execute`` is always single-shot regardless of
  ``session_key``.
- close(): no backend-local session state to release (binding lives in
  Modal's per-app name namespace).
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence

from starlette.datastructures import Secret

from .types import (
    ExecutionResult,
    ModalConfig,
    ProviderCredentialSpec,
    SandboxAdapter,
    SandboxBackend,
    compose_secret_values,
    compute_config_fingerprint,
)

if TYPE_CHECKING:
    from modal import App, Client
    from modal.image import Image
    from modal.sandbox import Sandbox

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 600
_DEFAULT_IDLE_TIMEOUT = 300
ENV_MODAL_TOKEN_ID = "MODAL_TOKEN_ID"
ENV_MODAL_TOKEN_SECRET = "MODAL_TOKEN_SECRET"


class ModalSandboxBackend(SandboxBackend):
    """Sandbox backend executing code in Modal cloud sandboxes.

    Session reuse is bound at Modal via the per-app sandbox-name namespace:
    ``find_or_create_session(session_key)`` derives a deterministic Modal name
    via ``provider_session_id`` (sha256 prefix of ``session_key``) and uses
    ``Sandbox.from_name`` / ``Sandbox.create(name=...)`` so two Phoenix
    replicas converge on a single remote sandbox without any process-local
    binding state.

    Credentials are passed explicitly to the SDK via ``modal.Client.from_credentials``
    rather than via ``os.environ``. The client + app are constructed lazily on
    first use so a missing/invalid token surfaces at sandbox-creation time —
    consistent with how the rest of Phoenix's SDK adapters fail.
    """

    def __init__(
        self,
        token_id: Secret,
        token_secret: Secret,
        *,
        timeout: int = _DEFAULT_TIMEOUT,
        idle_timeout: int = _DEFAULT_IDLE_TIMEOUT,
        app_name: str = "phoenix-sandbox",
        user_env: Optional[Mapping[str, str]] = None,
        packages: Optional[Sequence[str]] = None,
        block_network: bool = False,
    ) -> None:
        if not token_id or not token_secret:
            raise ValueError(
                "Modal sandbox requires both MODAL_TOKEN_ID and "
                "MODAL_TOKEN_SECRET. Set them via setSandboxCredential "
                "or as process environment variables."
            )

        import modal

        self._token_id = token_id
        self._token_secret = token_secret
        self._timeout = timeout
        self._idle_timeout = idle_timeout
        self._app_name = app_name
        self._user_env: dict[str, str] = dict(user_env or {})
        self._block_network = block_network
        self._client: Optional[Client] = None
        self._app: Optional[App] = None
        self._client_lock = asyncio.Lock()
        self._packages: list[str] = list(packages) if packages else []
        base_image = modal.Image.debian_slim()
        self._image: Image = (
            base_image.pip_install(self._packages) if self._packages else base_image
        )
        self.secret_values = compose_secret_values(user_env, token_id, token_secret)

    def config_fingerprint(self) -> str:
        return compute_config_fingerprint(
            family="MODAL",
            packages=self._packages,
            internet_access_mode="deny" if self._block_network else "allow",
            env_var_keys=list(self._user_env.keys()),
        )

    def provider_session_id(self, session_key: str) -> str:
        # Modal restricts sandbox names to alphanumeric + ``-`` and limits
        # length to 64 chars. sha256 hex prefix is deterministic across
        # replicas, alphanumeric, and well under the limit at 32 chars.
        return hashlib.sha256(session_key.encode()).hexdigest()[:32]

    async def _ensure_client(self) -> Client:
        """Construct (or reuse) a typed Modal Client bound to this backend's credentials.

        Double-checked locking: the unlocked fast path serves the steady-state
        cache hit, and the re-check inside the lock prevents two concurrent
        first-time callers from each constructing a client.
        """
        import modal

        if self._client is not None:
            return self._client
        async with self._client_lock:
            if self._client is None:
                self._client = await modal.Client.from_credentials.aio(
                    str(self._token_id), str(self._token_secret)
                )
        return self._client

    async def _ensure_app(self) -> App:
        """Look up (or create) the Modal App for sandbox association, using our client.

        Double-checked locking, same rationale as ``_ensure_client``.
        """
        import modal

        if self._app is not None:
            return self._app
        client = await self._ensure_client()
        async with self._client_lock:
            if self._app is None:
                self._app = await modal.App.lookup.aio(
                    self._app_name, client=client, create_if_missing=True
                )
        return self._app

    async def _create_sandbox(self, *, name: Optional[str] = None) -> Sandbox:
        import modal

        client = await self._ensure_client()
        app = await self._ensure_app()
        kwargs: dict[str, Any] = {
            "app": app,
            "client": client,
            "image": self._image,
            "timeout": self._timeout,
            "idle_timeout": self._idle_timeout,
        }
        if name is not None:
            kwargs["name"] = name
        if self._user_env:
            kwargs["env"] = self._user_env
        if self._block_network:
            kwargs["block_network"] = True
        return await modal.Sandbox.create.aio(**kwargs)

    async def _from_name_if_alive(self, name: str) -> Optional[Sandbox]:
        """Look up a named Modal sandbox; return None if missing or stale."""
        import modal
        from modal.exception import NotFoundError

        client = await self._ensure_client()
        try:
            sandbox = await modal.Sandbox.from_name.aio(self._app_name, name, client=client)
        except NotFoundError:
            return None
        # poll() returns None while running, else the exit code: a non-None
        # return means the sandbox has already exited and should not be reused.
        try:
            returncode = await sandbox.poll.aio()
        except Exception:
            return None
        if returncode is not None:
            return None
        return sandbox

    async def find_or_create_session(self, session_key: str) -> Sandbox:
        from modal.exception import AlreadyExistsError

        name = self.provider_session_id(session_key)
        existing = await self._from_name_if_alive(name)
        if existing is not None:
            logger.debug(f"Modal session '{name}' already exists; reusing")
            return existing
        try:
            sandbox = await self._create_sandbox(name=name)
            logger.debug(f"Created Modal session '{name}'")
            return sandbox
        except AlreadyExistsError:
            # Concurrent winner from another replica claimed the name first;
            # attach to it via from_name. The winner is by construction the
            # one we would have created.
            attached = await self._from_name_if_alive(name)
            if attached is None:
                raise
            logger.debug(f"Modal session '{name}' won by concurrent creator; attaching")
            return attached

    async def execute_in_session(
        self,
        handle: object,
        code: str,
        timeout: Optional[int] = None,
    ) -> ExecutionResult:
        # ``handle`` is the Modal Sandbox returned by find_or_create_session;
        # the type is opaque at the ABC level but always a modal.Sandbox here.
        sandbox: Sandbox = handle  # type: ignore[assignment]
        try:
            return await self._exec_code(sandbox, code)
        except Exception as exc:
            return ExecutionResult(stdout="", stderr=str(exc), error=str(exc))

    async def close_session(self, session_key: str) -> None:
        # No backend-local bookkeeping to pop: binding lives in Modal's per-app
        # name namespace, so the pop-before-await invariant is trivially
        # satisfied — there is nothing to pop.
        import modal
        from modal.exception import NotFoundError

        name = self.provider_session_id(session_key)
        client = await self._ensure_client()
        try:
            sandbox = await modal.Sandbox.from_name.aio(self._app_name, name, client=client)
        except NotFoundError:
            return
        try:
            await sandbox.terminate.aio()
            logger.debug(f"Stopped Modal session '{name}'")
        except NotFoundError:
            return

    async def _exec_code(self, sandbox: Sandbox, code: str) -> ExecutionResult:
        """Run code in a sandbox and collect stdout/stderr."""
        proc = await sandbox.exec.aio("python", "-c", code)
        stdout, stderr = await asyncio.gather(
            proc.stdout.read.aio(),
            proc.stderr.read.aio(),
        )
        exit_code = await proc.wait.aio()
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
        # Direct one-shot ephemeral execution: create, run, terminate.
        # Manager-mediated session reuse lives on ``find_or_create_session`` /
        # ``execute_in_session``; this path is always single-shot.
        try:
            sandbox = await self._create_sandbox()
            try:
                return await self._exec_code(sandbox, code)
            finally:
                await sandbox.terminate.aio()
        except Exception as exc:
            return ExecutionResult(stdout="", stderr=str(exc), error=str(exc))

    async def close(self) -> None:
        # Session bindings live in Modal's per-app name namespace; there is no
        # backend-local session map to drain. The SandboxSessionManager calls
        # close_session() for each tracked key during shutdown.
        return None


class ModalAdapter(SandboxAdapter):
    key = "MODAL"
    family = "MODAL"
    display_name = "Modal"
    language = "PYTHON"
    config_model = ModalConfig
    credential_specs = [
        ProviderCredentialSpec(
            key=ENV_MODAL_TOKEN_ID,
            display_name="Modal Token ID",
            description="Token ID issued by `modal token new`.",
        ),
        ProviderCredentialSpec(
            key=ENV_MODAL_TOKEN_SECRET,
            display_name="Modal Token Secret",
            description="Token secret issued by `modal token new`.",
        ),
    ]

    @classmethod
    def probe_dependencies(cls) -> None:
        """Verify ``modal`` is installed; ImportError → NOT_INSTALLED."""
        import modal  # noqa: F401

    def build_backend(
        self,
        config: Mapping[str, Any],
        user_env: Optional[Mapping[str, str]] = None,
    ) -> SandboxBackend:
        self._enforce_capabilities(config, user_env)
        token_id = config.get(ENV_MODAL_TOKEN_ID) or ""
        token_secret = config.get(ENV_MODAL_TOKEN_SECRET) or ""
        if not token_id or not token_secret:
            raise ValueError(
                "Modal sandbox authentication is not configured. Set both "
                "MODAL_TOKEN_ID and MODAL_TOKEN_SECRET "
                "via setSandboxCredential or as process environment variables."
            )
        deps = config.get("dependencies") or {}
        packages: list[str] = deps.get("packages", []) if isinstance(deps, dict) else []
        ia = config.get("internet_access") or {}
        mode = ia.get("mode") if isinstance(ia, dict) else getattr(ia, "mode", None)
        block_network: bool = mode == "deny"
        return ModalSandboxBackend(
            token_id=Secret(token_id),
            token_secret=Secret(token_secret),
            timeout=_DEFAULT_TIMEOUT,
            idle_timeout=_DEFAULT_IDLE_TIMEOUT,
            app_name="phoenix-sandbox",
            user_env=user_env,
            packages=packages or None,
            block_network=block_network,
        )
