"""
Core types for the sandbox backend system.

Only depends on stdlib and pydantic (a core Phoenix dependency). Safe to
import unconditionally regardless of optional sandbox extras.
"""

from __future__ import annotations

import hashlib
import json
import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import (
    Annotated,
    Any,
    Literal,
    Mapping,
    Optional,
    Type,
    Union,
    get_args,
)

from pydantic import BaseModel, ConfigDict, Field, field_validator
from starlette.datastructures import Secret


class UnsupportedOperation(Exception):
    """Raised when a sandbox backend does not support a requested operation."""


# ---------------------------------------------------------------------------
# Dependency-spec grammar. npm and Python requirement strings are validated
# independently — there is intentionally no cross-conversion: a TypeScript
# sandbox takes npm syntax (`lodash@^4.17`), a Python sandbox takes pip
# syntax (`numpy==1.26.0`), and a spec in the wrong dialect is rejected with
# a hint rather than silently translated. The frontend mirrors these in
# app/src/pages/settings/sandboxes/utils.tsx; keep the two in sync.
# ---------------------------------------------------------------------------

#: One identifier segment: starts/ends alphanumeric, allows ``.``/``_``/``-``/``~`` inside.
_IDENT = r"[A-Za-z0-9](?:[A-Za-z0-9._~-]*[A-Za-z0-9])?"

#: npm package name — optional ``@scope/`` prefix, then a name segment.
_NPM_NAME = rf"(?:@{_IDENT}/)?{_IDENT}"
#: npm version selector — anything non-empty without whitespace or ``@`` (covers
#: ranges like ``^1.2.0``, ``>=6.37.0``, ``1.2.3``, dist-tags, git refs).
_NPM_VERSION = r"[^@\s]+"
#: An npm requirement: ``name`` or ``name@version`` (incl. ``@scope/name``).
_NPM_REQUIREMENT_RE = re.compile(rf"^(?:{_NPM_NAME})(?:@{_NPM_VERSION})?$")

#: PEP 508 extras list, e.g. ``[socks,brotli]``.
_PY_EXTRAS = r"(?:\[\s*[A-Za-z0-9._-]+(?:\s*,\s*[A-Za-z0-9._-]+)*\s*\])?"
#: One PEP 440 version clause, e.g. ``>=1.2``, ``==1.*``, ``~=2.0``.
_PY_VERSION_CLAUSE = r"(?:===|==|!=|~=|<=|>=|<|>)\s*[A-Za-z0-9*][A-Za-z0-9.*+!_-]*"
#: A Python requirement: ``name[extras] <clause>[, <clause>...]`` (markers/URLs
#: are intentionally out of scope for the dependency-list UI).
_PYTHON_REQUIREMENT_RE = re.compile(
    rf"^{_IDENT}\s*{_PY_EXTRAS}\s*"
    rf"(?:{_PY_VERSION_CLAUSE}(?:\s*,\s*{_PY_VERSION_CLAUSE})*)?\s*$"
)


def validate_npm_package_spec(spec: str) -> str:
    """Strip and validate a single npm install spec; raise ValueError if invalid."""
    stripped = spec.strip()
    if not stripped or not _NPM_REQUIREMENT_RE.match(stripped):
        raise ValueError(
            f"invalid npm package spec {spec!r} "
            "(expected e.g. 'lodash', 'lodash@^4.17', '@scope/pkg@1.2.3')"
        )
    return stripped


def validate_python_package_spec(spec: str) -> str:
    """Strip and validate a single Python package spec; raise ValueError if invalid."""
    stripped = spec.strip()
    if not stripped or not _PYTHON_REQUIREMENT_RE.match(stripped):
        raise ValueError(
            f"invalid Python package spec {spec!r} "
            "(expected e.g. 'requests', 'numpy==1.26.0', 'httpx[http2]>=0.27,<1')"
        )
    return stripped


def _validated_package_list(packages: list[str], validate_one: Callable[[str], str]) -> list[str]:
    """Apply a per-entry validator across a package list.

    Re-raises the first failure with the offending index so the pydantic
    ValidationError points at the bad line.
    """
    out: list[str] = []
    for i, pkg in enumerate(packages):
        try:
            out.append(validate_one(pkg))
        except ValueError as exc:
            raise ValueError(f"packages[{i}]: {exc}") from exc
    return out


# ---------------------------------------------------------------------------
# SandboxProviderFamily — closed set of provider families used as the trust
# boundary for PHOENIX_ALLOWED_SANDBOX_PROVIDERS. Adding a new provider
# family requires adding its name here AND setting `family = "..."` on the
# adapter class. Pyright/mypy will reject mismatches at type-check time.
#
# The family of an adapter is the unit at which the allowlist operates:
# all (backend_type, language) variants of a family share one allowlist
# entry. Extending an existing family with a new language is just a new
# adapter class with the same `family`; extending Phoenix with a new
# provider family adds a literal here.
# ---------------------------------------------------------------------------
SandboxProviderFamily = Literal["WASM", "E2B", "DAYTONA", "VERCEL", "DENO", "MODAL"]


SANDBOX_PROVIDER_FAMILIES: frozenset[SandboxProviderFamily] = frozenset(
    get_args(SandboxProviderFamily)
)


# ---------------------------------------------------------------------------
# Shared config shapes — imported by per-adapter configs that opt in.
# ---------------------------------------------------------------------------


class EnvVarLiteral(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["literal"]
    name: str
    value: str


class EnvVarSecretRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["secret_ref"]
    name: str
    secret_key: str


EnvVarEntry = Annotated[
    Union[EnvVarLiteral, EnvVarSecretRef],
    Field(discriminator="kind"),
]


class InternetAccessConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["deny", "allow"] = "allow"


class PythonDependenciesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    packages: list[str] = Field(default_factory=list)
    lockfile: Optional[str] = None

    @field_validator("packages", mode="after")
    @classmethod
    def _validate_python_specs(cls, packages: list[str]) -> list[str]:
        return _validated_package_list(packages, validate_python_package_spec)


class TypescriptDependenciesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    packages: list[str] = Field(default_factory=list)
    lockfile: Optional[str] = None

    @field_validator("packages", mode="after")
    @classmethod
    def _validate_npm_specs(cls, packages: list[str]) -> list[str]:
        return _validated_package_list(packages, validate_npm_package_spec)


@dataclass
class ProviderCredentialSpec:
    """Describes a provider credential env var required by a sandbox adapter.

    Used by _resolve_sandbox_credentials() for DB secret lookup and by
    setSandboxCredential/deleteSandboxCredential mutations for key validation.
    """

    key: str
    display_name: str
    description: str = ""
    is_required: bool = True


# ---------------------------------------------------------------------------
# Per-adapter pydantic config models.
# extra="forbid" rejects unknown keys at validate_config.
# All config fields are optional — adapters use defaults for missing keys.
# ---------------------------------------------------------------------------


class E2BConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    env_vars: list[EnvVarEntry] = Field(
        default_factory=list,
        title="Environment Variables",
        description="Environment variables set at build time; not overridable per call.",
    )
    internet_access: Optional[InternetAccessConfig] = Field(
        default=None,
        title="Internet Access",
        description="Controls whether the sandbox can reach the internet.",
    )
    dependencies: Optional[PythonDependenciesConfig] = Field(
        default=None,
        title="Python Dependencies",
        description="Python packages to install before code execution.",
    )


class DaytonaPythonConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    env_vars: list[EnvVarEntry] = Field(
        default_factory=list,
        title="Environment Variables",
        description="Environment variables set at build time; not overridable per call.",
    )
    internet_access: Optional[InternetAccessConfig] = Field(
        default=None,
        title="Internet Access",
        description="Controls whether the sandbox can reach the internet.",
    )
    dependencies: Optional[PythonDependenciesConfig] = Field(
        default=None,
        title="Python Dependencies",
        description="Python packages to install before code execution.",
    )


class DaytonaTypescriptConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    env_vars: list[EnvVarEntry] = Field(
        default_factory=list,
        title="Environment Variables",
        description="Environment variables set at build time; not overridable per call.",
    )
    internet_access: Optional[InternetAccessConfig] = Field(
        default=None,
        title="Internet Access",
        description="Controls whether the sandbox can reach the internet.",
    )
    dependencies: Optional[TypescriptDependenciesConfig] = Field(
        default=None,
        title="TypeScript Dependencies",
        description="npm packages to install before code execution.",
    )


class DenoConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    env_vars: list[EnvVarEntry] = Field(
        default_factory=list,
        title="Environment Variables",
        description="Environment variables set at build time; not overridable per call.",
    )
    internet_access: Optional[InternetAccessConfig] = Field(
        default=None,
        title="Internet Access",
        description="Controls whether the sandbox can reach the internet.",
    )


class _VercelConfigBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    env_vars: list[EnvVarEntry] = Field(
        default_factory=list,
        title="Environment Variables",
        description="Environment variables set at build time; not overridable per call.",
    )
    internet_access: Optional[InternetAccessConfig] = Field(
        default=None,
        title="Internet Access",
        description="Controls whether the sandbox can reach the internet.",
    )


class VercelPythonConfig(_VercelConfigBase):
    dependencies: Optional[PythonDependenciesConfig] = Field(
        default=None,
        title="Python Dependencies",
        description="Python packages to install before code execution.",
    )


class VercelTypescriptConfig(_VercelConfigBase):
    dependencies: Optional[TypescriptDependenciesConfig] = Field(
        default=None,
        title="TypeScript Dependencies",
        description="npm packages to install before code execution.",
    )


class WASMConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    env_vars: list[EnvVarEntry] = Field(
        default_factory=list,
        title="Environment Variables",
        description="Environment variables set at build time; not overridable per call.",
    )
    internet_access: Optional[InternetAccessConfig] = Field(
        default=None,
        title="Internet Access",
        description="Controls whether the sandbox can reach the internet.",
    )
    dependencies: Optional[PythonDependenciesConfig] = Field(
        default=None,
        title="Python Dependencies",
        description="Python packages to install before code execution.",
    )


@dataclass
class ExecutionResult:
    """Result returned by a sandbox execution."""

    stdout: str
    stderr: str
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None


def compose_secret_values(
    user_env: Optional[Mapping[str, str]],
    *credentials: Optional[Secret],
) -> frozenset[str]:
    """Combine user-env plaintext values with provider credential plaintexts.

    Called by each ``SandboxBackend.__init__`` to populate ``self.secret_values``
    in a single place. Empty/None credential entries are dropped so adapters
    with partial credential sets (e.g. one missing key) don't introduce
    empty-string entries that would mask everywhere.

    Credentials are passed as ``starlette.datastructures.Secret`` and unwrapped
    via ``str()`` for the masking layer, which performs string replacement on
    emitted span attributes and exception messages and therefore needs the
    plaintext form to match against.
    """
    return frozenset((user_env or {}).values()) | frozenset(str(c) for c in credentials if c)


#: Sentinel handle returned by ``BaseNoSessionBackend.find_or_create_session``.
#: Stateless adapters (WASM, Deno) have no provider-side session to bind to —
#: the manager still stores a handle in its tracked-session table, so we
#: return this opaque singleton rather than ``None`` to keep the type uniform
#: and to make accidental dereferencing obvious in logs.
_NO_SESSION_HANDLE: object = object()


class SandboxBackend(ABC):
    """
    Protocol for sandbox backends.

    Surface: ``execute`` + ``find_or_create_session`` + ``execute_in_session``
    + ``close_session`` + ``close``.

    Session reuse is bound at the provider, not in Phoenix process state:
    ``find_or_create_session(session_key)`` returns an opaque remote handle
    that the adapter understands (e.g. a Modal ``Sandbox``, an E2B
    ``AsyncSandbox``, a Daytona ``Sandbox``, or a sentinel for stateless
    backends). Callers pass that handle back to ``execute_in_session`` to
    run code against the specific remote session. ``close_session`` releases
    the binding for one ``session_key``.

    ``execute(...)`` remains the direct, no-manager one-shot path used by
    callers that don't need session reuse (and by stateless backends, which
    delegate ``execute_in_session`` to it under the hood).

    ``secret_values`` is the union of user-env plaintexts and provider
    credential plaintexts that ``CodeEvaluatorRunner`` will mask out of
    emitted span attributes, status descriptions, and exception events.
    Subclasses populate it in ``__init__`` via ``compose_secret_values``;
    the class-level default ``frozenset()`` means a backend that takes no
    credentials and no user_env (e.g. WASM) needs no extra wiring, and
    mocks via ``MagicMock(spec=SandboxBackend)`` inherit it for free.
    """

    secret_values: frozenset[str] = frozenset()

    @abstractmethod
    async def find_or_create_session(self, session_key: str) -> object:
        """Return an opaque remote handle for the session identified by ``session_key``.

        Idempotent under the same ``session_key``: a second call with the same
        key MUST return the same remote session (either by lookup of an
        existing handle in backend-local bookkeeping, or by provider-native
        list/get on a stable identifier derived from ``session_key`` via
        ``provider_session_id``). Two replicas of Phoenix calling this method
        with the same ``session_key`` against the same provider must converge
        on a single remote sandbox where the provider supports it (Modal: by
        name; E2B/Daytona: by metadata-list). Adapters whose provider does not
        expose cross-process binding (currently Vercel) document the limit
        on their override.

        The returned value is opaque to the caller — it is passed back to
        ``execute_in_session`` unchanged. Adapters return the SDK-native
        session object (e.g. ``modal.Sandbox``, ``e2b.AsyncSandbox``,
        ``daytona.Sandbox``). ``BaseNoSessionBackend`` returns a sentinel
        handle.

        ``session_key`` is opaque to the adapter: callers compose it at the
        call site (e.g. ``f"evaluator:{evaluator.id}"``). Adapters that need
        to sanitize the key for provider char-class or length constraints do
        so via ``provider_session_id``.
        """
        ...

    @abstractmethod
    async def execute_in_session(
        self,
        handle: object,
        code: str,
        timeout: Optional[int] = None,
    ) -> ExecutionResult:
        """Execute ``code`` against a remote session referenced by ``handle``.

        ``handle`` is the value previously returned by
        ``find_or_create_session``. Adapters must not look up the session via
        a backend-local ``_sessions[session_key]`` map for manager-mediated
        execution — the manager passes the handle directly so that wrapper
        identity is not load-bearing for session dispatch.

        User-supplied environment variables are set at ``build_backend()``
        time via the ``user_env`` argument and carried by the adapter for the
        life of the session; there is no per-call env override by design.
        """
        ...

    @abstractmethod
    async def close_session(self, session_key: str) -> None:
        """Release the binding for ``session_key`` and stop the remote session.

        Idempotent: a no-op when ``session_key`` is absent from backend-local
        bookkeeping.

        Implementations MUST pop the session entry from any backend-local
        bookkeeping (e.g. a ``_sessions[session_key]`` dict) synchronously
        before the first ``await``. The ``SandboxSessionManager`` releases
        its per-key lock before awaiting ``close_session``, so any
        implementation that awaits before popping can race a concurrent
        same-key ``find_or_create_session`` against the same backend-local
        slot — the new session would be overwritten or shadowed by the close
        in progress.

        Failure semantics are best-effort: the manager logs and continues on
        any ``Exception`` raised here. Backends that mind orphaned
        provider-side resources (running container, billed minutes) should
        either retry internally or rely on the provider's idle-reclamation
        timeout (configured per-provider via the hard-coded TTL kwargs in
        ``find_or_create_session``) — the manager will not re-drive a failed
        ``close_session``.
        """
        ...

    @abstractmethod
    async def execute(
        self,
        code: str,
        session_key: str,
        timeout: Optional[int] = None,
    ) -> ExecutionResult:
        """Execute code in the sandbox session identified by session_key.

        Direct one-shot path that does not require a previously-obtained
        handle. Manager-mediated execution goes through
        ``find_or_create_session`` + ``execute_in_session`` instead.

        User-supplied environment variables are set at build_backend() time
        via the `user_env` argument and carried by the adapter for the life
        of the session. There is no per-call env override by design.
        """
        ...

    @abstractmethod
    async def close(self) -> None:
        """Release all resources held by this backend."""
        ...

    @abstractmethod
    def config_fingerprint(self) -> str:
        """Return a short, stable digest of the config fields that affect the
        remote runtime.

        Two backends with structurally-equal configs (same provider family,
        same package list, same internet-access mode, same env-var key set)
        must return the same fingerprint across replicas and across runs.
        Changes to a secret's plaintext value that do NOT change the env-var
        key set must NOT change the fingerprint — masking already handles
        plaintext rotation; the fingerprint scopes session reuse and must
        survive credential rotation.

        Used by ``SandboxSessionManager`` to compose an internal composite
        key (``f"{session_key}#{fingerprint}"``) so a mid-iteration config
        change under a stable frontend ``session_key`` produces a fresh
        session at both the Phoenix and provider layers.

        Stateless adapters (``BaseNoSessionBackend``) return ``""`` — the
        manager short-circuits stateless backends before composing, so the
        value is unused.
        """
        ...

    def provider_session_id(self, session_key: str) -> str:
        """Map an opaque ``session_key`` to the identifier the provider sees.

        Must be deterministic across calls and processes: two Phoenix replicas
        that compute ``provider_session_id`` from the same ``session_key``
        must produce identical output so that provider-side list/lookup
        converges on the same remote sandbox.

        The default implementation returns ``session_key`` unchanged. Override
        only when the provider has char-class or length restrictions on its
        sandbox identifiers (e.g. Modal restricts names to alphanumeric and
        ``-`` and limits length, so its adapter overrides this with a
        sha256-based digest).
        """
        return session_key


class BaseNoSessionBackend(SandboxBackend):
    """
    Mixin for stateless sandbox backends (e.g. WASM, Deno).

    ``find_or_create_session`` returns a sentinel handle — there is no
    provider-side session to bind to. ``execute_in_session`` delegates to
    ``execute``. ``close_session`` is a no-op. Subclasses only need to
    implement ``execute`` and ``close``.
    """

    async def find_or_create_session(self, session_key: str) -> object:
        return _NO_SESSION_HANDLE

    async def execute_in_session(
        self,
        handle: object,
        code: str,
        timeout: Optional[int] = None,
    ) -> ExecutionResult:
        # Stateless backends carry no per-session remote state; route through
        # the direct one-shot path. ``session_key`` is unused on the manager
        # side here because the manager only stores the sentinel handle.
        return await self.execute(code, session_key="", timeout=timeout)

    async def close_session(self, session_key: str) -> None:
        return None

    def config_fingerprint(self) -> str:
        # Stateless adapters never reach the manager's keying logic; the
        # sentinel value is uniform and makes accidental composition obvious.
        return ""


def compute_config_fingerprint(
    *,
    family: str,
    packages: Optional[Mapping[str, Any] | list[str] | tuple[str, ...]] = None,
    internet_access_mode: Optional[str] = None,
    env_var_keys: Optional[list[str] | tuple[str, ...] | set[str] | frozenset[str]] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> str:
    """Compute a stable sha256-prefix fingerprint over a config subset.

    Inputs are normalized into a canonical JSON document so two backends with
    structurally-equal configs produce identical output:

    - ``family``: provider family identifier (e.g. ``"MODAL"``).
    - ``packages``: dependency specs; sorted so list order does not matter.
    - ``internet_access_mode``: canonical mode token (e.g. ``"allow"``,
      ``"deny"``, or ``None`` when unset).
    - ``env_var_keys``: names of user-supplied env vars; sorted so insertion
      order does not matter. **Values are intentionally excluded** so a
      secret-plaintext rotation does not invalidate session reuse.
    - ``extra``: adapter-specific extras whose change should fragment the
      session (e.g. language for multi-language adapters); keys are sorted.

    Returns a truncated lowercase hex digest (16 chars) suitable for
    composing into a manager-internal key without bloating log lines.
    """
    packages_list: list[str]
    if packages is None:
        packages_list = []
    elif isinstance(packages, Mapping):
        # Defensive: callers may pass a dependencies dict directly.
        raw_pkgs = packages.get("packages") or []
        packages_list = sorted(str(p) for p in raw_pkgs)
    else:
        packages_list = sorted(str(p) for p in packages)

    env_keys_list: list[str] = sorted(env_var_keys) if env_var_keys else []
    extra_dict: dict[str, Any] = {}
    if extra:
        for k in sorted(extra):
            extra_dict[k] = extra[k]

    payload = {
        "family": family,
        "packages": packages_list,
        "internet_access_mode": internet_access_mode,
        "env_var_keys": env_keys_list,
        "extra": extra_dict,
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


class SandboxAdapter(ABC):
    """
    Abstract base class for sandbox adapters.

    An adapter bridges a SandboxConfig (DB row) and a SandboxBackend instance.
    It owns credential resolution and backend construction.
    """

    #: Unique key identifying this adapter (matches backend_type in sandbox_providers).
    key: str

    #: Provider family — the trust boundary for PHOENIX_ALLOWED_SANDBOX_PROVIDERS.
    #: All adapters that share infrastructure / SDK / credentials / isolation
    #: boundary (e.g. VercelPythonAdapter and VercelTypescriptAdapter both
    #: family="VERCEL") share an allowlist entry.
    family: SandboxProviderFamily

    #: Human-readable name for display in the UI.
    display_name: str

    #: Language this adapter supports (must match Language.name values in DB).
    language: Literal["PYTHON", "TYPESCRIPT"]

    #: Pydantic model used for config validation. Subclasses override at class level.
    config_model: Type[BaseModel] = BaseModel

    #: Specs for provider credential env vars required by this adapter.
    credential_specs: list["ProviderCredentialSpec"] = []

    @classmethod
    def probe_dependencies(cls) -> None:
        """Verify optional SDK dependencies are importable; raise ImportError otherwise.

        Called by ``phoenix.server.sandbox.__init__`` at registration time. Subclasses
        whose backend depends on an optional extra (wasmtime, e2b_code_interpreter,
        daytona_sdk, vercel, modal, ...) should override this to import their SDK
        and let the ImportError bubble. Adapters without optional SDK deps (e.g.
        Deno, which shells out to the ``deno`` CLI) inherit this no-op default.

        The registration block in ``phoenix.server.sandbox.__init__`` wraps the
        adapter import + ``probe_dependencies()`` + registration in a single
        ``try/except ImportError``: a failed probe results in the adapter being
        absent from ``_SANDBOX_ADAPTERS``, which the status resolver maps to
        ``status=NOT_INSTALLED`` (and surfaces the adapter's dependency hints).
        """
        return None

    @abstractmethod
    def build_backend(
        self,
        config: Mapping[str, Any],
        user_env: Optional[Mapping[str, str]] = None,
    ) -> SandboxBackend:
        """Construct and return a SandboxBackend from the provided config.

        The canonical capability contract is defined on AdapterMetadata
        (phoenix.server.sandbox.AdapterMetadata). Each flag on that class
        specifies the runtime obligation for this method:

        - supports_env_vars: if True, forward user_env to the backend at
          execute-time or creation-time as appropriate. If False, MUST raise
          UnsupportedOperation when user_env is non-empty.
        - internet_access_capability: if "none", MUST raise UnsupportedOperation
          when config.get("internet_access") resolves to a non-"none" mode.
        - dependencies_language: if None, MUST raise UnsupportedOperation when
          config.get("dependencies") contains non-empty packages.

        user_env is a pre-resolved plaintext mapping of user-supplied environment
        variables (name → value). It is passed as a sibling argument — NOT
        merged into config — to prevent collision with PHOENIX_SANDBOX_*
        credential keys that adapters read from config.
        """
        ...

    def validate_config(self, config: Mapping[str, Any]) -> dict[str, Any]:
        """
        Validate config via the adapter's pydantic config_model, then apply
        capability gates from AdapterMetadata.

        Returns the validated config dict. Raises ValueError on struct-validation
        failures and pydantic ValidationError when the validated config violates
        an advertised capability:
          - supports_env_vars is False and config has non-empty env_vars
          - internet_access_capability == "none" and config has an
            internet_access block
          - dependencies_language is None and config.dependencies.packages
            is non-empty

        The per-adapter build_backend capability guards remain in place as
        defense-in-depth (see _enforce_capabilities template method).
        """
        from pydantic import ValidationError

        try:
            validated = self.config_model.model_validate(config)
        except ValidationError as exc:
            raise ValueError(str(exc)) from exc
        # model_dump preserves extra fields because models use extra="allow"
        validated_dict = validated.model_dump()
        self._enforce_unique_env_var_names(validated_dict)
        self._enforce_capability_gates(validated_dict)
        return validated_dict

    def _enforce_unique_env_var_names(self, config: Mapping[str, Any]) -> None:
        """Reject duplicate ``name`` values in config.env_vars.

        Silent last-wins is unsafe: two entries with the same name but
        different kinds (e.g. literal vs secret_ref) would let one arbitrarily
        override the other at resolve time. Fail at write time instead so the
        caller sees a deterministic diagnostic.
        """
        from pydantic import ValidationError
        from pydantic_core import InitErrorDetails, PydanticCustomError

        env_vars = config.get("env_vars") or []
        seen: set[str] = set()
        duplicates: list[str] = []
        for entry in env_vars:
            name = entry.get("name") if isinstance(entry, dict) else getattr(entry, "name", None)
            if not isinstance(name, str):
                continue
            if name in seen and name not in duplicates:
                duplicates.append(name)
            seen.add(name)

        if not duplicates:
            return

        errors: list[InitErrorDetails] = [
            InitErrorDetails(
                type=PydanticCustomError(
                    "duplicate_env_var_name",
                    (
                        "Duplicate env_var name '{name}': env_var names must be "
                        "unique within a single SandboxConfig."
                    ),
                    {"name": name},
                ),
                loc=("env_vars",),
                input=env_vars,
            )
            for name in duplicates
        ]
        raise ValidationError.from_exception_data(type(self).__name__, errors)

    def _enforce_capability_gates(self, config: Mapping[str, Any]) -> None:
        """Raise pydantic ValidationError if config violates AdapterMetadata
        capability flags.

        Metadata is resolved via a lazy import to avoid a circular dependency
        between `types.py` and `sandbox.__init__`.
        """
        from pydantic import ValidationError
        from pydantic_core import InitErrorDetails, PydanticCustomError

        try:
            from phoenix.server.sandbox import SANDBOX_ADAPTER_METADATA
        except ImportError:
            return
        metadata = SANDBOX_ADAPTER_METADATA.get(self.key)
        if metadata is None:
            return

        errors: list[InitErrorDetails] = []

        env_vars = config.get("env_vars")
        if not metadata.supports_env_vars and env_vars:
            errors.append(
                InitErrorDetails(
                    type=PydanticCustomError(
                        "capability_violation",
                        (
                            "{adapter} adapter does not support user-defined "
                            "environment variables; remove env_vars or switch "
                            "to an adapter that supports them."
                        ),
                        {"adapter": self.key},
                    ),
                    loc=("env_vars",),
                    input=env_vars,
                )
            )

        internet_access = config.get("internet_access")
        if metadata.internet_access_capability == "none" and internet_access is not None:
            errors.append(
                InitErrorDetails(
                    type=PydanticCustomError(
                        "capability_violation",
                        (
                            "{adapter} adapter does not support internet_access "
                            "configuration; remove the internet_access field or "
                            "switch to an adapter that supports it."
                        ),
                        {"adapter": self.key},
                    ),
                    loc=("internet_access",),
                    input=internet_access,
                )
            )

        dependencies = config.get("dependencies")
        if metadata.dependencies_language is None and dependencies:
            packages = dependencies.get("packages") if isinstance(dependencies, dict) else None
            if packages:
                errors.append(
                    InitErrorDetails(
                        type=PydanticCustomError(
                            "capability_violation",
                            (
                                "{adapter} adapter does not support dependency "
                                "installation; remove dependencies.packages or "
                                "switch to an adapter that supports it."
                            ),
                            {"adapter": self.key},
                        ),
                        loc=("dependencies", "packages"),
                        input=packages,
                    )
                )

        # Runtime-install adapters install packages INSIDE the sandbox via
        # run_code, so a sandbox created with the network already denied has no
        # PyPI access and the install silently fails. Reject the combination
        # eagerly.
        if metadata.installs_packages_at_runtime and metadata.dependencies_language is not None:
            ia_mode: Optional[str] = None
            if isinstance(internet_access, dict):
                ia_mode = internet_access.get("mode")
            elif internet_access is not None:
                ia_mode = getattr(internet_access, "mode", None)
            packages_list: list[Any] = []
            if isinstance(dependencies, dict):
                packages_list = dependencies.get("packages") or []
            elif dependencies is not None:
                packages_list = getattr(dependencies, "packages", None) or []
            if ia_mode == "deny" and packages_list:
                errors.append(
                    InitErrorDetails(
                        type=PydanticCustomError(
                            "capability_violation",
                            (
                                "{adapter} adapter installs packages inside the "
                                "sandbox at runtime, so internet_access.mode='deny' "
                                "is incompatible with non-empty "
                                "dependencies.packages: pip cannot reach PyPI from "
                                "a network-denied sandbox. Set internet_access.mode "
                                "to 'allow' or remove dependencies.packages."
                            ),
                            {"adapter": self.key},
                        ),
                        loc=("dependencies", "packages"),
                        input=packages_list,
                    )
                )

        if errors:
            raise ValidationError.from_exception_data(
                type(self).__name__,
                errors,
            )

    def _enforce_capabilities(
        self,
        config: Mapping[str, Any],
        user_env: Optional[Mapping[str, str]] = None,
    ) -> None:
        """Raise UnsupportedOperation if config/user_env violates this adapter's
        advertised capabilities per SANDBOX_ADAPTER_METADATA.

        Build-time (second) capability guard. The first guard runs at
        validate_config time via ``_enforce_capability_gates`` and raises
        pydantic ValidationError. This method runs at ``build_backend`` time,
        enforcing the same contract against the effective runtime inputs
        (including per-execute ``user_env``) and raising UnsupportedOperation
        so executor surfaces (evaluators, chat_mutations) can surface the
        violation as an adapter error.

        Contract:
        - ``supports_env_vars`` is False → config's ``env_vars`` list must be
          empty AND ``user_env`` must be falsy.
        - ``internet_access_capability == "none"`` → config must not carry an
          ``internet_access`` block whose ``mode`` is non-None.
        - ``dependencies_language is None`` → config must not carry non-empty
          ``dependencies.packages``.

        ``config`` is a plain dict after validate_config (model_validate →
        model_dump). Nested shapes are dual-accessed via ``dict.get()`` /
        ``getattr()`` so callers passing pydantic instances still work.
        """
        # Lazy import to avoid circular dependency with sandbox/__init__.py.
        try:
            from phoenix.server.sandbox import SANDBOX_ADAPTER_METADATA
        except ImportError:
            return
        metadata = SANDBOX_ADAPTER_METADATA.get(self.key)
        if metadata is None:
            return

        if not metadata.supports_env_vars:
            env_vars = config.get("env_vars") or []
            if env_vars:
                raise UnsupportedOperation(
                    f"{self.display_name} backend does not support user-supplied "
                    "environment variables. Remove the `env_vars` field or switch "
                    "to a backend that supports env vars."
                )
            if user_env:
                raise UnsupportedOperation(
                    f"{self.display_name} backend does not support user-supplied "
                    "environment variables. Disable env_vars for this config or "
                    "switch to a backend that supports env vars."
                )

        if metadata.internet_access_capability == "none":
            internet_access = config.get("internet_access")
            if internet_access is not None:
                mode = (
                    internet_access.get("mode")
                    if isinstance(internet_access, dict)
                    else getattr(internet_access, "mode", None)
                )
                if mode is not None:
                    raise UnsupportedOperation(
                        f"{self.display_name} backend does not support "
                        "`internet_access` configuration. Remove the field or "
                        "switch to a backend that supports it."
                    )

        if metadata.dependencies_language is None:
            deps = config.get("dependencies")
            if deps is not None:
                packages = (
                    deps.get("packages")
                    if isinstance(deps, dict)
                    else getattr(deps, "packages", None)
                ) or []
                if packages:
                    raise UnsupportedOperation(
                        f"{self.display_name} backend does not support "
                        "dependency installation. Remove `dependencies.packages` "
                        "or switch to a backend that supports dependencies."
                    )

        # Runtime-install combo guard (mirrors _enforce_capability_gates).
        # Defense-in-depth: validate_config already rejects this at write time,
        # but a misconfigured stored config reaching build_backend (e.g. via a
        # path that bypassed validate_config) must still fail loudly rather
        # than silently producing a sandbox where pip install fails.
        if metadata.installs_packages_at_runtime and metadata.dependencies_language is not None:
            internet_access = config.get("internet_access")
            ia_mode: Optional[str] = None
            if isinstance(internet_access, dict):
                ia_mode = internet_access.get("mode")
            elif internet_access is not None:
                ia_mode = getattr(internet_access, "mode", None)
            deps = config.get("dependencies")
            packages_list: list[Any] = []
            if isinstance(deps, dict):
                packages_list = deps.get("packages") or []
            elif deps is not None:
                packages_list = getattr(deps, "packages", None) or []
            if ia_mode == "deny" and packages_list:
                raise UnsupportedOperation(
                    f"{self.display_name} backend installs packages inside the "
                    "sandbox at runtime; internet_access.mode='deny' blocks pip "
                    "from reaching PyPI. Set internet_access.mode to 'allow' or "
                    "remove dependencies.packages."
                )
