"""Unit tests for E2BSandboxBackend and E2BAdapter.

Scope: E2B-specific SDK kwarg shapes, pip-install-via-run_code wiring, and
``find_or_create_session`` provider-side binding via the
``phoenix_session_key`` metadata key.
"""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from starlette.datastructures import Secret

from phoenix.server.sandbox.e2b_backend import (
    _METADATA_SESSION_KEY,
    _SESSION_TIMEOUT_SECONDS,
    E2BAdapter,
    E2BSandboxBackend,
)

_API_KEY = Secret("k")
_CANONICAL_API_KEY = "E2B_API_KEY"


def _make_mock_sandbox_cls(create_result: Any = None) -> MagicMock:
    sandbox_instance = MagicMock()
    sandbox_instance.run_code = AsyncMock(
        return_value=MagicMock(logs=MagicMock(stdout=[], stderr=[]), error=None)
    )
    sandbox_instance.close = AsyncMock()
    sandbox_instance.is_running = AsyncMock(return_value=True)
    sandbox_cls = MagicMock()
    sandbox_cls.create = AsyncMock(return_value=create_result or sandbox_instance)
    sandbox_cls.connect = AsyncMock()
    sandbox_cls.kill = AsyncMock()
    # _list_sandboxes_for_key walks a paginator; default to "no existing".
    empty_paginator = MagicMock()
    empty_paginator.has_next = False
    sandbox_cls.list = MagicMock(return_value=empty_paginator)
    return sandbox_cls


def _make_paginator(items: list[Any]) -> MagicMock:
    """One-page paginator: ``has_next`` True once, then False after pull."""
    paginator = MagicMock()
    state = {"called": False}

    def _has_next_getter() -> bool:
        return not state["called"]

    type(paginator).has_next = property(lambda self: _has_next_getter())

    async def _next_items() -> list[Any]:
        state["called"] = True
        return items

    paginator.next_items = _next_items
    return paginator


def _make_sandbox_info(sandbox_id: str, started_at: int) -> MagicMock:
    info = MagicMock()
    info.sandbox_id = sandbox_id
    info.started_at = started_at
    return info


def _patch_sandbox_query() -> Any:
    """Make ``from e2b.sandbox.sandbox_api import SandboxQuery`` resolvable."""
    import sys

    api_mod = MagicMock()
    api_mod.SandboxQuery = lambda **kw: kw
    sandbox_pkg = MagicMock()
    sandbox_pkg.sandbox_api = api_mod
    e2b_pkg = MagicMock()
    e2b_pkg.sandbox = sandbox_pkg
    return patch.dict(
        sys.modules,
        {
            "e2b": e2b_pkg,
            "e2b.sandbox": sandbox_pkg,
            "e2b.sandbox.sandbox_api": api_mod,
        },
    )


def test_create_kwargs_defaults_to_allow_true() -> None:
    backend = E2BSandboxBackend(api_key=_API_KEY, template="base")
    assert backend._create_kwargs(session_key=None)["allow_internet_access"] is True


def test_create_kwargs_hardcodes_timeout() -> None:
    """Every create() call must carry ``timeout=600``."""
    backend = E2BSandboxBackend(api_key=_API_KEY, template="base")
    kwargs = backend._create_kwargs(session_key=None)
    assert kwargs["timeout"] == _SESSION_TIMEOUT_SECONDS == 600


def test_create_kwargs_tags_session_key_under_phoenix_metadata() -> None:
    """``find_or_create_session`` binds via the ``phoenix_session_key`` metadata
    key — assert ``_create_kwargs`` puts the opaque key there verbatim."""
    backend = E2BSandboxBackend(api_key=_API_KEY, template="base")
    kwargs = backend._create_kwargs(session_key="evaluator:42")
    md = kwargs["metadata"]
    assert md[_METADATA_SESSION_KEY] == "evaluator:42"


@pytest.mark.parametrize("allow", [True, False])
def test_create_kwargs_forwards_allow_internet_access(allow: bool) -> None:
    backend = E2BSandboxBackend(api_key=_API_KEY, template="base", allow_internet_access=allow)
    assert backend._create_kwargs(session_key=None)["allow_internet_access"] is allow


@pytest.mark.parametrize(
    "config,expected",
    [
        ({"internet_access": {"mode": "deny"}}, False),
        ({"internet_access": {"mode": "allow"}}, True),
        ({}, True),
    ],
)
def test_build_backend_translates_internet_access_to_allow_flag(
    config: dict[str, Any], expected: bool
) -> None:
    adapter = E2BAdapter()
    backend: E2BSandboxBackend = adapter.build_backend(  # type: ignore[assignment]
        {_CANONICAL_API_KEY: "k", **config}
    )
    assert backend._create_kwargs(session_key=None)["allow_internet_access"] is expected


@pytest.mark.asyncio
async def test_find_or_create_session_creates_when_no_existing() -> None:
    """List returns empty → create runs with the session_key in metadata."""
    mock_cls = _make_mock_sandbox_cls()
    backend = E2BSandboxBackend(api_key=_API_KEY, template="base")
    with _patch_sandbox_query():
        with patch.object(backend, "_get_sandbox_cls", return_value=mock_cls):
            handle = await backend.find_or_create_session("ev:1")

    mock_cls.create.assert_awaited_once()
    create_kwargs = mock_cls.create.call_args.kwargs
    assert create_kwargs["metadata"][_METADATA_SESSION_KEY] == "ev:1"
    assert create_kwargs["timeout"] == _SESSION_TIMEOUT_SECONDS
    assert handle is mock_cls.create.return_value


@pytest.mark.asyncio
async def test_find_or_create_session_connects_to_oldest_existing() -> None:
    """List returns two alive matches → connect to the oldest; create skipped."""
    older = _make_sandbox_info("sb-older", started_at=100)
    newer = _make_sandbox_info("sb-newer", started_at=200)

    mock_cls = _make_mock_sandbox_cls()
    mock_cls.list = MagicMock(return_value=_make_paginator([newer, older]))
    connected = MagicMock()
    connected.is_running = AsyncMock(return_value=True)
    mock_cls.connect = AsyncMock(return_value=connected)

    backend = E2BSandboxBackend(api_key=_API_KEY, template="base")
    with _patch_sandbox_query():
        with patch.object(backend, "_get_sandbox_cls", return_value=mock_cls):
            handle = await backend.find_or_create_session("ev:1")

    mock_cls.create.assert_not_awaited()
    mock_cls.connect.assert_awaited_once()
    connect_args = mock_cls.connect.call_args
    assert connect_args.args[0] == "sb-older", "must connect to the OLDEST sandbox by started_at"
    assert handle is connected


@pytest.mark.asyncio
async def test_find_or_create_session_falls_through_when_existing_is_stale() -> None:
    """Alive probe fails → ignore the stale candidate, create a fresh sandbox."""
    stale_info = _make_sandbox_info("sb-stale", started_at=100)
    stale = MagicMock()
    stale.is_running = AsyncMock(return_value=False)  # fails alive probe

    fresh = MagicMock()
    fresh.is_running = AsyncMock(return_value=True)
    fresh.sandbox_id = "sb-fresh"

    mock_cls = _make_mock_sandbox_cls()
    # First call: returns the stale candidate. Second call (dedup): only fresh.
    list_calls = {"count": 0}

    def _list(**kwargs: Any) -> Any:
        list_calls["count"] += 1
        if list_calls["count"] == 1:
            return _make_paginator([stale_info])
        return _make_paginator([_make_sandbox_info("sb-fresh", started_at=200)])

    mock_cls.list = _list
    mock_cls.connect = AsyncMock(return_value=stale)
    mock_cls.create = AsyncMock(return_value=fresh)

    backend = E2BSandboxBackend(api_key=_API_KEY, template="base")
    with _patch_sandbox_query():
        with patch.object(backend, "_get_sandbox_cls", return_value=mock_cls):
            handle = await backend.find_or_create_session("ev:1")

    mock_cls.create.assert_awaited_once()
    assert handle is fresh


@pytest.mark.asyncio
async def test_install_packages_runs_on_create_only() -> None:
    """Install runs on the CREATE branch only. Reusing an existing sandbox
    must NOT re-install."""
    older_info = _make_sandbox_info("sb-older", started_at=100)
    reused = MagicMock()
    reused.is_running = AsyncMock(return_value=True)
    reused.run_code = AsyncMock(return_value=MagicMock(error=None))

    mock_cls = _make_mock_sandbox_cls()
    mock_cls.list = MagicMock(return_value=_make_paginator([older_info]))
    mock_cls.connect = AsyncMock(return_value=reused)

    backend = E2BSandboxBackend(api_key=_API_KEY, template="base", packages=["cowsay"])
    with _patch_sandbox_query():
        with patch.object(backend, "_get_sandbox_cls", return_value=mock_cls):
            await backend.find_or_create_session("ev:1")

    # reused.run_code is the install probe; not called on the reuse path.
    reused.run_code.assert_not_called()


@pytest.mark.asyncio
async def test_install_packages_runs_via_run_code_on_create() -> None:
    """Non-empty packages on create path → run_code with a pip install snippet."""
    mock_cls = _make_mock_sandbox_cls()
    backend = E2BSandboxBackend(api_key=_API_KEY, template="base", packages=["cowsay"])
    with _patch_sandbox_query():
        with patch.object(backend, "_get_sandbox_cls", return_value=mock_cls):
            await backend.find_or_create_session("ev:1")
    instance = mock_cls.create.return_value
    instance.run_code.assert_called_once()
    code_arg = instance.run_code.call_args.args[0]
    assert "pip" in code_arg and "cowsay" in code_arg


@pytest.mark.asyncio
async def test_pip_install_failure_raises_from_find_or_create() -> None:
    """A failing install on create surfaces as RuntimeError, no cache."""
    mock_cls = _make_mock_sandbox_cls()
    mock_cls.create.return_value.run_code = AsyncMock(
        return_value=MagicMock(
            logs=MagicMock(stdout=[], stderr=[]),
            error="ModuleNotFoundError: No module named pip",
        )
    )
    backend = E2BSandboxBackend(api_key=_API_KEY, template="base", packages=["bad-pkg"])
    with _patch_sandbox_query():
        with patch.object(backend, "_get_sandbox_cls", return_value=mock_cls):
            with pytest.raises(RuntimeError):
                await backend.find_or_create_session("ev:1")


@pytest.mark.asyncio
async def test_package_specs_pass_through_to_subprocess_unmodified() -> None:
    """Version specifiers and extras must reach pip exactly as configured."""
    mock_cls = _make_mock_sandbox_cls()
    specs = ["numpy>=1.0", "requests[security]", "pandas==2.1.0"]
    backend = E2BSandboxBackend(api_key=_API_KEY, template="base", packages=specs)
    with _patch_sandbox_query():
        with patch.object(backend, "_get_sandbox_cls", return_value=mock_cls):
            await backend.find_or_create_session("ev:1")
    code_arg = mock_cls.create.return_value.run_code.call_args.args[0]
    for spec in specs:
        assert repr(spec) in code_arg, (
            f"Expected {spec!r} (Python repr) in generated code; got: {code_arg}"
        )


@pytest.mark.asyncio
async def test_close_session_kills_all_matches() -> None:
    """``close_session`` lists by the key and ``kill``s every match."""
    info_a = _make_sandbox_info("sb-a", started_at=100)
    info_b = _make_sandbox_info("sb-b", started_at=200)
    mock_cls = _make_mock_sandbox_cls()
    mock_cls.list = MagicMock(return_value=_make_paginator([info_a, info_b]))

    backend = E2BSandboxBackend(api_key=_API_KEY, template="base")
    with _patch_sandbox_query():
        with patch.object(backend, "_get_sandbox_cls", return_value=mock_cls):
            await backend.close_session("ev:1")

    killed_ids = sorted(call.args[0] for call in mock_cls.kill.call_args_list)
    assert killed_ids == ["sb-a", "sb-b"]


@pytest.mark.asyncio
async def test_close_session_is_idempotent_when_no_match() -> None:
    mock_cls = _make_mock_sandbox_cls()  # list returns empty paginator
    backend = E2BSandboxBackend(api_key=_API_KEY, template="base")
    with _patch_sandbox_query():
        with patch.object(backend, "_get_sandbox_cls", return_value=mock_cls):
            await backend.close_session("ev:1")
    mock_cls.kill.assert_not_called()


def test_provider_session_id_default_is_passthrough() -> None:
    """E2B does not override ``provider_session_id`` — input == output."""
    backend = E2BSandboxBackend(api_key=_API_KEY, template="base")
    for key in ("evaluator:42", "inline:abc-123", "x" * 200):
        assert backend.provider_session_id(key) == key


@pytest.mark.asyncio
async def test_ephemeral_execute_installs_packages_before_run_code() -> None:
    """Ephemeral execute() must install configured packages before user code."""
    mock_cls = _make_mock_sandbox_cls()
    sandbox_instance = mock_cls.create.return_value
    sandbox_instance.__aenter__ = AsyncMock(return_value=sandbox_instance)
    sandbox_instance.__aexit__ = AsyncMock(return_value=None)
    backend = E2BSandboxBackend(api_key=_API_KEY, template="base", packages=["cowsay"])
    with patch.object(backend, "_get_sandbox_cls", return_value=mock_cls):
        await backend.execute("print('hi')", session_key="s1")
    assert sandbox_instance.run_code.await_count == 2
    install_code = sandbox_instance.run_code.call_args_list[0].args[0]
    user_code = sandbox_instance.run_code.call_args_list[1].args[0]
    assert "pip" in install_code and "cowsay" in install_code
    assert user_code == "print('hi')"


@pytest.mark.parametrize(
    "config,expected_packages",
    [
        ({"dependencies": {"packages": ["cowsay"]}}, ["cowsay"]),
        ({}, []),
    ],
)
def test_build_backend_wires_packages(config: dict[str, Any], expected_packages: list[str]) -> None:
    adapter = E2BAdapter()
    backend: E2BSandboxBackend = adapter.build_backend(  # type: ignore[assignment]
        {_CANONICAL_API_KEY: "k", **config}
    )
    assert backend._packages == expected_packages


def test_build_backend_requires_api_key() -> None:
    adapter = E2BAdapter()
    with pytest.raises(ValueError, match=_CANONICAL_API_KEY):
        adapter.build_backend({})
    with pytest.raises(ValueError, match=_CANONICAL_API_KEY):
        adapter.build_backend({_CANONICAL_API_KEY: ""})


class TestE2BCrossWrapperConvergence:
    """Two FRESH ``E2BSandboxBackend`` wrappers calling ``find_or_create_session``
    with the same key must converge on a single remote sandbox via the
    metadata-list path. Mirrors the real factory's no-cache shape — backend
    instances are ephemeral, convergence is bound provider-side.
    """

    @pytest.mark.asyncio
    async def test_two_fresh_wrappers_converge_on_one_remote_sandbox(self) -> None:
        # Shared state across the two mock SDK shims simulates the provider's
        # cross-replica metadata store.
        provider_state: dict[str, MagicMock] = {}

        def _make_cls() -> MagicMock:
            cls = MagicMock()
            cls.create = AsyncMock()
            cls.connect = AsyncMock()
            cls.kill = AsyncMock()

            async def _create_side_effect(**kwargs: Any) -> MagicMock:
                key = kwargs["metadata"][_METADATA_SESSION_KEY]
                sandbox = MagicMock()
                sandbox.sandbox_id = f"sb-{key}"
                sandbox.is_running = AsyncMock(return_value=True)
                sandbox.run_code = AsyncMock(return_value=MagicMock(error=None))
                provider_state[key] = sandbox
                return sandbox

            cls.create.side_effect = _create_side_effect

            async def _connect_side_effect(sandbox_id: str, **kwargs: Any) -> MagicMock:
                for sb in provider_state.values():
                    if sb.sandbox_id == sandbox_id:
                        return sb
                raise RuntimeError(f"no such sandbox {sandbox_id}")

            cls.connect.side_effect = _connect_side_effect

            def _list(**kwargs: Any) -> Any:
                query = kwargs.get("query") or {}
                metadata = query.get("metadata") or {}
                key = metadata.get(_METADATA_SESSION_KEY)
                if key in provider_state:
                    sb = provider_state[key]
                    info = MagicMock()
                    info.sandbox_id = sb.sandbox_id
                    info.started_at = 100
                    return _make_paginator([info])
                return _make_paginator([])

            cls.list = _list
            return cls

        cls_a = _make_cls()
        cls_b = _make_cls()
        backend_a = E2BSandboxBackend(api_key=_API_KEY, template="base")
        backend_b = E2BSandboxBackend(api_key=_API_KEY, template="base")
        with _patch_sandbox_query():
            with patch.object(backend_a, "_get_sandbox_cls", return_value=cls_a):
                handle_a = await backend_a.find_or_create_session("ev:1")
            with patch.object(backend_b, "_get_sandbox_cls", return_value=cls_b):
                handle_b = await backend_b.find_or_create_session("ev:1")

        # Wrapper A created. Wrapper B connected to the existing one.
        cls_a.create.assert_awaited_once()
        cls_b.create.assert_not_awaited()
        cls_b.connect.assert_awaited_once()
        # The handles point at the same underlying sandbox_id (cross-wrapper
        # convergence). Cast through Any: find_or_create_session is typed
        # ``object`` (opaque handle) by contract, so the test reaches through
        # to assert on the concrete e2b-shaped attribute.
        assert cast(Any, handle_a).sandbox_id == cast(Any, handle_b).sandbox_id


def test_config_fingerprint_is_stable_and_changes_with_runtime_affecting_fields() -> None:
    """Same config → same fingerprint; packages / internet_access / env-var
    keys / template each fragment the fingerprint; secret plaintext rotation
    under the same env-var keys does NOT."""

    def _build(
        *,
        packages: list[str] | None = None,
        allow_internet_access: bool = True,
        user_env: dict[str, str] | None = None,
        template: str | None = "base",
    ) -> E2BSandboxBackend:
        return E2BSandboxBackend(
            api_key=_API_KEY,
            template=template,
            packages=packages,
            allow_internet_access=allow_internet_access,
            user_env=user_env,
        )

    base = _build(
        packages=["numpy"],
        allow_internet_access=True,
        user_env={"FOO": "bar"},
    )
    assert (
        base.config_fingerprint()
        == _build(
            packages=["numpy"],
            allow_internet_access=True,
            user_env={"FOO": "bar"},
        ).config_fingerprint()
    )
    assert (
        base.config_fingerprint()
        != _build(
            packages=["numpy", "pandas"],
            allow_internet_access=True,
            user_env={"FOO": "bar"},
        ).config_fingerprint()
    )
    assert (
        base.config_fingerprint()
        != _build(
            packages=["numpy"],
            allow_internet_access=False,
            user_env={"FOO": "bar"},
        ).config_fingerprint()
    )
    assert (
        base.config_fingerprint()
        != _build(
            packages=["numpy"],
            allow_internet_access=True,
            user_env={"FOO": "bar", "EXTRA": "x"},
        ).config_fingerprint()
    )
    # Secret-value-only change must NOT change the fingerprint.
    assert (
        base.config_fingerprint()
        == _build(
            packages=["numpy"],
            allow_internet_access=True,
            user_env={"FOO": "ROTATED"},
        ).config_fingerprint()
    )
