"""Unit tests for VercelSandboxBackend.

Scope: SDK kwarg shapes, runtime package install, network_policy forwarding,
``find_or_create_session`` via the module-level ``_session_id_map``, and
the hard-coded ``timeout=timedelta(seconds=600)`` create kwarg.
"""

from __future__ import annotations

import sys
from datetime import timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from starlette.datastructures import Secret

from phoenix.server.sandbox import vercel_backend
from phoenix.server.sandbox.vercel_backend import (
    _VERCEL_CREATE_TIMEOUT,
    VercelSandboxBackend,
)

_TOKEN = Secret("t")
_PROJECT = Secret("p")
_TEAM = Secret("m")


@pytest.fixture(autouse=True)
def _reset_vercel_module_state() -> Any:
    """Clear ``_session_id_map`` / ``_session_id_locks`` between tests so
    cross-test state from the shared module-level dict doesn't bleed in."""
    vercel_backend._session_id_map.clear()
    vercel_backend._session_id_locks.clear()
    yield
    vercel_backend._session_id_map.clear()
    vercel_backend._session_id_locks.clear()


def _make_vercel_sdk_mock(
    captured_kwargs: list[dict[str, Any]] | None = None,
    sandbox_ids: list[str] | None = None,
) -> tuple[MagicMock, dict[str, MagicMock]]:
    """Return (mock vercel.sandbox module, sandboxes-by-id index).

    ``AsyncSandbox.create`` builds a fresh sandbox with sequential id from
    ``sandbox_ids`` (defaults to ``["sb-1", "sb-2", ...]``).
    ``AsyncSandbox.get`` returns the sandbox previously stored under its id.
    """
    ids = list(sandbox_ids or [f"sb-{i}" for i in range(1, 100)])
    sandboxes: dict[str, MagicMock] = {}

    async def _create(**kwargs: Any) -> MagicMock:
        if captured_kwargs is not None:
            captured_kwargs.append(dict(kwargs))
        sandbox = MagicMock()
        sandbox.sandbox_id = ids[len(sandboxes)]
        sandbox.status = "running"
        sandbox.stop = AsyncMock()
        sandbox.client = MagicMock()
        sandbox.client.aclose = AsyncMock()
        sandboxes[sandbox.sandbox_id] = sandbox
        return sandbox

    async def _get(*, sandbox_id: str, **kwargs: Any) -> MagicMock:
        if sandbox_id not in sandboxes:
            raise RuntimeError(f"no such sandbox {sandbox_id}")
        return sandboxes[sandbox_id]

    sdk = MagicMock()
    sdk.AsyncSandbox = MagicMock()
    sdk.AsyncSandbox.create = _create
    sdk.AsyncSandbox.get = _get
    return sdk, sandboxes


@pytest.fixture
def patched_vercel_sdk_with_kwargs() -> Any:
    captured_kwargs: list[dict[str, Any]] = []
    sdk, _ = _make_vercel_sdk_mock(captured_kwargs=captured_kwargs)
    parent = MagicMock()
    parent.sandbox = sdk
    with patch.dict(sys.modules, {"vercel": parent, "vercel.sandbox": sdk}):
        yield captured_kwargs


def test_constructor_rejects_missing_credentials() -> None:
    with pytest.raises(ValueError, match="token, project_id, and team_id"):
        VercelSandboxBackend(
            token=Secret(""), project_id=_PROJECT, team_id=_TEAM, language="PYTHON"
        )
    with pytest.raises(ValueError, match="token, project_id, and team_id"):
        VercelSandboxBackend(token=_TOKEN, project_id=Secret(""), team_id=_TEAM, language="PYTHON")
    with pytest.raises(ValueError, match="token, project_id, and team_id"):
        VercelSandboxBackend(
            token=_TOKEN, project_id=_PROJECT, team_id=Secret(""), language="PYTHON"
        )


def _make_install_sandbox_mock(
    install_exit_code: int = 0,
    install_stderr: str = "",
) -> tuple[MagicMock, list[tuple[str, list[str]]]]:
    captured: list[tuple[str, list[str]]] = []

    async def _run_command(cmd: str, args: list[str], **kwargs: Any) -> Any:
        captured.append((cmd, list(args)))
        result = MagicMock()
        is_install = len(captured) == 1
        result.exit_code = install_exit_code if is_install else 0
        result.stdout = AsyncMock(return_value="")
        result.stderr = AsyncMock(return_value=install_stderr if is_install else "")
        return result

    sandbox = MagicMock()
    sandbox.run_command = _run_command
    sandbox.sandbox_id = "sb-1"
    sandbox.status = "running"
    sandbox.stop = AsyncMock()
    sandbox.client = MagicMock()
    sandbox.client.aclose = AsyncMock()
    return sandbox, captured


@pytest.mark.asyncio
async def test_find_or_create_session_installs_python_packages_with_pip_user(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PYTHON + packages → find_or_create_session issues
    ``python3 -m pip install --user <pkgs>`` on the create branch.
    """
    sandbox_mock, captured = _make_install_sandbox_mock()
    backend = VercelSandboxBackend(
        token=_TOKEN,
        project_id=_PROJECT,
        team_id=_TEAM,
        language="PYTHON",
        packages=["requests", "numpy"],
    )

    async def _fake_create_sandbox() -> Any:
        return sandbox_mock

    monkeypatch.setattr(backend, "_create_sandbox", _fake_create_sandbox)
    await backend.find_or_create_session("s1")

    assert captured == [("python3", ["-m", "pip", "install", "--user", "requests", "numpy"])]
    assert vercel_backend._session_id_map["s1"] == "sb-1"


@pytest.mark.asyncio
async def test_find_or_create_session_installs_typescript_packages_with_npm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sandbox_mock, captured = _make_install_sandbox_mock()
    backend = VercelSandboxBackend(
        token=_TOKEN,
        project_id=_PROJECT,
        team_id=_TEAM,
        language="TYPESCRIPT",
        packages=["lodash"],
    )

    async def _fake_create_sandbox() -> Any:
        return sandbox_mock

    monkeypatch.setattr(backend, "_create_sandbox", _fake_create_sandbox)
    await backend.find_or_create_session("s1")

    assert captured == [("npm", ["install", "lodash"])]
    assert vercel_backend._session_id_map["s1"] == "sb-1"


@pytest.mark.asyncio
async def test_find_or_create_session_install_failure_stops_sandbox_and_does_not_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Install failure → stop+aclose on the dead sandbox, no entry stored in
    ``_session_id_map``, RuntimeError propagated.
    """
    sandbox_mock, _captured = _make_install_sandbox_mock(
        install_exit_code=1,
        install_stderr="pip: package not found",
    )
    backend = VercelSandboxBackend(
        token=_TOKEN,
        project_id=_PROJECT,
        team_id=_TEAM,
        language="PYTHON",
        packages=["nonexistent-package"],
    )

    async def _fake_create_sandbox() -> Any:
        return sandbox_mock

    monkeypatch.setattr(backend, "_create_sandbox", _fake_create_sandbox)

    with pytest.raises(RuntimeError, match="pip: package not found"):
        await backend.find_or_create_session("s1")

    sandbox_mock.stop.assert_awaited_once()
    sandbox_mock.client.aclose.assert_awaited_once()
    assert "s1" not in vercel_backend._session_id_map


@pytest.mark.asyncio
async def test_ephemeral_execute_runs_install_before_user_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sandbox_mock, captured = _make_install_sandbox_mock()
    backend = VercelSandboxBackend(
        token=_TOKEN,
        project_id=_PROJECT,
        team_id=_TEAM,
        language="PYTHON",
        packages=["requests"],
    )

    async def _fake_create_sandbox() -> Any:
        return sandbox_mock

    monkeypatch.setattr(backend, "_create_sandbox", _fake_create_sandbox)

    result = await backend.execute("print('hello')", session_key="ephemeral")

    assert len(captured) >= 2
    assert captured[0] == ("python3", ["-m", "pip", "install", "--user", "requests"])
    sandbox_mock.stop.assert_awaited_once()
    sandbox_mock.client.aclose.assert_awaited_once()
    # Ephemeral execute MUST NOT register the sandbox in the shared map.
    assert "ephemeral" not in vercel_backend._session_id_map
    assert result.error is None or result.error == ""


@pytest.mark.asyncio
async def test_create_sandbox_passes_timeout_as_timedelta(
    patched_vercel_sdk_with_kwargs: list[dict[str, Any]],
) -> None:
    """Every ``AsyncSandbox.create`` must carry ``timeout=timedelta(seconds=600)``.

    The Vercel SDK interprets bare-int ``timeout`` as milliseconds, so we
    pass a ``timedelta`` explicitly to make the unit unambiguous.
    """
    captured_kwargs = patched_vercel_sdk_with_kwargs
    backend = VercelSandboxBackend(
        token=_TOKEN, project_id=_PROJECT, team_id=_TEAM, language="PYTHON"
    )
    await backend._create_sandbox()
    assert len(captured_kwargs) == 1
    timeout_kwarg = captured_kwargs[0].get("timeout")
    assert isinstance(timeout_kwarg, timedelta), (
        f"timeout kwarg must be timedelta; got {type(timeout_kwarg)!r}"
    )
    assert timeout_kwarg == _VERCEL_CREATE_TIMEOUT == timedelta(seconds=600)


@pytest.mark.asyncio
async def test_create_sandbox_forwards_access_token_triple_as_kwargs(
    patched_vercel_sdk_with_kwargs: list[dict[str, Any]],
) -> None:
    captured_kwargs = patched_vercel_sdk_with_kwargs
    backend = VercelSandboxBackend(
        token=Secret("db-resolved-token"),
        project_id=Secret("proj-id"),
        team_id=Secret("team-id"),
        language="PYTHON",
    )
    await backend._create_sandbox()
    assert len(captured_kwargs) == 1
    kwargs = captured_kwargs[0]
    assert kwargs.get("token") == "db-resolved-token"
    assert kwargs.get("project_id") == "proj-id"
    assert kwargs.get("team_id") == "team-id"
    assert backend.secret_values == frozenset({"db-resolved-token", "proj-id", "team-id"})


@pytest.mark.asyncio
async def test_create_sandbox_forwards_network_policy_allow_all(
    patched_vercel_sdk_with_kwargs: list[dict[str, Any]],
) -> None:
    captured_kwargs = patched_vercel_sdk_with_kwargs
    backend = VercelSandboxBackend(
        token=_TOKEN, project_id=_PROJECT, team_id=_TEAM, language="PYTHON", internet_access=True
    )
    await backend._create_sandbox()
    assert captured_kwargs[0].get("network_policy") == "allow-all"


@pytest.mark.asyncio
async def test_create_sandbox_forwards_network_policy_deny_all(
    patched_vercel_sdk_with_kwargs: list[dict[str, Any]],
) -> None:
    captured_kwargs = patched_vercel_sdk_with_kwargs
    backend = VercelSandboxBackend(
        token=_TOKEN, project_id=_PROJECT, team_id=_TEAM, language="PYTHON", internet_access=False
    )
    await backend._create_sandbox()
    assert captured_kwargs[0].get("network_policy") == "deny-all"


@pytest.mark.asyncio
async def test_create_sandbox_omits_network_policy_when_internet_access_unset(
    patched_vercel_sdk_with_kwargs: list[dict[str, Any]],
) -> None:
    captured_kwargs = patched_vercel_sdk_with_kwargs
    backend = VercelSandboxBackend(
        token=_TOKEN, project_id=_PROJECT, team_id=_TEAM, language="PYTHON"
    )
    await backend._create_sandbox()
    assert "network_policy" not in captured_kwargs[0]


@pytest.mark.parametrize(
    "adapter_module_path,adapter_cls_name,language",
    [
        ("phoenix.server.sandbox.vercel_backend", "VercelPythonAdapter", "PYTHON"),
        ("phoenix.server.sandbox.vercel_backend", "VercelTypescriptAdapter", "TYPESCRIPT"),
    ],
)
def test_adapter_build_backend_maps_internet_access_allow(
    adapter_module_path: str, adapter_cls_name: str, language: str
) -> None:
    import importlib

    mod = importlib.import_module(adapter_module_path)
    adapter = getattr(mod, adapter_cls_name)()
    backend = adapter.build_backend(
        {
            "internet_access": {"mode": "allow"},
            "VERCEL_TOKEN": "t",
            "VERCEL_PROJECT_ID": "p",
            "VERCEL_TEAM_ID": "m",
        }
    )
    assert backend._internet_access is True
    assert backend._language == language


@pytest.mark.parametrize(
    "adapter_module_path,adapter_cls_name",
    [
        ("phoenix.server.sandbox.vercel_backend", "VercelPythonAdapter"),
        ("phoenix.server.sandbox.vercel_backend", "VercelTypescriptAdapter"),
    ],
)
def test_adapter_build_backend_maps_internet_access_deny_no_packages(
    adapter_module_path: str, adapter_cls_name: str
) -> None:
    import importlib

    mod = importlib.import_module(adapter_module_path)
    adapter = getattr(mod, adapter_cls_name)()
    backend = adapter.build_backend(
        {
            "internet_access": {"mode": "deny"},
            "VERCEL_TOKEN": "t",
            "VERCEL_PROJECT_ID": "p",
            "VERCEL_TEAM_ID": "m",
        }
    )
    assert backend._internet_access is False


@pytest.mark.parametrize(
    "adapter_module_path,adapter_cls_name",
    [
        ("phoenix.server.sandbox.vercel_backend", "VercelPythonAdapter"),
        ("phoenix.server.sandbox.vercel_backend", "VercelTypescriptAdapter"),
    ],
)
def test_adapter_build_backend_omits_internet_access_when_absent(
    adapter_module_path: str, adapter_cls_name: str
) -> None:
    import importlib

    mod = importlib.import_module(adapter_module_path)
    adapter = getattr(mod, adapter_cls_name)()
    backend = adapter.build_backend(
        {
            "VERCEL_TOKEN": "t",
            "VERCEL_PROJECT_ID": "p",
            "VERCEL_TEAM_ID": "m",
        }
    )
    assert backend._internet_access is None


@pytest.mark.parametrize(
    "adapter_cls_name",
    ["VercelPythonAdapter", "VercelTypescriptAdapter"],
)
def test_adapter_build_backend_fails_closed_on_missing_triple(adapter_cls_name: str) -> None:
    import importlib

    mod = importlib.import_module("phoenix.server.sandbox.vercel_backend")
    adapter = getattr(mod, adapter_cls_name)()
    with pytest.raises(ValueError, match="Vercel sandbox authentication is not configured"):
        adapter.build_backend({"VERCEL_TOKEN": "t"})


class TestFindOrCreateSessionConvergesViaSessionIdMap:
    """The module-level ``_session_id_map`` lets two fresh wrappers in the
    same process converge on a single remote sandbox for the same key.
    """

    @pytest.mark.asyncio
    async def test_two_fresh_wrappers_share_one_sandbox_id(self) -> None:
        sdk, sandboxes = _make_vercel_sdk_mock()
        parent = MagicMock()
        parent.sandbox = sdk
        with patch.dict(sys.modules, {"vercel": parent, "vercel.sandbox": sdk}):
            backend_a = VercelSandboxBackend(
                token=_TOKEN, project_id=_PROJECT, team_id=_TEAM, language="PYTHON"
            )
            backend_b = VercelSandboxBackend(
                token=_TOKEN, project_id=_PROJECT, team_id=_TEAM, language="PYTHON"
            )
            handle_a = await backend_a.find_or_create_session("ev:1")
            handle_b = await backend_b.find_or_create_session("ev:1")

        # Only ONE remote sandbox was created across both wrappers.
        assert len(sandboxes) == 1
        # Both handles point at the same sandbox_id.
        assert handle_a.sandbox_id == handle_b.sandbox_id
        # The map binds the key to that sandbox_id.
        assert vercel_backend._session_id_map["ev:1"] == handle_a.sandbox_id

    @pytest.mark.asyncio
    async def test_idempotent_on_same_wrapper(self) -> None:
        sdk, sandboxes = _make_vercel_sdk_mock()
        parent = MagicMock()
        parent.sandbox = sdk
        with patch.dict(sys.modules, {"vercel": parent, "vercel.sandbox": sdk}):
            backend = VercelSandboxBackend(
                token=_TOKEN, project_id=_PROJECT, team_id=_TEAM, language="PYTHON"
            )
            h1 = await backend.find_or_create_session("ev:1")
            h2 = await backend.find_or_create_session("ev:1")

        assert len(sandboxes) == 1
        assert h1.sandbox_id == h2.sandbox_id

    @pytest.mark.asyncio
    async def test_stale_binding_drops_and_recreates(self) -> None:
        """If the previously-bound sandbox returns a dead status on get,
        the binding is dropped and a fresh sandbox is created."""
        sdk, sandboxes = _make_vercel_sdk_mock()
        parent = MagicMock()
        parent.sandbox = sdk
        with patch.dict(sys.modules, {"vercel": parent, "vercel.sandbox": sdk}):
            backend = VercelSandboxBackend(
                token=_TOKEN, project_id=_PROJECT, team_id=_TEAM, language="PYTHON"
            )
            h1 = await backend.find_or_create_session("ev:1")
            # Mark the bound sandbox as dead.
            sandboxes[h1.sandbox_id].status = "stopped"
            h2 = await backend.find_or_create_session("ev:1")

        assert h1.sandbox_id != h2.sandbox_id
        # Stale entry replaced; map now points at the fresh id.
        assert vercel_backend._session_id_map["ev:1"] == h2.sandbox_id


class TestCloseSession:
    @pytest.mark.asyncio
    async def test_close_session_pops_map_and_lock_and_stops_sandbox(self) -> None:
        sdk, sandboxes = _make_vercel_sdk_mock()
        parent = MagicMock()
        parent.sandbox = sdk
        with patch.dict(sys.modules, {"vercel": parent, "vercel.sandbox": sdk}):
            backend = VercelSandboxBackend(
                token=_TOKEN, project_id=_PROJECT, team_id=_TEAM, language="PYTHON"
            )
            handle = await backend.find_or_create_session("ev:1")
            assert "ev:1" in vercel_backend._session_id_map
            assert "ev:1" in vercel_backend._session_id_locks

            await backend.close_session("ev:1")

        sandbox = sandboxes[handle.sandbox_id]
        sandbox.stop.assert_awaited_once()
        sandbox.client.aclose.assert_awaited_once()
        # Both map and lock entries popped.
        assert "ev:1" not in vercel_backend._session_id_map
        assert "ev:1" not in vercel_backend._session_id_locks

    @pytest.mark.asyncio
    async def test_close_session_is_idempotent_on_unknown_key(self) -> None:
        sdk, _ = _make_vercel_sdk_mock()
        parent = MagicMock()
        parent.sandbox = sdk
        with patch.dict(sys.modules, {"vercel": parent, "vercel.sandbox": sdk}):
            backend = VercelSandboxBackend(
                token=_TOKEN, project_id=_PROJECT, team_id=_TEAM, language="PYTHON"
            )
            # No-op — must not raise.
            await backend.close_session("never-bound")


def test_provider_session_id_default_is_passthrough() -> None:
    backend = VercelSandboxBackend(
        token=_TOKEN, project_id=_PROJECT, team_id=_TEAM, language="PYTHON"
    )
    for key in ("evaluator:42", "inline:abc-123", "x" * 200):
        assert backend.provider_session_id(key) == key


def test_config_fingerprint_is_stable_and_changes_with_runtime_affecting_fields() -> None:
    """Same config → same fingerprint; packages / internet_access / env-var
    keys / language each fragment the fingerprint; secret plaintext rotation
    under the same env-var keys does NOT."""

    def _build(
        *,
        language: str = "PYTHON",
        packages: list[str] | None = None,
        internet_access: bool | None = True,
        user_env: dict[str, str] | None = None,
    ) -> VercelSandboxBackend:
        return VercelSandboxBackend(
            token=_TOKEN,
            project_id=_PROJECT,
            team_id=_TEAM,
            language=language,
            packages=packages,
            internet_access=internet_access,
            user_env=user_env,
        )

    base = _build(
        packages=["lodash@^4.17"],
        internet_access=True,
        user_env={"FOO": "bar"},
        language="TYPESCRIPT",
    )
    assert (
        base.config_fingerprint()
        == _build(
            packages=["lodash@^4.17"],
            internet_access=True,
            user_env={"FOO": "bar"},
            language="TYPESCRIPT",
        ).config_fingerprint()
    )
    assert (
        base.config_fingerprint()
        != _build(
            packages=["lodash@^4.17", "axios"],
            internet_access=True,
            user_env={"FOO": "bar"},
            language="TYPESCRIPT",
        ).config_fingerprint()
    )
    assert (
        base.config_fingerprint()
        != _build(
            packages=["lodash@^4.17"],
            internet_access=False,
            user_env={"FOO": "bar"},
            language="TYPESCRIPT",
        ).config_fingerprint()
    )
    assert (
        base.config_fingerprint()
        != _build(
            packages=["lodash@^4.17"],
            internet_access=True,
            user_env={"FOO": "bar", "BAZ": "qux"},
            language="TYPESCRIPT",
        ).config_fingerprint()
    )
    # Language (Python vs TypeScript) fragments because the remote runtime
    # is different.
    assert (
        base.config_fingerprint()
        != _build(
            packages=["lodash@^4.17"],
            internet_access=True,
            user_env={"FOO": "bar"},
            language="PYTHON",
        ).config_fingerprint()
    )
    # Secret-value-only change must NOT change the fingerprint.
    assert (
        base.config_fingerprint()
        == _build(
            packages=["lodash@^4.17"],
            internet_access=True,
            user_env={"FOO": "ROTATED"},
            language="TYPESCRIPT",
        ).config_fingerprint()
    )
