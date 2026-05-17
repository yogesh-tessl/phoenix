"""Unit tests for DaytonaSandboxBackend code_run kwarg correctness and
``find_or_create_session`` provider-side binding via the ``phoenix_session_key``
label.
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from starlette.datastructures import Secret

from phoenix.server.sandbox.daytona_backend import (
    _AUTO_ARCHIVE_INTERVAL_MIN,
    _AUTO_STOP_INTERVAL_MIN,
    _LABEL_SESSION_KEY,
)

_API_KEY = Secret("test-key")
_ALT_KEY = Secret("key")


class _CodeRunParams:
    def __init__(
        self,
        argv: list[str] | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        self.argv = argv
        self.env = env


class _CreateSandboxFromSnapshotParams:
    def __init__(
        self,
        language: str | None = None,
        network_block_all: bool | None = None,
        labels: dict[str, str] | None = None,
        auto_stop_interval: int | None = None,
        auto_archive_interval: int | None = None,
        **kwargs: object,
    ) -> None:
        self.language = language
        self.network_block_all = network_block_all
        self.labels = labels
        self.auto_stop_interval = auto_stop_interval
        self.auto_archive_interval = auto_archive_interval
        for key, value in kwargs.items():
            setattr(self, key, value)


class _DaytonaConfig:
    def __init__(
        self,
        api_key: str | None = None,
        api_url: str | None = None,
        **kwargs: object,
    ) -> None:
        self.api_key = api_key
        self.api_url = api_url
        for key, value in kwargs.items():
            setattr(self, key, value)


def _make_daytona_mocks() -> tuple[MagicMock, MagicMock]:
    process_mod = MagicMock()
    process_mod.CodeRunParams = _CodeRunParams

    daytona_mod = MagicMock()
    daytona_mod.CodeRunParams = _CodeRunParams
    daytona_mod.DaytonaConfig = _DaytonaConfig
    daytona_mod.CreateSandboxFromSnapshotParams = _CreateSandboxFromSnapshotParams

    workspace = MagicMock()
    workspace.process.code_run = AsyncMock(return_value=MagicMock(result="ok", exit_code=0))
    workspace.id = "sb-new"
    workspace.state = "STARTED"
    client = daytona_mod.AsyncDaytona.return_value
    client.create = AsyncMock(return_value=workspace)
    client.delete = AsyncMock()
    client.get = AsyncMock()
    # Default: list returns an empty result.
    list_response = MagicMock()
    list_response.items = []
    client.list = AsyncMock(return_value=list_response)

    return daytona_mod, process_mod


def _patch_sandbox_state(daytona_mod: MagicMock) -> Any:
    """Make ``from daytona_api_client_async.models.sandbox_state import SandboxState`` resolvable."""
    state_mod = MagicMock()
    state_mod.SandboxState = MagicMock()
    state_mod.SandboxState.STARTED = "STARTED"
    pkg = MagicMock()
    pkg.models = MagicMock()
    pkg.models.sandbox_state = state_mod
    return patch.dict(
        sys.modules,
        {
            "daytona_api_client_async": pkg,
            "daytona_api_client_async.models": pkg.models,
            "daytona_api_client_async.models.sandbox_state": state_mod,
        },
    )


class TestCodeRunParamsKwarg:
    @pytest.mark.asyncio
    async def test_execute_passes_params_not_envs(self) -> None:
        """code_run must receive params=CodeRunParams(env=...) instead of envs=."""
        daytona_mod, process_mod = _make_daytona_mocks()
        user_env = {"MY_KEY": "my_val"}

        modules = {
            "daytona_sdk": daytona_mod,
            "daytona_sdk.common": MagicMock(),
            "daytona_sdk.common.process": process_mod,
        }
        with patch.dict(sys.modules, modules):
            from phoenix.server.sandbox.daytona_backend import DaytonaSandboxBackend

            backend = DaytonaSandboxBackend(api_key=_API_KEY, user_env=user_env)
            await backend.execute("print('hi')", session_key="s1")

        workspace = daytona_mod.AsyncDaytona.return_value.create.return_value
        call_args = workspace.process.code_run.call_args
        assert call_args is not None
        assert "envs" not in call_args.kwargs
        assert "params" in call_args.kwargs
        params = call_args.kwargs["params"]
        assert isinstance(params, _CodeRunParams)
        assert params.env == user_env

    @pytest.mark.asyncio
    async def test_execute_empty_user_env_passes_none_env(self) -> None:
        daytona_mod, process_mod = _make_daytona_mocks()
        modules = {
            "daytona_sdk": daytona_mod,
            "daytona_sdk.common": MagicMock(),
            "daytona_sdk.common.process": process_mod,
        }
        with patch.dict(sys.modules, modules):
            from phoenix.server.sandbox.daytona_backend import DaytonaSandboxBackend

            backend = DaytonaSandboxBackend(api_key=_API_KEY, user_env={})
            await backend.execute("1+1", session_key="s1")

        workspace = daytona_mod.AsyncDaytona.return_value.create.return_value
        params = workspace.process.code_run.call_args.kwargs["params"]
        assert params.env is None


class TestCreateParamsTTLAndLabelKwargs:
    """``find_or_create_session`` must tag the create params with the TTL
    kwargs (``auto_stop_interval=5``, ``auto_archive_interval=15``) and the
    ``phoenix_session_key`` label so cross-replica list-by-label converges.
    """

    @pytest.mark.asyncio
    async def test_create_params_carry_ttl_and_session_label(self) -> None:
        daytona_mod, process_mod = _make_daytona_mocks()
        modules = {
            "daytona_sdk": daytona_mod,
            "daytona_sdk.common": MagicMock(),
            "daytona_sdk.common.process": process_mod,
        }
        with patch.dict(sys.modules, modules), _patch_sandbox_state(daytona_mod):
            from phoenix.server.sandbox.daytona_backend import DaytonaSandboxBackend

            backend = DaytonaSandboxBackend(api_key=_API_KEY)
            await backend.find_or_create_session("evaluator:42")

        client = daytona_mod.AsyncDaytona.return_value
        params = client.create.call_args.args[0]
        assert isinstance(params, _CreateSandboxFromSnapshotParams)
        assert params.auto_stop_interval == _AUTO_STOP_INTERVAL_MIN == 5
        assert params.auto_archive_interval == _AUTO_ARCHIVE_INTERVAL_MIN == 15
        assert params.labels == {_LABEL_SESSION_KEY: "evaluator:42"}


class TestNetworkBlockAll:
    @pytest.mark.asyncio
    async def test_deny_mode_passes_network_block_all_true(self) -> None:
        daytona_mod, process_mod = _make_daytona_mocks()
        modules = {
            "daytona_sdk": daytona_mod,
            "daytona_sdk.common": MagicMock(),
            "daytona_sdk.common.process": process_mod,
        }
        with patch.dict(sys.modules, modules):
            from phoenix.server.sandbox.daytona_backend import DaytonaSandboxBackend

            backend = DaytonaSandboxBackend(api_key=_ALT_KEY, network_block_all=True)
            await backend.execute("1", session_key="s1")

        create_call = daytona_mod.AsyncDaytona.return_value.create.call_args
        params = create_call.args[0] if create_call.args else create_call.kwargs.get("params")
        assert isinstance(params, _CreateSandboxFromSnapshotParams)
        assert params.network_block_all is True

    @pytest.mark.asyncio
    async def test_allow_mode_omits_network_block_all(self) -> None:
        daytona_mod, process_mod = _make_daytona_mocks()
        modules = {
            "daytona_sdk": daytona_mod,
            "daytona_sdk.common": MagicMock(),
            "daytona_sdk.common.process": process_mod,
        }
        with patch.dict(sys.modules, modules):
            from phoenix.server.sandbox.daytona_backend import DaytonaSandboxBackend

            backend = DaytonaSandboxBackend(api_key=_ALT_KEY, network_block_all=False)
            await backend.execute("1", session_key="s1")

        create_call = daytona_mod.AsyncDaytona.return_value.create.call_args
        params = create_call.args[0] if create_call.args else create_call.kwargs.get("params")
        assert isinstance(params, _CreateSandboxFromSnapshotParams)
        assert params.network_block_all is None

    @pytest.mark.asyncio
    async def test_find_or_create_session_deny_passes_network_block_all(self) -> None:
        """``find_or_create_session`` also sets ``network_block_all=True`` on
        create params under deny mode."""
        daytona_mod, process_mod = _make_daytona_mocks()
        modules = {
            "daytona_sdk": daytona_mod,
            "daytona_sdk.common": MagicMock(),
            "daytona_sdk.common.process": process_mod,
        }
        with patch.dict(sys.modules, modules), _patch_sandbox_state(daytona_mod):
            from phoenix.server.sandbox.daytona_backend import DaytonaSandboxBackend

            backend = DaytonaSandboxBackend(api_key=_ALT_KEY, network_block_all=True)
            await backend.find_or_create_session("sess")

        params = daytona_mod.AsyncDaytona.return_value.create.call_args.args[0]
        assert isinstance(params, _CreateSandboxFromSnapshotParams)
        assert params.network_block_all is True


class TestEphemeralTeardown:
    @pytest.mark.asyncio
    async def test_remove_called_when_code_run_raises(self) -> None:
        daytona_mod, process_mod = _make_daytona_mocks()
        client = daytona_mod.AsyncDaytona.return_value
        client.create.return_value.process.code_run = AsyncMock(
            side_effect=RuntimeError("code_run failed")
        )

        modules = {
            "daytona_sdk": daytona_mod,
            "daytona_sdk.common": MagicMock(),
            "daytona_sdk.common.process": process_mod,
        }
        with patch.dict(sys.modules, modules):
            from phoenix.server.sandbox.daytona_backend import DaytonaSandboxBackend

            backend = DaytonaSandboxBackend(api_key=_API_KEY)
            result = await backend.execute("raise RuntimeError()", session_key="ephemeral")

        assert result.error is not None
        assert client.create.call_count == 1
        assert client.delete.call_count == 1

    @pytest.mark.asyncio
    async def test_remove_called_on_cancellation(self) -> None:
        daytona_mod, process_mod = _make_daytona_mocks()
        client = daytona_mod.AsyncDaytona.return_value

        async def _slow_code_run(*args: object, **kwargs: object) -> None:
            await asyncio.sleep(10)

        client.create.return_value.process.code_run = AsyncMock(side_effect=_slow_code_run)

        modules = {
            "daytona_sdk": daytona_mod,
            "daytona_sdk.common": MagicMock(),
            "daytona_sdk.common.process": process_mod,
        }
        with patch.dict(sys.modules, modules):
            from phoenix.server.sandbox.daytona_backend import DaytonaSandboxBackend

            backend = DaytonaSandboxBackend(api_key=_API_KEY)
            with pytest.raises((asyncio.TimeoutError, TimeoutError)):
                await asyncio.wait_for(
                    backend.execute("sleep(10)", session_key="ephemeral"),
                    timeout=0.05,
                )

        assert client.create.call_count == 1
        assert client.delete.call_count == 1


class TestBuildBackendCredentialValidation:
    def test_missing_api_key_raises_value_error(self) -> None:
        from phoenix.server.sandbox.daytona_backend import DaytonaPythonAdapter

        adapter = DaytonaPythonAdapter()
        with pytest.raises(ValueError, match="DAYTONA_API_KEY"):
            adapter.build_backend({})

    def test_empty_api_key_raises_value_error(self) -> None:
        from phoenix.server.sandbox.daytona_backend import DaytonaPythonAdapter

        adapter = DaytonaPythonAdapter()
        with pytest.raises(ValueError, match="PHOENIX_SANDBOX_DAYTONA_API_KEY"):
            adapter.build_backend({"PHOENIX_SANDBOX_DAYTONA_API_KEY": ""})


class TestTypescriptRouting:
    @pytest.mark.asyncio
    async def test_create_params_uses_typescript_language(self) -> None:
        daytona_mod, process_mod = _make_daytona_mocks()
        daytona_mod.CodeLanguage = MagicMock()
        daytona_mod.CodeLanguage.PYTHON = "python"
        daytona_mod.CodeLanguage.TYPESCRIPT = "typescript"

        modules = {
            "daytona_sdk": daytona_mod,
            "daytona_sdk.common": MagicMock(),
            "daytona_sdk.common.process": process_mod,
        }
        with patch.dict(sys.modules, modules):
            from phoenix.server.sandbox.daytona_backend import DaytonaSandboxBackend

            backend = DaytonaSandboxBackend(api_key=_API_KEY, language="TYPESCRIPT")
            await backend.execute("console.log('hi')", session_key="s1")

        create_call = daytona_mod.AsyncDaytona.return_value.create.call_args
        params = create_call.args[0] if create_call.args else create_call.kwargs.get("params")
        assert isinstance(params, _CreateSandboxFromSnapshotParams)
        assert params.language == "typescript"

    @pytest.mark.asyncio
    async def test_install_packages_uses_npm_argv_shape(self) -> None:
        """``find_or_create_session`` runs the install on create; assert the
        generated TS source uses ``spawnSync('npm', ['install', ...])`` with
        packages embedded as a JSON array literal."""
        daytona_mod, process_mod = _make_daytona_mocks()
        daytona_mod.CodeLanguage = MagicMock()
        daytona_mod.CodeLanguage.PYTHON = "python"
        daytona_mod.CodeLanguage.TYPESCRIPT = "typescript"

        modules = {
            "daytona_sdk": daytona_mod,
            "daytona_sdk.common": MagicMock(),
            "daytona_sdk.common.process": process_mod,
        }
        with patch.dict(sys.modules, modules), _patch_sandbox_state(daytona_mod):
            from phoenix.server.sandbox.daytona_backend import DaytonaSandboxBackend

            backend = DaytonaSandboxBackend(
                api_key=_API_KEY,
                packages=["is-odd"],
                language="TYPESCRIPT",
            )
            await backend.find_or_create_session("sess-ts")

        workspace = daytona_mod.AsyncDaytona.return_value.create.return_value
        assert workspace.process.code_run.call_count >= 1
        install_source = workspace.process.code_run.call_args_list[0].args[0]
        assert "npm" in install_source
        assert "install" in install_source
        assert "pip install" not in install_source
        assert '["is-odd"]' in install_source
        assert ("spawnSync" in install_source) or ("execFileSync" in install_source)
        assert "cwd: '/tmp'" in install_source or 'cwd: "/tmp"' in install_source


class TestFindOrCreateSessionConvergence:
    """Cross-wrapper provider-side binding: list-by-label, then connect (get)."""

    @pytest.mark.asyncio
    async def test_two_fresh_wrappers_converge_on_one_remote_sandbox(self) -> None:
        """A second wrapper finds the first's sandbox via list-by-label
        and reuses it instead of creating a new one."""
        daytona_mod, process_mod = _make_daytona_mocks()

        provider_state: dict[str, MagicMock] = {}

        def _client_factory() -> MagicMock:
            client = MagicMock()
            client.create = AsyncMock()
            client.get = AsyncMock()
            client.delete = AsyncMock()

            async def _create_side(params: Any) -> MagicMock:
                key = params.labels[_LABEL_SESSION_KEY]
                sb = MagicMock()
                sb.id = f"sb-{key}"
                sb.state = "STARTED"
                sb.process.code_run = AsyncMock(return_value=MagicMock(exit_code=0))
                provider_state[key] = sb
                return sb

            client.create.side_effect = _create_side

            async def _get_side(sandbox_id: str) -> MagicMock:
                for sb in provider_state.values():
                    if sb.id == sandbox_id:
                        return sb
                raise RuntimeError("not found")

            client.get.side_effect = _get_side

            async def _list_side(**kwargs: Any) -> MagicMock:
                labels = kwargs.get("labels") or {}
                key = labels.get(_LABEL_SESSION_KEY)
                resp = MagicMock()
                resp.items = [provider_state[key]] if key in provider_state else []
                return resp

            client.list = AsyncMock(side_effect=_list_side)
            return client

        client_a = _client_factory()
        client_b = _client_factory()

        modules = {
            "daytona_sdk": daytona_mod,
            "daytona_sdk.common": MagicMock(),
            "daytona_sdk.common.process": process_mod,
        }
        with patch.dict(sys.modules, modules), _patch_sandbox_state(daytona_mod):
            from phoenix.server.sandbox.daytona_backend import DaytonaSandboxBackend

            backend_a = DaytonaSandboxBackend(api_key=_API_KEY)
            backend_b = DaytonaSandboxBackend(api_key=_API_KEY)
            with patch.object(backend_a, "_get_client", return_value=client_a):
                handle_a = await backend_a.find_or_create_session("ev:1")
            with patch.object(backend_b, "_get_client", return_value=client_b):
                handle_b = await backend_b.find_or_create_session("ev:1")

        client_a.create.assert_awaited_once()
        client_b.create.assert_not_awaited()
        client_b.get.assert_awaited_once()
        # find_or_create_session returns an opaque ``object`` handle by
        # contract; cast to read the daytona-shaped ``id`` attribute.
        assert cast(Any, handle_a).id == cast(Any, handle_b).id


class TestPostCreateDedup:
    """Two concurrent creates against the same ``session_key`` converge on one
    surviving sandbox via ``_dedupe_after_create`` re-list-and-kill."""

    @pytest.mark.asyncio
    async def test_two_concurrent_creates_converge_on_one_survivor(self) -> None:
        daytona_mod, process_mod = _make_daytona_mocks()
        client = daytona_mod.AsyncDaytona.return_value

        # Two sandboxes for the same session_key — simulates the race
        # window where both replicas listed empty and then both created.
        sb_a = MagicMock(id="sb-aaa", state="STARTED")
        sb_a.process.code_run = AsyncMock(return_value=MagicMock(result="ok", exit_code=0))
        sb_b = MagicMock(id="sb-bbb", state="STARTED")
        sb_b.process.code_run = AsyncMock(return_value=MagicMock(result="ok", exit_code=0))

        # Initial list returns empty (the race premise); the post-create
        # re-list returns both sandboxes so dedup picks the oldest by id.
        empty_resp = MagicMock()
        empty_resp.items = []
        both_resp = MagicMock()
        both_resp.items = [sb_b, sb_a]  # intentionally out of id order
        client.list = AsyncMock(side_effect=[empty_resp, both_resp])
        client.create = AsyncMock(return_value=sb_b)  # we "lost" the race
        client.delete = AsyncMock()

        modules = {
            "daytona_sdk": daytona_mod,
            "daytona_sdk.common": MagicMock(),
            "daytona_sdk.common.process": process_mod,
        }
        with patch.dict(sys.modules, modules), _patch_sandbox_state(daytona_mod):
            from phoenix.server.sandbox.daytona_backend import DaytonaSandboxBackend

            backend = DaytonaSandboxBackend(api_key=_API_KEY)
            handle = await backend.find_or_create_session("ev:race")

        # Survivor is sb_a (lower id); the just-created sb_b is the loser
        # and gets deleted.
        assert cast(Any, handle).id == "sb-aaa"
        client.delete.assert_awaited_once_with(sb_b)

    @pytest.mark.asyncio
    async def test_single_candidate_returns_just_created_without_delete(self) -> None:
        """Single-replica steady state: re-list returns one candidate, dedup
        short-circuits, no delete is issued."""
        daytona_mod, process_mod = _make_daytona_mocks()
        client = daytona_mod.AsyncDaytona.return_value

        sb_new = MagicMock(id="sb-new", state="STARTED")
        sb_new.process.code_run = AsyncMock(return_value=MagicMock(result="ok", exit_code=0))

        empty_resp = MagicMock()
        empty_resp.items = []
        single_resp = MagicMock()
        single_resp.items = [sb_new]
        client.list = AsyncMock(side_effect=[empty_resp, single_resp])
        client.create = AsyncMock(return_value=sb_new)
        client.delete = AsyncMock()

        modules = {
            "daytona_sdk": daytona_mod,
            "daytona_sdk.common": MagicMock(),
            "daytona_sdk.common.process": process_mod,
        }
        with patch.dict(sys.modules, modules), _patch_sandbox_state(daytona_mod):
            from phoenix.server.sandbox.daytona_backend import DaytonaSandboxBackend

            backend = DaytonaSandboxBackend(api_key=_API_KEY)
            handle = await backend.find_or_create_session("ev:solo")

        assert cast(Any, handle).id == "sb-new"
        client.delete.assert_not_awaited()


class TestCloseSession:
    @pytest.mark.asyncio
    async def test_close_session_deletes_all_matches(self) -> None:
        daytona_mod, process_mod = _make_daytona_mocks()
        client = daytona_mod.AsyncDaytona.return_value
        sb_a = MagicMock(id="sb-a")
        sb_b = MagicMock(id="sb-b")
        list_resp = MagicMock()
        list_resp.items = [sb_a, sb_b]
        client.list = AsyncMock(return_value=list_resp)

        modules = {
            "daytona_sdk": daytona_mod,
            "daytona_sdk.common": MagicMock(),
            "daytona_sdk.common.process": process_mod,
        }
        with patch.dict(sys.modules, modules):
            from phoenix.server.sandbox.daytona_backend import DaytonaSandboxBackend

            backend = DaytonaSandboxBackend(api_key=_API_KEY)
            await backend.close_session("ev:1")

        assert client.delete.await_count == 2


def test_provider_session_id_default_is_passthrough() -> None:
    """Daytona does not override ``provider_session_id`` — input == output."""
    daytona_mod, process_mod = _make_daytona_mocks()
    modules = {
        "daytona_sdk": daytona_mod,
        "daytona_sdk.common": MagicMock(),
        "daytona_sdk.common.process": process_mod,
    }
    with patch.dict(sys.modules, modules):
        from phoenix.server.sandbox.daytona_backend import DaytonaSandboxBackend

        backend = DaytonaSandboxBackend(api_key=_API_KEY)
        for key in ("evaluator:42", "inline:abc-123", "x" * 200):
            assert backend.provider_session_id(key) == key


def test_config_fingerprint_is_stable_and_changes_with_runtime_affecting_fields() -> None:
    """Same config → same fingerprint; packages / network_block_all /
    env-var keys / language each fragment the fingerprint; secret plaintext
    rotation under the same env-var keys does NOT."""
    daytona_mod, process_mod = _make_daytona_mocks()
    modules = {
        "daytona_sdk": daytona_mod,
        "daytona_sdk.common": MagicMock(),
        "daytona_sdk.common.process": process_mod,
    }
    with patch.dict(sys.modules, modules):
        from phoenix.server.sandbox.daytona_backend import DaytonaSandboxBackend

        def _build(
            *,
            packages: list[str] | None = None,
            network_block_all: bool = False,
            user_env: dict[str, str] | None = None,
            language: str = "PYTHON",
        ) -> DaytonaSandboxBackend:
            return DaytonaSandboxBackend(
                api_key=_API_KEY,
                packages=packages,
                network_block_all=network_block_all,
                user_env=user_env,
                language=language,
            )

        base = _build(
            packages=["numpy==1.26"],
            network_block_all=False,
            user_env={"FOO": "bar"},
            language="PYTHON",
        )
        assert (
            base.config_fingerprint()
            == _build(
                packages=["numpy==1.26"],
                network_block_all=False,
                user_env={"FOO": "bar"},
                language="PYTHON",
            ).config_fingerprint()
        )
        assert (
            base.config_fingerprint()
            != _build(
                packages=["numpy==1.26", "pandas"],
                network_block_all=False,
                user_env={"FOO": "bar"},
                language="PYTHON",
            ).config_fingerprint()
        )
        assert (
            base.config_fingerprint()
            != _build(
                packages=["numpy==1.26"],
                network_block_all=True,
                user_env={"FOO": "bar"},
                language="PYTHON",
            ).config_fingerprint()
        )
        assert (
            base.config_fingerprint()
            != _build(
                packages=["numpy==1.26"],
                network_block_all=False,
                user_env={"FOO": "bar", "EXTRA": "x"},
                language="PYTHON",
            ).config_fingerprint()
        )
        assert (
            base.config_fingerprint()
            != _build(
                packages=["numpy==1.26"],
                network_block_all=False,
                user_env={"FOO": "bar"},
                language="TYPESCRIPT",
            ).config_fingerprint()
        )
        # Secret-value-only change must NOT change the fingerprint.
        assert (
            base.config_fingerprint()
            == _build(
                packages=["numpy==1.26"],
                network_block_all=False,
                user_env={"FOO": "ROTATED"},
                language="PYTHON",
            ).config_fingerprint()
        )
