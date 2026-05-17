"""Unit tests for ModalSandboxBackend focusing on kwarg forwarding invariants.

Scope: Modal-specific SDK kwarg shapes that parametrized capability-matrix tests
can't express — `env` vs `env_dict`, `block_network`, `Image.pip_install` wiring,
and the explicit-client auth path (no os.environ mutation).
Generic "capability rejected when unsupported" coverage lives in
test_capability_matrix.py.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Mapping
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from starlette.datastructures import Secret


def _make_modal_mock() -> MagicMock:
    """Mock the modal SDK surface used by ModalSandboxBackend.

    Covers both sync and async (``.aio``) call shapes for ``Client.from_credentials``,
    ``App.lookup``, ``Sandbox.create``, and ``Sandbox.from_name`` — Phoenix
    uses the async wrappers everywhere, but the sync forms are mocked for
    completeness. ``modal.exception`` exposes the named exception classes
    that ``find_or_create_session`` / ``close_session`` catch.
    """
    modal = MagicMock()
    modal.App.lookup = MagicMock()
    modal.App.lookup.aio = AsyncMock(return_value=MagicMock())
    modal.Image.debian_slim.return_value = MagicMock()
    modal.Client.from_credentials = MagicMock()
    modal.Client.from_credentials.aio = AsyncMock(return_value=MagicMock())
    modal.Sandbox.create = MagicMock()
    modal.Sandbox.create.aio = AsyncMock(return_value=MagicMock())
    modal.Sandbox.from_name = MagicMock()
    modal.Sandbox.from_name.aio = AsyncMock()

    class _NotFoundError(Exception):
        pass

    class _AlreadyExistsError(Exception):
        pass

    modal.exception = MagicMock()
    modal.exception.NotFoundError = _NotFoundError
    modal.exception.AlreadyExistsError = _AlreadyExistsError
    return modal


_TOKEN_ID_RAW = "ak-test-id"
_TOKEN_SECRET_RAW = "as-test-secret"
_TOKEN_ID = Secret(_TOKEN_ID_RAW)
_TOKEN_SECRET = Secret(_TOKEN_SECRET_RAW)
_CANONICAL_TOKEN_ID = "MODAL_TOKEN_ID"
_CANONICAL_TOKEN_SECRET = "MODAL_TOKEN_SECRET"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "user_env,expect_env_kwarg",
    [
        ({"KEY": "value"}, {"KEY": "value"}),
        ({}, None),
    ],
)
async def test_user_env_reaches_sandbox_create_as_env_kwarg(
    user_env: Mapping[str, str], expect_env_kwarg: dict[str, str] | None
) -> None:
    """user_env must reach Sandbox.create.aio() as `env`, not `env_dict`; absent when empty."""
    modal_mock = _make_modal_mock()
    with patch.dict(sys.modules, {"modal": modal_mock, "modal.exception": modal_mock.exception}):
        from phoenix.server.sandbox.modal_backend import ModalSandboxBackend

        backend = ModalSandboxBackend(
            token_id=_TOKEN_ID, token_secret=_TOKEN_SECRET, user_env=user_env
        )
        await backend._create_sandbox()

    _, kwargs = modal_mock.Sandbox.create.aio.call_args
    assert "env_dict" not in kwargs
    if expect_env_kwarg is None:
        assert "env" not in kwargs
    else:
        assert kwargs.get("env") == expect_env_kwarg


def test_pip_install_invoked_only_when_packages_present() -> None:
    """Non-empty packages → Image.pip_install called with list; empty/None → not called."""
    modal_mock = _make_modal_mock()
    slim_image = modal_mock.Image.debian_slim.return_value
    installed_image = MagicMock()
    slim_image.pip_install.return_value = installed_image

    with patch.dict(sys.modules, {"modal": modal_mock, "modal.exception": modal_mock.exception}):
        from phoenix.server.sandbox.modal_backend import ModalAdapter

        adapter = ModalAdapter()
        with_pkgs: Any = adapter.build_backend(
            {
                "dependencies": {"packages": ["cowsay"]},
                _CANONICAL_TOKEN_ID: _TOKEN_ID_RAW,
                _CANONICAL_TOKEN_SECRET: _TOKEN_SECRET_RAW,
            }
        )
        slim_image.pip_install.assert_called_once_with(["cowsay"])
        assert with_pkgs._image is installed_image

        slim_image.pip_install.reset_mock()
        without_pkgs: Any = adapter.build_backend(
            {
                "dependencies": {"packages": []},
                _CANONICAL_TOKEN_ID: _TOKEN_ID_RAW,
                _CANONICAL_TOKEN_SECRET: _TOKEN_SECRET_RAW,
            }
        )
        slim_image.pip_install.assert_not_called()
        assert without_pkgs._image is slim_image


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "block_network,expect_kwarg",
    [(True, True), (False, None)],
)
async def test_block_network_kwarg_forwarding(
    block_network: bool, expect_kwarg: bool | None
) -> None:
    """block_network=True reaches SDK; block_network=False omits the kwarg entirely."""
    modal_mock = _make_modal_mock()
    with patch.dict(sys.modules, {"modal": modal_mock, "modal.exception": modal_mock.exception}):
        from phoenix.server.sandbox.modal_backend import ModalSandboxBackend

        backend = ModalSandboxBackend(
            token_id=_TOKEN_ID, token_secret=_TOKEN_SECRET, block_network=block_network
        )
        await backend._create_sandbox()

    _, kwargs = modal_mock.Sandbox.create.aio.call_args
    if expect_kwarg is None:
        assert "block_network" not in kwargs
    else:
        assert kwargs.get("block_network") is expect_kwarg


@pytest.mark.parametrize(
    "config,expected",
    [
        (
            {
                "internet_access": {"mode": "deny"},
                _CANONICAL_TOKEN_ID: _TOKEN_ID_RAW,
                _CANONICAL_TOKEN_SECRET: _TOKEN_SECRET_RAW,
            },
            True,
        ),
        (
            {
                "internet_access": {"mode": "allow"},
                _CANONICAL_TOKEN_ID: _TOKEN_ID_RAW,
                _CANONICAL_TOKEN_SECRET: _TOKEN_SECRET_RAW,
            },
            False,
        ),
        (
            {
                _CANONICAL_TOKEN_ID: _TOKEN_ID_RAW,
                _CANONICAL_TOKEN_SECRET: _TOKEN_SECRET_RAW,
            },
            False,
        ),
    ],
)
def test_build_backend_sets_block_network_from_internet_access(
    config: dict[str, Any], expected: bool
) -> None:
    """ModalAdapter.build_backend translates internet_access.mode → backend._block_network."""
    modal_mock = _make_modal_mock()
    with patch.dict(sys.modules, {"modal": modal_mock, "modal.exception": modal_mock.exception}):
        from phoenix.server.sandbox.modal_backend import ModalAdapter

        adapter = ModalAdapter()
        backend: Any = adapter.build_backend(config)
    assert backend._block_network is expected


def test_build_backend_requires_both_tokens() -> None:
    """Missing either token must raise ValueError at adapter.build_backend time."""
    modal_mock = _make_modal_mock()
    with patch.dict(sys.modules, {"modal": modal_mock, "modal.exception": modal_mock.exception}):
        from phoenix.server.sandbox.modal_backend import ModalAdapter

        adapter = ModalAdapter()
        with pytest.raises(ValueError, match=_CANONICAL_TOKEN_ID):
            adapter.build_backend({_CANONICAL_TOKEN_SECRET: _TOKEN_SECRET_RAW})
        with pytest.raises(ValueError, match=_CANONICAL_TOKEN_ID):
            adapter.build_backend({_CANONICAL_TOKEN_ID: _TOKEN_ID_RAW})


@pytest.mark.asyncio
async def test_credentials_passed_to_sdk_via_explicit_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The backend must construct ``modal.Client.from_credentials.aio(token_id, token_secret)``
    and thread that client into App.lookup + Sandbox.create as the ``client=`` kwarg —
    rather than mutating os.environ.
    """
    monkeypatch.delenv("MODAL_TOKEN_ID", raising=False)
    monkeypatch.delenv("MODAL_TOKEN_SECRET", raising=False)

    modal_mock = _make_modal_mock()
    sentinel_client = MagicMock(name="modal-client")
    modal_mock.Client.from_credentials.aio = AsyncMock(return_value=sentinel_client)
    sentinel_app = MagicMock(name="modal-app")
    modal_mock.App.lookup.aio = AsyncMock(return_value=sentinel_app)

    with patch.dict(sys.modules, {"modal": modal_mock, "modal.exception": modal_mock.exception}):
        from phoenix.server.sandbox.modal_backend import ModalSandboxBackend

        backend = ModalSandboxBackend(token_id=_TOKEN_ID, token_secret=_TOKEN_SECRET)
        await backend._create_sandbox()

    modal_mock.Client.from_credentials.aio.assert_awaited_once_with(
        _TOKEN_ID_RAW, _TOKEN_SECRET_RAW
    )
    _, lookup_kwargs = modal_mock.App.lookup.aio.call_args
    assert lookup_kwargs.get("client") is sentinel_client
    _, create_kwargs = modal_mock.Sandbox.create.aio.call_args
    assert create_kwargs.get("client") is sentinel_client
    assert create_kwargs.get("app") is sentinel_app
    assert "MODAL_TOKEN_ID" not in os.environ
    assert "MODAL_TOKEN_SECRET" not in os.environ


@pytest.mark.asyncio
async def test_client_construction_is_memoized_across_sandbox_creates() -> None:
    """Two _create_sandbox() calls on the same backend must reuse the same client + app,
    not re-construct them per call."""
    modal_mock = _make_modal_mock()
    with patch.dict(sys.modules, {"modal": modal_mock, "modal.exception": modal_mock.exception}):
        from phoenix.server.sandbox.modal_backend import ModalSandboxBackend

        backend = ModalSandboxBackend(token_id=_TOKEN_ID, token_secret=_TOKEN_SECRET)
        await backend._create_sandbox()
        await backend._create_sandbox()

    assert modal_mock.Client.from_credentials.aio.await_count == 1
    assert modal_mock.App.lookup.aio.await_count == 1
    assert modal_mock.Sandbox.create.aio.await_count == 2


# ---------------------------------------------------------------------------
# provider_session_id — sha256 prefix override (Modal-specific)
# ---------------------------------------------------------------------------


def test_provider_session_id_is_sha256_prefix_alphanumeric_le_32_chars() -> None:
    """Modal names are restricted to alphanumeric + ``-`` and limited to 64
    chars. ``provider_session_id`` produces a 32-char hex prefix (alphanumeric
    by definition, well under the limit) and is deterministic across calls.
    """
    modal_mock = _make_modal_mock()
    with patch.dict(sys.modules, {"modal": modal_mock, "modal.exception": modal_mock.exception}):
        from phoenix.server.sandbox.modal_backend import ModalSandboxBackend

        backend = ModalSandboxBackend(token_id=_TOKEN_ID, token_secret=_TOKEN_SECRET)
        sample = [
            "evaluator:42",
            "evaluator:43",
            "inline:abc-123",
            "inline:def-456",
            "very/long/key/with-special-chars and spaces!!!",
        ]
        ids = [backend.provider_session_id(k) for k in sample]

    # Determinism: same input → same output across calls.
    for key, derived in zip(sample, ids):
        assert backend.provider_session_id(key) == derived
    # Length / charset invariants.
    for derived in ids:
        assert len(derived) <= 32
        assert derived.isalnum()
    # Distinct inputs → distinct outputs across the sample set.
    assert len(set(ids)) == len(ids)


# ---------------------------------------------------------------------------
# find_or_create_session — Modal name-uniqueness binding + TTL kwargs
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_find_or_create_session_reuses_existing_named_sandbox() -> None:
    """from_name returns an alive sandbox → reuse; create.aio NOT awaited."""
    modal_mock = _make_modal_mock()
    existing = MagicMock()
    existing.poll = MagicMock()
    existing.poll.aio = AsyncMock(return_value=None)  # None == still running
    modal_mock.Sandbox.from_name.aio = AsyncMock(return_value=existing)

    with patch.dict(sys.modules, {"modal": modal_mock, "modal.exception": modal_mock.exception}):
        from phoenix.server.sandbox.modal_backend import ModalSandboxBackend

        backend = ModalSandboxBackend(token_id=_TOKEN_ID, token_secret=_TOKEN_SECRET)
        handle = await backend.find_or_create_session("evaluator:42")

    modal_mock.Sandbox.from_name.aio.assert_awaited()
    modal_mock.Sandbox.create.aio.assert_not_awaited()
    assert handle is existing


@pytest.mark.asyncio
async def test_find_or_create_session_creates_when_from_name_raises_not_found() -> None:
    """from_name → NotFoundError → fall through to create with name kwarg."""
    modal_mock = _make_modal_mock()
    modal_mock.Sandbox.from_name.aio = AsyncMock(side_effect=modal_mock.exception.NotFoundError())

    with patch.dict(sys.modules, {"modal": modal_mock, "modal.exception": modal_mock.exception}):
        from phoenix.server.sandbox.modal_backend import ModalSandboxBackend

        backend = ModalSandboxBackend(token_id=_TOKEN_ID, token_secret=_TOKEN_SECRET)
        handle = await backend.find_or_create_session("evaluator:42")

    modal_mock.Sandbox.create.aio.assert_awaited_once()
    create_kwargs = modal_mock.Sandbox.create.aio.call_args.kwargs
    # TTL kwargs hard-coded on every create.
    assert create_kwargs["timeout"] == 600
    assert create_kwargs["idle_timeout"] == 300
    # Name derives from provider_session_id, NOT the raw session_key.
    expected_name = backend.provider_session_id("evaluator:42")
    assert create_kwargs["name"] == expected_name
    assert handle is modal_mock.Sandbox.create.aio.return_value


@pytest.mark.asyncio
async def test_find_or_create_session_attaches_on_already_exists_race() -> None:
    """Concurrent winner from another replica → AlreadyExistsError on create
    → attach via a second from_name."""
    modal_mock = _make_modal_mock()
    # First from_name: not found (miss). Second from_name (after the race):
    # returns the concurrent winner.
    winner = MagicMock()
    winner.poll = MagicMock()
    winner.poll.aio = AsyncMock(return_value=None)
    modal_mock.Sandbox.from_name.aio = AsyncMock(
        side_effect=[
            modal_mock.exception.NotFoundError(),
            winner,
        ]
    )
    modal_mock.Sandbox.create.aio = AsyncMock(side_effect=modal_mock.exception.AlreadyExistsError())

    with patch.dict(sys.modules, {"modal": modal_mock, "modal.exception": modal_mock.exception}):
        from phoenix.server.sandbox.modal_backend import ModalSandboxBackend

        backend = ModalSandboxBackend(token_id=_TOKEN_ID, token_secret=_TOKEN_SECRET)
        handle = await backend.find_or_create_session("evaluator:42")

    assert modal_mock.Sandbox.from_name.aio.await_count == 2
    assert handle is winner


@pytest.mark.asyncio
async def test_find_or_create_session_is_idempotent_within_one_backend() -> None:
    """Two consecutive find_or_create_session(key) calls return the same handle.

    The Modal-side binding lives in the per-app name namespace; the second
    call resolves through from_name and never enters the create branch.
    """
    modal_mock = _make_modal_mock()
    existing = MagicMock()
    existing.poll = MagicMock()
    existing.poll.aio = AsyncMock(return_value=None)
    # First call: from_name MISS, then create succeeds. Second call: from_name HIT.
    modal_mock.Sandbox.from_name.aio = AsyncMock(
        side_effect=[
            modal_mock.exception.NotFoundError(),
            existing,
        ]
    )
    created = MagicMock()
    modal_mock.Sandbox.create.aio = AsyncMock(return_value=created)

    with patch.dict(sys.modules, {"modal": modal_mock, "modal.exception": modal_mock.exception}):
        from phoenix.server.sandbox.modal_backend import ModalSandboxBackend

        backend = ModalSandboxBackend(token_id=_TOKEN_ID, token_secret=_TOKEN_SECRET)
        first = await backend.find_or_create_session("evaluator:42")
        second = await backend.find_or_create_session("evaluator:42")

    assert modal_mock.Sandbox.create.aio.await_count == 1
    assert first is created
    assert second is existing


@pytest.mark.asyncio
async def test_close_session_terminates_named_sandbox() -> None:
    """``close_session`` looks up by provider_session_id and terminates it."""
    modal_mock = _make_modal_mock()
    sandbox = MagicMock()
    sandbox.terminate = MagicMock()
    sandbox.terminate.aio = AsyncMock()
    modal_mock.Sandbox.from_name.aio = AsyncMock(return_value=sandbox)

    with patch.dict(sys.modules, {"modal": modal_mock, "modal.exception": modal_mock.exception}):
        from phoenix.server.sandbox.modal_backend import ModalSandboxBackend

        backend = ModalSandboxBackend(token_id=_TOKEN_ID, token_secret=_TOKEN_SECRET)
        await backend.close_session("evaluator:42")

    sandbox.terminate.aio.assert_awaited_once()


@pytest.mark.asyncio
async def test_close_session_is_idempotent_on_unknown_name() -> None:
    modal_mock = _make_modal_mock()
    modal_mock.Sandbox.from_name.aio = AsyncMock(side_effect=modal_mock.exception.NotFoundError())
    with patch.dict(sys.modules, {"modal": modal_mock, "modal.exception": modal_mock.exception}):
        from phoenix.server.sandbox.modal_backend import ModalSandboxBackend

        backend = ModalSandboxBackend(token_id=_TOKEN_ID, token_secret=_TOKEN_SECRET)
        # Must NOT raise.
        await backend.close_session("evaluator:never-bound")


def test_config_fingerprint_is_stable_and_changes_with_runtime_affecting_fields() -> None:
    """``config_fingerprint`` is the manager-side disambiguator for
    mid-iteration config switches.

    Contract:
      - Structurally-equal configs produce the same fingerprint.
      - Changing the package list, internet-access mode, or env-var key set
        changes the fingerprint.
      - Rotating a secret's plaintext value WITHOUT changing the env-var
        key set leaves the fingerprint unchanged — masking handles
        plaintext rotation; the fingerprint must survive credential
        rotation so sessions don't churn whenever a secret rotates.
    """
    modal_mock = _make_modal_mock()
    with patch.dict(sys.modules, {"modal": modal_mock, "modal.exception": modal_mock.exception}):
        from phoenix.server.sandbox.modal_backend import ModalSandboxBackend

        def _build(
            *,
            packages: list[str] | None = None,
            block_network: bool = False,
            user_env: Mapping[str, str] | None = None,
        ) -> ModalSandboxBackend:
            return ModalSandboxBackend(
                token_id=_TOKEN_ID,
                token_secret=_TOKEN_SECRET,
                packages=packages,
                block_network=block_network,
                user_env=user_env,
            )

        base = _build(
            packages=["numpy==1.26.0"],
            block_network=False,
            user_env={"FOO": "bar"},
        )
        equal_shape = _build(
            packages=["numpy==1.26.0"],
            block_network=False,
            user_env={"FOO": "bar"},
        )
        assert base.config_fingerprint() == equal_shape.config_fingerprint()

        # Packages list change.
        assert (
            base.config_fingerprint()
            != _build(
                packages=["numpy==1.26.0", "requests"],
                block_network=False,
                user_env={"FOO": "bar"},
            ).config_fingerprint()
        )

        # Internet-access mode change.
        assert (
            base.config_fingerprint()
            != _build(
                packages=["numpy==1.26.0"],
                block_network=True,
                user_env={"FOO": "bar"},
            ).config_fingerprint()
        )

        # env_vars KEYS change.
        assert (
            base.config_fingerprint()
            != _build(
                packages=["numpy==1.26.0"],
                block_network=False,
                user_env={"FOO": "bar", "EXTRA": "x"},
            ).config_fingerprint()
        )

        # Secret-plaintext-only change (same key) — fingerprint must NOT change.
        rotated = _build(
            packages=["numpy==1.26.0"],
            block_network=False,
            user_env={"FOO": "ROTATED_VALUE"},
        )
        assert rotated.config_fingerprint() == base.config_fingerprint()
