"""Unit tests for ``SandboxSessionManager`` under the post-refactor shape.

``_tracked`` and ``_key_locks`` are keyed on the opaque ``session_key``.
``acquire(backend, session_key)`` invokes
``backend.find_or_create_session(session_key)``, stores the returned opaque
handle on the tracked entry, and routes ``session.execute`` through
``backend.execute_in_session(handle, code, timeout=...)``. Eviction is by
opaque key (``evict_for_session_key``) or by adapter family
(``evict_for_provider_family``); capacity caps are enforced per family via
``max_sessions_per_provider``.
"""

from __future__ import annotations

import asyncio
from asyncio import Event
from typing import AsyncIterator, Optional
from unittest.mock import patch

import pytest

from phoenix.server.sandbox.session_manager import (
    SandboxSessionManager,
    SessionInvalidated,
    SessionLimitExceeded,
)
from phoenix.server.sandbox.types import (
    BaseNoSessionBackend,
    ExecutionResult,
    SandboxBackend,
)


def _ck(session_key: str, fingerprint: str = "fp") -> str:
    """Compose the manager-internal composite key the way ``acquire`` does.

    Mirrors ``SandboxSessionManager._composite_key`` so tests can assert on
    the ``_tracked`` / ``_key_locks`` entries and the values
    ``backend.find_or_create_session`` / ``backend.close_session`` receive
    without re-deriving the format. The default ``fingerprint`` matches
    ``_FakeBackend._fingerprint`` so tests using the default fake don't have
    to thread the fingerprint through.
    """
    return f"{session_key}#{fingerprint}"


class _FakeHandle:
    """Opaque per-key remote handle the fake backend hands back."""

    def __init__(self, session_key: str) -> None:
        self.session_key = session_key


class _FakeBackend(SandboxBackend):
    """In-memory session-capable fake backend.

    Records ``find_or_create_session`` / ``close_session`` calls and the
    handles passed back through ``execute_in_session`` so tests can assert
    on lifecycle invariants without standing up a real provider SDK.
    """

    # Default for tests that don't override; matches the adapter family
    # the manager reads via ``getattr(backend, "family", type(...).__name__)``.
    family: str = "FAKE"

    # Constant fingerprint by default so the manager's composite key is
    # deterministic in tests that don't exercise config-switch behavior.
    # Tests covering config-switch override this on the instance.
    _fingerprint: str = "fp"

    def __init__(self) -> None:
        self.find_calls: list[str] = []
        self.close_calls: list[str] = []
        self.execute_in_session_calls: list[tuple[str, str, Optional[int]]] = []
        self._sessions: dict[str, _FakeHandle] = {}

    def config_fingerprint(self) -> str:
        return self._fingerprint

    async def find_or_create_session(self, session_key: str) -> object:
        self.find_calls.append(session_key)
        handle = self._sessions.get(session_key)
        if handle is None:
            handle = _FakeHandle(session_key)
            self._sessions[session_key] = handle
        return handle

    async def execute_in_session(
        self,
        handle: object,
        code: str,
        timeout: Optional[int] = None,
    ) -> ExecutionResult:
        assert isinstance(handle, _FakeHandle)
        self.execute_in_session_calls.append((handle.session_key, code, timeout))
        return ExecutionResult(stdout=code, stderr="")

    async def close_session(self, session_key: str) -> None:
        # Pop-before-await: synchronously remove the binding before any
        # subsequent await would run, mirroring the real adapter contract.
        self._sessions.pop(session_key, None)
        self.close_calls.append(session_key)

    async def execute(
        self,
        code: str,
        session_key: str,
        timeout: Optional[int] = None,
    ) -> ExecutionResult:
        return ExecutionResult(stdout=code, stderr="")

    async def close(self) -> None:
        self._sessions.clear()


class _StatelessFakeBackend(BaseNoSessionBackend):
    """Stateless fake backend — inherits ``find_or_create_session`` /
    ``execute_in_session`` / ``close_session`` from ``BaseNoSessionBackend``.
    """

    def __init__(self) -> None:
        self.execute_calls: list[tuple[str, str, Optional[int]]] = []

    async def execute(
        self,
        code: str,
        session_key: str,
        timeout: Optional[int] = None,
    ) -> ExecutionResult:
        self.execute_calls.append((code, session_key, timeout))
        return ExecutionResult(stdout=code, stderr="")

    async def close(self) -> None:
        pass


@pytest.fixture
async def sweep_trigger() -> AsyncIterator[Event]:
    """Patch ``session_manager.sleep`` so tests can drive the sweeper loop."""
    event = Event()

    async def wait_for_event(seconds: float) -> None:
        await event.wait()
        event.clear()

    with patch("phoenix.server.sandbox.session_manager.sleep", wait_for_event):
        yield event


@pytest.mark.asyncio
async def test_acquire_invokes_find_or_create_session_once_per_key() -> None:
    """First acquire calls ``find_or_create_session``; second acquire of the
    same key reuses — ``find_or_create_session`` called exactly once."""
    manager = SandboxSessionManager()
    backend = _FakeBackend()

    async with manager.acquire(backend, "k1") as session:
        result = await session.execute("print(1)")
    assert result.stdout == "print(1)"

    async with manager.acquire(backend, "k1") as session:
        await session.execute("print(2)")

    assert backend.find_calls == [_ck("k1")], (
        f"find_or_create_session must run once for k1; got {backend.find_calls!r}"
    )
    assert len(backend.execute_in_session_calls) == 2
    # Both executes routed through the same handle bound to k1's composite.
    assert all(call[0] == _ck("k1") for call in backend.execute_in_session_calls)


@pytest.mark.asyncio
async def test_idle_ttl_evicts_via_close_session(sweep_trigger: Event) -> None:
    manager = SandboxSessionManager(
        idle_ttl_seconds=0.0,
        sweep_interval_seconds=0.0,
    )
    backend = _FakeBackend()

    async with manager.acquire(backend, "k1"):
        pass

    await manager.start()
    try:
        await asyncio.sleep(0)
        await asyncio.sleep(0.01)
        sweep_trigger.set()
        for _ in range(50):
            await asyncio.sleep(0.01)
            if backend.close_calls:
                break
    finally:
        await manager.stop()

    assert backend.close_calls == [_ck("k1")], (
        f"sweeper must close idle session; got close_calls={backend.close_calls}"
    )


@pytest.mark.asyncio
async def test_idle_ttl_does_not_evict_when_inflight_nonzero(
    sweep_trigger: Event,
) -> None:
    manager = SandboxSessionManager(
        idle_ttl_seconds=0.0,
        sweep_interval_seconds=0.0,
    )
    backend = _FakeBackend()

    await manager.start()
    try:
        async with manager.acquire(backend, "k1"):
            sweep_trigger.set()
            for _ in range(20):
                await asyncio.sleep(0.01)
        assert backend.close_calls == [], (
            f"sweeper must NOT close in-flight session; got {backend.close_calls}"
        )
    finally:
        await manager.stop()


@pytest.mark.asyncio
async def test_acquire_raises_session_limit_exceeded() -> None:
    """At ``max_sessions_per_provider``, a new acquire for the same family
    raises ``SessionLimitExceeded``; existing keys remain usable."""
    manager = SandboxSessionManager(max_sessions_per_provider=2)
    backend = _FakeBackend()

    async with manager.acquire(backend, "k1"):
        async with manager.acquire(backend, "k2"):
            with pytest.raises(SessionLimitExceeded):
                async with manager.acquire(backend, "k3"):
                    pytest.fail("acquire should have raised before entering body")

            async with manager.acquire(backend, "k1") as session:
                await session.execute("print('still works')")

    assert sorted(backend.find_calls) == [_ck("k1"), _ck("k2")]


@pytest.mark.asyncio
async def test_capacity_is_per_provider_family() -> None:
    """``max_sessions_per_provider`` counts entries per adapter family, not
    per wrapper instance."""

    class _BackendA(_FakeBackend):
        family = "FAMILY_A"

    class _BackendB(_FakeBackend):
        family = "FAMILY_B"

    manager = SandboxSessionManager(max_sessions_per_provider=1)
    backend_a1 = _BackendA()
    backend_a2 = _BackendA()  # same family, different wrapper instance
    backend_b = _BackendB()

    async with manager.acquire(backend_a1, "a1"):
        # Same family ceiling reached — even a fresh wrapper for the same
        # family must hit the cap.
        with pytest.raises(SessionLimitExceeded):
            async with manager.acquire(backend_a2, "a2"):
                pytest.fail("FAMILY_A should be at capacity")
        # Different family has its own budget.
        async with manager.acquire(backend_b, "b1") as session:
            await session.execute("ok")

    assert backend_a1.find_calls == [_ck("a1")]
    assert backend_a2.find_calls == []
    assert backend_b.find_calls == [_ck("b1")]


@pytest.mark.asyncio
async def test_concurrent_acquires_respect_capacity_under_race() -> None:
    """Two acquires for different keys on the same family can't both pass
    the capacity check; the second must see the first's reservation."""
    a_in_find = asyncio.Event()
    release_a = asyncio.Event()

    class _GatedBackend(SandboxBackend):
        family = "GATED"

        def __init__(self) -> None:
            self.find_calls: list[str] = []

        async def find_or_create_session(self, session_key: str) -> object:
            self.find_calls.append(session_key)
            if session_key.startswith("a#"):
                a_in_find.set()
                await release_a.wait()
            return _FakeHandle(session_key)

        async def execute_in_session(
            self,
            handle: object,
            code: str,
            timeout: Optional[int] = None,
        ) -> ExecutionResult:
            return ExecutionResult(stdout="", stderr="")

        async def close_session(self, session_key: str) -> None:
            pass

        async def execute(
            self,
            code: str,
            session_key: str,
            timeout: Optional[int] = None,
        ) -> ExecutionResult:
            return ExecutionResult(stdout="", stderr="")

        async def close(self) -> None:
            pass

        def config_fingerprint(self) -> str:
            return "gated"

    backend = _GatedBackend()
    manager = SandboxSessionManager(max_sessions_per_provider=1)

    async def acquire_key(key: str) -> str:
        try:
            async with manager.acquire(backend, key):
                return "acquired"
        except SessionLimitExceeded:
            return "rejected"

    task_a = asyncio.create_task(acquire_key("a"))
    await a_in_find.wait()

    task_b = asyncio.create_task(acquire_key("b"))
    result_b = await task_b

    release_a.set()
    result_a = await task_a

    assert result_a == "acquired"
    assert result_b == "rejected"
    # find_or_create_session must never run for "b".
    assert backend.find_calls == [_ck("a", "gated")]


@pytest.mark.asyncio
async def test_find_or_create_session_failure_releases_reservation() -> None:
    """If ``find_or_create_session`` raises, the reserved slot is popped so
    the next acquire on the same key starts cleanly."""

    class _FailingBackend(SandboxBackend):
        family = "FAILING"

        def __init__(self) -> None:
            self.find_calls: list[str] = []
            self.fail_next = True

        def config_fingerprint(self) -> str:
            return "fail"

        async def find_or_create_session(self, session_key: str) -> object:
            self.find_calls.append(session_key)
            if self.fail_next:
                self.fail_next = False
                raise RuntimeError("simulated find_or_create failure")
            return _FakeHandle(session_key)

        async def execute_in_session(
            self,
            handle: object,
            code: str,
            timeout: Optional[int] = None,
        ) -> ExecutionResult:
            return ExecutionResult(stdout=code, stderr="")

        async def close_session(self, session_key: str) -> None:
            pass

        async def execute(
            self,
            code: str,
            session_key: str,
            timeout: Optional[int] = None,
        ) -> ExecutionResult:
            return ExecutionResult(stdout=code, stderr="")

        async def close(self) -> None:
            pass

    backend = _FailingBackend()
    manager = SandboxSessionManager(max_sessions_per_provider=1)

    with pytest.raises(RuntimeError, match="simulated find_or_create failure"):
        async with manager.acquire(backend, "k1"):
            pytest.fail("body should not run when find_or_create_session fails")

    # Reservation rolled back: both dicts cleared.
    assert manager._tracked == {}
    assert manager._key_locks == {}

    async with manager.acquire(backend, "k1") as session:
        result = await session.execute("print('ok')")
    assert result.stdout == "print('ok')"
    assert backend.find_calls == [_ck("k1", "fail"), _ck("k1", "fail")]


@pytest.mark.asyncio
async def test_key_locks_no_unbounded_growth_across_acquire_evict_cycles() -> None:
    """``_key_locks`` entries must pop alongside ``_tracked`` on every
    eviction site."""
    manager = SandboxSessionManager(max_sessions_per_provider=4)
    backend = _FakeBackend()

    for i in range(100):
        key = f"k{i}"
        async with manager.acquire(backend, key):
            pass
        await manager.evict_for_session_key(key)

    assert len(manager._key_locks) <= 1, (
        f"_key_locks should not grow unboundedly; got {len(manager._key_locks)}"
    )
    assert len(manager._tracked) <= 1, (
        f"_tracked should not grow unboundedly; got {len(manager._tracked)}"
    )


@pytest.mark.asyncio
async def test_release_after_lock_pop_still_decrements_inflight() -> None:
    """A coroutine that captured the per-key lock before eviction popped
    ``_key_locks`` must still decrement ``in_flight_count`` correctly on
    release."""
    manager = SandboxSessionManager(max_sessions_per_provider=2)
    backend = _FakeBackend()

    async with manager.acquire(backend, "k1"):
        pass

    captured_lock = manager._key_locks[_ck("k1")]

    await manager.evict_for_session_key("k1")
    assert _ck("k1") not in manager._key_locks
    assert _ck("k1") not in manager._tracked
    assert backend.close_calls == [_ck("k1")]

    async with manager.acquire(backend, "k1"):
        assert manager._key_locks[_ck("k1")] is not captured_lock
        tracked = manager._tracked[_ck("k1")]
        assert tracked.in_flight_count == 1
    assert manager._tracked[_ck("k1")].in_flight_count == 0


@pytest.mark.asyncio
async def test_same_key_parallel_acquires_under_popped_lock_waits_for_single_find() -> None:
    """Two acquires for the same key racing across a popped ``_key_locks``
    entry must result in exactly one ``find_or_create_session`` call AND the
    follower's body must not enter until the leader's find resolves."""

    start_gate = asyncio.Event()
    leader_in_find = asyncio.Event()
    follower_entered_body = asyncio.Event()

    class _GatedBackend(SandboxBackend):
        family = "GATED_PARALLEL"

        def __init__(self) -> None:
            self.find_calls: list[str] = []
            self.close_calls: list[str] = []

        def config_fingerprint(self) -> str:
            return "gp"

        async def find_or_create_session(self, session_key: str) -> object:
            self.find_calls.append(session_key)
            if session_key.startswith("k1#") and len(self.find_calls) >= 2:
                leader_in_find.set()
                await start_gate.wait()
            return _FakeHandle(session_key)

        async def execute_in_session(
            self,
            handle: object,
            code: str,
            timeout: Optional[int] = None,
        ) -> ExecutionResult:
            return ExecutionResult(stdout=code, stderr="")

        async def close_session(self, session_key: str) -> None:
            self.close_calls.append(session_key)

        async def execute(
            self,
            code: str,
            session_key: str,
            timeout: Optional[int] = None,
        ) -> ExecutionResult:
            return ExecutionResult(stdout=code, stderr="")

        async def close(self) -> None:
            pass

    backend = _GatedBackend()
    manager = SandboxSessionManager(max_sessions_per_provider=4)

    async with manager.acquire(backend, "k1"):
        pass
    await manager.evict_for_session_key("k1")
    assert _ck("k1", "gp") not in manager._key_locks
    assert _ck("k1", "gp") not in manager._tracked

    async def y_acquire() -> str:
        async with manager.acquire(backend, "k1"):
            return "y-acquired"

    task_y = asyncio.create_task(y_acquire())
    await leader_in_find.wait()

    async def z_acquire() -> str:
        async with manager.acquire(backend, "k1"):
            follower_entered_body.set()
            return "z-acquired"

    task_z = asyncio.create_task(z_acquire())

    for _ in range(20):
        await asyncio.sleep(0.01)
    assert not follower_entered_body.is_set(), (
        "follower must NOT enter body before leader's find_or_create_session resolves"
    )

    start_gate.set()
    result_y = await task_y
    result_z = await task_z
    assert result_y == "y-acquired"
    assert result_z == "z-acquired"

    assert backend.find_calls.count(_ck("k1", "gp")) == 2, (
        f"find_or_create_session must run once for X and once for Y, never for Z; "
        f"got find_calls={backend.find_calls}"
    )


@pytest.mark.asyncio
async def test_no_op_for_stateless_backend() -> None:
    """``BaseNoSessionBackend`` short-circuits — acquire bypasses lock/state
    tracking; ``execute_in_session`` routes to ``execute``."""
    manager = SandboxSessionManager()
    backend = _StatelessFakeBackend()

    async with manager.acquire(backend, "k1") as session:
        result = await session.execute("print('hi')")

    assert result.stdout == "print('hi')"
    # BaseNoSessionBackend.execute_in_session delegates to execute with key="".
    assert backend.execute_calls == [("print('hi')", "", None)]
    assert manager._tracked == {}
    assert manager._key_locks == {}


@pytest.mark.asyncio
async def test_daemontask_stop_cancels_sweeper_within_grace(
    sweep_trigger: Event,
) -> None:
    manager = SandboxSessionManager(
        idle_ttl_seconds=60.0,
        sweep_interval_seconds=0.0,
    )
    backend = _FakeBackend()

    await manager.start()
    sweep_trigger.set()
    await asyncio.sleep(0.01)

    async with manager.acquire(backend, "k1"):
        pass

    await manager.stop()
    assert manager._tasks == []


@pytest.mark.asyncio
async def test_acquire_raises_session_invalidated_when_marked_for_eviction() -> None:
    leader_in_body = asyncio.Event()
    release_leader = asyncio.Event()

    manager = SandboxSessionManager(
        max_sessions_per_provider=4,
        eviction_grace_seconds=0.0,
    )
    backend = _FakeBackend()

    async def long_acquire() -> None:
        async with manager.acquire(backend, "k1"):
            leader_in_body.set()
            await release_leader.wait()

    leader = asyncio.create_task(long_acquire())
    await leader_in_body.wait()

    await manager.evict_for_session_key("k1")

    with pytest.raises(SessionInvalidated):
        async with manager.acquire(backend, "k1"):
            pytest.fail("acquire should refuse admit onto a marked entry")

    release_leader.set()
    await leader

    assert backend.find_calls == [_ck("k1")]


@pytest.mark.asyncio
async def test_manager_stop_drains_tracked_via_close_session() -> None:
    """``stop`` must drive ``close_session`` for every tracked key before
    cancelling the sweeper."""
    manager = SandboxSessionManager(
        idle_ttl_seconds=60.0,
        sweep_interval_seconds=60.0,
        eviction_grace_seconds=0.5,
    )
    backend_a = _FakeBackend()
    backend_b = _FakeBackend()

    await manager.start()
    try:
        async with manager.acquire(backend_a, "ka1"):
            pass
        async with manager.acquire(backend_a, "ka2"):
            pass
        async with manager.acquire(backend_b, "kb1"):
            pass
    finally:
        await manager.stop()

    assert sorted(backend_a.close_calls) == [_ck("ka1"), _ck("ka2")]
    assert backend_b.close_calls == [_ck("kb1")]
    assert manager._tasks == []


@pytest.mark.asyncio
async def test_manager_stop_awaits_pending_tasks_before_cancelling_sweeper() -> None:
    """``schedule_eviction`` tasks must be awaited (not cancelled) by
    ``stop`` so the underlying ``close_session`` completes."""
    close_entered = asyncio.Event()
    release_close = asyncio.Event()
    close_completed = asyncio.Event()

    class _SlowCloseBackend(SandboxBackend):
        family = "SLOW_CLOSE"

        def __init__(self) -> None:
            self.find_calls: list[str] = []
            self.close_calls: list[str] = []

        def config_fingerprint(self) -> str:
            return "sc"

        async def find_or_create_session(self, session_key: str) -> object:
            self.find_calls.append(session_key)
            return _FakeHandle(session_key)

        async def execute_in_session(
            self,
            handle: object,
            code: str,
            timeout: Optional[int] = None,
        ) -> ExecutionResult:
            return ExecutionResult(stdout="", stderr="")

        async def close_session(self, session_key: str) -> None:
            close_entered.set()
            await release_close.wait()
            self.close_calls.append(session_key)
            close_completed.set()

        async def execute(
            self,
            code: str,
            session_key: str,
            timeout: Optional[int] = None,
        ) -> ExecutionResult:
            return ExecutionResult(stdout="", stderr="")

        async def close(self) -> None:
            pass

    backend = _SlowCloseBackend()
    manager = SandboxSessionManager(
        idle_ttl_seconds=60.0,
        sweep_interval_seconds=60.0,
        eviction_grace_seconds=1.0,
    )

    await manager.start()
    async with manager.acquire(backend, "k1"):
        pass

    manager.schedule_eviction("k1")
    await close_entered.wait()

    async def run_stop() -> None:
        await manager.stop()

    stop_task = asyncio.create_task(run_stop())
    await asyncio.sleep(0.05)
    assert not close_completed.is_set()

    release_close.set()
    await stop_task

    assert backend.close_calls == [_ck("k1", "sc")]


@pytest.mark.asyncio
async def test_manager_stop_does_not_cancel_slow_pending_eviction_past_grace() -> None:
    """A pending ``schedule_eviction`` task whose ``close_session`` runs
    longer than ``eviction_grace_seconds`` must NOT be cancelled by
    ``stop()`` — it must run to completion after ``stop()`` returns."""
    grace = 0.05
    close_entered = asyncio.Event()
    cancelled_during_close = False

    class _SlowCloseBackend(SandboxBackend):
        family = "SLOW_CLOSE_PAST_GRACE"

        def __init__(self) -> None:
            self.find_calls: list[str] = []
            self.close_calls: list[str] = []

        def config_fingerprint(self) -> str:
            return "scpg"

        async def find_or_create_session(self, session_key: str) -> object:
            self.find_calls.append(session_key)
            return _FakeHandle(session_key)

        async def execute_in_session(
            self,
            handle: object,
            code: str,
            timeout: Optional[int] = None,
        ) -> ExecutionResult:
            return ExecutionResult(stdout="", stderr="")

        async def close_session(self, session_key: str) -> None:
            nonlocal cancelled_during_close
            close_entered.set()
            try:
                await asyncio.sleep(grace * 2)
            except asyncio.CancelledError:
                cancelled_during_close = True
                raise
            self.close_calls.append(session_key)

        async def execute(
            self,
            code: str,
            session_key: str,
            timeout: Optional[int] = None,
        ) -> ExecutionResult:
            return ExecutionResult(stdout="", stderr="")

        async def close(self) -> None:
            pass

    backend = _SlowCloseBackend()
    manager = SandboxSessionManager(
        idle_ttl_seconds=60.0,
        sweep_interval_seconds=60.0,
        eviction_grace_seconds=grace,
    )

    await manager.start()
    async with manager.acquire(backend, "k1"):
        pass

    manager.schedule_eviction("k1")
    await close_entered.wait()
    assert manager._pending_tasks is not None
    pending_task = next(iter(manager._pending_tasks))

    await manager.stop()

    # stop() must have returned without cancelling the slow close_session.
    assert not pending_task.done(), (
        "slow pending eviction should still be running after stop() returns"
    )
    assert not cancelled_during_close
    assert backend.close_calls == []

    # The pending task is allowed to finish on its own; provider TTLs would
    # otherwise reap the orphan in production.
    await asyncio.wait_for(pending_task, timeout=grace * 10)
    assert backend.close_calls == [_ck("k1", "scpg")]
    assert not cancelled_during_close


@pytest.mark.asyncio
async def test_evict_drain_poll_identity_check_rejects_fresh_same_key_entry() -> None:
    """The drain-poll must identity-check the tracked entry so a fresh
    same-key acquire that pops the marked entry and inserts a new one is
    not misread."""
    in_first_body = asyncio.Event()
    release_first = asyncio.Event()

    manager = SandboxSessionManager(
        max_sessions_per_provider=4,
        eviction_grace_seconds=1.0,
    )
    backend = _FakeBackend()

    async def first_acquire() -> None:
        async with manager.acquire(backend, "k1"):
            in_first_body.set()
            await release_first.wait()

    first = asyncio.create_task(first_acquire())
    await in_first_body.wait()

    async def evict() -> None:
        await manager.evict_for_session_key("k1")

    evict_task = asyncio.create_task(evict())
    await asyncio.sleep(0.02)

    release_first.set()
    await first

    async with manager.acquire(backend, "k1") as session:
        result = await session.execute("print('post-evict')")

    await asyncio.wait_for(evict_task, timeout=2.0)

    assert result.stdout == "print('post-evict')"
    assert backend.close_calls == [_ck("k1")]


@pytest.mark.asyncio
async def test_evict_for_provider_family_targets_only_matching_family() -> None:
    """``evict_for_provider_family`` marks/closes only entries whose
    ``family`` matches; other families are untouched."""

    class _BackendModal(_FakeBackend):
        family = "MODAL"

    class _BackendE2B(_FakeBackend):
        family = "E2B"

    manager = SandboxSessionManager(max_sessions_per_provider=8)
    modal_a = _BackendModal()
    modal_b = _BackendModal()
    e2b = _BackendE2B()

    async with manager.acquire(modal_a, "m1"):
        pass
    async with manager.acquire(modal_b, "m2"):
        pass
    async with manager.acquire(e2b, "e1"):
        pass

    await manager.evict_for_provider_family("MODAL")

    assert sorted(modal_a.close_calls) == [_ck("m1")]
    assert sorted(modal_b.close_calls) == [_ck("m2")]
    assert e2b.close_calls == []
    assert _ck("e1") in manager._tracked


@pytest.mark.asyncio
async def test_execute_routes_through_execute_in_session_with_bound_handle() -> None:
    """``session.execute`` must call ``backend.execute_in_session(handle, ...)``
    with the handle stored on the tracked entry — NOT ``backend.execute``."""
    manager = SandboxSessionManager()
    backend = _FakeBackend()

    async with manager.acquire(backend, "k1") as session:
        await session.execute("print(1)", timeout=42)
        await session.execute("print(2)")

    # Both executes ran via execute_in_session against the same handle.
    assert [c[0] for c in backend.execute_in_session_calls] == [_ck("k1"), _ck("k1")]
    assert [c[1] for c in backend.execute_in_session_calls] == ["print(1)", "print(2)"]
    assert backend.execute_in_session_calls[0][2] == 42
    assert backend.execute_in_session_calls[1][2] is None


@pytest.mark.asyncio
async def test_config_switch_under_same_logical_key_produces_distinct_tracked_entries() -> None:
    """A mid-iteration config change under a stable frontend ``session_key``
    must produce a fresh session at both the manager and the provider layer.

    Two ``acquire`` calls for the same logical key, against two backends
    whose ``config_fingerprint`` differs, must:
      - allocate two distinct ``_tracked`` entries (keyed on the composite
        ``f"{session_key}#{fingerprint}"`` per backend),
      - call ``find_or_create_session`` once per composite (so the provider
        sees the fingerprint and can fragment its label / list-by-key
        convergence along the same boundary),
      - leave ``acquire(backend, session_key)`` itself unchanged at the
        call site — the caller still passes the logical key.
    """

    class _BackendOldConfig(_FakeBackend):
        _fingerprint = "old-cfg"

    class _BackendNewConfig(_FakeBackend):
        _fingerprint = "new-cfg"

    manager = SandboxSessionManager(max_sessions_per_provider=8)
    backend_old = _BackendOldConfig()
    backend_new = _BackendNewConfig()

    async with manager.acquire(backend_old, "frontend-session"):
        pass
    async with manager.acquire(backend_new, "frontend-session"):
        pass

    # Distinct tracked entries under the same logical key.
    assert _ck("frontend-session", "old-cfg") in manager._tracked
    assert _ck("frontend-session", "new-cfg") in manager._tracked
    assert len(manager._tracked) == 2

    # Each backend saw exactly one find_or_create_session with ITS composite.
    assert backend_old.find_calls == [_ck("frontend-session", "old-cfg")]
    assert backend_new.find_calls == [_ck("frontend-session", "new-cfg")]


@pytest.mark.asyncio
async def test_evict_for_session_key_drains_every_composite_under_logical_key() -> None:
    """``evict_for_session_key("k")`` evicts every tracked entry whose
    composite key prefix matches ``"k#"``, so a single logical-key eviction
    drains all config-variants under that key.

    This matches the documented ``stopEvaluatorSession`` contract: the
    frontend passes a logical session id and expects every backend session
    bound to it to stop, regardless of mid-iteration config drift.
    """

    class _BackendCfgA(_FakeBackend):
        _fingerprint = "cfg-a"

    class _BackendCfgB(_FakeBackend):
        _fingerprint = "cfg-b"

    manager = SandboxSessionManager(max_sessions_per_provider=8)
    backend_a = _BackendCfgA()
    backend_b = _BackendCfgB()

    async with manager.acquire(backend_a, "logical-k"):
        pass
    async with manager.acquire(backend_b, "logical-k"):
        pass
    assert len(manager._tracked) == 2

    await manager.evict_for_session_key("logical-k")

    assert manager._tracked == {}
    # Each variant's close_session ran exactly once with its own composite.
    assert backend_a.close_calls == [_ck("logical-k", "cfg-a")]
    assert backend_b.close_calls == [_ck("logical-k", "cfg-b")]
