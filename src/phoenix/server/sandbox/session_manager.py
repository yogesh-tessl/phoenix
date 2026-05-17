"""
SandboxSessionManager — central authority for sandbox session lifecycle.

Hoists per-key locking above the backend layer so all session-capable
backends (Modal, Vercel, E2B, Daytona) share one creation-dedup pathway.
Tracks idle TTL and in-flight reference counts; evicts unused sessions
in a background sweeper. Stateless backends (``BaseNoSessionBackend``)
are short-circuited — ``acquire`` returns a handle whose ``execute``
calls the backend directly without locking or state tracking.

The manager is constructed in ``create_app`` and registered in
``_lifespan``'s ``AsyncExitStack`` alongside other daemons (one per
process). Application code obtains the instance via
``info.context.sandbox_session_manager`` (GraphQL) or
``request.app.state.sandbox_session_manager`` (FastAPI) — never by
constructing a new manager.

Keying model: ``_tracked`` and ``_key_locks`` are keyed on an internal
composite ``f"{session_key}#{backend.config_fingerprint()}"`` string.
Callers pass their opaque logical ``session_key`` to ``acquire`` /
``evict_for_session_key`` / ``schedule_eviction`` — composition with the
backend's config fingerprint happens inside the manager. The fingerprint
captures the provider family, package list, internet-access mode, and
env-var key set, so a mid-iteration config change under a stable
``session_key`` produces a fresh tracked entry (and a fresh remote
session via ``find_or_create_session(composite_key)``) rather than
silently dispatching against the old container. Backend wrappers
remain ephemeral; wrapper identity is not load-bearing for session
dispatch. Sessions are bound at the provider via
``backend.find_or_create_session(composite_key)``, which returns an
opaque remote handle that the manager retains on ``_TrackedSession``
and passes back to ``backend.execute_in_session(handle, code,
timeout=...)`` on each execute.
"""

from __future__ import annotations

import asyncio
import logging
import os
from asyncio import Event, Lock, Task, sleep
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from time import monotonic
from typing import AsyncIterator, Optional

from phoenix.server.sandbox.types import (
    BaseNoSessionBackend,
    ExecutionResult,
    SandboxBackend,
)
from phoenix.server.types import DaemonTask

logger = logging.getLogger(__name__)


_DEFAULT_IDLE_TTL_SECONDS = 300.0
_DEFAULT_SWEEP_INTERVAL_SECONDS = 30.0
_DEFAULT_EVICTION_GRACE_SECONDS = 5.0
_DEFAULT_MAX_SESSIONS_PER_PROVIDER = 32


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid float for %s=%r; using default %s", name, raw, default)
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid int for %s=%r; using default %s", name, raw, default)
        return default


def _family_of(backend: SandboxBackend) -> str:
    # Falls back to ``type(backend).__name__`` because ``SandboxBackend``
    # wrappers do not carry a ``family`` attribute today; ``SandboxAdapter.family``
    # lives on the factory class. Aligning the namespaces is deferred — adding
    # ``family = "MODAL"`` etc. to each backend wrapper class is a one-line
    # change that this ``getattr`` picks up automatically when it happens.
    return getattr(backend, "family", type(backend).__name__)


class SessionLimitExceeded(Exception):
    """Raised by ``SandboxSessionManager.acquire`` when a provider family has
    reached ``max_sessions_per_provider``.

    Capacity refusal — not a backend execution failure. Raised BEFORE
    ``backend.find_or_create_session`` is called, and the manager's internal
    state is unchanged when the exception propagates. ``CodeEvaluatorRunner``
    converts this to an ``ExecutionResult`` with a stable error code so the
    UI sees a deterministic, user-actionable message.
    """

    MESSAGE = "session_limit_exceeded"

    def __init__(self, message: str = MESSAGE) -> None:
        super().__init__(message)


class SessionInvalidated(Exception):
    """Raised by ``SandboxSessionManager.acquire`` when a tracked session has
    been marked for eviction.

    Admit-block refusal — the existing entry is draining and will be stopped
    once in-flight users release. New admits must not extend its lifetime
    past the explicit stop request. Raised BEFORE
    ``backend.find_or_create_session`` is called; the manager's internal
    state is unchanged when the exception propagates. ``CodeEvaluatorRunner``
    converts this to an ``ExecutionResult`` with a stable error code so the
    UI sees a deterministic, user-actionable message rather than a silent
    restart.
    """

    MESSAGE = "session_invalidated"

    def __init__(self, message: str = MESSAGE) -> None:
        super().__init__(message)


@dataclass
class _TrackedSession:
    backend: SandboxBackend
    session_key: str
    family: str
    # Opaque remote handle returned by ``backend.find_or_create_session``.
    # Manager passes this back to ``backend.execute_in_session`` on each
    # execute; the manager treats it as opaque. ``None`` between reservation
    # and the leader's first successful find_or_create_session.
    handle: object = None
    in_flight_count: int = 0
    last_used: float = field(default_factory=monotonic)
    marked_for_eviction: bool = False
    # Startup-readiness gate: the leader (is_new=True acquire) flips
    # ``start_ready`` after ``backend.find_or_create_session`` resolves so
    # concurrent followers waiting on the same key block from crossing the
    # in_flight boundary until the remote session actually exists. On failure
    # the leader records the exception in ``start_error`` and signals
    # readiness so followers wake and re-raise the same error.
    start_ready: Event = field(default_factory=Event)
    start_error: Optional[BaseException] = None


class SandboxSession:
    """Handle returned by ``SandboxSessionManager.acquire``.

    Calls ``backend.execute_in_session(handle, code, timeout=timeout)``
    against the remote handle the manager bound at acquire time. In-flight
    ref-counting and last-used tracking are handled by the manager around
    the ``async with`` boundary.
    """

    def __init__(
        self,
        backend: SandboxBackend,
        session_key: str,
        handle: object,
    ) -> None:
        self._backend = backend
        self._session_key = session_key
        self._handle = handle

    @property
    def session_key(self) -> str:
        return self._session_key

    async def execute(
        self,
        code: str,
        timeout: Optional[int] = None,
    ) -> ExecutionResult:
        return await self._backend.execute_in_session(self._handle, code, timeout=timeout)


class SandboxSessionManager(DaemonTask):
    """Central session lifecycle authority for sandbox backends.

    - ``acquire(backend, session_key)`` — async context manager. Starts (or
      reuses) a session and yields a ``SandboxSession``. Enforces
      ``max_sessions_per_provider`` (counted by adapter family) before
      invoking ``backend.find_or_create_session``. Refuses admits onto
      entries marked for eviction by raising ``SessionInvalidated``.
    - ``evict_for_session_key(session_key)`` — invalidation by opaque
      session_key. Sessions with ``in_flight_count == 0`` are closed
      immediately; in-use sessions are marked for stop-on-release.
    - ``evict_for_provider_family(family)`` — scans ``_tracked`` and marks
      every entry whose ``family`` matches.
    - ``schedule_eviction(session_key)`` — fire-and-forget API for callers
      that cannot ``await`` the eviction inline (notably the per-execute
      timeout teardown in ``CodeEvaluatorRunner``, which runs in both
      GraphQL request and ``ExperimentRunner`` daemon scopes). The created
      task is retained on the manager so shutdown awaits it.
    - Background sweeper (``_run``) evicts entries whose ``last_used``
      crosses the idle TTL with ``in_flight_count == 0``.

    Stateless backends (``BaseNoSessionBackend``) bypass all locking and
    state tracking — ``acquire`` returns a handle wired directly to
    ``backend.execute_in_session`` (which itself delegates to
    ``backend.execute``).

    **Lock-order invariant.** Two locks coexist: a single ``_state_lock``
    that guards the ``_tracked`` / ``_key_locks`` dicts, and one
    ``asyncio.Lock`` per ``session_key`` in ``_key_locks``. ``acquire``
    takes the per-key lock OUTER and ``_state_lock`` INNER (existence
    check, capacity check, and reservation insertion all run under
    ``_state_lock`` so a fresh same-key acquire sees a leader's reservation
    immediately). Eviction paths take ``_state_lock`` only long enough to
    snapshot the target keys, then release it before acquiring each
    per-key lock — holding both locks across an ``await`` on
    ``backend.close_session`` would deadlock against an in-flight acquire
    on the same key.

    **Pop-before-await invariant.** The manager releases its per-key lock
    BEFORE awaiting ``backend.close_session`` (see ``_release``,
    ``_evict_targets``, and ``_sweep_idle``). Adapters are required to pop
    their backend-local bookkeeping (e.g. a ``_sessions[session_key]``
    map) synchronously before their first ``await`` so a concurrent
    same-key ``find_or_create_session`` cannot race the close.

    **Shutdown ordering.** ``stop()`` (overridden from ``DaemonTask``)
    drains in three phases under the inherited 10s ceiling:
    (1) await ``_pending_tasks`` (fire-and-forget evictions from
    ``schedule_eviction``) via ``asyncio.wait(timeout=eviction_grace_seconds)``
    so their underlying ``close_session`` calls complete; any tasks still
    pending past the grace window are left running (not cancelled) and
    logged as a warning; (2) snapshot every tracked session_key and
    drain via ``_evict_targets`` (which marks in-flight entries and
    waits up to ``eviction_grace_seconds`` for them to drain);
    (3) ``super().stop()`` cancels the sweeper. ``_pending_tasks`` is a
    set held adjacent to (not merged into) the inherited ``self._tasks``:
    ``self._tasks`` holds tasks spawned from ``_run`` and is drained by
    cancel-then-gather; ``_pending_tasks`` holds externally-scheduled
    work the manager owns through completion, so it is awaited (not
    cancelled) at shutdown.
    """

    def __init__(
        self,
        *,
        idle_ttl_seconds: Optional[float] = None,
        sweep_interval_seconds: Optional[float] = None,
        eviction_grace_seconds: Optional[float] = None,
        max_sessions_per_provider: Optional[int] = None,
    ) -> None:
        super().__init__()
        self._idle_ttl_seconds: float = (
            idle_ttl_seconds
            if idle_ttl_seconds is not None
            else _env_float(
                "PHOENIX_SANDBOX_SESSION_IDLE_TTL_SECONDS",
                _DEFAULT_IDLE_TTL_SECONDS,
            )
        )
        self._sweep_interval_seconds: float = (
            sweep_interval_seconds
            if sweep_interval_seconds is not None
            else _env_float(
                "PHOENIX_SANDBOX_SWEEP_INTERVAL_SECONDS",
                _DEFAULT_SWEEP_INTERVAL_SECONDS,
            )
        )
        self._eviction_grace_seconds: float = (
            eviction_grace_seconds
            if eviction_grace_seconds is not None
            else _env_float(
                "PHOENIX_SANDBOX_EVICTION_GRACE_SECONDS",
                _DEFAULT_EVICTION_GRACE_SECONDS,
            )
        )
        self._max_sessions_per_provider: int = (
            max_sessions_per_provider
            if max_sessions_per_provider is not None
            else _env_int(
                "PHOENIX_SANDBOX_MAX_SESSIONS_PER_PROVIDER",
                _DEFAULT_MAX_SESSIONS_PER_PROVIDER,
            )
        )

        # asyncio primitives are loop-bound; construct lazily inside the
        # running loop (start() / acquire()) rather than at __init__ time.
        self._state_lock: Optional[Lock] = None
        self._key_locks: dict[str, Lock] = {}
        self._tracked: dict[str, _TrackedSession] = {}
        # Fire-and-forget eviction tasks scheduled via schedule_eviction.
        # Held adjacent to (not merged into) self._tasks: self._tasks is the
        # daemon body spawned from _run() and is cancelled by DaemonTask.stop;
        # _pending_tasks is awaited (not cancelled) on stop so the underlying
        # backend.close_session completes rather than being cancelled mid-flight.
        self._pending_tasks: Optional[set[Task[None]]] = None

    @property
    def eviction_grace_seconds(self) -> float:
        """Maximum time eviction paths wait for in-flight sessions to drain
        before returning. Settable to compress shutdown / invalidation timing
        in tests."""
        return self._eviction_grace_seconds

    @eviction_grace_seconds.setter
    def eviction_grace_seconds(self, value: float) -> None:
        self._eviction_grace_seconds = value

    # ------------------------------------------------------------------
    # Public surface
    # ------------------------------------------------------------------

    @asynccontextmanager
    async def acquire(
        self,
        backend: SandboxBackend,
        session_key: str,
    ) -> AsyncIterator[SandboxSession]:
        """Yield a ``SandboxSession`` bound to ``session_key``.

        For session-capable backends: ensures the session is bound under a
        per-key lock (leader calls ``backend.find_or_create_session`` and
        stores the returned handle on ``_TrackedSession``; followers reuse
        the same handle), increments in-flight count on enter, decrements on
        exit, and fires ``backend.close_session`` on exit if the session was
        marked for eviction and in-flight reaches zero.

        For ``BaseNoSessionBackend`` backends: short-circuits — no locking,
        no state tracking, no eviction. The sentinel handle from
        ``find_or_create_session`` is passed through to
        ``execute_in_session`` (which itself delegates to ``execute``).
        """
        if isinstance(backend, BaseNoSessionBackend):
            sentinel = await backend.find_or_create_session(session_key)
            yield SandboxSession(backend, session_key, sentinel)
            return

        # Compose a composite internal key so a mid-iteration config change
        # under the same logical ``session_key`` produces a fresh entry. The
        # logical key remains the caller-facing surface; the fingerprint is
        # an internal disambiguator. The composite is also passed to
        # ``backend.find_or_create_session`` so provider-side label / list-by-key
        # convergence fragments along the same boundary.
        composite_key = self._composite_key(session_key, backend)

        await self._ensure_state_lock()
        key_lock = await self._get_or_create_key_lock(composite_key)

        async with key_lock:
            # Existence + capacity + insertion are atomic under
            # ``_state_lock``: when ``_key_locks`` becomes poppable, two
            # acquires for the same logical key may hold different per-key
            # locks, so the per-key lock is no longer the authority for
            # dedup. Folding the existence check into the capacity-check +
            # insertion critical section makes ``_state_lock`` the inner
            # authority and closes the duplicate-start gap.
            tracked, is_new = await self._get_or_reserve(backend, session_key, composite_key)
            if is_new:
                try:
                    handle = await backend.find_or_create_session(composite_key)
                except BaseException as exc:
                    # find_or_create_session failed: drop the reservation so
                    # capacity accounting stays correct, then wake any
                    # followers waiting on the readiness gate so they observe
                    # the error and re-raise. ``_key_locks`` is popped here
                    # too so a failure storm doesn't slow-leak lock entries —
                    # safe because no other coroutine can be parked on
                    # ``key_lock`` while this acquire holds it.
                    tracked.start_error = exc
                    assert self._state_lock is not None
                    async with self._state_lock:
                        self._tracked.pop(composite_key, None)
                        self._key_locks.pop(composite_key, None)
                    tracked.start_ready.set()
                    raise
                tracked.handle = handle
                tracked.start_ready.set()

            # All acquires (leader and follower) wait for startup to settle
            # before crossing the in_flight boundary. The leader sets
            # ``start_ready`` immediately above; a follower may park here
            # and wake only after the leader's find_or_create_session
            # resolves. If startup failed, propagate the leader's exception
            # so the follower doesn't yield a SandboxSession against a
            # remote session that does not exist.
            await tracked.start_ready.wait()
            if tracked.start_error is not None:
                raise tracked.start_error
            tracked.in_flight_count += 1
            tracked.last_used = monotonic()
            session_handle = tracked.handle

        try:
            yield SandboxSession(backend, session_key, session_handle)
        finally:
            await self._release(composite_key, key_lock)

    async def evict_for_session_key(self, session_key: str) -> None:
        """Evict every tracked entry for a logical ``session_key``.

        Internally the manager keys ``_tracked`` on a composite of the logical
        session key and the backend's config fingerprint, so a single logical
        key can have multiple tracked entries when the same frontend session
        has run against different backend configs (e.g. mid-iteration
        provider / packages / env-vars switch). A logical eviction drains
        every variant under the same ``session_key`` so callers using the
        chat ``stopEvaluatorSession`` mutation see the documented "stop this
        session" semantics regardless of config drift.

        Entries with ``in_flight_count == 0`` are closed immediately; in-use
        entries are marked for eviction and closed on release. No-op when no
        entry matches the prefix.
        """
        await self._ensure_state_lock()
        assert self._state_lock is not None
        prefix = f"{session_key}#"
        async with self._state_lock:
            targets = [k for k in self._tracked if k.startswith(prefix)]
        await self._evict_targets(targets)

    async def evict_for_provider_family(self, family: str) -> None:
        """Evict every tracked session whose adapter family matches ``family``.

        Sessions with ``in_flight_count == 0`` are closed immediately;
        in-use sessions are marked for close-on-release.
        """
        await self._ensure_state_lock()
        assert self._state_lock is not None
        async with self._state_lock:
            targets = [k for k, t in self._tracked.items() if t.family == family]
        await self._evict_targets(targets)

    def schedule_eviction(self, session_key: str) -> None:
        """Schedule a fire-and-forget eviction of ``session_key``.

        Used by callers that cannot await the eviction inline — notably the
        per-execute timeout teardown in ``CodeEvaluatorRunner.evaluate``,
        which runs in both a GraphQL request scope and the
        ``ExperimentRunner`` daemon scope. The created task is retained on
        the manager so it is awaited (not cancelled) during shutdown,
        completing the underlying ``backend.close_session`` rather than
        leaking the orphan.
        """
        if self._pending_tasks is None:
            self._pending_tasks = set()
        task: Task[None] = asyncio.create_task(self.evict_for_session_key(session_key))
        self._pending_tasks.add(task)
        task.add_done_callback(self._pending_tasks.discard)

    # ------------------------------------------------------------------
    # DaemonTask body
    # ------------------------------------------------------------------

    async def _run(self) -> None:
        await self._ensure_state_lock()
        while self._running:
            try:
                await self._sweep_idle()
            except Exception:
                logger.exception("Sandbox session manager sweep failed")
            await sleep(self._sweep_interval_seconds)

    async def stop(self) -> None:
        """Drain in-flight sessions, then cancel the sweeper.

        Ordering matters: ``_pending_tasks`` (fire-and-forget evictions
        scheduled via ``schedule_eviction``) are awaited first via
        ``asyncio.wait`` so the underlying ``backend.close_session`` calls
        complete instead of being cancelled mid-flight; tasks still pending
        past ``eviction_grace_seconds`` are left running (not cancelled)
        and logged as a warning. Then every tracked ``session_key`` is
        drained via ``_evict_targets`` (which marks in-flight entries for
        eviction and waits up to ``eviction_grace_seconds`` for them to
        release); finally ``super().stop()`` cancels the sweeper task.
        The combined wall clock budget must fit within
        ``DaemonTask.stop``'s 10s ceiling.
        """
        # Drain pending evictions first — they may already be the call that
        # would have popped a tracked entry. ``asyncio.wait`` returns the
        # (done, pending) split without cancelling the still-pending tasks,
        # so a slow ``backend.close_session`` past the grace window keeps
        # running to completion rather than being cancelled mid-flight.
        # Provider-side TTLs reap any orphan that does outlive the daemon.
        pending = self._pending_tasks
        if pending:
            _, still_pending = await asyncio.wait(pending, timeout=self._eviction_grace_seconds)
            if still_pending:
                logger.warning(
                    "Pending sandbox eviction tasks did not drain within %.2fs",
                    self._eviction_grace_seconds,
                )
        # Snapshot every tracked session_key under the state lock; drain
        # outside the lock so each per-key drain holds only its own per-key
        # lock (avoiding cross-key lock contention during shutdown).
        await self._ensure_state_lock()
        assert self._state_lock is not None
        async with self._state_lock:
            tracked_keys = list(self._tracked.keys())
        if tracked_keys:
            try:
                await self._evict_targets(tracked_keys)
            except Exception:
                logger.exception("Failed to drain tracked sandbox sessions during shutdown")
        await super().stop()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _ensure_state_lock(self) -> None:
        if self._state_lock is None:
            self._state_lock = Lock()

    async def _get_or_create_key_lock(self, session_key: str) -> Lock:
        assert self._state_lock is not None
        async with self._state_lock:
            lock = self._key_locks.get(session_key)
            if lock is None:
                lock = Lock()
                self._key_locks[session_key] = lock
            return lock

    def _composite_key(self, session_key: str, backend: SandboxBackend) -> str:
        """Compose the internal manager key from the caller's logical
        ``session_key`` and the backend's config fingerprint.

        Format: ``f"{session_key}#{fingerprint}"``. The ``#`` separator is
        a stable lexical delimiter that ``evict_for_session_key`` relies on
        for its prefix scan; callers never see this composition.
        """
        return f"{session_key}#{backend.config_fingerprint()}"

    async def _get_or_reserve(
        self,
        backend: SandboxBackend,
        session_key: str,
        composite_key: str,
    ) -> tuple[_TrackedSession, bool]:
        """Atomically resolve an existing tracking slot or reserve a fresh one.

        Under ``_state_lock``: (1) if ``_tracked[composite_key]`` exists,
        return ``(existing, False)`` without rechecking capacity — followers
        piggy-back on the leader's reservation. (2) Otherwise enforce
        ``max_sessions_per_provider`` (counted by adapter family) and insert
        a fresh ``_TrackedSession`` whose ``start_ready`` is unset; return
        ``(new, True)``. The leader (``is_new=True``) is responsible for
        invoking ``backend.find_or_create_session`` outside this lock and
        flipping ``start_ready`` when it resolves.

        ``session_key`` is the caller's logical key (recorded on
        ``_TrackedSession`` for diagnostic logs); ``composite_key`` is the
        internal manager key that includes the backend's config fingerprint
        and is also what the leader passes to ``find_or_create_session`` /
        ``close_session`` so provider-side label / list-by-key convergence
        fragments along the same boundary.

        Folding the existence check into the same critical section as the
        capacity check + insertion is what makes popping ``_key_locks`` safe:
        once the per-key lock is no longer the dedup authority, ``_state_lock``
        must be — so the existence check, capacity check, and insertion all
        run under it. The caller is responsible for popping the entry under
        ``_state_lock`` if ``backend.find_or_create_session`` fails (see
        ``acquire``'s rollback).
        """
        assert self._state_lock is not None
        family = _family_of(backend)
        async with self._state_lock:
            existing = self._tracked.get(composite_key)
            if existing is not None:
                # Refuse to admit new callers onto a session that another
                # caller has explicitly invalidated. The entry is draining
                # and will be stopped on its last release; piggy-backing a
                # fresh acquire would extend its lifetime past the stop
                # request. Surface a deterministic error rather than a
                # silent restart so the UI can decide whether to retry.
                if existing.marked_for_eviction:
                    raise SessionInvalidated()
                return existing, False
            count = sum(1 for t in self._tracked.values() if t.family == family)
            if count >= self._max_sessions_per_provider:
                raise SessionLimitExceeded()
            tracked = _TrackedSession(
                backend=backend,
                session_key=composite_key,
                family=family,
            )
            self._tracked[composite_key] = tracked
            return tracked, True

    async def _release(self, session_key: str, key_lock: Lock) -> None:
        # ``acquire`` passes the lock reference it captured at acquire-time,
        # so ``_release`` never has to look ``_key_locks`` up. The lock
        # object outlives its dict entry, which is what makes popping
        # ``_key_locks`` alongside ``_tracked`` safe — a coroutine that
        # captured the lock before a concurrent eviction popped the dict
        # entry still releases through the same Lock instance.
        # Snapshot decisions under the per-key lock, run close_session
        # outside the lock to avoid blocking other acquires on the same key.
        close_backend: Optional[SandboxBackend] = None
        close_key: Optional[str] = None
        async with key_lock:
            tracked = self._tracked.get(session_key)
            if tracked is None:
                return
            tracked.in_flight_count = max(0, tracked.in_flight_count - 1)
            tracked.last_used = monotonic()
            if tracked.marked_for_eviction and tracked.in_flight_count == 0:
                close_backend = tracked.backend
                close_key = tracked.session_key
                self._tracked.pop(session_key, None)
                self._key_locks.pop(session_key, None)
        if close_backend is not None and close_key is not None:
            try:
                await close_backend.close_session(close_key)
            except Exception:
                logger.exception(
                    "Failed to close sandbox session on release: backend=%r key=%r",
                    type(close_backend).__name__,
                    close_key,
                )

    async def _evict_targets(self, targets: list[str]) -> None:
        """Drain ``targets``: close idle entries immediately, mark in-flight
        entries for stop-on-release, then poll up to ``eviction_grace_seconds``
        for marked entries to drain.

        ``targets`` may include keys that have already been removed (e.g.
        by a concurrent release) — those are skipped silently.
        """
        await self._ensure_state_lock()
        # First pass: stop idle sessions immediately; mark in-flight sessions
        # for eviction. Track the latter (with the tracked instance, not just
        # the key) so the drain poll can identity-check against the same
        # entry — a fresh same-key acquire that pops the marked entry and
        # inserts a new one must not be misread as the marked entry having
        # drained.
        marked: list[tuple[str, _TrackedSession]] = []
        for session_key in targets:
            key_lock = self._key_locks.get(session_key)
            if key_lock is None:
                continue
            close_backend: Optional[SandboxBackend] = None
            close_key: Optional[str] = None
            async with key_lock:
                tracked = self._tracked.get(session_key)
                if tracked is None:
                    continue
                if tracked.in_flight_count == 0:
                    close_backend = tracked.backend
                    close_key = tracked.session_key
                    self._tracked.pop(session_key, None)
                    self._key_locks.pop(session_key, None)
                else:
                    tracked.marked_for_eviction = True
                    marked.append((session_key, tracked))
            if close_backend is not None and close_key is not None:
                try:
                    await close_backend.close_session(close_key)
                except Exception:
                    logger.exception(
                        "Failed to close sandbox session during eviction: backend=%r key=%r",
                        type(close_backend).__name__,
                        close_key,
                    )
        # Second pass: wait up to ``eviction_grace_seconds`` for marked
        # sessions to drain. Once ``in_flight_count`` reaches zero, the
        # corresponding ``_release`` callback fires ``backend.close_session``
        # and removes the entry from ``_tracked``. If the grace window
        # expires with sessions still in-flight, leave them marked — the
        # next release will still close them. This is a bounded best-effort
        # drain, not a guaranteed barrier.
        if not marked:
            return
        deadline = monotonic() + self._eviction_grace_seconds
        poll_interval = min(0.05, max(0.005, self._eviction_grace_seconds / 20.0))
        assert self._state_lock is not None
        while marked and monotonic() < deadline:
            await sleep(poll_interval)
            async with self._state_lock:
                # Identity check: the entry is still draining only if the
                # same ``_TrackedSession`` instance is still under the key.
                # A fresh acquire on the same key inserts a new instance
                # (the old one is popped by its last release) — that fresh
                # entry is not what we marked, so drop it from the wait set.
                marked = [(k, t) for (k, t) in marked if self._tracked.get(k) is t]

    async def _sweep_idle(self) -> None:
        await self._ensure_state_lock()
        now = monotonic()
        ttl = self._idle_ttl_seconds
        assert self._state_lock is not None
        async with self._state_lock:
            candidates = [
                k
                for k, t in list(self._tracked.items())
                if t.in_flight_count == 0 and now - t.last_used > ttl
            ]
        for session_key in candidates:
            key_lock = self._key_locks.get(session_key)
            if key_lock is None:
                continue
            close_backend: Optional[SandboxBackend] = None
            close_key: Optional[str] = None
            async with key_lock:
                tracked = self._tracked.get(session_key)
                if tracked is None:
                    continue
                if tracked.in_flight_count != 0:
                    continue
                if monotonic() - tracked.last_used <= ttl:
                    continue
                close_backend = tracked.backend
                close_key = tracked.session_key
                self._tracked.pop(session_key, None)
                self._key_locks.pop(session_key, None)
            if close_backend is not None and close_key is not None:
                try:
                    await close_backend.close_session(close_key)
                except Exception:
                    logger.exception(
                        "Failed to close idle sandbox session: backend=%r key=%r",
                        type(close_backend).__name__,
                        close_key,
                    )
