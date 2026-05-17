"""Tests for the asyncio.wait_for timeout wrapper in CodeEvaluatorRunner.evaluate."""

from __future__ import annotations

import asyncio
import logging

import pytest

from phoenix.db.types.annotation_configs import (
    CategoricalAnnotationValue,
    CategoricalOutputConfig,
    OptimizationDirection,
)
from phoenix.db.types.evaluators import InputMapping
from phoenix.server.api.evaluators import CodeEvaluatorRunner
from phoenix.server.sandbox.session_manager import SandboxSessionManager
from phoenix.server.sandbox.types import ExecutionResult, SandboxBackend


def _categorical_config() -> CategoricalOutputConfig:
    return CategoricalOutputConfig(
        type="CATEGORICAL",
        name="score",
        optimization_direction=OptimizationDirection.MAXIMIZE,
        description="",
        values=[
            CategoricalAnnotationValue(label="pass", score=1.0),
            CategoricalAnnotationValue(label="fail", score=0.0),
        ],
    )


def _make_runner(
    backend: SandboxBackend,
    timeout: int = 1,
    manager: SandboxSessionManager | None = None,
) -> CodeEvaluatorRunner:
    test_manager = manager if manager is not None else SandboxSessionManager()
    test_manager.eviction_grace_seconds = 0.05
    return CodeEvaluatorRunner(
        name="test-runner",
        description=None,
        source_code='def evaluate(**kw): return "pass"',
        stored_output_configs=[_categorical_config()],
        sandbox_backend=backend,
        language="PYTHON",
        timeout=timeout,
        sandbox_session_manager=test_manager,
    )


_EMPTY_MAPPING = InputMapping(literal_mapping={}, path_mapping={})


class _SlowBackend(SandboxBackend):
    """Backend whose execute_in_session sleeps indefinitely; close_session is tracked.

    Implements the session-bound ABC: ``find_or_create_session`` /
    ``execute_in_session`` / ``close_session`` / ``execute`` / ``close``.
    The runner now drives all execution through the manager, which calls
    ``execute_in_session`` — that's where the sleep lives.
    """

    # CodeEvaluatorRunner reads secret_values to seed SandboxSecretMasker.
    secret_values: frozenset[str] = frozenset()
    family: str = "TEST"

    def __init__(self, close_raises: Exception | None = None) -> None:
        self.close_session_calls: list[str] = []
        self._close_raises = close_raises

    def config_fingerprint(self) -> str:
        return "slow"

    async def find_or_create_session(self, session_key: str) -> object:
        return object()

    async def execute_in_session(
        self,
        handle: object,
        code: str,
        timeout: int | None = None,
    ) -> ExecutionResult:
        await asyncio.sleep(60)
        return ExecutionResult(stdout='"pass"', stderr="", error=None)

    async def close_session(self, session_key: str) -> None:
        self.close_session_calls.append(session_key)
        if self._close_raises is not None:
            raise self._close_raises

    async def execute(
        self,
        code: str,
        session_key: str = "",
        timeout: int | None = None,
    ) -> ExecutionResult:
        await asyncio.sleep(60)
        return ExecutionResult(stdout='"pass"', stderr="", error=None)

    async def close(self) -> None:
        pass


class _FastBackend(SandboxBackend):
    """Backend that returns immediately with a configurable result."""

    secret_values: frozenset[str] = frozenset()
    family: str = "TEST"

    def __init__(self, result: ExecutionResult) -> None:
        self._result = result
        self.close_session_calls: list[str] = []

    def config_fingerprint(self) -> str:
        return "fast"

    async def find_or_create_session(self, session_key: str) -> object:
        return object()

    async def execute_in_session(
        self,
        handle: object,
        code: str,
        timeout: int | None = None,
    ) -> ExecutionResult:
        return self._result

    async def close_session(self, session_key: str) -> None:
        self.close_session_calls.append(session_key)

    async def execute(
        self,
        code: str,
        session_key: str = "",
        timeout: int | None = None,
    ) -> ExecutionResult:
        return self._result

    async def close(self) -> None:
        pass


class TestTimeoutWrapper:
    @pytest.mark.asyncio
    async def test_slow_backend_returns_timeout_execution_result(self) -> None:
        backend = _SlowBackend()
        runner = _make_runner(backend, timeout=1)
        results = await runner.evaluate(
            context={},
            input_mapping=_EMPTY_MAPPING,
            name="test",
            output_configs=[_categorical_config()],
        )

        assert len(results) == 1
        assert results[0]["error"] == "timeout"

    @pytest.mark.asyncio
    async def test_slow_backend_schedules_close_session(self) -> None:
        backend = _SlowBackend()
        runner = _make_runner(backend, timeout=1)
        await runner.evaluate(
            context={},
            input_mapping=_EMPTY_MAPPING,
            name="test",
            output_configs=[_categorical_config()],
        )

        for _ in range(50):
            await asyncio.sleep(0.01)
            if backend.close_session_calls:
                break
        assert len(backend.close_session_calls) == 1

    @pytest.mark.asyncio
    async def test_close_session_exception_during_timeout_does_not_propagate(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        backend = _SlowBackend(close_raises=RuntimeError("close failed"))
        runner = _make_runner(backend, timeout=1)

        with caplog.at_level(logging.WARNING):
            results = await runner.evaluate(
                context={},
                input_mapping=_EMPTY_MAPPING,
                name="test",
                output_configs=[_categorical_config()],
            )
            for _ in range(50):
                await asyncio.sleep(0.01)
                if any("close" in r.message.lower() for r in caplog.records):
                    break

        assert len(results) == 1
        assert results[0]["error"] == "timeout"
        assert any("close" in r.message.lower() for r in caplog.records)

    @pytest.mark.asyncio
    async def test_fast_backend_result_passes_through_unchanged(self) -> None:
        backend = _FastBackend(ExecutionResult(stdout='"pass"', stderr="", error=None))
        runner = _make_runner(backend, timeout=30)
        results = await runner.evaluate(
            context={},
            input_mapping=_EMPTY_MAPPING,
            name="test",
            output_configs=[_categorical_config()],
        )

        assert len(results) == 1
        assert results[0]["error"] is None
        assert results[0]["label"] == "pass"
        assert len(backend.close_session_calls) == 0

    @pytest.mark.asyncio
    async def test_fast_backend_with_error_field_passes_through_unchanged(self) -> None:
        backend = _FastBackend(ExecutionResult(stdout="", stderr="", error="runtime error"))
        runner = _make_runner(backend, timeout=30)
        results = await runner.evaluate(
            context={},
            input_mapping=_EMPTY_MAPPING,
            name="test",
            output_configs=[_categorical_config()],
        )

        assert len(results) == 1
        assert results[0]["error"] == "runtime error"
        assert len(backend.close_session_calls) == 0


class TestTimeoutTeardownKeying:
    """Regression tests for the timeout teardown path.

    Guards two bugs:

    1. Wrong session key (resource leak). When ``session_key`` is overridden
       (the frontend-generated UUID path), teardown must use the override key
       — not ``self._name``. Otherwise the backend's ``close_session`` looks
       up a non-existent key and leaks the sandbox.

    2. Manager state desync (dead-session reuse). Teardown must route through
       ``schedule_eviction(session_key)`` so the manager's ``_tracked`` entry
       is removed atomically with the provider-side teardown.
    """

    @pytest.mark.asyncio
    async def test_override_session_key_used_for_teardown(self) -> None:
        backend = _SlowBackend()
        manager = SandboxSessionManager()
        manager.eviction_grace_seconds = 0.05
        override = "00000000-1111-2222-3333-444444444444"
        runner = CodeEvaluatorRunner(
            name="test-runner",
            description=None,
            source_code='def evaluate(**kw): return "pass"',
            stored_output_configs=[_categorical_config()],
            sandbox_backend=backend,
            language="PYTHON",
            timeout=1,
            session_key=override,
            sandbox_session_manager=manager,
        )

        await runner.evaluate(
            context={},
            input_mapping=_EMPTY_MAPPING,
            name="test",
            output_configs=[_categorical_config()],
        )

        for _ in range(50):
            await asyncio.sleep(0.01)
            if backend.close_session_calls:
                break

        # close_session must be called with a composite key derived from the
        # override key — NOT self._name. The manager composes
        # f"{session_key}#{backend.config_fingerprint()}" internally.
        assert backend.close_session_calls == [f"{override}#slow"]

    @pytest.mark.asyncio
    async def test_manager_teardown_evicts_tracked_entry(self) -> None:
        backend = _SlowBackend()
        manager = SandboxSessionManager()
        manager.eviction_grace_seconds = 0.05

        runner = CodeEvaluatorRunner(
            name="test-runner",
            description=None,
            source_code='def evaluate(**kw): return "pass"',
            stored_output_configs=[_categorical_config()],
            sandbox_backend=backend,
            language="PYTHON",
            timeout=1,
            sandbox_session_manager=manager,
        )

        await runner.evaluate(
            context={},
            input_mapping=_EMPTY_MAPPING,
            name="test",
            output_configs=[_categorical_config()],
        )

        for _ in range(50):
            await asyncio.sleep(0.01)
            if backend.close_session_calls and not manager._tracked:
                break

        assert backend.close_session_calls == ["test-runner#slow"]
        assert manager._tracked == {}
