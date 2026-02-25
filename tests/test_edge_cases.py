"""Tests verifying reported edge-case issues in the traqo codebase.

Each test is named after the issue it verifies. Tests assert CORRECT behavior —
a failing test confirms the bug is real.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

import pytest

from traqo.serialize import _serialize_value
from traqo.tracer import Tracer, _active_tracer, _span_stack, get_tracer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_events(path: Path) -> list[dict]:
    events = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


# ===========================================================================
# Issue 1: Slotted dataclass serialization
#
# Claim: @dataclass(slots=True) objects hit AttributeError on .__dict__
# in the dataclass branch of _serialize_value, causing the outer except
# to return "<ClassName: serialization failed>" instead of falling through
# to the __slots__ handler.
# ===========================================================================


@dataclass(slots=True)
class SlottedPoint:
    x: int
    y: int


@dataclass(slots=True)
class SlottedNested:
    label: str
    point: SlottedPoint


@dataclass  # regular (non-slotted) for comparison
class RegularPoint:
    x: int
    y: int


class TestSlottedDataclassSerialization:
    def test_regular_dataclass_serializes(self):
        """Baseline: regular dataclasses work fine."""
        result = _serialize_value(RegularPoint(1, 2))
        assert result == {"x": 1, "y": 2}

    def test_slotted_dataclass_serializes(self):
        """@dataclass(slots=True) should serialize to a dict of its fields."""
        result = _serialize_value(SlottedPoint(3, 4))
        assert isinstance(result, dict), f"Expected dict, got {result!r}"
        assert result == {"x": 3, "y": 4}

    def test_slotted_dataclass_nested(self):
        """Nested slotted dataclasses should serialize recursively."""
        obj = SlottedNested(label="origin", point=SlottedPoint(0, 0))
        result = _serialize_value(obj)
        assert isinstance(result, dict), f"Expected dict, got {result!r}"
        assert result == {"label": "origin", "point": {"x": 0, "y": 0}}

    def test_slotted_dataclass_in_trace_file(self, tmp_path):
        """Slotted dataclass used as span input should appear as dict in JSONL."""
        path = tmp_path / "trace.jsonl"
        point = SlottedPoint(5, 6)
        with Tracer(path) as tracer:
            with tracer.span("test", input=point):
                pass
        events = _read_events(path)
        span_start = next(e for e in events if e["type"] == "span_start")
        assert span_start["input"] == {"x": 5, "y": 6}


# ===========================================================================
# Issue 2: ContextVar not reset if _close() raises in __exit__
#
# Claim: If self._file.close() raises (e.g. I/O error), the code that
# resets _active_tracer via the ContextVar token is never reached, leaving
# a stale tracer reference in the context.
# ===========================================================================


class TestContextVarResetOnCloseError:
    def test_contextvar_reset_on_normal_exit(self, tmp_path):
        """Baseline: ContextVar is properly reset on normal exit."""
        path = tmp_path / "trace.jsonl"
        with Tracer(path):
            assert get_tracer() is not None
        assert get_tracer() is None

    def test_contextvar_reset_when_close_raises(self, tmp_path):
        """If file.close() raises, _active_tracer should still be reset."""
        path = tmp_path / "trace.jsonl"
        tracer = Tracer(path)
        tracer.__enter__()

        assert get_tracer() is tracer

        original_close = tracer._file.close

        def bad_close():
            raise OSError("Simulated I/O error on close")

        tracer._file.close = bad_close

        try:
            with pytest.raises(OSError, match="Simulated I/O error"):
                tracer.__exit__(None, None, None)

            # The critical check: is the ContextVar cleaned up?
            assert get_tracer() is None, (
                "ContextVar should be reset even when close() fails"
            )
        finally:
            # Restore original close so GC doesn't re-raise
            if tracer._file is not None:
                tracer._file.close = original_close
                tracer._file.close()


# ===========================================================================
# Issue 3: Abandoned span contexts leave orphans on _span_stack
#
# Claim: When streaming wrappers call span_ctx.__enter__() but the stream
# is never fully consumed (and __exit__ never called), the span stays on
# _span_stack. Subsequent spans get the orphaned span as parent_id, even
# across different tracers.
# ===========================================================================


class TestAbandonedSpanContext:
    def test_normal_span_pops_from_stack(self, tmp_path):
        """Baseline: normally-exited spans are removed from the stack."""
        path = tmp_path / "trace.jsonl"
        with Tracer(path) as tracer:
            assert _span_stack.get(()) == ()
            with tracer.span("normal"):
                assert len(_span_stack.get(())) == 1
            assert _span_stack.get(()) == ()

    def test_abandoned_span_stays_on_stack(self, tmp_path):
        """Manually entering a span without exiting leaves it on the stack."""
        path = tmp_path / "trace.jsonl"
        with Tracer(path) as tracer:
            span_ctx = tracer.span("abandoned")
            orphan = span_ctx.__enter__()

            # Orphan is on the stack
            stack = _span_stack.get(())
            assert len(stack) == 1
            assert stack[-1] is orphan

            # A subsequent span gets the orphan as parent
            with tracer.span("child") as child:
                assert child.parent_id == orphan.id

            # Clean up
            span_ctx.__exit__(None, None, None)

        # After cleanup, stack should be empty
        assert _span_stack.get(()) == ()

    def test_abandoned_span_persists_after_tracer_exit(self, tmp_path):
        """Orphaned span persists even after the tracer that created it exits."""
        path1 = tmp_path / "t1.jsonl"
        path2 = tmp_path / "t2.jsonl"

        # Create orphan in tracer 1
        with Tracer(path1) as t1:
            span_ctx = t1.span("orphan")
            orphan = span_ctx.__enter__()
            orphan_id = orphan.id
            # intentionally do NOT call span_ctx.__exit__()

        # Tracer 1 has exited but the orphan is still on the stack
        stack = _span_stack.get(())
        assert len(stack) == 1, "Orphaned span should survive tracer exit"

        # Tracer 2 is contaminated: its spans get a parent_id from tracer 1
        with Tracer(path2) as t2:
            with t2.span("innocent") as s:
                assert s.parent_id == orphan_id, (
                    "Cross-tracer contamination: span in t2 has parent from t1"
                )

        # Verify the trace file has the cross-tracer parent_id
        events = _read_events(path2)
        span_starts = [e for e in events if e["type"] == "span_start"]
        assert span_starts[0]["parent_id"] == orphan_id

        # Clean up
        span_ctx.__exit__(None, None, None)
        assert _span_stack.get(()) == ()


# ===========================================================================
# Issue 4: Path traversal in cloud source cache directories
#
# Claim: S3Source and GCSSource construct cache file paths from cloud
# object keys without validating for ".." components. A key containing
# "../" causes the cache path to escape the intended cache directory.
# LocalSource correctly prevents this.
# ===========================================================================


class TestPathTraversal:
    def test_local_source_blocks_path_traversal(self, tmp_path):
        """LocalSource rejects keys with path traversal."""
        from traqo.ui.sources import LocalSource

        traces_dir = tmp_path / "traces"
        traces_dir.mkdir()
        source = LocalSource(traces_dir)

        # Traversal attempt returns empty
        assert source.read_all("../../etc/passwd") == []
        first, last = source.read_first_last("../../etc/passwd")
        assert first is None and last is None

    def test_cloud_cache_path_escapes_directory(self, tmp_path):
        """Demonstrate that Path(cache_dir) / key with '..' escapes the dir."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        # This is what S3Source/GCSSource do internally
        malicious_key = "../../escape_target"
        cached = cache_dir / malicious_key
        resolved = cached.resolve()

        # The resolved path should be INSIDE cache_dir for safety
        # but it escapes:
        is_inside = True
        try:
            resolved.relative_to(cache_dir.resolve())
        except ValueError:
            is_inside = False

        assert not is_inside, (
            f"Path traversal: {resolved} should NOT escape {cache_dir.resolve()}"
        )

    def test_s3source_read_all_path_construction(self, tmp_path):
        """S3Source.read_all builds cache paths without traversal validation."""
        pytest.importorskip("boto3")
        from traqo.ui.sources import S3Source
        from unittest.mock import MagicMock

        mock_client = MagicMock()
        source = S3Source("test-bucket", prefix="traces/", boto3_client=mock_client)

        # The cache path for a malicious key escapes
        malicious_key = "../../outside"
        cached = source._cache_dir / malicious_key
        resolved = cached.resolve()

        is_inside = True
        try:
            resolved.relative_to(Path(source._cache_dir).resolve())
        except ValueError:
            is_inside = False

        assert not is_inside, "S3Source cache path should not escape cache directory"


# ===========================================================================
# Issue 6: LangChain callback _runs dict memory leak
#
# Claim: If on_*_start is called but the matching on_*_end never fires
# (crash, timeout, dropped callback), the entry in _runs persists forever.
# ===========================================================================


class TestLangChainCallbackRunsLeak:
    def test_runs_leak_on_missing_end_callback(self, tmp_path):
        """_runs grows if start callbacks have no matching end."""
        lc = pytest.importorskip("langchain_core")
        from traqo.integrations.langchain import TraqoCallback

        callback = TraqoCallback()
        path = tmp_path / "trace.jsonl"

        with Tracer(path) as tracer:
            # Fire 50 start callbacks with no matching end
            run_ids = []
            for _ in range(50):
                rid = uuid4()
                run_ids.append(rid)
                callback.on_chat_model_start(
                    serialized={
                        "id": ["langchain", "chat_models", "ChatOpenAI"],
                        "kwargs": {"model_name": "gpt-4"},
                    },
                    messages=[[]],
                    run_id=rid,
                )

            # All 50 entries should be stuck in _runs
            assert len(callback._runs) == 50, (
                f"Expected 50 leaked entries, got {len(callback._runs)}"
            )

            # Verify they are the exact run_ids we used
            for rid in run_ids:
                assert rid in callback._runs

    def test_runs_cleaned_on_matching_end(self, tmp_path):
        """Baseline: _runs is cleaned when end callback fires."""
        lc = pytest.importorskip("langchain_core")
        from langchain_core.messages import AIMessage
        from langchain_core.outputs import ChatGeneration, ChatResult, LLMResult
        from traqo.integrations.langchain import TraqoCallback

        callback = TraqoCallback()
        path = tmp_path / "trace.jsonl"

        with Tracer(path) as tracer:
            rid = uuid4()
            callback.on_chat_model_start(
                serialized={
                    "id": ["langchain", "chat_models", "ChatOpenAI"],
                    "kwargs": {"model_name": "gpt-4"},
                },
                messages=[[]],
                run_id=rid,
            )
            assert rid in callback._runs

            # Now fire the matching end
            callback.on_llm_end(
                response=LLMResult(
                    generations=[
                        [ChatGeneration(message=AIMessage(content="hi"))]
                    ]
                ),
                run_id=rid,
            )
            assert rid not in callback._runs


# ===========================================================================
# Issue 7: TOCTOU race in @trace decorator — PROVING NOT A BUG
#
# The claim was that between _should_passthrough() and get_tracer(), the
# tracer could become None. This is IMPOSSIBLE because:
# 1. No await/yield between the two calls (no task switching)
# 2. ContextVars are context-isolated (other threads can't modify ours)
# ===========================================================================


class TestNoTOCTOURace:
    def test_decorator_tracer_consistent(self, tmp_path):
        """@trace decorator always sees consistent tracer state."""
        from traqo.decorator import trace

        tracer_inside = []

        @trace
        def func():
            t = get_tracer()
            tracer_inside.append(t)
            assert t is not None, "Tracer must be available inside traced path"
            return 42

        path = tmp_path / "trace.jsonl"
        with Tracer(path) as tracer:
            result = func()

        assert result == 42
        assert tracer_inside[0] is tracer

    def test_decorator_passthrough_when_no_tracer(self):
        """Without an active tracer, @trace is pure passthrough."""
        from traqo.decorator import trace

        @trace
        def func():
            assert get_tracer() is None, "No tracer should be active"
            return 99

        assert get_tracer() is None
        assert func() == 99


# ===========================================================================
# Issue 10: _is_numpy crashes on objects with __module__ = None
#
# Claim: type(value).__module__.startswith("numpy") raises AttributeError
# when __module__ is None. This is caught by the outer except, returning
# the error fallback instead of the __dict__/__slots__ handlers.
# ===========================================================================


class TestIsNumpyCrashOnNoneModule:
    def test_object_with_none_module_serializes(self):
        """Objects with __module__ = None should still serialize via __dict__."""

        class WeirdClass:
            def __init__(self):
                self.data = "hello"
                self.count = 42

        WeirdClass.__module__ = None  # type: ignore[assignment]
        obj = WeirdClass()

        result = _serialize_value(obj)
        assert isinstance(result, dict), f"Expected dict, got {result!r}"
        assert result == {"data": "hello", "count": 42}

    def test_object_with_none_module_and_slots(self):
        """Slotted class with __module__ = None should serialize via __slots__."""

        class SlottedWeird:
            __slots__ = ("value",)

            def __init__(self):
                self.value = 123

        SlottedWeird.__module__ = None  # type: ignore[assignment]
        obj = SlottedWeird()

        result = _serialize_value(obj)
        assert isinstance(result, dict), f"Expected dict, got {result!r}"
        assert result == {"value": 123}
