"""Tests for traqo improvements: subtrace, child metadata, stats rollup, trace_end error, reader."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.conftest import read_events
from traqo import Tracer, get_tracer, subtrace
from traqo.reader import LLMSpan, aggregate_tokens, iter_llm_spans

# ---------------------------------------------------------------------------
# 1. child() metadata kwarg
# ---------------------------------------------------------------------------


class TestChildMetadata:
    def test_child_merges_metadata(self, tmp_path: Path):
        parent_path = tmp_path / "parent.jsonl"
        child_path = tmp_path / "child.jsonl"

        with Tracer(path=parent_path) as parent:
            child = parent.child(
                "agent_a",
                path=child_path,
                metadata={"agent_id": "agent_a", "phase": "validation"},
            )
            with child:
                pass

        child_events = read_events(child_path)
        meta = child_events[0]["metadata"]
        assert meta["parent_trace"] == str(parent_path)
        assert meta["agent_id"] == "agent_a"
        assert meta["phase"] == "validation"

    def test_child_metadata_none_keeps_parent_trace(self, tmp_path: Path):
        parent_path = tmp_path / "parent.jsonl"
        child_path = tmp_path / "child.jsonl"

        with Tracer(path=parent_path) as parent:
            child = parent.child("agent_a", path=child_path, metadata=None)
            with child:
                pass

        child_events = read_events(child_path)
        meta = child_events[0]["metadata"]
        assert meta["parent_trace"] == str(parent_path)

    def test_child_metadata_cannot_clobber_parent_trace(self, tmp_path: Path):
        """User metadata is merged after parent_trace, so it CAN override it.
        This is intentional — the user knows best."""
        parent_path = tmp_path / "parent.jsonl"
        child_path = tmp_path / "child.jsonl"

        with Tracer(path=parent_path) as parent:
            child = parent.child(
                "agent_a", path=child_path, metadata={"parent_trace": "custom"}
            )
            with child:
                pass

        child_events = read_events(child_path)
        assert child_events[0]["metadata"]["parent_trace"] == "custom"


# ---------------------------------------------------------------------------
# 2. subtrace()
# ---------------------------------------------------------------------------


class TestSubtrace:
    def test_subtrace_with_parent_creates_child(self, tmp_path: Path):
        parent_path = tmp_path / "parent.jsonl"

        with Tracer(path=parent_path):
            child_tracer = subtrace("agent_a")
            with child_tracer:
                assert get_tracer() is child_tracer

        # Child file should exist at parent_dir/agent_a_*.jsonl
        child_files = list(tmp_path.glob("agent_a_*.jsonl"))
        assert len(child_files) == 1

        # Parent should have child events
        parent_events = read_events(parent_path)
        names = [e.get("name") for e in parent_events if e["type"] == "event"]
        assert "child_started" in names
        assert "child_ended" in names

    def test_subtrace_without_parent_creates_root(self, tmp_path: Path):
        path = tmp_path / "standalone.jsonl"

        with subtrace("my_agent", path):
            tracer = get_tracer()
            assert tracer is not None
            tracer.log("hello")

        events = read_events(path)
        assert events[0]["type"] == "trace_start"
        assert events[-1]["type"] == "trace_end"

    def test_subtrace_without_parent_auto_generates(self, tmp_path: Path):
        with subtrace("agent", trace_dir=tmp_path):
            tracer = get_tracer()
            assert tracer is not None
        files = list(tmp_path.glob("agent_*.jsonl"))
        assert len(files) == 1

    def test_subtrace_passes_metadata_to_child(self, tmp_path: Path):
        parent_path = tmp_path / "parent.jsonl"

        with Tracer(path=parent_path):
            with subtrace("agent_a", metadata={"custom": "value"}):
                pass

        child_files = list(tmp_path.glob("agent_a_*.jsonl"))
        assert len(child_files) == 1
        child_events = read_events(child_files[0])
        meta = child_events[0]["metadata"]
        assert meta["parent_trace"] == str(parent_path)
        assert meta["custom"] == "value"

    def test_subtrace_passes_metadata_to_root(self, tmp_path: Path):
        path = tmp_path / "root.jsonl"

        with subtrace("agent", path, metadata={"job_id": "123"}):
            pass

        events = read_events(path)
        assert events[0]["metadata"]["job_id"] == "123"

    def test_subtrace_passes_backends_to_root(self, tmp_path: Path):
        path = tmp_path / "root.jsonl"
        completed = []

        class TestBackend:
            def on_event(self, event):
                pass

            def on_trace_complete(self, trace_path):
                completed.append(trace_path)

            def close(self):
                pass

        with subtrace("agent", path=path, backends=[TestBackend()]):
            pass

        assert len(completed) == 1

    def test_subtrace_disabled_no_error_without_path(self):
        import traqo

        traqo.disable()
        try:
            # Should not raise even without path when disabled
            with subtrace("agent"):
                pass
        finally:
            traqo.enable()


# ---------------------------------------------------------------------------
# 2b. child_file field in child events
# ---------------------------------------------------------------------------


class TestChildFileField:
    def test_child_started_has_child_file(self, tmp_path: Path):
        parent_path = tmp_path / "parent.jsonl"
        child_path = tmp_path / "my_agent.jsonl"

        with Tracer(path=parent_path) as parent:
            child = parent.child("my_agent", path=child_path)
            with child:
                pass

        parent_events = read_events(parent_path)
        started = [e for e in parent_events if e.get("name") == "child_started"][0]
        assert started["data"]["child_file"] == "my_agent.jsonl"

    def test_child_ended_has_child_file(self, tmp_path: Path):
        parent_path = tmp_path / "parent.jsonl"
        child_path = tmp_path / "my_agent.jsonl"

        with Tracer(path=parent_path) as parent:
            child = parent.child("my_agent", path=child_path)
            with child:
                pass

        parent_events = read_events(parent_path)
        ended = [e for e in parent_events if e.get("name") == "child_ended"][0]
        assert ended["data"]["child_file"] == "my_agent.jsonl"

    def test_trace_end_children_has_file(self, tmp_path: Path):
        parent_path = tmp_path / "parent.jsonl"
        child_path = tmp_path / "my_agent.jsonl"

        with Tracer(path=parent_path) as parent:
            child = parent.child("my_agent", path=child_path)
            with child:
                pass

        parent_events = read_events(parent_path)
        trace_end = parent_events[-1]
        assert trace_end["type"] == "trace_end"
        assert trace_end["children"][0]["file"] == "my_agent.jsonl"

    def test_child_file_with_default_path(self, tmp_path: Path):
        parent_path = tmp_path / "traces" / "parent.jsonl"

        with Tracer(path=parent_path) as parent:
            child = parent.child("agent_x")
            with child:
                pass

        parent_events = read_events(parent_path)
        started = [e for e in parent_events if e.get("name") == "child_started"][0]
        assert started["data"]["child_file"].startswith("agent_x_")

        ended = [e for e in parent_events if e.get("name") == "child_ended"][0]
        assert ended["data"]["child_file"].startswith("agent_x_")


# ---------------------------------------------------------------------------
# 3. Child stats roll up into parent
# ---------------------------------------------------------------------------


class TestChildStatsRollup:
    def test_parent_stats_include_child_tokens(self, tmp_path: Path):
        parent_path = tmp_path / "parent.jsonl"

        with Tracer(path=parent_path) as parent:
            # Parent's own span
            with parent.span(
                "parent_call",
                metadata={"token_usage": {"input_tokens": 100, "output_tokens": 50}},
                kind="llm",
            ):
                pass

            # Child with its own spans
            child = parent.child("agent_a")
            with (
                child,
                child.span(
                    "child_call",
                    metadata={
                        "token_usage": {"input_tokens": 200, "output_tokens": 100}
                    },
                    kind="llm",
                ),
            ):
                pass

        parent_events = read_events(parent_path)
        trace_end = parent_events[-1]
        assert trace_end["type"] == "trace_end"
        # Parent (100+200=300 input, 50+100=150 output)
        assert trace_end["stats"]["total_input_tokens"] == 300
        assert trace_end["stats"]["total_output_tokens"] == 150
        # Parent span + child span = 2
        assert trace_end["stats"]["spans"] == 2

    def test_parent_stats_include_child_errors(self, tmp_path: Path):
        parent_path = tmp_path / "parent.jsonl"

        with Tracer(path=parent_path) as parent:
            child = parent.child("agent_a")
            with child:
                try:
                    with child.span("fail", kind="tool"):
                        raise RuntimeError("boom")
                except RuntimeError:
                    pass

        parent_events = read_events(parent_path)
        trace_end = parent_events[-1]
        assert trace_end["stats"]["errors"] == 1
        assert trace_end["stats"]["spans"] == 1

    def test_multiple_children_roll_up(self, tmp_path: Path):
        parent_path = tmp_path / "parent.jsonl"

        with Tracer(path=parent_path) as parent:
            for i in range(3):
                child = parent.child(f"agent_{i}")
                with (
                    child,
                    child.span(
                        f"call_{i}",
                        metadata={
                            "token_usage": {"input_tokens": 10, "output_tokens": 5}
                        },
                        kind="llm",
                    ),
                ):
                    pass

        parent_events = read_events(parent_path)
        trace_end = parent_events[-1]
        assert trace_end["stats"]["total_input_tokens"] == 30
        assert trace_end["stats"]["total_output_tokens"] == 15
        assert trace_end["stats"]["spans"] == 3


# ---------------------------------------------------------------------------
# 4. trace_end error/status on exception
# ---------------------------------------------------------------------------


class TestTraceEndError:
    def test_trace_end_ok_on_normal_exit(self, tmp_path: Path):
        path = tmp_path / "trace.jsonl"

        with Tracer(path=path):
            pass

        events = read_events(path)
        trace_end = events[-1]
        assert trace_end["status"] == "ok"
        assert "error" not in trace_end

    def test_trace_end_error_on_exception(self, tmp_path: Path):
        path = tmp_path / "trace.jsonl"

        with pytest.raises(ValueError, match="test error"), Tracer(path=path):
            raise ValueError("test error")

        events = read_events(path)
        trace_end = events[-1]
        assert trace_end["status"] == "error"
        assert trace_end["error"]["type"] == "ValueError"
        assert "test error" in trace_end["error"]["message"]

    def test_trace_end_error_includes_traceback(self, tmp_path: Path):
        path = tmp_path / "trace.jsonl"

        with pytest.raises(RuntimeError), Tracer(path=path):
            raise RuntimeError("crash")

        events = read_events(path)
        trace_end = events[-1]
        assert "traceback" in trace_end["error"]
        assert "RuntimeError" in trace_end["error"]["traceback"]


# ---------------------------------------------------------------------------
# 5. iter_llm_spans() and aggregate_tokens() reader
# ---------------------------------------------------------------------------


def _write_trace_with_llm_spans(path: Path) -> None:
    """Write a realistic trace file with LLM spans."""
    events = [
        {
            "type": "trace_start",
            "ts": "2025-01-01T00:00:00+00:00",
            "tracer_version": "0.2.0",
        },
        {
            "type": "span_start",
            "id": "span1",
            "parent_id": None,
            "name": "gpt-4o",
            "ts": "2025-01-01T00:00:01+00:00",
            "kind": "llm",
        },
        {
            "type": "span_end",
            "id": "span1",
            "parent_id": None,
            "name": "gpt-4o",
            "ts": "2025-01-01T00:00:02+00:00",
            "duration_s": 1.0,
            "status": "ok",
            "kind": "llm",
            "metadata": {
                "model": "gpt-4o",
                "token_usage": {
                    "input_tokens": 1000,
                    "output_tokens": 200,
                    "reasoning_tokens": 50,
                    "cache_read_tokens": 100,
                },
            },
        },
        {
            "type": "span_start",
            "id": "span2",
            "parent_id": None,
            "name": "read_file",
            "ts": "2025-01-01T00:00:03+00:00",
            "kind": "tool",
        },
        {
            "type": "span_end",
            "id": "span2",
            "parent_id": None,
            "name": "read_file",
            "ts": "2025-01-01T00:00:03+00:00",
            "duration_s": 0.1,
            "status": "ok",
            "kind": "tool",
        },
        {
            "type": "span_start",
            "id": "span3",
            "parent_id": None,
            "name": "claude-3.5-sonnet",
            "ts": "2025-01-01T00:00:04+00:00",
            "kind": "llm",
        },
        {
            "type": "span_end",
            "id": "span3",
            "parent_id": None,
            "name": "claude-3.5-sonnet",
            "ts": "2025-01-01T00:00:06+00:00",
            "duration_s": 2.0,
            "status": "ok",
            "kind": "llm",
            "metadata": {
                "model": "claude-3.5-sonnet",
                "token_usage": {"input_tokens": 500, "output_tokens": 100},
            },
        },
        {
            "type": "trace_end",
            "ts": "2025-01-01T00:00:07+00:00",
            "duration_s": 7.0,
            "status": "ok",
            "stats": {
                "spans": 3,
                "events": 0,
                "total_input_tokens": 1500,
                "total_output_tokens": 300,
                "errors": 0,
            },
        },
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for event in events:
            f.write(json.dumps(event) + "\n")


class TestIterLLMSpans:
    def test_yields_only_llm_spans(self, tmp_path: Path):
        path = tmp_path / "trace.jsonl"
        _write_trace_with_llm_spans(path)

        spans = list(iter_llm_spans(path))
        assert len(spans) == 2
        assert all(isinstance(s, LLMSpan) for s in spans)

    def test_span_fields(self, tmp_path: Path):
        path = tmp_path / "trace.jsonl"
        _write_trace_with_llm_spans(path)

        spans = list(iter_llm_spans(path))
        gpt = spans[0]
        assert gpt.model == "gpt-4o"
        assert gpt.input_tokens == 1000
        assert gpt.output_tokens == 200
        assert gpt.reasoning_tokens == 50
        assert gpt.cache_read_tokens == 100
        assert gpt.cache_creation_tokens == 0
        assert gpt.duration_s == 1.0
        assert gpt.status == "ok"
        assert gpt.name == "gpt-4o"
        assert gpt.span_id == "span1"

        claude = spans[1]
        assert claude.model == "claude-3.5-sonnet"
        assert claude.input_tokens == 500
        assert claude.output_tokens == 100

    def test_skips_malformed_lines(self, tmp_path: Path):
        path = tmp_path / "trace.jsonl"
        with open(path, "w") as f:
            f.write("not json\n")
            f.write("\n")
            f.write(
                json.dumps(
                    {
                        "type": "span_end",
                        "id": "s1",
                        "kind": "llm",
                        "name": "model",
                        "duration_s": 1.0,
                        "status": "ok",
                        "metadata": {
                            "model": "test",
                            "token_usage": {"input_tokens": 10, "output_tokens": 5},
                        },
                    }
                )
                + "\n"
            )

        spans = list(iter_llm_spans(path))
        assert len(spans) == 1
        assert spans[0].input_tokens == 10

    def test_skips_llm_spans_without_token_usage(self, tmp_path: Path):
        path = tmp_path / "trace.jsonl"
        with open(path, "w") as f:
            f.write(
                json.dumps(
                    {
                        "type": "span_end",
                        "id": "s1",
                        "kind": "llm",
                        "name": "model",
                        "duration_s": 1.0,
                        "status": "ok",
                        "metadata": {"model": "test"},
                    }
                )
                + "\n"
            )

        spans = list(iter_llm_spans(path))
        assert len(spans) == 0


class TestAggregateTokens:
    def test_aggregates_by_model(self, tmp_path: Path):
        path = tmp_path / "trace.jsonl"
        _write_trace_with_llm_spans(path)

        result = aggregate_tokens(path)
        assert result == {
            "gpt-4o": {"input": 1000, "output": 200},
            "claude-3.5-sonnet": {"input": 500, "output": 100},
        }

    def test_empty_file(self, tmp_path: Path):
        path = tmp_path / "empty.jsonl"
        path.write_text("")

        result = aggregate_tokens(path)
        assert result == {}

    def test_integration_with_real_tracer(self, tmp_path: Path):
        """Test reader against actual Tracer output."""
        path = tmp_path / "trace.jsonl"

        with Tracer(path=path):
            t = get_tracer()
            assert t is not None
            with t.span(
                "call",
                kind="llm",
                metadata={
                    "model": "gpt-4o",
                    "token_usage": {"input_tokens": 100, "output_tokens": 50},
                },
            ):
                pass

        spans = list(iter_llm_spans(path))
        assert len(spans) == 1
        assert spans[0].model == "gpt-4o"
        assert spans[0].input_tokens == 100
        assert spans[0].output_tokens == 50

        agg = aggregate_tokens(path)
        assert agg == {"gpt-4o": {"input": 100, "output": 50}}
