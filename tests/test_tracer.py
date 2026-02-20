"""Tests for core Tracer class."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from traqo import Tracer, get_tracer
from tests.conftest import read_events


class TestTraceStartEnd:
    def test_trace_start_written_on_enter(self, trace_file: Path):
        with Tracer(trace_file):
            events = read_events(trace_file)
            assert len(events) == 1
            assert events[0]["type"] == "trace_start"
            assert "ts" in events[0]
            assert "tracer_version" in events[0]

    def test_trace_end_written_on_exit(self, trace_file: Path):
        with Tracer(trace_file):
            pass
        events = read_events(trace_file)
        assert events[-1]["type"] == "trace_end"
        assert "duration_s" in events[-1]
        assert "stats" in events[-1]

    def test_trace_end_stats_accurate(self, trace_file: Path):
        with Tracer(trace_file) as tracer:
            tracer.log("evt1", {"a": 1})
            tracer.log("evt2", {"b": 2})
            tracer.llm_event(
                model="test-model",
                input_messages=[{"role": "user", "content": "hi"}],
                output_text="hello",
                token_usage={"input_tokens": 10, "output_tokens": 5},
            )
        events = read_events(trace_file)
        stats = events[-1]["stats"]
        assert stats["events"] == 2
        assert stats["llm_calls"] == 1
        assert stats["spans"] == 0
        assert stats["errors"] == 0

    def test_trace_end_token_accumulation(self, trace_file: Path):
        with Tracer(trace_file) as tracer:
            tracer.llm_event(
                model="m1",
                token_usage={"input_tokens": 100, "output_tokens": 50},
            )
            tracer.llm_event(
                model="m2",
                token_usage={"input_tokens": 200, "output_tokens": 100},
            )
        events = read_events(trace_file)
        stats = events[-1]["stats"]
        assert stats["total_input_tokens"] == 300
        assert stats["total_output_tokens"] == 150


class TestMetadata:
    def test_metadata_in_trace_start(self, trace_file: Path):
        with Tracer(trace_file, metadata={"run_id": "abc", "model": "gpt-5"}):
            pass
        events = read_events(trace_file)
        assert events[0]["metadata"] == {"run_id": "abc", "model": "gpt-5"}

    def test_metadata_empty_by_default(self, trace_file: Path):
        with Tracer(trace_file):
            pass
        events = read_events(trace_file)
        assert "metadata" not in events[0]


class TestLogEvent:
    def test_log_event(self, trace_file: Path):
        with Tracer(trace_file) as tracer:
            tracer.log("checkpoint", {"count": 42})
        events = read_events(trace_file)
        evt = [e for e in events if e["type"] == "event"][0]
        assert evt["name"] == "checkpoint"
        assert evt["data"] == {"count": 42}
        assert "id" in evt
        assert "ts" in evt

    def test_log_event_no_data(self, trace_file: Path):
        with Tracer(trace_file) as tracer:
            tracer.log("ping")
        events = read_events(trace_file)
        evt = [e for e in events if e["type"] == "event"][0]
        assert evt["name"] == "ping"
        assert "data" not in evt


class TestLLMEvent:
    def test_llm_event(self, trace_file: Path):
        with Tracer(trace_file) as tracer:
            tracer.llm_event(
                model="gpt-5-mini",
                input_messages=[{"role": "user", "content": "hello"}],
                output_text="hi there",
                token_usage={"input_tokens": 10, "output_tokens": 5},
                duration_s=1.5,
                operation="greet",
            )
        events = read_events(trace_file)
        llm = [e for e in events if e["type"] == "llm_call"][0]
        assert llm["model"] == "gpt-5-mini"
        assert llm["input"] == [{"role": "user", "content": "hello"}]
        assert llm["output"] == "hi there"
        assert llm["token_usage"] == {"input_tokens": 10, "output_tokens": 5}
        assert llm["duration_s"] == 1.5
        assert llm["operation"] == "greet"

    def test_llm_event_within_span(self, trace_file: Path):
        with Tracer(trace_file) as tracer:
            with tracer.span("outer"):
                tracer.llm_event(model="m1")
        events = read_events(trace_file)
        llm = [e for e in events if e["type"] == "llm_call"][0]
        span_start = [e for e in events if e["type"] == "span_start"][0]
        assert llm["parent_id"] == span_start["id"]


class TestSpan:
    def test_span_start_end(self, trace_file: Path):
        with Tracer(trace_file) as tracer:
            with tracer.span("my_step", {"key": "val"}):
                pass
        events = read_events(trace_file)
        starts = [e for e in events if e["type"] == "span_start"]
        ends = [e for e in events if e["type"] == "span_end"]
        assert len(starts) == 1
        assert len(ends) == 1
        assert starts[0]["name"] == "my_step"
        assert starts[0]["input"] == {"key": "val"}
        assert ends[0]["status"] == "ok"
        assert ends[0]["id"] == starts[0]["id"]

    def test_span_error(self, trace_file: Path):
        with Tracer(trace_file) as tracer:
            with pytest.raises(ValueError, match="boom"):
                with tracer.span("failing"):
                    raise ValueError("boom")
        events = read_events(trace_file)
        end = [e for e in events if e["type"] == "span_end"][0]
        assert end["status"] == "error"
        assert end["error"]["type"] == "ValueError"
        assert end["error"]["message"] == "boom"

    def test_nested_spans_parent_ids(self, trace_file: Path):
        with Tracer(trace_file) as tracer:
            with tracer.span("outer") as outer_id:
                with tracer.span("inner") as inner_id:
                    pass
        events = read_events(trace_file)
        inner_start = [e for e in events if e["type"] == "span_start" and e["name"] == "inner"][0]
        assert inner_start["parent_id"] == outer_id


class TestGetTracer:
    def test_get_tracer_outside_context(self):
        assert get_tracer() is None

    def test_get_tracer_inside_context(self, trace_file: Path):
        with Tracer(trace_file) as tracer:
            assert get_tracer() is tracer

    def test_get_tracer_after_exit(self, trace_file: Path):
        with Tracer(trace_file):
            pass
        assert get_tracer() is None


class TestCaptureContent:
    def test_capture_content_true_includes_input_output(self, trace_file: Path):
        with Tracer(trace_file, capture_content=True) as tracer:
            tracer.llm_event(
                model="m1",
                input_messages=[{"role": "user", "content": "secret"}],
                output_text="response",
            )
        events = read_events(trace_file)
        llm = [e for e in events if e["type"] == "llm_call"][0]
        assert "input" in llm
        assert "output" in llm

    def test_capture_content_false_omits_input_output(self, trace_file: Path):
        with Tracer(trace_file, capture_content=False) as tracer:
            tracer.llm_event(
                model="m1",
                input_messages=[{"role": "user", "content": "secret"}],
                output_text="response",
                token_usage={"input_tokens": 10, "output_tokens": 5},
            )
        events = read_events(trace_file)
        llm = [e for e in events if e["type"] == "llm_call"][0]
        assert "input" not in llm
        assert "output" not in llm
        assert llm["model"] == "m1"
        assert llm["token_usage"]["input_tokens"] == 10

    def test_capture_content_does_not_affect_spans(self, trace_file: Path):
        with Tracer(trace_file, capture_content=False) as tracer:
            with tracer.span("step", {"data": "visible"}):
                pass
        events = read_events(trace_file)
        start = [e for e in events if e["type"] == "span_start"][0]
        assert start["input"] == {"data": "visible"}


class TestCreatesDirs:
    def test_creates_parent_dirs(self, tmp_path: Path):
        deep = tmp_path / "a" / "b" / "c" / "trace.jsonl"
        with Tracer(deep):
            pass
        assert deep.exists()
