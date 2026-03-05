"""Tests for core Tracer class."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.conftest import read_events
from traqo import Tracer, get_current_span, get_tracer, trace


class TestTraceStartEnd:
    def test_trace_start_written_on_enter(self, trace_file: Path):
        with Tracer(path=trace_file):
            events = read_events(trace_file)
            assert len(events) == 1
            assert events[0]["type"] == "trace_start"
            assert "ts" in events[0]
            assert "tracer_version" in events[0]

    def test_trace_end_written_on_exit(self, trace_file: Path):
        with Tracer(path=trace_file):
            pass
        events = read_events(trace_file)
        assert events[-1]["type"] == "trace_end"
        assert "duration_s" in events[-1]
        assert "stats" in events[-1]

    def test_trace_end_stats_accurate(self, trace_file: Path):
        with Tracer(path=trace_file) as tracer:
            tracer.log("evt1", {"a": 1})
            tracer.log("evt2", {"b": 2})
            with tracer.span(
                "llm_call",
                input=[{"role": "user", "content": "hi"}],
                metadata={
                    "model": "test-model",
                    "token_usage": {"input_tokens": 10, "output_tokens": 5},
                },
                kind="llm",
            ) as span:
                span.set_output("hello")
        events = read_events(trace_file)
        stats = events[-1]["stats"]
        assert stats["events"] == 2
        assert stats["spans"] == 1
        assert stats["errors"] == 0
        assert stats["total_input_tokens"] == 10
        assert stats["total_output_tokens"] == 5

    def test_trace_end_reasoning_tokens(self, trace_file: Path):
        with Tracer(path=trace_file) as tracer:
            with tracer.span(
                "reasoning_call",
                metadata={
                    "token_usage": {
                        "input_tokens": 100,
                        "output_tokens": 50,
                        "reasoning_tokens": 30,
                    }
                },
                kind="llm",
            ):
                pass
        events = read_events(trace_file)
        stats = events[-1]["stats"]
        assert stats["total_reasoning_tokens"] == 30
        assert stats["total_input_tokens"] == 100
        assert stats["total_output_tokens"] == 50

    def test_trace_end_cache_tokens(self, trace_file: Path):
        with Tracer(path=trace_file) as tracer:
            with tracer.span(
                "llm_call",
                metadata={
                    "token_usage": {
                        "input_tokens": 200,
                        "output_tokens": 50,
                        "cache_read_tokens": 150,
                        "cache_creation_tokens": 30,
                    }
                },
                kind="llm",
            ):
                pass
        events = read_events(trace_file)
        stats = events[-1]["stats"]
        assert stats["total_cache_read_tokens"] == 150
        assert stats["total_cache_creation_tokens"] == 30
        assert stats["total_input_tokens"] == 200

    def test_record_tokens_all_types(self, trace_file: Path):
        """record_tokens() must accumulate cache and reasoning tokens."""
        with Tracer(path=trace_file) as tracer:
            tracer.record_span()
            tracer.record_tokens(
                input_tokens=100,
                output_tokens=50,
                cache_read_tokens=80,
                cache_creation_tokens=20,
                reasoning_tokens=10,
            )
        events = read_events(trace_file)
        stats = events[-1]["stats"]
        assert stats["total_input_tokens"] == 100
        assert stats["total_output_tokens"] == 50
        assert stats["total_cache_read_tokens"] == 80
        assert stats["total_cache_creation_tokens"] == 20
        assert stats["total_reasoning_tokens"] == 10

    def test_trace_end_token_accumulation(self, trace_file: Path):
        with Tracer(path=trace_file) as tracer:
            with tracer.span(
                "call1",
                metadata={"token_usage": {"input_tokens": 100, "output_tokens": 50}},
                kind="llm",
            ):
                pass
            with tracer.span(
                "call2",
                metadata={"token_usage": {"input_tokens": 200, "output_tokens": 100}},
                kind="llm",
            ):
                pass
        events = read_events(trace_file)
        stats = events[-1]["stats"]
        assert stats["total_input_tokens"] == 300
        assert stats["total_output_tokens"] == 150


class TestMetadata:
    def test_metadata_in_trace_start(self, trace_file: Path):
        with Tracer(path=trace_file, metadata={"run_id": "abc", "model": "gpt-5"}):
            pass
        events = read_events(trace_file)
        assert events[0]["metadata"] == {"run_id": "abc", "model": "gpt-5"}

    def test_metadata_empty_by_default(self, trace_file: Path):
        with Tracer(path=trace_file):
            pass
        events = read_events(trace_file)
        assert "metadata" not in events[0]


class TestLogEvent:
    def test_log_event(self, trace_file: Path):
        with Tracer(path=trace_file) as tracer:
            tracer.log("checkpoint", {"count": 42})
        events = read_events(trace_file)
        evt = [e for e in events if e["type"] == "event"][0]
        assert evt["name"] == "checkpoint"
        assert evt["data"] == {"count": 42}
        assert "id" in evt
        assert "ts" in evt

    def test_log_event_no_data(self, trace_file: Path):
        with Tracer(path=trace_file) as tracer:
            tracer.log("ping")
        events = read_events(trace_file)
        evt = [e for e in events if e["type"] == "event"][0]
        assert evt["name"] == "ping"
        assert "data" not in evt


class TestSpanMetadata:
    """Test span metadata — replaces the old TestLLMEvent class."""

    def test_span_with_llm_metadata(self, trace_file: Path):
        with (
            Tracer(path=trace_file) as tracer,
            tracer.span(
                "chat",
                input=[{"role": "user", "content": "hello"}],
                metadata={
                    "model": "gpt-5-mini",
                    "provider": "openai",
                    "token_usage": {"input_tokens": 10, "output_tokens": 5},
                },
                kind="llm",
            ) as span,
        ):
            span.set_output("hi there")
        events = read_events(trace_file)
        start = [e for e in events if e["type"] == "span_start"][0]
        end = [e for e in events if e["type"] == "span_end"][0]
        assert start["kind"] == "llm"
        assert start["input"] == [{"role": "user", "content": "hello"}]
        assert start["metadata"]["model"] == "gpt-5-mini"
        assert end["output"] == "hi there"
        assert end["metadata"]["token_usage"] == {
            "input_tokens": 10,
            "output_tokens": 5,
        }

    def test_span_metadata_set_during_execution(self, trace_file: Path):
        with Tracer(path=trace_file) as tracer, tracer.span("step") as span:
            span.set_metadata("model", "claude-4")
            span.set_metadata("token_usage", {"input_tokens": 50, "output_tokens": 25})
        events = read_events(trace_file)
        end = [e for e in events if e["type"] == "span_end"][0]
        assert end["metadata"]["model"] == "claude-4"
        assert end["metadata"]["token_usage"]["input_tokens"] == 50

    def test_kind_field_written(self, trace_file: Path):
        with Tracer(path=trace_file) as tracer:
            with tracer.span("retrieval", kind="retriever"):
                pass
        events = read_events(trace_file)
        start = [e for e in events if e["type"] == "span_start"][0]
        end = [e for e in events if e["type"] == "span_end"][0]
        assert start["kind"] == "retriever"
        assert end["kind"] == "retriever"

    def test_kind_omitted_when_none(self, trace_file: Path):
        with Tracer(path=trace_file) as tracer, tracer.span("plain_step"):
            pass
        events = read_events(trace_file)
        start = [e for e in events if e["type"] == "span_start"][0]
        end = [e for e in events if e["type"] == "span_end"][0]
        assert "kind" not in start
        assert "kind" not in end


class TestSpan:
    def test_span_start_end(self, trace_file: Path):
        with Tracer(path=trace_file) as tracer:
            with tracer.span("my_step", input={"key": "val"}):
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
        with Tracer(path=trace_file) as tracer:
            with pytest.raises(ValueError, match="boom"):
                with tracer.span("failing"):
                    raise ValueError("boom")
        events = read_events(trace_file)
        end = [e for e in events if e["type"] == "span_end"][0]
        assert end["status"] == "error"
        assert end["error"]["type"] == "ValueError"
        assert end["error"]["message"] == "boom"

    def test_nested_spans_parent_ids(self, trace_file: Path):
        with Tracer(path=trace_file) as tracer, tracer.span("outer") as outer:
            with tracer.span("inner"):
                pass
        events = read_events(trace_file)
        inner_start = [
            e for e in events if e["type"] == "span_start" and e["name"] == "inner"
        ][0]
        assert inner_start["parent_id"] == outer.id

    def test_span_yields_span_object(self, trace_file: Path):
        with Tracer(path=trace_file) as tracer, tracer.span("test") as span:
            assert hasattr(span, "id")
            assert hasattr(span, "name")
            assert span.name == "test"
            span.set_output("result")
            span.set_metadata("key", "value")
        events = read_events(trace_file)
        end = [e for e in events if e["type"] == "span_end"][0]
        assert end["output"] == "result"
        assert end["metadata"]["key"] == "value"

    def test_span_update_metadata(self, trace_file: Path):
        with Tracer(path=trace_file) as tracer, tracer.span("test") as span:
            span.update_metadata({"a": 1, "b": 2})
            span.set_metadata("c", 3)
        events = read_events(trace_file)
        end = [e for e in events if e["type"] == "span_end"][0]
        assert end["metadata"] == {"a": 1, "b": 2, "c": 3}


class TestGetTracer:
    def test_get_tracer_outside_context(self):
        assert get_tracer() is None

    def test_get_tracer_inside_context(self, trace_file: Path):
        with Tracer(path=trace_file) as tracer:
            assert get_tracer() is tracer

    def test_get_tracer_after_exit(self, trace_file: Path):
        with Tracer(path=trace_file):
            pass
        assert get_tracer() is None


class TestCaptureContent:
    def test_span_input_always_written(self, trace_file: Path):
        """Span input is controlled by the caller, not capture_content."""
        with Tracer(path=trace_file, capture_content=False) as tracer:
            with tracer.span("step", input={"data": "visible"}):
                pass
        events = read_events(trace_file)
        start = [e for e in events if e["type"] == "span_start"][0]
        assert start["input"] == {"data": "visible"}


class TestTraceInputOutput:
    def test_trace_input_in_trace_start(self, trace_file: Path):
        with Tracer(path=trace_file, input={"query": "hello"}):
            pass
        events = read_events(trace_file)
        assert events[0]["input"] == {"query": "hello"}

    def test_trace_output_in_trace_end(self, trace_file: Path):
        with Tracer(path=trace_file) as tracer:
            tracer.set_output({"response": "world"})
        events = read_events(trace_file)
        assert events[-1]["output"] == {"response": "world"}

    def test_trace_no_input_omitted(self, trace_file: Path):
        with Tracer(path=trace_file):
            pass
        events = read_events(trace_file)
        assert "input" not in events[0]

    def test_trace_no_output_omitted(self, trace_file: Path):
        with Tracer(path=trace_file):
            pass
        events = read_events(trace_file)
        assert "output" not in events[-1]


class TestTags:
    def test_trace_tags(self, trace_file: Path):
        with Tracer(path=trace_file, tags=["production", "chatbot"]):
            pass
        events = read_events(trace_file)
        assert events[0]["tags"] == ["production", "chatbot"]

    def test_trace_no_tags_omitted(self, trace_file: Path):
        with Tracer(path=trace_file):
            pass
        events = read_events(trace_file)
        assert "tags" not in events[0]

    def test_span_tags(self, trace_file: Path):
        with Tracer(path=trace_file) as tracer:
            with tracer.span("classify", tags=["llm", "gpt-4o"]):
                pass
        events = read_events(trace_file)
        start = [e for e in events if e["type"] == "span_start"][0]
        end = [e for e in events if e["type"] == "span_end"][0]
        assert start["tags"] == ["llm", "gpt-4o"]
        assert end["tags"] == ["llm", "gpt-4o"]

    def test_span_no_tags_omitted(self, trace_file: Path):
        with Tracer(path=trace_file) as tracer, tracer.span("step"):
            pass
        events = read_events(trace_file)
        start = [e for e in events if e["type"] == "span_start"][0]
        end = [e for e in events if e["type"] == "span_end"][0]
        assert "tags" not in start
        assert "tags" not in end

    def test_span_tags_on_error(self, trace_file: Path):
        with Tracer(path=trace_file) as tracer, pytest.raises(ValueError):
            with tracer.span("failing", tags=["important"]):
                raise ValueError("boom")
        events = read_events(trace_file)
        end = [e for e in events if e["type"] == "span_end"][0]
        assert end["tags"] == ["important"]
        assert end["status"] == "error"

    def test_decorator_tags(self, trace_file: Path):
        @trace(tags=["auth", "v2"])
        def login(user: str) -> bool:
            return True

        with Tracer(path=trace_file):
            login("alice")

        events = read_events(trace_file)
        start = [e for e in events if e["type"] == "span_start"][0]
        assert start["tags"] == ["auth", "v2"]


class TestThreadId:
    def test_thread_id_in_trace_start(self, trace_file: Path):
        with Tracer(path=trace_file, thread_id="conv-123"):
            pass
        events = read_events(trace_file)
        assert events[0]["thread_id"] == "conv-123"

    def test_thread_id_omitted_when_none(self, trace_file: Path):
        with Tracer(path=trace_file):
            pass
        events = read_events(trace_file)
        assert "thread_id" not in events[0]


class TestGetCurrentSpan:
    def test_get_current_span_outside(self):
        assert get_current_span() is None

    def test_get_current_span_inside_span(self, trace_file: Path):
        with Tracer(path=trace_file) as tracer:
            assert get_current_span() is None
            with tracer.span("outer") as outer:
                assert get_current_span() is outer
                with tracer.span("inner") as inner:
                    assert get_current_span() is inner
                assert get_current_span() is outer
            assert get_current_span() is None

    def test_get_current_span_from_decorator(self, trace_file: Path):
        captured_span = None

        @trace()
        def my_fn() -> str:
            nonlocal captured_span
            captured_span = get_current_span()
            captured_span.set_metadata("custom", "value")
            return "done"

        with Tracer(path=trace_file):
            my_fn()

        assert captured_span is not None
        assert captured_span.name == "my_fn"
        events = read_events(trace_file)
        end = [e for e in events if e["type"] == "span_end"][0]
        assert end["metadata"]["custom"] == "value"

    def test_get_current_span_no_tracer(self):
        @trace()
        def my_fn() -> str:
            assert get_current_span() is None
            return "done"

        my_fn()


class TestCreatesDirs:
    def test_creates_parent_dirs(self, tmp_path: Path):
        deep = tmp_path / "a" / "b" / "c" / "trace.jsonl"
        with Tracer(path=deep):
            pass
        # Raw .jsonl is replaced by .jsonl.gz after compression
        gz = deep.parent / (deep.stem + ".jsonl.gz")
        assert gz.exists()


class TestAutoPath:
    def test_name_only_creates_file(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("TRAQO_TRACE_DIR", str(tmp_path))
        with Tracer("my_experiment"):
            pass
        files = list(tmp_path.glob("my_experiment_*.jsonl.gz"))
        assert len(files) == 1

    def test_no_args_creates_file(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("TRAQO_TRACE_DIR", str(tmp_path))
        with Tracer():
            pass
        files = list(tmp_path.glob("*.jsonl.gz"))
        assert len(files) == 1

    def test_name_in_trace_start(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("TRAQO_TRACE_DIR", str(tmp_path))
        with Tracer("my_experiment"):
            pass
        files = list(tmp_path.glob("my_experiment_*.jsonl.gz"))
        events = read_events(files[0])
        assert events[0]["type"] == "trace_start"
        assert events[0]["name"] == "my_experiment"

    def test_auto_path_with_backends_cleans_up(self, tmp_path: Path):
        from traqo.backend import flush_backends

        completed = []

        class TestBackend:
            def on_event(self, event):
                pass

            def on_trace_complete(self, trace_path):
                completed.append(trace_path)
                return None

            def close(self):
                pass

        with Tracer("ephemeral", trace_dir=tmp_path, backends=[TestBackend()]):
            pass

        flush_backends()
        # Raw .jsonl always deleted; auto_path + backends also cleans compressed
        assert list(tmp_path.glob("ephemeral_*.jsonl")) == []
        assert list(tmp_path.glob("ephemeral_*.jsonl.gz")) == []
        assert len(completed) == 1

    def test_explicit_path_not_cleaned_up(self, tmp_path: Path):
        from traqo.backend import flush_backends

        class TestBackend:
            def on_event(self, event):
                pass

            def on_trace_complete(self, trace_path):
                return None

            def close(self):
                pass

        path = tmp_path / "keep_me.jsonl"
        with Tracer(path=path, backends=[TestBackend()]):
            pass

        flush_backends()
        # Raw is deleted but compressed stays for explicit paths
        gz = path.parent / (path.stem + ".jsonl.gz")
        assert gz.exists()

    def test_auto_path_no_backends_not_cleaned_up(self, tmp_path: Path):
        with Tracer("keep_this", trace_dir=tmp_path):
            pass
        # Raw deleted, compressed stays (no backends to clean compressed)
        assert list(tmp_path.glob("keep_this_*.jsonl")) == []
        files = list(tmp_path.glob("keep_this_*.jsonl.gz"))
        assert len(files) == 1

    def test_trace_dir_kwarg(self, tmp_path: Path):
        custom_dir = tmp_path / "custom"
        with Tracer("experiment", trace_dir=custom_dir):
            pass
        files = list(custom_dir.glob("experiment_*.jsonl.gz"))
        assert len(files) == 1
