"""Tests for child tracers."""

from __future__ import annotations

from pathlib import Path

from tests.conftest import read_events
from traqo import Tracer, get_tracer


class TestChildTracer:
    def test_child_creates_separate_file(self, tmp_path: Path):
        parent_path = tmp_path / "parent.jsonl"
        child_path = tmp_path / "child.jsonl"

        with Tracer(path=parent_path) as parent:
            child = parent.child("agent_a", path=child_path)
            with child:
                child.log("from_child", {"x": 1})

        child_gz = child_path.parent / (child_path.stem + ".jsonl.gz")
        assert child_gz.exists()
        child_events = read_events(child_path)
        evt = [e for e in child_events if e["type"] == "event"]
        assert len(evt) == 1
        assert evt[0]["name"] == "from_child"

    def test_child_has_parent_trace_in_header(self, tmp_path: Path):
        parent_path = tmp_path / "parent.jsonl"
        child_path = tmp_path / "child.jsonl"

        with Tracer(path=parent_path) as parent:
            child = parent.child("agent_a", path=child_path)
            with child:
                pass

        child_events = read_events(child_path)
        assert child_events[0]["type"] == "trace_start"
        assert child_events[0]["metadata"]["parent_trace"] == str(parent_path)

    def test_child_writes_events_to_parent(self, tmp_path: Path):
        parent_path = tmp_path / "parent.jsonl"
        child_path = tmp_path / "child.jsonl"

        with Tracer(path=parent_path) as parent:
            child = parent.child("agent_a", path=child_path)
            with child:
                pass

        parent_events = read_events(parent_path)
        names = [e.get("name") for e in parent_events if e["type"] == "event"]
        assert "child_started" in names
        assert "child_ended" in names

    def test_child_ended_has_summary_stats(self, tmp_path: Path):
        parent_path = tmp_path / "parent.jsonl"
        child_path = tmp_path / "child.jsonl"

        with Tracer(path=parent_path) as parent:
            child = parent.child("agent_a", path=child_path)
            with child:
                with child.span(
                    "call1",
                    metadata={"token_usage": {"input_tokens": 10, "output_tokens": 5}},
                    kind="llm",
                ):
                    pass
                with child.span(
                    "call2",
                    metadata={"token_usage": {"input_tokens": 20, "output_tokens": 10}},
                    kind="llm",
                ):
                    pass

        parent_events = read_events(parent_path)
        ended = [
            e
            for e in parent_events
            if e["type"] == "event" and e.get("name") == "child_ended"
        ][0]
        assert ended["data"]["spans"] == 2
        assert ended["data"]["total_input_tokens"] == 30
        assert ended["data"]["total_output_tokens"] == 15
        assert "duration_s" in ended["data"]

    def test_child_cache_tokens_rolled_up(self, tmp_path: Path):
        """Cache and reasoning tokens from child must appear in parent stats and events."""
        parent_path = tmp_path / "parent.jsonl"
        child_path = tmp_path / "child.jsonl"

        with Tracer(path=parent_path) as parent:
            child = parent.child("agent_a", path=child_path)
            with child:
                with child.span(
                    "call1",
                    metadata={
                        "token_usage": {
                            "input_tokens": 100,
                            "output_tokens": 50,
                            "cache_read_tokens": 80,
                            "cache_creation_tokens": 10,
                            "reasoning_tokens": 5,
                        }
                    },
                    kind="llm",
                ):
                    pass

        parent_events = read_events(parent_path)
        ended = [
            e
            for e in parent_events
            if e["type"] == "event" and e.get("name") == "child_ended"
        ][0]
        assert ended["data"]["total_cache_read_tokens"] == 80
        assert ended["data"]["total_cache_creation_tokens"] == 10
        assert ended["data"]["total_reasoning_tokens"] == 5

        trace_end = [e for e in parent_events if e["type"] == "trace_end"][0]
        stats = trace_end["stats"]
        assert stats["total_cache_read_tokens"] == 80
        assert stats["total_cache_creation_tokens"] == 10
        assert stats["total_reasoning_tokens"] == 5

    def test_child_default_path(self, tmp_path: Path):
        parent_path = tmp_path / "traces" / "parent.jsonl"

        with Tracer(path=parent_path) as parent:
            child = parent.child("my_agent")
            with child:
                child.log("hi")

        files = list((tmp_path / "traces").glob("my_agent_*.jsonl.gz"))
        assert len(files) == 1

    def test_child_context_switches(self, tmp_path: Path):
        parent_path = tmp_path / "parent.jsonl"
        child_path = tmp_path / "child.jsonl"

        with Tracer(path=parent_path) as parent:
            assert get_tracer() is parent
            child = parent.child("agent_a", path=child_path)
            assert get_tracer() is parent
            with child:
                assert get_tracer() is child
            assert get_tracer() is parent

    def test_child_listed_in_parent_trace_end(self, tmp_path: Path):
        parent_path = tmp_path / "parent.jsonl"
        child_path = tmp_path / "child.jsonl"

        with Tracer(path=parent_path) as parent:
            child = parent.child("agent_a", path=child_path)
            with child, child.span("work", kind="llm"):
                pass

        parent_events = read_events(parent_path)
        trace_end = parent_events[-1]
        assert trace_end["type"] == "trace_end"
        assert len(trace_end["children"]) == 1
        assert trace_end["children"][0]["name"] == "agent_a"
        assert trace_end["children"][0]["file"] == "child.jsonl.gz"
        assert "path" not in trace_end["children"][0]

    def test_capture_content_inherited_by_child(self, tmp_path: Path):
        parent_path = tmp_path / "parent.jsonl"
        child_path = tmp_path / "child.jsonl"

        with Tracer(path=parent_path, capture_content=False) as parent:
            child = parent.child("agent_a", path=child_path)
            with child:
                assert child._capture_content is False
