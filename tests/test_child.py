"""Tests for child tracers."""

from __future__ import annotations

from pathlib import Path

import pytest

from traqo import Tracer, get_tracer
from tests.conftest import read_events


class TestChildTracer:
    def test_child_creates_separate_file(self, tmp_path: Path):
        parent_path = tmp_path / "parent.jsonl"
        child_path = tmp_path / "child.jsonl"

        with Tracer(parent_path) as parent:
            child = parent.child("agent_a", child_path)
            with child:
                child.log("from_child", {"x": 1})

        assert child_path.exists()
        child_events = read_events(child_path)
        evt = [e for e in child_events if e["type"] == "event"]
        assert len(evt) == 1
        assert evt[0]["name"] == "from_child"

    def test_child_has_parent_trace_in_header(self, tmp_path: Path):
        parent_path = tmp_path / "parent.jsonl"
        child_path = tmp_path / "child.jsonl"

        with Tracer(parent_path) as parent:
            child = parent.child("agent_a", child_path)
            with child:
                pass

        child_events = read_events(child_path)
        assert child_events[0]["type"] == "trace_start"
        assert child_events[0]["metadata"]["parent_trace"] == str(parent_path)

    def test_child_writes_events_to_parent(self, tmp_path: Path):
        parent_path = tmp_path / "parent.jsonl"
        child_path = tmp_path / "child.jsonl"

        with Tracer(parent_path) as parent:
            child = parent.child("agent_a", child_path)
            with child:
                pass

        parent_events = read_events(parent_path)
        names = [e.get("name") for e in parent_events if e["type"] == "event"]
        assert "child_started" in names
        assert "child_ended" in names

    def test_child_ended_has_summary_stats(self, tmp_path: Path):
        parent_path = tmp_path / "parent.jsonl"
        child_path = tmp_path / "child.jsonl"

        with Tracer(parent_path) as parent:
            child = parent.child("agent_a", child_path)
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
        ended = [e for e in parent_events if e["type"] == "event" and e.get("name") == "child_ended"][0]
        assert ended["data"]["spans"] == 2
        assert ended["data"]["total_input_tokens"] == 30
        assert ended["data"]["total_output_tokens"] == 15
        assert "duration_s" in ended["data"]

    def test_child_default_path(self, tmp_path: Path):
        parent_path = tmp_path / "traces" / "parent.jsonl"

        with Tracer(parent_path) as parent:
            child = parent.child("my_agent")
            with child:
                child.log("hi")

        expected_path = tmp_path / "traces" / "my_agent.jsonl"
        assert expected_path.exists()

    def test_child_context_switches(self, tmp_path: Path):
        parent_path = tmp_path / "parent.jsonl"
        child_path = tmp_path / "child.jsonl"

        with Tracer(parent_path) as parent:
            assert get_tracer() is parent
            child = parent.child("agent_a", child_path)
            assert get_tracer() is parent
            with child:
                assert get_tracer() is child
            assert get_tracer() is parent

    def test_child_listed_in_parent_trace_end(self, tmp_path: Path):
        parent_path = tmp_path / "parent.jsonl"
        child_path = tmp_path / "child.jsonl"

        with Tracer(parent_path) as parent:
            child = parent.child("agent_a", child_path)
            with child:
                with child.span("work", kind="llm"):
                    pass

        parent_events = read_events(parent_path)
        trace_end = parent_events[-1]
        assert trace_end["type"] == "trace_end"
        assert len(trace_end["children"]) == 1
        assert trace_end["children"][0]["name"] == "agent_a"
        assert trace_end["children"][0]["path"] == str(child_path)

    def test_capture_content_inherited_by_child(self, tmp_path: Path):
        parent_path = tmp_path / "parent.jsonl"
        child_path = tmp_path / "child.jsonl"

        with Tracer(parent_path, capture_content=False) as parent:
            child = parent.child("agent_a", child_path)
            with child:
                assert child._capture_content is False
