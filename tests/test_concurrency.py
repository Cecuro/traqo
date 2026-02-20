"""Tests for async concurrency and edge cases."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from unittest.mock import patch

import pytest

from traqo import Tracer, get_tracer, trace
from tests.conftest import read_events


class TestConcurrency:
    async def test_concurrent_tracers_isolated(self, tmp_path: Path):
        path_a = tmp_path / "a.jsonl"
        path_b = tmp_path / "b.jsonl"

        @trace()
        async def work(name: str) -> str:
            return f"done_{name}"

        async def task_a():
            async with Tracer(path_a, metadata={"task": "a"}):
                await work("a")

        async def task_b():
            async with Tracer(path_b, metadata={"task": "b"}):
                await work("b")

        await asyncio.gather(task_a(), task_b())

        events_a = read_events(path_a)
        events_b = read_events(path_b)

        # Each trace has its own events
        a_spans = [e for e in events_a if e["type"] == "span_start"]
        b_spans = [e for e in events_b if e["type"] == "span_start"]
        assert len(a_spans) == 1
        assert len(b_spans) == 1

    async def test_gather_branches_traced(self, trace_file: Path):
        @trace()
        async def branch(name: str) -> str:
            return name

        async with Tracer(trace_file):
            await asyncio.gather(
                branch("a"),
                branch("b"),
                branch("c"),
            )

        events = read_events(trace_file)
        starts = [e for e in events if e["type"] == "span_start"]
        assert len(starts) == 3


class TestEdgeCases:
    def test_tracer_outside_context_manager_still_writes(self, trace_file: Path):
        """Tracer used without 'with' still writes if opened manually."""
        tracer = Tracer(trace_file)
        tracer._open()
        tracer._start_time = __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        )
        tracer.log("manual", {"key": "val"})
        tracer._close()

        events = read_events(trace_file)
        assert len(events) == 1
        assert events[0]["type"] == "event"

    def test_write_failure_does_not_crash(self, trace_file: Path, caplog):
        """If writing fails, it logs a warning but doesn't crash."""
        with Tracer(trace_file) as tracer:
            # Close the file to force a write error
            tracer._file.close()
            with caplog.at_level(logging.WARNING):
                tracer.log("should_fail")
            # No exception raised
