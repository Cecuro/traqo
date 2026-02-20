"""Tests for @trace decorator."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from traqo import Tracer, trace
from tests.conftest import read_events


class TestSyncDecorator:
    def test_sync_function(self, trace_file: Path):
        @trace()
        def add(a: int, b: int) -> int:
            return a + b

        with Tracer(trace_file):
            result = add(1, 2)

        assert result == 3
        events = read_events(trace_file)
        starts = [e for e in events if e["type"] == "span_start"]
        ends = [e for e in events if e["type"] == "span_end"]
        assert len(starts) == 1
        assert starts[0]["name"] == "add"
        assert starts[0]["input"] == {"a": 1, "b": 2}
        assert len(ends) == 1
        assert ends[0]["status"] == "ok"
        assert ends[0]["output"] == 3

    def test_sync_error(self, trace_file: Path):
        @trace()
        def fail():
            raise RuntimeError("oops")

        with Tracer(trace_file):
            with pytest.raises(RuntimeError, match="oops"):
                fail()

        events = read_events(trace_file)
        end = [e for e in events if e["type"] == "span_end"][0]
        assert end["status"] == "error"
        assert end["error"]["type"] == "RuntimeError"

    def test_custom_name(self, trace_file: Path):
        @trace("my_custom_name")
        def do_stuff():
            return 42

        with Tracer(trace_file):
            do_stuff()

        events = read_events(trace_file)
        start = [e for e in events if e["type"] == "span_start"][0]
        assert start["name"] == "my_custom_name"

    def test_capture_input_false(self, trace_file: Path):
        @trace(capture_input=False)
        def secret_fn(password: str) -> str:
            return "ok"

        with Tracer(trace_file):
            secret_fn("hunter2")

        events = read_events(trace_file)
        start = [e for e in events if e["type"] == "span_start"][0]
        assert "input" not in start

    def test_capture_output_false(self, trace_file: Path):
        @trace(capture_output=False)
        def fn() -> str:
            return "sensitive"

        with Tracer(trace_file):
            fn()

        events = read_events(trace_file)
        end = [e for e in events if e["type"] == "span_end"][0]
        assert "output" not in end


class TestAsyncDecorator:
    async def test_async_function(self, trace_file: Path):
        @trace()
        async def async_add(a: int, b: int) -> int:
            return a + b

        async with Tracer(trace_file):
            result = await async_add(3, 4)

        assert result == 7
        events = read_events(trace_file)
        starts = [e for e in events if e["type"] == "span_start"]
        ends = [e for e in events if e["type"] == "span_end"]
        assert len(starts) == 1
        assert starts[0]["name"] == "async_add"
        assert len(ends) == 1
        assert ends[0]["output"] == 7

    async def test_async_error(self, trace_file: Path):
        @trace()
        async def async_fail():
            raise ValueError("async boom")

        async with Tracer(trace_file):
            with pytest.raises(ValueError, match="async boom"):
                await async_fail()

        events = read_events(trace_file)
        end = [e for e in events if e["type"] == "span_end"][0]
        assert end["status"] == "error"


class TestNesting:
    def test_nested_traces_parent_ids(self, trace_file: Path):
        @trace()
        def outer():
            return inner()

        @trace()
        def inner():
            return 42

        with Tracer(trace_file):
            outer()

        events = read_events(trace_file)
        outer_start = [e for e in events if e["type"] == "span_start" and e["name"] == "outer"][0]
        inner_start = [e for e in events if e["type"] == "span_start" and e["name"] == "inner"][0]
        assert inner_start["parent_id"] == outer_start["id"]

    async def test_async_nested_traces(self, trace_file: Path):
        @trace()
        async def pipeline():
            return await step()

        @trace()
        async def step():
            return "done"

        async with Tracer(trace_file):
            await pipeline()

        events = read_events(trace_file)
        pipeline_start = [e for e in events if e["type"] == "span_start" and e["name"] == "pipeline"][0]
        step_start = [e for e in events if e["type"] == "span_start" and e["name"] == "step"][0]
        assert step_start["parent_id"] == pipeline_start["id"]


class TestNoTracer:
    def test_no_tracer_passthrough(self):
        @trace()
        def add(a: int, b: int) -> int:
            return a + b

        result = add(1, 2)
        assert result == 3

    async def test_async_no_tracer_passthrough(self):
        @trace()
        async def add(a: int, b: int) -> int:
            return a + b

        result = await add(1, 2)
        assert result == 3


class TestSelfExclusion:
    def test_self_excluded_from_input(self, trace_file: Path):
        class MyClass:
            @trace()
            def method(self, x: int) -> int:
                return x * 2

        with Tracer(trace_file):
            MyClass().method(5)

        events = read_events(trace_file)
        start = [e for e in events if e["type"] == "span_start"][0]
        assert "self" not in start.get("input", {})
        assert start["input"]["x"] == 5
