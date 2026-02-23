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

    def test_metadata_parameter(self, trace_file: Path):
        @trace(metadata={"component": "auth", "version": 2})
        def login(user: str) -> bool:
            return True

        with Tracer(trace_file):
            login("alice")

        events = read_events(trace_file)
        start = [e for e in events if e["type"] == "span_start"][0]
        end = [e for e in events if e["type"] == "span_end"][0]
        assert start["metadata"]["component"] == "auth"
        assert start["metadata"]["version"] == 2
        assert end["metadata"]["component"] == "auth"

    def test_kind_parameter(self, trace_file: Path):
        @trace(kind="tool")
        def search(query: str) -> str:
            return "results"

        with Tracer(trace_file):
            search("hello")

        events = read_events(trace_file)
        start = [e for e in events if e["type"] == "span_start"][0]
        end = [e for e in events if e["type"] == "span_end"][0]
        assert start["kind"] == "tool"
        assert end["kind"] == "tool"


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

    async def test_async_metadata_and_kind(self, trace_file: Path):
        @trace(metadata={"provider": "openai"}, kind="llm")
        async def chat(prompt: str) -> str:
            return "response"

        async with Tracer(trace_file):
            await chat("hello")

        events = read_events(trace_file)
        start = [e for e in events if e["type"] == "span_start"][0]
        assert start["kind"] == "llm"
        assert start["metadata"]["provider"] == "openai"


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


class TestIgnoreArguments:
    def test_ignore_single_argument(self, trace_file: Path):
        @trace(ignore_arguments=["password"])
        def login(user: str, password: str) -> bool:
            return True

        with Tracer(trace_file):
            login("alice", "secret123")

        events = read_events(trace_file)
        start = [e for e in events if e["type"] == "span_start"][0]
        assert start["input"]["user"] == "alice"
        assert "password" not in start["input"]

    def test_ignore_multiple_arguments(self, trace_file: Path):
        @trace(ignore_arguments=["api_key", "token"])
        def call_api(url: str, api_key: str, token: str) -> str:
            return "ok"

        with Tracer(trace_file):
            call_api("https://example.com", "key123", "tok456")

        events = read_events(trace_file)
        start = [e for e in events if e["type"] == "span_start"][0]
        assert start["input"]["url"] == "https://example.com"
        assert "api_key" not in start["input"]
        assert "token" not in start["input"]

    def test_ignore_nonexistent_argument(self, trace_file: Path):
        @trace(ignore_arguments=["nonexistent"])
        def fn(x: int) -> int:
            return x

        with Tracer(trace_file):
            result = fn(42)

        assert result == 42
        events = read_events(trace_file)
        start = [e for e in events if e["type"] == "span_start"][0]
        assert start["input"]["x"] == 42

    async def test_ignore_arguments_async(self, trace_file: Path):
        @trace(ignore_arguments=["secret"])
        async def fetch(url: str, secret: str) -> str:
            return "data"

        async with Tracer(trace_file):
            result = await fetch("https://example.com", "s3cr3t")

        assert result == "data"
        events = read_events(trace_file)
        start = [e for e in events if e["type"] == "span_start"][0]
        assert start["input"]["url"] == "https://example.com"
        assert "secret" not in start["input"]


class TestGeneratorDecorator:
    def test_sync_generator(self, trace_file: Path):
        @trace()
        def count(n: int):
            for i in range(n):
                yield i

        with Tracer(trace_file):
            items = list(count(3))

        assert items == [0, 1, 2]
        events = read_events(trace_file)
        starts = [e for e in events if e["type"] == "span_start"]
        ends = [e for e in events if e["type"] == "span_end"]
        assert len(starts) == 1
        assert starts[0]["name"] == "count"
        assert starts[0]["input"] == {"n": 3}
        assert len(ends) == 1
        assert ends[0]["status"] == "ok"
        assert ends[0]["output"] == [0, 1, 2]

    def test_sync_generator_no_tracer_passthrough(self):
        @trace()
        def count(n: int):
            for i in range(n):
                yield i

        items = list(count(3))
        assert items == [0, 1, 2]

    def test_sync_generator_capture_output_false(self, trace_file: Path):
        @trace(capture_output=False)
        def count(n: int):
            for i in range(n):
                yield i

        with Tracer(trace_file):
            items = list(count(3))

        assert items == [0, 1, 2]
        events = read_events(trace_file)
        end = [e for e in events if e["type"] == "span_end"][0]
        assert "output" not in end

    def test_sync_generator_empty(self, trace_file: Path):
        @trace()
        def empty():
            return
            yield  # noqa: unreachable

        with Tracer(trace_file):
            items = list(empty())

        assert items == []
        events = read_events(trace_file)
        end = [e for e in events if e["type"] == "span_end"][0]
        assert "output" not in end  # Empty list means no output set

    async def test_async_generator(self, trace_file: Path):
        @trace()
        async def acount(n: int):
            for i in range(n):
                yield i

        async with Tracer(trace_file):
            items = [item async for item in acount(3)]

        assert items == [0, 1, 2]
        events = read_events(trace_file)
        starts = [e for e in events if e["type"] == "span_start"]
        ends = [e for e in events if e["type"] == "span_end"]
        assert len(starts) == 1
        assert starts[0]["name"] == "acount"
        assert starts[0]["input"] == {"n": 3}
        assert len(ends) == 1
        assert ends[0]["status"] == "ok"
        assert ends[0]["output"] == [0, 1, 2]

    async def test_async_generator_no_tracer_passthrough(self):
        @trace()
        async def acount(n: int):
            for i in range(n):
                yield i

        items = [item async for item in acount(3)]
        assert items == [0, 1, 2]

    def test_generator_with_ignore_arguments(self, trace_file: Path):
        @trace(ignore_arguments=["secret"])
        def gen(n: int, secret: str):
            for i in range(n):
                yield i

        with Tracer(trace_file):
            items = list(gen(2, "hidden"))

        assert items == [0, 1]
        events = read_events(trace_file)
        start = [e for e in events if e["type"] == "span_start"][0]
        assert start["input"]["n"] == 2
        assert "secret" not in start["input"]
