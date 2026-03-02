"""Tests for OpenAI integration — full traced completions path with real Tracer."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("openai")

from tests.conftest import read_events
from traqo import Tracer
from traqo.integrations.openai import (
    _TracedAsyncCompletions,
    _TracedCompletions,
    traced_openai,
)

# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


def _make_chat_response(
    content: str = "Hello",
    model: str = "gpt-4",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
    tool_calls: list[dict] | None = None,
) -> MagicMock:
    """Build a realistic mock ChatCompletion response."""
    response = MagicMock()
    msg = MagicMock()
    msg.content = content

    if tool_calls:
        mock_tcs = []
        for tc in tool_calls:
            mock_tc = MagicMock()
            mock_tc.id = tc["id"]
            mock_tc.type = tc.get("type", "function")
            mock_tc.function = MagicMock()
            mock_tc.function.name = tc["function"]["name"]
            mock_tc.function.arguments = tc["function"]["arguments"]
            mock_tcs.append(mock_tc)
        msg.tool_calls = mock_tcs
    else:
        msg.tool_calls = None

    choice = MagicMock()
    choice.message = msg
    response.choices = [choice]

    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    usage.prompt_tokens_details = None
    response.usage = usage
    response.model = model
    return response


def _make_stream_chunk(
    content: str | None = None,
    model: str | None = None,
    usage: dict[str, int] | None = None,
    tool_calls: list | None = None,
) -> MagicMock:
    """Build a mock ChatCompletionChunk."""
    chunk = MagicMock()
    chunk.model = model

    if usage:
        chunk.usage = MagicMock()
        chunk.usage.prompt_tokens = usage.get("prompt_tokens", 0)
        chunk.usage.completion_tokens = usage.get("completion_tokens", 0)
        chunk.usage.prompt_tokens_details = None
    else:
        chunk.usage = None

    if content is not None or tool_calls is not None:
        delta = MagicMock()
        delta.content = content
        delta.tool_calls = tool_calls
        choice = MagicMock()
        choice.delta = delta
        chunk.choices = [choice]
    else:
        chunk.choices = []

    return chunk


def _make_tool_call_delta(
    index: int,
    tc_id: str | None = None,
    name: str | None = None,
    arguments: str | None = None,
    tc_type: str | None = None,
) -> MagicMock:
    """Build a mock tool call delta for streaming."""
    tc = MagicMock()
    tc.index = index
    tc.id = tc_id
    tc.type = tc_type
    if name is not None or arguments is not None:
        tc.function = MagicMock()
        tc.function.name = name
        tc.function.arguments = arguments
    else:
        tc.function = None
    return tc


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCompletionsCreateTextResponse:
    def test_text_response(self, trace_file: Path):
        response = _make_chat_response("Hello world", "gpt-4", 10, 5)
        mock_completions = MagicMock()
        mock_completions.create.return_value = response

        traced = _TracedCompletions(mock_completions, "")

        with Tracer(path=trace_file):
            result = traced.create(
                model="gpt-4",
                messages=[{"role": "user", "content": "Hi"}],
            )

        assert result is response

        events = read_events(trace_file)
        span_starts = [e for e in events if e["type"] == "span_start"]
        span_ends = [e for e in events if e["type"] == "span_end"]
        trace_end = [e for e in events if e["type"] == "trace_end"][0]

        assert len(span_starts) == 1
        assert span_starts[0]["kind"] == "llm"
        assert span_starts[0]["metadata"]["provider"] == "openai"
        assert span_starts[0]["input"] == [{"role": "user", "content": "Hi"}]

        assert len(span_ends) == 1
        assert span_ends[0]["status"] == "ok"
        assert span_ends[0]["metadata"]["model"] == "gpt-4"
        assert span_ends[0]["metadata"]["token_usage"] == {
            "input_tokens": 10,
            "output_tokens": 5,
        }
        assert span_ends[0]["output"] == "Hello world"

        assert trace_end["stats"]["total_input_tokens"] == 10
        assert trace_end["stats"]["total_output_tokens"] == 5


class TestCompletionsCreateCachedTokens:
    def test_cached_tokens_in_metadata(self, trace_file: Path):
        response = _make_chat_response("Cached", "gpt-4o", 100, 50)
        # Add cached token details
        response.usage.prompt_tokens_details = MagicMock()
        response.usage.prompt_tokens_details.cached_tokens = 80

        mock_completions = MagicMock()
        mock_completions.create.return_value = response

        traced = _TracedCompletions(mock_completions, "")

        with Tracer(path=trace_file):
            traced.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Hi"}],
            )

        events = read_events(trace_file)
        span_end = [e for e in events if e["type"] == "span_end"][0]
        usage = span_end["metadata"]["token_usage"]
        # OpenAI prompt_tokens already includes cached, so input_tokens = 100
        assert usage["input_tokens"] == 100
        assert usage["output_tokens"] == 50
        assert usage["cached_input_tokens"] == 80

        trace_end = [e for e in events if e["type"] == "trace_end"][0]
        assert trace_end["stats"]["total_input_tokens"] == 100


class TestCompletionsCreateToolCalls:
    def test_tool_calls_in_response(self, trace_file: Path):
        tool_calls = [
            {
                "id": "call_abc",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city": "NYC"}'},
            }
        ]
        response = _make_chat_response("", "gpt-4", 15, 20, tool_calls=tool_calls)
        mock_completions = MagicMock()
        mock_completions.create.return_value = response

        traced = _TracedCompletions(mock_completions, "")

        with Tracer(path=trace_file):
            traced.create(
                model="gpt-4",
                messages=[{"role": "user", "content": "What's the weather?"}],
            )

        events = read_events(trace_file)
        span_end = [e for e in events if e["type"] == "span_end"][0]

        output = span_end["output"]
        assert output["content"] == ""
        assert len(output["tool_calls"]) == 1
        assert output["tool_calls"][0]["id"] == "call_abc"
        assert output["tool_calls"][0]["function"]["name"] == "get_weather"
        assert output["tool_calls"][0]["function"]["arguments"] == '{"city": "NYC"}'


class TestCompletionsCaptureContentFalse:
    def test_no_input_output_in_events(self, trace_file: Path):
        response = _make_chat_response("secret output", "gpt-4", 10, 5)
        mock_completions = MagicMock()
        mock_completions.create.return_value = response

        traced = _TracedCompletions(mock_completions, "")

        with Tracer(path=trace_file, capture_content=False):
            traced.create(
                model="gpt-4",
                messages=[{"role": "user", "content": "secret input"}],
            )

        events = read_events(trace_file)
        span_start = [e for e in events if e["type"] == "span_start"][0]
        span_end = [e for e in events if e["type"] == "span_end"][0]

        assert "input" not in span_start
        assert "output" not in span_end
        # Token usage and model are metadata, not content — still captured
        assert span_end["metadata"]["model"] == "gpt-4"
        assert span_end["metadata"]["token_usage"] == {
            "input_tokens": 10,
            "output_tokens": 5,
        }


class TestCompletionsStreaming:
    def test_streaming_full_path(self, trace_file: Path):
        chunk1 = _make_stream_chunk(content="Hello", model="gpt-4")
        chunk2 = _make_stream_chunk(content=" world", model="gpt-4")
        chunk3 = _make_stream_chunk(
            model="gpt-4",
            usage={"prompt_tokens": 10, "completion_tokens": 5},
        )

        mock_completions = MagicMock()
        mock_completions.create.return_value = iter([chunk1, chunk2, chunk3])

        traced = _TracedCompletions(mock_completions, "")

        with Tracer(path=trace_file):
            stream = traced.create(
                model="gpt-4",
                messages=[{"role": "user", "content": "Hi"}],
                stream=True,
            )
            chunks = list(stream)

        assert len(chunks) == 3

        # stream_options injected
        call_kwargs = mock_completions.create.call_args[1]
        assert call_kwargs["stream_options"] == {"include_usage": True}

        events = read_events(trace_file)
        span_end = [e for e in events if e["type"] == "span_end"][0]
        trace_end = [e for e in events if e["type"] == "trace_end"][0]

        assert span_end["status"] == "ok"
        assert span_end["output"] == "Hello world"
        assert span_end["metadata"]["model"] == "gpt-4"
        assert span_end["metadata"]["token_usage"] == {
            "input_tokens": 10,
            "output_tokens": 5,
        }
        assert "time_to_first_token_s" in span_end["metadata"]
        assert isinstance(span_end["metadata"]["time_to_first_token_s"], float)

        assert trace_end["stats"]["total_input_tokens"] == 10
        assert trace_end["stats"]["total_output_tokens"] == 5

    def test_streaming_preserves_existing_stream_options(self, trace_file: Path):
        chunk = _make_stream_chunk(content="Hi", model="gpt-4")
        mock_completions = MagicMock()
        mock_completions.create.return_value = iter([chunk])

        traced = _TracedCompletions(mock_completions, "")

        with Tracer(path=trace_file):
            stream = traced.create(
                model="gpt-4",
                messages=[{"role": "user", "content": "Hi"}],
                stream=True,
                stream_options={"include_usage": False},
            )
            list(stream)

        call_kwargs = mock_completions.create.call_args[1]
        assert call_kwargs["stream_options"] == {"include_usage": False}


class TestStreamingToolCallDeltas:
    def test_tool_call_assembly(self, trace_file: Path):
        # Chunk 1: start of tool call with id and name
        tc_delta1 = _make_tool_call_delta(
            0,
            tc_id="call_xyz",
            name="search",
            arguments="",
            tc_type="function",
        )
        chunk1 = _make_stream_chunk(model="gpt-4", tool_calls=[tc_delta1])

        # Chunk 2: first arguments fragment
        tc_delta2 = _make_tool_call_delta(0, arguments='{"query":')
        chunk2 = _make_stream_chunk(model="gpt-4", tool_calls=[tc_delta2])

        # Chunk 3: second arguments fragment
        tc_delta3 = _make_tool_call_delta(0, arguments=' "test"}')
        chunk3 = _make_stream_chunk(model="gpt-4", tool_calls=[tc_delta3])

        # Chunk 4: usage on final chunk
        chunk4 = _make_stream_chunk(
            model="gpt-4",
            usage={"prompt_tokens": 20, "completion_tokens": 15},
        )

        mock_completions = MagicMock()
        mock_completions.create.return_value = iter([chunk1, chunk2, chunk3, chunk4])

        traced = _TracedCompletions(mock_completions, "")

        with Tracer(path=trace_file):
            stream = traced.create(
                model="gpt-4",
                messages=[{"role": "user", "content": "search for test"}],
                stream=True,
            )
            list(stream)

        events = read_events(trace_file)
        span_end = [e for e in events if e["type"] == "span_end"][0]

        output = span_end["output"]
        assert output["content"] == ""
        assert len(output["tool_calls"]) == 1
        assert output["tool_calls"][0]["id"] == "call_xyz"
        assert output["tool_calls"][0]["function"]["name"] == "search"
        assert output["tool_calls"][0]["function"]["arguments"] == '{"query": "test"}'

    def test_multiple_tool_calls_by_index(self, trace_file: Path):
        tc0 = _make_tool_call_delta(
            0,
            tc_id="call_1",
            name="fn_a",
            arguments='{"a":1}',
            tc_type="function",
        )
        tc1 = _make_tool_call_delta(
            1,
            tc_id="call_2",
            name="fn_b",
            arguments='{"b":2}',
            tc_type="function",
        )
        chunk1 = _make_stream_chunk(model="gpt-4", tool_calls=[tc0])
        chunk2 = _make_stream_chunk(model="gpt-4", tool_calls=[tc1])
        chunk3 = _make_stream_chunk(
            model="gpt-4",
            usage={"prompt_tokens": 5, "completion_tokens": 10},
        )

        mock_completions = MagicMock()
        mock_completions.create.return_value = iter([chunk1, chunk2, chunk3])

        traced = _TracedCompletions(mock_completions, "")

        with Tracer(path=trace_file):
            stream = traced.create(model="gpt-4", messages=[], stream=True)
            list(stream)

        events = read_events(trace_file)
        span_end = [e for e in events if e["type"] == "span_end"][0]

        tool_calls = span_end["output"]["tool_calls"]
        assert len(tool_calls) == 2
        assert tool_calls[0]["function"]["name"] == "fn_a"
        assert tool_calls[1]["function"]["name"] == "fn_b"


class TestStreamingErrors:
    def test_error_during_create(self, trace_file: Path):
        mock_completions = MagicMock()
        mock_completions.create.side_effect = ConnectionError("API down")

        traced = _TracedCompletions(mock_completions, "")

        with Tracer(path=trace_file):
            with pytest.raises(ConnectionError, match="API down"):
                traced.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": "Hi"}],
                    stream=True,
                )

        events = read_events(trace_file)
        span_ends = [e for e in events if e["type"] == "span_end"]
        assert len(span_ends) == 1
        assert span_ends[0]["status"] == "error"


class _MockContextManagerStream:
    """A stream that supports both iteration and context manager protocol."""

    def __init__(self, chunks):
        self._chunks = chunks
        self._iter = None

    def __iter__(self):
        self._iter = iter(self._chunks)
        return self

    def __next__(self):
        if self._iter is None:
            self._iter = iter(self._chunks)
        return next(self._iter)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class TestStreamContextManager:
    def test_stream_as_context_manager(self, trace_file: Path):
        chunk1 = _make_stream_chunk(content="Hello", model="gpt-4")
        chunk2 = _make_stream_chunk(
            model="gpt-4",
            usage={"prompt_tokens": 10, "completion_tokens": 5},
        )

        mock_stream = _MockContextManagerStream([chunk1, chunk2])

        mock_completions = MagicMock()
        mock_completions.create.return_value = mock_stream

        traced = _TracedCompletions(mock_completions, "")

        with Tracer(path=trace_file):
            stream = traced.create(
                model="gpt-4",
                messages=[{"role": "user", "content": "Hi"}],
                stream=True,
            )
            with stream as s:
                chunks = list(s)

        assert len(chunks) == 2
        events = read_events(trace_file)
        span_ends = [e for e in events if e["type"] == "span_end"]
        assert len(span_ends) == 1
        assert span_ends[0]["status"] == "ok"


class TestNoTracerPassthrough:
    def test_returns_raw_response(self):
        response = _make_chat_response()
        mock_completions = MagicMock()
        mock_completions.create.return_value = response

        traced = _TracedCompletions(mock_completions, "")
        result = traced.create(model="gpt-4", messages=[])

        assert result is response
        mock_completions.create.assert_called_once()


class TestModelParamsInSpan:
    def test_model_params_captured(self, trace_file: Path):
        response = _make_chat_response()
        mock_completions = MagicMock()
        mock_completions.create.return_value = response

        traced = _TracedCompletions(mock_completions, "")

        with Tracer(path=trace_file):
            traced.create(
                model="gpt-4",
                messages=[{"role": "user", "content": "Hi"}],
                temperature=0.7,
                max_tokens=100,
            )

        events = read_events(trace_file)
        span_start = [e for e in events if e["type"] == "span_start"][0]
        assert span_start["metadata"]["model_parameters"]["temperature"] == 0.7
        assert span_start["metadata"]["model_parameters"]["max_tokens"] == 100


class TestTracedOpenAIEndToEnd:
    def test_full_proxy_chain(self, trace_file: Path):
        response = _make_chat_response("Hi there", "gpt-4o", 12, 8)
        mock_client = MagicMock()
        mock_client.__class__ = type("OpenAI", (), {})
        mock_client.chat.completions.create.return_value = response

        traced_client = traced_openai(mock_client)

        with Tracer(path=trace_file):
            result = traced_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Hello"}],
            )

        assert result is response

        events = read_events(trace_file)
        span_end = [e for e in events if e["type"] == "span_end"][0]
        assert span_end["metadata"]["model"] == "gpt-4o"
        assert span_end["output"] == "Hi there"

        trace_end = [e for e in events if e["type"] == "trace_end"][0]
        assert trace_end["stats"]["total_input_tokens"] == 12
        assert trace_end["stats"]["total_output_tokens"] == 8


class TestAsyncCompletions:
    async def test_async_create(self, trace_file: Path):
        response = _make_chat_response("Async hello", "gpt-4", 10, 5)
        mock_completions = MagicMock()
        mock_completions.create = AsyncMock(return_value=response)

        traced = _TracedAsyncCompletions(mock_completions, "")

        async with Tracer(path=trace_file):
            result = await traced.create(
                model="gpt-4",
                messages=[{"role": "user", "content": "Hi"}],
            )

        assert result is response

        events = read_events(trace_file)
        span_end = [e for e in events if e["type"] == "span_end"][0]
        assert span_end["status"] == "ok"
        assert span_end["output"] == "Async hello"
        assert span_end["metadata"]["model"] == "gpt-4"
        assert span_end["metadata"]["token_usage"] == {
            "input_tokens": 10,
            "output_tokens": 5,
        }
