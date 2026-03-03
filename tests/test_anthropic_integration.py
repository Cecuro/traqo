"""Tests for Anthropic integration — full traced messages path with real Tracer."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("anthropic")

from tests.conftest import read_events
from traqo import Tracer
from traqo.integrations.anthropic import (
    _TracedAsyncMessages,
    _TracedMessages,
    traced_anthropic,
)

# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


def _make_message_response(
    text: str = "Hello",
    model: str = "claude-3-sonnet-20240229",
    input_tokens: int = 10,
    output_tokens: int = 5,
    tool_use: list[dict] | None = None,
    cache_read: int | None = None,
    cache_creation: int | None = None,
) -> MagicMock:
    """Build a realistic mock Anthropic Message response."""
    response = MagicMock()

    content_blocks = []
    text_block = MagicMock()
    text_block.text = text
    text_block.type = "text"
    content_blocks.append(text_block)

    if tool_use:
        for tu in tool_use:
            block = MagicMock()
            block.type = "tool_use"
            block.id = tu["id"]
            block.name = tu["name"]
            block.input = tu["input"]
            content_blocks.append(block)

    response.content = content_blocks

    usage = MagicMock()
    usage.input_tokens = input_tokens
    usage.output_tokens = output_tokens
    usage.cache_read_input_tokens = cache_read
    usage.cache_creation_input_tokens = cache_creation
    response.usage = usage
    response.model = model
    return response


def _make_stream_events(
    text_parts: list[str],
    model: str = "claude-3-sonnet-20240229",
    input_tokens: int = 10,
    output_tokens: int = 5,
) -> list[MagicMock]:
    """Build the ordered event sequence for a text-only Anthropic stream."""
    events = []

    # message_start
    evt = MagicMock()
    evt.type = "message_start"
    evt.message = MagicMock()
    evt.message.model = model
    evt.message.usage = MagicMock()
    evt.message.usage.input_tokens = input_tokens
    evt.message.usage.cache_read_input_tokens = None
    evt.message.usage.cache_creation_input_tokens = None
    events.append(evt)

    # content_block_delta for each text part
    for part in text_parts:
        evt = MagicMock()
        evt.type = "content_block_delta"
        evt.delta = MagicMock()
        evt.delta.type = "text_delta"
        evt.delta.text = part
        events.append(evt)

    # message_delta with output tokens
    evt = MagicMock()
    evt.type = "message_delta"
    evt.delta = MagicMock()
    evt.usage = MagicMock()
    evt.usage.output_tokens = output_tokens
    events.append(evt)

    return events


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMessagesCreateTextResponse:
    def test_text_response(self, trace_file: Path):
        response = _make_message_response(
            "Hello world", "claude-3-sonnet-20240229", 10, 5
        )
        mock_messages = MagicMock()
        mock_messages.create.return_value = response

        traced = _TracedMessages(mock_messages, "")

        with Tracer(path=trace_file):
            result = traced.create(
                model="claude-3-sonnet-20240229",
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=100,
            )

        assert result is response

        events = read_events(trace_file)
        span_starts = [e for e in events if e["type"] == "span_start"]
        span_ends = [e for e in events if e["type"] == "span_end"]
        trace_end = [e for e in events if e["type"] == "trace_end"][0]

        assert len(span_starts) == 1
        assert span_starts[0]["kind"] == "llm"
        assert span_starts[0]["metadata"]["provider"] == "anthropic"
        assert span_starts[0]["input"] == [{"role": "user", "content": "Hi"}]

        assert len(span_ends) == 1
        assert span_ends[0]["status"] == "ok"
        assert span_ends[0]["metadata"]["model"] == "claude-3-sonnet-20240229"
        assert span_ends[0]["metadata"]["token_usage"]["input_tokens"] == 10
        assert span_ends[0]["metadata"]["token_usage"]["output_tokens"] == 5
        assert span_ends[0]["output"] == "Hello world"

        assert trace_end["stats"]["total_input_tokens"] == 10
        assert trace_end["stats"]["total_output_tokens"] == 5


class TestMessagesCreateCacheTokens:
    def test_cache_tokens_in_metadata(self, trace_file: Path):
        response = _make_message_response(
            "Cached response",
            "claude-3-sonnet-20240229",
            100,
            50,
            cache_read=30,
            cache_creation=10,
        )
        mock_messages = MagicMock()
        mock_messages.create.return_value = response

        traced = _TracedMessages(mock_messages, "")

        with Tracer(path=trace_file):
            traced.create(
                model="claude-3-sonnet-20240229",
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=100,
            )

        events = read_events(trace_file)
        span_end = [e for e in events if e["type"] == "span_end"][0]
        usage = span_end["metadata"]["token_usage"]
        # input_tokens includes base (100) + cache_read (30) + cache_creation (10)
        assert usage["input_tokens"] == 140
        assert usage["output_tokens"] == 50
        assert usage["cache_read_tokens"] == 30
        assert usage["cache_creation_tokens"] == 10

        trace_end = [e for e in events if e["type"] == "trace_end"][0]
        assert trace_end["stats"]["total_input_tokens"] == 140


class TestMessagesCreateToolUse:
    def test_tool_use_in_response(self, trace_file: Path):
        tool_use = [
            {"id": "toolu_123", "name": "get_weather", "input": {"city": "NYC"}}
        ]
        response = _make_message_response(
            "", "claude-3-sonnet-20240229", 15, 20, tool_use=tool_use
        )
        mock_messages = MagicMock()
        mock_messages.create.return_value = response

        traced = _TracedMessages(mock_messages, "")

        with Tracer(path=trace_file):
            traced.create(
                model="claude-3-sonnet-20240229",
                messages=[{"role": "user", "content": "What's the weather?"}],
                max_tokens=100,
            )

        events = read_events(trace_file)
        span_end = [e for e in events if e["type"] == "span_end"][0]

        output = span_end["output"]
        assert output["content"] == ""
        assert len(output["tool_use"]) == 1
        assert output["tool_use"][0]["id"] == "toolu_123"
        assert output["tool_use"][0]["name"] == "get_weather"
        assert output["tool_use"][0]["input"] == {"city": "NYC"}


class TestMessagesCreateCaptureContentFalse:
    def test_no_input_output_in_events(self, trace_file: Path):
        response = _make_message_response("secret", "claude-3-sonnet-20240229", 10, 5)
        mock_messages = MagicMock()
        mock_messages.create.return_value = response

        traced = _TracedMessages(mock_messages, "")

        with Tracer(path=trace_file, capture_content=False):
            traced.create(
                model="claude-3-sonnet-20240229",
                messages=[{"role": "user", "content": "secret input"}],
                max_tokens=100,
            )

        events = read_events(trace_file)
        span_start = [e for e in events if e["type"] == "span_start"][0]
        span_end = [e for e in events if e["type"] == "span_end"][0]

        assert "input" not in span_start
        assert "output" not in span_end
        assert span_end["metadata"]["model"] == "claude-3-sonnet-20240229"
        assert span_end["metadata"]["token_usage"]["input_tokens"] == 10


class TestMessagesCreateSystemMessage:
    def test_system_kwarg_extracted(self, trace_file: Path):
        response = _make_message_response("Hello", "claude-3-sonnet-20240229", 10, 5)
        mock_messages = MagicMock()
        mock_messages.create.return_value = response

        traced = _TracedMessages(mock_messages, "")

        with Tracer(path=trace_file):
            traced.create(
                model="claude-3-sonnet-20240229",
                messages=[{"role": "user", "content": "Hi"}],
                system="You are a helpful assistant",
                max_tokens=100,
            )

        events = read_events(trace_file)
        span_start = [e for e in events if e["type"] == "span_start"][0]

        assert span_start["input"][0] == {
            "role": "system",
            "content": "You are a helpful assistant",
        }
        assert span_start["input"][1] == {"role": "user", "content": "Hi"}


class TestStreamingCacheTokens:
    def test_streaming_cache_tokens_merged(self, trace_file: Path):
        events_list = []

        # message_start with cache tokens
        evt = MagicMock()
        evt.type = "message_start"
        evt.message = MagicMock()
        evt.message.model = "claude-3-sonnet-20240229"
        evt.message.usage = MagicMock()
        evt.message.usage.input_tokens = 10
        evt.message.usage.cache_read_input_tokens = 500
        evt.message.usage.cache_creation_input_tokens = 200
        events_list.append(evt)

        # text delta
        evt = MagicMock()
        evt.type = "content_block_delta"
        evt.delta = MagicMock()
        evt.delta.type = "text_delta"
        evt.delta.text = "Hello"
        events_list.append(evt)

        # message_delta
        evt = MagicMock()
        evt.type = "message_delta"
        evt.delta = MagicMock()
        evt.usage = MagicMock()
        evt.usage.output_tokens = 5
        events_list.append(evt)

        mock_messages = MagicMock()
        mock_messages.create.return_value = iter(events_list)

        traced = _TracedMessages(mock_messages, "")

        with Tracer(path=trace_file):
            stream = traced.create(
                model="claude-3-sonnet-20240229",
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=100,
                stream=True,
            )
            list(stream)

        events = read_events(trace_file)
        span_end = [e for e in events if e["type"] == "span_end"][0]
        usage = span_end["metadata"]["token_usage"]
        # input_tokens includes base (10) + cache_read (500) + cache_creation (200)
        assert usage["input_tokens"] == 710
        assert usage["output_tokens"] == 5
        assert usage["cache_read_tokens"] == 500
        assert usage["cache_creation_tokens"] == 200

        trace_end = [e for e in events if e["type"] == "trace_end"][0]
        assert trace_end["stats"]["total_input_tokens"] == 710


class TestMessagesStreaming:
    def test_streaming_via_create(self, trace_file: Path):
        stream_events = _make_stream_events(
            ["Hello", " world"], input_tokens=10, output_tokens=5
        )
        mock_messages = MagicMock()
        mock_messages.create.return_value = iter(stream_events)

        traced = _TracedMessages(mock_messages, "")

        with Tracer(path=trace_file):
            stream = traced.create(
                model="claude-3-sonnet-20240229",
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=100,
                stream=True,
            )
            result_events = list(stream)

        assert len(result_events) == len(stream_events)

        events = read_events(trace_file)
        span_end = [e for e in events if e["type"] == "span_end"][0]
        trace_end = [e for e in events if e["type"] == "trace_end"][0]

        assert span_end["status"] == "ok"
        assert span_end["output"] == "Hello world"
        assert span_end["metadata"]["model"] == "claude-3-sonnet-20240229"
        assert span_end["metadata"]["token_usage"]["input_tokens"] == 10
        assert span_end["metadata"]["token_usage"]["output_tokens"] == 5
        assert "time_to_first_token_s" in span_end["metadata"]

        assert trace_end["stats"]["total_input_tokens"] == 10
        assert trace_end["stats"]["total_output_tokens"] == 5


class TestMessagesStreamMethod:
    def test_stream_method(self, trace_file: Path):
        stream_events = _make_stream_events(
            ["Hi", " there"],
            model="claude-3-haiku-20240307",
            input_tokens=8,
            output_tokens=3,
        )
        mock_messages = MagicMock()
        mock_messages.stream.return_value = iter(stream_events)

        traced = _TracedMessages(mock_messages, "")

        with Tracer(path=trace_file):
            stream = traced.stream(
                model="claude-3-haiku-20240307",
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=50,
            )
            result_events = list(stream)

        assert len(result_events) == len(stream_events)

        events = read_events(trace_file)
        span_start = [e for e in events if e["type"] == "span_start"][0]
        span_end = [e for e in events if e["type"] == "span_end"][0]

        assert span_start["name"] == "anthropic.messages.stream"
        assert span_end["output"] == "Hi there"
        assert span_end["metadata"]["model"] == "claude-3-haiku-20240307"


class TestStreamingToolUseAggregation:
    def test_tool_use_json_reassembly(self, trace_file: Path):
        events_list = []

        # message_start
        evt = MagicMock()
        evt.type = "message_start"
        evt.message = MagicMock()
        evt.message.model = "claude-3-sonnet-20240229"
        evt.message.usage = MagicMock()
        evt.message.usage.input_tokens = 20
        evt.message.usage.cache_read_input_tokens = None
        evt.message.usage.cache_creation_input_tokens = None
        events_list.append(evt)

        # content_block_start — tool_use
        evt = MagicMock()
        evt.type = "content_block_start"
        evt.content_block = MagicMock()
        evt.content_block.type = "tool_use"
        evt.content_block.id = "toolu_abc"
        evt.content_block.name = "get_weather"
        events_list.append(evt)

        # content_block_delta — input_json_delta part 1
        evt = MagicMock()
        evt.type = "content_block_delta"
        evt.delta = MagicMock()
        evt.delta.type = "input_json_delta"
        evt.delta.partial_json = '{"city":'
        events_list.append(evt)

        # content_block_delta — input_json_delta part 2
        evt = MagicMock()
        evt.type = "content_block_delta"
        evt.delta = MagicMock()
        evt.delta.type = "input_json_delta"
        evt.delta.partial_json = ' "NYC"}'
        events_list.append(evt)

        # content_block_stop
        evt = MagicMock()
        evt.type = "content_block_stop"
        events_list.append(evt)

        # message_delta
        evt = MagicMock()
        evt.type = "message_delta"
        evt.delta = MagicMock()
        evt.usage = MagicMock()
        evt.usage.output_tokens = 15
        events_list.append(evt)

        mock_messages = MagicMock()
        mock_messages.create.return_value = iter(events_list)

        traced = _TracedMessages(mock_messages, "")

        with Tracer(path=trace_file):
            stream = traced.create(
                model="claude-3-sonnet-20240229",
                messages=[{"role": "user", "content": "What's the weather?"}],
                max_tokens=100,
                stream=True,
            )
            list(stream)

        events = read_events(trace_file)
        span_end = [e for e in events if e["type"] == "span_end"][0]

        output = span_end["output"]
        assert output["content"] == ""
        assert len(output["tool_use"]) == 1
        assert output["tool_use"][0]["id"] == "toolu_abc"
        assert output["tool_use"][0]["name"] == "get_weather"
        assert output["tool_use"][0]["input"] == {"city": "NYC"}


class TestStreamingErrors:
    def test_error_during_create(self, trace_file: Path):
        mock_messages = MagicMock()
        mock_messages.create.side_effect = ConnectionError("API unavailable")

        traced = _TracedMessages(mock_messages, "")

        with Tracer(path=trace_file):
            with pytest.raises(ConnectionError, match="API unavailable"):
                traced.create(
                    model="claude-3-sonnet-20240229",
                    messages=[{"role": "user", "content": "Hi"}],
                    max_tokens=100,
                    stream=True,
                )

        events = read_events(trace_file)
        span_ends = [e for e in events if e["type"] == "span_end"]
        assert len(span_ends) == 1
        assert span_ends[0]["status"] == "error"


class TestNoTracerPassthrough:
    def test_returns_raw_response(self):
        response = _make_message_response()
        mock_messages = MagicMock()
        mock_messages.create.return_value = response

        traced = _TracedMessages(mock_messages, "")
        result = traced.create(
            model="claude-3-sonnet-20240229",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=100,
        )

        assert result is response
        mock_messages.create.assert_called_once()


class TestTracedAnthropicEndToEnd:
    def test_full_proxy_chain(self, trace_file: Path):
        response = _make_message_response("Hi there", "claude-3-opus-20240229", 25, 12)
        mock_client = MagicMock()
        mock_client.__class__ = type("Anthropic", (), {})
        mock_client.messages.create.return_value = response

        traced_client = traced_anthropic(mock_client)

        with Tracer(path=trace_file):
            result = traced_client.messages.create(
                model="claude-3-opus-20240229",
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=200,
            )

        assert result is response

        events = read_events(trace_file)
        span_end = [e for e in events if e["type"] == "span_end"][0]
        assert span_end["metadata"]["model"] == "claude-3-opus-20240229"
        assert span_end["output"] == "Hi there"

        trace_end = [e for e in events if e["type"] == "trace_end"][0]
        assert trace_end["stats"]["total_input_tokens"] == 25
        assert trace_end["stats"]["total_output_tokens"] == 12


class TestAsyncMessages:
    async def test_async_create(self, trace_file: Path):
        response = _make_message_response(
            "Async hello", "claude-3-sonnet-20240229", 10, 5
        )
        mock_messages = MagicMock()
        mock_messages.create = AsyncMock(return_value=response)

        traced = _TracedAsyncMessages(mock_messages, "")

        async with Tracer(path=trace_file):
            result = await traced.create(
                model="claude-3-sonnet-20240229",
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=100,
            )

        assert result is response

        events = read_events(trace_file)
        span_end = [e for e in events if e["type"] == "span_end"][0]
        assert span_end["status"] == "ok"
        assert span_end["output"] == "Async hello"
        assert span_end["metadata"]["model"] == "claude-3-sonnet-20240229"
