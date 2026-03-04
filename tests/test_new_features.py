"""Tests for new features: bare @trace, kind constants, update_current_span,
TTFT tracking, model params, embeddings, Responses API, Gemini wrapper."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tests.conftest import read_events
from traqo import (
    AGENT,
    CHAIN,
    EMBEDDING,
    GUARDRAIL,
    LLM,
    RETRIEVER,
    TOOL,
    Tracer,
    trace,
    update_current_span,
)

# ===================================================================
# 1. Bare @trace decorator
# ===================================================================


class TestBareTrace:
    def test_bare_trace_no_parens(self, trace_file: Path):
        """@trace (no parens) should work."""

        @trace
        def add(a: int, b: int) -> int:
            return a + b

        with Tracer(path=trace_file):
            result = add(1, 2)

        assert result == 3
        events = read_events(trace_file)
        starts = [e for e in events if e["type"] == "span_start"]
        assert len(starts) == 1
        assert starts[0]["name"] == "add"

    def test_trace_with_name_kwarg(self, trace_file: Path):
        """@trace(name="custom") should work."""

        @trace(name="custom_name")
        def add(a: int, b: int) -> int:
            return a + b

        with Tracer(path=trace_file):
            add(1, 2)

        events = read_events(trace_file)
        starts = [e for e in events if e["type"] == "span_start"]
        assert starts[0]["name"] == "custom_name"

    def test_bare_trace_async(self, trace_file: Path):
        """@trace on async function (no parens)."""
        import asyncio

        @trace
        async def async_add(a: int, b: int) -> int:
            return a + b

        async def run():
            async with Tracer(path=trace_file):
                return await async_add(3, 4)

        result = asyncio.run(run())
        assert result == 7
        events = read_events(trace_file)
        starts = [e for e in events if e["type"] == "span_start"]
        assert starts[0]["name"] == "async_add"

    def test_bare_trace_generator(self, trace_file: Path):
        """@trace on generator (no parens)."""

        @trace
        def gen(n: int):
            yield from range(n)

        with Tracer(path=trace_file):
            items = list(gen(3))

        assert items == [0, 1, 2]
        events = read_events(trace_file)
        starts = [e for e in events if e["type"] == "span_start"]
        assert starts[0]["name"] == "gen"


# ===================================================================
# 2. Span kind constants
# ===================================================================


class TestKindConstants:
    def test_constants_values(self):
        assert LLM == "llm"
        assert TOOL == "tool"
        assert RETRIEVER == "retriever"
        assert CHAIN == "chain"
        assert AGENT == "agent"
        assert EMBEDDING == "embedding"
        assert GUARDRAIL == "guardrail"

    def test_constant_used_in_decorator(self, trace_file: Path):
        @trace(kind=LLM)
        def chat(prompt: str) -> str:
            return "response"

        with Tracer(path=trace_file):
            chat("hello")

        events = read_events(trace_file)
        start = [e for e in events if e["type"] == "span_start"][0]
        assert start["kind"] == "llm"


# ===================================================================
# 3. update_current_span convenience helper
# ===================================================================


class TestUpdateCurrentSpan:
    def test_update_output(self, trace_file: Path):
        with Tracer(path=trace_file) as tracer, tracer.span("test_span"):
            update_current_span(output="custom_output")

        events = read_events(trace_file)
        end = [e for e in events if e["type"] == "span_end"][0]
        assert end["output"] == "custom_output"

    def test_update_metadata_dict(self, trace_file: Path):
        @trace()
        def fn():
            update_current_span(metadata={"key1": "val1", "key2": 42})

        with Tracer(path=trace_file):
            fn()

        events = read_events(trace_file)
        end = [e for e in events if e["type"] == "span_end"][0]
        assert end["metadata"]["key1"] == "val1"
        assert end["metadata"]["key2"] == 42

    def test_update_kw_metadata(self, trace_file: Path):
        @trace()
        def fn():
            update_current_span(score=0.95, model="gpt-4")

        with Tracer(path=trace_file):
            fn()

        events = read_events(trace_file)
        end = [e for e in events if e["type"] == "span_end"][0]
        assert end["metadata"]["score"] == 0.95
        assert end["metadata"]["model"] == "gpt-4"

    def test_update_tags(self, trace_file: Path):
        @trace(tags=["existing"])
        def fn():
            update_current_span(tags=["new_tag"])

        with Tracer(path=trace_file):
            fn()

        events = read_events(trace_file)
        end = [e for e in events if e["type"] == "span_end"][0]
        assert "existing" in end["tags"]
        assert "new_tag" in end["tags"]

    def test_noop_when_no_span(self):
        """Should not raise when no span is active."""
        update_current_span(output="whatever", metadata={"k": "v"})

    def test_update_output_none(self, trace_file: Path):
        """Setting output=None should work (not be confused with 'not provided')."""

        @trace()
        def fn():
            update_current_span(output=None)

        with Tracer(path=trace_file):
            fn()

        events = read_events(trace_file)
        end = [e for e in events if e["type"] == "span_end"][0]
        # Output was explicitly set to None, so it should not appear (Tracer only writes if not None)
        assert "output" not in end


# ===================================================================
# 4. TTFT tracking (OpenAI stream wrapper, mocked)
# ===================================================================


class TestOpenAITTFT:
    @pytest.fixture(autouse=True)
    def _skip_no_openai(self):
        pytest.importorskip("openai")

    def test_sync_stream_ttft(self, trace_file: Path):
        """_StreamWrapper should record time_to_first_token_s."""
        from traqo.integrations.openai import _StreamWrapper

        # Create a mock span and chunks
        span = MagicMock()
        tracer = MagicMock()

        # Build fake chunks
        chunk1 = MagicMock()
        chunk1.model = "gpt-4"
        chunk1.usage = None
        chunk1.choices = [MagicMock()]
        chunk1.choices[0].delta = MagicMock()
        chunk1.choices[0].delta.content = "Hello"
        chunk1.choices[0].delta.tool_calls = None

        chunk2 = MagicMock()
        chunk2.model = "gpt-4"
        chunk2.usage = None
        chunk2.choices = [MagicMock()]
        chunk2.choices[0].delta = MagicMock()
        chunk2.choices[0].delta.content = " world"
        chunk2.choices[0].delta.tool_calls = None

        stream = iter([chunk1, chunk2])
        wrapper = _StreamWrapper(stream, span, tracer, True)
        chunks = list(wrapper)

        assert len(chunks) == 2
        # Check that time_to_first_token_s was set on first chunk
        ttft_calls = [
            c
            for c in span.set_metadata.call_args_list
            if c[0][0] == "time_to_first_token_s"
        ]
        assert len(ttft_calls) == 1
        ttft_val = ttft_calls[0][0][1]
        assert isinstance(ttft_val, float)
        assert ttft_val >= 0


class TestAnthropicTTFT:
    @pytest.fixture(autouse=True)
    def _skip_no_anthropic(self):
        pytest.importorskip("anthropic")

    def test_sync_stream_ttft(self, trace_file: Path):
        """Anthropic _StreamWrapper should record time_to_first_token_s."""
        from traqo.integrations.anthropic import _StreamWrapper

        span = MagicMock()
        tracer = MagicMock()

        event1 = MagicMock()
        event1.type = "message_start"
        event1.message = MagicMock()
        event1.message.model = "claude-3"
        event1.message.usage = MagicMock()
        event1.message.usage.input_tokens = 10

        event2 = MagicMock()
        event2.type = "content_block_delta"
        event2.delta = MagicMock()
        event2.delta.type = "text_delta"
        event2.delta.text = "Hello"

        stream = iter([event1, event2])
        wrapper = _StreamWrapper(stream, span, tracer, True)
        events = list(wrapper)

        assert len(events) == 2
        ttft_calls = [
            c
            for c in span.set_metadata.call_args_list
            if c[0][0] == "time_to_first_token_s"
        ]
        assert len(ttft_calls) == 1


# ===================================================================
# 5. Model parameters extraction
# ===================================================================


class TestOpenAIModelParams:
    @pytest.fixture(autouse=True)
    def _skip_no_openai(self):
        pytest.importorskip("openai")

    def test_model_params_extracted(self, trace_file: Path):
        """Model parameters should appear in span metadata."""
        from traqo.integrations.openai import _CHAT_MODEL_PARAMS, _extract_model_params

        kwargs = {
            "temperature": 0.7,
            "max_tokens": 100,
            "top_p": 0.9,
            "other_thing": True,
        }
        params = _extract_model_params(kwargs, _CHAT_MODEL_PARAMS)
        assert params == {"temperature": 0.7, "max_tokens": 100, "top_p": 0.9}

    def test_no_model_params(self):
        from traqo.integrations.openai import _CHAT_MODEL_PARAMS, _extract_model_params

        kwargs = {"messages": []}
        params = _extract_model_params(kwargs, _CHAT_MODEL_PARAMS)
        assert params is None


class TestAnthropicModelParams:
    @pytest.fixture(autouse=True)
    def _skip_no_anthropic(self):
        pytest.importorskip("anthropic")

    def test_model_params_extracted(self):
        from traqo.integrations.anthropic import (
            _ANTHROPIC_MODEL_PARAMS,
            _extract_model_params,
        )

        kwargs = {"temperature": 0.5, "max_tokens": 200, "top_k": 40}
        params = _extract_model_params(kwargs, _ANTHROPIC_MODEL_PARAMS)
        assert params == {"temperature": 0.5, "max_tokens": 200, "top_k": 40}


# ===================================================================
# 6. OpenAI Embeddings wrapper
# ===================================================================


class TestOpenAIEmbeddings:
    @pytest.fixture(autouse=True)
    def _skip_no_openai(self):
        pytest.importorskip("openai")

    def test_embeddings_no_tracer(self):
        """Without active tracer, should pass through."""
        from traqo.integrations.openai import _TracedEmbeddings

        mock_embeddings = MagicMock()
        mock_response = MagicMock()
        mock_response.model = "text-embedding-ada-002"
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 5
        mock_embeddings.create.return_value = mock_response

        traced = _TracedEmbeddings(mock_embeddings, "")
        result = traced.create(input="hello world", model="text-embedding-ada-002")
        assert result == mock_response
        mock_embeddings.create.assert_called_once()

    def test_embeddings_with_tracer(self, trace_file: Path):
        """With active tracer, should create embedding span."""
        from traqo.integrations.openai import _TracedEmbeddings

        mock_embeddings = MagicMock()
        mock_response = MagicMock()
        mock_response.model = "text-embedding-ada-002"
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 5
        mock_embeddings.create.return_value = mock_response

        traced = _TracedEmbeddings(mock_embeddings, "")

        with Tracer(path=trace_file):
            result = traced.create(input="hello world", model="text-embedding-ada-002")

        assert result == mock_response
        events = read_events(trace_file)
        starts = [e for e in events if e["type"] == "span_start"]
        ends = [e for e in events if e["type"] == "span_end"]
        assert len(starts) == 1
        assert starts[0]["kind"] == "embedding"
        assert starts[0]["input"] == "hello world"
        assert ends[0]["metadata"]["model"] == "text-embedding-ada-002"
        assert ends[0]["metadata"]["token_usage"]["input_tokens"] == 5


# ===================================================================
# 7. OpenAI Responses API wrapper
# ===================================================================


class TestOpenAIResponses:
    @pytest.fixture(autouse=True)
    def _skip_no_openai(self):
        pytest.importorskip("openai")

    def test_responses_no_tracer(self):
        """Without active tracer, should pass through."""
        from traqo.integrations.openai import _TracedResponses

        mock_responses = MagicMock()
        mock_response = MagicMock()
        mock_response.model = "gpt-4"
        mock_response.output_text = "Hello!"
        mock_response.output = []
        mock_response.usage = MagicMock()
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 5
        mock_responses.create.return_value = mock_response

        traced = _TracedResponses(mock_responses, "")
        result = traced.create(input="Hi", model="gpt-4")
        assert result == mock_response

    def test_responses_with_tracer(self, trace_file: Path):
        """With active tracer, should create LLM span."""
        from traqo.integrations.openai import _TracedResponses

        mock_responses = MagicMock()
        mock_response = MagicMock()
        mock_response.model = "gpt-4"
        mock_response.output_text = "Hello!"
        mock_response.output = []
        mock_response.usage = MagicMock()
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 5
        mock_responses.create.return_value = mock_response

        traced = _TracedResponses(mock_responses, "")

        with Tracer(path=trace_file):
            result = traced.create(input="Hi", model="gpt-4")

        assert result == mock_response
        events = read_events(trace_file)
        starts = [e for e in events if e["type"] == "span_start"]
        ends = [e for e in events if e["type"] == "span_end"]
        assert len(starts) == 1
        assert starts[0]["kind"] == "llm"
        assert starts[0]["input"]["input"] == "Hi"
        assert ends[0]["metadata"]["model"] == "gpt-4"
        assert ends[0]["metadata"]["token_usage"]["input_tokens"] == 10
        assert ends[0]["output"] == "Hello!"

    def test_responses_with_tool_calls(self, trace_file: Path):
        """Should extract function calls from response output."""
        from traqo.integrations.openai import _TracedResponses

        mock_responses = MagicMock()
        mock_response = MagicMock()
        mock_response.model = "gpt-4"
        mock_response.output_text = ""

        fc_item = MagicMock()
        fc_item.type = "function_call"
        fc_item.call_id = "call_123"
        fc_item.name = "get_weather"
        fc_item.arguments = '{"city": "NYC"}'
        mock_response.output = [fc_item]

        mock_response.usage = MagicMock()
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 20
        mock_responses.create.return_value = mock_response

        traced = _TracedResponses(mock_responses, "")

        with Tracer(path=trace_file):
            traced.create(input="What's the weather?", model="gpt-4")

        events = read_events(trace_file)
        ends = [e for e in events if e["type"] == "span_end"]
        output = ends[0]["output"]
        assert "tool_calls" in output
        assert output["tool_calls"][0]["name"] == "get_weather"

    def test_responses_streaming(self):
        """Streaming responses should track TTFT and extract output."""
        from traqo.integrations.openai import _ResponsesStreamWrapper

        span = MagicMock()
        tracer = MagicMock()

        # Create stream events
        text_delta_event = MagicMock()
        text_delta_event.type = "response.output_text.delta"

        completed_event = MagicMock()
        completed_event.type = "response.completed"
        completed_event.response = MagicMock()
        completed_event.response.model = "gpt-4"
        completed_event.response.output_text = "Hello world"
        completed_event.response.output = []
        completed_event.response.usage = MagicMock()
        completed_event.response.usage.input_tokens = 10
        completed_event.response.usage.output_tokens = 5

        mock_stream = iter([text_delta_event, completed_event])
        wrapper = _ResponsesStreamWrapper(mock_stream, span, tracer, True)
        events_received = list(wrapper)

        assert len(events_received) == 2
        # Check TTFT was set
        ttft_calls = [
            c
            for c in span.set_metadata.call_args_list
            if c[0][0] == "time_to_first_token_s"
        ]
        assert len(ttft_calls) == 1
        assert ttft_calls[0][0][1] >= 0
        # Check model was set
        model_calls = [
            c for c in span.set_metadata.call_args_list if c[0][0] == "model"
        ]
        assert model_calls[0][0][1] == "gpt-4"
        # Check output was set
        span.set_output.assert_called_once_with("Hello world")


# ===================================================================
# 8. _TracedOpenAIClient property tests
# ===================================================================


class TestTracedOpenAIClientProperties:
    @pytest.fixture(autouse=True)
    def _skip_no_openai(self):
        pytest.importorskip("openai")

    def test_has_embeddings_property(self):
        from traqo.integrations.openai import _TracedEmbeddings, _TracedOpenAIClient

        mock_client = MagicMock()
        mock_client.__class__ = type("OpenAI", (), {})
        traced = _TracedOpenAIClient(mock_client, "")
        assert isinstance(traced.embeddings, _TracedEmbeddings)

    def test_has_responses_property(self):
        from traqo.integrations.openai import _TracedOpenAIClient, _TracedResponses

        mock_client = MagicMock()
        mock_client.__class__ = type("OpenAI", (), {})
        traced = _TracedOpenAIClient(mock_client, "")
        assert isinstance(traced.responses, _TracedResponses)


# ===================================================================
# 9. Gemini wrapper (mocked)
# ===================================================================


class TestGeminiExtractors:
    @pytest.fixture(autouse=True)
    def _skip_no_gemini(self):
        pytest.importorskip("google.genai")

    def test_extract_model_params_from_dict(self):
        from traqo.integrations.gemini import _extract_model_params_from_config

        config = {"temperature": 0.7, "max_output_tokens": 100, "other": True}
        params = _extract_model_params_from_config(config)
        assert params == {"temperature": 0.7, "max_output_tokens": 100}

    def test_extract_model_params_from_object(self):
        from traqo.integrations.gemini import _extract_model_params_from_config

        config = MagicMock()
        config.temperature = 0.5
        config.max_output_tokens = 200
        config.top_p = None
        config.top_k = None
        params = _extract_model_params_from_config(config)
        assert params == {"temperature": 0.5, "max_output_tokens": 200}

    def test_extract_model_params_none(self):
        from traqo.integrations.gemini import _extract_model_params_from_config

        assert _extract_model_params_from_config(None) is None

    def test_extract_usage(self):
        from traqo.integrations.gemini import _extract_usage

        response = MagicMock()
        response.usage_metadata = MagicMock()
        response.usage_metadata.prompt_token_count = 10
        response.usage_metadata.candidates_token_count = 20
        usage = _extract_usage(response)
        assert usage == {"input_tokens": 10, "output_tokens": 20}

    def test_extract_usage_no_metadata(self):
        from traqo.integrations.gemini import _extract_usage

        response = MagicMock()
        response.usage_metadata = None
        assert _extract_usage(response) == {}

    def test_extract_output_text(self):
        from traqo.integrations.gemini import _extract_output

        response = MagicMock()
        response.text = "Hello world"
        response.candidates = []
        assert _extract_output(response) == "Hello world"

    def test_extract_output_function_call(self):
        from traqo.integrations.gemini import _extract_output

        fc = MagicMock()
        fc.function_call = MagicMock()
        fc.function_call.name = "get_weather"
        fc.function_call.args = {"city": "NYC"}

        part_text = MagicMock()
        part_text.function_call = None

        content = MagicMock()
        content.parts = [part_text, fc]

        candidate = MagicMock()
        candidate.content = content

        response = MagicMock()
        response.text = "Looking up weather..."
        response.candidates = [candidate]

        output = _extract_output(response)
        assert isinstance(output, dict)
        assert output["content"] == "Looking up weather..."
        assert len(output["function_calls"]) == 1
        assert output["function_calls"][0]["name"] == "get_weather"


class TestGeminiTracedModels:
    @pytest.fixture(autouse=True)
    def _skip_no_gemini(self):
        pytest.importorskip("google.genai")

    def test_generate_content_no_tracer(self):
        from traqo.integrations.gemini import _TracedModels

        mock_models = MagicMock()
        mock_response = MagicMock()
        mock_models.generate_content.return_value = mock_response

        traced = _TracedModels(mock_models, "")
        result = traced.generate_content(model="gemini-pro", contents="Hello")
        assert result == mock_response

    def test_generate_content_with_tracer(self, trace_file: Path):
        from traqo.integrations.gemini import _TracedModels

        mock_models = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Hello!"
        mock_response.candidates = []
        mock_response.usage_metadata = MagicMock()
        mock_response.usage_metadata.prompt_token_count = 5
        mock_response.usage_metadata.candidates_token_count = 10
        mock_models.generate_content.return_value = mock_response

        traced = _TracedModels(mock_models, "")

        with Tracer(path=trace_file):
            result = traced.generate_content(model="gemini-pro", contents="Hello")

        assert result == mock_response
        events = read_events(trace_file)
        starts = [e for e in events if e["type"] == "span_start"]
        ends = [e for e in events if e["type"] == "span_end"]
        assert starts[0]["kind"] == "llm"
        assert starts[0]["metadata"]["provider"] == "gemini"
        assert starts[0]["metadata"]["model"] == "gemini-pro"
        assert ends[0]["metadata"]["token_usage"]["input_tokens"] == 5
        assert ends[0]["metadata"]["token_usage"]["output_tokens"] == 10
        assert ends[0]["output"] == "Hello!"

    def test_generate_content_stream_ttft(self):
        """Gemini stream wrapper should track TTFT and aggregate chunks."""
        from traqo.integrations.gemini import _StreamWrapper

        span = MagicMock()
        tracer = MagicMock()

        chunk1 = MagicMock()
        chunk1.text = "Hi"
        chunk1.candidates = []
        chunk1.usage_metadata = None

        chunk2 = MagicMock()
        chunk2.text = " there"
        chunk2.candidates = []
        chunk2.usage_metadata = MagicMock()
        chunk2.usage_metadata.prompt_token_count = 5
        chunk2.usage_metadata.candidates_token_count = 10

        stream = iter([chunk1, chunk2])
        wrapper = _StreamWrapper(stream, span, tracer, True)
        chunks = list(wrapper)

        assert len(chunks) == 2
        # Check TTFT was set on first chunk
        ttft_calls = [
            c
            for c in span.set_metadata.call_args_list
            if c[0][0] == "time_to_first_token_s"
        ]
        assert len(ttft_calls) == 1
        assert ttft_calls[0][0][1] >= 0
        # Check token usage was set
        usage_calls = [
            c for c in span.set_metadata.call_args_list if c[0][0] == "token_usage"
        ]
        assert usage_calls[0][0][1] == {"input_tokens": 5, "output_tokens": 10}
        # Check output was set
        span.set_output.assert_called_once_with("Hi there")

    def test_embed_content(self, trace_file: Path):
        from traqo.integrations.gemini import _TracedModels

        mock_models = MagicMock()
        mock_response = MagicMock()
        mock_models.embed_content.return_value = mock_response

        traced = _TracedModels(mock_models, "")

        with Tracer(path=trace_file):
            result = traced.embed_content(model="embedding-001", contents="Hello")

        assert result == mock_response
        events = read_events(trace_file)
        starts = [e for e in events if e["type"] == "span_start"]
        assert starts[0]["kind"] == "embedding"
        assert starts[0]["metadata"]["provider"] == "gemini"


class TestGeminiClient:
    @pytest.fixture(autouse=True)
    def _skip_no_gemini(self):
        pytest.importorskip("google.genai")

    def test_traced_gemini_proxy_structure(self):
        from traqo.integrations.gemini import _TracedAio, _TracedModels, traced_gemini

        mock_client = MagicMock()
        traced = traced_gemini(mock_client)
        assert isinstance(traced.models, _TracedModels)
        assert isinstance(traced.aio, _TracedAio)

    def test_getattr_passthrough(self):
        from traqo.integrations.gemini import traced_gemini

        mock_client = MagicMock()
        mock_client.some_method.return_value = "test"
        traced = traced_gemini(mock_client)
        assert traced.some_method() == "test"


# ===================================================================
# 10. _extract_responses_output helper
# ===================================================================


class TestExtractResponsesOutput:
    @pytest.fixture(autouse=True)
    def _skip_no_openai(self):
        pytest.importorskip("openai")

    def test_text_only(self):
        from traqo.integrations.openai import _extract_responses_output

        response = MagicMock()
        response.output_text = "Hello world"
        response.output = []
        assert _extract_responses_output(response) == "Hello world"

    def test_with_function_calls(self):
        from traqo.integrations.openai import _extract_responses_output

        fc = MagicMock()
        fc.type = "function_call"
        fc.call_id = "call_1"
        fc.name = "search"
        fc.arguments = '{"q": "test"}'

        response = MagicMock()
        response.output_text = ""
        response.output = [fc]

        result = _extract_responses_output(response)
        assert isinstance(result, dict)
        assert result["tool_calls"][0]["name"] == "search"
