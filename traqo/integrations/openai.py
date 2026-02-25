"""OpenAI integration — wrap OpenAI client to auto-trace LLM calls as spans."""

from __future__ import annotations

import sys
import time
from typing import Any

try:
    import openai as _openai_mod
except ImportError as err:
    raise ImportError(
        "OpenAI not installed. Install with: pip install traqo[openai]"
    ) from err

from traqo.tracer import get_tracer

_CHAT_MODEL_PARAMS = (
    "temperature",
    "max_tokens",
    "max_completion_tokens",
    "top_p",
    "frequency_penalty",
    "presence_penalty",
)


def _extract_model_params(
    kwargs: dict[str, Any], param_names: tuple[str, ...]
) -> dict[str, Any] | None:
    """Extract generation parameters present in kwargs."""
    params = {k: kwargs[k] for k in param_names if k in kwargs}
    return params or None


def _extract_tool_calls(msg: Any) -> list[dict[str, Any]] | None:
    """Extract tool calls from an OpenAI message object."""
    tool_calls = getattr(msg, "tool_calls", None)
    if not tool_calls:
        return None
    result = []
    for tc in tool_calls:
        entry: dict[str, Any] = {"id": tc.id, "type": tc.type}
        if tc.function:
            entry["function"] = {
                "name": tc.function.name,
                "arguments": tc.function.arguments,
            }
        result.append(entry)
    return result


def _extract_response(response: Any) -> tuple[Any, dict[str, int], str]:
    """Extract output, token usage, and model from a chat completion response."""
    output: Any = ""
    if response.choices:
        msg = response.choices[0].message
        tool_calls = _extract_tool_calls(msg)
        if tool_calls:
            output = {"content": msg.content or "", "tool_calls": tool_calls}
        else:
            output = msg.content or ""

    usage: dict[str, int] = {}
    if response.usage:
        usage = {
            "input_tokens": response.usage.prompt_tokens or 0,
            "output_tokens": response.usage.completion_tokens or 0,
        }

    model = response.model or ""
    return output, usage, model


def _extract_messages(kwargs: dict[str, Any]) -> list[dict[str, Any]]:
    messages = kwargs.get("messages", [])
    return [
        {"role": m.get("role", ""), "content": m.get("content", "")}
        if isinstance(m, dict)
        else {"role": getattr(m, "role", ""), "content": getattr(m, "content", "")}
        for m in messages
    ]


# ---------------------------------------------------------------------------
# Responses API helpers
# ---------------------------------------------------------------------------


def _extract_responses_output(response: Any) -> Any:
    """Extract text + tool calls from a Responses API response."""
    text = getattr(response, "output_text", "") or ""
    tool_calls = []
    for item in getattr(response, "output", []):
        if getattr(item, "type", "") == "function_call":
            tool_calls.append(
                {
                    "id": getattr(item, "call_id", ""),
                    "name": getattr(item, "name", ""),
                    "arguments": getattr(item, "arguments", ""),
                }
            )
    if tool_calls:
        return {"content": text, "tool_calls": tool_calls}
    return text


# ---------------------------------------------------------------------------
# Streaming helpers
# ---------------------------------------------------------------------------


def _aggregate_stream_chunks(chunks: list[Any]) -> tuple[Any, dict[str, int], str]:
    """Aggregate streamed ChatCompletionChunk objects into a single result."""
    text_parts: list[str] = []
    tool_calls_map: dict[int, dict[str, Any]] = {}
    usage: dict[str, int] = {}
    model = ""

    for chunk in chunks:
        if chunk.model:
            model = chunk.model

        # Token usage is sometimes on the final chunk
        if chunk.usage:
            usage = {
                "input_tokens": getattr(chunk.usage, "prompt_tokens", 0) or 0,
                "output_tokens": getattr(chunk.usage, "completion_tokens", 0) or 0,
            }

        if not chunk.choices:
            continue

        delta = chunk.choices[0].delta
        if delta is None:
            continue

        # Text content
        if delta.content:
            text_parts.append(delta.content)

        # Tool calls (streamed incrementally)
        if delta.tool_calls:
            for tc in delta.tool_calls:
                idx = tc.index
                if idx not in tool_calls_map:
                    tool_calls_map[idx] = {
                        "id": tc.id or "",
                        "type": tc.type or "function",
                        "function": {"name": "", "arguments": ""},
                    }
                entry = tool_calls_map[idx]
                if tc.id:
                    entry["id"] = tc.id
                if tc.function:
                    if tc.function.name:
                        entry["function"]["name"] = tc.function.name
                    if tc.function.arguments:
                        entry["function"]["arguments"] += tc.function.arguments

    text = "".join(text_parts)
    if tool_calls_map:
        tool_calls = [tool_calls_map[i] for i in sorted(tool_calls_map)]
        output: Any = {"content": text, "tool_calls": tool_calls}
    else:
        output = text

    return output, usage, model


class _StreamWrapper:
    """Wraps an OpenAI sync stream — accumulates chunks and writes span on close."""

    def __init__(
        self,
        stream: Any,
        span: Any,
        tracer: Any,
        capture_content: bool,
        span_ctx: Any = None,
    ) -> None:
        self._stream = stream
        self._span = span
        self._tracer = tracer
        self._capture_content = capture_content
        self._span_ctx = span_ctx
        self._chunks: list[Any] = []
        self._t0 = time.perf_counter()
        self._finalized = False

    def __iter__(self):
        return self

    def __next__(self):
        try:
            chunk = next(self._stream)
            self._chunks.append(chunk)
            if len(self._chunks) == 1:
                ttft = time.perf_counter() - self._t0
                self._span.set_metadata("time_to_first_token_s", round(ttft, 3))
            return chunk
        except StopIteration:
            self._finalize()
            raise

    def _finalize(self) -> None:
        if self._finalized:
            return
        self._finalized = True
        output, usage, model = _aggregate_stream_chunks(self._chunks)
        self._span.set_metadata("model", model)
        if usage:
            self._span.set_metadata("token_usage", usage)
        if self._capture_content:
            self._span.set_output(output)
        if self._span_ctx is not None:
            self._span_ctx.__exit__(None, None, None)
            self._span_ctx = None

    def __enter__(self):
        self._stream.__enter__()
        return self

    def __exit__(self, *args):
        self._finalize()
        return self._stream.__exit__(*args)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


class _AsyncStreamWrapper:
    """Wraps an OpenAI async stream — accumulates chunks and writes span on close."""

    def __init__(
        self,
        stream: Any,
        span: Any,
        tracer: Any,
        capture_content: bool,
        span_ctx: Any = None,
    ) -> None:
        self._stream = stream
        self._span = span
        self._tracer = tracer
        self._capture_content = capture_content
        self._span_ctx = span_ctx
        self._chunks: list[Any] = []
        self._t0 = time.perf_counter()
        self._finalized = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            chunk = await self._stream.__anext__()
            self._chunks.append(chunk)
            if len(self._chunks) == 1:
                ttft = time.perf_counter() - self._t0
                self._span.set_metadata("time_to_first_token_s", round(ttft, 3))
            return chunk
        except StopAsyncIteration:
            self._finalize()
            raise

    def _finalize(self) -> None:
        if self._finalized:
            return
        self._finalized = True
        output, usage, model = _aggregate_stream_chunks(self._chunks)
        self._span.set_metadata("model", model)
        if usage:
            self._span.set_metadata("token_usage", usage)
        if self._capture_content:
            self._span.set_output(output)
        if self._span_ctx is not None:
            self._span_ctx.__exit__(None, None, None)
            self._span_ctx = None

    async def __aenter__(self):
        await self._stream.__aenter__()
        return self

    async def __aexit__(self, *args):
        self._finalize()
        return await self._stream.__aexit__(*args)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


# ---------------------------------------------------------------------------
# Responses API stream wrappers
# ---------------------------------------------------------------------------


class _ResponsesStreamWrapper:
    """Wraps an OpenAI Responses API sync stream."""

    def __init__(
        self,
        stream: Any,
        span: Any,
        tracer: Any,
        capture_content: bool,
        span_ctx: Any = None,
    ) -> None:
        self._stream = stream
        self._span = span
        self._tracer = tracer
        self._capture_content = capture_content
        self._span_ctx = span_ctx
        self._events: list[Any] = []
        self._t0 = time.perf_counter()
        self._got_first_text = False
        self._finalized = False

    def __iter__(self):
        return self

    def __next__(self):
        try:
            event = next(self._stream)
            self._events.append(event)
            self._check_ttft(event)
            self._check_completed(event)
            return event
        except StopIteration:
            self._finalize()
            raise

    def _check_ttft(self, event: Any) -> None:
        if self._got_first_text:
            return
        etype = getattr(event, "type", "")
        if etype == "response.output_text.delta":
            self._got_first_text = True
            ttft = time.perf_counter() - self._t0
            self._span.set_metadata("time_to_first_token_s", round(ttft, 3))

    def _check_completed(self, event: Any) -> None:
        etype = getattr(event, "type", "")
        if etype == "response.completed":
            response = getattr(event, "response", None)
            if response:
                usage = getattr(response, "usage", None)
                if usage:
                    self._span.set_metadata(
                        "token_usage",
                        {
                            "input_tokens": getattr(usage, "input_tokens", 0) or 0,
                            "output_tokens": getattr(usage, "output_tokens", 0) or 0,
                        },
                    )
                model = getattr(response, "model", "")
                if model:
                    self._span.set_metadata("model", model)
                if self._capture_content:
                    self._span.set_output(_extract_responses_output(response))

    def _finalize(self) -> None:
        if self._finalized:
            return
        self._finalized = True
        if self._span_ctx is not None:
            self._span_ctx.__exit__(None, None, None)
            self._span_ctx = None

    def __enter__(self):
        self._stream.__enter__()
        return self

    def __exit__(self, *args):
        self._finalize()
        return self._stream.__exit__(*args)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


class _AsyncResponsesStreamWrapper:
    """Wraps an OpenAI Responses API async stream."""

    def __init__(
        self,
        stream: Any,
        span: Any,
        tracer: Any,
        capture_content: bool,
        span_ctx: Any = None,
    ) -> None:
        self._stream = stream
        self._span = span
        self._tracer = tracer
        self._capture_content = capture_content
        self._span_ctx = span_ctx
        self._events: list[Any] = []
        self._t0 = time.perf_counter()
        self._got_first_text = False
        self._finalized = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            event = await self._stream.__anext__()
            self._events.append(event)
            self._check_ttft(event)
            self._check_completed(event)
            return event
        except StopAsyncIteration:
            self._finalize()
            raise

    def _check_ttft(self, event: Any) -> None:
        if self._got_first_text:
            return
        etype = getattr(event, "type", "")
        if etype == "response.output_text.delta":
            self._got_first_text = True
            ttft = time.perf_counter() - self._t0
            self._span.set_metadata("time_to_first_token_s", round(ttft, 3))

    def _check_completed(self, event: Any) -> None:
        etype = getattr(event, "type", "")
        if etype == "response.completed":
            response = getattr(event, "response", None)
            if response:
                usage = getattr(response, "usage", None)
                if usage:
                    self._span.set_metadata(
                        "token_usage",
                        {
                            "input_tokens": getattr(usage, "input_tokens", 0) or 0,
                            "output_tokens": getattr(usage, "output_tokens", 0) or 0,
                        },
                    )
                model = getattr(response, "model", "")
                if model:
                    self._span.set_metadata("model", model)
                if self._capture_content:
                    self._span.set_output(_extract_responses_output(response))

    def _finalize(self) -> None:
        if self._finalized:
            return
        self._finalized = True
        if self._span_ctx is not None:
            self._span_ctx.__exit__(None, None, None)
            self._span_ctx = None

    async def __aenter__(self):
        await self._stream.__aenter__()
        return self

    async def __aexit__(self, *args):
        self._finalize()
        return await self._stream.__aexit__(*args)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


# ---------------------------------------------------------------------------
# Traced wrapper classes
# ---------------------------------------------------------------------------


class _TracedCompletions:
    def __init__(self, completions: Any, operation: str) -> None:
        self._completions = completions
        self._operation = operation

    def create(self, **kwargs: Any) -> Any:
        tracer = get_tracer()
        if tracer is None:
            return self._completions.create(**kwargs)

        span_meta: dict[str, Any] = {"provider": "openai"}
        if self._operation:
            span_meta["operation"] = self._operation
        model_params = _extract_model_params(kwargs, _CHAT_MODEL_PARAMS)
        if model_params:
            span_meta["model_parameters"] = model_params
        input_data = _extract_messages(kwargs) if tracer.capture_content else None

        is_stream = kwargs.get("stream", False)

        if is_stream:
            # For streaming: open span, return wrapper that finalizes on exhaustion.
            # We use span as context manager manually.
            span_ctx = tracer.span(
                self._operation or "openai.chat.completions.create",
                input=input_data,
                metadata=span_meta,
                kind="llm",
            )
            span = span_ctx.__enter__()
            try:
                # Pass stream_options to get usage in the final chunk
                if "stream_options" not in kwargs:
                    kwargs["stream_options"] = {"include_usage": True}
                stream = self._completions.create(**kwargs)
                return _StreamWrapper(
                    stream, span, tracer, tracer.capture_content, span_ctx
                )
            except BaseException:
                span_ctx.__exit__(*sys.exc_info())
                raise
        else:
            with tracer.span(
                self._operation or "openai.chat.completions.create",
                input=input_data,
                metadata=span_meta,
                kind="llm",
            ) as span:
                response = self._completions.create(**kwargs)
                output, usage, model = _extract_response(response)
                span.set_metadata("model", model)
                if usage:
                    span.set_metadata("token_usage", usage)
                if tracer.capture_content:
                    span.set_output(output)
                return response

    def __getattr__(self, name: str) -> Any:
        return getattr(self._completions, name)


class _TracedAsyncCompletions:
    def __init__(self, completions: Any, operation: str) -> None:
        self._completions = completions
        self._operation = operation

    async def create(self, **kwargs: Any) -> Any:
        tracer = get_tracer()
        if tracer is None:
            return await self._completions.create(**kwargs)

        span_meta: dict[str, Any] = {"provider": "openai"}
        if self._operation:
            span_meta["operation"] = self._operation
        model_params = _extract_model_params(kwargs, _CHAT_MODEL_PARAMS)
        if model_params:
            span_meta["model_parameters"] = model_params
        input_data = _extract_messages(kwargs) if tracer.capture_content else None

        is_stream = kwargs.get("stream", False)

        if is_stream:
            span_ctx = tracer.span(
                self._operation or "openai.chat.completions.create",
                input=input_data,
                metadata=span_meta,
                kind="llm",
            )
            span = span_ctx.__enter__()
            try:
                if "stream_options" not in kwargs:
                    kwargs["stream_options"] = {"include_usage": True}
                stream = await self._completions.create(**kwargs)
                return _AsyncStreamWrapper(
                    stream, span, tracer, tracer.capture_content, span_ctx
                )
            except BaseException:
                span_ctx.__exit__(*sys.exc_info())
                raise
        else:
            with tracer.span(
                self._operation or "openai.chat.completions.create",
                input=input_data,
                metadata=span_meta,
                kind="llm",
            ) as span:
                response = await self._completions.create(**kwargs)
                output, usage, model = _extract_response(response)
                span.set_metadata("model", model)
                if usage:
                    span.set_metadata("token_usage", usage)
                if tracer.capture_content:
                    span.set_output(output)
                return response

    def __getattr__(self, name: str) -> Any:
        return getattr(self._completions, name)


class _TracedChat:
    def __init__(self, chat: Any, operation: str, is_async: bool = False) -> None:
        self._chat = chat
        self._operation = operation
        self._is_async = is_async

    @property
    def completions(self) -> _TracedCompletions | _TracedAsyncCompletions:
        if self._is_async:
            return _TracedAsyncCompletions(self._chat.completions, self._operation)
        return _TracedCompletions(self._chat.completions, self._operation)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._chat, name)


# ---------------------------------------------------------------------------
# Embeddings wrapper
# ---------------------------------------------------------------------------


class _TracedEmbeddings:
    def __init__(self, embeddings: Any, operation: str) -> None:
        self._embeddings = embeddings
        self._operation = operation

    def create(self, **kwargs: Any) -> Any:
        tracer = get_tracer()
        if tracer is None:
            return self._embeddings.create(**kwargs)

        span_meta: dict[str, Any] = {"provider": "openai"}
        if self._operation:
            span_meta["operation"] = self._operation
        input_data = kwargs.get("input") if tracer.capture_content else None

        with tracer.span(
            self._operation or "openai.embeddings.create",
            input=input_data,
            metadata=span_meta,
            kind="embedding",
        ) as span:
            response = self._embeddings.create(**kwargs)
            model = getattr(response, "model", "") or ""
            span.set_metadata("model", model)
            usage = getattr(response, "usage", None)
            if usage:
                span.set_metadata(
                    "token_usage",
                    {
                        "input_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                    },
                )
            return response

    def __getattr__(self, name: str) -> Any:
        return getattr(self._embeddings, name)


class _TracedAsyncEmbeddings:
    def __init__(self, embeddings: Any, operation: str) -> None:
        self._embeddings = embeddings
        self._operation = operation

    async def create(self, **kwargs: Any) -> Any:
        tracer = get_tracer()
        if tracer is None:
            return await self._embeddings.create(**kwargs)

        span_meta: dict[str, Any] = {"provider": "openai"}
        if self._operation:
            span_meta["operation"] = self._operation
        input_data = kwargs.get("input") if tracer.capture_content else None

        with tracer.span(
            self._operation or "openai.embeddings.create",
            input=input_data,
            metadata=span_meta,
            kind="embedding",
        ) as span:
            response = await self._embeddings.create(**kwargs)
            model = getattr(response, "model", "") or ""
            span.set_metadata("model", model)
            usage = getattr(response, "usage", None)
            if usage:
                span.set_metadata(
                    "token_usage",
                    {
                        "input_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                    },
                )
            return response

    def __getattr__(self, name: str) -> Any:
        return getattr(self._embeddings, name)


# ---------------------------------------------------------------------------
# Responses API wrapper
# ---------------------------------------------------------------------------


class _TracedResponses:
    def __init__(self, responses: Any, operation: str) -> None:
        self._responses = responses
        self._operation = operation

    def create(self, **kwargs: Any) -> Any:
        tracer = get_tracer()
        if tracer is None:
            return self._responses.create(**kwargs)

        span_meta: dict[str, Any] = {"provider": "openai"}
        if self._operation:
            span_meta["operation"] = self._operation
        model_params = _extract_model_params(kwargs, _CHAT_MODEL_PARAMS)
        if model_params:
            span_meta["model_parameters"] = model_params

        input_data = None
        if tracer.capture_content:
            input_parts: dict[str, Any] = {}
            if "input" in kwargs:
                input_parts["input"] = kwargs["input"]
            if "instructions" in kwargs:
                input_parts["instructions"] = kwargs["instructions"]
            input_data = input_parts or None

        is_stream = kwargs.get("stream", False)

        if is_stream:
            span_ctx = tracer.span(
                self._operation or "openai.responses.create",
                input=input_data,
                metadata=span_meta,
                kind="llm",
            )
            span = span_ctx.__enter__()
            try:
                stream = self._responses.create(**kwargs)
                return _ResponsesStreamWrapper(
                    stream, span, tracer, tracer.capture_content, span_ctx
                )
            except BaseException:
                span_ctx.__exit__(*sys.exc_info())
                raise
        else:
            with tracer.span(
                self._operation or "openai.responses.create",
                input=input_data,
                metadata=span_meta,
                kind="llm",
            ) as span:
                response = self._responses.create(**kwargs)
                model = getattr(response, "model", "") or ""
                span.set_metadata("model", model)
                usage = getattr(response, "usage", None)
                if usage:
                    span.set_metadata(
                        "token_usage",
                        {
                            "input_tokens": getattr(usage, "input_tokens", 0) or 0,
                            "output_tokens": getattr(usage, "output_tokens", 0) or 0,
                        },
                    )
                if tracer.capture_content:
                    span.set_output(_extract_responses_output(response))
                return response

    def __getattr__(self, name: str) -> Any:
        return getattr(self._responses, name)


class _TracedAsyncResponses:
    def __init__(self, responses: Any, operation: str) -> None:
        self._responses = responses
        self._operation = operation

    async def create(self, **kwargs: Any) -> Any:
        tracer = get_tracer()
        if tracer is None:
            return await self._responses.create(**kwargs)

        span_meta: dict[str, Any] = {"provider": "openai"}
        if self._operation:
            span_meta["operation"] = self._operation
        model_params = _extract_model_params(kwargs, _CHAT_MODEL_PARAMS)
        if model_params:
            span_meta["model_parameters"] = model_params

        input_data = None
        if tracer.capture_content:
            input_parts: dict[str, Any] = {}
            if "input" in kwargs:
                input_parts["input"] = kwargs["input"]
            if "instructions" in kwargs:
                input_parts["instructions"] = kwargs["instructions"]
            input_data = input_parts or None

        is_stream = kwargs.get("stream", False)

        if is_stream:
            span_ctx = tracer.span(
                self._operation or "openai.responses.create",
                input=input_data,
                metadata=span_meta,
                kind="llm",
            )
            span = span_ctx.__enter__()
            try:
                stream = await self._responses.create(**kwargs)
                return _AsyncResponsesStreamWrapper(
                    stream, span, tracer, tracer.capture_content, span_ctx
                )
            except BaseException:
                span_ctx.__exit__(*sys.exc_info())
                raise
        else:
            with tracer.span(
                self._operation or "openai.responses.create",
                input=input_data,
                metadata=span_meta,
                kind="llm",
            ) as span:
                response = await self._responses.create(**kwargs)
                model = getattr(response, "model", "") or ""
                span.set_metadata("model", model)
                usage = getattr(response, "usage", None)
                if usage:
                    span.set_metadata(
                        "token_usage",
                        {
                            "input_tokens": getattr(usage, "input_tokens", 0) or 0,
                            "output_tokens": getattr(usage, "output_tokens", 0) or 0,
                        },
                    )
                if tracer.capture_content:
                    span.set_output(_extract_responses_output(response))
                return response

    def __getattr__(self, name: str) -> Any:
        return getattr(self._responses, name)


# ---------------------------------------------------------------------------
# Top-level traced client
# ---------------------------------------------------------------------------


class _TracedOpenAIClient:
    def __init__(self, client: Any, operation: str) -> None:
        self._client = client
        self._operation = operation
        self._is_async = isinstance(client, _openai_mod.AsyncOpenAI)

    @property
    def chat(self) -> _TracedChat:
        return _TracedChat(self._client.chat, self._operation, self._is_async)

    @property
    def embeddings(self) -> _TracedEmbeddings | _TracedAsyncEmbeddings:
        if self._is_async:
            return _TracedAsyncEmbeddings(self._client.embeddings, self._operation)
        return _TracedEmbeddings(self._client.embeddings, self._operation)

    @property
    def responses(self) -> _TracedResponses | _TracedAsyncResponses:
        if self._is_async:
            return _TracedAsyncResponses(self._client.responses, self._operation)
        return _TracedResponses(self._client.responses, self._operation)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


def traced_openai(client: Any, operation: str = "") -> _TracedOpenAIClient:
    """Wrap an OpenAI client to auto-trace chat completion calls as spans."""
    return _TracedOpenAIClient(client, operation)
