"""OpenAI integration — wrap OpenAI client to auto-trace LLM calls as spans."""

from __future__ import annotations

from typing import Any

try:
    import openai as _openai_mod
except ImportError:
    raise ImportError(
        "OpenAI not installed. Install with: pip install traqo[openai]"
    )

from traqo.tracer import get_tracer


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

    def __init__(self, stream: Any, span: Any, tracer: Any, capture_content: bool) -> None:
        self._stream = stream
        self._span = span
        self._tracer = tracer
        self._capture_content = capture_content
        self._chunks: list[Any] = []

    def __iter__(self):
        return self

    def __next__(self):
        try:
            chunk = next(self._stream)
            self._chunks.append(chunk)
            return chunk
        except StopIteration:
            self._finalize()
            raise

    def _finalize(self) -> None:
        output, usage, model = _aggregate_stream_chunks(self._chunks)
        self._span.set_metadata("model", model)
        if usage:
            self._span.set_metadata("token_usage", usage)
        if self._capture_content:
            self._span.set_output(output)

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

    def __init__(self, stream: Any, span: Any, tracer: Any, capture_content: bool) -> None:
        self._stream = stream
        self._span = span
        self._tracer = tracer
        self._capture_content = capture_content
        self._chunks: list[Any] = []

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            chunk = await self._stream.__anext__()
            self._chunks.append(chunk)
            return chunk
        except StopAsyncIteration:
            self._finalize()
            raise

    def _finalize(self) -> None:
        output, usage, model = _aggregate_stream_chunks(self._chunks)
        self._span.set_metadata("model", model)
        if usage:
            self._span.set_metadata("token_usage", usage)
        if self._capture_content:
            self._span.set_output(output)

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
        input_data = _extract_messages(kwargs) if tracer._capture_content else None

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
                return _StreamWrapper(stream, span, tracer, tracer._capture_content)
            except BaseException:
                span_ctx.__exit__(*__import__("sys").exc_info())
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
                if tracer._capture_content:
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
        input_data = _extract_messages(kwargs) if tracer._capture_content else None

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
                return _AsyncStreamWrapper(stream, span, tracer, tracer._capture_content)
            except BaseException:
                span_ctx.__exit__(*__import__("sys").exc_info())
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
                if tracer._capture_content:
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


class _TracedOpenAIClient:
    def __init__(self, client: Any, operation: str) -> None:
        self._client = client
        self._operation = operation
        self._is_async = isinstance(client, _openai_mod.AsyncOpenAI)

    @property
    def chat(self) -> _TracedChat:
        return _TracedChat(self._client.chat, self._operation, self._is_async)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


def traced_openai(client: Any, operation: str = "") -> _TracedOpenAIClient:
    """Wrap an OpenAI client to auto-trace chat completion calls as spans."""
    return _TracedOpenAIClient(client, operation)
