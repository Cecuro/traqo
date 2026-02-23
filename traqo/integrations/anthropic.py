"""Anthropic integration — wrap Anthropic client to auto-trace LLM calls as spans."""

from __future__ import annotations

from typing import Any

try:
    import anthropic as _anthropic_mod
except ImportError:
    raise ImportError(
        "Anthropic not installed. Install with: pip install traqo[anthropic]"
    )

from traqo.tracer import get_tracer


def _extract_tool_use(content: list[Any]) -> list[dict[str, Any]] | None:
    """Extract tool_use blocks from Anthropic message content."""
    tool_uses = []
    for block in content:
        block_type = getattr(block, "type", None)
        if block_type == "tool_use":
            tool_uses.append({
                "id": getattr(block, "id", ""),
                "name": getattr(block, "name", ""),
                "input": getattr(block, "input", {}),
            })
    return tool_uses or None


def _extract_response(response: Any) -> tuple[Any, dict[str, int], str]:
    """Extract output, token usage, and model from an Anthropic response."""
    output: Any = ""
    if response.content:
        text_blocks = [b.text for b in response.content if hasattr(b, "text")]
        text = "\n".join(text_blocks)
        tool_uses = _extract_tool_use(response.content)
        if tool_uses:
            output = {"content": text, "tool_use": tool_uses}
        else:
            output = text

    usage: dict[str, int] = {}
    if response.usage:
        usage = {
            "input_tokens": response.usage.input_tokens or 0,
            "output_tokens": response.usage.output_tokens or 0,
        }
        # Anthropic cache tokens
        cache_read = getattr(response.usage, "cache_read_input_tokens", None)
        cache_creation = getattr(response.usage, "cache_creation_input_tokens", None)
        if cache_read:
            usage["cache_read_input_tokens"] = cache_read
        if cache_creation:
            usage["cache_creation_input_tokens"] = cache_creation

    model = response.model or ""
    return output, usage, model


def _extract_messages(kwargs: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    system = kwargs.get("system")
    if system:
        result.append({"role": "system", "content": system})
    messages = kwargs.get("messages", [])
    for m in messages:
        if isinstance(m, dict):
            result.append({"role": m.get("role", ""), "content": m.get("content", "")})
        else:
            result.append({"role": getattr(m, "role", ""), "content": getattr(m, "content", "")})
    return result


# ---------------------------------------------------------------------------
# Streaming helpers
# ---------------------------------------------------------------------------

def _aggregate_stream_events(events: list[Any]) -> tuple[Any, dict[str, int], str]:
    """Aggregate Anthropic stream events into a single result."""
    text_parts: list[str] = []
    tool_uses: list[dict[str, Any]] = []
    current_tool: dict[str, Any] | None = None
    usage: dict[str, int] = {}
    model = ""

    for event in events:
        event_type = getattr(event, "type", "")

        if event_type == "message_start":
            msg = getattr(event, "message", None)
            if msg:
                model = getattr(msg, "model", "") or ""
                msg_usage = getattr(msg, "usage", None)
                if msg_usage:
                    usage["input_tokens"] = getattr(msg_usage, "input_tokens", 0) or 0

        elif event_type == "content_block_start":
            block = getattr(event, "content_block", None)
            if block and getattr(block, "type", "") == "tool_use":
                current_tool = {
                    "id": getattr(block, "id", ""),
                    "name": getattr(block, "name", ""),
                    "input": "",
                }

        elif event_type == "content_block_delta":
            delta = getattr(event, "delta", None)
            if delta:
                delta_type = getattr(delta, "type", "")
                if delta_type == "text_delta":
                    text_parts.append(getattr(delta, "text", ""))
                elif delta_type == "input_json_delta" and current_tool is not None:
                    current_tool["input"] += getattr(delta, "partial_json", "")

        elif event_type == "content_block_stop":
            if current_tool is not None:
                # Try to parse accumulated JSON input
                raw = current_tool["input"]
                try:
                    import json
                    current_tool["input"] = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    pass  # Keep as string
                tool_uses.append(current_tool)
                current_tool = None

        elif event_type == "message_delta":
            delta = getattr(event, "delta", None)
            msg_usage = getattr(event, "usage", None)
            if msg_usage:
                usage["output_tokens"] = getattr(msg_usage, "output_tokens", 0) or 0

    text = "".join(text_parts)
    if tool_uses:
        output: Any = {"content": text, "tool_use": tool_uses}
    else:
        output = text

    return output, usage, model


class _StreamWrapper:
    """Wraps an Anthropic sync stream — accumulates events and writes span on close."""

    def __init__(self, stream: Any, span: Any, tracer: Any, capture_content: bool) -> None:
        self._stream = stream
        self._span = span
        self._tracer = tracer
        self._capture_content = capture_content
        self._events: list[Any] = []

    def __iter__(self):
        return self

    def __next__(self):
        try:
            event = next(self._stream)
            self._events.append(event)
            return event
        except StopIteration:
            self._finalize()
            raise

    def _finalize(self) -> None:
        output, usage, model = _aggregate_stream_events(self._events)
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
    """Wraps an Anthropic async stream — accumulates events and writes span on close."""

    def __init__(self, stream: Any, span: Any, tracer: Any, capture_content: bool) -> None:
        self._stream = stream
        self._span = span
        self._tracer = tracer
        self._capture_content = capture_content
        self._events: list[Any] = []

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            event = await self._stream.__anext__()
            self._events.append(event)
            return event
        except StopAsyncIteration:
            self._finalize()
            raise

    def _finalize(self) -> None:
        output, usage, model = _aggregate_stream_events(self._events)
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

class _TracedMessages:
    def __init__(self, messages: Any, operation: str) -> None:
        self._messages = messages
        self._operation = operation

    def create(self, **kwargs: Any) -> Any:
        tracer = get_tracer()
        if tracer is None:
            return self._messages.create(**kwargs)

        span_meta: dict[str, Any] = {"provider": "anthropic"}
        if self._operation:
            span_meta["operation"] = self._operation
        input_data = _extract_messages(kwargs) if tracer._capture_content else None

        is_stream = kwargs.get("stream", False)

        if is_stream:
            span_ctx = tracer.span(
                self._operation or "anthropic.messages.create",
                input=input_data,
                metadata=span_meta,
                kind="llm",
            )
            span = span_ctx.__enter__()
            try:
                stream = self._messages.create(**kwargs)
                return _StreamWrapper(stream, span, tracer, tracer._capture_content)
            except BaseException:
                span_ctx.__exit__(*__import__("sys").exc_info())
                raise
        else:
            with tracer.span(
                self._operation or "anthropic.messages.create",
                input=input_data,
                metadata=span_meta,
                kind="llm",
            ) as span:
                response = self._messages.create(**kwargs)
                output, usage, model = _extract_response(response)
                span.set_metadata("model", model)
                if usage:
                    span.set_metadata("token_usage", usage)
                if tracer._capture_content:
                    span.set_output(output)
                return response

    def stream(self, **kwargs: Any) -> Any:
        """Wrap Anthropic's messages.stream() context manager."""
        tracer = get_tracer()
        if tracer is None:
            return self._messages.stream(**kwargs)

        span_meta: dict[str, Any] = {"provider": "anthropic"}
        if self._operation:
            span_meta["operation"] = self._operation
        input_data = _extract_messages(kwargs) if tracer._capture_content else None

        span_ctx = tracer.span(
            self._operation or "anthropic.messages.stream",
            input=input_data,
            metadata=span_meta,
            kind="llm",
        )
        span = span_ctx.__enter__()
        try:
            stream = self._messages.stream(**kwargs)
            return _StreamWrapper(stream, span, tracer, tracer._capture_content)
        except BaseException:
            span_ctx.__exit__(*__import__("sys").exc_info())
            raise

    def __getattr__(self, name: str) -> Any:
        return getattr(self._messages, name)


class _TracedAsyncMessages:
    def __init__(self, messages: Any, operation: str) -> None:
        self._messages = messages
        self._operation = operation

    async def create(self, **kwargs: Any) -> Any:
        tracer = get_tracer()
        if tracer is None:
            return await self._messages.create(**kwargs)

        span_meta: dict[str, Any] = {"provider": "anthropic"}
        if self._operation:
            span_meta["operation"] = self._operation
        input_data = _extract_messages(kwargs) if tracer._capture_content else None

        is_stream = kwargs.get("stream", False)

        if is_stream:
            span_ctx = tracer.span(
                self._operation or "anthropic.messages.create",
                input=input_data,
                metadata=span_meta,
                kind="llm",
            )
            span = span_ctx.__enter__()
            try:
                stream = await self._messages.create(**kwargs)
                return _AsyncStreamWrapper(stream, span, tracer, tracer._capture_content)
            except BaseException:
                span_ctx.__exit__(*__import__("sys").exc_info())
                raise
        else:
            with tracer.span(
                self._operation or "anthropic.messages.create",
                input=input_data,
                metadata=span_meta,
                kind="llm",
            ) as span:
                response = await self._messages.create(**kwargs)
                output, usage, model = _extract_response(response)
                span.set_metadata("model", model)
                if usage:
                    span.set_metadata("token_usage", usage)
                if tracer._capture_content:
                    span.set_output(output)
                return response

    async def stream(self, **kwargs: Any) -> Any:
        """Wrap Anthropic's async messages.stream() context manager."""
        tracer = get_tracer()
        if tracer is None:
            return await self._messages.stream(**kwargs)

        span_meta: dict[str, Any] = {"provider": "anthropic"}
        if self._operation:
            span_meta["operation"] = self._operation
        input_data = _extract_messages(kwargs) if tracer._capture_content else None

        span_ctx = tracer.span(
            self._operation or "anthropic.messages.stream",
            input=input_data,
            metadata=span_meta,
            kind="llm",
        )
        span = span_ctx.__enter__()
        try:
            stream = await self._messages.stream(**kwargs)
            return _AsyncStreamWrapper(stream, span, tracer, tracer._capture_content)
        except BaseException:
            span_ctx.__exit__(*__import__("sys").exc_info())
            raise

    def __getattr__(self, name: str) -> Any:
        return getattr(self._messages, name)


class _TracedAnthropicClient:
    def __init__(self, client: Any, operation: str) -> None:
        self._client = client
        self._operation = operation
        self._is_async = isinstance(client, _anthropic_mod.AsyncAnthropic)

    @property
    def messages(self) -> _TracedMessages | _TracedAsyncMessages:
        if self._is_async:
            return _TracedAsyncMessages(self._client.messages, self._operation)
        return _TracedMessages(self._client.messages, self._operation)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


def traced_anthropic(client: Any, operation: str = "") -> _TracedAnthropicClient:
    """Wrap an Anthropic client to auto-trace message creation calls as spans."""
    return _TracedAnthropicClient(client, operation)
