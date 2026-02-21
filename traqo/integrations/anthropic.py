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


def _extract_response(response: Any) -> tuple[str, dict[str, int], str]:
    text = ""
    if response.content:
        text_blocks = [b.text for b in response.content if hasattr(b, "text")]
        text = "\n".join(text_blocks)

    usage: dict[str, int] = {}
    if response.usage:
        usage = {
            "input_tokens": response.usage.input_tokens or 0,
            "output_tokens": response.usage.output_tokens or 0,
        }

    model = response.model or ""
    return text, usage, model


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

        with tracer.span(
            self._operation or "anthropic.messages.create",
            input=input_data,
            metadata=span_meta,
            kind="llm",
        ) as span:
            response = self._messages.create(**kwargs)
            text, usage, model = _extract_response(response)
            span.set_metadata("model", model)
            if usage:
                span.set_metadata("token_usage", usage)
            if tracer._capture_content:
                span.set_output(text)
            return response

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

        with tracer.span(
            self._operation or "anthropic.messages.create",
            input=input_data,
            metadata=span_meta,
            kind="llm",
        ) as span:
            response = await self._messages.create(**kwargs)
            text, usage, model = _extract_response(response)
            span.set_metadata("model", model)
            if usage:
                span.set_metadata("token_usage", usage)
            if tracer._capture_content:
                span.set_output(text)
            return response

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
