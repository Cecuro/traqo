"""Anthropic integration — wrap Anthropic client to auto-trace LLM calls."""

from __future__ import annotations

import time
from typing import Any

try:
    import anthropic as _anthropic_mod
except ImportError:
    raise ImportError(
        "Anthropic not installed. Install with: pip install traqo[anthropic]"
    )

from traqo.tracer import get_tracer


def _extract_response(response: Any) -> tuple[str, dict[str, int], str]:
    """Extract text, token usage, and model from an Anthropic response."""
    text = ""
    if response.content:
        # Anthropic returns a list of content blocks
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
    """Extract messages from create() kwargs, including system prompt."""
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
    """Wrapper around client.messages that traces create() calls."""

    def __init__(self, messages: Any, operation: str) -> None:
        self._messages = messages
        self._operation = operation

    def create(self, **kwargs: Any) -> Any:
        input_messages = _extract_messages(kwargs)
        start = time.monotonic()
        response = self._messages.create(**kwargs)
        duration = time.monotonic() - start

        tracer = get_tracer()
        if tracer:
            text, usage, model = _extract_response(response)
            tracer.llm_event(
                model=model,
                input_messages=input_messages,
                output_text=text,
                token_usage=usage,
                duration_s=duration,
                operation=self._operation or None,
            )
        return response

    def __getattr__(self, name: str) -> Any:
        return getattr(self._messages, name)


class _TracedAsyncMessages:
    """Wrapper around async client.messages that traces create() calls."""

    def __init__(self, messages: Any, operation: str) -> None:
        self._messages = messages
        self._operation = operation

    async def create(self, **kwargs: Any) -> Any:
        input_messages = _extract_messages(kwargs)
        start = time.monotonic()
        response = await self._messages.create(**kwargs)
        duration = time.monotonic() - start

        tracer = get_tracer()
        if tracer:
            text, usage, model = _extract_response(response)
            tracer.llm_event(
                model=model,
                input_messages=input_messages,
                output_text=text,
                token_usage=usage,
                duration_s=duration,
                operation=self._operation or None,
            )
        return response

    def __getattr__(self, name: str) -> Any:
        return getattr(self._messages, name)


class _TracedAnthropicClient:
    """Wrapper around Anthropic client that traces messages.create()."""

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


def traced_anthropic(
    client: Any,
    operation: str = "",
) -> _TracedAnthropicClient:
    """Wrap an Anthropic client to auto-trace message creation calls.

    Works with both sync (Anthropic) and async (AsyncAnthropic) clients.
    """
    return _TracedAnthropicClient(client, operation)
