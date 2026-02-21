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


def _extract_response(response: Any) -> tuple[str, dict[str, int], str]:
    text = ""
    if response.choices:
        msg = response.choices[0].message
        text = msg.content or ""

    usage: dict[str, int] = {}
    if response.usage:
        usage = {
            "input_tokens": response.usage.prompt_tokens or 0,
            "output_tokens": response.usage.completion_tokens or 0,
        }

    model = response.model or ""
    return text, usage, model


def _extract_messages(kwargs: dict[str, Any]) -> list[dict[str, Any]]:
    messages = kwargs.get("messages", [])
    return [
        {"role": m.get("role", ""), "content": m.get("content", "")}
        if isinstance(m, dict)
        else {"role": getattr(m, "role", ""), "content": getattr(m, "content", "")}
        for m in messages
    ]


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

        with tracer.span(
            self._operation or "openai.chat.completions.create",
            input=input_data,
            metadata=span_meta,
            kind="llm",
        ) as span:
            response = self._completions.create(**kwargs)
            text, usage, model = _extract_response(response)
            span.set_metadata("model", model)
            if usage:
                span.set_metadata("token_usage", usage)
            if tracer._capture_content:
                span.set_output(text)
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

        with tracer.span(
            self._operation or "openai.chat.completions.create",
            input=input_data,
            metadata=span_meta,
            kind="llm",
        ) as span:
            response = await self._completions.create(**kwargs)
            text, usage, model = _extract_response(response)
            span.set_metadata("model", model)
            if usage:
                span.set_metadata("token_usage", usage)
            if tracer._capture_content:
                span.set_output(text)
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
