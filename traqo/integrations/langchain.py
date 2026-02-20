"""LangChain integration — wrap BaseChatModel to auto-trace LLM calls."""

from __future__ import annotations

import time
from typing import Any

try:
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage, BaseMessage
    from langchain_core.outputs import ChatGeneration, ChatResult
except ImportError:
    raise ImportError(
        "LangChain not installed. Install with: pip install traqo[langchain]"
    )

from traqo.tracer import get_tracer


def _extract_token_usage(result: ChatResult) -> dict[str, int]:
    """Extract token usage from ChatResult, checking both llm_output and usage_metadata."""
    usage: dict[str, int] = {}

    # Check llm_output (OpenAI-style)
    llm_output = result.llm_output or {}
    token_usage = llm_output.get("token_usage", {})
    if token_usage:
        usage["input_tokens"] = token_usage.get("prompt_tokens", 0)
        usage["output_tokens"] = token_usage.get("completion_tokens", 0)

    # Check usage_metadata on the message (newer LangChain)
    if result.generations:
        gen = result.generations[0]
        if isinstance(gen, ChatGeneration) and isinstance(gen.message, AIMessage):
            meta = getattr(gen.message, "usage_metadata", None)
            if meta:
                usage["input_tokens"] = getattr(meta, "input_tokens", 0) or meta.get("input_tokens", 0) if isinstance(meta, dict) else getattr(meta, "input_tokens", 0)
                usage["output_tokens"] = getattr(meta, "output_tokens", 0) or meta.get("output_tokens", 0) if isinstance(meta, dict) else getattr(meta, "output_tokens", 0)

    return usage


def _extract_output_text(result: ChatResult) -> str:
    """Extract the text content from a ChatResult."""
    if result.generations:
        gen = result.generations[0]
        if isinstance(gen, ChatGeneration) and gen.message:
            return gen.message.content
        return gen.text
    return ""


def _messages_to_dicts(messages: list[BaseMessage]) -> list[dict[str, Any]]:
    """Convert LangChain messages to plain dicts for logging."""
    result = []
    for msg in messages:
        d: dict[str, Any] = {"role": msg.type, "content": msg.content}
        result.append(d)
    return result


def _extract_model_name(model: BaseChatModel) -> str:
    """Extract the model name string from a BaseChatModel."""
    if hasattr(model, "model_name"):
        return model.model_name
    if hasattr(model, "model"):
        return model.model
    return type(model).__name__


class TracedChatModel(BaseChatModel):
    """Wrapper around a BaseChatModel that logs llm_call events to the active tracer."""

    wrapped: BaseChatModel
    operation: str = ""

    class Config:
        arbitrary_types_allowed = True

    @property
    def _llm_type(self) -> str:
        return self.wrapped._llm_type

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return self.wrapped._identifying_params

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        start = time.monotonic()
        result = self.wrapped._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        duration = time.monotonic() - start

        tracer = get_tracer()
        if tracer:
            tracer.llm_event(
                model=_extract_model_name(self.wrapped),
                input_messages=_messages_to_dicts(messages),
                output_text=_extract_output_text(result),
                token_usage=_extract_token_usage(result),
                duration_s=duration,
                operation=self.operation or None,
            )
        return result

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        start = time.monotonic()
        result = await self.wrapped._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)
        duration = time.monotonic() - start

        tracer = get_tracer()
        if tracer:
            tracer.llm_event(
                model=_extract_model_name(self.wrapped),
                input_messages=_messages_to_dicts(messages),
                output_text=_extract_output_text(result),
                token_usage=_extract_token_usage(result),
                duration_s=duration,
                operation=self.operation or None,
            )
        return result

    def __getattr__(self, name: str) -> Any:
        if name in ("wrapped", "operation"):
            raise AttributeError(name)
        return getattr(self.wrapped, name)


def traced_model(model: BaseChatModel, operation: str = "") -> TracedChatModel:
    """Wrap a LangChain chat model to auto-trace LLM calls.

    Returns a TracedChatModel that behaves identically to the original,
    except it logs llm_call events to the active traqo tracer.
    """
    return TracedChatModel(wrapped=model, operation=operation)
