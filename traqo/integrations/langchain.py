"""LangChain integration — wrap BaseChatModel to auto-trace LLM calls as spans."""

from __future__ import annotations

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
    usage: dict[str, int] = {}
    llm_output = result.llm_output or {}
    token_usage = llm_output.get("token_usage", {})
    if token_usage:
        usage["input_tokens"] = token_usage.get("prompt_tokens", 0)
        usage["output_tokens"] = token_usage.get("completion_tokens", 0)

    if result.generations:
        gen = result.generations[0]
        if isinstance(gen, ChatGeneration) and isinstance(gen.message, AIMessage):
            meta = getattr(gen.message, "usage_metadata", None)
            if meta:
                usage["input_tokens"] = getattr(meta, "input_tokens", 0) or meta.get("input_tokens", 0) if isinstance(meta, dict) else getattr(meta, "input_tokens", 0)
                usage["output_tokens"] = getattr(meta, "output_tokens", 0) or meta.get("output_tokens", 0) if isinstance(meta, dict) else getattr(meta, "output_tokens", 0)
    return usage


def _extract_output_text(result: ChatResult) -> str:
    if result.generations:
        gen = result.generations[0]
        if isinstance(gen, ChatGeneration) and gen.message:
            return gen.message.content
        return gen.text
    return ""


def _messages_to_dicts(messages: list[BaseMessage]) -> list[dict[str, Any]]:
    return [{"role": msg.type, "content": msg.content} for msg in messages]


def _extract_model_name(model: BaseChatModel) -> str:
    if hasattr(model, "model_name"):
        return model.model_name
    if hasattr(model, "model"):
        return model.model
    return type(model).__name__


class TracedChatModel(BaseChatModel):
    """Wrapper around a BaseChatModel that logs LLM calls as traced spans."""

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
        tracer = get_tracer()
        if tracer is None:
            return self.wrapped._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

        span_meta: dict[str, Any] = {
            "provider": "langchain",
            "model": _extract_model_name(self.wrapped),
        }
        if self.operation:
            span_meta["operation"] = self.operation
        input_data = _messages_to_dicts(messages) if tracer._capture_content else None

        with tracer.span(
            self.operation or "langchain.chat.generate",
            input=input_data,
            metadata=span_meta,
            kind="llm",
        ) as span:
            result = self.wrapped._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
            usage = _extract_token_usage(result)
            if usage:
                span.set_metadata("token_usage", usage)
            if tracer._capture_content:
                span.set_output(_extract_output_text(result))
            return result

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        tracer = get_tracer()
        if tracer is None:
            return await self.wrapped._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)

        span_meta: dict[str, Any] = {
            "provider": "langchain",
            "model": _extract_model_name(self.wrapped),
        }
        if self.operation:
            span_meta["operation"] = self.operation
        input_data = _messages_to_dicts(messages) if tracer._capture_content else None

        with tracer.span(
            self.operation or "langchain.chat.generate",
            input=input_data,
            metadata=span_meta,
            kind="llm",
        ) as span:
            result = await self.wrapped._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)
            usage = _extract_token_usage(result)
            if usage:
                span.set_metadata("token_usage", usage)
            if tracer._capture_content:
                span.set_output(_extract_output_text(result))
            return result

    def __getattr__(self, name: str) -> Any:
        if name in ("wrapped", "operation"):
            raise AttributeError(name)
        return getattr(self.wrapped, name)


def traced_model(model: BaseChatModel, operation: str = "") -> TracedChatModel:
    """Wrap a LangChain chat model to auto-trace LLM calls as spans."""
    return TracedChatModel(wrapped=model, operation=operation)
