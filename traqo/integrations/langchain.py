"""LangChain integration — callback handler and model wrapper for auto-tracing."""

from __future__ import annotations

import functools
import logging
import threading
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

try:
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage, BaseMessage, ChatMessage
    from langchain_core.outputs import ChatGeneration, ChatResult, LLMResult
    from pydantic import ConfigDict
except ImportError as err:
    raise ImportError(
        "LangChain not installed. Install with: pip install traqo[langchain]"
    ) from err

from traqo.serialize import serialize_error
from traqo.tracer import _get_parent_id, _now, _uuid, get_tracer

_logger = logging.getLogger("traqo.integrations.langchain")


def _parse_usage_metadata(meta: Any) -> dict[str, int]:
    """Parse LangChain usage_metadata into a flat token dict.

    Extracts reasoning tokens from output_token_details and cache tokens
    from input_token_details (Anthropic prompt caching, OpenAI cached prompts).
    """
    usage: dict[str, int] = {}
    if isinstance(meta, dict):
        usage["input_tokens"] = meta.get("input_tokens", 0)
        usage["output_tokens"] = meta.get("output_tokens", 0)
        output_details = meta.get("output_token_details", {}) or {}
        input_details = meta.get("input_token_details", {}) or {}
    else:
        usage["input_tokens"] = getattr(meta, "input_tokens", 0)
        usage["output_tokens"] = getattr(meta, "output_tokens", 0)
        output_details = getattr(meta, "output_token_details", {}) or {}
        input_details = getattr(meta, "input_token_details", {}) or {}

    def _get(d: Any, key: str) -> int:
        if isinstance(d, dict):
            return d.get(key, 0) or 0
        return getattr(d, key, 0) or 0

    # Reasoning tokens (OpenAI reasoning models)
    reasoning = _get(output_details, "reasoning")
    if reasoning:
        usage["reasoning_tokens"] = reasoning

    # Cache tokens (Anthropic prompt caching / OpenAI cached prompts)
    cache_read = _get(input_details, "cache_read")
    cache_creation = _get(input_details, "cache_creation")
    if cache_read:
        usage["cache_read_tokens"] = cache_read
    if cache_creation:
        usage["cache_creation_tokens"] = cache_creation

    return usage


def _extract_token_usage(result: ChatResult) -> dict[str, int]:
    usage: dict[str, int] = {}
    llm_output = result.llm_output or {}
    token_usage = llm_output.get("token_usage") or llm_output.get("usage") or {}
    if token_usage:
        usage["input_tokens"] = token_usage.get("prompt_tokens", 0)
        usage["output_tokens"] = token_usage.get("completion_tokens", 0)

    if result.generations:
        gen = result.generations[0]
        if isinstance(gen, ChatGeneration) and isinstance(gen.message, AIMessage):
            meta = getattr(gen.message, "usage_metadata", None)
            if meta:
                usage.update(_parse_usage_metadata(meta))
    return usage


def _extract_output(result: ChatResult) -> Any:
    if result.generations:
        gen = result.generations[0]
        if isinstance(gen, ChatGeneration) and gen.message:
            msg = gen.message
            if msg.content:
                return _message_content(msg)
            tool_calls = getattr(msg, "tool_calls", None)
            if tool_calls:
                return [{"name": tc["name"], "args": tc["args"]} for tc in tool_calls]
        return gen.text
    return ""


def _message_content(msg: BaseMessage) -> Any:
    """Extract content from a message, handling structured content blocks."""
    content = msg.content
    # Structured content (reasoning models return list of blocks)
    if isinstance(content, list):
        out: dict[str, Any] = {}
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "reasoning":
                    summaries = block.get("summary", [])
                    out["reasoning"] = " ".join(
                        s.get("text", "") for s in summaries if isinstance(s, dict)
                    )
                elif block.get("type") == "text":
                    out["text"] = block.get("text", "")
        return out if out else content
    return content


def _message_to_dict(msg: BaseMessage) -> dict[str, Any]:
    """Convert a message to a dict, preserving tool calls and structured content."""
    role = msg.role if isinstance(msg, ChatMessage) else msg.type
    d: dict[str, Any] = {"role": role, "content": _message_content(msg)}
    tool_calls = getattr(msg, "tool_calls", None)
    if tool_calls:
        d["tool_calls"] = [
            {"name": tc["name"], "args": tc["args"]} for tc in tool_calls
        ]
    tool_call_id = getattr(msg, "tool_call_id", None)
    if tool_call_id:
        d["tool_call_id"] = tool_call_id
    name = getattr(msg, "name", None)
    if name:
        d["name"] = name
    return d


def _messages_to_dicts(messages: list[BaseMessage]) -> list[dict[str, Any]]:
    return [_message_to_dict(msg) for msg in messages]


def _is_langgraph_interrupt(error: BaseException) -> bool:
    """Check if an error is a LangGraph interrupt (control flow, not an error).

    LangGraph uses ``GraphBubbleUp`` as the base class for all control-flow
    exceptions (``GraphInterrupt``, ``NodeInterrupt``, ``ParentCommand``, etc.).
    We check the MRO by class name so that ``langgraph`` remains an optional
    dependency and any current or future subclass is caught.
    """
    return any(
        cls.__name__ in ("GraphBubbleUp", "GraphInterrupt", "NodeInterrupt")
        for cls in type(error).__mro__
    )


def _interrupt_value(error: BaseException) -> Any:
    """Extract the interrupt payload from a LangGraph interrupt exception."""
    # GraphInterrupt stores the payload in .args[0] (a list of Interrupt objects)
    # or sometimes directly as .value
    value = getattr(error, "value", None)
    if value is not None:
        return value
    if error.args:
        return error.args[0]
    return None


def _extract_model_name(model: BaseChatModel) -> str:
    for attr in ("model_name", "model", "deployment_name", "azure_deployment"):
        val = getattr(model, attr, None)
        if val:
            return val
    return type(model).__name__


def _safe_callback(fn):  # type: ignore[no-untyped-def]
    """Wrap a callback so exceptions are logged instead of crashing the pipeline."""

    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        try:
            return fn(self, *args, **kwargs)
        except Exception:
            _logger.debug("Error in TraqoCallback.%s", fn.__name__, exc_info=True)

    return wrapper


class TraqoCallback(BaseCallbackHandler):
    """LangChain callback handler that auto-traces LLM and tool calls.

    Pass as a callback to any LangChain chain, agent, or model invocation.
    Requires an active Tracer context (via ``with Tracer(...)``).

    Example::

        callback = TraqoCallback()
        with Tracer("my_trace"):
            result = chain.invoke({"text": "hello"}, config={"callbacks": [callback]})
    """

    def __init__(self) -> None:
        super().__init__()
        self._runs: dict[UUID, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def _resolve_parent_id(self, parent_run_id: UUID | None) -> str | None:
        with self._lock:
            if parent_run_id and parent_run_id in self._runs:
                return self._runs[parent_run_id]["span_id"]
        return _get_parent_id()

    def _store_run(self, run_id: UUID, info: dict[str, Any]) -> None:
        with self._lock:
            self._runs[run_id] = info

    def _pop_run(self, run_id: UUID) -> dict[str, Any] | None:
        with self._lock:
            return self._runs.pop(run_id, None)

    @staticmethod
    def _name_from_serialized(
        serialized: dict[str, Any] | None, fallback: str, **kwargs: Any
    ) -> str:
        """Extract a span name, preferring kwargs['name'] (LangChain v0.3+)."""
        if kwargs.get("name"):
            return kwargs["name"]
        if serialized is not None:
            id_parts = serialized.get("id", [])
            if id_parts:
                return id_parts[-1]
            name = serialized.get("name")
            if name:
                return name
        return fallback

    def _extract_model_from_serialized(
        self, serialized: dict[str, Any] | None, **kwargs: Any
    ) -> str:
        if kwargs.get("name"):
            return kwargs["name"]
        if serialized is not None:
            ser_kwargs = serialized.get("kwargs", {})
            for key in ("model_name", "model", "deployment_name", "azure_deployment"):
                val = ser_kwargs.get(key)
                if val:
                    return val
        # Fallback: check invocation_params (some providers put model name there)
        invocation_params = kwargs.get("invocation_params", {})
        if invocation_params:
            for key in ("model_name", "model"):
                val = invocation_params.get(key)
                if val:
                    return val
        if serialized is not None:
            id_parts = serialized.get("id", [])
            if id_parts:
                return id_parts[-1]
        return "unknown"

    # -- LLM callbacks --

    @_safe_callback
    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Fallback for non-chat LLM types (e.g. completion models)."""
        tracer = get_tracer()
        if not tracer:
            return

        span_id = _uuid()
        parent_id = self._resolve_parent_id(parent_run_id)
        start = datetime.now(timezone.utc)
        model = self._extract_model_from_serialized(serialized, **kwargs)

        meta: dict[str, Any] = {"provider": "langchain", "model": model}

        # Extract model parameters from invocation_params
        invocation_params = kwargs.get("invocation_params", {})
        if invocation_params:
            model_params: dict[str, Any] = {}
            for key in (
                "temperature",
                "max_tokens",
                "max_completion_tokens",
                "top_p",
                "frequency_penalty",
                "presence_penalty",
                "stop",
            ):
                if key in invocation_params and invocation_params[key] is not None:
                    model_params[key] = invocation_params[key]
            if model_params:
                meta["model_parameters"] = model_params

        # Merge LangChain metadata from kwargs
        lc_metadata = kwargs.get("metadata")
        if lc_metadata:
            meta.update(lc_metadata)
            # Normalize LangChain's ls_model_name → model
            ls_model = meta.pop("ls_model_name", None)
            if ls_model and not meta.get("model"):
                meta["model"] = ls_model

        # Resolve final model name: prefer metadata (may have been fixed above)
        model = meta.get("model") or model

        start_event: dict[str, Any] = {
            "type": "span_start",
            "id": span_id,
            "parent_id": parent_id,
            "name": model,
            "ts": start.isoformat(),
            "kind": "llm",
            "metadata": meta.copy(),
        }
        if tracer.capture_content and prompts:
            start_event["input"] = prompts

        # Extract LangChain tags from kwargs
        lc_tags = kwargs.get("tags")
        if lc_tags:
            start_event["tags"] = lc_tags

        tracer.write_event(start_event)
        self._store_run(
            run_id,
            {
                "span_id": span_id,
                "parent_id": parent_id,
                "name": model,
                "start": start,
                "metadata": meta,
            },
        )

    @_safe_callback
    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        tracer = get_tracer()
        if not tracer:
            return

        span_id = _uuid()
        parent_id = self._resolve_parent_id(parent_run_id)
        start = datetime.now(timezone.utc)
        model = self._extract_model_from_serialized(serialized, **kwargs)

        meta: dict[str, Any] = {"provider": "langchain", "model": model}

        # Extract model parameters from invocation_params
        invocation_params = kwargs.get("invocation_params", {})
        if invocation_params:
            model_params: dict[str, Any] = {}
            for key in (
                "temperature",
                "max_tokens",
                "max_completion_tokens",
                "top_p",
                "frequency_penalty",
                "presence_penalty",
                "stop",
            ):
                if key in invocation_params and invocation_params[key] is not None:
                    model_params[key] = invocation_params[key]
            if model_params:
                meta["model_parameters"] = model_params

        # Merge LangChain metadata from kwargs
        lc_metadata = kwargs.get("metadata")
        if lc_metadata:
            meta.update(lc_metadata)
            # Normalize LangChain's ls_model_name → model
            ls_model = meta.pop("ls_model_name", None)
            if ls_model and not meta.get("model"):
                meta["model"] = ls_model

        # Resolve final model name: prefer metadata (may have been fixed above)
        model = meta.get("model") or model

        start_event: dict[str, Any] = {
            "type": "span_start",
            "id": span_id,
            "parent_id": parent_id,
            "name": model,
            "ts": start.isoformat(),
            "kind": "llm",
            "metadata": meta.copy(),
        }
        if tracer.capture_content and messages:
            start_event["input"] = _messages_to_dicts(messages[0])

        # Extract LangChain tags from kwargs
        lc_tags = kwargs.get("tags")
        if lc_tags:
            start_event["tags"] = lc_tags

        tracer.write_event(start_event)
        self._store_run(
            run_id,
            {
                "span_id": span_id,
                "parent_id": parent_id,
                "name": model,
                "start": start,
                "metadata": meta,
            },
        )

    @_safe_callback
    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        tracer = get_tracer()
        info = self._pop_run(run_id) if tracer else None
        if not info:
            return
        duration = (datetime.now(timezone.utc) - info["start"]).total_seconds()
        meta = info["metadata"]

        usage = _extract_token_usage_from_response(response)
        if usage:
            meta["token_usage"] = usage

        # Update model name from response (e.g. Azure OpenAI returns actual model)
        response_model = (response.llm_output or {}).get("model_name")
        if response_model:
            meta["model"] = response_model

        end_event: dict[str, Any] = {
            "type": "span_end",
            "id": info["span_id"],
            "parent_id": info["parent_id"],
            "name": info["name"],
            "ts": _now(),
            "duration_s": round(duration, 3),
            "status": "ok",
            "kind": "llm",
            "metadata": meta,
        }
        if tracer.capture_content:
            end_event["output"] = _extract_output_from_response(response)

        tracer.write_event(end_event)
        tracer.record_span()
        if usage:
            tracer.record_tokens(
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
                cache_read_tokens=usage.get("cache_read_tokens", 0),
                cache_creation_tokens=usage.get("cache_creation_tokens", 0),
                reasoning_tokens=usage.get("reasoning_tokens", 0),
            )

    @_safe_callback
    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        tracer = get_tracer()
        info = self._pop_run(run_id) if tracer else None
        if not info:
            return
        duration = (datetime.now(timezone.utc) - info["start"]).total_seconds()
        tracer.write_event(
            {
                "type": "span_end",
                "id": info["span_id"],
                "parent_id": info["parent_id"],
                "name": info["name"],
                "ts": _now(),
                "duration_s": round(duration, 3),
                "status": "error",
                "kind": "llm",
                "error": serialize_error(error),
                "metadata": info["metadata"],
            }
        )
        tracer.record_span()
        tracer.record_error()

    @_safe_callback
    def on_llm_new_token(
        self,
        token: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        with self._lock:
            info = self._runs.get(run_id)
            if info and "ttft_recorded" not in info:
                info["ttft_recorded"] = True
                ttft = (datetime.now(timezone.utc) - info["start"]).total_seconds()
                info["metadata"]["time_to_first_token_s"] = round(ttft, 3)

    # -- Tool callbacks --

    @_safe_callback
    def on_tool_start(
        self,
        serialized: dict[str, Any] | None,
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        tracer = get_tracer()
        if not tracer:
            return

        span_id = _uuid()
        parent_id = self._resolve_parent_id(parent_run_id)
        start = datetime.now(timezone.utc)
        name = self._name_from_serialized(serialized, "tool", **kwargs)

        meta: dict[str, Any] = {}

        # Merge LangChain metadata from kwargs
        lc_metadata = kwargs.get("metadata")
        if lc_metadata:
            meta.update(lc_metadata)

        start_event: dict[str, Any] = {
            "type": "span_start",
            "id": span_id,
            "parent_id": parent_id,
            "name": name,
            "ts": start.isoformat(),
            "kind": "tool",
        }
        if meta:
            start_event["metadata"] = meta.copy()
        if tracer.capture_content:
            start_event["input"] = input_str

        # Extract LangChain tags from kwargs
        lc_tags = kwargs.get("tags")
        if lc_tags:
            start_event["tags"] = lc_tags

        tracer.write_event(start_event)
        self._store_run(
            run_id,
            {
                "span_id": span_id,
                "parent_id": parent_id,
                "name": name,
                "start": start,
                "metadata": meta,
            },
        )

    @_safe_callback
    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        tracer = get_tracer()
        info = self._pop_run(run_id) if tracer else None
        if not info:
            return
        duration = (datetime.now(timezone.utc) - info["start"]).total_seconds()

        # Resolve tool name: prefer name from on_tool_start, but LangGraph
        # often passes name=None there. Fall back to output.name (ToolMessage).
        name = info["name"]
        if not name or name == "tool":
            tool_name = getattr(output, "name", None)
            if tool_name:
                name = tool_name

        end_event: dict[str, Any] = {
            "type": "span_end",
            "id": info["span_id"],
            "parent_id": info["parent_id"],
            "name": name,
            "ts": _now(),
            "duration_s": round(duration, 3),
            "status": "ok",
            "kind": "tool",
        }
        if tracer.capture_content:
            end_event["output"] = str(output)

        tracer.write_event(end_event)
        tracer.record_span()

    @_safe_callback
    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        tracer = get_tracer()
        info = self._pop_run(run_id) if tracer else None
        if not info:
            return
        duration = (datetime.now(timezone.utc) - info["start"]).total_seconds()
        tracer.write_event(
            {
                "type": "span_end",
                "id": info["span_id"],
                "parent_id": info["parent_id"],
                "name": info["name"],
                "ts": _now(),
                "duration_s": round(duration, 3),
                "status": "error",
                "kind": "tool",
                "error": serialize_error(error),
            }
        )
        tracer.record_span()
        tracer.record_error()

    # -- Retriever callbacks --

    @_safe_callback
    def on_retriever_start(
        self,
        serialized: dict[str, Any] | None,
        query: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        tracer = get_tracer()
        if not tracer:
            return

        span_id = _uuid()
        parent_id = self._resolve_parent_id(parent_run_id)
        start = datetime.now(timezone.utc)
        name = self._name_from_serialized(serialized, "retriever", **kwargs)

        meta: dict[str, Any] = {}

        # Merge LangChain metadata from kwargs
        lc_metadata = kwargs.get("metadata")
        if lc_metadata:
            meta.update(lc_metadata)

        start_event: dict[str, Any] = {
            "type": "span_start",
            "id": span_id,
            "parent_id": parent_id,
            "name": name,
            "ts": start.isoformat(),
            "kind": "retriever",
        }
        if meta:
            start_event["metadata"] = meta.copy()
        if tracer.capture_content:
            start_event["input"] = query

        # Extract LangChain tags from kwargs
        lc_tags = kwargs.get("tags")
        if lc_tags:
            start_event["tags"] = lc_tags

        tracer.write_event(start_event)
        self._store_run(
            run_id,
            {
                "span_id": span_id,
                "parent_id": parent_id,
                "name": name,
                "start": start,
                "metadata": meta,
            },
        )

    @_safe_callback
    def on_retriever_end(
        self,
        documents: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        tracer = get_tracer()
        info = self._pop_run(run_id) if tracer else None
        if not info:
            return
        duration = (datetime.now(timezone.utc) - info["start"]).total_seconds()
        end_event: dict[str, Any] = {
            "type": "span_end",
            "id": info["span_id"],
            "parent_id": info["parent_id"],
            "name": info["name"],
            "ts": _now(),
            "duration_s": round(duration, 3),
            "status": "ok",
            "kind": "retriever",
        }
        if tracer.capture_content and documents:
            end_event["output"] = [
                {
                    "page_content": getattr(doc, "page_content", str(doc)),
                    "metadata": getattr(doc, "metadata", {}),
                }
                for doc in documents
            ]

        tracer.write_event(end_event)
        tracer.record_span()

    @_safe_callback
    def on_retriever_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        tracer = get_tracer()
        info = self._pop_run(run_id) if tracer else None
        if not info:
            return
        duration = (datetime.now(timezone.utc) - info["start"]).total_seconds()
        tracer.write_event(
            {
                "type": "span_end",
                "id": info["span_id"],
                "parent_id": info["parent_id"],
                "name": info["name"],
                "ts": _now(),
                "duration_s": round(duration, 3),
                "status": "error",
                "kind": "retriever",
                "error": serialize_error(error),
            }
        )
        tracer.record_span()
        tracer.record_error()

    # -- Chain callbacks --

    @_safe_callback
    def on_chain_start(
        self,
        serialized: dict[str, Any] | None,
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        tracer = get_tracer()
        if not tracer:
            return

        span_id = _uuid()
        parent_id = self._resolve_parent_id(parent_run_id)
        start = datetime.now(timezone.utc)
        name = self._name_from_serialized(serialized, "chain", **kwargs)

        meta: dict[str, Any] = {}

        # Merge LangChain metadata from kwargs
        lc_metadata = kwargs.get("metadata")
        if lc_metadata:
            meta.update(lc_metadata)

        start_event: dict[str, Any] = {
            "type": "span_start",
            "id": span_id,
            "parent_id": parent_id,
            "name": name,
            "ts": start.isoformat(),
            "kind": "chain",
        }
        if meta:
            start_event["metadata"] = meta.copy()
        if tracer.capture_content:
            start_event["input"] = inputs

        # Extract LangChain tags from kwargs
        lc_tags = kwargs.get("tags")
        if lc_tags:
            start_event["tags"] = lc_tags

        tracer.write_event(start_event)
        self._store_run(
            run_id,
            {
                "span_id": span_id,
                "parent_id": parent_id,
                "name": name,
                "start": start,
                "metadata": meta,
            },
        )

    @_safe_callback
    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        tracer = get_tracer()
        info = self._pop_run(run_id) if tracer else None
        if not info:
            return
        duration = (datetime.now(timezone.utc) - info["start"]).total_seconds()
        end_event: dict[str, Any] = {
            "type": "span_end",
            "id": info["span_id"],
            "parent_id": info["parent_id"],
            "name": info["name"],
            "ts": _now(),
            "duration_s": round(duration, 3),
            "status": "ok",
            "kind": "chain",
        }
        if tracer.capture_content:
            end_event["output"] = outputs

        tracer.write_event(end_event)
        tracer.record_span()

    @_safe_callback
    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        tracer = get_tracer()
        info = self._pop_run(run_id) if tracer else None
        if not info:
            return
        duration = (datetime.now(timezone.utc) - info["start"]).total_seconds()

        # LangGraph interrupts are control flow, not errors
        if _is_langgraph_interrupt(error):
            end_event: dict[str, Any] = {
                "type": "span_end",
                "id": info["span_id"],
                "parent_id": info["parent_id"],
                "name": info["name"],
                "ts": _now(),
                "duration_s": round(duration, 3),
                "status": "interrupted",
                "kind": "chain",
            }
            payload = _interrupt_value(error)
            if tracer.capture_content and payload is not None:
                end_event["output"] = payload
            tracer.write_event(end_event)
            tracer.record_span()
            return

        tracer.write_event(
            {
                "type": "span_end",
                "id": info["span_id"],
                "parent_id": info["parent_id"],
                "name": info["name"],
                "ts": _now(),
                "duration_s": round(duration, 3),
                "status": "error",
                "kind": "chain",
                "error": serialize_error(error),
            }
        )
        tracer.record_span()
        tracer.record_error()

    # -- Agent callbacks --

    @_safe_callback
    def on_agent_action(
        self,
        action: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        tracer = get_tracer()
        if not tracer:
            return

        span_id = _uuid()
        parent_id = self._resolve_parent_id(parent_run_id)
        start = datetime.now(timezone.utc)
        tool = getattr(action, "tool", "agent_action")

        start_event: dict[str, Any] = {
            "type": "span_start",
            "id": span_id,
            "parent_id": parent_id,
            "name": tool,
            "ts": start.isoformat(),
            "kind": "agent",
        }
        if tracer.capture_content:
            start_event["input"] = {
                "tool": tool,
                "tool_input": getattr(action, "tool_input", ""),
                "log": getattr(action, "log", ""),
            }

        tracer.write_event(start_event)
        self._store_run(
            run_id,
            {
                "span_id": span_id,
                "parent_id": parent_id,
                "name": tool,
                "start": start,
                "metadata": {},
            },
        )

    @_safe_callback
    def on_agent_finish(
        self,
        finish: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        tracer = get_tracer()
        info = self._pop_run(run_id) if tracer else None
        if not info:
            return
        duration = (datetime.now(timezone.utc) - info["start"]).total_seconds()
        end_event: dict[str, Any] = {
            "type": "span_end",
            "id": info["span_id"],
            "parent_id": info["parent_id"],
            "name": info["name"],
            "ts": _now(),
            "duration_s": round(duration, 3),
            "status": "ok",
            "kind": "agent",
        }
        if tracer.capture_content:
            end_event["output"] = {
                "return_values": getattr(finish, "return_values", {}),
                "log": getattr(finish, "log", ""),
            }

        tracer.write_event(end_event)
        tracer.record_span()


def _extract_token_usage_from_response(response: LLMResult) -> dict[str, int]:
    """Extract token usage from an LLMResult (callback response format)."""
    usage: dict[str, int] = {}
    llm_output = response.llm_output or {}
    token_usage = llm_output.get("token_usage") or llm_output.get("usage") or {}
    if token_usage:
        usage["input_tokens"] = token_usage.get("prompt_tokens", 0)
        usage["output_tokens"] = token_usage.get("completion_tokens", 0)

    if response.generations and response.generations[0]:
        gen = response.generations[0][0]
        if isinstance(gen, ChatGeneration) and isinstance(gen.message, AIMessage):
            meta = getattr(gen.message, "usage_metadata", None)
            if meta:
                usage.update(_parse_usage_metadata(meta))
    return usage


def _extract_output_from_response(response: LLMResult) -> Any:
    """Extract output from an LLMResult (callback response format)."""
    if response.generations and response.generations[0]:
        gen = response.generations[0][0]
        if isinstance(gen, ChatGeneration) and gen.message:
            msg = gen.message
            if msg.content:
                return _message_content(msg)
            tool_calls = getattr(msg, "tool_calls", None)
            if tool_calls:
                return [{"name": tc["name"], "args": tc["args"]} for tc in tool_calls]
        return gen.text
    return ""


class TracedChatModel(BaseChatModel):
    """Wrapper around a BaseChatModel that logs LLM calls as traced spans."""

    wrapped: BaseChatModel
    operation: str = ""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def _llm_type(self) -> str:
        return self.wrapped._llm_type

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return dict(self.wrapped._identifying_params)

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        """Delegate tool binding to wrapped model, keeping tracing active."""
        bound = self.wrapped.bind_tools(tools, **kwargs)
        return self.bind(**bound.kwargs)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        tracer = get_tracer()
        if tracer is None:
            return self.wrapped._generate(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )

        span_meta: dict[str, Any] = {
            "provider": "langchain",
            "model": _extract_model_name(self.wrapped),
        }
        if self.operation:
            span_meta["operation"] = self.operation
        input_data = _messages_to_dicts(messages) if tracer.capture_content else None

        with tracer.span(
            self.operation or "langchain.chat.generate",
            input=input_data,
            metadata=span_meta,
            kind="llm",
        ) as span:
            result = self.wrapped._generate(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            usage = _extract_token_usage(result)
            if usage:
                span.set_metadata("token_usage", usage)
            if tracer.capture_content:
                span.set_output(_extract_output(result))
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
            return await self.wrapped._agenerate(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )

        span_meta: dict[str, Any] = {
            "provider": "langchain",
            "model": _extract_model_name(self.wrapped),
        }
        if self.operation:
            span_meta["operation"] = self.operation
        input_data = _messages_to_dicts(messages) if tracer.capture_content else None

        with tracer.span(
            self.operation or "langchain.chat.generate",
            input=input_data,
            metadata=span_meta,
            kind="llm",
        ) as span:
            result = await self.wrapped._agenerate(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            usage = _extract_token_usage(result)
            if usage:
                span.set_metadata("token_usage", usage)
            if tracer.capture_content:
                span.set_output(_extract_output(result))
            return result

    def __getattr__(self, name: str) -> Any:
        if name in ("wrapped", "operation"):
            raise AttributeError(name)
        return getattr(self.wrapped, name)


def traced_model(model: BaseChatModel, operation: str = "") -> TracedChatModel:
    """Wrap a LangChain chat model to auto-trace LLM calls as spans."""
    return TracedChatModel(wrapped=model, operation=operation)


def track_langgraph(graph: Any, callback: TraqoCallback | None = None) -> Any:
    """Auto-inject a TraqoCallback into a compiled LangGraph.

    After calling this, all ``invoke`` / ``ainvoke`` / ``stream`` /
    ``astream`` calls on *graph* will automatically include the callback
    — no need to pass ``config={"callbacks": [cb]}`` every time.

    Args:
        graph: A compiled LangGraph (``CompiledGraph`` or
            ``CompiledStateGraph``).
        callback: An existing :class:`TraqoCallback` to reuse.  If *None*,
            a new one is created.

    Returns:
        The same *graph* object (mutated in-place) for chaining.

    Example::

        from langgraph.prebuilt import create_react_agent

        agent = create_react_agent(llm, tools)
        track_langgraph(agent)

        with Tracer("my_trace"):
            result = agent.invoke({"messages": [HumanMessage("hi")]})
    """
    if callback is None:
        callback = TraqoCallback()

    _orig_invoke = graph.invoke
    _orig_ainvoke = graph.ainvoke
    _orig_stream = graph.stream
    _orig_astream = graph.astream

    def _inject_callback(config: dict[str, Any] | None) -> dict[str, Any]:
        config = dict(config) if config else {}
        cbs = list(config.get("callbacks") or [])
        if callback not in cbs:
            cbs.append(callback)
        config["callbacks"] = cbs
        return config

    def invoke(input: Any, config: dict[str, Any] | None = None, **kwargs: Any) -> Any:
        return _orig_invoke(input, config=_inject_callback(config), **kwargs)

    async def ainvoke(
        input: Any, config: dict[str, Any] | None = None, **kwargs: Any
    ) -> Any:
        return await _orig_ainvoke(input, config=_inject_callback(config), **kwargs)

    def stream(input: Any, config: dict[str, Any] | None = None, **kwargs: Any) -> Any:
        yield from _orig_stream(input, config=_inject_callback(config), **kwargs)

    async def astream(
        input: Any, config: dict[str, Any] | None = None, **kwargs: Any
    ) -> Any:
        async for chunk in _orig_astream(
            input, config=_inject_callback(config), **kwargs
        ):
            yield chunk

    graph.invoke = invoke
    graph.ainvoke = ainvoke
    graph.stream = stream
    graph.astream = astream

    return graph
