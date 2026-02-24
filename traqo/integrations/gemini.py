"""Gemini integration — wrap google.genai client to auto-trace LLM calls as spans."""

from __future__ import annotations

import sys
import time
from typing import Any

try:
    import google.genai as _genai_mod  # noqa: F401
except ImportError:
    raise ImportError(
        "Google GenAI not installed. Install with: pip install traqo[gemini]"
    )

from traqo.tracer import get_tracer

_GEMINI_MODEL_PARAMS = ("temperature", "max_output_tokens", "top_p", "top_k")


def _extract_model_params_from_config(config: Any) -> dict[str, Any] | None:
    """Extract generation parameters from a config dict or GenerateContentConfig."""
    if config is None:
        return None
    params: dict[str, Any] = {}
    for key in _GEMINI_MODEL_PARAMS:
        if isinstance(config, dict):
            if key in config:
                params[key] = config[key]
        else:
            val = getattr(config, key, None)
            if val is not None:
                params[key] = val
    return params or None


def _extract_usage(response: Any) -> dict[str, int]:
    """Extract token usage from a Gemini response."""
    usage_meta = getattr(response, "usage_metadata", None)
    if not usage_meta:
        return {}
    usage: dict[str, int] = {}
    prompt_tokens = getattr(usage_meta, "prompt_token_count", None)
    candidates_tokens = getattr(usage_meta, "candidates_token_count", None)
    if isinstance(prompt_tokens, (int, float)):
        usage["input_tokens"] = int(prompt_tokens)
    if isinstance(candidates_tokens, (int, float)):
        usage["output_tokens"] = int(candidates_tokens)
    return usage


def _extract_output(response: Any) -> Any:
    """Extract text and function calls from a Gemini response."""
    text = getattr(response, "text", None) or ""
    function_calls = []
    for candidate in getattr(response, "candidates", []):
        content = getattr(candidate, "content", None)
        if not content:
            continue
        for part in getattr(content, "parts", []):
            fc = getattr(part, "function_call", None)
            if fc:
                function_calls.append({
                    "name": getattr(fc, "name", ""),
                    "args": dict(getattr(fc, "args", {})) if getattr(fc, "args", None) else {},
                })
    if function_calls:
        return {"content": text, "function_calls": function_calls}
    return text


# ---------------------------------------------------------------------------
# Streaming helpers
# ---------------------------------------------------------------------------

class _StreamWrapper:
    """Wraps a Gemini sync stream — accumulates chunks and writes span on close."""

    def __init__(self, stream: Any, span: Any, tracer: Any, capture_content: bool, span_ctx: Any = None) -> None:
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
        # Accumulate text from all chunks
        text_parts: list[str] = []
        function_calls: list[dict[str, Any]] = []
        usage: dict[str, int] = {}

        for chunk in self._chunks:
            chunk_text = getattr(chunk, "text", None)
            if chunk_text:
                text_parts.append(chunk_text)
            # Usage metadata is typically on the last chunk
            chunk_usage = _extract_usage(chunk)
            if chunk_usage:
                usage = chunk_usage
            # Check for function calls
            for candidate in getattr(chunk, "candidates", []):
                content = getattr(candidate, "content", None)
                if not content:
                    continue
                for part in getattr(content, "parts", []):
                    fc = getattr(part, "function_call", None)
                    if fc:
                        function_calls.append({
                            "name": getattr(fc, "name", ""),
                            "args": dict(getattr(fc, "args", {})) if getattr(fc, "args", None) else {},
                        })

        text = "".join(text_parts)
        if function_calls:
            output: Any = {"content": text, "function_calls": function_calls}
        else:
            output = text

        if usage:
            self._span.set_metadata("token_usage", usage)
        if self._capture_content:
            self._span.set_output(output)
        if self._span_ctx is not None:
            self._span_ctx.__exit__(None, None, None)
            self._span_ctx = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


class _AsyncStreamWrapper:
    """Wraps a Gemini async stream — accumulates chunks and writes span on close."""

    def __init__(self, stream: Any, span: Any, tracer: Any, capture_content: bool, span_ctx: Any = None) -> None:
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
        text_parts: list[str] = []
        function_calls: list[dict[str, Any]] = []
        usage: dict[str, int] = {}

        for chunk in self._chunks:
            chunk_text = getattr(chunk, "text", None)
            if chunk_text:
                text_parts.append(chunk_text)
            chunk_usage = _extract_usage(chunk)
            if chunk_usage:
                usage = chunk_usage
            for candidate in getattr(chunk, "candidates", []):
                content = getattr(candidate, "content", None)
                if not content:
                    continue
                for part in getattr(content, "parts", []):
                    fc = getattr(part, "function_call", None)
                    if fc:
                        function_calls.append({
                            "name": getattr(fc, "name", ""),
                            "args": dict(getattr(fc, "args", {})) if getattr(fc, "args", None) else {},
                        })

        text = "".join(text_parts)
        if function_calls:
            output: Any = {"content": text, "function_calls": function_calls}
        else:
            output = text

        if usage:
            self._span.set_metadata("token_usage", usage)
        if self._capture_content:
            self._span.set_output(output)
        if self._span_ctx is not None:
            self._span_ctx.__exit__(None, None, None)
            self._span_ctx = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


# ---------------------------------------------------------------------------
# Traced wrapper classes
# ---------------------------------------------------------------------------

class _TracedModels:
    """Wraps client.models for sync generate_content / generate_content_stream / embed_content."""

    def __init__(self, models: Any, operation: str) -> None:
        self._models = models
        self._operation = operation

    def generate_content(self, *, model: str = "", contents: Any = None, config: Any = None, **kwargs: Any) -> Any:
        tracer = get_tracer()
        if tracer is None:
            return self._models.generate_content(model=model, contents=contents, config=config, **kwargs)

        span_meta: dict[str, Any] = {"provider": "gemini"}
        if self._operation:
            span_meta["operation"] = self._operation
        if model:
            span_meta["model"] = model
        model_params = _extract_model_params_from_config(config)
        if model_params:
            span_meta["model_parameters"] = model_params
        input_data = contents if tracer.capture_content else None

        with tracer.span(
            self._operation or "gemini.generate_content",
            input=input_data,
            metadata=span_meta,
            kind="llm",
        ) as span:
            response = self._models.generate_content(model=model, contents=contents, config=config, **kwargs)
            usage = _extract_usage(response)
            if usage:
                span.set_metadata("token_usage", usage)
            if tracer.capture_content:
                span.set_output(_extract_output(response))
            return response

    def generate_content_stream(self, *, model: str = "", contents: Any = None, config: Any = None, **kwargs: Any) -> Any:
        tracer = get_tracer()
        if tracer is None:
            return self._models.generate_content_stream(model=model, contents=contents, config=config, **kwargs)

        span_meta: dict[str, Any] = {"provider": "gemini"}
        if self._operation:
            span_meta["operation"] = self._operation
        if model:
            span_meta["model"] = model
        model_params = _extract_model_params_from_config(config)
        if model_params:
            span_meta["model_parameters"] = model_params
        input_data = contents if tracer.capture_content else None

        span_ctx = tracer.span(
            self._operation or "gemini.generate_content_stream",
            input=input_data,
            metadata=span_meta,
            kind="llm",
        )
        span = span_ctx.__enter__()
        try:
            stream = self._models.generate_content_stream(model=model, contents=contents, config=config, **kwargs)
            return _StreamWrapper(stream, span, tracer, tracer.capture_content, span_ctx)
        except BaseException:
            span_ctx.__exit__(*sys.exc_info())
            raise

    def embed_content(self, *, model: str = "", contents: Any = None, **kwargs: Any) -> Any:
        tracer = get_tracer()
        if tracer is None:
            return self._models.embed_content(model=model, contents=contents, **kwargs)

        span_meta: dict[str, Any] = {"provider": "gemini"}
        if self._operation:
            span_meta["operation"] = self._operation
        if model:
            span_meta["model"] = model
        input_data = contents if tracer.capture_content else None

        with tracer.span(
            self._operation or "gemini.embed_content",
            input=input_data,
            metadata=span_meta,
            kind="embedding",
        ) as span:
            response = self._models.embed_content(model=model, contents=contents, **kwargs)
            usage = _extract_usage(response)
            if usage:
                span.set_metadata("token_usage", usage)
            return response

    def __getattr__(self, name: str) -> Any:
        return getattr(self._models, name)


class _TracedAsyncModels:
    """Wraps client.aio.models for async generate_content / generate_content_stream."""

    def __init__(self, models: Any, operation: str) -> None:
        self._models = models
        self._operation = operation

    async def generate_content(self, *, model: str = "", contents: Any = None, config: Any = None, **kwargs: Any) -> Any:
        tracer = get_tracer()
        if tracer is None:
            return await self._models.generate_content(model=model, contents=contents, config=config, **kwargs)

        span_meta: dict[str, Any] = {"provider": "gemini"}
        if self._operation:
            span_meta["operation"] = self._operation
        if model:
            span_meta["model"] = model
        model_params = _extract_model_params_from_config(config)
        if model_params:
            span_meta["model_parameters"] = model_params
        input_data = contents if tracer.capture_content else None

        with tracer.span(
            self._operation or "gemini.generate_content",
            input=input_data,
            metadata=span_meta,
            kind="llm",
        ) as span:
            response = await self._models.generate_content(model=model, contents=contents, config=config, **kwargs)
            usage = _extract_usage(response)
            if usage:
                span.set_metadata("token_usage", usage)
            if tracer.capture_content:
                span.set_output(_extract_output(response))
            return response

    async def generate_content_stream(self, *, model: str = "", contents: Any = None, config: Any = None, **kwargs: Any) -> Any:
        tracer = get_tracer()
        if tracer is None:
            return await self._models.generate_content_stream(model=model, contents=contents, config=config, **kwargs)

        span_meta: dict[str, Any] = {"provider": "gemini"}
        if self._operation:
            span_meta["operation"] = self._operation
        if model:
            span_meta["model"] = model
        model_params = _extract_model_params_from_config(config)
        if model_params:
            span_meta["model_parameters"] = model_params
        input_data = contents if tracer.capture_content else None

        span_ctx = tracer.span(
            self._operation or "gemini.generate_content_stream",
            input=input_data,
            metadata=span_meta,
            kind="llm",
        )
        span = span_ctx.__enter__()
        try:
            stream = await self._models.generate_content_stream(model=model, contents=contents, config=config, **kwargs)
            return _AsyncStreamWrapper(stream, span, tracer, tracer.capture_content, span_ctx)
        except BaseException:
            span_ctx.__exit__(*sys.exc_info())
            raise

    def __getattr__(self, name: str) -> Any:
        return getattr(self._models, name)


class _TracedAio:
    """Wraps client.aio to provide traced async models."""

    def __init__(self, aio: Any, operation: str) -> None:
        self._aio = aio
        self._operation = operation

    @property
    def models(self) -> _TracedAsyncModels:
        return _TracedAsyncModels(self._aio.models, self._operation)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._aio, name)


class _TracedGeminiClient:
    """Traced wrapper around a google.genai.Client."""

    def __init__(self, client: Any, operation: str) -> None:
        self._client = client
        self._operation = operation

    @property
    def models(self) -> _TracedModels:
        return _TracedModels(self._client.models, self._operation)

    @property
    def aio(self) -> _TracedAio:
        return _TracedAio(self._client.aio, self._operation)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


def traced_gemini(client: Any, operation: str = "") -> _TracedGeminiClient:
    """Wrap a Google GenAI client to auto-trace generate_content calls as spans."""
    return _TracedGeminiClient(client, operation)
