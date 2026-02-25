"""traqo — Structured tracing for applications."""

from __future__ import annotations

import os

from traqo._version import __version__
from traqo.backend import Backend, flush_backends, shutdown_backends
from traqo.logging import setup_logging
from traqo.decorator import trace
from traqo.reader import LLMSpan, aggregate_tokens, iter_llm_spans
from traqo.tracer import (
    AGENT,
    CHAIN,
    EMBEDDING,
    GUARDRAIL,
    LLM,
    RETRIEVER,
    TOOL,
    Span,
    Tracer,
    get_current_span,
    get_tracer,
    subtrace,
    update_current_span,
)

_disabled: bool = os.environ.get("TRAQO_DISABLED", "").strip() in ("1", "true", "yes")


def disable() -> None:
    """Globally disable all tracing. Tracers become no-ops."""
    global _disabled
    _disabled = True


def enable() -> None:
    """Re-enable tracing after a disable() call."""
    global _disabled
    _disabled = False


__all__ = [
    "Backend",
    "LLMSpan",
    "Span",
    "Tracer",
    "trace",
    "aggregate_tokens",
    "get_tracer",
    "get_current_span",
    "iter_llm_spans",
    "subtrace",
    "update_current_span",
    "disable",
    "enable",
    "setup_logging",
    "flush_backends",
    "shutdown_backends",
    "__version__",
    # Span kind constants
    "LLM",
    "TOOL",
    "RETRIEVER",
    "CHAIN",
    "AGENT",
    "EMBEDDING",
    "GUARDRAIL",
]
