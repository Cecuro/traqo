"""traqo — Structured tracing for applications."""

from __future__ import annotations

import os

from traqo._version import __version__
from traqo.backend import Backend, flush_backends, shutdown_backends
from traqo.decorator import trace
from traqo.tracer import Span, Tracer, get_current_span, get_tracer

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
    "Span",
    "Tracer",
    "trace",
    "get_tracer",
    "get_current_span",
    "disable",
    "enable",
    "flush_backends",
    "shutdown_backends",
    "__version__",
]
