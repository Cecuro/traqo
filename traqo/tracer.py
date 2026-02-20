"""Core Tracer class — the trace session."""

from __future__ import annotations

import logging
import threading
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from traqo._version import __version__
from traqo.serialize import serialize_error, serialize_output, to_json

logger = logging.getLogger(__name__)

_active_tracer: ContextVar[Tracer | None] = ContextVar("_active_tracer", default=None)
_span_stack: ContextVar[list[str]] = ContextVar("_span_stack", default=[])


def get_tracer() -> Tracer | None:
    """Return the active Tracer for the current context, or None."""
    import traqo

    if traqo._disabled:
        return None
    return _active_tracer.get(None)


def _get_parent_id() -> str | None:
    """Return the current parent span ID from the stack, or None for root."""
    stack = _span_stack.get([])
    return stack[-1] if stack else None


def _push_span(span_id: str) -> None:
    """Push a span ID onto the current context's span stack."""
    stack = _span_stack.get([])
    _span_stack.set([*stack, span_id])


def _pop_span() -> None:
    """Pop the top span ID from the current context's span stack."""
    stack = _span_stack.get([])
    if stack:
        _span_stack.set(stack[:-1])


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _uuid() -> str:
    return uuid.uuid4().hex[:12]


class Tracer:
    """A trace session that writes structured JSONL events to a file.

    Use as a context manager to activate tracing in the current async context.
    """

    def __init__(
        self,
        path: Path,
        *,
        metadata: dict[str, Any] | None = None,
        capture_content: bool = True,
    ) -> None:
        import traqo

        self._path = Path(path)
        self._metadata = metadata or {}
        self._capture_content = capture_content
        self._disabled = traqo._disabled
        self._lock = threading.Lock()
        self._file = None
        self._token = None
        self._start_time: datetime | None = None

        # Stats
        self._stats_spans = 0
        self._stats_llm_calls = 0
        self._stats_events = 0
        self._stats_errors = 0
        self._stats_input_tokens = 0
        self._stats_output_tokens = 0
        self._children: list[dict[str, Any]] = []

    def _open(self) -> None:
        if self._disabled:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self._path, "a", encoding="utf-8")

    def _close(self) -> None:
        if self._file:
            self._file.close()
            self._file = None

    def _write(self, event: dict[str, Any]) -> None:
        if self._disabled or self._file is None:
            return
        try:
            line = to_json(event)
            with self._lock:
                self._file.write(line + "\n")
                self._file.flush()
        except Exception:
            logger.warning("traqo: failed to write event", exc_info=True)

    def __enter__(self) -> Tracer:
        self._open()
        self._start_time = datetime.now(timezone.utc)
        self._token = _active_tracer.set(self)
        self._write(
            {
                "type": "trace_start",
                "ts": self._start_time.isoformat(),
                "tracer_version": __version__,
                **({"metadata": self._metadata} if self._metadata else {}),
            }
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        duration = (
            (datetime.now(timezone.utc) - self._start_time).total_seconds()
            if self._start_time
            else 0.0
        )
        self._write(
            {
                "type": "trace_end",
                "ts": _now(),
                "duration_s": round(duration, 3),
                "stats": {
                    "spans": self._stats_spans,
                    "llm_calls": self._stats_llm_calls,
                    "events": self._stats_events,
                    "total_input_tokens": self._stats_input_tokens,
                    "total_output_tokens": self._stats_output_tokens,
                    "errors": self._stats_errors,
                },
                **({"children": self._children} if self._children else {}),
            }
        )
        self._close()
        if self._token is not None:
            _active_tracer.reset(self._token)
            self._token = None
        return False

    async def __aenter__(self) -> Tracer:
        return self.__enter__()

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        return self.__exit__(exc_type, exc_val, exc_tb)

    def log(self, name: str, data: dict[str, Any] | None = None) -> None:
        """Write a custom event."""
        self._stats_events += 1
        self._write(
            {
                "type": "event",
                "id": _uuid(),
                "parent_id": _get_parent_id(),
                "name": name,
                "ts": _now(),
                **({"data": data} if data else {}),
            }
        )

    def llm_event(
        self,
        *,
        model: str,
        input_messages: list[dict[str, Any]] | None = None,
        output_text: str | None = None,
        token_usage: dict[str, int] | None = None,
        duration_s: float | None = None,
        operation: str | None = None,
    ) -> None:
        """Write an llm_call event."""
        self._stats_llm_calls += 1
        if token_usage:
            self._stats_input_tokens += token_usage.get("input_tokens", 0)
            self._stats_output_tokens += token_usage.get("output_tokens", 0)

        event: dict[str, Any] = {
            "type": "llm_call",
            "id": _uuid(),
            "parent_id": _get_parent_id(),
            "ts": _now(),
            "model": model,
        }
        if self._capture_content:
            if input_messages is not None:
                event["input"] = input_messages
            if output_text is not None:
                event["output"] = output_text
        if duration_s is not None:
            event["duration_s"] = round(duration_s, 3)
        if token_usage:
            event["token_usage"] = token_usage
        if operation:
            event["operation"] = operation
        self._write(event)

    @contextmanager
    def span(self, name: str, inputs: dict[str, Any] | None = None):
        """Manual span context manager."""
        span_id = _uuid()
        parent_id = _get_parent_id()
        start = datetime.now(timezone.utc)

        self._write(
            {
                "type": "span_start",
                "id": span_id,
                "parent_id": parent_id,
                "name": name,
                "ts": start.isoformat(),
                **({"input": inputs} if inputs else {}),
            }
        )
        _push_span(span_id)
        try:
            yield span_id
        except BaseException as exc:
            duration = (datetime.now(timezone.utc) - start).total_seconds()
            self._stats_spans += 1
            self._stats_errors += 1
            self._write(
                {
                    "type": "span_end",
                    "id": span_id,
                    "parent_id": parent_id,
                    "name": name,
                    "ts": _now(),
                    "duration_s": round(duration, 3),
                    "status": "error",
                    "error": serialize_error(exc),
                }
            )
            raise
        finally:
            _pop_span()

        duration = (datetime.now(timezone.utc) - start).total_seconds()
        self._stats_spans += 1
        self._write(
            {
                "type": "span_end",
                "id": span_id,
                "parent_id": parent_id,
                "name": name,
                "ts": _now(),
                "duration_s": round(duration, 3),
                "status": "ok",
            }
        )

    def child(self, name: str, path: Path | None = None) -> Tracer:
        """Create a child tracer writing to a separate file.

        The child is linked to this parent via events and the trace_start header.
        """
        if path is None:
            path = self._path.parent / f"{name}.jsonl"

        child_tracer = Tracer(
            path,
            metadata={"parent_trace": str(self._path)},
            capture_content=self._capture_content,
        )
        # Store reference so we can write events to parent
        child_tracer._parent = self
        child_tracer._child_name = name

        return child_tracer

    def _write_child_started(self, name: str, path: Path) -> None:
        """Write child_started event to this (parent) tracer."""
        self._write(
            {
                "type": "event",
                "id": _uuid(),
                "parent_id": _get_parent_id(),
                "name": "child_started",
                "ts": _now(),
                "data": {"child_name": name, "child_path": str(path)},
            }
        )
        self._stats_events += 1

    def _write_child_ended(self, name: str, child: Tracer) -> None:
        """Write child_ended event to this (parent) tracer and record summary."""
        summary = {
            "name": name,
            "path": str(child._path),
            "duration_s": round(
                (datetime.now(timezone.utc) - child._start_time).total_seconds(), 3
            )
            if child._start_time
            else 0.0,
            "llm_calls": child._stats_llm_calls,
        }
        self._children.append(summary)
        self._write(
            {
                "type": "event",
                "id": _uuid(),
                "parent_id": _get_parent_id(),
                "name": "child_ended",
                "ts": _now(),
                "data": {
                    "child_name": name,
                    "duration_s": summary["duration_s"],
                    "llm_calls": summary["llm_calls"],
                },
            }
        )
        self._stats_events += 1


# Override __enter__/__exit__ for child tracers to write parent events
_original_enter = Tracer.__enter__
_original_exit = Tracer.__exit__


def _child_enter(self: Tracer) -> Tracer:
    result = _original_enter(self)
    parent = getattr(self, "_parent", None)
    if parent is not None:
        parent._write_child_started(self._child_name, self._path)
    return result


def _child_exit(self: Tracer, exc_type, exc_val, exc_tb):
    parent = getattr(self, "_parent", None)
    if parent is not None:
        parent._write_child_ended(self._child_name, self)
    return _original_exit(self, exc_type, exc_val, exc_tb)


# Patch enter/exit to handle child notifications
Tracer.__enter__ = _child_enter
Tracer.__exit__ = _child_exit


async def _child_aenter(self: Tracer) -> Tracer:
    return _child_enter(self)


async def _child_aexit(self: Tracer, exc_type, exc_val, exc_tb):
    return _child_exit(self, exc_type, exc_val, exc_tb)


Tracer.__aenter__ = _child_aenter
Tracer.__aexit__ = _child_aexit
