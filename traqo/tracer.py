"""Core Tracer and Span classes."""

from __future__ import annotations

import logging
import sys
import threading
import uuid
from collections.abc import Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from traqo.backend import Backend

from traqo._version import __version__
from traqo.serialize import serialize_error, to_json

logger = logging.getLogger(__name__)

# Span kind constants
LLM = "llm"
TOOL = "tool"
RETRIEVER = "retriever"
CHAIN = "chain"
AGENT = "agent"
EMBEDDING = "embedding"
GUARDRAIL = "guardrail"

_active_tracer: ContextVar[Tracer | None] = ContextVar("_active_tracer", default=None)
_span_stack: ContextVar[tuple[Span, ...]] = ContextVar("_span_stack", default=())


def get_tracer() -> Tracer | None:
    """Return the active Tracer for the current context, or None."""
    import traqo

    if traqo._disabled:
        return None
    return _active_tracer.get(None)


def get_current_span() -> Span | None:
    """Return the current active Span, or None if no span is active."""
    stack = _span_stack.get(())
    return stack[-1] if stack else None


def _get_parent_id() -> str | None:
    stack = _span_stack.get(())
    return stack[-1].id if stack else None


def _push_span(span_obj: Span) -> None:
    stack = _span_stack.get(())
    _span_stack.set(stack + (span_obj,))


def _pop_span() -> None:
    stack = _span_stack.get(())
    if stack:
        _span_stack.set(stack[:-1])


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _uuid() -> str:
    return uuid.uuid4().hex[:12]


class Span:
    """A live span handle yielded by tracer.span().

    Set output and metadata during execution — they are written to span_end.
    """

    __slots__ = ("id", "name", "parent_id", "output", "metadata", "tags")

    def __init__(self, span_id: str, name: str, parent_id: str | None) -> None:
        self.id = span_id
        self.name = name
        self.parent_id = parent_id
        self.output: Any = None
        self.metadata: dict[str, Any] = {}
        self.tags: list[str] = []

    def set_output(self, output: Any) -> None:
        self.output = output

    def set_metadata(self, key: str, value: Any) -> None:
        self.metadata[key] = value

    def update_metadata(self, data: dict[str, Any]) -> None:
        self.metadata.update(data)


class Tracer:
    """A trace session that writes structured JSONL events to a file.

    Use as a context manager to activate tracing in the current async context.
    """

    def __init__(
        self,
        path: Path,
        *,
        input: Any = None,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        thread_id: str | None = None,
        capture_content: bool = True,
        backends: Sequence[Backend] | None = None,
    ) -> None:
        import traqo

        self._path = Path(path)
        self._input = input
        self._metadata = metadata or {}
        self._tags = tags or []
        self._thread_id = thread_id
        self._capture_content = capture_content
        self._disabled = traqo._disabled
        self._lock = threading.Lock()
        self._file = None
        self._token = None
        self._start_time: datetime | None = None
        self._output: Any = None

        # Backends
        self._backends: list[Backend] = list(backends) if backends else []
        for b in self._backends:
            for method in ("on_event", "on_trace_complete", "close"):
                if not callable(getattr(b, method, None)):
                    raise TypeError(
                        f"Backend {b!r} is missing required callable method {method!r}"
                    )

        # Stats
        self._stats_spans = 0
        self._stats_events = 0
        self._stats_errors = 0
        self._stats_input_tokens = 0
        self._stats_output_tokens = 0
        self._children: list[dict[str, Any]] = []

        # Child tracer linkage (set by child())
        self._parent: Tracer | None = None
        self._child_name: str = ""

    @property
    def capture_content(self) -> bool:
        """Whether content capture is enabled for this tracer."""
        return self._capture_content

    def set_output(self, output: Any) -> None:
        """Set trace-level output. Written to trace_end."""
        self._output = output

    def write_event(self, event: dict[str, Any]) -> None:
        """Write a raw event dict to the trace file. For use by integrations."""
        self._write(event)

    def record_span(self) -> None:
        """Increment the span counter. For use by integrations."""
        with self._lock:
            self._stats_spans += 1

    def record_error(self) -> None:
        """Increment the error counter. For use by integrations."""
        with self._lock:
            self._stats_errors += 1

    def record_tokens(self, input_tokens: int = 0, output_tokens: int = 0) -> None:
        """Accumulate token counts. For use by integrations."""
        with self._lock:
            self._stats_input_tokens += input_tokens
            self._stats_output_tokens += output_tokens

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
            return
        # Notify backends (outside the lock, only on successful write).
        # The event dict is shared across backends and should not be mutated.
        self._notify_backends_event(event)

    def _notify_backends_event(self, event: dict[str, Any]) -> None:
        """Notify all backends of a new event. Never raises."""
        for backend in self._backends:
            try:
                backend.on_event(event)
            except Exception:
                logger.warning(
                    "traqo: backend on_event failed for %s",
                    type(backend).__name__,
                    exc_info=True,
                )

    def _notify_backends_complete(self) -> None:
        """Notify all backends that the trace file is complete. Never raises."""
        if self._disabled or not self._backends:
            return
        for backend in self._backends:
            try:
                backend.on_trace_complete(self._path)
            except Exception:
                logger.warning(
                    "traqo: backend on_trace_complete failed for %s",
                    type(backend).__name__,
                    exc_info=True,
                )

    def _accumulate_tokens(self, span_obj: Span) -> None:
        """If span metadata contains token_usage, accumulate into tracer stats."""
        token_usage = span_obj.metadata.get("token_usage")
        if token_usage and isinstance(token_usage, dict):
            with self._lock:
                self._stats_input_tokens += token_usage.get("input_tokens", 0)
                self._stats_output_tokens += token_usage.get("output_tokens", 0)

    def __enter__(self) -> Tracer:
        self._open()
        self._start_time = datetime.now(timezone.utc)
        self._token = _active_tracer.set(self)
        start_event: dict[str, Any] = {
            "type": "trace_start",
            "ts": self._start_time.isoformat(),
            "tracer_version": __version__,
        }
        if self._input is not None:
            start_event["input"] = self._input
        if self._metadata:
            start_event["metadata"] = self._metadata
        if self._tags:
            start_event["tags"] = self._tags
        if self._thread_id is not None:
            start_event["thread_id"] = self._thread_id
        self._write(start_event)
        if self._parent is not None:
            self._parent._write_child_started(self._child_name, self._path)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._parent is not None:
            self._parent._write_child_ended(self._child_name, self)
        duration = (
            (datetime.now(timezone.utc) - self._start_time).total_seconds()
            if self._start_time
            else 0.0
        )
        end_event: dict[str, Any] = {
            "type": "trace_end",
            "ts": _now(),
            "duration_s": round(duration, 3),
            "stats": {
                "spans": self._stats_spans,
                "events": self._stats_events,
                "total_input_tokens": self._stats_input_tokens,
                "total_output_tokens": self._stats_output_tokens,
                "errors": self._stats_errors,
            },
        }
        if self._output is not None:
            end_event["output"] = self._output
        if self._children:
            end_event["children"] = self._children
        self._write(end_event)
        self._close()
        self._notify_backends_complete()
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
        with self._lock:
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

    @contextmanager
    def span(
        self,
        name: str,
        *,
        input: Any = None,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        kind: str | None = None,
    ):
        """Span context manager. Yields a Span handle for setting output/metadata.

        Args:
            name: Span name.
            input: Input data written to span_start.
            metadata: Initial metadata. Can be updated via the yielded Span object.
            tags: List of string tags for filtering/categorization.
            kind: Optional span categorization (e.g. "llm", "tool", "retriever").
        """
        span_id = _uuid()
        parent_id = _get_parent_id()
        start = datetime.now(timezone.utc)

        span_obj = Span(span_id, name, parent_id)
        if metadata:
            span_obj.metadata.update(metadata)
        if tags:
            span_obj.tags = list(tags)

        start_event: dict[str, Any] = {
            "type": "span_start",
            "id": span_id,
            "parent_id": parent_id,
            "name": name,
            "ts": start.isoformat(),
        }
        if kind is not None:
            start_event["kind"] = kind
        if input is not None:
            start_event["input"] = input
        if span_obj.tags:
            start_event["tags"] = span_obj.tags
        if span_obj.metadata:
            start_event["metadata"] = span_obj.metadata.copy()
        self._write(start_event)

        _push_span(span_obj)
        try:
            yield span_obj
        except BaseException:
            duration = (datetime.now(timezone.utc) - start).total_seconds()
            with self._lock:
                self._stats_spans += 1
                self._stats_errors += 1
            try:
                self._accumulate_tokens(span_obj)
                end_event: dict[str, Any] = {
                    "type": "span_end",
                    "id": span_id,
                    "parent_id": parent_id,
                    "name": name,
                    "ts": _now(),
                    "duration_s": round(duration, 3),
                    "status": "error",
                    "error": serialize_error(sys.exc_info()[1]),
                }
                if kind is not None:
                    end_event["kind"] = kind
                if span_obj.tags:
                    end_event["tags"] = span_obj.tags
                if span_obj.metadata:
                    end_event["metadata"] = span_obj.metadata.copy()
                self._write(end_event)
            except Exception:
                logger.warning("traqo: failed to write span_end on error", exc_info=True)
            raise
        finally:
            _pop_span()

        duration = (datetime.now(timezone.utc) - start).total_seconds()
        with self._lock:
            self._stats_spans += 1
        self._accumulate_tokens(span_obj)
        end_event: dict[str, Any] = {
            "type": "span_end",
            "id": span_id,
            "parent_id": parent_id,
            "name": name,
            "ts": _now(),
            "duration_s": round(duration, 3),
            "status": "ok",
        }
        if kind is not None:
            end_event["kind"] = kind
        if span_obj.tags:
            end_event["tags"] = span_obj.tags
        if span_obj.output is not None:
            end_event["output"] = span_obj.output
        if span_obj.metadata:
            end_event["metadata"] = span_obj.metadata.copy()
        self._write(end_event)

    def child(self, name: str, path: Path | None = None) -> Tracer:
        """Create a child tracer writing to a separate file."""
        if path is None:
            path = self._path.parent / f"{name}.jsonl"

        child_tracer = Tracer(
            path,
            metadata={"parent_trace": str(self._path)},
            capture_content=self._capture_content,
            backends=self._backends,
        )
        child_tracer._parent = self  # noqa: SLF001
        child_tracer._child_name = name  # noqa: SLF001
        return child_tracer

    def _write_child_started(self, name: str, path: Path) -> None:
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
        with self._lock:
            self._stats_events += 1

    def _write_child_ended(self, name: str, child: Tracer) -> None:
        summary = {
            "name": name,
            "path": str(child._path),
            "duration_s": round(
                (datetime.now(timezone.utc) - child._start_time).total_seconds(), 3
            )
            if child._start_time
            else 0.0,
            "spans": child._stats_spans,
            "total_input_tokens": child._stats_input_tokens,
            "total_output_tokens": child._stats_output_tokens,
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
                    "spans": summary["spans"],
                    "total_input_tokens": summary["total_input_tokens"],
                    "total_output_tokens": summary["total_output_tokens"],
                },
            }
        )
        with self._lock:
            self._stats_events += 1



# ---------------------------------------------------------------------------
# Convenience helper
# ---------------------------------------------------------------------------

class _Unset:
    """Sentinel to distinguish 'not provided' from None."""
    __slots__ = ()
    def __repr__(self) -> str:
        return "<UNSET>"

_UNSET = _Unset()


def update_current_span(
    *,
    output: Any = _UNSET,
    metadata: dict[str, Any] | None = None,
    tags: list[str] | None = None,
    **kw_metadata: Any,
) -> None:
    """Update the current span's metadata/output. No-op if no span active."""
    span = get_current_span()
    if span is None:
        return
    if not isinstance(output, _Unset):
        span.set_output(output)
    if metadata:
        span.update_metadata(metadata)
    if kw_metadata:
        span.update_metadata(kw_metadata)
    if tags:
        span.tags = (span.tags or []) + tags
