"""Core Tracer and Span classes."""

from __future__ import annotations

import contextlib
import logging
import os
import sys
import threading
import time
import uuid
from collections.abc import Sequence
from concurrent.futures import Future
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
    _span_stack.set((*stack, span_obj))


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

    __slots__ = ("id", "metadata", "name", "output", "parent_id", "tags")

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
        name: str | None = None,
        *,
        path: Path | str | None = None,
        trace_dir: Path | str | None = None,
        input: Any = None,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        thread_id: str | None = None,
        capture_content: bool = True,
        backends: Sequence[Backend] | None = None,
        flush_interval: float = 2.0,
        flush_threshold: int = 256_000,
    ) -> None:
        """
        Args:
            flush_interval: Minimum seconds between disk flushes. Flushes are
                lazy — they happen on the next ``_write()`` call after the
                interval elapses, not on a background timer. Set to ``0`` for
                per-event flushing (old behavior).
            flush_threshold: Approximate buffer size in bytes before forcing a
                flush, regardless of the interval.
        """
        import traqo

        self._name = name
        self._path, self._auto_path = self._resolve_path(name, path, trace_dir)
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

        # Write buffer — accumulate lines and flush on interval or size threshold.
        if flush_interval < 0:
            raise ValueError("flush_interval must be >= 0")
        if flush_threshold < 0:
            raise ValueError("flush_threshold must be >= 0")
        self._flush_interval = flush_interval
        self._flush_threshold = flush_threshold
        self._buffer: list[str] = []
        self._buffer_bytes = 0
        self._last_flush = 0.0  # set properly in _open()

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
        self._stats_cache_read_tokens = 0
        self._stats_cache_creation_tokens = 0
        self._stats_reasoning_tokens = 0
        self._children: list[dict[str, Any]] = []

        # Child tracer linkage (set by child())
        self._parent: Tracer | None = None
        self._child_name: str = ""

    @staticmethod
    def _resolve_path(
        name: str | None,
        path: Path | str | None,
        trace_dir: Path | str | None,
    ) -> tuple[Path, bool]:
        """Return (resolved_path, auto_path)."""
        if path is not None:
            return Path(path), False
        base = (
            Path(trace_dir)
            if trace_dir
            else Path(os.environ.get("TRAQO_TRACE_DIR", "./traces"))
        )
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        short_id = uuid.uuid4().hex[:8]
        if name:
            # Sanitize: replace path separators and limit length for filesystem safety
            safe = name.replace("/", "_").replace("\\", "_").replace("\0", "")
            if len(safe) > 200:
                safe = safe[:200]
            stem = f"{safe}_{ts}_{short_id}"
        else:
            stem = f"{ts}_{short_id}"
        return base / f"{stem}.jsonl", True

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

    def record_tokens(
        self,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_read_tokens: int = 0,
        cache_creation_tokens: int = 0,
        reasoning_tokens: int = 0,
    ) -> None:
        """Accumulate token counts. For use by integrations."""
        with self._lock:
            self._stats_input_tokens += input_tokens
            self._stats_output_tokens += output_tokens
            self._stats_cache_read_tokens += cache_read_tokens
            self._stats_cache_creation_tokens += cache_creation_tokens
            self._stats_reasoning_tokens += reasoning_tokens

    def _open(self) -> None:
        if self._disabled:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self._path, "a", encoding="utf-8")  # noqa: SIM115
        self._last_flush = time.monotonic()

    def _close(self) -> None:
        if self._file:
            with self._lock:
                try:
                    self._flush_buffer()
                except Exception:
                    logger.warning(
                        "traqo: failed to flush buffer on close", exc_info=True
                    )
            self._file.close()
            self._file = None

    def _write(self, event: dict[str, Any]) -> None:
        if self._disabled or self._file is None:
            return
        try:
            line = to_json(event)
            with self._lock:
                self._buffer.append(line)
                self._buffer_bytes += len(line.encode("utf-8"))
                now = time.monotonic()
                if (
                    self._buffer_bytes >= self._flush_threshold
                    or (now - self._last_flush) >= self._flush_interval
                ):
                    self._flush_buffer()
        except Exception:
            logger.warning("traqo: failed to write event", exc_info=True)
            return
        # Notify backends (outside the lock, only on successful write).
        # The event dict is shared across backends and should not be mutated.
        self._notify_backends_event(event)

    def _flush_buffer(self) -> None:
        """Write buffered lines to disk. Must be called with self._lock held."""
        if not self._buffer or self._file is None:
            return
        self._file.write("\n".join(self._buffer) + "\n")
        self._file.flush()
        self._buffer.clear()
        self._buffer_bytes = 0
        self._last_flush = time.monotonic()

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

    def _prepare_for_upload(self) -> list[Path]:
        """Compress the trace and delete the raw ``.jsonl`` buffer.

        Returns compressed paths (``.jsonl.gz`` + optional ``.content.jsonl.zst``).
        Falls back to the raw file if compression fails.
        """
        if self._disabled or not self._path.is_file():
            return [self._path]
        try:
            from traqo.compress import split_and_compress

            main_path, content_path = split_and_compress(self._path)
        except Exception:
            logger.warning(
                "traqo: split_and_compress failed, uploading raw file",
                exc_info=True,
            )
            return [self._path]
        self._path.unlink(missing_ok=True)
        paths = [main_path]
        if content_path is not None:
            paths.append(content_path)
        return paths

    def _notify_backends_complete(
        self, upload_paths: list[Path] | None = None
    ) -> list[Future]:
        """Notify all backends that the trace file is complete. Returns futures."""
        futures: list[Future] = []
        if self._disabled or not self._backends:
            return futures
        paths = upload_paths or [self._path]
        for backend in self._backends:
            for path in paths:
                try:
                    result = backend.on_trace_complete(path)
                    if result is not None:
                        futures.append(result)
                except Exception:
                    logger.warning(
                        "traqo: backend on_trace_complete failed for %s",
                        type(backend).__name__,
                        exc_info=True,
                    )
        return futures

    def _schedule_cleanup(
        self, upload_futures: list[Future], upload_paths: list[Path] | None = None
    ) -> None:
        """Delete auto-generated compressed files after backends finish uploading."""
        if not self._auto_path or not self._backends:
            return
        from traqo.backend import submit_background

        paths_to_delete = upload_paths or []

        def _wait_and_delete():
            for fut in upload_futures:
                with contextlib.suppress(Exception):
                    fut.result(timeout=600)
            for p in paths_to_delete:
                try:
                    p.unlink(missing_ok=True)
                except Exception:
                    logger.warning(
                        "traqo: failed to clean up buffer %s", p, exc_info=True
                    )

        submit_background(_wait_and_delete)

    def _accumulate_tokens(self, span_obj: Span) -> None:
        """If span metadata contains token_usage, accumulate into tracer stats."""
        token_usage = span_obj.metadata.get("token_usage")
        if token_usage and isinstance(token_usage, dict):
            with self._lock:
                self._stats_input_tokens += token_usage.get("input_tokens", 0)
                self._stats_output_tokens += token_usage.get("output_tokens", 0)
                self._stats_cache_read_tokens += token_usage.get("cache_read_tokens", 0)
                self._stats_cache_creation_tokens += token_usage.get(
                    "cache_creation_tokens", 0
                )
                self._stats_reasoning_tokens += token_usage.get("reasoning_tokens", 0)

    def __enter__(self) -> Tracer:
        self._open()
        self._start_time = datetime.now(timezone.utc)
        self._token = _active_tracer.set(self)
        start_event: dict[str, Any] = {
            "type": "trace_start",
            "ts": self._start_time.isoformat(),
            "tracer_version": __version__,
        }
        if self._name:
            start_event["name"] = self._name
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
            "status": "error" if exc_val is not None else "ok",
            "stats": {
                "spans": self._stats_spans,
                "events": self._stats_events,
                "total_input_tokens": self._stats_input_tokens,
                "total_output_tokens": self._stats_output_tokens,
                "total_cache_read_tokens": self._stats_cache_read_tokens,
                "total_cache_creation_tokens": self._stats_cache_creation_tokens,
                "total_reasoning_tokens": self._stats_reasoning_tokens,
                "errors": self._stats_errors,
            },
        }
        if exc_val is not None:
            end_event["error"] = serialize_error(exc_val)
        if self._output is not None:
            end_event["output"] = self._output
        if self._children:
            end_event["children"] = self._children
        self._write(end_event)
        try:
            self._close()
            upload_paths = self._prepare_for_upload()
            futures = self._notify_backends_complete(upload_paths)
            self._schedule_cleanup(futures, upload_paths)
        finally:
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
                logger.warning(
                    "traqo: failed to write span_end on error", exc_info=True
                )
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

    def child(
        self,
        name: str,
        *,
        path: Path | str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Tracer:
        """Create a child tracer writing to a separate file.

        Args:
            name: Child tracer name (used in parent events and default filename).
            path: JSONL file path. If not given, an auto-generated path is used
                  in the same directory as the parent trace.
            metadata: Extra metadata merged into the child's trace_start event.
                      ``parent_trace`` is always included automatically.
        """
        merged_metadata: dict[str, Any] = {"parent_trace": str(self._path)}
        if metadata:
            merged_metadata.update(metadata)

        if path is not None:
            child_tracer = Tracer(
                name,
                path=path,
                metadata=merged_metadata,
                capture_content=self._capture_content,
                backends=self._backends,
                flush_interval=self._flush_interval,
                flush_threshold=self._flush_threshold,
            )
        else:
            child_tracer = Tracer(
                name,
                trace_dir=self._path.parent,
                metadata=merged_metadata,
                capture_content=self._capture_content,
                backends=self._backends,
                flush_interval=self._flush_interval,
                flush_threshold=self._flush_threshold,
            )
        child_tracer._parent = self
        child_tracer._child_name = name
        return child_tracer

    def _write_child_started(self, name: str, path: Path) -> None:
        self._write(
            {
                "type": "event",
                "id": _uuid(),
                "parent_id": _get_parent_id(),
                "name": "child_started",
                "ts": _now(),
                "data": {
                    "child_name": name,
                    "child_file": path.stem + ".jsonl.gz",
                },
            }
        )
        with self._lock:
            self._stats_events += 1

    def _write_child_ended(self, name: str, child: Tracer) -> None:
        summary = {
            "name": name,
            "file": child._path.stem + ".jsonl.gz",
            "duration_s": round(
                (datetime.now(timezone.utc) - child._start_time).total_seconds(), 3
            )
            if child._start_time
            else 0.0,
            "spans": child._stats_spans,
            "total_input_tokens": child._stats_input_tokens,
            "total_output_tokens": child._stats_output_tokens,
            "total_cache_read_tokens": child._stats_cache_read_tokens,
            "total_cache_creation_tokens": child._stats_cache_creation_tokens,
            "total_reasoning_tokens": child._stats_reasoning_tokens,
        }
        self._children.append(summary)
        # Roll up child stats so parent trace_end.stats reflects the full tree
        with self._lock:
            self._stats_spans += child._stats_spans
            self._stats_input_tokens += child._stats_input_tokens
            self._stats_output_tokens += child._stats_output_tokens
            self._stats_cache_read_tokens += child._stats_cache_read_tokens
            self._stats_cache_creation_tokens += child._stats_cache_creation_tokens
            self._stats_reasoning_tokens += child._stats_reasoning_tokens
            self._stats_errors += child._stats_errors
            self._stats_events += 1
        self._write(
            {
                "type": "event",
                "id": _uuid(),
                "parent_id": _get_parent_id(),
                "name": "child_ended",
                "ts": _now(),
                "data": {
                    "child_name": name,
                    "child_file": child._path.stem + ".jsonl.gz",
                    "duration_s": summary["duration_s"],
                    "spans": summary["spans"],
                    "total_input_tokens": summary["total_input_tokens"],
                    "total_output_tokens": summary["total_output_tokens"],
                    "total_cache_read_tokens": summary["total_cache_read_tokens"],
                    "total_cache_creation_tokens": summary[
                        "total_cache_creation_tokens"
                    ],
                    "total_reasoning_tokens": summary["total_reasoning_tokens"],
                },
            }
        )


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


def subtrace(
    name: str,
    path: Path | str | None = None,
    *,
    trace_dir: Path | str | None = None,
    metadata: dict[str, Any] | None = None,
    backends: Sequence[Backend] | None = None,
) -> Tracer:
    """Create a child tracer if inside a trace context, otherwise a new root tracer.

    This handles the common pattern where a sub-unit of work (e.g. an agent)
    may run inside a parent trace or standalone::

        with subtrace("my_agent"):
            await agent.run(...)

    When a parent tracer is active, ``path`` and ``backends`` are inherited from
    the parent (and any explicit values are ignored). When no parent is active,
    a path is auto-generated from the name.

    Args:
        name: Name for the child tracer / trace session.
        path: JSONL file path. When not given, auto-generated from *name*.
              Ignored when a parent exists (child derives path from parent).
        trace_dir: Directory for auto-generated paths. Only used when no parent.
        metadata: Extra metadata for the trace_start event.
        backends: Backends for uploading traces. Only used when no parent
                  (children inherit parent backends).
    """
    parent = get_tracer()
    if parent is not None:
        return parent.child(name, metadata=metadata)
    return Tracer(
        name, path=path, trace_dir=trace_dir, metadata=metadata, backends=backends
    )
