"""Storage backend protocol and background executor utilities."""

from __future__ import annotations

import atexit
import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

# Module-level lazy singleton executor.
_executor: ThreadPoolExecutor | None = None
_executor_lock = threading.Lock()
_atexit_registered = False


def _get_executor() -> ThreadPoolExecutor:
    """Return the shared background executor, creating it on first use."""
    global _executor, _atexit_registered
    if _executor is None:
        with _executor_lock:
            if _executor is None:
                _executor = ThreadPoolExecutor(
                    max_workers=2,
                    thread_name_prefix="traqo-backend",
                )
                if not _atexit_registered:
                    atexit.register(shutdown_backends)
                    _atexit_registered = True
    return _executor


@runtime_checkable
class Backend(Protocol):
    """Protocol for storage backends.

    Backends receive trace data and can upload/forward it to external storage.
    All methods must not raise — implementations should catch and log internally.

    Lifecycle:
    - ``on_event`` is called after each event is written to the local JSONL file.
    - ``on_trace_complete`` is called after the trace file is fully written and closed.
    - ``close`` is called during process shutdown (via ``shutdown_backends``) or
      explicitly by the user.  It is NOT called automatically from ``Tracer.__exit__``
      — backends are long-lived and may be shared across multiple tracers.
    """

    def on_event(self, event: dict[str, Any]) -> None:
        """Called after each event is written to the local JSONL file.

        For streaming backends. Batch-upload backends should no-op here.
        """
        ...

    def on_trace_complete(self, trace_path: Path) -> Future | None:
        """Called after the trace file is fully written and closed.

        Returns a Future for async work (e.g. background upload), or None
        if work completed synchronously.
        """
        ...

    def close(self) -> None:
        """Release resources held by this backend."""
        ...


def submit_background(fn, *args, **kwargs) -> Future | None:
    """Submit work to the shared background thread pool.

    Returns the Future, or None if submission failed.
    Exceptions in *fn* are logged but never propagated.
    """

    def _safe_wrapper():
        try:
            return fn(*args, **kwargs)
        except Exception:
            logger.warning("traqo: backend operation failed", exc_info=True)
            return None

    try:
        return _get_executor().submit(_safe_wrapper)
    except Exception:
        logger.warning("traqo: failed to submit backend operation", exc_info=True)
        return None


def flush_backends() -> None:
    """Wait for all pending backend operations to complete.

    The executor is recreated afterwards so new work can still be submitted.
    Use this in long-running servers to periodically ensure uploads are done.
    """
    global _executor
    with _executor_lock:
        if _executor is not None:
            _executor.shutdown(wait=True)
            _executor = ThreadPoolExecutor(
                max_workers=2,
                thread_name_prefix="traqo-backend",
            )


def shutdown_backends() -> None:
    """Wait for pending operations then tear down the executor permanently.

    Called automatically via ``atexit`` so that uploads complete before the
    process exits.  Can also be called explicitly.
    """
    global _executor
    with _executor_lock:
        if _executor is not None:
            _executor.shutdown(wait=True)
            _executor = None
