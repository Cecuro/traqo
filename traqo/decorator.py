"""@trace decorator for automatic span instrumentation."""

from __future__ import annotations

import asyncio
import functools
import inspect
from datetime import datetime, timezone
from typing import Any, Callable

from traqo.serialize import serialize_args, serialize_error, serialize_output
from traqo.tracer import _get_parent_id, _now, _pop_span, _push_span, _uuid, get_tracer


def trace(
    name: str | None = None,
    *,
    capture_input: bool = True,
    capture_output: bool = True,
) -> Callable:
    """Decorator that wraps a function in a tracing span.

    When no tracer is active: pure passthrough, zero overhead.
    """

    def decorator(func: Callable) -> Callable:
        span_name = name or func.__name__
        sig = inspect.signature(func)

        def _extract_args(args: tuple, kwargs: dict) -> dict[str, Any]:
            """Bind arguments to parameter names, excluding 'self'."""
            try:
                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()
                return {
                    k: v for k, v in bound.arguments.items() if k != "self"
                }
            except TypeError:
                return {}

        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                import traqo

                if traqo._disabled:
                    return await func(*args, **kwargs)

                tracer = get_tracer()
                if tracer is None:
                    return await func(*args, **kwargs)

                span_id = _uuid()
                parent_id = _get_parent_id()
                start = datetime.now(timezone.utc)

                start_event: dict[str, Any] = {
                    "type": "span_start",
                    "id": span_id,
                    "parent_id": parent_id,
                    "name": span_name,
                    "ts": start.isoformat(),
                }
                if capture_input:
                    start_event["input"] = serialize_args(
                        _extract_args(args, kwargs)
                    )
                tracer._write(start_event)

                _push_span(span_id)
                try:
                    result = await func(*args, **kwargs)
                except BaseException as exc:
                    duration = (
                        datetime.now(timezone.utc) - start
                    ).total_seconds()
                    tracer._stats_spans += 1
                    tracer._stats_errors += 1
                    tracer._write(
                        {
                            "type": "span_end",
                            "id": span_id,
                            "parent_id": parent_id,
                            "name": span_name,
                            "ts": _now(),
                            "duration_s": round(duration, 3),
                            "status": "error",
                            "error": serialize_error(exc),
                        }
                    )
                    raise
                finally:
                    _pop_span()

                duration = (
                    datetime.now(timezone.utc) - start
                ).total_seconds()
                tracer._stats_spans += 1
                end_event: dict[str, Any] = {
                    "type": "span_end",
                    "id": span_id,
                    "parent_id": parent_id,
                    "name": span_name,
                    "ts": _now(),
                    "duration_s": round(duration, 3),
                    "status": "ok",
                }
                if capture_output:
                    end_event["output"] = serialize_output(result)
                tracer._write(end_event)
                return result

            return async_wrapper

        else:

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                import traqo

                if traqo._disabled:
                    return func(*args, **kwargs)

                tracer = get_tracer()
                if tracer is None:
                    return func(*args, **kwargs)

                span_id = _uuid()
                parent_id = _get_parent_id()
                start = datetime.now(timezone.utc)

                start_event: dict[str, Any] = {
                    "type": "span_start",
                    "id": span_id,
                    "parent_id": parent_id,
                    "name": span_name,
                    "ts": start.isoformat(),
                }
                if capture_input:
                    start_event["input"] = serialize_args(
                        _extract_args(args, kwargs)
                    )
                tracer._write(start_event)

                _push_span(span_id)
                try:
                    result = func(*args, **kwargs)
                except BaseException as exc:
                    duration = (
                        datetime.now(timezone.utc) - start
                    ).total_seconds()
                    tracer._stats_spans += 1
                    tracer._stats_errors += 1
                    tracer._write(
                        {
                            "type": "span_end",
                            "id": span_id,
                            "parent_id": parent_id,
                            "name": span_name,
                            "ts": _now(),
                            "duration_s": round(duration, 3),
                            "status": "error",
                            "error": serialize_error(exc),
                        }
                    )
                    raise
                finally:
                    _pop_span()

                duration = (
                    datetime.now(timezone.utc) - start
                ).total_seconds()
                tracer._stats_spans += 1
                end_event: dict[str, Any] = {
                    "type": "span_end",
                    "id": span_id,
                    "parent_id": parent_id,
                    "name": span_name,
                    "ts": _now(),
                    "duration_s": round(duration, 3),
                    "status": "ok",
                }
                if capture_output:
                    end_event["output"] = serialize_output(result)
                tracer._write(end_event)
                return result

            return sync_wrapper

    return decorator
