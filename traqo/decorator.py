"""@trace decorator for automatic span instrumentation."""

from __future__ import annotations

import asyncio
import functools
import inspect
from typing import Any, Callable

from traqo.serialize import serialize_args, serialize_output
from traqo.tracer import get_tracer


def trace(
    name: str | None = None,
    *,
    capture_input: bool = True,
    capture_output: bool = True,
    metadata: dict[str, Any] | None = None,
    kind: str | None = None,
) -> Callable:
    """Decorator that wraps a function in a tracing span.

    When no tracer is active: pure passthrough, zero overhead.
    """

    def decorator(func: Callable) -> Callable:
        span_name = name or func.__name__
        sig = inspect.signature(func)

        def _extract_args(args: tuple, kwargs: dict) -> dict[str, Any]:
            try:
                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()
                return {k: v for k, v in bound.arguments.items() if k != "self"}
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

                input_data = (
                    serialize_args(_extract_args(args, kwargs))
                    if capture_input
                    else None
                )
                with tracer.span(
                    span_name, input=input_data, metadata=metadata, kind=kind
                ) as span:
                    result = await func(*args, **kwargs)
                    if capture_output:
                        span.set_output(serialize_output(result))
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

                input_data = (
                    serialize_args(_extract_args(args, kwargs))
                    if capture_input
                    else None
                )
                with tracer.span(
                    span_name, input=input_data, metadata=metadata, kind=kind
                ) as span:
                    result = func(*args, **kwargs)
                    if capture_output:
                        span.set_output(serialize_output(result))
                    return result

            return sync_wrapper

    return decorator
