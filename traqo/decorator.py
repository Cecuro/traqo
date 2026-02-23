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
    ignore_arguments: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    tags: list[str] | None = None,
    kind: str | None = None,
) -> Callable:
    """Decorator that wraps a function in a tracing span.

    When no tracer is active: pure passthrough, zero overhead.
    Supports sync functions, async functions, generators, and async generators.

    Args:
        ignore_arguments: List of argument names to exclude from captured input
            (e.g. ``["password", "api_key"]``).
    """

    _ignore = set(ignore_arguments) if ignore_arguments else set()

    def decorator(func: Callable) -> Callable:
        span_name = name or func.__name__
        sig = inspect.signature(func)

        def _extract_args(args: tuple, kwargs: dict) -> dict[str, Any]:
            try:
                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()
                return {
                    k: v
                    for k, v in bound.arguments.items()
                    if k != "self" and k not in _ignore
                }
            except TypeError:
                return {}

        def _should_passthrough() -> bool:
            import traqo

            if traqo._disabled:
                return True
            return get_tracer() is None

        def _make_input(args: tuple, kwargs: dict) -> Any:
            if not capture_input:
                return None
            return serialize_args(_extract_args(args, kwargs))

        if inspect.isasyncgenfunction(func):

            @functools.wraps(func)
            async def async_gen_wrapper(*args, **kwargs):
                if _should_passthrough():
                    async for item in func(*args, **kwargs):
                        yield item
                    return

                tracer = get_tracer()
                input_data = _make_input(args, kwargs)
                with tracer.span(
                    span_name, input=input_data, metadata=metadata, tags=tags, kind=kind
                ) as span:
                    collected: list[Any] = []
                    async for item in func(*args, **kwargs):
                        collected.append(item)
                        yield item
                    if capture_output and collected:
                        span.set_output(serialize_output(collected))

            return async_gen_wrapper

        elif inspect.isgeneratorfunction(func):

            @functools.wraps(func)
            def gen_wrapper(*args, **kwargs):
                if _should_passthrough():
                    yield from func(*args, **kwargs)
                    return

                tracer = get_tracer()
                input_data = _make_input(args, kwargs)
                with tracer.span(
                    span_name, input=input_data, metadata=metadata, tags=tags, kind=kind
                ) as span:
                    collected: list[Any] = []
                    for item in func(*args, **kwargs):
                        collected.append(item)
                        yield item
                    if capture_output and collected:
                        span.set_output(serialize_output(collected))

            return gen_wrapper

        elif asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                if _should_passthrough():
                    return await func(*args, **kwargs)

                tracer = get_tracer()
                input_data = _make_input(args, kwargs)
                with tracer.span(
                    span_name, input=input_data, metadata=metadata, tags=tags, kind=kind
                ) as span:
                    result = await func(*args, **kwargs)
                    if capture_output:
                        span.set_output(serialize_output(result))
                    return result

            return async_wrapper

        else:

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                if _should_passthrough():
                    return func(*args, **kwargs)

                tracer = get_tracer()
                input_data = _make_input(args, kwargs)
                with tracer.span(
                    span_name, input=input_data, metadata=metadata, tags=tags, kind=kind
                ) as span:
                    result = func(*args, **kwargs)
                    if capture_output:
                        span.set_output(serialize_output(result))
                    return result

            return sync_wrapper

    return decorator
