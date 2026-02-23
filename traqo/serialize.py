"""Serialization helpers for trace events."""

from __future__ import annotations

import dataclasses
import json
import math
import traceback as _traceback_mod
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import UUID

_MAX_ERROR_LENGTH = 500
_MAX_TRACEBACK_LENGTH = 2000


def _is_numpy(value: Any) -> bool:
    """Check if a value is a numpy type without importing numpy."""
    return type(value).__module__.startswith("numpy")


def _serialize_numpy(value: Any, *, _seen: set[int] | None = None) -> Any:
    """Serialize numpy types to JSON-safe representations."""
    import numpy as np

    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return _serialize_value(value.tolist(), _seen=_seen)
    return str(value)


def _serialize_value(value: Any, *, _seen: set[int] | None = None) -> Any:
    """Serialize a value to a JSON-safe representation for span inputs/outputs.

    No limits on string length, depth, or collection size — traqo writes to
    local files so there is no reason to truncate user data. Safety guards:
    circular reference detection and a try/except wrapper for graceful
    degradation.
    """
    try:
        # 1. Primitives
        if value is None or isinstance(value, bool):
            return value
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            if math.isnan(value) or math.isinf(value):
                return None
            return value
        if isinstance(value, str):
            return value

        # 2. Common stdlib types
        if isinstance(value, (datetime, date)):
            return value.isoformat()
        if isinstance(value, UUID):
            return str(value)
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, bytes):
            return f"<{len(value)} bytes>"

        # 3. Circular reference guard
        if _seen is None:
            _seen = set()
        obj_id = id(value)
        if obj_id in _seen:
            return "<circular ref>"
        _seen.add(obj_id)
        try:
            # 4. Collections
            if isinstance(value, dict):
                return {k: _serialize_value(v, _seen=_seen) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [_serialize_value(v, _seen=_seen) for v in value]
            if isinstance(value, (set, frozenset)):
                return [_serialize_value(v, _seen=_seen) for v in sorted(value, key=str)]

            # 5. Pydantic
            if hasattr(value, "model_dump"):
                return _serialize_value(value.model_dump(), _seen=_seen)

            # 6. Dataclass
            if dataclasses.is_dataclass(value) and not isinstance(value, type):
                return _serialize_value(value.__dict__, _seen=_seen)

            # 7. numpy (optional, no hard dependency)
            if _is_numpy(value):
                return _serialize_numpy(value, _seen=_seen)

            # 8. __slots__ objects
            if hasattr(value, "__slots__"):
                d = {s: getattr(value, s) for s in value.__slots__ if hasattr(value, s)}
                return _serialize_value(d, _seen=_seen)

            # 9. Generic objects with __dict__
            if hasattr(value, "__dict__"):
                return _serialize_value(vars(value), _seen=_seen)

            # 10. Fallback
            return str(value)
        finally:
            _seen.discard(obj_id)

    except Exception:
        return f"<{type(value).__name__}: serialization failed>"


def serialize_args(args: dict[str, Any]) -> dict[str, Any]:
    """Serialize function arguments for span_start input."""
    return {k: _serialize_value(v) for k, v in args.items()}


def serialize_output(value: Any) -> Any:
    """Serialize a return value for span_end output."""
    return _serialize_value(value)


def serialize_error(exc: BaseException) -> dict[str, str]:
    """Serialize an exception for span_end error field."""
    msg = str(exc)
    if len(msg) > _MAX_ERROR_LENGTH:
        msg = msg[:_MAX_ERROR_LENGTH] + "..."
    tb = _traceback_mod.format_exc()
    if tb == "NoneType: None\n":
        tb = "".join(_traceback_mod.format_exception(type(exc), exc, exc.__traceback__))
    if len(tb) > _MAX_TRACEBACK_LENGTH:
        tb = tb[:_MAX_TRACEBACK_LENGTH] + "..."
    return {"type": type(exc).__name__, "message": msg, "traceback": tb}


def json_default(obj: Any) -> Any:
    """Default handler for json.dumps — delegates to _serialize_value."""
    return _serialize_value(obj)


def to_json(event: dict[str, Any]) -> str:
    """Serialize an event dict to a JSON string.

    Pre-processes through _serialize_value to handle NaN/Infinity and
    non-JSON-native types before json.dumps sees them.
    """
    return json.dumps(
        _serialize_value(event), default=json_default, ensure_ascii=False, allow_nan=False
    )
