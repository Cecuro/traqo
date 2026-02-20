"""Serialization helpers for trace events."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_MAX_STRING_LENGTH = 1000
_MAX_ERROR_LENGTH = 500


def _serialize_value(value: Any) -> Any:
    """Serialize a value to a JSON-safe summary for span inputs/outputs.

    Primitives pass through. Complex objects become type summaries.
    """
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        if len(value) > _MAX_STRING_LENGTH:
            return value[:_MAX_STRING_LENGTH] + "..."
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return f"<{len(value)} bytes>"
    if isinstance(value, list):
        return f"[{len(value)} items]"
    if isinstance(value, tuple):
        return f"({len(value)} items)"
    if isinstance(value, set):
        return f"{{{len(value)} items}}"
    if isinstance(value, dict):
        return f"{{{len(value)} keys}}"
    # Pydantic models
    if hasattr(value, "model_fields"):
        return f"<{type(value).__name__}>"
    return f"<{type(value).__name__}>"


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
    return {"type": type(exc).__name__, "message": msg}


def json_default(obj: Any) -> Any:
    """Default handler for json.dumps — converts non-serializable objects to strings."""
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return str(obj)


def to_json(event: dict[str, Any]) -> str:
    """Serialize an event dict to a JSON string."""
    return json.dumps(event, default=json_default, ensure_ascii=False)
