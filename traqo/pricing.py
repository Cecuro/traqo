"""Model pricing data and cost estimation.

Prices are per 1M tokens, sourced from https://models.dev (sst/models.dev).
Run ``python scripts/update_pricing.py`` to refresh pricing_data.json.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_DATA_PATH = Path(__file__).parent / "pricing_data.json"

# {model_id: {"input": $/1M, "output": $/1M, "cache_read": $/1M, "cache_write": $/1M}}
_PRICING: dict[str, dict[str, float]] = json.loads(_DATA_PATH.read_text("utf-8"))

# Provider prefixes to strip (litellm / router conventions)
_PREFIXES = ("azure/", "openai/", "anthropic/", "google/", "bedrock/", "vertex_ai/")


def _normalize_model(model: str) -> str | None:
    """Normalize model name to a known pricing key, or None."""
    # Strip provider prefixes
    for prefix in _PREFIXES:
        if model.startswith(prefix):
            model = model[len(prefix) :]
            break

    # Direct match
    if model in _PRICING:
        return model

    # Try stripping date suffix (e.g. "gpt-4o-2024-11-20" → "gpt-4o")
    # Walk from the end looking for a known base
    parts = model.split("-")
    for i in range(len(parts) - 1, 0, -1):
        candidate = "-".join(parts[:i])
        if candidate in _PRICING:
            return candidate

    return None


def estimate_cost(
    model: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read_tokens: int = 0,
    cache_creation_tokens: int = 0,
) -> float | None:
    """Estimate cost in USD for a model call. Returns None for unknown models.

    ``input_tokens`` is the **total** input token count (including cached).
    Cached tokens are subtracted before applying the base input price so they
    are only charged at their respective cache rate.
    """
    key = _normalize_model(model)
    if key is None:
        return None
    prices = _PRICING[key]
    non_cached = max(0, input_tokens - cache_read_tokens - cache_creation_tokens)
    cost = 0.0
    cost += non_cached * prices.get("input", 0) / 1_000_000
    cost += output_tokens * prices.get("output", 0) / 1_000_000
    cost += cache_read_tokens * prices.get("cache_read", 0) / 1_000_000
    cost += cache_creation_tokens * prices.get("cache_write", 0) / 1_000_000
    return cost


def add_cost(usage: dict[str, Any], model: str) -> None:
    """Add estimated cost to a usage dict in-place (no-op for unknown models)."""

    def _int(v: Any) -> int:
        return v if isinstance(v, int) else 0

    cost = estimate_cost(
        model,
        input_tokens=_int(usage.get("input_tokens", 0)),
        output_tokens=_int(usage.get("output_tokens", 0)),
        cache_read_tokens=_int(usage.get("cache_read_tokens", 0)),
        cache_creation_tokens=_int(usage.get("cache_creation_tokens", 0)),
    )
    if cost is not None:
        usage["cost"] = cost
