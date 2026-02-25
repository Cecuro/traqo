"""Read and parse traqo JSONL trace files.

Provides typed access to trace data so consumers don't need to know the
raw JSONL event schema.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class LLMSpan:
    """A parsed LLM span from a trace file."""

    model: str
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    cache_read_tokens: int
    cache_creation_tokens: int
    duration_s: float
    status: str
    name: str
    span_id: str


def iter_llm_spans(path: Path) -> Iterator[LLMSpan]:
    """Yield LLM spans from a traqo JSONL trace file.

    Only yields ``span_end`` events with ``kind="llm"`` that contain
    token usage metadata. Silently skips malformed lines.

    Args:
        path: Path to a ``.jsonl`` trace file.

    Yields:
        LLMSpan for each completed LLM call in the trace.
    """
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            if event.get("type") != "span_end" or event.get("kind") != "llm":
                continue

            metadata = event.get("metadata")
            if not isinstance(metadata, dict):
                continue

            usage = metadata.get("token_usage")
            if not isinstance(usage, dict):
                continue

            yield LLMSpan(
                model=metadata.get("model", "unknown"),
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
                reasoning_tokens=usage.get("reasoning_tokens", 0),
                cache_read_tokens=usage.get("cache_read_tokens", 0),
                cache_creation_tokens=usage.get("cache_creation_tokens", 0),
                duration_s=event.get("duration_s", 0.0),
                status=event.get("status", "ok"),
                name=event.get("name", ""),
                span_id=event.get("id", ""),
            )


def aggregate_tokens(
    path: Path,
) -> dict[str, dict[str, int]]:
    """Aggregate token usage by model from a trace file.

    Args:
        path: Path to a ``.jsonl`` trace file.

    Returns:
        Dict mapping model name to ``{"input": N, "output": N}`` totals.
    """
    by_model: dict[str, dict[str, int]] = {}
    for span in iter_llm_spans(path):
        if span.model not in by_model:
            by_model[span.model] = {"input": 0, "output": 0}
        by_model[span.model]["input"] += span.input_tokens
        by_model[span.model]["output"] += span.output_tokens
    return by_model
