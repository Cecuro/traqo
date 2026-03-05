"""Shared test fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import traqo


@pytest.fixture(autouse=True)
def _reset_traqo():
    """Ensure traqo is enabled before each test."""
    traqo.enable()
    yield
    traqo.enable()


@pytest.fixture
def trace_file(tmp_path: Path) -> Path:
    return tmp_path / "test_trace.jsonl"


def _resolve_trace_path(path: Path) -> Path:
    """Find the actual trace file, preferring .jsonl.gz over raw .jsonl."""
    if path.name.endswith(".jsonl.gz"):
        if path.is_file():
            return path
        raise FileNotFoundError(f"No trace file found: {path}")
    gz_path = path.parent / (path.stem + ".jsonl.gz")
    if gz_path.is_file():
        return gz_path
    if path.is_file():
        return path
    raise FileNotFoundError(f"No trace file found: {path} or {gz_path}")


def read_events(path: Path) -> list[dict]:
    """Read all JSONL events from a trace file (.jsonl or .jsonl.gz)."""
    import gzip

    resolved = _resolve_trace_path(path)
    events = []
    with (
        gzip.open(resolved, "rt", encoding="utf-8")
        if resolved.name.endswith(".gz")
        else open(resolved, encoding="utf-8") as f
    ):
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events
