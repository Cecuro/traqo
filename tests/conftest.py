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


def read_events(path: Path) -> list[dict]:
    """Read all JSONL events from a trace file."""
    events = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events
