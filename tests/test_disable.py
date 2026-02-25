"""Tests for global disable/enable."""

from __future__ import annotations

from pathlib import Path

import traqo
from traqo import Tracer, get_tracer, trace


class TestDisable:
    def test_disable_prevents_file_creation(self, tmp_path: Path):
        path = tmp_path / "should_not_exist.jsonl"
        traqo.disable()
        with Tracer(path):
            pass
        assert not path.exists()

    def test_disable_trace_decorator_passthrough(self, trace_file: Path):
        @trace()
        def add(a: int, b: int) -> int:
            return a + b

        traqo.disable()
        with Tracer(trace_file):
            result = add(1, 2)

        assert result == 3
        assert not trace_file.exists()

    def test_disable_get_tracer_returns_none(self, trace_file: Path):
        traqo.disable()
        with Tracer(trace_file):
            assert get_tracer() is None

    def test_enable_after_disable(self, trace_file: Path):
        traqo.disable()
        assert get_tracer() is None
        traqo.enable()
        with Tracer(trace_file) as t:
            assert get_tracer() is t

    def test_env_var_disable(self, monkeypatch, tmp_path: Path):
        """Test that TRAQO_DISABLED env var works.

        Note: Since the module is already imported, we test the mechanism
        by directly setting the flag.
        """
        traqo._disabled = True
        path = tmp_path / "env_test.jsonl"
        with Tracer(path):
            assert get_tracer() is None
        assert not path.exists()
        traqo._disabled = False
