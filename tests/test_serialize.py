"""Tests for serialization helpers."""

from __future__ import annotations

import dataclasses
import json
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from traqo.serialize import _serialize_value, serialize_args, serialize_error, to_json


class TestSerializeValue:
    # -- Primitives --

    def test_none(self):
        assert _serialize_value(None) is None

    def test_bool(self):
        assert _serialize_value(True) is True
        assert _serialize_value(False) is False

    def test_int(self):
        assert _serialize_value(42) == 42

    def test_large_int(self):
        big = 2**64
        assert _serialize_value(big) == big

    def test_float(self):
        assert _serialize_value(3.14) == 3.14

    def test_short_string(self):
        assert _serialize_value("hello") == "hello"

    def test_long_string_preserved(self):
        """No string truncation — local files have no size limit."""
        long = "x" * 100_000
        assert _serialize_value(long) == long

    # -- NaN / Infinity --

    def test_nan(self):
        assert _serialize_value(float("nan")) is None

    def test_infinity(self):
        assert _serialize_value(float("inf")) is None

    def test_neg_infinity(self):
        assert _serialize_value(float("-inf")) is None

    # -- stdlib types --

    def test_datetime(self):
        dt = datetime(2026, 2, 23, 12, 0, 0, tzinfo=timezone.utc)
        assert _serialize_value(dt) == "2026-02-23T12:00:00+00:00"

    def test_date(self):
        d = date(2026, 2, 23)
        assert _serialize_value(d) == "2026-02-23"

    def test_uuid(self):
        u = uuid4()
        assert _serialize_value(u) == str(u)

    def test_uuid_specific(self):
        u = UUID("12345678-1234-5678-1234-567812345678")
        assert _serialize_value(u) == "12345678-1234-5678-1234-567812345678"

    def test_enum(self):
        class Color(Enum):
            RED = "red"
            BLUE = "blue"

        assert _serialize_value(Color.RED) == "red"

    def test_enum_int_value(self):
        class Status(Enum):
            OK = 200
            ERROR = 500

        assert _serialize_value(Status.OK) == 200

    def test_path(self):
        assert _serialize_value(Path("/tmp/test")) == "/tmp/test"

    def test_bytes(self):
        assert _serialize_value(b"hello") == "<5 bytes>"

    # -- Collections --

    def test_list(self):
        assert _serialize_value([1, 2, 3]) == [1, 2, 3]

    def test_empty_list(self):
        assert _serialize_value([]) == []

    def test_tuple(self):
        assert _serialize_value((1, 2)) == [1, 2]

    def test_set(self):
        result = _serialize_value({1, 2, 3})
        assert sorted(result) == [1, 2, 3]

    def test_frozenset(self):
        result = _serialize_value(frozenset({1, 2, 3}))
        assert sorted(result) == [1, 2, 3]

    def test_dict(self):
        assert _serialize_value({"a": 1, "b": 2}) == {"a": 1, "b": 2}

    def test_nested_dict(self):
        result = _serialize_value({"city": "Tokyo", "report": {"weather": "Sunny"}})
        assert result == {"city": "Tokyo", "report": {"weather": "Sunny"}}

    def test_deep_nesting_preserved(self):
        """No depth limit — deep structures pass through."""
        deep = {"a": {"b": {"c": {"d": {"e": {"f": {"g": "deep value"}}}}}}}
        result = _serialize_value(deep)
        assert result["a"]["b"]["c"]["d"]["e"]["f"]["g"] == "deep value"

    def test_large_list_preserved(self):
        """No item limit — large collections pass through."""
        big = list(range(10_000))
        result = _serialize_value(big)
        assert len(result) == 10_000
        assert result[-1] == 9999

    def test_large_dict_preserved(self):
        """No item limit — large dicts pass through."""
        big = {f"k{i}": i for i in range(10_000)}
        result = _serialize_value(big)
        assert len(result) == 10_000

    # -- Circular references --

    def test_circular_dict(self):
        a: dict = {"key": "value"}
        a["self"] = a
        result = _serialize_value(a)
        assert result["key"] == "value"
        assert result["self"] == "<circular ref>"

    def test_circular_list(self):
        a: list = [1, 2]
        a.append(a)
        result = _serialize_value(a)
        assert result[0] == 1
        assert result[2] == "<circular ref>"

    def test_sibling_refs_not_circular(self):
        """Same object in two positions is NOT a cycle — both should serialize."""
        shared = {"x": 1}
        parent = [shared, shared]
        result = _serialize_value(parent)
        assert result == [{"x": 1}, {"x": 1}]

    def test_circular_object(self):
        class Node:
            def __init__(self, name):
                self.name = name
                self.child = None

        a = Node("a")
        b = Node("b")
        a.child = b
        b.child = a
        result = _serialize_value(a)
        assert result["name"] == "a"
        assert result["child"]["name"] == "b"
        assert result["child"]["child"] == "<circular ref>"

    # -- Pydantic --

    def test_pydantic_model(self):
        class FakeModel:
            def model_dump(self):
                return {"name": "test", "value": 42}

        result = _serialize_value(FakeModel())
        assert result == {"name": "test", "value": 42}

    # -- Dataclass --

    def test_dataclass(self):
        @dataclasses.dataclass
        class Point:
            x: int
            y: int

        result = _serialize_value(Point(1, 2))
        assert result == {"x": 1, "y": 2}

    def test_nested_dataclass(self):
        @dataclasses.dataclass
        class Inner:
            val: str

        @dataclasses.dataclass
        class Outer:
            inner: Inner
            name: str

        result = _serialize_value(Outer(inner=Inner(val="hello"), name="test"))
        assert result == {"inner": {"val": "hello"}, "name": "test"}

    # -- __slots__ objects --

    def test_slots_object(self):
        class SlotObj:
            __slots__ = ("x", "y")

            def __init__(self, x, y):
                self.x = x
                self.y = y

        result = _serialize_value(SlotObj(1, 2))
        assert result == {"x": 1, "y": 2}

    # -- Generic objects with __dict__ --

    def test_object_with_dict(self):
        class Foo:
            def __init__(self):
                self.a = 1
                self.b = "hello"

        result = _serialize_value(Foo())
        assert result == {"a": 1, "b": "hello"}

    def test_empty_slots_object(self):
        """Object with empty __slots__ serializes to empty dict."""

        class Opaque:
            __slots__ = ()

        result = _serialize_value(Opaque())
        assert result == {}

    # -- Graceful degradation --

    def test_graceful_degradation(self):
        """If serialization of any part fails, return a placeholder string."""

        class Bomb:
            @property
            def __dict__(self):
                raise RuntimeError("boom")

        result = _serialize_value(Bomb())
        assert "serialization failed" in result

    # -- numpy (optional) --

    @pytest.fixture
    def np(self):
        return pytest.importorskip("numpy")

    def test_numpy_int(self, np):
        assert _serialize_value(np.int64(42)) == 42
        assert isinstance(_serialize_value(np.int64(42)), int)

    def test_numpy_float(self, np):
        result = _serialize_value(np.float32(3.14))
        assert isinstance(result, float)

    def test_numpy_bool(self, np):
        assert _serialize_value(np.bool_(True)) is True

    def test_numpy_array(self, np):
        arr = np.array([1, 2, 3])
        assert _serialize_value(arr) == [1, 2, 3]

    def test_numpy_2d_array(self, np):
        arr = np.array([[1, 2], [3, 4]])
        assert _serialize_value(arr) == [[1, 2], [3, 4]]


class TestSerializeArgs:
    def test_basic(self):
        result = serialize_args({"a": 1, "b": "hello", "c": [1, 2, 3]})
        assert result == {"a": 1, "b": "hello", "c": [1, 2, 3]}

    def test_with_datetime(self):
        dt = datetime(2026, 1, 1, tzinfo=timezone.utc)
        result = serialize_args({"ts": dt})
        assert result == {"ts": "2026-01-01T00:00:00+00:00"}


class TestSerializeError:
    def test_basic(self):
        result = serialize_error(ValueError("test error"))
        assert result["type"] == "ValueError"
        assert result["message"] == "test error"
        assert "traceback" in result
        assert "ValueError: test error" in result["traceback"]

    def test_truncation(self):
        long_msg = "x" * 1000
        result = serialize_error(RuntimeError(long_msg))
        assert len(result["message"]) == 503  # 500 + "..."

    def test_traceback_from_handler(self):
        """Traceback captured from a real exception handler."""
        try:
            raise RuntimeError("boom")
        except RuntimeError as exc:
            result = serialize_error(exc)
        assert "RuntimeError: boom" in result["traceback"]
        assert "raise RuntimeError" in result["traceback"]

    def test_traceback_truncation(self):
        """Very long tracebacks get truncated."""
        try:
            raise RuntimeError("x" * 3000)
        except RuntimeError as exc:
            result = serialize_error(exc)
        assert (
            len(result["traceback"]) <= 2003 + 10
        )  # _MAX_TRACEBACK_LENGTH + "..." + margin


class TestToJson:
    def test_nan_safe(self):
        """to_json should never produce invalid JSON with NaN/Infinity."""
        event = {"value": float("nan"), "other": float("inf")}
        result = to_json(event)
        parsed = json.loads(result)
        assert parsed["value"] is None
        assert parsed["other"] is None

    def test_datetime_in_event(self):
        """Datetime values in events should serialize via json_default."""
        dt = datetime(2026, 2, 23, tzinfo=timezone.utc)
        event = {"ts": dt}
        result = to_json(event)
        parsed = json.loads(result)
        assert parsed["ts"] == "2026-02-23T00:00:00+00:00"
