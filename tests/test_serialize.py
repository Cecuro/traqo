"""Tests for serialization helpers."""

from __future__ import annotations

from pathlib import Path

from traqo.serialize import _serialize_value, serialize_args, serialize_error


class TestSerializeValue:
    def test_none(self):
        assert _serialize_value(None) is None

    def test_bool(self):
        assert _serialize_value(True) is True

    def test_int(self):
        assert _serialize_value(42) == 42

    def test_float(self):
        assert _serialize_value(3.14) == 3.14

    def test_short_string(self):
        assert _serialize_value("hello") == "hello"

    def test_truncation(self):
        long = "x" * 2000
        result = _serialize_value(long)
        assert len(result) == 1003  # 1000 + "..."
        assert result.endswith("...")

    def test_path(self):
        assert _serialize_value(Path("/tmp/test")) == "/tmp/test"

    def test_bytes(self):
        assert _serialize_value(b"hello") == "<5 bytes>"

    def test_list(self):
        assert _serialize_value([1, 2, 3]) == "[3 items]"

    def test_empty_list(self):
        assert _serialize_value([]) == "[0 items]"

    def test_tuple(self):
        assert _serialize_value((1, 2)) == "(2 items)"

    def test_set(self):
        assert _serialize_value({1, 2, 3}) == "{3 items}"

    def test_dict(self):
        assert _serialize_value({"a": 1, "b": 2}) == "{2 keys}"

    def test_object(self):
        class Foo:
            pass

        assert _serialize_value(Foo()) == "<Foo>"


class TestSerializeArgs:
    def test_basic(self):
        result = serialize_args({"a": 1, "b": "hello", "c": [1, 2, 3]})
        assert result == {"a": 1, "b": "hello", "c": "[3 items]"}


class TestSerializeError:
    def test_basic(self):
        result = serialize_error(ValueError("test error"))
        assert result == {"type": "ValueError", "message": "test error"}

    def test_truncation(self):
        long_msg = "x" * 1000
        result = serialize_error(RuntimeError(long_msg))
        assert len(result["message"]) == 503  # 500 + "..."
