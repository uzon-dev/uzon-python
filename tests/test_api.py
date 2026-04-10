# SPDX-FileCopyrightText: © 2026 Suho Kang
# SPDX-License-Identifier: MIT
"""Tests for the public API (loads, dumps, load, dump)."""

import os
import tempfile
import pytest

import uzon
from uzon.types import UzonInt, UzonFloat, UzonEnum


class TestLoads:
    def test_basic(self):
        r = uzon.loads("x is 42")
        assert r["x"] == 42

    def test_struct(self):
        r = uzon.loads("s is { a is 1, b is 2 }")
        assert r["s"]["a"] == 1

    def test_plain_strips_enum(self):
        r = uzon.loads("x is red from red, green, blue", plain=True)
        assert r["x"] == "red"
        assert isinstance(r["x"], str)

    def test_plain_strips_typed_int(self):
        r = uzon.loads("x is 42", plain=True)
        assert r["x"] == 42
        assert type(r["x"]) is int

    def test_plain_strips_typed_float(self):
        r = uzon.loads("x is 3.14", plain=True)
        assert abs(r["x"] - 3.14) < 1e-10
        assert type(r["x"]) is float

    def test_plain_strips_tagged_union(self):
        r = uzon.loads("x is 7 named n from n as i32, s as string", plain=True)
        assert r["x"] == 7

    def test_plain_nested_struct(self):
        r = uzon.loads("s is { color is red from red, blue }", plain=True)
        assert r["s"]["color"] == "red"

    def test_plain_list(self):
        r = uzon.loads("xs is [1, 2, 3] as [i64]", plain=True)
        assert r["xs"] == [1, 2, 3]
        assert all(type(x) is int for x in r["xs"])

    def test_plain_tuple(self):
        r = uzon.loads("t is (1, 2)", plain=True)
        assert type(r["t"][0]) is int


class TestDumps:
    def test_basic(self):
        text = uzon.dumps({"x": 42})
        assert "x is 42" in text

    def test_indent(self):
        text = uzon.dumps({"s": {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}}, indent=2)
        assert "  a is 1" in text

    def test_roundtrip_simple(self):
        original = uzon.loads("x is 42\ny is true")
        text = uzon.dumps(original)
        roundtrip = uzon.loads(text)
        assert roundtrip["x"] == original["x"]
        assert roundtrip["y"] == original["y"]


class TestLoadDump:
    def test_load_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".uzon", delete=False) as f:
            f.write('x is 42\nname is "test"')
            f.flush()
            path = f.name
        try:
            r = uzon.load(path)
            assert r["x"] == 42
            assert r["name"] == "test"
        finally:
            os.unlink(path)

    def test_load_plain(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".uzon", delete=False) as f:
            f.write("x is 42 as i32")
            f.flush()
            path = f.name
        try:
            r = uzon.load(path, plain=True)
            assert type(r["x"]) is int
        finally:
            os.unlink(path)

    def test_dump_file(self):
        data = {"x": 42, "name": "test"}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".uzon", delete=False) as f:
            path = f.name
        try:
            uzon.dump(data, path)
            r = uzon.load(path)
            assert r["x"] == 42
            assert r["name"] == "test"
        finally:
            os.unlink(path)

    def test_dump_roundtrip(self):
        data = {"a": 1, "b": "hello", "c": True, "d": None}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".uzon", delete=False) as f:
            path = f.name
        try:
            uzon.dump(data, path)
            r = uzon.load(path)
            assert r["a"] == 1
            assert r["b"] == "hello"
            assert r["c"] is True
            assert r["d"] is None
        finally:
            os.unlink(path)


class TestVersion:
    def test_version_exists(self):
        assert hasattr(uzon, "__version__")
        assert isinstance(uzon.__version__, str)

    def test_version_format(self):
        parts = uzon.__version__.split(".")
        assert len(parts) == 3
        assert all(p.isdigit() for p in parts)


class TestExports:
    def test_all_exports(self):
        for name in uzon.__all__:
            assert hasattr(uzon, name), f"Missing export: {name}"


class TestErrors:
    def test_syntax_error(self):
        with pytest.raises(uzon.UzonSyntaxError):
            uzon.loads("x is is")

    def test_type_error(self):
        with pytest.raises(uzon.UzonTypeError):
            uzon.loads("x is 1 + true")

    def test_circular_error(self):
        with pytest.raises(uzon.UzonCircularError):
            uzon.loads("a is self.b\nb is self.a")

    def test_runtime_error(self):
        with pytest.raises(uzon.UzonRuntimeError):
            uzon.loads("x is undefined")
