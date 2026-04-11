# SPDX-FileCopyrightText: © 2026 Suho Kang
# SPDX-License-Identifier: MIT
"""Tests for the val factory (uzon.val)."""

import pytest

import uzon
from uzon import val
from uzon.types import UzonEnum, UzonFloat, UzonInt, UzonStruct, UzonTaggedUnion, UzonUnion


# ── integer types ─────────────────────────────────────────────────


class TestIntTypes:
    def test_i8(self):
        v = val.i8(127)
        assert isinstance(v, UzonInt)
        assert int(v) == 127
        assert v.type_name == "i8"

    def test_i8_min(self):
        v = val.i8(-128)
        assert int(v) == -128

    def test_i16(self):
        v = val.i16(32767)
        assert v.type_name == "i16"

    def test_i32(self):
        v = val.i32(2_147_483_647)
        assert v.type_name == "i32"

    def test_i64(self):
        v = val.i64(0)
        assert v.type_name == "i64"

    def test_u8(self):
        v = val.u8(255)
        assert v.type_name == "u8"
        assert int(v) == 255

    def test_u8_zero(self):
        v = val.u8(0)
        assert int(v) == 0

    def test_u16(self):
        v = val.u16(8080)
        assert v.type_name == "u16"
        assert int(v) == 8080

    def test_u32(self):
        v = val.u32(4_294_967_295)
        assert v.type_name == "u32"

    def test_u64(self):
        v = val.u64(0)
        assert v.type_name == "u64"

    def test_arbitrary_width_i128(self):
        v = val.i128(2**100)
        assert v.type_name == "i128"
        assert int(v) == 2**100


# ── integer overflow ──────────────────────────────────────────────


class TestIntOverflow:
    def test_i8_overflow(self):
        with pytest.raises(OverflowError, match="out of i8 range"):
            val.i8(128)

    def test_i8_underflow(self):
        with pytest.raises(OverflowError, match="out of i8 range"):
            val.i8(-129)

    def test_u8_negative(self):
        with pytest.raises(OverflowError, match="out of u8 range"):
            val.u8(-1)

    def test_u8_overflow(self):
        with pytest.raises(OverflowError, match="out of u8 range"):
            val.u8(256)

    def test_u16_overflow(self):
        with pytest.raises(OverflowError, match="out of u16 range"):
            val.u16(65536)


# ── integer type errors ───────────────────────────────────────────


class TestIntTypeErrors:
    def test_bool_rejected(self):
        with pytest.raises(TypeError, match="expects int"):
            val.i32(True)

    def test_float_rejected(self):
        with pytest.raises(TypeError, match="expects int"):
            val.i32(3.14)

    def test_str_rejected(self):
        with pytest.raises(TypeError, match="expects int"):
            val.i32("42")


# ── float types ───────────────────────────────────────────────────


class TestFloatTypes:
    def test_f32(self):
        v = val.f32(1.5)
        assert isinstance(v, UzonFloat)
        assert float(v) == 1.5
        assert v.type_name == "f32"

    def test_f64(self):
        v = val.f64(3.14)
        assert v.type_name == "f64"

    def test_f16(self):
        v = val.f16(0.5)
        assert v.type_name == "f16"

    def test_f80(self):
        v = val.f80(2.718)
        assert v.type_name == "f80"

    def test_f128(self):
        v = val.f128(1e100)
        assert v.type_name == "f128"

    def test_int_coerced_to_float(self):
        v = val.f64(42)
        assert isinstance(v, UzonFloat)
        assert float(v) == 42.0

    def test_bool_rejected(self):
        with pytest.raises(TypeError, match="expects number, got bool"):
            val.f64(True)

    def test_str_rejected(self):
        with pytest.raises(TypeError, match="expects number"):
            val.f64("1.5")


# ── unknown type ──────────────────────────────────────────────────


class TestUnknownType:
    def test_unknown_attr(self):
        with pytest.raises(AttributeError, match="Unknown UZON type"):
            val.xyz

    def test_unknown_prefix(self):
        with pytest.raises(AttributeError, match="Unknown UZON type"):
            val.g32


# ── struct ─────────────────────────────────────────────────────────


class TestStruct:
    def test_basic(self):
        v = val.struct({"host": "localhost", "port": val.u16(8080)})
        assert isinstance(v, UzonStruct)
        assert isinstance(v, dict)
        assert v["host"] == "localhost"
        assert v["port"] == 8080

    def test_with_type_name(self):
        v = val.struct({"host": "localhost"}, type_name="Server")
        assert v.type_name == "Server"

    def test_no_type_name(self):
        v = val.struct({"x": 1})
        assert v.type_name is None

    def test_dict_compatible(self):
        v = val.struct({"a": 1, "b": 2}, type_name="Pair")
        assert len(v) == 2
        assert list(v.keys()) == ["a", "b"]
        assert "a" in v

    def test_generator_emits_called(self):
        data = {"server": val.struct({"host": "localhost"}, type_name="Server")}
        text = uzon.dumps(data)
        assert "called Server" in text

    def test_generator_emits_as_for_repeated(self):
        s1 = val.struct({"x": 1}, type_name="Point")
        s2 = val.struct({"x": 2}, type_name="Point")
        data = {"a": s1, "b": s2}
        text = uzon.dumps(data)
        assert "called Point" in text
        assert "as Point" in text

    def test_roundtrip_named_struct(self):
        src = 'p is { x is 1, y is 2 } called Point'
        result = uzon.loads(src)
        assert isinstance(result["p"], UzonStruct)
        assert result["p"].type_name == "Point"
        assert result["p"]["x"] == 1


# ── enum ──────────────────────────────────────────────────────────


class TestEnum:
    def test_basic(self):
        v = val.enum("Red", ["Red", "Green", "Blue"])
        assert isinstance(v, UzonEnum)
        assert v.value == "Red"
        assert v.variants == ["Red", "Green", "Blue"]

    def test_with_type_name(self):
        v = val.enum("Red", ["Red", "Green", "Blue"], type_name="Color")
        assert v.type_name == "Color"

    def test_invalid_variant(self):
        with pytest.raises(ValueError, match="not a valid variant"):
            val.enum("Yellow", ["Red", "Green", "Blue"])


# ── union ─────────────────────────────────────────────────────────


class TestUnion:
    def test_basic(self):
        v = val.union(42, ["string", "i32"])
        assert isinstance(v, UzonUnion)
        assert v.value == 42
        assert v.types == ["string", "i32"]

    def test_with_type_name(self):
        v = val.union("hello", ["string", "i32"], type_name="StringOrInt")
        assert v.type_name == "StringOrInt"


# ── tagged union ──────────────────────────────────────────────────


class TestTaggedUnion:
    def test_basic(self):
        v = val.tagged("Ok", "done", {"Ok": "string", "Err": "string"})
        assert isinstance(v, UzonTaggedUnion)
        assert v.tag == "Ok"
        assert v.value == "done"
        assert v.variants == {"Ok": "string", "Err": "string"}

    def test_with_type_name(self):
        v = val.tagged("Ok", 0, {"Ok": "i32", "Err": "string"}, type_name="Result")
        assert v.type_name == "Result"

    def test_invalid_tag(self):
        with pytest.raises(ValueError, match="not a valid variant"):
            val.tagged("Unknown", 1, {"Ok": "i32", "Err": "string"})


# ── repr / dir ────────────────────────────────────────────────────


class TestMeta:
    def test_repr(self):
        assert repr(val) == "uzon.val"

    def test_dir_contains_common_types(self):
        d = dir(val)
        for name in ["i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64",
                      "f32", "f64", "enum", "union", "tagged"]:
            assert name in d
