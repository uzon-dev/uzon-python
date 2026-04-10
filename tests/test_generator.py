# SPDX-FileCopyrightText: © 2026 Suho Kang
# SPDX-License-Identifier: MIT
"""Tests for the UZON generator."""

import pytest

from uzon.generator import generate
from uzon.types import UzonInt, UzonFloat, UzonEnum, UzonUnion, UzonTaggedUnion, UzonUndefined


class TestGenerateBasic:
    def test_integer(self):
        text = generate({"x": 42})
        assert "x is 42" in text

    def test_float(self):
        text = generate({"pi": 3.14})
        assert "pi is 3.14" in text

    def test_bool_true(self):
        text = generate({"flag": True})
        assert "flag is true" in text

    def test_bool_false(self):
        text = generate({"flag": False})
        assert "flag is false" in text

    def test_null(self):
        text = generate({"x": None})
        assert "x is null" in text

    def test_string(self):
        text = generate({"name": "hello"})
        assert 'name is "hello"' in text

    def test_string_escape_newline(self):
        text = generate({"s": 'line\n"end'})
        assert '\\n' in text
        assert '\\"' in text

    def test_string_escape_brace(self):
        """§4.4.1: Braces must be escaped in generated strings."""
        text = generate({"s": "x{y}"})
        assert "\\{" in text

    def test_string_escape_tab(self):
        text = generate({"s": "a\tb"})
        assert "\\t" in text

    def test_string_escape_null(self):
        text = generate({"s": "a\0b"})
        assert "\\0" in text

    def test_multiple_bindings(self):
        text = generate({"a": 1, "b": 2, "c": 3})
        lines = text.strip().split("\n")
        assert len(lines) == 3


class TestGenerateTypedNumerics:
    def test_typed_int(self):
        text = generate({"x": UzonInt(5, "i32")})
        assert "5 as i32" in text

    def test_adoptable_int_no_annotation(self):
        text = generate({"x": UzonInt(5, "i64", adoptable=True)})
        assert "x is 5" in text
        assert "as" not in text

    def test_typed_float(self):
        text = generate({"x": UzonFloat(1.5, "f32")})
        assert "1.5 as f32" in text

    def test_adoptable_float_no_annotation(self):
        text = generate({"x": UzonFloat(1.5, "f64", adoptable=True)})
        assert "x is 1.5" in text
        assert "as" not in text


class TestGenerateCompounds:
    def test_struct_single_line(self):
        text = generate({"s": {"a": 1, "b": 2}})
        assert "a is 1" in text
        assert "b is 2" in text

    def test_empty_struct(self):
        text = generate({"s": {}})
        assert "{}" in text

    def test_nested_struct(self):
        text = generate({"s": {"inner": {"x": 1}}})
        assert "inner is" in text
        assert "x is 1" in text

    def test_list(self):
        text = generate({"arr": [1, 2, 3]})
        assert "[ 1, 2, 3 ]" in text

    def test_empty_list(self):
        text = generate({"arr": []})
        assert "[]" in text

    def test_tuple(self):
        text = generate({"t": (1, 2, 3)})
        assert "(1, 2, 3)" in text

    def test_single_tuple(self):
        text = generate({"t": (42,)})
        assert "(42,)" in text

    def test_empty_tuple(self):
        text = generate({"t": ()})
        assert "()" in text


class TestGenerateEnum:
    def test_basic_enum(self):
        text = generate({"color": UzonEnum("red", ["red", "green", "blue"])})
        assert "red from red, green, blue" in text

    def test_named_enum(self):
        text = generate({"color": UzonEnum("red", ["red", "green", "blue"], "Color")})
        assert "called Color" in text


class TestGenerateUnion:
    def test_untagged_union(self):
        text = generate({"x": UzonUnion(42, ["i32", "string"])})
        assert "42 from union i32, string" in text

    def test_tagged_union(self):
        text = generate({"x": UzonTaggedUnion(7, "n", {"n": "i32", "s": "string"})})
        assert "named n" in text
        assert "n as i32" in text
        assert "s as string" in text


class TestGenerateSpecial:
    def test_inf(self):
        text = generate({"x": float("inf")})
        assert "inf" in text

    def test_nan(self):
        text = generate({"x": float("nan")})
        assert "nan" in text

    def test_negative_zero(self):
        text = generate({"x": -0.0})
        assert "-0.0" in text

    def test_undefined(self):
        text = generate({"x": UzonUndefined})
        assert "undefined" in text


class TestGenerateKeywordEscape:
    def test_keyword_name(self):
        """§2.4: Keywords used as identifiers need @ escape."""
        text = generate({"true": 1})
        assert "@true is 1" in text

    def test_special_chars_quoting(self):
        """§2.3: Special characters in names need quoting."""
        text = generate({"my name": "x"})
        assert "'my name'" in text

    def test_empty_name(self):
        text = generate({"": "x"})
        assert "''" in text


class TestGenerateErrors:
    def test_non_dict_error(self):
        with pytest.raises(TypeError, match="must be a dict"):
            generate([1, 2, 3])

    def test_function_error(self):
        from uzon.types import UzonFunction
        with pytest.raises(TypeError, match="function"):
            generate({"f": UzonFunction([], "i64", [], None, None)})


class TestGenerateRoundTrip:
    """Test that generate output can be parsed back and evaluated to the same values."""

    def test_roundtrip_simple(self):
        from uzon.lexer import Lexer
        from uzon.parser import Parser
        from uzon.evaluator import Evaluator

        original = {"x": 42, "name": "hello", "flag": True, "n": None}
        text = generate(original)
        tokens = Lexer(text).tokenize()
        doc = Parser(tokens).parse()
        result = Evaluator().evaluate(doc)
        assert result["x"] == 42
        assert result["name"] == "hello"
        assert result["flag"] is True
        assert result["n"] is None

    def test_roundtrip_nested(self):
        from uzon.lexer import Lexer
        from uzon.parser import Parser
        from uzon.evaluator import Evaluator

        original = {"s": {"a": 1, "b": "two"}, "xs": [10, 20]}
        text = generate(original)
        tokens = Lexer(text).tokenize()
        doc = Parser(tokens).parse()
        result = Evaluator().evaluate(doc)
        assert result["s"]["a"] == 1
        assert result["s"]["b"] == "two"
        assert result["xs"] == [10, 20]
