# SPDX-FileCopyrightText: © 2026 Suho Kang
# SPDX-License-Identifier: MIT
"""Tests for the UZON evaluator."""

import math
import os
import pytest

from uzon.lexer import Lexer
from uzon.parser import Parser
from uzon.evaluator import Evaluator
from uzon.types import UzonInt, UzonFloat, UzonEnum, UzonUnion, UzonTaggedUnion, UzonUndefined
from uzon.errors import UzonCircularError, UzonRuntimeError, UzonSyntaxError, UzonTypeError


def evaluate(source: str) -> dict:
    tokens = Lexer(source).tokenize()
    doc = Parser(tokens).parse()
    return Evaluator().evaluate(doc)


# ── Literals ───────────────────────────────────────────────────────

class TestLiterals:
    def test_integer(self):
        r = evaluate("x is 42")
        assert r["x"] == 42
        assert isinstance(r["x"], UzonInt)

    def test_negative_integer(self):
        r = evaluate("x is -7")
        assert r["x"] == -7

    def test_float(self):
        r = evaluate("x is 3.14")
        assert isinstance(r["x"], UzonFloat)
        assert abs(r["x"] - 3.14) < 1e-10

    def test_bool_true(self):
        r = evaluate("x is true")
        assert r["x"] is True

    def test_bool_false(self):
        r = evaluate("x is false")
        assert r["x"] is False

    def test_null(self):
        r = evaluate("x is null")
        assert r["x"] is None

    def test_string(self):
        r = evaluate('x is "hello"')
        assert r["x"] == "hello"

    def test_inf(self):
        r = evaluate("x is inf")
        assert r["x"] == float("inf")

    def test_nan(self):
        r = evaluate("x is nan")
        assert math.isnan(r["x"])

    def test_hex_integer(self):
        r = evaluate("x is 0xFF")
        assert r["x"] == 255

    def test_binary_integer(self):
        r = evaluate("x is 0b1010")
        assert r["x"] == 10

    def test_octal_integer(self):
        r = evaluate("x is 0o17")
        assert r["x"] == 15

    def test_negative_zero(self):
        r = evaluate("x is -0.0")
        assert r["x"] == 0.0
        assert math.copysign(1.0, r["x"]) < 0


# ── Arithmetic ─────────────────────────────────────────────────────

class TestArithmetic:
    def test_add(self):
        r = evaluate("x is 1 + 2")
        assert r["x"] == 3

    def test_subtract(self):
        r = evaluate("x is 10 - 4")
        assert r["x"] == 6

    def test_multiply(self):
        r = evaluate("x is 3 * 7")
        assert r["x"] == 21

    def test_divide_int_truncating(self):
        """§5.4: Integer division truncates toward zero."""
        r = evaluate("x is 7 / 2")
        assert r["x"] == 3

    def test_divide_neg_truncating(self):
        """§5.4: Negative division truncates toward zero, not floor."""
        r = evaluate("x is -7 / 2")
        assert r["x"] == -3

    def test_modulo(self):
        r = evaluate("x is 10 % 3")
        assert r["x"] == 1

    def test_exponent(self):
        r = evaluate("x is 2 ^ 10")
        assert r["x"] == 1024

    def test_float_divide(self):
        r = evaluate("x is 1.0 / 3.0")
        assert abs(r["x"] - 1.0 / 3.0) < 1e-10

    def test_divide_by_zero_int(self):
        with pytest.raises(UzonRuntimeError, match="Division by zero"):
            evaluate("x is 1 / 0")

    def test_divide_by_zero_float(self):
        r = evaluate("x is 1.0 / 0.0")
        assert r["x"] == float("inf")

    def test_precedence(self):
        r = evaluate("x is 2 + 3 * 4")
        assert r["x"] == 14

    def test_grouping(self):
        r = evaluate("x is (2 + 3) * 4")
        assert r["x"] == 20


# ── Comparison ─────────────────────────────────────────────────────

class TestComparison:
    def test_less_than(self):
        r = evaluate("x is 1 < 2")
        assert r["x"] is True

    def test_greater_than(self):
        r = evaluate("x is 5 > 3")
        assert r["x"] is True

    def test_less_equal(self):
        r = evaluate("x is 2 <= 2")
        assert r["x"] is True

    def test_greater_equal(self):
        r = evaluate("x is 1 >= 2")
        assert r["x"] is False

    def test_string_compare(self):
        r = evaluate('x is "a" < "b"')
        assert r["x"] is True


# ── Equality ───────────────────────────────────────────────────────

class TestEquality:
    def test_is_equal(self):
        r = evaluate("x is 1 is 1")
        assert r["x"] is True

    def test_is_not_equal(self):
        r = evaluate("x is 1 is not 2")
        assert r["x"] is True

    def test_null_equality(self):
        r = evaluate("x is null is null")
        assert r["x"] is True

    def test_nan_not_equal_to_self(self):
        """§5.6: NaN is not equal to itself."""
        r = evaluate("x is nan is nan")
        assert r["x"] is False


# ── Logic ──────────────────────────────────────────────────────────

class TestLogic:
    def test_and_true(self):
        r = evaluate("x is true and true")
        assert r["x"] is True

    def test_and_short_circuit(self):
        r = evaluate("x is false and true")
        assert r["x"] is False

    def test_or_true(self):
        r = evaluate("x is false or true")
        assert r["x"] is True

    def test_or_short_circuit(self):
        r = evaluate("x is true or false")
        assert r["x"] is True

    def test_not(self):
        r = evaluate("x is not false")
        assert r["x"] is True


# ── Bare identifier reference and dependencies ──────────────────

class TestSelfReference:
    def test_forward_ref(self):
        """Dependency graph resolves forward references."""
        r = evaluate("a is b + 1\nb is 10")
        assert r["a"] == 11

    def test_self_exclusion(self):
        """§5.12: bare identifier inside its own binding skips to parent scope."""
        r = evaluate("x is { a is 10, b is a * 2 }")
        assert r["x"]["b"] == 20

    def test_circular_error(self):
        with pytest.raises(UzonCircularError):
            evaluate("a is b\nb is a")

    def test_duplicate_binding_error(self):
        with pytest.raises(UzonSyntaxError, match="Duplicate binding"):
            evaluate("x is 1\nx is 2")


# ── Or else ────────────────────────────────────────────────────────

class TestOrElse:
    def test_defined_value(self):
        r = evaluate("x is 42\ny is x or else 0")
        assert r["y"] == 42

    def test_undefined_fallback(self):
        r = evaluate("y is missing or else 99")
        assert r["y"] == 99


# ── Struct ─────────────────────────────────────────────────────────

class TestStruct:
    def test_basic_struct(self):
        r = evaluate("x is { a is 1, b is 2 }")
        assert r["x"]["a"] == 1
        assert r["x"]["b"] == 2

    def test_nested_struct(self):
        r = evaluate("x is { inner is { v is 99 } }")
        assert r["x"]["inner"]["v"] == 99

    def test_struct_self_ref(self):
        r = evaluate("x is { a is 10, b is a + 5 }")
        assert r["x"]["b"] == 15

    def test_struct_member_access(self):
        r = evaluate("s is { x is 1, y is 2 }\nv is s.x")
        assert r["v"] == 1

    def test_nested_member_access(self):
        r = evaluate("s is { inner is { val is 99 } }\nv is s.inner.val")
        assert r["v"] == 99


# ── With / Extends ─────────────────────────────────────────────────

class TestWithExtends:
    def test_with_override(self):
        r = evaluate("base is { x is 1, y is 2 }\nresult is base with { x is 10 }")
        assert r["result"]["x"] == 10
        assert r["result"]["y"] == 2

    def test_with_nonexistent_field_error(self):
        with pytest.raises(UzonRuntimeError, match="does not exist"):
            evaluate("base is { x is 1 }\nresult is base with { z is 5 }")

    def test_plus_add_field(self):
        r = evaluate("base is { x is 1 }\nresult is base plus { y is 2 }")
        assert r["result"]["x"] == 1
        assert r["result"]["y"] == 2

    def test_plus_no_new_field_error(self):
        with pytest.raises(UzonTypeError, match="must add at least one"):
            evaluate("base is { x is 1 }\nresult is base plus { x is 2 }")

    def test_plus_override_and_add(self):
        r = evaluate("base is { x is 1 }\nresult is base plus { x is 10, y is 2 }")
        assert r["result"]["x"] == 10
        assert r["result"]["y"] == 2


# ── List ───────────────────────────────────────────────────────────

class TestList:
    def test_basic_list(self):
        r = evaluate("x is [1, 2, 3] as [i64]")
        assert len(r["x"]) == 3

    def test_empty_list_typed(self):
        r = evaluate("x is [] as [i32]")
        assert r["x"] == []

    def test_empty_list_untyped_error(self):
        with pytest.raises(UzonTypeError, match="Empty list"):
            evaluate("x is []")

    def test_list_indexing(self):
        """List elements accessible by ordinal names."""
        r = evaluate("x is [10, 20, 30] as [i64]\ny is x.first")
        assert r["y"] == 10


# ── Tuple ──────────────────────────────────────────────────────────

class TestTuple:
    def test_basic_tuple(self):
        r = evaluate("x is (1, 2, 3)")
        assert r["x"] == (1, 2, 3)

    def test_empty_tuple(self):
        r = evaluate("x is ()")
        assert r["x"] == ()

    def test_tuple_ordinal_access(self):
        r = evaluate("x is (10, 20, 30)\ny is x.second")
        assert r["y"] == 20


# ── String interpolation ──────────────────────────────────────────

class TestStringInterpolation:
    def test_basic_interpolation(self):
        r = evaluate('n is 42\nx is "value: {n}"')
        assert r["x"] == "value: 42"

    def test_undefined_interpolation_error(self):
        with pytest.raises(UzonRuntimeError, match="interpolate undefined"):
            evaluate('x is "val: {missing}"')

    def test_multiple_interpolation(self):
        r = evaluate('a is "hello"\nb is "world"\nc is "{a} {b}"')
        assert r["c"] == "hello world"


# ── If expression ──────────────────────────────────────────────────

class TestIfExpr:
    def test_if_true(self):
        r = evaluate("x is if true then 1 else 2")
        assert r["x"] == 1

    def test_if_false(self):
        r = evaluate("x is if false then 1 else 2")
        assert r["x"] == 2

    def test_non_bool_condition_error(self):
        with pytest.raises(UzonTypeError, match="must be bool"):
            evaluate("x is if 1 then 2 else 3")

    def test_if_with_self_ref(self):
        r = evaluate("flag is true\nx is if flag then 10 else 20")
        assert r["x"] == 10


# ── Case expression ───────────────────────────────────────────────

class TestCaseExpr:
    def test_basic_case(self):
        r = evaluate('x is case 2 when 1 then "a" when 2 then "b" else "c"')
        assert r["x"] == "b"

    def test_case_else(self):
        r = evaluate('x is case 99 when 1 then "a" else "other"')
        assert r["x"] == "other"

    def test_case_with_tagged_union(self):
        r = evaluate(
            'x is "hi" named s from n as i32, s as string\n'
            'y is case named x when n then "number" when s then "string" else "unknown"'
        )
        assert r["y"] == "string"


# ── Type annotation (as) ──────────────────────────────────────────

class TestTypeAnnotation:
    def test_as_i32(self):
        r = evaluate("x is 42 as i32")
        assert isinstance(r["x"], UzonInt)
        assert r["x"].type_name == "i32"

    def test_as_f64(self):
        r = evaluate("x is 1.5 as f64")
        assert isinstance(r["x"], UzonFloat)
        assert r["x"].type_name == "f64"

    def test_as_wrong_type_error(self):
        with pytest.raises(UzonTypeError, match="requires an integer"):
            evaluate('x is "hello" as i32')


# ── Type conversion (to) ──────────────────────────────────────────

class TestConversion:
    def test_float_to_int(self):
        r = evaluate("x is 3.7 to i32")
        assert r["x"] == 3
        assert isinstance(r["x"], UzonInt)

    def test_int_to_float(self):
        r = evaluate("x is 5 to f64")
        assert r["x"] == 5.0
        assert isinstance(r["x"], UzonFloat)

    def test_int_to_string(self):
        r = evaluate("x is 42 to string")
        assert r["x"] == "42"

    def test_string_to_int(self):
        r = evaluate('x is "100" to i32')
        assert r["x"] == 100

    def test_null_to_string(self):
        r = evaluate("x is null to string")
        assert r["x"] == "null"

    def test_bool_to_string(self):
        r = evaluate("x is true to string")
        assert r["x"] == "true"


# ── Enum ───────────────────────────────────────────────────────────

class TestEnum:
    def test_basic_enum(self):
        r = evaluate("x is red from red, green, blue")
        assert isinstance(r["x"], UzonEnum)
        assert r["x"].value == "red"

    def test_invalid_variant_error(self):
        with pytest.raises(UzonTypeError, match="is not a variant"):
            evaluate("x is yellow from red, green, blue")


# ── Union / Tagged union ──────────────────────────────────────────

class TestUnion:
    def test_untagged_union(self):
        r = evaluate("x is 42 from union i32, f64, string")
        assert isinstance(r["x"], UzonUnion)
        assert r["x"].value == 42

    def test_tagged_union(self):
        r = evaluate("x is 7 named n from n as i32, s as string")
        assert isinstance(r["x"], UzonTaggedUnion)
        assert r["x"].tag == "n"
        assert r["x"].value == 7

    def test_is_named(self):
        r = evaluate("x is 7 named n from n as i32, s as string\ny is x is named n")
        assert r["y"] is True

    def test_is_not_named(self):
        r = evaluate("x is 7 named n from n as i32, s as string\ny is x is not named s")
        assert r["y"] is True


# ── Functions ──────────────────────────────────────────────────────

class TestFunctions:
    def test_basic_function(self):
        r = evaluate("f is function n as i32 returns i32 { n + 1 }\nx is f(5)")
        assert r["x"] == 6

    def test_function_multiple_args(self):
        r = evaluate("f is function a as i32, b as i32 returns i32 { a + b }\nx is f(3, 4)")
        assert r["x"] == 7

    def test_function_wrong_arg_type_error(self):
        with pytest.raises(UzonTypeError, match="type mismatch"):
            evaluate('f is function n as i32 returns i32 { n }\nx is f("hello")')

    def test_function_wrong_return_type_error(self):
        with pytest.raises(UzonTypeError, match="return type mismatch"):
            evaluate('f is function returns i32 { "hello" }\nx is f()')

    def test_function_with_closure(self):
        r = evaluate(
            "factor is 10\n"
            "scale is function n as i64 returns i64 { n * factor }\n"
            "y is scale(5)"
        )
        assert r["y"] == 50


# ── Are binding ────────────────────────────────────────────────────

class TestAreBinding:
    def test_basic_are(self):
        r = evaluate('names are "a", "b", "c"')
        assert r["names"] == ["a", "b", "c"]

    def test_are_with_type(self):
        r = evaluate("ids are 1, 2, 3 as [i32]")
        assert len(r["ids"]) == 3
        assert all(isinstance(v, UzonInt) for v in r["ids"])


# ── Field extraction ──────────────────────────────────────────────

class TestFieldExtraction:
    def test_is_of(self):
        r = evaluate("config is { port is 8080 }\nport is of config")
        assert r["port"] == 8080


# ── Concat / Repeat ───────────────────────────────────────────────

class TestConcatRepeat:
    def test_string_concat(self):
        r = evaluate('x is "hello" ++ " " ++ "world"')
        assert r["x"] == "hello world"

    def test_list_concat(self):
        r = evaluate("x is [1, 2] ++ [3, 4] as [i64]")
        assert r["x"] == [1, 2, 3, 4]

    def test_string_repeat(self):
        r = evaluate('x is "ab" ** 3')
        assert r["x"] == "ababab"

    def test_list_repeat(self):
        r = evaluate("x is ([1, 2] ** 2) as [i64]")
        assert r["x"] == [1, 2, 1, 2]

    def test_repeat_zero(self):
        """Bug fix: ** 0 is allowed and produces empty."""
        r = evaluate('x is "ab" ** 0')
        assert r["x"] == ""


# ── In operator ────────────────────────────────────────────────────

class TestInOperator:
    def test_in_list(self):
        r = evaluate("x is [1, 2, 3] as [i64]\ny is 2 in x")
        assert r["y"] is True

    def test_not_in_list(self):
        r = evaluate("x is [1, 2, 3] as [i64]\ny is 5 in x")
        assert r["y"] is False

    def test_in_struct(self):
        r = evaluate('x is { a is 1 }\ny is "a" in std.keys(x)')
        assert r["y"] is True


# ── Env ────────────────────────────────────────────────────────────

class TestEnv:
    def test_env_defined(self):
        os.environ["UZON_TEST_VAR"] = "hello"
        try:
            r = evaluate("x is env.UZON_TEST_VAR")
            assert r["x"] == "hello"
        finally:
            del os.environ["UZON_TEST_VAR"]

    def test_env_undefined(self):
        r = evaluate("x is env.NONEXISTENT_UZON_VAR or else \"default\"")
        assert r["x"] == "default"


# ── Called (type naming) ───────────────────────────────────────────

class TestCalled:
    def test_called_enum(self):
        r = evaluate("x is red from red, green, blue called Color")
        assert isinstance(r["x"], UzonEnum)
        assert r["x"].type_name == "Color"

    def test_called_struct(self):
        r = evaluate("p is { x is 1, y is 2 } called Point")
        assert r["p"]["x"] == 1


# ── Std library ────────────────────────────────────────────────────

class TestStdLibrary:
    def test_std_len_list(self):
        r = evaluate("x is [1, 2, 3] as [i64]\ny is std.len(x)")
        assert r["y"] == 3

    def test_std_len_string(self):
        r = evaluate('x is std.len("hello")')
        assert r["x"] == 5

    def test_std_has_list(self):
        r = evaluate("x is [1, 2, 3] as [i64]\ny is std.has(x, 2)")
        assert r["y"] is True

    def test_std_has_struct(self):
        r = evaluate('x is { a is 1 }\ny is std.has(x, "a")')
        assert r["y"] is True

    def test_std_get_list(self):
        r = evaluate("x is [10, 20, 30] as [i64]\ny is std.get(x, 1)")
        assert r["y"] == 20

    def test_std_keys(self):
        r = evaluate("x is { a is 1, b is 2 }\ny is std.keys(x)")
        assert set(r["y"]) == {"a", "b"}

    def test_std_values(self):
        r = evaluate("x is { a is 1, b is 2 }\ny is std.values(x)")
        assert set(r["y"]) == {1, 2}

    def test_std_trim(self):
        r = evaluate('x is std.trim("  hello  ")')
        assert r["x"] == "hello"

    def test_std_join(self):
        r = evaluate('parts are "a", "b", "c"\nx is std.join(parts, "-")')
        assert r["x"] == "a-b-c"

    def test_std_replace(self):
        r = evaluate('x is std.replace("hello world", "world", "there")')
        assert r["x"] == "hello there"

    def test_std_split(self):
        r = evaluate('x is std.split("a,b,c", ",")')
        assert r["x"] == ["a", "b", "c"]

    def test_std_map(self):
        r = evaluate(
            "xs is [1, 2, 3] as [i64]\n"
            "ys is std.map(xs, function x as i64 returns i64 { x * 2 })"
        )
        assert r["ys"] == [2, 4, 6]

    def test_std_filter(self):
        r = evaluate(
            "xs is [1, 2, 3, 4] as [i64]\n"
            "ys is std.filter(xs, function x as i64 returns bool { x > 2 })"
        )
        assert r["ys"] == [3, 4]

    def test_std_reduce(self):
        r = evaluate(
            "xs is [1, 2, 3, 4] as [i64]\n"
            "sum is std.reduce(xs, 0, function acc as i64, x as i64 returns i64 { acc + x })"
        )
        assert r["sum"] == 10

    def test_std_sort(self):
        r = evaluate(
            "xs is [3, 1, 2] as [i64]\n"
            "sorted is std.sort(xs, function a as i64, b as i64 returns bool { a < b })"
        )
        assert r["sorted"] == [1, 2, 3]

    def test_std_isNan(self):
        r = evaluate("x is std.isNan(nan)")
        assert r["x"] is True

    def test_std_isInf(self):
        r = evaluate("x is std.isInf(inf)")
        assert r["x"] is True

    def test_std_isFinite(self):
        r = evaluate("x is std.isFinite(1.0)")
        assert r["x"] is True

    def test_std_lower(self):
        r = evaluate('x is std.lower("Hello, World!")')
        assert r["x"] == "hello, world!"

    def test_std_lower_no_change(self):
        r = evaluate('x is std.lower("already lower")')
        assert r["x"] == "already lower"

    def test_std_lower_type_error(self):
        with pytest.raises(UzonTypeError, match="std.lower"):
            evaluate("x is std.lower(42)")

    def test_std_upper(self):
        r = evaluate('x is std.upper("hello")')
        assert r["x"] == "HELLO"

    def test_std_upper_no_change(self):
        r = evaluate('x is std.upper("ALREADY UPPER")')
        assert r["x"] == "ALREADY UPPER"

    def test_std_upper_type_error(self):
        with pytest.raises(UzonTypeError, match="std.upper"):
            evaluate("x is std.upper(true)")

    def test_std_get_struct(self):
        r = evaluate('x is { a is 1 }\ny is std.get(x, "a")')
        assert r["y"] == 1

    def test_std_get_out_of_bounds(self):
        r = evaluate("x is [1, 2] as [i64]\ny is std.get(x, 10) or else -1")
        assert r["y"] == -1


# ── Undefined assignment ──────────────────────────────────────────

class TestUndefined:
    def test_literal_undefined_error(self):
        with pytest.raises(UzonRuntimeError, match="Cannot assign literal"):
            evaluate("x is undefined")

    def test_undefined_propagation(self):
        r = evaluate("x is missing or else 42")
        assert r["x"] == 42

    def test_undefined_member_access(self):
        r = evaluate("s is { a is 1 }\nx is s.missing or else 0")
        assert r["x"] == 0


# ── Struct import from string (§7) ──────────────────────────────

class TestStructImportFromString:
    """§7: struct import should work when evaluating from a string (loads)."""

    def test_import_from_string_uses_cwd(self, tmp_path, monkeypatch):
        """struct import in loads() resolves paths relative to CWD."""
        (tmp_path / "data.uzon").write_text("x is 42", encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        r = evaluate('d is struct "data.uzon"')
        assert r["d"]["x"] == 42

    def test_import_from_string_nested(self, tmp_path, monkeypatch):
        """Nested struct imports from string context."""
        (tmp_path / "inner.uzon").write_text("val is 99", encoding="utf-8")
        (tmp_path / "outer.uzon").write_text('inner is struct "inner.uzon"', encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        r = evaluate('o is struct "outer.uzon"')
        assert r["o"]["inner"]["val"] == 99

    def test_import_chain_error_message(self, tmp_path, monkeypatch):
        """Errors in imported files show import chain stack trace."""
        (tmp_path / "bad.uzon").write_text("x is 1 + true", encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        with pytest.raises(UzonTypeError) as exc_info:
            evaluate('d is struct "bad.uzon"')
        msg = str(exc_info.value)
        assert "imported from" in msg

    def test_import_chain_nested_error(self, tmp_path, monkeypatch):
        """Nested import errors show full chain."""
        (tmp_path / "bad.uzon").write_text("x is 1 + true", encoding="utf-8")
        (tmp_path / "mid.uzon").write_text('b is struct "bad.uzon"', encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        with pytest.raises(UzonTypeError) as exc_info:
            evaluate('d is struct "mid.uzon"')
        msg = str(exc_info.value)
        # Should have two levels of import chain
        assert msg.count("imported from") == 2

    def test_import_chain_shows_string_origin(self, tmp_path, monkeypatch):
        """Import chain traces back to <string> origin."""
        (tmp_path / "bad.uzon").write_text("x is 1 + true", encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        with pytest.raises(UzonTypeError) as exc_info:
            evaluate('d is struct "bad.uzon"')
        msg = str(exc_info.value)
        assert "<string>" in msg

    def test_import_not_found_from_string(self, tmp_path, monkeypatch):
        """File not found error still works from string context."""
        monkeypatch.chdir(tmp_path)
        with pytest.raises(UzonRuntimeError, match="Import file not found"):
            evaluate('d is struct "nonexistent.uzon"')

    def test_circular_import_from_string(self, tmp_path, monkeypatch):
        """Circular import detection works from string context."""
        (tmp_path / "a.uzon").write_text('b is struct "b.uzon"', encoding="utf-8")
        (tmp_path / "b.uzon").write_text('a is struct "a.uzon"', encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        with pytest.raises(UzonCircularError, match="Circular import"):
            evaluate('x is struct "a.uzon"')
