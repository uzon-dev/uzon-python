# SPDX-FileCopyrightText: © 2026 Suho Kang
# SPDX-License-Identifier: MIT
"""Tests for the UZON parser."""

import pytest

from uzon.ast_nodes import (
    AreBinding, BinaryOp, Binding, BoolLiteral, CaseExpr, Conversion,
    Document, EnvRef, FieldExtraction, FloatLiteral, FromEnum, FromUnion,
    FunctionCall, FunctionExpr, Grouping, Identifier, IfExpr,
    InfLiteral, IntegerLiteral, ListLiteral, MemberAccess,
    NamedVariant, NanLiteral, NullLiteral, OrElse,
    StringLiteral, StructExtension, StructImport, StructLiteral,
    StructOverride, TupleLiteral, TypeAnnotation, TypeExpr,
    UnaryOp, UndefinedLiteral, WhenClause,
)
from uzon.errors import UzonSyntaxError
from uzon.lexer import Lexer
from uzon.parser import Parser


def parse(source: str) -> Document:
    """Tokenize and parse source into a Document AST."""
    tokens = Lexer(source).tokenize()
    return Parser(tokens).parse()


# ── Basic bindings ──────────────────────────────────────────────────

class TestBasicBindings:
    def test_integer_binding(self):
        doc = parse("x is 42")
        assert len(doc.bindings) == 1
        b = doc.bindings[0]
        assert isinstance(b, Binding)
        assert b.name == "x"
        assert isinstance(b.value, IntegerLiteral)
        assert b.value.value == "42"

    def test_negative_integer(self):
        doc = parse("x is -42")
        b = doc.bindings[0]
        assert isinstance(b.value, IntegerLiteral)
        assert b.value.value == "-42"

    def test_float_binding(self):
        doc = parse("x is 3.14")
        b = doc.bindings[0]
        assert isinstance(b.value, FloatLiteral)
        assert b.value.value == "3.14"

    def test_string_binding(self):
        doc = parse('name is "hello"')
        b = doc.bindings[0]
        assert isinstance(b.value, StringLiteral)
        assert b.value.parts == ["hello"]

    def test_bool_binding(self):
        doc = parse("flag is true")
        assert isinstance(doc.bindings[0].value, BoolLiteral)
        assert doc.bindings[0].value.value is True

    def test_bool_false(self):
        doc = parse("flag is false")
        assert isinstance(doc.bindings[0].value, BoolLiteral)
        assert doc.bindings[0].value.value is False

    def test_null_binding(self):
        doc = parse("x is null")
        assert isinstance(doc.bindings[0].value, NullLiteral)

    def test_undefined_binding(self):
        doc = parse("x is undefined")
        assert isinstance(doc.bindings[0].value, UndefinedLiteral)

    def test_inf_binding(self):
        doc = parse("x is inf")
        assert isinstance(doc.bindings[0].value, InfLiteral)

    def test_nan_binding(self):
        doc = parse("x is nan")
        assert isinstance(doc.bindings[0].value, NanLiteral)

    def test_multiple_bindings_newline(self):
        doc = parse("x is 1\ny is 2")
        assert len(doc.bindings) == 2
        assert doc.bindings[0].name == "x"
        assert doc.bindings[1].name == "y"

    def test_multiple_bindings_comma(self):
        doc = parse("x is 1, y is 2")
        assert len(doc.bindings) == 2

    def test_three_bindings(self):
        doc = parse("x is 1\ny is 2\nz is 3")
        assert len(doc.bindings) == 3

    def test_identifier_binding(self):
        doc = parse("x is hello")
        b = doc.bindings[0]
        assert isinstance(b.value, Identifier)
        assert b.value.name == "hello"

    def test_called(self):
        doc = parse("x is 42 called MyType")
        assert doc.bindings[0].called == "MyType"


# ── Arithmetic expressions (§5.3) ──────────────────────────────────

class TestArithmetic:
    def test_addition(self):
        doc = parse("x is 1 + 2")
        b = doc.bindings[0]
        assert isinstance(b.value, BinaryOp)
        assert b.value.op == "+"

    def test_subtraction(self):
        doc = parse("x is 5 - 3")
        b = doc.bindings[0]
        assert isinstance(b.value, BinaryOp)
        assert b.value.op == "-"

    def test_multiplication(self):
        doc = parse("x is 2 * 3")
        b = doc.bindings[0]
        assert isinstance(b.value, BinaryOp)
        assert b.value.op == "*"

    def test_division(self):
        doc = parse("x is 10 / 2")
        b = doc.bindings[0]
        assert isinstance(b.value, BinaryOp)
        assert b.value.op == "/"

    def test_modulo(self):
        doc = parse("x is 10 % 3")
        b = doc.bindings[0]
        assert isinstance(b.value, BinaryOp)
        assert b.value.op == "%"

    def test_precedence_mul_over_add(self):
        """§5.5: Multiplication binds tighter than addition."""
        doc = parse("x is 1 + 2 * 3")
        b = doc.bindings[0]
        assert b.value.op == "+"
        assert isinstance(b.value.right, BinaryOp)
        assert b.value.right.op == "*"

    def test_left_associativity(self):
        """§5.5: Addition is left-associative."""
        doc = parse("x is 1 + 2 + 3")
        b = doc.bindings[0]
        assert b.value.op == "+"
        assert isinstance(b.value.left, BinaryOp)
        assert b.value.left.op == "+"

    def test_exponentiation(self):
        doc = parse("x is 2 ^ 3")
        b = doc.bindings[0]
        assert isinstance(b.value, BinaryOp)
        assert b.value.op == "^"

    def test_exponentiation_right_assoc(self):
        """§5.5: Exponentiation is right-associative."""
        doc = parse("x is 2 ^ 3 ^ 2")
        b = doc.bindings[0]
        assert b.value.op == "^"
        assert isinstance(b.value.right, BinaryOp)
        assert b.value.right.op == "^"


# ── Comparison (§5.4) ──────────────────────────────────────────────

class TestComparison:
    def test_less_than(self):
        doc = parse("x is 1 < 2")
        assert doc.bindings[0].value.op == "<"

    def test_less_equal(self):
        doc = parse("x is 1 <= 2")
        assert doc.bindings[0].value.op == "<="

    def test_greater_than(self):
        doc = parse("x is 1 > 2")
        assert doc.bindings[0].value.op == ">"

    def test_greater_equal(self):
        doc = parse("x is 1 >= 2")
        assert doc.bindings[0].value.op == ">="


# ── Equality (§5.1, §5.2) ──────────────────────────────────────────

class TestEquality:
    def test_is_equality(self):
        doc = parse("x is 1 is 1")
        b = doc.bindings[0]
        assert isinstance(b.value, BinaryOp)
        assert b.value.op == "is"

    def test_is_not(self):
        doc = parse("x is 1 is not 0")
        b = doc.bindings[0]
        assert isinstance(b.value, BinaryOp)
        assert b.value.op == "is not"

    def test_is_named(self):
        doc = parse("x is y is named ok")
        b = doc.bindings[0]
        assert isinstance(b.value, BinaryOp)
        assert b.value.op == "is named"

    def test_is_not_named(self):
        doc = parse("x is y is not named err")
        b = doc.bindings[0]
        assert isinstance(b.value, BinaryOp)
        assert b.value.op == "is not named"


# ── Logic (§5.6) ───────────────────────────────────────────────────

class TestLogic:
    def test_and(self):
        doc = parse("x is true and false")
        b = doc.bindings[0]
        assert isinstance(b.value, BinaryOp)
        assert b.value.op == "and"

    def test_or(self):
        doc = parse("x is true or false")
        b = doc.bindings[0]
        assert isinstance(b.value, BinaryOp)
        assert b.value.op == "or"

    def test_not(self):
        doc = parse("x is not true")
        b = doc.bindings[0]
        assert isinstance(b.value, UnaryOp)
        assert b.value.op == "not"

    def test_and_or_precedence(self):
        """§5.5: AND binds tighter than OR."""
        doc = parse("x is a or b and c")
        b = doc.bindings[0]
        assert b.value.op == "or"
        assert isinstance(b.value.right, BinaryOp)
        assert b.value.right.op == "and"


# ── Membership (§5.8.1) ────────────────────────────────────────────

class TestMembership:
    def test_in(self):
        doc = parse("x is 1 in [1, 2, 3]")
        b = doc.bindings[0]
        assert isinstance(b.value, BinaryOp)
        assert b.value.op == "in"

    def test_not_in_list(self):
        """§5.8.1: 'not (x in list)' via logical not."""
        doc = parse("x is not (1 in [1, 2, 3])")
        b = doc.bindings[0]
        assert isinstance(b.value, UnaryOp)
        assert b.value.op == "not"


# ── Or else (§5.7) ─────────────────────────────────────────────────

class TestOrElse:
    def test_or_else(self):
        doc = parse("x is y or else 5")
        b = doc.bindings[0]
        assert isinstance(b.value, OrElse)

    def test_or_else_lowest_precedence(self):
        """§5.5: or else has lowest precedence."""
        doc = parse("x is 1 + 2 or else 0")
        b = doc.bindings[0]
        assert isinstance(b.value, OrElse)
        assert isinstance(b.value.left, BinaryOp)


# ── Collection operators (§5.8) ────────────────────────────────────

class TestCollectionOps:
    def test_concat(self):
        doc = parse('x is "a" ++ "b"')
        b = doc.bindings[0]
        assert isinstance(b.value, BinaryOp)
        assert b.value.op == "++"

    def test_repeat(self):
        doc = parse('x is "ha" ** 3')
        b = doc.bindings[0]
        assert isinstance(b.value, BinaryOp)
        assert b.value.op == "**"


# ── Member access (§5.5) ──────────────────────────────────────────

class TestMemberAccess:
    def test_identifier_member(self):
        doc = parse("a is 1\nx is a")
        b = doc.bindings[1]
        assert isinstance(b.value, Identifier)
        assert b.value.name == "a"

    def test_chained_member(self):
        doc = parse("config is { port is 8080 }\nx is config.port")
        b = doc.bindings[1]
        assert isinstance(b.value, MemberAccess)
        assert b.value.member == "port"
        assert isinstance(b.value.object, Identifier)

    def test_env_member(self):
        doc = parse("x is env.HOME")
        b = doc.bindings[0]
        assert isinstance(b.value, MemberAccess)
        assert isinstance(b.value.object, EnvRef)
        assert b.value.member == "HOME"

    def test_numeric_member(self):
        """§5.5: Numeric indexing via dot."""
        doc = parse("list is [1, 2]\nx is list.0")
        b = doc.bindings[1]
        assert isinstance(b.value, MemberAccess)
        assert b.value.member == "0"


# ── Unary operators (§5.5) ────────────────────────────────────────

class TestUnary:
    def test_unary_minus(self):
        doc = parse("y is 5\nx is -y")
        b = doc.bindings[1]
        assert isinstance(b.value, UnaryOp)
        assert b.value.op == "-"

    def test_not(self):
        doc = parse("x is not false")
        b = doc.bindings[0]
        assert isinstance(b.value, UnaryOp)
        assert b.value.op == "not"


# ── Type system (§6) ──────────────────────────────────────────────

class TestTypeSystem:
    def test_type_annotation(self):
        doc = parse("x is 42 as Integer")
        b = doc.bindings[0]
        assert isinstance(b.value, TypeAnnotation)
        assert b.value.type.name == "Integer"

    def test_conversion(self):
        doc = parse("x is 3.14 to Integer")
        b = doc.bindings[0]
        assert isinstance(b.value, Conversion)
        assert b.value.type.name == "Integer"

    def test_list_type(self):
        doc = parse("x is [1, 2] as [Integer]")
        b = doc.bindings[0]
        assert isinstance(b.value, TypeAnnotation)
        assert b.value.type.is_list
        assert b.value.type.inner.name == "Integer"

    def test_tuple_type(self):
        doc = parse("x is (1, 2) as (Integer, String)")
        b = doc.bindings[0]
        assert isinstance(b.value, TypeAnnotation)
        assert b.value.type.is_tuple
        assert len(b.value.type.elements) == 2

    def test_null_type(self):
        doc = parse("x is null as null")
        b = doc.bindings[0]
        assert isinstance(b.value, TypeAnnotation)
        assert b.value.type.name == "null"

    def test_dotted_type(self):
        doc = parse("inner is { color is red from red, green, blue called RGB }\nx is inner.color as inner.RGB")
        b = doc.bindings[1]
        assert isinstance(b.value, TypeAnnotation)
        assert b.value.type.path == ["inner", "RGB"]

    def test_called(self):
        doc = parse("x is red from red, green, blue called RGB")
        b = doc.bindings[0]
        assert b.called == "RGB"


# ── Compounds (§3.2, §3.3, §3.4) ──────────────────────────────────

class TestCompounds:
    def test_struct_literal(self):
        doc = parse("x is { a is 1, b is 2 }")
        b = doc.bindings[0]
        assert isinstance(b.value, StructLiteral)
        assert len(b.value.fields) == 2
        assert b.value.fields[0].name == "a"
        assert b.value.fields[1].name == "b"

    def test_struct_newline_separated(self):
        doc = parse("x is {\n  a is 1\n  b is 2\n}")
        b = doc.bindings[0]
        assert isinstance(b.value, StructLiteral)
        assert len(b.value.fields) == 2

    def test_nested_struct(self):
        doc = parse("x is { inner is { y is 1 } }")
        b = doc.bindings[0]
        assert isinstance(b.value, StructLiteral)
        inner = b.value.fields[0].value
        assert isinstance(inner, StructLiteral)

    def test_empty_struct(self):
        doc = parse("x is {}")
        b = doc.bindings[0]
        assert isinstance(b.value, StructLiteral)
        assert len(b.value.fields) == 0

    def test_list_literal(self):
        doc = parse("x is [1, 2, 3]")
        b = doc.bindings[0]
        assert isinstance(b.value, ListLiteral)
        assert len(b.value.elements) == 3

    def test_empty_list(self):
        doc = parse("x is []")
        b = doc.bindings[0]
        assert isinstance(b.value, ListLiteral)
        assert len(b.value.elements) == 0

    def test_list_trailing_comma(self):
        doc = parse("x is [1, 2, 3,]")
        b = doc.bindings[0]
        assert isinstance(b.value, ListLiteral)
        assert len(b.value.elements) == 3

    def test_tuple_literal(self):
        doc = parse("x is (1, 2, 3)")
        b = doc.bindings[0]
        assert isinstance(b.value, TupleLiteral)
        assert len(b.value.elements) == 3

    def test_empty_tuple(self):
        doc = parse("x is ()")
        b = doc.bindings[0]
        assert isinstance(b.value, TupleLiteral)
        assert len(b.value.elements) == 0

    def test_single_tuple(self):
        doc = parse("x is (1,)")
        b = doc.bindings[0]
        assert isinstance(b.value, TupleLiteral)
        assert len(b.value.elements) == 1

    def test_grouping(self):
        doc = parse("x is (1 + 2)")
        b = doc.bindings[0]
        assert isinstance(b.value, Grouping)


# ── Enum / Union / Tagged union (§3.5, §3.6, §3.7) ────────────────

class TestEnumUnionTagged:
    def test_enum(self):
        doc = parse("x is red from red, green, blue")
        b = doc.bindings[0]
        assert isinstance(b.value, FromEnum)
        assert b.value.variants == ["red", "green", "blue"]

    def test_enum_with_called(self):
        doc = parse("x is red from red, green, blue called Color")
        b = doc.bindings[0]
        assert isinstance(b.value, FromEnum)
        assert b.called == "Color"

    def test_union(self):
        doc = parse("x is 42 from union Integer, String")
        b = doc.bindings[0]
        assert isinstance(b.value, FromUnion)
        assert len(b.value.types) == 2

    def test_tagged_union(self):
        doc = parse("x is 7 named ln from n as Integer, ln as String")
        b = doc.bindings[0]
        assert isinstance(b.value, NamedVariant)
        assert b.value.tag == "ln"
        assert len(b.value.variants) == 2

    def test_tagged_union_no_variants(self):
        """§3.7: Tag without from clause — type from outer annotation."""
        doc = parse("x is 7 named ok")
        b = doc.bindings[0]
        assert isinstance(b.value, NamedVariant)
        assert b.value.tag == "ok"
        assert len(b.value.variants) == 0


# ── Control flow (§5.9, §5.10) ─────────────────────────────────────

class TestControlFlow:
    def test_if_then_else(self):
        doc = parse("x is if true then 1 else 0")
        b = doc.bindings[0]
        assert isinstance(b.value, IfExpr)
        assert isinstance(b.value.condition, BoolLiteral)

    def test_nested_if(self):
        doc = parse("x is if true then if false then 1 else 2 else 3")
        b = doc.bindings[0]
        assert isinstance(b.value, IfExpr)
        assert isinstance(b.value.then_branch, IfExpr)

    def test_case_when_else(self):
        doc = parse('x is case 5 when 5 then "yes" else "no"')
        b = doc.bindings[0]
        assert isinstance(b.value, CaseExpr)
        assert len(b.value.when_clauses) == 1

    def test_case_multiple_when(self):
        doc = parse('x is case n when 1 then "a" when 2 then "b" else "c"')
        b = doc.bindings[0]
        assert isinstance(b.value, CaseExpr)
        assert len(b.value.when_clauses) == 2

    def test_case_no_when_error(self):
        with pytest.raises(UzonSyntaxError, match="at least one"):
            parse("x is case 5 else 0")


# ── With / Extends (§3.2.1, §3.2.2) ───────────────────────────────

class TestWithExtends:
    def test_with(self):
        doc = parse("x is base with { a is 1 }")
        b = doc.bindings[0]
        assert isinstance(b.value, StructOverride)
        assert isinstance(b.value.overrides, StructLiteral)

    def test_plus(self):
        doc = parse("x is base plus { b is 2 }")
        b = doc.bindings[0]
        assert isinstance(b.value, StructExtension)
        assert isinstance(b.value.extensions, StructLiteral)

    def test_chained_with_error(self):
        with pytest.raises(UzonSyntaxError, match="Cannot chain"):
            parse("x is base with { a is 1 } with { b is 2 }")

    def test_chained_plus_error(self):
        with pytest.raises(UzonSyntaxError, match="Cannot chain"):
            parse("x is base plus { a is 1 } plus { b is 2 }")


# ── Functions (§3.8) ──────────────────────────────────────────────

class TestFunctions:
    def test_function_basic(self):
        doc = parse("f is function n as Integer returns Integer { n + 1 }")
        b = doc.bindings[0]
        assert isinstance(b.value, FunctionExpr)
        assert len(b.value.params) == 1
        assert b.value.params[0].name == "n"
        assert b.value.return_type.name == "Integer"

    def test_function_zero_params(self):
        doc = parse("f is function returns Integer { 42 }")
        b = doc.bindings[0]
        assert isinstance(b.value, FunctionExpr)
        assert len(b.value.params) == 0

    def test_function_multiple_params(self):
        doc = parse("f is function a as Integer, b as Integer returns Integer { a + b }")
        b = doc.bindings[0]
        assert isinstance(b.value, FunctionExpr)
        assert len(b.value.params) == 2

    def test_function_with_body_bindings(self):
        doc = parse("f is function n as Integer returns Integer { x is n + 1\n x * 2 }")
        b = doc.bindings[0]
        assert isinstance(b.value, FunctionExpr)
        assert len(b.value.body_bindings) == 1
        assert b.value.body_bindings[0].name == "x"

    def test_function_default_param(self):
        doc = parse("f is function n as Integer default 0 returns Integer { n + 1 }")
        b = doc.bindings[0]
        p = b.value.params[0]
        assert p.default is not None

    def test_function_required_after_default_error(self):
        with pytest.raises(UzonSyntaxError, match="Required parameter"):
            parse("f is function a as Integer default 0, b as Integer returns Integer { b }")

    def test_function_call(self):
        doc = parse("x is f(1, 2)")
        b = doc.bindings[0]
        assert isinstance(b.value, FunctionCall)
        assert len(b.value.args) == 2

    def test_function_call_no_args(self):
        doc = parse("x is f()")
        b = doc.bindings[0]
        assert isinstance(b.value, FunctionCall)
        assert len(b.value.args) == 0

    def test_function_no_body_expr_error(self):
        with pytest.raises(UzonSyntaxError, match="must end with an expression"):
            parse("f is function returns Integer { x is 1 }")


# ── Are binding (§3.4.1) ──────────────────────────────────────────

class TestAreBinding:
    def test_are_basic(self):
        doc = parse('names are "a", "b", "c"')
        b = doc.bindings[0]
        assert isinstance(b, AreBinding)
        assert len(b.elements) == 3

    def test_are_with_type(self):
        doc = parse("ids are 1, 2, 3 as [Integer]")
        b = doc.bindings[0]
        assert isinstance(b, AreBinding)
        assert b.type_annotation is not None
        assert b.type_annotation.is_list

    def test_are_with_called(self):
        doc = parse("ids are 1, 2, 3 called Numbers")
        b = doc.bindings[0]
        assert isinstance(b, AreBinding)
        assert b.called == "Numbers"


# ── Field extraction (§5.14) ──────────────────────────────────────

class TestFieldExtraction:
    def test_is_of(self):
        doc = parse("port is of config")
        b = doc.bindings[0]
        assert isinstance(b.value, FieldExtraction)
        assert isinstance(b.value.source, Identifier)

    def test_is_of_chained(self):
        doc = parse("x is of a.b")
        b = doc.bindings[0]
        assert isinstance(b.value, FieldExtraction)


# ── Import (§7.1) ─────────────────────────────────────────────────

class TestImport:
    def test_struct_import(self):
        doc = parse('q is struct "./config.uzon"')
        b = doc.bindings[0]
        assert isinstance(b.value, StructImport)
        assert b.value.path == "./config.uzon"


# ── Binding decomposition (§9) ────────────────────────────────────

class TestBindingDecomposition:
    def test_is_not_decomposition(self):
        """§9: 'x is not true' → x = (not true), not x (is not) true."""
        doc = parse("x is not true")
        b = doc.bindings[0]
        assert isinstance(b.value, UnaryOp)
        assert b.value.op == "not"

    def test_is_named_decomposition(self):
        """§9: 'x is named' → x = identifier 'named'."""
        doc = parse("x is named")
        b = doc.bindings[0]
        assert isinstance(b.value, Identifier)
        assert b.value.name == "named"

    def test_is_not_named_decomposition(self):
        """§9: 'x is not named' → x = (not <identifier 'named'>)."""
        doc = parse("x is not named")
        b = doc.bindings[0]
        assert isinstance(b.value, UnaryOp)
        assert b.value.op == "not"


# ── String interpolation (§4.4.1) ─────────────────────────────────

class TestStringInterpolation:
    def test_interpolation_parts(self):
        doc = parse('name is "world"\nx is "hello {name}"')
        b = doc.bindings[1]
        assert isinstance(b.value, StringLiteral)
        assert len(b.value.parts) == 2
        assert b.value.parts[0] == "hello "
        assert isinstance(b.value.parts[1], Identifier)

    def test_multiple_interpolations(self):
        doc = parse('a is 1\nb is 2\nx is "{a} and {b}"')
        b = doc.bindings[2]
        assert isinstance(b.value, StringLiteral)
        assert len(b.value.parts) >= 3


# ── Multiline strings (§4.4.2) ────────────────────────────────────

class TestMultilineStrings:
    def test_multiline_string(self):
        doc = parse('x is "hello"\n"world"')
        b = doc.bindings[0]
        assert isinstance(b.value, StringLiteral)
        assert b.value.parts == ["hello", "\n", "world"]

    def test_multiline_disabled_in_function(self):
        """§4.4.2: Multiline strings disabled inside function bodies."""
        doc = parse('f is function returns String { "hello" }')
        b = doc.bindings[0]
        assert isinstance(b.value, FunctionExpr)


# ── NEWLINE_SEP (§8) ─────────────────────────────────────────────

class TestNewlineSep:
    def test_expression_continuation(self):
        """§8: Operator on next line continues expression."""
        doc = parse("x is 1\n+ 2")
        b = doc.bindings[0]
        assert isinstance(b.value, BinaryOp)
        assert b.value.op == "+"

    def test_newline_as_separator(self):
        """§8: identifier + is on next line starts new binding."""
        doc = parse("x is 1\ny is 2")
        assert len(doc.bindings) == 2

    def test_struct_newlines(self):
        """§8: Newlines inside struct separate fields."""
        doc = parse("s is {\n  a is 1\n  b is 2\n}")
        assert len(doc.bindings[0].value.fields) == 2


# ── References (§5.12, §5.13) ────────────────────────────────────

class TestReferences:
    def test_env(self):
        doc = parse("x is env.HOME")
        b = doc.bindings[0]
        assert isinstance(b.value, MemberAccess)
        assert isinstance(b.value.object, EnvRef)

    def test_bare_identifier(self):
        doc = parse("a is 1\nx is a")
        b = doc.bindings[1]
        assert isinstance(b.value, Identifier)


# ── Error cases ──────────────────────────────────────────────────

class TestParserErrors:
    def test_missing_is(self):
        with pytest.raises(UzonSyntaxError):
            parse("x 42")

    def test_unexpected_token(self):
        with pytest.raises(UzonSyntaxError):
            parse("x is")

    def test_missing_rbrace(self):
        with pytest.raises(UzonSyntaxError):
            parse("x is { a is 1")

    def test_missing_rbracket(self):
        with pytest.raises(UzonSyntaxError):
            parse("x is [1, 2")

    def test_missing_rparen(self):
        with pytest.raises(UzonSyntaxError):
            parse("x is (1 + 2")

    def test_case_no_when(self):
        with pytest.raises(UzonSyntaxError, match="at least one"):
            parse("x is case 5 else 0")


# ── Source positions ─────────────────────────────────────────────

class TestPositions:
    def test_binding_position(self):
        doc = parse("x is 42")
        b = doc.bindings[0]
        assert b.line == 1
        assert b.col == 1

    def test_second_line_position(self):
        doc = parse("x is 1\ny is 2")
        b = doc.bindings[1]
        assert b.line == 2
