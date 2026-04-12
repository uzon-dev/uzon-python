# SPDX-FileCopyrightText: © 2026 Suho Kang
# SPDX-License-Identifier: MIT
"""Tests for the UZON lexer."""

import pytest

from uzon.errors import UzonSyntaxError
from uzon.lexer import Lexer
from uzon.tokens import TokenType


def lex(source: str) -> list[tuple[TokenType, str]]:
    """Helper: tokenize and return (type, value) pairs, excluding EOF."""
    tokens = Lexer(source).tokenize()
    return [(t.type, t.value) for t in tokens if t.type != TokenType.EOF]


# ── Basic tokens ──────────────────────────────────────────────────────

class TestBasicTokens:
    def test_integer_decimal(self):
        assert lex("42") == [(TokenType.INTEGER, "42")]

    def test_integer_hex(self):
        assert lex("0xff") == [(TokenType.INTEGER, "0xff")]

    def test_integer_octal(self):
        assert lex("0o77") == [(TokenType.INTEGER, "0o77")]

    def test_integer_binary(self):
        assert lex("0b1010") == [(TokenType.INTEGER, "0b1010")]

    def test_integer_with_underscores(self):
        assert lex("1_000_000") == [(TokenType.INTEGER, "1000000")]

    def test_float_basic(self):
        assert lex("3.14") == [(TokenType.FLOAT, "3.14")]

    def test_float_with_exponent(self):
        assert lex("1.0e10") == [(TokenType.FLOAT, "1.0e10")]

    def test_float_negative_exponent(self):
        assert lex("2.5E-3") == [(TokenType.FLOAT, "2.5E-3")]

    def test_float_exponent_only(self):
        assert lex("1e5") == [(TokenType.FLOAT, "1e5")]

    def test_inf(self):
        assert lex("inf") == [(TokenType.INF, "inf")]

    def test_nan(self):
        assert lex("nan") == [(TokenType.NAN, "nan")]

    def test_negative_inf(self):
        assert lex("-inf") == [(TokenType.FLOAT, "-inf")]

    def test_negative_nan(self):
        assert lex("-nan") == [(TokenType.FLOAT, "-nan")]

    def test_string(self):
        assert lex('"hello"') == [(TokenType.STRING, "hello")]

    def test_string_escapes(self):
        assert lex(r'"tab:\there"') == [(TokenType.STRING, "tab:\there")]

    def test_string_all_escapes(self):
        result = lex(r'"\\\"\n\r\t\0\{"')
        assert result == [(TokenType.STRING, '\\\"\n\r\t\0{')]

    def test_string_hex_escape(self):
        assert lex(r'"\x41"') == [(TokenType.STRING, "A")]

    def test_string_unicode_escape(self):
        assert lex(r'"\u{1F600}"') == [(TokenType.STRING, "\U0001F600")]

    def test_boolean_true(self):
        assert lex("true") == [(TokenType.TRUE, "true")]

    def test_boolean_false(self):
        assert lex("false") == [(TokenType.FALSE, "false")]

    def test_null(self):
        assert lex("null") == [(TokenType.NULL, "null")]

    def test_undefined(self):
        assert lex("undefined") == [(TokenType.UNDEFINED, "undefined")]


# ── Keywords and composite operators ──────────────────────────────────

class TestKeywordsAndComposite:
    def test_is_keyword(self):
        result = lex("x is 5")
        assert result[1] == (TokenType.IS, "is")

    def test_is_not_composite(self):
        types = [t for t, _ in lex("x is not 0")]
        assert TokenType.IS_NOT in types

    def test_is_named_composite(self):
        types = [t for t, _ in lex("x is named ok")]
        assert TokenType.IS_NAMED in types

    def test_is_not_named_composite(self):
        types = [t for t, _ in lex("x is not named ok")]
        assert TokenType.IS_NOT_NAMED in types

    def test_or_else_composite(self):
        types = [t for t, _ in lex("x or else 5")]
        assert TokenType.OR_ELSE in types

    def test_or_alone(self):
        types = [t for t, _ in lex("true or false")]
        assert TokenType.OR in types
        assert TokenType.OR_ELSE not in types

    def test_all_keywords(self):
        """All non-composite keywords tokenize correctly."""
        for kw in ("from", "called", "as", "named", "with", "plus",
                    "union", "function", "returns", "default", "to", "of",
                    "and", "not", "if", "then", "else", "case", "when",
                    "type", "env", "struct", "in", "are"):
            result = lex(kw)
            assert len(result) == 1 and result[0][1] == kw


# ── Identifiers ───────────────────────────────────────────────────────

class TestIdentifiers:
    def test_simple(self):
        assert lex("hello") == [(TokenType.IDENTIFIER, "hello")]

    def test_unicode(self):
        assert lex("안녕") == [(TokenType.IDENTIFIER, "안녕")]

    def test_emoji(self):
        assert lex("😀😀") == [(TokenType.IDENTIFIER, "😀😀")]

    def test_digit_start(self):
        """§2.3: '1st' does not match number grammar → identifier."""
        assert lex("1st") == [(TokenType.IDENTIFIER, "1st")]

    def test_quoted(self):
        assert lex("'Content-Type'") == [(TokenType.IDENTIFIER, "Content-Type")]

    def test_quoted_with_spaces(self):
        assert lex("'this is a key'") == [(TokenType.IDENTIFIER, "this is a key")]

    def test_quoted_empty(self):
        assert lex("''") == [(TokenType.IDENTIFIER, "")]

    def test_quoted_keyword_stays_keyword(self):
        assert lex("'is'") == [(TokenType.IS, "is")]

    def test_keyword_escape(self):
        assert lex("@is") == [(TokenType.IDENTIFIER, "is")]

    def test_keyword_escape_true(self):
        assert lex("@true") == [(TokenType.IDENTIFIER, "true")]


# ── Operators and delimiters ──────────────────────────────────────────

class TestOperators:
    def test_arithmetic(self):
        types = [t for t, _ in lex("+ - * / % ^")]
        assert types == [TokenType.PLUS, TokenType.MINUS, TokenType.STAR,
                         TokenType.SLASH, TokenType.PERCENT, TokenType.CARET]

    def test_concat_and_repeat(self):
        types = [t for t, _ in lex("++ **")]
        assert types == [TokenType.PLUS_PLUS, TokenType.STAR_STAR]

    def test_comparison(self):
        types = [t for t, _ in lex("< <= > >=")]
        assert types == [TokenType.LT, TokenType.LE, TokenType.GT, TokenType.GE]

    def test_delimiters(self):
        types = [t for t, _ in lex("{ } [ ] ( )")]
        assert types == [TokenType.LBRACE, TokenType.RBRACE, TokenType.LBRACKET,
                         TokenType.RBRACKET, TokenType.LPAREN, TokenType.RPAREN]

    def test_comma_dot(self):
        types = [t for t, _ in lex(", .")]
        assert types == [TokenType.COMMA, TokenType.DOT]


# ── Context-sensitive minus ───────────────────────────────────────────

class TestMinus:
    def test_negative_literal(self):
        assert lex("-5") == [(TokenType.INTEGER, "-5")]

    def test_binary_subtraction(self):
        assert lex("3 - 5")[1] == (TokenType.MINUS, "-")

    def test_subtraction_no_spaces(self):
        assert lex("3-5") == [
            (TokenType.INTEGER, "3"), (TokenType.MINUS, "-"), (TokenType.INTEGER, "5")
        ]

    def test_unary_minus_after_operator(self):
        """Minus after operator → unary minus."""
        result = lex("x + -3")
        assert (TokenType.INTEGER, "-3") in result

    def test_minus_after_rparen_is_binary(self):
        result = lex("(1) - 2")
        assert (TokenType.MINUS, "-") in result


# ── String interpolation ─────────────────────────────────────────────

class TestStringInterpolation:
    def test_basic(self):
        types = [t for t, _ in lex('"hello {name}"')]
        assert TokenType.INTERP_START in types
        assert TokenType.INTERP_END in types

    def test_struct_in_interpolation(self):
        """§4.4.1: Nested braces inside interpolation."""
        types = [t for t, _ in lex('"value: {x}"')]
        assert TokenType.INTERP_START in types

    def test_escaped_brace(self):
        """§4.4: \\{ suppresses interpolation."""
        result = lex(r'"\{not interpolation}"')
        assert result == [(TokenType.STRING, "{not interpolation}")]

    def test_escaped_quote_in_interpolation(self):
        r"""§4.4.1: Escaped-quote strings \"...\" inside interpolation."""
        result = lex(r'"join: {std.join(list, \", \")}"')
        types = [t for t, _ in result]
        assert TokenType.INTERP_START in types
        assert TokenType.STRING in types


# ── Error cases ───────────────────────────────────────────────────────

class TestLexerErrors:
    def test_unterminated_string(self):
        with pytest.raises(UzonSyntaxError, match="Unterminated"):
            Lexer('"hello').tokenize()

    def test_invalid_escape(self):
        with pytest.raises(UzonSyntaxError, match="Invalid escape"):
            Lexer('"\\q"').tokenize()

    def test_hex_escape_above_ascii(self):
        with pytest.raises(UzonSyntaxError, match="above ASCII"):
            Lexer('"\\x80"').tokenize()

    def test_unicode_surrogate(self):
        with pytest.raises(UzonSyntaxError, match="surrogate"):
            Lexer('"\\u{D800}"').tokenize()

    def test_unicode_too_large(self):
        with pytest.raises(UzonSyntaxError, match="exceeds"):
            Lexer('"\\u{110000}"').tokenize()

    def test_consecutive_underscores(self):
        with pytest.raises(UzonSyntaxError, match="Consecutive underscores"):
            Lexer("1__000").tokenize()

    def test_unmatched_quoted_identifier(self):
        with pytest.raises(UzonSyntaxError, match="Unmatched"):
            Lexer("'hello").tokenize()

    def test_reserved_keyword(self):
        with pytest.raises(UzonSyntaxError, match="Reserved keyword"):
            Lexer("lazy").tokenize()

    def test_control_char_in_string(self):
        with pytest.raises(UzonSyntaxError, match="Control character"):
            Lexer('"hello\x01world"').tokenize()

    def test_at_non_keyword(self):
        with pytest.raises(UzonSyntaxError, match="not a keyword"):
            Lexer("@hello").tokenize()


# ── BOM handling ──────────────────────────────────────────────────────

class TestBOM:
    def test_bom_stripped(self):
        assert lex("\ufeff42") == [(TokenType.INTEGER, "42")]


# ── Comments ──────────────────────────────────────────────────────────

class TestComments:
    def test_line_comment(self):
        result = lex("42 // comment")
        assert (TokenType.INTEGER, "42") in result

    def test_comment_preserves_newline(self):
        result = lex("42 // comment\n43")
        vals = [v for _, v in result if v not in ("\n", "")]
        assert "42" in vals and "43" in vals


# ── Newlines ──────────────────────────────────────────────────────────

class TestNewlines:
    def test_newline_token(self):
        types = [t for t, _ in lex("a\nb")]
        assert TokenType.NEWLINE in types

    def test_crlf(self):
        types = [t for t, _ in lex("a\r\nb")]
        assert TokenType.NEWLINE in types


# ── Complete binding ──────────────────────────────────────────────────

class TestFullBindings:
    def test_simple_binding(self):
        result = lex("x is 42")
        assert result == [
            (TokenType.IDENTIFIER, "x"),
            (TokenType.IS, "is"),
            (TokenType.INTEGER, "42"),
        ]

    def test_struct_binding(self):
        result = lex('config is { port is 8080 }')
        types = [t for t, _ in result]
        assert TokenType.LBRACE in types and TokenType.RBRACE in types
