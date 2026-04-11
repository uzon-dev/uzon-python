# SPDX-FileCopyrightText: © 2026 Suho Kang
# SPDX-License-Identifier: MIT
"""UZON token types and Token class per §2.5, §2.6, §9.

TokenType covers all keywords (§2.5), operators and punctuation (§2.6),
composite operators (§9 lexer rules), literals (§4), and structural tokens.
"""

from __future__ import annotations

from enum import Enum, auto


class TokenType(Enum):
    # ── Literals (§4) ──────────────────────────────────────────────
    INTEGER = auto()    # §4.2: decimal, hex, octal, binary
    FLOAT = auto()      # §4.3: IEEE 754 float
    STRING = auto()     # §4.4: double-quoted with escapes
    TRUE = auto()       # §4.1: boolean true
    FALSE = auto()      # §4.1: boolean false
    NULL = auto()       # §4.5: absence of value
    INF = auto()        # §4.3: positive infinity
    NAN = auto()        # §4.3: not a number
    UNDEFINED = auto()  # §3.1: state, not a value

    # ── Identifier (§2.3) ─────────────────────────────────────────
    IDENTIFIER = auto()

    # ── Keywords — binding (§2.5) ─────────────────────────────────
    IS = auto()         # §5.1: assignment / equality
    ARE = auto()        # §3.4.1: list binding sugar

    # ── Keywords — type system (§2.5) ─────────────────────────────
    FROM = auto()       # §3.5/§3.6/§3.7: enum/union/tagged union source
    CALLED = auto()     # §6.2: type naming
    AS = auto()         # §6.1: type annotation
    NAMED = auto()      # §3.7: tagged union variant
    WITH = auto()       # §3.2.1: struct override
    EXTENDS = auto()    # §3.2.2: struct extension
    UNION = auto()      # §3.6: union type marker

    # ── Keywords — functions (§3.8) ───────────────────────────────
    FUNCTION = auto()   # §3.8: function definition
    RETURNS = auto()    # §3.8: return type declaration
    DEFAULT = auto()    # §3.8: default parameter value

    # ── Keywords — conversion / extraction (§2.5) ─────────────────
    TO = auto()         # §5.11: type conversion
    OF = auto()         # §5.14: field extraction

    # ── Keywords — logic (§2.5) ───────────────────────────────────
    AND = auto()        # §5.6: logical AND
    OR = auto()         # §5.6: logical OR
    NOT = auto()        # §5.6: logical NOT

    # ── Keywords — control (§2.5) ─────────────────────────────────
    IF = auto()         # §5.9: conditional
    THEN = auto()       # §5.9: conditional branch
    ELSE = auto()       # §5.9: conditional fallback
    CASE = auto()       # §5.10: multi-branch conditional
    WHEN = auto()       # §5.10: case clause

    # ── Keywords — references (§2.5) ──────────────────────────────
    SELF = auto()       # §5.12: reserved keyword
    ENV = auto()        # §5.13: environment variable access

    # ── Keywords — import (§2.5) ──────────────────────────────────
    STRUCT = auto()     # §7.1: file import

    # ── Keywords — membership (§2.5) ──────────────────────────────
    IN = auto()         # §5.8.1: membership test

    # ── Composite operators — single tokens (§9 lexer rules) ─────
    OR_ELSE = auto()       # §5.7: undefined coalescing
    IS_NOT = auto()        # §5.2: inequality
    IS_NAMED = auto()      # §3.7.2: variant check
    IS_NOT_NAMED = auto()  # §3.7.2: negated variant check

    # ── Arithmetic operators (§2.6, §5.3) ─────────────────────────
    PLUS = auto()          # +
    MINUS = auto()         # -
    STAR = auto()          # *
    SLASH = auto()         # /
    PERCENT = auto()       # %
    CARET = auto()         # ^  exponentiation
    PLUS_PLUS = auto()     # ++ concatenation
    STAR_STAR = auto()     # ** repetition

    # ── Comparison operators (§2.6, §5.4) ─────────────────────────
    LT = auto()            # <
    LE = auto()            # <=
    GT = auto()            # >
    GE = auto()            # >=

    # ── Punctuation (§2.6) ────────────────────────────────────────
    COMMA = auto()         # ,
    DOT = auto()           # .
    AT = auto()            # @ keyword escape (§2.4)

    # ── Delimiters (§2.6) ─────────────────────────────────────────
    LBRACE = auto()        # {
    RBRACE = auto()        # }
    LBRACKET = auto()      # [
    RBRACKET = auto()      # ]
    LPAREN = auto()        # (
    RPAREN = auto()        # )

    # ── String interpolation (§4.4.1) ─────────────────────────────
    INTERP_START = auto()  # { inside string
    INTERP_END = auto()    # } closing interpolation

    # ── Structural ────────────────────────────────────────────────
    NEWLINE = auto()       # line boundary for NEWLINE_SEP (§8)
    EOF = auto()


# ── Keyword → token type mapping (§2.5) ───────────────────────────
KEYWORDS: dict[str, TokenType] = {
    "is": TokenType.IS,
    "are": TokenType.ARE,
    "from": TokenType.FROM,
    "called": TokenType.CALLED,
    "as": TokenType.AS,
    "named": TokenType.NAMED,
    "with": TokenType.WITH,
    "extends": TokenType.EXTENDS,
    "union": TokenType.UNION,
    "function": TokenType.FUNCTION,
    "returns": TokenType.RETURNS,
    "default": TokenType.DEFAULT,
    "to": TokenType.TO,
    "of": TokenType.OF,
    "and": TokenType.AND,
    "or": TokenType.OR,
    "not": TokenType.NOT,
    "if": TokenType.IF,
    "then": TokenType.THEN,
    "else": TokenType.ELSE,
    "case": TokenType.CASE,
    "when": TokenType.WHEN,
    "self": TokenType.SELF,
    "env": TokenType.ENV,
    "struct": TokenType.STRUCT,
    "in": TokenType.IN,
    "true": TokenType.TRUE,
    "false": TokenType.FALSE,
    "null": TokenType.NULL,
    "undefined": TokenType.UNDEFINED,
    "inf": TokenType.INF,
    "nan": TokenType.NAN,
}

# Reserved keywords (§2.5) — recognized but not yet assigned semantics.
RESERVED_KEYWORDS: set[str] = {"lazy", "type"}

ALL_KEYWORDS: set[str] = set(KEYWORDS) | RESERVED_KEYWORDS

# Token boundary characters per §2.3 — always terminate unquoted identifiers.
TOKEN_BOUNDARIES: set[str] = set("{}[](),.\"'@+-*/%^<>=!?:;|&$~#\\")


class Token:
    """A single lexer token carrying type, value, and source position (§11.2.0)."""

    __slots__ = ("type", "value", "line", "col")

    def __init__(self, type: TokenType, value: str, line: int, col: int):
        self.type = type
        self.value = value
        self.line = line
        self.col = col

    def __repr__(self) -> str:
        return f"Token({self.type.name}, {self.value!r}, {self.line}:{self.col})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Token):
            return self.type == other.type and self.value == other.value
        return NotImplemented
