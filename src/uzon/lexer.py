# SPDX-FileCopyrightText: © 2026 Suho Kang
# SPDX-License-Identifier: MIT
"""UZON lexer — tokenizes source text into a stream of tokens.

Implements the lexical structure defined in §2 and §9 (Lexer Rules):
  - UTF-8 BOM handling (§2.1)
  - Line comments (§2.2)
  - Identifiers: bare, quoted, keyword-escaped (§2.3, §2.4)
  - Keywords (§2.5) and reserved words
  - Operators and punctuation (§2.6)
  - String literals with escapes and interpolation (§4.4, §4.4.1)
  - Integer/float literals in all bases (§4.2, §4.3)
  - Composite operators: or else, is not, is named, is not named (§9)
  - Context-sensitive minus: unary vs binary (§9 lexer note)
"""

from __future__ import annotations

from .errors import UzonSyntaxError
from .tokens import (
    ALL_KEYWORDS,
    KEYWORDS,
    RESERVED_KEYWORDS,
    TOKEN_BOUNDARIES,
    Token,
    TokenType,
)

# Token types that make a following minus binary subtraction (§9 lexer note).
_VALUE_TOKEN_TYPES: frozenset[TokenType] = frozenset({
    TokenType.INTEGER, TokenType.FLOAT, TokenType.STRING,
    TokenType.IDENTIFIER,
    TokenType.TRUE, TokenType.FALSE, TokenType.NULL,
    TokenType.INF, TokenType.NAN, TokenType.UNDEFINED,
    TokenType.ENV,
    TokenType.RPAREN, TokenType.RBRACKET, TokenType.RBRACE,
})


class Lexer:
    """Tokenizes UZON source text per §2 and §9."""

    def __init__(self, source: str, filename: str = "<string>"):
        self._src = source
        self._file = filename
        self._pos = 0
        self._line = 1
        self._col = 1
        self._tokens: list[Token] = []

    # ── public API ────────────────────────────────────────────────────

    def tokenize(self) -> list[Token]:
        """Tokenize the entire source and return the token list."""
        # Strip UTF-8 BOM if present (§2.1)
        if self._src.startswith("\ufeff"):
            self._src = self._src[1:]

        while self._pos < len(self._src):
            self._skip_ws()
            if self._pos >= len(self._src):
                break

            ch = self._src[self._pos]

            if ch == "\n" or (ch == "\r" and self._peek_at(1) == "\n"):
                self._read_newline()
            elif ch == '"':
                self._read_string()
            elif ch == "'":
                self._read_quoted_ident()
            elif ch == "@":
                self._read_keyword_escape()
            elif ch in TOKEN_BOUNDARIES:
                self._read_op_or_delim()
            else:
                self._read_word()

        self._tokens.append(Token(TokenType.EOF, "", self._line, self._col))
        return self._tokens

    # ── character helpers ─────────────────────────────────────────────

    def _peek(self) -> str:
        return self._src[self._pos] if self._pos < len(self._src) else ""

    def _peek_at(self, offset: int) -> str:
        p = self._pos + offset
        return self._src[p] if p < len(self._src) else ""

    def _advance(self) -> str:
        ch = self._src[self._pos]
        self._pos += 1
        if ch == "\n":
            self._line += 1
            self._col = 1
        else:
            self._col += 1
        return ch

    def _error(self, msg: str) -> UzonSyntaxError:
        return UzonSyntaxError(msg, self._line, self._col, file=self._file)

    def _last_type(self) -> TokenType | None:
        return self._tokens[-1].type if self._tokens else None

    # ── whitespace / comments (§2.2) ──────────────────────────────────

    def _skip_ws(self) -> None:
        while self._pos < len(self._src):
            ch = self._src[self._pos]
            if ch in (" ", "\t"):
                self._advance()
            elif ch == "/" and self._peek_at(1) == "/":
                self._skip_comment()
            elif ch == "\r" and self._peek_at(1) != "\n":
                self._advance()  # bare CR as whitespace
            else:
                break

    def _skip_ws_and_newlines(self) -> None:
        """Skip whitespace, comments, and newlines.

        Used by composite keyword lookahead (§9): ``or else``, ``is not``,
        etc. must be recognized even when split across lines.
        """
        while self._pos < len(self._src):
            ch = self._src[self._pos]
            if ch in (" ", "\t"):
                self._advance()
            elif ch == "\n":
                self._advance()
            elif ch == "\r":
                self._advance()
            elif ch == "/" and self._peek_at(1) == "/":
                self._skip_comment()
            else:
                break

    def _skip_comment(self) -> None:
        """§2.2: Line comment — // to end of line."""
        self._advance()  # /
        self._advance()  # /
        while self._pos < len(self._src) and self._src[self._pos] != "\n":
            self._advance()

    # ── newline (§8) ──────────────────────────────────────────────────

    def _read_newline(self) -> None:
        line, col = self._line, self._col
        if self._src[self._pos] == "\r":
            self._advance()
        self._advance()  # \n
        self._tokens.append(Token(TokenType.NEWLINE, "\n", line, col))

    # ── strings (§4.4) ────────────────────────────────────────────────

    def _read_string(self) -> None:
        """§4.4: Read a double-quoted string, handling escapes and interpolation."""
        line, col = self._line, self._col
        self._advance()  # opening "
        parts: list[str] = []
        buf: list[str] = []

        while self._pos < len(self._src):
            ch = self._src[self._pos]

            if ch == '"':
                self._advance()
                if parts:
                    if buf:
                        self._tokens.append(Token(TokenType.STRING, "".join(buf), line, col))
                else:
                    self._tokens.append(Token(TokenType.STRING, "".join(buf), line, col))
                return

            if ch == "\\":
                buf.append(self._read_escape())
            elif ch == "{":
                # §4.4.1: string interpolation
                if buf or not parts:
                    self._tokens.append(Token(TokenType.STRING, "".join(buf), line, col))
                    parts.append("".join(buf))
                    buf = []
                self._advance()  # {
                self._tokens.append(Token(TokenType.INTERP_START, "{", self._line, self._col))
                self._read_interp_expr()
                self._tokens.append(Token(TokenType.INTERP_END, "}", self._line, self._col))
                self._advance()  # }
                parts.append(None)  # type: ignore[arg-type]
                line, col = self._line, self._col
            elif ch in ("\n", "\r"):
                raise self._error("Unterminated string literal")
            elif ord(ch) < 0x20:
                raise self._error(
                    f"Control character U+{ord(ch):04X} not allowed in string"
                    " — use an escape sequence"
                )
            else:
                buf.append(self._advance())

        raise self._error("Unterminated string literal")

    def _read_interp_expr(self) -> None:
        """§4.4.1: Lex tokens inside string interpolation until matching }."""
        depth = 1
        while self._pos < len(self._src) and depth > 0:
            self._skip_ws()
            if self._pos >= len(self._src):
                break

            ch = self._src[self._pos]

            if ch == "}":
                depth -= 1
                if depth == 0:
                    return
                self._tokens.append(Token(TokenType.RBRACE, "}", self._line, self._col))
                self._advance()
            elif ch == "{":
                depth += 1
                self._tokens.append(Token(TokenType.LBRACE, "{", self._line, self._col))
                self._advance()
            elif ch in ("\n", "\r") and (ch == "\n" or self._peek_at(1) == "\n"):
                self._read_newline()
            elif ch == "\\" and self._peek_at(1) == '"':
                # §4.4.1: Escaped-quote string delimiters inside interpolation
                self._advance()  # consume backslash
                self._read_interp_string()
            elif ch == '"':
                self._read_string()
            elif ch == "'":
                self._read_quoted_ident()
            elif ch == "@":
                self._read_keyword_escape()
            elif ch in TOKEN_BOUNDARIES:
                self._read_op_or_delim()
            else:
                self._read_word()

        if depth > 0:
            raise self._error("Unterminated string interpolation")

    def _read_interp_string(self) -> None:
        r"""§4.4.1: Read a \"...\" delimited string inside interpolation."""
        line, col = self._line, self._col
        self._advance()  # opening "
        buf: list[str] = []
        while self._pos < len(self._src):
            ch = self._src[self._pos]
            if ch == "\\" and self._peek_at(1) == '"':
                self._advance()  # backslash
                self._advance()  # "
                self._tokens.append(Token(TokenType.STRING, "".join(buf), line, col))
                return
            if ch in ("\n", "\r"):
                raise self._error("Unterminated string literal in interpolation")
            buf.append(self._advance())
        raise self._error("Unterminated string literal in interpolation")

    # ── escape sequences (§4.4) ───────────────────────────────────────

    def _read_escape(self) -> str:
        self._advance()  # backslash
        if self._pos >= len(self._src):
            raise self._error("Unexpected end of string after backslash")
        ch = self._advance()
        match ch:
            case "\\": return "\\"
            case '"':  return '"'
            case "n":  return "\n"
            case "r":  return "\r"
            case "t":  return "\t"
            case "0":  return "\0"
            case "{":  return "{"
            case "x":  return self._read_hex_escape()
            case "u":  return self._read_unicode_escape()
            case _:    raise self._error(f"Invalid escape sequence: \\{ch}")

    def _read_hex_escape(self) -> str:
        """§4.4: \\xHH — byte value, 0x00-0x7F only."""
        digits = ""
        for _ in range(2):
            if self._pos >= len(self._src):
                raise self._error("Incomplete \\x escape sequence")
            digits += self._advance()
        try:
            value = int(digits, 16)
        except ValueError:
            raise self._error(f"Invalid hex escape: \\x{digits}")
        if value > 0x7F:
            raise self._error(
                f"\\x{digits} is above ASCII range (0x00-0x7F); use \\u{{...}} for Unicode"
            )
        return chr(value)

    def _read_unicode_escape(self) -> str:
        """§4.4: \\u{HHHHHH} — 1-6 hex digits, valid Unicode scalar value."""
        if self._pos >= len(self._src) or self._src[self._pos] != "{":
            raise self._error("Expected '{' after \\u")
        self._advance()  # {
        digits = ""
        while self._pos < len(self._src) and self._src[self._pos] != "}":
            digits += self._advance()
        if self._pos >= len(self._src):
            raise self._error("Unterminated \\u{...} escape")
        self._advance()  # }
        if not digits or len(digits) > 6:
            raise self._error(f"\\u{{}} requires 1-6 hex digits, got {len(digits)}")
        try:
            value = int(digits, 16)
        except ValueError:
            raise self._error(f"Invalid hex digits in \\u{{{digits}}}")
        if value > 0x10FFFF:
            raise self._error(f"Unicode value U+{value:X} exceeds U+10FFFF")
        if 0xD800 <= value <= 0xDFFF:
            raise self._error(f"Unicode surrogate U+{value:X} is not allowed")
        return chr(value)

    # ── quoted identifier (§2.3) ──────────────────────────────────────

    def _read_quoted_ident(self) -> None:
        """§2.3: 'Content-Type' — quoted identifier, same name as unquoted."""
        line, col = self._line, self._col
        self._advance()  # opening '
        chars: list[str] = []
        while self._pos < len(self._src):
            ch = self._src[self._pos]
            if ch == "'":
                self._advance()
                name = "".join(chars)
                if name in KEYWORDS:
                    self._tokens.append(Token(KEYWORDS[name], name, line, col))
                elif name in RESERVED_KEYWORDS:
                    raise UzonSyntaxError(
                        f"Reserved keyword '{name}' cannot be used as identifier",
                        line, col,
                    )
                else:
                    self._tokens.append(Token(TokenType.IDENTIFIER, name, line, col))
                return
            if ch in ("\n", "\r"):
                raise self._error("Unmatched ' — quoted identifier must close on the same line")
            chars.append(self._advance())
        raise self._error("Unmatched ' — quoted identifier not closed")

    # ── keyword escape (§2.4) ─────────────────────────────────────────

    def _read_keyword_escape(self) -> None:
        """§2.4: @keyword — escape to use reserved words as identifiers."""
        line, col = self._line, self._col
        self._advance()  # @
        word = self._read_raw_word()
        if not word:
            raise self._error("Expected keyword after @")
        if word not in ALL_KEYWORDS:
            raise self._error(f"@ can only escape keywords, '{word}' is not a keyword")
        self._tokens.append(Token(TokenType.IDENTIFIER, word, line, col))

    # ── operators and delimiters (§2.6) ───────────────────────────────

    def _read_op_or_delim(self) -> None:
        line, col = self._line, self._col
        ch = self._src[self._pos]

        if ch == "{":
            self._advance(); self._tokens.append(Token(TokenType.LBRACE, "{", line, col))
        elif ch == "}":
            self._advance(); self._tokens.append(Token(TokenType.RBRACE, "}", line, col))
        elif ch == "[":
            self._advance(); self._tokens.append(Token(TokenType.LBRACKET, "[", line, col))
        elif ch == "]":
            self._advance(); self._tokens.append(Token(TokenType.RBRACKET, "]", line, col))
        elif ch == "(":
            self._advance(); self._tokens.append(Token(TokenType.LPAREN, "(", line, col))
        elif ch == ")":
            self._advance(); self._tokens.append(Token(TokenType.RPAREN, ")", line, col))
        elif ch == ",":
            self._advance(); self._tokens.append(Token(TokenType.COMMA, ",", line, col))
        elif ch == ".":
            self._advance(); self._tokens.append(Token(TokenType.DOT, ".", line, col))
        elif ch == "+":
            self._advance()
            if self._peek() == "+":
                self._advance()
                self._tokens.append(Token(TokenType.PLUS_PLUS, "++", line, col))
            else:
                self._tokens.append(Token(TokenType.PLUS, "+", line, col))
        elif ch == "-":
            self._read_minus(line, col)
        elif ch == "*":
            self._advance()
            if self._peek() == "*":
                self._advance()
                self._tokens.append(Token(TokenType.STAR_STAR, "**", line, col))
            else:
                self._tokens.append(Token(TokenType.STAR, "*", line, col))
        elif ch == "/":
            self._advance()
            self._tokens.append(Token(TokenType.SLASH, "/", line, col))
        elif ch == "%":
            self._advance()
            self._tokens.append(Token(TokenType.PERCENT, "%", line, col))
        elif ch == "^":
            self._advance()
            self._tokens.append(Token(TokenType.CARET, "^", line, col))
        elif ch == "<":
            self._advance()
            if self._peek() == "=":
                self._advance()
                self._tokens.append(Token(TokenType.LE, "<=", line, col))
            else:
                self._tokens.append(Token(TokenType.LT, "<", line, col))
        elif ch == ">":
            self._advance()
            if self._peek() == "=":
                self._advance()
                self._tokens.append(Token(TokenType.GE, ">=", line, col))
            else:
                self._tokens.append(Token(TokenType.GT, ">", line, col))
        else:
            raise self._error(f"Unexpected character: {ch!r}")

    def _read_minus(self, line: int, col: int) -> None:
        """§9 lexer note: Context-sensitive minus.

        After a value token → binary subtraction.
        Otherwise, minus + digits/inf/nan → negative literal.
        Fallback → unary minus operator.
        """
        self._advance()  # -
        last = self._last_type()

        if last in _VALUE_TOKEN_TYPES:
            self._tokens.append(Token(TokenType.MINUS, "-", line, col))
            return

        # Negative numeric literal
        if self._pos < len(self._src) and self._src[self._pos].isdigit():
            self._read_number(line, col, negative=True)
            return

        # Negative inf/nan (§4.3)
        if self._pos < len(self._src):
            word = self._peek_word()
            if word in ("inf", "nan"):
                self._consume_word(word)
                self._tokens.append(Token(TokenType.FLOAT, f"-{word}", line, col))
                return

        self._tokens.append(Token(TokenType.MINUS, "-", line, col))

    # ── numbers (§4.2, §4.3) ──────────────────────────────────────────

    def _read_number(self, line: int, col: int, negative: bool = False) -> None:
        """Parse integer or float literal per §4.2 and §4.3."""
        prefix = "-" if negative else ""
        start = self._pos

        # Base prefixes (§4.2)
        if self._src[self._pos] == "0" and self._pos + 1 < len(self._src):
            nx = self._src[self._pos + 1].lower()
            third = self._src[self._pos + 2] if self._pos + 2 < len(self._src) else ""
            if nx == "x" and third in "0123456789abcdefABCDEF":
                self._read_based_int(line, col, prefix, "x", "0123456789abcdefABCDEF")
                return
            if nx == "o" and third in "01234567":
                self._read_based_int(line, col, prefix, "o", "01234567")
                return
            if nx == "b" and third in "01":
                self._read_based_int(line, col, prefix, "b", "01")
                return

        # Decimal integer or float
        digits = self._read_digits()
        is_float = False

        # Decimal point (§4.3)
        if (self._pos < len(self._src) and self._src[self._pos] == "."
                and self._pos + 1 < len(self._src) and self._src[self._pos + 1].isdigit()):
            is_float = True
            digits += self._advance()  # .
            digits += self._read_digits()

        # Exponent (§4.3)
        if self._pos < len(self._src) and self._src[self._pos] in ("e", "E"):
            is_float = True
            digits += self._advance()
            if self._pos < len(self._src) and self._src[self._pos] in ("+", "-"):
                digits += self._advance()
            exp = self._read_digits()
            if not exp:
                raise self._error("Exponent requires at least one digit")
            digits += exp

        value = prefix + digits

        # §2.3: If followed by non-boundary → identifier, not number
        if self._pos < len(self._src):
            ch = self._src[self._pos]
            if ch not in (" ", "\t", "\n", "\r") and ch not in TOKEN_BOUNDARIES and ch != "/":
                if negative:
                    raise self._error(f"Invalid numeric literal: {value}{ch}")
                self._pos = start
                self._col = col
                self._line = line
                word = self._read_raw_word()
                self._tokens.append(Token(TokenType.IDENTIFIER, word, line, col))
                return

        tok = TokenType.FLOAT if is_float else TokenType.INTEGER
        self._tokens.append(Token(tok, value, line, col))

    def _read_based_int(
        self, line: int, col: int, prefix: str, base_ch: str, valid: str
    ) -> None:
        """§4.2: Read hex/octal/binary integer with underscore separators."""
        self._advance()  # 0
        self._advance()  # x/o/b
        digits = ""
        last_us = False
        while self._pos < len(self._src):
            ch = self._src[self._pos]
            if ch in valid:
                digits += self._advance()
                last_us = False
            elif ch == "_":
                if last_us:
                    raise self._error("Consecutive underscores in numeric literal")
                self._advance()
                last_us = True
            else:
                break
        if last_us and digits:
            raise self._error("Trailing underscore in numeric literal")
        if not digits:
            value = prefix + "0" + base_ch
            self._pos -= len(value) - len(prefix)
            self._col = col + (1 if prefix else 0)
            if prefix:
                raise self._error(f"Invalid numeric literal: {value}")
            self._read_word()
            return
        self._tokens.append(Token(TokenType.INTEGER, prefix + "0" + base_ch + digits, line, col))

    def _read_digits(self) -> str:
        """Read a sequence of decimal digits with underscore separators (§4.2)."""
        digits = ""
        last_us = False
        while self._pos < len(self._src):
            ch = self._src[self._pos]
            if ch.isdigit():
                digits += self._advance()
                last_us = False
            elif ch == "_":
                if not digits:
                    break
                if last_us:
                    raise self._error("Consecutive underscores in numeric literal")
                self._advance()
                last_us = True
            else:
                break
        if last_us and digits:
            raise self._error("Trailing underscore in numeric literal")
        return digits

    # ── words (identifiers / keywords) ────────────────────────────────

    def _read_raw_word(self) -> str:
        """Read a contiguous non-whitespace, non-boundary token (§2.3)."""
        word = ""
        while self._pos < len(self._src):
            ch = self._src[self._pos]
            if ch in (" ", "\t", "\n", "\r") or ch in TOKEN_BOUNDARIES:
                break
            if ch == "/" and self._peek_at(1) == "/":
                break
            cp = ord(ch)
            if cp in (0x200E, 0x200F) or 0x202A <= cp <= 0x202E or 0x2066 <= cp <= 0x2069:
                raise self._error(
                    f"RTL/bidi mark U+{cp:04X} is not allowed outside string literals (§2.3)"
                )
            word += self._advance()
        return word

    def _read_word(self) -> None:
        """Read a word and classify as keyword, identifier, or number."""
        line, col = self._line, self._col

        if self._src[self._pos].isdigit():
            self._read_number(line, col, negative=False)
            return

        word = self._read_raw_word()
        if not word:
            raise self._error(f"Unexpected character: {self._src[self._pos]!r}")

        if word in KEYWORDS:
            tt = KEYWORDS[word]
            if tt == TokenType.IS:
                self._emit_is_composite(line, col)
                return
            if tt == TokenType.OR:
                self._emit_or_composite(line, col)
                return
            self._tokens.append(Token(tt, word, line, col))
            return

        if word in RESERVED_KEYWORDS:
            raise UzonSyntaxError(
                f"Reserved keyword '{word}' cannot be used as identifier; use @{word} to escape",
                line, col,
            )

        self._tokens.append(Token(TokenType.IDENTIFIER, word, line, col))

    # ── composite operator lookahead (§9 lexer rules) ─────────────────

    def _emit_is_composite(self, line: int, col: int) -> None:
        """§9: Handle ``is``, ``is not``, ``is named``, ``is not named``, ``is type``, ``is not type``."""
        sp, sl, sc = self._pos, self._line, self._col

        self._skip_ws_and_newlines()
        nxt = self._peek_word()

        if nxt == "not":
            self._consume_word(nxt)
            self._skip_ws_and_newlines()
            nxt2 = self._peek_word()
            if nxt2 == "named":
                self._consume_word(nxt2)
                self._tokens.append(Token(TokenType.IS_NOT_NAMED, "is not named", line, col))
                return
            if nxt2 == "type":
                self._consume_word(nxt2)
                self._tokens.append(Token(TokenType.IS_NOT_TYPE, "is not type", line, col))
                return
            self._tokens.append(Token(TokenType.IS_NOT, "is not", line, col))
            return

        if nxt == "named":
            self._consume_word(nxt)
            self._tokens.append(Token(TokenType.IS_NAMED, "is named", line, col))
            return

        if nxt == "type":
            self._consume_word(nxt)
            self._tokens.append(Token(TokenType.IS_TYPE, "is type", line, col))
            return

        self._pos, self._line, self._col = sp, sl, sc
        self._tokens.append(Token(TokenType.IS, "is", line, col))

    def _emit_or_composite(self, line: int, col: int) -> None:
        """§9: Handle ``or`` and ``or else``."""
        sp, sl, sc = self._pos, self._line, self._col

        self._skip_ws_and_newlines()
        nxt = self._peek_word()

        if nxt == "else":
            self._consume_word(nxt)
            self._tokens.append(Token(TokenType.OR_ELSE, "or else", line, col))
            return

        self._pos, self._line, self._col = sp, sl, sc
        self._tokens.append(Token(TokenType.OR, "or", line, col))

    def _peek_word(self) -> str:
        """Peek at the next word without consuming it."""
        sp, sl, sc = self._pos, self._line, self._col
        word = self._read_raw_word()
        self._pos, self._line, self._col = sp, sl, sc
        return word

    def _consume_word(self, word: str) -> None:
        """Advance past a word we already peeked at."""
        for _ in word:
            self._advance()
