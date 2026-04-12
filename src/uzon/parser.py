# SPDX-FileCopyrightText: © 2026 Suho Kang
# SPDX-License-Identifier: MIT
"""UZON recursive descent parser — builds AST from token stream.

Implements the grammar defined in §9 with precedence climbing per §5.5.
Handles: NEWLINE_SEP rule (§8), binding decomposition (§9 lexer rules),
multiline string continuation (§4.4.2), and all expression forms.
"""

from __future__ import annotations

from .ast_nodes import (
    AreBinding, BinaryOp, Binding, BoolLiteral, CaseExpr, Conversion,
    Document, EnvRef, FieldExtraction, FloatLiteral, FromEnum, FromUnion,
    FunctionCall, FunctionExpr, FunctionParam, Grouping, Identifier,
    IfExpr, InfLiteral, IntegerLiteral, ListLiteral, MemberAccess,
    NamedVariant, NanLiteral, Node, NullLiteral, OrElse,
    StringLiteral, StructExtension, StructImport, StructLiteral,
    StructOverride, TupleLiteral, TypeAnnotation, TypeExpr, UnaryOp,
    UndefinedLiteral, WhenClause,
)
from .errors import UzonSyntaxError
from .tokens import ALL_KEYWORDS, KEYWORDS, Token, TokenType


class Parser:
    """Recursive descent parser for UZON per §9."""

    def __init__(self, tokens: list[Token], filename: str = "<string>"):
        self._tokens = tokens
        self._file = filename
        self._pos = 0
        self._no_multiline_string = False

    def parse(self) -> Document:
        """§9 document: Parse the entire token stream into a Document AST."""
        doc = Document(line=1, col=1)
        doc.bindings = self._parse_bindings(until=TokenType.EOF)
        return doc

    # ── token helpers ─────────────────────────────────────────────────

    def _peek(self) -> Token:
        return self._tokens[self._pos]

    def _peek_type(self) -> TokenType:
        return self._tokens[self._pos].type

    def _peek_at(self, offset: int) -> Token:
        idx = self._pos + offset
        if idx >= len(self._tokens):
            return self._tokens[-1]
        return self._tokens[idx]

    def _advance(self) -> Token:
        tok = self._tokens[self._pos]
        self._pos += 1
        return tok

    def _expect(self, tt: TokenType) -> Token:
        tok = self._peek()
        if tok.type != tt:
            raise self._error(f"Expected {tt.name}, got {tok.type.name} ({tok.value!r})")
        return self._advance()

    def _match(self, *types: TokenType) -> Token | None:
        if self._peek_type() in types:
            return self._advance()
        return None

    # Token types that are valid as type names in ``is type`` / ``case type``
    _TYPE_NAME_TOKENS: frozenset[TokenType] = frozenset({
        TokenType.IDENTIFIER, TokenType.NULL, TokenType.STRUCT,
    })

    def _expect_type_name(self) -> Token:
        """Expect a type name — IDENTIFIER or keyword that doubles as a type (e.g. null)."""
        tok = self._peek()
        if tok.type in self._TYPE_NAME_TOKENS:
            return self._advance()
        raise self._error(f"Expected type name, got {tok.type.name} ({tok.value!r})")

    def _error(self, msg: str) -> UzonSyntaxError:
        tok = self._peek()
        return UzonSyntaxError(msg, tok.line, tok.col, file=self._file)

    def _skip_nl(self) -> None:
        while self._peek_type() == TokenType.NEWLINE:
            self._advance()

    def _skip_nl_if_cont(self) -> None:
        """§8: Skip newlines only if what follows is NOT a new binding."""
        if self._peek_type() == TokenType.NEWLINE and not self._is_binding_start_ahead():
            self._skip_nl()

    # ── NEWLINE_SEP detection (§8) ────────────────────────────────────

    def _is_binding_start_at(self, pos: int) -> bool:
        """§8: Check if tokens at *pos* form a new binding (name is/are)."""
        i = pos
        while i < len(self._tokens) and self._tokens[i].type == TokenType.NEWLINE:
            i += 1
        if i >= len(self._tokens):
            return False
        t1 = self._tokens[i]
        if t1.type in (TokenType.RBRACE, TokenType.EOF):
            return True
        if t1.type == TokenType.IDENTIFIER and i + 1 < len(self._tokens):
            t2 = self._tokens[i + 1]
            if t2.type in (TokenType.IS, TokenType.IS_NOT, TokenType.IS_NAMED,
                           TokenType.IS_NOT_NAMED, TokenType.IS_TYPE,
                           TokenType.IS_NOT_TYPE, TokenType.ARE):
                return True
        return False

    def _is_binding_start_ahead(self) -> bool:
        return self._is_binding_start_at(self._pos)

    # ── binding list (§9 document / struct_literal) ───────────────────

    def _parse_bindings(self, until: TokenType) -> list[Binding | AreBinding]:
        bindings: list[Binding | AreBinding] = []
        self._skip_nl()
        while self._peek_type() != until:
            bindings.append(self._parse_binding())
            if self._match(TokenType.COMMA):
                self._skip_nl()
            elif self._peek_type() == TokenType.NEWLINE:
                self._skip_nl()
        return bindings

    def _parse_binding(self) -> Binding | AreBinding:
        """§9 binding: name (is_binding | are_binding).

        Handles binding decomposition (§9 lexer rules):
        composite ``is not``/``is named``/``is not named`` at binding position
        decomposes into ``is`` (binding) + remaining tokens (value expression).
        """
        name_tok = self._expect(TokenType.IDENTIFIER)
        name, line, col = name_tok.value, name_tok.line, name_tok.col
        tok = self._peek()

        if tok.type == TokenType.ARE:
            self._advance()
            return self._parse_are_binding(name, line, col)

        if tok.type == TokenType.IS:
            self._advance()
            return self._parse_is_binding(name, line, col)

        # §9 binding decomposition
        if tok.type == TokenType.IS_NOT:
            self._advance()
            self._tokens.insert(self._pos, Token(TokenType.NOT, "not", tok.line, tok.col))
            expr = self._parse_expression()
            return Binding(name=name, value=expr, called=self._try_called(), line=line, col=col)

        if tok.type == TokenType.IS_NAMED:
            self._advance()
            self._tokens.insert(self._pos, Token(TokenType.IDENTIFIER, "named", tok.line, tok.col))
            expr = self._parse_expression()
            return Binding(name=name, value=expr, called=self._try_called(), line=line, col=col)

        if tok.type == TokenType.IS_NOT_NAMED:
            self._advance()
            self._tokens.insert(self._pos, Token(TokenType.NOT, "not", tok.line, tok.col))
            self._tokens.insert(self._pos + 1, Token(TokenType.IDENTIFIER, "named", tok.line, tok.col))
            expr = self._parse_expression()
            return Binding(name=name, value=expr, called=self._try_called(), line=line, col=col)

        if tok.type == TokenType.IS_TYPE:
            self._advance()
            self._tokens.insert(self._pos, Token(TokenType.IDENTIFIER, "type", tok.line, tok.col))
            expr = self._parse_expression()
            return Binding(name=name, value=expr, called=self._try_called(), line=line, col=col)

        if tok.type == TokenType.IS_NOT_TYPE:
            self._advance()
            self._tokens.insert(self._pos, Token(TokenType.NOT, "not", tok.line, tok.col))
            self._tokens.insert(self._pos + 1, Token(TokenType.IDENTIFIER, "type", tok.line, tok.col))
            expr = self._parse_expression()
            return Binding(name=name, value=expr, called=self._try_called(), line=line, col=col)

        raise self._error(f"Expected 'is' or 'are' after binding name '{name}'")

    def _parse_is_binding(self, name: str, line: int, col: int) -> Binding:
        """§9 is_binding: 'is' ('of' member_access | expression) ['called' name]."""
        if self._peek_type() == TokenType.OF:
            self._advance()
            source = self._parse_member_access()
            return Binding(
                name=name,
                value=FieldExtraction(source=source, line=line, col=col),
                called=self._try_called(), line=line, col=col,
            )
        expr = self._parse_expression()
        return Binding(name=name, value=expr, called=self._try_called(), line=line, col=col)

    def _parse_are_binding(self, name: str, line: int, col: int) -> AreBinding:
        """§9 are_binding: 'are' expression {',' expression} ['as' type] ['called' name].

        Per §3.4.1: trailing ``as`` is list-level annotation, not element-level.
        """
        self._skip_nl()
        elements: list[Node] = [self._parse_expression()]

        while self._match(TokenType.COMMA):
            self._skip_nl()
            # §3.5 enum termination: comma + binding start → stop
            if self._peek_type() == TokenType.IDENTIFIER:
                nxt = self._peek_at(1)
                if nxt.type in (TokenType.IS, TokenType.IS_NOT, TokenType.IS_NAMED,
                                TokenType.IS_NOT_NAMED, TokenType.IS_TYPE,
                                TokenType.IS_NOT_TYPE, TokenType.ARE):
                    break
            elements.append(self._parse_expression())

        # §3.4.1: lift trailing ``as`` from last element to binding level
        type_ann = None
        if elements and isinstance(elements[-1], TypeAnnotation):
            type_ann = elements[-1].type
            elements[-1] = elements[-1].expr

        return AreBinding(
            name=name, elements=elements, type_annotation=type_ann,
            called=self._try_called(), line=line, col=col,
        )

    def _try_called(self) -> str | None:
        """§6.2: Optionally consume ``called name`` for type naming."""
        self._skip_nl_if_cont()
        if self._peek_type() == TokenType.CALLED:
            self._advance()
            return self._expect(TokenType.IDENTIFIER).value
        return None

    # ── expression parsing — precedence climbing per §5.5 ─────────────

    def _parse_expression(self) -> Node:
        return self._parse_or_else()

    def _parse_or_else(self) -> Node:
        """§5.7: or else — precedence 18 (lowest), left-associative."""
        left = self._parse_or()
        while True:
            self._skip_nl_if_cont()
            if self._peek_type() != TokenType.OR_ELSE:
                break
            tok = self._advance()
            left = OrElse(left=left, right=self._parse_or(), line=tok.line, col=tok.col)
        return left

    def _parse_or(self) -> Node:
        """§5.6: or — precedence 17, left-associative."""
        left = self._parse_and()
        while True:
            self._skip_nl_if_cont()
            if self._peek_type() != TokenType.OR:
                break
            tok = self._advance()
            left = BinaryOp(op="or", left=left, right=self._parse_and(), line=tok.line, col=tok.col)
        return left

    def _parse_and(self) -> Node:
        """§5.6: and — precedence 16, left-associative."""
        left = self._parse_not()
        while True:
            self._skip_nl_if_cont()
            if self._peek_type() != TokenType.AND:
                break
            tok = self._advance()
            left = BinaryOp(op="and", left=left, right=self._parse_not(), line=tok.line, col=tok.col)
        return left

    def _parse_not(self) -> Node:
        """§5.6: not — precedence 15, right-associative (prefix)."""
        if self._peek_type() == TokenType.NOT:
            tok = self._advance()
            return UnaryOp(op="not", operand=self._parse_not(), line=tok.line, col=tok.col)
        return self._parse_equality()

    def _parse_equality(self) -> Node:
        """§5.1/§5.2: is, is not, is named, is not named, is type, is not type — precedence 14."""
        left = self._parse_membership()
        self._skip_nl_if_cont()
        tok = self._peek()

        if tok.type == TokenType.IS_NOT_NAMED:
            self._advance()
            n = self._expect(TokenType.IDENTIFIER)
            return BinaryOp(op="is not named", left=left,
                            right=Identifier(name=n.value, line=n.line, col=n.col),
                            line=tok.line, col=tok.col)
        if tok.type == TokenType.IS_NOT_TYPE:
            self._advance()
            n = self._expect_type_name()
            return BinaryOp(op="is not type", left=left,
                            right=Identifier(name=n.value, line=n.line, col=n.col),
                            line=tok.line, col=tok.col)
        if tok.type == TokenType.IS_NAMED:
            self._advance()
            n = self._expect(TokenType.IDENTIFIER)
            return BinaryOp(op="is named", left=left,
                            right=Identifier(name=n.value, line=n.line, col=n.col),
                            line=tok.line, col=tok.col)
        if tok.type == TokenType.IS_TYPE:
            self._advance()
            n = self._expect_type_name()
            return BinaryOp(op="is type", left=left,
                            right=Identifier(name=n.value, line=n.line, col=n.col),
                            line=tok.line, col=tok.col)
        if tok.type == TokenType.IS_NOT:
            self._advance()
            return BinaryOp(op="is not", left=left, right=self._parse_membership(),
                            line=tok.line, col=tok.col)
        if tok.type == TokenType.IS:
            self._advance()
            return BinaryOp(op="is", left=left, right=self._parse_membership(),
                            line=tok.line, col=tok.col)
        return left

    def _parse_membership(self) -> Node:
        """§5.8.1: in — precedence 13, no chaining."""
        left = self._parse_relational()
        self._skip_nl_if_cont()
        if self._peek_type() == TokenType.IN:
            tok = self._advance()
            return BinaryOp(op="in", left=left, right=self._parse_relational(),
                            line=tok.line, col=tok.col)
        return left

    def _parse_relational(self) -> Node:
        """§5.4: <, <=, >, >= — precedence 12, no chaining."""
        left = self._parse_concat()
        self._skip_nl_if_cont()
        tok = self._peek()
        if tok.type in (TokenType.LT, TokenType.LE, TokenType.GT, TokenType.GE):
            self._advance()
            return BinaryOp(op=tok.value, left=left, right=self._parse_concat(),
                            line=tok.line, col=tok.col)
        return left

    def _parse_concat(self) -> Node:
        """§5.8.2: ++ — precedence 11, left-associative."""
        left = self._parse_addition()
        while True:
            self._skip_nl_if_cont()
            if self._peek_type() != TokenType.PLUS_PLUS:
                break
            tok = self._advance()
            self._skip_nl_if_cont()
            left = BinaryOp(op="++", left=left, right=self._parse_addition(),
                            line=tok.line, col=tok.col)
        return left

    def _parse_addition(self) -> Node:
        """§5.3: +, - — precedence 10, left-associative."""
        left = self._parse_multiplication()
        while True:
            self._skip_nl_if_cont()
            if self._peek_type() not in (TokenType.PLUS, TokenType.MINUS):
                break
            tok = self._advance()
            left = BinaryOp(op=tok.value, left=left, right=self._parse_multiplication(),
                            line=tok.line, col=tok.col)
        return left

    def _parse_multiplication(self) -> Node:
        """§5.3/§5.8.3: *, /, %, ** — precedence 9, left-associative."""
        left = self._parse_unary()
        while True:
            self._skip_nl_if_cont()
            if self._peek_type() not in (TokenType.STAR, TokenType.SLASH,
                                         TokenType.PERCENT, TokenType.STAR_STAR):
                break
            tok = self._advance()
            left = BinaryOp(op=tok.value, left=left, right=self._parse_unary(),
                            line=tok.line, col=tok.col)
        return left

    def _parse_unary(self) -> Node:
        """§5.5: Unary negation — precedence 8, right-associative."""
        if self._peek_type() == TokenType.MINUS:
            tok = self._advance()
            return UnaryOp(op="-", operand=self._parse_power(), line=tok.line, col=tok.col)
        return self._parse_power()

    def _parse_power(self) -> Node:
        """§5.3: ^ exponentiation — precedence 7, right-associative."""
        base = self._parse_type_decl()
        self._skip_nl_if_cont()
        if self._peek_type() == TokenType.CARET:
            tok = self._advance()
            return BinaryOp(op="^", left=base, right=self._parse_unary(),
                            line=tok.line, col=tok.col)
        return base

    # ── postfix chain ─────────────────────────────────────────────────

    def _parse_type_decl(self) -> Node:
        """§9 type_decl = type_annot_level [from_clause | named_clause]."""
        node = self._parse_type_annot()
        self._skip_nl_if_cont()
        if self._peek_type() == TokenType.FROM:
            return self._parse_from_clause(node)
        if self._peek_type() == TokenType.NAMED:
            return self._parse_named_clause(node)
        return node

    def _parse_from_clause(self, value: Node) -> Node:
        """§9 from_clause: ``from`` (enum variants | ``union`` types)."""
        tok = self._advance()  # from

        # §3.6: union
        if self._peek_type() == TokenType.UNION:
            self._advance()
            types = [self._parse_type_expr()]
            while self._peek_type() == TokenType.COMMA:
                if self._is_binding_start_at(self._pos + 1):
                    break
                self._advance()
                self._skip_nl()
                types.append(self._parse_type_expr())
            return FromUnion(value=value, types=types, line=tok.line, col=tok.col)

        # §3.5: enum variants with termination rules
        variants = [self._parse_variant_name()]
        while self._peek_type() == TokenType.COMMA:
            if self._is_binding_start_at(self._pos + 1):
                break
            # §3.5: called terminates enum
            np = self._pos + 1
            while np < len(self._tokens) and self._tokens[np].type == TokenType.NEWLINE:
                np += 1
            if np < len(self._tokens) and self._tokens[np].type == TokenType.CALLED:
                break
            self._advance()
            self._skip_nl()
            variants.append(self._parse_variant_name())

        return FromEnum(value=value, variants=variants, line=tok.line, col=tok.col)

    def _parse_variant_name(self) -> str:
        """§9 variant_name: name or keyword — keywords are valid variant names."""
        tok = self._peek()
        if tok.type == TokenType.IDENTIFIER:
            self._advance()
            return tok.value
        if tok.value in KEYWORDS:
            self._advance()
            return tok.value
        raise self._error(f"Expected variant name, got {tok.type.name} ({tok.value!r})")

    def _parse_named_clause(self, value: Node) -> Node:
        """§9 named_clause: ``named`` tag [``from`` variant ``as`` type, ...]."""
        tok = self._advance()  # named
        tag = self._parse_variant_name()

        variants: list[tuple[str, TypeExpr]] = []
        self._skip_nl_if_cont()
        if self._peek_type() == TokenType.FROM:
            self._advance()
            self._skip_nl()
            vn = self._parse_variant_name()
            self._expect(TokenType.AS)
            vt = self._parse_type_expr()
            variants.append((vn, vt))
            while self._peek_type() == TokenType.COMMA:
                if self._is_binding_start_at(self._pos + 1):
                    break
                self._advance()
                self._skip_nl()
                vn = self._parse_variant_name()
                self._expect(TokenType.AS)
                vt = self._parse_type_expr()
                variants.append((vn, vt))

        return NamedVariant(value=value, tag=tag, variants=variants, line=tok.line, col=tok.col)

    def _parse_type_annot(self) -> Node:
        """§9: struct_override [``as`` type_expr]."""
        node = self._parse_struct_override()
        self._skip_nl_if_cont()
        if self._peek_type() == TokenType.AS:
            tok = self._advance()
            te = self._parse_type_expr()
            node = TypeAnnotation(expr=node, type=te, line=tok.line, col=tok.col)
        # Allow ``to`` after ``as``: e.g. ``100 as u8 to u16``
        self._skip_nl_if_cont()
        if self._peek_type() == TokenType.TO:
            tok = self._advance()
            te = self._parse_type_expr()
            node = Conversion(expr=node, type=te, line=tok.line, col=tok.col)
        return node

    def _parse_struct_override(self) -> Node:
        """§9: conversion [(``with`` | ``plus``) struct_literal]."""
        node = self._parse_conversion()
        self._skip_nl_if_cont()
        if self._peek_type() == TokenType.WITH:
            tok = self._advance()
            s = self._parse_struct_literal()
            node = StructOverride(base=node, overrides=s, line=tok.line, col=tok.col)
            self._skip_nl_if_cont()
            if self._peek_type() in (TokenType.WITH, TokenType.KW_PLUS):
                raise self._error("Cannot chain 'with'/'plus' — use an intermediate binding")
        elif self._peek_type() == TokenType.KW_PLUS:
            tok = self._advance()
            s = self._parse_struct_literal()
            node = StructExtension(base=node, extensions=s, line=tok.line, col=tok.col)
            self._skip_nl_if_cont()
            if self._peek_type() in (TokenType.WITH, TokenType.KW_PLUS):
                raise self._error("Cannot chain 'with'/'plus' — use an intermediate binding")
        return node

    def _parse_conversion(self) -> Node:
        """§9: call_or_access [``to`` type_expr]."""
        node = self._parse_call_or_access()
        self._skip_nl_if_cont()
        if self._peek_type() == TokenType.TO:
            tok = self._advance()
            node = Conversion(expr=node, type=self._parse_type_expr(), line=tok.line, col=tok.col)
        return node

    def _parse_member_access(self) -> Node:
        """§9 member_access: primary {'.' (name | integer)} — no calls.
        Used only by field extraction (§5.14).
        """
        node = self._parse_primary()
        self._skip_nl_if_cont()
        while self._peek_type() == TokenType.DOT:
            self._advance()
            tok = self._peek()
            if tok.type in (TokenType.INTEGER, TokenType.IDENTIFIER):
                self._advance()
                node = MemberAccess(object=node, member=tok.value, line=tok.line, col=tok.col)
            else:
                raise self._error(f"Expected member name after '.', got {tok.type.name}")
            self._skip_nl_if_cont()
        return node

    @staticmethod
    def _is_non_callable(node: Node) -> bool:
        """Struct and list literals are never callable; a trailing '(' would
        incorrectly consume the next expression."""
        return isinstance(node, (StructLiteral, ListLiteral))

    def _parse_call_or_access(self) -> Node:
        """§9 call_or_access: primary {'.' member | '(' args ')'}."""
        node = self._parse_primary()
        self._skip_nl_if_cont()
        while True:
            tt = self._peek_type()
            if tt == TokenType.DOT:
                self._advance()
                tok = self._peek()
                if tok.type in (TokenType.INTEGER, TokenType.IDENTIFIER):
                    self._advance()
                    node = MemberAccess(object=node, member=tok.value, line=tok.line, col=tok.col)
                else:
                    raise self._error(f"Expected member name after '.', got {tok.type.name}")
            elif tt == TokenType.LPAREN and not self._is_non_callable(node):
                tok = self._advance()  # (
                self._skip_nl()
                args: list[Node] = []
                if self._peek_type() != TokenType.RPAREN:
                    args.append(self._parse_expression())
                    self._skip_nl()
                    while self._peek_type() == TokenType.COMMA:
                        self._advance()
                        self._skip_nl()
                        if self._peek_type() == TokenType.RPAREN:
                            break
                        args.append(self._parse_expression())
                        self._skip_nl()
                self._expect(TokenType.RPAREN)
                node = FunctionCall(callee=node, args=args, line=tok.line, col=tok.col)
            else:
                break
            self._skip_nl_if_cont()
        return node

    # ── primary (§9) ──────────────────────────────────────────────────

    def _parse_primary(self) -> Node:
        tok = self._peek()

        if tok.type == TokenType.INTEGER:
            self._advance()
            return IntegerLiteral(value=tok.value, line=tok.line, col=tok.col)
        if tok.type == TokenType.FLOAT:
            self._advance()
            return FloatLiteral(value=tok.value, line=tok.line, col=tok.col)
        if tok.type == TokenType.STRING:
            return self._parse_string_literal()
        if tok.type == TokenType.TRUE:
            self._advance()
            return BoolLiteral(value=True, line=tok.line, col=tok.col)
        if tok.type == TokenType.FALSE:
            self._advance()
            return BoolLiteral(value=False, line=tok.line, col=tok.col)
        if tok.type == TokenType.NULL:
            self._advance()
            return NullLiteral(line=tok.line, col=tok.col)
        if tok.type == TokenType.UNDEFINED:
            self._advance()
            return UndefinedLiteral(line=tok.line, col=tok.col)
        if tok.type == TokenType.INF:
            self._advance()
            return InfLiteral(line=tok.line, col=tok.col)
        if tok.type == TokenType.NAN:
            self._advance()
            return NanLiteral(line=tok.line, col=tok.col)
        if tok.type == TokenType.ENV:
            self._advance()
            return EnvRef(line=tok.line, col=tok.col)
        if tok.type == TokenType.IDENTIFIER:
            self._advance()
            return Identifier(name=tok.value, line=tok.line, col=tok.col)
        if tok.type == TokenType.LBRACE:
            return self._parse_struct_literal()
        if tok.type == TokenType.LBRACKET:
            return self._parse_list_literal()
        if tok.type == TokenType.LPAREN:
            return self._parse_tuple_or_group()
        if tok.type == TokenType.IF:
            return self._parse_if_expr()
        if tok.type == TokenType.CASE:
            return self._parse_case_expr()
        if tok.type == TokenType.STRUCT:
            return self._parse_struct_import()
        if tok.type == TokenType.FUNCTION:
            return self._parse_function_expr()

        raise self._error(f"Unexpected token: {tok.type.name} ({tok.value!r})")

    def _parse_string_literal(self) -> Node:
        """§4.4/§4.4.2: String with optional interpolation and multiline continuation."""
        first = self._peek()
        parts: list[str | Node] = []
        self._parse_string_segment(parts)

        # §4.4.2: Multiline strings — adjacent strings joined with \n
        while not self._no_multiline_string:
            if self._peek_type() == TokenType.NEWLINE:
                i, nc = self._pos, 0
                while i < len(self._tokens) and self._tokens[i].type == TokenType.NEWLINE:
                    nc += 1; i += 1
                if nc == 1 and i < len(self._tokens) and self._tokens[i].type == TokenType.STRING:
                    self._advance()  # consume single NEWLINE
                    parts.append("\n")
                    self._parse_string_segment(parts)
                    continue
            break

        return StringLiteral(parts=parts, line=first.line, col=first.col)

    def _parse_string_segment(self, parts: list[str | Node]) -> None:
        """Parse one string token and any following interpolation tokens."""
        tok = self._peek()
        if tok.type == TokenType.STRING:
            self._advance()
            parts.append(tok.value)
            while self._peek_type() == TokenType.INTERP_START:
                self._advance()
                parts.append(self._parse_expression())
                self._expect(TokenType.INTERP_END)
                if self._peek_type() == TokenType.STRING:
                    tok = self._advance()
                    parts.append(tok.value)

    def _parse_struct_literal(self) -> StructLiteral:
        """§3.2: Struct literal — { binding, ... }."""
        tok = self._expect(TokenType.LBRACE)
        self._skip_nl()
        fields = self._parse_bindings(until=TokenType.RBRACE)
        self._expect(TokenType.RBRACE)
        return StructLiteral(fields=fields, line=tok.line, col=tok.col)

    def _parse_list_literal(self) -> ListLiteral:
        """§3.4: List literal — [ expr, ... ]. Commas required."""
        tok = self._expect(TokenType.LBRACKET)
        self._skip_nl()
        elems: list[Node] = []
        if self._peek_type() != TokenType.RBRACKET:
            elems.append(self._parse_expression())
            while self._match(TokenType.COMMA):
                self._skip_nl()
                if self._peek_type() == TokenType.RBRACKET:
                    break
                elems.append(self._parse_expression())
        self._skip_nl()
        self._expect(TokenType.RBRACKET)
        return ListLiteral(elements=elems, line=tok.line, col=tok.col)

    def _parse_tuple_or_group(self) -> Node:
        """§3.3/§9: () empty tuple, (expr) grouping, (expr,) 1-tuple, (expr,...) tuple."""
        tok = self._expect(TokenType.LPAREN)
        self._skip_nl()
        if self._peek_type() == TokenType.RPAREN:
            self._advance()
            return TupleLiteral(elements=[], line=tok.line, col=tok.col)
        first = self._parse_expression()
        self._skip_nl()
        if self._match(TokenType.COMMA):
            self._skip_nl()
            elems = [first]
            if self._peek_type() != TokenType.RPAREN:
                elems.append(self._parse_expression())
                while self._match(TokenType.COMMA):
                    self._skip_nl()
                    if self._peek_type() == TokenType.RPAREN:
                        break
                    elems.append(self._parse_expression())
            self._skip_nl()
            self._expect(TokenType.RPAREN)
            return TupleLiteral(elements=elems, line=tok.line, col=tok.col)
        self._expect(TokenType.RPAREN)
        return Grouping(expr=first, line=tok.line, col=tok.col)

    def _parse_if_expr(self) -> IfExpr:
        """§5.9: if condition then expr else expr."""
        tok = self._expect(TokenType.IF)
        cond = self._parse_expression()
        self._skip_nl()
        self._expect(TokenType.THEN)
        then_ = self._parse_expression()
        self._skip_nl()
        self._expect(TokenType.ELSE)
        else_ = self._parse_expression()
        return IfExpr(condition=cond, then_branch=then_, else_branch=else_,
                      line=tok.line, col=tok.col)

    def _parse_case_expr(self) -> CaseExpr:
        """§5.10: case [type|named] expr when ... then expr ... else expr."""
        tok = self._expect(TokenType.CASE)
        case_kind = "value"
        if self._peek_type() == TokenType.TYPE:
            self._advance()
            case_kind = "type"
        elif self._peek_type() == TokenType.NAMED:
            self._advance()
            case_kind = "named"
        scrutinee = self._parse_expression()
        whens: list[WhenClause] = []
        self._skip_nl()
        while self._peek_type() == TokenType.WHEN:
            wt = self._advance()
            self._skip_nl()
            if case_kind == "named":
                nt = self._expect(TokenType.IDENTIFIER)
                val: Node = Identifier(name=nt.value, line=nt.line, col=nt.col)
            elif case_kind == "type":
                nt = self._expect_type_name()
                val = Identifier(name=nt.value, line=nt.line, col=nt.col)
            else:
                val = self._parse_expression()
            self._expect(TokenType.THEN)
            result = self._parse_expression()
            whens.append(WhenClause(value=val, result=result, kind=case_kind,
                                    line=wt.line, col=wt.col))
            self._skip_nl()
        if not whens:
            raise self._error("'case' requires at least one 'when' clause")
        self._expect(TokenType.ELSE)
        else_ = self._parse_expression()
        return CaseExpr(scrutinee=scrutinee, when_clauses=whens, else_branch=else_,
                        line=tok.line, col=tok.col)

    def _parse_struct_import(self) -> Node:
        """§7.1: struct \"path\" — file import. Interpolated paths rejected."""
        tok = self._expect(TokenType.STRUCT)
        path_tok = self._expect(TokenType.STRING)
        if self._peek_type() == TokenType.INTERP_START:
            raise self._error("Struct import path cannot contain interpolation")
        node: Node = StructImport(path=path_tok.value, line=tok.line, col=tok.col)
        while self._peek_type() == TokenType.DOT:
            self._advance()
            mt = self._peek()
            if mt.type in (TokenType.IDENTIFIER, TokenType.INTEGER):
                self._advance()
                node = MemberAccess(object=node, member=mt.value, line=mt.line, col=mt.col)
            else:
                raise self._error(f"Expected member name after '.', got {mt.type.name}")
        return node

    # ── functions (§3.8) ──────────────────────────────────────────────

    def _parse_function_expr(self) -> FunctionExpr:
        """§3.8: function [params] returns type { body }."""
        tok = self._expect(TokenType.FUNCTION)
        params: list[FunctionParam] = []
        has_default = False
        while self._peek_type() != TokenType.RETURNS:
            if params:
                self._expect(TokenType.COMMA)
                self._skip_nl()
            p = self._parse_function_param()
            if p.default is not None:
                has_default = True
            elif has_default:
                raise self._error(f"Required parameter '{p.name}' after defaulted parameter")
            params.append(p)

        self._expect(TokenType.RETURNS)
        ret_type = self._parse_type_expr()
        self._expect(TokenType.LBRACE)
        self._skip_nl()

        body_bindings: list[Binding] = []
        body_expr: Node | None = None
        saved = self._no_multiline_string
        self._no_multiline_string = True

        while self._peek_type() != TokenType.RBRACE:
            if self._is_func_binding_start():
                nt = self._advance()
                self._expect(TokenType.IS)
                v = self._parse_expression()
                body_bindings.append(Binding(name=nt.value, value=v, line=nt.line, col=nt.col))
                self._skip_nl()
                if self._peek_type() == TokenType.COMMA:
                    self._advance()
                    self._skip_nl()
            else:
                body_expr = self._parse_expression()
                self._skip_nl()
                break

        self._no_multiline_string = saved
        if body_expr is None:
            raise self._error("Function body must end with an expression")
        self._expect(TokenType.RBRACE)
        return FunctionExpr(params=params, return_type=ret_type,
                            body_bindings=body_bindings, body_expr=body_expr,
                            line=tok.line, col=tok.col)

    def _parse_function_param(self) -> FunctionParam:
        """§9 param: name ``as`` type_expr [``default`` expression]."""
        nt = self._peek()
        if nt.type != TokenType.IDENTIFIER:
            raise self._error(f"Expected parameter name, got {nt.type.name}")
        self._advance()
        self._expect(TokenType.AS)
        te = self._parse_type_expr()
        default: Node | None = None
        if self._peek_type() == TokenType.DEFAULT:
            self._advance()
            default = self._parse_expression()
        return FunctionParam(name=nt.value, type=te, default=default, line=nt.line, col=nt.col)

    def _is_func_binding_start(self) -> bool:
        if self._pos + 1 >= len(self._tokens):
            return False
        return (self._tokens[self._pos].type == TokenType.IDENTIFIER
                and self._tokens[self._pos + 1].type == TokenType.IS)

    # ── type expressions (§9 type_expr) ───────────────────────────────

    def _parse_type_expr(self) -> TypeExpr:
        """§9: named_type | list_type | tuple_type | ``null``."""
        tok = self._peek()

        if tok.type == TokenType.LBRACKET:
            self._advance()
            inner = self._parse_type_expr()
            self._expect(TokenType.RBRACKET)
            return TypeExpr(name=f"[{inner.name}]", is_list=True, inner=inner,
                            line=tok.line, col=tok.col)

        if tok.type == TokenType.LPAREN:
            self._advance()
            elems = [self._parse_type_expr()]
            self._expect(TokenType.COMMA)
            elems.append(self._parse_type_expr())
            while self._peek_type() == TokenType.COMMA:
                self._advance()
                if self._peek_type() == TokenType.RPAREN:
                    break
                elems.append(self._parse_type_expr())
            self._expect(TokenType.RPAREN)
            name = "(" + ", ".join(e.name for e in elems) + ")"
            return TypeExpr(name=name, is_tuple=True, elements=elems,
                            line=tok.line, col=tok.col)

        if tok.type == TokenType.NULL:
            self._advance()
            return TypeExpr(name="null", line=tok.line, col=tok.col)

        nt = self._expect(TokenType.IDENTIFIER)
        path = [nt.value]
        while self._peek_type() == TokenType.DOT:
            self._advance()
            path.append(self._expect(TokenType.IDENTIFIER).value)

        if len(path) == 1:
            return TypeExpr(name=path[0], line=nt.line, col=nt.col)
        return TypeExpr(name=".".join(path), path=path, line=nt.line, col=nt.col)
