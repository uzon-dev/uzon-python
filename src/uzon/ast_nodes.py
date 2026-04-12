# SPDX-FileCopyrightText: © 2026 Suho Kang
# SPDX-License-Identifier: MIT
"""UZON AST node definitions per §9 (Formal Grammar).

Each node maps to a production rule in the EBNF grammar.
All nodes carry source position (line, col) for error reporting (§11.2.0).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Node:
    """Base AST node with source position for error reporting (§11.2.0)."""
    line: int = 0
    col: int = 0


# ── Literals (§4) ──────────────────────────────────────────────────

@dataclass
class IntegerLiteral(Node):
    """§4.2: Integer literal — decimal, hex, octal, or binary."""
    value: str = ""       # raw string preserves base prefix
    negative: bool = False

@dataclass
class FloatLiteral(Node):
    """§4.3: Float literal — decimal with optional exponent."""
    value: str = ""       # raw string
    negative: bool = False

@dataclass
class BoolLiteral(Node):
    """§4.1: Boolean literal — true or false."""
    value: bool = False

@dataclass
class StringLiteral(Node):
    """§4.4: String literal with optional interpolation (§4.4.1).

    parts: list of str (literal text) and Node (interpolated expressions).
    A plain string has a single str element.
    """
    parts: list[str | Node] = field(default_factory=list)

@dataclass
class NullLiteral(Node):
    """§4.5: Null — intentional absence of value."""

@dataclass
class UndefinedLiteral(Node):
    """§3.1: Undefined state, not a value. Cannot appear as RHS of binding."""

@dataclass
class InfLiteral(Node):
    """§4.3: Positive infinity (IEEE 754)."""

@dataclass
class NanLiteral(Node):
    """§4.3: NaN — not a number (IEEE 754)."""


# ── Identifiers (§2.3) ─────────────────────────────────────────────

@dataclass
class Identifier(Node):
    """§2.3: Bare or quoted identifier for names and variant references."""
    name: str = ""


# ── References (§5.12, §5.13) ──────────────────────────────────────

@dataclass
class EnvRef(Node):
    """§5.13: The ``env`` keyword — environment variable access."""


# ── Expressions (§5) ───────────────────────────────────────────────

@dataclass
class MemberAccess(Node):
    """§5.5 precedence 1: dot-notation member access (env.X, list.0, etc.)."""
    object: Node = field(default_factory=Node)
    member: str = ""       # field name or numeric index as string

@dataclass
class BinaryOp(Node):
    """Binary operator expression: arithmetic (§5.3), comparison (§5.4),
    logic (§5.6), collection (§5.8), equality (§5.1/§5.2), membership (§5.8.1)."""
    op: str = ""
    left: Node = field(default_factory=Node)
    right: Node = field(default_factory=Node)

@dataclass
class UnaryOp(Node):
    """Unary operator: negation (-) per §5.5, logical not (§5.6)."""
    op: str = ""
    operand: Node = field(default_factory=Node)

@dataclass
class OrElse(Node):
    """§5.7: Undefined coalescing — ``or else`` composite operator."""
    left: Node = field(default_factory=Node)
    right: Node = field(default_factory=Node)

@dataclass
class IfExpr(Node):
    """§5.9: Conditional — if/then/else. Both branches required."""
    condition: Node = field(default_factory=Node)
    then_branch: Node = field(default_factory=Node)
    else_branch: Node = field(default_factory=Node)

@dataclass
class WhenClause(Node):
    """§5.10: A single ``when`` clause in a ``case`` expression."""
    value: Node = field(default_factory=Node)
    result: Node = field(default_factory=Node)
    kind: str = "value"  # "value" | "type" | "named" — set by case [type|named]

@dataclass
class CaseExpr(Node):
    """§5.10: Multi-branch conditional — case/when/else. At least one when required."""
    scrutinee: Node = field(default_factory=Node)
    when_clauses: list[WhenClause] = field(default_factory=list)
    else_branch: Node = field(default_factory=Node)


# ── Type system (§6) ───────────────────────────────────────────────

@dataclass
class TypeExpr(Node):
    """§9 type_expr: A type reference — simple name, dotted path, [T], or (T, T)."""
    name: str = ""
    path: list[str] = field(default_factory=list)       # dotted path, e.g. ["inner", "RGB"]
    is_list: bool = False                                # [Type]
    inner: TypeExpr | None = None                        # element type for list types
    is_tuple: bool = False                               # (T, T, ...)
    elements: list[TypeExpr] = field(default_factory=list)  # element types for tuple types

@dataclass
class TypeAnnotation(Node):
    """§6.1: Type annotation — expr ``as`` type."""
    expr: Node = field(default_factory=Node)
    type: TypeExpr = field(default_factory=TypeExpr)

@dataclass
class Conversion(Node):
    """§5.11: Type conversion — expr ``to`` type."""
    expr: Node = field(default_factory=Node)
    type: TypeExpr = field(default_factory=TypeExpr)

@dataclass
class StructOverride(Node):
    """§3.2.1: Struct override — expr ``with`` struct_literal."""
    base: Node = field(default_factory=Node)
    overrides: StructLiteral = None  # type: ignore[assignment]

@dataclass
class StructExtension(Node):
    """§3.2.2: Struct extension — expr ``plus`` struct_literal."""
    base: Node = field(default_factory=Node)
    extensions: StructLiteral = None  # type: ignore[assignment]

@dataclass
class FromEnum(Node):
    """§3.5: Enum definition — value ``from`` variant1, variant2, ..."""
    value: Node = field(default_factory=Node)
    variants: list[str] = field(default_factory=list)

@dataclass
class FromUnion(Node):
    """§3.6: Union definition — value ``from union`` type1, type2, ..."""
    value: Node = field(default_factory=Node)
    types: list[TypeExpr] = field(default_factory=list)

@dataclass
class NamedVariant(Node):
    """§3.7: Tagged union — value ``named`` tag [``from`` variant ``as`` type, ...]."""
    value: Node = field(default_factory=Node)
    tag: str = ""
    variants: list[tuple[str, TypeExpr]] = field(default_factory=list)
    # empty variants → type comes from an outer ``as`` annotation

@dataclass
class FieldExtraction(Node):
    """§5.14: Field extraction — ``is of`` member_access."""
    source: Node = field(default_factory=Node)


# ── Compounds (§3.2, §3.3, §3.4) ──────────────────────────────────

@dataclass
class StructLiteral(Node):
    """§3.2: Struct literal — { field1 is v1, field2 is v2, ... }."""
    fields: list[Binding] = field(default_factory=list)

@dataclass
class ListLiteral(Node):
    """§3.4: List literal — [ elem1, elem2, ... ]."""
    elements: list[Node] = field(default_factory=list)

@dataclass
class TupleLiteral(Node):
    """§3.3: Tuple literal — (elem1, elem2, ...) or (elem,) for 1-tuple."""
    elements: list[Node] = field(default_factory=list)

@dataclass
class Grouping(Node):
    """Parenthesized expression — (expr) without comma. Not a tuple."""
    expr: Node = field(default_factory=Node)


# ── Functions (§3.8) ───────────────────────────────────────────────

@dataclass
class FunctionParam(Node):
    """§3.8: Function parameter — name ``as`` type [``default`` expr]."""
    name: str = ""
    type: TypeExpr = field(default_factory=TypeExpr)
    default: Node | None = None

@dataclass
class FunctionExpr(Node):
    """§3.8: Function definition — function [params] returns type { body }."""
    params: list[FunctionParam] = field(default_factory=list)
    return_type: TypeExpr = field(default_factory=TypeExpr)
    body_bindings: list[Binding] = field(default_factory=list)
    body_expr: Node = field(default_factory=Node)

@dataclass
class FunctionCall(Node):
    """§5.15: Function call — callee(arg1, arg2, ...)."""
    callee: Node = field(default_factory=Node)
    args: list[Node] = field(default_factory=list)


# ── Import (§7.1) ──────────────────────────────────────────────────

@dataclass
class StructImport(Node):
    """§7.1: File import — struct \"path\"."""
    path: str = ""


# ── Bindings (§5.1, §3.4.1) ───────────────────────────────────────

@dataclass
class Binding(Node):
    """§5.1: Binding — name ``is`` value [``called`` TypeName]."""
    name: str = ""
    value: Node = field(default_factory=Node)
    called: str | None = None   # optional type naming (§6.2)

@dataclass
class AreBinding(Node):
    """§3.4.1: List binding — name ``are`` elem1, elem2, ... [``as`` Type] [``called`` Name]."""
    name: str = ""
    elements: list[Node] = field(default_factory=list)
    type_annotation: TypeExpr | None = None   # trailing list-level ``as``
    called: str | None = None


# ── Document (§1, §7.2) ───────────────────────────────────────────

@dataclass
class Document(Node):
    """A UZON document is an anonymous struct (§1) — a sequence of bindings."""
    bindings: list[Binding | AreBinding] = field(default_factory=list)
