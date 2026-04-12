# SPDX-FileCopyrightText: © 2026 Suho Kang
# SPDX-License-Identifier: MIT
"""UZON evaluator core — walks AST and produces Python values.

Implements the main evaluation loop: binding resolution and node dispatch
(§3, §5). Delegates to mixin classes for dependency resolution, type system,
operators, control flow, struct operations, functions, and standard library.
"""

from __future__ import annotations

from typing import Any

from ..ast_nodes import (
    AreBinding, BinaryOp, Binding, BoolLiteral, CaseExpr, Conversion,
    Document, EnvRef, FieldExtraction, FloatLiteral, FromEnum, FromUnion,
    FunctionCall, FunctionExpr, Grouping, Identifier, IfExpr, InfLiteral,
    IntegerLiteral, ListLiteral, MemberAccess, NamedVariant, NanLiteral,
    Node, NullLiteral, OrElse, StringLiteral, StructImport,
    StructLiteral, StructExtension, StructOverride, TupleLiteral,
    TypeAnnotation, UnaryOp, UndefinedLiteral,
)
from ..errors import UzonRuntimeError, UzonSyntaxError, UzonTypeError
from ..scope import Scope
from ..types import UzonFloat, UzonInt, UzonUndefined
from ._constants import I64_MIN, I64_MAX, SPECULATIVE_FAILED
from ._control import ControlMixin
from ._dependencies import DependencyMixin
from ._functions import FunctionMixin
from ._operators import OperatorMixin
from ._stdlib import StdlibMixin
from ._structs import StructMixin
from ._type_annotation import TypeAnnotationMixin
from ._type_checks import TypeChecksMixin
from ._type_conversion import TypeConversionMixin


class Evaluator(
    DependencyMixin,
    TypeAnnotationMixin,
    TypeConversionMixin,
    TypeChecksMixin,
    OperatorMixin,
    ControlMixin,
    StructMixin,
    FunctionMixin,
    StdlibMixin,
):
    """Evaluate a UZON AST into Python values."""

    def __init__(self, filename: str = "<string>"):
        self._filename = filename
        self._import_cache: dict[str, dict[str, Any]] = {}
        self._import_stack: list[str] = []  # §7.2: circular import detection
        self._scope_of: dict[int, Scope] = {}   # id(result_dict) → Scope
        self._called_of: dict[int, str] = {}    # id(result_dict) → type_name (§6.2)
        self._call_stack: list[int] = []  # §3.8: recursion detection

    def evaluate(self, doc: Document) -> dict[str, Any]:
        """§3: Evaluate a Document AST into a Python dict."""
        scope = Scope()
        scope.define("std", self._build_std())
        self._evaluate_bindings(doc.bindings, scope)
        result = scope.to_dict()
        result.pop("std", None)
        return result

    # ── binding evaluation ─────────────────────────────────────────────

    def _evaluate_bindings(
        self, bindings: list[Binding | AreBinding], scope: Scope,
        *, struct_context: bool = False,
    ) -> None:
        """Evaluate bindings with dependency resolution and duplicate detection."""
        # §3.1: detect duplicates; self-referencing overrides are allowed
        seen: dict[str, Binding | AreBinding] = {}
        override_bindings: list[Binding | AreBinding] = []
        primary_bindings: list[Binding | AreBinding] = []
        for b in bindings:
            if b.name in seen:
                is_of = isinstance(b, Binding) and isinstance(b.value, FieldExtraction)
                if not is_of and not self._references_own_name(b, b.name):
                    raise UzonSyntaxError(
                        f"Duplicate binding '{b.name}' in same scope",
                        b.line, b.col, file=self._filename,
                    )
                override_bindings.append(b)
            else:
                seen[b.name] = b
                primary_bindings.append(b)
        bindings = primary_bindings

        # §3.8: Static recursion check — function bodies must not call themselves
        for b in bindings:
            if isinstance(b, Binding) and isinstance(b.value, FunctionExpr):
                body_refs: set[str] = set()
                for body_b in b.value.body_bindings:
                    self._collect_bare_refs(body_b.value, body_refs)
                self._collect_bare_refs(b.value.body_expr, body_refs)
                if b.name in body_refs:
                    raise UzonTypeError(
                        "Recursive function call detected — call graph must be a DAG",
                        b.line, b.col, file=self._filename,
                    )

        deps = self._build_dependencies(bindings)
        order = self._topological_sort(bindings, deps)

        for b in order:
            value = self._eval_binding(b, scope, struct_context=struct_context)
            scope.define(b.name, value)

        for b in override_bindings:
            value = self._eval_binding(b, scope, struct_context=struct_context)
            scope.define(b.name, value)

    def _eval_binding(
        self, b: Binding | AreBinding, scope: Scope,
        struct_context: bool = False,
    ) -> Any:
        """Evaluate a single binding."""
        if isinstance(b, AreBinding):
            return self._eval_are_binding(b, scope)

        value = self._eval_node(b.value, scope, exclude=b.name)

        # §6.1: Empty list requires type annotation (relaxed inside struct literals)
        if not struct_context and isinstance(value, list) and len(value) == 0:
            if not isinstance(b.value, (TypeAnnotation, IfExpr, CaseExpr, OrElse, FunctionCall, BinaryOp)):
                raise UzonTypeError(
                    "Empty list requires explicit type annotation: [] as [Type]",
                    b.value.line, b.value.col, file=self._filename,
                )

        # All-null list requires type annotation (relaxed inside struct literals)
        if (not struct_context and isinstance(value, list) and value
                and not isinstance(b.value, TypeAnnotation)
                and all(e is None for e in value)):
            raise UzonTypeError(
                "All-null list requires explicit type annotation: [...] as [Type]",
                b.value.line, b.value.col, file=self._filename,
            )

        if b.called:
            value = self._register_called(b.called, value, b, scope)
        return value

    def _eval_are_binding(self, b: AreBinding, scope: Scope) -> list:
        """Evaluate an `are` binding (list sugar)."""
        elements = []
        for elem in b.elements:
            v = self._eval_node(elem, scope, exclude=b.name)
            if v is UzonUndefined:
                raise UzonRuntimeError(
                    "List element is undefined", elem.line, elem.col,
                    file=self._filename,
                )
            elements.append(v)
        self._check_list_homogeneity(elements, b)
        if elements and all(e is None for e in elements) and not b.type_annotation:
            raise UzonTypeError(
                "All-null list requires explicit type annotation via 'as [Type]'",
                b.line, b.col, file=self._filename,
            )
        if b.type_annotation:
            elements = self._apply_are_type_annotation(elements, b, scope)
        if b.called:
            elements = self._register_called(b.called, elements, b, scope)
        return elements

    def _apply_are_type_annotation(
        self, elements: list, b: AreBinding, scope: Scope
    ) -> list:
        """Apply type annotation to are-binding elements."""
        if b.type_annotation.is_list and b.type_annotation.inner:
            elem_type = b.type_annotation.inner
        elif not b.type_annotation.is_list:
            type_info = scope.get_type(b.type_annotation.name)
            if type_info is not None and type_info.get("kind") == "list":
                elem_type = b.type_annotation
            else:
                raise UzonTypeError(
                    f"'are' binding produces a list — type annotation must be a list type"
                    f" like [{b.type_annotation.name}], not bare {b.type_annotation.name}",
                    b.type_annotation.line, b.type_annotation.col,
                    file=self._filename,
                )
        else:
            elem_type = b.type_annotation
        for i, elem in enumerate(elements):
            if elem is not None:
                self._check_type_assertion(elem, elem_type, b, scope)
                elements[i] = self._wrap_typed(elem, elem_type)
        return elements

    # ── helpers ──────────────────────────────────────────────────────────

    # ── node evaluation ────────────────────────────────────────────────

    def _eval_node(self, node: Node, scope: Scope, exclude: str | None = None) -> Any:
        """§5: Main dispatch — evaluate an AST node to a Python value."""
        # §3.3: Literals
        if isinstance(node, IntegerLiteral):
            val = int(node.value, 0)
            if not (I64_MIN <= val <= I64_MAX):
                raise UzonRuntimeError(
                    f"Integer literal {val} overflows i64 "
                    f"(range {I64_MIN}..{I64_MAX})",
                    node.line, node.col, file=self._filename,
                )
            return UzonInt(val, "i64", adoptable=True)

        if isinstance(node, FloatLiteral):
            return UzonFloat(float(node.value), "f64", adoptable=True)

        if isinstance(node, BoolLiteral):
            return node.value

        if isinstance(node, StringLiteral):
            return self._eval_string(node, scope, exclude)

        if isinstance(node, NullLiteral):
            return None

        if isinstance(node, UndefinedLiteral):
            return UzonUndefined

        if isinstance(node, InfLiteral):
            return UzonFloat(float("inf"), "f64", adoptable=True)

        if isinstance(node, NanLiteral):
            return UzonFloat(float("nan"), "f64", adoptable=True)

        if isinstance(node, EnvRef):
            raise UzonTypeError(
                "'env' must be followed by .NAME", node.line, node.col,
                file=self._filename,
            )

        # §5.12: Member access (env.X, struct.field)
        if isinstance(node, MemberAccess):
            return self._eval_member_access(node, scope, exclude)

        # §5.1: Grouping
        if isinstance(node, Grouping):
            return self._eval_node(node.expr, scope, exclude)

        # §3.2: Struct literal
        if isinstance(node, StructLiteral):
            return self._eval_struct_literal(node, scope)

        # §3.2.1: with — copy-and-update
        if isinstance(node, StructOverride):
            return self._eval_struct_override(node, scope, exclude)

        # §3.2.2: plus — copy, override, and add
        if isinstance(node, StructExtension):
            return self._eval_struct_extension(node, scope, exclude)

        # §3.4: List literal
        if isinstance(node, ListLiteral):
            elements = []
            for e in node.elements:
                v = self._eval_node(e, scope, exclude)
                if v is UzonUndefined:
                    raise UzonRuntimeError(
                        "List element is undefined", e.line, e.col,
                        file=self._filename,
                    )
                elements.append(v)
            self._check_list_homogeneity(elements, node)
            return elements

        # §3.4.1: Tuple literal
        if isinstance(node, TupleLiteral):
            elements = []
            for e in node.elements:
                v = self._eval_node(e, scope, exclude)
                if v is UzonUndefined:
                    raise UzonRuntimeError(
                        "Tuple element is undefined", e.line, e.col,
                        file=self._filename,
                    )
                elements.append(v)
            return tuple(elements)

        # §5.2–§5.8: Binary operators
        if isinstance(node, BinaryOp):
            return self._eval_binary(node, scope, exclude)

        # §5.3: Unary operators
        if isinstance(node, UnaryOp):
            return self._eval_unary(node, scope, exclude)

        # §5.7: or else
        if isinstance(node, OrElse):
            left = self._eval_node(node.left, scope, exclude)
            if left is UzonUndefined:
                return self._eval_node(node.right, scope, exclude)
            right_spec = self._speculative_eval(node.right, scope, exclude)
            if right_spec is not SPECULATIVE_FAILED:
                self._check_branch_type_compat([left, right_spec], node)
            return left

        # §6: Type annotation (as)
        if isinstance(node, TypeAnnotation):
            return self._eval_type_annotation(node, scope, exclude)

        # §6.3: Type conversion (to)
        if isinstance(node, Conversion):
            return self._eval_conversion(node, scope, exclude)

        # §7: File import
        if isinstance(node, StructImport):
            return self._eval_struct_import(node)

        # §5.9: if/then/else
        if isinstance(node, IfExpr):
            return self._eval_if(node, scope, exclude)

        # §5.10: case/when
        if isinstance(node, CaseExpr):
            return self._eval_case(node, scope, exclude)

        # §5.14: Field extraction (is of)
        if isinstance(node, FieldExtraction):
            return self._eval_field_extraction(node, scope, exclude)

        # §3.5: Enum
        if isinstance(node, FromEnum):
            return self._eval_from_enum(node, scope, exclude)

        # §3.6: Union
        if isinstance(node, FromUnion):
            return self._eval_from_union(node, scope, exclude)

        # §3.7: Tagged union
        if isinstance(node, NamedVariant):
            return self._eval_named_variant(node, scope, exclude)

        # §3.8: Function expression
        if isinstance(node, FunctionExpr):
            return self._eval_function_expr(node, scope, exclude)

        # §5.15: Function call
        if isinstance(node, FunctionCall):
            return self._eval_function_call(node, scope, exclude)

        # §5.12: Identifier lookup via lexical scope chain
        if isinstance(node, Identifier):
            value = scope.get(node.name, exclude=exclude)
            return value

        raise UzonRuntimeError(
            f"Evaluation not yet implemented for {type(node).__name__}",
            node.line, node.col, file=self._filename,
        )
