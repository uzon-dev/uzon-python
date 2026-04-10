# SPDX-FileCopyrightText: © 2026 Suho Kang
# SPDX-License-Identifier: MIT
"""UZON evaluator core — walks AST and produces Python values.

Implements the main evaluation loop: binding resolution, dependency
graph construction, topological sort, and node dispatch (§3, §5).
Delegates to mixin classes for type system, operators, control flow,
struct operations, functions, and standard library.
"""

from __future__ import annotations

from typing import Any

from ..ast_nodes import (
    AreBinding, BinaryOp, Binding, BoolLiteral, CaseExpr, Conversion,
    Document, EnvRef, FieldExtraction, FloatLiteral, FromEnum, FromUnion,
    FunctionCall, FunctionExpr, Grouping, Identifier, IfExpr, InfLiteral,
    IntegerLiteral, ListLiteral, MemberAccess, NamedVariant, NanLiteral,
    Node, NullLiteral, OrElse, SelfRef, StringLiteral, StructImport,
    StructLiteral, StructExtension, StructOverride, TupleLiteral,
    TypeAnnotation, UnaryOp, UndefinedLiteral,
)
from ..errors import UzonCircularError, UzonRuntimeError, UzonSyntaxError, UzonTypeError
from ..scope import Scope
from ..types import UzonFloat, UzonInt, UzonUndefined
from ._constants import I64_MIN, I64_MAX, SPECULATIVE_FAILED
from ._control import ControlMixin
from ._functions import FunctionMixin
from ._operators import OperatorMixin
from ._stdlib import StdlibMixin
from ._structs import StructMixin
from ._types import TypeMixin


class Evaluator(
    TypeMixin,
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
        self, bindings: list[Binding | AreBinding], scope: Scope
    ) -> None:
        """Evaluate bindings with dependency resolution and duplicate detection."""
        # §3.1: detect duplicates; self-referencing overrides are allowed
        seen: dict[str, Binding | AreBinding] = {}
        override_bindings: list[Binding | AreBinding] = []
        primary_bindings: list[Binding | AreBinding] = []
        for b in bindings:
            if b.name in seen:
                is_of = isinstance(b, Binding) and isinstance(b.value, FieldExtraction)
                if not is_of and not self._references_self(b, b.name):
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
                    self._collect_self_refs(body_b.value, body_refs)
                self._collect_self_refs(b.value.body_expr, body_refs)
                if b.name in body_refs:
                    raise UzonTypeError(
                        "Recursive function call detected — call graph must be a DAG",
                        b.line, b.col, file=self._filename,
                    )

        deps = self._build_dependencies(bindings, scope)
        order = self._topological_sort(bindings, deps)

        for b in order:
            value = self._eval_binding(b, scope)
            scope.define(b.name, value)

        for b in override_bindings:
            value = self._eval_binding(b, scope)
            scope.define(b.name, value)

    def _eval_binding(self, b: Binding | AreBinding, scope: Scope) -> Any:
        """Evaluate a single binding."""
        if isinstance(b, AreBinding):
            return self._eval_are_binding(b, scope)

        if isinstance(b.value, UndefinedLiteral):
            raise UzonRuntimeError(
                f"Cannot assign literal 'undefined' to '{b.name}'",
                b.value.line, b.value.col, file=self._filename,
            )

        value = self._eval_node(b.value, scope, exclude=b.name)

        # §6.1: Empty list requires type annotation
        if isinstance(value, list) and len(value) == 0:
            if not isinstance(b.value, (TypeAnnotation, IfExpr, CaseExpr, OrElse, FunctionCall, BinaryOp)):
                raise UzonTypeError(
                    "Empty list requires explicit type annotation: [] as [Type]",
                    b.value.line, b.value.col, file=self._filename,
                )

        # All-null list requires type annotation
        if (isinstance(value, list) and value
                and not isinstance(b.value, TypeAnnotation)
                and all(e is None for e in value)):
            raise UzonTypeError(
                "All-null list requires explicit type annotation: [...] as [Type]",
                b.value.line, b.value.col, file=self._filename,
            )

        if b.called:
            self._register_called(b.called, value, b, scope)
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
            self._register_called(b.called, elements, b, scope)
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

    @staticmethod
    def _references_self(b: Binding | AreBinding, name: str) -> bool:
        """Check if a binding's value AST references self.<name>."""
        nodes: list[Any] = []
        if isinstance(b, Binding):
            nodes.append(b.value)
        else:
            nodes.extend(b.elements)
        while nodes:
            node = nodes.pop()
            if node is None:
                continue
            if isinstance(node, MemberAccess):
                if isinstance(node.object, SelfRef) and node.member == name:
                    return True
            for attr in vars(node).values():
                if isinstance(attr, list):
                    nodes.extend(v for v in attr if hasattr(v, '__dict__'))
                elif hasattr(attr, '__dict__') and attr is not node:
                    nodes.append(attr)
        return False

    # ── dependency graph ───────────────────────────────────────────────

    def _build_dependencies(
        self,
        bindings: list[Binding | AreBinding],
        scope: Scope,
    ) -> dict[str, set[str]]:
        """Build dependency graph for topological sort."""
        names = {b.name for b in bindings}
        deps: dict[str, set[str]] = {}
        for b in bindings:
            refs: set[str] = set()
            bare_refs: set[str] = set()
            if isinstance(b, AreBinding):
                for elem in b.elements:
                    self._collect_self_refs(elem, refs)
                    self._collect_bare_refs(elem, bare_refs)
            else:
                self._collect_self_refs(b.value, refs)
                self._collect_bare_refs(b.value, bare_refs)
            refs |= bare_refs
            deps[b.name] = (refs & names) - {b.name}
        return deps

    def _collect_bare_refs(self, node: Node, refs: set[str]) -> None:
        """Collect bare Identifier references (not inside function bodies)."""
        if isinstance(node, Identifier):
            refs.add(node.name)
            return
        # §3.8: Don't descend into function bodies — only collect default refs
        if isinstance(node, FunctionExpr):
            for param in node.params:
                if param.default is not None:
                    self._collect_bare_refs(param.default, refs)
            return
        for attr in vars(node).values():
            if isinstance(attr, list):
                for item in attr:
                    if isinstance(item, Node):
                        self._collect_bare_refs(item, refs)
            elif isinstance(attr, Node):
                self._collect_bare_refs(attr, refs)

    def _collect_self_refs(self, node: Node, refs: set[str]) -> None:
        """Walk AST to find all self.name references."""
        if isinstance(node, MemberAccess):
            name = self._find_self_member(node)
            if name is not None:
                refs.add(name)
                return
            self._collect_self_refs(node.object, refs)
        elif isinstance(node, BinaryOp):
            self._collect_self_refs(node.left, refs)
            self._collect_self_refs(node.right, refs)
        elif isinstance(node, UnaryOp):
            self._collect_self_refs(node.operand, refs)
        elif isinstance(node, OrElse):
            self._collect_self_refs(node.left, refs)
            self._collect_self_refs(node.right, refs)
        elif isinstance(node, IfExpr):
            self._collect_self_refs(node.condition, refs)
            self._collect_self_refs(node.then_branch, refs)
            self._collect_self_refs(node.else_branch, refs)
        elif isinstance(node, CaseExpr):
            self._collect_self_refs(node.scrutinee, refs)
            for clause in node.when_clauses:
                self._collect_self_refs(clause.value, refs)
                self._collect_self_refs(clause.result, refs)
            self._collect_self_refs(node.else_branch, refs)
        elif isinstance(node, TypeAnnotation):
            self._collect_self_refs(node.expr, refs)
        elif isinstance(node, Conversion):
            self._collect_self_refs(node.expr, refs)
        elif isinstance(node, StructOverride):
            self._collect_self_refs(node.base, refs)
            if node.overrides:
                for field in node.overrides.fields:
                    if isinstance(field, AreBinding):
                        for elem in field.elements:
                            self._collect_self_refs(elem, refs)
                    else:
                        self._collect_self_refs(field.value, refs)
        elif isinstance(node, StructExtension):
            self._collect_self_refs(node.base, refs)
            if node.extensions:
                for field in node.extensions.fields:
                    if isinstance(field, AreBinding):
                        for elem in field.elements:
                            self._collect_self_refs(elem, refs)
                    else:
                        self._collect_self_refs(field.value, refs)
        elif isinstance(node, FromEnum):
            self._collect_self_refs(node.value, refs)
        elif isinstance(node, FromUnion):
            self._collect_self_refs(node.value, refs)
        elif isinstance(node, NamedVariant):
            self._collect_self_refs(node.value, refs)
        elif isinstance(node, FieldExtraction):
            self._collect_self_refs(node.source, refs)
        elif isinstance(node, StructLiteral):
            for field in node.fields:
                if isinstance(field, (Binding, AreBinding)):
                    if isinstance(field, AreBinding):
                        for elem in field.elements:
                            self._collect_self_refs(elem, refs)
                    else:
                        self._collect_self_refs(field.value, refs)
        elif isinstance(node, ListLiteral):
            for elem in node.elements:
                self._collect_self_refs(elem, refs)
        elif isinstance(node, TupleLiteral):
            for elem in node.elements:
                self._collect_self_refs(elem, refs)
        elif isinstance(node, Grouping):
            self._collect_self_refs(node.expr, refs)
        elif isinstance(node, FunctionExpr):
            for param in node.params:
                if param.default is not None:
                    self._collect_self_refs(param.default, refs)
            for binding in node.body_bindings:
                self._collect_self_refs(binding.value, refs)
            self._collect_self_refs(node.body_expr, refs)
        elif isinstance(node, FunctionCall):
            self._collect_self_refs(node.callee, refs)
            for arg in node.args:
                self._collect_self_refs(arg, refs)
        elif isinstance(node, StringLiteral):
            for part in node.parts:
                if isinstance(part, Node):
                    self._collect_self_refs(part, refs)

    @staticmethod
    def _find_self_member(node: MemberAccess) -> str | None:
        """If member access chain starts with self, return first member name."""
        obj = node
        while isinstance(obj, MemberAccess):
            if isinstance(obj.object, SelfRef):
                return obj.member
            obj = obj.object
        return None

    def _topological_sort(
        self,
        bindings: list[Binding | AreBinding],
        deps: dict[str, set[str]],
    ) -> list[Binding | AreBinding]:
        """Kahn's algorithm. Raises UzonCircularError on cycles."""
        by_name: dict[str, Binding | AreBinding] = {b.name: b for b in bindings}
        in_degree: dict[str, int] = {b.name: len(deps.get(b.name, set())) for b in bindings}

        reverse: dict[str, list[str]] = {b.name: [] for b in bindings}
        for name, dep_set in deps.items():
            for d in dep_set:
                if d in reverse:
                    reverse[d].append(name)

        queue = [name for name, deg in in_degree.items() if deg == 0]
        order: list[str] = []

        while queue:
            name = queue.pop(0)
            order.append(name)
            for dependent in reverse.get(name, []):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        if len(order) != len(bindings):
            remaining = [b.name for b in bindings if b.name not in set(order)]
            raise UzonCircularError(
                f"Circular dependency among: {', '.join(remaining)}",
                by_name[remaining[0]].line,
                by_name[remaining[0]].col,
                file=self._filename,
            )

        return [by_name[name] for name in order]

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

        # §5.12: self / env must be followed by .name
        if isinstance(node, SelfRef):
            raise UzonTypeError(
                "'self' must be followed by .name", node.line, node.col,
                file=self._filename,
            )

        if isinstance(node, EnvRef):
            raise UzonTypeError(
                "'env' must be followed by .NAME", node.line, node.col,
                file=self._filename,
            )

        # §5.12: Member access (self.x, env.X, struct.field)
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

        # §3.2.2: extends — copy, override, and add
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

        # §5.12: Identifier lookup
        if isinstance(node, Identifier):
            if scope.has(node.name):
                return scope.get(node.name)
            raise UzonRuntimeError(
                f"Bare identifier '{node.name}' — use self.{node.name} to reference a binding",
                node.line, node.col, file=self._filename,
            )

        raise UzonRuntimeError(
            f"Evaluation not yet implemented for {type(node).__name__}",
            node.line, node.col, file=self._filename,
        )
