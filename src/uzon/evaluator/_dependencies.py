# SPDX-FileCopyrightText: © 2026 Suho Kang
# SPDX-License-Identifier: MIT
"""Dependency graph construction and topological sort.

Implements dependency resolution for binding evaluation order:
self-reference collection, bare identifier collection, and
Kahn's algorithm for topological sorting with cycle detection.
"""

from __future__ import annotations

from typing import Any

from ..ast_nodes import (
    AreBinding, BinaryOp, Binding, CaseExpr, Conversion,
    FieldExtraction, FromEnum, FromUnion, FunctionCall, FunctionExpr,
    Grouping, Identifier, IfExpr, ListLiteral, MemberAccess, NamedVariant,
    Node, OrElse, SelfRef, StringLiteral, StructExtension, StructLiteral,
    StructOverride, TupleLiteral, TypeAnnotation, UnaryOp,
)
from ..errors import UzonCircularError


class DependencyMixin:
    """Dependency graph and topological sort methods mixed into the Evaluator."""

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

    # ── dependency graph ──────────────────────────────────────────────

    def _build_dependencies(
        self,
        bindings: list[Binding | AreBinding],
        scope: 'Scope',
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

    # ── topological sort ──────────────────────────────────────────────

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
