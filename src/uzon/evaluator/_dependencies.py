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
    AreBinding, Binding, FunctionExpr,
    Identifier, Node,
)
from ..errors import UzonCircularError


class DependencyMixin:
    """Dependency graph and topological sort methods mixed into the Evaluator."""

    @staticmethod
    def _references_self(b: Binding | AreBinding, name: str) -> bool:
        """Check if a binding's value AST references <name> as a bare identifier."""
        nodes: list[Any] = []
        if isinstance(b, Binding):
            nodes.append(b.value)
        else:
            nodes.extend(b.elements)
        while nodes:
            node = nodes.pop()
            if node is None:
                continue
            if isinstance(node, Identifier) and node.name == name:
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
    ) -> dict[str, set[str]]:
        """Build dependency graph for topological sort."""
        names = {b.name for b in bindings}
        deps: dict[str, set[str]] = {}
        for b in bindings:
            refs: set[str] = set()
            if isinstance(b, AreBinding):
                for elem in b.elements:
                    self._collect_bare_refs(elem, refs)
            else:
                self._collect_bare_refs(b.value, refs)
            deps[b.name] = (refs & names) - {b.name}
        return deps

    def _collect_bare_refs(self, node: Node, refs: set[str]) -> None:
        """Collect bare Identifier references, including inside function bodies."""
        if isinstance(node, Identifier):
            refs.add(node.name)
            return
        if isinstance(node, FunctionExpr):
            # Collect references from default parameter values
            for param in node.params:
                if param.default is not None:
                    self._collect_bare_refs(param.default, refs)
            # Collect references from function body for mutual recursion detection
            for binding in node.body_bindings:
                self._collect_bare_refs(binding.value, refs)
            self._collect_bare_refs(node.body_expr, refs)
            return
        for attr in vars(node).values():
            if isinstance(attr, list):
                for item in attr:
                    if isinstance(item, Node):
                        self._collect_bare_refs(item, refs)
            elif isinstance(attr, Node):
                self._collect_bare_refs(attr, refs)

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
