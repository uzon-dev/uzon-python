# SPDX-FileCopyrightText: © 2026 Suho Kang
# SPDX-License-Identifier: MIT
"""Type annotation and registration mixin.

Implements §6.1 (type annotation via `as`) and §6.2 (named types via `called`).
"""

from __future__ import annotations

from typing import Any

from ..ast_nodes import (
    AreBinding, Binding, Identifier, IntegerLiteral, ListLiteral,
    Node, TypeAnnotation, TypeExpr,
)
from ..errors import UzonSyntaxError, UzonTypeError
from ..scope import Scope
from ..types import (
    UzonEnum, UzonFloat, UzonFunction, UzonInt, UzonStruct,
    UzonTaggedUnion, UzonTypedList, UzonUndefined, UzonUnion,
)
from ._constants import FLOAT_TYPES, INT_TYPE_RE


class TypeAnnotationMixin:
    """Type annotation and registration methods mixed into the Evaluator."""

    # ── type registration (§6.2) ──────────────────────────────────────

    def _register_called(
        self, type_name: str, value: Any, binding: Binding | AreBinding, scope: Scope
    ) -> Any:
        """§6.2: Register a named type via `called`.

        Returns the (possibly wrapped) value — dicts become UzonStruct.
        """
        if scope.has_own_type(type_name):
            raise UzonSyntaxError(
                f"Duplicate type name '{type_name}' in same scope",
                binding.line, binding.col,
                file=self._filename,
            )
        type_info: dict[str, Any] = {"name": type_name, "binding": binding.name}
        if isinstance(value, UzonEnum):
            type_info["kind"] = "enum"
            type_info["variants"] = value.variants
            value.type_name = type_name
        elif isinstance(value, UzonUnion):
            type_info["kind"] = "union"
            type_info["types"] = value.types
            value.type_name = type_name
        elif isinstance(value, UzonTaggedUnion):
            type_info["kind"] = "tagged_union"
            type_info["variants"] = value.variants
            value.type_name = type_name
        elif isinstance(value, UzonFunction):
            type_info["kind"] = "function"
            type_info["signature"] = value.signature()
            value.type_name = type_name
        elif isinstance(value, dict):
            type_info["kind"] = "struct"
            type_info["fields"] = set(value.keys())
            type_info["field_values"] = dict(value)
            old_id = id(value)
            if not isinstance(value, UzonStruct):
                value = UzonStruct(value, type_name)
            else:
                value.type_name = type_name
            # Migrate _scope_of from old dict id to new UzonStruct id
            if old_id != id(value) and old_id in self._scope_of:
                self._scope_of[id(value)] = self._scope_of.pop(old_id)
            self._called_of[id(value)] = type_name
        elif isinstance(value, list):
            type_info["kind"] = "list"
            for i, elem in enumerate(value):
                old_elem_id = id(elem)
                if isinstance(elem, dict) and not isinstance(elem, UzonStruct):
                    value[i] = UzonStruct(elem, type_name)
                    if old_elem_id in self._scope_of:
                        self._scope_of[id(value[i])] = self._scope_of.pop(old_elem_id)
                    self._called_of[id(value[i])] = type_name
                elif isinstance(elem, UzonStruct):
                    elem.type_name = type_name
                    self._called_of[id(elem)] = type_name
                elif isinstance(elem, dict):
                    self._called_of[id(elem)] = type_name
        else:
            type_info["kind"] = "value"
        scope.define_type(type_name, type_info)
        return value

    # ── type annotation (as) — §6.1 ──────────────────────────────────

    def _eval_type_annotation(
        self, node: TypeAnnotation, scope: Scope, exclude: str | None
    ) -> Any:
        """§6.1: Evaluate `expr as Type` — assertion, not conversion."""
        # §6.3: Check for enum variant resolution
        type_info = self._resolve_named_type(node.type, scope, node)
        if type_info and type_info.get("kind") == "enum" and isinstance(node.expr, Identifier):
            variant_name = node.expr.name
            variants = type_info["variants"]
            if variant_name not in variants:
                raise UzonTypeError(
                    f"'{variant_name}' is not a variant of {type_info['name']}",
                    node.line, node.col,
                    file=self._filename,
                )
            return UzonEnum(variant_name, variants, type_info["name"])

        # List of enum variants: [id, id, ...] as [EnumType]
        if node.type.is_list and node.type.inner and isinstance(node.expr, ListLiteral):
            inner_type_info = self._resolve_named_type(node.type.inner, scope, node)
            if inner_type_info and inner_type_info.get("kind") == "enum":
                elements = []
                for elem in node.expr.elements:
                    if isinstance(elem, Identifier) and elem.name in inner_type_info["variants"]:
                        elements.append(UzonEnum(
                            elem.name, inner_type_info["variants"], inner_type_info["name"]
                        ))
                    else:
                        elements.append(self._eval_node(elem, scope, exclude))
                return elements

        # Bypass i64 range check when target is a wider integer type
        if INT_TYPE_RE.match(node.type.name) and isinstance(node.expr, IntegerLiteral):
            value = int(node.expr.value, 0)
        else:
            value = self._eval_node(node.expr, scope, exclude)

        # §6.1: undefined propagates through `as`
        if value is UzonUndefined:
            self._validate_type_name(node.type, node, scope)
            return UzonUndefined

        self._check_type_assertion(value, node.type, node, scope)
        return self._wrap_typed(value, node.type)

    def _wrap_typed(self, value: Any, type_expr: TypeExpr) -> Any:
        """Wrap a value in a typed wrapper after assertion/conversion."""
        if value is None or isinstance(value, bool):
            return value
        if type_expr.is_list:
            if isinstance(value, list) and type_expr.inner:
                wrapped = [self._wrap_typed(e, type_expr.inner) if e is not None else None
                           for e in value]
                return UzonTypedList(wrapped, type_expr.inner.name)
            return value
        name = type_expr.name
        if INT_TYPE_RE.match(name) and isinstance(value, int) and not isinstance(value, bool):
            if isinstance(value, UzonInt) and value.type_name == name and not value.adoptable:
                return value
            return UzonInt(int(value), name)
        if name in FLOAT_TYPES and isinstance(value, float):
            if isinstance(value, UzonFloat) and value.type_name == name and not value.adoptable:
                return value
            return UzonFloat(float(value), name)
        return value
