# SPDX-FileCopyrightText: © 2026 Suho Kang
# SPDX-License-Identifier: MIT
"""Type assertion checks and shared type helpers.

Implements §6.1 assertion checks (`as` type validation), §6.3 named type
resolution, and shared type-checking utilities used across all evaluator modules.
"""

from __future__ import annotations

import math
from typing import Any

from ..ast_nodes import Node, TypeExpr
from ..errors import UzonRuntimeError, UzonTypeError
from ..scope import Scope
from ..types import (
    UzonBuiltinFunction, UzonEnum, UzonFloat, UzonFunction,
    UzonInt, UzonStruct, UzonTaggedUnion, UzonUndefined, UzonUnion,
)
from ._constants import FLOAT_TYPES, INT_TYPE_RE, SIMPLE_TYPES


class TypeChecksMixin:
    """Type assertion checks and helpers mixed into the Evaluator."""

    # ── type assertion checks (§6.1) ──────────────────────────────────

    def _check_type_assertion(
        self, value: Any, type_expr: TypeExpr, node: Node, scope: Scope
    ) -> None:
        """§6.1: Validate that value conforms to the asserted type."""
        if type_expr.is_list:
            self._check_type_assertion_list(value, type_expr, node, scope)
            return
        if type_expr.is_tuple:
            self._check_type_assertion_tuple(value, type_expr, node, scope)
            return
        type_name = type_expr.name
        m = INT_TYPE_RE.match(type_name)
        if m:
            bits = int(m.group(2))
            if bits > 65535:
                raise UzonTypeError(
                    f"Integer type width {bits} exceeds maximum (65535)",
                    node.line, node.col, file=self._filename,
                )
            self._check_type_assertion_int(value, type_name, m.group(1), bits, node)
            return
        if type_name in FLOAT_TYPES:
            self._check_type_assertion_float(value, type_name, node)
            return
        if type_name in ("string", "bool", "null"):
            self._check_type_assertion_simple(value, type_name, node)
            return
        self._check_type_assertion_named(value, type_expr, node, scope)

    def _check_type_assertion_list(
        self, value: Any, type_expr: TypeExpr, node: Node, scope: Scope
    ) -> None:
        if not isinstance(value, list):
            raise UzonTypeError(
                f"'as [{type_expr.inner.name if type_expr.inner else '?'}]' requires a list, "
                f"got {self._type_name(value)}",
                node.line, node.col, file=self._filename,
            )
        if type_expr.inner and value:
            for elem in value:
                if elem is None:
                    continue
                self._check_type_assertion(elem, type_expr.inner, node, scope)

    def _check_type_assertion_tuple(
        self, value: Any, type_expr: TypeExpr, node: Node, scope: Scope
    ) -> None:
        if not isinstance(value, tuple):
            raise UzonTypeError(
                f"'as {type_expr.name}' requires a tuple, got {self._type_name(value)}",
                node.line, node.col, file=self._filename,
            )
        if len(value) != len(type_expr.elements):
            raise UzonTypeError(
                f"'as {type_expr.name}' requires {len(type_expr.elements)} elements, "
                f"got {len(value)}",
                node.line, node.col, file=self._filename,
            )
        for elem, elem_type in zip(value, type_expr.elements):
            self._check_type_assertion(elem, elem_type, node, scope)

    def _check_type_assertion_int(
        self, value: Any, type_name: str, sign: str, bits: int, node: Node
    ) -> None:
        if isinstance(value, bool):
            raise UzonTypeError(
                f"'as {type_name}' requires an integer, got bool",
                node.line, node.col, file=self._filename,
            )
        if isinstance(value, UzonInt):
            if value.type_name != type_name:
                if not value.adoptable:
                    raise UzonTypeError(
                        f"'as {type_name}' type mismatch: value is {value.type_name}",
                        node.line, node.col, file=self._filename,
                    )
                self._check_int_range(int(value), sign, bits, type_name, node)
            return
        if not isinstance(value, int):
            raise UzonTypeError(
                f"'as {type_name}' requires an integer, got {self._type_name(value)}",
                node.line, node.col, file=self._filename,
            )
        self._check_int_range(value, sign, bits, type_name, node)

    def _check_type_assertion_float(self, value: Any, type_name: str, node: Node) -> None:
        if isinstance(value, UzonFloat):
            if value.type_name != type_name and not value.adoptable:
                raise UzonTypeError(
                    f"'as {type_name}' type mismatch: value is {value.type_name}",
                    node.line, node.col, file=self._filename,
                )
            return
        if not isinstance(value, float):
            raise UzonTypeError(
                f"'as {type_name}' requires a float, got {self._type_name(value)}",
                node.line, node.col, file=self._filename,
            )

    def _check_type_assertion_simple(self, value: Any, type_name: str, node: Node) -> None:
        if type_name == "string":
            if not isinstance(value, str):
                raise UzonTypeError(
                    f"'as string' requires a string, got {self._type_name(value)}",
                    node.line, node.col, file=self._filename,
                )
        elif type_name == "bool":
            if not isinstance(value, bool):
                raise UzonTypeError(
                    f"'as bool' requires a bool, got {self._type_name(value)}",
                    node.line, node.col, file=self._filename,
                )
        elif type_name == "null":
            if value is not None:
                raise UzonTypeError(
                    f"'as null' requires null, got {self._type_name(value)}",
                    node.line, node.col, file=self._filename,
                )

    def _check_type_assertion_named(
        self, value: Any, type_expr: TypeExpr, node: Node, scope: Scope
    ) -> None:
        """§6.3: Check assertion against user-defined types (struct, union, function, etc.)."""
        type_name = type_expr.name
        type_info = self._resolve_named_type(type_expr, scope, node)
        if type_info is None:
            raise UzonTypeError(
                f"Unknown type '{type_name}'", node.line, node.col,
                file=self._filename,
            )
        kind = type_info.get("kind")
        if kind == "tagged_union":
            raise UzonTypeError(
                f"'as {type_name}' requires 'named' to specify the active variant",
                node.line, node.col, file=self._filename,
            )
        if kind == "struct":
            self._check_struct_assertion(value, type_name, type_info, node)
            return
        if kind == "union":
            member_types = type_info.get("types", [])
            if not self._value_matches_union(value, member_types):
                raise UzonTypeError(
                    f"'as {type_name}' requires value to match one of: {', '.join(member_types)}",
                    node.line, node.col, file=self._filename,
                )
            return
        if kind == "function":
            self._check_function_assertion(value, type_name, type_info, node)

    def _check_struct_assertion(
        self, value: Any, type_name: str, type_info: dict, node: Node
    ) -> None:
        """§6.3: `as StructType` — check field name and type conformance."""
        if not isinstance(value, dict):
            raise UzonTypeError(
                f"'as {type_name}' requires a struct, got {self._type_name(value)}",
                node.line, node.col, file=self._filename,
            )
        # §3.2.1 rule 5 / §6.3: nominal identity — a value already named
        # as a different struct type cannot be re-asserted as another.
        existing_name = None
        if isinstance(value, UzonStruct) and value.type_name:
            existing_name = value.type_name
        else:
            existing_name = self._called_of.get(id(value))
        if existing_name and existing_name != type_name:
            raise UzonTypeError(
                f"'as {type_name}' type mismatch: value is {existing_name} "
                "(nominal identity — separately named struct types are incompatible)",
                node.line, node.col, file=self._filename,
            )
        expected_fields = type_info["fields"]
        actual_fields = set(value.keys())
        if expected_fields != actual_fields:
            extra = actual_fields - expected_fields
            missing = expected_fields - actual_fields
            parts = []
            if extra:
                parts.append(f"extra: {', '.join(sorted(extra))}")
            if missing:
                parts.append(f"missing: {', '.join(sorted(missing))}")
            raise UzonTypeError(
                f"'as {type_name}' field mismatch — {'; '.join(parts)}",
                node.line, node.col, file=self._filename,
            )
        original = type_info.get("field_values")
        if original:
            for key in expected_fields:
                orig_val = original[key]
                new_val = value[key]
                if orig_val is not None and new_val is not None:
                    if not self._same_uzon_type(orig_val, new_val):
                        raise UzonTypeError(
                            f"'as {type_name}' field '{key}' type mismatch: "
                            f"expected {self._type_name(orig_val)}, got {self._type_name(new_val)}",
                            node.line, node.col, file=self._filename,
                        )
        if isinstance(value, UzonStruct):
            value.type_name = type_name
        self._called_of[id(value)] = type_name

    def _check_function_assertion(
        self, value: Any, type_name: str, type_info: dict, node: Node
    ) -> None:
        """§3.8: `as FunctionType` — structural conformance check."""
        if not isinstance(value, UzonFunction):
            raise UzonTypeError(
                f"'as {type_name}' requires a function, got {self._type_name(value)}",
                node.line, node.col, file=self._filename,
            )
        if value.type_name and value.type_name != type_name:
            raise UzonTypeError(
                f"'as {type_name}' type mismatch: function is {value.type_name} "
                f"(nominal identity — separately named types are incompatible)",
                node.line, node.col, file=self._filename,
            )
        expected_sig = type_info["signature"]
        actual_sig = value.signature()
        if expected_sig != actual_sig:
            raise UzonTypeError(
                f"'as {type_name}' signature mismatch: expected "
                f"({', '.join(expected_sig[0])}) -> {expected_sig[1]}, got "
                f"({', '.join(actual_sig[0])}) -> {actual_sig[1]}",
                node.line, node.col, file=self._filename,
            )

    # ── type resolution ───────────────────────────────────────────────

    def _validate_type_name(
        self, type_expr: TypeExpr, node: Node, scope: Scope
    ) -> None:
        """Validate that a type name exists (for undefined propagation through `as`)."""
        name = type_expr.name
        if type_expr.is_list:
            if type_expr.inner:
                self._validate_type_name(type_expr.inner, node, scope)
            return
        if type_expr.is_tuple:
            for elem in type_expr.elements:
                self._validate_type_name(elem, node, scope)
            return
        m = INT_TYPE_RE.match(name)
        if m:
            if int(m.group(2)) > 65535:
                raise UzonTypeError(
                    f"Integer type width {m.group(2)} exceeds maximum (65535)",
                    node.line, node.col, file=self._filename,
                )
            return
        if name in FLOAT_TYPES or name in SIMPLE_TYPES:
            return
        if self._resolve_named_type(type_expr, scope, node) is None:
            raise UzonTypeError(
                f"Unknown type '{name}'", node.line, node.col,
                file=self._filename,
            )

    def _resolve_named_type(
        self, type_expr: TypeExpr, scope: Scope, node: Node
    ) -> dict[str, Any] | None:
        """§6.2: Resolve a type name (possibly dotted path) from scope."""
        if type_expr.path:
            current_scope = scope
            for segment in type_expr.path[:-1]:
                binding_val = current_scope.get(segment)
                if binding_val is UzonUndefined or not isinstance(binding_val, dict):
                    return None
                child_scope = self._scope_of.get(id(binding_val))
                if child_scope is None:
                    return None
                current_scope = child_scope
            return current_scope.get_type(type_expr.path[-1])
        return scope.get_type(type_expr.name)

    def _value_matches_union(self, value: Any, member_types: list[str]) -> bool:
        """Check if a value's type matches at least one union member type."""
        for mt in member_types:
            m = INT_TYPE_RE.match(mt)
            if m:
                if isinstance(value, int) and not isinstance(value, bool):
                    return True
                continue
            if mt in FLOAT_TYPES:
                if isinstance(value, float):
                    return True
                continue
            if mt == "string" and isinstance(value, str):
                return True
            if mt == "bool" and isinstance(value, bool):
                return True
            if mt == "null" and value is None:
                return True
        return False

    # ── numeric range and repr ────────────────────────────────────────

    @staticmethod
    def _check_int_range(
        value: int, sign: str, bits: int, type_name: str, node: Node,
        error_cls: type = UzonTypeError,
    ) -> None:
        """Check that an integer value fits in the specified type range."""
        if bits == 0:
            if value != 0:
                raise error_cls(
                    f"Value {value} overflows {type_name} (only 0 fits)",
                    node.line, node.col,
                )
            return
        if sign == "u":
            lo, hi = 0, (1 << bits) - 1
        else:
            lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
        if not (lo <= value <= hi):
            raise error_cls(
                f"Value {value} overflows {type_name} (range {lo}..{hi})",
                node.line, node.col,
            )

    @staticmethod
    def _float_repr(val: float) -> str:
        if math.isnan(val):
            return "nan"
        if val == float("inf"):
            return "inf"
        if val == float("-inf"):
            return "-inf"
        return str(val)

    # ── type comparison helpers ────────────────────────────────────────

    def _require_bool(self, val: Any, op: str, node: Node) -> None:
        if val is UzonUndefined:
            raise UzonRuntimeError(
                f"Cannot use '{op}' with undefined", node.line, node.col,
                file=self._filename,
            )
        if not isinstance(val, bool):
            raise UzonTypeError(
                f"'{op}' requires bool, got {self._type_name(val)}",
                node.line, node.col, file=self._filename,
            )

    def _require_numeric(self, val: Any, op: str, node: Node) -> None:
        if isinstance(val, bool):
            raise UzonTypeError(
                f"'{op}' requires a number, got bool", node.line, node.col,
                file=self._filename,
            )
        if not isinstance(val, (int, float)):
            raise UzonTypeError(
                f"'{op}' requires a number, got {self._type_name(val)}",
                node.line, node.col, file=self._filename,
            )

    def _check_list_homogeneity(self, elements: list[Any], node: Node) -> None:
        """§3.4: Check that all list elements are the same type (null compatible)."""
        non_null = [e for e in elements if e is not None]
        if len(non_null) >= 2:
            first = non_null[0]
            for other in non_null[1:]:
                if not self._same_uzon_type(first, other, for_homogeneity=True):
                    raise UzonTypeError(
                        f"List elements must be same type, got {self._type_name(first)} and {self._type_name(other)}",
                        node.line, node.col, file=self._filename,
                    )
            # §3.4 + §3.2.1: Anonymous struct elements must share field shapes
            if isinstance(first, dict) and not (
                (isinstance(first, UzonStruct) and first.type_name)
                or self._called_of.get(id(first))
            ):
                self._check_struct_list_shape(non_null, node)

    def _check_struct_list_shape(self, elements: list[dict], node: Node) -> None:
        """§3.4: Anonymous struct list elements must have same keys and value type categories."""
        first = elements[0]
        first_keys = first.keys()
        for other in elements[1:]:
            if other.keys() != first_keys:
                raise UzonTypeError(
                    "List elements must be same type, got struct and struct",
                    node.line, node.col, file=self._filename,
                )
            for k in first_keys:
                va, vb = first[k], other[k]
                if va is not None and vb is not None:
                    if self._type_category(va) != self._type_category(vb):
                        raise UzonTypeError(
                            "List elements must be same type, got struct and struct",
                            node.line, node.col, file=self._filename,
                        )

    @staticmethod
    def _type_category(value: Any) -> str:
        """Return coarse type category for struct field value comparison."""
        if isinstance(value, bool):
            return "bool"
        if isinstance(value, int):
            return "int"
        if isinstance(value, float):
            return "float"
        if isinstance(value, str):
            return "string"
        if isinstance(value, dict):
            return "struct"
        if isinstance(value, list):
            return "list"
        if isinstance(value, tuple):
            return "tuple"
        return type(value).__name__

    def _require_same_type(self, left: Any, right: Any, op: str, node: Node) -> None:
        if not self._same_uzon_type(left, right):
            raise UzonTypeError(
                f"'{op}' requires same type, got {self._type_name(left)} and {self._type_name(right)}",
                node.line, node.col, file=self._filename,
            )

    def _same_uzon_type(self, a: Any, b: Any, *, for_homogeneity: bool = False) -> bool:
        """§5.2: Check if two values represent the same UZON type.

        Args:
            for_homogeneity: When True (list homogeneity checks), anonymous structs
                are always compatible. When False (comparison operators), anonymous
                structs must have matching field shapes.
        """
        if isinstance(a, bool) or isinstance(b, bool):
            return isinstance(a, bool) and isinstance(b, bool)
        if isinstance(a, int) and isinstance(b, int):
            if isinstance(a, UzonInt) and isinstance(b, UzonInt):
                if a.adoptable or b.adoptable:
                    return True
                return a.type_name == b.type_name
            return True
        if isinstance(a, float) and isinstance(b, float):
            if isinstance(a, UzonFloat) and isinstance(b, UzonFloat):
                if a.adoptable or b.adoptable:
                    return True
                return a.type_name == b.type_name
            return True
        # §5 line 1220: Cross-category — adoptable integer literal → float type
        if isinstance(a, UzonInt) and a.adoptable and isinstance(b, float):
            return True
        if isinstance(b, UzonInt) and b.adoptable and isinstance(a, float):
            return True
        if isinstance(a, tuple) and isinstance(b, tuple):
            if len(a) != len(b):
                return False
            return all(x is None or y is None or self._same_uzon_type(x, y, for_homogeneity=for_homogeneity)
                       for x, y in zip(a, b))
        if isinstance(a, dict) and isinstance(b, dict):
            a_type = a.type_name if isinstance(a, UzonStruct) else self._called_of.get(id(a))
            b_type = b.type_name if isinstance(b, UzonStruct) else self._called_of.get(id(b))
            if a_type and b_type:
                return a_type == b_type
            if a_type or b_type:
                return False
            if for_homogeneity:
                return True
            if a.keys() != b.keys():
                return False
            return all(
                va is None or vb is None or self._same_uzon_type(va, vb)
                for va, vb in ((a[k], b[k]) for k in a)
            )
        if isinstance(a, list) and isinstance(b, list):
            a_rep = next((e for e in a if e is not None), None)
            b_rep = next((e for e in b if e is not None), None)
            if a_rep is None or b_rep is None:
                return True
            return self._same_uzon_type(a_rep, b_rep, for_homogeneity=for_homogeneity)
        if isinstance(a, UzonEnum) and isinstance(b, UzonEnum):
            if a.type_name and b.type_name:
                return a.type_name == b.type_name
            return True
        if isinstance(a, UzonFunction) and isinstance(b, UzonFunction):
            if a.type_name and b.type_name:
                return a.type_name == b.type_name
            return a.signature() == b.signature()
        if isinstance(a, UzonUnion) and isinstance(b, UzonUnion):
            if a.type_name and b.type_name:
                return a.type_name == b.type_name
            if a.type_name or b.type_name:
                return False
            return set(a.types) == set(b.types)
        return type(a) is type(b)

    def _type_name(self, val: Any) -> str:
        if val is UzonUndefined:
            return "undefined"
        if val is None:
            return "null"
        if isinstance(val, UzonEnum):
            return f"enum({val.type_name or 'anonymous'})"
        if isinstance(val, UzonTaggedUnion):
            return f"tagged_union({val.type_name or 'anonymous'})"
        if isinstance(val, UzonUnion):
            return f"union({val.type_name or 'anonymous'})"
        if isinstance(val, bool):
            return "bool"
        if isinstance(val, UzonInt):
            return "integer" if val.adoptable else val.type_name
        if isinstance(val, int):
            return "integer"
        if isinstance(val, UzonFloat):
            return "float" if val.adoptable else val.type_name
        if isinstance(val, float):
            return "float"
        if isinstance(val, str):
            return "string"
        if isinstance(val, dict):
            if isinstance(val, UzonStruct) and val.type_name:
                return val.type_name
            called = self._called_of.get(id(val))
            return called if called else "struct"
        if isinstance(val, list):
            return "list"
        if isinstance(val, tuple):
            return "tuple"
        if isinstance(val, UzonFunction):
            return f"function({val.type_name or 'anonymous'})"
        if isinstance(val, UzonBuiltinFunction):
            return f"builtin({val.name})"
        return type(val).__name__
