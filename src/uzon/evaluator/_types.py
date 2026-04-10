# SPDX-FileCopyrightText: © 2026 Suho Kang
# SPDX-License-Identifier: MIT
"""Type system mixin — assertions, conversions, and type helpers.

Implements §6 (type assertions via `as`), §5.11 (conversions via `to`),
and shared type-checking utilities used across all evaluator modules.
"""

from __future__ import annotations

import math
from typing import Any

from ..ast_nodes import (
    AreBinding, Binding, Conversion, Identifier, IntegerLiteral,
    ListLiteral, MemberAccess, EnvRef, Node, TypeAnnotation, TypeExpr,
)
from ..errors import UzonRuntimeError, UzonSyntaxError, UzonTypeError
from .._format import format_float as _format_float
from ..scope import Scope
from ..types import (
    UzonBuiltinFunction, UzonEnum, UzonFloat, UzonFunction,
    UzonInt, UzonTaggedUnion, UzonUndefined, UzonUnion,
)
from ._constants import FLOAT_TYPES, I64_MIN, I64_MAX, INT_TYPE_RE, SIMPLE_TYPES


class TypeMixin:
    """Type system methods mixed into the Evaluator."""

    # ── type registration ────────────────────────────────────────────

    def _register_called(
        self, type_name: str, value: Any, binding: Binding | AreBinding, scope: Scope
    ) -> None:
        """§6.2: Register a named type via `called`."""
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
            self._called_of[id(value)] = type_name
        elif isinstance(value, list):
            type_info["kind"] = "list"
            for elem in value:
                if isinstance(elem, dict):
                    self._called_of[id(elem)] = type_name
        else:
            type_info["kind"] = "value"
        scope.define_type(type_name, type_info)

    # ── type annotation (as) ─────────────────────────────────────────

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
                return [self._wrap_typed(e, type_expr.inner) if e is not None else None
                        for e in value]
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

    # ── type conversion (to) ─────────────────────────────────────────

    def _eval_conversion(
        self, node: Conversion, scope: Scope, exclude: str | None
    ) -> Any:
        """§5.11: Evaluate `expr to Type` — explicit type conversion."""
        # Bypass i64 range check when converting to a wider integer type
        if INT_TYPE_RE.match(node.type.name) and isinstance(node.expr, IntegerLiteral):
            value = int(node.expr.value, 0)
        else:
            value = self._eval_node(node.expr, scope, exclude)

        # §5.11: undefined propagates, but conversion must be valid for source type
        if value is UzonUndefined:
            self._validate_conversion_type(node)
            return UzonUndefined

        # §5.11.0: tagged/untagged unions can only convert to string
        if isinstance(value, (UzonTaggedUnion, UzonUnion)):
            if node.type.name != "string":
                raise UzonTypeError(
                    f"Cannot convert {self._type_name(value)} to {node.type.name} — "
                    f"only 'to string' is permitted for union types",
                    node.line, node.col,
                    file=self._filename,
                )
            self._check_to_string_convertible(value, node)
            return self._value_to_string(value, node)

        return self._convert_value(value, node.type, node, scope)

    def _validate_conversion_type(self, node: Conversion) -> None:
        """Validate `to` target against known source type even when value is undefined."""
        expr = node.expr
        if isinstance(expr, MemberAccess) and isinstance(expr.object, EnvRef):
            target = node.type.name
            if target == "bool":
                raise UzonTypeError(
                    "Cannot convert string to bool",
                    node.line, node.col, file=self._filename,
                )
            if target == "null":
                raise UzonTypeError(
                    "Cannot convert string to null",
                    node.line, node.col, file=self._filename,
                )

    # ── type assertion checks ────────────────────────────────────────

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
            self._check_type_assertion_int(value, type_name, m.group(1), int(m.group(2)), node)
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
                # Bug fix: null elements skipped in list type assertion
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

    # ── type conversion implementation ───────────────────────────────

    def _convert_value(
        self, value: Any, type_expr: TypeExpr, node: Node, scope: Scope
    ) -> Any:
        """§5.11: Perform explicit type conversion."""
        if type_expr.is_list:
            raise UzonTypeError(
                "Cannot convert to list type with 'to'", node.line, node.col,
                file=self._filename,
            )
        type_name = type_expr.name
        if value is None:
            return self._convert_null(type_name, node)
        if isinstance(value, str):
            return self._convert_string(value, type_expr, node, scope)
        return self._convert_non_string(value, type_name, type_expr, node, scope)

    def _convert_null(self, type_name: str, node: Node) -> Any:
        """§5.11.0: null conversions."""
        if type_name == "string":
            return "null"
        if type_name == "null":
            return None
        raise UzonTypeError(
            f"Cannot convert null to {type_name}", node.line, node.col,
            file=self._filename,
        )

    def _convert_non_string(
        self, value: Any, type_name: str, type_expr: TypeExpr, node: Node, scope: Scope
    ) -> Any:
        """Convert a non-string, non-null value to target type."""
        m = INT_TYPE_RE.match(type_name)
        if m:
            return self._convert_to_int(value, m.group(1), int(m.group(2)), type_name, node)
        if type_name in FLOAT_TYPES:
            return self._convert_to_float(value, type_name, node)
        if type_name == "string":
            self._check_to_string_convertible(value, node)
            return self._value_to_string(value, node)
        if type_name == "bool":
            if isinstance(value, bool):
                return value
            raise UzonTypeError(
                f"Cannot convert {self._type_name(value)} to bool",
                node.line, node.col, file=self._filename,
            )
        if type_name == "null":
            raise UzonTypeError(
                "'to null' is not permitted — null cannot be a conversion target",
                node.line, node.col, file=self._filename,
            )
        # User-defined type
        type_info = self._resolve_named_type(type_expr, scope, node)
        if type_info is None:
            raise UzonTypeError(
                f"Unknown type '{type_name}'", node.line, node.col,
                file=self._filename,
            )
        raise UzonTypeError(
            f"Cannot convert {self._type_name(value)} to {type_name}",
            node.line, node.col, file=self._filename,
        )

    def _convert_to_int(
        self, value: Any, sign: str, bits: int, type_name: str, node: Node
    ) -> UzonInt:
        """Convert numeric value to integer type."""
        if isinstance(value, bool):
            raise UzonTypeError(
                f"Cannot convert bool to {type_name}", node.line, node.col,
                file=self._filename,
            )
        if isinstance(value, float):
            if math.isnan(value) or math.isinf(value):
                raise UzonRuntimeError(
                    f"Cannot convert {self._float_repr(value)} to {type_name}",
                    node.line, node.col, file=self._filename,
                )
            int_val = int(value)
            self._check_int_range(int_val, sign, bits, type_name, node, error_cls=UzonRuntimeError)
            return UzonInt(int_val, type_name)
        if isinstance(value, int):
            int_val = int(value)
            self._check_int_range(int_val, sign, bits, type_name, node, error_cls=UzonRuntimeError)
            return UzonInt(int_val, type_name)
        raise UzonTypeError(
            f"Cannot convert {self._type_name(value)} to {type_name}",
            node.line, node.col, file=self._filename,
        )

    def _convert_to_float(self, value: Any, type_name: str, node: Node) -> UzonFloat:
        """Convert numeric value to float type."""
        if isinstance(value, bool):
            raise UzonTypeError(
                f"Cannot convert bool to {type_name}", node.line, node.col,
                file=self._filename,
            )
        if isinstance(value, (int, float)):
            return UzonFloat(float(value), type_name)
        raise UzonTypeError(
            f"Cannot convert {self._type_name(value)} to {type_name}",
            node.line, node.col, file=self._filename,
        )

    def _convert_string(self, value: str, type_expr: TypeExpr, node: Node, scope: Scope) -> Any:
        """§5.11.1: Convert a string to a target type."""
        type_name = type_expr.name

        # §5.11.1: Reject strings with leading/trailing whitespace
        if type_name != "string" and value != value.strip():
            raise UzonRuntimeError(
                f"Cannot convert string {value!r} to {type_name} — no surrounding whitespace allowed",
                node.line, node.col, file=self._filename,
            )

        m = INT_TYPE_RE.match(type_name)
        if m:
            return self._convert_string_to_int(value, m.group(1), int(m.group(2)), type_name, node)
        if type_name in FLOAT_TYPES:
            return self._convert_string_to_float(value, type_name, node)
        if type_name == "string":
            return value
        if type_name == "bool":
            raise UzonTypeError(
                "Cannot convert string to bool", node.line, node.col,
                file=self._filename,
            )
        if type_name == "null":
            raise UzonTypeError(
                "'to null' is not permitted — null cannot be a conversion target",
                node.line, node.col, file=self._filename,
            )
        # §5.11.1a: String → enum conversion
        type_info = self._resolve_named_type(type_expr, scope, node)
        if type_info is None:
            raise UzonTypeError(
                f"Unknown type '{type_name}'", node.line, node.col,
                file=self._filename,
            )
        if type_info.get("kind") == "enum":
            variants = type_info["variants"]
            if value not in variants:
                raise UzonRuntimeError(
                    f"String {value!r} does not match any variant of {type_name}",
                    node.line, node.col, file=self._filename,
                )
            return UzonEnum(value, variants, type_info["name"])
        raise UzonTypeError(
            f"Cannot convert string to {type_name}",
            node.line, node.col, file=self._filename,
        )

    def _convert_string_to_int(
        self, value: str, sign: str, bits: int, type_name: str, node: Node
    ) -> UzonInt:
        try:
            if value.startswith(('0x', '0X', '0o', '0O', '0b', '0B')):
                int_val = int(value, 0)
            else:
                int_val = int(value, 10)
        except ValueError:
            raise UzonRuntimeError(
                f"Cannot convert string {value!r} to {type_name}",
                node.line, node.col, file=self._filename,
            )
        self._check_int_range(int_val, sign, bits, type_name, node, error_cls=UzonRuntimeError)
        return UzonInt(int_val, type_name)

    def _convert_string_to_float(self, value: str, type_name: str, node: Node) -> UzonFloat:
        # §5.11.2: only exact "inf", "-inf", "nan", "-nan" are valid special values
        if value in ("inf", "-inf", "nan", "-nan"):
            return UzonFloat(float(value), type_name)
        lower = value.lower()
        if "inf" in lower or "nan" in lower:
            raise UzonRuntimeError(
                f"Cannot convert string {value!r} to {type_name} — "
                f'only "inf", "-inf", "nan", "-nan" are valid special float strings',
                node.line, node.col, file=self._filename,
            )
        try:
            return UzonFloat(float(value), type_name)
        except ValueError:
            raise UzonRuntimeError(
                f"Cannot convert string {value!r} to {type_name}",
                node.line, node.col, file=self._filename,
            )

    # ── type resolution ──────────────────────────────────────────────

    def _validate_type_name(
        self, type_expr: TypeExpr, node: Node, scope: Scope
    ) -> None:
        """Validate that a type name exists (for undefined propagation through `as`)."""
        name = type_expr.name
        if type_expr.is_list:
            if type_expr.inner:
                self._validate_type_name(type_expr.inner, node, scope)
            return
        if INT_TYPE_RE.match(name) or name in FLOAT_TYPES or name in SIMPLE_TYPES:
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

    # ── numeric range and repr ───────────────────────────────────────

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

    # ── type comparison helpers ──────────────────────────────────────

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
                if not self._same_uzon_type(first, other):
                    raise UzonTypeError(
                        f"List elements must be same type, got {self._type_name(first)} and {self._type_name(other)}",
                        node.line, node.col, file=self._filename,
                    )

    def _require_same_type(self, left: Any, right: Any, op: str, node: Node) -> None:
        if not self._same_uzon_type(left, right):
            raise UzonTypeError(
                f"'{op}' requires same type, got {self._type_name(left)} and {self._type_name(right)}",
                node.line, node.col, file=self._filename,
            )

    def _same_uzon_type(self, a: Any, b: Any) -> bool:
        """§5.2: Check if two values represent the same UZON type."""
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
        if isinstance(a, tuple) and isinstance(b, tuple):
            if len(a) != len(b):
                return False
            return all(self._same_uzon_type(x, y) for x, y in zip(a, b))
        if isinstance(a, dict) and isinstance(b, dict):
            a_type = self._called_of.get(id(a))
            b_type = self._called_of.get(id(b))
            if a_type and b_type:
                return a_type == b_type
            if set(a.keys()) != set(b.keys()):
                return False
            return all(self._same_uzon_type(a[k], b[k]) for k in a)
        if isinstance(a, list) and isinstance(b, list):
            a_rep = next((e for e in a if e is not None), None)
            b_rep = next((e for e in b if e is not None), None)
            if a_rep is None or b_rep is None:
                return True
            return self._same_uzon_type(a_rep, b_rep)
        if isinstance(a, UzonEnum) and isinstance(b, UzonEnum):
            if a.type_name and b.type_name:
                return a.type_name == b.type_name
            return True
        if isinstance(a, UzonFunction) and isinstance(b, UzonFunction):
            if a.type_name and b.type_name:
                return a.type_name == b.type_name
            return a.signature() == b.signature()
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
