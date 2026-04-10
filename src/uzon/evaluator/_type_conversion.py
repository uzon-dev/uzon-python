# SPDX-FileCopyrightText: © 2026 Suho Kang
# SPDX-License-Identifier: MIT
"""Type conversion mixin — explicit `to` conversions.

Implements §5.11 (type conversions via `to`): numeric, string, null,
and user-defined type conversions.
"""

from __future__ import annotations

import math
from typing import Any

from ..ast_nodes import Conversion, EnvRef, IntegerLiteral, MemberAccess, Node, TypeExpr
from ..errors import UzonRuntimeError, UzonTypeError
from ..scope import Scope
from ..types import (
    UzonEnum, UzonFloat, UzonInt, UzonTaggedUnion, UzonUndefined, UzonUnion,
)
from ._constants import FLOAT_TYPES, INT_TYPE_RE


class TypeConversionMixin:
    """Type conversion methods mixed into the Evaluator."""

    # ── type conversion entry point (§5.11) ───────────────────────────

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

    # ── conversion dispatch ───────────────────────────────────────────

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

    # ── null conversion ───────────────────────────────────────────────

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

    # ── non-string conversion ─────────────────────────────────────────

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

    # ── string conversion (§5.11.1) ───────────────────────────────────

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
