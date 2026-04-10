# SPDX-FileCopyrightText: © 2026 Suho Kang
# SPDX-License-Identifier: MIT
"""Operator evaluation mixin — arithmetic, comparison, equality, collection ops.

Implements §5.2 (equality), §5.3 (arithmetic), §5.4 (comparison),
§5.6 (short-circuit logic), §5.8 (collection operators).
"""

from __future__ import annotations

import math
from typing import Any

from ..ast_nodes import BinaryOp, Identifier, Node, UnaryOp
from ..errors import UzonRuntimeError, UzonTypeError
from ..types import (
    UzonBuiltinFunction, UzonEnum, UzonFloat, UzonFunction,
    UzonInt, UzonTaggedUnion, UzonUndefined, UzonUnion,
)
from ._constants import I64_MIN, I64_MAX, INT_TYPE_RE, SPECULATIVE_FAILED


class OperatorMixin:
    """Operator evaluation methods mixed into the Evaluator."""

    @staticmethod
    def _unwrap_transparent(val: Any) -> Any:
        """§3.7.1: Unwrap tagged union to inner value for transparent evaluation."""
        if isinstance(val, UzonTaggedUnion):
            return val.value
        return val

    # ── unary ────────────────────────────────────────────────────────

    def _eval_unary(self, node: UnaryOp, scope, exclude: str | None) -> Any:
        operand = self._eval_node(node.operand, scope, exclude)
        if node.op == "-":
            return self._eval_unary_minus(operand, node)
        if node.op == "not":
            return self._eval_unary_not(operand, node)
        raise UzonRuntimeError(
            f"Unknown unary operator: {node.op}", node.line, node.col,
            file=self._filename,
        )

    def _eval_unary_minus(self, operand: Any, node: Node) -> Any:
        if operand is UzonUndefined:
            raise UzonRuntimeError(
                "Cannot negate undefined", node.line, node.col,
                file=self._filename,
            )
        operand = self._unwrap_transparent(operand)
        if not isinstance(operand, (int, float)) or isinstance(operand, bool):
            raise UzonTypeError(
                f"Cannot negate {self._type_name(operand)}", node.line, node.col,
                file=self._filename,
            )
        if isinstance(operand, UzonInt):
            raw = -int(operand)
            return self._typed_int_result(raw, operand.type_name, node, adoptable=operand.adoptable)
        if isinstance(operand, UzonFloat):
            raw = -float(operand)
            return self._typed_float_result(raw, operand.type_name, adoptable=operand.adoptable)
        result = -operand
        if isinstance(result, int):
            if not (I64_MIN <= result <= I64_MAX):
                raise UzonRuntimeError(
                    f"Integer negation overflow: -{operand} exceeds i64 range "
                    f"({I64_MIN}..{I64_MAX})",
                    node.line, node.col, file=self._filename,
                )
        return result

    def _eval_unary_not(self, operand: Any, node: Node) -> bool:
        if operand is UzonUndefined:
            raise UzonRuntimeError(
                "Cannot apply 'not' to undefined", node.line, node.col,
                file=self._filename,
            )
        operand = self._unwrap_transparent(operand)
        if not isinstance(operand, bool):
            raise UzonTypeError(
                f"'not' requires bool, got {self._type_name(operand)}",
                node.line, node.col, file=self._filename,
            )
        return not operand

    # ── binary dispatch ──────────────────────────────────────────────

    def _eval_binary(self, node: BinaryOp, scope, exclude: str | None) -> Any:
        op = node.op

        # §5.6: Short-circuit — and, or
        if op == "and":
            return self._eval_and(node, scope, exclude)
        if op == "or":
            return self._eval_or(node, scope, exclude)

        # §3.7.2: is named / is not named
        if op in ("is named", "is not named"):
            left = self._eval_node(node.left, scope, exclude)
            return self._eval_is_named(op, left, node)

        # §5.8: `in` — evaluate right first for enum variant inference
        if op == "in":
            return self._eval_in_op(node, scope, exclude)

        # Evaluate both sides
        left = self._eval_node(node.left, scope, exclude)
        right = self._eval_node(node.right, scope, exclude)

        # §5.2: Equality — allows undefined and null
        if op == "is":
            return self._eval_is(left, right, node)
        if op == "is not":
            return not self._eval_is(left, right, node)

        # Everything else errors on undefined
        if left is UzonUndefined or right is UzonUndefined:
            raise UzonRuntimeError(
                f"Cannot use '{op}' with undefined — use 'or else' to provide a fallback",
                node.line, node.col, file=self._filename,
            )

        # §3.7.1: Transparent tagged union unwrapping
        both_tu = isinstance(left, UzonTaggedUnion) and isinstance(right, UzonTaggedUnion)

        # §5.3: Arithmetic
        if op in ("+", "-", "*", "/", "%", "^"):
            return self._eval_arithmetic(
                op, self._unwrap_transparent(left),
                self._unwrap_transparent(right), node)

        # §5.4: Comparison
        if op in ("<", "<=", ">", ">="):
            if both_tu:
                return self._eval_comparison(op, left, right, node)
            return self._eval_comparison(
                op, self._unwrap_transparent(left),
                self._unwrap_transparent(right), node)

        # §5.8: Concatenation and repetition
        if op == "++":
            return self._eval_concat(
                self._unwrap_transparent(left),
                self._unwrap_transparent(right), node)
        if op == "**":
            return self._eval_repeat(
                self._unwrap_transparent(left),
                self._unwrap_transparent(right), node)

        raise UzonRuntimeError(
            f"Unknown operator: {op}", node.line, node.col,
            file=self._filename,
        )

    # ── short-circuit logic ──────────────────────────────────────────

    def _eval_and(self, node: BinaryOp, scope, exclude: str | None) -> bool:
        left = self._unwrap_transparent(self._eval_node(node.left, scope, exclude))
        self._require_bool(left, "and", node)
        if not left:
            right_spec = self._speculative_eval(node.right, scope, exclude)
            if right_spec is not SPECULATIVE_FAILED:
                right_spec = self._unwrap_transparent(right_spec)
                self._require_bool(right_spec, "and", node)
            return False
        right = self._unwrap_transparent(self._eval_node(node.right, scope, exclude))
        self._require_bool(right, "and", node)
        return right

    def _eval_or(self, node: BinaryOp, scope, exclude: str | None) -> bool:
        left = self._unwrap_transparent(self._eval_node(node.left, scope, exclude))
        self._require_bool(left, "or", node)
        if left:
            right_spec = self._speculative_eval(node.right, scope, exclude)
            if right_spec is not SPECULATIVE_FAILED:
                right_spec = self._unwrap_transparent(right_spec)
                self._require_bool(right_spec, "or", node)
            return True
        right = self._unwrap_transparent(self._eval_node(node.right, scope, exclude))
        self._require_bool(right, "or", node)
        return right

    # ── `in` operator ────────────────────────────────────────────────

    def _eval_in_op(self, node: BinaryOp, scope, exclude: str | None) -> bool:
        """§5.8: Evaluate `in` with enum variant inference."""
        right = self._eval_node(node.right, scope, exclude)
        if right is UzonUndefined:
            raise UzonRuntimeError(
                "Cannot use 'in' with undefined — use 'or else' to provide a fallback",
                node.line, node.col, file=self._filename,
            )
        if (isinstance(node.left, Identifier) and isinstance(right, list) and right
                and isinstance(right[0], UzonEnum)):
            variant_name = node.left.name
            enum_val = right[0]
            if variant_name in enum_val.variants:
                left = UzonEnum(variant_name, enum_val.variants, enum_val.type_name)
            else:
                left = self._eval_node(node.left, scope, exclude)
        else:
            left = self._eval_node(node.left, scope, exclude)
        if left is UzonUndefined:
            raise UzonRuntimeError(
                "Cannot use 'in' with undefined — use 'or else' to provide a fallback",
                node.line, node.col, file=self._filename,
            )
        return self._eval_in(left, right, node)

    # ── equality ─────────────────────────────────────────────────────

    def _eval_is(self, left: Any, right: Any, node: Node) -> bool:
        """§5.2: Equality comparison."""
        if isinstance(left, (UzonFunction, UzonBuiltinFunction)) or isinstance(right, (UzonFunction, UzonBuiltinFunction)):
            raise UzonTypeError(
                "Cannot compare function values with 'is' / 'is not'",
                node.line, node.col, file=self._filename,
            )
        if isinstance(left, UzonUnion):
            left = left.value
        if isinstance(right, UzonUnion):
            right = right.value
        if left is None or right is None:
            return left is None and right is None
        if left is UzonUndefined or right is UzonUndefined:
            return left is UzonUndefined and right is UzonUndefined
        # §4.3: NaN is never equal to anything
        if isinstance(left, float) and left != left:
            return False
        if isinstance(right, float) and right != right:
            return False
        left_is_tu = isinstance(left, UzonTaggedUnion)
        right_is_tu = isinstance(right, UzonTaggedUnion)
        if left_is_tu != right_is_tu:
            raise UzonTypeError(
                f"Cannot compare tagged union with {self._type_name(right if left_is_tu else left)} "
                f"using 'is' — tagged unions can only be compared with other tagged unions",
                node.line, node.col, file=self._filename,
            )
        self._require_same_type(left, right, "is", node)
        return left == right

    # ── arithmetic ───────────────────────────────────────────────────

    @staticmethod
    def _resolve_numeric_type(left: Any, right: Any) -> tuple[str | None, bool]:
        """Return (type_name, adoptable) for binary numeric op result."""
        if isinstance(left, UzonInt) and isinstance(right, UzonInt):
            adoptable = left.adoptable and right.adoptable
            if left.adoptable and not right.adoptable:
                return right.type_name, adoptable
            return left.type_name, adoptable
        if isinstance(left, UzonInt):
            return left.type_name, False
        if isinstance(right, UzonInt):
            return right.type_name, False
        if isinstance(left, UzonFloat) and isinstance(right, UzonFloat):
            adoptable = left.adoptable and right.adoptable
            if left.adoptable and not right.adoptable:
                return right.type_name, adoptable
            return left.type_name, adoptable
        if isinstance(left, UzonFloat):
            return left.type_name, False
        if isinstance(right, UzonFloat):
            return right.type_name, False
        return None, False

    def _typed_int_result(
        self, result: int, type_name: str | None, node: Node, *, adoptable: bool = False
    ) -> int:
        """Wrap int result in UzonInt if typed, or range-check against i64."""
        if type_name:
            if adoptable:
                if not (I64_MIN <= result <= I64_MAX):
                    raise UzonRuntimeError(
                        f"Integer arithmetic overflow: result {result} "
                        f"exceeds i64 range ({I64_MIN}..{I64_MAX})",
                        node.line, node.col, file=self._filename,
                    )
            else:
                m = INT_TYPE_RE.match(type_name)
                if m:
                    self._check_int_range(result, m.group(1), int(m.group(2)), type_name, node, error_cls=UzonRuntimeError)
            return UzonInt(result, type_name, adoptable=adoptable)
        if not (I64_MIN <= result <= I64_MAX):
            raise UzonRuntimeError(
                f"Integer arithmetic overflow: result {result} "
                f"exceeds i64 range ({I64_MIN}..{I64_MAX})",
                node.line, node.col, file=self._filename,
            )
        return result

    @staticmethod
    def _typed_float_result(
        result: float, type_name: str | None, *, adoptable: bool = False
    ) -> float:
        """Wrap float result in UzonFloat if typed."""
        if type_name:
            return UzonFloat(result, type_name, adoptable=adoptable)
        return result

    def _eval_arithmetic(self, op: str, left: Any, right: Any, node: Node) -> Any:
        """§5.3: Evaluate arithmetic operators."""
        if left is None or right is None:
            raise UzonTypeError(
                f"Cannot use '{op}' with null", node.line, node.col,
                file=self._filename,
            )
        self._require_numeric(left, op, node)
        self._require_numeric(right, op, node)
        self._require_same_type(left, right, op, node)

        result_type, result_adoptable = self._resolve_numeric_type(left, right)

        if op == "+":
            result = int(left) + int(right) if isinstance(left, int) else float(left) + float(right)
        elif op == "-":
            result = int(left) - int(right) if isinstance(left, int) else float(left) - float(right)
        elif op == "*":
            result = int(left) * int(right) if isinstance(left, int) else float(left) * float(right)
        elif op == "/":
            result = self._eval_division(left, right, result_type, result_adoptable, node)
            if isinstance(result, (UzonInt, UzonFloat)):
                return result
        elif op == "%":
            result = self._eval_modulo(left, right, result_type, result_adoptable, node)
            if isinstance(result, (UzonInt, UzonFloat)):
                return result
        elif op == "^":
            return self._eval_power(left, right, result_type, result_adoptable, node)
        else:
            raise UzonRuntimeError(
                f"Unknown arithmetic operator: {op}", node.line, node.col,
                file=self._filename,
            )

        if isinstance(result, int) and not isinstance(result, bool):
            return self._typed_int_result(result, result_type, node, adoptable=result_adoptable)
        if isinstance(result, float):
            return self._typed_float_result(result, result_type, adoptable=result_adoptable)
        return result

    def _eval_division(
        self, left: Any, right: Any, result_type: str | None,
        result_adoptable: bool, node: Node,
    ) -> Any:
        """§5.3: Division with zero handling."""
        if right == 0:
            if isinstance(left, float):
                if left == 0.0:
                    return self._typed_float_result(float("nan"), result_type, adoptable=result_adoptable)
                return self._typed_float_result(
                    math.copysign(float("inf"), math.copysign(1.0, left) * math.copysign(1.0, right)),
                    result_type, adoptable=result_adoptable)
            raise UzonRuntimeError(
                "Division by zero", node.line, node.col,
                file=self._filename,
            )
        if isinstance(left, int):
            return _trunc_div(int(left), int(right))
        return float(left) / float(right)

    def _eval_modulo(
        self, left: Any, right: Any, result_type: str | None,
        result_adoptable: bool, node: Node,
    ) -> Any:
        """§5.3: Modulo with zero handling."""
        if right == 0:
            if isinstance(left, float):
                return self._typed_float_result(float("nan"), result_type, adoptable=result_adoptable)
            raise UzonRuntimeError(
                "Modulo by zero", node.line, node.col,
                file=self._filename,
            )
        if isinstance(left, int):
            return _trunc_mod(int(left), int(right))
        return math.fmod(float(left), float(right))

    def _eval_power(
        self, left: Any, right: Any, result_type: str | None,
        result_adoptable: bool, node: Node,
    ) -> Any:
        """§5.3: Exponentiation."""
        if isinstance(left, int) and isinstance(right, int) and right < 0:
            raise UzonRuntimeError(
                "Integer exponentiation requires non-negative exponent",
                node.line, node.col, file=self._filename,
            )
        if isinstance(left, float) or isinstance(right, float):
            fl, fr = float(left), float(right)
            if fl == 0.0 and fr < 0:
                r = math.copysign(float("inf"), fl) if (fr == int(fr) and int(fr) % 2 != 0) else float("inf")
                return self._typed_float_result(r, result_type, adoptable=result_adoptable)
            if fl < 0 and fr != int(fr):
                return self._typed_float_result(float("nan"), result_type, adoptable=result_adoptable)
            try:
                return self._typed_float_result(math.pow(fl, fr), result_type, adoptable=result_adoptable)
            except (ValueError, OverflowError):
                r = float("inf") if fl > 0 else float("nan")
                return self._typed_float_result(r, result_type, adoptable=result_adoptable)
        # Bug fix: ** allows 0 exponent (e.g., 2^0 = 1)
        result = int(left) ** int(right)
        return self._typed_int_result(result, result_type, node, adoptable=result_adoptable)

    # ── comparison ───────────────────────────────────────────────────

    def _eval_comparison(self, op: str, left: Any, right: Any, node: Node) -> bool:
        """§5.4: Evaluate comparison operators."""
        if left is None or right is None:
            raise UzonTypeError(
                f"Cannot use '{op}' with null", node.line, node.col,
                file=self._filename,
            )
        self._require_same_type(left, right, op, node)
        if not isinstance(left, (int, float, str)) or isinstance(left, bool):
            raise UzonTypeError(
                f"Cannot compare {self._type_name(left)} with '{op}'",
                node.line, node.col, file=self._filename,
            )
        if op == "<":
            return left < right
        if op == "<=":
            return left <= right
        if op == ">":
            return left > right
        return left >= right

    # ── collection operators ─────────────────────────────────────────

    @staticmethod
    def _concretize(val: Any) -> Any:
        """Return a non-adoptable version of a value for strict type comparison."""
        if isinstance(val, UzonInt) and val.adoptable:
            return UzonInt(int(val), val.type_name)
        if isinstance(val, UzonFloat) and val.adoptable:
            return UzonFloat(float(val), val.type_name)
        return val

    def _eval_in(self, left: Any, right: Any, node: Node) -> bool:
        """§5.8: Evaluate `in` membership operator."""
        if not isinstance(right, list):
            raise UzonTypeError(
                f"'in' requires a list on the right, got {self._type_name(right)}",
                node.line, node.col, file=self._filename,
            )
        if left is not None and right:
            representative = next((e for e in right if e is not None), None)
            if representative is not None:
                c_rep = self._concretize(representative)
                if not self._same_uzon_type(left, c_rep):
                    raise UzonTypeError(
                        f"Type mismatch in 'in': {self._type_name(left)} vs {self._type_name(c_rep)}",
                        node.line, node.col, file=self._filename,
                    )
        return left in right

    def _eval_concat(self, left: Any, right: Any, node: Node) -> Any:
        """§5.8.2: Evaluate `++` concatenation operator."""
        if isinstance(left, str) and isinstance(right, str):
            return left + right
        if isinstance(left, list) and isinstance(right, list):
            left_rep = next((e for e in left if e is not None), None)
            right_rep = next((e for e in right if e is not None), None)
            if left_rep is not None and right_rep is not None:
                if not self._same_uzon_type(left_rep, right_rep):
                    raise UzonTypeError(
                        f"'++' list element type mismatch: {self._type_name(left_rep)} vs {self._type_name(right_rep)}",
                        node.line, node.col, file=self._filename,
                    )
            return left + right
        raise UzonTypeError(
            f"'++' requires two strings or two lists, got {self._type_name(left)} and {self._type_name(right)}",
            node.line, node.col, file=self._filename,
        )

    def _eval_repeat(self, left: Any, right: Any, node: Node) -> Any:
        """§5.8.3: Evaluate `**` repetition operator."""
        if not isinstance(right, int) or isinstance(right, bool):
            raise UzonTypeError(
                f"'**' requires an integer on the right, got {self._type_name(right)}",
                node.line, node.col, file=self._filename,
            )
        # Bug fix: ** allows 0 repetition
        if right < 0:
            raise UzonRuntimeError(
                "'**' requires a non-negative integer (>= 0)", node.line, node.col,
                file=self._filename,
            )
        if isinstance(left, str):
            return left * right
        if isinstance(left, list):
            return left * right
        raise UzonTypeError(
            f"'**' requires a string or list on the left, got {self._type_name(left)}",
            node.line, node.col, file=self._filename,
        )


# ── module-level pure functions ──────────────────────────────────────

def _trunc_div(a: int, b: int) -> int:
    """§5.3: Integer division truncating toward zero (not floor)."""
    q, _ = divmod(abs(a), abs(b))
    return -q if (a < 0) != (b < 0) else q


def _trunc_mod(a: int, b: int) -> int:
    """§5.3: Modulo with truncation semantics (sign follows dividend)."""
    q = _trunc_div(a, b)
    return a - q * b
