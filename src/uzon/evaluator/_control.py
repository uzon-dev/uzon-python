# SPDX-FileCopyrightText: © 2026 Suho Kang
# SPDX-License-Identifier: MIT
"""Control flow and variant evaluation mixin.

Implements §4.4.1 (string interpolation), §5.7 (or else), §5.9 (if/then/else),
§5.10 (case/when), §5.14 (field extraction), §3.5–§3.7 (enum, union, tagged union),
and §D.5 (speculative evaluation).
"""

from __future__ import annotations

from typing import Any

from ..ast_nodes import (
    BinaryOp, FieldExtraction, FromEnum, FromUnion, Identifier,
    IfExpr, CaseExpr, MemberAccess, NamedVariant, Node,
    StringLiteral, TypeAnnotation, UndefinedLiteral,
)
from ..errors import UzonRuntimeError, UzonTypeError
from .._format import format_float as _format_float
from ..types import (
    UzonBuiltinFunction, UzonEnum, UzonFloat, UzonFunction,
    UzonInt, UzonTaggedUnion, UzonUndefined, UzonUnion,
)
from ._constants import SPECULATIVE_FAILED


class ControlMixin:
    """Control flow and variant evaluation methods mixed into the Evaluator."""

    # ── string evaluation ────────────────────────────────────────────

    def _eval_string(
        self, node: StringLiteral, scope, exclude: str | None
    ) -> str:
        if len(node.parts) == 1 and isinstance(node.parts[0], str):
            return node.parts[0]
        result: list[str] = []
        for part in node.parts:
            if isinstance(part, str):
                result.append(part)
            else:
                val = self._eval_node(part, scope, exclude)
                result.append(self._value_to_string(val, part))
        return "".join(result)

    def _check_to_string_convertible(self, val: Any, node: Node) -> None:
        """Check that a value can be converted to string (§5.11.2)."""
        check = val
        if isinstance(check, (UzonTaggedUnion, UzonUnion)):
            check = check.value
        if isinstance(check, (dict, list, tuple)):
            raise UzonTypeError(
                f"Cannot convert {type(check).__name__} to string",
                node.line, node.col, file=self._filename,
            )
        if isinstance(check, (UzonFunction, UzonBuiltinFunction)):
            raise UzonTypeError(
                "Cannot convert function to string",
                node.line, node.col, file=self._filename,
            )

    def _value_to_string(self, val: Any, node: Node) -> str:
        """Convert a value to its string representation (§5.11.2)."""
        if val is UzonUndefined:
            raise UzonRuntimeError(
                "Cannot interpolate undefined value — use 'or else' to provide a fallback",
                node.line, node.col, file=self._filename,
            )
        if val is None:
            return "null"
        if isinstance(val, bool):
            return "true" if val else "false"
        if isinstance(val, int):
            return str(val)
        if isinstance(val, float):
            return _format_float(val)
        if isinstance(val, str):
            return val
        if isinstance(val, UzonEnum):
            return val.value
        if isinstance(val, UzonTaggedUnion):
            return self._value_to_string(val.value, node)
        if isinstance(val, UzonUnion):
            return self._value_to_string(val.value, node)
        if isinstance(val, (dict, list, tuple)):
            raise UzonRuntimeError(
                f"Cannot convert {type(val).__name__} to string",
                node.line, node.col, file=self._filename,
            )
        if isinstance(val, (UzonFunction, UzonBuiltinFunction)):
            raise UzonRuntimeError(
                "Cannot convert function to string",
                node.line, node.col, file=self._filename,
            )
        return str(val)

    # ── speculative evaluation (§D.5) ────────────────────────────────

    def _speculative_eval(self, node: Node, scope, exclude: str | None) -> Any:
        """§D.5: Evaluate speculatively, suppressing runtime errors only."""
        try:
            return self._eval_node(node, scope, exclude)
        except UzonRuntimeError:
            return SPECULATIVE_FAILED

    def _check_branch_type_compat(self, values: list[Any], node: Node) -> None:
        """Check that all branch results are type-compatible (null exempt)."""
        non_null = [v for v in values if v is not None and v is not UzonUndefined]
        if len(non_null) < 2:
            return
        first = non_null[0]
        for other in non_null[1:]:
            if not self._same_uzon_type(first, other):
                raise UzonTypeError(
                    f"Branch type mismatch: {self._type_name(first)} vs {self._type_name(other)}",
                    node.line, node.col, file=self._filename,
                )

    # ── if/then/else (§5.9) ──────────────────────────────────────────

    def _eval_if(self, node: IfExpr, scope, exclude: str | None) -> Any:
        """§5.9: Evaluate `if cond then a else b` with speculative branch type check."""
        cond = self._unwrap_transparent(self._eval_node(node.condition, scope, exclude))
        if not isinstance(cond, bool):
            raise UzonTypeError(
                f"'if' condition must be bool, got {self._type_name(cond)}",
                node.line, node.col, file=self._filename,
            )
        if cond:
            result = self._eval_node(node.then_branch, scope, exclude)
            other = self._speculative_eval(node.else_branch, scope, exclude)
        else:
            result = self._eval_node(node.else_branch, scope, exclude)
            other = self._speculative_eval(node.then_branch, scope, exclude)

        if other is not SPECULATIVE_FAILED:
            self._check_branch_type_compat([result, other], node)
        return result

    # ── case/when (§5.10) ────────────────────────────────────────────

    def _eval_case(self, node: CaseExpr, scope, exclude: str | None) -> Any:
        """§5.10: Evaluate `case scrutinee when v1 then r1 ... else default`."""
        scrutinee = self._eval_node(node.scrutinee, scope, exclude)

        if scrutinee is UzonUndefined:
            raise UzonRuntimeError(
                "Cannot use 'case' on undefined — check with 'is undefined' first",
                node.line, node.col, file=self._filename,
            )
        if isinstance(scrutinee, UzonUnion):
            raise UzonRuntimeError(
                "Cannot use 'case' on untagged union — use tagged union with 'when named'",
                node.line, node.col, file=self._filename,
            )

        result = None
        matched_idx: int | None = None

        for i, clause in enumerate(node.when_clauses):
            if isinstance(clause.value, UndefinedLiteral):
                raise UzonRuntimeError(
                    "'when undefined' is not allowed",
                    clause.value.line, clause.value.col,
                    file=self._filename,
                )
            if clause.is_named:
                if not isinstance(scrutinee, UzonTaggedUnion):
                    raise UzonTypeError(
                        "'when named' requires a tagged union scrutinee",
                        clause.line, clause.col, file=self._filename,
                    )
                if not isinstance(clause.value, Identifier):
                    raise UzonRuntimeError(
                        "'when named' requires a variant name",
                        clause.line, clause.col, file=self._filename,
                    )
                if scrutinee.variants and clause.value.name not in scrutinee.variants:
                    raise UzonTypeError(
                        f"'{clause.value.name}' is not a variant of this tagged union",
                        clause.value.line, clause.value.col,
                        file=self._filename,
                    )
            if matched_idx is not None:
                continue
            if clause.is_named:
                if scrutinee.tag == clause.value.name:
                    result = self._eval_node(clause.result, scope, exclude)
                    matched_idx = i
            else:
                when_val = self._resolve_when_value(clause.value, scrutinee, scope, exclude)
                if self._eval_is(scrutinee, when_val, node):
                    result = self._eval_node(clause.result, scope, exclude)
                    matched_idx = i

        if matched_idx is None:
            result = self._eval_node(node.else_branch, scope, exclude)

        # §D.5: speculatively evaluate all non-selected branches for type checking
        branch_values = [result]
        for i, clause in enumerate(node.when_clauses):
            if i != matched_idx:
                spec = self._speculative_eval(clause.result, scope, exclude)
                if spec is not SPECULATIVE_FAILED:
                    branch_values.append(spec)
        if matched_idx is not None:
            spec_else = self._speculative_eval(node.else_branch, scope, exclude)
            if spec_else is not SPECULATIVE_FAILED:
                branch_values.append(spec_else)
        self._check_branch_type_compat(branch_values, node)

        return result

    def _resolve_when_value(
        self, value_node: Node, scrutinee: Any, scope, exclude: str | None
    ) -> Any:
        """Resolve a when clause value, with enum variant inference from scrutinee."""
        if (isinstance(scrutinee, UzonEnum) and scrutinee.type_name
                and isinstance(value_node, Identifier)):
            variant_name = value_node.name
            if variant_name in scrutinee.variants:
                return UzonEnum(variant_name, scrutinee.variants, scrutinee.type_name)
        return self._eval_node(value_node, scope, exclude)

    # ── field extraction (§5.14) ─────────────────────────────────────

    def _eval_field_extraction(
        self, node: FieldExtraction, scope, exclude: str | None
    ) -> Any:
        """§5.14: Evaluate `name is of expr` — equivalent to `name is expr.name`."""
        source = self._eval_node(node.source, scope, exclude)
        if source is UzonUndefined:
            return UzonUndefined
        if isinstance(source, UzonTaggedUnion):
            source = source.value
        if not isinstance(source, dict):
            raise UzonTypeError(
                f"'of' requires a struct, got {self._type_name(source)}",
                node.line, node.col, file=self._filename,
            )
        if exclude is None:
            raise UzonRuntimeError(
                "'of' can only be used in binding position",
                node.line, node.col, file=self._filename,
            )
        return source.get(exclude, UzonUndefined)

    # ── enum (§3.5) ──────────────────────────────────────────────────

    def _eval_from_enum(
        self, node: FromEnum, scope, exclude: str | None
    ) -> UzonEnum:
        """§3.5: Evaluate `value from variant1, variant2, ...`."""
        if not isinstance(node.value, Identifier):
            raise UzonRuntimeError(
                "Enum value must be a bare identifier (variant name)",
                node.line, node.col, file=self._filename,
            )
        if len(node.variants) < 2:
            raise UzonTypeError(
                "Enum must have at least 2 variants",
                node.line, node.col, file=self._filename,
            )
        variant_name = node.value.name
        if variant_name not in node.variants:
            raise UzonTypeError(
                f"'{variant_name}' is not a variant in [{', '.join(node.variants)}]",
                node.line, node.col, file=self._filename,
            )
        return UzonEnum(variant_name, list(node.variants))

    # ── union (§3.6) ─────────────────────────────────────────────────

    def _eval_from_union(
        self, node: FromUnion, scope, exclude: str | None
    ) -> UzonUnion:
        """§3.6: Evaluate `value from union type1, type2, ...`."""
        value = self._eval_node(node.value, scope, exclude)
        if value is UzonUndefined:
            raise UzonRuntimeError(
                "Union operand is undefined", node.line, node.col,
                file=self._filename,
            )
        type_names = [t.name for t in node.types]
        if len(type_names) < 2:
            raise UzonTypeError(
                "Union must have at least 2 member types",
                node.line, node.col, file=self._filename,
            )
        return UzonUnion(value, type_names)

    # ── tagged union (§3.7) ──────────────────────────────────────────

    def _eval_named_variant(
        self, node: NamedVariant, scope, exclude: str | None
    ) -> UzonTaggedUnion:
        """§3.7: Evaluate `value named tag [from variant as type, ...]`."""
        if node.variants and len(node.variants) < 2:
            raise UzonTypeError(
                "Tagged union must have at least 2 variants",
                node.line, node.col, file=self._filename,
            )
        variants: dict[str, str | None] = {}
        type_name: str | None = None
        for var_name, var_type in node.variants:
            variants[var_name] = var_type.name if var_type else None

        if isinstance(node.value, TypeAnnotation):
            type_info = self._resolve_named_type(node.value.type, scope, node)
            if type_info and type_info.get("kind") == "tagged_union":
                value = self._eval_node(node.value.expr, scope, exclude)
                if not variants:
                    variants = dict(type_info.get("variants", {}))
                    type_name = type_info.get("name")
            else:
                value = self._eval_node(node.value, scope, exclude)
        else:
            value = self._eval_node(node.value, scope, exclude)

        if value is UzonUndefined:
            raise UzonRuntimeError(
                "Tagged union operand is undefined", node.line, node.col,
                file=self._filename,
            )
        tag = node.tag

        if tag not in variants and variants:
            raise UzonTypeError(
                f"Active variant '{tag}' is not in variant definitions",
                node.line, node.col, file=self._filename,
            )
        result = UzonTaggedUnion(value, tag, variants)
        if type_name:
            result.type_name = type_name
        return result

    # ── is named (§3.7.2) ───────────────────────────────────────────

    def _eval_is_named(self, op: str, left: Any, node: BinaryOp) -> bool:
        """§3.7.2: Evaluate `value is named tag` / `value is not named tag`."""
        if left is UzonUndefined:
            raise UzonRuntimeError(
                "'is named' requires a tagged union, got undefined",
                node.line, node.col, file=self._filename,
            )
        if not isinstance(left, UzonTaggedUnion):
            raise UzonTypeError(
                f"'is named' requires a tagged union, got {self._type_name(left)}",
                node.line, node.col, file=self._filename,
            )
        if not isinstance(node.right, Identifier):
            raise UzonRuntimeError(
                "'is named' requires a variant name on the right",
                node.line, node.col, file=self._filename,
            )
        tag_name = node.right.name
        if left.variants and tag_name not in left.variants:
            raise UzonTypeError(
                f"'{tag_name}' is not a variant of this tagged union",
                node.line, node.col, file=self._filename,
            )
        result = left.tag == tag_name
        return result if op == "is named" else not result
