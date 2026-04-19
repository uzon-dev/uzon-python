# SPDX-FileCopyrightText: © 2026 Suho Kang
# SPDX-License-Identifier: MIT
"""Function evaluation mixin.

Implements §3.8 (function expressions), §5.15 (function calls),
and parameter/return type checking.
"""

from __future__ import annotations

from typing import Any

from ..ast_nodes import FunctionCall, FunctionExpr, Node
from ..errors import UzonCircularError, UzonRuntimeError, UzonTypeError
from ..scope import Scope
from ..types import (
    UzonBuiltinFunction, UzonEnum, UzonFloat, UzonFunction, UzonInt, UzonTaggedUnion,
    UzonUndefined,
)
from ._constants import FLOAT_TYPES, INT_TYPE_RE


class FunctionMixin:
    """Function evaluation methods mixed into the Evaluator."""

    def _eval_function_expr(
        self, node: FunctionExpr, scope: Scope, exclude: str | None
    ) -> UzonFunction:
        """§3.8: Evaluate a function expression — creates a closure."""
        params: list[tuple[str, str, Any]] = []
        for param in node.params:
            type_name = param.type.name
            # §6.2: Validate parameter type exists in scope at definition time
            self._validate_type_name(param.type, node, scope)
            default = None
            if param.default is not None:
                default = self._eval_node(param.default, scope, exclude)
                # §3.8: default value must be defined and match the param type
                if default is UzonUndefined:
                    raise UzonTypeError(
                        f"Default for parameter '{param.name}' is undefined",
                        param.default.line, param.default.col,
                        file=self._filename,
                    )
                self._check_type_assertion(default, param.type, param.default, scope)
                default = self._wrap_typed(default, param.type)
            params.append((param.name, type_name, default))

        # §6.2: Validate return type exists in scope at definition time
        self._validate_type_name(node.return_type, node, scope)

        return UzonFunction(
            params=params,
            return_type=node.return_type.name,
            body_bindings=node.body_bindings,
            body_expr=node.body_expr,
            closure_scope=scope,
        )

    def _eval_function_call(
        self, node: FunctionCall, scope: Scope, exclude: str | None
    ) -> Any:
        """§5.15: Evaluate a function call."""
        callee = self._eval_node(node.callee, scope, exclude)

        if isinstance(callee, UzonBuiltinFunction):
            args = [self._eval_node(a, scope, exclude) for a in node.args]
            if not (callee.min_args <= len(args) <= callee.max_args):
                raise UzonTypeError(
                    f"std.{callee.name} expects {callee.min_args}"
                    + (f"-{callee.max_args}" if callee.max_args != callee.min_args else "")
                    + f" arguments, got {len(args)}",
                    node.line, node.col, file=self._filename,
                )
            return callee.func(args, node)

        if callee is UzonUndefined:
            raise UzonRuntimeError(
                "Cannot call undefined — not a function",
                node.line, node.col, file=self._filename,
            )
        if not isinstance(callee, UzonFunction):
            raise UzonTypeError(
                f"Cannot call {self._type_name(callee)} — not a function",
                node.line, node.col, file=self._filename,
            )

        func_id = id(callee)
        if func_id in self._call_stack:
            raise UzonCircularError(
                "Recursive function call detected — call graph must be a DAG",
                node.line, node.col, file=self._filename,
            )

        args = [self._eval_node(a, scope, exclude) for a in node.args]
        func = callee
        self._check_arg_count(func, args, node)

        # §3.8: Two-level scope for function bodies
        param_scope: dict[str, Any] = {}
        for i, (pname, ptype, pdefault) in enumerate(func.params):
            val = args[i] if i < len(args) else pdefault
            self._check_param_type(val, ptype, pname, node)
            val = self._coerce_to_param_type(val, ptype)
            param_scope[pname] = val

        body_binding_scope = Scope(parent=func.closure_scope)
        body_scope = Scope(parent=body_binding_scope, closure_scope=body_binding_scope)
        for pname, val in param_scope.items():
            body_scope.define(pname, val)

        self._call_stack.append(func_id)
        try:
            for binding in func.body_bindings:
                val = self._eval_node(binding.value, body_scope, binding.name)
                body_scope.define(binding.name, val)
                body_binding_scope.define(binding.name, val)
            result = self._eval_body_with_return_hint(
                func.body_expr, func.return_type, body_scope,
            )
        finally:
            self._call_stack.pop()

        self._check_return_type(result, func.return_type, node)
        return result

    def _eval_body_with_return_hint(
        self, body_expr: Any, return_type: str, scope: Scope,
    ) -> Any:
        """§3.5 R7 v0.10: If the function body is a bare Identifier and
        the return type resolves to a named enum whose variants include
        that identifier, return the corresponding UzonEnum."""
        from ..ast_nodes import Identifier as _Ident
        if isinstance(body_expr, _Ident) and return_type:
            type_info = scope.get_type(return_type)
            if (type_info and type_info.get("kind") == "enum"
                    and body_expr.name in type_info["variants"]):
                from ..types import UzonEnum as _UE
                return _UE(
                    body_expr.name, type_info["variants"], type_info["name"],
                )
        return self._eval_node(body_expr, scope, None)

    def _apply_function(self, func: UzonFunction, args: list, node: Node) -> Any:
        """Apply a UzonFunction to arguments (used by std.map/filter/reduce/sort)."""
        func_id = id(func)
        if func_id in self._call_stack:
            raise UzonCircularError(
                "Recursive function call detected — call graph must be a DAG",
                node.line, node.col, file=self._filename,
            )
        self._check_arg_count(func, args, node)

        # §3.8: Two-level scope for function bodies (matches _eval_function_call)
        param_scope: dict[str, Any] = {}
        for i, (pname, ptype, pdefault) in enumerate(func.params):
            val = args[i] if i < len(args) else pdefault
            self._check_param_type(val, ptype, pname, node)
            val = self._coerce_to_param_type(val, ptype)
            param_scope[pname] = val

        body_binding_scope = Scope(parent=func.closure_scope)
        body_scope = Scope(parent=body_binding_scope, closure_scope=body_binding_scope)
        for pname, val in param_scope.items():
            body_scope.define(pname, val)

        self._call_stack.append(func_id)
        try:
            for binding in func.body_bindings:
                val = self._eval_node(binding.value, body_scope, binding.name)
                body_scope.define(binding.name, val)
                body_binding_scope.define(binding.name, val)
            result = self._eval_node(func.body_expr, body_scope, None)
        finally:
            self._call_stack.pop()
        self._check_return_type(result, func.return_type, node)
        return result

    # ── argument helpers ─────────────────────────────────────────────

    def _check_arg_count(self, func: UzonFunction, args: list, node: Node) -> None:
        min_required = sum(1 for _, _, d in func.params if d is None)
        max_params = len(func.params)
        if not (min_required <= len(args) <= max_params):
            raise UzonTypeError(
                f"Function expects {min_required}"
                + (f"-{max_params}" if max_params != min_required else "")
                + f" arguments, got {len(args)}",
                node.line, node.col, file=self._filename,
            )

    def _coerce_to_param_type(self, value: Any, type_name: str) -> Any:
        """Coerce an adoptable literal to the declared parameter type."""
        tn = self._normalize_type_name(type_name)
        if isinstance(value, UzonInt) and value.adoptable and INT_TYPE_RE.match(tn):
            return UzonInt(int(value), tn)
        if isinstance(value, UzonFloat) and value.adoptable and tn in FLOAT_TYPES:
            return UzonFloat(float(value), tn)
        return value

    def _value_matches_type(self, value: Any, type_name: str) -> bool:
        """Check if a value matches a type name (handles compound types)."""
        expected = self._normalize_type_name(type_name)
        actual = self._type_name(value)
        if expected == actual:
            return True
        if isinstance(value, UzonInt) and value.adoptable and expected.startswith(("i", "u")):
            return True
        if isinstance(value, UzonFloat) and value.adoptable and expected.startswith("f"):
            return True
        if expected.startswith("[") and expected.endswith("]") and isinstance(value, list):
            elem_type = expected[1:-1]
            return all(self._value_matches_type(e, elem_type) for e in value)
        if expected.startswith("(") and expected.endswith(")") and isinstance(value, tuple):
            inner = expected[1:-1]
            type_parts = [t.strip() for t in inner.split(",")]
            if len(type_parts) != len(value):
                return False
            return all(self._value_matches_type(v, t) for v, t in zip(value, type_parts))
        if isinstance(value, dict) and actual == "struct":
            return False
        if isinstance(value, UzonTaggedUnion) and value.type_name:
            return self._normalize_type_name(value.type_name) == expected
        # Bug fix: UzonEnum checked in function type matching
        if isinstance(value, UzonEnum) and value.type_name:
            return self._normalize_type_name(value.type_name) == expected
        return False

    def _value_matches_type_strict(self, value: Any, type_name: str) -> bool:
        """Check if a value matches a type name using concrete types only.

        Unlike _value_matches_type, adoptable literals match only their
        default type (i64/f64), not any integer/float type.
        """
        expected = self._normalize_type_name(type_name)
        actual = self._type_name(value)
        if expected == actual:
            return True
        # Adoptable literals: match only their default type (e.g. i64, f64)
        if isinstance(value, UzonInt) and value.adoptable:
            return expected == value.type_name
        if isinstance(value, UzonFloat) and value.adoptable:
            return expected == value.type_name
        if expected.startswith("[") and expected.endswith("]") and isinstance(value, list):
            elem_type = expected[1:-1]
            return all(self._value_matches_type_strict(e, elem_type) for e in value)
        if expected.startswith("(") and expected.endswith(")") and isinstance(value, tuple):
            inner = expected[1:-1]
            type_parts = [t.strip() for t in inner.split(",")]
            if len(type_parts) != len(value):
                return False
            return all(self._value_matches_type_strict(v, t) for v, t in zip(value, type_parts))
        if isinstance(value, UzonTaggedUnion) and value.type_name:
            return self._normalize_type_name(value.type_name) == expected
        if isinstance(value, UzonEnum) and value.type_name:
            return self._normalize_type_name(value.type_name) == expected
        return False

    def _check_param_type(self, value: Any, type_name: str, param_name: str, node: Node) -> None:
        if self._value_matches_type(value, type_name):
            return
        actual = self._type_name(value)
        raise UzonTypeError(
            f"Argument type mismatch for parameter '{param_name}': "
            f"expected {type_name}, got {actual}",
            node.line, node.col, file=self._filename,
        )

    def _check_return_type(self, value: Any, type_name: str, node: Node) -> None:
        if self._value_matches_type(value, type_name):
            return
        actual = self._type_name(value)
        raise UzonTypeError(
            f"Function return type mismatch: expected {type_name}, got {actual}",
            node.line, node.col, file=self._filename,
        )

    @staticmethod
    def _normalize_type_name(type_name: str) -> str:
        """Normalize type names for comparison."""
        if "." in type_name:
            type_name = type_name.rsplit(".", 1)[-1]
        mapping = {
            "i8": "i8", "i16": "i16", "i32": "i32", "i64": "i64",
            "u8": "u8", "u16": "u16", "u32": "u32", "u64": "u64",
            "f32": "f32", "f64": "f64",
            "bool": "bool", "string": "string", "null": "null",
        }
        return mapping.get(type_name, type_name)
