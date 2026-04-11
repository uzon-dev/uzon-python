# SPDX-FileCopyrightText: © 2026 Suho Kang
# SPDX-License-Identifier: MIT
"""Standard library mixin.

Implements §5.16: the 18 built-in functions available as std.*,
including std.lower and std.upper (§5.16.4).
"""

from __future__ import annotations

import functools
import math
from typing import Any

from ..ast_nodes import Node
from ..errors import UzonTypeError
from ..types import UzonBuiltinFunction, UzonFloat, UzonFunction, UzonInt


class StdlibMixin:
    """Standard library methods mixed into the Evaluator."""

    def _build_std(self) -> dict[str, UzonBuiltinFunction]:
        """§5.16: Build the std library as a struct of built-in functions."""
        return {
            "len": UzonBuiltinFunction("len", self._std_len, 1, 1),
            "has": UzonBuiltinFunction("has", self._std_has, 2, 2),
            "get": UzonBuiltinFunction("get", self._std_get, 2, 2),
            "keys": UzonBuiltinFunction("keys", self._std_keys, 1, 1),
            "values": UzonBuiltinFunction("values", self._std_values, 1, 1),
            "map": UzonBuiltinFunction("map", self._std_map, 2, 2),
            "filter": UzonBuiltinFunction("filter", self._std_filter, 2, 2),
            "reduce": UzonBuiltinFunction("reduce", self._std_reduce, 3, 3),
            "sort": UzonBuiltinFunction("sort", self._std_sort, 2, 2),
            "isNan": UzonBuiltinFunction("isNan", self._std_isNan, 1, 1),
            "isInf": UzonBuiltinFunction("isInf", self._std_isInf, 1, 1),
            "isFinite": UzonBuiltinFunction("isFinite", self._std_isFinite, 1, 1),
            "join": UzonBuiltinFunction("join", self._std_join, 2, 2),
            "replace": UzonBuiltinFunction("replace", self._std_replace, 3, 3),
            "split": UzonBuiltinFunction("split", self._std_split, 2, 2),
            "trim": UzonBuiltinFunction("trim", self._std_trim, 1, 1),
            "lower": UzonBuiltinFunction("lower", self._std_lower, 1, 1),
            "upper": UzonBuiltinFunction("upper", self._std_upper, 1, 1),
        }

    # ── collection functions ─────────────────────────────────────────

    def _std_len(self, args: list, node: Node) -> UzonInt:
        val = args[0]
        if isinstance(val, (list, tuple, dict, str)):
            return UzonInt(len(val), "i64")
        raise UzonTypeError(
            f"std.len expects a list, tuple, struct, or string, got {self._type_name(val)}",
            node.line, node.col, file=self._filename,
        )

    def _std_has(self, args: list, node: Node) -> bool:
        collection, key = args[0], args[1]
        if isinstance(collection, list):
            if collection:
                if not self._same_uzon_type(collection[0], key):
                    raise UzonTypeError(
                        f"std.has: key type {self._type_name(key)} does not match "
                        f"element type {self._type_name(collection[0])}",
                        node.line, node.col, file=self._filename,
                    )
            return key in collection
        if isinstance(collection, dict):
            if not isinstance(key, str):
                raise UzonTypeError(
                    "std.has: struct key must be a string",
                    node.line, node.col, file=self._filename,
                )
            return key in collection
        raise UzonTypeError(
            f"std.has expects a list or struct, got {self._type_name(collection)}",
            node.line, node.col, file=self._filename,
        )

    def _std_get(self, args: list, node: Node) -> Any:
        collection, key = args[0], args[1]
        if isinstance(collection, list):
            if not isinstance(key, int):
                raise UzonTypeError(
                    "std.get: list index must be an integer",
                    node.line, node.col, file=self._filename,
                )
            from ..types import UzonUndefined
            idx = int(key)
            if 0 <= idx < len(collection):
                return collection[idx]
            return UzonUndefined
        if isinstance(collection, dict):
            if not isinstance(key, str):
                raise UzonTypeError(
                    "std.get: struct key must be a string",
                    node.line, node.col, file=self._filename,
                )
            from ..types import UzonUndefined
            return collection.get(key, UzonUndefined)
        raise UzonTypeError(
            f"std.get expects a list or struct, got {self._type_name(collection)}",
            node.line, node.col, file=self._filename,
        )

    def _std_keys(self, args: list, node: Node) -> list:
        val = args[0]
        if not isinstance(val, dict):
            raise UzonTypeError(
                f"std.keys expects a struct, got {self._type_name(val)}",
                node.line, node.col, file=self._filename,
            )
        return list(val.keys())

    def _std_values(self, args: list, node: Node) -> tuple:
        val = args[0]
        if not isinstance(val, dict):
            raise UzonTypeError(
                f"std.values expects a struct, got {self._type_name(val)}",
                node.line, node.col, file=self._filename,
            )
        return tuple(val.values())

    # ── higher-order functions ───────────────────────────────────────

    def _std_map(self, args: list, node: Node) -> list:
        lst, func = args[0], args[1]
        if not isinstance(lst, list):
            raise UzonTypeError(
                f"std.map expects a list as first argument, got {self._type_name(lst)}",
                node.line, node.col, file=self._filename,
            )
        if not isinstance(func, UzonFunction):
            raise UzonTypeError(
                "std.map expects a function as second argument",
                node.line, node.col, file=self._filename,
            )
        return [self._apply_function(func, [elem], node) for elem in lst]

    def _std_filter(self, args: list, node: Node) -> list:
        lst, func = args[0], args[1]
        if not isinstance(lst, list):
            raise UzonTypeError(
                f"std.filter expects a list, got {self._type_name(lst)}",
                node.line, node.col, file=self._filename,
            )
        if not isinstance(func, UzonFunction):
            raise UzonTypeError(
                "std.filter expects a function as second argument",
                node.line, node.col, file=self._filename,
            )
        result = []
        for elem in lst:
            val = self._apply_function(func, [elem], node)
            if not isinstance(val, bool):
                raise UzonTypeError(
                    "std.filter function must return bool",
                    node.line, node.col, file=self._filename,
                )
            if val:
                result.append(elem)
        return result

    def _std_sort(self, args: list, node: Node) -> list:
        lst, func = args[0], args[1]
        if not isinstance(lst, list):
            raise UzonTypeError(
                f"std.sort expects a list, got {self._type_name(lst)}",
                node.line, node.col, file=self._filename,
            )
        if not isinstance(func, UzonFunction):
            raise UzonTypeError(
                "std.sort expects a function as second argument",
                node.line, node.col, file=self._filename,
            )
        def cmp(a: Any, b: Any) -> int:
            val = self._apply_function(func, [a, b], node)
            if not isinstance(val, bool):
                raise UzonTypeError(
                    "std.sort comparator must return bool",
                    node.line, node.col, file=self._filename,
                )
            if val:
                return -1
            val2 = self._apply_function(func, [b, a], node)
            if val2:
                return 1
            return 0
        return sorted(lst, key=functools.cmp_to_key(cmp))

    def _std_reduce(self, args: list, node: Node) -> Any:
        lst, initial, func = args[0], args[1], args[2]
        if not isinstance(lst, list):
            raise UzonTypeError(
                f"std.reduce expects a list, got {self._type_name(lst)}",
                node.line, node.col, file=self._filename,
            )
        if not isinstance(func, UzonFunction):
            raise UzonTypeError(
                "std.reduce expects a function as third argument",
                node.line, node.col, file=self._filename,
            )
        acc = initial
        for elem in lst:
            acc = self._apply_function(func, [acc, elem], node)
        return acc

    # ── numeric inspection ───────────────────────────────────────────

    def _std_isNan(self, args: list, node: Node) -> bool:
        val = args[0]
        if not isinstance(val, float):
            raise UzonTypeError(
                f"std.isNan expects a float, got {self._type_name(val)}",
                node.line, node.col, file=self._filename,
            )
        return math.isnan(val)

    def _std_isInf(self, args: list, node: Node) -> bool:
        val = args[0]
        if not isinstance(val, float):
            raise UzonTypeError(
                f"std.isInf expects a float, got {self._type_name(val)}",
                node.line, node.col, file=self._filename,
            )
        return math.isinf(val)

    def _std_isFinite(self, args: list, node: Node) -> bool:
        val = args[0]
        if not isinstance(val, float):
            raise UzonTypeError(
                f"std.isFinite expects a float, got {self._type_name(val)}",
                node.line, node.col, file=self._filename,
            )
        return math.isfinite(val)

    # ── string functions (§5.16.4) ───────────────────────────────────

    def _std_join(self, args: list, node: Node) -> str:
        lst, sep = args[0], args[1]
        if not isinstance(lst, list):
            raise UzonTypeError(
                f"std.join expects a [string] list, got {self._type_name(lst)}",
                node.line, node.col, file=self._filename,
            )
        if not isinstance(sep, str):
            raise UzonTypeError(
                f"std.join separator must be a string, got {self._type_name(sep)}",
                node.line, node.col, file=self._filename,
            )
        for i, elem in enumerate(lst):
            if not isinstance(elem, str):
                raise UzonTypeError(
                    f"std.join: element {i} is {self._type_name(elem)}, expected string",
                    node.line, node.col, file=self._filename,
                )
        return sep.join(lst)

    def _std_replace(self, args: list, node: Node) -> str:
        s, target, replacement = args[0], args[1], args[2]
        if not isinstance(s, str):
            raise UzonTypeError(
                f"std.replace expects a string, got {self._type_name(s)}",
                node.line, node.col, file=self._filename,
            )
        if not isinstance(target, str):
            raise UzonTypeError(
                f"std.replace target must be a string, got {self._type_name(target)}",
                node.line, node.col, file=self._filename,
            )
        if not isinstance(replacement, str):
            raise UzonTypeError(
                f"std.replace replacement must be a string, got {self._type_name(replacement)}",
                node.line, node.col, file=self._filename,
            )
        if target == "":
            return s
        return s.replace(target, replacement)

    def _std_split(self, args: list, node: Node) -> list:
        s, delimiter = args[0], args[1]
        if not isinstance(s, str):
            raise UzonTypeError(
                f"std.split expects a string, got {self._type_name(s)}",
                node.line, node.col, file=self._filename,
            )
        if not isinstance(delimiter, str):
            raise UzonTypeError(
                f"std.split delimiter must be a string, got {self._type_name(delimiter)}",
                node.line, node.col, file=self._filename,
            )
        if s == "":
            return [""]
        if delimiter == "":
            return list(s)
        return s.split(delimiter)

    def _std_trim(self, args: list, node: Node) -> str:
        s = args[0]
        if not isinstance(s, str):
            raise UzonTypeError(
                f"std.trim expects a string, got {self._type_name(s)}",
                node.line, node.col, file=self._filename,
            )
        return s.strip()

    def _std_lower(self, args: list, node: Node) -> str:
        s = args[0]
        if not isinstance(s, str):
            raise UzonTypeError(
                f"std.lower expects a string, got {self._type_name(s)}",
                node.line, node.col, file=self._filename,
            )
        return s.lower()

    def _std_upper(self, args: list, node: Node) -> str:
        s = args[0]
        if not isinstance(s, str):
            raise UzonTypeError(
                f"std.upper expects a string, got {self._type_name(s)}",
                node.line, node.col, file=self._filename,
            )
        return s.upper()
