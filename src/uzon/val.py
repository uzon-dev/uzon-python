# SPDX-FileCopyrightText: © 2026 Suho Kang
# SPDX-License-Identifier: MIT
"""Factory helpers for creating typed UZON values from Python.

Usage::

    from uzon import val

    port = val.u16(8080)          # UzonInt(8080, 'u16')
    score = val.f32(9.5)          # UzonFloat(9.5, 'f32')
    big = val.i128(2**100)        # UzonInt(..., 'i128')

    color = val.enum("Red", ["Red", "Green", "Blue"], type_name="Color")
    result = val.tagged("Ok", "done", {"Ok": "string", "Err": "string"})
    flexible = val.union(42, ["string", "i32"])

Integer and float type names are resolved dynamically via attribute access,
so any bit width supported by the UZON spec works without code changes.
"""

from __future__ import annotations

import re
from typing import Any

from .types import (
    UzonEnum,
    UzonFloat,
    UzonInt,
    UzonStruct,
    UzonTaggedUnion,
    UzonUnion,
)

_INT_TYPE_RE = re.compile(r'^([iu])(\d+)$')
_FLOAT_TYPES = frozenset({'f16', 'f32', 'f64', 'f80', 'f128'})


class _ValFactory:
    """Dynamic factory for typed UZON values.

    Numeric types are resolved via ``__getattr__``::

        val.i32(10)   → UzonInt(10, 'i32')
        val.u16(8080) → UzonInt(8080, 'u16')
        val.f64(1.5)  → UzonFloat(1.5, 'f64')

    Variant types have explicit methods: ``enum()``, ``union()``, ``tagged()``.
    """

    # ── numeric types via __getattr__ ─────────────────────────────

    def __getattr__(self, name: str) -> Any:
        m = _INT_TYPE_RE.match(name)
        if m:
            signed = m.group(1) == 'i'
            width = int(m.group(2))
            if signed:
                lo, hi = -(1 << (width - 1)), (1 << (width - 1)) - 1
            else:
                lo, hi = 0, (1 << width) - 1

            def make_int(value: int) -> UzonInt:
                if not isinstance(value, int) or isinstance(value, bool):
                    raise TypeError(f"val.{name}() expects int, got {type(value).__name__}")
                if not (lo <= value <= hi):
                    raise OverflowError(
                        f"{value} out of {name} range [{lo}, {hi}]"
                    )
                return UzonInt(value, name)

            make_int.__name__ = make_int.__qualname__ = name
            return make_int

        if name in _FLOAT_TYPES:
            def make_float(value: float | int) -> UzonFloat:
                if isinstance(value, bool):
                    raise TypeError(f"val.{name}() expects number, got bool")
                if not isinstance(value, (int, float)):
                    raise TypeError(f"val.{name}() expects number, got {type(value).__name__}")
                return UzonFloat(float(value), name)

            make_float.__name__ = make_float.__qualname__ = name
            return make_float

        raise AttributeError(f"Unknown UZON type: '{name}'")

    def __dir__(self) -> list[str]:
        common = [
            'i8', 'i16', 'i32', 'i64',
            'u8', 'u16', 'u32', 'u64',
            'f16', 'f32', 'f64', 'f80', 'f128',
        ]
        methods = ['struct', 'enum', 'union', 'tagged']
        return common + methods

    # ── struct type ────────────────────────────────────────────────

    @staticmethod
    def struct(
        fields: dict[str, Any],
        *,
        type_name: str | None = None,
    ) -> UzonStruct:
        """Create a UZON named struct.

        Args:
            fields: Dict of field names to values.
            type_name: Optional named type (e.g. "Server").

        Returns:
            A UzonStruct (dict subclass) with type_name metadata.
        """
        return UzonStruct(fields, type_name)

    # ── variant types ─────────────────────────────────────────────

    @staticmethod
    def enum(
        value: str,
        variants: list[str],
        *,
        type_name: str | None = None,
    ) -> UzonEnum:
        """Create a UZON enum value.

        Args:
            value: The active variant name.
            variants: All allowed variant names.
            type_name: Optional named type (e.g. "Color").

        Raises:
            ValueError: If *value* is not in *variants*.
        """
        if value not in variants:
            raise ValueError(
                f"'{value}' is not a valid variant of "
                f"{type_name or 'enum'} — expected one of {variants}"
            )
        return UzonEnum(value, variants, type_name)

    @staticmethod
    def union(
        value: Any,
        types: list[str],
        *,
        type_name: str | None = None,
    ) -> UzonUnion:
        """Create a UZON untagged union value.

        Args:
            value: The inner value.
            types: List of allowed type names.
            type_name: Optional named type (e.g. "StringOrInt").
        """
        return UzonUnion(value, types, type_name)

    @staticmethod
    def tagged(
        tag: str,
        value: Any,
        variants: dict[str, str | None],
        *,
        type_name: str | None = None,
    ) -> UzonTaggedUnion:
        """Create a UZON tagged union value.

        Args:
            tag: The active variant tag.
            value: The inner value.
            variants: Map of variant names to their payload types.
            type_name: Optional named type (e.g. "Result").

        Raises:
            ValueError: If *tag* is not in *variants*.
        """
        if tag not in variants:
            raise ValueError(
                f"'{tag}' is not a valid variant of "
                f"{type_name or 'tagged union'} — expected one of {list(variants)}"
            )
        return UzonTaggedUnion(value, tag, variants, type_name)

    def __repr__(self) -> str:
        return "uzon.val"
