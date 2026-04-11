# SPDX-FileCopyrightText: © 2026 Suho Kang
# SPDX-License-Identifier: MIT
"""UZON generator — converts Python values to UZON text.

Produces valid UZON source from evaluated Python values (§4, §8).
"""

from __future__ import annotations

from typing import Any

from ._format import format_float as _format_float
from .tokens import ALL_KEYWORDS
from .types import (
    UzonBuiltinFunction, UzonEnum, UzonFloat, UzonFunction,
    UzonInt, UzonTaggedUnion, UzonTypedList, UzonUndefined, UzonUnion,
)

# §2.3: Characters that require a quoted identifier
_NEEDS_QUOTING = set(' \t\n\r!@#$%^&*()-+=[]{}|;:\'",.<>?/\\')


def generate(value: dict[str, Any], indent: int = 4) -> str:
    """Generate UZON text from a Python dict (top-level struct).

    Args:
        value: A dict representing a UZON document.
        indent: Number of spaces per indentation level.

    Returns:
        A string of UZON text.
    """
    if not isinstance(value, dict):
        raise TypeError(f"Top-level value must be a dict, got {type(value).__name__}")
    gen = _Generator(indent=indent)
    return gen.emit_bindings(value, level=0)


class _Generator:
    def __init__(self, indent: int = 4):
        self._indent = indent
        self._emitted_types: set[str] = set()

    def emit_bindings(self, d: dict[str, Any], level: int) -> str:
        """§8: Emit bindings separated by newlines."""
        lines: list[str] = []
        prefix = " " * (self._indent * level)
        for key, val in d.items():
            name = self._emit_name(key)
            val_str = self._emit_value(val, level, in_collection=False)
            lines.append(f"{prefix}{name} is {val_str}")
        return "\n".join(lines)

    def _emit_name(self, name: str) -> str:
        """Emit an identifier, quoting or @-escaping if needed (§2.3, §2.4)."""
        if not name:
            return "''"
        # §2.4: Keywords need @ escape
        if name in ALL_KEYWORDS:
            return f"@{name}"
        # §2.3: Special characters need quoting (quoted idents are literal, no escaping)
        if any(c in _NEEDS_QUOTING for c in name):
            return f"'{name}'"
        return name

    def _emit_value(self, val: Any, level: int, *, in_collection: bool = False) -> str:
        """Emit a UZON value."""
        if val is None:
            return "null"
        if val is UzonUndefined:
            return "undefined"
        if isinstance(val, bool):
            return "true" if val else "false"
        if isinstance(val, UzonEnum):
            return self._emit_enum(val, in_collection=in_collection)
        if isinstance(val, UzonTaggedUnion):
            return self._emit_tagged_union(val, level, in_collection=in_collection)
        if isinstance(val, UzonUnion):
            return self._emit_union(val, level, in_collection=in_collection)
        if isinstance(val, (UzonFunction, UzonBuiltinFunction)):
            raise TypeError("Cannot generate UZON for function values")
        if isinstance(val, UzonInt):
            s = str(int(val))
            return f"{s} as {val.type_name}" if not val.adoptable else s
        if isinstance(val, UzonFloat):
            s = self._emit_float(float(val))
            return f"{s} as {val.type_name}" if not val.adoptable else s
        if isinstance(val, int):
            return str(val)
        if isinstance(val, float):
            return self._emit_float(val)
        if isinstance(val, str):
            return self._emit_string(val)
        if isinstance(val, dict):
            return self._emit_struct(val, level)
        if isinstance(val, list):
            return self._emit_list(val, level)
        if isinstance(val, tuple):
            return self._emit_tuple(val, level)
        raise TypeError(f"Cannot generate UZON for {type(val).__name__}")

    def _emit_float(self, val: float) -> str:
        """§5.11.2: Format float using spec formatting rules."""
        return _format_float(val)

    def _emit_string(self, val: str) -> str:
        """§4.4: Emit a string literal with proper escaping."""
        escaped = val.replace("\\", "\\\\")
        escaped = escaped.replace('"', '\\"')
        escaped = escaped.replace("\n", "\\n")
        escaped = escaped.replace("\r", "\\r")
        escaped = escaped.replace("\t", "\\t")
        escaped = escaped.replace("\0", "\\0")
        escaped = escaped.replace("{", "\\{")
        return f'"{escaped}"'

    def _emit_struct(self, d: dict[str, Any], level: int) -> str:
        """§3.2: Emit struct literal."""
        if not d:
            return "{}"
        # Try single-line for small structs without nested compounds
        if self._is_simple_struct(d):
            parts = [f"{self._emit_name(k)} is {self._emit_value(v, level + 1)}"
                     for k, v in d.items()]
            one_line = "{ " + ", ".join(parts) + " }"
            if len(one_line) <= 80:
                return one_line
        # §8: Multi-line — newline separates bindings
        inner = self.emit_bindings(d, level + 1)
        prefix = " " * (self._indent * level)
        return "{\n" + inner + "\n" + prefix + "}"

    def _is_simple_struct(self, d: dict[str, Any]) -> bool:
        """Check if a struct can be rendered on a single line."""
        if len(d) > 4:
            return False
        for v in d.values():
            if isinstance(v, (dict, list, tuple)):
                return False
        return True

    def _emit_list(self, lst: list[Any], level: int) -> str:
        """§3.4: Emit list literal with type annotation when needed."""
        type_suffix = self._list_type_suffix(lst)
        if not lst:
            return f"[] as [{type_suffix}]" if type_suffix else "[]"
        parts = [self._emit_value(v, level, in_collection=True) for v in lst]
        one_line = "[ " + ", ".join(parts) + " ]"
        if type_suffix:
            one_line += f" as [{type_suffix}]"
        if len(one_line) <= 80 and "\n" not in one_line:
            return one_line
        prefix = " " * (self._indent * (level + 1))
        inner_prefix = " " * (self._indent * level)
        inner_parts = [f"{prefix}{self._emit_value(v, level + 1, in_collection=True)}" for v in lst]
        result = "[\n" + ",\n".join(inner_parts) + "\n" + inner_prefix + "]"
        if type_suffix:
            result += f" as [{type_suffix}]"
        return result

    @staticmethod
    def _list_type_suffix(lst: list[Any]) -> str | None:
        """Determine the type annotation suffix for a list, if needed."""
        if isinstance(lst, UzonTypedList) and lst.element_type:
            return lst.element_type
        return None

    def _emit_tuple(self, t: tuple[Any, ...], level: int) -> str:
        """§3.4.1: Emit tuple literal."""
        if not t:
            return "()"
        if len(t) == 1:
            return f"({self._emit_value(t[0], level, in_collection=True)},)"
        parts = [self._emit_value(v, level, in_collection=True) for v in t]
        one_line = "(" + ", ".join(parts) + ")"
        if len(one_line) <= 80 and "\n" not in one_line:
            return one_line
        prefix = " " * (self._indent * (level + 1))
        inner_prefix = " " * (self._indent * level)
        inner_parts = [f"{prefix}{self._emit_value(v, level + 1, in_collection=True)}" for v in t]
        return "(\n" + ",\n".join(inner_parts) + "\n" + inner_prefix + ")"

    # ── variant types ─────────────────────────────────────────────

    def _emit_enum(self, val: UzonEnum, *, in_collection: bool = False) -> str:
        """§3.5: Emit enum value — reference via `as` if type already emitted."""
        esc = self._escape_variant
        if val.type_name and (val.type_name in self._emitted_types or in_collection):
            return f"{esc(val.value)} as {val.type_name}"
        parts = [esc(val.value), "from", ", ".join(esc(v) for v in val.variants)]
        if val.type_name:
            parts.extend(["called", val.type_name])
            self._emitted_types.add(val.type_name)
        return " ".join(parts)

    def _emit_union(self, val: UzonUnion, level: int, *, in_collection: bool = False) -> str:
        """§3.6: Emit union value."""
        inner = self._emit_value(val.value, level)
        if val.type_name and (val.type_name in self._emitted_types or in_collection):
            return f"{inner} as {val.type_name}"
        types_str = ", ".join(val.types)
        parts = [inner, "from union", types_str]
        if val.type_name:
            parts.extend(["called", val.type_name])
            self._emitted_types.add(val.type_name)
        return " ".join(parts)

    def _emit_tagged_union(self, val: UzonTaggedUnion, level: int, *, in_collection: bool = False) -> str:
        """§3.7: Emit tagged union value — reference via `as` if type already emitted."""
        esc = self._escape_variant
        inner = self._emit_value(val.value, level)
        if val.type_name and (val.type_name in self._emitted_types or in_collection):
            return f"{inner} as {val.type_name} named {esc(val.tag)}"
        variant_parts = [f"{esc(name)} as {type_name}" for name, type_name in val.variants.items()
                         if type_name]
        parts = [inner, "named", esc(val.tag)]
        if variant_parts:
            parts.extend(["from", ", ".join(variant_parts)])
        if val.type_name:
            parts.extend(["called", val.type_name])
            self._emitted_types.add(val.type_name)
        return " ".join(parts)

    @staticmethod
    def _escape_variant(name: str) -> str:
        """§2.4: Escape a variant/tag name if it's a keyword."""
        if name in ALL_KEYWORDS:
            return f"@{name}"
        return name
