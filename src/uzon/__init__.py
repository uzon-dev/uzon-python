# SPDX-FileCopyrightText: © 2026 Suho Kang
# SPDX-License-Identifier: MIT
"""UZON parser and generator for Python — spec v0.8."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .errors import UzonCircularError, UzonError, UzonRuntimeError, UzonSyntaxError, UzonTypeError
from .evaluator import Evaluator
from .generator import generate
from .lexer import Lexer
from .parser import Parser
from .types import (
    UzonBuiltinFunction, UzonEnum, UzonFloat, UzonFunction, UzonInt,
    UzonStruct, UzonTaggedUnion, UzonTypedList, UzonUndefined, UzonUnion,
)
from .val import _ValFactory

__version__ = "0.8.1"

val = _ValFactory()

__all__ = [
    # Core API
    "loads",
    "dumps",
    "load",
    "dump",
    "json_default",
    "merge",
    "pretty_format",
    "val",
    # Errors
    "UzonError",
    "UzonSyntaxError",
    "UzonTypeError",
    "UzonRuntimeError",
    "UzonCircularError",
    # Types
    "UzonInt",
    "UzonFloat",
    "UzonEnum",
    "UzonUnion",
    "UzonStruct",
    "UzonTaggedUnion",
    "UzonTypedList",
    "UzonFunction",
    "UzonBuiltinFunction",
    "UzonUndefined",
]


def loads(text: str, *, plain: bool = False) -> dict[str, Any]:
    """Parse and evaluate a UZON string.

    Args:
        text: UZON source text.
        plain: If True, strip all type wrappers to plain Python equivalents.

    Returns:
        A dict of evaluated bindings.

    Raises:
        UzonError: On failure. The ``errors`` attribute contains all
            individual errors when multiple problems are detected.
    """
    tokens = Lexer(text).tokenize()
    doc = Parser(tokens).parse()
    evaluator = Evaluator()
    try:
        result = evaluator.evaluate(doc)
    except UzonError as e:
        if evaluator._collected_errors and not e.errors:
            e.errors = list(evaluator._collected_errors)
        raise
    if plain:
        return _to_plain(result)
    return result


def dumps(value: dict[str, Any], *, indent: int = 4) -> str:
    """Generate UZON text from a Python dict.

    Args:
        value: A dict representing a UZON document.
        indent: Number of spaces per indentation level.

    Returns:
        A string of UZON text.
    """
    return generate(value, indent=indent)


def load(source: str | Path, *, plain: bool = False) -> dict[str, Any]:
    """Parse and evaluate a UZON file.

    Args:
        source: Path to a .uzon file (str or Path).
        plain: If True, strip all type wrappers.

    Returns:
        A dict of evaluated bindings.

    Raises:
        UzonError: On failure. The ``errors`` attribute contains all
            individual errors when multiple problems are detected.
    """
    path = Path(source).resolve()  # realpath for consistent import detection
    text = path.read_text(encoding="utf-8")
    tokens = Lexer(text).tokenize()
    doc = Parser(tokens).parse()
    evaluator = Evaluator(filename=str(path))
    try:
        result = evaluator.evaluate(doc)
    except UzonError as e:
        if evaluator._collected_errors and not e.errors:
            e.errors = list(evaluator._collected_errors)
        raise
    if plain:
        return _to_plain(result)
    return result


def dump(value: dict[str, Any], dest: str | Path, *, indent: int = 4) -> None:
    """Write UZON text to a file.

    Args:
        value: A dict representing a UZON document.
        dest: Path to write to (str or Path).
        indent: Number of spaces per indentation level.
    """
    text = generate(value, indent=indent)
    Path(dest).write_text(text + "\n", encoding="utf-8")


# ── plain mode ─────────────────────────────────────────────────────

def _to_plain(value: Any) -> Any:
    """Recursively strip type wrappers to plain Python types."""
    if isinstance(value, UzonEnum):
        return value.value
    if isinstance(value, UzonTaggedUnion):
        return _to_plain(value.value)
    if isinstance(value, UzonUnion):
        return _to_plain(value.value)
    if isinstance(value, UzonInt):
        return int(value)
    if isinstance(value, UzonFloat):
        return float(value)
    if isinstance(value, dict):
        return {k: _to_plain(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_plain(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_to_plain(v) for v in value)
    return value


# ── JSON serialization ────────────────────────────────────────────


def json_default(obj: Any) -> Any:
    """JSON serialization hook for UZON types.

    Usage::

        import json, uzon
        data = uzon.loads(text)
        json.dumps(data, default=uzon.json_default)

    Mapping:
        UzonInt / UzonFloat → number (handled natively by json)
        UzonEnum → string (variant name)
        UzonUnion → inner value (recursive)
        UzonTaggedUnion → ``{"_tag": str, "_value": ...}``
        UzonUndefined → None
        UzonFunction → raises TypeError
    """
    if isinstance(obj, UzonEnum):
        return obj.value
    if isinstance(obj, UzonUnion):
        return obj.value
    if isinstance(obj, UzonTaggedUnion):
        return {"_tag": obj.tag, "_value": obj.value}
    if obj is UzonUndefined:
        return None
    if isinstance(obj, (UzonFunction, UzonBuiltinFunction)):
        raise TypeError(f"UZON {type(obj).__name__} is not JSON serializable")
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


# ── deep merge ───────────────────────────────────────────────────


def merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge two UZON structs (dicts), returning a new dict.

    - Nested dicts are merged recursively.
    - All other values in *override* replace *base*.
    - Type metadata (UzonStruct.type_name) is preserved from *override*
      if present, otherwise from *base*.

    Usage::

        base = uzon.loads('host is "localhost", port is 8080')
        override = uzon.loads('port is 9090, debug is true')
        merged = uzon.merge(base, override)
        # → {"host": "localhost", "port": 9090, "debug": True}
    """
    if not isinstance(base, dict):
        raise TypeError(f"merge() base must be a dict, got {type(base).__name__}")
    if not isinstance(override, dict):
        raise TypeError(f"merge() override must be a dict, got {type(override).__name__}")
    return _merge_dicts(base, override)


def _merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge two dicts."""
    # Determine type_name for result
    base_tn = base.type_name if isinstance(base, UzonStruct) else None
    over_tn = override.type_name if isinstance(override, UzonStruct) else None
    result_tn = over_tn or base_tn

    merged: dict[str, Any] = {}
    for key in base:
        if key in override:
            merged[key] = _merge_values(base[key], override[key])
        else:
            merged[key] = base[key]
    for key in override:
        if key not in base:
            merged[key] = override[key]

    if result_tn:
        return UzonStruct(merged, result_tn)
    return merged


def _merge_values(base_val: Any, over_val: Any) -> Any:
    """Merge a single value: recurse into dicts, otherwise override wins."""
    if isinstance(base_val, dict) and isinstance(over_val, dict):
        return _merge_dicts(base_val, over_val)
    return over_val


# ── pretty format ────────────────────────────────────────────────


def pretty_format(value: Any, *, indent: int = 2) -> str:
    """Pretty-format a UZON value for debugging.

    Unlike ``dumps()`` which produces valid UZON source, ``pformat()``
    produces a human-readable representation that shows type information.

    Usage::

        data = uzon.loads(text)
        print(uzon.pformat(data))
    """
    lines: list[str] = []
    _pformat_value(value, lines, level=0, indent=indent)
    return "\n".join(lines)


def _pformat_value(value: Any, lines: list[str], level: int, indent: int) -> None:
    """Recursively format a value into lines."""
    pad = " " * (indent * level)

    if isinstance(value, UzonTaggedUnion):
        tag_info = f" (tag={value.tag!r})" if value.tag else ""
        type_info = f" as {value.type_name}" if value.type_name else ""
        inner_repr = _pformat_inline(value.value, indent)
        lines.append(f"{pad}TaggedUnion{tag_info}{type_info}: {inner_repr}")
        return

    if isinstance(value, UzonUnion):
        type_info = f" as {value.type_name}" if value.type_name else ""
        inner_repr = _pformat_inline(value.value, indent)
        lines.append(f"{pad}Union{type_info}: {inner_repr}")
        return

    if isinstance(value, UzonEnum):
        type_info = f" as {value.type_name}" if value.type_name else ""
        lines.append(f"{pad}Enum{type_info}: {value.value}")
        return

    if isinstance(value, dict):
        type_info = ""
        if isinstance(value, UzonStruct) and value.type_name:
            type_info = f" as {value.type_name}"
        if not value:
            lines.append(f"{pad}{{}}{type_info}")
            return
        lines.append(f"{pad}{{{type_info}")
        for k, v in value.items():
            child_pad = " " * (indent * (level + 1))
            if _is_simple(v):
                lines.append(f"{child_pad}{k}: {_pformat_scalar(v)}")
            else:
                lines.append(f"{child_pad}{k}:")
                _pformat_value(v, lines, level + 2, indent)
        lines.append(f"{pad}}}")
        return

    if isinstance(value, list):
        if not value:
            lines.append(f"{pad}[]")
            return
        if all(_is_simple(e) for e in value):
            items = ", ".join(_pformat_scalar(e) for e in value)
            one_line = f"[{items}]"
            if len(one_line) <= 80:
                lines.append(f"{pad}{one_line}")
                return
        lines.append(f"{pad}[")
        for e in value:
            if _is_simple(e):
                lines.append(f"{pad}{' ' * indent}{_pformat_scalar(e)}")
            else:
                _pformat_value(e, lines, level + 1, indent)
        lines.append(f"{pad}]")
        return

    if isinstance(value, tuple):
        if not value:
            lines.append(f"{pad}()")
            return
        items = ", ".join(_pformat_scalar(e) if _is_simple(e) else "..." for e in value)
        lines.append(f"{pad}({items})")
        return

    lines.append(f"{pad}{_pformat_scalar(value)}")


def _is_simple(value: Any) -> bool:
    """Check if a value is a scalar (not compound)."""
    if isinstance(value, (dict, list, tuple)):
        return False
    if isinstance(value, (UzonUnion, UzonTaggedUnion)):
        return _is_simple(value.value)
    return True


def _pformat_scalar(value: Any) -> str:
    """Format a scalar value with type info."""
    if value is None:
        return "null"
    if value is UzonUndefined:
        return "undefined"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, UzonInt):
        if value.adoptable:
            return str(int(value))
        return f"{int(value)} as {value.type_name}"
    if isinstance(value, UzonFloat):
        if value.adoptable:
            return str(float(value))
        return f"{float(value)} as {value.type_name}"
    if isinstance(value, UzonEnum):
        type_info = f" as {value.type_name}" if value.type_name else ""
        return f"{value.value}{type_info}"
    if isinstance(value, UzonUnion):
        type_info = f" as {value.type_name}" if value.type_name else ""
        return f"{_pformat_scalar(value.value)}{type_info}"
    if isinstance(value, UzonTaggedUnion):
        type_info = f" as {value.type_name}" if value.type_name else ""
        return f"{_pformat_scalar(value.value)} (tag={value.tag!r}){type_info}"
    if isinstance(value, str):
        return repr(value)
    return str(value)


def _pformat_inline(value: Any, indent: int) -> str:
    """Format a value as inline (single line) for union/tagged union display."""
    if _is_simple(value):
        return _pformat_scalar(value)
    if isinstance(value, dict) and len(value) <= 3:
        parts = [f"{k}: {_pformat_scalar(v)}" for k, v in value.items() if _is_simple(v)]
        if len(parts) == len(value):
            return "{ " + ", ".join(parts) + " }"
    if isinstance(value, list) and len(value) <= 5 and all(_is_simple(e) for e in value):
        return "[" + ", ".join(_pformat_scalar(e) for e in value) + "]"
    return f"({type(value).__name__}, {len(value)} items)"
