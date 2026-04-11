# SPDX-FileCopyrightText: © 2026 Suho Kang
# SPDX-License-Identifier: MIT
"""UZON parser and generator for Python — spec v0.5."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .errors import UzonCircularError, UzonRuntimeError, UzonSyntaxError, UzonTypeError
from .evaluator import Evaluator
from .generator import generate
from .lexer import Lexer
from .parser import Parser
from .types import (
    UzonBuiltinFunction, UzonEnum, UzonFloat, UzonFunction, UzonInt,
    UzonStruct, UzonTaggedUnion, UzonTypedList, UzonUndefined, UzonUnion,
)
from .val import _ValFactory

__version__ = "0.5.0"

val = _ValFactory()

__all__ = [
    # Core API
    "loads",
    "dumps",
    "load",
    "dump",
    "json_default",
    "val",
    # Errors
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
    "UzonUndefined",
]


def loads(text: str, *, plain: bool = False) -> dict[str, Any]:
    """Parse and evaluate a UZON string.

    Args:
        text: UZON source text.
        plain: If True, strip all type wrappers to plain Python equivalents.

    Returns:
        A dict of evaluated bindings.
    """
    tokens = Lexer(text).tokenize()
    doc = Parser(tokens).parse()
    result = Evaluator().evaluate(doc)
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
    """
    path = Path(source)
    text = path.read_text(encoding="utf-8")
    tokens = Lexer(text).tokenize()
    doc = Parser(tokens).parse()
    result = Evaluator(filename=str(path)).evaluate(doc)
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
