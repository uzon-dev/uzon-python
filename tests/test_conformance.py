# SPDX-FileCopyrightText: © 2026 Suho Kang
# SPDX-License-Identifier: MIT
"""Conformance test suite — runs all tests from ../conformance/."""

from __future__ import annotations

import math
import os
from pathlib import Path

import pytest

import uzon
from uzon.types import UzonEnum, UzonFloat, UzonInt, UzonTaggedUnion, UzonUnion

CONFORMANCE_DIR = Path(__file__).resolve().parent.parent.parent / "conformance"


def _skip_if_missing():
    if not CONFORMANCE_DIR.is_dir():
        pytest.skip(f"Conformance directory not found: {CONFORMANCE_DIR}")


# ── helpers ───────────────────────────────────────────────────────


def _nan_aware_eq(a: object, b: object) -> bool:
    """Deep equality that handles NaN, UzonInt, UzonFloat, etc."""
    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) and math.isnan(b):
            return True
        return a == b
    if isinstance(a, UzonEnum) and isinstance(b, UzonEnum):
        return a.value == b.value and a.type_name == b.type_name
    if isinstance(a, UzonUnion) and isinstance(b, UzonUnion):
        return _nan_aware_eq(a.value, b.value)
    if isinstance(a, UzonTaggedUnion) and isinstance(b, UzonTaggedUnion):
        return a.tag == b.tag and _nan_aware_eq(a.value, b.value)
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return False
        return all(_nan_aware_eq(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return False
        return all(_nan_aware_eq(x, y) for x, y in zip(a, b))
    return a == b


# ── eval tests ────────────────────────────────────────────────────


def _collect_eval_tests():
    _dir = CONFORMANCE_DIR / "eval"
    if not _dir.is_dir():
        return []
    sources = sorted(f for f in _dir.iterdir() if f.suffix == ".uzon" and ".expected" not in f.name)
    return [f.stem for f in sources if (f.parent / f"{f.stem}.expected.uzon").exists()]


@pytest.mark.parametrize("name", _collect_eval_tests())
def test_eval(name: str):
    _skip_if_missing()
    src_file = CONFORMANCE_DIR / "eval" / f"{name}.uzon"
    exp_file = CONFORMANCE_DIR / "eval" / f"{name}.expected.uzon"

    result = uzon.load(src_file)
    expected = uzon.load(exp_file)

    for key, exp_val in expected.items():
        assert key in result, f"Missing key '{key}' in result"
        assert _nan_aware_eq(result[key], exp_val), (
            f"Key '{key}': expected {exp_val!r}, got {result[key]!r}"
        )


# ── parse/valid tests ────────────────────────────────────────────


def _collect_parse_valid():
    _dir = CONFORMANCE_DIR / "parse" / "valid"
    if not _dir.is_dir():
        return []
    return sorted(f.stem for f in _dir.iterdir() if f.suffix == ".uzon")


@pytest.mark.parametrize("name", _collect_parse_valid())
def test_parse_valid(name: str):
    _skip_if_missing()
    src_file = CONFORMANCE_DIR / "parse" / "valid" / f"{name}.uzon"
    uzon.load(src_file)  # should not raise


# ── parse/invalid tests ──────────────────────────────────────────


def _collect_parse_invalid():
    _dir = CONFORMANCE_DIR / "parse" / "invalid"
    if not _dir.is_dir():
        return []
    return sorted(f.stem for f in _dir.iterdir() if f.suffix == ".uzon")


@pytest.mark.parametrize("name", _collect_parse_invalid())
def test_parse_invalid(name: str):
    _skip_if_missing()
    src_file = CONFORMANCE_DIR / "parse" / "invalid" / f"{name}.uzon"
    with pytest.raises(Exception):
        uzon.load(src_file)


# ── roundtrip tests ──────────────────────────────────────────────


def _collect_roundtrip():
    _dir = CONFORMANCE_DIR / "roundtrip"
    if not _dir.is_dir():
        return []
    return sorted(f.stem for f in _dir.iterdir() if f.suffix == ".uzon")


@pytest.mark.parametrize("name", _collect_roundtrip())
def test_roundtrip(name: str):
    _skip_if_missing()
    src_file = CONFORMANCE_DIR / "roundtrip" / f"{name}.uzon"
    text = src_file.read_text(encoding="utf-8")

    original = uzon.loads(text)
    regenerated = uzon.dumps(original)
    reparsed = uzon.loads(regenerated)

    assert _nan_aware_eq(original, reparsed), (
        f"Roundtrip mismatch:\n"
        f"  original keys: {sorted(original.keys())}\n"
        f"  reparsed keys: {sorted(reparsed.keys())}"
    )
