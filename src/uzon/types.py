# SPDX-FileCopyrightText: © 2026 Suho Kang
# SPDX-License-Identifier: MIT
"""UZON typed value wrappers per §3, §5.

UzonUndefined — §3.1: state sentinel for missing values.
UzonInt — §4.2/§5: typed integer with width annotation and adoptable flag.
UzonFloat — §4.3/§5: typed float with width annotation and adoptable flag.
UzonEnum — §3.5: enum value with variant set and type name.
UzonUnion — §3.6: untagged union value.
UzonTaggedUnion — §3.7: tagged union value.
UzonFunction — §3.8: function value (closure).
UzonBuiltinFunction — §5.16: std library built-in function.
"""

from __future__ import annotations

import copy as _copy


class _UzonUndefinedType:
    """§3.1: Singleton sentinel for UZON undefined state."""

    _instance: _UzonUndefinedType | None = None

    def __new__(cls) -> _UzonUndefinedType:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "UzonUndefined"

    def __bool__(self) -> bool:
        return False

    def __copy__(self) -> _UzonUndefinedType:
        return self

    def __deepcopy__(self, memo: dict) -> _UzonUndefinedType:
        return self


UzonUndefined = _UzonUndefinedType()


class UzonInt(int):
    """§4.2/§5: Typed UZON integer preserving width annotation.

    adoptable=True means untyped literal (defaults to i64) that can adopt
    another integer type when combined with a typed operand (§5 untyped literal compatibility).

    Arithmetic operators preserve type_name::

        >>> x = UzonInt(10, 'i32')
        >>> x + 5
        UzonInt(15, 'i32')
    """

    type_name: str
    adoptable: bool

    def __new__(cls, value: int, type_name: str, *, adoptable: bool = False) -> UzonInt:
        obj = super().__new__(cls, value)
        obj.type_name = type_name
        obj.adoptable = adoptable
        return obj

    def __repr__(self) -> str:
        return f"UzonInt({int(self)}, {self.type_name!r})"

    def __str__(self) -> str:
        return int.__repr__(self)

    def __copy__(self) -> UzonInt:
        return self  # immutable

    def __deepcopy__(self, memo: dict) -> UzonInt:
        return self  # immutable

    def to_plain(self) -> int:
        return int(self)


class UzonFloat(float):
    """§4.3/§5: Typed UZON float preserving width annotation.

    adoptable=True means untyped literal (defaults to f64) that can adopt
    another float type when combined with a typed operand.

    Arithmetic operators preserve type_name::

        >>> x = UzonFloat(1.5, 'f32')
        >>> x * 2.0
        UzonFloat(3.0, 'f32')
    """

    type_name: str
    adoptable: bool

    def __new__(cls, value: float, type_name: str, *, adoptable: bool = False) -> UzonFloat:
        obj = super().__new__(cls, value)
        obj.type_name = type_name
        obj.adoptable = adoptable
        return obj

    def __repr__(self) -> str:
        return f"UzonFloat({float(self)}, {self.type_name!r})"

    def __str__(self) -> str:
        return float.__repr__(self)

    def __copy__(self) -> UzonFloat:
        return self  # immutable

    def __deepcopy__(self, memo: dict) -> UzonFloat:
        return self  # immutable

    def to_plain(self) -> float:
        return float(self)


# ── type-preserving arithmetic for UzonInt / UzonFloat ─────────


def _make_int_op(name: str):
    """Generate an arithmetic method for UzonInt that preserves type_name."""
    base = getattr(int, name)

    def method(self, *args):
        result = base(self, *args)
        if result is NotImplemented:
            return result
        if isinstance(result, int) and not isinstance(result, bool):
            return UzonInt(result, self.type_name)
        return result

    method.__name__ = method.__qualname__ = name
    return method


def _make_float_op(name: str):
    """Generate an arithmetic method for UzonFloat that preserves type_name."""
    base = getattr(float, name)

    def method(self, *args):
        result = base(self, *args)
        if result is NotImplemented:
            return result
        if isinstance(result, float):
            return UzonFloat(result, self.type_name)
        return result

    method.__name__ = method.__qualname__ = name
    return method


for _op in (
    "__add__", "__radd__", "__sub__", "__rsub__",
    "__mul__", "__rmul__", "__floordiv__", "__rfloordiv__",
    "__mod__", "__rmod__", "__pow__", "__rpow__",
    "__and__", "__rand__", "__or__", "__ror__",
    "__xor__", "__rxor__", "__lshift__", "__rlshift__",
    "__rshift__", "__rrshift__",
    "__neg__", "__pos__", "__abs__", "__invert__",
    "__round__",
):
    setattr(UzonInt, _op, _make_int_op(_op))

for _op in (
    "__add__", "__radd__", "__sub__", "__rsub__",
    "__mul__", "__rmul__", "__truediv__", "__rtruediv__",
    "__floordiv__", "__rfloordiv__", "__mod__", "__rmod__",
    "__pow__", "__rpow__",
    "__neg__", "__pos__", "__abs__",
    "__round__",
):
    setattr(UzonFloat, _op, _make_float_op(_op))


# ── collection types ───────────────────────────────────────────


class UzonStruct(dict):
    """Dict subclass that preserves a named type annotation for round-trip fidelity.

    Used when a struct has an explicit ``called TypeName`` annotation (§6.2),
    so the generator can re-emit the type name.
    """

    type_name: str | None

    def __init__(self, mapping: dict | None = None, type_name: str | None = None, **kwargs):
        super().__init__(mapping or {}, **kwargs)
        self.type_name = type_name

    def __repr__(self) -> str:
        return f"UzonStruct({dict.__repr__(self)}, type_name={self.type_name!r})"

    def __copy__(self) -> UzonStruct:
        return UzonStruct(dict(self), self.type_name)

    def __deepcopy__(self, memo: dict) -> UzonStruct:
        return UzonStruct(
            {k: _copy.deepcopy(v, memo) for k, v in self.items()},
            self.type_name,
        )


class UzonTypedList(list):
    """List that preserves element type annotation for round-trip fidelity.

    Used when a list has an explicit `as [Type]` annotation (§6.1), so the
    generator can re-emit the annotation even for empty or all-null lists.
    """

    element_type: str | None

    def __init__(self, elements: list, element_type: str | None = None):
        super().__init__(elements)
        self.element_type = element_type

    def __repr__(self) -> str:
        return f"UzonTypedList({list.__repr__(self)}, element_type={self.element_type!r})"

    def __copy__(self) -> UzonTypedList:
        return UzonTypedList(list(self), self.element_type)

    def __deepcopy__(self, memo: dict) -> UzonTypedList:
        return UzonTypedList([_copy.deepcopy(e, memo) for e in self], self.element_type)


# ── variant types ──────────────────────────────────────────────


class UzonEnum:
    """§3.5: Enum value — one variant from a defined set."""

    __slots__ = ("value", "variants", "type_name")
    __match_args__ = ("value",)

    def __init__(self, value: str, variants: list[str], type_name: str | None = None):
        self.value = value
        self.variants = variants
        self.type_name = type_name

    def __eq__(self, other: object) -> bool:
        if isinstance(other, UzonEnum):
            return self.value == other.value and self.type_name == other.type_name
        if isinstance(other, str):
            return self.value == other
        return NotImplemented

    def __hash__(self) -> int:
        return hash((self.value, self.type_name))

    def __repr__(self) -> str:
        name = self.type_name or "anonymous"
        return f"UzonEnum({self.value!r}, type={name})"

    def __str__(self) -> str:
        return self.value

    def __copy__(self) -> UzonEnum:
        return UzonEnum(self.value, self.variants, self.type_name)

    def __deepcopy__(self, memo: dict) -> UzonEnum:
        return UzonEnum(self.value, list(self.variants), self.type_name)

    def to_plain(self) -> str:
        return self.value


class UzonUnion:
    """§3.6: Untagged union — value with one of several possible types.

    §3.6: Equality compares inner values only — no tag to compare.
    Transparent access: ``[]``, ``len()``, ``iter()``, ``in`` delegate to inner value.
    """

    __slots__ = ("value", "types", "type_name")
    __match_args__ = ("value",)

    def __init__(self, value: object, types: list[str], type_name: str | None = None):
        self.value = value
        self.types = types
        self.type_name = type_name

    def __eq__(self, other: object) -> bool:
        if isinstance(other, UzonUnion):
            return self.value == other.value
        return self.value == other

    def __hash__(self) -> int:
        return hash(self.value)

    def __repr__(self) -> str:
        name = self.type_name or "anonymous"
        return f"UzonUnion({self.value!r}, types={self.types}, type={name})"

    def __str__(self) -> str:
        return str(self.value)

    def __bool__(self) -> bool:
        return bool(self.value)

    def __getitem__(self, key):
        return self.value[key]

    def __len__(self) -> int:
        return len(self.value)

    def __iter__(self):
        return iter(self.value)

    def __contains__(self, item) -> bool:
        return item in self.value

    def __copy__(self) -> UzonUnion:
        return UzonUnion(self.value, self.types, self.type_name)

    def __deepcopy__(self, memo: dict) -> UzonUnion:
        return UzonUnion(
            _copy.deepcopy(self.value, memo),
            list(self.types),
            self.type_name,
        )

    def to_plain(self) -> object:
        return self.value


class UzonTaggedUnion:
    """§3.7: Tagged union — value paired with explicit variant tag.

    §3.7.2: Equality compares tag AND inner value, not type_name.
    Transparent access: ``[]``, ``len()``, ``iter()``, ``in`` delegate to inner value.
    """

    __slots__ = ("value", "tag", "variants", "type_name")
    __match_args__ = ("tag", "value")

    def __init__(
        self,
        value: object,
        tag: str,
        variants: dict[str, str | None],
        type_name: str | None = None,
    ):
        self.value = value
        self.tag = tag
        self.variants = variants
        self.type_name = type_name

    def __eq__(self, other: object) -> bool:
        if isinstance(other, UzonTaggedUnion):
            return self.value == other.value and self.tag == other.tag
        return self.value == other

    def __hash__(self) -> int:
        return hash((self.value, self.tag))

    def __repr__(self) -> str:
        name = self.type_name or "anonymous"
        return f"UzonTaggedUnion({self.value!r}, tag={self.tag!r}, type={name})"

    def __str__(self) -> str:
        return str(self.value)

    def __getitem__(self, key):
        return self.value[key]

    def __len__(self) -> int:
        return len(self.value)

    def __iter__(self):
        return iter(self.value)

    def __contains__(self, item) -> bool:
        return item in self.value

    def __copy__(self) -> UzonTaggedUnion:
        return UzonTaggedUnion(self.value, self.tag, self.variants, self.type_name)

    def __deepcopy__(self, memo: dict) -> UzonTaggedUnion:
        return UzonTaggedUnion(
            _copy.deepcopy(self.value, memo),
            self.tag,
            _copy.deepcopy(self.variants, memo),
            self.type_name,
        )

    def to_plain(self) -> object:
        return self.value


# ── function types ─────────────────────────────────────────────


class UzonFunction:
    """§3.8: Function value — a closure capturing its definition scope."""

    __slots__ = ("params", "return_type", "body_bindings", "body_expr",
                 "closure_scope", "type_name")

    def __init__(
        self,
        params: list[tuple[str, str, object]],  # [(name, type_name, default_or_None)]
        return_type: str,
        body_bindings: list,
        body_expr: object,
        closure_scope: object,
        type_name: str | None = None,
    ):
        self.params = params
        self.return_type = return_type
        self.body_bindings = body_bindings
        self.body_expr = body_expr
        self.closure_scope = closure_scope
        self.type_name = type_name

    def __repr__(self) -> str:
        name = self.type_name or "anonymous"
        param_str = ", ".join(f"{n}: {t}" for n, t, _ in self.params)
        return f"UzonFunction({param_str} -> {self.return_type}, type={name})"

    def signature(self) -> tuple[tuple[str, ...], str]:
        """Return (param_types, return_type) for structural comparison (§3.8)."""
        return tuple(t for _, t, _ in self.params), self.return_type


class UzonBuiltinFunction:
    """§5.16: Built-in std library function."""

    __slots__ = ("name", "func", "min_args", "max_args")

    def __init__(self, name: str, func: object, min_args: int, max_args: int):
        self.name = name
        self.func = func
        self.min_args = min_args
        self.max_args = max_args

    def __repr__(self) -> str:
        return f"UzonBuiltinFunction({self.name!r})"
