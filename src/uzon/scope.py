# SPDX-FileCopyrightText: © 2026 Suho Kang
# SPDX-License-Identifier: MIT
"""UZON scope — lexical scoping with self-exclusion per §5.12.

The scope chain implements:
- Lexical parent chain: self.name walks current → parent → ... → file scope.
- Self-exclusion rule (§5.12): when evaluating a binding, the binding's
  own name is excluded from the current scope lookup.
- Type namespace (§6.2): separate from binding namespace.
- File boundary isolation (§7.3): self does not cross file boundaries.
"""

from __future__ import annotations

from typing import Any

from .types import UzonUndefined


class Scope:
    """A scope in the UZON evaluation environment."""

    def __init__(self, parent: Scope | None = None, *, self_scope: Scope | None = None):
        self._bindings: dict[str, Any] = {}
        self._parent = parent
        self._types: dict[str, Any] = {}
        self._self_scope = self_scope

    def define(self, name: str, value: Any) -> None:
        self._bindings[name] = value

    def get(self, name: str, exclude: str | None = None) -> Any:
        """Look up a name with optional self-exclusion (§5.12).

        If *exclude* matches *name* at this scope level, skip this scope
        and search the parent — implementing the spec's self-exclusion rule.
        """
        if name in self._bindings and name != exclude:
            return self._bindings[name]
        if self._parent is not None:
            return self._parent.get(name)
        return UzonUndefined

    def define_type(self, name: str, type_info: Any) -> None:
        """§6.2: Define a named type in this scope's type namespace."""
        self._types[name] = type_info

    def has_own_type(self, name: str) -> bool:
        return name in self._types

    def get_type(self, name: str) -> Any:
        """Look up a type name through the scope chain (§6.2)."""
        if name in self._types:
            return self._types[name]
        if self._parent is not None:
            return self._parent.get_type(name)
        return None

    def has(self, name: str) -> bool:
        if name in self._bindings:
            return True
        return self._parent.has(name) if self._parent else False

    def to_dict(self) -> dict[str, Any]:
        """Return bindings as a dict, excluding undefined values."""
        return {k: v for k, v in self._bindings.items() if v is not UzonUndefined}

    @property
    def parent(self) -> Scope | None:
        return self._parent

    @property
    def self_scope(self) -> Scope:
        """The scope that `self.name` resolves against.

        For function body scopes (§3.8), this is the closure scope (skipping params).
        For normal scopes, this is the scope itself.
        """
        return self._self_scope if self._self_scope is not None else self

    @property
    def names(self) -> set[str]:
        return set(self._bindings.keys())
