# SPDX-FileCopyrightText: © 2026 Suho Kang
# SPDX-License-Identifier: MIT
"""Struct and member access evaluation mixin.

Implements §3.2 (struct literal), §3.2.1 (with — copy-and-update),
§3.2.2 (extends — copy, override, and add), §5.12 (member access),
§5.13 (env reference), and §7 (file import).
"""

from __future__ import annotations

import os
from typing import Any

from ..ast_nodes import (
    AreBinding, Binding, MemberAccess, Node, EnvRef,
    StructImport, StructLiteral, StructExtension, StructOverride,
)
from ..errors import UzonCircularError, UzonError, UzonRuntimeError, UzonTypeError
from ..scope import Scope
from ..types import (
    UzonFloat, UzonFunction, UzonInt, UzonStruct, UzonTaggedUnion, UzonUndefined,
)
from ._constants import INT_TYPE_RE


_ORDINALS = {
    "first": 0, "second": 1, "third": 2, "fourth": 3, "fifth": 4,
    "sixth": 5, "seventh": 6, "eighth": 7, "ninth": 8, "tenth": 9,
}


class StructMixin:
    """Struct and member access methods mixed into the Evaluator."""

    # ── member access (§5.12, §5.13) ─────────────────────────────────

    @staticmethod
    def _deadopt(value: Any) -> Any:
        """Strip adoptability from values retrieved via member access."""
        if isinstance(value, UzonInt) and value.adoptable:
            return UzonInt(int(value), value.type_name)
        if isinstance(value, UzonFloat) and value.adoptable:
            return UzonFloat(float(value), value.type_name)
        return value

    def _eval_member_access(
        self, node: MemberAccess, scope: Scope, exclude: str | None
    ) -> Any:
        if isinstance(node.object, EnvRef):
            return os.environ.get(node.member, UzonUndefined)

        obj = self._eval_node(node.object, scope, exclude)
        if obj is UzonUndefined:
            return UzonUndefined
        if obj is None:
            raise UzonTypeError(
                "Member access on null — null is a value, not a missing state",
                node.line, node.col, file=self._filename,
            )
        if isinstance(obj, UzonFunction):
            raise UzonTypeError(
                "Member access on function value — functions have no fields",
                node.line, node.col, file=self._filename,
            )
        if isinstance(obj, dict):
            return obj.get(node.member, UzonUndefined)
        if isinstance(obj, (list, tuple)):
            return self._access_indexed(obj, node.member)
        if isinstance(obj, UzonTaggedUnion):
            return self._access_tagged_union(obj, node.member)
        return UzonUndefined

    @staticmethod
    def _access_indexed(obj: list | tuple, member: str) -> Any:
        """Access a list/tuple by numeric index or named ordinal."""
        try:
            idx = int(member)
            if 0 <= idx < len(obj):
                return obj[idx]
            return UzonUndefined
        except ValueError:
            pass
        idx = _ORDINALS.get(member)
        if idx is not None and 0 <= idx < len(obj):
            return obj[idx]
        return UzonUndefined

    @staticmethod
    def _access_tagged_union(obj: UzonTaggedUnion, member: str) -> Any:
        """§3.7.1: Transparent member access on tagged union's inner value."""
        inner = obj.value
        if isinstance(inner, dict):
            return inner.get(member, UzonUndefined)
        if isinstance(inner, (list, tuple)):
            try:
                idx = int(member)
                if 0 <= idx < len(inner):
                    return inner[idx]
                return UzonUndefined
            except ValueError:
                pass
            idx = _ORDINALS.get(member)
            if idx is not None and 0 <= idx < len(inner):
                return inner[idx]
        return UzonUndefined

    # ── struct literal (§3.2) ────────────────────────────────────────

    def _eval_struct_literal(self, node: StructLiteral, scope: Scope) -> dict[str, Any]:
        """§3.2: Evaluate a struct literal into a dict with a child scope."""
        child_scope = Scope(parent=scope)
        self._evaluate_bindings(node.fields, child_scope, struct_context=True)
        result = child_scope.to_dict()
        self._scope_of[id(result)] = child_scope
        return result

    # ── struct override (§3.2.1) ─────────────────────────────────────

    def _eval_struct_override(
        self, node: StructOverride, scope: Scope, exclude: str | None
    ) -> dict[str, Any]:
        """§3.2.1: Evaluate `base with { overrides }` — copy-and-update."""
        base = self._eval_node(node.base, scope, exclude)
        if base is UzonUndefined:
            raise UzonRuntimeError(
                "'with' requires a concrete struct, got undefined",
                node.base.line, node.base.col, file=self._filename,
            )
        if isinstance(base, UzonTaggedUnion):
            raise UzonTypeError(
                "'with' requires a struct, not a tagged union — "
                "apply 'with' to the inner struct explicitly",
                node.line, node.col, file=self._filename,
            )
        if not isinstance(base, dict):
            raise UzonTypeError(
                f"'with' requires a struct, got {self._type_name(base)}",
                node.line, node.col, file=self._filename,
            )

        override_scope = self._augment_scope_with_types(scope, base)
        overrides: dict[str, Any] = {}

        # §3.2.1: Lookup named type field definitions for null-priority rule
        named_fields = self._get_named_type_fields(base)

        for field in node.overrides.fields:
            name = field.name
            if name not in base:
                raise UzonTypeError(
                    f"'with' override: field '{name}' does not exist in base struct",
                    field.line, field.col, file=self._filename,
                )
            value = self._eval_field_value(field, override_scope, exclude)
            if value is UzonUndefined:
                raise UzonRuntimeError(
                    f"'with' override: field '{name}' cannot be undefined",
                    field.line, field.col, file=self._filename,
                )
            value = self._check_override_compat(
                base[name], value, name, "with", field,
                named_type_original=named_fields.get(name),
            )
            overrides[name] = value

        if isinstance(base, UzonStruct) and base.type_name:
            result = UzonStruct(base, base.type_name)
            result.update(overrides)
            self._called_of[id(result)] = base.type_name
        else:
            result = dict(base)
            result.update(overrides)
            base_type = self._called_of.get(id(base))
            if base_type:
                self._called_of[id(result)] = base_type
        # Propagate scope from base so nested type defs remain accessible
        base_scope = self._scope_of.get(id(base))
        if base_scope:
            self._scope_of[id(result)] = base_scope
        return result

    # ── struct extension (§3.2.2) ────────────────────────────────────

    def _eval_struct_extension(
        self, node: StructExtension, scope: Scope, exclude: str | None
    ) -> dict[str, Any]:
        """§3.2.2: Evaluate `base plus { fields }` — copy, override, and add."""
        base = self._eval_node(node.base, scope, exclude)
        if base is UzonUndefined:
            raise UzonRuntimeError(
                "'plus' requires a concrete struct, got undefined",
                node.base.line, node.base.col, file=self._filename,
            )
        if isinstance(base, UzonTaggedUnion):
            raise UzonTypeError(
                "'plus' requires a struct, not a tagged union",
                node.line, node.col, file=self._filename,
            )
        if not isinstance(base, dict):
            raise UzonTypeError(
                f"'plus' requires a struct, got {self._type_name(base)}",
                node.line, node.col, file=self._filename,
            )

        ext_scope = self._augment_scope_with_types(scope, base)
        overrides: dict[str, Any] = {}
        additions: dict[str, Any] = {}

        named_fields = self._get_named_type_fields(base)

        for field in node.extensions.fields:
            name = field.name
            is_existing = name in base
            value = self._eval_field_value(field, ext_scope, exclude)
            if value is UzonUndefined:
                raise UzonRuntimeError(
                    f"'plus' field: '{name}' cannot be undefined",
                    field.line, field.col, file=self._filename,
                )
            if is_existing:
                value = self._check_override_compat(
                    base[name], value, name, "plus", field,
                    named_type_original=named_fields.get(name),
                )
                overrides[name] = value
            else:
                additions[name] = value

        if not additions:
            raise UzonTypeError(
                "'plus' must add at least one new field — use 'with' for override-only",
                node.line, node.col, file=self._filename,
            )

        # Note: plus adds new fields, so the result is NOT the same type as base.
        # We intentionally do NOT preserve base's type_name here — the shape changed.
        result = dict(base)
        result.update(overrides)
        result.update(additions)
        # Propagate scope so nested type defs remain accessible
        base_scope = self._scope_of.get(id(base))
        if base_scope:
            self._scope_of[id(result)] = base_scope
        return result

    # ── shared struct helpers ────────────────────────────────────────

    def _get_named_type_fields(self, base: dict) -> dict[str, Any]:
        """Return the named type's original field values, or empty dict."""
        type_name = None
        if isinstance(base, UzonStruct) and base.type_name:
            type_name = base.type_name
        else:
            type_name = self._called_of.get(id(base))
        if not type_name:
            return {}
        base_scope = self._scope_of.get(id(base))
        if not base_scope:
            return {}
        type_info = base_scope.get_type(type_name)
        if type_info and isinstance(type_info, dict):
            return type_info.get("field_values", {})
        return {}

    def _augment_scope_with_types(self, scope: Scope, base: dict) -> Scope:
        """Create a scope augmented with the base struct's type definitions."""
        base_scope = self._scope_of.get(id(base))
        if base_scope and base_scope._types:
            augmented = Scope(parent=scope)
            for tname, tinfo in base_scope._types.items():
                augmented.define_type(tname, tinfo)
            return augmented
        return scope

    def _eval_field_value(
        self, field: Binding | AreBinding, scope: Scope, exclude: str | None
    ) -> Any:
        """Evaluate a struct field (binding or are-binding)."""
        if isinstance(field, AreBinding):
            elements = []
            for elem in field.elements:
                v = self._eval_node(elem, scope, exclude)
                if v is UzonUndefined:
                    raise UzonRuntimeError(
                        "List element is undefined",
                        elem.line, elem.col, file=self._filename,
                    )
                elements.append(v)
            self._check_list_homogeneity(elements, field)
            return elements
        return self._eval_node(field.value, scope, exclude)

    def _check_override_compat(
        self, original: Any, value: Any, field_name: str, context: str, field: Node,
        *, named_type_original: Any = None,
    ) -> Any:
        """Check type compatibility for struct field override, with adoptable coercion.

        *named_type_original* is the field value from the named type definition
        (§3.2.1).  When the current field is null but the named type defines a
        concrete type, the named type takes priority.
        """
        # §3.2.1: When field is null but named type defines a concrete type,
        # use the named type's original value for type checking.
        effective = original
        if effective is None and named_type_original is not None:
            effective = named_type_original

        if effective is not None and value is not None:
            if not self._same_uzon_type(effective, value):
                raise UzonTypeError(
                    f"'{context}' override: field '{field_name}' type mismatch — "
                    f"original is {self._type_name(effective)}, override is {self._type_name(value)}",
                    field.line, field.col, file=self._filename,
                )
            if (isinstance(effective, UzonInt) and isinstance(value, UzonInt)
                    and value.adoptable and effective.type_name != value.type_name):
                m = INT_TYPE_RE.match(effective.type_name)
                if m:
                    self._check_int_range(int(value), m.group(1), int(m.group(2)), effective.type_name, field)
                value = UzonInt(int(value), effective.type_name)
            elif (isinstance(effective, UzonFloat) and isinstance(value, UzonFloat)
                    and value.adoptable and effective.type_name != value.type_name):
                value = UzonFloat(float(value), effective.type_name)
        return value

    # ── file import (§7) ─────────────────────────────────────────────

    def _eval_struct_import(self, node: StructImport) -> dict[str, Any]:
        """§7.1: Evaluate `struct "path"` — import a .uzon file as a struct."""
        from ..lexer import Lexer
        from ..parser import Parser

        raw_path = node.path
        # §7: Use CWD as base directory when evaluating from a string
        if self._filename == "<string>":
            base_dir = os.getcwd()
        else:
            base_dir = os.path.dirname(os.path.abspath(self._filename))
        if "." not in os.path.basename(raw_path):
            raw_path = raw_path + ".uzon"
        resolved = os.path.realpath(os.path.join(base_dir, raw_path))

        if resolved in self._import_stack:
            raise UzonCircularError(
                f"Circular import: {resolved}", node.line, node.col,
                file=self._filename,
            )
        if resolved in self._import_cache:
            return self._import_cache[resolved]

        if not os.path.isfile(resolved):
            raise UzonRuntimeError(
                f"Import file not found: {resolved}", node.line, node.col,
                file=self._filename,
            )
        with open(resolved, encoding="utf-8") as f:
            source = f.read()

        saved_collected = self._collected_errors
        self._collected_errors = []
        self._import_stack.append(resolved)
        old_filename = self._filename
        self._filename = resolved
        try:
            tokens = Lexer(source).tokenize()
            doc = Parser(tokens).parse()
            scope = Scope()
            scope.define("std", self._build_std())
            self._evaluate_bindings(doc.bindings, scope)
            result = scope.to_dict()
            result.pop("std", None)
            self._scope_of[id(result)] = scope
        except UzonError as e:
            # Build import chain stack trace
            parts: list[str] = []
            if old_filename != "<string>":
                parts.append(f"File {old_filename}")
            else:
                parts.append("<string>")
            if node.line is not None and node.col is not None:
                parts.append(f"Line {node.line}, col {node.col}")
            elif node.line is not None:
                parts.append(f"Line {node.line}")
            raise type(e)(f"{e}\n  imported from {', '.join(parts)}") from None
        finally:
            self._filename = old_filename
            self._import_stack.pop()
            self._collected_errors = saved_collected

        self._import_cache[resolved] = result
        return result
