# SPDX-FileCopyrightText: © 2026 Suho Kang
# SPDX-License-Identifier: MIT
"""Type annotation and registration mixin.

Implements §6.1 (type annotation via `as`) and §6.2 (named types via `called`).
"""

from __future__ import annotations

from typing import Any

from ..ast_nodes import (
    AreBinding, Binding, Identifier, IntegerLiteral, ListLiteral,
    NamedVariant, Node, StandaloneStruct, StructLiteral, TypeAnnotation,
    TypeExpr,
)
from ..errors import UzonSyntaxError, UzonTypeError
from ..scope import Scope
from ..types import (
    UzonEnum, UzonFloat, UzonFunction, UzonInt, UzonStruct,
    UzonTaggedUnion, UzonTypedList, UzonUndefined, UzonUnion,
)
from ._constants import FLOAT_TYPES, INT_TYPE_RE


class TypeAnnotationMixin:
    """Type annotation and registration methods mixed into the Evaluator."""

    # ── type registration (§6.2) ──────────────────────────────────────

    def _register_called(
        self, type_name: str, value: Any, binding: Binding | AreBinding, scope: Scope
    ) -> Any:
        """§6.2: Register a named type via `called`.

        Returns the (possibly wrapped) value — dicts become UzonStruct.
        """
        if scope.has_own_type(type_name):
            raise UzonSyntaxError(
                f"Duplicate type name '{type_name}' in same scope",
                binding.line, binding.col,
                file=self._filename,
            )
        # §7.3: qual_id distinguishes same-named types across files.
        qual_id = f"{self._filename}#{type_name}"
        type_info: dict[str, Any] = {
            "name": type_name, "binding": binding.name, "qual_id": qual_id,
        }
        if isinstance(value, UzonEnum):
            type_info["kind"] = "enum"
            type_info["variants"] = value.variants
            value.type_name = type_name
        elif isinstance(value, UzonUnion):
            type_info["kind"] = "union"
            type_info["types"] = value.types
            value.type_name = type_name
        elif isinstance(value, UzonTaggedUnion):
            type_info["kind"] = "tagged_union"
            type_info["variants"] = value.variants
            value.type_name = type_name
        elif isinstance(value, UzonFunction):
            type_info["kind"] = "function"
            type_info["signature"] = value.signature()
            value.type_name = type_name
        elif isinstance(value, dict):
            type_info["kind"] = "struct"
            type_info["fields"] = set(value.keys())
            type_info["field_values"] = dict(value)
            type_info["field_types"] = self._extract_struct_field_types(binding)
            old_id = id(value)
            if not isinstance(value, UzonStruct):
                value = UzonStruct(value, type_name)
            else:
                value.type_name = type_name
            # Migrate _scope_of from old dict id to new UzonStruct id
            if old_id != id(value) and old_id in self._scope_of:
                self._scope_of[id(value)] = self._scope_of.pop(old_id)
            self._called_of[id(value)] = type_name
            self._qual_of[id(value)] = qual_id
        elif isinstance(value, list):
            type_info["kind"] = "list"
            self._called_of[id(value)] = type_name
            for i, elem in enumerate(value):
                old_elem_id = id(elem)
                if isinstance(elem, dict) and not isinstance(elem, UzonStruct):
                    value[i] = UzonStruct(elem, type_name)
                    if old_elem_id in self._scope_of:
                        self._scope_of[id(value[i])] = self._scope_of.pop(old_elem_id)
                    self._called_of[id(value[i])] = type_name
                    self._qual_of[id(value[i])] = qual_id
                elif isinstance(elem, UzonStruct):
                    elem.type_name = type_name
                    self._called_of[id(elem)] = type_name
                    self._qual_of[id(elem)] = qual_id
                elif isinstance(elem, dict):
                    self._called_of[id(elem)] = type_name
                    self._qual_of[id(elem)] = qual_id
        else:
            type_info["kind"] = "value"
        scope.define_type(type_name, type_info)
        return value

    # ── type annotation (as) — §6.1 ──────────────────────────────────

    def _eval_type_annotation(
        self, node: TypeAnnotation, scope: Scope, exclude: str | None
    ) -> Any:
        """§6.1: Evaluate `expr as Type` — assertion, not conversion."""
        # §6.3: Check for enum variant resolution
        type_info = self._resolve_named_type(node.type, scope, node)

        # §3.7 v0.10: variant_shorthand inside `as TaggedUnionType`.
        if type_info and type_info.get("kind") == "tagged_union":
            result = self._try_tagged_shorthand(node, type_info, scope, exclude)
            if result is not None:
                return result

        # §3.5 v0.10: bare enum variants inside `{ ... } as StructType`
        # resolve via the struct's field_values (which carry the declared
        # enum type for each field). _eval_struct_literal consumes the
        # pushed hints; on any abrupt path we defensively clean up.
        if (type_info and type_info.get("kind") == "struct"
                and isinstance(node.expr, StructLiteral)):
            hints = self._struct_field_enum_hints(type_info, scope)
            if hints:
                depth = len(self._field_enum_hints_stack)
                self._field_enum_hints_stack.append(hints)
                try:
                    value = self._eval_node(node.expr, scope, exclude)
                finally:
                    if len(self._field_enum_hints_stack) > depth:
                        self._field_enum_hints_stack.pop()
                self._check_type_assertion(value, node.type, node, scope)
                return self._wrap_typed(value, node.type)

        if type_info and type_info.get("kind") == "enum" and isinstance(node.expr, Identifier):
            variant_name = node.expr.name
            variants = type_info["variants"]
            if variant_name not in variants:
                raise UzonTypeError(
                    f"'{variant_name}' is not a variant of {type_info['name']}",
                    node.line, node.col,
                    file=self._filename,
                )
            return UzonEnum(variant_name, variants, type_info["name"])

        # §3.5 R7 v0.10: bare Identifier asserted against a named union
        # whose members include multiple enums sharing this variant name.
        # The assertion carries no further discriminator, so the reference
        # is ambiguous — raise rather than pick one.
        if (type_info and type_info.get("kind") == "union"
                and isinstance(node.expr, Identifier)
                and not scope.has(node.expr.name)):
            member_types = type_info.get("types", [])
            matching = scope.enum_members_with_variant(
                node.expr.name, member_types,
            )
            if len(matching) > 1:
                raise UzonTypeError(
                    f"Bare variant '{node.expr.name}' is ambiguous under "
                    f"union {type_info['name']} — variant declared by "
                    f"{', '.join(matching)}. Qualify with the specific "
                    "enum, e.g., `X.v as " + type_info["name"] + "`.",
                    node.line, node.col, file=self._filename,
                )
            if len(matching) == 1:
                only = matching[0]
                enum_info = scope.get_type(only)
                return UzonEnum(
                    node.expr.name, enum_info["variants"], only,
                )

        # List of enum variants: [id, id, ...] as [EnumType]
        if node.type.is_list and node.type.inner and isinstance(node.expr, ListLiteral):
            inner_type_info = self._resolve_named_type(node.type.inner, scope, node)
            if inner_type_info and inner_type_info.get("kind") == "enum":
                elements = []
                for elem in node.expr.elements:
                    if isinstance(elem, Identifier) and elem.name in inner_type_info["variants"]:
                        elements.append(UzonEnum(
                            elem.name, inner_type_info["variants"], inner_type_info["name"]
                        ))
                    else:
                        elements.append(self._eval_node(elem, scope, exclude))
                return UzonTypedList(elements, inner_type_info["name"])

            # §3.2.1 line 491 + §3.4: `[{...}, {...}] as [NamedStruct]` —
            # promote each struct-literal element to the named type before
            # the list-level homogeneity check. Missing fields fill from
            # declared defaults (§3.2 v0.10), and each element takes on the
            # nominal type, so homogeneity is satisfied by nominal identity
            # even when elements specify different subsets of fields (the
            # sparse-config idiom).
            if inner_type_info and inner_type_info.get("kind") == "struct":
                elements = []
                for elem in node.expr.elements:
                    if isinstance(elem, StructLiteral):
                        synthetic = TypeAnnotation(
                            expr=elem, type=node.type.inner,
                            line=elem.line, col=elem.col,
                        )
                        elements.append(
                            self._eval_type_annotation(synthetic, scope, exclude)
                        )
                    else:
                        v = self._eval_node(elem, scope, exclude)
                        if v is UzonUndefined:
                            raise UzonTypeError(
                                "List element is undefined",
                                elem.line, elem.col, file=self._filename,
                            )
                        if v is not None:
                            self._check_type_assertion(
                                v, node.type.inner, node, scope,
                            )
                        elements.append(v)
                self._check_list_homogeneity(elements, node)
                return UzonTypedList(elements, node.type.inner.name)

        # Bypass i64 range check when target is a wider integer type
        if INT_TYPE_RE.match(node.type.name) and isinstance(node.expr, IntegerLiteral):
            value = int(node.expr.value, 0)
        else:
            value = self._eval_node(node.expr, scope, exclude)

        # §6.1: undefined propagates through `as`
        if value is UzonUndefined:
            self._validate_type_name(node.type, node, scope)
            return UzonUndefined

        self._check_type_assertion(value, node.type, node, scope)
        # §6.3 R7 v0.10: adoptable numeric literal applied to a named
        # union adopts the first member type whose category matches
        # (integer/float). Integer-to-float promotion applies only when
        # no integer member exists.
        if type_info and type_info.get("kind") == "union":
            value = self._adopt_literal_to_union(value, type_info.get("types", []))
        return self._wrap_typed(value, node.type)

    @staticmethod
    def _adopt_literal_to_union(value: Any, members: list[str]) -> Any:
        """§6.3 R7 v0.10: Promote an adoptable numeric literal to the
        first matching union member type.

        * adoptable ``UzonInt`` → first ``i*``/``u*`` member; otherwise
          first ``f*`` member via integer-to-float promotion.
        * adoptable ``UzonFloat`` → first ``f*`` member.
        """
        if isinstance(value, UzonInt) and value.adoptable:
            for mt in members:
                if INT_TYPE_RE.match(mt):
                    return UzonInt(int(value), mt)
            for mt in members:
                if mt in FLOAT_TYPES:
                    return UzonFloat(float(int(value)), mt)
        elif isinstance(value, UzonFloat) and value.adoptable:
            for mt in members:
                if mt in FLOAT_TYPES:
                    return UzonFloat(float(value), mt)
        return value

    @staticmethod
    def _extract_struct_field_types(
        binding: Binding | AreBinding,
    ) -> dict[str, TypeExpr]:
        """Capture each struct field's declared TypeExpr from the `as T`
        annotation on its default. Used by §3.5 R7 to detect fields whose
        declared type is a named union.
        """
        value = getattr(binding, "value", None)
        if isinstance(value, StandaloneStruct):
            value = value.struct
        if not isinstance(value, StructLiteral):
            return {}
        field_types: dict[str, TypeExpr] = {}
        for f in value.fields:
            if isinstance(f, Binding) and isinstance(f.value, TypeAnnotation):
                field_types[f.name] = f.value.type
        return field_types

    def _struct_field_enum_hints(
        self, type_info: dict, scope: Scope,
    ) -> dict[str, dict]:
        """Return per-field hints for resolving bare variants inside a
        struct literal (§3.5 R7 v0.10).

        Hint shape:
          - ``{"kind": "enum", "name": EnumType, "variants": {...}}``
            for fields declared with a single enum type.
          - ``{"kind": "union", "name": UnionType, "members": [...]}``
            for fields declared with a union whose members are enums.
            Each entry is ``{"name": EnumType, "variants": {...}}``.
        """
        hints: dict[str, dict] = {}
        field_values = type_info.get("field_values", {})
        field_types: dict[str, TypeExpr] = type_info.get("field_types", {})

        for fname, ftype in field_types.items():
            if ftype.is_list or ftype.is_tuple:
                continue
            info = scope.get_type(ftype.name)
            if info is None:
                continue
            if info.get("kind") == "union":
                members: list[dict] = []
                for mt in info.get("types", []):
                    minfo = scope.get_type(mt)
                    if minfo and minfo.get("kind") == "enum":
                        members.append({
                            "name": minfo["name"],
                            "variants": minfo["variants"],
                        })
                if members:
                    hints[fname] = {
                        "kind": "union",
                        "name": info["name"],
                        "members": members,
                    }

        # Fallback/augment from resolved default values for fields that do
        # not carry an explicit `as` annotation but default to an enum.
        for fname, fval in field_values.items():
            if fname in hints:
                continue
            if isinstance(fval, UzonEnum) and fval.type_name:
                hints[fname] = {
                    "kind": "enum",
                    "name": fval.type_name,
                    "variants": fval.variants,
                }
        return hints

    def _try_tagged_shorthand(
        self, node: TypeAnnotation, type_info: dict, scope: Scope, exclude: str | None
    ) -> Any | None:
        """§3.7 v0.10: Resolve variant_shorthand under `as TaggedUnionType`.

        Handles two forms:
          - ``variant_name inner_value as Type`` (parsed as NamedVariant
            with empty variants)
          - ``variant_name as Type`` (parsed as Identifier) — only valid
            when the variant's type is null.

        Returns the resulting UzonTaggedUnion, or None if the expression
        does not match either shorthand form (falls through to default).
        """
        variants = type_info["variants"]
        type_name = type_info["name"]

        # Form 1: NamedVariant with empty variants → resolve via outer type.
        if isinstance(node.expr, NamedVariant) and not node.expr.variants:
            tag = node.expr.tag
            if tag not in variants:
                raise UzonTypeError(
                    f"'{tag}' is not a variant of {type_name}",
                    node.line, node.col, file=self._filename,
                )
            variant_type = variants[tag]
            if variant_type in (None, "null"):
                raise UzonTypeError(
                    f"Variant '{tag}' of {type_name} is nullary; "
                    "cannot take an inner value",
                    node.line, node.col, file=self._filename,
                )
            inner_value = self._eval_variant_inner(
                node.expr.value, variant_type, scope, exclude, node,
            )
            result = UzonTaggedUnion(inner_value, tag, variants)
            result.type_name = type_name
            return result

        # Form 2: bare Identifier naming a nullary variant.
        if isinstance(node.expr, Identifier):
            tag = node.expr.name
            if tag in variants and variants[tag] in (None, "null"):
                result = UzonTaggedUnion(None, tag, variants)
                result.type_name = type_name
                return result

        return None

    def _eval_variant_inner(
        self, inner_node: Node, variant_type: str,
        scope: Scope, exclude: str | None, outer: Node,
    ) -> Any:
        """Evaluate a variant_shorthand's inner value under an expected type.

        If the inner is itself a NamedVariant with empty variants and the
        expected variant_type resolves to another tagged_union, recurse
        with that type as context (§3.7 nested shorthand).
        """
        inner_type_info = None
        if variant_type:
            inner_type_info = scope.get_type(variant_type)

        if (inner_type_info and inner_type_info.get("kind") == "tagged_union"
                and isinstance(inner_node, NamedVariant)
                and not inner_node.variants):
            synthetic = TypeAnnotation(
                expr=inner_node,
                type=TypeExpr(name=variant_type, line=outer.line, col=outer.col),
                line=outer.line, col=outer.col,
            )
            return self._eval_type_annotation(synthetic, scope, exclude)

        if (inner_type_info and inner_type_info.get("kind") == "enum"
                and isinstance(inner_node, Identifier)):
            variants = inner_type_info["variants"]
            name = inner_node.name
            if name not in variants:
                raise UzonTypeError(
                    f"'{name}' is not a variant of {variant_type}",
                    outer.line, outer.col, file=self._filename,
                )
            return UzonEnum(name, variants, variant_type)

        value = self._eval_node(inner_node, scope, exclude)
        if variant_type:
            self._check_type_assertion(
                value,
                TypeExpr(name=variant_type, line=outer.line, col=outer.col),
                outer, scope,
            )
        return value

    def _wrap_typed(self, value: Any, type_expr: TypeExpr) -> Any:
        """Wrap a value in a typed wrapper after assertion/conversion."""
        if value is None or isinstance(value, bool):
            return value
        if type_expr.is_list:
            if isinstance(value, list) and type_expr.inner:
                wrapped = [self._wrap_typed(e, type_expr.inner) if e is not None else None
                           for e in value]
                return UzonTypedList(wrapped, type_expr.inner.name)
            return value
        name = type_expr.name
        if INT_TYPE_RE.match(name) and isinstance(value, int) and not isinstance(value, bool):
            if isinstance(value, UzonInt) and value.type_name == name and not value.adoptable:
                return value
            return UzonInt(int(value), name)
        if name in FLOAT_TYPES:
            if isinstance(value, UzonInt) and value.adoptable:
                return UzonFloat(float(int(value)), name)
            if isinstance(value, float):
                if isinstance(value, UzonFloat) and value.type_name == name and not value.adoptable:
                    return value
                return UzonFloat(float(value), name)
        return value
