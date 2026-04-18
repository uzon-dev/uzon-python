# SPDX-FileCopyrightText: © 2026 Suho Kang
# SPDX-License-Identifier: MIT
"""UZON evaluator core — walks AST and produces Python values.

Implements the main evaluation loop: binding resolution and node dispatch
(§3, §5). Delegates to mixin classes for dependency resolution, type system,
operators, control flow, struct operations, functions, and standard library.
"""

from __future__ import annotations

from typing import Any

from ..ast_nodes import (
    AreBinding, BinaryOp, Binding, BoolLiteral, CaseExpr, Conversion,
    Document, EnvRef, FieldExtraction, FloatLiteral, FromEnum, FromUnion,
    FunctionCall, FunctionExpr, Grouping, Identifier, IfExpr, InfLiteral,
    IntegerLiteral, ListLiteral, MemberAccess, NamedVariant, NanLiteral,
    Node, NullLiteral, OrElse,
    StandaloneEnum, StandaloneStruct, StandaloneTaggedUnion, StandaloneUnion,
    StringLiteral, StructImport,
    StructLiteral, StructExtension, StructOverride, TupleLiteral,
    TypeAnnotation, UnaryOp, UndefinedLiteral,
)
from ..errors import UzonCircularError, UzonRuntimeError, UzonSyntaxError, UzonTypeError
from ..scope import Scope
from ..types import (
    UzonEnum, UzonFloat, UzonInt, UzonStruct, UzonTaggedUnion,
    UzonTypedList, UzonUndefined, UzonUnion,
)
from ._constants import I64_MIN, I64_MAX, SPECULATIVE_FAILED
from ._control import ControlMixin
from ._dependencies import DependencyMixin
from ._functions import FunctionMixin
from ._operators import OperatorMixin
from ._stdlib import StdlibMixin
from ._structs import StructMixin
from ._type_annotation import TypeAnnotationMixin
from ._type_checks import TypeChecksMixin
from ._type_conversion import TypeConversionMixin


class Evaluator(
    DependencyMixin,
    TypeAnnotationMixin,
    TypeConversionMixin,
    TypeChecksMixin,
    OperatorMixin,
    ControlMixin,
    StructMixin,
    FunctionMixin,
    StdlibMixin,
):
    """Evaluate a UZON AST into Python values."""

    def __init__(self, filename: str = "<string>"):
        self._filename = filename
        self._import_cache: dict[str, dict[str, Any]] = {}
        self._import_stack: list[str] = []  # §7.2: circular import detection
        self._scope_of: dict[int, Scope] = {}   # id(result_dict) → Scope
        self._called_of: dict[int, str] = {}    # id(result_dict) → type_name (§6.2)
        self._call_stack: list[int] = []  # §3.8: recursion detection
        self._collected_errors: list = []  # multi-error collection

    def evaluate(self, doc: Document) -> dict[str, Any]:
        """§3: Evaluate a Document AST into a Python dict."""
        # Push entry file to import stack for circular import detection
        if self._filename != "<string>":
            self._import_stack.append(self._filename)
        scope = Scope()
        scope.define("std", self._build_std())
        try:
            self._evaluate_bindings(doc.bindings, scope)
        finally:
            if self._filename != "<string>" and self._import_stack:
                self._import_stack.pop()
        result = scope.to_dict()
        result.pop("std", None)
        return result

    # ── binding evaluation ─────────────────────────────────────────────

    def _evaluate_bindings(
        self, bindings: list[Binding | AreBinding], scope: Scope,
        *, struct_context: bool = False,
    ) -> None:
        """Evaluate bindings with dependency resolution and duplicate detection."""
        # §3.1: detect duplicates; self-referencing overrides are allowed
        seen: dict[str, Binding | AreBinding] = {}
        override_bindings: list[Binding | AreBinding] = []
        primary_bindings: list[Binding | AreBinding] = []
        for b in bindings:
            if b.name in seen:
                is_of = isinstance(b, Binding) and isinstance(b.value, FieldExtraction)
                if not is_of and not self._references_own_name(b, b.name):
                    raise UzonSyntaxError(
                        f"Duplicate binding '{b.name}' in same scope",
                        b.line, b.col, file=self._filename,
                    )
                override_bindings.append(b)
            else:
                seen[b.name] = b
                primary_bindings.append(b)
        bindings = primary_bindings

        had_errors = False

        # §3.8: Function call DAG check — detect direct and mutual recursion
        fn_cycle_names: set[str] = set()
        func_bindings = {
            b.name: b for b in bindings
            if isinstance(b, Binding) and isinstance(b.value, FunctionExpr)
        }
        if func_bindings:
            # Build call graph among functions
            call_graph: dict[str, set[str]] = {}
            for name, b in func_bindings.items():
                refs: set[str] = set()
                for body_b in b.value.body_bindings:
                    self._collect_bare_refs(body_b.value, refs)
                self._collect_bare_refs(b.value.body_expr, refs)
                call_graph[name] = refs & set(func_bindings.keys())

            # DFS cycle detection (0=white, 1=gray, 2=black)
            color: dict[str, int] = {n: 0 for n in func_bindings}

            def _dfs(node: str) -> bool:
                color[node] = 1
                for neighbor in call_graph.get(node, set()):
                    if color.get(neighbor, 0) == 1:
                        return True
                    if color.get(neighbor, 0) == 0 and _dfs(neighbor):
                        return True
                color[node] = 2
                return False

            for fname in func_bindings:
                if color[fname] == 0:
                    if _dfs(fname):
                        for n, c in color.items():
                            if c == 1:
                                fn_cycle_names.add(n)
                                color[n] = 2

            for name in fn_cycle_names:
                b = func_bindings[name]
                # Report at the call site, not the definition site
                other = fn_cycle_names - {name}
                loc = self._find_ref_location(b.value, other | {name})
                err_line = loc[0] if loc else b.line
                err_col = loc[1] if loc else b.col
                self._collected_errors.append(UzonCircularError(
                    "Recursive function call detected — call graph must be a DAG",
                    err_line, err_col, file=self._filename,
                ))
                had_errors = True

        deps = self._build_dependencies(bindings)
        order, cycle_groups = self._topological_sort(bindings, deps)

        # Collect cycle errors per binding (excluding already-reported fn cycles)
        by_name = {b.name: b for b in bindings}
        for comp in cycle_groups:
            for name in comp:
                if name in fn_cycle_names:
                    continue
                b = by_name[name]
                self._collected_errors.append(UzonCircularError(
                    f"Circular dependency among: {', '.join(comp)}",
                    b.line, b.col, file=self._filename,
                ))
            had_errors = True

        # Evaluate non-cycle bindings, skipping function cycle participants
        for b in order:
            if b.name in fn_cycle_names:
                continue
            pre_count = len(self._collected_errors)
            try:
                value = self._eval_binding(b, scope, struct_context=struct_context)
                scope.define(b.name, value)
            except UzonCircularError as e:
                if len(self._collected_errors) == pre_count:
                    self._collected_errors.append(e)
                had_errors = True
                continue

        for b in override_bindings:
            pre_count = len(self._collected_errors)
            try:
                value = self._eval_binding(b, scope, struct_context=struct_context)
                scope.define(b.name, value)
            except UzonCircularError as e:
                if len(self._collected_errors) == pre_count:
                    self._collected_errors.append(e)
                had_errors = True
                continue

        if had_errors:
            last = self._collected_errors[-1]
            last.errors = list(self._collected_errors)
            raise last

    def _eval_binding(
        self, b: Binding | AreBinding, scope: Scope,
        struct_context: bool = False,
    ) -> Any:
        """Evaluate a single binding."""
        if isinstance(b, AreBinding):
            return self._eval_are_binding(b, scope)

        # §3.1: undefined is not a literal — cannot be assigned directly.
        if isinstance(b.value, UndefinedLiteral):
            raise UzonTypeError(
                f"Cannot assign literal 'undefined' to '{b.name}'",
                b.value.line, b.value.col, file=self._filename,
            )

        # §3.2/§3.5/§3.6/§3.7: Standalone type declaration — the binding name
        # becomes the type name. Mixing with ``called`` is a syntax error.
        if isinstance(b.value, (StandaloneStruct, StandaloneEnum,
                                StandaloneUnion, StandaloneTaggedUnion)):
            if b.called is not None:
                raise UzonSyntaxError(
                    "Standalone type declaration cannot be combined with 'called'",
                    b.value.line, b.value.col, file=self._filename,
                )
            self._check_no_self_type_reference(b.value, b.name, b)
            value = self._eval_standalone_decl(b.value, scope)
            return self._register_called(b.name, value, b, scope)

        # §6.4: Recursive type definitions are forbidden. When this binding will
        # name a type (via `called`), detect self-reference in its value AST.
        if b.called is not None:
            self._check_no_self_type_reference(b.value, b.called, b)

        value = self._eval_node(b.value, scope, exclude=b.name)

        # §6.1: Empty list requires type annotation (relaxed inside struct literals)
        if not struct_context and isinstance(value, list) and len(value) == 0:
            if not isinstance(b.value, (TypeAnnotation, IfExpr, CaseExpr, OrElse, FunctionCall, BinaryOp)):
                raise UzonTypeError(
                    "Empty list requires explicit type annotation: [] as [Type]",
                    b.value.line, b.value.col, file=self._filename,
                )

        # All-null list requires type annotation (relaxed inside struct literals)
        if (not struct_context and isinstance(value, list) and value
                and not isinstance(b.value, TypeAnnotation)
                and all(e is None for e in value)):
            raise UzonTypeError(
                "All-null list requires explicit type annotation: [...] as [Type]",
                b.value.line, b.value.col, file=self._filename,
            )

        if b.called:
            value = self._register_called(b.called, value, b, scope)
        return value

    def _eval_are_binding(self, b: AreBinding, scope: Scope) -> list:
        """Evaluate an `are` binding (list sugar)."""
        elements = []
        for elem in b.elements:
            v = self._eval_node(elem, scope, exclude=b.name)
            if v is UzonUndefined:
                raise UzonRuntimeError(
                    "List element is undefined", elem.line, elem.col,
                    file=self._filename,
                )
            elements.append(v)
        self._check_list_homogeneity(elements, b)
        if elements and all(e is None for e in elements) and not b.type_annotation:
            raise UzonTypeError(
                "All-null list requires explicit type annotation via 'as [Type]'",
                b.line, b.col, file=self._filename,
            )
        if b.type_annotation:
            elements = self._apply_are_type_annotation(elements, b, scope)
        if b.called:
            elements = self._register_called(b.called, elements, b, scope)
        return elements

    def _apply_are_type_annotation(
        self, elements: list, b: AreBinding, scope: Scope
    ) -> list:
        """Apply type annotation to are-binding elements."""
        if b.type_annotation.is_list and b.type_annotation.inner:
            elem_type = b.type_annotation.inner
        elif not b.type_annotation.is_list:
            type_info = scope.get_type(b.type_annotation.name)
            if type_info is not None and type_info.get("kind") == "list":
                elem_type = b.type_annotation
            else:
                raise UzonTypeError(
                    f"'are' binding produces a list — type annotation must be a list type"
                    f" like [{b.type_annotation.name}], not bare {b.type_annotation.name}",
                    b.type_annotation.line, b.type_annotation.col,
                    file=self._filename,
                )
        else:
            elem_type = b.type_annotation
        for i, elem in enumerate(elements):
            if elem is not None:
                self._check_type_assertion(elem, elem_type, b, scope)
                elements[i] = self._wrap_typed(elem, elem_type)
        return elements

    # ── helpers ──────────────────────────────────────────────────────────

    # ── node evaluation ────────────────────────────────────────────────

    def _eval_node(self, node: Node, scope: Scope, exclude: str | None = None) -> Any:
        """§5: Main dispatch — evaluate an AST node to a Python value."""
        # §3.3: Literals
        if isinstance(node, IntegerLiteral):
            val = int(node.value, 0)
            if not (I64_MIN <= val <= I64_MAX):
                raise UzonRuntimeError(
                    f"Integer literal {val} overflows i64 "
                    f"(range {I64_MIN}..{I64_MAX})",
                    node.line, node.col, file=self._filename,
                )
            return UzonInt(val, "i64", adoptable=True)

        if isinstance(node, FloatLiteral):
            return UzonFloat(float(node.value), "f64", adoptable=True)

        if isinstance(node, BoolLiteral):
            return node.value

        if isinstance(node, StringLiteral):
            return self._eval_string(node, scope, exclude)

        if isinstance(node, NullLiteral):
            return None

        if isinstance(node, UndefinedLiteral):
            return UzonUndefined

        if isinstance(node, InfLiteral):
            return UzonFloat(float("inf"), "f64", adoptable=True)

        if isinstance(node, NanLiteral):
            return UzonFloat(float("nan"), "f64", adoptable=True)

        if isinstance(node, EnvRef):
            raise UzonTypeError(
                "'env' must be followed by .NAME", node.line, node.col,
                file=self._filename,
            )

        # §5.12: Member access (env.X, struct.field)
        if isinstance(node, MemberAccess):
            return self._eval_member_access(node, scope, exclude)

        # §5.1: Grouping
        if isinstance(node, Grouping):
            return self._eval_node(node.expr, scope, exclude)

        # §3.2: Struct literal
        if isinstance(node, StructLiteral):
            return self._eval_struct_literal(node, scope)

        # §3.2.1: with — copy-and-update
        if isinstance(node, StructOverride):
            return self._eval_struct_override(node, scope, exclude)

        # §3.2.2: plus — copy, override, and add
        if isinstance(node, StructExtension):
            return self._eval_struct_extension(node, scope, exclude)

        # §3.4: List literal
        if isinstance(node, ListLiteral):
            elements = []
            for e in node.elements:
                v = self._eval_node(e, scope, exclude)
                if v is UzonUndefined:
                    raise UzonRuntimeError(
                        "List element is undefined", e.line, e.col,
                        file=self._filename,
                    )
                elements.append(v)
            self._check_list_homogeneity(elements, node)
            return elements

        # §3.4.1: Tuple literal
        if isinstance(node, TupleLiteral):
            elements = []
            for e in node.elements:
                v = self._eval_node(e, scope, exclude)
                if v is UzonUndefined:
                    raise UzonRuntimeError(
                        "Tuple element is undefined", e.line, e.col,
                        file=self._filename,
                    )
                elements.append(v)
            return tuple(elements)

        # §5.2–§5.8: Binary operators
        if isinstance(node, BinaryOp):
            return self._eval_binary(node, scope, exclude)

        # §5.3: Unary operators
        if isinstance(node, UnaryOp):
            return self._eval_unary(node, scope, exclude)

        # §5.7: or else
        if isinstance(node, OrElse):
            left = self._eval_node(node.left, scope, exclude)
            if left is UzonUndefined:
                return self._eval_node(node.right, scope, exclude)
            right_spec = self._speculative_eval(node.right, scope, exclude)
            if right_spec is not SPECULATIVE_FAILED:
                self._check_branch_type_compat([left, right_spec], node)
            return left

        # §6: Type annotation (as)
        if isinstance(node, TypeAnnotation):
            return self._eval_type_annotation(node, scope, exclude)

        # §6.3: Type conversion (to)
        if isinstance(node, Conversion):
            return self._eval_conversion(node, scope, exclude)

        # §7: File import
        if isinstance(node, StructImport):
            return self._eval_struct_import(node)

        # §5.9: if/then/else
        if isinstance(node, IfExpr):
            return self._eval_if(node, scope, exclude)

        # §5.10: case/when
        if isinstance(node, CaseExpr):
            return self._eval_case(node, scope, exclude)

        # §5.14: Field extraction (is of)
        if isinstance(node, FieldExtraction):
            return self._eval_field_extraction(node, scope, exclude)

        # §3.5: Enum
        if isinstance(node, FromEnum):
            return self._eval_from_enum(node, scope, exclude)

        # §3.6: Union
        if isinstance(node, FromUnion):
            return self._eval_from_union(node, scope, exclude)

        # §3.7: Tagged union
        if isinstance(node, NamedVariant):
            return self._eval_named_variant(node, scope, exclude)

        # §3.2/§3.5/§3.6/§3.7: Standalone type declarations (anonymous fallback).
        # At a binding position, _eval_binding intercepts these and applies the
        # binding name as the type name. When nested, we still produce a value
        # but leave the type name unset.
        if isinstance(node, (StandaloneStruct, StandaloneEnum,
                             StandaloneUnion, StandaloneTaggedUnion)):
            return self._eval_standalone_decl(node, scope)

        # §3.8: Function expression
        if isinstance(node, FunctionExpr):
            return self._eval_function_expr(node, scope, exclude)

        # §5.15: Function call
        if isinstance(node, FunctionCall):
            return self._eval_function_call(node, scope, exclude)

        # §5.12: Identifier lookup via lexical scope chain
        if isinstance(node, Identifier):
            value = scope.get(node.name, exclude=exclude)
            return value

        raise UzonRuntimeError(
            f"Evaluation not yet implemented for {type(node).__name__}",
            node.line, node.col, file=self._filename,
        )

    # ── recursive type detection (§6.4) ───────────────────────────────

    def _check_no_self_type_reference(
        self, node: Node, type_name: str, binding: Binding
    ) -> None:
        """§6.4: Reject recursive type definitions.

        Walks the value AST and raises UzonTypeError if any TypeExpr references
        ``type_name`` — the name the binding is about to claim. Since type
        declarations require forward references to resolve and UZON has no
        forward declarations, direct self-reference is the only form of
        recursion possible within a single binding.
        """
        from ..ast_nodes import (
            TypeExpr as _TypeExpr,
            TypeAnnotation as _TypeAnnotation,
            Conversion as _Conversion,
            FunctionParam as _FunctionParam,
            FunctionExpr as _FunctionExpr,
            FromUnion as _FromUnion,
            NamedVariant as _NamedVariant,
        )

        def check_type(te: _TypeExpr | None) -> None:
            if te is None:
                return
            if te.is_list:
                check_type(te.inner)
                return
            if te.is_tuple:
                for elem in te.elements:
                    check_type(elem)
                return
            if te.name == type_name:
                raise UzonTypeError(
                    f"Recursive type definition: '{type_name}' references itself",
                    te.line or binding.line, te.col or binding.col,
                    file=self._filename,
                )

        def walk(n: Node | None) -> None:
            if n is None:
                return
            if isinstance(n, _TypeAnnotation):
                check_type(n.type)
                walk(n.expr)
                return
            if isinstance(n, _Conversion):
                check_type(n.type)
                walk(n.expr)
                return
            if isinstance(n, _FunctionParam):
                check_type(n.type)
                walk(n.default)
                return
            if isinstance(n, _FunctionExpr):
                for p in n.params:
                    walk(p)
                check_type(n.return_type)
                for bb in n.body_bindings:
                    walk(bb)
                walk(n.body_expr)
                return
            if isinstance(n, _FromUnion):
                for t in n.types:
                    check_type(t)
                walk(n.value)
                return
            if isinstance(n, _NamedVariant):
                for _tag, t in n.variants:
                    check_type(t)
                walk(n.value)
                return
            if isinstance(n, StandaloneStruct):
                walk(n.struct)
                return
            if isinstance(n, StandaloneUnion):
                for t in n.types:
                    check_type(t)
                return
            if isinstance(n, StandaloneTaggedUnion):
                for _tag, t in n.variants:
                    check_type(t)
                return
            # Recurse into compound AST fields
            for _fname, fval in vars(n).items():
                if isinstance(fval, Node):
                    walk(fval)
                elif isinstance(fval, list):
                    for item in fval:
                        if isinstance(item, Node):
                            walk(item)

        walk(node)

    # ── standalone type declarations (§3.2 / §3.5 / §3.6 / §3.7) ─────

    def _eval_standalone_decl(self, node: Node, scope: Scope) -> Any:
        """§3.2/§3.5/§3.6/§3.7: Evaluate a standalone type declaration.

        Returns the default value per the spec's default value tables. When
        invoked from _eval_binding, the caller wraps the result with
        _register_called using the binding name so the binding name becomes
        the type name. When invoked from _eval_node at a nested position,
        the type remains anonymous.
        """
        if isinstance(node, StandaloneStruct):
            return self._eval_struct_literal(node.struct, scope)

        if isinstance(node, StandaloneEnum):
            if len(node.variants) < 2:
                raise UzonTypeError(
                    "Enum must have at least 2 variants",
                    node.line, node.col, file=self._filename,
                )
            seen: set[str] = set()
            for v in node.variants:
                if v in seen:
                    raise UzonTypeError(
                        f"Duplicate variant '{v}' in enum definition",
                        node.line, node.col, file=self._filename,
                    )
                seen.add(v)
            return UzonEnum(node.variants[0], list(node.variants))

        if isinstance(node, StandaloneUnion):
            if len(node.types) < 2:
                raise UzonTypeError(
                    "Union must have at least 2 member types",
                    node.line, node.col, file=self._filename,
                )
            type_names = [t.name for t in node.types]
            seen_t: set[str] = set()
            for tn in type_names:
                if tn in seen_t:
                    raise UzonTypeError(
                        f"Duplicate type '{tn}' in union definition",
                        node.line, node.col, file=self._filename,
                    )
                seen_t.add(tn)
            default = self._default_for_type(node.types[0], scope, node)
            return UzonUnion(default, type_names)

        if isinstance(node, StandaloneTaggedUnion):
            if len(node.variants) < 2:
                raise UzonTypeError(
                    "Tagged union must have at least 2 variants",
                    node.line, node.col, file=self._filename,
                )
            variants_map: dict[str, str | None] = {}
            for var_name, var_type in node.variants:
                if var_name in variants_map:
                    raise UzonTypeError(
                        f"Duplicate variant '{var_name}' in tagged union definition",
                        node.line, node.col, file=self._filename,
                    )
                variants_map[var_name] = var_type.name if var_type else None
            first_name, first_type = node.variants[0]
            default = self._default_for_type(first_type, scope, node)
            return UzonTaggedUnion(default, first_name, variants_map)

        raise UzonRuntimeError(
            f"Unknown standalone decl: {type(node).__name__}",
            node.line, node.col, file=self._filename,
        )

    def _default_for_type(self, te, scope: Scope, node: Node) -> Any:
        """§3.6 default value table — compute default for a type expression."""
        # List types [T]
        if te.is_list:
            inner_name = te.inner.name if te.inner else ""
            return UzonTypedList([], inner_name)
        # Tuple types (T1, T2, ...)
        if te.is_tuple:
            return tuple()
        name = te.name
        # Primitives
        if name == "bool":
            return False
        if name == "string":
            return ""
        if name == "null":
            return None
        # Integers iN / uN
        if name and len(name) >= 2 and name[0] in ("i", "u") and name[1:].isdigit():
            return UzonInt(0, name)
        # Floats fN
        if name in ("f16", "f32", "f64", "f80", "f128"):
            return UzonFloat(0.0, name)
        # Function type — not permitted as standalone default
        if name == "function":
            raise UzonTypeError(
                "Standalone declaration with 'function' as first/only member type is "
                "not permitted — use inline form with explicit value",
                node.line, node.col, file=self._filename,
            )
        # Named types via scope
        type_info = scope.get_type(name)
        if type_info is None:
            raise UzonTypeError(
                f"Unknown type '{name}' in standalone declaration",
                node.line, node.col, file=self._filename,
            )
        kind = type_info.get("kind")
        if kind == "enum":
            variants = type_info["variants"]
            return UzonEnum(variants[0], list(variants), type_info["name"])
        if kind == "struct":
            fields = type_info.get("field_values", {})
            return UzonStruct(dict(fields), type_info["name"])
        if kind == "tagged_union":
            variants_map = type_info.get("variants", {})
            if not variants_map:
                raise UzonTypeError(
                    f"Named tagged union '{name}' has no variants",
                    node.line, node.col, file=self._filename,
                )
            first_name = next(iter(variants_map))
            first_type = variants_map[first_name]
            inner_default: Any = None
            if first_type:
                from ..ast_nodes import TypeExpr
                inner_default = self._default_for_type(
                    TypeExpr(name=first_type), scope, node
                )
            result = UzonTaggedUnion(inner_default, first_name, dict(variants_map))
            result.type_name = type_info["name"]
            return result
        if kind == "union":
            raise UzonTypeError(
                f"Nested union '{name}' is not permitted as standalone default — "
                "use inline form with explicit value",
                node.line, node.col, file=self._filename,
            )
        raise UzonTypeError(
            f"No default value defined for type '{name}'",
            node.line, node.col, file=self._filename,
        )
