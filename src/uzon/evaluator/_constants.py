# SPDX-FileCopyrightText: © 2026 Suho Kang
# SPDX-License-Identifier: MIT
"""Shared constants for the evaluator package."""

import re

# §4.2: Pattern matching integer type names (i8, u16, i32, u64, etc.)
INT_TYPE_RE = re.compile(r'^([iu])(\d+)$')
# §4.3: Valid float type names
FLOAT_TYPES = frozenset({'f16', 'f32', 'f64', 'f80', 'f128'})
# §3: Built-in simple type names
SIMPLE_TYPES = frozenset({'bool', 'string', 'null'})
# §4.2: Default integer range — untyped integer literals default to i64
I64_MIN = -(1 << 63)
I64_MAX = (1 << 63) - 1
# §D.5: Sentinel for speculative evaluation failure
SPECULATIVE_FAILED = object()
