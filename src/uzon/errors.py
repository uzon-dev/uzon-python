# SPDX-FileCopyrightText: © 2026 Suho Kang
# SPDX-License-Identifier: MIT
"""UZON error hierarchy per §11.2.

Error priority (§11.2): syntax → circular → type → runtime.
All errors carry source location (file, line, col) per §11.2.0.
"""


class UzonError(Exception):
    """Base class for all UZON errors."""

    def __init__(
        self,
        message: str,
        line: int | None = None,
        col: int | None = None,
        file: str | None = None,
    ):
        self.line = line
        self.col = col
        self.file = file
        parts: list[str] = []
        if file is not None and file != "<string>":
            parts.append(f"File {file}")
        if line is not None and col is not None:
            parts.append(f"Line {line}, col {col}")
        elif line is not None:
            parts.append(f"Line {line}")
        if parts:
            super().__init__(f"{', '.join(parts)}: {message}")
        else:
            super().__init__(message)


class UzonSyntaxError(UzonError):
    """Lexer and parser errors (§11.2 priority 1)."""


class UzonTypeError(UzonError):
    """Type annotation, conversion, and compatibility errors (§11.2 priority 3)."""


class UzonRuntimeError(UzonError):
    """Evaluation errors: overflow, division by zero, etc. (§11.2 priority 4)."""


class UzonCircularError(UzonError):
    """Circular dependency or circular import errors (§11.2 priority 2)."""
