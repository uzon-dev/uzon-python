# SPDX-FileCopyrightText: © 2026 Suho Kang
# SPDX-License-Identifier: MIT
"""Float-to-string formatting per §5.11.2.

Rules:
- Shortest decimal string that round-trips to the same float value.
- If 0 < n <= 21: plain decimal notation (e.g., 3.14, 10000000.0).
- If -6 < n <= 0: plain decimal with leading zeros (e.g., 0.000001).
- Otherwise: scientific notation with one digit before the decimal point
  (e.g., 1.5e100, 3.0e-8).
- Result always contains a decimal point.
- inf → "inf", -inf → "-inf", nan/-nan → "nan".
"""

from __future__ import annotations

import math


def format_float(val: float) -> str:
    """Format a float according to §5.11.2."""
    if math.isnan(val):
        return "nan"  # §5.2: -nan is semantically identical to nan
    if val == float("inf"):
        return "inf"
    if val == float("-inf"):
        return "-inf"

    # Handle zero (including -0.0)
    if val == 0.0:
        if math.copysign(1.0, val) < 0:
            return "-0.0"
        return "0.0"

    # repr() gives the shortest round-trip representation
    r = float.__repr__(val)

    abs_val = abs(val)
    n = math.floor(math.log10(abs_val)) + 1

    negative = val < 0
    sign = "-" if negative else ""

    if 0 < n <= 21:
        # Plain decimal notation
        if "e" in r or "E" in r:
            parts = r.lstrip("-").lower().split("e")
            mantissa_str = parts[0]
            exp = int(parts[1])

            if "." in mantissa_str:
                int_part, frac_part = mantissa_str.split(".")
            else:
                int_part = mantissa_str
                frac_part = ""

            digits = int_part + frac_part
            dot_pos = 1 + exp

            if dot_pos >= len(digits):
                s = digits + "0" * (dot_pos - len(digits)) + ".0"
            else:
                s = digits[:dot_pos] + "." + digits[dot_pos:]
                s = s.rstrip("0")
                if s.endswith("."):
                    s += "0"

            return sign + s
        else:
            if "." not in r:
                r += ".0"
            return r

    elif -6 < n <= 0:
        # Plain decimal with leading zeros (e.g., 0.000001)
        if "e" in r or "E" in r:
            parts = r.lstrip("-").lower().split("e")
            mantissa_str = parts[0]
            exp = int(parts[1])

            if "." in mantissa_str:
                int_part, frac_part = mantissa_str.split(".")
            else:
                int_part = mantissa_str
                frac_part = ""

            digits = int_part + frac_part
            dot_pos = 1 + exp
            s = "0." + "0" * (-dot_pos) + digits
            s = s.rstrip("0")
            if s.endswith("."):
                s += "0"
            return sign + s
        else:
            if "." not in r:
                r += ".0"
            return r

    else:
        # Scientific notation: one digit before the decimal point
        if "e" in r or "E" in r:
            parts = r.lstrip("-").lower().split("e")
            mantissa = parts[0]
            exp = int(parts[1])
            if "." not in mantissa:
                mantissa += ".0"
            return f"{sign}{mantissa}e{exp}"
        else:
            exp = n - 1
            shifted = abs_val / (10.0**exp)
            mantissa = repr(shifted)
            if "." not in mantissa:
                mantissa += ".0"
            return f"{sign}{mantissa}e{exp}"
