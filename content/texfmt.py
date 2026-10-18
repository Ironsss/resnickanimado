"""Formato de numeros para TeX en las soluciones paso a paso."""
from __future__ import annotations

import math


def sci(x: float, d: int = 4) -> str:
    """Notacion cientifica TeX: sci(3.0e5) -> '3.000\\times10^{5}'. Usa decimales fijos si 0.01 <= |x| < 1e4."""
    if x == 0:
        return "0"
    if 1e-2 <= abs(x) < 1e4:
        dec = max(d - 1 - math.floor(math.log10(abs(x))), 0)  # d cifras significativas, sin notacion e
        return f"{x:.{dec}f}"
    m, e = f"{x:.{d - 1}e}".split("e")
    return f"{m}\\times10^{{{int(e)}}}"


def fx(x: float, d: int = 4) -> str:
    """Decimal fijo con d decimales."""
    return f"{x:.{d}f}"
