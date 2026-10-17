"""Capitulo 2. Movimiento en una dimension."""
from __future__ import annotations

import numpy as np

from .integrators import euler_step, integrate, rk4_step, riemann_left, trapezoid

# ------------------------------------------- velocidad instantanea como limite
# x(t) = 2 + 3 t - t^2/2 + t^3/10   [m, s]


def x_of_t(t):
    return 2.0 + 3.0 * t - 0.5 * t**2 + 0.1 * t**3


def v_analytic(t):
    return 3.0 - t + 0.3 * t**2


def a_analytic(t):
    return -1.0 + 0.6 * t


def secant_slope(t0: float, dt: float) -> float:
    return float((x_of_t(t0 + dt) - x_of_t(t0)) / dt)


def central_second_difference(t0: float, h: float) -> float:
    return float((x_of_t(t0 + h) - 2 * x_of_t(t0) + x_of_t(t0 - h)) / h**2)


def secant_table(t0: float = 2.0, dts=(1.0, 0.5, 0.1, 0.01, 0.001)) -> list[dict[str, float]]:
    v_exact = float(v_analytic(t0))
    return [
        {"dt": dt, "slope": secant_slope(t0, dt), "error": abs(secant_slope(t0, dt) - v_exact)}
        for dt in dts
    ]


def displacement_from_velocity(t0: float, t1: float, n: int) -> dict[str, float]:
    """Desplazamiento como area bajo v(t): analitico, Riemann y trapecio."""
    t = np.linspace(t0, t1, n + 1)
    exact = float(x_of_t(t1) - x_of_t(t0))
    return {
        "exact": exact,
        "riemann": riemann_left(v_analytic(t), t),
        "trapezoid": trapezoid(v_analytic(t), t),
    }


# ------------------------------------------------ aceleracion constante / caida
V0 = 20.0
G = 9.81


def free_fall_analytic(t, v0: float = V0, g: float = G):
    t = np.asarray(t, dtype=float)
    return v0 * t - 0.5 * g * t**2, v0 - g * t


def free_fall_summary(v0: float = V0, g: float = G) -> dict[str, float]:
    return {
        "t_peak": v0 / g,
        "y_max": v0**2 / (2 * g),
        "t_return": 2 * v0 / g,
        "v_return": -v0,
    }


def free_fall_numeric(n: int, method: str = "rk4", t1: float = 3.0, v0: float = V0, g: float = G):
    step = rk4_step if method == "rk4" else euler_step

    def rhs(_t, s):
        return np.array([s[1], -g])

    return integrate(rhs, np.array([0.0, v0]), 0.0, t1, n, step)


def free_fall_convergence(t1: float = 3.0, ns=(10, 20, 40, 80, 160)) -> list[dict[str, float]]:
    y_exact = free_fall_analytic(t1)[0]
    rows = []
    for n in ns:
        _, ye = free_fall_numeric(n, "euler", t1)
        _, yr = free_fall_numeric(n, "rk4", t1)
        rows.append(
            {
                "n": n,
                "dt": t1 / n,
                "err_euler": abs(ye[-1, 0] - float(y_exact)),
                "err_rk4": abs(yr[-1, 0] - float(y_exact)),
            }
        )
    return rows
