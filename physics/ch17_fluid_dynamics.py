"""Capitulo 17. Dinamica de fluidos."""
from __future__ import annotations

import numpy as np

from .integrators import integrate, rk4_step

G = 9.81
RHO = 1000.0

# ---------------------------------------------------------------------- Venturi
D1, D2 = 0.20, 0.10
Q = 0.030  # m^3/s
P1 = 2.0e5  # Pa


def areas(d1: float = D1, d2: float = D2):
    return np.pi * d1**2 / 4, np.pi * d2**2 / 4


def venturi_analytic(q: float = Q, p1: float = P1, rho: float = RHO):
    a1, a2 = areas()
    v1, v2 = q / a1, q / a2
    return {"v1": v1, "v2": v2, "p2": p1 + 0.5 * rho * (v1**2 - v2**2)}


def flow_from_pressure_drop_analytic(dp: float, rho: float = RHO) -> float:
    a1, a2 = areas()
    return float(a2 * np.sqrt(2 * dp / (rho * (1 - (a2 / a1) ** 2))))


def flow_from_pressure_drop_numeric(dp: float, rho: float = RHO) -> float:
    """Biseccion sobre Q usando continuidad + Bernoulli: f(Q) = p1 - p2(Q) - dp."""
    a1, a2 = areas()
    f = lambda q: 0.5 * rho * ((q / a2) ** 2 - (q / a1) ** 2) - dp  # noqa: E731
    lo, hi = 0.0, 1.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if f(mid) < 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# --------------------------------------------------------------- Torricelli
A_TANK = 0.5
A_HOLE = 2.0e-4
H0 = 1.5


def torricelli_speed(h, g: float = G):
    return np.sqrt(2 * g * np.asarray(h, dtype=float))


def height_analytic(t, h0: float = H0, a: float = A_HOLE, area: float = A_TANK, g: float = G):
    t = np.asarray(t, dtype=float)
    return np.maximum(np.sqrt(h0) - (a / area) * np.sqrt(g / 2) * t, 0.0) ** 2


def drain_time_analytic(h0: float = H0, a: float = A_HOLE, area: float = A_TANK, g: float = G) -> float:
    return float((area / a) * np.sqrt(2 * h0 / g))


def height_numeric(n: int = 4000, h0: float = H0, a: float = A_HOLE, area: float = A_TANK, g: float = G, t_end: float | None = None):
    t_end = t_end or 1.05 * drain_time_analytic(h0, a, area, g)
    f = lambda _t, y: np.array([-(a / area) * np.sqrt(2 * g * max(y[0], 0.0))])  # noqa: E731
    return integrate(f, np.array([h0]), 0.0, t_end, n, rk4_step)


def drain_time_numeric(n: int = 20000) -> float:
    """Instante en que h cruza 1 mm (la raiz cuadrada hace singular el final); se corrige con la solucion local."""
    t, y = height_numeric(n)
    h = np.maximum(y[:, 0], 0.0)
    i = int(np.argmax(h < 1e-3))
    # cerca de h = 0: h = (a/A sqrt(g/2) (t_e - t))^2  =>  t_e = t + sqrt(h) / ((a/A) sqrt(g/2))
    k = (A_HOLE / A_TANK) * np.sqrt(G / 2)
    return float(t[i - 1] + np.sqrt(h[i - 1]) / k)
