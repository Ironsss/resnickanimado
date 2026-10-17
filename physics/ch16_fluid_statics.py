"""Capitulo 16. Estatica de fluidos."""
from __future__ import annotations

import numpy as np

from .integrators import integrate, rk4_step

G = 9.81
P0 = 101325.0
RHO_WATER = 1000.0

# ------------------------------------------------------ presion vs profundidad / altura
M_AIR = 0.02897
R_GAS = 8.314
T_AIR = 288.15


def water_pressure(depth: float, rho: float = RHO_WATER) -> float:
    return P0 + rho * G * depth


def atmosphere_analytic(h, t: float = T_AIR):
    return P0 * np.exp(-M_AIR * G * np.asarray(h, dtype=float) / (R_GAS * t))


def atmosphere_numeric(h_end: float = 10000.0, n: int = 200, t: float = T_AIR):
    """dp/dh = -rho(p) g con rho = p M / (R T) (gas ideal isotermo)."""
    f = lambda _h, y: np.array([-y[0] * M_AIR * G / (R_GAS * t)])  # noqa: E731
    h, y = integrate(f, np.array([P0]), 0.0, h_end, n, rk4_step)
    return h, y[:, 0]


def atmosphere_constant_density(h, t: float = T_AIR):
    rho0 = P0 * M_AIR / (R_GAS * t)
    return P0 - rho0 * G * np.asarray(h, dtype=float)


# --------------------------------------------------------- cilindro flotante
RHO_OBJ = 600.0
AREA = 0.01
LEN = 0.5
MASS = RHO_OBJ * AREA * LEN


def submerged_depth_analytic(rho_obj: float = RHO_OBJ, rho_f: float = RHO_WATER, length: float = LEN) -> float:
    return rho_obj / rho_f * length


def buoyancy(depth: float, rho_f: float = RHO_WATER, area: float = AREA) -> float:
    return rho_f * G * area * max(depth, 0.0)


def omega_analytic(rho_f: float = RHO_WATER, area: float = AREA, mass: float = MASS) -> float:
    return float(np.sqrt(rho_f * G * area / mass))


def bobbing_numeric(damping: float = 0.0, y0: float = 0.05, t_end: float = 4.0, n: int = 8000):
    """y = profundidad sumergida. m y'' = m g - rho_f g A y - c y'."""
    h_eq = submerged_depth_analytic()
    rhs = lambda _t, s: np.array([s[1], (MASS * G - buoyancy(s[0]) - damping * s[1]) / MASS])  # noqa: E731
    return integrate(rhs, np.array([h_eq + y0, 0.0]), 0.0, t_end, n, rk4_step)


def period_numeric(y0: float = 0.02) -> float:
    t, s = bobbing_numeric(0.0, y0, 2.0, 40000)
    h_eq = submerged_depth_analytic()
    d = s[:, 0] - h_eq
    cross = [i for i in range(1, len(d)) if d[i - 1] > 0 >= d[i]]
    t1 = t[cross[0]] - d[cross[0]] * (t[cross[0]] - t[cross[0] - 1]) / (d[cross[0]] - d[cross[0] - 1])
    t2 = t[cross[1]] - d[cross[1]] * (t[cross[1]] - t[cross[1] - 1]) / (d[cross[1]] - d[cross[1] - 1])
    return float(t2 - t1)


def equilibrium_bisection(rho_obj: float = RHO_OBJ) -> float:
    lo, hi = 0.0, LEN
    mass = rho_obj * AREA * LEN
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if buoyancy(mid) < mass * G:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
