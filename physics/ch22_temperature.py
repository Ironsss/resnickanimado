"""Capitulo 22. Temperatura y ley cero."""
from __future__ import annotations

import numpy as np

from .integrators import integrate, rk4_step

# ------------------------------------------------------------- dilatacion termica
L0 = 30.0  # m
ALPHA0 = 12e-6  # 1/K
BETA = 2.0e-3  # 1/K: alpha(T) = alpha0 (1 + beta (T - T0))
T0, T1 = 20.0, 120.0  # grados Celsius


def alpha(temp, alpha0: float = ALPHA0, beta: float = BETA, t0: float = T0):
    return alpha0 * (1.0 + beta * (np.asarray(temp, dtype=float) - t0))


def length_linear(t_final: float = T1, length: float = L0, alpha0: float = ALPHA0, t0: float = T0) -> float:
    return length * (1.0 + alpha0 * (t_final - t0))


def length_analytic(t_final: float = T1, length: float = L0, alpha0: float = ALPHA0, beta: float = BETA, t0: float = T0) -> float:
    """dL/dT = alpha(T) L  =>  L = L0 exp( alpha0 (dT + beta dT^2 / 2) )."""
    dt = t_final - t0
    return float(length * np.exp(alpha0 * (dt + beta * dt**2 / 2)))


def length_numeric(t_final: float = T1, n: int = 400, length: float = L0) -> float:
    t, y = integrate(lambda temp, y: np.array([alpha(temp) * y[0]]), np.array([length]), T0, t_final, n, rk4_step)
    return float(y[-1, 0])


def expansion_table(temps=(40.0, 70.0, 120.0)):
    return [{"T": t, "linear": length_linear(t) - L0, "exact": length_analytic(t) - L0, "numeric": length_numeric(t) - L0} for t in temps]


# ------------------------------------------------------------- equilibrio termico
M1, C1, T1_0 = 0.30, 900.0, 150.0  # aluminio
M2, C2, T2_0 = 0.50, 4186.0, 20.0  # agua
H = 20.0  # W/K


def final_temperature_analytic() -> float:
    return (M1 * C1 * T1_0 + M2 * C2 * T2_0) / (M1 * C1 + M2 * C2)


def rate_analytic() -> float:
    return H * (1 / (M1 * C1) + 1 / (M2 * C2))


def temperatures_analytic(t):
    tf = final_temperature_analytic()
    return tf + (T1_0 - tf) * np.exp(-rate_analytic() * np.asarray(t, dtype=float)), tf + (T2_0 - tf) * np.exp(-rate_analytic() * np.asarray(t, dtype=float))


def temperatures_numeric(t_end: float = 60.0, n: int = 1200):
    rhs = lambda _t, y: np.array([-H * (y[0] - y[1]) / (M1 * C1), H * (y[0] - y[1]) / (M2 * C2)])  # noqa: E731
    return integrate(rhs, np.array([T1_0, T2_0]), 0.0, t_end, n, rk4_step)


def to_fahrenheit(c): return c * 9 / 5 + 32  # noqa: E704
def to_kelvin(c): return c + 273.15  # noqa: E704
