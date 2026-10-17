"""Integradores de EDO de uso general (Euler, punto medio, RK4)."""
from __future__ import annotations

from typing import Callable

import numpy as np

Rhs = Callable[[float, np.ndarray], np.ndarray]


def euler_step(f: Rhs, t: float, y: np.ndarray, h: float) -> np.ndarray:
    return y + h * f(t, y)


def rk4_step(f: Rhs, t: float, y: np.ndarray, h: float) -> np.ndarray:
    k1 = f(t, y)
    k2 = f(t + h / 2, y + h / 2 * k1)
    k3 = f(t + h / 2, y + h / 2 * k2)
    k4 = f(t + h, y + h * k3)
    return y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def integrate(
    f: Rhs,
    y0: np.ndarray,
    t0: float,
    t1: float,
    n: int,
    step: Callable[..., np.ndarray] = rk4_step,
) -> tuple[np.ndarray, np.ndarray]:
    """Integra y' = f(t, y) con n pasos uniformes.

    Devuelve (t, y) con t de forma (n+1,) y y de forma (n+1, dim).
    """
    t = np.linspace(t0, t1, n + 1)
    h = (t1 - t0) / n
    y = np.empty((n + 1, np.size(y0)))
    y[0] = y0
    for i in range(n):
        y[i + 1] = step(f, t[i], y[i], h)
    return t, y


def trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))


def riemann_left(y: np.ndarray, x: np.ndarray) -> float:
    return float(np.sum(y[:-1] * np.diff(x)))
