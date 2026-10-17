"""Capitulo 11. Cinematica rotacional."""
from __future__ import annotations

import numpy as np

from .integrators import trapezoid
from .ch07_work_energy import simpson

ALPHA = 1.5  # rad/s^2
T_END = 8.0
RIM = 0.4  # m


def omega_analytic(t, alpha: float = ALPHA):
    return alpha * np.asarray(t, dtype=float)


def theta_analytic(t, alpha: float = ALPHA):
    return 0.5 * alpha * np.asarray(t, dtype=float) ** 2


def rim_accelerations(t: float, alpha: float = ALPHA, r: float = RIM):
    w = alpha * t
    return {"a_t": alpha * r, "a_c": w**2 * r, "a": np.hypot(alpha * r, w**2 * r), "v": w * r}


def spin_up_numeric(n: int = 400, alpha: float = ALPHA, t_end: float = T_END):
    """Integra alpha(t) dos veces con la regla del trapecio acumulada."""
    t = np.linspace(0, t_end, n + 1)
    a = np.full_like(t, alpha)
    w = np.concatenate([[0.0], np.cumsum(0.5 * (a[1:] + a[:-1]) * np.diff(t))])
    th = np.concatenate([[0.0], np.cumsum(0.5 * (w[1:] + w[:-1]) * np.diff(t))])
    return t, w, th


def spin_up_numeric_variable(n: int = 4000, t_end: float = T_END):
    """alpha(t) = 0.25 t: omega = 0.125 t^2, theta = t^3/24 (comprueba el metodo con aceleracion variable)."""
    t = np.linspace(0, t_end, n + 1)
    a = 0.25 * t
    w = np.concatenate([[0.0], np.cumsum(0.5 * (a[1:] + a[:-1]) * np.diff(t))])
    th = np.concatenate([[0.0], np.cumsum(0.5 * (w[1:] + w[:-1]) * np.diff(t))])
    return t, w, th


# ------------------------------------------------------- rodadura sin deslizar (cicloide)
R_WHEEL = 0.5


def cycloid(theta, r: float = R_WHEEL):
    theta = np.asarray(theta, dtype=float)
    return r * (theta - np.sin(theta)), r * (1 - np.cos(theta))


def arch_length_analytic(r: float = R_WHEEL) -> float:
    return 8.0 * r


def arch_area_analytic(r: float = R_WHEEL) -> float:
    return 3.0 * np.pi * r**2


def arch_length_numeric(n: int = 2000, r: float = R_WHEEL) -> float:
    th = np.linspace(0, 2 * np.pi, n + 1)
    speed = 2 * r * np.abs(np.sin(th / 2))  # |dr/dtheta| para v_cm = r (omega = 1)
    return simpson(speed, th)


def arch_area_numeric(n: int = 2000, r: float = R_WHEEL) -> float:
    th = np.linspace(0, 2 * np.pi, n + 1)
    y = r * (1 - np.cos(th))
    dxdth = r * (1 - np.cos(th))
    return simpson(y * dxdth, th)


def arch_length_polyline(n: int = 2000, r: float = R_WHEEL) -> float:
    x, y = cycloid(np.linspace(0, 2 * np.pi, n + 1), r)
    return float(np.sum(np.hypot(np.diff(x), np.diff(y))))
