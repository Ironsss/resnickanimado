"""Volumen 2, capitulo 1. Carga electrica y ley de Coulomb."""
from __future__ import annotations

import numpy as np

from .constants import K_E

# ----------------------------------------------------------- fuerza sobre q3 en el plano
Q = np.array([3e-6, -2e-6, 1e-6])
POS = np.array([[0.0, 0.0], [0.4, 0.0], [0.2, 0.3]])


def force_analytic(index: int = 2, q=Q, pos=POS) -> np.ndarray:
    """Suma vectorial de las fuerzas de Coulomb sobre la carga `index`."""
    f = np.zeros(2)
    for j in range(len(q)):
        if j == index:
            continue
        r = pos[index] - pos[j]
        f += K_E * q[index] * q[j] * r / np.linalg.norm(r) ** 3
    return f


def potential_energy(pos, q=Q) -> float:
    u = 0.0
    for i in range(len(q)):
        for j in range(i + 1, len(q)):
            u += K_E * q[i] * q[j] / np.linalg.norm(pos[i] - pos[j])
    return float(u)


def force_numeric(index: int = 2, h: float = 1e-6, q=Q, pos=POS) -> np.ndarray:
    """F = -grad U con diferencias centradas respecto de la posicion de la carga."""
    f = np.zeros(2)
    for axis in range(2):
        p_plus, p_minus = pos.copy(), pos.copy()
        p_plus[index, axis] += h
        p_minus[index, axis] -= h
        f[axis] = -(potential_energy(p_plus, q) - potential_energy(p_minus, q)) / (2 * h)
    return f


def newton_third_law_residual(q=Q, pos=POS) -> float:
    total = sum(force_analytic(i, q, pos) for i in range(len(q)))
    return float(np.linalg.norm(total))


# --------------------------------------------------------- punto de equilibrio en una recta
Q1, Q2, SEP = 4e-6, 1e-6, 0.6


def zero_force_point_analytic(q1: float = Q1, q2: float = Q2, d: float = SEP) -> float:
    """Entre dos cargas del mismo signo: x = d sqrt(q1) / (sqrt(q1) + sqrt(q2))."""
    return float(d * np.sqrt(abs(q1)) / (np.sqrt(abs(q1)) + np.sqrt(abs(q2))))


def net_force_on_test(x, q1: float = Q1, q2: float = Q2, d: float = SEP, qt: float = 1e-9):
    x = np.asarray(x, dtype=float)
    return K_E * qt * (q1 / x**2 - q2 / (d - x) ** 2)


def zero_force_point_numeric(q1: float = Q1, q2: float = Q2, d: float = SEP) -> float:
    lo, hi = 1e-6 * d, d * (1 - 1e-6)
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if net_force_on_test(mid, q1, q2, d) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def opposite_charges_point_analytic(q1: float = 4e-6, q2: float = -1e-6, d: float = SEP) -> float:
    """Con signos opuestos el punto de equilibrio esta fuera del segmento, del lado de la carga menor."""
    r = np.sqrt(abs(q1 / q2))
    return float(d * r / (r - 1.0))  # medida desde q1, hacia q2 y mas alla


def opposite_charges_point_numeric(q1: float = 4e-6, q2: float = -1e-6, d: float = SEP) -> float:
    f = lambda x: q1 / x**2 + q2 / (x - d) ** 2  # noqa: E731  (q2 negativo; x > d)
    lo, hi = d * (1 + 1e-9), 100 * d
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)
