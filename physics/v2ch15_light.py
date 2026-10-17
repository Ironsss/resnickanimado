"""Volumen 2, capitulo 15. Naturaleza y propagacion de la luz."""
from __future__ import annotations

import numpy as np

from .constants import C_LIGHT

N1, N2 = 1.0, 1.5
H1, H2, THETA1_DEG = 1.0, 1.5, 40.0


def snell_angle(theta1: float, n1: float = N1, n2: float = N2) -> float:
    return float(np.arcsin(n1 * np.sin(theta1) / n2))


def refraction_point_analytic(h1: float = H1, theta1: float = np.deg2rad(THETA1_DEG)) -> float:
    """Abscisa del punto de incidencia de un rayo que sale de (0, h1) con angulo de incidencia theta1."""
    return float(h1 * np.tan(theta1))


def target_point(h1: float = H1, h2: float = H2, theta1: float = np.deg2rad(THETA1_DEG)):
    x = refraction_point_analytic(h1, theta1)
    return x + h2 * np.tan(snell_angle(theta1)), -h2


def travel_time(x: np.ndarray, d: float, h1: float = H1, h2: float = H2, n1: float = N1, n2: float = N2):
    x = np.asarray(x, dtype=float)
    return (n1 * np.hypot(x, h1) + n2 * np.hypot(d - x, h2)) / C_LIGHT


def fermat_numeric(d: float, h1: float = H1, h2: float = H2) -> float:
    """Minimiza el tiempo de viaje (principio de Fermat) por busqueda de la seccion aurea."""
    a, b = 0.0, d
    g = (np.sqrt(5) - 1) / 2
    c, e = b - g * (b - a), a + g * (b - a)
    for _ in range(200):
        if travel_time(c, d, h1, h2) < travel_time(e, d, h1, h2):
            b = e
        else:
            a = c
        c, e = b - g * (b - a), a + g * (b - a)
    return float(0.5 * (a + b))


def critical_angle_analytic(n_dense: float = 1.5, n_thin: float = 1.0) -> float:
    return float(np.degrees(np.arcsin(n_thin / n_dense)))


def critical_angle_numeric(n_dense: float = 1.5, n_thin: float = 1.0) -> float:
    """Biseccion sobre el angulo de incidencia: el rayo refractado deja de existir cuando n1 sen(t) > n2."""
    lo, hi = 0.0, 90.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if n_dense * np.sin(np.deg2rad(mid)) <= n_thin:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ----------------------------------------------------------------------- prisma y dispersion
APEX_DEG = 60.0
CAUCHY_A, CAUCHY_B = 1.50, 5000.0  # n = A + B / lambda^2, lambda en nm


def cauchy(wavelength_nm):
    return CAUCHY_A + CAUCHY_B / np.asarray(wavelength_nm, dtype=float) ** 2


def min_deviation_analytic(n, apex_deg: float = APEX_DEG):
    a = np.deg2rad(apex_deg)
    return np.degrees(2 * np.arcsin(np.asarray(n) * np.sin(a / 2)) - a)


def deviation(theta_in_deg: float, n: float, apex_deg: float = APEX_DEG) -> float:
    """Traza un rayo por las dos caras del prisma aplicando la ley de Snell en cada una."""
    a = np.deg2rad(apex_deg)
    t1 = np.deg2rad(theta_in_deg)
    r1 = np.arcsin(np.sin(t1) / n)
    r2 = a - r1
    s = n * np.sin(r2)
    if s >= 1:
        return float("nan")
    t2 = np.arcsin(s)
    return float(np.degrees(t1 + t2 - a))


def min_deviation_numeric(n: float, apex_deg: float = APEX_DEG) -> float:
    lo, hi = 10.0, 85.0
    g = (np.sqrt(5) - 1) / 2
    c, e = hi - g * (hi - lo), lo + g * (hi - lo)
    for _ in range(100):
        if deviation(c, n, apex_deg) < deviation(e, n, apex_deg):
            hi = e
        else:
            lo = c
        c, e = hi - g * (hi - lo), lo + g * (hi - lo)
    return deviation(0.5 * (lo + hi), n, apex_deg)
