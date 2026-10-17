"""Volumen 2, capitulo 17. Interferencia."""
from __future__ import annotations

import numpy as np

LAM, D_SLIT, L_SCR = 600e-9, 0.25e-3, 1.5


def fringe_position_analytic(m: int, lam: float = LAM, d: float = D_SLIT, big_l: float = L_SCR) -> float:
    return float(big_l * np.tan(np.arcsin(m * lam / d)))


def intensity_analytic(y, lam: float = LAM, d: float = D_SLIT, big_l: float = L_SCR):
    y = np.asarray(y, dtype=float)
    sin_t = y / np.hypot(y, big_l)
    return np.cos(np.pi * d * sin_t / lam) ** 2


def intensity_numeric(y, lam: float = LAM, d: float = D_SLIT, big_l: float = L_SCR):
    """Suma de dos ondas esfericas con las distancias exactas a cada rendija."""
    y = np.asarray(y, dtype=float)
    r1 = np.hypot(big_l, y - d / 2)
    r2 = np.hypot(big_l, y + d / 2)
    k = 2 * np.pi / lam
    field = np.exp(1j * k * r1) / r1 + np.exp(1j * k * r2) / r2
    ref = 2.0 / big_l
    return (np.abs(field) / ref) ** 2


def bright_fringe_numeric(m: int, lam: float = LAM, d: float = D_SLIT, big_l: float = L_SCR) -> float:
    """Biseccion sobre la diferencia de camino exacta r2 - r1 = m lambda."""
    f = lambda y: np.hypot(big_l, y + d / 2) - np.hypot(big_l, y - d / 2) - m * lam  # noqa: E731
    lo, hi = 0.0, 3 * m * lam * big_l / d + 1e-9
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if f(mid) < 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------- pelicula delgada
N_FILM, N_SUB, T_FILM, LAM0 = 1.38, 1.52, 550e-9 / (4 * 1.38), 550e-9


def reflectance_analytic(lam, n_film: float = N_FILM, n_sub: float = N_SUB, t: float = T_FILM):
    """Formula de Airy para una pelicula sobre un sustrato, incidencia normal desde el aire."""
    lam = np.asarray(lam, dtype=float)
    r1 = (1 - n_film) / (1 + n_film)
    r2 = (n_film - n_sub) / (n_film + n_sub)
    delta = 2 * np.pi * n_film * t / lam
    e = np.exp(2j * delta)
    return np.abs((r1 + r2 * e) / (1 + r1 * r2 * e)) ** 2


def reflectance_matrix(lam, n_film: float = N_FILM, n_sub: float = N_SUB, t: float = T_FILM):
    """Metodo de la matriz de transferencia (matriz caracteristica de la capa)."""
    lam = np.atleast_1d(np.asarray(lam, dtype=float))
    out = np.empty_like(lam)
    for i, l in enumerate(lam):
        delta = 2 * np.pi * n_film * t / l
        m = np.array([[np.cos(delta), 1j * np.sin(delta) / n_film], [1j * n_film * np.sin(delta), np.cos(delta)]])
        b, c = m @ np.array([1.0, n_sub])
        r = (1 * b - c) / (1 * b + c)
        out[i] = abs(r) ** 2
    return out


def bare_reflectance(n_sub: float = N_SUB) -> float:
    return float(((1 - n_sub) / (1 + n_sub)) ** 2)
