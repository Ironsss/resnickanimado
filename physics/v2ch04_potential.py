"""Volumen 2, capitulo 4. Potencial electrico."""
from __future__ import annotations

import numpy as np

from .constants import EPS0, K_E
from .ch07_work_energy import simpson

# ------------------------------------------------------------ anillo y disco
Q_RING = 10e-9
A_RING = 0.10
SIGMA = 2e-6  # C/m^2 (disco)
R_DISK = 0.10


def ring_potential_analytic(z, q: float = Q_RING, a: float = A_RING):
    return K_E * q / np.sqrt(np.asarray(z, dtype=float) ** 2 + a**2)


def disk_potential_analytic(z, sigma: float = SIGMA, r: float = R_DISK):
    z = np.asarray(z, dtype=float)
    return sigma / (2 * EPS0) * (np.sqrt(z**2 + r**2) - z)


def disk_potential_numeric(z: float, sigma: float = SIGMA, r: float = R_DISK, n: int = 400) -> float:
    """Superposicion de anillos: dV = k dq / sqrt(s^2 + z^2) con dq = sigma 2 pi s ds."""
    s = np.linspace(0, r, n + 1)
    dq_ds = sigma * 2 * np.pi * s
    return simpson(K_E * dq_ds / np.sqrt(s**2 + z**2), s)


def field_from_potential_numeric(v_func, z: float, h: float = 1e-6) -> float:
    return float(-(v_func(z + h) - v_func(z - h)) / (2 * h))


def disk_field_analytic(z, sigma: float = SIGMA, r: float = R_DISK):
    z = np.asarray(z, dtype=float)
    return sigma / (2 * EPS0) * (1 - z / np.sqrt(z**2 + r**2))


# ----------------------------------------------------------- dos cargas puntuales
QA, QB, D = 4e-9, -4e-9, 0.20


def potential_pair(x, y, qa: float = QA, qb: float = QB, d: float = D):
    ra = np.hypot(x + d / 2, y)
    rb = np.hypot(x - d / 2, y)
    return K_E * (qa / ra + qb / rb)


def field_pair_analytic(x, y, qa: float = QA, qb: float = QB, d: float = D):
    ra = np.hypot(x + d / 2, y)
    rb = np.hypot(x - d / 2, y)
    ex = K_E * (qa * (x + d / 2) / ra**3 + qb * (x - d / 2) / rb**3)
    ey = K_E * (qa * y / ra**3 + qb * y / rb**3)
    return ex, ey


def field_grid_numeric(extent: float = 0.5, n: int = 801):
    """E = -grad V sobre una malla con np.gradient (diferencias centradas)."""
    xs = np.linspace(-extent, extent, n)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    V = potential_pair(X, Y)
    dx = xs[1] - xs[0]
    gx, gy = np.gradient(V, dx, dx)
    return xs, -gx, -gy


def field_grid_error(extent: float = 0.5, n: int = 801, mask_radius: float = 0.12) -> float:
    xs, ex, ey = field_grid_numeric(extent, n)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    eax, eay = field_pair_analytic(X, Y)
    far = (np.hypot(X + D / 2, Y) > mask_radius) & (np.hypot(X - D / 2, Y) > mask_radius)
    mag = np.hypot(eax, eay)
    return float(np.max(np.hypot(ex - eax, ey - eay)[far] / mag[far]))


def interaction_energy(qa: float = QA, qb: float = QB, d: float = D) -> float:
    return K_E * qa * qb / d
