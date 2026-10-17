"""Volumen 2, capitulo 22. Estructura atomica."""
from __future__ import annotations

import numpy as np
from scipy.linalg import eigh_tridiagonal

from .ch07_work_energy import simpson

HARTREE_EV = 27.211386245988
RYDBERG_EV = 13.605693122994
H_C_EV_NM = 1239.84198  # h c en eV nm
RINF_M = 10973731.568160  # constante de Rydberg, 1/m


def level_analytic(n):
    return -RYDBERG_EV / np.asarray(n, dtype=float) ** 2


def radial_numeric(n_states: int = 4, rmax: float = 160.0, npts: int = 32000):
    """Ecuacion radial de l = 0 en unidades atomicas: -u''/2 - u/r = E u, diagonalizada en una malla uniforme. Devuelve E [eV], r, u."""
    h = rmax / (npts + 1)
    r = np.arange(1, npts + 1) * h
    diag = 1.0 / h**2 - 1.0 / r
    off = np.full(npts - 1, -0.5 / h**2)
    w, v = eigh_tridiagonal(diag, off, select="i", select_range=(0, n_states - 1))
    u = v / np.sqrt(h)
    return w * HARTREE_EV, r, u


def balmer_wavelength_analytic(n: int) -> float:
    return float(1e9 / (RINF_M * (0.25 - 1.0 / n**2)))


def balmer_wavelength_numeric(n: int, energies_ev: np.ndarray) -> float:
    return float(H_C_EV_NM / (energies_ev[n - 1] - energies_ev[1]))


def radial_probability_analytic_1s(r):
    r = np.asarray(r, dtype=float)
    return 4 * r**2 * np.exp(-2 * r)


def mean_radius_1s_numeric() -> float:
    _, r, u = radial_numeric(1)
    p = u[:, 0] ** 2
    m = len(r) - (1 if len(r) % 2 == 0 else 0)  # Simpson necesita un numero impar de nodos
    return float(simpson(r[:m] * p[:m], r[:m]))


def most_probable_radius_numeric() -> float:
    _, r, u = radial_numeric(1)
    p = u[:, 0] ** 2
    k = int(np.argmax(p))
    y0, y1, y2 = p[k - 1], p[k], p[k + 1]
    return float(r[k] + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2) * (r[1] - r[0]))
