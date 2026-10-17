"""Volumen 2, capitulo 21. Ecuacion de Schrodinger."""
from __future__ import annotations

import numpy as np
from scipy.linalg import eigh_tridiagonal

from .constants import E_CH, HBAR, M_E

L_WELL = 1.0e-9


def well_energy_analytic(n, length: float = L_WELL, m: float = M_E):
    n = np.asarray(n, dtype=float)
    return n**2 * np.pi**2 * HBAR**2 / (2 * m * length**2) / E_CH  # eV


def well_numeric(n_states: int = 4, length: float = L_WELL, m: float = M_E, npts: int = 2000):
    """Diagonalizacion del hamiltoniano de diferencias finitas (tridiagonal) en el pozo infinito. Devuelve E [eV], x, psi."""
    h = length / (npts + 1)
    t = HBAR**2 / (2 * m * h**2)
    diag = np.full(npts, 2 * t)
    off = np.full(npts - 1, -t)
    w, v = eigh_tridiagonal(diag, off, select="i", select_range=(0, n_states - 1))
    x = np.linspace(h, length - h, npts)
    return w / E_CH, x, v / np.sqrt(h)


def well_wavefunction(n: int, x, length: float = L_WELL):
    return np.sqrt(2 / length) * np.sin(n * np.pi * np.asarray(x) / length)


def wavefunction_overlap(n: int = 1, npts: int = 2000) -> float:
    _, x, v = well_numeric(n, npts=npts)
    psi = v[:, n - 1]
    ref = well_wavefunction(n, x)
    h = x[1] - x[0]
    return float(abs(np.sum(psi * ref) * h))


# ----------------------------------------------------------------------------------- barrera
E_BEAM_EV, V0_EV, A_BAR = 5.0, 10.0, 0.5e-9


def transmission_analytic(e_ev: float = E_BEAM_EV, v0_ev: float = V0_EV, a: float = A_BAR, m: float = M_E) -> float:
    e, v0 = e_ev * E_CH, v0_ev * E_CH
    if e < v0:
        kappa = np.sqrt(2 * m * (v0 - e)) / HBAR
        return float(1 / (1 + v0**2 * np.sinh(kappa * a) ** 2 / (4 * e * (v0 - e))))
    q = np.sqrt(2 * m * (e - v0)) / HBAR
    return float(1 / (1 + v0**2 * np.sin(q * a) ** 2 / (4 * e * (e - v0))))


def transmission_numeric(e_ev: float = E_BEAM_EV, v0_ev: float = V0_EV, a: float = A_BAR, m: float = M_E, n: int = 4000) -> float:
    """Integra con RK4 la ecuacion de Schrodinger estacionaria desde la derecha (onda transmitida pura) y descompone en incidente/reflejada."""
    e, v0 = e_ev * E_CH, v0_ev * E_CH
    k = np.sqrt(2 * m * e) / HBAR
    x_right = a
    psi = np.array([1.0 + 0j, 1j * k])  # psi = exp(ik(x - a)), psi' = ik psi
    h = -a / n
    f = lambda y: np.array([y[1], -(2 * m / HBAR**2) * (e - v0) * y[0]])  # noqa: E731
    for _ in range(n):
        k1 = f(psi)
        k2 = f(psi + 0.5 * h * k1)
        k3 = f(psi + 0.5 * h * k2)
        k4 = f(psi + h * k3)
        psi = psi + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    # en x = 0: psi = A exp(ikx) + B exp(-ikx)
    amp_inc = 0.5 * (psi[0] + psi[1] / (1j * k))
    return float(1.0 / abs(amp_inc) ** 2)
