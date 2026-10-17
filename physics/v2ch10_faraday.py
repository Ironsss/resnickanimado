"""Volumen 2, capitulo 10. Ley de induccion de Faraday."""
from __future__ import annotations

import numpy as np

from .integrators import integrate, rk4_step
from .ch07_work_energy import simpson

# --------------------------------------------------------------- bobina giratoria
N_COIL, AREA, B0, FREQ = 100, 0.02, 0.5, 50.0
OMEGA = 2 * np.pi * FREQ


def emf_peak_analytic() -> float:
    return N_COIL * B0 * AREA * OMEGA


def emf_analytic(t):
    return emf_peak_analytic() * np.sin(OMEGA * np.asarray(t, dtype=float))


def flux(t):
    return N_COIL * B0 * AREA * np.cos(OMEGA * np.asarray(t, dtype=float))


def emf_numeric(t: np.ndarray, h: float = 1e-6) -> np.ndarray:
    return -(flux(t + h) - flux(t - h)) / (2 * h)


def emf_rms_numeric(n: int = 4000) -> float:
    t = np.linspace(0, 1 / FREQ, n + 1)
    return float(np.sqrt(simpson(emf_analytic(t) ** 2, t) * FREQ))


# ------------------------------------------------------------------ barra deslizante
B_ROD, L_ROD, M_ROD, R_CIRC, V0_ROD = 0.8, 0.5, 0.2, 2.0, 4.0


def tau_rod() -> float:
    return M_ROD * R_CIRC / (B_ROD**2 * L_ROD**2)


def rod_velocity_analytic(t):
    return V0_ROD * np.exp(-np.asarray(t, dtype=float) / tau_rod())


def rod_numeric(t_end: float = 12.0, n: int = 2400):
    k = B_ROD**2 * L_ROD**2 / (M_ROD * R_CIRC)
    return integrate(lambda _t, y: np.array([y[1], -k * y[1]]), np.array([0.0, V0_ROD]), 0.0, t_end, n, rk4_step)


def rod_energy_dissipated_numeric(t_end: float = 25.0, n: int = 10000) -> float:
    t, y = rod_numeric(t_end, n)
    emf = B_ROD * L_ROD * y[:, 1]
    current = emf / R_CIRC
    return simpson(current**2 * R_CIRC, t)


def rod_distance_analytic() -> float:
    return V0_ROD * tau_rod()
