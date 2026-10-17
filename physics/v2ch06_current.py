"""Volumen 2, capitulo 6. Corriente y resistencia."""
from __future__ import annotations

import numpy as np

from .constants import E_CH, M_E, K_B
from .ch07_work_energy import simpson

# ----------------------------------------------------------------- modelo de Drude
E_FIELD = 100.0  # V/m
TAU = 2.5e-14  # s
N_CU = 8.5e28
T_K = 300.0


def drift_speed_analytic(e: float = E_FIELD, tau: float = TAU) -> float:
    return E_CH * e * tau / M_E


def conductivity_analytic(n: float = N_CU, tau: float = TAU) -> float:
    return n * E_CH**2 * tau / M_E


E_SIM = 2.0e6  # V/m: campo grande solo para que la senal supere el ruido estadistico (el modelo es lineal en E)


def drift_mobility_analytic(tau: float = TAU) -> float:
    return E_CH * tau / M_E


def drift_mobility_monte_carlo(n_e: int = 20000, steps: int = 4000, e: float = E_SIM, tau: float = TAU, seed: int = 9):
    """Electrones con aceleracion constante -eE/m y colisiones de Poisson (probabilidad dt/tau por paso) que reinician la velocidad termica.

    El modelo de Drude es lineal en E, asi que la movilidad mu = v_d / E medida con un campo grande es la misma que con uno pequeno,
    pero con una relacion senal/ruido manejable. Devuelve (mu, error estadistico, serie temporal de <v>/E).
    """
    rng = np.random.default_rng(seed)
    dt = tau / 50
    v_th = np.sqrt(K_B * T_K / M_E)
    v = rng.normal(0.0, v_th, n_e)
    a = -E_CH * e / M_E
    mean = np.empty(steps)
    for k in range(steps):
        v += a * dt
        hit = rng.uniform(size=n_e) < dt / tau
        v[hit] = rng.normal(0.0, v_th, hit.sum())
        mean[k] = v.mean()
    tail = -mean[steps // 2:] / e
    # error: ruido de la media por paso / sqrt(muestras independientes ~ n_e * pasos / 50)
    sigma = v_th / np.sqrt(n_e * (steps // 2) / 50) / e
    return float(tail.mean()), float(sigma), -mean / e


def drift_speed_analytic(e: float = E_FIELD, tau: float = TAU) -> float:
    return E_CH * e * tau / M_E


def conductivity_analytic(n: float = N_CU, tau: float = TAU) -> float:
    return n * E_CH**2 * tau / M_E


# --------------------------------------------------------- resistencia de un cable cónico
RHO = 1.68e-8
LENGTH = 2.0
R1, R2 = 0.5e-3, 1.5e-3


def tapered_resistance_analytic(rho: float = RHO, length: float = LENGTH, r1: float = R1, r2: float = R2) -> float:
    return float(rho * length / (np.pi * r1 * r2))


def tapered_resistance_numeric(n: int = 400, rho: float = RHO, length: float = LENGTH, r1: float = R1, r2: float = R2) -> float:
    x = np.linspace(0, length, n + 1)
    r = r1 + (r2 - r1) * x / length
    return simpson(rho / (np.pi * r**2), x)


def uniform_resistance(rho: float = RHO, length: float = LENGTH, r: float = R1) -> float:
    return float(rho * length / (np.pi * r**2))


ALPHA_T = 0.0039


def resistance_at_temperature(r0: float, delta_t: float, alpha: float = ALPHA_T) -> float:
    return r0 * (1 + alpha * delta_t)
