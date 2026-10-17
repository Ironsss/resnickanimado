"""Volumen 2, capitulo 19. Fotones y naturaleza ondulatoria de la luz."""
from __future__ import annotations

import numpy as np

from .constants import C_LIGHT, E_CH, H_PLANCK, K_B, M_E, SIGMA_SB
from .ch07_work_energy import simpson

# ----------------------------------------------------------------------------- cuerpo negro
WIEN_B = 2.897771955e-3


def planck(lam, temp: float):
    lam = np.asarray(lam, dtype=float)
    return 2 * np.pi * H_PLANCK * C_LIGHT**2 / lam**5 / np.expm1(H_PLANCK * C_LIGHT / (lam * K_B * temp))


def wien_peak_analytic(temp: float) -> float:
    return WIEN_B / temp


def wien_peak_numeric(temp: float) -> float:
    """Maximo de la funcion de Planck por busqueda de la seccion aurea."""
    lo, hi = 0.2 * WIEN_B / temp, 5 * WIEN_B / temp
    g = (np.sqrt(5) - 1) / 2
    c, e = hi - g * (hi - lo), lo + g * (hi - lo)
    for _ in range(200):
        if planck(c, temp) > planck(e, temp):
            hi = e
        else:
            lo = c
        c, e = hi - g * (hi - lo), lo + g * (hi - lo)
    return float(0.5 * (lo + hi))


def stefan_analytic(temp: float) -> float:
    return SIGMA_SB * temp**4


def stefan_numeric(temp: float, n: int = 20000) -> float:
    """Integral de la distribucion de Planck con Simpson en lambda (de 0.02 a 200 veces lambda_max)."""
    lam_max = WIEN_B / temp
    lam = np.linspace(0.02 * lam_max, 200 * lam_max, n + 1)
    return simpson(planck(lam, temp), lam)


# ------------------------------------------------------------------------------ fotoelectrico
PHI_EV = 2.30  # sodio
F_THRESH = PHI_EV * E_CH / H_PLANCK


def stopping_potential_analytic(f):
    return (H_PLANCK * np.asarray(f, dtype=float) - PHI_EV * E_CH) / E_CH


def millikan_fit(seed: int = 3, noise: float = 0.01):
    """Datos sinteticos con ruido gaussiano (10 mV); ajuste lineal V0 = (h/e) f - phi/e. Devuelve h, phi[eV] y sus errores."""
    rng = np.random.default_rng(seed)
    f = np.linspace(6.0e14, 12.0e14, 25)
    v = stopping_potential_analytic(f) + rng.normal(0, noise, f.size)
    f14 = f / 1e14  # variable escalada para el condicionamiento del ajuste
    a = np.vstack([f14, np.ones_like(f14)]).T
    coef, *_ = np.linalg.lstsq(a, v, rcond=None)
    resid = v - a @ coef
    cov = np.linalg.inv(a.T @ a) * resid.var(ddof=2)
    h = coef[0] * E_CH / 1e14
    phi = -coef[1]
    return float(h), float(phi), float(np.sqrt(cov[0, 0]) * E_CH / 1e14), float(np.sqrt(cov[1, 1]))


# --------------------------------------------------------------------------------- Compton
LAMBDA_C = H_PLANCK / (M_E * C_LIGHT)


def compton_shift_analytic(theta):
    return LAMBDA_C * (1 - np.cos(np.asarray(theta, dtype=float)))


def compton_shift_numeric(theta: float, lam: float = 71.0e-12) -> float:
    """Conservacion de energia y momento: se busca por biseccion la energia del foton dispersado."""
    e0 = H_PLANCK * C_LIGHT / lam
    mc2 = M_E * C_LIGHT**2

    def resid(e1):
        p0 = e0 / C_LIGHT
        p1 = e1 / C_LIGHT
        pe2 = p0**2 + p1**2 - 2 * p0 * p1 * np.cos(theta)
        # energia del electron: E_e^2 = (pe c)^2 + mc2^2 ; conservacion: e0 + mc2 = e1 + E_e
        return (e0 + mc2 - e1) ** 2 - (pe2 * C_LIGHT**2 + mc2**2)

    lo, hi = e0 * 1e-6, e0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if resid(lo) * resid(mid) <= 0:
            hi = mid
        else:
            lo = mid
    e1 = 0.5 * (lo + hi)
    return float(H_PLANCK * C_LIGHT / e1 - lam)
