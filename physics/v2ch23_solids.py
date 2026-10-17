"""Volumen 2, capitulo 23. Conduccion electrica en solidos."""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.special import lambertw

# ------------------------------------------------------------------ Kronig-Penney (peine de deltas)
# unidades: hbar = m = 1, constante de red a = 1; potencial V = g * sum delta(x - n a)
G_STRENGTH = 4.0


def _f(q, g: float = G_STRENGTH):
    return np.cos(q) + g / q * np.sin(q)


def kp_residual(e: float, k: float, g: float = G_STRENGTH) -> float:
    return _f(np.sqrt(2 * e), g) - np.cos(k)  # cos(ka) = cos(qa) + (m g / hbar^2 q) sin(qa), con a = m = hbar = 1


def band_limits_analytic(n_bands: int = 3, g: float = G_STRENGTH):
    """Lista de (E_inferior, E_superior) de cada banda: transiciones de |cos q + (g/q) sin q| <= 1, refinadas por brentq."""
    qs = np.linspace(1e-6, (n_bands + 1) * np.pi, 80001)
    gq = np.abs(_f(qs, g)) - 1.0
    edges = []
    for i in range(len(qs) - 1):
        if (gq[i] <= 0) != (gq[i + 1] <= 0):
            edges.append(brentq(lambda x: abs(_f(x, g)) - 1.0, qs[i], qs[i + 1]))
    edges = np.array(edges) ** 2 / 2
    return [(float(edges[2 * i]), float(edges[2 * i + 1])) for i in range(min(n_bands, len(edges) // 2))]


def band_edges_analytic(n_bands: int = 3, g: float = G_STRENGTH):
    return np.sort(np.array(band_limits_analytic(n_bands, g)).ravel())


def bands_plane_waves(k: float, n_bands: int = 3, n_g: int = 201, g: float = G_STRENGTH) -> np.ndarray:
    """Diagonaliza el hamiltoniano en una base de ondas planas e^{i(k + 2 pi m)x}; V_{mm'} = g (a = 1)."""
    m = np.arange(-(n_g // 2), n_g // 2 + 1)
    h = np.full((len(m), len(m)), g, dtype=float)
    h[np.diag_indices_from(h)] += 0.5 * (k + 2 * np.pi * m) ** 2
    return np.linalg.eigvalsh(h)[:n_bands]


def band_edges_numeric(n_bands: int = 3, n_g: int = 201):
    lo = bands_plane_waves(0.0, n_bands, n_g)
    hi = bands_plane_waves(np.pi, n_bands, n_g)
    return np.sort(np.concatenate([lo, hi]))


def band_energy_analytic(k: float, band: int = 0, g: float = G_STRENGTH) -> float:
    """Energia de la banda 'band' en k, por busqueda de raices de la relacion de dispersion dentro de sus limites."""
    lo, hi = band_limits_analytic(band + 1, g)[band]
    f = lambda e: kp_residual(e, k, g)  # noqa: E731
    return float(brentq(f, lo - 1e-9, hi + 1e-9))


# ------------------------------------------------------------------------------------ diodo
I_S, N_IDEAL, V_T = 1.0e-12, 1.0, 0.025852
R_SERIES = 100.0


def diode_voltage_analytic(e: float, r: float = R_SERIES) -> float:
    """Solucion exacta del diodo de Shockley en serie con R mediante la funcion W de Lambert."""
    nvt = N_IDEAL * V_T
    arg = I_S * r / nvt * np.exp((e + I_S * r) / nvt)
    return float(e + I_S * r - nvt * lambertw(arg).real)


def diode_voltage_numeric(e: float, r: float = R_SERIES) -> float:
    """Biseccion sobre V_d de f(V) = I_S (exp(V/nVT) - 1) - (E - V)/R."""
    nvt = N_IDEAL * V_T
    f = lambda v: I_S * np.expm1(v / nvt) - (e - v) / r  # noqa: E731
    lo, hi = 0.0, e
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def diode_current(e: float, v_d: float, r: float = R_SERIES) -> float:
    return (e - v_d) / r
