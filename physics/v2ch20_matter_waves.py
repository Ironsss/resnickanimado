"""Volumen 2, capitulo 20. Ondas de materia."""
from __future__ import annotations

import numpy as np

from .constants import C_LIGHT, E_CH, H_PLANCK, HBAR, M_E

# ----------------------------------------------------------------- interferencia de electrones
KE_EV, D_SLIT, L_SCR = 100.0, 1.0e-6, 1.0


def de_broglie(ke_ev: float = KE_EV) -> float:
    ke = ke_ev * E_CH
    mc2 = M_E * C_LIGHT**2
    p = np.sqrt(ke * (ke + 2 * mc2)) / C_LIGHT  # relativista
    return float(H_PLANCK / p)


def fringe_spacing_analytic(ke_ev: float = KE_EV, d: float = D_SLIT, big_l: float = L_SCR) -> float:
    return de_broglie(ke_ev) * big_l / d


def electron_hits(n: int = 20000, ke_ev: float = KE_EV, d: float = D_SLIT, big_l: float = L_SCR, seed: int = 7) -> np.ndarray:
    """Monte Carlo: cada electron llega a una posicion y muestreada de I(y) = cos^2(pi d y/(lambda L)) por rechazo."""
    rng = np.random.default_rng(seed)
    lam = de_broglie(ke_ev)
    span = 12 * lam * big_l / d
    out = []
    while len(out) < n:
        y = rng.uniform(-span / 2, span / 2, 4 * n)
        keep = rng.uniform(size=y.size) < np.cos(np.pi * d * y / (lam * big_l)) ** 2
        out.extend(y[keep].tolist())
    return np.array(out[:n])


def fringe_spacing_from_hits(hits: np.ndarray, ke_ev: float = KE_EV, d: float = D_SLIT, big_l: float = L_SCR):
    """Espaciado de franjas por la frecuencia espacial dominante del histograma (FFT con relleno de ceros)."""
    lam = de_broglie(ke_ev)
    span = 12 * lam * big_l / d
    nb = 600
    hist, edges = np.histogram(hits, bins=nb, range=(-span / 2, span / 2))
    hist = hist - hist.mean()
    dy = edges[1] - edges[0]
    spec = np.abs(np.fft.rfft(hist, 16 * nb))
    freqs = np.fft.rfftfreq(16 * nb, dy)
    k = int(np.argmax(spec))
    # refinamiento parabolico
    y0, y1, y2 = spec[k - 1], spec[k], spec[k + 1]
    shift = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
    f = freqs[k] + shift * (freqs[1] - freqs[0])
    return float(1.0 / f), float(len(hits))


def bragg_angle_analytic(ke_ev: float = 54.0, d: float = 0.215e-9) -> float:
    return float(np.degrees(np.arcsin(de_broglie(ke_ev) / (2 * d))))


def bragg_angle_numeric(ke_ev: float = 54.0, d: float = 0.215e-9) -> float:
    """Biseccion sobre el angulo: la diferencia de camino 2 d sen(theta) igual a una longitud de onda."""
    lam = de_broglie(ke_ev)
    lo, hi = 0.0, 90.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if 2 * d * np.sin(np.deg2rad(mid)) < lam:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ------------------------------------------------------------------ incertidumbre y paquetes
SIGMA0 = 1.0e-10


def packet_width_analytic(t, sigma0: float = SIGMA0, m: float = M_E):
    t = np.asarray(t, dtype=float)
    return sigma0 * np.sqrt(1 + (HBAR * t / (2 * m * sigma0**2)) ** 2)


def packet_numeric(t: float, sigma0: float = SIGMA0, m: float = M_E, n: int = 4096, box: float = 400e-10):
    """Evolucion libre exacta en el espacio de momentos con FFT. Devuelve sigma_x, sigma_p."""
    x = np.linspace(-box / 2, box / 2, n, endpoint=False)
    dx = x[1] - x[0]
    psi = np.exp(-(x**2) / (4 * sigma0**2))
    psi /= np.sqrt(np.sum(np.abs(psi) ** 2) * dx)
    k = 2 * np.pi * np.fft.fftfreq(n, dx)
    phi = np.fft.fft(psi)
    phit = phi * np.exp(-1j * HBAR * k**2 * t / (2 * m))
    psit = np.fft.ifft(phit)
    px = np.abs(psit) ** 2
    px /= px.sum() * dx
    sx = float(np.sqrt(np.sum(x**2 * px) * dx - (np.sum(x * px) * dx) ** 2))
    pk = np.abs(phit) ** 2
    pk /= pk.sum()
    sk = float(np.sqrt(np.sum(k**2 * pk) - np.sum(k * pk) ** 2))
    return sx, HBAR * sk
