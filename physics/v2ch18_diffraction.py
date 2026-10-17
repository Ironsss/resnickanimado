"""Volumen 2, capitulo 18. Difraccion y polarizacion."""
from __future__ import annotations

import numpy as np

from .ch07_work_energy import simpson

LAM, A_SLIT, D_GRAT, N_SLITS = 550e-9, 2.0e-6, 6.0e-6, 5


def single_slit_analytic(sin_t, lam: float = LAM, a: float = A_SLIT):
    beta = np.pi * a * np.asarray(sin_t, dtype=float) / lam
    return np.sinc(beta / np.pi) ** 2


def single_slit_numeric(sin_t, lam: float = LAM, a: float = A_SLIT, n: int = 400):
    """Principio de Huygens: se suman 'n' emisores uniformes a lo ancho de la rendija con Simpson (n par)."""
    x = np.linspace(-a / 2, a / 2, n + 1)
    k = 2 * np.pi / lam
    out = []
    for s in np.atleast_1d(sin_t):
        amp = simpson(np.exp(1j * k * x * s).real, x) / a
        out.append(amp**2)
    return np.array(out)


def grating_analytic(sin_t, lam: float = LAM, a: float = A_SLIT, d: float = D_GRAT, n: int = N_SLITS):
    s = np.asarray(sin_t, dtype=float)
    gamma = np.pi * d * s / lam
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(np.abs(np.sin(gamma)) < 1e-12, 1.0, np.sin(n * gamma) / (n * np.sin(gamma)))
    return single_slit_analytic(s, lam, a) * ratio**2


def grating_numeric(sin_t, lam: float = LAM, a: float = A_SLIT, d: float = D_GRAT, n: int = N_SLITS, m: int = 200):
    """Suma coherente de n rendijas, cada una con 'm' fuentes puntuales (regla del punto medio)."""
    k = 2 * np.pi / lam
    centers = (np.arange(n) - (n - 1) / 2) * d
    xs = (np.arange(m) + 0.5) / m * a - a / 2
    pos = (centers[:, None] + xs[None, :]).ravel()
    out = []
    for s in np.atleast_1d(sin_t):
        amp = np.exp(1j * k * pos * s).sum() / (n * m)
        out.append(abs(amp) ** 2)
    return np.array(out)


def principal_maxima_analytic(lam: float = LAM, d: float = D_GRAT):
    """Angulos de los maximos principales d sen(theta) = m lambda."""
    ms = np.arange(0, int(d / lam) + 1)
    return ms, np.degrees(np.arcsin(ms * lam / d))


def principal_maxima_numeric(lam: float = LAM, d: float = D_GRAT, n: int = N_SLITS):
    """Se buscan numericamente los maximos locales de la intensidad (fraccion de rendijas puntuales)."""
    th = np.linspace(0, np.deg2rad(89.0), 20001)
    i = grating_numeric(np.sin(th), lam, 1e-9, d, n, 1)
    idx = [j for j in range(1, len(th) - 1) if i[j] > i[j - 1] and i[j] >= i[j + 1] and i[j] > 0.9]
    idx = [0] + idx
    return np.degrees(th[idx])


# ---------------------------------------------------------------------------- polarizacion
def malus_analytic(theta, i0: float = 1.0):
    return i0 * np.cos(np.asarray(theta, dtype=float)) ** 2


def polarizer_jones(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c * c, c * s], [c * s, s * s]])


def jones_intensity(angles: list[float], e_in=(1.0, 0.0)) -> float:
    e = np.array(e_in, dtype=complex)
    for a in angles:
        e = polarizer_jones(a) @ e
    return float(np.vdot(e, e).real)


def three_polarizers_analytic(a_mid: float, a_out: float = np.pi / 2) -> float:
    return float(np.cos(a_mid) ** 2 * np.cos(a_out - a_mid) ** 2)
