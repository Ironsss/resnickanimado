"""Volumen 2, capitulo 24. Fisica nuclear."""
from __future__ import annotations

import numpy as np

from .integrators import integrate, rk4_step

# ------------------------------------------------------------------------ cadena de decaimiento
LAM_A, LAM_B = 1.0, 0.25  # 1/s (escala adimensional)
N0 = 1.0


def bateman_b(t):
    t = np.asarray(t, dtype=float)
    return N0 * LAM_A / (LAM_B - LAM_A) * (np.exp(-LAM_A * t) - np.exp(-LAM_B * t))


def n_a(t):
    return N0 * np.exp(-LAM_A * np.asarray(t, dtype=float))


def chain_numeric(t_end: float = 20.0, n: int = 4000):
    rhs = lambda _t, y: np.array([-LAM_A * y[0], LAM_A * y[0] - LAM_B * y[1], LAM_B * y[1]])  # noqa: E731
    return integrate(rhs, np.array([N0, 0.0, 0.0]), 0.0, t_end, n, rk4_step)


def peak_time_analytic() -> float:
    return float(np.log(LAM_A / LAM_B) / (LAM_A - LAM_B))


def peak_time_numeric() -> float:
    t, y = chain_numeric(20.0, 40000)
    k = int(np.argmax(y[:, 1]))
    y0, y1, y2 = y[k - 1, 1], y[k, 1], y[k + 1, 1]
    return float(t[k] + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2) * (t[1] - t[0]))


def mean_life_monte_carlo(lam: float = LAM_B, n: int = 50_000, seed: int = 11):
    """Tiempos de decaimiento exponenciales (inversion de la CDF); la vida media es el promedio y su error sigma/sqrt(N)."""
    rng = np.random.default_rng(seed)
    t = -np.log(1 - rng.uniform(size=n)) / lam
    return float(t.mean()), float(t.std(ddof=1) / np.sqrt(n))


# ------------------------------------------------------------------------ formula semiempirica
A_V, A_S, A_C, A_A, A_P = 15.75, 17.8, 0.711, 23.7, 11.18  # MeV


def binding_energy(a: int, z: float, pairing: bool = True) -> float:
    n = a - z
    pair = 0.0
    if not pairing:
        pass
    elif int(round(z)) % 2 == 0 and int(round(n)) % 2 == 0:
        pair = A_P / a**0.5
    elif int(round(z)) % 2 == 1 and int(round(n)) % 2 == 1:
        pair = -A_P / a**0.5
    return A_V * a - A_S * a ** (2 / 3) - A_C * z * (z - 1) / a ** (1 / 3) - A_A * (a - 2 * z) ** 2 / a + pair


def z_stable_analytic(a: int) -> float:
    """Minimo continuo de la parte regular de B: dB/dZ = 0 (se desprecia la diferencia n-p de masa)."""
    return a / 2 / (1 + A_C * a ** (2 / 3) / (4 * A_A))


def z_stable_numeric(a: int, pairing: bool = True) -> int:
    zs = np.arange(1, a)
    b = np.array([binding_energy(a, int(z), pairing) for z in zs])
    return int(zs[int(np.argmax(b))])


def b_per_a(a: int, pairing: bool = True) -> float:
    return binding_energy(a, z_stable_numeric(a, pairing), pairing) / a


def b_per_a_analytic(a) -> np.ndarray:
    """B/A evaluada en el Z continuo que minimiza la masa (sin termino de apareamiento)."""
    a = np.asarray(a, dtype=float)
    z = a / 2 / (1 + A_C * a ** (2 / 3) / (4 * A_A))
    return (A_V * a - A_S * a ** (2 / 3) - A_C * z * (z - 1) / a ** (1 / 3) - A_A * (a - 2 * z) ** 2 / a) / a


def peak_a_numeric() -> int:
    aa = np.arange(20, 240)
    v = np.array([b_per_a(int(a), False) for a in aa])
    return int(aa[int(np.argmax(v))])
