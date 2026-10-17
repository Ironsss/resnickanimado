"""Capitulo 18. Oscilaciones."""
from __future__ import annotations

import numpy as np

from .integrators import integrate, rk4_step
from .ch01_measurement import pendulum_period_numeric

G = 9.81
M, K, AMP = 0.5, 200.0, 0.08

# ------------------------------------------------------------ oscilador armonico simple
def omega0(k: float = K, m: float = M) -> float:
    return float(np.sqrt(k / m))


def x_analytic(t, amp: float = AMP, k: float = K, m: float = M):
    return amp * np.cos(omega0(k, m) * np.asarray(t, dtype=float))


def shm_numeric(t_end: float = 1.0, n: int = 2000, amp: float = AMP, k: float = K, m: float = M):
    return integrate(lambda _t, y: np.array([y[1], -(k / m) * y[0]]), np.array([amp, 0.0]), 0.0, t_end, n, rk4_step)


def period_numeric(amp: float = AMP, k: float = K, m: float = M) -> float:
    t, y = shm_numeric(2.0, 40000, amp, k, m)
    x = y[:, 0]
    # cruces descendentes por cero con interpolacion lineal
    idx = [i for i in range(1, len(x)) if x[i - 1] > 0 >= x[i]]
    t1 = t[idx[0]] - x[idx[0]] * (t[idx[0]] - t[idx[0] - 1]) / (x[idx[0]] - x[idx[0] - 1])
    t2 = t[idx[1]] - x[idx[1]] * (t[idx[1]] - t[idx[1] - 1]) / (x[idx[1]] - x[idx[1] - 1])
    return float(t2 - t1)


def energy_analytic(amp: float = AMP, k: float = K) -> float:
    return 0.5 * k * amp**2


# ------------------------------------------------------------------ pendulo (no lineal)
def agm(a: float, b: float, tol: float = 1e-15) -> float:
    while abs(a - b) > tol:
        a, b = 0.5 * (a + b), np.sqrt(a * b)
    return float(a)


def pendulum_period_exact(theta0_deg: float, length: float = 1.0, g: float = G) -> float:
    """Periodo exacto T = 2 pi sqrt(L/g) / AGM(1, cos(theta0/2)) (media aritmetico-geometrica)."""
    return float(2 * np.pi * np.sqrt(length / g) / agm(1.0, np.cos(np.deg2rad(theta0_deg) / 2)))


def pendulum_period_small(length: float = 1.0, g: float = G) -> float:
    return float(2 * np.pi * np.sqrt(length / g))


def pendulum_table(angles=(10, 30, 60, 90)):
    return [{"theta": a, "exact": pendulum_period_exact(a), "numeric": pendulum_period_numeric(np.deg2rad(a), 1.0, G, 20000),
             "small": pendulum_period_small()} for a in angles]


# --------------------------------------------------------- oscilador amortiguado y forzado
M_D, K_D, B_D, F0 = 1.0, 100.0, 2.0, 5.0  # w0 = 10 rad/s, gamma = 2 1/s


def driven_amplitude_analytic(w, m: float = M_D, k: float = K_D, b: float = B_D, f0: float = F0):
    w = np.asarray(w, dtype=float)
    return (f0 / m) / np.sqrt((k / m - w**2) ** 2 + (b / m * w) ** 2)


def resonance_frequency_analytic(m: float = M_D, k: float = K_D, b: float = B_D) -> float:
    return float(np.sqrt(k / m - (b / m) ** 2 / 2))


def driven_amplitude_numeric(w: float, m: float = M_D, k: float = K_D, b: float = B_D, f0: float = F0) -> float:
    """Simula hasta el regimen estacionario y mide el maximo de |x| en los ultimos ciclos."""
    t_end = 24.0
    n = int(t_end * 400 / min(1.0, 2 * np.pi / w) / 2 * 1.0) + 4000
    n = max(n, 20000)
    f = lambda t, y: np.array([y[1], (f0 * np.cos(w * t) - b * y[1] - k * y[0]) / m])  # noqa: E731
    t, y = integrate(f, np.zeros(2), 0.0, t_end, n, rk4_step)
    tail = t > t_end - max(3 * 2 * np.pi / w, 4.0)
    return float(np.max(np.abs(y[tail, 0])))


def resonance_curve_numeric(ws):
    return np.array([driven_amplitude_numeric(float(w)) for w in ws])


def decay_envelope_analytic(t, amp: float = 1.0, b: float = B_D, m: float = M_D):
    return amp * np.exp(-b * np.asarray(t, dtype=float) / (2 * m))


def damped_free(t_end: float = 6.0, n: int = 6000, m: float = M_D, k: float = K_D, b: float = B_D):
    return integrate(lambda _t, y: np.array([y[1], (-k * y[0] - b * y[1]) / m]), np.array([1.0, 0.0]), 0.0, t_end, n, rk4_step)
