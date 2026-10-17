"""Volumen 2, capitulo 11. Inductancia."""
from __future__ import annotations

import numpy as np

from .integrators import integrate, rk4_step
from .ch07_work_energy import simpson

EPS, R, L = 12.0, 4.0, 0.2


def tau_rl() -> float:
    return L / R


def i_growth_analytic(t, eps: float = EPS, r: float = R, l_: float = L):
    return (eps / r) * (1 - np.exp(-r * np.asarray(t, dtype=float) / l_))


def rl_numeric(t_end: float = 0.4, n: int = 2000):
    return integrate(lambda _t, y: np.array([(EPS - R * y[0]) / L]), np.array([0.0]), 0.0, t_end, n, rk4_step)


def time_to_half_numeric() -> float:
    """Tiempo de decaimiento al 50 % tras retirar la fuente (RK4 de L di/dt = -R i)."""
    t, y = integrate(lambda _t, i: np.array([-R * i[0] / L]), np.array([EPS / R]), 0.0, 0.3, 3000, rk4_step)
    target = 0.5 * EPS / R
    k = int(np.argmax(y[:, 0] <= target))
    return float(t[k - 1] + (y[k - 1, 0] - target) * (t[k] - t[k - 1]) / (y[k - 1, 0] - y[k, 0]))


def energy_magnetic(current: float = EPS / R, l_: float = L) -> float:
    return 0.5 * l_ * current**2


def energy_integral_numeric(t_end: float = 0.5, n: int = 5000) -> float:
    """Energia entregada al inductor: integral de L i di/dt dt = integral de i v_L dt, con v_L = eps - R i."""
    t, y = rl_numeric(t_end, n)
    i = y[:, 0]
    v_l = EPS - R * i
    return simpson(i * v_l, t)


# ----------------------------------------------------------------------- RLC serie
L2, C2, R2 = 0.1, 100e-6, 5.0
Q0 = 1e-3


def omega0() -> float:
    return float(1 / np.sqrt(L2 * C2))


def gamma_() -> float:
    return R2 / (2 * L2)


def omega_d() -> float:
    return float(np.sqrt(omega0() ** 2 - gamma_() ** 2))


def q_analytic(t):
    t = np.asarray(t, dtype=float)
    g, wd = gamma_(), omega_d()
    return Q0 * np.exp(-g * t) * (np.cos(wd * t) + g / wd * np.sin(wd * t))


def rlc_numeric(t_end: float = 0.15, n: int = 6000, r: float = R2):
    rhs = lambda _t, y: np.array([y[1], -(r / L2) * y[1] - y[0] / (L2 * C2)])  # noqa: E731
    return integrate(rhs, np.array([Q0, 0.0]), 0.0, t_end, n, rk4_step)


def period_numeric() -> float:
    t, y = rlc_numeric(0.2, 20000)
    q = y[:, 0]
    idx = [i for i in range(1, len(q)) if q[i - 1] > 0 >= q[i]]
    def cross(i):
        return t[i] - q[i] * (t[i] - t[i - 1]) / (q[i] - q[i - 1])
    return float(cross(idx[1]) - cross(idx[0]))


def quality_factor() -> float:
    return omega0() * L2 / R2


def energy_ratio_after_period(r: float = R2) -> float:
    """U(T)/U(0) por integracion; analiticamente exp(-2 gamma T) (aprox. de la envolvente)."""
    t, y = rlc_numeric(2 * np.pi / omega_d() + 1e-9, 8000, r)
    q, i = y[-1]
    u = q**2 / (2 * C2) + 0.5 * L2 * i**2
    u0 = Q0**2 / (2 * C2)
    return float(u / u0)
