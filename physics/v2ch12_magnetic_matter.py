"""Volumen 2, capitulo 12. Propiedades magneticas de la materia."""
from __future__ import annotations

import numpy as np

from .constants import K_B
from .integrators import integrate, rk4_step
from .ch18_oscillations import agm

MU = 0.1  # A m^2
B_FIELD = 5e-5  # T
INERTIA = 1e-6  # kg m^2


def omega_small(mu: float = MU, b: float = B_FIELD, inertia: float = INERTIA) -> float:
    return float(np.sqrt(mu * b / inertia))


def torque(theta, mu: float = MU, b: float = B_FIELD):
    return mu * b * np.sin(np.asarray(theta, dtype=float))


def potential_energy(theta, mu: float = MU, b: float = B_FIELD):
    return -mu * b * np.cos(np.asarray(theta, dtype=float))


def period_exact(theta0_deg: float) -> float:
    return float(2 * np.pi / omega_small() / agm(1.0, np.cos(np.deg2rad(theta0_deg) / 2)))


def period_numeric(theta0_deg: float) -> float:
    w2 = omega_small() ** 2
    rhs = lambda _t, y: np.array([y[1], -w2 * np.sin(y[0])])  # noqa: E731
    t, y = integrate(rhs, np.array([np.deg2rad(theta0_deg), 0.0]), 0.0, 6.0, 40000, rk4_step)
    th = y[:, 0]
    idx = [i for i in range(1, len(th)) if th[i - 1] > 0 >= th[i]]
    # primer cruce descendente es T/4; el siguiente descendente esta a T
    def cross(i):
        return t[i] - th[i] * (t[i] - t[i - 1]) / (th[i] - th[i - 1])
    t_desc = [cross(i) for i in idx]
    return float(t_desc[1] - t_desc[0]) if len(t_desc) >= 2 else float("nan")


def energy_conservation_drift(theta0_deg: float = 60.0) -> float:
    w2 = omega_small() ** 2
    rhs = lambda _t, y: np.array([y[1], -w2 * np.sin(y[0])])  # noqa: E731
    t, y = integrate(rhs, np.array([np.deg2rad(theta0_deg), 0.0]), 0.0, 3.0, 6000, rk4_step)
    e = 0.5 * INERTIA * y[:, 1] ** 2 + potential_energy(y[:, 0])
    return float(np.max(np.abs(e - e[0])) / abs(e[0]))


# ----------------------------------------------------------------- paramagneto de espin 1/2
MU_B = 9.2740100783e-24


def magnetization_analytic(x):
    return np.tanh(np.asarray(x, dtype=float))


def x_param(b: float, temp: float, mu: float = MU_B) -> float:
    return mu * b / (K_B * temp)


def magnetization_metropolis(x: float, n: int = 100_000, sweeps: int = 80, seed: int = 4) -> tuple[float, float]:
    """Metropolis para espines independientes: energia -/+ x en unidades de kT. Devuelve <sigma> y su error estadistico."""
    rng = np.random.default_rng(seed)
    s = rng.choice([-1, 1], size=n)
    vals = []
    for sweep in range(sweeps):
        de = 2 * x * s  # cambio de energia (en kT) al invertir un espin: E = -x s
        accept = (de <= 0) | (rng.uniform(size=n) < np.exp(-np.clip(de, 0, 50)))
        s = np.where(accept, -s, s)
        if sweep >= sweeps // 2:
            vals.append(s.mean())
    vals = np.array(vals)
    return float(vals.mean()), float(np.sqrt(1.0 / (n * len(vals))))


def curie_low_field(x: float) -> float:
    return x  # M/Ms ~ x para x << 1 (ley de Curie)
