"""Capitulo 21. Teoria especial de la relatividad."""
from __future__ import annotations

import numpy as np

from .ch07_work_energy import simpson

C = 299_792_458.0
TAU_MUON = 2.197e-6
BETA_MUON = 0.995
D_MUON = 10_000.0


def gamma(beta: float) -> float:
    return float(1.0 / np.sqrt(1.0 - beta**2))


def survival_analytic(d: float = D_MUON, beta: float = BETA_MUON, tau: float = TAU_MUON, relativistic: bool = True) -> float:
    v = beta * C
    life = gamma(beta) * tau if relativistic else tau
    return float(np.exp(-d / (v * life)))


def survival_monte_carlo(n: int = 2_000_000, d: float = D_MUON, beta: float = BETA_MUON, tau: float = TAU_MUON,
                         relativistic: bool = True, seed: int = 3):
    rng = np.random.default_rng(seed)
    life = (gamma(beta) if relativistic else 1.0) * tau
    t_decay = rng.exponential(life, n)
    flight = d / (beta * C)
    p = float(np.mean(t_decay > flight))
    return p, float(np.sqrt(p * (1 - p) / n))


# --------------------------------------------------------- adicion de velocidades
def velocity_addition_analytic(u: float, v: float) -> float:
    return (u + v) / (1 + u * v)  # en unidades de c


def velocity_addition_numeric(u: float, v: float) -> float:
    """Componiendo boosts como matrices de Lorentz en (ct, x) y leyendo la velocidad del resultado."""
    def boost(beta):
        g = gamma(beta)
        return np.array([[g, g * beta], [g * beta, g]])
    m = boost(v) @ boost(u)
    return float(m[1, 0] / m[0, 0])


def rapidity_addition(u: float, v: float) -> float:
    return float(np.tanh(np.arctanh(u) + np.arctanh(v)))


# ------------------------------------------------------- energia cinetica relativista
def kinetic_analytic(beta: float, m: float = 1.0) -> float:
    return (gamma(beta) - 1.0) * m * C**2


def kinetic_numeric(beta: float, m: float = 1.0, n: int = 4000) -> float:
    """K = integral de v dp con p = gamma m v (trabajo de una fuerza que acelera desde el reposo)."""
    b = np.linspace(0.0, beta, n + 1)
    g = 1.0 / np.sqrt(1 - b**2)
    p = g * m * b * C
    v = b * C
    # dp/db analitica del integrando evita derivar numericamente: dp = m c gamma^3 db
    dp_db = m * C * g**3
    return simpson(v * dp_db, b)


def kinetic_classical(beta: float, m: float = 1.0) -> float:
    return 0.5 * m * (beta * C) ** 2
