"""Capitulo 5. Fuerza y leyes de Newton."""
from __future__ import annotations

import numpy as np

from .integrators import integrate, rk4_step

G = 9.81

# ---------------------------------------------------- equilibrio estatico
WEIGHT = 500.0  # N
ALPHA_DEG = 30.0  # cable izquierdo respecto a la horizontal
BETA_DEG = 45.0  # cable derecho respecto a la horizontal


def tensions_analytic(w: float = WEIGHT, alpha_deg: float = ALPHA_DEG, beta_deg: float = BETA_DEG):
    a, b = np.deg2rad(alpha_deg), np.deg2rad(beta_deg)
    s = np.sin(a + b)
    return float(w * np.cos(b) / s), float(w * np.cos(a) / s)


def tensions_numeric(w: float = WEIGHT, alpha_deg: float = ALPHA_DEG, beta_deg: float = BETA_DEG):
    """Resuelve la suma de fuerzas nula como sistema lineal 2x2."""
    a, b = np.deg2rad(alpha_deg), np.deg2rad(beta_deg)
    m = np.array([[np.cos(a), -np.cos(b)], [np.sin(a), np.sin(b)]])
    t = np.linalg.solve(m, np.array([0.0, w]))
    return float(t[0]), float(t[1])


def net_force(t1: float, t2: float, w: float = WEIGHT, alpha_deg: float = ALPHA_DEG, beta_deg: float = BETA_DEG):
    a, b = np.deg2rad(alpha_deg), np.deg2rad(beta_deg)
    return np.array([-t1 * np.cos(a) + t2 * np.cos(b), t1 * np.sin(a) + t2 * np.sin(b) - w])


def tension_vs_angle(alphas_deg, beta_deg: float = BETA_DEG, w: float = WEIGHT):
    return np.array([tensions_numeric(w, a, beta_deg) for a in alphas_deg])


# --------------------------------------------- plano inclinado y polea ideal
M1 = 4.0  # kg sobre el plano sin friccion
M2 = 3.0  # kg colgante
THETA_DEG = 30.0


def pulley_analytic(m1: float = M1, m2: float = M2, theta_deg: float = THETA_DEG, g: float = G):
    s = np.sin(np.deg2rad(theta_deg))
    a = (m2 - m1 * s) * g / (m1 + m2)
    t = m1 * (a + g * s)
    return float(a), float(t)


def pulley_numeric(m1: float = M1, m2: float = M2, theta_deg: float = THETA_DEG, g: float = G):
    """Segunda ley para cada cuerpo como sistema lineal en (a, T)."""
    s = np.sin(np.deg2rad(theta_deg))
    # m1 a - T = -m1 g s ;  m2 a + T = m2 g
    m = np.array([[m1, -1.0], [m2, 1.0]])
    rhs = np.array([-m1 * g * s, m2 * g])
    a, t = np.linalg.solve(m, rhs)
    return float(a), float(t)


def pulley_trajectory(t1: float = 3.0, n: int = 300, m1: float = M1, m2: float = M2,
                      theta_deg: float = THETA_DEG, g: float = G):
    a, _ = pulley_numeric(m1, m2, theta_deg, g)

    def rhs(_t, s):
        return np.array([s[1], a])

    return integrate(rhs, np.array([0.0, 0.0]), 0.0, t1, n, rk4_step)


def acceleration_vs_angle(thetas_deg, m1: float = M1, m2: float = M2, g: float = G):
    return np.array([pulley_numeric(m1, m2, th, g)[0] for th in thetas_deg])


def balance_angle_analytic(m1: float = M1, m2: float = M2) -> float:
    return float(np.rad2deg(np.arcsin(m2 / m1)))


def balance_angle_numeric(m1: float = M1, m2: float = M2, g: float = G) -> float:
    """Raiz de a(theta) = 0 por biseccion."""
    lo, hi = 0.0, 89.9
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if pulley_numeric(m1, m2, lo, g)[0] * pulley_numeric(m1, m2, mid, g)[0] <= 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)
