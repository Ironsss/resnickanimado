"""Capitulo 6. Dinamica de la particula: friccion y resistencia del medio."""
from __future__ import annotations

import numpy as np

from .integrators import euler_step, integrate, rk4_step

G = 9.81

# ----------------------------------------------------- friccion en plano inclinado
THETA_DEG = 35.0
MU_S = 0.45
MU_K = 0.30


def critical_angle_analytic(mu_s: float = MU_S) -> float:
    return float(np.rad2deg(np.arctan(mu_s)))


def incline_acceleration_analytic(theta_deg: float, mu_s: float = MU_S, mu_k: float = MU_K, g: float = G) -> float:
    """Aceleracion a lo largo del plano (positiva hacia abajo), partiendo del reposo."""
    th = np.deg2rad(theta_deg)
    if np.tan(th) <= mu_s:
        return 0.0
    return float(g * (np.sin(th) - mu_k * np.cos(th)))


def simulate_incline(theta_deg: float, t_end: float, mu_s: float = MU_S, mu_k: float = MU_K,
                     g: float = G, dt: float = 1e-3):
    """Integra el bloque con friccion estatica y cinetica (maquina de estados).

    Mientras v = 0 la friccion estatica compensa la componente del peso si no excede mu_s N;
    en caso contrario el bloque desliza con friccion cinetica. Devuelve (t, s, v).
    """
    th = np.deg2rad(theta_deg)
    n = int(round(t_end / dt))
    t = np.linspace(0.0, t_end, n + 1)
    s = np.zeros(n + 1)
    v = np.zeros(n + 1)
    moving = False
    for i in range(n):
        if not moving:
            if g * np.sin(th) > mu_s * g * np.cos(th):
                moving = True
            else:
                s[i + 1], v[i + 1] = s[i], 0.0
                continue
        y = rk4_step(lambda _t, y_: np.array([y_[1], g * (np.sin(th) - mu_k * np.cos(th))]), t[i],
                     np.array([s[i], v[i]]), dt)
        s[i + 1], v[i + 1] = y
    return t, s, v


def critical_angle_numeric(mu_s: float = MU_S, mu_k: float = MU_K, g: float = G) -> float:
    """Biseccion sobre el angulo: el menor angulo con desplazamiento apreciable en 1 s."""
    lo, hi = 1.0, 80.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        _, s, _ = simulate_incline(mid, 1.0, mu_s, mu_k, g, dt=0.01)
        if s[-1] > 1e-9:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def incline_sweep(angles_deg, mu_s: float = MU_S, mu_k: float = MU_K):
    return np.array([incline_acceleration_analytic(a, mu_s, mu_k) for a in angles_deg])


# ------------------------------------------------------- resistencia lineal (Stokes)
MASS = 0.5
DRAG_B = 0.8  # kg/s


def terminal_speed(m: float = MASS, b: float = DRAG_B, g: float = G) -> float:
    return m * g / b


def tau(m: float = MASS, b: float = DRAG_B) -> float:
    return m / b


def drag_velocity_analytic(t, m: float = MASS, b: float = DRAG_B, g: float = G):
    t = np.asarray(t, dtype=float)
    return terminal_speed(m, b, g) * (1.0 - np.exp(-t / tau(m, b)))


def drag_position_analytic(t, m: float = MASS, b: float = DRAG_B, g: float = G):
    t = np.asarray(t, dtype=float)
    ta = tau(m, b)
    return terminal_speed(m, b, g) * (t - ta * (1.0 - np.exp(-t / ta)))


def drag_numeric(n: int, method: str = "rk4", t_end: float = 4.0, m: float = MASS, b: float = DRAG_B, g: float = G):
    step = rk4_step if method == "rk4" else euler_step
    return integrate(lambda _t, s: np.array([s[1], g - (b / m) * s[1]]), np.array([0.0, 0.0]), 0.0, t_end, n, step)


def time_to_fraction_analytic(frac: float = 0.95, m: float = MASS, b: float = DRAG_B) -> float:
    return float(-tau(m, b) * np.log(1.0 - frac))


def time_to_fraction_numeric(frac: float = 0.95, m: float = MASS, b: float = DRAG_B, g: float = G) -> float:
    t, y = drag_numeric(4000, "rk4", 6.0, m, b, g)
    target = frac * terminal_speed(m, b, g)
    i = int(np.argmax(y[:, 1] >= target))
    t0, t1, v0, v1 = t[i - 1], t[i], y[i - 1, 1], y[i, 1]
    return float(t0 + (target - v0) * (t1 - t0) / (v1 - v0))


def drag_convergence(t_end: float = 1.0, ns=(8, 16, 32, 64, 128)):
    ex = float(drag_velocity_analytic(t_end))
    rows = []
    for n in ns:
        _, ye = drag_numeric(n, "euler", t_end)
        _, yr = drag_numeric(n, "rk4", t_end)
        rows.append({"n": n, "err_euler": abs(ye[-1, 1] - ex), "err_rk4": abs(yr[-1, 1] - ex)})
    return rows
