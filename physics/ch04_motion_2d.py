"""Capitulo 4. Movimiento en dos y tres dimensiones."""
from __future__ import annotations

import numpy as np

from .integrators import integrate, rk4_step

G = 9.81
V0 = 25.0
THETA_DEG = 40.0


def projectile_analytic(v0: float = V0, theta_deg: float = THETA_DEG, g: float = G) -> dict[str, float]:
    th = np.deg2rad(theta_deg)
    return {
        "t_flight": 2 * v0 * np.sin(th) / g,
        "range": v0**2 * np.sin(2 * th) / g,
        "height": (v0 * np.sin(th)) ** 2 / (2 * g),
        "t_apex": v0 * np.sin(th) / g,
    }


def trajectory_analytic(t, v0: float = V0, theta_deg: float = THETA_DEG, g: float = G):
    th = np.deg2rad(theta_deg)
    t = np.asarray(t, dtype=float)
    return v0 * np.cos(th) * t, v0 * np.sin(th) * t - 0.5 * g * t**2


def projectile_numeric(v0: float = V0, theta_deg: float = THETA_DEG, g: float = G, dt: float = 0.01):
    """Integracion RK4 hasta que y cruza 0; el cruce se refina por interpolacion."""
    th = np.deg2rad(theta_deg)

    def rhs(_t, s):
        return np.array([s[2], s[3], 0.0, -g])

    s = np.array([0.0, 0.0, v0 * np.cos(th), v0 * np.sin(th)])
    ts, states = [0.0], [s.copy()]
    t = 0.0
    while True:
        s_new = rk4_step(rhs, t, s, dt)
        if s_new[1] < 0.0 and t > 0.0:
            frac = s[1] / (s[1] - s_new[1])
            s_hit = s + frac * (s_new - s)
            ts.append(t + frac * dt)
            states.append(s_hit)
            break
        t += dt
        s = s_new
        ts.append(t)
        states.append(s.copy())
    return np.array(ts), np.array(states)


def range_numeric(theta_deg: float, v0: float = V0, g: float = G, dt: float = 0.002) -> float:
    return float(projectile_numeric(v0, theta_deg, g, dt)[1][-1, 0])


def optimal_angle_numeric(v0: float = V0, g: float = G) -> float:
    """Maximiza el alcance con busqueda de seccion aurea sobre el angulo."""
    phi = (np.sqrt(5) - 1) / 2
    a, b = 5.0, 85.0
    c, d = b - phi * (b - a), a + phi * (b - a)
    fc, fd = range_numeric(c, v0, g), range_numeric(d, v0, g)
    for _ in range(40):
        if fc > fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = range_numeric(c, v0, g)
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = range_numeric(d, v0, g)
    return float((a + b) / 2)


# ------------------------------------------------------ movimiento circular
RADIUS = 2.0
PERIOD = 4.0


def omega(period: float = PERIOD) -> float:
    return 2 * np.pi / period


def position(t, radius: float = RADIUS, period: float = PERIOD):
    w = omega(period)
    t = np.asarray(t, dtype=float)
    return radius * np.array([np.cos(w * t), np.sin(w * t)])


def circular_analytic(radius: float = RADIUS, period: float = PERIOD) -> dict[str, float]:
    w = omega(period)
    return {"omega": w, "speed": w * radius, "a_c": w**2 * radius, "a_c_from_v": (w * radius) ** 2 / radius}


def velocity_fd(t: float, h: float, radius: float = RADIUS, period: float = PERIOD) -> np.ndarray:
    return (position(t + h, radius, period) - position(t - h, radius, period)) / (2 * h)


def acceleration_fd(t: float, h: float, radius: float = RADIUS, period: float = PERIOD) -> np.ndarray:
    return (
        position(t + h, radius, period) - 2 * position(t, radius, period) + position(t - h, radius, period)
    ) / h**2


def fd_error_table(t: float = 0.7, hs=(0.5, 0.25, 0.1, 0.05, 0.01)) -> list[dict[str, float]]:
    ca = circular_analytic()
    rows = []
    for h in hs:
        v = np.linalg.norm(velocity_fd(t, h))
        a = np.linalg.norm(acceleration_fd(t, h))
        rows.append({"h": h, "speed": float(v), "a_c": float(a),
                     "err_speed": abs(float(v) - ca["speed"]), "err_a": abs(float(a) - ca["a_c"])})
    return rows
