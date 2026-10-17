"""Capitulo 15. Gravitacion."""
from __future__ import annotations

import numpy as np

from .integrators import integrate, rk4_step

# ----------------------------------------------------------- orbita de Kepler (UA, anio)
GM = 4 * np.pi**2  # UA^3 / anio^2 (Sol)
A_ORBIT = 2.0
E_ORBIT = 0.5


def period_analytic(a: float = A_ORBIT) -> float:
    return float(a**1.5)


def perihelion_speed(a: float = A_ORBIT, e: float = E_ORBIT) -> float:
    return float(np.sqrt(GM * (1 + e) / (a * (1 - e))))


def orbit_numeric(a: float = A_ORBIT, e: float = E_ORBIT, revolutions: float = 1.2, n_per_orbit: int = 8000):
    rp = a * (1 - e)
    y0 = np.array([rp, 0.0, 0.0, perihelion_speed(a, e)])
    rhs = lambda _t, s: np.array([s[2], s[3], -GM * s[0] / np.hypot(s[0], s[1]) ** 3, -GM * s[1] / np.hypot(s[0], s[1]) ** 3])  # noqa: E731
    t_end = revolutions * period_analytic(a)
    n = int(n_per_orbit * revolutions)
    return integrate(rhs, y0, 0.0, t_end, n, rk4_step)


def period_numeric(a: float = A_ORBIT, e: float = E_ORBIT) -> float:
    """Periodo por el segundo paso del cuerpo por el perihelio (minimo de r), refinado con parabola."""
    t, y = orbit_numeric(a, e, 1.3, 20000)
    r = np.hypot(y[:, 0], y[:, 1])
    idx = [i for i in range(1, len(r) - 1) if r[i] < r[i - 1] and r[i] <= r[i + 1]]
    i = idx[0]
    # ajuste parabolico para el instante del minimo
    y0, y1, y2 = r[i - 1], r[i], r[i + 1]
    dt = t[1] - t[0]
    shift = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
    return float(t[i] + shift * dt)


def areal_velocity(t, y):
    return 0.5 * (y[:, 0] * y[:, 3] - y[:, 1] * y[:, 2])


def energy(y):
    r = np.hypot(y[:, 0], y[:, 1])
    return 0.5 * (y[:, 2] ** 2 + y[:, 3] ** 2) - GM / r


def kepler_third_law_table(semi_axes=(0.5, 1.0, 2.0, 4.0)):
    return [{"a": a, "T_an": period_analytic(a), "T_nu": period_numeric(a, 0.3)} for a in semi_axes]


# ------------------------------------------------------------- velocidad de escape
G_N = 6.674e-11
M_EARTH = 5.972e24
R_EARTH = 6.371e6


def escape_speed_analytic(m_planet: float = M_EARTH, radius: float = R_EARTH) -> float:
    return float(np.sqrt(2 * G_N * m_planet / radius))


def rmax_analytic(v0: float, m_planet: float = M_EARTH, radius: float = R_EARTH) -> float:
    mu = G_N * m_planet
    return float(radius / (1 - v0**2 * radius / (2 * mu)))


def radial_launch_numeric(v0: float, m_planet: float = M_EARTH, radius: float = R_EARTH, t_end: float = 4000.0, n: int = 40000):
    mu = G_N * m_planet
    return integrate(lambda _t, s: np.array([s[1], -mu / s[0] ** 2]), np.array([radius, v0]), 0.0, t_end, n, rk4_step)


def rmax_numeric(v0: float, **kw) -> float:
    _, y = radial_launch_numeric(v0, **kw)
    return float(np.max(y[:, 0]))


def escape_speed_numeric(m_planet: float = M_EARTH, radius: float = R_EARTH) -> float:
    """Biseccion sobre v0: escapa si la velocidad sigue positiva tras un tiempo largo."""
    mu = G_N * m_planet
    lo, hi = 0.5 * np.sqrt(mu / radius), 2.0 * np.sqrt(mu / radius)
    for _ in range(45):
        mid = 0.5 * (lo + hi)
        # integracion en fase con variable r: se detiene al volver a la superficie o superar 10^4 R
        s = np.array([radius, mid])
        dt = 2.0
        escaped = True
        for _ in range(400000):
            s = rk4_step(lambda _t, y: np.array([y[1], -mu / y[0] ** 2]), 0.0, s, dt)
            if s[1] <= 0:
                escaped = False
                break
            if s[0] > 1e4 * radius:
                break
            dt = min(1e5, max(2.0, 0.002 * s[0] / max(abs(s[1]), 1.0)))
        if escaped:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)
