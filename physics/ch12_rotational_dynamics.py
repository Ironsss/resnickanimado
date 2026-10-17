"""Capitulo 12. Dinamica rotacional."""
from __future__ import annotations

import numpy as np

from .integrators import integrate, rk4_step
from .ch07_work_energy import simpson

G = 9.81

# ------------------------------------------------------------ momentos de inercia
M_ROD = 3.0
L_ROD = 1.2


def rod_inertia_analytic(axis: str, m: float = M_ROD, length: float = L_ROD) -> float:
    return {"end": m * length**2 / 3, "center": m * length**2 / 12}[axis]


def rod_inertia_numeric(axis: str, m: float = M_ROD, length: float = L_ROD, n: int = 400) -> float:
    x = np.linspace(0, length, n + 1)
    pivot = {"end": 0.0, "center": length / 2}[axis]
    return simpson((x - pivot) ** 2 * (m / length) * np.ones_like(x), x)


def disc_inertia_analytic(m: float = 4.0, r: float = 0.25) -> float:
    return 0.5 * m * r**2


def disc_inertia_numeric(m: float = 4.0, r: float = 0.25, n: int = 400) -> float:
    """Integra anillos: dm = sigma 2 pi s ds, I = int s^2 dm."""
    s = np.linspace(0, r, n + 1)
    sigma = m / (np.pi * r**2)
    return simpson(s**2 * sigma * 2 * np.pi * s, s)


def parallel_axis(i_cm: float, m: float, d: float) -> float:
    return i_cm + m * d**2


# -------------------------------------------------------- polea con masa (disco)
M_DISC, R_DISC, M_HANG = 4.0, 0.25, 2.0


def hanging_analytic(m: float = M_HANG, big_m: float = M_DISC, r: float = R_DISC, g: float = G):
    a = m * g / (m + big_m / 2)
    return {"a": a, "T": m * (g - a), "alpha": a / r}


def hanging_numeric(t_end: float = 1.0, n: int = 1000, m: float = M_HANG, big_m: float = M_DISC, r: float = R_DISC, g: float = G):
    """Integra el sistema acoplado (y, v, theta, omega) con la cuerda inextensible como ligadura.

    Las ecuaciones de cada cuerpo se plantean por separado: m a = m g - T, I alpha = T r, y se resuelve
    el sistema 2x2 en cada paso (T, a) antes de avanzar con RK4.
    """
    inertia = 0.5 * big_m * r**2

    def accel(_state):
        mat = np.array([[m, 1.0], [-r * r / inertia * 0 + 0.0, 0.0]])  # placeholder para claridad
        # incognitas (a, T):  m a + T = m g ;  I a / r - T r = 0
        mat = np.array([[m, 1.0], [inertia / r, -r]])
        rhs = np.array([m * g, 0.0])
        a, T = np.linalg.solve(mat, rhs)
        return a, T

    def rhs(_t, s):
        a, _ = accel(s)
        return np.array([s[1], a, s[3], a / r])

    t, y = integrate(rhs, np.zeros(4), 0.0, t_end, n, rk4_step)
    a, T = accel(y[-1])
    energy_in = m * g * y[-1, 0]
    energy_out = 0.5 * m * y[-1, 1] ** 2 + 0.5 * inertia * y[-1, 3] ** 2
    return {"t": t, "y": y, "a": a, "T": T, "energy_error": abs(energy_in - energy_out)}
