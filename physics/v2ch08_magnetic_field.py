"""Volumen 2, capitulo 8. El campo magnetico."""
from __future__ import annotations

import numpy as np

from .constants import E_CH, M_P, M_E
from .integrators import integrate, rk4_step

B = 0.1  # T
V_PERP, V_PAR = 3.0e5, 1.0e5


def radius_analytic(v_perp: float = V_PERP, b: float = B, m: float = M_P, q: float = E_CH) -> float:
    return m * v_perp / (q * b)


def period_analytic(b: float = B, m: float = M_P, q: float = E_CH) -> float:
    return 2 * np.pi * m / (q * b)


def pitch_analytic(v_par: float = V_PAR) -> float:
    return v_par * period_analytic()


def _accel(v, b=B, m=M_P, q=E_CH):
    return (q / m) * np.cross(v, np.array([0.0, 0.0, b]))


def helix_rk4(turns: float = 4.0, steps_per_turn: int = 50):
    t_end = turns * period_analytic()
    n = int(turns * steps_per_turn)
    rhs = lambda _t, s: np.concatenate([s[3:], _accel(s[3:])])  # noqa: E731
    return integrate(rhs, np.array([0.0, 0.0, 0.0, V_PERP, 0.0, V_PAR]), 0.0, t_end, n, rk4_step)


def helix_boris(turns: float = 4.0, steps_per_turn: int = 50):
    """Integrador de Boris (conserva |v| exactamente para un campo magnetico puro)."""
    t_end = turns * period_analytic()
    n = int(turns * steps_per_turn)
    dt = t_end / n
    x = np.zeros((n + 1, 3))
    v = np.zeros((n + 1, 3))
    v[0] = [V_PERP, 0.0, V_PAR]
    tvec = (E_CH / M_P) * np.array([0.0, 0.0, B]) * dt / 2
    svec = 2 * tvec / (1 + tvec @ tvec)
    for i in range(n):
        v_prime = v[i] + np.cross(v[i], tvec)
        v[i + 1] = v[i] + np.cross(v_prime, svec)
        x[i + 1] = x[i] + v[i + 1] * dt
    t = np.linspace(0, t_end, n + 1)
    return t, np.hstack([x, v])


def measured_radius(y: np.ndarray) -> float:
    return float((y[:, 1].max() - y[:, 1].min()) / 2)  # el circulo esta centrado en y = r (v inicial en +x)


def speed_drift(y: np.ndarray) -> float:
    s = np.linalg.norm(y[:, 3:], axis=1)
    return float(np.max(np.abs(s - s[0])) / s[0])


def measured_period(t: np.ndarray, y: np.ndarray) -> float:
    """Tiempo entre dos pasos ascendentes de y por su valor medio (interpolacion lineal)."""
    mid = 0.5 * (y[:, 1].max() + y[:, 1].min())
    f = y[:, 1] - mid
    idx = [i for i in range(1, len(f)) if f[i - 1] < 0 <= f[i]]
    t0 = t[idx[0]] - f[idx[0]] * (t[idx[0]] - t[idx[0] - 1]) / (f[idx[0]] - f[idx[0] - 1])
    t1 = t[idx[1]] - f[idx[1]] * (t[idx[1]] - t[idx[1] - 1]) / (f[idx[1]] - f[idx[1] - 1])
    return float(t1 - t0)


# --------------------------------------------------------------------------- efecto Hall
N_CU = 8.5e28
THICK = 1.0e-4
WIDTH = 1.0e-2
I_HALL = 5.0
B_HALL = 0.5
TAU = 2.5e-14


def hall_voltage_analytic(i: float = I_HALL, b: float = B_HALL, n: float = N_CU, t: float = THICK) -> float:
    return i * b / (n * E_CH * t)


def _vy_steady(ey: float, ex: float, b: float) -> float:
    """Velocidad transversal estacionaria de los electrones (integracion RK4 de la ecuacion de Drude con campo magnetico)."""
    k = E_CH / M_E
    def rhs(_t, v):
        return np.array([-k * (ex + v[1] * b) - v[0] / TAU, -k * (ey - v[0] * b) - v[1] / TAU])
    _, y = integrate(rhs, np.zeros(2), 0.0, 40 * TAU, 800, rk4_step)
    return float(y[-1, 1])


def hall_voltage_numeric(i: float = I_HALL, b: float = B_HALL, n: float = N_CU, t: float = THICK, w: float = WIDTH) -> float:
    v_d = i / (n * E_CH * w * t)
    ex = M_E * v_d / (E_CH * TAU)  # campo longitudinal que sostiene la deriva
    lo, hi = -2 * v_d * b, 2 * v_d * b
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if _vy_steady(mid, ex, b) > 0:
            lo = mid
        else:
            hi = mid
    ey = 0.5 * (lo + hi)
    return abs(ey) * w


def carrier_density_from_hall(v_h: float, i: float = I_HALL, b: float = B_HALL, t: float = THICK) -> float:
    return i * b / (E_CH * t * v_h)
