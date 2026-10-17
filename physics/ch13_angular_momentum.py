"""Capitulo 13. Momento angular."""
from __future__ import annotations

import numpy as np

from .integrators import integrate, rk4_step

G = 9.81

# ------------------------------------------------------ patinadora (L constante)
I_I, I_F = 4.0, 1.6
W_I = 2.0
T_PULL = 2.0


def omega_final_analytic(i_i: float = I_I, i_f: float = I_F, w_i: float = W_I) -> float:
    return i_i * w_i / i_f


def inertia_profile(t, i_i: float = I_I, i_f: float = I_F, t_pull: float = T_PULL):
    """I(t) con suavizado cubico entre t = 0 y t_pull (derivada nula en los extremos)."""
    s = np.clip(np.asarray(t, dtype=float) / t_pull, 0.0, 1.0)
    return i_i + (i_f - i_i) * (3 * s**2 - 2 * s**3)


def inertia_rate(t, i_i: float = I_I, i_f: float = I_F, t_pull: float = T_PULL):
    s = np.clip(np.asarray(t, dtype=float) / t_pull, 0.0, 1.0)
    inside = (np.asarray(t) > 0) & (np.asarray(t) < t_pull)
    return np.where(inside, (i_f - i_i) * (6 * s - 6 * s**2) / t_pull, 0.0)


def skater_numeric(t_end: float = 4.0, n: int = 4000):
    """Sin torque externo: d(I w)/dt = 0  =>  w' = -(I'/I) w. Se integra con RK4."""
    f = lambda t, y: np.array([-(inertia_rate(t) / inertia_profile(t)) * y[0]])  # noqa: E731
    t, y = integrate(f, np.array([W_I]), 0.0, t_end, n, rk4_step)
    w = y[:, 0]
    inertia = inertia_profile(t)
    return t, w, inertia, inertia * w, 0.5 * inertia * w**2


def kinetic_energy_change_analytic(i_i: float = I_I, i_f: float = I_F, w_i: float = W_I) -> float:
    return 0.5 * i_f * omega_final_analytic(i_i, i_f, w_i) ** 2 - 0.5 * i_i * w_i**2


# -------------------------------------------------------------- precesion giroscopica
M_GYRO = 0.5
R_GYRO = 0.10
I_GYRO = 2.5e-3
W_SPIN = 200.0


def precession_analytic(w_spin: float = W_SPIN, m: float = M_GYRO, r: float = R_GYRO, inertia: float = I_GYRO, g: float = G) -> float:
    return m * g * r / (inertia * w_spin)


def precession_numeric(w_spin: float = W_SPIN, m: float = M_GYRO, r: float = R_GYRO, inertia: float = I_GYRO,
                       g: float = G, t_end: float = 6.0, n: int = 6000):
    """Integra dL/dt = r x (m g) en 3D con L = I w_spin * eje; el eje queda horizontal."""
    big_l = inertia * w_spin

    def rhs(_t, s):
        axis = s / np.linalg.norm(s)
        torque = np.cross(r * axis, np.array([0.0, 0.0, -m * g]))
        return torque

    t, y = integrate(rhs, np.array([big_l, 0.0, 0.0]), 0.0, t_end, n, rk4_step)
    phi = np.unwrap(np.arctan2(y[:, 1], y[:, 0]))
    omega_p = np.polyfit(t, phi, 1)[0]
    return t, y, phi, float(omega_p)
