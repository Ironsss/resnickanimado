"""Volumen 2, capitulo 7. Circuitos de corriente directa."""
from __future__ import annotations

import numpy as np

from .integrators import integrate, rk4_step
from .ch07_work_energy import simpson

# ----------------------------------------------------------------- corrientes de malla
E1, E2 = 12.0, 6.0
R1, R2, R3 = 4.0, 2.0, 6.0  # R1 en la malla izquierda, R2 en la derecha, R3 compartida


def mesh_currents_numeric(e1: float = E1, e2: float = E2, r1: float = R1, r2: float = R2, r3: float = R3):
    """Mallas: (R1 + R3) i1 - R3 i2 = E1 ; -R3 i1 + (R2 + R3) i2 = -E2."""
    a = np.array([[r1 + r3, -r3], [-r3, r2 + r3]])
    b = np.array([e1, -e2])
    return np.linalg.solve(a, b)


def mesh_currents_analytic(e1: float = E1, e2: float = E2, r1: float = R1, r2: float = R2, r3: float = R3):
    det = (r1 + r3) * (r2 + r3) - r3**2
    i1 = (e1 * (r2 + r3) - r3 * e2) / det
    i2 = (r3 * e1 - (r1 + r3) * e2) / det
    return np.array([i1, i2])


def branch_currents(i):
    return {"R1": i[0], "R2": i[1], "R3": i[0] - i[1]}


def kirchhoff_residuals(i, e1: float = E1, e2: float = E2, r1: float = R1, r2: float = R2, r3: float = R3):
    b = branch_currents(i)
    loop_left = e1 - r1 * b["R1"] - r3 * b["R3"]
    loop_right = -e2 - r2 * b["R2"] + r3 * b["R3"]
    power_src = e1 * i[0] - e2 * i[1]
    power_res = r1 * b["R1"] ** 2 + r2 * b["R2"] ** 2 + r3 * b["R3"] ** 2
    return {"loop_left": loop_left, "loop_right": loop_right, "power_balance": power_src - power_res}


# --------------------------------------------------------------------- carga de un RC
EPS, R, C = 12.0, 2.0e3, 100e-6


def q_analytic(t, eps: float = EPS, r: float = R, c: float = C):
    return c * eps * (1 - np.exp(-np.asarray(t, dtype=float) / (r * c)))


def i_analytic(t, eps: float = EPS, r: float = R, c: float = C):
    return (eps / r) * np.exp(-np.asarray(t, dtype=float) / (r * c))


def rc_numeric(t_end: float = 1.2, n: int = 2400, eps: float = EPS, r: float = R, c: float = C):
    rhs = lambda _t, y: np.array([(eps - y[0] / c) / r])  # noqa: E731
    return integrate(rhs, np.array([0.0]), 0.0, t_end, n, rk4_step)


def rc_energies_numeric(eps: float = EPS, r: float = R, c: float = C, t_end: float = 3.0, n: int = 6000):
    t, y = rc_numeric(t_end, n, eps, r, c)
    q = y[:, 0]
    i = (eps - q / c) / r
    e_diss = simpson(i**2 * r, t)
    e_cap = q[-1] ** 2 / (2 * c)
    e_batt = simpson(eps * i, t)
    return {"dissipated": e_diss, "capacitor": e_cap, "battery": e_batt}


def time_to_fraction_numeric(frac: float = 0.632, **kw) -> float:
    t, y = rc_numeric(2.0, 4000, **kw)
    target = frac * C * EPS
    i = int(np.argmax(y[:, 0] >= target))
    return float(t[i - 1] + (target - y[i - 1, 0]) * (t[i] - t[i - 1]) / (y[i, 0] - y[i - 1, 0]))
