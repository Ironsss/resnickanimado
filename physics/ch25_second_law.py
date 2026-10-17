"""Capitulo 25. Segunda ley y entropia."""
from __future__ import annotations

import numpy as np

from .integrators import integrate, rk4_step
from .ch07_work_energy import simpson
from . import ch22_temperature as c22

R = 8.314
N_MOL = 1.0
GAMMA = 5 / 3
TH, TC = 500.0, 300.0
V_A = 1.0e-3
V_B = 2.0e-3


def carnot_efficiency_analytic(th: float = TH, tc: float = TC) -> float:
    return 1.0 - tc / th


def carnot_states(th: float = TH, tc: float = TC, v_a: float = V_A, v_b: float = V_B, gamma: float = GAMMA):
    p_a = N_MOL * R * th / v_a
    p_b = N_MOL * R * th / v_b
    v_c = v_b * (th / tc) ** (1 / (gamma - 1))
    v_d = v_a * (th / tc) ** (1 / (gamma - 1))
    p_c = N_MOL * R * tc / v_c
    p_d = N_MOL * R * tc / v_d
    return {"A": (p_a, v_a), "B": (p_b, v_b), "C": (p_c, v_c), "D": (p_d, v_d), "v_c": v_c, "v_d": v_d}


def carnot_numeric(th: float = TH, tc: float = TC, n: int = 2000):
    """Integra p dV por tramos: isotermas (p = nRT/V) y adiabatas (RK4 sobre dp/dV = -gamma p / V)."""
    st = carnot_states(th, tc)
    (pa, va), (pb, vb), (pc, vc), (pd, vd) = st["A"], st["B"], st["C"], st["D"]
    v_ab = np.linspace(va, vb, n + 1)
    w_ab = simpson(N_MOL * R * th / v_ab, v_ab)
    _, y_bc = integrate(lambda v, y: np.array([-GAMMA * y[0] / v]), np.array([pb]), vb, vc, n, rk4_step)
    v_bc = np.linspace(vb, vc, n + 1)
    w_bc = simpson(y_bc[:, 0], v_bc)
    v_cd = np.linspace(vc, vd, n + 1)
    w_cd = simpson(N_MOL * R * tc / v_cd, v_cd)  # negativo: vd < vc
    _, y_da = integrate(lambda v, y: np.array([-GAMMA * y[0] / v]), np.array([pd]), vd, va, n, rk4_step)
    v_da = np.linspace(vd, va, n + 1)
    w_da = simpson(y_da[:, 0], v_da)
    w_net = w_ab + w_bc + w_cd + w_da
    q_h = w_ab  # isoterma: Q = W
    return {"W_net": w_net, "Q_h": q_h, "eta": w_net / q_h, "pa_end": float(y_da[-1, 0]), "pa": pa,
            "paths": {"AB": (v_ab, N_MOL * R * th / v_ab), "BC": (v_bc, y_bc[:, 0]), "CD": (v_cd, N_MOL * R * tc / v_cd), "DA": (v_da, y_da[:, 0])}}


# --------------------------------------------------------------------- entropia
def heating_entropy_analytic(m: float = 1.0, c: float = 4186.0, t1: float = 20.0, t2: float = 80.0) -> float:
    return m * c * np.log((t2 + 273.15) / (t1 + 273.15))


def heating_entropy_numeric(m: float = 1.0, c: float = 4186.0, t1: float = 20.0, t2: float = 80.0, n: int = 600) -> float:
    temp = np.linspace(t1 + 273.15, t2 + 273.15, n + 1)
    return simpson(m * c / temp, temp)  # dS = dQ/T = m c dT / T


def contact_entropy_analytic() -> float:
    tf = c22.final_temperature_analytic() + 273.15
    return float(c22.M1 * c22.C1 * np.log(tf / (c22.T1_0 + 273.15)) + c22.M2 * c22.C2 * np.log(tf / (c22.T2_0 + 273.15)))


def contact_entropy_numeric(t_end: float = 120.0, n: int = 6000) -> float:
    """Integra la produccion de entropia dS/dt = h (T1 - T2)^2 / (T1 T2) a lo largo de la evolucion RK4."""
    rhs = lambda _t, y: np.array([-c22.H * (y[0] - y[1]) / (c22.M1 * c22.C1), c22.H * (y[0] - y[1]) / (c22.M2 * c22.C2),
                                   c22.H * (y[0] - y[1]) ** 2 / ((y[0] + 273.15) * (y[1] + 273.15))])  # noqa: E731
    t, y = integrate(rhs, np.array([c22.T1_0, c22.T2_0, 0.0]), 0.0, t_end, n, rk4_step)
    return float(y[-1, 2])
