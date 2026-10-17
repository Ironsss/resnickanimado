"""Capitulo 23. Primera ley de la termodinamica."""
from __future__ import annotations

import numpy as np

from .integrators import integrate, rk4_step
from .ch07_work_energy import simpson

R = 8.314
N_MOL = 1.0
GAMMA_MONO = 5 / 3
GAMMA_DIA = 7 / 5
P1, V1, V2 = 2.0e5, 0.010, 0.020


def temperature(p: float, v: float, n: float = N_MOL) -> float:
    return p * v / (n * R)


def work_analytic(kind: str, p1: float = P1, v1: float = V1, v2: float = V2, gamma: float = GAMMA_MONO) -> float:
    if kind == "isothermal":
        return p1 * v1 * np.log(v2 / v1)
    if kind == "isobaric":
        return p1 * (v2 - v1)
    p2 = p1 * (v1 / v2) ** gamma
    return (p1 * v1 - p2 * v2) / (gamma - 1)


def pressure_path(kind: str, v, p1: float = P1, v1: float = V1, gamma: float = GAMMA_MONO):
    v = np.asarray(v, dtype=float)
    if kind == "isothermal":
        return p1 * v1 / v
    if kind == "isobaric":
        return np.full_like(v, p1)
    return p1 * (v1 / v) ** gamma


def adiabat_numeric(v_end: float = V2, n: int = 400, p1: float = P1, v1: float = V1, gamma: float = GAMMA_MONO):
    """Integra dp/dV = -gamma p / V con RK4 (independiente de la forma cerrada p V^gamma = cte)."""
    return integrate(lambda v, y: np.array([-gamma * y[0] / v]), np.array([p1]), v1, v_end, n, rk4_step)


def work_numeric(kind: str, n: int = 400, **kw) -> float:
    v = np.linspace(V1, V2, n + 1)
    if kind == "adiabatic":
        gamma = kw.get("gamma", GAMMA_MONO)
        _, y = adiabat_numeric(n=n, gamma=gamma)
        p = y[:, 0]
    else:
        p = pressure_path(kind, v)
    return simpson(p, v)


def first_law_table(gamma: float = GAMMA_MONO):
    cv = R / (gamma - 1)
    rows = []
    t1 = temperature(P1, V1)
    for kind in ("isothermal", "isobaric", "adiabatic"):
        w = work_numeric(kind)
        if kind == "isothermal":
            t2 = t1
        elif kind == "isobaric":
            t2 = temperature(P1, V2)
        else:
            t2 = t1 * (V1 / V2) ** (gamma - 1)
        du = N_MOL * cv * (t2 - t1)
        rows.append({"kind": kind, "W": w, "dU": du, "Q": du + w, "T2": t2})
    return rows


# ------------------------------------------------------------------ ciclo de Otto
RATIO = 8.0
T_A = 300.0
T_C_HEAT = 1800.0  # temperatura tras la combustion (fin de la isocora de calentamiento)


def otto_states(r: float = RATIO, gamma: float = GAMMA_DIA, t_a: float = T_A, t_c: float = T_C_HEAT, v_a: float = 0.025):
    """Estados A (admision) -> B (compresion adiabatica) -> C (calentamiento isocoro) -> D (expansion adiabatica)."""
    v_b = v_a / r
    p_a = N_MOL * R * t_a / v_a
    t_b = t_a * r ** (gamma - 1)
    p_b = N_MOL * R * t_b / v_b
    p_c = N_MOL * R * t_c / v_b
    t_d = t_c * r ** (1 - gamma)
    p_d = N_MOL * R * t_d / v_a
    return {"A": (p_a, v_a, t_a), "B": (p_b, v_b, t_b), "C": (p_c, v_b, t_c), "D": (p_d, v_a, t_d)}


def otto_efficiency_analytic(r: float = RATIO, gamma: float = GAMMA_DIA) -> float:
    return 1.0 - r ** (1.0 - gamma)


def otto_numeric(r: float = RATIO, gamma: float = GAMMA_DIA, n: int = 800):
    st = otto_states(r, gamma)
    pa, va, _ = st["A"]
    pb, vb, _ = st["B"]
    pc, vc, tc = st["C"]
    pd, vd, _ = st["D"]
    # compresion A->B: integra dp/dV con RK4 desde V_A hasta V_B
    _, yab = integrate(lambda v, y: np.array([-gamma * y[0] / v]), np.array([pa]), va, vb, n, rk4_step)
    v_ab = np.linspace(va, vb, n + 1)
    w_ab = simpson(yab[:, 0], v_ab)  # negativo (trabajo sobre el gas)
    _, ycd = integrate(lambda v, y: np.array([-gamma * y[0] / v]), np.array([pc]), vc, vd, n, rk4_step)
    v_cd = np.linspace(vc, vd, n + 1)
    w_cd = simpson(ycd[:, 0], v_cd)
    cv = R / (gamma - 1)
    q_in = N_MOL * cv * (tc - st["B"][2])
    w_net = w_ab + w_cd
    return {"W_net": w_net, "Q_in": q_in, "eta": w_net / q_in, "states": st, "ab": (v_ab, yab[:, 0]), "cd": (v_cd, ycd[:, 0])}
