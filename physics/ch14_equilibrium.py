"""Capitulo 14. Equilibrio de cuerpos rigidos."""
from __future__ import annotations

import numpy as np

G = 9.81

# -------------------------------------------------------------------- escalera
L_LAD = 5.0
W_LAD = 150.0  # N
W_MAN = 700.0
THETA_DEG = 60.0
MU = 0.40


def ladder_forces_analytic(d: float, theta_deg: float = THETA_DEG, length: float = L_LAD,
                           wl: float = W_LAD, wp: float = W_MAN):
    th = np.deg2rad(theta_deg)
    nw = (wl * length / 2 + wp * d) * np.cos(th) / (length * np.sin(th))
    return {"N_floor": wl + wp, "f": nw, "N_wall": nw}


def ladder_forces_numeric(d: float, theta_deg: float = THETA_DEG, length: float = L_LAD,
                          wl: float = W_LAD, wp: float = W_MAN):
    """Equilibrio como sistema lineal: sum Fx = 0, sum Fy = 0, sum tau (respecto al pie) = 0."""
    th = np.deg2rad(theta_deg)
    # incognitas: [N_floor, f, N_wall]
    a = np.array([
        [0.0, 1.0, -1.0],  # Fx: f - N_wall = 0
        [1.0, 0.0, 0.0],  # Fy: N_floor = wl + wp
        [0.0, 0.0, length * np.sin(th)],  # torque pared = pesos
    ])
    b = np.array([0.0, wl + wp, (wl * length / 2 + wp * d) * np.cos(th)])
    n_floor, f, n_wall = np.linalg.solve(a, b)
    return {"N_floor": n_floor, "f": f, "N_wall": n_wall}


def d_max_analytic(mu: float = MU, theta_deg: float = THETA_DEG, length: float = L_LAD, wl: float = W_LAD, wp: float = W_MAN) -> float:
    th = np.deg2rad(theta_deg)
    return (mu * (wl + wp) * length * np.tan(th) - wl * length / 2) / wp


def d_max_numeric(mu: float = MU, theta_deg: float = THETA_DEG, length: float = L_LAD) -> float:
    lo, hi = 0.0, length
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        fr = ladder_forces_numeric(mid, theta_deg, length)
        if fr["f"] <= mu * fr["N_floor"]:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def min_angle_analytic(mu: float = MU, d_frac: float = 1.0, length: float = L_LAD, wl: float = W_LAD, wp: float = W_MAN) -> float:
    """Angulo minimo para subir hasta d = d_frac L: tan(theta) = (wl/2 + wp d_frac) / (mu (wl+wp))."""
    return float(np.rad2deg(np.arctan((wl / 2 + wp * d_frac) / (mu * (wl + wp)))))


def min_angle_numeric(mu: float = MU, d_frac: float = 1.0, length: float = L_LAD) -> float:
    lo, hi = 20.0, 89.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        fr = ladder_forces_numeric(d_frac * length, mid, length)
        if fr["f"] <= mu * fr["N_floor"]:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


# ------------------------------------------------------------------------ viga
BEAM_L = 6.0
BEAM_W = 400.0
PERSON_W = 700.0


def beam_reactions_analytic(x: float, length: float = BEAM_L, wb: float = BEAM_W, wp: float = PERSON_W):
    ra = wb / 2 + wp * (length - x) / length
    rb = wb / 2 + wp * x / length
    return ra, rb


def beam_reactions_numeric(x: float, pivot: float = 0.0, length: float = BEAM_L, wb: float = BEAM_W, wp: float = PERSON_W):
    """Sistema lineal con torques respecto a un punto arbitrario (el resultado no depende del pivote)."""
    a = np.array([[1.0, 1.0], [0.0 - pivot, length - pivot]])
    b = np.array([wb + wp, wb * (length / 2 - pivot) + wp * (x - pivot)])
    return tuple(np.linalg.solve(a, b))
