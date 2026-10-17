"""Volumen 2, capitulo 13. Circuitos de corriente alterna."""
from __future__ import annotations

import numpy as np

from .integrators import integrate, rk4_step
from .ch07_work_energy import simpson

R, L, C, V0 = 20.0, 0.1, 10e-6, 50.0


def omega0() -> float:
    return float(1 / np.sqrt(L * C))


def impedance(w, r: float = R, l_: float = L, c: float = C):
    w = np.asarray(w, dtype=float)
    return np.sqrt(r**2 + (w * l_ - 1 / (w * c)) ** 2)


def current_amplitude_analytic(w):
    return V0 / impedance(w)


def phase_analytic(w):
    w = np.asarray(w, dtype=float)
    return np.arctan((w * L - 1 / (w * C)) / R)  # i atrasa a v cuando es positivo


def steady_state_numeric(w: float, cycles_settle: int = 25, cycles_measure: int = 6):
    """RK4 en el dominio del tiempo y deteccion sincrona (proyeccion sobre sen y cos) en un numero entero de ciclos."""
    period = 2 * np.pi / w
    t_end = (cycles_settle + cycles_measure) * period
    n = int((cycles_settle + cycles_measure) * 400)
    rhs = lambda t, y: np.array([y[1], (V0 * np.sin(w * t) - R * y[1] - y[0] / C) / L])  # noqa: E731
    t, y = integrate(rhs, np.zeros(2), 0.0, t_end, n, rk4_step)
    i = y[:, 1]
    k0 = cycles_settle * 400
    tm, im = t[k0:], i[k0:]
    # i = I0 sin(wt - phi) = I0 cos(phi) sin(wt) - I0 sin(phi) cos(wt)
    a = 2 * simpson(im * np.sin(w * tm), tm) / (tm[-1] - tm[0])
    b = 2 * simpson(im * np.cos(w * tm), tm) / (tm[-1] - tm[0])
    amp = float(np.hypot(a, b))
    phi = float(np.arctan2(-b, a))
    return amp, phi


def resonance_curve_numeric(ws):
    return [steady_state_numeric(float(w))[0] for w in ws]


def q_factor() -> float:
    return omega0() * L / R


# -------------------------------------------------------------- potencia y transformador
V_PEAK, I_PEAK, PHI_DEG = 170.0, 8.0, 40.0
N_P, N_S = 200, 50


def power_average_analytic() -> float:
    return 0.5 * V_PEAK * I_PEAK * np.cos(np.deg2rad(PHI_DEG))


def rms_numeric(amplitude: float, n: int = 4000) -> float:
    t = np.linspace(0, 1.0, n + 1)  # un periodo normalizado
    return float(np.sqrt(simpson((amplitude * np.sin(2 * np.pi * t)) ** 2, t)))


def power_average_numeric(n: int = 4000) -> float:
    t = np.linspace(0, 1.0, n + 1)
    v = V_PEAK * np.sin(2 * np.pi * t)
    i = I_PEAK * np.sin(2 * np.pi * t - np.deg2rad(PHI_DEG))
    return simpson(v * i, t)


def transformer_analytic(vp_rms: float = 120.0, ip_rms: float = 2.0):
    return {"vs": vp_rms * N_S / N_P, "is": ip_rms * N_P / N_S}


def transformer_numeric(vp_rms: float = 120.0, ip_rms: float = 2.0):
    """Ondas muestreadas: la razon de vueltas escala v (x N_S/N_P) y conserva v*i (transformador ideal)."""
    t = np.linspace(0, 1.0, 4001)
    vp = np.sqrt(2) * vp_rms * np.sin(2 * np.pi * t)
    ip = np.sqrt(2) * ip_rms * np.sin(2 * np.pi * t)
    vs = vp * N_S / N_P
    is_ = ip * N_P / N_S
    p_p = simpson(vp * ip, t)
    p_s = simpson(vs * is_, t)
    return {"vs": float(np.sqrt(simpson(vs**2, t))), "is": float(np.sqrt(simpson(is_**2, t))), "p_p": p_p, "p_s": p_s}
