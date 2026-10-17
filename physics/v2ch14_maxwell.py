"""Volumen 2, capitulo 14. Ecuaciones de Maxwell y ondas electromagneticas."""
from __future__ import annotations

import numpy as np

from .constants import C_LIGHT, EPS0, MU0
from .ch07_work_energy import simpson


def c_from_constants() -> float:
    return float(1 / np.sqrt(MU0 * EPS0))


def yee_pulse(n: int = 1200, courant: float = 0.5, steps: int = 600, dx: float = 1e-3):
    """FDTD 1D de Yee en el vacio (SI): Ez y By en mallas escalonadas; pulso gaussiano que viaja hacia +x."""
    dt = courant * dx / C_LIGHT
    ez = np.zeros(n)
    hy = np.zeros(n - 1)
    x = np.arange(n) * dx
    x0, sigma = 0.2 * n * dx, 20 * dx
    # pulso que viaja a +x: E(x) = f(x - x0), Hy = -Ez / (mu0 c) con la convencion de signos de las ecuaciones de Faraday y Ampere-Maxwell
    ez[:] = np.exp(-((x - x0) ** 2) / (2 * sigma**2))
    xh = (np.arange(n - 1) + 0.5) * dx
    hy[:] = -np.exp(-((xh - x0 + 0.5 * C_LIGHT * dt) ** 2) / (2 * sigma**2)) / (MU0 * C_LIGHT)
    peaks = []
    for k in range(steps):
        hy += dt / (MU0 * dx) * (ez[1:] - ez[:-1])
        ez[1:-1] += dt / (EPS0 * dx) * (hy[1:] - hy[:-1])
        if k % 20 == 0 or k == steps - 1:
            idx = int(np.argmax(ez))
            # refinamiento parabolico del maximo
            y0, y1, y2 = ez[idx - 1], ez[idx], ez[idx + 1]
            shift = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
            peaks.append(((k + 1) * dt, (idx + shift) * dx))
    return x, ez, np.array(peaks)


def speed_numeric(courant: float = 0.5) -> float:
    _, _, peaks = yee_pulse(courant=courant)
    return float(np.polyfit(peaks[:, 0], peaks[:, 1], 1)[0])


# ------------------------------------------------------- Poynting y corriente de desplazamiento
E0 = 100.0  # V/m


def intensity_analytic(e0: float = E0) -> float:
    return 0.5 * C_LIGHT * EPS0 * e0**2


def intensity_numeric(e0: float = E0, n: int = 4000) -> float:
    """Promedio temporal del modulo del vector de Poynting S = E x B / mu0 con B = E / c."""
    t = np.linspace(0, 1.0, n + 1)
    e = e0 * np.sin(2 * np.pi * t)
    b = e / C_LIGHT
    s = e * b / MU0
    return simpson(s, t)


def radiation_pressure(intensity: float, perfect_absorber: bool = True) -> float:
    return intensity / C_LIGHT if perfect_absorber else 2 * intensity / C_LIGHT


AREA, GAP, DVDT = 0.01, 1.0e-3, 1.0e6


def conduction_current_analytic() -> float:
    return EPS0 * AREA / GAP * DVDT


def displacement_current_numeric(h: float = 1e-9) -> float:
    """eps0 dPhi_E/dt con Phi_E = (V(t)/d) A y V(t) = DVDT t, derivando por diferencias centradas."""
    phi = lambda t: (DVDT * t / GAP) * AREA  # noqa: E731
    t0 = 1e-6
    return float(EPS0 * (phi(t0 + h) - phi(t0 - h)) / (2 * h))
