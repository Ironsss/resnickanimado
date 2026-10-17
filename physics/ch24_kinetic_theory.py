"""Capitulo 24. Teoria cinetica de los gases."""
from __future__ import annotations

import numpy as np

from .ch07_work_energy import simpson

KB = 1.380649e-23
M_HE = 6.646e-27  # kg (helio)
M_N2 = 4.652e-26  # kg (N2)
T_GAS = 300.0

# ------------------------------------------------- presion por transferencia de momento
BOX = 1.0e-6  # lado del cubo [m]


def pressure_analytic(n: int, temperature: float = T_GAS, volume: float = BOX**3) -> float:
    return n * KB * temperature / volume


def pressure_simulation(n: int = 4000, temperature: float = T_GAS, m: float = M_HE, box: float = BOX,
                        duration: float = 2.0e-8, dt: float = 2.0e-12, seed: int = 21):
    """Moleculas independientes que rebotan elasticamente en las paredes de un cubo.

    La presion se mide como impulso total transferido a la pared x = L entre el tiempo, dividido por su area.
    La temperatura se estima a partir de la energia cinetica media de la propia simulacion (3/2 kT = <K>).
    """
    rng = np.random.default_rng(seed)
    sigma = np.sqrt(KB * temperature / m)
    v = rng.normal(0.0, sigma, size=(n, 3))
    x = rng.uniform(0, box, size=(n, 3))
    steps = int(round(duration / dt))
    impulse = 0.0
    for _ in range(steps):
        x += v * dt
        hi = x[:, 0] > box
        impulse += float(np.sum(2 * m * v[hi, 0]))
        x[hi, 0] = 2 * box - x[hi, 0]
        v[hi, 0] *= -1
        lo = x[:, 0] < 0
        x[lo, 0] *= -1
        v[lo, 0] *= -1
        for ax in (1, 2):
            o = (x[:, ax] > box) | (x[:, ax] < 0)
            x[o, ax] = np.where(x[o, ax] < 0, -x[o, ax], 2 * box - x[o, ax])
            v[o, ax] *= -1
    p_sim = impulse / (duration * box**2)
    t_sim = m * float(np.mean(np.sum(v**2, axis=1))) / (3 * KB)
    return p_sim, t_sim


# --------------------------------------------------------- Maxwell-Boltzmann
def mb_pdf(v, temperature: float = T_GAS, m: float = M_N2):
    v = np.asarray(v, dtype=float)
    a = m / (2 * np.pi * KB * temperature)
    return 4 * np.pi * a**1.5 * v**2 * np.exp(-m * v**2 / (2 * KB * temperature))


def speeds_analytic(temperature: float = T_GAS, m: float = M_N2):
    return {"v_mp": np.sqrt(2 * KB * temperature / m), "v_mean": np.sqrt(8 * KB * temperature / (np.pi * m)),
            "v_rms": np.sqrt(3 * KB * temperature / m)}


def speeds_sampled(n: int = 400_000, temperature: float = T_GAS, m: float = M_N2, seed: int = 8):
    rng = np.random.default_rng(seed)
    comp = rng.normal(0.0, np.sqrt(KB * temperature / m), size=(n, 3))
    v = np.linalg.norm(comp, axis=1)
    hist, edges = np.histogram(v, bins=60, range=(0, 1500), density=True)
    # moda aproximada: centro del bin mas poblado (el ancho de bin es 25 m/s)
    centers = 0.5 * (edges[1:] + edges[:-1])
    return {"v": v, "v_mean": float(v.mean()), "v_rms": float(np.sqrt(np.mean(v**2))),
            "v_mp": float(centers[np.argmax(hist)]), "hist": hist, "centers": centers}


def moments_by_quadrature(temperature: float = T_GAS, m: float = M_N2, n: int = 4000):
    v = np.linspace(0, 6000, n + 1)
    f = mb_pdf(v, temperature, m)
    return {"norm": simpson(f, v), "v_mean": simpson(v * f, v), "v_rms": float(np.sqrt(simpson(v**2 * f, v)))}
