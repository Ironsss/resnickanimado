"""Capitulo 1. Medicion: analisis dimensional y propagacion de incertidumbre."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .integrators import integrate, rk4_step

# Dimensiones base como vectores de exponentes (M, L, T).
DIM = {
    "m": np.array([1.0, 0.0, 0.0]),
    "L": np.array([0.0, 1.0, 0.0]),
    "g": np.array([0.0, 1.0, -2.0]),
    "T": np.array([0.0, 0.0, 1.0]),
}


@dataclass(frozen=True)
class DimensionalResult:
    exponents: dict[str, float]
    residual: float
    k_numeric: float
    k_analytic: float


def pendulum_exponents() -> dict[str, float]:
    """Resuelve T = L^a g^b m^c exigiendo coherencia dimensional.

    El sistema A x = d se resuelve por minimos cuadrados (SVD). Las filas de A
    son las dimensiones M, L, T y las columnas las magnitudes L, g, m.
    """
    a = np.column_stack([DIM["L"], DIM["g"], DIM["m"]])
    x, *_ = np.linalg.lstsq(a, DIM["T"], rcond=None)
    return {"L": float(x[0]), "g": float(x[1]), "m": float(x[2])}


def pendulum_period_numeric(theta0: float, length: float = 1.0, g: float = 9.81, n: int = 4000) -> float:
    """Periodo del pendulo por integracion RK4 de theta'' = -(g/L) sin(theta).

    El periodo se obtiene del primer cruce por cero (cuarto de periodo) con
    interpolacion lineal del instante de cruce.
    """
    def rhs(_t, y):
        return np.array([y[1], -(g / length) * np.sin(y[0])])

    t_max = 2.0 * np.pi * np.sqrt(length / g)  # cota superior holgada del cuarto de periodo
    t, y = integrate(rhs, np.array([theta0, 0.0]), 0.0, t_max, n, rk4_step)
    idx = int(np.argmax(y[:, 0] <= 0.0))
    t_a, t_b = t[idx - 1], t[idx]
    th_a, th_b = y[idx - 1, 0], y[idx, 0]
    t_cross = t_a + (0.0 - th_a) * (t_b - t_a) / (th_b - th_a)
    return 4.0 * float(t_cross)


def pendulum_dimensionless_constant(theta0: float = 0.01) -> DimensionalResult:
    exps = pendulum_exponents()
    a = np.column_stack([DIM["L"], DIM["g"], DIM["m"]])
    x = np.array([exps["L"], exps["g"], exps["m"]])
    resid = float(np.linalg.norm(a @ x - DIM["T"]))
    length, g = 1.0, 9.81
    period = pendulum_period_numeric(theta0, length, g)
    k_num = period / (length ** exps["L"] * g ** exps["g"])
    return DimensionalResult(exps, resid, k_num, 2.0 * np.pi)


# ---------------------------------------------------------------- incertidumbre

@dataclass(frozen=True)
class Measurement:
    value: float
    sigma: float
    unit: str


CYLINDER = {
    "m": Measurement(111.4, 0.1, "g"),
    "r": Measurement(1.250, 0.005, "cm"),
    "h": Measurement(8.40, 0.02, "cm"),
}


def density(m: float, r: float, h: float) -> float:
    return m / (np.pi * r**2 * h)


def density_analytic() -> dict[str, float]:
    """Densidad y su incertidumbre por propagacion de primer orden.

    rho = m / (pi r^2 h)  =>  (s_rho/rho)^2 = (s_m/m)^2 + (2 s_r/r)^2 + (s_h/h)^2
    """
    m, r, h = CYLINDER["m"], CYLINDER["r"], CYLINDER["h"]
    rho = density(m.value, r.value, h.value)
    rel_terms = np.array([m.sigma / m.value, 2 * r.sigma / r.value, h.sigma / h.value])
    rel_quad = float(np.sqrt(np.sum(rel_terms**2)))
    rel_worst = float(np.sum(rel_terms))
    return {
        "rho": float(rho),
        "sigma_quadrature": float(rho * rel_quad),
        "sigma_worst_case": float(rho * rel_worst),
        "rel_m": float(rel_terms[0]),
        "rel_r": float(rel_terms[1]),
        "rel_h": float(rel_terms[2]),
    }


def density_monte_carlo(n: int = 400_000, seed: int = 7) -> dict[str, float | np.ndarray]:
    rng = np.random.default_rng(seed)
    m = rng.normal(CYLINDER["m"].value, CYLINDER["m"].sigma, n)
    r = rng.normal(CYLINDER["r"].value, CYLINDER["r"].sigma, n)
    h = rng.normal(CYLINDER["h"].value, CYLINDER["h"].sigma, n)
    rho = density(m, r, h)
    return {"mean": float(rho.mean()), "std": float(rho.std(ddof=1)), "samples": rho}
