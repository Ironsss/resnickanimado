"""Volumen 2, capitulo 5. Capacitancia y dielectricos."""
from __future__ import annotations

import numpy as np

from .constants import EPS0

# ----------------------------------------------- capacitor coaxial por relajacion de Laplace
A_IN = 0.005
B_OUT = 0.02


def coax_capacitance_per_length_analytic(a: float = A_IN, b: float = B_OUT) -> float:
    return float(2 * np.pi * EPS0 / np.log(b / a))


def coax_laplace_numeric(n: int = 161, iterations: int = 6000, a: float = A_IN, b: float = B_OUT, v0: float = 1.0):
    """Resuelve Laplace en el plano con SOR rojo-negro (malla cuadrada) con el conductor interno a v0 y el externo a 0."""
    xs = np.linspace(-b * 1.02, b * 1.02, n)
    h = xs[1] - xs[0]
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    R = np.hypot(X, Y)
    inner = R <= a
    outer = R >= b
    fixed = inner | outer
    V = np.where(inner, v0, 0.0)
    omega = 2.0 / (1.0 + np.sin(np.pi / n))
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    red = ((ii + jj) % 2 == 0) & ~fixed
    black = ((ii + jj) % 2 == 1) & ~fixed
    for _ in range(iterations):
        for mask in (red, black):
            avg = 0.25 * (np.roll(V, 1, 0) + np.roll(V, -1, 0) + np.roll(V, 1, 1) + np.roll(V, -1, 1))
            V = np.where(mask, V + omega * (avg - V), V)
    return xs, V, h, fixed


def coax_capacitance_numeric(**kw) -> float:
    """C por unidad de longitud a partir de la energia: U = 1/2 eps0 int |E|^2 dA = 1/2 C V0^2."""
    xs, V, h, fixed = coax_laplace_numeric(**kw)
    gx, gy = np.gradient(V, h, h)
    e2 = gx**2 + gy**2
    region = ~fixed
    u = 0.5 * EPS0 * np.sum(e2[region]) * h * h
    return float(2 * u / 1.0**2)


# ------------------------------------------------------------------ red de capacitores
V_SRC = 12.0
C1, C2, C3 = 2e-6, 3e-6, 6e-6


def network_ceq_analytic(c1: float = C1, c2: float = C2, c3: float = C3) -> float:
    """C1 en serie con (C2 en paralelo con C3)."""
    return float(1.0 / (1.0 / c1 + 1.0 / (c2 + c3)))


def network_nodal_numeric(v: float = V_SRC, c1: float = C1, c2: float = C2, c3: float = C3):
    """Potencial del nodo intermedio por conservacion de carga: c1 (v - vx) = (c2 + c3) vx."""
    a = np.array([[c1 + c2 + c3]])
    b = np.array([c1 * v])
    vx = float(np.linalg.solve(a, b)[0])
    q1 = c1 * (v - vx)
    return {"vx": vx, "q": q1, "q2": c2 * vx, "q3": c3 * vx, "c_eq": q1 / v, "energy": 0.5 * q1 * v}


def dielectric_energies(kappa: float = 4.0, c0: float = 2e-6, v: float = 12.0):
    """Insercion del dielectrico: a V constante (bateria conectada) y a Q constante (aislado)."""
    u0 = 0.5 * c0 * v**2
    u_v = 0.5 * kappa * c0 * v**2
    q0 = c0 * v
    u_q = q0**2 / (2 * kappa * c0)
    return {"u0": u0, "u_const_v": u_v, "u_const_q": u_q}
