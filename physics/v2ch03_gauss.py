"""Volumen 2, capitulo 3. Ley de Gauss."""
from __future__ import annotations

import numpy as np

from .constants import EPS0, K_E

Q = 2e-9


def flux_sphere_analytic(q: float = Q) -> float:
    return q / EPS0


def _field(points: np.ndarray, charge_pos: np.ndarray, q: float) -> np.ndarray:
    r = points - charge_pos
    d = np.linalg.norm(r, axis=-1, keepdims=True)
    return K_E * q * r / d**3


def flux_sphere_numeric(radius: float = 0.3, q: float = Q, charge_pos=(0.0, 0.0, 0.0), n_theta: int = 200, n_phi: int = 400) -> float:
    """Integral de superficie E.n dA con la regla del punto medio en (theta, phi)."""
    th = (np.arange(n_theta) + 0.5) * np.pi / n_theta
    ph = (np.arange(n_phi) + 0.5) * 2 * np.pi / n_phi
    T, P = np.meshgrid(th, ph, indexing="ij")
    nrm = np.stack([np.sin(T) * np.cos(P), np.sin(T) * np.sin(P), np.cos(T)], axis=-1)
    pts = radius * nrm
    e = _field(pts, np.asarray(charge_pos, dtype=float), q)
    dA = radius**2 * np.sin(T) * (np.pi / n_theta) * (2 * np.pi / n_phi)
    return float(np.sum(np.sum(e * nrm, axis=-1) * dA))


def flux_cube_numeric(side: float = 0.4, q: float = Q, charge_pos=(0.0, 0.0, 0.0), n: int = 400) -> float:
    """Flujo a traves de un cubo centrado en el origen: suma de seis caras con la regla del punto medio."""
    s = (np.arange(n) + 0.5) / n * side - side / 2
    U, V = np.meshgrid(s, s, indexing="ij")
    dA = (side / n) ** 2
    total = 0.0
    cp = np.asarray(charge_pos, dtype=float)
    for axis in range(3):
        for sign in (+1, -1):
            pts = np.zeros(U.shape + (3,))
            idx = [i for i in range(3) if i != axis]
            pts[..., idx[0]], pts[..., idx[1]], pts[..., axis] = U, V, sign * side / 2
            e = _field(pts, cp, q)
            total += float(np.sum(sign * e[..., axis]) * dA)
    return total


# ---------------------------------------------------- esfera uniformemente cargada
R_SPH = 0.10
Q_SPH = 5e-9


def sphere_field_analytic(r, q: float = Q_SPH, big_r: float = R_SPH):
    r = np.asarray(r, dtype=float)
    return np.where(r < big_r, K_E * q * r / big_r**3, K_E * q / np.maximum(r, 1e-12) ** 2)


def sphere_field_monte_carlo(r: float, q: float = Q_SPH, big_r: float = R_SPH, n: int = 400_000, seed: int = 17):
    """Campo de una esfera uniforme: suma de n cargas puntuales muestreadas uniformemente en el volumen."""
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    rad = big_r * rng.uniform(size=(n, 1)) ** (1 / 3)
    src = v * rad
    obs = np.array([r, 0.0, 0.0])
    d = obs - src
    dist = np.linalg.norm(d, axis=1, keepdims=True)
    e = K_E * (q / n) * d / dist**3
    ex = e[:, 0]
    return float(ex.sum()), float(ex.std(ddof=1) * np.sqrt(n))
