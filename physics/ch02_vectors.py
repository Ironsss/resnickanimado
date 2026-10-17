"""Capitulo 2. Vectores."""
from __future__ import annotations

import numpy as np


def polar(magnitude: float, angle_deg: float) -> np.ndarray:
    a = np.deg2rad(angle_deg)
    return magnitude * np.array([np.cos(a), np.sin(a)])


# Dos desplazamientos consecutivos [m].
D1 = polar(40.0, 30.0)
D2 = polar(25.0, 120.0)


def resultant_components(vectors=(D1, D2)) -> np.ndarray:
    return np.sum(np.array(vectors), axis=0)


def resultant_analytic() -> dict[str, float]:
    """Resultante por la ley de los cosenos y la ley de los senos.

    Para dos vectores con angulo entre ellos gamma = 90 grados:
    |R|^2 = a^2 + b^2 + 2ab cos(gamma).
    """
    a, b = 40.0, 25.0
    gamma = np.deg2rad(120.0 - 30.0)
    mag = float(np.sqrt(a**2 + b**2 + 2 * a * b * np.cos(gamma)))
    # Angulo respecto a D1 por la ley de los senos, luego se suma el de D1.
    delta = float(np.arcsin(b * np.sin(gamma) / mag))
    return {"magnitude": mag, "angle_deg": float(np.rad2deg(delta) + 30.0)}


def resultant_numeric() -> dict[str, float]:
    r = resultant_components()
    return {
        "x": float(r[0]),
        "y": float(r[1]),
        "magnitude": float(np.hypot(*r)),
        "angle_deg": float(np.rad2deg(np.arctan2(r[1], r[0]))),
    }


# ---------------------------------------------------------------- productos
A = np.array([4.0, 1.0])
B = np.array([2.0, 3.0])


def dot(a=A, b=B) -> float:
    return float(np.dot(a, b))


def cross_z(a=A, b=B) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def angle_between_deg(a=A, b=B) -> float:
    c = dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.rad2deg(np.arccos(c)))


def products_analytic(a=A, b=B) -> dict[str, float]:
    ma, mb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    th = np.deg2rad(angle_between_deg(a, b))
    return {
        "dot_by_angle": ma * mb * float(np.cos(th)),
        "cross_by_angle": ma * mb * float(np.sin(th)),
    }


def parallelogram_area_monte_carlo(a=A, b=B, n: int = 400_000, seed: int = 11):
    """Area del paralelogramo contando puntos aleatorios en su caja envolvente."""
    rng = np.random.default_rng(seed)
    corners = np.array([[0, 0], a, b, a + b], dtype=float)
    lo, hi = corners.min(axis=0), corners.max(axis=0)
    pts = rng.uniform(lo, hi, size=(n, 2))
    m = np.column_stack([a, b])  # p = s a + t b
    st = np.linalg.solve(m, pts.T).T
    inside = np.all((st >= 0) & (st <= 1), axis=1)
    box = float(np.prod(hi - lo))
    return box * inside.mean(), pts, inside


def rotate_frame(v: np.ndarray, phi: float) -> np.ndarray:
    """Componentes de v en ejes rotados un angulo phi."""
    c, s = np.cos(phi), np.sin(phi)
    return np.array([c * v[0] + s * v[1], -s * v[0] + c * v[1]])


def invariance_sweep(a=A, b=B, n: int = 361):
    phis = np.linspace(0.0, 2 * np.pi, n)
    ra = np.array([rotate_frame(a, p) for p in phis])
    rb = np.array([rotate_frame(b, p) for p in phis])
    dots = np.sum(ra * rb, axis=1)
    return phis, ra, rb, dots
