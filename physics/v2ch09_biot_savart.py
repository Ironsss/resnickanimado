"""Volumen 2, capitulo 9. El campo magnetico de una corriente."""
from __future__ import annotations

import numpy as np

from .constants import MU0

# ------------------------------------------------------------------------ espira
A_LOOP = 0.05
I = 3.0


def loop_axis_analytic(z, a: float = A_LOOP, current: float = I):
    z = np.asarray(z, dtype=float)
    return MU0 * current * a**2 / (2 * (a**2 + z**2) ** 1.5)


def biot_savart(points: np.ndarray, segments: np.ndarray, current: float) -> np.ndarray:
    """B en `points` (m, 3) debido a segmentos rectos (n, 2, 3) con corriente: suma de dl x r / r^3 en el punto medio."""
    mid = 0.5 * (segments[:, 0] + segments[:, 1])
    dl = segments[:, 1] - segments[:, 0]
    b = np.zeros_like(points)
    for k, p in enumerate(points):
        r = p - mid
        d = np.linalg.norm(r, axis=1, keepdims=True)
        b[k] = MU0 * current / (4 * np.pi) * np.sum(np.cross(dl, r) / d**3, axis=0)
    return b


def loop_segments(a: float = A_LOOP, n: int = 720, z0: float = 0.0) -> np.ndarray:
    phi = np.linspace(0, 2 * np.pi, n + 1)
    pts = np.stack([a * np.cos(phi), a * np.sin(phi), np.full_like(phi, z0)], axis=1)
    return np.stack([pts[:-1], pts[1:]], axis=1)


def loop_axis_numeric(z: float, a: float = A_LOOP, current: float = I, n: int = 720) -> float:
    b = biot_savart(np.array([[0.0, 0.0, z]]), loop_segments(a, n), current)
    return float(b[0, 2])


# -------------------------------------------------------------------- hilo recto finito
LEN_WIRE = 0.30
DIST = 0.04


def wire_field_analytic(length: float = LEN_WIRE, d: float = DIST, current: float = I) -> float:
    s = (length / 2) / np.hypot(length / 2, d)
    return MU0 * current / (4 * np.pi * d) * 2 * s


def wire_field_numeric(length: float = LEN_WIRE, d: float = DIST, current: float = I, n: int = 2000) -> float:
    z = np.linspace(-length / 2, length / 2, n + 1)
    seg = np.stack([np.stack([np.zeros(n), np.zeros(n), z[:-1]], axis=1), np.stack([np.zeros(n), np.zeros(n), z[1:]], axis=1)], axis=1)
    b = biot_savart(np.array([[d, 0.0, 0.0]]), seg, current)
    return float(np.linalg.norm(b[0]))


def infinite_wire(d: float = DIST, current: float = I) -> float:
    return MU0 * current / (2 * np.pi * d)


# ----------------------------------------------------------------------- solenoide
N_TURNS = 500
L_SOL = 0.5
R_SOL = 0.02
I_SOL = 2.0


def solenoid_center_infinite(n: int = N_TURNS, length: float = L_SOL, current: float = I_SOL) -> float:
    return MU0 * n / length * current


def solenoid_axis_exact(z, n: int = N_TURNS, length: float = L_SOL, r: float = R_SOL, current: float = I_SOL):
    """Solenoide finito sobre el eje: B = mu0 n I / 2 (cos a2 - cos a1) (integral analitica de las espiras)."""
    z = np.asarray(z, dtype=float)
    z1, z2 = z + length / 2, z - length / 2
    return MU0 * (n / length) * current / 2 * (z1 / np.hypot(z1, r) - z2 / np.hypot(z2, r))


def solenoid_axis_numeric(z: float, n: int = N_TURNS, length: float = L_SOL, r: float = R_SOL, current: float = I_SOL, seg: int = 60) -> float:
    """Biot-Savart sobre las N espiras (cada una con `seg` segmentos), agrupadas en bloques para vectorizar."""
    zs = np.linspace(-length / 2 + length / (2 * n), length / 2 - length / (2 * n), n)
    phi = np.linspace(0, 2 * np.pi, seg + 1)
    ring = np.stack([r * np.cos(phi), r * np.sin(phi), np.zeros_like(phi)], axis=1)
    all_pts = np.concatenate([ring[None] + np.array([0.0, 0.0, zz]) for zz in zs], axis=0)  # (n, seg+1, 3)
    segments = np.stack([all_pts[:, :-1], all_pts[:, 1:]], axis=2).reshape(-1, 2, 3)
    b = biot_savart(np.array([[0.0, 0.0, z]]), segments, current)
    return float(b[0, 2])
