"""Volumen 2, capitulo 2. El campo electrico."""
from __future__ import annotations

import numpy as np

from .constants import K_E
from .integrators import rk4_step
from .ch07_work_energy import simpson

# ------------------------------------------------------------------------ dipolo
QD = 5e-9
DSEP = 0.02


def dipole_field_axis_exact(z, q: float = QD, d: float = DSEP):
    z = np.asarray(z, dtype=float)
    return K_E * q * (1 / (z - d / 2) ** 2 - 1 / (z + d / 2) ** 2)


def dipole_field_axis_far(z, q: float = QD, d: float = DSEP):
    return 2 * K_E * q * d / np.asarray(z, dtype=float) ** 3


def potential_dipole(x, y, q: float = QD, d: float = DSEP):
    r1 = np.hypot(x - d / 2, y)
    r2 = np.hypot(x + d / 2, y)
    return K_E * q * (1 / r1 - 1 / r2)


def field_from_potential_numeric(x: float, y: float, h: float = 1e-6, q: float = QD, d: float = DSEP):
    ex = -(potential_dipole(x + h, y, q, d) - potential_dipole(x - h, y, q, d)) / (2 * h)
    ey = -(potential_dipole(x, y + h, q, d) - potential_dipole(x, y - h, q, d)) / (2 * h)
    return np.array([ex, ey])


def field_dipole_vector(x: float, y: float, q: float = QD, d: float = DSEP):
    e = np.zeros(2)
    for sign, cx in ((+1, d / 2), (-1, -d / 2)):
        r = np.array([x - cx, y])
        e += K_E * sign * q * r / np.linalg.norm(r) ** 3
    return e


def trace_field_line(angle_deg: float, q: float = QD, d: float = DSEP, ds: float = 5e-5, max_steps: int = 20000):
    """Integra dr/ds = E/|E| con RK4 desde la carga positiva hasta llegar a la negativa (o escapar)."""
    a = np.deg2rad(angle_deg)
    start = np.array([d / 2, 0.0]) + 2e-4 * np.array([np.cos(a), np.sin(a)])
    rhs = lambda _s, r: (lambda e: e / np.linalg.norm(e))(field_dipole_vector(r[0], r[1], q, d))  # noqa: E731
    pts = [start]
    r = start
    for _ in range(max_steps):
        r = rk4_step(rhs, 0.0, r, ds)
        pts.append(r)
        if np.linalg.norm(r - np.array([-d / 2, 0.0])) < 4e-4:
            return np.array(pts), True
        if np.linalg.norm(r) > 40 * d:
            break
    return np.array(pts), False


# ---------------------------------------------------------------- anillo cargado
RING_R = 0.10
RING_Q = 10e-9


def ring_axis_field_analytic(z, q: float = RING_Q, a: float = RING_R):
    z = np.asarray(z, dtype=float)
    return K_E * q * z / (z**2 + a**2) ** 1.5


def ring_axis_field_numeric(z: float, q: float = RING_Q, a: float = RING_R, n: int = 400) -> float:
    """Suma vectorial de elementos dq del anillo (Simpson sobre el angulo); solo sobrevive la componente axial."""
    phi = np.linspace(0, 2 * np.pi, n + 1)
    dq = q / (2 * np.pi)  # por radian
    src = np.stack([a * np.cos(phi), a * np.sin(phi), np.zeros_like(phi)], axis=1)
    r = np.array([0.0, 0.0, z]) - src
    dist = np.linalg.norm(r, axis=1)
    contrib = K_E * dq * r / dist[:, None] ** 3
    ez = simpson(contrib[:, 2], phi)
    ex, ey = simpson(contrib[:, 0], phi), simpson(contrib[:, 1], phi)
    assert abs(ex) < 1e-6 * abs(ez) + 1e-12 and abs(ey) < 1e-6 * abs(ez) + 1e-12
    return float(ez)


def ring_max_field_position_analytic(a: float = RING_R) -> float:
    return a / np.sqrt(2.0)


def ring_max_field_position_numeric(a: float = RING_R) -> float:
    zs = np.linspace(0.001, 3 * a, 2000)
    e = np.array([ring_axis_field_numeric(float(z), n=200) for z in zs[::10]])
    zc = zs[::10]
    i = int(np.argmax(e))
    # refinar con ajuste parabolico alrededor del maximo
    lo, hi = max(i - 3, 0), min(i + 4, len(zc))
    c = np.polyfit(zc[lo:hi], e[lo:hi], 2)
    return float(-c[1] / (2 * c[0]))
