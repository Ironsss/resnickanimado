"""Capitulo 10. Colisiones."""
from __future__ import annotations

import numpy as np

from .integrators import integrate, rk4_step

G = 9.81
M1, M2 = 2.0, 1.0
V1 = 3.0


def outcome_analytic(e: float, m1: float = M1, m2: float = M2, v1: float = V1):
    """Colision unidimensional con coeficiente de restitucion e (m2 en reposo)."""
    v1p = (m1 - e * m2) / (m1 + m2) * v1
    v2p = (1 + e) * m1 / (m1 + m2) * v1
    return v1p, v2p


def restitution_from_damping(zeta: float) -> float:
    zeta = float(zeta)
    return float(np.exp(-np.pi * zeta / np.sqrt(1 - zeta**2)))


def contact_simulation(zeta: float, m1: float = M1, m2: float = M2, v1: float = V1,
                       k: float = 1.0e4, dt: float = 1e-5):
    """Dos particulas con contacto resorte-amortiguador lineal. Devuelve velocidades finales."""
    meff = m1 * m2 / (m1 + m2)
    c = 2 * zeta * np.sqrt(k * meff)

    def rhs(_t, s):
        x1, x2, u1, u2 = s
        overlap = x1 - x2 + 0.0  # contacto cuando x1 > x2 (posiciones de los centros, radio nulo)
        if overlap <= 0.0:
            return np.array([u1, u2, 0.0, 0.0])
        f = k * overlap + c * (u1 - u2)  # el contacto termina cuando el solape vuelve a cero
        return np.array([u1, u2, -f / m1, f / m2])

    s = np.array([-0.02, 0.0, v1, 0.0])
    t = 0.0
    touched = False
    while True:
        s = rk4_step(rhs, t, s, dt)
        t += dt
        if s[0] - s[1] > 0:
            touched = True
        if touched and s[0] - s[1] <= 0:
            break
        if t > 1.0:
            break
    return float(s[2]), float(s[3])


def restitution_measured(zeta: float, **kw) -> float:
    v1p, v2p = contact_simulation(zeta, **kw)
    return (v2p - v1p) / V1


# --------------------------------------------------------------- pendulo balistico
BULLET = 0.020
BLOCK = 2.0
BULLET_V = 250.0
ROPE = 1.2


def bullet_block_analytic(m: float = BULLET, M: float = BLOCK, v: float = BULLET_V, g: float = G):
    V = m * v / (m + M)
    h = V**2 / (2 * g)
    return {"V": V, "h": h, "loss_fraction": M / (m + M), "v_from_h": (m + M) / m * np.sqrt(2 * g * h)}


def swing_numeric(V: float, length: float = ROPE, g: float = G, dt: float = 1e-4):
    """Pendulo rigido con velocidad inicial V en el punto mas bajo: devuelve altura maxima."""
    rhs = lambda _t, y: np.array([y[1], -(g / length) * np.sin(y[0])])  # noqa: E731
    y = np.array([0.0, V / length])
    t, hmax = 0.0, 0.0
    while t < 3.0:
        y = rk4_step(rhs, t, y, dt)
        t += dt
        hmax = max(hmax, length * (1 - np.cos(y[0])))
        if y[1] < 0:
            break
    return hmax


def bullet_block_numeric(m: float = BULLET, M: float = BLOCK, v: float = BULLET_V, g: float = G):
    V = m * v / (m + M)  # conservacion del momento (colision perfectamente inelastica)
    h = swing_numeric(V, g=g)
    k_before = 0.5 * m * v**2
    k_after = 0.5 * (m + M) * V**2
    return {"V": V, "h": h, "loss_fraction": 1 - k_after / k_before}


def contact_history(zeta: float, m1: float = M1, m2: float = M2, v1: float = V1, k: float = 1.0e4, dt: float = 2e-5):
    """Velocidades v1(t), v2(t) durante el contacto (para graficar)."""
    meff = m1 * m2 / (m1 + m2)
    c = 2 * zeta * np.sqrt(k * meff)

    def rhs(_t, s):
        x1, x2, u1, u2 = s
        overlap = x1 - x2
        if overlap <= 0.0:
            return np.array([u1, u2, 0.0, 0.0])
        f = k * overlap + c * (u1 - u2)
        return np.array([u1, u2, -f / m1, f / m2])

    s = np.array([-1e-4, 0.0, v1, 0.0])
    t = 0.0
    ts, us = [0.0], [(v1, 0.0)]
    touched = False
    while t < 0.2:
        s = rk4_step(rhs, t, s, dt)
        t += dt
        ts.append(t)
        us.append((s[2], s[3]))
        if s[0] - s[1] > 0:
            touched = True
        if touched and s[0] - s[1] <= 0:
            break
    return np.array(ts), np.array(us)


def swing_history(V: float, length: float = ROPE, g: float = G, dt: float = 1e-3):
    """theta(t) del pendulo desde el punto mas bajo con velocidad V hasta el primer maximo."""
    rhs = lambda _t, y: np.array([y[1], -(g / length) * np.sin(y[0])])  # noqa: E731
    y = np.array([0.0, V / length])
    t = 0.0
    ts, th = [0.0], [0.0]
    while t < 3.0:
        y = rk4_step(rhs, t, y, dt)
        t += dt
        ts.append(t)
        th.append(y[0])
        if y[1] < 0:
            break
    return np.array(ts), np.array(th)
