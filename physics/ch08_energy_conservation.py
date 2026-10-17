"""Capitulo 8. Conservacion de la energia."""
from __future__ import annotations

import numpy as np

from .integrators import euler_step, integrate, rk4_step

G = 9.81
LENGTH = 1.5
THETA0_DEG = 60.0
MASS = 1.0


def bottom_speed_analytic(theta0_deg: float = THETA0_DEG, length: float = LENGTH, g: float = G) -> float:
    return float(np.sqrt(2 * g * length * (1 - np.cos(np.deg2rad(theta0_deg)))))


def pendulum_rhs(length: float = LENGTH, g: float = G):
    return lambda _t, y: np.array([y[1], -(g / length) * np.sin(y[0])])


def energy(theta, omega, length: float = LENGTH, g: float = G, m: float = MASS):
    k = 0.5 * m * (length * omega) ** 2
    u = m * g * length * (1 - np.cos(theta))
    return k, u


def pendulum_run(method: str, t_end: float = 6.0, n: int = 600, theta0_deg: float = THETA0_DEG,
                 length: float = LENGTH, g: float = G):
    y0 = np.array([np.deg2rad(theta0_deg), 0.0])
    if method == "verlet":
        h = t_end / n
        t = np.linspace(0, t_end, n + 1)
        th = np.empty(n + 1)
        om = np.empty(n + 1)
        th[0], om[0] = y0
        acc = lambda x: -(g / length) * np.sin(x)  # noqa: E731
        for i in range(n):
            a = acc(th[i])
            th_new = th[i] + om[i] * h + 0.5 * a * h * h
            om[i + 1] = om[i] + 0.5 * (a + acc(th_new)) * h
            th[i + 1] = th_new
        return t, np.column_stack([th, om])
    step = rk4_step if method == "rk4" else euler_step
    return integrate(pendulum_rhs(length, g), y0, 0.0, t_end, n, step)


def energy_drift(method: str, **kw) -> float:
    t, y = pendulum_run(method, **kw)
    k, u = energy(y[:, 0], y[:, 1])
    e = k + u
    return float(np.max(np.abs(e - e[0])) / e[0])


def max_speed_numeric(**kw) -> float:
    t, y = pendulum_run("rk4", n=6000, **kw)
    return float(np.max(np.abs(y[:, 1])) * LENGTH)


# ------------------------------------- resorte con friccion (fuerza no conservativa)
KS = 100.0
M = 1.0
MU = 0.25
X0 = 0.20


def delta(mu: float = MU, m: float = M, k: float = KS, g: float = G) -> float:
    """Reduccion de amplitud por semiciclo = 2*delta; delta = mu m g / k."""
    return mu * m * g / k


def turning_points_analytic(x0: float = X0, mu: float = MU, m: float = M, k: float = KS, g: float = G):
    d = delta(mu, m, k, g)
    pts = [x0]
    x = x0
    sign = -1.0
    while abs(x) > d:
        x = abs(x) - 2 * d
        pts.append(sign * x)
        sign = -sign
        if x <= d:
            break
    return pts


def spring_friction_numeric(x0: float = X0, mu: float = MU, m: float = M, k: float = KS, g: float = G,
                            dt: float = 1e-4, t_max: float = 5.0):
    """RK4 con friccion cinetica -mu m g sgn(v) y parada cuando |kx| <= mu m g en reposo."""
    t, x, v = 0.0, x0, 0.0
    ts, xs, vs = [0.0], [x0], [0.0]
    turning = [x0]
    prev_v = 0.0
    direction = 0.0
    while t < t_max:
        if v == 0.0 and abs(k * x) <= mu * m * g:
            break
        if v == 0.0:
            direction = -np.sign(x)
        def rhs(_t, y, d=direction):
            return np.array([y[1], (-k * y[0] - d * mu * m * g) / m]) if d != 0 else np.array([y[1], -k * y[0] / m])
        y_new = rk4_step(rhs, t, np.array([x, v]), dt)
        # el sentido del movimiento no debe invertirse dentro del paso
        if v != 0.0 and np.sign(y_new[1]) != np.sign(v):
            frac = v / (v - y_new[1])
            x_turn = x + frac * (y_new[0] - x)
            t += frac * dt
            x, v = x_turn, 0.0
            turning.append(x)
            ts.append(t); xs.append(x); vs.append(0.0)
            continue
        x, v = y_new[0], y_new[1]
        t += dt
        ts.append(t); xs.append(x); vs.append(v)
    return np.array(ts), np.array(xs), np.array(vs), turning
