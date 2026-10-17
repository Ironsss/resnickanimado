"""Capitulo 19. Movimiento ondulatorio en una cuerda."""
from __future__ import annotations

import numpy as np

TENSION = 80.0
MU = 0.05
V = float(np.sqrt(TENSION / MU))  # 40 m/s
LENGTH = 20.0
SIGMA = 0.6


def pulse(x, x0: float = 10.0, sigma: float = SIGMA):
    return np.exp(-((np.asarray(x, dtype=float) - x0) ** 2) / (2 * sigma**2))


def dalembert(x, t: float, v: float = V):
    """Pulso gaussiano inicialmente en reposo: se divide en dos mitades que viajan en sentidos opuestos."""
    return 0.5 * (pulse(np.asarray(x) - v * t) + pulse(np.asarray(x) + v * t))


def fdtd(n: int = 800, courant: float = 0.9, t_snap=(0.0, 0.075, 0.15), v: float = V, length: float = LENGTH):
    """Diferencias finitas centradas (leapfrog) para y_tt = v^2 y_xx con extremos fijos lejanos."""
    x = np.linspace(0, length, n + 1)
    dx = x[1] - x[0]
    dt = courant * dx / v
    y_prev = pulse(x)
    c2 = courant**2
    y = y_prev.copy()
    y[1:-1] = y_prev[1:-1] + 0.5 * c2 * (y_prev[2:] - 2 * y_prev[1:-1] + y_prev[:-2])  # primer paso con velocidad nula
    snaps = {}
    t = dt
    targets = sorted(t_snap)
    steps = {round(tt / dt): tt for tt in targets}
    if 0 in steps:
        snaps[steps[0]] = y_prev.copy()
    k = 1
    y_old, y_cur = y_prev, y
    max_k = int(round(max(targets) / dt))
    while k <= max_k:
        if k in steps:
            snaps[k] = y_cur.copy()
        y_new = np.zeros_like(y_cur)
        y_new[1:-1] = 2 * y_cur[1:-1] - y_old[1:-1] + c2 * (y_cur[2:] - 2 * y_cur[1:-1] + y_cur[:-2])
        y_old, y_cur = y_cur, y_new
        k += 1
    return x, dt, {k_ * dt: s for k_, s in snaps.items()}


def fdtd_error(courant: float = 0.9, n: int = 800, t: float = 0.15) -> float:
    x, dt, snaps = fdtd(n, courant, (0.0, t))
    tt, y = max(snaps.items())
    return float(np.max(np.abs(y - dalembert(x, tt))))


# ---------------------------------------------------------------- modos normales
STRING_L = 1.2


def mode_frequency_analytic(n: int, length: float = STRING_L, v: float = V) -> float:
    return n * v / (2 * length)


def modes_numeric(n_points: int = 400, length: float = STRING_L, v: float = V, count: int = 5):
    """Autovalores de la matriz tridiagonal de la segunda derivada con extremos fijos."""
    dx = length / (n_points + 1)
    main = np.full(n_points, 2.0)
    off = np.full(n_points - 1, -1.0)
    a = (np.diag(main) + np.diag(off, 1) + np.diag(off, -1)) / dx**2
    w2, vec = np.linalg.eigh(a)
    freqs = v * np.sqrt(w2[:count]) / (2 * np.pi)
    x = np.linspace(dx, length - dx, n_points)
    shapes = []
    for j in range(count):
        s = vec[:, j]
        shapes.append(s / np.max(np.abs(s)) * np.sign(s[np.argmax(np.abs(s))]))
    return freqs, x, shapes
