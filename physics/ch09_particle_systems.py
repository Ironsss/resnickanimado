"""Capitulo 9. Sistemas de particulas y centro de masa."""
from __future__ import annotations

import numpy as np

from .integrators import rk4_step
from .ch07_work_energy import simpson

G = 9.81
MASSES = np.array([2.0, 3.0, 5.0])
POSITIONS = np.array([[0.0, 0.0], [4.0, 0.0], [2.0, 3.0]])


def cm_discrete_analytic(masses=MASSES, pos=POSITIONS) -> np.ndarray:
    return (masses[:, None] * pos).sum(axis=0) / masses.sum()


def cm_discrete_numeric(masses=MASSES, pos=POSITIONS) -> np.ndarray:
    # Promedio ponderado por componentes, con producto matricial
    return (masses @ pos) / masses.sum()


ROD_L = 2.0


def rod_cm_analytic(length: float = ROD_L) -> float:
    """Barra con densidad lineal lambda(x) = lambda0 x / L: x_cm = 2L/3."""
    return 2.0 * length / 3.0


def rod_cm_numeric(n: int = 200, length: float = ROD_L) -> float:
    x = np.linspace(0.0, length, n + 1)
    lam = x / length
    return simpson(x * lam, x) / simpson(lam, x)


def rod_cm_monte_carlo(n: int = 400_000, length: float = ROD_L, seed: int = 5):
    """Muestrea x con densidad proporcional a x (transformada inversa: x = L sqrt(u))."""
    rng = np.random.default_rng(seed)
    x = length * np.sqrt(rng.uniform(size=n))
    return float(x.mean()), x


# ---------------------------------------------- explosion de un proyectil en el apice
V0 = 20.0
ANGLE_DEG = 60.0
M_A, M_B = 1.0, 3.0


def apex_state(v0: float = V0, angle_deg: float = ANGLE_DEG, g: float = G):
    th = np.deg2rad(angle_deg)
    vx, vy = v0 * np.cos(th), v0 * np.sin(th)
    t_apex = vy / g
    return {"t": t_apex, "x": vx * t_apex, "y": vy * t_apex - 0.5 * g * t_apex**2, "vx": vx}


def range_analytic(v0: float = V0, angle_deg: float = ANGLE_DEG, g: float = G) -> float:
    return v0**2 * np.sin(2 * np.deg2rad(angle_deg)) / g


def fragments_analytic(v0: float = V0, angle_deg: float = ANGLE_DEG, g: float = G):
    """Fragmento A cae verticalmente desde el reposo; B conserva el momento horizontal."""
    ap = apex_state(v0, angle_deg, g)
    vb = (M_A + M_B) * ap["vx"] / M_B
    t_fall = np.sqrt(2 * ap["y"] / g)
    xa = ap["x"]
    xb = ap["x"] + vb * t_fall
    x_cm = (M_A * xa + M_B * xb) / (M_A + M_B)
    return {"vb": vb, "t_fall": t_fall, "xa": xa, "xb": xb, "x_cm": x_cm}


def fragments_numeric(v0: float = V0, angle_deg: float = ANGLE_DEG, g: float = G, dt: float = 1e-3):
    """Integra ambos fragmentos con RK4 desde el apice y reconstruye el centro de masa."""
    ap = apex_state(v0, angle_deg, g)
    vb = (M_A + M_B) * ap["vx"] / M_B

    def fall(state):
        rhs = lambda _t, s: np.array([s[2], s[3], 0.0, -g])  # noqa: E731
        ts, ys = [0.0], [state]
        while ys[-1][1] > 0:
            ys.append(rk4_step(rhs, ts[-1], ys[-1], dt))
            ts.append(ts[-1] + dt)
        ys = np.array(ys)
        frac = ys[-2, 1] / (ys[-2, 1] - ys[-1, 1])
        ys[-1] = ys[-2] + frac * (ys[-1] - ys[-2])
        ts[-1] = ts[-2] + frac * dt
        return np.array(ts), ys

    ta, ya = fall(np.array([ap["x"], ap["y"], 0.0, 0.0]))
    tb, yb = fall(np.array([ap["x"], ap["y"], vb, 0.0]))
    return {"A": (ta, ya), "B": (tb, yb), "xa": ya[-1, 0], "xb": yb[-1, 0],
            "x_cm": (M_A * ya[-1, 0] + M_B * yb[-1, 0]) / (M_A + M_B)}
