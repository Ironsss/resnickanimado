"""Capitulo 7. Trabajo y energia cinetica."""
from __future__ import annotations

import numpy as np

from .integrators import integrate, rk4_step, riemann_left, trapezoid

# ------------------------------------------------------------ trabajo de un resorte
K = 200.0  # N/m
X1, X2 = 0.10, 0.30  # m


def spring_force(x, k: float = K):
    return k * np.asarray(x, dtype=float)


def spring_work_analytic(x1: float = X1, x2: float = X2, k: float = K) -> float:
    return 0.5 * k * (x2**2 - x1**2)


def simpson(y: np.ndarray, x: np.ndarray) -> float:
    n = len(x) - 1
    if n % 2:
        raise ValueError("Simpson requiere un numero par de subintervalos")
    h = (x[-1] - x[0]) / n
    return float(h / 3 * (y[0] + y[-1] + 4 * y[1:-1:2].sum() + 2 * y[2:-1:2].sum()))


def spring_work_numeric(n: int, method: str, x1: float = X1, x2: float = X2, k: float = K) -> float:
    x = np.linspace(x1, x2, n + 1)
    f = spring_force(x, k)
    return {"riemann": riemann_left, "trapezoid": trapezoid, "simpson": simpson}[method](f, x)


def work_convergence(ns=(2, 4, 8, 16, 32)):
    ex = spring_work_analytic()
    return [{"n": n, **{m: abs(spring_work_numeric(n, m) - ex) for m in ("riemann", "trapezoid", "simpson")}} for n in ns]


# ------------------------------------------- teorema trabajo-energia con fuerza variable
MASS = 2.0
F0 = 30.0
LENGTH = 2.0


def force_profile(x, f0: float = F0, length: float = LENGTH):
    return f0 * (1.0 - np.asarray(x, dtype=float) / length)


def kinetic_energy_analytic(x, f0: float = F0, length: float = LENGTH):
    x = np.asarray(x, dtype=float)
    return f0 * (x - x**2 / (2 * length))


def final_speed_analytic(m: float = MASS, f0: float = F0, length: float = LENGTH) -> float:
    return float(np.sqrt(2 * kinetic_energy_analytic(length, f0, length) / m))


def newton_numeric(m: float = MASS, f0: float = F0, length: float = LENGTH, dt: float = 1e-3):
    """Integra m x'' = F(x) con RK4 hasta x = L; devuelve (t, x, v)."""
    def rhs(_t, y):
        return np.array([y[1], force_profile(y[0], f0, length) / m])

    t, y = [0.0], [np.array([0.0, 0.0])]
    while y[-1][0] < length:
        y.append(rk4_step(rhs, t[-1], y[-1], dt))
        t.append(t[-1] + dt)
        if len(t) > 100000:
            break
    y = np.array(y)
    # refinar el cruce x = L por interpolacion lineal
    if y[-1, 0] > length:
        frac = (length - y[-2, 0]) / (y[-1, 0] - y[-2, 0])
        y[-1] = y[-2] + frac * (y[-1] - y[-2])
        t[-1] = t[-2] + frac * dt
    return np.array(t), y[:, 0], y[:, 1]


def final_speed_numeric(**kw) -> float:
    return float(newton_numeric(**kw)[2][-1])


def work_integral_numeric(n: int = 400, f0: float = F0, length: float = LENGTH) -> float:
    x = np.linspace(0, length, n + 1)
    return simpson(force_profile(x, f0, length), x)


def final_power_analytic(m: float = MASS, f0: float = F0, length: float = LENGTH) -> float:
    return 0.0  # F(L) = 0, la fuerza se anula al final


def max_power(m: float = MASS, f0: float = F0, length: float = LENGTH):
    """Potencia instantanea P = F v a lo largo del recorrido (numerica)."""
    _, x, v = newton_numeric(m, f0, length)
    p = force_profile(x, f0, length) * v
    return float(p.max()), float(x[int(np.argmax(p))])
