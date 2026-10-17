"""Volumen 2, capitulo 25. Particulas elementales y cosmologia."""
from __future__ import annotations

import numpy as np

from .integrators import integrate, rk4_step
from .ch07_work_energy import simpson

# --------------------------------------------------------------- decaimiento a dos cuerpos (MeV, c = 1)
M_PARENT, M_1, M_2 = 1115.683, 938.272, 139.570  # Lambda -> p + pi-


def p_star_analytic(m: float = M_PARENT, m1: float = M_1, m2: float = M_2) -> float:
    return float(np.sqrt((m**2 - (m1 + m2) ** 2) * (m**2 - (m1 - m2) ** 2)) / (2 * m))


def p_star_numeric(m: float = M_PARENT, m1: float = M_1, m2: float = M_2) -> float:
    """Biseccion de E1(p) + E2(p) = M (conservacion de energia en el sistema en reposo)."""
    f = lambda p: np.hypot(p, m1) + np.hypot(p, m2) - m  # noqa: E731
    lo, hi = 0.0, m
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) < 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def decay_monte_carlo(n: int = 100_000, p_lab: float = 2000.0, seed: int = 5, m: float = M_PARENT, m1: float = M_1, m2: float = M_2):
    """Direccion isotropa en el sistema en reposo; impulso al sistema del laboratorio con un boost a lo largo de z.
    Devuelve la masa invariante reconstruida (media y dispersion) y el impulso maximo del protón en el laboratorio."""
    rng = np.random.default_rng(seed)
    ps = p_star_analytic(m, m1, m2)
    cost = rng.uniform(-1, 1, n)
    sint = np.sqrt(1 - cost**2)
    phi = rng.uniform(0, 2 * np.pi, n)
    p1 = ps * np.stack([sint * np.cos(phi), sint * np.sin(phi), cost], axis=1)
    e1 = np.hypot(ps, m1)
    e2 = np.hypot(ps, m2)
    gamma = np.hypot(p_lab, m) / m
    beta = p_lab / np.hypot(p_lab, m)
    boost = lambda e, pz: (gamma * (e + beta * pz), gamma * (pz + beta * e))  # noqa: E731
    e1l, p1z = boost(e1, p1[:, 2])
    e2l, p2z = boost(e2, -p1[:, 2])
    px, py = p1[:, 0], p1[:, 1]
    et = e1l + e2l
    pzt = p1z + p2z
    ptx = px + (-px)
    inv = np.sqrt(et**2 - pzt**2 - ptx**2)
    p1mag = np.sqrt(px**2 + py**2 + p1z**2)
    return float(inv.mean()), float(inv.std()), float(p1mag.max()), float(p1mag.min())


def lab_momentum_limits_analytic(p_lab: float = 2000.0, m: float = M_PARENT, m1: float = M_1, m2: float = M_2):
    """Extremos del impulso del protón en el laboratorio: gamma(p* cos +- beta E*) con cos = +-1."""
    ps = p_star_analytic(m, m1, m2)
    e1 = np.hypot(ps, m1)
    gamma = np.hypot(p_lab, m) / m
    beta = p_lab / np.hypot(p_lab, m)
    return float(gamma * (ps + beta * e1)), float(abs(gamma * (-ps + beta * e1)))


# ---------------------------------------------------------------------------- cosmologia plana
H0_PER_GYR = 0.0703  # 68.7 km/s/Mpc en 1/Gyr
OMEGA_M, OMEGA_L = 0.31, 0.69


def age_analytic(h0: float = H0_PER_GYR, om: float = OMEGA_M, ol: float = OMEGA_L) -> float:
    return float(2 / (3 * h0 * np.sqrt(ol)) * np.arcsinh(np.sqrt(ol / om)))


def age_numeric(h0: float = H0_PER_GYR, om: float = OMEGA_M, ol: float = OMEGA_L, n: int = 20000) -> float:
    """t0 = integral de da / (a H(a)) de 0 a 1; el cambio a = u^2 elimina la singularidad en a = 0."""
    u = np.linspace(0.0, 1.0, n + 1)
    a = u**2
    with np.errstate(divide="ignore", invalid="ignore"):
        integrand = np.where(u > 0, 2 * u / (a * h0 * np.sqrt(om / a**3 + ol)), 0.0)
    integrand[0] = 0.0
    return simpson(integrand, u) / 1.0


def scale_factor_analytic(t, h0: float = H0_PER_GYR, om: float = OMEGA_M, ol: float = OMEGA_L):
    t = np.asarray(t, dtype=float)
    return (om / ol) ** (1 / 3) * np.sinh(1.5 * np.sqrt(ol) * h0 * t) ** (2 / 3)


def scale_factor_numeric(t_end: float = 25.0, n: int = 5000, h0: float = H0_PER_GYR, om: float = OMEGA_M, ol: float = OMEGA_L):
    """RK4 de da/dt = H0 a sqrt(om/a^3 + ol), arrancando con la solucion de materia a un instante pequeno."""
    t0 = 1e-3
    a0 = float(scale_factor_analytic(t0, h0, om, ol))
    rhs = lambda _t, y: np.array([h0 * np.sqrt(om / y[0] + ol * y[0] ** 2)])  # noqa: E731
    return integrate(rhs, np.array([a0]), t0, t_end, n, rk4_step)


def lab_energy_samples(n: int = 200_000, p_lab: float = 2000.0, seed: int = 5, m: float = M_PARENT, m1: float = M_1, m2: float = M_2) -> np.ndarray:
    """Energia del proton en el laboratorio para decaimientos isotropos en el sistema en reposo."""
    rng = np.random.default_rng(seed)
    ps = p_star_analytic(m, m1, m2)
    e1 = np.hypot(ps, m1)
    gamma = np.hypot(p_lab, m) / m
    beta = p_lab / np.hypot(p_lab, m)
    cost = rng.uniform(-1, 1, n)
    return gamma * (e1 + beta * ps * cost)


def lab_energy_range_analytic(p_lab: float = 2000.0, m: float = M_PARENT, m1: float = M_1, m2: float = M_2):
    ps = p_star_analytic(m, m1, m2)
    e1 = np.hypot(ps, m1)
    gamma = np.hypot(p_lab, m) / m
    beta = p_lab / np.hypot(p_lab, m)
    return float(gamma * (e1 - beta * ps)), float(gamma * (e1 + beta * ps))
