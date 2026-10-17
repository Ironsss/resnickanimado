"""Volumen 2, capitulo 16. Optica geometrica: espejos y lentes."""
from __future__ import annotations

import numpy as np

N_GLASS = 1.5
R1, R2 = 0.10, -0.10


def focal_lensmaker(n: float = N_GLASS, r1: float = R1, r2: float = R2) -> float:
    return 1.0 / ((n - 1) * (1 / r1 - 1 / r2))


def image_distance_analytic(s: float, f: float) -> float:
    return 1.0 / (1.0 / f - 1.0 / s)


def _refract(p, d, center_x, radius, n_from, n_to):
    """Refraccion exacta (vectorial) de un rayo 2D en un arco esferico con centro (center_x, 0)."""
    c = np.array([center_x, 0.0])
    oc = p - c
    b = oc @ d
    disc = b * b - (oc @ oc - radius**2)
    t_candidates = [-b - np.sqrt(disc), -b + np.sqrt(disc)]
    t = min(t for t in t_candidates if t > 1e-12) if radius > 0 else max(t_candidates)
    hit = p + t * d
    nrm = (hit - c) / abs(radius)
    if nrm @ d > 0:
        nrm = -nrm
    cos_i = -nrm @ d
    ratio = n_from / n_to
    k = 1 - ratio**2 * (1 - cos_i**2)
    return hit, ratio * d + (ratio * cos_i - np.sqrt(k)) * nrm


def lens_image_numeric(s: float, h: float = 1e-3, n: float = N_GLASS, r1: float = R1, r2: float = R2, thickness: float = 2e-5) -> float:
    """Traza un rayo paraxial desde un objeto axial a distancia s por las dos superficies esfericas (Snell exacto) y halla donde corta el eje."""
    p = np.array([-s, 0.0])
    d = np.array([s, h]) / np.hypot(s, h)
    # superficie 1: vertice en x=0, centro en r1; superficie 2: vertice en x=thickness, centro en thickness + r2
    c1 = r1
    hit1, d1 = _refract_conv(p, d, c1, r1, 1.0, n)
    c2 = thickness + r2
    hit2, d2 = _refract_conv(hit1, d1, c2, r2, n, 1.0)
    t = -hit2[1] / d2[1]
    return float(hit2[0] + t * d2[0])


def _refract_conv(p, d, center_x, radius, n_from, n_to):
    """Interseccion con un arco esferico de radio con signo (centro a la derecha si radius>0) y refraccion vectorial."""
    c = np.array([center_x, 0.0])
    oc = p - c
    b = oc @ d
    disc = b * b - (oc @ oc - radius**2)
    ts = [-b - np.sqrt(disc), -b + np.sqrt(disc)]
    # la interseccion buscada es la mas cercana al vertice (x = center_x - radius)
    vx = center_x - radius
    hits = [p + t * d for t in ts if t > 0]
    hit = min(hits, key=lambda q: abs(q[0] - vx))
    nrm = (hit - c) / abs(radius)
    if nrm @ d > 0:
        nrm = -nrm
    cos_i = -(nrm @ d)
    ratio = n_from / n_to
    k = 1 - ratio**2 * (1 - cos_i**2)
    return hit, ratio * d + (ratio * cos_i - np.sqrt(k)) * nrm


def mirror_image_analytic(s: float, radius: float) -> float:
    return image_distance_analytic(s, radius / 2)


def mirror_image_numeric(s: float, radius: float, h: float = 1e-3) -> float:
    """Reflexion exacta de un rayo paraxial en un espejo concavo (centro a la izquierda del vertice)."""
    c = np.array([-radius, 0.0])
    p = np.array([-s, 0.0])
    d = np.array([s, h]) / np.hypot(s, h)
    oc = p - c
    b = oc @ d
    disc = b * b - (oc @ oc - radius**2)
    t = -b + np.sqrt(disc)
    hit = p + t * d
    nrm = (hit - c) / radius
    d2 = d - 2 * (d @ nrm) * nrm
    t2 = -hit[1] / d2[1]
    return float(-(hit[0] + t2 * d2[0]))


# ------------------------------------------------------------------ dos lentes (matrices ABCD)
F1, F2, SEP = 0.20, 0.10, 0.15


def two_lens_analytic(s: float = 0.5, f1: float = F1, f2: float = F2, d: float = SEP):
    s1p = image_distance_analytic(s, f1)
    s2 = d - s1p
    s2p = image_distance_analytic(s2, f2)
    return s2p, (-s1p / s) * (-s2p / s2)


def thin_lens_matrix(f: float) -> np.ndarray:
    return np.array([[1.0, 0.0], [-1.0 / f, 1.0]])


def free_matrix(d: float) -> np.ndarray:
    return np.array([[1.0, d], [0.0, 1.0]])


def two_lens_matrix(s: float = 0.5, f1: float = F1, f2: float = F2, d: float = SEP):
    """Imagen a partir de la matriz del sistema objeto -> L1 -> L2 -> plano imagen: B = 0 define el plano conjugado."""
    sys_ = thin_lens_matrix(f2) @ free_matrix(d) @ thin_lens_matrix(f1) @ free_matrix(s)
    # buscamos z tal que el elemento B de free(z) @ sys_ sea 0
    a, b, c, dd = sys_[0, 0], sys_[0, 1], sys_[1, 0], sys_[1, 1]
    z = -b / dd
    m = a + z * c
    return float(z), float(m)
