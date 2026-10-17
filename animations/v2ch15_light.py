"""Volumen 2, capitulo 15. Naturaleza y propagacion de la luz."""
from __future__ import annotations

import numpy as np

from animations.common import ACCENT, ANALYTIC, NUMERIC, PhysicsScene, curve_scene
from physics import v2ch15_light as m15

CT = "Naturaleza y propagación de la luz"


class SnellFermat(PhysicsScene):
    volume = 2
    chapter_number = 15
    chapter_title = CT

    def construct(self):
        d = m15.target_point()[0]
        xs = np.linspace(0, d, 200)
        t = m15.travel_time(xs, d) * 1e9
        xn = m15.fermat_numeric(d)
        th1 = np.deg2rad(m15.THETA1_DEG)
        th2 = m15.snell_angle(th1)
        curve_scene(
            self,
            found=[r"n_1\sin\theta_1 = n_2\sin\theta_2\qquad(\text{ley de Snell})", r"T = \dfrac{1}{c}\left(n_1 s_1 + n_2 s_2\right)\ \text{es estacionario}\ (\text{principio de Fermat})"],
            note="El camino real extremiza el tiempo de viaje; la ley de Snell es la condición de mínimo.",
            fcap="Refracción y principio de Fermat.",
            problem=["Un rayo sale de un punto a 1.0 m sobre una interfaz aire-vidrio (n = 1.5) con incidencia de 40° y llega a un punto a 1.5 m bajo ella.",
                     "Halle el ángulo refractado y el punto de incidencia; luego halle el ángulo crítico vidrio-aire."],
            pcap="El punto de llegada está sobre el rayo de Snell.",
            analytic=[(rf"\theta_2 = \arcsin\!\left(\dfrac{{\sin\theta_1}}{{n}}\right) = {np.degrees(th2):.4f}^\circ", ACCENT), rf"x^* = h_1\tan\theta_1 = {m15.refraction_point_analytic():.5f}\ \mathrm{{m}},\qquad \theta_c = {m15.critical_angle_analytic():.4f}^\circ"],
            acap="Más allá de θ_c hay reflexión total interna.",
            numeric="Se minimiza el tiempo de viaje T(x) por búsqueda de la sección áurea; el ángulo crítico se halla por bisección.",
            xr=(0, d), yr=(float(t.min()) - 0.3, float(t.max()) + 0.3), xticks=(0, 0.5, 1.0, 1.5), yticks=(round(float(t.min()), 1), round(float(t.max()), 1)), xlabel="x [m]", ylabel="T [ns]",
            curves=[(xs, t, ANALYTIC)], dots=([xn], [float(m15.travel_time(xn, d)) * 1e9]), legend=[(ANALYTIC, "T(x)"), (NUMERIC, "mínimo (sección áurea)")],
            table=(["magnitud", "numérico", "analítico"],
                   [("x* [m]", f"{xn:.6f}", f"{m15.refraction_point_analytic():.6f}"), ("θ_c [°]", f"{m15.critical_angle_numeric():.6f}", f"{m15.critical_angle_analytic():.6f}")]),
            final="El mínimo de T coincide con el punto dado por la ley de Snell.", fsize=32)


class PrismDispersion(PhysicsScene):
    volume = 2
    chapter_number = 15
    chapter_title = CT

    def construct(self):
        lam = np.linspace(400, 700, 100)
        ln = np.array([400, 450, 500, 550, 600, 650, 700], dtype=float)
        dn = [m15.min_deviation_numeric(float(m15.cauchy(x))) for x in ln]
        n550 = float(m15.cauchy(550))
        curve_scene(
            self,
            found=[r"n(\lambda) = A + \dfrac{B}{\lambda^2}\qquad(\text{Cauchy})", r"\delta_{min} = 2\arcsin\!\left(n\sin\dfrac{\alpha}{2}\right) - \alpha"],
            note="La dispersión descompone la luz blanca porque el índice depende de la longitud de onda.",
            fcap="Dispersión en un prisma.",
            problem=["Un prisma de ángulo α = 60° tiene n = 1.50 + 5000/λ² (λ en nm).",
                     "Halle la desviación mínima para λ entre 400 y 700 nm y la separación angular entre el violeta y el rojo."],
            pcap="La desviación mínima ocurre con paso simétrico.",
            analytic=[(rf"n(550) = {n550:.5f},\qquad \delta_{{min}}(550) = {float(m15.min_deviation_analytic(n550)):.4f}^\circ", ACCENT),
                      rf"\delta(400) - \delta(700) = {float(m15.min_deviation_analytic(m15.cauchy(400)) - m15.min_deviation_analytic(m15.cauchy(700))):.4f}^\circ"],
            acap="El azul se desvía más que el rojo.",
            numeric="Se traza el rayo por ambas caras (Snell en cada una) y se minimiza la desviación respecto del ángulo de incidencia con sección áurea.",
            xr=(400, 700), yr=(36, 42.5), xticks=(400, 500, 600, 700), yticks=(36, 38, 40, 42), xlabel="λ [nm]", ylabel="δ_min [°]",
            curves=[(lam, m15.min_deviation_analytic(m15.cauchy(lam)), ANALYTIC)], dots=(ln, dn), legend=[(ANALYTIC, "fórmula del prisma"), (NUMERIC, "trazado de rayos")],
            table=(["magnitud", "numérico", "analítico"],
                   [("δ(400) [°]", f"{dn[0]:.6f}", f"{float(m15.min_deviation_analytic(m15.cauchy(400))):.6f}"), ("δ(550) [°]", f"{dn[3]:.6f}", f"{float(m15.min_deviation_analytic(n550)):.6f}"),
                    ("δ(700) [°]", f"{dn[6]:.6f}", f"{float(m15.min_deviation_analytic(m15.cauchy(700))):.6f}")]),
            final="El trazado de rayos reproduce la fórmula de desviación mínima.", fsize=32)
