"""Volumen 2, capitulo 3. Ley de Gauss."""
from __future__ import annotations

import numpy as np
from manim import DOWN, LEFT, RIGHT, UP, Arrow, Circle, Create, DashedLine, Dot, FadeIn, FadeOut, Rectangle, VGroup, LaggedStart

from animations.common import (
    ACCENT, ANALYTIC, ERROR, FAINT, FG, MUTED, NUMERIC, M, PhysicsScene, T, axes_block, legend_block, mono_table, polyline,
    text_lines,
)
from physics import v2ch03_gauss as e3
from physics.constants import EPS0


class GaussFlux(PhysicsScene):
    volume = 2
    chapter_number = 3
    chapter_title = "Ley de Gauss"

    def construct(self):
        q = e3.Q
        phi_an = e3.flux_sphere_analytic()
        rows = [("esfera centrada", e3.flux_sphere_numeric()), ("esfera descentrada", e3.flux_sphere_numeric(charge_pos=(0.1, 0.05, 0.0))),
                ("cubo", e3.flux_cube_numeric()), ("carga fuera", e3.flux_sphere_numeric(radius=0.1, charge_pos=(0.3, 0.0, 0.0)))]
        self.foundations(
            [r"\Phi_E = \oint \vec E\cdot d\vec A = \dfrac{q_{enc}}{\varepsilon_0}",
             r"\text{Simetría} \Rightarrow E \text{ sale de la integral}"],
            note="El flujo depende solo de la carga encerrada, no de la forma de la superficie ni de dónde esté la carga dentro.",
            caption="Ley de Gauss.", size=42)
        self.problem(
            ["Una carga de 2 nC se rodea de una esfera de 0.3 m y de un cubo de 0.4 m de lado. Calcule el flujo en cada caso,",
             "con la carga centrada, descentrada y fuera de la superficie."],
            caption="Se integra E·n dA sobre toda la superficie.")
        self.analytic(
            [(rf"\Phi_E = \dfrac{{q}}{{\varepsilon_0}} = {phi_an:.2f}\ \mathrm{{N\,m^2/C}}", ACCENT),
             r"\text{Carga fuera de la superficie: } \Phi_E = 0"],
            size=40, caption="El resultado es el mismo para cualquier superficie cerrada que encierre la carga.")

        self.numeric_phase("Integración numérica de superficie con la regla del punto medio (200×400 celdas en la esfera, 6 × 400² en el cubo).")
        c1 = np.array([-4.3, 0.1, 0.0])
        sph = Circle(radius=1.5, stroke_color=NUMERIC, stroke_width=3).move_to(c1)
        cube = Rectangle(width=2.4, height=2.4, stroke_color=ACCENT, stroke_width=3).move_to(np.array([-1.0, 0.1, 0.0]))
        charges = VGroup(Dot(c1, radius=0.1, color=ERROR), Dot(np.array([-1.0, 0.1, 0.0]), radius=0.1, color=ERROR))
        arrows = VGroup()
        for cen in (c1, np.array([-1.0, 0.1, 0.0])):
            for ang in np.linspace(0, 2 * np.pi, 12, endpoint=False):
                u = np.array([np.cos(ang), np.sin(ang), 0])
                arrows.add(Arrow(cen + 0.25 * u, cen + 1.0 * u, buff=0, color=FAINT, stroke_width=2, max_tip_length_to_length_ratio=0.3))
        self.play(Create(sph), Create(cube), FadeIn(charges), run_time=1.0)
        self.play(FadeIn(arrows), run_time=1.0)
        tab = mono_table(["superficie", "Φ numérico", "q/ε₀"],
                         [(n_, f"{v:.3f}", f"{phi_an:.3f}" if n_ != "carga fuera" else "0") for n_, v in rows], size=15,
                         colors=[MUTED, NUMERIC, ANALYTIC], gap=0.3).move_to(RIGHT * 3.6 + UP * 0.7)
        unit = T("unidades: N·m²/C", 14, MUTED).next_to(tab, DOWN, buff=0.3).align_to(tab, LEFT)
        self.say("El flujo numérico es q/ε₀ para esfera y cubo, esté o no la carga en el centro, y es nulo si la carga queda fuera.", FadeIn(tab), FadeIn(unit), run_time=1.4)
        self.wait(2.5)


class SphereFieldProfile(PhysicsScene):
    volume = 2
    chapter_number = 3
    chapter_title = "Ley de Gauss"

    def construct(self):
        q, R = e3.Q_SPH, e3.R_SPH
        from physics.constants import K_E
        self.foundations(
            [r"r < R:\quad E = \dfrac{kQ\,r}{R^{3}}\qquad\qquad r \ge R:\quad E = \dfrac{kQ}{r^{2}}",
             r"E\,(4\pi r^{2}) = \dfrac{q_{enc}}{\varepsilon_0},\qquad q_{enc} = Q\,\dfrac{r^{3}}{R^{3}}\ (r<R)"],
            note="Fuera de la esfera el campo es el de una carga puntual en el centro; dentro crece linealmente con r.",
            caption="Esfera aislante con carga uniforme en su volumen.", size=36)
        self.problem(
            ["Una esfera aislante de radio 10 cm tiene una carga uniforme de 5 nC. Halle E(r) dentro y fuera.",
             "Verifique el resultado sumando el campo de 400 000 cargas puntuales repartidas al azar en el volumen."],
            caption="Distribución de carga uniforme.")
        e_surf = K_E * q / R**2
        self.analytic(
            [r"E_{in} = \dfrac{kQ}{R^{3}}\,r,\qquad E_{out} = \dfrac{kQ}{r^{2}}",
             (rf"E(R) = \dfrac{{kQ}}{{R^2}} = {e_surf:.1f}\ \mathrm{{N/C}}", ACCENT)],
            size=40, caption="El campo es continuo en la superficie y máximo ahí.")

        self.numeric_phase("Monte Carlo: se suma el campo coulombiano de 400 000 elementos de carga muestreados uniformemente en el volumen.")
        ax, deco = axes_block((0, 0.3), (0, 5500), 7.2, 3.6, LEFT * 2.8 + DOWN * 0.15, (0, 0.1, 0.2, 0.3), (0, 1000, 2000, 3000, 4000, 5000), "r [m]", "E [N/C]")
        rs = np.linspace(0.002, 0.3, 200)
        curve = polyline(ax, rs, e3.sphere_field_analytic(rs), ANALYTIC, 3.5)
        edge = DashedLine(ax.c2p(R, 0), ax.c2p(R, 5500), color=FAINT, stroke_width=1.5, dash_length=0.1)
        self.play(Create(ax), FadeIn(deco), Create(edge), Create(curve), run_time=2.0)
        rows = []
        dots = VGroup()
        for r in (0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.28):
            mc, sig = e3.sphere_field_monte_carlo(r, n=200_000)
            dots.add(Dot(ax.c2p(r, mc), radius=0.07, color=NUMERIC))
            if r in (0.05, 0.10, 0.20):
                rows.append((f"{r:.2f}", f"{mc:.1f} ± {sig:.1f}", f"{float(e3.sphere_field_analytic(r)):.1f}"))
        self.play(LaggedStart(*[FadeIn(d_, scale=1.5) for d_ in dots], lag_ratio=0.15), run_time=2.0)
        tab = mono_table(["r [m]", "E Monte Carlo", "E analítico"], rows, size=15, colors=[FG, NUMERIC, ANALYTIC], gap=0.3).move_to(RIGHT * 4.6 + UP * 0.7)
        note = text_lines([("N = 200 000 elementos por punto", MUTED), ("línea gris: superficie r = R", MUTED)], 14).move_to(RIGHT * 4.6 + DOWN * 0.9)
        self.say("Los puntos Monte Carlo siguen la curva de Gauss, con el quiebre de pendiente exactamente en la superficie de la esfera.", FadeIn(tab), FadeIn(note), run_time=1.4)
        self.wait(2.5)
