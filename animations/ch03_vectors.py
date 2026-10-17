"""Capitulo 3. Vectores."""
from __future__ import annotations

import numpy as np
from manim import (
    DOWN, LEFT, RIGHT, UP, Arrow, Create, DashedLine, Dot, FadeIn, FadeOut, Line, Polygon, Arc,
    ValueTracker, VGroup, Write, always_redraw, LaggedStart, linear, ORIGIN,
)

from animations.common import (
    ACCENT, ANALYTIC, BORDER, ERROR, FAINT, FG, MUTED, NUMERIC, M, PhysicsScene, T, make_axes, mono_table,
)
from physics import ch03_vectors as c3


def vec_arrow(axes, start, end, color, width=4.5):
    a = Arrow(axes.c2p(*start), axes.c2p(*end), buff=0, color=color, stroke_width=width,
              max_tip_length_to_length_ratio=0.12, max_stroke_width_to_length_ratio=8)
    return a


class VectorAddition(PhysicsScene):
    chapter_number = 3
    chapter_title = "Vectores"

    def construct(self):
        d1, d2 = c3.D1, c3.D2
        r_an, r_nu = c3.resultant_analytic(), c3.resultant_numeric()

        # ---------------------------------------------------------- fundamentos
        self.phase("Fundamentos")
        f = VGroup(
            M(r"A_x = A\cos\theta,\qquad A_y = A\sin\theta", 44),
            M(r"A = \sqrt{A_x^{2} + A_y^{2}},\qquad \theta = \tan^{-1}\!\dfrac{A_y}{A_x}", 44),
            M(r"\vec{R} = \vec{A} + \vec{B}\;\Longleftrightarrow\; R_x = A_x + B_x,\ \ R_y = A_y + B_y", 40),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.55).move_to(DOWN * 0.1)
        self.say("Un vector se describe por su módulo y dirección, o por sus componentes; la suma opera por componentes.",
                 *[Write(x) for x in f], run_time=4.0)
        self.wait(1.0)
        self.clear_stage()

        # -------------------------------------------------------------- problema
        self.phase("Problema")
        q = VGroup(
            T("Una persona camina 40 m a 30° sobre el eje x (vector A)", 26, FG),
            T("y luego 25 m a 120° (vector B).", 26, FG),
            T("¿Cuál es su desplazamiento resultante respecto al punto de partida?", 25, MUTED),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to(DOWN * 0.1)
        self.say("Los ángulos se miden desde el eje x positivo, en sentido antihorario.", FadeIn(q), run_time=1.6)
        self.wait(1.5)
        self.clear_stage()

        # ------------------------------------------------ analitica (geometria)
        self.phase("Solución analítica")
        s = 0.09
        axes = make_axes([-5, 45, 10], [-5, 50, 10], 50 * s, 55 * s, numbers=False).move_to(LEFT * 3.9 + DOWN * 0.15)
        ticks = VGroup(
            *[T(str(k), 14, MUTED, mono=True).next_to(axes.c2p(k, -5), DOWN, buff=0.05) for k in (0, 20, 40)],
            *[T(str(k), 14, MUTED, mono=True).next_to(axes.c2p(-5, k), LEFT, buff=0.08) for k in (20, 40)],
        )
        unit = T("m", 14, MUTED, mono=True).next_to(axes.x_axis, RIGHT, buff=0.1)
        tip1 = d1
        tip2 = d1 + d2
        arrow_a = vec_arrow(axes, (0, 0), tuple(tip1), FG)
        arrow_b = vec_arrow(axes, tuple(tip1), tuple(tip2), NUMERIC)
        arrow_r = vec_arrow(axes, (0, 0), tuple(tip2), ACCENT, width=5)
        lab_a = T("A", 22, FG).next_to(arrow_a.get_center(), DOWN + RIGHT * 0.4, buff=0.08)
        lab_b = T("B", 22, NUMERIC).next_to(arrow_b.get_center(), RIGHT, buff=0.12)
        lab_r = T("R", 22, ACCENT).next_to(arrow_r.get_center(), LEFT, buff=0.12)
        self.say("Método del polígono: B se coloca con su origen en la punta de A.",
                 Create(axes), FadeIn(ticks), FadeIn(unit), run_time=1.0)
        self.play(Create(arrow_a), FadeIn(lab_a), run_time=1.0)
        self.play(Create(arrow_b), FadeIn(lab_b), run_time=1.0)
        self.play(Create(arrow_r), FadeIn(lab_r), run_time=1.0)

        an_steps = VGroup(
            M(r"\gamma = 120^\circ - 30^\circ = 90^\circ", 36),
            M(r"R^{2} = A^{2} + B^{2} + 2AB\cos\gamma", 36),
            M(rf"R = \sqrt{{40^2 + 25^2}} = {r_an['magnitude']:.2f}\ \mathrm{{m}}", 38, ACCENT),
            M(r"\dfrac{\sin\delta}{B} = \dfrac{\sin\gamma}{R}\;\Rightarrow\;\delta = 32.00^\circ", 36),
            M(rf"\theta_R = 30^\circ + \delta = {r_an['angle_deg']:.2f}^\circ", 38, ACCENT),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.34).move_to(RIGHT * 2.7 + DOWN * 0.1)
        self.say("Ley de los cosenos para el módulo y ley de los senos para la dirección.", run_time=0.4)
        for st in an_steps:
            self.play(Write(st), run_time=1.1)
        self.wait(1.5)
        self.play(FadeOut(an_steps), run_time=0.5)

        # ---------------------------------------------------------- numerica
        self.phase("Solución numérica")
        comps = VGroup()
        for origin, vec, col in ((np.zeros(2), d1, FG), (d1, d2, NUMERIC)):
            end = origin + vec
            comps.add(
                DashedLine(axes.c2p(*origin), axes.c2p(end[0], origin[1]), color=col, stroke_width=2, dash_length=0.08),
                DashedLine(axes.c2p(end[0], origin[1]), axes.c2p(*end), color=col, stroke_width=2, dash_length=0.08),
            )
        tab = mono_table(
            ["vector", "x [m]", "y [m]"],
            [("A", f"{d1[0]:+.3f}", f"{d1[1]:+.3f}"),
             ("B", f"{d2[0]:+.3f}", f"{d2[1]:+.3f}"),
             ("R = A + B", f"{r_nu['x']:+.3f}", f"{r_nu['y']:+.3f}")],
            size=20, colors=[MUTED, FG, FG], gap=0.7,
        ).move_to(RIGHT * 2.9 + UP * 1.2)
        out = VGroup(
            T(f"|R|  = hypot(Rx, Ry)    = {r_nu['magnitude']:.4f} m", 19, NUMERIC, mono=True),
            T(f"θ_R  = atan2(Ry, Rx)    = {r_nu['angle_deg']:.4f}°", 19, NUMERIC, mono=True),
            T(f"|R| analítico           = {r_an['magnitude']:.4f} m", 19, ANALYTIC, mono=True),
            T(f"θ_R analítico           = {r_an['angle_deg']:.4f}°", 19, ANALYTIC, mono=True),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2).move_to(RIGHT * 2.9 + DOWN * 1.3)
        self.say("Se proyecta cada vector sobre los ejes y se suman las componentes; luego se recupera módulo y ángulo.",
                 LaggedStart(*[Create(c) for c in comps], lag_ratio=0.2), run_time=1.6)
        self.play(FadeIn(tab), run_time=1.2)
        self.play(FadeIn(out), run_time=1.2)
        self.say("Ambos caminos dan el mismo resultado: R = 47.17 m a 62.0°.", run_time=0.6)
        self.wait(2.5)


class VectorProducts(PhysicsScene):
    chapter_number = 3
    chapter_title = "Vectores"

    def construct(self):
        a, b = c3.A, c3.B
        pa = c3.products_analytic()
        theta = c3.angle_between_deg()
        area_mc, pts, inside = c3.parallelogram_area_monte_carlo()

        # ---------------------------------------------------------- fundamentos
        self.phase("Fundamentos")
        f = VGroup(
            M(r"\vec{A}\cdot\vec{B} = AB\cos\theta = A_xB_x + A_yB_y", 44),
            M(r"\left|\vec{A}\times\vec{B}\right| = AB\sin\theta = \left|A_xB_y - A_yB_x\right|", 44),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.7).move_to(DOWN * 0.1)
        note = T("El producto escalar da una proyección; el módulo del vectorial, el área del paralelogramo.", 22, MUTED).move_to(DOWN * 1.8)
        self.say("Dos productos con significado geométrico distinto.", *[Write(x) for x in f], run_time=3.0)
        self.play(FadeIn(note))
        self.wait(1.0)
        self.clear_stage()

        # -------------------------------------------------------------- problema
        self.phase("Problema")
        q = VGroup(
            M(r"\vec{A} = (4,\,1)\qquad \vec{B} = (2,\,3)", 50),
            T("Calcule el ángulo entre ambos y el área del paralelogramo que definen.", 25, MUTED),
        ).arrange(DOWN, buff=0.6).move_to(DOWN * 0.1)
        self.say("Vectores en el plano xy, sin unidades.", FadeIn(q), run_time=1.4)
        self.wait(1.4)
        self.clear_stage()

        # ---------------------------------------------------------- analitica
        self.phase("Solución analítica")
        s = 0.6
        axes = make_axes([-1, 7, 1], [-1, 5, 1], 8 * s, 6 * s, numbers=False).move_to(LEFT * 2.9 + DOWN * 0.1)
        ticks = VGroup(
            *[T(str(k), 14, MUTED, mono=True).next_to(axes.c2p(k, -1), DOWN, buff=0.05) for k in (0, 2, 4, 6)],
            *[T(str(k), 14, MUTED, mono=True).next_to(axes.c2p(-1, k), LEFT, buff=0.08) for k in (2, 4)],
        )
        para = Polygon(axes.c2p(0, 0), axes.c2p(*a), axes.c2p(*(a + b)), axes.c2p(*b),
                       stroke_color=FAINT, stroke_width=2, fill_color=NUMERIC, fill_opacity=0.12)
        arrow_a = vec_arrow(axes, (0, 0), tuple(a), FG)
        arrow_b = vec_arrow(axes, (0, 0), tuple(b), NUMERIC)
        lab_a = T("A", 22, FG).next_to(arrow_a.get_end(), RIGHT, buff=0.1)
        lab_b = T("B", 22, NUMERIC).next_to(arrow_b.get_end(), UP, buff=0.1)
        arc = Arc(radius=0.9, start_angle=np.arctan2(a[1], a[0]), angle=np.deg2rad(theta), arc_center=axes.c2p(0, 0), color=ACCENT, stroke_width=3)
        arc_lab = M(r"\theta", 28, ACCENT).move_to(axes.c2p(0, 0) + 1.25 * np.array([np.cos(np.deg2rad(34)), np.sin(np.deg2rad(34)), 0]) * 0.8 + RIGHT * 0.25)
        self.play(Create(axes), FadeIn(ticks), run_time=0.8)
        self.play(Create(arrow_a), Create(arrow_b), FadeIn(lab_a), FadeIn(lab_b), run_time=1.0)
        self.play(FadeIn(para), Create(arc), FadeIn(arc_lab), run_time=1.0)

        st = VGroup(
            M(rf"\vec{{A}}\cdot\vec{{B}} = 4(2) + 1(3) = {c3.dot():.0f}", 38),
            M(rf"\cos\theta = \dfrac{{11}}{{\sqrt{{17}}\sqrt{{13}}}} \;\Rightarrow\; \theta = {theta:.2f}^\circ", 38, ACCENT),
            M(rf"A_xB_y - A_yB_x = 4(3) - 1(2) = {c3.cross_z():.0f}", 38),
            M(rf"\text{{área}} = \left|\vec{{A}}\times\vec{{B}}\right| = {abs(c3.cross_z()):.0f}", 38, ACCENT),
            M(rf"AB\sin\theta = {pa['cross_by_angle']:.4f}", 34),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to(RIGHT * 3.2 + DOWN * 0.1)
        self.say("El ángulo sale del producto escalar; el área, de la componente z del producto vectorial.", run_time=0.4)
        for x in st:
            self.play(Write(x), run_time=1.0)
        self.wait(1.2)
        self.play(FadeOut(st), run_time=0.5)

        # ---------------------------------- numerica 1: area por Monte Carlo
        self.phase("Solución numérica")
        sample = 2400
        dots = VGroup()
        for p, ins in zip(pts[:sample], inside[:sample]):
            dots.add(Dot(axes.c2p(*p), radius=0.022, color=NUMERIC if ins else FAINT))
        est = T(f"área ≈ caja · (dentro / total) = {area_mc:.3f}", 19, NUMERIC, mono=True)
        est2 = T(f"|A × B| = {abs(c3.cross_z()):.3f}      N = 400 000", 19, ANALYTIC, mono=True)
        out = VGroup(est, est2).arrange(DOWN, aligned_edge=LEFT, buff=0.25).move_to(RIGHT * 3.2 + UP * 0.5)
        how = T("Se lanzan puntos al azar en la caja que\nenvuelve al paralelogramo y se cuenta la\nfracción que cae dentro.", 20, MUTED).move_to(RIGHT * 3.2 + UP * 1.7)
        self.say("Comprobación numérica independiente del área: método de Monte Carlo.", FadeIn(how), run_time=0.8)
        self.play(LaggedStart(*[FadeIn(d, run_time=0.15) for d in dots], lag_ratio=0.0015), run_time=3.5)
        self.play(FadeIn(out), run_time=1.0)
        self.wait(1.5)
        self.play(FadeOut(dots), FadeOut(out), FadeOut(how), FadeOut(para), FadeOut(arc), FadeOut(arc_lab), run_time=0.7)

        # ------------------------------ numerica 2: invariancia bajo rotacion
        phi = ValueTracker(0.0)
        base = axes.c2p(0, 0)

        def frame_axes():
            p = phi.get_value()
            ex = np.array([np.cos(p), np.sin(p), 0.0]) * 1.5
            ey = np.array([-np.sin(p), np.cos(p), 0.0]) * 1.5
            return VGroup(
                Line(base - ex * 0.3, base + ex, color=ACCENT, stroke_width=2.5),
                Line(base - ey * 0.3, base + ey, color=ACCENT, stroke_width=2.5),
                T("x'", 17, ACCENT).move_to(base + ex * 1.12),
                T("y'", 17, ACCENT).move_to(base + ey * 1.12),
            )

        frame = always_redraw(frame_axes)

        def readout():
            p = phi.get_value()
            ra, rb = c3.rotate_frame(a, p), c3.rotate_frame(b, p)
            return VGroup(
                T(f"φ = {np.rad2deg(p):6.1f}°", 20, ACCENT, mono=True),
                T(f"A' = ({ra[0]:+.3f}, {ra[1]:+.3f})", 20, FG, mono=True),
                T(f"B' = ({rb[0]:+.3f}, {rb[1]:+.3f})", 20, NUMERIC, mono=True),
                T(f"A'·B' = {np.dot(ra, rb):.6f}", 20, ANALYTIC, mono=True),
                T(f"A'×B' = {ra[0] * rb[1] - ra[1] * rb[0]:.6f}", 20, ANALYTIC, mono=True),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.25).move_to(RIGHT * 3.4 + DOWN * 0.1)

        ro = always_redraw(readout)
        self.say("Al girar los ejes cambian las componentes, pero A·B y A×B permanecen constantes: son invariantes.",
                 FadeIn(frame), FadeIn(ro), run_time=1.2)
        self.play(phi.animate.set_value(2 * np.pi), run_time=9.0, rate_func=linear)
        self.wait(1.5)
