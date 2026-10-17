"""Capitulo 5. Fuerza y leyes de Newton."""
from __future__ import annotations

import numpy as np
from manim import (
    DOWN, LEFT, RIGHT, UP, Arc, Arrow, Create, DashedLine, Dot, FadeIn, FadeOut, Line, Polygon, Square,
    ValueTracker, VGroup, Write, always_redraw, LaggedStart, linear, VMobject, PI, Rectangle,
)

from animations.common import (
    ACCENT, ANALYTIC, BORDER, ERROR, FAINT, FG, MUTED, NUMERIC, PANEL, M, PhysicsScene, T, make_axes, mono_table,
)
from physics import ch05_newton as c5


def force_arrow(start, direction_deg, length, color, width=5):
    ang = np.deg2rad(direction_deg)
    end = np.array(start) + length * np.array([np.cos(ang), np.sin(ang), 0.0])
    return Arrow(np.array(start), end, buff=0, color=color, stroke_width=width,
                 max_tip_length_to_length_ratio=0.16, max_stroke_width_to_length_ratio=9)


class StaticEquilibrium(PhysicsScene):
    chapter_number = 5
    chapter_title = "Fuerza y leyes de Newton"

    def construct(self):
        W, al, be = c5.WEIGHT, c5.ALPHA_DEG, c5.BETA_DEG
        t_an, t_nu = c5.tensions_analytic(), c5.tensions_numeric()
        resid = np.linalg.norm(c5.net_force(*t_nu))

        # ---------------------------------------------------------- fundamentos
        self.phase("Fundamentos")
        f = VGroup(
            M(r"\sum \vec{F} = m\,\vec{a}", 56),
            M(r"\vec{a} = 0 \;\Longleftrightarrow\; \sum F_x = 0,\quad \sum F_y = 0", 44),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.7).move_to(DOWN * 0.1)
        note = T("Primera ley: sin fuerza neta, un cuerpo en reposo permanece en reposo.", 23, MUTED).move_to(DOWN * 1.9)
        self.say("La segunda ley se aplica componente a componente; el equilibrio es el caso a = 0.",
                 *[Write(x) for x in f], run_time=3.0)
        self.play(FadeIn(note))
        self.wait(1.0)
        self.clear_stage()

        # -------------------------------------------------------------- problema
        self.phase("Problema")
        q = VGroup(
            T("Un letrero de 500 N cuelga en reposo de dos cables.", 27, FG),
            T("El cable izquierdo forma 30° con la horizontal y el derecho, 45°.", 25, FG),
            T("Determine la tensión de cada cable.", 25, MUTED),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to(DOWN * 0.1)
        self.say("Diagrama de cuerpo libre sobre el punto donde concurren los cables.", FadeIn(q), run_time=1.4)
        self.wait(1.5)
        self.clear_stage()

        # ---------------------------------------------------------- analitica
        self.phase("Solución analítica")
        k = 0.0048
        node = np.array([-3.5, 0.4, 0.0])
        ray1 = DashedLine(node, node + 3.3 * np.array([-np.cos(np.deg2rad(al)), np.sin(np.deg2rad(al)), 0]), color=FAINT, stroke_width=1.5, dash_length=0.1)
        ray2 = DashedLine(node, node + 3.3 * np.array([np.cos(np.deg2rad(be)), np.sin(np.deg2rad(be)), 0]), color=FAINT, stroke_width=1.5, dash_length=0.1)
        hline = DashedLine(node + LEFT * 2.6, node + RIGHT * 2.6, color=FAINT, stroke_width=1.5, dash_length=0.1)
        a1 = force_arrow(node, 180 - al, k * t_an[0], FG)
        a2 = force_arrow(node, be, k * t_an[1], NUMERIC)
        aw = force_arrow(node, 270, k * W, ERROR)
        arc1 = Arc(radius=0.7, start_angle=np.deg2rad(180 - al), angle=np.deg2rad(al), arc_center=node, color=ACCENT, stroke_width=2.5)
        arc2 = Arc(radius=0.9, start_angle=0, angle=np.deg2rad(be), arc_center=node, color=ACCENT, stroke_width=2.5)
        l1 = M(r"\vec{T}_1", 30, FG).next_to(a1.get_end(), LEFT, buff=0.1)
        l2 = M(r"\vec{T}_2", 30, NUMERIC).next_to(a2.get_end(), RIGHT, buff=0.1)
        lw = M(r"\vec{W}", 30, ERROR).next_to(aw.get_end(), DOWN, buff=0.1)
        la = M(r"\alpha", 26, ACCENT).move_to(node + 1.05 * np.array([-np.cos(np.deg2rad(al / 2)), np.sin(np.deg2rad(al / 2)), 0]))
        lb = M(r"\beta", 26, ACCENT).move_to(node + 1.3 * np.array([np.cos(np.deg2rad(be / 2)), np.sin(np.deg2rad(be / 2)), 0]))
        pt = Dot(node, radius=0.07, color=FG)
        self.say("Tres fuerzas concurrentes: dos tensiones y el peso.",
                 FadeIn(ray1), FadeIn(ray2), FadeIn(hline), FadeIn(pt), run_time=0.8)
        self.play(Create(aw), FadeIn(lw), run_time=0.8)
        self.play(Create(a1), Create(a2), FadeIn(l1), FadeIn(l2), Create(arc1), Create(arc2), FadeIn(la), FadeIn(lb), run_time=1.4)

        steps = VGroup(
            M(r"\sum F_x = 0:\quad T_1\cos\alpha = T_2\cos\beta", 34),
            M(r"\sum F_y = 0:\quad T_1\sin\alpha + T_2\sin\beta = W", 34),
            M(rf"T_1 = \dfrac{{W\cos\beta}}{{\sin(\alpha+\beta)}} = {t_an[0]:.2f}\ \mathrm{{N}}", 38, ACCENT),
            M(rf"T_2 = \dfrac{{W\cos\alpha}}{{\sin(\alpha+\beta)}} = {t_an[1]:.2f}\ \mathrm{{N}}", 38, ACCENT),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.38).move_to(RIGHT * 3.0 + DOWN * 0.1)
        self.say("Se despeja T₂ de la ecuación horizontal y se sustituye en la vertical.", run_time=0.4)
        for st in steps:
            self.play(Write(st), run_time=1.2)
        self.wait(1.5)
        self.play(FadeOut(steps), run_time=0.5)

        # ----------------------------------------------------------- numerica
        self.phase("Solución numérica")
        mat = M(r"\begin{bmatrix}\cos\alpha & -\cos\beta\\ \sin\alpha & \sin\beta\end{bmatrix}"
                r"\begin{bmatrix}T_1\\T_2\end{bmatrix}="
                r"\begin{bmatrix}0\\W\end{bmatrix}", 36).move_to(RIGHT * 3.0 + UP * 1.5)
        res = VGroup(
            mono_table(["método", "T1 [N]", "T2 [N]"],
                       [("numpy.linalg.solve", f"{t_nu[0]:.4f}", f"{t_nu[1]:.4f}"),
                        ("analítico", f"{t_an[0]:.4f}", f"{t_an[1]:.4f}")],
                       size=18, colors=[MUTED, NUMERIC, NUMERIC], gap=0.5),
            T(f"|ΣF| residual = {resid:.1e} N", 18, MUTED, mono=True),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to(RIGHT * 3.3 + DOWN * 0.5)
        self.say("Las dos ecuaciones forman un sistema lineal 2×2; la suma de fuerzas resultante es nula.",
                 Write(mat), run_time=1.6)
        self.play(FadeIn(res), run_time=1.2)
        self.wait(2.0)
        self.play(*[FadeOut(m) for m in (mat, res, ray1, ray2, hline, a1, a2, aw, l1, l2, lw, arc1, arc2, la, lb, pt)], run_time=0.7)

        # barrido de alfa: tensiones numericas
        self.say("Con β = 45° fijo, se resuelve el sistema para cada α entre 0° y 90°.", run_time=0.5)
        axes = make_axes([0, 90, 15], [0, 800, 200], 7.6, 3.7, numbers=False).move_to(LEFT * 2.4 + DOWN * 0.1)
        xt = VGroup(*[T(str(v), 14, MUTED, mono=True).next_to(axes.c2p(v, 0), DOWN, buff=0.1) for v in (0, 30, 60, 90)])
        yt = VGroup(*[T(str(v), 14, MUTED, mono=True).next_to(axes.c2p(0, v), LEFT, buff=0.1) for v in (200, 400, 600, 800)])
        xl = T("α [°]", 16, MUTED).next_to(axes.x_axis, RIGHT, buff=0.45)
        yl = T("T [N]", 16, MUTED).next_to(axes.y_axis, UP, buff=0.1)
        self.play(Create(axes), FadeIn(xt), FadeIn(yt), FadeIn(xl), FadeIn(yl), run_time=1.0)
        grid = np.linspace(0, 89, 90)
        tens = c5.tension_vs_angle(grid, be, W)
        pts1 = [axes.c2p(a, v) for a, v in zip(grid, tens[:, 0])]
        pts2 = [axes.c2p(a, v) for a, v in zip(grid, tens[:, 1])]
        c1_ = VMobject(color=FG, stroke_width=3.5)
        c1_.set_points_as_corners(pts1)
        c2_ = VMobject(color=NUMERIC, stroke_width=3.5)
        c2_.set_points_as_corners(pts2)
        leg = VGroup(
            VGroup(Line(LEFT * 0.15, RIGHT * 0.15, color=FG, stroke_width=4), T("T₁", 17, MUTED, mono=True)).arrange(RIGHT, buff=0.2),
            VGroup(Line(LEFT * 0.15, RIGHT * 0.15, color=NUMERIC, stroke_width=4), T("T₂", 17, MUTED, mono=True)).arrange(RIGHT, buff=0.2),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15).move_to(RIGHT * 4.3 + UP * 1.5)
        self.play(Create(c1_), Create(c2_), FadeIn(leg), run_time=2.0)
        mark = VGroup(
            Dot(axes.c2p(al, t_nu[0]), radius=0.08, color=ACCENT),
            Dot(axes.c2p(al, t_nu[1]), radius=0.08, color=ACCENT),
            DashedLine(axes.c2p(al, 0), axes.c2p(al, t_nu[1]), color=ACCENT, stroke_width=1.5, dash_length=0.08),
        )
        t90 = c5.tensions_numeric(W, 90.0, be)
        t45 = c5.tensions_numeric(W, 45.0, be)
        info = VGroup(
            T(f"α = 30°   T₁ {t_nu[0]:6.1f}   T₂ {t_nu[1]:6.1f}", 17, ACCENT, mono=True),
            T(f"α = 45°   T₁ {t45[0]:6.1f}   T₂ {t45[1]:6.1f}", 17, MUTED, mono=True),
            T(f"α = 90°   T₁ {t90[0]:6.1f}   T₂ {abs(t90[1]):6.1f}", 17, MUTED, mono=True),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2).move_to(RIGHT * 4.3 + DOWN * 0.4)
        self.say("A medida que el cable izquierdo se verticaliza, soporta todo el peso y el derecho se relaja.",
                 FadeIn(mark), FadeIn(info), run_time=1.2)
        self.wait(2.5)


class InclinePulley(PhysicsScene):
    chapter_number = 5
    chapter_title = "Fuerza y leyes de Newton"

    def construct(self):
        m1, m2, th, g = c5.M1, c5.M2, c5.THETA_DEG, c5.G
        a_an, T_an = c5.pulley_analytic()
        a_nu, T_nu = c5.pulley_numeric()
        ts, ys = c5.pulley_trajectory(t1=2.2, n=220)

        # ---------------------------------------------------------- fundamentos
        self.phase("Fundamentos")
        f = VGroup(
            M(r"\sum \vec{F} = m\,\vec{a}", 52),
            M(r"\text{Polea ideal, cuerda inextensible:}\quad a_1 = a_2 = a,\quad T_1 = T_2 = T", 38),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.7).move_to(DOWN * 0.1)
        note = T("Se aplica la segunda ley a cada cuerpo por separado y se acoplan por la cuerda.", 23, MUTED).move_to(DOWN * 1.9)
        self.say("Un diagrama de cuerpo libre por cuerpo y una ecuación de movimiento por cada eje relevante.",
                 *[Write(x) for x in f], run_time=3.0)
        self.play(FadeIn(note))
        self.wait(1.0)
        self.clear_stage()

        # -------------------------------------------------------------- problema
        self.phase("Problema")
        q = VGroup(
            T("Un bloque de 4 kg descansa sobre un plano sin fricción inclinado 30°.", 26, FG),
            T("Una cuerda sobre una polea ideal lo une a un bloque colgante de 3 kg.", 26, FG),
            T("Determine la aceleración del sistema y la tensión en la cuerda.", 25, MUTED),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to(DOWN * 0.1)
        self.say("Se suelta desde el reposo.", FadeIn(q), run_time=1.4)
        self.wait(1.5)
        self.clear_stage()

        # ------------------------------------------------- geometria del sistema
        self.phase("Solución analítica")
        th_r = np.deg2rad(th)
        n = np.array([-np.sin(th_r), np.cos(th_r), 0.0])
        u = np.array([np.cos(th_r), np.sin(th_r), 0.0])
        p0 = np.array([-6.2, -2.75, 0.0])
        length = 5.2
        p1 = p0 + length * u
        wedge = Polygon(p0, p1, np.array([p1[0], p0[1], 0.0]), stroke_color=FAINT, stroke_width=2, fill_color="#18181b", fill_opacity=1)
        floor = Line(p0 + LEFT * 0.3, np.array([p1[0] + 3.0, p0[1], 0.0]), color=FAINT, stroke_width=2)
        radius = 0.3
        center = p1 + (0.35 - radius) * n
        pulley = Circle_(center, radius)
        tangent_pt = center + radius * n
        right_pt = center + np.array([radius, 0.0, 0.0])
        wrap = Arc(radius=radius, start_angle=np.deg2rad(90 + th), angle=-np.deg2rad(90 + th), arc_center=center, color=FG, stroke_width=2.5)
        scale = 0.42
        s0 = 1.1
        rope_len = 0.55
        half = 0.25
        s_t = ValueTracker(0.0)

        def s_now():
            return scale * float(np.interp(s_t.get_value(), ts, ys[:, 0]))

        def block1_center():
            return p0 + (s0 + s_now()) * u + 0.3 * n

        def block1():
            b = Square(side_length=0.6, stroke_color=FG, stroke_width=2.5, fill_color=PANEL, fill_opacity=1)
            b.rotate(th_r).move_to(block1_center())
            lbl = T("m₁", 20, FG).move_to(block1_center())
            return VGroup(b, lbl)

        def block2_center():
            return right_pt + np.array([0.0, -rope_len - half - s_now(), 0.0])

        def block2():
            b = Rectangle(width=0.5, height=0.5, stroke_color=NUMERIC, stroke_width=2.5, fill_color=PANEL, fill_opacity=1)
            b.move_to(block2_center())
            lbl = T("m₂", 18, NUMERIC).move_to(block2_center())
            return VGroup(b, lbl)

        def ropes():
            return VGroup(
                Line(block1_center(), tangent_pt, color=FG, stroke_width=2),
                Line(right_pt, block2_center() + UP * half, color=FG, stroke_width=2),
            )

        b1, b2, rp = always_redraw(block1), always_redraw(block2), always_redraw(ropes)
        ang_arc = Arc(radius=0.9, start_angle=0, angle=th_r, arc_center=p0, color=ACCENT, stroke_width=2.5)
        ang_lab = M(r"30^\circ", 24, ACCENT).move_to(p0 + np.array([1.25, 0.2, 0.0]))
        self.say("Sistema inicial: ambos cuerpos en reposo.",
                 FadeIn(floor), FadeIn(wedge), FadeIn(pulley), Create(wrap), FadeIn(rp), FadeIn(b1), FadeIn(b2), FadeIn(ang_arc), FadeIn(ang_lab),
                 run_time=1.6)

        steps = VGroup(
            M(r"m_1:\quad T - m_1 g\sin\theta = m_1 a", 34),
            M(r"m_2:\quad m_2 g - T = m_2 a", 34),
            M(rf"a = \dfrac{{m_2 - m_1\sin\theta}}{{m_1+m_2}}\,g = {a_an:.4f}\ \mathrm{{m/s^2}}", 36, ACCENT),
            M(rf"T = m_1\,(a + g\sin\theta) = {T_an:.3f}\ \mathrm{{N}}", 36, ACCENT),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.36).move_to(RIGHT * 3.4 + UP * 0.3)
        self.say("Al sumar las dos ecuaciones desaparece la tensión; luego se sustituye para hallar T.", run_time=0.4)
        for st in steps:
            self.play(Write(st), run_time=1.2)
        self.wait(1.2)
        self.play(FadeOut(steps), run_time=0.5)

        # ----------------------------------------------------------- numerica 1
        self.phase("Solución numérica")
        mat = M(r"\begin{bmatrix}m_1 & -1\\ m_2 & 1\end{bmatrix}\begin{bmatrix}a\\T\end{bmatrix}="
                r"\begin{bmatrix}-m_1 g\sin\theta\\ m_2 g\end{bmatrix}", 34).move_to(RIGHT * 3.4 + UP * 1.7)
        res = mono_table(["método", "a [m/s²]", "T [N]"],
                         [("numpy.linalg.solve", f"{a_nu:.6f}", f"{T_nu:.6f}"),
                          ("analítico", f"{a_an:.6f}", f"{T_an:.6f}")],
                         size=18, colors=[MUTED, NUMERIC, NUMERIC], gap=0.45).move_to(RIGHT * 3.6 + UP * 0.2)
        self.say("Las dos ecuaciones forman un sistema lineal en (a, T); con a conocida se integra el movimiento por RK4.",
                 Write(mat), run_time=1.6)
        self.play(FadeIn(res), run_time=1.0)
        readout = always_redraw(lambda: VGroup(
            T(f"t = {s_t.get_value():4.2f} s", 18, MUTED, mono=True),
            T(f"desplazamiento = {float(np.interp(s_t.get_value(), ts, ys[:, 0])):5.2f} m", 18, NUMERIC, mono=True),
            T(f"velocidad      = {float(np.interp(s_t.get_value(), ts, ys[:, 1])):5.2f} m/s", 18, NUMERIC, mono=True),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.16).move_to(RIGHT * 3.7 + DOWN * 1.5))
        self.add(readout)
        self.play(s_t.animate.set_value(2.2), run_time=4.5, rate_func=linear)
        exact = 0.5 * a_an * 2.2**2
        self.say(f"A los 2.2 s: s = ½at² = {exact:.3f} m, igual al valor integrado numéricamente.", run_time=0.6)
        self.wait(2.0)
        self.play(*[FadeOut(m) for m in (mat, res, readout, floor, wedge, pulley, wrap, rp, b1, b2, ang_arc, ang_lab)], run_time=0.7)

        # ----------------------------------------------------------- numerica 2
        self.say("Barrido del ángulo del plano: la aceleración cambia de signo donde el sistema queda en equilibrio.", run_time=0.5)
        axes = make_axes([0, 90, 15], [-2, 5, 1], 7.6, 3.7, numbers=False).move_to(LEFT * 2.4 + DOWN * 0.1)
        xt = VGroup(*[T(str(v), 14, MUTED, mono=True).next_to(axes.c2p(v, -2), DOWN, buff=0.1) for v in (0, 30, 60, 90)])
        yt = VGroup(*[T(str(v), 14, MUTED, mono=True).next_to(axes.c2p(0, v), LEFT, buff=0.1) for v in (-2, 0, 2, 4)])
        xl = T("θ [°]", 16, MUTED).next_to(axes.c2p(90, -2), RIGHT, buff=0.45)
        yl = T("a [m/s²]", 16, MUTED).next_to(axes.y_axis, UP, buff=0.1)
        zero = DashedLine(axes.c2p(0, 0), axes.c2p(90, 0), color=FAINT, stroke_width=1.5, dash_length=0.1)
        self.play(Create(axes), FadeIn(xt), FadeIn(yt), FadeIn(xl), FadeIn(yl), FadeIn(zero), run_time=1.0)
        grid = np.linspace(0, 90, 91)
        acc = c5.acceleration_vs_angle(grid)
        curve = VMobject(color=NUMERIC, stroke_width=3.5)
        curve.set_points_as_corners([axes.c2p(x, y) for x, y in zip(grid, acc)])
        self.play(Create(curve), run_time=2.0)
        th_num = c5.balance_angle_numeric()
        th_an = c5.balance_angle_analytic()
        root = Dot(axes.c2p(th_num, 0), radius=0.09, color=ACCENT)
        info = VGroup(
            T(f"a(0°)  = {acc[0]:+.3f} m/s²", 17, MUTED, mono=True),
            T(f"a(90°) = {acc[-1]:+.3f} m/s²", 17, MUTED, mono=True),
            T(f"bisección  θ = {th_num:.4f}°", 17, NUMERIC, mono=True),
            T(f"arcsen(m₂/m₁) = {th_an:.4f}°", 17, ANALYTIC, mono=True),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2).move_to(RIGHT * 4.6 + DOWN * 0.2)
        self.say("La raíz numérica coincide con sen θ = m₂/m₁, la condición de equilibrio.", FadeIn(root), FadeIn(info), run_time=1.4)
        self.wait(2.5)


def Circle_(center, radius):
    from manim import Circle
    c = Circle(radius=radius, stroke_color=FG, stroke_width=2.5, fill_color=PANEL, fill_opacity=1)
    c.move_to(center)
    return c
