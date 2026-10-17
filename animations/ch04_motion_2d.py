"""Capitulo 4. Movimiento en dos y tres dimensiones."""
from __future__ import annotations

import numpy as np
from manim import (
    DOWN, LEFT, RIGHT, UP, Arrow, Circle, Create, DashedLine, Dot, FadeIn, FadeOut, Line, ValueTracker,
    VGroup, Write, always_redraw, LaggedStart, linear, ORIGIN, DashedVMobject, Transform,
)

from animations.common import (
    ACCENT, ANALYTIC, ERROR, FAINT, FG, MUTED, NUMERIC, M, PhysicsScene, T, make_axes, mono_table,
)
from physics import ch04_motion_2d as c4


class Projectile(PhysicsScene):
    chapter_number = 4
    chapter_title = "Movimiento en dos y tres dimensiones"

    def construct(self):
        v0, th, g = c4.V0, c4.THETA_DEG, c4.G
        an = c4.projectile_analytic()
        t_nu, s_nu = c4.projectile_numeric(dt=0.005)

        # ---------------------------------------------------------- fundamentos
        self.phase("Fundamentos")
        f = VGroup(
            M(r"x(t) = v_0\cos\theta\; t", 46),
            M(r"y(t) = v_0\sin\theta\; t - \tfrac{1}{2}g\,t^{2}", 46),
            M(r"R = \dfrac{v_0^{2}\sin 2\theta}{g}", 46),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.5).move_to(LEFT * 3.0 + DOWN * 0.1)
        side = T("Los movimientos horizontal\ny vertical son independientes:\nax = 0,  ay = −g.", 24, MUTED).move_to(RIGHT * 3.5 + DOWN * 0.1)
        self.say("Se descomponen la posición y la velocidad inicial; cada eje se trata como movimiento en una dimensión.",
                 *[Write(x) for x in f], run_time=3.2)
        self.play(FadeIn(side))
        self.wait(1.0)
        self.clear_stage()

        # -------------------------------------------------------------- problema
        self.phase("Problema")
        q = VGroup(
            T("Un proyectil se lanza desde el suelo con v₀ = 25 m/s a 40° sobre la horizontal.", 26, FG),
            T("Determine el tiempo de vuelo, la altura máxima y el alcance.", 25, MUTED),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.35).move_to(DOWN * 0.1)
        self.say("Terreno plano y sin resistencia del aire; g = 9.81 m/s².", FadeIn(q), run_time=1.4)
        self.wait(1.4)
        self.clear_stage()

        # ---------------------------------------------------------- analitica
        self.phase("Solución analítica")
        d = VGroup(
            M(rf"v_{{0x}} = {v0 * np.cos(np.deg2rad(th)):.3f}\quad v_{{0y}} = {v0 * np.sin(np.deg2rad(th)):.3f}\ \mathrm{{m/s}}", 38),
            M(rf"y = 0 \;\Rightarrow\; T = \dfrac{{2v_{{0y}}}}{{g}} = {an['t_flight']:.3f}\ \mathrm{{s}}", 38, ACCENT),
            M(rf"H = \dfrac{{v_{{0y}}^{{2}}}}{{2g}} = {an['height']:.3f}\ \mathrm{{m}}", 38, ACCENT),
            M(rf"R = v_{{0x}}\,T = {an['range']:.3f}\ \mathrm{{m}}", 38, ACCENT),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.42).move_to(LEFT * 0.6 + DOWN * 0.1)
        self.say("El tiempo de vuelo sale de la ecuación vertical; el alcance, de la horizontal.", *[Write(x) for x in d], run_time=4.5)
        self.wait(1.5)
        self.clear_stage()

        # ----------------------------------------------- numerica 1: trayectoria
        self.phase("Solución numérica")
        sc = 0.115
        axes = make_axes([0, 72, 10], [0, 20, 5], 72 * sc, 20 * sc, numbers=False).move_to(LEFT * 2.2 + DOWN * 0.7)
        xt = VGroup(*[T(str(k), 14, MUTED, mono=True).next_to(axes.c2p(k, 0), DOWN, buff=0.1) for k in (0, 20, 40, 60)])
        yt = VGroup(*[T(str(k), 14, MUTED, mono=True).next_to(axes.c2p(0, k), LEFT, buff=0.1) for k in (10, 20)])
        unit = T("x, y [m]", 15, MUTED).next_to(axes.x_axis, RIGHT, buff=0.1).shift(DOWN * 0.18)
        curve = axes.plot(lambda x: x * np.tan(np.deg2rad(th)) - g * x**2 / (2 * v0**2 * np.cos(np.deg2rad(th))**2),
                          x_range=[0, an["range"]], color=ANALYTIC, stroke_width=3.5)
        self.say("Integración RK4 de (x, y, vx, vy) con paso 0.25 s; los puntos azules se comparan con la parábola.",
                 Create(axes), FadeIn(xt), FadeIn(yt), FadeIn(unit), run_time=1.0)
        self.play(Create(curve), run_time=1.6)
        t_m, s_m = c4.projectile_numeric(dt=0.25)
        dots = VGroup(*[Dot(axes.c2p(s[0], s[1]), radius=0.075, color=NUMERIC) for s in s_m])
        self.play(LaggedStart(*[FadeIn(x, scale=1.6) for x in dots], lag_ratio=0.25), run_time=3.0)

        rows = mono_table(
            ["magnitud", "RK4", "teoría"],
            [("alcance [m]", f"{s_nu[-1, 0]:.4f}", f"{an['range']:.4f}"),
             ("vuelo [s]", f"{t_nu[-1]:.4f}", f"{an['t_flight']:.4f}"),
             ("altura [m]", f"{s_nu[:, 1].max():.4f}", f"{an['height']:.4f}")],
            size=17, colors=[MUTED, NUMERIC, ANALYTIC], gap=0.4,
        ).move_to(RIGHT * 4.1 + UP * 1.3)
        rows_g = VGroup(rows)
        self.play(FadeIn(rows_g), run_time=1.0)
        self.wait(2.0)
        self.play(*[FadeOut(m) for m in (curve, dots, rows_g, axes, xt, yt, unit)], run_time=0.7)

        # ------------------------------------ numerica 2: barrido del angulo
        self.say("Barrido del ángulo de lanzamiento: el alcance máximo se localiza con búsqueda de sección áurea.", run_time=0.6)
        angles = [15, 30, 45, 60, 75]
        axes2 = make_axes([0, 72, 10], [0, 32, 5], 72 * 0.115, 32 * 0.115, numbers=False).move_to(LEFT * 2.2 + DOWN * 0.3)
        xt2 = VGroup(*[T(str(k), 14, MUTED, mono=True).next_to(axes2.c2p(k, 0), DOWN, buff=0.1) for k in (0, 20, 40, 60)])
        self.play(Create(axes2), FadeIn(xt2), run_time=0.8)
        cols = {15: FAINT, 30: MUTED, 45: ACCENT, 60: MUTED, 75: FAINT}
        trajs = []
        for ang in angles:
            tt, ss = c4.projectile_numeric(v0, ang, g, 0.02)
            pts = [axes2.c2p(s[0], s[1]) for s in ss]
            path = Line(pts[0], pts[1])
            from manim import VMobject
            path = VMobject(color=cols[ang], stroke_width=3.2 if ang == 45 else 2.2)
            path.set_points_as_corners(pts)
            trajs.append((ang, path, ss[-1, 0]))
        for ang, path, rng in trajs:
            self.play(Create(path), run_time=0.8)
        best = c4.optimal_angle_numeric()
        tbl = mono_table(["θ [°]", "alcance [m]"],
                         [(f"{a}", f"{r:.2f}") for a, _, r in trajs],
                         size=19, colors=[FG, NUMERIC], gap=0.6).move_to(RIGHT * 4.0 + UP * 0.9)
        res = VGroup(
            T(f"θ óptimo numérico   {best:.2f}°", 18, NUMERIC, mono=True),
            T("θ óptimo analítico  45.00°", 18, ANALYTIC, mono=True),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2).move_to(RIGHT * 4.3 + DOWN * 1.3)
        self.play(FadeIn(tbl), run_time=1.0)
        self.play(FadeIn(res), run_time=1.0)
        self.wait(2.5)


class CircularMotion(PhysicsScene):
    chapter_number = 4
    chapter_title = "Movimiento en dos y tres dimensiones"

    def construct(self):
        R, per = c4.RADIUS, c4.PERIOD
        ca = c4.circular_analytic()

        # ---------------------------------------------------------- fundamentos
        self.phase("Fundamentos")
        f = VGroup(
            M(r"\vec{r}(t) = R\,(\cos\omega t,\ \sin\omega t),\qquad \omega = \dfrac{2\pi}{T}", 42),
            M(r"v = \omega R,\qquad a_c = \dfrac{v^{2}}{R} = \omega^{2}R", 42),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.6).move_to(DOWN * 0.1)
        note = T("La velocidad es tangente a la trayectoria; la aceleración apunta al centro.", 23, MUTED).move_to(DOWN * 1.8)
        self.say("El módulo de la velocidad es constante, pero su dirección cambia: hay aceleración.",
                 *[Write(x) for x in f], run_time=3.0)
        self.play(FadeIn(note))
        self.wait(1.0)
        self.clear_stage()

        # -------------------------------------------------------------- problema
        self.phase("Problema")
        q = VGroup(
            T("Una partícula recorre una circunferencia de radio R = 2 m con periodo T = 4 s.", 26, FG),
            T("Determine su rapidez y su aceleración centrípeta.", 25, MUTED),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.35).move_to(DOWN * 0.1)
        self.say("Movimiento circular uniforme.", FadeIn(q), run_time=1.2)
        self.wait(1.4)
        self.clear_stage()

        # ---------------------------------------------------------- analitica
        self.phase("Solución analítica")
        d = VGroup(
            M(rf"\omega = \dfrac{{2\pi}}{{T}} = {ca['omega']:.4f}\ \mathrm{{rad/s}}", 40),
            M(rf"v = \omega R = {ca['speed']:.4f}\ \mathrm{{m/s}}", 40, ACCENT),
            M(rf"a_c = \omega^{{2}}R = {ca['a_c']:.4f}\ \mathrm{{m/s^2}}", 40, ACCENT),
            M(rf"a_c = \dfrac{{v^{{2}}}}{{R}} = {ca['a_c_from_v']:.4f}\ \mathrm{{m/s^2}}\ \ (\approx 0.50\,g)", 40),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.42).move_to(LEFT * 0.6 + DOWN * 0.1)
        self.say("Las dos expresiones de la aceleración centrípeta coinciden.", *[Write(x) for x in d], run_time=4.0)
        self.wait(1.5)
        self.clear_stage()

        # ----------------------------------------------------------- numerica
        self.phase("Solución numérica")
        sc = 0.9
        center = np.array([-3.4, -0.15, 0.0])
        circle = Circle(radius=R * sc, color=FAINT, stroke_width=2).move_to(center)
        center_dot = Dot(center, radius=0.04, color=MUTED)
        t = ValueTracker(0.0)
        h = 0.05

        def pos_at(tt):
            p = c4.position(tt)
            return center + sc * np.array([p[0], p[1], 0.0])

        particle = always_redraw(lambda: Dot(pos_at(t.get_value()), radius=0.12, color=ANALYTIC))

        def v_arrow():
            v = c4.velocity_fd(t.get_value(), h)
            p0 = pos_at(t.get_value())
            return Arrow(p0, p0 + 0.42 * np.array([v[0], v[1], 0.0]), buff=0, color=NUMERIC, stroke_width=4,
                         max_tip_length_to_length_ratio=0.22)

        def a_arrow():
            a = c4.acceleration_fd(t.get_value(), h)
            p0 = pos_at(t.get_value())
            return Arrow(p0, p0 + 0.2 * np.array([a[0], a[1], 0.0]), buff=0, color=ACCENT, stroke_width=4,
                         max_tip_length_to_length_ratio=0.3)

        va, aa = always_redraw(v_arrow), always_redraw(a_arrow)

        def readout():
            v = np.linalg.norm(c4.velocity_fd(t.get_value(), h))
            a = np.linalg.norm(c4.acceleration_fd(t.get_value(), h))
            return VGroup(
                T(f"t = {t.get_value():5.2f} s", 19, MUTED, mono=True),
                T(f"|v| = {v:.4f} m/s", 19, NUMERIC, mono=True),
                T(f"|a| = {a:.4f} m/s²", 19, ACCENT, mono=True),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.18).move_to(RIGHT * 3.6 + UP * 1.7)

        ro = always_redraw(readout)
        legend = VGroup(
            VGroup(Line(LEFT * 0.15, RIGHT * 0.15, color=NUMERIC, stroke_width=4), T("velocidad", 17, MUTED, mono=True)).arrange(RIGHT, buff=0.2),
            VGroup(Line(LEFT * 0.15, RIGHT * 0.15, color=ACCENT, stroke_width=4), T("aceleración (escala propia)", 17, MUTED, mono=True)).arrange(RIGHT, buff=0.2),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        legend.move_to(RIGHT * 3.6 + UP * 0.3)
        self.say("Solo se usa la posición: v y a se estiman por diferencias finitas con h = 0.05 s.",
                 Create(circle), FadeIn(center_dot), FadeIn(particle), FadeIn(va), FadeIn(aa), FadeIn(ro), FadeIn(legend), run_time=1.4)
        self.play(t.animate.set_value(per), run_time=8.0, rate_func=linear)
        rows = c4.fd_error_table()
        tab = mono_table(["h [s]", "|v| [m/s]", "|a| [m/s²]", "err a"],
                         [(f"{r['h']:g}", f"{r['speed']:.4f}", f"{r['a_c']:.4f}", f"{r['err_a']:.1e}") for r in rows],
                         size=17, colors=[FG, NUMERIC, ACCENT, MUTED], gap=0.4).move_to(RIGHT * 3.7 + DOWN * 1.2)
        ref = T(f"teoría: v = {ca['speed']:.4f}   a = {ca['a_c']:.4f}", 17, ANALYTIC, mono=True).next_to(tab, DOWN, buff=0.28).align_to(tab, LEFT)
        self.say("Al reducir h las estimaciones convergen a v = ωR y a = ω²R (error cuadrático en h).",
                 FadeOut(ro), FadeOut(legend), FadeIn(tab), run_time=1.2)
        self.play(FadeIn(ref))
        self.wait(2.5)
