"""Capitulo 11. Cinematica rotacional."""
from __future__ import annotations

import numpy as np
from manim import DOWN, LEFT, RIGHT, UP, Circle, Create, Dot, FadeIn, FadeOut, Line, ValueTracker, VGroup, always_redraw, linear, DashedLine, Arrow

from animations.common import (
    ACCENT, ANALYTIC, ERROR, FAINT, FG, MUTED, NUMERIC, PANEL, M, PhysicsScene, T, axes_block, legend_block, mono_table,
    polyline, text_lines,
)
from physics import ch11_rotational_kinematics as c11


class SpinUp(PhysicsScene):
    chapter_number = 11
    chapter_title = "Cinemática rotacional"

    def construct(self):
        al, tf, r = c11.ALPHA, c11.T_END, c11.RIM
        t, w, th = c11.spin_up_numeric()
        rim = c11.rim_accelerations(tf)
        self.foundations(
            [r"\omega = \omega_0 + \alpha t,\qquad \theta = \omega_0 t + \tfrac{1}{2}\alpha t^{2}",
             r"v = \omega r,\qquad a_t = \alpha r,\qquad a_c = \omega^{2} r"],
            note="Las ecuaciones son idénticas a las de la cinemática lineal con x → θ, v → ω y a → α.",
            caption="Rotación con aceleración angular constante.")
        self.problem(
            ["Una rueda de radio 0.40 m parte del reposo con α = 1.5 rad/s² durante 8 s.",
             "Halle ω, el ángulo girado y las aceleraciones de un punto del borde al final."],
            caption="El eje de giro está fijo.")
        self.analytic(
            [rf"\omega = \alpha t = {c11.omega_analytic(tf):.1f}\ \mathrm{{rad/s}},\quad \theta = \tfrac12\alpha t^2 = {c11.theta_analytic(tf):.1f}\ \mathrm{{rad}}\ ({c11.theta_analytic(tf) / (2 * np.pi):.2f}\ \mathrm{{rev}})",
             rf"a_t = \alpha r = {rim['a_t']:.2f}\ \mathrm{{m/s^2}},\quad a_c = \omega^2 r = {rim['a_c']:.1f}\ \mathrm{{m/s^2}}",
             (rf"|a| = \sqrt{{a_t^2 + a_c^2}} = {rim['a']:.2f}\ \mathrm{{m/s^2}}", ACCENT)],
            size=34, caption="La aceleración centrípeta domina: crece con ω², mientras la tangencial es constante.")

        self.numeric_phase("Se integra α dos veces con la regla del trapecio acumulada y se anima la rueda con el ángulo obtenido.")
        center = np.array([-4.0, -0.15, 0.0])
        radius = 1.6
        clock = ValueTracker(0.0)

        def ang():
            return float(np.interp(clock.get_value(), t, th))

        def wheel():
            a = ang()
            spokes = VGroup(*[Line(center, center + radius * np.array([np.cos(a + k * np.pi / 2), np.sin(a + k * np.pi / 2), 0]), color=FAINT, stroke_width=3) for k in range(4)])
            marker = Dot(center + radius * np.array([np.cos(a), np.sin(a), 0]), radius=0.1, color=ACCENT)
            return VGroup(Circle(radius=radius, stroke_color=FG, stroke_width=3).move_to(center), spokes, marker)

        wh = always_redraw(wheel)
        readout = always_redraw(lambda: text_lines([
            f"t = {clock.get_value():4.2f} s",
            (f"ω = {float(np.interp(clock.get_value(), t, w)):5.2f} rad/s", NUMERIC),
            (f"θ = {ang() / (2 * np.pi):5.2f} rev", NUMERIC),
            (f"v = {float(np.interp(clock.get_value(), t, w)) * r:5.2f} m/s", MUTED)], 19).move_to(RIGHT * 1.4 + UP * 1.2))
        self.play(FadeIn(wh), FadeIn(readout), run_time=0.8)
        self.play(clock.animate.set_value(tf), run_time=7.0, rate_func=linear)
        t2, w2, th2 = c11.spin_up_numeric_variable()
        tab = mono_table(["magnitud", "numérico", "analítico"],
                         [("ω [rad/s]", f"{w[-1]:.4f}", f"{c11.omega_analytic(tf):.4f}"),
                          ("θ [rad]", f"{th[-1]:.4f}", f"{c11.theta_analytic(tf):.4f}"),
                          ("θ con α(t)=0.25t", f"{th2[-1]:.3f}", f"{tf**3 / 24:.3f}")],
                         size=17, colors=[MUTED, NUMERIC, ANALYTIC], gap=0.4).move_to(RIGHT * 4.4 + DOWN * 1.2)
        self.say("La integración numérica reproduce las fórmulas, también con aceleración angular variable (última fila).", FadeIn(tab), run_time=1.2)
        self.wait(2.5)


class RollingCycloid(PhysicsScene):
    chapter_number = 11
    chapter_title = "Cinemática rotacional"

    def construct(self):
        r = c11.R_WHEEL
        L_an, A_an = c11.arch_length_analytic(), c11.arch_area_analytic()
        self.foundations(
            [r"v_{cm} = \omega R\quad(\text{rodadura sin deslizar})",
             r"x = R(\theta - \sin\theta),\qquad y = R(1 - \cos\theta)"],
            note="El punto del borde que toca el suelo está instantáneamente en reposo.",
            caption="Un punto del borde de una rueda que rueda sin deslizar describe una cicloide.")
        self.problem(
            ["Una rueda de radio R = 0.50 m rueda sin deslizar sobre el suelo.",
             "Halle la longitud de un arco de la cicloide y el área bajo él."],
            caption="Un arco corresponde a una vuelta completa, 0 ≤ θ ≤ 2π.")
        self.analytic(
            [r"ds = \sqrt{dx^2 + dy^2} = 2R\sin\tfrac{\theta}{2}\,d\theta",
             (rf"s = \int_0^{{2\pi}} 2R\sin\tfrac{{\theta}}{{2}}\,d\theta = 8R = {L_an:.3f}\ \mathrm{{m}}", ACCENT),
             (rf"A = \int y\,dx = 3\pi R^2 = {A_an:.4f}\ \mathrm{{m^2}}", ACCENT)],
            caption="El área bajo el arco es el triple del área del círculo generador.")

        self.numeric_phase("Se anima la rueda y se calculan la longitud y el área con Simpson y con la poligonal de la curva.")
        sc = 1.75
        ax, deco = axes_block((0, 3.4), (0, 1.2), 3.4 * sc, 1.2 * sc, LEFT * 2.7 + DOWN * 0.8, (0, 1, 2, 3), (0, 0.5, 1.0), "x [m]", "y [m]")
        self.play(Create(ax), FadeIn(deco), run_time=0.8)
        clock = ValueTracker(0.0)
        origin = ax.c2p(0, 0)

        def wheel():
            th = clock.get_value()
            cx, cy = r * th, r
            c = ax.c2p(cx, cy)
            px, py = c11.cycloid(th)
            return VGroup(Circle(radius=r * sc, stroke_color=FG, stroke_width=2.5).move_to(c),
                          Line(c, ax.c2p(float(px), float(py)), color=FAINT, stroke_width=2),
                          Dot(ax.c2p(float(px), float(py)), radius=0.08, color=ACCENT))

        def trace():
            th = np.linspace(0, max(clock.get_value(), 1e-3), 80)
            x, y = c11.cycloid(th)
            return polyline(ax, x, y, NUMERIC, 3)

        wh, tr = always_redraw(wheel), always_redraw(trace)
        self.add(tr, wh)
        self.play(clock.animate.set_value(2 * np.pi), run_time=5.0, rate_func=linear)
        tab = mono_table(["magnitud", "numérico", "analítico"],
                         [("s Simpson [m]", f"{c11.arch_length_numeric():.6f}", f"{L_an:.6f}"),
                          ("s poligonal [m]", f"{c11.arch_length_polyline():.6f}", f"{L_an:.6f}"),
                          ("A Simpson [m²]", f"{c11.arch_area_numeric():.6f}", f"{A_an:.6f}")],
                         size=16, colors=[MUTED, NUMERIC, ANALYTIC], gap=0.35).move_to(RIGHT * 4.5 + UP * 0.8)
        self.say("La longitud 8R y el área 3πR² coinciden con los valores numéricos.", FadeIn(tab), run_time=1.2)
        self.wait(2.5)
