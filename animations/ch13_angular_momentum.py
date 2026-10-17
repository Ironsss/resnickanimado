"""Capitulo 13. Momento angular."""
from __future__ import annotations

import numpy as np
from manim import DOWN, LEFT, RIGHT, UP, Arrow, Circle, Create, DashedLine, Dot, FadeIn, FadeOut, Line, ValueTracker, VGroup, always_redraw, linear

from animations.common import (
    ACCENT, ANALYTIC, ERROR, FAINT, FG, MUTED, NUMERIC, PANEL, M, PhysicsScene, T, axes_block, legend_block, mono_table,
    polyline, text_lines,
)
from physics import ch13_angular_momentum as c13


class SkaterSpin(PhysicsScene):
    chapter_number = 13
    chapter_title = "Momento angular"

    def construct(self):
        wf = c13.omega_final_analytic()
        dk = c13.kinetic_energy_change_analytic()
        t, w, inertia, ang, k = c13.skater_numeric()
        self.foundations(
            [r"\vec L = I\vec\omega,\qquad \vec\tau_{ext} = \dfrac{d\vec L}{dt}",
             r"\tau_{ext} = 0 \;\Rightarrow\; I_i\omega_i = I_f\omega_f"],
            note="Sin torque externo el momento angular se conserva, aunque la energía cinética pueda cambiar.",
            caption="Conservación del momento angular.")
        self.problem(
            ["Una patinadora gira a 2 rad/s con los brazos extendidos (I = 4.0 kg m²).",
             "Al recogerlos, su inercia pasa a 1.6 kg m². Halle ω final y el cambio de energía cinética."],
            caption="El hielo no ejerce torque apreciable sobre el eje vertical.")
        self.analytic(
            [(rf"\omega_f = \dfrac{{I_i\omega_i}}{{I_f}} = {wf:.2f}\ \mathrm{{rad/s}}", ACCENT),
             rf"K_i = \tfrac{{1}}{{2}}I_i\omega_i^2 = {0.5 * c13.I_I * c13.W_I**2:.1f}\ \mathrm{{J}},\quad K_f = {0.5 * c13.I_F * wf**2:.1f}\ \mathrm{{J}}",
             (rf"\Delta K = {dk:.1f}\ \mathrm{{J}}\ \text{{(trabajo interno de los músculos)}}", ACCENT)],
            caption="La energía cinética aumenta porque la patinadora realiza trabajo al acercar los brazos.")

        self.numeric_phase("Sin torque: d(Iω)/dt = 0, es decir ω' = −(I'/I) ω. Se integra con RK4 mientras I(t) disminuye suavemente.")
        ax, deco = axes_block((0, 4), (0, 9), 6.4, 3.6, LEFT * 3.5 + DOWN * 0.15, (0, 1, 2, 3, 4), (0, 3, 6, 9), "t [s]", "")
        cw, ci, cl = polyline(ax, t, w, NUMERIC, 3.2), polyline(ax, t, inertia, ACCENT, 3.2), polyline(ax, t, ang, ANALYTIC, 3.2)
        leg = legend_block([(NUMERIC, "ω [rad/s]"), (ACCENT, "I [kg m²]"), (ANALYTIC, "L = Iω [kg m²/s]")], LEFT * 3.5 + UP * 2.4)
        leg.arrange(RIGHT, buff=0.5).move_to(LEFT * 3.5 + UP * 2.4)
        center = np.array([3.9, 0.9, 0.0])
        clock = ValueTracker(0.0)
        phase = np.concatenate([[0.0], np.cumsum(0.5 * (w[1:] + w[:-1]) * np.diff(t))])

        def skater():
            ph = float(np.interp(clock.get_value(), t, phase))
            rr = 0.35 + 0.95 * np.sqrt(float(np.interp(clock.get_value(), t, inertia)) / c13.I_I)
            arms = VGroup(*[Dot(center + rr * np.array([np.cos(ph + s), np.sin(ph + s), 0.0]), radius=0.1, color=ACCENT) for s in (0, np.pi)])
            return VGroup(Circle(radius=0.3, stroke_color=FG, stroke_width=3, fill_color=PANEL, fill_opacity=1).move_to(center),
                          Line(center + rr * np.array([np.cos(ph), np.sin(ph), 0]), center - rr * np.array([np.cos(ph), np.sin(ph), 0]), color=FG, stroke_width=3), arms)

        sk = always_redraw(skater)
        cursor = always_redraw(lambda: DashedLine(ax.c2p(clock.get_value(), 0), ax.c2p(clock.get_value(), 9), color=FAINT, stroke_width=1.5, dash_length=0.08))
        self.play(Create(ax), FadeIn(deco), FadeIn(leg), FadeIn(sk), FadeIn(cursor), run_time=1.0)
        self.add(cw, ci, cl)
        self.play(clock.animate.set_value(4.0), run_time=6.0, rate_func=linear)
        self.play(FadeOut(cursor), run_time=0.3)
        tab = mono_table(["magnitud", "RK4", "analítico"],
                         [("ω final", f"{w[-1]:.5f}", f"{wf:.5f}"), ("ΔL/L", f"{np.ptp(ang) / ang[0]:.1e}", "0"),
                          ("ΔK [J]", f"{k[-1] - k[0]:.4f}", f"{dk:.4f}")],
                         size=16, colors=[MUTED, NUMERIC, ANALYTIC], gap=0.35).move_to(RIGHT * 3.9 + DOWN * 1.5)
        self.say("L permanece constante (línea blanca) y ω crece como 1/I: el resultado numérico coincide con la ley de conservación.", FadeIn(tab), run_time=1.3)
        self.wait(2.5)


class GyroscopePrecession(PhysicsScene):
    chapter_number = 13
    chapter_title = "Momento angular"

    def construct(self):
        m, r, inertia, ws, g = c13.M_GYRO, c13.R_GYRO, c13.I_GYRO, c13.W_SPIN, c13.G
        Om = c13.precession_analytic()
        t, y, phi, Om_n = c13.precession_numeric()
        self.foundations(
            [r"\vec\tau = \vec r \times m\vec g,\qquad \dfrac{d\vec L}{dt} = \vec\tau",
             r"\Omega = \dfrac{\tau}{L} = \dfrac{mgr}{I\omega}"],
            note="El torque es perpendicular a L: cambia su dirección, no su módulo. El giróscopo precede en vez de caer.",
            caption="Precesión de un giróscopo con el eje horizontal.")
        self.problem(
            ["Un giróscopo de 0.5 kg tiene su centro de masa a 0.10 m del pivote y gira con ω = 200 rad/s (I = 2.5×10⁻³ kg m²).",
             "Halle su velocidad de precesión y cómo cambia con ω."],
            caption="El eje del rotor se mantiene horizontal.")
        self.analytic(
            [rf"L = I\omega = {inertia * ws:.2f}\ \mathrm{{kg\,m^2/s}},\qquad \tau = mgr = {m * g * r:.3f}\ \mathrm{{N\,m}}",
             (rf"\Omega = \dfrac{{\tau}}{{L}} = {Om:.3f}\ \mathrm{{rad/s}}", ACCENT),
             r"\Omega \propto \dfrac{1}{\omega}"],
            caption="Un giróscopo más rápido precede más despacio.")

        self.numeric_phase("Se integra dL/dt = r × mg en tres dimensiones con RK4 y se mide el ángulo azimutal de L en el plano horizontal.")
        center = np.array([-4.0, -0.1, 0.0])
        rad = 1.9
        circ = Circle(radius=rad, stroke_color=FAINT, stroke_width=2).move_to(center)
        clock = ValueTracker(0.0)

        def vec():
            a = float(np.interp(clock.get_value(), t, phi))
            return Arrow(center, center + rad * np.array([np.cos(a), np.sin(a), 0]), buff=0, color=ACCENT, stroke_width=5, max_tip_length_to_length_ratio=0.15)

        arr = always_redraw(vec)
        pivot = Dot(center, radius=0.07, color=FG)
        read = always_redraw(lambda: text_lines([f"t = {clock.get_value():4.2f} s",
                                                 (f"φ = {np.rad2deg(float(np.interp(clock.get_value(), t, phi))):6.1f}°", NUMERIC)], 19).move_to(RIGHT * 0.3 + UP * 1.4))
        top = T("vista desde arriba", 15, MUTED).next_to(circ, DOWN, buff=0.15)
        self.play(Create(circ), FadeIn(pivot), FadeIn(arr), FadeIn(read), FadeIn(top), run_time=0.9)
        self.play(clock.animate.set_value(6.0), run_time=6.0, rate_func=linear)
        rows = []
        for w_ in (100.0, 200.0, 400.0):
            rows.append((f"{w_:.0f}", f"{c13.precession_numeric(w_spin=w_)[3]:.4f}", f"{c13.precession_analytic(w_):.4f}"))
        tab = mono_table(["ω [rad/s]", "Ω numérico", "Ω analítico"], rows, size=17, colors=[FG, NUMERIC, ANALYTIC], gap=0.4).move_to(RIGHT * 3.6 + DOWN * 0.7)
        self.say("La pendiente de φ(t) da Ω; duplicar ω reduce Ω a la mitad, como predice Ω = mgr/(Iω).", FadeIn(tab), run_time=1.3)
        self.wait(2.5)
