"""Capitulo 12. Dinamica rotacional."""
from __future__ import annotations

import numpy as np
from manim import DOWN, LEFT, RIGHT, UP, Circle, Create, Dot, FadeIn, FadeOut, Line, Rectangle, ValueTracker, VGroup, always_redraw, linear, ReplacementTransform

from animations.common import (
    ACCENT, ANALYTIC, ERROR, FAINT, FG, MUTED, NUMERIC, PANEL, M, PhysicsScene, T, axes_block, legend_block, mono_table,
    polyline, text_lines,
)
from physics import ch12_rotational_dynamics as c12


class MomentOfInertia(PhysicsScene):
    chapter_number = 12
    chapter_title = "Dinámica rotacional"

    def construct(self):
        m, L = c12.M_ROD, c12.L_ROD
        ie_a, ic_a = c12.rod_inertia_analytic("end"), c12.rod_inertia_analytic("center")
        self.foundations(
            [r"I = \sum_i m_i r_i^{2} \;\to\; I = \int r^{2}\,dm",
             r"I = I_{cm} + Md^{2}\qquad\text{(teorema de los ejes paralelos)}"],
            note="El momento de inercia depende del eje elegido: mide cómo se distribuye la masa respecto a él.",
            caption="La inercia rotacional juega el papel de la masa en la segunda ley para la rotación.")
        self.problem(
            ["Una barra uniforme de 3 kg y 1.2 m gira alrededor de uno de sus extremos y alrededor de su centro.",
             "Calcule I en ambos casos, verifique el teorema de ejes paralelos y halle I de un disco de 4 kg y 0.25 m."],
            caption="Las masas se distribuyen de forma continua.")
        self.analytic(
            [rf"I_{{extremo}} = \int_0^L x^2\dfrac{{M}}{{L}}dx = \dfrac{{ML^2}}{{3}} = {ie_a:.4f}\ \mathrm{{kg\,m^2}}",
             rf"I_{{cm}} = \dfrac{{ML^2}}{{12}} = {ic_a:.4f}\ \mathrm{{kg\,m^2}}",
             (rf"I_{{cm}} + M(L/2)^2 = {c12.parallel_axis(ic_a, m, L / 2):.4f} = I_{{extremo}}", ACCENT),
             rf"I_{{disco}} = \tfrac{{1}}{{2}}MR^2 = {c12.disc_inertia_analytic():.5f}\ \mathrm{{kg\,m^2}}"],
            size=32, caption="El teorema de ejes paralelos transforma un resultado en el otro.")

        self.numeric_phase("Se integra r² dm con Simpson: el área bajo r²λ es el momento de inercia (n = 4 y n = 400).")
        ax, deco = axes_block((0, L), (0, 4.0), 6.8, 3.6, LEFT * 3.0 + DOWN * 0.15, (0, 0.3, 0.6, 0.9, 1.2), (0, 1, 2, 3), "x [m]", "x² λ [kg m]", "{:g}")
        lam = m / L
        xs = np.linspace(0, L, 100)
        c_end = polyline(ax, xs, lam * xs**2, ANALYTIC, 3.5)
        c_cm = polyline(ax, xs, lam * (xs - L / 2) ** 2, ACCENT, 3.5)
        leg = legend_block([(ANALYTIC, "eje en el extremo"), (ACCENT, "eje en el centro")], RIGHT * 4.2 + UP * 2.0)
        self.play(Create(ax), FadeIn(deco), FadeIn(leg), run_time=1.0)
        self.play(Create(c_end), Create(c_cm), run_time=2.0)
        rows = [("extremo", f"{c12.rod_inertia_numeric('end', n=4):.5f}", f"{c12.rod_inertia_numeric('end'):.5f}", f"{ie_a:.5f}"),
                ("centro", f"{c12.rod_inertia_numeric('center', n=4):.5f}", f"{c12.rod_inertia_numeric('center'):.5f}", f"{ic_a:.5f}"),
                ("disco", "", f"{c12.disc_inertia_numeric():.5f}", f"{c12.disc_inertia_analytic():.5f}")]
        tab = mono_table(["eje", "n = 4", "n = 400", "analítico"], rows, size=15, colors=[MUTED, MUTED, NUMERIC, ANALYTIC], gap=0.3).move_to(RIGHT * 4.3 + DOWN * 0.3)
        self.say("Simpson es exacto para polinomios de grado ≤ 3, así que incluso n = 4 reproduce I de la barra.", FadeIn(tab), run_time=1.4)
        self.wait(2.5)


class HangingMassDisc(PhysicsScene):
    chapter_number = 12
    chapter_title = "Dinámica rotacional"

    def construct(self):
        m, big_m, r, g = c12.M_HANG, c12.M_DISC, c12.R_DISC, c12.G
        an = c12.hanging_analytic()
        nu = c12.hanging_numeric()
        self.foundations(
            [r"\sum \tau = I\alpha,\qquad \sum F = ma",
             r"a = \alpha R\quad(\text{cuerda sin deslizar}),\qquad I = \tfrac{1}{2}MR^{2}"],
            note="La tensión no es mg: la polea con masa consume parte de la energía en rotar.",
            caption="Dinámica combinada: traslación de la masa y rotación del disco.")
        self.problem(
            ["Una masa de 2 kg cuelga de una cuerda enrollada en un disco uniforme de 4 kg y radio 0.25 m.",
             "Halle la aceleración de la masa, la tensión y la aceleración angular del disco."],
            caption="El disco gira sin fricción en su eje.")
        self.analytic(
            [r"mg - T = ma,\qquad TR = \tfrac{1}{2}MR^{2}\,\alpha = \tfrac{1}{2}MRa",
             (rf"a = \dfrac{{mg}}{{m + M/2}} = {an['a']:.3f}\ \mathrm{{m/s^2}}", ACCENT),
             (rf"T = m(g - a) = {an['T']:.3f}\ \mathrm{{N}},\qquad \alpha = a/R = {an['alpha']:.2f}\ \mathrm{{rad/s^2}}", ACCENT)],
            caption="Si el disco no tuviera masa, a sería g.")

        self.numeric_phase("Las dos ecuaciones forman un sistema lineal en (a, T); luego se integra el movimiento con RK4 durante 1 s.")
        t, y = nu["t"], nu["y"]
        center = np.array([-4.0, 1.3, 0.0])
        rad = 0.9
        clock = ValueTracker(0.0)
        scale = 0.9
        right = center + np.array([rad, 0.0, 0.0])

        def disc():
            a = float(np.interp(clock.get_value(), t, y[:, 2]))
            return VGroup(Circle(radius=rad, stroke_color=FG, stroke_width=3, fill_color=PANEL, fill_opacity=1).move_to(center),
                          Line(center, center + rad * np.array([np.cos(-a), np.sin(-a), 0]), color=ACCENT, stroke_width=3))

        def block():
            dy = float(np.interp(clock.get_value(), t, y[:, 0])) * scale
            pos = right + np.array([0.0, -0.9 - dy, 0.0])
            return VGroup(Line(right, pos + UP * 0.25, color=FG, stroke_width=2),
                          Rectangle(width=0.5, height=0.5, stroke_color=NUMERIC, stroke_width=2.5, fill_color=PANEL, fill_opacity=1).move_to(pos))

        dsk, blk = always_redraw(disc), always_redraw(block)
        readout = always_redraw(lambda: text_lines([f"t = {clock.get_value():4.2f} s",
                                                    (f"y = {float(np.interp(clock.get_value(), t, y[:, 0])):4.2f} m", NUMERIC),
                                                    (f"v = {float(np.interp(clock.get_value(), t, y[:, 1])):4.2f} m/s", NUMERIC)], 19).move_to(RIGHT * 0.2 + UP * 1.2))
        self.play(FadeIn(dsk), FadeIn(blk), FadeIn(readout), run_time=0.8)
        self.play(clock.animate.set_value(1.0), run_time=3.5, rate_func=linear)
        tab = mono_table(["magnitud", "numérico", "analítico"],
                         [("a [m/s²]", f"{nu['a']:.5f}", f"{an['a']:.5f}"), ("T [N]", f"{nu['T']:.5f}", f"{an['T']:.5f}"),
                          ("ΔE [J]", f"{nu['energy_error']:.1e}", "0")],
                         size=17, colors=[MUTED, NUMERIC, ANALYTIC], gap=0.4).move_to(RIGHT * 3.6 + DOWN * 0.8)
        self.say("La energía potencial perdida es igual a la energía cinética de traslación y rotación (ΔE ≈ 0).", FadeIn(tab), run_time=1.3)
        self.wait(2.5)
