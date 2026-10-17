"""Volumen 2, capitulo 2. El campo electrico."""
from __future__ import annotations

import numpy as np
from manim import DOWN, LEFT, RIGHT, UP, Create, DashedLine, Dot, FadeIn, FadeOut, VGroup, LaggedStart, VMobject

from animations.common import (
    ACCENT, ANALYTIC, ERROR, FAINT, FG, MUTED, NUMERIC, M, PhysicsScene, T, axes_block, legend_block, mono_table, polyline,
    text_lines,
)
from physics import v2ch02_field as e2


class DipoleField(PhysicsScene):
    volume = 2
    chapter_number = 2
    chapter_title = "El campo eléctrico"

    def construct(self):
        q, d = e2.QD, e2.DSEP
        z0 = 0.5
        ex_exact, ex_far = float(e2.dipole_field_axis_exact(z0)), float(e2.dipole_field_axis_far(z0))
        e_num, e_vec = e2.field_from_potential_numeric(0.3, 0.2), e2.field_dipole_vector(0.3, 0.2)
        self.foundations(
            [r"\vec E = \dfrac{\vec F}{q_0} = k\dfrac{q}{r^{2}}\,\hat r,\qquad \vec E = -\nabla V",
             r"p = qd,\qquad E_{eje}(z \gg d) \approx \dfrac{2kp}{z^{3}}"],
            note="Las líneas de campo parten de las cargas positivas y terminan en las negativas; su densidad mide |E|.",
            caption="Campo eléctrico de cargas puntuales y del dipolo.", size=40)
        self.problem(
            ["Un dipolo está formado por +5 nC y −5 nC separadas 2 cm. Halle el campo sobre su eje a 0.50 m del centro con la fórmula exacta y la de campo lejano,",
             "y trace sus líneas de campo."],
            caption="Momento dipolar p = qd = 1×10⁻¹⁰ C·m.")
        self.analytic(
            [r"E = kq\left[\dfrac{1}{(z - d/2)^2} - \dfrac{1}{(z + d/2)^2}\right]",
             (rf"E_{{exacto}}(0.50\ \mathrm{{m}}) = {ex_exact:.5f}\ \mathrm{{N/C}}", ACCENT),
             rf"E_{{lejano}} = 2kp/z^3 = {ex_far:.5f}\ \mathrm{{N/C}}\quad(\text{{diferencia }}{abs(ex_far - ex_exact) / ex_exact * 100:.3f}\,\%)"],
            size=34, caption="A 25 veces la separación, la aproximación dipolar es excelente.")

        self.numeric_phase("Las líneas se obtienen integrando dr/ds = E/|E| con RK4 desde la carga positiva; el campo también se calcula como −∇V.")
        sc = 5.0 / (14 * d)  # 14 d de ancho visible
        org = np.array([-1.5, -0.1, 0.0])

        def P(pt):
            return org + sc * np.array([pt[0], pt[1], 0.0])

        pos_dot = Dot(P((d / 2, 0)), radius=0.12, color=ERROR)
        neg_dot = Dot(P((-d / 2, 0)), radius=0.12, color=NUMERIC)
        self.play(FadeIn(pos_dot), FadeIn(neg_dot), run_time=0.8)
        lines = VGroup()
        closed_n = 0
        for ang in np.arange(15, 180, 15):
            for sgn in (1, -1):
                pts, closed = e2.trace_field_line(float(sgn * ang))
                closed_n += int(closed)
                m = VMobject(color=ANALYTIC, stroke_width=1.8)
                pp = [P(p) for p in pts[::6] if np.linalg.norm(P(p) - org) < 3.7]
                if len(pp) > 1:
                    m.set_points_as_corners(pp)
                    lines.add(m)
        self.play(LaggedStart(*[Create(l_) for l_ in lines], lag_ratio=0.05), run_time=4.0)
        tab = mono_table(["magnitud", "numérico", "analítico"],
                         [("E eje exacto [N/C]", f"{ex_far:.5f}", f"{ex_exact:.5f}"),
                          ("Ex en (0.3, 0.2) [N/C]", f"{e_num[0]:.4f}", f"{e_vec[0]:.4f}"), ("Ey en (0.3, 0.2) [N/C]", f"{e_num[1]:.4f}", f"{e_vec[1]:.4f}")],
                         size=14, colors=[MUTED, NUMERIC, ANALYTIC], gap=0.28).move_to(RIGHT * 4.3 + UP * 1.2)
        note = text_lines([(f"líneas que terminan en −q: {closed_n} de 22", MUTED), ("columna numérica: −∇V y campo lejano", MUTED)], 14).move_to(RIGHT * 4.3 + DOWN * 0.5)
        self.say("El gradiente numérico de V coincide con la suma vectorial de Coulomb y las líneas trazadas conectan +q con −q.", FadeIn(tab), FadeIn(note), run_time=1.3)
        self.wait(2.5)


class ChargedRing(PhysicsScene):
    volume = 2
    chapter_number = 2
    chapter_title = "El campo eléctrico"

    def construct(self):
        q, a = e2.RING_Q, e2.RING_R
        zmax_a, zmax_n = e2.ring_max_field_position_analytic(), e2.ring_max_field_position_numeric()
        emax = float(e2.ring_axis_field_analytic(zmax_a))
        self.foundations(
            [r"d\vec E = k\,\dfrac{dq}{r^{2}}\,\hat r,\qquad \vec E = \int d\vec E",
             r"E_z = \dfrac{kQz}{(z^{2} + a^{2})^{3/2}}"],
            note="Por simetría las componentes perpendiculares al eje se cancelan; solo sobrevive E_z.",
            caption="Campo de una distribución continua de carga.", size=40)
        self.problem(
            ["Un anillo de radio 10 cm tiene una carga total de 10 nC repartida uniformemente.",
             "Halle E sobre su eje y la posición donde el campo es máximo."],
            caption="La carga se supone distribuida de forma continua.")
        self.analytic(
            [(r"E_z(z) = \dfrac{kQz}{(z^2 + a^2)^{3/2}}", FG),
             r"\dfrac{dE_z}{dz} = 0 \;\Rightarrow\; z = \dfrac{a}{\sqrt{2}}",
             (rf"z_{{max}} = {zmax_a * 100:.3f}\ \mathrm{{cm}},\qquad E_{{max}} = {emax:.1f}\ \mathrm{{N/C}}", ACCENT)],
            size=36, caption="El campo es cero en el centro, crece, alcanza un máximo y decae como 1/z².")

        self.numeric_phase("Se suman 400 elementos dq del anillo con Simpson sobre el ángulo y se compara con la fórmula cerrada.")
        ax, deco = axes_block((0, 0.4), (0, 1.0e4), 7.0, 3.6, LEFT * 2.8 + DOWN * 0.15, (0, 0.1, 0.2, 0.3, 0.4), (0, 2500, 5000, 7500, 10000), "z [m]", "E_z [N/C]")
        zs = np.linspace(0, 0.4, 120)
        curve = polyline(ax, zs, e2.ring_axis_field_analytic(zs), ANALYTIC, 3.5)
        self.play(Create(ax), FadeIn(deco), Create(curve), run_time=2.0)
        pz = np.linspace(0.02, 0.39, 14)
        dots = VGroup(*[Dot(ax.c2p(float(z), e2.ring_axis_field_numeric(float(z))), radius=0.07, color=NUMERIC) for z in pz])
        self.play(LaggedStart(*[FadeIn(d_, scale=1.5) for d_ in dots], lag_ratio=0.1), run_time=2.0)
        mk = DashedLine(ax.c2p(zmax_n, 0), ax.c2p(zmax_n, emax), color=ACCENT, stroke_width=2, dash_length=0.08)
        tab = mono_table(["z [m]", "E numérico", "E analítico"],
                         [(f"{z:.2f}", f"{e2.ring_axis_field_numeric(z):.2f}", f"{float(e2.ring_axis_field_analytic(z)):.2f}") for z in (0.05, 0.10, 0.30)],
                         size=15, colors=[FG, NUMERIC, ANALYTIC], gap=0.3).move_to(RIGHT * 4.5 + UP * 1.0)
        res = text_lines([(f"z_max numérico   {zmax_n * 100:.3f} cm", NUMERIC), (f"z_max analítico  {zmax_a * 100:.3f} cm", ANALYTIC)], 15).move_to(RIGHT * 4.5 + DOWN * 0.8)
        self.say("La integración numérica reproduce la fórmula cerrada y localiza el máximo en a/√2.", Create(mk), FadeIn(tab), FadeIn(res), run_time=1.3)
        self.wait(2.5)
