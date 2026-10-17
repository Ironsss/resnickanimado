"""Capitulo 20. Ondas sonoras."""
from __future__ import annotations

import numpy as np
from manim import DOWN, LEFT, RIGHT, UP, Circle, Create, DashedLine, Dot, FadeIn, FadeOut, Line, ValueTracker, VGroup, always_redraw, linear

from animations.common import (
    ACCENT, ANALYTIC, ERROR, FAINT, FG, MUTED, NUMERIC, PANEL, M, PhysicsScene, T, axes_block, legend_block, mono_table,
    polyline, text_lines,
)
from physics import ch20_sound as c20


class DopplerEffect(PhysicsScene):
    chapter_number = 20
    chapter_title = "Ondas sonoras"

    def construct(self):
        f0, vs, c = c20.F0, c20.VS, c20.C
        ahead, behind = c20.doppler_numeric(vs=vs), c20.doppler_numeric(vs=-vs)
        self.foundations(
            [r"f' = f_0\,\dfrac{c + v_o}{c - v_s}",
             r"v_s > 0:\ \text{fuente que se acerca},\qquad v_o > 0:\ \text{observador que se acerca}"],
            note="Si la fuente avanza, los frentes de onda se comprimen por delante y se alargan por detrás.",
            caption="Efecto Doppler para el sonido.", size=38)
        self.problem(
            ["Un avión emite un tono de 440 Hz y vuela a 120 m/s hacia un observador en reposo; c = 343 m/s.",
             "¿Qué frecuencia oye el observador delante del avión, y qué frecuencia oiría uno situado detrás?"],
            caption="Aire en calma.")
        self.analytic(
            [(rf"f'_{{delante}} = f_0\dfrac{{c}}{{c - v_s}} = {c20.doppler_analytic():.2f}\ \mathrm{{Hz}}", ACCENT),
             (rf"f'_{{detrás}} = f_0\dfrac{{c}}{{c + v_s}} = {c20.doppler_analytic(vs=-vs):.2f}\ \mathrm{{Hz}}", ACCENT)],
            caption="Delante se oye más agudo; detrás, más grave.")

        self.numeric_phase("Se emiten pulsos cada 1/f₀ desde una fuente en movimiento y se mide el intervalo entre sus llegadas al observador.")
        # animacion en camara lenta con v_s = 0.35 c
        c_vis, vs_vis, Te = 1.6, 0.35 * 1.6, 0.42
        clock = ValueTracker(0.0)
        y0 = 0.3
        x_start = -4.6

        def frame():
            t = clock.get_value()
            grp = VGroup()
            k = 0
            while k * Te <= t:
                te = k * Te
                cx = x_start + vs_vis * te
                r = c_vis * (t - te)
                if r > 0.02 and r < 9:
                    grp.add(Circle(radius=r, stroke_color=NUMERIC, stroke_width=1.8, stroke_opacity=0.8).move_to(np.array([cx, y0, 0])))
                k += 1
            grp.add(Dot(np.array([x_start + vs_vis * t, y0, 0]), radius=0.1, color=ACCENT))
            return grp

        wf = always_redraw(frame)
        obs_r = Dot(np.array([3.2, y0, 0]), radius=0.08, color=FG)
        obs_l = Dot(np.array([-6.2, y0, 0]), radius=0.08, color=FG)
        lab_r = T("observador delante", 14, MUTED).next_to(obs_r, DOWN, buff=0.2)
        lab_l = T("observador detrás", 14, MUTED).next_to(obs_l, DOWN, buff=0.2).shift(RIGHT * 0.5)
        note = T("cámara lenta: ilustración con v_s = 0.35 c", 14, MUTED).move_to(DOWN * 1.9 + LEFT * 2.3)
        self.play(FadeIn(obs_r), FadeIn(obs_l), FadeIn(lab_r), FadeIn(lab_l), FadeIn(note), run_time=0.8)
        self.add(wf)
        self.play(clock.animate.set_value(4.4), run_time=6.0, rate_func=linear)
        self.play(FadeOut(wf), FadeOut(obs_r), FadeOut(obs_l), FadeOut(lab_r), FadeOut(lab_l), FadeOut(note), run_time=0.6)
        rows = []
        for label, vs_, vo_ in (("fuente se acerca", vs, 0.0), ("fuente se aleja", -vs, 0.0), ("observador se acerca (30 m/s)", 0.0, 30.0)):
            rows.append((label, f"{c20.doppler_numeric(vs=vs_, vo=vo_):.3f}", f"{c20.doppler_analytic(vs=vs_, vo=vo_):.3f}"))
        tab = mono_table(["caso", "f' numérica [Hz]", "f' analítica [Hz]"], rows, size=20, colors=[MUTED, NUMERIC, ANALYTIC], gap=0.5).move_to(DOWN * 0.1)
        self.say("El intervalo entre llegadas de pulsos reproduce la fórmula de Doppler en los tres casos.", FadeIn(tab), run_time=1.3)
        self.wait(2.5)


class BeatsAndPipes(PhysicsScene):
    chapter_number = 20
    chapter_title = "Ondas sonoras"

    def construct(self):
        f1, f2 = c20.F1, c20.F2
        t, s = c20.beat_signal(16000.0, 0.5)
        env = c20.envelope(s)
        fb_fft, fb_env = c20.beat_frequency_numeric(), c20.beat_envelope_frequency_numeric()
        self.foundations(
            [r"y = \sin 2\pi f_1t + \sin 2\pi f_2t = 2\cos\!\left(\pi(f_2 - f_1)t\right)\sin\!\left(\pi(f_1 + f_2)t\right)",
             r"f_{pulsación} = |f_2 - f_1|,\qquad f_n^{abierto} = \dfrac{nc}{2L},\quad f_n^{cerrado} = \dfrac{(2n-1)c}{4L}"],
            note="Dos tonos cercanos producen una oscilación rápida cuya amplitud varía lentamente: las pulsaciones.",
            caption="Superposición de ondas sonoras y resonancia en tubos.", size=32)
        self.problem(
            ["Dos diapasones suenan a 440 Hz y 444 Hz. ¿Cuántas pulsaciones por segundo se oyen?",
             "Además, halle las frecuencias de resonancia de un tubo de 0.75 m abierto en ambos extremos y de otro cerrado en uno."],
            caption="c = 343 m/s.")
        self.analytic(
            [(rf"f_{{pulsación}} = |444 - 440| = {c20.beat_frequency_analytic():.0f}\ \mathrm{{Hz}}", ACCENT),
             rf"\text{{Abierto: }}f_n = {', '.join(f'{x:.1f}' for x in c20.pipe_frequencies_analytic('open')[:3])}\ \mathrm{{Hz}}",
             rf"\text{{Cerrado: }}f_n = {', '.join(f'{x:.1f}' for x in c20.pipe_frequencies_analytic('closed')[:3])}\ \mathrm{{Hz}}"],
            size=34, caption="El tubo cerrado solo tiene armónicos impares.")

        self.numeric_phase("Se muestrea la suma de los tonos, se calcula su espectro (FFT) y la envolvente; los modos de los tubos salen de una matriz tridiagonal.")
        ax, deco = axes_block((0, 0.5), (-2.2, 2.2), 11.2, 2.4, LEFT * 0.4 + UP * 1.2, (0, 0.1, 0.2, 0.3, 0.4, 0.5), (-2, 0, 2), "t [s]", "y")
        step = 3
        wave = polyline(ax, t[::step], s[::step], NUMERIC, 1.2)
        top, bot = polyline(ax, t[::40], env[::40], ACCENT, 2.5), polyline(ax, t[::40], -env[::40], ACCENT, 2.5)
        self.play(Create(ax), FadeIn(deco), run_time=0.9)
        self.play(Create(wave), run_time=2.0)
        self.play(Create(top), Create(bot), run_time=1.5)
        rows = []
        for kind, name in (("open", "abierto"), ("closed", "cerrado")):
            nu, an = c20.pipe_frequencies_numeric(kind), c20.pipe_frequencies_analytic(kind)
            rows += [(f"{name} n={i + 1}", f"{nu[i]:.2f}", f"{an[i]:.2f}") for i in range(3)]
        tab = mono_table(["tubo", "f numérica [Hz]", "f analítica [Hz]"], rows, size=14, colors=[MUTED, NUMERIC, ANALYTIC], gap=0.3, row_buff=0.18).move_to(RIGHT * 4.3 + DOWN * 1.8 + LEFT * 0.0)
        res = text_lines([(f"pulsación (FFT)       {fb_fft:.3f} Hz", NUMERIC), (f"pulsación (envolvente) {fb_env:.3f} Hz", NUMERIC), (f"pulsación analítica   {c20.beat_frequency_analytic():.3f} Hz", ANALYTIC)], 14).move_to(LEFT * 3.2 + DOWN * 1.6)
        self.say("La envolvente (ámbar) late a 4 Hz; la FFT separa los dos tonos y los modos de los tubos coinciden con las fórmulas.", FadeIn(tab), FadeIn(res), run_time=1.3)
        self.wait(2.5)
