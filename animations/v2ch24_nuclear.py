"""Volumen 2, capitulo 24. Fisica nuclear."""
from __future__ import annotations

import numpy as np

from animations.common import ACCENT, ANALYTIC, ERROR, NUMERIC, PhysicsScene, curve_scene
from physics import v2ch24_nuclear as m24

CT = "Física nuclear"


class DecayChain(PhysicsScene):
    volume = 2
    chapter_number = 24
    chapter_title = CT

    def construct(self):
        t = np.linspace(0, 20, 300)
        tt, y = m24.chain_numeric(20.0, 4000)
        mean, err = m24.mean_life_monte_carlo()
        curve_scene(
            self,
            found=[r"\dfrac{dN}{dt} = -\lambda N,\qquad N = N_0 e^{-\lambda t},\qquad T_{1/2} = \dfrac{\ln 2}{\lambda}", r"N_B(t) = N_0\dfrac{\lambda_A}{\lambda_B - \lambda_A}\left(e^{-\lambda_A t} - e^{-\lambda_B t}\right)\qquad(\text{Bateman})"],
            note="La actividad de la hija pasa por un máximo cuando la de la madre y la de la hija se igualan.",
            fcap="Radiactividad y cadenas de decaimiento.",
            problem=["Un núcleo A (λ_A = 1.0 s⁻¹) decae a B (λ_B = 0.25 s⁻¹), que decae a un estado estable.",
                     "Halle N_B(t), el instante de su máximo y verifique la vida media con una simulación de Monte Carlo."],
            pcap="Escala de tiempo adimensional; N₀ = 1.",
            analytic=[(rf"t_{{max}} = \dfrac{{\ln(\lambda_A/\lambda_B)}}{{\lambda_A - \lambda_B}} = {m24.peak_time_analytic():.6f}\ \mathrm{{s}}", ACCENT),
                      rf"N_B(t_{{max}}) = {float(m24.bateman_b(m24.peak_time_analytic())):.6f}"],
            acap="Con λ_A > λ_B la cadena no está en equilibrio secular.",
            numeric="RK4 del sistema de tres ecuaciones y Monte Carlo (50 000 tiempos exponenciales) para la vida media de B.",
            xr=(0, 20), yr=(0, 1.05), xticks=(0, 5, 10, 15, 20), yticks=(0, 0.5, 1.0), xlabel="t [s]", ylabel="N / N₀",
            curves=[(t, m24.n_a(t), ERROR), (t, m24.bateman_b(t), ANALYTIC)], dots=(tt[::200], y[::200, 1]), legend=[(ERROR, "N_A (analítico)"), (ANALYTIC, "N_B (Bateman)"), (NUMERIC, "N_B (RK4)")],
            table=(["magnitud", "numérico", "analítico"],
                   [("t_max [s]", f"{m24.peak_time_numeric():.6f}", f"{m24.peak_time_analytic():.6f}"), ("N_B(20 s)", f"{y[-1, 1]:.8f}", f"{float(m24.bateman_b(20.0)):.8f}"),
                    ("vida media B [s]", f"{mean:.3f} ± {err:.3f}", f"{1 / m24.LAM_B:.3f}")]),
            final="El Monte Carlo concuerda con 1/λ_B dentro de su incertidumbre estadística.", fsize=30, asize=32, dot_every=1)


class BindingEnergy(PhysicsScene):
    volume = 2
    chapter_number = 24
    chapter_title = CT

    def construct(self):
        a = np.linspace(20, 238, 200)
        an = np.arange(20, 239, 12)
        bn = [m24.b_per_a(int(x), False) for x in an]
        za, zn = m24.z_stable_analytic(56), m24.z_stable_numeric(56, False)
        curve_scene(
            self,
            found=[r"B = a_VA - a_SA^{2/3} - a_C\dfrac{Z(Z-1)}{A^{1/3}} - a_A\dfrac{(A-2Z)^2}{A} + \delta", r"\dfrac{\partial B}{\partial Z} = 0\ \Rightarrow\ Z_0 = \dfrac{A/2}{1 + a_CA^{2/3}/4a_A}"],
            note="La competencia entre los términos de superficie y de Coulomb produce el máximo de B/A cerca del hierro y el níquel.",
            fcap="Fórmula semiempírica de masa.",
            problem=["Con los coeficientes de la fórmula semiempírica (sin apareamiento), halle el núcleo estable para A = 56 y la curva B/A en función de A.",
                     "Identifique la región de máxima energía de enlace por nucleón."],
            pcap="Se comparan el mínimo continuo y la búsqueda sobre Z entero.",
            analytic=[(rf"Z_0(56) = {za:.4f},\qquad B/A\ \text{{en el máximo}}\ \approx {float(np.max(m24.b_per_a_analytic(a))):.3f}\ \mathrm{{MeV}}", ACCENT)],
            acap="Los núcleos más ligados están alrededor de A ≈ 56–62.",
            numeric="Para cada A se evalúa B para todos los Z enteros y se toma el máximo; el máximo en A se busca sobre 20 ≤ A ≤ 239.",
            xr=(20, 240), yr=(6, 9), xticks=(20, 80, 140, 200), yticks=(6, 7, 8, 9), xlabel="A", ylabel="B/A [MeV]",
            curves=[(a, m24.b_per_a_analytic(a), ANALYTIC)], dots=(an, bn), legend=[(ANALYTIC, "Z continuo"), (NUMERIC, "Z entero")],
            table=(["magnitud", "numérico", "analítico"],
                   [("Z estable (A=56)", f"{zn}", f"{za:.3f}"), ("B/A(56) [MeV]", f"{m24.b_per_a(56, False):.4f}", f"{float(m24.b_per_a_analytic(56)):.4f}"), ("A del máximo", f"{m24.peak_a_numeric()}", f"{int(np.argmax(m24.b_per_a_analytic(np.arange(20, 240)))) + 20}")]),
            final="Con enteros y continuo, B/A difiere solo en la cuarta cifra; el máximo está en A ≈ 58.", fsize=30)
