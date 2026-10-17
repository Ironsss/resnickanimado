"""Volumen 2, capitulo 12. Propiedades magneticas de la materia."""
from __future__ import annotations

import numpy as np

from animations.common import ACCENT, ANALYTIC, NUMERIC, PhysicsScene, curve_scene
from physics import v2ch12_magnetic_matter as m12


class DipoleTorque(PhysicsScene):
    volume = 2
    chapter_number = 12
    chapter_title = "Propiedades magnéticas de la materia"

    def construct(self):
        ang = np.array([5, 15, 30, 45, 60, 75, 90, 120, 150, 170], dtype=float)
        ag = np.linspace(2, 175, 120)
        t0 = 2 * np.pi / m12.omega_small()
        curve_scene(
            self,
            found=[r"\vec\tau = \vec\mu\times\vec B,\qquad U = -\vec\mu\cdot\vec B", r"I\ddot\theta = -\mu B\sin\theta\ \Rightarrow\ T = \dfrac{T_0}{\mathrm{AGM}(1,\cos(\theta_0/2))}"],
            note="Para ángulos pequeños el dipolo oscila armónicamente con ω² = μB/I; para amplitudes grandes el periodo crece.",
            fcap="Dipolo magnético en un campo uniforme.",
            problem=["Una aguja imanada (μ = 0.10 A·m², I = 1.0×10⁻⁶ kg·m²) oscila en el campo terrestre B = 50 µT.",
                     "Halle el periodo para amplitud pequeña y su dependencia con la amplitud."],
            pcap="El campo es horizontal y no hay fricción.",
            analytic=[(rf"\omega = \sqrt{{\mu B/I}} = {m12.omega_small():.4f}\ \mathrm{{rad/s}},\qquad T_0 = {t0:.4f}\ \mathrm{{s}}", ACCENT),
                      r"T(\theta_0) = \dfrac{T_0}{\mathrm{AGM}\left(1,\cos(\theta_0/2)\right)}\ \text{(media aritmético-geométrica)}"],
            acap="Solución exacta del péndulo no lineal.",
            numeric="RK4 de la ecuación no lineal (40000 pasos); el periodo se mide entre dos cruces descendentes por cero.",
            xr=(0, 180), yr=(1.0, 4.0), xticks=(0, 60, 120, 180), yticks=(1, 2, 3, 4), xlabel="θ₀ [°]", ylabel="T / T₀",
            curves=[(ag, np.array([m12.period_exact(a) for a in ag]) / t0, ANALYTIC)], dots=(ang, [m12.period_numeric(float(a)) / t0 for a in ang]),
            legend=[(ANALYTIC, "T₀/AGM"), (NUMERIC, "RK4")],
            table=(["magnitud", "numérico", "analítico"],
                   [("T(30°) [s]", f"{m12.period_numeric(30.0):.5f}", f"{m12.period_exact(30.0):.5f}"),
                    ("T(90°) [s]", f"{m12.period_numeric(90.0):.5f}", f"{m12.period_exact(90.0):.5f}"),
                    ("deriva de E", f"{m12.energy_conservation_drift():.1e}", "0")]),
            final="El periodo exacto crece sin límite cuando θ₀ → 180°.", fsize=32)


class ParamagnetCurie(PhysicsScene):
    volume = 2
    chapter_number = 12
    chapter_title = "Propiedades magnéticas de la materia"

    def construct(self):
        xs = np.array([0.1, 0.25, 0.5, 0.8, 1.2, 1.6, 2.0, 2.6, 3.2])
        xg = np.linspace(0, 3.4, 120)
        mm = [m12.magnetization_metropolis(float(x)) for x in xs]
        x300 = m12.x_param(1.0, 300.0)
        curve_scene(
            self,
            found=[r"U = \mp\mu_B B,\qquad \dfrac{M}{M_s} = \tanh\!\left(\dfrac{\mu_B B}{kT}\right)", r"\dfrac{M}{M_s}\approx \dfrac{\mu_B B}{kT}\ \ (\text{ley de Curie, } x \ll 1)"],
            note="La agitación térmica compite con la alineación: la magnetización satura solo para μB ≫ kT.",
            fcap="Paramagnetismo de espín 1/2.",
            problem=["Un paramagneto ideal de espines independientes con momento μ_B está en B y T dados.",
                     "Halle M/M_s en función de x = μ_B B/kT y verifique la ley de Curie a campo débil."],
            pcap="Para B = 1 T y T = 300 K se tiene x ≈ 2×10⁻³.",
            analytic=[(r"\dfrac{M}{M_s} = \tanh x", ACCENT), rf"x(1\ \mathrm{{T}},\,300\ \mathrm{{K}}) = {x300:.4e}\ \Rightarrow\ M/M_s \approx {np.tanh(x300):.4e}"],
            acap="Distribución de Boltzmann de dos niveles.",
            numeric="Monte Carlo de Metropolis con 100 000 espines y 80 barridos; la incertidumbre estadística es 1/√N de muestras.",
            xr=(0, 3.4), yr=(0, 1.05), xticks=(0, 1, 2, 3), yticks=(0, 0.5, 1.0), xlabel="x = μB/kT", ylabel="M / Ms",
            curves=[(xg, m12.magnetization_analytic(xg), ANALYTIC)], dots=(xs, [v[0] for v in mm]),
            legend=[(ANALYTIC, "tanh x"), (NUMERIC, "Metropolis")],
            table=(["x", "Metropolis", "tanh x"],
                   [(f"{xs[i]:.2f}", f"{mm[i][0]:.4f} ± {mm[i][1]:.4f}", f"{np.tanh(xs[i]):.4f}") for i in (0, 3, 5, 8)]),
            final="Los resultados de Metropolis concuerdan con tanh x dentro de la incertidumbre.", fsize=34)
