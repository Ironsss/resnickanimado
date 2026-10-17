"""Volumen 2, capitulo 14. Ecuaciones de Maxwell."""
from __future__ import annotations

import numpy as np

from animations.common import ACCENT, ANALYTIC, ERROR, NUMERIC, PhysicsScene, curve_scene
from physics import v2ch14_maxwell as m14
from physics.constants import C_LIGHT


class YeeFDTD(PhysicsScene):
    volume = 2
    chapter_number = 14
    chapter_title = "Ecuaciones de Maxwell y ondas electromagnéticas"

    def construct(self):
        x, ez, peaks = m14.yee_pulse()
        dx = 1e-3
        n = len(x)
        t_end = peaks[-1, 0]
        x0 = 0.2 * n * dx
        xg = np.linspace(0, x[-1], 400)
        xc = x0 + C_LIGHT * t_end
        sg = 20 * dx
        sel = np.arange(300, 800, 12)
        curve_scene(
            self,
            found=[r"\nabla\times\vec E = -\dfrac{\partial\vec B}{\partial t},\qquad \nabla\times\vec B = \mu_0\vec J + \mu_0\varepsilon_0\dfrac{\partial\vec E}{\partial t}",
                   r"\dfrac{\partial^2 E}{\partial t^2} = \dfrac{1}{\mu_0\varepsilon_0}\dfrac{\partial^2 E}{\partial x^2}\ \Rightarrow\ c = \dfrac{1}{\sqrt{\mu_0\varepsilon_0}}"],
            note="La corriente de desplazamiento cierra las ecuaciones y predice ondas electromagnéticas con velocidad c.",
            fcap="Ecuaciones de Maxwell.",
            problem=["Un pulso gaussiano (σ = 2 cm) viaja en el vacío en la dirección +x. Halle c a partir de μ₀ y ε₀",
                     "y mida su velocidad con un esquema de diferencias finitas de Yee (Courant 0.5, 600 pasos)."],
            pcap="E y B se alternan en una malla escalonada.",
            analytic=[(rf"c = (\mu_0\varepsilon_0)^{{-1/2}} = {m14.c_from_constants():.6e}\ \mathrm{{m/s}}", ACCENT),
                      rf"x_{{pico}}(t) = x_0 + ct,\qquad x_{{pico}}(t_f) = {xc:.4f}\ \mathrm{{m}}"],
            acap="La solución es f(x − ct).",
            numeric="FDTD de Yee: dos actualizaciones de salto de rana (B desde E y E desde B), y regresión lineal del máximo del pulso.",
            xr=(0, 1.2), yr=(-0.1, 1.15), xticks=(0, 0.4, 0.8, 1.2), yticks=(0, 0.5, 1.0), xlabel="x [m]", ylabel="E / E₀",
            curves=[(xg, np.exp(-((xg - xc) ** 2) / (2 * sg**2)), ANALYTIC)], dots=(x[sel], ez[sel]),
            legend=[(ANALYTIC, "f(x − ct)"), (NUMERIC, "Yee t = t_f")],
            table=(["magnitud", "numérico", "analítico"],
                   [("c [10⁸ m/s]", f"{m14.speed_numeric(0.5) / 1e8:.6f}", f"{m14.c_from_constants() / 1e8:.6f}"),
                    ("c, Courant 0.9", f"{m14.speed_numeric(0.9) / 1e8:.6f}", f"{m14.c_from_constants() / 1e8:.6f}")]),
            final="El esquema numérico propaga el pulso a la velocidad c que predice Maxwell.", fsize=32)


class PoyntingDisplacement(PhysicsScene):
    volume = 2
    chapter_number = 14
    chapter_title = "Ecuaciones de Maxwell y ondas electromagnéticas"

    def construct(self):
        e = np.linspace(0, 200, 100)
        en = np.array([20, 50, 80, 110, 140, 170, 200], dtype=float)
        ia = m14.intensity_analytic()
        curve_scene(
            self,
            found=[r"\vec S = \dfrac{1}{\mu_0}\vec E\times\vec B,\qquad I = \langle S\rangle = \tfrac12 c\varepsilon_0 E_0^2,\qquad p_{rad} = \dfrac{I}{c}",
                   r"I_d = \varepsilon_0\dfrac{d\Phi_E}{dt}"],
            note="El vector de Poynting da el flujo de energía; la corriente de desplazamiento equivale a la de conducción en un capacitor.",
            fcap="Energía y momento de la onda, y corriente de desplazamiento.",
            problem=["Una onda plana con E₀ = 100 V/m incide sobre un absorbente perfecto. Halle la intensidad y la presión de radiación.",
                     "Un capacitor de placas de 100 cm² separadas 1.0 mm se carga a dV/dt = 10⁶ V/s: halle la corriente de desplazamiento."],
            pcap="B = E/c en la onda.",
            analytic=[(rf"I = \tfrac12 c\varepsilon_0E_0^2 = {ia:.4f}\ \mathrm{{W/m^2}},\qquad p = I/c = {m14.radiation_pressure(ia) * 1e9:.4f}\ \mathrm{{nPa}}", ACCENT),
                      rf"I_d = \varepsilon_0 A\,\dfrac{{dV/dt}}{{d}} = {m14.conduction_current_analytic() * 1e6:.4f}\ \mu\mathrm{{A}}"],
            acap="Reflector perfecto: la presión se duplica.",
            numeric="El promedio temporal de E×B/μ₀ se integra con Simpson en un periodo; la corriente de desplazamiento se deriva con diferencias centrales.",
            xr=(0, 200), yr=(0, 55), xticks=(0, 100, 200), yticks=(0, 25, 50), xlabel="E₀ [V/m]", ylabel="I [W/m²]",
            curves=[(e, m14.intensity_analytic(e), ANALYTIC)], dots=(en, [m14.intensity_numeric(float(v)) for v in en]),
            legend=[(ANALYTIC, "½ c ε₀ E₀²"), (NUMERIC, "⟨E×B⟩/μ₀")],
            table=(["magnitud", "numérico", "analítico"],
                   [("I [W/m²]", f"{m14.intensity_numeric():.6f}", f"{ia:.6f}"), ("I_d [µA]", f"{m14.displacement_current_numeric() * 1e6:.5f}", f"{m14.conduction_current_analytic() * 1e6:.5f}"),
                    ("p absorb. [nPa]", "n.a.", f"{m14.radiation_pressure(ia) * 1e9:.4f}"), ("p reflect. [nPa]", "n.a.", f"{m14.radiation_pressure(ia, False) * 1e9:.4f}")]),
            final="La intensidad crece como E₀² y I_d reproduce la corriente de conducción.", fsize=32)
