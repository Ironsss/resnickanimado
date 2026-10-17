"""Volumen 2, capitulo 23. Conduccion electrica en solidos."""
from __future__ import annotations

import numpy as np

from animations.common import ACCENT, ANALYTIC, ERROR, NUMERIC, PhysicsScene, curve_scene
from physics import v2ch23_solids as m23

CT = "Conducción eléctrica en sólidos"


class BandStructure(PhysicsScene):
    volume = 2
    chapter_number = 23
    chapter_title = CT

    def construct(self):
        k = np.linspace(0.02, np.pi - 0.02, 40)
        kn = np.linspace(0.02, np.pi - 0.02, 12)
        lim = m23.band_limits_analytic(2)
        num_edges = m23.band_edges_numeric(3)
        gap = lim[1][0] - lim[0][1]
        pw = np.array([m23.bands_plane_waves(float(x), 2) for x in kn])
        curve_scene(
            self,
            found=[r"\cos ka = \cos qa + \dfrac{mg}{\hbar^2 q}\sin qa,\qquad q = \dfrac{\sqrt{2mE}}{\hbar}\qquad(\text{Kronig-Penney})",
                   r"\text{las energías permitidas cumplen } |\cos qa + (mg/\hbar^2q)\sin qa|\le 1"],
            note="El potencial periódico abre brechas de energía en los bordes de la zona de Brillouin: es el origen de las bandas.",
            fcap="Teoría de bandas.",
            problem=["Un cristal unidimensional se modela con un peine de deltas de intensidad g = 4 (unidades atómicas, a = 1).",
                     "Halle las dos primeras bandas, la brecha entre ellas y la relación E(k)."],
            pcap="Se compara con la diagonalización en ondas planas.",
            analytic=[(rf"\text{{banda 1}}:\ [{lim[0][0]:.4f},\,{lim[0][1]:.4f}],\qquad \text{{banda 2}}:\ [{lim[1][0]:.4f},\,{lim[1][1]:.4f}]\ \text{{Ha}}", ACCENT), rf"E_{{gap}} = {gap:.4f}\ \mathrm{{Ha}}"],
            acap="E(k) se obtiene resolviendo la relación de dispersión para cada k.",
            numeric="Hamiltoniano en una base de 201 ondas planas e^{i(k+2πm)x}, con V_mm′ = g; se diagonaliza para cada k.",
            xr=(0, 3.2), yr=(0, 22), xticks=(0, 1.57, 3.14), yticks=(0, 5, 10, 15, 20), xlabel="k a", ylabel="E [Ha]",
            curves=[(k, [m23.band_energy_analytic(float(x), 0) for x in k], ANALYTIC), (k, [m23.band_energy_analytic(float(x), 1) for x in k], ERROR)],
            dots=(np.concatenate([kn, kn]), np.concatenate([pw[:, 0], pw[:, 1]])), legend=[(ANALYTIC, "banda 1"), (ERROR, "banda 2"), (NUMERIC, "ondas planas")],
            table=(["borde", "numérico", "analítico"], [(f"E{i + 1} [Ha]", f"{num_edges[i]:.4f}", f"{m23.band_edges_analytic(3)[i]:.4f}") for i in range(4)]),
            final="La convergencia en ondas planas es lenta (≈1/N) porque el potencial es singular.", fsize=30, asize=30)


class DiodeCircuit(PhysicsScene):
    volume = 2
    chapter_number = 23
    chapter_title = CT

    def construct(self):
        e = np.linspace(0.6, 10, 100)
        en = np.array([0.6, 1, 2, 3, 4, 5, 6, 8, 10])
        va = [m23.diode_voltage_analytic(float(x)) for x in e]
        vn = [m23.diode_voltage_numeric(float(x)) for x in en]
        curve_scene(
            self,
            found=[r"I = I_s\left(e^{V_d/nV_T} - 1\right)\qquad(\text{ecuación de Shockley})", r"E = V_d + IR\ \Rightarrow\ V_d = E + I_sR - nV_T\,W\!\left(\dfrac{I_sR}{nV_T}e^{(E+I_sR)/nV_T}\right)"],
            note="La función W de Lambert da la solución exacta del diodo en serie con una resistencia.",
            fcap="Unión p-n y diodo en un circuito.",
            problem=["Un diodo (I_s = 1 pA, n = 1, V_T = 25.85 mV) está en serie con R = 100 Ω y una fuente E.",
                     "Halle la tensión V_d en el diodo en función de E y la corriente para E = 5 V."],
            pcap="Temperatura de 300 K.",
            analytic=[(rf"V_d(5\ \mathrm{{V}}) = {m23.diode_voltage_analytic(5.0):.6f}\ \mathrm{{V}}", ACCENT), rf"I(5\ \mathrm{{V}}) = {m23.diode_current(5.0, m23.diode_voltage_analytic(5.0)) * 1e3:.4f}\ \mathrm{{mA}}"],
            acap="V_d varía solo logarítmicamente con E.",
            numeric="Bisección de f(V) = I_s(e^{V/nV_T} − 1) − (E − V)/R sobre [0, E].",
            xr=(0, 10), yr=(0.4, 0.8), xticks=(0, 2, 4, 6, 8, 10), yticks=(0.4, 0.6, 0.8), xlabel="E [V]", ylabel="V_d [V]",
            curves=[(e, va, ANALYTIC)], dots=(en, vn), legend=[(ANALYTIC, "función W de Lambert"), (NUMERIC, "bisección")],
            table=(["E [V]", "numérico [V]", "analítico [V]"], [(f"{x:g}", f"{m23.diode_voltage_numeric(x):.8f}", f"{m23.diode_voltage_analytic(x):.8f}") for x in (1.0, 5.0, 10.0)]),
            final="La solución exacta y la numérica coinciden a ~10⁻¹⁰ V.", fsize=30, asize=32)
