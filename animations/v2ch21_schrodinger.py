"""Volumen 2, capitulo 21. Ecuacion de Schrodinger."""
from __future__ import annotations

import numpy as np

from animations.common import ACCENT, ANALYTIC, NUMERIC, PhysicsScene, curve_scene
from physics import v2ch21_schrodinger as m21

CT = "Ecuación de Schrödinger"


class InfiniteWell(PhysicsScene):
    volume = 2
    chapter_number = 21
    chapter_title = CT

    def construct(self):
        e, x, v = m21.well_numeric(4)
        xa = np.linspace(0, 1e-9, 300)
        psi_n = v[:, 2] * np.sign(np.sum(v[:, 2] * m21.well_wavefunction(3, x)))
        sel = np.arange(40, 2000, 80)
        ea = m21.well_energy_analytic([1, 2, 3, 4])
        curve_scene(
            self,
            found=[r"-\dfrac{\hbar^2}{2m}\dfrac{d^2\psi}{dx^2} + V\psi = E\psi", r"E_n = \dfrac{n^2\pi^2\hbar^2}{2mL^2},\qquad \psi_n = \sqrt{\dfrac{2}{L}}\sin\dfrac{n\pi x}{L}"],
            note="Las condiciones de contorno ψ(0) = ψ(L) = 0 cuantizan la energía.",
            fcap="Partícula en un pozo infinito.",
            problem=["Un electrón está confinado en un pozo infinito de L = 1.0 nm.",
                     "Halle los cuatro primeros niveles de energía y la función de onda del tercer estado."],
            pcap="Se diagonaliza el hamiltoniano discretizado.",
            analytic=[(rf"E_n = {ea[0]:.5f}\,n^2\ \mathrm{{eV}}:\quad E_1 = {ea[0]:.5f},\ E_2 = {ea[1]:.5f},\ E_3 = {ea[2]:.5f},\ E_4 = {ea[3]:.5f}\ \mathrm{{eV}}", ACCENT)],
            acap="La energía crece como n².",
            numeric="Hamiltoniano de diferencias finitas (matriz tridiagonal de 2000 nodos) diagonalizado con eigh_tridiagonal; se normaliza ∫|ψ|² = 1.",
            xr=(0, 1), yr=(-1.8, 1.8), xticks=(0, 0.5, 1.0), yticks=(-1, 0, 1), xlabel="x / L", ylabel="ψ₃ √L",
            curves=[(xa * 1e9, m21.well_wavefunction(3, xa) * np.sqrt(1e-9), ANALYTIC)], dots=(x[sel] * 1e9, psi_n[sel] * np.sqrt(1e-9)),
            legend=[(ANALYTIC, "√(2/L) sen(3πx/L)"), (NUMERIC, "diagonalización")],
            table=(["nivel", "numérico [eV]", "analítico [eV]"], [(f"E_{n + 1}", f"{e[n]:.6f}", f"{ea[n]:.6f}") for n in range(4)] + [("traslape ψ₁", f"{m21.wavefunction_overlap(1):.8f}", "1")]),
            final="Los autovalores concuerdan con n²π²ħ²/2mL² con error del orden de 10⁻⁶.", fsize=32, asize=30)


class BarrierTunneling(PhysicsScene):
    volume = 2
    chapter_number = 21
    chapter_title = CT

    def construct(self):
        e = np.linspace(0.5, 15, 200)
        en = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], dtype=float)
        ta = m21.transmission_analytic()
        curve_scene(
            self,
            found=[r"T = \left[1 + \dfrac{V_0^2\sinh^2\kappa a}{4E(V_0 - E)}\right]^{-1},\quad \kappa = \dfrac{\sqrt{2m(V_0-E)}}{\hbar}\ \ (E<V_0)",
                   r"T = \left[1 + \dfrac{V_0^2\sin^2 qa}{4E(E - V_0)}\right]^{-1},\quad q = \dfrac{\sqrt{2m(E-V_0)}}{\hbar}\ \ (E>V_0)"],
            note="Para E < V₀ la onda decae en la barrera pero no se anula: hay transmisión por efecto túnel.",
            fcap="Efecto túnel a través de una barrera rectangular.",
            problem=["Un electrón de 5 eV incide sobre una barrera de 10 eV y 0.50 nm de ancho.",
                     "Halle la probabilidad de transmisión y su dependencia con la energía."],
            pcap="Se supone incidencia desde la izquierda.",
            analytic=[(rf"\kappa = {np.sqrt(2 * 9.1093837015e-31 * 5 * 1.602176634e-19) / 1.054571817e-34 * 1e-9:.4f}\ \mathrm{{nm^{{-1}}}},\qquad T(5\ \mathrm{{eV}}) = {ta:.4e}", ACCENT),
                      rf"T(12\ \mathrm{{eV}}) = {m21.transmission_analytic(12.0):.5f}\ \ (\text{{resonancias para }}E>V_0)"],
            acap="Por encima de la barrera hay reflexión parcial y resonancias de transmisión.",
            numeric="RK4 de la ecuación de Schrödinger estacionaria integrada desde la derecha (solo onda transmitida) hasta x = 0, donde se descompone en incidente y reflejada.",
            xr=(0, 15), yr=(0, 1.05), xticks=(0, 5, 10, 15), yticks=(0, 0.5, 1.0), xlabel="E [eV]", ylabel="T",
            curves=[(e, [m21.transmission_analytic(float(x)) for x in e], ANALYTIC)], dots=(en, [m21.transmission_numeric(float(x)) for x in en]),
            legend=[(ANALYTIC, "fórmula analítica"), (NUMERIC, "integración RK4")],
            table=(["magnitud", "numérico", "analítico"],
                   [("T(5 eV)", f"{m21.transmission_numeric(5.0):.6e}", f"{ta:.6e}"), ("T(8 eV)", f"{m21.transmission_numeric(8.0):.6e}", f"{m21.transmission_analytic(8.0):.6e}"),
                    ("T(12 eV)", f"{m21.transmission_numeric(12.0):.6e}", f"{m21.transmission_analytic(12.0):.6e}")]),
            final="La integración numérica reproduce la transmisión analítica, incluidas las resonancias.", fsize=28)
