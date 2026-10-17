"""Volumen 2, capitulo 20. Ondas de materia."""
from __future__ import annotations

import numpy as np

from animations.common import ACCENT, ANALYTIC, NUMERIC, PhysicsScene, curve_scene
from physics import v2ch20_matter_waves as m20

CT = "Ondas de materia"


class ElectronInterference(PhysicsScene):
    volume = 2
    chapter_number = 20
    chapter_title = CT

    def construct(self):
        lam = m20.de_broglie()
        sp = m20.fringe_spacing_analytic()
        hits = m20.electron_hits()
        span = 12 * sp
        edges = np.linspace(-span / 2, span / 2, 600 + 1)
        # histograma agrupado en 120 contenedores
        hist, e2 = np.histogram(hits, bins=120, range=(-span / 2, span / 2))
        centers = 0.5 * (e2[1:] + e2[:-1])
        norm = hist.max()
        y = np.linspace(-span / 4, span / 4, 400)
        est, _ = m20.fringe_spacing_from_hits(hits)
        sel = slice(30, 90)  # bins dentro de +-span/4
        curve_scene(
            self,
            found=[r"\lambda = \dfrac{h}{p},\qquad p = \sqrt{K(K+2mc^2)}/c", r"\Delta y = \dfrac{\lambda L}{d}\qquad(\text{franjas de interferencia de electrones})"],
            note="Cada electrón llega como una partícula localizada; la distribución de muchos reproduce el patrón ondulatorio.",
            fcap="Hipótesis de De Broglie e interferencia de electrones.",
            problem=["Electrones de 100 eV atraviesan dos rendijas separadas 1.0 µm y se detectan en una pantalla a 1.0 m.",
                     "Halle λ y el espaciado de las franjas; luego simule 20 000 detecciones individuales."],
            pcap="Cada electrón se muestrea de I(y) por rechazo.",
            analytic=[(rf"\lambda = {lam * 1e12:.4f}\ \mathrm{{pm}}", ACCENT), rf"\Delta y = \dfrac{{\lambda L}}{{d}} = {sp * 1e6:.3f}\ \mu\mathrm{{m}}"],
            acap="Aproximación de ángulos pequeños.",
            numeric="Monte Carlo de 20 000 electrones; el espaciado se estima con la frecuencia espacial dominante del histograma (FFT con relleno de ceros).",
            xr=(-span * 1e6 / 4, span * 1e6 / 4), yr=(0, 1.15), xticks=(-300, 0, 300), yticks=(0, 0.5, 1.0), xlabel="y [µm]", ylabel="cuentas (norm.)",
            curves=[(y * 1e6, np.cos(np.pi * m20.D_SLIT * y / (lam * m20.L_SCR)) ** 2, ANALYTIC)], dots=(centers[sel] * 1e6, hist[sel] / norm),
            legend=[(ANALYTIC, "cos²(πdy/λL)"), (NUMERIC, "histograma MC")],
            table=(["magnitud", "numérico", "analítico"],
                   [("λ [pm]", "n.a.", f"{lam * 1e12:.4f}"), ("Δy [µm]", f"{est * 1e6:.3f}", f"{sp * 1e6:.3f}"), ("Bragg θ [°]", f"{m20.bragg_angle_numeric():.5f}", f"{m20.bragg_angle_analytic():.5f}")]),
            final="El espaciado de las franjas medido sobre el histograma coincide con λL/d.", fsize=32)


class WavePacketUncertainty(PhysicsScene):
    volume = 2
    chapter_number = 20
    chapter_title = CT

    def construct(self):
        t = np.linspace(0, 1e-15, 100)
        tn = np.linspace(0, 1e-15, 11)
        sx0, sp0 = m20.packet_numeric(0.0)
        curve_scene(
            self,
            found=[r"\Delta x\,\Delta p \ge \dfrac{\hbar}{2}", r"\sigma(t) = \sigma_0\sqrt{1 + \left(\dfrac{\hbar t}{2m\sigma_0^2}\right)^2}\qquad(\text{paquete gaussiano libre})"],
            note="El paquete gaussiano minimiza el producto de incertidumbres; su ensanchamiento temporal es consecuencia de la dispersión.",
            fcap="Principio de incertidumbre y paquetes de onda.",
            problem=["Un electrón libre está descrito por un paquete gaussiano con σ₀ = 0.100 nm.",
                     "Halle Δx Δp y cómo se ensancha el paquete con el tiempo."],
            pcap="Se evoluciona exactamente en el espacio de momentos.",
            analytic=[(rf"\Delta x\,\Delta p = \hbar/2 = {m20.HBAR / 2:.5e}\ \mathrm{{J\,s}}", ACCENT), rf"\sigma(0.5\ \mathrm{{fs}}) = {float(m20.packet_width_analytic(0.5e-15)) * 1e10:.5f}\ \text{{Å}}"],
            acap="Con σ₀ = 1 Å el tiempo característico es 2mσ₀²/ħ ≈ 0.17 fs.",
            numeric="FFT de 4096 puntos: se aplica la fase e^(−iħk²t/2m) en el espacio de momentos, se regresa al espacio y se calculan las desviaciones estándar.",
            xr=(0, 1), yr=(0, 6.5), xticks=(0, 0.5, 1.0), yticks=(0, 2, 4, 6), xlabel="t [fs]", ylabel="σ(t) [Å]",
            curves=[(t * 1e15, m20.packet_width_analytic(t) * 1e10, ANALYTIC)], dots=(tn * 1e15, [m20.packet_numeric(float(x))[0] * 1e10 for x in tn]),
            legend=[(ANALYTIC, "σ₀√(1 + (ħt/2mσ₀²)²)"), (NUMERIC, "evolución por FFT")],
            table=(["magnitud", "numérico", "analítico"],
                   [("Δx Δp / (ħ/2)", f"{sx0 * sp0 / (m20.HBAR / 2):.8f}", "1"), ("σ(1 fs) [Å]", f"{m20.packet_numeric(1e-15)[0] * 1e10:.6f}", f"{float(m20.packet_width_analytic(1e-15)) * 1e10:.6f}"),
                    ("σ(0.5 fs) [Å]", f"{m20.packet_numeric(0.5e-15)[0] * 1e10:.6f}", f"{float(m20.packet_width_analytic(0.5e-15)) * 1e10:.6f}")]),
            final="El ancho en el espacio de momentos no cambia; el producto crece solo por el ensanchamiento.", fsize=32)
