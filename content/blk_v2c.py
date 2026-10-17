"""Volumen 2, capitulos 15 a 25."""
from __future__ import annotations

import numpy as np

from content.model import Chapter, Row, Scene
from physics import v2ch15_light as m15
from physics import v2ch16_lenses as m16
from physics import v2ch17_interference as m17
from physics import v2ch18_diffraction as m18
from physics import v2ch19_photons as m19
from physics import v2ch20_matter_waves as m20
from physics import v2ch21_schrodinger as m21
from physics import v2ch22_atom as m22
from physics import v2ch23_solids as m23
from physics import v2ch24_nuclear as m24
from physics import v2ch25_particles as m25

OPT, QM = "Óptica", "Física cuántica y moderna"


def _ch15() -> Chapter:
    d = m15.target_point()[0]
    xn = m15.fermat_numeric(d)
    th1 = np.deg2rad(m15.THETA1_DEG)
    th2 = np.degrees(m15.snell_angle(th1))
    dn = {x: m15.min_deviation_numeric(float(m15.cauchy(x))) for x in (400, 550, 700)}
    da = {x: float(m15.min_deviation_analytic(m15.cauchy(x))) for x in dn}
    s1 = Scene(
        id="SnellFermat", title="Refracción: ley de Snell y principio de Fermat",
        statement="Un rayo sale de un punto a 1.0 m sobre una interfaz aire-vidrio (n = 1.5) con incidencia de 40° y llega a un punto a 1.5 m bajo ella. Hallar el ángulo refractado, el punto de incidencia y el ángulo crítico vidrio-aire.",
        foundations=[(r"n_1\sin\theta_1=n_2\sin\theta_2", "Ley de Snell."), (r"T=\dfrac1c(n_1s_1+n_2s_2)", "Tiempo de viaje, estacionario para el camino real (Fermat).")],
        analytic=[(rf"\theta_2={th2:.4f}^\circ,\quad x^*=h_1\tan\theta_1={m15.refraction_point_analytic():.5f}\ \mathrm{{m}}", "Refracción."), (rf"\theta_c=\arcsin(1/1.5)={m15.critical_angle_analytic():.4f}^\circ", "Ángulo crítico.")],
        numeric_method="Se minimiza T(x) por búsqueda de la sección áurea (200 iteraciones); el ángulo crítico se halla por bisección sobre el ángulo de incidencia.",
        rows=[Row("Punto de incidencia", "x^*", m15.refraction_point_analytic(), xn, "m", 6), Row("Ángulo crítico", r"\theta_c", m15.critical_angle_analytic(), m15.critical_angle_numeric(), "°", 8)],
    )
    s2 = Scene(
        id="PrismDispersion", title="Dispersión en un prisma",
        statement="Un prisma de ángulo 60° tiene n = 1.50 + 5000/λ² (λ en nm). Hallar la desviación mínima entre 400 y 700 nm.",
        foundations=[(r"n(\lambda)=A+B/\lambda^2", "Dispersión de Cauchy."), (r"\delta_{min}=2\arcsin\!\left(n\sin\tfrac\alpha2\right)-\alpha", "Desviación mínima.")],
        analytic=[(rf"\delta_{{min}}(400)={da[400]:.4f}^\circ,\quad\delta_{{min}}(700)={da[700]:.4f}^\circ", "Violeta y rojo."), (rf"\Delta\delta={da[400]-da[700]:.4f}^\circ", "Dispersión angular.")],
        numeric_method="Se traza un rayo por las dos caras con la ley de Snell y se minimiza la desviación sobre el ángulo de incidencia con sección áurea.",
        rows=[Row(f"Desviación mínima en {x} nm", r"\delta_{min}", da[x], dn[x], "°", 7) for x in (400, 550, 700)],
    )
    return Chapter(number=15, volume=2, part=OPT, slug="v2ch15", module="v2ch15_light", title="Naturaleza y propagación de la luz",
                   summary="Velocidad de la luz, reflexión, refracción, ángulo crítico y dispersión.",
                   concepts=["Velocidad de la luz", "Reflexión y refracción", "Principio de Fermat", "Dispersión"], scenes=[s1, s2])


def _ch16() -> Chapter:
    f = m16.focal_lensmaker()
    za, ma = m16.two_lens_analytic()
    zn, mn = m16.two_lens_matrix()
    s1 = Scene(
        id="ThinLensImage", title="Imagen de una lente delgada y de un espejo esférico",
        statement="Una lente biconvexa de vidrio (n = 1.5, R₁ = +10 cm, R₂ = −10 cm) forma la imagen de un objeto axial a 30 cm. Hallar f y s′; resolver también el caso de un espejo cóncavo de R = 40 cm con el objeto a 60 cm.",
        foundations=[(r"\dfrac1f=(n-1)\left(\dfrac1{R_1}-\dfrac1{R_2}\right)", "Ecuación del fabricante."), (r"\dfrac1s+\dfrac1{s'}=\dfrac1f,\quad m=-\dfrac{s'}s", "Ecuación de las lentes delgadas.")],
        analytic=[(rf"f={f*100:.3f}\ \mathrm{{cm}},\quad s'={m16.image_distance_analytic(0.3, f)*100:.4f}\ \mathrm{{cm}}", "Lente."), (rf"s'_{{espejo}}={m16.mirror_image_analytic(0.6, 0.4)*100:.4f}\ \mathrm{{cm}}", "Espejo con f = R/2.")],
        numeric_method="Trazado de un rayo paraxial (altura 1 mm) con la ley de Snell vectorial exacta en cada superficie esférica y reflexión exacta en el espejo; se halla el corte con el eje.",
        rows=[Row("Imagen de la lente", "s'", m16.image_distance_analytic(0.3, f) * 100, m16.lens_image_numeric(0.3) * 100, "cm", 4), Row("Imagen del espejo", "s'", m16.mirror_image_analytic(0.6, 0.4) * 100, m16.mirror_image_numeric(0.6, 0.4) * 100, "cm", 5)],
        notes=["La diferencia en la quinta cifra proviene de la aberración esférica y del espesor finito de la lente."],
    )
    s2 = Scene(
        id="TwoLensSystem", title="Sistema de dos lentes y matrices de rayos",
        statement="Dos lentes delgadas, f₁ = 20 cm y f₂ = 10 cm, separadas 15 cm. Un objeto está a 50 cm de la primera. Hallar la posición y el aumento de la imagen final.",
        foundations=[(r"\dfrac1s+\dfrac1{s'}=\dfrac1f", "Aplicada lente por lente."), (r"M=M_{L_2}M_dM_{L_1}M_s,\quad B=0", "Condición de imagen en el método matricial.")],
        analytic=[(rf"s_1'={m16.image_distance_analytic(0.5, 0.2)*100:.3f}\ \mathrm{{cm}}", "Imagen intermedia."), (rf"s_2'={za*100:.4f}\ \mathrm{{cm}},\quad m={ma:.5f}", "Imagen final y aumento.")],
        numeric_method="Producto de matrices de propagación libre y de lente delgada; el plano imagen se halla imponiendo B = 0 en la matriz total.",
        rows=[Row("Posición de la imagen final", "s_2'", za * 100, zn * 100, "cm", 7), Row("Aumento lateral", "m", ma, mn, "", 7)],
    )
    return Chapter(number=16, volume=2, part=OPT, slug="v2ch16", module="v2ch16_lenses", title="Óptica geométrica: espejos y lentes",
                   summary="Formación de imágenes con espejos esféricos, lentes delgadas y sistemas de lentes.",
                   concepts=["Espejos esféricos", "Lentes delgadas", "Sistemas de lentes", "Matrices de transferencia de rayos"], scenes=[s1, s2])


def _ch17() -> Chapter:
    sp = m17.LAM * m17.L_SCR / m17.D_SLIT
    r0 = m17.bare_reflectance()
    s1 = Scene(
        id="YoungSlits", title="Experimento de Young",
        statement="Dos rendijas separadas 0.25 mm se iluminan con luz de 600 nm y se observan en una pantalla a 1.5 m. Hallar la posición de las franjas brillantes y su espaciado.",
        foundations=[(r"d\sin\theta=m\lambda", "Máximos de interferencia."), (r"I=4I_0\cos^2\!\left(\dfrac{\pi d\sin\theta}\lambda\right)", "Intensidad.")],
        analytic=[(rf"y_m=L\tan\!\left(\arcsin\dfrac{{m\lambda}}d\right),\quad\Delta y\approx\dfrac{{\lambda L}}d={sp*1e3:.4f}\ \mathrm{{mm}}", "Posiciones y espaciado."), (rf"y_3={m17.fringe_position_analytic(3)*1e3:.5f}\ \mathrm{{mm}}", "Tercer máximo.")],
        numeric_method="Suma compleja de dos ondas esféricas con las distancias exactas a cada rendija; las posiciones de las franjas se hallan por bisección sobre r₂ − r₁ = mλ.",
        rows=[Row(f"Franja m = {m}", f"y_{m}", m17.fringe_position_analytic(m) * 1e3, m17.bright_fringe_numeric(m) * 1e3, "mm", 7) for m in (1, 3, 5)],
    )
    s2 = Scene(
        id="ThinFilmCoating", title="Recubrimiento antirreflejo de una película delgada",
        statement="Una película de MgF₂ (n = 1.38) de espesor λ₀/4n con λ₀ = 550 nm recubre vidrio (n = 1.52). Hallar la reflectancia en función de λ y compararla con la del vidrio desnudo.",
        foundations=[(r"R_{vidrio}=\left(\dfrac{n_s-1}{n_s+1}\right)^2", "Reflectancia sin recubrir."), (r"R=\left|\dfrac{r_{01}+r_{12}e^{2i\delta}}{1+r_{01}r_{12}e^{2i\delta}}\right|^2", "Fórmula de Airy.")],
        analytic=[(rf"R_{{vidrio}}={r0*100:.4f}\,\%", "Vidrio desnudo."), (rf"R(550\ \mathrm{{nm}})={float(m17.reflectance_analytic(550e-9))*100:.4f}\,\%", "Con la película.")],
        numeric_method="Método de la matriz de transferencia: matriz característica de la capa y coeficiente de reflexión de la pila completa.",
        rows=[Row(f"Reflectancia a {int(l*1e9)} nm", "R", float(m17.reflectance_analytic(l)) * 100, float(m17.reflectance_matrix(l)[0]) * 100, "%", 8) for l in (450e-9, 550e-9, 650e-9)],
    )
    return Chapter(number=17, volume=2, part=OPT, slug="v2ch17", module="v2ch17_interference", title="Interferencia",
                   summary="Interferencia de dos fuentes, experimento de Young y películas delgadas.",
                   concepts=["Coherencia", "Experimento de Young", "Películas delgadas", "Recubrimientos antirreflejo"], scenes=[s1, s2])


def _ch18() -> Chapter:
    ms, ang = m18.principal_maxima_analytic()
    num = m18.principal_maxima_numeric()
    s1 = Scene(
        id="SlitAndGrating", title="Difracción de una rendija y rejilla de varias rendijas",
        statement="Una rejilla de 5 rendijas de ancho a = 2.0 µm y paso d = 6.0 µm se ilumina con luz de 550 nm. Hallar los ángulos de los máximos principales y el patrón de intensidad.",
        foundations=[(r"I=I_0\,\mathrm{sinc}^2\!\left(\dfrac{\pi a\sin\theta}\lambda\right)", "Rendija única."), (r"I\propto\mathrm{sinc}^2\beta\left(\dfrac{\sin N\gamma}{N\sin\gamma}\right)^2", "N rendijas.")],
        analytic=[(rf"\theta_1={ang[1]:.4f}^\circ,\ \theta_2={ang[2]:.4f}^\circ,\ \theta_4={ang[4]:.4f}^\circ", "d sen θ = mλ."), (r"d/a=3\ \Rightarrow\ \text{el orden 3 queda suprimido}", "Envolvente de difracción.")],
        numeric_method="Suma coherente de 5 rendijas con 200 fuentes puntuales cada una; los máximos son extremos locales de la intensidad sobre una malla de 20 001 ángulos.",
        rows=[Row(f"Máximo de orden {m}", rf"\theta_{m}", float(ang[m]), float(num[m]), "°", 3) for m in (1, 2, 4)],
        notes=["La resolución angular está limitada por la malla de ángulos (0.004°)."],
    )
    s2 = Scene(
        id="PolarizersMalus", title="Polarizadores y ley de Malus",
        statement="Luz polarizada a 0° pasa por dos polarizadores cruzados (0° y 90°) con un tercero intercalado a un ángulo θ. Hallar la transmisión en función de θ y su máximo.",
        foundations=[(r"I=I_0\cos^2\theta", "Ley de Malus."), (r"P(\theta)=\begin{pmatrix}\cos^2\theta&\cos\theta\sin\theta\\\cos\theta\sin\theta&\sin^2\theta\end{pmatrix}", "Matriz de Jones de un polarizador lineal.")],
        analytic=[(r"I/I_0=\cos^2\theta\sin^2\theta=\tfrac14\sin^22\theta", "Tres polarizadores."), (rf"\theta=45^\circ:\ I/I_0={m18.three_polarizers_analytic(np.pi/4):.4f}", "Máximo.")],
        numeric_method="Producto de matrices de Jones aplicado al vector de campo incidente; se evalúa la intensidad como el producto escalar del campo con su conjugado.",
        rows=[Row("Tres polarizadores a 30°", "I/I_0", m18.three_polarizers_analytic(np.pi / 6), m18.jones_intensity([0, np.pi / 6, np.pi / 2]), "", 8), Row("Tres polarizadores a 45°", "I/I_0", m18.three_polarizers_analytic(np.pi / 4), m18.jones_intensity([0, np.pi / 4, np.pi / 2]), "", 8),
              Row("Ley de Malus a 60°", "I/I_0", float(m18.malus_analytic(np.pi / 3)), m18.jones_intensity([0, np.pi / 3]), "", 8)],
    )
    return Chapter(number=18, volume=2, part=OPT, slug="v2ch18", module="v2ch18_diffraction", title="Difracción y polarización",
                   summary="Difracción de Fraunhofer, rejillas y polarización de la luz.",
                   concepts=["Difracción de una rendija", "Rejillas de difracción", "Polarización", "Ley de Malus"], scenes=[s1, s2])


def _ch19() -> Chapter:
    t = 5800.0
    h, phi, dh, dphi = m19.millikan_fit()
    s1 = Scene(
        id="BlackbodySpectrum", title="Radiación de cuerpo negro",
        statement="Un cuerpo negro a 5800 K emite según la ley de Planck. Hallar la longitud de onda del máximo y la potencia emitida por unidad de área.",
        foundations=[(r"u(\lambda,T)=\dfrac{2\pi hc^2}{\lambda^5}\dfrac1{e^{hc/\lambda kT}-1}", "Ley de Planck."), (r"\lambda_{max}T=b,\quad P=\sigma AT^4", "Wien y Stefan-Boltzmann.")],
        analytic=[(rf"\lambda_{{max}}={m19.wien_peak_analytic(t)*1e9:.4f}\ \mathrm{{nm}}", "Máximo."), (rf"P/A={m19.stefan_analytic(t)/1e6:.4f}\ \mathrm{{MW/m^2}}", "Potencia por área.")],
        numeric_method="Sección áurea sobre la función de Planck para el máximo; Simpson (20 000 subintervalos) entre 0.02 λ_max y 200 λ_max para la potencia total.",
        rows=[Row("Máximo a 5800 K", r"\lambda_{max}", m19.wien_peak_analytic(t) * 1e9, m19.wien_peak_numeric(t) * 1e9, "nm", 7), Row("Máximo a 3000 K", r"\lambda_{max}", m19.wien_peak_analytic(3000.0) * 1e9, m19.wien_peak_numeric(3000.0) * 1e9, "nm", 7),
              Row("Potencia por área a 5800 K", "P/A", m19.stefan_analytic(t) / 1e6, m19.stefan_numeric(t) / 1e6, "MW/m²", 5)],
        notes=["La pequeña diferencia en la potencia se debe a la cola del espectro que queda fuera del intervalo de integración."],
    )
    s2 = Scene(
        id="ComptonPhotoelectric", title="Efecto Compton y efecto fotoeléctrico",
        statement="Rayos X de 71 pm se dispersan por electrones libres: hallar Δλ(θ). Con potenciales de frenado sintéticos del sodio (ruido de 10 mV), determinar h y la función de trabajo.",
        foundations=[(r"\Delta\lambda=\dfrac h{m_ec}(1-\cos\theta)", "Efecto Compton."), (r"eV_0=hf-\phi", "Ecuación de Einstein.")],
        analytic=[(rf"\lambda_C={m19.LAMBDA_C*1e12:.5f}\ \mathrm{{pm}},\quad\Delta\lambda(90^\circ)={float(m19.compton_shift_analytic(np.pi/2))*1e12:.5f}\ \mathrm{{pm}}", "Compton."), (rf"h={m19.H_PLANCK:.6e}\ \mathrm{{J\,s}},\quad\phi={m19.PHI_EV:.2f}\ \mathrm{{eV}}", "Valores de entrada de los datos sintéticos.")],
        numeric_method="Conservación relativista de energía y momento: bisección de la energía del fotón dispersado. Ajuste lineal por mínimos cuadrados de V₀(f) con 25 puntos entre 6×10¹⁴ y 12×10¹⁴ Hz.",
        rows=[Row("Corrimiento a 90°", r"\Delta\lambda", float(m19.compton_shift_analytic(np.pi / 2)) * 1e12, m19.compton_shift_numeric(np.pi / 2) * 1e12, "pm", 7), Row("Corrimiento a 180°", r"\Delta\lambda", float(m19.compton_shift_analytic(np.pi)) * 1e12, m19.compton_shift_numeric(np.pi) * 1e12, "pm", 7),
              Row("Constante de Planck ajustada", "h", m19.H_PLANCK * 1e34, h * 1e34, "10⁻³⁴ J s", 3), Row("Función de trabajo ajustada", r"\phi", m19.PHI_EV, phi, "eV", 3)],
        notes=[f"Incertidumbres del ajuste: h ± {dh*1e34:.4f}×10⁻³⁴ J s, φ ± {dphi:.4f} eV."],
    )
    return Chapter(number=19, volume=2, part=QM, slug="v2ch19", module="v2ch19_photons", title="Fotones y naturaleza ondulatoria de la luz",
                   summary="Cuerpo negro, efecto fotoeléctrico y efecto Compton.",
                   concepts=["Radiación de cuerpo negro", "Efecto fotoeléctrico", "Efecto Compton", "Cuantización de la energía"], scenes=[s1, s2])


def _ch20() -> Chapter:
    lam, sp = m20.de_broglie(), m20.fringe_spacing_analytic()
    est, _ = m20.fringe_spacing_from_hits(m20.electron_hits())
    sx0, sp0 = m20.packet_numeric(0.0)
    s1 = Scene(
        id="ElectronInterference", title="Interferencia de electrones y longitud de onda de De Broglie",
        statement="Electrones de 100 eV atraviesan dos rendijas separadas 1.0 µm y se detectan a 1.0 m. Hallar λ y el espaciado de las franjas; simular 20 000 detecciones.",
        foundations=[(r"\lambda=\dfrac hp,\quad p=\sqrt{K(K+2mc^2)}/c", "Longitud de onda de De Broglie."), (r"\Delta y=\dfrac{\lambda L}d", "Espaciado de franjas.")],
        analytic=[(rf"\lambda={lam*1e12:.4f}\ \mathrm{{pm}}", "Longitud de onda."), (rf"\Delta y={sp*1e6:.3f}\ \mu\mathrm{{m}}", "Espaciado.")],
        numeric_method="Monte Carlo de 20 000 electrones muestreados de I(y) por rechazo; el espaciado se estima con la frecuencia dominante del histograma (FFT con relleno de ceros) y el ángulo de Bragg por bisección.",
        rows=[Row("Espaciado de franjas", r"\Delta y", sp * 1e6, est * 1e6, "µm", 3), Row("Ángulo de Bragg (54 eV, d = 0.215 nm)", r"\theta", m20.bragg_angle_analytic(), m20.bragg_angle_numeric(), "°", 8)],
        notes=["La precisión de la estimación está limitada por el ruido de conteo (0.1 % con 20 000 electrones)."],
    )
    s2 = Scene(
        id="WavePacketUncertainty", title="Paquete de onda y principio de incertidumbre",
        statement="Un electrón libre está descrito por un paquete gaussiano con σ₀ = 0.100 nm. Hallar Δx Δp y su ensanchamiento temporal.",
        foundations=[(r"\Delta x\,\Delta p\ge\hbar/2", "Principio de incertidumbre."), (r"\sigma(t)=\sigma_0\sqrt{1+\left(\dfrac{\hbar t}{2m\sigma_0^2}\right)^2}", "Ensanchamiento libre.")],
        analytic=[(rf"\Delta x\,\Delta p=\hbar/2={m20.HBAR/2:.5e}\ \mathrm{{J\,s}}", "Estado de mínima incertidumbre."), (rf"\sigma(0.5\ \mathrm{{fs}})={float(m20.packet_width_analytic(0.5e-15))*1e10:.5f}\ \text{{Å}}", "Ancho a 0.5 fs.")],
        numeric_method="FFT de 4096 puntos: evolución libre exacta en el espacio de momentos (fase e^(−iħk²t/2m)); las desviaciones estándar se calculan de |ψ|² y |φ|².",
        rows=[Row("Producto de incertidumbres", r"\Delta x\Delta p/(\hbar/2)", 1.0, sx0 * sp0 / (m20.HBAR / 2), "", 8), Row("Ancho a 0.5 fs", r"\sigma", float(m20.packet_width_analytic(0.5e-15)) * 1e10, m20.packet_numeric(0.5e-15)[0] * 1e10, "Å", 7),
              Row("Ancho a 1 fs", r"\sigma", float(m20.packet_width_analytic(1e-15)) * 1e10, m20.packet_numeric(1e-15)[0] * 1e10, "Å", 7)],
    )
    return Chapter(number=20, volume=2, part=QM, slug="v2ch20", module="v2ch20_matter_waves", title="Ondas de materia",
                   summary="Hipótesis de De Broglie, difracción de electrones y principio de incertidumbre.",
                   concepts=["Hipótesis de De Broglie", "Difracción de electrones", "Principio de incertidumbre", "Paquetes de onda"], scenes=[s1, s2])


def _ch21() -> Chapter:
    e, _, _ = m21.well_numeric(4)
    ea = m21.well_energy_analytic([1, 2, 3, 4])
    ta = m21.transmission_analytic()
    s1 = Scene(
        id="InfiniteWell", title="Partícula en un pozo infinito",
        statement="Un electrón está confinado en un pozo infinito de L = 1.0 nm. Hallar los cuatro primeros niveles de energía y la función de onda del tercer estado.",
        foundations=[(r"-\dfrac{\hbar^2}{2m}\psi''+V\psi=E\psi", "Ecuación de Schrödinger estacionaria."), (r"E_n=\dfrac{n^2\pi^2\hbar^2}{2mL^2},\quad\psi_n=\sqrt{\tfrac2L}\sin\tfrac{n\pi x}L", "Solución.")],
        analytic=[(rf"E_1={ea[0]:.5f},\ E_2={ea[1]:.5f},\ E_3={ea[2]:.5f},\ E_4={ea[3]:.5f}\ \mathrm{{eV}}", "Niveles.")],
        numeric_method="Hamiltoniano de diferencias finitas (matriz tridiagonal de 2000 nodos) diagonalizado con eigh_tridiagonal y normalizado a ∫|ψ|² = 1.",
        rows=[Row(f"Nivel n = {n + 1}", f"E_{n + 1}", float(ea[n]), float(e[n]), "eV", 5) for n in range(4)] + [Row("Traslape del estado fundamental", r"|\langle\psi|\psi_1\rangle|", 1.0, m21.wavefunction_overlap(1), "", 7)],
    )
    s2 = Scene(
        id="BarrierTunneling", title="Efecto túnel a través de una barrera rectangular",
        statement="Un electrón de 5 eV incide sobre una barrera de 10 eV y 0.50 nm de ancho. Hallar la probabilidad de transmisión y su dependencia con la energía.",
        foundations=[(r"T=\left[1+\dfrac{V_0^2\sinh^2\kappa a}{4E(V_0-E)}\right]^{-1}", "Transmisión para E < V₀."), (r"T=\left[1+\dfrac{V_0^2\sin^2qa}{4E(E-V_0)}\right]^{-1}", "Transmisión para E > V₀.")],
        analytic=[(rf"T(5\ \mathrm{{eV}})={ta:.4e}", "Túnel."), (rf"T(12\ \mathrm{{eV}})={m21.transmission_analytic(12.0):.5f}", "Por encima de la barrera.")],
        numeric_method="RK4 de la ecuación estacionaria integrada desde la derecha (solo onda transmitida) hasta el borde izquierdo, donde se descompone en onda incidente y reflejada.",
        rows=[Row(f"Transmisión a {ev:g} eV", "T", m21.transmission_analytic(ev), m21.transmission_numeric(ev), "", 7) for ev in (5.0, 8.0, 12.0)],
    )
    return Chapter(number=21, volume=2, part=QM, slug="v2ch21", module="v2ch21_schrodinger", title="Ecuación de Schrödinger",
                   summary="Funciones de onda, pozos de potencial y tunelamiento.",
                   concepts=["Ecuación de Schrödinger", "Pozo infinito", "Barreras de potencial", "Efecto túnel"], scenes=[s1, s2])


def _ch22() -> Chapter:
    e, _, _ = m22.radial_numeric(8)
    la = {n: m22.balmer_wavelength_analytic(n) for n in (3, 4)}
    ln = {n: m22.balmer_wavelength_numeric(n, e) for n in (3, 4)}
    s1 = Scene(
        id="HydrogenSpectrum", title="Niveles del hidrógeno y serie de Balmer",
        statement="Hallar los niveles de energía del hidrógeno y las longitudes de onda de la serie de Balmer; verificarlos resolviendo numéricamente la ecuación radial.",
        foundations=[(r"E_n=-\dfrac{13.6\ \mathrm{eV}}{n^2}", "Niveles de Bohr."), (r"\dfrac1\lambda=R_\infty\left(\dfrac1{2^2}-\dfrac1{n^2}\right)", "Serie de Balmer.")],
        analytic=[(rf"E_1={m22.level_analytic(1):.4f},\ E_2={m22.level_analytic(2):.4f},\ E_3={m22.level_analytic(3):.4f}\ \mathrm{{eV}}", "Niveles."), (rf"\lambda_{{H\alpha}}={la[3]:.3f}\ \mathrm{{nm}},\ \lambda_{{H\beta}}={la[4]:.3f}\ \mathrm{{nm}}", "Balmer.")],
        numeric_method="Diagonalización tridiagonal de −u''/2 − u/r = E u en unidades atómicas (32 000 nodos hasta 160 a₀); λ = hc/(Eₙ − E₂).",
        rows=[Row(f"Nivel n = {n}", f"E_{n}", float(m22.level_analytic(n)), float(e[n - 1]), "eV", 4) for n in (1, 2, 3)] + [Row("Hα", r"\lambda", la[3], ln[3], "nm", 5), Row("Hβ", r"\lambda", la[4], ln[4], "nm", 5)],
        notes=["Se usa la constante de Rydberg para núcleo infinitamente pesado; la masa reducida desplaza las líneas unas décimas de nanómetro (0.05 %)."],
    )
    s2 = Scene(
        id="RadialProbability", title="Probabilidad radial del estado fundamental",
        statement="Para el estado fundamental del hidrógeno hallar el radio más probable y el radio medio y compararlos con la solución numérica de la ecuación radial.",
        foundations=[(r"\psi_{1s}=\dfrac1{\sqrt{\pi a_0^3}}e^{-r/a_0}", "Función de onda."), (r"P(r)=\dfrac{4r^2}{a_0^3}e^{-2r/a_0}", "Probabilidad radial.")],
        analytic=[(r"r_{mp}=a_0,\quad\langle r\rangle=\tfrac32a_0", "Radios característicos.")],
        numeric_method="Se usa u(r) del autovector fundamental, P = u²; ⟨r⟩ por Simpson sobre los 32 000 nodos y r_mp por ajuste parabólico del máximo.",
        rows=[Row("Radio medio", r"\langle r\rangle", 1.5, m22.mean_radius_1s_numeric(), "a₀", 5), Row("Radio más probable", r"r_{mp}", 1.0, m22.most_probable_radius_numeric(), "a₀", 5)],
    )
    return Chapter(number=22, volume=2, part=QM, slug="v2ch22", module="v2ch22_atom", title="Estructura atómica",
                   summary="Modelo de Bohr, átomo de hidrógeno y espectros atómicos.",
                   concepts=["Modelo de Bohr", "Átomo de hidrógeno", "Niveles de energía", "Probabilidad radial"], scenes=[s1, s2])


def _ch23() -> Chapter:
    lim = m23.band_limits_analytic(2)
    ana = m23.band_edges_analytic(3)
    num = m23.band_edges_numeric(3)
    va = {e: m23.diode_voltage_analytic(e) for e in (1.0, 5.0, 10.0)}
    s1 = Scene(
        id="BandStructure", title="Bandas de energía en un cristal unidimensional",
        statement="Un cristal unidimensional se modela con un peine de deltas de intensidad g = 4 (unidades atómicas, a = 1). Hallar las bandas, la brecha entre las dos primeras y E(k).",
        foundations=[(r"\cos ka=\cos qa+\dfrac{mg}{\hbar^2q}\sin qa", "Relación de dispersión de Kronig-Penney."), (r"\left|\cos qa+\dfrac{mg}{\hbar^2q}\sin qa\right|\le1", "Energías permitidas.")],
        analytic=[(rf"\text{{banda 1}}:[{lim[0][0]:.4f},{lim[0][1]:.4f}],\ \text{{banda 2}}:[{lim[1][0]:.4f},{lim[1][1]:.4f}]\ \mathrm{{Ha}}", "Bandas."), (rf"E_{{gap}}={lim[1][0]-lim[0][1]:.4f}\ \mathrm{{Ha}}", "Brecha.")],
        numeric_method="Hamiltoniano en una base de 201 ondas planas e^{i(k+2πm)x} con V_mm′ = g, diagonalizado en cada k; los bordes se evalúan en k = 0 y k = π.",
        rows=[Row(f"Borde {i + 1}", f"E_{i + 1}", float(ana[i]), float(num[i]), "Ha", 2) for i in range(4)],
        notes=["La convergencia en ondas planas para un potencial de deltas es lenta (error ~ 1/N), de modo que el acuerdo es del orden del 1 %."],
    )
    s2 = Scene(
        id="DiodeCircuit", title="Diodo de unión p-n en serie con una resistencia",
        statement="Un diodo (I_s = 1 pA, n = 1, V_T = 25.85 mV) está en serie con R = 100 Ω y una fuente E. Hallar V_d en función de E y la corriente para E = 5 V.",
        foundations=[(r"I=I_s\left(e^{V_d/nV_T}-1\right)", "Ecuación de Shockley."), (r"V_d=E+I_sR-nV_T\,W\!\left(\dfrac{I_sR}{nV_T}e^{(E+I_sR)/nV_T}\right)", "Solución exacta con la función W de Lambert.")],
        analytic=[(rf"V_d(5\ \mathrm{{V}})={va[5.0]:.6f}\ \mathrm{{V}}", "Tensión en el diodo."), (rf"I={m23.diode_current(5.0, va[5.0])*1e3:.4f}\ \mathrm{{mA}}", "Corriente.")],
        numeric_method="Bisección (200 iteraciones) de f(V) = I_s(e^{V/nV_T} − 1) − (E − V)/R sobre [0, E].",
        rows=[Row(f"Tensión del diodo con E = {e:g} V", "V_d", va[e], m23.diode_voltage_numeric(e), "V", 9) for e in (1.0, 5.0, 10.0)],
    )
    return Chapter(number=23, volume=2, part=QM, slug="v2ch23", module="v2ch23_solids", title="Conducción eléctrica en sólidos",
                   summary="Teoría de bandas, semiconductores y uniones p-n.",
                   concepts=["Teoría de bandas", "Modelo de Kronig-Penney", "Semiconductores", "Unión p-n"], scenes=[s1, s2])


def _ch24() -> Chapter:
    mean, err = m24.mean_life_monte_carlo()
    tt, y = m24.chain_numeric(20.0, 4000)
    za, zn = m24.z_stable_analytic(56), m24.z_stable_numeric(56, False)
    amax = int(np.argmax(m24.b_per_a_analytic(np.arange(20, 240)))) + 20
    s1 = Scene(
        id="DecayChain", title="Cadena de decaimiento radiactivo",
        statement="Un núcleo A (λ_A = 1.0 s⁻¹) decae a B (λ_B = 0.25 s⁻¹), que decae a un estado estable. Hallar N_B(t), el instante de su máximo y verificar la vida media con Monte Carlo.",
        foundations=[(r"\dfrac{dN}{dt}=-\lambda N,\quad T_{1/2}=\dfrac{\ln2}\lambda", "Decaimiento exponencial."), (r"N_B=N_0\dfrac{\lambda_A}{\lambda_B-\lambda_A}\left(e^{-\lambda_At}-e^{-\lambda_Bt}\right)", "Solución de Bateman.")],
        analytic=[(rf"t_{{max}}=\dfrac{{\ln(\lambda_A/\lambda_B)}}{{\lambda_A-\lambda_B}}={m24.peak_time_analytic():.6f}\ \mathrm{{s}}", "Máximo de N_B.")],
        numeric_method="RK4 del sistema de tres ecuaciones (4000 pasos) y Monte Carlo con 50 000 tiempos exponenciales obtenidos por inversión de la función de distribución.",
        rows=[Row("Instante del máximo de N_B", r"t_{max}", m24.peak_time_analytic(), m24.peak_time_numeric(), "s", 6), Row("N_B(20 s)", r"N_B", float(m24.bateman_b(20.0)), float(y[-1, 1]), "", 7), Row("Vida media de B (Monte Carlo)", r"\tau", 1 / m24.LAM_B, mean, "s", 2)],
        notes=[f"Incertidumbre estadística de la vida media: ±{err:.3f} s."],
    )
    s2 = Scene(
        id="BindingEnergy", title="Energía de enlace por nucleón: fórmula semiempírica",
        statement="Con los coeficientes de la fórmula semiempírica (sin apareamiento) hallar el núcleo estable para A = 56 y la curva B/A en función de A; identificar el máximo.",
        foundations=[(r"B=a_VA-a_SA^{2/3}-a_C\dfrac{Z(Z-1)}{A^{1/3}}-a_A\dfrac{(A-2Z)^2}A", "Fórmula semiempírica de masa."), (r"Z_0=\dfrac{A/2}{1+a_CA^{2/3}/4a_A}", "Mínimo de masa para A fijo.")],
        analytic=[(rf"Z_0(56)={za:.3f},\quad B/A_{{max}}\approx{float(np.max(m24.b_per_a_analytic(np.arange(20,240)))):.3f}\ \mathrm{{MeV}}", "Estabilidad y máximo.")],
        numeric_method="Para cada A se evalúa B en todos los Z enteros y se toma el máximo; el máximo de B/A se busca sobre 20 ≤ A ≤ 239.",
        rows=[Row("Z estable para A = 56 (continuo vs entero)", "Z", za, float(zn), "", 2), Row("B/A para A = 56", "B/A", float(m24.b_per_a_analytic(56)), m24.b_per_a(56, False), "MeV", 3), Row("A del máximo de B/A", "A", float(amax), float(m24.peak_a_numeric()), "", 2)],
        notes=["La comparación de la primera fila es entre un valor continuo y su redondeo al entero más cercano."],
    )
    return Chapter(number=24, volume=2, part=QM, slug="v2ch24", module="v2ch24_nuclear", title="Física nuclear",
                   summary="Estructura del núcleo, radiactividad, energía de enlace, fisión y fusión.",
                   concepts=["Estructura del núcleo", "Radiactividad", "Energía de enlace", "Fisión y fusión"], scenes=[s1, s2])


def _ch25() -> Chapter:
    lo, hi = m25.lab_energy_range_analytic()
    mean, std, pmax, pmin = m25.decay_monte_carlo()
    ph, pl = m25.lab_momentum_limits_analytic()
    s1 = Scene(
        id="TwoBodyDecay", title="Decaimiento a dos cuerpos y masa invariante",
        statement="Un Λ (1115.683 MeV) con p = 2000 MeV/c decae en p + π⁻. Hallar p* y el rango de energía del protón en el laboratorio; simular 200 000 decaimientos isótropos y reconstruir la masa invariante.",
        foundations=[(r"p^*=\dfrac{\sqrt{[M^2-(m_1+m_2)^2][M^2-(m_1-m_2)^2]}}{2M}", "Impulso en el sistema en reposo."), (r"E_{lab}=\gamma(E^*+\beta p^*\cos\theta^*)", "Boost de Lorentz.")],
        analytic=[(rf"p^*={m25.p_star_analytic():.4f}\ \mathrm{{MeV}}/c", "Impulso de las hijas."), (rf"E_p\in[{lo:.2f},{hi:.2f}]\ \mathrm{{MeV}}", "Distribución uniforme en E.")],
        numeric_method="Bisección de E₁(p) + E₂(p) = M para p*; Monte Carlo con dirección isótropa, boost a lo largo de z y reconstrucción de M² = (ΣE)² − |Σp|² evento por evento.",
        rows=[Row("Impulso en reposo", "p^*", m25.p_star_analytic(), m25.p_star_numeric(), "MeV/c", 9), Row("Masa invariante reconstruida", "M", m25.M_PARENT, mean, "MeV", 9), Row("Impulso máximo del protón en el laboratorio", r"p_{max}", ph, pmax, "MeV/c", 4),
              Row("Impulso mínimo del protón en el laboratorio", r"p_{min}", pl, pmin, "MeV/c", 4)],
        notes=[f"Dispersión de la masa reconstruida: {std:.1e} MeV (redondeo)."],
    )
    s2 = Scene(
        id="FriedmannAge", title="Edad del universo en el modelo de Friedmann plano",
        statement="Un universo plano con Ω_m = 0.31, Ω_Λ = 0.69 y H₀ = 0.0703 Gyr⁻¹ (68.7 km/s/Mpc). Hallar su edad y el factor de escala a(t).",
        foundations=[(r"\left(\dfrac{\dot a}a\right)^2=H_0^2\left(\dfrac{\Omega_m}{a^3}+\Omega_\Lambda\right)", "Ecuación de Friedmann."), (r"a(t)=\left(\dfrac{\Omega_m}{\Omega_\Lambda}\right)^{1/3}\sinh^{2/3}\!\left(\tfrac32\sqrt{\Omega_\Lambda}H_0t\right)", "Solución analítica.")],
        analytic=[(rf"t_0=\dfrac2{{3H_0\sqrt{{\Omega_\Lambda}}}}\mathrm{{arcsinh}}\sqrt{{\Omega_\Lambda/\Omega_m}}={m25.age_analytic():.4f}\ \mathrm{{Gyr}}", "Edad.")],
        numeric_method="Simpson de t₀ = ∫ da/(aH) con el cambio a = u² (elimina la singularidad) y RK4 de da/dt = H₀a√(Ω_m/a³ + Ω_Λ) desde un instante inicial pequeño.",
        rows=[Row("Edad del universo", "t_0", m25.age_analytic(), m25.age_numeric(), "Gyr", 7), Row("Factor de escala a 25 Gyr", "a", float(m25.scale_factor_analytic(25.0)), float(m25.scale_factor_numeric()[1][-1, 0]), "", 5)],
    )
    return Chapter(number=25, volume=2, part=QM, slug="v2ch25", module="v2ch25_particles", title="Partículas elementales y cosmología",
                   summary="Cinemática de decaimientos, partículas elementales y modelo cosmológico de Friedmann.",
                   concepts=["Leptones y quarks", "Masa invariante y decaimientos", "Fuerzas fundamentales", "Cosmología de Friedmann"], scenes=[s1, s2])


def chapters() -> list[Chapter]:
    return [_ch15(), _ch16(), _ch17(), _ch18(), _ch19(), _ch20(), _ch21(), _ch22(), _ch23(), _ch24(), _ch25()]
