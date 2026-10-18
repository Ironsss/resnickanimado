"""Volumen 2, capitulos 8 a 14."""
from __future__ import annotations

import numpy as np

from content.model import Chapter, Row, Scene
from physics import v2ch08_magnetic_field as m8
from physics import v2ch09_biot_savart as m9
from physics import v2ch10_faraday as m10
from physics import v2ch11_inductance as m11
from physics import v2ch12_magnetic_matter as m12
from physics import v2ch13_ac as m13
from physics import v2ch14_maxwell as m14
from physics.constants import C_LIGHT

PART = "Electromagnetismo"


def _ch8() -> Chapter:
    tb, yb = m8.helix_boris()
    tr, yr = m8.helix_rk4()
    va, vn = m8.hall_voltage_analytic(), m8.hall_voltage_numeric()
    s1 = Scene(
        id="HelicalMotion", title="Movimiento helicoidal de un protón en un campo magnético",
        statement=("Un protón entra en B = 0.10 T con v⊥ = 3.0×10⁵ m/s y v∥ = 1.0×10⁵ m/s. Hallar el radio, el periodo y el paso de la hélice y comprobar la conservación de la rapidez."),
        foundations=[(r"\vec F=q\vec v\times\vec B", "Fuerza de Lorentz magnética."), (r"r=\dfrac{mv_\perp}{qB},\quad T=\dfrac{2\pi m}{qB}", "Radio y periodo ciclotrón.")],
        analytic=[(rf"r={m8.radius_analytic()*100:.4f}\ \mathrm{{cm}},\quad T={m8.period_analytic()*1e6:.4f}\ \mu\mathrm{{s}}", "Radio y periodo."), (rf"p=v_\parallel T={m8.pitch_analytic()*100:.4f}\ \mathrm{{cm}}", "Paso de la hélice.")],
        numeric_method="Integración de m dv/dt = q v×B con RK4 y con el integrador de Boris (50 pasos por vuelta, 4 vueltas). El radio y el periodo se miden sobre la trayectoria.",
        rows=[Row("Radio", "r", m8.radius_analytic() * 100, m8.measured_radius(yb) * 100, "cm", 3), Row("Periodo", "T", m8.period_analytic() * 1e6, m8.measured_period(tb, yb) * 1e6, "µs", 3),
              Row("Deriva relativa de |v| (Boris)", r"\Delta v/v", 0.0, m8.speed_drift(yb), "", 2), Row("Deriva relativa de |v| (RK4)", r"\Delta v/v", 0.0, m8.speed_drift(yr), "", 2)],
        notes=["Boris conserva la rapidez a precisión de máquina; RK4 la degrada lentamente con el número de vueltas."],
    )
    s2 = Scene(
        id="HallEffect", title="Efecto Hall en una cinta de cobre",
        statement="Una cinta de cobre (n = 8.5×10²⁸ m⁻³, espesor 0.10 mm, ancho 10 mm) conduce 5.0 A en B = 0.5 T. Hallar la tensión de Hall y recuperar la densidad de portadores.",
        foundations=[(r"qE_H=qv_dB", "Equilibrio de fuerzas transversales."), (r"V_H=\dfrac{IB}{nqt}", "Tensión de Hall.")],
        analytic=[(rf"V_H={va*1e6:.4f}\ \mu\mathrm{{V}}", "Tensión de Hall."), (rf"n=\dfrac{{IB}}{{qtV_H}}={m8.carrier_density_from_hall(va):.4e}\ \mathrm{{m^{{-3}}}}", "Densidad de portadores.")],
        numeric_method="Modelo de Drude con campo magnético integrado con RK4 hasta el estado estacionario; el campo transversal se obtiene por bisección imponiendo velocidad transversal nula.",
        rows=[Row("Tensión de Hall", "V_H", va * 1e6, vn * 1e6, "µV", 5), Row("Densidad de portadores", "n", m8.N_CU / 1e28, m8.carrier_density_from_hall(vn) / 1e28, "10²⁸ m⁻³", 5)],
    )
    return Chapter(number=8, volume=2, part=PART, slug="v2ch08", module="v2ch08_magnetic_field", title="El campo magnético",
                   summary="Fuerza magnética sobre cargas, movimiento ciclotrón y efecto Hall.",
                   concepts=["Fuerza magnética sobre una carga", "Movimiento circular y helicoidal", "Efecto Hall"], scenes=[s1, s2])


def _ch9() -> Chapter:
    b0, w, wi = float(m9.loop_axis_analytic(0.0)), m9.wire_field_analytic(), m9.infinite_wire()
    be = float(m9.solenoid_axis_exact(0.0))
    s1 = Scene(
        id="BiotSavartLoop", title="Campo de una espira y de un hilo finito",
        statement="Una espira de radio 5.0 cm conduce 3.0 A. Hallar B sobre su eje; verificar además el campo de un hilo recto de 30 cm a 4.0 cm de su punto medio.",
        foundations=[(r"d\vec B=\dfrac{\mu_0I}{4\pi}\dfrac{d\vec\ell\times\hat r}{r^2}", "Ley de Biot-Savart."), (r"B_z=\dfrac{\mu_0Ia^2}{2(a^2+z^2)^{3/2}}", "Eje de una espira.")],
        analytic=[(rf"B(0)=\dfrac{{\mu_0I}}{{2a}}={b0*1e6:.4f}\ \mu\mathrm{{T}}", "Centro de la espira."), (rf"B_{{hilo}}={w*1e6:.4f}\ \mu\mathrm{{T}}", "Hilo finito.")],
        numeric_method="Suma vectorial de Biot-Savart sobre 720 segmentos de la espira y 2000 del hilo, con producto vectorial explícito.",
        rows=[Row("B(0) de la espira", "B", b0 * 1e6, m9.loop_axis_numeric(0.0) * 1e6, "µT", 5), Row("B(5 cm) de la espira", "B", float(m9.loop_axis_analytic(0.05)) * 1e6, m9.loop_axis_numeric(0.05) * 1e6, "µT", 5),
              Row("Hilo finito", "B", w * 1e6, m9.wire_field_numeric() * 1e6, "µT", 5)],
        notes=[f"El hilo infinito daría {wi*1e6:.4f} µT: truncar el hilo reduce el campo en {(1 - w / wi) * 100:.1f} %."],
    )
    s2 = Scene(
        id="SolenoidAxis", title="Campo sobre el eje de un solenoide finito",
        statement="Un solenoide de 500 vueltas, 50 cm de largo y 2.0 cm de radio conduce 2.0 A. Hallar B en el centro y su perfil sobre el eje; comparar con el solenoide infinito.",
        foundations=[(r"\oint\vec B\cdot d\vec\ell=\mu_0I_{enc}", "Ley de Ampère."), (r"B=\mu_0nI", "Solenoide largo."), (r"B(z)=\dfrac{\mu_0nI}2(\cos\alpha_2-\cos\alpha_1)", "Solenoide finito.")],
        analytic=[(rf"B_\infty=\mu_0nI={m9.solenoid_center_infinite()*1e3:.4f}\ \mathrm{{mT}}", "Límite infinito."), (rf"B(0)={be*1e3:.4f}\ \mathrm{{mT}}", "Solenoide finito, centro.")],
        numeric_method="Biot-Savart sobre las 500 espiras, cada una con 60 segmentos, evaluado en puntos del eje.",
        rows=[Row("B(0)", "B", be * 1e3, m9.solenoid_axis_numeric(0.0) * 1e3, "mT", 3), Row("B en el extremo", "B", float(m9.solenoid_axis_exact(0.25)) * 1e3, m9.solenoid_axis_numeric(0.25) * 1e3, "mT", 3)],
        notes=["La discretización de cada espira en 60 segmentos introduce un error sistemático de unos 0.09 % en el centro, visible en la tercera cifra."],
    )
    return Chapter(number=9, volume=2, part=PART, slug="v2ch09", module="v2ch09_biot_savart", title="Campo magnético de una corriente",
                   summary="Ley de Biot-Savart, ley de Ampère y campos de espiras, hilos y solenoides.",
                   concepts=["Ley de Biot-Savart", "Ley de Ampère", "Campo de una espira y de un solenoide"], scenes=[s1, s2])


def _ch10() -> Chapter:
    e0 = m10.emf_peak_analytic()
    t, y = m10.rod_numeric()
    tau = m10.tau_rod()
    s1 = Scene(
        id="RotatingLoop", title="Generador: bobina giratoria en un campo uniforme",
        statement="Una bobina de 100 vueltas y 0.020 m² gira a 50 Hz en B = 0.50 T. Hallar la fem inducida, su máximo y su valor eficaz.",
        foundations=[(r"\varepsilon=-\dfrac{d\Phi_B}{dt}", "Ley de Faraday."), (r"\Phi_B=NBA\cos\omega t", "Flujo a través de la bobina.")],
        analytic=[(rf"\varepsilon_0=NBA\omega={e0:.4f}\ \mathrm{{V}}", "Fem máxima."), (rf"\varepsilon_{{rms}}=\varepsilon_0/\sqrt2={e0/np.sqrt(2):.4f}\ \mathrm{{V}}", "Valor eficaz.")],
        numeric_method="La fem se obtiene derivando numéricamente el flujo con diferencias centrales (h = 1 µs); el valor eficaz se integra con Simpson sobre un periodo.",
        rows=[Row("ε a 5 ms", r"\varepsilon", float(m10.emf_analytic(0.005)), float(m10.emf_numeric(np.array([0.005]))[0]), "V", 5), Row("Valor eficaz", r"\varepsilon_{rms}", e0 / np.sqrt(2), m10.emf_rms_numeric(), "V", 5)],
    )
    s2 = Scene(
        id="SlidingRod", title="Barra deslizante: fem de movimiento y frenado magnético",
        statement="Una barra de 0.20 kg y 0.50 m desliza sin fricción sobre rieles unidos por R = 2.0 Ω en B = 0.80 T, con v₀ = 4.0 m/s. Hallar v(t), la distancia recorrida y la energía disipada.",
        foundations=[(r"\varepsilon=B\ell v,\qquad F=-\dfrac{B^2\ell^2}{R}v", "Fem de movimiento y fuerza de frenado."), (r"v=v_0e^{-t/\tau},\quad\tau=\dfrac{mR}{B^2\ell^2}", "Solución.")],
        analytic=[(rf"\tau={tau:.3f}\ \mathrm{{s}},\quad x_\infty={m10.rod_distance_analytic():.4f}\ \mathrm{{m}}", "Constante de tiempo y alcance."), (rf"E=\tfrac12mv_0^2={0.5*m10.M_ROD*m10.V0_ROD**2:.4f}\ \mathrm{{J}}", "Energía disipada.")],
        numeric_method="RK4 del sistema (x, v) y integración de Simpson de I²R sobre 25 s.",
        rows=[Row("v(4 s)", "v", float(m10.rod_velocity_analytic(4.0)), float(np.interp(4.0, t, y[:, 1])), "m/s", 5), Row("x(12 s)", "x", m10.rod_distance_analytic() * (1 - np.exp(-12 / tau)), y[-1, 0], "m", 5),
              Row("Energía disipada", "E", 0.5 * m10.M_ROD * m10.V0_ROD**2, m10.rod_energy_dissipated_numeric(), "J", 5)],
    )
    return Chapter(number=10, volume=2, part=PART, slug="v2ch10", module="v2ch10_faraday", title="Ley de inducción de Faraday",
                   summary="Flujo magnético, ley de Faraday, ley de Lenz y fem de movimiento.",
                   concepts=["Ley de Faraday", "Ley de Lenz", "Fem de movimiento"], scenes=[s1, s2])


def _ch11() -> Chapter:
    tau = m11.tau_rl()
    th = 2 * np.pi / m11.omega_d()
    s1 = Scene(
        id="RLCircuit", title="Crecimiento de la corriente en un circuito RL",
        statement="Un inductor de 0.20 H en serie con 4.0 Ω se conecta a 12 V en t = 0. Hallar i(t), la constante de tiempo y la energía almacenada.",
        foundations=[(r"\varepsilon_L=-L\dfrac{di}{dt}", "Autoinducción."), (r"i=\dfrac\varepsilon R(1-e^{-t/\tau_L}),\quad U_B=\tfrac12Li^2", "Corriente y energía.")],
        analytic=[(rf"\tau_L=L/R={tau*1e3:.2f}\ \mathrm{{ms}}", "Constante de tiempo."), (rf"U_B=\tfrac12Li_\infty^2={m11.energy_magnetic():.4f}\ \mathrm{{J}}", "Energía final.")],
        numeric_method="RK4 de L di/dt = ε − Ri; el tiempo de decaimiento al 50 % tras retirar la fuente se interpola; la energía es ∫ i v_L dt con Simpson.",
        rows=[Row("Tiempo de decaimiento al 50 %", r"t_{1/2}", tau * np.log(2) * 1e3, m11.time_to_half_numeric() * 1e3, "ms", 6), Row("Energía almacenada", "U_B", m11.energy_magnetic(), m11.energy_integral_numeric(), "J", 4)],
    )
    s2 = Scene(
        id="RLCOscillator", title="Oscilaciones amortiguadas en un circuito RLC",
        statement="Un capacitor de 100 µF con 1.0 mC se conecta a L = 0.10 H y R = 5.0 Ω. Hallar q(t), la frecuencia amortiguada, el factor de calidad y la energía tras un periodo.",
        foundations=[(r"L\ddot q+R\dot q+q/C=0", "Ecuación del circuito."), (r"\omega_d=\sqrt{\omega_0^2-\gamma^2},\quad\gamma=\dfrac R{2L}", "Frecuencia amortiguada.")],
        analytic=[(rf"\omega_d={m11.omega_d():.3f}\ \mathrm{{rad/s}},\quad Q={m11.quality_factor():.3f}", "Frecuencia y calidad."), (rf"U(T)/U(0)=e^{{-2\gamma T}}={np.exp(-2*m11.gamma_()*th):.5f}", "Energía tras un periodo.")],
        numeric_method="RK4 del sistema (q, i) con 20 000 pasos; el periodo se mide entre cruces descendentes por cero con interpolación lineal.",
        rows=[Row("Periodo amortiguado", "T", th * 1e3, m11.period_numeric() * 1e3, "ms", 5), Row("Razón de energías tras un periodo", "U(T)/U(0)", np.exp(-2 * m11.gamma_() * th), m11.energy_ratio_after_period(), "", 5)],
    )
    return Chapter(number=11, volume=2, part=PART, slug="v2ch11", module="v2ch11_inductance", title="Inductancia",
                   summary="Autoinductancia, circuitos RL, energía magnética y oscilaciones RLC.",
                   concepts=["Autoinductancia", "Circuitos RL", "Energía en el campo magnético", "Oscilaciones LC y RLC"], scenes=[s1, s2])


def _ch12() -> Chapter:
    t0 = 2 * np.pi / m12.omega_small()
    xs = (0.1, 0.8, 1.6, 3.2)
    mm = {x: m12.magnetization_metropolis(x) for x in xs}
    s1 = Scene(
        id="DipoleTorque", title="Oscilaciones de un dipolo magnético en un campo uniforme",
        statement="Una aguja imanada (μ = 0.10 A·m², I = 1.0×10⁻⁶ kg·m²) oscila en B = 50 µT. Hallar el periodo para pequeñas amplitudes y su dependencia con la amplitud.",
        foundations=[(r"\vec\tau=\vec\mu\times\vec B,\quad U=-\vec\mu\cdot\vec B", "Dipolo en un campo."), (r"T=\dfrac{T_0}{\mathrm{AGM}(1,\cos(\theta_0/2))}", "Periodo exacto (péndulo no lineal).")],
        analytic=[(rf"T_0=2\pi\sqrt{{I/\mu B}}={t0:.4f}\ \mathrm{{s}}", "Pequeñas oscilaciones."), (rf"T(90^\circ)={m12.period_exact(90.0):.4f}\ \mathrm{{s}}", "Amplitud de 90°.")],
        numeric_method="RK4 de I θ'' = −μB sen θ con 40 000 pasos; el periodo se mide entre dos cruces descendentes por cero y se comprueba la conservación de la energía.",
        rows=[Row("Periodo a 30°", "T", m12.period_exact(30.0), m12.period_numeric(30.0), "s", 5), Row("Periodo a 90°", "T", m12.period_exact(90.0), m12.period_numeric(90.0), "s", 5),
              Row("Deriva de la energía a 60°", r"\Delta E/E", 0.0, m12.energy_conservation_drift(), "", 2)],
    )
    s2 = Scene(
        id="ParamagnetCurie", title="Paramagneto ideal: ley de Curie por Monte Carlo",
        statement="Un paramagneto ideal de espines 1/2 independientes con momento μ_B está en B y T. Hallar M/M_s en función de x = μ_B B/kT y verificar la ley de Curie a campo débil.",
        foundations=[(r"U=\mp\mu_BB", "Dos niveles."), (r"\dfrac M{M_s}=\tanh\dfrac{\mu_BB}{kT}", "Distribución de Boltzmann.")],
        analytic=[(r"\dfrac M{M_s}\approx x\quad(x\ll1)", "Ley de Curie."), (rf"x(1\ \mathrm{{T}},300\ \mathrm{{K}})={m12.x_param(1.0, 300.0):.3e}", "Valor típico.")],
        numeric_method="Monte Carlo de Metropolis con 100 000 espines y 80 barridos, promediando la segunda mitad; el error estadístico es 1/√N.",
        rows=[Row(f"M/M_s con x = {x}", "M/M_s", float(np.tanh(x)), mm[x][0], "", 3) for x in xs],
        notes=["Incertidumbres estadísticas: " + ", ".join(f"±{mm[x][1]:.4f} (x = {x})" for x in xs) + "."],
    )
    return Chapter(number=12, volume=2, part=PART, slug="v2ch12", module="v2ch12_magnetic_matter", title="Propiedades magnéticas de la materia",
                   summary="Momento dipolar, oscilaciones de un dipolo y paramagnetismo.",
                   concepts=["Dipolo magnético", "Torque y energía", "Paramagnetismo y ley de Curie"], scenes=[s1, s2])


def _ch13() -> Chapter:
    w0 = m13.omega0()
    pk = float(m13.current_amplitude_analytic(w0))
    a0 = m13.steady_state_numeric(w0)
    a1 = m13.steady_state_numeric(1250.0)
    pa = m13.power_average_analytic()
    tr_a, tr_n = m13.transformer_analytic(), m13.transformer_numeric()
    s1 = Scene(
        id="SeriesRLC", title="Circuito RLC en serie: impedancia y resonancia",
        statement="Un circuito RLC en serie (R = 20 Ω, L = 0.10 H, C = 10 µF) se alimenta con V₀ = 50 V de frecuencia variable. Hallar la resonancia, el factor de calidad y I₀(ω).",
        foundations=[(r"Z=\sqrt{R^2+(\omega L-1/\omega C)^2}", "Impedancia."), (r"\tan\phi=\dfrac{\omega L-1/\omega C}R", "Fase de la corriente.")],
        analytic=[(rf"\omega_0={w0:.3f}\ \mathrm{{rad/s}},\quad Q={m13.q_factor():.3f}", "Resonancia y calidad."), (rf"I_{{max}}=V_0/R={pk:.4f}\ \mathrm{{A}}", "Corriente máxima.")],
        numeric_method="RK4 de la ecuación diferencial en el dominio del tiempo y detección síncrona (proyección sobre sen y cos) en 6 ciclos tras 25 ciclos de transitorio.",
        rows=[Row("Amplitud de corriente en ω₀", "I_0", pk, a0[0], "A", 5), Row("Fase en ω₀", r"\phi", 0.0, a0[1], "rad", 3), Row("Fase en 1250 rad/s", r"\phi", float(m13.phase_analytic(1250.0)), a1[1], "rad", 5)],
    )
    s2 = Scene(
        id="PowerTransformer", title="Potencia media y transformador ideal",
        statement="Una carga recibe v = 170 sen ωt V con i = 8.0 sen(ωt − 40°) A. Hallar la potencia media y los valores eficaces. Un transformador ideal 200:50 con 120 V y 2.0 A eficaces en el primario: hallar el secundario.",
        foundations=[(r"\bar P=\tfrac12V_0I_0\cos\phi", "Potencia media."), (r"\dfrac{V_s}{V_p}=\dfrac{N_s}{N_p},\quad V_pI_p=V_sI_s", "Transformador ideal.")],
        analytic=[(rf"\bar P={pa:.4f}\ \mathrm{{W}}", "Potencia media."), (rf"V_s={tr_a['vs']:.2f}\ \mathrm{{V}},\quad I_s={tr_a['is']:.2f}\ \mathrm{{A}}", "Secundario.")],
        numeric_method="Simpson sobre un periodo de v(t)·i(t) y de v²; en el transformador se muestrean las ondas, se integran los valores eficaces y se comprueba v·i en ambos lados.",
        rows=[Row("Potencia media", r"\bar P", pa, m13.power_average_numeric(), "W", 5), Row("Tensión eficaz", r"V_{rms}", m13.V_PEAK / np.sqrt(2), m13.rms_numeric(m13.V_PEAK), "V", 5),
              Row("Tensión del secundario", "V_s", tr_a["vs"], tr_n["vs"], "V", 5), Row("Corriente del secundario", "I_s", tr_a["is"], tr_n["is"], "A", 5),
              Row("Diferencia de potencia", r"P_p-P_s", 0.0, tr_n["p_p"] - tr_n["p_s"], "W", 3)],
    )
    return Chapter(number=13, volume=2, part=PART, slug="v2ch13", module="v2ch13_ac", title="Circuitos de corriente alterna",
                   summary="Impedancia, resonancia, potencia en corriente alterna y transformadores.",
                   concepts=["Circuito RLC en serie", "Resonancia", "Potencia en CA", "Transformadores"], scenes=[s1, s2])


def _ch14() -> Chapter:
    ia = m14.intensity_analytic()
    c0 = m14.c_from_constants()
    s1 = Scene(
        id="YeeFDTD", title="Propagación de un pulso electromagnético por diferencias finitas",
        statement="Un pulso gaussiano (σ = 2 cm) viaja en el vacío en +x. Hallar c a partir de μ₀ y ε₀ y medir la velocidad del pulso con el esquema de Yee (Courant 0.5, 600 pasos).",
        foundations=[(r"\nabla\times\vec E=-\partial_t\vec B,\quad\nabla\times\vec B=\mu_0\vec J+\mu_0\varepsilon_0\partial_t\vec E", "Faraday y Ampère-Maxwell."), (r"c=\dfrac1{\sqrt{\mu_0\varepsilon_0}}", "Velocidad de la onda.")],
        analytic=[(rf"c={c0:.6e}\ \mathrm{{m/s}}", "Calculada con las constantes.")],
        numeric_method="FDTD de Yee en una dimensión: E y B en mallas escalonadas con actualizaciones de salto de rana; la velocidad es la pendiente de la regresión del máximo del pulso contra el tiempo.",
        rows=[Row("Velocidad con Courant 0.5", "c", c0 / 1e8, m14.speed_numeric(0.5) / 1e8, "10⁸ m/s", 5), Row("Velocidad con Courant 0.9", "c", c0 / 1e8, m14.speed_numeric(0.9) / 1e8, "10⁸ m/s", 5)],
        notes=["La dispersión numérica disminuye al acercarse el número de Courant a 1, donde el esquema es exacto en 1D."],
    )
    s2 = Scene(
        id="PoyntingDisplacement", title="Vector de Poynting, presión de radiación y corriente de desplazamiento",
        statement="Una onda plana con E₀ = 100 V/m incide sobre un absorbente perfecto: hallar la intensidad y la presión. Un capacitor de 100 cm² con 1.0 mm de separación se carga a dV/dt = 10⁶ V/s: hallar la corriente de desplazamiento.",
        foundations=[(r"\vec S=\dfrac1{\mu_0}\vec E\times\vec B,\quad I=\tfrac12c\varepsilon_0E_0^2", "Poynting."), (r"I_d=\varepsilon_0\dfrac{d\Phi_E}{dt}", "Corriente de desplazamiento.")],
        analytic=[(rf"I={ia:.4f}\ \mathrm{{W/m^2}},\quad p={m14.radiation_pressure(ia)*1e9:.4f}\ \mathrm{{nPa}}", "Intensidad y presión."), (rf"I_d={m14.conduction_current_analytic()*1e6:.4f}\ \mu\mathrm{{A}}", "Corriente de desplazamiento.")],
        numeric_method="Promedio temporal de E×B/μ₀ con Simpson en un periodo; la corriente de desplazamiento se deriva numéricamente (diferencias centrales).",
        rows=[Row("Intensidad", "I", ia, m14.intensity_numeric(), "W/m²", 6), Row("Corriente de desplazamiento", "I_d", m14.conduction_current_analytic() * 1e6, m14.displacement_current_numeric() * 1e6, "µA", 5)],
    )
    return Chapter(number=14, volume=2, part=PART, slug="v2ch14", module="v2ch14_maxwell", title="Ecuaciones de Maxwell y ondas electromagnéticas",
                   summary="Corriente de desplazamiento, ondas electromagnéticas, vector de Poynting y presión de radiación.",
                   concepts=["Ecuaciones de Maxwell", "Corriente de desplazamiento", "Ondas electromagnéticas", "Vector de Poynting"], scenes=[s1, s2])


def chapters() -> list[Chapter]:
    return [_ch8(), _ch9(), _ch10(), _ch11(), _ch12(), _ch13(), _ch14()]
