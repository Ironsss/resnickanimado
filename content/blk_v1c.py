"""Volumen 1, capitulos 18 a 25."""
from __future__ import annotations

import numpy as np

from content.model import Chapter, Row, Scene
from physics import ch18_oscillations as c18
from physics import ch19_waves as c19
from physics import ch20_sound as c20
from physics import ch21_relativity as c21
from physics import ch22_temperature as c22
from physics import ch23_first_law as c23
from physics import ch24_kinetic_theory as c24
from physics import ch25_second_law as c25


def _ch18() -> Chapter:
    w0 = c18.omega0()
    t, y = c18.shm_numeric()
    tab = c18.pendulum_table((10, 60))
    wr = c18.resonance_frequency_analytic()
    ws = np.array([6.0, 9.0, 12.0])
    num = c18.resonance_curve_numeric(ws)
    s1 = Scene(
        id="SimpleHarmonic", title="Movimiento armónico simple y péndulo exacto",
        statement=("Un bloque de 0.5 kg unido a un resorte de k = 200 N/m se suelta desde x = 8 cm. Hallar ω, T y la energía; después comparar "
                   "el periodo exacto de un péndulo de 1 m con el de pequeñas oscilaciones."),
        foundations=[(r"m\ddot x=-kx\;\Rightarrow\;x=A\cos(\omega t+\varphi),\quad\omega=\sqrt{k/m}", "Oscilador armónico."),
                     (r"T_{exacto}=\dfrac{2\pi\sqrt{L/g}}{\mathrm{AGM}(1,\cos(\theta_0/2))}", "Péndulo de amplitud arbitraria.")],
        analytic=[(rf"\omega={w0:.1f}\ \mathrm{{rad/s}},\quad T={2*np.pi/w0:.4f}\ \mathrm{{s}}", "Frecuencia y periodo."),
                  (rf"v_{{max}}=A\omega={c18.AMP*w0:.2f}\ \mathrm{{m/s}},\quad E={c18.energy_analytic():.3f}\ \mathrm{{J}}", "Velocidad máxima y energía.")],
        numeric_method="RK4 sobre x'' = −(k/m)x (2000 pasos); el periodo se mide por cruces de cero con interpolación. El periodo del péndulo se integra con RK4 y se compara con la media aritmético-geométrica.",
        rows=[Row("Periodo del resorte", "T", 2 * np.pi / w0, c18.period_numeric(), "s", 6),
              Row("Error máximo de x(t)", r"\max|\Delta x|", 0.0, float(np.max(np.abs(y[:, 0] - c18.x_analytic(t)))), "m", 12),
              Row("Periodo del péndulo, θ₀ = 10°", "T", tab[0]["exact"], tab[0]["numeric"], "s", 5),
              Row("Periodo del péndulo, θ₀ = 60°", "T", tab[1]["exact"], tab[1]["numeric"], "s", 5)],
    )
    s2 = Scene(
        id="DrivenResonance", title="Oscilador forzado y curva de resonancia",
        statement=("Un oscilador de m = 1 kg, k = 100 N/m y b = 2 kg/s se fuerza con F₀ = 5 N. Hallar la amplitud estacionaria en función de ω y la "
                   "frecuencia de máxima respuesta."),
        foundations=[(r"m\ddot x+b\dot x+kx=F_0\cos\omega t", "Ecuación de movimiento."),
                     (r"A(\omega)=\dfrac{F_0/m}{\sqrt{(\omega_0^2-\omega^2)^2+(\gamma\omega)^2}}", "Amplitud estacionaria.")],
        analytic=[(rf"\omega_r=\sqrt{{\omega_0^2-\gamma^2/2}}={wr:.3f}\ \mathrm{{rad/s}}", "Frecuencia de resonancia de amplitud."),
                  (rf"A(\omega_r)={float(c18.driven_amplitude_analytic(wr)):.4f}\ \mathrm{{m}},\quad Q=5", "Amplitud en resonancia.")],
        numeric_method="Simulación RK4 hasta el régimen estacionario (24 s) para cada ω y medición del máximo de |x| en los últimos ciclos.",
        rows=[Row(f"Amplitud con ω = {w_:.0f} rad/s", "A", float(c18.driven_amplitude_analytic(w_)), float(a), "m", 5) for w_, a in zip(ws, num)],
    )
    return Chapter(number=18, volume=1, part="Oscilaciones y ondas", slug="ch18", module="ch18_oscillations", title="Oscilaciones",
                   summary="Movimiento armónico simple, péndulos, oscilaciones amortiguadas y forzadas.",
                   concepts=["Oscilaciones armónicas simples", "Energía en el movimiento armónico simple", "Péndulos", "Oscilaciones amortiguadas",
                             "Oscilaciones forzadas y resonancia"], scenes=[s1, s2])


def _ch19() -> Chapter:
    v = c19.V
    err1, err9 = c19.fdtd_error(1.0), c19.fdtd_error(0.9)
    freqs, xs, shapes = c19.modes_numeric()
    s1 = Scene(
        id="WavePulse", title="Pulso en una cuerda: ecuación de onda",
        statement=("Una cuerda con T = 80 N y μ = 0.05 kg/m recibe un pulso gaussiano (σ = 0.6 m) en x = 10 m, inicialmente en reposo. "
                   "Hallar la velocidad de la onda y la forma del pulso a los 0.15 s."),
        foundations=[(r"\dfrac{\partial^2y}{\partial t^2}=v^2\dfrac{\partial^2y}{\partial x^2},\qquad v=\sqrt{T/\mu}", "Ecuación de onda."),
                     (r"y=f(x-vt)+g(x+vt)", "Solución general de d'Alembert.")],
        analytic=[(rf"v=\sqrt{{T/\mu}}={v:.1f}\ \mathrm{{m/s}}", "Velocidad de la onda."),
                  (r"y(x,t)=\tfrac12\left[f(x-vt)+f(x+vt)\right]", "Dos pulsos de amplitud 1/2.")],
        numeric_method="Diferencias finitas centradas (leapfrog) con 800 intervalos y número de Courant 1, 0.9 y 0.5; error máximo frente a d'Alembert a los 0.15 s.",
        rows=[Row("Velocidad de la onda", "v", v, float(np.sqrt(c19.TENSION / c19.MU)), "m/s", 3),
              Row("Error máx. con Courant 1.0", r"\varepsilon", 0.0, err1, "m", 12),
              Row("Error máx. con Courant 0.9", r"\varepsilon", 0.0, err9, "m", 7)],
        notes=["Con Courant = 1 el esquema es exacto para la ecuación de onda; con Courant < 1 aparece dispersión numérica."],
    )
    s2 = Scene(
        id="StandingModes", title="Modos normales de una cuerda fija",
        statement="Una cuerda de 1.2 m con v = 40 m/s está fija en ambos extremos. Hallar las frecuencias de los primeros modos y verificar sus formas.",
        foundations=[(r"y_n=A_n\sin\dfrac{n\pi x}L\cos\omega_nt", "Modos normales."), (r"f_n=\dfrac{nv}{2L}", "Frecuencias.")],
        analytic=[(rf"f_1={c19.mode_frequency_analytic(1):.2f}\ \mathrm{{Hz}},\quad f_2={c19.mode_frequency_analytic(2):.2f}\ \mathrm{{Hz}}", "Armónicos.")],
        numeric_method="Autovalores y autovectores de la matriz tridiagonal de la segunda derivada con extremos fijos (400 puntos).",
        rows=[Row(f"Frecuencia del modo {n}", f"f_{n}", c19.mode_frequency_analytic(n), float(freqs[n - 1]), "Hz", 4) for n in (1, 2, 3, 5)]
              + [Row("Error máx. de la forma del modo 1", r"\varepsilon", 0.0, float(np.max(np.abs(shapes[0] - np.sin(np.pi * xs / c19.STRING_L)))), "", 8)],
    )
    return Chapter(number=19, volume=1, part="Oscilaciones y ondas", slug="ch19", module="ch19_waves", title="Movimiento ondulatorio",
                   summary="Ondas transversales y longitudinales, velocidad de onda, superposición y ondas estacionarias.",
                   concepts=["Ondas mecánicas", "Ondas viajeras", "Velocidad de una onda", "Superposición", "Ondas estacionarias y resonancia"], scenes=[s1, s2])


def _ch20() -> Chapter:
    vs = c20.VS
    s1 = Scene(
        id="DopplerEffect", title="Efecto Doppler",
        statement=("Un avión emite un tono de 440 Hz y vuela a 120 m/s hacia un observador en reposo (c = 343 m/s). Hallar la frecuencia que oye "
                   "el observador delante y detrás del avión."),
        foundations=[(r"f'=f_0\dfrac{c+v_o}{c-v_s}", "Fórmula de Doppler.")],
        analytic=[(rf"f'_{{delante}}=f_0\dfrac c{{c-v_s}}={c20.doppler_analytic():.2f}\ \mathrm{{Hz}}", "Fuente que se acerca."),
                  (rf"f'_{{detrás}}=f_0\dfrac c{{c+v_s}}={c20.doppler_analytic(vs=-vs):.2f}\ \mathrm{{Hz}}", "Fuente que se aleja.")],
        numeric_method="Se emiten pulsos cada 1/f₀ desde la fuente en movimiento; el tiempo de llegada de cada pulso al observador se calcula con la ecuación del frente de onda y la frecuencia es la inversa del intervalo medio.",
        rows=[Row("Fuente que se acerca", "f'", c20.doppler_analytic(), c20.doppler_numeric(), "Hz", 3),
              Row("Fuente que se aleja", "f'", c20.doppler_analytic(vs=-vs), c20.doppler_numeric(vs=-vs), "Hz", 3),
              Row("Observador que se acerca a 30 m/s", "f'", c20.doppler_analytic(vs=0.0, vo=30.0), c20.doppler_numeric(vs=0.0, vo=30.0), "Hz", 3)],
    )
    op_n, op_a = c20.pipe_frequencies_numeric("open"), c20.pipe_frequencies_analytic("open")
    cl_n, cl_a = c20.pipe_frequencies_numeric("closed"), c20.pipe_frequencies_analytic("closed")
    s2 = Scene(
        id="BeatsAndPipes", title="Pulsaciones y resonancia en tubos",
        statement=("Dos diapasones suenan a 440 y 444 Hz. Hallar la frecuencia de pulsación. Hallar también las frecuencias de resonancia de un tubo de "
                   "0.75 m abierto en ambos extremos y de otro cerrado en un extremo."),
        foundations=[(r"y=2\cos\!\left(\pi\Delta f\,t\right)\sin\!\left(\pi(f_1+f_2)t\right)", "Suma de dos tonos."),
                     (r"f_n^{ab}=\dfrac{nc}{2L},\qquad f_n^{ce}=\dfrac{(2n-1)c}{4L}", "Modos de un tubo.")],
        analytic=[(rf"f_{{puls}}=|f_2-f_1|={c20.beat_frequency_analytic():.0f}\ \mathrm{{Hz}}", "Pulsación.")],
        numeric_method="FFT de la señal con ventana de Hann e interpolación parabólica de los picos; envolvente por transformada de Hilbert; modos de los tubos como autovalores de una matriz tridiagonal.",
        rows=[Row("Pulsación (FFT)", r"f_{puls}", 4.0, c20.beat_frequency_numeric(), "Hz", 4),
              Row("Pulsación (envolvente)", r"f_{puls}", 4.0, c20.beat_envelope_frequency_numeric(), "Hz", 4),
              Row("Tubo abierto, n = 1", "f_1", float(op_a[0]), float(op_n[0]), "Hz", 2),
              Row("Tubo cerrado, n = 1", "f_1", float(cl_a[0]), float(cl_n[0]), "Hz", 2)],
    )
    return Chapter(number=20, volume=1, part="Oscilaciones y ondas", slug="ch20", module="ch20_sound", title="Ondas sonoras",
                   summary="Propagación del sonido, efecto Doppler, interferencia, pulsaciones y resonancia.",
                   concepts=["Velocidad del sonido", "Ondas longitudinales viajeras", "Interferencia y pulsaciones", "Tubos resonantes", "Efecto Doppler"], scenes=[s1, s2])


def _ch21() -> Chapter:
    tau, beta, d = c21.TAU_MUON, c21.BETA_MUON, c21.D_MUON
    p_rel, p_cl = c21.survival_analytic(), c21.survival_analytic(relativistic=False)
    p_mc, err = c21.survival_monte_carlo()
    s1 = Scene(
        id="MuonDecay", title="Dilatación del tiempo: muones atmosféricos",
        statement=("Muones con v = 0.995c y vida media propia τ = 2.197 µs se producen a 10 km de altura. Hallar qué fracción llega al suelo con "
                   "y sin dilatación del tiempo."),
        foundations=[(r"\Delta t=\gamma\Delta t_0,\qquad\gamma=\dfrac1{\sqrt{1-v^2/c^2}}", "Dilatación del tiempo."),
                     (r"P=\exp\!\left(-\dfrac d{v\gamma\tau}\right)", "Supervivencia de una población que decae.")],
        analytic=[(rf"\gamma={c21.gamma(beta):.2f},\quad P_{{rel}}={p_rel:.4f}", "Con dilatación."),
                  (rf"P_{{cl}}={p_cl:.2e}", "Sin dilatación.")],
        numeric_method="Monte Carlo con 2×10⁶ vidas exponenciales en el laboratorio (media γτ): se cuenta la fracción que supera el tiempo de vuelo d/(βc).",
        rows=[Row("Fracción que sobrevive (con dilatación)", r"P_{rel}", p_rel, p_mc, "", 5)],
        notes=[f"Incertidumbre estadística del Monte Carlo: {err:.5f}. Sin dilatación el suceso es demasiado raro (P = {p_cl:.1e}) para estimarlo con muestreo directo."],
    )
    s2 = Scene(
        id="VelocityAndEnergy", title="Adición de velocidades y energía cinética relativista",
        statement=("Una nave a 0.8c lanza un proyectil a 0.8c respecto de la nave en la misma dirección. Hallar su velocidad respecto de la Tierra y "
                   "comparar las energías cinéticas relativista y clásica a 0.9c."),
        foundations=[(r"u'=\dfrac{u+v}{1+uv/c^2}", "Adición relativista de velocidades."),
                     (r"K=(\gamma-1)mc^2=\int_0^vv\,dp,\quad p=\gamma mv", "Energía cinética relativista.")],
        analytic=[(rf"u'={c21.velocity_addition_analytic(0.8, 0.8):.5f}\,c", "Velocidad compuesta."),
                  (rf"K_{{rel}}(0.9c)={c21.kinetic_analytic(0.9)/c21.C**2:.4f}\,mc^2", "Energía cinética relativista.")],
        numeric_method="Composición de dos matrices de Lorentz 2×2 en (ct, x) y cuadratura de Simpson de ∫v dp con 4000 subintervalos.",
        rows=[Row("Velocidad compuesta (matrices)", "u'", c21.velocity_addition_analytic(0.8, 0.8), c21.velocity_addition_numeric(0.8, 0.8), "c", 6),
              Row("Velocidad compuesta (rapideces)", "u'", c21.velocity_addition_analytic(0.8, 0.8), c21.rapidity_addition(0.8, 0.8), "c", 6),
              Row("Energía cinética a 0.9c", "K", c21.kinetic_analytic(0.9) / c21.C**2, c21.kinetic_numeric(0.9) / c21.C**2, "mc²", 5)],
        notes=[f"La energía cinética clásica a 0.9c sería {c21.kinetic_classical(0.9)/c21.C**2:.4f} mc², menos de la mitad de la relativista."],
    )
    return Chapter(number=21, volume=1, part="Oscilaciones y ondas", slug="ch21", module="ch21_relativity", title="Teoría especial de la relatividad",
                   summary="Postulados de Einstein, dilatación del tiempo, transformación de Lorentz y dinámica relativista.",
                   concepts=["Postulados de la relatividad especial", "Dilatación del tiempo", "Contracción de la longitud", "Transformaciones de Lorentz",
                             "Energía y momento relativistas"], scenes=[s1, s2])


def _ch22() -> Chapter:
    L0 = c22.L0
    tf = c22.final_temperature_analytic()
    lam = c22.rate_analytic()
    t, y = c22.temperatures_numeric(100.0, 2000)
    t2, y2 = c22.temperatures_numeric(300.0, 6000)
    s1 = Scene(
        id="ThermalExpansion", title="Dilatación térmica con coeficiente variable",
        statement=("Un riel de acero de 30 m (α₀ = 12×10⁻⁶ K⁻¹ a 20 °C) se calienta hasta 120 °C. Calcular ΔL con α constante y con "
                   "α(T) = α₀[1 + 0.002 (T − 20 °C)]."),
        foundations=[(r"\Delta L=\alpha L_0\Delta T", "Dilatación lineal."), (r"\dfrac{dL}{dT}=\alpha(T)L", "Ecuación diferencial con α variable.")],
        analytic=[(rf"\Delta L_{{lineal}}={(c22.length_linear()-L0)*1000:.2f}\ \mathrm{{mm}}", "α constante."),
                  (rf"\Delta L_{{exacto}}={(c22.length_analytic()-L0)*1000:.2f}\ \mathrm{{mm}}", "L = L₀ exp[α₀(ΔT + βΔT²/2)].")],
        numeric_method="RK4 de dL/dT = α(T) L de 20 a 120 °C con 400 pasos.",
        rows=[Row("ΔL a 120 °C (α variable)", r"\Delta L", (c22.length_analytic() - L0) * 1000, (c22.length_numeric() - L0) * 1000, "mm", 5),
              Row("ΔL a 70 °C (α variable)", r"\Delta L", (c22.length_analytic(70.0) - L0) * 1000, (c22.length_numeric(70.0) - L0) * 1000, "mm", 5)],
        notes=[f"Con α constante el resultado es {(c22.length_linear()-L0)*1000:.2f} mm: subestima la dilatación en {((c22.length_analytic()-c22.length_linear())*1000):.2f} mm."],
    )
    s2 = Scene(
        id="ThermalEquilibrium", title="Equilibrio térmico entre dos cuerpos",
        statement=("Un bloque de aluminio de 0.30 kg a 150 °C se sumerge en 0.50 kg de agua a 20 °C; la conductancia térmica entre ambos es 20 W/K. "
                   "Hallar la temperatura final y la constante de tiempo."),
        foundations=[(r"C_1\dfrac{dT_1}{dt}=-h(T_1-T_2),\quad C_2\dfrac{dT_2}{dt}=h(T_1-T_2)", "Intercambio de calor entre dos cuerpos.")],
        analytic=[(rf"T_f=\dfrac{{C_1T_1+C_2T_2}}{{C_1+C_2}}={tf:.2f}\ ^\circ\mathrm{{C}}", "Temperatura final."),
                  (rf"\lambda=h\left(\tfrac1{{C_1}}+\tfrac1{{C_2}}\right)={lam:.4f}\ \mathrm{{s^{{-1}}}},\quad\tau={1/lam:.1f}\ \mathrm{{s}}", "Constante de tiempo.")],
        numeric_method="RK4 del sistema acoplado de dos ecuaciones (2000 pasos hasta 100 s y 6000 hasta 300 s para el valor final).",
        rows=[Row("Temperatura final", "T_f", tf, float(y2[-1, 0]), "°C", 4),
              Row("Temperatura del aluminio a 20 s", "T_1", float(c22.temperatures_analytic(20.0)[0]), float(np.interp(20, t, y[:, 0])), "°C", 4),
              Row("Temperatura del agua a 20 s", "T_2", float(c22.temperatures_analytic(20.0)[1]), float(np.interp(20, t, y[:, 1])), "°C", 4)],
    )
    return Chapter(number=22, volume=1, part="Termodinámica", slug="ch22", module="ch22_temperature", title="Temperatura y ley cero",
                   summary="Equilibrio térmico, escalas de temperatura y dilatación térmica.",
                   concepts=["Equilibrio térmico y ley cero", "Medición de la temperatura", "Escalas Celsius y Fahrenheit", "Dilatación térmica"], scenes=[s1, s2])


def _ch23() -> Chapter:
    rows = c23.first_law_table()
    o = c23.otto_numeric()
    eta_an = c23.otto_efficiency_analytic()
    s1 = Scene(
        id="GasProcesses", title="Procesos de un gas ideal: trabajo, calor y energía interna",
        statement=("Un mol de gas ideal monoatómico a 2.0×10⁵ Pa y 10 L se expande hasta 20 L por tres caminos: isotérmico, isobárico y adiabático. "
                   "Hallar W, ΔU y Q en cada uno."),
        foundations=[(r"\Delta U=Q-W,\qquad W=\int p\,dV", "Primera ley."), (r"\Delta U=nC_V\Delta T,\quad C_V=\tfrac32R", "Energía interna del gas monoatómico.")],
        analytic=[(rf"W_{{isot}}={c23.work_analytic('isothermal'):.1f}\ \mathrm{{J}},\quad W_{{isob}}={c23.work_analytic('isobaric'):.1f}\ \mathrm{{J}},\quad W_{{ad}}={c23.work_analytic('adiabatic'):.1f}\ \mathrm{{J}}", "Trabajos.")],
        numeric_method="Simpson de ∫p dV con 400 subintervalos; en la adiabática, RK4 de dp/dV = −γp/V como camino independiente de la forma cerrada.",
        rows=[Row(f"Trabajo {r_['kind']}", "W", c23.work_analytic(r_["kind"]), r_["W"], "J", 3) for r_ in rows]
              + [Row("Calor adiabático (primera ley)", "Q", 0.0, rows[2]["Q"], "J", 8)],
    )
    s2 = Scene(
        id="OttoCycle", title="Ciclo de Otto",
        statement=("Un motor de gasolina ideal con gas diatómico (γ = 7/5) y relación de compresión 8 parte de 300 K y 25 L (1 mol) y absorbe calor "
                   "a volumen constante hasta 1800 K. Hallar el trabajo neto por ciclo y la eficiencia."),
        foundations=[(r"\oint dU=0\;\Rightarrow\;W_{neto}=Q_{entra}-Q_{sale}", "Ciclo completo."),
                     (r"\eta=\dfrac{W_{neto}}{Q_{entra}}=1-r^{1-\gamma}", "Eficiencia del ciclo de Otto.")],
        analytic=[(rf"\eta=1-8^{{-0.4}}={eta_an:.4f}", "Eficiencia."),
                  (rf"Q_{{entra}}=nC_V(T_C-T_B)={o['Q_in']:.0f}\ \mathrm{{J}}", "Calor absorbido.")],
        numeric_method="Adiabáticas por RK4 sobre dp/dV = −γp/V (800 pasos) y trabajo neto por Simpson sobre cada tramo; Q_entra de la isocora.",
        rows=[Row("Eficiencia", r"\eta", eta_an, o["eta"], "", 6), Row("Trabajo neto", r"W_{neto}", eta_an * o["Q_in"], o["W_net"], "J", 3)],
    )
    return Chapter(number=23, volume=1, part="Termodinámica", slug="ch23", module="ch23_first_law", title="Primera ley de la termodinámica",
                   summary="Calor, trabajo termodinámico y energía interna; procesos y ciclos de un gas ideal.",
                   concepts=["Calor y trabajo", "Energía interna", "Primera ley", "Procesos isotérmico, isobárico y adiabático", "Ciclos"], scenes=[s1, s2])


def _ch24() -> Chapter:
    n = 4000
    p_sim, t_sim = c24.pressure_simulation(n=n)
    an = c24.speeds_analytic()
    sp = c24.speeds_sampled()
    q = c24.moments_by_quadrature()
    s1 = Scene(
        id="GasPressure", title="Presión de un gas por transferencia de momento",
        statement=("4000 átomos de helio a 300 K se mueven libremente en un cubo de 1 µm de lado y rebotan elásticamente en las paredes. Calcular la presión "
                   "por el impulso transferido a una pared y compararla con p = NkT/V."),
        foundations=[(r"p=\dfrac13\dfrac NVm\langle v^2\rangle=\dfrac{NkT}V", "Presión del gas ideal."), (r"\tfrac12m\langle v^2\rangle=\tfrac32kT", "Temperatura cinética.")],
        analytic=[(rf"p=\dfrac{{NkT}}V={c24.pressure_analytic(n):.3f}\ \mathrm{{Pa}}", "Presión esperada a 300 K.")],
        numeric_method="Simulación de 4000 partículas independientes con pasos de 2 ps y reflexión elástica; la presión es el impulso total en la pared x = L dividido por tiempo y área. T se mide de la energía cinética de la simulación.",
        rows=[Row("Presión (con T medida)", "p", c24.pressure_analytic(n, t_sim), p_sim, "Pa", 4),
              Row("Temperatura medida", "T", 300.0, t_sim, "K", 2)],
        notes=["La diferencia entre T medida y 300 K es fluctuación estadística de una muestra de 4000 partículas."],
    )
    s2 = Scene(
        id="MaxwellBoltzmann", title="Distribución de rapideces de Maxwell-Boltzmann",
        statement=("Nitrógeno (N₂) a 300 K. Hallar las rapideces más probable, media y cuadrática media, y verificarlas con un muestreo de 400 000 moléculas y "
                   "con cuadratura de la densidad."),
        foundations=[(r"f(v)=4\pi\left(\dfrac m{2\pi kT}\right)^{3/2}v^2e^{-mv^2/2kT}", "Densidad de rapideces."),
                     (r"v_p=\sqrt{\tfrac{2kT}m},\ \bar v=\sqrt{\tfrac{8kT}{\pi m}},\ v_{rms}=\sqrt{\tfrac{3kT}m}", "Rapideces características.")],
        analytic=[(rf"v_p={an['v_mp']:.1f},\ \bar v={an['v_mean']:.1f},\ v_{{rms}}={an['v_rms']:.1f}\ \mathrm{{m/s}}", "Valores a 300 K.")],
        numeric_method="Muestreo de las tres componentes con distribución normal (400 000 moléculas) y cuadratura de Simpson de los momentos de f(v) con 4000 subintervalos.",
        rows=[Row("Rapidez media (muestreo)", r"\bar v", float(an["v_mean"]), sp["v_mean"], "m/s", 2),
              Row("Rapidez rms (muestreo)", r"v_{rms}", float(an["v_rms"]), sp["v_rms"], "m/s", 2),
              Row("Rapidez media (cuadratura)", r"\bar v", float(an["v_mean"]), q["v_mean"], "m/s", 4),
              Row("Normalización", r"\int f\,dv", 1.0, q["norm"], "", 6)],
    )
    return Chapter(number=24, volume=1, part="Termodinámica", slug="ch24", module="ch24_kinetic_theory", title="Teoría cinética de los gases",
                   summary="Presión y temperatura desde el movimiento molecular y distribución de rapideces.",
                   concepts=["Modelo del gas ideal", "Cálculo cinético de la presión", "Interpretación cinética de la temperatura",
                             "Distribución de rapideces de Maxwell"], scenes=[s1, s2])


def _ch25() -> Chapter:
    o = c25.carnot_numeric()
    eta_an = c25.carnot_efficiency_analytic()
    s_h_a, s_h_n = c25.heating_entropy_analytic(), c25.heating_entropy_numeric()
    s_c_a, s_c_n = c25.contact_entropy_analytic(), c25.contact_entropy_numeric()
    s1 = Scene(
        id="CarnotCycle", title="Ciclo de Carnot",
        statement=("Una máquina de Carnot con 1 mol de gas ideal monoatómico opera entre 500 K y 300 K; en la isoterma caliente el volumen va de 1 L a 2 L. "
                   "Hallar el trabajo neto y la eficiencia."),
        foundations=[(r"\eta=1-\dfrac{Q_c}{Q_h},\qquad\eta_C=1-\dfrac{T_c}{T_h}", "Eficiencia de una máquina térmica."),
                     (r"pV=\text{cte},\qquad pV^\gamma=\text{cte}", "Isotermas y adiabatas.")],
        analytic=[(rf"\eta=1-\dfrac{{300}}{{500}}={eta_an:.3f}", "Eficiencia de Carnot."),
                  (rf"W=\eta Q_h={eta_an*o['Q_h']:.1f}\ \mathrm{{J}}", "Trabajo neto con Q_h = nRT_h ln 2.")],
        numeric_method="Simpson de p dV en las isotermas y RK4 de dp/dV = −γp/V en las adiabatas (2000 pasos); se suma el trabajo de los cuatro tramos.",
        rows=[Row("Eficiencia", r"\eta", eta_an, o["eta"], "", 6), Row("Trabajo neto", r"W", eta_an * o["Q_h"], o["W_net"], "J", 3),
              Row("Cierre del ciclo en p_A", r"p_A", o["pa"] / 1e6, o["pa_end"] / 1e6, "MPa", 6)],
    )
    s2 = Scene(
        id="EntropyProduction", title="Cambio de entropía y producción de entropía",
        statement=("a) Un kilogramo de agua se calienta de 20 °C a 80 °C: hallar ΔS. b) Un bloque de aluminio caliente y agua fría se igualan por contacto: "
                   "hallar ΔS total."),
        foundations=[(r"dS=\dfrac{dQ_{rev}}T,\quad\Delta S=\int\dfrac{dQ}T", "Definición de entropía."), (r"\Delta S_{universo}\ge0", "Segunda ley.")],
        analytic=[(rf"\Delta S_a=mc\ln\dfrac{{T_2}}{{T_1}}={s_h_a:.1f}\ \mathrm{{J/K}}", "Calentamiento."),
                  (rf"\Delta S_b=C_1\ln\dfrac{{T_f}}{{T_1}}+C_2\ln\dfrac{{T_f}}{{T_2}}={s_c_a:.2f}\ \mathrm{{J/K}}", "Contacto térmico.")],
        numeric_method="Simpson de m c dT/T para el calentamiento; para el contacto, RK4 de la producción de entropía dS/dt = h(T₁ − T₂)²/(T₁T₂) junto con las temperaturas.",
        rows=[Row("ΔS del calentamiento", r"\Delta S", s_h_a, s_h_n, "J/K", 4), Row("ΔS del contacto", r"\Delta S", s_c_a, s_c_n, "J/K", 4)],
        notes=["El signo positivo de ΔS en el contacto refleja la irreversibilidad del proceso."],
    )
    return Chapter(number=25, volume=1, part="Termodinámica", slug="ch25", module="ch25_second_law", title="Segunda ley y entropía",
                   summary="Máquinas térmicas, ciclo de Carnot y entropía.",
                   concepts=["Procesos reversibles e irreversibles", "Máquinas térmicas", "Ciclo de Carnot", "Entropía", "Segunda ley"], scenes=[s1, s2])


def chapters() -> list[Chapter]:
    return [_ch18(), _ch19(), _ch20(), _ch21(), _ch22(), _ch23(), _ch24(), _ch25()]
