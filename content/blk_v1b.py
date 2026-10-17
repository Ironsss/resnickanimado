"""Volumen 1, capitulos 11 a 17."""
from __future__ import annotations

import numpy as np

from content.model import Chapter, Row, Scene
from physics import ch11_rotational_kinematics as c11
from physics import ch12_rotational_dynamics as c12
from physics import ch13_angular_momentum as c13
from physics import ch14_equilibrium as c14
from physics import ch15_gravitation as c15
from physics import ch16_fluid_statics as c16
from physics import ch17_fluid_dynamics as c17

PART = "Mecánica clásica"


def _ch11() -> Chapter:
    tf = c11.T_END
    t, w, th = c11.spin_up_numeric()
    rim = c11.rim_accelerations(tf)
    _, _, th2 = c11.spin_up_numeric_variable()
    s1 = Scene(
        id="SpinUp", title="Rotación con aceleración angular constante",
        statement=("Una rueda de radio 0.40 m parte del reposo con α = 1.5 rad/s² durante 8 s. Hallar ω, el ángulo girado y las "
                   "aceleraciones de un punto del borde al final."),
        foundations=[
            (r"\omega=\omega_0+\alpha t,\qquad\theta=\omega_0t+\tfrac12\alpha t^2", "Cinemática rotacional con α constante."),
            (r"v=\omega r,\qquad a_t=\alpha r,\qquad a_c=\omega^2r", "Variables lineales de un punto del borde."),
        ],
        analytic=[
            (rf"\omega={c11.omega_analytic(tf):.1f}\ \mathrm{{rad/s}},\quad\theta={c11.theta_analytic(tf):.1f}\ \mathrm{{rad}}", "Valores a los 8 s."),
            (rf"a_t={rim['a_t']:.2f}\ \mathrm{{m/s^2}},\quad a_c={rim['a_c']:.1f}\ \mathrm{{m/s^2}},\quad|a|={rim['a']:.2f}\ \mathrm{{m/s^2}}", "Aceleraciones del borde."),
        ],
        numeric_method="Regla del trapecio acumulada aplicada dos veces a α(t) (400 pasos). Se repite con α(t) = 0.25 t para comprobar el método con aceleración variable.",
        rows=[Row("Velocidad angular final", r"\omega", c11.omega_analytic(tf), float(w[-1]), "rad/s", 4),
              Row("Ángulo girado", r"\theta", c11.theta_analytic(tf), float(th[-1]), "rad", 4),
              Row("Ángulo con α(t) = 0.25 t", r"\theta", tf**3 / 24, float(th2[-1]), "rad", 3)],
    )
    s2 = Scene(
        id="RollingCycloid", title="Rodadura sin deslizar y cicloide",
        statement="Una rueda de radio 0.50 m rueda sin deslizar. Hallar la longitud de un arco de la cicloide de un punto del borde y el área bajo él.",
        foundations=[
            (r"v_{cm}=\omega R", "Condición de rodadura sin deslizar."),
            (r"x=R(\theta-\sin\theta),\qquad y=R(1-\cos\theta)", "Trayectoria de un punto del borde."),
        ],
        analytic=[
            (rf"s=\int_0^{{2\pi}}2R\sin\tfrac\theta2\,d\theta=8R={c11.arch_length_analytic():.3f}\ \mathrm{{m}}", "Longitud de un arco."),
            (rf"A=\int y\,dx=3\pi R^2={c11.arch_area_analytic():.4f}\ \mathrm{{m^2}}", "Área bajo el arco."),
        ],
        numeric_method="Regla de Simpson (2000 subintervalos) sobre |dr/dθ| y sobre y dx/dθ, y longitud de la poligonal de la curva con 2000 segmentos.",
        rows=[Row("Longitud (Simpson)", "s", c11.arch_length_analytic(), c11.arch_length_numeric(), "m", 6),
              Row("Longitud (poligonal)", "s", c11.arch_length_analytic(), c11.arch_length_polyline(), "m", 6),
              Row("Área bajo el arco", "A", c11.arch_area_analytic(), c11.arch_area_numeric(), "m²", 6)],
    )
    return Chapter(number=11, volume=1, part=PART, slug="ch11", module="ch11_rotational_kinematics", title="Cinemática rotacional",
                   summary="Variables rotacionales, aceleración angular constante y relación entre movimiento circular y lineal.",
                   concepts=["Variables rotacionales", "Rotación con aceleración angular constante", "Cantidades rotacionales como vectores",
                             "Relaciones entre variables lineales y angulares"], scenes=[s1, s2])


def _ch12() -> Chapter:
    m, L = c12.M_ROD, c12.L_ROD
    ie_a, ic_a = c12.rod_inertia_analytic("end"), c12.rod_inertia_analytic("center")
    an, nu = c12.hanging_analytic(), c12.hanging_numeric()
    s1 = Scene(
        id="MomentOfInertia", title="Momentos de inercia por integración",
        statement=("Una barra uniforme de 3 kg y 1.2 m gira alrededor de un extremo y de su centro. Calcular I en ambos casos, "
                   "verificar el teorema de ejes paralelos y hallar I de un disco de 4 kg y radio 0.25 m."),
        foundations=[
            (r"I=\sum m_ir_i^2\;\to\;I=\int r^2\,dm", "Definición de momento de inercia."),
            (r"I=I_{cm}+Md^2", "Teorema de los ejes paralelos."),
        ],
        analytic=[
            (rf"I_{{extremo}}=\dfrac{{ML^2}}3={ie_a:.4f}\ \mathrm{{kg\,m^2}},\quad I_{{cm}}=\dfrac{{ML^2}}{{12}}={ic_a:.4f}\ \mathrm{{kg\,m^2}}", "Barra uniforme."),
            (rf"I_{{disco}}=\tfrac12MR^2={c12.disc_inertia_analytic():.5f}\ \mathrm{{kg\,m^2}}", "Disco uniforme."),
        ],
        numeric_method="Simpson sobre r² dm con dm = (M/L) dx (barra) y sobre anillos dm = σ 2π s ds (disco), con 400 subintervalos.",
        rows=[Row("Barra, eje en el extremo", "I", ie_a, c12.rod_inertia_numeric("end"), "kg m²", 6),
              Row("Barra, eje en el centro", "I", ic_a, c12.rod_inertia_numeric("center"), "kg m²", 6),
              Row("Disco, eje central", "I", c12.disc_inertia_analytic(), c12.disc_inertia_numeric(), "kg m²", 6),
              Row("Ejes paralelos: I_cm + M(L/2)²", "I", ie_a, c12.parallel_axis(c12.rod_inertia_numeric("center"), m, L / 2), "kg m²", 6)],
    )
    s2 = Scene(
        id="HangingMassDisc", title="Masa colgante y polea con inercia",
        statement=("Una masa de 2 kg cuelga de una cuerda enrollada en un disco uniforme de 4 kg y radio 0.25 m. Hallar la aceleración "
                   "de la masa, la tensión y la aceleración angular."),
        foundations=[
            (r"\sum\tau=I\alpha,\qquad\sum F=ma", "Segunda ley para rotación y traslación."),
            (r"a=\alpha R,\qquad I=\tfrac12MR^2", "Cuerda sin deslizar y disco uniforme."),
        ],
        analytic=[
            (rf"a=\dfrac{{mg}}{{m+M/2}}={an['a']:.3f}\ \mathrm{{m/s^2}},\quad T=m(g-a)={an['T']:.3f}\ \mathrm{{N}}", "Solución del sistema."),
            (rf"\alpha=a/R={an['alpha']:.2f}\ \mathrm{{rad/s^2}}", "Aceleración angular."),
        ],
        numeric_method="Sistema lineal 2×2 en (a, T) resuelto con numpy.linalg.solve; movimiento integrado con RK4 (1000 pasos) y balance de energía.",
        rows=[Row("Aceleración", "a", an["a"], nu["a"], "m/s²", 5), Row("Tensión", "T", an["T"], nu["T"], "N", 5),
              Row("Error de energía a 1 s", r"\Delta E", 0.0, nu["energy_error"], "J", 3)],
    )
    return Chapter(number=12, volume=1, part=PART, slug="ch12", module="ch12_rotational_dynamics", title="Dinámica rotacional",
                   summary="Torque, momento de inercia y la segunda ley de Newton para la rotación.",
                   concepts=["Torque sobre una partícula", "Momento de inercia", "Teorema de ejes paralelos", "Segunda ley para la rotación",
                             "Energía cinética de rotación"], scenes=[s1, s2])


def _ch13() -> Chapter:
    wf = c13.omega_final_analytic()
    dk = c13.kinetic_energy_change_analytic()
    t, w, inertia, ang, k = c13.skater_numeric()
    Om = c13.precession_analytic()
    Om_n = c13.precession_numeric()[3]
    Om2 = c13.precession_numeric(w_spin=400.0)[3]
    s1 = Scene(
        id="SkaterSpin", title="Conservación del momento angular: la patinadora",
        statement=("Una patinadora gira a 2 rad/s con I = 4.0 kg m². Al recoger los brazos, I pasa a 1.6 kg m². Hallar la velocidad angular "
                   "final y el cambio de energía cinética."),
        foundations=[
            (r"\vec L=I\vec\omega,\qquad\vec\tau_{ext}=\dfrac{d\vec L}{dt}", "Momento angular y segunda ley."),
            (r"\tau_{ext}=0\;\Rightarrow\;I_i\omega_i=I_f\omega_f", "Conservación."),
        ],
        analytic=[(rf"\omega_f=\dfrac{{I_i\omega_i}}{{I_f}}={wf:.2f}\ \mathrm{{rad/s}}", "Velocidad angular final."),
                  (rf"\Delta K=K_f-K_i={dk:.1f}\ \mathrm{{J}}", "Trabajo interno de los músculos.")],
        numeric_method="Sin torque, d(Iω)/dt = 0 implica ω' = −(I'/I) ω; se integra con RK4 (4000 pasos) con I(t) suavizado entre 0 y 2 s.",
        rows=[Row("Velocidad angular final", r"\omega_f", wf, float(w[-1]), "rad/s", 5),
              Row("Variación de L", r"\Delta L", 0.0, float(np.ptp(ang)), "kg m²/s", 12),
              Row("Cambio de energía cinética", r"\Delta K", dk, float(k[-1] - k[0]), "J", 4)],
    )
    s2 = Scene(
        id="GyroscopePrecession", title="Precesión de un giróscopo",
        statement=("Un giróscopo de 0.5 kg tiene su centro de masa a 0.10 m del pivote y gira con ω = 200 rad/s (I = 2.5×10⁻³ kg m²). "
                   "Hallar su velocidad de precesión y cómo cambia con ω."),
        foundations=[
            (r"\vec\tau=\vec r\times m\vec g,\qquad\dfrac{d\vec L}{dt}=\vec\tau", "El torque cambia la dirección de L."),
            (r"\Omega=\dfrac{mgr}{I\omega}", "Velocidad de precesión."),
        ],
        analytic=[(rf"L=I\omega={c13.I_GYRO*c13.W_SPIN:.2f}\ \mathrm{{kg\,m^2/s}},\quad\tau=mgr={c13.M_GYRO*c13.G*c13.R_GYRO:.3f}\ \mathrm{{N\,m}}", "Momento angular y torque."),
                  (rf"\Omega=\tau/L={Om:.3f}\ \mathrm{{rad/s}}", "Precesión.")],
        numeric_method="Integración RK4 de dL/dt = r × mg en 3D (6000 pasos) y ajuste lineal del ángulo azimutal de L para obtener Ω.",
        rows=[Row("Precesión con ω = 200 rad/s", r"\Omega", Om, Om_n, "rad/s", 4),
              Row("Precesión con ω = 400 rad/s", r"\Omega", c13.precession_analytic(400.0), Om2, "rad/s", 4)],
    )
    return Chapter(number=13, volume=1, part=PART, slug="ch13", module="ch13_angular_momentum", title="Momento angular",
                   summary="Momento angular de una partícula y de un cuerpo rígido, su conservación y los giróscopos.",
                   concepts=["Momento angular de una partícula", "Segunda ley para sistemas de partículas",
                             "Conservación del momento angular", "Giróscopos y precesión"], scenes=[s1, s2])


def _ch14() -> Chapter:
    dmax_a, dmax_n = c14.d_max_analytic(), c14.d_max_numeric()
    amin_a, amin_n = c14.min_angle_analytic(), c14.min_angle_numeric()
    ra, rb = c14.beam_reactions_analytic(2.0)
    s1 = Scene(
        id="LadderFriction", title="Escalera apoyada en una pared lisa",
        statement=("Una escalera de 5 m y 150 N se apoya a 60° en una pared lisa; el suelo tiene μs = 0.40. Un hombre de 700 N sube por ella. "
                   "Hallar hasta qué distancia puede subir sin que resbale y el ángulo mínimo para llegar al extremo."),
        foundations=[(r"\sum F_x=0,\quad\sum F_y=0,\quad\sum\tau=0", "Condiciones de equilibrio."),
                     (r"f\le\mu_sN_{suelo}", "Fricción estática máxima.")],
        analytic=[(r"N_{pared}=f=\dfrac{(W_eL/2+W_hd)\cos\theta}{L\sin\theta}", "Torque respecto al pie."),
                  (rf"d_{{max}}=\dfrac{{\mu_s(W_e+W_h)L\tan\theta-W_eL/2}}{{W_h}}={dmax_a:.3f}\ \mathrm{{m}}", "Distancia máxima.")],
        numeric_method="Sistema lineal 3×3 (Fx, Fy, torque) con numpy.linalg.solve para cada d; d_max y θ_min por bisección sobre la condición f ≤ μs N.",
        rows=[Row("Distancia máxima", r"d_{max}", dmax_a, dmax_n, "m", 5), Row("Ángulo mínimo", r"\theta_{min}", amin_a, amin_n, "°", 4),
              Row("Fuerza de fricción en d = 2.5 m", "f", c14.ladder_forces_analytic(2.5)["f"], c14.ladder_forces_numeric(2.5)["f"], "N", 4)],
    )
    s2 = Scene(
        id="BeamReactions", title="Reacciones de una viga con carga móvil",
        statement=("Una viga uniforme de 400 N y 6 m descansa sobre dos apoyos en sus extremos. Una persona de 700 N camina sobre ella. "
                   "Hallar las reacciones en función de su posición."),
        foundations=[(r"R_A+R_B=W_b+W_p", "Equilibrio de fuerzas."),
                     (r"R_BL=W_b\tfrac L2+W_px", "Torques respecto al apoyo A.")],
        analytic=[(r"R_B=\dfrac{W_b}2+W_p\dfrac xL,\qquad R_A=\dfrac{W_b}2+W_p\left(1-\dfrac xL\right)", "Reacciones."),
                  (rf"x=2\ \mathrm{{m}}:\ R_A={ra:.1f}\ \mathrm{{N}},\ R_B={rb:.1f}\ \mathrm{{N}}", "Ejemplo.")],
        numeric_method="Sistema lineal 2×2 con torques respecto a tres pivotes distintos (0, 3 y 6 m): el resultado no depende del eje.",
        rows=[Row(f"R_A con pivote en {p:.0f} m", "R_A", ra, c14.beam_reactions_numeric(2.0, p)[0], "N", 4) for p in (0.0, 3.0, 6.0)]
              + [Row("R_B con pivote en 6 m", "R_B", rb, c14.beam_reactions_numeric(2.0, 6.0)[1], "N", 4)],
    )
    return Chapter(number=14, volume=1, part=PART, slug="ch14", module="ch14_equilibrium", title="Equilibrio de los cuerpos rígidos",
                   summary="Condiciones de equilibrio estático, centro de gravedad y aplicaciones a escaleras y vigas.",
                   concepts=["Condiciones de equilibrio", "Centro de gravedad", "Ejemplos de equilibrio", "Elasticidad"], scenes=[s1, s2])


def _ch15() -> Chapter:
    T_an, T_nu = c15.period_analytic(), c15.period_numeric()
    t, y = c15.orbit_numeric(revolutions=1.0)
    area = c15.areal_velocity(t, y)
    en = c15.energy(y)
    rm = c15.rmax_analytic(8000.0)
    v_a, v_n = c15.escape_speed_analytic(), c15.escape_speed_numeric()
    s1 = Scene(
        id="KeplerOrbit", title="Órbita de Kepler: periodo y áreas barridas",
        statement=("Un planeta orbita el Sol con semieje a = 2 UA y excentricidad e = 0.5, partiendo del perihelio. Hallar su periodo y "
                   "verificar numéricamente la segunda y la tercera ley de Kepler."),
        foundations=[(r"F=\dfrac{GMm}{r^2}", "Gravitación universal."),
                     (r"T^2=a^3\quad(\text{UA, años})", "Tercera ley con GM = 4π².")],
        analytic=[(rf"T=a^{{3/2}}={T_an:.4f}\ \text{{años}}", "Periodo."),
                  (rf"v_p=\sqrt{{\dfrac{{GM(1+e)}}{{a(1-e)}}}}={c15.perihelion_speed():.3f}\ \text{{UA/año}}", "Velocidad en el perihelio.")],
        numeric_method="RK4 de r'' = −GM r/|r|³ con 8000 pasos por órbita; el periodo se mide en el segundo paso por el perihelio con un ajuste parabólico.",
        rows=[Row("Periodo", "T", T_an, T_nu, "años", 5), Row("Variación de dA/dt", r"\Delta(dA/dt)", 0.0, float(np.ptp(area) / abs(area[0])), "", 9),
              Row("Variación de la energía", r"\Delta E/E", 0.0, float(np.ptp(en) / abs(en[0])), "", 9)],
    )
    s2 = Scene(
        id="EscapeVelocity", title="Lanzamiento radial y velocidad de escape",
        statement=("Un proyectil se lanza radialmente desde la superficie de la Tierra con rapidez v₀. Hallar la distancia máxima para v₀ = 8 km/s "
                   "y la velocidad de escape."),
        foundations=[(r"U(r)=-\dfrac{GMm}r,\qquad E=\tfrac12mv^2-\dfrac{GMm}r", "Energía potencial y total."),
                     (r"E=0\;\Rightarrow\;v_{esc}=\sqrt{2GM/R}", "Velocidad de escape.")],
        analytic=[(rf"r_{{max}}=\dfrac{{R}}{{1-v_0^2R/(2GM)}}={rm/c15.R_EARTH:.3f}\,R", "Distancia máxima a 8 km/s."),
                  (rf"v_{{esc}}={v_a/1000:.3f}\ \mathrm{{km/s}}", "Velocidad de escape terrestre.")],
        numeric_method="RK4 de r'' = −GM/r² para v₀ = 6, 8, 10 y 11 km/s; v_esc por bisección sobre v₀ con el criterio de alcanzar 10⁴ R.",
        rows=[Row("r_max/R a 8 km/s", r"r_{max}/R", rm / c15.R_EARTH, c15.rmax_numeric(8000.0, t_end=40000.0, n=40000) / c15.R_EARTH, "", 4),
              Row("r_max/R a 10 km/s", r"r_{max}/R", c15.rmax_analytic(10000.0) / c15.R_EARTH, c15.rmax_numeric(10000.0, t_end=40000.0, n=40000) / c15.R_EARTH, "", 4),
              Row("Velocidad de escape", r"v_{esc}", v_a / 1000, v_n / 1000, "km/s", 4)],
    )
    return Chapter(number=15, volume=1, part=PART, slug="ch15", module="ch15_gravitation", title="Gravitación",
                   summary="Ley de la gravitación universal, energía potencial gravitacional, leyes de Kepler y movimiento planetario.",
                   concepts=["Ley de la gravitación universal", "Energía potencial gravitacional", "Velocidad de escape",
                             "Leyes de Kepler", "Movimiento de planetas y satélites"], scenes=[s1, s2])


def _ch16() -> Chapter:
    p_w = c16.water_pressure(100.0)
    h_n, p_n = c16.atmosphere_numeric()
    p5_an = float(c16.atmosphere_analytic(5000.0))
    p5_nu = float(np.interp(5000.0, h_n, p_n))
    h_a, h_b = c16.submerged_depth_analytic(), c16.equilibrium_bisection()
    w_a, T_n = c16.omega_analytic(), c16.period_numeric()
    s1 = Scene(
        id="PressureDepth", title="Presión en un líquido y en la atmósfera",
        statement=("Calcular la presión absoluta a 100 m de profundidad en agua y la presión atmosférica a 5000 m de altura a 15 °C, "
                   "tratando la atmósfera como gas ideal isotermo."),
        foundations=[(r"\dfrac{dp}{dy}=-\rho g", "Ecuación fundamental."),
                     (r"p=p_0e^{-Mgh/RT}", "Gas ideal isotermo.")],
        analytic=[(rf"p=p_0+\rho gh={p_w/1e6:.3f}\ \mathrm{{MPa}}", "Agua a 100 m."),
                  (rf"p(5000)={p5_an/1000:.2f}\ \mathrm{{kPa}}", "Atmósfera a 5000 m.")],
        numeric_method="RK4 de dp/dh = −ρ(p) g con ρ = pM/RT (200 pasos hasta 10 km), comparado con un modelo de densidad constante.",
        rows=[Row("Presión a 5000 m", "p", p5_an / 1000, p5_nu / 1000, "kPa", 4),
              Row("Presión a 10 km", "p", float(c16.atmosphere_analytic(10000.0)) / 1000, float(p_n[-1]) / 1000, "kPa", 4)],
        notes=[f"Un modelo de densidad constante daría {float(c16.atmosphere_constant_density(5000.0))/1000:.2f} kPa a 5000 m, claramente incorrecto."],
    )
    s2 = Scene(
        id="FloatingCylinder", title="Flotación y oscilación vertical de un cilindro",
        statement=("Un cilindro de madera (ρ = 600 kg/m³, L = 0.50 m, A = 0.010 m²) flota vertical en agua. Hallar la profundidad sumergida de "
                   "equilibrio y el periodo de sus pequeñas oscilaciones verticales."),
        foundations=[(r"B=\rho_fgV_{sum}", "Principio de Arquímedes."),
                     (r"B=mg\;\Rightarrow\;h/L=\rho/\rho_f", "Condición de flotación.")],
        analytic=[(rf"h=\dfrac\rho{{\rho_f}}L={h_a:.3f}\ \mathrm{{m}}", "Profundidad de equilibrio."),
                  (rf"\omega=\sqrt{{\rho_fgA/m}}={w_a:.3f}\ \mathrm{{rad/s}},\quad T={2*np.pi/w_a:.3f}\ \mathrm{{s}}", "Oscilaciones pequeñas.")],
        numeric_method="Bisección sobre B(h) = mg para el equilibrio; RK4 de m y'' = mg − ρ_f g A y (40 000 pasos) para el periodo sin amortiguar.",
        rows=[Row("Profundidad de equilibrio", "h", h_a, h_b, "m", 6), Row("Periodo de oscilación", "T", 2 * np.pi / w_a, T_n, "s", 5)],
    )
    return Chapter(number=16, volume=1, part=PART, slug="ch16", module="ch16_fluid_statics", title="Estática de fluidos",
                   summary="Presión en un fluido, principio de Pascal y principio de Arquímedes.",
                   concepts=["Presión y densidad", "Variación de la presión en un fluido en reposo", "Principio de Pascal",
                             "Principio de Arquímedes"], scenes=[s1, s2])


def _ch17() -> Chapter:
    an = c17.venturi_analytic()
    dp = c17.P1 - an["p2"]
    q_a, q_n = c17.flow_from_pressure_drop_analytic(dp), c17.flow_from_pressure_drop_numeric(dp)
    te_a, te_n = c17.drain_time_analytic(), c17.drain_time_numeric()
    t, y = c17.height_numeric(2000)
    mid = len(t) // 2
    s1 = Scene(
        id="VenturiTube", title="Tubo de Venturi: continuidad y Bernoulli",
        statement=("Por un tubo horizontal de 20 cm de diámetro que se estrecha a 10 cm circula agua con Q = 0.030 m³/s y p₁ = 200 kPa. Hallar las "
                   "velocidades, la presión en la garganta y recuperar Q a partir de la caída de presión."),
        foundations=[(r"A_1v_1=A_2v_2", "Ecuación de continuidad."),
                     (r"p_1+\tfrac12\rho v_1^2=p_2+\tfrac12\rho v_2^2", "Ecuación de Bernoulli.")],
        analytic=[(rf"v_1={an['v1']:.3f}\ \mathrm{{m/s}},\quad v_2={an['v2']:.3f}\ \mathrm{{m/s}}", "Velocidades."),
                  (rf"p_2=p_1+\tfrac12\rho(v_1^2-v_2^2)={an['p2']/1000:.2f}\ \mathrm{{kPa}}", "Presión en la garganta.")],
        numeric_method="Bisección sobre Q de f(Q) = ½ρ(Q²/A₂² − Q²/A₁²) − Δp con la caída de presión Δp conocida.",
        rows=[Row("Caudal recuperado", "Q", q_a, q_n, "m³/s", 7), Row("Caudal de referencia", "Q", c17.Q, q_n, "m³/s", 7)],
    )
    s2 = Scene(
        id="TorricelliDrain", title="Vaciado de un depósito: ley de Torricelli",
        statement=("Un depósito abierto de 0.50 m² de sección contiene agua hasta 1.5 m. Se abre un orificio de 2.0 cm² en el fondo. Hallar la "
                   "velocidad inicial de salida, h(t) y el tiempo de vaciado."),
        foundations=[(r"v=\sqrt{2gh}", "Ley de Torricelli."), (r"A\dfrac{dh}{dt}=-a\sqrt{2gh}", "Balance de volumen.")],
        analytic=[(r"h(t)=\left(\sqrt{h_0}-\dfrac aA\sqrt{\dfrac g2}\,t\right)^2", "Solución."),
                  (rf"t_{{vac}}=\dfrac Aa\sqrt{{\dfrac{{2h_0}}g}}={te_a:.1f}\ \mathrm{{s}}", "Tiempo de vaciado.")],
        numeric_method="RK4 de dh/dt = −(a/A)√(2gh) con 4000 pasos; el vaciado se estima con una corrección local cerca de h = 0, donde la derivada diverge.",
        rows=[Row("Tiempo de vaciado", r"t_{vac}", te_a, te_n, "s", 2),
              Row("Altura a la mitad del tiempo", "h", float(c17.height_analytic(t[mid])), float(y[mid, 0]), "m", 6)],
    )
    return Chapter(number=17, volume=1, part=PART, slug="ch17", module="ch17_fluid_dynamics", title="Dinámica de fluidos",
                   summary="Fluido ideal, ecuación de continuidad, ecuación de Bernoulli y aplicaciones.",
                   concepts=["Conceptos generales del flujo de fluidos", "Trayectoria de una partícula de fluido",
                             "Ecuación de continuidad", "Ecuación de Bernoulli", "Aplicaciones de Bernoulli y continuidad"], scenes=[s1, s2])


def chapters() -> list[Chapter]:
    return [_ch11(), _ch12(), _ch13(), _ch14(), _ch15(), _ch16(), _ch17()]
