"""Volumen 1, capitulos 6 a 10."""
from __future__ import annotations

import numpy as np

from content.model import Chapter, Row, Scene
from physics import ch06_dynamics as c6
from physics import ch07_work_energy as c7
from physics import ch08_energy_conservation as c8
from physics import ch09_particle_systems as c9
from physics import ch10_collisions as c10


def _ch6() -> Chapter:
    th = c6.THETA_DEG
    a_an = c6.incline_acceleration_analytic(th)
    t, s, v = c6.simulate_incline(th, 2.0)
    thc_an, thc_nu = c6.critical_angle_analytic(), c6.critical_angle_numeric()
    vt, ta = c6.terminal_speed(), c6.tau()
    t95_an, t95_nu = c6.time_to_fraction_analytic(), c6.time_to_fraction_numeric()
    tt, yy = c6.drag_numeric(2000, "rk4", 8.0)
    row_conv = c6.drag_convergence()

    s1 = Scene(
        id="FrictionIncline",
        title="Fricción estática y cinética en un plano inclinado",
        statement=("Un bloque descansa sobre un plano inclinado 35° con μs = 0.45 y μk = 0.30. Determinar si desliza, "
                   "con qué aceleración y a partir de qué ángulo comienza a deslizar."),
        foundations=[
            (r"f_s\le\mu_sN,\qquad f_k=\mu_kN,\qquad N=mg\cos\theta", "Modelo de fricción seca."),
            (r"\sum F_\parallel=mg\sin\theta-f_k=ma", "Segunda ley a lo largo del plano."),
        ],
        analytic=[
            (rf"\tan\theta={np.tan(np.deg2rad(th)):.3f}>\mu_s=0.45\;\Rightarrow\;\text{{desliza}}", "Condición de deslizamiento."),
            (rf"a=g(\sin\theta-\mu_k\cos\theta)={a_an:.3f}\ \mathrm{{m/s^2}}", "Aceleración con fricción cinética."),
            (rf"\theta_c=\tan^{{-1}}\mu_s={thc_an:.2f}^\circ", "Ángulo crítico."),
        ],
        numeric_method=("RK4 con una máquina de estados: mientras v = 0 la fricción estática compensa el peso si no excede μs N; "
                        "en caso contrario el bloque desliza con fricción cinética. El ángulo crítico se halla por bisección "
                        "sobre el desplazamiento en 1 s."),
        rows=[
            Row("Aceleración", "a", a_an, float(v[-1] / 2.0), "m/s²", 4),
            Row("Desplazamiento a los 2 s", "s", 0.5 * a_an * 4.0, float(s[-1]), "m", 4),
            Row("Ángulo crítico", r"\theta_c", thc_an, thc_nu, "°", 3),
        ],
    )
    s2 = Scene(
        id="DragFall",
        title="Caída con resistencia lineal y rapidez terminal",
        statement=("Una esfera de 0.5 kg cae desde el reposo en un medio con b = 0.8 kg/s. Determinar v(t), la rapidez "
                   "terminal y el tiempo que tarda en alcanzar el 95 % de ella."),
        foundations=[
            (r"\vec F_{res}=-b\vec v,\qquad m\dfrac{dv}{dt}=mg-bv", "Resistencia proporcional a la velocidad."),
            (r"v_t=\dfrac{mg}{b},\qquad \tau=\dfrac{m}{b}", "Rapidez terminal y constante de tiempo."),
        ],
        analytic=[
            (r"v(t)=v_t\left(1-e^{-t/\tau}\right)", "Solución con v(0) = 0."),
            (rf"v_t={vt:.3f}\ \mathrm{{m/s}},\qquad\tau={ta:.3f}\ \mathrm{{s}}", "Valores del problema."),
            (rf"t_{{95}}=\tau\ln20={t95_an:.3f}\ \mathrm{{s}}", "Tiempo al 95 % de v_t."),
        ],
        numeric_method=("Integración RK4 de (y, v) con 4000 pasos; t95 por interpolación lineal del cruce. Con Δt = 0.5 s se compara "
                        "Euler (orden 1) con RK4 (orden 4)."),
        rows=[
            Row("Rapidez terminal (v a los 8 s)", "v_t", vt, float(yy[-1, 1]), "m/s", 4),
            Row("Tiempo al 95 %", "t_{95}", t95_an, t95_nu, "s", 4),
            Row("Error de Euler en v(1 s), n = 8", r"\varepsilon", 0.0, float(row_conv[0]["err_euler"]), "m/s", 3),
            Row("Error de RK4 en v(1 s), n = 8", r"\varepsilon", 0.0, float(row_conv[0]["err_rk4"]), "m/s", 6),
        ],
        notes=["Las dos últimas filas muestran el error absoluto de cada integrador: la solución exacta es el cero de referencia."],
    )
    return Chapter(
        number=6, volume=1, part="Mecánica clásica", slug="ch06", module="ch06_dynamics",
        title="Dinámica de la partícula",
        summary="Fricción estática y cinética, resistencia del medio y velocidad terminal.",
        concepts=["Fuerzas de fricción", "Propiedades de la fricción", "Fuerza de arrastre y velocidad terminal",
                  "Movimiento circular uniforme", "Dinámica del movimiento circular"],
        scenes=[s1, s2],
    )


def _ch7() -> Chapter:
    w = c7.spring_work_analytic()
    conv = c7.work_convergence((2, 32))
    v_an, v_nu = c7.final_speed_analytic(), c7.final_speed_numeric()
    s1 = Scene(
        id="SpringWork",
        title="Trabajo de una fuerza variable: el resorte",
        statement="Un resorte de k = 200 N/m se estira desde x = 0.10 m hasta x = 0.30 m. Calcular el trabajo del agente externo.",
        foundations=[
            (r"W=\int_{x_1}^{x_2}F(x)\,dx", "Trabajo de una fuerza que depende de la posición."),
            (r"F=kx\;\Rightarrow\;W=\tfrac12k\,(x_2^2-x_1^2)", "Ley de Hooke."),
        ],
        analytic=[(rf"W=\tfrac12(200)(0.30^2-0.10^2)={w:.2f}\ \mathrm{{J}}", "Integral de F = kx entre los extremos.")],
        numeric_method="Suma de Riemann por la izquierda, regla del trapecio y regla de Simpson, con 2, 4, 8, 16 y 32 subintervalos.",
        rows=[
            Row("Trabajo (Riemann, n = 32)", "W", w, c7.spring_work_numeric(32, "riemann"), "J", 4),
            Row("Trabajo (trapecio, n = 32)", "W", w, c7.spring_work_numeric(32, "trapezoid"), "J", 4),
            Row("Trabajo (Simpson, n = 32)", "W", w, c7.spring_work_numeric(32, "simpson"), "J", 4),
        ],
        notes=[f"Error de Riemann: {conv[0]['riemann']:.1e} J con n = 2 y {conv[1]['riemann']:.1e} J con n = 32 (orden 1)."],
    )
    s2 = Scene(
        id="WorkEnergyTheorem",
        title="Teorema trabajo-energía con fuerza variable",
        statement=("Un bloque de 2 kg parte del reposo sobre una superficie sin fricción y lo empuja F(x) = 30(1 − x/2) N, "
                   "de x = 0 a x = 2 m. Calcular su rapidez final."),
        foundations=[
            (r"W_{neto}=\Delta K=\tfrac12mv^2-\tfrac12mv_0^2", "Teorema trabajo-energía."),
            (r"W=\int F(x)\,dx,\qquad P=Fv", "Trabajo y potencia instantánea."),
        ],
        analytic=[
            (r"W=\int_0^2 30\left(1-\tfrac{x}{2}\right)dx=30\ \mathrm{J}", "El área bajo F(x) es un triángulo."),
            (rf"v=\sqrt{{2W/m}}={v_an:.3f}\ \mathrm{{m/s}}", "Rapidez final."),
        ],
        numeric_method="Integración RK4 de m x'' = F(x) hasta x = 2 m (paso de 1 ms) y cuadratura de Simpson del trabajo con 400 subintervalos.",
        rows=[
            Row("Rapidez final", "v", v_an, v_nu, "m/s", 4),
            Row("Trabajo total", "W", 30.0, c7.work_integral_numeric(), "J", 4),
        ],
        notes=[f"Potencia máxima numérica: {c7.max_power()[0]:.1f} W en x = {c7.max_power()[1]:.2f} m."],
    )
    return Chapter(number=7, volume=1, part="Mecánica clásica", slug="ch07", module="ch07_work_energy",
                   title="Trabajo y energía",
                   summary="Trabajo de fuerzas constantes y variables, energía cinética y el teorema trabajo-energía.",
                   concepts=["Trabajo de una fuerza constante", "Trabajo de una fuerza variable", "Energía cinética y teorema trabajo-energía",
                             "Potencia"], scenes=[s1, s2])


def _ch8() -> Chapter:
    v_an, v_nu = c8.bottom_speed_analytic(), c8.max_speed_numeric()
    drift = {m: c8.energy_drift(m, t_end=6.0, n=600) for m in ("euler", "verlet", "rk4")}
    an = c8.turning_points_analytic()
    _, _, _, turn = c8.spring_friction_numeric()
    d = c8.delta()
    s1 = Scene(
        id="PendulumEnergy",
        title="Conservación de la energía mecánica en un péndulo",
        statement=("Un péndulo de L = 1.5 m y m = 1 kg se suelta desde el reposo a 60°. Calcular la rapidez en el punto más bajo "
                   "y comprobar numéricamente la conservación de la energía."),
        foundations=[
            (r"U=mgL(1-\cos\theta),\qquad K=\tfrac12mv^2", "Energías potencial y cinética."),
            (r"E=K+U=\text{constante}", "Solo fuerzas conservativas."),
        ],
        analytic=[
            (r"h=L(1-\cos60^\circ)=0.750\ \mathrm{m}", "Altura inicial sobre el punto más bajo."),
            (rf"v=\sqrt{{2gh}}={v_an:.3f}\ \mathrm{{m/s}}", "Conservación de la energía."),
        ],
        numeric_method=("RK4, Euler y Verlet sobre θ'' = −(g/L) sen θ durante 6 s con 600 pasos. Se mide la deriva relativa máxima de la "
                        "energía total y la rapidez máxima con 6000 pasos."),
        rows=[Row("Rapidez en el punto más bajo", "v", v_an, v_nu, "m/s", 4),
              Row("Deriva de E con RK4", r"\Delta E/E", 0.0, drift["rk4"], "", 9)],
        notes=[f"Deriva relativa de la energía en 6 s: Euler {drift['euler']:.1e}, Verlet {drift['verlet']:.1e}, RK4 {drift['rk4']:.1e}."],
    )
    s2 = Scene(
        id="SpringFriction",
        title="Resorte con fricción: fuerza no conservativa",
        statement=("Un bloque de 1 kg unido a un resorte de k = 100 N/m se comprime 0.20 m sobre una superficie con μk = 0.25. "
                   "Hallar los puntos de retorno sucesivos y dónde se detiene."),
        foundations=[
            (r"W_{nc}=\Delta E_{mec}", "El trabajo de las fuerzas no conservativas es el cambio de energía mecánica."),
            (r"f_k=\mu_kmg,\qquad\delta=\dfrac{\mu_kmg}{k}", "Fricción cinética y reducción de amplitud."),
        ],
        analytic=[
            (rf"\delta={d:.4f}\ \mathrm{{m}},\qquad x_n=\pm(x_0-2n\delta)", "La amplitud disminuye 2δ por semiciclo."),
            (rf"\text{{se detiene en }}x={an[-1]:+.3f}\ \mathrm{{m}}", "El bloque se detiene cuando |x| ≤ δ."),
        ],
        numeric_method="RK4 con fricción −μmg·signo(v) (paso 0.1 ms), detección de retornos por interpolación del cambio de signo de v y parada por fricción estática.",
        rows=[Row(f"Punto de retorno n = {i}", f"x_{i}", an[i], turn[i], "m", 4) for i in range(len(an))],
    )
    return Chapter(number=8, volume=1, part="Mecánica clásica", slug="ch08", module="ch08_energy_conservation",
                   title="Conservación de la energía",
                   summary="Energía potencial, fuerzas conservativas y el papel de las fuerzas no conservativas.",
                   concepts=["Fuerzas conservativas", "Energía potencial", "Conservación de la energía mecánica",
                             "Trabajo de fuerzas no conservativas"], scenes=[s1, s2])


def _ch9() -> Chapter:
    cm_a, cm_n = c9.cm_discrete_analytic(), c9.cm_discrete_numeric()
    rod_a, rod_n = c9.rod_cm_analytic(), c9.rod_cm_numeric()
    mc, _ = c9.rod_cm_monte_carlo()
    an, nu, R = c9.fragments_analytic(), c9.fragments_numeric(), c9.range_analytic()
    s1 = Scene(
        id="CenterOfMass",
        title="Centro de masa de partículas y de una barra no uniforme",
        statement=("Tres partículas de 2, 3 y 5 kg están en (0, 0), (4, 0) y (2, 3) m. Una barra de 2 m tiene densidad lineal λ(x) = λ₀x/L. "
                   "Hallar el centro de masa de ambos sistemas."),
        foundations=[
            (r"\vec r_{cm}=\dfrac1M\sum_im_i\vec r_i", "Sistema discreto."),
            (r"x_{cm}=\dfrac{\int x\,dm}{\int dm},\quad dm=\lambda(x)\,dx", "Sistema continuo."),
        ],
        analytic=[
            (rf"(x_{{cm}},y_{{cm}})=({cm_a[0]:.2f},\ {cm_a[1]:.2f})\ \mathrm{{m}}", "Promedio ponderado por masa."),
            (rf"x_{{cm}}=\dfrac{{2L}}{{3}}={rod_a:.4f}\ \mathrm{{m}}", "Barra con λ proporcional a x."),
        ],
        numeric_method="Producto matricial para las partículas; cuadratura de Simpson (200 subintervalos) y Monte Carlo (400 000 muestras con densidad ∝ x) para la barra.",
        rows=[Row("x_cm de las partículas", "x_{cm}", cm_a[0], cm_n[0], "m", 4),
              Row("y_cm de las partículas", "y_{cm}", cm_a[1], cm_n[1], "m", 4),
              Row("x_cm de la barra (Simpson)", "x_{cm}", rod_a, rod_n, "m", 6),
              Row("x_cm de la barra (Monte Carlo)", "x_{cm}", rod_a, mc, "m", 4)],
    )
    s2 = Scene(
        id="ExplodingProjectile",
        title="Explosión de un proyectil en el punto más alto",
        statement=("Un proyectil de 4 kg se lanza a 20 m/s y 60°. En el punto más alto explota en un fragmento A (1 kg) que cae "
                   "verticalmente desde el reposo y otro B (3 kg). Hallar dónde cae cada uno y la posición final del centro de masa."),
        foundations=[
            (r"\vec F_{ext}=M\vec a_{cm}", "Solo las fuerzas externas mueven al centro de masa."),
            (r"\vec P=\text{constante en la explosión}", "Las fuerzas internas conservan el momento."),
        ],
        analytic=[
            (rf"v_B=\dfrac{{4(10)}}{{3}}={an['vb']:.2f}\ \mathrm{{m/s}}", "Conservación del momento horizontal."),
            (rf"x_A={an['xa']:.2f}\ \mathrm{{m}},\quad x_B={an['xb']:.2f}\ \mathrm{{m}}", "Posiciones de caída."),
            (rf"x_{{cm}}={an['x_cm']:.2f}\ \mathrm{{m}}=R", "El alcance sin explosión."),
        ],
        numeric_method="Integración RK4 de cada fragmento desde el apice (paso de 1 ms) y reconstrucción del centro de masa en el impacto.",
        rows=[Row("Posición de caída de A", "x_A", an["xa"], nu["xa"], "m", 4),
              Row("Posición de caída de B", "x_B", an["xb"], nu["xb"], "m", 4),
              Row("Centro de masa final", "x_{cm}", R, nu["x_cm"], "m", 4)],
    )
    return Chapter(number=9, volume=1, part="Mecánica clásica", slug="ch09", module="ch09_particle_systems",
                   title="Sistemas de partículas",
                   summary="Centro de masa, ímpetu lineal y su conservación en sistemas de partículas.",
                   concepts=["Centro de masa", "Segunda ley de Newton para un sistema", "Ímpetu lineal de una partícula y de un sistema",
                             "Conservación del ímpetu lineal"], scenes=[s1, s2])


def _ch10() -> Chapter:
    an, nu = c10.bullet_block_analytic(), c10.bullet_block_numeric()
    e_rows = []
    for z in (0.0, 0.3, 0.7):
        e_rows.append(Row(f"Coeficiente e, ζ = {z}", "e", c10.restitution_from_damping(z), c10.restitution_measured(z), "", 4))
    s1 = Scene(
        id="Collision1D",
        title="Colisión unidimensional y coeficiente de restitución",
        statement=("Un cuerpo de 2 kg a 3 m/s choca con otro de 1 kg en reposo. Hallar las velocidades finales para e = 1, 0.5 y 0, y "
                   "verificar el coeficiente de restitución con una simulación de contacto."),
        foundations=[
            (r"m_1v_1+m_2v_2=m_1v_1'+m_2v_2',\qquad e=\dfrac{v_2'-v_1'}{v_1-v_2}", "Momento y restitución."),
            (r"v_1'=\dfrac{m_1-em_2}{m_1+m_2}v_1,\quad v_2'=\dfrac{(1+e)m_1}{m_1+m_2}v_1", "Velocidades finales."),
        ],
        analytic=[
            (r"e=1:\ v_1'=1,\ v_2'=4\ \mathrm{m/s}\qquad e=0:\ v_1'=v_2'=2\ \mathrm{m/s}", "Casos elástico y perfectamente inelástico."),
            (r"e=\exp\!\left(-\dfrac{\pi\zeta}{\sqrt{1-\zeta^2}}\right)", "Restitución de un contacto resorte-amortiguador."),
        ],
        numeric_method="Dos partículas con contacto de resorte lineal y amortiguador (k = 10⁴ N/m) integradas con RK4 (paso de 10 µs); e se mide de las velocidades finales.",
        rows=e_rows,
    )
    s2 = Scene(
        id="BallisticPendulum",
        title="Péndulo balístico",
        statement=("Una bala de 20 g a 250 m/s se incrusta en un bloque de 2 kg colgado de una cuerda de 1.2 m. Hallar la velocidad del "
                   "conjunto, la altura alcanzada y la fracción de energía cinética perdida."),
        foundations=[
            (r"mv=(m+M)V", "Momento en la colisión perfectamente inelástica."),
            (r"\tfrac12(m+M)V^2=(m+M)gh", "Energía en el ascenso."),
        ],
        analytic=[
            (rf"V={an['V']:.3f}\ \mathrm{{m/s}},\qquad h={an['h']:.4f}\ \mathrm{{m}}", "Velocidad y altura."),
            (rf"\Delta K/K_0=\dfrac{{M}}{{m+M}}={an['loss_fraction']*100:.2f}\%", "Fracción de energía disipada."),
        ],
        numeric_method="Integración RK4 del péndulo rígido (paso de 0.1 ms) con velocidad inicial V hasta su primer máximo.",
        rows=[Row("Velocidad del conjunto", "V", an["V"], nu["V"], "m/s", 4),
              Row("Altura máxima", "h", an["h"], nu["h"], "m", 5),
              Row("Fracción de energía perdida", r"\Delta K/K_0", an["loss_fraction"], nu["loss_fraction"], "", 5)],
    )
    return Chapter(number=10, volume=1, part="Mecánica clásica", slug="ch10", module="ch10_collisions",
                   title="Colisiones",
                   summary="Impulso y colisiones elásticas e inelásticas en una dimensión.",
                   concepts=["Impulso", "Colisiones elásticas", "Colisiones inelásticas", "Coeficiente de restitución"], scenes=[s1, s2])


def chapters() -> list[Chapter]:
    return [_ch6(), _ch7(), _ch8(), _ch9(), _ch10()]
