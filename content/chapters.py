"""Contenido de los capitulos 1 a 5.

Las filas de verificacion se calculan en tiempo de construccion con
`physics`, de modo que el sitio nunca muestra un numero que no se haya
recalculado.
"""
from __future__ import annotations

import numpy as np

from content.model import Chapter, Row, Scene
from physics import ch01_measurement as c1
from physics import ch03_motion_1d as m1d
from physics import ch02_vectors as vec
from physics import ch04_motion_2d as c4
from physics import ch05_newton as c5


# ---------------------------------------------------------------- capitulo 1
def _ch1() -> Chapter:
    dim = c1.pendulum_dimensionless_constant(0.01)
    den_an, den_mc = c1.density_analytic(), c1.density_monte_carlo()

    s1 = Scene(
        id="DimensionalAnalysis",
        title="Análisis dimensional del péndulo simple",
        statement=(
            "Determinar de qué magnitudes depende el periodo T de un péndulo simple "
            "usando únicamente coherencia dimensional, y fijar después la constante "
            "adimensional que el análisis dimensional no puede dar."
        ),
        foundations=[
            (r"[m]=\mathsf{M},\quad [L]=\mathsf{L},\quad [g]=\mathsf{L}\,\mathsf{T}^{-2},\quad [T]=\mathsf{T}",
             "Dimensiones de las magnitudes involucradas."),
            (r"T = k\,L^{a}\,g^{b}\,m^{c}",
             "Ley de potencias propuesta; k es adimensional."),
        ],
        analytic=[
            (r"\mathsf{M}:\; c=0 \qquad \mathsf{L}:\; a+b=0 \qquad \mathsf{T}:\; -2b=1",
             "Igualar los exponentes de cada dimensión da un sistema lineal."),
            (r"a=\tfrac12,\quad b=-\tfrac12,\quad c=0 \;\Rightarrow\; T = k\sqrt{L/g}",
             "La masa no interviene."),
        ],
        numeric_method=(
            "Los exponentes se obtienen resolviendo A x = d por mínimos cuadrados (SVD) con "
            "numpy.linalg.lstsq. La constante k se mide integrando la ecuación del péndulo, "
            r"$\ddot\theta=-(g/L)\sin\theta$, con RK4 (4000 pasos) y amplitud inicial de 0.01 rad."
        ),
        rows=[
            Row("Exponente de L", "a", 0.5, dim.exponents["L"], "", 6),
            Row("Exponente de g", "b", -0.5, dim.exponents["g"], "", 6),
            Row("Exponente de m", "c", 0.0, dim.exponents["m"], "", 6),
            Row("Constante de proporcionalidad", "k", 2 * np.pi, dim.k_numeric, "", 5),
        ],
        notes=["El valor numérico de k converge a 2π cuando la amplitud tiende a cero."],
    )

    s2 = Scene(
        id="ErrorPropagation",
        title="Propagación de incertidumbre en una densidad",
        statement=(
            "Un cilindro de aluminio se mide con m = 111.4 ± 0.1 g, r = 1.250 ± 0.005 cm y "
            "h = 8.40 ± 0.02 cm. Calcular su densidad y la incertidumbre asociada."
        ),
        foundations=[
            (r"\rho = \dfrac{m}{\pi r^{2} h}", "Densidad de un cilindro."),
            (r"\left(\dfrac{\sigma_\rho}{\rho}\right)^{2}=\left(\dfrac{\sigma_m}{m}\right)^{2}+\left(2\dfrac{\sigma_r}{r}\right)^{2}+\left(\dfrac{\sigma_h}{h}\right)^{2}",
             "Propagación de primer orden para magnitudes independientes."),
        ],
        analytic=[
            (rf"\dfrac{{\sigma_m}}{{m}}={den_an['rel_m']*100:.2f}\%,\quad 2\dfrac{{\sigma_r}}{{r}}={den_an['rel_r']*100:.2f}\%,\quad \dfrac{{\sigma_h}}{{h}}={den_an['rel_h']*100:.2f}\%",
             "La contribución del radio domina por el exponente 2."),
            (rf"\rho={den_an['rho']:.3f}\ \mathrm{{g/cm^3}},\qquad \sigma_\rho={den_an['sigma_quadrature']:.3f}\ \mathrm{{g/cm^3}}",
             f"La suma lineal de errores daría {den_an['sigma_worst_case']:.3f}, una cota conservadora."),
        ],
        numeric_method=(
            "Monte Carlo con 400 000 realizaciones: cada medición se muestrea de una distribución "
            "normal con su incertidumbre y se calcula ρ para cada terna (semilla fija)."
        ),
        rows=[
            Row("Densidad", r"\rho", den_an["rho"], den_mc["mean"], "g/cm³", 4),
            Row("Incertidumbre", r"\sigma_\rho", den_an["sigma_quadrature"], den_mc["std"], "g/cm³", 5),
        ],
    )

    return Chapter(
        number=1, slug="ch01", module="ch01_measurement", title="Medición",
        summary="Magnitudes, unidades, coherencia dimensional y cuantificación de la incertidumbre de una medida.",
        concepts=[
            "Patrones de longitud, tiempo y masa", "Sistema Internacional de unidades",
            "Conversión de unidades", "Cifras significativas y precisión",
            "Análisis dimensional", "Propagación de incertidumbre",
        ],
        scenes=[s1, s2],
        part="Mecánica clásica",
    )


# ---------------------------------------------------------------- capitulo 3 (movimiento 1D)
def _ch3m() -> Chapter:
    t0 = 2.0
    sec = m1d.secant_table(t0)[-1]
    a_fd = m1d.central_second_difference(t0, 1e-2)
    disp = m1d.displacement_from_velocity(0.0, 4.0, 32)
    summ = m1d.free_fall_summary()
    tt, yy = m1d.free_fall_numeric(2000, "rk4", t1=2.5)
    y3_an = float(m1d.free_fall_analytic(3.0)[0])
    y3_rk4 = float(m1d.free_fall_numeric(10, "rk4")[1][-1, 0])
    y3_eu = float(m1d.free_fall_numeric(10, "euler")[1][-1, 0])

    s1 = Scene(
        id="InstantaneousVelocity",
        title="Velocidad instantánea y área bajo v(t)",
        statement=(
            "Una partícula se mueve con x(t) = 2 + 3t − t²/2 + t³/10 (m, s). Determinar v y a en "
            "t = 2 s y el desplazamiento entre t = 0 y t = 4 s."
        ),
        foundations=[
            (r"v=\lim_{\Delta t\to0}\dfrac{\Delta x}{\Delta t}=\dfrac{dx}{dt},\qquad a=\dfrac{dv}{dt}", "Definiciones de velocidad y aceleración instantáneas."),
            (r"\Delta x=\int_{t_0}^{t_1} v\,dt", "El desplazamiento es el área bajo la curva v(t)."),
        ],
        analytic=[
            (r"v(t)=3-t+0.3\,t^{2}\;\Rightarrow\; v(2)=2.20\ \mathrm{m/s}", "Derivada de x(t)."),
            (r"a(t)=-1+0.6\,t\;\Rightarrow\; a(2)=0.20\ \mathrm{m/s^2}", "Derivada de v(t)."),
            (r"\Delta x=x(4)-x(0)=10.40\ \mathrm{m}", "Evaluación directa o integral de v(t)."),
        ],
        numeric_method=(
            "La velocidad se estima como pendiente de secantes con Δt = 1, 0.5, 0.1, 0.01 y 0.001 s; "
            "la aceleración, con la segunda diferencia centrada (h = 0.01 s); el desplazamiento, con "
            "la regla del trapecio sobre 32 subintervalos."
        ),
        rows=[
            Row("Velocidad en t = 2 s", "v(2)", float(m1d.v_analytic(t0)), sec["slope"], "m/s", 4),
            Row("Aceleración en t = 2 s", "a(2)", float(m1d.a_analytic(t0)), a_fd, "m/s²", 4),
            Row("Desplazamiento en [0, 4] s", r"\Delta x", disp["exact"], disp["trapezoid"], "m", 4),
        ],
        notes=["La suma de Riemann por la izquierda con 32 rectángulos da "
               f"{disp['riemann']:.4f} m; el trapecio converge más rápido."],
    )

    s2 = Scene(
        id="ConstantAcceleration",
        title="Aceleración constante y caída libre",
        statement=(
            "Una pelota se lanza verticalmente hacia arriba desde el suelo con v₀ = 20 m/s. "
            "Calcular la altura máxima, el tiempo de vuelo y la altura en t = 3 s (g = 9.81 m/s²)."
        ),
        foundations=[
            (r"v=v_0+at,\quad x=x_0+v_0t+\tfrac12at^{2},\quad v^{2}=v_0^{2}+2a(x-x_0)", "Ecuaciones cinemáticas para aceleración constante."),
            (r"a=-g,\qquad g=9.81\ \mathrm{m/s^2}", "Caída libre sin resistencia del aire."),
        ],
        analytic=[
            (rf"t_{{\max}}=\dfrac{{v_0}}{{g}}={summ['t_peak']:.3f}\ \mathrm{{s}},\quad y_{{\max}}=\dfrac{{v_0^2}}{{2g}}={summ['y_max']:.3f}\ \mathrm{{m}}", "En el punto más alto v = 0."),
            (rf"T=\dfrac{{2v_0}}{{g}}={summ['t_return']:.3f}\ \mathrm{{s}},\quad y(3)={y3_an:.3f}\ \mathrm{{m}}", "Subida y bajada tardan lo mismo."),
        ],
        numeric_method=(
            "Se integra el sistema y' = v, v' = −g con Euler y con RK4 usando 10 pasos hasta t = 3 s, y "
            "luego se refina el paso para medir el orden de convergencia. La altura máxima se toma del "
            "máximo de la solución RK4 con 2000 pasos."
        ),
        rows=[
            Row("Altura máxima", r"y_{\max}", summ["y_max"], float(yy[:, 0].max()), "m", 4),
            Row("Altura en t = 3 s (RK4, 10 pasos)", "y(3)", y3_an, y3_rk4, "m", 4),
            Row("Altura en t = 3 s (Euler, 10 pasos)", "y(3)", y3_an, y3_eu, "m", 4),
        ],
        notes=[
            "Euler es de primer orden: el error se divide por dos al duplicar el número de pasos.",
            "RK4 integra exactamente polinomios de grado cuatro o menor, por lo que el error es del orden del redondeo.",
        ],
    )

    return Chapter(
        number=3, slug="ch03", module="ch03_motion_1d", title="Movimiento en una dimensión",
        summary="Cinemática de la partícula: posición, velocidad y aceleración, con el cálculo diferencial e integral como lenguaje.",
        concepts=[
            "Cinemática de la partícula", "Velocidad media e instantánea",
            "Aceleración", "Movimiento con aceleración constante",
            "Cuerpos en caída libre", "Aceleración variable e integración",
        ],
        scenes=[s1, s2],
        part="Mecánica clásica",
    )


# ---------------------------------------------------------------- capitulo 2 (vectores)
def _ch2v() -> Chapter:
    ra, rn = vec.resultant_analytic(), vec.resultant_numeric()
    pa = vec.products_analytic()
    area_mc, _, _ = vec.parallelogram_area_monte_carlo()
    _, _, _, dots = vec.invariance_sweep()

    s1 = Scene(
        id="VectorAddition",
        title="Suma de vectores: método geométrico y por componentes",
        statement=(
            "Una persona camina 40 m a 30° sobre el eje x y luego 25 m a 120°. Determinar el "
            "módulo y la dirección de su desplazamiento resultante."
        ),
        foundations=[
            (r"A_x=A\cos\theta,\quad A_y=A\sin\theta,\quad A=\sqrt{A_x^2+A_y^2},\quad \theta=\tan^{-1}\dfrac{A_y}{A_x}", "Componentes cartesianas de un vector plano."),
            (r"\vec R=\vec A+\vec B\;\Longleftrightarrow\;R_x=A_x+B_x,\; R_y=A_y+B_y", "La suma opera componente a componente."),
        ],
        analytic=[
            (r"\gamma=120^\circ-30^\circ=90^\circ,\qquad R^2=A^2+B^2+2AB\cos\gamma", "Ley de los cosenos."),
            (rf"R={ra['magnitude']:.2f}\ \mathrm{{m}},\qquad \theta_R=30^\circ+\sin^{{-1}}\!\left(\dfrac{{B\sin\gamma}}{{R}}\right)={ra['angle_deg']:.2f}^\circ", "Ley de los senos para la dirección."),
        ],
        numeric_method="Se proyecta cada vector sobre los ejes, se suman las componentes con numpy y se recuperan módulo y ángulo con hypot y atan2.",
        rows=[
            Row("Módulo del resultante", "R", ra["magnitude"], rn["magnitude"], "m", 4),
            Row("Dirección del resultante", r"\theta_R", ra["angle_deg"], rn["angle_deg"], "°", 4),
        ],
    )

    s2 = Scene(
        id="VectorProducts",
        title="Producto escalar y vectorial",
        statement="Dados A = (4, 1) y B = (2, 3), calcular el ángulo entre ellos y el área del paralelogramo que definen.",
        foundations=[
            (r"\vec A\cdot\vec B=AB\cos\theta=A_xB_x+A_yB_y", "Producto escalar: proyección."),
            (r"\left|\vec A\times\vec B\right|=AB\sin\theta=\left|A_xB_y-A_yB_x\right|", "Producto vectorial: área del paralelogramo."),
        ],
        analytic=[
            (rf"\cos\theta=\dfrac{{11}}{{\sqrt{{17}}\sqrt{{13}}}}\;\Rightarrow\;\theta={vec.angle_between_deg():.2f}^\circ", "El ángulo sale del producto escalar."),
            (r"A_xB_y-A_yB_x=4\cdot3-1\cdot2=10", "Área del paralelogramo."),
        ],
        numeric_method=(
            "El área se verifica con Monte Carlo (400 000 puntos en la caja envolvente) y la invarianza "
            "de A·B y A×B se comprueba girando el sistema de ejes 360° en pasos de 1°."
        ),
        rows=[
            Row("Producto escalar", r"\vec A\cdot\vec B", pa["dot_by_angle"], vec.dot(), "", 4),
            Row("Área del paralelogramo", r"|\vec A\times\vec B|", pa["cross_by_angle"], area_mc, "", 3),
        ],
        notes=[f"Variación máxima de A·B al rotar los ejes: {np.ptp(dots):.1e}."],
    )

    return Chapter(
        number=2, slug="ch02", module="ch02_vectors", title="Vectores",
        summary="Magnitudes escalares y vectoriales, suma geométrica y analítica, y los dos productos de vectores.",
        concepts=[
            "Escalares y vectores", "Suma geométrica de vectores",
            "Componentes de un vector", "Suma por componentes",
            "Producto escalar", "Producto vectorial",
        ],
        scenes=[s1, s2],
        part="Mecánica clásica",
    )


# ---------------------------------------------------------------- capitulo 4
def _ch4() -> Chapter:
    an = c4.projectile_analytic()
    t, s = c4.projectile_numeric(dt=0.005)
    best = c4.optimal_angle_numeric()
    ca = c4.circular_analytic()
    fd = c4.fd_error_table(hs=(0.01,))[0]

    s1 = Scene(
        id="Projectile",
        title="Movimiento de proyectiles",
        statement=(
            "Un proyectil se lanza desde el suelo con v₀ = 25 m/s a 40° sobre la horizontal. "
            "Determinar el tiempo de vuelo, la altura máxima, el alcance y el ángulo de alcance máximo."
        ),
        foundations=[
            (r"x=v_0\cos\theta\,t,\qquad y=v_0\sin\theta\,t-\tfrac12gt^{2}", "Movimientos horizontal y vertical independientes."),
            (r"R=\dfrac{v_0^{2}\sin2\theta}{g}", "Alcance en terreno plano."),
        ],
        analytic=[
            (rf"T=\dfrac{{2v_0\sin\theta}}{{g}}={an['t_flight']:.3f}\ \mathrm{{s}},\quad H=\dfrac{{(v_0\sin\theta)^2}}{{2g}}={an['height']:.3f}\ \mathrm{{m}}", "Tiempo de vuelo y altura máxima."),
            (rf"R={an['range']:.3f}\ \mathrm{{m}},\qquad \dfrac{{d}}{{d\theta}}\sin2\theta=0\Rightarrow\theta=45^\circ", "Alcance y ángulo óptimo."),
        ],
        numeric_method=(
            "Integración RK4 del estado (x, y, vₓ, v_y) con paso 0.005 s; el impacto se refina por "
            "interpolación lineal. El ángulo óptimo se busca con sección áurea sobre el alcance numérico."
        ),
        rows=[
            Row("Tiempo de vuelo", "T", an["t_flight"], float(t[-1]), "s", 4),
            Row("Altura máxima", "H", an["height"], float(s[:, 1].max()), "m", 4),
            Row("Alcance", "R", an["range"], float(s[-1, 0]), "m", 4),
            Row("Ángulo de alcance máximo", r"\theta^*", 45.0, best, "°", 3),
        ],
    )

    s2 = Scene(
        id="CircularMotion",
        title="Movimiento circular uniforme",
        statement="Una partícula recorre una circunferencia de radio R = 2 m con periodo T = 4 s. Determinar su rapidez y su aceleración centrípeta.",
        foundations=[
            (r"\vec r(t)=R(\cos\omega t,\sin\omega t),\qquad \omega=\dfrac{2\pi}{T}", "Posición en movimiento circular uniforme."),
            (r"v=\omega R,\qquad a_c=\dfrac{v^2}{R}=\omega^2R", "Rapidez y aceleración centrípeta."),
        ],
        analytic=[
            (rf"\omega={ca['omega']:.4f}\ \mathrm{{rad/s}},\qquad v={ca['speed']:.4f}\ \mathrm{{m/s}}", "Velocidad angular y rapidez."),
            (rf"a_c=\omega^2R={ca['a_c']:.4f}\ \mathrm{{m/s^2}}", "Aceleración dirigida al centro."),
        ],
        numeric_method=(
            "A partir solo de la posición r(t) se estiman v y a con diferencias finitas centradas "
            "(h = 0.01 s); el error decrece como h², como se observa al reducir el paso."
        ),
        rows=[
            Row("Rapidez", "v", ca["speed"], fd["speed"], "m/s", 4),
            Row("Aceleración centrípeta", "a_c", ca["a_c"], fd["a_c"], "m/s²", 4),
        ],
    )

    return Chapter(
        number=4, slug="ch04", module="ch04_motion_2d", title="Movimiento en dos y tres dimensiones",
        summary="Vectores posición, velocidad y aceleración; proyectiles y movimiento circular uniforme.",
        concepts=[
            "Vectores posición, velocidad y aceleración", "Movimiento con aceleración constante en el plano",
            "Movimiento de proyectiles", "Movimiento circular uniforme",
            "Velocidad y aceleración relativas",
        ],
        scenes=[s1, s2],
        part="Mecánica clásica",
    )


# ---------------------------------------------------------------- capitulo 5
def _ch5() -> Chapter:
    t_an, t_nu = c5.tensions_analytic(), c5.tensions_numeric()
    a_an, T_an = c5.pulley_analytic()
    a_nu, T_nu = c5.pulley_numeric()
    _, ys = c5.pulley_trajectory(t1=2.2, n=220)
    th_an, th_nu = c5.balance_angle_analytic(), c5.balance_angle_numeric()

    s1 = Scene(
        id="StaticEquilibrium",
        title="Equilibrio estático con dos cables",
        statement=(
            "Un letrero de 500 N cuelga en reposo de dos cables que forman 30° y 45° con la horizontal. "
            "Determinar la tensión de cada cable."
        ),
        foundations=[
            (r"\sum\vec F=m\vec a", "Segunda ley de Newton."),
            (r"\vec a=0\;\Longleftrightarrow\;\sum F_x=0,\;\sum F_y=0", "Condición de equilibrio de la partícula."),
        ],
        analytic=[
            (r"T_1\cos\alpha=T_2\cos\beta,\qquad T_1\sin\alpha+T_2\sin\beta=W", "Componentes horizontal y vertical."),
            (rf"T_1=\dfrac{{W\cos\beta}}{{\sin(\alpha+\beta)}}={t_an[0]:.2f}\ \mathrm{{N}},\quad T_2=\dfrac{{W\cos\alpha}}{{\sin(\alpha+\beta)}}={t_an[1]:.2f}\ \mathrm{{N}}", "Solución cerrada del sistema."),
        ],
        numeric_method=(
            "El sistema 2×2 se resuelve con numpy.linalg.solve y se verifica que la fuerza neta "
            "residual sea del orden del redondeo. Luego se barre α entre 0° y 90° con β fijo."
        ),
        rows=[
            Row("Tensión del cable izquierdo", "T_1", t_an[0], t_nu[0], "N", 4),
            Row("Tensión del cable derecho", "T_2", t_an[1], t_nu[1], "N", 4),
        ],
        notes=[f"Fuerza neta residual: {np.linalg.norm(c5.net_force(*t_nu)):.1e} N."],
    )

    s2 = Scene(
        id="InclinePulley",
        title="Plano inclinado con polea ideal",
        statement=(
            "Un bloque de 4 kg sobre un plano sin fricción inclinado 30° está unido por una cuerda, a "
            "través de una polea ideal, a un bloque colgante de 3 kg. Determinar la aceleración y la tensión."
        ),
        foundations=[
            (r"\sum\vec F=m\vec a", "Segunda ley aplicada a cada cuerpo."),
            (r"a_1=a_2=a,\qquad T_1=T_2=T", "Cuerda inextensible y polea ideal."),
        ],
        analytic=[
            (r"m_1:\;T-m_1g\sin\theta=m_1a,\qquad m_2:\;m_2g-T=m_2a", "Una ecuación por cuerpo."),
            (rf"a=\dfrac{{m_2-m_1\sin\theta}}{{m_1+m_2}}\,g={a_an:.4f}\ \mathrm{{m/s^2}},\qquad T={T_an:.3f}\ \mathrm{{N}}", "Solución cerrada."),
            (rf"a=0\;\Rightarrow\;\sin\theta=\dfrac{{m_2}}{{m_1}}\;\Rightarrow\;\theta={th_an:.4f}^\circ", "Ángulo de equilibrio."),
        ],
        numeric_method=(
            "El sistema lineal en (a, T) se resuelve con numpy.linalg.solve; el movimiento se integra con "
            "RK4 (220 pasos hasta 2.2 s) y el ángulo de equilibrio se halla por bisección sobre a(θ)."
        ),
        rows=[
            Row("Aceleración", "a", a_an, a_nu, "m/s²", 5),
            Row("Tensión", "T", T_an, T_nu, "N", 4),
            Row("Desplazamiento a los 2.2 s", "s", 0.5 * a_an * 2.2**2, float(ys[-1, 0]), "m", 4),
            Row("Ángulo de equilibrio", r"\theta_{eq}", th_an, th_nu, "°", 4),
        ],
    )

    return Chapter(
        number=5, slug="ch05", module="ch05_newton", title="Fuerza y leyes de Newton",
        summary="Las tres leyes de Newton, el diagrama de cuerpo libre y su aplicación a equilibrio y sistemas acoplados.",
        concepts=[
            "Mecánica clásica", "Primera ley de Newton", "Fuerza y masa",
            "Segunda ley de Newton", "Tercera ley de Newton", "Peso y masa",
            "Aplicaciones de las leyes de Newton",
        ],
        scenes=[s1, s2],
        part="Mecánica clásica",
    )


def _core() -> list[Chapter]:
    return [_ch1(), _ch2v(), _ch3m(), _ch4(), _ch5()]


def load_chapters() -> list[Chapter]:
    """Capitulos 1-5 mas todos los modulos `blk_*` del paquete, ordenados por (volumen, numero)."""
    import importlib
    import pkgutil

    import content

    chapters = _core()
    for mod in pkgutil.iter_modules(content.__path__):
        if mod.name.startswith("blk_"):
            chapters += importlib.import_module(f"content.{mod.name}").chapters()
    chapters.sort(key=lambda c: (c.volume, c.number))
    return chapters
