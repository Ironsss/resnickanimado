"""Explicacion y pasos a mano: Volumen 1, capitulos 1 a 4 (bloque A).

Escenas: DimensionalAnalysis, ErrorPropagation, VectorAddition, VectorProducts,
InstantaneousVelocity, ConstantAcceleration, Projectile, CircularMotion.
Todas las cifras se calculan con `physics/` o con aritmetica explicita.
"""
from __future__ import annotations

import numpy as np

from content.texfmt import fx, sci
from physics import ch01_measurement as c1
from physics import ch02_vectors as vec
from physics import ch03_motion_1d as m1d
from physics import ch04_motion_2d as c4
from physics.integrators import euler_step, integrate, rk4_step, trapezoid


def _v(a, d=4) -> str:
    """Vector fila TeX con d decimales."""
    return r"\left(" + ",\\ ".join(f"{x:.{d}f}" for x in a) + r"\right)"


# ====================================================================== cap. 1
def _dimensional() -> dict:
    g, L, th0, n = 9.81, 1.0, 0.01, 4000
    res = c1.pendulum_dimensionless_constant(th0)
    ex = res.exponents
    A = np.column_stack([c1.DIM["L"], c1.DIM["g"], c1.DIM["m"]])
    w2 = g / L
    tmax = 2.0 * np.pi * np.sqrt(L / g)
    h = tmax / n

    def f(y):
        return np.array([y[1], -w2 * np.sin(y[0])])

    y0 = np.array([th0, 0.0])
    k1 = f(y0)
    k2 = f(y0 + h / 2 * k1)
    k3 = f(y0 + h / 2 * k2)
    k4 = f(y0 + h * k3)
    y1 = y0 + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    # reproducir la integracion completa para ubicar el cruce por cero
    t, y = integrate(lambda _t, s: f(s), y0, 0.0, tmax, n, rk4_step)
    idx = int(np.argmax(y[:, 0] <= 0.0))
    ta, tb, tha, thb = t[idx - 1], t[idx], y[idx - 1, 0], y[idx, 0]
    tc = ta + (0.0 - tha) * (tb - ta) / (thb - tha)
    period = 4.0 * tc
    k_num = res.k_numeric
    t_small = 2 * np.pi * np.sqrt(L / g)
    corr = 1 + th0**2 / 16

    am = r"\begin{pmatrix}0&0&1\\1&1&0\\0&-2&0\end{pmatrix}"
    return dict(
        explanation=[
            "El análisis dimensional parte de una idea simple: una ecuación física solo tiene sentido si sus dos miembros tienen las mismas dimensiones. No se puede igualar una longitud con un tiempo. Esta exigencia, sin resolver ninguna ecuación de movimiento, restringe mucho la forma en que una magnitud puede depender de otras.",
            "Para el péndulo simple se supone que el periodo $T$ depende de la longitud $L$, de la aceleración de la gravedad $g$ y de la masa $m$, combinadas como un producto de potencias. Cada dimensión base (masa, longitud, tiempo) debe quedar con el mismo exponente a ambos lados, lo que produce un sistema lineal para los exponentes desconocidos.",
            "El resultado es que la masa no interviene y que $T\\propto\\sqrt{L/g}$. Esto tiene consecuencias físicas verificables: cuadruplicar la longitud duplica el periodo, y el mismo péndulo oscila más despacio en la Luna, donde $g$ es menor.",
            "El método tiene un límite claro: las constantes adimensionales (números como $2\\pi$ o $1/2$) son invisibles para él, porque no tienen dimensiones. Aquí queda una constante $k$ que solo se fija resolviendo la dinámica o midiendo. Además supone que no hay otras magnitudes relevantes; si se olvida una (por ejemplo el ángulo de amplitud), el análisis es incompleto.",
            "Por eso se mide $k$ integrando numéricamente la ecuación real del péndulo con una amplitud muy pequeña, donde $\\sin\\theta\\approx\\theta$ y el periodo no depende de la amplitud. El valor obtenido debe acercarse a $2\\pi\\approx 6.283$.",
        ],
        analytic_steps=[
            ("Proponer una ley de potencias", "Se supone que el periodo es un producto de potencias de las magnitudes de las que puede depender el sistema; $k$ es un número sin dimensiones que aún no conocemos.",
             [r"T = k\,L^{a}\,g^{b}\,m^{c}"]),
            ("Escribir la ecuación dimensional", "Se reemplaza cada magnitud por su dimensión: $[T]=\\mathsf{T}$, $[L]=\\mathsf{L}$, $[g]=\\mathsf{L}\\,\\mathsf{T}^{-2}$, $[m]=\\mathsf{M}$. Al agrupar potencias iguales queda un solo monomio.",
             [r"\mathsf{T} = \mathsf{L}^{a}\,\left(\mathsf{L}\,\mathsf{T}^{-2}\right)^{b}\,\mathsf{M}^{c} = \mathsf{M}^{c}\,\mathsf{L}^{a+b}\,\mathsf{T}^{-2b}"]),
            ("Igualar los exponentes de cada dimensión", "El miembro izquierdo es $\\mathsf{M}^0\\mathsf{L}^0\\mathsf{T}^1$. Igualando exponente por exponente se obtiene un sistema lineal de tres ecuaciones con tres incógnitas.",
             [r"\mathsf{M}:\ c = 0,\qquad \mathsf{L}:\ a + b = 0,\qquad \mathsf{T}:\ -2b = 1"]),
            ("Resolver el sistema", "La ecuación del tiempo da $b$ directamente; con $b$ se obtiene $a$ de la ecuación de longitud, y $c$ ya estaba dado.",
             [r"b = -\dfrac{1}{2},\qquad a = -b = \dfrac{1}{2},\qquad c = 0",
              rf"a = {ex['L']:.4f},\quad b = {ex['g']:.4f},\quad c = {ex['m']:.4f}"]),
            ("Escribir la forma del periodo", "Con esos exponentes la masa desaparece y el periodo queda proporcional a la raíz de $L/g$.",
             [r"T = k\,L^{1/2}\,g^{-1/2} = k\sqrt{\dfrac{L}{g}}"]),
            ("Fijar la constante con la ecuación linealizada", "El análisis dimensional no da $k$. Para ángulos pequeños, $\\ddot\\theta = -(g/L)\\theta$ es un oscilador armónico de frecuencia angular $\\omega=\\sqrt{g/L}$, cuyo periodo es $2\\pi/\\omega$. Por comparación $k=2\\pi$.",
             [r"T = \dfrac{2\pi}{\omega} = 2\pi\sqrt{\dfrac{L}{g}}\ \Rightarrow\ k = 2\pi = " + fx(2 * np.pi, 5)]),
            ("Evaluar para L = 1 m", "Se sustituye $L=1$ m y $g=9.81$ m/s$^2$ para tener un valor que comparar con la simulación.",
             [rf"T = 2\pi\sqrt{{\dfrac{{{L:.0f}\ \mathrm{{m}}}}{{{g}\ \mathrm{{m/s^2}}}}}} = 2\pi\,({fx(np.sqrt(L / g), 5)}\ \mathrm{{s}}) = {fx(t_small, 5)}\ \mathrm{{s}}"]),
        ],
        numeric_steps=[
            ("Plantear el sistema lineal en forma matricial", "Las filas de $A$ son las dimensiones $\\mathsf{M},\\mathsf{L},\\mathsf{T}$ y las columnas las magnitudes $L, g, m$. Se resuelve $A\\vec x=\\vec d$ con $\\vec d=(0,0,1)$, el vector de dimensiones de $T$.",
             [r"A = " + am + r",\qquad \vec x = \begin{pmatrix}a\\b\\c\end{pmatrix},\qquad \vec d = \begin{pmatrix}0\\0\\1\end{pmatrix}"]),
            ("Resolver por mínimos cuadrados (SVD)", "El código usa numpy.linalg.lstsq, que descompone $A$ por valores singulares y minimiza $\\|A\\vec x-\\vec d\\|$. Como $A$ es cuadrada e invertible, el mínimo es exacto. A mano equivale a sustituir hacia atrás: fila 3 da $b$, fila 2 da $a$, fila 1 da $c$.",
             [rf"\vec x = ({ex['L']:.6f},\ {ex['g']:.6f},\ {ex['m']:.6f}),\qquad \|A\vec x-\vec d\| = {sci(res.residual, 2)}"]),
            ("Preparar la simulación del péndulo", "Para medir $k$ se integra $\\dot\\theta=\\omega_a$, $\\dot\\omega_a=-(g/L)\\sin\\theta$ con RK4. Se parte de la amplitud $\\theta_0=0.01$ rad en reposo y se toman 4000 pasos hasta una cota $t_{\\max}$ mayor que el cuarto de periodo.",
             [rf"\theta_0 = {th0}\ \mathrm{{rad}},\quad \omega_a(0)=0,\quad g/L = {w2}\ \mathrm{{s^{{-2}}}}",
              rf"t_{{\max}} = 2\pi\sqrt{{L/g}} = {fx(tmax, 5)}\ \mathrm{{s}},\qquad h = \dfrac{{t_{{\max}}}}{{4000}} = {sci(h, 5)}\ \mathrm{{s}}"]),
            ("Primer paso de RK4 a mano", "Se evalúa la derivada en cuatro puntos del intervalo y se combinan con pesos $1,2,2,1$. Cada $k$ es un par (derivada de $\\theta$, derivada de $\\omega_a$).",
             [rf"k_1 = ({sci(k1[0], 4)},\ {sci(k1[1], 5)}),\qquad k_2 = ({sci(k2[0], 5)},\ {sci(k2[1], 5)})",
              rf"k_3 = ({sci(k3[0], 5)},\ {sci(k3[1], 5)}),\qquad k_4 = ({sci(k4[0], 5)},\ {sci(k4[1], 5)})",
              rf"y_1 = y_0 + \dfrac{{h}}{{6}}(k_1+2k_2+2k_3+k_4) = ({sci(y1[0], 8)},\ {sci(y1[1], 5)})"]),
            ("Avanzar hasta el cruce por cero", "Se repite el paso 4000 veces como máximo y se busca el primer índice donde $\\theta\\le 0$. Ese cruce ocurre al cuarto de periodo, porque el péndulo parte del extremo.",
             [rf"\text{{índice del cruce}}={idx},\quad (t_a,\theta_a) = ({fx(ta, 5)},\ {sci(tha, 4)}),\quad (t_b,\theta_b) = ({fx(tb, 5)},\ {sci(thb, 4)})"]),
            ("Interpolar el instante del cruce", "Entre dos muestras se aproxima $\\theta(t)$ por una recta y se halla donde vale cero; después se multiplica por 4 para pasar de cuarto de periodo a periodo.",
             [rf"t_c = t_a + \dfrac{{(0-\theta_a)(t_b-t_a)}}{{\theta_b-\theta_a}} = {fx(tc, 6)}\ \mathrm{{s}},\qquad T_{{num}} = 4t_c = {fx(period, 6)}\ \mathrm{{s}}"]),
            ("Obtener k y compararla con 2π", "Con los exponentes ya hallados, $k = T/(L^{a}g^{b})$. La pequeña diferencia con $2\\pi$ es la corrección de amplitud finita, $T\\approx T_0(1+\\theta_0^2/16)$.",
             [rf"k_{{num}} = \dfrac{{T_{{num}}}}{{L^{{{ex['L']:.1f}}}\,g^{{{ex['g']:.1f}}}}} = {fx(k_num, 6)},\qquad 2\pi = {fx(2*np.pi, 6)}",
              rf"\dfrac{{k_{{num}}-2\pi}}{{2\pi}} = {sci((k_num - 2*np.pi)/(2*np.pi), 3)} \approx \dfrac{{\theta_0^2}}{{16}} = {sci(th0**2/16, 3)}"]),
        ],
    )


def _error_prop() -> dict:
    m, r, hh = c1.CYLINDER["m"], c1.CYLINDER["r"], c1.CYLINDER["h"]
    an = c1.density_analytic()
    mc = c1.density_monte_carlo()
    n = 400_000
    rng = np.random.default_rng(7)
    ms = rng.normal(m.value, m.sigma, n)
    rs = rng.normal(r.value, r.sigma, n)
    hs = rng.normal(hh.value, hh.sigma, n)
    rho_s = c1.density(ms, rs, hs)
    rho = an["rho"]
    vol = np.pi * r.value**2 * hh.value
    rel = np.sqrt(an["rel_m"]**2 + an["rel_r"]**2 + an["rel_h"]**2)
    first = [(ms[i], rs[i], hs[i], rho_s[i]) for i in range(3)]
    tab = r"\begin{array}{c|ccc|c} i & m\ (\mathrm{g}) & r\ (\mathrm{cm}) & h\ (\mathrm{cm}) & \rho_i\ (\mathrm{g/cm^3})\\ \hline "
    for i, (a, b, c, d) in enumerate(first, 1):
        tab += rf"{i} & {a:.4f} & {b:.5f} & {c:.5f} & {d:.5f}\\ "
    tab += r"\end{array}"
    ten = rho_s[:10]
    return dict(
        explanation=[
            "Toda medición viene con una incertidumbre: el valor verdadero está cerca del medido, pero no sabemos exactamente dónde. Cuando se calcula una magnitud a partir de varias mediciones (aquí la densidad a partir de masa, radio y altura), esas incertidumbres se propagan y el resultado también tiene una incertidumbre que hay que cuantificar.",
            "Si la magnitud es un producto de potencias, $\\rho = m\\,r^{-2}h^{-1}/\\pi$, lo que se suma son las incertidumbres relativas, cada una multiplicada por el valor absoluto de su exponente. Por eso el radio, que aparece al cuadrado, pesa el doble que su incertidumbre relativa. Si los errores son independientes se combinan en cuadratura (raíz de la suma de cuadrados), porque es poco probable que todos coincidan en el peor sentido a la vez.",
            "Sumar los errores linealmente da una cota conservadora, válida cuando todo conspira en la misma dirección. La combinación en cuadratura da el error típico (una desviación estándar), siempre menor o igual que la suma lineal.",
            "La fórmula de primer orden es una aproximación: reemplaza la función por su plano tangente, lo cual es válido si las incertidumbres relativas son pequeñas (aquí del orden del 1 %). Un método más general es el Monte Carlo, que no linealiza: se generan muchas «mediciones posibles» y se calcula la densidad de cada una.",
            "El orden de magnitud es razonable: una densidad cercana a $2.7\\ \\mathrm{g/cm^3}$ corresponde al aluminio, y una incertidumbre relativa de aproximadamente 1 % es compatible con medir longitudes con un error de décimas de milímetro.",
        ],
        analytic_steps=[
            ("Plantear la densidad", "La densidad es masa entre volumen, y el volumen de un cilindro es el área de la base por la altura.",
             [r"\rho = \dfrac{m}{V} = \dfrac{m}{\pi r^2 h}"]),
            ("Sustituir los valores medios", "Se usan los valores centrales de las tres mediciones, con $r$ y $h$ en cm para que $\\rho$ salga en g/cm$^3$. Primero el volumen y luego el cociente.",
             [rf"V = \pi\,({r.value})^2\,({hh.value}) = {fx(vol, 4)}\ \mathrm{{cm^3}}",
              rf"\rho = \dfrac{{{m.value}\ \mathrm{{g}}}}{{{fx(vol, 4)}\ \mathrm{{cm^3}}}} = {fx(rho, 4)}\ \mathrm{{g/cm^3}}"]),
            ("Plantear la propagación en cuadratura", "Como $\\ln\\rho = \\ln m - 2\\ln r - \\ln h - \\ln\\pi$, al derivar y tomar cuadrados de términos independientes queda la suma de incertidumbres relativas, con el factor 2 del radio.",
             [r"\left(\dfrac{\sigma_\rho}{\rho}\right)^2 = \left(\dfrac{\sigma_m}{m}\right)^2 + \left(2\dfrac{\sigma_r}{r}\right)^2 + \left(\dfrac{\sigma_h}{h}\right)^2"]),
            ("Calcular cada contribución relativa", "Se divide cada incertidumbre por su valor central; el del radio se duplica.",
             [rf"\dfrac{{\sigma_m}}{{m}} = \dfrac{{{m.sigma}}}{{{m.value}}} = {an['rel_m']*100:.3f}\%,\qquad 2\dfrac{{\sigma_r}}{{r}} = \dfrac{{2({r.sigma})}}{{{r.value}}} = {an['rel_r']*100:.3f}\%,\qquad \dfrac{{\sigma_h}}{{h}} = \dfrac{{{hh.sigma}}}{{{hh.value}}} = {an['rel_h']*100:.3f}\%"]),
            ("Combinar en cuadratura", "Se suman los cuadrados de las tres contribuciones y se saca la raíz. El radio domina: su cuadrado es la mayor parte de la suma.",
             [rf"\dfrac{{\sigma_\rho}}{{\rho}} = \sqrt{{({an['rel_m']:.5f})^2+({an['rel_r']:.5f})^2+({an['rel_h']:.5f})^2}} = {rel:.5f} = {rel*100:.3f}\%"]),
            ("Convertir a incertidumbre absoluta", "Se multiplica la incertidumbre relativa por la densidad calculada.",
             [rf"\sigma_\rho = \rho\cdot{rel:.5f} = ({fx(rho, 4)})({rel:.5f}) = {fx(an['sigma_quadrature'], 4)}\ \mathrm{{g/cm^3}}"]),
            ("Comparar con la cota lineal y escribir el resultado", "La suma lineal de las tres contribuciones da una cota máxima, mayor que la de cuadratura. El resultado se reporta con la incertidumbre estándar.",
             [rf"\sigma_{{\rho,\max}} = \rho\,({an['rel_m']:.5f}+{an['rel_r']:.5f}+{an['rel_h']:.5f}) = {fx(an['sigma_worst_case'], 4)}\ \mathrm{{g/cm^3}}",
              rf"\rho = {fx(rho, 3)} \pm {fx(an['sigma_quadrature'], 3)}\ \mathrm{{g/cm^3}}"]),
        ],
        numeric_steps=[
            ("Idea del método Monte Carlo", "En lugar de linealizar, cada medición se trata como una variable aleatoria normal centrada en su valor medido y con desviación igual a su incertidumbre. Se sortean muchas ternas posibles y se calcula $\\rho$ para cada una.",
             [rf"m\sim\mathcal N({m.value},\,{m.sigma}^2),\quad r\sim\mathcal N({r.value},\,{r.sigma}^2),\quad h\sim\mathcal N({hh.value},\,{hh.sigma}^2)"]),
            ("Parámetros", "Se usan $N=400\\,000$ realizaciones y una semilla fija (7) para que el resultado sea reproducible. Primero se sortean las $N$ masas, luego los $N$ radios y luego las $N$ alturas.",
             [r"N = 400\,000,\qquad \text{semilla}=7,\qquad \rho_i = \dfrac{m_i}{\pi\,r_i^2\,h_i}"]),
            ("Primeras tres muestras calculadas a mano", "Con cada terna sorteada se calcula una densidad. Las tres primeras filas del generador son:",
             [tab,
              rf"\rho_1 = \dfrac{{{first[0][0]:.4f}}}{{\pi({first[0][1]:.5f})^2({first[0][2]:.5f})}} = {first[0][3]:.5f}\ \mathrm{{g/cm^3}}"]),
            ("Promedio y desviación con 10 muestras", "Con solo diez realizaciones ya se ve la mecánica: el promedio estima $\\rho$ y la desviación muestral (con $N-1$ en el denominador) estima $\\sigma_\\rho$, aunque todavía con mucho ruido.",
             [rf"\bar\rho_{{10}} = \dfrac{{1}}{{10}}\sum_{{i=1}}^{{10}}\rho_i = {ten.mean():.5f},\qquad s_{{10}} = \sqrt{{\dfrac{{\sum(\rho_i-\bar\rho)^2}}{{9}}}} = {ten.std(ddof=1):.5f}"]),
            ("Resultado con las 400 000 muestras", "Se promedian y se calcula la desviación estándar de todas las densidades. El error estadístico del promedio es $s/\\sqrt N$, mucho menor que la incertidumbre del método.",
             [rf"\bar\rho = \dfrac{{1}}{{N}}\sum\rho_i = {mc['mean']:.5f}\ \mathrm{{g/cm^3}},\qquad s = {mc['std']:.5f}\ \mathrm{{g/cm^3}}",
              rf"\dfrac{{s}}{{\sqrt N}} = {sci(mc['std']/np.sqrt(n), 3)}\ \mathrm{{g/cm^3}}"]),
            ("Comparar con el resultado analítico", "El promedio coincide con $\\rho$ salvo un pequeño sesgo, porque la densidad no es lineal en $r$ y $h$ (promediar $1/r^2$ no es lo mismo que evaluar en $\\bar r$). La desviación coincide con la fórmula de primer orden.",
             [rf"\bar\rho-\rho = {sci(mc['mean']-rho, 3)}\ \mathrm{{g/cm^3}},\qquad s-\sigma_\rho = {sci(mc['std']-an['sigma_quadrature'], 3)}\ \mathrm{{g/cm^3}}"]),
        ],
    )


# ====================================================================== cap. 2
def _vec_add() -> dict:
    ra, rn = vec.resultant_analytic(), vec.resultant_numeric()
    a, b = 40.0, 25.0
    t1, t2 = 30.0, 120.0
    d1, d2 = vec.D1, vec.D2
    gamma = t2 - t1
    c, s = np.cos(np.deg2rad(gamma)), np.sin(np.deg2rad(gamma))
    r2 = a**2 + b**2 + 2 * a * b * c
    delta = np.rad2deg(np.arcsin(b * s / ra["magnitude"]))
    return dict(
        explanation=[
            "Un vector es una magnitud con módulo y dirección; el desplazamiento es el ejemplo más claro. Dos desplazamientos sucesivos equivalen a uno solo, el resultante, que va del punto de partida al punto final sin importar el camino. Gráficamente se coloca la cola del segundo vector en la punta del primero y el resultante cierra el triángulo.",
            "El método geométrico (leyes de los cosenos y de los senos) es útil cuando se conoce el ángulo entre los vectores. El método por componentes es más general: cada vector se proyecta sobre los ejes $x$ e $y$, las proyecciones se suman por separado, y al final se reconstruye el módulo con Pitágoras y el ángulo con la arcotangente.",
            "La suma de vectores no es la suma de módulos: $40+25=65$ m sería la distancia recorrida, pero el desplazamiento depende del ángulo entre tramos. Aquí los tramos forman un ángulo de $90^\\circ$, así que el módulo resultante es la hipotenusa de un triángulo rectángulo de catetos 40 y 25.",
            "Al calcular el ángulo con $\\tan^{-1}(R_y/R_x)$ hay que cuidar el cuadrante; la función atan2 del código lo hace automáticamente. Aquí ambas componentes son positivas, así que el resultado está en el primer cuadrante y entre $30^\\circ$ y $120^\\circ$, como debe ser para un vector que combina dos vectores situados en ese intervalo.",
        ],
        analytic_steps=[
            ("Identificar los datos y el ángulo entre vectores", "El primer vector forma $30^\\circ$ con el eje $x$ y el segundo $120^\\circ$. El ángulo entre ellos, medido al colocarlos cola con punta, es la diferencia.",
             [rf"A = {a:.0f}\ \mathrm{{m}}\ (30^\circ),\qquad B = {b:.0f}\ \mathrm{{m}}\ (120^\circ),\qquad \gamma = 120^\circ-30^\circ = {gamma:.0f}^\circ"]),
            ("Aplicar la ley de los cosenos", "Para dos vectores con ángulo $\\gamma$ entre sus direcciones el cuadrado del resultante es $A^2+B^2+2AB\\cos\\gamma$ (signo $+$ porque $\\gamma$ es el ángulo entre las direcciones, no el interior del triángulo).",
             [r"R^2 = A^2 + B^2 + 2AB\cos\gamma"]),
            ("Sustituir los números", f"Como $\\cos 90^\\circ = 0$, el término cruzado se anula y queda el teorema de Pitágoras.",
             [rf"R^2 = ({a:.0f})^2 + ({b:.0f})^2 + 2({a:.0f})({b:.0f})\cos{gamma:.0f}^\circ = {a**2:.0f} + {b**2:.0f} + {2*a*b*c:.0f} = {r2:.0f}\ \mathrm{{m^2}}"]),
            ("Calcular el módulo", "Se extrae la raíz cuadrada del resultado anterior.",
             [rf"R = \sqrt{{{r2:.0f}}}\ \mathrm{{m}} = {ra['magnitude']:.4f}\ \mathrm{{m}}"]),
            ("Hallar la dirección con la ley de los senos", "En el triángulo, el ángulo $\\delta$ que el resultante forma con el primer vector cumple $\\sin\\delta/B = \\sin\\gamma/R$. El ángulo respecto al eje $x$ es $30^\\circ+\\delta$.",
             [r"\delta = \sin^{-1}\!\left(\dfrac{B\sin\gamma}{R}\right)",
              rf"\delta = \sin^{{-1}}\!\left(\dfrac{{({b:.0f})\sin{gamma:.0f}^\circ}}{{{ra['magnitude']:.4f}}}\right) = {delta:.4f}^\circ"]),
            ("Dar el ángulo respecto del eje x", "Se suma la dirección del primer vector.",
             [rf"\theta_R = 30^\circ + {delta:.4f}^\circ = {ra['angle_deg']:.4f}^\circ"]),
            ("Verificar por componentes", "Es un chequeo independiente: se proyecta cada vector, se suman las componentes y se reconstruyen módulo y ángulo.",
             [rf"R_x = {a:.0f}\cos30^\circ + {b:.0f}\cos120^\circ = {d1[0]:.4f} + ({d2[0]:.4f}) = {d1[0]+d2[0]:.4f}\ \mathrm{{m}}",
              rf"R_y = {a:.0f}\sin30^\circ + {b:.0f}\sin120^\circ = {d1[1]:.4f} + {d2[1]:.4f} = {d1[1]+d2[1]:.4f}\ \mathrm{{m}}"]),
        ],
        numeric_steps=[
            ("Representar cada vector con sus componentes", "El código convierte cada vector de (módulo, ángulo) a (x, y) con $A_x=A\\cos\\theta$ y $A_y=A\\sin\\theta$ (los ángulos pasan a radianes: $\\theta_{rad}=\\theta^\\circ\\pi/180$).",
             [rf"30^\circ = {np.deg2rad(30):.6f}\ \mathrm{{rad}},\qquad 120^\circ = {np.deg2rad(120):.6f}\ \mathrm{{rad}}"]),
            ("Proyectar el primer vector", "Se evalúan coseno y seno de 30° y se multiplican por 40 m.",
             [rf"\vec D_1 = 40\,(\cos30^\circ,\ \sin30^\circ) = 40\,({np.cos(np.deg2rad(30)):.6f},\ {np.sin(np.deg2rad(30)):.6f}) = {_v(d1)}\ \mathrm{{m}}"]),
            ("Proyectar el segundo vector", "Igual con 25 m a 120°; la componente $x$ es negativa porque el ángulo supera $90^\\circ$.",
             [rf"\vec D_2 = 25\,(\cos120^\circ,\ \sin120^\circ) = 25\,({np.cos(np.deg2rad(120)):.6f},\ {np.sin(np.deg2rad(120)):.6f}) = {_v(d2)}\ \mathrm{{m}}"]),
            ("Sumar componente a componente", "Esta es la operación central: numpy.sum a lo largo del eje de los vectores.",
             [rf"\vec R = \vec D_1+\vec D_2 = ({d1[0]:.4f}+({d2[0]:.4f}),\ {d1[1]:.4f}+{d2[1]:.4f}) = {_v([rn['x'], rn['y']])}\ \mathrm{{m}}"]),
            ("Recuperar el módulo con hipot", "Se aplica Pitágoras a las componentes; hypot lo calcula evitando desbordamientos.",
             [rf"R = \sqrt{{R_x^2+R_y^2}} = \sqrt{{({rn['x']:.4f})^2+({rn['y']:.4f})^2}} = {rn['magnitude']:.4f}\ \mathrm{{m}}"]),
            ("Recuperar el ángulo con atan2", "atan2$(R_y,R_x)$ devuelve el ángulo en el cuadrante correcto, en radianes, y se convierte a grados.",
             [rf"\theta_R = \mathrm{{atan2}}({rn['y']:.4f},\,{rn['x']:.4f}) = {np.arctan2(rn['y'], rn['x']):.6f}\ \mathrm{{rad}} = {rn['angle_deg']:.4f}^\circ"]),
            ("Comparar con el resultado analítico", "Ambos caminos coinciden hasta el redondeo de la aritmética de punto flotante.",
             [rf"R_{{num}}-R_{{an}} = {sci(rn['magnitude']-ra['magnitude'], 2)}\ \mathrm{{m}},\qquad \theta_{{num}}-\theta_{{an}} = {sci(rn['angle_deg']-ra['angle_deg'], 2)}^\circ"]),
        ],
    )


def _vec_prod() -> dict:
    A, B = vec.A, vec.B
    pa = vec.products_analytic()
    area_mc, pts, inside = vec.parallelogram_area_monte_carlo()
    n = len(pts)
    ma, mb = np.linalg.norm(A), np.linalg.norm(B)
    th = vec.angle_between_deg()
    dot, cr = vec.dot(), vec.cross_z()
    corners = np.array([[0, 0], A, B, A + B], dtype=float)
    lo, hi = corners.min(axis=0), corners.max(axis=0)
    box = float(np.prod(hi - lo))
    M = np.column_stack([A, B])
    st = np.linalg.solve(M, pts.T).T
    phi = np.deg2rad(30.0)
    ra, rb = vec.rotate_frame(A, phi), vec.rotate_frame(B, phi)
    _, _, _, dots = vec.invariance_sweep()
    tab = r"\begin{array}{c|cc|cc|c} i & x & y & s & t & \text{dentro}\\ \hline "
    for i in range(5):
        tab += rf"{i+1} & {pts[i,0]:.4f} & {pts[i,1]:.4f} & {st[i,0]:.4f} & {st[i,1]:.4f} & {'sí' if inside[i] else 'no'}\\ "
    tab += r"\end{array}"
    ctab = ""
    return dict(
        explanation=[
            "Hay dos formas de multiplicar vectores y miden cosas distintas. El producto escalar $\\vec A\\cdot\\vec B$ da un número: el módulo de uno por la proyección del otro sobre él. Es máximo cuando los vectores son paralelos, cero cuando son perpendiculares y negativo si forman más de $90^\\circ$.",
            "El producto vectorial $\\vec A\\times\\vec B$ da otro vector, perpendicular a ambos, cuyo módulo es $AB\\sin\\theta$. Ese módulo es el área del paralelogramo que forman los dos vectores; es máximo cuando son perpendiculares y cero cuando son paralelos. En el plano $xy$ solo tiene componente $z$.",
            "Ambos productos tienen una expresión muy sencilla en componentes, sin necesidad de conocer el ángulo: $A_xB_x+A_yB_y$ para el escalar y $A_xB_y-A_yB_x$ para la componente $z$ del vectorial. De la primera se despeja el ángulo, $\\cos\\theta=\\vec A\\cdot\\vec B/(AB)$.",
            "Una propiedad fundamental es que ambos son invariantes: si se giran los ejes coordenados, las componentes cambian pero el valor del producto escalar y el módulo del vectorial no. Eso es lo que permite decir que expresan relaciones geométricas reales y no artefactos del sistema de referencia.",
        ],
        analytic_steps=[
            ("Calcular los módulos", "Se necesitan para convertir el producto escalar en un ángulo.",
             [rf"A = \sqrt{{4^2+1^2}} = \sqrt{{17}} = {ma:.5f},\qquad B = \sqrt{{2^2+3^2}} = \sqrt{{13}} = {mb:.5f}"]),
            ("Producto escalar por componentes", "Se multiplican componentes homólogas y se suman.",
             [rf"\vec A\cdot\vec B = A_xB_x + A_yB_y = (4)(2)+(1)(3) = {dot:.0f}"]),
            ("Despejar el ángulo", "Se igualan las dos expresiones del producto escalar, $AB\\cos\\theta=\\vec A\\cdot\\vec B$, y se aplica el arcocoseno.",
             [rf"\cos\theta = \dfrac{{{dot:.0f}}}{{\sqrt{{17}}\sqrt{{13}}}} = \dfrac{{{dot:.0f}}}{{{ma*mb:.5f}}} = {dot/(ma*mb):.6f}",
              rf"\theta = \cos^{{-1}}({dot/(ma*mb):.6f}) = {th:.4f}^\circ"]),
            ("Producto vectorial en el plano", "Para vectores en el plano $xy$ solo sobrevive la componente $z$ del producto cruz, dada por el determinante $2\\times2$.",
             [rf"(\vec A\times\vec B)_z = A_xB_y - A_yB_x = (4)(3)-(1)(2) = {cr:.0f}"]),
            ("Interpretar como área", "El módulo del producto vectorial es el área del paralelogramo de lados $\\vec A$ y $\\vec B$.",
             [rf"\text{{Área}} = |\vec A\times\vec B| = {abs(cr):.0f}"]),
            ("Verificar con el ángulo", "Se recupera el mismo valor con $AB\\sin\\theta$, y el producto escalar con $AB\\cos\\theta$. Es una comprobación de consistencia entre las dos fórmulas.",
             [rf"AB\sin\theta = ({ma*mb:.5f})\sin{th:.4f}^\circ = {pa['cross_by_angle']:.4f}",
              rf"AB\cos\theta = ({ma*mb:.5f})\cos{th:.4f}^\circ = {pa['dot_by_angle']:.4f}"]),
            ("Comprobar el triángulo", "Como el triángulo formado por $\\vec A$ y $\\vec B$ tiene la mitad del área, su área es 5, lo que concuerda con lo esperado para estos vértices.",
             [rf"\text{{Área del triángulo}} = \dfrac{{1}}{{2}}|\vec A\times\vec B| = {abs(cr)/2:.1f}"]),
        ],
        numeric_steps=[
            ("Idea: área por conteo de puntos", "Se encierra el paralelogramo en el rectángulo más pequeño con lados paralelos a los ejes, se lanzan puntos aleatorios uniformes y la fracción que cae dentro estima la fracción de área.",
             [r"\text{Área} \approx \text{Área}_{caja}\cdot\dfrac{N_{dentro}}{N}"]),
            ("Caja envolvente", "Los vértices del paralelogramo son $(0,0)$, $\\vec A$, $\\vec B$ y $\\vec A+\\vec B$; la caja se obtiene de sus mínimos y máximos.",
             [rf"\text{{vértices: }}(0,0),\ (4,1),\ (2,3),\ (6,4)\ \Rightarrow\ x\in[{lo[0]:.0f},{hi[0]:.0f}],\ y\in[{lo[1]:.0f},{hi[1]:.0f}],\ \text{{Área}}_{{caja}}={box:.0f}"]),
            ("Prueba de pertenencia", "Un punto $\\vec p$ está dentro si se puede escribir $\\vec p=s\\vec A+t\\vec B$ con $0\\le s,t\\le1$. Se resuelve el sistema $2\\times2$ para cada punto.",
             [r"\begin{pmatrix}4&2\\1&3\end{pmatrix}\begin{pmatrix}s\\t\end{pmatrix}=\begin{pmatrix}x\\y\end{pmatrix}\ \Rightarrow\ s=\dfrac{3x-2y}{10},\quad t=\dfrac{4y-x}{10}"]),
            ("Primeras cinco muestras", "Se sortean puntos con la semilla 11 y se aplica la prueba. Por ejemplo, el primero tiene $s=(3x-2y)/10$ y $t=(4y-x)/10$ como en la tabla; está dentro solo si ambos están en $[0,1]$.",
             [tab]),
            ("Conteo con N = 400 000", "El código repite la prueba para todos los puntos y cuenta los que están dentro. La fracción por el área de la caja da la estimación.",
             [rf"N_{{dentro}} = {int(inside.sum())},\qquad \dfrac{{N_{{dentro}}}}{{N}} = {inside.mean():.5f}",
              rf"\text{{Área}} \approx {box:.0f}\times{inside.mean():.5f} = {area_mc:.4f}"]),
            ("Error esperado", "El conteo sigue una distribución binomial: el error estándar del área es del orden de $\\text{Área}_{caja}\\sqrt{p(1-p)/N}$ con $p$ la fracción dentro. La diferencia observada es de ese orden.",
             [rf"\Delta = {area_mc-abs(cr):+.4f},\qquad \text{{Área}}_{{caja}}\sqrt{{p(1-p)/N}} = {box*np.sqrt(inside.mean()*(1-inside.mean())/n):.4f}"]),
            ("Invarianza al girar los ejes", "Además se gira el sistema de coordenadas de 0° a 360° en pasos de 1° y se recalcula $\\vec A\\cdot\\vec B$ en los ejes nuevos con $x'=x\\cos\\varphi+y\\sin\\varphi$, $y'=-x\\sin\\varphi+y\\cos\\varphi$. Ejemplo con $\\varphi=30^\\circ$:",
             [rf"\vec A' = {_v(ra)},\quad \vec B' = {_v(rb)},\quad \vec A'\cdot\vec B' = {np.dot(ra, rb):.6f}",
              rf"\text{{máx}}-\text{{mín}}\ \text{{de }}\vec A\cdot\vec B\ \text{{en los 361 giros}} = {sci(np.ptp(dots), 2)}"]),
        ],
    )


# ====================================================================== cap. 3
def _inst_vel() -> dict:
    t0 = 2.0
    x = m1d.x_of_t
    tab_secant = m1d.secant_table(t0)
    a_fd = m1d.central_second_difference(t0, 1e-2)
    disp = m1d.displacement_from_velocity(0.0, 4.0, 32)
    disp4 = m1d.displacement_from_velocity(0.0, 4.0, 4)
    t32 = np.linspace(0, 4, 33)
    v32 = m1d.v_analytic(t32)
    h = 0.01
    st = r"\begin{array}{c|cc|c} \Delta t\ (\mathrm{s}) & x(2+\Delta t) & \Delta x/\Delta t\ (\mathrm{m/s}) & \text{error}\\ \hline "
    for r in tab_secant:
        st += rf"{r['dt']:g} & {x(t0 + r['dt']):.6f} & {r['slope']:.6f} & {sci(r['error'], 2)}\\ "
    st += r"\end{array}"
    t4 = np.linspace(0, 4, 5)
    v4 = m1d.v_analytic(t4)
    return dict(
        explanation=[
            "La velocidad media entre dos instantes es el cociente $\\Delta x/\\Delta t$: la pendiente de la recta secante a la curva $x(t)$. Si el intervalo se hace cada vez más corto, la secante se aproxima a la tangente, y su pendiente es la velocidad instantánea, la derivada $dx/dt$.",
            "La aceleración es la rapidez con que cambia la velocidad, $a=dv/dt=d^2x/dt^2$. Gráficamente es la pendiente de $v(t)$ o la curvatura de $x(t)$. Con $x(t)$ polinómica, derivar es inmediato; con datos medidos hay que aproximar derivadas con diferencias finitas.",
            "El proceso inverso también tiene interpretación geométrica: el desplazamiento entre dos instantes es el área bajo la curva $v(t)$, es decir $\\Delta x=\\int v\\,dt$. Una velocidad negativa aporta área negativa (la partícula retrocede), de modo que la integral da el desplazamiento neto, no la distancia recorrida.",
            "En la práctica, las derivadas numéricas son delicadas: reducir $\\Delta t$ disminuye el error de truncamiento, pero si se hace demasiado pequeño aparece el error de redondeo por restar números casi iguales. Por eso la aceleración se calcula con una diferencia centrada y un paso moderado.",
        ],
        analytic_steps=[
            ("Posición dada", "Se parte de la función de posición del enunciado, con $x$ en metros y $t$ en segundos.",
             [r"x(t) = 2 + 3t - \dfrac{t^2}{2} + \dfrac{t^3}{10}"]),
            ("Derivar para obtener la velocidad", "Se deriva término a término con la regla $d(t^n)/dt=nt^{n-1}$; las constantes desaparecen.",
             [r"v(t)=\dfrac{dx}{dt} = 3 - t + \dfrac{3}{10}t^2"]),
            ("Evaluar la velocidad en t = 2 s", "Se sustituye $t=2$ s.",
             [rf"v(2) = 3 - 2 + 0.3(2)^2 = 3 - 2 + {0.3*4:.1f} = {float(m1d.v_analytic(2.0)):.2f}\ \mathrm{{m/s}}"]),
            ("Derivar de nuevo para obtener la aceleración", "La aceleración es la derivada de la velocidad.",
             [r"a(t)=\dfrac{dv}{dt} = -1 + 0.6\,t",
              rf"a(2) = -1 + 0.6(2) = {float(m1d.a_analytic(2.0)):.2f}\ \mathrm{{m/s^2}}"]),
            ("Plantear el desplazamiento como integral", "El desplazamiento neto entre 0 y 4 s es la integral de la velocidad. Se integra el polinomio término a término.",
             [r"\Delta x = \int_0^4\left(3 - t + 0.3t^2\right)dt = \left[3t - \dfrac{t^2}{2} + 0.1t^3\right]_0^4"]),
            ("Evaluar los límites", "Se sustituye el límite superior; el inferior da cero.",
             [rf"\Delta x = 3(4) - \dfrac{{(4)^2}}{{2}} + 0.1(4)^3 = {12:.1f} - {8:.1f} + {6.4:.1f} = {disp['exact']:.2f}\ \mathrm{{m}}"]),
            ("Verificar con las posiciones", "Como comprobación, se evalúan las posiciones inicial y final y se restan.",
             [rf"x(0) = {float(x(0.0)):.2f}\ \mathrm{{m}},\qquad x(4) = {float(x(4.0)):.2f}\ \mathrm{{m}},\qquad x(4)-x(0) = {float(x(4.0)-x(0.0)):.2f}\ \mathrm{{m}}"]),
        ],
        numeric_steps=[
            ("Velocidad como límite de pendientes de secantes", "Se calcula $x(2+\\Delta t)$ para varios $\\Delta t$ y se forma $(x(2+\\Delta t)-x(2))/\\Delta t$. Primero la posición base:",
             [rf"x(2) = 2 + 3(2) - \dfrac{{(2)^2}}{{2}} + \dfrac{{(2)^3}}{{10}} = {float(x(2.0)):.6f}\ \mathrm{{m}}"]),
            ("Primera secante a mano (Δt = 1 s)", "Se evalúa $x(3)$ y se resta.",
             [rf"x(3) = 2 + 9 - 4.5 + 2.7 = {float(x(3.0)):.4f}\ \mathrm{{m}}",
              rf"\dfrac{{x(3)-x(2)}}{{1}} = \dfrac{{{float(x(3.0)):.4f}-{float(x(2.0)):.4f}}}{{1}} = {tab_secant[0]['slope']:.4f}\ \mathrm{{m/s}}"]),
            ("Tabla de convergencia", "Al reducir $\\Delta t$ las secantes se acercan a $v(2)=2.20$ m/s. El error disminuye aproximadamente proporcional a $\\Delta t$ (primer orden): vale $\\approx a\\,\\Delta t/2$.",
             [st]),
            ("Aceleración con la segunda diferencia centrada", "Se combinan tres posiciones simétricas alrededor de $t=2$ con $h=0.01$ s: $a\\approx[x(t+h)-2x(t)+x(t-h)]/h^2$. Es de segundo orden: exacta para polinomios cuadráticos y con error $O(h^2)$.",
             [rf"x(2.01) = {float(x(2.01)):.8f},\quad x(2) = {float(x(2.0)):.8f},\quad x(1.99) = {float(x(1.99)):.8f}",
              rf"a \approx \dfrac{{{float(x(2.01)):.8f}-2({float(x(2.0)):.8f})+{float(x(1.99)):.8f}}}{{(0.01)^2}} = {a_fd:.4f}\ \mathrm{{m/s^2}}"]),
            ("Regla del trapecio con 4 subintervalos (versión reducida)", "El área bajo $v(t)$ se aproxima con trapecios de ancho $h$: $\\Delta x\\approx h\\left[\\tfrac12 v_0+v_1+\\dots+\\tfrac12 v_n\\right]$. Con $n=4$ ($h=1$ s) se ve la mecánica.",
             [rf"v(0..4) = ({', '.join(f'{u:.2f}' for u in v4)})\ \mathrm{{m/s}}",
              rf"\Delta x \approx 1\left[\tfrac12({v4[0]:.2f}) + {v4[1]:.2f} + {v4[2]:.2f} + {v4[3]:.2f} + \tfrac12({v4[4]:.2f})\right] = {disp4['trapezoid']:.4f}\ \mathrm{{m}}"]),
            ("Trapecio con 32 subintervalos (el valor del código)", "Ahora $h=4/32=0.125$ s. Primeros dos trapecios, cada uno con área $\\tfrac12(v_i+v_{i+1})h$:",
             [rf"v_0 = {v32[0]:.4f},\ v_1 = {v32[1]:.6f},\ v_2 = {v32[2]:.6f}\ \mathrm{{m/s}}",
              rf"A_1 = \tfrac12({v32[0]:.4f}+{v32[1]:.6f})(0.125) = {0.5*(v32[0]+v32[1])*0.125:.6f}\ \mathrm{{m}},\qquad A_2 = \tfrac12({v32[1]:.6f}+{v32[2]:.6f})(0.125) = {0.5*(v32[1]+v32[2])*0.125:.6f}\ \mathrm{{m}}"]),
            ("Sumar los 32 trapecios y comparar", "La suma total se compara con el valor exacto $10.40$ m. El error del trapecio es $O(h^2)$ y crece con la curvatura de $v$ (aquí $v''=0.6$). La suma de Riemann por la izquierda, de primer orden, da un error mucho mayor.",
             [rf"\Delta x_{{trap}} = {disp['trapezoid']:.6f}\ \mathrm{{m}},\quad \Delta x_{{exacto}} = {disp['exact']:.6f}\ \mathrm{{m}},\quad \text{{error}} = {sci(disp['trapezoid']-disp['exact'], 3)}\ \mathrm{{m}}",
              rf"\Delta x_{{Riemann}} = {disp['riemann']:.6f}\ \mathrm{{m}}"]),
        ],
    )


def _const_acc() -> dict:
    v0, g = m1d.V0, m1d.G
    sm = m1d.free_fall_summary()
    y3 = float(m1d.free_fall_analytic(3.0)[0])
    v3 = float(m1d.free_fall_analytic(3.0)[1])
    tt, yy = m1d.free_fall_numeric(2000, "rk4", t1=2.5)
    tr, yr = m1d.free_fall_numeric(10, "rk4")
    te, ye = m1d.free_fall_numeric(10, "euler")
    n, T = 10, 3.0
    h = T / n

    def f(s):
        return np.array([s[1], -g])

    s0 = np.array([0.0, v0])
    k1 = f(s0)
    k2 = f(s0 + h / 2 * k1)
    k3 = f(s0 + h / 2 * k2)
    k4 = f(s0 + h * k3)
    s1 = s0 + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    e1 = s0 + h * k1
    conv = m1d.free_fall_convergence()
    tab = r"\begin{array}{c|cc|cc} k & t_k & y^{\text{Euler}}_k & y^{\mathrm{RK4}}_k & y_k^{\text{exacto}}\\ \hline "
    for i in range(0, 11, 2):
        tab += rf"{i} & {tr[i]:.1f} & {ye[i,0]:.4f} & {yr[i,0]:.4f} & {float(m1d.free_fall_analytic(tr[i])[0]):.4f}\\ "
    tab += r"\end{array}"
    ctab = r"\begin{array}{c|cc|cc} N & h\ (\mathrm{s}) & |\text{err}_{\text{Euler}}| & |\text{err}_{\mathrm{RK4}}| \\ \hline "
    for r in conv:
        ctab += rf"{r['n']} & {r['dt']:.5f} & {sci(r['err_euler'], 3)} & {sci(r['err_rk4'], 2)}\\ "
    ctab = ctab.replace(r"c|cc|cc", r"c|cc|c", 1) + r"\end{array}"
    ctab = ctab.replace(r"\end{array}\end{array}", r"\end{array}")
    imax = int(np.argmax(yy[:, 0]))
    return dict(
        explanation=[
            "Cuando la aceleración es constante, la velocidad cambia linealmente con el tiempo y la posición es una parábola. Las tres ecuaciones cinemáticas ($v=v_0+at$, $x=x_0+v_0t+\\tfrac12at^2$ y $v^2=v_0^2+2a\\Delta x$) salen de integrar $a$ una y dos veces; la tercera elimina el tiempo.",
            "La caída libre es el caso más importante: sin resistencia del aire, todos los cuerpos aceleran hacia abajo con la misma $g\\approx9.81$ m/s$^2$, sea cual sea su masa. Si se elige el eje $y$ positivo hacia arriba, entonces $a=-g$ tanto durante la subida como durante la bajada.",
            "Eso implica que el movimiento es simétrico: en el punto más alto la velocidad es cero, el tiempo de subida es igual al de bajada, y el cuerpo regresa al punto de partida con la misma rapidez con que salió, pero dirigida hacia abajo ($v=-v_0$).",
            "El signo importa: una velocidad positiva significa que el cuerpo sube y una negativa que baja, pero la aceleración es negativa en todo el trayecto. Por eso en la cima $v=0$ no implica $a=0$: la pelota está instantáneamente en reposo, pero su velocidad sigue cambiando 9.81 m/s cada segundo.",
            "La integración numérica de $\\dot y=v$, $\\dot v=-g$ sirve para comparar métodos. Euler usa solo la pendiente al inicio del paso, es de primer orden y acumula error; RK4 evalúa cuatro pendientes por paso y es exacto para este problema porque la solución es un polinomio de grado 2.",
        ],
        analytic_steps=[
            ("Elegir el sistema de referencia", "Se toma el eje $y$ hacia arriba con origen en el suelo. Entonces $y_0=0$, $v_0=+20$ m/s y la aceleración es $a=-g=-9.81$ m/s$^2$.",
             [rf"y_0 = 0,\qquad v_0 = {v0:.0f}\ \mathrm{{m/s}},\qquad a = -g = -{g}\ \mathrm{{m/s^2}}"]),
            ("Ecuaciones del movimiento", "Se particularizan las ecuaciones cinemáticas a este caso.",
             [r"v(t) = v_0 - gt,\qquad y(t) = v_0 t - \tfrac{1}{2}gt^2"]),
            ("Tiempo de ascenso", "En el punto más alto la velocidad se anula. Se iguala $v(t)=0$ y se despeja el tiempo.",
             [rf"0 = v_0 - g\,t_{{\max}}\ \Rightarrow\ t_{{\max}} = \dfrac{{v_0}}{{g}} = \dfrac{{{v0:.0f}}}{{{g}}} = {sm['t_peak']:.4f}\ \mathrm{{s}}"]),
            ("Altura máxima", "Se evalúa $y$ en ese instante; es equivalente a usar $v^2=v_0^2-2gy$ con $v=0$.",
             [rf"y_{{\max}} = v_0t_{{\max}} - \tfrac12 g t_{{\max}}^2 = \dfrac{{v_0^2}}{{2g}} = \dfrac{{({v0:.0f})^2}}{{2({g})}} = {sm['y_max']:.4f}\ \mathrm{{m}}"]),
            ("Tiempo de vuelo", "La pelota regresa al suelo cuando $y=0$ (distinto de $t=0$). Se factoriza $t(v_0-\\tfrac12gt)=0$ y se toma la raíz no nula; resulta el doble del tiempo de ascenso.",
             [rf"T = \dfrac{{2v_0}}{{g}} = \dfrac{{2({v0:.0f})}}{{{g}}} = {sm['t_return']:.4f}\ \mathrm{{s}}"]),
            ("Altura y velocidad en t = 3 s", "Como $3\\ \\mathrm{s}<T$, la pelota aún está en el aire y ya está bajando (velocidad negativa), pues $3 > t_{\\max}$.",
             [rf"y(3) = (20)(3) - \tfrac12({g})(3)^2 = {20*3:.0f} - {0.5*g*9:.3f} = {y3:.4f}\ \mathrm{{m}}",
              rf"v(3) = 20 - ({g})(3) = {v3:.2f}\ \mathrm{{m/s}}"]),
            ("Verificar la velocidad de retorno", "Al regresar al suelo, $v(T)=v_0-gT=-v_0$: misma rapidez, sentido opuesto.",
             [rf"v(T) = {v0:.0f} - ({g})({sm['t_return']:.4f}) = {v0 - g*sm['t_return']:.2f}\ \mathrm{{m/s}}"]),
        ],
        numeric_steps=[
            ("Pasar a un sistema de primer orden", "Se llama $\\vec s=(y,v)$ y se integra $\\dot{\\vec s}=(v,-g)$. Con 10 pasos hasta $t=3$ s el paso es $h=0.3$ s.",
             [rf"\vec s_0 = (0,\ {v0:.0f}),\qquad f(\vec s)=(v,\,-{g}),\qquad h = \dfrac{{{T:.0f}}}{{{n}}} = {h:.1f}\ \mathrm{{s}}"]),
            ("Primer paso de Euler", "Euler avanza con la pendiente al inicio del intervalo, $\\vec s_1=\\vec s_0+h\\,f(\\vec s_0)$. Es una línea recta tangente: subestima la curvatura.",
             [rf"f(\vec s_0) = ({v0:.0f},\ -{g}),\qquad \vec s_1 = ({0+h*k1[0]:.4f},\ {v0 + h*k1[1]:.4f}) = ({e1[0]:.4f},\ {e1[1]:.4f})"]),
            ("Primer paso de RK4 a mano", "RK4 evalúa la pendiente al inicio ($k_1$), dos veces en el punto medio ($k_2,k_3$) y al final ($k_4$), y las combina con pesos $1,2,2,1$.",
             [rf"k_1 = ({k1[0]:.4f},\ {k1[1]:.2f}),\qquad k_2 = f(\vec s_0+\tfrac{{h}}{{2}}k_1) = ({k2[0]:.4f},\ {k2[1]:.2f})",
              rf"k_3 = f(\vec s_0+\tfrac{{h}}{{2}}k_2) = ({k3[0]:.4f},\ {k3[1]:.2f}),\qquad k_4 = f(\vec s_0+hk_3) = ({k4[0]:.4f},\ {k4[1]:.2f})",
              rf"\vec s_1 = \vec s_0+\dfrac{{h}}{{6}}(k_1+2k_2+2k_3+k_4) = ({s1[0]:.4f},\ {s1[1]:.4f})"]),
            ("Comparar el primer paso con la solución exacta", "La exacta en $t=0.3$ s es $y=v_0t-\\tfrac12gt^2$. RK4 la reproduce porque $y(t)$ es un polinomio de grado 2; Euler se queda corto porque ignora el término $-\\tfrac12gt^2$.",
             [rf"y_{{exacto}}(0.3) = (20)(0.3)-\tfrac12({g})(0.09) = {float(m1d.free_fall_analytic(0.3)[0]):.4f}\ \mathrm{{m}}",
              rf"\text{{error Euler}} = {abs(e1[0]-float(m1d.free_fall_analytic(0.3)[0])):.4f}\ \mathrm{{m}},\qquad \text{{error RK4}} = {sci(abs(s1[0]-float(m1d.free_fall_analytic(0.3)[0])), 2)}\ \mathrm{{m}}"]),
            ("Repetir los 10 pasos", "Cada paso usa el resultado anterior como punto de partida. Se muestran los pasos pares:",
             [tab]),
            ("Resultado en t = 3 s y orden de convergencia", "Con 10 pasos Euler queda lejos de $y(3)$, mientras que RK4 coincide con la solución exacta salvo el redondeo. Al duplicar $N$ el error de Euler se divide aproximadamente por 2 (orden 1).",
             [rf"y^{{Euler}}(3) = {ye[-1,0]:.4f},\quad y^{{RK4}}(3) = {yr[-1,0]:.4f},\quad y_{{exacto}}(3) = {y3:.4f}\ \mathrm{{m}}",
              ctab]),
            ("Altura máxima con 2000 pasos de RK4", "Se integra hasta $t=2.5$ s con $h=2.5/2000=0.00125$ s y se toma el máximo de las muestras de $y$. El máximo cae en un nodo vecino a $t_{\\max}=2.0387$ s, de modo que la muestra difiere del máximo exacto solo en una cantidad del orden $\\tfrac12 g\\,\\delta t^2$.",
             [rf"y_{{\max}}^{{num}} = {yy[:,0].max():.6f}\ \mathrm{{m}}\ \text{{en }}t={tt[imax]:.5f}\ \mathrm{{s}},\qquad y_{{\max}}^{{an}} = {sm['y_max']:.6f}\ \mathrm{{m}}",
              rf"\text{{error}} = {sci(yy[:,0].max()-sm['y_max'], 2)}\ \mathrm{{m}}"]),
        ],
    )


# ====================================================================== cap. 4
def _projectile() -> dict:
    v0, th, g = c4.V0, c4.THETA_DEG, c4.G
    an = c4.projectile_analytic()
    t, s = c4.projectile_numeric(dt=0.005)
    best = c4.optimal_angle_numeric()
    thr = np.deg2rad(th)
    vx, vy = v0 * np.cos(thr), v0 * np.sin(thr)
    dt = 0.005

    def f(q):
        return np.array([q[2], q[3], 0.0, -g])

    q0 = np.array([0.0, 0.0, vx, vy])
    k1 = f(q0)
    k2 = f(q0 + dt / 2 * k1)
    k3 = f(q0 + dt / 2 * k2)
    k4 = f(q0 + dt * k3)
    q1 = q0 + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    # ultimo paso antes del impacto
    qa, qb = s[-2], None
    n_steps = len(t) - 1
    ta = t[-2]
    qb = rk4_step(lambda _t, q: f(q), ta, qa, dt)
    frac = qa[1] / (qa[1] - qb[1])
    hit = qa + frac * (qb - qa)
    # seccion aurea: primeras iteraciones
    phi = (np.sqrt(5) - 1) / 2
    a_, b_ = 5.0, 85.0
    c_, d_ = b_ - phi * (b_ - a_), a_ + phi * (b_ - a_)
    fc, fd = c4.range_numeric(c_), c4.range_numeric(d_)
    rows_gs = [(a_, b_, c_, d_, fc, fd)]
    for _ in range(3):
        if fc > fd:
            b_, d_, fd = d_, c_, fc
            c_ = b_ - phi * (b_ - a_)
            fc = c4.range_numeric(c_)
        else:
            a_, c_, fc = c_, d_, fd
            d_ = a_ + phi * (b_ - a_)
            fd = c4.range_numeric(d_)
        rows_gs.append((a_, b_, c_, d_, fc, fd))
    tab = r"\begin{array}{c|cc|cc|cc} k & a & b & c & d & R(c) & R(d)\\ \hline "
    for i, (a1, b1, c1_, d1, fc1, fd1) in enumerate(rows_gs):
        tab += rf"{i} & {a1:.4f} & {b1:.4f} & {c1_:.4f} & {d1:.4f} & {fc1:.4f} & {fd1:.4f}\\ "
    tab += r"\end{array}"
    return dict(
        explanation=[
            "El movimiento de un proyectil se descompone en dos movimientos independientes. Horizontalmente no hay aceleración (se ignora el aire), así que la velocidad horizontal $v_0\\cos\\theta$ es constante. Verticalmente actúa la gravedad, que da una aceleración $-g$ y produce un movimiento de caída libre con velocidad inicial $v_0\\sin\\theta$.",
            "La independencia de ambos movimientos es la idea central: el tiempo es el único parámetro común. Por eso el tiempo de vuelo depende solo del movimiento vertical y el alcance es el producto de la velocidad horizontal por ese tiempo. La trayectoria resultante es una parábola.",
            "Combinando ambos se obtiene $R=v_0^2\\sin2\\theta/g$. Como $\\sin2\\theta$ vale como máximo 1 en $\\theta=45^\\circ$, ese es el ángulo de alcance máximo en terreno plano; dos ángulos complementarios ($\\theta$ y $90^\\circ-\\theta$) dan el mismo alcance, aunque uno tarda más y sube más.",
            "Las fórmulas suponen que el punto de lanzamiento y el de llegada están a la misma altura y que no hay resistencia del aire. Con aire, el alcance real es menor y el ángulo óptimo baja de $45^\\circ$. En este caso el orden de magnitud es el de un lanzamiento deportivo: unos 60 m con 25 m/s.",
        ],
        analytic_steps=[
            ("Descomponer la velocidad inicial", "Se proyecta $\\vec v_0$ sobre los ejes horizontal y vertical.",
             [rf"v_{{0x}} = v_0\cos\theta = {v0:.0f}\cos{th:.0f}^\circ = {vx:.4f}\ \mathrm{{m/s}},\qquad v_{{0y}} = v_0\sin\theta = {v0:.0f}\sin{th:.0f}^\circ = {vy:.4f}\ \mathrm{{m/s}}"]),
            ("Escribir las ecuaciones del movimiento", "Movimiento uniforme en $x$ y uniformemente acelerado en $y$.",
             [r"x(t) = v_{0x}\,t,\qquad y(t) = v_{0y}\,t - \tfrac{1}{2}gt^2,\qquad v_y(t)=v_{0y}-gt"]),
            ("Tiempo de vuelo", "El proyectil regresa al suelo cuando $y=0$ con $t\\neq0$. De $t(v_{0y}-\\tfrac12gt)=0$ se toma la segunda raíz.",
             [rf"T = \dfrac{{2v_{{0y}}}}{{g}} = \dfrac{{2({vy:.4f})}}{{{g}}} = {an['t_flight']:.4f}\ \mathrm{{s}}"]),
            ("Altura máxima", "En la cima $v_y=0$, lo que ocurre en $t=T/2$. Con $v_y^2=v_{0y}^2-2gy$ y $v_y=0$:",
             [rf"H = \dfrac{{v_{{0y}}^2}}{{2g}} = \dfrac{{({vy:.4f})^2}}{{2({g})}} = {an['height']:.4f}\ \mathrm{{m}}",
              rf"t_{{apex}} = \dfrac{{v_{{0y}}}}{{g}} = {an['t_apex']:.4f}\ \mathrm{{s}}"]),
            ("Alcance horizontal", "Se multiplica la velocidad horizontal por el tiempo de vuelo. Con la identidad $2\\sin\\theta\\cos\\theta=\\sin2\\theta$ queda la fórmula compacta.",
             [rf"R = v_{{0x}}T = ({vx:.4f})({an['t_flight']:.4f}) = {vx*an['t_flight']:.4f}\ \mathrm{{m}}",
              rf"R = \dfrac{{v_0^2\sin2\theta}}{{g}} = \dfrac{{({v0:.0f})^2\sin{2*th:.0f}^\circ}}{{{g}}} = {an['range']:.4f}\ \mathrm{{m}}"]),
            ("Ángulo de alcance máximo", "Con $v_0$ fijo, $R(\\theta)\\propto\\sin2\\theta$. Se deriva e iguala a cero: $2\\cos2\\theta=0$, es decir $2\\theta=90^\\circ$.",
             [rf"\dfrac{{dR}}{{d\theta}} = \dfrac{{2v_0^2\cos2\theta}}{{g}} = 0\ \Rightarrow\ \theta^* = 45^\circ,\qquad R_{{\max}} = \dfrac{{v_0^2}}{{g}} = {v0**2/g:.4f}\ \mathrm{{m}}"]),
            ("Comprobar con la altura en la cima", "Un chequeo: la altura sobre el suelo a $t_{apex}$ calculada con $y(t)$ debe coincidir con $H$.",
             [rf"y(t_{{apex}}) = ({vy:.4f})({an['t_apex']:.4f}) - \tfrac12({g})({an['t_apex']:.4f})^2 = {vy*an['t_apex']-0.5*g*an['t_apex']**2:.4f}\ \mathrm{{m}}"]),
        ],
        numeric_steps=[
            ("Sistema de primer orden", "Se integra el vector de estado $\\vec q=(x,y,v_x,v_y)$ con $\\dot{\\vec q}=(v_x,v_y,0,-g)$. El paso es $h=0.005$ s y el estado inicial sale de los datos.",
             [rf"\vec q_0 = (0,\ 0,\ {vx:.4f},\ {vy:.4f}),\qquad h = 0.005\ \mathrm{{s}}"]),
            ("Primer paso RK4: las cuatro pendientes", "Como $f$ no depende de $x,y$ y la aceleración es constante, las cuatro pendientes difieren solo por el valor de $v_y$ en cada punto de evaluación.",
             [rf"k_1 = ({k1[0]:.4f},\ {k1[1]:.4f},\ 0,\ {k1[3]:.2f}),\qquad k_2 = ({k2[0]:.4f},\ {k2[1]:.4f},\ 0,\ {k2[3]:.2f})",
              rf"k_3 = ({k3[0]:.4f},\ {k3[1]:.4f},\ 0,\ {k3[3]:.2f}),\qquad k_4 = ({k4[0]:.4f},\ {k4[1]:.4f},\ 0,\ {k4[3]:.2f})"]),
            ("Combinar las pendientes", "El nuevo estado es $\\vec q_1=\\vec q_0+\\tfrac{h}{6}(k_1+2k_2+2k_3+k_4)$. Coincide con la solución exacta en $t=h$ porque el movimiento es polinómico de grado 2.",
             [rf"\vec q_1 = ({q1[0]:.6f},\ {q1[1]:.6f},\ {q1[2]:.4f},\ {q1[3]:.5f})",
              rf"x_{{exacto}}(h) = {vx*dt:.6f},\qquad y_{{exacto}}(h) = {vy*dt-0.5*g*dt**2:.6f}"]),
            ("Iterar hasta el impacto", "Se repite el paso mientras $y\\ge0$. El primer paso que deja $y<0$ cierra la trayectoria. Aquí la integración dura " + f"{n_steps} pasos completos (más una fracción) hasta aproximadamente $t={t[-2]:.3f}$ s.",
             [rf"\text{{último estado con }}y\ge0:\ t_a={ta:.3f}\ \mathrm{{s}},\ (x_a,y_a)=({qa[0]:.5f},\ {qa[1]:.5f})",
              rf"\text{{primer estado con }}y<0:\ t_b={ta+dt:.3f}\ \mathrm{{s}},\ (x_b,y_b)=({qb[0]:.5f},\ {qb[1]:.5f})"]),
            ("Refinar el impacto por interpolación lineal", "Se busca la fracción del paso en la que la recta entre ambos estados cruza $y=0$ y con ella se interpolan el tiempo y la posición horizontal.",
             [rf"f = \dfrac{{y_a}}{{y_a-y_b}} = \dfrac{{{qa[1]:.5f}}}{{{qa[1]:.5f}-({qb[1]:.5f})}} = {frac:.5f}",
              rf"T_{{num}} = t_a + f\,h = {t[-1]:.5f}\ \mathrm{{s}},\qquad R_{{num}} = x_a + f(x_b-x_a) = {s[-1,0]:.5f}\ \mathrm{{m}}"]),
            ("Altura máxima y comparación con lo analítico", "La altura máxima es el mayor $y$ de las muestras; al ser un nodo próximo a la cima, coincide con $H$ salvo un error muy pequeño de discretización (la parábola es plana cerca de la cima).",
             [rf"H_{{num}} = {s[:,1].max():.6f}\ \mathrm{{m}}\ \ (H_{{an}} = {an['height']:.6f})",
              rf"T_{{num}}-T_{{an}} = {sci(t[-1]-an['t_flight'], 2)}\ \mathrm{{s}},\qquad R_{{num}}-R_{{an}} = {sci(s[-1,0]-an['range'], 2)}\ \mathrm{{m}}"]),
            ("Buscar el ángulo óptimo: sección áurea", "Para maximizar $R(\\theta)$ sin derivar, se encierra el máximo en $[a,b]=[5^\\circ,85^\\circ]$ y se reduce el intervalo por el factor $\\varphi=(\\sqrt5-1)/2=0.618$ en cada iteración, evaluando el alcance en dos puntos interiores $c=b-\\varphi(b-a)$ y $d=a+\\varphi(b-a)$. Si $R(c)>R(d)$, el máximo está en $[a,d]$; si no, en $[c,b]$. El alcance se calcula con RK4 y $h=0.002$ s.",
             [tab]),
            ("Criterio de parada y resultado", "El código realiza 40 iteraciones; el ancho final es $80\\varphi^{40}\\approx$ " + f"${sci(80*phi**40, 2)}$" + " grados, y el ángulo es el punto medio del intervalo.",
             [rf"\theta^*_{{num}} = \dfrac{{a+b}}{{2}} = {best:.4f}^\circ,\qquad \theta^*_{{an}} = 45^\circ,\qquad \text{{diferencia}} = {sci(best-45, 2)}^\circ"]),
        ],
    )


def _circular() -> dict:
    R, T = c4.RADIUS, c4.PERIOD
    ca = c4.circular_analytic()
    t0, h = 0.7, 0.01
    w = ca["omega"]
    rp, r0, rm = c4.position(t0 + h), c4.position(t0), c4.position(t0 - h)
    vfd = c4.velocity_fd(t0, h)
    afd = c4.acceleration_fd(t0, h)
    tab_fd = c4.fd_error_table(t=t0, hs=(0.5, 0.25, 0.1, 0.05, 0.01))
    tb = r"\begin{array}{c|cc|cc} h\ (\mathrm{s}) & v_{fd} & a_{fd} & |v_{fd}-v| & |a_{fd}-a_c|\\ \hline "
    for r in tab_fd:
        tb += rf"{r['h']:g} & {r['speed']:.6f} & {r['a_c']:.6f} & {sci(r['err_speed'], 2)} & {sci(r['err_a'], 2)}\\ "
    tb += r"\end{array}"
    return dict(
        explanation=[
            "En el movimiento circular uniforme la rapidez es constante, pero la velocidad cambia de dirección a cada instante, y por eso hay aceleración. Esa aceleración apunta hacia el centro de la circunferencia (es centrípeta) y no cambia la rapidez, solo la dirección.",
            "La descripción natural usa la frecuencia angular $\\omega=2\\pi/T$, el ángulo barrido por unidad de tiempo. La rapidez lineal es entonces $v=\\omega R$ (arco recorrido por unidad de tiempo) y la aceleración centrípeta es $a_c=v^2/R=\\omega^2R$. Si se duplica la rapidez con el mismo radio, la aceleración se multiplica por cuatro.",
            "Hay una forma de entender por qué aparece $v^2/R$: en un intervalo corto $\\Delta t$ la velocidad gira un ángulo $\\omega\\Delta t$ y su cambio tiene módulo $v\\,\\omega\\Delta t$; dividiendo entre $\\Delta t$ queda $v\\omega=v^2/R$, dirigido hacia el centro. El resultado vale para cualquier trayectoria curva cuando se usa el radio de curvatura local.",
            "El movimiento circular es la proyección de dos oscilaciones armónicas en cuadratura, $x=R\\cos\\omega t$ y $y=R\\sin\\omega t$. Con $R=2$ m y $T=4$ s, la partícula da un cuarto de vuelta cada segundo y su aceleración es del orden de 5 m/s$^2$, aproximadamente la mitad de $g$.",
        ],
        analytic_steps=[
            ("Frecuencia angular", "En un periodo se barren $2\\pi$ radianes.",
             [rf"\omega = \dfrac{{2\pi}}{{T}} = \dfrac{{2\pi}}{{{T:.0f}\ \mathrm{{s}}}} = {w:.4f}\ \mathrm{{rad/s}}"]),
            ("Escribir la posición", "La posición en cada instante es un vector de módulo $R$ y ángulo $\\omega t$.",
             [r"\vec r(t) = R\,(\cos\omega t,\ \sin\omega t)"]),
            ("Derivar para obtener la velocidad", "La derivada de la posición es perpendicular a ella y su módulo no depende del tiempo.",
             [r"\vec v(t) = \dot{\vec r} = \omega R\,(-\sin\omega t,\ \cos\omega t),\qquad |\vec v| = \omega R"]),
            ("Calcular la rapidez", "Se sustituyen $\\omega$ y $R$. Equivale a la longitud de la circunferencia entre el periodo.",
             [rf"v = \omega R = ({w:.4f})({R:.0f}) = {ca['speed']:.4f}\ \mathrm{{m/s}},\qquad \dfrac{{2\pi R}}{{T}} = \dfrac{{2\pi({R:.0f})}}{{{T:.0f}}} = {2*np.pi*R/T:.4f}\ \mathrm{{m/s}}"]),
            ("Derivar de nuevo para obtener la aceleración", "La segunda derivada es proporcional a $-\\vec r$: apunta al centro.",
             [r"\vec a(t) = \ddot{\vec r} = -\omega^2\vec r(t),\qquad |\vec a| = \omega^2 R"]),
            ("Aceleración centrípeta", "Se calcula con las dos formas equivalentes, $\\omega^2R$ y $v^2/R$.",
             [rf"a_c = \omega^2R = ({w:.4f})^2({R:.0f}) = {ca['a_c']:.4f}\ \mathrm{{m/s^2}}",
              rf"a_c = \dfrac{{v^2}}{{R}} = \dfrac{{({ca['speed']:.4f})^2}}{{{R:.0f}}} = {ca['a_c_from_v']:.4f}\ \mathrm{{m/s^2}}"]),
            ("Comparar con g", "Un chequeo de orden de magnitud.",
             [rf"\dfrac{{a_c}}{{g}} = \dfrac{{{ca['a_c']:.4f}}}{{9.81}} = {ca['a_c']/9.81:.3f}"]),
        ],
        numeric_steps=[
            ("Idea: derivar solo con valores de la posición", "El código supone que solo se conoce $\\vec r(t)$ y estima velocidad y aceleración con diferencias finitas centradas: usan puntos simétricos a ambos lados de $t$, y por eso el error baja como $h^2$.",
             [r"\vec v \approx \dfrac{\vec r(t+h)-\vec r(t-h)}{2h},\qquad \vec a \approx \dfrac{\vec r(t+h)-2\vec r(t)+\vec r(t-h)}{h^2}"]),
            ("Parámetros", "Se evalúa en $t=0.7$ s con $h=0.01$ s. La posición se calcula con $\\omega t$ en radianes.",
             [rf"\omega = {w:.6f}\ \mathrm{{rad/s}},\quad \omega(t-h) = {w*(t0-h):.6f},\quad \omega t = {w*t0:.6f},\quad \omega(t+h) = {w*(t0+h):.6f}\ \mathrm{{rad}}"]),
            ("Tres posiciones", "Se calcula $\\vec r=R(\\cos\\omega t,\\sin\\omega t)$ en los tres instantes.",
             [rf"\vec r(0.69) = {_v(rm, 7)},\quad \vec r(0.70) = {_v(r0, 7)},\quad \vec r(0.71) = {_v(rp, 7)}\ \mathrm{{m}}"]),
            ("Diferencia centrada para la velocidad", "Se resta la posición posterior de la anterior y se divide por $2h=0.02$ s.",
             [rf"\vec v \approx \dfrac{{{_v(rp, 7)}-{_v(rm, 7)}}}{{0.02}} = {_v(vfd, 5)}\ \mathrm{{m/s}}",
              rf"|\vec v| = \sqrt{{({vfd[0]:.5f})^2+({vfd[1]:.5f})^2}} = {np.linalg.norm(vfd):.6f}\ \mathrm{{m/s}}"]),
            ("Segunda diferencia para la aceleración", "Se combinan las tres posiciones con pesos $1,-2,1$ y se divide por $h^2=10^{-4}$ s$^2$. La resta de números casi iguales amplifica el redondeo, por lo que no conviene usar un $h$ demasiado pequeño.",
             [rf"\vec a \approx \dfrac{{{_v(rp, 7)}-2{_v(r0, 7)}+{_v(rm, 7)}}}{{10^{{-4}}}} = {_v(afd, 4)}\ \mathrm{{m/s^2}}",
              rf"|\vec a| = {np.linalg.norm(afd):.6f}\ \mathrm{{m/s^2}},\qquad \text{{dirección: }}\vec a\cdot\vec r/(|a||r|) = {np.dot(afd, r0)/(np.linalg.norm(afd)*R):.6f}"]),
            ("Reducir el paso y ver la convergencia", "Al repetir con distintos $h$ el error baja aproximadamente por un factor de 4 cuando $h$ se reduce a la mitad, lo que confirma el orden 2. Con $h$ muy pequeño el redondeo empieza a competir en la aceleración.",
             [tb]),
            ("Resultado final y comparación", "Con $h=0.01$ s la rapidez y la aceleración numéricas coinciden con las analíticas en las cifras mostradas.",
             [rf"v_{{num}} = {tab_fd[-1]['speed']:.6f}\ \mathrm{{m/s}}\ (v_{{an}}={ca['speed']:.6f}),\qquad a_{{num}} = {tab_fd[-1]['a_c']:.6f}\ \mathrm{{m/s^2}}\ (a_{{an}}={ca['a_c']:.6f})"]),
        ],
    )


DETAIL = {
    "DimensionalAnalysis": _dimensional(),
    "ErrorPropagation": _error_prop(),
    "VectorAddition": _vec_add(),
    "VectorProducts": _vec_prod(),
    "InstantaneousVelocity": _inst_vel(),
    "ConstantAcceleration": _const_acc(),
    "Projectile": _projectile(),
    "CircularMotion": _circular(),
}
