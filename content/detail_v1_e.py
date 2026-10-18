"""Explicacion y pasos a mano: capitulos 17 a 20 (VenturiTube ... BeatsAndPipes). Cifras calculadas con physics/."""
from __future__ import annotations

import numpy as np

from physics import ch17_fluid_dynamics as c17
from physics import ch18_oscillations as c18
from physics import ch19_waves as c19
from physics import ch20_sound as c20
from physics.integrators import rk4_step
from content.texfmt import fx, sci


# =============================================================== Venturi
def _venturi() -> dict:
    a1, a2 = c17.areas()
    an = c17.venturi_analytic()
    rho, g = c17.RHO, c17.G
    dp = c17.P1 - an["p2"]
    qa = c17.flow_from_pressure_drop_analytic(dp)
    qn = c17.flow_from_pressure_drop_numeric(dp)
    k = 0.5 * rho * (1 / a2**2 - 1 / a1**2)
    f = lambda q: k * q * q - dp  # noqa: E731
    lo, hi = 0.0, 1.0
    rows = []
    for i in range(1, 7):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        rows.append((i, lo, hi, mid, fm))
        if fm < 0:
            lo = mid
        else:
            hi = mid
    tab = r"\begin{array}{c|c|c|c|c} i & Q_{\min} & Q_{\max} & Q_{med} & f(Q_{med})\ [\mathrm{Pa}]\\ \hline " + r"\\ ".join(
        f"{i} & {fx(l, 5)} & {fx(h, 5)} & {fx(m, 5)} & {sci(fm, 4)}" for i, l, h, m, fm in rows) + r"\end{array}"
    return dict(
        explanation=[
            "Un fluido ideal es incompresible y no tiene viscosidad. Si por un tubo pasa un caudal constante $Q$, el volumen que entra por una sección debe salir por la otra en el mismo tiempo; esto es la ecuación de continuidad $A_1v_1 = A_2v_2$ y dice que donde el tubo se estrecha el fluido tiene que ir más rápido.",
            "La ecuación de Bernoulli es la conservación de la energía por unidad de volumen a lo largo de una línea de corriente: $p + \\tfrac12\\rho v^2 + \\rho g y$ es constante. En un tubo horizontal el término de altura no cambia, de modo que si la rapidez aumenta la presión estática tiene que disminuir.",
            "El resultado es contraintuitivo: en la garganta del Venturi hay más velocidad y menos presión. Esa diferencia de presión se puede medir con un manómetro y, invirtiendo el cálculo, sirve para determinar el caudal sin interrumpir el flujo; por eso el Venturi es un medidor de caudal.",
            "El modelo es válido para flujo estacionario, sin turbulencia ni pérdidas por fricción y con fluido incompresible. En un caso real la caída medida es algo mayor que la ideal y se corrige con un coeficiente de descarga cercano a 1. Aquí la velocidad pasa de pocos metros por segundo a más de diez, y la presión baja del orden de cien kilopascales.",
        ],
        analytic_steps=[
            ("Áreas de las secciones", "Las secciones son circulares, así que cada área es $\\pi d^2/4$. Se calculan con $d_1 = 0.20$ m y $d_2 = 0.10$ m.",
             [rf"A_1 = \dfrac{{\pi (0.20)^2}}{{4}} = {sci(a1, 5)}\ \mathrm{{m^2}},\qquad A_2 = \dfrac{{\pi (0.10)^2}}{{4}} = {sci(a2, 5)}\ \mathrm{{m^2}}"]),
            ("Velocidades con la continuidad", "El caudal es $Q = Av$ en cada sección, por lo que $v = Q/A$. Como $A_2 = A_1/4$, la velocidad en la garganta es cuatro veces mayor.",
             [rf"v_1 = \dfrac{{Q}}{{A_1}} = \dfrac{{0.030}}{{{sci(a1, 5)}}} = {an['v1']:.3f}\ \mathrm{{m/s}}",
              rf"v_2 = \dfrac{{Q}}{{A_2}} = \dfrac{{0.030}}{{{sci(a2, 5)}}} = {an['v2']:.3f}\ \mathrm{{m/s}}"]),
            ("Plantear Bernoulli entre las dos secciones", "El tubo es horizontal, así que los términos $\\rho g y$ se cancelan. Se despeja la presión desconocida $p_2$.",
             [r"p_1 + \tfrac12\rho v_1^2 = p_2 + \tfrac12\rho v_2^2 \quad\Rightarrow\quad p_2 = p_1 + \tfrac12\rho\,(v_1^2 - v_2^2)"]),
            ("Sustituir y obtener la presión en la garganta", "Se usa $\\rho = 1000$ kg/m³ y $p_1 = 200$ kPa. Como $v_2 > v_1$ el paréntesis es negativo y la presión baja.",
             [rf"p_2 = {sci(c17.P1, 3)} + \tfrac12 ({rho:.0f})\left[({an['v1']:.3f})^2 - ({an['v2']:.3f})^2\right]\ \mathrm{{Pa}}",
              rf"p_2 = {sci(c17.P1, 3)} + {fx(0.5 * rho * (an['v1']**2 - an['v2']**2), 1)} = {fx(an['p2'], 1)}\ \mathrm{{Pa}} = {an['p2'] / 1000:.2f}\ \mathrm{{kPa}}"]),
            ("Caída de presión medida", "El manómetro diferencial mide la diferencia entre ambas presiones. Es el dato que se usará para recuperar el caudal.",
             [rf"\Delta p = p_1 - p_2 = {fx(c17.P1, 1)} - {fx(an['p2'], 1)} = {fx(dp, 1)}\ \mathrm{{Pa}}"]),
            ("Despejar el caudal a partir de $\\Delta p$", "Se sustituye $v_1 = Q/A_1$ y $v_2 = Q/A_2$ en Bernoulli y se agrupa $Q^2$. Esto invierte el problema: del dato de presión sale el caudal.",
             [r"\Delta p = \tfrac12\rho Q^2\left(\dfrac{1}{A_2^2} - \dfrac{1}{A_1^2}\right) \quad\Rightarrow\quad Q = A_2\sqrt{\dfrac{2\,\Delta p}{\rho\,[1-(A_2/A_1)^2]}}",
              rf"Q = ({sci(a2, 5)})\sqrt{{\dfrac{{2({fx(dp, 1)})}}{{({rho:.0f})\,[1-({a2 / a1:.2f})^2]}}}} = {qa:.6f}\ \mathrm{{m^3/s}}"]),
        ],
        numeric_steps=[
            ("Convertir la ecuación en una raíz", "El método numérico no despeja $Q$: define $f(Q) = \\tfrac12\\rho Q^2(1/A_2^2 - 1/A_1^2) - \\Delta p$ y busca el $Q$ donde $f = 0$. Es creciente para $Q > 0$, así que hay una única raíz.",
             [rf"f(Q) = {sci(k, 5)}\,Q^2 - {fx(dp, 1)}\ \mathrm{{Pa}},\qquad \tfrac12\rho\left(\dfrac{{1}}{{A_2^2}} - \dfrac{{1}}{{A_1^2}}\right) = {sci(k, 5)}"]),
            ("Intervalo inicial y regla de bisección", "Se parte de un intervalo que encierra la raíz, $[0, 1]$ m³/s: $f(0) < 0$ y $f(1) > 0$. En cada paso se evalúa el punto medio; si $f<0$ el extremo inferior pasa al punto medio, y si no, el superior.",
             [rf"Q_{{\min}} = 0,\quad Q_{{\max}} = 1,\qquad f(0) = {sci(f(0.0), 4)}\ \mathrm{{Pa}},\quad f(1) = {sci(f(1.0), 4)}\ \mathrm{{Pa}}"]),
            ("Primera bisección", "El punto medio es $Q = 0.5$. Su $f$ es muy positivo, así que la raíz está en la mitad inferior y se descarta $[0.5, 1]$.",
             [rf"Q_{{med}} = \dfrac{{0 + 1}}{{2}} = 0.5,\qquad f(0.5) = {sci(k, 5)}(0.5)^2 - {fx(dp, 1)} = {sci(rows[0][4], 4)}\ \mathrm{{Pa}} > 0 \;\Rightarrow\; Q_{{\max}} = 0.5"]),
            ("Siguientes bisecciones", "Cada paso reduce a la mitad la longitud del intervalo. Las primeras seis iteraciones son:",
             [tab]),
            ("Criterio de parada", "El código repite exactamente 80 bisecciones, por lo que el intervalo final mide $1/2^{80}$, muy por debajo de la precisión de la aritmética de 64 bits. El resultado es el punto medio del último intervalo.",
             [rf"\text{{ancho final}} = 2^{{-80}} = {sci(2.0**-80, 3)}\ \mathrm{{m^3/s}}"]),
            ("Resultado y error", "Se compara el caudal numérico con el analítico y con el caudal original del enunciado, $0.030$ m³/s.",
             [rf"Q_{{num}} = {qn:.9f}\ \mathrm{{m^3/s}},\qquad Q_{{anal}} = {qa:.9f}\ \mathrm{{m^3/s}}",
              rf"|Q_{{num}} - Q_{{anal}}| = {sci(abs(qn - qa), 3)}\ \mathrm{{m^3/s}},\qquad |Q_{{num}} - 0.030| = {sci(abs(qn - c17.Q), 3)}\ \mathrm{{m^3/s}}"]),
        ],
    )


# =============================================================== Torricelli
def _torricelli() -> dict:
    A, a, h0, g = c17.A_TANK, c17.A_HOLE, c17.H0, c17.G
    te = c17.drain_time_analytic()
    v0 = float(c17.torricelli_speed(h0))
    kk = (a / A) * np.sqrt(g / 2)
    t, y = c17.height_numeric(2000)
    mid = len(t) // 2
    h = t[1] - t[0]
    fdot = lambda _t, yy: np.array([-(a / A) * np.sqrt(2 * g * max(yy[0], 0.0))])  # noqa: E731
    y0 = np.array([h0])
    k1 = fdot(0, y0)[0]
    k2 = fdot(0, y0 + h / 2 * k1)[0]
    k3 = fdot(0, y0 + h / 2 * np.array([k2]))[0]
    k4 = fdot(0, y0 + h * np.array([k3]))[0]
    y1 = rk4_step(fdot, 0.0, y0, h)[0]
    tn = c17.drain_time_numeric()
    t2, y2 = c17.height_numeric(20000)
    hh = np.maximum(y2[:, 0], 0.0)
    i = int(np.argmax(hh < 1e-3))
    ha = float(c17.height_analytic(t[mid]))
    return dict(
        explanation=[
            "Un depósito abierto con un pequeño orificio en el fondo se vacía porque la presión del agua en la profundidad $h$ supera a la atmosférica en $\\rho g h$. Aplicando Bernoulli entre la superficie libre (donde el agua casi no se mueve) y el chorro (a presión atmosférica) la energía potencial $\\rho g h$ se convierte en energía cinética $\\tfrac12\\rho v^2$.",
            "De ahí sale la ley de Torricelli, $v = \\sqrt{2gh}$: el agua sale con la misma rapidez que tendría un cuerpo que cae libremente desde la altura $h$. Esta rapidez no depende de la densidad ni del tamaño del orificio.",
            "A medida que el nivel baja, la velocidad de salida disminuye y el vaciado se hace cada vez más lento. El balance de volumen, $A\\,dh/dt = -a\\sqrt{2gh}$, es una ecuación diferencial separable cuya solución es una parábola en el tiempo y que alcanza $h=0$ en un tiempo finito.",
            "La aproximación requiere que el orificio sea mucho menor que la sección del depósito ($a \\ll A$) para despreciar la velocidad de la superficie, y no incluye contracción del chorro ni viscosidad. Con un orificio de 2 cm² en un depósito de 0.5 m² el tiempo de vaciado resulta del orden de media hora.",
        ],
        analytic_steps=[
            ("Velocidad inicial de salida", "Se aplica Bernoulli entre la superficie y el orificio, con $v_{sup}\\approx 0$ y presión atmosférica en ambos puntos, y se despeja $v$.",
             [r"\rho g h = \tfrac12\rho v^2 \;\Rightarrow\; v = \sqrt{2gh}",
              rf"v_0 = \sqrt{{2({g})({h0})}} = {v0:.4f}\ \mathrm{{m/s}}"]),
            ("Balance de volumen", "En un intervalo $dt$ el volumen que pierde el depósito, $-A\\,dh$, es el que sale por el orificio, $a\\,v\\,dt$. Esto da la ecuación diferencial.",
             [r"A\,\dfrac{dh}{dt} = -a\sqrt{2gh}"]),
            ("Separar variables", "Se pasan los $h$ a un lado y los $t$ al otro, y se define la constante $\\kappa = (a/A)\\sqrt{g/2}$ para simplificar.",
             [r"\dfrac{dh}{\sqrt h} = -\dfrac{a}{A}\sqrt{2g}\,dt \;\Rightarrow\; \dfrac{d\sqrt h}{dt} = -\dfrac{a}{A}\sqrt{\dfrac g2} = -\kappa",
              rf"\kappa = \dfrac{{{sci(a, 3)}}}{{{A}}}\sqrt{{\dfrac{{{g}}}{{2}}}} = {sci(kk, 5)}\ \mathrm{{m^{{1/2}}/s}}"]),
            ("Integrar", "Como $d\\sqrt h/dt$ es constante, $\\sqrt h$ decrece linealmente. Se impone $h(0) = h_0$.",
             [r"\sqrt{h(t)} = \sqrt{h_0} - \kappa t \;\Rightarrow\; h(t) = \left(\sqrt{h_0} - \dfrac{a}{A}\sqrt{\dfrac g2}\,t\right)^2",
              rf"h(t) = \left({np.sqrt(h0):.5f} - {sci(kk, 5)}\,t\right)^2\ \mathrm{{m}}"]),
            ("Tiempo de vaciado", "El depósito se vacía cuando $\\sqrt h = 0$, es decir en $t = \\sqrt{h_0}/\\kappa$, que equivale a la expresión del enunciado.",
             [r"t_{vac} = \dfrac{\sqrt{h_0}}{\kappa} = \dfrac{A}{a}\sqrt{\dfrac{2h_0}{g}}",
              rf"t_{{vac}} = \dfrac{{{A}}}{{{sci(a, 3)}}}\sqrt{{\dfrac{{2({h0})}}{{{g}}}}} = {A / a:.0f}\,({np.sqrt(2 * h0 / g):.5f}) = {te:.2f}\ \mathrm{{s}}\approx {te / 60:.1f}\ \mathrm{{min}}"]),
            ("Altura a mitad del tiempo numérico", "Se evalúa la fórmula en el instante $t_*$ que usa la verificación numérica (la mitad del intervalo de integración, $0.525\\,t_{vac}$), para poder comparar.",
             [rf"t_* = {t[mid]:.3f}\ \mathrm{{s}}\;\Rightarrow\; h(t_*) = \left({np.sqrt(h0):.5f} - {sci(kk, 5)}({t[mid]:.3f})\right)^2 = {ha:.6f}\ \mathrm{{m}}"]),
        ],
        numeric_steps=[
            ("Pasar a la forma $y' = f(t,y)$", "Se integra $dh/dt = -(a/A)\\sqrt{2gh}$ con RK4 de paso fijo. La integración se hace hasta $1.05\\,t_{vac}$ para que el final quede dentro del intervalo; $\\max(h,0)$ evita la raíz de un número negativo.",
             [rf"t_{{fin}} = 1.05\,t_{{vac}} = {1.05 * te:.2f}\ \mathrm{{s}},\qquad N = 2000,\qquad \Delta t = {h:.5f}\ \mathrm{{s}}"]),
            ("Pendientes de RK4 en el primer paso", "Con $h_0 = 1.5$ m se calculan las cuatro pendientes: al inicio, dos en el punto medio y una al final del paso.",
             [rf"k_1 = f(h_0) = {k1:.7f}\ \mathrm{{m/s}}",
              rf"k_2 = f\!\left(h_0 + \tfrac{{\Delta t}}{{2}}k_1\right) = {k2:.7f},\qquad k_3 = f\!\left(h_0 + \tfrac{{\Delta t}}{{2}}k_2\right) = {k3:.7f},\qquad k_4 = f(h_0 + \Delta t\,k_3) = {k4:.7f}"]),
            ("Primera actualización", "La nueva altura es el promedio ponderado de las pendientes por el paso.",
             [r"h_1 = h_0 + \dfrac{\Delta t}{6}\,(k_1 + 2k_2 + 2k_3 + k_4)",
              rf"h_1 = {h0} + \dfrac{{{h:.5f}}}{{6}}\,({k1:.6f} + 2({k2:.6f}) + 2({k3:.6f}) + {k4:.6f}) = {y1:.8f}\ \mathrm{{m}}",
              rf"h_{{anal}}(\Delta t) = {float(c17.height_analytic(h)):.8f}\ \mathrm{{m}}"]),
            ("Repetir hasta $t_*$", "El mismo paso se aplica 1000 veces, cada uno partiendo del valor anterior. En la mitad del intervalo se compara con la fórmula analítica.",
             [rf"h_{{num}}(t_*) = {float(y[mid, 0]):.8f}\ \mathrm{{m}},\qquad h_{{anal}}(t_*) = {ha:.8f}\ \mathrm{{m}},\qquad |\Delta h| = {sci(abs(float(y[mid, 0]) - ha), 3)}\ \mathrm{{m}}"]),
            ("Por qué el final es delicado", "Cerca de $h=0$ la derivada $-\\kappa'\\sqrt h$ tiene pendiente infinita en $h$ y RK4 pierde precisión. Por eso el tiempo de vaciado no se toma del último paso sino de una corrección local.",
             [r"\text{Cerca de }h=0:\quad h = \kappa^2\,(t_e - t)^2 \;\Rightarrow\; t_e = t + \dfrac{\sqrt h}{\kappa}"]),
            ("Detección y corrección", "Con 20000 pasos se busca el primer paso en que $h<1$ mm, y desde el paso anterior se aplica la fórmula local con $\\kappa$.",
             [rf"i = {i},\quad t_{{i-1}} = {t2[i - 1]:.4f}\ \mathrm{{s}},\quad h_{{i-1}} = {hh[i - 1]:.6f}\ \mathrm{{m}}",
              rf"t_e = {t2[i - 1]:.4f} + \dfrac{{\sqrt{{{hh[i - 1]:.6f}}}}}{{{sci(kk, 5)}}} = {tn:.4f}\ \mathrm{{s}}"]),
            ("Error del tiempo de vaciado", "Se compara con el valor analítico.",
             [rf"t_{{num}} = {tn:.4f}\ \mathrm{{s}},\qquad t_{{anal}} = {te:.4f}\ \mathrm{{s}},\qquad |\Delta t| = {sci(abs(tn - te), 3)}\ \mathrm{{s}}"]),
        ],
    )


# =============================================================== SimpleHarmonic
def _shm() -> dict:
    m, k, A = c18.M, c18.K, c18.AMP
    w = c18.omega0()
    T = 2 * np.pi / w
    E = c18.energy_analytic()
    t, y = c18.shm_numeric()
    h = t[1] - t[0]
    f = lambda _t, yy: np.array([yy[1], -(k / m) * yy[0]])  # noqa: E731
    y0 = np.array([A, 0.0])
    k1 = f(0, y0)
    k2 = f(0, y0 + h / 2 * k1)
    k3 = f(0, y0 + h / 2 * k2)
    k4 = f(0, y0 + h * k3)
    y1 = rk4_step(f, 0.0, y0, h)
    tn = c18.period_numeric()
    err = float(np.max(np.abs(y[:, 0] - c18.x_analytic(t))))
    tab = c18.pendulum_table((10, 60))
    ts = c18.pendulum_period_small()

    def agm_steps(th):
        a, b = 1.0, float(np.cos(np.deg2rad(th) / 2))
        out = [(a, b)]
        for _ in range(3):
            a, b = 0.5 * (a + b), float(np.sqrt(a * b))
            out.append((a, b))
        return out

    ag = agm_steps(60)
    th = np.deg2rad(60)
    hp = 4000 and 2.0 * np.pi * np.sqrt(1 / c18.G) / 20000
    fp = lambda _t, yy: np.array([yy[1], -(c18.G / 1.0) * np.sin(yy[0])])  # noqa: E731
    yp0 = np.array([th, 0.0])
    p1 = rk4_step(fp, 0.0, yp0, hp)
    return dict(
        explanation=[
            "Un oscilador armónico simple es cualquier sistema cuya fuerza restauradora es proporcional al desplazamiento, $F = -kx$. Con la segunda ley de Newton resulta $\\ddot x = -(k/m)x$, cuyas soluciones son senos y cosenos: el cuerpo va y viene con una frecuencia angular $\\omega = \\sqrt{k/m}$ que no depende de la amplitud.",
            "La energía total se reparte entre cinética y potencial elástica, pero su suma $E = \\tfrac12 kA^2$ es constante. En los extremos toda la energía es potencial y la velocidad es cero; al pasar por el equilibrio la energía es toda cinética y la velocidad es máxima, $v_{max} = A\\omega$.",
            "El péndulo simple solo es armónico de forma aproximada. La fuerza restauradora es proporcional a $\\sin\\theta$ y no a $\\theta$, y para amplitudes grandes el periodo real es mayor que $2\\pi\\sqrt{L/g}$. El periodo exacto se expresa con una integral elíptica, que se puede calcular rápidamente con la media aritmético-geométrica.",
            "Para ángulos pequeños la corrección es mínima (unas décimas de porcentaje a 10°), pero a 60° ya es cercana al 7 %. Esto muestra hasta dónde es útil la linealización y por qué los relojes de péndulo usan amplitudes pequeñas.",
        ],
        analytic_steps=[
            ("Frecuencia angular", "La ecuación de movimiento del bloque es $m\\ddot x = -kx$. Probando $x = A\\cos\\omega t$ resulta $\\omega^2 = k/m$.",
             [rf"\omega = \sqrt{{\dfrac{{k}}{{m}}}} = \sqrt{{\dfrac{{{k:.0f}}}{{{m}}}}} = {w:.1f}\ \mathrm{{rad/s}}"]),
            ("Periodo", "El periodo es el tiempo de una oscilación completa, $2\\pi/\\omega$.",
             [rf"T = \dfrac{{2\pi}}{{\omega}} = \dfrac{{2\pi}}{{{w:.1f}}} = {T:.6f}\ \mathrm{{s}}"]),
            ("Velocidad máxima y energía", "Soltado desde el reposo en $x = 8$ cm la amplitud es $A = 0.08$ m. La velocidad es máxima al pasar por el equilibrio; la energía se evalúa en el extremo, donde es solo elástica.",
             [rf"v_{{max}} = A\omega = ({A})({w:.1f}) = {A * w:.2f}\ \mathrm{{m/s}}",
              rf"E = \tfrac12 kA^2 = \tfrac12({k:.0f})({A})^2 = {E:.3f}\ \mathrm{{J}} = \tfrac12 m v_{{max}}^2 = \tfrac12({m})({A * w:.2f})^2"]),
            ("Péndulo: periodo de pequeñas oscilaciones", "Con $L = 1$ m y $g = 9.81$ m/s² se linealiza $\\sin\\theta \\approx \\theta$. Este es el periodo de referencia.",
             [rf"T_0 = 2\pi\sqrt{{\dfrac{{L}}{{g}}}} = 2\pi\sqrt{{\dfrac{{1}}{{{c18.G}}}}} = {ts:.5f}\ \mathrm{{s}}"]),
            ("Péndulo: media aritmético-geométrica", "Para amplitud $\\theta_0$ el periodo exacto es $T = T_0 / \\mathrm{AGM}(1,\\cos(\\theta_0/2))$. La AGM se calcula repitiendo $a' = (a+b)/2$, $b' = \\sqrt{ab}$ hasta que ambos coincidan. Para 60°, $\\cos 30^\\circ = 0.86603$:",
             [r"\begin{array}{c|c|c} n & a_n & b_n\\ \hline " + r"\\ ".join(f"{i} & {fx(a_, 8)} & {fx(b_, 8)}" for i, (a_, b_) in enumerate(ag)) + r"\end{array}",
              rf"\mathrm{{AGM}}(1,\cos 30^\circ) = {c18.agm(1.0, float(np.cos(np.deg2rad(30)))):.8f}"]),
            ("Péndulo: resultados", "Se divide el periodo de referencia por la AGM. La última columna es el aumento respecto de la aproximación lineal.",
             [rf"\theta_0 = 10^\circ:\ \ T = {tab[0]['exact']:.5f}\ \mathrm{{s}}\quad(+{(tab[0]['exact'] / ts - 1) * 100:.2f}\,\%)",
              rf"\theta_0 = 60^\circ:\ \ T = \dfrac{{{ts:.5f}}}{{{c18.agm(1.0, float(np.cos(np.deg2rad(30)))):.8f}}} = {tab[1]['exact']:.5f}\ \mathrm{{s}}\quad(+{(tab[1]['exact'] / ts - 1) * 100:.2f}\,\%)"]),
        ],
        numeric_steps=[
            ("Sistema de primer orden", "Se define $y = (x, v)$ y $y' = (v,\\,-(k/m)x)$ para poder aplicar RK4. Se integra 1 s con 2000 pasos.",
             [rf"\Delta t = \dfrac{{1}}{{2000}} = {sci(h, 3)}\ \mathrm{{s}},\qquad y_0 = ({A},\,0),\qquad k/m = {k / m:.0f}\ \mathrm{{s^{{-2}}}}"]),
            ("Pendientes del primer paso", "Cada pendiente es $(v, -400x)$ evaluada en un punto distinto del paso.",
             [rf"k_1 = (0,\,{k1[1]:.3f}),\qquad k_2 = ({k2[0]:.5f},\,{k2[1]:.5f})",
              rf"k_3 = ({k3[0]:.5f},\,{k3[1]:.5f}),\qquad k_4 = ({k4[0]:.5f},\,{k4[1]:.5f})"]),
            ("Primera actualización", "Se combinan las pendientes y se compara con la solución exacta $A\\cos\\omega\\Delta t$.",
             [rf"x_1 = {A} + \dfrac{{\Delta t}}{{6}}(k_{{1x}} + 2k_{{2x}} + 2k_{{3x}} + k_{{4x}}) = {y1[0]:.10f}\ \mathrm{{m}}",
              rf"v_1 = {y1[1]:.8f}\ \mathrm{{m/s}},\qquad x_{{anal}}(\Delta t) = {A * np.cos(w * h):.10f}\ \mathrm{{m}}"]),
            ("Error global de la trayectoria", "Tras 2000 pasos se compara cada $x_k$ con $A\\cos\\omega t_k$ y se toma el mayor desvío. RK4 tiene error local $O(\\Delta t^5)$, por eso resulta casi nulo.",
             [rf"\max_k |x_k - A\cos\omega t_k| = {sci(err, 3)}\ \mathrm{{m}}"]),
            ("Medición del periodo", "Se integran 2 s con 40000 pasos y se buscan dos cruces descendentes por cero, interpolando linealmente entre las muestras que cambian de signo: $t_c = t_{i} - x_i\\,(t_i - t_{i-1})/(x_i - x_{i-1})$. El periodo es la diferencia entre ambos cruces.",
             [rf"T_{{num}} = t_{{c2}} - t_{{c1}} = {tn:.8f}\ \mathrm{{s}},\qquad T_{{anal}} = {T:.8f}\ \mathrm{{s}},\qquad |\Delta T| = {sci(abs(tn - T), 3)}\ \mathrm{{s}}"]),
            ("Péndulo exacto: ecuación y primer paso", "Se integra $\\ddot\\theta = -(g/L)\\sin\\theta$ con RK4 y 20000 pasos hasta $2\\pi\\sqrt{L/g}$ (cota del cuarto de periodo). Primer paso para $\\theta_0 = 60^\\circ = 1.0472$ rad:",
             [rf"\Delta t = {sci(hp, 4)}\ \mathrm{{s}},\qquad \dot\omega_0 = -g\sin\theta_0 = {-c18.G * np.sin(th):.4f}\ \mathrm{{rad/s^2}}",
              rf"\theta_1 = {p1[0]:.8f}\ \mathrm{{rad}},\qquad \omega_1 = {p1[1]:.8f}\ \mathrm{{rad/s}}"]),
            ("Periodo del péndulo", "El primer paso en que $\\theta \\le 0$ marca el cuarto de periodo; se interpola linealmente el cruce y se multiplica por 4.",
             [rf"T_{{num}}(10^\circ) = {tab[0]['numeric']:.6f}\ \mathrm{{s}}\ \ (\text{{AGM: }}{tab[0]['exact']:.6f})",
              rf"T_{{num}}(60^\circ) = {tab[1]['numeric']:.6f}\ \mathrm{{s}}\ \ (\text{{AGM: }}{tab[1]['exact']:.6f}),\qquad |\Delta T| = {sci(abs(tab[1]['numeric'] - tab[1]['exact']), 3)}\ \mathrm{{s}}"]),
        ],
    )


# =============================================================== DrivenResonance
def _driven() -> dict:
    m, k, b, F0 = c18.M_D, c18.K_D, c18.B_D, c18.F0
    w0 = np.sqrt(k / m)
    gam = b / m
    wr = c18.resonance_frequency_analytic()
    Ar = float(c18.driven_amplitude_analytic(wr))
    wd = 9.0
    ws = np.array([6.0, 9.0, 12.0])
    num = c18.resonance_curve_numeric(ws)
    A = lambda w_: float(c18.driven_amplitude_analytic(w_))  # noqa: E731
    t_end = 24.0
    n = int(t_end * 400 / min(1.0, 2 * np.pi / wd) / 2 * 1.0) + 4000
    n = max(n, 20000)
    h = t_end / n
    f = lambda t_, yy: np.array([yy[1], (F0 * np.cos(wd * t_) - b * yy[1] - k * yy[0]) / m])  # noqa: E731
    y0 = np.zeros(2)
    k1 = f(0.0, y0)
    k2 = f(h / 2, y0 + h / 2 * k1)
    k3 = f(h / 2, y0 + h / 2 * k2)
    k4 = f(h, y0 + h * k3)
    y1 = rk4_step(f, 0.0, y0, h)
    win = max(3 * 2 * np.pi / wd, 4.0)
    A0 = F0 / m
    d = lambda w_: (w0**2 - w_**2) ** 2 + (gam * w_) ** 2  # noqa: E731
    return dict(
        explanation=[
            "Cuando una fuerza periódica $F_0\\cos\\omega t$ actúa sobre un oscilador amortiguado, después de un transitorio inicial el sistema oscila con la frecuencia de la fuerza y no con la suya propia. La amplitud de ese movimiento estacionario depende de qué tan cerca esté $\\omega$ de la frecuencia natural $\\omega_0 = \\sqrt{k/m}$.",
            "Si la frecuencia de la fuerza es mucho menor que $\\omega_0$, el sistema sigue lentamente la fuerza con una amplitud cercana a la estática $F_0/k$. Si es mucho mayor, la masa no alcanza a responder y la amplitud cae. Entre ambos extremos hay un máximo: la resonancia.",
            "El amortiguamiento limita el pico. Con coeficiente $\\gamma = b/m$, la amplitud es máxima en $\\omega_r = \\sqrt{\\omega_0^2 - \\gamma^2/2}$, ligeramente menor que $\\omega_0$, y el factor de calidad $Q \\approx \\omega_0/\\gamma$ indica qué tan alto y angosto es el pico. Aquí $Q = 5$, por lo que la amplitud en resonancia es unas cinco veces la estática.",
            "La resonancia es la razón por la que empujar un columpio al ritmo correcto produce grandes amplitudes con poca fuerza, y también por la que una estructura puede dañarse si una vibración externa coincide con una de sus frecuencias propias.",
        ],
        analytic_steps=[
            ("Ecuación de movimiento y solución estacionaria", "La ecuación es $m\\ddot x + b\\dot x + kx = F_0\\cos\\omega t$. Una vez extinguido el transitorio se busca $x = A\\cos(\\omega t - \\delta)$ y al sustituir se obtiene la amplitud $A(\\omega)$.",
             [r"A(\omega) = \dfrac{F_0/m}{\sqrt{(\omega_0^2 - \omega^2)^2 + (\gamma\omega)^2}}"]),
            ("Parámetros", "Se calculan la frecuencia natural, el coeficiente de amortiguamiento y la amplitud por unidad de masa.",
             [rf"\omega_0 = \sqrt{{\dfrac{{{k:.0f}}}{{{m:.0f}}}}} = {w0:.1f}\ \mathrm{{rad/s}},\qquad \gamma = \dfrac{{b}}{{m}} = \dfrac{{{b:.0f}}}{{{m:.0f}}} = {gam:.1f}\ \mathrm{{s^{{-1}}}},\qquad \dfrac{{F_0}}{{m}} = {A0:.1f}\ \mathrm{{m/s^2}}"]),
            ("Condición de máximo", "Maximizar $A$ es lo mismo que minimizar el radicando $D(\\omega) = (\\omega_0^2-\\omega^2)^2 + \\gamma^2\\omega^2$. Se deriva respecto a $\\omega^2$ y se iguala a cero.",
             [r"\dfrac{dD}{d(\omega^2)} = -2(\omega_0^2 - \omega^2) + \gamma^2 = 0 \;\Rightarrow\; \omega_r^2 = \omega_0^2 - \dfrac{\gamma^2}{2}"]),
            ("Frecuencia de resonancia", "Se sustituyen los valores.",
             [rf"\omega_r = \sqrt{{{w0**2:.0f} - \dfrac{{({gam:.1f})^2}}{{2}}}} = \sqrt{{{w0**2 - gam**2 / 2:.1f}}} = {wr:.4f}\ \mathrm{{rad/s}}"]),
            ("Amplitud en resonancia", "Se evalúa $A$ en $\\omega_r$. El denominador se simplifica porque $(\\omega_0^2-\\omega_r^2) = \\gamma^2/2$.",
             [rf"A(\omega_r) = \dfrac{{{A0:.1f}}}{{\sqrt{{({gam**2 / 2:.1f})^2 + ({gam:.1f})^2({wr**2:.1f})}}}} = \dfrac{{{A0:.1f}}}{{{np.sqrt(d(wr)):.5f}}} = {Ar:.5f}\ \mathrm{{m}}"]),
            ("Factor de calidad", "La amplitud estática es $F_0/k$. El cociente con la amplitud de resonancia es aproximadamente $Q = \\omega_0/\\gamma$.",
             [rf"Q \approx \dfrac{{\omega_0}}{{\gamma}} = \dfrac{{{w0:.0f}}}{{{gam:.0f}}} = {w0 / gam:.0f},\qquad \dfrac{{A(\omega_r)}}{{F_0/k}} = \dfrac{{{Ar:.5f}}}{{{F0 / k:.3f}}} = {Ar / (F0 / k):.3f}"]),
            ("Tres frecuencias de comparación", "La verificación numérica mide la amplitud en $\\omega = 6, 9$ y $12$ rad/s; aquí se calculan con la fórmula.",
             [rf"A(6) = \dfrac{{{A0:.1f}}}{{\sqrt{{({w0**2 - 36:.0f})^2 + ({gam * 6:.0f})^2}}}} = {A(6.0):.5f}\ \mathrm{{m}}",
              rf"A(9) = \dfrac{{{A0:.1f}}}{{\sqrt{{({w0**2 - 81:.0f})^2 + ({gam * 9:.0f})^2}}}} = {A(9.0):.5f}\ \mathrm{{m}}",
              rf"A(12) = \dfrac{{{A0:.1f}}}{{\sqrt{{({w0**2 - 144:.0f})^2 + ({gam * 12:.0f})^2}}}} = {A(12.0):.5f}\ \mathrm{{m}}"]),
        ],
        numeric_steps=[
            ("Sistema de primer orden", "Con $y = (x, v)$ la ecuación queda $y' = (v,\\ [F_0\\cos\\omega t - bv - kx]/m)$. Se parte del reposo en el origen y se integra hasta $t = 24$ s, tiempo suficiente para que el transitorio $e^{-\\gamma t/2} = e^{-t}$ desaparezca.",
             [rf"y_0 = (0,\,0),\qquad e^{{-\gamma t/2}}\big|_{{t=24}} = {sci(np.exp(-gam * 24 / 2), 3)}"]),
            ("Número de pasos", "El código usa al menos 20000 pasos y más si $\\omega$ es grande. Para $\\omega = 9$ rad/s resulta:",
             [rf"N = {n},\qquad \Delta t = \dfrac{{24}}{{{n}}} = {sci(h, 4)}\ \mathrm{{s}}"]),
            ("Pendientes del primer paso ($\\omega = 9$ rad/s)", "El término de fuerza cambia con el tiempo, así que $k_2$ y $k_3$ se evalúan en $t=\\Delta t/2$ y $k_4$ en $t=\\Delta t$.",
             [rf"k_1 = ({k1[0]:.6f},\ {k1[1]:.6f}),\qquad k_2 = ({k2[0]:.6f},\ {k2[1]:.6f})",
              rf"k_3 = ({k3[0]:.6f},\ {k3[1]:.6f}),\qquad k_4 = ({k4[0]:.6f},\ {k4[1]:.6f})"]),
            ("Primera actualización", "Se combinan con los pesos $1,2,2,1$ y el paso $\\Delta t/6$. Al arrancar del reposo, la fuerza acelera la masa y aparece una pequeña velocidad.",
             [rf"y_1 = y_0 + \dfrac{{\Delta t}}{{6}}(k_1 + 2k_2 + 2k_3 + k_4) = ({sci(y1[0], 5)},\ {sci(y1[1], 5)})"]),
            ("Régimen estacionario y medición", "Se repite el paso hasta 24 s. Después se toma el máximo de $|x|$ en la ventana final, de duración el mayor entre tres periodos de la fuerza y 4 s. Con esa ventana se mide la amplitud con la que se queda el oscilador.",
             [rf"\text{{ventana}}:\ t > 24 - {win:.3f} = {24 - win:.3f}\ \mathrm{{s}}\quad(\max\{{3T_f,\,4\}} = {win:.3f}\ \mathrm{{s}})"]),
            ("Resultados para las tres frecuencias", "Se repite todo el proceso para $\\omega = 6, 9, 12$ rad/s y se compara con la fórmula analítica.",
             [r"\begin{array}{c|c|c|c} \omega\ [\mathrm{rad/s}] & A_{anal}\ [\mathrm{m}] & A_{num}\ [\mathrm{m}] & |\Delta A|\ [\mathrm{m}]\\ \hline " + r"\\ ".join(
                 f"{w_:.0f} & {A(w_):.6f} & {a_:.6f} & {sci(abs(a_ - A(w_)), 2)}" for w_, a_ in zip(ws, num)) + r"\end{array}"]),
            ("Origen del error residual", "La ventana final cubre varios ciclos pero el transitorio no es exactamente cero, y la amplitud medida es un máximo de muestras discretas. Ambos efectos son mucho menores que la amplitud, por eso el acuerdo es de varias cifras significativas.",
             [rf"e^{{-t}}\big|_{{t=24}} = {sci(np.exp(-24.0), 3)}\ \ll\ A"]),
        ],
    )


# =============================================================== WavePulse
def _pulse() -> dict:
    T, mu, v = c19.TENSION, c19.MU, c19.V
    sg, L = c19.SIGMA, c19.LENGTH
    n = 800
    dx = L / n
    t_end = 0.15
    out = {}
    for cr in (1.0, 0.9, 0.5):
        out[cr] = (cr * dx / v, c19.fdtd_error(cr))
    dt = out[1.0][0]
    c2 = 1.0
    xi = 10.0
    u = lambda xx: float(c19.pulse(xx))  # noqa: E731
    i0 = 400
    xs = np.array([xi - dx, xi, xi + dx])
    y0v = [u(xx) for xx in xs]
    y1c = y0v[1] + 0.5 * c2 * (y0v[2] - 2 * y0v[1] + y0v[0])
    # un punto del flanco, x = 10.6 (indice 424)
    ip = 424
    xp = ip * dx
    yp = [u(xp - dx), u(xp), u(xp + dx)]
    y1p = yp[1] + 0.5 * (yp[2] - 2 * yp[1] + yp[0])
    # segundo paso en x = xp con Courant 1: necesita y1 en vecinos
    y1_at = lambda xx: u(xx) + 0.5 * (u(xx + dx) - 2 * u(xx) + u(xx - dx))  # noqa: E731
    y2p = 2 * y1p - yp[1] + (y1_at(xp + dx) - 2 * y1p + y1_at(xp - dx))
    y2d = float(c19.dalembert(xp, 2 * dt))
    steps1 = int(round(t_end / dt))
    steps9 = int(round(t_end / out[0.9][0]))
    return dict(
        explanation=[
            "Una onda transversal en una cuerda tensa es una perturbación que se propaga sin que el material de la cuerda viaje con ella. Cada elemento de cuerda solo sube y baja; lo que avanza es la forma. La razón es que la tensión actúa como fuerza restauradora y la masa por unidad de longitud como inercia.",
            "Aplicando la segunda ley de Newton a un trocito de cuerda con pequeña pendiente se obtiene la ecuación de onda, $\\partial_t^2 y = v^2\\partial_x^2 y$, con $v = \\sqrt{T/\\mu}$. Una cuerda más tensa o más ligera transporta las ondas más rápido.",
            "La solución general de d'Alembert, $y = f(x - vt) + g(x + vt)$, dice que cualquier forma que viaja a velocidad $v$ en un sentido o en otro satisface la ecuación. Un pulso inicial en reposo se divide en dos mitades idénticas, de la mitad de la amplitud, que se alejan en sentidos opuestos.",
            "En este caso $v = 40$ m/s, de modo que en 0.15 s cada mitad recorre 6 m. Como el ancho del pulso es solo $\\sigma = 0.6$ m, las dos mitades ya están completamente separadas, una en $x = 4$ m y otra en $x = 16$ m. La forma no cambia porque esta cuerda ideal no es dispersiva.",
        ],
        analytic_steps=[
            ("Velocidad de la onda", "Se sustituyen la tensión y la densidad lineal en $v = \\sqrt{T/\\mu}$.",
             [rf"v = \sqrt{{\dfrac{{T}}{{\mu}}}} = \sqrt{{\dfrac{{{T:.0f}\ \mathrm{{N}}}}{{{mu}\ \mathrm{{kg/m}}}}}} = \sqrt{{{T / mu:.0f}}} = {v:.1f}\ \mathrm{{m/s}}"]),
            ("Condiciones iniciales", "El pulso es una gaussiana centrada en $x_0 = 10$ m con $\\sigma = 0.6$ m y la cuerda parte del reposo, $\\partial_t y(x,0) = 0$.",
             [r"y(x,0) = f_0(x) = \exp\!\left[-\dfrac{(x-x_0)^2}{2\sigma^2}\right],\qquad \partial_t y(x,0) = 0"]),
            ("Aplicar d'Alembert", "Con $y = f(x-vt) + g(x+vt)$, la condición de velocidad nula obliga a $g = f$ y la de posición da $f + g = f_0$. Por tanto $f = g = f_0/2$.",
             [r"y(x,t) = \tfrac12\left[f_0(x - vt) + f_0(x + vt)\right]"]),
            ("Posición de los dos pulsos", "A $t = 0.15$ s cada mitad se ha desplazado $vt$ hacia un lado.",
             [rf"vt = ({v:.1f})({t_end}) = {v * t_end:.1f}\ \mathrm{{m}}\;\Rightarrow\; x_\pm = 10 \pm {v * t_end:.1f} = {10 - v * t_end:.1f}\ \mathrm{{m}},\ {10 + v * t_end:.1f}\ \mathrm{{m}}"]),
            ("Amplitud y forma a 0.15 s", "Cada mitad tiene altura $1/2$ y la misma anchura $\\sigma$. En su centro y a un $\\sigma$ de distancia:",
             [rf"y(4,\,0.15) = \tfrac12\left[e^{{-(4-4)^2/(2\cdot{sg}^2)}} + e^{{-(4-16)^2/(2\cdot{sg}^2)}}\right] = {float(c19.dalembert(4.0, t_end)):.4f}\ \mathrm{{m}}",
              rf"y(4.6,\,0.15) = \tfrac12\,e^{{-(0.6)^2/(2\cdot{sg}^2)}} = \tfrac12 e^{{-1/2}} = {float(c19.dalembert(4.6, t_end)):.4f}\ \mathrm{{m}}"]),
            ("Entre los pulsos", "En el centro original, $x = 10$ m, la distancia a ambos pulsos es de 6 m, es decir $10\\sigma$, y la contribución es prácticamente cero.",
             [rf"y(10,\,0.15) = \tfrac12\cdot 2\,e^{{-36/(2\cdot{sg}^2)}} = e^{{-50}} = {sci(float(c19.dalembert(10.0, t_end)), 3)}\ \mathrm{{m}}"]),
        ],
        numeric_steps=[
            ("Discretizar espacio y tiempo", "La cuerda de 20 m se divide en $N = 800$ intervalos. El paso de tiempo se fija con el número de Courant $C = v\\Delta t/\\Delta x$.",
             [rf"\Delta x = \dfrac{{{L:.0f}}}{{{n}}} = {dx} \ \mathrm{{m}},\qquad \Delta t = \dfrac{{C\,\Delta x}}{{v}} = {sci(dt, 4)}\ \mathrm{{s}}\ \ (C=1)"]),
            ("Esquema leapfrog", "Se reemplazan las segundas derivadas por diferencias centradas y se despeja el valor futuro. Con $C=1$ el esquema coincide con la solución exacta en la malla.",
             [r"y_i^{k+1} = 2y_i^k - y_i^{k-1} + C^2\left(y_{i+1}^k - 2y_i^k + y_{i-1}^k\right)"]),
            ("Primer paso (velocidad inicial nula)", "No existe $y^{-1}$, así que el primer paso usa un desarrollo de Taylor con $\\partial_t y = 0$: $y^1 = y^0 + \\tfrac12 C^2(y_{i+1}^0 - 2y_i^0 + y_{i-1}^0)$. Para el punto central $x=10$ m:",
             [rf"y^0_{{i-1}} = {y0v[0]:.8f},\quad y^0_i = {y0v[1]:.1f},\quad y^0_{{i+1}} = {y0v[2]:.8f}",
              rf"y^1_i = {y0v[1]:.1f} + \tfrac12\,({y0v[2]:.8f} - 2({y0v[1]:.1f}) + {y0v[0]:.8f}) = {y1c:.8f}"]),
            ("Un punto en el flanco del pulso, dos pasos", "Para $x = 10.6$ m (un $\\sigma$ a la derecha del centro) se aplica el primer paso y luego leapfrog. El segundo paso usa los valores $y^1$ de los vecinos, calculados de la misma forma.",
             [rf"y^0 = {yp[1]:.8f},\qquad y^1 = {y1p:.8f},\qquad y^2 = 2y^1 - y^0 + (y^1_{{i+1}} - 2y^1_i + y^1_{{i-1}}) = {y2p:.8f}",
              rf"\text{{d'Alembert en }}t=2\Delta t:\ \ y = {y2d:.8f}"]),
            ("Número de pasos", "Se avanza hasta $t = 0.15$ s. Para cada $C$ cambia $\\Delta t$ y por tanto el número de pasos.",
             [rf"C=1:\ \ \Delta t = {sci(out[1.0][0], 4)}\ \mathrm{{s}},\ \ {steps1}\ \text{{pasos}};\qquad C=0.9:\ \ \Delta t = {sci(out[0.9][0], 4)}\ \mathrm{{s}},\ \ {steps9}\ \text{{pasos}}"]),
            ("Error máximo frente a d'Alembert", "En cada malla se calcula $\\varepsilon = \\max_i |y_i - y_{dAl}(x_i, t)|$.",
             [r"\begin{array}{c|c} C & \varepsilon\ [\mathrm{m}]\\ \hline " + r"\\ ".join(f"{cr} & {sci(e, 3) if e > 0 else '0'}" for cr, (_, e) in out.items()) + r"\end{array}"]),
            ("Interpretación", "Con $C=1$ la información viaja exactamente una celda por paso y el error es de redondeo. Con $C<1$ el esquema es estable pero dispersivo: las componentes de longitud de onda corta viajan algo más lento que $v$, lo que deforma el pulso. Es lo que mide el error de $C=0.9$.",
             [rf"\varepsilon(C=1) = {sci(out[1.0][1], 3)}\ \mathrm{{m}}\ \ll\ \varepsilon(C=0.9) = {sci(out[0.9][1], 3)}\ \mathrm{{m}}"]),
        ],
    )


# =============================================================== StandingModes
def _modes() -> dict:
    L, v = c19.STRING_L, c19.V
    freqs, xs, shapes = c19.modes_numeric()
    N = 3
    dxs = L / (N + 1)
    a3 = (np.diag(np.full(N, 2.0)) + np.diag(np.full(N - 1, -1.0), 1) + np.diag(np.full(N - 1, -1.0), -1)) / dxs**2
    w2 = np.linalg.eigvalsh(a3)
    f3 = v * np.sqrt(w2) / (2 * np.pi)
    vec = np.linalg.eigh(a3)[1]
    s1 = vec[:, 0]
    s1 = s1 / np.max(np.abs(s1)) * np.sign(s1[np.argmax(np.abs(s1))])
    dx = L / 401
    e1 = float(np.max(np.abs(shapes[0] - np.sin(np.pi * xs / L))))
    fa = lambda n_: c19.mode_frequency_analytic(n_)  # noqa: E731
    return dict(
        explanation=[
            "Cuando una cuerda está fija en ambos extremos, las ondas que viajan en un sentido se reflejan y se superponen con las del sentido contrario. Solo ciertas formas sobreviven: aquellas en las que las ondas reflejadas refuerzan a las incidentes. Son los modos normales, y cada punto de la cuerda oscila con la misma frecuencia y fase, aunque con distinta amplitud.",
            "Que los extremos estén fijos impone $y(0) = y(L) = 0$. Eso exige que quepa un número entero de medias longitudes de onda en la cuerda, $L = n\\lambda/2$. Con $\\lambda f = v$ se obtienen las frecuencias $f_n = nv/2L$, múltiplos enteros de la fundamental.",
            "Los puntos que no se mueven son los nodos, y los de máxima oscilación son los vientres. El modo $n$ tiene $n-1$ nodos interiores. Una cuerda real vibra con una mezcla de estos modos, y el timbre de un instrumento depende de cuánta amplitud tiene cada uno.",
            "Matemáticamente los modos son los autovectores del operador segunda derivada con condiciones de frontera, y $\\omega_n^2$ son sus autovalores. Eso permite calcularlos numéricamente como un problema de álgebra lineal, sin conocer la solución analítica.",
        ],
        analytic_steps=[
            ("Separar variables", "Se busca $y(x,t) = X(x)\\,\\cos\\omega t$. Sustituyendo en la ecuación de onda queda una ecuación para $X$.",
             [r"-\omega^2 X = v^2 X'' \;\Rightarrow\; X'' + k^2 X = 0,\qquad k = \dfrac{\omega}{v}"]),
            ("Condiciones de contorno", "La solución general es $X = a\\sin kx + b\\cos kx$. La condición $X(0)=0$ elimina el coseno y $X(L) = 0$ restringe $k$.",
             [r"X(L) = a\sin kL = 0 \;\Rightarrow\; k_n L = n\pi,\quad k_n = \dfrac{n\pi}{L},\quad n = 1, 2, 3,\dots"]),
            ("Frecuencias", "De $\\omega_n = v k_n$ y $f = \\omega/2\\pi$ resulta la fórmula de los armónicos.",
             [r"f_n = \dfrac{\omega_n}{2\pi} = \dfrac{n v}{2L}"]),
            ("Sustituir valores", "Con $v = 40$ m/s y $L = 1.2$ m, la fundamental es $v/2L$ y los demás son múltiplos.",
             [rf"f_1 = \dfrac{{40}}{{2(1.2)}} = {fa(1):.4f}\ \mathrm{{Hz}},\qquad f_2 = {fa(2):.4f}\ \mathrm{{Hz}},\qquad f_3 = {fa(3):.4f}\ \mathrm{{Hz}},\qquad f_5 = {fa(5):.4f}\ \mathrm{{Hz}}"]),
            ("Longitud de onda y nodos", "Para cada modo $\\lambda_n = 2L/n$. El modo 1 tiene sus extremos como únicos nodos; el modo 2 añade uno en el centro.",
             [rf"\lambda_1 = {2 * L:.1f}\ \mathrm{{m}},\qquad \lambda_2 = {L:.1f}\ \mathrm{{m}},\qquad \lambda_3 = {2 * L / 3:.1f}\ \mathrm{{m}}",
              rf"\text{{nodos del modo 3 en }}x = {L / 3:.1f},\ {2 * L / 3:.1f}\ \mathrm{{m}}"]),
            ("Forma del modo", "La forma espacial es una semionda senoidal para el modo 1. Es lo que se comparará con el vector propio.",
             [rf"X_1(x) = \sin\dfrac{{\pi x}}{{{L}}},\qquad X_1(L/2) = 1"]),
        ],
        numeric_steps=[
            ("Discretizar la cuerda", "Se toman $N$ puntos interiores, equiespaciados $\\Delta x = L/(N+1)$. Los extremos fijos no son incógnitas. Para mostrarlo a mano se usa $N=3$.",
             [rf"N = 3:\quad \Delta x = \dfrac{{{L}}}{{4}} = {dxs:.1f}\ \mathrm{{m}},\qquad x_j = {dxs:.1f},\ {2 * dxs:.1f},\ {3 * dxs:.1f}\ \mathrm{{m}}"]),
            ("Segunda derivada como matriz", "La derivada discreta es $X'' \\approx (X_{j+1} - 2X_j + X_{j-1})/\\Delta x^2$. Con $X_0 = X_{N+1} = 0$ el operador es una matriz tridiagonal $\\mathbf{A}$ y la ecuación es $\\mathbf{A}X = k^2 X$.",
             [rf"\mathbf{{A}} = \dfrac{{1}}{{\Delta x^2}}\begin{{pmatrix}} 2 & -1 & 0\\ -1 & 2 & -1\\ 0 & -1 & 2\end{{pmatrix}},\qquad \dfrac{{1}}{{\Delta x^2}} = {1 / dxs**2:.4f}\ \mathrm{{m^{{-2}}}}"]),
            ("Autovalores de la versión $3\\times3$", "Los autovalores de la matriz $(2,-1)$ son $2 - 2\\cos(j\\pi/4)$, es decir $2-\\sqrt2,\\ 2,\\ 2+\\sqrt2$. Se multiplican por $1/\\Delta x^2$.",
             [rf"k_j^2 = {w2[0]:.4f},\ {w2[1]:.4f},\ {w2[2]:.4f}\ \mathrm{{m^{{-2}}}}"]),
            ("Pasar a frecuencia", "De $k^2$ se obtiene $f = v\\sqrt{k^2}/2\\pi$. Con solo 3 puntos ya se acerca a la solución exacta, pero con error apreciable.",
             [rf"f_j = \dfrac{{40\sqrt{{k_j^2}}}}{{2\pi}} = {f3[0]:.3f},\ {f3[1]:.3f},\ {f3[2]:.3f}\ \mathrm{{Hz}}",
              rf"\text{{exacto:}}\ {fa(1):.3f},\ {fa(2):.3f},\ {fa(3):.3f}\ \mathrm{{Hz}}\quad(\text{{error del modo 1: }}{(f3[0] / fa(1) - 1) * 100:.1f}\,\%)"]),
            ("Autovector del modo 1", "El vector propio, normalizado a máximo 1, es la forma del modo en los puntos de la malla. Debe ser $\\sin(\\pi x/L)$ evaluado en ellos.",
             [rf"X^{{(1)}} = ({s1[0]:.4f},\ {s1[1]:.4f},\ {s1[2]:.4f}),\qquad \sin(\pi x_j/L) = ({np.sin(np.pi * dxs / L):.4f},\ {np.sin(2 * np.pi * dxs / L):.4f},\ {np.sin(3 * np.pi * dxs / L):.4f})"]),
            ("Lo que hace el código a escala", "El código repite el mismo cálculo con $N = 400$ puntos, así que $\\Delta x = 1.2/401 = %s$ m. La matriz es de $400\\times400$, se diagonalizan con `eigh` y se toman los cinco menores autovalores." % sci(dx, 4),
             [r"\begin{array}{c|c|c|c} n & f_n^{anal}\ [\mathrm{Hz}] & f_n^{num}\ [\mathrm{Hz}] & |\Delta f|\ [\mathrm{Hz}]\\ \hline " + r"\\ ".join(
                 f"{n_} & {fa(n_):.5f} & {float(freqs[n_ - 1]):.5f} & {sci(abs(float(freqs[n_ - 1]) - fa(n_)), 2)}" for n_ in (1, 2, 3, 5)) + r"\end{array}"]),
            ("Error y forma", "El error de la discretización es $O(\\Delta x^2)$ y crece con el número de modo, porque los modos altos oscilan más rápido sobre la misma malla. La forma numérica del modo 1 coincide con la senoide.",
             [rf"\max_j\,\left|X^{{(1)}}_j - \sin\dfrac{{\pi x_j}}{{L}}\right| = {sci(e1, 3)}"]),
        ],
    )


# =============================================================== Doppler
def _doppler() -> dict:
    c, f0, vs = c20.C, c20.F0, c20.VS
    fa, fb = c20.doppler_analytic(), c20.doppler_analytic(vs=-vs)
    fo = c20.doppler_analytic(vs=0.0, vo=30.0)
    na, nb, no = c20.doppler_numeric(), c20.doppler_numeric(vs=-vs), c20.doppler_numeric(vs=0.0, vo=30.0)
    te = np.arange(4) / f0
    xs = 1000.0 - vs * te
    ta = (xs + c * te) / c
    return dict(
        explanation=[
            "Cuando una fuente de sonido se mueve respecto del aire, el observador oye una frecuencia distinta de la emitida. No cambia el tono de la fuente, cambia la separación entre los frentes de onda: si la fuente se acerca, cada frente se emite un poco más cerca del anterior y las crestas llegan más seguidas.",
            "El sonido viaja a velocidad $c$ respecto del aire, independientemente de cómo se mueva la fuente. Entre dos emisiones sucesivas, separadas por $1/f_0$, la fuente avanza $v_s/f_0$, por lo que la longitud de onda delante de ella se acorta a $(c - v_s)/f_0$ y la frecuencia percibida crece a $f' = f_0\\,c/(c - v_s)$.",
            "Si la fuente se aleja, $v_s$ cambia de signo y se oye un tono más grave. El observador en movimiento también modifica la frecuencia, pero por otro mecanismo: encuentra los frentes con más o menos frecuencia, y el numerador pasa a ser $c + v_o$. Ambos efectos no son simétricos, porque el medio (el aire) define un sistema de referencia privilegiado para el sonido.",
            "Con un avión a 120 m/s, es decir alrededor del 35 % de la velocidad del sonido, el cambio es muy notorio: el tono sube casi una cuarta al acercarse y baja de manera comparable al alejarse. La fórmula deja de valer si $v_s \\ge c$, donde aparece el estampido sónico.",
        ],
        analytic_steps=[
            ("Longitud de onda delante de la fuente", "En el tiempo $T_0 = 1/f_0$ entre dos emisiones, el frente anterior avanza $cT_0$ y la fuente $v_sT_0$. Esa es la distancia entre crestas consecutivas delante de ella.",
             [rf"T_0 = \dfrac{{1}}{{{f0:.0f}}} = {sci(1 / f0, 4)}\ \mathrm{{s}},\qquad \lambda' = (c - v_s)T_0 = ({c:.0f} - {vs:.0f})({sci(1 / f0, 4)}) = {(c - vs) / f0:.5f}\ \mathrm{{m}}"]),
            ("Frecuencia que oye el observador", "El observador en reposo percibe la frecuencia con que le llegan las crestas, que viajan a $c$ separadas por $\\lambda'$.",
             [rf"f' = \dfrac{{c}}{{\lambda'}} = f_0\,\dfrac{{c}}{{c - v_s}} = {f0:.0f}\cdot\dfrac{{{c:.0f}}}{{{c:.0f} - {vs:.0f}}} = {fa:.2f}\ \mathrm{{Hz}}"]),
            ("Fuente que se aleja", "Detrás del avión el desplazamiento de la fuente alarga la distancia entre crestas: se cambia el signo de $v_s$.",
             [rf"f' = f_0\,\dfrac{{c}}{{c + v_s}} = {f0:.0f}\cdot\dfrac{{{c:.0f}}}{{{c:.0f} + {vs:.0f}}} = {fb:.2f}\ \mathrm{{Hz}}"]),
            ("Cambio relativo", "Se comparan ambas frecuencias con la emitida.",
             [rf"\dfrac{{f'_{{delante}}}}{{f_0}} = {fa / f0:.4f},\qquad \dfrac{{f'_{{detrás}}}}{{f_0}} = {fb / f0:.4f}"]),
            ("Observador que se acerca (fuente en reposo)", "Si es el observador quien se mueve hacia la fuente a $v_o = 30$ m/s, la velocidad relativa del sonido respecto de él es $c + v_o$ pero la longitud de onda no cambia.",
             [rf"f' = f_0\,\dfrac{{c + v_o}}{{c}} = {f0:.0f}\cdot\dfrac{{{c:.0f} + 30}}{{{c:.0f}}} = {fo:.2f}\ \mathrm{{Hz}}"]),
            ("Fórmula general", "Las dos situaciones se combinan en una sola expresión, con $v_s$ y $v_o$ positivas cuando fuente y observador se acercan.",
             [r"f' = f_0\,\dfrac{c + v_o}{c - v_s}"]),
        ],
        numeric_steps=[
            ("Montar el experimento", "El observador está en $x = 0$ y la fuente parte de $x_0 = 1000$ m moviéndose hacia él a $v_s$. Se emiten $400$ pulsos, uno cada $1/f_0$.",
             [rf"t_{{e,k}} = \dfrac{{k}}{{f_0}} = k\,({sci(1 / f0, 5)})\ \mathrm{{s}},\qquad x_{{s,k}} = 1000 - v_s\,t_{{e,k}},\qquad k = 0,\dots,399"]),
            ("Tiempo de llegada de un pulso", "El pulso sale de $x_{s,k}$ en $t_{e,k}$ y viaja a $c$. Si el observador se mueve con $x_o = v_ot$, llega cuando $x_{s,k} - c(t - t_{e,k}) = v_ot$.",
             [r"t_{a,k} = \dfrac{x_{s,k} + c\,t_{e,k}}{c + v_o}"]),
            ("Primeros pulsos (fuente que se acerca, $v_o=0$)", "Se calculan los primeros cuatro instantes de emisión, posición de la fuente y llegada.",
             [r"\begin{array}{c|c|c|c} k & t_e\ [\mathrm{s}] & x_s\ [\mathrm{m}] & t_a\ [\mathrm{s}]\\ \hline " + r"\\ ".join(
                 f"{k_} & {te[k_]:.6f} & {xs[k_]:.5f} & {ta[k_]:.6f}" for k_ in range(4)) + r"\end{array}"]),
            ("Intervalo entre llegadas", "Cada pulso llega antes de lo que le tocaría porque salió más cerca. El intervalo constante entre llegadas es menor que $1/f_0$.",
             [rf"\Delta t_a = t_{{a,1}} - t_{{a,0}} = {ta[1] - ta[0]:.8f}\ \mathrm{{s}} = \dfrac{{c - v_s}}{{c}}\cdot\dfrac{{1}}{{f_0}}"]),
            ("Frecuencia como inverso del intervalo medio", "El código promedia los 399 intervalos y toma el inverso. Como son todos iguales, el promedio coincide con $\\Delta t_a$.",
             [rf"f'_{{num}} = \dfrac{{1}}{{\overline{{\Delta t_a}}}} = \dfrac{{1}}{{{(1 / na):.8f}}} = {na:.4f}\ \mathrm{{Hz}}"]),
            ("Fuente que se aleja y observador que se acerca", "Se repite con $v_s = -120$ m/s (la fuente se aleja) y con $v_s=0$, $v_o = 30$ m/s.",
             [rf"f'_{{detrás}} = {nb:.4f}\ \mathrm{{Hz}}\ \ (\text{{anal. }}{fb:.4f}),\qquad f'_{{obs}} = {no:.4f}\ \mathrm{{Hz}}\ \ (\text{{anal. }}{fo:.4f})"]),
            ("Error", "La diferencia es de redondeo de punto flotante, porque el método reproduce de forma exacta la geometría del problema y no aproxima derivadas.",
             [rf"|f'_{{num}} - f'_{{anal}}| = {sci(abs(na - fa), 2)},\ {sci(abs(nb - fb), 2)},\ {sci(abs(no - fo), 2)}\ \mathrm{{Hz}}"]),
        ],
    )


# =============================================================== BeatsAndPipes
def _beats() -> dict:
    c, Lp = c20.C, c20.LEN
    f1, f2 = c20.F1, c20.F2
    fb = c20.beat_frequency_analytic()
    fs, dur = 16000.0, 4.0
    t, s = c20.beat_signal(fs, dur)
    spec = np.abs(np.fft.rfft(s * np.hanning(len(s))))
    fr = np.fft.rfftfreq(len(s), 1 / fs)
    pk = [i for i in range(1, len(spec) - 1) if spec[i] > spec[i - 1] and spec[i] >= spec[i + 1] and spec[i] > 0.2 * spec.max()]
    i1 = pk[0]
    la, lb, lc = np.log(spec[i1 - 1]), np.log(spec[i1]), np.log(spec[i1 + 1])
    dlt = 0.5 * (la - lc) / (la - 2 * lb + lc)
    fpk = [fr[i] + 0.5 * (np.log(spec[i - 1]) - np.log(spec[i + 1])) / (np.log(spec[i - 1]) - 2 * np.log(spec[i]) + np.log(spec[i + 1])) * (fr[1] - fr[0]) for i in pk]
    fbn = c20.beat_frequency_numeric()
    fbe = c20.beat_envelope_frequency_numeric()
    t2, s2 = c20.beat_signal(fs, 2.0)
    env = c20.envelope(s2)
    pe = [i for i in range(1, len(env) - 1) if env[i] > env[i - 1] and env[i] >= env[i + 1] and env[i] > 1.8]
    op_a, cl_a = c20.pipe_frequencies_analytic("open"), c20.pipe_frequencies_analytic("closed")
    op_n, cl_n = c20.pipe_frequencies_numeric("open"), c20.pipe_frequencies_numeric("closed")

    def small(kind, n=3):
        dx = Lp / (n + 0.5 if kind == "closed" else n + 1)
        a = (np.diag(np.full(n, 2.0)) + np.diag(np.full(n - 1, -1.0), 1) + np.diag(np.full(n - 1, -1.0), -1)) / dx**2
        if kind == "closed":
            a[0, 0] = 1.0 / dx**2
        return dx, c * np.sqrt(np.linalg.eigvalsh(a)) / (2 * np.pi)

    dxo, fo3 = small("open")
    dxc, fc3 = small("closed")
    return dict(
        explanation=[
            "Cuando dos ondas sonoras de frecuencias muy próximas llegan juntas al oído, se suman y la amplitud total sube y baja periódicamente: son las pulsaciones. A ratos las crestas coinciden y el sonido es fuerte; a ratos una cresta cae sobre un valle de la otra y el sonido casi desaparece. Los afinadores de instrumentos usan este fenómeno para igualar dos notas.",
            "Matemáticamente, la suma de dos senos es un tono de frecuencia promedio $(f_1+f_2)/2$ multiplicado por una envolvente lenta de frecuencia $(f_2-f_1)/2$. Como el oído percibe tanto los máximos positivos como los negativos de la envolvente, el ritmo percibido es el doble, $|f_2 - f_1|$.",
            "En un tubo de aire, el sonido se refleja en los extremos. En un extremo abierto la presión es casi la atmosférica (un nodo de presión); en un extremo cerrado el aire no se mueve y la presión varía al máximo (un antinodo de presión). Solo caben ondas estacionarias compatibles con ambas condiciones.",
            "Un tubo abierto en ambos extremos resuena en $f_n = nc/2L$ con todos los armónicos, mientras que uno cerrado en un extremo cabe un cuarto de longitud de onda y solo tiene armónicos impares, $f_n = (2n-1)c/4L$. Por eso, a igual longitud, el cerrado suena una octava más grave y con timbre distinto.",
        ],
        analytic_steps=[
            ("Suma de los dos tonos", "Se suman dos ondas de la misma amplitud y se usa la identidad $\\sin a + \\sin b = 2\\sin\\frac{a+b}{2}\\cos\\frac{a-b}{2}$.",
             [r"y = \sin 2\pi f_1 t + \sin 2\pi f_2 t = 2\cos\!\big(\pi (f_2 - f_1)\,t\big)\,\sin\!\big(\pi (f_1 + f_2)\,t\big)"]),
            ("Tono medio y envolvente", "El factor seno es la portadora, y el coseno modula la amplitud lentamente.",
             [rf"f_{{medio}} = \dfrac{{f_1 + f_2}}{{2}} = \dfrac{{{f1:.0f} + {f2:.0f}}}{{2}} = {(f1 + f2) / 2:.0f}\ \mathrm{{Hz}},\qquad f_{{env}} = \dfrac{{f_2 - f_1}}{{2}} = {(f2 - f1) / 2:.0f}\ \mathrm{{Hz}}"]),
            ("Frecuencia de pulsación", "La amplitud $|\\cos(\\pi\\Delta f\\,t)|$ alcanza un máximo dos veces por ciclo del coseno, así que los máximos se repiten con frecuencia $\\Delta f$.",
             [rf"f_{{puls}} = |f_2 - f_1| = |{f2:.0f} - {f1:.0f}| = {fb:.0f}\ \mathrm{{Hz}},\qquad T_{{puls}} = \dfrac{{1}}{{{fb:.0f}}} = {1 / fb:.2f}\ \mathrm{{s}}"]),
            ("Tubo abierto: condición", "Presión nula (nodo) en ambos extremos, igual que una cuerda fija: caben $n$ medias longitudes de onda.",
             [r"L = \dfrac{n\lambda}{2} \;\Rightarrow\; f_n = \dfrac{c}{\lambda} = \dfrac{n\,c}{2L}"]),
            ("Tubo abierto: números", "Con $c = 343$ m/s y $L = 0.75$ m se obtienen los primeros cuatro armónicos.",
             [rf"f_1 = \dfrac{{{c:.0f}}}{{2({Lp})}} = {op_a[0]:.3f}\ \mathrm{{Hz}},\quad f_2 = {op_a[1]:.3f},\quad f_3 = {op_a[2]:.3f},\quad f_4 = {op_a[3]:.3f}\ \mathrm{{Hz}}"]),
            ("Tubo cerrado: condición", "Un extremo es antinodo y el otro nodo de presión: cabe un número impar de cuartos de onda.",
             [r"L = \dfrac{(2n-1)\lambda}{4} \;\Rightarrow\; f_n = \dfrac{(2n-1)\,c}{4L}"]),
            ("Tubo cerrado: números", "Se calculan con los mismos datos.",
             [rf"f_1 = \dfrac{{{c:.0f}}}{{4({Lp})}} = {cl_a[0]:.3f}\ \mathrm{{Hz}},\quad f_2 = {cl_a[1]:.3f},\quad f_3 = {cl_a[2]:.3f},\quad f_4 = {cl_a[3]:.3f}\ \mathrm{{Hz}}"]),
        ],
        numeric_steps=[
            ("Muestreo de la señal", "Se genera $s(t) = \\sin 2\\pi f_1t + \\sin 2\\pi f_2 t$ con frecuencia de muestreo $16000$ Hz. La resolución en frecuencia de una FFT es el inverso de la duración, por eso se usan 4 s para resolver los dos picos.",
             [rf"N = {len(t)},\qquad \Delta f_{{bin}} = \dfrac{{1}}{{T}} = \dfrac{{1}}{{{dur:.0f}\ \mathrm{{s}}}} = {1 / dur}\ \mathrm{{Hz}}"]),
            ("Ventana de Hann y FFT", "Se multiplica por $w_j = 0.5[1 - \\cos(2\\pi j/(N-1))]$ para reducir las fugas espectrales y se calcula la transformada $S_m = \\sum_j s_jw_j e^{-2\\pi i jm/N}$. Con solo dos tonos el espectro tiene dos máximos, que el código detecta como puntos mayores que sus vecinos y mayores que el 20 % del máximo global.",
             [rf"\text{{picos en los bines }}{pk[0]},\ {pk[-1]}:\quad f = {fr[pk[0]]:.2f},\ {fr[pk[-1]]:.2f}\ \mathrm{{Hz}}"]),
            ("Interpolación parabólica del primer pico", "El pico verdadero cae entre bines. Se ajusta una parábola a $\\ln|S|$ en tres puntos $(\\alpha,\\beta,\\gamma)$ y se desplaza el pico $\\delta\\,\\Delta f_{bin}$.",
             [rf"\alpha = {la:.5f},\quad \beta = {lb:.5f},\quad \gamma = {lc:.5f}",
              rf"\delta = \dfrac{{\alpha - \gamma}}{{2(\alpha - 2\beta + \gamma)}} = {dlt:.5f},\qquad f_1^{{num}} = {fr[i1]:.2f} + ({dlt:.5f})({1 / dur}) = {fpk[0]:.4f}\ \mathrm{{Hz}}"]),
            ("Pulsación a partir del espectro", "Se refina el segundo pico de la misma forma y se restan.",
             [rf"f_2^{{num}} = {fpk[-1]:.4f}\ \mathrm{{Hz}},\qquad f_{{puls}} = {fpk[-1]:.4f} - {fpk[0]:.4f} = {fbn:.5f}\ \mathrm{{Hz}}"]),
            ("Pulsación por la envolvente", "En el dominio del tiempo, la envolvente se obtiene de la señal analítica: se anulan las frecuencias negativas en la FFT (se duplican las positivas), se invierte la transformada y se toma el módulo. Los máximos de la envolvente por encima de 1.8 marcan los latidos.",
             [rf"t_{{pico}} = {t2[pe[0]]:.5f},\ {t2[pe[1]]:.5f},\ {t2[pe[2]]:.5f}\ \mathrm{{s}}\ \ \Rightarrow\ \ \overline{{\Delta t}} = {np.mean(np.diff(t2[pe])):.6f}\ \mathrm{{s}}",
              rf"f_{{puls}} = \dfrac{{1}}{{\overline{{\Delta t}}}} = {fbe:.5f}\ \mathrm{{Hz}}\quad(\text{{exacto }}{fb:.0f}\ \mathrm{{Hz}})"]),
            ("Tubos como autovalores: versión de 3 puntos", "Como en la cuerda, $p'' = -k^2p$ se discretiza con una matriz tridiagonal y los modos son los autovalores. Abierto: $p=0$ en ambos extremos, $\\Delta x = L/(N+1)$. Cerrado: $p'=0$ en el extremo cerrado, lo que cambia el primer elemento diagonal a $1/\\Delta x^2$ y usa $\\Delta x = L/(N+0.5)$. Con $N=3$:",
             [rf"\text{{abierto}}:\ \Delta x = {dxo:.4f}\ \mathrm{{m}},\ \ f = {fo3[0]:.2f},\ {fo3[1]:.2f},\ {fo3[2]:.2f}\ \mathrm{{Hz}}\ \ (\text{{exacto }}{op_a[0]:.2f},\ {op_a[1]:.2f},\ {op_a[2]:.2f})",
              rf"\text{{cerrado}}:\ \Delta x = {dxc:.4f}\ \mathrm{{m}},\ \ f = {fc3[0]:.2f},\ {fc3[1]:.2f},\ {fc3[2]:.2f}\ \mathrm{{Hz}}\ \ (\text{{exacto }}{cl_a[0]:.2f},\ {cl_a[1]:.2f},\ {cl_a[2]:.2f})"]),
            ("Escala real y errores", "El código usa $N=600$ puntos y toma los cuatro menores autovalores. El error disminuye como $\\Delta x^2$.",
             [r"\begin{array}{l|c|c|c} \text{tubo} & f_1^{anal}\ [\mathrm{Hz}] & f_1^{num}\ [\mathrm{Hz}] & |\Delta f_1|\ [\mathrm{Hz}]\\ \hline "
              + rf"\text{{abierto}} & {op_a[0]:.4f} & {float(op_n[0]):.4f} & {sci(abs(float(op_n[0]) - op_a[0]), 2)}\\ "
              + rf"\text{{cerrado}} & {cl_a[0]:.4f} & {float(cl_n[0]):.4f} & {sci(abs(float(cl_n[0]) - cl_a[0]), 2)}" + r"\end{array}"]),
        ],
    )


DETAIL = {
    "VenturiTube": _venturi(),
    "TorricelliDrain": _torricelli(),
    "SimpleHarmonic": _shm(),
    "DrivenResonance": _driven(),
    "WavePulse": _pulse(),
    "StandingModes": _modes(),
    "DopplerEffect": _doppler(),
    "BeatsAndPipes": _beats(),
}
