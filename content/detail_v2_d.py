"""Explicacion y pasos a mano, Volumen 2 capitulos 14 a 18.

Escenas: YeeFDTD, PoyntingDisplacement, SnellFermat, PrismDispersion, ThinLensImage,
TwoLensSystem, YoungSlits, ThinFilmCoating, SlitAndGrating, PolarizersMalus.
Todas las cifras se calculan con funciones de `physics/` o replicando su algoritmo.
"""
from __future__ import annotations

import numpy as np

from content.texfmt import fx, sci
from physics import v2ch14_maxwell as m14
from physics import v2ch15_light as m15
from physics import v2ch16_lenses as m16
from physics import v2ch17_interference as m17
from physics import v2ch18_diffraction as m18
from physics.ch07_work_energy import simpson
from physics.constants import C_LIGHT, EPS0, MU0


def _tab(cols: str, rows: list[list[str]]) -> str:
    """Tabla TeX (array) a partir de filas de cadenas ya formateadas."""
    return r"\begin{array}{" + cols + r"}" + r" \\ ".join(" & ".join(r) for r in rows) + r"\end{array}"


# ===================================================================== Capitulo 14
def _yee() -> dict:
    c0 = m14.c_from_constants()
    dx, cour = 1e-3, 0.5
    dt = cour * dx / C_LIGHT
    n = 1200
    x = np.arange(n) * dx
    x0, sg = 0.2 * n * dx, 20 * dx
    ez = np.exp(-((x - x0) ** 2) / (2 * sg**2))
    xh = (np.arange(n - 1) + 0.5) * dx
    hy = -np.exp(-((xh - x0 + 0.5 * C_LIGHT * dt) ** 2) / (2 * sg**2)) / (MU0 * C_LIGHT)
    j = 220
    hy0 = hy[j]
    ca = dt / (MU0 * dx)
    cb = dt / (EPS0 * dx)
    hy_new = hy.copy()
    hy_new += ca * (ez[1:] - ez[:-1])
    ez_new = ez.copy()
    ez_new[1:-1] += cb * (hy_new[1:] - hy_new[:-1])
    _, _, pk = m14.yee_pulse()
    slope = float(np.polyfit(pk[:, 0], pk[:, 1], 1)[0])
    s9 = m14.speed_numeric(0.9)
    sa = m14.speed_numeric(0.5)
    tab = _tab("rrr", [[r"t\ (\mathrm{ns})", r"x_{pico}\ (\mathrm{m})", r"\text{muestra}"]] +
               [[fx(pk[i, 0] * 1e9, 4), fx(pk[i, 1], 5), str(i + 1)] for i in (0, 1, 2, len(pk) - 1)])
    return dict(
        explanation=[
            "Las ecuaciones de Maxwell acoplan los dos campos: un campo magnético que cambia en el tiempo crea un campo eléctrico (Faraday), y un campo eléctrico que cambia en el tiempo crea un campo magnético (Ampère con el término de Maxwell). En el vacío, sin cargas ni corrientes, cada campo es la fuente del otro y la perturbación se sostiene sola.",
            "Al combinar ambas ecuaciones aparece una ecuación de onda cuya velocidad depende únicamente de las constantes eléctrica y magnética del vacío, $c = 1/\\sqrt{\\mu_0\\varepsilon_0}$. Que ese número coincida con la velocidad medida de la luz fue la señal de que la luz es una onda electromagnética.",
            "En una onda plana que viaja en $+x$ los campos son perpendiculares entre sí y a la dirección de propagación, y sus módulos cumplen $E = cB$. El pulso gaussiano de esta escena, de ancho $\\sigma = 2$ cm, es una superposición de muchas ondas planas que se mueven juntas sin deformarse porque en el vacío no hay dispersión.",
            "El esquema de Yee aproxima las derivadas por diferencias finitas y coloca $E$ y $B$ en mallas desfasadas media celda en el espacio y medio paso en el tiempo, de modo que cada derivada es centrada y el error es de segundo orden. El número de Courant $S = c\\,\\Delta t/\\Delta x$ mide cuántas celdas avanza la luz por paso; debe ser $S\\le 1$ para que el esquema sea estable, y con $S = 1$ en una dimensión el esquema reproduce la onda exactamente.",
        ],
        analytic_steps=[
            ("Escribir las dos ecuaciones en el vacío", "Para una onda con $\\vec E = E_z(x,t)\\hat z$ y $\\vec B = B_y(x,t)\\hat y$ los rotacionales se reducen a derivadas en $x$ y las ecuaciones de Faraday y de Ampère-Maxwell (con $\\vec J = 0$) quedan acopladas.",
             [r"\dfrac{\partial B_y}{\partial t} = \dfrac{\partial E_z}{\partial x},\qquad \dfrac{\partial E_z}{\partial t} = \dfrac{1}{\mu_0\varepsilon_0}\,\dfrac{\partial B_y}{\partial x}"]),
            ("Eliminar un campo", "Se deriva la primera respecto de $t$, la segunda respecto de $x$, y se sustituye una en la otra. Queda una sola ecuación para $E_z$.",
             [r"\dfrac{\partial^2 E_z}{\partial t^2} = \dfrac{1}{\mu_0\varepsilon_0}\,\dfrac{\partial^2 E_z}{\partial x^2}"]),
            ("Identificar la velocidad de la onda", "Esta es la ecuación de onda $\\partial_t^2 f = v^2\\partial_x^2 f$, de modo que el coeficiente que multiplica a la derivada espacial es $v^2$.",
             [r"v^2 = \dfrac{1}{\mu_0\varepsilon_0}\quad\Rightarrow\quad v = c = \dfrac{1}{\sqrt{\mu_0\varepsilon_0}}"]),
            ("Sustituir las constantes", "Con $\\mu_0 = 4\\pi\\times10^{-7}$ T m/A y $\\varepsilon_0 = 8.854\\times10^{-12}$ F/m, expresadas con más cifras para no perder precisión.",
             [rf"\mu_0\varepsilon_0 = ({sci(MU0,6)})({sci(EPS0,6)}) = {sci(MU0*EPS0,6)}\ \mathrm{{s^2/m^2}}",
              rf"c = \dfrac{{1}}{{\sqrt{{{sci(MU0*EPS0,6)}}}}} = {sci(c0,7)}\ \mathrm{{m/s}}"]),
            ("Relación entre los campos del pulso", "Una solución que viaja a $+x$ es $E_z = f(x-ct)$; al sustituirla en Faraday se obtiene $B_y$ proporcional. Por eso en el código $H_y = -E_z/(\\mu_0 c)$ al iniciar el pulso.",
             [rf"\dfrac{{E}}{{B}} = c = {sci(c0,7)}\ \mathrm{{m/s}}"]),
            ("Tiempo esperado para recorrer la ventana", "Con 600 pasos de $\\Delta t = 0.5\\,\\Delta x/c$ el pulso debe avanzar una distancia $c\\,t$; esto sirve para comprobar el resultado numérico.",
             [rf"t = 600\,\Delta t = {sci(600*dt,4)}\ \mathrm{{s}},\qquad \Delta x_{{pulso}} = c\,t = {600*cour*dx:.4f}\ \mathrm{{m}}"]),
        ],
        numeric_steps=[
            ("Malla y paso temporal", "Se toma $\\Delta x = 1$ mm con 1200 nodos para $E_z$ y 1199 para $H_y$ (medias celdas), y el paso se fija con el número de Courant $S = 0.5$. El pulso inicial es una gaussiana centrada en 24 cm con $\\sigma = 20\\,\\Delta x$.",
             [rf"\Delta t = \dfrac{{S\,\Delta x}}{{c}} = \dfrac{{0.5\,({sci(dx,2)})}}{{{sci(C_LIGHT,7)}}} = {sci(dt,5)}\ \mathrm{{s}}"]),
            ("Fórmulas de actualización (salto de rana)", "Primero se avanza $H_y$ con la diferencia de $E_z$ en sus dos nodos vecinos y luego se avanza $E_z$ con la diferencia de $H_y$ ya actualizado. Los coeficientes son constantes en el vacío.",
             [r"H_y^{n+1}[j] = H_y^{n}[j] + \dfrac{\Delta t}{\mu_0\Delta x}\left(E_z[j+1]-E_z[j]\right)",
              r"E_z^{n+1}[j] = E_z^{n}[j] + \dfrac{\Delta t}{\varepsilon_0\Delta x}\left(H_y[j]-H_y[j-1]\right)",
              rf"\dfrac{{\Delta t}}{{\mu_0\Delta x}} = {sci(ca,5)},\qquad \dfrac{{\Delta t}}{{\varepsilon_0\Delta x}} = {sci(cb,5)}"]),
            ("Primera actualización de H en el nodo j = 220", "Es el flanco anterior del pulso, donde la pendiente de $E_z$ es grande. Se parte de los valores iniciales y se suma la corrección.",
             [rf"E_z[220]={ez[220]:.6f},\quad E_z[221]={ez[221]:.6f},\quad H_y^{{0}}[220]={sci(hy0,5)}\ \mathrm{{A/m}}",
              rf"H_y^{{1}}[220] = {sci(hy0,5)} + ({sci(ca,5)})({ez[221]-ez[220]:.6f}) = {sci(hy_new[j],5)}\ \mathrm{{A/m}}"]),
            ("Primera actualización de E en el nodo j = 221", "Se usa el $H_y$ recién calculado en los nodos 220 y 221 (la diferencia es el rotacional discreto).",
             [rf"E_z^{{1}}[221] = {ez[221]:.6f} + ({sci(cb,5)})\left({sci(hy_new[221],5)} - ({sci(hy_new[220],5)})\right) = {ez_new[221]:.6f}",
              rf"\text{{El pico avanzó de }}x=0.240\text{{ m a }}{(np.argmax(ez_new))*dx:.3f}\text{{ m en el primer paso (media celda por paso a }}S=0.5)."]),
            ("Seguimiento del pico", "Cada 20 pasos se ubica el máximo de $E_z$ y se refina con una parábola por los tres nodos vecinos: $x = (j + \\delta)\\Delta x$ con $\\delta = \\tfrac12\\,(y_0-y_2)/(y_0-2y_1+y_2)$. Se guardan parejas $(t, x_{pico})$.",
             [tab]),
            ("Regresión lineal de la velocidad", "La velocidad es la pendiente de la recta de mínimos cuadrados $x_{pico} = x_0 + v\\,t$ sobre las 31 muestras.",
             [rf"v_{{num}}(S=0.5) = {sci(sa,7)}\ \mathrm{{m/s}} = {sa/1e8:.6f}\times10^{{8}}\ \mathrm{{m/s}}",
              rf"\text{{error relativo}} = \dfrac{{|v_{{num}}-c|}}{{c}} = {sci(abs(sa-c0)/c0,3)}"]),
            ("Efecto del número de Courant", "Con $S = 0.9$ el esquema se acerca al paso exacto $S = 1$ y la dispersión numérica disminuye: el pulso viaja más cerca de $c$.",
             [rf"v_{{num}}(S=0.9) = {s9/1e8:.6f}\times10^{{8}}\ \mathrm{{m/s}},\qquad \text{{error relativo}} = {sci(abs(s9-c0)/c0,3)}"]),
        ],
    )


def _poynting() -> dict:
    e0 = m14.E0
    ia = m14.intensity_analytic()
    b0 = e0 / C_LIGHT
    s0 = e0 * b0 / MU0
    p = m14.radiation_pressure(ia)
    p2 = m14.radiation_pressure(ia, False)
    idis = m14.conduction_current_analytic()
    inum = m14.displacement_current_numeric()
    t8 = np.linspace(0, 1, 9)
    s8 = (e0 * np.sin(2 * np.pi * t8)) ** 2 / C_LIGHT / MU0
    q8 = simpson(s8, t8)
    q4000 = m14.intensity_numeric()
    t0, h = 1e-6, 1e-9
    phi = lambda t: (m14.DVDT * t / m14.GAP) * m14.AREA  # noqa: E731
    simp = _tab("cc", [[r"t_k/T", r"S(t_k)\ (\mathrm{W/m^2})"]] + [[f"{t8[k]:.3f}", fx(s8[k], 3)] for k in range(9)])
    return dict(
        explanation=[
            "Una onda electromagnética transporta energía. El vector de Poynting $\\vec S = \\vec E\\times\\vec B/\\mu_0$ da la energía por unidad de área y de tiempo, y su dirección es la de propagación. Para una onda plana $B = E/c$, por lo que $S = E^2/(\\mu_0 c)$ oscila al doble de la frecuencia de la onda y nunca es negativo.",
            "Lo que miden los detectores es el promedio temporal, la intensidad $I = \\langle S\\rangle = \\tfrac12 c\\varepsilon_0 E_0^2$; el factor $\\tfrac12$ viene de promediar $\\sin^2$ en un periodo. Con $E_0 = 100$ V/m, un campo comparable al de la luz solar de unos pocos W/m² se queda corto, así que la intensidad es del orden de decenas de W/m².",
            "La onda también transporta cantidad de movimiento. Un absorbente perfecto recibe presión $p = I/c$ y un reflector perfecto $2I/c$. Como $c$ es enorme la presión es de nanopascales: real pero minúscula, apreciable solo en velas solares o con láseres muy intensos.",
            "En un capacitor que se carga no hay carga que atraviese el espacio entre placas, pero el campo eléctrico cambia. Maxwell añadió la corriente de desplazamiento $I_d = \\varepsilon_0\\,d\\Phi_E/dt$, que produce campo magnético igual que una corriente de conducción y hace que la corriente sea continua a lo largo del circuito; sin ella la ley de Ampère sería inconsistente.",
        ],
        analytic_steps=[
            ("Campo magnético de la onda", "En el vacío los módulos de los campos de una onda plana se relacionan por $E = cB$.",
             [rf"B_0 = \dfrac{{E_0}}{{c}} = \dfrac{{{e0:.1f}}}{{{sci(C_LIGHT,7)}}} = {sci(b0,5)}\ \mathrm{{T}}"]),
            ("Valor instantáneo y máximo de S", "El módulo del vector de Poynting es $S = EB/\\mu_0$ con $E = E_0\\sin\\omega t$. Su valor pico es:",
             [r"S(t) = \dfrac{E_0^2}{\mu_0 c}\,\sin^2\omega t",
              rf"S_{{max}} = \dfrac{{({e0:.1f})^2}}{{({sci(MU0,6)})({sci(C_LIGHT,7)})}} = {s0:.5f}\ \mathrm{{W/m^2}}"]),
            ("Promedio temporal (intensidad)", "El promedio de $\\sin^2$ en un periodo es $1/2$. Usando $\\mu_0 = 1/(\\varepsilon_0 c^2)$, el resultado se escribe con $\\varepsilon_0$.",
             [r"I = \langle S\rangle = \dfrac{E_0^2}{2\mu_0 c} = \tfrac12 c\,\varepsilon_0 E_0^2",
              rf"I = \tfrac12 ({sci(C_LIGHT,7)})({sci(EPS0,6)})({e0:.1f})^2 = {ia:.5f}\ \mathrm{{W/m^2}}"]),
            ("Presión de radiación", "Un absorbente perfecto absorbe toda la cantidad de movimiento $I/c$ por unidad de área y de tiempo. Si fuera un espejo perfecto la presión sería el doble.",
             [rf"p = \dfrac{{I}}{{c}} = \dfrac{{{ia:.5f}}}{{{sci(C_LIGHT,7)}}} = {sci(p,5)}\ \mathrm{{Pa}} = {p*1e9:.4f}\ \mathrm{{nPa}}",
              rf"p_{{espejo}} = 2I/c = {p2*1e9:.4f}\ \mathrm{{nPa}}"]),
            ("Campo eléctrico y flujo en el capacitor", "Entre placas paralelas $E = V/d$, y el flujo a través del área $A$ es $\\Phi_E = EA = VA/d$.",
             [r"\Phi_E = \dfrac{V A}{d},\qquad \dfrac{d\Phi_E}{dt} = \dfrac{A}{d}\,\dfrac{dV}{dt}"]),
            ("Corriente de desplazamiento", "Se multiplica por $\\varepsilon_0$ y se sustituyen $A = 100\\ \\mathrm{cm^2} = 0.010\\ \\mathrm{m^2}$, $d = 1.0$ mm y $dV/dt = 10^6$ V/s.",
             [r"I_d = \varepsilon_0\dfrac{A}{d}\dfrac{dV}{dt}",
              rf"I_d = ({sci(EPS0,6)})\dfrac{{{m14.AREA}}}{{{sci(m14.GAP,2)}}}({sci(m14.DVDT,2)}) = {sci(idis,5)}\ \mathrm{{A}} = {idis*1e6:.4f}\ \mu\mathrm{{A}}"]),
            ("Interpretación", "La corriente de desplazamiento es idéntica a la que cargaría un capacitor $C = \\varepsilon_0 A/d$ con la misma rapidez de cambio de voltaje: $I = C\\,dV/dt$.",
             [rf"C = \dfrac{{\varepsilon_0 A}}{{d}} = {sci(EPS0*m14.AREA/m14.GAP,5)}\ \mathrm{{F}},\qquad I = C\,\dfrac{{dV}}{{dt}} = {idis*1e6:.4f}\ \mu\mathrm{{A}}"]),
        ],
        numeric_steps=[
            ("Plantear el promedio como integral", "La intensidad es el promedio de $S(t)$ en un periodo, es decir una integral dividida entre la duración. Se usa un periodo normalizado $t\\in[0,1]$, así el promedio es directamente la integral.",
             [r"I = \dfrac{1}{T}\int_0^T S(t)\,dt,\qquad S_k = \dfrac{E_k B_k}{\mu_0},\ \ E_k = E_0\sin(2\pi t_k),\ B_k = E_k/c"]),
            ("Regla de Simpson", "Con $N$ subintervalos pares de ancho $h = 1/N$ se combinan las muestras con pesos 1, 4, 2, 4, ..., 4, 1.",
             [r"\int S\,dt \approx \dfrac{h}{3}\left[S_0 + 4S_1 + 2S_2 + 4S_3 + \cdots + 4S_{N-1} + S_N\right]"]),
            ("Versión reducida con N = 8", "Se calculan las 9 muestras de $S$ (la onda es senoidal, de modo que $S\\propto\\sin^2$).",
             [simp]),
            ("Suma ponderada con N = 8", "Los puntos impares (peso 4) valen $S_{max}/2$ y los pares interiores (peso 2) valen $S_{max}$ o 0. Con $h = 1/8$:",
             [rf"\dfrac{{h}}{{3}}\left[0 + 4(4\cdot{s8[1]:.3f}) + 2({s8[2]:.3f}+{s8[4]:.3f}+{s8[6]:.3f}) + 0\right] = \dfrac{{1}}{{24}}({4*4*s8[1]+2*(s8[2]+s8[4]+s8[6]):.3f}) = {q8:.5f}\ \mathrm{{W/m^2}}"]),
            ("Resultado con el N real", "El código usa $N = 4000$; el resultado coincide con $\\tfrac12 S_{max}$ analítico a precisión de máquina porque Simpson integra exactamente un periodo completo de $\\sin^2$.",
             [rf"I_{{num}} = {q4000:.6f}\ \mathrm{{W/m^2}},\qquad I_{{anal}} = {ia:.6f},\qquad |I_{{num}}-I_{{anal}}| = {sci(abs(q4000-ia),2)}"]),
            ("Derivada numérica por diferencias centrales", "Para la corriente de desplazamiento se evalúa el flujo en $t_0 \\pm h$ con $t_0 = 1\\ \\mu$s y $h = 10^{-9}$ s, y se aplica $\\Phi'(t_0)\\approx[\\Phi(t_0+h)-\\Phi(t_0-h)]/2h$.",
             [rf"\Phi(t_0+h) = {phi(t0+h):.7f},\quad \Phi(t_0-h) = {phi(t0-h):.7f}\ \mathrm{{V\,m}}",
              rf"\dfrac{{d\Phi_E}}{{dt}} \approx \dfrac{{{sci(phi(t0+h)-phi(t0-h),5)}}}{{2({sci(h,1)})}} = {sci((phi(t0+h)-phi(t0-h))/(2*h),5)}\ \mathrm{{V\,m/s}}"]),
            ("Corriente y error", "Como $\\Phi$ es lineal en $t$ la diferencia central es exacta; el error que queda es redondeo de punto flotante.",
             [rf"I_d^{{num}} = \varepsilon_0\,\dfrac{{d\Phi_E}}{{dt}} = {inum*1e6:.6f}\ \mu\mathrm{{A}},\qquad |I_d^{{num}}-I_d| = {sci(abs(inum-idis)*1e6,2)}\ \mu\mathrm{{A}}"]),
        ],
    )


# ===================================================================== Capitulo 15
def _snell() -> dict:
    h1, h2 = m15.H1, m15.H2
    th1 = np.deg2rad(m15.THETA1_DEG)
    th2 = m15.snell_angle(th1)
    xs = m15.refraction_point_analytic()
    d, _ = m15.target_point()
    xn = m15.fermat_numeric(d)
    n1, n2 = m15.N1, m15.N2
    s1, s2 = np.hypot(xs, h1), np.hypot(d - xs, h2)
    tc = m15.critical_angle_analytic()
    tcn = m15.critical_angle_numeric()
    g = (np.sqrt(5) - 1) / 2
    a, b = 0.0, d
    c, e = b - g * (b - a), a + g * (b - a)
    opl = lambda x: float(m15.travel_time(x, d) * C_LIGHT)  # noqa: E731
    rows, it = [], []
    for k in range(5):
        fc, fe = opl(c), opl(e)
        rows.append([str(k + 1), fx(a, 5), fx(b, 5), fx(c, 5), fx(e, 5), fx(fc, 6), fx(fe, 6)])
        if fc < fe:
            b = e
        else:
            a = c
        c, e = b - g * (b - a), a + g * (b - a)
    tabg = _tab("rrrrrrr", [[r"k", r"a", r"b", r"x_c", r"x_e", r"L(x_c)", r"L(x_e)"]] + rows)
    lo, hi = 0.0, 90.0
    br = []
    for k in range(7):
        mid = 0.5 * (lo + hi)
        v = n2 * np.sin(np.deg2rad(mid))
        ok = v <= n1
        br.append([str(k + 1), fx(lo, 4), fx(hi, 4), fx(mid, 4), fx(v, 4), r"\text{sí}" if ok else r"\text{no}"])
        if ok:
            lo = mid
        else:
            hi = mid
    tabb = _tab("rrrrrc", [[r"k", r"\theta_{lo}", r"\theta_{hi}", r"\theta_{mid}", r"n\sin\theta", r"\le 1"]] + br)
    return dict(
        explanation=[
            "Cuando la luz pasa de un medio a otro su velocidad cambia a $v = c/n$ y el rayo se desvía en la interfaz. La ley de Snell, $n_1\\sin\\theta_1 = n_2\\sin\\theta_2$, relaciona los ángulos medidos respecto de la normal: al entrar en un medio más denso (mayor $n$) el rayo se acerca a la normal.",
            "El principio de Fermat da una explicación más profunda: entre todos los caminos posibles, la luz sigue aquel para el cual el tiempo de viaje es estacionario (en general mínimo). Minimizar $T(x)$ respecto del punto de incidencia $x$ conduce exactamente a la ley de Snell, así que ambos enfoques deben dar el mismo punto, y eso es lo que se verifica numéricamente.",
            "Cuando la luz va de un medio denso a uno menos denso, $\\sin\\theta_2 = (n_1/n_2)\\sin\\theta_1$ puede exceder 1. Para ángulos mayores que el ángulo crítico, $\\theta_c = \\arcsin(n_2/n_1)$, no existe rayo refractado y toda la luz se refleja (reflexión total interna), lo que explica las fibras ópticas y el brillo de los diamantes.",
        ],
        analytic_steps=[
            ("Ley de Snell en la interfaz aire-vidrio", "Se aplica a la incidencia de 40° con $n_1 = 1.0$ y $n_2 = 1.5$ y se despeja el seno del ángulo refractado.",
             [r"\sin\theta_2 = \dfrac{n_1}{n_2}\sin\theta_1",
              rf"\sin\theta_2 = \dfrac{{1.0}}{{1.5}}\sin 40^\circ = \dfrac{{{np.sin(th1):.5f}}}{{1.5}} = {np.sin(th2):.5f}"]),
            ("Ángulo refractado", "Se toma el arcoseno.",
             [rf"\theta_2 = \arcsin({np.sin(th2):.5f}) = {np.degrees(th2):.4f}^\circ"]),
            ("Punto de incidencia", "El rayo sale de $(0, h_1)$ con $h_1 = 1.0$ m. Su ángulo con la vertical es $\\theta_1$, de modo que el recorrido horizontal hasta la interfaz es $h_1\\tan\\theta_1$.",
             [rf"x^* = h_1\tan\theta_1 = (1.0)\tan 40^\circ = {xs:.5f}\ \mathrm{{m}}"]),
            ("Punto de llegada", "En el vidrio el rayo baja $h_2 = 1.5$ m con ángulo $\\theta_2$, así que el punto de llegada está a una distancia horizontal $d$ del origen.",
             [rf"d = x^* + h_2\tan\theta_2 = {xs:.5f} + 1.5\,({np.tan(th2):.5f}) = {d:.5f}\ \mathrm{{m}}"]),
            ("Condición de Fermat", "El tiempo de viaje es $T(x) = [n_1\\sqrt{x^2+h_1^2} + n_2\\sqrt{(d-x)^2+h_2^2}]/c$. Al derivar e igualar a cero aparecen los senos de los ángulos.",
             [r"\dfrac{dT}{dx} = \dfrac1c\left[n_1\dfrac{x}{\sqrt{x^2+h_1^2}} - n_2\dfrac{d-x}{\sqrt{(d-x)^2+h_2^2}}\right] = 0\ \Rightarrow\ n_1\sin\theta_1 = n_2\sin\theta_2",
              rf"n_1\dfrac{{x^*}}{{s_1}} = {n1*xs/s1:.5f},\qquad n_2\dfrac{{d-x^*}}{{s_2}} = {n2*(d-xs)/s2:.5f}"]),
            ("Ángulo crítico vidrio-aire", "Del vidrio al aire el máximo ángulo de refracción es $90^\\circ$, y el ángulo de incidencia que lo produce es el crítico: $n_1\\sin\\theta_c = n_2\\sin 90^\\circ$.",
             [rf"\theta_c = \arcsin\dfrac{{1.0}}{{1.5}} = \arcsin({1/1.5:.5f}) = {tc:.4f}^\circ"]),
        ],
        numeric_steps=[
            ("Función a minimizar", "Se trabaja con la longitud de camino óptico $L(x) = c\\,T(x) = n_1\\sqrt{x^2+h_1^2} + n_2\\sqrt{(d-x)^2+h_2^2}$ en el intervalo $[0, d]$ con $d = " + f"{d:.5f}" + "$ m.",
             [rf"L(0) = {opl(0.0):.6f},\qquad L(d) = {opl(d):.6f},\qquad L(x^*) = {opl(xs):.6f}\ \mathrm{{m}}"]),
            ("Búsqueda de la sección áurea", "Se mantienen dos puntos interiores $x_c = b - g(b-a)$ y $x_e = a + g(b-a)$ con $g = (\\sqrt5-1)/2 = " + f"{g:.6f}" + "$. Si $L(x_c) < L(x_e)$ el mínimo está en $[a, x_e]$; si no, en $[x_c, b]$. El intervalo se reduce un factor $g$ por iteración.",
             [r"L(x_c) < L(x_e)\ \Rightarrow\ b\leftarrow x_e;\qquad \text{si no}\ \Rightarrow\ a\leftarrow x_c"]),
            ("Primeras cinco iteraciones", "La primera usa $a=0$, $b=d$. Obsérvese cómo el intervalo se contrae hacia $x^* = " + f"{xs:.5f}" + "$.",
             [tabg]),
            ("Parada y resultado", "El código hace 200 iteraciones; el intervalo queda del tamaño de la precisión de máquina y se toma su centro.",
             [rf"x_{{num}} = {xn:.8f}\ \mathrm{{m}},\qquad x^* = {xs:.8f}\ \mathrm{{m}},\qquad |x_{{num}}-x^*| = {sci(abs(xn-xs),2)}\ \mathrm{{m}}"]),
            ("Ángulo crítico por bisección", "Se busca el ángulo en $[0^\\circ, 90^\\circ]$ donde $1.5\\sin\\theta$ cruza 1. En cada paso se evalúa el punto medio: si $1.5\\sin\\theta_{mid}\\le 1$ hay rayo refractado y se sube el extremo inferior; si no, se baja el superior.",
             [tabb]),
            ("Convergencia de la bisección", "Cada paso divide el intervalo a la mitad, de modo que tras 80 pasos su ancho es $90^\\circ/2^{80}\\approx 7\\times10^{-23}$ grados, mucho menor que la precisión de máquina.",
             [rf"\theta_{{c,num}} = {tcn:.8f}^\circ,\qquad \theta_{{c}} = {tc:.8f}^\circ,\qquad |\Delta| = {sci(abs(tcn-tc),2)}^\circ"]),
        ],
    )


def _prism() -> dict:
    A = m15.APEX_DEG
    lams = (400, 550, 700)
    ns = {l: float(m15.cauchy(l)) for l in lams}
    da = {l: float(m15.min_deviation_analytic(ns[l])) for l in lams}
    dn = {l: m15.min_deviation_numeric(ns[l]) for l in lams}
    n = ns[550]
    th_m = np.degrees(np.arcsin(n * np.sin(np.deg2rad(A / 2))))
    g = (np.sqrt(5) - 1) / 2
    lo, hi = 10.0, 85.0
    c, e = hi - g * (hi - lo), lo + g * (hi - lo)
    rows = []
    for k in range(5):
        dc, de = m15.deviation(c, n), m15.deviation(e, n)
        rows.append([str(k + 1), fx(lo, 3), fx(hi, 3), fx(c, 3), fx(e, 3), fx(dc, 4), fx(de, 4)])
        if dc < de:
            hi = e
        else:
            lo = c
        c, e = hi - g * (hi - lo), lo + g * (hi - lo)
    tab = _tab("rrrrrrr", [[r"k", r"\mathrm{lo}", r"\mathrm{hi}", r"\theta_c", r"\theta_e", r"\delta(\theta_c)", r"\delta(\theta_e)"]] + rows)
    t_in = 50.0
    t1 = np.deg2rad(t_in)
    r1 = np.arcsin(np.sin(t1) / n)
    r2 = np.deg2rad(A) - r1
    sx = n * np.sin(r2)
    t2 = np.arcsin(sx)
    return dict(
        explanation=[
            "El índice de refracción de un vidrio depende de la longitud de onda: es mayor para el violeta que para el rojo. Esta dispersión se describe bien con la fórmula empírica de Cauchy, $n(\\lambda) = A + B/\\lambda^2$, y es la razón por la que un prisma separa la luz blanca en colores.",
             "Un rayo que atraviesa el prisma se refracta en la primera cara, recorre el vidrio y se refracta de nuevo en la segunda. La desviación total $\\delta = \\theta_1 + \\theta_2 - \\alpha$ depende del ángulo de incidencia y tiene un mínimo cuando el rayo cruza el prisma simétricamente, es decir paralelo a la base.",
            "En el mínimo de desviación el ángulo de incidencia y el de salida son iguales y el rayo interior forma $\\alpha/2$ con la normal de cada cara, lo que da $n = \\sin[(\\alpha+\\delta_{min})/2]/\\sin(\\alpha/2)$. Es el método estándar para medir el índice de un vidrio.",
            "Como $n$ es mayor para 400 nm, el violeta se desvía más que el rojo. La diferencia entre ambas desviaciones mínimas es la dispersión angular del prisma, de unos pocos grados para un vidrio común.",
        ],
        analytic_steps=[
            ("Índice para cada color", "Se evalúa la ley de Cauchy con $A = 1.50$ y $B = 5000\\ \\mathrm{nm^2}$ en 400, 550 y 700 nm.",
             [rf"n({l}\ \mathrm{{nm}}) = 1.50 + \dfrac{{5000}}{{{l}^2}} = {ns[l]:.5f}" for l in lams]),
            ("Condición de mínima desviación", "Por simetría, en el mínimo el rayo interior forma $r_1 = r_2 = \\alpha/2 = 30^\\circ$ con las normales, y entonces $\\theta_1 = \\theta_2$.",
             [r"\sin\theta_1 = n\sin\dfrac{\alpha}{2}\ \Rightarrow\ \theta_1 = \arcsin\!\left(n\sin\dfrac\alpha2\right)"]),
            ("Desviación mínima", "La desviación es $\\delta = \\theta_1+\\theta_2-\\alpha = 2\\theta_1-\\alpha$.",
             [r"\delta_{min} = 2\arcsin\!\left(n\sin\dfrac\alpha2\right) - \alpha"]),
            ("Caso de 550 nm", "Se sustituye paso a paso, con $\\sin 30^\\circ = 0.5$.",
             [rf"\sin\theta_1 = {n:.5f}\,(0.5) = {n*0.5:.5f}\ \Rightarrow\ \theta_1 = {th_m:.4f}^\circ",
              rf"\delta_{{min}}(550) = 2({th_m:.4f}^\circ) - 60^\circ = {da[550]:.4f}^\circ"]),
            ("Casos extremos", "Se repite el cálculo con los índices de violeta y rojo.",
             [rf"\delta_{{min}}(400) = 2\arcsin({ns[400]*0.5:.5f}) - 60^\circ = {da[400]:.4f}^\circ",
              rf"\delta_{{min}}(700) = 2\arcsin({ns[700]*0.5:.5f}) - 60^\circ = {da[700]:.4f}^\circ"]),
            ("Dispersión angular", "La diferencia entre las desviaciones extremas mide el ancho angular del espectro visible.",
             [rf"\Delta\delta = \delta_{{min}}(400) - \delta_{{min}}(700) = {da[400]:.4f}^\circ - {da[700]:.4f}^\circ = {da[400]-da[700]:.4f}^\circ"]),
        ],
        numeric_steps=[
            ("Trazado de un rayo por el prisma", "Dado el ángulo de incidencia $\\theta_1$ se aplica Snell en la primera cara, se usa la geometría $r_2 = \\alpha - r_1$ y se aplica Snell en la segunda. Con $n = " + f"{n:.5f}" + "$ (550 nm) y $\\theta_1 = 50^\\circ$:",
             [rf"r_1 = \arcsin\dfrac{{\sin 50^\circ}}{{{n:.5f}}} = {np.degrees(r1):.4f}^\circ,\qquad r_2 = 60^\circ - r_1 = {np.degrees(r2):.4f}^\circ"]),
            ("Salida y desviación", "En la segunda cara $\\sin\\theta_2 = n\\sin r_2$ (si fuera mayor que 1 habría reflexión total y el código devuelve NaN). La desviación total sigue.",
             [rf"\sin\theta_2 = {n:.5f}\sin({np.degrees(r2):.4f}^\circ) = {sx:.5f}\ \Rightarrow\ \theta_2 = {np.degrees(t2):.4f}^\circ",
              rf"\delta(50^\circ) = 50^\circ + {np.degrees(t2):.4f}^\circ - 60^\circ = {m15.deviation(50.0, n):.4f}^\circ"]),
            ("Minimizar por sección áurea", "Se busca el mínimo de $\\delta(\\theta_1)$ en $[10^\\circ, 85^\\circ]$ con $g = 0.618034$: se comparan $\\delta(\\theta_c)$ y $\\delta(\\theta_e)$ y se descarta el tramo donde no puede estar el mínimo.",
             [r"\delta(\theta_c)<\delta(\theta_e)\ \Rightarrow\ \mathrm{hi}\leftarrow\theta_e;\qquad \text{si no}\ \Rightarrow\ \mathrm{lo}\leftarrow\theta_c"]),
            ("Primeras cinco iteraciones para 550 nm", "El intervalo se estrecha hacia el ángulo de mínima desviación (" + f"{th_m:.3f}" + "$^\\circ$ según el cálculo analítico).",
             [tab]),
            ("Parada", "El código hace 100 iteraciones, y evalúa $\\delta$ en el centro del intervalo final. Como cerca del mínimo $\\delta$ es plana, el valor de $\\delta$ converge mucho mejor que el ángulo.",
             [rf"\delta_{{num}}(550) = {dn[550]:.8f}^\circ,\qquad \delta_{{anal}}(550) = {da[550]:.8f}^\circ"]),
            ("Resultados y errores", "Se repite el procedimiento para los otros dos colores.",
             [_tab("rrrr", [[r"\lambda\ (\mathrm{nm})", r"\delta_{num}\ (^\circ)", r"\delta_{anal}\ (^\circ)", r"|\Delta|"]] +
                   [[str(l), fx(dn[l], 6), fx(da[l], 6), sci(abs(dn[l]-da[l]), 2)] for l in lams])]),
        ],
    )


# ===================================================================== Capitulo 16
def _lens() -> dict:
    n, r1, r2 = m16.N_GLASS, m16.R1, m16.R2
    f = m16.focal_lensmaker()
    s = 0.3
    sp = m16.image_distance_analytic(s, f)
    sn = m16.lens_image_numeric(s)
    rm, sm = 0.4, 0.6
    fm = rm / 2
    spm = m16.mirror_image_analytic(sm, rm)
    spn = m16.mirror_image_numeric(sm, rm)
    h, th = 1e-3, 2e-5
    p = np.array([-s, 0.0])
    d = np.array([s, h]) / np.hypot(s, h)
    hit1, d1 = m16._refract_conv(p, d, r1, r1, 1.0, n)
    hit2, d2 = m16._refract_conv(hit1, d1, th + r2, r2, n, 1.0)
    tcut = -hit2[1] / d2[1]
    # espejo
    c = np.array([-rm, 0.0])
    pm = np.array([-sm, 0.0])
    dm = np.array([sm, h]) / np.hypot(sm, h)
    oc = pm - c
    b = oc @ dm
    disc = b * b - (oc @ oc - rm**2)
    t = -b + np.sqrt(disc)
    hit = pm + t * dm
    nrm = (hit - c) / rm
    dr = dm - 2 * (dm @ nrm) * nrm
    t2 = -hit[1] / dr[1]
    return dict(
        explanation=[
            "Una lente delgada hace converger o diverger los rayos porque cada una de sus dos superficies esféricas refracta la luz. Para rayos paraxiales (cercanos al eje y poco inclinados) todos los rayos de un punto objeto se cruzan en un punto imagen, y la potencia de la lente queda resumida en la distancia focal $f$ dada por la ecuación del fabricante.",
            "La ecuación $1/s + 1/s' = 1/f$ relaciona la distancia objeto $s$ con la distancia imagen $s'$. Con la convención usada aquí, $s'>0$ indica una imagen real (del lado de salida de la luz) y $s'<0$ una imagen virtual; el aumento lateral $m = -s'/s$ tiene signo negativo cuando la imagen está invertida.",
            "Un espejo esférico cóncavo obedece la misma ecuación con $f = R/2$, porque un rayo paraxial que llega paralelo al eje se refleja pasando por el punto medio entre el vértice y el centro de curvatura. Por eso un espejo de $R = 40$ cm enfoca a 20 cm.",
            "La aproximación paraxial es la hipótesis clave. Cuando los rayos se alejan del eje o la lente es gruesa aparecen aberraciones: rayos de distinta altura no se cruzan exactamente en el mismo punto. El trazado numérico con la ley de Snell exacta muestra cuán pequeña es esa desviación para un rayo a 1 mm del eje.",
        ],
        analytic_steps=[
            ("Ecuación del fabricante", "Se parte de los radios con signo: $R_1 = +10$ cm (la cara convexa tiene su centro del lado de salida) y $R_2 = -10$ cm.",
             [r"\dfrac1f = (n-1)\left(\dfrac1{R_1}-\dfrac1{R_2}\right)"]),
            ("Sustituir y despejar f", "Con $n = 1.5$ y los radios en metros.",
             [rf"\dfrac1f = (0.5)\left(\dfrac{{1}}{{{r1}}}-\dfrac{{1}}{{{r2}}}\right) = (0.5)({1/r1:.1f}+{1/abs(r2):.1f}) = {1/f:.2f}\ \mathrm{{m^{{-1}}}}",
              rf"f = {f:.3f}\ \mathrm{{m}} = {f*100:.3f}\ \mathrm{{cm}}"]),
            ("Distancia imagen de la lente", "Con el objeto a $s = 30$ cm se despeja $1/s'$ de la ecuación de las lentes delgadas.",
             [r"\dfrac{1}{s'} = \dfrac1f-\dfrac1s",
              rf"\dfrac1{{s'}} = \dfrac{{1}}{{{f}}}-\dfrac{{1}}{{{s}}} = {1/f:.4f}-{1/s:.4f} = {1/f-1/s:.4f}\ \mathrm{{m^{{-1}}}}\ \Rightarrow\ s' = {sp*100:.4f}\ \mathrm{{cm}}"]),
            ("Aumento lateral", "El signo negativo indica imagen invertida.",
             [rf"m = -\dfrac{{s'}}{{s}} = -\dfrac{{{sp*100:.3f}}}{{{s*100:.1f}}} = {-sp/s:.4f}"]),
            ("Espejo cóncavo", "La focal es la mitad del radio: $f = R/2 = 20$ cm, y el objeto está a $s = 60$ cm.",
             [rf"f = \dfrac{{R}}{{2}} = {fm*100:.1f}\ \mathrm{{cm}}"]),
            ("Imagen del espejo", "Se usa la misma ecuación que para la lente.",
             [rf"\dfrac{{1}}{{s'}} = \dfrac{{1}}{{{fm}}}-\dfrac{{1}}{{{sm}}} = {1/fm-1/sm:.4f}\ \mathrm{{m^{{-1}}}}\ \Rightarrow\ s' = {spm*100:.4f}\ \mathrm{{cm}},\qquad m = -\dfrac{{s'}}{{s}} = {-spm/sm:.4f}"]),
        ],
        numeric_steps=[
            ("Rayo de prueba", "Se lanza desde el objeto axial $(-s, 0)$ un rayo que llega a una altura $h = 1$ mm sobre la lente; su dirección es el vector unitario hacia $(0, h)$. Se usa la ley de Snell vectorial exacta, sin aproximación paraxial.",
             [rf"\hat d_0 = \dfrac{{(s,\,h)}}{{\sqrt{{s^2+h^2}}}} = ({d[0]:.8f},\ {d[1]:.8f})"]),
            ("Primera superficie", "La superficie 1 es un arco de centro $(R_1,0)$ y radio $R_1 = 10$ cm. Se resuelve la intersección rayo-esfera (una ecuación cuadrática en el parámetro $t$) y se refracta con la forma vectorial, donde $\\vec N$ es la normal y $\\eta = n_1/n_2$.",
             [r"\hat d' = \eta\,\hat d + \left(\eta\cos\theta_i-\sqrt{1-\eta^2\sin^2\theta_i}\right)\hat N",
              rf"\text{{punto de impacto}} = ({hit1[0]:.8f},\ {hit1[1]:.8f})\ \mathrm{{m}}",
              rf"\hat d_1 = ({d1[0]:.8f},\ {d1[1]:.8f})"]),
            ("Segunda superficie", "El vidrio tiene un espesor finito de $20\\ \\mu$m en el eje. La segunda cara es un arco de centro $(t+R_2,0)$ con $R_2=-10$ cm, y el rayo sale hacia el aire ($\\eta = 1.5$).",
             [rf"\text{{punto de impacto}} = ({hit2[0]:.8f},\ {hit2[1]:.8f})\ \mathrm{{m}}",
              rf"\hat d_2 = ({d2[0]:.8f},\ {d2[1]:.8f})"]),
            ("Corte con el eje", "La imagen está donde el rayo saliente cruza $y=0$: se despeja el parámetro y se mide la posición.",
             [rf"t = -\dfrac{{y}}{{d_y}} = -\dfrac{{{hit2[1]:.8f}}}{{{d2[1]:.8f}}} = {tcut:.6f}\ \mathrm{{m}}",
              rf"s'_{{num}} = x + t\,d_x = {hit2[0]:.8f} + ({tcut:.6f})({d2[0]:.8f}) = {sn*100:.5f}\ \mathrm{{cm}}"]),
            ("Comparación con el resultado analítico", "La diferencia en la cuarta cifra proviene de la aberración esférica del rayo a 1 mm y del espesor finito de la lente.",
             [rf"s'_{{anal}} = {sp*100:.5f}\ \mathrm{{cm}},\qquad |s'_{{num}}-s'_{{anal}}| = {abs(sn-sp)*100:.5f}\ \mathrm{{cm}}"]),
            ("Espejo: reflexión exacta", "El espejo cóncavo es un arco de centro $(-R,0)$. Se intersecta el rayo con la esfera, se calcula la normal $\\hat N$ y se refleja con $\\hat d' = \\hat d - 2(\\hat d\\cdot\\hat N)\\hat N$.",
             [rf"\text{{impacto}} = ({hit[0]:.8f},\ {hit[1]:.8f})\ \mathrm{{m}},\qquad \hat d' = ({dr[0]:.8f},\ {dr[1]:.8f})"]),
            ("Imagen del espejo y error", "Como en la lente, se busca el corte con el eje y se toma su valor absoluto como distancia (la imagen es real, frente al espejo).",
             [rf"s'_{{num}} = {spn*100:.5f}\ \mathrm{{cm}},\quad s'_{{anal}} = {spm*100:.5f}\ \mathrm{{cm}},\quad |\Delta| = {abs(spn-spm)*100:.5f}\ \mathrm{{cm}}"]),
        ],
    )


def _twolens() -> dict:
    f1, f2, dd = m16.F1, m16.F2, m16.SEP
    s = 0.5
    s1p = m16.image_distance_analytic(s, f1)
    s2 = dd - s1p
    s2p, ma = m16.two_lens_analytic()
    zn, mn = m16.two_lens_matrix()
    m1, m2 = -s1p / s, -s2p / s2
    Fs, L1, D, L2 = m16.free_matrix(s), m16.thin_lens_matrix(f1), m16.free_matrix(dd), m16.thin_lens_matrix(f2)
    p1 = Fs
    p2 = L1 @ p1
    p3 = D @ p2
    sysm = L2 @ p3
    a_, b_, c_, d_ = sysm.ravel()

    def mat(m_):
        return rf"\begin{{pmatrix}}{m_[0,0]:.5g}&{m_[0,1]:.5g}\\{m_[1,0]:.5g}&{m_[1,1]:.5g}\end{{pmatrix}}"
    return dict(
        explanation=[
            "Un sistema de dos lentes se resuelve en cadena: la imagen que forma la primera lente actúa como objeto de la segunda. La única precaución es la posición de ese objeto intermedio. Si la imagen intermedia cae más allá de la segunda lente, el objeto es virtual y su distancia se toma negativa en la ecuación de las lentes.",
            "El aumento total es el producto de los aumentos de cada etapa, $m = m_1 m_2$. El signo de $m$ indica la orientación: $m>0$ es una imagen derecha y $m<0$ una invertida. Con los datos de esta escena $m<0$, de modo que la imagen final está invertida respecto del objeto.",
            "El método matricial codifica la misma física con matrices $2\\times2$ que actúan sobre el vector $(y,\\theta)$, altura y pendiente del rayo. Propagar una distancia $d$ es $\\begin{pmatrix}1&d\\\\0&1\\end{pmatrix}$ y pasar por una lente delgada es $\\begin{pmatrix}1&0\\\\-1/f&1\\end{pmatrix}$. La matriz de todo el sistema se obtiene multiplicando en orden inverso al del recorrido.",
            "El plano imagen es aquel donde la altura final no depende de la pendiente inicial del rayo, es decir donde el elemento $B$ de la matriz total vale cero. En ese plano el elemento $A$ es el aumento lateral.",
        ],
        analytic_steps=[
            ("Imagen de la primera lente", "El objeto está a $s=50$ cm de $L_1$ con $f_1=20$ cm.",
             [rf"\dfrac{{1}}{{s_1'}} = \dfrac{{1}}{{{f1}}}-\dfrac{{1}}{{{s}}} = {1/f1-1/s:.4f}\ \mathrm{{m^{{-1}}}}\ \Rightarrow\ s_1' = {s1p*100:.4f}\ \mathrm{{cm}}"]),
            ("Aumento de la primera etapa", "Imagen real e invertida.",
             [rf"m_1 = -\dfrac{{s_1'}}{{s}} = -\dfrac{{{s1p*100:.3f}}}{{{s*100:.0f}}} = {m1:.5f}"]),
            ("Objeto de la segunda lente", "La imagen intermedia está a 33.3 cm de $L_1$, pero $L_2$ está a solo 15 cm. El objeto de $L_2$ queda detrás de ella y es virtual.",
             [rf"s_2 = d - s_1' = {dd*100:.0f} - {s1p*100:.3f} = {s2*100:.4f}\ \mathrm{{cm}}\quad(<0)"]),
            ("Imagen final", "Se aplica la ecuación de las lentes con $f_2=10$ cm y $s_2<0$.",
             [rf"\dfrac{{1}}{{s_2'}} = \dfrac{{1}}{{{f2}}}-\dfrac{{1}}{{{s2:.5f}}} = {1/f2-1/s2:.4f}\ \mathrm{{m^{{-1}}}}\ \Rightarrow\ s_2' = {s2p*100:.4f}\ \mathrm{{cm}}"]),
            ("Aumento de la segunda etapa", "Con $s_2<0$ el aumento $m_2$ resulta positivo.",
             [rf"m_2 = -\dfrac{{s_2'}}{{s_2}} = -\dfrac{{{s2p*100:.4f}}}{{({s2*100:.4f})}} = {m2:.5f}"]),
            ("Aumento total", "Es el producto de las etapas.",
             [rf"m = m_1 m_2 = ({m1:.5f})({m2:.5f}) = {ma:.5f}"]),
        ],
        numeric_steps=[
            ("Matrices elementales", "Se trabaja con $(y,\\theta)$ en metros y radianes. Propagación libre $T(d)$ y lente $L(f)$:",
             [r"T(d)=\begin{pmatrix}1&d\\0&1\end{pmatrix},\qquad L(f)=\begin{pmatrix}1&0\\-1/f&1\end{pmatrix}"]),
            ("Construir el producto paso a paso", "Se parte del objeto y se aplican, de derecha a izquierda, la propagación $s$, la lente 1, la propagación $d$ y la lente 2.",
             [rf"T(0.5)={mat(p1)}",
              rf"L_1T(0.5)={mat(p2)}",
              rf"T(0.15)L_1T(0.5)={mat(p3)}"]),
            ("Matriz del sistema", "Se multiplica por $L_2 = L(0.10)$.",
             [rf"M = L_2\,T(0.15)\,L_1\,T(0.5) = {mat(sysm)}"]),
            ("Condición de imagen", "Se añade una propagación final $z$: el nuevo elemento $B' = B + zD$ debe valer cero, lo que fija el plano imagen.",
             [rf"z = -\dfrac{{B}}{{D}} = -\dfrac{{{b_:.5f}}}{{{d_:.5f}}} = {zn*100:.4f}\ \mathrm{{cm}}"]),
            ("Aumento lateral", "En ese plano el nuevo elemento $A' = A + zC$ es el aumento.",
             [rf"m = A + zC = {a_:.5f} + ({zn:.5f})({c_:.5f}) = {mn:.5f}"]),
            ("Verificación y error", "Ambos métodos usan la misma óptica paraxial y deben coincidir a precisión de máquina.",
             [rf"z - s_2' = {sci(abs(zn-s2p),2)}\ \mathrm{{m}},\qquad m_{{mat}} - m_{{anal}} = {sci(abs(mn-ma),2)}"]),
        ],
    )


# ===================================================================== Capitulo 17
def _young() -> dict:
    lam, d, L = m17.LAM, m17.D_SLIT, m17.L_SCR
    sp = lam * L / d
    ya = {m: m17.fringe_position_analytic(m) for m in (1, 3, 5)}
    yn = {m: m17.bright_fringe_numeric(m) for m in (1, 3, 5)}
    m = 1
    f = lambda y: np.hypot(L, y + d / 2) - np.hypot(L, y - d / 2) - m * lam  # noqa: E731
    lo, hi = 0.0, 3 * m * lam * L / d + 1e-9
    rows = []
    for k in range(7):
        mid = 0.5 * (lo + hi)
        v = f(mid)
        rows.append([str(k + 1), fx(lo * 1e3, 4), fx(hi * 1e3, 4), fx(mid * 1e3, 4), fx(v * 1e9, 2), r"<0" if v < 0 else r">0"])
        if v < 0:
            lo = mid
        else:
            hi = mid
    tab = _tab("rrrrrc", [[r"k", r"y_{lo}\,(\mathrm{mm})", r"y_{hi}\,(\mathrm{mm})", r"y_{mid}\,(\mathrm{mm})", r"f\,(\mathrm{nm})", r"\mathrm{signo}"]] + rows)
    return dict(
        explanation=[
            "En el experimento de Young dos rendijas estrechas actúan como fuentes coherentes de ondas cilíndricas. En cada punto de la pantalla las dos ondas se suman con una diferencia de fase que depende de la diferencia de camino $\\Delta r = r_2 - r_1$: si es un número entero de longitudes de onda hay interferencia constructiva (franja brillante) y si es un semientero, destructiva (franja oscura).",
            "Para una pantalla lejana ($L\\gg d$) los rayos son casi paralelos y $\\Delta r\\approx d\\sin\\theta$, de modo que las franjas brillantes están en $d\\sin\\theta = m\\lambda$. La intensidad tiene forma $I = 4I_0\\cos^2(\\pi d\\sin\\theta/\\lambda)$ y las franjas están casi igualmente espaciadas con separación $\\lambda L/d$.",
            "Con $\\lambda = 600$ nm, $d = 0.25$ mm y $L = 1.5$ m las franjas distan unos milímetros, perfectamente visibles. El resultado muestra que la luz se comporta como onda: si fuera un chorro de partículas se verían dos manchas, no un patrón alternado.",
            "La fórmula exacta $y_m = L\\tan(\\arcsin(m\\lambda/d))$ usa el ángulo real de cada franja; la aproximación $y_m\\approx m\\lambda L/d$ solo vale para ángulos pequeños. El cálculo numérico no hace esa aproximación y usa las distancias exactas a cada rendija.",
        ],
        analytic_steps=[
            ("Condición de máximo", "Los rayos desde las dos rendijas hacia un punto lejano a ángulo $\\theta$ difieren en $d\\sin\\theta$; para interferencia constructiva esa diferencia es un múltiplo de $\\lambda$.",
             [r"d\sin\theta_m = m\lambda\ \Rightarrow\ \sin\theta_m = \dfrac{m\lambda}{d}"]),
            ("Ángulo del orden m", "Se sustituyen $\\lambda = 600$ nm y $d = 0.25$ mm.",
             [rf"\sin\theta_m = \dfrac{{m\,({sci(lam,3)})}}{{{sci(d,3)}}} = {lam/d:.5f}\,m",
              rf"\theta_1 = \arcsin({lam/d:.5f}) = {np.degrees(np.arcsin(lam/d)):.5f}^\circ,\qquad \theta_3 = \arcsin({3*lam/d:.5f}) = {np.degrees(np.arcsin(3*lam/d)):.5f}^\circ"]),
            ("Posición sobre la pantalla", "La franja está a una altura $y_m = L\\tan\\theta_m$ sobre el eje central, con $L=1.5$ m.",
             [rf"y_1 = 1.5\tan({np.degrees(np.arcsin(lam/d)):.5f}^\circ) = {ya[1]*1e3:.5f}\ \mathrm{{mm}}",
              rf"y_3 = 1.5\tan({np.degrees(np.arcsin(3*lam/d)):.5f}^\circ) = {ya[3]*1e3:.5f}\ \mathrm{{mm}},\qquad y_5 = {ya[5]*1e3:.5f}\ \mathrm{{mm}}"]),
            ("Espaciado entre franjas", "Para ángulos pequeños $\\tan\\theta\\approx\\sin\\theta$ y las franjas quedan igualmente separadas.",
             [rf"\Delta y \approx \dfrac{{\lambda L}}{{d}} = \dfrac{{({sci(lam,3)})(1.5)}}{{{sci(d,3)}}} = {sp*1e3:.4f}\ \mathrm{{mm}}"]),
            ("Intensidad en la pantalla", "La fase relativa de las dos ondas es $\\pi d\\sin\\theta/\\lambda$; en el máximo $m=1$ vale $\\pi$ y $\\cos^2 = 1$.",
             [rf"I(y_1)/I_{{max}} = \cos^2\!\left(\pi\cdot{lam/d*d/lam:.1f}\right) = {float(m17.intensity_analytic(ya[1])):.5f}"]),
        ],
        numeric_steps=[
            ("Diferencia de camino exacta", "Con las rendijas en $y=\\pm d/2$, las distancias exactas a un punto $y$ de la pantalla son $r_{1,2} = \\sqrt{L^2+(y\\mp d/2)^2}$. Un máximo cumple $r_2-r_1 = m\\lambda$ sin aproximar nada.",
             [r"f(y) = \sqrt{L^2+(y+d/2)^2} - \sqrt{L^2+(y-d/2)^2} - m\lambda = 0"]),
            ("Intervalo de búsqueda", "La diferencia de camino crece con $y$, de modo que $f$ cambia de signo una sola vez. Se toma $y\\in[0,\\ 3m\\lambda L/d]$, tres veces el valor aproximado.",
             [rf"y_{{lo}} = 0,\qquad y_{{hi}} = 3(1){sci(sp,4)} = {hi*1e3:.5f}\ \mathrm{{mm}}"]),
            ("Primera bisección (m = 1)", "El punto medio es $y_{mid}=" + f"{0.5*(0+3*lam*L/d+1e-9)*1e3:.4f}" + "$ mm. Se calcula $f$ y se conserva la mitad donde cambia de signo.",
             [rf"f(y_{{mid}}) = {rows[0][4]}\ \mathrm{{nm}}\ \ ({'<' if rows[0][5]=='<0' else '>'}0)\ \Rightarrow\ {'y_{lo}\\leftarrow y_{mid}' if rows[0][5]=='<0' else 'y_{hi}\\leftarrow y_{mid}'}"]),
            ("Siete primeras iteraciones", "La diferencia $f$ se muestra en nanómetros. El intervalo se reduce a la mitad en cada paso.",
             [tab]),
            ("Parada", "El código itera 100 veces; el ancho final es $\\sim10^{-30}$ m, así que el límite es la precisión de máquina del cálculo de $f$.",
             [rf"y_1^{{num}} = {yn[1]*1e3:.7f}\ \mathrm{{mm}},\qquad y_1^{{anal}} = {ya[1]*1e3:.7f}\ \mathrm{{mm}}"]),
            ("Resultados para m = 1, 3, 5", "Las pequeñas diferencias vienen de que la fórmula analítica usa $d\\sin\\theta$ (rayos paralelos) y la numérica la diferencia de camino exacta a distancia finita.",
             [_tab("rrrr", [[r"m", r"y^{num}\,(\mathrm{mm})", r"y^{anal}\,(\mathrm{mm})", r"|\Delta|\,(\mathrm{mm})"]] +
                   [[str(k), fx(yn[k]*1e3, 6), fx(ya[k]*1e3, 6), sci(abs(yn[k]-ya[k])*1e3, 2)] for k in (1, 3, 5)])]),
        ],
    )


def _film() -> dict:
    n0, nf, ns, t, lam0 = 1.0, m17.N_FILM, m17.N_SUB, m17.T_FILM, m17.LAM0
    r0 = m17.bare_reflectance()
    r01 = (n0 - nf) / (n0 + nf)
    r12 = (nf - ns) / (nf + ns)
    lams = (450e-9, 550e-9, 650e-9)
    R = {l: float(m17.reflectance_analytic(l)) for l in lams}
    Rm = {l: float(m17.reflectance_matrix(l)[0]) for l in lams}
    l = 550e-9
    dl = 2 * np.pi * nf * t / l
    e2 = np.exp(2j * dl)
    num = r01 + r12 * e2
    den = 1 + r01 * r12 * e2
    M = np.array([[np.cos(dl), 1j * np.sin(dl) / nf], [1j * nf * np.sin(dl), np.cos(dl)]])
    b, c = M @ np.array([1.0, ns])
    rr = (b - c) / (b + c)
    l2 = 450e-9
    d2 = 2 * np.pi * nf * t / l2
    M2 = np.array([[np.cos(d2), 1j * np.sin(d2) / nf], [1j * nf * np.sin(d2), np.cos(d2)]])
    b2, c2 = M2 @ np.array([1.0, ns])
    r2_ = (b2 - c2) / (b2 + c2)

    def cx(z):
        return f"({z.real:.5f}{z.imag:+.5f}i)"
    return dict(
        explanation=[
            "Cuando la luz incide sobre una película delgada transparente se refleja parcialmente en la cara superior y en la inferior. Las dos ondas reflejadas interfieren, y que el resultado sea refuerzo o cancelación depende de la diferencia de camino dentro de la película y de los cambios de fase en la reflexión (media vuelta, $\\pi$, cuando se refleja en un medio de mayor índice).",
            "Un recubrimiento antirreflejo aprovecha esto. Si el índice de la película está entre el del aire y el del vidrio, ambas reflexiones sufren el mismo cambio de fase de $\\pi$; con un espesor de un cuarto de longitud de onda dentro de la película, el camino de ida y vuelta es $\\lambda/2$ y las dos ondas llegan en oposición de fase, cancelándose casi por completo.",
            "La cancelación es perfecta solo para la longitud de onda de diseño y si las dos amplitudes son iguales (aquí $n_f=\\sqrt{n_s}\\approx 1.23$ lo sería; con MgF$_2$ ($1.38$) es una buena aproximación). Para otros colores la reflectancia crece, por eso las lentes recubiertas tienen un tinte purpúreo: reflejan más de los extremos del espectro visible.",
            "El cálculo exacto suma todas las reflexiones múltiples dentro de la película, lo que conduce a la fórmula de Airy. El método de la matriz de transferencia obtiene el mismo resultado con álgebra de matrices $2\\times2$ y se generaliza sin esfuerzo a pilas de muchas capas.",
        ],
        analytic_steps=[
            ("Reflectancia del vidrio desnudo", "A incidencia normal, la fracción reflejada en una interfaz aire-vidrio es el cuadrado del coeficiente de Fresnel.",
             [rf"R_0 = \left(\dfrac{{n_s-1}}{{n_s+1}}\right)^2 = \left(\dfrac{{{ns}-1}}{{{ns}+1}}\right)^2 = {r0:.6f} = {r0*100:.4f}\,\%"]),
            ("Espesor de la película", "Un cuarto de longitud de onda medida dentro de la película: $t=\\lambda_0/4n_f$.",
             [rf"t = \dfrac{{{sci(lam0,3)}}}{{4({nf})}} = {t*1e9:.4f}\ \mathrm{{nm}}"]),
            ("Coeficientes de reflexión de cada interfaz", "Aire-película ($r_{01}$) y película-vidrio ($r_{12}$); ambos son negativos porque el índice aumenta en ambos casos.",
             [rf"r_{{01}} = \dfrac{{1-{nf}}}{{1+{nf}}} = {r01:.5f},\qquad r_{{12}} = \dfrac{{{nf}-{ns}}}{{{nf}+{ns}}} = {r12:.5f}"]),
            ("Fase de ida en la película", "El desfase de una pasada es $\\delta = 2\\pi n_f t/\\lambda$ y la ida y vuelta introduce $e^{2i\\delta}$. A $\\lambda = 550$ nm:",
             [rf"\delta = \dfrac{{2\pi({nf})({t*1e9:.4f}\ \mathrm{{nm}})}}{{550\ \mathrm{{nm}}}} = {dl:.5f}\ \mathrm{{rad}} = \dfrac{{\pi}}{{2}},\qquad e^{{2i\delta}} = {e2.real:.5f}"]),
            ("Fórmula de Airy", "El coeficiente de reflexión total suma las reflexiones múltiples (serie geométrica) y la reflectancia es su módulo cuadrado.",
             [r"r = \dfrac{r_{01}+r_{12}e^{2i\delta}}{1+r_{01}r_{12}e^{2i\delta}},\qquad R = |r|^2",
              rf"r = \dfrac{{{r01:.5f} + ({r12:.5f})({e2.real:.1f})}}{{1 + ({r01:.5f})({r12:.5f})({e2.real:.1f})}} = \dfrac{{{num.real:.5f}}}{{{den.real:.5f}}} = {(num/den).real:.5f}"]),
            ("Reflectancia con película a 550 nm", "Se eleva al cuadrado.",
             [rf"R(550\ \mathrm{{nm}}) = ({(num/den).real:.5f})^2 = {R[550e-9]:.6f} = {R[550e-9]*100:.4f}\,\%"]),
            ("Otros colores", "Para 450 y 650 nm el espesor ya no es exactamente $\\lambda/4$ y la cancelación es parcial.",
             [rf"R(450) = {R[450e-9]*100:.4f}\,\%,\qquad R(650) = {R[650e-9]*100:.4f}\,\%,\qquad R_0 = {r0*100:.4f}\,\%"]),
        ],
        numeric_steps=[
            ("Matriz característica de la capa", "En el método de la matriz de transferencia cada capa de índice $n_f$ y fase $\\delta$ se describe por una matriz $2\\times2$ que relaciona los campos en sus dos caras.",
             [r"M = \begin{pmatrix}\cos\delta & i\sin\delta/n_f\\ i\,n_f\sin\delta & \cos\delta\end{pmatrix}"]),
            ("Evaluar a 550 nm", "Con $\\delta=\\pi/2$ los términos diagonales se anulan.",
             [rf"M = \begin{{pmatrix}}{M[0,0].real:.5f}&{M[0,1].imag:.5f}\,i\\ {M[1,0].imag:.5f}\,i&{M[1,1].real:.5f}\end{{pmatrix}}"]),
            ("Condiciones en el sustrato", "Se multiplica la matriz por el vector del sustrato $(1,\\ n_s)^T$ para obtener $(B,C)$, que son los campos equivalentes en la entrada.",
             [rf"\begin{{pmatrix}}B\\C\end{{pmatrix}} = M\begin{{pmatrix}}1\\{ns}\end{{pmatrix}} = \begin{{pmatrix}}{cx(b)}\\ {cx(c)}\end{{pmatrix}}"]),
            ("Coeficiente de reflexión", "Con índice del aire $n_0=1$.",
             [rf"r = \dfrac{{n_0B-C}}{{n_0B+C}} = \dfrac{{{cx(b-c)}}}{{{cx(b+c)}}} = {rr.real:.5f},\qquad R=|r|^2 = {abs(rr)**2*100:.4f}\,\%"]),
            ("Mismo cálculo a 450 nm", "El desfase cambia a $\\delta = 2\\pi n_f t/\\lambda$ con $\\lambda=450$ nm y las componentes de $M$ ya no son simples.",
             [rf"\delta = {d2:.5f}\ \mathrm{{rad}},\quad B={cx(b2)},\ C={cx(c2)}",
              rf"r = {cx(r2_)},\qquad R = {abs(r2_)**2*100:.4f}\,\%"]),
            ("Resumen y concordancia con Airy", "El código repite el procedimiento para cada longitud de onda y compara con la fórmula de Airy; ambos son algebraicamente equivalentes, así que la diferencia es de redondeo.",
             [_tab("rrrr", [[r"\lambda\,(\mathrm{nm})", r"R_{matriz}\,(\%)", r"R_{Airy}\,(\%)", r"|\Delta|"]] +
                   [[str(int(x*1e9)), fx(Rm[x]*100, 6), fx(R[x]*100, 6), sci(abs(Rm[x]-R[x])*100, 2)] for x in lams])]),
        ],
    )


# ===================================================================== Capitulo 18
def _grating() -> dict:
    lam, a, d, N = m18.LAM, m18.A_SLIT, m18.D_GRAT, m18.N_SLITS
    ms, ang = m18.principal_maxima_analytic()
    num = m18.principal_maxima_numeric()
    th = np.linspace(0, np.deg2rad(89.0), 20001)
    dth = np.degrees(th[1] - th[0])
    j1 = int(np.argmin(np.abs(np.degrees(th) - ang[1])))
    I = m18.grating_numeric(np.sin(th[j1 - 1:j1 + 2]), lam, 1e-9, d, N, 1)
    gam = np.pi * d * np.sin(np.deg2rad(ang[1])) / lam
    # N=5 phasor example: first minimum
    gmin = np.pi / N
    thmin = np.degrees(np.arcsin(gmin * lam / (np.pi * d)))
    b3 = np.pi * a * (3 * lam / d) / lam
    return dict(
        explanation=[
            "La difracción es la flexión y el ensanchamiento que sufre una onda al pasar por una abertura de tamaño comparable a su longitud de onda. Cada punto de una rendija de ancho $a$ se comporta como una fuente de ondas, y la suma de todas produce una envolvente $\\mathrm{sinc}^2(\\pi a\\sin\\theta/\\lambda)$ con ceros en $a\\sin\\theta = m\\lambda$.",
            "Una rejilla de $N$ rendijas separadas por $d$ añade interferencia entre rendijas. Los máximos principales ocurren cuando todas las rendijas están en fase, $d\\sin\\theta = m\\lambda$, y son tanto más estrechos cuanto mayor es $N$; entre dos máximos principales hay $N-1$ mínimos y $N-2$ máximos secundarios débiles.",
            "El patrón final es el producto de ambos efectos: las posiciones de los máximos las fija $d$, y la envolvente de una rendija $a$ modula su intensidad. Como aquí $d/a = 3$, el orden $m=3$ cae exactamente sobre el primer cero de la envolvente y desaparece (orden suprimido o faltante).",
            "Esta es la base de la espectroscopia: como $\\sin\\theta_m = m\\lambda/d$, cada longitud de onda sale a un ángulo distinto y la rejilla separa los colores. Con $\\lambda = 550$ nm y $d = 6$ µm solo hay órdenes hasta $m = 10$, pues $\\sin\\theta$ no puede superar 1.",
        ],
        analytic_steps=[
            ("Condición de máximo principal", "Para que las $N$ ondas se sumen en fase, la diferencia de camino entre rendijas adyacentes debe ser un número entero de longitudes de onda.",
             [r"d\sin\theta_m = m\lambda\ \Rightarrow\ \theta_m = \arcsin\dfrac{m\lambda}{d}"]),
            ("Órdenes posibles", "Como $\\sin\\theta\\le1$, $m\\le d/\\lambda$.",
             [rf"m_{{max}} \le \dfrac{{d}}{{\lambda}} = \dfrac{{{sci(d,3)}}}{{{sci(lam,3)}}} = {d/lam:.4f}\ \Rightarrow\ m = 0,1,\ldots,{int(ms[-1])}"]),
            ("Ángulos de los órdenes 1, 2 y 4", "Se sustituye en cada orden.",
             [rf"\theta_{{{k}}} = \arcsin\dfrac{{{k}({sci(lam,3)})}}{{{sci(d,3)}}} = \arcsin({k*lam/d:.5f}) = {ang[k]:.4f}^\circ" for k in (1, 2, 4)]),
            ("Orden suprimido", "La envolvente de una rendija se anula cuando $a\\sin\\theta = n\\lambda$. Se comparan las dos condiciones.",
             [r"\dfrac{d\sin\theta = m\lambda}{a\sin\theta = n\lambda}\ \Rightarrow\ \dfrac{m}{n} = \dfrac{d}{a}",
              rf"\dfrac{{d}}{{a}} = \dfrac{{{sci(d,3)}}}{{{sci(a,3)}}} = {d/a:.1f}\ \Rightarrow\ m = 3,6,9\ \text{{coinciden con ceros}}"]),
            ("Comprobación para m = 3", "En $\\theta_3$ el argumento de la envolvente es exactamente $\\pi$, donde $\\mathrm{sinc}$ vale cero.",
             [rf"\beta_3 = \dfrac{{\pi a\sin\theta_3}}{{\lambda}} = \dfrac{{\pi({sci(a,3)})({3*lam/d:.4f})}}{{{sci(lam,3)}}} = {b3:.5f} = \pi,\qquad \mathrm{{sinc}}^2\beta_3 = {sci(float(m18.single_slit_analytic(3*lam/d)),3)}"]),
            ("Estructura entre máximos", "Con $N=5$ el numerador $\\sin N\\gamma$ se anula antes que el denominador: hay 4 mínimos entre máximos, el primero en $\\gamma=\\pi/5$.",
             [rf"\gamma = \dfrac{{\pi d\sin\theta}}{{\lambda}}=\dfrac{{\pi}}{{5}}\ \Rightarrow\ \sin\theta = \dfrac{{\lambda}}{{5d}} = {lam/(5*d):.5f},\ \theta = {thmin:.4f}^\circ"]),
        ],
        numeric_steps=[
            ("Modelo de suma coherente", "Cada rendija se divide en fuentes puntuales y la amplitud en la dirección $\\theta$ es la suma de las fases $e^{ikx\\sin\\theta}$, con $k=2\\pi/\\lambda$ y $x$ la posición de cada fuente. La intensidad relativa es $|A|^2$ con $A=\\frac1M\\sum e^{ikx_j\\sin\\theta}$.",
             [r"A(\theta) = \dfrac1M\sum_{j=1}^{M} e^{\,ik\,x_j\sin\theta},\qquad I(\theta)=|A|^2"]),
            ("Reducción para buscar máximos", "Para localizar los máximos principales el código usa una fuente puntual por rendija ($M=N=5$, posiciones $x_j=(j-2)d$ con $j=0..4$), pues los máximos de interferencia no dependen del ancho. La envolvente de una rendija se omite.",
             [r"x_j = \{-2d,-d,0,d,2d\}"]),
            ("Evaluación a mano en θ = 0", "Con $\\theta=0$ todas las fases son 0 y las cinco contribuciones se suman completamente.",
             [r"A(0) = \dfrac15(1+1+1+1+1) = 1,\qquad I(0)=1"]),
            ("Evaluación cerca del primer orden", "En la malla (20 001 ángulos entre 0° y 89°, paso " + f"{dth:.5f}" + "°) el punto más cercano a $\\theta_1 = " + f"{ang[1]:.4f}" + "^\\circ$ es $\\theta_j = " + f"{np.degrees(th[j1]):.5f}" + "^\\circ$. Allí $\\gamma\\approx 2\\pi$, así que todas las fases son múltiplos casi enteros de $2\\pi$.",
             [rf"I(\theta_{{j-1}}) = {I[0]:.6f},\quad I(\theta_j) = {I[1]:.6f},\quad I(\theta_{{j+1}}) = {I[2]:.6f}"]),
            ("Criterio de máximo local", "Un nodo $j$ es un máximo si $I_j > I_{j-1}$, $I_j\\ge I_{j+1}$ e $I_j>0.9$ (esto descarta los máximos secundarios, que no pasan de 0.05 aproximadamente). Se agrega el orden 0 en $\\theta=0$.",
             [rf"I_{{j-1}}<I_j\ge I_{{j+1}}\ \text{{y}}\ I_j>0.9\ \Rightarrow\ \theta_{{1,num}} = {num[1]:.5f}^\circ"]),
            ("Resultados y error", "El error está limitado por el paso de la malla ($\\Delta\\theta = " + f"{dth:.4f}" + "^\\circ$), pues el máximo es un nodo de la malla.",
             [_tab("rrrr", [[r"m", r"\theta_{num}\,(^\circ)", r"\theta_{anal}\,(^\circ)", r"|\Delta|\,(^\circ)"]] +
                   [[str(k), fx(num[k], 5), fx(ang[k], 5), fx(abs(num[k]-ang[k]), 5)] for k in (1, 2, 4)])]),
        ],
    )


def _pol() -> dict:
    t30, t45, t60 = np.pi / 6, np.pi / 4, np.pi / 3
    e0 = np.array([1.0, 0.0])
    p0, p1, p2 = m18.polarizer_jones(0.0), m18.polarizer_jones(t30), m18.polarizer_jones(np.pi / 2)
    e1 = p0 @ e0
    e2 = p1 @ e1
    e3 = p2 @ e2

    def mat(m_):
        return rf"\begin{{pmatrix}}{m_[0,0]:.4f}&{m_[0,1]:.4f}\\{m_[1,0]:.4f}&{m_[1,1]:.4f}\end{{pmatrix}}"
    ia = {k: m18.three_polarizers_analytic(v) for k, v in (("30", t30), ("45", t45))}
    ij = {k: m18.jones_intensity([0, v, np.pi / 2]) for k, v in (("30", t30), ("45", t45))}
    ma = float(m18.malus_analytic(t60))
    mj = m18.jones_intensity([0, t60])
    return dict(
        explanation=[
            "La luz es una onda transversal: su campo eléctrico oscila perpendicular a la dirección de propagación. Un polarizador lineal solo deja pasar la componente del campo paralela a su eje de transmisión y absorbe la perpendicular.",
            "Si luz polarizada linealmente incide sobre un polarizador cuyo eje forma el ángulo $\\theta$ con la polarización, la amplitud transmitida es $E_0\\cos\\theta$. Como la intensidad es proporcional al cuadrado de la amplitud, $I = I_0\\cos^2\\theta$ (ley de Malus).",
            "Dos polarizadores cruzados (0° y 90°) no transmiten nada. Pero al intercalar un tercero a un ángulo $\\theta$ la luz vuelve a pasar: el segundo polarizador gira la polarización hacia el tercero. Es un resultado contraintuitivo, porque añadir un elemento absorbente aumenta la transmisión; la intensidad máxima es una cuarta parte de la inicial y ocurre a 45°.",
            "El formalismo de Jones representa la polarización con un vector de dos componentes complejas y cada elemento óptico con una matriz $2\\times2$. La matriz de un polarizador lineal es el proyector sobre su eje, $P(\\theta)=\\hat u\\hat u^T$, y la intensidad es $\\vec E^\\dagger\\vec E$.",
        ],
        analytic_steps=[
            ("Primera etapa: polarizador a 0°", "La luz ya está polarizada a 0°, por lo que el primer polarizador no reduce la intensidad (se normaliza $I_0=1$).",
             [r"E_1 = E_0,\qquad I_1 = I_0"]),
            ("Segunda etapa: polarizador a θ", "La ley de Malus aplicada entre la polarización 0° y el eje a $\\theta$.",
             [r"I_2 = I_1\cos^2\theta"]),
            ("Tercera etapa: polarizador a 90°", "Ahora la luz está polarizada a $\\theta$ y el último eje está a $90^\\circ-\\theta$ de ella.",
             [r"I_3 = I_2\cos^2(90^\circ-\theta) = I_2\sin^2\theta"]),
            ("Transmisión total", "Se combinan y se usa $\\sin2\\theta = 2\\sin\\theta\\cos\\theta$.",
             [r"\dfrac{I_3}{I_0} = \cos^2\theta\,\sin^2\theta = \tfrac14\sin^2 2\theta"]),
            ("Máximo", "El seno al cuadrado alcanza 1 en $2\\theta=90^\\circ$.",
             [rf"\theta = 45^\circ:\ \dfrac{{I_3}}{{I_0}} = \cos^2 45^\circ\sin^2 45^\circ = ({np.cos(t45)**2:.4f})({np.sin(t45)**2:.4f}) = {m18.three_polarizers_analytic(t45):.4f}"]),
            ("Valores numéricos a 30° y 60°", "A 30° la transmisión es menor que el máximo; la ley de Malus simple a 60° da el mismo valor que la expresión doble.",
             [rf"\theta=30^\circ:\ I_3/I_0=({np.cos(t30)**2:.4f})({np.sin(t30)**2:.4f})={ia['30']:.4f}",
              rf"\text{{Malus a }}60^\circ:\ I/I_0=\cos^2 60^\circ={ma:.4f}"]),
        ],
        numeric_steps=[
            ("Matriz de Jones del polarizador", "Para un eje a ángulo $\\theta$ la matriz es el proyector sobre el vector unitario $(\\cos\\theta,\\sin\\theta)$.",
             [r"P(\theta)=\begin{pmatrix}\cos^2\theta&\cos\theta\sin\theta\\ \cos\theta\sin\theta&\sin^2\theta\end{pmatrix}"]),
            ("Matrices de la cadena 0°, 30°, 90°", "Se evalúan numéricamente las tres matrices.",
             [rf"P(0^\circ)={mat(p0)},\quad P(30^\circ)={mat(p1)},\quad P(90^\circ)={mat(p2)}"]),
            ("Aplicar sucesivamente al campo", "El campo incidente es $\\vec E_0=(1,0)^T$ (polarizado a 0°). Se multiplica por cada matriz en el orden en que la luz las atraviesa.",
             [rf"\vec E_1 = P(0^\circ)\vec E_0 = ({e1[0]:.4f},\ {e1[1]:.4f})",
              rf"\vec E_2 = P(30^\circ)\vec E_1 = ({e2[0]:.4f},\ {e2[1]:.4f})",
              rf"\vec E_3 = P(90^\circ)\vec E_2 = ({e3[0]:.4f},\ {e3[1]:.4f})"]),
            ("Intensidad", "La intensidad es el producto escalar del campo con su conjugado: $\\vec E^\\dagger\\vec E$.",
             [rf"I = |E_{{3x}}|^2+|E_{{3y}}|^2 = {e3[0]**2:.4f}+{e3[1]**2:.4f} = {ij['30']:.8f}"]),
            ("Comparación a 30° y 45°", "El resultado coincide con $\\cos^2\\theta\\sin^2\\theta$ (3/16 = 0.1875 a 30°).",
             [_tab("rrrr", [[r"\theta", r"I_{Jones}", r"I_{anal}", r"|\Delta|"]] +
                   [[k + r"^\circ", fx(ij[k], 8), fx(ia[k], 8), sci(abs(ij[k]-ia[k]), 2)] for k in ("30", "45")])]),
            ("Ley de Malus a 60°", "Con solo dos polarizadores (0° y 60°) el campo transmitido es $P(60^\\circ)(1,0)^T=(\\cos^2 60^\\circ,\\ \\cos60^\\circ\\sin60^\\circ)$.",
             [rf"I = {mj:.8f},\qquad \cos^2 60^\circ = {ma:.8f},\qquad |\Delta| = {sci(abs(mj-ma),2)}"]),
        ],
    )


DETAIL = {
    "YeeFDTD": _yee(),
    "PoyntingDisplacement": _poynting(),
    "SnellFermat": _snell(),
    "PrismDispersion": _prism(),
    "ThinLensImage": _lens(),
    "TwoLensSystem": _twolens(),
    "YoungSlits": _young(),
    "ThinFilmCoating": _film(),
    "SlitAndGrating": _grating(),
    "PolarizersMalus": _pol(),
}
