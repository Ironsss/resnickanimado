"""Concepto y desarrollo analitico general (capitulos 17 a 20): VenturiTube ... BeatsAndPipes.

Cada entrada es DETAIL[id] = dict(concept=[(titulo, prosa, [ecuaciones TeX]), ...]). Las cifras se calculan con physics/.
"""
from __future__ import annotations

import numpy as np

from physics import ch17_fluid_dynamics as c17
from physics import ch18_oscillations as c18
from physics import ch19_waves as c19
from physics import ch20_sound as c20
from content.texfmt import fx, sci


# =============================================================== VenturiTube
def _venturi() -> dict:
    a1, a2 = c17.areas()
    an = c17.venturi_analytic()
    rho = c17.RHO
    dp = c17.P1 - an["p2"]
    qa = c17.flow_from_pressure_drop_analytic(dp)
    beta = a2 / a1
    d1, d2 = c17.D1, c17.D2
    re1 = rho * an["v1"] * d1 / 1.0e-3
    eu = dp / (0.5 * rho * an["v1"] ** 2)
    concept = [
        ("Qué se estudia y por qué importa",
         r"Cuando un líquido fluye por una tubería de sección variable, la velocidad y la presión cambian de un punto a otro aunque el caudal sea el mismo. El tubo de Venturi aprovecha ese acoplamiento: midiendo la caída de presión entre la sección ancha y la garganta se obtiene el caudal sin interrumpir el flujo. Se usa en redes de agua, en carburadores, en aspiradores de laboratorio y en medición de gases industriales. El mismo principio explica por qué una ventana abierta puede arrastrar una cortina hacia afuera cuando sopla viento.",
         []),
        ("Magnitudes y convenciones",
         r"Se emplean la densidad $\rho$ (kg/m³, unos $10^3$ para el agua), la presión $p$ (Pa), la velocidad media $v$ (m/s), el área $A$ (m²) y el caudal volumétrico $Q$ (m³/s). Los subíndices 1 y 2 designan la sección ancha y la garganta; $g$ es la aceleración de la gravedad y $z$ la cota vertical. Las presiones en juego son manométricas o absolutas indistintamente, porque solo importan diferencias. En redes domésticas las velocidades son de 1 a 3 m/s y las caídas de presión del orden de $10^3$ a $10^5$ Pa.",
         []),
        ("Intuición y analogías",
         r"Un río que atraviesa un desfiladero se acelera porque el mismo volumen por segundo debe pasar por una sección menor. Esa aceleración necesita una fuerza neta en la dirección del flujo, y en un fluido sin fricción esa fuerza solo puede venir de que la presión aguas arriba sea mayor que en la garganta. Por eso la presión baja donde el fluido va más rápido: la energía no se crea, se redistribuye entre energía de presión y energía cinética.",
         []),
        ("Fluido ideal y ecuaciones de partida",
         r"Se modela el líquido como un medio continuo, incompresible ($\rho$ constante) y sin viscosidad. La conservación de la masa y la segunda ley de Newton aplicada a un elemento de fluido dan la ecuación de continuidad y la ecuación de Euler. Se supone además flujo estacionario ($\partial/\partial t = 0$) y fuerzas de volumen conservativas, $\vec g = -\nabla(gz)$.",
         [r"\nabla\cdot\vec v = 0",
          r"\rho\left[\dfrac{\partial\vec v}{\partial t} + (\vec v\cdot\nabla)\vec v\right] = -\nabla p - \rho g\,\hat z"]),
        ("Continuidad en forma integral",
         r"Se integra $\nabla\cdot\vec v = 0$ sobre el volumen limitado por dos secciones transversales y las paredes del tubo. Por el teorema de la divergencia el flujo neto a través de la frontera es nulo, y como la velocidad es tangente a las paredes solo contribuyen las dos secciones. Con velocidad media uniforme en cada sección resulta que el caudal es el mismo en ambas.",
         [r"\oint_S \vec v\cdot d\vec S = 0 \;\Rightarrow\; -v_1A_1 + v_2A_2 = 0",
          r"Q = A_1v_1 = A_2v_2,\qquad \dfrac{v_2}{v_1} = \dfrac{A_1}{A_2}"]),
        ("Bernoulli a partir de la ecuación de Euler",
         r"Se usa la identidad vectorial $(\vec v\cdot\nabla)\vec v = \nabla(v^2/2) - \vec v\times(\nabla\times\vec v)$. En régimen estacionario se proyecta la ecuación de Euler sobre una línea de corriente, es decir, se multiplica escalarmente por $\vec v$; el término con el rotacional desaparece porque $\vec v\times\vec\omega$ es perpendicular a $\vec v$. Queda una derivada a lo largo de la línea que se anula, de modo que la cantidad entre paréntesis es constante a lo largo de ella.",
         [r"\vec v\cdot\left[\rho\nabla\dfrac{v^2}{2} + \nabla p + \rho g\nabla z\right] = 0 \;\Rightarrow\; \dfrac{d}{ds}\left(\dfrac{\rho v^2}{2} + p + \rho g z\right) = 0",
          r"p + \tfrac12\rho v^2 + \rho g z = \text{constante a lo largo de una línea de corriente}"]),
        ("Aplicación a una sección variable y parámetro adimensional",
         r"Para un tubo horizontal ($z_1 = z_2$) se escribe Bernoulli entre las secciones 1 y 2 y se elimina $v_2$ con la continuidad. El resultado depende solo de la razón de áreas $\beta = A_2/A_1 < 1$ y de la presión dinámica de entrada. La caída de presión crece como $v_1^2$ y diverge al cerrar la garganta ($\beta\to0$), lo que fija el límite de validez del modelo.",
         [r"p_1 - p_2 = \tfrac12\rho\,(v_2^2 - v_1^2) = \tfrac12\rho v_1^2\left(\dfrac{1}{\beta^2} - 1\right)",
          r"\mathrm{Eu} \equiv \dfrac{\Delta p}{\tfrac12\rho v_1^2} = \dfrac{1}{\beta^2} - 1,\qquad \beta = \dfrac{A_2}{A_1} = \left(\dfrac{d_2}{d_1}\right)^2"]),
        ("Problema inverso: el caudal a partir de la caída de presión",
         r"En un medidor real se conoce $\Delta p$ y se busca $Q$. Se sustituye $v_1 = Q/A_1$ y $v_2 = Q/A_2$ en la ecuación anterior y se despeja $Q$, lo cual es posible porque $\Delta p$ es monótona en $Q^2$. Las pérdidas por fricción y la contracción del chorro se absorben en un coeficiente de descarga empírico $C_d \lesssim 1$ que depende del número de Reynolds.",
         [r"\Delta p = \tfrac12\rho Q^2\left(\dfrac{1}{A_2^2} - \dfrac{1}{A_1^2}\right)",
          r"Q = C_d\,A_2\sqrt{\dfrac{2\,\Delta p}{\rho\,(1-\beta^2)}}"]),
        ("Resultado general",
         r"Para un fluido ideal en un tubo horizontal que pasa de la sección $A_1$ a la sección $A_2$, las velocidades, la presión en la garganta y el caudal inferido son los siguientes. Dimensionalmente $[\rho v^2] = \mathrm{kg\,m^{-3}\,m^2\,s^{-2}} = \mathrm{Pa}$ y $[A\sqrt{\Delta p/\rho}] = \mathrm{m^2\,m\,s^{-1}} = \mathrm{m^3/s}$, de modo que todas las expresiones son consistentes.",
         [r"v_1 = \dfrac{Q}{A_1},\qquad v_2 = \dfrac{Q}{A_2},\qquad p_2 = p_1 - \dfrac{\rho Q^2}{2}\left(\dfrac{1}{A_2^2} - \dfrac{1}{A_1^2}\right)",
          r"\boxed{\,Q = A_2\sqrt{\dfrac{2\,(p_1-p_2)}{\rho\,[\,1-(A_2/A_1)^2\,]}}\,}"]),
        ("Particularización al problema",
         rf"Con $d_1 = {d1}$ m, $d_2 = {d2}$ m se tiene $\beta = {beta:.2f}$, y con $Q = {c17.Q}$ m³/s, $p_1 = {c17.P1/1000:.0f}$ kPa y $\rho = {rho:.0f}$ kg/m³ la fórmula general reproduce las cifras de la escena. Además, el número de Euler vale $1/\beta^2 - 1 = {1/beta**2 - 1:.0f}$, de modo que la caída de presión es quince veces la presión dinámica de entrada.",
         [rf"v_1 = \dfrac{{{c17.Q}}}{{{sci(a1,5)}}} = {an['v1']:.3f}\ \mathrm{{m/s}},\qquad v_2 = \dfrac{{{c17.Q}}}{{{sci(a2,5)}}} = {an['v2']:.3f}\ \mathrm{{m/s}}",
          rf"p_2 = {sci(c17.P1,3)} - \dfrac{{{rho:.0f}\,({c17.Q})^2}}{{2}}\left(\dfrac{{1}}{{({sci(a2,5)})^2}} - \dfrac{{1}}{{({sci(a1,5)})^2}}\right) = {fx(an['p2'],1)}\ \mathrm{{Pa}} = {an['p2']/1000:.2f}\ \mathrm{{kPa}}",
          rf"Q = ({sci(a2,5)})\sqrt{{\dfrac{{2({fx(dp,1)})}}{{{rho:.0f}\,[1-{beta**2:.4f}]}}}} = {qa:.6f}\ \mathrm{{m^3/s}}"]),
        ("Errores comunes y límites de validez",
         rf"Un error frecuente es creer que la presión sube donde el fluido va más rápido, o aplicar Bernoulli entre puntos no unidos por una línea de corriente en flujo rotacional. También se confunde la velocidad de la sección con la velocidad en el eje, que es mayor cuando el perfil no es uniforme. El modelo falla si el número de Reynolds es bajo (con $\mu\approx10^{{-3}}$ Pa·s resulta $\mathrm{{Re}}\approx {sci(re1,2)}$ aquí, claramente turbulento pero aún bien descrito por $C_d\approx1$), si la presión en la garganta cae por debajo de la presión de vapor (cavitación) o si el fluido es compresible, caso en que se usa la ecuación de Bernoulli para gases con la densidad variable.",
         [r"\mathrm{Re} = \dfrac{\rho v d}{\mu},\qquad p_2 > p_{vap}\ \text{(condición para evitar cavitación)}"]),
    ]
    return dict(concept=concept)


# =============================================================== TorricelliDrain
def _torricelli() -> dict:
    A, a, h0, g = c17.A_TANK, c17.A_HOLE, c17.H0, c17.G
    te = c17.drain_time_analytic()
    v0 = float(c17.torricelli_speed(h0))
    kk = (a / A) * np.sqrt(g / 2)
    r = a / A
    corr = 1 / np.sqrt(1 - r**2) - 1
    concept = [
        ("Qué se estudia y por qué importa",
         r"El vaciado de un depósito por un orificio es uno de los problemas clásicos de la hidrodinámica: de él se obtiene la rapidez de salida del chorro y el tiempo que tarda en agotarse el contenido. Aplica al diseño de tanques de agua, buretas, clepsidras (relojes de agua), sistemas de drenaje y a la estimación de fugas. Además es un ejemplo sencillo de una ecuación diferencial no lineal con solución cerrada y con un tiempo finito de extinción.",
         []),
        ("Magnitudes y convenciones",
         r"Intervienen la altura del líquido sobre el orificio $h(t)$ (m), la sección del depósito $A$ y la del orificio $a$ (m²), la rapidez de salida $v$ (m/s), la gravedad $g = 9.81$ m/s² y el tiempo $t$ (s). La altura se mide desde el nivel del orificio hacia arriba y es positiva. La relación $a/A$ es adimensional y en los depósitos reales suele estar entre $10^{-4}$ y $10^{-2}$, lo que hace que el vaciado sea lento.",
         []),
        ("Intuición y analogías",
         r"El agua que sale con rapidez $\sqrt{2gh}$ es la misma rapidez que alcanzaría un cuerpo que cae libremente desde la altura $h$: la energía potencial de una porción de agua de la superficie se convierte en cinética al llegar al orificio. Como el nivel baja, la rapidez de salida disminuye y el vaciado se enlentece, pero en la raíz cuadrada que gobierna el proceso el nivel llega a cero en un tiempo finito, a diferencia de un decaimiento exponencial.",
         []),
        ("Bernoulli generalizado entre la superficie y el orificio",
         r"Para flujo irrotacional $\vec v = \nabla\phi$ de un fluido ideal incompresible, la ecuación de Euler se integra en todo el volumen y da la forma no estacionaria de Bernoulli. Se aplica entre la superficie libre (punto S, velocidad $v_s$, presión atmosférica) y el chorro (punto O, velocidad $v$, presión atmosférica), con $z_S - z_O = h$. Si el nivel varía lentamente el término $\partial\phi/\partial t$ es despreciable frente a los demás.",
         [r"\dfrac{\partial\phi}{\partial t} + \tfrac12 v^2 + \dfrac{p}{\rho} + gz = f(t)",
          r"p_{atm} + \tfrac12\rho v_s^2 + \rho g h = p_{atm} + \tfrac12\rho v^2"]),
        ("Ley de Torricelli con corrección por la razón de áreas",
         r"La continuidad entre la superficie y el orificio da $v_s = (a/A)\,v$, que se sustituye en la expresión anterior. Se despeja $v$ sin aproximar. Para $a\ll A$ el denominador tiende a 1 y se recupera la ley de Torricelli; el factor extra es la corrección por velocidad finita de la superficie.",
         [r"v^2\left[1 - \left(\dfrac{a}{A}\right)^2\right] = 2gh",
          r"v = \sqrt{\dfrac{2gh}{1-(a/A)^2}} \;\xrightarrow{\;a\ll A\;}\; v = \sqrt{2gh}"]),
        ("Balance de volumen para una sección cualquiera",
         r"En un intervalo $dt$ el volumen perdido por el depósito es $-A(h)\,dh$ y debe ser igual al que sale por el orificio, $a\,v\,dt$. Esto es válido para un depósito de forma arbitraria con área $A(h)$ a la altura $h$ (la hipótesis de la sección constante no es necesaria). Es una ecuación de variables separables y su integral da el tiempo de descenso entre dos niveles en términos de una cuadratura.",
         [r"A(h)\,\dfrac{dh}{dt} = -a\sqrt{2gh}",
          r"t(h_0\to h) = \dfrac{1}{a\sqrt{2g}}\int_{h}^{h_0}\dfrac{A(h')}{\sqrt{h'}}\,dh'"]),
        ("Integración exacta para sección constante",
         r"Con $A$ constante la cuadratura es elemental, ya que $\int h^{-1/2}dh = 2\sqrt h$. Se obtiene que $\sqrt h$ decrece linealmente con el tiempo, es decir, $h(t)$ es una parábola que toca a cero con pendiente nula. Con la escala de tiempo $\tau = (A/a)\sqrt{2h_0/g}$ la solución adimensional no depende de ningún otro parámetro.",
         [r"\dfrac{d\sqrt h}{dt} = -\dfrac{a}{A}\sqrt{\dfrac{g}{2}} \equiv -\kappa \;\Rightarrow\; \sqrt{h(t)} = \sqrt{h_0} - \kappa t",
          r"\dfrac{h}{h_0} = \left(1 - \dfrac{t}{\tau}\right)^2,\qquad \tau = \dfrac{\sqrt{h_0}}{\kappa} = \dfrac{A}{a}\sqrt{\dfrac{2h_0}{g}}"]),
        ("Casos límite y efectos reales",
         r"Si el depósito se ensancha hacia arriba el vaciado inicial es más lento; si se estrecha (como en una clepsidra bien diseñada, $A\propto\sqrt h$) el nivel baja a ritmo constante. Para orificios reales el chorro se contrae y el caudal es $Q = C_d\,a\sqrt{2gh}$ con $C_d\approx0.6$, que alarga el tiempo de vaciado en el factor $1/C_d$. El modelo ideal no incluye viscosidad ni tensión superficial.",
         [r"\tau_{real} = \dfrac{\tau}{C_d},\qquad A(h) = A_0\sqrt{h/h_0}\;\Rightarrow\; \dfrac{dh}{dt} = -\text{constante}"]),
        ("Resultado general",
         r"Para un depósito prismático de sección $A$, con un orificio de área $a\ll A$ a la altura inicial $h_0$, el nivel y el tiempo de vaciado son los siguientes. La verificación dimensional: $[(A/a)\sqrt{h/g}] = \sqrt{\mathrm{m/(m\,s^{-2})}} = \mathrm{s}$ y $[\kappa] = \mathrm{m^{1/2}/s}$ coincide con $[\sqrt h/t]$.",
         [r"h(t) = \left(\sqrt{h_0} - \dfrac{a}{A}\sqrt{\dfrac g2}\,t\right)^2,\qquad 0\le t\le t_{vac}",
          r"\boxed{\,t_{vac} = \dfrac{A}{a}\sqrt{\dfrac{2h_0}{g}}\,}"]),
        ("Particularización al problema",
         rf"Con $A = {A}$ m², $a = {sci(a,3)}$ m², $h_0 = {h0}$ m y $g = {g}$ m/s² se obtiene $\kappa = {sci(kk,5)}$ m$^{{1/2}}$/s. La corrección por razón de áreas es de apenas ${sci(corr,2)}$ en la velocidad, lo que justifica despreciarla, y el vaciado dura aproximadamente media hora, igual que en la escena.",
         [rf"v_0 = \sqrt{{2({g})({h0})}} = {v0:.4f}\ \mathrm{{m/s}},\qquad \dfrac{{1}}{{\sqrt{{1-(a/A)^2}}}} - 1 = {sci(corr,3)}",
          rf"t_{{vac}} = \dfrac{{{A}}}{{{sci(a,3)}}}\sqrt{{\dfrac{{2({h0})}}{{{g}}}}} = {A/a:.0f}\,({np.sqrt(2*h0/g):.5f}) = {te:.2f}\ \mathrm{{s}}\approx {te/60:.1f}\ \mathrm{{min}}"]),
        ("Errores comunes y límites de validez",
         r"Es habitual pensar que el tiempo de vaciado es el volumen dividido por el caudal inicial; en realidad el caudal disminuye y el vaciado dura el doble, $t_{vac} = 2V/Q_0$. Tampoco la velocidad de salida depende de la densidad ni del tamaño del orificio. La ley falla si el orificio es grande ($a/A$ no pequeño), si el líquido es viscoso o el orificio capilar, y cuando el nivel es tan bajo que aparece un vórtice de drenaje que arrastra aire.",
         [r"t_{vac} = \dfrac{2\,A\,h_0}{a\sqrt{2gh_0}} = \dfrac{2V_0}{Q_0}"]),
    ]
    return dict(concept=concept)


# =============================================================== SimpleHarmonic
def _shm() -> dict:
    m, k, A = c18.M, c18.K, c18.AMP
    w = c18.omega0()
    T = 2 * np.pi / w
    E = c18.energy_analytic()
    ts = c18.pendulum_period_small()
    t60 = c18.pendulum_period_exact(60.0)
    ag = c18.agm(1.0, float(np.cos(np.deg2rad(30))))
    ser = ts * (1 + (np.deg2rad(60)) ** 2 / 16 + 11 * np.deg2rad(60) ** 4 / 3072)
    concept = [
        ("Qué se estudia y por qué importa",
         r"Casi cualquier sistema que se desplaza poco de una posición de equilibrio estable oscila de forma armónica: un resorte, un péndulo, una molécula, un puente, un circuito LC. Por eso el oscilador armónico es el modelo universal de la física, y es la base de los relojes, de la espectroscopia vibracional y de la teoría cuántica de campos. Aquí se deduce de forma general cuándo y por qué ocurre, y cuánto se aparta la realidad (el péndulo exacto) del modelo lineal.",
         []),
        ("Magnitudes y convenciones",
         r"Se usan la coordenada generalizada $q$ medida desde el equilibrio, la masa efectiva $m$ (kg), la constante restauradora $k$ (N/m), la frecuencia angular $\omega$ (rad/s), el periodo $T = 2\pi/\omega$ (s) y la amplitud $A$ (m). Para el péndulo la coordenada es el ángulo $\theta$ (rad) y $L$ la longitud (m). La energía total $E$ se mide en julios; los osciladores mecánicos cotidianos tienen frecuencias de $0.1$ a $10^3$ Hz.",
         []),
        ("Intuición y analogías",
         r"El potencial en la vecindad de un mínimo siempre parece una parábola, igual que cualquier curva suave parece una recta vista de cerca. La fuerza restauradora es entonces lineal y la partícula rueda hacia atrás y hacia adelante en un cuenco. En un cuenco real, más empinado en los bordes que una parábola o más plano, el periodo depende de la altura de la que se suelta; es lo que pasa con el péndulo, cuyo potencial es más suave que una parábola y por eso se vuelve más lento a gran amplitud.",
         []),
        ("Lagrangiano general y ecuaciones de Lagrange",
         r"Para un sistema con coordenadas generalizadas $q_i$, ligaduras independientes del tiempo y fuerzas conservativas, la energía cinética es cuadrática en las velocidades. Se postula la acción $S = \int L\,dt$ y su estacionariedad, $\delta S = 0$, conduce a las ecuaciones de Euler-Lagrange, válidas en cualquier sistema de coordenadas. La matriz $T_{ij}$ es simétrica y definida positiva.",
         [r"L(q,\dot q) = T - V = \tfrac12\sum_{i,j}T_{ij}(q)\,\dot q_i\dot q_j - V(q)",
          r"\dfrac{d}{dt}\dfrac{\partial L}{\partial\dot q_i} - \dfrac{\partial L}{\partial q_i} = 0"]),
        ("Equilibrio y linealización: pequeñas oscilaciones",
         r"Un equilibrio $q^{(0)}$ cumple $\partial V/\partial q_i = 0$. Se desarrolla $V$ hasta segundo orden en los desplazamientos $\eta_i = q_i - q_i^{(0)}$ y se evalúa $T_{ij}$ en el equilibrio, porque los términos de orden superior en $\eta$ multiplican velocidades que ya son pequeñas. El equilibrio es estable si la matriz hessiana $K$ es definida positiva. El Lagrangiano resultante es cuadrático y las ecuaciones de movimiento son lineales.",
         [r"V \simeq V_0 + \tfrac12\sum_{ij}K_{ij}\eta_i\eta_j,\qquad K_{ij} = \left.\dfrac{\partial^2V}{\partial q_i\partial q_j}\right|_{q^{(0)}},\qquad M_{ij} = T_{ij}(q^{(0)})",
          r"L = \tfrac12\,\dot{\vec\eta}^{\,T}M\dot{\vec\eta} - \tfrac12\,\vec\eta^{\,T}K\vec\eta \;\Rightarrow\; M\ddot{\vec\eta} + K\vec\eta = 0"]),
        ("Modos normales y frecuencias propias",
         r"Se buscan soluciones $\vec\eta = \vec a\,e^{i\omega t}$, lo que lleva al problema de autovalores generalizado $(K - \omega^2M)\vec a = 0$. Tiene solución no trivial solo si el determinante se anula: las raíces $\omega_\alpha^2$ son reales y positivas porque $K$ y $M$ son simétricas y definidas positivas. Las coordenadas normales $Q_\alpha$ desacoplan el sistema en $f$ osciladores independientes.",
         [r"\det(K - \omega^2M) = 0,\qquad \vec a_\alpha^{\,T}M\vec a_\beta = \delta_{\alpha\beta}",
          r"L = \sum_\alpha\tfrac12\left(\dot Q_\alpha^2 - \omega_\alpha^2Q_\alpha^2\right)"]),
        ("Un grado de libertad: solución, energía y formalismo hamiltoniano",
         r"Con una sola coordenada la masa efectiva es $m$ y la constante es $k = V''(q_0)$. La ecuación $m\ddot x + kx = 0$ tiene solución general con dos constantes fijadas por las condiciones iniciales. El momento canónico es $p = m\dot x$ y el hamiltoniano $H = p^2/2m + kx^2/2$ es independiente del tiempo, de modo que la energía se conserva (simetría de traslación temporal, teorema de Noether). Las trayectorias en el espacio de fases son elipses de semiejes $A$ y $m\omega A$.",
         [r"\omega = \sqrt{\dfrac{V''(q_0)}{m_{ef}}},\qquad x(t) = x_0\cos\omega t + \dfrac{v_0}{\omega}\sin\omega t = A\cos(\omega t + \varphi)",
          r"H = \dfrac{p^2}{2m} + \tfrac12kx^2 = \tfrac12kA^2,\qquad A^2 = x_0^2 + \dfrac{v_0^2}{\omega^2},\qquad \dfrac{x^2}{A^2} + \dfrac{p^2}{(m\omega A)^2} = 1"]),
        ("El péndulo plano: integral de energía y periodo exacto",
         r"Para un péndulo de longitud $L$ el Lagrangiano es $L = \tfrac12mL^2\dot\theta^2 + mgL\cos\theta$ y la energía se conserva. Despejando $\dot\theta$ y separando variables, el periodo se expresa como cuarto de oscilación; con la sustitución $\sin(\theta/2) = k\sin\phi$, $k = \sin(\theta_0/2)$, resulta una integral elíptica completa de primera especie, que se calcula con rapidez con la media aritmético-geométrica. Para amplitudes pequeñas se desarrolla en serie, lo que muestra la anarmonicidad.",
         [r"\dfrac{1}{2}\dot\theta^2 - \dfrac{g}{L}\cos\theta = -\dfrac{g}{L}\cos\theta_0 \;\Rightarrow\; T = 4\sqrt{\dfrac Lg}\int_0^{\pi/2}\dfrac{d\phi}{\sqrt{1-k^2\sin^2\phi}} = 4\sqrt{\dfrac Lg}\,K(k)",
          r"T = \dfrac{2\pi\sqrt{L/g}}{\mathrm{AGM}\!\left(1,\cos\tfrac{\theta_0}{2}\right)} \simeq T_0\left[1 + \dfrac{\theta_0^2}{16} + \dfrac{11\,\theta_0^4}{3072} + \cdots\right]"]),
        ("Resultado general",
         r"Cerca de un equilibrio estable de un grado de libertad, la frecuencia depende únicamente de la curvatura del potencial y de la masa efectiva, y es independiente de la amplitud (isocronismo). Las unidades son consistentes: $[k/m] = \mathrm{N\,m^{-1}\,kg^{-1}} = \mathrm{s^{-2}}$. Fuera del régimen lineal, el periodo crece con la amplitud según la fórmula del péndulo.",
         [r"\boxed{\,\omega = \sqrt{\dfrac{V''(q_0)}{m_{ef}}},\qquad T = \dfrac{2\pi}{\omega},\qquad E = \tfrac12kA^2,\qquad v_{max} = A\omega\,}",
          r"T_{pend}(\theta_0) = \dfrac{2\pi\sqrt{L/g}}{\mathrm{AGM}(1,\cos(\theta_0/2))}"]),
        ("Particularización al problema",
         rf"Para el bloque, $V = \tfrac12kx^2$ da $V'' = k = {k:.0f}$ N/m y $m_{{ef}} = {m}$ kg. Para el péndulo de $L = 1$ m, $V = mgL(1-\cos\theta)$ da $V'' = mgL$ y $m_{{ef}} = mL^2$, de modo que $\omega_0^2 = g/L$. La serie con dos términos da {ser:.5f} s para 60°, frente al valor exacto {t60:.5f} s de la AGM.",
         [rf"\omega = \sqrt{{\dfrac{{{k:.0f}}}{{{m}}}}} = {w:.1f}\ \mathrm{{rad/s}},\qquad T = \dfrac{{2\pi}}{{{w:.1f}}} = {T:.6f}\ \mathrm{{s}}",
          rf"v_{{max}} = ({A})({w:.1f}) = {A*w:.2f}\ \mathrm{{m/s}},\qquad E = \tfrac12({k:.0f})({A})^2 = {E:.3f}\ \mathrm{{J}}",
          rf"T_0 = 2\pi\sqrt{{\dfrac{{1}}{{{c18.G}}}}} = {ts:.5f}\ \mathrm{{s}},\qquad T(60^\circ) = \dfrac{{{ts:.5f}}}{{{ag:.8f}}} = {t60:.5f}\ \mathrm{{s}}"]),
        ("Errores comunes y límites de validez",
         r"No es cierto que el periodo del resorte dependa de la amplitud, ni que el del péndulo deba ser independiente de ella: solo lo es en el límite lineal, y a 60° ya difiere en cerca de 7 %. También se confunde la fase $\varphi$ con la fase inicial del movimiento (dependen de la condición inicial) o se olvida que $\omega$ no es la frecuencia en hercios. La aproximación armónica deja de valer cuando la amplitud alcanza la región donde el potencial se aparta de la parábola; el sistema pasa entonces a ser anarmónico y aparecen armónicos superiores y desplazamientos de frecuencia.",
         [r"\dfrac{T-T_0}{T_0} \simeq \dfrac{\theta_0^2}{16}\quad(\theta_0\ll1)"]),
    ]
    return dict(concept=concept)


# =============================================================== DrivenResonance
def _driven() -> dict:
    m, k, b, F0 = c18.M_D, c18.K_D, c18.B_D, c18.F0
    w0 = np.sqrt(k / m)
    gam = b / m
    wr = c18.resonance_frequency_analytic()
    Ar = float(c18.driven_amplitude_analytic(wr))
    A0 = F0 / m
    wd = np.sqrt(w0**2 - gam**2 / 4)
    Astat = F0 / k
    dl = np.arctan2(gam * w0, 0.0)
    pm = 0.5 * F0 * float(c18.driven_amplitude_analytic(w0)) * w0 * np.sin(np.pi / 2)
    concept = [
        ("Qué se estudia y por qué importa",
         r"Un oscilador real pierde energía por rozamiento y, a la vez, puede recibir energía de un agente externo periódico. El balance entre ambos procesos determina la respuesta estacionaria y da lugar a la resonancia. Este fenómeno decide el diseño de puentes y edificios, la sintonía de radios, los relojes de cuarzo, los instrumentos musicales y las resonancias magnéticas.",
         []),
        ("Magnitudes y convenciones",
         r"Se usan la masa $m$ (kg), la constante del resorte $k$ (N/m), el coeficiente de amortiguamiento viscoso $b$ (kg/s), la amplitud de la fuerza $F_0$ (N) y su frecuencia angular $\omega$ (rad/s). Se definen la frecuencia natural $\omega_0 = \sqrt{k/m}$, la tasa de amortiguamiento $\gamma = b/m$ (s$^{-1}$) y el factor de calidad $Q = \omega_0/\gamma$ (sin unidades). Un diapasón tiene $Q\sim10^3$; un amortiguador de automóvil, $Q\lesssim1$.",
         []),
        ("Intuición y analogías",
         r"Empujar un columpio es el ejemplo clásico: si cada empuje llega en fase con la velocidad, el trabajo realizado es siempre positivo y la amplitud crece hasta que la pérdida por rozamiento por ciclo iguala a la energía aportada. Una fuerza de frecuencia muy baja sigue la posición lentamente (respuesta estática) y una muy alta no consigue mover la masa. La resonancia aparece cuando la fuerza y la velocidad están en fase.",
         []),
        ("Ecuación de movimiento: Lagrangiano con función de disipación",
         r"Las fuerzas viscosas $-b\dot x$ no derivan de un potencial. Se extienden las ecuaciones de Lagrange con la función de disipación de Rayleigh $\mathcal F = \tfrac12b\dot x^2$ y la fuerza generalizada externa. Con $L = \tfrac12m\dot x^2 - \tfrac12kx^2$ se llega a una ecuación lineal de segundo orden con coeficientes constantes, y los parámetros $\omega_0$, $\gamma$ y $F_0/m$ la normalizan.",
         [r"\dfrac{d}{dt}\dfrac{\partial L}{\partial\dot x} - \dfrac{\partial L}{\partial x} = -\dfrac{\partial\mathcal F}{\partial\dot x} + F_0\cos\omega t",
          r"\ddot x + \gamma\dot x + \omega_0^2x = \dfrac{F_0}{m}\cos\omega t,\qquad \gamma = \dfrac bm,\quad \omega_0^2 = \dfrac km"]),
        ("Solución homogénea: el transitorio",
         r"La solución general es la suma de la solución homogénea y una solución particular. Con $x\propto e^{rt}$ la ecuación característica da dos raíces. Si $\gamma<2\omega_0$ (subamortiguado) el transitorio es una oscilación de frecuencia $\omega_d$ con envolvente $e^{-\gamma t/2}$; si $\gamma = 2\omega_0$ es críticamente amortiguado y si $\gamma>2\omega_0$ la respuesta es la suma de dos exponenciales. En todos los casos el transitorio se extingue, de modo que el movimiento a tiempos largos queda fijado solo por la fuerza.",
         [r"r^2 + \gamma r + \omega_0^2 = 0 \;\Rightarrow\; r_\pm = -\dfrac\gamma2 \pm\sqrt{\dfrac{\gamma^2}{4} - \omega_0^2}",
          r"x_h(t) = e^{-\gamma t/2}\left[C_1\cos\omega_dt + C_2\sin\omega_dt\right],\qquad \omega_d = \sqrt{\omega_0^2 - \dfrac{\gamma^2}{4}}"]),
        ("Solución estacionaria por amplitud compleja",
         r"Como la ecuación es lineal, se reemplaza $\cos\omega t$ por $e^{i\omega t}$ y se toma la parte real al final. Se propone $x_p = \mathrm{Re}[\hat x\,e^{i\omega t}]$ y se obtiene $\hat x$ como un cociente complejo (función de respuesta $\chi(\omega)$). Su módulo es la amplitud y su argumento es el desfase respecto de la fuerza; $\chi$ es analítica en el semiplano superior, consecuencia de la causalidad.",
         [r"\hat x = \chi(\omega)\dfrac{F_0}{m},\qquad \chi(\omega) = \dfrac{1}{\omega_0^2 - \omega^2 + i\gamma\omega}",
          r"x_p(t) = A(\omega)\cos(\omega t - \delta),\qquad A = \dfrac{F_0}{m}|\chi|,\qquad \tan\delta = \dfrac{\gamma\omega}{\omega_0^2 - \omega^2}"]),
        ("Amplitud, fase y resonancia de amplitud",
         r"La amplitud es máxima cuando el radicando $D(\omega) = (\omega_0^2-\omega^2)^2 + \gamma^2\omega^2$ es mínimo. Derivando respecto de $\omega^2$ se obtiene la frecuencia de resonancia de amplitud, que es menor que $\omega_0$ y solo existe si $\gamma<\sqrt2\,\omega_0$; si no, $A$ decrece monótonamente. El desfase pasa de 0 (fuerza lenta) a $\pi$ (fuerza rápida) y vale $\pi/2$ en $\omega = \omega_0$ independientemente del amortiguamiento.",
         [r"\dfrac{dD}{d(\omega^2)} = -2(\omega_0^2-\omega^2) + \gamma^2 = 0 \;\Rightarrow\; \omega_r = \sqrt{\omega_0^2 - \dfrac{\gamma^2}{2}}",
          r"A_{max} = \dfrac{F_0/m}{\gamma\sqrt{\omega_0^2 - \gamma^2/4}} = \dfrac{F_0/m}{\gamma\,\omega_d}"]),
        ("Potencia absorbida, factor de calidad y ancho de la resonancia",
         r"La potencia media entregada por la fuerza es igual a la disipada por la viscosidad; se promedia sobre un ciclo y resulta una lorentziana de máximo en $\omega_0$ (resonancia de potencia, distinta de la de amplitud) y ancho a mitad de altura $\gamma$. Por eso $Q = \omega_0/\gamma$ mide la nitidez del pico, y también el número de ciclos que dura el transitorio (energía decae a $e^{-2\pi}$ en $Q$ ciclos aproximadamente).",
         [r"\overline P = \tfrac12 b\omega^2A^2 = \dfrac{F_0^2}{2m}\,\dfrac{\gamma\omega^2}{(\omega_0^2-\omega^2)^2 + \gamma^2\omega^2}",
          r"\overline P_{max} = \dfrac{F_0^2}{2b},\qquad \Delta\omega_{1/2}\simeq\gamma,\qquad Q = \dfrac{\omega_0}{\gamma}"]),
        ("Resultado general",
         r"Después del transitorio el oscilador responde a la frecuencia de la fuerza con amplitud y fase dadas abajo. Las unidades son consistentes: $[F_0/(m\omega^2)] = \mathrm{N/(kg\,s^{-2})} = \mathrm{m}$. Para $\omega\to0$ se recupera la deflexión estática $F_0/k$, y para $\omega\to\infty$ la amplitud decae como $F_0/(m\omega^2)$.",
         [r"\boxed{\,A(\omega) = \dfrac{F_0/m}{\sqrt{(\omega_0^2-\omega^2)^2 + (\gamma\omega)^2}},\qquad \omega_r = \sqrt{\omega_0^2 - \dfrac{\gamma^2}{2}}\,}",
          r"\delta(\omega) = \arctan\dfrac{\gamma\omega}{\omega_0^2-\omega^2},\qquad Q = \dfrac{\omega_0}{\gamma}"]),
        ("Particularización al problema",
         rf"Con $m = {m:.0f}$ kg, $k = {k:.0f}$ N/m, $b = {b:.0f}$ kg/s y $F_0 = {F0:.0f}$ N, los parámetros son $\omega_0 = {w0:.0f}$ rad/s, $\gamma = {gam:.0f}$ s$^{{-1}}$, $Q = {w0/gam:.0f}$ y $F_0/m = {A0:.0f}$ m/s². La frecuencia de resonancia y la amplitud reproducen las cifras de la escena; la amplitud estacionaria vale ${Astat:.3f}$ m a baja frecuencia y a $\omega_r$ es unas cinco veces mayor. El transitorio tiene frecuencia $\omega_d = {wd:.4f}$ rad/s.",
         [rf"\omega_r = \sqrt{{{w0**2:.0f} - \dfrac{{({gam:.0f})^2}}{{2}}}} = {wr:.4f}\ \mathrm{{rad/s}}",
          rf"A(\omega_r) = \dfrac{{{A0:.0f}}}{{\sqrt{{({gam**2/2:.1f})^2 + ({gam:.0f})^2({wr**2:.1f})}}}} = {Ar:.5f}\ \mathrm{{m}},\qquad \dfrac{{A(\omega_r)}}{{F_0/k}} = {Ar/Astat:.3f}",
          rf"\overline P_{{max}} = \dfrac{{F_0^2}}{{2b}} = \dfrac{{{F0**2:.0f}}}{{{2*b:.0f}}} = {F0**2/(2*b):.2f}\ \mathrm{{W}}"]),
        ("Errores comunes y límites de validez",
         r"Se confunde la resonancia de amplitud ($\omega_r$) con la de potencia ($\omega_0$) y la frecuencia natural no amortiguada con la del transitorio $\omega_d$; las tres difieren cuando $\gamma$ no es pequeño. También se olvida que la solución estacionaria no depende de las condiciones iniciales y que la amplitud en resonancia no crece sin límite mientras $\gamma>0$. El modelo lineal deja de valer cuando la amplitud llega a la región anarmónica (el pico se inclina y aparece histéresis, como en el oscilador de Duffing) o cuando el amortiguamiento no es viscoso.",
         [r"\omega_r<\omega_d<\omega_0,\qquad \text{resonancia de potencia en }\omega_0"]),
    ]
    return dict(concept=concept)


# =============================================================== WavePulse
def _pulse() -> dict:
    T, mu, v = c19.TENSION, c19.MU, c19.V
    sg = c19.SIGMA
    t = 0.15
    xs = v * t
    n = 800
    dx = c19.LENGTH / n
    dt1 = dx / v
    err1, err9 = c19.fdtd_error(1.0), c19.fdtd_error(0.9)
    # dispersion numerica para C=0.9 y lambda tipica = 2*pi*sigma
    kk = 1.0 / sg
    cr = 0.9
    wnum = (2 / dt1 / cr) * np.arcsin(cr * np.sin(kk * dx / 2))
    vph = wnum / kk
    concept = [
        ("Qué se estudia y por qué importa",
         r"Una perturbación localizada en un medio elástico se propaga como una onda sin que el medio se desplace en bloque. La ecuación de onda gobierna cuerdas, columnas de aire, ondas electromagnéticas en el vacío y vibraciones de sólidos. Entender cómo un pulso se divide en dos mitades que viajan en sentidos opuestos es la base de la acústica, de las líneas de transmisión y de la sismología.",
         []),
        ("Magnitudes y convenciones",
         r"Se usan el desplazamiento transversal $y(x,t)$ (m), la tensión $T$ (N), la densidad lineal de masa $\mu$ (kg/m), la velocidad de fase $v$ (m/s) y la energía lineal $u$ (J/m). Las pendientes $\partial y/\partial x$ se suponen pequeñas. Una cuerda de guitarra tiene $v$ del orden de $10^2$ m/s; un cable tenso, algunas decenas.",
         []),
        ("Intuición y analogías",
         r"Cada elemento de la cuerda es tirado por sus vecinos: si la cuerda está curvada hacia arriba en un punto, la resultante de las tensiones empuja hacia abajo. La aceleración local es proporcional a la curvatura, y esa relación lineal entre curvatura y aceleración es la ecuación de onda. El pulso se transmite porque cada elemento hace mover al siguiente, y la rapidez es mayor cuanto mayor es la tensión y menor la inercia.",
         []),
        ("Densidad lagrangiana de la cuerda y ecuación de onda",
         r"Para un sistema continuo la acción es la integral de una densidad lagrangiana, aquí $\mathcal L = \tfrac12\mu y_t^2 - \tfrac12Ty_x^2$ (energía cinética menos energía potencial elástica, esta última por el alargamiento $\Delta s\simeq\tfrac12y_x^2dx$ a pequeñas pendientes). La estacionariedad de la acción con extremos fijados da la ecuación de Euler-Lagrange para campos, que es la ecuación de onda con velocidad $v = \sqrt{T/\mu}$.",
         [r"S = \int\!\!\int\mathcal L\,dx\,dt,\qquad \mathcal L = \tfrac12\mu\left(\dfrac{\partial y}{\partial t}\right)^2 - \tfrac12T\left(\dfrac{\partial y}{\partial x}\right)^2",
          r"\dfrac{\partial}{\partial t}\dfrac{\partial\mathcal L}{\partial y_t} + \dfrac{\partial}{\partial x}\dfrac{\partial\mathcal L}{\partial y_x} = 0 \;\Rightarrow\; \mu\,y_{tt} = T\,y_{xx},\qquad v = \sqrt{T/\mu}"]),
        ("Conservación de la energía y flujo",
         r"La invariancia de $\mathcal L$ bajo traslaciones en el tiempo da, por el teorema de Noether, una ley de conservación local. La densidad de energía $u$ y la potencia transmitida $S$ (flujo de energía) cumplen una ecuación de continuidad. Para una onda viajera $y = f(x - vt)$ se tiene $u = \mu v^2f'^2$ y $S = vu$: la energía viaja con la velocidad de la onda.",
         [r"u = \tfrac12\mu y_t^2 + \tfrac12Ty_x^2,\qquad S = -T\,y_xy_t",
          r"\dfrac{\partial u}{\partial t} + \dfrac{\partial S}{\partial x} = 0"]),
        ("Solución de d'Alembert por características",
         r"Con el cambio de variables $\xi = x - vt$, $\eta = x + vt$ la ecuación de onda se vuelve $\partial^2y/\partial\xi\,\partial\eta = 0$, cuya solución general es la suma de una función de $\xi$ y otra de $\eta$. La primera es una onda que se mueve hacia $+x$ sin cambiar de forma y la segunda hacia $-x$. Este resultado depende de que la ecuación sea lineal, de coeficientes constantes y no dispersiva.",
         [r"\xi = x-vt,\quad \eta = x+vt:\qquad \dfrac{\partial^2y}{\partial\xi\,\partial\eta} = 0 \;\Rightarrow\; y = F(x-vt) + G(x+vt)"]),
        ("Problema de valores iniciales",
         r"Se imponen $y(x,0) = f(x)$ y $y_t(x,0) = g(x)$. Derivando la solución general y despejando $F$ y $G$ se obtiene la fórmula de Stokes. Para un pulso inicialmente en reposo ($g = 0$) la solución se reduce a la mitad de $f$ trasladada en cada sentido: el pulso se divide en dos mitades de amplitud $1/2$ que se separan con velocidad $v$.",
         [r"y(x,t) = \tfrac12\left[f(x-vt) + f(x+vt)\right] + \dfrac{1}{2v}\int_{x-vt}^{x+vt}g(s)\,ds",
          r"g = 0:\qquad y(x,t) = \tfrac12\left[f(x-vt) + f(x+vt)\right]"]),
        ("Discretización y dispersión numérica",
         r"En diferencias finitas centradas con paso $\Delta x$ y $\Delta t$ la relación de dispersión discreta es $\sin(\omega\Delta t/2) = C\sin(k\Delta x/2)$ con el número de Courant $C = v\Delta t/\Delta x$. Es estable solo si $C\le1$, y para $C=1$ coincide exactamente con la continua. Para $C<1$ las componentes de número de onda $k$ viajan con velocidad de fase ligeramente menor, lo que deforma el pulso.",
         [r"\sin\dfrac{\omega\Delta t}{2} = C\sin\dfrac{k\Delta x}{2},\qquad C = \dfrac{v\Delta t}{\Delta x}\le1",
          r"v_{fase}^{num} = \dfrac{\omega}{k}\simeq v\left[1 - \dfrac{(k\Delta x)^2}{24}(1-C^2)\right]"]),
        ("Resultado general",
         r"Una cuerda uniforme con tensión $T$ y densidad $\mu$ transporta perturbaciones de forma invariante con la velocidad indicada. Un pulso inicial $f(x)$ en reposo se descompone en dos réplicas de amplitud mitad. Dimensionalmente $[T/\mu] = \mathrm{N\,m/kg} = \mathrm{m^2/s^2}$.",
         [r"\boxed{\,v = \sqrt{\dfrac T\mu},\qquad y(x,t) = \tfrac12\left[f(x-vt) + f(x+vt)\right]\,}",
          r"u = \mu v^2 f'^2,\qquad E_{pulso} = \int u\,dx"]),
        ("Particularización al problema",
         rf"Con $T = {T:.0f}$ N y $\mu = {mu}$ kg/m resulta $v = {v:.0f}$ m/s; el pulso gaussiano de $\sigma = {sg}$ m, centrado en $x_0 = 10$ m, produce a $t = {t}$ s dos gaussianas de altura $0.5$ separadas de $x_0$ por $vt = {xs:.1f}$ m. En la discretización de la escena ($N = {n}$, $\Delta x = {dx}$ m) el paso temporal con $C=1$ es ${sci(dt1,4)}$ s, y el error frente a d'Alembert es {sci(err1,2)} m con $C=1$ y {sci(err9,2)} m con $C=0.9$.",
         [rf"v = \sqrt{{\dfrac{{{T:.0f}}}{{{mu}}}}} = {v:.1f}\ \mathrm{{m/s}},\qquad vt = ({v:.0f})({t}) = {xs:.1f}\ \mathrm{{m}}",
          rf"y(x,{t}) = \tfrac12\left[e^{{-(x-{10-xs:.1f})^2/(2\cdot{sg}^2)}} + e^{{-(x-{10+xs:.1f})^2/(2\cdot{sg}^2)}}\right],\qquad y_{{max}} = 0.5",
          rf"v_{{fase}}^{{num}}(k=1/\sigma,\ C=0.9) = {vph:.4f}\ \mathrm{{m/s}}\quad(\text{{frente a }}{v:.1f})"]),
        ("Errores comunes y límites de validez",
         r"Se confunde la velocidad de la onda (propiedad del medio) con la velocidad transversal de los elementos de la cuerda, que depende de la amplitud: $y_t = -vf'$. También se cree que el pulso se atenúa al dividirse, cuando la energía total se conserva (cada mitad lleva la mitad de la energía). El modelo falla para pendientes grandes (no linealidad), cuerdas rígidas (la rigidez añade dispersión, $\omega^2 = v^2k^2 + \beta k^4$) o con pérdidas (atenuación). Las condiciones de contorno reflejan el pulso y no están contempladas en la solución sobre la recta infinita.",
         [r"\omega^2 = v^2k^2 + \beta k^4\ \Rightarrow\ v_{fase}=\omega/k\ \text{depende de }k\ (\text{dispersión})"]),
    ]
    return dict(concept=concept)


# =============================================================== StandingModes
def _modes() -> dict:
    v, L = c19.V, c19.STRING_L
    f = [c19.mode_frequency_analytic(n) for n in (1, 2, 3, 5)]
    n_mass = 400
    fnum1 = (v / (L / (n_mass + 1))) / np.pi * np.sin(np.pi / (2 * (n_mass + 1)))
    concept = [
        ("Qué se estudia y por qué importa",
         r"Una cuerda fija en ambos extremos solo puede vibrar establemente con ciertas formas y frecuencias discretas, los modos normales. Son el origen de las notas de los instrumentos de cuerda, de las líneas espectrales de sistemas confinados y de los niveles de energía de una partícula en una caja en mecánica cuántica. Cualquier movimiento de la cuerda es una superposición de modos, de modo que conocerlos permite predecir todo el comportamiento.",
         []),
        ("Magnitudes y convenciones",
         r"Se emplean la longitud $L$ (m), la velocidad de la onda $v = \sqrt{T/\mu}$ (m/s), el número de onda $k_n$ (rad/m), la frecuencia $f_n$ (Hz) y la amplitud modal $A_n$ (m). El índice $n=1,2,\dots$ cuenta los vientres; $n-1$ es el número de nodos interiores. En una guitarra, $L\simeq0.65$ m y $f_1$ entre 80 y 330 Hz.",
         []),
        ("Intuición y analogías",
         r"Un modo normal es un movimiento en el que todos los puntos de la cuerda oscilan con la misma frecuencia y en fase o en oposición de fase, como un columpio gigante con una forma fija. Es el análogo continuo de los modos de un sistema de masas acopladas. Los modos son independientes: se puede excitar uno sin que se transfiera energía a otro, de forma que el sonido de una cuerda es la suma de sus armónicos con distintos pesos.",
         []),
        ("Separación de variables: el problema de Sturm-Liouville",
         r"Se parte de la ecuación de onda $y_{tt} = v^2y_{xx}$ con $y(0,t) = y(L,t) = 0$. Se busca $y = X(x)\,\Theta(t)$ y se divide por $X\Theta$: un lado depende solo de $x$ y el otro solo de $t$, de modo que ambos son iguales a una constante, que se escribe $-k^2$. La constante negativa es la única compatible con condiciones de contorno de Dirichlet no triviales, como muestra una integración por partes de $\int X X''dx$.",
         [r"\dfrac{X''}{X} = \dfrac{1}{v^2}\dfrac{\ddot\Theta}{\Theta} = -k^2",
          r"X'' + k^2X = 0,\qquad X(0) = X(L) = 0,\qquad \ddot\Theta + \omega^2\Theta = 0,\quad \omega = vk"]),
        ("Condiciones de contorno y espectro",
         r"La solución espacial es $X = a\sin kx + b\cos kx$. La condición en $x=0$ fuerza $b=0$ y la de $x=L$ exige $\sin kL = 0$, lo que cuantiza el número de onda. El espectro resultante es equiespaciado (armónico): las frecuencias son múltiplos enteros de la fundamental, y la longitud de onda de cada modo es una fracción entera de $2L$. Con otras condiciones de contorno el espectro cambia, como se indica abajo.",
         [r"k_n = \dfrac{n\pi}{L},\qquad \lambda_n = \dfrac{2L}{n},\qquad \omega_n = \dfrac{n\pi v}{L},\qquad f_n = \dfrac{nv}{2L}",
          r"\text{libre-libre: } f_n = \dfrac{nv}{2L};\qquad \text{fija-libre: } f_n = \dfrac{(2n-1)v}{4L}"]),
        ("Ortogonalidad, completitud y solución general",
         r"Las funciones $\sin(n\pi x/L)$ son ortogonales en $[0,L]$ y forman una base completa (series de Fourier de senos) para funciones que se anulan en los extremos. La solución general es la superposición de todos los modos, y los coeficientes se fijan proyectando las condiciones iniciales sobre cada modo.",
         [r"\int_0^L\sin\dfrac{n\pi x}{L}\sin\dfrac{m\pi x}{L}\,dx = \dfrac L2\,\delta_{nm}",
          r"y(x,t) = \sum_{n=1}^\infty\sin\dfrac{n\pi x}{L}\left[a_n\cos\omega_nt + b_n\sin\omega_nt\right],\quad a_n = \dfrac2L\!\int_0^L\!f\sin\dfrac{n\pi x}{L}dx,\ \ b_n = \dfrac{2}{n\pi v}\!\int_0^L\!g\sin\dfrac{n\pi x}{L}dx"]),
        ("Energía modal y cuerda como suma de osciladores",
         r"Al sustituir el desarrollo en el hamiltoniano de la cuerda, los términos cruzados se anulan por ortogonalidad y la energía total es la suma de las energías de osciladores armónicos independientes, uno por modo. Cada coordenada normal $q_n(t)$ obedece una ecuación de oscilador armónico. Esto explica que la energía de cada modo se conserve por separado en el modelo lineal.",
         [r"H = \sum_n\tfrac12\mu\dfrac L2\left(\dot q_n^2 + \omega_n^2q_n^2\right),\qquad E_n = \tfrac14\mu L\,\omega_n^2A_n^2",
          r"y = \sum_nq_n(t)\sin\dfrac{n\pi x}{L},\qquad \ddot q_n + \omega_n^2q_n = 0"]),
        ("Cadena discreta y relación con el método numérico",
         r"Una cuerda se aproxima por $N$ masas puntuales separadas $\Delta x = L/(N+1)$ y unidas por muelles. Las ecuaciones $\ddot y_j = (v/\Delta x)^2(y_{j+1} - 2y_j + y_{j-1})$ con extremos fijos tienen modos exactos $\sin(jn\pi/(N+1))$ y frecuencias que se desvían del espectro continuo para $n$ grande. Los autovalores de la matriz tridiagonal con ese patrón son los mismos que los de la cadena de masas.",
         [r"\omega_n^{(N)} = \dfrac{2v}{\Delta x}\sin\dfrac{n\pi}{2(N+1)}\ \xrightarrow{\;N\to\infty\;}\ \dfrac{n\pi v}{L}",
          r"\dfrac{\omega_n^{(N)}}{\omega_n} \simeq 1 - \dfrac{1}{24}\left(\dfrac{n\pi}{N+1}\right)^2"]),
        ("Resultado general",
         r"Las frecuencias propias de una cuerda de longitud $L$ con extremos fijos y velocidad de onda $v$ son los múltiplos enteros de la fundamental, y la forma de cada modo es sinusoidal. Unidades: $[v/L] = \mathrm{s^{-1}}$.",
         [r"\boxed{\,f_n = \dfrac{n\,v}{2L},\qquad y_n(x,t) = A_n\sin\dfrac{n\pi x}{L}\cos(\omega_nt+\varphi_n)\,}",
          r"f_n = n\,f_1,\qquad f_1 = \dfrac{1}{2L}\sqrt{\dfrac T\mu}"]),
        ("Particularización al problema",
         rf"Con $L = {L}$ m y $v = {v:.0f}$ m/s la fundamental es $f_1 = {f[0]:.2f}$ Hz y los modos 2, 3 y 5 siguen la progresión armónica de la escena. La discretización de 400 puntos desvía el modo 1 en el orden de ${sci(abs(fnum1/f[0]-1),2)}$ en términos relativos, ya que $n\pi/(N+1)$ es muy pequeño.",
         [rf"f_n = \dfrac{{n\,({v:.0f})}}{{2({L})}} = {v/(2*L):.4f}\,n\ \mathrm{{Hz}}",
          rf"f_1 = {f[0]:.2f},\quad f_2 = {f[1]:.2f},\quad f_3 = {f[2]:.2f},\quad f_5 = {f[3]:.2f}\ \ \mathrm{{Hz}}",
          rf"\lambda_1 = 2L = {2*L:.1f}\ \mathrm{{m}},\qquad k_1 = \dfrac{{\pi}}{{{L}}} = {np.pi/L:.4f}\ \mathrm{{rad/m}}"]),
        ("Errores comunes y límites de validez",
         r"Se confunde el número de modo con el número de nodos (hay $n-1$ nodos interiores) y se atribuye la altura del sonido a la amplitud en lugar de a $f_n$. También se supone que la cuerda real tiene armónicos exactos: la rigidez a la flexión eleva los armónicos altos ($f_n\simeq nf_1\sqrt{1+Bn^2}$, inarmonicidad). Los extremos nunca son perfectamente fijos y hay pérdidas, de modo que cada modo decae con su propio tiempo. Con amplitudes grandes aparecen acoplamientos no lineales entre modos.",
         [r"f_n \simeq n f_1\sqrt{1 + Bn^2},\qquad B\ll1\ \text{(rigidez)}"]),
    ]
    return dict(concept=concept)


# =============================================================== DopplerEffect
def _doppler() -> dict:
    c, f0, vs = c20.C, c20.F0, c20.VS
    fa, fb = c20.doppler_analytic(), c20.doppler_analytic(vs=-vs)
    fo = c20.doppler_analytic(vs=0.0, vo=30.0)
    mach = vs / c
    lam0 = c / f0
    concept = [
        ("Qué se estudia y por qué importa",
         r"La frecuencia que percibe un observador depende del movimiento relativo entre la fuente, el observador y el medio. Es lo que se oye cuando pasa una ambulancia, y se usa en el radar de velocidad de la policía, en la ecografía Doppler del flujo sanguíneo, en la medición de corrimientos al rojo de galaxias y en la meteorología con radar. El análisis general muestra además cómo se llega a las ondas de choque cuando la fuente supera la velocidad del sonido.",
         []),
        ("Magnitudes y convenciones",
         r"Se usan la frecuencia emitida $f_0$ (Hz), la percibida $f'$ (Hz), la velocidad del sonido respecto del aire $c$ (m/s), las velocidades de la fuente $\vec v_s$ y del observador $\vec v_o$ respecto del aire (m/s) y el vector unitario $\hat n$ que apunta de la fuente al observador. El número de Mach de la fuente es $M = v_s/c$. En el aire a 20 °C, $c\simeq343$ m/s.",
         []),
        ("Intuición y analogías",
         r"Si una fuente que se mueve emite una cresta cada $1/f_0$ segundos, cada cresta sale de un punto distinto y las ondas se amontonan delante y se separan detrás, como las ondulaciones delante de un bote. El observador solo cuenta cuántas crestas recibe por segundo. Un observador que avanza hacia la fuente se encuentra con ellas más seguido sin que cambie la longitud de onda; así, el movimiento de la fuente cambia la longitud de onda y el del observador cambia la tasa de recepción.",
         []),
        ("Tiempo retardado y cinemática de la emisión",
         r"Una cresta emitida en el instante $t_e$ desde la posición $\vec x_s(t_e)$ llega al observador en el instante $t_r$ si recorre la distancia a velocidad $c$ respecto del aire. Esta condición define $t_r(t_e)$ implícitamente. La frecuencia percibida es el cociente entre el número de crestas y el intervalo de recepción, es decir, $f' = f_0\,dt_e/dt_r$.",
         [r"c\,(t_r - t_e) = \left|\vec x_o(t_r) - \vec x_s(t_e)\right|",
          r"f' = f_0\,\dfrac{dt_e}{dt_r}"]),
        ("Derivación para velocidades constantes",
         r"Se deriva la condición de retardo respecto de $t_e$, con $\vec x_s = \vec x_s(0)+\vec v_st_e$ y $\vec x_o = \vec x_o(0)+\vec v_ot_r$. La derivada del módulo de $\vec x_o-\vec x_s$ es su proyección sobre $\hat n$. Despejando $dt_r/dt_e$ resulta una razón entre dos proyecciones de velocidad sobre la línea de visión. Los vectores se miden respecto del medio y por eso la fórmula no es simétrica en fuente y observador.",
         [r"c\left(\dfrac{dt_r}{dt_e} - 1\right) = \hat n\cdot\left(\vec v_o\dfrac{dt_r}{dt_e} - \vec v_s\right)",
          r"\dfrac{dt_r}{dt_e} = \dfrac{c - \hat n\cdot\vec v_s}{c - \hat n\cdot\vec v_o}"]),
        ("Fórmula general y casos particulares",
         r"Invirtiendo el cociente se obtiene la fórmula de Doppler con direcciones arbitrarias. Si la fuente y el observador se mueven sobre la línea que los une y se toman positivas las velocidades de acercamiento ($v_s$ hacia el observador, $v_o$ hacia la fuente), se recupera la forma usual. Para un movimiento perpendicular a la línea de visión ($\hat n\cdot\vec v = 0$) el efecto desaparece en primer orden.",
         [r"f' = f_0\,\dfrac{c - \hat n\cdot\vec v_o}{c - \hat n\cdot\vec v_s}",
          r"f' = f_0\,\dfrac{c + v_o}{c - v_s}\quad\text{(colineal, velocidades de acercamiento positivas)}"]),
        ("Fuente supersónica y ondas de choque",
         r"Si $v_s>c$ el denominador cambia de signo y la fórmula carece de sentido: los frentes emitidos no alcanzan a la fuente y su envolvente es un cono (cono de Mach), con ángulo de semiapertura dado por la razón $c/v_s$. Sobre ese cono la superposición es coherente y aparece una onda de choque (estampido sónico). Para $v_s<c$ el efecto es una simple corrección de la frecuencia.",
         [r"\sin\alpha = \dfrac{c}{v_s} = \dfrac1M,\qquad M = \dfrac{v_s}{c}>1"]),
        ("Comparación con el Doppler de la luz",
         r"Para la luz no hay medio privilegiado, de manera que solo importa la velocidad relativa $\beta = v/c$ entre fuente y observador, y la fórmula resulta de la invariancia de la fase $k_\mu x^\mu$ bajo transformaciones de Lorentz. Aparece el factor $\gamma$, que da un efecto Doppler transversal que no existe para el sonido. Para $\beta\ll1$ ambos coinciden en primer orden.",
         [r"f' = f_0\,\dfrac{\sqrt{1-\beta^2}}{1-\beta\cos\theta_s},\qquad \beta = \dfrac vc",
          r"f' \simeq f_0\left(1 + \beta\cos\theta\right)\quad(\beta\ll1)"]),
        ("Resultado general",
         r"Para velocidades constantes y subsónicas medidas respecto del medio, la frecuencia recibida es la dada abajo. Es adimensional en la razón $f'/f_0$; las velocidades y $c$ comparten unidades.",
         [r"\boxed{\,f' = f_0\,\dfrac{c - \hat n\cdot\vec v_o}{c - \hat n\cdot\vec v_s}\,}",
          r"\text{colineal:}\quad f'_{acerca} = f_0\dfrac{c}{c-v_s},\qquad f'_{aleja} = f_0\dfrac{c}{c+v_s},\qquad f' = f_0\dfrac{c+v_o}{c}"]),
        ("Particularización al problema",
         rf"Con $f_0 = {f0:.0f}$ Hz, $v_s = {vs:.0f}$ m/s y $c = {c:.0f}$ m/s el número de Mach es $M = {mach:.3f}$. Delante del avión $\hat n\cdot\vec v_s = v_s$ y detrás $-v_s$; para el observador que avanza a 30 m/s hacia una fuente en reposo, $\hat n\cdot\vec v_o = -30$. Se recuperan las tres frecuencias de la escena. La longitud de onda delante es {(c-vs)/f0:.4f} m, frente a $\lambda_0 = {lam0:.4f}$ m.",
         [rf"f'_{{delante}} = {f0:.0f}\cdot\dfrac{{{c:.0f}}}{{{c:.0f}-{vs:.0f}}} = {fa:.2f}\ \mathrm{{Hz}}",
          rf"f'_{{aleja}} = {f0:.0f}\cdot\dfrac{{{c:.0f}}}{{{c:.0f}+{vs:.0f}}} = {fb:.2f}\ \mathrm{{Hz}}",
          rf"f'_{{obs}} = {f0:.0f}\cdot\dfrac{{{c:.0f}+30}}{{{c:.0f}}} = {fo:.2f}\ \mathrm{{Hz}}"]),
        ("Errores comunes y límites de validez",
         r"Es frecuente pensar que el sonido de la fuente se mueve más rápido por la velocidad de la fuente; la velocidad de la onda solo depende del medio. También se aplica la misma fórmula a fuente y observador en movimiento sin notar que las velocidades son respecto del aire, y no relativas entre sí. Con viento hay que sumar la velocidad del medio a las proyecciones. La fórmula deja de valer para $v_s\ge c$ (se sustituye por el cono de Mach), para $v\sim c$ en ondas electromagnéticas (se usa la versión relativista) y cuando las velocidades cambian durante la emisión (aceleración).",
         [r"v_s\to c:\ f'\to\infty\ \ \text{(la fórmula pierde sentido; ver cono de Mach)}"]),
    ]
    return dict(concept=concept)


# =============================================================== BeatsAndPipes
def _beats() -> dict:
    c, Lp = c20.C, c20.LEN
    f1, f2 = c20.F1, c20.F2
    fb = c20.beat_frequency_analytic()
    op = c20.pipe_frequencies_analytic("open")
    cl = c20.pipe_frequencies_analytic("closed")
    fm = 0.5 * (f1 + f2)
    concept = [
        ("Qué se estudia y por qué importa",
         r"Dos fenómenos de ondas sonoras confinadas o superpuestas se tratan juntos. Cuando dos tonos de frecuencias próximas suenan a la vez, la intensidad percibida oscila lentamente (pulsaciones), un método sencillo para afinar instrumentos. Un tubo de aire, en cambio, resuena solo a ciertas frecuencias (flautas, órganos, tubos de escape y conductos de ventilación). Ambos se explican con la superposición y las condiciones de contorno de la onda sonora.",
         []),
        ("Magnitudes y convenciones",
         r"Se usan las frecuencias $f_1$, $f_2$ (Hz), la frecuencia de pulsación $f_p$ (Hz), la presión acústica $p'$ (Pa), el desplazamiento de las moléculas $s$ (m), la velocidad del sonido $c$ (m/s), la longitud del tubo $L$ (m) y los modos $n=1,2,\dots$ El oído percibe pulsaciones de hasta unos 15 Hz; por encima se oyen como aspereza y después como dos tonos separados.",
         []),
        ("Intuición y analogías",
         r"Dos corredores que dan vueltas a una pista con ritmos ligeramente distintos coinciden periódicamente: cuando van en fase sus efectos se suman (sonido fuerte) y cuando van en contrafase se cancelan (silencio). El tubo es un resonador: una onda que se refleja en los extremos solo se refuerza consigo misma si, tras ir y volver, vuelve en fase; eso selecciona un conjunto discreto de longitudes de onda.",
         []),
        ("Superposición de dos tonos: pulsaciones",
         r"Para tonos de igual amplitud $a$ se aplica la identidad trigonométrica para la suma de cosenos y se separa el factor rápido (frecuencia media) del factor lento (envolvente). La amplitud instantánea $2a\cos(\pi\Delta f\,t)$ cambia de signo cada $1/\Delta f$, pero la intensidad, proporcional al cuadrado de la envolvente, tiene máximos cada $1/\Delta f$, con lo que la frecuencia de pulsación es $|f_1-f_2|$ y no $|f_1-f_2|/2$.",
         [r"y = a\cos2\pi f_1t + a\cos2\pi f_2t = 2a\cos\!\left(\pi\Delta f\,t\right)\cos\!\left(2\pi\bar f\,t\right),\quad \Delta f = f_1-f_2,\ \bar f = \tfrac{f_1+f_2}{2}",
          r"I\propto\overline{y^2} \propto 4a^2\cos^2(\pi\Delta f\,t) = 2a^2\left[1 + \cos(2\pi\Delta f\,t)\right]\ \Rightarrow\ f_p = |f_1-f_2|"]),
        ("Ecuación de onda acústica",
         r"Para perturbaciones pequeñas de un gas en reposo se linealizan la continuidad, la ecuación de Euler y la ecuación de estado adiabática. Eliminando la velocidad y la densidad queda la ecuación de onda para la presión acústica $p'$, con velocidad que depende de la compresibilidad adiabática. Para un gas ideal $c = \sqrt{\gamma_{ad}RT/M}$, que aumenta con la temperatura.",
         [r"\dfrac{\partial\rho'}{\partial t} + \rho_0\nabla\cdot\vec u = 0,\qquad \rho_0\dfrac{\partial\vec u}{\partial t} = -\nabla p',\qquad p' = c^2\rho'",
          r"\dfrac{\partial^2p'}{\partial t^2} = c^2\nabla^2p',\qquad c = \sqrt{\dfrac{\gamma_{ad}p_0}{\rho_0}} = \sqrt{\dfrac{\gamma_{ad}RT}{M}}"]),
        ("Condiciones de contorno en los extremos del tubo",
         r"En un extremo abierto el aire se comunica con la atmósfera y la presión acústica es nula (nodo de presión, vientre de desplazamiento). En un extremo cerrado la velocidad del aire es nula, lo que por la ecuación de Euler implica $\partial p'/\partial x = 0$ (vientre de presión). Con $p' = P(x)\cos\omega t$ la ecuación de onda se reduce a $P'' + k^2P = 0$ con $k = \omega/c$, un problema de Sturm-Liouville como el de la cuerda.",
         [r"\text{abierto: } P = 0;\qquad \text{cerrado: } \dfrac{dP}{dx} = 0;\qquad P'' + k^2P = 0"]),
        ("Espectro de los tubos",
         r"Para un tubo abierto en ambos extremos, $P = \sin kx$ con $\sin kL = 0$. Para un tubo cerrado en $x=0$ y abierto en $x=L$, $P = \cos kx$ con $\cos kL = 0$, que solo admite múltiplos impares de un cuarto de onda. El cerrado resuena a una frecuencia una octava más grave que el abierto de igual longitud y solo con armónicos impares, lo que da su timbre característico. En el tubo real, el aire de los extremos abiertos añade una corrección de longitud efectiva.",
         [r"\text{abierto-abierto: } k_n = \dfrac{n\pi}{L},\quad f_n = \dfrac{nc}{2L};\qquad \text{cerrado-abierto: } k_n = \dfrac{(2n-1)\pi}{2L},\quad f_n = \dfrac{(2n-1)c}{4L}",
          r"L_{ef}\simeq L + 0.6\,r\ \ (\text{por extremo abierto, radio }r)"]),
        ("Resultado general",
         r"Dos tonos de frecuencias $f_1$ y $f_2$ producen pulsaciones a la diferencia, y un tubo de longitud $L$ resuena en los espectros de abajo. En ambos casos la frecuencia es una velocidad entre una longitud, y $[c/L] = \mathrm{s^{-1}}$.",
         [r"\boxed{\,f_p = |f_1-f_2|,\qquad f_n^{ab} = \dfrac{nc}{2L},\qquad f_n^{ce} = \dfrac{(2n-1)c}{4L}\,}"]),
        ("Particularización al problema",
         rf"Con $f_1 = {f1:.0f}$ Hz y $f_2 = {f2:.0f}$ Hz la frecuencia media es ${fm:.0f}$ Hz y la pulsación es {fb:.0f} Hz. Con $L = {Lp}$ m y $c = {c:.0f}$ m/s, las frecuencias de resonancia reproducen las de la escena.",
         [rf"f_p = |{f2:.0f}-{f1:.0f}| = {fb:.0f}\ \mathrm{{Hz}},\qquad \bar f = {fm:.0f}\ \mathrm{{Hz}}",
          rf"f_n^{{ab}} = \dfrac{{n\,({c:.0f})}}{{2({Lp})}} = {c/(2*Lp):.2f}\,n\ \mathrm{{Hz}}\ \Rightarrow\ {op[0]:.2f},\ {op[1]:.2f},\ {op[2]:.2f},\ {op[3]:.2f}\ \mathrm{{Hz}}",
          rf"f_n^{{ce}} = \dfrac{{(2n-1)({c:.0f})}}{{4({Lp})}} = {c/(4*Lp):.2f}\,(2n-1)\ \mathrm{{Hz}}\ \Rightarrow\ {cl[0]:.2f},\ {cl[1]:.2f},\ {cl[2]:.2f},\ {cl[3]:.2f}\ \mathrm{{Hz}}"]),
        ("Errores comunes y límites de validez",
         r"Se confunde la frecuencia de pulsación con la frecuencia del factor lento $\Delta f/2$ de la expresión de $y$, o se dice que las pulsaciones son un tono nuevo (no hay energía en la frecuencia $\Delta f$ del espectro de la señal). Con amplitudes distintas la envolvente no se anula y la modulación es menor. En los tubos, el error típico es ignorar que el extremo cerrado es un vientre de presión y no un nodo, y despreciar la corrección de extremo. El modelo unidimensional vale para $\lambda\gg$ el diámetro; para frecuencias mayores aparecen modos transversales.",
         [r"f_{corte}\simeq\dfrac{1.84\,c}{2\pi r}\ \ (\text{primer modo transversal en un tubo circular})"]),
    ]
    return dict(concept=concept)


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
