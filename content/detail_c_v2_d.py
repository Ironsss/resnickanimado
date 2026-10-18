"""Concepto y desarrollo analitico general, Volumen 2 capitulos 14 a 18 (optica y ondas electromagneticas).

Escenas: YeeFDTD, PoyntingDisplacement, SnellFermat, PrismDispersion, ThinLensImage,
TwoLensSystem, YoungSlits, ThinFilmCoating, SlitAndGrating, PolarizersMalus.
Cada entrada `concept` es una lista de (titulo, prosa, [ecuaciones TeX]): conceptos, deduccion
general desde principios (Maxwell, Fermat, matrices ABCD, Fraunhofer como transformada de Fourier),
formula general y particularizacion a los datos de la escena. Las cifras se calculan con `physics/`.
"""
from __future__ import annotations

import numpy as np

from content.texfmt import fx, sci
from physics import v2ch14_maxwell as m14
from physics import v2ch15_light as m15
from physics import v2ch16_lenses as m16
from physics import v2ch17_interference as m17
from physics import v2ch18_diffraction as m18
from physics.constants import C_LIGHT, EPS0, MU0


# ===================================================================== YeeFDTD
def _yee() -> dict:
    c0 = m14.c_from_constants()
    dx, cour, sg = 1e-3, 0.5, 2e-2
    dt = cour * dx / C_LIGHT
    z0 = MU0 * c0
    s5, s9 = m14.speed_numeric(0.5), m14.speed_numeric(0.9)
    e5, e9 = s5 / c0 - 1, s9 / c0 - 1
    p5, p9 = (1 - 0.5**2), (1 - 0.9**2)
    kd = dx / sg
    return dict(concept=[
        ("Qué se estudia y por qué importa",
         r"Las ecuaciones de Maxwell gobiernan toda la radiación electromagnética, pero solo unas pocas geometrías admiten solución exacta. El método de diferencias finitas en el dominio del tiempo (FDTD) integra las ecuaciones directamente, paso a paso, y es la herramienta habitual para diseñar antenas, guías de onda, circuitos de alta frecuencia, estructuras fotónicas y para estimar la sección eficaz de radar de un objeto. Esta escena estudia el caso más simple, un pulso en el vacío en una dimensión, porque allí se conoce la respuesta exacta y se puede medir con precisión cuánto se equivoca el esquema numérico.",
         []),
        ("Magnitudes y convenciones",
         r"El campo eléctrico $E$ se mide en V/m y la inducción magnética $B$ en T; el campo auxiliar $H=B/\mu_0$ se mide en A/m. Las constantes del vacío son $\varepsilon_0\approx 8.854\times10^{-12}$ F/m y $\mu_0=4\pi\times10^{-7}$ T m/A, y de ellas sale la velocidad $c\approx 3\times10^{8}$ m/s y la impedancia del vacío $Z_0=\sqrt{\mu_0/\varepsilon_0}\approx 377\ \Omega$. En la malla, $\Delta x$ es el paso espacial, $\Delta t$ el temporal y $S=c\,\Delta t/\Delta x$ el número de Courant, adimensional. Para resolver una onda de longitud $\lambda$ se piden al menos unas 10 a 20 celdas por longitud de onda.",
         [r"Z_0=\sqrt{\dfrac{\mu_0}{\varepsilon_0}},\qquad S=\dfrac{c\,\Delta t}{\Delta x}"]),
        ("Intuición y analogías",
         r"Cada campo es la fuente del otro: un $B$ que cambia induce un $E$ y un $E$ que cambia induce un $B$, como dos eslabones que se empujan alternadamente. Esa es la misma estructura que tiene una cuerda tensa, donde la altura y la velocidad se alimentan mutuamente. El esquema de Yee imita esta alternancia: $E$ se calcula en instantes enteros y $B$ en instantes semienteros, en un «salto de rana» en el que cada campo usa el valor más reciente del otro. Que ambos campos vivan en puntos distintos de la malla hace que cada derivada sea centrada y por tanto de segundo orden.",
         []),
        ("Las ecuaciones de Maxwell reducidas a una dimensión",
         r"Se parte de la ley de Faraday y de la ley de Ampère con el término de Maxwell en el vacío, sin cargas ni corrientes. Se supone una onda plana que depende solo de $x$ y $t$, con $\vec E=E_z(x,t)\hat z$ y $\vec B=B_y(x,t)\hat y$; esta elección es consistente porque $\nabla\cdot\vec E=\nabla\cdot\vec B=0$ se cumple de forma idéntica y porque los rotacionales de estos campos solo producen derivadas respecto de $x$. Al calcular los rotacionales, cada ley se reduce a una ecuación escalar, y las dos juntas forman un sistema hiperbólico de primer orden.",
         [r"\nabla\times\vec E=-\dfrac{\partial\vec B}{\partial t},\qquad \nabla\times\vec B=\mu_0\varepsilon_0\dfrac{\partial\vec E}{\partial t}",
          r"(\nabla\times\vec E)_y=-\dfrac{\partial E_z}{\partial x},\qquad (\nabla\times\vec B)_z=\dfrac{\partial B_y}{\partial x}",
          r"\dfrac{\partial B_y}{\partial t}=\dfrac{\partial E_z}{\partial x},\qquad \dfrac{\partial E_z}{\partial t}=\dfrac{1}{\mu_0\varepsilon_0}\dfrac{\partial B_y}{\partial x}"]),
        ("Ecuación de onda y solución de d'Alembert",
         r"Derivar la primera ecuación respecto de $t$ y la segunda respecto de $x$ permite eliminar $B_y$, porque las derivadas cruzadas conmutan si los campos son suficientemente suaves. Resulta la ecuación de onda de velocidad $c=1/\sqrt{\mu_0\varepsilon_0}$, cuya solución general es la suma de una onda que viaja a la derecha y otra a la izquierda. Sustituir una solución que viaja a la derecha en la ley de Faraday fija el cociente entre los campos, de modo que $E$ y $B$ están en fase y su relación es la velocidad de la onda. Con el campo auxiliar, el cociente $E/H$ es la impedancia del vacío.",
         [r"\dfrac{\partial^2E_z}{\partial t^2}=c^2\dfrac{\partial^2E_z}{\partial x^2},\qquad c=\dfrac{1}{\sqrt{\mu_0\varepsilon_0}}",
          r"E_z(x,t)=f(x-ct)+g(x+ct)",
          r"E_z=f(x-ct)\ \Rightarrow\ B_y=-\dfrac{f(x-ct)}{c},\qquad \dfrac{E_z}{H_y}=-Z_0"]),
        ("Conservación de la energía: teorema de Poynting unidimensional",
         r"Multiplicar la ley de Faraday por $B_y/\mu_0$ y la de Ampère por $\varepsilon_0E_z$, y sumar, conduce a una ley de conservación local sin necesidad de resolver la ecuación. La densidad de energía es $u=\tfrac12\varepsilon_0E^2+B^2/2\mu_0$ y el flujo es $S=-E_zB_y/\mu_0$ (el signo viene de la orientación $\hat z\times\hat y=-\hat x$). Esta ley es la que un buen esquema numérico debe respetar: el esquema de Yee conserva una energía discreta análoga, lo que explica su excelente comportamiento a tiempos largos.",
         [r"\dfrac{\partial u}{\partial t}+\dfrac{\partial S_x}{\partial x}=0,\qquad u=\tfrac12\varepsilon_0E_z^2+\dfrac{B_y^2}{2\mu_0}",
          r"S_x=-\dfrac{E_zB_y}{\mu_0}=\dfrac{E_z^2}{\mu_0c}\quad(\text{onda hacia }+x)"]),
        ("Discretización de Yee",
         r"Se colocan $E$ en los nodos $x_i=i\Delta x$ y $H$ en los puntos medios $x_{i+1/2}$, y $E$ en los tiempos $n\Delta t$ y $H$ en $(n+\tfrac12)\Delta t$. Cada derivada se aproxima por una diferencia centrada alrededor del punto donde se evalúa la ecuación, con error $O(\Delta x^2)$ y $O(\Delta t^2)$ (desarrollo de Taylor de $f(x\pm\Delta x/2)$). Al despejar el valor futuro resultan actualizaciones explícitas: no hay que resolver sistemas lineales, solo recorrer la malla.",
         [r"H_y\big|_{i+1/2}^{n+1/2}=H_y\big|_{i+1/2}^{n-1/2}+\dfrac{\Delta t}{\mu_0\Delta x}\left(E_z\big|_{i+1}^{n}-E_z\big|_{i}^{n}\right)",
          r"E_z\big|_{i}^{n+1}=E_z\big|_{i}^{n}+\dfrac{\Delta t}{\varepsilon_0\Delta x}\left(H_y\big|_{i+1/2}^{n+1/2}-H_y\big|_{i-1/2}^{n+1/2}\right)",
          r"\dfrac{f(x+\tfrac{\Delta x}2)-f(x-\tfrac{\Delta x}2)}{\Delta x}=f'(x)+\dfrac{\Delta x^2}{24}f'''(x)+\cdots"]),
        ("Relación de dispersión numérica y estabilidad",
         r"Se sustituye una onda plana discreta $E=e^{i(k\,i\Delta x-\omega n\Delta t)}$ (y la $H$ correspondiente) en las dos actualizaciones y se eliminan las amplitudes; es un análisis de von Neumann y es legítimo porque el esquema es lineal y de coeficientes constantes. El resultado liga la frecuencia numérica con el número de onda. Si $S\le1$ el argumento del arcoseno nunca excede 1 y $\omega$ es real para todo $k$ de la malla; si $S>1$, $\omega$ adquiere parte imaginaria y los modos crecen exponencialmente (inestabilidad). Al desarrollar para $k\Delta x\ll1$ se obtiene el error de velocidad de fase, que es de segundo orden y se anula en $S=1$.",
         [r"\sin\!\left(\dfrac{\omega\Delta t}{2}\right)=S\,\sin\!\left(\dfrac{k\Delta x}{2}\right)\ \Rightarrow\ \omega=\dfrac{2}{\Delta t}\arcsin\!\left[S\sin\dfrac{k\Delta x}{2}\right]",
          r"\dfrac{v_{fase}}{c}=\dfrac{\omega}{ck}=1-\dfrac{(1-S^2)\,(k\Delta x)^2}{24}+O\big((k\Delta x)^4\big)",
          r"S\le1\quad(\text{criterio de Courant-Friedrichs-Lewy en 1D})"]),
        ("Resultado general",
         r"La física exacta es $c=1/\sqrt{\mu_0\varepsilon_0}$, con $E=cB$ para una onda progresiva; el esquema discreto la reproduce con un error relativo de velocidad que depende solo del cociente $\Delta x$ sobre la escala del pulso y de $S$. El error es siempre negativo (la onda numérica va más lenta, y los componentes de $k$ alto más aún, por lo que el pulso se ensancha con una cola oscilante). Dimensionalmente, $\mu_0\varepsilon_0$ tiene unidades de s$^2$/m$^2$, de modo que $c$ resulta en m/s.",
         [r"\boxed{\,c=\dfrac{1}{\sqrt{\mu_0\varepsilon_0}},\qquad \dfrac{v_{num}-c}{c}\simeq-\dfrac{(1-S^2)}{24}\left(\dfrac{\Delta x}{\ell}\right)^2\,}",
          r"[\mu_0\varepsilon_0]=\dfrac{\mathrm{T\,m}}{\mathrm{A}}\cdot\dfrac{\mathrm{F}}{\mathrm{m}}=\dfrac{\mathrm{s^2}}{\mathrm{m^2}}"]),
        ("Particularización al problema",
         rf"Con las constantes del vacío sale $c$; con $\Delta x=1$ mm y $S=0.5$ el paso temporal es $\Delta t=S\Delta x/c$ y en 600 pasos el pulso avanza $0.3$ m. El pulso tiene $\sigma=2$ cm, así que $\Delta x/\sigma=0.05$ y la longitud característica $\ell$ es del orden de $\sigma$. El error predicho es proporcional a $1-S^2$: la razón entre los errores de $S=0.5$ y $S=0.9$ debe ser {p5/p9:.3f}, y la medida numérica da {e5/e9:.3f}, lo que confirma la ley. Los errores absolutos medidos son del orden de $10^{{-4}}$, pequeños pero no nulos, y se reducen al acercar $S$ a 1.",
         [rf"c=\dfrac{{1}}{{\sqrt{{({sci(MU0,6)})({sci(EPS0,6)})}}}}={sci(c0,7)}\ \mathrm{{m/s}},\qquad Z_0={fx(z0,3)}\ \Omega",
          rf"\Delta t=\dfrac{{S\,\Delta x}}{{c}}=\dfrac{{(0.5)(10^{{-3}})}}{{{sci(c0,7)}}}={sci(dt,4)}\ \mathrm{{s}},\qquad c\,(600\,\Delta t)={fx(C_LIGHT*600*dt,4)}\ \mathrm{{m}}",
          rf"\dfrac{{v_{{num}}-c}}{{c}}\Big|_{{S=0.5}}={sci(e5,3)},\qquad \dfrac{{v_{{num}}-c}}{{c}}\Big|_{{S=0.9}}={sci(e9,3)},\qquad \dfrac{{e_{{0.5}}}}{{e_{{0.9}}}}={e5/e9:.3f}\ \ \left(\text{{teoría }}\dfrac{{1-0.25}}{{1-0.81}}={p5/p9:.3f}\right)"]),
        ("Errores comunes y límites de validez",
         r"Un error frecuente es creer que reducir $\Delta t$ con $\Delta x$ fijo mejora la solución: como el error depende de $1-S^2$, disminuir $S$ lo empeora; la mejor elección en 1D es $S=1$, donde el esquema es exacto. Otra confusión es olvidar la condición de estabilidad: con $S>1$ la solución explota en pocas decenas de pasos. También se suele colocar $E$ y $B$ en el mismo nodo y en el mismo instante, lo que pierde la precisión de segundo orden. El modelo vale para el vacío lineal y sin fronteras; con paredes, medios dispersivos o dimensiones superiores se necesitan condiciones de contorno absorbentes (PML), relaciones constitutivas con memoria y un criterio de estabilidad más estricto ($S\le1/\sqrt{D}$ en $D$ dimensiones con mallas cúbicas).",
         [r"S_{max}=\dfrac{1}{\sqrt D}\quad(D=1,2,3)"]),
    ])


# ===================================================================== PoyntingDisplacement
def _poynting() -> dict:
    e0 = m14.E0
    ia = m14.intensity_analytic()
    b0 = e0 / C_LIGHT
    s0 = e0 * b0 / MU0
    p = m14.radiation_pressure(ia)
    p2 = m14.radiation_pressure(ia, False)
    idis = m14.conduction_current_analytic()
    cap = EPS0 * m14.AREA / m14.GAP
    u = EPS0 * e0**2 / 2
    return dict(concept=[
        ("Qué se estudia y por qué importa",
         r"Una onda electromagnética no solo viaja: transporta energía y cantidad de movimiento. Eso es lo que permite que el Sol caliente la Tierra, que un horno de microondas cocine y que una vela solar pueda acelerar una nave sin combustible. La escena reúne dos consecuencias de las ecuaciones de Maxwell: el flujo de energía de una onda plana (vector de Poynting y presión de radiación) y la corriente de desplazamiento que Maxwell añadió a la ley de Ampère para que la carga se conserve en circuitos con capacitores.",
         []),
        ("Magnitudes y convenciones",
         r"El vector de Poynting $\vec S$ se mide en W/m$^2$ (potencia por unidad de área perpendicular a la propagación); la intensidad $I=\langle S\rangle$ es su promedio temporal; la presión de radiación $p$ se mide en Pa. La densidad de energía $u$ se mide en J/m$^3$ y la densidad de cantidad de movimiento $g$ en kg/(m$^2$ s). La corriente de desplazamiento $I_d$ se mide en A y su densidad $\vec J_d=\varepsilon_0\partial_t\vec E$ en A/m$^2$. Los órdenes de magnitud son contrastantes: la luz solar da $I\sim10^{3}$ W/m$^2$ pero $p\sim10^{-6}$ Pa, porque $c$ es enorme.",
         [r"[\vec S]=\mathrm{W/m^2},\qquad [p]=\dfrac{[I]}{[c]}=\mathrm{Pa},\qquad [\varepsilon_0\partial_tE]=\mathrm{A/m^2}"]),
        ("Intuición y analogías",
         r"Piénsese en el campo como un fluido que lleva energía: la energía contenida en una columna de longitud $c\,\mathrm{d}t$ y sección $A$ cruza el área $A$ en un tiempo $\mathrm{d}t$, de modo que el flujo es la densidad de energía por la velocidad. Para la presión, la analogía es una lluvia de «granos» de cantidad de movimiento $u/c$ por unidad de volumen: si el absorbente los detiene, recibe $I/c$ por área y por segundo; si los devuelve, recibe el doble. La corriente de desplazamiento es la continuación de la corriente de conducción dentro del vacío entre las placas: lo que ya no es carga en movimiento es campo que cambia, y actúa como fuente de campo magnético exactamente igual.",
         []),
        ("Teorema de Poynting: balance local de energía",
         r"Se parte de las ecuaciones de Maxwell con corriente libre $\vec J$ en el vacío. Multiplicando la ley de Faraday por $\vec H=\vec B/\mu_0$ y la de Ampère-Maxwell por $\vec E$, restando, y usando la identidad vectorial $\nabla\cdot(\vec E\times\vec H)=\vec H\cdot\nabla\times\vec E-\vec E\cdot\nabla\times\vec H$, se obtiene el balance. El término $\vec J\cdot\vec E$ es la potencia por unidad de volumen que el campo entrega a las cargas, así que $\vec S=\vec E\times\vec H$ debe interpretarse como flujo de energía del campo. Integrando sobre un volumen se obtiene la forma global: la energía almacenada cambia por el trabajo sobre las cargas y por el flujo a través de la frontera.",
         [r"\nabla\times\vec E=-\mu_0\dfrac{\partial\vec H}{\partial t},\qquad \nabla\times\vec H=\vec J+\varepsilon_0\dfrac{\partial\vec E}{\partial t}",
          r"\dfrac{\partial u}{\partial t}+\nabla\cdot\vec S=-\vec J\cdot\vec E,\qquad u=\tfrac12\varepsilon_0E^2+\tfrac12\mu_0H^2,\qquad \vec S=\vec E\times\vec H",
          r"\dfrac{\mathrm{d}}{\mathrm{d}t}\int_V u\,\mathrm{d}V=-\oint_{\partial V}\vec S\cdot\mathrm{d}\vec A-\int_V\vec J\cdot\vec E\,\mathrm{d}V"]),
        ("Onda plana: campos, flujo e intensidad",
         r"Para una onda plana linealmente polarizada que viaja en $+x$ en el vacío, la ecuación de onda y la ley de Faraday dan $B=E/c$ y los campos en fase. Entonces $\vec S=\vec E\times\vec B/\mu_0$ apunta en la dirección de propagación y su módulo es $E^2/\mu_0c$, que oscila al doble de la frecuencia y nunca es negativo. Como la energía eléctrica y la magnética son iguales en cada instante, $u=\varepsilon_0E^2$ y $S=uc$. Los detectores promedian en tiempos largos frente al periodo, y el promedio de $\sin^2$ vale $1/2$; con $\mu_0=1/\varepsilon_0c^2$ la intensidad queda en función de la amplitud $E_0$.",
         [r"E=E_0\sin(kx-\omega t),\quad B=\dfrac{E}{c},\quad u_E=u_B=\tfrac12\varepsilon_0E^2",
          r"S(t)=\dfrac{E^2}{\mu_0c}=\varepsilon_0cE_0^2\sin^2(kx-\omega t)=uc",
          r"I=\langle S\rangle=\dfrac{E_0^2}{2\mu_0c}=\tfrac12c\,\varepsilon_0E_0^2=\dfrac{E_{rms}^2}{\mu_0c}"]),
        ("Cantidad de movimiento y presión de radiación",
         r"Un campo electromagnético transporta cantidad de movimiento con densidad $\vec g=\vec S/c^2$, resultado que sale del tensor de esfuerzos de Maxwell o, de forma elemental, de que una carga que oscila en el campo $E$ sufre la fuerza magnética $q\vec v\times\vec B$ en la dirección de propagación. Si el material absorbe toda la onda, el momento que llega por unidad de área y de tiempo es $g\,c=I/c$; si la refleja perfectamente, el momento se invierte y el cambio es el doble. La presión es entonces proporcional a la intensidad con el factor $1/c$, y vale entre $1$ (absorbente) y $2$ (espejo) veces $I/c$, según la reflectividad $\rho$.",
         [r"\vec g=\dfrac{\vec S}{c^2},\qquad \dfrac{\mathrm{d}p_{mec}}{\mathrm{d}A\,\mathrm{d}t}=\dfrac{I}{c}\ (\text{absorbente})",
          r"p=\dfrac{(1+\rho)\,I}{c},\qquad 0\le\rho\le1,\qquad u=\dfrac{I}{c}\ \Rightarrow\ p=u_{incidente}(1+\rho)"]),
        ("Corriente de desplazamiento y conservación de la carga",
         r"La ley de Ampère $\nabla\times\vec B=\mu_0\vec J$ es inconsistente en régimen variable: tomando la divergencia, el lado izquierdo es siempre cero pero $\nabla\cdot\vec J=-\partial_t\rho\neq0$ cuando se carga un capacitor. Con la ley de Gauss $\nabla\cdot\vec E=\rho/\varepsilon_0$ se ve que el término que falta es $\varepsilon_0\partial_t\vec E$. Añadirlo restituye la consistencia con la ecuación de continuidad y hace que la corriente total $\vec J+\varepsilon_0\partial_t\vec E$ sea de divergencia nula (sin fuentes ni sumideros), de modo que cierra todos los circuitos, incluso a través del vacío entre placas.",
         [r"\nabla\cdot(\nabla\times\vec B)=0\ \Rightarrow\ \mu_0\nabla\cdot\vec J=0\quad(\text{contradice }\nabla\cdot\vec J=-\partial_t\rho)",
          r"\nabla\cdot\left(\vec J+\varepsilon_0\dfrac{\partial\vec E}{\partial t}\right)=\nabla\cdot\vec J+\dfrac{\partial\rho}{\partial t}=0",
          r"\nabla\times\vec B=\mu_0\left(\vec J+\vec J_d\right),\qquad \vec J_d=\varepsilon_0\dfrac{\partial\vec E}{\partial t}"]),
        ("Capacitor de placas paralelas",
         r"Entre placas de área $A$ y separación $d$ pequeña frente a las dimensiones laterales, el campo es uniforme, $E=V/d$ (se desprecian los bordes), y el flujo es $\Phi_E=EA$. La corriente de desplazamiento es la integral de $\varepsilon_0\partial_tE$ sobre el área, que no es sino $\varepsilon_0\,\mathrm{d}\Phi_E/\mathrm{d}t$. Al sustituir se reconoce la capacidad $C=\varepsilon_0A/d$, y el resultado coincide con la corriente de conducción $\mathrm{d}Q/\mathrm{d}t=C\,\mathrm{d}V/\mathrm{d}t$ en los cables: la corriente es continua a lo largo del circuito.",
         [r"I_d=\int_A\varepsilon_0\dfrac{\partial E}{\partial t}\mathrm{d}A=\varepsilon_0\dfrac{\mathrm{d}\Phi_E}{\mathrm{d}t}=\varepsilon_0\dfrac{A}{d}\dfrac{\mathrm{d}V}{\mathrm{d}t}",
          r"C=\dfrac{\varepsilon_0A}{d},\qquad I_d=C\dfrac{\mathrm{d}V}{\mathrm{d}t}=\dfrac{\mathrm{d}Q}{\mathrm{d}t}=I_{cond}"]),
        ("Resultado general",
         r"Los resultados generales son tres: la intensidad de una onda plana de amplitud $E_0$, la presión de radiación para una reflectividad $\rho$ y la corriente de desplazamiento de un capacitor de geometría arbitraria. La verificación dimensional se hace con $[\varepsilon_0]=\mathrm{F/m}$, $[c]=\mathrm{m/s}$ y $[E]=\mathrm{V/m}$; las tres fórmulas crecen como $E_0^2$ las dos primeras y linealmente con $\mathrm{d}V/\mathrm{d}t$ la tercera. Para $\rho=1$ se recupera el factor 2 del espejo y para $\rho=0$ el absorbente.",
         [r"\boxed{\,I=\tfrac12c\,\varepsilon_0E_0^2,\qquad p=\dfrac{(1+\rho)\,I}{c},\qquad I_d=\varepsilon_0\dfrac{\mathrm{d}\Phi_E}{\mathrm{d}t}=\dfrac{\varepsilon_0A}{d}\dfrac{\mathrm{d}V}{\mathrm{d}t}\,}",
          r"\left[\tfrac12c\varepsilon_0E_0^2\right]=\dfrac{\mathrm m}{\mathrm s}\dfrac{\mathrm F}{\mathrm m}\dfrac{\mathrm V^2}{\mathrm m^2}=\dfrac{\mathrm{J}}{\mathrm{s\,m^2}}=\dfrac{\mathrm W}{\mathrm m^2}"]),
        ("Particularización al problema",
         rf"Con $E_0=100$ V/m, $\rho=0$ para el absorbente perfecto y $\rho=1$ para el espejo, la intensidad, la densidad de energía y las presiones salen directamente de las fórmulas generales. Para el capacitor se usan $A=100\ \mathrm{{cm^2}}=0.010\ \mathrm{{m^2}}$, $d=1.0$ mm y $\mathrm{{d}}V/\mathrm{{d}}t=10^6$ V/s. Los resultados coinciden con los de la solución analítica de la escena: $I={ia:.4f}$ W/m$^2$, $p={p*1e9:.4f}$ nPa e $I_d={idis*1e6:.4f}\ \mu$A.",
         [rf"I=\tfrac12({sci(C_LIGHT,7)})({sci(EPS0,6)})({e0:.0f})^2={ia:.4f}\ \mathrm{{W/m^2}},\qquad u=\dfrac{{I}}{{c}}={sci(u,4)}\ \mathrm{{J/m^3}}",
          rf"p_{{\rho=0}}=\dfrac{{I}}{{c}}={p*1e9:.4f}\ \mathrm{{nPa}},\qquad p_{{\rho=1}}=\dfrac{{2I}}{{c}}={p2*1e9:.4f}\ \mathrm{{nPa}},\qquad S_{{max}}={s0:.4f}\ \mathrm{{W/m^2}}",
          rf"C=\dfrac{{({sci(EPS0,6)})({m14.AREA})}}{{{sci(m14.GAP,2)}}}={sci(cap,5)}\ \mathrm{{F}},\qquad I_d=C\dfrac{{\mathrm{{d}}V}}{{\mathrm{{d}}t}}=({sci(cap,5)})({sci(m14.DVDT,2)})={idis*1e6:.4f}\ \mu\mathrm{{A}}"]),
        ("Errores comunes y límites de validez",
         r"Se confunde con frecuencia la intensidad con el valor pico: $S_{max}=2I$, porque el promedio de $\sin^2$ es $1/2$. Otro error es usar $E_0$ como valor eficaz o viceversa, lo que introduce un factor 2 en $I$. La presión de radiación no depende de la frecuencia en este modelo clásico, pero la luz solar real tiene que considerar absorción parcial y, a incidencia oblicua, el factor $\cos^2\theta$. Finalmente, la corriente de desplazamiento no es una corriente de cargas: no calienta ni transporta carga, solo genera campo magnético. La fórmula del capacitor desprecia efectos de borde y supone variación lenta (régimen cuasiestático), válida mientras el tamaño del capacitor sea mucho menor que la longitud de onda de las variaciones $c/f$.",
         []),
    ])


# ===================================================================== SnellFermat
def _snell() -> dict:
    h1, h2, n1, n2 = m15.H1, m15.H2, m15.N1, m15.N2
    th1 = np.deg2rad(m15.THETA1_DEG)
    th2 = m15.snell_angle(th1)
    xs = m15.refraction_point_analytic()
    d, _ = m15.target_point()
    s1, s2 = np.hypot(xs, h1), np.hypot(d - xs, h2)
    tc = m15.critical_angle_analytic()
    opl = n1 * s1 + n2 * s2
    t_min = opl / C_LIGHT
    t2 = (n1 * h1**2 / s1**3 + n2 * h2**2 / s2**3) / C_LIGHT
    return dict(concept=[
        ("Qué se estudia y por qué importa",
         r"Cuando la luz cruza la frontera entre dos medios cambia de dirección. Esa desviación es la base de las lentes, de los lentes de contacto, de la fibra óptica y del espejismo sobre el asfalto caliente. La ley de Snell la describe, pero ¿por qué existe? El principio de Fermat responde: la luz sigue entre dos puntos el camino de tiempo estacionario. Esta escena enlaza ambas ideas y calcula también el ángulo crítico, que gobierna la reflexión total interna.",
         []),
        ("Magnitudes y convenciones",
         r"El índice de refracción $n=c/v$ es adimensional ($n=1$ en el vacío, $\approx1.0003$ en el aire, $1.33$ en el agua, $1.5$ en un vidrio común). Los ángulos $\theta_1,\theta_2$ se miden respecto de la normal a la interfaz, no respecto de la superficie. El camino óptico es $\mathrm{OPL}=\int n\,\mathrm{d}s$, medido en metros, y el tiempo de viaje es $T=\mathrm{OPL}/c$. El ángulo crítico existe solo al pasar de un medio de mayor índice a otro de menor índice.",
         [r"n=\dfrac{c}{v},\qquad \mathrm{OPL}=\int_A^Bn\,\mathrm{d}s,\qquad T=\dfrac{\mathrm{OPL}}{c}"]),
        ("Intuición y analogías",
         r"Un salvavidas corre más rápido por la arena que nadando; para llegar lo antes posible a un bañista no va en línea recta, sino que alarga el tramo en la arena y acorta el tramo en el agua. Eso es exactamente la refracción: el rayo «quiebra» donde cambia la velocidad. El tren de ruedas que pasa de pavimento a barro y gira hacia el lado más lento es otra imagen mecánica. En ambos casos la ley de Snell es la condición de óptimo y no una regla ad hoc.",
         []),
        ("Principio de Fermat y funcional de tiempo",
         r"Un camino $y(x)$ entre dos puntos fijos tiene un tiempo $T[y]=\frac1c\int n(x,y)\sqrt{1+y'^2}\,\mathrm{d}x$. El principio de Fermat postula $\delta T=0$ (estacionario; casi siempre mínimo local). Es el análogo óptico del principio de mínima acción: el integrando $\mathcal L=n\sqrt{1+y'^2}$ hace el papel de lagrangiano y sus ecuaciones de Euler-Lagrange son las ecuaciones del rayo. En un medio homogéneo $n$ es constante, el lagrangiano solo depende de $y'$, y los rayos son rectas.",
         [r"\delta\int_A^Bn(\vec r)\,\mathrm{d}s=0",
          r"\dfrac{\mathrm{d}}{\mathrm{d}x}\dfrac{\partial\mathcal L}{\partial y'}-\dfrac{\partial\mathcal L}{\partial y}=0,\qquad \mathcal L=n\sqrt{1+y'^2}",
          r"\dfrac{\mathrm{d}}{\mathrm{d}s}\!\left(n\dfrac{\mathrm{d}\vec r}{\mathrm{d}s}\right)=\nabla n\quad(\text{ecuación del rayo})"]),
        ("Simetría y ley de conservación: ley de Snell",
         r"Si $n$ depende solo de $z$ (medio estratificado, interfaz plana), la configuración es invariante por traslaciones a lo largo de la interfaz y el teorema de Noether da una cantidad conservada: la componente tangencial de $n\,\hat s$, es decir $n\sin\theta$. En una interfaz abrupta esa constancia se verifica en el salto, lo que es la ley de Snell. Alternativamente, se escribe el tiempo de viaje con un punto de incidencia $x$ y se impone $\mathrm{d}T/\mathrm{d}x=0$; ambos procedimientos son equivalentes, el segundo es más directo para dos medios homogéneos.",
         [r"T(x)=\dfrac{1}{c}\left[n_1\sqrt{x^2+h_1^2}+n_2\sqrt{(d-x)^2+h_2^2}\right]",
          r"\dfrac{\mathrm{d}T}{\mathrm{d}x}=\dfrac{1}{c}\left[n_1\sin\theta_1-n_2\sin\theta_2\right]=0,\quad \sin\theta_1=\dfrac{x}{s_1},\ \sin\theta_2=\dfrac{d-x}{s_2}",
          r"n_1\sin\theta_1=n_2\sin\theta_2"]),
        ("Es un mínimo y la geometría completa",
         r"La segunda derivada de $T$ respecto de $x$ es suma de dos términos positivos, de modo que el punto estacionario es un mínimo y es único: el camino real es el más rápido y no un máximo ni una silla. El problema geométrico se cierra relacionando la distancia horizontal total con los dos ángulos. A partir de estas tres relaciones se resuelven los problemas directo (conocidos $\theta_1$ y los medios, hallar $\theta_2$ y el destino) e inverso (conocidos ambos puntos, hallar el punto de incidencia).",
         [r"\dfrac{\mathrm{d}^2T}{\mathrm{d}x^2}=\dfrac{1}{c}\left[\dfrac{n_1h_1^2}{s_1^3}+\dfrac{n_2h_2^2}{s_2^3}\right]>0",
          r"d=h_1\tan\theta_1+h_2\tan\theta_2,\qquad x^*=h_1\tan\theta_1",
          r"\theta_2=\arcsin\!\left(\dfrac{n_1}{n_2}\sin\theta_1\right)"]),
        ("Reflexión total y ángulo crítico",
         r"Si el rayo pasa a un medio de menor índice ($n_1>n_2$), la ley de Snell exige $\sin\theta_2=(n_1/n_2)\sin\theta_1$, que supera 1 si $\theta_1$ excede un valor límite; entonces no hay solución real, la onda refractada es evanescente y toda la energía se refleja. El ángulo límite se obtiene de imponer $\theta_2=90^\circ$. El parámetro adimensional relevante es el cociente de índices $n_2/n_1$; cuanto menor es, más estrecho es el cono de transmisión, lo que explica el brillo del diamante ($n=2.42$) y el confinamiento en fibras.",
         [r"\sin\theta_c=\dfrac{n_2}{n_1}\quad(n_1>n_2),\qquad \theta_c=\arcsin\dfrac{n_2}{n_1}",
          r"\theta_1>\theta_c\ \Rightarrow\ \text{reflexión total},\qquad k_{2z}=\mathrm i\,k_0\sqrt{n_1^2\sin^2\theta_1-n_2^2}"]),
        ("Resultado general",
         r"Los resultados generales son la ley de Snell, el punto de incidencia en función de la geometría y el ángulo crítico, todos adimensionales o en metros y válidos para cualquier par de medios homogéneos e isótropos y para cualquier distancia. La verificación dimensional es inmediata: los ángulos y los índices no tienen unidades y $x^*$ tiene las de $h_1$. El tiempo mínimo $T_{min}=(n_1s_1+n_2s_2)/c$ es el camino óptico dividido entre $c$ y es invariante bajo la inversión del rayo (reciprocidad).",
         [r"\boxed{\,n_1\sin\theta_1=n_2\sin\theta_2,\quad x^*=h_1\tan\theta_1,\quad \theta_c=\arcsin\dfrac{n_2}{n_1}\,}",
          r"T_{min}=\dfrac{n_1s_1+n_2s_2}{c},\qquad s_1=\dfrac{h_1}{\cos\theta_1},\ s_2=\dfrac{h_2}{\cos\theta_2}"]),
        ("Particularización al problema",
         rf"Se sustituyen $n_1=1.0$ (aire), $n_2=1.5$ (vidrio), $h_1=1.0$ m, $h_2=1.5$ m y $\theta_1=40^\circ$. El resultado recupera el ángulo refractado, el punto de incidencia y el ángulo crítico de la solución analítica de la escena, y la convexidad de $T$ confirma que el camino es mínimo.",
         [rf"\theta_2=\arcsin\!\left(\dfrac{{\sin 40^\circ}}{{1.5}}\right)={np.degrees(th2):.4f}^\circ,\qquad x^*=(1.0)\tan 40^\circ={xs:.5f}\ \mathrm{{m}},\qquad d={d:.5f}\ \mathrm{{m}}",
          rf"s_1={s1:.5f}\ \mathrm{{m}},\ s_2={s2:.5f}\ \mathrm{{m}},\ \ \mathrm{{OPL}}={opl:.5f}\ \mathrm{{m}},\ \ T_{{min}}={sci(t_min,5)}\ \mathrm{{s}}",
          rf"c\,T''(x^*)={t2*C_LIGHT:.5f}\ \mathrm{{m^{{-1}}}}>0,\qquad \theta_c=\arcsin\dfrac{{1.0}}{{1.5}}={tc:.4f}^\circ"]),
        ("Errores comunes y límites de validez",
         r"Es frecuente medir los ángulos respecto de la superficie en lugar de la normal, o aplicar el ángulo crítico al pasar de un medio menos denso a uno más denso, donde no existe. Tampoco el rayo sigue «el camino más corto», sino el de tiempo estacionario: en lentes o espejos curvos puede ser un máximo o un punto de inflexión. El modelo de rayos falla cuando las dimensiones son comparables a $\lambda$ (difracción), cuando el índice varía en escalas menores que $\lambda$, y en la reflexión total, donde aparece una onda evanescente que penetra del orden de $\lambda$ y permite el efecto túnel óptico entre dos medios cercanos (reflexión total frustrada). La reflectividad en la interfaz la dan las fórmulas de Fresnel, no la ley de Snell.",
         []),
    ])


# ===================================================================== PrismDispersion
def _prism() -> dict:
    A = m15.APEX_DEG
    ar = np.deg2rad(A)
    lams = (400, 550, 700)
    ns = {l: float(m15.cauchy(l)) for l in lams}
    da = {l: float(m15.min_deviation_analytic(ns[l])) for l in lams}
    th = {l: np.degrees(np.arcsin(ns[l] * np.sin(ar / 2))) for l in lams}
    n = ns[550]
    dd_dn = 2 * np.sin(ar / 2) / np.sqrt(1 - (n * np.sin(ar / 2)) ** 2)  # rad por unidad de n
    dn_dl = -2 * 5000 / 550**3  # 1/nm
    disp = np.degrees(dd_dn * dn_dl) * 1e3  # grados por micrometro
    amax = 2 * np.degrees(np.arcsin(1 / ns[400]))
    return dict(concept=[
        ("Qué se estudia y por qué importa",
         r"Un prisma de vidrio descompone la luz blanca porque el índice de refracción depende del color. Es el fenómeno detrás del arco iris, de la aberración cromática de las lentes y del espectrómetro de prisma que, desde Newton, permite medir la composición de una fuente luminosa y el índice de un vidrio. Aquí se obtiene la desviación de un rayo en función de la longitud de onda, se halla su valor mínimo (el que se usa en la práctica) y se cuantifica la separación angular entre el violeta y el rojo.",
         []),
        ("Magnitudes y convenciones",
         r"El ángulo de vértice $\alpha$ (60° aquí) es el que forman las dos caras; $\theta_1$ es el ángulo de incidencia en la primera cara, $r_1$ y $r_2$ los ángulos internos medidos desde cada normal y $\theta_2$ el de salida. La desviación $\delta$ es el ángulo entre el rayo incidente y el emergente. La ley de Cauchy $n(\lambda)=A+B/\lambda^2$ (con $A$ adimensional y $B$ en nm$^2$) describe la dispersión normal en el visible: $n$ decrece al crecer $\lambda$, así que el violeta se desvía más que el rojo.",
         [r"n(\lambda)=A+\dfrac{B}{\lambda^2},\qquad \frac{\mathrm{d}n}{\mathrm{d}\lambda}=-\frac{2B}{\lambda^3}<0"]),
        ("Intuición y analogías",
         r"Cada color atraviesa el vidrio a una velocidad distinta, $v=c/n(\lambda)$, así que cada color «gira» una cantidad diferente en la interfaz, igual que un carrito de juguete cuyas ruedas de un lado tocan arena antes que las del otro. Como las dos caras del prisma no son paralelas, la desviación de la entrada y la de la salida se suman en vez de cancelarse (en una placa de caras paralelas el rayo sale paralelo al incidente y solo se desplaza lateralmente). Esa falta de paralelismo es lo que convierte un cambio pequeño de $n$ en una separación angular apreciable.",
         []),
        ("Geometría del rayo y ley de Snell en las dos caras",
         r"Se aplica la ley de Snell en cada cara y se relaciona la geometría interior del prisma. Los dos normales de las caras forman un ángulo $\alpha$ y el rayo interior, con los ángulos $r_1$ y $r_2$ respecto de ellas, cierra un triángulo con el vértice, de modo que $r_1+r_2=\alpha$ (suma de ángulos del triángulo). La desviación total es la suma de lo que se desvía en cada refracción, $(\theta_1-r_1)+(\theta_2-r_2)$.",
         [r"\sin\theta_1=n\sin r_1,\qquad n\sin r_2=\sin\theta_2",
          r"r_1+r_2=\alpha,\qquad \delta=(\theta_1-r_1)+(\theta_2-r_2)=\theta_1+\theta_2-\alpha"]),
        ("Condición de desviación mínima",
         r"Como $\delta$ depende de $\theta_1$ (con $\alpha$ y $n$ fijos), se busca el extremo: se deriva $\delta$ respecto de $\theta_1$ usando $\mathrm{d}r_2=-\mathrm{d}r_1$ y las derivadas de las leyes de Snell. La condición $\mathrm{d}\theta_2/\mathrm{d}\theta_1=-1$ se reduce, tras elevar al cuadrado y usar $\cos^2 r=1-\sin^2\theta/n^2$, a $(n^2-1)(\sin^2\theta_2-\sin^2\theta_1)=0$. Como $n>1$, la única solución es $\theta_1=\theta_2$: el rayo atraviesa el prisma simétricamente. Es un mínimo porque $\delta\to$ grande cerca de la incidencia rasante y cerca de la reflexión total.",
         [r"\dfrac{\mathrm{d}\theta_2}{\mathrm{d}\theta_1}=-\dfrac{\cos\theta_1\cos r_2}{\cos\theta_2\cos r_1}=-1\ \Rightarrow\ \cos\theta_1\cos r_2=\cos\theta_2\cos r_1",
          r"(n^2-1)\left(\sin^2\theta_2-\sin^2\theta_1\right)=0\ \Rightarrow\ \theta_1=\theta_2,\quad r_1=r_2=\dfrac\alpha2"]),
        ("Fórmula de la desviación mínima y medida del índice",
         r"En el mínimo, $r_1=\alpha/2$ y $\sin\theta_1=n\sin(\alpha/2)$, y la desviación es $\delta_{min}=2\theta_1-\alpha$. Invirtiendo la relación se obtiene el índice a partir de dos ángulos medibles en un goniómetro, que es el método estándar de caracterización de vidrios. La fórmula exige $n\sin(\alpha/2)\le1$; si no, el rayo no emerge (reflexión total en la segunda cara) y por tanto existe un ángulo de vértice máximo que depende del color.",
         [r"\delta_{min}=2\arcsin\!\left(n\sin\dfrac\alpha2\right)-\alpha",
          r"n=\dfrac{\sin\!\left[(\alpha+\delta_{min})/2\right]}{\sin(\alpha/2)}",
          r"\alpha<\alpha_{max}=2\arcsin\dfrac1n"]),
        ("Dispersión angular y poder dispersor",
         r"La separación angular entre dos colores se obtiene por la regla de la cadena: la variación de $\delta_{min}$ con $n$ (propiedad geométrica del prisma) por la de $n$ con $\lambda$ (propiedad del material). Para ángulos pequeños, $\delta\approx(n-1)\alpha$ y la dispersión es $\alpha\,\mathrm{d}n/\mathrm{d}\lambda$. El poder resolutor de un prisma de base $b$ es el producto de esa base por $|\mathrm{d}n/\mathrm{d}\lambda|$: prismas mayores y vidrios más dispersivos resuelven mejor.",
         [r"\dfrac{\mathrm{d}\delta_{min}}{\mathrm{d}n}=\dfrac{2\sin(\alpha/2)}{\sqrt{1-n^2\sin^2(\alpha/2)}}=\dfrac{2\sin(\alpha/2)}{\cos\theta_1}",
          r"\dfrac{\mathrm{d}\delta_{min}}{\mathrm{d}\lambda}=\dfrac{\mathrm{d}\delta_{min}}{\mathrm{d}n}\,\dfrac{\mathrm{d}n}{\mathrm{d}\lambda}=-\dfrac{4B\sin(\alpha/2)}{\lambda^3\cos\theta_1}",
          r"\mathcal R=\dfrac{\lambda}{\Delta\lambda}=b\left|\dfrac{\mathrm{d}n}{\mathrm{d}\lambda}\right|"]),
        ("Resultado general",
         r"La desviación mínima y su dispersión quedan determinadas por el ángulo de vértice y la función $n(\lambda)$. Todo es adimensional (ángulos) salvo $\mathrm{d}\delta/\mathrm{d}\lambda$, que se expresa en grados por nanómetro. Casos límite: para $\alpha\to0$ se recupera la placa de caras paralelas ($\delta\to0$); para $\alpha$ pequeño $\delta_{min}\approx(n-1)\alpha$.",
         [r"\boxed{\,\delta_{min}(\lambda)=2\arcsin\!\left[n(\lambda)\sin\dfrac\alpha2\right]-\alpha,\qquad \Delta\delta=\delta_{min}(\lambda_1)-\delta_{min}(\lambda_2)\,}",
          r"\delta_{min}\xrightarrow[\alpha\ll1]{}(n-1)\,\alpha,\qquad \left[\dfrac{\mathrm{d}\delta}{\mathrm{d}\lambda}\right]=\dfrac{\mathrm{rad}}{\mathrm{nm}}"]),
        ("Particularización al problema",
         rf"Se usa $\alpha=60^\circ$ y $n=1.50+5000/\lambda^2$ con $\lambda$ en nm. En el mínimo el rayo interior forma $30^\circ$ con cada normal. Los resultados coinciden con la solución analítica de la escena. Además, la estimación diferencial en 550 nm reproduce el orden de magnitud de la pendiente (cuyo valor medio entre 400 y 700 nm es {(da[400]-da[700])/0.3:.4f} grados por micrómetro, frente a {abs(disp):.4f} en 550 nm: la dispersión es mayor en el azul).",
         [rf"n(400)={ns[400]:.5f},\quad n(550)={ns[550]:.5f},\quad n(700)={ns[700]:.5f}",
          rf"\theta_1(400)={th[400]:.4f}^\circ,\ \delta_{{min}}(400)={da[400]:.4f}^\circ;\qquad \delta_{{min}}(550)={da[550]:.4f}^\circ;\qquad \delta_{{min}}(700)={da[700]:.4f}^\circ",
          rf"\Delta\delta=\delta_{{min}}(400)-\delta_{{min}}(700)={da[400]-da[700]:.4f}^\circ,\qquad \dfrac{{\mathrm{{d}}\delta}}{{\mathrm{{d}}\lambda}}\Big|_{{550}}={np.degrees(dd_dn*dn_dl):.5f}^\circ/\mathrm{{nm}}"]),
        ("Errores comunes y límites de validez",
         r"Se suele confundir el ángulo de vértice con el ángulo de incidencia, o aplicar la fórmula de mínima desviación a una incidencia cualquiera (para otros ángulos hay que trazar el rayo con Snell dos veces). También se cree que el rojo se desvía más: en dispersión normal ocurre lo contrario. La fórmula de Cauchy es empírica y falla cerca de bandas de absorción (dispersión anómala), donde $n$ crece con $\lambda$; hay que usar la ecuación de Sellmeier o el modelo de Lorentz. Para rayos que pasan cerca de la base o del vértice, la reflexión total en la segunda cara limita el ángulo $\alpha$ útil; y la aproximación geométrica ignora la difracción por la abertura, que limita la resolución real.",
         []),
    ])


# ===================================================================== ThinLensImage
def _lens() -> dict:
    n, r1, r2 = m16.N_GLASS, m16.R1, m16.R2
    f = m16.focal_lensmaker()
    s = 0.3
    sp = m16.image_distance_analytic(s, f)
    rm, sm = 0.4, 0.6
    spm = m16.mirror_image_analytic(sm, rm)
    pw = 1 / f
    return dict(concept=[
        ("Qué se estudia y por qué importa",
         r"Casi todo instrumento óptico (cámara, ojo, microscopio, telescopio, proyector) se basa en formar una imagen con lentes y espejos. Esta escena obtiene la ley que relaciona las posiciones de objeto e imagen y la potencia de una lente a partir de la geometría de sus superficies, y la aplica a una lente biconvexa y a un espejo cóncavo. El mismo formalismo, escrito con matrices, es el que usan los programas de diseño óptico.",
         []),
        ("Magnitudes y convenciones",
         r"La distancia focal $f$ y las distancias $s$ (objeto) y $s'$ (imagen) se miden en metros desde el vértice; $R_1,R_2$ son los radios de curvatura con signo (positivo si el centro de curvatura está del lado de salida de la luz). Aquí $s>0$ para objeto real, $s'>0$ para imagen real y $s'<0$ para imagen virtual. El aumento lateral $m=-s'/s$ es adimensional y su signo indica si la imagen está invertida ($m<0$). La potencia $P=1/f$ se mide en dioptrías ($\mathrm{m^{-1}}$): una lente de $f=10$ cm tiene $P=10$ D.",
         [r"P=\dfrac1f\ \ [\mathrm{D}=\mathrm{m^{-1}}],\qquad m=\dfrac{y'}{y}=-\dfrac{s'}{s}"]),
        ("Intuición y analogías",
         r"Una lente convergente actúa como un prisma cuya desviación crece con la distancia al eje: los rayos más alejados se doblan más y por eso todos los que salen de un punto se vuelven a cruzar en otro. Un espejo cóncavo cumple la misma función por reflexión. La aproximación paraxial dice que, cerca del eje, la desviación es proporcional a la altura del rayo, de modo que toda la óptica de primer orden es lineal; esa linealidad es lo que permite resumir una lente en un solo número, $f$, y representar cualquier sistema con una matriz $2\times2$.",
         []),
        ("Principio de Fermat en una superficie esférica",
         r"Se considera una superficie esférica de radio $R$ entre los medios $n_1$ y $n_2$, un punto objeto axial a distancia $s$ y su imagen a $s'$. Un rayo que toca la superficie a la altura $h$ tiene un camino óptico que se desarrolla a segundo orden en $h$ usando $\sqrt{a^2+h^2}\approx a+h^2/2a$ y la altura de la superficie (sagita) $z\approx h^2/2R$. Para que haya imagen perfecta en el orden paraxial, el camino óptico debe ser estacionario respecto de $h$ para todo rayo: el coeficiente de $h^2$ debe anularse, lo que da la ecuación de la superficie refractante.",
         [r"\ell_1\approx s+\dfrac{h^2}{2R}+\dfrac{h^2}{2s},\qquad \ell_2\approx s'-\dfrac{h^2}{2R}+\dfrac{h^2}{2s'}",
          r"\mathrm{OPL}=n_1\ell_1+n_2\ell_2=\text{cte}+\dfrac{h^2}{2}\left[\dfrac{n_1}{s}+\dfrac{n_2}{s'}-\dfrac{n_2-n_1}{R}\right]",
          r"\dfrac{n_1}{s}+\dfrac{n_2}{s'}=\dfrac{n_2-n_1}{R}"]),
        ("Lente delgada y ecuación del fabricante",
         r"Una lente son dos superficies; la imagen de la primera es el objeto de la segunda. En el límite de espesor nulo, la distancia de la imagen intermedia se suma con signo negativo a la de objeto de la segunda y los términos intermedios se cancelan al sumar las dos ecuaciones. Con aire ($n_1=n_3=1$) y vidrio ($n_2=n$) queda la ecuación de las lentes delgadas, y de ella la focal (imagen de un objeto en el infinito). Las potencias de las dos superficies simplemente se suman cuando están en contacto.",
         [r"\dfrac{1}{s}+\dfrac{n}{s_1'}=\dfrac{n-1}{R_1},\qquad -\dfrac{n}{s_1'}+\dfrac{1}{s'}=\dfrac{1-n}{R_2}",
          r"\dfrac1s+\dfrac1{s'}=(n-1)\left(\dfrac1{R_1}-\dfrac1{R_2}\right)\equiv\dfrac1f",
          r"P=P_1+P_2,\qquad P_i=\dfrac{n_{sal}-n_{ent}}{R_i}"]),
        ("Ecuación de las lentes, aumento y forma de Newton",
         r"La ecuación gaussiana liga $s$, $s'$ y $f$. El aumento lateral sale de la semejanza de triángulos con el rayo que pasa por el centro óptico (que no se desvía en una lente delgada). Si las distancias se miden desde los focos, $x=s-f$ y $x'=s'-f$, la relación se simplifica a $xx'=f^2$, útil para ver cómo se mueve la imagen: acercar un objeto hacia el foco la aleja hacia el infinito. Los casos límite son: objeto en el infinito ($s'=f$); objeto en $2f$ ($s'=2f$, $m=-1$); objeto dentro de $f$ (imagen virtual, derecha y mayor, la lupa).",
         [r"s'=\dfrac{sf}{s-f},\qquad m=-\dfrac{s'}{s}=-\dfrac{f}{s-f}",
          r"(s-f)(s'-f)=f^2",
          r"\text{Aumento angular de una lupa: }M\approx1+\dfrac{25\ \mathrm{cm}}{f}"]),
        ("Espejo esférico como caso particular",
         r"La reflexión se obtiene de la misma ecuación de superficie con $n_2=-n_1$ (el rayo vuelve al mismo medio, lo que se codifica como un índice efectivo negativo). Entonces desaparece $n$ y queda una ecuación universal con $f=R/2$: para un rayo paraxial paralelo al eje, la ley de reflexión sobre la normal radial lo envía al punto medio entre el vértice y el centro. La coincidencia con la lente muestra que lo importante es la potencia $P=2/R$, no el mecanismo físico.",
         [r"\dfrac{1}{s}-\dfrac{1}{s'}\Big|_{n_2=-n_1}\ \to\ \dfrac1s+\dfrac1{s'}=\dfrac2R=\dfrac1f,\qquad f=\dfrac R2"]),
        ("Formulación matricial y límites paraxiales",
         r"Para rayos paraxiales, un rayo se describe por su altura $y$ y su pendiente $\theta$, y cada elemento actúa linealmente: propagar una distancia $d$ y atravesar una lente delgada son matrices de determinante 1 (se conserva el producto de índice por la invariante de Lagrange). La corrección de tercer orden de $\sin\theta\approx\theta-\theta^3/6$ da la aberración esférica longitudinal, proporcional al cuadrado de la altura del rayo, y el modelo paraxial deja de valer cuando $h/f$ deja de ser pequeño.",
         [r"\begin{pmatrix}y'\\\theta'\end{pmatrix}=\begin{pmatrix}1&0\\-1/f&1\end{pmatrix}\begin{pmatrix}y\\\theta\end{pmatrix},\qquad \det=1",
          r"\Delta s'_{aberr}\propto\dfrac{h^2}{f}\quad(\text{aberración esférica})"]),
        ("Resultado general",
         r"Las fórmulas finales son la focal por la ecuación del fabricante, la posición de la imagen y el aumento, para lente o espejo. Dimensionalmente, todas las inversas de longitud son dioptrías y el aumento es adimensional. El resultado vale para lentes delgadas ($t\ll f$) en la aproximación paraxial y en medios con aire a ambos lados; si los medios exteriores son $n_1$ y $n_3$ se sustituye $n-1$ por $(n-n_1)/R_1$ y $(n_3-n)/R_2$ en cada superficie.",
         [r"\boxed{\,\dfrac1f=(n-1)\left(\dfrac1{R_1}-\dfrac1{R_2}\right),\qquad s'=\dfrac{sf}{s-f},\qquad m=-\dfrac{s'}{s},\qquad f_{espejo}=\dfrac R2\,}"]),
        ("Particularización al problema",
         rf"Para la lente biconvexa con $n=1.5$, $R_1=+10$ cm y $R_2=-10$ cm, la focal es {f*100:.3f} cm y, con el objeto a 30 cm, la imagen queda en {sp*100:.4f} cm, real e invertida con $m={-sp/s:.4f}$. Para el espejo cóncavo de $R=40$ cm ($f=20$ cm) con el objeto a 60 cm la imagen está en {spm*100:.4f} cm con $m={-spm/sm:.4f}$. Son las mismas cifras de la solución analítica de la escena.",
         [rf"\dfrac1f=(0.5)\left(\dfrac1{{{r1}}}-\dfrac1{{{r2}}}\right)={pw:.2f}\ \mathrm{{D}}\ \Rightarrow\ f={f*100:.3f}\ \mathrm{{cm}}",
          rf"s'=\dfrac{{sf}}{{s-f}}=\dfrac{{({s})({f:.3f})}}{{{s}-{f:.3f}}}={sp*100:.4f}\ \mathrm{{cm}},\qquad m=-\dfrac{{{sp*100:.3f}}}{{{s*100:.1f}}}={-sp/s:.4f}",
          rf"s'_{{espejo}}=\dfrac{{(0.6)(0.2)}}{{0.6-0.2}}={spm*100:.4f}\ \mathrm{{cm}},\qquad m_{{espejo}}={-spm/sm:.4f}"]),
        ("Errores comunes y límites de validez",
         r"Se confunden con frecuencia los signos: usar $R_2=+10$ cm en una lente biconvexa da una focal infinita o negativa. Otra confusión es suponer que una lente siempre invierte la imagen; solo lo hace con imagen real. En un espejo, $s'>0$ está del lado del objeto, a diferencia de la lente. Más allá del modelo, las lentes gruesas necesitan planos principales; los rayos marginales sufren aberración esférica, coma y astigmatismo, y la dependencia de $n$ con $\lambda$ produce aberración cromática, que se corrige con dobletes. Para aberturas pequeñas, la difracción impone el límite de resolución $1.22\lambda f/D$.",
         []),
    ])


# ===================================================================== TwoLensSystem
def _twolens() -> dict:
    f1, f2, dd = m16.F1, m16.F2, m16.SEP
    s = 0.5
    s1p = m16.image_distance_analytic(s, f1)
    s2p, ma = m16.two_lens_analytic()
    zn, mn = m16.two_lens_matrix()
    M = m16.thin_lens_matrix(f2) @ m16.free_matrix(dd) @ m16.thin_lens_matrix(f1)
    a, b, c, dm = M.ravel()
    spm = -(a * s + b) / (c * s + dm)
    mag = 1 / (c * s + dm)
    feff = -1 / c
    return dict(concept=[
        ("Qué se estudia y por qué importa",
         r"Casi ningún instrumento tiene una sola lente: el telescopio, el microscopio, el objetivo de una cámara y el ojo con gafas son sistemas de varias lentes. Para analizarlos sin repetir la ecuación de las lentes etapa por etapa, el formalismo de matrices de rayos (matrices ABCD) reduce todo el sistema a una matriz $2\times2$ y a unas pocas cantidades: distancia focal efectiva, posición de los planos principales y aumento. Es el método que usan los diseñadores y la base de la teoría de cavidades láser y haces gaussianos.",
         []),
        ("Magnitudes y convenciones",
         r"Un rayo paraxial queda descrito en cada plano transversal por el vector $(y,\theta)$, con $y$ la altura sobre el eje (m) y $\theta$ la pendiente (rad). Las matrices son adimensionales salvo el elemento $B$ (metros) y $C$ (m$^{-1}$, dioptrías). Se multiplica de derecha a izquierda, en orden inverso al de recorrido. Para esta escena, $f_1$ y $f_2$ son las distancias focales, $d$ la separación entre lentes y $s$ la distancia del objeto a la primera lente.",
         [r"\begin{pmatrix}y_{sal}\\\theta_{sal}\end{pmatrix}=\begin{pmatrix}A&B\\C&D\end{pmatrix}\begin{pmatrix}y_{ent}\\\theta_{ent}\end{pmatrix}"]),
        ("Intuición y analogías",
         r"Cada elemento óptico transforma rayos de forma lineal, como una caja negra con un «engranaje» de dos ruedas: el desplazamiento es una cizalladura (la altura crece con la pendiente) y la lente es la otra cizalladura (la pendiente cambia con la altura). Componer cizalladuras da cualquier transformación lineal de determinante 1. La conservación del determinante es la conservación del espacio de fases de los rayos, análoga al teorema de Liouville: la óptica paraxial es una mecánica hamiltoniana lineal con $y$ como coordenada y $n\theta$ como momento.",
         []),
        ("Matrices de propagación y de lente delgada",
         r"En un medio homogéneo, un rayo es una recta: $y'=y+d\,\theta$ y la pendiente no cambia. En una lente delgada, la altura no cambia (espesor nulo) y la pendiente cambia $\Delta\theta=-y/f$, que es la expresión paraxial de la ley de las lentes: un rayo a altura $y$ se desvía hacia el eje una cantidad proporcional a $y$. Ambas operaciones son lineales y tienen determinante 1. El mismo rayo cruzando varios elementos produce el producto ordenado de las matrices.",
         [r"\mathbf{F}(d)=\begin{pmatrix}1&d\\0&1\end{pmatrix},\qquad \mathbf{L}(f)=\begin{pmatrix}1&0\\-1/f&1\end{pmatrix}",
          r"\det\mathbf F=\det\mathbf L=1\ \Rightarrow\ \det\mathbf M=1"]),
        ("Matriz del sistema de dos lentes",
         r"Se componen primero la lente 1, luego la propagación $d$ y por último la lente 2, con el producto en orden inverso. El resultado, simbólico, muestra que el elemento $C$ es el negativo de la potencia del sistema, y que la potencia no es la suma de las potencias, sino que incluye un término de separación. Dos casos notables: lentes en contacto ($d=0$) suman potencias; si $d=f_1+f_2$ resulta $C=0$ (sistema afocal, un telescopio) y $A=-f_2/f_1$ es el aumento angular inverso.",
         [r"\mathbf M=\mathbf L(f_2)\,\mathbf F(d)\,\mathbf L(f_1)=\begin{pmatrix}1-\dfrac{d}{f_1}&d\\[2mm]-\dfrac1{f_1}-\dfrac1{f_2}+\dfrac{d}{f_1f_2}&1-\dfrac{d}{f_2}\end{pmatrix}",
          r"\dfrac1{f_{ef}}=-C=\dfrac1{f_1}+\dfrac1{f_2}-\dfrac{d}{f_1f_2}"]),
        ("Condición de imagen y aumento",
         r"Un objeto a distancia $s$ antes del sistema y una imagen a $s'$ después se conectan con la matriz total $\mathbf F(s')\,\mathbf M\,\mathbf F(s)$. Hay imagen cuando la altura final no depende de la pendiente inicial de los rayos del punto objeto, es decir cuando el elemento $B$ de la matriz total se anula: todos los rayos de un punto se cruzan en un punto. El elemento $A$ en ese plano es el aumento lateral. Como el determinante vale 1, $A=1/D_{tot}$.",
         [r"\mathbf T=\mathbf F(s')\,\mathbf M\,\mathbf F(s)=\begin{pmatrix}a+s'c&\ a s+b+s'(cs+d_{22})\\ c&cs+d_{22}\end{pmatrix}",
          r"B_{tot}=0\ \Rightarrow\ s'=-\dfrac{as+b}{cs+d_{22}}",
          r"m=A_{tot}=a+s'c=\dfrac{1}{cs+d_{22}}"]),
        ("Equivalencia con la ecuación de lentes en cadena",
         r"El resultado matricial coincide con aplicar la ecuación $1/s+1/s'=1/f$ lente por lente, con una precaución: si la imagen intermedia cae después de la segunda lente, el objeto de esta es virtual y su distancia es negativa, $s_2=d-s_1'<0$. En el formalismo matricial esta distinción no existe, ya que el signo sale del álgebra. Para $N$ elementos se multiplican $N$ matrices; el aumento total es el producto de los aumentos de cada etapa, lo que concuerda con $A_{tot}$.",
         [r"s_1'=\dfrac{sf_1}{s-f_1},\qquad s_2=d-s_1',\qquad s_2'=\dfrac{s_2f_2}{s_2-f_2},\qquad m=\left(-\dfrac{s_1'}{s}\right)\left(-\dfrac{s_2'}{s_2}\right)"]),
        ("Resultado general",
         r"Para dos lentes delgadas, la posición de la imagen medida desde la segunda lente y el aumento total se obtienen en forma cerrada a partir de $f_1,f_2,d,s$. Se verifican dimensionalmente porque $B$ es una longitud y $C$ una inversa de longitud, y se comprueban en los límites: $d\to0$ da una lente de potencia suma; $f_2\to\infty$ (sin segunda lente) devuelve la lente simple. La estabilidad de una cavidad, el diseño de un expansor de haz y el cálculo de un telescopio se hacen con estas mismas expresiones.",
         [r"\boxed{\,s'=-\dfrac{as+b}{cs+d_{22}},\qquad m=\dfrac{1}{cs+d_{22}},\qquad \dfrac1{f_{ef}}=\dfrac1{f_1}+\dfrac1{f_2}-\dfrac{d}{f_1f_2}\,}"]),
        ("Particularización al problema",
         rf"Con $f_1=20$ cm, $f_2=10$ cm, $d=15$ cm y $s=50$ cm, la matriz del sistema, la focal efectiva, la posición de la imagen y el aumento coinciden con el método paso a paso de la escena. La imagen de la primera lente cae a {s1p*100:.3f} cm de ella, es decir {s1p*100-dd*100:.2f} cm después de la segunda lente: el objeto de esta es virtual, y la imagen final queda a {s2p*100:.4f} cm de $L_2$ con aumento $m={ma:.5f}$ (imagen invertida y reducida). La comparación con el valor matricial confirma la coincidencia.",
         [rf"\mathbf M=\begin{{pmatrix}}{a:.5g}&{b:.5g}\\{c:.5g}&{dm:.5g}\end{{pmatrix}},\qquad \det\mathbf M={a*dm-b*c:.5f},\qquad f_{{ef}}=-\dfrac1C={feff*100:.4f}\ \mathrm{{cm}}",
          rf"s'=-\dfrac{{({a:.5g})({s})+({b:.5g})}}{{({c:.5g})({s})+({dm:.5g})}}={spm*100:.4f}\ \mathrm{{cm}},\qquad m=\dfrac{{1}}{{({c:.5g})({s})+({dm:.5g})}}={mag:.5f}",
          rf"\text{{Cadena de lentes: }}\ s_2'={s2p*100:.4f}\ \mathrm{{cm}},\ \ m={ma:.5f};\qquad \text{{matricial: }}\ {zn*100:.4f}\ \mathrm{{cm}},\ {mn:.5f}"]),
        ("Errores comunes y límites de validez",
         r"El error más común es multiplicar las matrices en el orden del recorrido en lugar del orden inverso, o medir $s'$ desde la primera lente en vez de la última. Otro es olvidar el signo negativo de un objeto virtual. Las matrices ABCD son válidas solo en el régimen paraxial, con lentes delgadas alineadas y sin aberraciones; para lentes gruesas se usan matrices de superficie con índices y espesor, para aberraciones se traza el rayo exacto (óptica de tercer orden y superiores) y para haces coherentes se emplea la ley ABCD sobre el parámetro complejo $q$ del haz gaussiano.",
         []),
    ])


# ===================================================================== YoungSlits
def _young() -> dict:
    lam, d, L = m17.LAM, m17.D_SLIT, m17.L_SCR
    sp = lam * L / d
    ya = {m: m17.fringe_position_analytic(m) for m in (1, 3, 5)}
    th = {m: np.degrees(np.arcsin(m * lam / d)) for m in (1, 3, 5)}
    nf = d**2 / (lam * L)
    mmax = int(np.floor(d / lam))
    y1_par = lam * L / d
    err3 = (3 * y1_par - ya[3]) / ya[3]
    dr = lambda y: np.hypot(L, y + d / 2) - np.hypot(L, y - d / 2)  # noqa: E731
    return dict(concept=[
        ("Qué se estudia y por qué importa",
         r"El experimento de Young (1801) mostró que la luz produce franjas de interferencia, algo que un chorro de partículas no haría, y fue la prueba decisiva de su naturaleza ondulatoria. Hoy el mismo montaje, con electrones, átomos o moléculas, demuestra la dualidad onda-partícula, y la interferencia de dos haces es la base de los interferómetros que miden distancias con precisión de una fracción de longitud de onda (LIGO, metrología, espectroscopia por transformada de Fourier).",
         []),
        ("Magnitudes y convenciones",
         r"Intervienen la longitud de onda $\lambda$ (m), la separación $d$ entre rendijas (m), la distancia $L$ a la pantalla (m) y la altura $y$ sobre el eje central. La diferencia de camino $\Delta r=r_2-r_1$ (m) y la diferencia de fase $\Delta\varphi=k\Delta r$ (rad, con $k=2\pi/\lambda$) controlan el resultado. El orden $m$ es un entero (positivo y negativo); $m=0$ es la franja central. En el visible $\lambda\sim5\times10^{-7}$ m, de modo que $d$ debe ser de décimas de milímetro y $L$ de metros para que las franjas se vean a simple vista.",
         [r"k=\dfrac{2\pi}{\lambda},\qquad \Delta\varphi=k\,\Delta r,\qquad \Delta y=\dfrac{\lambda L}{d}"]),
        ("Intuición y analogías",
         r"Dos piedras lanzadas al agua generan círculos que se cruzan; donde coinciden dos crestas el agua sube el doble, y donde una cresta encuentra un valle el agua queda quieta. Con luz ocurre lo mismo, pero lo que se suma son campos y lo que se ve es la intensidad, el cuadrado de la suma. Por eso la energía no se destruye en las franjas oscuras: se redistribuye hacia las brillantes, que reciben hasta cuatro veces la intensidad de una rendija sola mientras que el promedio es solo el doble.",
         []),
        ("Superposición de dos fuentes coherentes",
         r"La ecuación de onda es lineal y por ello vale el principio de superposición. Cada rendija, estrecha frente a $\lambda$, actúa como una fuente cilíndrica; en el plano de observación, el campo complejo es la suma de dos ondas con amplitud $E_0$ (se desprecia la variación de $1/\sqrt r$ entre ambas por ser $d\ll L$) y fases $kr_1$ y $kr_2$. La intensidad es el módulo cuadrado, que se simplifica con la identidad de la suma de exponenciales a $4I_0\cos^2(\Delta\varphi/2)$. La condición de coherencia, que la diferencia de fase de las fuentes sea constante, es esencial; las fuentes independientes no interfieren de forma estable.",
         [r"E=E_0\left(e^{\mathrm ikr_1}+e^{\mathrm ikr_2}\right)=2E_0\,e^{\mathrm ik(r_1+r_2)/2}\cos\dfrac{k(r_2-r_1)}{2}",
          r"I=|E|^2=4I_0\cos^2\!\left(\dfrac{\pi\,\Delta r}{\lambda}\right),\qquad I_0=|E_0|^2",
          r"\Delta r=\lambda m\ \Rightarrow\ I_{max}=4I_0,\qquad \Delta r=\left(m+\tfrac12\right)\lambda\ \Rightarrow\ I=0"]),
        ("Geometría exacta de la diferencia de camino",
         r"Con las rendijas en $(\pm d/2,0)$ y un punto de la pantalla en $(y,L)$ las distancias se calculan por Pitágoras. La diferencia de camino exacta es una función de $y$; sus franjas brillantes son las ramas de hipérbolas con focos en las rendijas, que la pantalla corta. Para $L\gg d$ se desarrolla en serie y aparece $d\sin\theta$ con $\tan\theta=y/L$: este es el límite de Fraunhofer (rayos casi paralelos), que corresponde a un desfase cuadrático despreciable.",
         [r"r_{1,2}=\sqrt{L^2+\left(y\mp\tfrac d2\right)^2},\qquad \Delta r=r_2-r_1",
          r"\Delta r=d\sin\theta\left[1+O\!\left(\dfrac{d^2}{L^2}\right)\right],\qquad \tan\theta=\dfrac yL",
          r"\Delta r\approx\dfrac{d\,y}{L}\quad(y\ll L)"]),
        ("Posiciones de las franjas y espaciado",
         r"Imponer $\Delta r=m\lambda$ da los ángulos de los máximos y, con la geometría $y=L\tan\theta$, sus alturas exactas. Para ángulos pequeños, $\tan\theta\approx\sin\theta\approx\theta$ y las franjas quedan igualmente espaciadas con $\Delta y=\lambda L/d$. Existe un orden máximo, porque $\sin\theta\le1$ obliga a $|m|\le d/\lambda$. La intensidad se escribe también en función de la altura. La constancia del espaciado permite medir $\lambda$ de manera directa: $\lambda=d\,\Delta y/L$.",
         [r"\sin\theta_m=\dfrac{m\lambda}{d},\qquad y_m=L\tan\!\left[\arcsin\dfrac{m\lambda}{d}\right]",
          r"y_m\approx\dfrac{m\lambda L}{d},\qquad \Delta y=y_{m+1}-y_m=\dfrac{\lambda L}{d}",
          r"|m|\le\dfrac d\lambda,\qquad I(y)=4I_0\cos^2\!\left(\dfrac{\pi d\,y}{\lambda L}\right)"]),
        ("Rendijas de ancho finito y transformada de Fourier",
         r"Si cada rendija tiene ancho $a$, el campo en el infinito es la transformada de Fourier espacial de la función de apertura $T(x')$, con frecuencia espacial $k\sin\theta$. La apertura de dos rendijas es una rectángulo convolucionado con dos deltas; por el teorema de convolución, su transformada es el producto de la transformada del rectángulo por la de las deltas. De ahí el patrón: las franjas del coseno multiplicadas por la envolvente de difracción $\mathrm{sinc}^2$ de una rendija. Cuanto más angosta la rendija, más ancha la envolvente, y para $a\ll\lambda$ vuelve la expresión ideal de rendijas puntuales.",
         [r"T(x')=\mathrm{rect}\!\left(\dfrac{x'-d/2}{a}\right)+\mathrm{rect}\!\left(\dfrac{x'+d/2}{a}\right),\qquad \tilde T(\kappa)=\int T(x')\,e^{-\mathrm i\kappa x'}\mathrm dx'",
          r"\tilde T(\kappa)=2a\,\mathrm{sinc}\!\left(\dfrac{\kappa a}{2}\right)\cos\dfrac{\kappa d}{2},\qquad \kappa=k\sin\theta",
          r"I(\theta)=4I_0\,\mathrm{sinc}^2\beta\,\cos^2\gamma,\qquad \beta=\dfrac{\pi a\sin\theta}{\lambda},\ \gamma=\dfrac{\pi d\sin\theta}{\lambda}"]),
        ("Resultado general",
         r"Las posiciones exactas y aproximadas de las franjas y su espaciado definen el resultado, junto con el criterio de validez del régimen de Fraunhofer. La expresión es dimensionalmente consistente: $\lambda L/d$ es una longitud y $m$ es adimensional. Los casos límite son $d\to\infty$ (franjas tan finas que se promedian en un fondo uniforme) y $d\to\lambda$ (sólo el máximo central y dos órdenes, $m=\pm1$ rasantes). El criterio de campo lejano es que el número de Fresnel $N_F=d^2/(\lambda L)$ sea mucho menor que 1.",
         [r"\boxed{\,y_m=L\tan\!\left[\arcsin\dfrac{m\lambda}d\right]\approx\dfrac{m\lambda L}{d},\qquad \Delta y=\dfrac{\lambda L}{d},\qquad N_F=\dfrac{d^2}{\lambda L}\ll1\,}"]),
        ("Particularización al problema",
         rf"Con $\lambda=600$ nm, $d=0.25$ mm y $L=1.5$ m, el espaciado es {sp*1e3:.4f} mm, el orden máximo es $m_{{max}}={mmax}$ y el número de Fresnel es {nf:.4f}, así que el régimen es de campo lejano. Las posiciones exactas de la solución analítica difieren de la aproximación paraxial en el tercer orden en un {abs(err3):.1e} relativo, imperceptible; y la diferencia de camino exacta evaluada en $y_1$ vale $\lambda$ con la precisión numérica.",
         [rf"\Delta y=\dfrac{{({sci(lam,3)})(1.5)}}{{{sci(d,3)}}}={sp*1e3:.4f}\ \mathrm{{mm}},\qquad N_F=\dfrac{{({sci(d,3)})^2}}{{({sci(lam,3)})(1.5)}}={nf:.4f}",
          rf"\theta_1={th[1]:.5f}^\circ,\ y_1={ya[1]*1e3:.5f}\ \mathrm{{mm}};\qquad \theta_3={th[3]:.5f}^\circ,\ y_3={ya[3]*1e3:.5f}\ \mathrm{{mm}};\qquad y_5={ya[5]*1e3:.5f}\ \mathrm{{mm}}",
          rf"r_2-r_1\big|_{{y_1}}={dr(ya[1])*1e9:.4f}\ \mathrm{{nm}}\ \ (\lambda={lam*1e9:.0f}\ \mathrm{{nm}}),\qquad \dfrac{{3\lambda L/d-y_3}}{{y_3}}={sci(err3,2)}"]),
        ("Errores comunes y límites de validez",
         r"Un error frecuente es creer que las franjas oscuras «destruyen» energía; la energía total promedio sigue siendo la suma de las dos rendijas. También se confunde $\Delta y=\lambda L/d$ con la posición de la primera franja contada desde la central, o se usa en ángulos grandes, donde ya no vale y hay que emplear $y_m=L\tan\theta_m$. Si la fuente tiene ancho o es poco monocromática, la visibilidad de las franjas cae (coherencia espacial y temporal) y con luz blanca solo se ve nítida la franja central. Si la pantalla está cerca ($N_F\gtrsim1$) hay que usar difracción de Fresnel. El modelo escalar ignora la polarización, importante si $d\sim\lambda$.",
         []),
    ])


# ===================================================================== ThinFilmCoating
def _film() -> dict:
    n0, nf, ns, t, lam0 = 1.0, m17.N_FILM, m17.N_SUB, m17.T_FILM, m17.LAM0
    r0 = m17.bare_reflectance()
    r01 = (n0 - nf) / (n0 + nf)
    r12 = (nf - ns) / (nf + ns)
    qw = (n0 * ns - nf**2) / (n0 * ns + nf**2)
    lams = (450e-9, 550e-9, 650e-9)
    R = {l: float(m17.reflectance_analytic(l)) for l in lams}
    Rm = {l: float(m17.reflectance_matrix(l)[0]) for l in lams}
    nopt = np.sqrt(n0 * ns)
    l = 450e-9
    dl = 2 * np.pi * nf * t / l
    airy = (r01**2 + r12**2 + 2 * r01 * r12 * np.cos(2 * dl)) / (1 + r01**2 * r12**2 + 2 * r01 * r12 * np.cos(2 * dl))
    return dict(concept=[
        ("Qué se estudia y por qué importa",
         r"Una superficie de vidrio sin tratar refleja alrededor del 4 % de la luz que le llega; en un objetivo con quince lentes esa pérdida se acumula y el contraste se destruye con reflejos fantasma. Los recubrimientos antirreflejo, capas de espesor de una fracción de longitud de onda, reducen la reflectancia a cerca del 1 % o menos. El mismo cálculo explica los colores de una película de jabón o de aceite, los espejos dieléctricos y los filtros de interferencia.",
         []),
        ("Magnitudes y convenciones",
         r"Los índices son $n_0$ (medio de incidencia), $n_f$ (película) y $n_s$ (sustrato). La película tiene espesor $t$ (m) y el desfase en una pasada es $\delta=2\pi n_ft\cos\theta_f/\lambda$ (rad); el camino óptico es $n_ft$. Los coeficientes de Fresnel $r_{ij}$ son amplitudes adimensionales (negativas si hay cambio de fase de $\pi$) y la reflectancia $R=|r|^2$ es una fracción de la potencia, normalmente expresada en %. Un recubrimiento de cuarto de onda tiene $n_ft=\lambda_0/4$.",
         [r"\delta=\dfrac{2\pi n_ft}{\lambda},\qquad r_{ij}=\dfrac{n_i-n_j}{n_i+n_j},\qquad R=|r|^2"]),
        ("Intuición y analogías",
         r"La luz se refleja en la cara superior y en la inferior de la película; las dos ondas reflejadas vuelven juntas y se suman. Si la segunda se retrasa media longitud de onda con respecto a la primera, se cancelan, como dos pulsos en una cuerda que llegan con signos opuestos. Hay un cuarto de onda de ida y otro de vuelta: de ahí el espesor $\lambda/4$ dentro de la película. La cancelación sería total si las dos reflexiones tuvieran la misma amplitud, y esa es la condición sobre los índices.",
         []),
        ("Condiciones de contorno y matriz característica de una capa",
         r"Las ecuaciones de Maxwell exigen que las componentes tangenciales de $E$ y $H$ sean continuas en cada interfaz. Dentro de una capa homogénea los campos son combinaciones de ondas progresiva y regresiva con fase $e^{\pm\mathrm i\delta}$, y la continuidad liga los campos en la cara superior con los de la inferior mediante una matriz $2\times2$ de determinante 1 (la admitancia óptica del medio, a incidencia normal, es proporcional a $n$). Esta matriz característica es la herramienta general: para una pila de capas, se multiplican las matrices individuales.",
         [r"\begin{pmatrix}E_a\\H_a\end{pmatrix}=\begin{pmatrix}\cos\delta&\dfrac{\mathrm i}{n_f}\sin\delta\\[1mm]\mathrm in_f\sin\delta&\cos\delta\end{pmatrix}\begin{pmatrix}E_b\\H_b\end{pmatrix}",
          r"\det\mathbf M=\cos^2\delta+\sin^2\delta=1",
          r"\mathbf M_{pila}=\mathbf M_1\mathbf M_2\cdots\mathbf M_N"]),
        ("Coeficiente de reflexión de la pila",
         r"Con el sustrato semiinfinito sin onda regresiva, los campos en la cara inferior son proporcionales a $(1,n_s)$; se aplica la matriz y se calcula el cociente que define la admitancia de entrada $Y=C/B$, de donde el coeficiente de reflexión resulta de comparar $Y$ con la admitancia $n_0$ del medio incidente. Aplica a incidencia normal; la reflexión de potencia es el módulo cuadrado.",
         [r"\begin{pmatrix}B\\C\end{pmatrix}=\mathbf M\begin{pmatrix}1\\n_s\end{pmatrix},\qquad Y=\dfrac CB",
          r"r=\dfrac{n_0B-C}{n_0B+C},\qquad R=|r|^2,\qquad T=\dfrac{4n_0\,\mathrm{Re}(n_s)}{|n_0B+C|^2}"]),
        ("Fórmula de Airy: reflexiones múltiples",
         r"El mismo resultado se obtiene sumando todas las reflexiones internas: la onda se refleja con amplitud $r_{01}$ en la cara superior, y la que entra se refleja en la inferior y sale con un retardo $e^{2\mathrm i\delta}$ y factores de transmisión, y así sucesivamente. La suma geométrica converge porque $|r_{01}r_{12}|<1$. Al calcular el módulo cuadrado, resulta una reflectancia que depende de $\cos2\delta$, es decir, periódica en $\lambda^{-1}$.",
         [r"r=r_{01}+\dfrac{t_{01}t_{10}\,r_{12}\,e^{2\mathrm i\delta}}{1-r_{10}r_{12}e^{2\mathrm i\delta}}=\dfrac{r_{01}+r_{12}e^{2\mathrm i\delta}}{1+r_{01}r_{12}e^{2\mathrm i\delta}}",
          r"R=\dfrac{r_{01}^2+r_{12}^2+2r_{01}r_{12}\cos2\delta}{1+r_{01}^2r_{12}^2+2r_{01}r_{12}\cos2\delta}"]),
        ("Capa de cuarto de onda y condición antirreflejo",
         r"Con $n_ft=\lambda_0/4$ resulta $\delta=\pi/2$ a la longitud de diseño; la matriz se vuelve antidiagonal y el cálculo del coeficiente de reflexión es inmediato. La reflexión se anula si $n_f^2=n_0n_s$, es decir, la película debe tener el índice medio geométrico entre aire y sustrato. Como no existen materiales robustos con $n=1.23$ para vidrio común, se usa MgF$_2$ ($1.38$), que deja una reflexión residual pequeña. Para ampliar el ancho de banda se apilan varias capas.",
         [r"\delta=\dfrac\pi2:\quad\mathbf M=\begin{pmatrix}0&\mathrm i/n_f\\\mathrm in_f&0\end{pmatrix},\qquad B=\dfrac{\mathrm in_s}{n_f},\ C=\mathrm in_f",
          r"r=\dfrac{n_0n_s-n_f^2}{n_0n_s+n_f^2},\qquad R=0\iff n_f=\sqrt{n_0n_s}"]),
        ("Resultado general",
         r"El resultado para la reflectancia depende de los tres índices y del desfase $\delta(\lambda)$, y sus dos casos relevantes son la reflectancia del sustrato desnudo y la de la capa de cuarto de onda. Como $\delta$ es adimensional, la reflectancia es una función periódica de $1/\lambda$ con mínimos donde $2\delta=(2m+1)\pi$; la verificación dimensional es trivial porque solo aparecen razones de índices. La energía se conserva en materiales sin absorción: $R+T=1$.",
         [r"\boxed{\,R_{vidrio}=\left(\dfrac{n_s-n_0}{n_s+n_0}\right)^2,\qquad R_{\lambda_0/4}=\left(\dfrac{n_0n_s-n_f^2}{n_0n_s+n_f^2}\right)^2,\qquad R(\lambda)\ \text{(Airy)}\,}"]),
        ("Particularización al problema",
         rf"Para aire ($n_0=1$), MgF$_2$ ($n_f=1.38$) y vidrio ($n_s={ns}$), con $t=\lambda_0/4n_f={t*1e9:.4f}$ nm y $\lambda_0=550$ nm, el vidrio desnudo refleja {r0*100:.4f} % y con película, a 550 nm, {R[550e-9]*100:.4f} %. El índice óptimo sería $\sqrt{{n_s}}={nopt:.4f}$, que explica por qué el residuo no es nulo. A 450 nm la fórmula de Airy da {airy*100:.4f} %, que coincide con la matriz de transferencia ({Rm[450e-9]*100:.4f} %) y con la solución analítica de la escena.",
         [rf"R_{{vidrio}}=\left(\dfrac{{{ns}-1}}{{{ns}+1}}\right)^2={r0*100:.4f}\,\%",
          rf"r_{{\lambda_0/4}}=\dfrac{{(1)({ns})-({nf})^2}}{{(1)({ns})+({nf})^2}}={qw:.5f},\qquad R(550\ \mathrm{{nm}})={qw**2*100:.4f}\,\%\ \ (\text{{escena: }}{R[550e-9]*100:.4f}\,\%)",
          rf"\delta(450)=\dfrac{{2\pi({nf})({t*1e9:.4f})}}{{450}}={dl:.5f}\ \mathrm{{rad}},\qquad R(450)={airy*100:.4f}\,\%,\qquad R(650)={R[650e-9]*100:.4f}\,\%"]),
        ("Errores comunes y límites de validez",
         r"Se piensa a veces que basta cualquier película de espesor $\lambda/4$: sin la condición de índices la cancelación es solo parcial. Otro error es medir $\lambda/4$ en el aire en vez de dentro de la película ($\lambda/4n_f$). También se olvida que el cambio de fase de $\pi$ ocurre al reflejarse en un medio de mayor índice, de modo que con $n_0<n_f<n_s$ ambas reflexiones lo sufren y la cancelación exige el cuarto de onda; si $n_f>n_s$ la condición cambia a media onda para reducir reflexión. El modelo supone incidencia normal, capas planas y homogéneas, sin absorción ni dispersión; a incidencia oblicua $\delta$ incluye $\cos\theta_f$ y los coeficientes de Fresnel dependen de la polarización, y una luz poco coherente (película gruesa, $t\gg$ longitud de coherencia) deja de producir interferencia.",
         []),
    ])


# ===================================================================== SlitAndGrating
def _grating() -> dict:
    lam, a, d, N = m18.LAM, m18.A_SLIT, m18.D_GRAT, m18.N_SLITS
    ms, ang = m18.principal_maxima_analytic()
    mmax = int(ms[-1])
    env = {k: float(np.sinc(k * a / d) ** 2) for k in (1, 2, 4)}
    thmin = np.degrees(np.arcsin(lam / (N * d)))
    dth1 = np.degrees(lam / (N * d * np.cos(np.deg2rad(ang[1]))))
    disp = {k: np.degrees(k / (d * np.cos(np.deg2rad(ang[k])))) * 1e-3 for k in (1, 2, 4)}  # grados por nm
    rp = 1 * N
    return dict(concept=[
        ("Qué se estudia y por qué importa",
         r"La difracción limita la nitidez de cualquier imagen y es la base de la espectroscopia: una rejilla separa los colores con mayor resolución que un prisma, y es el elemento central de los espectrómetros astronómicos, de los monocromadores y de los discos ópticos (el CD refleja colores porque es una rejilla). Esta escena deduce el patrón de una apertura general como transformada de Fourier y luego lo particulariza a una rendija y a una rejilla de $N$ rendijas.",
         []),
        ("Magnitudes y convenciones",
         r"El ancho de cada rendija es $a$ y el paso de la rejilla $d$ (m); $N$ es el número de rendijas, $\lambda$ la longitud de onda y $\theta$ el ángulo respecto de la normal. Se definen $\beta=\pi a\sin\theta/\lambda$ y $\gamma=\pi d\sin\theta/\lambda$ (rad). El orden $m$ de un máximo principal es un entero. La función $\mathrm{sinc}\,x=\sin x/x$ vale 1 en el origen y se anula en $x=n\pi$. Las intensidades se expresan relativas a la del máximo central de una rendija.",
         [r"\beta=\dfrac{\pi a\sin\theta}{\lambda},\qquad \gamma=\dfrac{\pi d\sin\theta}{\lambda},\qquad \mathrm{sinc}\,x=\dfrac{\sin x}{x}"]),
        ("Intuición y analogías",
         r"Una rendija se puede pensar como una fila de muchísimas fuentes puntuales vecinas; a ciertos ángulos, las contribuciones del extremo superior y del inferior llegan con fases opuestas y se cancelan: son los ceros del patrón. Cuando se tienen varias rendijas idénticas, cada una repite el mismo patrón, pero las contribuciones de rendijas distintas se suman con fases que dependen del paso; es una «orquesta» donde cada instrumento (la rendija) tiene su timbre (la envolvente) y el director (el paso $d$) decide cuándo todos coinciden. A más rendijas, más estrechos los picos.",
         []),
        ("Difracción de Fraunhofer como transformada de Fourier",
         r"Según el principio de Huygens-Fresnel, el campo en un punto lejano es la suma de ondas esféricas emitidas por cada punto de la apertura. En el campo lejano, la distancia desde un punto $x'$ de la apertura al observador es $r\approx r_0-x'\sin\theta$ (el término cuadrático es despreciable si $a^2/(\lambda r_0)\ll1$). Sacando los factores comunes, el campo es proporcional a la transformada de Fourier de la función de apertura $T(x')$ evaluada en la frecuencia espacial $\kappa=k\sin\theta$. Esta identidad explica el comportamiento inverso: una apertura estrecha da un patrón ancho (relación de incertidumbre $\Delta x\,\Delta\kappa\gtrsim1$).",
         [r"E(\theta)\propto\int_{-\infty}^{\infty}T(x')\,e^{-\mathrm i\kappa x'}\,\mathrm dx'=\tilde T(\kappa),\qquad \kappa=k\sin\theta",
          r"I(\theta)\propto|\tilde T(\kappa)|^2,\qquad \int|\tilde T|^2\,\mathrm d\kappa=2\pi\int|T|^2\,\mathrm dx'\quad(\text{Parseval})"]),
        ("Una rendija",
         r"La apertura es un rectángulo de ancho $a$, $T=1$ si $|x'|<a/2$. La integral se hace de manera elemental y resulta una función seno sobre argumento, es decir, un sinc con el argumento $\kappa a/2=\beta$. La intensidad normalizada es $\mathrm{sinc}^2\beta$, con ceros en $a\sin\theta=n\lambda$ ($n=\pm1,\pm2,\ldots$) y un lóbulo central de semianchura angular $\lambda/a$ que contiene el 90 % de la potencia. Si $a<\lambda$ no hay ceros y la luz se difunde en todo el semiespacio.",
         [r"\tilde T(\kappa)=\int_{-a/2}^{a/2}e^{-\mathrm i\kappa x'}\mathrm dx'=\dfrac{2\sin(\kappa a/2)}{\kappa}=a\,\mathrm{sinc}\,\beta",
          r"I(\theta)=I_0\,\mathrm{sinc}^2\beta,\qquad \text{ceros: }a\sin\theta=n\lambda"]),
        ("N rendijas: teorema de convolución",
         r"La rejilla es una rendija convolucionada con un peine de $N$ deltas separadas por $d$. El teorema de convolución dice que la transformada de una convolución es el producto de las transformadas, de modo que el campo es el factor de una rendija por la suma geométrica de $N$ fases $e^{-\mathrm ij\kappa d}$. Esta suma se calcula en forma cerrada (serie geométrica) y su módulo es el núcleo de Dirichlet $\sin N\gamma/\sin\gamma$. El patrón es entonces el producto de la envolvente de difracción por el factor de interferencia.",
         [r"T(x')=\mathrm{rect}_a(x')*\sum_{j=0}^{N-1}\delta(x'-jd)\ \Rightarrow\ \tilde T=a\,\mathrm{sinc}\,\beta\ \sum_{j=0}^{N-1}e^{-\mathrm ij\kappa d}",
          r"\left|\sum_{j=0}^{N-1}e^{-\mathrm i2j\gamma}\right|=\left|\dfrac{\sin N\gamma}{\sin\gamma}\right|",
          r"I(\theta)=I_0\,\mathrm{sinc}^2\beta\left(\dfrac{\sin N\gamma}{N\sin\gamma}\right)^2"]),
        ("Máximos principales, órdenes suprimidos y estructura fina",
         r"El factor de interferencia alcanza su máximo cuando $\gamma=m\pi$, es decir $d\sin\theta=m\lambda$, con valor $N^2$ (todas las rendijas en fase). Entre dos máximos principales el numerador se anula en $\gamma=j\pi/N$ con $j$ no múltiplo de $N$, lo que da $N-1$ mínimos y $N-2$ máximos secundarios. Si un máximo principal coincide con un cero de la envolvente, $m/n=d/a$, ese orden desaparece. El primer cero a ambos lados del máximo de orden $m$ está a $\Delta\sin\theta=\lambda/(Nd)$, de modo que el ancho angular decrece con $N$.",
         [r"d\sin\theta_m=m\lambda,\qquad m=0,\pm1,\ldots,\ |m|\le\dfrac d\lambda,\qquad I_{pico}/I_{1}=N^2",
          r"\Delta\theta_{1/2}\simeq\dfrac{\lambda}{Nd\cos\theta_m},\qquad \text{orden }m\text{ ausente si }\dfrac md=\dfrac na",
          r"\mathcal D=\dfrac{\mathrm d\theta}{\mathrm d\lambda}=\dfrac{m}{d\cos\theta_m},\qquad \mathcal R=\dfrac{\lambda}{\Delta\lambda}=mN"]),
        ("Resultado general",
         r"El patrón de una rejilla de $N$ rendijas de ancho $a$ y paso $d$ es el producto de una envolvente y de un factor de interferencia. El resultado es adimensional y homogéneo: $\beta$ y $\gamma$ son ángulos de fase; la dispersión se mide en rad/m y el poder resolutor es adimensional. Los casos límite recuperan la rendija simple ($N=1$), las dos rendijas de Young ($N=2$, donde $\sin2\gamma/2\sin\gamma=\cos\gamma$) y la rendija infinitesimal ($a\to0$, envolvente constante).",
         [r"\boxed{\,I(\theta)=I_0\,\mathrm{sinc}^2\!\left(\dfrac{\pi a\sin\theta}{\lambda}\right)\left[\dfrac{\sin(N\pi d\sin\theta/\lambda)}{N\sin(\pi d\sin\theta/\lambda)}\right]^2\,}",
          r"\theta_m=\arcsin\dfrac{m\lambda}{d},\qquad N=1,2:\ \text{rendija simple, Young}"]),
        ("Particularización al problema",
         rf"Con $\lambda=550$ nm, $a=2.0\ \mu$m, $d=6.0\ \mu$m y $N=5$, los máximos principales están en $\theta_m=\arcsin(m\lambda/d)$ hasta $m_{{max}}={mmax}$; el orden 3 (y 6, 9) queda suprimido porque $d/a=3$. La envolvente reduce la intensidad relativa de los órdenes 1, 2 y 4 a {env[1]:.4f}, {env[2]:.4f} y {env[4]:.4f} del máximo central. El primer mínimo junto al orden central está a {thmin:.4f}° y la anchura de pico del orden 1 es de {dth1:.4f}° aproximadamente. Los ángulos coinciden con los de la solución analítica de la escena.",
         [rf"\theta_m=\arcsin\dfrac{{m({sci(lam,3)})}}{{{sci(d,3)}}}:\quad \theta_1={ang[1]:.4f}^\circ,\ \theta_2={ang[2]:.4f}^\circ,\ \theta_4={ang[4]:.4f}^\circ,\qquad m_{{max}}=\lfloor d/\lambda\rfloor={mmax}",
          rf"\dfrac{{d}}{{a}}={d/a:.1f}\ \Rightarrow\ \beta_3=3\pi\dfrac{{a}}{{d}}=\pi\ \Rightarrow\ \mathrm{{sinc}}^2\beta_3={sci(float(np.sinc(1.0)**2),2)}\ (m=3\text{{ ausente}})",
          rf"\dfrac{{I(\theta_m)}}{{I(0)}}=\mathrm{{sinc}}^2\!\left(\dfrac{{m\pi a}}{{d}}\right):\ \ m=1:\ {env[1]:.4f},\ \ m=2:\ {env[2]:.4f},\ \ m=4:\ {env[4]:.4f}",
          rf"\sin\theta_{{min}}=\dfrac{{\lambda}}{{Nd}}={lam/(N*d):.5f}\ \Rightarrow\ \theta_{{min}}={thmin:.4f}^\circ,\qquad \mathcal R_{{m=1}}=mN={rp}"]),
        ("Errores comunes y límites de validez",
         r"Se confunde con frecuencia el papel de $a$ y $d$: $d$ fija las posiciones de los máximos y $a$ la envolvente de intensidad. Se cree también que añadir rendijas hace los máximos más intensos en el mismo ángulo sin afinarlos: el pico crece como $N^2$ pero su ancho decrece como $1/N$, de modo que la energía integrada crece como $N$. El modelo de Fraunhofer exige campo lejano (número de Fresnel $a^2/\lambda L\ll1$; de lo contrario, patrón de Fresnel), luz monocromática y coherente, aberturas planas y escalares: ignora polarización y espesor de la rejilla. Para ángulos grandes hay que añadir el factor de oblicuidad $\cos\theta$, y en incidencia oblicua la condición es $d(\sin\theta_m-\sin\theta_i)=m\lambda$.",
         []),
    ])


# ===================================================================== PolarizersMalus
def _pol() -> dict:
    t30, t45, t60 = np.pi / 6, np.pi / 4, np.pi / 3
    e0 = np.array([1.0, 0.0])
    P = m18.polarizer_jones
    e1 = P(0.0) @ e0
    e2 = P(t30) @ e1
    e3 = P(np.pi / 2) @ e2
    ia = {k: m18.three_polarizers_analytic(v) for k, v in (("30", t30), ("45", t45))}
    ij = {k: m18.jones_intensity([0, v, np.pi / 2]) for k, v in (("30", t30), ("45", t45))}
    ma = float(m18.malus_analytic(t60))
    mj = m18.jones_intensity([0, t60])
    zeno = {n: np.cos(np.pi / 2 / (n + 1)) ** (2 * (n + 1)) for n in (1, 2, 5, 10)}

    def mat(m_):
        return rf"\begin{{pmatrix}}{m_[0,0]:.4f}&{m_[0,1]:.4f}\\{m_[1,0]:.4f}&{m_[1,1]:.4f}\end{{pmatrix}}"
    return dict(concept=[
        ("Qué se estudia y por qué importa",
         r"La polarización describe la dirección en que oscila el campo eléctrico de la luz. Controlarla es fundamental en pantallas de cristal líquido, en gafas de sol que eliminan reflejos, en la fotografía, en microscopios de luz polarizada que revelan tensiones en materiales y en comunicaciones ópticas. La ley de Malus y el cálculo de Jones permiten predecir cuánta luz pasa por una cadena de polarizadores, y la sorpresa de que un tercer polarizador intercalado aumenta la transmisión revela la naturaleza vectorial del campo.",
         []),
        ("Magnitudes y convenciones",
         r"El campo $\vec E$ (V/m) es transversal a la propagación $\hat z$; su estado de polarización lineal se describe por el ángulo $\psi$ con el eje $x$. El eje de transmisión de un polarizador forma el ángulo $\theta$ con $x$ (positivo antihorario visto contra la luz). La intensidad $I\propto|\vec E|^2$ se expresa en W/m$^2$; la transmisión $I/I_0$ es adimensional y está entre 0 y 1. Luz no polarizada (natural) que cruza un polarizador ideal pierde la mitad de su intensidad.",
         [r"\vec E=E_x\hat x+E_y\hat y,\qquad I\propto|E_x|^2+|E_y|^2,\qquad \dfrac{I}{I_0}\in[0,1]"]),
        ("Intuición y analogías",
         r"Un polarizador es una reja de barrotes que solo deja pasar la proyección del campo sobre su dirección. Si la polarización es perpendicular a esa reja, no pasa nada; si está inclinada, pasa solo su proyección, y a la salida la luz queda alineada con la reja: el polarizador «reinicia» la dirección. Eso es lo que hace útil al tercero intercalado: toma la polarización de 0° y la rota hacia 90° en dos saltos, cada uno con pérdida moderada, en lugar de uno solo con pérdida total.",
         []),
        ("Ley de Malus: proyección del campo",
         r"El campo incidente, linealmente polarizado según $\psi$, se descompone en las componentes paralela y perpendicular al eje de transmisión, que forma el ángulo $\theta-\psi$. El polarizador ideal anula la perpendicular y transmite la paralela sin cambio, con una amplitud $E_0\cos(\theta-\psi)$. Como la intensidad es proporcional al cuadrado de la amplitud, se obtiene la ley de Malus. El campo emergente queda polarizado a lo largo del eje, un hecho clave para encadenar polarizadores.",
         [r"E_\parallel=E_0\cos(\theta-\psi),\qquad E_\perp\to0",
          r"I=I_0\cos^2(\theta-\psi),\qquad \vec E_{sal}=E_\parallel\,\hat u(\theta)"]),
        ("Cálculo de Jones y operador de proyección",
         r"El estado de polarización se representa por un vector columna de dos componentes complejas y cada elemento óptico lineal por una matriz $2\times2$. El polarizador ideal es el proyector sobre su eje: $P(\theta)=\hat u\hat u^T$, con $\hat u=(\cos\theta,\sin\theta)$. Es idempotente ($P^2=P$, aplicar dos veces el mismo polarizador no cambia nada) y de rango 1 (autovalores 0 y 1), lo que expresa el filtrado. La intensidad es la norma cuadrática del vector de Jones, $\vec E^\dagger\vec E$.",
         [r"P(\theta)=\begin{pmatrix}\cos^2\theta&\cos\theta\sin\theta\\\cos\theta\sin\theta&\sin^2\theta\end{pmatrix}=\hat u\hat u^{T},\qquad \hat u=\begin{pmatrix}\cos\theta\\\sin\theta\end{pmatrix}",
          r"P^2=P,\qquad \det P=0,\qquad \mathrm{tr}P=1,\qquad I=\vec E^\dagger\vec E"]),
        ("Cadena de polarizadores",
         r"Para polarizadores en $\theta_1,\theta_2,\ldots,\theta_n$ el campo final es el producto ordenado de los proyectores. Como $\hat u_j^T\hat u_k=\cos(\theta_j-\theta_k)$, cada proyector tras otro deja el factor coseno del ángulo relativo y el campo final queda a lo largo del último eje. La luz de entrada polarizada a 0° da la fórmula con productos de cosenos de los ángulos relativos entre polarizadores consecutivos. Es un resultado exacto de la teoría de Jones.",
         [r"\vec E_{sal}=P(\theta_n)\cdots P(\theta_2)P(\theta_1)\vec E_{ent}",
          r"\vec E_{sal}=\left[\prod_{k=1}^{n-1}\cos(\theta_{k+1}-\theta_k)\right]\cos(\theta_1-\psi)\,E_0\,\hat u_n",
          r"\dfrac I{I_0}=\cos^2(\theta_1-\psi)\prod_{k=1}^{n-1}\cos^2(\theta_{k+1}-\theta_k)"]),
        ("Tres polarizadores y límite de muchos pasos",
         r"Para luz polarizada a 0°, un polarizador a $0^\circ$, otro a $\theta$ y otro a $\varphi$ dan $\cos^2\theta\cos^2(\varphi-\theta)$. Para $\varphi=90^\circ$ resulta $\cos^2\theta\sin^2\theta=\tfrac14\sin^22\theta$, máximo $1/4$ en $\theta=45^\circ$ (se obtiene derivando respecto de $\theta$, o por simetría). En general, el reparto uniforme del ángulo total en $n+1$ saltos iguales da una transmisión que tiende a 1 cuando $n\to\infty$, porque $\cos^{2(n+1)}[\varphi/(n+1)]\to1$: es la analogía óptica del efecto Zenón cuántico.",
         [r"\dfrac{I_3}{I_0}(\theta)=\cos^2\theta\,\cos^2(\varphi-\theta),\qquad \dfrac{\mathrm d}{\mathrm d\theta}=0\ \Rightarrow\ \theta=\dfrac\varphi2",
          r"\left.\dfrac{I_3}{I_0}\right|_{max}=\cos^4\dfrac\varphi2\ \xrightarrow{\ \varphi=90^\circ\ }\ \tfrac14",
          r"T_n=\cos^{2(n+1)}\!\left(\dfrac{\varphi}{n+1}\right)\xrightarrow[n\to\infty]{}1"]),
        ("Resultado general",
         r"Los resultados generales son la ley de Malus, la transmisión de tres polarizadores con el intermedio a un ángulo cualquiera (donde el máximo se alcanza en la mitad del ángulo total) y la fórmula de cadena. Todos son adimensionales y están acotados por 1, y respetan los casos límite: para $\theta=0$ o $\theta=\varphi$ la transmisión es $\cos^2\varphi$ (el intermedio no cambia nada); para $\varphi=90^\circ$ y $\theta=0$ se obtiene 0 (polarizadores cruzados).",
         [r"\boxed{\,\dfrac I{I_0}=\cos^2\Delta\psi,\qquad \dfrac{I_3}{I_0}=\cos^2\theta\cos^2(\varphi-\theta)\xrightarrow{\varphi=90^\circ}\tfrac14\sin^22\theta\,}"]),
        ("Particularización al problema",
         rf"Con luz polarizada a 0° y el primero y último polarizador a 0° y 90°, el vector de Jones tras el segundo polarizador a 30° y el tercero se calcula con las matrices. A 45° la transmisión es {ia['45']:.4f} (máximo); a 30°, {ia['30']:.4f}; la ley de Malus simple a 60° da {ma:.4f}. El cálculo con matrices de Jones ({ij['30']:.6f}, {ij['45']:.6f} y {mj:.6f}) reproduce los valores analíticos de la escena.",
         [rf"\vec E_1=({e1[0]:.4f},\,{e1[1]:.4f}),\quad \vec E_2=P(30^\circ)\vec E_1=({e2[0]:.4f},\,{e2[1]:.4f}),\quad \vec E_3=P(90^\circ)\vec E_2=({e3[0]:.4f},\,{e3[1]:.4f})",
          rf"P(30^\circ)={mat(P(t30))},\qquad \dfrac{{I_3}}{{I_0}}=|\vec E_3|^2={float(e3@e3):.4f}",
          rf"\dfrac{{I_3}}{{I_0}}\Big|_{{30^\circ}}={ia['30']:.4f},\quad \dfrac{{I_3}}{{I_0}}\Big|_{{45^\circ}}={ia['45']:.4f},\quad \cos^2 60^\circ={ma:.4f}",
          rf"\text{{Zenón (}}\varphi=90^\circ\text{{): }}\ n=1:\ {zeno[1]:.4f},\ n=2:\ {zeno[2]:.4f},\ n=5:\ {zeno[5]:.4f},\ n=10:\ {zeno[10]:.4f}"]),
        ("Errores comunes y límites de validez",
         r"El error típico es creer que tres polarizadores transmiten menos que dos: cada polarizador intermedio reorienta la polarización y puede hacer pasar luz donde antes no pasaba. También se aplica la ley de Malus a luz natural (donde se pierde la mitad independientemente del ángulo) o con el ángulo equivocado (medido entre ejes y no entre la polarización y el eje). El modelo supone polarizadores ideales (extinción infinita, sin pérdidas), luz monocromática y totalmente polarizada y propagación en la dirección del eje; los polarizadores reales tienen absorción residual y relación de extinción finita, y la luz parcialmente polarizada requiere los parámetros de Stokes y las matrices de Mueller. La reflexión en el ángulo de Brewster, la birrefringencia y la actividad óptica no entran en este modelo.",
         []),
    ])


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
