"""Concepto y desarrollo analitico general: capitulos 9 a 13 del volumen 2.

Escenas: BiotSavartLoop, SolenoidAxis, RotatingLoop, SlidingRod, RLCircuit,
RLCOscillator, DipoleTorque, ParamagnetCurie, SeriesRLC, PowerTransformer.
Cada escena: concept = [(titulo, prosa, [ecuaciones TeX]), ...] con la version general
(simbolos) y, al final, la particularizacion numerica calculada con `physics/`.
"""
from __future__ import annotations

import numpy as np

from content.texfmt import fx, sci
from physics import v2ch09_biot_savart as m9
from physics import v2ch10_faraday as m10
from physics import v2ch11_inductance as m11
from physics import v2ch12_magnetic_matter as m12
from physics import v2ch13_ac as m13
from physics.constants import K_B, MU0


# =============================================================================== BiotSavartLoop
def _loop() -> list:
    a, cur = m9.A_LOOP, m9.I
    b0 = float(m9.loop_axis_analytic(0.0))
    b5 = float(m9.loop_axis_analytic(0.05))
    wl, wd = m9.LEN_WIRE, m9.DIST
    bw, bi = m9.wire_field_analytic(), m9.infinite_wire()
    sn = (wl / 2) / np.hypot(wl / 2, wd)
    mom = cur * np.pi * a**2
    return [
        ("Qué se estudia y por qué importa",
         r"Toda corriente estacionaria crea un campo magnético a su alrededor, y la manera de calcularlo a partir de la geometría del conductor es la ley de Biot-Savart. La espira circular es el elemento básico de los electroimanes, las bobinas de Helmholtz, las sondas de resonancia magnética y los giróscopos; el hilo recto finito es el ladrillo con el que se construyen los campos de circuitos poligonales. Entender este cálculo en general permite obtener el campo de cualquier circuito filiforme como una suma (integral) de contribuciones elementales, y comprobar los resultados con un cálculo numérico independiente.",
         []),
        ("Magnitudes y convenciones",
         r"El campo $\vec B$ se mide en tesla ($1\ \mathrm{T}=1\ \mathrm{N/(A\cdot m)}$) y la constante $\mu_0=4\pi\times10^{-7}\ \mathrm{T\cdot m/A}$ fija la escala del vacío. El elemento de corriente $I\,d\vec\ell$ apunta en el sentido convencional de la corriente, $\vec r$ va de la fuente al punto de campo y el sentido de $\vec B$ sale de la regla de la mano derecha: los dedos siguen la corriente y el pulgar indica $\vec B$ en el eje. Con $I\sim1$ A y radios de centímetros, el campo está en el rango de decenas a cientos de microtesla, comparable con el campo terrestre ($\approx50\ \mu\mathrm{T}$).",
         [r"[B]=\mathrm{T},\qquad [\mu_0 I\,d\ell/r^2]=\dfrac{\mathrm{T\cdot m}}{\mathrm{A}}\cdot\dfrac{\mathrm{A\cdot m}}{\mathrm{m^2}}=\mathrm{T}"]),
        ("Intuición y analogías",
         r"La ley de Biot-Savart es la gemela magnética de la ley de Coulomb: cada elemento de corriente contribuye con un término que decae como $1/r^2$, pero la dirección no es radial sino perpendicular al plano que forman $d\vec\ell$ y $\vec r$. Por eso el campo envuelve al conductor en lazos cerrados y nunca tiene fuentes puntuales (no hay monopolos). En el eje de una espira, las componentes transversales de elementos opuestos se anulan por simetría y solo sobrevive la componente axial, como la resultante vertical de una corona de cuerdas que sostiene un peso.",
         []),
        ("Desarrollo 1: de las ecuaciones de la magnetostática a Biot-Savart",
         r"Para corrientes estacionarias, las ecuaciones de Maxwell se reducen a $\nabla\cdot\vec B=0$ y $\nabla\times\vec B=\mu_0\vec J$. La primera permite escribir $\vec B=\nabla\times\vec A$ con un potencial vectorial, y en la gauge de Coulomb ($\nabla\cdot\vec A=0$) la segunda se convierte en una ecuación de Poisson vectorial componente a componente, $\nabla^2\vec A=-\mu_0\vec J$. Su solución en el espacio sin fronteras es la superposición de las soluciones de fuente puntual, y al tomar el rotacional (el rotor actúa sobre $\vec r$, no sobre la fuente $\vec r'$) aparece la ley de Biot-Savart.",
         [r"\nabla\cdot\vec B=0,\qquad \nabla\times\vec B=\mu_0\vec J,\qquad \vec B=\nabla\times\vec A,\qquad \nabla^2\vec A=-\mu_0\vec J",
          r"\vec A(\vec r)=\dfrac{\mu_0}{4\pi}\int\dfrac{\vec J(\vec r')}{|\vec r-\vec r'|}\,dV'",
          r"\vec B=\nabla\times\vec A=\dfrac{\mu_0}{4\pi}\int\vec J(\vec r')\times\nabla\dfrac{1}{|\vec r-\vec r'|}\cdot(-1)\,dV'\ \Rightarrow\ \vec B(\vec r)=\dfrac{\mu_0}{4\pi}\int\dfrac{\vec J(\vec r')\times(\vec r-\vec r')}{|\vec r-\vec r'|^3}\,dV'",
          r"\vec J\,dV'\to I\,d\vec\ell'\quad\Rightarrow\quad d\vec B=\dfrac{\mu_0 I}{4\pi}\dfrac{d\vec\ell'\times\hat r}{r^2},\qquad \vec r=\vec r-\vec r'"]),
        ("Desarrollo 2: hilo recto finito de longitud arbitraria",
         r"Se coloca el hilo sobre el eje $z$ entre $z_1$ y $z_2$ y el punto de campo a una distancia perpendicular $d$. Por simetría axial, $\vec B$ es azimutal y basta integrar su módulo. Con el ángulo $\theta$ entre $\hat r$ y el plano perpendicular al hilo, se tiene $z'=d\tan\theta$, $r=d/\cos\theta$ y $|d\vec\ell\times\hat r|=dz'\cos\theta$, de modo que el integrando se reduce a $\cos\theta\,d\theta$ y la integral es elemental.",
         [r"dB=\dfrac{\mu_0I}{4\pi}\dfrac{dz'\cos\theta}{r^2},\qquad z'=d\tan\theta,\quad dz'=\dfrac{d\,d\theta}{\cos^2\theta},\quad r=\dfrac{d}{\cos\theta}",
          r"B=\dfrac{\mu_0I}{4\pi d}\int_{\theta_1}^{\theta_2}\cos\theta\,d\theta=\dfrac{\mu_0I}{4\pi d}\left(\sin\theta_2-\sin\theta_1\right)",
          r"\text{Segmento simétrico de longitud }\ell:\quad \sin\theta_2=-\sin\theta_1=\dfrac{\ell/2}{\sqrt{(\ell/2)^2+d^2}}\ \Rightarrow\ B=\dfrac{\mu_0I}{2\pi d}\dfrac{\ell/2}{\sqrt{(\ell/2)^2+d^2}}",
          r"\ell\to\infty:\quad B\to\dfrac{\mu_0I}{2\pi d}"]),
        ("Desarrollo 3: espira circular, campo sobre el eje",
         r"Sea una espira de radio $a$ en el plano $z=0$ y un punto del eje a altura $z$. Todos los elementos están a la misma distancia $r=\sqrt{a^2+z^2}$ y $d\vec\ell\perp\vec r$, así que $|d\vec\ell\times\hat r|=d\ell$. Al integrar sobre la circunferencia, las componentes perpendiculares al eje se cancelan por pares diametralmente opuestos, y solo sobrevive la proyección axial, de factor $a/r$. La integral angular es trivial porque el integrando es constante sobre la espira.",
         [r"dB_z=\dfrac{\mu_0I}{4\pi}\dfrac{d\ell}{r^2}\cdot\dfrac{a}{r},\qquad \oint d\ell=2\pi a",
          r"B_z(z)=\dfrac{\mu_0I}{4\pi}\dfrac{2\pi a\cdot a}{(a^2+z^2)^{3/2}}=\dfrac{\mu_0Ia^2}{2(a^2+z^2)^{3/2}}",
          r"B_z(0)=\dfrac{\mu_0I}{2a},\qquad B_z\xrightarrow{|z|\gg a}\dfrac{\mu_0Ia^2}{2|z|^3}"]),
        ("Desarrollo 4: momento dipolar, campo lejano y estructura fuera del eje",
         r"Para $|z|\gg a$ el campo de la espira coincide con el de un dipolo magnético de momento $m=I\pi a^2$ (corriente por área, con dirección normal por la regla de la mano derecha). Fuera del eje, $\nabla\cdot\vec B=0$ obliga a que exista una componente radial cuando $B_z$ varía con $z$: en coordenadas cilíndricas y cerca del eje, $B_\rho\approx-(\rho/2)\,\partial B_z/\partial z$. Esto explica por qué el campo de la espira diverge hacia afuera por encima y por debajo del plano, y es el origen de la fuerza sobre dipolos en gradientes de campo.",
         [r"m=I\,\pi a^2,\qquad B_z\simeq\dfrac{\mu_0}{4\pi}\dfrac{2m}{|z|^3}",
          r"\dfrac1\rho\dfrac{\partial(\rho B_\rho)}{\partial\rho}+\dfrac{\partial B_z}{\partial z}=0\ \Rightarrow\ B_\rho(\rho,z)\simeq-\dfrac\rho2\dfrac{\partial B_z}{\partial z}=\dfrac{3\mu_0Ia^2}{4}\dfrac{z\,\rho}{(a^2+z^2)^{5/2}}",
          r"\dfrac{B_z(z)}{B_z(0)}=\left(1+\dfrac{z^2}{a^2}\right)^{-3/2}=1-\dfrac32\dfrac{z^2}{a^2}+\mathcal O\!\left(\dfrac{z^4}{a^4}\right)"]),
        ("Resultado general",
         r"Las dos configuraciones se resumen en dos fórmulas cerradas válidas para cualquier radio, longitud, corriente y distancia. El campo es lineal en $I$ y depende de la geometría solo mediante cocientes de longitudes; la comprobación dimensional muestra que ambas dan tesla. El caso del hilo contiene al hilo infinito como límite y el de la espira contiene al dipolo como límite.",
         [r"\boxed{B_z^{\,\mathrm{espira}}(z)=\dfrac{\mu_0 I a^2}{2\,(a^2+z^2)^{3/2}}}\qquad\boxed{B^{\,\mathrm{hilo}}(d)=\dfrac{\mu_0I}{4\pi d}\left(\sin\theta_2-\sin\theta_1\right)}",
          r"[\mu_0Ia^2/a^3]=\dfrac{\mathrm{T\cdot m}}{\mathrm{A}}\cdot\dfrac{\mathrm{A}}{\mathrm{m}}=\mathrm{T}\ \checkmark"]),
        ("Particularización al problema",
         rf"Con $a={a*100:.1f}$ cm e $I={cur:.1f}$ A, el centro de la espira da $B(0)=\mu_0I/2a$ y a $z=5.0$ cm el factor geométrico vale $(1+1)^{{-3/2}}=2^{{-3/2}}$. Para el hilo de {wl*100:.0f} cm a {wd*100:.1f} cm de su punto medio, $\sin\theta={sn:.5f}$. Se recuperan las cifras analíticas de la escena.",
         [rf"B(0)=\dfrac{{({sci(MU0,5)})({cur:.1f})}}{{2({a:.3f})}}={b0*1e6:.4f}\ \mu\mathrm{{T}},\qquad B(0.05)=B(0)\,2^{{-3/2}}={b5*1e6:.4f}\ \mu\mathrm{{T}}",
          rf"B_{{hilo}}=\dfrac{{({sci(MU0,5)})({cur:.1f})}}{{2\pi({wd:.3f})}}({sn:.5f})={bw*1e6:.4f}\ \mu\mathrm{{T}},\qquad B_\infty={bi*1e6:.4f}\ \mu\mathrm{{T}}",
          rf"m=I\pi a^2={sci(mom,4)}\ \mathrm{{A\cdot m^2}}"]),
        ("Errores comunes y límites de validez",
         r"Un error frecuente es aplicar $\mu_0I/2\pi d$ a un hilo corto: el factor $\sin\theta<1$ reduce el campo, aquí en más de un 6 %. Otro es sumar magnitudes en lugar de vectores, olvidando que las componentes transversales de la espira se cancelan, o usar la fórmula del eje fuera del eje. Biot-Savart supone corrientes estacionarias y filiformes; para corrientes que varían rápidamente hay que incluir la corriente de desplazamiento y los efectos de retardo, y para grosores finitos hay que integrar $\vec J\,dV$ en lugar de $I\,d\vec\ell$.",
         [r"\text{Hilo corto frente a infinito:}\quad \dfrac{B_{\mathrm{hilo}}}{B_\infty}=\sin\theta_{\max}=\dfrac{\ell/2}{\sqrt{(\ell/2)^2+d^2}}<1"]),
    ]


# =============================================================================== SolenoidAxis
def _sol() -> list:
    n, ln, r, cur = m9.N_TURNS, m9.L_SOL, m9.R_SOL, m9.I_SOL
    dens = n / ln
    binf = m9.solenoid_center_infinite()
    b0 = float(m9.solenoid_axis_exact(0.0))
    be = float(m9.solenoid_axis_exact(ln / 2))
    ca = (ln / 2) / np.hypot(ln / 2, r)
    ratio = b0 / binf
    return [
        ("Qué se estudia y por qué importa",
         r"El solenoide es la forma práctica de producir un campo magnético intenso y casi uniforme en un volumen acotado: se usa en electroimanes, relés, bobinas de resonancia magnética, espectrómetros y en el interior de los aceleradores. El problema general consiste en conocer el campo en todos los puntos del eje, en particular cuán uniforme es y cuánto se aparta del ideal de longitud infinita. La respuesta combina dos herramientas: la ley de Ampère, que da el límite ideal sin integrales, y la superposición de espiras, que da el resultado finito exacto.",
         []),
        ("Magnitudes y convenciones",
         r"Un solenoide de $N$ vueltas, longitud $\ell$ y radio $a$ tiene densidad lineal de vueltas $n=N/\ell$ (en $\mathrm{m^{-1}}$); la corriente $I$ se mide en amperes y el campo en tesla. El eje $z$ se toma con origen en el centro; los ángulos $\alpha_1$ y $\alpha_2$ son los que forman con el eje las rectas que unen el punto de campo con los bordes de ambos extremos. La regla de la mano derecha da el sentido del campo: los dedos siguen la corriente de las espiras y el pulgar apunta a lo largo de $+\hat z$. Con $n\sim10^3\ \mathrm{m^{-1}}$ y $I\sim1$ A, $B$ es del orden del militesla.",
         [r"n=\dfrac N\ell,\qquad [\mu_0nI]=\dfrac{\mathrm{T\cdot m}}{\mathrm{A}}\cdot\dfrac{1}{\mathrm{m}}\cdot\mathrm{A}=\mathrm{T}"]),
        ("Intuición y analogías",
         r"Dentro de un solenoide largo, las contribuciones de las espiras se refuerzan; fuera, las de un lado cancelan las del otro y el campo es muy débil. Es el análogo magnético del capacitor de placas paralelas: campo casi uniforme dentro, casi nulo fuera, y las desviaciones aparecen en los bordes. En los extremos, el campo cae a la mitad porque solo la mitad de las espiras queda de un lado del punto, como en el borde de una fila de personas que empujan: en el interior hay vecinos por ambos lados.",
         []),
        ("Desarrollo 1: solenoide infinito con la ley de Ampère",
         r"Por simetría traslacional y azimutal, y porque $\nabla\cdot\vec B=0$ prohíbe una componente radial uniforme, el campo es axial y depende solo del radio: $\vec B=B(\rho)\hat z$. Se aplica $\oint\vec B\cdot d\vec\ell=\mu_0I_{enc}$ a un rectángulo con un lado dentro y otro fuera. Los lados transversales no aportan y, como a gran distancia el campo debe anularse, el campo exterior es cero, y se obtiene el interior uniforme. El rectángulo de longitud $h$ encierra $nh$ vueltas.",
         [r"\oint\vec B\cdot d\vec\ell=B_{\mathrm{int}}\,h-B_{\mathrm{ext}}\,h=\mu_0\,(n h)\,I",
          r"B_{\mathrm{ext}}\to0\ \Rightarrow\ B_{\mathrm{int}}=\mu_0nI,\qquad \text{independiente de }\rho\text{ y de }a"]),
        ("Desarrollo 2: solenoide finito por superposición de espiras",
         r"Un tramo $dz'$ del solenoide contiene $n\,dz'$ espiras, cada una con el campo axial de la espira circular conocido. Si el punto está en $z$, la distancia al tramo es $s=z-z'$ y la superposición es una integral en $z'$ de $-\ell/2$ a $\ell/2$. La sustitución $s=a\cot\alpha$ convierte $(a^2+s^2)^{-3/2}ds$ en $-\sin\alpha\,d\alpha/a$, y la integral queda como diferencia de cosenos de los semiángulos que subtienden los extremos.",
         [r"dB=\dfrac{\mu_0\,(n\,dz')\,I\,a^2}{2\left[a^2+(z-z')^2\right]^{3/2}}",
          r"B(z)=\dfrac{\mu_0nIa^2}{2}\int_{-\ell/2}^{\ell/2}\dfrac{dz'}{\left[a^2+(z-z')^2\right]^{3/2}}=\dfrac{\mu_0nI}{2}\left[\dfrac{z'-z}{\sqrt{a^2+(z'-z)^2}}\right]_{-\ell/2}^{\ell/2}",
          r"B(z)=\dfrac{\mu_0nI}{2}\left(\cos\alpha_2-\cos\alpha_1\right)=\dfrac{\mu_0nI}{2}\left[\dfrac{z+\ell/2}{\sqrt{a^2+(z+\ell/2)^2}}-\dfrac{z-\ell/2}{\sqrt{a^2+(z-\ell/2)^2}}\right]"]),
        ("Desarrollo 3: casos límite y parámetro adimensional",
         r"El único parámetro adimensional del problema es la razón de aspecto $\ell/a$. En el centro, el resultado se simplifica a una forma que tiende a $\mu_0nI$ cuando $\ell\gg a$: el solenoide infinito es el límite de aspecto grande. En un extremo ($z=\ell/2$), el segundo término se anula y el primero da casi la mitad del valor central. Para $\ell\ll a$ se recupera la espira de $N$ vueltas, $B\to\mu_0NI/2a$.",
         [r"B(0)=\mu_0nI\,\dfrac{\ell/2}{\sqrt{(\ell/2)^2+a^2}}=\mu_0nI\,\dfrac{1}{\sqrt{1+(2a/\ell)^2}}",
          r"B(\pm\ell/2)=\dfrac{\mu_0nI}{2}\dfrac{\ell}{\sqrt{\ell^2+a^2}}\ \xrightarrow{\ell\gg a}\ \dfrac12\mu_0nI",
          r"\ell\ll a:\quad B(0)\to\dfrac{\mu_0nI\,\ell}{2a}=\dfrac{\mu_0NI}{2a}"]),
        ("Desarrollo 4: uniformidad cerca del centro y componente radial",
         r"La derivada de $B(z)$ en el centro es nula por simetría, y la primera corrección es cuadrática. Derivar la forma con cosenos permite estimar el tamaño de la región donde el campo es uniforme dentro de una tolerancia dada. La componente radial cerca del eje, exigida por $\nabla\cdot\vec B=0$, es proporcional al gradiente axial y es máxima cerca de los bordes, donde las líneas de campo divergen.",
         [r"\dfrac{B(z)}{B(0)}=1+\dfrac{1}{2}\dfrac{B''(0)}{B(0)}z^2+\dots,\qquad B''(0)=-\dfrac{\mu_0nI\,\ell\,a^2\cdot 3}{\left[(\ell/2)^2+a^2\right]^{5/2}}\cdot\dfrac{1}{8}\cdot 4\cdot\dfrac{1}{2}\cdot 0+\dots",
          r"B_\rho(\rho,z)\simeq-\dfrac\rho2\dfrac{dB}{dz}"]),
        ("Resultado general",
         r"El campo sobre el eje de un solenoide finito depende de la posición solo mediante los cosenos de los ángulos de los extremos, y se reduce al valor ideal $\mu_0nI$ cuando ambos extremos están lejos. La verificación dimensional da tesla, y la dependencia es lineal en $I$ y $n$ pero solo geométrica en $\ell/a$ y $z/a$.",
         [r"\boxed{B(z)=\dfrac{\mu_0nI}{2}\left[\dfrac{z+\ell/2}{\sqrt{a^2+(z+\ell/2)^2}}-\dfrac{z-\ell/2}{\sqrt{a^2+(z-\ell/2)^2}}\right]}",
          r"B_\infty=\mu_0nI,\qquad B(0)=B_\infty\left[1+(2a/\ell)^2\right]^{-1/2},\qquad B(\ell/2)\simeq\tfrac12B_\infty"]),
        ("Particularización al problema",
         rf"Con $N={n}$, $\ell={ln:.2f}$ m, $a={r*100:.1f}$ cm e $I={cur:.1f}$ A, la densidad es $n={dens:.0f}\ \mathrm{{m^{{-1}}}}$ y la razón $2a/\ell={2*r/ln:.3f}$. El campo del solenoide infinito supera al del finito en un factor $\sqrt{{1+(2a/\ell)^2}}$, apenas una fracción de porcentaje.",
         [rf"B_\infty=\mu_0nI=({sci(MU0,5)})({dens:.0f})({cur:.1f})={binf*1e3:.4f}\ \mathrm{{mT}}",
          rf"B(0)=B_\infty\left[1+({2*r/ln:.3f})^2\right]^{{-1/2}}={b0*1e3:.4f}\ \mathrm{{mT}}\quad(B(0)/B_\infty={ratio:.5f})",
          rf"B(\ell/2)=\dfrac{{\mu_0nI}}{{2}}\dfrac{{\ell}}{{\sqrt{{\ell^2+a^2}}}}={be*1e3:.4f}\ \mathrm{{mT}}\quad(\cos\alpha={ca:.5f}\text{{ en el centro}})"]),
        ("Errores comunes y límites de validez",
         r"Es habitual usar $\mu_0nI$ también en los extremos o fuera del solenoide, donde es incorrecto: el extremo vale aproximadamente la mitad. Otro error es confundir $n$ (vueltas por metro) con $N$ (vueltas totales). El modelo supone espiras apiladas con corriente puramente azimutal; un devanado helicoidal real añade una corriente axial pequeña, y un núcleo ferromagnético reemplaza $\mu_0$ por $\mu$ con efectos de saturación e histéresis.",
         [r"B_{\mathrm{ext}}\neq0\ \text{ en realidad: }\ B_{\mathrm{ext}}\sim\dfrac{\mu_0 N I a^2}{4\,d^3}\ \text{(dipolar, lejos)}"]),
    ]


# =============================================================================== RotatingLoop
def _rot() -> list:
    nn, ar, b, f = m10.N_COIL, m10.AREA, m10.B0, m10.FREQ
    w = m10.OMEGA
    e0 = m10.emf_peak_analytic()
    e5 = float(m10.emf_analytic(0.005))
    return [
        ("Qué se estudia y por qué importa",
         r"Un generador de corriente alterna convierte energía mecánica en eléctrica haciendo girar una bobina en un campo magnético: la variación temporal del flujo induce una fem sinusoidal. Es el principio de las centrales hidroeléctricas y térmicas, de los alternadores de automóvil y de las dinamos de bicicleta. El análisis general permite predecir la forma de onda, su amplitud, su valor eficaz y la potencia mecánica que debe aportar el motor que impulsa la bobina.",
         []),
        ("Magnitudes y convenciones",
         r"La bobina tiene $N$ vueltas de área $A$ y gira con velocidad angular $\omega=2\pi f$ alrededor de un eje perpendicular a un campo uniforme $\vec B$. El ángulo $\theta$ es el que forma la normal a la bobina con $\vec B$; el flujo se mide en weber ($\mathrm{Wb}=\mathrm{T\cdot m^2}$) y la fem en voltios. El signo de la fem lo fija la ley de Lenz: se opone a la variación del flujo. En este caso $NBA\sim1\ \mathrm{Wb}$ y a 50 Hz la fem pico es de cientos de voltios.",
         [r"[\Phi_B]=\mathrm{T\cdot m^2}=\mathrm{Wb},\qquad \mathrm{V}=\mathrm{Wb/s}"]),
        ("Intuición y analogías",
         r"El flujo es máximo cuando la bobina está de cara al campo y nulo cuando está de canto, pero la fem depende de la rapidez de cambio del flujo, no del flujo: es máxima cuando el flujo pasa por cero. Es como un péndulo: la posición y la velocidad están desfasadas $90^\circ$. La energía eléctrica generada no sale de la nada: el motor debe vencer un momento magnético de frenado que crece cuanta más corriente se extraiga.",
         []),
        ("Desarrollo 1: ley de Faraday desde la fuerza de Lorentz",
         r"En un conductor que se mueve con velocidad $\vec v$ en un campo $\vec B$, los portadores sienten la fuerza de Lorentz $q\vec v\times\vec B$, equivalente a un campo eléctrico efectivo $\vec v\times\vec B$. La fem es la circulación de ese campo a lo largo del circuito. Con la identidad $\oint(\vec v\times\vec B)\cdot d\vec\ell=-\,d\Phi_B/dt$ para un circuito que se deforma o gira en un campo estático, se obtiene la ley de Faraday; en forma diferencial se escribe $\nabla\times\vec E=-\partial_t\vec B$ y vale también cuando el campo cambia en un circuito fijo.",
         [r"\varepsilon=\oint(\vec v\times\vec B)\cdot d\vec\ell=-\dfrac{d\Phi_B}{dt},\qquad \Phi_B=\int_S\vec B\cdot d\vec A",
          r"\oint\vec E\cdot d\vec\ell=-\dfrac{d}{dt}\int_S\vec B\cdot d\vec A\ \Longleftrightarrow\ \nabla\times\vec E=-\dfrac{\partial\vec B}{\partial t}"]),
        ("Desarrollo 2: flujo y fem de la bobina giratoria",
         r"Para $\vec B$ uniforme, el flujo por las $N$ vueltas es $N B A\cos\theta$ con $\theta=\omega t$ si la rotación es uniforme. La derivada temporal se obtiene con la regla de la cadena, y para una rotación no uniforme aparece directamente $\dot\theta(t)$. La verificación independiente con la fem de movimiento en los dos lados activos (longitud $h$, a distancia $b/2$ del eje, de modo que $A=hb$) da el mismo resultado, lo que confirma la consistencia entre las dos formulaciones.",
         [r"\Phi_B=NBA\cos\theta,\qquad \varepsilon=-\dfrac{d\Phi_B}{dt}=NBA\,\dot\theta\sin\theta",
          r"\dot\theta=\omega:\quad \varepsilon(t)=NBA\omega\,\sin\omega t=\varepsilon_0\sin\omega t",
          r"\text{Lados activos: }v=\omega\dfrac b2,\ \ \varepsilon=2\cdot N\,B\,h\,v\sin\omega t=NBhb\,\omega\sin\omega t=NBA\omega\sin\omega t"]),
        ("Desarrollo 3: valor eficaz y armónicos",
         r"El valor eficaz de una señal periódica se define como la raíz cuadrada del promedio de su cuadrado en un periodo, para que dé la misma potencia en una resistencia que una tensión continua equivalente. Para una sinusoide, el promedio de $\sin^2$ es $1/2$. Si la rotación no es uniforme o el campo no es uniforme, la fem tiene armónicos, y su contenido se obtiene por desarrollo de Fourier del flujo.",
         [r"\varepsilon_{rms}=\sqrt{\dfrac1T\int_0^T\varepsilon_0^2\sin^2\omega t\,dt}=\dfrac{\varepsilon_0}{\sqrt2}",
          r"\Phi(\theta)=\sum_k\Phi_k\cos k\theta\ \Rightarrow\ \varepsilon(t)=\sum_k k\,\omega\,\Phi_k\sin k\omega t"]),
        ("Desarrollo 4: carga, momento de frenado y balance de energía",
         r"Si la bobina alimenta una resistencia $R$ (autoinductancia despreciada), circula $i=\varepsilon/R$. La corriente genera un momento magnético $\vec\mu=NiA\hat n$ y el campo ejerce sobre él un momento $\tau=\mu B\sin\theta$ que frena la rotación. La potencia mecánica necesaria, $\tau\omega$, coincide instante a instante con la potencia eléctrica $\varepsilon i$: la conservación de energía queda verificada. Con inductancia propia $L$, la amplitud de corriente es $\varepsilon_0/\sqrt{R^2+\omega^2L^2}$ y aparece un desfase.",
         [r"\tau=NiAB\sin\theta=\dfrac{(NBA)^2\omega}{R}\sin^2\omega t,\qquad P_{mec}=\tau\omega=\dfrac{\varepsilon^2}{R}=P_{el}",
          r"\bar P=\dfrac{\varepsilon_0^2}{2R}=\dfrac{\varepsilon_{rms}^2}{R},\qquad I_0=\dfrac{\varepsilon_0}{\sqrt{R^2+\omega^2L^2}},\ \ \tan\phi=\dfrac{\omega L}{R}"]),
        ("Resultado general",
         r"La fem es sinusoidal, proporcional a $N$, $B$, $A$ y $\omega$, y adelantada $90^\circ$ respecto al flujo. Es lineal en cada parámetro: duplicar la frecuencia duplica la fem. Dimensionalmente, $[BA\omega]=\mathrm{T\cdot m^2/s}=\mathrm{V}$.",
         [r"\boxed{\varepsilon(t)=NBA\,\omega\,\sin\omega t,\qquad \varepsilon_0=NBA\omega,\qquad \varepsilon_{rms}=\dfrac{NBA\omega}{\sqrt2}}",
          r"[NBA\omega]=\mathrm{T\cdot m^2\cdot s^{-1}}=\mathrm{V}\ \checkmark"]),
        ("Particularización al problema",
         rf"Con $N={nn}$, $A={ar:.3f}\ \mathrm{{m^2}}$, $B={b:.2f}$ T y $f={f:.0f}$ Hz se tiene $\omega=2\pi f={w:.4f}\ \mathrm{{rad/s}}$. La fem pico, la eficaz y el valor instantáneo a 5 ms se obtienen sustituyendo en la fórmula general; en $t=5$ ms, $\omega t=\pi/2$ y la fem alcanza su máximo.",
         [rf"\varepsilon_0=({nn})({b:.2f})({ar:.3f})({w:.4f})={e0:.4f}\ \mathrm{{V}}",
          rf"\varepsilon_{{rms}}=\varepsilon_0/\sqrt2={e0/np.sqrt(2):.4f}\ \mathrm{{V}},\qquad \varepsilon(5\ \mathrm{{ms}})={e5:.4f}\ \mathrm{{V}}"]),
        ("Errores comunes y límites de validez",
         r"No debe confundirse el máximo del flujo con el máximo de la fem: están en cuadratura. Tampoco debe olvidarse el factor $N$ ni usar el valor pico donde se pide el eficaz. El modelo supone campo uniforme, rotación estable y autoinductancia despreciable; con carga inductiva aparece el desfase $\phi$, y a gran corriente la reacción de armadura distorsiona el campo, por lo que la fem de un generador real deja de ser puramente sinusoidal.",
         []),
    ]


# =============================================================================== SlidingRod
def _rod() -> list:
    b, ln, m, r, v0 = m10.B_ROD, m10.L_ROD, m10.M_ROD, m10.R_CIRC, m10.V0_ROD
    tau = m10.tau_rod()
    k = b**2 * ln**2 / (m * r)
    xinf = m10.rod_distance_analytic()
    e0 = 0.5 * m * v0**2
    return [
        ("Qué se estudia y por qué importa",
         r"Una barra conductora que se desplaza en un campo magnético sobre un circuito cerrado genera una fem de movimiento, la corriente resultante siente una fuerza magnética y la barra se frena: es el freno magnético por corrientes de Foucault de los trenes, de las atracciones de feria y de los medidores de energía. El problema contiene en pequeño toda la conversión electromecánica: energía cinética que se transforma en calor Joule. Su análisis general da la ley de movimiento, la distancia recorrida y el balance energético exacto.",
         []),
        ("Magnitudes y convenciones",
         r"Los datos son la masa $m$, la longitud efectiva $\ell$, la resistencia total $R$ del circuito, el campo $B$ perpendicular al plano y la velocidad inicial $v_0$. La fem de movimiento es $\varepsilon=B\ell v$ (en voltios), la corriente $i=\varepsilon/R$ y la fuerza $F=iB\ell$ (en newtons) se opone al movimiento por la ley de Lenz. El coeficiente $B^2\ell^2/R$ tiene unidades de coeficiente de amortiguamiento viscoso (kg/s).",
         [r"\left[\dfrac{B^2\ell^2}{R}\right]=\dfrac{\mathrm{T^2m^2}}{\Omega}=\dfrac{\mathrm{N^2}}{\mathrm{A^2}}\dfrac{\mathrm{A^2 s}}{\mathrm{N\,m}}\cdot\dfrac{\mathrm{m^2}}{\mathrm{m^2}}=\dfrac{\mathrm{kg}}{\mathrm{s}}"]),
        ("Intuición y analogías",
         r"La barra se comporta como un cuerpo que se mueve dentro de un fluido viscoso: la fuerza de frenado es proporcional a la velocidad y su coeficiente de amortiguamiento lo fijan $B$, $\ell$ y $R$. Si el circuito está abierto ($R\to\infty$) no hay corriente ni frenado; si $R\to0$ el amortiguamiento es enorme. El balance es limpio: toda la energía cinética se convierte en calor en la resistencia.",
         []),
        ("Desarrollo 1: fem de movimiento y corriente",
         r"Un portador de carga $q$ en la barra, que se mueve con velocidad $v\hat x$ en $\vec B=B\hat z$, siente $q\vec v\times\vec B$ a lo largo de la barra ($\hat y$). La fem es el trabajo por unidad de carga a lo largo de la barra; equivale a $-d\Phi_B/dt$ porque el área del circuito crece como $\ell x$. El sentido de la corriente es tal que la fuerza sobre ella se opone al movimiento.",
         [r"\varepsilon=\int_0^{\ell}(\vec v\times\vec B)\cdot d\vec\ell=B\ell v=-\dfrac{d}{dt}\left(-B\ell x\right)",
          r"i=\dfrac{\varepsilon}{R}=\dfrac{B\ell v}{R},\qquad \vec F=i\,\vec\ell\times\vec B=-\dfrac{B^2\ell^2}{R}\,v\,\hat x"]),
        ("Desarrollo 2: ecuación de movimiento y solución exacta",
         r"La segunda ley de Newton para la única fuerza horizontal da una ecuación lineal de primer orden en $v$, que se integra por separación de variables. Integrando una vez más se obtiene la posición, que converge a un alcance finito: la barra recorre $v_0\tau$ aunque tarde un tiempo infinito en detenerse. La solución depende de un solo parámetro con dimensión de tiempo, $\tau=mR/B^2\ell^2$.",
         [r"m\dfrac{dv}{dt}=-\dfrac{B^2\ell^2}{R}v\ \Rightarrow\ \dfrac{dv}{v}=-\dfrac{dt}{\tau},\qquad \tau=\dfrac{mR}{B^2\ell^2}",
          r"v(t)=v_0e^{-t/\tau},\qquad x(t)=v_0\tau\left(1-e^{-t/\tau}\right),\qquad x_\infty=v_0\tau",
          r"i(t)=\dfrac{B\ell v_0}{R}e^{-t/\tau}"]),
        ("Desarrollo 3: balance de energía",
         r"La potencia disipada en la resistencia es $i^2R$, y la potencia mecánica extraída por la fuerza de frenado es $-Fv$. Ambas coinciden en cada instante, lo que confirma que el trabajo mecánico se convierte íntegramente en calor Joule. Integrando desde 0 hasta infinito se recupera la energía cinética inicial.",
         [r"i^2R=\dfrac{B^2\ell^2v^2}{R}=-Fv=-m\,v\dot v=-\dfrac{d}{dt}\left(\tfrac12mv^2\right)",
          r"E_J=\int_0^\infty\dfrac{B^2\ell^2v_0^2}{R}e^{-2t/\tau}dt=\dfrac{B^2\ell^2v_0^2}{R}\dfrac\tau2=\tfrac12mv_0^2"]),
        ("Desarrollo 4: generalizaciones, fuerza externa e inductancia",
         r"Con una fuerza externa constante $F_0$ la ecuación es $m\dot v=F_0-m v/\tau$ y la velocidad tiende a la velocidad terminal $v_T=F_0R/B^2\ell^2$, el principio del generador lineal. Si el circuito tiene autoinductancia $L$, el sistema pasa a ser de segundo orden en $(v,i)$: $m\dot v=B\ell i$ y $L\,\dot i+Ri=-B\ell v$. Eliminando $i$ aparece un oscilador amortiguado cuyo caso sobreamortiguado reproduce la solución sin inductancia cuando $L/R\ll\tau$.",
         [r"v(t)=v_T+(v_0-v_T)e^{-t/\tau},\qquad v_T=\dfrac{F_0R}{B^2\ell^2}",
          r"\ddot v+\dfrac RL\dot v+\dfrac{B^2\ell^2}{mL}v=0,\qquad s_\pm=-\dfrac{R}{2L}\pm\sqrt{\dfrac{R^2}{4L^2}-\dfrac{B^2\ell^2}{mL}}",
          r"\dfrac{L}{R}\ll\tau\ \Rightarrow\ s_-\simeq-\dfrac{1}{\tau}"]),
        ("Resultado general",
         r"La velocidad decae exponencialmente con constante de tiempo $\tau$, la barra recorre una distancia finita proporcional a $v_0$ y a $\tau$, y la energía disipada es la cinética inicial. Dimensionalmente, $\tau$ es un tiempo y $x_\infty$ una longitud.",
         [r"\boxed{v(t)=v_0\,e^{-t/\tau},\qquad \tau=\dfrac{mR}{B^2\ell^2},\qquad x_\infty=\dfrac{m R v_0}{B^2\ell^2},\qquad E_J=\tfrac12mv_0^2}",
          r"[\tau]=\dfrac{\mathrm{kg\cdot\Omega}}{\mathrm{T^2m^2}}=\mathrm{s}\ \checkmark"]),
        ("Particularización al problema",
         rf"Con $m={m:.2f}$ kg, $\ell={ln:.2f}$ m, $R={r:.1f}\ \Omega$, $B={b:.2f}$ T y $v_0={v0:.1f}$ m/s, el coeficiente de frenado es $B^2\ell^2/R={b**2*ln**2/r:.4f}\ \mathrm{{kg/s}}$, es decir $1/\tau={k:.4f}\ \mathrm{{s^{{-1}}}}$.",
         [rf"\tau=\dfrac{{({m:.2f})({r:.1f})}}{{({b:.2f})^2({ln:.2f})^2}}={tau:.3f}\ \mathrm{{s}},\qquad x_\infty=v_0\tau={xinf:.4f}\ \mathrm{{m}}",
          rf"v(4\ \mathrm{{s}})={v0:.1f}\,e^{{-4/{tau:.3f}}}={float(m10.rod_velocity_analytic(4.0)):.5f}\ \mathrm{{m/s}}",
          rf"E_J=\tfrac12({m:.2f})({v0:.1f})^2={e0:.4f}\ \mathrm{{J}}"]),
        ("Errores comunes y límites de validez",
         r"Suele suponerse que la barra se detiene en un tiempo finito: en realidad $v\to0$ solo asintóticamente. Otro error es olvidar que la fuerza de frenado es cuadrática en $B\ell$ (una potencia de $B\ell$ para la fem y otra para la fuerza). El modelo supone rieles ideales sin fricción ni resistencia, campo uniforme y autoinductancia nula; con inductancia aparece el sistema de segundo orden, y a velocidades enormes hay que considerar efectos relativistas y calentamiento que cambia $R$.",
         []),
    ]


# =============================================================================== RLCircuit
def _rl() -> list:
    eps, r, ln = m11.EPS, m11.R, m11.L
    tau = m11.tau_rl()
    ifin = eps / r
    u = m11.energy_magnetic()
    return [
        ("Qué se estudia y por qué importa",
         r"Un inductor se opone a los cambios de corriente: al cerrar un circuito, la corriente no salta al valor de régimen sino que crece de forma gradual. Esto limita las corrientes de arranque en motores y transformadores, protege circuitos con diodos de rueda libre y determina la rapidez con que se magnetiza un electroimán. Con la ecuación general se estudia la respuesta a cualquier fuente $\varepsilon(t)$ y se identifica la escala de tiempo de todo el sistema.",
         []),
        ("Magnitudes y convenciones",
         r"La inductancia $L$ se mide en henrios ($1\ \mathrm{H}=1\ \mathrm{V\cdot s/A}$), la resistencia $R$ en ohmios y la fem $\varepsilon$ en voltios. La fem autoinducida es $\varepsilon_L=-L\,di/dt$ y se opone al cambio de corriente (ley de Lenz). La constante $\tau_L=L/R$ tiene unidades de tiempo; típicamente va de microsegundos en bobinas pequeñas a segundos en electroimanes grandes.",
         [r"[L/R]=\dfrac{\mathrm{V\,s/A}}{\mathrm{V/A}}=\mathrm{s}"]),
        ("Intuición y analogías",
         r"La inductancia es la inercia del circuito: así como una masa se opone a cambios de velocidad, un inductor se opone a cambios de corriente. La analogía mecánica es una masa $L$ sometida a una fuerza $\varepsilon$ con rozamiento viscoso $R$; la corriente corresponde a la velocidad y el régimen estacionario a la velocidad terminal $\varepsilon/R$. La energía $\tfrac12Li^2$ es la energía cinética de esa inercia, almacenada en el campo magnético.",
         []),
        ("Desarrollo 1: la inductancia desde el campo",
         r"El flujo propio que atraviesa un circuito es proporcional a su corriente, $\Phi=Li$, porque el campo lo es (Biot-Savart es lineal). Si la corriente cambia, la ley de Faraday da una fem autoinducida. Para un solenoide largo, con $B=\mu_0Ni/\ell$ y flujo por vuelta $BA$, se obtiene $L=\mu_0N^2A/\ell$, que depende solo de la geometría.",
         [r"\Phi_{tot}=Li,\qquad \varepsilon_L=-\dfrac{d\Phi_{tot}}{dt}=-L\dfrac{di}{dt}",
          r"\Phi_{tot}=N\cdot\dfrac{\mu_0Ni}{\ell}A\ \Rightarrow\ L=\dfrac{\mu_0N^2A}{\ell}"]),
        ("Desarrollo 2: ecuación del circuito y solución con condición inicial arbitraria",
         r"La segunda ley de Kirchhoff para la malla da una ecuación diferencial lineal de primer orden. Se resuelve con el factor integrante $e^{t/\tau}$, que convierte el miembro izquierdo en una derivada exacta; la constante queda fijada por $i(0)=i_0$. La solución es la suma de una solución permanente y una transitoria que decae.",
         [r"\varepsilon-iR-L\dfrac{di}{dt}=0\ \Rightarrow\ \dfrac{di}{dt}+\dfrac{i}{\tau}=\dfrac{\varepsilon}{L}",
          r"\dfrac{d}{dt}\left(i\,e^{t/\tau}\right)=\dfrac{\varepsilon}{L}e^{t/\tau}\ \Rightarrow\ i(t)=\dfrac\varepsilon R+\left(i_0-\dfrac\varepsilon R\right)e^{-t/\tau}",
          r"i_0=0:\quad i(t)=\dfrac{\varepsilon}{R}\left(1-e^{-t/\tau}\right);\qquad \varepsilon=0:\quad i(t)=i_0e^{-t/\tau}"]),
        ("Desarrollo 3: fuente arbitraria y respuesta impulsional",
         r"Como la ecuación es lineal e invariante en el tiempo, la respuesta a una fuente $\varepsilon(t)$ cualquiera es la convolución de la fuente con la respuesta a un impulso (función de Green) $G(t)=e^{-t/\tau}/L$. Esta fórmula incluye el escalón como caso particular, y para una fuente sinusoidal conduce, tras el transitorio, a la impedancia $Z=R+i\omega L$.",
         [r"i(t)=i_0e^{-t/\tau}+\dfrac1L\int_0^t e^{-(t-s)/\tau}\varepsilon(s)\,ds",
          r"\varepsilon=\varepsilon_m\sin\omega t:\quad i_{ss}(t)=\dfrac{\varepsilon_m}{\sqrt{R^2+\omega^2L^2}}\sin(\omega t-\phi),\quad \tan\phi=\dfrac{\omega L}{R}"]),
        ("Desarrollo 4: balance de energía y tiempos característicos",
         r"Multiplicando la ecuación del circuito por $i$ se obtiene el balance de potencias: lo que entrega la fuente se reparte entre el calor Joule y el almacenamiento en el campo. La energía magnética, integrando $Li\,di$, vale $\tfrac12Li^2$; en términos del campo es $\int B^2/2\mu_0\,dV$. El tiempo para alcanzar una fracción $f$ del valor final es $-\tau\ln(1-f)$, y el de decaimiento a la mitad es $\tau\ln2$.",
         [r"\varepsilon\,i=i^2R+\dfrac{d}{dt}\left(\tfrac12Li^2\right),\qquad U_B=\int_0^{i}Li'\,di'=\tfrac12Li^2=\int\dfrac{B^2}{2\mu_0}dV",
          r"t_f=-\tau\ln(1-f),\qquad t_{1/2}^{\,\mathrm{decae}}=\tau\ln2"]),
        ("Resultado general",
         r"La corriente se acerca al valor de régimen $\varepsilon/R$ con una exponencial de constante $L/R$, y la energía final es $L\varepsilon^2/2R^2$. Solo el cociente $L/R$ controla la dinámica; $L$ y $R$ por separado fijan la energía y la corriente.",
         [r"\boxed{i(t)=\dfrac{\varepsilon}{R}\left(1-e^{-t/\tau_L}\right),\qquad \tau_L=\dfrac LR,\qquad U_B=\dfrac{L\varepsilon^2}{2R^2}}",
          r"[LE^2/R^2]=\mathrm{H\cdot A^2}=\mathrm{J}\ \checkmark"]),
        ("Particularización al problema",
         rf"Con $L={ln:.2f}$ H, $R={r:.1f}\ \Omega$ y $\varepsilon={eps:.0f}$ V se obtiene la corriente final $i_\infty=\varepsilon/R={ifin:.2f}$ A. La fracción de la corriente final en un tiempo $\tau$ es $1-e^{{-1}}$, y el tiempo de decaimiento al 50 % tras retirar la fuente es $\tau\ln2$, que coincide con el valor de la fila de la escena.",
         [rf"\tau_L=\dfrac{{{ln:.2f}}}{{{r:.1f}}}={tau*1e3:.2f}\ \mathrm{{ms}},\qquad i_\infty={ifin:.2f}\ \mathrm{{A}},\qquad i(\tau)=i_\infty(1-e^{{-1}})={ifin*(1-np.exp(-1)):.4f}\ \mathrm{{A}}",
          rf"t_{{1/2}}=\tau\ln2={tau*np.log(2)*1e3:.4f}\ \mathrm{{ms}},\qquad U_B=\tfrac12({ln:.2f})({ifin:.2f})^2={u:.4f}\ \mathrm{{J}}"]),
        ("Errores comunes y límites de validez",
         r"Es común creer que la corriente alcanza el valor final en un tiempo $\tau$: solo llega al 63 %; se necesitan unas $5\tau$ para el 99.3 %. También se confunde la fem autoinducida con la fuente. El modelo supone $L$ constante (sin núcleo saturable), resistencia constante (sin calentamiento) y circuito concentrado; a frecuencias altas la capacidad parásita entre espiras modifica la respuesta, y con núcleos ferromagnéticos $L$ depende de $i$.",
         []),
    ]


# =============================================================================== RLCOscillator
def _rlc() -> list:
    l_, c, r, q0 = m11.L2, m11.C2, m11.R2, m11.Q0
    w0, g, wd = m11.omega0(), m11.gamma_(), m11.omega_d()
    t = 2 * np.pi / wd
    qf = m11.quality_factor()
    u0 = q0**2 / (2 * c)
    return [
        ("Qué se estudia y por qué importa",
         r"Un capacitor cargado que se conecta a un inductor intercambia energía con él de forma oscilatoria, mientras la resistencia la disipa gradualmente. Este circuito RLC es el modelo del oscilador amortiguado y subyace a los sintonizadores de radio, a los filtros, a los osciladores de reloj y a los circuitos de ignición. El análisis general determina los regímenes posibles (subamortiguado, crítico, sobreamortiguado), la frecuencia, el decaimiento y el factor de calidad.",
         []),
        ("Magnitudes y convenciones",
         r"La carga del capacitor $q$ se mide en coulomb, la corriente es $i=\dot q$ (amperes), $L$ en henrios, $C$ en faradios y $R$ en ohmios. Se definen la frecuencia natural $\omega_0=1/\sqrt{LC}$, el coeficiente de amortiguamiento $\gamma=R/2L$ y el factor de calidad $Q=\omega_0L/R$. La energía total es $U=q^2/2C+Li^2/2$ en joule.",
         [r"[\omega_0]=\dfrac{1}{\sqrt{\mathrm{H\cdot F}}}=\mathrm{s^{-1}},\qquad [\gamma]=\dfrac{\Omega}{\mathrm H}=\mathrm{s^{-1}}"]),
        ("Intuición y analogías",
         r"Es el equivalente eléctrico del sistema masa-resorte con amortiguador: $L$ hace de masa, $1/C$ de constante del resorte y $R$ de coeficiente de rozamiento; $q$ es la posición e $i$ la velocidad. La energía oscila entre la forma eléctrica del capacitor (potencial) y la magnética del inductor (cinética), y en cada ciclo una fracción $\sim2\pi/Q$ se pierde en calor.",
         []),
        ("Desarrollo 1: formulación lagrangiana con disipación",
         r"Tomando $q$ como coordenada generalizada, la energía cinética magnética es $T=\tfrac12L\dot q^2$ y la potencial eléctrica $V=q^2/2C$. Las fuerzas disipativas se incorporan con la función de Rayleigh $\mathcal F=\tfrac12R\dot q^2$, de manera que las ecuaciones de Lagrange tienen un término adicional. El resultado coincide con la ley de Kirchhoff para la malla.",
         [r"\mathcal L=\tfrac12L\dot q^2-\dfrac{q^2}{2C},\qquad \mathcal F=\tfrac12R\dot q^2",
          r"\dfrac{d}{dt}\dfrac{\partial\mathcal L}{\partial\dot q}-\dfrac{\partial\mathcal L}{\partial q}=-\dfrac{\partial\mathcal F}{\partial\dot q}\ \Rightarrow\ L\ddot q+R\dot q+\dfrac qC=0",
          r"\dfrac{dU}{dt}=-R\,i^2\le0"]),
        ("Desarrollo 2: ecuación característica y regímenes",
         r"La ecuación es lineal con coeficientes constantes, de modo que se buscan soluciones $q=e^{st}$. La ecuación característica es cuadrática y sus raíces dependen del signo del discriminante $\gamma^2-\omega_0^2$. Si $\gamma<\omega_0$, las raíces son complejas conjugadas y hay oscilación; si $\gamma=\omega_0$ el amortiguamiento es crítico (el retorno más rápido sin oscilar); si $\gamma>\omega_0$ ambas raíces son reales y negativas.",
         [r"\ddot q+2\gamma\dot q+\omega_0^2q=0\ \Rightarrow\ s^2+2\gamma s+\omega_0^2=0\ \Rightarrow\ s_\pm=-\gamma\pm\sqrt{\gamma^2-\omega_0^2}",
          r"\gamma<\omega_0:\ s_\pm=-\gamma\pm i\omega_d,\quad \omega_d=\sqrt{\omega_0^2-\gamma^2}",
          r"\gamma=\omega_0:\ q=(A+Bt)e^{-\gamma t};\qquad \gamma>\omega_0:\ q=Ae^{s_+t}+Be^{s_-t}"]),
        ("Desarrollo 3: solución subamortiguada con condiciones iniciales",
         r"Para $q(0)=q_0$ e $i(0)=0$ (capacitor cargado, corriente nula), las constantes de la combinación de $e^{-\gamma t}\cos\omega_dt$ y $e^{-\gamma t}\sin\omega_dt$ quedan fijadas en términos de $q_0$ y de $\gamma/\omega_d$. La corriente se obtiene derivando; es proporcional a $\sin\omega_dt$ y se anula en los instantes en que $\omega_dt$ es múltiplo de $\pi$.",
         [r"q(t)=q_0\,e^{-\gamma t}\left(\cos\omega_dt+\dfrac{\gamma}{\omega_d}\sin\omega_dt\right)",
          r"i(t)=\dot q=-q_0\dfrac{\omega_0^2}{\omega_d}\,e^{-\gamma t}\sin\omega_dt",
          r"T_d=\dfrac{2\pi}{\omega_d},\qquad \delta=\ln\dfrac{q(t)}{q(t+T_d)}=\gamma T_d\ \ (\text{decremento logarítmico})"]),
        ("Desarrollo 4: energía y factor de calidad",
         r"La energía instantánea decae al ritmo $-Ri^2$. Como $i=0$ cuando $\omega_dt=n\pi$ y $\sin\omega_dT_d=0$, en $t=T_d$ toda la energía está en el capacitor con carga $q_0e^{-\gamma T_d}$, y por tanto la razón de energías tras un periodo es exactamente $e^{-2\gamma T_d}$. Para amortiguamiento débil, $2\gamma T_d\simeq2\pi/Q$, lo que da la interpretación de $Q$ como $2\pi$ veces la energía almacenada dividida por la perdida por ciclo.",
         [r"\dfrac{U(T_d)}{U(0)}=\dfrac{q^2(T_d)/2C}{q_0^2/2C}=e^{-2\gamma T_d},\qquad Q=\dfrac{\omega_0}{2\gamma}=\dfrac{\omega_0L}{R}=\dfrac1R\sqrt{\dfrac LC}",
          r"Q\gg1:\quad \dfrac{\Delta U}{U}\simeq\dfrac{2\pi}{Q},\qquad \omega_d\simeq\omega_0\left(1-\dfrac{1}{8Q^2}\right)"]),
        ("Resultado general",
         r"La carga oscila con la frecuencia amortiguada, dentro de la envolvente $q_0e^{-\gamma t}$ de tiempo de vida $1/\gamma=2L/R$. Dimensionalmente, todas las frecuencias son $\mathrm{s^{-1}}$.",
         [r"\boxed{q(t)=q_0e^{-\gamma t}\left(\cos\omega_dt+\dfrac\gamma{\omega_d}\sin\omega_dt\right),\quad \gamma=\dfrac R{2L},\ \omega_d=\sqrt{\dfrac1{LC}-\dfrac{R^2}{4L^2}}}",
          r"\boxed{Q=\dfrac{\omega_0L}{R},\qquad \dfrac{U(T_d)}{U(0)}=e^{-2\gamma T_d}}"]),
        ("Particularización al problema",
         rf"Con $L={l_:.2f}$ H, $C={c*1e6:.0f}\ \mu\mathrm{{F}}$, $R={r:.1f}\ \Omega$ y $q_0={q0*1e3:.1f}$ mC se obtienen los parámetros y el periodo amortiguado de la escena; el régimen es subamortiguado porque $\gamma<\omega_0$.",
         [rf"\omega_0=\dfrac{{1}}{{\sqrt{{LC}}}}={w0:.3f}\ \mathrm{{rad/s}},\qquad \gamma=\dfrac{{R}}{{2L}}={g:.2f}\ \mathrm{{s^{{-1}}}},\qquad \omega_d={wd:.3f}\ \mathrm{{rad/s}}",
          rf"T_d=\dfrac{{2\pi}}{{\omega_d}}={t*1e3:.5f}\ \mathrm{{ms}},\qquad Q=\dfrac{{\omega_0L}}{{R}}={qf:.3f}",
          rf"\dfrac{{U(T_d)}}{{U(0)}}=e^{{-2({g:.2f})({sci(t,5)})}}={np.exp(-2*g*t):.5f},\qquad U(0)=\dfrac{{q_0^2}}{{2C}}={u0:.4f}\ \mathrm{{J}}"]),
        ("Errores comunes y límites de validez",
         r"Es frecuente usar $\omega_0$ en lugar de $\omega_d$ para el periodo (la diferencia es pequeña si $Q\gg1$, pero no nula) y olvidar que la energía decae con $2\gamma$ y no con $\gamma$. El modelo supone elementos ideales concentrados: a altas frecuencias hay capacidad parásita del inductor y resistencia dependiente de la frecuencia (efecto pelicular); además, la radiación del circuito añade un amortiguamiento que el modelo no incluye.",
         []),
    ]


# =============================================================================== DipoleTorque
def _dip() -> list:
    mu, b, ii = m12.MU, m12.B_FIELD, m12.INERTIA
    w0 = m12.omega_small()
    t0 = 2 * np.pi / w0
    t30, t90 = m12.period_exact(30.0), m12.period_exact(90.0)
    return [
        ("Qué se estudia y por qué importa",
         r"Una aguja imanada en el campo terrestre oscila alrededor de la dirección del campo: es el principio de la brújula, y mediante la medición del periodo se determina el producto $\mu B$ (magnetómetro de oscilación de Gauss). Con la dinámica general de un dipolo se obtiene el periodo exacto con su dependencia con la amplitud, y se identifica la diferencia entre oscilación y rotación libre. La misma ecuación aparece en el péndulo y en resonancia magnética.",
         []),
        ("Magnitudes y convenciones",
         r"El momento dipolar $\mu$ se mide en $\mathrm{A\cdot m^2}$, el campo $B$ en tesla y el momento de inercia $I$ en $\mathrm{kg\,m^2}$. El ángulo $\theta$ se mide desde la dirección de equilibrio ($\vec\mu\parallel\vec B$). La energía potencial $U=-\vec\mu\cdot\vec B=-\mu B\cos\theta$ es mínima en $\theta=0$. Para una brújula, $\mu\sim0.1\ \mathrm{A\,m^2}$, $B\sim50\ \mu\mathrm{T}$ y $\mu B\sim5\ \mu\mathrm{J}$.",
         [r"[\mu B]=\mathrm{A\,m^2\cdot T}=\mathrm{J},\qquad [\sqrt{\mu B/I}]=\mathrm{s^{-1}}"]),
        ("Intuición y analogías",
         r"El dipolo en un campo uniforme es un péndulo: el momento restaurador $-\mu B\sin\theta$ es el análogo de $-mg\ell\sin\theta$, con $\mu B\leftrightarrow mg\ell$ e $I\leftrightarrow m\ell^2$. Como en el péndulo, para amplitudes pequeñas el periodo es independiente de la amplitud, y para amplitudes grandes se alarga hasta divergir cuando la aguja casi alcanza el equilibrio inestable $\theta=\pi$.",
         []),
        ("Desarrollo 1: el momento sobre una espira de corriente",
         r"Una espira rectangular de lados $a$ y $b$ con corriente $I$ en un campo uniforme siente fuerzas $I\,\vec\ell\times\vec B$ en cada lado. Las fuerzas netas se anulan, pero los dos lados opuestos de longitud $a$ forman un par con brazo $b\sin\theta$, y el momento resultante tiene módulo $IabB\sin\theta$. Esto se escribe vectorialmente con $\vec\mu=IA\hat n$, y como el trabajo del par es $-d(\ldots)$, la energía es $U=-\vec\mu\cdot\vec B$.",
         [r"\vec F_{neta}=I\oint d\vec\ell\times\vec B=\vec 0,\qquad \vec\tau=\vec\mu\times\vec B,\quad \vec\mu=IA\,\hat n",
          r"dW=\tau\,d\theta\ \Rightarrow\ U(\theta)=-\mu B\cos\theta"]),
        ("Desarrollo 2: lagrangiana, hamiltoniana e integral primera",
         r"Para rotación alrededor de un eje fijo perpendicular a $\vec B$, la coordenada es $\theta$ y la lagrangiana es $\mathcal L=\tfrac12I\dot\theta^2+\mu B\cos\theta$. La ecuación de Euler-Lagrange es la del péndulo. Como $\mathcal L$ no depende explícitamente del tiempo, el hamiltoniano $H=p_\theta^2/2I-\mu B\cos\theta$ se conserva (energía), y el sistema se reduce por cuadratura.",
         [r"\mathcal L=\tfrac12I\dot\theta^2+\mu B\cos\theta\ \Rightarrow\ I\ddot\theta=-\mu B\sin\theta",
          r"p_\theta=I\dot\theta,\quad H=\dfrac{p_\theta^2}{2I}-\mu B\cos\theta=E,\qquad \dot\theta=\pm\sqrt{\dfrac{2}{I}\left[E+\mu B\cos\theta\right]}",
          r"\omega_0^2=\dfrac{\mu B}{I}"]),
        ("Desarrollo 3: periodo exacto para amplitud arbitraria",
         r"Con la amplitud $\theta_0$ (velocidad nula allí), la energía es $E=-\mu B\cos\theta_0$. El cuarto de periodo es la integral del tiempo entre $0$ y $\theta_0$, y la sustitución $\sin(\theta/2)=k\sin\varphi$ con $k=\sin(\theta_0/2)$ la convierte en la integral elíptica completa de primera especie. Esta se evalúa con la media aritmético-geométrica, que converge cuadráticamente.",
         [r"T=4\int_0^{\theta_0}\dfrac{d\theta}{\sqrt{(2\mu B/I)\,(\cos\theta-\cos\theta_0)}}=\dfrac{4}{\omega_0}K(k),\qquad k=\sin\dfrac{\theta_0}{2}",
          r"K(k)=\int_0^{\pi/2}\dfrac{d\varphi}{\sqrt{1-k^2\sin^2\varphi}}=\dfrac{\pi}{2\,\mathrm{AGM}(1,\sqrt{1-k^2})}",
          r"T=\dfrac{2\pi}{\omega_0\,\mathrm{AGM}\!\left(1,\cos\frac{\theta_0}{2}\right)}"]),
        ("Desarrollo 4: pequeñas oscilaciones, correcciones y espacio de fases",
         r"Para $\theta_0\ll1$, se linealiza $\sin\theta\simeq\theta$ y la ecuación es la de un oscilador armónico de frecuencia $\omega_0$; la primera corrección anarmónica es cuadrática en la amplitud. En el espacio de fases $(\theta,p_\theta)$, las curvas con $E<\mu B$ son cerradas (libración) y las de $E>\mu B$ son abiertas (rotación); la separatriz $E=\mu B$ pasa por el equilibrio inestable $\theta=\pi$, donde el periodo diverge.",
         [r"T\simeq T_0\left(1+\dfrac{\theta_0^2}{16}+\dfrac{11\,\theta_0^4}{3072}+\dots\right),\qquad T_0=\dfrac{2\pi}{\omega_0}=2\pi\sqrt{\dfrac I{\mu B}}",
          r"E<\mu B:\ \text{libración};\qquad E=\mu B:\ \text{separatriz};\qquad E>\mu B:\ \text{rotación}"]),
        ("Resultado general",
         r"El periodo depende de la amplitud mediante una función universal, AGM, y de los parámetros físicos solo mediante $I/\mu B$. El análisis dimensional confirma que $T_0$ es un tiempo y que la medición de $T_0$ permite determinar $\mu B=4\pi^2I/T_0^2$.",
         [r"\boxed{T(\theta_0)=\dfrac{T_0}{\mathrm{AGM}\!\left(1,\cos\frac{\theta_0}{2}\right)},\qquad T_0=2\pi\sqrt{\dfrac{I}{\mu B}}}",
          r"\left[\sqrt{\dfrac{\mathrm{kg\,m^2}}{\mathrm{J}}}\right]=\mathrm{s}\ \checkmark"]),
        ("Particularización al problema",
         rf"Con $\mu={mu:.2f}\ \mathrm{{A\,m^2}}$, $B={b*1e6:.0f}\ \mu\mathrm{{T}}$ e $I={sci(ii,2)}\ \mathrm{{kg\,m^2}}$ resulta $\omega_0^2=\mu B/I={w0**2:.4f}\ \mathrm{{s^{{-2}}}}$. Los periodos exactos para 30° y 90° se obtienen con el AGM y exceden a $T_0$ en un {100*(t30/t0-1):.3f} % y un {100*(t90/t0-1):.3f} %, respectivamente.",
         [rf"T_0=2\pi\sqrt{{\dfrac{{{sci(ii,2)}}}{{({mu:.2f})({sci(b,2)})}}}}={t0:.4f}\ \mathrm{{s}}",
          rf"T(30^\circ)={t30:.4f}\ \mathrm{{s}},\qquad T(90^\circ)={t90:.4f}\ \mathrm{{s}},\qquad \mu B={sci(mu*b,3)}\ \mathrm{{J}}"]),
        ("Errores comunes y límites de validez",
         r"Se confunde con frecuencia el periodo de pequeñas oscilaciones con el exacto: a 90° la diferencia ya es de aproximadamente 18 %. También se aplica $\tau=\mu B$ sin el seno. El modelo supone campo uniforme (en un gradiente aparece además una fuerza neta), momento $\mu$ fijo (un ferromagnético real puede reorientar sus dominios) y sin fricción; la fricción del pivote y la radiación amortiguan el movimiento y, a escala atómica, el momento angular intrínseco (espín) hace que el dipolo preceda en lugar de oscilar.",
         []),
    ]


# =============================================================================== ParamagnetCurie
def _para() -> list:
    xs = (0.1, 0.8, 1.6, 3.2)
    x1 = m12.x_param(1.0, 300.0)
    mu_b = m12.MU_B
    return [
        ("Qué se estudia y por qué importa",
         r"En un paramagneto los átomos poseen momentos magnéticos permanentes que no interactúan entre sí; el campo externo los orienta parcialmente, y la agitación térmica desordena. El resultado es una magnetización que aumenta con el campo y disminuye con la temperatura (ley de Curie). Este es el modelo base para sales paramagnéticas usadas en refrigeración por desmagnetización adiabática, para la resonancia de espín y para entender el magnetismo colectivo: el modelo de Ising y el ferromagnetismo lo extienden agregando interacciones.",
         []),
        ("Magnitudes y convenciones",
         r"Cada espín tiene momento $\pm\mu_B$ (magnetón de Bohr, $9.274\times10^{-24}\ \mathrm{J/T}$) y energía $E=\mp\mu_BB$ según esté alineado o antialineado con el campo. La temperatura $T$ está en kelvin y la constante de Boltzmann es $k=1.381\times10^{-23}\ \mathrm{J/K}$. El parámetro adimensional $x=\mu_BB/kT$ compara la energía magnética con la térmica. La magnetización por espín, normalizada con su valor de saturación, $M/M_s$, va de 0 a 1.",
         [r"x=\dfrac{\mu_BB}{kT},\qquad [\mu_BB]=\mathrm{J},\qquad \dfrac{M}{M_s}\in[0,1]"]),
        ("Intuición y analogías",
         r"Es una competencia entre orden y desorden: el campo favorece la alineación, pero la temperatura equilibra casi por igual ambas orientaciones. Si $x\ll1$ solo un pequeño exceso de espines apunta a favor y la respuesta es lineal; si $x\gg1$ casi todos están alineados y el sistema satura. Es análogo a una moneda sesgada: la probabilidad de cara es $e^x/(e^x+e^{-x})$.",
         []),
        ("Desarrollo 1: ensamble canónico y función de partición",
         r"El sistema está en contacto con un baño térmico a temperatura $T$, así que cada espín sigue la distribución canónica: la probabilidad de un microestado es proporcional al factor de Boltzmann. Como los espines son independientes, la función de partición de $N$ espines es el producto de las funciones de partición individuales, $Z_N=Z_1^N$, y todas las magnitudes extensivas son $N$ veces las de un espín.",
         [r"p(s)=\dfrac{e^{-\beta E_s}}{Z_1},\qquad E_s=-s\,\mu_BB,\ \ s=\pm1,\ \ \beta=\dfrac1{kT}",
          r"Z_1=e^{x}+e^{-x}=2\cosh x,\qquad Z_N=Z_1^N"]),
        ("Desarrollo 2: magnetización media y energía libre",
         r"La energía libre de Helmholtz es $F=-kT\ln Z_N$, y la magnetización total es la derivada de $-F$ respecto de $B$ (la variable conjugada del momento). Equivalentemente, es el promedio de $s\mu_B$ con la distribución canónica. Ambas rutas dan el mismo resultado, la tangente hiperbólica, que está acotada entre $-1$ y $1$.",
         [r"F=-NkT\ln\left(2\cosh x\right),\qquad M=-\dfrac{\partial F}{\partial B}=N\mu_B\tanh x",
          r"\langle s\rangle=\dfrac{e^{x}-e^{-x}}{e^{x}+e^{-x}}=\tanh x,\qquad \dfrac{M}{M_s}=\tanh x,\ \ M_s=N\mu_B"]),
        ("Desarrollo 3: límites, ley de Curie y susceptibilidad",
         r"Para $x\ll1$ se desarrolla $\tanh x=x-x^3/3+\dots$ y la magnetización es lineal en $B/T$, con constante de Curie $C=N\mu_0\mu_B^2/k$ en términos de la susceptibilidad volumétrica. Para $x\gg1$, $\tanh x\to1-2e^{-2x}$ y el sistema satura exponencialmente. La susceptibilidad $\chi\propto1/T$ es la ley de Curie, y deja de ser válida cuando $x\sim1$.",
         [r"x\ll1:\quad \dfrac M{M_s}=x-\dfrac{x^3}{3}+\dfrac{2x^5}{15}-\dots\simeq\dfrac{\mu_BB}{kT}",
          r"M\simeq\dfrac{N\mu_B^2}{kT}B\ \Rightarrow\ \chi=\dfrac{\mu_0 n\mu_B^2}{kT}=\dfrac{C}{T}\quad(n=N/V)",
          r"x\gg1:\quad \dfrac M{M_s}\simeq1-2e^{-2x}"]),
        ("Desarrollo 4: entropía, fluctuaciones y el algoritmo de Metropolis",
         r"La entropía se obtiene de $S=-\partial F/\partial T$ y baja de $Nk\ln2$ (alto $T$) a cero (alto campo), principio de la desmagnetización adiabática. La varianza de un espín, $1-\tanh^2x$, determina el error estadístico de un promedio sobre $N$ espines independientes. El método de Metropolis genera la distribución canónica proponiendo inversiones y aceptándolas con probabilidad $\min(1,e^{-\beta\Delta E})$; esta regla satisface el balance detallado, que garantiza la distribución estacionaria de Boltzmann.",
         [r"S=Nk\left[\ln\left(2\cosh x\right)-x\tanh x\right],\qquad \mathrm{Var}(s)=1-\tanh^2x=\dfrac1{\cosh^2x}",
          r"\dfrac{p(s\to-s)}{p(-s\to s)}=e^{-\beta\Delta E}=\dfrac{\pi(-s)}{\pi(s)}\ \Rightarrow\ \pi(s)\,W(s\to s')=\pi(s')\,W(s'\to s)",
          r"\sigma_{\bar s}=\dfrac{\sqrt{1-\tanh^2x}}{\sqrt{N_{\mathrm{ef}}}}\le\dfrac1{\sqrt{N_{\mathrm{ef}}}}"]),
        ("Resultado general",
         r"La magnetización por espín depende solo de $x=\mu_BB/kT$, es impar, está acotada y tiene los límites lineal y de saturación indicados. Para espín $J$ arbitrario la tangente hiperbólica se reemplaza por la función de Brillouin $B_J(x)$, que reduce a $\tanh x$ para $J=1/2$ y $g=2$.",
         [r"\boxed{\dfrac{M}{M_s}=\tanh\dfrac{\mu_BB}{kT},\qquad \chi=\dfrac{C}{T}\ (x\ll1)}",
          r"\dfrac{M}{M_s}=B_J(x),\quad B_J(x)=\dfrac{2J+1}{2J}\coth\dfrac{(2J+1)x}{2J}-\dfrac1{2J}\coth\dfrac{x}{2J}"]),
        ("Particularización al problema",
         rf"Con $B=1$ T y $T=300$ K, $x=\mu_BB/kT={sci(x1,4)}$: la ley de Curie es excelente porque $x^2/3={sci(x1**2/3,2)}$. Para los cuatro valores de $x$ de la escena, la solución exacta es $\tanh x$ y la aproximación lineal solo es buena en el primero.",
         [rf"x=\dfrac{{({sci(mu_b,5)})(1)}}{{({sci(K_B,5)})(300)}}={sci(x1,4)}",
          r"\dfrac{M}{M_s}=\tanh x:\quad " + r",\ \ ".join(rf"x={xv}\to{np.tanh(xv):.4f}" for xv in xs),
          r"\text{Curie }(M/M_s\simeq x):\quad " + r",\ \ ".join(rf"{xv}\to{xv:.4f}\ ({(xv/np.tanh(xv)-1)*100:+.1f}\,\%)" for xv in xs)]),
        ("Errores comunes y límites de validez",
         r"Se confunde la magnetización con el momento de un espín: $M$ es el promedio sobre el ensamble. Tampoco debe aplicarse la ley de Curie a campos fuertes o temperaturas muy bajas, donde se satura. El modelo asume espines independientes: con interacción de intercambio aparece la ley de Curie-Weiss $\chi\propto1/(T-T_c)$ y el ferromagnetismo bajo $T_c$; además, los electrones de conducción aportan un paramagnetismo de Pauli casi independiente de $T$, y el diamagnetismo orbital siempre compite.",
         []),
    ]


# =============================================================================== SeriesRLC
def _srlc() -> list:
    r, l_, c, v0 = m13.R, m13.L, m13.C, m13.V0
    w0 = m13.omega0()
    qf = m13.q_factor()
    pk = float(m13.current_amplitude_analytic(w0))
    bw = r / l_
    z125 = float(m13.impedance(1250.0))
    ph125 = float(m13.phase_analytic(1250.0))
    return [
        ("Qué se estudia y por qué importa",
         r"El circuito RLC en serie conectado a una fuente sinusoidal es el modelo de los sintonizadores de radio, los filtros pasa-banda y las líneas de transmisión resonantes: responde con intensidad solo a frecuencias cercanas a su frecuencia natural. Se busca la amplitud y la fase de la corriente para cualquier frecuencia, el ancho de la resonancia y el factor de calidad. Es el caso forzado del oscilador amortiguado.",
         []),
        ("Magnitudes y convenciones",
         r"La fuente impone $v=V_0\sin\omega t$ con $\omega$ en rad/s. La impedancia $Z$ en ohmios generaliza la resistencia: la reactancia inductiva es $X_L=\omega L$ y la capacitiva $X_C=1/\omega C$. El ángulo de fase $\phi$ es el atraso de la corriente respecto de la tensión: $i=I_0\sin(\omega t-\phi)$, con $\phi>0$ si el circuito es inductivo ($\omega>\omega_0$). La frecuencia de resonancia es $\omega_0=1/\sqrt{LC}$.",
         [r"[X_L]=[\omega L]=\mathrm{s^{-1}\cdot H}=\Omega,\qquad [X_C]=\dfrac{1}{\mathrm{s^{-1}F}}=\Omega"]),
        ("Intuición y analogías",
         r"Es un oscilador forzado: empujar un columpio con la frecuencia adecuada produce amplitudes grandes con poco esfuerzo. En resonancia, el efecto inductivo y el capacitivo se cancelan, el circuito parece una resistencia pura y la fuente solo debe compensar la disipación. Aunque la corriente sea máxima, las tensiones en $L$ y en $C$ pueden ser mucho mayores que la de la fuente, por un factor $Q$.",
         []),
        ("Desarrollo 1: ecuación diferencial del forzamiento",
         r"La ley de Kirchhoff en la malla da una ecuación lineal de segundo orden con un término forzante sinusoidal. Su solución general es la suma de la solución homogénea (transitorio, ya estudiada en el RLC libre), que decae como $e^{-Rt/2L}$, y una solución particular estacionaria con la frecuencia de la fuente. Tras un tiempo de varias constantes $2L/R$ solo sobrevive la solución estacionaria.",
         [r"L\dfrac{di}{dt}+Ri+\dfrac qC=V_0\sin\omega t,\qquad i=\dfrac{dq}{dt}",
          r"i(t)=i_{tr}(t)+i_{ss}(t),\qquad i_{tr}\propto e^{-Rt/2L}\to0"]),
        ("Desarrollo 2: método fasorial",
         r"Como la ecuación es lineal, se sustituye la fuente real por la compleja $V_0e^{i\omega t}$ (la parte imaginaria da el resultado físico) y se propone la corriente $\tilde I e^{i\omega t}$. La derivada y la integral se vuelven multiplicaciones por $i\omega$ y $1/i\omega$, de modo que la ecuación diferencial se convierte en una ecuación algebraica, $\tilde V=Z\tilde I$, con impedancia compleja. El módulo y el argumento de $Z$ dan la amplitud y la fase.",
         [r"\tilde V=V_0,\qquad \tilde I\left[R+i\omega L+\dfrac{1}{i\omega C}\right]=\tilde V\ \Rightarrow\ Z=R+i\left(\omega L-\dfrac1{\omega C}\right)",
          r"|Z|=\sqrt{R^2+\left(\omega L-\dfrac1{\omega C}\right)^2},\qquad \tan\phi=\dfrac{\omega L-1/\omega C}{R}",
          r"I_0=\dfrac{V_0}{|Z|},\qquad i_{ss}(t)=I_0\sin(\omega t-\phi)"]),
        ("Desarrollo 3: resonancia, ancho de banda y factor de calidad",
         r"La amplitud $I_0$ es máxima cuando el término reactivo se anula, es decir, cuando $\omega L=1/\omega C$: $\omega=\omega_0$, $\phi=0$ y $I_0=V_0/R$. Las frecuencias de media potencia, donde $|Z|=\sqrt2R$, satisfacen $|\omega L-1/\omega C|=R$; la diferencia entre ellas es el ancho de banda $\Delta\omega=R/L$. El cociente $\omega_0/\Delta\omega$ define el factor de calidad.",
         [r"\omega_0=\dfrac1{\sqrt{LC}},\qquad I_{max}=\dfrac{V_0}{R}",
          r"\omega_\pm=\mp\dfrac{R}{2L}+\sqrt{\dfrac{R^2}{4L^2}+\omega_0^2},\qquad \Delta\omega=\omega_+-\omega_-=\dfrac RL",
          r"Q=\dfrac{\omega_0}{\Delta\omega}=\dfrac{\omega_0L}{R}=\dfrac1R\sqrt{\dfrac LC}"]),
        ("Desarrollo 4: potencia, sobretensión y forma universal",
         r"La potencia media disipada es $\tfrac12I_0^2R$, que puede escribirse mediante la forma universal de la curva de resonancia en función del desafinamiento adimensional $\xi=Q(\omega/\omega_0-\omega_0/\omega)$. La tensión en el capacitor a resonancia es $Q$ veces la de la fuente: $V_C=I_0/\omega_0C=QV_0$. Con $Q\gg1$ la curva es una lorentziana de ancho $\Delta\omega$ alrededor de $\omega_0$.",
         [r"\bar P=\tfrac12I_0^2R=\dfrac{V_0^2}{2R}\dfrac{1}{1+\xi^2},\qquad \xi=Q\left(\dfrac\omega{\omega_0}-\dfrac{\omega_0}\omega\right)",
          r"V_C^{\,res}=\dfrac{I_{max}}{\omega_0C}=QV_0,\qquad Q\gg1:\ \ \dfrac{I_0}{I_{max}}\simeq\dfrac{1}{\sqrt{1+4Q^2(\omega-\omega_0)^2/\omega_0^2}}"]),
        ("Resultado general",
         r"La corriente estacionaria queda determinada por tres parámetros, $R$, $L$ y $C$, y por la frecuencia: amplitud $V_0/|Z|$ y fase $\arctan(X/R)$. La dimensión de $|Z|$ es de ohmio y la de $Q$ es adimensional.",
         [r"\boxed{I_0(\omega)=\dfrac{V_0}{\sqrt{R^2+\left(\omega L-\dfrac1{\omega C}\right)^2}},\qquad \phi(\omega)=\arctan\dfrac{\omega L-1/\omega C}{R}}",
          r"\boxed{\omega_0=\dfrac1{\sqrt{LC}},\qquad Q=\dfrac{\omega_0L}{R},\qquad \Delta\omega=\dfrac RL}"]),
        ("Particularización al problema",
         rf"Con $R={r:.0f}\ \Omega$, $L={l_:.2f}$ H, $C={c*1e6:.0f}\ \mu\mathrm{{F}}$ y $V_0={v0:.0f}$ V se obtienen la resonancia, la calidad y el ancho de banda. Evaluando también en $\omega=1250$ rad/s se recuperan la amplitud y la fase de la escena.",
         [rf"\omega_0=\dfrac1{{\sqrt{{({l_:.2f})({sci(c,3)})}}}}={w0:.3f}\ \mathrm{{rad/s}},\qquad Q=\dfrac{{({w0:.3f})({l_:.2f})}}{{{r:.0f}}}={qf:.3f},\qquad \Delta\omega={bw:.1f}\ \mathrm{{rad/s}}",
          rf"I_{{max}}=\dfrac{{V_0}}{{R}}=\dfrac{{{v0:.0f}}}{{{r:.0f}}}={pk:.4f}\ \mathrm{{A}},\qquad V_C^{{res}}=QV_0={qf*v0:.2f}\ \mathrm{{V}}",
          rf"\omega=1250:\ \ |Z|=\sqrt{{{r:.0f}^2+({1250*l_:.2f}-{1/(1250*c):.2f})^2}}={z125:.4f}\ \Omega,\ \ I_0={v0/z125:.4f}\ \mathrm{{A}},\ \ \phi={ph125:.5f}\ \mathrm{{rad}}"]),
        ("Errores comunes y límites de validez",
         r"Se confunde la frecuencia de resonancia de la corriente $\omega_0$ con la de máxima tensión en el capacitor, que es ligeramente menor para $Q$ pequeño. También se suma $R$, $X_L$ y $X_C$ aritméticamente en lugar de vectorialmente. Las fórmulas valen en régimen estacionario: tras conectar la fuente hay un transitorio de duración $\sim2L/R$; además, los componentes reales tienen resistencia en el inductor, capacidad parásita y pérdidas dieléctricas, que modifican $Q$ en la práctica.",
         []),
    ]


# =============================================================================== PowerTransformer
def _pow() -> list:
    vp, ip, phi = m13.V_PEAK, m13.I_PEAK, m13.PHI_DEG
    npr, nsc = m13.N_P, m13.N_S
    pa = m13.power_average_analytic()
    tr = m13.transformer_analytic()
    vr, ir = vp / np.sqrt(2), ip / np.sqrt(2)
    s_app = vr * ir
    pf = np.cos(np.deg2rad(phi))
    q_re = s_app * np.sin(np.deg2rad(phi))
    return [
        ("Qué se estudia y por qué importa",
         r"En corriente alterna, la potencia entregada a una carga fluctúa en el tiempo, y lo que importa económicamente es su promedio; el transformador permite elevar o reducir la tensión con pérdidas mínimas, lo que hace posible transportar energía a gran distancia con corrientes bajas. El análisis general define los valores eficaces, el factor de potencia y la potencia reactiva, y deduce las relaciones del transformador ideal a partir de la ley de Faraday.",
         []),
        ("Magnitudes y convenciones",
         r"La tensión y la corriente son $v=V_0\sin\omega t$ e $i=I_0\sin(\omega t-\phi)$, con amplitudes $V_0$ e $I_0$ y desfase $\phi$. Los valores eficaces ($V_{rms}=V_0/\sqrt2$, $I_{rms}=I_0/\sqrt2$) son los que dan la misma potencia en una resistencia que una señal continua. La potencia media se mide en vatios, la aparente $S=V_{rms}I_{rms}$ en voltamperios y la reactiva en var; el factor de potencia es $\cos\phi$. El transformador tiene $N_p$ vueltas en el primario y $N_s$ en el secundario.",
         [r"[VI]=\mathrm{V\cdot A}=\mathrm{W},\qquad \mathrm{pf}=\cos\phi=\dfrac{\bar P}{S}"]),
        ("Intuición y analogías",
         r"Cuando la corriente está desfasada respecto de la tensión, en parte del ciclo la carga devuelve energía a la fuente: la potencia instantánea toma valores negativos, y la potencia media es menor que el producto de eficaces. En el transformador, ambos arrollamientos comparten el mismo flujo, como dos engranajes: la relación de vueltas multiplica la tensión por un factor y divide la corriente por el mismo, de manera que la potencia se conserva, como en una palanca.",
         []),
        ("Desarrollo 1: potencia instantánea y media",
         r"El producto $vi$ se transforma con la identidad del producto de senos en un término constante más un término que oscila al doble de la frecuencia. El promedio sobre un periodo anula el término oscilante (integral de un coseno sobre dos ciclos completos) y deja la parte constante. Así, la potencia media es el producto de los eficaces por el factor de potencia.",
         [r"p(t)=v\,i=V_0I_0\sin\omega t\,\sin(\omega t-\phi)=\tfrac12V_0I_0\left[\cos\phi-\cos(2\omega t-\phi)\right]",
          r"\bar P=\dfrac1T\int_0^Tp\,dt=\tfrac12V_0I_0\cos\phi=V_{rms}I_{rms}\cos\phi"]),
        ("Desarrollo 2: valores eficaces y potencia compleja",
         r"El valor eficaz es la raíz del promedio del cuadrado, y para una sinusoide es la amplitud dividida por $\sqrt2$ porque el promedio de $\sin^2$ es $1/2$. Con fasores eficaces, la potencia compleja $\tilde S=\tilde V\tilde I^{*}$ tiene como parte real la potencia media y como parte imaginaria la reactiva, y su módulo es la aparente. El triángulo de potencias resume la relación.",
         [r"V_{rms}=\sqrt{\dfrac1T\int_0^T v^2dt}=\dfrac{V_0}{\sqrt2}",
          r"\tilde S=\tilde V\tilde I^{*}=P+iQ_r,\quad P=S\cos\phi,\ \ Q_r=S\sin\phi,\ \ S^2=P^2+Q_r^2"]),
        ("Desarrollo 3: transformador ideal desde la ley de Faraday",
         r"Dos bobinas con $N_p$ y $N_s$ vueltas comparten un núcleo de alta permeabilidad. Si el flujo por vuelta $\Phi$ es el mismo, cada tensión es $v_k=N_k\,d\Phi/dt$, y el cociente de tensiones es el de vueltas. El balance de fuerza magnetomotriz en un núcleo con permeabilidad infinita exige $N_pi_p=N_si_s$ (el flujo no puede necesitar corriente neta). Por tanto $v_pi_p=v_si_s$: la potencia pasa íntegra.",
         [r"v_p=N_p\dfrac{d\Phi}{dt},\quad v_s=N_s\dfrac{d\Phi}{dt}\ \Rightarrow\ \dfrac{v_s}{v_p}=\dfrac{N_s}{N_p}",
          r"\mathcal R\Phi=N_pi_p-N_si_s\ \xrightarrow{\ \mu\to\infty\ }\ N_pi_p=N_si_s\ \Rightarrow\ \dfrac{i_s}{i_p}=\dfrac{N_p}{N_s}",
          r"v_si_s=v_pi_p"]),
        ("Desarrollo 4: transformador real como límite y reflexión de impedancias",
         r"Con inductancias propias $L_p$, $L_s$ y mutua $M=k\sqrt{L_pL_s}$, las ecuaciones de las mallas son lineales. El transformador ideal es el límite $k\to1$, $L_p\to\infty$ con $L_s/L_p=(N_s/N_p)^2$. Una carga $Z_L$ en el secundario se ve desde el primario como $Z'=(N_p/N_s)^2Z_L$, lo que permite acoplar impedancias. La eficiencia real es menor por pérdidas en el cobre y el hierro (histéresis y corrientes de Foucault).",
         [r"v_p=L_p\dfrac{di_p}{dt}+M\dfrac{di_s}{dt},\qquad v_s=M\dfrac{di_p}{dt}+L_s\dfrac{di_s}{dt}",
          r"M=k\sqrt{L_pL_s},\quad k\to1,\quad \dfrac{L_s}{L_p}=\left(\dfrac{N_s}{N_p}\right)^2\ \Rightarrow\ \dfrac{v_s}{v_p}=\dfrac{M}{L_p}=\dfrac{N_s}{N_p}",
          r"Z'_L=\left(\dfrac{N_p}{N_s}\right)^2Z_L,\qquad \eta=\dfrac{P_s}{P_p}\le1"]),
        ("Resultado general",
         r"La potencia media es el producto de los valores eficaces y del factor de potencia; en el transformador ideal, las tensiones se escalan con la razón de vueltas y las corrientes con la razón inversa. Ambos resultados son adimensionales en su estructura y dan vatios, voltios y amperes respectivamente.",
         [r"\boxed{\bar P=\tfrac12V_0I_0\cos\phi=V_{rms}I_{rms}\cos\phi,\qquad V_{rms}=\dfrac{V_0}{\sqrt2}}",
          r"\boxed{V_s=V_p\dfrac{N_s}{N_p},\qquad I_s=I_p\dfrac{N_p}{N_s},\qquad V_sI_s=V_pI_p}"]),
        ("Particularización al problema",
         rf"Con $V_0={vp:.0f}$ V, $I_0={ip:.1f}$ A y $\phi={phi:.0f}^\circ$ se evalúan potencia media, eficaces y factor de potencia. Para el transformador se tiene $N_p={npr}$, $N_s={nsc}$, primario eficaz de 120 V y 2.0 A: el secundario reduce la tensión en un factor {npr/nsc:.0f} y eleva la corriente por el mismo factor.",
         [rf"\bar P=\tfrac12({vp:.0f})({ip:.1f})\cos{phi:.0f}^\circ={pa:.4f}\ \mathrm{{W}},\qquad \cos\phi={pf:.5f}",
          rf"V_{{rms}}=\dfrac{{{vp:.0f}}}{{\sqrt2}}={vr:.4f}\ \mathrm{{V}},\qquad I_{{rms}}=\dfrac{{{ip:.1f}}}{{\sqrt2}}={ir:.4f}\ \mathrm{{A}},\qquad S={s_app:.4f}\ \mathrm{{VA}},\ \ Q_r={q_re:.4f}\ \mathrm{{var}}",
          rf"V_s=120\cdot\dfrac{{{nsc}}}{{{npr}}}={tr['vs']:.2f}\ \mathrm{{V}},\qquad I_s=2.0\cdot\dfrac{{{npr}}}{{{nsc}}}={tr['is']:.2f}\ \mathrm{{A}},\qquad V_sI_s={tr['vs']*tr['is']:.1f}\ \mathrm{{W}}=V_pI_p={120*2.0:.1f}\ \mathrm{{W}}"]),
        ("Errores comunes y límites de validez",
         r"Es frecuente usar $V_0I_0$ como potencia media, olvidando el factor $1/2$ y $\cos\phi$, o mezclar valores pico y eficaces. Con el transformador se cree que se gana potencia al elevar tensión: la potencia se conserva y lo que se gana en tensión se pierde en corriente. El modelo ideal ignora la resistencia de los arrollamientos, el flujo de dispersión, la corriente de magnetización y la saturación del núcleo; no funciona con corriente continua, porque sin $d\Phi/dt$ no hay tensión inducida, y el calentamiento limita la potencia nominal.",
         []),
    ]


DETAIL = {
    "BiotSavartLoop": dict(concept=_loop()),
    "SolenoidAxis": dict(concept=_sol()),
    "RotatingLoop": dict(concept=_rot()),
    "SlidingRod": dict(concept=_rod()),
    "RLCircuit": dict(concept=_rl()),
    "RLCOscillator": dict(concept=_rlc()),
    "DipoleTorque": dict(concept=_dip()),
    "ParamagnetCurie": dict(concept=_para()),
    "SeriesRLC": dict(concept=_srlc()),
    "PowerTransformer": dict(concept=_pow()),
}
