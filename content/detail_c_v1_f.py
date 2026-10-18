"""Concepto y desarrollo analitico general: Volumen 1, capitulos 21 a 25.

Escenas: MuonDecay, VelocityAndEnergy, ThermalExpansion, ThermalEquilibrium, GasProcesses, OttoCycle, GasPressure,
MaxwellBoltzmann, CarnotCycle, EntropyProduction. Cada escena define `concept`: lista de (titulo, prosa, [ecuaciones TeX]).
Todas las cifras de la particularizacion se calculan con `physics/`.
"""
from __future__ import annotations

import numpy as np

from content.texfmt import fx, sci
from physics import ch21_relativity as c21
from physics import ch22_temperature as c22
from physics import ch23_first_law as c23
from physics import ch24_kinetic_theory as c24
from physics import ch25_second_law as c25


# =============================================================== MuonDecay
def _muon() -> dict:
    tau, beta, d, C = c21.TAU_MUON, c21.BETA_MUON, c21.D_MUON, c21.C
    g = c21.gamma(beta)
    v = beta * C
    tf = d / v
    life = g * tau
    p_rel, p_cl = c21.survival_analytic(), c21.survival_analytic(relativistic=False)
    dprime = d / g
    xi = d / (C * tau)
    concept = [
        ("Qué se estudia y por qué importa",
         "Los muones se crean a unos 10 km de altura cuando los rayos cósmicos chocan con la atmósfera. Su vida media en reposo es de apenas unos microsegundos, de modo que sin relatividad casi ninguno llegaría al suelo; sin embargo se detectan en abundancia. La explicación es que el tiempo propio de una partícula rápida transcurre más despacio medido desde el laboratorio. El mismo principio aparece en aceleradores de partículas, que mantienen vivos a los muones el tiempo suficiente para estudiarlos, y en la corrección de relojes de los sistemas de navegación por satélite.",
         []),
        ("Magnitudes y convenciones",
         r"Se usan la velocidad $v$ del muón en el laboratorio (m/s), el parámetro $\beta=v/c$ (adimensional, $0\le\beta<1$), el factor de Lorentz $\gamma\ge1$, la vida media propia $\tau$ (s, medida en el sistema donde el muón está en reposo) y la distancia $d$ recorrida en el laboratorio (m). Para el muón $\tau\approx2.2\ \mu\mathrm{s}$ y $c\tau\approx 659$ m; con $\beta=0.995$ aparece $\gamma\approx10$. El tiempo propio es el que marca un reloj que acompaña a la partícula y es el único que no depende del observador.",
         [r"\beta=\dfrac vc,\qquad \gamma=\dfrac1{\sqrt{1-\beta^2}},\qquad c\tau\approx 659\ \mathrm{m}"]),
        ("Intuición y analogías",
         r"Un viaje en carretera sirve de analogía: la distancia entre dos ciudades medida sobre un camino torcido es mayor que la medida en línea recta, y ambas son legítimas porque dependen del trayecto. En relatividad el «camino» es la línea de universo en el espacio-tiempo, y el tiempo propio es su longitud. Una trayectoria más rápida en el espacio es «más corta» en tiempo propio, igual que una geometría con signo opuesto en la métrica. Por eso el muón rápido envejece menos que lo que indica el reloj del suelo.",
         []),
        ("Postulados y el intervalo invariante",
         r"Los postulados son: las leyes físicas tienen la misma forma en todo sistema inercial y la velocidad de la luz en el vacío es la misma en todos ellos. Un pulso de luz emitido en el origen satisface $c^2t^2-x^2-y^2-z^2=0$ en un sistema y, por el segundo postulado, $c^2t'^2-x'^2-y'^2-z'^2=0$ en otro. Como la transformación entre sistemas inerciales debe ser lineal (el movimiento libre debe seguir siendo rectilíneo), la igualdad de los conos de luz obliga a que la forma cuadrática completa se conserve salvo un factor, que la isotropía y la reciprocidad fijan en 1. Así existe un invariante, el intervalo.",
         [r"s^2=c^2\Delta t^2-\Delta x^2-\Delta y^2-\Delta z^2,\qquad s'^2=s^2",
          r"\eta_{\mu\nu}=\mathrm{diag}(+1,-1,-1,-1),\qquad s^2=\eta_{\mu\nu}\,\Delta x^\mu\Delta x^\nu"]),
        ("Tiempo propio y dilatación del tiempo",
         r"Para un reloj que se mueve con velocidad $v(t)$, el intervalo entre dos sucesos infinitamente próximos de su línea de universo es $ds^2=c^2dt^2-v^2dt^2$. El tiempo propio se define como $d\tau=ds/c$, y es el mismo en cualquier sistema porque $ds$ es invariante. De ahí se integra a lo largo de la trayectoria; si la velocidad es constante el resultado es inmediato. Esta fórmula es también el origen del principio variacional de la partícula libre: la acción $S=-mc^2\int d\tau$ se hace extrema en las líneas rectas del espacio-tiempo.",
         [r"d\tau=\sqrt{1-\dfrac{v^2(t)}{c^2}}\,dt,\qquad \tau_{AB}=\int_A^B\sqrt{1-\dfrac{v^2}{c^2}}\,dt",
          r"v=\text{cte}\ \Rightarrow\ \Delta t=\gamma\,\Delta\tau\ \ge\ \Delta\tau",
          r"S=-mc^2\int d\tau=-mc^2\int\sqrt{1-v^2/c^2}\,dt\ \Rightarrow\ L=-mc^2\sqrt{1-v^2/c^2}"]),
        ("Transformación de Lorentz y contracción de longitudes",
         r"La transformación lineal que conserva $s^2$ y reduce a la identidad cuando $v\to0$ es el «boost». Con el sistema $S'$ moviéndose con velocidad $v$ a lo largo de $x$, se obtiene la forma matricial en las coordenadas $(ct,x)$; el parámetro $\varphi$ (rapidez) cumple $\tanh\varphi=\beta$. Dos sucesos en el mismo lugar de $S'$ (un reloj en reposo) están separados en $S$ por $\gamma$ veces su tiempo propio, y una regla en reposo en $S$ mide en $S'$ la longitud contraída. Ambas son la misma propiedad de la matriz vista desde lados distintos.",
         [r"\begin{pmatrix}ct\\x\end{pmatrix}=\begin{pmatrix}\cosh\varphi&\sinh\varphi\\ \sinh\varphi&\cosh\varphi\end{pmatrix}\begin{pmatrix}ct'\\x'\end{pmatrix},\qquad \tanh\varphi=\beta,\quad \cosh\varphi=\gamma",
          r"\Delta t=\gamma\left(\Delta t'+\dfrac{v\,\Delta x'}{c^2}\right),\qquad \Delta x=\gamma(\Delta x'+v\,\Delta t')",
          r"\Delta x'=0\ \Rightarrow\ \Delta t=\gamma\Delta t'\ ;\qquad \ell=\dfrac{\ell_0}{\gamma}\ \text{(longitud medida con ambos extremos a la vez)}"]),
        ("Decaimiento exponencial en el sistema propio",
         r"La desintegración es un proceso aleatorio sin memoria: la probabilidad de decaer en un intervalo $d\tau$ es $d\tau/\tau$, independiente de la edad. Si $N$ es el número de muones que sobreviven, $dN/d\tau=-N/\tau$, y la solución es una exponencial en el tiempo propio. La probabilidad de supervivencia se obtiene dividiendo entre $N_0$. Esta ecuación está formulada en el sistema propio, el único donde $\tau$ es la vida media de un reloj en reposo, y se traslada al laboratorio con la relación del apartado anterior.",
         [r"\dfrac{dN}{d\tau}=-\dfrac N\tau\ \Rightarrow\ N(\tau)=N_0\,e^{-\tau/\tau_0},\qquad P(\tau)=e^{-\tau/\tau_0}",
          r"f(\tau)\,d\tau=\dfrac1{\tau_0}e^{-\tau/\tau_0}d\tau,\qquad \langle\tau\rangle=\tau_0"]),
        ("Supervivencia en el laboratorio: dos puntos de vista equivalentes",
         r"En el laboratorio el muón recorre la distancia $d$ con velocidad $v$ en un tiempo $t=d/v$, al que corresponde un tiempo propio $\tau=t/\gamma$. Sustituyendo en la exponencial aparece la vida media dilatada $\gamma\tau_0$. En el sistema del muón el análisis es distinto pero da el mismo número: la atmósfera se mueve hacia él con velocidad $v$ y su espesor está contraído a $d/\gamma$, que se recorre en un tiempo $d/(\gamma v)$ del reloj propio. Que ambos resultados coincidan es una comprobación de la consistencia de la teoría, no de dos efectos distintos.",
         [r"P=\exp\!\left(-\dfrac{t}{\gamma\tau_0}\right)=\exp\!\left(-\dfrac{d}{\gamma v\tau_0}\right)",
          r"\text{muón: }\ \tau_{\mathrm{vuelo}}=\dfrac{d/\gamma}{v}\ \Rightarrow\ P=\exp\!\left(-\dfrac{d}{\gamma v\tau_0}\right)",
          r"\gamma\beta=\dfrac{p}{mc}\ \Rightarrow\ P=\exp\!\left(-\dfrac{d\,mc}{p\,c\,\tau_0}\right)"]),
        ("Resultado general",
         r"La supervivencia depende de un único grupo adimensional, la distancia recorrida medida en unidades del recorrido característico $\gamma\beta c\tau_0$. En el límite no relativista ($\beta\ll1$, $\gamma\to1$) se recupera la exponencial clásica $e^{-d/(v\tau_0)}$; en el límite ultrarrelativista el recorrido característico crece como $\gamma c\tau_0$ y la probabilidad tiende a 1 con $\gamma$. El exponente es adimensional: $d$ en metros, $v\tau_0$ en metros.",
         [r"\boxed{\,P(d)=\exp\!\left(-\dfrac{d}{\gamma\beta\,c\,\tau_0}\right),\qquad \gamma\beta=\dfrac{\beta}{\sqrt{1-\beta^2}}\,}",
          r"\lambda=\gamma\beta c\tau_0,\qquad \xi\equiv\dfrac{d}{c\tau_0},\qquad P=e^{-\xi/(\gamma\beta)}",
          r"[\,d/(\gamma\beta c\tau_0)\,]=\dfrac{\mathrm{m}}{(\mathrm{m/s})\,\mathrm{s}}=1"]),
        ("Particularización al problema",
         rf"Con $\beta={beta}$, $\tau_0={sci(tau,4)}\ \mathrm{{s}}$ y $d={sci(d,3)}\ \mathrm{{m}}$ se tiene $\gamma={g:.4f}$ y $c\tau_0={sci(C*tau,4)}\ \mathrm{{m}}$. El recorrido característico relativista resulta casi diez veces el clásico, y la probabilidad pasa de insignificante a cercana a un quinto. Es el mismo resultado analítico de la escena.",
         [rf"\xi=\dfrac{{d}}{{c\tau_0}}={fx(xi,3)},\qquad \gamma\beta={fx(g*beta,4)},\qquad \lambda=\gamma\beta c\tau_0={fx(g*beta*C*tau,1)}\ \mathrm{{m}}",
          rf"P_{{rel}}=\exp\!\left(-\dfrac{{{sci(d,3)}}}{{{fx(g*beta*C*tau,1)}}}\right)={p_rel:.4f}",
          rf"P_{{cl}}=\exp\!\left(-\dfrac{{{sci(d,3)}}}{{{fx(beta*C*tau,1)}}}\right)={sci(p_cl,3)}",
          rf"\text{{(sistema del muón) }}d/\gamma={fx(dprime,1)}\ \mathrm{{m}},\quad t_{{vuelo}}={sci(tf/g,4)}\ \mathrm{{s}}\ \Rightarrow\ P={p_rel:.4f}"]),
        ("Errores comunes y límites de validez",
         r"Es frecuente pensar que la dilatación del tiempo es una ilusión óptica o que ambos observadores ven el reloj del otro retrasarse y por tanto hay una contradicción: la simetría desaparece al comparar relojes que se reúnen, porque entonces uno de ellos ha cambiado de sistema inercial. Otro error es aplicar la dilatación y la contracción de longitudes a la vez en el mismo sistema, lo que cuenta dos veces el efecto. También se confunde la vida media con el tiempo de vida de cada muón, que es aleatorio. El modelo supone velocidad constante y atmósfera de densidad uniforme; los muones reales pierden energía por ionización y se producen a alturas variables, y la relatividad general sólo importa para campos gravitatorios intensos.",
         []),
    ]
    return dict(concept=concept)


# =============================================================== VelocityAndEnergy
def _velocity() -> dict:
    C = c21.C
    u = v = 0.8
    up = c21.velocity_addition_analytic(u, v)
    phi_u = np.arctanh(u)
    b = 0.9
    g = c21.gamma(b)
    kr = c21.kinetic_analytic(b) / C**2
    kc = c21.kinetic_classical(b) / C**2
    concept = [
        ("Qué se estudia y por qué importa",
         "En la mecánica clásica las velocidades se suman: si una nave se mueve respecto de la Tierra y lanza un proyectil, la velocidad del proyectil respecto de la Tierra es la suma. Eso permitiría superar la velocidad de la luz con la sola repetición del lanzamiento, y es incompatible con el hecho de que ninguna señal supera $c$. La relatividad corrige la regla de composición y, al mismo tiempo, la energía cinética, que crece sin límite cuando la rapidez se aproxima a $c$. Son resultados indispensables en el diseño de aceleradores, en astrofísica de chorros relativistas y en la interpretación de reacciones nucleares.",
         []),
        ("Magnitudes y convenciones",
         r"Se tratan velocidades en una sola dirección expresadas en unidades de $c$: $u$ es la velocidad del proyectil respecto de la nave, $v$ la de la nave respecto de la Tierra y $u'$ la del proyectil respecto de la Tierra. La energía cinética $K$ se mide en julios, o en unidades de la energía en reposo $mc^2$ (por ejemplo $0.511\ \mathrm{MeV}$ para el electrón y $938\ \mathrm{MeV}$ para el protón). El momento es $p=\gamma mv$ y la energía total es $E=\gamma mc^2$. Convención: velocidad positiva a lo largo de $+x$.",
         [r"u,\,v,\,u'\in(-c,c),\qquad E=\gamma mc^2,\qquad K=E-mc^2"]),
        ("Intuición y analogías",
         r"La suma de velocidades se parece a la suma de ángulos en lugar de a la suma de longitudes. Cada boost es una rotación hiperbólica en el plano $(ct,x)$, y lo que se suma al componer boosts es el ángulo hiperbólico (la rapidez), no la pendiente de la línea de universo. Como la tangente hiperbólica está acotada por 1, ningún número de sumas lleva a superar $c$. La energía se comporta como la hipotenusa de un triángulo: al aproximarse a $c$ casi toda la energía añadida va a aumentar la inercia, no la rapidez.",
         []),
        ("Los boosts como grupo de rotaciones hiperbólicas",
         r"La transformación que conserva el intervalo en el plano $(ct,x)$ y es continua con la identidad puede escribirse con un parámetro $\varphi$. Las matrices $\Lambda(\varphi)$ forman un grupo abeliano uniparamétrico, generado por una matriz $K$ que cumple $K^2=1$. De ahí $\Lambda(\varphi)=e^{\varphi K}$ y, en consecuencia, $\Lambda(\varphi_1)\Lambda(\varphi_2)=\Lambda(\varphi_1+\varphi_2)$. La velocidad relativa asociada es $\beta=\tanh\varphi$.",
         [r"\Lambda(\varphi)=\begin{pmatrix}\cosh\varphi&\sinh\varphi\\ \sinh\varphi&\cosh\varphi\end{pmatrix}=e^{\varphi K},\qquad K=\begin{pmatrix}0&1\\1&0\end{pmatrix},\ K^2=\mathbb 1",
          r"\Lambda^{T}\eta\,\Lambda=\eta,\qquad \eta=\mathrm{diag}(1,-1)"]),
        ("Adición de velocidades",
         r"Si el proyectil tiene rapidez $\varphi_u$ respecto de la nave y la nave $\varphi_v$ respecto de la Tierra, la matriz compuesta es el boost de rapidez $\varphi_u+\varphi_v$. La velocidad respecto de la Tierra es la tangente hiperbólica de esa suma, y la fórmula de adición de la tangente hiperbólica da la ley relativista. Es posible justificar el resultado también leyendo directamente el cociente de elementos de la matriz producto. En el límite $uv\ll c^2$ se recupera la suma galileana.",
         [r"\Lambda(\varphi_v)\Lambda(\varphi_u)=\Lambda(\varphi_u+\varphi_v)\ \Rightarrow\ u'=c\tanh(\varphi_u+\varphi_v)",
          r"\tanh(a+b)=\dfrac{\tanh a+\tanh b}{1+\tanh a\tanh b}\ \Rightarrow\ u'=\dfrac{u+v}{1+uv/c^2}",
          r"1-\dfrac{u'^2}{c^2}=\dfrac{(1-u^2/c^2)(1-v^2/c^2)}{(1+uv/c^2)^2}>0\ \Rightarrow\ |u'|<c"]),
        ("Cuadrivelocidad, cuadrimomento y energía",
         r"Con el tiempo propio $d\tau$ se define la cuadrivelocidad $u^\mu=dx^\mu/d\tau=\gamma(c,\vec v)$, cuyo módulo cuadrado es $c^2$ para cualquier trayectoria. El cuadrimomento $p^\mu=mu^\mu$ hereda esa invariancia: $p^\mu p_\mu=m^2c^2$. Identificando su componente temporal con $E/c$ (la que se conserva por invarianza traslacional en el tiempo) se llega a la relación entre energía, momento y masa. Los tres límites básicos son masa nula (fotón, $E=pc$), reposo ($E=mc^2$) y rapidez pequeña.",
         [r"u^\mu=\gamma(c,\vec v),\qquad u^\mu u_\mu=c^2",
          r"p^\mu=(E/c,\vec p),\qquad p^\mu p_\mu=\dfrac{E^2}{c^2}-p^2=m^2c^2",
          r"E^2=(pc)^2+(mc^2)^2,\qquad \vec p=\dfrac{E}{c^2}\vec v"]),
        ("Lagrangiana relativista y energía cinética",
         r"Del principio variacional $S=-mc^2\int d\tau$ resulta $L=-mc^2\sqrt{1-v^2/c^2}$. El momento canónico $p=\partial L/\partial v$ es $\gamma mv$ y el hamiltoniano $H=pv-L$ es la energía total $\gamma mc^2$. Al no depender $L$ explícitamente del tiempo, $H$ se conserva (teorema de Noether). El trabajo que la fuerza hace sobre la partícula es $dK=v\,dp$, y la integral desde el reposo da la energía cinética, que se compara con el primer término del desarrollo en potencias de $\beta^2$.",
         [r"p=\dfrac{\partial L}{\partial v}=\gamma mv,\qquad H=pv-L=\gamma mc^2",
          r"K=\int_0^{v}v'\,dp=\int_0^{v}v'\,d(\gamma mv')=\gamma mv^2-\int_0^v\gamma mv'\,dv'=(\gamma-1)mc^2",
          r"K=mc^2\left(\dfrac{\beta^2}{2}+\dfrac{3\beta^4}{8}+\dfrac{5\beta^6}{16}+\dots\right)"]),
        ("Límites y parámetros adimensionales",
         r"El único parámetro adimensional es $\beta$. Para $\beta\ll1$ el primer término del desarrollo reproduce $\tfrac12mv^2$ y la primera corrección relativa es $\tfrac34\beta^2$. Cuando $\beta\to1$, $\gamma$ diverge y se necesita energía infinita para alcanzar $c$: $K\approx E$ y $p\approx E/c$. Para la adición de velocidades, si $u=c$ el resultado es $u'=c$ para cualquier $v$, que es el segundo postulado; si $u,v\ll c$ el denominador vale 1.",
         [r"\dfrac{K_{rel}-K_{cl}}{K_{cl}}\simeq\dfrac34\beta^2,\qquad \beta\ll1",
          r"u=c\ \Rightarrow\ u'=\dfrac{c+v}{1+v/c}=c"]),
        ("Resultado general",
         r"Las dos fórmulas que resumen el análisis son la adición de velocidades colineales y la energía cinética en función de la rapidez. Ambas son homogéneas en las unidades: el cociente $uv/c^2$ es adimensional y $K$ escala con $mc^2$. En términos de rapideces, la adición es lineal, lo que explica que la composición de $n$ boosts iguales conduzca a $\tanh(n\varphi)$.",
         [r"\boxed{\,u'=\dfrac{u+v}{1+uv/c^2}=c\tanh\!\left(\operatorname{artanh}\tfrac uc+\operatorname{artanh}\tfrac vc\right)\,}",
          r"\boxed{\,K=(\gamma-1)\,mc^2=mc^2\left(\dfrac1{\sqrt{1-\beta^2}}-1\right)\,}",
          r"[K]=\mathrm{kg}\cdot\mathrm{m^2\,s^{-2}}=\mathrm{J}"]),
        ("Particularización al problema",
         rf"Para $u=v=0.8c$ las rapideces valen $\varphi={fx(phi_u,5)}$ cada una, y su suma da $\tanh({fx(2*phi_u,5)})$. A $0.9c$ el factor de Lorentz es $\gamma={g:.4f}$. Las cifras coinciden con las de la solución analítica de la escena.",
         [rf"u'=\dfrac{{0.8+0.8}}{{1+0.8\cdot0.8}}=\dfrac{{1.6}}{{{fx(1+u*v,2)}}}={up:.5f}\,c",
          rf"u'=\tanh(2\cdot{fx(phi_u,5)})=\tanh({fx(2*phi_u,5)})={np.tanh(2*phi_u):.5f}\,c",
          rf"K_{{rel}}=(\gamma-1)mc^2=({g:.4f}-1)\,mc^2={kr:.4f}\,mc^2",
          rf"K_{{cl}}=\tfrac12\beta^2mc^2={kc:.4f}\,mc^2,\qquad \dfrac{{K_{{rel}}}}{{K_{{cl}}}}={kr/kc:.3f}"]),
        ("Errores comunes y límites de validez",
         r"Un error habitual es sumar velocidades en sentido contrario como si bastara cambiar el signo de $v$: la fórmula es válida con signo, pero la ley no es simétrica en los resultados si las velocidades no son colineales, donde se necesita la versión vectorial con la rotación de Wigner-Thomas. Otro es usar $K=\tfrac12\gamma mv^2$ o $K=\gamma mc^2$: lo primero no es el trabajo y lo segundo es la energía total. También se confunde «masa relativista» con una masa que cambia; aquí $m$ es invariante. Las fórmulas valen en espacio-tiempo plano; en presencia de gravedad intensa hay que sustituir los boosts por la geometría local.",
         []),
    ]
    return dict(concept=concept)


# =============================================================== ThermalExpansion
def _expansion() -> dict:
    L0, a0, bt, t0, t1 = c22.L0, c22.ALPHA0, c22.BETA, c22.T0, c22.T1
    dT = t1 - t0
    lin = c22.length_linear() - L0
    ex = c22.length_analytic() - L0
    a1 = float(c22.alpha(t1))
    ab = a0 * (1 + bt * dT / 2)
    concept = [
        ("Qué se estudia y por qué importa",
         "Casi todos los sólidos se alargan al calentarse porque la separación media entre átomos aumenta con la agitación térmica. En estructuras largas, como rieles, puentes, tuberías y líneas eléctricas, esa variación debe absorberse con juntas de dilatación para evitar esfuerzos que deforman o rompen la pieza. Bimetales y termómetros aprovechan el efecto. El coeficiente de dilatación casi nunca es exactamente constante, y para intervalos grandes de temperatura el error de usar un valor constante se acumula.",
         []),
        ("Magnitudes y convenciones",
         r"La longitud $L(T)$ se mide en metros, la temperatura $T$ en kelvin o grados Celsius (sólo importan diferencias $\Delta T=T-T_0$), y el coeficiente de dilatación lineal $\alpha$ en $\mathrm{K^{-1}}$. En metales vale típicamente $10^{-5}$ a $3\times10^{-5}\ \mathrm{K^{-1}}$; el acero ronda $12\times10^{-6}\ \mathrm{K^{-1}}$. El parámetro $\beta$ de la variación $\alpha(T)=\alpha_0[1+\beta(T-T_0)]$ tiene unidades de $\mathrm{K^{-1}}$. Una dilatación positiva significa un aumento de longitud.",
         [r"[\alpha]=\mathrm{K^{-1}},\qquad [\beta]=\mathrm{K^{-1}},\qquad [\Delta L]=\mathrm{m}"]),
        ("Intuición y analogías",
         r"Se puede imaginar la red cristalina como bolas unidas por resortes. Si el resorte fuera perfectamente armónico, la bola oscilaría simétricamente y su posición media no cambiaría al aumentar la energía; la dilatación existe porque la pared de energía potencial es más empinada hacia dentro que hacia fuera. Cuanto más energía, más se asoma la bola hacia el lado suave. De ahí que la dilatación sea un efecto anarmónico y que el coeficiente cambie con la temperatura.",
         []),
        ("Origen microscópico: potencial anarmónico",
         r"Para un par de átomos se modela la energía potencial cerca del mínimo como $U(x)=cx^2-gx^3$, con $x$ el desplazamiento respecto de la posición de equilibrio. Con la distribución de Boltzmann $e^{-U/kT}$ el promedio de $x$ se calcula desarrollando el término cúbico, que es pequeño. Solo sobrevive el término impar acoplado con $x^4$, y el resultado es lineal en $T$. Si $\langle x\rangle$ crece como $T$, la longitud de la barra crece proporcionalmente y $\alpha=\dfrac1L\dfrac{d\langle x\rangle}{dT}$.",
         [r"\langle x\rangle=\dfrac{\int x\,e^{-U/kT}dx}{\int e^{-U/kT}dx}\simeq\dfrac{3g}{4c^2}\,kT",
          r"\alpha=\dfrac1{r_0}\dfrac{d\langle x\rangle}{dT}=\dfrac{3gk}{4c^2r_0}"]),
        ("Definición local de α y ecuación diferencial",
         r"El coeficiente de dilatación lineal se define por la variación relativa de longitud por unidad de temperatura, y la definición precisa es local: debe referirse a un $\Delta T$ infinitesimal porque $\alpha$ puede depender de $T$. Al pasar al límite se obtiene una ecuación diferencial de variables separables para $L(T)$ cuya solución general es una exponencial de la integral de $\alpha$. Todo esto supone que la barra es homogénea y está en equilibrio térmico, de modo que $L$ sólo depende de $T$ y no de la rapidez de calentamiento.",
         [r"\alpha(T)\equiv\dfrac1L\dfrac{dL}{dT}\ \Rightarrow\ \dfrac{dL}{dT}=\alpha(T)\,L",
          r"\dfrac{dL}{L}=\alpha(T')\,dT'\ \Rightarrow\ \ln\dfrac{L}{L_0}=\int_{T_0}^{T}\alpha(T')\,dT'"]),
        ("Integración con α(T) general y con α lineal",
         r"La solución exacta para cualquier $\alpha(T)$ es $L=L_0\exp\int\alpha\,dT'$. Para el modelo $\alpha=\alpha_0[1+\beta(T'-T_0)]$ la integral se calcula con el cambio $s=T'-T_0$ y resulta un polinomio cuadrático en $\Delta T$. Si además $\alpha_0\Delta T\ll1$, la exponencial se desarrolla en serie y la primera aproximación es la fórmula lineal $L_0(1+\bar\alpha\,\Delta T)$, con $\bar\alpha$ el coeficiente promedio en el intervalo.",
         [r"\ln\dfrac{L}{L_0}=\alpha_0\int_0^{\Delta T}(1+\beta s)\,ds=\alpha_0\left(\Delta T+\dfrac{\beta\,\Delta T^2}{2}\right)",
          r"L=L_0\exp\!\left[\alpha_0\left(\Delta T+\dfrac{\beta\,\Delta T^2}{2}\right)\right]",
          r"\bar\alpha=\dfrac1{\Delta T}\int_0^{\Delta T}\alpha\,ds=\alpha_0\left(1+\dfrac{\beta\Delta T}{2}\right)"]),
        ("Desarrollo en serie y parámetros adimensionales",
         r"Todo el efecto depende de dos números adimensionales: $\varepsilon=\alpha_0\Delta T$, la deformación térmica lineal, y $\beta\Delta T$, la variación relativa del coeficiente en el intervalo. Al desarrollar la exponencial se obtiene $\Delta L/L_0$ como serie de $\varepsilon$ con correcciones en $\beta\Delta T$ y $\varepsilon^2$. Como $\varepsilon$ es del orden de $10^{-3}$ en la práctica, el término cuadrático en $\varepsilon$ es despreciable y la corrección dominante respecto de la ley lineal proviene de $\beta$.",
         [r"\dfrac{\Delta L}{L_0}=\varepsilon\left(1+\dfrac{\beta\Delta T}{2}\right)+\dfrac{\varepsilon^2}{2}\left(1+\dfrac{\beta\Delta T}{2}\right)^2+\dots",
          r"\dfrac{\Delta L_{\text{exacto}}-\Delta L_{\text{lineal}}}{\Delta L_{\text{lineal}}}\simeq\dfrac{\beta\Delta T}{2}"]),
        ("Extensión a área, volumen y esfuerzo térmico",
         r"Para un sólido isótropo cada dimensión lineal se dilata igual, de modo que las deformaciones térmicas forman el tensor $\varepsilon_{ij}=\alpha\Delta T\,\delta_{ij}$. El área y el volumen cambian entonces con $2\alpha$ y $3\alpha$ a primer orden; el coeficiente de volumen es la traza del tensor. Si la barra está empotrada y no puede dilatarse, aparece un esfuerzo $\sigma=-E\alpha\Delta T$ con $E$ el módulo de Young, que explica las juntas de dilatación. Estas extensiones usan el mismo $\alpha(T)$ en la integral.",
         [r"\dfrac{\Delta A}{A_0}\simeq2\alpha\Delta T,\qquad \dfrac{\Delta V}{V_0}\simeq3\alpha\Delta T=\mathrm{tr}(\varepsilon_{ij})",
          r"\sigma=-E\int_{T_0}^{T}\alpha(T')\,dT'"]),
        ("Resultado general",
         r"La longitud y la dilatación absoluta para un coeficiente que varía linealmente con la temperatura se escriben en una sola fórmula. Recuperan la expresión clásica en el límite $\beta\to0$, $\alpha_0\Delta T\to0$, y las unidades son consistentes porque $\alpha_0\Delta T$ y $\beta\Delta T$ son adimensionales.",
         [r"\boxed{\,L(T)=L_0\exp\!\left[\alpha_0\left(\Delta T+\tfrac12\beta\,\Delta T^2\right)\right],\qquad \Delta L=L(T)-L_0\,}",
          r"\beta\to0:\ \ L=L_0e^{\alpha_0\Delta T}\simeq L_0(1+\alpha_0\Delta T)",
          r"[\alpha_0\Delta T]=\mathrm{K^{-1}\cdot K}=1,\qquad [\beta\Delta T]=1"]),
        ("Particularización al problema",
         rf"Con $L_0={fx(L0,0)}\ \mathrm{{m}}$, $\alpha_0={sci(a0,3)}\ \mathrm{{K^{{-1}}}}$, $\beta={bt}\ \mathrm{{K^{{-1}}}}$ y $\Delta T={fx(dT,0)}\ \mathrm{{K}}$, el exponente vale $\alpha_0(\Delta T+\beta\Delta T^2/2)={sci(a0*(dT+bt*dT**2/2),5)}$. El coeficiente crece de $\alpha_0$ a ${sci(a1,4)}\ \mathrm{{K^{{-1}}}}$ en el extremo caliente, y su promedio es ${sci(ab,4)}\ \mathrm{{K^{{-1}}}}$. Estas cifras reproducen las de la escena.",
         [rf"\Delta L_{{lineal}}=L_0\alpha_0\Delta T=({fx(L0,0)})({sci(a0,3)})({fx(dT,0)})={lin*1000:.2f}\ \mathrm{{mm}}",
          rf"\Delta L_{{exacto}}=L_0\left(e^{{{sci(a0*(dT+bt*dT**2/2),5)}}}-1\right)={ex*1000:.2f}\ \mathrm{{mm}}",
          rf"\Delta L_{{exacto}}-\Delta L_{{lineal}}={(ex-lin)*1000:.2f}\ \mathrm{{mm}}\quad\left(\text{{relativo}}\approx\dfrac{{\beta\Delta T}}{{2}}={fx(bt*dT/2,2)}\right)"]),
        ("Errores comunes y límites de validez",
         r"Es común aplicar $\Delta L=\alpha L_0\Delta T$ en intervalos amplios, o tomar el $\alpha$ de una tabla medida a otra temperatura; aquí con $\beta\Delta T=0.2$ el error supera 10 %. Otra confusión es emplear $\alpha$ de volumen en una fórmula lineal (hay un factor 3). Los grados Celsius sirven para $\Delta T$, pero no para fórmulas con $T$ absoluta. El modelo no incluye cambios de fase, anisotropía de cristales (el grafito, por ejemplo, tiene $\alpha$ distinto por eje) ni dilatación negativa (agua entre 0 y 4 °C), donde $\alpha(T)$ cambia de signo y la ecuación diferencial debe resolverse con el $\alpha(T)$ medido.",
         []),
    ]
    return dict(concept=concept)


# =============================================================== ThermalEquilibrium
def _equilibrium() -> dict:
    m1, c1, t10 = c22.M1, c22.C1, c22.T1_0
    m2, c2, t20 = c22.M2, c22.C2, c22.T2_0
    h = c22.H
    C1, C2 = m1 * c1, m2 * c2
    tf = c22.final_temperature_analytic()
    lam = c22.rate_analytic()
    tau = 1 / lam
    cred = C1 * C2 / (C1 + C2)
    t20s = float(c22.temperatures_analytic(20.0)[0])
    t2_20 = float(c22.temperatures_analytic(20.0)[1])
    concept = [
        ("Qué se estudia y por qué importa",
         "Cuando dos cuerpos a distinta temperatura se ponen en contacto, el más caliente cede energía al más frío hasta que ambos alcanzan una temperatura común. Este proceso es la base de la calorimetría, del enfriamiento de componentes electrónicos, de la inercia térmica de edificios y de todo diseño de intercambiadores de calor. Lo que interesa saber es cuál es la temperatura final y qué tan rápido se alcanza.",
         []),
        ("Magnitudes y convenciones",
         r"Cada cuerpo se caracteriza por su capacidad calorífica $C_i=m_ic_i$ en $\mathrm{J/K}$, con $m_i$ la masa y $c_i$ el calor específico. La conductancia térmica $h$ (W/K) mide cuánta potencia fluye por kelvin de diferencia entre ambos. La rapidez de variación es una tasa $\lambda$ en $\mathrm{s^{-1}}$ y la constante de tiempo es $\tau=1/\lambda$. Las temperaturas pueden expresarse en grados Celsius mientras sólo aparezcan diferencias o medias ponderadas, pero deben ir en kelvin dentro de logaritmos.",
         [r"[C_i]=\mathrm{J\,K^{-1}},\qquad [h]=\mathrm{W\,K^{-1}},\qquad [\lambda]=\dfrac{[h]}{[C]}=\mathrm{s^{-1}}"]),
        ("Intuición y analogías",
         r"Es el equivalente térmico de dos depósitos de agua unidos por un tubo: el caudal depende de la diferencia de nivel y los niveles se igualan en un valor intermedio que depende de las áreas de los depósitos, es decir, de sus capacidades. También se parece a un circuito $RC$: la diferencia de temperatura decae exponencialmente con una constante de tiempo que depende de la «resistencia» $1/h$ y de la «capacidad» efectiva, que es menor que las dos reales. El cuerpo con menor capacidad cambia más su temperatura.",
         []),
        ("Ley cero y existencia de la temperatura",
         r"La ley cero postula que si $A$ está en equilibrio térmico con $B$ y $B$ con $C$, entonces $A$ lo está con $C$. Esa transitividad convierte el equilibrio en una relación de equivalencia en el conjunto de estados, y las clases de equivalencia pueden etiquetarse con un número: la temperatura empírica. Sin esta propiedad no habría termómetros, ya que dos lecturas iguales de un mismo termómetro no implicarían equilibrio entre los cuerpos. La temperatura es una función de estado que determina la dirección del flujo de calor.",
         [r"A\sim B,\ B\sim C\ \Rightarrow\ A\sim C,\qquad \theta_A=\theta_B\iff A\sim B"]),
        ("Balance de energía y ley de conducción",
         r"Con el sistema aislado del exterior, el calor que pierde un cuerpo lo gana el otro. Si el flujo entre ellos es proporcional a la diferencia de temperatura (ley de Fourier en su forma de parámetros concentrados, válida cuando la temperatura de cada cuerpo es casi uniforme), la primera ley aplicada a cada cuerpo da un sistema de ecuaciones diferenciales lineales acopladas. La suma de ambas ecuaciones demuestra que la energía total se conserva, y por ello $C_1T_1+C_2T_2$ es una integral primera.",
         [r"C_1\dfrac{dT_1}{dt}=-h\,(T_1-T_2),\qquad C_2\dfrac{dT_2}{dt}=+h\,(T_1-T_2)",
          r"\dfrac{d}{dt}\big(C_1T_1+C_2T_2\big)=0\ \Rightarrow\ C_1T_1+C_2T_2=C_1T_{10}+C_2T_{20}"]),
        ("Reducción a un grado de libertad",
         r"Con la integral primera el sistema de dos variables se reduce a una sola ecuación, para la diferencia $\Delta=T_1-T_2$. Se resta la segunda ecuación (dividida entre $C_2$) de la primera (dividida entre $C_1$), y aparece la capacidad reducida $C_r=C_1C_2/(C_1+C_2)$, igual que la masa reducida en un problema de dos cuerpos. La ecuación resultante es lineal de primer orden y su solución es una exponencial decreciente.",
         [r"\dfrac{d\Delta}{dt}=-h\left(\dfrac1{C_1}+\dfrac1{C_2}\right)\Delta=-\dfrac{h}{C_r}\,\Delta,\qquad C_r=\dfrac{C_1C_2}{C_1+C_2}",
          r"\Delta(t)=\Delta_0\,e^{-\lambda t},\qquad \lambda=\dfrac h{C_r},\quad \tau=\dfrac1\lambda=\dfrac{C_r}{h}"]),
        ("Solución para cada temperatura y modos normales",
         r"Al combinar $\Delta(t)$ con la integral primera se despeja cada temperatura. La temperatura de equilibrio $T_f$ es la media de las temperaturas iniciales ponderada por las capacidades. En forma matricial $\dot{\vec T}=-A\vec T$, la matriz $A$ tiene un autovalor nulo, asociado al modo uniforme (equilibrio), y otro $\lambda$, asociado al modo de diferencia que decae. Esto confirma que la relajación es monótona, sin oscilaciones: el sistema es sobreamortiguado.",
         [r"T_f=\dfrac{C_1T_{10}+C_2T_{20}}{C_1+C_2}",
          r"T_1(t)=T_f+\dfrac{C_2}{C_1+C_2}\,\Delta_0\,e^{-\lambda t},\qquad T_2(t)=T_f-\dfrac{C_1}{C_1+C_2}\,\Delta_0\,e^{-\lambda t}",
          r"A=h\begin{pmatrix}1/C_1&-1/C_1\\-1/C_2&1/C_2\end{pmatrix},\qquad \mathrm{spec}(A)=\{0,\ \lambda\}"]),
        ("Generalización a N cuerpos",
         r"Para $N$ cuerpos con conductancias $h_{ij}=h_{ji}$ entre pares, el sistema es $C_i\dot T_i=\sum_jh_{ij}(T_j-T_i)$, es decir $\mathbf C\dot{\vec T}=-\mathbf L\vec T$ con $\mathbf L$ el laplaciano ponderado del grafo de conexiones, simétrico y semidefinido positivo. Por ello los autovalores de $\mathbf C^{-1/2}\mathbf L\,\mathbf C^{-1/2}$ son reales y no negativos: el sistema relaja sin oscilar, y si el grafo es conexo el único modo no amortiguado es el uniforme. La temperatura final es el promedio ponderado con las capacidades y el tiempo de relajación lo fija el menor autovalor positivo.",
         [r"\mathbf L_{ij}=-h_{ij}\ (i\ne j),\qquad \mathbf L_{ii}=\sum_{j\ne i}h_{ij},\qquad \vec{1}^{T}\mathbf L=0",
          r"T_f=\dfrac{\sum_iC_iT_{i0}}{\sum_iC_i},\qquad \tau=\dfrac1{\lambda_{\min}^{(+)}}"]),
        ("Resultado general",
         r"La temperatura final no depende de $h$ (que sólo fija la rapidez), y la constante de tiempo no depende de las temperaturas iniciales. Si una de las capacidades es mucho mayor que la otra, $T_f$ coincide con la temperatura del cuerpo grande (un termostato o foco térmico) y $\lambda\to h/C_{pequeña}$. Dimensionalmente $\lambda=h/C_r$ tiene unidades de $\mathrm{s^{-1}}$.",
         [r"\boxed{\,T_f=\dfrac{C_1T_{10}+C_2T_{20}}{C_1+C_2},\qquad \lambda=h\left(\dfrac1{C_1}+\dfrac1{C_2}\right),\qquad \tau=\dfrac1\lambda\,}",
          r"T_{1,2}(t)=T_f\pm\dfrac{C_{2,1}}{C_1+C_2}\,(T_{10}-T_{20})\,e^{-\lambda t}",
          r"C_2\gg C_1:\ \ T_f\to T_{20},\quad \lambda\to h/C_1"]),
        ("Particularización al problema",
         rf"Las capacidades son $C_1=({m1})({fx(c1,0)})={fx(C1,1)}\ \mathrm{{J/K}}$ para el aluminio y $C_2=({m2})({fx(c2,0)})={fx(C2,1)}\ \mathrm{{J/K}}$ para el agua, con $h={fx(h,0)}\ \mathrm{{W/K}}$. La capacidad reducida es $C_r={fx(cred,2)}\ \mathrm{{J/K}}$. Se recupera la temperatura final y la constante de tiempo de la escena, y de la solución de cada cuerpo, los valores a 20 s.",
         [rf"T_f=\dfrac{{({fx(C1,1)})({fx(t10,0)})+({fx(C2,1)})({fx(t20,0)})}}{{{fx(C1+C2,1)}}}={tf:.2f}\ ^\circ\mathrm{{C}}",
          rf"\lambda={fx(h,0)}\left(\dfrac1{{{fx(C1,1)}}}+\dfrac1{{{fx(C2,1)}}}\right)={lam:.4f}\ \mathrm{{s^{{-1}}}},\qquad \tau={tau:.1f}\ \mathrm{{s}}",
          rf"T_1(20\ \mathrm{{s}})={t20s:.4f}\ ^\circ\mathrm{{C}},\qquad T_2(20\ \mathrm{{s}})={t2_20:.4f}\ ^\circ\mathrm{{C}}"]),
        ("Errores comunes y límites de validez",
         r"Un error frecuente es promediar las temperaturas sin ponderar por las capacidades, lo cual sólo es válido si $C_1=C_2$. Otro es creer que el cuerpo caliente baja tanto como sube el frío: la variación de temperatura es inversamente proporcional a la capacidad. También se confunde $h$ con el calor específico: la constante de tiempo depende de ambos. El modelo supone que cada cuerpo es isotermo (número de Biot pequeño), que $c_i$ y $h$ no dependen de la temperatura y que no hay pérdidas al ambiente ni cambio de fase (ebullición); si falla la hipótesis de cuerpo isotermo hay que resolver la ecuación del calor $\partial_tT=D\nabla^2T$.",
         []),
    ]
    return dict(concept=concept)


# =============================================================== GasProcesses
def _gas_processes() -> dict:
    R, n = c23.R, c23.N_MOL
    p1, v1, v2, g = c23.P1, c23.V1, c23.V2, c23.GAMMA_MONO
    t1 = c23.temperature(p1, v1)
    cv = R / (g - 1)
    rows = c23.first_law_table()
    wi, wb, wa = (c23.work_analytic(k) for k in ("isothermal", "isobaric", "adiabatic"))
    t_b = c23.temperature(p1, v2)
    t_a = t1 * (v1 / v2) ** (g - 1)
    p2a = p1 * (v1 / v2) ** g
    concept = [
        ("Qué se estudia y por qué importa",
         "La primera ley de la termodinámica expresa la conservación de la energía cuando un sistema intercambia calor y trabajo con su entorno. Un gas que se expande empuja un pistón y entrega trabajo, pero cuánto trabajo y cuánto calor intervienen depende del camino seguido entre el estado inicial y el final. Entender ese reparto es el fundamento del diseño de motores, compresores, refrigeradores y turbinas.",
         []),
        ("Magnitudes y convenciones",
         r"El estado de un gas ideal está fijado por $p$ (Pa), $V$ ($\mathrm{m^3}$), $T$ (K) y la cantidad de sustancia $n$ (mol), relacionados por $pV=nRT$ con $R=8.314\ \mathrm{J/(mol\,K)}$. Se usa la convención $W>0$ cuando el sistema realiza trabajo sobre el entorno y $Q>0$ cuando recibe calor, de modo que $\Delta U=Q-W$. Las capacidades caloríficas molares $C_V$ y $C_P$ se miden en $\mathrm{J/(mol\,K)}$ y su cociente $\gamma=C_P/C_V$ es adimensional. Un litro vale $10^{-3}\ \mathrm{m^3}$.",
         [r"pV=nRT,\qquad \Delta U=Q-W,\qquad \gamma=\dfrac{C_P}{C_V}"]),
        ("Intuición y analogías",
         r"La energía interna es como el saldo de una cuenta bancaria: depende sólo del estado, mientras que el calor y el trabajo son los movimientos que lo modifican, y existen muchas combinaciones de depósitos y retiros con el mismo saldo final. Por eso $Q$ y $W$ no son funciones de estado, y se denotan con $\delta Q$ y $\delta W$. En el diagrama $p$-$V$ el trabajo es el área bajo la curva, y una curva distinta entre los mismos extremos encierra un área distinta.",
         []),
        ("Energía interna del gas ideal",
         r"Para un gas ideal las moléculas no interactúan, y por tanto la energía interna es la suma de energías cinéticas internas, que dependen sólo de $T$ (experimento de Joule de expansión libre). El teorema de equipartición asigna $\tfrac12kT$ por grado de libertad cuadrático, lo que da $U=\tfrac f2nRT$, con $f=3$ para un gas monoatómico y $f=5$ para un diatómico a temperatura ambiente. Entonces $C_V=\tfrac f2R$ y $\gamma=1+2/f$, y el cambio de energía interna es el mismo en cualquier proceso que una dos temperaturas dadas.",
         [r"U=\dfrac f2\,nRT,\qquad C_V=\dfrac f2R,\qquad \gamma=1+\dfrac2f",
          r"\Delta U=nC_V\,\Delta T\quad\text{(cualquier proceso)}"]),
        ("Primera ley y trabajo en procesos cuasiestáticos",
         r"En un proceso cuasiestático (sucesión de estados de equilibrio) la presión del gas es igual a la externa, de modo que el trabajo sobre un pistón es $\delta W=p\,dV$. La primera ley en forma diferencial relaciona calor, trabajo y energía interna, y da el calor como la suma de un término de estado y otro de camino. El trabajo total se obtiene integrando $p(V)$ a lo largo del camino, que se debe especificar. De la ecuación de estado se despeja $p$ en función de $V$ para cada proceso.",
         [r"dU=\delta Q-\delta W,\qquad \delta W=p\,dV,\qquad W=\int_{V_1}^{V_2}p(V)\,dV",
          r"\delta Q=nC_V\,dT+p\,dV"]),
        ("Isoterma e isobara",
         r"En la isoterma $T$ es constante, $\Delta U=0$ y $p=nRT/V$, de modo que todo el calor absorbido se convierte en trabajo. En la isobara $p$ es constante y el trabajo es el rectángulo $p\Delta V$; el calor intercambiado define $C_P$ y, combinando con $\Delta U$ y la ecuación de estado, se llega a la relación de Mayer. Esta relación explica por qué $C_P>C_V$: a presión constante parte del calor se emplea en realizar trabajo de expansión.",
         [r"W_{iso}=nRT\ln\dfrac{V_2}{V_1},\qquad \Delta U=0,\qquad Q_{iso}=W_{iso}",
          r"W_{isob}=p\,(V_2-V_1)=nR\,\Delta T,\qquad Q_{isob}=nC_V\Delta T+nR\Delta T=nC_P\Delta T",
          r"C_P-C_V=R"]),
        ("Adiabata y proceso politrópico",
         r"En una adiabática $\delta Q=0$ y la primera ley da $nC_V\,dT=-p\,dV$. Sustituyendo $T=pV/(nR)$ se obtiene una ecuación separable cuya solución es $pV^\gamma=\text{cte}$. El trabajo es la disminución de energía interna. El resultado se generaliza a procesos politrópicos $pV^k=\text{cte}$, que incluyen los anteriores como casos particulares ($k=0$ isobara, $k=1$ isoterma, $k=\gamma$ adiabática), y su capacidad calorífica efectiva sale de eliminar $dp$ con la ecuación de estado.",
         [r"nC_V\,dT=-p\,dV\ \Rightarrow\ \dfrac{dp}{p}=-\gamma\dfrac{dV}{V}\ \Rightarrow\ pV^\gamma=\text{cte},\quad TV^{\gamma-1}=\text{cte}",
          r"W_{ad}=\int_{V_1}^{V_2}\dfrac{p_1V_1^\gamma}{V^\gamma}dV=\dfrac{p_1V_1-p_2V_2}{\gamma-1}=nC_V(T_1-T_2)",
          r"pV^k=\text{cte}:\ \ W=\dfrac{p_1V_1-p_2V_2}{k-1},\qquad C_k=C_V+\dfrac R{1-k}"]),
        ("Camino, funciones de estado y balance global",
         r"Entre los mismos estados inicial y final, $\Delta U$ y $\Delta T$ se determinan sólo por los extremos mientras que $W$ y $Q$ dependen del camino: el trabajo en una expansión entre $V_1$ y $V_2$ es mayor cuanto más calor se absorbe (isoterma) que cuando no se absorbe (adiabática). La integral cerrada de $\delta W$ no es cero y su valor es el área encerrada del ciclo. Verificar $Q=\Delta U+W$ en cada proceso es la comprobación básica de coherencia de los cálculos.",
         [r"\oint dU=0,\qquad \oint\delta W=\oint\delta Q\neq0",
          r"\Delta U_{iso}=0\ne\Delta U_{isob}\ne\Delta U_{ad},\qquad Q=\Delta U+W"]),
        ("Resultado general",
         r"Las expresiones para los tres procesos se resumen con las magnitudes iniciales $(p_1,V_1,T_1)$ y la razón de volúmenes $V_2/V_1$. Todas tienen unidades de energía ($\mathrm{Pa\cdot m^3}=\mathrm{J}$).",
         [r"\boxed{\,W_{iso}=p_1V_1\ln\dfrac{V_2}{V_1},\qquad W_{isob}=p_1(V_2-V_1),\qquad W_{ad}=\dfrac{p_1V_1}{\gamma-1}\left[1-\left(\dfrac{V_1}{V_2}\right)^{\gamma-1}\right]\,}",
          r"\Delta U_{iso}=0,\qquad \Delta U_{isob}=\dfrac{p_1}{\gamma-1}\,(V_2-V_1),\qquad \Delta U_{ad}=-W_{ad}",
          r"Q=\Delta U+W:\ \ Q_{iso}=W_{iso},\quad Q_{isob}=\dfrac{\gamma}{\gamma-1}\,p_1\Delta V,\quad Q_{ad}=0"]),
        ("Particularización al problema",
         rf"Con $n=1$ mol, $p_1={sci(p1,3)}\ \mathrm{{Pa}}$, $V_1={fx(v1*1000,0)}\ \mathrm{{L}}$ y $V_2={fx(v2*1000,0)}\ \mathrm{{L}}$ la temperatura inicial es $T_1=p_1V_1/(nR)={fx(t1,2)}\ \mathrm{{K}}$ y $C_V=\tfrac32R={fx(cv,3)}\ \mathrm{{J/(mol\,K)}}$ ($\gamma=5/3$). Los resultados coinciden con los de la solución analítica de la escena.",
         [rf"W_{{iso}}=p_1V_1\ln2=({sci(p1,3)})({fx(v1,3)})(0.6931)={wi:.1f}\ \mathrm{{J}},\qquad \Delta U=0,\qquad Q={rows[0]['Q']:.1f}\ \mathrm{{J}}",
          rf"W_{{isob}}=p_1\Delta V={wb:.1f}\ \mathrm{{J}},\quad T_2={fx(t_b,2)}\ \mathrm{{K}},\quad \Delta U=nC_V\Delta T={rows[1]['dU']:.1f}\ \mathrm{{J}},\quad Q={rows[1]['Q']:.1f}\ \mathrm{{J}}",
          rf"p_2=p_1(V_1/V_2)^{{\gamma}}={sci(p2a,4)}\ \mathrm{{Pa}},\quad T_2={fx(t_a,2)}\ \mathrm{{K}},\quad W_{{ad}}={wa:.1f}\ \mathrm{{J}}",
          rf"\Delta U_{{ad}}=nC_V(T_2-T_1)={rows[2]['dU']:.1f}\ \mathrm{{J}}\ \Rightarrow\ Q_{{ad}}=\Delta U+W={fx(rows[2]['dU']+wa,6)}\ \mathrm{{J}}"]),
        ("Errores comunes y límites de validez",
         r"Se confunde con frecuencia el signo de $W$ (trabajo del sistema frente a trabajo sobre el sistema), de modo que la primera ley cambia a $\Delta U=Q+W_{\text{sobre}}$. También es habitual suponer $Q=0$ en una isoterma o $\Delta U=0$ en una isobara. Otro error es usar $\Delta U=nC_P\Delta T$ en procesos a presión constante: $\Delta U$ siempre usa $C_V$. Los resultados valen para procesos cuasiestáticos de un gas ideal con $C_V$ constante; en una expansión rápida contra el vacío, o cuando $C_V$ depende de la temperatura (grados de libertad vibracionales a alta $T$) o hay interacciones moleculares (gas de van der Waals), las fórmulas deben modificarse.",
         []),
    ]
    return dict(concept=concept)


# =============================================================== OttoCycle
def _otto() -> dict:
    R, n = c23.R, c23.N_MOL
    r, g, ta, tc = c23.RATIO, c23.GAMMA_DIA, c23.T_A, c23.T_C_HEAT
    st = c23.otto_states()
    o = c23.otto_numeric()
    eta = c23.otto_efficiency_analytic()
    cv = R / (g - 1)
    tb, td = st["B"][2], st["D"][2]
    qin = n * cv * (tc - tb)
    qout = n * cv * (td - ta)
    eta_c = 1 - ta / tc
    concept = [
        ("Qué se estudia y por qué importa",
         "El ciclo de Otto es el modelo ideal de los motores de gasolina: el gas se comprime casi sin intercambiar calor, recibe energía de la combustión a volumen casi constante, se expande, y cede calor al escape. Interesa cuánto trabajo neto se extrae por ciclo y qué fracción del calor absorbido se aprovecha. Saber de qué parámetros depende esa fracción orienta el diseño: por qué se busca una relación de compresión alta y qué limita su aumento.",
         []),
        ("Magnitudes y convenciones",
         r"Intervienen la relación de compresión $r=V_{max}/V_{min}$ (adimensional), el índice adiabático $\gamma$, las temperaturas de los cuatro estados A, B, C, D (K), el calor absorbido $Q_{in}$ y cedido $Q_{out}$ (J) y el trabajo neto $W$ (J). La eficiencia térmica es el cociente entre lo que se obtiene y lo que se paga. Para el aire $\gamma\approx1.4$ y los motores reales tienen $r$ entre 8 y 12.",
         [r"r=\dfrac{V_A}{V_B},\qquad \eta=\dfrac{W}{Q_{in}},\qquad Q_{in},Q_{out}>0\ \text{(magnitudes)}"]),
        ("Intuición y analogías",
         r"El motor es como una cascada: la energía que pasa de una fuente caliente a un sumidero frío mueve una rueda, pero no toda la que cae se convierte en trabajo; una parte debe seguir hasta el sumidero. Al comprimir más, la altura efectiva de la cascada aumenta y el aprovechamiento mejora. El resultado notable es que el rendimiento depende sólo de cuánto se comprime el gas, y no de la cantidad de calor que se quema.",
         []),
        ("El ciclo como camino cerrado",
         r"El ciclo consta de cuatro tramos reversibles: compresión adiabática A→B, calentamiento isocoro B→C, expansión adiabática C→D y enfriamiento isocoro D→A. Al volver al estado inicial, $\Delta U=0$ y, según la primera ley, el trabajo neto es la diferencia entre el calor absorbido y el cedido. Las adiabáticas no intercambian calor y en las isocoras no hay trabajo, por lo cual todo el calor entra y sale por los dos tramos a volumen constante.",
         [r"\oint dU=0\ \Rightarrow\ W=Q_{in}-Q_{out}",
          r"\eta=\dfrac{W}{Q_{in}}=1-\dfrac{Q_{out}}{Q_{in}}"]),
        ("Adiabáticas: relaciones entre estados",
         r"En un tramo adiabático reversible de gas ideal se cumple $TV^{\gamma-1}=\text{cte}$, que se deduce de $nC_V\,dT=-p\,dV$ con $p=nRT/V$. Como $V_A=rV_B$ y $V_D=V_A$, $V_C=V_B$, ambos tramos relacionan las temperaturas por el mismo factor $r^{\gamma-1}$. Esto fija $T_B$ y $T_D$ en función de las temperaturas de los extremos del calentamiento, y es lo que hace que los cocientes de temperaturas de los dos tramos sean iguales.",
         [r"T_AV_A^{\gamma-1}=T_BV_B^{\gamma-1}\ \Rightarrow\ T_B=T_A\,r^{\gamma-1}",
          r"T_CV_B^{\gamma-1}=T_DV_A^{\gamma-1}\ \Rightarrow\ T_D=T_C\,r^{1-\gamma}",
          r"\dfrac{T_D}{T_A}=\dfrac{T_C}{T_B}"]),
        ("Calores en las isocoras",
         r"En un tramo a volumen constante $\delta W=0$, de modo que $Q=\Delta U=nC_V\Delta T$. El calor absorbido es el del tramo B→C y el cedido es el del tramo D→A. Se toma su valor absoluto para que ambos sean positivos. Aquí se supone $C_V$ independiente de $T$, lo que es una aproximación razonable para un gas diatómico a las temperaturas del ciclo.",
         [r"Q_{in}=nC_V\,(T_C-T_B),\qquad Q_{out}=nC_V\,(T_D-T_A)",
          r"W=nC_V\big[(T_C-T_B)-(T_D-T_A)\big]"]),
        ("Eficiencia térmica del ciclo de Otto",
         r"Al dividir $Q_{out}$ entre $Q_{in}$ se usa la igualdad entre cocientes demostrada arriba: $T_D-T_A=(T_C-T_B)\,r^{1-\gamma}$, porque $T_D=T_Cr^{1-\gamma}$ y $T_A=T_Br^{1-\gamma}$. Por tanto el cociente no depende de cuánto se caliente el gas. El rendimiento sólo depende de $r$ y de $\gamma$, y aumenta con ambos. Es un resultado exacto del modelo ideal, no una aproximación.",
         [r"\dfrac{Q_{out}}{Q_{in}}=\dfrac{T_D-T_A}{T_C-T_B}=r^{1-\gamma}",
          r"\eta_{Otto}=1-r^{1-\gamma}=1-\dfrac1{r^{\gamma-1}}"]),
        ("Comparación con el límite de Carnot",
         r"La segunda ley impone que ninguna máquina entre dos focos puede superar el rendimiento de Carnot. El ciclo de Otto toma calor a temperaturas que van de $T_B$ a $T_C$ y lo cede entre $T_D$ y $T_A$; como $T_B>T_A$ y $T_C>T_D$, su rendimiento queda por debajo del de una máquina de Carnot entre $T_C$ y $T_A$. El cálculo es inmediato con los resultados del apartado previo.",
         [r"\eta_{Otto}=1-\dfrac{T_A}{T_B}<1-\dfrac{T_A}{T_C}=\eta_{Carnot}(T_C,T_A)\quad\text{pues }T_B<T_C",
          r"\Delta S_{ciclo}=nC_V\ln\dfrac{T_C}{T_B}+nC_V\ln\dfrac{T_A}{T_D}=0"]),
        ("Resultado general",
         r"El trabajo neto y la eficiencia del ciclo se escriben con la relación de compresión, el índice adiabático y las temperaturas extremas. La eficiencia es independiente del calor aportado y el trabajo es proporcional a él. Dimensionalmente $nC_V\Delta T$ es una energía y $\eta$ es adimensional.",
         [r"\boxed{\,\eta=1-r^{1-\gamma},\qquad W=nC_V\,(T_C-T_A\,r^{\gamma-1})\,\big(1-r^{1-\gamma}\big)\,}",
          r"Q_{in}=nC_V(T_C-T_Ar^{\gamma-1}),\qquad Q_{out}=r^{1-\gamma}\,Q_{in}",
          r"\dfrac{\partial\eta}{\partial r}=(\gamma-1)\,r^{-\gamma}>0"]),
        ("Particularización al problema",
         rf"Con $r={r:.0f}$, $\gamma=7/5$, $T_A={ta:.0f}\ \mathrm{{K}}$, $T_C={tc:.0f}\ \mathrm{{K}}$ y $n=1$ mol, se tiene $C_V=\tfrac52R={fx(cv,3)}\ \mathrm{{J/(mol\,K)}}$. Las temperaturas de los estados son $T_B={fx(tb,2)}\ \mathrm{{K}}$ y $T_D={fx(td,2)}\ \mathrm{{K}}$, y la eficiencia y el trabajo coinciden con los de la escena.",
         [rf"T_B=T_A r^{{\gamma-1}}={ta:.0f}\cdot8^{{0.4}}={fx(tb,2)}\ \mathrm{{K}},\qquad T_D=T_C r^{{1-\gamma}}={fx(td,2)}\ \mathrm{{K}}",
          rf"Q_{{in}}=nC_V(T_C-T_B)={fx(cv,3)}\,({fx(tc-tb,2)})={qin:.0f}\ \mathrm{{J}},\qquad Q_{{out}}=nC_V(T_D-T_A)={qout:.0f}\ \mathrm{{J}}",
          rf"\eta=1-8^{{-0.4}}={eta:.4f},\qquad W=\eta\,Q_{{in}}={eta*qin:.0f}\ \mathrm{{J}}=Q_{{in}}-Q_{{out}}={qin-qout:.0f}\ \mathrm{{J}}",
          rf"\eta_{{Carnot}}(T_C,T_A)=1-\dfrac{{{ta:.0f}}}{{{tc:.0f}}}={eta_c:.4f}>\eta"]),
        ("Errores comunes y límites de validez",
         r"Un error común es calcular la eficiencia como $1-T_A/T_C$, que es la de Carnot y no se alcanza. Otro es creer que calentar más (mayor $T_C$) mejora la eficiencia: sólo aumenta el trabajo. También se suele olvidar que $r$ es una razón de volúmenes y no de presiones. El modelo supone aire como gas ideal con $\gamma$ constante, procesos reversibles y combustión instantánea; en motores reales la fricción, la pérdida de calor por las paredes, la combustión finita y la variación de $\gamma$ con la temperatura reducen el rendimiento, y una relación de compresión demasiado alta produce detonación.",
         []),
    ]
    return dict(concept=concept)


# =============================================================== GasPressure
def _pressure() -> dict:
    kb, m, T, N = c24.KB, c24.M_HE, c24.T_GAS, 4000
    V = c24.BOX**3
    A = c24.BOX**2
    p = c24.pressure_analytic(N, T, V)
    vrms = float(np.sqrt(3 * kb * T / m))
    vx2 = kb * T / m
    nd = N / V
    fl = float(np.sqrt(2 / (3 * N)))
    k_mean = 1.5 * kb * T
    concept = [
        ("Qué se estudia y por qué importa",
         "La presión que un gas ejerce sobre las paredes de su recipiente es el efecto promedio de un número enorme de choques moleculares. La teoría cinética explica una magnitud macroscópica, la presión, y una propiedad térmica, la temperatura, desde la mecánica de las moléculas. Esa conexión es el modelo mínimo de la física estadística, y permite entender la ley del gas ideal, la difusión, la efusión y el funcionamiento de bombas y sensores.",
         []),
        ("Magnitudes y convenciones",
         r"Se usan el número de moléculas $N$, el volumen $V$ ($\mathrm{m^3}$), la densidad numérica $n=N/V$ ($\mathrm{m^{-3}}$), la masa molecular $m$ (kg), la constante de Boltzmann $k=1.381\times10^{-23}\ \mathrm{J/K}$ y la presión $p$ en pascales. Los promedios sobre moléculas se escriben con paréntesis angulares, $\langle\cdot\rangle$. La relación con la constante de los gases es $R=N_Ak$. A temperatura y presión ambientes hay unas $2.5\times10^{25}$ moléculas por metro cúbico y rapideces del orden de 500 m/s.",
         [r"p=\dfrac{F}{A},\qquad n=\dfrac NV,\qquad R=N_A\,k"]),
        ("Intuición y analogías",
         r"Una lluvia de granizo sobre un techo produce una fuerza aparentemente continua aunque cada piedra golpea de forma discreta. La presión es el análogo: millones de impactos por microsegundo que se suavizan en un valor constante. Cuanto más rápida y más numerosa es la lluvia, mayor es la fuerza, y de ahí que la presión crezca con la densidad y con la agitación térmica.",
         []),
        ("Hipótesis del modelo",
         "El modelo del gas ideal supone un número muy grande de moléculas puntuales con movimiento aleatorio e isótropo, sin interacciones entre sí salvo choques breves, y con colisiones perfectamente elásticas contra paredes lisas. La isotropía significa que los promedios de las componentes de la velocidad son iguales por simetría y que no hay dirección privilegiada, y la ausencia de interacciones implica que las moléculas se mueven en línea recta entre colisiones. Son las hipótesis que se usan para calcular el impulso transferido.",
         [r"\langle v_x\rangle=\langle v_y\rangle=\langle v_z\rangle=0,\qquad \langle v_x^2\rangle=\langle v_y^2\rangle=\langle v_z^2\rangle"]),
        ("Impulso en una colisión y flujo sobre la pared",
         r"En un choque elástico con una pared perpendicular a $x$ la componente $v_x$ se invierte y el impulso entregado a la pared es $2mv_x$. En un intervalo $dt$ alcanzan un área $A$ las moléculas con $v_x>0$ situadas a una distancia menor que $v_xdt$; su número es $\tfrac12\,nAv_x\,dt$, pues la mitad de las moléculas tiene $v_x>0$. La fuerza media es el impulso total por unidad de tiempo y la presión, esa fuerza dividida entre $A$.",
         [r"\Delta p_x=2mv_x,\qquad dN=\dfrac12\,n\,A\,v_x\,dt\quad(v_x>0)",
          r"F=\dfrac{1}{dt}\sum\Delta p_x=\dfrac12\,nA\,\big\langle 2m\,v_x^2\big\rangle,\qquad p=\dfrac FA=nm\langle v_x^2\rangle"]),
        ("Isotropía y presión en función de la energía",
         r"Por isotropía $\langle v^2\rangle=\langle v_x^2\rangle+\langle v_y^2\rangle+\langle v_z^2\rangle=3\langle v_x^2\rangle$, de donde la presión es un tercio de la densidad de momento cuadrático. Expresado con la energía cinética media por molécula $\langle K\rangle=\tfrac12m\langle v^2\rangle$, la presión es dos tercios de la densidad de energía cinética de traslación. Este resultado no depende de la distribución de rapideces, sólo de su segundo momento.",
         [r"p=\dfrac13\,n\,m\,\langle v^2\rangle=\dfrac23\,n\,\langle K\rangle",
          r"pV=\dfrac23\,N\langle K\rangle=\dfrac23\,U_{tr}"]),
        ("Temperatura cinética y ley del gas ideal",
         r"Comparando $pV=\tfrac23N\langle K\rangle$ con la ley empírica $pV=NkT$ se identifica la energía cinética media de traslación con $\tfrac32kT$. En mecánica estadística el mismo resultado sale del teorema de equipartición: cada término cuadrático del hamiltoniano contribuye con $\tfrac12kT$ al promedio canónico. De aquí se obtiene la rapidez cuadrática media, que crece con $\sqrt T$ y disminuye con la masa.",
         [r"\langle K\rangle=\dfrac32\,kT\ \Rightarrow\ pV=NkT",
          r"v_{rms}=\sqrt{\langle v^2\rangle}=\sqrt{\dfrac{3kT}{m}},\qquad \langle v_x^2\rangle=\dfrac{kT}{m}"]),
        ("Cálculo con paredes finitas y fluctuaciones",
         r"Un cálculo alternativo sigue a una sola molécula en una caja de lado $L$: tras chocar con la pared tarda $2L/|v_x|$ en volver, de modo que la fuerza media de esa molécula es $mv_x^2/L$; sumando sobre $N$ moléculas se reproduce $p=Nm\langle v_x^2\rangle/V$. La presión instantánea fluctúa porque el número de choques es finito, y la fluctuación relativa de la energía cinética media estimada con $N$ moléculas es $\sqrt{2/(3N)}$, lo que da el orden del error estadístico de una simulación pequeña.",
         [r"\bar F_1=\dfrac{2m|v_x|}{2L/|v_x|}=\dfrac{mv_x^2}{L},\qquad p=\dfrac{N\bar F_1}{L^2}=\dfrac{Nm\langle v_x^2\rangle}{L^3}=\dfrac{Nm\langle v_x^2\rangle}{V}",
          r"\dfrac{\delta T}{T}\simeq\sqrt{\dfrac{2}{3N}}"]),
        ("Resultado general",
         r"La presión se expresa de tres maneras equivalentes, según se prefiera la microscópica, la energética o la termodinámica. La última es la ley del gas ideal. En unidades: $kT$ es una energía, $N/V$ es $\mathrm{m^{-3}}$ y su producto es $\mathrm{J/m^3}=\mathrm{Pa}$.",
         [r"\boxed{\,p=\dfrac13\,\dfrac NV\,m\langle v^2\rangle=\dfrac23\,\dfrac NV\langle K\rangle=\dfrac{N\,kT}{V}\,}",
          r"[p]=\mathrm{m^{-3}\cdot J}=\mathrm{N\,m^{-2}}=\mathrm{Pa}"]),
        ("Particularización al problema",
         rf"Para $N={N}$ átomos de helio ($m={sci(m,4)}\ \mathrm{{kg}}$) a $T={T:.0f}\ \mathrm{{K}}$ en un cubo de lado $L=1\ \mu\mathrm{{m}}$ ($V={sci(V,3)}\ \mathrm{{m^3}}$) la densidad es $n={sci(nd,4)}\ \mathrm{{m^{{-3}}}}$. La fluctuación relativa esperada de la temperatura medida con $N={N}$ es del orden de $\sqrt{{2/(3N)}}={fl*100:.2f}\,\%$. El valor esperado de la presión coincide con el de la escena.",
         [rf"\langle K\rangle=\tfrac32kT={sci(k_mean,4)}\ \mathrm{{J}},\qquad v_{{rms}}=\sqrt{{3kT/m}}={vrms:.1f}\ \mathrm{{m/s}},\qquad \langle v_x^2\rangle={sci(vx2,4)}\ \mathrm{{m^2/s^2}}",
          rf"p=nm\langle v_x^2\rangle=({sci(nd,4)})({sci(m,4)})({sci(vx2,4)})={nd*m*vx2:.3f}\ \mathrm{{Pa}}",
          rf"p=\dfrac{{NkT}}{{V}}=\dfrac{{({N})({sci(kb,5)})({T:.0f})}}{{{sci(V,3)}}}={p:.3f}\ \mathrm{{Pa}}",
          rf"F=pA={sci(p*A,4)}\ \mathrm{{N}}\ \text{{sobre una cara de área }}A={sci(A,3)}\ \mathrm{{m^2}}"]),
        ("Errores comunes y límites de validez",
         r"Se suele olvidar el factor $1/2$ (sólo la mitad de las moléculas se dirige a la pared) o el factor 2 del impulso, que se compensan, o usar $\tfrac12m\langle v^2\rangle=kT$ en lugar de $\tfrac32kT$. Otro error es pensar que cada molécula tiene la rapidez $v_{rms}$: la distribución es ancha. También se debe recordar que la presión mide el flujo de momento, no de energía. El modelo falla cuando las interacciones importan (gases densos, ecuación de van der Waals), cuando la longitud de onda térmica de de Broglie es comparable a la separación (gases degenerados, a baja $T$ y alta densidad: estadística de Fermi o Bose) y a muy baja densidad, donde el recorrido libre medio supera al recipiente y el régimen es molecular, no de equilibrio.",
         []),
    ]
    return dict(concept=concept)


# =============================================================== MaxwellBoltzmann
def _maxwell() -> dict:
    kb, m, T = c24.KB, c24.M_N2, c24.T_GAS
    an = c24.speeds_analytic()
    vp, vm, vr = float(an["v_mp"]), float(an["v_mean"]), float(an["v_rms"])
    a = m / (2 * kb * T)
    sigma = float(np.sqrt(kb * T / m))
    fp = float(c24.mb_pdf(vp))
    e0 = 10 * kb * T
    concept = [
        ("Qué se estudia y por qué importa",
         "Las moléculas de un gas no tienen todas la misma rapidez: colisionan constantemente y se reparten la energía de manera aleatoria. La distribución de Maxwell-Boltzmann dice qué fracción de moléculas tiene cada rapidez a una temperatura dada. De ella dependen la velocidad de las reacciones químicas (sólo reaccionan las moléculas con energía suficiente), la evaporación, la escape de la atmósfera de los planetas, la efusión a través de pequeñas aberturas y el ensanchamiento Doppler de las líneas espectrales.",
         []),
        ("Magnitudes y convenciones",
         r"La densidad de probabilidad $f(v)$ tiene unidades de $\mathrm{s/m}$ de modo que $f(v)\,dv$ sea la fracción de moléculas con rapidez entre $v$ y $v+dv$ (adimensional). Se definen tres rapideces características: la más probable $v_p$, la media $\bar v$ y la cuadrática media $v_{rms}$. La masa molecular $m$ se da en kg (para $\mathrm{N_2}$, $28\,u=4.65\times10^{-26}\ \mathrm{kg}$). El parámetro natural es $kT/m$, que tiene unidades de velocidad al cuadrado.",
         [r"\int_0^\infty f(v)\,dv=1,\qquad [f]=\mathrm{s/m},\qquad [kT/m]=\mathrm{m^2/s^2}"]),
        ("Intuición y analogías",
         r"La forma de la curva es la de dos factores que compiten. Hay muchas más maneras de repartir la rapidez cuando $v$ es grande (el volumen del espacio de velocidades crece como $v^2$) pero la energía necesaria hace que esas maneras sean exponencialmente raras. El resultado es una curva que parte de cero, tiene un máximo y una cola larga. Es análogo a la distribución de alturas de las personas en una población: hay valores típicos, y valores extremos muy poco frecuentes.",
         []),
        ("Distribución canónica y factor de Boltzmann",
         r"Un gas en contacto térmico con un baño a temperatura $T$ se describe con la distribución canónica: la densidad de probabilidad de un microestado del espacio de fases es proporcional a $e^{-H/kT}$. Para un gas de moléculas sin interacción, el hamiltoniano de una molécula es $H=\vec p^{\,2}/2m+U(\vec r)$ y la distribución se factoriza en un factor espacial y otro de velocidades. Con $U=0$ en el interior de la caja el factor de velocidades es independiente de la posición, y las tres componentes son variables independientes.",
         [r"\rho(\vec r,\vec p)\propto e^{-H/kT},\qquad H=\dfrac{\vec p^{\,2}}{2m}+U(\vec r)",
          r"g(\vec v)\,d^3v\propto\exp\!\left(-\dfrac{m(v_x^2+v_y^2+v_z^2)}{2kT}\right)d^3v"]),
        ("Derivación de Maxwell por independencia e isotropía",
         r"El mismo resultado puede obtenerse sin hamiltoniano (argumento de Maxwell): si las componentes son independientes, $g(\vec v)=\phi(v_x)\phi(v_y)\phi(v_z)$, y si además es isótropa depende sólo del módulo, $g=\psi(v^2)$. La única función que es producto de funciones de cada coordenada y depende sólo de la suma de cuadrados es una gaussiana, como se ve derivando el logaritmo respecto de cada componente. La constante $a$ queda fijada por el valor de $\langle v_x^2\rangle$ (equipartición, $m\langle v_x^2\rangle=kT$).",
         [r"\ln g=\ln\phi(v_x)+\ln\phi(v_y)+\ln\phi(v_z)=\ln\psi(v^2)\ \Rightarrow\ \dfrac{\phi'(v_x)}{v_x\phi(v_x)}=-2a=\text{cte}",
          r"\phi(v_x)=\sqrt{\dfrac a\pi}\,e^{-av_x^2},\qquad \langle v_x^2\rangle=\dfrac1{2a}=\dfrac{kT}{m}\ \Rightarrow\ a=\dfrac m{2kT}"]),
        ("Densidad de rapideces",
         r"Para pasar de la densidad de velocidades a la de rapideces se integra sobre la dirección: el elemento de volumen en coordenadas esféricas es $4\pi v^2dv$ una vez integrados los ángulos, que no aparecen en la gaussiana. La normalización se comprueba con la integral gaussiana $\int_0^\infty v^2e^{-av^2}dv=\sqrt\pi/(4a^{3/2})$. El resultado es la distribución de Maxwell-Boltzmann. En términos de la energía cinética $E=\tfrac12mv^2$ la distribución es proporcional a $\sqrt E\,e^{-E/kT}$.",
         [r"f(v)=4\pi\left(\dfrac m{2\pi kT}\right)^{3/2}v^2\,e^{-mv^2/2kT}",
          r"\int_0^\infty f\,dv=4\pi\left(\dfrac a\pi\right)^{3/2}\dfrac{\sqrt\pi}{4a^{3/2}}=1",
          r"F(E)\,dE=\dfrac{2}{\sqrt\pi}\,(kT)^{-3/2}\sqrt E\,e^{-E/kT}\,dE"]),
        ("Momentos y rapideces características",
         r"El máximo de $f$ se halla de $df/dv=0$ y da $v_p$. Los momentos se calculan con la integral general $\int_0^\infty v^{n}e^{-av^2}dv=\tfrac12\,\Gamma\!\left(\tfrac{n+1}2\right)a^{-(n+1)/2}$, por lo que $\langle v^n\rangle=\Gamma\!\left(\tfrac{n+3}2\right)\big/\Gamma(\tfrac32)\cdot(2kT/m)^{n/2}$. Los casos $n=1$ y $n=2$ dan la rapidez media y la cuadrática media; esta última concuerda con la energía cinética media $\tfrac32kT$ de la teoría cinética.",
         [r"\dfrac{df}{dv}=0\ \Rightarrow\ v_p=\sqrt{\dfrac{2kT}m}",
          r"\langle v^n\rangle=\dfrac{\Gamma\!\left(\frac{n+3}2\right)}{\Gamma\!\left(\frac32\right)}\left(\dfrac{2kT}m\right)^{n/2}",
          r"\bar v=\sqrt{\dfrac{8kT}{\pi m}},\qquad v_{rms}=\sqrt{\dfrac{3kT}{m}}"]),
        ("Forma adimensional y cola de la distribución",
         r"Con la variable $x=v/v_p$ la distribución depende de una sola curva universal, $f\,dv=\tfrac{4}{\sqrt\pi}x^2e^{-x^2}dx$, y los cocientes entre rapideces son fijos: $v_p:\bar v:v_{rms}=1:2/\sqrt\pi:\sqrt{3/2}$. Por ello todas las rapideces escalan con $\sqrt{T/m}$. La fracción de moléculas con energía superior a $E_0\gg kT$ es aproximadamente $e^{-E_0/kT}$ multiplicada por un factor algebraico, el origen de la ley de Arrhenius de la cinética química.",
         [r"x=\dfrac v{v_p},\qquad \dfrac{\bar v}{v_p}=\dfrac2{\sqrt\pi}\simeq1.128,\qquad \dfrac{v_{rms}}{v_p}=\sqrt{\dfrac32}\simeq1.225",
          r"P(E>E_0)\simeq\dfrac{2}{\sqrt\pi}\sqrt{\dfrac{E_0}{kT}}\,e^{-E_0/kT},\qquad E_0\gg kT"]),
        ("Resultado general",
         r"La distribución y sus tres rapideces características dependen sólo de la combinación $kT/m$. Las unidades de $f$ son $\mathrm{s/m}$ y las de $v_p,\bar v,v_{rms}$ son $\mathrm{m/s}$, ya que $kT/m=\mathrm{J/kg}=\mathrm{m^2/s^2}$. A igual temperatura, las moléculas ligeras son más rápidas, y a igual masa, el gas más caliente es más rápido.",
         [r"\boxed{\,f(v)=4\pi\left(\dfrac m{2\pi kT}\right)^{3/2}v^2e^{-mv^2/2kT}\,}",
          r"\boxed{\,v_p=\sqrt{\dfrac{2kT}m},\qquad \bar v=\sqrt{\dfrac{8kT}{\pi m}},\qquad v_{rms}=\sqrt{\dfrac{3kT}m}\,}",
          r"\sigma_{v_x}=\sqrt{\dfrac{kT}m},\qquad v_p<\bar v<v_{rms}"]),
        ("Particularización al problema",
         rf"Para $\mathrm{{N_2}}$ ($m={sci(m,4)}\ \mathrm{{kg}}$) a $T={T:.0f}\ \mathrm{{K}}$ se tiene $kT/m={sci(kb*T/m,4)}\ \mathrm{{m^2/s^2}}$, es decir, una desviación típica por componente de $\sigma={sigma:.1f}\ \mathrm{{m/s}}$ (que es la que se usa para generar el muestreo). Los valores de las rapideces coinciden con los de la escena. En el máximo la densidad vale $f(v_p)={sci(fp,4)}\ \mathrm{{s/m}}$.",
         [rf"a=\dfrac m{{2kT}}={sci(a,4)}\ \mathrm{{s^2/m^2}}",
          rf"v_p=\sqrt{{\dfrac{{2({sci(kb,5)})({T:.0f})}}{{{sci(m,4)}}}}}={vp:.1f}\ \mathrm{{m/s}},\qquad \bar v={vm:.1f}\ \mathrm{{m/s}},\qquad v_{{rms}}={vr:.1f}\ \mathrm{{m/s}}",
          rf"\dfrac{{\bar v}}{{v_p}}={vm/vp:.4f}\ \left(\dfrac2{{\sqrt\pi}}={2/np.sqrt(np.pi):.4f}\right),\qquad \dfrac{{v_{{rms}}}}{{v_p}}={vr/vp:.4f}\ \left(\sqrt{{3/2}}={np.sqrt(1.5):.4f}\right)",
          rf"P(E>10\,kT)\simeq\dfrac{{2}}{{\sqrt\pi}}\sqrt{{10}}\,e^{{-10}}={sci(2/np.sqrt(np.pi)*np.sqrt(10)*np.exp(-10),3)}\quad(E_0={sci(e0,3)}\ \mathrm{{J}})"]),
        ("Errores comunes y límites de validez",
         r"Es común confundir las tres rapideces o suponer que la rapidez media es la más probable; son distintas porque la distribución es asimétrica. También se confunde la distribución de la componente $v_x$ (gaussiana centrada en cero) con la de la rapidez (que no es gaussiana). Otro error es olvidar el factor $v^2$ al calcular promedios. La distribución es válida para un gas clásico en equilibrio térmico sin interacciones, con energía cinética no relativista; deja de valer cuando hay gases degenerados (se aplican las estadísticas de Fermi-Dirac o Bose-Einstein), cuando $kT\sim mc^2$ (se sustituye por la de Maxwell-Jüttner) y para gases fuera de equilibrio, como haces moleculares o plasmas con campos intensos.",
         []),
    ]
    return dict(concept=concept)


# =============================================================== CarnotCycle
def _carnot() -> dict:
    R, n, g = c25.R, c25.N_MOL, c25.GAMMA
    th, tc, va, vb = c25.TH, c25.TC, c25.V_A, c25.V_B
    st = c25.carnot_states()
    eta = c25.carnot_efficiency_analytic()
    qh = n * R * th * np.log(vb / va)
    qc = n * R * tc * np.log(vb / va)
    w = eta * qh
    vc, vd = st["v_c"], st["v_d"]
    ds = n * R * np.log(vb / va)
    eta_ca = 1 - np.sqrt(tc / th)
    concept = [
        ("Qué se estudia y por qué importa",
         "Toda máquina térmica toma calor de una fuente caliente, convierte parte en trabajo y cede el resto a una fuente fría. El ciclo de Carnot es el ciclo reversible más eficiente que puede operar entre dos temperaturas dadas, y su eficiencia, que sólo depende de ellas, es la cota superior que impone la naturaleza a todos los motores, plantas eléctricas y refrigeradores. Sirve de referencia para medir cuánto se pierde en un diseño real y para definir la escala termodinámica de temperatura.",
         []),
        ("Magnitudes y convenciones",
         r"Intervienen las temperaturas absolutas $T_h>T_c$ (K), los calores absorbido $Q_h$ y cedido $Q_c$ (J, ambos tomados como magnitudes positivas), el trabajo neto $W$ (J) y la eficiencia $\eta=W/Q_h$ (adimensional, entre 0 y 1). Los volúmenes de los vértices se denotan $V_A,\dots,V_D$ y el índice adiabático es $\gamma$. En una central térmica típica $T_h\approx800\ \mathrm{K}$ y $T_c\approx300\ \mathrm{K}$, para una cota de Carnot cercana al 60 %.",
         [r"\eta=\dfrac{W}{Q_h},\qquad W=Q_h-Q_c,\qquad 0\le\eta<1"]),
        ("Intuición y analogías",
         r"Como en una rueda hidráulica, no basta con que haya agua que baja: hace falta una diferencia de altura, que aquí es una diferencia de temperaturas. Se puede extraer más trabajo cuanto mayor es el desnivel relativo $\Delta T/T_h$. El ciclo de Carnot evita pérdidas: toda transferencia de calor se hace sin diferencia de temperatura apreciable (isotermas) y toda variación de temperatura sin transferencia de calor (adiabáticas), de modo que nada se disipa.",
         []),
        ("Las cuatro etapas y la primera ley",
         r"El ciclo consta de una expansión isoterma A→B a $T_h$ (absorbe $Q_h$), una expansión adiabática B→C que enfría hasta $T_c$, una compresión isoterma C→D a $T_c$ (cede $Q_c$) y una compresión adiabática D→A que cierra el ciclo. Al ser un ciclo, $\Delta U=0$, y la primera ley implica que el trabajo neto es la diferencia de calores. Todas las etapas son cuasiestáticas, por lo que el ciclo es reversible y puede recorrerse en sentido inverso como refrigerador.",
         [r"\oint dU=0\ \Rightarrow\ W=Q_h-Q_c",
          r"A\xrightarrow{T_h}B\xrightarrow{\ \delta Q=0\ }C\xrightarrow{T_c}D\xrightarrow{\ \delta Q=0\ }A"]),
        ("Calores en las isotermas",
         r"En una isoterma de gas ideal $\Delta U=0$, y por la primera ley el calor es igual al trabajo $\int p\,dV$ con $p=nRT/V$. Esto da un logaritmo de la razón de volúmenes en cada isoterma. El calor absorbido en la rama caliente y el cedido en la rama fría tienen el mismo tipo de expresión, con las temperaturas y los cocientes de volúmenes correspondientes.",
         [r"Q_h=nRT_h\ln\dfrac{V_B}{V_A},\qquad Q_c=nRT_c\ln\dfrac{V_C}{V_D}"]),
        ("Adiabáticas: relación entre los volúmenes",
         r"Para un gas ideal en una adiabática reversible vale $TV^{\gamma-1}=\text{cte}$. Aplicada a B→C y a D→A, con las mismas temperaturas extremas, da dos igualdades cuyo cociente muestra que las razones de volumen de las dos isotermas son iguales. Este es el paso clave: el logaritmo que aparece en $Q_h$ y en $Q_c$ es el mismo, y se cancela al hacer el cociente.",
         [r"T_hV_B^{\gamma-1}=T_cV_C^{\gamma-1},\qquad T_hV_A^{\gamma-1}=T_cV_D^{\gamma-1}",
          r"\dfrac{V_C}{V_B}=\dfrac{V_D}{V_A}=\left(\dfrac{T_h}{T_c}\right)^{1/(\gamma-1)}\ \Rightarrow\ \dfrac{V_C}{V_D}=\dfrac{V_B}{V_A}"]),
        ("Eficiencia de Carnot y su universalidad",
         r"Con la igualdad anterior, el cociente $Q_c/Q_h$ se reduce a $T_c/T_h$ y la eficiencia depende sólo de las temperaturas. Que valga para cualquier sustancia se demuestra por el teorema de Carnot: si una máquina reversible con otra sustancia tuviera mayor eficiencia, acoplada a una de Carnot funcionando como refrigerador produciría trabajo neto sin más efecto que enfriar un solo foco, violando el enunciado de Kelvin-Planck. Todas las máquinas reversibles entre los mismos focos comparten, entonces, la misma eficiencia, y las irreversibles tienen una menor.",
         [r"\dfrac{Q_c}{Q_h}=\dfrac{T_c}{T_h},\qquad \eta_C=1-\dfrac{Q_c}{Q_h}=1-\dfrac{T_c}{T_h}",
          r"\eta_{\text{irreversible}}<\eta_C<1\quad(T_c>0)"]),
        ("Interpretación entrópica y diagrama T-S",
         r"La relación $Q_h/T_h=Q_c/T_c$ equivale a $\oint\delta Q/T=0$, la igualdad de Clausius para un ciclo reversible, que lleva a definir la entropía como función de estado. En el plano $T$-$S$ el ciclo de Carnot es un rectángulo con base $\Delta S=Q_h/T_h$ y altura $T_h-T_c$, y su área, igual al trabajo neto, se lee directamente. Las isotermas son los lados horizontales y las adiabáticas reversibles, los verticales (isentrópicas).",
         [r"\oint\dfrac{\delta Q}{T}=\dfrac{Q_h}{T_h}-\dfrac{Q_c}{T_c}=0",
          r"W=\oint T\,dS=(T_h-T_c)\,\Delta S,\qquad \Delta S=nR\ln\dfrac{V_B}{V_A}"]),
        ("Límites y variantes",
         r"Los casos límite orientan el diseño: si $T_c\to0$ la eficiencia tiende a 1 pero es inalcanzable (tercera ley); si $T_c\to T_h$ el trabajo se anula. Una máquina que opera a máxima potencia, con transferencia de calor a ritmo finito, tiene una eficiencia menor, $\eta_{CA}=1-\sqrt{T_c/T_h}$ (Curzon-Ahlborn), más cercana a la de las centrales reales. Recorrido en sentido inverso, el ciclo es un refrigerador de coeficiente de operación $T_c/(T_h-T_c)$ y una bomba de calor de $T_h/(T_h-T_c)$.",
         [r"\mathrm{COP}_{ref}=\dfrac{Q_c}{W}=\dfrac{T_c}{T_h-T_c},\qquad \mathrm{COP}_{bc}=\dfrac{Q_h}{W}=\dfrac{T_h}{T_h-T_c}",
          r"\eta_{CA}=1-\sqrt{\dfrac{T_c}{T_h}}<\eta_C"]),
        ("Resultado general",
         r"El ciclo queda completamente caracterizado por las dos temperaturas y la razón de volúmenes de la isoterma caliente. La eficiencia es adimensional y el trabajo tiene unidades de energía ($nRT$ en julios).",
         [r"\boxed{\,\eta_C=1-\dfrac{T_c}{T_h},\qquad Q_h=nRT_h\ln\dfrac{V_B}{V_A},\qquad W=nR\,(T_h-T_c)\ln\dfrac{V_B}{V_A}\,}",
          r"Q_c=nRT_c\ln\dfrac{V_B}{V_A}=\dfrac{T_c}{T_h}Q_h,\qquad V_C=V_B\left(\dfrac{T_h}{T_c}\right)^{\frac1{\gamma-1}},\quad V_D=V_A\left(\dfrac{T_h}{T_c}\right)^{\frac1{\gamma-1}}"]),
        ("Particularización al problema",
         rf"Con $n=1$ mol, $T_h={th:.0f}\ \mathrm{{K}}$, $T_c={tc:.0f}\ \mathrm{{K}}$, $V_A={fx(va*1000,0)}\ \mathrm{{L}}$, $V_B={fx(vb*1000,0)}\ \mathrm{{L}}$ y $\gamma=5/3$ la razón de temperaturas elevada a $1/(\gamma-1)=3/2$ da $V_C={fx(vc*1000,4)}\ \mathrm{{L}}$ y $V_D={fx(vd*1000,4)}\ \mathrm{{L}}$. Los resultados reproducen los de la escena.",
         [rf"\eta=1-\dfrac{{{tc:.0f}}}{{{th:.0f}}}={eta:.3f}",
          rf"Q_h=nRT_h\ln2=({R})({th:.0f})(0.6931)={qh:.1f}\ \mathrm{{J}},\qquad Q_c={qc:.1f}\ \mathrm{{J}}",
          rf"W=Q_h-Q_c={qh-qc:.1f}\ \mathrm{{J}}=\eta Q_h={w:.1f}\ \mathrm{{J}}",
          rf"\Delta S=\dfrac{{Q_h}}{{T_h}}={ds:.4f}\ \mathrm{{J/K}},\qquad \dfrac{{Q_h}}{{T_h}}-\dfrac{{Q_c}}{{T_c}}={fx(qh/th-qc/tc,6)},\qquad \eta_{{CA}}={eta_ca:.3f}"]),
        ("Errores comunes y límites de validez",
         r"Es frecuente usar temperaturas en grados Celsius en $1-T_c/T_h$, lo que da un resultado incorrecto: deben ser absolutas. También se piensa que un motor real puede alcanzar la eficiencia de Carnot si se elimina la fricción; no es así porque transferir calor a ritmo finito exige diferencias de temperatura y por tanto irreversibilidad. Otro error es creer que la eficiencia depende del gas o de $V_B/V_A$; esa razón sólo cambia el trabajo por ciclo. El ciclo es una idealización: un ciclo reversible tarda un tiempo infinito, de modo que su potencia es cero, y los motores reales se comparan con modelos con irreversibilidad finita.",
         []),
    ]
    return dict(concept=concept)


# =============================================================== EntropyProduction
def _entropy() -> dict:
    m, c, t1, t2 = 1.0, 4186.0, 20.0, 80.0
    sa = c25.heating_entropy_analytic()
    sb = c25.contact_entropy_analytic()
    m1, c1_, t10 = c22.M1, c22.C1, c22.T1_0
    m2, c2_, t20 = c22.M2, c22.C2, c22.T2_0
    C1, C2 = m1 * c1_, m2 * c2_
    tfc = c22.final_temperature_analytic()
    tfk = tfc + 273.15
    s1 = C1 * np.log(tfk / (t10 + 273.15))
    s2 = C2 * np.log(tfk / (t20 + 273.15))
    h = c22.H
    sig0 = h * (t10 - t20) ** 2 / ((t10 + 273.15) * (t20 + 273.15))
    t1k, t2k = t1 + 273.15, t2 + 273.15
    s_foco = -m * c * (t2k - t1k) / t2k
    s_univ = sa + s_foco
    concept = [
        ("Qué se estudia y por qué importa",
         "El calor fluye espontáneamente del cuerpo caliente al frío, nunca al revés; una taza de café se enfría pero no se calienta sola. La entropía es la magnitud que cuantifica esa dirección privilegiada del tiempo: nunca disminuye en un sistema aislado. Medir cuánta entropía se produce en un proceso permite saber cuánta capacidad de producir trabajo se pierde y cuál es el límite de eficiencia de cualquier dispositivo.",
         []),
        ("Magnitudes y convenciones",
         r"La entropía $S$ se mide en $\mathrm{J/K}$; el cambio $\Delta S$ de un sistema y la entropía generada $S_{gen}$ (siempre no negativa) son conceptos distintos. Las temperaturas deben ir en kelvin dentro de $dQ/T$. Se denota $C=mc$ la capacidad calorífica, y $\delta Q_{rev}$ el calor intercambiado a lo largo de un camino reversible. Un sistema aislado cumple $\Delta S\ge0$, con igualdad sólo si el proceso es reversible.",
         [r"[S]=\mathrm{J\,K^{-1}},\qquad S_{gen}\ge0,\qquad \Delta S_{univ}=\Delta S_{sis}+\Delta S_{ent}"]),
        ("Intuición y analogías",
         r"La entropía mide el número de maneras microscópicas de realizar un estado macroscópico. Un mazo de cartas ordenada es una configuración entre muchísimas posibles: barajar la lleva casi siempre a una desordenada y es improbable que el azar la reordene. El calor que pasa del cuerpo caliente al frío dispersa la energía entre más moléculas, aumentando el número de configuraciones accesibles. Es una ley estadística: no prohíbe que la entropía baje, pero lo hace tan improbable que no se observa.",
         []),
        ("Segunda ley y entropía como función de estado",
         r"Los enunciados de Clausius (el calor no pasa solo de frío a caliente) y de Kelvin-Planck (no hay ciclo que convierta calor de un solo foco en trabajo) son equivalentes. De ellos se demuestra la desigualdad de Clausius para cualquier ciclo, con igualdad si es reversible. La igualdad implica que $\int\delta Q_{rev}/T$ entre dos estados no depende del camino reversible, y define una función de estado $S$. Para un proceso real irreversible se calcula $\Delta S$ con un camino reversible imaginario entre los mismos extremos.",
         [r"\oint\dfrac{\delta Q}{T}\le0\ \ (=0\text{ si es reversible})",
          r"dS=\dfrac{\delta Q_{rev}}{T},\qquad S_B-S_A=\int_A^B\dfrac{\delta Q_{rev}}{T}",
          r"\Delta S_{aislado}\ge0"]),
        ("Entropía de calentamiento y de un gas ideal",
         r"Para una sustancia de capacidad calorífica constante $C$ calentada de $T_1$ a $T_2$, el camino reversible pone el cuerpo en contacto con una sucesión de focos infinitamente próximos y $\delta Q_{rev}=C\,dT$. El resultado es el logaritmo del cociente de temperaturas absolutas, positivo si se calienta. Para un gas ideal la primera ley con $\delta W=p\,dV=nRT\,dV/V$ da una entropía que depende de $\ln T$ y de $\ln V$ y reproduce, por ejemplo, el resultado de la expansión libre.",
         [r"\Delta S=\int_{T_1}^{T_2}\dfrac{C\,dT}{T}=C\ln\dfrac{T_2}{T_1}\quad(C=mc\ \text{constante})",
          r"dS=nC_V\dfrac{dT}{T}+nR\dfrac{dV}{V}\ \Rightarrow\ \Delta S=nC_V\ln\dfrac{T_2}{T_1}+nR\ln\dfrac{V_2}{V_1}",
          r"\text{expansión libre: }\ \Delta S=nR\ln\dfrac{V_2}{V_1}>0"]),
        ("Producción de entropía en la conducción de calor",
         r"Si fluye un calor $\delta Q$ del cuerpo 1 al 2 a través de una conductancia, cada cuerpo, considerado en su propio equilibrio, cambia su entropía en $\mp\delta Q/T_i$. La suma es la entropía generada, que se escribe como la tasa de calor por la diferencia de inversos de temperatura. Con la ley de conducción $\dot Q=h(T_1-T_2)$ se obtiene una tasa que es una suma de cuadrados dividida entre positivos y por tanto no negativa. Se anula sólo en equilibrio, $T_1=T_2$.",
         [r"dS_1=-\dfrac{\delta Q}{T_1},\qquad dS_2=+\dfrac{\delta Q}{T_2}\ \Rightarrow\ dS=\delta Q\left(\dfrac1{T_2}-\dfrac1{T_1}\right)",
          r"\dot S_{gen}=h\,(T_1-T_2)\left(\dfrac1{T_2}-\dfrac1{T_1}\right)=\dfrac{h\,(T_1-T_2)^2}{T_1T_2}\ \ge0"]),
        ("Entropía total del contacto térmico",
         r"Integrar la tasa anterior sobre toda la relajación es equivalente a sumar los cambios de entropía de cada cuerpo entre su estado inicial y la temperatura final común, porque $S$ es función de estado. El resultado no depende de $h$. La positividad de la suma se demuestra con la concavidad del logaritmo: como $T_f$ es una media de las $T_i$ ponderada con los pesos $C_i/\sum C$, la desigualdad de Jensen da $\Delta S\ge0$, con igualdad sólo si todas las temperaturas iniciales coinciden.",
         [r"\Delta S=\int_0^\infty\dot S_{gen}\,dt=C_1\ln\dfrac{T_f}{T_{1}}+C_2\ln\dfrac{T_f}{T_{2}}",
          r"\dfrac{\Delta S}{C_1+C_2}=-\sum_iw_i\ln\dfrac{T_i}{T_f}\ge-\ln\sum_iw_i\dfrac{T_i}{T_f}=0,\qquad w_i=\dfrac{C_i}{C_1+C_2}"]),
        ("Interpretación estadística y trabajo perdido",
         r"En mecánica estadística la entropía es $S=k\ln\Omega$, con $\Omega$ el número de microestados compatibles con el macroestado; la segunda ley expresa que el sistema evoluciona hacia los macroestados con mayor $\Omega$. La entropía generada tiene además una lectura práctica (teorema de Gouy-Stodola): el trabajo que se deja de obtener respecto del proceso reversible es la temperatura del ambiente por la entropía generada. Por eso un contacto térmico con diferencia finita de temperaturas desperdicia capacidad de producir trabajo.",
         [r"S=k\ln\Omega,\qquad \Delta S=k\ln\dfrac{\Omega_f}{\Omega_i}",
          r"W_{perdido}=T_0\,S_{gen}"]),
        ("Resultado general",
         r"El cambio de entropía de un cuerpo de capacidad constante y la del contacto de varios cuerpos aislados se expresan con logaritmos de cocientes de temperaturas absolutas. Las expresiones son homogéneas: $C$ en $\mathrm{J/K}$ por un logaritmo adimensional da $\mathrm{J/K}$. La producción es siempre no negativa y es de segundo orden en la diferencia de temperaturas cuando ésta es pequeña.",
         [r"\boxed{\,\Delta S=C\ln\dfrac{T_2}{T_1},\qquad \Delta S_{contacto}=\sum_iC_i\ln\dfrac{T_f}{T_{i}}\ge0,\qquad T_f=\dfrac{\sum C_iT_i}{\sum C_i}\,}",
          r"\Delta S_{contacto}\simeq\dfrac{C_1C_2}{2(C_1+C_2)}\left(\dfrac{T_1-T_2}{\bar T}\right)^2\quad(|T_1-T_2|\ll\bar T)"]),
        ("Particularización al problema",
         rf"En (a), $C=mc={m*c:.0f}\ \mathrm{{J/K}}$ y las temperaturas absolutas son $T_1={t1k:.2f}\ \mathrm{{K}}$ y $T_2={t2k:.2f}\ \mathrm{{K}}$. En (b), las capacidades son $C_1={fx(C1,1)}\ \mathrm{{J/K}}$ (aluminio) y $C_2={fx(C2,1)}\ \mathrm{{J/K}}$ (agua), con $T_f={tfk:.2f}\ \mathrm{{K}}$. La entropía generada inicialmente por la conducción es $\dot S_{{gen}}={sig0:.3f}\ \mathrm{{W/K}}$ con $h=20\ \mathrm{{W/K}}$. Los resultados coinciden con los de la escena.",
         [rf"\Delta S_a=mc\ln\dfrac{{T_2}}{{T_1}}={m*c:.0f}\ln\dfrac{{{t2k:.2f}}}{{{t1k:.2f}}}={sa:.1f}\ \mathrm{{J/K}}",
          rf"\Delta S_1=C_1\ln\dfrac{{T_f}}{{T_{{10}}}}={fx(s1,3)}\ \mathrm{{J/K}},\qquad \Delta S_2=C_2\ln\dfrac{{T_f}}{{T_{{20}}}}=+{fx(s2,3)}\ \mathrm{{J/K}}",
          rf"\Delta S_b=\Delta S_1+\Delta S_2={sb:.2f}\ \mathrm{{J/K}}>0",
          rf"\text{{(a) con un foco a }}80\ ^\circ\mathrm{{C}}:\ \Delta S_{{foco}}={s_foco:.1f}\ \mathrm{{J/K}},\quad \Delta S_{{univ}}={s_univ:.1f}\ \mathrm{{J/K}}>0"]),
        ("Errores comunes y límites de validez",
         r"El error más frecuente es usar grados Celsius dentro del logaritmo. Otro es creer que la entropía de un sistema no puede disminuir: un cuerpo que se enfría pierde entropía, pero el entorno gana más, y lo que no puede disminuir es la del sistema aislado completo. También se confunde $\Delta S$ del sistema, que depende sólo de los estados extremos, con la entropía generada, que depende del proceso: un mismo $\Delta S_{sis}$ puede tener distinta $S_{gen}$. El cálculo supone capacidades constantes, equilibrio local dentro de cada cuerpo y un entorno aislado; cerca de transiciones de fase hay que añadir el término $L/T$, y para sistemas muy pequeños las fluctuaciones permiten decrementos transitorios (teoremas de fluctuación).",
         []),
    ]
    return dict(concept=concept)


DETAIL = {
    "MuonDecay": _muon(),
    "VelocityAndEnergy": _velocity(),
    "ThermalExpansion": _expansion(),
    "ThermalEquilibrium": _equilibrium(),
    "GasProcesses": _gas_processes(),
    "OttoCycle": _otto(),
    "GasPressure": _pressure(),
    "MaxwellBoltzmann": _maxwell(),
    "CarnotCycle": _carnot(),
    "EntropyProduction": _entropy(),
}
