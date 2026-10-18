"""Desarrollo conceptual y analitico GENERAL (campo `concept`) de: CoaxialLaplace, CapacitorNetwork, DrudeElectrons,
TaperedWire, MeshCircuit, RCCharging, HelicalMotion y HallEffect. Todas las cifras se calculan con `physics/`."""
from __future__ import annotations

import numpy as np

from content.texfmt import fx, sci
from physics import v2ch05_capacitance as e5
from physics import v2ch06_current as e6
from physics import v2ch07_circuits as e7
from physics import v2ch08_magnetic_field as m8
from physics.constants import E_CH, EPS0, K_B, M_E, M_P


# ------------------------------------------------------------------------------------------ CoaxialLaplace
def _coax() -> dict:
    a, b, v0 = e5.A_IN, e5.B_OUT, 1.0
    ln = float(np.log(b / a))
    c_an = e5.coax_capacitance_per_length_analytic()
    e_a = v0 / (a * ln)
    e_b = v0 / (b * ln)
    lam = c_an * v0
    sig = EPS0 * e_a
    u = 0.5 * c_an * v0**2
    concept = [
        ("Qué se estudia y por qué importa",
         r"Un cable coaxial es un par de conductores cilíndricos concéntricos separados por un aislante. Es la geometría de las líneas de transmisión de radiofrecuencia, de los cables de antena y de los sensores capacitivos, porque confina el campo eléctrico entre los dos conductores y lo aísla del exterior. La pregunta es cuánta carga por unidad de longitud almacena el cable por cada volt aplicado, es decir, su capacitancia por unidad de longitud $C'$. Esa magnitud fija, junto con la inductancia, la impedancia característica y la velocidad de propagación de las señales.",
         [r"C'=\dfrac{\lambda}{\Delta V}\quad[\mathrm{F/m}]"]),
        ("Magnitudes y convenciones",
         r"Se llama $a$ al radio del conductor interno y $b$ al radio interno del conductor externo; $\lambda$ es la carga por unidad de longitud del conductor interno (C/m) y $V_0$ su potencial respecto del externo, que se toma como referencia. La permitividad $\varepsilon$ del medio vale $\varepsilon_0=8.854\times10^{-12}$ F/m en el vacío o el aire, y $\kappa\varepsilon_0$ en un dieléctrico. Los cables coaxiales comerciales rondan 50-100 pF/m, de modo que el valor de decenas de pF por metro es el orden de magnitud correcto.",
         [r"[\lambda]=\mathrm{C/m},\qquad[E]=\mathrm{V/m},\qquad[\varepsilon_0]=\mathrm{F/m}=\mathrm{C^2/(N\,m^2)}"]),
        ("Intuición y analogías",
         r"Cada anillo de carga del conductor interno produce un campo radial que decae como $1/r$, igual que el flujo de un fluido incompresible que sale de una fuente lineal en un plano. Por eso el potencial varía con el logaritmo del radio y no linealmente: la mayor parte de la caída de tensión ocurre cerca del conductor interno, donde el campo es más intenso. El cable es también la versión cilíndrica del capacitor de placas, con la que coincide cuando el espesor $b-a$ es mucho menor que $a$.",
         [r"E(r)\propto\dfrac1r\ \Longrightarrow\ V(r)\propto\ln r"]),
        ("Punto de partida: electrostática en un dieléctrico lineal",
         r"Las ecuaciones de Maxwell para campos estáticos se reducen a $\nabla\times\vec E=0$ y $\nabla\cdot(\varepsilon\vec E)=\rho_{\ell}$. La primera garantiza que $\vec E$ es un gradiente, y se introduce el potencial escalar $V$. Con $\varepsilon$ uniforme, la segunda se convierte en la ecuación de Poisson, que en una región sin carga libre es la de Laplace. Las cargas del problema residen únicamente en las superficies de los conductores, que actúan como condiciones de contorno.",
         [r"\nabla\times\vec E=0\ \Rightarrow\ \vec E=-\nabla V",
          r"\nabla\cdot(\varepsilon\vec E)=\rho_\ell\ \Rightarrow\ \nabla^2V=-\dfrac{\rho_\ell}{\varepsilon}",
          r"\rho_\ell=0\ \text{(entre conductores)}\ \Rightarrow\ \nabla^2V=0"]),
        ("Simetría y reducción a una ecuación diferencial ordinaria",
         r"Si el cable es suficientemente largo frente a $b$ se desprecian los efectos de borde y el problema es invariante bajo traslaciones en $z$ y rotaciones en $\varphi$. Entonces $V=V(r)$ y el laplaciano cilíndrico conserva solo el término radial. La ecuación resultante se integra dos veces porque $r\,V'$ es constante, lo que da una combinación de una constante y un logaritmo.",
         [r"\nabla^2V=\dfrac1r\dfrac{d}{dr}\!\left(r\dfrac{dV}{dr}\right)=0",
          r"r\dfrac{dV}{dr}=A\ \Longrightarrow\ V(r)=A\ln r+B"]),
        ("Condiciones de contorno y unicidad",
         r"Las constantes $A$ y $B$ se fijan con $V(a)=V_0$ y $V(b)=0$. El teorema de unicidad garantiza que esta es la única solución de Laplace con esos valores en la frontera, de modo que no hace falta verificar nada más. Resolviendo el sistema lineal de dos ecuaciones se obtiene el potencial entre los conductores, que decrece monótonamente de $V_0$ a 0 y no tiene máximos ni mínimos interiores, como exige la propiedad del valor medio.",
         [r"A\ln a+B=V_0,\qquad A\ln b+B=0\ \Longrightarrow\ A=-\dfrac{V_0}{\ln(b/a)}",
          r"V(r)=V_0\,\dfrac{\ln(b/r)}{\ln(b/a)},\qquad a\le r\le b"]),
        ("Campo, densidad superficial y carga",
         r"El campo se obtiene derivando: $E_r=-dV/dr$. En la superficie de un conductor se cumple $\sigma=\varepsilon E_n$, que se integra sobre la circunferencia $2\pi a$ para hallar la carga por unidad de longitud. Se comprueba que la ley de Gauss se satisface para cualquier radio intermedio, porque $rE_r$ es constante, y que el conductor externo lleva exactamente $-\lambda$.",
         [r"E_r(r)=\dfrac{V_0}{\ln(b/a)}\dfrac1r",
          r"\sigma_a=\varepsilon E_r(a)=\dfrac{\varepsilon V_0}{a\ln(b/a)},\qquad\lambda=2\pi a\,\sigma_a=\dfrac{2\pi\varepsilon V_0}{\ln(b/a)}"]),
        ("Energía, principio variacional y relación con la relajación",
         r"La energía por unidad de longitud es la integral de la densidad $\tfrac12\varepsilon E^2$ sobre la sección. Al calcularla y compararla con $\tfrac12C'V_0^2$ se confirma el mismo valor de $C'$ obtenido por Gauss. La solución de Laplace es además el campo que minimiza esa energía entre todos los potenciales con las mismas condiciones de contorno (principio de Dirichlet). Esta propiedad justifica que el método de relajación converja: cada promedio de vecinos reduce la energía discretizada de la malla.",
         [r"U'=\dfrac{\varepsilon}{2}\int_a^b E_r^2\,2\pi r\,dr=\dfrac{\pi\varepsilon V_0^2}{\ln(b/a)}=\tfrac12C'V_0^2",
          r"\delta\!\int\dfrac{\varepsilon}{2}|\nabla V|^2\,dA=0\ \Longleftrightarrow\ \nabla^2V=0"]),
        ("Resultado general",
         r"Dividiendo la carga por unidad de longitud entre la diferencia de potencial, $V_0$ y todas las constantes de integración se cancelan: $C'$ depende solo de la geometría y del medio. La dependencia logarítmica implica que escalar ambos radios por un mismo factor no altera $C'$, y que duplicar $b/a$ solo lo reduce en una fracción moderada. Las dimensiones son $[\varepsilon]=\mathrm{F/m}$ y $\ln$ es adimensional, por lo que $C'$ se mide en F/m, como corresponde. Para $b/a\to1$ se recupera $C'\to\varepsilon\,2\pi a/(b-a)$, la capacitancia de placas paralelas por unidad de longitud.",
         [r"\boxed{\,C'=\dfrac{2\pi\varepsilon}{\ln(b/a)}=\dfrac{2\pi\kappa\varepsilon_0}{\ln(b/a)}\,}",
          r"\dfrac{1}{C'}=\dfrac{\ln(b/a)}{2\pi\varepsilon},\qquad Z_0=\dfrac{1}{2\pi}\sqrt{\dfrac{\mu}{\varepsilon}}\ln\dfrac ba"]),
        ("Particularización al problema",
         rf"Con $a={a*100:.1f}$ cm, $b={b*100:.1f}$ cm, $\kappa=1$ y $V_0=1$ V, el cociente $b/a$ es 4. Al sustituir en la fórmula general se recupera el valor analítico de la escena, que el método numérico reproduce con un error de discretización de pocos por ciento. Además se obtienen los campos en ambas superficies y la energía por metro.",
         [rf"C'=\dfrac{{2\pi({sci(EPS0,5)})}}{{\ln({fx(b/a,0)})}}=\dfrac{{{sci(2*np.pi*EPS0,5)}}}{{{ln:.5f}}}={c_an*1e12:.3f}\ \mathrm{{pF/m}}",
          rf"E(a)=\dfrac{{V_0}}{{a\ln(b/a)}}={fx(e_a,1)}\ \mathrm{{V/m}},\qquad E(b)={fx(e_b,1)}\ \mathrm{{V/m}}",
          rf"\lambda=C'V_0={sci(lam,4)}\ \mathrm{{C/m}},\qquad \sigma_a={sci(sig,4)}\ \mathrm{{C/m^2}},\qquad U'=\tfrac12C'V_0^2={sci(u,4)}\ \mathrm{{J/m}}"]),
        ("Errores comunes y límites de validez",
         r"Un error frecuente es creer que $C'$ depende de $V_0$ o de $\lambda$: ambos se cancelan porque el sistema es lineal. Otro es usar $a$ y $b$ como diámetros, o interpretar $\ln$ como logaritmo decimal, lo que cambia el resultado en un factor 2.3. El modelo falla cerca de los extremos del cable (efectos de borde), si el conductor interno es excéntrico (hay que usar coordenadas bipolares) o cuando la frecuencia es tan alta que la longitud de onda se acerca a las dimensiones del cable y la electrostática deja de valer. En una malla cartesiana, además, las fronteras circulares se aproximan por escalones y generan el error numérico observado.",
         [r"C'_{\text{excéntrico}}=\dfrac{2\pi\varepsilon}{\operatorname{arccosh}\!\left(\dfrac{a^2+b^2-d^2}{2ab}\right)},\qquad d=\text{excentricidad}"]),
    ]
    return dict(concept=concept)


# ----------------------------------------------------------------------------------------- CapacitorNetwork
def _network() -> dict:
    c1, c2, c3, v = e5.C1, e5.C2, e5.C3, e5.V_SRC
    ceq = e5.network_ceq_analytic()
    q = ceq * v
    vx = c1 * v / (c1 + c2 + c3)
    v1 = v - vx
    u = 0.5 * ceq * v**2
    kap = 4.0
    d = e5.dielectric_energies(kap, c1, v)
    c0 = c1
    w_bat = (kap - 1) * c0 * v**2
    concept = [
        ("Qué se estudia y por qué importa",
         r"Los capacitores casi nunca aparecen aislados: en filtros, fuentes de alimentación, memorias dinámicas y flashes de cámara se combinan en redes cuyo comportamiento global se resume en una capacitancia equivalente. Aquí se estudia cómo se reparten carga, tensión y energía en una red serie-paralelo y qué ocurre cuando se introduce un dieléctrico, con y sin batería conectada. La diferencia entre ambas situaciones es un ejemplo claro de cómo las condiciones de contorno, y no solo los componentes, determinan el resultado.",
         [r"C_{eq}=\dfrac{Q}{V},\qquad U=\tfrac12C_{eq}V^2"]),
        ("Magnitudes y convenciones",
         r"La capacitancia $C$ (faradio, F) relaciona la carga $Q$ de la placa positiva con la diferencia de potencial $V$ entre placas. La energía $U$ se mide en joules y la constante dieléctrica $\kappa\ge1$ es adimensional. En la práctica los valores van de picofaradios a faradios; los microfaradios del problema son típicos de circuitos de potencia. Se asigna signo a la carga de cada placa y se trabaja siempre con la magnitud $Q$ de la placa positiva.",
         [r"[C]=\mathrm{F}=\mathrm{C/V},\qquad[U]=\mathrm{J}=\mathrm{C\,V}"]),
        ("Intuición y analogías",
         r"En paralelo los capacitores comparten la tensión y es como aumentar el área de las placas: las capacitancias se suman. En serie comparten la carga y es como alejar las placas: el equivalente es menor que el menor de ellos. La analogía hidráulica es un conjunto de membranas elásticas en un tubo: el volumen que desplazan es la carga y la presión es la tensión. Un dieléctrico se comporta como un material que apantalla el campo, de manera que cabe más carga con la misma tensión.",
         [r"C_{par}\ge\max C_i,\qquad C_{ser}\le\min C_i"]),
        ("Dos leyes de conservación",
         r"En una red cuasiestática hay dos principios: la tensión es un potencial, por lo que la suma de caídas en cualquier lazo cerrado es nula, y la carga se conserva, por lo que la carga neta en cualquier conjunto de placas aislado del resto es constante. Estas son las versiones electrostáticas de las leyes de Kirchhoff. Como cada capacitor es lineal ($Q=CV$), la red completa es un sistema lineal cuyas incógnitas son los potenciales de los nodos.",
         [r"\oint\vec E\cdot d\vec\ell=0\ \Rightarrow\ \sum_{\text{lazo}}V_k=0",
          r"\sum_{\text{placas unidas a un nodo aislado}}Q_k=\text{cte}"]),
        ("Capacitores en paralelo y en serie",
         r"En paralelo, ambos capacitores tienen la misma tensión por estar unidos por conductores equipotenciales, y la carga total es la suma. En serie, la placa intermedia está aislada y era neutra, así que las cargas de sus caras opuestas son iguales y la carga es la misma en todos los capacitores; las tensiones se suman. Las dos fórmulas se obtienen eliminando las variables comunes en las ecuaciones de cada capacitor.",
         [r"Q=\sum_iC_iV=C_{par}V\ \Rightarrow\ C_{par}=\sum_iC_i",
          r"V=\sum_i\dfrac{Q}{C_i}\ \Rightarrow\ \dfrac1{C_{ser}}=\sum_i\dfrac1{C_i}"]),
        ("Red general mediante potenciales nodales",
         r"Si se llama $V_x$ al potencial del nodo común, la conservación de la carga en la placa aislada de $C_1$ y las placas de $C_2$ y $C_3$ unidas a él da una ecuación lineal en $V_x$. En una red con $N$ nodos libres el mismo procedimiento produce un sistema $\mathbf{K}\vec V=\vec b$, donde $\mathbf{K}$ es simétrica y definida positiva (análoga a una matriz de rigidez). La definición positiva garantiza solución única y se relaciona con que la energía $\tfrac12\vec V^{T}\mathbf K\vec V$ es positiva.",
         [r"C_1(V-V_x)=(C_2+C_3)V_x\ \Longrightarrow\ V_x=\dfrac{C_1}{C_1+C_2+C_3}\,V",
          r"K_{ii}=\sum_jC_{ij},\qquad K_{ij}=-C_{ij}\ (i\ne j),\qquad \mathbf K\vec V=\vec b"]),
        ("Energía y reparto en la red",
         r"La energía total es la suma de las energías de cada capacitor y coincide con $\tfrac12C_{eq}V^2$, que es el trabajo realizado por la fuente menos lo que no se almacena (en un proceso ideal, la mitad se disipa en resistencias de carga). Al estar $C_1$ en serie con el paralelo, a la rama de menor capacitancia le corresponde la mayor caída de tensión. Ese criterio de reparto $V_i\propto1/C_i$ en serie y $Q_i\propto C_i$ en paralelo sirve para estimar tensiones sin resolver todo el sistema.",
         [r"U=\sum_i\tfrac12C_iV_i^2=\tfrac12C_{eq}V^2",
          r"V_1=\dfrac{C_2+C_3}{C_1+C_2+C_3}\,V,\qquad Q_2=C_2V_x,\quad Q_3=C_3V_x"]),
        ("Dieléctrico: polarización y constante dieléctrica",
         r"Un dieléctrico lineal se polariza en respuesta al campo y las cargas ligadas superficiales generan un campo opuesto que reduce el campo neto. En un capacitor de placas con el dieléctrico llenándolo, el campo libre $\vec D=\varepsilon\vec E$ se mantiene fijado por la carga libre, mientras que $E$ se reduce en $\kappa$. Para la misma carga la tensión cae en $\kappa$, por lo que la capacitancia aumenta en $\kappa$.",
         [r"\vec D=\varepsilon_0\vec E+\vec P=\kappa\varepsilon_0\vec E,\qquad E=\dfrac{E_0}{\kappa}",
          r"C=\dfrac{Q}{V}=\dfrac{Q}{E\,d}=\kappa\,C_0"]),
        ("Dieléctrico con batería o aislado",
         r"Con la batería conectada $V$ permanece fija: la carga aumenta a $\kappa Q_0$ y la energía crece en $\kappa$; la batería aporta $(\kappa-1)C_0V^2$, de los cuales la mitad aumenta la energía almacenada y la otra mitad es el trabajo de atracción sobre el dieléctrico. Con el capacitor aislado $Q$ queda fija: la tensión cae en $\kappa$ y la energía disminuye en el mismo factor, es decir, el dieléctrico es atraído hacia el interior y esa diferencia se convierte en trabajo mecánico. En ambos casos el signo del trabajo coincide con el de una fuerza que tiende a introducir el material.",
         [r"V\ \text{fijo}:\ U_V=\tfrac12\kappa C_0V^2=\kappa U_0,\qquad W_{bat}=\Delta Q\,V=(\kappa-1)C_0V^2",
          r"Q\ \text{fijo}:\ U_Q=\dfrac{Q^2}{2\kappa C_0}=\dfrac{U_0}{\kappa},\qquad F=-\left.\dfrac{\partial U}{\partial x}\right|_Q=+\left.\dfrac{\partial U}{\partial x}\right|_V"]),
        ("Resultado general",
         r"Para la topología serie-paralelo estudiada la capacitancia equivalente es una función racional homogénea de grado 1 en las capacitancias, con unidades de faradio. Si todas las capacitancias se multiplican por un factor, $C_{eq}$ también, por lo que la carga y la energía escalan igual. La verificación dimensional es inmediata: $C_1C_{23}/(C_1+C_{23})$ tiene unidades de F. Los límites $C_1\to\infty$ y $C_1\to0$ dan $C_{eq}\to C_2+C_3$ y $C_{eq}\to0$.",
         [r"\boxed{\,C_{eq}=\dfrac{C_1(C_2+C_3)}{C_1+C_2+C_3}\,}",
          r"Q=C_{eq}V,\qquad U=\tfrac12C_{eq}V^2,\qquad U_V=\kappa U_0,\quad U_Q=\dfrac{U_0}{\kappa}"]),
        ("Particularización al problema",
         rf"Con $C_1={c1*1e6:.0f}\ \mu$F, $C_2={c2*1e6:.0f}\ \mu$F, $C_3={c3*1e6:.0f}\ \mu$F y $V={v:.0f}$ V, se evalúa la fórmula general. Las tensiones y cargas de cada rama se recuperan con los criterios de reparto y se obtienen las energías con dieléctrico $\kappa={kap:.0f}$ sobre el capacitor de $C_0={c0*1e6:.0f}\ \mu$F.",
         [rf"C_{{eq}}=\dfrac{{({c1*1e6:.0f})({(c2+c3)*1e6:.0f})}}{{{(c1+c2+c3)*1e6:.0f}}}={ceq*1e6:.4f}\ \mu\mathrm{{F}},\qquad Q={q*1e6:.3f}\ \mu\mathrm{{C}},\qquad U={u*1e6:.2f}\ \mu\mathrm{{J}}",
          rf"V_x=\dfrac{{{c1*1e6:.0f}}}{{{(c1+c2+c3)*1e6:.0f}}}({v:.0f})={vx:.4f}\ \mathrm{{V}},\qquad V_1={v1:.4f}\ \mathrm{{V}}",
          rf"U_0=\tfrac12C_0V^2={d['u0']*1e6:.1f}\ \mu\mathrm{{J}},\quad U_V={d['u_const_v']*1e6:.1f}\ \mu\mathrm{{J}},\quad U_Q={d['u_const_q']*1e6:.1f}\ \mu\mathrm{{J}},\quad W_{{bat}}={w_bat*1e6:.0f}\ \mu\mathrm{{J}}"]),
        ("Errores comunes y límites de validez",
         r"Es frecuente invertir las reglas: sumar capacitancias en serie o tomar la suma de inversas en paralelo. También se confunde la carga del equivalente con la de cada elemento en serie: la carga total $Q$ pasa por $C_1$, pero se reparte entre $C_2$ y $C_3$ en proporción a sus capacitancias. Otro error es olvidar si el dieléctrico se inserta con la batería conectada o sin ella, lo que cambia el sentido en que varía la energía. El modelo ideal deja de valer con efectos de borde, fugas, tensión de ruptura dieléctrica ($E<E_{ruptura}$), frecuencias en que importa la inductancia de los terminales, o dieléctricos no lineales como los ferroeléctricos.",
         [r"Q_2+Q_3=Q_1,\qquad\dfrac{Q_2}{Q_3}=\dfrac{C_2}{C_3}"]),
    ]
    return dict(concept=concept)


# ----------------------------------------------------------------------------------------- DrudeElectrons
def _drude() -> dict:
    n, tau, e_f = e6.N_CU, e6.TAU, e6.E_FIELD
    mu = E_CH * tau / M_E
    vd = mu * e_f
    sig = n * E_CH**2 * tau / M_E
    rho = 1 / sig
    j = n * E_CH * vd
    vth = float(np.sqrt(3 * K_B * e6.T_K / M_E))
    mfp = vth * tau
    concept = [
        ("Qué se estudia y por qué importa",
         r"La ley de Ohm es una regla empírica; el modelo de Drude (1900) la explica tratando a los electrones de conducción como un gas de partículas clásicas que aceleran con el campo y chocan con la red. Es el primer modelo microscópico de la conductividad y, a pesar de sus defectos, predice correctamente la forma de la ley de Ohm, la conducción en corriente alterna, el efecto Hall y la proporcionalidad entre conductividad térmica y eléctrica. Sigue siendo la base cualitativa para entender resistencia, movilidad y portadores en semiconductores.",
         [r"\vec J=\sigma\vec E"]),
        ("Magnitudes y convenciones",
         r"Los símbolos son: $n$ densidad de electrones libres (m$^{-3}$), $\tau$ tiempo medio entre colisiones (s), $\vec v_d$ velocidad de deriva (m/s), $\vec J$ densidad de corriente (A/m$^2$), $\sigma$ conductividad (S/m) y $\mu$ movilidad (m$^2$/(V s)). Por convención la corriente tiene el sentido de las cargas positivas, de modo que los electrones se desplazan en sentido opuesto a $\vec E$ y $\vec J$ es paralela a $\vec E$. En metales $n\sim10^{28}$-$10^{29}$ m$^{-3}$ y $\tau\sim10^{-14}$ s.",
         [r"[\sigma]=\mathrm{S/m}=\mathrm{A/(V\,m)},\qquad[\mu]=\mathrm{m^2/(V\,s)}"]),
        ("Intuición y analogías",
         r"Un electrón de conducción se mueve con velocidades térmicas enormes (del orden de $10^5$-$10^6$ m/s) pero en direcciones aleatorias, así que su promedio es nulo. El campo agrega un sesgo diminuto: entre choques el electrón acelera de manera uniforme y en cada choque pierde la memoria de su velocidad. La imagen es la de una bola de pinball que rebota en clavijas mientras la mesa está ligeramente inclinada: avanza lento hacia abajo aunque su rapidez instantánea sea alta.",
         [r"v_d\ll v_{térm}"]),
        ("Hipótesis del modelo",
         r"Se supone que (i) entre colisiones el electrón obedece la mecánica newtoniana bajo la fuerza $-e\vec E$, sin interacciones con otros electrones ni con los iones; (ii) las colisiones son instantáneas y ocurren con probabilidad $dt/\tau$ en cada intervalo $dt$, sin memoria; (iii) tras cada colisión la velocidad es aleatoria con promedio nulo, y no depende del campo. Estas hipótesis definen un proceso de Poisson. El campo $\vec E$ se supone uniforme y estático.",
         [r"m\dfrac{d\vec v}{dt}=-e\vec E\quad\text{(entre colisiones)},\qquad P(\text{choque en }dt)=\dfrac{dt}{\tau}"]),
        ("Distribución de tiempos libres",
         r"Sea $P(s)$ la probabilidad de que un electrón no haya chocado en un intervalo $s$. La probabilidad de sobrevivir un $ds$ adicional es $1-ds/\tau$, de donde $P'(s)=-P(s)/\tau$ y la solución es una exponencial. Como el proceso no tiene memoria, el tiempo transcurrido desde el último choque tiene la misma distribución que el tiempo hasta el próximo, y su media es $\tau$.",
         [r"P(s+ds)=P(s)\left(1-\dfrac{ds}{\tau}\right)\ \Rightarrow\ P(s)=e^{-s/\tau}",
          r"\langle s\rangle=\int_0^\infty s\,\dfrac{e^{-s/\tau}}{\tau}\,ds=\tau"]),
        ("Velocidad de deriva",
         r"Desde su último choque, un electrón ha acelerado durante un tiempo $s$ con $\vec a=-e\vec E/m$, de modo que su velocidad es $\vec v_0-e\vec E s/m$. Al promediar sobre electrones, $\langle\vec v_0\rangle=0$ por la hipótesis (iii), y solo queda el término de aceleración multiplicado por $\langle s\rangle=\tau$. Es el resultado central: la deriva es proporcional al campo.",
         [r"\vec v_d=\langle\vec v\rangle=\langle\vec v_0\rangle-\dfrac{e\vec E}{m}\langle s\rangle=-\dfrac{e\tau}{m}\vec E",
          r"\vec v_d=-\mu\vec E,\qquad\mu=\dfrac{e\tau}{m}"]),
        ("Ecuación de balance de cantidad de movimiento",
         r"El mismo resultado sigue de promediar el momento: en $dt$ una fracción $dt/\tau$ de los electrones pierde su momento medio, y el campo aporta $-e\vec E\,dt$. Esto da una ecuación lineal con un término de fricción efectiva, cuya solución muestra que el régimen estacionario se alcanza en un tiempo del orden de $\tau\sim10^{-14}$ s, prácticamente instantáneo en circuitos de corriente continua.",
         [r"\dfrac{d\langle\vec p\rangle}{dt}=-e\vec E-\dfrac{\langle\vec p\rangle}{\tau}",
          r"\langle\vec p\rangle(t)=-e\tau\vec E\left(1-e^{-t/\tau}\right)"]),
        ("Ley de Ohm microscópica y conductividad",
         r"La densidad de corriente es la carga que atraviesa la unidad de área por unidad de tiempo: $n$ portadores de carga $-e$ con velocidad media $\vec v_d$. Sustituyendo la deriva se obtiene $\vec J\propto\vec E$, que es la ley de Ohm local con una conductividad que depende de los parámetros microscópicos. Integrando sobre un conductor de longitud $L$ y sección $A$ se recupera $V=IR$ con $R=L/(\sigma A)$.",
         [r"\vec J=n(-e)\vec v_d=\dfrac{ne^2\tau}{m}\vec E=\sigma\vec E",
          r"\sigma=\dfrac{ne^2\tau}{m}=ne\mu,\qquad\rho=\dfrac1\sigma=\dfrac{m}{ne^2\tau}"]),
        ("Resultado general",
         r"La velocidad de deriva es lineal en el campo, con constante de proporcionalidad la movilidad. La conductividad crece con la densidad de portadores y con el tiempo de relajación, y es independiente del campo mientras el modelo sea lineal. Dimensionalmente $e\tau/m$ tiene unidades C s/kg $=$ m$^2$/(V s), y $ne\mu$ tiene unidades C/m$^3\cdot$m$^2$/(V s) $=$ S/m. Una consecuencia es que el tiempo de colisión puede estimarse a partir de la resistividad medida.",
         [r"\boxed{\,v_d=\dfrac{e\tau}{m}E,\qquad\sigma=\dfrac{ne^2\tau}{m},\qquad\mu=\dfrac{v_d}{E}=\dfrac{e\tau}{m}\,}",
          r"\tau=\dfrac{m}{\rho ne^2}"]),
        ("Particularización al problema",
         rf"Para cobre con $n={sci(n,3)}\ \mathrm{{m^{{-3}}}}$, $\tau={sci(tau,3)}$ s y $E={e_f:.0f}$ V/m, se evalúan las fórmulas generales. La movilidad coincide con la que la simulación de Monte Carlo busca verificar (a un campo mayor, pues el modelo es lineal). La rapidez de deriva resulta mucho menor que la térmica, como pide la hipótesis del modelo.",
         [rf"\mu=\dfrac{{({sci(E_CH,5)})({sci(tau,3)})}}{{{sci(M_E,5)}}}={sci(mu,4)}\ \mathrm{{m^2/(V\,s)}}",
          rf"v_d=\mu E={sci(vd,4)}\ \mathrm{{m/s}},\qquad\sigma=ne\mu={sci(sig,4)}\ \mathrm{{S/m}},\qquad\rho={sci(rho,4)}\ \Omega\cdot\mathrm{{m}}",
          rf"J=nev_d={sci(j,3)}\ \mathrm{{A/m^2}},\qquad v_{{térm}}=\sqrt{{3k_BT/m}}={sci(vth,3)}\ \mathrm{{m/s}},\qquad\dfrac{{v_d}}{{v_{{térm}}}}={sci(vd/vth,2)}"]),
        ("Errores comunes y límites de validez",
         rf"No debe confundirse la velocidad de deriva ({sci(vd,2)} m/s, menos de un metro por segundo) con la velocidad con que se propaga la señal, cercana a la de la luz, ni con la velocidad térmica. El tiempo $\tau$ no es un tiempo fijo entre choques, sino la media de una distribución exponencial. El modelo clásico falla porque los electrones obedecen la estadística de Fermi-Dirac: solo participan los cercanos a la superficie de Fermi ($v_F\sim10^6$ m/s), y el camino libre real en cobre a 300 K es de decenas de nanómetros y no de unos pocos, como daría $v_{{térm}}\tau={mfp*1e9:.1f}$ nm. Además $\tau$ depende de la temperatura y de las impurezas (regla de Matthiessen).",
         [r"\rho(T)=\rho_{\text{imp}}+\rho_{\text{fonones}}(T),\qquad\ell=v_F\tau"]),
    ]
    return dict(concept=concept)


# ------------------------------------------------------------------------------------------ TaperedWire
def _tapered() -> dict:
    rho, L, r1, r2 = e6.RHO, e6.LENGTH, e6.R1, e6.R2
    ra = e6.tapered_resistance_analytic()
    ru = e6.uniform_resistance()
    k = (r2 - r1) / L
    mid = r1 + k * L / 2
    tan2 = k**2
    area1, area2 = np.pi * r1**2, np.pi * r2**2
    concept = [
        ("Qué se estudia y por qué importa",
         r"En un conductor real la sección rara vez es constante: conos de conexión, hilos estirados, pistas de circuito impreso que se ensanchan o electrodos de punta. La resistencia determina las pérdidas $I^2R$, el calentamiento y la caída de tensión, y por eso es necesario saber calcularla cuando el área cambia a lo largo del recorrido. El problema muestra cómo una geometría variable se resuelve integrando la contribución de rebanadas infinitesimales conectadas en serie.",
         [r"R=\dfrac{\Delta V}{I}"]),
        ("Magnitudes y convenciones",
         r"La resistividad $\rho$ (Ω·m) es una propiedad del material; la resistencia $R$ (Ω) depende además de la geometría. $I$ es la corriente total (A), $\vec J$ la densidad de corriente (A/m$^2$) y $A(x)$ el área de la sección transversal a la distancia $x$ del extremo. El valor de $\rho$ del cobre a temperatura ambiente es del orden de $10^{-8}$ Ω·m, de modo que los cables de unos metros tienen resistencias de miliohmios.",
         [r"[\rho]=\Omega\cdot\mathrm{m},\qquad[R]=\Omega=\mathrm{V/A}"]),
        ("Intuición y analogías",
         r"La corriente es como agua en una tubería de diámetro variable: el caudal es el mismo en todas las secciones, pero la velocidad es mayor donde el tubo es estrecho y es allí donde ocurre la mayor parte de la caída de presión. Análogamente, la mayor parte de la resistencia del cono está en el extremo delgado, donde $J$ y $E$ son más grandes. El cable cónico no equivale al de radio promedio, porque la resistencia depende de $1/r^2$ y es dominada por las zonas delgadas.",
         [r"J(x)=\dfrac{I}{A(x)},\qquad E(x)=\rho J(x)"]),
        ("Ecuaciones de partida",
         r"En régimen estacionario la carga no se acumula, de modo que $\nabla\cdot\vec J=0$ en el interior, y en la superficie lateral $\vec J\cdot\hat n=0$ porque la corriente no escapa del conductor. En un material óhmico $\vec J=\vec E/\rho$ y el campo es conservativo, de modo que el potencial satisface la ecuación de Laplace con condiciones de aislamiento en la pared y de potencial fijo en los extremos. Esa es la formulación exacta del problema.",
         [r"\nabla\cdot\vec J=0,\qquad\vec J=\dfrac{1}{\rho}\vec E=-\dfrac1\rho\nabla V",
          r"\nabla^2V=0,\qquad\left.\dfrac{\partial V}{\partial n}\right|_{\text{pared}}=0,\qquad V(0)-V(L)=\Delta V"]),
        ("Aproximación de conductor delgado",
         r"Si el radio varía lentamente, $|dr/dx|\ll1$, las líneas de corriente son casi paralelas al eje y las secciones son casi equipotenciales. Entonces la corriente total es la misma en cada sección y $J$ es aproximadamente uniforme dentro de cada una. La conservación de la corriente es la integral de $\nabla\cdot\vec J=0$ sobre el volumen entre dos secciones, y permite escribir $J(x)=I/A(x)$. El error de esta aproximación es del orden de $(dr/dx)^2$.",
         [r"\iint_{A(x)}\vec J\cdot d\vec A=I=\text{cte}\ \Rightarrow\ J(x)\simeq\dfrac{I}{A(x)}",
          r"\varepsilon_{rel}\sim\left(\dfrac{dr}{dx}\right)^2"]),
        ("Integración de la caída de potencial",
         r"Cada rebanada de espesor $dx$ es un resistor elemental $dR=\rho\,dx/A(x)$ y todas están conectadas en serie, por lo que sus resistencias se suman. Equivalentemente, se integra $E=\rho J$ a lo largo del eje. La resistencia total es la integral de $\rho/A(x)$ y depende del perfil $A(x)$ completo.",
         [r"dV=-E\,dx=-\dfrac{\rho I}{A(x)}\,dx",
          r"R=\dfrac{V(0)-V(L)}{I}=\int_0^L\dfrac{\rho\,dx}{A(x)}"]),
        ("Perfil lineal del radio",
         r"Para un cono truncado $r(x)=r_1+kx$ con $k=(r_2-r_1)/L$ el integrando es $\rho/(\pi r^2)$. Con el cambio de variable $u=r(x)$ se tiene $dx=du/k$, y la integral de $u^{-2}$ es elemental. Al evaluar entre $r_1$ y $r_2$ y sustituir $k$, $L$ aparece multiplicando y $r_2-r_1$ se cancela, lo que da una forma compacta y simétrica en los radios.",
         [r"R=\dfrac{\rho}{\pi}\int_0^L\dfrac{dx}{(r_1+kx)^2}=\dfrac{\rho}{\pi k}\left[\dfrac{1}{r_1}-\dfrac{1}{r_2}\right]",
          r"R=\dfrac{\rho L}{\pi(r_2-r_1)}\cdot\dfrac{r_2-r_1}{r_1r_2}=\dfrac{\rho L}{\pi r_1r_2}"]),
        ("Casos límite y parámetro adimensional",
         r"Si $r_2\to r_1$ se recupera la fórmula del cable uniforme $\rho L/(\pi r_1^2)$, lo que verifica la deducción. El cociente $R_{cono}/R_{unif}=r_1/r_2$ muestra cómo la resistencia disminuye al ensancharse un extremo, pero siempre es mayor que la de un cable uniforme del radio mayor. La validez de la aproximación se mide con el parámetro adimensional $\tan\theta=k$ (semiángulo del cono). Por el principio de Rayleigh, imponer secciones equipotenciales equivale a añadir cortocircuitos entre ellas, que solo pueden reducir la resistencia: la fórmula unidimensional es, a este orden, una cota inferior de la resistencia exacta.",
         [r"\dfrac{R_{cono}}{R_{unif}(r_1)}=\dfrac{r_1}{r_2},\qquad\lim_{r_2\to r_1}R=\dfrac{\rho L}{\pi r_1^2}",
          r"\tan\theta=\dfrac{r_2-r_1}{L}\ll1"]),
        ("Resultado general",
         r"La resistencia depende de la resistividad, la longitud y el producto de los radios extremos, y es independiente de $I$ y $V$ mientras $\rho$ sea constante. Tiene dimensiones $\Omega\cdot\mathrm{m}\cdot\mathrm{m}/\mathrm{m}^2=\Omega$. Para un perfil arbitrario $A(x)$ la fórmula general es la integral de línea; para el perfil lineal se cierra de forma exacta. La caída de potencial acumulada hasta $x$ es $IR(x)$, con $R(x)$ la integral parcial.",
         [r"\boxed{\,R=\int_0^L\dfrac{\rho\,dx}{A(x)}\ \xrightarrow{\ r\text{ lineal}\ }\ R=\dfrac{\rho L}{\pi r_1r_2}\,}",
          r"V(x)=V(0)-I\int_0^x\dfrac{\rho\,dx'}{A(x')}"]),
        ("Particularización al problema",
         rf"Con $\rho={sci(rho,3)}\ \Omega\cdot\mathrm{{m}}$, $L={L:.0f}$ m, $r_1={r1*1e3:.1f}$ mm y $r_2={r2*1e3:.1f}$ mm, se sustituye en las fórmulas generales y se comparan con el cable uniforme de radio $r_1$. La pendiente del radio es muy pequeña, lo que justifica la aproximación de conductor delgado.",
         [rf"R=\dfrac{{({sci(rho,3)})({L:.0f})}}{{\pi({sci(r1,3)})({sci(r2,3)})}}={ra*1e3:.4f}\ \mathrm{{m\Omega}}",
          rf"R_{{unif}}=\dfrac{{\rho L}}{{\pi r_1^2}}={ru*1e3:.4f}\ \mathrm{{m\Omega}},\qquad\dfrac{{R}}{{R_{{unif}}}}=\dfrac{{r_1}}{{r_2}}={r1/r2:.4f}",
          rf"k={sci(k,3)},\qquad k^2={sci(tan2,2)}\ \ll1,\qquad R(r\!=\!\bar r)=\dfrac{{\rho L}}{{\pi\bar r^2}}={rho*L/(np.pi*mid**2)*1e3:.4f}\ \mathrm{{m\Omega}}\ (\bar r={mid*1e3:.1f}\ \mathrm{{mm}})"]),
        ("Errores comunes y límites de validez",
         rf"Es común usar el radio medio ($\bar r={mid*1e3:.1f}$ mm) como si el cable fuera uniforme; esto subestima la resistencia porque la integral pondera más las secciones delgadas (da {rho*L/(np.pi*mid**2)*1e3:.2f} mΩ en lugar de {ra*1e3:.2f} mΩ). Otro error es promediar las áreas o los radios en lugar de integrar $1/A$. La aproximación unidimensional falla en conos muy abiertos, en el contacto de un conductor con un electrodo puntual (resistencia de constricción, que diverge como $1/r$) y a altas frecuencias, donde el efecto pelicular concentra la corriente cerca de la superficie. También se supone $\rho$ constante, lo que no es cierto si el cable se calienta de forma no uniforme.",
         [r"R_{\bar r}=\dfrac{\rho L}{\pi\bar r^2}<R\ \ (\bar r=\tfrac12(r_1+r_2)),\qquad\delta=\sqrt{\dfrac{2\rho}{\omega\mu_0}}\ \text{(profundidad pelicular)}"]),
    ]
    return dict(concept=concept)


# ------------------------------------------------------------------------------------------- MeshCircuit
def _mesh() -> dict:
    e1, e2, r1, r2, r3 = e7.E1, e7.E2, e7.R1, e7.R2, e7.R3
    i = e7.mesh_currents_analytic()
    br = e7.branch_currents(i)
    det = r1 * r2 + r1 * r3 + r2 * r3
    p_src = e1 * i[0] - e2 * i[1]
    p_res = r1 * i[0] ** 2 + r2 * i[1] ** 2 + r3 * (i[0] - i[1]) ** 2
    concept = [
        ("Qué se estudia y por qué importa",
         r"Casi cualquier aparato eléctrico contiene redes de fuentes y resistencias con más de un lazo. Las leyes de Kirchhoff permiten calcular las corrientes en todas las ramas a partir de las fuerzas electromotrices y las resistencias, y son la base del análisis de circuitos, de los simuladores como SPICE y del diseño de redes eléctricas. El problema de dos mallas es el caso mínimo en que aparece un elemento compartido y se hace necesario plantear un sistema de ecuaciones.",
         [r"\sum I_{nodo}=0,\qquad\sum\Delta V_{lazo}=0"]),
        ("Magnitudes y convenciones",
         r"Las corrientes de malla $i_k$ (A) son corrientes ficticias que circulan por cada lazo en sentido horario; las corrientes de rama son combinaciones de ellas. Una fuente se cuenta como $+\varepsilon$ al recorrerla de menos a más y una resistencia como $-IR$ al recorrerla en el sentido de la corriente. Un resultado negativo indica que el sentido real es opuesto al supuesto, no un error. Las unidades son voltio, ampere y ohmio, con $\mathrm{V}=\mathrm{A}\,\Omega$.",
         [r"[\varepsilon]=\mathrm{V},\qquad[R]=\Omega,\qquad[P]=\mathrm{W}=\mathrm{V\,A}=\Omega\,\mathrm{A^2}"]),
        ("Intuición y analogías",
         r"Un circuito se parece a una red de tuberías con bombas (fuentes) y estrechamientos (resistencias): el agua no se acumula en las uniones, y al recorrer cualquier circuito cerrado la presión que ganan las bombas se pierde en los estrechamientos. La corriente de malla es una forma económica de describir la red: al suponer que cada lazo tiene su propio flujo, la conservación en los nodos queda satisfecha automáticamente. Solo hay que imponer la segunda ley en cada lazo independiente.",
         [r"\#\text{lazos independientes}=b-n+1"]),
        ("Fundamentos físicos de las leyes de Kirchhoff",
         r"La ley de los nodos expresa la conservación de la carga en régimen estacionario: de $\nabla\cdot\vec J=-\partial\rho/\partial t$ y $\partial\rho/\partial t=0$ se sigue que el flujo neto de corriente por cualquier superficie cerrada es nulo. La ley de las mallas es la forma integral de $\nabla\times\vec E=-\partial\vec B/\partial t$ cuando el flujo magnético que atraviesa el lazo es despreciable (circuito cuasiestático), es decir, el potencial es una función univaluada. En una fuente real el trabajo no electrostático por unidad de carga es la fem.",
         [r"\oint_S\vec J\cdot d\vec A=0\ \Rightarrow\ \sum_kI_k=0",
          r"\oint\vec E\cdot d\vec\ell=-\dfrac{d\Phi_B}{dt}\approx0\ \Rightarrow\ \sum_k\varepsilon_k-\sum_kI_kR_k=0"]),
        ("Planteamiento general por mallas",
         r"Para $N$ mallas con corrientes $i_1,\dots,i_N$, la segunda ley en el lazo $j$ da una ecuación lineal. El coeficiente diagonal $R_{jj}$ es la suma de todas las resistencias del lazo $j$, y el coeficiente $R_{jk}$ es menos la suma de las resistencias compartidas con el lazo $k$ (signo menos si ambas corrientes circulan en sentido horario, pues en la rama común van en direcciones opuestas). El lado derecho $\varepsilon_j$ suma las fem del lazo. El resultado es un sistema matricial simétrico.",
         [r"\sum_{k=1}^NR_{jk}\,i_k=\varepsilon_j,\qquad\mathbf R\,\vec i=\vec\varepsilon",
          r"R_{jj}=\sum_{\text{lazo }j}R,\qquad R_{jk}=-\sum_{\text{comunes}}R\ (j\ne k),\qquad\mathbf R=\mathbf R^T"]),
        ("Existencia y unicidad de la solución",
         r"La potencia disipada total es $P=\vec i^{\,T}\mathbf R\,\vec i$, y como cada resistencia aporta $R\,I^2\ge0$, la forma cuadrática es definida positiva si todas las resistencias son positivas. Esto implica $\det\mathbf R>0$ y por tanto solución única. Más aún, la corriente real es el mínimo de la función $\tfrac12\vec i^{\,T}\mathbf R\,\vec i-\vec i\cdot\vec\varepsilon$ (teorema del mínimo calor de Maxwell-Thomson), un principio variacional que garantiza estabilidad y que justifica los métodos iterativos.",
         [r"P=\vec i^{\,T}\mathbf R\,\vec i=\sum_kR_kI_k^2>0",
          r"\nabla_{\vec i}\!\left(\tfrac12\vec i^{\,T}\mathbf R\,\vec i-\vec i\cdot\vec\varepsilon\right)=0\ \Longleftrightarrow\ \mathbf R\,\vec i=\vec\varepsilon"]),
        ("Solución para dos mallas con resistencia compartida",
         r"Con $R_1$ y $\varepsilon_1$ en el lazo izquierdo, $R_2$ y $\varepsilon_2$ en el derecho (con polaridad opuesta en el sentido horario) y $R_3$ compartida, el sistema es de $2\times2$. Su determinante es la suma de los productos de resistencias tomadas de a dos, y se resuelve con la regla de Cramer. Aparecen sumas y diferencias de fem, cada una pesada por una resistencia.",
         [r"\begin{pmatrix}R_1+R_3&-R_3\\-R_3&R_2+R_3\end{pmatrix}\begin{pmatrix}i_1\\i_2\end{pmatrix}=\begin{pmatrix}\varepsilon_1\\-\varepsilon_2\end{pmatrix}",
          r"D=R_1R_2+R_1R_3+R_2R_3",
          r"i_1=\dfrac{\varepsilon_1(R_2+R_3)-\varepsilon_2R_3}{D},\qquad i_2=\dfrac{\varepsilon_1R_3-\varepsilon_2(R_1+R_3)}{D}"]),
        ("Balance de potencia y superposición",
         r"El trabajo de las fuentes por unidad de tiempo debe igualar el calor disipado: multiplicar cada ecuación de malla por su $i_j$ y sumar es una demostración del teorema de Tellegen. Además, la linealidad permite aplicar superposición: la corriente total es la suma de la que produciría cada fuente por separado con las demás reemplazadas por cortocircuitos. La corriente por la rama compartida es la diferencia $i_1-i_2$.",
         [r"\sum_j\varepsilon_ji_j=\vec i^{\,T}\mathbf R\,\vec i=\sum_kR_kI_k^2",
          r"I_{R_3}=i_1-i_2=\dfrac{\varepsilon_1R_2+\varepsilon_2R_1}{D}"]),
        ("Resultado general",
         r"Las corrientes de malla son funciones racionales de las fem y las resistencias, homogéneas de grado 1 en las fem y de grado $-1$ en las resistencias, de modo que sus dimensiones son V/Ω $=$ A. Si todas las resistencias se multiplican por un factor, las corrientes se dividen entre él. Si $R_3\to\infty$ la rama central se abre y ambas corrientes tienden a $(\varepsilon_1-\varepsilon_2)/(R_1+R_2)$, la de un lazo único; si $R_3\to0$ los lazos se desacoplan y $i_1\to\varepsilon_1/R_1$, $i_2\to-\varepsilon_2/R_2$.",
         [r"\boxed{\,\vec i=\mathbf R^{-1}\vec\varepsilon,\qquad I_{R_3}=\dfrac{\varepsilon_1R_2+\varepsilon_2R_1}{R_1R_2+R_1R_3+R_2R_3}\,}"]),
        ("Particularización al problema",
         rf"Con $\varepsilon_1={e1:.0f}$ V, $\varepsilon_2={e2:.0f}$ V, $R_1={r1:.0f}\ \Omega$, $R_2={r2:.0f}\ \Omega$ y $R_3={r3:.0f}\ \Omega$ se evalúan el determinante y las dos corrientes. Se verifica el balance de potencia entre la fuente y las resistencias.",
         [rf"D=({r1:.0f})({r2:.0f})+({r1:.0f})({r3:.0f})+({r2:.0f})({r3:.0f})={det:.0f}\ \Omega^2",
          rf"i_1=\dfrac{{{e1:.0f}({r2+r3:.0f})-{e2:.0f}({r3:.0f})}}{{{det:.0f}}}={i[0]:.4f}\ \mathrm{{A}},\qquad i_2=\dfrac{{{e1:.0f}({r3:.0f})-{e2:.0f}({r1+r3:.0f})}}{{{det:.0f}}}={i[1]:.4f}\ \mathrm{{A}}",
          rf"I_{{R_3}}={br['R3']:.4f}\ \mathrm{{A}},\qquad\sum\varepsilon i={p_src:.4f}\ \mathrm{{W}}=\sum RI^2={p_res:.4f}\ \mathrm{{W}}"]),
        ("Errores comunes y límites de validez",
         r"Los errores típicos son olvidar el signo de una fem cuya polaridad es opuesta al sentido de recorrido, contar la resistencia compartida con el mismo signo en los términos cruzados y diagonales, y confundir la corriente de malla con la de rama en la resistencia común. Las leyes de Kirchhoff dejan de ser válidas cuando el flujo magnético variable no es despreciable (inductancias, alta frecuencia, líneas largas, donde hay que usar ecuaciones de telegrafista) y cuando los componentes no son lineales; en este último caso el sistema se resuelve iterativamente (Newton-Raphson). Con fuentes reales hay que incluir su resistencia interna.",
         [r"\text{no lineal: }\ \mathbf F(\vec i)=\vec 0,\quad\vec i_{k+1}=\vec i_k-\mathbf J^{-1}\mathbf F(\vec i_k)"]),
    ]
    return dict(concept=concept)


# --------------------------------------------------------------------------------------------- RCCharging
def _rc() -> dict:
    eps, R, C = e7.EPS, e7.R, e7.C
    tau = R * C
    q_inf = C * eps
    i0 = eps / R
    t632 = -tau * np.log(1 - 0.632)
    t99 = -tau * np.log(0.01)
    u_half = 0.5 * C * eps**2
    l_crit = R**2 * C / 4
    concept = [
        ("Qué se estudia y por qué importa",
         r"La carga de un capacitor a través de una resistencia es el prototipo de los procesos de relajación exponencial: temporizadores, filtros paso bajo, circuitos de disparo, flashes, la membrana de una neurona y el rebote de señales digitales. Interesa saber cuánto tarda el capacitor en cargarse, cómo varía la corriente en el tiempo y a dónde va la energía suministrada por la fuente. Un resultado notable es que la resistencia disipa justo la mitad de la energía de la batería, sea cual sea su valor.",
         [r"\varepsilon=IR+\dfrac{q}{C}"]),
        ("Magnitudes y convenciones",
         r"La carga $q(t)$ de la placa positiva se mide en coulomb, la corriente $I=dq/dt$ en ampere, la capacitancia en faradios y la resistencia en ohmios. La constante de tiempo $\tau=RC$ tiene unidades de segundo, pues $\Omega\cdot\mathrm{F}=\mathrm{(V/A)(C/V)}=\mathrm{C/A}=\mathrm{s}$. Se supone el capacitor inicialmente descargado y el interruptor se cierra en $t=0$. Se toma $I>0$ cuando la carga de la placa positiva aumenta.",
         [r"[\tau]=\Omega\cdot\mathrm{F}=\mathrm{s}"]),
        ("Intuición y analogías",
         r"Cuando el capacitor está vacío toda la tensión de la fuente cae sobre la resistencia y la corriente es máxima. A medida que se carga, su propia tensión se opone a la fuente y la corriente disminuye, de modo que cada vez se carga más despacio. Es como llenar un depósito por un tubo estrecho: al principio el nivel sube rápido y al acercarse al nivel de la fuente casi no entra agua. Un proceso donde la velocidad es proporcional a lo que falta produce siempre una exponencial.",
         [r"\dfrac{dq}{dt}\propto q_\infty-q"]),
        ("Ecuación del circuito",
         r"La segunda ley de Kirchhoff en el lazo único da la ecuación de la carga: la fem es igual a la caída en la resistencia más la tensión en el capacitor, con $I=dq/dt$ por conservación de la carga. Es una ecuación diferencial lineal de primer orden con coeficientes constantes y término fuente constante. Su estructura es la misma que la de un sistema mecánico sobreamortiguado sin inercia.",
         [r"R\dfrac{dq}{dt}+\dfrac{q}{C}=\varepsilon",
          r"\dfrac{dq}{dt}+\dfrac{q}{\tau}=\dfrac{\varepsilon}{R},\qquad\tau=RC"]),
        ("Solución con factor integrante",
         r"Se multiplica la ecuación por $e^{t/\tau}$, que convierte el lado izquierdo en la derivada de un producto. Se integra desde $0$ hasta $t$ con $q(0)=q_0$. El resultado es la suma de la solución particular (estado estacionario $C\varepsilon$) y la homogénea (transitorio que decae con $\tau$), lo que permite escribir la respuesta para cualquier condición inicial.",
         [r"\dfrac{d}{dt}\!\left(q\,e^{t/\tau}\right)=\dfrac{\varepsilon}{R}\,e^{t/\tau}",
          r"q(t)=C\varepsilon+\left(q_0-C\varepsilon\right)e^{-t/\tau}\ \xrightarrow{\ q_0=0\ }\ q(t)=C\varepsilon\left(1-e^{-t/\tau}\right)"]),
        ("Corriente, tensiones y tiempos característicos",
         r"Derivando se obtiene la corriente, que decae exponencialmente desde $\varepsilon/R$. La tensión en el capacitor sube y la de la resistencia baja de forma complementaria, de modo que su suma es siempre $\varepsilon$. A $t=\tau$ el capacitor alcanza $1-e^{-1}\approx63.2\ \%$ de su carga final, y el tiempo para llegar a una fracción $f$ es $-\tau\ln(1-f)$. A $t\approx5\tau$ la carga supera el 99 %.",
         [r"I(t)=\dfrac{dq}{dt}=\dfrac{\varepsilon}{R}\,e^{-t/\tau},\qquad V_R=\varepsilon e^{-t/\tau},\qquad V_C=\varepsilon\left(1-e^{-t/\tau}\right)",
          r"t_f=-\tau\ln(1-f)"]),
        ("Balance de energía",
         r"Multiplicando la ecuación del circuito por $I$ se obtiene la potencia: la que entrega la fuente es igual a la disipada en la resistencia más la tasa de aumento de energía del campo del capacitor. Integrando en el tiempo se comprueba que la batería entrega $C\varepsilon^2$, el capacitor almacena $\tfrac12C\varepsilon^2$ y la resistencia disipa los otros $\tfrac12C\varepsilon^2$. La mitad disipada no depende de $R$: una resistencia menor carga más rápido pero con corriente mayor, y el producto es el mismo.",
         [r"\varepsilon I=I^2R+\dfrac{d}{dt}\!\left(\dfrac{q^2}{2C}\right)",
          r"E_{bat}=\int_0^\infty\varepsilon I\,dt=C\varepsilon^2,\qquad E_{dis}=\int_0^\infty I^2R\,dt=\dfrac{\varepsilon^2}{R}\cdot\dfrac{\tau}{2}=\tfrac12C\varepsilon^2"]),
        ("Límite de validez: inductancia y régimen oscilatorio",
         r"Todo circuito real tiene una inductancia parásita $L$. Entonces la ecuación es de segundo orden y la respuesta depende del factor de amortiguamiento. Si $R^2>4L/C$ el sistema es sobreamortiguado y la solución tiende a la exponencial simple cuando $L\to0$; en caso contrario aparecen oscilaciones. El modelo RC es válido para tiempos mucho mayores que $L/R$.",
         [r"L\dfrac{d^2q}{dt^2}+R\dfrac{dq}{dt}+\dfrac qC=\varepsilon",
          r"\zeta=\dfrac R2\sqrt{\dfrac CL}>1\ \Rightarrow\ \text{sobreamortiguado}"]),
        ("Resultado general",
         r"La carga evoluciona exponencialmente hacia $C\varepsilon$ con constante de tiempo $RC$, y la energía de la fuente se reparte por igual entre el capacitor y la resistencia. $\tau$ aparece como única escala de tiempo: toda la dinámica es una función universal de $t/\tau$. La verificación dimensional es directa: $C\varepsilon$ está en C, $\varepsilon/R$ en A y $C\varepsilon^2$ en J. Es el límite $L\to0$ del circuito RLC y la respuesta a un escalón de un filtro paso bajo de frecuencia de corte $f_c=1/(2\pi RC)$.",
         [r"\boxed{\,q(t)=C\varepsilon\left(1-e^{-t/RC}\right),\qquad I(t)=\dfrac\varepsilon Re^{-t/RC}\,}",
          r"E_{bat}=C\varepsilon^2,\qquad E_{cap}=E_{dis}=\tfrac12C\varepsilon^2"]),
        ("Particularización al problema",
         rf"Con $\varepsilon={eps:.0f}$ V, $R={R/1e3:.0f}\ \mathrm{{k\Omega}}$ y $C={C*1e6:.0f}\ \mu$F se calculan la constante de tiempo, los valores extremos y las energías. El tiempo para alcanzar 63.2 % de la carga final es aproximadamente $\tau$ y el de 99 % es de unos 4.6 $\tau$.",
         [rf"\tau=RC=({R:.0f})({sci(C,3)})={tau:.3f}\ \mathrm{{s}},\qquad q_\infty=C\varepsilon={q_inf*1e3:.2f}\ \mathrm{{mC}},\qquad I_0=\dfrac{{\varepsilon}}{{R}}={i0*1e3:.2f}\ \mathrm{{mA}}",
          rf"t_{{63.2\%}}=-\tau\ln(1-0.632)={t632:.5f}\ \mathrm{{s}},\qquad t_{{99\%}}=-\tau\ln(0.01)={t99:.4f}\ \mathrm{{s}}",
          rf"E_{{bat}}=C\varepsilon^2={C*eps**2*1e3:.2f}\ \mathrm{{mJ}},\qquad E_{{cap}}=E_{{dis}}={u_half*1e3:.2f}\ \mathrm{{mJ}}"]),
        ("Errores comunes y límites de validez",
         rf"Se suele pensar que la carga llega a su valor final en $\tau$, cuando solo alcanza 63 %; en teoría nunca lo alcanza exactamente. También es erróneo suponer que la energía disipada depende de $R$ o que la batería almacena la energía $C\varepsilon^2$ en el capacitor: solo la mitad queda almacenada. El modelo falla si el capacitor tiene fugas (resistencia en paralelo, que cambia la carga final a $C\varepsilon R_p/(R+R_p)$), si hay inductancia significativa (para estos datos la inductancia crítica sería $L_c=R^2C/4={l_crit:.0f}$ H, inalcanzable en la práctica) o si el dieléctrico es no lineal o tiene absorción.",
         [r"q_\infty=\dfrac{C\varepsilon R_p}{R+R_p},\qquad\tau'=\dfrac{RR_pC}{R+R_p}"]),
    ]
    return dict(concept=concept)


# ----------------------------------------------------------------------------------------- HelicalMotion
def _helical() -> dict:
    b, vp, vz = m8.B, m8.V_PERP, m8.V_PAR
    q, m = E_CH, M_P
    om = q * b / m
    r = m * vp / (q * b)
    T = 2 * np.pi / om
    p = vz * T
    f = om / (2 * np.pi)
    ang = np.degrees(np.arctan2(vz, vp))
    ek = 0.5 * m * (vp**2 + vz**2) / q
    mu_m = 0.5 * m * vp**2 / b
    flux = np.pi * r**2 * b
    pc = m * vp
    concept = [
        ("Qué se estudia y por qué importa",
         r"El movimiento de una partícula cargada en un campo magnético uniforme está en el origen de los ciclotrones, los espectrómetros de masa, el confinamiento del plasma en tokamaks, las auroras y los cinturones de Van Allen. La fuerza magnética no realiza trabajo y solo curva la trayectoria, por lo que la trayectoria es una hélice cuyo eje es la línea de campo. Medir el radio permite determinar el cociente entre la cantidad de movimiento y la carga, lo que identifica a las partículas.",
         [r"\vec F=q\,\vec v\times\vec B"]),
        ("Magnitudes y convenciones",
         r"La carga $q$ (C), la masa $m$ (kg), la inducción magnética $B$ (tesla) y la velocidad $\vec v$ (m/s) se relacionan por la fuerza de Lorentz. Se descompone $\vec v=\vec v_\perp+\vec v_\parallel$ respecto de $\hat B$. El radio $r$ se llama de Larmor o giroradio, la frecuencia $\omega=|q|B/m$ es la ciclotrónica y el paso $p$ es el avance axial por vuelta. Un protón en 0.1 T gira con frecuencia de unos 1.5 MHz, y el sentido de giro es antihorario visto desde $+\hat B$ para cargas negativas y horario para positivas.",
         [r"[B]=\mathrm{T}=\mathrm{N/(A\,m)}=\mathrm{kg/(C\,s)},\qquad[\omega]=\mathrm{rad/s}"]),
        ("Intuición y analogías",
         r"Como la fuerza es perpendicular a la velocidad, actúa como la tensión de una cuerda que hace girar una piedra: cambia la dirección y no la rapidez. Una partícula más rápida describe un círculo mayor, pero tarda lo mismo en cada vuelta porque debe recorrer una distancia proporcionalmente mayor. La componente paralela al campo no percibe la fuerza, así que se superpone un avance uniforme y la trayectoria es un resorte enrollado alrededor de la línea de campo.",
         [r"r\propto v_\perp,\qquad T\ \text{independiente de}\ v"]),
        ("Lagrangiano de una partícula en un campo electromagnético",
         r"La fuerza de Lorentz deriva de un potencial dependiente de la velocidad. El lagrangiano es $L=T-U$ con $U=q\phi-q\vec v\cdot\vec A$, donde $\vec B=\nabla\times\vec A$. Para un campo uniforme $B\hat z$ se elige la gauge simétrica $\vec A=\tfrac12\vec B\times\vec r$. El resultado físico no depende de la gauge, pues una transformación $\vec A\to\vec A+\nabla\chi$ añade al lagrangiano una derivada total del tiempo, que no afecta las ecuaciones de movimiento.",
         [r"L=\tfrac12m|\dot{\vec r}|^2+q\,\dot{\vec r}\cdot\vec A-q\phi",
          r"\vec A=\tfrac{B}{2}(-y,\,x,\,0)\ \Rightarrow\ L=\tfrac12m(\dot x^2+\dot y^2+\dot z^2)+\dfrac{qB}{2}(x\dot y-y\dot x)"]),
        ("Ecuaciones de Euler-Lagrange",
         r"Se calcula el momento canónico $p_i=\partial L/\partial\dot q_i$, que difiere del momento cinético por el término $qA_i$, y se aplican las ecuaciones de Euler-Lagrange a cada coordenada. Los términos magnéticos de las ecuaciones en $x$ e $y$ se suman y dan la fuerza $q\vec v\times\vec B$. La coordenada $z$ no aparece en el potencial, y su ecuación es la de una partícula libre.",
         [r"p_x=m\dot x-\dfrac{qB}{2}y,\quad p_y=m\dot y+\dfrac{qB}{2}x,\quad p_z=m\dot z",
          r"\dfrac{d}{dt}\!\left(m\dot x-\dfrac{qB}{2}y\right)=\dfrac{qB}{2}\dot y\ \Rightarrow\ m\ddot x=qB\dot y",
          r"m\ddot y=-qB\dot x,\qquad m\ddot z=0"]),
        ("Hamiltoniano y leyes de conservación por simetrías",
         r"El hamiltoniano es la energía cinética expresada con el momento cinético $\vec p-q\vec A$, y no depende explícitamente del tiempo, por lo que se conserva (teorema de Noether para traslaciones temporales): la fuerza magnética no realiza trabajo. La invariancia ante traslaciones a lo largo de $z$ conserva $p_z$, y la invariancia ante rotaciones alrededor de $z$ conserva el momento angular canónico, que incluye el término magnético. En total hay tres integrales primeras para tres grados de libertad, suficientes para que el problema sea integrable.",
         [r"H=\dfrac{(\vec p-q\vec A)^2}{2m}=\tfrac12mv^2=E,\qquad\dfrac{dH}{dt}=0",
          r"p_z=mv_\parallel=\text{cte},\qquad P_\varphi=m(x\dot y-y\dot x)+\dfrac{qB}{2}(x^2+y^2)=\text{cte}"]),
        ("Integración exacta en el plano perpendicular",
         r"Con la variable compleja $w=v_x+iv_y$ las dos ecuaciones de movimiento se unen en una ecuación lineal de primer orden, cuya solución es una rotación uniforme de la velocidad con frecuencia $\omega=qB/m$. Integrando una vez más se obtiene la trayectoria: una circunferencia recorrida con rapidez constante $v_\perp$. Su radio se deduce de $v_\perp=\omega r$, y es el mismo que resulta de igualar la fuerza magnética a la centrípeta.",
         [r"\dot w=-i\omega w\ \Rightarrow\ w(t)=v_\perp e^{-i\omega t},\qquad\omega=\dfrac{qB}{m}",
          r"x(t)=\dfrac{v_\perp}{\omega}\sin\omega t,\qquad y(t)=\dfrac{v_\perp}{\omega}(\cos\omega t-1),\qquad z(t)=v_\parallel t",
          r"r=\dfrac{v_\perp}{\omega}=\dfrac{mv_\perp}{qB}=\dfrac{p_\perp}{qB}"]),
        ("Hélice, centro de guía e invariante adiabático",
         r"La trayectoria es una hélice de radio $r$ y paso $p=v_\parallel T$. El centro de la circunferencia (centro de guía) se mueve a lo largo de la línea de campo con velocidad $v_\parallel$. En un campo que varía lentamente, el flujo magnético que abarca la órbita, o de forma equivalente el momento magnético orbital $\mu=mv_\perp^2/(2B)$, es un invariante adiabático. Este es el principio de los espejos magnéticos: al entrar en una región de $B$ mayor, $v_\perp$ crece y $v_\parallel$ disminuye hasta que la partícula se refleja. La condición de validez es $r\ll L_B$, con $L_B=B/|\nabla B|$ la escala de variación del campo.",
         [r"\tan\alpha=\dfrac{v_\parallel}{v_\perp},\qquad p=v_\parallel T=\dfrac{2\pi m v_\parallel}{qB}",
          r"\mu=\dfrac{mv_\perp^2}{2B}=\dfrac{q^2}{2\pi m}\,\Phi,\qquad\Phi=B\pi r^2"]),
        ("Resultado general",
         r"El giroradio es proporcional al momento perpendicular e inversamente proporcional a la carga y al campo; el periodo y la frecuencia no dependen de la velocidad. La energía cinética se conserva, y el paso es proporcional a $v_\parallel$. Dimensionalmente, $m v/(qB)$ tiene unidades kg·m/s / (C·kg/(C·s)) $=$ m, y $2\pi m/(qB)$ está en segundos. En el límite relativista basta cambiar $m$ por $\gamma m$: el periodo depende entonces de la energía, y por eso los ciclotrones deben sincronizarse.",
         [r"\boxed{\,r=\dfrac{mv_\perp}{|q|B},\qquad T=\dfrac{2\pi m}{|q|B},\qquad p=v_\parallel T=\dfrac{2\pi mv_\parallel}{|q|B}\,}",
          r"\gamma\ne1:\ r=\dfrac{\gamma mv_\perp}{|q|B},\quad T=\dfrac{2\pi\gamma m}{|q|B}"]),
        ("Particularización al problema",
         rf"Para un protón ($m={sci(m,5)}$ kg, $q={sci(q,5)}$ C) con $B={b}$ T, $v_\perp={sci(vp,2)}$ m/s y $v_\parallel={sci(vz,2)}$ m/s, se sustituyen los datos en las fórmulas generales y se recuperan los valores de la escena. También se calculan la frecuencia, el ángulo de paso, la energía cinética y el momento magnético.",
         [rf"\omega=\dfrac{{qB}}{{m}}={sci(om,5)}\ \mathrm{{rad/s}},\qquad f={sci(f,4)}\ \mathrm{{Hz}},\qquad T={T*1e6:.5f}\ \mu\mathrm{{s}}",
          rf"r=\dfrac{{mv_\perp}}{{qB}}={r*100:.4f}\ \mathrm{{cm}},\qquad p=v_\parallel T={p*100:.4f}\ \mathrm{{cm}},\qquad\alpha=\arctan\dfrac{{v_\parallel}}{{v_\perp}}={ang:.2f}^\circ",
          rf"E_k=\tfrac12m(v_\perp^2+v_\parallel^2)={ek:.1f}\ \mathrm{{eV}},\qquad\mu={sci(mu_m,4)}\ \mathrm{{J/T}},\qquad p_\perp={sci(pc,4)}\ \mathrm{{kg\,m/s}}"]),
        ("Errores comunes y límites de validez",
         r"Se suele creer que el campo magnético acelera a la partícula: solo cambia la dirección de la velocidad, y la energía solo puede cambiar con un campo eléctrico. Otro error es calcular el radio con la rapidez total en lugar de $v_\perp$, o tomar el paso de la hélice con $v_\perp$. El modelo supone un campo uniforme y estático, sin campo eléctrico, sin radiación y no relativista. Falla si $B$ varía apreciablemente en una órbita (espejos, deriva por gradiente), si hay un campo eléctrico transversal (deriva $\vec E\times\vec B$), si la partícula es relativista, o si los efectos cuánticos importan: la energía transversal está cuantizada en niveles de Landau.",
         [r"E_n=\hbar\omega\left(n+\tfrac12\right)+\dfrac{p_z^2}{2m},\qquad\vec v_{E\times B}=\dfrac{\vec E\times\vec B}{B^2}"]),
    ]
    return dict(concept=concept)


# ----------------------------------------------------------------------------------------------- HallEffect
def _hall() -> dict:
    n, t, w, i_c, b = m8.N_CU, m8.THICK, m8.WIDTH, m8.I_HALL, m8.B_HALL
    tau = m8.TAU
    vh = m8.hall_voltage_analytic()
    rh = 1.0 / (n * E_CH)
    j = i_c / (w * t)
    vd = j / (n * E_CH)
    eh = vd * b
    om = E_CH * b / M_E
    wt = om * tau
    sig = n * E_CH**2 * tau / M_E
    ex = j / sig
    n_rec = m8.carrier_density_from_hall(vh)
    concept = [
        ("Qué se estudia y por qué importa",
         r"En 1879 Edwin Hall observó que una corriente en una lámina conductora sometida a un campo magnético perpendicular genera una tensión transversal. El efecto permite determinar el signo y la densidad de los portadores de carga, algo que la resistencia por sí sola no revela, y es el fundamento de los sensores de campo magnético, de los medidores de corriente sin contacto y de la caracterización de semiconductores. Su versión cuántica es la base del patrón de resistencia eléctrica.",
         [r"V_H=\dfrac{IB}{nqt}"]),
        ("Magnitudes y convenciones",
         r"La cinta tiene espesor $t$ en la dirección del campo y ancho $w$ en la dirección transversal; la corriente $I$ fluye a lo largo de $x$ y $\vec B=B\hat z$. La densidad de corriente es $J=I/(wt)$. El coeficiente de Hall $R_H=1/(nq)$ se mide en m$^3$/C y tiene el signo de la carga $q$ de los portadores (negativo para electrones). Los voltajes de Hall son pequeños (microvoltios) en metales y grandes en semiconductores, donde $n$ es menor.",
         [r"[R_H]=\mathrm{m^3/C},\qquad[V_H]=\mathrm{V}"]),
        ("Intuición y analogías",
         r"Los portadores en movimiento sienten una fuerza lateral que los empuja hacia un borde de la cinta. Esa acumulación de carga en un borde y su ausencia en el otro crea un campo eléctrico transversal que crece hasta compensar exactamente la fuerza magnética. Es como un río que se curva por la fuerza de Coriolis hasta que la pendiente de la superficie del agua contrarresta ese efecto: se establece una inclinación transversal estacionaria.",
         [r"q\vec E_H+q\vec v_d\times\vec B=0"]),
        ("Ecuación de movimiento de los portadores (modelo de Drude)",
         r"Se aplica el modelo de Drude con campo magnético: la velocidad promedio $\vec v$ de los portadores obedece una ecuación de balance de momento donde la fricción efectiva representa las colisiones. En régimen estacionario la derivada temporal es nula y queda una relación lineal entre $\vec v$, $\vec E$ y $\vec B$. Se usa el momento promedio, de modo que los resultados no dependen de detalles de las colisiones.",
         [r"m\left(\dfrac{d\vec v}{dt}+\dfrac{\vec v}{\tau}\right)=q\left(\vec E+\vec v\times\vec B\right)",
          r"\vec v=\mu_q\left(\vec E+\vec v\times\vec B\right),\qquad\mu_q=\dfrac{q\tau}{m}"]),
        ("Tensor de conductividad y de resistividad",
         r"Con $\vec B=B\hat z$ y $\vec J=nq\vec v$, la relación anterior se escribe en componentes. Se obtiene un tensor de conductividad con elementos fuera de la diagonal proporcionales a $\omega_c\tau$, donde $\omega_c=qB/m$ es la frecuencia ciclotrónica. Invirtiéndolo, el tensor de resistividad es particularmente simple: $\rho_{xx}$ es independiente de $B$ y $\rho_{xy}$ es lineal en $B$.",
         [r"\sigma_{xx}=\dfrac{\sigma_0}{1+(\omega_c\tau)^2},\qquad\sigma_{xy}=-\sigma_{yx}=\dfrac{\sigma_0\,\omega_c\tau}{1+(\omega_c\tau)^2},\qquad\sigma_0=\dfrac{nq^2\tau}{m}",
          r"\rho_{xx}=\dfrac{1}{\sigma_0},\qquad\rho_{yx}=-\rho_{xy}=\dfrac{B}{nq}=R_HB,\qquad\omega_c\tau=\dfrac{qB\tau}{m}"]),
        ("Condición de circuito abierto transversal",
         r"En una cinta finita no puede haber corriente neta en la dirección $y$, porque las caras laterales están aisladas: $J_y=0$. Esa condición determina el campo transversal $E_y$ y deja una corriente que fluye solo a lo largo de $x$. En $y$ queda $E_y=\rho_{yx}J_x$, mientras que en $x$ el campo longitudinal es el óhmico $E_x=\rho_0J_x$, independiente de $B$. Para electrones $R_H<0$ y $E_y$ apunta hacia el borde que se carga negativamente.",
         [r"J_y=0\ \Longrightarrow\ E_y=\rho_{yx}J_x=R_H\,B\,J_x",
          r"E_x=\rho_0J_x,\qquad\dfrac{E_y}{E_x}=\omega_c\tau\ (\text{ángulo de Hall})"]),
        ("Tensión de Hall y dependencia con la geometría",
         r"Se integra $E_y$ sobre el ancho $w$ de la cinta y se expresa $J_x=I/(wt)$. El ancho $w$ se cancela: la tensión de Hall solo depende de la corriente, del campo, del espesor y de la densidad de portadores, lo que justifica el uso de cintas delgadas para aumentar la señal. En términos de la deriva, $E_H=v_dB$, con lo que la fórmula coincide con el balance de fuerzas.",
         [r"V_H=|E_y|\,w=|R_H|\,B\,\dfrac{I}{wt}\,w=\dfrac{IB}{n|q|\,t}",
          r"E_H=v_dB,\qquad v_d=\dfrac{J}{n|q|}=\dfrac{I}{n|q|wt}"]),
        ("Varios tipos de portadores y signo",
         r"El signo de $V_H$ revela el signo de los portadores: electrones y huecos se desvían hacia el mismo lado, pero la carga acumulada es opuesta, de modo que $R_H$ cambia de signo. Cuando coexisten electrones (densidad $n$, movilidad $\mu_e$) y huecos (densidad $p$, movilidad $\mu_h$), cada especie aporta una contribución y el coeficiente de Hall mide una combinación ponderada. En un metal monovalente como el cobre el modelo da $R_H=-1/(ne)$ con $n$ igual a la densidad atómica, lo que concuerda con el experimento.",
         [r"R_H=\dfrac{p\mu_h^2-n\mu_e^2}{e\,(p\mu_h+n\mu_e)^2}",
          r"R_H=-\dfrac{1}{ne}\ \text{(electrones)},\qquad R_H=+\dfrac{1}{pe}\ \text{(huecos)}"]),
        ("Resultado general",
         r"La tensión de Hall es proporcional a $I$ y a $B$ e inversamente proporcional al espesor y a la densidad de portadores; no depende del ancho ni, en el modelo de Drude, del tiempo de colisión $\tau$. La verificación dimensional es $\mathrm{A\,T}/(\mathrm{m^{-3}\,C\,m})=\mathrm{A\,T\,m^2/C}=\mathrm{V}$, pues $\mathrm{T}=\mathrm{V\,s/m^2}$. Si se mide $V_H$, se puede invertir la fórmula para obtener $n$.",
         [r"\boxed{\,V_H=\dfrac{IB}{nqt}=\dfrac{R_H\,IB}{t},\qquad n=\dfrac{IB}{qtV_H}\,}"]),
        ("Particularización al problema",
         rf"Para la cinta de cobre con $n={sci(n,3)}\ \mathrm{{m^{{-3}}}}$, $t={t*1e3:.2f}$ mm, $w={w*1e3:.0f}$ mm, $I={i_c:.1f}$ A y $B={b}$ T se evalúan las magnitudes. La tensión es de microvoltios y la densidad que se recupera invirtiendo la fórmula coincide con la de partida.",
         [rf"R_H=-\dfrac{{1}}{{ne}}=-{sci(rh*-1,4)}\ \mathrm{{m^3/C}},\qquad J=\dfrac{{I}}{{wt}}={sci(j,4)}\ \mathrm{{A/m^2}},\qquad v_d=\dfrac{{J}}{{ne}}={sci(vd,4)}\ \mathrm{{m/s}}",
          rf"E_H=v_dB={sci(eh,4)}\ \mathrm{{V/m}},\qquad V_H=\dfrac{{IB}}{{net}}={vh*1e6:.4f}\ \mu\mathrm{{V}}",
          rf"n=\dfrac{{IB}}{{qtV_H}}={sci(n_rec,4)}\ \mathrm{{m^{{-3}}}},\qquad\omega_c\tau={sci(wt,3)},\qquad\dfrac{{E_H}}{{E_x}}={sci(eh/ex,3)}"]),
        ("Errores comunes y límites de validez",
         r"Un error frecuente es usar el ancho $w$ en lugar del espesor $t$ (la dimensión paralela a $\vec B$), lo que da un factor $w/t=100$ de diferencia. También se confunde el signo de $V_H$ con el de la corriente convencional: depende del signo de los portadores. Además, en metales $\omega_c\tau\ll1$, de modo que el ángulo de Hall es pequeño y el campo Hall es minúsculo frente al longitudinal. El modelo de un solo portador deja de valer con varias bandas, a campos tan intensos que $\omega_c\tau\gtrsim1$ (magnetorresistencia) o a baja temperatura, donde aparece el efecto Hall cuántico con $\rho_{xy}=h/(\nu e^2)$ escalonada.",
         [r"\rho_{xy}=\dfrac{h}{\nu e^2},\qquad\nu\in\mathbb Z\ \ (\text{o racional})"]),
    ]
    return dict(concept=concept)


DETAIL = {
    "CoaxialLaplace": _coax(),
    "CapacitorNetwork": _network(),
    "DrudeElectrons": _drude(),
    "TaperedWire": _tapered(),
    "MeshCircuit": _mesh(),
    "RCCharging": _rc(),
    "HelicalMotion": _helical(),
    "HallEffect": _hall(),
}
