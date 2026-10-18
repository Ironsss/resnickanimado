"""Desarrollo conceptual y analitico general (campo `concept`). Volumen 2, capitulos 23 a 25:
BandStructure, DiodeCircuit, DecayChain, BindingEnergy, TwoBodyDecay, FriedmannAge.
Cada escena parte de principios, deduce la formula general con simbolos y solo al final la particulariza
(las cifras se calculan con physics/ o con las mismas formulas)."""
from __future__ import annotations

import numpy as np

from content.texfmt import fx, sci
from physics import v2ch23_solids as m23
from physics import v2ch24_nuclear as m24
from physics import v2ch25_particles as m25


# ====================================================================== BandStructure
def _band() -> list:
    g = m23.G_STRENGTH
    lim = m23.band_limits_analytic(3)
    (l1, u1), (l2, u2), (l3, u3) = lim
    gap = l2 - u1
    q1l, q1u, q2l = np.sqrt(2 * l1), np.sqrt(2 * u1), np.sqrt(2 * l2)
    w1, w2 = u1 - l1, u2 - l2
    # energia en el centro de la zona y en un k intermedio
    kmid = np.pi / 2
    e_mid = m23.band_energy_analytic(kmid, 0)
    qm = np.sqrt(2 * e_mid)
    f_mid = np.cos(qm) + g / qm * np.sin(qm)
    # masa efectiva en el fondo de la banda 1 (diferencias finitas)
    dk = 1e-3
    e0 = m23.band_energy_analytic(0.0, 0)
    ed = m23.band_energy_analytic(dk, 0)
    mstar = dk**2 / (2 * (ed - e0))
    f_at = lambda q: np.cos(q) + g / q * np.sin(q)  # noqa: E731
    return [
        ("Qué se estudia y por qué importa",
         r"Un electrón dentro de un cristal no se mueve en el vacío ni en un pozo aislado: siente el potencial periódico de millones de iones. De esa periodicidad nace la estructura de bandas, que decide si un material es aislante, semiconductor o metal. De ella dependen la conductividad, el color, la masa efectiva de los portadores y, en última instancia, todos los dispositivos electrónicos modernos, desde el diodo hasta el procesador. El modelo de Kronig-Penney sustituye los iones por una fila de barreras delta, de modo que la ecuación de Schrödinger se resuelve de forma exacta y se ve con claridad cómo aparecen bandas permitidas y brechas prohibidas.",
         [r"\left[-\dfrac{\hbar^2}{2m}\dfrac{d^2}{dx^2}+V(x)\right]\psi(x)=E\,\psi(x),\qquad V(x+a)=V(x)"]),
        ("Magnitudes y convenciones",
         r"La constante de red $a$ es la distancia entre iones (del orden de $10^{-10}$ m en sólidos reales). El vector de onda cristalino $k$ se define módulo $2\pi/a$ y basta tomarlo en la primera zona de Brillouin, $-\pi/a<k\le\pi/a$. La intensidad $g$ de cada delta tiene unidades de energía por longitud; $q=\sqrt{2mE}/\hbar$ es el número de onda local de un electrón libre de energía $E$. En unidades atómicas ($\hbar=m=e=1$) la energía se mide en hartrees (1 Ha $=27.211$ eV) y la longitud en radios de Bohr. Los electrones de valencia de un sólido real tienen bandas de anchura de unos pocos eV.",
         [r"\hbar=m=1,\qquad [g]=\text{energía}\cdot\text{longitud},\qquad [q]=\mathrm{longitud}^{-1},\qquad P\equiv\dfrac{mga}{\hbar^2}\ \ (\text{adimensional})"]),
        ("Intuición y analogías",
         r"Una onda que atraviesa una fila periódica de obstáculos se refleja parcialmente en cada uno de ellos, igual que la luz en una pila de láminas o que la onda en un collar de cuentas. Cuando la longitud de onda es tal que los reflejos de todos los obstáculos se suman en fase (condición de Bragg, $2a=n\lambda$), la onda no puede propagarse y aparece una brecha. Entre brechas existen estados extendidos por todo el cristal que forman una banda casi continua. Si las barreras son muy opacas, cada celda se comporta como un pozo casi aislado y las bandas se estrechan hacia los niveles discretos del pozo.",
         []),
        ("Teorema de Bloch: la simetría de traslación",
         r"El hamiltoniano es invariante bajo la traslación discreta $T_a:\ x\to x+a$, de modo que $[\hat H,\hat T_a]=0$ y ambos pueden diagonalizarse a la vez. Como $\hat T_a$ es unitario, sus autovalores tienen módulo uno y se escriben $e^{ika}$ con $k$ real: ese es el contenido del teorema de Bloch. Una autofunción puede escribirse como onda plana modulada por una función periódica. Dos valores $k$ y $k+2\pi/a$ dan el mismo autovalor y por eso la numeración queda restringida a la primera zona de Brillouin. Es la analogía exacta de la conservación del momento en el espacio homogéneo, pero ahora solo se conserva módulo $2\pi\hbar/a$ (cuasi-momento).",
         [r"\hat T_a\psi(x)=\psi(x+a)=e^{ika}\psi(x),\qquad \psi_{nk}(x)=e^{ikx}u_{nk}(x),\quad u_{nk}(x+a)=u_{nk}(x)",
          r"\psi_{n,k+G}=\psi_{n,k},\qquad G=\dfrac{2\pi}{a}\,j,\ \ j\in\mathbb Z,\qquad E_n(k)=E_n(-k)"]),
        ("Planteamiento general: peine de deltas y condiciones de empalme",
         r"Se toma $V(x)=g\sum_n\delta(x-na)$ con $g>0$ (barreras repulsivas). Entre dos deltas la partícula es libre, de modo que en la celda $0<x<a$ la solución general es combinación de dos ondas planas de número de onda $q$. La delta impone dos condiciones en $x=0$: continuidad de $\psi$ y un salto de la derivada que se obtiene integrando la ecuación de Schrödinger sobre un intervalo infinitesimal. La función en la celda vecina $-a<x<0$ no es independiente: por Bloch vale $e^{-ika}$ veces la de la celda desplazada.",
         [r"\psi(x)=A\,e^{iqx}+B\,e^{-iqx}\ \ (0<x<a),\qquad q=\dfrac{\sqrt{2mE}}{\hbar}",
          r"\psi(x)=e^{-ika}\left[A\,e^{iq(x+a)}+B\,e^{-iq(x+a)}\right]\ \ (-a<x<0)",
          r"\psi(0^+)=\psi(0^-),\qquad \psi'(0^+)-\psi'(0^-)=\dfrac{2mg}{\hbar^2}\,\psi(0)"]),
        ("Condición de compatibilidad: relación de dispersión",
         r"Las dos condiciones de empalme forman un sistema lineal homogéneo en $(A,B)$. Sólo tiene solución no trivial si su determinante se anula, y esa es la ecuación que liga $E$ con $k$. Con $u=e^{-ika}$ y $\alpha=2mg/\hbar^2$ el sistema queda escrito abajo; el determinante se simplifica usando $u+u^{-1}=2\cos ka$ y $e^{iqa}-e^{-iqa}=2i\sin qa$. El resultado es la relación de Kronig-Penney, válida para cualquier $g$, $a$ y energía $E>0$.",
         [r"\begin{aligned}A\,(1-u\,e^{iqa})+B\,(1-u\,e^{-iqa})&=0\\ A\,[\,iq(1-u\,e^{iqa})-\alpha\,]+B\,[-iq(1-u\,e^{-iqa})-\alpha\,]&=0\end{aligned}",
          r"q\left(u+u^{-1}-2\cos qa\right)=\alpha\sin qa\ \Longrightarrow\ \cos ka=\cos qa+\dfrac{mg}{\hbar^2 q}\sin qa"]),
        ("Bandas permitidas, brechas y parámetro adimensional",
         r"Como $\cos ka$ solo puede valer entre $-1$ y $1$, una energía es permitida únicamente si el miembro derecho cumple $|F(q)|\le1$; fuera de ese rango $k$ es complejo y la onda decae exponencialmente (brecha). Definiendo $P=mga/\hbar^2$, la función $F$ depende solo de $qa$ y de $P$, por lo que toda la física del peine de deltas está contenida en un único parámetro. Para $P>0$ los extremos superiores de banda están siempre en $qa=n\pi$, donde $\sin qa=0$ y $F=(-1)^n$: son estados con nodos en cada delta, que no sienten la barrera. Los extremos inferiores se encuentran resolviendo $F=\pm1$.",
         [r"F(q)\equiv\cos qa+P\,\dfrac{\sin qa}{qa},\qquad \text{energía permitida}\iff |F(q)|\le1",
          r"E_n^{\max}=\dfrac{\hbar^2}{2m}\left(\dfrac{n\pi}{a}\right)^2,\qquad \left(k=\tfrac{\pi}{a}\ \text{si }n\text{ impar};\ \ k=0\ \text{si }n\text{ par}\right)",
          r"E_{\mathrm{gap},n}=E_{n+1}^{\min}-E_n^{\max},\qquad \cos(k_{\mathrm{borde}}a)=\pm1\Rightarrow F(q)=\pm1"]),
        ("Velocidad de grupo, masa efectiva y casos límite",
         r"Derivando implícitamente la relación de dispersión respecto de $k$ se obtiene la velocidad de grupo $v=\hbar^{-1}dE/dk$; se anula en los bordes de banda, donde $\sin ka=0$, y allí la onda es estacionaria. La curvatura de $E(k)$ define la masa efectiva $m^*$, que reemplaza a $m$ en la dinámica de un paquete de ondas cerca de un extremo. Cuando $P\to0$ se recupera la parábola del electrón libre plegada en la zona de Brillouin; cuando $P\to\infty$ las bandas se colapsan en los niveles $(n\pi/a)^2\hbar^2/2m$ del pozo infinito de ancho $a$, con anchura de banda que decrece como $1/P$.",
         [r"-a\sin ka\ dk=F'(q)\,dq\ \Longrightarrow\ v(k)=\dfrac1\hbar\dfrac{dE}{dk}=\dfrac{\hbar q}{m}\,\dfrac{dq}{dk}=-\dfrac{\hbar q\,a\sin ka}{m\,F'(q)}",
          r"\dfrac1{m^*}=\dfrac1{\hbar^2}\dfrac{d^2E}{dk^2}\bigg|_{k_0},\qquad P\to0:\ E=\dfrac{\hbar^2k^2}{2m},\qquad P\to\infty:\ E_n\to\dfrac{\hbar^2n^2\pi^2}{2ma^2},\ \ W_n\sim\dfrac{1}{P}"]),
        ("Resultado general",
         r"El resultado completo es la relación de dispersión implícita $\cos ka=F(q)$: dado un $k$ se obtiene $E_n(k)$ en cada banda $n$ resolviendo para $q$, y las bandas son los intervalos de $q$ donde $|F|\le1$. Los bordes de la banda $n$ son $q_n^{\max}a=n\pi$ y la raíz de $F=\pm1$ inmediatamente anterior, de modo que el ancho de brecha depende solo de $P$ y de la escala $\hbar^2/ma^2$. La verificación dimensional es directa: $P$ es adimensional, $E\sim\hbar^2/(ma^2)$ (hartree en unidades atómicas con $a=1$) y $v\sim\hbar/(ma)$.",
         [r"\boxed{\ \cos ka=\cos qa+\dfrac{mg}{\hbar^2q}\sin qa,\qquad E_n(k)=\dfrac{\hbar^2q_n(k)^2}{2m}\ }",
          r"\left[\dfrac{mg}{\hbar^2q}\right]=\dfrac{\mathrm{kg}\cdot\mathrm{J\,m}}{(\mathrm{J\,s})^2\cdot\mathrm{m^{-1}}}=\dfrac{\mathrm{kg\,m^2\,J}}{\mathrm{J^2\,s^2}}=1,\qquad \left[\dfrac{\hbar^2}{ma^2}\right]=\dfrac{(\mathrm{J\,s})^2}{\mathrm{kg\,m^2}}=\mathrm{J}"]),
        ("Particularización al problema",
         rf"Con $m=\hbar=a=1$ y $g={g:g}$ el parámetro es $P=mga/\hbar^2={g:g}$. Se buscan las raíces de $F(q)=\pm1$ y se usan $q a=n\pi$ para los techos de banda, con $E=q^2/2$. Se recuperan las cifras de la escena. A modo de ejemplo, en $k=\pi/2$ la primera banda tiene $E={e_mid:.4f}$ Ha, que cumple $F(q)=\cos ka=0$.",
         [rf"\text{{Banda 1: }}q_{{\min}}={q1l:.4f},\ q_{{\max}}=\pi\ \Rightarrow\ E\in\left[{l1:.4f},\ \dfrac{{\pi^2}}{{2}}={u1:.4f}\right]\ \mathrm{{Ha}}",
          rf"\text{{Banda 2: }}q_{{\min}}={q2l:.4f},\ q_{{\max}}=2\pi\ \Rightarrow\ E\in\left[{l2:.4f},\ {u2:.4f}\right]\ \mathrm{{Ha}}",
          rf"E_{{gap}}={l2:.4f}-{u1:.4f}={gap:.4f}\ \mathrm{{Ha}},\qquad W_1={w1:.4f}\ \mathrm{{Ha}},\ \ W_2={w2:.4f}\ \mathrm{{Ha}}",
          rf"F(q_{{1,\min}})={f_at(q1l):.4f}=+1,\quad F(\pi)={f_at(np.pi):.4f}=-1,\quad m^*_{{\text{{fondo banda 1}}}}\approx{mstar:.4f}\,m"]),
        ("Errores comunes y límites de validez",
         r"Es frecuente confundir $k$ con el momento: $\hbar k$ es solo el cuasi-momento y no es la cantidad de movimiento de la partícula. También se cree que la brecha aparece por el potencial medio, cuando lo hace por su componente de Fourier a $G=2\pi/a$; un potencial constante no abre ninguna brecha. Otro error es olvidar que con $P$ grande las bandas siguen siendo de ancho finito, pues el efecto túnel acopla celdas vecinas. El modelo es unidimensional, no incluye la interacción electrón-electrón ni los fonones, y la delta ideal es una abstracción; en sólidos reales se usan pseudopotenciales y cálculos en ondas planas, cuya convergencia para deltas es lenta (error $\sim1/N$).",
         [r"V_G=\dfrac1a\int_0^aV(x)\,e^{-iGx}dx,\qquad E_{\mathrm{gap}}\approx2|V_{G}|\ \ (\text{potencial d\'ebil})"]),
    ]


# ====================================================================== DiodeCircuit
def _diode() -> list:
    i_s, n, vt, r = m23.I_S, m23.N_IDEAL, m23.V_T, m23.R_SERIES
    nvt = n * vt
    e = 5.0
    vd = m23.diode_voltage_analytic(e)
    cur = m23.diode_current(e, vd)
    lam = i_s * r / nvt
    z = lam * np.exp((e + i_s * r) / nvt)
    from scipy.special import lambertw
    w = float(lambertw(z).real)
    v_log = nvt * np.log(1 + cur / i_s)
    rd = nvt / (cur + i_s)
    pw_d, pw_r, pw_e = vd * cur, cur**2 * r, e * cur
    # iteracion de punto fijo
    v = 0.6
    for _ in range(6):
        v = nvt * np.log(1 + (e - v) / (r * i_s))
    return [
        ("Qué se estudia y por qué importa",
         r"La unión p-n es el bloque elemental de la electrónica: rectificadores, LED, celdas solares, fotodiodos y la base de los transistores. Su característica esencial es la asimetría, pues conduce con facilidad en un sentido y casi nada en el contrario. Aquí se estudia el diodo ideal de Shockley conectado en serie con una resistencia a una fuente, un circuito que aparece en toda fuente de alimentación. El análisis ilustra un patrón general en física aplicada: un elemento no lineal exponencial más un elemento lineal conducen a una ecuación trascendente que se resuelve de manera exacta con una función especial (la función $W$ de Lambert).",
         []),
        ("Magnitudes y convenciones",
         r"La corriente de saturación inversa $I_s$ es del orden de picoamperios en silicio. El voltaje térmico $V_T=k_BT/q$ vale $25.85$ mV a 300 K y el factor de idealidad $n$ está entre 1 y 2. La tensión $V_d$ se mide del ánodo (lado p) al cátodo (lado n), y $I>0$ circula en el sentido directo. $E$ es la fem de la fuente y $R$ la resistencia en serie. En polarización directa de silicio $V_d$ cae típicamente entre 0.5 y 0.8 V, con una ligera dependencia logarítmica de la corriente.",
         [r"V_T=\dfrac{k_BT}{q},\qquad [I_s]=\mathrm{A},\ \ [V_d]=\mathrm{V},\ \ [R]=\Omega,\qquad \lambda\equiv\dfrac{I_sR}{nV_T},\ \ \varepsilon\equiv\dfrac{E}{nV_T}\ \ (\text{adimensionales})"]),
        ("Intuición y analogías",
         r"El diodo se comporta como una válvula de retención hidráulica con una apertura gradual: casi cerrada hasta que la presión supera un umbral difuso y, pasado éste, deja pasar un caudal que crece exponencialmente. Como la fuente y la resistencia imponen una relación lineal entre tensión y corriente (la recta de carga) y el diodo impone otra curva (exponencial), el punto de operación es la intersección de ambas. Al ser la exponencial tan empinada, un cambio enorme de la corriente solo mueve la tensión del diodo unas décimas de voltio; por eso el diodo actúa casi como una fuente de tensión de unos 0.6 V en directa.",
         []),
        ("Origen de la característica de Shockley",
         r"En la unión polarizada con $V_d$, la barrera de potencial se reduce en $qV_d$ y la población de portadores minoritarios inyectados en el borde de la zona de carga espacial crece según el factor de Boltzmann. Esos minoritarios difunden y se recombinan en una longitud $L$, y la corriente de difusión resultante es proporcional al exceso de portadores. Se obtiene una corriente neta que es la diferencia entre el flujo directo, que crece exponencialmente con $V_d$, y el inverso, constante. La hipótesis es baja inyección y despreciar la recombinación en la zona de carga espacial (de ahí el factor $n=1$).",
         [r"n_p(0)=n_{p0}\,e^{qV_d/k_BT},\qquad J_n=qD_n\dfrac{n_p(0)-n_{p0}}{L_n}",
          r"I=I_s\left(e^{V_d/nV_T}-1\right),\qquad I_s=Aqn_i^2\left(\dfrac{D_n}{L_nN_A}+\dfrac{D_p}{L_pN_D}\right)"]),
        ("Ley de Kirchhoff y existencia de solución única",
         r"El circuito serie cumple la ley de las mallas, $E=V_d+IR$, que junto con la característica del diodo da una ecuación para la sola incógnita $V_d$. Definiendo $f(V)=I_s(e^{V/nV_T}-1)-(E-V)/R$, la derivada $f'(V)=\frac{I_s}{nV_T}e^{V/nV_T}+\frac1R$ es positiva para todo $V$, de modo que $f$ es estrictamente creciente y tiene a lo sumo una raíz. Para $E>0$ se cumple $f(0)=-E/R<0$ y $f(E)=I_s(e^{E/nV_T}-1)>0$, luego la raíz existe y está en el intervalo $(0,E)$: eso justifica la bisección y garantiza la unicidad física del punto de operación.",
         [r"E=V_d+IR,\qquad I=I_s\left(e^{V_d/nV_T}-1\right)",
          r"f(V)=I_s\left(e^{V/nV_T}-1\right)-\dfrac{E-V}{R},\qquad f'(V)=\dfrac{I_s}{nV_T}\,e^{V/nV_T}+\dfrac1R>0"]),
        ("Solución exacta con la función W de Lambert",
         r"Eliminando $I$ se obtiene $E-V_d=RI_s(e^{V_d/nV_T}-1)$. Con el cambio de variable $w=(E+RI_s-V_d)/nV_T$ la exponencial se reescribe como $e^{V_d/nV_T}=e^{(E+RI_s)/nV_T}e^{-w}$ y la ecuación queda $we^{w}=\lambda\,e^{\varepsilon+\lambda}$. Esta es exactamente la ecuación que define la función $W$ de Lambert, $W(z)e^{W(z)}=z$, cuya rama principal es real y única para $z>0$. Se obtiene así una expresión cerrada para la tensión del diodo, y de ella la corriente.",
         [r"w\,e^{w}=\lambda\,e^{\varepsilon+\lambda}\ \Longrightarrow\ w=W\!\left(\lambda\,e^{\varepsilon+\lambda}\right)",
          r"V_d=E+I_sR-nV_T\,W\!\left(\dfrac{I_sR}{nV_T}\,e^{(E+I_sR)/nV_T}\right),\qquad I=\dfrac{E-V_d}{R}"]),
        ("Comportamiento asintótico y resistencia dinámica",
         r"Como el argumento de $W$ es enorme en directa ($E\gg nV_T$), se usa el desarrollo asintótico $W(z)\simeq\ln z-\ln\ln z$. Conduce a $V_d\simeq nV_T\ln(1+I/I_s)$, es decir, invirtiendo la ley de Shockley con la corriente impuesta por el resto del circuito. Esto sugiere una iteración de punto fijo que converge muy rápido porque la dependencia es logarítmica. La resistencia dinámica $r_d=dV_d/dI=nV_T/(I+I_s)$ es pequeña en directa (decenas de ohmios o menos) y gigantesca en inversa. En el otro límite, $E<0$ con $|E|\gg nV_T$, resulta $I\to-I_s$ y $V_d\to E+I_sR$.",
         [r"V_d\simeq nV_T\ln\!\left(1+\dfrac{I}{I_s}\right),\qquad I=\dfrac{E-V_d}{R}\ \ \Rightarrow\ \ V_{k+1}=nV_T\ln\!\left(1+\dfrac{E-V_k}{RI_s}\right)",
          r"r_d=\dfrac{dV_d}{dI}=\dfrac{nV_T}{I+I_s},\qquad E\to-\infty:\ \ I\to-I_s,\ \ V_d\to E+I_sR"]),
        ("Resultado general",
         r"Para un diodo de parámetros $(I_s,n,V_T)$ en serie con $R$ y una fuente $E$ cualquiera, la tensión en el diodo y la corriente quedan dadas en forma cerrada por la función $W$. Dependencia: $V_d$ crece solo logarítmicamente con $E$ (para $E\gg nV_T$), mientras la corriente lo hace casi linealmente, $I\approx(E-V_d)/R$. Dimensionalmente, $I_sR/nV_T$ y $E/nV_T$ son adimensionales, $nV_T$ tiene unidades de voltio y $W$ es adimensional; así $V_d$ resulta en voltios e $I$ en amperios. El balance de potencias es $EI=V_dI+I^2R$.",
         [r"\boxed{\ V_d(E)=E+I_sR-nV_T\,W\!\left(\dfrac{I_sR}{nV_T}\,e^{(E+I_sR)/nV_T}\right)\ }",
          r"I=\dfrac{E-V_d}{R}=I_s\left(e^{V_d/nV_T}-1\right),\qquad EI=V_dI+RI^2"]),
        ("Particularización al problema",
         rf"Con $I_s={sci(i_s,1)}\ \mathrm{{A}}$, $n=1$, $V_T={fx(vt*1e3,2)}$ mV y $R={r:g}\ \Omega$ se tiene $\lambda={sci(lam,4)}$. Para $E={e:g}$ V el argumento de $W$ es $z={sci(z,4)}$, $W(z)={w:.4f}$, y se recupera la tensión de la escena. El resultado es coherente con la forma asintótica logarítmica y con el balance de potencias.",
         [rf"V_d={e:g}+{sci(i_s*r,2)}-({fx(nvt,6)})({w:.4f})={vd:.6f}\ \mathrm{{V}}",
          rf"I=\dfrac{{{e:g}-{vd:.6f}}}{{{r:g}}}={cur*1e3:.4f}\ \mathrm{{mA}}",
          rf"nV_T\ln\!\left(1+\dfrac{{I}}{{I_s}}\right)={fx(v_log,6)}\ \mathrm{{V}},\qquad r_d=\dfrac{{nV_T}}{{I}}={rd:.4f}\ \Omega",
          rf"EI={pw_e*1e3:.4f}\ \mathrm{{mW}}=V_dI+RI^2={pw_d*1e3:.4f}+{pw_r*1e3:.4f}\ \mathrm{{mW}},\qquad V_{{6}}\ (\text{{punto fijo}})={fx(v,6)}\ \mathrm{{V}}"]),
        ("Errores comunes y límites de validez",
         r"Un error frecuente es suponer que el diodo cae siempre 0.7 V: es una aproximación válida solo para corrientes del orden de miliamperios en silicio, ya que $V_d$ depende logarítmicamente de $I$. Otro es olvidar el $-1$ de Shockley, que es esencial en inversa y se desprecia sin riesgo en directa. El modelo ideal deja de valer a corrientes grandes, donde la resistencia óhmica de los semiconductores y la alta inyección hacen que $V_d$ crezca linealmente; en inversa fuerte aparece la ruptura por avalancha o Zener, que el modelo no contiene. Además, $I_s$ crece muy rápido con la temperatura (se duplica cada unos 5 a 10 K), de modo que $V_d$ disminuye cerca de $2$ mV/K a corriente fija.",
         [r"\dfrac{dV_d}{dT}\bigg|_I\simeq-\dfrac{E_g/q-V_d}{T}\approx-2\ \mathrm{mV/K},\qquad V_d\to V_d+IR_s\ \ (\text{resistencia parásita})"]),
    ]


# ====================================================================== DecayChain
def _chain() -> list:
    la, lb, n0 = m24.LAM_A, m24.LAM_B, m24.N0
    tmax = m24.peak_time_analytic()
    nbmax = (lb / la) ** (lb / (la - lb))
    nb20 = float(m24.bateman_b(20.0))
    na20 = float(m24.n_a(20.0))
    nc20 = n0 - na20 - nb20
    mean, err = m24.mean_life_monte_carlo()
    th_a, th_b = np.log(2) / la, np.log(2) / lb
    nbtm = float(m24.bateman_b(tmax))
    return [
        ("Qué se estudia y por qué importa",
         r"Los núcleos inestables se transforman en otros al emitir partículas, y con frecuencia el producto también es radiactivo, formando una cadena como las del uranio, el torio o el neptunio. El análisis de cadenas permite fechar rocas, calcular el inventario de residuos nucleares, dimensionar generadores de isótopos médicos (por ejemplo el par molibdeno-99 y tecnecio-99m) y estimar la dosis de un paciente. La pregunta central es cuántos núcleos de cada especie quedan al cabo de un tiempo, dados los ritmos de decaimiento de todos los eslabones.",
         []),
        ("Magnitudes y convenciones",
         r"La constante de decaimiento $\lambda_i$ (unidad s$^{-1}$) es la probabilidad por unidad de tiempo de que un núcleo de la especie $i$ decaiga. La vida media $\tau_i=1/\lambda_i$ y la semivida $T_{1/2}=\ln2/\lambda_i$ miden la misma escala temporal. $N_i(t)$ es el número de núcleos y la actividad $A_i=\lambda_iN_i$ (becquerel, desintegraciones por segundo) es lo que realmente miden los detectores. Se numera la cadena $1\to2\to\dots\to n$; el último eslabón puede ser estable ($\lambda_n=0$). Las semividas van desde fracciones de segundo hasta $10^{10}$ años.",
         [r"A_i=\lambda_iN_i,\qquad \tau_i=\dfrac1{\lambda_i},\qquad T_{1/2,i}=\dfrac{\ln2}{\lambda_i},\qquad [\lambda]=\mathrm{s^{-1}},\ \ [A]=\mathrm{Bq}"]),
        ("Intuición y analogías",
         r"Una cadena de decaimiento se parece a una cascada de depósitos de agua: cada depósito se vacía por un orificio hacia el siguiente a un ritmo proporcional a su nivel. El primero solo pierde, el último solo gana, y los intermedios primero se llenan y luego se vacían, de modo que su contenido presenta un máximo. Si el depósito intermedio drena mucho más rápido que el primero, su nivel sigue de cerca al del primero (equilibrio); si drena más lento, se acumula. La aleatoriedad individual es la de un reloj sin memoria: un núcleo no envejece.",
         []),
        ("Ley de decaimiento: hipótesis y origen estadístico",
         r"La hipótesis física es que cada núcleo decae de forma independiente, con probabilidad por unidad de tiempo $\lambda$ constante, sin memoria de su pasado. Entonces la probabilidad de que un núcleo sobreviva un tiempo $t+dt$ es la de sobrevivir hasta $t$ multiplicada por $1-\lambda\,dt$, lo que lleva a una ecuación diferencial cuya solución es una exponencial. El tiempo de vida es una variable aleatoria con densidad exponencial, y su media es $1/\lambda$. Para $N$ grande el número de supervivientes es el valor esperado con fluctuaciones relativas de orden $1/\sqrt N$ (estadística de Poisson).",
         [r"P(t+dt)=P(t)\,(1-\lambda\,dt)\ \Rightarrow\ \dfrac{dP}{dt}=-\lambda P,\qquad P(t)=e^{-\lambda t}",
          r"f(t)=\lambda e^{-\lambda t},\qquad \langle t\rangle=\int_0^\infty t\,\lambda e^{-\lambda t}dt=\dfrac1\lambda,\qquad \dfrac{\sigma_N}{\langle N\rangle}\sim\dfrac1{\sqrt N}"]),
        ("Sistema de Bateman para una cadena de n especies",
         r"Cada especie $i$ pierde núcleos a razón $\lambda_iN_i$ y los gana de la especie anterior a razón $\lambda_{i-1}N_{i-1}$. El resultado es un sistema lineal con matriz triangular inferior y coeficientes constantes, el sistema de Bateman. Como la matriz es triangular, sus autovalores son los elementos diagonales $-\lambda_i$, y la dinámica es una suma de exponenciales con esas tasas (si son distintas). Además, la suma $\sum N_i$ se conserva, pues la suma de las columnas de la matriz es cero.",
         [r"\dfrac{dN_1}{dt}=-\lambda_1N_1,\qquad \dfrac{dN_i}{dt}=\lambda_{i-1}N_{i-1}-\lambda_iN_i\quad(i=2,\dots,n)",
          r"\dfrac{d\vec N}{dt}=M\vec N,\qquad M=\begin{pmatrix}-\lambda_1&&&\\ \lambda_1&-\lambda_2&&\\ &\lambda_2&-\lambda_3&\\ &&\ddots&\ddots\end{pmatrix},\qquad \sum_iN_i=N_0"]),
        ("Solución general por transformada de Laplace",
         r"La transformada de Laplace convierte las ecuaciones diferenciales en algebraicas: $s\tilde N_i-N_i(0)=\lambda_{i-1}\tilde N_{i-1}-\lambda_i\tilde N_i$. Con la condición inicial de sólo la especie 1 presente ($N_1(0)=N_0$) se resuelve recursivamente y se obtiene un producto de factores $1/(s+\lambda_j)$. Si los $\lambda_j$ son todos distintos, la descomposición en fracciones simples y la inversa de cada polo $s=-\lambda_i$ dan la fórmula de Bateman. El caso de dos especies es el que se pide en el problema.",
         [r"\tilde N_n(s)=N_0\,\dfrac{\lambda_1\lambda_2\cdots\lambda_{n-1}}{(s+\lambda_1)(s+\lambda_2)\cdots(s+\lambda_n)}",
          r"N_n(t)=N_0\left(\prod_{j=1}^{n-1}\lambda_j\right)\sum_{i=1}^{n}\dfrac{e^{-\lambda_it}}{\prod_{j\ne i}(\lambda_j-\lambda_i)}",
          r"n=2:\ \ N_2(t)=N_0\,\dfrac{\lambda_1}{\lambda_2-\lambda_1}\left(e^{-\lambda_1t}-e^{-\lambda_2t}\right)"]),
        ("Casos especiales: degeneración y equilibrios",
         r"Si $\lambda_1=\lambda_2=\lambda$ la fórmula tiene un denominador nulo, pero el límite existe y aparece un polo doble: $N_2=N_0\lambda t\,e^{-\lambda t}$. Cuando el padre vive mucho más que el hijo ($\lambda_1\ll\lambda_2$) se establece un equilibrio secular en el que las actividades se igualan, $\lambda_2N_2\simeq\lambda_1N_1$. Si el padre vive más que el hijo pero no tanto ($\lambda_1<\lambda_2$) se alcanza un equilibrio transitorio con razón constante de actividades. Y si el hijo vive más ($\lambda_2<\lambda_1$) no hay equilibrio, y a tiempos largos domina la cola del hijo.",
         [r"\lambda_1\to\lambda_2:\ \ N_2\to N_0\,\lambda\,t\,e^{-\lambda t}",
          r"\lambda_1\ll\lambda_2:\ \ \dfrac{A_2}{A_1}\to1,\qquad \lambda_1<\lambda_2:\ \ \dfrac{A_2}{A_1}\to\dfrac{\lambda_2}{\lambda_2-\lambda_1}"]),
        ("Máximo del eslabón intermedio y balance",
         r"El máximo de $N_2$ ocurre cuando su derivada se anula, es decir, cuando la tasa de producción iguala a la de pérdida: $\lambda_2N_2=\lambda_1N_1$ (la actividad del hijo cruza a la del padre). Derivando la solución de Bateman y despejando se obtiene el instante $t_{\max}$, simétrico en los dos índices. Sustituyendo en la solución se halla el valor del máximo en forma cerrada. La población del estado final estable se obtiene por conservación, $N_3=N_0-N_1-N_2$.",
         [r"\dfrac{dN_2}{dt}=0\iff\lambda_2N_2=\lambda_1N_1\ \Longrightarrow\ t_{\max}=\dfrac{\ln(\lambda_1/\lambda_2)}{\lambda_1-\lambda_2}",
          r"N_2(t_{\max})=N_0\left(\dfrac{\lambda_2}{\lambda_1}\right)^{\lambda_2/(\lambda_1-\lambda_2)},\qquad N_3(t)=N_0-N_1(t)-N_2(t)"]),
        ("Resultado general",
         r"La población de cualquier eslabón de una cadena lineal con constantes distintas viene dada por la suma de exponenciales de Bateman, y la del eslabón intermedio de una cadena de tres especies por la fórmula enmarcada. Depende de las tasas sólo a través de $\lambda_1/\lambda_2$ y del producto $\lambda_1t$, así que el comportamiento es universal una vez medido el tiempo en vidas medias. Dimensionalmente, $\lambda t$ es adimensional y $N_2$ se mide en las mismas unidades que $N_0$ (núcleos); $t_{\max}$ tiene unidades de tiempo.",
         [r"\boxed{\ N_2(t)=N_0\,\dfrac{\lambda_1}{\lambda_2-\lambda_1}\left(e^{-\lambda_1t}-e^{-\lambda_2t}\right),\qquad t_{\max}=\dfrac{\ln(\lambda_1/\lambda_2)}{\lambda_1-\lambda_2}\ }",
          r"N_{2,\max}=N_0\left(\dfrac{\lambda_2}{\lambda_1}\right)^{\lambda_2/(\lambda_1-\lambda_2)},\qquad \tau_2=\dfrac1{\lambda_2}"]),
        ("Particularización al problema",
         rf"Con $\lambda_A={la:g}\ \mathrm{{s^{{-1}}}}$, $\lambda_B={lb:g}\ \mathrm{{s^{{-1}}}}$ y $N_0=1$ (poblaciones relativas) se aplican las fórmulas generales con $1=A$ y $2=B$. Se recupera el tiempo del máximo de la escena y se evalúa $N_B$ en 20 s. La vida media de B es $1/\lambda_B={1/lb:g}$ s, y el experimento de Monte Carlo la reproduce dentro de su incertidumbre estadística.",
         [rf"t_{{\max}}=\dfrac{{\ln({la:g}/{lb:g})}}{{{la:g}-{lb:g}}}=\dfrac{{{np.log(la/lb):.6f}}}{{{la-lb:g}}}={tmax:.6f}\ \mathrm{{s}}",
          rf"N_B(t_{{\max}})=\left(\dfrac{{{lb:g}}}{{{la:g}}}\right)^{{{lb:g}/{la-lb:g}}}={nbtm:.6f}",
          rf"N_B(20\ \mathrm{{s}})=\dfrac{{{la:g}}}{{{lb-la:g}}}\left(e^{{-20}}-e^{{-5}}\right)={sci(nb20,7)},\qquad N_A={sci(na20,4)},\ \ N_C={nc20:.7f}",
          rf"\langle t\rangle_{{MC}}={mean:.3f}\pm{err:.3f}\ \mathrm{{s}}\ \ \text{{frente a}}\ \ \tau_B={1/lb:g}\ \mathrm{{s}},\qquad T_{{1/2,A}}={th_a:.4f}\ \mathrm{{s}},\ \ T_{{1/2,B}}={th_b:.4f}\ \mathrm{{s}}"]),
        ("Errores comunes y límites de validez",
         r"Es habitual creer que la vida media es el tiempo en que decae todo el material, cuando es el tiempo en que queda $1/e$ de la muestra; la semivida es $\ln2$ veces la vida media. También se confunde el máximo de $N_2$ con el máximo de la actividad $A_2$: ambas coinciden solo en el cruce de actividades, y el máximo de $N_2$ ocurre en el cruce $A_2=A_1$, y a partir de ahí $A_2$ ya está decreciendo, de modo que ambos máximos no coinciden. Con constantes casi iguales la fórmula de Bateman sufre cancelación numérica, y conviene usar el límite degenerado. El modelo falla con muestras pequeñas (fluctuaciones de Poisson dominan), con ramificaciones (hay que sumar las tasas parciales) y cuando se añade alimentación externa, como en un reactor o un generador, donde aparece un término fuente.",
         [r"\lambda_{\mathrm{tot}}=\sum_b\lambda_b,\qquad \dfrac{dN_i}{dt}=\sum_{j}b_{ji}\lambda_jN_j-\lambda_iN_i+S_i(t)"]),
    ]


# ====================================================================== BindingEnergy
def _binding() -> list:
    av, as_, ac, aa = m24.A_V, m24.A_S, m24.A_C, m24.A_A
    a = 56
    za = m24.z_stable_analytic(a)
    zn = m24.z_stable_numeric(a, False)
    bpa = float(m24.b_per_a_analytic(a))
    bpa_int = m24.b_per_a(a, False)
    amax = int(np.argmax(m24.b_per_a_analytic(np.arange(20, 240)))) + 20
    bmax = float(np.max(m24.b_per_a_analytic(np.arange(20, 240))))
    anum = m24.peak_a_numeric()
    # punto fijo A = a_S / (2 a_C zeta^2)
    aa_ = 60.0
    for _ in range(200):
        z0 = aa_ / 2 / (1 + ac * aa_ ** (2 / 3) / (4 * aa))
        aa_ = as_ / (2 * ac * (z0 / aa_) ** 2)
    zeta = za / a
    term_v, term_s = av * a, as_ * a ** (2 / 3)
    term_c = ac * za * (za - 1) / a ** (1 / 3)
    term_a = aa * (a - 2 * za) ** 2 / a
    ef = (197.327**2 / (2 * 938.919)) * (3 * np.pi**2 * 0.16 / 2) ** (2 / 3)
    ac_th = 0.6 * 197.327 / 137.036 / 1.2
    return [
        ("Qué se estudia y por qué importa",
         r"La energía de enlace mide cuánto más ligado está un núcleo que sus nucleones por separado, y su valor por nucleón explica por qué la fusión libera energía en los núcleos ligeros y la fisión en los pesados. La curva $B/A$ crece rápido hasta unos 8 MeV, tiene un máximo cercano al hierro y níquel y baja lentamente. La fórmula semiempírica de masa, propuesta con el modelo de la gota líquida, reproduce esa curva con cinco términos de origen físico claro y predice qué isóbaros son estables, la dirección de los decaimientos beta y la energía que liberan las reacciones nucleares. Es una herramienta central en astrofísica nuclear y en ingeniería de reactores.",
         []),
        ("Magnitudes y convenciones",
         r"La energía de enlace $B(A,Z)$ se expresa en MeV y es positiva para un núcleo ligado: es el trabajo necesario para disgregarlo en $Z$ protones y $N=A-Z$ neutrones. Los coeficientes $a_V,a_S,a_C,a_A$ también están en MeV; sus valores típicos son 16, 18, 0.7 y 23 MeV respectivamente. La energía de enlace por nucleón $B/A$ ronda 8 MeV, y el radio nuclear sigue $R=r_0A^{1/3}$ con $r_0\approx1.2$ fm. La masa atómica se relaciona con $B$ por el defecto de masa.",
         [r"M(A,Z)c^2=Zm_pc^2+Nm_nc^2-B(A,Z),\qquad R=r_0A^{1/3},\qquad \dfrac{e^2}{4\pi\varepsilon_0}=\alpha\hbar c=1.44\ \mathrm{MeV\,fm}"]),
        ("Intuición y analogías",
         r"El núcleo se parece a una gota de líquido incompresible: densidad casi constante, fuerzas de corto alcance que saturan (cada nucleón interactúa solo con sus vecinos inmediatos) y una tensión superficial. A esa gota se le añaden dos efectos de la mecánica cuántica y del electromagnetismo: la repulsión coulombiana entre protones, que crece con $Z^2$, y el principio de exclusión de Pauli, que penaliza los desequilibrios entre neutrones y protones al obligar a ocupar niveles más altos. El equilibrio entre estas tendencias opuestas fija el valle de estabilidad.",
         []),
        ("Términos de volumen y superficie",
         r"Como las fuerzas nucleares tienen corto alcance y saturan, cada nucleón contribuye con una energía de enlace aproximadamente constante: $B_V=a_VA$. Pero los nucleones de la superficie tienen menos vecinos, y por ello menos enlace; el número de nucleones superficiales es proporcional al área $4\pi R^2\propto A^{2/3}$, lo que da un término negativo $-a_SA^{2/3}$. El cociente $B_S/B_V\sim A^{-1/3}$ explica por qué los núcleos ligeros están menos ligados por nucleón que los pesados: tienen una gran fracción de nucleones en la superficie.",
         [r"B_V=a_VA,\qquad B_S=-a_SA^{2/3}\ \ (\text{área}\propto R^2=r_0^2A^{2/3})",
          r"\dfrac{B_V+B_S}{A}=a_V-\dfrac{a_S}{A^{1/3}}"]),
        ("Término de Coulomb",
         r"Los $Z$ protones se repelen. Modelando el núcleo como esfera uniformemente cargada de radio $R$, la energía electrostática es $\frac35\frac{Q^2}{4\pi\varepsilon_0R}$ con $Q=Ze$. Descontando la autoenergía de cada protón (cada uno no se repele a sí mismo) $Z^2$ se sustituye por $Z(Z-1)$. Reemplazando $R=r_0A^{1/3}$ se obtiene la dependencia $Z(Z-1)/A^{1/3}$ y el coeficiente $a_C=\frac35\,\alpha\hbar c/r_0$, que con $r_0=1.2$ fm concuerda con el valor empírico.",
         [r"U_C=\dfrac35\,\dfrac{Z(Z-1)\,e^2}{4\pi\varepsilon_0\,r_0A^{1/3}}\ \Rightarrow\ B_C=-a_C\,\dfrac{Z(Z-1)}{A^{1/3}}",
          rf"a_C=\dfrac35\,\dfrac{{\alpha\hbar c}}{{r_0}}=\dfrac35\cdot\dfrac{{1.44\ \mathrm{{MeV\,fm}}}}{{1.2\ \mathrm{{fm}}}}\approx{ac_th:.3f}\ \mathrm{{MeV}}\ \ (\text{{empírico: }}{ac})"]),
        ("Término de simetría: gas de Fermi",
         r"Neutrones y protones son fermiones distintos, cada especie llena sus propios niveles hasta una energía de Fermi. Con densidad fija, la energía cinética de $n$ fermiones con espín 1/2 en un volumen $V\propto A$ crece como $n^{5/3}$. Desarrollando en torno a $N=Z=A/2$ hasta orden cuadrático en la asimetría, el término lineal se cancela por simetría y queda una penalización $\propto(N-Z)^2/A$. El gas de Fermi da $a_A=\varepsilon_F/3\approx12$ MeV, la mitad del valor empírico, pues el potencial nuclear también depende del isospín.",
         [r"K(n)=\dfrac35\,n\,\varepsilon_F(n),\quad \varepsilon_F=\dfrac{\hbar^2}{2m}\left(\dfrac{3\pi^2n}{V}\right)^{2/3}\ \Rightarrow\ K\propto n^{5/3}V^{-2/3}",
          r"K(\tfrac A2+\delta)+K(\tfrac A2-\delta)\simeq 2K(\tfrac A2)\left[1+\dfrac59\left(\dfrac{N-Z}{A}\right)^2\right]\ \Rightarrow\ \Delta K=\dfrac{\varepsilon_F}{3}\,\dfrac{(N-Z)^2}{A}",
          r"B_A=-a_A\,\dfrac{(A-2Z)^2}{A}=-a_A\,\dfrac{(N-Z)^2}{A}"]),
        ("Estabilidad: el valle de estabilidad Z0(A)",
         r"Para $A$ fijo, el núcleo más estable es el de mayor $B$ (menor masa), que se halla anulando la derivada respecto a $Z$. Se usa $Z(Z-1)\simeq Z^2$ (válido si $A\gg1$) y el hecho de que, a $A$ fijo, los términos de volumen y superficie son constantes. La segunda derivada es negativa, así que se trata de un máximo de $B$. El resultado muestra que $Z_0\to A/2$ para núcleos ligeros, donde domina la simetría, y que $Z_0/A$ disminuye al crecer $A$, porque la repulsión coulombiana (que crece como $A^{2/3}$ por nucleón) favorece el exceso de neutrones.",
         [r"\dfrac{\partial B}{\partial Z}=-\dfrac{2a_CZ}{A^{1/3}}+\dfrac{4a_A(A-2Z)}{A}=0\ \Rightarrow\ Z_0=\dfrac{A/2}{1+\dfrac{a_CA^{2/3}}{4a_A}}",
          r"\dfrac{\partial^2B}{\partial Z^2}=-\dfrac{2a_C}{A^{1/3}}-\dfrac{8a_A}{A}<0,\qquad \dfrac{Z_0}{A}\xrightarrow{A\ll4a_A/a_C}\dfrac12"]),
        ("La curva B/A y su máximo",
         r"Sustituyendo $Z_0(A)$ en la fórmula se obtiene $B/A$ como función de $A$ únicamente. Como $B$ ya es estacionaria respecto de $Z$, la derivada total respecto de $A$ es igual a la parcial a $\zeta=Z/A$ fijo, y se calcula sin derivar $Z_0(A)$. La condición de extremo exige que la energía superficial sea el doble de la de Coulomb por nucleón, resultado clásico de la teoría de la gota. Como $\zeta$ depende de $A$ la condición es implícita y se resuelve iterando; el máximo cae en el grupo del hierro.",
         [r"\dfrac BA=a_V-\dfrac{a_S}{A^{1/3}}-a_C\zeta^2A^{2/3}-a_A(1-2\zeta)^2,\qquad \zeta=\dfrac ZA",
          r"\dfrac{d(B/A)}{dA}\bigg|_{\zeta=\zeta_0}=\dfrac{a_S}{3A^{4/3}}-\dfrac{2a_C\zeta^2}{3A^{1/3}}=0\ \Rightarrow\ A_{\max}=\dfrac{a_S}{2a_C\,\zeta_0(A_{\max})^2}\ \ (E_S=2E_C)"]),
        ("Resultado general",
         r"La energía de enlace de un núcleo $(A,Z)$ se obtiene de la fórmula semiempírica; la estabilidad a $A$ fijo la da $Z_0(A)$, y la curva $B/A(A)$ resultante tiene un máximo que satisface $E_S=2E_C$. Todas las energías son proporcionales a los coeficientes en MeV y las potencias de $A$ son las dictadas por la geometría (volumen, área, carga). Dimensionalmente $Z(Z-1)/A^{1/3}$, $(A-2Z)^2/A$ y $A^{2/3}$ son adimensionales, así que cada término hereda la unidad de su coeficiente (MeV).",
         [r"\boxed{\ B(A,Z)=a_VA-a_SA^{2/3}-a_C\dfrac{Z(Z-1)}{A^{1/3}}-a_A\dfrac{(A-2Z)^2}{A}\ }",
          r"Z_0(A)=\dfrac{A/2}{1+a_CA^{2/3}/4a_A},\qquad \left(\dfrac BA\right)_{\max}\ \text{en}\ \ a_SA^{2/3}=2a_C\zeta_0^2A^{5/3}"]),
        ("Particularización al problema",
         rf"Con $a_V={av}$, $a_S={as_}$, $a_C={ac}$ y $a_A={aa}$ MeV se aplica $Z_0(A)$ a $A=56$ y se evalúa $B/A$ en ese $Z_0$ continuo. Los términos de la fórmula en $A=56$ y $Z_0$ son los siguientes. La condición $E_S=2E_C$, resuelta por iteración, da $A_{{\max}}\approx{aa_:.1f}$, y la búsqueda sobre $A$ entera de la escena da {amax} (continuo) y {anum} (con $Z$ entero), siendo la diferencia consecuencia de $Z(Z-1)\to Z^2$ y de la discretización.",
         [rf"Z_0(56)=\dfrac{{28}}{{1+{ac}\cdot56^{{2/3}}/(4\cdot{aa})}}={za:.3f}\ \ (\text{{entero que maximiza }}B\text{{: }}Z={zn})",
          rf"B=\underbrace{{{term_v:.2f}}}_{{\mathrm{{vol}}}}-\underbrace{{{term_s:.2f}}}_{{\mathrm{{sup}}}}-\underbrace{{{term_c:.2f}}}_{{\mathrm{{Coul}}}}-\underbrace{{{term_a:.2f}}}_{{\mathrm{{sim}}}}={term_v-term_s-term_c-term_a:.2f}\ \mathrm{{MeV}}",
          rf"\dfrac BA(56)={bpa:.3f}\ \mathrm{{MeV}}\ \ (\text{{con }}Z\ \text{{entero: }}{bpa_int:.3f}\ \mathrm{{MeV}}),\qquad \left(\dfrac BA\right)_{{\max}}\approx{bmax:.3f}\ \mathrm{{MeV}}",
          rf"\varepsilon_F\approx{ef:.1f}\ \mathrm{{MeV}}\ \Rightarrow\ \dfrac{{\varepsilon_F}}{{3}}\approx{ef/3:.1f}\ \mathrm{{MeV}}\ \ (a_A=23.7\ \text{{empírico}})"]),
        ("Errores comunes y límites de validez",
         r"Un error común es creer que el máximo de $B/A$ está en el elemento más estable por decaimiento: la estabilidad la fija $B$ a $A$ fijo (valle), mientras que el máximo de $B/A$ compara distintos $A$. También se confunde el signo de las energías: $B>0$ es energía de ligadura y no energía del núcleo. La fórmula es una media suave y no reproduce los números mágicos (2, 8, 20, 28, 50, 82, 126) ligados al modelo de capas, ni la deformación; falla para $A\lesssim20$. Se corrige con el término de apareamiento $\delta=\pm a_P/\sqrt A$ (positivo si $Z$ y $N$ son pares, negativo si ambos impares), que se omitió aquí y explica por qué hay muchos más núcleos estables par-par que impar-impar.",
         [r"B\to B+\delta(A,Z),\qquad \delta=\begin{cases}+a_P/\sqrt A&Z,N\ \text{pares}\\ 0&A\ \text{impar}\\ -a_P/\sqrt A&Z,N\ \text{impares}\end{cases}"]),
    ]


# ====================================================================== TwoBodyDecay
def _twobody() -> list:
    m, m1, m2 = m25.M_PARENT, m25.M_1, m25.M_2
    ps = m25.p_star_analytic()
    e1s = np.hypot(ps, m1)
    e2s = np.hypot(ps, m2)
    e1s_f = (m**2 + m1**2 - m2**2) / (2 * m)
    p_lab = 2000.0
    el = np.hypot(p_lab, m)
    gam, bet = el / m, p_lab / el
    bs = ps / e1s
    lo, hi = m25.lab_energy_range_analytic()
    ph, pl = m25.lab_momentum_limits_analytic()
    kallen = (m**2 - (m1 + m2) ** 2) * (m**2 - (m1 - m2) ** 2)
    q = m - m1 - m2
    mu = m1 * m2 / (m1 + m2)
    p_nr = np.sqrt(2 * mu * q)
    th = np.degrees(np.arctan(bs / (gam * np.sqrt(bet**2 - bs**2))))
    mean, std, pmax, pmin = m25.decay_monte_carlo()
    return [
        ("Qué se estudia y por qué importa",
         r"Casi todas las partículas inestables observadas en aceleradores y rayos cósmicos se detectan a través de sus productos de decaimiento, no directamente. Al reconstruir la masa invariante de las hijas se descubren resonancias (el bosón $Z$, el $J/\psi$, el bosón de Higgs en dos fotones) como picos sobre un fondo suave. El decaimiento a dos cuerpos es el caso más sencillo y completamente determinado por la cinemática: las energías e impulsos de las hijas en el sistema en reposo quedan fijados por las tres masas. Aquí se estudia el $\Lambda\to p\pi^-$, una desintegración débil observada como una «V» en cámaras de burbujas.",
         []),
        ("Magnitudes y convenciones",
         r"Se usa el sistema natural $c=1$: energías, impulsos y masas en MeV (o MeV/$c$ y MeV/$c^2$ al restituir $c$). Las cantidades con asterisco ($E^*,p^*$) se miden en el sistema en reposo de la partícula madre; las sin asterisco, en el laboratorio. La energía total de una partícula es $E=\sqrt{p^2+m^2}$, la velocidad $\beta=p/E$ y el factor de Lorentz $\gamma=E/m$. Los cuadrivectores tienen la forma $p^\mu=(E,\vec p)$ con métrica $(+,-,-,-)$, de modo que $p^\mu p_\mu=E^2-|\vec p|^2=m^2$.",
         [r"p^\mu=(E,\vec p),\qquad p^\mu p_\mu=E^2-|\vec p|^2=m^2,\qquad \vec\beta=\dfrac{\vec p}{E},\quad \gamma=\dfrac Em"]),
        ("Intuición y analogías",
         r"En el sistema en reposo del padre el decaimiento se parece a una explosión de un cuerpo que se parte en dos fragmentos: por conservación del impulso, las hijas salen espalda con espalda con impulsos iguales y opuestos, y la energía liberada se reparte según sus masas. En el laboratorio, el padre se mueve y el patrón se arrastra hacia adelante como la chispa de un petardo lanzado en un tren en marcha. Lo que se mantiene igual en ambos sistemas es la masa invariante, una especie de «longitud» del cuadrivector energía-impulso, y por eso sirve para identificar la partícula original.",
         []),
        ("Invariantes de Lorentz y conservación del cuadrivector",
         r"La conservación de energía e impulso se resume en una ecuación de cuadrivectores: $P^\mu=p_1^\mu+p_2^\mu$. El cuadrado $P^\mu P_\mu$ es un invariante bajo transformaciones de Lorentz, de modo que su valor es el mismo en cualquier sistema inercial y se define como el cuadrado de la masa invariante del sistema. Para una partícula aislada que decae es $M^2$ (su masa en reposo). Aplicado al sistema de las dos hijas, la masa invariante reconstruida debe igualar $M$ evento por evento, cualquiera que sea el ángulo de emisión.",
         [r"P^\mu=p_1^\mu+p_2^\mu,\qquad M^2=P^\mu P_\mu=(E_1+E_2)^2-|\vec p_1+\vec p_2|^2",
          r"M^2=m_1^2+m_2^2+2\left(E_1E_2-\vec p_1\cdot\vec p_2\right)\qquad(\text{invariante})"]),
        ("Cinemática en el sistema en reposo del padre",
         r"En el sistema en reposo $P^\mu=(M,\vec0)$ y $\vec p_1^{\,*}=-\vec p_2^{\,*}$. Pasando $p_1$ al otro lado, $p_2=P-p_1$ y elevando al cuadrado (usando que $p_2^2=m_2^2$ es invariante) se despeja $E_1^*$ sin necesidad de conocer $p^*$. La energía de la segunda hija sigue por simetría intercambiando índices. El impulso $p^*$ sale de la relación $p^{*2}=E_1^{*2}-m_1^2$, que tras un poco de álgebra es la función de Källén de $(M^2,m_1^2,m_2^2)$. Es real si y solo si $M\ge m_1+m_2$, que es la condición de umbral.",
         [r"m_2^2=(P-p_1)^2=M^2+m_1^2-2ME_1^*\ \Rightarrow\ E_1^*=\dfrac{M^2+m_1^2-m_2^2}{2M},\qquad E_2^*=\dfrac{M^2+m_2^2-m_1^2}{2M}",
          r"p^*=\sqrt{E_1^{*2}-m_1^2}=\dfrac{\sqrt{\lambda(M^2,m_1^2,m_2^2)}}{2M}=\dfrac{\sqrt{[M^2-(m_1+m_2)^2][M^2-(m_1-m_2)^2]}}{2M}",
          r"\lambda(x,y,z)=x^2+y^2+z^2-2xy-2yz-2zx"]),
        ("Transformación al laboratorio",
         r"El padre se mueve con impulso $p$ a lo largo de $z$ y, por tanto, con $\beta=p/E$ y $\gamma=E/M$. Las componentes del cuadrivector de la hija en el laboratorio se obtienen aplicando el boost de Lorentz a las del sistema en reposo, donde la dirección de emisión $\theta^*$ respecto de $z$ es una variable libre. La componente transversal no cambia. Estas ecuaciones se verifican porque conservan $E^2-p^2=m_1^2$ para cualquier $\theta^*$, como corresponde a una transformación de Lorentz.",
         [r"E_1=\gamma\left(E_1^*+\beta p^*\cos\theta^*\right),\qquad p_{1\parallel}=\gamma\left(p^*\cos\theta^*+\beta E_1^*\right),\qquad p_{1\perp}=p^*\sin\theta^*",
          r"E_1^2-p_{1\parallel}^2=E_1^{*2}-p^{*2}\cos^2\theta^*\ \Rightarrow\ E_1^2-p_{1\parallel}^2-p_{1\perp}^2=E_1^{*2}-p^{*2}=m_1^2"]),
        ("Distribución en energía y cono de emisión",
         r"Si el decaimiento es isótropo en el sistema en reposo, el ángulo sólido es uniforme y $\cos\theta^*$ se distribuye uniformemente en $[-1,1]$. Como $E_1$ es función lineal de $\cos\theta^*$, la energía del laboratorio también se reparte de manera uniforme entre dos extremos, y el espectro es un rectángulo cuyo ancho $2\gamma\beta p^*$ es proporcional al impulso del padre. Los extremos corresponden a emisión colineal hacia adelante o hacia atrás. Si $\beta>\beta^*=p^*/E_1^*$ la hija nunca sale hacia atrás, y el ángulo de apertura máximo del cono es el dado abajo.",
         [r"\dfrac{dN}{dE_1}=\dfrac{dN}{d\cos\theta^*}\left|\dfrac{d\cos\theta^*}{dE_1}\right|=\dfrac{1}{2\gamma\beta p^*}\qquad E_1\in\left[\gamma(E_1^*-\beta p^*),\ \gamma(E_1^*+\beta p^*)\right]",
          r"\tan\theta_{\max}=\dfrac{\beta^*}{\gamma\sqrt{\beta^2-\beta^{*2}}}\quad(\beta>\beta^*),\qquad \beta^*=\dfrac{p^*}{E_1^*}"]),
        ("Masa invariante reconstruida y límite no relativista",
         r"Al medir $(E_i,\vec p_i)$ de las dos hijas en el laboratorio se calcula $M_{\mathrm{inv}}^2=(E_1+E_2)^2-|\vec p_1+\vec p_2|^2$; por invarianza de Lorentz el resultado es $M^2$ sea cual sea la dirección y el boost, y en un detector real el ancho del pico refleja la resolución experimental y el ancho natural de la resonancia. Cuando la energía liberada $Q=M-m_1-m_2$ es pequeña frente a las masas, el impulso se aproxima por el resultado no relativista de dos cuerpos con masa reducida $\mu$, que sirve de comprobación.",
         [r"M_{\mathrm{inv}}^2=(E_1+E_2)^2-|\vec p_1+\vec p_2|^2=M^2,\qquad Q=M-m_1-m_2",
          r"Q\ll m_i:\ \ p^*\simeq\sqrt{2\mu Q},\qquad \mu=\dfrac{m_1m_2}{m_1+m_2}"]),
        ("Resultado general",
         r"Para una partícula de masa $M$ que se desintegra en dos de masas $m_1$ y $m_2$ y se mueve con impulso $p$ en el laboratorio, el impulso en reposo, el espectro plano de energía y la masa invariante vienen dados por las fórmulas enmarcadas. Las dependencias son: $p^*$ depende solo de las tres masas, el ancho del espectro es proporcional a $p\,p^*/M$, y la masa invariante no depende del ángulo ni del boost. Dimensionalmente, con $c=1$ todas las cantidades están en MeV; al restituir $c$, $p^*$ está en MeV/$c$ y $M$ en MeV/$c^2$.",
         [r"\boxed{\ p^*=\dfrac{\sqrt{[M^2-(m_1+m_2)^2][M^2-(m_1-m_2)^2]}}{2M},\qquad E_1^*=\dfrac{M^2+m_1^2-m_2^2}{2M}\ }",
          r"E_1\in\left[\gamma(E_1^*-\beta p^*),\ \gamma(E_1^*+\beta p^*)\right],\ \ \gamma=\dfrac{\sqrt{p^2+M^2}}{M},\ \ \beta=\dfrac{p}{\sqrt{p^2+M^2}};\qquad M^2=(p_1+p_2)^\mu(p_1+p_2)_\mu"]),
        ("Particularización al problema",
         rf"Se sustituyen $M={m}$ MeV, $m_1={m1}$ MeV (protón) y $m_2={m2}$ MeV ($\pi^-$) y el impulso del laboratorio $p={p_lab:g}$ MeV/$c$. El producto de Källén vale ${sci(kallen,5)}\ \mathrm{{MeV^4}}$. El protón no puede salir hacia atrás, pues $\beta={bet:.4f}>\beta^*={bs:.4f}$. La masa reconstruida con 200 000 eventos reproduce $M$ con dispersión de redondeo.",
         [rf"p^*=\dfrac{{\sqrt{{{sci(kallen,5)}}}}}{{2({m})}}={ps:.4f}\ \mathrm{{MeV}}/c,\qquad E_p^*={e1s:.4f}\ \mathrm{{MeV}}=\dfrac{{M^2+m_1^2-m_2^2}}{{2M}}={e1s_f:.4f}\ \mathrm{{MeV}}",
          rf"\gamma=\dfrac{{\sqrt{{{p_lab:g}^2+{m}^2}}}}{{{m}}}={gam:.4f},\qquad \beta={bet:.4f},\qquad \theta_{{\max}}={th:.3f}^\circ",
          rf"E_p\in[{lo:.2f},\ {hi:.2f}]\ \mathrm{{MeV}},\qquad p_p\in[{pl:.2f},\ {ph:.2f}]\ \mathrm{{MeV}}/c",
          rf"M_{{inv}}={mean:.4f}\ \mathrm{{MeV}},\qquad Q={q:.3f}\ \mathrm{{MeV}},\ \ \sqrt{{2\mu Q}}={p_nr:.2f}\ \mathrm{{MeV}}/c\ \ (\text{{no relativista, }}p^*={ps:.2f})"]),
        ("Errores comunes y límites de validez",
         r"Es un error aplicar la conservación de energía en el laboratorio como si $E_1+E_2=M$: eso solo vale en el sistema en reposo, y en el laboratorio la suma es $\gamma M$. Otro es confundir el impulso $p^*$ con el que se mide en el laboratorio, que depende del ángulo. También se olvida que con una masa de dos partículas sin identificación hay que suponer hipótesis de masa, y una asignación errada desplaza el pico. El tratamiento supone partículas estables en la escala del experimento y ancho nulo; para resonancias anchas la masa invariante sigue una distribución de Breit-Wigner. Si el padre tiene espín y está polarizado, la distribución angular deja de ser isótropa (en el $\Lambda$ aparece $1+\alpha_\Lambda P\cos\theta$ con $\alpha_\Lambda\approx0.75$), y el espectro de energía deja de ser plano, aunque $p^*$ y la masa invariante no cambian.",
         [r"\dfrac{dN}{d\cos\theta^*}\propto1+\alpha P\cos\theta^*,\qquad \dfrac{d\Gamma}{dM_{\mathrm{inv}}}\propto\dfrac{1}{(M_{\mathrm{inv}}^2-M^2)^2+M^2\Gamma^2}"]),
    ]


# ====================================================================== FriedmannAge
def _friedmann() -> list:
    h0, om, ol = m25.H0_PER_GYR, m25.OMEGA_M, m25.OMEGA_L
    t0 = m25.age_analytic()
    arg = np.sqrt(ol / om)
    ash = np.arcsinh(arg)
    a_eq = (om / ol) ** (1 / 3)
    a_acc = (om / (2 * ol)) ** (1 / 3)
    t_of = lambda a: 2 / (3 * h0 * np.sqrt(ol)) * np.arcsinh(a ** 1.5 * np.sqrt(ol / om))  # noqa: E731
    t_acc, t_eq = t_of(a_acc), t_of(a_eq)
    a25 = float(m25.scale_factor_analytic(25.0))
    q0 = om / 2 - ol
    t_hub = 1 / h0
    t_eds = 2 / (3 * h0)
    kms = h0 / 1.0227e-3
    return [
        ("Qué se estudia y por qué importa",
         r"La cosmología pregunta cómo evoluciona el universo en su conjunto y cuánto tiempo ha transcurrido desde su estado denso inicial. Las observaciones de la radiación de fondo, de las supernovas lejanas y de la distribución de galaxias convergen en un universo prácticamente plano, con materia y una componente de energía oscura que lo acelera. La edad que resulta debe superar la de las estrellas más antiguas; esa comprobación de consistencia es una de las pruebas más sencillas del modelo. El modelo de Friedmann plano con materia y constante cosmológica es el núcleo del modelo estándar $\Lambda$CDM.",
         []),
        ("Magnitudes y convenciones",
         r"El factor de escala $a(t)$ mide cómo se estiran las distancias cosmológicas, normalizado a $a(t_0)=1$ hoy. El parámetro de Hubble $H=\dot a/a$ tiene unidades de inverso del tiempo; su valor actual $H_0$ se da en km s$^{-1}$ Mpc$^{-1}$ (68.7 en este problema) y equivale a $0.0703\ \mathrm{Gyr}^{-1}$. Los parámetros de densidad $\Omega_i=\rho_i/\rho_c$ son adimensionales y miden cada componente respecto de la densidad crítica. En un universo plano $\Omega_m+\Omega_\Lambda=1$ (se desprecia la radiación, que es importante solo antes de $a\sim10^{-4}$).",
         [r"H=\dfrac{\dot a}{a},\qquad \rho_c=\dfrac{3H_0^2}{8\pi G},\qquad \Omega_i=\dfrac{\rho_i}{\rho_c},\qquad 1\ \mathrm{km\,s^{-1}Mpc^{-1}}=1.0227\times10^{-3}\ \mathrm{Gyr^{-1}}"]),
        ("Intuición y analogías",
         r"Un modelo mental útil es el de pasas en una masa de pan que crece: ninguna pasa es el centro y todas se alejan de las demás con una velocidad proporcional a su distancia (ley de Hubble). La expansión no es un movimiento a través del espacio sino un estiramiento del propio espacio. La gravedad de la materia frena la expansión, de modo que en el pasado la expansión fue más rápida y la edad resulta menor que el tiempo de Hubble $1/H_0$; la energía oscura, con densidad constante, empuja y hace crecer la edad.",
         []),
        ("Principio cosmológico y deducción newtoniana de la ecuación de Friedmann",
         r"A grandes escalas se supone que el universo es homogéneo e isótropo (principio cosmológico). Considérese una esfera de radio $r=a(t)\,\chi$ y densidad $\rho$ uniforme: según el teorema de Newton solo importa la masa interior $M=\frac{4\pi}3\rho r^3$, y la energía por unidad de masa de una capa en el borde se conserva. Esto equivale a la ecuación de Friedmann, con la constante de integración $-k/2$ que representa la curvatura. La relatividad general da el mismo resultado exacto (con $k$ la curvatura espacial) y además aporta la ecuación para la aceleración, donde la presión también gravita.",
         [r"\dfrac12\dot r^2-\dfrac{GM}{r}=\text{cte}\ \Rightarrow\ \left(\dfrac{\dot a}a\right)^2=\dfrac{8\pi G}{3}\rho-\dfrac{kc^2}{a^2}\quad\left(+\dfrac{\Lambda c^2}{3}\right)",
          r"\dfrac{\ddot a}{a}=-\dfrac{4\pi G}{3}\left(\rho+\dfrac{3p}{c^2}\right)+\dfrac{\Lambda c^2}{3}"]),
        ("Métrica FLRW y ecuaciones de Einstein",
         r"La métrica más general homogénea e isótropa es la de Friedmann-Lemaître-Robertson-Walker, con el factor de escala dependiente solo del tiempo y la curvatura espacial constante $k=0,\pm1$. Insertándola en las ecuaciones de Einstein con un fluido perfecto de densidad $\rho$ y presión $p$ más la constante cosmológica, la componente temporal da la ecuación de Friedmann y las componentes espaciales dan la de aceleración. La conservación de energía-momento, $\nabla_\mu T^{\mu\nu}=0$, sale de ambas y no es independiente.",
         [r"ds^2=-c^2dt^2+a(t)^2\left[\dfrac{dr^2}{1-kr^2}+r^2d\Omega^2\right]",
          r"G_{\mu\nu}+\Lambda g_{\mu\nu}=\dfrac{8\pi G}{c^4}T_{\mu\nu},\qquad T^{\mu}{}_{\nu}=\mathrm{diag}(-\rho c^2,p,p,p)"]),
        ("Contenido material: ecuación de estado y parámetros de densidad",
         r"La conservación de energía en el volumen comóvil, $d(\rho c^2a^3)=-p\,d(a^3)$, con ecuación de estado $p=w\rho c^2$ da $\rho\propto a^{-3(1+w)}$. Para la materia sin presión ($w=0$) se tiene $\rho_m\propto a^{-3}$ (la masa se diluye con el volumen), y para la constante cosmológica ($w=-1$) $\rho_\Lambda$ es constante. Normalizando con la densidad crítica actual y con curvatura nula, $\Omega_m+\Omega_\Lambda=1$, se obtiene la ecuación que gobierna a $a(t)$ y que depende solo de $H_0$ y $\Omega_m$.",
         [r"\dot\rho=-3H(\rho+p/c^2)\ \Rightarrow\ \rho\propto a^{-3(1+w)},\qquad \rho_m=\rho_{m0}a^{-3},\quad \rho_\Lambda=\text{cte}",
          r"H^2=\dfrac{8\pi G}3\left(\rho_{m0}a^{-3}+\rho_\Lambda\right)\ \Rightarrow\ \boxed{\ \left(\dfrac{\dot a}a\right)^2=H_0^2\left(\dfrac{\Omega_m}{a^3}+\Omega_\Lambda\right)\ }"]),
        ("Integración exacta del universo plano",
         r"El cambio de variable $y=a^{3/2}$ linealiza la ecuación: $\dot y=\frac32 a^{1/2}\dot a=\frac32 H a^{3/2}=\frac32H_0\sqrt{\Omega_m+\Omega_\Lambda y^2}$. Esta ecuación es de variables separables, y su integral es un arcoseno hiperbólico, con $y(0)=0$ como condición inicial (la singularidad inicial $a=0$ en $t=0$). Invirtiendo se obtiene $a(t)$ explícito, y la edad es el tiempo en que $a=1$. Todo el cálculo es posible porque en un universo plano con $w=0$ y $w=-1$ el coeficiente sólo depende de $y^2$.",
         [r"\dot y=\dfrac32H_0\sqrt{\Omega_m+\Omega_\Lambda y^2}\ \Rightarrow\ \dfrac1{\sqrt{\Omega_\Lambda}}\,\mathrm{arcsinh}\!\left(y\sqrt{\dfrac{\Omega_\Lambda}{\Omega_m}}\right)=\dfrac32H_0t",
          r"a(t)=\left(\dfrac{\Omega_m}{\Omega_\Lambda}\right)^{1/3}\sinh^{2/3}\!\left(\dfrac32\sqrt{\Omega_\Lambda}\,H_0t\right)",
          r"a(t_0)=1\ \Rightarrow\ t_0=\dfrac{2}{3H_0\sqrt{\Omega_\Lambda}}\,\mathrm{arcsinh}\sqrt{\dfrac{\Omega_\Lambda}{\Omega_m}}"]),
        ("Casos límite, aceleración y transiciones",
         r"A tiempos tempranos $\sinh x\simeq x$, de donde $a\propto t^{2/3}$ (universo de Einstein-de Sitter dominado por materia); para $t$ grande $\sinh x\simeq e^{x}/2$ y $a\propto e^{\sqrt{\Omega_\Lambda}H_0t}$ (de Sitter). Si $\Omega_\Lambda\to0$ la edad tiende a $2/(3H_0)$, y si $\Omega_m\to0$ crece sin límite. El parámetro de desaceleración hoy, $q_0=\Omega_m/2-\Omega_\Lambda$, es negativo cuando $\Omega_\Lambda>\Omega_m/2$. La aceleración comienza cuando $\ddot a=0$, es decir $\rho_m=2\rho_\Lambda$, y la igualdad de densidades se da en $a_{eq}$.",
         [r"\Omega_\Lambda\to0:\ \ t_0\to\dfrac{2}{3H_0\sqrt{\Omega_m}},\qquad \Omega_m\to0:\ \ t_0\to\infty,\qquad q_0=-\dfrac{a\ddot a}{\dot a^2}\bigg|_0=\dfrac{\Omega_m}{2}-\Omega_\Lambda",
          r"a_{\mathrm{acel}}=\left(\dfrac{\Omega_m}{2\Omega_\Lambda}\right)^{1/3},\qquad a_{\mathrm{eq}}=\left(\dfrac{\Omega_m}{\Omega_\Lambda}\right)^{1/3}"]),
        ("Resultado general",
         r"Para un universo plano con materia y constante cosmológica, el factor de escala y la edad quedan en forma cerrada en función de $H_0$ y $\Omega_m$ (con $\Omega_\Lambda=1-\Omega_m$). La edad se escribe $t_0=H_0^{-1}f(\Omega_m)$ con $f$ adimensional; por eso el tiempo de Hubble $1/H_0$ fija la escala y el modelo fija el factor numérico. Dimensionalmente, $H_0t$ es adimensional y $a$ también. Para $\Omega_m=0.31$, $f$ es cercano a $1$, un poco menor que el tiempo de Hubble.",
         [r"\boxed{\ t_0=\dfrac{2}{3H_0\sqrt{\Omega_\Lambda}}\,\mathrm{arcsinh}\sqrt{\dfrac{\Omega_\Lambda}{\Omega_m}},\qquad a(t)=\left(\dfrac{\Omega_m}{\Omega_\Lambda}\right)^{1/3}\sinh^{2/3}\!\left(\tfrac32\sqrt{\Omega_\Lambda}H_0t\right)\ }",
          r"t_0=\dfrac{1}{H_0}\,f(\Omega_m),\qquad f=\dfrac{2}{3\sqrt{\Omega_\Lambda}}\,\mathrm{arcsinh}\sqrt{\dfrac{\Omega_\Lambda}{\Omega_m}}"]),
        ("Particularización al problema",
         rf"Con $H_0={h0}\ \mathrm{{Gyr^{{-1}}}}$ ({kms:.1f} km s$^{{-1}}$ Mpc$^{{-1}}$), $\Omega_m={om}$ y $\Omega_\Lambda={ol}$ se evalúa la fórmula general. El tiempo de Hubble es $1/H_0={t_hub:.3f}$ Gyr, la edad resulta una fracción $f=t_0H_0={t0*h0:.4f}$ de él y supera la de un universo sólo de materia con $\Omega_m=1$ ($2/3H_0={t_eds:.3f}$ Gyr). La aceleración empezó cuando $a={a_acc:.4f}$ (hace unos {t0-t_acc:.2f} Gyr).",
         [rf"\sqrt{{\Omega_\Lambda/\Omega_m}}=\sqrt{{{ol}/{om}}}={arg:.4f},\qquad \mathrm{{arcsinh}}({arg:.4f})={ash:.4f}",
          rf"t_0=\dfrac{{2}}{{3({h0})\sqrt{{{ol}}}}}\cdot{ash:.4f}=\dfrac{{2}}{{{3*h0*np.sqrt(ol):.5f}}}\cdot{ash:.4f}={t0:.4f}\ \mathrm{{Gyr}}",
          rf"a(25\ \mathrm{{Gyr}})=({om}/{ol})^{{1/3}}\sinh^{{2/3}}\!\left(\tfrac32\sqrt{{{ol}}}({h0})(25)\right)={a25:.5f}",
          rf"q_0=\dfrac{{{om}}}{{2}}-{ol}={q0:.3f}<0,\qquad a_{{eq}}={a_eq:.4f}\ (t={t_eq:.2f}\ \mathrm{{Gyr}}),\qquad a_{{\mathrm{{acel}}}}={a_acc:.4f}\ (t={t_acc:.2f}\ \mathrm{{Gyr}})"]),
        ("Errores comunes y límites de validez",
         r"Se piensa con frecuencia que el universo se expande «hacia» algún lugar o desde un punto central; en realidad no hay centro, y las galaxias lejanas pueden alejarse más rápido que $c$ sin violar la relatividad, ya que es el espacio el que se estira. Otro error es identificar $1/H_0$ con la edad: solo coincide en un universo con expansión uniforme (vacío). El modelo no incluye radiación, indispensable para los primeros $\sim10^5$ años y que corrige la edad en menos del $0.1\,\%$, ni la curvatura ($\Omega_k\ne0$), que impide la forma cerrada elemental. Tampoco describe la inflación ni las inhomogeneidades, y asume que la energía oscura es una constante cosmológica ($w=-1$), hipótesis en estudio observacional.",
         [r"H^2=H_0^2\left(\dfrac{\Omega_r}{a^4}+\dfrac{\Omega_m}{a^3}+\dfrac{\Omega_k}{a^2}+\Omega_\Lambda\right),\qquad t_0=\dfrac1{H_0}\int_0^1\dfrac{da}{a\sqrt{\Omega_r a^{-4}+\Omega_ma^{-3}+\Omega_ka^{-2}+\Omega_\Lambda}}"]),
    ]


DETAIL = {
    "BandStructure": dict(concept=_band()),
    "DiodeCircuit": dict(concept=_diode()),
    "DecayChain": dict(concept=_chain()),
    "BindingEnergy": dict(concept=_binding()),
    "TwoBodyDecay": dict(concept=_twobody()),
    "FriedmannAge": dict(concept=_friedmann()),
}
