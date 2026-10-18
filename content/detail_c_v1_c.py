"""Concepto y analisis analitico general. Escenas: CenterOfMass, ExplodingProjectile, Collision1D,
BallisticPendulum, SpinUp, RollingCycloid, MomentOfInertia, HangingMassDisc.

Cada entrada es dict(concept=[(titulo, prosa, [ecuaciones TeX]), ...]); los numeros de la
particularizacion se calculan con physics/ (nunca a mano)."""
from __future__ import annotations

import numpy as np

from content.texfmt import fx, sci
from physics import ch09_particle_systems as c9
from physics import ch10_collisions as c10
from physics import ch11_rotational_kinematics as c11
from physics import ch12_rotational_dynamics as c12


# ====================================================================== CenterOfMass
def _cm() -> dict:
    m, p = c9.MASSES, c9.POSITIONS
    M = float(m.sum())
    cm = c9.cm_discrete_analytic()
    L = c9.ROD_L
    rod = c9.rod_cm_analytic()
    lam_n = [(n, (n + 1) * L / (n + 2)) for n in (0, 1, 2)]
    xs = " + ".join(f"({fx(mi,0)})({fx(pi,0)})" for mi, pi in zip(m, p[:, 0]))
    ys = " + ".join(f"({fx(mi,0)})({fx(pi,0)})" for mi, pi in zip(m, p[:, 1]))
    return dict(concept=[
        ("Qué se estudia y por qué importa",
         "El centro de masa es el punto que representa el movimiento de traslación de todo un sistema, sea un conjunto de partículas, una barra o una nave con combustible. Permite reemplazar un problema de muchos cuerpos por el de una sola partícula ficticia sometida a la fuerza externa total. Aparece al diseñar vehículos y estructuras, al estimar la estabilidad de un objeto, al analizar explosiones y choques, y en astronomía al describir sistemas binarios, donde ambas estrellas giran alrededor de un punto común.",
         []),
        ("Magnitudes y convenciones",
         r"Las masas $m_i$ se miden en kg, las posiciones $\vec r_i$ en m y la densidad lineal $\lambda(x)$ en kg/m, de modo que $dm=\lambda\,dx$ tiene unidades de masa. La masa total es $M=\sum m_i$ o $M=\int dm$, y el centro de masa $\vec R$ tiene unidades de longitud y es independiente de la escala de masas: solo importan las razones $m_i/M$. El resultado depende del origen elegido para las posiciones, pero el punto físico no.",
         [r"\vec R=\dfrac{1}{M}\sum_i m_i\vec r_i,\qquad [\vec R]=\mathrm{m}"]),
        ("Intuición y analogías",
         "El centro de masa es un promedio ponderado: cada partícula tira de la posición promedio en proporción a su masa, como en un balancín donde el punto de apoyo se desplaza hacia el lado más pesado. Si se cuelga un cuerpo de ese punto, queda en equilibrio en cualquier orientación bajo gravedad uniforme. En un sistema aislado, el centro de masa se mueve en línea recta aunque las partes se muevan de forma complicada, porque las fuerzas internas no pueden empujar al conjunto.",
         []),
        ("Separación de coordenadas: centro de masa y relativas",
         r"Para $N$ partículas con posiciones $\vec r_i$ se define $\vec R=\frac1M\sum m_i\vec r_i$ y las posiciones relativas $\vec r_i'=\vec r_i-\vec R$. La definición de $\vec R$ equivale exactamente a la condición $\sum m_i\vec r_i'=0$, de modo que de las $3N$ coordenadas solo $3N-3$ relativas son independientes. Esta descomposición es un cambio de coordenadas generalizadas, legítimo porque es invertible, y es el paso previo para escribir el lagrangiano.",
         [r"\vec r_i=\vec R+\vec r_i',\qquad \sum_i m_i\vec r_i'=\vec 0",
          r"\vec v_i=\vec V+\vec v_i',\qquad \vec V=\dot{\vec R},\quad \sum_i m_i\vec v_i'=\vec 0"]),
        ("Energía cinética y teorema de König",
         r"Al sustituir $\vec v_i=\vec V+\vec v_i'$ en $T=\sum\tfrac12m_iv_i^2$, el término cruzado $\vec V\cdot\sum m_i\vec v_i'$ se anula por la condición anterior. Queda una suma de dos energías independientes: la del movimiento del punto $\vec R$ con toda la masa y la del movimiento relativo. Este es el teorema de König, y justifica tratar la traslación del conjunto por separado de la dinámica interna.",
         [r"T=\sum_i\tfrac12m_i\left|\vec V+\vec v_i'\right|^2=\tfrac12MV^2+\vec V\cdot\underbrace{\sum_i m_i\vec v_i'}_{=0}+\sum_i\tfrac12m_iv_i'^2",
          r"T=\underbrace{\tfrac12MV^2}_{T_{cm}}+\underbrace{\sum_i\tfrac12m_iv_i'^2}_{T_{rel}}"]),
        ("Ecuación de movimiento del centro de masa",
         r"Sea $L=T-U$ con $U=\sum U_i^{ext}(\vec r_i)+\tfrac12\sum_{i\neq j}U_{ij}(|\vec r_i-\vec r_j|)$. El potencial interno depende solo de diferencias de posición y por tanto no depende de $\vec R$; el externo aporta la fuerza $\vec F_{ext}=\sum\vec F_i^{ext}$. La ecuación de Euler-Lagrange para la coordenada $\vec R$ da directamente el movimiento del centro de masa, sin invocar la tercera ley: la cancelación de fuerzas internas es consecuencia de la invariancia traslacional de $U_{int}$.",
         [r"\dfrac{d}{dt}\dfrac{\partial L}{\partial\dot{\vec R}}-\dfrac{\partial L}{\partial\vec R}=0\;\Rightarrow\;M\ddot{\vec R}=\sum_i\vec F_i^{ext}=\vec F_{ext}",
          r"\vec P=\dfrac{\partial L}{\partial\dot{\vec R}}=M\vec V,\qquad \dot{\vec P}=\vec F_{ext}"]),
        ("Simetría y conservación (Noether)",
         r"Si $\vec F_{ext}=0$, el lagrangiano no depende de $\vec R$: es invariante bajo traslaciones, y el teorema de Noether asegura que $\vec P$ se conserva. Si la fuerza externa es uniforme, como la gravedad $m_i\vec g$, el centro de masa cae como una sola partícula de masa $M$ con aceleración $\vec g$ sin importar cómo se mueven las partes. La conservación vale componente a componente: basta que una componente de $\vec F_{ext}$ sea nula para que la correspondiente de $\vec P$ se conserve.",
         [r"\vec F_{ext}=M\vec g\;\Rightarrow\;\vec R(t)=\vec R_0+\vec V_0t+\tfrac12\vec gt^2",
          r"\vec F_{ext}\cdot\hat n=0\;\Rightarrow\;P_n=\text{constante}"]),
        ("Límite continuo y familia de densidades",
         r"Cuando el número de partículas crece, la suma se convierte en integral con $dm=\rho\,dV$; en una dimensión $dm=\lambda(x)\,dx$. Como familia general se toma $\lambda(x)=\lambda_0(x/L)^n$ con $n\ge0$ en una barra de longitud $L$, que contiene el caso uniforme ($n=0$) y el del enunciado ($n=1$). La integral converge para $n>-1$ porque el integrando es una potencia en un intervalo finito.",
         [r"M=\int_0^L\lambda_0\left(\dfrac xL\right)^ndx=\dfrac{\lambda_0L}{n+1}",
          r"X_{cm}=\dfrac{1}{M}\int_0^Lx\,\lambda_0\left(\dfrac xL\right)^ndx=\dfrac{\lambda_0L^2/(n+2)}{\lambda_0L/(n+1)}=\dfrac{n+1}{n+2}L"]),
        ("Casos límite, simetrías y cambio de origen",
         r"Si el origen se traslada una cantidad $\vec a$, cada $\vec r_i\to\vec r_i-\vec a$ y $\vec R\to\vec R-\vec a$ porque $\sum m_i/M=1$: el centro de masa es un punto físico y no depende de la elección del origen. Si un cuerpo tiene un plano, eje o centro de simetría y su densidad es simétrica, $\vec R$ está sobre ese elemento. Si dos subsistemas tienen masas $M_1$, $M_2$ y centros $\vec R_1$, $\vec R_2$, el centro del conjunto es el promedio ponderado de ambos, lo que permite calcular por partes figuras compuestas.",
         [r"\vec R=\dfrac{M_1\vec R_1+M_2\vec R_2}{M_1+M_2}",
          r"n\to\infty:\ X_{cm}\to L\ \text{(toda la masa en el extremo)},\qquad n=0:\ X_{cm}=\dfrac L2"]),
        ("Resultado general",
         r"Para un sistema discreto o continuo el centro de masa es el primer momento de la distribución de masa dividido por la masa total, y su movimiento está gobernado únicamente por la fuerza externa. Para la barra con densidad potencial, el resultado crece de $L/2$ a $L$ al aumentar $n$. Dimensionalmente, $\sum m_ir_i/M$ es longitud, y la razón $(n+1)/(n+2)$ es adimensional.",
         [r"\boxed{\;\vec R=\dfrac{\sum_im_i\vec r_i}{\sum_im_i}\ \ \text{o}\ \ \vec R=\dfrac{\int\vec r\,dm}{\int dm},\qquad M\ddot{\vec R}=\vec F_{ext}\;}",
          r"\boxed{\;X_{cm}=\dfrac{n+1}{n+2}\,L\quad\text{para}\ \lambda=\lambda_0(x/L)^n\;}"]),
        ("Particularización al problema",
         rf"Con masas $2$, $3$ y $5$ kg en $(0,0)$, $(4,0)$ y $(2,3)$ m se tiene $M={fx(M,0)}$ kg. Cada coordenada es el promedio ponderado por masa. Para la barra de $L={fx(L,0)}$ m con $\lambda\propto x$ se toma $n=1$ en la fórmula general, y se recupera el valor de la escena.",
         [rf"x_{{cm}}=\dfrac{{{xs}}}{{{fx(M,0)}}}={fx(cm[0],2)}\ \mathrm{{m}},\qquad y_{{cm}}=\dfrac{{{ys}}}{{{fx(M,0)}}}={fx(cm[1],2)}\ \mathrm{{m}}",
          rf"X_{{cm}}=\dfrac{{n+1}}{{n+2}}L=\dfrac{{2}}{{3}}({fx(L,0)})={fx(rod,4)}\ \mathrm{{m}}",
          rf"n=0:\ {fx(lam_n[0][1],4)}\ \mathrm{{m}},\qquad n=1:\ {fx(lam_n[1][1],4)}\ \mathrm{{m}},\qquad n=2:\ {fx(lam_n[2][1],4)}\ \mathrm{{m}}"]),
        ("Errores comunes y límites de validez",
         r"Un error frecuente es promediar las posiciones sin pesar por la masa, o creer que el centro de masa está dentro del cuerpo: en un anillo o en una herradura cae en el vacío. Otro es confundir centro de masa con centro de gravedad; coinciden solo si $\vec g$ es uniforme sobre el cuerpo. Tampoco es cierto que $\vec R$ se mueva sin fuerzas si hay fuerzas externas, aunque sean pequeñas. El modelo clásico falla a velocidades cercanas a $c$, donde el centro de energía depende del observador y se debe usar el centro de momento relativista.",
         []),
    ])


# ============================================================== ExplodingProjectile
def _explode() -> dict:
    v0, g, th = c9.V0, c9.G, np.deg2rad(c9.ANGLE_DEG)
    mA, mB = c9.M_A, c9.M_B
    M = mA + mB
    ap = c9.apex_state()
    an = c9.fragments_analytic()
    R = c9.range_analytic()
    vx = ap["vx"]
    q = 0.5 * mB * an["vb"] ** 2 - 0.5 * M * vx**2
    xb_gen = ap["x"] * (1 + M / mB)
    return dict(concept=[
        ("Qué se estudia y por qué importa",
         "Una explosión es un proceso en el que fuerzas internas muy intensas y de corta duración redistribuyen el momento y liberan energía almacenada, como en un cohete que se separa por etapas, un fuego artificial o la desintegración de un núcleo. Aunque el detalle de la ruptura es complejo, la física global se resuelve con dos ideas: el momento se conserva durante el proceso y el centro de masa sigue su trayectoria original. Esto permite predecir dónde caen los fragmentos sin conocer cómo se rompió el objeto.",
         []),
        ("Magnitudes y convenciones",
         r"El proyectil tiene masa $M$ (kg), rapidez inicial $v_0$ (m/s) y ángulo $\theta$ sobre la horizontal; $g$ es la aceleración de la gravedad (m/s$^2$). Los fragmentos tienen masas $m_A$, $m_B$ con $m_A+m_B=M$ y velocidades horizontales justo tras la explosión $u_A$, $u_B$ (positivas hacia adelante). La energía liberada $Q$ se mide en joules y es siempre positiva, porque una explosión convierte energía interna en cinética.",
         []),
        ("Intuición y analogías",
         "Imagine una persona sobre patines que lanza un objeto: ella retrocede para que el momento total siga siendo cero. En la explosión ocurre lo mismo: si un fragmento pierde velocidad horizontal, el otro la gana. El centro de masa es como el punto de equilibrio de la pareja de fragmentos, y mientras ambos vuelan, ese punto sigue dibujando la misma parábola, como si nada hubiera pasado, hasta que el primer fragmento toca el suelo.",
         []),
        ("Movimiento previo: la parábola y el vértice",
         r"Antes de la explosión el proyectil es una partícula con $L=\tfrac12M(\dot x^2+\dot y^2)-Mgy$. La coordenada $x$ es cíclica, de modo que $p_x=M\dot x$ es constante; la ecuación de $y$ da aceleración uniforme $-g$. El vértice es el punto donde $\dot y=0$, y allí la velocidad es puramente horizontal, lo que simplifica el balance de momento.",
         [r"x(t)=v_0\cos\theta\,t,\qquad y(t)=v_0\sin\theta\,t-\tfrac12gt^2",
          r"t_a=\dfrac{v_0\sin\theta}{g},\quad x_a=\dfrac{v_0^2\sin\theta\cos\theta}{g}=\dfrac R2,\quad y_a=\dfrac{v_0^2\sin^2\theta}{2g},\quad R=\dfrac{v_0^2\sin2\theta}{g}"]),
        ("La explosión: balance de impulso",
         r"Se integra la ecuación de movimiento de cada fragmento en el intervalo corto $\Delta t$ de la explosión. El impulso de las fuerzas internas es finito (son intensas), mientras que el de la gravedad es $O(m g\,\Delta t)\to0$. Por tanto la suma de los momentos justo antes y justo después coincide, aunque la energía mecánica no se conserve. Las fuerzas internas son iguales y opuestas, de modo que sus impulsos se cancelan en la suma.",
         [r"\Delta\vec p_A=\int\vec F_{AB}\,dt+O(\Delta t),\qquad \Delta\vec p_B=-\int\vec F_{AB}\,dt+O(\Delta t)",
          r"M v_x=m_Au_A+m_Bu_B\quad(\text{horizontal}),\qquad 0=m_Av_{Ay}+m_Bv_{By}\quad(\text{vertical})"]),
        ("Velocidades de los fragmentos",
         r"Si el fragmento A cae verticalmente a partir del reposo tras la explosión, $u_A=0$. En general se deja $u_A$ libre y se despeja $u_B$ del balance horizontal. Tanto el momento vertical antes como después es nulo, así que ambos fragmentos comienzan su caída con $v_y=0$. La energía liberada es la diferencia entre la energía cinética final e inicial, que es positiva y que acota qué repartos de momento son posibles.",
         [r"u_B=\dfrac{Mv_x-m_Au_A}{m_B},\qquad v_x=v_0\cos\theta",
          r"Q=\tfrac12m_Au_A^2+\tfrac12m_Bu_B^2-\tfrac12Mv_x^2\;\ge\;0"]),
        ("Vuelo de los fragmentos y posición de caída",
         r"Tras la explosión cada fragmento tiene $L_k=\tfrac12m_k(\dot x_k^2+\dot y_k^2)-m_kgy_k$ y se mueve independientemente, con $x$ cíclica. Ambos parten de la altura $y_a$ con $v_y=0$ y tardan el mismo tiempo en caer, $t_f=\sqrt{2y_a/g}$, que resulta igual al tiempo de ascenso $t_a$. Las posiciones de caída son entonces lineales en las velocidades horizontales.",
         [r"t_f=\sqrt{\dfrac{2y_a}{g}}=\dfrac{v_0\sin\theta}{g}=t_a",
          r"x_k=x_a+u_k\,t_f,\qquad k=A,B"]),
        ("El centro de masa sigue su parábola",
         r"Como la única fuerza externa es la gravedad, $M\ddot{\vec R}=M\vec g$ antes y después de la explosión, y la posición y la velocidad de $\vec R$ son continuas en la explosión. Por tanto, si ningún fragmento hubiera tocado el suelo, $\vec R(t)$ seguiría la parábola original. Al instante del impacto ($t=2t_a$), $X_{cm}=R$. Esto equivale a decir que $m_Ax_A+m_Bx_B=MR$, una verificación independiente que no requiere conocer $u_A$.",
         [r"x_{cm}(t_a+t_f)=\dfrac{m_Ax_A+m_Bx_B}{M}=x_a+\dfrac{m_Au_A+m_Bu_B}{M}t_f=x_a+v_xt_f",
          r"v_xt_f=\dfrac{v_0^2\sin\theta\cos\theta}{g}=x_a\;\Rightarrow\;x_{cm}=2x_a=R"]),
        ("Parámetros adimensionales y casos límite",
         r"El reparto de masa $\mu=m_A/M$ controla el alcance del fragmento B: si $u_A=0$, $x_B=x_a(1+M/m_B)=x_a\,(2-\mu)/(1-\mu)$, que diverge cuando $\mu\to1$ porque B tiene poca masa y recibe toda la velocidad. Si $u_A=-v_x$ (A sale hacia atrás con la rapidez del proyectil) cae en el origen del lanzamiento. Si la explosión ocurre fuera del vértice, la misma conservación del momento vale, pero hay que incluir la componente vertical y los tiempos de caída ya no coinciden.",
         [r"u_A=0:\ x_B=x_a\left(1+\dfrac{M}{m_B}\right),\qquad u_A=-v_x\ \Rightarrow\ x_A=x_a-v_xt_f=0"]),
        ("Resultado general",
         r"La posición de caída de cada fragmento es la del vértice más su velocidad horizontal tras la explosión por el tiempo de caída, y el centro de masa final coincide con el alcance sin explosión. La energía liberada depende solo del reparto de velocidades. Las unidades son metros para $x_k$, m/s para $u_k$ y joules para $Q$.",
         [r"\boxed{\;x_k=\dfrac{v_0^2\sin\theta\cos\theta}{g}+u_k\dfrac{v_0\sin\theta}{g},\qquad \dfrac{m_Ax_A+m_Bx_B}{M}=\dfrac{v_0^2\sin2\theta}{g}\;}",
          r"\boxed{\;u_B=\dfrac{Mv_0\cos\theta-m_Au_A}{m_B}\;}"]),
        ("Particularización al problema",
         rf"Con $M={fx(M,0)}$ kg, $m_A={fx(mA,0)}$ kg, $m_B={fx(mB,0)}$ kg, $v_0={fx(v0,0)}$ m/s, $\theta=60^\circ$ y $u_A=0$, se sustituye en la fórmula general y se recuperan las cifras de la escena.",
         [rf"v_x={fx(vx,2)}\ \mathrm{{m/s}},\quad x_a={fx(ap['x'],3)}\ \mathrm{{m}},\quad y_a={fx(ap['y'],3)}\ \mathrm{{m}},\quad t_f={fx(an['t_fall'],4)}\ \mathrm{{s}}",
          rf"u_B=\dfrac{{({fx(M,0)})({fx(vx,2)})}}{{{fx(mB,0)}}}={fx(an['vb'],2)}\ \mathrm{{m/s}},\qquad x_A={fx(an['xa'],2)}\ \mathrm{{m}}",
          rf"x_B=x_a\left(1+\dfrac{{M}}{{m_B}}\right)={fx(xb_gen,2)}\ \mathrm{{m}}={fx(an['xb'],2)}\ \mathrm{{m}},\qquad x_{{cm}}=\dfrac{{({fx(mA,0)})({fx(an['xa'],2)})+({fx(mB,0)})({fx(an['xb'],2)})}}{{{fx(M,0)}}}={fx(an['x_cm'],2)}\ \mathrm{{m}}=R",
          rf"Q={fx(q,2)}\ \mathrm{{J}}"]),
        ("Errores comunes y límites de validez",
         r"Es erróneo conservar la energía cinética en la explosión: se libera energía interna y $Q>0$. También se suele olvidar que el centro de masa sigue la parábola solo mientras ningún fragmento haya tocado el suelo, pues el suelo introduce una fuerza externa. Otra confusión es suponer que el fragmento que cae vertical no se mueve: su velocidad horizontal es nula respecto del suelo, no respecto del proyectil. La aproximación de impulso instantáneo falla si la explosión es lenta, pues entonces la gravedad y el arrastre actúan durante el proceso.",
         []),
    ])


# ==================================================================== Collision1D
def _collision() -> dict:
    m1, m2, v1 = c10.M1, c10.M2, c10.V1
    M = m1 + m2
    mu = m1 * m2 / M
    V = m1 * v1 / M
    u = v1
    k_rel = 0.5 * mu * u**2
    out = {e: c10.outcome_analytic(e) for e in (1.0, 0.5, 0.0)}
    th_max = float(np.degrees(np.arcsin(m2 / m1))) if m1 > m2 else 90.0
    zetas = [(z, c10.restitution_from_damping(z)) for z in (0.0, 0.3, 0.7)]
    rows = "".join(
        rf"e={e:.1f}:\ v_1'={fx(out[e][0],3)},\ v_2'={fx(out[e][1],3)},\ \Delta K={fx(-(k_rel*(1-e**2)),3)}\ \mathrm{{J}}\\ " for e in out)
    return dict(concept=[
        ("Qué se estudia y por qué importa",
         "Una colisión es una interacción intensa y breve entre cuerpos que cambia sus velocidades. Su estudio sostiene la física de partículas (los aceleradores se interpretan midiendo desviaciones), la seguridad vial, el billar, la deformación de materiales y la reconstrucción de accidentes. En muchos casos no se conoce la fuerza de contacto, pero la conservación del momento y una sola cantidad que describe la pérdida de energía bastan para predecir el resultado.",
         []),
        ("Magnitudes y convenciones",
         r"Las masas $m_1$, $m_2$ se dan en kg y las velocidades $v_1$, $v_2$ en m/s, con signo según un eje fijo. El coeficiente de restitución $e\in[0,1]$ es adimensional: $e=1$ es elástica y $e=0$ perfectamente inelástica. La masa reducida $\mu=m_1m_2/(m_1+m_2)$ es una masa efectiva en kg, siempre menor que la menor de las dos masas.",
         []),
        ("Intuición y analogías",
         "Visto desde el centro de masa, una colisión es sencillo: ambos cuerpos se acercan con momentos iguales y opuestos y se alejan con momentos iguales y opuestos. Una colisión elástica solo invierte las velocidades relativas, como una pelota que rebota sobre una pared muy pesada; una inelástica las atenúa. La energía que se pierde es exactamente la asociada al movimiento relativo, nunca la del movimiento del conjunto.",
         []),
        ("Reducción a un problema de un cuerpo",
         r"Para dos partículas con potencial de interacción $U(|\vec r_1-\vec r_2|)$ el lagrangiano es $L=\tfrac12m_1\dot{\vec r}_1^2+\tfrac12m_2\dot{\vec r}_2^2-U$. Con $\vec R=(m_1\vec r_1+m_2\vec r_2)/M$ y $\vec r=\vec r_1-\vec r_2$ resulta un lagrangiano separado: un término libre en $\vec R$ y un problema de una partícula de masa $\mu$ en el potencial $U(r)$. La coordenada $\vec R$ es cíclica, de modo que $\vec P$ se conserva (simetría de traslación).",
         [r"L=\tfrac12M\dot{\vec R}^2+\tfrac12\mu\dot{\vec r}^2-U(r),\qquad \mu=\dfrac{m_1m_2}{m_1+m_2}",
          r"\vec r_1=\vec R+\dfrac{m_2}{M}\vec r,\qquad \vec r_2=\vec R-\dfrac{m_1}{M}\vec r"]),
        ("Cinemática en el sistema del centro de masa",
         r"En el sistema CM, que se mueve con $\vec V=(m_1\vec v_1+m_2\vec v_2)/M$, los momentos son $\pm\vec p^{*}$ con $\vec p^{*}=\mu\vec u$ y $\vec u=\vec v_1-\vec v_2$ la velocidad relativa. La energía cinética total se separa en la del CM, que no cambia, y la relativa $\tfrac12\mu u^2$, que es la única disponible para disiparse. Es el mismo teorema de König del tema de sistemas de partículas.",
         [r"\vec v_1=\vec V+\dfrac{m_2}{M}\vec u,\qquad \vec v_2=\vec V-\dfrac{m_1}{M}\vec u",
          r"K=\tfrac12MV^2+\tfrac12\mu u^2,\qquad \vec p_1^{*}=-\vec p_2^{*}=\mu\vec u"]),
        ("Colisión unidimensional con restitución",
         r"En una dimensión la conservación del momento fija $V$ y no puede cambiar $u$ más que en módulo y sentido. La restitución se define como $u'=-eu$: la velocidad relativa de separación es $e$ veces la de aproximación. Con esto las dos incógnitas $v_1'$, $v_2'$ quedan determinadas sin conocer el contacto. En el caso $m_2$ en reposo se reducen a la expresión conocida con $V=m_1v_1/M$ y $u=v_1$.",
         [r"v_1'=V-e\dfrac{m_2}{M}u,\qquad v_2'=V+e\dfrac{m_1}{M}u",
          r"v_1'=\dfrac{m_1-em_2}{m_1+m_2}v_1,\qquad v_2'=\dfrac{(1+e)m_1}{m_1+m_2}v_1\quad(v_2=0)"]),
        ("Energía perdida y origen físico de la restitución",
         r"La energía cinética perdida es $\Delta K=\tfrac12\mu u'^2-\tfrac12\mu u^2=-\tfrac12\mu u^2(1-e^2)$, siempre no positiva y nula solo para $e=1$. Un modelo físico de $e$ es un contacto con resorte $k$ y amortiguador $c$ para la coordenada relativa: $\mu\ddot x+c\dot x+kx=0$. Durante el contacto, que dura media oscilación amortiguada, la velocidad relativa se reduce por el factor $e=\exp(-\pi\zeta/\sqrt{1-\zeta^2})$ con $\zeta=c/(2\sqrt{k\mu})$.",
         [r"\Delta K=-\tfrac12\mu u^2\,(1-e^2)",
          r"e=\exp\!\left(-\dfrac{\pi\zeta}{\sqrt{1-\zeta^2}}\right),\qquad \zeta=\dfrac{c}{2\sqrt{k\mu}}"]),
        ("Dispersión en dos dimensiones y ángulos",
         r"Con un blanco en reposo y una colisión elástica, en el CM las partículas solo giran un ángulo $\Theta$ sin cambiar el módulo de sus velocidades. Pasar al laboratorio suma $\vec V$. El ángulo de dispersión de la partícula incidente en el laboratorio y la fracción de energía transferida son funciones de $\Theta$; si $m_1>m_2$ existe un ángulo máximo de desviación.",
         [r"\tan\theta_1=\dfrac{\sin\Theta}{m_1/m_2+\cos\Theta},\qquad \dfrac{T_2}{T_1}=\dfrac{4m_1m_2}{M^2}\sin^2\dfrac\Theta2",
          r"m_1>m_2:\ \sin\theta_{1,\max}=\dfrac{m_2}{m_1};\qquad m_1=m_2:\ \theta_1+\theta_2=\dfrac\pi2"]),
        ("Ángulo de dispersión y sección eficaz",
         r"Para un potencial central, el momento angular $l=\mu u b$ (con $b$ el parámetro de impacto) y la energía en el CM $E^{*}=\tfrac12\mu u^2$ se conservan, y el ángulo $\Theta$ sigue de integrar la ecuación orbital. La sección eficaz diferencial cuenta cuántas partículas se desvían a un ángulo dado. Para esferas duras de radio $a$ resulta isótropa en el CM, con un valor que no depende del ángulo.",
         [r"\Theta(b)=\pi-2\int_{r_{\min}}^{\infty}\dfrac{b\,dr/r^2}{\sqrt{1-b^2/r^2-U(r)/E^{*}}}",
          r"\dfrac{d\sigma}{d\Omega}=\dfrac{b}{\sin\Theta}\left|\dfrac{db}{d\Theta}\right|,\qquad \text{esfera dura: } b=a\cos\dfrac\Theta2\ \Rightarrow\ \dfrac{d\sigma}{d\Omega}=\dfrac{a^2}{4}"]),
        ("Resultado general",
         r"La velocidad final de cada cuerpo se obtiene de la velocidad del CM y de la velocidad relativa invertida y atenuada por $e$; la energía disipada depende solo de $\mu$, $u$ y $e$. Las unidades son m/s para velocidades y J para $\Delta K$, y las relaciones son homogéneas en las masas (dependen solo de $m_2/m_1$), como debe ser.",
         [r"\boxed{\;v_{1}'=V-e\,\dfrac{m_2}{M}(v_1-v_2),\qquad v_{2}'=V+e\,\dfrac{m_1}{M}(v_1-v_2),\qquad V=\dfrac{m_1v_1+m_2v_2}{M}\;}",
          r"\boxed{\;\Delta K=-\tfrac12\mu\,(v_1-v_2)^2\,(1-e^2)\;}"]),
        ("Particularización al problema",
         rf"Con $m_1={fx(m1,0)}$ kg, $m_2={fx(m2,0)}$ kg, $v_1={fx(v1,0)}$ m/s y $v_2=0$ se tiene $M={fx(M,0)}$ kg, $\mu={fx(mu,4)}$ kg, $V={fx(V,2)}$ m/s y $u={fx(u,1)}$ m/s. Se sustituye para $e=1$, $0.5$ y $0$; las velocidades coinciden con las de la escena. Para $m_1>m_2$ el ángulo máximo de dispersión es {th_max:.1f}$^\circ$.",
         [rf"\begin{{aligned}}{rows}\end{{aligned}}",
          rf"K_{{rel}}=\tfrac12\mu u^2={fx(k_rel,3)}\ \mathrm{{J}},\qquad \theta_{{1,\max}}=\arcsin\dfrac{{m_2}}{{m_1}}={th_max:.1f}^\circ",
          rf"\zeta=0,\,0.3,\,0.7\ \Rightarrow\ e={fx(zetas[0][1],4)},\ {fx(zetas[1][1],4)},\ {fx(zetas[2][1],4)}"]),
        ("Errores comunes y límites de validez",
         r"Es común aplicar conservación de la energía cinética en cualquier choque, cuando solo el momento se conserva siempre (si no hay impulso externo). También se confunde $e$ con una propiedad de un solo cuerpo, cuando depende del par de materiales y de la velocidad de impacto. La reducción al CM y la restitución constante dejan de valer si el contacto es largo y hay fuerzas externas durante el choque, si la velocidad se acerca a $c$ (hay que usar cuadrimomentos) o en escalas cuánticas, donde la dispersión se describe con amplitudes y desfases.",
         []),
    ])


# ============================================================== BallisticPendulum
def _pendulum() -> dict:
    m, M, v, l, g = c10.BULLET, c10.BLOCK, c10.BULLET_V, c10.ROPE, c10.G
    an = c10.bullet_block_analytic()
    k0 = 0.5 * m * v**2
    th = float(np.degrees(np.arccos(1 - an["h"] / l)))
    return dict(concept=[
        ("Qué se estudia y por qué importa",
         "El péndulo balístico fue durante siglos el instrumento para medir la velocidad de un proyectil, antes de existir los cronógrafos electrónicos. Combina dos fases con leyes distintas: una colisión brevísima, donde se conserva el momento pero no la energía, y un ascenso lento, donde se conserva la energía mecánica pero no el momento. Comprender por qué cada ley vale en su fase es la lección central, y el mismo esquema aparece en el análisis de impactos contra blindajes y en ensayos de materiales.",
         []),
        ("Magnitudes y convenciones",
         r"Se usan $m$ y $M$ para las masas de la bala y del bloque (kg), $v$ para la rapidez de la bala (m/s), $\ell$ para la longitud de la cuerda (m) y $g$ para la gravedad. $V$ es la velocidad del conjunto justo tras el impacto, $h$ la altura máxima ganada y $\theta_{\max}$ el ángulo de la cuerda en el punto más alto. La fracción de energía perdida es adimensional y vale entre 0 y 1.",
         []),
        ("Intuición y analogías",
         "Al incrustarse, la bala cede su momento a un objeto mucho más pesado, y la velocidad resultante es pequeña: el bloque se lleva casi todo el momento pero muy poca energía. Casi toda la energía de la bala se convierte en deformación y calor. Después el bloque se comporta como una pelota que sube por una rampa circular, intercambiando energía cinética por potencial.",
         []),
        ("Fase 1: colisión impulsiva",
         r"Durante el impacto actúan fuerzas internas muy grandes y de corta duración; la tensión de la cuerda y el peso son finitos y su impulso tiende a cero con la duración del choque. Además la tensión pasa por el punto de suspensión y no produce torque respecto de él. Por eso se conservan tanto el momento lineal horizontal como el momento angular respecto de la suspensión, y ambos dan el mismo resultado cuando la bala se incrusta a la distancia $\ell$.",
         [r"mv=(m+M)V,\qquad L_O:\ mv\ell=(m+M)V\ell",
          r"V=\dfrac{m}{m+M}\,v"]),
        ("Energía disipada en la colisión",
         r"La energía cinética antes y después no coincide porque el choque es perfectamente inelástico. La pérdida se escribe en términos de la masa reducida $\mu=mM/(m+M)$ como $\tfrac12\mu v^2$, que es la energía relativa disponible. La fracción perdida resulta $M/(m+M)$, cercana a 1 cuando $M\gg m$.",
         [r"K_0=\tfrac12mv^2,\qquad K_1=\tfrac12(m+M)V^2=\dfrac{m}{m+M}K_0",
          r"\Delta K=K_1-K_0=-\tfrac12\mu v^2,\qquad \dfrac{\Delta K}{K_0}=-\dfrac{M}{m+M}"]),
        ("Fase 2: el péndulo como sistema lagrangiano",
         r"Con el ángulo $\theta$ como coordenada generalizada (la ligadura de cuerda inextensible elimina la otra coordenada), $T=\tfrac12(m+M)\ell^2\dot\theta^2$ y $U=-(m+M)g\ell\cos\theta$. La ecuación de Euler-Lagrange es la del péndulo simple y no depende de la masa total. Como $L$ no depende explícitamente del tiempo, el hamiltoniano $H=T+U$ se conserva.",
         [r"L=\tfrac12(m+M)\ell^2\dot\theta^2+(m+M)g\ell\cos\theta",
          r"\ddot\theta+\dfrac g\ell\sin\theta=0,\qquad p_\theta=(m+M)\ell^2\dot\theta,\quad H=\dfrac{p_\theta^2}{2(m+M)\ell^2}-(m+M)g\ell\cos\theta"]),
        ("Integral de energía y altura máxima",
         r"Se evalúa $H$ en el instante inicial de la fase ($\theta=0$, $\dot\theta=V/\ell$) y en el punto de retorno ($\dot\theta=0$). El resultado vincula el ángulo máximo con $V$ y es independiente de las masas. La altura es $h=\ell(1-\cos\theta_{\max})$, y el ascenso con cuerda tensa y retorno exige $\theta_{\max}\le90^\circ$, es decir $V^2\le2g\ell$; si $V^2\ge5g\ell$ el bloque completa la vuelta.",
         [r"\tfrac12V^2=g\ell(1-\cos\theta_{\max})\;\Rightarrow\;\cos\theta_{\max}=1-\dfrac{V^2}{2g\ell}",
          r"h=\dfrac{V^2}{2g},\qquad \text{cuerda tensa si } V^2\le2g\ell\ (\theta_{\max}\le90^\circ)"]),
        ("Cálculo de la velocidad de la bala y sensibilidad",
         r"Invirtiendo las dos fases se obtiene la velocidad de la bala a partir de una altura medida, que es el uso histórico del aparato. Como $v\propto\sqrt h$ y $h$ es pequeña, una incertidumbre relativa $\delta h/h$ produce la mitad de incertidumbre relativa en $v$. Para oscilaciones pequeñas, el tiempo hasta la altura máxima es un cuarto del periodo, $\tfrac\pi2\sqrt{\ell/g}$, pero para amplitudes grandes el periodo crece con $\theta_{\max}$.",
         [r"v=\dfrac{m+M}{m}\sqrt{2gh}=\dfrac{m+M}{m}\sqrt{2g\ell(1-\cos\theta_{\max})}",
          r"\dfrac{\delta v}{v}=\tfrac12\dfrac{\delta h}{h},\qquad T_{pend}\simeq2\pi\sqrt{\dfrac{\ell}{g}}\left(1+\dfrac{\theta_{\max}^2}{16}\right)"]),
        ("Resultado general",
         r"La velocidad del conjunto es la de la bala reducida por el cociente de masas, la altura sigue de la energía y la fracción disipada depende solo de la razón de masas. Todas las relaciones son homogéneas: $V$ y $v$ en m/s, $h$ en m y la fracción adimensional. Para $m\ll M$ resulta $V\approx mv/M$ y la pérdida casi total.",
         [r"\boxed{\;V=\dfrac{m\,v}{m+M},\qquad h=\dfrac{1}{2g}\left(\dfrac{m\,v}{m+M}\right)^2,\qquad \dfrac{|\Delta K|}{K_0}=\dfrac{M}{m+M}\;}",
          r"\boxed{\;\theta_{\max}=\arccos\!\left[1-\dfrac{1}{2g\ell}\left(\dfrac{mv}{m+M}\right)^2\right]\;}"]),
        ("Particularización al problema",
         rf"Con $m={fx(m,3)}$ kg, $M={fx(M,1)}$ kg, $v={fx(v,0)}$ m/s y $\ell={fx(l,1)}$ m se sustituye en las fórmulas generales; se obtienen las mismas cifras que en la escena.",
         [rf"V=\dfrac{{({fx(m,3)})({fx(v,0)})}}{{{fx(m+M,3)}}}={fx(an['V'],3)}\ \mathrm{{m/s}},\qquad h=\dfrac{{V^2}}{{2g}}={fx(an['h'],4)}\ \mathrm{{m}}",
          rf"\theta_{{\max}}=\arccos\!\left(1-\dfrac{{{fx(an['h'],4)}}}{{{fx(l,1)}}}\right)={th:.2f}^\circ,\qquad K_0={fx(k0,1)}\ \mathrm{{J}}",
          rf"\dfrac{{|\Delta K|}}{{K_0}}=\dfrac{{{fx(M,1)}}}{{{fx(m+M,3)}}}={fx(an['loss_fraction']*100,2)}\%,\qquad v=\dfrac{{m+M}}{{m}}\sqrt{{2gh}}={fx(an['v_from_h'],2)}\ \mathrm{{m/s}}"]),
        ("Errores comunes y límites de validez",
         r"El error clásico es igualar la energía antes y después de la colisión, o el momento antes y después del ascenso (la tensión lo cambia). El modelo supone choque instantáneo, bala puntual incrustada a la altura del centro de masa y cuerda sin masa; si el bloque es extenso hay que usar $I\omega$ con el momento de inercia del cuerpo rígido, y el ángulo cambia. También deja de valer cuando $\theta_{\max}$ es grande y la cuerda pierde tensión, o cuando la bala atraviesa el bloque y se lleva momento.",
         []),
    ])


# ============================================================================ SpinUp
def _spinup() -> dict:
    al, tf, R = c11.ALPHA, c11.T_END, c11.RIM
    w, th = c11.omega_analytic(tf), c11.theta_analytic(tf)
    rim = c11.rim_accelerations(tf)
    phi = float(np.degrees(np.arctan2(al, w**2)))
    kk = 0.25
    return dict(concept=[
        ("Qué se estudia y por qué importa",
         "La rotación alrededor de un eje fijo es el movimiento de ruedas, turbinas, discos duros, volantes y planetas. Su descripción cinemática reproduce la de la traslación cambiando $x,v,a$ por $\\theta,\\omega,\\alpha$, y su dinámica sigue del torque neto. Entender cómo se acelera un punto del borde, con una componente tangencial que cambia la rapidez y otra centrípeta que cambia la dirección, es indispensable para dimensionar cojinetes, correas y frenos.",
         []),
        ("Magnitudes y convenciones",
         r"El ángulo $\theta$ se mide en radianes (adimensional), la velocidad angular $\omega=\dot\theta$ en rad/s y la aceleración angular $\alpha=\dot\omega$ en rad/s$^2$. El signo sigue la regla de la mano derecha: antihorario es positivo respecto del eje de salida. Para un punto a distancia $r$ del eje, $v=\omega r$ en m/s, $a_t=\alpha r$ y $a_c=\omega^2r$ en m/s$^2$; el radián es adimensional, de modo que estas relaciones cierran dimensionalmente.",
         []),
        ("Intuición y analogías",
         "Cada punto de una rueda recorre una circunferencia distinta pero todos giran el mismo ángulo en el mismo tiempo, de ahí que $\\omega$ y $\\alpha$ describan al cuerpo entero. Al acelerar, el borde va más rápido que el interior, igual que en una pista circular el corredor exterior debe cubrir más distancia. La aceleración centrípeta crece con el cuadrado de $\\omega$ y pronto supera a la tangencial, por lo que la aceleración total se acerca a apuntar hacia el centro.",
         []),
        ("Dinámica de un eje fijo: ecuación de Euler-Lagrange",
         r"Un cuerpo rígido que gira alrededor de un eje fijo tiene un único grado de libertad, $\theta$. Su energía cinética es $T=\tfrac12I\dot\theta^2$ y el torque externo $\tau(t)$ actúa como fuerza generalizada $Q_\theta=\tau$. La ecuación de Euler-Lagrange da la segunda ley rotacional, de la cual $\alpha(t)=\tau(t)/I$ es la aceleración angular que se tomará como dato (en el enunciado, $\alpha$ constante).",
         [r"\dfrac{d}{dt}\dfrac{\partial T}{\partial\dot\theta}-\dfrac{\partial T}{\partial\theta}=Q_\theta\;\Rightarrow\;I\ddot\theta=\tau(t)",
          r"\alpha(t)=\ddot\theta=\dfrac{\tau(t)}{I}"]),
        ("Integración general de la cinemática",
         r"Dada $\alpha(t)$ cualquiera, la velocidad y el ángulo se obtienen integrando una y dos veces, con constantes $\omega_0$ y $\theta_0$ fijadas por las condiciones iniciales. Esta integral siempre existe si $\alpha$ es continua a trozos. La eliminación del tiempo da una relación directa entre $\omega$ y $\Delta\theta$, útil cuando el tiempo no es dato.",
         [r"\omega(t)=\omega_0+\int_0^t\alpha(t')\,dt',\qquad \theta(t)=\theta_0+\omega_0t+\int_0^t\!\!\int_0^{t'}\alpha(t'')\,dt''dt'",
          r"\alpha=\text{cte}:\ \omega=\omega_0+\alpha t,\quad \theta=\theta_0+\omega_0t+\tfrac12\alpha t^2,\quad \omega^2=\omega_0^2+2\alpha\,\Delta\theta"]),
        ("Trabajo y energía en la rotación",
         r"La potencia del torque es $\tau\omega$, y su integral temporal es el trabajo, que se convierte en energía cinética de rotación: es el teorema trabajo-energía para el grado de libertad $\theta$. Si $\tau$ es constante, el trabajo es $\tau\Delta\theta$. Esta relación permite calcular la energía entregada sin integrar la cinemática en el tiempo.",
         [r"\dfrac{dK}{dt}=\tau\omega,\qquad K=\tfrac12I\omega^2",
          r"W=\int\tau\,d\theta=\Delta K=\tfrac12I\left(\omega^2-\omega_0^2\right)"]),
        ("Punto del borde: velocidad y aceleración vectoriales",
         r"Un punto a distancia $R$ del eje está en $\vec r=R(\cos\theta,\sin\theta)$. Derivando, la velocidad es tangencial con módulo $\omega R$, y la aceleración tiene una componente tangencial $\alpha R$ y una centrípeta $\omega^2R$ dirigida al eje. Como son ortogonales, el módulo total sigue de Pitágoras, y el ángulo con la dirección radial depende solo del cociente $\alpha/\omega^2$.",
         [r"\vec v=\dot\theta R(-\sin\theta,\cos\theta)=\vec\omega\times\vec r",
          r"\vec a=\alpha R\,\hat\theta-\omega^2R\,\hat r,\qquad |\vec a|=R\sqrt{\alpha^2+\omega^4},\quad \tan\varphi=\dfrac{\alpha}{\omega^2}"]),
        ("Aceleración variable y casos límite",
         r"Para $\alpha=k\,t$ las integrales dan $\omega=kt^2/2$ y $\theta=kt^3/6$, que confirma que el método general vale para aceleración no constante. A tiempos largos $\omega^2\sim t^4\gg\alpha\sim t$, así que la aceleración se vuelve casi puramente centrípeta; a tiempos cortos domina la tangencial. Si $\alpha=0$ se recupera la rotación uniforme.",
         [r"\alpha=kt:\quad \omega=\omega_0+\dfrac{k}{2}t^2,\quad \theta=\theta_0+\omega_0t+\dfrac{k}{6}t^3",
          r"\dfrac{a_t}{a_c}=\dfrac{\alpha}{\omega^2}\ \to\ \infty\ (t\to0),\qquad \to0\ (t\to\infty)"]),
        ("Resultado general",
         r"Con aceleración angular constante la velocidad crece linealmente y el ángulo cuadráticamente; la aceleración de un punto del borde tiene módulo creciente con el tiempo. Unidades: rad/s, rad, m/s$^2$. Verificación dimensional: $\alpha t$ tiene unidades de rad/s y $\alpha^2+\omega^4$ de s$^{-4}$, de modo que $R\sqrt{\cdots}$ es m/s$^2$.",
         [r"\boxed{\;\omega=\omega_0+\alpha t,\qquad \theta=\omega_0t+\tfrac12\alpha t^2,\qquad |\vec a|=R\sqrt{\alpha^2+\left(\omega_0+\alpha t\right)^4}\;}",
          r"\boxed{\;a_t=\alpha R,\qquad a_c=(\omega_0+\alpha t)^2R\;}"]),
        ("Particularización al problema",
         rf"Con $\omega_0=0$, $\alpha={fx(al,1)}$ rad/s$^2$, $t={fx(tf,0)}$ s y $R={fx(R,2)}$ m, las fórmulas generales dan las cifras de la escena. Para aceleración variable $\alpha=0.25\,t$ el ángulo es $kt^3/6$ con $k={fx(kk,2)}$ rad/s$^3$.",
         [rf"\omega=({fx(al,1)})({fx(tf,0)})={fx(w,1)}\ \mathrm{{rad/s}},\qquad \theta=\tfrac12({fx(al,1)})({fx(tf,0)})^2={fx(th,1)}\ \mathrm{{rad}}",
          rf"a_t=({fx(al,1)})({fx(R,2)})={fx(rim['a_t'],2)}\ \mathrm{{m/s^2}},\qquad a_c=({fx(w,1)})^2({fx(R,2)})={fx(rim['a_c'],1)}\ \mathrm{{m/s^2}}",
          rf"|\vec a|={fx(R,2)}\sqrt{{{fx(al,1)}^2+{fx(w,1)}^4}}={fx(rim['a'],2)}\ \mathrm{{m/s^2}},\qquad \varphi=\arctan\dfrac{{\alpha}}{{\omega^2}}={phi:.2f}^\circ",
          rf"\theta_{{var}}=\dfrac{{({fx(kk,2)})({fx(tf,0)})^3}}{{6}}={fx(kk*tf**3/6,3)}\ \mathrm{{rad}}"]),
        ("Errores comunes y límites de validez",
         r"Es frecuente usar grados con las fórmulas de $\omega$ y $\alpha$ (que requieren radianes), o aplicar las ecuaciones de $\alpha$ constante cuando el torque varía. También se olvida la componente centrípeta y se toma $a=\alpha R$ como aceleración total. El modelo supone un cuerpo rígido: a velocidades grandes el material se deforma por el esfuerzo centrífugo $\rho\omega^2R^2$, y cerca de $c$ el borde no puede superar esa velocidad.",
         []),
    ])


# ================================================================== RollingCycloid
def _cycloid() -> dict:
    R = c11.R_WHEEL
    s = c11.arch_length_analytic()
    A = c11.arch_area_analytic()
    circ = 2 * np.pi * R
    area_disc = np.pi * R**2
    d = 0.5 * R
    a_troc = 2 * np.pi * R**2 + np.pi * d**2
    return dict(concept=[
        ("Qué se estudia y por qué importa",
         "Rodar sin deslizar es la forma más eficiente de mover cargas: ruedas de vehículos, rodamientos y engranajes. El movimiento de un punto del borde, la cicloide, no es un círculo ni una recta, sino una curva con picos donde el punto se detiene instantáneamente. Esta curva tiene propiedades notables: es la curva de descenso más rápido (braquistócrona) y la del reloj isócrono (tautócrona), y se usa en el diseño de dientes de engranaje y de perfiles de rodadura.",
         []),
        ("Magnitudes y convenciones",
         r"El radio $R$ y el desplazamiento $x$ están en metros; $\theta$ es el ángulo girado en radianes y la condición de rodadura liga $x_c=R\theta$. El punto de contacto con el suelo tiene velocidad nula en el sistema del suelo (no desliza). Para un punto situado a la distancia $d$ del centro, $d=R$ corresponde al borde (cicloide) y $d<R$ a un punto interior (trocoide acortada).",
         []),
        ("Intuición y analogías",
         "Visto desde el suelo, la rueda gira alrededor del punto de contacto en cada instante, como si fuera una palanca que pivota sobre él. Por eso el punto más alto se mueve el doble de rápido que el centro y el de contacto no se mueve. Un punto del borde sube y baja mientras avanza, y traza arcos idénticos que se repiten cada vuelta, separados por picos.",
         []),
        ("La ligadura de rodadura",
         r"La condición de no deslizamiento exige que la velocidad del punto de contacto sea cero: $v_{cm}-\omega R=0$. Como esta condición es integrable (se puede escribir como relación entre coordenadas), la ligadura es holónoma: $x_c=R\theta$. Con ello los dos grados de libertad $(x_c,\theta)$ se reducen a uno. En una rueda ideal la fuerza de rozamiento estático no realiza trabajo, porque se aplica en un punto instantáneamente en reposo.",
         [r"v_{cm}=\omega R,\qquad x_c=R\theta,\qquad y_c=R",
          r"P_{contacto}:\ \vec v=\vec v_{cm}+\vec\omega\times(-R\hat y)=\vec0"]),
        ("Ecuaciones paramétricas de la trocoide",
         r"Un punto fijo a la rueda a distancia $d$ del centro, que en $\theta=0$ está en el suelo, tiene posición igual a la del centro más un vector que gira con la rueda. En los puntos del borde ($d=R$) se obtiene la cicloide. La velocidad se deriva respecto de $\theta$ y se multiplica por $\dot\theta=\omega$.",
         [r"x=R\theta-d\sin\theta,\qquad y=R-d\cos\theta",
          r"\dot x=\omega\,(R-d\cos\theta),\qquad \dot y=\omega\,d\sin\theta"]),
        ("Velocidad, energía y curvatura de la cicloide",
         r"Para $d=R$ el módulo de la velocidad es $|\vec v|=2\omega R\sin(\theta/2)$: nula en el contacto ($\theta=0$) y $2v_{cm}$ en el punto más alto. La energía cinética de la rueda sigue de rotar alrededor del punto de contacto, con inercia $I_P=I_{cm}+MR^2$ por el teorema de ejes paralelos, es decir, suma de la traslación del centro y la rotación propia. El radio de curvatura vale $4R\sin(\theta/2)$: es nulo en los picos y $4R$ en la cima.",
         [r"|\vec v|=\omega R\sqrt{2-2\cos\theta}=2\omega R\sin\dfrac\theta2",
          r"K=\tfrac12I_P\omega^2=\tfrac12Mv_{cm}^2+\tfrac12I_{cm}\omega^2,\qquad \rho_{curv}=4R\sin\dfrac\theta2"]),
        ("Longitud de un arco",
         r"El elemento de arco es $ds=|\vec v|\,dt=|d\vec r/d\theta|\,d\theta$. Con la identidad $1-\cos\theta=2\sin^2(\theta/2)$ la raíz cuadrada se simplifica y la integral se evalúa en forma cerrada en $0\le\theta\le2\pi$. El resultado es exactamente $8R$, no relacionado con $\pi$, y cuatro veces el diámetro. Para una trocoide con $d\neq R$ la integral se vuelve elíptica.",
         [r"s=\int_0^{2\pi}R\sqrt{2-2\cos\theta}\,d\theta=2R\int_0^{2\pi}\sin\dfrac\theta2\,d\theta=2R\left[-2\cos\dfrac\theta2\right]_0^{2\pi}=8R",
          r"s_{troc}=\int_0^{2\pi}\sqrt{R^2+d^2-2Rd\cos\theta}\,d\theta\quad(\text{integral elíptica})"]),
        ("Área bajo el arco",
         r"El área entre el arco y el suelo es $\int y\,dx$ con $dx=(R-d\cos\theta)\,d\theta$; el cambio de variable a $\theta$ es válido porque $x(\theta)$ es monótona creciente (su derivada no es negativa). Al integrar $(1-\cos\theta)^2=1-2\cos\theta+\cos^2\theta$ sobre un periodo, el término lineal se anula y el cuadrático aporta $\pi$. El área vale tres veces la del disco que rueda. Para la trocoide el resultado general es $2\pi R^2+\pi d^2$.",
         [r"A=\int_0^{2\pi}y\,\dfrac{dx}{d\theta}\,d\theta=\int_0^{2\pi}(R-d\cos\theta)^2\,d\theta=2\pi R^2+\pi d^2",
          r"d=R:\ A=R^2\int_0^{2\pi}(1-\cos\theta)^2d\theta=3\pi R^2"]),
        ("Propiedad variacional y casos límite",
         r"El perfil invertido de la cicloide minimiza el tiempo de descenso entre dos puntos (braquistócrona): el funcional $T=\int ds/\sqrt{2gy}$ tiene como extremal una cicloide, que se obtiene de la ecuación de Euler-Lagrange cuando el integrando no depende de $x$ (integral primera de Beltrami). Además, el periodo de una partícula sobre ella es independiente de la amplitud. Si $d\to0$ la trocoide se reduce a una recta y la longitud tiende a $2\pi R$; si $d\to\infty$ se vuelve casi circular.",
         [r"T=\int\sqrt{\dfrac{1+y'^2}{2gy}}\,dx,\qquad \dfrac{\partial\mathcal L}{\partial y'}y'-\mathcal L=\text{constante}\ \Rightarrow\ y\,(1+y'^2)=2R",
          r"T_{taut}=4\pi\sqrt{\dfrac Rg}\quad(\text{independiente de la amplitud})"]),
        ("Resultado general",
         r"La longitud de un arco depende linealmente de $R$ y el área de su cuadrado, con coeficientes numéricos fijos que no dependen de ninguna escala. Verificación dimensional: $8R$ es una longitud en m y $3\pi R^2$ un área en m$^2$. En términos de la trayectoria del centro, el arco es $4/\pi$ veces más largo que el camino $2\pi R$ del centro.",
         [r"\boxed{\;s=8R,\qquad A=3\pi R^2\;}",
          r"\boxed{\;x=R(\theta-\sin\theta),\quad y=R(1-\cos\theta),\quad |\vec v|=2v_{cm}\sin\dfrac\theta2\;}"]),
        ("Particularización al problema",
         rf"Con $R={fx(R,2)}$ m se sustituye en $s=8R$ y $A=3\pi R^2$. Como control, el camino del centro es $2\pi R$ y el disco que rueda tiene área $\pi R^2$.",
         [rf"s=8({fx(R,2)})={fx(s,3)}\ \mathrm{{m}},\qquad A=3\pi({fx(R,2)})^2={fx(A,4)}\ \mathrm{{m^2}}",
          rf"2\pi R={fx(circ,4)}\ \mathrm{{m}},\qquad \dfrac{{s}}{{2\pi R}}={fx(s/circ,4)}\approx\dfrac4\pi,\qquad \dfrac{{A}}{{\pi R^2}}={fx(A/area_disc,2)}",
          rf"\text{{Trocoide }}d=R/2:\ A=2\pi R^2+\pi d^2={fx(a_troc,4)}\ \mathrm{{m^2}}"]),
        ("Errores comunes y límites de validez",
         r"Se suele creer que el arco de la cicloide mide $2\pi R$, que es el camino del centro, o que el área bajo el arco es $\pi R^2$ (la del disco); en realidad es el triple. También se confunde la velocidad del punto del borde con $\omega R$: solo el centro tiene $v_{cm}=\omega R$, y el borde varía entre $0$ y $2\omega R$. El modelo falla si hay deslizamiento (la ligadura deja de ser válida) o si la rueda se deforma, como en un neumático blando, donde el contacto es un área y el punto de velocidad nula se desplaza.",
         []),
    ])


# =============================================================== MomentOfInertia
def _inertia() -> dict:
    m, L = c12.M_ROD, c12.L_ROD
    ie, ic = c12.rod_inertia_analytic("end"), c12.rod_inertia_analytic("center")
    md, rd = 4.0, 0.25
    idisc = c12.disc_inertia_analytic()
    ix = 0.25 * md * rd**2
    return dict(concept=[
        ("Qué se estudia y por qué importa",
         "El momento de inercia mide la resistencia de un cuerpo a cambiar su rotación, igual que la masa mide la resistencia a cambiar la traslación. Depende no solo de cuánta masa hay, sino de cómo está distribuida respecto del eje: la misma masa lejos del eje cuesta mucho más girar. Determina el diseño de volantes, el comportamiento de patinadores, la estabilidad de satélites y la respuesta de cualquier máquina rotativa.",
         []),
        ("Magnitudes y convenciones",
         r"La inercia rotacional $I$ se expresa en kg$\cdot$m$^2$, y siempre depende del eje elegido y no solo del cuerpo. La densidad puede ser lineal $\lambda$ (kg/m), superficial $\sigma$ (kg/m$^2$) o volumétrica $\rho$ (kg/m$^3$). Es útil escribir $I=\kappa MR^2$ con $\kappa$ adimensional que depende de la forma: anillo $1$, disco $1/2$, esfera maciza $2/5$.",
         []),
        ("Intuición y analogías",
         "Un patinador que abre los brazos aumenta su inercia porque la masa de los brazos se aleja del eje, y por eso gira más despacio. De igual modo, una barra es más fácil de girar alrededor de su centro que de un extremo, porque en el primer caso la masa promedio está más cerca del eje. Como el cuadrado de la distancia pesa, las partes lejanas dominan la integral.",
         []),
        ("Energía cinética y tensor de inercia",
         r"Para un cuerpo rígido que gira con velocidad angular $\vec\omega$ alrededor de un punto fijo, cada elemento tiene velocidad $\vec v=\vec\omega\times\vec r$. La energía cinética es $T=\tfrac12\int\rho\,|\vec\omega\times\vec r|^2dV$. Usando la identidad $|\vec\omega\times\vec r|^2=\omega^2r^2-(\vec\omega\cdot\vec r)^2$ se llega a una forma cuadrática en $\vec\omega$ cuyos coeficientes forman el tensor de inercia.",
         [r"T=\tfrac12\int\rho\left[\omega^2r^2-(\vec\omega\cdot\vec r)^2\right]dV=\tfrac12\,\omega_iI_{ij}\omega_j",
          r"I_{ij}=\int\rho(\vec r)\left(r^2\delta_{ij}-x_ix_j\right)dV"]),
        ("Propiedades del tensor y ejes principales",
         r"$I_{ij}$ es simétrico y definido positivo, de modo que es diagonalizable por una rotación ortogonal: existen tres ejes principales con momentos $I_1,I_2,I_3>0$ (nunca todos pueden superar la suma de los otros dos). El momento angular $\vec L=\mathsf I\vec\omega$ es paralelo a $\vec\omega$ solo si éste está sobre un eje principal. Para un eje fijo de dirección $\hat n$, el momento de inercia escalar es $I=\hat n\cdot\mathsf I\,\hat n$.",
         [r"\mathsf I=\begin{pmatrix}I_1&0&0\\0&I_2&0\\0&0&I_3\end{pmatrix},\qquad \vec L=\mathsf I\vec\omega,\qquad T=\tfrac12\left(I_1\omega_1^2+I_2\omega_2^2+I_3\omega_3^2\right)",
          r"I_1+I_2\ge I_3\ \text{(y permutaciones)},\qquad I_{\hat n}=\hat n\cdot\mathsf I\,\hat n"]),
        ("Ecuaciones de Euler-Lagrange del cuerpo rígido",
         r"Con los ángulos de Euler $(\phi,\theta,\psi)$ como coordenadas generalizadas, el lagrangiano de un trompo simétrico libre ($I_1=I_2$) es una función de $\dot\theta$, $\dot\phi$, $\dot\psi$. Las coordenadas $\phi$ y $\psi$ son cíclicas, por lo que sus momentos conjugados (componentes de $\vec L$ y $\omega_3$) son constantes de movimiento por simetría de rotación. En el sistema del cuerpo, las ecuaciones equivalentes son las de Euler.",
         [r"L=\tfrac12I_1\left(\dot\theta^2+\dot\phi^2\sin^2\theta\right)+\tfrac12I_3\left(\dot\psi+\dot\phi\cos\theta\right)^2-V(\theta)",
          r"p_\psi=I_3\omega_3=\text{cte},\qquad p_\phi=I_1\dot\phi\sin^2\theta+I_3\omega_3\cos\theta=\text{cte}",
          r"I_1\dot\omega_1=(I_2-I_3)\omega_2\omega_3+\tau_1\ \text{y permutaciones cíclicas}"]),
        ("Momento de inercia por integración: barra y disco",
         r"Para una barra delgada de masa $M$ y longitud $L$ con eje perpendicular que pasa por el punto $s$ medido desde su centro, $dm=(M/L)\,dx$ y la integral se hace en forma cerrada. Para un disco de masa superficial $\sigma=M/\pi R^2$, la simetría axial sugiere anillos de radio $r$ y ancho $dr$, con $dm=\sigma\,2\pi r\,dr$. En ambos casos las integrales son de potencias de grado finito en un intervalo acotado, de modo que existen.",
         [r"I_{barra}(s)=\dfrac ML\int_{-L/2-s}^{L/2-s}x^2dx=M\left(\dfrac{L^2}{12}+s^2\right)",
          r"I_{disco}=\int_0^R r^2\sigma\,2\pi r\,dr=\dfrac{2\pi\sigma R^4}{4}=\tfrac12MR^2"]),
        ("Teoremas de ejes paralelos y perpendiculares",
         r"Si el eje se traslada por un vector $\vec a$ desde el centro de masa, el término lineal en las coordenadas relativas se anula por la definición del CM y queda el teorema de Steiner, que en forma tensorial añade $M(a^2\delta_{ij}-a_ia_j)$. El mínimo de $I$ entre ejes paralelos se alcanza siempre sobre el centro de masa. Para una lámina plana en el plano $xy$, $I_z=I_x+I_y$ porque $z=0$ en todo elemento.",
         [r"I_{ij}=I_{ij}^{cm}+M\left(a^2\delta_{ij}-a_ia_j\right)\ \Rightarrow\ I=I_{cm}+Md^2",
          r"\text{Lámina: }I_z=I_x+I_y\ \Rightarrow\ \text{disco: }I_x=I_y=\tfrac14MR^2"]),
        ("Resultado general",
         r"El momento de inercia es una integral de la densidad por el cuadrado de la distancia al eje; para cuerpos uniformes es $\kappa MR^2$ con $\kappa$ geométrico. Los resultados clave son los siguientes, todos en kg$\cdot$m$^2$ y con verificación dimensional inmediata (masa por longitud al cuadrado).",
         [r"\boxed{\;I=\int r_\perp^2\,dm,\qquad I_{ij}=\int\rho\left(r^2\delta_{ij}-x_ix_j\right)dV,\qquad I=I_{cm}+Md^2\;}",
          r"\boxed{\;I_{barra,cm}=\dfrac{ML^2}{12},\quad I_{barra,ext}=\dfrac{ML^2}{3},\quad I_{disco}=\dfrac{MR^2}{2}\;}"]),
        ("Particularización al problema",
         rf"Con $M={fx(m,0)}$ kg y $L={fx(L,1)}$ m para la barra, y $M={fx(md,0)}$ kg y $R={fx(rd,2)}$ m para el disco, se sustituye en los resultados generales. Se verifica el teorema de Steiner con $d=L/2$ y el de ejes perpendiculares en el disco.",
         [rf"I_{{cm}}=\dfrac{{({fx(m,0)})({fx(L,1)})^2}}{{12}}={fx(ic,4)}\ \mathrm{{kg\,m^2}},\qquad I_{{ext}}=\dfrac{{({fx(m,0)})({fx(L,1)})^2}}{{3}}={fx(ie,4)}\ \mathrm{{kg\,m^2}}",
          rf"I_{{cm}}+M\left(\dfrac L2\right)^2={fx(ic,4)}+({fx(m,0)})({fx(L/2,1)})^2={fx(ic+m*(L/2)**2,4)}\ \mathrm{{kg\,m^2}}",
          rf"I_{{disco}}=\tfrac12({fx(md,0)})({fx(rd,2)})^2={fx(idisc,5)}\ \mathrm{{kg\,m^2}},\qquad I_x+I_y=2({fx(ix,5)})={fx(2*ix,5)}\ \mathrm{{kg\,m^2}}"]),
        ("Errores comunes y límites de validez",
         r"Se suele olvidar que $I$ depende del eje y usar $ML^2/12$ para un extremo, o aplicar el teorema de Steiner entre dos ejes que no pasan por el centro de masa (solo vale desde el CM). También se confunde $\vec L=\mathsf I\vec\omega$ con $L=I\omega$ escalar, que solo es correcto para rotación en torno a un eje principal. El tratamiento supone cuerpo rígido; la barra delgada ignora su grosor (corrección de orden $a^2/L^2$), y a velocidades relativistas o con deformación elástica la inercia efectiva cambia.",
         []),
    ])


# ============================================================== HangingMassDisc
def _hanging() -> dict:
    m, M, R, g = c12.M_HANG, c12.M_DISC, c12.R_DISC, c12.G
    an = c12.hanging_analytic()
    I = 0.5 * M * R**2
    meff = m + I / R**2
    v1 = np.sqrt(2 * an["a"] * 1.0)
    return dict(concept=[
        ("Qué se estudia y por qué importa",
         "Una masa que cuelga de una cuerda enrollada en una polea es el modelo básico de grúas, ascensores, cabrestantes y de la máquina de Atwood, y sirve en el laboratorio para medir momentos de inercia. A diferencia del caso con polea sin masa, aquí la polea almacena energía cinética de rotación y la tensión no es la misma a ambos lados. La pregunta es cuánto se acelera la masa y qué tensión soporta la cuerda cuando el disco también debe acelerarse.",
         []),
        ("Magnitudes y convenciones",
         r"Se denotan $m$ la masa colgante (kg), $M$ y $R$ la masa (kg) y el radio (m) del disco, $I$ su momento de inercia (kg$\cdot$m$^2$) y $y$ el descenso medido hacia abajo (m). Aparecen la aceleración $a$ (m/s$^2$), la tensión $T$ (N) y la aceleración angular $\alpha$ (rad/s$^2$). Se supone cuerda inextensible y sin masa que no desliza sobre el disco, y eje sin fricción.",
         []),
        ("Intuición y analogías",
         "El peso de la masa tiene que acelerar dos cosas a la vez: la propia masa y el disco, que necesita un torque para empezar a girar. La inercia del disco se comporta como una masa extra $I/R^2$ que se suma a la de la carga. Por eso la aceleración es menor que $g$ y la tensión es menor que $mg$ pero mayor que cero: la cuerda debe entregar el torque al disco.",
         []),
        ("Ligadura y coordenada generalizada",
         r"La cuerda inextensible y sin deslizamiento liga la altura de la masa con el ángulo del disco: $\dot y=\omega R$. Por eso hay un solo grado de libertad, que se toma como $y$. Esta ligadura es holónoma; la tensión es una fuerza de ligadura que no aparece en el lagrangiano y se recuperará después a partir de la segunda ley aplicada a la masa.",
         [r"y=R\theta,\qquad \dot y=R\omega,\qquad \ddot y=a=R\alpha"]),
        ("Lagrangiano y ecuación de Euler-Lagrange",
         r"La energía cinética es la de traslación de la masa más la de rotación del disco, y la potencial es gravitatoria. Sustituyendo $\omega=\dot y/R$, la inercia del disco aparece como una masa efectiva. Como $L$ no depende del tiempo, la energía se conserva, y su derivada temporal reproduce la ecuación de movimiento.",
         [r"L=\tfrac12m\dot y^2+\tfrac12I\omega^2+mgy=\tfrac12\left(m+\dfrac{I}{R^2}\right)\dot y^2+mgy",
          r"\dfrac{d}{dt}\dfrac{\partial L}{\partial\dot y}-\dfrac{\partial L}{\partial y}=0\;\Rightarrow\;\left(m+\dfrac{I}{R^2}\right)\ddot y=mg"]),
        ("Formulación hamiltoniana y energía",
         r"El momento conjugado es $p=m_{ef}\dot y$ con $m_{ef}=m+I/R^2$, y el hamiltoniano es $H=p^2/(2m_{ef})-mgy$. Como $\partial H/\partial t=0$, es constante, e igual a cero si el sistema parte del reposo en $y=0$. De ello sale la velocidad después de descender una altura $h$ sin integrar el movimiento, y ofrece una verificación independiente de la aceleración.",
         [r"\dot p=-\dfrac{\partial H}{\partial y}=mg,\qquad \dot y=\dfrac{\partial H}{\partial p}=\dfrac{p}{m_{ef}}",
          r"mgh=\tfrac12m_{ef}v^2\;\Rightarrow\;v=\sqrt{2ah},\qquad a=\dfrac{mg}{m_{ef}}"]),
        ("Tensión y aceleración angular (ligadura)",
         r"La tensión se obtiene de la segunda ley de Newton aplicada a la masa, $mg-T=ma$, y debe coincidir con el torque sobre el disco, $TR=I\alpha$; la igualdad de ambas ecuaciones con $a=R\alpha$ es una comprobación de consistencia que cierra el sistema. La tensión es siempre menor que $mg$ y positiva.",
         [r"T=m(g-a)=\dfrac{mg\,(I/R^2)}{m+I/R^2},\qquad \alpha=\dfrac aR",
          r"\sum\tau=TR=I\alpha\ \Rightarrow\ T=\dfrac{I a}{R^2}"]),
        ("Parámetros adimensionales, casos límite y generalización",
         r"Con $I=\kappa MR^2$ el único parámetro es $\beta=\kappa M/m$ y $a=g/(1+\beta)$. Si $\beta\to0$ (disco ligero) se obtiene caída libre y $T\to0$; si $\beta\to\infty$ el disco es tan pesado que $a\to0$ y $T\to mg$. Con dos masas $m_1<m_2$ a ambos lados de la polea (Atwood con polea masiva) el mismo método da $a$, con tensiones distintas a cada lado, cuya diferencia acelera el disco.",
         [r"a=\dfrac{g}{1+\beta},\quad T=mg\dfrac{\beta}{1+\beta},\qquad \beta=\dfrac{I}{mR^2}",
          r"a_{Atwood}=\dfrac{(m_2-m_1)\,g}{m_1+m_2+I/R^2},\qquad T_1=m_1(g+a),\ T_2=m_2(g-a)"]),
        ("Resultado general",
         r"La aceleración es la fuerza motriz $mg$ dividida por la inercia total del sistema, medida en unidades de masa. La tensión y la aceleración angular son consecuencias algebraicas. Unidades: m/s$^2$, N y rad/s$^2$; $I/R^2$ tiene unidades de masa, así que el denominador es consistente.",
         [r"\boxed{\;a=\dfrac{mg}{m+I/R^2},\qquad T=\dfrac{mg\,(I/R^2)}{m+I/R^2},\qquad \alpha=\dfrac aR\;}",
          r"\boxed{\;I=\tfrac12MR^2\ \Rightarrow\ a=\dfrac{mg}{m+M/2}\;}"]),
        ("Particularización al problema",
         rf"Con $m={fx(m,0)}$ kg, $M={fx(M,0)}$ kg y $R={fx(R,2)}$ m, el disco uniforme tiene $I={fx(I,4)}$ kg$\cdot$m$^2$ y la masa efectiva es ${fx(meff,0)}$ kg. Se sustituye en las fórmulas generales y se recupera la solución de la escena.",
         [rf"a=\dfrac{{({fx(m,0)})({fx(g,2)})}}{{{fx(m,0)}+{fx(M,0)}/2}}={fx(an['a'],3)}\ \mathrm{{m/s^2}},\qquad T=({fx(m,0)})({fx(g,2)}-{fx(an['a'],3)})={fx(an['T'],3)}\ \mathrm{{N}}",
          rf"\alpha=\dfrac{{a}}{{R}}={fx(an['alpha'],2)}\ \mathrm{{rad/s^2}},\qquad TR={fx(an['T']*R,4)}=I\alpha={fx(I*an['alpha'],4)}\ \mathrm{{N\,m}}",
          rf"v(h=1\ \mathrm{{m}})=\sqrt{{2ah}}={fx(v1,3)}\ \mathrm{{m/s}},\qquad \beta=\dfrac{{I}}{{mR^2}}={fx(I/(m*R**2),1)}\ \Rightarrow\ a=\dfrac{{g}}{{1+\beta}}={fx(g/(1+I/(m*R**2)),3)}\ \mathrm{{m/s^2}}"]),
        ("Errores comunes y límites de validez",
         r"El error típico es tomar $T=mg$, lo que ignora que la masa acelera, o escribir $a=g$ ignorando la inercia del disco. También se olvida que, con dos masas, las tensiones a cada lado de una polea con inercia son diferentes. El modelo supone cuerda sin masa ni deslizamiento y eje sin fricción; con rozamiento aparece un torque resistivo constante, y si la cuerda tiene masa o se desenrolla cambia el radio efectivo y la inercia es función del tiempo.",
         []),
    ])


DETAIL = {
    "CenterOfMass": _cm(),
    "ExplodingProjectile": _explode(),
    "Collision1D": _collision(),
    "BallisticPendulum": _pendulum(),
    "SpinUp": _spinup(),
    "RollingCycloid": _cycloid(),
    "MomentOfInertia": _inertia(),
    "HangingMassDisc": _hanging(),
}
