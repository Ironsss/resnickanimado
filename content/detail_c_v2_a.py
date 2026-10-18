"""Desarrollo conceptual y analitico GENERAL, Volumen 2, capitulos 1 a 4 (Coulomb, campo, Gauss, potencial).

DETAIL = {id: dict(concept=[(titulo, prosa, [ecuaciones TeX]), ...])}. Todos los numeros de las secciones de
particularizacion se calculan con `physics/` y las constantes del proyecto.
"""
from __future__ import annotations

import numpy as np

from content.texfmt import fx, sci
from physics import v2ch01_coulomb as e1
from physics import v2ch02_field as e2
from physics import v2ch03_gauss as e3
from physics import v2ch04_potential as e4
from physics.constants import EPS0, K_E


def _mn(v) -> str:
    return "(" + ",\\ ".join(fx(float(c) * 1000, 3) for c in v) + ")"


# =========================================================================== CoulombForces
def _coulomb() -> dict:
    q, pos = e1.Q, e1.POS
    r13, r23 = pos[2] - pos[0], pos[2] - pos[1]
    d13, d23 = float(np.linalg.norm(r13)), float(np.linalg.norm(r23))
    f31 = K_E * q[2] * q[0] * r13 / d13**3
    f32 = K_E * q[2] * q[1] * r23 / d23**3
    fa = e1.force_analytic()
    f_all = [e1.force_analytic(i) for i in range(3)]
    tot = sum(f_all)
    u = e1.potential_energy(pos)
    return dict(concept=[
        ("Qué se estudia y por qué importa",
         r"La interacción entre cargas en reposo es la base de la química (enlaces, fuerzas intermoleculares), de la materia condensada y de dispositivos como filtros electrostáticos, impresoras láser y cámaras de deriva. A diferencia de la gravedad, la fuerza eléctrica puede ser atractiva o repulsiva, y es unos $10^{36}$ veces más intensa entre dos protones que su atracción gravitatoria. Por eso la materia ordinaria es casi neutra: los excesos de carga se compensan a escalas muy pequeñas. Aquí se parte de la ley de Coulomb para $N$ cargas puntuales en el espacio y se obtiene la fuerza sobre cada una, antes de aplicarla al caso de tres cargas en el plano.",
         []),
        ("Magnitudes y convenciones",
         r"La carga $q$ se mide en coulomb (C); la constante de Coulomb $k=1/(4\pi\varepsilon_0)\approx 8.99\times10^{9}\ \mathrm{N\,m^2/C^2}$ y la permitividad del vacío es $\varepsilon_0\approx 8.85\times10^{-12}\ \mathrm{C^2/(N\,m^2)}$. La fuerza $\vec F_{ij}$ es la que la carga $j$ ejerce sobre la carga $i$, y el vector $\vec r_{ij}=\vec r_i-\vec r_j$ apunta de $j$ hacia $i$: así un producto $q_iq_j>0$ da una fuerza a lo largo de $+\vec r_{ij}$ (repulsión). Los órdenes de magnitud típicos en el laboratorio son microcoulombs y fuerzas de milinewtons a newtons; un coulomb es una carga enorme.",
         [r"[k]=\mathrm{N\,m^2\,C^{-2}},\qquad [\varepsilon_0]=\mathrm{C^2\,N^{-1}\,m^{-2}},\qquad \vec r_{ij}=\vec r_i-\vec r_j"]),
        ("Intuición y analogías",
         r"La ley de Coulomb tiene la misma forma que la ley de gravitación de Newton, con la masa sustituida por la carga y con signo: cargas iguales se repelen y distintas se atraen. La forma $1/r^2$ equivale a decir que la influencia de una carga se reparte sobre la superficie de una esfera de área $4\pi r^2$, de modo que a doble distancia llega la cuarta parte. Lo más importante conceptualmente es la superposición: las fuerzas sobre una carga se suman vectorialmente, sin que la presencia de una tercera carga altere la fuerza entre otras dos.",
         []),
        ("Principios y planteamiento general",
         r"Se consideran $N$ cargas puntuales $q_1,\dots,q_N$ en reposo en posiciones $\vec r_1,\dots,\vec r_N$ del vacío. Las hipótesis son: (i) las cargas están fijas (electrostática, sin radiación ni campo magnético), (ii) la interacción es de a pares y central, y (iii) vale el principio de superposición, que expresa la linealidad de las ecuaciones de Maxwell en las fuentes. La fuerza de la carga $j$ sobre la $i$ es la ley de Coulomb vectorial; es central porque va a lo largo de la recta que une las cargas y depende solo de la distancia.",
         [r"\vec F_{ij}=k\,\dfrac{q_iq_j}{|\vec r_{ij}|^{2}}\,\hat r_{ij}=k\,q_iq_j\,\dfrac{\vec r_i-\vec r_j}{|\vec r_i-\vec r_j|^{3}},\qquad \hat r_{ij}=\dfrac{\vec r_{ij}}{|\vec r_{ij}|}",
          r"\vec F_{ji}=-\vec F_{ij}\qquad(\text{al cambiar }i\leftrightarrow j\text{ el vector }\vec r_{ij}\text{ cambia de signo})"]),
        ("Superposición y fuerza neta sobre una carga",
         r"Como la fuerza es lineal en el campo creado por las demás cargas, la fuerza neta sobre la carga $i$ es la suma de las contribuciones individuales; de ahí se define el campo eléctrico $\vec E_i$ que producen las demás en la posición $\vec r_i$ y se escribe $\vec F_i=q_i\vec E_i$. Separar $q_i$ del resto tiene sentido porque $\vec E_i$ no depende de $q_i$ (una carga no actúa sobre sí misma, de ahí la exclusión $j\neq i$). Con ello el problema de $N$ cuerpos queda reducido a evaluar una suma de $N-1$ términos vectoriales para cada carga.",
         [r"\vec F_i=\sum_{j\neq i}\vec F_{ij}=k\,q_i\sum_{j\neq i}q_j\,\dfrac{\vec r_i-\vec r_j}{|\vec r_i-\vec r_j|^{3}}=q_i\,\vec E_i",
          r"\vec E_i=k\sum_{j\neq i}q_j\,\dfrac{\vec r_i-\vec r_j}{|\vec r_i-\vec r_j|^{3}}"]),
        ("Energía potencial, carácter conservativo y simetrías",
         r"El campo $\vec r/r^3$ es gradiente de $-1/r$ y por tanto tiene rotacional nulo ($\nabla\times\vec E=0$, ecuación de Maxwell estática); en consecuencia el trabajo no depende del camino y existe una energía potencial. La energía total del sistema es la suma sobre pares distintos, y cada fuerza se obtiene como menos el gradiente respecto de la posición de su propia carga. La invariancia de $U$ ante una traslación común de todas las cargas (solo depende de diferencias $\vec r_i-\vec r_j$) implica, por el teorema de Noether, la conservación del momento: $\sum_i\vec F_i=0$, es decir, la tercera ley de Newton; la invariancia ante rotaciones implica que el momento de las fuerzas internas también suma cero.",
         [r"U=k\sum_{i<j}\dfrac{q_iq_j}{|\vec r_i-\vec r_j|},\qquad \vec F_i=-\nabla_{\vec r_i}U",
          r"\sum_i\vec F_i=-\sum_i\nabla_{\vec r_i}U=-\dfrac{d}{d\vec a}\,U(\vec r_1+\vec a,\dots,\vec r_N+\vec a)\Big|_{\vec a=0}=0",
          r"\sum_i\vec r_i\times\vec F_i=0\qquad(\text{invariancia de }U\text{ ante rotaciones})"]),
        ("Casos límite, parámetros adimensionales y equilibrio",
         r"Para dos cargas la fuerza se reduce a $kq_1q_2/r^2$ sobre la recta que las une, y a distancias grandes comparadas con la separación de un grupo neutro la fuerza decae más rápido que $1/r^2$ porque las contribuciones se compensan. Midiendo las longitudes en una escala $L$ (la separación típica) y las cargas en $q_0$, toda la fuerza se escribe como $kq_0^2/L^2$ por un vector adimensional que solo depende de la geometría y de los cocientes $q_i/q_0$. Además, como $\nabla\cdot\vec E=0$ fuera de las cargas, ninguna configuración de cargas fijas puede ser un equilibrio estable en las tres direcciones a la vez (teorema de Earnshaw).",
         [r"\vec F_i=\dfrac{k\,q_0^{2}}{L^{2}}\,\vec f_i\!\left(\dfrac{\vec r_j}{L},\dfrac{q_j}{q_0}\right),\qquad \vec f_i\ \text{adimensional}",
          r"\vec F_{12}\xrightarrow{\ r\to\infty\ }\dfrac{kq_1q_2}{r^{2}}\hat r\qquad\text{(par neutro: }F\sim r^{-3}\text{ o menor)}"]),
        ("Resultado general",
         r"La fuerza sobre la carga $i$ de un sistema de $N$ cargas puntuales en reposo es la suma vectorial de las fuerzas coulombianas de las demás; equivalente a $\vec F_i=-\nabla_i U$ y sujeta a $\sum_i\vec F_i=0$. Depende de cada carga linealmente, de cada distancia como $r^{-2}$, y su unidad es el newton: $[k][q]^2/[r]^2=\mathrm{N\,m^2C^{-2}\cdot C^2\cdot m^{-2}}=\mathrm{N}$, lo que verifica la dimensión.",
         [r"\boxed{\ \vec F_i=k\,q_i\sum_{j\neq i}q_j\,\dfrac{\vec r_i-\vec r_j}{|\vec r_i-\vec r_j|^{3}}\ },\qquad \sum_{i=1}^{N}\vec F_i=\vec 0"]),
        ("Particularización al problema",
         rf"Con $q_1=+3\ \mu\mathrm{{C}}$ en el origen, $q_2=-2\ \mu\mathrm{{C}}$ en $(0.4,0)$ m y $q_3=+1\ \mu\mathrm{{C}}$ en $(0.2,0.3)$ m, la fórmula general para $i=3$ da dos términos: $\vec r_{{31}}=(0.2,0.3)$ m con $|\vec r_{{31}}|={fx(d13,4)}$ m y $\vec r_{{32}}=(-0.2,0.3)$ m con $|\vec r_{{32}}|={fx(d23,4)}$ m. La primera fuerza es repulsiva y la segunda atractiva. La suma coincide con el resultado de la escena y las tres fuerzas netas suman cero.",
         [rf"\vec F_{{31}}=k\,q_3q_1\dfrac{{\vec r_{{31}}}}{{|\vec r_{{31}}|^3}}={_mn(f31)}\ \mathrm{{mN}},\qquad \vec F_{{32}}=k\,q_3q_2\dfrac{{\vec r_{{32}}}}{{|\vec r_{{32}}|^3}}={_mn(f32)}\ \mathrm{{mN}}",
          rf"\vec F_3=\vec F_{{31}}+\vec F_{{32}}={_mn(fa)}\ \mathrm{{mN}},\qquad |\vec F_3|={fx(float(np.linalg.norm(fa)) * 1000, 3)}\ \mathrm{{mN}}",
          rf"\sum_i\vec F_i={_mn(tot)}\ \mathrm{{mN}},\qquad U={fx(u * 1000, 4)}\ \mathrm{{mJ}}"]),
        ("Errores comunes y límites de validez",
         r"Un error frecuente es usar $|\vec r|$ en el numerador y olvidar que el vector unitario ya contiene una división por la distancia: la forma correcta tiene $r^3$ en el denominador cuando se escribe con $\vec r_i-\vec r_j$. Otro es el signo: la fuerza sobre $i$ debe medirse con $\vec r_i-\vec r_j$ (no al revés) y los signos de las cargas entran en el producto $q_iq_j$. También se confunde la fuerza sobre $q_3$ con la suma de todas las fuerzas del sistema, que es cero por la tercera ley. El modelo falla si las cargas se mueven con velocidades relativistas o aceleran (aparecen campos magnéticos y retardo, ecuaciones de Liénard-Wiechert), a distancias subatómicas (la carga deja de ser puntual y entra la mecánica cuántica) y en medios materiales, donde $\varepsilon_0$ se sustituye por $\varepsilon=\kappa\varepsilon_0$ y hay cargas ligadas.",
         []),
    ])


# =========================================================================== EquilibriumPoint
def _equilibrium() -> dict:
    q1, q2, d = e1.Q1, e1.Q2, e1.SEP
    rho = float(np.sqrt(abs(q1 / q2)))
    xa = e1.zero_force_point_analytic()
    xo = e1.opposite_charges_point_analytic()
    qt = 1e-9
    k_stiff = -2 * K_E * qt * (q1 / xa**3 + q2 / (d - xa) ** 3)
    return dict(concept=[
        ("Qué se estudia y por qué importa",
         r"Un punto de equilibrio es aquel donde la fuerza eléctrica neta sobre una carga de prueba se anula. Su estudio es la versión más simple de problemas de gran interés práctico: trampas de iones, puntos nulos de un campo, focalización de haces y análisis de estabilidad de configuraciones de cargas. Que exista o no un punto de equilibrio y dónde, depende solo de los cocientes entre cargas y de la geometría. Aquí se resuelve el caso general de dos cargas fijas de valores y signos arbitrarios sobre una recta.",
         []),
        ("Magnitudes y convenciones",
         r"Se coloca $q_1$ en el origen y $q_2$ en $x=d$ sobre el eje $x$; la carga de prueba $q_t$ está en la posición $x$ y se mide en coulomb. Se trabaja con cargas $q_1,q_2$ con signo y se define el cociente adimensional $\rho=\sqrt{|q_1/q_2|}$. La carga de prueba debe ser pequeña para no perturbar la distribución; su valor no influye en la posición de equilibrio, solo en la magnitud de la fuerza, que es del orden de $kq_tq/d^2$.",
         [r"\rho=\sqrt{\left|\dfrac{q_1}{q_2}\right|},\qquad [x]=\mathrm{m},\qquad F\sim\dfrac{kq_tq}{d^2}"]),
        ("Intuición y analogías",
         r"Es el análogo del punto de Lagrange $L_1$ entre dos astros: allí las atracciones de los dos cuerpos se compensan. Con cargas del mismo signo ambas repelen a la prueba y el equilibrio ocurre entre ellas, más cerca de la carga menor porque su campo es más débil; con signos opuestos las fuerzas entre las cargas tienen el mismo sentido y no pueden compensarse, de modo que el equilibrio solo es posible fuera del segmento, en el lado de la carga menor, donde una fuerza atrae y la otra repele.",
         []),
        ("Planteamiento general y simetrías",
         r"La fuerza sobre la prueba es la superposición de dos fuerzas de Coulomb que actúan a lo largo del eje, ya que por simetría de revolución alrededor de la recta que une las cargas no hay componente transversal en puntos del eje. Se escribe con las funciones signo para cubrir los tres tramos del eje con una sola expresión. La distancia a cada carga es $|x|$ y $|x-d|$, y el sentido de cada fuerza lo da el producto de la carga por el signo de la posición relativa.",
         [r"F(x)=k\,q_t\left[\,q_1\,\dfrac{\operatorname{sgn}(x)}{x^{2}}+q_2\,\dfrac{\operatorname{sgn}(x-d)}{(x-d)^{2}}\right],\qquad x\neq0,\ d"]),
        ("Condición de equilibrio y raíces",
         r"Igualar $F(x)=0$ y cancelar $kq_t$ (por eso el equilibrio no depende de la carga de prueba) obliga a que los dos términos tengan igual módulo y sentido opuesto. La igualdad de módulos da $|q_1|/x^2=|q_2|/(x-d)^2$, es decir $x/(x-d)=\pm\rho$; al elevar al cuadrado se pierde el signo y por eso hay que decidir después en qué tramo es válida cada raíz. Las dos raíces son lineales en $x$.",
         [r"\dfrac{|q_1|}{x^{2}}=\dfrac{|q_2|}{(x-d)^{2}}\ \Longrightarrow\ \dfrac{x}{x-d}=\pm\rho",
          r"x_{-}=\dfrac{\rho\,d}{\rho+1},\qquad x_{+}=\dfrac{\rho\,d}{\rho-1}\quad(\rho\neq1)"]),
        ("Cuál raíz es física: criterio del signo",
         r"Entre las cargas ($0<x<d$) los dos términos de $F$ tienen la forma $q_1/x^2-q_2/(d-x)^2$: se oponen solo si $q_1$ y $q_2$ tienen el mismo signo, de modo que la raíz interior $x_-$ vale solo para cargas del mismo signo (pues $x_-\in(0,d)$ siempre). Fuera del segmento ambos términos tienen signo relativo $q_1q_2$ cambiado: se oponen si $q_1q_2<0$, y la raíz $x_+$ lo describe, con $x_+>d$ si $\rho>1$ (la carga mayor es $q_1$) o $x_+<0$ si $\rho<1$. Para $\rho=1$ con signos opuestos no hay equilibrio a distancia finita (el punto está en el infinito), y con $q_1=q_2$ iguales el equilibrio es el punto medio.",
         [r"q_1q_2>0:\ x^{*}=\dfrac{\rho\,d}{\rho+1}\in(0,d);\qquad q_1q_2<0:\ x^{*}=\dfrac{\rho\,d}{\rho-1}\notin[0,d]",
          r"x^{*}=\dfrac{\sqrt{|q_1|}\,d}{\sqrt{|q_1|}+\sqrt{|q_2|}}\ \ (\text{mismo signo}),\qquad \dfrac{x^{*}}{d-x^{*}}=\sqrt{\left|\dfrac{q_1}{q_2}\right|}"]),
        ("Estabilidad y teorema de Earnshaw",
         r"Que haya fuerza nula no garantiza un equilibrio estable. Para una prueba del mismo signo que las dos cargas, la derivada de $F$ en el punto interior es negativa: es una fuerza restauradora axial. Pero como $\nabla\cdot\vec E=0$ en el vacío, la suma de las tres derivadas $\partial E_i/\partial x_i$ es nula; por simetría axial, $\partial E_y/\partial y=\partial E_z/\partial z=-\tfrac12\,\partial E_x/\partial x$, de modo que si el equilibrio es estable a lo largo del eje, es inestable en las direcciones transversales (punto silla). Con un potencial armónico $U\approx\tfrac12 \kappa x^2$ se obtiene la rigidez axial $\kappa=-F'(x^{*})$.",
         [r"\kappa_x=-\dfrac{dF}{dx}\Big|_{x^{*}}=2kq_t\left[\dfrac{q_1}{x^{*3}}+\dfrac{q_2}{(d-x^{*})^{3}}\right],\qquad \kappa_y=\kappa_z=-\dfrac{\kappa_x}{2}",
          r"\kappa_x+\kappa_y+\kappa_z=0\quad\Longleftrightarrow\quad \nabla\cdot\vec E=0"]),
        ("Resultado general",
         r"El punto de equilibrio de una carga de prueba entre dos cargas fijas depende solo del cociente $\rho=\sqrt{|q_1/q_2|}$ y de la separación $d$, es independiente de $q_t$, y es lineal en $d$. Es adimensional en $x^*/d$, de modo que se verifica dimensionalmente: $[x^*]=[d]=\mathrm{m}$. Cumple $x^*\to d/2$ cuando $\rho\to1$ (cargas del mismo signo) y se acerca a la carga menor cuando $\rho$ crece.",
         [r"\boxed{\ x^{*}=\dfrac{\rho}{\rho\pm1}\,d,\qquad \rho=\sqrt{\left|\dfrac{q_1}{q_2}\right|},\quad \text{signo }+\text{ (mismo signo de las cargas), signo }-\text{ (signos opuestos)}\ }"]),
        ("Particularización al problema",
         rf"Con $q_1=+4\ \mu\mathrm{{C}}$, $q_2=+1\ \mu\mathrm{{C}}$ y $d=0.60$ m se obtiene $\rho=\sqrt{{4/1}}={fx(rho, 3)}$ y la raíz interior. Para $q_2=-1\ \mu\mathrm{{C}}$ el cociente es el mismo, pero ahora la raíz válida es la exterior, que coincide con $2d$. Ambos valores son los de la escena. Para una prueba positiva en el caso del mismo signo, la rigidez axial es positiva (equilibrio estable a lo largo del eje) con una carga de prueba de $1$ nC.",
         [rf"x^{{*}}=\dfrac{{\rho d}}{{\rho+1}}=\dfrac{{({fx(rho, 3)})({fx(d, 2)})}}{{{fx(rho, 3)}+1}}={fx(xa, 4)}\ \mathrm{{m}}\qquad(q_1q_2>0)",
          rf"x^{{*}}=\dfrac{{\rho d}}{{\rho-1}}=\dfrac{{({fx(rho, 3)})({fx(d, 2)})}}{{{fx(rho, 3)}-1}}={fx(xo, 4)}\ \mathrm{{m}}=2d\qquad(q_1q_2<0)",
          rf"\kappa_x=2kq_t\left[\dfrac{{q_1}}{{x^{{*3}}}}+\dfrac{{q_2}}{{(d-x^{{*}})^3}}\right]={sci(k_stiff, 4)}\ \mathrm{{N/m}}"]),
        ("Errores comunes y límites de validez",
         r"Es habitual buscar el equilibrio entre las cargas cuando son de signo opuesto, o suponer que está a mitad de camino aunque las cargas difieran; la distancia correcta se reparte en la razón de las raíces de las cargas, no de las cargas. Otra confusión es pensar que la carga de prueba debe ser positiva: el punto de fuerza nula no depende del signo de $q_t$, solo la estabilidad cambia. Fuera del eje la fuerza no se anula salvo en el propio eje. El modelo supone cargas puntuales fijas en el vacío; si las cargas pueden moverse (por ejemplo, conductores), la carga de prueba induce redistribución y el equilibrio cambia.",
         []),
    ])


# =========================================================================== DipoleField
def _dipole() -> dict:
    q, d = e2.QD, e2.DSEP
    p = q * d
    z = 0.5
    ex, ef = float(e2.dipole_field_axis_exact(z)), float(e2.dipole_field_axis_far(z))
    eps = d / z
    corr = 1 + eps**2 / 2
    return dict(concept=[
        ("Qué se estudia y por qué importa",
         r"Un dipolo eléctrico son dos cargas iguales y opuestas muy próximas: es la configuración de carga neutra más simple y también la primera aproximación a cualquier distribución neutra. Describe moléculas polares como el agua, antenas elementales, átomos polarizados y el comportamiento de los dieléctricos, y explica por qué las fuerzas entre moléculas neutras decaen con la distancia. Se obtiene aquí el potencial y el campo en un punto cualquiera, las líneas de campo y la aproximación de campo lejano que usa la escena.",
         []),
        ("Magnitudes y convenciones",
         r"El dipolo tiene cargas $+q$ y $-q$ separadas una distancia $d$; su momento dipolar $\vec p=q\vec d$ apunta de la carga negativa a la positiva y se mide en $\mathrm{C\,m}$ (una molécula de agua tiene $6.2\times10^{-30}\ \mathrm{C\,m}$). Se usan coordenadas esféricas $(r,\theta)$ con el eje $z$ a lo largo de $\vec p$ y el origen en el centro. El campo se mide en N/C y el potencial en voltios.",
         [r"\vec p=q\,\vec d,\qquad [p]=\mathrm{C\,m},\qquad [E]=\mathrm{N/C}=\mathrm{V/m}"]),
        ("Intuición y analogías",
         r"Cada carga produce un campo de tipo $1/r^2$, pero a gran distancia los campos de las dos casi se cancelan: la diferencia entre ellos es una derivada, lo que da una potencia más, $1/r^3$. Es análogo al par de imanes en un imán de barra, con líneas que salen del polo positivo, se curvan y entran por el negativo. A una distancia grande el dipolo solo ve su orientación y su momento, no los detalles de cómo están distribuidas las cargas.",
         []),
        ("Potencial exacto y simetrías",
         r"Se sitúan $+q$ en $z=+d/2$ y $-q$ en $z=-d/2$. El potencial es la superposición de dos potenciales coulombianos, y es antisimétrico bajo la reflexión $z\to-z$; además tiene simetría axial, de modo que solo depende de $(r,\theta)$. Las distancias a cada carga se obtienen con la ley de los cosenos. Con el potencial escalar el cálculo es más simple que con el campo vectorial, ya que $\vec E=-\nabla V$ siempre se puede recuperar después.",
         [r"V(r,\theta)=kq\left[\dfrac{1}{r_{+}}-\dfrac{1}{r_{-}}\right],\qquad r_{\pm}=\sqrt{r^{2}\mp r\,d\cos\theta+\dfrac{d^{2}}{4}}"]),
        ("Desarrollo multipolar",
         r"Con $t=d/(2r)<1$ la función generatriz de los polinomios de Legendre permite expandir cada inverso de distancia en potencias de $t$. Al restar, los términos pares se cancelan (por la antisimetría) y la carga total nula elimina el monopolo; sobrevive el dipolo $l=1$, seguido del octupolo $l=3$. La serie converge para $r>d/2$, y cada término es menor que el anterior en un factor $(d/2r)^2$, por eso a gran distancia basta el primer término.",
         [r"\dfrac{1}{r_{\pm}}=\dfrac{1}{r}\sum_{l=0}^{\infty}\left(\dfrac{d}{2r}\right)^{l}P_l(\pm\cos\theta)=\dfrac1r\sum_{l=0}^{\infty}(\pm1)^{l}\left(\dfrac{d}{2r}\right)^{l}P_l(\cos\theta)",
          r"V=\dfrac{2kq}{r}\sum_{l\ \mathrm{impar}}\left(\dfrac{d}{2r}\right)^{l}P_l(\cos\theta)=\dfrac{kp\cos\theta}{r^{2}}+\dfrac{kp\,d^{2}}{4}\,\dfrac{P_3(\cos\theta)}{r^{4}}+\cdots"]),
        ("Campo del dipolo y líneas de campo",
         r"El campo dipolar sale de $\vec E=-\nabla V$ en coordenadas esféricas, sin componente azimutal por la simetría. Su módulo cae como $r^{-3}$ y depende del ángulo; en el eje es doble que en el plano ecuatorial y de sentido opuesto. Las líneas de campo son curvas tangentes a $\vec E$, es decir, cumplen $dr/(r\,d\theta)=E_r/E_\theta$; la integración da una familia de curvas con parámetro $C$. Las equipotenciales cumplen $\cos\theta/r^2=\text{cte}$, y son ortogonales a las líneas de campo.",
         [r"E_r=-\dfrac{\partial V}{\partial r}=\dfrac{2kp\cos\theta}{r^{3}},\qquad E_\theta=-\dfrac1r\dfrac{\partial V}{\partial\theta}=\dfrac{kp\sin\theta}{r^{3}},\qquad |\vec E|=\dfrac{kp}{r^{3}}\sqrt{1+3\cos^{2}\theta}",
          r"\vec E=k\,\dfrac{3(\vec p\cdot\hat r)\hat r-\vec p}{r^{3}},\qquad \dfrac{dr}{r\,d\theta}=2\cot\theta\ \Longrightarrow\ r=C\sin^{2}\theta"]),
        ("Eje del dipolo: fórmula exacta y corrección",
         r"Sobre el eje la suma exacta de las dos cargas se simplifica con una diferencia de cuadrados. Al introducir el parámetro adimensional $\varepsilon=d/z$ y desarrollar en serie, el primer término es el de campo lejano y el siguiente es una corrección relativa de orden $\varepsilon^2/2$; el error de la aproximación dipolar es por tanto cuadrático en $d/z$ y siempre por defecto. El límite puntual $d\to0$ con $p$ fijo recupera exactamente $2kp/z^3$.",
         [r"E_z=kq\left[\dfrac{1}{(z-d/2)^{2}}-\dfrac{1}{(z+d/2)^{2}}\right]=\dfrac{2kp\,z}{\left(z^{2}-d^{2}/4\right)^{2}}",
          r"E_z=\dfrac{2kp}{z^{3}}\left(1-\dfrac{\varepsilon^{2}}{4}\right)^{-2}=\dfrac{2kp}{z^{3}}\left[1+\dfrac{\varepsilon^{2}}{2}+O(\varepsilon^{4})\right],\qquad \varepsilon=\dfrac dz"]),
        ("Interacción con un campo externo",
         r"Si el dipolo se sitúa en un campo externo $\vec E_0$ aproximadamente uniforme, las fuerzas sobre las dos cargas son opuestas y producen un momento de torsión que lo alinea con el campo. Su energía potencial es mínima cuando $\vec p\parallel\vec E_0$. Para ángulos pequeños, el dipolo oscila como un péndulo con momento de inercia $I$, con frecuencia $\omega=\sqrt{pE_0/I}$; si el campo no es uniforme, aparece además una fuerza neta $(\vec p\cdot\nabla)\vec E$, que explica por qué un objeto neutro polarizable es atraído hacia donde el campo es más intenso.",
         [r"\vec\tau=\vec p\times\vec E_0,\qquad U=-\vec p\cdot\vec E_0,\qquad \vec F=(\vec p\cdot\nabla)\vec E_0",
          r"I\ddot\theta=-pE_0\sin\theta\ \Longrightarrow\ \omega=\sqrt{\dfrac{pE_0}{I}}\quad(\theta\ll1)"]),
        ("Resultado general",
         r"El potencial y el campo de un dipolo puntual (con $r\gg d$) son los de la expresión siguiente. Dependen linealmente del momento dipolar, decrecen con $r^{-2}$ y $r^{-3}$ respectivamente, y la fórmula del eje es $2kp/z^3$. Dimensionalmente $[kp/r^2]=\mathrm{N\,m^2C^{-2}\cdot C\,m\cdot m^{-2}}=\mathrm{N/C\cdot m}$ y $[kp/r^3]=\mathrm{N/C}$, como debe ser para un campo.",
         [r"\boxed{\ V=\dfrac{k\,\vec p\cdot\hat r}{r^{2}},\qquad \vec E=k\,\dfrac{3(\vec p\cdot\hat r)\hat r-\vec p}{r^{3}},\qquad E_{\mathrm{eje}}=\dfrac{2kp}{z^{3}}\ \ }"]),
        ("Particularización al problema",
         rf"Con $q=5$ nC y $d=2$ cm, el momento dipolar es $p=qd={sci(p, 4)}\ \mathrm{{C\,m}}$. A $z=0.50$ m el parámetro de la serie es $\varepsilon=d/z={fx(eps, 3)}$, tan pequeño que la corrección relativa $\varepsilon^2/2$ es del orden de $10^{{-3}}$. Se obtienen los dos valores de la escena: la fórmula exacta es mayor que la de campo lejano por un factor $1+\varepsilon^2/2$.",
         [rf"E_{{lejano}}=\dfrac{{2kp}}{{z^3}}=\dfrac{{2({sci(K_E, 5)})({sci(p, 4)})}}{{({fx(z, 2)})^3}}={sci(ef, 5)}\ \mathrm{{N/C}}",
          rf"E_{{exacto}}=\dfrac{{2kp\,z}}{{(z^2-d^2/4)^2}}={sci(ex, 5)}\ \mathrm{{N/C}},\qquad \dfrac{{E_{{exacto}}}}{{E_{{lejano}}}}={fx(ex / ef, 6)}\approx 1+\dfrac{{\varepsilon^2}}{{2}}={fx(corr, 6)}"]),
        ("Errores comunes y límites de validez",
         r"Suele confundirse el momento dipolar con la carga o con el campo, y olvidar que $p$ es un vector que va de la carga negativa a la positiva. También es frecuente creer que el campo del dipolo decae como $1/r^2$ o que es igual en todas las direcciones: es anisótropo, el doble en el eje que en el ecuador y de sentido contrario. La aproximación dipolar solo vale para $r\gg d$; cerca del dipolo hay que sumar las dos cargas, y para una distribución con momento dipolar nulo domina el cuadrupolo. En medios materiales se cambia $\varepsilon_0$ por $\varepsilon$, y en moléculas reales el dipolo puede ser inducido y depender del campo.",
         []),
    ])


# =========================================================================== ChargedRing
def _ring() -> dict:
    q, a = e2.RING_Q, e2.RING_R
    zm = e2.ring_max_field_position_analytic()
    emax = 2 * K_E * q / (3 * np.sqrt(3) * a**2)
    e5 = float(e2.ring_axis_field_analytic(0.05))
    e30 = float(e2.ring_axis_field_analytic(0.30))
    e_scale = K_E * q / a**2
    return dict(concept=[
        ("Qué se estudia y por qué importa",
         r"El anillo cargado es el bloque elemental de las distribuciones con simetría de revolución: apilando anillos se construyen discos, cilindros, esferas y solenoides eléctricos. Aparece en lentes electrostáticas de haces de partículas, anillos de acumulación y en modelos de moléculas anulares. Su estudio introduce la técnica general de integrar el campo de una distribución continua usando simetrías para reducir las integrales vectoriales a escalares.",
         []),
        ("Magnitudes y convenciones",
         r"El anillo tiene radio $a$ y carga total $Q$ distribuida con densidad lineal uniforme $\lambda=Q/(2\pi a)$ (C/m). El eje $z$ es perpendicular al plano del anillo y pasa por su centro; $z$ mide la distancia al plano. El campo se mide en N/C, y la escala natural es $kQ/a^2$, el campo que crearía una carga puntual $Q$ a la distancia $a$.",
         [r"\lambda=\dfrac{Q}{2\pi a},\qquad [\lambda]=\mathrm{C/m},\qquad E_0=\dfrac{kQ}{a^{2}}"]),
        ("Intuición y analogías",
         r"En el centro del anillo el campo es nulo porque cada elemento de carga queda compensado por el diametralmente opuesto. Lejos, el anillo se ve como una carga puntual. Entre ambos extremos el campo debe crecer y luego decrecer, así que tiene un máximo. Es como una rueda: su influencia a lo largo del eje de giro es nula en el centro y alcanza un máximo a una distancia comparable con el radio.",
         []),
        ("Planteamiento y simetría",
         r"Cada elemento $dq=\lambda a\,d\varphi$ está a la misma distancia $R=\sqrt{z^2+a^2}$ de un punto del eje y su campo forma con el eje un ángulo cuyo coseno es $z/R$. Se descompone $d\vec E$ en una componente axial y otra radial. Por la simetría de revolución, a cada elemento en $\varphi$ le corresponde otro en $\varphi+\pi$ con la misma componente axial y componente radial opuesta, de modo que las componentes perpendiculares al eje se cancelan sin necesidad de integrarlas.",
         [r"dE_z=\dfrac{k\,dq}{R^{2}}\cos\alpha=\dfrac{k\,dq}{z^{2}+a^{2}}\,\dfrac{z}{\sqrt{z^{2}+a^{2}}},\qquad \vec E_\perp=\vec 0"]),
        ("Integración sobre el anillo",
         r"Como $R$ y $\cos\alpha$ son los mismos para todos los elementos, salen de la integral y solo queda $\int dq=Q$. El resultado es exacto y no depende de cómo esté repartida la carga, siempre que la distribución sea uniforme en el ángulo (basta que sea simétrica). Si el anillo tuviera densidad $\lambda(\varphi)$ variable, la componente axial seguiría siendo la misma y aparecerían componentes transversales no nulas.",
         [r"E_z(z)=\int_{0}^{2\pi}\dfrac{k\,z\,\lambda a\,d\varphi}{(z^{2}+a^{2})^{3/2}}=\dfrac{k\,Q\,z}{(z^{2}+a^{2})^{3/2}}",
          r"E_z(z)=\dfrac{kQ}{a^{2}}\,\dfrac{\zeta}{(1+\zeta^{2})^{3/2}},\qquad \zeta=\dfrac za"]),
        ("Vía potencial y campo fuera del eje",
         r"El potencial se calcula sin vectores: todos los elementos están a la misma distancia, luego $V=kQ/\sqrt{z^2+a^2}$ y $E_z=-dV/dz$ reproduce el resultado anterior con una sola derivada. Como $\nabla^2V=0$ fuera del anillo, el desarrollo cerca del eje es $V(\rho,z)\approx V(z)-\tfrac{\rho^2}{4}V''(z)$, lo que da la componente radial del campo cercana al eje; para un punto cualquiera la solución exacta es una integral elíptica completa de primera especie $K(m)$.",
         [r"V(z)=\dfrac{kQ}{\sqrt{z^{2}+a^{2}}},\qquad E_z=-\dfrac{dV}{dz}=\dfrac{kQ\,z}{(z^{2}+a^{2})^{3/2}}",
          r"E_\rho\simeq-\dfrac{\rho}{2}\,\dfrac{dE_z}{dz},\qquad V(\rho,z)=\dfrac{2kQ}{\pi}\,\dfrac{K(m)}{\sqrt{(a+\rho)^{2}+z^{2}}},\quad m=\dfrac{4a\rho}{(a+\rho)^{2}+z^{2}}"]),
        ("Máximo del campo y casos límite",
         r"El máximo de $E_z$ se obtiene anulando la derivada, que lleva a una ecuación en $z^2$. Cerca del centro el campo es lineal en $z$, por lo que una carga puntual de signo opuesto y masa $m$ realiza oscilaciones armónicas con $\omega=\sqrt{kQq/(ma^3)}$. Lejos del anillo, $E_z\to kQ/z^2$ con una corrección de orden $a^2/z^2$ (el momento cuadrupolar del anillo).",
         [r"\dfrac{dE_z}{dz}=kQ\,\dfrac{a^{2}-2z^{2}}{(z^{2}+a^{2})^{5/2}}=0\ \Longrightarrow\ z_{\max}=\dfrac{a}{\sqrt2},\qquad E_{\max}=\dfrac{2}{3\sqrt3}\,\dfrac{kQ}{a^{2}}",
          r"E_z\simeq\dfrac{kQ\,z}{a^{3}}\ (z\ll a),\qquad E_z\simeq\dfrac{kQ}{z^{2}}\left(1-\dfrac{3a^{2}}{2z^{2}}\right)\ (z\gg a)"]),
        ("Resultado general",
         r"El campo sobre el eje de un anillo uniformemente cargado es una función impar de $z$, nula en el centro, máxima en $|z|=a/\sqrt2$ y con cola $kQ/z^2$. Es lineal en $Q$ y homogénea de grado $-2$ en las longitudes. Dimensionalmente $[kQz/R^3]=\mathrm{N\,m^2C^{-2}\cdot C\cdot m\cdot m^{-3}}=\mathrm{N/C}$.",
         [r"\boxed{\ E_z(z)=\dfrac{k\,Q\,z}{\left(z^{2}+a^{2}\right)^{3/2}},\qquad z_{\max}=\dfrac{a}{\sqrt2},\qquad E_{\max}=\dfrac{2kQ}{3\sqrt3\,a^{2}}\ }"]),
        ("Particularización al problema",
         rf"Con $Q=10$ nC y $a=10$ cm la escala de campo es $kQ/a^2={sci(e_scale, 5)}\ \mathrm{{N/C}}$. La fórmula general da los valores de la escena a $5$ cm y a $30$ cm, y el máximo en $z_{{\max}}=a/\sqrt2={fx(zm * 100, 3)}$ cm con el valor $E_{{\max}}=2kQ/(3\sqrt3 a^2)$.",
         [rf"E_z(0.05)=\dfrac{{({sci(K_E, 5)})({sci(q, 3)})(0.05)}}{{(0.05^2+0.10^2)^{{3/2}}}}={fx(e5, 3)}\ \mathrm{{N/C}},\qquad E_z(0.30)={fx(e30, 3)}\ \mathrm{{N/C}}",
          rf"z_{{\max}}=\dfrac{{a}}{{\sqrt2}}={fx(zm * 100, 3)}\ \mathrm{{cm}},\qquad E_{{\max}}=\dfrac{{2kQ}}{{3\sqrt3\,a^2}}={fx(emax, 2)}\ \mathrm{{N/C}}={fx(float(e2.ring_axis_field_analytic(zm)), 2)}\ \mathrm{{N/C}}\ \ (\text{{fórmula directa}})"]),
        ("Errores comunes y límites de validez",
         r"Un error común es sumar los módulos de los $d\vec E$ en lugar de proyectarlos sobre el eje, olvidando que las componentes radiales se cancelan. También se confunde la distancia $z$ al plano con la distancia $R=\sqrt{z^2+a^2}$ al anillo. La fórmula vale solo sobre el eje y para carga uniforme en el ángulo; fuera del eje hay que usar las integrales elípticas o el desarrollo cercano. Para anillos muy delgados, el campo en puntos del propio anillo diverge (carga lineal ideal), de modo que hay que dar un grosor finito. En conductores la carga no se reparte uniformemente.",
         []),
    ])


# =========================================================================== GaussFlux
def _gauss_flux() -> dict:
    q = e3.Q
    phi = e3.flux_sphere_analytic()
    r = 0.3
    e_s = K_E * q / r**2
    area = 4 * np.pi * r**2
    side = 0.4
    face = phi / 6
    return dict(concept=[
        ("Qué se estudia y por qué importa",
         r"La ley de Gauss relaciona el flujo del campo eléctrico a través de una superficie cerrada con la carga encerrada. Es una de las cuatro ecuaciones de Maxwell, y permite calcular campos con simetría sin integrar vectores, justificar que dentro de un conductor en equilibrio no hay carga neta y entender por qué la fuerza eléctrica es del tipo $1/r^2$. Aquí se deduce en forma integral y diferencial desde la ley de Coulomb y se aplica a una carga puntual con superficies distintas.",
         []),
        ("Magnitudes y convenciones",
         r"El flujo $\Phi_E=\oint\vec E\cdot d\vec A$ se mide en $\mathrm{N\,m^2/C}$ (equivalente a V$\cdot$m). El vector $d\vec A=\hat n\,dA$ apunta hacia afuera de la superficie cerrada; el flujo es positivo si el campo sale. La carga $q_{enc}$ es la contenida en el interior, con signo. El flujo por una carga unidad vale $1/\varepsilon_0\approx1.13\times10^{11}\ \mathrm{N\,m^2/C}$.",
         [r"[\Phi_E]=\mathrm{N\,m^{2}\,C^{-1}}=\mathrm{V\,m},\qquad d\vec A=\hat n\,dA\ \ (\hat n\text{ exterior})"]),
        ("Intuición y analogías",
         r"Si las líneas de campo representan un fluido incompresible que sale de la carga, el flujo mide cuánto fluido atraviesa la superficie: lo que sale es lo que produce la fuente en el interior, sin importar la forma de la superficie. Una línea que entra y sale por una superficie sin carga dentro no aporta flujo neto. El flujo depende solo de cuántas líneas nacen dentro, no de la geometría ni de que la carga esté o no centrada.",
         []),
        ("De Coulomb al ángulo sólido",
         r"El flujo infinitesimal de una carga puntual a través de un elemento de área se escribe con el ángulo $\alpha$ entre $\hat n$ y $\hat r$. La proyección $dA\cos\alpha$ es el área perpendicular a la línea visual, y dividida por $r^2$ es el ángulo sólido $d\Omega$ bajo el que se ve el elemento desde la carga. La dependencia $1/r^2$ del campo cancela exactamente el crecimiento $r^2$ del área y por eso el resultado es independiente de la distancia y de la forma. Esta cancelación solo ocurre con exponente $2$.",
         [r"d\Phi=\vec E\cdot\hat n\,dA=\dfrac{kq}{r^{2}}\cos\alpha\,dA=\dfrac{q}{4\pi\varepsilon_0}\,d\Omega,\qquad d\Omega=\dfrac{\cos\alpha\,dA}{r^{2}}"]),
        ("Superficie cerrada: dentro y fuera",
         r"Para una superficie que encierra la carga, cada rayo desde la carga la cruza (de modo neto) una vez hacia afuera, así que $\oint d\Omega=4\pi$ y el flujo es $q/\varepsilon_0$, sea cual sea la forma. Si la carga está fuera, cada rayo que entra por la superficie sale por otro punto, y las contribuciones $\pm d\Omega$ se cancelan, con flujo nulo. Por superposición, para un conjunto de cargas el flujo total es la suma de las contribuciones, y solo las cargas interiores cuentan.",
         [r"\oint_S d\Omega=\begin{cases}4\pi,& \text{carga dentro}\\ 0,& \text{carga fuera}\end{cases}\qquad\Longrightarrow\qquad \oint_S\vec E\cdot d\vec A=\dfrac{q_{enc}}{\varepsilon_0}",
          r"\oint_S\vec E\cdot d\vec A=\dfrac{1}{\varepsilon_0}\int_V\rho\,dV=\dfrac{Q_{enc}}{\varepsilon_0}\qquad(\text{distribución continua})"]),
        ("Forma diferencial y función delta",
         r"Aplicando el teorema de la divergencia a un volumen arbitrario se obtiene la forma local $\nabla\cdot\vec E=\rho/\varepsilon_0$. Para una carga puntual, $\nabla\cdot(\hat r/r^2)$ es nulo en todo punto salvo el origen, pero su flujo total es $4\pi$: la divergencia es, por lo tanto, una distribución delta de Dirac. Esto reconcilia la forma integral con la diferencial y muestra que la ley de Gauss y la de Coulomb son equivalentes cuando se añade $\nabla\times\vec E=0$.",
         [r"\int_V\nabla\cdot\vec E\,dV=\oint_{\partial V}\vec E\cdot d\vec A\ \Longrightarrow\ \nabla\cdot\vec E=\dfrac{\rho}{\varepsilon_0}",
          r"\nabla\cdot\dfrac{\hat r}{r^{2}}=4\pi\,\delta^{3}(\vec r),\qquad \nabla\times\vec E=\vec 0"]),
        ("Simetría, partes del flujo y casos límite",
         r"Cuando hay simetría (esfera con la carga en el centro, cubo con la carga en el centro), el flujo se reparte de forma igual entre partes equivalentes: cada cara de un cubo centrado recibe $1/6$ del total, lo que se verifica con el ángulo sólido de un rectángulo de semilados $\alpha,\beta$ a distancia $h$. Si la carga está sobre la superficie, el flujo es la fracción de ángulo sólido (la mitad en una superficie lisa). Un casquete esférico de semiángulo $\theta_0$ recibe una fracción proporcional a $1-\cos\theta_0$. Si la ley de fuerza fuera $r^{-(2+\delta)}$, el flujo dependería del radio, lo que hace de la ley de Gauss una prueba experimental de la ley de Coulomb.",
         [r"\Omega_{\mathrm{rect}}=4\arctan\dfrac{\alpha\beta}{h\sqrt{h^{2}+\alpha^{2}+\beta^{2}}},\qquad \alpha=\beta=h=\dfrac a2\ \Rightarrow\ \Omega=\dfrac{4\pi}{6}",
          r"\Phi_{\mathrm{casquete}}=\dfrac{q}{2\varepsilon_0}\,(1-\cos\theta_0),\qquad \Phi\propto R^{-\delta}\ \ \text{si}\ \ E\propto r^{-(2+\delta)}"]),
        ("Resultado general",
         r"El flujo del campo eléctrico a través de cualquier superficie cerrada es la carga neta encerrada dividida por $\varepsilon_0$, con independencia de la forma, tamaño o posición de las cargas dentro. En forma local, la divergencia del campo es la densidad de carga por $1/\varepsilon_0$. Dimensionalmente $[q/\varepsilon_0]=\mathrm{C}/(\mathrm{C^2N^{-1}m^{-2}})=\mathrm{N\,m^2/C}$, igual a $[E][A]$.",
         [r"\boxed{\ \oint_S\vec E\cdot d\vec A=\dfrac{q_{enc}}{\varepsilon_0}\quad\Longleftrightarrow\quad \nabla\cdot\vec E=\dfrac{\rho}{\varepsilon_0}\ }"]),
        ("Particularización al problema",
         rf"Con $q=2$ nC el flujo vale $\Phi=q/\varepsilon_0={fx(phi, 2)}\ \mathrm{{N\,m^2/C}}$ para la esfera de $0.3$ m, para el cubo de $0.4$ m, y con la carga descentrada; en la esfera con carga centrada puede comprobarse con el producto del campo en la superficie por su área. Para el cubo con la carga en el centro, cada cara recibe un sexto del total. Si la carga queda fuera de la superficie, el flujo es cero.",
         [rf"E(0.3\ \mathrm{{m}})=\dfrac{{kq}}{{r^2}}={fx(e_s, 3)}\ \mathrm{{N/C}},\qquad A=4\pi r^2={fx(area, 4)}\ \mathrm{{m^2}}",
          rf"\Phi_{{esfera}}=E\,A={fx(e_s * area, 2)}\ \mathrm{{N\,m^2/C}}=\dfrac{{q}}{{\varepsilon_0}}={fx(phi, 2)}\ \mathrm{{N\,m^2/C}}",
          rf"\Phi_{{cara}}=\dfrac{{\Phi}}{{6}}={fx(face, 2)}\ \mathrm{{N\,m^2/C}},\qquad \Phi_{{cubo}}=6\,\Phi_{{cara}}={fx(6 * face, 2)}\ \mathrm{{N\,m^2/C}},\qquad \Phi_{{fuera}}=0"]),
        ("Errores comunes y límites de validez",
         r"Se suele creer que el campo en la superficie depende solo de la carga interior: lo que depende de ella es el flujo, y el campo local incluye las cargas externas. Otro error es usar la ley de Gauss para calcular $E$ sin simetría: es válida siempre, pero solo permite despejar $E$ cuando la simetría lo saca de la integral. Tampoco hay que confundir flujo nulo con campo nulo. La ley vale en electrostática y también para campos dependientes del tiempo (es una de las ecuaciones de Maxwell), pero en medios materiales debe escribirse con $\vec D$ y la carga libre.",
         []),
    ])


# =========================================================================== SphereFieldProfile
def _sphere() -> dict:
    q, big_r = e3.Q_SPH, e3.R_SPH
    rho = 3 * q / (4 * np.pi * big_r**3)
    e_r = float(e3.sphere_field_analytic(big_r))
    e_h = float(e3.sphere_field_analytic(0.05))
    e_o = float(e3.sphere_field_analytic(0.20))
    v0 = 1.5 * K_E * q / big_r
    vs = K_E * q / big_r
    u = 3 * K_E * q**2 / (5 * big_r)
    return dict(concept=[
        ("Qué se estudia y por qué importa",
         r"La esfera uniformemente cargada es el modelo más simple de un núcleo atómico, de una gota cargada o de una nube de plasma; es también el ejemplo canónico de la ley de Gauss con simetría esférica. Su campo interior, lineal en la distancia al centro, explica los modelos atómicos de J. J. Thomson y el potencial armónico en el interior de distribuciones de carga. Se deduce el perfil radial del campo, el potencial, la energía y su generalización a perfiles de densidad arbitrarios.",
         []),
        ("Magnitudes y convenciones",
         r"La esfera aislante de radio $R$ tiene una carga total $Q$ distribuida con densidad volumétrica uniforme $\rho=3Q/(4\pi R^3)$ (C/m$^3$). Se usa la coordenada radial $r$ medida desde el centro y el vector unitario radial $\hat r$. El campo se mide en N/C, el potencial en voltios, y la energía almacenada en joules.",
         [r"\rho=\dfrac{Q}{\tfrac43\pi R^{3}},\qquad [\rho]=\mathrm{C/m^{3}},\qquad Q_{enc}(r)=Q\,\dfrac{r^{3}}{R^{3}}\ \ (r<R)"]),
        ("Intuición y analogías",
         r"Fuera de la esfera el campo es idéntico al de una carga puntual en el centro, lo mismo que ocurre con la gravedad de una Tierra esférica. Dentro, solo la carga que queda dentro de la esfera de radio $r$ contribuye al campo, y esa carga crece como $r^3$ mientras que el área crece como $r^2$, por lo que el campo crece linealmente desde cero en el centro. El campo máximo está en la superficie.",
         []),
        ("Simetría y forma del campo",
         r"Como la distribución es invariante ante cualquier rotación en torno al centro, el campo también lo es: no puede tener componente tangencial (una dirección tangente privilegiada rompería la simetría) y su módulo solo depende de $r$. Así $\vec E=E(r)\hat r$. Esto permite elegir como superficie de Gauss una esfera concéntrica de radio $r$, en la cual $E$ es constante y paralelo al elemento de área.",
         [r"\vec E(\vec r)=E(r)\,\hat r,\qquad \oint_{S_r}\vec E\cdot d\vec A=E(r)\,4\pi r^{2}"]),
        ("Ley de Gauss para un perfil de densidad arbitrario",
         r"Con la superficie de Gauss esférica se despeja directamente $E(r)$ en función de la carga encerrada. La carga encerrada se obtiene integrando la densidad en capas esféricas. Esta fórmula vale para cualquier $\rho(r)$ con simetría esférica y contiene el teorema de las capas: una capa esférica no produce campo en su interior y se comporta como una carga puntual fuera.",
         [r"E(r)=\dfrac{1}{4\pi\varepsilon_0}\,\dfrac{Q_{enc}(r)}{r^{2}},\qquad Q_{enc}(r)=4\pi\int_{0}^{r}\rho(r')\,r'^{2}\,dr'",
          r"\dfrac{1}{r^{2}}\dfrac{d}{dr}\left(r^{2}E\right)=\dfrac{\rho(r)}{\varepsilon_0}\qquad(\text{forma diferencial: }\nabla\cdot\vec E=\rho/\varepsilon_0)"]),
        ("Esfera uniforme: campo interior y exterior",
         r"Para $\rho$ constante, $Q_{enc}=Q\,(r/R)^3$ dentro y $Q$ fuera. El campo crece linealmente hasta la superficie y decae como $1/r^2$ después. Es continuo en $r=R$ porque no hay carga superficial, aunque su derivada no lo es. La forma diferencial da el mismo perfil de manera independiente: $E=\rho r/(3\varepsilon_0)$ en el interior.",
         [r"E(r)=\begin{cases}\dfrac{kQ\,r}{R^{3}}=\dfrac{\rho\,r}{3\varepsilon_0}, & r\le R\\[2mm] \dfrac{kQ}{r^{2}}, & r\ge R\end{cases}",
          r"E_{\max}=E(R)=\dfrac{kQ}{R^{2}}"]),
        ("Potencial, ecuación de Poisson y energía",
         r"El potencial se obtiene integrando $-\vec E\cdot d\vec r$ desde el infinito, con $V\to0$ allí; es continuo en $r=R$ y es máximo en el centro. Debe cumplir la ecuación de Poisson $\nabla^2V=-\rho/\varepsilon_0$, lo que se verifica derivando el interior. La energía almacenada se obtiene integrando la densidad de energía $\tfrac12\varepsilon_0E^2$ en todo el espacio; el exterior aporta $kQ^2/(2R)$ y el interior $kQ^2/(10R)$.",
         [r"V(r)=\begin{cases}\dfrac{kQ\,(3R^{2}-r^{2})}{2R^{3}}, & r\le R\\[2mm] \dfrac{kQ}{r}, & r\ge R\end{cases},\qquad \dfrac1{r^{2}}\dfrac{d}{dr}\!\left(r^{2}\dfrac{dV}{dr}\right)=-\dfrac{3kQ}{R^{3}}=-\dfrac{\rho}{\varepsilon_0}",
          r"U=\dfrac{\varepsilon_0}{2}\int E^{2}\,dV=\dfrac{kQ^{2}}{2R}+\dfrac{kQ^{2}}{10R}=\dfrac{3}{5}\,\dfrac{kQ^{2}}{R}"]),
        ("Resultado general",
         r"El campo de una esfera uniformemente cargada es lineal en el interior y coulombiano en el exterior, con máximo en la superficie. Es lineal en $Q$ y se escala con $R$. Dimensionalmente $[kQr/R^3]=\mathrm{N\,m^2C^{-2}\cdot C\cdot m\cdot m^{-3}}=\mathrm{N/C}$. Una carga de prueba de signo opuesto en el interior ejecuta oscilaciones armónicas de frecuencia $\omega=\sqrt{kQq/(mR^3)}$.",
         [r"\boxed{\ E(r)=\dfrac{kQ}{R^{3}}\,r\ \ (r\le R),\qquad E(r)=\dfrac{kQ}{r^{2}}\ \ (r\ge R)\ }"]),
        ("Particularización al problema",
         rf"Con $Q=5$ nC y $R=10$ cm, la escala del campo es $kQ/R^2={fx(e_r, 1)}\ \mathrm{{N/C}}$ y la densidad es $\rho={sci(rho, 4)}\ \mathrm{{C/m^3}}$. A mitad del radio, el campo es la mitad del máximo; a $r=2R$, un cuarto. El potencial en el centro es $1.5$ veces el de la superficie, y la energía almacenada es $3kQ^2/(5R)$. Los valores coinciden con los de la escena, que usa el Monte Carlo para verificarlos.",
         [rf"E(0.05)=\dfrac{{kQ\,r}}{{R^3}}={fx(e_h, 1)}\ \mathrm{{N/C}},\qquad E(0.10)=\dfrac{{kQ}}{{R^2}}={fx(e_r, 1)}\ \mathrm{{N/C}},\qquad E(0.20)=\dfrac{{kQ}}{{r^2}}={fx(e_o, 1)}\ \mathrm{{N/C}}",
          rf"V(0)=\dfrac{{3kQ}}{{2R}}={fx(v0, 1)}\ \mathrm{{V}},\qquad V(R)=\dfrac{{kQ}}{{R}}={fx(vs, 1)}\ \mathrm{{V}},\qquad U=\dfrac{{3kQ^2}}{{5R}}={sci(u, 4)}\ \mathrm{{J}}"]),
        ("Errores comunes y límites de validez",
         r"Un error clásico es aplicar $E=kQ/r^2$ dentro de la esfera usando la carga total, en lugar de la encerrada. También se confunde una esfera aislante (carga en el volumen) con una conductora (carga solo en la superficie, campo interior nulo). La simetría esférica es necesaria: si la densidad no es esférica la ley de Gauss sigue valiendo, pero no permite despejar el campo. El modelo vale para el vacío y para $\rho$ fija; en un dieléctrico real, la polarización modifica el campo, y en un núcleo la distribución es de tipo Fermi, no uniforme.",
         []),
    ])


# =========================================================================== DiskPotential
def _disk() -> dict:
    sg, big_r = e4.SIGMA, e4.R_DISK
    z = 0.05
    v = float(e4.disk_potential_analytic(z))
    ez = float(e4.disk_field_analytic(z))
    plano = sg / (2 * EPS0)
    qd = sg * np.pi * big_r**2
    cosal = z / np.sqrt(z**2 + big_r**2)
    v0 = sg * big_r / (2 * EPS0)
    return dict(concept=[
        ("Qué se estudia y por qué importa",
         r"El disco cargado conecta el anillo con el plano infinito y es la base de los condensadores de placas, las lentes electrostáticas de apertura y los modelos de discos de acreción. Su estudio muestra cómo se pasa del potencial al campo, y cómo los límites geométricos (disco muy grande o muy pequeño) reproducen los resultados de simetría plana y de carga puntual. Se obtienen el potencial y el campo sobre el eje para un disco de radio y densidad superficial arbitrarios.",
         []),
        ("Magnitudes y convenciones",
         r"El disco tiene radio $R$ y densidad superficial uniforme $\sigma$ (C/m$^2$), con carga total $Q=\sigma\pi R^2$. El eje $z$ es perpendicular al disco por su centro. El potencial se mide en voltios, con $V\to0$ en el infinito, y el campo en V/m. El parámetro adimensional natural es $\zeta=z/R$, y la escala del campo es $\sigma/(2\varepsilon_0)$, el campo de un plano infinito.",
         [r"Q=\sigma\pi R^{2},\qquad [\sigma]=\mathrm{C/m^{2}},\qquad E_{\mathrm{plano}}=\dfrac{\sigma}{2\varepsilon_0},\qquad \zeta=\dfrac zR"]),
        ("Intuición y analogías",
         r"El disco es un apilamiento de anillos concéntricos. Muy cerca de su centro, comparado con su tamaño, parece un plano infinito y el campo es uniforme e igual a $\sigma/2\varepsilon_0$; muy lejos, parece una carga puntual y el campo decae como $1/z^2$. El campo, a diferencia del potencial, salta al atravesar la superficie cargada, igual que el campo de un plano infinito invierte su sentido al cruzarlo.",
         []),
        ("Planteamiento: suma de anillos",
         r"Se divide el disco en anillos de radio $s$ y ancho $ds$, cada uno con carga $dq=\sigma\,2\pi s\,ds$. Todos los puntos de un anillo están a la misma distancia $\sqrt{s^2+z^2}$ del punto del eje, y el potencial es escalar: se suma sin descomponer vectores. La superposición es válida porque la ecuación de Poisson es lineal. Se integra desde $s=0$ hasta $s=R$.",
         [r"dV=\dfrac{k\,dq}{\sqrt{s^{2}+z^{2}}}=\dfrac{\sigma}{2\varepsilon_0}\,\dfrac{s\,ds}{\sqrt{s^{2}+z^{2}}}",
          r"V(z)=\dfrac{\sigma}{2\varepsilon_0}\int_{0}^{R}\dfrac{s\,ds}{\sqrt{s^{2}+z^{2}}}"]),
        ("Integración del potencial",
         r"La integral es inmediata con el cambio $u=s^2+z^2$, con $du=2s\,ds$. El valor absoluto de $z$ aparece porque la raíz es siempre positiva: el potencial es par en $z$ por la simetría de reflexión respecto del plano del disco. Es continuo en $z=0$ y allí vale $\sigma R/2\varepsilon_0$, finito pese a que la densidad es superficial.",
         [r"V(z)=\dfrac{\sigma}{4\varepsilon_0}\int_{z^{2}}^{z^{2}+R^{2}}\dfrac{du}{\sqrt u}=\dfrac{\sigma}{2\varepsilon_0}\left(\sqrt{z^{2}+R^{2}}-|z|\right)",
          r"V(z)=\dfrac{\sigma R}{2\varepsilon_0}\left(\sqrt{1+\zeta^{2}}-|\zeta|\right)"]),
        ("Campo a partir del potencial",
         r"El campo sobre el eje es $E_z=-dV/dz$, válido porque en el eje la simetría axial elimina la componente transversal. El resultado se puede interpretar geométricamente: es proporcional al ángulo sólido $\Omega$ bajo el que se ve el disco desde el punto, con $\Omega=2\pi(1-\cos\alpha)$ y $\tan\alpha=R/z$. Sobre la superficie el campo salta de $-\sigma/2\varepsilon_0$ a $+\sigma/2\varepsilon_0$, un salto total de $\sigma/\varepsilon_0$, consistente con la condición de contorno de la ley de Gauss.",
         [r"E_z(z)=-\dfrac{dV}{dz}=\dfrac{\sigma}{2\varepsilon_0}\left[\operatorname{sgn}(z)-\dfrac{z}{\sqrt{z^{2}+R^{2}}}\right],\qquad E_z(z)=\dfrac{\sigma\,\Omega}{4\pi\varepsilon_0}\ \ (z>0)",
          r"E_z(0^{+})-E_z(0^{-})=\dfrac{\sigma}{\varepsilon_0}"]),
        ("Casos límite y campo fuera del eje",
         r"Para $R\to\infty$ el campo tiende a $\sigma/2\varepsilon_0$ (plano infinito) y el potencial diverge, lo que obliga a elegir otra referencia. Para $z\gg R$ se recupera la carga puntual $Q$ con una corrección cuadrupolar. Fuera del eje, como $\nabla^2V=0$, el desarrollo cercano da una componente radial proporcional a la distancia al eje, que sale de la derivada de $E_z$ y es lo que hace divergir las líneas de campo a la vez que se alejan del disco.",
         [r"V\simeq\dfrac{kQ}{z}\left(1-\dfrac{R^{2}}{4z^{2}}\right),\qquad E_z\simeq\dfrac{kQ}{z^{2}}\left(1-\dfrac{3R^{2}}{4z^{2}}\right)\qquad(z\gg R)",
          r"E_\rho\simeq-\dfrac{\rho}{2}\,\dfrac{dE_z}{dz}=\dfrac{\sigma R^{2}\,\rho}{4\varepsilon_0\,(z^{2}+R^{2})^{3/2}}\qquad(\rho\ll R)"]),
        ("Resultado general",
         r"El potencial y el campo de un disco uniformemente cargado sobre su eje son los de la expresión siguiente. Son lineales en $\sigma$ y tienen escala $\sigma R/(2\varepsilon_0)$ para $V$ y $\sigma/(2\varepsilon_0)$ para $E$. Dimensionalmente $[\sigma R/\varepsilon_0]=\mathrm{C\,m^{-2}\cdot m\cdot N\,m^2C^{-2}}=\mathrm{N\,m/C}=\mathrm{V}$.",
         [r"\boxed{\ V(z)=\dfrac{\sigma}{2\varepsilon_0}\left(\sqrt{z^{2}+R^{2}}-|z|\right),\qquad E_z(z)=\dfrac{\sigma}{2\varepsilon_0}\left(1-\dfrac{z}{\sqrt{z^{2}+R^{2}}}\right)\ \ (z>0)\ }"]),
        ("Particularización al problema",
         rf"Con $\sigma=2\ \mu\mathrm{{C/m^2}}$, $R=10$ cm y $z=5$ cm ($\zeta=0.5$), la escala del campo es $\sigma/2\varepsilon_0={sci(plano / 1000, 4)}\ \mathrm{{kV/m}}$, y $\cos\alpha=z/\sqrt{{z^2+R^2}}={fx(cosal, 4)}$. La fórmula general reproduce los valores de la escena. El campo de un disco es menor que el de un plano porque el factor entre paréntesis es menor que $1$, y la carga total es $Q={sci(qd, 4)}$ C.",
         [rf"V(z)=\dfrac{{\sigma}}{{2\varepsilon_0}}\left(\sqrt{{z^2+R^2}}-z\right)={sci(plano, 4)}\,(0.11180-0.05)={fx(v / 1000, 3)}\ \mathrm{{kV}}",
          rf"E_z(z)=\dfrac{{\sigma}}{{2\varepsilon_0}}\left(1-\dfrac{{z}}{{\sqrt{{z^2+R^2}}}}\right)={sci(plano, 4)}\,(1-{fx(cosal, 4)})={fx(ez / 1000, 3)}\ \mathrm{{kV/m}}",
          rf"V(0)=\dfrac{{\sigma R}}{{2\varepsilon_0}}={fx(v0 / 1000, 3)}\ \mathrm{{kV}}"]),
        ("Errores comunes y límites de validez",
         r"Un error común es sumar los módulos de los campos de los anillos, en vez de proyectar sobre el eje o, más fácil, calcular el potencial y derivar. También es habitual olvidar el $|z|$ y obtener un potencial no simétrico, o usar $E=\sigma/2\varepsilon_0$ del plano infinito sin comprobar que $z\ll R$. La fórmula vale solo sobre el eje; fuera de él el campo no es axial. En un conductor la carga no se reparte uniformemente (se acumula en el borde), y en un disco real de espesor finito hay que modelar el volumen.",
         []),
    ])


# =========================================================================== EquipotentialPair
def _pair() -> dict:
    qa, qb, d = e4.QA, e4.QB, e4.D
    u = e4.interaction_energy()
    x0, y0 = 0.05, 0.10
    ra, rb = float(np.hypot(x0 + d / 2, y0)), float(np.hypot(x0 - d / 2, y0))
    v_f = K_E * (qa / ra + qb / rb)
    v_n = float(e4.potential_pair(x0, y0))
    e_mid = 2 * K_E * abs(qa) / (d / 2) ** 2
    f_im = K_E * qa**2 / d**2
    p = abs(qa) * d
    return dict(concept=[
        ("Qué se estudia y por qué importa",
         r"El potencial de dos cargas iguales y opuestas es el modelo de un dipolo finito, de la membrana de una célula, de las placas de un condensador visto de perfil y de un par de electrodos. Las equipotenciales son las curvas de nivel del potencial: dibujan el relieve que determina cómo se moverían las cargas, y las líneas de campo son sus pendientes más pronunciadas. Se obtiene el potencial, el campo, la geometría de las equipotenciales, la energía del par y la relación con el método de imágenes.",
         []),
        ("Magnitudes y convenciones",
         r"Las cargas $q_a=+q$ y $q_b=-q$ están en $(-d/2,0)$ y $(d/2,0)$. El potencial se mide en voltios (J/C), con cero en el infinito; la energía de interacción en joules. El signo de $U$ distingue las configuraciones ligadas ($U<0$) de las repulsivas ($U>0$). Las distancias a las cargas son $r_a$ y $r_b$.",
         [r"r_a=\sqrt{(x+d/2)^{2}+y^{2}},\qquad r_b=\sqrt{(x-d/2)^{2}+y^{2}},\qquad [V]=\mathrm{J/C},\qquad [U]=\mathrm{J}"]),
        ("Intuición y analogías",
         r"El potencial es un paisaje con una colina sobre la carga positiva y un pozo sobre la negativa; las equipotenciales son las curvas de nivel, y una carga positiva rueda cuesta abajo siguiendo las líneas de campo, siempre perpendiculares a esas curvas. El plano que equidista de ambas cargas está a nivel del infinito, porque las contribuciones de ambas se anulan.",
         []),
        ("Potencial y simetrías",
         r"El potencial del par es la suma de los potenciales coulombianos, por superposición. Bajo la reflexión $x\to-x$ las cargas se intercambian pero cambian de signo, por lo que $V(-x,y)=-V(x,y)$: el potencial es impar en $x$ y se anula en toda la mediatriz $x=0$. Además tiene simetría de revolución en torno al eje que une las cargas, así que el problema en 3D se reduce al plano $(x,y)$ con $y\ge0$.",
         [r"V(x,y)=k\left(\dfrac{q_a}{r_a}+\dfrac{q_b}{r_b}\right)=kq\left(\dfrac{1}{r_a}-\dfrac{1}{r_b}\right),\qquad V(-x,y)=-V(x,y)"]),
        ("Geometría de las equipotenciales",
         r"Las equipotenciales son curvas $V=V_0$ implícitas (de cuarto grado en general, no círculos; los círculos aparecen en el problema bidimensional de dos hilos). Cerca de cada carga son aproximadamente círculos, porque domina el término de esa carga; a gran distancia son las curvas del dipolo, con $\cos\theta/r^2=\text{cte}$. La superficie $V=0$ es el plano mediatriz. Como $\vec E=-\nabla V$, el campo es perpendicular a cada equipotencial en todos sus puntos y apunta hacia el potencial decreciente.",
         [r"V\simeq\dfrac{k\,p\,\cos\theta}{r^{2}},\quad p=q\,d,\ \ r\gg d;\qquad V\simeq\dfrac{kq}{r_a}\ \ \text{cerca de }q_a",
          r"\vec E=-\nabla V=k\left[q_a\,\dfrac{\vec r-\vec r_a}{r_a^{3}}+q_b\,\dfrac{\vec r-\vec r_b}{r_b^{3}}\right],\qquad \vec E\perp\{V=\text{cte}\}"]),
        ("El campo sobre la mediatriz",
         r"En los puntos de la mediatriz las dos cargas están a la misma distancia $\varrho=\sqrt{y^2+d^2/4}$, y las componentes paralelas al plano se cancelan mientras que las componentes perpendiculares se suman. El campo es entonces normal al plano $V=0$, como corresponde a una superficie equipotencial. En el punto medio vale $8kq/d^2$, el doble del de una carga a la distancia $d/2$.",
         [r"E_x(0,y)=\dfrac{k\,q\,d}{\left(y^{2}+d^{2}/4\right)^{3/2}},\qquad E_y(0,y)=0",
          r"E(0,0)=\dfrac{8kq}{d^{2}}"]),
        ("Energía del par y energía del campo",
         r"La energía del sistema es el trabajo que hay que hacer para armarlo trayendo una carga desde el infinito hasta el potencial de la otra. Es negativa para cargas opuestas (se atraen, el sistema está ligado). Se puede ver en el campo: la densidad de energía $\tfrac12\varepsilon_0E^2$ contiene dos términos propios (que divergen para cargas puntuales y se asignan a la energía de formación de cada carga) y un término cruzado finito, $\varepsilon_0\int\vec E_a\cdot\vec E_b\,dV$, que vale exactamente $kq_aq_b/d$.",
         [r"U=q_b\,V_a(\vec r_b)=k\,\dfrac{q_aq_b}{d}=-\dfrac{kq^{2}}{d}",
          r"\varepsilon_0\int\vec E_a\cdot\vec E_b\,dV=k\,\dfrac{q_aq_b}{d}"]),
        ("Unicidad y método de imágenes",
         r"Las soluciones de la ecuación de Laplace que toman valores dados en el contorno son únicas. Como el plano $x=0$ está a $V=0$, el semiespacio $x>0$ con la carga $-q$ y las condiciones $V=0$ en el plano y $V\to0$ en el infinito tiene la misma solución que una carga $-q$ frente a un plano conductor a tierra, con una carga imagen $+q$ simétrica. De ahí se obtienen la fuerza sobre la carga, que es la de dos cargas a distancia $d$, y la energía del sistema carga-conductor, que es la mitad de la del par, porque el campo solo existe en medio espacio.",
         [r"F_{\mathrm{imagen}}=\dfrac{kq^{2}}{(2h)^{2}},\quad h=\dfrac d2;\qquad U_{\mathrm{carga+plano}}=\dfrac12\,U_{\mathrm{par}}=-\dfrac{kq^{2}}{2d}"]),
        ("Resultado general",
         r"El potencial de un par de cargas opuestas es la diferencia de dos términos coulombianos, impar bajo la reflexión del par, con el plano mediatriz como equipotencial $V=0$; el campo es su gradiente cambiado de signo y la energía del par es $kq_aq_b/d$. Dimensionalmente $[kq^2/d]=\mathrm{N\,m^2C^{-2}\cdot C^2\cdot m^{-1}}=\mathrm{N\,m}=\mathrm{J}$.",
         [r"\boxed{\ V=kq\left(\dfrac{1}{r_a}-\dfrac{1}{r_b}\right),\qquad \vec E=-\nabla V,\qquad U=\dfrac{k\,q_aq_b}{d}\ }"]),
        ("Particularización al problema",
         rf"Con $q_a=+4$ nC, $q_b=-4$ nC y $d=20$ cm, la energía es $U=kq_aq_b/d={fx(u * 1e6, 4)}\ \mu\mathrm{{J}}$ (negativa: par ligado). En el plano mediatriz $V=0$, y el campo en el punto medio es $8kq/d^2={sci(e_mid, 4)}\ \mathrm{{N/C}}$. Como comprobación en el punto $({x0},{y0})$ m, con $r_a={fx(ra, 4)}$ m y $r_b={fx(rb, 4)}$ m, la fórmula coincide con la función de la escena.",
         [rf"U=\dfrac{{({sci(K_E, 5)})({sci(qa, 1)})({sci(qb, 1)})}}{{{fx(d, 2)}}}={fx(u * 1e6, 4)}\ \mu\mathrm{{J}}",
          rf"V(0,y)=kq\left(\dfrac{{1}}{{r}}-\dfrac{{1}}{{r}}\right)=0,\qquad E(0,0)=\dfrac{{8kq}}{{d^2}}={sci(e_mid, 4)}\ \mathrm{{N/C}}",
          rf"V({x0},{y0})=kq\left(\dfrac{{1}}{{r_a}}-\dfrac{{1}}{{r_b}}\right)={fx(v_f, 4)}\ \mathrm{{V}}\ \ ({fx(v_n, 4)}\ \mathrm{{V}}\ \text{{en la escena}}),\qquad F_{{imagen}}={sci(f_im, 4)}\ \mathrm{{N}}"]),
        ("Errores comunes y límites de validez",
         r"Es muy frecuente suponer que las equipotenciales de un par de cargas son círculos; solo el plano mediatriz es exacto. También se confunde la energía de interacción con la energía total (que incluye energías propias) y se olvida el signo negativo del par ligado. El campo eléctrico no es el gradiente del módulo del potencial, sino $-\nabla V$ con signo. La aproximación dipolar solo vale a $r\gg d$ y falla cerca de las cargas, donde $V$ diverge. En la práctica el gradiente numérico de una malla pierde precisión cerca de las cargas, por lo que la comparación se hace a distancias mayores que unos pocos pasos de malla.",
         []),
    ])


DETAIL = {
    "CoulombForces": _coulomb(),
    "EquilibriumPoint": _equilibrium(),
    "DipoleField": _dipole(),
    "ChargedRing": _ring(),
    "GaussFlux": _gauss_flux(),
    "SphereFieldProfile": _sphere(),
    "DiskPotential": _disk(),
    "EquipotentialPair": _pair(),
}
