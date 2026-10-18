"""Explicacion y pasos a mano: Volumen 2, capitulos 5 a 8 (CoaxialLaplace, CapacitorNetwork, DrudeElectrons,
TaperedWire, MeshCircuit, RCCharging, HallEffect). Todas las cifras se calculan con `physics/`."""
from __future__ import annotations

import numpy as np

from content.texfmt import fx, sci
from physics import v2ch05_capacitance as e5
from physics import v2ch06_current as e6
from physics import v2ch07_circuits as e7
from physics import v2ch08_magnetic_field as m8
from physics.constants import E_CH, EPS0, K_B, M_E
from physics.integrators import integrate, rk4_step


def _table(head: list[str], rows: list[list[str]]) -> str:
    cols = "c" * len(head)
    body = r"\\ ".join(" & ".join(r) for r in rows)
    return r"\begin{array}{" + cols + r"}" + " & ".join(head) + r"\\ \hline " + body + r"\end{array}"


# ------------------------------------------------------------------------------------------ CoaxialLaplace
def _coax() -> dict:
    a, b, v0 = e5.A_IN, e5.B_OUT, 1.0
    c_an = e5.coax_capacitance_per_length_analytic()
    n, its = 161, 4000
    xs, V, h, fixed = e5.coax_laplace_numeric(n=n, iterations=its)
    c_nu = e5.coax_capacitance_numeric(n=n, iterations=its)
    omega = 2.0 / (1.0 + np.sin(np.pi / n))
    ratio = np.log(b / a)
    R = np.hypot(*np.meshgrid(xs, xs, indexing="ij"))
    n_in, n_out, n_free = int((R <= a).sum()), int((R >= b).sum()), int((~fixed).sum())
    # primer nodo rojo libre sobre el eje (fila central) pegado al conductor interno
    jc = n // 2
    ii = next(i for i in range(jc, n) if (not fixed[i, jc]) and (i + jc) % 2 == 0 and fixed[i - 1, jc])
    nb = [0.0 + (1.0 if fixed[ii - 1, jc] and R[ii - 1, jc] <= a else 0.0), 0.0, 0.0, 0.0]
    avg0 = sum(nb) / 4
    v_new = omega * avg0
    # energia
    gx, gy = np.gradient(V, h, h)
    e2 = gx**2 + gy**2
    s_e2 = float(e2[~fixed].sum())
    u = 0.5 * EPS0 * s_e2 * h * h
    # nodo de ejemplo del gradiente en la solucion final
    ig = ii + 10
    dvx = (V[ig + 1, jc] - V[ig - 1, jc]) / (2 * h)
    e_exact = v0 / (R[ig, jc] * ratio)
    # convergencia
    conv = []
    for k in (1, 10, 100, 500, 2000, 4000):
        conv.append((k, e5.coax_capacitance_numeric(n=n, iterations=k)))
    tab = _table([r"\text{iteraciones}", r"C'\ (\mathrm{pF/m})", r"\text{error}"],
                 [[str(k), fx(c * 1e12, 3), f"{abs(c - c_an) / c_an * 100:.2f}\\,\\%"] for k, c in conv])
    return dict(
        explanation=[
            "Un cable coaxial es un capacitor cilíndrico: el conductor interno lleva una carga por unidad de longitud $+\\lambda$ y el externo $-\\lambda$, de modo que todo el campo queda confinado entre ambos. La capacitancia $C=Q/\\Delta V$ depende solo de la geometría (radios y longitud) y del medio, no de la carga ni del potencial aplicado.",
            "En la región entre conductores no hay cargas libres, por lo que el potencial cumple la ecuación de Laplace $\\nabla^2 V=0$. Una consecuencia es que el valor del potencial en un punto es el promedio de los valores a su alrededor; esa propiedad de promedio es la que permite resolver el problema con una malla.",
            "La energía almacenada se puede escribir de dos maneras equivalentes, $U=\\tfrac12 C\\,\\Delta V^2$ y $U=\\tfrac12\\varepsilon_0\\int E^2\\,dA$ (por unidad de longitud). Igualarlas permite obtener $C'$ a partir del campo calculado en la malla, sin necesidad de conocer la carga.",
            "Como la capacitancia por unidad de longitud depende del logaritmo de $b/a$, duplicar ambos radios no cambia $C'$ y hacer más delgado el conductor interno la reduce muy lentamente. Para $a=0.5$ cm y $b=2$ cm resulta del orden de 40 pF por metro, valor típico de un cable coaxial con aire.",
        ],
        analytic_steps=[
            ("Campo entre los conductores", "Se aplica la ley de Gauss con una superficie cilíndrica coaxial de radio $r$ y longitud $\\ell$ entre $a$ y $b$. Por simetría $E$ es radial y constante sobre la superficie.",
             [r"E(2\pi r\ell)=\dfrac{\lambda\ell}{\varepsilon_0}\quad\Longrightarrow\quad E(r)=\dfrac{\lambda}{2\pi\varepsilon_0 r}"]),
            ("Diferencia de potencial", "Se integra el campo desde el conductor interno hasta el externo.",
             [r"\Delta V=\int_a^b E\,dr=\dfrac{\lambda}{2\pi\varepsilon_0}\ln\dfrac{b}{a}"]),
            ("Capacitancia por unidad de longitud", "Se divide la carga por unidad de longitud entre la diferencia de potencial; $\\lambda$ se cancela.",
             [r"C'=\dfrac{\lambda}{\Delta V}=\dfrac{2\pi\varepsilon_0}{\ln(b/a)}"]),
            ("Sustituir los números", f"Con $a=0.5$ cm, $b=2$ cm y $\\varepsilon_0={sci(EPS0,5)}$ F/m, el cociente $b/a$ vale 4.",
             [rf"\ln\dfrac{{b}}{{a}}=\ln\dfrac{{{fx(b*100,1)}}}{{{fx(a*100,1)}}}=\ln 4={ratio:.5f}",
              rf"C'=\dfrac{{2\pi({sci(EPS0,5)})}}{{{ratio:.5f}}}={sci(c_an,5)}\ \mathrm{{F/m}}={c_an*1e12:.3f}\ \mathrm{{pF/m}}"]),
            ("Comprobar con la energía", f"Con $V_0=1$ V, la energía por unidad de longitud es $U'=\\tfrac12C'V_0^2$. Esta es la cantidad que el método numérico va a calcular integrando $E^2$.",
             [rf"U'=\tfrac12({sci(c_an,5)})(1)^2={sci(0.5*c_an,5)}\ \mathrm{{J/m}}"]),
        ],
        numeric_steps=[
            ("Construir la malla", f"Se cubre el cuadrado $[-1.02b,\\,1.02b]$ con $n={n}$ puntos por lado. Los nodos dentro del conductor interno ($r\\le a$) se fijan a $V_0=1$ V y los del exterior ($r\\ge b$) a 0; el resto son incógnitas.",
             [rf"h=\dfrac{{2(1.02)(0.02)}}{{{n-1}}}={sci(h,5)}\ \mathrm{{m}}",
              rf"\text{{nodos fijos interiores: }}{n_in},\quad \text{{exteriores: }}{n_out},\quad \text{{incógnitas: }}{n_free}"]),
            ("Discretizar Laplace", "Aproximando las segundas derivadas por diferencias centrales, $\\nabla^2V=0$ se convierte en que cada nodo libre es el promedio de sus cuatro vecinos.",
             [r"V_{i,j}=\dfrac{V_{i+1,j}+V_{i-1,j}+V_{i,j+1}+V_{i,j-1}}{4}"]),
            ("Sobrerrelajación y orden rojo-negro", f"En lugar de reemplazar $V_{{i,j}}$ por el promedio, se avanza un factor $\\omega>1$ más allá. Los nodos se colorean como tablero de ajedrez (rojo si $i+j$ es par): se actualizan primero todos los rojos y luego los negros, porque cada color solo depende del otro.",
             [r"V_{i,j}^{\text{nuevo}}=V_{i,j}+\omega\left(\bar V_{i,j}-V_{i,j}\right),\qquad \omega=\dfrac{2}{1+\sin(\pi/n)}",
              rf"\omega=\dfrac{{2}}{{1+\sin(\pi/{n})}}={omega:.5f}"]),
            ("Primera actualización a mano", f"Se toma el primer nodo rojo libre junto al conductor interno (sobre el eje, índice $i={ii}$). Al inicio todos los nodos libres valen 0 y solo su vecino interior vale 1 V.",
             [rf"\bar V=\dfrac{{1+0+0+0}}{{4}}={avg0:.4f}",
              rf"V^{{\text{{nuevo}}}}=0+{omega:.5f}\,({avg0:.4f}-0)={v_new:.5f}\ \mathrm{{V}}"]),
            ("Convergencia con las iteraciones", "Cada iteración es un barrido rojo y uno negro. A medida que la información de la frontera se propaga hacia la capa exterior, la capacitancia calculada se aproxima a un valor estable.",
             [tab]),
            ("Calcular la energía del campo", f"Con la solución final se obtiene $\\vec E=-\\nabla V$ con diferencias centrales (np.gradient), por ejemplo en $i={ig}$ sobre el eje; luego se suma $E^2$ sobre los nodos libres y se multiplica por el área de celda $h^2$.",
             [rf"E_x\approx-\dfrac{{V_{{i+1}}-V_{{i-1}}}}{{2h}}={fx(-dvx,3)}\ \mathrm{{V/m}}\quad(\text{{exacto }}{fx(e_exact,3)}\ \mathrm{{V/m}})",
              rf"\sum E^2={sci(s_e2,5)}\ \mathrm{{V^2/m^2}},\qquad U'=\tfrac12\varepsilon_0\sum E^2h^2={sci(u,5)}\ \mathrm{{J/m}}"]),
            ("Resultado y error", f"De $U'=\\tfrac12C'V_0^2$ se despeja $C'=2U'/V_0^2$ y se compara con el valor analítico.",
             [rf"C'_{{num}}=2({sci(u,5)})/1^2={c_nu*1e12:.3f}\ \mathrm{{pF/m}},\qquad C'_{{an}}={c_an*1e12:.3f}\ \mathrm{{pF/m}}",
              rf"\text{{error relativo}}=\dfrac{{|{c_nu*1e12:.3f}-{c_an*1e12:.3f}|}}{{{c_an*1e12:.3f}}}={abs(c_nu-c_an)/c_an*100:.2f}\,\%"]),
        ],
    )


# ----------------------------------------------------------------------------------------- CapacitorNetwork
def _network() -> dict:
    c1, c2, c3, v = e5.C1, e5.C2, e5.C3, e5.V_SRC
    cpar = c2 + c3
    ceq = e5.network_ceq_analytic()
    q = ceq * v
    vx = q / cpar
    v1 = q / c1
    u = 0.5 * ceq * v**2
    nd = e5.network_nodal_numeric()
    d = e5.dielectric_energies()
    k, c0 = 4.0, 2e-6
    q0 = c0 * v
    mu = lambda x: x * 1e6  # noqa: E731
    return dict(
        explanation=[
            "Un capacitor almacena carga y energía en el campo eléctrico entre sus placas. Cuando varios capacitores se conectan, lo que decide la combinación es qué magnitud comparten: en paralelo comparten la diferencia de potencial y las cargas se suman; en serie comparten la misma carga y los potenciales se suman.",
            "Por eso las capacitancias en paralelo se suman, $C_{par}=\\sum C_i$, mientras que en serie se suman sus inversas. Una red mixta se reduce paso a paso, empezando por los bloques más internos, hasta obtener un único $C_{eq}$ que determina la carga total y la energía suministrada por la fuente.",
            "Un dieléctrico es un material aislante cuyas moléculas se polarizan y debilitan el campo dentro del capacitor. A igual carga, el potencial cae por un factor $\\kappa$, de modo que la capacitancia aumenta a $C=\\kappa C_0$.",
            "Lo que ocurre con la energía depende de qué se mantiene fijo. Con la batería conectada el potencial es constante, la carga crece y la energía $\\tfrac12CV^2$ aumenta $\\kappa$ veces (la batería aporta el exceso). Con el capacitor aislado la carga es constante y la energía $Q^2/2C$ disminuye $\\kappa$ veces: el dieléctrico es atraído hacia dentro y realiza trabajo.",
        ],
        analytic_steps=[
            ("Reducir el bloque en paralelo", f"Los capacitores de 3 y 6 $\\mu$F están conectados entre los mismos dos nodos, así que comparten el potencial y sus capacitancias se suman.",
             [rf"C_{{23}}=C_2+C_3={mu(c2):.0f}+{mu(c3):.0f}={mu(cpar):.0f}\ \mu\mathrm{{F}}"]),
            ("Combinar en serie con el primero", "El capacitor de 2 $\\mu$F queda en serie con el bloque de 9 $\\mu$F; se suman los inversos.",
             [r"\dfrac{1}{C_{eq}}=\dfrac{1}{C_1}+\dfrac{1}{C_{23}}\ \Rightarrow\ C_{eq}=\dfrac{C_1C_{23}}{C_1+C_{23}}",
              rf"C_{{eq}}=\dfrac{{({mu(c1):.0f})({mu(cpar):.0f})}}{{{mu(c1):.0f}+{mu(cpar):.0f}}}=\dfrac{{{mu(c1)*mu(cpar):.0f}}}{{{mu(c1)+mu(cpar):.0f}}}={mu(ceq):.4f}\ \mu\mathrm{{F}}"]),
            ("Carga y energía totales", "La carga de la combinación en serie es la misma que la del capacitor $C_1$ y la del bloque; la energía es la de un solo capacitor equivalente.",
             [rf"Q=C_{{eq}}V=({mu(ceq):.4f}\ \mu\mathrm{{F}})({v:.0f}\ \mathrm{{V}})={mu(q):.3f}\ \mu\mathrm{{C}}",
              rf"U=\tfrac12C_{{eq}}V^2=\tfrac12({mu(ceq):.4f}\times10^{{-6}})({v:.0f})^2={mu(u):.2f}\ \mu\mathrm{{J}}"]),
            ("Potencial del nodo intermedio", "Con la carga conocida se reparten los potenciales: el capacitor $C_1$ toma $Q/C_1$ y el bloque paralelo el resto.",
             [rf"V_1=\dfrac{{Q}}{{C_1}}=\dfrac{{{mu(q):.3f}}}{{{mu(c1):.0f}}}={v1:.4f}\ \mathrm{{V}},\qquad V_x=\dfrac{{Q}}{{C_{{23}}}}=\dfrac{{{mu(q):.3f}}}{{{mu(cpar):.0f}}}={vx:.4f}\ \mathrm{{V}}",
              rf"V_1+V_x={v1+vx:.4f}\ \mathrm{{V}}={v:.0f}\ \mathrm{{V}}\ \checkmark"]),
            ("Dieléctrico con la batería conectada", f"El potencial se mantiene en 12 V, así que la capacitancia pasa de $C_0=2\\ \\mu$F a $\\kappa C_0$ y la energía se multiplica por $\\kappa=4$.",
             [rf"U_0=\tfrac12C_0V^2=\tfrac12({mu(c0):.0f}\times10^{{-6}})({v:.0f})^2={mu(d['u0']):.1f}\ \mu\mathrm{{J}}",
              rf"U_V=\tfrac12(\kappa C_0)V^2=\kappa U_0=4({mu(d['u0']):.1f})={mu(d['u_const_v']):.1f}\ \mu\mathrm{{J}}"]),
            ("Dieléctrico con el capacitor aislado", f"Ahora la carga $Q_0=C_0V={mu(q0):.0f}\\ \\mu$C no cambia; la capacitancia aumenta y el potencial y la energía disminuyen.",
             [rf"U_Q=\dfrac{{Q_0^2}}{{2\kappa C_0}}=\dfrac{{({mu(q0):.0f}\times10^{{-6}})^2}}{{2(4)({mu(c0):.0f}\times10^{{-6}})}}={mu(d['u_const_q']):.1f}\ \mu\mathrm{{J}}=\dfrac{{U_0}}{{\kappa}}"]),
        ],
        numeric_steps=[
            ("Plantear el nodo como incógnita", "En lugar de reducir la red, se toma como incógnita el potencial $V_x$ del nodo entre $C_1$ y el bloque paralelo. La fuente fija 12 V en un extremo y 0 V en el otro.",
             [rf"V_{{fuente}}={v:.0f}\ \mathrm{{V}},\qquad V_{{tierra}}=0"]),
            ("Conservación de la carga en el nodo", "La placa del nodo es aislada, por lo que la carga que llega por $C_1$ debe ser igual a la que se reparte en $C_2$ y $C_3$.",
             [r"C_1(V-V_x)=C_2V_x+C_3V_x\ \Rightarrow\ (C_1+C_2+C_3)\,V_x=C_1V"]),
            ("Armar el sistema lineal", "Con una sola incógnita es una matriz $1\\times1$ (numpy.linalg.solve la resuelve igual que lo haría con una red de muchos nodos, donde la matriz tiene un renglón por nodo).",
             [rf"[\,{mu(c1+c2+c3):.0f}\times10^{{-6}}\,]\,[V_x]=[\,{mu(c1*v):.0f}\times10^{{-6}}\,]"]),
            ("Resolver", "La solución es el cociente de los dos números anteriores.",
             [rf"V_x=\dfrac{{{mu(c1*v):.0f}}}{{{mu(c1+c2+c3):.0f}}}={nd['vx']:.5f}\ \mathrm{{V}}\quad(\text{{analítico }}24/11={24/11:.5f})"]),
            ("Cargas, capacitancia y energía", "Con $V_x$ se calculan las cargas de cada capacitor y a partir de ellas $C_{eq}=Q/V$ y $U=\\tfrac12QV$.",
             [rf"Q=C_1(V-V_x)=({mu(c1):.0f})({v-nd['vx']:.5f})={mu(nd['q']):.4f}\ \mu\mathrm{{C}}",
              rf"Q_2=C_2V_x={mu(nd['q2']):.4f}\ \mu\mathrm{{C}},\quad Q_3=C_3V_x={mu(nd['q3']):.4f}\ \mu\mathrm{{C}},\quad Q_2+Q_3={mu(nd['q2']+nd['q3']):.4f}\ \mu\mathrm{{C}}=Q",
              rf"C_{{eq}}=Q/V={mu(nd['c_eq']):.5f}\ \mu\mathrm{{F}},\qquad U=\tfrac12QV={mu(nd['energy']):.3f}\ \mu\mathrm{{J}}"]),
            ("Dieléctrico: dos casos límite", "No hay iteración: se aplican las dos reglas de actualización según qué magnitud se mantiene constante.",
             [rf"V\ \text{{cte.}}:\ U=\kappa U_0=4({mu(d['u0']):.1f})={mu(d['u_const_v']):.1f}\ \mu\mathrm{{J}}\qquad Q\ \text{{cte.}}:\ U=U_0/\kappa={mu(d['u_const_q']):.1f}\ \mu\mathrm{{J}}"]),
            ("Error frente al resultado analítico", "Como el sistema es lineal y se resuelve directamente, la diferencia es solo error de redondeo de la máquina.",
             [rf"|C_{{eq}}^{{num}}-C_{{eq}}^{{an}}|={sci(abs(nd['c_eq']-ceq),2)}\ \mathrm{{F}}"]),
        ],
    )


# ------------------------------------------------------------------------------------------ DrudeElectrons
def _drude() -> dict:
    tau, n, e = e6.TAU, e6.N_CU, e6.E_FIELD
    vd = e6.drift_speed_analytic()
    sig = e6.conductivity_analytic()
    mu_a = e6.drift_mobility_analytic()
    mu_n, sg, series = e6.drift_mobility_monte_carlo()
    rho = 1 / sig
    j = n * E_CH * vd
    vth = np.sqrt(K_B * e6.T_K / M_E)
    es = e6.E_SIM
    dt = tau / 50
    a = -E_CH * es / M_E
    # replica reducida: 5 electrones, 2 pasos
    rng = np.random.default_rng(9)
    ne = 5
    v = rng.normal(0.0, vth, ne)
    v0 = v.copy()
    v = v + a * dt
    v1a = v.copy()
    hit = rng.uniform(size=ne) < dt / tau
    v[hit] = rng.normal(0.0, vth, hit.sum())
    v1 = v.copy()
    hits1 = hit.copy()
    s0 = lambda x: sci(x, 3)  # noqa: E731
    rows = [[str(i + 1), s0(v0[i]), s0(v1a[i]), "sí" if hits1[i] else "no", s0(v1[i])] for i in range(ne)]
    tab = _table([r"e^-", r"v_0\ (\mathrm{m/s})", r"v_0+a\,\Delta t", r"\text{colisión}", r"v_1\ (\mathrm{m/s})"], rows)
    p = dt / tau
    steady_discrete = a * dt * (1 - p) / p
    tail = float(-series[len(series) // 2:].mean())
    return dict(
        explanation=[
            "En el modelo de Drude los electrones de conducción se tratan como partículas libres que aceleran bajo el campo eléctrico y de vez en cuando chocan con los iones de la red. Cada choque borra la memoria de la velocidad previa: tras él, el electrón sale en una dirección al azar con velocidad térmica.",
            "Entre choques consecutivos el electrón gana, en promedio, una velocidad $a\\tau=-eE\\tau/m$, donde $\\tau$ es el tiempo medio entre colisiones. Esa velocidad media ordenada es la velocidad de deriva $v_d$, y es muy pequeña frente a la velocidad térmica aleatoria (cerca de $10^5$ m/s a 300 K).",
            "La densidad de corriente $J=nev_d$ resulta proporcional a $E$, lo que reproduce la ley de Ohm con una conductividad $\\sigma=ne^2\\tau/m$. La movilidad $\\mu=v_d/E=e\\tau/m$ resume cuánto se desplaza un electrón por cada V/m de campo.",
            "El modelo es una descripción clásica y por eso falla en detalles (el calor específico electrónico, la dependencia exacta con la temperatura), pero acierta en el orden de magnitud: en el cobre da una deriva de apenas unos milímetros por segundo para un campo de 100 V/m, pese a que la señal eléctrica se propaga casi a la velocidad de la luz.",
        ],
        analytic_steps=[
            ("Planteamiento de la aceleración media", "Entre colisiones el electrón experimenta una fuerza constante $-eE$. Su velocidad crece linealmente con el tiempo durante un intervalo medio $\\tau$ y luego se reinicia.",
             [r"a=-\dfrac{eE}{m},\qquad \langle \vec v\rangle=\vec a\,\tau"]),
            ("Velocidad de deriva", f"Se toma el módulo y se sustituye $e={sci(E_CH,5)}$ C, $m={sci(M_E,5)}$ kg, $E=100$ V/m y $\\tau={sci(tau,2)}$ s.",
             [r"v_d=\dfrac{eE\tau}{m}",
              rf"v_d=\dfrac{{({sci(E_CH,5)})(100)({sci(tau,2)})}}{{{sci(M_E,5)}}}={vd:.4f}\ \mathrm{{m/s}}"]),
            ("Movilidad", "Dividiendo la deriva entre el campo se obtiene una cantidad independiente de $E$.",
             [rf"\mu=\dfrac{{v_d}}{{E}}=\dfrac{{e\tau}}{{m}}=\dfrac{{({sci(E_CH,5)})({sci(tau,2)})}}{{{sci(M_E,5)}}}={sci(mu_a,4)}\ \mathrm{{m^2/(V\,s)}}"]),
            ("Densidad de corriente", f"Con $n={sci(n,2)}$ m$^{{-3}}$ electrones por metro cúbico, cada uno con velocidad media $v_d$:",
             [rf"J=nev_d=({sci(n,2)})({sci(E_CH,5)})({vd:.4f})={sci(j,4)}\ \mathrm{{A/m^2}}"]),
            ("Conductividad", "De $J=\\sigma E$ se despeja $\\sigma$; al sustituir $J=nev_d$ queda una fórmula que depende solo de $n$, $\\tau$ y constantes.",
             [r"\sigma=\dfrac{J}{E}=\dfrac{ne^2\tau}{m}",
              rf"\sigma=\dfrac{{{sci(j,4)}}}{{100}}={sci(sig,4)}\ \mathrm{{S/m}},\qquad \rho=\dfrac{{1}}{{\sigma}}={sci(rho,3)}\ \Omega\cdot\mathrm{{m}}"]),
        ],
        numeric_steps=[
            ("Por qué un campo grande", f"A 100 V/m la deriva ({vd:.3f} m/s) es minúscula frente a la velocidad térmica ($\\sqrt{{k_BT/m}}={sci(vth,3)}$ m/s), así que el ruido estadístico dominaría. Como el modelo es lineal en $E$, se simula con $E={sci(es,2)}$ V/m y se compara la movilidad $\\mu=v_d/E$, que no depende del campo.",
             [rf"\sqrt{{\dfrac{{k_BT}}{{m}}}}=\sqrt{{\dfrac{{({sci(K_B,5)})(300)}}{{{sci(M_E,5)}}}}}={sci(vth,4)}\ \mathrm{{m/s}}"]),
            ("Parámetros del Monte Carlo", f"Se simulan 20 000 electrones durante 4000 pasos con semilla 9. El paso es $\\Delta t=\\tau/50$ y la aceleración es la misma para todos.",
             [rf"\Delta t=\dfrac{{\tau}}{{50}}={sci(dt,3)}\ \mathrm{{s}},\qquad a=-\dfrac{{eE}}{{m}}={sci(a,4)}\ \mathrm{{m/s^2}},\qquad a\,\Delta t={sci(a*dt,4)}\ \mathrm{{m/s}}"]),
            ("Regla de actualización por paso", "En cada paso: (1) se suma $a\\,\\Delta t$ a todos; (2) cada electrón sortea un número uniforme $u\\in[0,1)$ y, si $u<\\Delta t/\\tau=0.02$, choca y su velocidad se reemplaza por una muestra de una normal de media 0 y desviación $v_{th}$; (3) se guarda el promedio de $v$.",
             [r"v\leftarrow v+a\Delta t;\qquad \text{si } u<\Delta t/\tau:\ v\leftarrow \mathcal N(0,v_{th}^2);\qquad \bar v_k=\dfrac{1}{N}\sum v"]),
            ("Dos pasos a mano con 5 electrones", "Con la misma semilla se muestran las velocidades iniciales (normales), el efecto de la aceleración y qué electrones chocaron en el primer paso. Con probabilidad 0.02 por paso, lo esperado es que casi ninguno choque.",
             [tab]),
            ("Cómo se acumula la media", f"La media obedece $\\bar v_{{k+1}}=(\\bar v_k+a\\Delta t)(1-p)$ con $p=0.02$, porque en promedio una fracción $p$ de los electrones pierde su velocidad. El punto fijo de esa recurrencia es la deriva simulada.",
             [rf"\bar v_\infty=\dfrac{{a\Delta t\,(1-p)}}{{p}}={sci(steady_discrete,4)}\ \mathrm{{m/s}}\ \approx\ 0.98\,a\tau,\qquad a\tau={sci(a*tau,4)}\ \mathrm{{m/s}}"]),
            ("Promedio de la segunda mitad", "El régimen transitorio dura unos pocos $\\tau$ (50 pasos cada uno). Se descartan los primeros 2000 pasos y se promedia $-\\bar v_k/E$ sobre los 2000 restantes (80 $\\tau$ en total).",
             [rf"\mu_{{num}}=\Big\langle-\dfrac{{\bar v_k}}{{E}}\Big\rangle_{{k>2000}}={sci(tail,4)}\ \mathrm{{m^2/(V\,s)}}"]),
            ("Resultado, incertidumbre y error", f"La incertidumbre estadística estimada es $\\sigma_\\mu={sci(sg,2)}$. Con la movilidad numérica la deriva a 100 V/m se obtiene como $\\mu E$.",
             [rf"\mu_{{num}}={mu_n*1e3:.4f}\times10^{{-3}}\ \mathrm{{m^2/(V\,s)}},\qquad \mu_{{an}}={mu_a*1e3:.4f}\times10^{{-3}}",
              rf"v_d^{{num}}=\mu_{{num}}(100)={mu_n*100:.4f}\ \mathrm{{m/s}}\quad(\text{{error }}{abs(mu_n-mu_a)/mu_a*100:.2f}\,\%)"]),
        ],
    )


# -------------------------------------------------------------------------------------------- TaperedWire
def _tapered() -> dict:
    rho, L, r1, r2 = e6.RHO, e6.LENGTH, e6.R1, e6.R2
    ra, rn, ru = e6.tapered_resistance_analytic(), e6.tapered_resistance_numeric(), e6.uniform_resistance()
    slope = (r2 - r1) / L
    f = lambda x: rho / (np.pi * (r1 + slope * x) ** 2)  # noqa: E731
    xs4 = np.linspace(0, L, 5)
    f4 = [f(x) for x in xs4]
    h4 = L / 4
    s4 = h4 / 3 * (f4[0] + 4 * f4[1] + 2 * f4[2] + 4 * f4[3] + f4[4])
    n = 400
    h = L / n
    x = np.linspace(0, L, n + 1)
    fv = rho / (np.pi * (r1 + slope * x) ** 2)
    odd, even = fv[1:-1:2].sum(), fv[2:-1:2].sum()
    rows = [[str(i), fx(xs4[i], 2), fx((r1 + slope * xs4[i]) * 1e3, 2), sci(f4[i], 4), str([1, 4, 2, 4, 1][i])] for i in range(5)]
    tab = _table([r"i", r"x\ (\mathrm{m})", r"r\ (\mathrm{mm})", r"\rho/(\pi r^2)\ (\Omega/\mathrm{m})", r"\text{peso}"], rows)
    r_mid = r1 + slope * L / 2
    return dict(
        explanation=[
            "La resistencia de un conductor mide cuánto se opone al paso de corriente para una diferencia de potencial dada. En un cable uniforme vale $R=\\rho L/A$: crece con la longitud y con la resistividad del material, y disminuye al aumentar la sección.",
            "Si la sección cambia a lo largo del cable, esa fórmula ya no vale directamente, pero sí vale para cada rebanada delgada de espesor $dx$. Como la corriente es la misma en todas las rebanadas (conservación de la carga), las resistencias de las rebanadas están en serie y se suman: $R=\\int \\rho\\,dx/A(x)$.",
            "En un cable cónico el tramo delgado domina: como el área varía con el cuadrado del radio, el extremo de 0.5 mm aporta mucho más que el de 1.5 mm. Por eso la resistencia total no es la del radio medio, sino bastante mayor.",
            "Este mismo procedimiento sirve para cualquier perfil $A(x)$: basta integrar. Cuando la integral no tiene forma cerrada se recurre a un método numérico como la regla de Simpson.",
        ],
        analytic_steps=[
            ("Describir el radio", "El radio crece linealmente desde $r_1$ en $x=0$ hasta $r_2$ en $x=L$.",
             [rf"r(x)=r_1+\dfrac{{r_2-r_1}}{{L}}x=0.5\ \mathrm{{mm}}+({fx(slope,4)}\ \mathrm{{m/m}})\,x\quad\Longrightarrow\quad r(x)={sci(r1,2)}+{sci(slope,3)}\,x\ \ (\mathrm{{m}})"]),
            ("Resistencia de una rebanada", "La rebanada de espesor $dx$ tiene área $\\pi r(x)^2$ y su resistencia es la de un cable corto uniforme.",
             [r"dR=\dfrac{\rho\,dx}{\pi r(x)^2}"]),
            ("Integrar a lo largo del cable", "Se cambia la variable a $r$ con $dx=\\dfrac{L}{r_2-r_1}dr$, y se integra $1/r^2$.",
             [r"R=\dfrac{\rho L}{\pi(r_2-r_1)}\int_{r_1}^{r_2}\dfrac{dr}{r^2}=\dfrac{\rho L}{\pi(r_2-r_1)}\left(\dfrac1{r_1}-\dfrac1{r_2}\right)=\dfrac{\rho L}{\pi r_1 r_2}"]),
            ("Sustituir los números", f"Con $\\rho={sci(rho,3)}\\ \\Omega\\cdot$m, $L=2$ m, $r_1=0.5$ mm y $r_2=1.5$ mm:",
             [rf"R=\dfrac{{({sci(rho,3)})(2)}}{{\pi({sci(r1,2)})({sci(r2,2)})}}=\dfrac{{{sci(rho*L,4)}}}{{{sci(np.pi*r1*r2,4)}}}={sci(ra,5)}\ \Omega={ra*1e3:.3f}\ \mathrm{{m\Omega}}"]),
            ("Cable uniforme de comparación", "Para el cable de radio 0.5 mm en toda su longitud se aplica $R=\\rho L/(\\pi r_1^2)$.",
             [rf"R_u=\dfrac{{({sci(rho,3)})(2)}}{{\pi({sci(r1,2)})^2}}={sci(ru,5)}\ \Omega={ru*1e3:.3f}\ \mathrm{{m\Omega}}"]),
            ("Interpretar el cociente", "La relación entre ambas resistencias depende solo de los radios; confirma que el tramo delgado domina.",
             [rf"\dfrac{{R}}{{R_u}}=\dfrac{{r_1}}{{r_2}}={ra/ru:.4f}\quad\Longrightarrow\quad R=\dfrac13R_u;\qquad \text{{con radio medio }}{r_mid*1e3:.1f}\ \mathrm{{mm}}:\ \dfrac{{\rho L}}{{\pi r_{{med}}^2}}={rho*L/(np.pi*r_mid**2)*1e3:.3f}\ \mathrm{{m\Omega}}"]),
        ],
        numeric_steps=[
            ("Integrando", "Se integra numéricamente $dR/dx=\\rho/(\\pi r(x)^2)$ entre 0 y $L$ con la regla de Simpson compuesta, que ajusta una parábola a cada par de subintervalos.",
             [r"R\approx\dfrac{h}{3}\left[f_0+4f_1+2f_2+4f_3+\cdots+4f_{N-1}+f_N\right],\qquad f_i=\dfrac{\rho}{\pi r(x_i)^2}"]),
            ("Versión reducida a mano (N = 4)", f"Con $N=4$ el paso es $h=L/4={h4:.2f}$ m. Se evalúa el integrando en cinco puntos y se asignan los pesos 1, 4, 2, 4, 1.",
             [tab]),
            ("Suma de Simpson con N = 4", "Se multiplican los valores por sus pesos y se suman.",
             [rf"R_4=\dfrac{{{h4:.2f}}}{{3}}\left[{sci(f4[0],4)}+4({sci(f4[1],4)})+2({sci(f4[2],4)})+4({sci(f4[3],4)})+{sci(f4[4],4)}\right]={sci(s4,5)}\ \Omega={s4*1e3:.3f}\ \mathrm{{m\Omega}}",
              rf"\text{{error con }}N=4:\ \dfrac{{|{s4*1e3:.3f}-{ra*1e3:.3f}|}}{{{ra*1e3:.3f}}}={abs(s4-ra)/ra*100:.3f}\,\%"]),
            ("Parámetros reales", f"El código usa $N={n}$ subintervalos: $h=L/N={sci(h,3)}$ m. Los puntos $x_i=ih$ llevan radios $r_i$; los índices impares pesan 4 y los pares interiores 2.",
             [rf"f_0=\dfrac{{\rho}}{{\pi r_1^2}}={sci(fv[0],5)}\ \Omega/\mathrm{{m}},\qquad f_N=\dfrac{{\rho}}{{\pi r_2^2}}={sci(fv[-1],5)}\ \Omega/\mathrm{{m}}"]),
            ("Sumas de índices impares y pares", "Se acumulan por separado los 200 términos impares y los 199 pares interiores.",
             [rf"\sum_{{\text{{impar}}}}f_i={sci(odd,6)},\qquad \sum_{{\text{{par int.}}}}f_i={sci(even,6)}"]),
            ("Ensamblar la regla", "Se combinan los extremos y las dos sumas con sus pesos.",
             [rf"R_{{400}}=\dfrac{{{sci(h,3)}}}{{3}}\left[{sci(fv[0],5)}+{sci(fv[-1],5)}+4({sci(odd,6)})+2({sci(even,6)})\right]={sci(rn,7)}\ \Omega"]),
            ("Resultado y error", "El error de Simpson decrece como $h^4$; con 400 subintervalos queda en el límite de la precisión de la máquina para este integrando suave.",
             [rf"R_{{num}}={rn*1e3:.6f}\ \mathrm{{m\Omega}},\qquad R_{{an}}={ra*1e3:.6f}\ \mathrm{{m\Omega}},\qquad \dfrac{{|R_{{num}}-R_{{an}}|}}{{R_{{an}}}}={sci(abs(rn-ra)/ra,2)}"]),
        ],
    )


# --------------------------------------------------------------------------------------------- MeshCircuit
def _mesh() -> dict:
    e1, e2, r1, r2, r3 = e7.E1, e7.E2, e7.R1, e7.R2, e7.R3
    ia = e7.mesh_currents_analytic()
    inum = e7.mesh_currents_numeric()
    br = e7.branch_currents(ia)
    res = e7.kirchhoff_residuals(inum)
    det = (r1 + r3) * (r2 + r3) - r3**2
    a11, a12, a22 = r1 + r3, -r3, r2 + r3
    m = a12 / a11  # multiplicador de eliminacion
    a22p = a22 - m * a12
    b2p = -e2 - m * e1
    i2 = b2p / a22p
    i1 = (e1 - a12 * i2) / a11
    ps = e1 * ia[0] - e2 * ia[1]
    pr = r1 * br["R1"] ** 2 + r2 * br["R2"] ** 2 + r3 * br["R3"] ** 2
    return dict(
        explanation=[
            "Las leyes de Kirchhoff expresan dos conservaciones. La de los nodos dice que la carga no se acumula en una unión: lo que entra de corriente es igual a lo que sale. La de las mallas dice que el potencial es una función de punto: al recorrer un lazo cerrado y volver al inicio, la suma de subidas (fuentes) y caídas ($IR$) es cero.",
            "En el método de corrientes de malla se asigna una corriente circulante a cada lazo independiente. La ley de los nodos queda satisfecha automáticamente, y solo hay que escribir la ley de las mallas, una ecuación por lazo. La resistencia compartida lleva la diferencia de las dos corrientes de malla.",
            "El signo importa: si al resolver una corriente sale negativa, solo significa que circula en sentido contrario al supuesto. En este circuito la segunda fuente se opone al sentido elegido, por eso aparece con signo negativo en la ecuación de la segunda malla.",
            "Como comprobación, la potencia entregada por las fuentes debe ser igual a la disipada en las resistencias. Esta igualdad es una forma de conservación de la energía y detecta errores de signo o de aritmética.",
        ],
        analytic_steps=[
            ("Elegir las corrientes de malla", "Se supone que $i_1$ circula por la malla izquierda ($\\varepsilon_1$, $R_1$, $R_3$) y $i_2$ por la derecha ($\\varepsilon_2$, $R_2$, $R_3$), ambas en sentido horario. En $R_3$ pasa $i_1-i_2$.",
             [rf"\varepsilon_1={e1:.0f}\ \mathrm{{V}},\ \varepsilon_2={e2:.0f}\ \mathrm{{V}},\ R_1={r1:.0f}\ \Omega,\ R_2={r2:.0f}\ \Omega,\ R_3={r3:.0f}\ \Omega"]),
            ("Ley de las mallas en la izquierda", "Se recorre el lazo sumando la subida de la fuente y las caídas en $R_1$ y en $R_3$.",
             [r"\varepsilon_1-R_1i_1-R_3(i_1-i_2)=0\ \Rightarrow\ (R_1+R_3)i_1-R_3i_2=\varepsilon_1",
              rf"({r1:.0f}+{r3:.0f})i_1-{r3:.0f}\,i_2={e1:.0f}\ \Rightarrow\ {r1+r3:.0f}\,i_1-{r3:.0f}\,i_2={e1:.0f}"]),
            ("Ley de las mallas en la derecha", "La fuente $\\varepsilon_2$ está orientada contra el sentido horario, por lo que resta; $R_3$ se recorre ahora en sentido contrario al anterior.",
             [r"-\varepsilon_2-R_2i_2-R_3(i_2-i_1)=0\ \Rightarrow\ -R_3i_1+(R_2+R_3)i_2=-\varepsilon_2",
              rf"-{r3:.0f}\,i_1+({r2:.0f}+{r3:.0f})\,i_2=-{e2:.0f}\ \Rightarrow\ -{r3:.0f}\,i_1+{r2+r3:.0f}\,i_2=-{e2:.0f}"]),
            ("Resolver con determinantes (Cramer)", "El determinante de coeficientes es el mismo para ambas incógnitas.",
             [rf"\Delta=({r1+r3:.0f})({r2+r3:.0f})-({r3:.0f})^2={det:.0f}\ \Omega^2",
              rf"i_1=\dfrac{{\varepsilon_1(R_2+R_3)-R_3\varepsilon_2}}{{\Delta}}=\dfrac{{({e1:.0f})({r2+r3:.0f})-({r3:.0f})({e2:.0f})}}{{{det:.0f}}}=\dfrac{{{e1*(r2+r3)-r3*e2:.0f}}}{{{det:.0f}}}={ia[0]:.4f}\ \mathrm{{A}}",
              rf"i_2=\dfrac{{R_3\varepsilon_1-(R_1+R_3)\varepsilon_2}}{{\Delta}}=\dfrac{{({r3:.0f})({e1:.0f})-({r1+r3:.0f})({e2:.0f})}}{{{det:.0f}}}=\dfrac{{{r3*e1-(r1+r3)*e2:.0f}}}{{{det:.0f}}}={ia[1]:.4f}\ \mathrm{{A}}"]),
            ("Corrientes de rama", "Las ramas con $R_1$ y $R_2$ llevan la corriente de su malla; en la rama central circula la diferencia.",
             [rf"I_{{R_1}}=i_1={br['R1']:.4f}\ \mathrm{{A}},\quad I_{{R_2}}=i_2={br['R2']:.4f}\ \mathrm{{A}},\quad I_{{R_3}}=i_1-i_2={br['R3']:.4f}\ \mathrm{{A}}"]),
            ("Balance de potencia", "Las fuentes entregan $\\varepsilon_1 i_1$ y la segunda, contraria al sentido de $i_2$, aporta $-\\varepsilon_2i_2$; las resistencias disipan $I^2R$.",
             [rf"P_{{fuentes}}=({e1:.0f})({ia[0]:.4f})-({e2:.0f})({ia[1]:.4f})={ps:.4f}\ \mathrm{{W}}",
              rf"P_{{R}}={r1:.0f}({br['R1']:.4f})^2+{r2:.0f}({br['R2']:.4f})^2+{r3:.0f}({br['R3']:.4f})^2={pr:.4f}\ \mathrm{{W}}\ \Rightarrow\ \Delta P=0"]),
        ],
        numeric_steps=[
            ("Forma matricial", "Las dos ecuaciones de malla se escriben como $A\\,\\vec i=\\vec b$, que es lo que recibe numpy.linalg.solve.",
             [rf"\begin{{pmatrix}}{r1+r3:.0f}&-{r3:.0f}\\-{r3:.0f}&{r2+r3:.0f}\end{{pmatrix}}\begin{{pmatrix}}i_1\\i_2\end{{pmatrix}}=\begin{{pmatrix}}{e1:.0f}\\-{e2:.0f}\end{{pmatrix}}"]),
            ("Eliminación gaussiana: multiplicador", "Se usa el elemento mayor de la primera columna (10) como pivote; el multiplicador elimina el término $-6$ del segundo renglón.",
             [rf"m=\dfrac{{a_{{21}}}}{{a_{{11}}}}=\dfrac{{-{r3:.0f}}}{{{a11:.0f}}}={m:.2f}"]),
            ("Eliminación gaussiana: segundo renglón", "Se resta $m$ veces el primer renglón al segundo, tanto en la matriz como en el lado derecho.",
             [rf"a_{{22}}'={a22:.0f}-({m:.2f})({a12:.0f})={a22p:.2f},\qquad b_2'=-{e2:.0f}-({m:.2f})({e1:.0f})={b2p:.2f}"]),
            ("Sustitución hacia atrás", "Del renglón triangular se despeja primero $i_2$, y luego $i_1$ del primer renglón.",
             [rf"i_2=\dfrac{{b_2'}}{{a_{{22}}'}}=\dfrac{{{b2p:.2f}}}{{{a22p:.2f}}}={i2:.6f}\ \mathrm{{A}}",
              rf"i_1=\dfrac{{{e1:.0f}-({a12:.0f})({i2:.6f})}}{{{a11:.0f}}}={i1:.6f}\ \mathrm{{A}}"]),
            ("Resultado de numpy.linalg.solve", "La función realiza esta misma factorización LU con pivoteo parcial, con aritmética de doble precisión.",
             [rf"i_1={inum[0]:.6f}\ \mathrm{{A}},\quad i_2={inum[1]:.6f}\ \mathrm{{A}};\qquad |i^{{num}}-i^{{an}}|\le{sci(float(np.max(np.abs(inum-ia))),2)}\ \mathrm{{A}}"]),
            ("Residuos de Kirchhoff", "Se reintroducen las corrientes en las dos ecuaciones de lazo y en el balance de potencia; si todo está bien, deben dar cero salvo redondeo.",
             [rf"\sum\Delta V_{{izq}}={e1:.0f}-{r1:.0f}({br['R1']:.4f})-{r3:.0f}({br['R3']:.4f})={sci(res['loop_left'],2) if abs(res['loop_left'])>0 else '0'}\ \mathrm{{V}}",
              rf"\sum\Delta V_{{der}}={sci(res['loop_right'],2) if abs(res['loop_right'])>0 else '0'}\ \mathrm{{V}},\qquad \Delta P=P_{{fuentes}}-P_R={sci(res['power_balance'],2) if abs(res['power_balance'])>0 else '0'}\ \mathrm{{W}}"]),
        ],
    )


# --------------------------------------------------------------------------------------------- RCCharging
def _rc() -> dict:
    eps, R, C = e7.EPS, e7.R, e7.C
    tau = R * C
    qinf = C * eps
    i0 = eps / R
    en = e7.rc_energies_numeric()
    t_an = -tau * np.log(1 - 0.632)
    t_nu = e7.time_to_fraction_numeric()
    t, y = e7.rc_numeric()
    h = 1.2 / 2400
    f = lambda q: (eps - q / C) / R  # noqa: E731
    k1 = f(0.0)
    k2 = f(h / 2 * k1)
    k3 = f(h / 2 * k2)
    k4 = f(h * k3)
    q1 = h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    q1_exact = float(e7.q_analytic(h))
    tgt = 0.632 * qinf
    idx = int(np.argmax(y[:, 0] >= tgt))
    # energias
    h3 = 3.0 / 6000
    t3, y3 = e7.rc_numeric(3.0, 6000)
    q3 = y3[:, 0]
    i3 = (eps - q3 / C) / R
    rows = [[str(k), fx(t[k], 4), sci(y[k, 0], 5), sci(float(e7.q_analytic(t[k])), 5)] for k in (0, 1, 2, 3)]
    rows.append([str(2400), fx(t[2400], 4), sci(y[2400, 0], 5), sci(float(e7.q_analytic(t[2400])), 5)])
    tab = _table([r"k", r"t_k\ (\mathrm{s})", r"q_k^{RK4}\ (\mathrm{C})", r"q(t_k)\ (\mathrm{C})"], rows)
    return dict(
        explanation=[
            "Cuando se cierra el interruptor, la batería empuja carga hacia las placas del capacitor a través de la resistencia. Al principio el capacitor está descargado y toda la fem cae en $R$, por lo que la corriente es máxima, $\\varepsilon/R$; conforme el capacitor se carga su voltaje $q/C$ se opone a la batería y la corriente decae.",
            "La ley de las mallas da la ecuación diferencial $R\\,dq/dt=\\varepsilon-q/C$. Su solución es una exponencial que se acerca a la carga final $q_\\infty=C\\varepsilon$ con una escala de tiempo $\\tau=RC$: tras un tiempo $\\tau$ el capacitor alcanzó el 63.2 % de su carga y tras $5\\tau$ más del 99 %.",
            "El balance de energía tiene un resultado notable e independiente de $R$: la batería entrega $C\\varepsilon^2$, de los cuales la mitad se almacena en el capacitor y la otra mitad se disipa como calor en la resistencia. Una resistencia menor solo hace que el proceso sea más rápido, no más eficiente.",
            "Con $R=2$ k$\\Omega$ y $C=100\\ \\mu$F la constante de tiempo es 0.2 s: un fenómeno lo bastante lento para observarse con un voltímetro, y la base de temporizadores y filtros.",
        ],
        analytic_steps=[
            ("Plantear la ecuación del circuito", "Recorriendo el lazo: la fem sube el potencial, y el resistor y el capacitor lo bajan, con $I=dq/dt$.",
             [r"\varepsilon-IR-\dfrac{q}{C}=0\ \Rightarrow\ \dfrac{dq}{dt}=\dfrac{1}{R}\left(\varepsilon-\dfrac{q}{C}\right)"]),
            ("Separar variables e integrar", "Con $q(0)=0$ se agrupan $q$ y $t$ a lados distintos.",
             [r"\int_0^q\dfrac{dq'}{C\varepsilon-q'}=\dfrac{1}{RC}\int_0^t dt'\ \Rightarrow\ -\ln\!\left(1-\dfrac{q}{C\varepsilon}\right)=\dfrac{t}{RC}",
              r"q(t)=C\varepsilon\left(1-e^{-t/RC}\right),\qquad I(t)=\dfrac{\varepsilon}{R}e^{-t/RC}"]),
            ("Constante de tiempo y valores límite", f"Se sustituyen $R=2\\times10^3\\ \\Omega$, $C=10^{{-4}}$ F y $\\varepsilon=12$ V.",
             [rf"\tau=RC=({sci(R,2)})({sci(C,2)})={tau:.3f}\ \mathrm{{s}}",
              rf"q_\infty=C\varepsilon=({sci(C,2)})(12)={qinf*1e3:.2f}\ \mathrm{{mC}},\qquad I_0=\dfrac{{\varepsilon}}{{R}}=\dfrac{{12}}{{{sci(R,2)}}}={i0*1e3:.1f}\ \mathrm{{mA}}"]),
            ("Tiempo para el 63.2 %", "Se impone $q=0.632\\,q_\\infty$ en la solución y se despeja $t$.",
             [rf"t=-\tau\ln(1-0.632)=-({tau:.3f})\ln(0.368)={t_an:.5f}\ \mathrm{{s}}\ \ (\approx\tau)"]),
            ("Energía entregada por la batería", "La batería entrega una carga total $q_\\infty$ a potencial constante $\\varepsilon$.",
             [rf"E_{{bat}}=\varepsilon q_\infty=C\varepsilon^2=({sci(C,2)})(12)^2={C*eps**2*1e3:.2f}\ \mathrm{{mJ}}"]),
            ("Energía almacenada y disipada", "La energía del capacitor es $q_\\infty^2/2C$; la disipada se obtiene integrando $I^2R$.",
             [rf"E_{{cap}}=\dfrac{{q_\infty^2}}{{2C}}=\tfrac12C\varepsilon^2={0.5*C*eps**2*1e3:.2f}\ \mathrm{{mJ}}",
              r"E_{dis}=\int_0^\infty I^2R\,dt=\dfrac{\varepsilon^2}{R}\int_0^\infty e^{-2t/RC}dt=\dfrac{\varepsilon^2}{R}\cdot\dfrac{RC}{2}=\tfrac12C\varepsilon^2",
              rf"E_{{dis}}={0.5*C*eps**2*1e3:.2f}\ \mathrm{{mJ}}=E_{{bat}}-E_{{cap}}"]),
        ],
        numeric_steps=[
            ("Parámetros de Runge-Kutta", f"Se integra $dq/dt=f(q)=(\\varepsilon-q/C)/R$ con RK4, $q(0)=0$, sobre 1.2 s con 2400 pasos. Como $f$ no depende de $t$, solo se evalúa en $q$.",
             [rf"h=\dfrac{{1.2}}{{2400}}={sci(h,3)}\ \mathrm{{s}},\qquad f(q)=\dfrac{{12-q/({sci(C,2)})}}{{{sci(R,2)}}}"]),
            ("Las cuatro pendientes del primer paso", "Se evalúa la pendiente al inicio, dos veces en el punto medio (con dos estimaciones distintas de $q$) y al final.",
             [rf"k_1=f(0)={sci(k1,5)}\ \mathrm{{A}}",
              rf"k_2=f\!\left(q_0+\tfrac h2k_1\right)=f({sci(h/2*k1,5)})={sci(k2,5)}\ \mathrm{{A}}",
              rf"k_3=f\!\left(q_0+\tfrac h2k_2\right)=f({sci(h/2*k2,5)})={sci(k3,5)}\ \mathrm{{A}}",
              rf"k_4=f(q_0+hk_3)=f({sci(h*k3,5)})={sci(k4,5)}\ \mathrm{{A}}"]),
            ("Primer valor de q", "La nueva carga es la anterior más el promedio ponderado de pendientes por el paso.",
             [rf"q_1=q_0+\dfrac{{h}}{{6}}(k_1+2k_2+2k_3+k_4)={sci(q1,7)}\ \mathrm{{C}}",
              rf"q(h)=C\varepsilon(1-e^{{-h/\tau}})={sci(q1_exact,7)}\ \mathrm{{C}}\quad\Rightarrow\quad |q_1-q(h)|={sci(abs(q1-q1_exact),2)}\ \mathrm{{C}}"]),
            ("Seguir avanzando", "Se repite el mismo cálculo con $q_{k+1}=q_k+\\frac h6(\\cdots)$. La tabla compara RK4 con la solución exacta.",
             [tab]),
            ("Tiempo para el 63.2 % por interpolación", f"Se busca el primer índice con $q\\ge 0.632\\,q_\\infty={sci(tgt,5)}$ C (ese índice es $k={idx}$) y se interpola linealmente entre $k-1$ y $k$ (en la sección de 2 s, 4000 pasos).",
             [rf"t^*=t_{{k-1}}+\dfrac{{q^*-q_{{k-1}}}}{{q_k-q_{{k-1}}}}(t_k-t_{{k-1}})={t_nu:.5f}\ \mathrm{{s}}\quad(\text{{analítico }}{t_an:.5f}\ \mathrm{{s}})"]),
            ("Energías por Simpson", f"Con la solución hasta 3 s (15$\\tau$) en 6000 pasos, $h={sci(h3,3)}$ s, se obtiene la corriente $i_k=(\\varepsilon-q_k/C)/R$ y se integran con Simpson $\\int i^2R\\,dt$ y $\\int\\varepsilon i\\,dt$. Primer y último valor de la corriente:",
             [rf"i_0={sci(i3[0],5)}\ \mathrm{{A}},\qquad i_N={sci(i3[-1],3)}\ \mathrm{{A}}",
              rf"E_{{dis}}=\dfrac{{h}}{{3}}\left[\,g_0+4g_1+2g_2+\cdots+g_N\right],\ g_k=i_k^2R;\qquad g_0={sci(i3[0]**2*R,5)}\ \mathrm{{W}}"]),
            ("Resultados y errores", "La energía del capacitor se calcula directamente como $q_N^2/2C$ con la carga final de la integración.",
             [rf"E_{{dis}}={en['dissipated']*1e3:.5f},\quad E_{{cap}}={en['capacitor']*1e3:.5f},\quad E_{{bat}}={en['battery']*1e3:.5f}\ \mathrm{{mJ}}",
              rf"E_{{dis}}+E_{{cap}}={(en['dissipated']+en['capacitor'])*1e3:.5f}\ \mathrm{{mJ}}\approx E_{{bat}};\qquad \text{{error de }}E_{{dis}}:\ {sci(abs(en['dissipated']-0.5*C*eps**2)/(0.5*C*eps**2),2)}"]),
        ],
    )


# ---------------------------------------------------------------------------------------------- HallEffect
def _hall() -> dict:
    n, t_, w, i_, b = m8.N_CU, m8.THICK, m8.WIDTH, m8.I_HALL, m8.B_HALL
    tau = m8.TAU
    va = m8.hall_voltage_analytic()
    vn = m8.hall_voltage_numeric()
    vd = i_ / (n * E_CH * w * t_)
    eh = vd * b
    k = E_CH / M_E
    ex = M_E * vd / (E_CH * tau)
    lo, hi = -2 * vd * b, 2 * vd * b
    h = 40 * tau / 800
    # primer paso RK4 con la primera prueba (mid = 0)
    ey0 = 0.5 * (lo + hi)
    rhs = lambda tt, v: np.array([-k * (ex + v[1] * b) - v[0] / tau, -k * (ey0 - v[0] * b) - v[1] / tau])  # noqa: E731
    y0 = np.zeros(2)
    k1 = rhs(0, y0)
    k2 = rhs(0, y0 + h / 2 * k1)
    k3 = rhs(0, y0 + h / 2 * k2)
    k4 = rhs(0, y0 + h * k3)
    y1 = y0 + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    # biseccion, primeras iteraciones
    trace = []
    l2, h2 = lo, hi
    for it in range(80):
        mid = 0.5 * (l2 + h2)
        vy = m8._vy_steady(mid, ex, b)
        if it < 5 or it == 79:
            trace.append((it + 1, l2, h2, mid, vy))
        if vy > 0:
            l2 = mid
        else:
            h2 = mid
    ey = 0.5 * (l2 + h2)
    rows = [[str(it), sci(a_, 3), sci(b_, 3), sci(m_, 3), sci(v_, 3)] for it, a_, b_, m_, v_ in trace]
    tab = _table([r"\text{iter.}", r"E_{lo}", r"E_{hi}", r"E_y\ (\mathrm{V/m})", r"v_y^{est}\ (\mathrm{m/s})"], rows)
    vy_end = m8._vy_steady(ey, ex, b)
    n_back = m8.carrier_density_from_hall(vn)
    return dict(
        explanation=[
            "Cuando una corriente circula por una cinta conductora dentro de un campo magnético perpendicular a ella, la fuerza de Lorentz empuja a los portadores hacia un borde. Esa acumulación de carga en un lado crea un campo eléctrico transversal, el campo de Hall, que crece hasta compensar exactamente la fuerza magnética sobre los portadores.",
            "En el equilibrio la fuerza neta transversal es nula, $qE_H=qv_dB$, de modo que $E_H=v_dB$ y la tensión entre los bordes es $V_H=E_Hw=v_dBw$. Al escribir la velocidad de deriva en función de la corriente, $v_d=I/(nqwt)$, el ancho se cancela y queda $V_H=IB/(nqt)$.",
            "La importancia del efecto es doble. Su signo revela si los portadores son positivos o negativos, y su magnitud permite medir la densidad de portadores $n$. En metales como el cobre, con $n$ enorme, el efecto es muy pequeño (microvoltios); en semiconductores, con $n$ mucho menor, es grande y útil para fabricar sensores de campo magnético.",
            "Una confirmación microscópica viene del modelo de Drude con campo magnético: la ecuación de movimiento incluye aceleración eléctrica, fuerza magnética y frenado por colisiones $-\\vec v/\\tau$. En el estado estacionario, la condición de que no haya corriente transversal fija el campo de Hall.",
        ],
        analytic_steps=[
            ("Equilibrio de fuerzas transversales", "En el estado estacionario los portadores no se desvían más: la fuerza eléctrica del campo de Hall iguala a la magnética.",
             [r"qE_H=qv_dB\ \Rightarrow\ E_H=v_dB"]),
            ("Velocidad de deriva", f"Se obtiene de la corriente: $I=nqv_d(wt)$ con $w=10$ mm y $t=0.10$ mm, de modo que $wt={sci(w*t_,3)}\\ \\mathrm{{m^2}}$.",
             [r"v_d=\dfrac{I}{nqwt}",
              rf"v_d=\dfrac{{{i_:.1f}}}{{({sci(n,2)})({sci(E_CH,5)})({sci(w*t_,3)})}}={sci(vd,4)}\ \mathrm{{m/s}}"]),
            ("Campo de Hall", "Se sustituye la deriva y el campo magnético dado, $B=0.5$ T.",
             [rf"E_H=v_dB=({sci(vd,4)})({b})={sci(eh,4)}\ \mathrm{{V/m}}"]),
            ("Tensión de Hall", "El campo es uniforme a lo ancho de la cinta, así que la tensión es el campo por el ancho.",
             [rf"V_H=E_Hw=({sci(eh,4)})({sci(w,2)})={sci(eh*w,5)}\ \mathrm{{V}}={va*1e6:.4f}\ \mu\mathrm{{V}}"]),
            ("Fórmula directa", "Se comprueba que el ancho se cancela.",
             [rf"V_H=\dfrac{{IB}}{{nqt}}=\dfrac{{({i_:.1f})({b})}}{{({sci(n,2)})({sci(E_CH,5)})({sci(t_,2)})}}={va*1e6:.4f}\ \mu\mathrm{{V}}"]),
            ("Recuperar la densidad de portadores", "Si se mide $V_H$, se despeja $n=IB/(qtV_H)$. Con el valor analítico debe resultar el $n$ de partida.",
             [rf"n=\dfrac{{({i_:.1f})({b})}}{{({sci(E_CH,5)})({sci(t_,2)})({sci(va,5)})}}={sci(m8.carrier_density_from_hall(va),4)}\ \mathrm{{m^{{-3}}}}"]),
        ],
        numeric_steps=[
            ("Ecuaciones de Drude con B", f"Las componentes $(v_x,v_y)$ de la velocidad de un electrón (carga $-e$, $k=e/m={sci(k,4)}$ C/kg) cumplen $m\\,d\\vec v/dt=-e(\\vec E+\\vec v\\times\\vec B)-m\\vec v/\\tau$ con $\\vec B=B\\hat z$. La corriente va en $x$ y el campo de Hall en $y$.",
             [r"\dfrac{dv_x}{dt}=-k(E_x+v_yB)-\dfrac{v_x}{\tau},\qquad \dfrac{dv_y}{dt}=-k(E_y-v_xB)-\dfrac{v_y}{\tau}"]),
            ("Campo longitudinal y rango de búsqueda", f"El campo $E_x$ es el que sostiene la deriva a través del frenado; la incógnita es $E_y$, que se busca por bisección en el intervalo $[-2v_dB,\\,2v_dB]$.",
             [rf"E_x=\dfrac{{mv_d}}{{e\tau}}={sci(ex,4)}\ \mathrm{{V/m}},\qquad v_dB={sci(eh,4)}\ \mathrm{{V/m}}",
              rf"E_y\in[{sci(lo,4)},\ {sci(hi,4)}]\ \mathrm{{V/m}}"]),
            ("Primer paso de RK4 con la primera prueba", f"Para cada $E_y$ de prueba se integra hasta el estado estacionario con 800 pasos de $h=40\\tau/800={sci(h,3)}$ s. La primera prueba es el punto medio $E_y=0$, con $\\vec v(0)=0$.",
             [rf"\vec k_1=({sci(k1[0],4)},\ {sci(k1[1],4)}),\quad \vec k_2=({sci(k2[0],4)},\ {sci(k2[1],4)})\ \mathrm{{m/s^2}}",
              rf"\vec k_3=({sci(k3[0],4)},\ {sci(k3[1],4)}),\quad \vec k_4=({sci(k4[0],4)},\ {sci(k4[1],4)})",
              rf"\vec v_1=\dfrac h6(\vec k_1+2\vec k_2+2\vec k_3+\vec k_4)=({sci(y1[0],4)},\ {sci(y1[1],4)})\ \mathrm{{m/s}}"]),
            ("Estado estacionario de cada prueba", "Tras 40 pasos de tiempo $\\tau$ la velocidad ha alcanzado su valor límite (el transitorio decae como $e^{-t/\\tau}$). Se lee la componente transversal $v_y$.",
             [rf"\text{{con }}E_y=0:\ v_y^{{est}}={sci(trace[0][4],4)}\ \mathrm{{m/s}}\ \Rightarrow\ \text{{signo }}{'+' if trace[0][4] > 0 else '-'}"]),
            ("Criterio de bisección", "Se busca el $E_y$ para el cual $v_y^{est}=0$ (no hay deriva transversal). Si $v_y>0$, el cero está en la mitad superior (se sube el extremo inferior); si no, en la inferior.",
             [r"v_y^{est}(E_y)>0\ \Rightarrow\ E_{lo}\leftarrow E_y;\qquad v_y^{est}\le 0\ \Rightarrow\ E_{hi}\leftarrow E_y"]),
            ("Iteraciones de bisección", "En cada iteración el intervalo se reduce a la mitad; tras 80 iteraciones es del orden de $10^{-24}$ del original.",
             [tab]),
            ("Tensión de Hall y densidad", f"Con $E_y$ final se calcula $V_H=|E_y|w$ y con la fórmula inversa se recupera $n$. Comprobación: con ese campo, $v_y^{{est}}={sci(vy_end,2)}$ m/s.",
             [rf"E_y={sci(ey,6)}\ \mathrm{{V/m}}\ \Rightarrow\ V_H=|E_y|w={vn*1e6:.5f}\ \mu\mathrm{{V}}\quad(\text{{analítico }}{va*1e6:.5f}\ \mu\mathrm{{V}})",
              rf"n_{{num}}=\dfrac{{IB}}{{qtV_H}}={n_back/1e28:.5f}\times10^{{28}}\ \mathrm{{m^{{-3}}}},\qquad \text{{error }}={abs(vn-va)/va*100:.3f}\,\%"]),
        ],
    )


DETAIL = {
    "CoaxialLaplace": _coax(),
    "CapacitorNetwork": _network(),
    "DrudeElectrons": _drude(),
    "TaperedWire": _tapered(),
    "MeshCircuit": _mesh(),
    "RCCharging": _rc(),
    "HallEffect": _hall(),
}
