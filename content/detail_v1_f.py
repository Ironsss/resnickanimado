"""Explicacion y pasos a mano: Volumen 1, capitulos 21 a 25 (relatividad, temperatura, primera y segunda ley, teoria cinetica).

Escenas: MuonDecay, VelocityAndEnergy, ThermalExpansion, ThermalEquilibrium, GasProcesses, OttoCycle, GasPressure,
MaxwellBoltzmann, CarnotCycle, EntropyProduction. Todas las cifras se calculan con `physics/`.
"""
from __future__ import annotations

import numpy as np

from content.texfmt import fx, sci
from physics import ch21_relativity as c21
from physics import ch22_temperature as c22
from physics import ch23_first_law as c23
from physics import ch24_kinetic_theory as c24
from physics import ch25_second_law as c25
from physics.ch07_work_energy import simpson
from physics.integrators import integrate, rk4_step

NL = r" \\ "


def _rk4(f, t, y, h):
    """Un paso de RK4 devolviendo las cuatro pendientes y el nuevo valor."""
    y = np.asarray(y, dtype=float)
    k1 = f(t, y)
    k2 = f(t + h / 2, y + h / 2 * k1)
    k3 = f(t + h / 2, y + h / 2 * k2)
    k4 = f(t + h, y + h * k3)
    return k1, k2, k3, k4, y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def _g(x, d=4):
    """Numero generico en TeX (decimal o cientifico)."""
    return sci(float(x), d)


# =============================================================== MuonDecay
def _muon() -> dict:
    tau, beta, d, C = c21.TAU_MUON, c21.BETA_MUON, c21.D_MUON, c21.C
    g = c21.gamma(beta)
    v = beta * C
    tf = d / v
    life = g * tau
    p_rel, p_cl = c21.survival_analytic(), c21.survival_analytic(relativistic=False)
    n = 2_000_000
    p_mc, err = c21.survival_monte_carlo()
    rng = np.random.default_rng(3)
    s = rng.exponential(life, 8)
    ok = s > tf
    tab = NL.join(rf"{i + 1} & {s[i] * 1e6:.3f} & {'\\text{sí}' if ok[i] else '\\text{no}'}" for i in range(8))
    return dict(
        explanation=[
            "Un muón es una partícula inestable: en su propio sistema de referencia vive en promedio $\\tau = 2.197\\ \\mu\\mathrm{s}$ y decae de forma aleatoria, de modo que la fracción que sobrevive un tiempo $t$ es $e^{-t/\\tau}$. Con esa vida media y la rapidez de la luz, un muón recorrería apenas unos 660 m antes de decaer.",
            "Sin embargo, los muones creados en la alta atmósfera por rayos cósmicos llegan al suelo en cantidades medibles. La razón es la dilatación del tiempo: para un observador en la Tierra, un reloj que se mueve con rapidez $v$ marca intervalos más largos en un factor $\\gamma = 1/\\sqrt{1-v^2/c^2}$. El muón no \"vive más\" para sí mismo; es el tiempo de laboratorio el que transcurre más rápido que el suyo.",
            "El cálculo se hace en el sistema de la Tierra: el tiempo de vuelo es $d/v$ y la vida media dilatada es $\\gamma\\tau$. La supervivencia es el exponencial del cociente de ambos. Como $\\gamma\\approx 10$ a $0.995c$, el exponente baja de unos 15 a unos 1.5, y la fracción pasa de casi cero a alrededor de un cuarto.",
            "El resultado es muy sensible a $\\gamma$ porque $\\gamma$ crece rápidamente cuando $v\\to c$. Este experimento es una de las comprobaciones más directas de la relatividad especial, y solo es válido mientras el muón se mueva con rapidez constante, como se supone aquí.",
        ],
        analytic_steps=[
            ("Plantear la condición de supervivencia", "Una población que decae con vida media $\\tau'$ (medida en el sistema donde se observa) cumple $P(t)=e^{-t/\\tau'}$. En el sistema de la Tierra, $t$ es el tiempo de vuelo y $\\tau'=\\gamma\\tau$.",
             [r"P = \exp\!\left(-\dfrac{t_{vuelo}}{\gamma\tau}\right),\qquad t_{vuelo}=\dfrac{d}{\beta c}"]),
            ("Calcular el factor de Lorentz", "Se sustituye $\\beta = 0.995$ en la definición de $\\gamma$.",
             [rf"\gamma = \dfrac{{1}}{{\sqrt{{1-{beta}^2}}}} = \dfrac{{1}}{{\sqrt{{{1 - beta**2:.6f}}}}} = {g:.4f}"]),
            ("Calcular el tiempo de vuelo en la Tierra", "El muón recorre $d = 10\\ \\mathrm{km}$ con rapidez $v=\\beta c$.",
             [rf"v = {beta}\,({sci(C, 6)}\ \mathrm{{m/s}}) = {sci(v, 6)}\ \mathrm{{m/s}}", rf"t_{{vuelo}} = \dfrac{{{sci(d, 3)}\ \mathrm{{m}}}}{{{sci(v, 6)}\ \mathrm{{m/s}}}} = {sci(tf, 5)}\ \mathrm{{s}} = {tf * 1e6:.3f}\ \mu\mathrm{{s}}"]),
            ("Vida media dilatada y exponente", "La vida media vista desde la Tierra es $\\gamma\\tau$; el exponente es el número de vidas medias que dura el vuelo.",
             [rf"\gamma\tau = ({g:.4f})({sci(tau, 4)}\ \mathrm{{s}}) = {life * 1e6:.3f}\ \mu\mathrm{{s}}", rf"\dfrac{{t_{{vuelo}}}}{{\gamma\tau}} = \dfrac{{{tf * 1e6:.3f}}}{{{life * 1e6:.3f}}} = {tf / life:.4f}"]),
            ("Fracción que llega con dilatación", "Se evalúa el exponencial.",
             [rf"P_{{rel}} = e^{{-{tf / life:.4f}}} = {p_rel:.5f}\approx {100 * p_rel:.1f}\,\%"]),
            ("Fracción sin dilatación (caso clásico)", "Si el tiempo no se dilatara, la vida media sería solo $\\tau$ y el exponente sería $\\gamma$ veces mayor.",
             [rf"\dfrac{{t_{{vuelo}}}}{{\tau}} = \dfrac{{{tf * 1e6:.3f}}}{{{tau * 1e6:.3f}}} = {tf / tau:.3f},\qquad P_{{cl}} = e^{{-{tf / tau:.3f}}} = {sci(p_cl, 3)}"]),
            ("Interpretar", "La dilatación del tiempo aumenta la supervivencia en un factor enorme, lo que explica que los muones sí lleguen al suelo.",
             [rf"\dfrac{{P_{{rel}}}}{{P_{{cl}}}} = \dfrac{{{p_rel:.5f}}}{{{sci(p_cl, 3)}}} \approx {sci(p_rel / p_cl, 3)}"]),
        ],
        numeric_steps=[
            ("Idea del método Monte Carlo", "En lugar de usar la fórmula, se simulan muones: cada uno recibe un tiempo de decaimiento aleatorio con distribución exponencial de media $\\gamma\\tau$ (en el laboratorio) y se cuenta cuántos superan el tiempo de vuelo. La fracción de éxitos estima $P$.",
             [rf"t_i\sim \mathrm{{Exp}}(\gamma\tau),\qquad P\approx \dfrac{{\#\{{t_i>t_{{vuelo}}\}}}}{{N}},\qquad N={n}"]),
            ("Parámetros", "Se fija la semilla del generador (3) para que el resultado sea reproducible, y se calcula el umbral del tiempo de vuelo.",
             [rf"\gamma\tau = {life * 1e6:.3f}\ \mu\mathrm{{s}},\qquad t_{{vuelo}} = {tf * 1e6:.3f}\ \mu\mathrm{{s}},\qquad \text{{semilla}}=3"]),
            ("Primeras muestras a mano", "Las primeras ocho vidas generadas son las siguientes; cada una se compara con el umbral $t_{vuelo}$.",
             [rf"\begin{{array}}{{c|c|c}} i & t_i\ (\mu\mathrm{{s}}) & t_i>{tf * 1e6:.3f}\ \mu\mathrm{{s}} \\ \hline {tab} \end{{array}}"]),
            ("Promedio de las primeras muestras", "Con solo ocho muones la fracción es muy burda: depende de unos pocos sucesos.",
             [rf"\hat P_8 = \dfrac{{{int(ok.sum())}}}{{8}} = {ok.mean():.3f}"]),
            ("Escalar a dos millones de muestras", "El código repite lo mismo con $N = 2\\times10^6$. La incertidumbre estadística de una proporción es $\\sqrt{p(1-p)/N}$.",
             [rf"\hat P = \dfrac{{{int(round(p_mc * n))}}}{{{n}}} = {p_mc:.5f},\qquad \sigma = \sqrt{{\dfrac{{\hat P(1-\hat P)}}{{N}}}} = {err:.5f}"]),
            ("Comparar con el resultado analítico", "La diferencia debe ser del orden de $\\sigma$ (un error típico de un Monte Carlo disminuye como $1/\\sqrt N$).",
             [rf"|\hat P - P_{{rel}}| = |{p_mc:.5f}-{p_rel:.5f}| = {abs(p_mc - p_rel):.5f}\approx {abs(p_mc - p_rel) / err:.2f}\,\sigma"]),
            ("Por qué no se simula el caso clásico", "Sin dilatación $P_{cl}$ es tan pequeña que entre dos millones de muones se esperan menos de un evento, así que el muestreo directo no puede estimarla.",
             [rf"N\,P_{{cl}} = ({n})({sci(p_cl, 3)}) = {n * p_cl:.2f}"]),
        ],
    )


# =============================================================== VelocityAndEnergy
def _velocity() -> dict:
    C = c21.C
    u = v = 0.8
    ua = c21.velocity_addition_analytic(u, v)
    g8 = c21.gamma(0.8)
    bm = lambda b: np.array([[c21.gamma(b), c21.gamma(b) * b], [c21.gamma(b) * b, c21.gamma(b)]])  # noqa: E731
    B = bm(0.8)
    M = B @ B
    rap = np.arctanh(0.8)
    b9 = 0.9
    g9 = c21.gamma(b9)
    kr, kc = c21.kinetic_analytic(b9) / C**2, c21.kinetic_classical(b9) / C**2
    kn = c21.kinetic_numeric(b9) / C**2
    xs = np.linspace(0, b9, 5)
    f = xs * (1 / np.sqrt(1 - xs**2)) ** 3
    k4 = simpson(f, xs)
    rows = NL.join(rf"{i} & {xs[i]:.3f} & {f[i]:.5f}" for i in range(5))
    h4 = b9 / 4
    return dict(
        explanation=[
            "En la mecánica clásica las velocidades simplemente se suman, pero si lo hicieran siempre se podría superar $c$ lanzando un objeto desde otro objeto veloz. La relatividad corrige esto: si una nave se mueve con $v$ respecto de la Tierra y lanza algo con $u$ respecto de la nave, la velocidad respecto de la Tierra es $(u+v)/(1+uv/c^2)$, que nunca supera $c$.",
            "Una forma más natural de ver la regla es con la rapidez (rapidity) $\\phi$, definida por $\\beta=\\tanh\\phi$. Las rapideces sí se suman linealmente, igual que las velocidades en la física clásica, y la fórmula de adición no es más que $\\tanh(\\phi_1+\\phi_2)$. Cada transformación de Lorentz en una dimensión es una rotación hiperbólica de ángulo $\\phi$.",
            "Para la energía, el trabajo que realiza una fuerza al acelerar una partícula desde el reposo es $K=\\int v\\,dp$, pero ahora $p=\\gamma m v$, no $mv$. Al integrar se obtiene $K=(\\gamma-1)mc^2$, que a baja velocidad se reduce a $\\tfrac12 mv^2$ y que diverge cuando $v\\to c$: acelerar hasta $c$ requeriría energía infinita.",
            "A $0.9c$ la diferencia es muy grande: la energía cinética clásica ($0.405\\,mc^2$) sería menos de la mitad de la relativista (cercana a $1.29\\,mc^2$). Para velocidades de pocos por ciento de $c$ ambas coinciden prácticamente, que es la razón por la que la mecánica de Newton funciona en la vida diaria.",
        ],
        analytic_steps=[
            ("Plantear la adición de velocidades", "La nave se mueve con $v=0.8c$ respecto de la Tierra y el proyectil con $u=0.8c$ respecto de la nave. Se aplica la fórmula relativista (en unidades de $c$).",
             [r"u' = \dfrac{u+v}{1+uv}"]),
            ("Sustituir", "Se reemplazan los valores numéricos. Observa que la suma clásica daría $1.6c$, imposible.",
             [rf"u' = \dfrac{{0.8+0.8}}{{1+(0.8)(0.8)}} = \dfrac{{1.6}}{{{1 + 0.64:.2f}}} = {ua:.5f}\,c"]),
            ("Comprobar con rapideces", "Cada velocidad corresponde a una rapidez; se suman y se vuelve a velocidad.",
             [rf"\phi = \operatorname{{artanh}}(0.8) = {rap:.5f},\qquad u' = \tanh(2\phi) = \tanh({2 * rap:.5f}) = {np.tanh(2 * rap):.5f}\,c"]),
            ("Plantear la energía cinética", "El trabajo de la fuerza es $dK = v\\,dp$ con $p=\\gamma m v$. Al derivar $p$ respecto de $\\beta$ se obtiene $dp = mc\\,\\gamma^3 d\\beta$, y el integrando queda explícito.",
             [r"K = \int_0^{v} v\,dp = mc^2\int_0^{\beta}\beta'\,\gamma(\beta')^3\,d\beta' = mc^2\left[\gamma(\beta)-1\right]"]),
            ("Calcular γ a 0.9c", "Se evalúa el factor de Lorentz para $\\beta = 0.9$.",
             [rf"\gamma = \dfrac{{1}}{{\sqrt{{1-0.81}}}} = \dfrac{{1}}{{\sqrt{{0.19}}}} = {g9:.5f}"]),
            ("Energía cinética relativista", "Se resta $1$ (la energía en reposo) y se expresa en unidades de $mc^2$.",
             [rf"K_{{rel}} = ({g9:.5f}-1)\,mc^2 = {kr:.5f}\,mc^2"]),
            ("Energía cinética clásica y cociente", "Con la fórmula de Newton, a la misma velocidad.",
             [rf"K_{{cl}} = \tfrac12 m (0.9c)^2 = {kc:.4f}\,mc^2,\qquad \dfrac{{K_{{rel}}}}{{K_{{cl}}}} = {kr / kc:.3f}"]),
        ],
        numeric_steps=[
            ("Boost como matriz", "En coordenadas $(ct, x)$ una transformación de Lorentz de velocidad $\\beta$ es una matriz $2\\times2$ con $\\gamma$ en la diagonal y $\\gamma\\beta$ fuera de ella. Componer dos boosts es multiplicar matrices.",
             [rf"\Lambda(0.8) = \begin{{pmatrix}}\gamma & \gamma\beta\\ \gamma\beta & \gamma\end{{pmatrix}} = \begin{{pmatrix}}{B[0, 0]:.5f} & {B[0, 1]:.5f}\\ {B[1, 0]:.5f} & {B[1, 1]:.5f}\end{{pmatrix}}"]),
            ("Multiplicar las matrices", "El producto $\\Lambda(0.8)\\,\\Lambda(0.8)$ se calcula entrada por entrada.",
             [rf"M_{{00}} = {B[0, 0]:.5f}^2 + {B[0, 1]:.5f}^2 = {M[0, 0]:.5f},\qquad M_{{10}} = 2({B[0, 0]:.5f})({B[1, 0]:.5f}) = {M[1, 0]:.5f}"]),
            ("Leer la velocidad", "La velocidad de la transformación compuesta es el cociente $M_{10}/M_{00}$.",
             [rf"u' = \dfrac{{M_{{10}}}}{{M_{{00}}}} = \dfrac{{{M[1, 0]:.5f}}}{{{M[0, 0]:.5f}}} = {c21.velocity_addition_numeric(u, v):.6f}\,c"]),
            ("Resultado y error", "El código también suma rapideces con $\\tanh(\\operatorname{artanh}u+\\operatorname{artanh}v)$; ambos caminos coinciden con la fórmula cerrada a precisión de máquina.",
             [rf"|u'_{{mat}}-u'_{{anal}}| = {sci(abs(c21.velocity_addition_numeric(u, v) - ua), 2)},\qquad |u'_{{rap}}-u'_{{anal}}| = {sci(abs(c21.rapidity_addition(u, v) - ua), 2)}"]),
            ("Simpson para la energía: reducción a N = 4", f"Para $K/mc^2=\\int_0^{{0.9}}\\beta\\gamma^3\\,d\\beta$ la regla de Simpson usa $h=(b-a)/N$ y pesos $1,4,2,4,1$. Con $N=4$, $h={h4:.4f}$, la tabla del integrando es:",
             [rf"\begin{{array}}{{c|c|c}} i & \beta_i & \beta_i\gamma_i^3 \\ \hline {rows} \end{{array}}"]),
            ("Suma de Simpson con N = 4", "Se aplica $\\frac h3\\,[f_0+4f_1+2f_2+4f_3+f_4]$.",
             [rf"K\approx \dfrac{{{h4:.4f}}}{{3}}\left[{f[0]:.4f}+4({f[1]:.4f})+2({f[2]:.4f})+4({f[3]:.4f})+{f[4]:.4f}\right] = {k4:.5f}\,mc^2"]),
            ("N real = 4000", "Con $4000$ subintervalos ($h=2.25\\times10^{-4}$) la regla de Simpson, cuyo error decrece como $h^4$, reproduce el valor exacto.",
             [rf"K_{{num}} = {kn:.7f}\,mc^2,\qquad K_{{anal}} = {kr:.7f}\,mc^2,\qquad \text{{error}} = {sci(abs(kn - kr), 2)}\,mc^2"]),
        ],
    )


# =============================================================== ThermalExpansion
def _expansion() -> dict:
    L0, a0, bt, T0, T1 = c22.L0, c22.ALPHA0, c22.BETA, c22.T0, c22.T1
    dT = T1 - T0
    dl_lin = (c22.length_linear() - L0)
    dl_ex = c22.length_analytic() - L0
    dl_num = c22.length_numeric() - L0
    n = 400
    h = dT / n
    f = lambda t, y: np.array([c22.alpha(t) * y[0]])  # noqa: E731
    k1, k2, k3, k4, y1 = _rk4(f, T0, [L0], h)
    corr = bt * dT / 2
    return dict(
        explanation=[
            "Cuando se calienta un sólido, la agitación de sus átomos aumenta y la distancia media entre ellos crece. Para variaciones moderadas el cambio relativo de longitud es proporcional al cambio de temperatura, $\\Delta L/L_0=\\alpha\\,\\Delta T$, donde $\\alpha$ es el coeficiente de dilatación lineal del material (para el acero, del orden de $10^{-5}\\ \\mathrm{K^{-1}}$).",
            "Un riel de 30 m calentado 100 K se alarga unas decenas de milímetros: poco en términos relativos ($\\sim 0.04\\,\\%$), pero suficiente para deformar una vía si no hay juntas de dilatación.",
            "En realidad $\\alpha$ no es constante: crece algo con la temperatura. Aquí se modela como $\\alpha(T)=\\alpha_0[1+\\beta(T-T_0)]$. Entonces ya no vale la fórmula lineal, hay que integrar la ecuación diferencial $dL/dT=\\alpha(T)\\,L$, que dice que el alargamiento relativo en un intervalo infinitesimal es $\\alpha\\,dT$.",
            "La solución exacta es una exponencial cuyo exponente es la integral de $\\alpha$ entre $T_0$ y $T$. Cuando $\\alpha\\,\\Delta T\\ll 1$ (como aquí), ese exponencial es casi lineal y toda la diferencia con el cálculo de $\\alpha$ constante viene de la corrección de $\\alpha(T)$ de un 10 % en este caso.",
        ],
        analytic_steps=[
            ("Dilatación con α constante", "Es la aproximación de primer orden: se usa $\\alpha_0$ en todo el intervalo, $\\Delta T = 120-20 = 100\\ \\mathrm{K}$.",
             [rf"\Delta L_{{lin}} = \alpha_0 L_0\Delta T = ({sci(a0, 3)}\ \mathrm{{K^{{-1}}}})({L0:.0f}\ \mathrm{{m}})({dT:.0f}\ \mathrm{{K}}) = {dl_lin * 1000:.2f}\ \mathrm{{mm}}"]),
            ("Plantear la ecuación diferencial", "Con coeficiente variable, el alargamiento relativo infinitesimal es $dL/L = \\alpha(T)\\,dT$. Se separan variables.",
             [r"\dfrac{dL}{L} = \alpha_0\left[1+\beta(T-T_0)\right]dT"]),
            ("Integrar", "Se integra de $T_0$ a $T$ con $\\Delta T = T-T_0$ y se aplica el exponencial.",
             [r"\ln\dfrac{L}{L_0} = \alpha_0\left(\Delta T + \dfrac{\beta\,\Delta T^2}{2}\right)\quad\Rightarrow\quad L = L_0\exp\!\left[\alpha_0\left(\Delta T+\dfrac{\beta\,\Delta T^2}{2}\right)\right]"]),
            ("Evaluar el exponente", "Se sustituyen $\\alpha_0 = 12\\times10^{-6}\\ \\mathrm{K^{-1}}$, $\\beta=0.002\\ \\mathrm{K^{-1}}$ y $\\Delta T = 100\\ \\mathrm{K}$.",
             [rf"\Delta T + \dfrac{{\beta\Delta T^2}}{{2}} = {dT:.0f} + \dfrac{{({bt})({dT:.0f})^2}}{{2}} = {dT:.0f}+{bt * dT**2 / 2:.0f} = {dT + bt * dT**2 / 2:.0f}\ \mathrm{{K}}",
              rf"\alpha_0(\ldots) = ({sci(a0, 3)})({dT + bt * dT**2 / 2:.0f}) = {sci(a0 * (dT + bt * dT**2 / 2), 5)}"]),
            ("Longitud final y alargamiento", "Se evalúa el exponencial y se resta $L_0$.",
             [rf"L = {L0:.0f}\,e^{{{sci(a0 * (dT + bt * dT**2 / 2), 5)}}} = {c22.length_analytic():.6f}\ \mathrm{{m}}", rf"\Delta L_{{exacto}} = L-L_0 = {dl_ex * 1000:.2f}\ \mathrm{{mm}}"]),
            ("Comparar con el caso lineal", "El coeficiente creciente aumenta el alargamiento. El cociente coincide con la corrección $1+\\beta\\Delta T/2$.",
             [rf"\dfrac{{\Delta L_{{exacto}}}}{{\Delta L_{{lin}}}} = \dfrac{{{dl_ex * 1000:.2f}}}{{{dl_lin * 1000:.2f}}} = {dl_ex / dl_lin:.4f}\approx 1+\dfrac{{\beta\Delta T}}{{2}} = {1 + corr:.2f}",
              rf"\Delta L_{{exacto}}-\Delta L_{{lin}} = {(dl_ex - dl_lin) * 1000:.2f}\ \mathrm{{mm}}"]),
        ],
        numeric_steps=[
            ("Método y paso", "Se integra $dL/dT=\\alpha(T)L$ de $20$ a $120\\,^\\circ\\mathrm{C}$ con RK4 y $N=400$ pasos. La variable independiente es la temperatura y el paso es $h=\\Delta T/N$.",
             [rf"h = \dfrac{{{dT:.0f}}}{{{n}}} = {h:.2f}\ \mathrm{{K}},\qquad f(T,L)=\alpha(T)\,L,\qquad L_0={L0:.0f}\ \mathrm{{m}}"]),
            ("Fórmula de RK4", "En cada paso se promedian cuatro pendientes (inicio, dos puntos medios y fin).",
             [r"L_{k+1}=L_k+\dfrac h6\left(k_1+2k_2+2k_3+k_4\right)"]),
            ("Primeras dos pendientes", "Con $T_0=20\\,^\\circ\\mathrm{C}$, $L_0=30$ m: $k_1$ usa $\\alpha(20)=\\alpha_0$; $k_2$ evalúa en el punto medio $T=20.125$ con $L$ avanzado media pendiente.",
             [rf"k_1 = \alpha(20)\,L_0 = ({sci(a0, 3)})({L0:.0f}) = {sci(k1[0], 5)}\ \mathrm{{m/K}}",
              rf"k_2 = \alpha(20.125)\,(L_0+\tfrac h2 k_1) = ({sci(c22.alpha(20 + h / 2), 6)})({L0 + h / 2 * k1[0]:.8f}) = {sci(k2[0], 6)}\ \mathrm{{m/K}}"]),
            ("Últimas dos pendientes", "$k_3$ repite el punto medio con la pendiente $k_2$; $k_4$ usa el extremo $T=20.25$ y la pendiente $k_3$.",
             [rf"k_3 = {sci(k3[0], 6)}\ \mathrm{{m/K}},\qquad k_4 = \alpha(20.25)\,(L_0+h\,k_3) = {sci(k4[0], 6)}\ \mathrm{{m/K}}"]),
            ("Primer valor L₁", "Se combinan las cuatro pendientes con pesos $1,2,2,1$.",
             [rf"L_1 = {L0:.0f}+\dfrac{{{h:.2f}}}{{6}}\left[{sci(k1[0], 6)}+2({sci(k2[0], 6)})+2({sci(k3[0], 6)})+{sci(k4[0], 6)}\right] = {y1[0]:.9f}\ \mathrm{{m}}",
              rf"\Delta L_1 = {(y1[0] - L0) * 1000:.5f}\ \mathrm{{mm}}"]),
            ("Repetir 400 veces", "El mismo cálculo se repite hasta llegar a $T=120\\,^\\circ\\mathrm{C}$; en $T=70$ (paso 200) y $T=120$ (paso 400) se leen los resultados del cuadro de verificación.",
             [rf"\Delta L_{{num}}(120) = {dl_num * 1000:.5f}\ \mathrm{{mm}}"]),
            ("Error frente al analítico", "El error local de RK4 es $O(h^5)$ y el global $O(h^4)$; con $h=0.25$ K y una solución casi lineal se llega al límite de la aritmética.",
             [rf"\Delta L_{{anal}} = {dl_ex * 1000:.5f}\ \mathrm{{mm}},\qquad |\Delta L_{{num}}-\Delta L_{{anal}}| = {sci(abs(dl_num - dl_ex) * 1000, 2)}\ \mathrm{{mm}}"]),
        ],
    )


# =============================================================== ThermalEquilibrium
def _equilibrium() -> dict:
    m1, c1, t10, m2, c2, t20, hh = c22.M1, c22.C1, c22.T1_0, c22.M2, c22.C2, c22.T2_0, c22.H
    C1, C2 = m1 * c1, m2 * c2
    tf = c22.final_temperature_analytic()
    lam = c22.rate_analytic()
    tau = 1 / lam
    a20 = c22.temperatures_analytic(20.0)
    t, y = c22.temperatures_numeric(100.0, 2000)
    t2, y2 = c22.temperatures_numeric(300.0, 6000)
    n20 = [float(np.interp(20, t, y[:, 0])), float(np.interp(20, t, y[:, 1]))]
    rhs = lambda _t, yy: np.array([-hh * (yy[0] - yy[1]) / C1, hh * (yy[0] - yy[1]) / C2])  # noqa: E731
    h = 100.0 / 2000
    k1, k2, k3, k4, y1 = _rk4(rhs, 0.0, [t10, t20], h)
    q = C1 * (t10 - tf)

    def v(a):
        return rf"\begin{{pmatrix}}{a[0]:.5f}\\ {a[1]:.5f}\end{{pmatrix}}"
    return dict(
        explanation=[
            "Dos cuerpos con temperaturas distintas, en contacto térmico, intercambian calor desde el más caliente hacia el más frío hasta que ambos alcanzan la misma temperatura (equilibrio térmico, base de la ley cero). La capacidad calorífica $C=mc$ mide cuánta energía hace falta para cambiar 1 K la temperatura del cuerpo.",
            "Aquí el agua tiene una capacidad mucho mayor que el bloque de aluminio, de modo que la temperatura final queda cerca de la del agua. Si no hay pérdidas al exterior, el calor que cede uno es el que gana el otro, y de ahí sale $T_f=(C_1T_1+C_2T_2)/(C_1+C_2)$, un promedio ponderado por las capacidades.",
            "El camino hacia el equilibrio lo describe la conductancia $h$ (W/K): el flujo de calor es $h(T_1-T_2)$. Las dos ecuaciones acopladas se reducen a una sola para la diferencia $T_1-T_2$, que decae exponencialmente con rapidez $\\lambda=h(1/C_1+1/C_2)$; su inverso es la constante de tiempo $\\tau$, que fija cuánto tarda el sistema en acercarse al equilibrio.",
            "Un conductor más pobre (menor $h$) o capacidades mayores alargan $\\tau$. Después de unos cuatro o cinco $\\tau$ la diferencia de temperaturas ya es menor que un 2 % de la inicial.",
        ],
        analytic_steps=[
            ("Capacidades caloríficas", "Se calculan $C=mc$ con $c_{Al}=900\\ \\mathrm{J/(kg\\,K)}$ y $c_{agua}=4186\\ \\mathrm{J/(kg\\,K)}$.",
             [rf"C_1 = (0.30)(900) = {C1:.1f}\ \mathrm{{J/K}},\qquad C_2 = (0.50)(4186) = {C2:.1f}\ \mathrm{{J/K}}"]),
            ("Conservación de la energía", "Sin pérdidas, $C_1(T_1-T_f)=C_2(T_f-T_2)$. Se despeja $T_f$.",
             [r"C_1(T_1-T_f)=C_2(T_f-T_2)\ \Rightarrow\ T_f=\dfrac{C_1T_1+C_2T_2}{C_1+C_2}"]),
            ("Temperatura final", "Se sustituyen los números.",
             [rf"T_f = \dfrac{{({C1:.1f})(150)+({C2:.1f})(20)}}{{{C1:.1f}+{C2:.1f}}} = \dfrac{{{C1 * t10 + C2 * t20:.1f}}}{{{C1 + C2:.1f}}} = {tf:.4f}\ ^\circ\mathrm{{C}}"]),
            ("Calor transferido", "El aluminio cede el calor que absorbe el agua.",
             [rf"Q = C_1(T_1-T_f) = ({C1:.1f})({t10 - tf:.4f}) = {q:.1f}\ \mathrm{{J}}"]),
            ("Ecuación para la diferencia", "Restando las dos ecuaciones, $\\Delta=T_1-T_2$ cumple $d\\Delta/dt=-\\lambda\\Delta$ con $\\lambda=h(1/C_1+1/C_2)$.",
             [rf"\lambda = {hh:.0f}\left(\dfrac{{1}}{{{C1:.1f}}}+\dfrac{{1}}{{{C2:.1f}}}\right) = {hh:.0f}({1 / C1:.6f}+{1 / C2:.6f}) = {lam:.5f}\ \mathrm{{s^{{-1}}}}"]),
            ("Constante de tiempo", "Es el inverso de $\\lambda$.",
             [rf"\tau = \dfrac{{1}}{{\lambda}} = \dfrac{{1}}{{{lam:.5f}}} = {tau:.2f}\ \mathrm{{s}}"]),
            ("Temperaturas en función del tiempo", "Cada cuerpo tiende exponencialmente a $T_f$. Se evalúa en $t=20$ s.",
             [r"T_1(t)=T_f+(T_1^0-T_f)e^{-\lambda t},\qquad T_2(t)=T_f+(T_2^0-T_f)e^{-\lambda t}",
              rf"e^{{-\lambda\,20}} = e^{{-{lam * 20:.4f}}} = {np.exp(-lam * 20):.5f}",
              rf"T_1(20) = {tf:.4f}+({t10 - tf:.4f})({np.exp(-lam * 20):.5f}) = {a20[0]:.4f}\ ^\circ\mathrm{{C}},\qquad T_2(20) = {a20[1]:.4f}\ ^\circ\mathrm{{C}}"]),
        ],
        numeric_steps=[
            ("Sistema de EDO", "Se integra el sistema acoplado para $\\vec y=(T_1,T_2)$ con RK4. La función derivada es $f=(-h(T_1-T_2)/C_1,\\ h(T_1-T_2)/C_2)$.",
             [rf"h_{{paso}}=\dfrac{{100}}{{2000}} = {h:.2f}\ \mathrm{{s}},\qquad \vec y_0=({t10:.0f},\,{t20:.0f})\ ^\circ\mathrm{{C}}"]),
            ("Pendiente k₁", "Se evalúa $f$ en el estado inicial; la diferencia es $130$ K.",
             [rf"\vec k_1 = \left(-\dfrac{{20(130)}}{{{C1:.1f}}},\ \dfrac{{20(130)}}{{{C2:.1f}}}\right) = {v(k1)}\ \mathrm{{K/s}}"]),
            ("Pendientes k₂ y k₃", "Se avanza media pendiente ($h/2=0.025$ s) y se vuelve a evaluar, dos veces.",
             [rf"\vec y_0+\tfrac h2\vec k_1 = ({t10 + h / 2 * k1[0]:.5f},\ {t20 + h / 2 * k1[1]:.5f}),\qquad \vec k_2 = {v(k2)}",
              rf"\vec k_3 = {v(k3)}"]),
            ("Pendiente k₄ y primer paso", "$\\vec k_4$ se evalúa al final del paso con $\\vec y_0+h\\vec k_3$; después se combinan con pesos $1,2,2,1$.",
             [rf"\vec k_4 = {v(k4)}",
              rf"\vec y_1 = \vec y_0+\dfrac{{{h:.2f}}}{{6}}\left(\vec k_1+2\vec k_2+2\vec k_3+\vec k_4\right) = ({y1[0]:.5f},\ {y1[1]:.5f})\ ^\circ\mathrm{{C}}"]),
            ("Seguir hasta t = 20 s", "Se repiten los pasos y se interpola. Esto da las temperaturas a $20$ s.",
             [rf"T_1^{{num}}(20) = {n20[0]:.4f}\ ^\circ\mathrm{{C}}\ \ (\text{{analítico }}{a20[0]:.4f}),\qquad T_2^{{num}}(20) = {n20[1]:.4f}\ ^\circ\mathrm{{C}}\ \ (\text{{analítico }}{a20[1]:.4f})"]),
            ("Criterio para el valor final", "Para la temperatura final se integra hasta $300\\ \\mathrm{s}$ ($\\approx 25\\,\\tau$) con 6000 pasos, cuando la diferencia ya es despreciable.",
             [rf"T_1(300)-T_2(300) = {y2[-1, 0] - y2[-1, 1]:.2e}\ \mathrm{{K}}".replace("e-", r"\times10^{-").replace(" \\mathrm", "}\\ \\mathrm") if False else rf"e^{{-\lambda\,300}} = {sci(float(np.exp(-lam * 300)), 3)}"]),
            ("Resultado y error", "La temperatura numérica de equilibrio coincide con el promedio ponderado.",
             [rf"T_f^{{num}} = {y2[-1, 0]:.6f}\ ^\circ\mathrm{{C}},\qquad T_f^{{anal}} = {tf:.6f}\ ^\circ\mathrm{{C}},\qquad \text{{error}} = {sci(abs(y2[-1, 0] - tf), 2)}\ \mathrm{{K}}"]),
        ],
    )


# =============================================================== GasProcesses
def _gas_processes() -> dict:
    R, n, g = c23.R, c23.N_MOL, c23.GAMMA_MONO
    p1, v1, v2 = c23.P1, c23.V1, c23.V2
    t1 = c23.temperature(p1, v1)
    cv = R / (g - 1)
    rows = c23.first_law_table()
    wi, wb, wa = (c23.work_analytic(k) for k in ("isothermal", "isobaric", "adiabatic"))
    p2a = p1 * (v1 / v2) ** g
    t2a = t1 * (v1 / v2) ** (g - 1)
    t2b = c23.temperature(p1, v2)
    vv = np.linspace(v1, v2, 5)
    pp = p1 * v1 / vv
    s4 = simpson(pp, vv)
    nn = 400
    h = (v2 - v1) / nn
    f = lambda v, y: np.array([-g * y[0] / v])  # noqa: E731
    k1, k2, k3, k4, y1 = _rk4(f, v1, [p1], h)
    p_ex = p1 * (v1 / (v1 + h)) ** g
    wn = [r["W"] for r in rows]
    tb = NL.join(rf"{i} & {vv[i] * 1000:.2f} & {pp[i] / 1e5:.4f}" for i in range(5))
    return dict(
        explanation=[
            "La primera ley de la termodinámica es la conservación de la energía aplicada a un gas: el calor que recibe se reparte entre aumentar su energía interna y realizar trabajo sobre el entorno, $Q=\\Delta U+W$. El trabajo de expansión es el área bajo la curva $p(V)$, por lo que depende del camino, igual que $Q$; en cambio $\\Delta U$ solo depende de los estados inicial y final.",
            "Para un gas ideal monoatómico la energía interna es puramente cinética de traslación, $U=\\tfrac32 nRT$, así que $\\Delta U=nC_V\\Delta T$ con $C_V=\\tfrac32R$ sin importar el proceso. Esto permite hallar $\\Delta U$ con solo conocer las temperaturas extremas.",
            "Los tres caminos tienen la misma expansión de $10$ a $20$ L, pero distinto efecto. En el isotérmico la temperatura no cambia, $\\Delta U=0$ y todo el calor recibido se convierte en trabajo. En el isobárico el gas se calienta y recibe más calor que el trabajo que entrega. En el adiabático no hay intercambio de calor: el trabajo sale de la propia energía interna, por lo que el gas se enfría.",
            "A igual volumen final, el trabajo es mayor en el isobárico que en el isotérmico, y este mayor que en el adiabático, porque la presión cae más rápido en la curva adiabática ($pV^\\gamma=\\text{cte}$, con $\\gamma=5/3$) que en la isoterma ($pV=\\text{cte}$).",
        ],
        analytic_steps=[
            ("Estado inicial", "La temperatura se obtiene de la ecuación de estado con $n=1$ mol y $R=8.314\\ \\mathrm{J/(mol\\,K)}$.",
             [rf"T_1 = \dfrac{{p_1V_1}}{{nR}} = \dfrac{{({sci(p1, 3)})({v1})}}{{(1)({R})}} = {t1:.2f}\ \mathrm{{K}},\qquad C_V=\tfrac32R={cv:.3f}\ \mathrm{{J/(mol\,K)}}"]),
            ("Isotérmico", "Con $p=p_1V_1/V$ el trabajo es la integral de $1/V$; $\\Delta U=0$ y $Q=W$.",
             [r"W = p_1V_1\ln\dfrac{V_2}{V_1}", rf"W = ({sci(p1, 3)})({v1})\ln 2 = {p1 * v1:.0f}\,({np.log(2):.5f}) = {wi:.1f}\ \mathrm{{J}},\qquad \Delta U=0,\qquad Q={wi:.1f}\ \mathrm{{J}}"]),
            ("Isobárico", "A presión constante $W=p\\Delta V$, la temperatura final se obtiene con la ley del gas y $\\Delta U = nC_V\\Delta T$.",
             [rf"W = p_1(V_2-V_1) = ({sci(p1, 3)})({v2 - v1:.3f}) = {wb:.1f}\ \mathrm{{J}}",
              rf"T_2 = \dfrac{{p_1V_2}}{{nR}} = {t2b:.2f}\ \mathrm{{K}},\qquad \Delta U = (1)({cv:.3f})({t2b - t1:.2f}) = {rows[1]['dU']:.1f}\ \mathrm{{J}}",
              rf"Q = \Delta U+W = {rows[1]['dU']:.1f}+{wb:.1f} = {rows[1]['Q']:.1f}\ \mathrm{{J}}"]),
            ("Adiabático: presión final", "Con $pV^\\gamma=\\text{cte}$ y $\\gamma=5/3$:",
             [rf"p_2 = p_1\left(\dfrac{{V_1}}{{V_2}}\right)^{{\gamma}} = ({sci(p1, 3)})(0.5)^{{{g:.4f}}} = {p2a:.1f}\ \mathrm{{Pa}}"]),
            ("Adiabático: trabajo", "Se integra $p\\,dV$ con $p=p_1V_1^\\gamma V^{-\\gamma}$ y se simplifica con $p_1V_1^\\gamma=p_2V_2^\\gamma$.",
             [r"W = \dfrac{p_1V_1-p_2V_2}{\gamma-1}", rf"W = \dfrac{{({p1 * v1:.1f})-({p2a * v2:.2f})}}{{{g - 1:.4f}}} = {wa:.1f}\ \mathrm{{J}}"]),
            ("Adiabático: ΔU y Q", "La temperatura final sale de $T V^{\\gamma-1}=\\text{cte}$. Como $Q=0$, $\\Delta U=-W$, lo que sirve de comprobación.",
             [rf"T_2 = T_1\left(\dfrac{{V_1}}{{V_2}}\right)^{{\gamma-1}} = ({t1:.2f})(0.5)^{{{g - 1:.4f}}} = {t2a:.2f}\ \mathrm{{K}}",
              rf"\Delta U = nC_V(T_2-T_1) = ({cv:.3f})({t2a - t1:.2f}) = {rows[2]['dU']:.1f}\ \mathrm{{J}} = -W,\qquad Q = \Delta U+W \approx {rows[2]['Q']:.1e}".replace("e-", r"\times10^{-").replace("e+", r"\times10^{") + "}" if False else rf"\Delta U = nC_V(T_2-T_1) = ({cv:.3f})({t2a - t1:.2f}) = {rows[2]['dU']:.1f}\ \mathrm{{J}} = -W,\qquad Q = \Delta U+W \approx 0"]),
        ],
        numeric_steps=[
            ("Qué hace el código", "El trabajo se calcula como la integral numérica $\\int p\\,dV$ con la regla de Simpson. En isotérmico e isobárico $p(V)$ se conoce; en el adiabático se obtiene resolviendo $dp/dV=-\\gamma p/V$ con RK4, sin usar $pV^\\gamma=\\text{cte}$.",
             [rf"V_1={v1}\ \mathrm{{m^3}},\quad V_2={v2}\ \mathrm{{m^3}},\quad N=400,\quad h=\dfrac{{V_2-V_1}}{{N}}={sci(h, 3)}\ \mathrm{{m^3}}"]),
            ("Simpson: mecánica con N = 4 (isotérmico)", "Para ver el algoritmo se toma $N=4$, $h=2.5\\times10^{-3}$ m³, y la fórmula $\\frac h3[p_0+4p_1+2p_2+4p_3+p_4]$.",
             [rf"\begin{{array}}{{c|c|c}} i & V_i\ (\mathrm{{L}}) & p_i\ (10^5\ \mathrm{{Pa}}) \\ \hline {tb} \end{{array}}",
              rf"W_{{N=4}} = \dfrac{{{(v2 - v1) / 4:.4f}}}{{3}}\left[{pp[0]:.0f}+4({pp[1]:.0f})+2({pp[2]:.0f})+4({pp[3]:.0f})+{pp[4]:.0f}\right] = {s4:.2f}\ \mathrm{{J}}\quad(\text{{exacto }}{wi:.2f})"]),
            ("Simpson con N = 400", "Con 400 subintervalos el error de Simpson (proporcional a $h^4$) se reduce muchísimo.",
             [rf"W_{{isot}}^{{num}} = {wn[0]:.3f}\ \mathrm{{J}},\qquad W_{{isob}}^{{num}} = {wn[1]:.3f}\ \mathrm{{J}}"]),
            ("RK4 para la adiabática: primer paso", "Se parte de $p(V_1)=2\\times10^5$ Pa. Las pendientes usan $f(V,p)=-\\gamma p/V$ y el paso $h=2.5\\times10^{-5}$ m³.",
             [rf"k_1 = -\dfrac{{\gamma p_0}}{{V_0}} = -\dfrac{{({g:.4f})({sci(p1, 3)})}}{{{v1}}} = {sci(k1[0], 5)}\ \mathrm{{Pa/m^3}}",
              rf"k_2 = {sci(k2[0], 5)},\qquad k_3 = {sci(k3[0], 5)},\qquad k_4 = {sci(k4[0], 5)}"]),
            ("Combinar", "$p_1 = p_0 + \\frac h6(k_1+2k_2+2k_3+k_4)$ y comparar con la solución exacta en $V_0+h$.",
             [rf"p(V_0+h) = {sci(p1, 3)}+\dfrac{{{sci(h, 3)}}}{{6}}\,(\ldots) = {y1[0]:.4f}\ \mathrm{{Pa}}",
              rf"p_{{exacto}} = p_1\left(\dfrac{{V_1}}{{V_1+h}}\right)^{{\gamma}} = {p_ex:.4f}\ \mathrm{{Pa}},\qquad \text{{diferencia}}={sci(abs(y1[0] - p_ex), 2)}\ \mathrm{{Pa}}"]),
            ("Trabajo adiabático", "Los 401 valores de $p$ así obtenidos se integran con Simpson.",
             [rf"W_{{ad}}^{{num}} = {wn[2]:.3f}\ \mathrm{{J}}\quad(\text{{analítico }}{wa:.3f}\ \mathrm{{J}})"]),
            ("Primera ley y errores", "Con $\\Delta U=nC_V(T_2-T_1)$ y $Q=\\Delta U+W$ se completa la tabla. En el adiabático $Q$ debe salir cero.",
             [rf"\begin{{array}}{{l|c|c|c}} & W\ (\mathrm{{J}}) & \Delta U\ (\mathrm{{J}}) & Q\ (\mathrm{{J}}) \\ \hline \text{{isotérmico}} & {rows[0]['W']:.2f} & {rows[0]['dU']:.2f} & {rows[0]['Q']:.2f} \\ \text{{isobárico}} & {rows[1]['W']:.2f} & {rows[1]['dU']:.2f} & {rows[1]['Q']:.2f} \\ \text{{adiabático}} & {rows[2]['W']:.2f} & {rows[2]['dU']:.2f} & {rows[2]['Q']:.2e} \end{{array}}".replace(f"{rows[2]['Q']:.2e}", "0.00"),
              rf"|W^{{num}}-W^{{anal}}|:\quad {sci(abs(wn[0] - wi), 2)},\ {sci(abs(wn[1] - wb), 2)},\ {sci(abs(wn[2] - wa), 2)}\ \mathrm{{J}}"]),
        ],
    )


# =============================================================== OttoCycle
def _otto() -> dict:
    R, n, gm, r = c23.R, c23.N_MOL, c23.GAMMA_DIA, c23.RATIO
    st = c23.otto_states()
    o = c23.otto_numeric()
    cv = R / (gm - 1)
    (pa, va, ta), (pb, vb, tb), (pc, vc, tc), (pd, vd, td) = st["A"], st["B"], st["C"], st["D"]
    qin, qout = cv * (tc - tb), cv * (td - ta)
    wnet = qin - qout
    eta = c23.otto_efficiency_analytic()
    nn = 800
    h = (vb - va) / nn
    f = lambda v, y: np.array([-gm * y[0] / v])  # noqa: E731
    k1, k2, k3, k4, y1 = _rk4(f, va, [pa], h)
    pex = pa * (va / (va + h)) ** gm
    v_ab, p_ab = o["ab"]
    v_cd, p_cd = o["cd"]
    wab, wcd = simpson(p_ab, v_ab), simpson(p_cd, v_cd)
    return dict(
        explanation=[
            "El ciclo de Otto es el modelo ideal del motor de gasolina de cuatro tiempos. Consta de cuatro procesos: compresión adiabática del gas ($A\\to B$), combustión a volumen constante que le agrega calor ($B\\to C$), expansión adiabática en la que se obtiene trabajo ($C\\to D$) y enfriamiento a volumen constante que expulsa calor ($D\\to A$).",
            "En un ciclo completo el gas vuelve a su estado inicial, por lo que $\\Delta U=0$ y el trabajo neto es la diferencia entre el calor absorbido y el cedido, $W_{neto}=Q_{entra}-Q_{sale}$. Las adiabáticas no intercambian calor, así que solo las dos isocoras cuentan en el balance.",
            "Como en las isocoras $Q=nC_V\\Delta T$ y en las adiabáticas $TV^{\\gamma-1}=\\text{cte}$, todas las temperaturas se relacionan con la relación de compresión $r=V_A/V_B$ y la eficiencia depende solo de $r$ y de $\\gamma$: $\\eta=1-r^{1-\\gamma}$. A mayor compresión, mayor eficiencia (por eso los motores buscan alta compresión, limitados por la detonación del combustible).",
            "Con $r=8$ y $\\gamma=1.4$ la eficiencia teórica es cercana al 56 %. Los motores reales rinden bastante menos por fricción, pérdidas de calor y combustión incompleta; el resultado aquí es un límite ideal.",
        ],
        analytic_steps=[
            ("Estado A (inicio)", "Se parte de $T_A = 300$ K y $V_A = 25$ L con 1 mol.",
             [rf"p_A = \dfrac{{nRT_A}}{{V_A}} = \dfrac{{(1)({R})({ta:.0f})}}{{{va}}} = {pa:.1f}\ \mathrm{{Pa}}"]),
            ("Compresión adiabática A→B", "Con $r=8$, $V_B=V_A/r$ y $TV^{\\gamma-1}=\\text{cte}$ ($\\gamma-1=0.4$).",
             [rf"V_B = \dfrac{{{va}}}{{8}} = {vb:.6f}\ \mathrm{{m^3}},\qquad T_B = T_A\,r^{{\gamma-1}} = {ta:.0f}\,(8)^{{0.4}} = {tb:.2f}\ \mathrm{{K}}",
              rf"p_B = \dfrac{{nRT_B}}{{V_B}} = {pb:.1f}\ \mathrm{{Pa}}"]),
            ("Calentamiento isocoro B→C", "A $V_C=V_B$ el gas llega a $T_C=1800$ K. El calor absorbido es $nC_V\\Delta T$ con $C_V=\\tfrac52R$.",
             [rf"C_V = \dfrac{{R}}{{\gamma-1}} = {cv:.3f}\ \mathrm{{J/(mol\,K)}},\qquad p_C = \dfrac{{nRT_C}}{{V_B}} = {pc:.1f}\ \mathrm{{Pa}}",
              rf"Q_{{entra}} = nC_V(T_C-T_B) = ({cv:.3f})({tc:.0f}-{tb:.2f}) = {qin:.1f}\ \mathrm{{J}}"]),
            ("Expansión adiabática C→D", "Se aplica de nuevo $TV^{\\gamma-1}=\\text{cte}$ de $V_B$ a $V_A$.",
             [rf"T_D = T_C\,r^{{1-\gamma}} = {tc:.0f}\,(8)^{{-0.4}} = {td:.2f}\ \mathrm{{K}},\qquad p_D = \dfrac{{nRT_D}}{{V_A}} = {pd:.1f}\ \mathrm{{Pa}}"]),
            ("Enfriamiento isocoro D→A", "El gas cede calor a volumen constante hasta $T_A$.",
             [rf"Q_{{sale}} = nC_V(T_D-T_A) = ({cv:.3f})({td:.2f}-{ta:.0f}) = {qout:.1f}\ \mathrm{{J}}"]),
            ("Trabajo neto", "El ciclo cierra, así que el trabajo es la diferencia de calores.",
             [rf"W_{{neto}} = Q_{{entra}}-Q_{{sale}} = {qin:.1f}-{qout:.1f} = {wnet:.1f}\ \mathrm{{J}}"]),
            ("Eficiencia", "Se calcula con la definición y con la fórmula cerrada; coinciden.",
             [rf"\eta = \dfrac{{W_{{neto}}}}{{Q_{{entra}}}} = \dfrac{{{wnet:.1f}}}{{{qin:.1f}}} = {wnet / qin:.5f},\qquad 1-8^{{-0.4}} = 1-{r ** (1 - gm):.5f} = {eta:.5f}"]),
        ],
        numeric_steps=[
            ("Qué calcula el código", "El trabajo de cada adiabática se obtiene resolviendo $dp/dV=-\\gamma p/V$ con RK4 ($N=800$ pasos) e integrando $p$ con Simpson; las isocoras no realizan trabajo. $Q_{entra}$ viene de la isocora $B\\to C$.",
             [rf"h_{{AB}}=\dfrac{{V_B-V_A}}{{800}} = {sci(h, 4)}\ \mathrm{{m^3}}\ (\text{{negativo: compresión}}),\qquad p(V_A)={pa:.1f}\ \mathrm{{Pa}}"]),
            ("RK4: pendientes del primer paso", "$f(V,p)=-\\gamma p/V$ con $\\gamma=1.4$.",
             [rf"k_1 = -\dfrac{{(1.4)({pa:.1f})}}{{{va}}} = {sci(k1[0], 5)}\ \mathrm{{Pa/m^3}}",
              rf"k_2 = {sci(k2[0], 5)},\qquad k_3 = {sci(k3[0], 5)},\qquad k_4 = {sci(k4[0], 5)}"]),
            ("Primer valor de presión", "Se combinan con pesos $1,2,2,1$ y se compara con la adiabata exacta.",
             [rf"p_1 = {pa:.1f}+\dfrac{{{sci(h, 4)}}}{{6}}\,(k_1+2k_2+2k_3+k_4) = {y1[0]:.4f}\ \mathrm{{Pa}}",
              rf"p_{{exacto}} = p_A\left(\dfrac{{V_A}}{{V_A+h}}\right)^{{1.4}} = {pex:.4f}\ \mathrm{{Pa}},\qquad \text{{dif.}}={sci(abs(y1[0] - pex), 2)}\ \mathrm{{Pa}}"]),
            ("Final de la adiabática A→B", "Tras 800 pasos la presión llega al valor esperado de $B$.",
             [rf"p_B^{{num}} = {p_ab[-1]:.3f}\ \mathrm{{Pa}}\quad(\text{{analítico }}{pb:.3f})"]),
            ("Simpson sobre cada tramo", "Se integra $\\int p\\,dV$ con $h=(V_B-V_A)/800$ en $A\\to B$ y $(V_A-V_B)/800$ en $C\\to D$ (pesos $1,4,2,4,\\ldots,4,1$).",
             [rf"W_{{AB}} = {wab:.2f}\ \mathrm{{J}}\ (\text{{trabajo sobre el gas}}),\qquad W_{{CD}} = {wcd:.2f}\ \mathrm{{J}}"]),
            ("Trabajo neto y eficiencia", "Se suman los dos tramos y se divide por el calor de la isocora.",
             [rf"W_{{neto}}^{{num}} = {wab:.2f}+{wcd:.2f} = {o['W_net']:.2f}\ \mathrm{{J}},\qquad Q_{{entra}} = {o['Q_in']:.2f}\ \mathrm{{J}}",
              rf"\eta^{{num}} = \dfrac{{{o['W_net']:.2f}}}{{{o['Q_in']:.2f}}} = {o['eta']:.7f}"]),
            ("Error", "Comparación con el resultado analítico.",
             [rf"\eta^{{anal}} = {eta:.7f},\qquad |\Delta\eta| = {sci(abs(o['eta'] - eta), 2)};\qquad |\Delta W| = {sci(abs(o['W_net'] - eta * o['Q_in']), 2)}\ \mathrm{{J}}"]),
        ],
    )


# =============================================================== GasPressure
def _pressure() -> dict:
    kb, m, T, box = c24.KB, c24.M_HE, c24.T_GAS, c24.BOX
    n, dur, dt = 4000, 2.0e-8, 2.0e-12
    steps = int(round(dur / dt))
    V = box**3
    p_an = c24.pressure_analytic(n)
    p_sim, t_sim = c24.pressure_simulation(n=n)
    sigma = np.sqrt(kb * T / m)
    rng = np.random.default_rng(21)
    v = rng.normal(0.0, sigma, size=(n, 3))
    x = rng.uniform(0, box, size=(n, 3))
    v0, x0 = v[0].copy(), x[0].copy()
    # replica 1D del movimiento en x (la pared x=L solo depende de x)
    vx, xx = v[:, 0].copy(), x[:, 0].copy()
    hits, imp = 0, 0.0
    for _ in range(steps):
        xx += vx * dt
        hi = xx > box
        hits += int(hi.sum())
        imp += float(np.sum(2 * m * vx[hi]))
        xx[hi] = 2 * box - xx[hi]
        vx[hi] *= -1
        lo = xx < 0
        xx[lo] *= -1
        vx[lo] *= -1
    p_rep = imp / (dur * box**2)
    v2m = float(np.mean(np.sum(v**2, axis=1)))
    p_t = c24.pressure_analytic(n, t_sim)
    return dict(
        explanation=[
            "La presión de un gas es el efecto promedio de innumerables choques de moléculas contra las paredes. Cada rebote elástico contra una pared perpendicular a $x$ invierte la componente $v_x$ y entrega a la pared un impulso $2mv_x$. La fuerza sobre la pared es la suma de estos impulsos por unidad de tiempo.",
            "Si una molécula rebota entre dos paredes separadas $L$, golpea una de ellas cada $2L/v_x$ y ejerce en promedio una fuerza $mv_x^2/L$. Sumando sobre las $N$ moléculas y dividiendo por el área $L^2$ se obtiene $p=\\dfrac{N m\\langle v_x^2\\rangle}{V}$, y como el movimiento es isótropo, $\\langle v_x^2\\rangle=\\tfrac13\\langle v^2\\rangle$.",
            "Comparando con $pV=NkT$ resulta la interpretación cinética de la temperatura: $\\tfrac12 m\\langle v^2\\rangle=\\tfrac32kT$. La temperatura mide la energía cinética media de traslación por molécula, no la de un objeto macroscópico.",
            "Para el helio a 300 K, $\\sqrt{kT/m}\\approx 790$ m/s por componente. En una caja de 1 µm una molécula cruza en poco más de un nanosegundo, de modo que en 20 ns cada una rebota decenas de veces y la presión se promedia bien aun con solo 4000 átomos, aunque con fluctuaciones estadísticas de algunos por ciento.",
        ],
        analytic_steps=[
            ("Una molécula entre dos paredes", "Con rebote elástico el impulso por choque es $2m|v_x|$ y los choques con la pared derecha ocurren cada $2L/|v_x|$.",
             [r"\bar F_1 = \dfrac{2m|v_x|}{2L/|v_x|} = \dfrac{m\,v_x^2}{L}"]),
            ("Sumar N moléculas y dividir por el área", "La presión es la fuerza promedio total entre $L^2$ y se usa $V=L^3$.",
             [r"p = \dfrac{1}{L^2}\sum_{i=1}^{N}\dfrac{m\,v_{x,i}^2}{L} = \dfrac{N\,m\langle v_x^2\rangle}{V}"]),
            ("Isotropía", "En un gas en equilibrio ninguna dirección es especial, así que $\\langle v_x^2\\rangle=\\langle v^2\\rangle/3$.",
             [r"p = \dfrac13\,\dfrac{N}{V}\,m\langle v^2\rangle = \dfrac23\,\dfrac NV\,\langle K\rangle"]),
            ("Relación con la temperatura", "La energía cinética media de traslación es $\\tfrac32kT$, por lo que se obtiene la ley del gas ideal.",
             [r"\langle K\rangle = \tfrac32 kT\ \Rightarrow\ p = \dfrac{NkT}{V}"]),
            ("Datos", "Con $N=4000$, $k=1.381\\times10^{-23}$ J/K, $T=300$ K y $V=(10^{-6}\\ \\mathrm{m})^3 = 10^{-18}\\ \\mathrm{m^3}$.",
             [rf"NkT = ({n})({sci(kb, 6)})(300) = {sci(n * kb * T, 5)}\ \mathrm{{J}}"]),
            ("Presión esperada", "Se divide por el volumen.",
             [rf"p = \dfrac{{{sci(n * kb * T, 5)}\ \mathrm{{J}}}}{{{sci(V, 1)}\ \mathrm{{m^3}}}} = {p_an:.3f}\ \mathrm{{Pa}}"]),
            ("Rapidez típica por componente", "Cada componente de la velocidad es normal con desviación $\\sigma=\\sqrt{kT/m}$ ($m=6.646\\times10^{-27}$ kg).",
             [rf"\sigma = \sqrt{{\dfrac{{({sci(kb, 6)})(300)}}{{{sci(m, 4)}}}}} = {sigma:.2f}\ \mathrm{{m/s}}"]),
        ],
        numeric_steps=[
            ("Esquema de la simulación", "Se simulan $N=4000$ átomos independientes (no chocan entre sí) en un cubo de $L=1\\ \\mu\\mathrm{m}$ con paso $\\Delta t=2$ ps durante $20$ ns, es decir, $10^4$ pasos. En cada paso se avanza $x\\leftarrow x+v\\Delta t$ y se refleja en las paredes.",
             [rf"\Delta t = {sci(dt, 2)}\ \mathrm{{s}},\qquad \text{{pasos}} = \dfrac{{{sci(dur, 2)}}}{{{sci(dt, 2)}}} = {steps},\qquad \sigma\Delta t = {sci(sigma * dt, 3)}\ \mathrm{{m}}\ll L"]),
            ("Condiciones iniciales (semilla 21)", "Las velocidades se sortean con la distribución normal de desviación $\\sigma$ y las posiciones de modo uniforme en el cubo. Para la molécula número 1:",
             [rf"\vec v_1 = ({sci(v0[0], 4)},\,{sci(v0[1], 4)},\,{sci(v0[2], 4)})\ \mathrm{{m/s}},\qquad x_1 = {sci(x0[0], 4)}\ \mathrm{{m}}"]),
            ("Primer paso de esa molécula", "Se avanza la posición y se verifica si cruzó la pared $x=L$ (solo entonces se acumula impulso). Como $|v_x|\\Delta t\\sim 10^{-9}$ m y $L=10^{-6}$ m, los rebotes son raros por paso.",
             [rf"x_1\to x_1+v_x\Delta t = {sci(x0[0], 4)}+({sci(v0[0], 4)})({sci(dt, 2)}) = {sci(x0[0] + v0[0] * dt, 5)}\ \mathrm{{m}}",
              rf"\text{{¿}}x>L={sci(box, 1)}\text{{?}}\ \ {'\\text{sí: impulso }2mv_x' if x0[0] + v0[0] * dt > box else '\\text{no}'}"]),
            ("Regla de rebote", "Si $x>L$, se suma el impulso $2mv_x$ a la pared, se refleja la posición ($x\\to2L-x$) y se invierte $v_x$. En la pared $x=0$ solo se refleja, sin contarla.",
             [rf"\text{{impulso por choque típico }}2m\sigma = 2({sci(m, 4)})({sigma:.1f}) = {sci(2 * m * sigma, 3)}\ \mathrm{{kg\,m/s}}"]),
            ("Impulso total y presión", "Tras los $10^4$ pasos se obtiene el impulso total en la pared y se divide por el tiempo y el área $L^2$.",
             [rf"\text{{choques en }}x=L:\ {hits},\qquad J_{{tot}} = \sum 2mv_x = {sci(imp, 4)}\ \mathrm{{kg\,m/s}}",
              rf"p_{{sim}} = \dfrac{{J_{{tot}}}}{{t\,L^2}} = \dfrac{{{sci(imp, 4)}}}{{({sci(dur, 2)})({sci(box**2, 1)})}} = {p_rep:.3f}\ \mathrm{{Pa}}"]),
            ("Temperatura medida", "Como la muestra es finita, la energía cinética media de las 4000 moléculas no es exactamente $\\tfrac32kT$. La simulación estima $T=m\\langle v^2\\rangle/(3k)$.",
             [rf"\langle v^2\rangle = {sci(v2m, 4)}\ \mathrm{{m^2/s^2}},\qquad T_{{sim}} = \dfrac{{m\langle v^2\rangle}}{{3k}} = {t_sim:.2f}\ \mathrm{{K}}"]),
            ("Comparar con p = NkT/V", "Se usa la temperatura medida, ya que lo que se verifica es la ley $p=NkT/V$ con la energía real de la muestra.",
             [rf"p_{{NkT/V}}(T_{{sim}}) = {p_t:.3f}\ \mathrm{{Pa}},\qquad p_{{sim}} = {p_sim:.3f}\ \mathrm{{Pa}},\qquad \text{{dif. relativa}} = {abs(p_sim - p_t) / p_t * 100:.2f}\,\%"]),
        ],
    )


# =============================================================== MaxwellBoltzmann
def _maxwell() -> dict:
    kb, m, T = c24.KB, c24.M_N2, c24.T_GAS
    an = c24.speeds_analytic()
    sp = c24.speeds_sampled()
    q = c24.moments_by_quadrature()
    a = m / (2 * np.pi * kb * T)
    sigma = np.sqrt(kb * T / m)
    vmp, vm, vr = float(an["v_mp"]), float(an["v_mean"]), float(an["v_rms"])
    rng = np.random.default_rng(8)
    comp = rng.normal(0.0, sigma, size=(400_000, 3))
    c5 = comp[:5]
    s5 = np.linalg.norm(c5, axis=1)
    rows = NL.join(rf"{i + 1} & {c5[i, 0]:.1f} & {c5[i, 1]:.1f} & {c5[i, 2]:.1f} & {s5[i]:.1f}" for i in range(5))
    nq = 4000
    h = 6000 / nq
    vq = np.array([0.0, h, 2 * h, 3 * h, 4 * h])
    fq = c24.mb_pdf(vq)
    return dict(
        explanation=[
            "Las moléculas de un gas no tienen todas la misma rapidez: los choques las redistribuyen continuamente y, en equilibrio a temperatura $T$, la fracción con rapidez entre $v$ y $v+dv$ es $f(v)\\,dv$ con la distribución de Maxwell-Boltzmann. Su forma resulta del producto de un factor de Boltzmann $e^{-mv^2/2kT}$ (las energías altas son improbables) y del volumen del espacio de velocidades, $4\\pi v^2$ (hay muchas más direcciones disponibles cuando $v$ es grande).",
            "Como la distribución es asimétrica, el valor más probable $v_p$ (el máximo), la rapidez media $\\bar v$ y la rapidez cuadrática media $v_{rms}$ son distintos y cumplen $v_p<\\bar v<v_{rms}$. La $v_{rms}$ es la que se relaciona con la energía cinética media, $\\tfrac12mv_{rms}^2=\\tfrac32kT$.",
            "Las tres rapideces escalan como $\\sqrt{T/m}$: calentar el gas o usar moléculas más ligeras las desplaza hacia valores mayores. Para N$_2$ a 300 K están en el rango de 420 a 520 m/s, comparable a la rapidez del sonido en el aire.",
            "Una propiedad útil para simular: cada componente cartesiana de la velocidad es normal con desviación $\\sqrt{kT/m}$ e independiente de las otras. La rapidez de una molécula es entonces la norma de un vector de tres normales, y su histograma reproduce $f(v)$.",
        ],
        analytic_steps=[
            ("Parámetros", "Para N$_2$, $m=4.652\\times10^{-26}$ kg, y $T=300$ K. Se define $\\alpha=m/2kT$, que aparece en el exponente.",
             [rf"2kT/m = \dfrac{{2({sci(kb, 6)})(300)}}{{{sci(m, 4)}}} = {2 * kb * T / m:.2f}\ \mathrm{{m^2/s^2}}"]),
            ("Rapidez más probable", "Se deriva $f(v)$ respecto de $v$ y se iguala a cero; $d/dv\\,[v^2e^{-mv^2/2kT}]=0$ da $v_p$.",
             [r"v_p=\sqrt{\dfrac{2kT}{m}}", rf"v_p = \sqrt{{{2 * kb * T / m:.2f}}} = {vmp:.1f}\ \mathrm{{m/s}}"]),
            ("Rapidez media", "Se integra $\\bar v=\\int_0^\\infty v f(v)\\,dv$ (integral gaussiana tipo $\\int v^3e^{-av^2}dv$).",
             [r"\bar v=\sqrt{\dfrac{8kT}{\pi m}}", rf"\bar v = \sqrt{{\dfrac{{8({sci(kb, 6)})(300)}}{{\pi\,({sci(m, 4)})}}}} = \sqrt{{{8 * kb * T / (np.pi * m):.2f}}} = {vm:.1f}\ \mathrm{{m/s}}"]),
            ("Rapidez cuadrática media", "De $\\langle v^2\\rangle=\\int v^2f\\,dv=3kT/m$, que es la equipartición de energía.",
             [r"v_{rms}=\sqrt{\dfrac{3kT}{m}}", rf"v_{{rms}} = \sqrt{{{3 * kb * T / m:.2f}}} = {vr:.1f}\ \mathrm{{m/s}}"]),
            ("Orden y cociente", "Las tres rapideces mantienen proporciones fijas independientes de $T$ y $m$.",
             [rf"v_p:\bar v:v_{{rms}} = 1 : {vm / vmp:.4f} : {vr / vmp:.4f},\qquad v_p<\bar v<v_{{rms}}"]),
            ("Valor máximo de la densidad", "Se evalúa $f(v_p)$ para la normalización (unidades de s/m); el área total debe ser 1.",
             [rf"f(v)=4\pi\left(\dfrac{{m}}{{2\pi kT}}\right)^{{3/2}}v^2e^{{-mv^2/2kT}},\qquad f(v_p) = {sci(float(c24.mb_pdf(vmp)), 4)}\ \mathrm{{s/m}}"]),
        ],
        numeric_steps=[
            ("Método 1: muestreo", "Para cada molécula se sortean tres componentes normales con desviación $\\sigma=\\sqrt{kT/m}$ y se calcula $v=\\sqrt{v_x^2+v_y^2+v_z^2}$. Con $N=400\\,000$ y semilla 8, los promedios aproximan los momentos de $f$.",
             [rf"\sigma = \sqrt{{kT/m}} = {sigma:.2f}\ \mathrm{{m/s}},\qquad N=400000"]),
            ("Primeras cinco moléculas", "Las componentes (en m/s) y la rapidez de las primeras muestras:",
             [rf"\begin{{array}}{{c|c|c|c|c}} i & v_x & v_y & v_z & v \\ \hline {rows} \end{{array}}"]),
            ("Promedio con pocas muestras", "Con solo cinco moléculas el promedio es muy burdo, pero ilustra la idea.",
             [rf"\bar v_5 = \dfrac{{1}}{{5}}\sum v_i = {s5.mean():.1f}\ \mathrm{{m/s}},\qquad v_{{rms,5}} = \sqrt{{\dfrac{{1}}{{5}}\sum v_i^2}} = {np.sqrt((s5**2).mean()):.1f}\ \mathrm{{m/s}}"]),
            ("Promedios con N = 400 000", "El error estadístico baja como $1/\\sqrt N$ (unas 600 veces menos que con una muestra).",
             [rf"\bar v_{{mues}} = {sp['v_mean']:.2f}\ \mathrm{{m/s}}\ (\text{{anal. }}{vm:.2f}),\qquad v_{{rms,mues}} = {sp['v_rms']:.2f}\ \mathrm{{m/s}}\ (\text{{anal. }}{vr:.2f})",
              rf"\text{{moda del histograma (bin de 25 m/s): }}{sp['v_mp']:.1f}\ \mathrm{{m/s}}"]),
            ("Método 2: cuadratura de Simpson", f"Se integra $f(v)$ entre $0$ y $6000$ m/s con $N=4000$, $h={h:.2f}$ m/s: normalización, $\\int vf\\,dv$ y $\\int v^2f\\,dv$. Los primeros valores de $f$ (en 1/(m/s)):",
             [rf"f(0)={sci(fq[0], 3)},\ f(h)={sci(fq[1], 4)},\ f(2h)={sci(fq[2], 4)},\ f(3h)={sci(fq[3], 4)},\ f(4h)={sci(fq[4], 4)}"]),
            ("Fórmula de Simpson para el primer momento", "Con pesos $1,4,2,4,\\ldots,4,1$ sobre el integrando $vf(v)$; los primeros términos de la suma son pequeños porque $f\\propto v^2$ cerca del origen.",
             [rf"\bar v\approx\dfrac h3\left[0\cdot f_0+4(h f_1)+2(2hf_2)+4(3hf_3)+\ldots\right],\qquad 4hf_1 = {sci(4 * vq[1] * fq[1], 4)}"]),
            ("Resultados de la cuadratura", "Los momentos obtenidos y sus errores respecto de las fórmulas cerradas.",
             [rf"\int f\,dv = {q['norm']:.8f},\qquad \bar v_{{cuad}} = {q['v_mean']:.4f}\ \mathrm{{m/s}},\qquad v_{{rms,cuad}} = {q['v_rms']:.4f}\ \mathrm{{m/s}}",
              rf"|\bar v_{{cuad}}-\bar v| = {sci(abs(q['v_mean'] - vm), 2)}\ \mathrm{{m/s}},\qquad |\bar v_{{mues}}-\bar v| = {abs(sp['v_mean'] - vm):.2f}\ \mathrm{{m/s}}"]),
        ],
    )


# =============================================================== CarnotCycle
def _carnot() -> dict:
    R, n, g = c25.R, c25.N_MOL, c25.GAMMA
    th, tc = c25.TH, c25.TC
    st = c25.carnot_states()
    o = c25.carnot_numeric()
    eta = c25.carnot_efficiency_analytic()
    (pa, va), (pb, vb), (pc, vc), (pd, vd) = st["A"], st["B"], st["C"], st["D"]
    qh = n * R * th * np.log(vb / va)
    qc = n * R * tc * np.log(vc / vd)
    w = qh - qc
    nn = 2000
    vv = np.linspace(va, vb, 5)
    pp = n * R * th / vv
    s4 = simpson(pp, vv)
    h = (vc - vb) / nn
    f = lambda v, y: np.array([-g * y[0] / v])  # noqa: E731
    k1, k2, k3, k4, y1 = _rk4(f, vb, [pb], h)
    pex = pb * (vb / (vb + h)) ** g
    wn = o["W_net"]
    tb = NL.join(rf"{i} & {vv[i] * 1000:.3f} & {pp[i] / 1e6:.4f}" for i in range(5))
    return dict(
        explanation=[
            "El ciclo de Carnot es el ciclo reversible entre dos focos a temperaturas $T_h$ y $T_c$: dos isotermas (en las que el gas intercambia calor con los focos) unidas por dos adiabáticas (en las que cambia de temperatura sin intercambiar calor). Es el ciclo más eficiente posible entre esos dos focos.",
            "En la isoterma caliente el gas se expande y absorbe $Q_h=nRT_h\\ln(V_B/V_A)$; en la isoterma fría se comprime y cede $Q_c$. Las adiabáticas ($pV^\\gamma=\\text{cte}$) fuerzan que los volúmenes cumplan $V_C/V_D=V_B/V_A$, por lo que $Q_c/Q_h=T_c/T_h$.",
            "De ahí la eficiencia de Carnot $\\eta_C=1-T_c/T_h$, que depende solo de las temperaturas y no del gas ni de la máquina. Para que $\\eta$ sea alta hay que aumentar el cociente de temperaturas; ninguna máquina que opere entre los mismos focos puede superarla.",
            "Con $500$ K y $300$ K resulta $\\eta=40\\,\\%$: de cada 100 J absorbidos solo 40 J se convierten en trabajo. Es un límite teórico (proceso cuasiestático, sin fricción) que los motores reales no alcanzan.",
        ],
        analytic_steps=[
            ("Estado A y B (isoterma caliente)", "A $T_h=500$ K el gas pasa de $V_A=1$ L a $V_B=2$ L; la presión sale de $p=nRT/V$.",
             [rf"p_A = \dfrac{{(1)({R})({th:.0f})}}{{{va}}} = {pa:.0f}\ \mathrm{{Pa}},\qquad p_B = \dfrac{{nRT_h}}{{V_B}} = {pb:.0f}\ \mathrm{{Pa}}"]),
            ("Calor absorbido", "En la isoterma $\\Delta U=0$, por lo que $Q_h=W_{AB}=nRT_h\\ln(V_B/V_A)$.",
             [rf"Q_h = (1)({R})({th:.0f})\ln 2 = {n * R * th:.1f}\,({np.log(2):.5f}) = {qh:.2f}\ \mathrm{{J}}"]),
            ("Volúmenes de las adiabáticas", "Con $T V^{\\gamma-1}=\\text{cte}$ y $\\gamma=5/3$ ($1/(\\gamma-1)=1.5$): $V_C=V_B(T_h/T_c)^{1.5}$ y $V_D=V_A(T_h/T_c)^{1.5}$.",
             [rf"V_C = ({vb})\left(\dfrac{{500}}{{300}}\right)^{{1.5}} = {vc * 1e3:.4f}\ \mathrm{{L}},\qquad V_D = {vd * 1e3:.4f}\ \mathrm{{L}}"]),
            ("Calor cedido", "En la isoterma fría ($C\\to D$) el gas se comprime y cede $Q_c=nRT_c\\ln(V_C/V_D)$, con el mismo cociente de volúmenes 2.",
             [rf"Q_c = (1)({R})({tc:.0f})\ln\dfrac{{{vc * 1e3:.4f}}}{{{vd * 1e3:.4f}}} = {qc:.2f}\ \mathrm{{J}}"]),
            ("Trabajo neto", "Las adiabáticas se cancelan entre sí; el trabajo neto es $Q_h-Q_c$.",
             [rf"W = Q_h-Q_c = {qh:.2f}-{qc:.2f} = {w:.2f}\ \mathrm{{J}}"]),
            ("Eficiencia", "Por definición y con la fórmula de Carnot.",
             [rf"\eta = \dfrac{{W}}{{Q_h}} = \dfrac{{{w:.2f}}}{{{qh:.2f}}} = {w / qh:.4f},\qquad \eta_C = 1-\dfrac{{300}}{{500}} = {eta:.4f}"]),
            ("Verificación con Q/T", "Una comprobación de reversibilidad: la suma de $Q/T$ del ciclo debe anularse.",
             [rf"\dfrac{{Q_h}}{{T_h}}-\dfrac{{Q_c}}{{T_c}} = \dfrac{{{qh:.2f}}}{{500}}-\dfrac{{{qc:.2f}}}{{300}} = {qh / th - qc / tc:.2e}".replace("e-", r"\times10^{-").replace("e+", r"\times10^{") + "}" if abs(qh / th - qc / tc) > 1e-9 else rf"\dfrac{{Q_h}}{{T_h}}-\dfrac{{Q_c}}{{T_c}} = \dfrac{{{qh:.2f}}}{{500}}-\dfrac{{{qc:.2f}}}{{300}} = 0"]),
        ],
        numeric_steps=[
            ("Qué hace el código", "Se calcula el trabajo de cada tramo y se suman. En las isotermas se integra $p=nRT/V$ con Simpson; en las adiabáticas se resuelve $dp/dV=-\\gamma p/V$ con RK4 y luego se integra con Simpson. Cada tramo usa $N=2000$ subintervalos.",
             [r"W_{neto}=W_{AB}+W_{BC}+W_{CD}+W_{DA},\qquad Q_h=W_{AB}"]),
            ("Simpson en A→B con N = 4", "Para ver la mecánica: $h=(V_B-V_A)/4=2.5\\times10^{-4}$ m³ y pesos $1,4,2,4,1$.",
             [rf"\begin{{array}}{{c|c|c}} i & V_i\ (\mathrm{{L}}) & p_i\ (\mathrm{{MPa}}) \\ \hline {tb} \end{{array}}",
              rf"W_{{AB}}^{{N=4}} = \dfrac{{{(vb - va) / 4:.5f}}}{{3}}\left[{pp[0]:.0f}+4({pp[1]:.0f})+2({pp[2]:.0f})+4({pp[3]:.0f})+{pp[4]:.0f}\right] = {s4:.3f}\ \mathrm{{J}}\quad(\text{{exacto }}{qh:.3f})"]),
            ("Simpson en A→B con N = 2000", "Con el N real el error es mínimo, y es el valor de $Q_h$ del código.",
             [rf"Q_h^{{num}} = {o['Q_h']:.4f}\ \mathrm{{J}}\quad(\text{{anal. }}{qh:.4f}),\qquad |\Delta Q_h| = {sci(abs(o['Q_h'] - qh), 2)}\ \mathrm{{J}}"]),
            ("RK4 en la adiabática B→C: primer paso", "Parte de $p_B$ con $h=(V_C-V_B)/2000$ y $f=-\\gamma p/V$.",
             [rf"h = {sci(h, 4)}\ \mathrm{{m^3}},\qquad k_1 = -\dfrac{{\gamma p_B}}{{V_B}} = {sci(k1[0], 5)}\ \mathrm{{Pa/m^3}}",
              rf"k_2={sci(k2[0], 5)},\ k_3={sci(k3[0], 5)},\ k_4={sci(k4[0], 5)}"]),
            ("Presión tras el primer paso", "Se compara con la adiabata exacta.",
             [rf"p_1 = p_B+\dfrac h6(k_1+2k_2+2k_3+k_4) = {y1[0]:.5f}\ \mathrm{{Pa}},\qquad p_{{exacto}} = {pex:.5f}\ \mathrm{{Pa}}"]),
            ("Resto del ciclo", "Se hace lo mismo en $C\\to D$ (isoterma de compresión, trabajo negativo) y $D\\to A$ (adiabática de compresión, trabajo negativo) y se suman los cuatro tramos.",
             [rf"W_{{neto}}^{{num}} = {wn:.4f}\ \mathrm{{J}},\qquad W^{{anal}} = \eta_C Q_h = {eta * o['Q_h']:.4f}\ \mathrm{{J}},\qquad |\Delta W| = {sci(abs(wn - eta * o['Q_h']), 2)}\ \mathrm{{J}}"]),
            ("Cierre del ciclo y eficiencia", "El RK4 de $D\\to A$ debe volver a $p_A$ si el ciclo cierra; la eficiencia numérica se compara con $1-T_c/T_h$.",
             [rf"p_A^{{fin}} = {o['pa_end'] / 1e6:.6f}\ \mathrm{{MPa}}\ \ (p_A={o['pa'] / 1e6:.6f}),\qquad \eta^{{num}} = \dfrac{{{wn:.3f}}}{{{o['Q_h']:.3f}}} = {o['eta']:.7f}\ \ (\eta_C={eta:.7f})"]),
        ],
    )


# =============================================================== EntropyProduction
def _entropy() -> dict:
    m, c, t1, t2 = 1.0, 4186.0, 20.0, 80.0
    T1, T2 = t1 + 273.15, t2 + 273.15
    sa, sn = c25.heating_entropy_analytic(), c25.heating_entropy_numeric()
    nn = 600
    h = (T2 - T1) / nn
    tt = np.linspace(T1, T2, 3)
    ff = m * c / tt
    s2 = simpson(ff, tt)
    C1, C2 = c22.M1 * c22.C1, c22.M2 * c22.C2
    ta, tw = c22.T1_0, c22.T2_0
    tf = c22.final_temperature_analytic()
    tfk = tf + 273.15
    sb = c25.contact_entropy_analytic()
    sbn = c25.contact_entropy_numeric()
    dS1, dS2 = C1 * np.log(tfk / (ta + 273.15)), C2 * np.log(tfk / (tw + 273.15))
    hh = 120.0 / 6000
    H = c22.H
    rhs = lambda _t, y: np.array([-H * (y[0] - y[1]) / C1, H * (y[0] - y[1]) / C2,  # noqa: E731
                                  H * (y[0] - y[1]) ** 2 / ((y[0] + 273.15) * (y[1] + 273.15))])
    k1, k2, k3, k4, y1 = _rk4(rhs, 0.0, [ta, tw, 0.0], hh)
    return dict(
        explanation=[
            "La entropía $S$ es una función de estado que cuantifica la irreversibilidad. Se define por el calor intercambiado en un camino reversible: $dS=dQ_{rev}/T$. Aunque un proceso real sea irreversible, el $\\Delta S$ de un sistema se calcula imaginando un camino reversible entre los mismos estados, porque $S$ solo depende de los estados extremos.",
            "Al calentar agua con $dQ=mc\\,dT$, cada joule agregado a baja temperatura aumenta más la entropía que a alta temperatura (por el factor $1/T$), de donde $\\Delta S=mc\\ln(T_2/T_1)$. Siempre hay que usar kelvin: el logaritmo no tiene sentido con temperaturas en Celsius.",
            "La segunda ley establece que la entropía del universo no disminuye: $\\Delta S_{universo}\\ge0$, con igualdad solo si el proceso es reversible. Cuando dos cuerpos a distinta temperatura entran en contacto, el cuerpo caliente pierde entropía y el frío la gana, pero el aumento del frío siempre es mayor, porque el mismo calor se divide entre una temperatura menor.",
            "La producción de entropía durante el contacto es $dS/dt=h(T_1-T_2)^2/(T_1T_2)\\ge0$, positiva mientras haya diferencia de temperaturas y nula en el equilibrio. Este es un criterio cuantitativo de irreversibilidad: cuanto mayor la diferencia inicial, mayor el $\\Delta S$ generado.",
        ],
        analytic_steps=[
            ("a) Plantear dS = dQ/T", "Para 1 kg de agua con $c=4186\\ \\mathrm{J/(kg\\,K)}$ constante, $dQ_{rev}=mc\\,dT$. Se integra entre las temperaturas absolutas.",
             [r"\Delta S = \int_{T_1}^{T_2}\dfrac{mc\,dT}{T} = mc\ln\dfrac{T_2}{T_1}"]),
            ("a) Convertir a kelvin", "Se suman $273.15$ a cada temperatura.",
             [rf"T_1 = 20+273.15 = {T1:.2f}\ \mathrm{{K}},\qquad T_2 = 80+273.15 = {T2:.2f}\ \mathrm{{K}}"]),
            ("a) Evaluar", "Se calcula el cociente, su logaritmo y el producto.",
             [rf"\ln\dfrac{{{T2:.2f}}}{{{T1:.2f}}} = \ln({T2 / T1:.5f}) = {np.log(T2 / T1):.5f}",
              rf"\Delta S_a = (1)({c:.0f})({np.log(T2 / T1):.5f}) = {sa:.2f}\ \mathrm{{J/K}}"]),
            ("b) Estado final del contacto", "Con $C_1=m_1c_1=270$ J/K (aluminio a 150 °C) y $C_2=2093$ J/K (agua a 20 °C), la temperatura final es la de la escena de equilibrio térmico.",
             [rf"T_f = \dfrac{{C_1T_1+C_2T_2}}{{C_1+C_2}} = {tf:.4f}\ ^\circ\mathrm{{C}} = {tfk:.3f}\ \mathrm{{K}}"]),
            ("b) Entropía de cada cuerpo", "Cada cuerpo se lleva a $T_f$ por un camino reversible imaginario; el aluminio se enfría ($\\Delta S<0$) y el agua se calienta ($\\Delta S>0$).",
             [rf"\Delta S_1 = C_1\ln\dfrac{{T_f}}{{T_1}} = {C1:.1f}\ln\dfrac{{{tfk:.3f}}}{{{ta + 273.15:.2f}}} = {dS1:.3f}\ \mathrm{{J/K}}",
              rf"\Delta S_2 = C_2\ln\dfrac{{T_f}}{{T_2}} = {C2:.1f}\ln\dfrac{{{tfk:.3f}}}{{{tw + 273.15:.2f}}} = {dS2:.3f}\ \mathrm{{J/K}}"]),
            ("b) Entropía total", "La suma es positiva, como exige la segunda ley.",
             [rf"\Delta S_b = \Delta S_1+\Delta S_2 = ({dS1:.3f})+({dS2:.3f}) = {sb:.3f}\ \mathrm{{J/K}} > 0"]),
        ],
        numeric_steps=[
            ("a) Integrar 1/T con Simpson", f"El código evalúa $\\Delta S=\\int mc/T\\,dT$ con la regla de Simpson y $N=600$ subintervalos de ancho $h={h:.2f}$ K. Primero, la mecánica con $N=2$: $h=30$ K y pesos $1,4,1$.",
             [rf"T_0={tt[0]:.2f},\ T_1={tt[1]:.2f},\ T_2={tt[2]:.2f}\ \mathrm{{K}};\qquad f=\dfrac{{mc}}{{T}} = {ff[0]:.4f},\ {ff[1]:.4f},\ {ff[2]:.4f}\ \mathrm{{J/K^2}}",
              rf"\Delta S_{{N=2}} = \dfrac{{30}}{{3}}\left[{ff[0]:.4f}+4({ff[1]:.4f})+{ff[2]:.4f}\right] = {s2:.3f}\ \mathrm{{J/K}}"]),
            ("a) Con N = 600", "Se repite con los 601 nodos; el error de Simpson baja como $h^4$.",
             [rf"\Delta S_a^{{num}} = {sn:.6f}\ \mathrm{{J/K}},\qquad \Delta S_a^{{anal}} = {sa:.6f}\ \mathrm{{J/K}},\qquad \text{{error}} = {sci(abs(sn - sa), 2)}\ \mathrm{{J/K}}"]),
            ("b) Sistema de tres variables", "Para el contacto se integra con RK4 el vector $(T_1,T_2,S)$: las dos temperaturas del problema de equilibrio y la entropía generada $S$ con $\\dot S=h(T_1-T_2)^2/(T_1T_2)$ (temperaturas en K). Paso $h_t=120/6000=0.02$ s.",
             [rf"\vec y_0=({ta:.0f},\,{tw:.0f},\,0),\qquad \dot S_0 = \dfrac{{20(130)^2}}{{({ta + 273.15:.2f})({tw + 273.15:.2f})}} = {rhs(0, np.array([ta, tw, 0.0]))[2]:.4f}\ \mathrm{{W/K}}"]),
            ("b) Primer paso RK4", "Se calculan las cuatro pendientes de las tres componentes (K/s, K/s y W/K).",
             [rf"\vec k_1 = ({k1[0]:.5f},\ {k1[1]:.5f},\ {k1[2]:.5f})",
              rf"\vec k_2 = ({k2[0]:.5f},\ {k2[1]:.5f},\ {k2[2]:.5f}),\qquad \vec k_3 = ({k3[0]:.5f},\ {k3[1]:.5f},\ {k3[2]:.5f})",
              rf"\vec k_4 = ({k4[0]:.5f},\ {k4[1]:.5f},\ {k4[2]:.5f})"]),
            ("b) Combinar", "Se aplica $\\vec y_1=\\vec y_0+\\frac{h_t}{6}(\\vec k_1+2\\vec k_2+2\\vec k_3+\\vec k_4)$.",
             [rf"\vec y_1 = ({y1[0]:.5f},\ {y1[1]:.5f},\ {y1[2]:.6f})",
              rf"S_1 = {y1[2]:.6f}\ \mathrm{{J/K}}\ \text{{generados en el primer paso de }}0.02\ \mathrm{{s}}"]),
            ("b) Acumulación hasta el equilibrio", "El proceso se repite 6000 veces. La producción $\\dot S$ decae junto con $(T_1-T_2)^2$, con constante de tiempo $\\tau/2$, de modo que a los $120$ s ya casi no queda diferencia de temperaturas y $S$ ha saturado.",
             [rf"S(120\ \mathrm{{s}}) = {sbn:.6f}\ \mathrm{{J/K}}"]),
            ("b) Comparación y consistencia", "El resultado coincide con la suma analítica $C_1\\ln+C_2\\ln$ y es positivo, como exige la segunda ley.",
             [rf"\Delta S_b^{{anal}} = {sb:.6f}\ \mathrm{{J/K}},\qquad |\Delta S_b^{{num}}-\Delta S_b^{{anal}}| = {sci(abs(sbn - sb), 2)}\ \mathrm{{J/K}}"]),
        ],
    )


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
