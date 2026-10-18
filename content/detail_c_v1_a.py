"""Concepto y desarrollo analitico general: Volumen 1, capitulos 1 a 4 (bloque A).

Escenas: DimensionalAnalysis, ErrorPropagation, VectorAddition, VectorProducts,
InstantaneousVelocity, ConstantAcceleration, Projectile, CircularMotion.
Cada escena define `concept`: lista de (titulo, prosa, [ecuaciones TeX]).
Las cifras de la particularizacion se calculan con `physics/`.
"""
from __future__ import annotations

import numpy as np

from content.texfmt import fx, sci
from physics import ch01_measurement as c1
from physics import ch02_vectors as vec
from physics import ch03_motion_1d as m1d
from physics import ch04_motion_2d as c4


# ====================================================================== cap. 1
def _dimensional() -> dict:
    g, L, th0 = 9.81, 1.0, 0.01
    res = c1.pendulum_dimensionless_constant(th0)
    ex = res.exponents
    A = np.array([[0, 0, 0, 1], [0, 1, 1, 0], [1, 0, -2, 0]], dtype=float)  # filas M, L, T; columnas T, L, g, m
    rank = int(np.linalg.matrix_rank(A))
    nvar = A.shape[1]
    t0 = 2 * np.pi * np.sqrt(L / g)
    agm_a, agm_b = 1.0, np.cos(th0 / 2)
    for _ in range(30):
        agm_a, agm_b = (agm_a + agm_b) / 2, np.sqrt(agm_a * agm_b)
    t_exact = t0 / agm_a
    f_series = 1 + th0**2 / 16 + 11 * th0**4 / 3072
    k_num = res.k_numeric
    concept = [
        ("Qué se estudia y por qué importa",
         r"El análisis dimensional es la herramienta que permite saber de qué puede depender una magnitud antes de resolver ninguna ecuación de movimiento. Se usa en el diseño de modelos a escala (túneles de viento, canales hidráulicos), en la comprobación de fórmulas y en problemas donde la dinámica completa es desconocida, como la turbulencia o la explosión de una bomba. Su idea es que las leyes físicas no pueden depender del sistema de unidades elegido, y esa invariancia restringe de forma fuerte la forma funcional de cualquier relación. El péndulo simple sirve de ejemplo porque su dinámica es conocida y el resultado dimensional se puede contrastar con la solución exacta.",
         [r"\text{relación física}\ \Phi(q_1,\dots,q_n)=0\ \ \text{independiente de las unidades} \;\Longrightarrow\; \tilde\Phi(\Pi_1,\dots,\Pi_{n-r})=0"]),
        ("Magnitudes, dimensiones y convenciones",
         r"Se toman como dimensiones base la masa $\mathsf{M}$ (kg), la longitud $\mathsf{L}$ (m) y el tiempo $\mathsf{T}$ (s); toda magnitud mecánica tiene dimensión $[q]=\mathsf{M}^{\alpha}\mathsf{L}^{\beta}\mathsf{T}^{\gamma}$ con exponentes racionales. Un ángulo es adimensional (razón de longitudes), de modo que $\theta_0$ no entra en la matriz dimensional. Aquí $L\sim 1\ \mathrm{m}$, $g\simeq 9.81\ \mathrm{m/s^2}$ y $T\sim 2\ \mathrm{s}$ son los órdenes de magnitud relevantes. Los exponentes pueden ser negativos o fraccionarios; lo que nunca puede ocurrir es sumar magnitudes de dimensiones distintas.",
         [r"[L]=\mathsf{L},\quad [m]=\mathsf{M},\quad [g]=\mathsf{L}\mathsf{T}^{-2},\quad [T]=\mathsf{T},\quad [\theta_0]=1"]),
        ("Intuición y analogías",
         r"Una receta de cocina que dice «sumar 3 metros con 2 segundos» es absurda; lo mismo vale para una ecuación física. Cambiar de unidades es como cambiar el idioma de una receta: el resultado del plato no puede cambiar. De ahí sale que una ley solo puede depender de combinaciones sin dimensiones, igual que una razón de ingredientes no depende de si se mide en tazas o en gramos. En el péndulo la única escala de tiempo que se puede formar con $L$ y $g$ es $\sqrt{L/g}$; la masa no puede aportar tiempo porque ninguna otra variable tiene masa con la que cancelarla.",
         [r"\text{escala de tiempo única con } L,g:\quad \tau=\sqrt{L/g},\qquad [\tau]=\left(\mathsf{L}/\mathsf{L}\mathsf{T}^{-2}\right)^{1/2}=\mathsf{T}"]),
        ("Matriz dimensional y rango",
         r"Dadas $n$ variables $q_1,\dots,q_n$ que intervienen en el problema, se construye la matriz dimensional $D$ de tamaño (dimensiones base)$\times n$ cuya columna $j$ contiene los exponentes de $[q_j]$. Un producto $\prod_j q_j^{x_j}$ es adimensional si y solo si $D\,\vec x=\vec 0$, es decir, si $\vec x$ pertenece al núcleo de $D$. Por el teorema del rango, el número de grupos adimensionales independientes es $n-r$, con $r=\operatorname{rango}D$. Esto reduce el problema de $n$ variables a una relación entre $n-r$ números puros.",
         [r"D=\begin{pmatrix}0&0&0&1\\0&1&1&0\\1&0&-2&0\end{pmatrix}\ \begin{matrix}\mathsf{M}\\ \mathsf{L}\\ \mathsf{T}\end{matrix},\qquad \text{columnas}: (T,\,L,\,g,\,m)",
          r"\dim\ker D=n-r,\qquad D\,\vec x=\vec 0\;\Longleftrightarrow\;\prod_j q_j^{x_j}\ \text{adimensional}"]),
        ("Teorema Π de Buckingham",
         r"Toda ecuación dimensionalmente homogénea entre $n$ variables con $r$ dimensiones independientes puede reescribirse como una relación entre $n-r$ productos adimensionales. La demostración usa que el cambio de unidades $q_j\to\lambda_1^{\alpha_{1j}}\lambda_2^{\alpha_{2j}}\lambda_3^{\alpha_{3j}}\,q_j$ es un grupo de escalamiento de $r$ parámetros, y que una función invariante bajo ese grupo solo puede depender de los invariantes, que son los $\Pi_i$. Aquí $n=4$ (más el ángulo, que ya es un $\Pi$ por sí solo) y $r=3$, de modo que queda un único grupo con dimensiones, que contiene a $T$. La selección de variables es la hipótesis física crucial: omitir una variable relevante falsea el resultado, incluir una superflua solo añade grupos.",
         [rf"n={nvar},\quad r={rank}\;\Rightarrow\; n-r={nvar - rank}\ \text{{grupo}};\qquad \vec x\in\ker D:\ x_T=1,\; x_L=-\tfrac12,\; x_g=\tfrac12,\; x_m=0",
          r"\Pi_1=T\sqrt{\dfrac{g}{L}},\qquad \Pi_2=\theta_0,\qquad \Phi(\Pi_1,\Pi_2)=0\;\Longrightarrow\;\Pi_1=F(\theta_0)"]),
        ("Forma general del periodo",
         r"Despejando $T$ de la relación entre los grupos se obtiene que el periodo es la escala $\sqrt{L/g}$ multiplicada por una función adimensional de la amplitud. Esa función $F(\theta_0)$ no puede determinarse por argumentos dimensionales; contiene toda la dinámica. Obsérvese que la independencia de la masa no es un resultado de la mecánica sino de la estructura dimensional: sin otra magnitud con masa, $m$ no puede aparecer en un grupo. Cualquier ley de potencias $T=kL^ag^bm^c$ es el caso particular $F=k$ constante, válido cuando $F$ es casi constante (amplitudes pequeñas).",
         [r"T=\sqrt{\dfrac{L}{g}}\;F(\theta_0),\qquad F\ \text{adimensional, independiente de}\ L,g,m",
          r"T=kL^{a}g^{b}m^{c}\;\Rightarrow\;\begin{cases}\mathsf{M}: c=0\\ \mathsf{L}: a+b=0\\ \mathsf{T}: -2b=1\end{cases}\;\Rightarrow\; a=\tfrac12,\ b=-\tfrac12,\ c=0"]),
        ("Cálculo de la función F desde la conservación de la energía",
         r"El lagrangiano $\mathcal L=\tfrac12mL^2\dot\theta^2+mgL\cos\theta$ no depende explícitamente del tiempo, de modo que la energía $E=\tfrac12mL^2\dot\theta^2-mgL\cos\theta$ es constante (teorema de Noether para traslaciones temporales). Partiendo del reposo en $\theta_0$ se despeja $\dot\theta$ y, por simetría, el periodo es cuatro veces el tiempo de $0$ a $\theta_0$. El cambio de variable $\sin(\theta/2)=\sin(\theta_0/2)\sin\varphi$ convierte la integral en una integral elíptica completa de primera especie $K$, con módulo $\kappa=\sin(\theta_0/2)$. La masa se cancela en la energía, lo que confirma el resultado dimensional.",
         [r"\dfrac12 L^2\dot\theta^2=g L\,(\cos\theta-\cos\theta_0)\;\Rightarrow\;T=4\sqrt{\dfrac{L}{2g}}\int_0^{\theta_0}\dfrac{d\theta}{\sqrt{\cos\theta-\cos\theta_0}}",
          r"\sin\dfrac{\theta}{2}=\kappa\sin\varphi,\quad \kappa=\sin\dfrac{\theta_0}{2}\;\Rightarrow\;T=4\sqrt{\dfrac{L}{g}}\,K(\kappa),\qquad K(\kappa)=\int_0^{\pi/2}\dfrac{d\varphi}{\sqrt{1-\kappa^2\sin^2\varphi}}"]),
        ("Desarrollo en serie y casos límite",
         r"Para amplitudes pequeñas se desarrolla $K(\kappa)=\tfrac{\pi}{2}\left(1+\tfrac14\kappa^2+\tfrac{9}{64}\kappa^4+\dots\right)$ y $\kappa^2=\sin^2(\theta_0/2)\approx\theta_0^2/4$. El primer término da $F\to2\pi$, que identifica la constante que el análisis dimensional deja libre. La primera corrección es de orden $\theta_0^2$ y positiva: un péndulo más amplio oscila más despacio, pues la fuerza restauradora crece menos que linealmente. En el límite opuesto $\theta_0\to\pi$ el módulo $\kappa\to1$, $K$ diverge logarítmicamente y el periodo tiende a infinito: el péndulo tarda indefinidamente en abandonar la posición invertida inestable.",
         [r"F(\theta_0)=2\pi\left[1+\dfrac{\theta_0^{2}}{16}+\dfrac{11\,\theta_0^{4}}{3072}+\mathcal O(\theta_0^{6})\right]",
          r"F(\theta_0)=\dfrac{2\pi}{\operatorname{AGM}\!\left(1,\cos\tfrac{\theta_0}{2}\right)},\qquad \lim_{\theta_0\to0}F=2\pi,\qquad \lim_{\theta_0\to\pi}F=\infty"]),
        ("Resultado general",
         r"El periodo del péndulo simple de longitud $L$ en un campo $g$ con amplitud $\theta_0$ es la escala dimensional $\sqrt{L/g}$ por la función $F$ de la amplitud. En la aproximación armónica $F=2\pi$ y $k=2\pi$. Dimensionalmente es un tiempo, $\mathsf{T}=(\mathsf{L}/\mathsf{L}\mathsf{T}^{-2})^{1/2}$, y depende de $L$ crecientemente, de $g$ decrecientemente, y no de $m$.",
         [r"\boxed{\,T=2\pi\sqrt{\dfrac{L}{g}}\;\Big[1+\dfrac{\theta_0^{2}}{16}+\mathcal O(\theta_0^{4})\Big]\,}",
          r"[T]=\sqrt{\dfrac{\mathsf{L}}{\mathsf{L}\,\mathsf{T}^{-2}}}=\mathsf{T},\qquad \dfrac{\partial\ln T}{\partial\ln L}=\dfrac12,\quad \dfrac{\partial\ln T}{\partial\ln g}=-\dfrac12,\quad \dfrac{\partial\ln T}{\partial\ln m}=0"]),
        ("Particularización al problema",
         rf"Con $L=1$ m, $g=9.81\ \mathrm{{m/s^2}}$ y la amplitud de prueba $\theta_0=0.01$ rad, el sistema dimensional da exactamente los exponentes de la escena. La constante armónica es $2\pi$ y la corrección de amplitud es del orden de $6\times10^{{-6}}$, por lo que la constante medida numéricamente difiere de $2\pi$ solo en la sexta cifra. El valor exacto con la media aritmético-geométrica coincide con la serie hasta el orden mostrado.",
         [rf"a={fx(ex['L'], 4)},\quad b={fx(ex['g'], 4)},\quad c={fx(ex['m'], 4)}",
          rf"T_0=2\pi\sqrt{{\dfrac{{{L:.0f}}}{{{g}}}}}={fx(t0, 5)}\ \mathrm{{s}},\qquad F(\theta_0)/2\pi=1+\dfrac{{\theta_0^2}}{{16}}={fx(1 + th0**2 / 16, 8)}",
          rf"T_{{\mathrm{{exacto}}}}=\dfrac{{2\pi\sqrt{{L/g}}}}{{\operatorname{{AGM}}(1,\cos\theta_0/2)}}={fx(t_exact, 7)}\ \mathrm{{s}},\qquad k_{{num}}={fx(k_num, 6)}\ \ (2\pi={fx(2*np.pi, 6)})"]),
        ("Errores comunes y límites de validez",
         r"Un error frecuente es creer que el análisis dimensional prueba que la masa «no influye» en cualquier péndulo: solo lo muestra si no existe otra magnitud con masa (resistencia del aire, péndulo físico con densidad); con arrastre, aparece el cociente $m/b$. Otro error es olvidar que las constantes numéricas son invisibles, y que $F(\theta_0)$ puede variar mucho aunque sea adimensional. También se confunde un grupo $\Pi$ con una ley: la relación $\Pi_1=F(\Pi_2)$ debe fijarse con teoría o con experimentos. El método falla si el conjunto de variables es incompleto, y se reemplaza entonces por la dinámica explícita (ecuación de Euler-Lagrange) o por mediciones que determinan $F$.",
         [r"T\ne k\sqrt{L/g}\ \text{si}\ \theta_0=\mathcal O(1)\ \text{o hay disipación};\qquad T=\sqrt{L/g}\,F(\theta_0,\,m\sqrt{g/L}/b,\dots)"]),
    ]
    return dict(concept=concept)


def _error_prop() -> dict:
    m, r, hh = c1.CYLINDER["m"], c1.CYLINDER["r"], c1.CYLINDER["h"]
    an = c1.density_analytic()
    mc = c1.density_monte_carlo()
    rho = an["rho"]
    vol = np.pi * r.value**2 * hh.value
    rel = float(np.sqrt(an["rel_m"] ** 2 + an["rel_r"] ** 2 + an["rel_h"] ** 2))
    # sesgo de segundo orden del cociente: <rho> ~ rho (1 + 3 (sr/r)^2 + (sh/h)^2) para exponentes -2 y -1
    rr, rh = r.sigma / r.value, hh.sigma / hh.value
    bias = 3 * rr**2 + rh**2
    # peso (cuadrado) de cada contribucion
    tot = an["rel_m"] ** 2 + an["rel_r"] ** 2 + an["rel_h"] ** 2
    concept = [
        ("Qué se estudia y por qué importa",
         r"Toda cantidad medida es una variable aleatoria: el instrumento tiene resolución finita, el entorno fluctúa y el observador introduce sesgos. Reportar un número sin su incertidumbre carece de contenido científico, porque no permite comparar con una predicción teórica ni con otra medición. Cuando la magnitud de interés se calcula a partir de varias mediciones directas, la incertidumbre se transmite según la sensibilidad de la fórmula. Este problema aparece en metrología, ingeniería de tolerancias y en cualquier ajuste de datos.",
         [r"y=f(x_1,\dots,x_n),\qquad x_i=\bar x_i\pm\sigma_i\;\Longrightarrow\;y=\bar y\pm\sigma_y\ \ ?"]),
        ("Magnitudes y convenciones",
         r"Se denomina $\sigma_i$ a la desviación estándar de la variable $x_i$, que se interpreta como la incertidumbre estándar; la incertidumbre relativa es $\sigma_i/|\bar x_i|$ y es adimensional. Las unidades de $\sigma_i$ son las de $x_i$ y la covarianza $\sigma_{ij}=\operatorname{cov}(x_i,x_j)$ tiene unidades del producto. Una incertidumbre del $1\ \%$ es típica de mediciones con calibre; del orden de $10^{-6}$ en metrología de precisión. Es habitual reportar con una o dos cifras significativas en $\sigma$ y redondear el valor central a la misma posición decimal.",
         [r"\sigma_i^2=\langle(x_i-\bar x_i)^2\rangle,\qquad \sigma_{ij}=\langle(x_i-\bar x_i)(x_j-\bar x_j)\rangle,\qquad \rho_{ij}=\dfrac{\sigma_{ij}}{\sigma_i\sigma_j}\in[-1,1]"]),
        ("Intuición y analogías",
         r"Un paseo aleatorio ayuda a ver por qué los errores independientes se suman en cuadratura: cada paso aporta una distancia $\sigma$ en un sentido azaroso, y tras $N$ pasos el desplazamiento típico es $\sigma\sqrt N$, no $N\sigma$. Si los errores fueran todos del mismo signo (correlación $+1$) se sumarían linealmente, como si todos los pasos fueran en la misma dirección. La sensibilidad $\partial f/\partial x_i$ actúa como una palanca: una variable a la que $f$ es muy sensible amplifica su incertidumbre. En una potencia $x^p$ la palanca relativa es $p$, por lo que el radio (al cuadrado) pesa el doble.",
         [r"\sigma_{\sum}=\sqrt{\sigma_1^2+\dots+\sigma_N^2}\quad(\rho=0),\qquad \sigma_{\sum}=\sigma_1+\dots+\sigma_N\quad(\rho=+1)"]),
        ("Linealización de la función",
         r"Se desarrolla $f$ en serie de Taylor alrededor del vector de valores medios $\bar{\vec x}$. Esto es legítimo si las desviaciones $\delta x_i=x_i-\bar x_i$ son pequeñas frente a la escala de variación de $f$, de modo que los términos de segundo orden sean despreciables. Se mantiene el término lineal en las desviaciones, que es la proyección de las fluctuaciones sobre el gradiente de $f$. El valor medio de $\delta x_i$ es cero por definición, así que a este orden $\langle y\rangle=f(\bar{\vec x})$.",
         [r"y=f(\bar{\vec x}+\delta\vec x)=f(\bar{\vec x})+\sum_{i=1}^{n}\dfrac{\partial f}{\partial x_i}\bigg|_{\bar{\vec x}}\delta x_i+\mathcal O(\delta x^2)",
          r"\langle\delta x_i\rangle=0\;\Rightarrow\;\langle y\rangle=f(\bar{\vec x})+\mathcal O(\sigma^2)"]),
        ("Ley general de propagación",
         r"La varianza de $y$ es el valor esperado de $(y-\langle y\rangle)^2$, que a primer orden es el cuadrado de la forma lineal. Al desarrollar el cuadrado aparecen los términos diagonales $\sigma_i^2$ y los cruzados con covarianzas. En forma matricial es la forma cuadrática del gradiente con la matriz de covarianza $C$. Para variables independientes ($C$ diagonal) quedan solo las contribuciones en cuadratura; esta hipótesis debe verificarse, pues mediciones con el mismo instrumento pueden estar correlacionadas.",
         [r"\sigma_y^2=\Big\langle\Big(\sum_i f_i\,\delta x_i\Big)^2\Big\rangle=\sum_i f_i^2\sigma_i^2+2\sum_{i<j}f_if_j\,\sigma_{ij},\qquad f_i\equiv\dfrac{\partial f}{\partial x_i}",
          r"\sigma_y^2=\nabla f^{\mathsf T}\,C\,\nabla f,\qquad C_{ij}=\sigma_{ij};\qquad C\ \text{diagonal}\;\Rightarrow\;\sigma_y^2=\sum_i\left(\dfrac{\partial f}{\partial x_i}\sigma_i\right)^2"]),
        ("Producto de potencias: incertidumbres relativas",
         r"Si $f=C\prod_i x_i^{p_i}$ con $C$ constante exacta, la derivada logarítmica es $\partial\ln f/\partial x_i=p_i/x_i$, y por tanto $\partial f/\partial x_i=f\,p_i/x_i$. Sustituyendo en la ley general todas las incertidumbres aparecen en forma relativa, pesadas por el cuadrado del exponente. Esto explica que la densidad $\rho=m/(\pi r^2h)$, con exponentes $(1,-2,-1)$, tenga como factor $2$ la contribución del radio. El signo del exponente no importa para independientes, pero sí en presencia de correlación.",
         [r"\ln f=\ln C+\sum_i p_i\ln x_i\;\Rightarrow\;\dfrac{\delta f}{f}=\sum_i p_i\,\dfrac{\delta x_i}{x_i}",
          r"\left(\dfrac{\sigma_f}{f}\right)^2=\sum_i p_i^2\left(\dfrac{\sigma_i}{x_i}\right)^2+2\sum_{i<j}p_ip_j\,\dfrac{\sigma_{ij}}{x_ix_j}"]),
        ("Cota del peor caso, sesgo y validez",
         r"La cota conservadora corresponde a la suma lineal de los valores absolutos, $\sigma_y^{\max}=\sum_i|f_i|\sigma_i$, y se tiene siempre $\sigma_y\le\sigma_y^{\max}$ por la desigualdad triangular en $\mathbb R^n$ (norma $\ell_2\le\ell_1$). El primer término omitido del desarrollo da un sesgo del valor medio: para una función no lineal $\langle y\rangle\ne f(\bar{\vec x})$, con corrección $\tfrac12\sum_i f_{ii}\sigma_i^2$. La linealización es fiable si las incertidumbres relativas son mucho menores que uno; en caso contrario o si la distribución de $y$ no es aproximadamente normal, se recurre a Monte Carlo, que propaga la distribución completa sin derivar.",
         [r"\sigma_y=\|\vec w\|_2\le\|\vec w\|_1=\sum_i|f_i|\sigma_i,\qquad w_i=f_i\sigma_i",
          r"\langle y\rangle\simeq f(\bar{\vec x})+\dfrac12\sum_i\dfrac{\partial^2f}{\partial x_i^2}\,\sigma_i^2,\qquad \text{validez: } \sigma_i/|\bar x_i|\ll1"]),
        ("Resultado general",
         r"Para magnitudes independientes y una función diferenciable, la incertidumbre estándar del resultado es la raíz de la suma cuadrática de las contribuciones $|\partial f/\partial x_i|\,\sigma_i$, cada una con las unidades de $y$. Para productos de potencias se reduce a la suma cuadrática de incertidumbres relativas pesadas por los exponentes. Es una expresión homogénea: $\sigma_y$ tiene las unidades de $y$ y la forma relativa es adimensional.",
         [r"\boxed{\,\sigma_y=\sqrt{\sum_{i=1}^{n}\left(\dfrac{\partial f}{\partial x_i}\right)^{2}\sigma_i^{2}}\,}\qquad\Longrightarrow\qquad \boxed{\,\dfrac{\sigma_f}{f}=\sqrt{\sum_i p_i^2\left(\dfrac{\sigma_i}{x_i}\right)^2}\,}",
          r"[\sigma_y]=[y],\qquad \left[\dfrac{\sigma_f}{f}\right]=1"]),
        ("Particularización al problema",
         rf"Para la densidad del cilindro, $f=m/(\pi r^2 h)$ con exponentes $p_m=1$, $p_r=-2$, $p_h=-1$ y datos $m={m.value}\pm{m.sigma}$ g, $r={r.value}\pm{r.sigma}$ cm, $h={hh.value}\pm{hh.sigma}$ cm. El radio aporta el {an['rel_r']**2 / tot * 100:.1f} % de la varianza total, de modo que mejorar la medición de $r$ es la vía más eficaz para reducir $\sigma_\rho$. El sesgo de segundo orden es menor que la incertidumbre y el Monte Carlo confirma el resultado de primer orden.",
         [rf"\rho=\dfrac{{{m.value}}}{{\pi({r.value})^2({hh.value})}}={fx(rho, 4)}\ \mathrm{{g/cm^3}},\qquad V={fx(vol, 3)}\ \mathrm{{cm^3}}",
          rf"\dfrac{{\sigma_\rho}}{{\rho}}=\sqrt{{(1)^2({an['rel_m']:.5f})^2+(2)^2({rr:.5f})^2+(1)^2({rh:.5f})^2}}={rel:.5f},\qquad \sigma_\rho=\rho\cdot{rel:.5f}={fx(an['sigma_quadrature'], 4)}\ \mathrm{{g/cm^3}}",
          rf"\sigma_\rho^{{\max}}=\rho\,(0.00090+0.01000+0.00238)={fx(an['sigma_worst_case'], 4)}\ \mathrm{{g/cm^3}};\qquad \text{{Monte Carlo}}:\ \sigma_\rho={fx(mc['std'], 4)}\ \mathrm{{g/cm^3}}",
          rf"\text{{sesgo relativo}}\approx3\left(\dfrac{{\sigma_r}}{{r}}\right)^2+\left(\dfrac{{\sigma_h}}{{h}}\right)^2={sci(bias, 3)}\ll\dfrac{{\sigma_\rho}}{{\rho}}"]),
        ("Errores comunes y límites de validez",
         r"Es frecuente sumar las incertidumbres relativas de forma lineal cuando son independientes, lo que sobrestima el error, o sumarlas en cuadratura cuando están correlacionadas (por ejemplo dos longitudes medidas con la misma regla descalibrada), lo que lo subestima. Otro error es olvidar el exponente: para $r^2$ la contribución es $2\,\sigma_r/r$, no $\sigma_r/r$. Se confunde también la incertidumbre estándar con un intervalo de confianza: el resultado $\bar y\pm\sigma_y$ cubre alrededor del $68\ \%$ si $y$ es normal, no el $100\ \%$. Cuando las incertidumbres relativas dejan de ser pequeñas, o $f$ tiene un punto singular cerca de $\bar{\vec x}$, el modelo de primer orden se reemplaza por el desarrollo de segundo orden o por Monte Carlo.",
         [r"\sigma_y^{\text{indep}}\le\sigma_y^{\text{peor caso}},\qquad \sigma_y^{\text{correl.}}\ \text{puede ser}\ >\sigma_y^{\text{indep}}\ \text{o}\ <\sigma_y^{\text{indep}}\ \text{según el signo de}\ \sigma_{ij}f_if_j"]),
    ]
    return dict(concept=concept)


# ====================================================================== cap. 2
def _vec_add() -> dict:
    ra = vec.resultant_analytic()
    rn = vec.resultant_numeric()
    a, b = 40.0, 25.0
    al, be = 30.0, 120.0
    gam = be - al
    D1, D2 = vec.D1, vec.D2
    rx, ry = D1[0] + D2[0], D1[1] + D2[1]
    r_gen = float(np.sqrt(a**2 + b**2 + 2 * a * b * np.cos(np.deg2rad(gam))))
    th_gen = float(np.rad2deg(np.arctan2(a * np.sin(np.deg2rad(al)) + b * np.sin(np.deg2rad(be)),
                                         a * np.cos(np.deg2rad(al)) + b * np.cos(np.deg2rad(be)))))
    concept = [
        ("Qué se estudia y por qué importa",
         r"Las magnitudes con dirección (desplazamientos, velocidades, fuerzas, campos) no se combinan como números ordinarios: lo que importa es la combinación de efectos en el espacio. La suma vectorial permite predecir el desplazamiento neto de un caminante, la fuerza resultante sobre un puente, la velocidad de un avión con viento cruzado o la superposición de campos eléctricos. Su fundamento es la estructura de espacio vectorial del espacio físico, que hace posible descomponer cualquier problema en componentes independientes. En este problema se parte de dos desplazamientos consecutivos y se pregunta por el desplazamiento neto.",
         [r"\vec R=\sum_{k=1}^{N}\vec A_k,\qquad \vec A_k=A_k(\cos\alpha_k,\ \sin\alpha_k)"]),
        ("Magnitudes y convenciones",
         r"Un vector plano se describe por su módulo $A\ge0$ (en metros para un desplazamiento) y su ángulo $\alpha$ medido desde el eje $+x$ en sentido antihorario, de modo que $\alpha\in(-180^\circ,180^\circ]$. Las componentes son proyecciones con signo: $A_x<0$ si el vector apunta hacia $-x$. El ángulo entre dos vectores es la diferencia de sus ángulos polares, $\gamma=\beta-\alpha$, y no debe confundirse con el ángulo exterior del polígono. En el caso del enunciado los órdenes de magnitud son decenas de metros y el ángulo entre los tramos es de $90^\circ$.",
         [r"A=\sqrt{A_x^2+A_y^2}\ \mathrm{(m)},\qquad \alpha=\operatorname{atan2}(A_y,A_x),\qquad \gamma=\beta-\alpha"]),
        ("Intuición y analogías",
         r"Sumar vectores es encadenar flechas, punta con cola; el resultado es la flecha que va del inicio del primer tramo al final del último. Si los vectores apuntan en la misma dirección los módulos se suman, y si son opuestos se restan, de modo que la suma de módulos es siempre una cota superior. Con ángulo recto el resultado es la hipotenusa, como en el teorema de Pitágoras. Es análogo a dos remeros que tiran de una barca en direcciones distintas: solo la componente común acelera la barca.",
         [r"\big|\,|\vec A|-|\vec B|\,\big|\le|\vec A+\vec B|\le|\vec A|+|\vec B|"]),
        ("Estructura de espacio vectorial y componentes",
         r"El conjunto de desplazamientos con la suma de flechas y el producto por escalares cumple los axiomas de espacio vectorial real: la suma es conmutativa, asociativa, tiene neutro y opuesto, y el producto por escalares es distributivo. Eligiendo una base ortonormal $\{\hat e_x,\hat e_y\}$ todo vector se escribe de forma única como combinación lineal, y las operaciones se vuelven aritmética de componentes. Esto convierte un problema geométrico en uno algebraico, y por inducción vale para cualquier número de sumandos.",
         [r"\vec A=A_x\hat e_x+A_y\hat e_y,\qquad \vec A+\vec B=(A_x+B_x)\hat e_x+(A_y+B_y)\hat e_y",
          r"\vec R=\sum_k\vec A_k\;\Longleftrightarrow\;R_x=\sum_kA_k\cos\alpha_k,\quad R_y=\sum_kA_k\sin\alpha_k"]),
        ("Invariancia bajo rotaciones",
         r"Una rotación de ejes un ángulo $\varphi$ cambia las componentes mediante la matriz ortogonal $R(\varphi)$, pero no el vector físico. La suma es covariante: $R(\varphi)(\vec A+\vec B)=R(\varphi)\vec A+R(\varphi)\vec B$ por linealidad. El módulo se conserva porque $R^{\mathsf T}R=I$, y por ello el módulo del resultante no depende de la orientación de ejes elegida. Esta isotropía del espacio es la razón física de que solo los invariantes (módulos y ángulos entre vectores) tengan significado independiente del observador.",
         [r"\begin{pmatrix}A_x'\\A_y'\end{pmatrix}=\begin{pmatrix}\cos\varphi&\sin\varphi\\-\sin\varphi&\cos\varphi\end{pmatrix}\begin{pmatrix}A_x\\A_y\end{pmatrix},\qquad R^{\mathsf T}R=I",
          r"A_x'^2+A_y'^2=A_x^2+A_y^2"]),
        ("Módulo del resultante: ley de los cosenos",
         r"Para dos vectores se calcula el módulo cuadrado como producto punto del resultante consigo mismo, lo que es lícito porque $|\vec R|^2=\vec R\cdot\vec R$ y el producto punto es bilineal. Al desarrollar aparece el producto cruzado $2\vec A\cdot\vec B=2AB\cos\gamma$, con $\gamma$ el ángulo entre los vectores (no el del triángulo de la suma, que es $180^\circ-\gamma$). El resultado es la ley de los cosenos, que se reduce a Pitágoras para $\gamma=90^\circ$. Los casos $\gamma=0$ y $\gamma=180^\circ$ dan las cotas de la desigualdad triangular.",
         [r"R^2=(\vec A+\vec B)\cdot(\vec A+\vec B)=A^2+B^2+2\vec A\cdot\vec B=A^2+B^2+2AB\cos\gamma",
          r"\gamma=0:\ R=A+B;\qquad \gamma=90^\circ:\ R=\sqrt{A^2+B^2};\qquad \gamma=180^\circ:\ R=|A-B|"]),
        ("Dirección del resultante: ley de los senos y atan2",
         r"En el triángulo formado por $\vec A$, $\vec B$ y $\vec R$ se aplica la ley de los senos para obtener el ángulo $\delta$ que $\vec R$ forma con $\vec A$. Es una alternativa a calcular el ángulo directamente con las componentes, que da el ángulo polar absoluto sin ambigüedad de cuadrante si se usa la función atan2. La expresión por componentes es preferible para $N>2$ vectores, pues la ley de senos solo sirve en el caso de dos. La coincidencia de ambos caminos es una comprobación útil.",
         [r"\dfrac{\sin\delta}{B}=\dfrac{\sin(180^\circ-\gamma)}{R}\;\Rightarrow\;\delta=\sin^{-1}\!\left(\dfrac{B\sin\gamma}{R}\right),\qquad \theta_R=\alpha+\delta",
          r"\theta_R=\operatorname{atan2}\!\big(A\sin\alpha+B\sin\beta,\ A\cos\alpha+B\cos\beta\big)"]),
        ("Resultado general",
         r"El desplazamiento resultante de $N$ vectores tiene componentes dadas por la suma de las componentes individuales; en el caso de dos vectores el módulo y la dirección se expresan en forma cerrada. Ambos tienen las unidades del vector sumado (el ángulo es adimensional) y son invariantes de forma frente a rotaciones del sistema de referencia.",
         [r"\boxed{\,R=\sqrt{A^{2}+B^{2}+2AB\cos(\beta-\alpha)},\qquad \theta_R=\operatorname{atan2}\!\big(A\sin\alpha+B\sin\beta,\ A\cos\alpha+B\cos\beta\big)\,}",
          r"[R]=[A]=[B]=\mathrm{m},\qquad [\theta_R]=\mathrm{rad}\ \text{(adimensional)}"]),
        ("Particularización al problema",
         rf"Con $A={a:.0f}$ m, $\alpha={al:.0f}^\circ$, $B={b:.0f}$ m y $\beta={be:.0f}^\circ$ el ángulo entre los tramos es $\gamma={gam:.0f}^\circ$ y $\cos\gamma=0$. La fórmula general reproduce el valor analítico de la escena y coincide con la suma por componentes que realiza el código.",
         [rf"R=\sqrt{{{a:.0f}^2+{b:.0f}^2+2({a:.0f})({b:.0f})\cos{gam:.0f}^\circ}}=\sqrt{{{a**2 + b**2:.0f}}}={fx(r_gen, 2)}\ \mathrm{{m}}\quad({fx(ra['magnitude'], 2)})",
          rf"R_x={fx(D1[0], 3)}+({fx(D2[0], 3)})={fx(rx, 3)}\ \mathrm{{m}},\qquad R_y={fx(D1[1], 3)}+{fx(D2[1], 3)}={fx(ry, 3)}\ \mathrm{{m}}",
          rf"\theta_R=\operatorname{{atan2}}({fx(ry, 3)},\,{fx(rx, 3)})={fx(th_gen, 2)}^\circ\quad(\text{{ley de senos}}:\ {fx(ra['angle_deg'], 2)}^\circ),\qquad R_{{num}}={fx(rn['magnitude'], 2)}\ \mathrm{{m}}"]),
        ("Errores comunes y límites de validez",
         r"Es habitual sumar módulos en lugar de componentes, o usar $\gamma$ equivocado en la ley de los cosenos: el signo del término cruzado es $+$ cuando $\gamma$ es el ángulo entre los vectores dibujados con el mismo origen, y $-$ si se usa el ángulo interior del triángulo. Con $\tan^{-1}(R_y/R_x)$ se pierde el cuadrante; atan2 lo resuelve. Por otro lado, la suma vectorial vale para desplazamientos en un plano euclídeo; sobre una esfera (navegación a gran escala) los desplazamientos finitos no conmutan ni se suman, y se reemplaza por trigonometría esférica. Tampoco vale para magnitudes que tienen módulo y dirección pero no se suman como flechas, como las rotaciones finitas en 3D.",
         [r"\text{rotaciones finitas:}\quad R_1R_2\ne R_2R_1\ \text{en}\ SO(3)"]),
    ]
    return dict(concept=concept)


def _vec_prod() -> dict:
    A, B = vec.A, vec.B
    pa = vec.products_analytic()
    dot, cz = vec.dot(), vec.cross_z()
    na, nb = float(np.linalg.norm(A)), float(np.linalg.norm(B))
    th = vec.angle_between_deg()
    lag = na**2 * nb**2 - dot**2
    concept = [
        ("Qué se estudia y por qué importa",
         r"Dos vectores pueden combinarse para dar un escalar (producto escalar) o un nuevo vector (producto vectorial). El escalar mide cuánto de un vector está alineado con otro y aparece en el trabajo $W=\vec F\cdot\Delta\vec r$ y el flujo $\Phi=\vec E\cdot\vec S$. El vectorial mide el área orientada que definen y aparece en el momento $\vec\tau=\vec r\times\vec F$, la fuerza magnética $q\vec v\times\vec B$ y el momento angular. Ambos son los dos únicos productos bilineales naturales en el espacio físico compatibles con la rotación.",
         [r"W=\vec F\cdot\Delta\vec r,\qquad \vec\tau=\vec r\times\vec F,\qquad \vec F_B=q\,\vec v\times\vec B"]),
        ("Magnitudes y convenciones",
         r"El producto escalar es un número con unidades del producto de las unidades de los factores (por ejemplo N$\cdot$m = J) y puede ser negativo; el vectorial tiene el módulo de un área, $AB\sin\theta\ge0$, y su dirección se fija por la regla de la mano derecha. En el plano, $\vec A\times\vec B$ solo tiene componente $z$, de signo positivo si $\vec B$ está a la izquierda de $\vec A$. El ángulo $\theta\in[0,\pi]$ entre los vectores es el mismo en ambos productos. Aquí los vectores son adimensionales (unidades arbitrarias).",
         [r"\theta\in[0,\pi],\qquad \hat n=\dfrac{\vec A\times\vec B}{|\vec A\times\vec B|}\ \text{(mano derecha)}"]),
        ("Intuición y analogías",
         r"El producto escalar es la sombra de un vector sobre el otro, multiplicada por el módulo del segundo: si son perpendiculares no hay sombra y vale cero. El vectorial es el área del paralelogramo, y por ello se anula cuando los vectores son paralelos, ya que entonces el paralelogramo se aplasta. Ambos son complementarios: uno detecta alineación, el otro detecta no alineación. Su combinación $\cos^2+\sin^2=1$ expresa que los dos juntos agotan la información del ángulo.",
         [r"(\vec A\cdot\vec B)^2+|\vec A\times\vec B|^2=A^2B^2"]),
        ("Definición axiomática del producto escalar",
         r"El producto escalar se define como la forma bilineal simétrica y definida positiva que induce la norma, $\vec A\cdot\vec A=|\vec A|^2$. En una base ortonormal esto implica que $\hat e_i\cdot\hat e_j=\delta_{ij}$, y la bilinealidad da la expresión por componentes. La invariancia frente a rotaciones, $R^{\mathsf T}R=I$, es lo que le da significado geométrico: el valor no depende de los ejes. De la desigualdad de Cauchy-Schwarz $|\vec A\cdot\vec B|\le AB$ se garantiza que $\cos\theta$ definido a partir de él cae en $[-1,1]$.",
         [r"\vec A\cdot\vec B=\sum_{i}A_iB_i=\sum_{i,j}A_i\delta_{ij}B_j,\qquad \vec A\cdot\vec B=AB\cos\theta",
          r"(R\vec A)\cdot(R\vec B)=\vec A^{\mathsf T}R^{\mathsf T}R\vec B=\vec A\cdot\vec B,\qquad |\vec A\cdot\vec B|\le AB"]),
        ("Producto vectorial con el símbolo de Levi-Civita",
         r"El producto vectorial se define por componentes con el símbolo totalmente antisimétrico $\varepsilon_{ijk}$, que es bilineal y antisimétrico: $\vec A\times\vec B=-\vec B\times\vec A$. La antisimetría se sigue de intercambiar dos índices de $\varepsilon$. En forma de determinante es la regla familiar. Bajo rotaciones propias ($\det R=+1$) se transforma como un vector, pero bajo inversión $\vec r\to-\vec r$ no cambia de signo, por lo que es un pseudovector (vector axial): por eso el momento angular y el campo magnético tienen ese carácter.",
         [r"(\vec A\times\vec B)_i=\sum_{j,k}\varepsilon_{ijk}A_jB_k,\qquad \vec A\times\vec B=\begin{vmatrix}\hat e_x&\hat e_y&\hat e_z\\A_x&A_y&A_z\\B_x&B_y&B_z\end{vmatrix}",
          r"\vec A\times\vec B=-\vec B\times\vec A,\qquad \vec A\times\vec A=\vec 0,\qquad \vec A\cdot(\vec A\times\vec B)=0"]),
        ("Identidad de Lagrange y módulo del producto vectorial",
         r"La identidad $\varepsilon_{ijk}\varepsilon_{ilm}=\delta_{jl}\delta_{km}-\delta_{jm}\delta_{kl}$ permite demostrar que el módulo cuadrado del producto vectorial es $A^2B^2$ menos el cuadrado del escalar. Con $\vec A\cdot\vec B=AB\cos\theta$ esto da $|\vec A\times\vec B|=AB\sin\theta$ sin ninguna hipótesis geométrica adicional (la raíz es positiva porque $\sin\theta\ge0$ en $[0,\pi]$). En el plano se reduce a un determinante $2\times2$, y es igual al área orientada del paralelogramo.",
         [r"|\vec A\times\vec B|^2=A^2B^2-(\vec A\cdot\vec B)^2=A^2B^2\sin^2\theta",
          r"\text{plano}:\quad \vec A\times\vec B=(A_xB_y-A_yB_x)\,\hat z=\det\begin{pmatrix}A_x&A_y\\B_x&B_y\end{pmatrix}\hat z"]),
        ("Ángulo, área y producto triple",
         r"Del par de productos se obtiene el ángulo sin ambigüedad: el coseno da el módulo del ángulo y el signo del seno, ligado a la orientación, lo da la componente $z$ del vectorial. Con un tercer vector, el producto triple escalar $\vec C\cdot(\vec A\times\vec B)$ es el volumen orientado del paralelepípedo y la identidad $\vec A\times(\vec B\times\vec C)=\vec B(\vec A\cdot\vec C)-\vec C(\vec A\cdot\vec B)$ simplifica dobles productos. El determinante de Gram generaliza la relación de Lagrange a $n$ vectores.",
         [r"\theta=\operatorname{atan2}\big(|\vec A\times\vec B|,\ \vec A\cdot\vec B\big),\qquad \sin\theta_{\text{orient.}}=\dfrac{(\vec A\times\vec B)_z}{AB}",
          r"V=\big|\vec C\cdot(\vec A\times\vec B)\big|,\qquad \vec A\times(\vec B\times\vec C)=\vec B(\vec A\cdot\vec C)-\vec C(\vec A\cdot\vec B)"]),
        ("Resultado general",
         r"Para dos vectores planos de componentes $(A_x,A_y)$ y $(B_x,B_y)$ el producto escalar es la suma de productos de componentes, la componente $z$ del vectorial es el determinante, y el ángulo entre ellos se obtiene de ambos. Las unidades son las del producto de los factores y los dos productos satisfacen la identidad de Lagrange.",
         [r"\boxed{\,\vec A\cdot\vec B=A_xB_x+A_yB_y,\qquad (\vec A\times\vec B)_z=A_xB_y-A_yB_x,\qquad \theta=\operatorname{atan2}\!\big(|A_xB_y-A_yB_x|,\ A_xB_x+A_yB_y\big)\,}",
          r"[\vec A\cdot\vec B]=[\,|\vec A\times\vec B|\,]=[A][B],\qquad (\vec A\cdot\vec B)^2+|\vec A\times\vec B|^2=A^2B^2"]),
        ("Particularización al problema",
         rf"Con $\vec A=(4,1)$ y $\vec B=(2,3)$ los módulos son $A=\sqrt{{17}}={fx(na, 4)}$ y $B=\sqrt{{13}}={fx(nb, 4)}$. Los valores obtenidos coinciden con los de la escena, y la identidad de Lagrange se cumple de forma exacta ($221=121+100$).",
         [rf"\vec A\cdot\vec B=4\cdot2+1\cdot3={dot:.0f},\qquad (\vec A\times\vec B)_z=4\cdot3-1\cdot2={cz:.0f}",
          rf"\theta=\operatorname{{atan2}}({cz:.0f},\,{dot:.0f})={fx(th, 2)}^\circ\quad(\text{{escena}}:\ {fx(th, 2)}^\circ)",
          rf"A^2B^2-(\vec A\cdot\vec B)^2={na**2 * nb**2:.0f}-{dot**2:.0f}={lag:.0f}={cz**2:.0f}=|\vec A\times\vec B|^2,\qquad \text{{área}}={fx(pa['cross_by_angle'], 4)}"]),
        ("Errores comunes y límites de validez",
         r"Se confunde con frecuencia el orden: $\vec A\cdot\vec B$ es conmutativo, pero $\vec A\times\vec B$ cambia de signo al intercambiar los factores. Otro error es suponer que $\vec A\cdot\vec B=\vec A\cdot\vec C$ implica $\vec B=\vec C$; solo implica que $\vec B-\vec C\perp\vec A$. También se aplica el producto vectorial «en el plano» olvidando que su resultado es perpendicular al plano. El producto vectorial como vector solo existe en tres dimensiones (y en siete); en otras dimensiones se sustituye por el producto exterior $A\wedge B$, un bivector que en 2D es el escalar $A_xB_y-A_yB_x$ que se calcula aquí.",
         [r"\vec A\cdot\vec B=\vec A\cdot\vec C\;\not\Rightarrow\;\vec B=\vec C;\qquad A\wedge B=(A_xB_y-A_yB_x)\,\hat e_x\wedge\hat e_y"]),
    ]
    return dict(concept=concept)


# ====================================================================== cap. 3
def _inst_vel() -> dict:
    c0, c1_, c2, c3 = 2.0, 3.0, -0.5, 0.1
    t0 = 2.0
    v2 = float(m1d.v_analytic(t0))
    a2 = float(m1d.a_analytic(t0))
    j = 6 * c3
    tab = m1d.secant_table(t0)
    d1 = tab[0]
    disp = m1d.displacement_from_velocity(0.0, 4.0, 32)
    dx = float(m1d.x_of_t(4.0) - m1d.x_of_t(0.0))
    h = 0.01
    a_fd = m1d.central_second_difference(t0, h)
    err_sec = a2 * 1.0 / 2 + j * 1.0**2 / 6  # error teorico de la secante dt=1
    trap_err = -((4.0 / 32) ** 2 / 12) * (float(m1d.a_analytic(4.0)) - float(m1d.a_analytic(0.0)))
    concept = [
        ("Qué se estudia y por qué importa",
         r"La velocidad instantánea formaliza la idea de «rapidez en este instante» para un movimiento que cambia continuamente. Es el concepto que dio origen al cálculo diferencial y es la base de toda la cinemática: un velocímetro, un GPS y un radar miden aproximaciones a esa derivada. Su operación inversa, integrar la velocidad, reconstruye el desplazamiento y conecta la derivada con el área bajo una curva. En la práctica solo se dispone de muestras discretas de la posición, y por eso importa saber cuánto error comete cada aproximación.",
         [r"v(t)=\lim_{\Delta t\to0}\dfrac{x(t+\Delta t)-x(t)}{\Delta t}=\dot x,\qquad \Delta x=\int_{t_0}^{t_1}v\,dt"]),
        ("Magnitudes y convenciones",
         r"La posición $x(t)$ se da en metros respecto de un origen, y el eje fija el signo: $v>0$ indica movimiento hacia $+x$. La velocidad tiene unidades de m/s, la aceleración de m/s$^2$ y el «tirón» (jerk) de m/s$^3$. La velocidad es distinta de la rapidez y de la velocidad media $\Delta x/\Delta t$, que depende del intervalo; el desplazamiento neto tampoco es la distancia recorrida si $v$ cambia de signo. Los coeficientes de la posición del enunciado tienen unidades m, m/s, m/s$^2$ y m/s$^3$ para que cada término sea una longitud.",
         [r"x(t)=\sum_{n=0}^{3}c_nt^n,\qquad [c_n]=\mathrm{m\,s^{-n}}"]),
        ("Intuición y analogías",
         r"La velocidad instantánea es la pendiente de la tangente a la gráfica $x(t)$, y las secantes son aproximaciones que giran hacia la tangente cuando el intervalo se estrecha. Es como ampliar una curva con una lupa: a escala suficientemente pequeña toda curva diferenciable parece una recta. Inversamente, el desplazamiento es el área bajo $v(t)$, como el agua que llena un depósito con caudal variable. Derivar e integrar son operaciones inversas, lo que enuncia el teorema fundamental del cálculo.",
         [r"x(t+\Delta t)\simeq x(t)+v(t)\,\Delta t\quad(\text{linealización local})"]),
        ("Velocidad y aceleración como derivadas",
         r"Para una posición suficientemente suave se define la velocidad como la derivada y la aceleración como su derivada segunda. La existencia del límite requiere que $x$ sea diferenciable; la velocidad puede ser discontinua (un choque ideal) sin que el movimiento deje de tener sentido, pero entonces la aceleración no existe como función. Para un polinomio la derivada se obtiene término a término, porque la derivación es lineal y $\frac{d}{dt}t^n=nt^{n-1}$ se prueba con el binomio de Newton.",
         [r"v=\dfrac{dx}{dt},\qquad a=\dfrac{dv}{dt}=\dfrac{d^{2}x}{dt^{2}},\qquad \dfrac{d}{dt}t^{n}=\lim_{\Delta t\to0}\dfrac{(t+\Delta t)^n-t^n}{\Delta t}=nt^{n-1}",
          r"x=c_0+c_1t+c_2t^2+c_3t^3\;\Rightarrow\;v=c_1+2c_2t+3c_3t^2,\quad a=2c_2+6c_3t,\quad j=\dot a=6c_3"]),
        ("Error de las aproximaciones por diferencias",
         r"Con el desarrollo de Taylor de $x(t\pm\Delta t)$ se cuantifica el error de cada esquema. La secante hacia adelante tiene un error de orden $\Delta t$ proporcional a la aceleración, porque la pendiente de la cuerda es la velocidad en el punto medio del intervalo. La diferencia centrada cancela los términos pares y gana un orden. La segunda diferencia centrada estima la aceleración con error de orden $h^2$ proporcional a la derivada cuarta, que para un cúbico es nula: su error se debe únicamente al redondeo, que crece como $\varepsilon/h^2$.",
         [r"\dfrac{x(t+\Delta t)-x(t)}{\Delta t}=v+\dfrac{a}{2}\Delta t+\dfrac{j}{6}\Delta t^{2}+\dots",
          r"\dfrac{x(t+h)-x(t-h)}{2h}=v+\dfrac{j}{6}h^{2}+\dots,\qquad \dfrac{x(t+h)-2x(t)+x(t-h)}{h^{2}}=a+\dfrac{x^{(4)}}{12}h^{2}+\dots"]),
        ("Desplazamiento como integral y regla del trapecio",
         r"El teorema fundamental del cálculo afirma que, si $v=\dot x$ es continua, $\int_{t_0}^{t_1}v\,dt=x(t_1)-x(t_0)$. Esto permite hallar el desplazamiento con la primitiva sin conocer la trayectoria intermedia. La regla compuesta del trapecio con $n$ subintervalos aproxima el área por trapecios y su error global es proporcional a $h^2$ y a la variación de $v'$ en el intervalo. Si $v$ es parabólica, $v''=6c_3\cdot$ constante y el error del trapecio es calculable en forma exacta.",
         [r"\Delta x=\int_{t_0}^{t_1}v\,dt=\Big[c_1t+c_2t^2+c_3t^3\Big]_{t_0}^{t_1}",
          r"T_n=\dfrac{h}{2}\Big[v_0+2\sum_{k=1}^{n-1}v_k+v_n\Big],\qquad \Delta x-T_n=-\dfrac{h^{2}}{12}\,\big[a(t_1)-a(t_0)\big]+\mathcal O(h^4)"]),
        ("Resultado general",
         r"Para una posición cúbica general quedan expresiones cerradas para la velocidad, la aceleración y el desplazamiento. Cada una tiene las unidades correctas término a término. La velocidad media en $[t_0,t_1]$ coincide con la instantánea en el punto medio solo si $j=0$.",
         [r"\boxed{\,v(t)=c_1+2c_2t+3c_3t^{2},\qquad a(t)=2c_2+6c_3t,\qquad \Delta x=c_1\Delta t+c_2\,(t_1^2-t_0^2)+c_3\,(t_1^3-t_0^3)\,}",
          r"[c_1t]=[c_2t^2]=[c_3t^3]=\mathrm{m},\qquad \bar v=\dfrac{\Delta x}{\Delta t}=v\!\left(\dfrac{t_0+t_1}{2}\right)+\dfrac{c_3\,\Delta t^2}{4}"]),
        ("Particularización al problema",
         rf"Con $c_0={c0:.0f}$ m, $c_1={c1_:.0f}$ m/s, $c_2={c2}$ m/s$^2$ y $c_3={c3}$ m/s$^3$, la fórmula general en $t=2$ s da los valores de la escena. La secante con $\Delta t=1$ s tiene un error dado exactamente por $a\Delta t/2+j\Delta t^2/6$, y la segunda diferencia centrada acierta hasta el redondeo porque la derivada cuarta es nula.",
         [rf"v(2)={c1_:.0f}+2({c2})(2)+3({c3})(4)={fx(v2, 4)}\ \mathrm{{m/s}},\qquad a(2)=2({c2})+6({c3})(2)={fx(a2, 4)}\ \mathrm{{m/s^2}}",
          rf"\text{{secante }}\Delta t=1:\ {fx(d1['slope'], 4)}=v+\dfrac{{a}}{{2}}+\dfrac{{j}}{{6}}={fx(v2, 4)}+{fx(a2 / 2, 4)}+{fx(j / 6, 4)}={fx(v2 + a2 / 2 + j / 6, 4)}",
          rf"\Delta x=c_1(4)+c_2(16)+c_3(64)=12-8+6.4={fx(dx, 4)}\ \mathrm{{m}}\quad(\text{{trapecio, }}n=32:\ {fx(disp['trapezoid'], 4)})",
          rf"a_{{FD}}(h={h})={fx(a_fd, 4)}\ \mathrm{{m/s^2}}\ \text{{(exacto }}{fx(a2, 4)}\text{{)}},\qquad \text{{error del trapecio}}={sci(trap_err, 3)}\ \mathrm{{m}}"]),
        ("Errores comunes y límites de validez",
         r"Un error clásico es identificar la velocidad media sobre un intervalo con la instantánea en su extremo; difieren en $a\Delta t/2$. Otro es creer que $v=0$ implica $a=0$: en el punto de retorno la velocidad se anula y la aceleración no. También se confunde el desplazamiento con el camino recorrido, que requiere integrar $|v|$. En las diferencias finitas reducir $h$ indefinidamente no mejora el resultado, porque el error de redondeo crece como $\varepsilon/h$ (primera derivada) o $\varepsilon/h^2$ (segunda). El modelo deja de valer en movimientos con cambios bruscos o ruidosos; allí se usan derivadas suavizadas o filtros de Kalman, y la relatividad reemplaza a la cinemática newtoniana cuando $v\sim c$.",
         [r"\bar v\ne v(t_1);\qquad v=0\not\Rightarrow a=0;\qquad \text{error total}\sim\dfrac{M}{6}h^2+\dfrac{\varepsilon}{h}"]),
    ]
    return dict(concept=concept)


def _const_acc() -> dict:
    v0, g = m1d.V0, m1d.G
    summ = m1d.free_fall_summary()
    y3 = float(m1d.free_fall_analytic(3.0)[0])
    v3 = float(m1d.free_fall_analytic(3.0)[1])
    y0h = 10.0  # ejemplo de lanzamiento desde altura para el caso general
    t_h = (v0 + np.sqrt(v0**2 + 2 * g * y0h)) / g
    concept = [
        ("Qué se estudia y por qué importa",
         r"La aceleración constante es el caso más simple de movimiento acelerado y describe con muy buena aproximación la caída de cuerpos densos cerca de la superficie terrestre, el frenado de un vehículo con fuerza constante o el movimiento de una carga en un campo eléctrico uniforme. Sus ecuaciones cierran la cinemática de una dimensión y son el punto de partida de los proyectiles. Además, es el modelo con solución exacta que sirve para validar métodos numéricos de integración de ecuaciones diferenciales.",
         [r"\ddot y=-g,\qquad y(0)=y_0,\quad \dot y(0)=v_0"]),
        ("Magnitudes y convenciones",
         r"Se elige el eje $y$ vertical hacia arriba con origen en el suelo, de modo que $a=-g$ con $g=9.81\ \mathrm{m/s^2}$ (el valor varía de $9.78$ en el ecuador a $9.83$ en los polos). La velocidad inicial $v_0$ es positiva si el lanzamiento es hacia arriba. Con la convención de signos elegida el tiempo de subida, la altura máxima y el tiempo de vuelo son cantidades positivas, mientras que la velocidad de regreso es negativa. Un lanzamiento de $20\ \mathrm{m/s}$ implica alturas de decenas de metros y tiempos de unos segundos.",
         [r"[y]=\mathrm{m},\quad [v]=\mathrm{m\,s^{-1}},\quad [g]=\mathrm{m\,s^{-2}},\qquad a=-g\ \ \text{(eje hacia arriba)}"]),
        ("Intuición y analogías",
         r"Una aceleración constante significa que la velocidad cambia la misma cantidad en cada segundo: cada segundo la pelota pierde $9.81\ \mathrm{m/s}$ de velocidad ascendente. Subir y bajar son simétricos como un vídeo reproducido al revés: la pelota vuelve con la misma rapidez con que salió, y el tiempo de subida es igual al de bajada. La posición, al integrar una velocidad lineal, es una parábola, de ahí que la trayectoria vertical en función del tiempo sea una parábola cóncava hacia abajo.",
         [r"v(t)=v_0-gt\ \ (\text{recta}),\qquad y(t)\ \ (\text{parábola})"]),
        ("Formulación variacional: lagrangiano y ecuación de movimiento",
         r"Se toma como coordenada generalizada la altura $y$. La energía cinética es $T=\tfrac12m\dot y^2$ y la potencial gravitatoria cerca de la superficie es $V=mgy$, de modo que $\mathcal L=T-V$. El principio de Hamilton exige que la acción $S=\int\mathcal L\,dt$ sea estacionaria, lo que conduce a la ecuación de Euler-Lagrange. La masa se cancela, y se obtiene $\ddot y=-g$ para toda masa (principio de equivalencia). Esta formulación hace evidentes las leyes de conservación que siguen.",
         [r"\mathcal L=\tfrac12m\dot y^{2}-mgy,\qquad \delta S=\delta\!\int_{t_a}^{t_b}\mathcal L\,dt=0",
          r"\dfrac{d}{dt}\dfrac{\partial\mathcal L}{\partial\dot y}-\dfrac{\partial\mathcal L}{\partial y}=m\ddot y+mg=0\;\Longrightarrow\;\ddot y=-g"]),
        ("Integración y condiciones iniciales",
         r"La ecuación diferencial de segundo orden con segunda derivada constante se integra dos veces, introduciendo dos constantes que se fijan con las condiciones iniciales. La primera integración da $v(t)$, y la segunda $y(t)$. La unicidad de la solución para condiciones iniciales dadas se sigue del teorema de Picard-Lindelöf, pues el segundo miembro es lipschitziano. Eliminando $t$ entre ambas se obtiene la relación entre velocidad y posición, que es equivalente a la conservación de la energía.",
         [r"\dot y=\int(-g)\,dt=v_0-gt,\qquad y=y_0+v_0t-\tfrac12gt^{2}",
          r"t=\dfrac{v_0-v}{g}\;\Rightarrow\;v^{2}=v_0^{2}-2g\,(y-y_0)"]),
        ("Integral primera: energía y simetría temporal",
         r"El lagrangiano no depende explícitamente de $t$, por lo que el teorema de Noether (simetría ante traslaciones temporales) asegura que la función energía $E=\dot y\,\partial\mathcal L/\partial\dot y-\mathcal L$ es constante. Se obtiene $E=\tfrac12m\dot y^2+mgy$, la energía mecánica total. La igualdad $v^2+2gy=\mathrm{const}$ ofrece la altura máxima sin resolver para el tiempo: en el punto más alto $v=0$. No hay simetría ante traslaciones en $y$, porque $V$ depende de $y$, y así el momento $m\dot y$ no se conserva (la gravedad lo cambia a razón $-mg$).",
         [r"E=\dot y\,\dfrac{\partial\mathcal L}{\partial\dot y}-\mathcal L=\tfrac12m\dot y^{2}+mgy=\text{const},\qquad \dfrac{dp}{dt}=-mg",
          r"\tfrac12mv_0^{2}+mgy_0=mgy_{\max}\;\Longrightarrow\;y_{\max}=y_0+\dfrac{v_0^{2}}{2g}"]),
        ("Tiempos característicos y caso general con altura inicial",
         r"El instante del vértice se halla anulando $v(t)$. El tiempo de vuelo es la raíz positiva de $y(t)=0$; para lanzamiento desde el nivel de referencia es $2v_0/g$, y si se lanza desde una altura $y_0$ se elige la raíz positiva de la cuadrática. El parámetro adimensional $\Lambda=2gy_0/v_0^2$ mide la altura inicial relativa a la de ascenso: para $\Lambda\to0$ se recupera el caso simétrico y para $\Lambda\gg1$ la caída es casi libre. La velocidad de impacto sigue de la conservación de la energía.",
         [r"t_{\mathrm{ap}}=\dfrac{v_0}{g},\qquad t_{\mathrm{vuelo}}=\dfrac{v_0+\sqrt{v_0^{2}+2gy_0}}{g}=\dfrac{v_0}{g}\left(1+\sqrt{1+\Lambda}\right),\quad \Lambda=\dfrac{2gy_0}{v_0^{2}}",
          r"v_{\mathrm{impacto}}=-\sqrt{v_0^{2}+2gy_0},\qquad \Lambda\to0:\ t_{\mathrm{vuelo}}\to\dfrac{2v_0}{g}"]),
        ("Resultado general",
         r"Las cantidades principales del movimiento con aceleración constante $-g$ quedan en función de los parámetros $v_0$, $g$ y $y_0$. Todas son dimensionalmente consistentes: $v_0^2/g$ es una longitud y $v_0/g$ un tiempo, las únicas escalas que pueden formarse con $v_0$ y $g$.",
         [r"\boxed{\,y(t)=y_0+v_0t-\tfrac12gt^{2},\qquad y_{\max}=y_0+\dfrac{v_0^{2}}{2g},\qquad t_{\mathrm{vuelo}}=\dfrac{v_0+\sqrt{v_0^{2}+2gy_0}}{g}\,}",
          r"\left[\dfrac{v_0^{2}}{g}\right]=\mathrm{m},\qquad \left[\dfrac{v_0}{g}\right]=\mathrm{s}"]),
        ("Particularización al problema",
         rf"Con $y_0=0$, $v_0={v0:.0f}$ m/s y $g={g}$ m/s$^2$ ($\Lambda=0$) se recuperan los valores de la escena. La altura en $t=3$ s es menor que la máxima porque la pelota ya ha pasado el vértice en $t={fx(summ['t_peak'], 3)}$ s y su velocidad es negativa. Como ilustración del parámetro $\Lambda$, desde $y_0={y0h:.0f}$ m el tiempo de vuelo sería {fx(t_h, 3)} s.",
         [rf"t_{{ap}}=\dfrac{{{v0:.0f}}}{{{g}}}={fx(summ['t_peak'], 3)}\ \mathrm{{s}},\qquad y_{{\max}}=\dfrac{{{v0:.0f}^2}}{{2({g})}}={fx(summ['y_max'], 3)}\ \mathrm{{m}},\qquad t_{{vuelo}}=\dfrac{{2v_0}}{{g}}={fx(summ['t_return'], 3)}\ \mathrm{{s}}",
          rf"y(3)={v0:.0f}(3)-\tfrac12({g})(9)={fx(y3, 3)}\ \mathrm{{m}},\qquad v(3)={v0:.0f}-{g}(3)={fx(v3, 2)}\ \mathrm{{m/s}}",
          rf"E/m=\tfrac12v^2+gy=\tfrac12({fx(v3, 2)})^2+{g}({fx(y3, 3)})={fx(0.5 * v3**2 + g * y3, 3)}\ \mathrm{{J/kg}}=\tfrac12v_0^2={fx(0.5 * v0**2, 3)}"]),
        ("Errores comunes y límites de validez",
         r"Un error frecuente es creer que en el punto más alto la aceleración es cero: la velocidad es nula pero la aceleración sigue siendo $-g$. Otro es tomar $g$ con el signo equivocado al mezclar convenciones (eje hacia arriba o hacia abajo). También se aplica $t_{\mathrm{vuelo}}=2v_0/g$ cuando el aterrizaje ocurre a otra altura, o se descarta la raíz negativa sin justificar. El modelo falla cuando la resistencia del aire es importante (cuerpos ligeros, velocidades altas), donde aparece una velocidad terminal $v_T$; con alturas comparables al radio terrestre $g$ deja de ser constante y la trayectoria es una elipse kepleriana.",
         [r"m\ddot y=-mg-b\dot y\;\Rightarrow\;v_T=\dfrac{mg}{b};\qquad g(h)=g_0\left(\dfrac{R_T}{R_T+h}\right)^{2}"]),
    ]
    return dict(concept=concept)


# ====================================================================== cap. 4
def _projectile() -> dict:
    v0, th, g = c4.V0, c4.THETA_DEG, c4.G
    an = c4.projectile_analytic()
    thr = np.deg2rad(th)
    vx, vy = v0 * np.cos(thr), v0 * np.sin(thr)
    h0 = 10.0
    r_h = vx / g * (vy + np.sqrt(vy**2 + 2 * g * h0))
    th_h = np.rad2deg(np.arctan(v0 / np.sqrt(v0**2 + 2 * g * h0)))
    r_max = v0**2 / g
    concept = [
        ("Qué se estudia y por qué importa",
         r"El movimiento de proyectiles describe cualquier cuerpo que, tras un impulso inicial, se mueve solo bajo la gravedad: balones, chorros de agua, piedras, y también bolas de cañón en balística elemental. Su estudio histórico por Galileo mostró que los movimientos horizontal y vertical son independientes, un paso decisivo hacia la mecánica newtoniana. Hoy sigue siendo la base de la balística exterior, del diseño de instalaciones deportivas y de la intuición sobre órbitas (una órbita es un proyectil que cae alrededor de la Tierra).",
         [r"\ddot x=0,\qquad \ddot y=-g"]),
        ("Magnitudes y convenciones",
         r"Se usa un sistema con $x$ horizontal e $y$ vertical hacia arriba, y el origen en el punto de lanzamiento. Las magnitudes son la rapidez inicial $v_0$ (m/s), el ángulo de elevación $\theta$ sobre la horizontal y $g=9.81\ \mathrm{m/s^2}$. De ellas se derivan el tiempo de vuelo (s), la altura máxima $H$ (m) y el alcance $R$ (m). Para $v_0=25\ \mathrm{m/s}$ los órdenes de magnitud son segundos y decenas de metros; para un proyectil de artillería, de kilómetros.",
         [r"v_{0x}=v_0\cos\theta,\quad v_{0y}=v_0\sin\theta,\qquad [R]=[H]=\mathrm{m},\quad [T]=\mathrm{s}"]),
        ("Intuición y analogías",
         r"El proyectil se mueve con velocidad horizontal constante mientras cae con aceleración vertical constante, como un objeto que rueda con rapidez uniforme por una mesa y que se deja caer a la vez: ambas bolas llegan al suelo en el mismo instante. La trayectoria es una parábola porque la posición horizontal es lineal en $t$ y la vertical cuadrática. Sin gravedad el proyectil seguiría una recta, y la gravedad añade la «caída» $\tfrac12gt^2$ respecto de esa recta.",
         [r"\vec r(t)=\vec v_0\,t+\tfrac12\vec g\,t^{2}"]),
        ("Lagrangiano y simetrías",
         r"En el plano la energía cinética es $\tfrac12m(\dot x^2+\dot y^2)$ y la potencial $mgy$, de modo que $\mathcal L=T-V$ no depende de $x$: la coordenada $x$ es cíclica. Por Noether (invariancia frente a traslaciones horizontales) se conserva el momento conjugado $p_x$, es decir, la velocidad horizontal. Además $\mathcal L$ no depende explícitamente de $t$ y se conserva la energía. Las ecuaciones de Euler-Lagrange desacoplan el problema en dos movimientos unidimensionales independientes, lo que justifica a Galileo.",
         [r"\mathcal L=\tfrac12m(\dot x^{2}+\dot y^{2})-mgy,\qquad \dfrac{\partial\mathcal L}{\partial x}=0\;\Rightarrow\;p_x=m\dot x=\text{const}",
          r"m\ddot x=0,\qquad m\ddot y=-mg,\qquad E=\tfrac12m(\dot x^{2}+\dot y^{2})+mgy=\text{const}"]),
        ("Integración y trayectoria",
         r"Con las condiciones iniciales $(x,y)=(0,0)$ y $\vec v=v_0(\cos\theta,\sin\theta)$ la integración de cada ecuación es inmediata. Eliminando el tiempo ($t=x/v_0\cos\theta$, posible si $\cos\theta\ne0$) se obtiene la ecuación de la trayectoria, una parábola de eje vertical. La forma en términos de $\tan\theta$ es útil para el problema inverso: dado el blanco $(x,y)$, hallar los ángulos de tiro.",
         [r"x=v_0\cos\theta\,t,\qquad y=v_0\sin\theta\,t-\tfrac12gt^{2}",
          r"y=x\tan\theta-\dfrac{g\,x^{2}}{2v_0^{2}\cos^{2}\theta}=x\tan\theta-\dfrac{g\,x^{2}}{2v_0^{2}}\left(1+\tan^{2}\theta\right)"]),
        ("Tiempo de vuelo, altura máxima y alcance",
         r"El vértice ocurre cuando $v_y=0$ y el impacto cuando $y=0$ de nuevo (terreno plano, raíz distinta de cero). La simetría de la parábola implica que el tiempo de vuelo es el doble del de ascenso. El alcance es la distancia horizontal recorrida en ese tiempo, y el producto $2\sin\theta\cos\theta=\sin2\theta$ lo escribe de forma compacta. El resultado es simétrico respecto de $45^\circ$: los ángulos complementarios $\theta$ y $90^\circ-\theta$ dan el mismo alcance, con distintas alturas y tiempos.",
         [r"T=\dfrac{2v_0\sin\theta}{g},\qquad H=\dfrac{v_0^{2}\sin^{2}\theta}{2g},\qquad R=v_0\cos\theta\,T=\dfrac{v_0^{2}\sin2\theta}{g}",
          r"R(\theta)=R(90^\circ-\theta),\qquad \dfrac{H}{R}=\dfrac14\tan\theta"]),
        ("Optimización, envolvente y altura de lanzamiento",
         r"El alcance es máximo cuando $dR/d\theta=0$, es decir $\cos2\theta=0$, y es un máximo porque $d^2R/d\theta^2<0$ allí: $\theta^*=45^\circ$ y $R_{\max}=v_0^2/g$. Si se lanza desde una altura $h_0$ sobre el suelo, el alcance depende de la raíz de la cuadrática de vuelo y el ángulo óptimo es menor que $45^\circ$. Todos los puntos alcanzables están dentro de la parábola de seguridad, envolvente de la familia de trayectorias, que se obtiene exigiendo que la ecuación cuadrática en $\tan\theta$ tenga discriminante nulo.",
         [r"\dfrac{dR}{d\theta}=\dfrac{2v_0^{2}\cos2\theta}{g}=0\;\Rightarrow\;\theta^{*}=45^\circ,\qquad R_{\max}=\dfrac{v_0^{2}}{g}",
          r"R(\theta,h_0)=\dfrac{v_0\cos\theta}{g}\Big(v_0\sin\theta+\sqrt{v_0^{2}\sin^{2}\theta+2gh_0}\Big),\qquad \tan\theta^{*}=\dfrac{v_0}{\sqrt{v_0^{2}+2gh_0}}",
          r"y_{\mathrm{env}}(x)=\dfrac{v_0^{2}}{2g}-\dfrac{g\,x^{2}}{2v_0^{2}}"]),
        ("Resultado general",
         r"Para un lanzamiento sobre terreno plano con rapidez $v_0$ y ángulo $\theta$ los resultados son función de los dos parámetros y la escala $v_0^2/g$ (longitud) o $v_0/g$ (tiempo). Con las variables adimensionales $\tilde x=gx/v_0^2$, $\tilde y=gy/v_0^2$ la trayectoria no depende de $g$ ni de $v_0$ por separado, solo de $\theta$.",
         [r"\boxed{\,T=\dfrac{2v_0\sin\theta}{g},\qquad H=\dfrac{v_0^{2}\sin^{2}\theta}{2g},\qquad R=\dfrac{v_0^{2}\sin2\theta}{g},\qquad \theta^{*}=45^\circ\,}",
          r"\tilde y=\tilde x\tan\theta-\dfrac{\tilde x^{2}}{2\cos^{2}\theta},\qquad [R]=\dfrac{(\mathrm{m/s})^2}{\mathrm{m/s^2}}=\mathrm{m}"]),
        ("Particularización al problema",
         rf"Con $v_0={v0:.0f}$ m/s, $\theta={th:.0f}^\circ$ y $g={g}$ m/s$^2$ las componentes iniciales son $v_{{0x}}={fx(vx, 3)}$ m/s y $v_{{0y}}={fx(vy, 3)}$ m/s. La fórmula general reproduce los valores analíticos de la escena. Para ilustrar el efecto de la altura, desde $h_0={h0:.0f}$ m el mismo tiro llegaría a {fx(r_h, 2)} m y el ángulo óptimo bajaría a {fx(th_h, 2)}$^\circ$.",
         [rf"T=\dfrac{{2({v0:.0f})\sin{th:.0f}^\circ}}{{{g}}}={fx(an['t_flight'], 3)}\ \mathrm{{s}},\qquad H=\dfrac{{({v0:.0f}\sin{th:.0f}^\circ)^2}}{{2({g})}}={fx(an['height'], 3)}\ \mathrm{{m}}",
          rf"R=\dfrac{{({v0:.0f})^2\sin{2 * th:.0f}^\circ}}{{{g}}}={fx(an['range'], 3)}\ \mathrm{{m}},\qquad R_{{\max}}=\dfrac{{v_0^2}}{{g}}={fx(r_max, 3)}\ \mathrm{{m}}",
          rf"R=v_{{0x}}T={fx(vx, 3)}({fx(an['t_flight'], 3)})={fx(vx * an['t_flight'], 3)}\ \mathrm{{m}},\qquad \dfrac{{R}}{{R_{{\max}}}}=\sin{2 * th:.0f}^\circ={fx(np.sin(2 * thr), 4)}"]),
        ("Errores comunes y límites de validez",
         r"Un error habitual es pensar que la componente horizontal de la velocidad disminuye durante el vuelo, o que el proyectil «pierde» impulso al subir; solo cambia $v_y$. También se aplica $R=v_0^2\sin2\theta/g$ cuando el blanco está a otra altura, o se supone que $45^\circ$ es óptimo con altura inicial no nula. En el vacío el modelo es exacto en campo uniforme; con aire, la resistencia reduce el alcance y el ángulo óptimo baja a unos $30^\circ$ a $40^\circ$, la trayectoria es asimétrica y la ecuación pasa a ser $m\ddot{\vec r}=m\vec g-c|\vec v|\vec v$, sin solución cerrada. A alcances de cientos de kilómetros se añaden la curvatura terrestre y la fuerza de Coriolis.",
         [r"m\ddot{\vec r}=m\vec g-c\,|\vec v|\,\vec v;\qquad \text{Coriolis:}\ \vec a_C=-2\vec\Omega\times\vec v"]),
    ]
    return dict(concept=concept)


def _circular() -> dict:
    R0, P = c4.RADIUS, c4.PERIOD
    ca = c4.circular_analytic()
    om, v, ac = ca["omega"], ca["speed"], ca["a_c"]
    fd = c4.fd_error_table(hs=(0.01,))[0]
    h = 0.01
    err_h2 = om**4 * R0 * h**2 / 12  # error relativo-absoluto de la 2a diferencia sobre a: a*(w h)^2/12
    freq = 1 / P
    m_ej = 1.0
    concept = [
        ("Qué se estudia y por qué importa",
         r"El movimiento circular uniforme describe satélites en órbita circular, ruedas y rotores, electrones en un campo magnético y la rotación de la Tierra. Es el caso más simple de movimiento acelerado con rapidez constante: la aceleración existe porque cambia la dirección de la velocidad, no su módulo. Su comprensión es indispensable para dinámica de rotación, para ondas (el fasor que gira) y para la teoría de órbitas. Revela la fuerza centrípeta como una necesidad cinemática y no como una nueva fuerza.",
         [r"\vec r(t)=R(\cos\omega t,\ \sin\omega t),\qquad \omega=\dfrac{2\pi}{T}"]),
        ("Magnitudes y convenciones",
         r"El movimiento se caracteriza por el radio $R$ (m), el periodo $T$ (s), la frecuencia $f=1/T$ (Hz) y la velocidad angular $\omega=2\pi/T$ (rad/s). La rapidez lineal es $v=\omega R$ (m/s) y la aceleración centrípeta $a_c=\omega^2R=v^2/R$ (m/s$^2$) apunta siempre hacia el centro. Se adopta sentido antihorario como positivo. El radián es adimensional, por lo que $\mathrm{rad/s}$ y $\mathrm{s^{-1}}$ son equivalentes dimensionalmente aunque se distinguen por claridad. Las ruedas de un auto tienen $\omega\sim 10^2\ \mathrm{rad/s}$ y la Tierra $7.3\times10^{-5}\ \mathrm{rad/s}$.",
         [r"[\omega]=\mathrm{s^{-1}},\quad [v]=\mathrm{m/s},\quad [a_c]=\mathrm{m/s^{2}},\quad f=\dfrac1T=\dfrac{\omega}{2\pi}"]),
        ("Intuición y analogías",
         r"Una piedra atada a una cuerda gira porque la cuerda tira constantemente hacia el centro; si la cuerda se rompe, la piedra sale por la tangente con la velocidad que tenía, porque la inercia no cambia la dirección por sí sola. La aceleración centrípeta es la rapidez con que la velocidad gira: al recorrer un ángulo $d\varphi$ el vector velocidad cambia $v\,d\varphi$. Es perpendicular a $\vec v$ y por eso no altera el módulo, igual que la fuerza magnética no realiza trabajo.",
         [r"\left|\dfrac{d\vec v}{dt}\right|=v\,\dfrac{d\varphi}{dt}=v\,\omega=\dfrac{v^{2}}{R}"]),
        ("Cinemática en coordenadas polares",
         r"Se usan los vectores unitarios locales $\hat r$ y $\hat\varphi$, que dependen del ángulo: $d\hat r/d\varphi=\hat\varphi$ y $d\hat\varphi/d\varphi=-\hat r$. Aplicando la regla de la cadena se obtienen velocidad y aceleración para un movimiento plano general. Esta descomposición separa la parte radial de la azimutal y es válida en todo punto salvo el origen. La aceleración tiene un término radial $\ddot r-r\dot\varphi^2$ (que incluye la centrípeta) y uno azimutal $r\ddot\varphi+2\dot r\dot\varphi$ (con la parte de Coriolis).",
         [r"\vec r=r\hat r,\qquad \vec v=\dot r\,\hat r+r\dot\varphi\,\hat\varphi",
          r"\vec a=\big(\ddot r-r\dot\varphi^{2}\big)\hat r+\big(r\ddot\varphi+2\dot r\dot\varphi\big)\hat\varphi"]),
        ("Condición de movimiento circular uniforme",
         r"La restricción $r=R=\text{const}$ elimina las derivadas de $r$, y la condición de uniformidad, $\dot\varphi=\omega=\text{const}$, anula $\ddot\varphi$. Solo sobrevive el término radial negativo: la aceleración es puramente centrípeta. Es también la solución general con $\ddot\varphi=0$, porque integrando se obtiene $\varphi=\varphi_0+\omega t$. El módulo de la velocidad es constante, pero su vector no: $\vec v\cdot\vec a=0$ en todo instante.",
         [r"r=R,\ \dot\varphi=\omega\;\Rightarrow\;\vec v=R\omega\,\hat\varphi,\qquad \vec a=-R\omega^{2}\,\hat r=-\omega^{2}\vec r",
          r"\vec v\cdot\vec a=0,\qquad \dfrac{d}{dt}|\vec v|^{2}=2\vec v\cdot\vec a=0"]),
        ("Dinámica: lagrangiano con ligadura y fuerza centrípeta",
         r"En coordenadas polares la energía cinética libre es $T=\tfrac12m(\dot r^2+r^2\dot\varphi^2)$. La ligadura $r=R$ se impone con un multiplicador de Lagrange $\lambda$, que resulta ser (menos) la fuerza de ligadura. La ecuación radial da la tensión necesaria, $F_c=m\omega^2R=mv^2/R$, y la azimutal da la conservación del momento angular $p_\varphi=mR^2\dot\varphi$, consecuencia de la simetría rotacional (Noether). Sin fuerza tangencial el movimiento es uniforme, y la fuerza centrípeta no realiza trabajo porque es perpendicular al desplazamiento.",
         [r"\mathcal L=\tfrac12m(\dot r^{2}+r^{2}\dot\varphi^{2})+\lambda\,(r-R)\;\Rightarrow\;\begin{cases}m\ddot r-mr\dot\varphi^{2}=\lambda\\ \dfrac{d}{dt}\big(mr^{2}\dot\varphi\big)=0\end{cases}",
          r"\lambda=-mR\omega^{2}=-\dfrac{mv^{2}}{R},\qquad p_\varphi=mR^{2}\omega=\text{const},\qquad W=\int\vec F_c\cdot d\vec r=0"]),
        ("Representación compleja y fasores",
         r"Identificando el plano con $\mathbb C$, la posición es $z=Re^{i\omega t}$ y derivar es multiplicar por $i\omega$, es decir, girar $90^\circ$ y escalar. Así se ve de inmediato que $v$ está adelantada $90^\circ$ respecto de la posición y $a$ lo está $180^\circ$. Cada proyección del movimiento circular es un movimiento armónico simple, por lo que el círculo es la unión de dos oscilaciones en cuadratura. Esta equivalencia explica por qué las oscilaciones se estudian con fasores.",
         [r"z(t)=Re^{i\omega t},\quad \dot z=i\omega z,\quad \ddot z=-\omega^{2}z,\qquad x=R\cos\omega t,\ \ \ddot x=-\omega^{2}x"]),
        ("Resultado general",
         r"Para radio $R$ y periodo $T$ la velocidad angular, la rapidez y la aceleración centrípeta son: todas dependen solo de $R$ y $T$ y tienen las dimensiones adecuadas. La fuerza centrípeta necesaria para una masa $m$ es $m a_c$, proporcional a $R/T^2$. Para mantener la fuerza fija, duplicar la rapidez exige cuadruplicar el cociente $1/R$ (es decir, reducir $R$ a la cuarta parte).",
         [r"\boxed{\,\omega=\dfrac{2\pi}{T},\qquad v=\dfrac{2\pi R}{T},\qquad a_c=\dfrac{v^{2}}{R}=\dfrac{4\pi^{2}R}{T^{2}},\qquad F_c=\dfrac{4\pi^{2}mR}{T^{2}}\,}",
          r"[a_c]=\dfrac{\mathrm{m}}{\mathrm{s}^2},\qquad [F_c]=\mathrm{kg\,m\,s^{-2}}=\mathrm{N}"]),
        ("Particularización al problema",
         rf"Con $R={R0:.0f}$ m y $T={P:.0f}$ s la frecuencia es $f={fx(freq, 2)}$ Hz. Las expresiones generales reproducen los valores de la escena. Para ilustrar la fuerza, una masa de {m_ej:.0f} kg requeriría {fx(m_ej * ac, 4)} N. Las diferencias finitas centradas de la escena estiman $v$ y $a_c$ con error de orden $h^2$; con $h={h}$ s el error esperado de $a_c$ es {sci(err_h2, 3)} m/s$^2$.",
         [rf"\omega=\dfrac{{2\pi}}{{{P:.0f}}}={fx(om, 4)}\ \mathrm{{rad/s}},\qquad v=\omega R={fx(v, 4)}\ \mathrm{{m/s}}",
          rf"a_c=\dfrac{{4\pi^2({R0:.0f})}}{{({P:.0f})^2}}={fx(ac, 4)}\ \mathrm{{m/s^2}}=\dfrac{{v^2}}{{R}}={fx(v**2 / R0, 4)}\ \mathrm{{m/s^2}}",
          rf"v_{{FD}}={fx(fd['speed'], 4)}\ \mathrm{{m/s}},\qquad a_{{c,FD}}={fx(fd['a_c'], 4)}\ \mathrm{{m/s^2}}"]),
        ("Errores comunes y límites de validez",
         r"Se suele afirmar que, como la rapidez es constante, la aceleración es nula; la aceleración es un vector y cambia con la dirección. Se confunde la fuerza centrípeta con una fuerza adicional, cuando es la resultante de fuerzas reales (tensión, gravedad, rozamiento); la «fuerza centrífuga» solo aparece en el sistema rotante. Es común usar $\omega$ en grados por segundo con fórmulas pensadas para radianes. El modelo uniforme deja de valer si hay fuerza tangencial ($\dot\varphi$ variable, aparece $a_t=R\dot\omega$), si el radio cambia (órbita elíptica, regida por la conservación de $L$ y la ley de Kepler), o a velocidades relativistas, donde la masa efectiva $\gamma m$ reemplaza a $m$.",
         [r"\vec a=-\omega^{2}R\,\hat r+R\dot\omega\,\hat\varphi\ \ (\dot\omega\ne0);\qquad F_c=\gamma m\,\dfrac{v^{2}}{R}\ \ (v\sim c)"]),
    ]
    return dict(concept=concept)


DETAIL = {
    "DimensionalAnalysis": _dimensional(),
    "ErrorPropagation": _error_prop(),
    "VectorAddition": _vec_add(),
    "VectorProducts": _vec_prod(),
    "InstantaneousVelocity": _inst_vel(),
    "ConstantAcceleration": _const_acc(),
    "Projectile": _projectile(),
    "CircularMotion": _circular(),
}
