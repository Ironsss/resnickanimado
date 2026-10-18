"""Desarrollo analitico general (concept) para Volumen 1, capitulos 5 a 8.

Escenas: StaticEquilibrium, InclinePulley, FrictionIncline, DragFall,
SpringWork, WorkEnergyTheorem, PendulumEnergy, SpringFriction.
Cada escena: conceptual, desarrollo lagrangiano/hamiltoniano general, formula general,
particularizacion numerica (calculada con physics/), errores y limites.
"""
from __future__ import annotations

import numpy as np

from content.texfmt import fx, sci
from physics import ch05_newton as c5
from physics import ch06_dynamics as c6
from physics import ch07_work_energy as c7
from physics import ch08_energy_conservation as c8


# =========================================================== 1. StaticEquilibrium
def _static() -> dict:
    w, al, be = c5.WEIGHT, c5.ALPHA_DEG, c5.BETA_DEG
    a, b = np.deg2rad(al), np.deg2rad(be)
    t1, t2 = c5.tensions_analytic()
    sab = np.sin(a + b)
    sym = w / (2 * np.sin(np.deg2rad(40.0)))
    return dict(concept=[
        ("Qué se estudia y por qué importa",
         r"El equilibrio estático de una partícula suspendida de cables es el problema básico de cualquier estructura que cuelga: lámparas, puentes colgantes, grúas, tirantes de torres y arneses de seguridad. La pregunta práctica es qué tensión debe soportar cada cable, porque de ella depende el diámetro y el material elegidos. Un cable que trabaja casi horizontal soporta tensiones mucho mayores que el peso colgado, y por eso los cables de una línea eléctrica nunca quedan perfectamente tensos. Este análisis parte del principio general de la estática y lo resuelve para ángulos y peso arbitrarios antes de usar cifras.",
         []),
        ("Magnitudes y convenciones",
         r"El peso $W=mg$ se mide en newton y actúa verticalmente hacia abajo; los ángulos $\alpha$ y $\beta$ se miden desde la horizontal hacia arriba, de modo que ambos pertenecen al intervalo $(0,\pi/2)$. Las tensiones $T_1$ y $T_2$ son magnitudes positivas porque un cable solo puede tirar, nunca empujar; un resultado negativo señalaría una hipótesis geométrica imposible. Con $W$ de cientos de newton, los cables de uso doméstico trabajan en el rango de cientos de newton y los estructurales en el de kilonewton. El eje $x$ es horizontal y el eje $y$ vertical hacia arriba.",
         [r"[W]=\mathrm{N},\qquad [T_i]=\mathrm{N},\qquad \alpha,\beta\ \text{(rad o grados, adimensionales)}"]),
        ("Intuición y analogías",
         r"Tres fuerzas que mantienen una partícula en reposo forman un triángulo cerrado cuando se colocan una a continuación de otra; los lados del triángulo son proporcionales a las magnitudes. La situación es análoga a un tira y afloja en el que dos equipos tiran en direcciones distintas contra una carga: cuanto más abierto el ángulo entre los equipos, mayor es el esfuerzo que cada uno debe hacer para anular la carga. Los cables casi horizontales son ineficaces para sostener peso, porque su componente vertical es pequeña y la mayor parte de su tensión se emplea en tirar uno contra otro.",
         []),
        ("Principio variacional: trabajos virtuales y multiplicadores de Lagrange",
         r"El punto de partida es el principio de los trabajos virtuales: una partícula está en equilibrio si y solo si el trabajo de las fuerzas activas en cualquier desplazamiento virtual compatible con las ligaduras es nulo. Aquí las ligaduras son que cada cable conserva su longitud $\ell_i$ al estar tenso, $g_i(\vec r)=|\vec r-\vec A_i|-\ell_i=0$, con $\vec A_i$ el punto de anclaje. La energía potencial es solo la gravitatoria, $V=Wy$, y el equilibrio es un punto estacionario de $V$ restringido, que se trata con multiplicadores. Los multiplicadores resultan ser precisamente las tensiones, porque son las fuerzas de ligadura que mantienen cada longitud.",
         [r"\Lambda(\vec r,\lambda_1,\lambda_2)=W\,y+\lambda_1\,g_1(\vec r)+\lambda_2\,g_2(\vec r)",
          r"\nabla\Lambda=\vec 0\;\Longrightarrow\; W\,\hat y+\lambda_1\hat n_1+\lambda_2\hat n_2=\vec 0,\qquad \hat n_i=\dfrac{\vec r-\vec A_i}{|\vec r-\vec A_i|}",
          r"T_i\equiv-\lambda_i\;\Longrightarrow\;\vec T_1+\vec T_2+\vec W=\vec 0,\quad \vec T_i=-T_i\hat n_i"]),
        ("Condiciones de equilibrio en componentes",
         r"Con $\hat n_1$ y $\hat n_2$ expresados mediante los ángulos de los cables, la ecuación vectorial se descompone en dos ecuaciones escalares independientes, una por eje. Descomponer es legítimo porque los vectores de la base cartesiana son linealmente independientes, de modo que cada componente de una suma nula debe anularse por separado. El cable izquierdo tira hacia arriba y a la izquierda, el derecho hacia arriba y a la derecha. El resultado es un sistema lineal de dos ecuaciones con las incógnitas $T_1$ y $T_2$, cuyos coeficientes dependen solo de la geometría.",
         [r"\sum F_x=0:\qquad -T_1\cos\alpha+T_2\cos\beta=0",
          r"\sum F_y=0:\qquad T_1\sin\alpha+T_2\sin\beta-W=0",
          r"\begin{pmatrix}\cos\alpha&-\cos\beta\\ \sin\alpha&\sin\beta\end{pmatrix}\begin{pmatrix}T_1\\T_2\end{pmatrix}=\begin{pmatrix}0\\W\end{pmatrix}"]),
        ("Resolución del sistema y teorema de Lamy",
         r"El sistema tiene solución única si el determinante de la matriz de coeficientes no se anula. Ese determinante es $\cos\alpha\sin\beta+\sin\alpha\cos\beta=\sin(\alpha+\beta)$, que es positivo mientras $0<\alpha+\beta<\pi$, y por tanto la solución existe y es única en todo el dominio físico. La regla de Cramer entrega las tensiones directamente. El resultado equivale al teorema de Lamy, que afirma que cada fuerza es proporcional al seno del ángulo formado por las otras dos, y es la expresión matemática de la regla del triángulo de fuerzas.",
         [r"\Delta=\cos\alpha\sin\beta+\sin\alpha\cos\beta=\sin(\alpha+\beta)",
          r"T_1=\dfrac{1}{\Delta}\begin{vmatrix}0&-\cos\beta\\W&\sin\beta\end{vmatrix}=\dfrac{W\cos\beta}{\sin(\alpha+\beta)},\qquad T_2=\dfrac{1}{\Delta}\begin{vmatrix}\cos\alpha&0\\ \sin\alpha&W\end{vmatrix}=\dfrac{W\cos\alpha}{\sin(\alpha+\beta)}",
          r"\dfrac{T_1}{\cos\beta}=\dfrac{T_2}{\cos\alpha}=\dfrac{W}{\sin(\alpha+\beta)}"]),
        ("Parámetros adimensionales y casos límite",
         r"Al dividir por el peso, las tensiones relativas $\tau_i=T_i/W$ dependen solo de los dos ángulos, de modo que el problema tiene un único factor de escala, $W$. Si los dos cables son simétricos, $\alpha=\beta$, ambas tensiones coinciden y valen $W/(2\sin\alpha)$, con la mitad del peso soportada por cada rama. Cuando $\alpha+\beta\to0$ el denominador se anula y las tensiones crecen sin cota como $W/(\alpha+\beta)$: un cable perfectamente horizontal no puede sostener ningún peso sin tensión infinita. Si $\alpha\to\pi/2$ con $\beta$ fijo, $T_2\to0$ y todo el peso pasa al cable vertical.",
         [r"\tau_1=\dfrac{\cos\beta}{\sin(\alpha+\beta)},\qquad \tau_2=\dfrac{\cos\alpha}{\sin(\alpha+\beta)}",
          r"\alpha=\beta:\quad T_1=T_2=\dfrac{W}{2\sin\alpha};\qquad \alpha+\beta\to0:\quad T_i\simeq\dfrac{W}{\alpha+\beta}\to\infty",
          r"\dfrac{\partial T_1}{\partial\alpha}=-\dfrac{W\cos\beta\cos(\alpha+\beta)}{\sin^2(\alpha+\beta)}"]),
        ("Resultado general",
         r"La fórmula general muestra que cada tensión es el peso multiplicado por el coseno del ángulo del otro cable y dividido por el seno del ángulo total entre los cables. Es homogénea de grado uno en $W$, depende de los ángulos mediante funciones trigonométricas y es dimensionalmente consistente: $W$ en newton multiplica factores adimensionales, de modo que $T_i$ queda en newton. Es válida para cualesquiera $W>0$, $\alpha,\beta\in(0,\pi/2)$ y contiene todos los casos particulares, incluido el simétrico.",
         [r"\boxed{\;T_1=\dfrac{W\cos\beta}{\sin(\alpha+\beta)},\qquad T_2=\dfrac{W\cos\alpha}{\sin(\alpha+\beta)}\;}",
          r"[T_i]=\mathrm{N}\cdot 1=\mathrm{N}"]),
        ("Particularización al problema",
         r"Se sustituyen los datos del enunciado en la fórmula general: el peso, el ángulo izquierdo y el ángulo derecho. El denominador común es el seno de la suma de ambos ángulos, y los numeradores usan el coseno del ángulo opuesto. Las cifras coinciden con las del cálculo paso a paso de la escena y con la solución numérica del sistema lineal.",
         [rf"W={sci(w,4)}\ \mathrm{{N}},\quad \alpha={fx(al,1)}^\circ,\quad \beta={fx(be,1)}^\circ,\quad \sin(\alpha+\beta)=\sin 75^\circ={sci(sab,5)}",
          rf"T_1=\dfrac{{({sci(w,4)})\cos {fx(be,1)}^\circ}}{{{sci(sab,5)}}}={sci(t1,5)}\ \mathrm{{N}}",
          rf"T_2=\dfrac{{({sci(w,4)})\cos {fx(al,1)}^\circ}}{{{sci(sab,5)}}}={sci(t2,5)}\ \mathrm{{N}}",
          rf"\text{{Control simétrico }}(\alpha=\beta=40^\circ):\ T={sci(sym,4)}\ \mathrm{{N}}"]),
        ("Errores comunes y límites de validez",
         r"Un error frecuente es suponer que las tensiones son iguales solo porque el peso cuelga del centro del nudo; solo lo son si los ángulos coinciden. Otro es medir los ángulos desde la vertical, lo que intercambia senos y cosenos y produce tensiones erróneas. También se confunde la suma de las tensiones con el peso: la suma de las magnitudes siempre supera a $W$, porque solo las componentes verticales se suman. El modelo supone cables inextensibles y sin masa, un nudo puntual y equilibrio estático; si el cable tiene masa apreciable adopta forma de catenaria, y si se estira, la geometría cambia con la carga y debe resolverse un problema elástico acoplado.",
         []),
    ])


# =========================================================== 2. InclinePulley
def _pulley() -> dict:
    m1, m2, th, g = c5.M1, c5.M2, c5.THETA_DEG, c5.G
    a, t = c5.pulley_analytic()
    s = np.sin(np.deg2rad(th))
    mm = m1 + m2
    tb = c5.balance_angle_analytic()
    atw = (m2 - m1) * g / mm
    return dict(concept=[
        ("Qué se estudia y por qué importa",
         r"Dos cuerpos unidos por una cuerda que pasa por una polea forman el sistema más sencillo en el que una ligadura acopla dos movimientos. Aparece en ascensores con contrapeso, grúas, funiculares y en la máquina de Atwood, con la que se midió la aceleración de la gravedad de forma casi estática. El interés conceptual es que el problema tiene un solo grado de libertad aunque haya dos masas, y que la tensión de la cuerda es una fuerza de ligadura, no un dato. Este análisis la obtiene a partir del formalismo de Lagrange para $m_1$, $m_2$ y ángulo $\theta$ arbitrarios.",
         []),
        ("Magnitudes y convenciones",
         r"El cuerpo $m_1$ se apoya sobre un plano inclinado liso un ángulo $\theta$ sobre la horizontal y $m_2$ cuelga verticalmente. La coordenada $s$ mide el desplazamiento de $m_1$ hacia arriba por el plano, y es positiva cuando $m_2$ desciende la misma distancia. La aceleración $a=\ddot s$ se da en $\mathrm{m/s^2}$ y la tensión $T$ en newton; la aceleración de la gravedad es $g\simeq9{,}81\ \mathrm{m/s^2}$. La cuerda es inextensible y sin masa, y la polea carece de masa y de rozamiento.",
         [r"[a]=\mathrm{m/s^2},\qquad [T]=\mathrm{N},\qquad [m_i]=\mathrm{kg},\qquad 0\le\theta\le\pi/2"]),
        ("Intuición y analogías",
         r"El sistema se comporta como un solo cuerpo de masa $m_1+m_2$ que se mueve a lo largo de una coordenada curvada, empujado por el peso neto de $m_2$ y frenado por la componente del peso de $m_1$ a lo largo del plano. La aceleración es, por tanto, la fuerza neta motriz dividida por la masa total que debe acelerarse. La cuerda solo transmite fuerza: es como un pasamanos que reparte la carga entre las dos masas.",
         []),
        ("Lagrangiano con ligadura holónoma",
         r"Las coordenadas cartesianas de las dos masas están ligadas por la inextensibilidad de la cuerda, que es una ligadura holónoma: $m_1$ sube $s$ por el plano y $m_2$ baja $s$. Se elige $s$ como única coordenada generalizada, lo que elimina la ligadura por construcción y deja un sistema con un grado de libertad. La energía cinética total es la de ambas masas con la misma rapidez $\dot s$ y la energía potencial reúne el peso de cada cuerpo. El lagrangiano es $L=T-V$ y no depende explícitamente del tiempo.",
         [r"T_{\rm cin}=\tfrac12 m_1\dot s^{2}+\tfrac12 m_2\dot s^{2}=\tfrac12(m_1+m_2)\dot s^{2}",
          r"V=m_1 g\,s\sin\theta-m_2 g\,s=(m_1\sin\theta-m_2)\,g\,s",
          r"L(s,\dot s)=\tfrac12(m_1+m_2)\dot s^{2}-(m_1\sin\theta-m_2)\,g\,s"]),
        ("Ecuación de Euler-Lagrange y aceleración",
         r"La ecuación de Euler-Lagrange garantiza que la trayectoria real hace estacionaria la acción y es equivalente a la segunda ley de Newton para la coordenada generalizada. Al derivar, el momento canónico es $p=(m_1+m_2)\dot s$ y la fuerza generalizada es el término constante procedente del potencial lineal. La aceleración resulta uniforme, puesto que la fuerza neta no depende de $s$, y la integración es inmediata con las condiciones de partida del reposo. El signo de $a$ indica el sentido del movimiento: positivo si desciende $m_2$.",
         [r"\dfrac{d}{dt}\dfrac{\partial L}{\partial\dot s}-\dfrac{\partial L}{\partial s}=0\;\Longrightarrow\;(m_1+m_2)\ddot s+(m_1\sin\theta-m_2)\,g=0",
          r"a=\ddot s=\dfrac{m_2-m_1\sin\theta}{m_1+m_2}\,g",
          r"s(t)=\tfrac12 a t^{2},\qquad \dot s(t)=a\,t\qquad(s_0=\dot s_0=0)"]),
        ("Hamiltoniano, energía y simetrías",
         r"La transformada de Legendre del lagrangiano da el hamiltoniano, que aquí coincide con la energía mecánica porque la ligadura no depende del tiempo. Como $L$ no depende explícitamente de $t$, el teorema de Noether para traslaciones temporales asegura que $H$ es una integral primera. La traslación en $s$ no es una simetría exacta, ya que el potencial lineal cambia por una constante, y por eso el momento no se conserva: $\dot p=(m_2-m_1\sin\theta)\,g$ es justamente la fuerza neta. La energía conservada da $\dot s^2=2as$, la relación habitual entre velocidad y desplazamiento.",
         [r"H=p\dot s-L=\dfrac{p^{2}}{2(m_1+m_2)}+(m_1\sin\theta-m_2)\,g\,s",
          r"\dot s=\dfrac{\partial H}{\partial p},\qquad \dot p=-\dfrac{\partial H}{\partial s}=(m_2-m_1\sin\theta)\,g",
          r"\dfrac{dH}{dt}=-\dfrac{\partial L}{\partial t}=0\;\Longrightarrow\;\dot s^{2}=2a\,s"]),
        ("La tensión como fuerza de ligadura",
         r"Para obtener la tensión hay que relajar la ligadura e introducirla con un multiplicador de Lagrange $\lambda$, que representa la fuerza que la cuerda ejerce sobre cada masa. Las ecuaciones de cada cuerpo, con coordenadas independientes $s_1$ y $s_2$ y ligadura $s_1-s_2=0$, dan dos ecuaciones que, junto con la ligadura, determinan $a$ y $T$. Sustituyendo la aceleración ya calculada en la ecuación de la masa colgante se obtiene la tensión, que es siempre menor que el peso de $m_2$ cuando $m_2$ acelera hacia abajo.",
         [r"m_1\ddot s=-m_1 g\sin\theta+T,\qquad m_2\ddot s=m_2 g-T",
          r"T=m_2(g-a)=\dfrac{m_1m_2\,g\,(1+\sin\theta)}{m_1+m_2}"]),
        ("Parámetros adimensionales y casos límite",
         r"El único parámetro adimensional relevante es $\mu=m_2/m_1$, junto con $\theta$: la aceleración relativa a $g$ vale $(\mu-\sin\theta)/(1+\mu)$. El sistema está en equilibrio cuando $a=0$, es decir, cuando $\sin\theta=m_2/m_1$, y solo existe si $m_2\le m_1$. Con $\theta=\pi/2$ se recupera la máquina de Atwood y con $\theta=0$ el bloque horizontal arrastrado por un peso. Si $m_1\to\infty$, la aceleración tiende a $-g\sin\theta$ y la tensión a un valor finito.",
         [r"\dfrac{a}{g}=\dfrac{\mu-\sin\theta}{1+\mu},\qquad \theta_{eq}=\arcsin\dfrac{m_2}{m_1}",
          r"\theta=\dfrac{\pi}{2}:\ a=\dfrac{m_2-m_1}{m_1+m_2}\,g,\ T=\dfrac{2m_1m_2\,g}{m_1+m_2};\qquad \theta=0:\ a=\dfrac{m_2g}{m_1+m_2}"]),
        ("Resultado general",
         r"La aceleración es la diferencia entre el peso motriz y la componente resistente, repartida sobre la masa total, y la tensión es la media armónica de las masas multiplicada por $g(1+\sin\theta)$. Ambas expresiones son dimensionalmente correctas: la primera en $\mathrm{m/s^2}$ y la segunda en newton, ya que $m_1m_2/(m_1+m_2)$ tiene dimensión de masa. Son válidas mientras la cuerda esté tensa, es decir, mientras $T>0$, condición que siempre se cumple aquí.",
         [r"\boxed{\;a=\dfrac{m_2-m_1\sin\theta}{m_1+m_2}\,g,\qquad T=\dfrac{m_1m_2}{m_1+m_2}\,g\,(1+\sin\theta)\;}",
          r"[T]=\mathrm{kg}\cdot\mathrm{m/s^2}=\mathrm{N}"]),
        ("Particularización al problema",
         r"Se sustituyen las masas y el ángulo del enunciado en las fórmulas generales. El resultado reproduce la aceleración y la tensión halladas en la solución paso a paso, y el ángulo de equilibrio sirve de comprobación del caso límite.",
         [rf"m_1={sci(m1,3)}\ \mathrm{{kg}},\ m_2={sci(m2,3)}\ \mathrm{{kg}},\ \theta={fx(th,1)}^\circ,\ \sin\theta={sci(s,4)}",
          rf"a=\dfrac{{{sci(m2,3)}-({sci(m1,3)})({sci(s,4)})}}{{{sci(mm,3)}}}({sci(g,4)})={sci(a,5)}\ \mathrm{{m/s^2}}",
          rf"T=\dfrac{{({sci(m1,3)})({sci(m2,3)})}}{{{sci(mm,3)}}}({sci(g,4)})(1+{sci(s,4)})={sci(t,5)}\ \mathrm{{N}}",
          rf"\theta_{{eq}}=\arcsin\dfrac{{{sci(m2,3)}}}{{{sci(m1,3)}}}={fx(tb,3)}^\circ;\qquad \text{{Atwood }}(\theta=90^\circ):\ a={sci(atw,4)}\ \mathrm{{m/s^2}}"]),
        ("Errores comunes y límites de validez",
         r"Es habitual suponer que la tensión es igual al peso de la masa colgante, lo que solo es cierto cuando $a=0$. También se olvida que el signo de $a$ cambia con el ángulo y que bajo $\theta_{eq}$ el movimiento se invierte, de modo que la masa $m_2$ asciende. El modelo supone cuerda sin masa, polea ideal sin inercia y plano sin rozamiento; una polea con momento de inercia $I$ añade el término $I/R^2$ a la masa total y hace distintas las tensiones a cada lado. Si la cuerda se afloja, $T$ se anula y el sistema deja de ser de un solo grado de libertad.",
         []),
    ])


# =========================================================== 3. FrictionIncline
def _friction() -> dict:
    th, ms, mk, g = c6.THETA_DEG, c6.MU_S, c6.MU_K, c6.G
    tr = np.deg2rad(th)
    thc = c6.critical_angle_analytic()
    a = c6.incline_acceleration_analytic(th)
    sn, cs = np.sin(tr), np.cos(tr)
    return dict(concept=[
        ("Qué se estudia y por qué importa",
         r"Un bloque sobre un plano inclinado con rozamiento seco permite decidir si un objeto se queda quieto o desliza, y con qué aceleración. Interviene en el diseño de rampas, cintas transportadoras, frenos, neumáticos y en la estabilidad de laderas, donde el ángulo máximo estable de un talud coincide con el ángulo de rozamiento del material. La novedad respecto de los sistemas sin pérdidas es que la fuerza de rozamiento no es una función simple de la posición, sino una ley con dos regímenes, estático y cinético. El análisis general determina el ángulo crítico y la aceleración para coeficientes arbitrarios.",
         []),
        ("Magnitudes y convenciones",
         r"El ángulo $\theta$ del plano se mide desde la horizontal, y el eje $s$ se orienta hacia abajo del plano. El coeficiente de rozamiento estático $\mu_s$ y el cinético $\mu_k$ son adimensionales, con $\mu_k\le\mu_s$ y valores típicos entre $0{,}05$ y $1{,}5$. La normal $N$ y la fuerza de rozamiento $f$ se expresan en newton. Para madera sobre madera $\mu_s$ ronda $0{,}4$ y para goma sobre asfalto seco supera la unidad.",
         [r"[\mu_s]=[\mu_k]=1,\qquad [N]=[f]=\mathrm{N},\qquad f\le\mu_sN\ \text{(reposo)},\quad f=\mu_kN\ \text{(deslizamiento)}"]),
        ("Intuición y analogías",
         r"El rozamiento estático se comporta como un pestillo: se ajusta a la fuerza que lo solicita hasta un umbral y luego cede de golpe. Una vez que el bloque desliza, la resistencia cae al valor cinético, que es algo menor, por eso cuesta más iniciar un movimiento que mantenerlo. Cada ángulo del plano es una orden distinta para el pestillo: tangente grande significa mucha componente paralela por unidad de apriete, y a partir de cierto valor el pestillo ya no puede sostener.",
         []),
        ("Lagrangiano con ligadura unilateral y normal",
         r"Se toman coordenadas $(s,n)$ a lo largo y perpendicular al plano, y se impone la ligadura holónoma $n=0$ con un multiplicador $\lambda$ que representa la normal. La energía potencial gravitatoria incluye ambas componentes del peso. La ecuación de Euler-Lagrange para $n$ fija el valor del multiplicador y, con él, la normal, y la ligadura es unilateral porque requiere $N\ge0$ para que el bloque no se despegue. La ecuación para $s$ queda con la componente del peso paralela al plano como única fuerza conservativa.",
         [r"L=\tfrac12 m(\dot s^{2}+\dot n^{2})+mg\,s\sin\theta-mg\,n\cos\theta,\qquad \Phi=n=0",
          r"m\ddot n=-mg\cos\theta+\lambda\;\xrightarrow{\ \ddot n=0\ }\;N=\lambda=mg\cos\theta",
          r"m\ddot s=mg\sin\theta+Q_s"]),
        ("Rozamiento como fuerza generalizada disipativa",
         r"El rozamiento seco no deriva de un potencial y se introduce como fuerza generalizada no conservativa $Q_s$ en el miembro derecho de las ecuaciones de Lagrange. La ley de Coulomb tiene dos ramas: en reposo $f$ toma el valor que mantiene $\ddot s=0$ siempre que no supere $\mu_sN$, y en movimiento vale $\mu_kN$ en sentido contrario a $\dot s$. Esta estructura se expresa como una desigualdad en reposo y una igualdad en deslizamiento, y es la razón por la que el problema se resuelve por casos. El balance de energía muestra que el rozamiento siempre disipa, ya que $Q_s\dot s\le0$.",
         [r"\text{Reposo }(\dot s=0):\quad f=mg\sin\theta,\qquad |f|\le\mu_sN=\mu_s\,mg\cos\theta",
          r"\text{Deslizamiento }(\dot s>0):\quad Q_s=-\mu_kN=-\mu_k\,mg\cos\theta",
          r"\dfrac{d}{dt}\Big(\tfrac12 m\dot s^{2}-mg\,s\sin\theta\Big)=Q_s\dot s=-\mu_kmg\cos\theta\ \dot s\le0"]),
        ("Ángulo crítico y condición de deslizamiento",
         r"En reposo el rozamiento debe igualar la componente paralela del peso, de modo que el cociente $f/N=\tan\theta$ crece al inclinar el plano. El bloque se mantiene mientras ese cociente no exceda $\mu_s$, y el deslizamiento comienza exactamente cuando $\tan\theta=\mu_s$. La masa y la gravedad desaparecen de la condición, ya que tanto el peso motriz como el apriete son proporcionales a $mg$. Geométricamente, el cono de rozamiento, de semiángulo $\arctan\mu_s$, debe contener la dirección del peso.",
         [r"\dfrac{f}{N}=\dfrac{mg\sin\theta}{mg\cos\theta}=\tan\theta\le\mu_s",
          r"\boxed{\theta_c=\arctan\mu_s}"]),
        ("Dinámica del deslizamiento y casos límite",
         r"Para $\theta>\theta_c$ el movimiento obedece a una aceleración constante, de modo que la integración es elemental, y la velocidad y la posición crecen linealmente y cuadráticamente en el tiempo. La relación entre velocidad y recorrido sigue de la conservación del trabajo y la energía con pérdida, y la aceleración se anula en $\tan\theta=\mu_k$, lo que implica que un bloque lanzado con velocidad en un plano de ángulo $\arctan\mu_k$ conserva su rapidez. Si $\mu_k=0$ se recupera $g\sin\theta$, y si $\theta=90^\circ$, la caída libre.",
         [r"a=g(\sin\theta-\mu_k\cos\theta),\qquad v(t)=a\,t,\qquad s(t)=\tfrac12 a\,t^{2}",
          r"v^{2}=2a\,s,\qquad t=\sqrt{2s/a}",
          r"\theta=\arctan\mu_k\Rightarrow a=0;\quad \mu_k=0\Rightarrow a=g\sin\theta;\quad \theta=\tfrac{\pi}{2}\Rightarrow a=g"]),
        ("Resultado general",
         r"El resultado es una función por tramos: el bloque está en reposo con rozamiento estático $f=mg\sin\theta$ si $\tan\theta\le\mu_s$, y desliza con aceleración $g(\sin\theta-\mu_k\cos\theta)$ en caso contrario. La aceleración es independiente de la masa y tiene dimensión de $\mathrm{m/s^2}$; los coeficientes de rozamiento son adimensionales. Hay una discontinuidad en $\theta_c$, porque al cruzarlo la aceleración salta de $0$ a un valor positivo, consecuencia de que $\mu_k<\mu_s$.",
         [r"\boxed{\;a(\theta)=\begin{cases}0,&\tan\theta\le\mu_s\\ g(\sin\theta-\mu_k\cos\theta),&\tan\theta>\mu_s\end{cases}\qquad \theta_c=\arctan\mu_s\;}",
          r"[a]=\mathrm{m/s^2}"]),
        ("Particularización al problema",
         r"Con los coeficientes y el ángulo del enunciado se evalúa primero el ángulo crítico para saber en qué régimen se está, y luego la aceleración. El resultado coincide con el de la escena y con la integración numérica con máquina de estados.",
         [rf"\mu_s={fx(ms,2)},\quad \mu_k={fx(mk,2)},\quad \theta={fx(th,1)}^\circ,\quad \theta_c=\arctan({fx(ms,2)})={fx(thc,3)}^\circ",
          rf"\tan\theta={fx(np.tan(tr),4)}>\mu_s={fx(ms,2)}\ \Rightarrow\ \text{{desliza}}",
          rf"a={sci(g,4)}\,\big({sci(sn,4)}-{fx(mk,2)}\cdot{sci(cs,4)}\big)={sci(a,5)}\ \mathrm{{m/s^2}}"]),
        ("Errores comunes y límites de validez",
         r"Se confunde con frecuencia la fuerza de rozamiento con $\mu_sN$: esa cantidad es solo el máximo, mientras que el valor real en reposo es $mg\sin\theta$. Otro error es usar $\mu_s$ para calcular la aceleración, cuando durante el deslizamiento corresponde $\mu_k$. También se cree que un bloque más pesado resbala antes; la masa se cancela. La ley de Coulomb es empírica, y deja de valer con velocidades muy altas, superficies lubricadas o con adhesión, y en el efecto de agarrar y soltar (stick-slip), donde $\mu$ depende de la velocidad.",
         []),
    ])


# =========================================================== 4. DragFall
def _drag() -> dict:
    m, b, g = c6.MASS, c6.DRAG_B, c6.G
    vt, ta = c6.terminal_speed(), c6.tau()
    t95 = c6.time_to_fraction_analytic()
    fr = 1 - np.exp(-1.0)
    x2 = float(c6.drag_position_analytic(2.0))
    return dict(concept=[
        ("Qué se estudia y por qué importa",
         r"La caída con resistencia del medio explica por qué una gota de lluvia no llega al suelo a cientos de kilómetros por hora, cómo funciona un paracaídas y por qué una esfera en un fluido viscoso sedimenta con velocidad constante. La velocidad límite es la clave de la sedimentación, de los viscosímetros de bola y del frenado de partículas en un medio. El objetivo es resolver la ecuación de movimiento con una resistencia proporcional a la velocidad para una masa y un coeficiente arbitrarios. De esa solución general se obtienen el tiempo característico y la velocidad terminal.",
         []),
        ("Magnitudes y convenciones",
         r"La coordenada $x$ se mide hacia abajo desde el punto de partida, de modo que la gravedad y la velocidad inicial positiva coinciden. El coeficiente $b$, en $\mathrm{kg/s}$, es la constante de la ley lineal de resistencia $F_d=-b\,v$, válida a números de Reynolds bajos; en la ley de Stokes $b=6\pi\eta R$. Aparecen dos escalas: la velocidad terminal $v_t$ en $\mathrm{m/s}$ y el tiempo característico $\tau$ en segundos. Una gota de aceite en aire tiene $\tau$ de milisegundos, y un cuerpo en miel, de segundos.",
         [r"[b]=\mathrm{kg/s},\qquad [v_t]=\mathrm{m/s},\qquad [\tau]=\mathrm{s},\qquad F_d=-b\,\dot x"]),
        ("Intuición y analogías",
         r"La resistencia crece con la velocidad, de modo que el cuerpo acelera mientras el peso supera a la resistencia y deja de hacerlo cuando ambas se igualan. El proceso es análogo a la carga de un condensador a través de una resistencia: el sistema se aproxima a un valor final siguiendo una exponencial decreciente. La velocidad terminal es el análogo de la tensión de la fuente y el tiempo $\tau$, de la constante $RC$.",
         []),
        ("Lagrangiano con función de disipación de Rayleigh",
         r"El rozamiento viscoso no deriva de un potencial dependiente solo de las coordenadas, pero sí de una función de disipación de Rayleigh $\mathcal F=\tfrac12 b\dot x^2$ cuya derivada respecto de la velocidad da la fuerza de resistencia. Las ecuaciones de Lagrange adquieren entonces el término disipativo en el miembro derecho. El lagrangiano contiene la energía cinética y el potencial gravitatorio con $x$ hacia abajo. Esta formulación generaliza de manera directa a muchos grados de libertad con matriz de amortiguamiento.",
         [r"L=\tfrac12 m\dot x^{2}+mg\,x,\qquad \mathcal F=\tfrac12 b\,\dot x^{2}",
          r"\dfrac{d}{dt}\dfrac{\partial L}{\partial\dot x}-\dfrac{\partial L}{\partial x}=-\dfrac{\partial\mathcal F}{\partial\dot x}",
          r"m\ddot x=mg-b\dot x"]),
        ("Ruptura de la simetría temporal y balance de energía",
         r"El lagrangiano no depende de $t$ y, sin disipación, el teorema de Noether daría la conservación de la energía. Con la función de Rayleigh la energía deja de ser una integral primera: su pérdida por unidad de tiempo es $2\mathcal F=b\dot x^2$, la potencia disipada. La energía no desaparece, sino que pasa al medio como calor. En régimen terminal, la potencia del peso, $mg\,v_t$, se emplea íntegramente en esa disipación.",
         [r"E=\tfrac12 m\dot x^{2}-mg\,x,\qquad \dfrac{dE}{dt}=-b\dot x^{2}",
          r"\text{Régimen terminal:}\quad mg\,v_t=b\,v_t^{2}\;\Longrightarrow\;v_t=\dfrac{mg}{b}"]),
        ("Integración exacta de la ecuación de movimiento",
         r"La ecuación diferencial es lineal de primer orden en $v$ con coeficientes constantes, por lo que se resuelve separando variables o con factor integrante. Al definir $\tau=m/b$ y $v_t=mg/b$ queda $\dot v=(v_t-v)/\tau$, de modo que la velocidad se aproxima exponencialmente a $v_t$ con constante de tiempo $\tau$. Integrando una vez más se obtiene la posición, y eliminando el tiempo se obtiene la relación $x(v)$. Las condiciones iniciales de partida del reposo fijan las constantes.",
         [r"\dot v=\dfrac{v_t-v}{\tau}\;\Longrightarrow\;\int_0^{v}\dfrac{dv'}{v_t-v'}=\int_0^{t}\dfrac{dt'}{\tau}",
          r"v(t)=v_t\left(1-e^{-t/\tau}\right),\qquad x(t)=v_t\left[t-\tau\left(1-e^{-t/\tau}\right)\right]",
          r"x(v)=-\,v_t\tau\left[\dfrac{v}{v_t}+\ln\!\left(1-\dfrac{v}{v_t}\right)\right]"]),
        ("Variables adimensionales y casos límite",
         r"Con $u=v/v_t$ y $s=t/\tau$ la ecuación se reduce a $du/ds=1-u$, que no contiene ningún parámetro, lo que revela que toda caída lineal es la misma curva reescalada. Para $t\ll\tau$ se recupera la caída libre, $x\simeq gt^2/2$, y para $t\gg\tau$ el movimiento es uniforme, $x\simeq v_t(t-\tau)$. El tiempo para alcanzar una fracción $f$ de la velocidad terminal es independiente de la masa si se mide en unidades de $\tau$. En el límite $b\to0$, $\tau\to\infty$ y $v_t\to\infty$, es decir, caída libre.",
         [r"\dfrac{du}{ds}=1-u,\qquad u=1-e^{-s}",
          r"t\ll\tau:\ v\simeq g\,t,\ x\simeq\tfrac12 g\,t^{2};\qquad t\gg\tau:\ v\to v_t,\ x\simeq v_t(t-\tau)",
          r"t_f=-\tau\ln(1-f)"]),
        ("Resultado general",
         r"La velocidad es una exponencial creciente saturada en $v_t=mg/b$ con escala $\tau=m/b$, y la posición es su integral. Dimensionalmente, $mg/b$ tiene unidades $\mathrm{kg\cdot m/s^2}/(\mathrm{kg/s})=\mathrm{m/s}$ y $m/b$ unidades de segundo. La fórmula contiene el caso sin resistencia como límite y es válida mientras la resistencia sea lineal en la velocidad.",
         [r"\boxed{\;v(t)=\dfrac{mg}{b}\left(1-e^{-bt/m}\right),\qquad x(t)=\dfrac{mg}{b}\left[t-\dfrac{m}{b}\left(1-e^{-bt/m}\right)\right]\;}",
          r"v_t=\dfrac{mg}{b},\qquad \tau=\dfrac{m}{b}"]),
        ("Particularización al problema",
         r"Con la masa y el coeficiente del enunciado se calculan la velocidad terminal, el tiempo característico y el tiempo en que se alcanza el $95\%$ de la velocidad límite. Las cifras coinciden con las de la escena y con la integración numérica por Runge-Kutta.",
         [rf"m={fx(m,2)}\ \mathrm{{kg}},\quad b={fx(b,2)}\ \mathrm{{kg/s}},\quad v_t=\dfrac{{({fx(m,2)})({sci(g,4)})}}{{{fx(b,2)}}}={sci(vt,5)}\ \mathrm{{m/s}}",
          rf"\tau=\dfrac{{{fx(m,2)}}}{{{fx(b,2)}}}={sci(ta,4)}\ \mathrm{{s}},\qquad t_{{95}}=-\tau\ln0{{,}}05={sci(t95,5)}\ \mathrm{{s}}",
          rf"v(\tau)={sci(vt,5)}\,(1-e^{{-1}})={sci(vt*fr,5)}\ \mathrm{{m/s}},\qquad x(2\ \mathrm{{s}})={sci(x2,5)}\ \mathrm{{m}}"]),
        ("Errores comunes y límites de validez",
         r"Un error habitual es pensar que el cuerpo alcanza la velocidad terminal en un tiempo finito: la aproximación es exponencial y solo se llega asintóticamente. También se confunde $\tau$ con el tiempo de llegada a $v_t$, cuando en $t=\tau$ la velocidad es el $63\%$ del valor límite. La ley lineal solo vale con Reynolds menor que la unidad; a velocidades mayores la resistencia cuadrática $F_d=-\tfrac12 C_D\rho A v^2$ domina, la velocidad terminal pasa a ser $\sqrt{2mg/(C_D\rho A)}$ y la solución pasa a ser una tangente hiperbólica.",
         []),
    ])


# =========================================================== 5. SpringWork
def _springwork() -> dict:
    k, x1, x2 = c7.K, c7.X1, c7.X2
    w = c7.spring_work_analytic()
    wn = c7.spring_work_numeric(2, "trapezoid")
    return dict(concept=[
        ("Qué se estudia y por qué importa",
         r"El resorte es el modelo universal de cualquier fuerza restauradora en el entorno de un equilibrio: amortiguadores, balanzas, enlaces moleculares, cuerdas de instrumentos y estructuras elásticas. Calcular el trabajo necesario para deformarlo ilustra el caso en que la fuerza varía con la posición y la integral reemplaza al producto de fuerza por distancia. La energía así almacenada es la energía potencial elástica, que reaparece en el oscilador armónico y en la cuantización de las vibraciones. El análisis general resuelve el trabajo para cualquier ley de fuerza y destaca el papel del potencial.",
         []),
        ("Magnitudes y convenciones",
         r"La constante $k$, en $\mathrm{N/m}$, mide la rigidez, y $x$ es la elongación respecto de la longitud natural, positiva al estirar. La fuerza que debe aplicar un agente externo para sostener una deformación cuasiestática es $F=kx$, y la que ejerce el resorte sobre quien lo deforma es $-kx$ (ley de Hooke). El trabajo se expresa en julios y es positivo cuando el agente externo cede energía al resorte. Valores típicos de $k$ van de unos $10\ \mathrm{N/m}$ en un resorte de juguete a $10^5\ \mathrm{N/m}$ en una suspensión de automóvil.",
         [r"[k]=\mathrm{N/m},\qquad [W]=\mathrm{N\cdot m}=\mathrm{J},\qquad F_{\rm ext}=kx,\qquad F_{\rm res}=-kx"]),
        ("Intuición y analogías",
         r"Estirar un resorte es cada vez más costoso, porque la fuerza crece con la elongación; por eso el trabajo no es la fuerza final por la distancia, sino el área bajo la recta fuerza-posición. La analogía es la de llenar un depósito cónico: la energía almacenada depende del cuadrado de la deformación, de modo que duplicar la elongación cuadruplica la energía. Ese carácter cuadrático explica que un pequeño exceso de estiramiento cueste mucho más que el primer tramo.",
         []),
        ("Trabajo de una fuerza que depende de la posición",
         r"Para una fuerza unidimensional $F(x)$ el trabajo de $x_1$ a $x_2$ se define como el límite de las sumas de Riemann de $F\,dx$ sobre una partición cada vez más fina. Es la única definición coherente porque en cada tramo infinitesimal la fuerza puede considerarse constante. El resultado es una integral definida, y el trabajo es una cantidad que depende de los extremos, no del tiempo empleado. En una dimensión toda fuerza que depende solo de la posición es conservativa, porque siempre admite una primitiva.",
         [r"W_{1\to2}=\lim_{n\to\infty}\sum_{i}F(x_i)\,\Delta x_i=\int_{x_1}^{x_2}F(x)\,dx",
          r"U(x)=-\int_{x_0}^{x}F_{\rm res}(x')\,dx',\qquad W_{\rm ext}=\Delta U"]),
        ("Lagrangiano y potencial elástico",
         r"La ley de Hooke deriva del potencial cuadrático $U=\tfrac12kx^2$, que es el primer término no trivial del desarrollo de Taylor de cualquier potencial en torno a un mínimo estable. El lagrangiano $L=T-U$ no depende explícitamente del tiempo, de modo que el teorema de Noether implica la conservación de la energía. La ecuación de Euler-Lagrange da el movimiento armónico, y el hamiltoniano es la suma de cinética y potencial. El trabajo cuasiestático del agente externo es, en consecuencia, la variación de $U$.",
         [r"L=\tfrac12 m\dot x^{2}-\tfrac12 k x^{2},\qquad m\ddot x=-kx,\qquad \omega=\sqrt{k/m}",
          r"H=\dfrac{p^{2}}{2m}+\tfrac12kx^{2}=E,\qquad \dfrac{dH}{dt}=0",
          r"U(x)=\tfrac12 k x^{2}+\text{cte},\qquad U''(0)=k"]),
        ("Cálculo del trabajo cuasiestático",
         r"El agente externo estira el resorte tan despacio que la energía cinética es despreciable y su fuerza equilibra en cada instante la del resorte, $F_{\rm ext}=kx$. En esas condiciones el trabajo externo es la integral de $kx$, y como la primitiva es elemental se evalúa en cierre. El resultado es la diferencia de $U$ entre los estados final e inicial, y no depende de cómo se lleve el resorte de uno al otro. El trabajo del resorte es el opuesto, $-\Delta U$.",
         [r"W_{\rm ext}=\int_{x_1}^{x_2}kx\,dx=\tfrac12k\left(x_2^{2}-x_1^{2}\right)=U(x_2)-U(x_1)",
          r"W_{\rm res}=-\int_{x_1}^{x_2}kx\,dx=-\Delta U"]),
        ("Generalización, casos límite y regla numérica",
         r"Si la ley fuese $F=\kappa x^n$ el trabajo valdría $\kappa(x_2^{n+1}-x_1^{n+1})/(n+1)$ y $n=1$ lo reduce al caso lineal. Partiendo del reposo $x_1=0$, la energía es $\tfrac12kx_2^2$, y el trabajo entre dos elongaciones no es proporcional a su diferencia. Para la integración numérica, la regla del trapecio es exacta con integrandos lineales y la de Simpson lo es hasta grado tres; por eso el error de ambas en este problema es de redondeo.",
         [r"F=\kappa x^{n}\;\Rightarrow\;W=\dfrac{\kappa}{n+1}\left(x_2^{\,n+1}-x_1^{\,n+1}\right)",
          r"x_1=0:\ W=\tfrac12kx_2^{2};\qquad x_2\to x_1:\ W\simeq kx_1\,\Delta x",
          r"W_{\rm trap}=\dfrac{\Delta x}{2}\big[F(x_1)+F(x_2)\big]=\dfrac{k}{2}(x_2-x_1)(x_1+x_2)=W"]),
        ("Resultado general",
         r"El trabajo para deformar un resorte lineal entre dos elongaciones es el producto de la mitad de la constante por la diferencia de cuadrados. Es dimensionalmente $\mathrm{N/m\cdot m^2}=\mathrm{J}$, es simétrico respecto del signo de $x$ y depende solo de los estados extremos. Contiene el caso desde el reposo como $x_1=0$.",
         [r"\boxed{\;W_{1\to2}=\tfrac12\,k\left(x_2^{2}-x_1^{2}\right)\;}",
          r"[W]=\mathrm{N\,m^{-1}\cdot m^{2}}=\mathrm{J}"]),
        ("Particularización al problema",
         r"Con la constante y las elongaciones del enunciado, el resultado coincide con la integral analítica de la escena. La regla del trapecio con solo dos subintervalos reproduce el mismo valor, porque el integrando es lineal.",
         [rf"k={sci(k,4)}\ \mathrm{{N/m}},\quad x_1={fx(x1,2)}\ \mathrm{{m}},\quad x_2={fx(x2,2)}\ \mathrm{{m}}",
          rf"W=\tfrac12({sci(k,4)})\big(({fx(x2,2)})^2-({fx(x1,2)})^2\big)={sci(w,5)}\ \mathrm{{J}}",
          rf"W_{{\rm trap}}(n=2)={sci(wn,5)}\ \mathrm{{J}};\qquad \text{{desde el reposo }}(x_1=0):\ W={sci(0.5*k*x2**2,4)}\ \mathrm{{J}}"]),
        ("Errores comunes y límites de validez",
         r"El error más difundido es calcular $W=F\,\Delta x$ con la fuerza final o la media aritmética de las elongaciones, que no coincide con el área bajo la recta. También se confunde el trabajo del agente con el del resorte, que tienen signos opuestos, y se olvida que la energía depende de la diferencia de cuadrados, no del cuadrado de la diferencia. La ley de Hooke solo vale para deformaciones pequeñas frente a la longitud natural; más allá del límite elástico el material se deforma de modo permanente y la fuerza ya no es función univaluada de $x$.",
         []),
    ])


# =========================================================== 6. WorkEnergyTheorem
def _workenergy() -> dict:
    m, f0, L = c7.MASS, c7.F0, c7.LENGTH
    kf = float(c7.kinetic_energy_analytic(L))
    v = c7.final_speed_analytic()
    om = np.sqrt(f0 / (m * L))
    tf = np.pi / (2 * om)
    pmax = 0.5 * f0 * L * om
    return dict(concept=[
        ("Qué se estudia y por qué importa",
         r"El teorema trabajo-energía permite obtener la rapidez de un cuerpo sin resolver su trayectoria en el tiempo, lo que es decisivo cuando la fuerza depende de la posición: motores con curva de par, catapultas, frenos magnéticos o colisiones con superficies blandas. Resolver la segunda ley paso a paso es posible, pero el teorema entrega el resultado con una sola integral. Este análisis lo deduce del lagrangiano y de la simetría temporal, y lo aplica a una fuerza que decrece linealmente con la posición. Además da la potencia instantánea, necesaria para dimensionar el motor.",
         []),
        ("Magnitudes y convenciones",
         r"La energía cinética $K=\tfrac12mv^2$ se mide en julios, el trabajo $W=\int F\,dx$ en julios y la potencia $P=Fv$ en vatios. En este modelo la fuerza vale $F_0$ al partir y decrece linealmente hasta anularse a la distancia $L$, de modo que $F_0$ está en newton y $L$ en metro. La partícula parte del reposo en $x=0$. El teorema se aplica a la fuerza neta, sumando todas las fuerzas que actúan.",
         [r"F(x)=F_0\left(1-\dfrac{x}{L}\right),\qquad [K]=[W]=\mathrm{J},\qquad [P]=\mathrm{W}"]),
        ("Intuición y analogías",
         r"El cuerpo recibe energía mientras la fuerza empuja en el sentido del movimiento, y el aporte por unidad de recorrido es precisamente el valor de la fuerza. Es como llenar un tanque cuyo caudal disminuye con el nivel: la cantidad acumulada es el área bajo la curva. Como la fuerza se anula en $x=L$, la energía cinética alcanza ahí su máximo y la potencia instantánea es nula, pues el cuerpo ya no es empujado.",
         []),
        ("De la segunda ley al teorema por multiplicación por la velocidad",
         r"Partiendo de la segunda ley unidimensional, se multiplica ambos miembros por la velocidad. El miembro izquierdo se convierte en la derivada temporal de la energía cinética por la regla de la cadena, y el derecho es la potencia de la fuerza. La integración respecto del tiempo, con el cambio de variable $dx=v\,dt$, lleva a la integral de la fuerza sobre el camino. Esta deducción no supone fuerza constante; solo exige que $F$ sea integrable a lo largo de la trayectoria.",
         [r"m\ddot x=F(x)\;\xrightarrow{\ \times\dot x\ }\;\dfrac{d}{dt}\Big(\tfrac12m\dot x^{2}\Big)=F\,\dot x",
          r"\int_{0}^{t}\dfrac{dK}{dt'}dt'=\int_{0}^{t}F\,\dot x\,dt'\;\Longrightarrow\;K(x)-K(0)=\int_{0}^{x}F(x')\,dx'"]),
        ("Formulación lagrangiana y simetría temporal",
         r"En una dimensión toda fuerza que depende solo de la posición deriva de un potencial $U(x)=-\int F\,dx$, por lo que existe un lagrangiano $L=\tfrac12m\dot x^2-U$. Como $L$ no depende explícitamente del tiempo, el teorema de Noether para traslaciones temporales garantiza que $E=K+U$ es una integral primera. El teorema trabajo-energía es entonces la expresión de la homogeneidad del tiempo: si las leyes no cambian con el instante, la energía se conserva. El hamiltoniano $H=p^2/2m+U$ coincide con $E$.",
         [r"U(x)=-F_0\left(x-\dfrac{x^{2}}{2L}\right),\qquad L_{\rm lag}=\tfrac12m\dot x^{2}-U(x)",
          r"E=\tfrac12 m\dot x^{2}+U(x)=\text{cte}=H,\qquad \dot x=\dfrac{\partial H}{\partial p},\ \dot p=F",
          r"\tfrac12mv^{2}=E-U(x)=F_0\left(x-\dfrac{x^{2}}{2L}\right)"]),
        ("Solución exacta del movimiento",
         r"La ecuación de movimiento es lineal con una fuerza restauradora hacia $x=L$, de modo que equivale a un oscilador armónico desplazado de frecuencia $\omega=\sqrt{F_0/(mL)}$. Con partida del reposo en el origen, la solución es un coseno desplazado, y la partícula alcanza $x=L$ en un cuarto de período. Derivando se obtiene la velocidad, y verificando que $K(x)$ obtenida por el teorema coincide con $\tfrac12m\dot x^2$, se comprueba la consistencia. Esta solución exacta sirve para validar el integrador numérico.",
         [r"\ddot x=\dfrac{F_0}{m}\left(1-\dfrac xL\right)\;\Rightarrow\;x(t)=L\left(1-\cos\omega t\right),\quad \omega=\sqrt{\dfrac{F_0}{mL}}",
          r"\dot x=L\omega\sin\omega t,\qquad x=L\ \text{en}\ t_L=\dfrac{\pi}{2\omega}",
          r"P(t)=F\dot x=F_0L\omega\sin\omega t\cos\omega t=\tfrac12F_0L\omega\sin2\omega t"]),
        ("Casos límite y escalas",
         r"El trabajo total hasta $x=L$ es el área del triángulo de base $L$ y altura $F_0$, es decir, $\tfrac12F_0L$, y la rapidez final $\sqrt{F_0L/m}$ define la escala natural de velocidades. La potencia máxima se alcanza a mitad de camino en términos de ángulo, para $\omega t=\pi/4$, y vale $\tfrac12F_0L\omega$. Si la fuerza fuese constante, $F=F_0$, la energía final sería $F_0L$, el doble. Si $F_0$ o $L$ tienden a cero, la rapidez final tiende a cero.",
         [r"W_{0\to L}=\int_0^{L}F_0\left(1-\dfrac xL\right)dx=\tfrac12F_0L",
          r"v_f=\sqrt{\dfrac{2W}{m}}=\sqrt{\dfrac{F_0L}{m}},\qquad P_{\max}=\tfrac12F_0L\,\omega\ \ (\omega t=\pi/4)",
          r"F=F_0\ \text{(constante)}:\quad v_f=\sqrt{2F_0L/m}"]),
        ("Resultado general",
         r"La energía cinética en cualquier punto es la integral de la fuerza hasta ese punto, y la rapidez es su raíz. Las fórmulas son dimensionalmente consistentes: $F_0L$ tiene dimensión de energía y $\sqrt{F_0L/m}$ de velocidad. Son válidas para $0\le x\le L$, donde la fuerza es positiva.",
         [r"\boxed{\;K(x)=F_0\left(x-\dfrac{x^{2}}{2L}\right),\qquad v(x)=\sqrt{\dfrac{2F_0}{m}\left(x-\dfrac{x^{2}}{2L}\right)},\qquad v_f=\sqrt{\dfrac{F_0L}{m}}\;}",
          r"[v_f]=\sqrt{\mathrm{N\,m/kg}}=\mathrm{m/s}"]),
        ("Particularización al problema",
         r"Con la masa, la fuerza inicial y la longitud del enunciado se obtiene la energía cinética final y la rapidez, que coinciden con las de la escena y con la integración de la segunda ley. La solución exacta aporta además el tiempo de recorrido y la potencia máxima.",
         [rf"m={fx(m,1)}\ \mathrm{{kg}},\quad F_0={fx(f0,1)}\ \mathrm{{N}},\quad L={fx(L,1)}\ \mathrm{{m}}",
          rf"K(L)=F_0\dfrac{{L}}{{2}}={sci(kf,4)}\ \mathrm{{J}},\qquad v_f=\sqrt{{\dfrac{{({fx(f0,1)})({fx(L,1)})}}{{{fx(m,1)}}}}}={sci(v,5)}\ \mathrm{{m/s}}",
          rf"\omega={sci(om,5)}\ \mathrm{{s^{{-1}}}},\quad t_L=\dfrac{{\pi}}{{2\omega}}={sci(tf,4)}\ \mathrm{{s}},\quad P_{{\max}}={sci(pmax,4)}\ \mathrm{{W}},\quad P(L)=0"]),
        ("Errores comunes y límites de validez",
         r"Se suele multiplicar la fuerza inicial por la distancia, ignorando que la fuerza decrece; el trabajo correcto es el área bajo la curva. También se supone que la rapidez es máxima cuando la fuerza es máxima, cuando en realidad lo es cuando la fuerza se anula. El teorema vale solo para la fuerza neta y para una partícula puntual; con un cuerpo extenso, el trabajo de las fuerzas internas puede cambiar la energía interna. Si la fuerza dependiese del tiempo o de la velocidad no habría potencial, y la energía mecánica no se conservaría, aunque el teorema trabajo-energía seguiría valiendo.",
         []),
    ])


# =========================================================== 7. PendulumEnergy
def _pendulum() -> dict:
    L, g, t0d, m = c8.LENGTH, c8.G, c8.THETA0_DEG, c8.MASS
    t0 = np.deg2rad(t0d)
    vb = c8.bottom_speed_analytic()
    tens = m * g * (3 - 2 * np.cos(t0))
    w0 = np.sqrt(g / L)
    tp0 = 2 * np.pi / w0
    a_, b_ = 1.0, np.cos(t0 / 2)
    for _ in range(30):
        a_, b_ = (a_ + b_) / 2, np.sqrt(a_ * b_)
    tper = tp0 / a_
    return dict(concept=[
        ("Qué se estudia y por qué importa",
         r"El péndulo es el ejemplo clásico de oscilación y de conservación de la energía: relojes, columpios, sismómetros, grúas con carga suspendida y, por analogía, la dinámica de una molécula alrededor de un enlace. Resolver su movimiento exacto exige integrales elípticas, pero la rapidez en cada ángulo se obtiene de un balance de energía elemental. Este análisis parte del lagrangiano para amplitud arbitraria, deduce la integral primera de energía y de ella la rapidez, la tensión y el período, mostrando dónde falla la aproximación de pequeñas oscilaciones.",
         []),
        ("Magnitudes y convenciones",
         r"El ángulo $\theta$ se mide desde la vertical hacia abajo, la longitud $\ell$ de la varilla o hilo en metros y la masa $m$ en kilogramos. La energía potencial se toma nula en el punto más bajo, $U=mg\ell(1-\cos\theta)$, y la energía cinética es $\tfrac12m\ell^2\dot\theta^2$. La rapidez lineal es $v=\ell\dot\theta$. Un péndulo de un metro tiene un período cercano a dos segundos.",
         [r"[\theta]=\mathrm{rad},\qquad [\omega]=[\dot\theta]=\mathrm{rad/s},\qquad U(\theta)=mg\ell\,(1-\cos\theta)"]),
        ("Intuición y analogías",
         r"La masa cambia energía potencial por cinética al descender y viceversa al subir, sin pérdida en el caso ideal. Es como una bola en un valle: la altura inicial determina cuánta velocidad tendrá en el fondo y hasta dónde sube al otro lado. El movimiento es periódico porque la energía total no alcanza para salir del valle, y el período depende de la amplitud, a diferencia del oscilador armónico.",
         []),
        ("Lagrangiano y ecuación de movimiento",
         r"El péndulo tiene un grado de libertad, $\theta$, porque la longitud fija es una ligadura holónoma que se elimina al usar coordenadas polares. El lagrangiano es la energía cinética menos la potencial, con el origen de potencial arbitrario. La ecuación de Euler-Lagrange es la ecuación del péndulo no lineal, equivalente a la segunda ley en la dirección tangencial. Para pequeñas amplitudes $\sin\theta\simeq\theta$ y se recupera el oscilador armónico.",
         [r"L=\tfrac12m\ell^{2}\dot\theta^{2}+mg\ell\cos\theta",
          r"\dfrac{d}{dt}\left(m\ell^{2}\dot\theta\right)+mg\ell\sin\theta=0\;\Longrightarrow\;\ddot\theta=-\dfrac{g}{\ell}\sin\theta",
          r"\theta\ll1:\quad \ddot\theta\simeq-\omega_0^{2}\theta,\qquad \omega_0=\sqrt{g/\ell}"]),
        ("Hamiltoniano, teorema de Noether e integral de energía",
         r"El lagrangiano no depende explícitamente del tiempo, de modo que el teorema de Noether asocia a la traslación temporal la conservación de la energía, que es el hamiltoniano. El momento canónico es el momento angular respecto del pivote, y no se conserva porque el peso ejerce par. La ligadura no trabaja porque la tensión es perpendicular al desplazamiento, y esa es la razón de que solo cuente el peso. Con la energía conservada se obtiene una ecuación de primer orden en $\dot\theta$.",
         [r"p_\theta=m\ell^{2}\dot\theta,\qquad H=\dfrac{p_\theta^{2}}{2m\ell^{2}}-mg\ell\cos\theta",
          r"\dot\theta=\dfrac{\partial H}{\partial p_\theta},\qquad \dot p_\theta=-\dfrac{\partial H}{\partial\theta}=-mg\ell\sin\theta,\qquad \dfrac{dH}{dt}=0",
          r"\dot\theta^{2}=\dfrac{2g}{\ell}\left(\cos\theta-\cos\theta_0\right)"]),
        ("Rapidez y tensión en función del ángulo",
         r"Como la energía total es la potencial en el punto de reposo inicial, la rapidez en cualquier ángulo sale de igualar $K+U$ con $U(\theta_0)$. La rapidez es máxima donde $U$ es mínima, es decir, en $\theta=0$. La tensión se obtiene de la ecuación radial, que tiene que proporcionar la fuerza centrípeta además de equilibrar la componente radial del peso. En el punto más bajo supera al peso incluso para amplitud pequeña, porque la masa describe un arco.",
         [r"v(\theta)=\ell\dot\theta=\sqrt{2g\ell\,(\cos\theta-\cos\theta_0)},\qquad v_{\max}=\sqrt{2g\ell\,(1-\cos\theta_0)}=2\sqrt{g\ell}\,\sin\dfrac{\theta_0}{2}",
          r"T(\theta)-mg\cos\theta=\dfrac{mv^{2}}{\ell}\;\Longrightarrow\;T(\theta)=mg\,(3\cos\theta-2\cos\theta_0)",
          r"T(0)=mg\,(3-2\cos\theta_0)"]),
        ("Período exacto y retrato de fase",
         r"Integrando la ecuación de primer orden entre $0$ y $\theta_0$ se obtiene el período como una integral elíptica completa de primera especie, con módulo $k=\sin(\theta_0/2)$. Se evalúa de forma eficiente con la media aritmético-geométrica. En el plano de fases $(\theta,p_\theta)$ las órbitas son curvas de nivel de $H$: cerradas si $E<mg\ell$ y la separatriz en $E=mg\ell$, que corresponde a un ascenso hasta la vertical. Más energía da rotación completa. Para amplitudes pequeñas el período tiende a $2\pi/\omega_0$.",
         [r"T=4\sqrt{\dfrac{\ell}{g}}\,K(k)=\dfrac{2\pi}{\omega_0}\,\dfrac{1}{\mathrm{AGM}\!\left(1,\cos\frac{\theta_0}{2}\right)},\qquad k=\sin\dfrac{\theta_0}{2}",
          r"T\simeq\dfrac{2\pi}{\omega_0}\left(1+\dfrac{\theta_0^{2}}{16}+\cdots\right)",
          r"E_{\rm sep}=mg\ell\ \ (\theta_0=\pi,\ T\to\infty)"]),
        ("Resultado general",
         r"La rapidez en el fondo solo depende de $g$, $\ell$ y la amplitud, y es independiente de la masa. Dimensionalmente $g\ell$ tiene unidades de $\mathrm{m^2/s^2}$. La tensión sí es proporcional a la masa, porque es una fuerza. El caso general contiene la caída libre desde la altura $h=\ell(1-\cos\theta_0)$, pues $v=\sqrt{2gh}$.",
         [r"\boxed{\;v_{\max}=\sqrt{2g\ell\,(1-\cos\theta_0)},\qquad T_{\max}=mg\,(3-2\cos\theta_0)\;}",
          r"[v]=\sqrt{\mathrm{m/s^{2}\cdot m}}=\mathrm{m/s}"]),
        ("Particularización al problema",
         r"Con la longitud y la amplitud inicial del enunciado se obtiene la rapidez en el fondo, que coincide con la de la escena y con la energía total constante en la integración. La misma fórmula general da la tensión y el período exacto, que difiere del lineal.",
         [rf"\ell={fx(L,1)}\ \mathrm{{m}},\quad \theta_0={fx(t0d,0)}^\circ,\quad 1-\cos\theta_0={fx(1-np.cos(t0),3)}",
          rf"v_{{\max}}=\sqrt{{2({sci(g,4)})({fx(L,1)})({fx(1-np.cos(t0),3)})}}={sci(vb,5)}\ \mathrm{{m/s}}",
          rf"T_{{\max}}=({fx(m,1)})({sci(g,4)})(3-2\cos{fx(t0d,0)}^\circ)={sci(tens,5)}\ \mathrm{{N}}",
          rf"T_{{\rm per}}={sci(tper,5)}\ \mathrm{{s}}\quad(\text{{lineal: }}{sci(tp0,5)}\ \mathrm{{s}})"]),
        ("Errores comunes y límites de validez",
         r"Se supone a menudo que el período es independiente de la amplitud, cuando solo lo es en la aproximación lineal, y a $60^\circ$ la diferencia es de alrededor del 7%. También se afirma que la tensión iguala al peso en el fondo, lo que desprecia el término centrípeto. El modelo supone varilla o hilo sin masa, pivote sin rozamiento y ausencia de resistencia del aire; con disipación la energía decae y el péndulo termina en reposo. Un péndulo físico cuerpo extenso sustituye $m\ell^2$ por el momento de inercia $I$.",
         []),
    ])


# =========================================================== 8. SpringFriction
def _springfric() -> dict:
    k, m, mu, g, x0 = c8.KS, c8.M, c8.MU, c8.G, c8.X0
    d = c8.delta()
    pts = c8.turning_points_analytic()
    n_half = len(pts) - 1
    xn = pts[-1]
    th_ = np.pi * np.sqrt(m / k)
    ttot = n_half * th_
    ediss = 0.5 * k * (x0**2 - xn**2)
    path = x0 + 2 * sum(abs(p) for p in pts[1:-1]) + abs(xn)
    return dict(concept=[
        ("Qué se estudia y por qué importa",
         r"Un oscilador con rozamiento seco muestra cómo se pierde energía en sistemas reales: relojes de cuerda, suspensiones con fricción, estructuras que disipan en las juntas y el sismo bajo una cimentación. A diferencia del amortiguamiento viscoso, aquí el movimiento se detiene en un tiempo finito y a una distancia del equilibrio distinta de cero. El interés es entender que la fuerza no conservativa se trata con el balance de energía y con ecuaciones lineales por tramos. El análisis general entrega la ley de decaimiento de las amplitudes y el punto de parada.",
         []),
        ("Magnitudes y convenciones",
         r"La elongación $x$ se mide desde la posición natural del resorte, con constante $k$ en $\mathrm{N/m}$ y masa $m$ en kg. El coeficiente de rozamiento cinético $\mu$ es adimensional y la fuerza de rozamiento $\mu mg$ es constante en módulo. Se define la longitud característica $\delta=\mu mg/k$, el desplazamiento al que el resorte equilibra al rozamiento. El movimiento arranca del reposo en $x_0$.",
         [r"[\delta]=\dfrac{1\cdot\mathrm{kg}\cdot\mathrm{m/s^2}}{\mathrm{N/m}}=\mathrm{m},\qquad f=\mu mg=\text{cte},\qquad E=\tfrac12kx^{2}\ \text{(en reposo)}"]),
        ("Intuición y analogías",
         r"El rozamiento seco actúa como un freno de intensidad constante que siempre se opone al movimiento. En cada media oscilación la masa oscila alrededor de un centro desplazado hacia el lado hacia el que se mueve, y ese centro cambia de lado en cada retorno. Como cada semiciclo resta una cantidad fija de amplitud, el decaimiento es lineal y no exponencial, y la oscilación se apaga de golpe cuando el resorte ya no logra vencer al rozamiento.",
         []),
        ("Lagrangiano con fuerza generalizada no conservativa",
         r"El lagrangiano del resorte es conservativo y el rozamiento seco entra como fuerza generalizada $Q=-\mu mg\,\mathrm{sgn}(\dot x)$ en el miembro derecho de la ecuación de Lagrange. No se puede derivar de un potencial ni de una función de Rayleigh cuadrática, porque su módulo no depende de la velocidad. La no analiticidad de $\mathrm{sgn}$ obliga a resolver por tramos de signo constante. El teorema de Noether temporal no da conservación de energía, ya que la fuerza disipa a razón $Q\dot x\le0$.",
         [r"L=\tfrac12m\dot x^{2}-\tfrac12kx^{2},\qquad \dfrac{d}{dt}\dfrac{\partial L}{\partial\dot x}-\dfrac{\partial L}{\partial x}=Q",
          r"m\ddot x+kx=-\mu mg\,\mathrm{sgn}(\dot x)",
          r"\dfrac{dE}{dt}=Q\,\dot x=-\mu mg\,|\dot x|\le0,\qquad E=\tfrac12m\dot x^{2}+\tfrac12kx^{2}"]),
        ("Solución por semiciclos",
         r"En un tramo con $\dot x<0$ la ecuación es lineal con término constante, $m\ddot x+k(x-\delta)=0$, que es un oscilador armónico con el equilibrio desplazado a $x=+\delta$. La solución es un coseno de frecuencia $\omega=\sqrt{k/m}$ centrado en $\delta$ y parte del reposo en $x_n>0$. Dura exactamente medio período, porque la velocidad vuelve a anularse en $x=2\delta-x_n$. En un tramo con $\dot x>0$ el equilibrio se desplaza a $-\delta$ y el razonamiento es idéntico por simetría.",
         [r"\dot x<0:\quad x(t)=\delta+(x_n-\delta)\cos\omega t,\qquad \omega=\sqrt{k/m}",
          r"\dot x(t)=0\ \Rightarrow\ \omega t=\pi,\qquad x=\delta-(x_n-\delta)=2\delta-x_n",
          r"\dot x>0:\quad x(t)=-\delta+(x_n+\delta)\cos\omega t"]),
        ("Recurrencia de amplitudes y energía disipada",
         r"Aplicando el resultado de un semiciclo repetidas veces, el valor absoluto de la elongación en cada retorno disminuye en $2\delta$, lo que da una progresión aritmética. El balance de energía confirma el resultado, ya que la energía perdida en un semiciclo es el trabajo de la fricción sobre el recorrido, $\mu mg(|x_n|+|x_{n+1}|)$. El movimiento termina cuando el retorno cae en la zona muerta $|x_n|\le\delta$, donde el resorte ya no supera al rozamiento. El número de semiciclos es el menor entero que cumple esa condición.",
         [r"|x_{n+1}|=|x_n|-2\delta\;\Longrightarrow\;|x_n|=|x_0|-2n\delta,\qquad x_n=(-1)^{n}\,(x_0-2n\delta)",
          r"\tfrac12k\left(x_n^{2}-x_{n+1}^{2}\right)=\mu mg\left(|x_n|+|x_{n+1}|\right)",
          r"N=\left\lceil\dfrac{x_0-\delta}{2\delta}\right\rceil,\qquad t_{\rm stop}=N\,\pi\sqrt{m/k}"]),
        ("Casos límite y comparación con el amortiguamiento viscoso",
         r"Si el rozamiento es pequeño, $\delta\to0$, la envolvente decae linealmente con pendiente $-2\delta/(\pi\sqrt{m/k})$ por unidad de tiempo, mientras que la viscosa decae exponencialmente y nunca se detiene. Si $x_0\le\delta$ el resorte no logra mover la masa: no hay movimiento. Si $\mu=0$ se recupera el oscilador armónico, con $N\to\infty$. El parámetro adimensional del problema es $x_0/\delta$, que fija el número de semiciclos.",
         [r"\dfrac{d|x|_{\rm env}}{dt}=-\dfrac{2\delta}{\pi\sqrt{m/k}}=-\dfrac{2\delta\,\omega}{\pi}=-\dfrac{2\mu g}{\pi\,\omega}",
          r"x_0\le\delta\Rightarrow N=0;\qquad \mu\to0\Rightarrow\delta\to0,\ N\to\infty",
          r"N\simeq\dfrac{x_0}{2\delta}=\dfrac{k\,x_0}{2\mu mg}\quad(x_0\gg\delta)"]),
        ("Resultado general",
         r"Los puntos de retorno forman una sucesión aritmética de razón $2\delta$ con signos alternados, y la masa se detiene tras $N$ semiciclos con una elongación residual menor que $\delta$ en módulo. Dimensionalmente $\delta$ es una longitud y $\pi\sqrt{m/k}$ un tiempo. La fórmula depende de $k$, $m$, $\mu$ y $g$ solo mediante $\delta$ y $\omega$.",
         [r"\boxed{\;x_n=(-1)^{n}\,(x_0-2n\delta),\qquad \delta=\dfrac{\mu mg}{k},\qquad t_n=n\pi\sqrt{\dfrac mk}\;}",
          r"E_{\rm disip}=\tfrac12k\left(x_0^{2}-x_N^{2}\right)=\mu mg\,\ell_{\rm rec}"]),
        ("Particularización al problema",
         r"Con los datos del enunciado se calculan la longitud característica, los puntos de retorno y el estado final. La sucesión coincide con la de la escena y con la integración numérica por Runge-Kutta con parada en reposo.",
         [rf"k={sci(k,3)}\ \mathrm{{N/m}},\ m={fx(m,1)}\ \mathrm{{kg}},\ \mu={fx(mu,2)},\ x_0={fx(x0,2)}\ \mathrm{{m}},\ \delta=\dfrac{{({fx(mu,2)})({fx(m,1)})({sci(g,4)})}}{{{sci(k,3)}}}={sci(d,4)}\ \mathrm{{m}}",
          r"x_n=(-1)^n\,(x_0-2n\delta)\ :\ \ " + r",\ ".join(sci(p, 4) for p in pts) + r"\ \mathrm{m}",
          rf"N={n_half}\ \text{{semiciclos}},\quad t_{{\rm stop}}={n_half}\pi\sqrt{{m/k}}={sci(ttot,4)}\ \mathrm{{s}},\quad |x_N|={sci(abs(xn),3)}\ \mathrm{{m}}<\delta",
          rf"E_{{\rm disip}}=\tfrac12k(x_0^2-x_N^2)={sci(ediss,4)}\ \mathrm{{J}}"]),
        ("Errores comunes y límites de validez",
         r"Se suele suponer que el movimiento decae exponencialmente, como en el caso viscoso, o que la masa regresa al equilibrio $x=0$: se detiene en cualquier punto de la zona muerta $|x|\le\delta$. Otro error es usar la energía inicial menos $\mu mg\,x_0$ para cada ciclo, olvidando que el recorrido cambia. El modelo supone $\mu_s=\mu_k$; si $\mu_s>\mu_k$ la zona muerta se ensancha a $\mu_smg/k$. Además, el rozamiento real puede depender de la velocidad, y entonces aparece el movimiento a tirones (stick-slip).",
         []),
    ])


DETAIL = {
    "StaticEquilibrium": _static(),
    "InclinePulley": _pulley(),
    "FrictionIncline": _friction(),
    "DragFall": _drag(),
    "SpringWork": _springwork(),
    "WorkEnergyTheorem": _workenergy(),
    "PendulumEnergy": _pendulum(),
    "SpringFriction": _springfric(),
}
