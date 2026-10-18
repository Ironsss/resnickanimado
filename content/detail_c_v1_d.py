"""Concepto y desarrollo analitico general: Volumen 1, capitulos 13 a 16 (SkaterSpin ... FloatingCylinder).

Cada entrada: concept = [(titulo, prosa, [ecuaciones TeX]), ...]. Las cifras de la particularizacion se calculan
con `physics/` y coinciden con las de las filas de la escena.
"""
from __future__ import annotations

import numpy as np

from content.texfmt import fx, sci
from physics import ch13_angular_momentum as c13
from physics import ch14_equilibrium as c14
from physics import ch15_gravitation as c15
from physics import ch16_fluid_statics as c16


# ============================================================================ SkaterSpin
def _skater() -> list:
    ii, i_f, wi = c13.I_I, c13.I_F, c13.W_I
    wf = c13.omega_final_analytic()
    dk = c13.kinetic_energy_change_analytic()
    ki, kf = 0.5 * ii * wi**2, 0.5 * i_f * wf**2
    L = ii * wi
    return [
        ("Qué se estudia y por qué importa",
         r"Cuando un cuerpo que gira cambia la distribución de su masa respecto del eje, su velocidad angular cambia aunque no actúe ningún torque externo. Es lo que hace una patinadora al cerrar los brazos, un clavadista al encogerse en el aire o una estrella que colapsa y se convierte en pulsar. El fenómeno no es una curiosidad: es la manifestación de una ley de conservación asociada a la isotropía del espacio. En ingeniería aparece en volantes de inercia, satélites con ruedas de reacción y mecanismos de control de actitud.",
         []),
        ("Magnitudes y convenciones",
         r"El momento de inercia $I=\sum m_i\rho_i^2$ (kg m²) mide cuánta masa hay y qué tan lejos del eje está; $\omega$ (rad/s) es la rapidez angular y $L=I\omega$ (kg m²/s) el momento angular respecto del eje. Se toma como positivo el sentido de giro inicial. Para una persona con los brazos extendidos $I$ ronda 3 a 5 kg m², y con los brazos recogidos baja a 1 a 2 kg m²; una patinadora en trompo alcanza entre 5 y 10 rev/s. La energía cinética de rotación $K=\tfrac12 I\omega^2$ se mide en joules.",
         [r"[I]=\mathrm{kg\,m^2},\qquad [\omega]=\mathrm{rad/s},\qquad [L]=\mathrm{kg\,m^2/s}=\mathrm{J\,s}"]),
        ("Intuición y analogías",
         r"La rotación es el análogo de la traslación: $I$ juega el papel de la masa, $\omega$ el de la velocidad y $L$ el del momento lineal. Si el cuerpo no recibe impulso externo, el producto $I\omega$ se mantiene; si $I$ baja a la mitad, $\omega$ debe duplicarse. La energía, en cambio, no se conserva: los músculos realizan trabajo al tirar de los brazos contra la fuerza centrífuga aparente, de modo que la energía cinética crece a costa de la energía química interna. Es como acercar dos imanes que se repelen: el sistema gana energía porque alguien hace trabajo.",
         []),
        ("Planteamiento general: lagrangiana con coordenada de forma",
         r"Sea un sistema que gira alrededor de un eje fijo $z$ con ángulo $\varphi$ y cuya configuración interna (posición de los brazos) se describe por una coordenada generalizada $s$. Para un cuerpo con eje principal alineado con $z$, la energía cinética es $T=\tfrac12 I(s)\dot\varphi^2+\tfrac12\mu\dot s^2$, donde $\mu$ es una masa efectiva del movimiento interno. Se supone que no hay torque externo: el potencial $V(s)$ no depende de $\varphi$. Esa independencia es la hipótesis de simetría del problema.",
         [r"\mathcal L(\varphi,s,\dot\varphi,\dot s)=\tfrac12 I(s)\,\dot\varphi^{2}+\tfrac12\mu\,\dot s^{2}-V(s)",
          r"I(s)=\sum_i m_i\,\rho_i^2(s)"]),
        ("Ecuaciones de Euler-Lagrange y teorema de Noether",
         r"Como $\varphi$ no aparece en $\mathcal L$ (coordenada cíclica), su momento conjugado es una constante del movimiento; esto es el teorema de Noether para invariancia ante rotaciones alrededor de $z$. El momento conjugado resulta ser el momento angular. La ecuación para $s$ contiene el término centrífugo $\tfrac12 I'(s)\dot\varphi^2$, que acopla ambos movimientos.",
         [r"\dfrac{d}{dt}\dfrac{\partial\mathcal L}{\partial\dot\varphi}=\dfrac{\partial\mathcal L}{\partial\varphi}=0\;\Rightarrow\; p_\varphi=I(s)\,\dot\varphi=L=\mathrm{cte}",
          r"\mu\ddot s=\tfrac12 I'(s)\,\dot\varphi^{2}-V'(s)=\dfrac{L^2\,I'(s)}{2I(s)^2}-V'(s)"]),
        ("Cuerpo rígido general: ecuaciones de Euler",
         r"En un cuerpo no simétrico el momento angular es $\vec L=\mathsf I\vec\omega$ con tensor de inercia $\mathsf I$, y en ejes principales móviles $\vec L$ solo es paralelo a $\vec\omega$ si $\vec\omega$ es un eje principal. La ley $d\vec L/dt=\vec\tau$ escrita en el cuerpo da las ecuaciones de Euler, donde el término $\vec\omega\times\mathsf I\vec\omega$ es el efecto giroscópico. Para rotación alrededor de un eje principal $\omega_1=\omega$, $\omega_2=\omega_3=0$ ese término se anula y queda $I_1\dot\omega_1=\tau_1$, que es el caso del problema. Cuando además el momento de inercia cambia con el tiempo, el miembro izquierdo es $d(I\omega)/dt$.",
         [r"I_1\dot\omega_1-(I_2-I_3)\,\omega_2\omega_3=\tau_1,\quad I_2\dot\omega_2-(I_3-I_1)\,\omega_3\omega_1=\tau_2,\quad I_3\dot\omega_3-(I_1-I_2)\,\omega_1\omega_2=\tau_3",
          r"\tau_{ext}=0,\ \omega_2=\omega_3=0\;\Rightarrow\;\dfrac{d}{dt}\big(I\omega\big)=0\;\Rightarrow\;\dot\omega=-\dfrac{\dot I}{I}\,\omega"]),
        ("Solución para cualquier historia I(t) y balance de energía",
         r"La ecuación $\dot\omega=-(\dot I/I)\,\omega$ es separable y se integra de inmediato: el resultado depende solo de los valores inicial y final de $I$, no de cómo se recojan los brazos. La energía cinética es $K=L^2/(2I)$, por lo que disminuir $I$ a $L$ fija la aumenta. El trabajo lo realiza la fuerza interna contra la inercia centrífuga y se obtiene integrando la potencia $dK/dt=-\tfrac12\omega^2\dot I$; con $\omega=L/I$ esa integral da exactamente $\Delta K$ sin importar el camino.",
         [r"\dfrac{d\omega}{\omega}=-\dfrac{dI}{I}\;\Rightarrow\;\omega(t)=\omega_i\,\dfrac{I_i}{I(t)}",
          r"\dfrac{dK}{dt}=-\tfrac12\,\omega^{2}\dot I,\qquad W_{int}=\Delta K=\dfrac{L^2}{2}\left(\dfrac1{I_f}-\dfrac1{I_i}\right)"]),
        ("Parámetro adimensional y casos límite",
         r"Todo el problema depende del cociente $\lambda=I_i/I_f$: la velocidad angular se multiplica por $\lambda$ y la energía cinética también. Si $\lambda=1$ nada cambia; si $\lambda\to\infty$ (colapso estelar) la energía rotacional crece sin cota y deja de ser válida la hipótesis de cuerpo rígido, porque la fuerza centrífuga alcanza a la gravedad o a la cohesión. En el caso opuesto $\lambda<1$ (extender los brazos) hay que realizar trabajo contra la inercia para frenar el giro.",
         [r"\lambda\equiv\dfrac{I_i}{I_f},\qquad \dfrac{\omega_f}{\omega_i}=\lambda,\qquad \dfrac{K_f}{K_i}=\lambda"]),
        ("Resultado general",
         r"Para un sistema con eje de rotación principal fijo y sin torque externo, la velocidad angular final y el cambio de energía cinética quedan expresados únicamente por los momentos de inercia inicial y final y por la velocidad angular inicial. Las dimensiones son consistentes: $I\omega$ conserva kg m²/s y $\tfrac12 I\omega^2$ tiene unidades de joule. El resultado depende linealmente de $\omega_i$ en la velocidad y cuadráticamente en la energía.",
         [r"\boxed{\;\omega_f=\dfrac{I_i}{I_f}\,\omega_i,\qquad \Delta K=\tfrac12 I_i\omega_i^{2}\left(\dfrac{I_i}{I_f}-1\right)\;}",
          r"[\Delta K]=\mathrm{kg\,m^2}\cdot\mathrm{s^{-2}}=\mathrm{J}"]),
        ("Particularización al problema",
         r"Se sustituyen $I_i=4.0$ kg m², $I_f=1.6$ kg m² y $\omega_i=2$ rad/s en las fórmulas generales. El cociente $\lambda=2.5$ multiplica tanto la velocidad como la energía, y el momento angular se mantiene en su valor inicial.",
         [rf"\lambda=\dfrac{{{fx(ii,1)}}}{{{fx(i_f,1)}}}={fx(ii/i_f,2)},\qquad \omega_f=\lambda\,\omega_i={fx(wf,2)}\ \mathrm{{rad/s}}",
          rf"L=I_i\omega_i={fx(L,1)}\ \mathrm{{kg\,m^2/s}}=I_f\omega_f={fx(i_f*wf,1)}\ \mathrm{{kg\,m^2/s}}",
          rf"K_i={fx(ki,1)}\ \mathrm{{J}},\quad K_f={fx(kf,1)}\ \mathrm{{J}},\quad \Delta K=\tfrac12({fx(ii,1)})({fx(wi,1)})^2({fx(ii/i_f-1,2)})={fx(dk,1)}\ \mathrm{{J}}"]),
        ("Errores comunes y límites de validez",
         r"Un error frecuente es afirmar que la energía se conserva porque no hay torque externo: se conserva $L$, no $K$, ya que las fuerzas internas sí realizan trabajo. Otro es olvidar que el eje debe ser principal; si no lo es, aparece el término $\vec\omega\times\mathsf I\vec\omega$ y el eje se bambolea. Además, la rotación alrededor del eje intermedio de un cuerpo asimétrico es inestable, de modo que un giro limpio exige un eje de mayor o menor momento de inercia. El modelo falla cuando hay fricción con el hielo, que ejerce un torque pequeño pero no nulo y hace decaer $L$ lentamente, o cuando el cuerpo se deforma tanto que el tensor de inercia deja de ser constante en el cuerpo.",
         [r"\vec\tau_{ext}\neq0\;\Rightarrow\;\dfrac{dL}{dt}=\tau_{ext}\ \ (\text{ya no se conserva})"]),
    ]


# ============================================================================ GyroscopePrecession
def _gyro() -> list:
    m, r, ii, w, g = c13.M_GYRO, c13.R_GYRO, c13.I_GYRO, c13.W_SPIN, c13.G
    big_l = ii * w
    tau = m * g * r
    om = c13.precession_analytic()
    om2 = c13.precession_analytic(2 * w)
    return [
        ("Qué se estudia y por qué importa",
         r"Un giróscopo que gira rápido no cae cuando se apoya en un extremo: su eje describe lentamente un círculo horizontal, fenómeno llamado precesión. Se aprovecha en brújulas giroscópicas, sistemas inerciales de navegación, estabilización de barcos y satélites, y explica por qué una bicicleta o una peonza se mantienen erguidas. La precesión de los equinoccios de la Tierra, con periodo cercano a 26 000 años, es el mismo efecto producido por el torque de la Luna y el Sol sobre el abultamiento ecuatorial.",
         []),
        ("Magnitudes y convenciones",
         r"Intervienen la masa $m$ (kg), la distancia $r$ (m) del centro de masa al pivote, el momento de inercia axial $I$ (kg m²), la rapidez de giro $\omega$ (rad/s) y la gravedad $g$. El momento angular de espín es $L=I\omega$ y el torque del peso respecto del pivote, $\tau=mgr$ (N m). La precesión $\Omega$ (rad/s) se mide en el plano horizontal. Para un giróscopo de laboratorio $\omega$ ronda 100 a 1000 rad/s y $\Omega$ unas fracciones de rad/s a pocos rad/s.",
         [r"[\Omega]=\dfrac{\mathrm{N\,m}}{\mathrm{kg\,m^2\cdot rad/s}}=\mathrm{rad/s}"]),
        ("Intuición y analogías",
         r"El torque no inclina el eje hacia abajo, sino que cambia $\vec L$ en la dirección del propio torque, que es horizontal y perpendicular a $\vec L$. Un cambio perpendicular a un vector solo lo rota sin modificar su módulo, igual que una fuerza centrípeta cambia la dirección de la velocidad sin cambiar la rapidez. Cuanto mayor es $L$, más difícil es desviarlo y más lenta es la precesión. Por eso un giróscopo muy rápido parece desafiar la gravedad.",
         []),
        ("Ecuaciones de Euler del cuerpo simétrico",
         r"Un giróscopo es un cuerpo con simetría axial, $I_1=I_2\neq I_3$, y con un punto fijo. La dinámica del cuerpo rígido en ejes principales es regida por las ecuaciones de Euler, que expresan $d\vec L/dt=\vec\tau$ en el sistema que rota con el cuerpo. Con $I_1=I_2$ la tercera ecuación implica que la componente de $\vec\omega$ sobre el eje de simetría es constante cuando el torque no tiene componente axial, lo cual ocurre con el peso, porque su torque es perpendicular al eje.",
         [r"I_1\dot\omega_1-(I_1-I_3)\,\omega_2\omega_3=\tau_1,\qquad I_1\dot\omega_2-(I_3-I_1)\,\omega_3\omega_1=\tau_2,\qquad I_3\dot\omega_3=\tau_3=0",
          r"\omega_3=\mathrm{cte}\equiv\omega"]),
        ("Lagrangiana del trompo con ángulos de Euler",
         r"Los ángulos de Euler $(\varphi,\theta,\psi)$ dan la precesión, la inclinación respecto de la vertical y el espín. La energía cinética es la de un cuerpo simétrico con punto fijo y el potencial es el del peso a la altura $l\cos\theta$ del centro de masa (con $l=r$ la distancia al pivote). Esta formulación convierte la dinámica en un problema con dos coordenadas cíclicas.",
         [r"\mathcal L=\tfrac12 I_1\big(\dot\theta^2+\dot\varphi^2\sin^2\theta\big)+\tfrac12 I_3\big(\dot\psi+\dot\varphi\cos\theta\big)^2-mgl\cos\theta"]),
        ("Integrales primeras por simetría",
         r"Como $\varphi$ y $\psi$ no aparecen explícitamente en $\mathcal L$, sus momentos conjugados se conservan (Noether para rotaciones alrededor de la vertical y del eje del cuerpo); además el lagrangiano no depende del tiempo y se conserva la energía. Estas tres integrales reducen el sistema a un problema unidimensional en $\theta$ con potencial efectivo, cuyo mínimo da la inclinación estable y cuyas pequeñas oscilaciones son la nutación.",
         [r"p_\psi=I_3\big(\dot\psi+\dot\varphi\cos\theta\big)=I_3\omega_3,\qquad p_\varphi=I_1\dot\varphi\sin^2\theta+p_\psi\cos\theta",
          r"E=\tfrac12 I_1\big(\dot\theta^2+\dot\varphi^2\sin^2\theta\big)+\tfrac12 I_3\omega_3^2+mgl\cos\theta"]),
        ("Precesión estacionaria y aproximación giroscópica",
         r"Una precesión uniforme a ángulo $\theta$ constante exige $\ddot\theta=0$, que en la ecuación de Lagrange para $\theta$ da una condición cuadrática en $\dot\varphi$. Cuando el eje es horizontal ($\theta=\pi/2$, $\cos\theta=0$) la cuadrática desaparece y la precesión es exactamente lineal en $1/\omega_3$. Para otros ángulos, el espín rápido da dos raíces: una precesión lenta, que es la observada, y otra rápida, que no se mantiene por sí sola. Aquí $\omega_3$ es el espín propio $\dot\psi+\dot\varphi\cos\theta$.",
         [r"I_1\dot\varphi^{2}\cos\theta-I_3\omega_3\,\dot\varphi+mgl=0",
          r"\dot\varphi_{\pm}=\dfrac{I_3\omega_3\pm\sqrt{I_3^2\omega_3^2-4I_1mgl\cos\theta}}{2I_1\cos\theta}\;\xrightarrow{\ I_3\omega_3\gg\sqrt{I_1mgl}\ }\;\Omega=\dfrac{mgl}{I_3\omega_3}"]),
        ("Parámetros adimensionales, nutación y casos límite",
         r"La precesión lenta existe si el discriminante es positivo, es decir si $I_3^2\omega_3^2>4I_1mgl\cos\theta$; por debajo de ese espín crítico un trompo inclinado cae. La razón $\Omega/\omega$ mide la validez de la aproximación giroscópica y debe ser mucho menor que 1. Superpuesta a la precesión hay una nutación de frecuencia $\omega_n\approx I_3\omega_3/I_1$, que se amortigua en la práctica por fricción. Si $\omega\to0$ la precesión diverge, signo de que el modelo se rompe y el cuerpo simplemente cae como un péndulo.",
         [r"\varepsilon\equiv\dfrac{\Omega}{\omega}=\dfrac{mgl}{I_3\omega^2}\ll1,\qquad \omega_n\simeq\dfrac{I_3\omega}{I_1}"]),
        ("Resultado general",
         r"Para un giróscopo rápido con eje horizontal la velocidad de precesión es el cociente entre el torque del peso y el momento angular de espín, y apunta alrededor de la vertical en el sentido de $\vec L\times\vec\tau$ visto desde arriba. Es inversamente proporcional a $\omega$ y directamente proporcional a $mgr$. Dimensionalmente, N m entre kg m² rad/s da rad/s.",
         [r"\boxed{\;\Omega=\dfrac{\tau}{L}=\dfrac{m\,g\,r}{I\,\omega}\;}",
          r"\vec\Omega\ \text{vertical},\qquad \dfrac{d\vec L}{dt}=\vec\Omega\times\vec L=\vec r\times m\vec g"]),
        ("Particularización al problema",
         r"Con $m=0.5$ kg, $r=0.10$ m, $I=2.5\times10^{-3}$ kg m² y $\omega=200$ rad/s se calculan el momento angular, el torque y la precesión; al duplicar $\omega$ la precesión se reduce a la mitad.",
         [rf"L=I\omega=({sci(ii,2)})({fx(w,0)})={fx(big_l,2)}\ \mathrm{{kg\,m^2/s}},\qquad \tau=mgr=({fx(m,1)})({fx(g,2)})({fx(r,2)})={fx(tau,3)}\ \mathrm{{N\,m}}",
          rf"\Omega=\dfrac{{\tau}}{{L}}=\dfrac{{{fx(tau,3)}}}{{{fx(big_l,2)}}}={fx(om,3)}\ \mathrm{{rad/s}},\qquad \dfrac{{\Omega}}{{\omega}}={sci(om/w,2)}\ll1",
          rf"\omega=400\ \mathrm{{rad/s}}:\quad \Omega={fx(om2,3)}\ \mathrm{{rad/s}}"]),
        ("Errores comunes y límites de validez",
         r"Se suele creer que el torque debería hacer caer el eje; esa idea vale para un cuerpo sin espín, no para uno con gran $L$. Otro error es usar $\omega$ total en vez del espín propio, o sustituir $I_3$ por el momento de inercia respecto del pivote en la fórmula del espín. La aproximación giroscópica falla si $\omega$ es pequeño, si hay fricción apreciable en el pivote, que hace perder espín, o si el ángulo está cerca de la vertical, donde hay que usar la cuadrática completa. Además, la precesión del giróscopo real incluye la nutación inicial: la lenta es el promedio.",
         []),
    ]


# ============================================================================ LadderFriction
def _ladder() -> list:
    L, we, wh, th, mu = c14.L_LAD, c14.W_LAD, c14.W_MAN, c14.THETA_DEG, c14.MU
    t = np.deg2rad(th)
    dmax = c14.d_max_analytic()
    amin = c14.min_angle_analytic()
    fr = c14.ladder_forces_analytic(2.5)
    nf = we + wh
    return [
        ("Qué se estudia y por qué importa",
         r"Una escalera apoyada en una pared se mantiene en equilibrio gracias a la fricción en el suelo; si el pie resbala, la escalera cae. El problema de determinar hasta dónde puede subir una persona o cuál es el ángulo seguro es un caso típico de estática de cuerpos rígidos con rozamiento, y el mismo razonamiento se emplea al analizar andamios, grúas, puertas y estructuras ligeras. La seguridad real depende de que el cociente entre la fuerza de fricción necesaria y la normal quede por debajo del coeficiente estático.",
         []),
        ("Magnitudes y convenciones",
         r"La longitud de la escalera es $L$ (m), su peso $W_e$ (N) actúa en el punto medio y el del hombre, $W_h$ (N), a una distancia $d$ del pie medida a lo largo de la escalera. El ángulo $\theta$ con el suelo se mide en la base. $N_f$ y $f$ son la normal y la fricción en el suelo; $N_p$, la reacción de la pared, que por ser lisa es solo normal. El coeficiente de fricción estática $\mu_s$ es adimensional, típicamente entre 0.3 y 0.8 para caucho sobre pavimento.",
         [r"[W]=\mathrm{N},\qquad [d]=[L]=\mathrm{m},\qquad f\le\mu_s N_f"]),
        ("Intuición y analogías",
         r"La pared empuja la parte alta hacia afuera y el suelo debe contrarrestar ese empuje con fricción: si la escalera es más vertical, la pared empuja menos; si es más tendida, empuja más. Subir aumenta el torque de los pesos respecto del pie, por lo que la pared empuja más y la fricción requerida crece. Es como un arco: la reacción horizontal en los apoyos equilibra el empuje y se necesita un suelo que lo soporte.",
         []),
        ("Planteamiento general: equilibrio y trabajo virtual",
         r"Un cuerpo rígido está en equilibrio si la fuerza neta y el torque neto son nulos; de forma equivalente, si el trabajo virtual de todas las fuerzas activas es cero para cualquier desplazamiento compatible con los vínculos. Se modela la escalera como una barra con extremos en la pared y en el suelo, un solo grado de libertad $\theta$ y reacciones desconocidas $N_f$, $f$, $N_p$. Hipótesis: barra rígida, pared lisa, contacto puntual en el pie y problema plano.",
         [r"\sum F_x=f-N_p=0,\qquad \sum F_y=N_f-W_e-W_h=0",
          r"\sum \tau_{pie}=N_pL\sin\theta-\Big(W_e\tfrac L2+W_hd\Big)\cos\theta=0"]),
        ("Reacciones en función de la posición",
         r"De las dos ecuaciones de fuerza, la normal del suelo es la suma de los pesos y la fricción coincide con la reacción de la pared; la ecuación de torques respecto del pie, donde no actúan ni $N_f$ ni $f$, determina esa reacción. El sistema es isostático: tres ecuaciones, tres incógnitas. La fricción necesaria crece linealmente con $d$ y con $\cot\theta$.",
         [r"N_f=W_e+W_h",
          r"f=N_p=\dfrac{\big(W_eL/2+W_hd\big)\cos\theta}{L\sin\theta}=\dfrac{\big(W_e/2+W_h\,\xi\big)}{\tan\theta},\qquad \xi\equiv\dfrac dL"]),
        ("Verificación por el principio de los trabajos virtuales",
         r"Con la pared en $x=0$ y el pie en $x=L\cos\theta$, el centro de masa de la barra queda a altura $\tfrac L2\sin\theta$ y la persona a $d\sin\theta$. Un desplazamiento virtual $\delta\theta$ mueve el pie $\delta x=-L\sin\theta\,\delta\theta$; la fuerza de fricción actúa en sentido contrario al empuje de la pared. La pared lisa y la normal del suelo no realizan trabajo virtual, porque sus fuerzas son perpendiculares a los desplazamientos de sus puntos de aplicación. Anular el trabajo total reproduce el resultado del balance de torques.",
         [r"V(\theta)=\Big(W_e\tfrac L2+W_hd\Big)\sin\theta,\qquad \delta W=-V'(\theta)\,\delta\theta+f\,L\sin\theta\,\delta\theta=0",
          r"f=\dfrac{V'(\theta)}{L\sin\theta}=\dfrac{\big(W_eL/2+W_hd\big)\cos\theta}{L\sin\theta}"]),
        ("Condición de no deslizamiento",
         r"La ley de Coulomb exige que la fricción estática no supere el valor límite $\mu_sN_f$. Como $f$ crece con $d$, el deslizamiento comienza cuando se alcanza la igualdad; esto fija la distancia máxima que puede ascender la persona. Si el valor obtenido supera $L$, la escalera no resbala en ningún punto. Y si es negativo, resbala incluso sin carga adicional.",
         [r"f\le\mu_sN_f\;\Rightarrow\;\dfrac{W_eL/2+W_hd}{L}\cot\theta\le\mu_s\,(W_e+W_h)",
          r"d\le d_{max}=\dfrac{\mu_s(W_e+W_h)L\tan\theta-W_eL/2}{W_h}"]),
        ("Ángulo mínimo, parámetros adimensionales y límites",
         r"Despejando el ángulo para una posición dada $\xi=d/L$ se obtiene la inclinación mínima segura; llegar al extremo corresponde a $\xi=1$. El resultado depende de tres números adimensionales: $\mu_s$, la razón $W_h/W_e$ y $\xi$. Si el hombre es muy pesado respecto de la escalera ($W_h\gg W_e$), el ángulo mínimo tiende a $\arctan(\xi/\mu_s)$; si $\mu_s\to0$ ninguna inclinación finita basta y se necesita sujetar el pie; si la pared tuviera fricción, el problema se vuelve hiperestático.",
         [r"\tan\theta_{min}=\dfrac{W_e/2+W_h\,\xi}{\mu_s\,(W_e+W_h)},\qquad \xi=\dfrac dL",
          r"W_h\gg W_e:\quad \tan\theta_{min}\to\dfrac{\xi}{\mu_s}"]),
        ("Resultado general",
         r"La fricción requerida, la distancia máxima y el ángulo mínimo se expresan con los pesos, la longitud, el ángulo y el coeficiente de fricción. Las tres fórmulas son homogéneas en las fuerzas, de modo que solo importan sus cocientes, y las longitudes son proporcionales a $L$.",
         [r"\boxed{\;d_{max}=\dfrac{\mu_s(W_e+W_h)\,L\tan\theta-W_eL/2}{W_h},\qquad \theta_{min}=\arctan\dfrac{W_e/2+W_h\,\xi}{\mu_s(W_e+W_h)}\;}",
          r"f(d)=\dfrac{\big(W_e/2+W_h\,d/L\big)}{\tan\theta},\qquad [d]=\mathrm{m},\ [\theta]=\mathrm{rad}"]),
        ("Particularización al problema",
         r"Con $L=5$ m, $W_e=150$ N, $W_h=700$ N, $\theta=60^\circ$ y $\mu_s=0.40$ se obtienen la distancia máxima, el ángulo mínimo para $\xi=1$ y la fricción cuando el hombre está a 2.5 m.",
         [rf"N_f=W_e+W_h={fx(nf,0)}\ \mathrm{{N}},\qquad \mu_sN_f={fx(mu*nf,0)}\ \mathrm{{N}}",
          rf"d_{{max}}=\dfrac{{({fx(mu,2)})({fx(nf,0)})({fx(L,0)})\tan{fx(th,0)}^\circ-({fx(we,0)})({fx(L,0)})/2}}{{{fx(wh,0)}}}={fx(dmax,3)}\ \mathrm{{m}}",
          rf"\theta_{{min}}=\arctan\dfrac{{{fx(we/2+wh,0)}}}{{({fx(mu,2)})({fx(nf,0)})}}={fx(amin,2)}^\circ",
          rf"d=2.5\ \mathrm{{m}}:\quad f=\dfrac{{{fx(we/2+wh*2.5/L,1)}}}{{\tan{fx(th,0)}^\circ}}={fx(fr['f'],2)}\ \mathrm{{N}}"]),
        ("Errores comunes y límites de validez",
         r"Un error común es suponer que la fricción vale siempre $\mu_sN_f$; esa es solo su cota, y en equilibrio vale lo que exija el balance. Otro es tomar torques respecto de puntos donde aparecen reacciones desconocidas, lo que complica el álgebra. El modelo supone pared lisa, escalera rígida y suelo con fricción uniforme; con pared rugosa el sistema es indeterminado y requiere deformaciones. Tampoco incluye el vuelco de la persona ni la dinámica del deslizamiento, que sigue otras ecuaciones una vez que se supera la cota.",
         []),
    ]


# ============================================================================ BeamReactions
def _beam() -> list:
    L, wb, wp = c14.BEAM_L, c14.BEAM_W, c14.PERSON_W
    x = 2.0
    ra, rb = c14.beam_reactions_analytic(x)
    return [
        ("Qué se estudia y por qué importa",
         r"Las vigas apoyadas en dos extremos sostienen puentes, pisos y pasarelas, y cada apoyo debe resistir una reacción que depende de dónde esté la carga. Conocer esas reacciones es el primer paso del cálculo estructural: de ellas salen los esfuerzos cortantes y los momentos flectores que dimensionan la sección. El problema muestra cómo se reparte una carga entre dos soportes según su posición.",
         []),
        ("Magnitudes y convenciones",
         r"La viga tiene longitud $L$ (m) y peso $W_b$ (N), uniformemente distribuido; la carga puntual $W_p$ (N) se aplica en la abscisa $x$ medida desde el apoyo A. Las reacciones $R_A$ y $R_B$ (N) son verticales y positivas hacia arriba. La razón adimensional $\xi=x/L$ varía entre 0 y 1. En un puente peatonal $W_p$ puede ser comparable al peso propio de la viga.",
         [r"[R_A]=[R_B]=[W]=\mathrm{N},\qquad \xi=\dfrac xL\in[0,1]"]),
        ("Intuición y analogías",
         r"Si la persona está sobre un apoyo, ese apoyo la sostiene completamente; si está en el centro, se reparte por igual. Entre medias, el reparto es proporcional a la cercanía: es la regla de la palanca. El peso propio, simétrico, contribuye con la mitad a cada apoyo. Es el mismo reparto de una carga cargada por dos personas que sostienen una tabla.",
         []),
        ("Planteamiento general: equilibrio con carga distribuida",
         r"Se considera una carga vertical por unidad de longitud $q(x)$, que incluye el peso propio y cualquier carga adicional, y reacciones verticales en $x=0$ y $x=L$. El equilibrio exige que la resultante de fuerzas y el momento resultante sean nulos; al ser dos incógnitas y dos ecuaciones, el problema es isostático. La carga puntual se modela con una delta de Dirac.",
         [r"R_A+R_B=\int_0^L q(x)\,dx\equiv W",
          r"R_BL=\int_0^L q(x)\,x\,dx,\qquad q(x)=\dfrac{W_b}{L}+W_p\,\delta(x-x_p)"]),
        ("Reacciones para una carga arbitraria",
         r"Resolviendo el sistema, cada reacción es un promedio ponderado de la carga con pesos lineales, de modo que cada elemento de carga $q\,dx$ aporta a los apoyos según la regla de la palanca. Estos pesos son las líneas de influencia de las reacciones y son aditivas por linealidad. La conservación del momento se verifica con cualquier eje de torque, no solo con el apoyo A.",
         [r"R_A=\dfrac1L\int_0^L q(x)\,(L-x)\,dx,\qquad R_B=\dfrac1L\int_0^L q(x)\,x\,dx",
          r"\sum\tau_P=0\ \text{para todo }P:\quad R_B(L-p)-R_A\,p=\int_0^L q(x)\,(x-p)\,dx"]),
        ("Líneas de influencia por trabajo virtual",
         r"Para hallar $R_B$ basta suprimir el apoyo B y dar a la viga una rotación virtual rígida alrededor de A: el punto de abscisa $x$ sube $x\,\delta\theta$ y el extremo B sube $L\,\delta\theta$. El trabajo virtual de una carga unitaria y de la reacción debe anularse, y eso da directamente la línea de influencia. Este método evita resolver ecuaciones simultáneas y generaliza a estructuras más complejas.",
         [r"R_B\,L\,\delta\theta-1\cdot x\,\delta\theta=0\;\Rightarrow\;\eta_B(x)=\dfrac xL,\qquad \eta_A(x)=1-\dfrac xL"]),
        ("Esfuerzos internos y límite de validez isostático",
         r"Con las reacciones conocidas, el cortante y el momento flector se obtienen cortando la viga: el cortante es la suma de fuerzas a un lado y el momento, su integral. Con un tercer apoyo el sistema pasa a ser hiperestático y hay que añadir la ecuación de la elástica $EI\,w''''=q$ para determinar las reacciones. Aquí, al ser rígida y de dos apoyos, la estática basta.",
         [r"V(s)=R_A-\int_0^s q\,dx,\qquad M(s)=R_As-\int_0^s q(x)\,(s-x)\,dx",
          r"EI\,\dfrac{d^4w}{dx^4}=q(x)\quad(\text{si hay más de dos apoyos})"]),
        ("Caso de carga uniforme más una puntual y límites",
         r"Para una viga uniforme y una persona en $x$, el peso propio se reparte a la mitad y la carga puntual se divide linealmente. En los extremos $\xi=0$ y $\xi=1$ un apoyo recibe todo el peso de la persona más la mitad de la viga, y el otro solo la mitad de la viga. La reacción mínima nunca baja de $W_b/2$, de modo que ningún apoyo se descarga. Si la carga se aplicara fuera de los apoyos (voladizo), una reacción podría hacerse negativa.",
         [r"R_A+R_B=W_b+W_p,\qquad R_B-R_A=W_p\,(2\xi-1)"]),
        ("Resultado general",
         r"Las reacciones son funciones lineales de la posición de la carga, cuya suma es el peso total; la dimensión es fuerza y el reparto depende únicamente de $\xi$. La verificación dimensional es inmediata: $W\,x/L$ es una fuerza porque $x/L$ es adimensional.",
         [r"\boxed{\;R_A=\dfrac{W_b}{2}+W_p\Big(1-\dfrac xL\Big),\qquad R_B=\dfrac{W_b}{2}+W_p\,\dfrac xL\;}",
          r"R_A+R_B=W_b+W_p"]),
        ("Particularización al problema",
         r"Con $L=6$ m, $W_b=400$ N, $W_p=700$ N y la persona a $x=2$ m se evalúan las reacciones; el resultado no depende del pivote elegido para los torques.",
         [rf"\xi=\dfrac xL=\dfrac{{{fx(x,0)}}}{{{fx(L,0)}}}={fx(x/L,4)}",
          rf"R_A=\dfrac{{{fx(wb,0)}}}{{2}}+{fx(wp,0)}\,(1-{fx(x/L,4)})={fx(ra,1)}\ \mathrm{{N}}",
          rf"R_B=\dfrac{{{fx(wb,0)}}}{{2}}+{fx(wp,0)}\,({fx(x/L,4)})={fx(rb,1)}\ \mathrm{{N}},\qquad R_A+R_B={fx(ra+rb,1)}\ \mathrm{{N}}"]),
        ("Errores comunes y límites de validez",
         r"Es frecuente repartir la carga puntual por igual entre los apoyos o ubicar el peso de la viga en un extremo en lugar de en su centro. Otro error es olvidar que el resultado no depende del punto de torque: elegir un apoyo solo elimina una incógnita. El modelo es válido para viga rígida, apoyos puntuales y cargas verticales; con apoyos intermedios, vigas flexibles que se apoyan con deformación apreciable o cargas inclinadas hace falta la teoría de elasticidad o incorporar reacciones horizontales.",
         []),
    ]


# ============================================================================ KeplerOrbit
def _kepler() -> list:
    gm, a, e = c15.GM, c15.A_ORBIT, c15.E_ORBIT
    T = c15.period_analytic()
    vp = c15.perihelion_speed()
    rp, ra = a * (1 - e), a * (1 + e)
    p = a * (1 - e**2)
    b = a * np.sqrt(1 - e**2)
    h = rp * vp
    en = -gm / (2 * a)
    return [
        ("Qué se estudia y por qué importa",
         r"Los planetas, los cometas y los satélites artificiales se mueven bajo una fuerza central que decrece como el inverso del cuadrado de la distancia. Kepler formuló tres leyes empíricas con las observaciones de Tycho Brahe, y Newton las dedujo de la gravitación universal. Esa deducción es el modelo de todo el análisis de fuerzas centrales: sirve para planear misiones espaciales, calcular tránsitos y también para el átomo de Rutherford, donde la fuerza coulombiana tiene la misma forma.",
         []),
        ("Magnitudes y convenciones",
         r"El semieje mayor $a$ (UA o m), la excentricidad $e$ (adimensional), el semilado recto $p$, el periodo $T$, la energía específica $\varepsilon=E/m$ y el momento angular específico $h=\ell/m$ (UA²/año o m²/s). Con unidades de UA y años, $GM_\odot=4\pi^2\ \mathrm{UA^3/yr^2}$; en SI, $GM_\odot=1.327\times10^{20}\ \mathrm{m^3/s^2}$. Las órbitas son elipses si $0\le e<1$, parábolas si $e=1$ e hipérbolas si $e>1$.",
         [r"[GM]=\mathrm{m^3/s^2},\qquad [h]=\mathrm{m^2/s},\qquad [\varepsilon]=\mathrm{J/kg}"]),
        ("Intuición y analogías",
         r"El astro se mueve más rápido cuando está cerca del Sol y más lento cuando está lejos, porque el radio vector barre áreas iguales en tiempos iguales. Es como un patinador que acerca los brazos: la conservación del momento angular obliga a girar más rápido al reducir la distancia. La energía total negativa significa que el cuerpo está ligado, igual que una bola en un pozo de potencial que no tiene energía para salir.",
         []),
        ("Planteamiento: lagrangiana en un campo central",
         r"Se reduce el problema de dos cuerpos al de una partícula de masa reducida $m$ en un campo central $V(r)=-k/r$, con $k=GMm$ para gravitación (o $k>0$ en general). La fuerza central conserva el plano de la órbita, de modo que bastan las coordenadas polares $(r,\varphi)$. Hipótesis: masas puntuales, fuerza conservativa, sin otras perturbaciones y $M\gg m$ o masa reducida.",
         [r"\mathcal L=\tfrac12 m\big(\dot r^2+r^2\dot\varphi^2\big)+\dfrac{k}{r}"]),
        ("Integrales primeras: Noether y la segunda ley",
         r"La coordenada $\varphi$ es cíclica porque el campo es isótropo (simetría de rotación); su momento conjugado es el momento angular $\ell$, constante. Geométricamente, el área barrida por el radio vector por unidad de tiempo es $\ell/2m$, que es la segunda ley de Kepler y vale para cualquier fuerza central. La energía se conserva porque $\mathcal L$ no depende del tiempo.",
         [r"p_\varphi=m r^2\dot\varphi=\ell=\mathrm{cte},\qquad \dfrac{dA}{dt}=\dfrac12 r^2\dot\varphi=\dfrac{\ell}{2m}",
          r"E=\tfrac12 m\dot r^{2}+\dfrac{\ell^2}{2mr^2}-\dfrac kr=\mathrm{cte}"]),
        ("Potencial efectivo y clasificación del movimiento",
         r"Eliminando $\dot\varphi$ queda un problema radial unidimensional con potencial efectivo $U_{ef}=\ell^2/(2mr^2)-k/r$: el término centrífugo repulsivo domina cerca del origen y el atractivo, lejos. Su mínimo, en $r=\ell^2/(mk)$, es la órbita circular de energía $-mk^2/(2\ell^2)$. Si $E<0$ hay dos puntos de retorno (elipse); si $E=0$, uno (parábola); si $E>0$, uno (hipérbola).",
         [r"U_{ef}(r)=\dfrac{\ell^2}{2mr^2}-\dfrac kr,\qquad m\ddot r=-\dfrac{dU_{ef}}{dr}",
          r"\dot r=\pm\sqrt{\dfrac2m\big[E-U_{ef}(r)\big]},\qquad r_{\pm}=\dfrac{-k\pm\sqrt{k^2+2E\ell^2/m}}{2E}"]),
        ("Ecuación de la órbita r(φ) y excentricidad",
         r"Con el cambio $u=1/r$ y $d/dt=(\ell u^2/m)\,d/d\varphi$ (fórmula de Binet), la ecuación radial se vuelve una ecuación lineal de oscilador armónico forzado, cuya solución general es una cónica con foco en el centro de fuerzas. Las constantes de integración son la excentricidad, ligada a $E$ y $\ell$, y el ángulo del perihelio, que se fija en $\varphi=0$. La conservación adicional del vector de Runge-Lenz explica que la órbita sea cerrada.",
         [r"\dfrac{d^2u}{d\varphi^2}+u=\dfrac{mk}{\ell^2}\;\Rightarrow\;r(\varphi)=\dfrac{p}{1+e\cos\varphi},\quad p=\dfrac{\ell^2}{mk}",
          r"e=\sqrt{1+\dfrac{2E\ell^2}{mk^2}},\qquad \vec A=\vec v\times\vec\ell-\dfrac{k}{m}\hat r=\mathrm{cte}"]),
        ("Parámetros de la elipse, tercera ley y ecuación de Kepler",
         r"Para $E<0$ los puntos de retorno son perihelio y afelio, y el semieje mayor depende solo de la energía. El área de la elipse es $\pi ab$ y se barre a ritmo constante, por lo que el periodo sale al dividir. Resulta $T^2\propto a^3$ con constante $4\pi^2/GM$, la tercera ley, válida para toda excentricidad. La posición en función del tiempo se obtiene con la anomalía excéntrica $\eta$ mediante la ecuación de Kepler.",
         [r"a=\dfrac{p}{1-e^2}=\dfrac{k}{2|E|},\quad r_p=a(1-e),\quad r_a=a(1+e),\quad b=a\sqrt{1-e^2}",
          r"T=\dfrac{\pi ab}{\ell/2m}=2\pi\sqrt{\dfrac{ma^3}{k}}=2\pi\sqrt{\dfrac{a^3}{GM}}",
          r"r=a(1-e\cos\eta),\qquad \dfrac{2\pi}{T}\,t=\eta-e\sin\eta"]),
        ("Resultado general",
         r"El periodo depende solo del semieje mayor y de la masa central, no de la excentricidad ni de la masa orbitante (si es pequeña). La rapidez en cada punto se obtiene de la integral de energía (vis-viva), y en el perihelio coincide con la fórmula que usa la escena. Dimensionalmente $\sqrt{a^3/GM}$ tiene unidades de tiempo.",
         [r"\boxed{\;T=2\pi\sqrt{\dfrac{a^3}{GM}},\qquad v^2=GM\Big(\dfrac2r-\dfrac1a\Big),\qquad v_p=\sqrt{\dfrac{GM}{a}\,\dfrac{1+e}{1-e}}\;}",
          r"\dfrac{dA}{dt}=\dfrac h2=\dfrac12\sqrt{GM\,a(1-e^2)}"]),
        ("Particularización al problema",
         r"Con unidades de UA y años, $GM=4\pi^2$, $a=2$ UA y $e=0.5$ se calculan los parámetros de la elipse, el periodo, la rapidez en el perihelio y se comprueba que el momento angular es consistente con $p$ y $GM$.",
         [rf"p=a(1-e^2)={fx(p,3)}\ \mathrm{{UA}},\quad r_p={fx(rp,2)}\ \mathrm{{UA}},\quad r_a={fx(ra,2)}\ \mathrm{{UA}},\quad b={fx(b,4)}\ \mathrm{{UA}}",
          rf"T=2\pi\sqrt{{\dfrac{{a^3}}{{4\pi^2}}}}=a^{{3/2}}=({fx(a,0)})^{{3/2}}={fx(T,4)}\ \text{{ años}}",
          rf"v_p=\sqrt{{\dfrac{{4\pi^2}}{{{fx(a,0)}}}\cdot\dfrac{{{fx(1+e,1)}}}{{{fx(1-e,1)}}}}}={fx(vp,3)}\ \mathrm{{UA/yr}},\qquad h=r_pv_p={fx(h,3)}\ \mathrm{{UA^2/yr}}",
          rf"h^2={fx(h**2,2)}={fx(gm*p,2)}=GM\,p,\qquad \varepsilon=-\dfrac{{GM}}{{2a}}={fx(en,3)}\ \mathrm{{UA^2/yr^2}}"]),
        ("Errores comunes y límites de validez",
         r"Se confunde con frecuencia el centro de la elipse con el foco: el Sol está en un foco. Otro error es suponer que la velocidad es constante en módulo, o que la tercera ley depende de $e$. Si la masa orbitante no es despreciable, hay que usar la masa reducida y $G(M+m)$. El modelo deja de valer con perturbaciones de otros planetas, achatamiento del cuerpo central, arrastre atmosférico o efectos relativistas, como el avance del perihelio de Mercurio, que rompen el cierre de la órbita.",
         []),
    ]


# ============================================================================ EscapeVelocity
def _escape() -> list:
    mu = c15.G_N * c15.M_EARTH
    R = c15.R_EARTH
    ve = c15.escape_speed_analytic()
    v0 = 8000.0
    v1 = 10000.0
    rm = c15.rmax_analytic(v0)
    rm1 = c15.rmax_analytic(v1)
    return [
        ("Qué se estudia y por qué importa",
         r"Para enviar una sonda a otro planeta hay que vencer la atracción gravitatoria del propio, y la energía mínima necesaria define la velocidad de escape. Con ella se estima cuánto combustible exige un lanzamiento, se explica por qué la Luna no retiene atmósfera y por qué la Tierra retiene la suya, y se razona sobre los agujeros negros, donde la velocidad de escape alcanza la de la luz. El caso radial es el más simple y revela la estructura energética de cualquier trayectoria.",
         []),
        ("Magnitudes y convenciones",
         r"Se usan la constante gravitacional $G$, la masa del planeta $M$, su radio $R$, la distancia al centro $r$ y la rapidez de lanzamiento $v_0$. La energía potencial se define nula en el infinito y por eso es negativa: $U=-GMm/r$. La aceleración en superficie es $g=GM/R^2$, de modo que el producto $GM=gR^2$ resulta cómodo. Para la Tierra $GM\approx3.986\times10^{14}\ \mathrm{m^3/s^2}$ y $v_{esc}\approx11.2$ km/s.",
         [r"[GM]=\mathrm{m^3/s^2},\qquad GM=gR^2,\qquad U(\infty)=0"]),
        ("Intuición y analogías",
         r"Lanzar un objeto hacia arriba es subir un pozo de potencial cuya profundidad es $GMm/R$. Con energía menor que esa profundidad el objeto alcanza una altura máxima y regresa; con energía igual o mayor, sube indefinidamente. La velocidad de escape es entonces la que lleva la energía cinética a cancelar exactamente la energía potencial, de modo que el objeto llega al infinito con rapidez nula.",
         []),
        ("Planteamiento: lagrangiana del movimiento radial",
         r"En el movimiento radial el momento angular es cero y solo importa la distancia $r$. La lagrangiana de la partícula en el campo gravitatorio de una masa esférica (que atrae como si estuviera concentrada en el centro) es la de un grado de libertad. Hipótesis: masa del planeta mucho mayor que la del proyectil, planeta esférico, no rota, sin atmósfera, sin otros cuerpos.",
         [r"\mathcal L(r,\dot r)=\tfrac12 m\dot r^{2}+\dfrac{GMm}{r}",
          r"m\ddot r=-\dfrac{GMm}{r^2}"]),
        ("Integral de energía",
         r"Como $\mathcal L$ no depende explícitamente del tiempo, se conserva la energía (teorema de Noether para traslaciones temporales). Esa integral primera reduce la ecuación de segundo orden a una de primer orden, donde el signo de $E$ decide el destino: ligado, parabólico o libre. Igualando con la energía en el lanzamiento, $E=\tfrac12mv_0^2-GMm/R$.",
         [r"E=\tfrac12 m\dot r^{2}-\dfrac{GMm}{r}=\tfrac12 mv_0^2-\dfrac{GMm}{R}",
          r"\dot r^{2}=v_0^2-2GM\Big(\dfrac1R-\dfrac1r\Big)"]),
        ("Altura máxima",
         r"El punto de retorno se alcanza cuando $\dot r=0$, lo que da $r_{max}$ explícito. La expresión solo tiene sentido físico si el denominador es positivo, es decir si $v_0<v_{esc}$; en el límite $v_0\to v_{esc}$ la distancia máxima diverge. Con $\beta=v_0^2R/(2GM)=(v_0/v_{esc})^2$ se obtiene una forma adimensional que muestra que la altura depende únicamente de la fracción de la velocidad de escape.",
         [r"\dfrac1{r_{max}}=\dfrac1R-\dfrac{v_0^2}{2GM}\;\Rightarrow\;r_{max}=\dfrac{R}{1-\beta},\qquad \beta=\dfrac{v_0^2R}{2GM}",
          r"v_0\ll v_{esc}:\quad r_{max}-R\simeq\dfrac{v_0^2}{2g}\ \ (\text{caso de gravedad uniforme})"]),
        ("Velocidad de escape y tipos de trayectoria",
         r"El caso límite es $E=0$: el objeto llega al infinito con velocidad nula. Resulta $v_{esc}=\sqrt{2GM/R}=\sqrt2\,v_c$, donde $v_c=\sqrt{GM/R}$ es la velocidad circular rasante. Para $E=0$ la ecuación $\dot r^2=2GM/r$ integra a $r\propto t^{2/3}$, y para $E>0$ la rapidez asintótica es $v_\infty=\sqrt{v_0^2-v_{esc}^2}$. La velocidad de escape no depende de la dirección del lanzamiento ni de la masa del proyectil.",
         [r"E=0\;\Rightarrow\;v_{esc}=\sqrt{\dfrac{2GM}{R}}=\sqrt{2gR}",
          r"r(t)=\Big(\tfrac92\,GM\Big)^{1/3}t^{2/3},\qquad v_\infty=\sqrt{v_0^2-v_{esc}^2}\ \ (E>0)"]),
        ("Tiempo de vuelo y límites",
         r"Para $E<0$ la integración exacta es paramétrica: con $r=r_{max}\cos^2\eta$ el tiempo de subida resulta una función de $\eta$, lo que evita la singularidad en el punto de retorno. Cuando $\beta\to1$ el tiempo de ascenso diverge, y cuando $\beta\to0$ se recupera el tiro vertical clásico. Para lanzamientos no radiales el criterio de escape es el mismo, pues solo depende de la energía; lo que cambia es la forma de la trayectoria.",
         [r"t(\eta)=\sqrt{\dfrac{r_{max}^3}{2GM}}\Big(\eta+\sin\eta\cos\eta\Big),\qquad r=r_{max}\cos^2\eta"]),
        ("Resultado general",
         r"La distancia máxima de un lanzamiento radial subcrítico y la velocidad de escape se expresan con $GM$, $R$ y $v_0$. Las dimensiones son correctas: $GM/R$ tiene unidades de velocidad al cuadrado.",
         [r"\boxed{\;r_{max}=\dfrac{R}{1-\dfrac{v_0^2R}{2GM}},\qquad v_{esc}=\sqrt{\dfrac{2GM}{R}}\;}",
          r"\dfrac{r_{max}}{R}=\dfrac{1}{1-(v_0/v_{esc})^2}"]),
        ("Particularización al problema",
         r"Con $G=6.674\times10^{-11}$, $M=5.972\times10^{24}$ kg y $R=6.371\times10^6$ m se obtiene la velocidad de escape terrestre y la distancia máxima para $v_0=8$ y 10 km/s.",
         [rf"GM={sci(mu,4)}\ \mathrm{{m^3/s^2}},\qquad v_{{esc}}=\sqrt{{\dfrac{{2({sci(mu,4)})}}{{{sci(R,4)}}}}}={fx(ve/1000,3)}\ \mathrm{{km/s}}",
          rf"v_0=8\ \mathrm{{km/s}}:\ \beta=\left(\dfrac{{8}}{{{fx(ve/1000,3)}}}\right)^2={fx(v0**2*R/(2*mu),4)},\quad \dfrac{{r_{{max}}}}{{R}}={fx(rm/R,3)}",
          rf"v_0=10\ \mathrm{{km/s}}:\ \beta={fx(v1**2*R/(2*mu),4)},\quad \dfrac{{r_{{max}}}}{{R}}={fx(rm1/R,3)}"]),
        ("Errores comunes y límites de validez",
         r"Es habitual usar $g$ constante a gran altura: con $v_0=8$ km/s ese modelo daría una altura de más de 3000 km sobre la superficie en vez de la correcta, porque $g$ decrece como $1/r^2$. También se confunde la velocidad de escape con la orbital; la de escape es $\sqrt2$ veces mayor. El modelo desprecia la atmósfera, que frena el cohete, la rotación terrestre, que ayuda hacia el este, y la atracción del Sol y la Luna; en escapes del sistema solar hay que considerar también la energía respecto del Sol.",
         []),
    ]


# ============================================================================ PressureDepth
def _pressure() -> list:
    P0, rho, g = c16.P0, c16.RHO_WATER, c16.G
    M, Rg, Tt = c16.M_AIR, c16.R_GAS, c16.T_AIR
    hh = Rg * Tt / (M * g)
    pw = c16.water_pressure(100.0)
    p5 = float(c16.atmosphere_analytic(5000.0))
    pc = float(c16.atmosphere_constant_density(5000.0))
    rho0 = P0 * M / (Rg * Tt)
    return [
        ("Qué se estudia y por qué importa",
         r"La presión en un fluido en reposo aumenta con la profundidad, y ese hecho explica el diseño de presas, submarinos, buzos, barómetros y aviones presurizados. En la atmósfera la presión disminuye con la altura y condiciona la respiración y el punto de ebullición. El problema muestra cómo el mismo principio de equilibrio da leyes distintas según la densidad del fluido sea constante (líquido) o dependa de la presión (gas).",
         []),
        ("Magnitudes y convenciones",
         r"La presión $p$ es fuerza por unidad de área, en pascales ($1\ \mathrm{Pa}=1\ \mathrm{N/m^2}$); la atmósfera estándar vale $101\,325$ Pa. La densidad $\rho$ se da en kg/m³ (1000 para el agua, 1.2 para el aire a nivel del mar). Se mide la profundidad $h$ hacia abajo o la altura $y$ hacia arriba; con $M$ masa molar, $R$ constante de los gases y $T$ la temperatura en kelvin. La presión absoluta incluye la atmosférica; la manométrica es $p-p_0$.",
         [r"[p]=\mathrm{Pa}=\mathrm{kg\,m^{-1}\,s^{-2}},\qquad R=8.314\ \mathrm{J/(mol\,K)}"]),
        ("Intuición y analogías",
         r"Cada capa de fluido sostiene el peso de todas las capas superiores, igual que el estante inferior de una pila soporta todo lo de encima. Por eso la presión crece con la profundidad, y en un gas lo hace de forma no lineal porque las capas más bajas están más comprimidas y pesan más. Es una cadena de pequeños equilibrios, cada uno entre la diferencia de presión y el peso de una rebanada fina.",
         []),
        ("Planteamiento: equilibrio de un elemento de fluido",
         r"Sea un elemento de fluido en reposo sometido a la presión sobre sus caras y a una fuerza de volumen por unidad de masa $\vec g$. El equilibrio de fuerzas sobre un volumen finito, usando el teorema de la divergencia $\oint p\,d\vec S=\int\nabla p\,dV$, se cumple para cualquier volumen solo si se cumple punto a punto. Resulta la ecuación fundamental de la estática de fluidos, de la cual se concluye que $\rho\vec g$ debe ser un gradiente.",
         [r"-\oint p\,d\vec S+\int\rho\vec g\,dV=\vec 0\;\Rightarrow\;\nabla p=\rho\,\vec g",
          r"\nabla\times(\rho\vec g)=\vec0"]),
        ("Consecuencias: presión solo función de la altura, Pascal",
         r"En un campo gravitatorio uniforme $\vec g=-g\hat z$, el rotacional exige que $\rho$ dependa solo de $z$ (superficies isobaras horizontales), así que $p=p(z)$. Esto implica que la presión es igual en todos los puntos a la misma altura de un fluido continuo, y que un aumento de presión aplicado en un punto se transmite íntegramente (principio de Pascal). De aquí que las vasijas conectadas alcancen la misma altura, independientemente de su forma.",
         [r"\dfrac{dp}{dz}=-\rho(z)\,g,\qquad p(z_1)-p(z_2)=g\int_{z_1}^{z_2}\rho\,dz"]),
        ("Fluido incompresible",
         r"Para un líquido la densidad es prácticamente constante, y la integral es inmediata. La presión absoluta a profundidad $h$ bajo la superficie libre es la presión atmosférica sumada al peso de la columna por unidad de área. Esto explica la paradoja hidrostática: la presión depende de la altura y no de la cantidad de líquido ni de la forma del recipiente.",
         [r"p(h)=p_0+\rho\,g\,h,\qquad p_{man}=\rho g h"]),
        ("Gas ideal isotermo: atmósfera exponencial",
         r"Para un gas ideal, $\rho=pM/RT$; con temperatura constante la ecuación se vuelve lineal homogénea en $p$ y se integra con la condición $p(0)=p_0$. La altura de escala $H=RT/Mg$ es la distancia en que la presión se reduce en el factor $e$. Si $h\ll H$ el desarrollo lineal recupera el modelo de densidad constante. Una atmósfera adiabática o con gradiente de temperatura da otras leyes, de la forma $p\propto(1-\alpha h)^n$.",
         [r"\dfrac{dp}{dh}=-\dfrac{Mg}{RT}\,p\;\Rightarrow\;p(h)=p_0\,e^{-h/H},\qquad H=\dfrac{RT}{Mg}",
          r"h\ll H:\quad p\simeq p_0\Big(1-\dfrac hH\Big)=p_0-\rho_0gh"]),
        ("Resultado general",
         r"En un líquido incompresible la presión crece linealmente con la profundidad con pendiente $\rho g$; en un gas ideal isotermo decae exponencialmente con escala $H$. Ambas leyes son casos de $dp/dh=-\rho(p)g$. Las unidades son correctas: $\rho g h$ equivale a kg m⁻³ m s⁻² m = Pa, y $h/H$ es adimensional.",
         [r"\boxed{\;p_{liq}(h)=p_0+\rho g h,\qquad p_{gas}(h)=p_0\,\exp\!\Big(-\dfrac{Mgh}{RT}\Big)\;}"]),
        ("Particularización al problema",
         r"Con $p_0=101\,325$ Pa, $\rho=1000$ kg/m³, $g=9.81$ m/s², $M=0.02897$ kg/mol, $R=8.314$ J/(mol K) y $T=288.15$ K se obtienen la presión a 100 m de profundidad y a 5000 m de altura.",
         [rf"p(100\ \mathrm{{m}})={sci(P0,6)}+({fx(rho,0)})({fx(g,2)})(100)={fx(pw/1e6,3)}\ \mathrm{{MPa}}",
          rf"H=\dfrac{{({fx(Rg,3)})({fx(Tt,2)})}}{{({fx(M,5)})({fx(g,2)})}}={fx(hh,0)}\ \mathrm{{m}}",
          rf"p(5000\ \mathrm{{m}})={sci(P0,6)}\,e^{{-5000/{fx(hh,0)}}}={fx(p5/1000,2)}\ \mathrm{{kPa}}",
          rf"\text{{Con }}\rho_0={fx(rho0,4)}\ \mathrm{{kg/m^3}}\text{{ constante: }}p={fx(pc/1000,2)}\ \mathrm{{kPa}}\ \text{{(incorrecto)}}"]),
        ("Errores comunes y límites de validez",
         r"Se confunde la presión absoluta con la manométrica, o se aplica $p_0+\rho gh$ a la atmósfera suponiendo densidad constante, lo que a 5000 m da un valor muy distinto. Otro error es pensar que la presión depende del volumen del recipiente. El modelo exponencial supone temperatura uniforme y gas ideal; en la atmósfera real la temperatura baja con la altura (troposfera) y la presión se aparta de la exponencial. Para el agua, a grandes profundidades hay compresibilidad (módulo de volumen) y la densidad aumenta ligeramente.",
         []),
    ]


# ============================================================================ FloatingCylinder
def _cylinder() -> list:
    rho_o, rho_f, A, Lc = c16.RHO_OBJ, c16.RHO_WATER, c16.AREA, c16.LEN
    g = c16.G
    m = c16.MASS
    h0 = c16.submerged_depth_analytic()
    w = c16.omega_analytic()
    T = 2 * np.pi / w
    bm = A / (4 * np.pi * h0)
    return [
        ("Qué se estudia y por qué importa",
         r"Un cuerpo sumergido en un fluido recibe un empuje igual al peso del fluido desalojado (principio de Arquímedes). De él dependen el diseño de barcos, boyas, submarinos y globos, y la medida de densidades con hidrómetros. Además, un cuerpo que flota y se desplaza ligeramente de su equilibrio oscila: es un oscilador armónico cuyo periodo se calcula con la misma teoría de pequeñas oscilaciones de la mecánica analítica.",
         []),
        ("Magnitudes y convenciones",
         r"El cilindro tiene masa $m$, densidad $\rho$, longitud $L$ y sección $A$; el fluido, densidad $\rho_f$. La profundidad sumergida $h$ (m) se mide hacia abajo desde la superficie libre hasta la base del cilindro. El empuje es $B=\rho_fgAh$ (N) y la fracción sumergida en equilibrio es $\rho/\rho_f$. Para madera en agua $\rho/\rho_f$ está entre 0.4 y 0.8.",
         [r"m=\rho AL,\qquad B=\rho_f\,g\,A\,h,\qquad [B]=\mathrm{N}"]),
        ("Intuición y analogías",
         r"El cilindro se hunde hasta que el peso del agua desalojada iguala al suyo. Si se empuja un poco más, el empuje aumenta y lo devuelve hacia arriba; si se levanta, el empuje disminuye y lo hunde de nuevo. Esa fuerza restauradora proporcional al desplazamiento es la de un resorte, cuya rigidez es el peso de la columna de agua de sección $A$ y altura unitaria.",
         []),
        ("Planteamiento: lagrangiana con energía potencial de flotación",
         r"Se toma como coordenada generalizada la profundidad sumergida $h$ y se supone que el cilindro se mueve verticalmente, con paredes verticales y sin ser cubierto por completo. La energía potencial total tiene dos partes: la gravitatoria, que baja al descender, y la del fluido desplazado, que sube al elevar el nivel de agua desalojada. El empuje es la derivada de esa energía de desplazamiento. Se desprecian la masa añadida y la viscosidad.",
         [r"T=\tfrac12 m\dot h^{2},\qquad V(h)=\tfrac12\rho_f\,g\,A\,h^{2}-m\,g\,h",
          r"\mathcal L=\tfrac12 m\dot h^{2}-\tfrac12\rho_fgAh^{2}+mgh"]),
        ("Ecuación de movimiento y equilibrio",
         r"La ecuación de Euler-Lagrange reproduce la segunda ley de Newton: el peso hacia abajo menos el empuje hacia arriba. El equilibrio es el punto estacionario de $V$, donde empuje y peso se igualan; de ahí sale la profundidad sumergida, proporcional a la razón de densidades. La condición $h_0\le L$ exige $\rho\le\rho_f$ para que el cilindro flote.",
         [r"m\ddot h=mg-\rho_fgAh",
          r"V'(h_0)=0\;\Rightarrow\;h_0=\dfrac{m}{\rho_fA}=\dfrac{\rho}{\rho_f}\,L"]),
        ("Estabilidad y pequeñas oscilaciones",
         r"El equilibrio es estable si $V''(h_0)>0$; aquí $V''=\rho_fgA>0$. Escribiendo $h=h_0+\xi$, la ecuación se reduce a la de un oscilador armónico con $\omega^2=V''/m$. Como el empuje es exactamente lineal en $h$ para paredes verticales, la oscilación es armónica para cualquier amplitud que no saque el cilindro del agua ni lo hunda por completo, es decir $|\xi|<\min(h_0,L-h_0)$. Se puede escribir $\omega^2=g/h_0$, análoga a la de un péndulo de longitud $h_0$.",
         [r"\ddot\xi+\omega^{2}\xi=0,\qquad \omega^{2}=\dfrac{V''(h_0)}{m}=\dfrac{\rho_fgA}{m}=\dfrac{g}{h_0}",
          r"T=2\pi\sqrt{\dfrac{m}{\rho_fgA}}=2\pi\sqrt{\dfrac{\rho L}{\rho_fg}}"]),
        ("Generalizaciones: amortiguamiento y otras formas",
         r"Si el agua opone una resistencia proporcional a la velocidad, la ecuación se convierte en la del oscilador amortiguado y la frecuencia disminuye. Para un cuerpo con sección variable la rigidez es $\rho_fgA_{wl}(h_0)$, donde $A_{wl}$ es el área de la línea de flotación en el equilibrio, y la oscilación deja de ser exactamente armónica. Para la estabilidad ante inclinaciones el criterio es la altura metacéntrica: el equilibrio vertical es estable solo si el radio metacéntrico $I/V$ supera la distancia del centro de gravedad al de empuje.",
         [r"\ddot\xi+2\gamma\dot\xi+\omega^{2}\xi=0,\qquad \omega_d=\sqrt{\omega^2-\gamma^2}",
          r"\overline{BM}=\dfrac{I_{wl}}{V_{sum}}>\overline{BG}\quad(\text{estabilidad al balanceo})"]),
        ("Resultado general",
         r"La profundidad de equilibrio y el periodo dependen solo de las densidades, de la longitud y de $g$; el área de la sección se cancela. Un cilindro más ligero se sumerge menos y oscila más lentamente. Las unidades son correctas: $\sqrt{\rho L/(\rho_fg)}$ tiene unidades de s.",
         [r"\boxed{\;h_0=\dfrac{\rho}{\rho_f}\,L,\qquad \omega=\sqrt{\dfrac{\rho_f gA}{m}}=\sqrt{\dfrac{g}{h_0}},\qquad T=2\pi\sqrt{\dfrac{\rho L}{\rho_f g}}\;}"]),
        ("Particularización al problema",
         r"Con $\rho=600$ kg/m³, $L=0.50$ m, $A=0.010$ m² y $\rho_f=1000$ kg/m³ se calcula la masa, la profundidad de equilibrio, la frecuencia y el periodo.",
         [rf"m=\rho AL=({fx(rho_o,0)})({fx(A,3)})({fx(Lc,2)})={fx(m,2)}\ \mathrm{{kg}},\qquad h_0=\dfrac{{{fx(rho_o,0)}}}{{{fx(rho_f,0)}}}({fx(Lc,2)})={fx(h0,3)}\ \mathrm{{m}}",
          rf"\omega=\sqrt{{\dfrac{{({fx(rho_f,0)})({fx(g,2)})({fx(A,3)})}}{{{fx(m,2)}}}}}={fx(w,3)}\ \mathrm{{rad/s}}=\sqrt{{\dfrac{{{fx(g,2)}}}{{{fx(h0,3)}}}}}",
          rf"T=\dfrac{{2\pi}}{{\omega}}={fx(T,3)}\ \mathrm{{s}}"]),
        ("Errores comunes y límites de validez",
         r"Un error es olvidar que solo el volumen sumergido produce empuje y usar el volumen total del cilindro. Otro es creer que el periodo depende del área o del amplitud; mientras el cilindro siga parcialmente sumergido, no depende de ellos. El modelo ignora la masa de agua arrastrada (masa añadida), que alarga el periodo real, y la viscosidad y las olas, que lo amortiguan. Además, un cilindro delgado y largo suele ser inestable en posición vertical frente al balanceo: para estas dimensiones, suponiendo sección circular, el radio metacéntrico es "
         + f"{bm*100:.2f}" + r" cm, menor que la distancia $\overline{BG}$ de " + f"{(Lc - h0)/2*100:.1f}" + r" cm, y en la práctica volcaría; la oscilación vertical descrita es válida solo si se impide ese giro.",
         []),
    ]


DETAIL = {
    "SkaterSpin": dict(concept=_skater()),
    "GyroscopePrecession": dict(concept=_gyro()),
    "LadderFriction": dict(concept=_ladder()),
    "BeamReactions": dict(concept=_beam()),
    "KeplerOrbit": dict(concept=_kepler()),
    "EscapeVelocity": dict(concept=_escape()),
    "PressureDepth": dict(concept=_pressure()),
    "FloatingCylinder": dict(concept=_cylinder()),
}
