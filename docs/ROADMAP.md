# Hoja de ruta

Cobertura de *Física*, vol. 1 y 2, de Resnick, Halliday y Krane, según el índice de referencia del proyecto.
La numeración exacta de capítulos varía entre ediciones; aquí se identifica cada tema por su título.

## Volumen 1: Mecánica, ondas y termodinámica

Avance: 5 de 25 temas.

### Mecánica clásica

| Tema | Conceptos | Estado |
| --- | --- | --- |
| Medición | Sistemas de unidades, patrones de tiempo, longitud y masa | Publicado |
| Vectores | Suma, componentes, productos escalar y vectorial | Publicado |
| Movimiento en una dimensión | Posición, velocidad, aceleración y caída libre | Publicado |
| Movimiento en dos y tres dimensiones | Tiro parabólico y movimiento circular | Publicado |
| Fuerza y leyes de Newton | Leyes del movimiento y aplicaciones | Publicado |
| Dinámica de la partícula | Fricción, fuerza centrípeta, dinámica avanzada | Planificado |
| Trabajo y energía | Trabajo, energía cinética, teorema trabajo-energía | Planificado |
| Conservación de la energía | Energía potencial, fuerzas conservativas y no conservativas | Planificado |
| Sistemas de partículas | Centro de masa, ímpetu lineal y su conservación | Planificado |
| Colisiones | Impulso, colisiones elásticas e inelásticas | Planificado |
| Cinemática rotacional | Variables rotacionales, velocidad y aceleración angular | Planificado |
| Dinámica rotacional | Torque, momento de inercia, segunda ley para la rotación | Planificado |
| Momento angular | Conservación del momento angular, giroscopios | Planificado |
| Equilibrio de cuerpos rígidos | Condiciones de equilibrio, elasticidad | Planificado |
| Gravitación | Gravitación universal, leyes de Kepler, movimiento planetario | Planificado |
| Estática de fluidos | Presión, principios de Pascal y Arquímedes | Planificado |
| Dinámica de fluidos | Continuidad, ecuación de Bernoulli | Planificado |

### Oscilaciones y ondas

| Tema | Conceptos | Estado |
| --- | --- | --- |
| Oscilaciones | Movimiento armónico simple, péndulos, amortiguamiento y forzamiento | Planificado |
| Movimiento ondulatorio | Ondas transversales y longitudinales, velocidad y potencia | Planificado |
| Ondas sonoras | Propagación, efecto Doppler, interferencia, resonancia | Planificado |
| Teoría especial de la relatividad | Cinemática y dinámica relativistas | Planificado |

### Termodinámica

| Tema | Conceptos | Estado |
| --- | --- | --- |
| Temperatura y ley cero | Escalas de temperatura, expansión térmica | Planificado |
| Primera ley de la termodinámica | Calor, trabajo, energía interna | Planificado |
| Teoría cinética de los gases | Presión del gas ideal, temperatura, distribución de velocidades | Planificado |
| Segunda ley y entropía | Máquinas térmicas, ciclo de Carnot, entropía | Planificado |

## Volumen 2: Electromagnetismo, óptica y física cuántica

Avance: 0 de 25 temas.

### Electromagnetismo

| Tema | Conceptos | Estado |
| --- | --- | --- |
| Carga eléctrica y ley de Coulomb | Carga, aislantes y conductores, fuerza electrostática | Planificado |
| El campo eléctrico | Líneas de campo, distribuciones continuas | Planificado |
| Ley de Gauss | Flujo eléctrico, simetrías, conductores en equilibrio | Planificado |
| Potencial eléctrico | Energía potencial, diferencia de potencial, cálculo del campo | Planificado |
| Capacitancia y dieléctricos | Capacitores, serie y paralelo, energía almacenada | Planificado |
| Corriente y resistencia | Ley de Ohm, modelos de conducción | Planificado |
| Circuitos de corriente directa | Leyes de Kirchhoff, circuitos RC, fem | Planificado |
| El campo magnético | Fuerza magnética, efecto Hall, movimiento de cargas | Planificado |
| Campo magnético de una corriente | Ley de Biot-Savart, ley de Ampère | Planificado |
| Ley de inducción de Faraday | Flujo magnético, ley de Lenz, fem inducida | Planificado |
| Inductancia | Autoinductancia, circuitos RL, oscilaciones LC | Planificado |
| Propiedades magnéticas de la materia | Dipolos, dia, para y ferromagnetismo; ley de Gauss del magnetismo | Planificado |
| Circuitos de corriente alterna | Fasores, circuitos RLC, resonancia, transformadores | Planificado |
| Ecuaciones de Maxwell y ondas electromagnéticas | Corriente de desplazamiento, espectro, vector de Poynting | Planificado |

### Óptica

| Tema | Conceptos | Estado |
| --- | --- | --- |
| Naturaleza y propagación de la luz | Velocidad de la luz, reflexión, refracción, dispersión | Planificado |
| Óptica geométrica: espejos y lentes | Formación de imágenes con espejos y lentes delgadas | Planificado |
| Interferencia | Dos fuentes, experimento de Young, películas delgadas, interferómetros | Planificado |
| Difracción y polarización | Difracción de Fraunhofer, rejillas, polarización | Planificado |

### Física cuántica y moderna

| Tema | Conceptos | Estado |
| --- | --- | --- |
| Fotones y naturaleza ondulatoria de la luz | Cuerpo negro, efecto fotoeléctrico, efecto Compton | Planificado |
| Ondas de materia | Hipótesis de De Broglie, principio de incertidumbre | Planificado |
| Ecuación de Schrödinger | Funciones de onda, pozos de potencial, tunelamiento | Planificado |
| Estructura atómica | Modelo de Bohr, átomo de hidrógeno, niveles de energía | Planificado |
| Conducción eléctrica en sólidos | Teoría de bandas, semiconductores, uniones p-n | Planificado |
| Física nuclear | Estructura del núcleo, radiactividad, fisión y fusión | Planificado |
| Partículas elementales y cosmología | Quarks, leptones, fuerzas fundamentales, Big Bang | Planificado |

## Criterios de entrada para cada tema nuevo

1. Índice de conceptos tomado del índice de referencia (solo títulos, no texto del libro).
2. Al menos dos escenas por capítulo, cada una con un problema original.
3. Solución analítica en forma cerrada en `physics/` y un método numérico independiente.
4. Pruebas que comparen ambas con una tolerancia justificada por el método.
5. Fila de verificación en `content/chapters.py` y revisión visual del video a 1080p.
