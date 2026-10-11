# Fichas: Aprendizaje por refuerzo

## agente-entorno-y-recompensa
frase: Un agente actúa en un entorno, observa el resultado y recibe recompensas; aprende a actuar para maximizarlas.
desc: En el aprendizaje por refuerzo no hay respuestas correctas: el agente observa el estado del entorno, elige una acción, el entorno cambia de estado y devuelve una recompensa. El objetivo es maximizar la recompensa acumulada a largo plazo, lo que puede exigir sacrificar recompensas inmediatas. Se formaliza como un proceso de decisión de Markov: estados, acciones, transiciones, recompensas y un factor de descuento.
ej: Un agente que aprende a jugar al ajedrez recibe +1 al ganar, −1 al perder y 0 en cualquier otra jugada, y debe descubrir qué movimientos llevan a la victoria.
py: gymnasium.make ; gymnasium.Env
rel: politica-y-funcion-de-valor, exploracion-vs-explotacion, paradigmas-de-aprendizaje, multi-armed-bandits
en: agent, environment & reward

## politica-y-funcion-de-valor
frase: La política dice qué hacer en cada estado; la función de valor, cuánta recompensa futura cabe esperar.
desc: La política π(a|s) es la estrategia del agente: la probabilidad de elegir cada acción en cada estado. La función de valor V(s) estima la recompensa acumulada esperada a partir de un estado siguiendo la política, y la función Q(s, a), la de tomar una acción concreta y seguir la política después. Los métodos basados en valor aprenden Q y actúan de forma voraz respecto a ella; los de gradiente de política optimizan π directamente; los actor-crítico combinan ambas ideas.
ej: En un laberinto, V es alto en las casillas cercanas a la salida, y la política consiste en moverse siempre hacia la casilla vecina de mayor valor.
rel: agente-entorno-y-recompensa, ppo, exploracion-vs-explotacion, regret
en: policy & value function

## exploracion-vs-explotacion
frase: El dilema entre aprovechar lo que ya sabes que funciona y probar opciones nuevas que podrían ser mejores.
desc: Explotar maximiza la recompensa inmediata con el conocimiento actual; explorar sacrifica algo de recompensa para obtener información que puede mejorar las decisiones futuras. Si se explota demasiado pronto, el agente se queda con una opción mediocre; si se explora demasiado, desperdicia oportunidades. Estrategias como ε-greedy, UCB o el muestreo de Thompson resuelven el dilema de formas distintas.
ej: Ir siempre a tu restaurante favorito (explotar) o probar uno nuevo que podría convertirse en tu nuevo favorito (explorar).
rel: multi-armed-bandits, epsilon-greedy, ucb, muestreo-de-thompson, arranque-en-frio
en: exploration vs exploitation

## multi-armed-bandits
frase: Elegir repetidamente entre varias opciones de recompensa incierta, aprendiendo cuál es la mejor mientras se decide.
desc: Es el problema de aprendizaje por refuerzo más simple: no hay estados, solo K acciones (los brazos de una máquina tragaperras) cuyas recompensas siguen distribuciones desconocidas. En cada ronda se elige un brazo, se observa su recompensa y se actualiza lo que se sabe. Se usa como alternativa adaptativa a los tests A/B, porque desvía el tráfico hacia las mejores opciones mientras aprende, y en recomendación, publicidad y ensayos clínicos.
cuando: Para elegir en línea entre varias alternativas (versiones de una web, anuncios, precios) minimizando lo que se pierde mientras se aprende.
ej: Una web prueba cinco titulares para una noticia y, en pocas horas, el bandit muestra casi siempre el que más clics consigue.
py: mabwiser.mab.MAB
rel: exploracion-vs-explotacion, epsilon-greedy, ucb, muestreo-de-thompson, regret
en: multi-armed bandits

## epsilon-greedy
frase: Elegir casi siempre la mejor opción conocida y, con una pequeña probabilidad ε, una al azar.
desc: La estrategia voraz (greedy) elige siempre el brazo con la mejor recompensa media estimada, y puede quedarse atascada en uno mediocre si al principio tuvo suerte. ε-greedy explora con probabilidad ε eligiendo un brazo al azar, lo que garantiza que todos se sigan probando. Es simple y eficaz, y es habitual reducir ε con el tiempo para explorar mucho al principio y poco después.
cuando: Como estrategia de referencia sencilla en bandits y en aprendizaje por refuerzo.
ej: Con ε = 0,1, nueve de cada diez visitantes ven el titular que mejor ha funcionado hasta ahora y uno de cada diez, un titular al azar.
hp: epsilon: probabilidad de explorar con una acción al azar
py: mabwiser.mab.LearningPolicy.EpsilonGreedy
rel: multi-armed-bandits, exploracion-vs-explotacion, valores-iniciales-optimistas, ucb
en: greedy & ε-greedy

## valores-iniciales-optimistas
frase: Empezar suponiendo que todas las opciones son excelentes, para que el agente se vea obligado a probarlas todas.
desc: Si las estimaciones iniciales de recompensa son muy altas, cualquier opción probada decepciona y su estimación baja, así que el agente voraz pasa a probar las que todavía conservan su valor optimista. El resultado es una exploración sistemática al principio, sin aleatoriedad. Funciona bien en problemas estacionarios, pero no ayuda si las recompensas cambian con el tiempo, porque la exploración ocurre solo al inicio.
cuando: En bandits estacionarios con pocas opciones, como forma sencilla de forzar la exploración inicial.
ej: Con recompensas entre 0 y 1, iniciar todas las estimaciones en 5 hace que el agente pruebe cada opción varias veces antes de decidirse.
py: `Q = np.full(n_brazos, 5.0)`
rel: epsilon-greedy, exploracion-vs-explotacion, multi-armed-bandits, prior-verosimilitud-y-posterior
en: optimistic initial values

## ucb
frase: Elegir la opción con la estimación más optimista: su media más un margen que crece cuanto menos se ha probado.
desc: Upper Confidence Bound suma a la recompensa media de cada brazo un término de incertidumbre proporcional a √(ln t / n), donde t es el número total de rondas y n, las veces que se ha probado ese brazo. Los brazos poco explorados tienen un margen grande y acaban probándose; a medida que se prueban, el margen se reduce. Es el principio de "optimismo ante la incertidumbre" y tiene garantías teóricas de regret logarítmico.
cuando: En bandits con recompensas estacionarias cuando se quiere una exploración dirigida, sin aleatoriedad.
ej: Un titular con una tasa de clics media del 4 % probado 10 veces puede elegirse antes que otro del 5 % probado 1.000 veces, porque su incertidumbre es mucho mayor.
hp: alpha: peso del término de exploración
py: mabwiser.mab.LearningPolicy.UCB1
rel: multi-armed-bandits, exploracion-vs-explotacion, intervalo-de-confianza, regret
en: UCB (upper confidence bound)

## muestreo-de-thompson
frase: Elegir cada opción con la probabilidad de que sea la mejor, muestreando de la distribución posterior de su recompensa.
desc: Mantiene una distribución de creencia bayesiana sobre la recompensa de cada brazo, por ejemplo una Beta para tasas de éxito. En cada ronda toma una muestra de cada distribución, elige el brazo con la muestra más alta y actualiza su posterior con el resultado. Explora de forma natural las opciones inciertas y converge a la mejor; en la práctica suele superar a ε-greedy y a UCB.
cuando: Como estrategia por defecto en bandits, sobre todo con recompensas binarias como clics o conversiones.
ej: Con dos versiones de un botón, una con 30 clics en 1.000 visitas y otra con 5 en 100, el muestreo de Thompson sigue mostrando a veces la segunda porque su tasa real aún es muy incierta.
py: mabwiser.mab.LearningPolicy.ThompsonSampling ; numpy.random.Generator.beta
rel: prior-verosimilitud-y-posterior, multi-armed-bandits, ucb, teorema-de-bayes
en: Thompson sampling

## bandits-contextuales
frase: Bandits que eligen la acción teniendo en cuenta el contexto de cada decisión, como las características del usuario.
desc: En lugar de buscar la mejor opción para todos, aprenden qué opción es mejor en cada contexto. LinUCB modela la recompensa de cada brazo como una función lineal del contexto y aplica UCB sobre esa estimación. Están a medio camino entre los bandits simples y el aprendizaje por refuerzo completo: hay contexto, pero las acciones no cambian el estado futuro.
cuando: Para personalizar decisiones en línea (qué noticia, oferta o diseño mostrar) según las características de cada usuario o situación.
ej: Una app de noticias elige el titular de portada para cada lector según su ubicación, la hora y los temas que lee, aprendiendo de cada clic.
hp: alpha: peso del término de exploración en LinUCB | l2_lambda: regularización de los modelos lineales
py: mabwiser.mab.LearningPolicy.LinUCB
rel: multi-armed-bandits, ucb, regresion-lineal, recomendacion-con-deep-learning
en: contextual bandits (LinUCB)

## regret
frase: Lo que se pierde por no haber elegido siempre la mejor opción: la medida de calidad de una estrategia de bandit.
desc: El regret acumulado es la diferencia entre la recompensa que se habría obtenido eligiendo siempre la mejor acción y la que realmente se obtuvo. Una buena estrategia tiene un regret que crece cada vez más despacio, como el logaritmo del número de rondas, porque aprende pronto cuál es la mejor opción. Sirve para comparar estrategias en simulación y para cuantificar el coste de aprender.
ej: Si el mejor anuncio da 5 clics por cada 100 visitas y la estrategia ha obtenido 4,2 de media en 10.000 visitas, el regret acumulado es de 80 clics.
rel: multi-armed-bandits, exploracion-vs-explotacion, ucb, muestreo-de-thompson
en: regret
