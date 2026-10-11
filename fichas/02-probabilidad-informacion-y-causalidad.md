# Fichas: Probabilidad, información y causalidad

## probabilidad-y-estadistica
frase: El lenguaje para describir la incertidumbre y sacar conclusiones fiables a partir de datos.
desc: La probabilidad modela fenómenos inciertos; la estadística usa datos para estimar sus parámetros y contrastar hipótesis. Casi todo modelo de ML es, en el fondo, un modelo probabilístico: predice una distribución, se entrena maximizando una verosimilitud y se evalúa teniendo en cuenta la incertidumbre.
ej: Estimar la tasa de conversión de una web a partir de 1.000 visitas y acompañarla de un intervalo de confianza.
py: scipy.stats
rel: teorema-de-bayes, maxima-verosimilitud, contraste-de-hipotesis, distribuciones-de-probabilidad
en: probability & statistics

## probabilidad-frecuentista-vs-bayesiana
frase: Probabilidad como frecuencia a largo plazo o como grado de creencia que se actualiza con datos.
desc: Para la visión frecuentista, la probabilidad es la frecuencia con la que ocurre un suceso si se repite el experimento muchas veces, y los parámetros son valores fijos desconocidos. Para la bayesiana, es un grado de creencia: los parámetros tienen una distribución que se actualiza con los datos mediante el teorema de Bayes. La primera da p-valores e intervalos de confianza; la segunda, distribuciones posteriores e intervalos de credibilidad.
ej: Tras 7 caras en 10 lanzamientos, el frecuentista estima p = 0,7 y contrasta si la moneda es justa. El bayesiano parte de una creencia previa centrada en 0,5 y obtiene una distribución posterior para p.
rel: teorema-de-bayes, prior-verosimilitud-y-posterior, p-valor, intervalo-de-confianza
en: frequentist vs Bayesian probability

## probabilidad-condicional
frase: La probabilidad de un suceso cuando ya sabemos que ha ocurrido otro.
desc: P(A|B) es la probabilidad de A restringida a los casos en que ocurre B: P(A|B) = P(A y B) / P(B). Es la base del teorema de Bayes y de casi todo modelo predictivo, que en el fondo estima P(y|x). Confundir P(A|B) con P(B|A) es uno de los errores de razonamiento más frecuentes.
ej: La probabilidad de que un correo sea spam es del 20 %, pero entre los correos que contienen la palabra "gratis" sube al 70 %.
py: pandas.crosstab
rel: teorema-de-bayes, independencia-estadistica, dependencia-condicional, red-bayesiana
en: conditional probability

## teorema-de-bayes
frase: Regla para actualizar una creencia con nueva evidencia: el posterior es proporcional a verosimilitud por prior.
desc: Relaciona las dos probabilidades condicionales: P(A|B) = P(B|A)·P(A) / P(B). Permite invertir el razonamiento, de la probabilidad de los síntomas dada la enfermedad a la de la enfermedad dados los síntomas. Es la base de la inferencia bayesiana y del clasificador Naive Bayes, que lo aplica suponiendo que las variables son independientes dentro de cada clase.
cuando: Para actualizar una estimación con evidencia nueva o invertir una probabilidad condicional; Naive Bayes, como clasificador rápido de texto.
ej: Una prueba que acierta el 99 % de las veces, con enfermos y con sanos, para una enfermedad que tiene 1 de cada 1.000 personas: un positivo solo implica un 9 % de probabilidad de estar enfermo.
py: sklearn.naive_bayes.GaussianNB ; sklearn.naive_bayes.MultinomialNB
rel: probabilidad-condicional, prior-verosimilitud-y-posterior, probabilidad-frecuentista-vs-bayesiana, red-bayesiana
en: Bayes' theorem

## prior-verosimilitud-y-posterior
frase: Las tres piezas de la inferencia bayesiana: creencia previa, evidencia de los datos y creencia actualizada.
desc: El prior P(θ) recoge lo que se cree sobre un parámetro antes de ver datos. La verosimilitud P(datos|θ) mide lo compatibles que son los datos con cada valor del parámetro. El posterior P(θ|datos) combina ambos y es el resultado de la inferencia. Con pocos datos pesa más el prior; con muchos, domina la verosimilitud.
ej: Para la tasa de clics de un anuncio nuevo, un prior Beta(2, 98), centrado en el 2 % habitual, y 20 clics en 500 impresiones (4 %) dan un posterior Beta(22, 578), centrado en el 3,7 %.
py: scipy.stats.beta
rel: teorema-de-bayes, maxima-verosimilitud, regresion-lineal-bayesiana, muestreo-de-thompson
en: prior, likelihood & posterior

## distribuciones-de-probabilidad
frase: Modelos matemáticos de cómo se reparten los valores posibles de una variable aleatoria.
desc: Describen qué valores puede tomar una variable y con qué probabilidad. Las discretas, como la binomial o la de Poisson, asignan probabilidad a valores contables. Las continuas, como la normal o la exponencial, usan una función de densidad. Elegir la distribución adecuada para los datos o los errores es el primer paso de muchos modelos.
ej: El número de llamadas que recibe un centro de atención en una hora suele seguir una Poisson; la altura de los adultos de un país, aproximadamente una normal.
py: scipy.stats.norm ; scipy.stats.binom ; scipy.stats.poisson
rel: maxima-verosimilitud, glm, estimacion-de-densidad, kde
en: probability distributions

## maxima-verosimilitud
frase: Elegir los parámetros que hacen más probables los datos observados.
desc: Busca los valores de los parámetros que maximizan la verosimilitud, es decir, la probabilidad de haber observado los datos que tenemos. En la práctica se maximiza su logaritmo, que convierte productos en sumas. Muchas funciones de pérdida son máxima verosimilitud encubierta: el error cuadrático corresponde a errores gaussianos y la entropía cruzada, a salidas categóricas.
cuando: Para ajustar los parámetros de una distribución o de un modelo probabilístico a unos datos.
ej: Si 30 de 100 clientes renuevan su suscripción, la estimación de máxima verosimilitud de la probabilidad de renovación es 0,30.
py: scipy.stats.rv_continuous.fit ; statsmodels.base.model.GenericLikelihoodModel
rel: nll, prior-verosimilitud-y-posterior, entropia-cruzada, funciones-de-perdida
en: maximum likelihood estimation (MLE)

## nll
frase: La verosimilitud en logaritmo y con signo negativo: la pérdida que se minimiza al entrenar.
desc: Es −log P(datos|θ). Minimizarla equivale a maximizar la verosimilitud, pero es numéricamente más estable y encaja con los optimizadores, que minimizan. Penaliza mucho las predicciones seguras y equivocadas. La entropía cruzada de clasificación es la NLL de una distribución categórica, y el MSE es, salvo constantes, la NLL de una normal.
ej: Un clasificador que asigna una probabilidad de 0,01 a la clase correcta suma −ln(0,01) ≈ 4,6 a la pérdida; si le asigna 0,9, solo 0,1.
py: sklearn.metrics.log_loss ; torch.nn.GaussianNLLLoss
rel: maxima-verosimilitud, entropia-cruzada, funciones-de-perdida, perplejidad
en: negative log-likelihood (NLL)

## contraste-de-hipotesis
frase: Procedimiento para decidir si los datos aportan evidencia suficiente contra una hipótesis de partida.
desc: Se plantea una hipótesis nula (no hay efecto, no hay diferencia) y se calcula lo extraño que sería observar los datos si fuera cierta. Si resulta muy improbable, por debajo de un nivel de significación fijado de antemano (típicamente 0,05), se rechaza. Hay tests para comparar medias, proporciones, distribuciones o la independencia entre variables.
cuando: Para decidir si una diferencia observada, como la mejora de un modelo o el resultado de un test A/B, es real o puede deberse al azar.
ej: Un test A/B compara dos versiones de una web; el test t indica si la diferencia en el tiempo medio de permanencia es significativa.
py: scipy.stats.ttest_ind ; scipy.stats.mannwhitneyu
rel: p-valor, intervalo-de-confianza, test-chi-cuadrado, probabilidad-frecuentista-vs-bayesiana
en: hypothesis testing

## p-valor
frase: Probabilidad de ver un resultado al menos tan extremo como el observado si la hipótesis nula fuera cierta.
desc: Mide la compatibilidad de los datos con la hipótesis nula: un p-valor pequeño indica que los datos serían raros si no hubiera efecto. No es la probabilidad de que la hipótesis nula sea cierta, ni mide el tamaño del efecto. Con muestras enormes, efectos irrelevantes dan p-valores minúsculos, y probar muchas hipótesis a la vez produce falsos positivos si no se corrige.
ej: Un p-valor de 0,03 al comparar dos modelos significa que, si rindieran igual, una diferencia como la observada aparecería solo en un 3 % de las repeticiones.
py: scipy.stats.permutation_test
rel: contraste-de-hipotesis, intervalo-de-confianza, correlacion-espuria
en: p-value

## intervalo-de-confianza
frase: Rango de valores plausibles para un parámetro, con un nivel de confianza asociado.
desc: En lugar de un único número, da un rango que contiene el valor real con una confianza dada, por ejemplo del 95 %. Su interpretación es frecuentista: si se repitiera el muestreo muchas veces, el 95 % de los intervalos construidos así contendrían el valor real. El bootstrap permite calcularlo para casi cualquier métrica remuestreando los datos.
cuando: Para informar de una métrica o una estimación junto con su incertidumbre, sobre todo con pocos datos.
ej: "AUC de 0,87, con intervalo de confianza del 95 % entre 0,83 y 0,91" informa mucho más que el 0,87 a secas.
py: scipy.stats.bootstrap ; statsmodels.stats.proportion.proportion_confint
rel: p-valor, contraste-de-hipotesis, incertidumbre, validacion-cruzada
en: confidence interval

## test-chi-cuadrado
frase: Test para comprobar si dos variables categóricas son independientes.
desc: Compara las frecuencias observadas en una tabla de contingencia con las que se esperarían si las variables fueran independientes. Cuanto mayor es la discrepancia, mayor es el estadístico χ² y menor el p-valor. En ML también se usa para seleccionar variables categóricas o de conteo relacionadas con la clase.
cuando: Para comprobar si dos variables categóricas están relacionadas o para filtrar variables de conteo antes de clasificar.
ej: Comprobar si la tasa de abandono de clientes depende del tipo de contrato: mensual, anual o bianual.
py: scipy.stats.chi2_contingency ; sklearn.feature_selection.chi2
rel: independencia-estadistica, contraste-de-hipotesis, metodos-filtro, informacion-mutua
en: chi-squared test

## teoria-de-la-informacion
frase: Matemáticas para medir la información, la incertidumbre y lo que comparten las variables.
desc: Fundada por Claude Shannon en 1948, cuantifica la información en bits. Sus medidas clave son la entropía (incertidumbre de una variable), la información mutua (lo que una variable dice de otra) y la divergencia de Kullback-Leibler (la diferencia entre dos distribuciones). En ML aparece en las funciones de pérdida, los árboles de decisión, la selección de variables y los modelos generativos.
ej: La entropía cruzada que se minimiza al entrenar casi cualquier clasificador es una medida de teoría de la información.
py: scipy.stats.entropy
rel: entropia, informacion-mutua, divergencia-de-kullback-leibler, entropia-cruzada
en: information theory

## informacion-shannon
frase: Cuanto menos probable es un suceso, más información aporta cuando ocurre.
desc: La información, o sorpresa, de un suceso con probabilidad p es −log₂ p bits. Un suceso seguro no aporta nada; uno muy improbable aporta mucho. Con logaritmo en base 2 se mide en bits y con logaritmo natural, en nats. La media de la información de todos los sucesos posibles es la entropía.
ej: Conocer el resultado de lanzar una moneda aporta 1 bit; conocer el número que sale en un dado de 8 caras, 3 bits.
py: `-np.log2(p)`
rel: entropia, teoria-de-la-informacion, nll
en: Shannon information (self-information)

## entropia
frase: La incertidumbre media de una variable: cuánta información aporta, de media, conocer su valor.
desc: Es la información media de una variable, H(X) = −Σ p(x) log p(x). Es máxima cuando todos los valores son igual de probables y nula cuando el resultado es seguro. Los árboles de decisión la usan para elegir divisiones que reduzcan la incertidumbre sobre la clase, y la entropía cruzada se basa en ella para comparar predicciones con etiquetas.
ej: Una moneda justa tiene 1 bit de entropía; una trucada que sale cara el 90 % de las veces, solo 0,47 bits.
py: scipy.stats.entropy ; scipy.stats.differential_entropy
rel: informacion-shannon, entropia-cruzada, criterios-de-division, informacion-mutua
en: entropy

## divergencia-de-kullback-leibler
frase: Mide cuánto se aleja una distribución de probabilidad de otra de referencia.
desc: KL(P‖Q) = Σ p(x) log(p(x)/q(x)) cuantifica la información que se pierde al aproximar P con Q. Siempre es mayor o igual que cero y solo vale cero si las distribuciones son iguales, pero no es simétrica, así que no es una distancia. Aparece en la pérdida de los VAE, en la destilación de conocimiento, en t-SNE y en el ajuste de LLMs con RLHF.
ej: En un VAE, el término KL obliga a que la distribución latente de cada imagen se parezca a una normal estándar.
py: scipy.special.rel_entr ; keras.losses.KLDivergence ; torch.nn.KLDivLoss
rel: entropia, elbo, destilacion-de-conocimiento, t-sne, rlhf
en: Kullback-Leibler divergence

## informacion-mutua
frase: Cuánta información comparten dos variables: detecta cualquier dependencia, no solo la lineal.
desc: I(X;Y) mide cuánto se reduce la incertidumbre sobre una variable al conocer la otra. Vale cero solo si son independientes y, a diferencia de la correlación de Pearson, capta relaciones no lineales. Se usa para seleccionar variables, comparar agrupamientos y como objetivo en el aprendizaje de representaciones.
cuando: Para medir o clasificar por importancia la relación entre variables cuando puede ser no lineal, por ejemplo al seleccionar variables.
ej: Si y = x² con x simétrica en torno a cero, la correlación de Pearson es casi nula, pero la información mutua es alta.
py: sklearn.feature_selection.mutual_info_classif ; sklearn.feature_selection.mutual_info_regression ; sklearn.metrics.mutual_info_score
rel: entropia, icc, metodos-filtro, correlacion-de-pearson, independencia-estadistica
en: mutual information

## correlacion-total
frase: Generalización de la información mutua a muchas variables: cuánta redundancia hay en el conjunto.
desc: Es la suma de las entropías de cada variable menos la entropía conjunta. Vale cero si todas las variables son independientes y crece con la redundancia entre ellas. Medirla en muchas dimensiones es difícil; métodos como la gaussianización iterativa (RBIG) la estiman transformando los datos hasta hacerlos independientes.
ej: En una red de estaciones meteorológicas cercanas, la correlación total es alta porque muchas miden casi lo mismo.
rel: informacion-mutua, entropia, rbig, ica
en: total correlation (multi-information)

## medidas-de-dependencia
frase: Herramientas para cuantificar si dos variables están relacionadas, con qué forma y con qué intensidad.
desc: Van de lo más simple a lo más general. Pearson capta relaciones lineales; Spearman, monótonas; la información mutua y HSIC detectan cualquier dependencia, incluso no monótona. La dependencia condicional pregunta si la relación persiste al controlar por una tercera variable, el primer paso para hablar de causalidad.
ej: La relación entre temperatura y consumo eléctrico tiene forma de U (calefacción en invierno, aire acondicionado en verano): Pearson la subestima y la información mutua la detecta.
py: pandas.DataFrame.corr
rel: correlacion-de-pearson, correlacion-de-spearman, informacion-mutua, hsic, dependencia-condicional
en: dependence measures

## independencia-estadistica
frase: Dos variables son independientes si conocer una no dice nada sobre la otra.
desc: Formalmente, P(X, Y) = P(X)·P(Y). La independencia implica correlación cero, pero no al revés: dos variables pueden tener correlación nula y depender fuertemente de forma no lineal. Muchos métodos la suponen, como Naive Bayes, ICA o el supuesto de datos i.i.d. (independientes e idénticamente distribuidos).
ej: Los resultados de dos lanzamientos de un dado son independientes; la altura y el peso de una persona, no.
py: scipy.stats.chi2_contingency
rel: probabilidad-condicional, dependencia-condicional, informacion-mutua, test-chi-cuadrado, hsic
en: statistical independence

## correlacion-de-pearson
frase: Mide la fuerza de la relación lineal entre dos variables, de −1 a 1.
desc: Es la covarianza dividida por el producto de las desviaciones típicas. Vale 1 o −1 si los puntos caen sobre una recta y 0 si no hay relación lineal. Es sensible a los valores atípicos y ciega a las relaciones no lineales, por eso conviene mirar siempre el diagrama de dispersión.
cuando: Para medir relaciones aproximadamente lineales entre variables numéricas sin valores atípicos fuertes.
ej: Entre los metros cuadrados y el precio de las viviendas de un barrio, una correlación de 0,8 indica una relación lineal fuerte y positiva.
py: scipy.stats.pearsonr ; numpy.corrcoef
rel: correlacion-de-spearman, informacion-mutua, correlacion-no-implica-causalidad, eda
en: Pearson correlation

## correlacion-de-spearman
frase: Correlación entre rangos: mide relaciones monótonas aunque no sean lineales.
desc: Calcula la correlación de Pearson sobre las posiciones (rangos) de los datos en lugar de sobre sus valores. Así detecta cualquier relación que siempre crece o siempre decrece, y es robusta a valores atípicos. La tau de Kendall es una alternativa basada en contar los pares de puntos ordenados en el mismo sentido.
cuando: Con relaciones monótonas no lineales, datos ordinales o valores atípicos.
ej: Si y = eˣ, la relación es perfectamente monótona: Spearman vale 1, mientras que Pearson queda claramente por debajo.
py: scipy.stats.spearmanr ; scipy.stats.kendalltau
rel: correlacion-de-pearson, medidas-de-dependencia, outliers
en: Spearman rank correlation

## icc
frase: La información mutua reescalada para leerse como un coeficiente de correlación entre 0 y 1.
desc: El coeficiente de correlación de información transforma la información mutua I, medida en nats, en ICC = √(1 − e^(−2I)). Para variables gaussianas coincide con el valor absoluto de la correlación de Pearson, pero sigue siendo válido cuando la relación no es lineal. Combina la generalidad de la información mutua con una escala fácil de interpretar.
ej: Para y = sen(x) con algo de ruido en un rango amplio de x, Pearson sale cerca de 0, mientras que el ICC refleja una dependencia fuerte.
py: `np.sqrt(1 - np.exp(-2 * mi))`
rel: informacion-mutua, correlacion-de-pearson, medidas-de-dependencia
en: information coefficient of correlation (ICC)

## hsic
frase: Medida de dependencia basada en kernels que detecta relaciones no lineales entre variables.
desc: El criterio de independencia de Hilbert-Schmidt compara las matrices de similitud (kernels) de las dos variables: si sus estructuras de similitud varían juntas, hay dependencia. Con kernels adecuados, como el gaussiano, vale cero solo si las variables son independientes. Funciona en varias dimensiones sin estimar densidades y se usa en tests de independencia y en descubrimiento causal.
cuando: Para contrastar la independencia entre variables, incluso multidimensionales, cuando la relación puede ser no lineal.
ej: Comprobar si los residuos de una regresión son independientes de la entrada, la condición que usan los modelos de ruido aditivo para orientar causas.
py: hyppo.independence.Hsic
rel: truco-del-kernel, informacion-mutua, modelos-de-ruido-aditivo, independencia-estadistica
en: Hilbert-Schmidt Independence Criterion (HSIC)

## dependencia-condicional
frase: Si dos variables siguen relacionadas cuando ya se conoce una tercera.
desc: X e Y son condicionalmente independientes dado Z si, una vez conocida Z, X ya no informa sobre Y. Dos variables pueden estar correlacionadas solo porque comparten una causa común, y la relación desaparece al condicionar en ella. Los tests de independencia condicional son la pieza básica de algoritmos de descubrimiento causal como PC.
ej: Las ventas de helados y los ahogamientos están correlacionados, pero son independientes dada la temperatura: ambos suben en verano.
py: pingouin.partial_corr
rel: independencia-estadistica, algoritmo-pc, red-bayesiana, correlacion-espuria
en: conditional dependence

## modelos-graficos-probabilisticos
frase: Grafos que representan qué variables dependen directamente de cuáles.
desc: Combinan grafos y probabilidad: cada nodo es una variable y las aristas indican dependencias directas. La estructura del grafo descompone una distribución conjunta enorme en factores pequeños, lo que facilita aprenderla, razonar con ella e interpretarla. Los dos grandes tipos son los dirigidos (redes bayesianas) y los no dirigidos (campos aleatorios de Markov).
cuando: Para modelar dominios con muchas variables inciertas y relaciones conocidas o interpretables entre ellas.
ej: Un sistema de diagnóstico de averías que relaciona síntomas del coche (no arranca, luces débiles) con sus posibles causas (batería, motor de arranque).
py: pgmpy.models.DiscreteBayesianNetwork
rel: red-bayesiana, hmm, aprendizaje-de-estructura, causalidad
en: probabilistic graphical models

## red-bayesiana
frase: Grafo dirigido sin ciclos en el que cada variable depende solo de sus padres.
desc: Cada nodo tiene una tabla de probabilidades condicionada a sus padres en el grafo, y la distribución conjunta es el producto de todas esas tablas. Permite inferir variables no observadas a partir de las observadas en cualquier dirección: de causas a efectos y de efectos a causas. Es interpretable y admite incorporar conocimiento experto en la estructura.
cuando: Para razonar con incertidumbre en dominios con relaciones causales conocidas, combinando datos y conocimiento experto.
ej: Una red con Tabaquismo → Cáncer de pulmón → Tos permite calcular la probabilidad de cáncer de un paciente fumador que tose.
py: pgmpy.models.DiscreteBayesianNetwork ; pgmpy.inference.VariableElimination
rel: modelos-graficos-probabilisticos, teorema-de-bayes, aprendizaje-de-estructura, causalidad
en: Bayesian network

## aprendizaje-de-estructura
frase: Descubrir a partir de los datos qué aristas tiene el grafo de un modelo gráfico.
desc: En lugar de dibujar el grafo a mano, se infiere de los datos. Hay dos familias: los métodos basados en restricciones, como PC, eliminan aristas mediante tests de independencia condicional; los basados en puntuación, como hill climbing, buscan el grafo que maximiza una métrica como el BIC. El resultado suele dejar algunas direcciones sin fijar, porque varios grafos explican los datos igual de bien.
cuando: Cuando no conoces las relaciones entre variables y quieres obtener un grafo inicial e interpretable a partir de los datos.
ej: A partir de registros de pacientes, recuperar que la edad influye en la tensión arterial y esta, en el riesgo cardiaco.
py: pgmpy.causal_discovery.PC ; pgmpy.causal_discovery.HillClimbSearch
rel: algoritmo-pc, hill-climbing, red-bayesiana, descubrimiento-causal
en: structure learning

## algoritmo-pc
frase: Reconstruye el grafo quitando las aristas entre variables que resultan condicionalmente independientes.
desc: Parte de un grafo completo y elimina la arista entre cada par de variables que un test declara independientes, primero sin condicionar y después condicionando en conjuntos crecientes de vecinos. Después orienta las aristas que puede mediante estructuras en V (X → Z ← Y). Escala bien con muchas variables, pero depende de la fiabilidad de los tests.
cuando: Para obtener un primer grafo causal con muchas variables y datos suficientes para que los tests de independencia sean fiables.
ej: Con datos de lluvia, aspersor con temporizador y césped mojado, PC detecta que lluvia y aspersor son independientes, pero dependientes dado el césped, y orienta ambas flechas hacia él.
hp: significance_level: nivel de significación de los tests; más bajo, menos aristas | ci_test: test de independencia condicional que se usa | max_cond_vars: tamaño máximo del conjunto de variables en que se condiciona
py: pgmpy.causal_discovery.PC ; causallearn.search.ConstraintBased.PC.pc
rel: aprendizaje-de-estructura, dependencia-condicional, descubrimiento-causal, hill-climbing
en: PC algorithm

## hill-climbing
frase: Busca el mejor grafo añadiendo, quitando o invirtiendo aristas mientras mejore la puntuación.
desc: Método de aprendizaje de estructura basado en puntuación. En cada paso prueba todos los cambios de una sola arista y aplica el que más mejora una métrica como el BIC, que premia el ajuste a los datos y penaliza la complejidad. Es rápido, pero puede quedarse en óptimos locales; una lista tabú, que prohíbe deshacer los cambios recientes, ayuda a evitarlos.
cuando: Para aprender la estructura de una red bayesiana con datos discretos o gaussianos y un número moderado de variables.
ej: Aprender la red de dependencias entre síntomas y diagnósticos a partir de historiales clínicos.
hp: scoring_method: puntuación que se maximiza (BIC, K2, BDeu…) | max_indegree: número máximo de padres por nodo | tabu_length: cuántos cambios recientes no se pueden deshacer
py: pgmpy.causal_discovery.HillClimbSearch
rel: aprendizaje-de-estructura, algoritmo-pc, red-bayesiana, optimizacion
en: hill climbing search

## regresion-lineal-bayesiana
frase: Regresión lineal que da una distribución para los coeficientes y para cada predicción, no solo un valor.
desc: Pone una distribución a priori sobre los pesos y obtiene su posterior a partir de los datos. Cada predicción viene con su incertidumbre, que crece lejos de los datos de entrenamiento. El prior actúa como regularización: un prior gaussiano equivale a Ridge, y la fuerza de la regularización se puede estimar a partir de los propios datos.
cuando: Con pocos datos o cuando necesitas un intervalo de incertidumbre en cada predicción de un modelo lineal.
ej: Predecir la demanda de un producto recién lanzado con pocas semanas de historia, con un intervalo que se estrecha a medida que llegan datos.
py: sklearn.linear_model.BayesianRidge ; sklearn.linear_model.ARDRegression
rel: regresion-lineal, ridge-lasso-y-elastic-net, prior-verosimilitud-y-posterior, procesos-gaussianos, incertidumbre
en: Bayesian linear regression

## hmm
frase: Secuencia de estados ocultos que saltan entre sí y generan las observaciones que vemos.
desc: Supone que el sistema pasa por estados que no se observan directamente, con unas probabilidades de transición entre ellos, y que cada estado emite observaciones con su propia distribución. Se entrena con el algoritmo EM (Baum-Welch), y el algoritmo de Viterbi recupera la secuencia de estados más probable. Fue la base del reconocimiento de voz antes del deep learning.
cuando: Para secuencias que alternan entre regímenes o estados no observables, cuando quieres un modelo interpretable de esos cambios.
ej: Detectar con el acelerómetro de un móvil si una persona está quieta, andando o corriendo en cada momento.
hp: n_components: número de estados ocultos | covariance_type: forma de la covarianza de las emisiones de cada estado | n_iter: iteraciones máximas del algoritmo EM
py: hmmlearn.hmm.GaussianHMM ; hmmlearn.hmm.CategoricalHMM
rel: modelos-graficos-probabilisticos, rnn, gmm, componentes-de-una-serie
en: hidden Markov model (HMM)

## causalidad
frase: Saber qué variables provocan cambios en otras, no solo cuáles se mueven juntas.
desc: Un modelo predictivo aprende asociaciones; una pregunta causal plantea qué pasaría si se interviniera sobre una variable. Responderla con datos observacionales exige supuestos explícitos, normalmente expresados como un grafo causal. Las pistas para construirlo son los experimentos aleatorizados, la dependencia condicional, el orden temporal y las asimetrías entre causa y efecto.
cuando: Cuando la decisión implica intervenir (cambiar un precio, aplicar un tratamiento) y no basta con predecir.
ej: Un modelo puede predecir bien las ventas a partir del gasto en publicidad, pero solo un análisis causal dice cuánto subirían si se duplicara ese gasto.
rel: correlacion-no-implica-causalidad, descubrimiento-causal, red-bayesiana, causalidad-de-granger
en: causality

## correlacion-no-implica-causalidad
frase: Que dos variables se muevan juntas no significa que una cause la otra.
desc: Una correlación puede deberse a una causa común (variable de confusión), a que el efecto va en sentido contrario, al azar o a cómo se seleccionaron los datos. Por eso un modelo que predice bien no sirve automáticamente para decidir intervenciones. Establecer causalidad exige experimentos o supuestos adicionales sobre cómo se generan los datos.
ej: En los incendios a los que acuden más bomberos se producen más daños, pero no porque los bomberos los causen: ambas cosas dependen del tamaño del incendio.
rel: correlacion-espuria, causalidad, dependencia-condicional, correlacion-de-pearson
en: correlation does not imply causation

## correlacion-espuria
frase: Correlación sin relación real, fruto del azar, de una tercera variable o de tendencias compartidas.
desc: Surge cuando dos variables parecen relacionadas sin estarlo. Las causas típicas son una variable de confusión común, la coincidencia al explorar muchas variables y, en series temporales, la tendencia: dos series que crecen con el tiempo salen muy correlacionadas aunque no tengan nada que ver. Diferenciar las series o controlar por la variable común suele hacerla desaparecer.
ej: En EE. UU., el consumo de queso por habitante y las muertes por enredarse en las sábanas estuvieron muy correlacionados durante una década, por pura coincidencia.
rel: correlacion-no-implica-causalidad, estacionariedad, p-valor, dependencia-condicional
en: spurious correlation

## causalidad-de-granger
frase: X causa en sentido de Granger a Y si su pasado mejora la predicción de Y.
desc: Compara dos modelos que predicen Y: uno solo con el pasado de Y y otro que añade el pasado de X. Si el segundo predice significativamente mejor, se dice que X causa en sentido de Granger a Y. Es causalidad predictiva, no causalidad real: una causa común no observada puede producir el mismo resultado. Supone series estacionarias y relaciones lineales.
cuando: Para explorar qué series temporales anticipan a otras, por ejemplo al elegir variables para un modelo de predicción.
ej: Comprobar si la temperatura de los días anteriores mejora la predicción del consumo eléctrico de mañana más allá del propio consumo pasado.
hp: maxlag: número máximo de retardos que se prueban
py: statsmodels.tsa.stattools.grangercausalitytests
rel: informacion-dirigida, estacionariedad, autocorrelacion, causalidad
en: Granger causality

## informacion-dirigida
frase: Granger en clave de teoría de la información: cuánta información fluye del pasado de X al futuro de Y.
desc: Mide cuánta información aporta la historia de una serie a la siguiente observación de otra, más allá de la que aporta la historia de esta última. Al basarse en la información mutua capta relaciones no lineales, y al ser asimétrica permite poner dirección a las flechas de un grafo. La transferencia de entropía es una medida estrechamente relacionada.
ej: En una red de sensores de un río, comprobar que la información fluye de la estación de aguas arriba a la de aguas abajo y no al revés.
rel: causalidad-de-granger, informacion-mutua, descubrimiento-causal, entropia
en: directed information

## modelos-de-ruido-aditivo
frase: Orientan la flecha causal comprobando en qué dirección el residuo es independiente de la causa.
desc: Suponen que el efecto es una función de la causa más un ruido independiente: y = f(x) + n. Se ajusta una regresión en cada sentido y se examina el residuo: solo en la dirección causal correcta es independiente de la variable de entrada. Así pueden distinguir causa y efecto con solo dos variables, siempre que la relación sea no lineal o el ruido no gaussiano.
cuando: Para decidir la dirección causal entre dos variables cuando no hay información temporal ni experimental.
ej: Con la altitud y la temperatura media de muchas estaciones meteorológicas, el residuo solo es independiente al modelar la temperatura en función de la altitud, lo que apunta a altitud → temperatura.
py: causallearn.search.FCMBased.ANM.ANM.ANM
rel: hsic, descubrimiento-causal, regresion-lineal, causalidad
en: additive noise models (ANM)

## descubrimiento-causal
frase: Inferir el grafo de causas y efectos entre variables a partir de datos observacionales.
desc: Combina varias pistas para dibujar flechas causales sin hacer experimentos: independencias condicionales (algoritmo PC), puntuaciones de ajuste (GES), asimetrías entre causa y efecto (modelos de ruido aditivo, LiNGAM) y orden temporal (Granger). Todos dependen de supuestos, como que no haya causas comunes ocultas, así que el resultado es una hipótesis que conviene contrastar.
cuando: Para generar hipótesis causales en sistemas donde no se puede experimentar, como el clima, la biología o los procesos industriales.
ej: Inferir a partir de mediciones de expresión génica qué genes regulan a cuáles.
py: causallearn.search.ConstraintBased.PC.pc ; causallearn.search.ScoreBased.GES.ges ; causallearn.search.FCMBased.lingam.DirectLiNGAM
rel: algoritmo-pc, modelos-de-ruido-aditivo, aprendizaje-de-estructura, causalidad-de-granger
en: causal discovery
