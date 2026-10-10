# Fichas: Aprendizaje supervisado

## regresion-vs-clasificacion
frase: Predecir un número (regresión) o una categoría (clasificación).
desc: Son los dos tipos de problema supervisado. En regresión la salida es un valor numérico continuo, como un precio o una temperatura. En clasificación es una categoría, como spam/no spam o la especie de una flor. Muchos clasificadores devuelven en realidad una probabilidad por clase, y la etiqueta final sale de aplicar un umbral.
ej: Predecir el precio de una vivienda es regresión. Predecir si se venderá en menos de un mes es clasificación binaria.
py: sklearn.linear_model.LinearRegression ; sklearn.linear_model.LogisticRegression
rel: metricas-de-regresion, metricas-de-clasificacion, umbral-de-decision, regresion-logistica
en: regression vs classification

## clasificacion-multiclase-y-multietiqueta
frase: Elegir una clase entre varias, o asignar varias etiquetas a la vez.
desc: En la clasificación multiclase cada ejemplo pertenece a una sola de K clases, y el modelo suele dar K probabilidades con una softmax. En la multietiqueta un ejemplo puede tener varias etiquetas a la vez, y cada una se predice por separado con una sigmoide. La estrategia One-vs-Rest convierte cualquier clasificador binario en multiclase entrenando un modelo por clase.
cuando: Cuando hay más de dos clases posibles o cada ejemplo puede tener varias etiquetas simultáneas.
ej: Reconocer qué dígito del 0 al 9 aparece en una imagen es multiclase. Etiquetar una película como "comedia" y "romántica" a la vez es multietiqueta.
py: sklearn.multiclass.OneVsRestClassifier ; sklearn.multioutput.MultiOutputClassifier
rel: softmax, promedios-macro-y-micro, entropia-cruzada, regresion-logistica
en: multiclass & multilabel classification

## modelos-lineales
frase: Predicción como suma ponderada de las variables: simple, rápida e interpretable.
desc: Modelos cuya predicción es una combinación lineal de las variables, ŷ = w₀ + w₁x₁ + … + wₙxₙ, a veces pasada por una función como la sigmoide. Son rápidos, estables e interpretables, porque cada peso indica el efecto de su variable. Suelen ser la primera línea base que conviene probar.
cuando: Como primera línea base y cuando necesitas un modelo fácil de explicar con relaciones aproximadamente lineales.
ej: Estimar el consumo de un coche a partir de su peso y su potencia, y leer directamente cuántos litros añade cada 100 kg.
py: sklearn.linear_model
rel: regresion-lineal, regresion-logistica, glm, linea-base
en: linear models

## regresion-lineal
frase: La recta que mejor se ajusta a los datos minimizando el error al cuadrado.
desc: Ajusta la recta o el hiperplano que minimiza la suma de los errores al cuadrado entre predicción y valor real. Tiene solución analítica (las ecuaciones normales), aunque también se puede resolver por descenso por gradiente. Supone relación lineal, errores independientes y varianza constante.
cuando: Para predecir una variable numérica cuando la relación es aproximadamente lineal o necesitas interpretar el efecto de cada variable.
ej: Predecir el precio de una vivienda como una constante más 2.000 € por m² y 15.000 € por cada habitación.
py: sklearn.linear_model.LinearRegression ; statsmodels.api.OLS
rel: mse-rmse-y-mae, ridge-lasso-y-elastic-net, descenso-por-gradiente, regresion-lineal-bayesiana
en: linear regression (ordinary least squares)

## regresion-polinomica-e-interacciones
frase: Añadir potencias y productos de variables para captar curvas e interacciones.
desc: Extiende el modelo lineal añadiendo como variables nuevas potencias (x², x³) y productos entre variables (x₁·x₂). Sigue siendo lineal en los parámetros, pero capta curvas e interacciones, es decir, efectos de una variable que dependen del valor de otra. Con muchos términos crece el riesgo de sobreajuste.
cuando: Cuando un modelo lineal se queda corto porque la relación es curva o el efecto de una variable depende de otra.
ej: La publicidad puede vender más en temporada alta. El término publicidad × temporada recoge esa interacción.
hp: degree: grado del polinomio; más grado, más flexibilidad y más riesgo de sobreajuste | interaction_only: genera solo productos entre variables, sin potencias
py: sklearn.preprocessing.PolynomialFeatures
rel: regresion-lineal, feature-engineering, sobreajuste, gam
en: polynomial regression & interaction terms

## ridge-lasso-y-elastic-net
frase: Regresión lineal con penalización de los pesos para evitar el sobreajuste.
desc: Versiones regularizadas de la regresión lineal que penalizan los pesos grandes para reducir el sobreajuste. Ridge (L2) encoge todos los coeficientes hacia cero. Lasso (L1) lleva algunos exactamente a cero, y con ello selecciona variables. Elastic Net combina ambas penalizaciones. La fuerza de la penalización, α, es un hiperparámetro.
cuando: Cuando tienes muchas variables, correlacionadas entre sí o con pocos datos; Lasso, además, cuando quieres que el modelo seleccione variables.
ej: Con 500 variables candidatas y pocos datos, Lasso se queda con las 20 que tienen efecto real y anula el resto.
hp: alpha: fuerza de la penalización; más alta, coeficientes más pequeños | l1_ratio: en Elastic Net, mezcla de penalizaciones (0 = Ridge, 1 = Lasso)
py: sklearn.linear_model.Ridge ; sklearn.linear_model.Lasso ; sklearn.linear_model.ElasticNet
rel: regularizacion, regresion-lineal, metodos-integrados, penalizacion-l1-y-l2
en: Ridge, Lasso & Elastic Net

## regresion-logistica
frase: Modelo lineal que convierte una suma ponderada en la probabilidad de una clase.
desc: Modelo lineal de clasificación binaria. Calcula una combinación lineal de las variables y la transforma con la sigmoide en una probabilidad entre 0 y 1. Se entrena minimizando la entropía cruzada, lo que equivale a máxima verosimilitud. Sus coeficientes se leen en log-odds, por eso se usa mucho cuando hay que explicar las decisiones.
cuando: Para clasificación binaria como línea base sólida, sobre todo si necesitas probabilidades fiables y explicar el peso de cada variable.
ej: Estimar la probabilidad de que un paciente tenga diabetes a partir de su edad, su IMC y su glucosa, y marcarlo como positivo si supera 0,5.
hp: C: inversa de la regularización; más pequeño, modelo más simple | l1_ratio: tipo de penalización (0 = L2, 1 = L1, valores intermedios = Elastic Net) | class_weight: compensa clases desbalanceadas ("balanced")
py: sklearn.linear_model.LogisticRegression
rel: odds-y-odds-ratio, sigmoide-y-tanh, entropia-cruzada, glm
en: logistic regression

## odds-y-odds-ratio
frase: Por cuánto se multiplican las probabilidades a favor de un suceso al cambiar una variable.
desc: Las odds son el cociente p/(1−p) entre la probabilidad de que ocurra un suceso y la de que no ocurra. En la regresión logística, cada coeficiente β dice cuánto cambian las log-odds por cada unidad de su variable. e^β es el odds ratio, el factor por el que se multiplican las odds; un odds ratio de 2 significa que se duplican.
cuando: Para interpretar y comunicar el efecto de cada variable en una regresión logística.
ej: Si fumar tiene un odds ratio de 3 en un modelo de enfermedad cardíaca, las odds de un fumador triplican las de un no fumador con el resto de variables iguales.
py: `np.exp(modelo.coef_)`
rel: regresion-logistica, modelos-interpretables-por-diseno, intervalo-de-confianza
en: odds & odds ratio

## glm
frase: Regresión lineal generalizada a conteos, proporciones o cantidades positivas mediante una función de enlace.
desc: Generalizan la regresión lineal a respuestas que no siguen una normal. Conectan una combinación lineal de las variables con la media de la respuesta mediante una función de enlace, y suponen una distribución de la familia exponencial. La regresión lineal (enlace identidad, normal) y la logística (logit, binomial) son casos particulares; la de Poisson modela conteos.
cuando: Cuando la variable a predecir no es normal: conteos (Poisson), sí/no (binomial) o cantidades positivas y asimétricas (gamma).
ej: Predecir cuántos siniestros tendrá un conductor en un año con una regresión de Poisson y enlace logarítmico.
hp: alpha: fuerza de la regularización en los modelos de scikit-learn | family: distribución de la respuesta y su enlace en statsmodels (Poisson, Binomial, Gamma)
py: sklearn.linear_model.PoissonRegressor ; statsmodels.api.GLM
rel: regresion-lineal, regresion-logistica, glmm, gam
en: generalized linear models

## glmm
frase: Modelos lineales con efectos aleatorios para datos agrupados o medidas repetidas.
desc: Modelos lineales (o generalizados) que añaden efectos aleatorios para datos agrupados, jerárquicos o con medidas repetidas. Los efectos fijos son comunes a todos los datos, mientras que los aleatorios dejan que cada grupo (un paciente, un colegio, una tienda) tenga su propia desviación. Así se recoge la correlación entre observaciones del mismo grupo.
cuando: Cuando las observaciones vienen en grupos (pacientes, colegios, tiendas) que comparten características dentro de cada grupo.
ej: Analizar las notas de alumnos de muchos colegios dejando que cada colegio tenga su propio nivel de partida (intercepto aleatorio).
hp: groups: variable que define los grupos | re_formula: qué efectos varían de un grupo a otro (por defecto, solo el intercepto)
py: statsmodels.formula.api.mixedlm ; statsmodels.genmod.bayes_mixed_glm.BinomialBayesMixedGLM
rel: glm, regresion-lineal-bayesiana, modelos-interpretables-por-diseno
en: (generalized) linear mixed models

## gam
frase: Suma de curvas suaves, una por variable: no lineal pero interpretable.
desc: Sustituyen cada término lineal βⱼxⱼ por una función suave fⱼ(xⱼ), normalmente un spline, y suman los efectos: g(E[y]) = β₀ + f₁(x₁) + … + fₙ(xₙ). Captan relaciones no lineales sin perder interpretabilidad, porque el efecto de cada variable se puede dibujar por separado.
cuando: Cuando las relaciones no son lineales pero necesitas ver y explicar el efecto de cada variable por separado.
ej: Modelar el riesgo de hospitalización según la edad con una curva en U, más alta en niños y ancianos, algo que una recta no puede captar.
hp: n_splines: número de tramos de cada curva; más tramos, curvas más flexibles | lam: penalización de la rugosidad; más alta, curvas más suaves
py: pygam.LinearGAM ; interpret.glassbox.ExplainableBoostingClassifier
rel: glm, ebm, regresion-polinomica-e-interacciones
en: generalized additive models

## knn
frase: Predice copiando a los k ejemplos más parecidos.
desc: Para predecir un caso nuevo, busca los k ejemplos de entrenamiento más cercanos según una distancia. Después vota su clase (clasificación) o promedia su valor (regresión). No entrena ningún modelo: memoriza los datos, y el coste está en la predicción. Es sensible a la escala de las variables y a la maldición de la dimensionalidad.
cuando: Con conjuntos pequeños o medianos y pocas variables, cuando los casos parecidos suelen tener resultados parecidos.
ej: Estimar el alquiler de un piso promediando el de los 5 pisos más parecidos en tamaño, barrio y planta.
hp: n_neighbors: número de vecinos; pocos, predicción nerviosa; muchos, predicción suavizada | weights: si los vecinos cercanos pesan más (distance) o todos igual (uniform) | metric: distancia con la que se mide el parecido
py: sklearn.neighbors.KNeighborsClassifier ; sklearn.neighbors.KNeighborsRegressor
rel: distancia-y-similitud, escalado-de-variables, maldicion-de-la-dimensionalidad, busqueda-por-similitud
en: k-nearest neighbors

## arboles-de-decision
frase: Una cadena de preguntas sí/no que termina en una predicción.
desc: Dividen el espacio de las variables con preguntas sucesivas del tipo "¿x₃ > 5?" hasta llegar a hojas que dan la predicción. Se leen como un diagrama de flujo, captan no linealidades e interacciones y no necesitan escalar los datos. Su punto débil es la inestabilidad, porque pequeños cambios en los datos cambian mucho el árbol, y la tendencia a sobreajustar si crece demasiado.
cuando: Cuando necesitas reglas fáciles de explicar, o como pieza base de ensembles como Random Forest y gradient boosting.
ej: Triaje en urgencias: "¿fiebre > 39 °C?" → sí → "¿dificultad para respirar?" → sí → prioridad alta.
hp: max_depth: profundidad máxima; limita la complejidad del árbol | min_samples_leaf: mínimo de ejemplos por hoja; suaviza las predicciones | criterion: medida de impureza para elegir las divisiones (gini, entropy)
py: sklearn.tree.DecisionTreeClassifier ; sklearn.tree.DecisionTreeRegressor ; sklearn.tree.plot_tree
rel: criterios-de-division, poda-y-criterios-de-parada, random-forest, gradient-boosting
en: decision trees

## criterios-de-division
frase: Cómo elige el árbol la mejor pregunta: la que deja los grupos más puros.
desc: En cada nodo, el árbol elige la variable y el umbral que dejan los grupos resultantes lo más puros posible. En clasificación, la impureza se mide con el índice de Gini (la probabilidad de clasificar mal un elemento al azar) o con la entropía; la ganancia de información es cuánto reduce la entropía una división. En regresión se usa la reducción de la varianza (MSE).
ej: Un nodo con un 50 % de cada clase tiene un Gini de 0,5. Si una división lo parte en dos nodos 90/10, cada uno baja a 0,18.
py: sklearn.tree.DecisionTreeClassifier(criterion="gini" | "entropy")
rel: arboles-de-decision, entropia, informacion-mutua
en: split criteria: Gini impurity & entropy

## poda-y-criterios-de-parada
frase: Frenar o recortar el crecimiento del árbol para que no memorice el ruido.
desc: Un árbol que crece sin límite acaba con casi una hoja por ejemplo y memoriza el ruido. Se evita parando antes (profundidad máxima, mínimo de muestras por hoja o mejora mínima exigida) o podando después: se construye el árbol completo y se quitan las ramas que no mejoran la validación, por ejemplo con la poda por coste-complejidad.
cuando: Siempre que entrenes un árbol individual, para controlar el sobreajuste.
ej: Limitar la profundidad a 4 baja algo el acierto en entrenamiento, pero a menudo mejora el de test.
hp: ccp_alpha: intensidad de la poda por coste-complejidad | max_depth: profundidad máxima del árbol | min_samples_split: mínimo de ejemplos para dividir un nodo
py: sklearn.tree.DecisionTreeClassifier(max_depth=…, min_samples_leaf=…, ccp_alpha=…)
rel: arboles-de-decision, sobreajuste, compromiso-sesgo-varianza
en: pruning & stopping criteria

## metodos-kernel
frase: Resolver problemas no lineales con herramientas lineales trabajando con similitudes entre puntos.
desc: Familia de métodos que trabajan con similitudes entre pares de puntos, llamadas kernels, en lugar de con las variables originales. Equivale a proyectar los datos a un espacio de muchas más dimensiones, donde un problema no lineal se vuelve lineal, sin llegar a calcular esa proyección. Aquí entran las SVM, las SVR, la kernel ridge regression y los procesos gaussianos.
cuando: Con conjuntos pequeños o medianos y fronteras no lineales, cuando no hay datos suficientes para una red neuronal.
ej: Dos clases dispuestas en círculos concéntricos no se separan con una recta, pero un kernel RBF las separa sin problema.
py: sklearn.metrics.pairwise.rbf_kernel
rel: truco-del-kernel, svm, procesos-gaussianos, hsic
en: kernel methods

## truco-del-kernel
frase: Calcular similitudes como si los datos vivieran en un espacio enorme, sin construirlo.
desc: Muchos algoritmos solo necesitan productos escalares entre puntos. Si se sustituyen por una función kernel k(x, x′), el resultado es el mismo que si los datos se hubieran transformado a un espacio de mayor dimensión, pero sin construir esa transformación. Los kernels más comunes son el lineal, el polinómico y el RBF (gaussiano).
ej: El kernel RBF mide la similitud como exp(−γ‖x−x′‖²), lo que equivale a trabajar en un espacio de dimensión infinita.
py: sklearn.svm.SVC(kernel="rbf")
rel: metodos-kernel, svm, distancia-euclidea
en: kernel trick

## svm
frase: Separa las clases con la frontera de máximo margen.
desc: Clasificador que busca el hiperplano que separa las clases con el máximo margen, es decir, lo más lejos posible de los puntos más cercanos de cada clase (los vectores soporte). Con margen blando, controlado por el parámetro C, tolera algunos errores, y con kernels traza fronteras no lineales. Rinde muy bien con conjuntos medianos y de alta dimensión.
cuando: Para clasificar conjuntos pequeños o medianos, incluso con muchísimas variables, como textos o datos genómicos.
ej: Clasificar noticias en deportes o política a partir de miles de variables TF-IDF fue durante años terreno clásico de las SVM lineales.
hp: C: cuánto se penalizan los errores; alto, frontera más pegada a los datos | kernel: forma de la frontera (linear, rbf, poly) | gamma: alcance de cada punto en el kernel RBF; alto, frontera más irregular
py: sklearn.svm.SVC ; sklearn.svm.LinearSVC
rel: truco-del-kernel, svr, regresion-logistica, tf-idf
en: support vector machines

## svr
frase: Regresión que ignora los errores pequeños y penaliza solo los grandes.
desc: Versión de las SVM para regresión. Ajusta una función que deja la mayoría de los puntos dentro de un tubo de anchura ε alrededor de la predicción y solo penaliza los que quedan fuera. Ignorar los errores pequeños la hace robusta al ruido, y con kernels capta relaciones no lineales.
cuando: Para regresión no lineal con conjuntos pequeños o medianos y datos con ruido moderado.
ej: Predecir la demanda horaria de un servicio de bicicletas con un kernel RBF, sin penalizar desviaciones menores de 5 bicis (ε = 5).
hp: C: cuánto se penalizan los errores fuera del tubo | epsilon: anchura del tubo de errores tolerados | kernel: forma de la función (linear, rbf, poly) | gamma: alcance de cada punto en el kernel RBF
py: sklearn.svm.SVR
rel: svm, truco-del-kernel, kernel-ridge-regression
en: support vector regression

## kernel-ridge-regression
frase: Regresión Ridge con kernels: no lineal y con solución exacta.
desc: Combina la regresión Ridge con el truco del kernel: una regresión lineal regularizada en el espacio que induce el kernel, con solución cerrada. Su predicción coincide con la media de un proceso gaussiano equivalente, pero no da incertidumbre. En conjuntos medianos suele entrenar más rápido que una SVR.
cuando: Para regresión no lineal y suave con hasta unos pocos miles de datos.
ej: Ajustar una curva suave y no lineal a 2.000 medidas de un sensor con un kernel RBF y una sola línea de código.
hp: alpha: fuerza de la regularización | kernel: tipo de kernel (rbf, polynomial, linear) | gamma: alcance del kernel
py: sklearn.kernel_ridge.KernelRidge
rel: ridge-lasso-y-elastic-net, truco-del-kernel, procesos-gaussianos, svr
en: kernel ridge regression

## procesos-gaussianos
frase: Predicciones acompañadas de su incertidumbre, modelando una distribución sobre funciones.
desc: Modelo bayesiano no paramétrico que define una distribución sobre funciones: cualquier conjunto de puntos sigue una normal multivariante cuya covarianza la marca un kernel. Al condicionar en los datos, cada predicción nueva sale acompañada de su incertidumbre. Escala mal con muchos datos, porque su coste crece con el cubo de n.
cuando: Cuando tienes pocos datos y necesitas saber cuánto fiarte de cada predicción, como en la optimización bayesiana o en experimentos caros.
ej: En la optimización bayesiana de hiperparámetros, un GP modela el rendimiento del modelo en función de los hiperparámetros y propone dónde probar a continuación.
hp: kernel: forma de la covarianza (RBF, Matern o periódico) | alpha: ruido supuesto en las observaciones | n_restarts_optimizer: reinicios al ajustar los parámetros del kernel
py: sklearn.gaussian_process.GaussianProcessRegressor
rel: metodos-kernel, kernel-ridge-regression, optimizacion-bayesiana-de-hiperparametros, incertidumbre
en: Gaussian processes

## ensembles
frase: Combinar varios modelos para que juntos acierten más que cada uno.
desc: Combinan varios modelos para lograr predicciones más precisas y estables que las de cualquiera de ellos por separado. Hay tres estrategias principales. El bagging entrena modelos en paralelo sobre remuestreos y promedia, y reduce la varianza. El boosting entrena en secuencia corrigiendo errores, y reduce el sesgo. El stacking aprende cómo combinar los modelos.
cuando: Cuando quieres exprimir el rendimiento o estabilizar un modelo inestable; con datos tabulares son la opción más competitiva.
ej: En las competiciones de Kaggle con datos tabulares, las soluciones ganadoras suelen ser ensembles de varios modelos de gradient boosting.
py: sklearn.ensemble
rel: bagging, boosting, stacking, por-que-funcionan-los-ensembles
en: ensemble methods

## por-que-funcionan-los-ensembles
frase: Si los modelos se equivocan de forma distinta, al combinarlos los errores se compensan.
desc: Si los modelos se equivocan de forma distinta, con errores poco correlacionados, al combinarlos esos errores se compensan: baja la varianza del promedio y el voto mayoritario acierta más que cada miembro. Por eso la diversidad (distintos datos, variables o algoritmos) importa tanto como la calidad individual.
ej: Cinco clasificadores independientes que aciertan un 70 % cada uno llegan a cerca de un 84 % votando por mayoría.
rel: compromiso-sesgo-varianza, bagging, combinacion-por-votacion
en: why ensembles work

## combinacion-por-votacion
frase: Cada modelo vota una clase y gana la opción más respaldada.
desc: Formas de combinar las etiquetas que predicen varios clasificadores. En el voto por mayoría gana la clase más votada, y el voto puede ponderarse según la calidad de cada modelo. En el recuento de Borda cada modelo ordena las clases y se suman puntos según la posición. El Behavior Knowledge Space (BKS) asigna, con una tabla aprendida en entrenamiento, la clase real más frecuente a cada combinación de votos.
cuando: Cuando ya tienes varios clasificadores razonables y diversos y quieres combinarlos de la forma más sencilla.
ej: Tres modelos votan "gato", "gato" y "perro": por mayoría, la predicción final es "gato".
hp: voting: hard (vota la etiqueta) o soft (promedia las probabilidades) | weights: peso de cada modelo en la votación
py: sklearn.ensemble.VotingClassifier
rel: combinadores-algebraicos, stacking, ensembles
en: voting: majority, Borda count, BKS

## combinadores-algebraicos
frase: Promediar, multiplicar o tomar la mediana de las probabilidades de varios modelos.
desc: Combinan las salidas continuas de los modelos, como probabilidades o puntuaciones, con una operación algebraica: media, media ponderada, mediana, máximo, mínimo o producto. La media es robusta y casi siempre funciona bien. El producto castiga mucho a cualquier modelo muy seguro de que la clase es incorrecta, y la mediana resiste a los modelos atípicos.
cuando: Cuando los modelos dan probabilidades o puntuaciones y quieres combinarlas sin entrenar nada más.
ej: Promediar las probabilidades de fraude de tres modelos (0,9, 0,6 y 0,7) da 0,73, que después se compara con el umbral.
hp: weights: peso de cada modelo en la media
py: sklearn.ensemble.VotingClassifier(voting="soft")
rel: combinacion-por-votacion, stacking, ensembles
en: algebraic combiners

## stacking
frase: Un metamodelo aprende a combinar las predicciones de otros modelos.
desc: También llamado generalización apilada. Entrena varios modelos base y, encima, un metamodelo que aprende a combinar sus predicciones. Para que el metamodelo no aprenda de predicciones sobreajustadas, se entrena con predicciones fuera de muestra obtenidas por validación cruzada. Suele aportar mejoras pequeñas a cambio de más complejidad.
cuando: Cuando tienes modelos diversos con rendimientos parecidos y quieres sacar la última mejora de precisión.
ej: Combinar un Random Forest, un XGBoost y una red neuronal mediante una regresión logística que aprende cuánto fiarse de cada uno.
hp: final_estimator: el metamodelo que combina las predicciones | cv: cómo se generan las predicciones fuera de muestra | passthrough: si el metamodelo ve también las variables originales
py: sklearn.ensemble.StackingClassifier ; sklearn.ensemble.StackingRegressor
rel: combinacion-por-votacion, validacion-cruzada, mezcla-de-expertos, fuga-de-datos
en: stacked generalization

## mezcla-de-expertos
frase: Expertos especializados y una compuerta que decide a cuál escuchar en cada caso.
desc: Varios modelos expertos se especializan en distintas regiones del espacio de entrada, y una red de compuerta (gating) decide cuánto pesa cada uno para cada entrada concreta. A diferencia de un voto fijo, la combinación depende de la entrada. Hoy es una pieza clave de muchos LLMs: en cada capa solo se activan unos pocos expertos por token, lo que permite modelos enormes con un coste de cómputo moderado.
cuando: Cuando los datos tienen regímenes muy distintos y conviene especializar un modelo en cada uno; en los LLMs, para crecer sin multiplicar el cómputo.
ej: Mixtral 8x7B tiene 8 expertos por capa y activa 2 en cada token, de modo que solo usa una fracción de sus parámetros en cada paso.
rel: stacking, ensembles, llm
en: mixture of experts

## bagging
frase: Muchos modelos entrenados con remuestreos de los datos y promediados.
desc: Abreviatura de bootstrap aggregating. Entrena muchos modelos del mismo tipo, cada uno con una muestra bootstrap (con reemplazo) de los datos, y promedia o vota sus predicciones. Reduce sobre todo la varianza, por eso funciona muy bien con modelos inestables como los árboles profundos. Los ejemplos que no entran en cada muestra (out-of-bag) permiten estimar el error sin un conjunto de validación aparte.
cuando: Cuando tu modelo base es potente pero inestable, como un árbol profundo, y quieres reducir su varianza.
ej: Entrenar 100 árboles de regresión, cada uno con un remuestreo distinto de los datos, y promediar sus predicciones del precio de una casa.
hp: n_estimators: número de modelos | max_samples: tamaño de cada remuestreo | max_features: fracción de variables que ve cada modelo
py: sklearn.ensemble.BaggingClassifier ; sklearn.ensemble.BaggingRegressor
rel: random-forest, por-que-funcionan-los-ensembles, compromiso-sesgo-varianza, arboles-de-decision
en: bootstrap aggregating

## random-forest
frase: Muchos árboles distintos que votan: más estable y preciso que uno solo.
desc: Bagging de árboles de decisión con un ingrediente extra: en cada división el árbol solo puede elegir entre un subconjunto aleatorio de variables. Eso hace que los árboles se parezcan menos entre sí y que el promedio sea más eficaz. Es robusto, necesita poco ajuste y ofrece de serie una estimación del error (out-of-bag) y de la importancia de cada variable.
cuando: Como primera opción robusta con datos tabulares, cuando quieres buen rendimiento con muy poco ajuste.
ej: Clasificar los usos del suelo (cultivo, bosque, ciudad) en imágenes de satélite con 500 árboles, cada uno entrenado con datos y variables distintos.
hp: n_estimators: número de árboles; más árboles estabilizan el resultado a cambio de tiempo | max_features: variables candidatas en cada división; cuantas menos, más distintos son los árboles | max_depth: profundidad máxima de cada árbol | min_samples_leaf: mínimo de ejemplos por hoja; suaviza las predicciones
py: sklearn.ensemble.RandomForestClassifier ; sklearn.ensemble.RandomForestRegressor
rel: bagging, arboles-de-decision, importancia-por-permutacion, gradient-boosting
en: random forest

## boosting
frase: Modelos débiles en cadena, cada uno corrigiendo los errores del anterior.
desc: Construye el ensemble en secuencia: cada nuevo modelo débil, normalmente un árbol pequeño, se centra en los errores de los anteriores, y la predicción final es una suma ponderada de todos. Reduce sobre todo el sesgo. Es la familia más potente para datos tabulares, aunque más sensible al ruido y a los hiperparámetros que el bagging.
cuando: Cuando buscas la máxima precisión con datos tabulares y puedes dedicar tiempo a ajustar hiperparámetros.
ej: Cada árbol nuevo corrige un poco lo que los anteriores predijeron mal, hasta sumar cientos de árboles pequeños que juntos son muy precisos.
hp: max_iter: número de árboles (iteraciones) | learning_rate: cuánto aporta cada árbol | max_leaf_nodes: tamaño de cada árbol
py: sklearn.ensemble.HistGradientBoostingClassifier
rel: adaboost, gradient-boosting, xgboost-y-lightgbm, bagging
en: boosting

## adaboost
frase: Boosting que da más peso, en cada ronda, a los ejemplos mal clasificados.
desc: Adaptive Boosting, propuesto por Freund y Schapire. Entrena clasificadores débiles en secuencia y, tras cada uno, sube el peso de los ejemplos mal clasificados para que el siguiente se fije más en ellos. Cada clasificador recibe un peso según su error, y la predicción final es un voto ponderado. Equivale a minimizar una pérdida exponencial.
cuando: Con clasificadores débiles muy simples y datos poco ruidosos, cuando quieres un boosting clásico y ligero.
ej: En el detector de caras de Viola-Jones, AdaBoost combina miles de rasgos muy simples, cada uno apenas mejor que el azar, en un detector rápido y preciso.
hp: n_estimators: número de clasificadores débiles | learning_rate: cuánto aporta cada uno | estimator: el modelo débil (por defecto, un árbol de un solo nivel)
py: sklearn.ensemble.AdaBoostClassifier ; sklearn.ensemble.AdaBoostRegressor
rel: boosting, gradient-boosting, combinacion-por-votacion
en: adaptive boosting

## gradient-boosting
frase: Cada árbol nuevo corrige el error que dejan los anteriores, como un descenso por gradiente.
desc: Generaliza el boosting a cualquier función de pérdida diferenciable. Cada árbol nuevo se ajusta al gradiente negativo de la pérdida (con MSE, simplemente a los residuos) y se suma al modelo multiplicado por una tasa de aprendizaje pequeña. Es, en esencia, descenso por gradiente en el espacio de funciones.
cuando: Con datos tabulares cuando necesitas la máxima precisión, para regresión, clasificación o ranking.
ej: Predecir el tiempo de entrega de un pedido: el primer árbol da una estimación gruesa y cada uno de los 300 siguientes corrige parte del error que queda.
hp: n_estimators: número de árboles | learning_rate: tamaño de cada paso; más bajo, más árboles y mejor generalización | max_depth: profundidad de cada árbol | subsample: fracción de datos por árbol; por debajo de 1 añade aleatoriedad y regulariza
py: sklearn.ensemble.GradientBoostingRegressor ; sklearn.ensemble.HistGradientBoostingRegressor
rel: boosting, xgboost-y-lightgbm, descenso-por-gradiente, learning-rate
en: gradient boosting

## xgboost-y-lightgbm
frase: Gradient boosting industrial: rápido, escalable y estándar con datos tabulares.
desc: Implementaciones muy optimizadas de gradient boosting y estándar de facto con datos tabulares. XGBoost añade regularización en los árboles, manejo nativo de valores faltantes y paralelización. LightGBM acelera el entrenamiento agrupando las variables en histogramas y haciendo crecer los árboles hoja a hoja. CatBoost, una tercera opción, destaca con variables categóricas.
cuando: Con datos tabulares medianos o grandes, cuando quieres el mejor rendimiento en un tiempo de entrenamiento razonable.
ej: Predecir qué usuarios harán clic en un anuncio a partir de millones de registros y cientos de variables, con un entrenamiento de minutos en LightGBM.
hp: n_estimators: número de árboles | learning_rate: tamaño del paso de cada árbol | max_depth / num_leaves: tamaño de cada árbol (en XGBoost / en LightGBM) | subsample / colsample_bytree: fracción de filas / de columnas que ve cada árbol | reg_lambda: regularización L2 de las hojas
py: xgboost.XGBClassifier ; lightgbm.LGBMClassifier
rel: gradient-boosting, valores-faltantes-e-imputacion, shap, codificacion-de-variables-categoricas
en: XGBoost & LightGBM
