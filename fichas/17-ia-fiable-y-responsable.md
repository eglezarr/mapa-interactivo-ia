# Fichas: IA fiable y responsable

## xai
frase: Técnicas para entender por qué un modelo decide lo que decide, en general o en cada caso.
desc: Buscan que las decisiones de un modelo resulten comprensibles para quienes lo desarrollan, lo supervisan o se ven afectados por él. Hay dos caminos: usar modelos interpretables por diseño o aplicar métodos post hoc a un modelo ya entrenado, que lo explican en conjunto (explicaciones globales) o predicción a predicción (explicaciones locales). Sirven para depurar modelos, detectar sesgos y atajos, generar confianza y cumplir requisitos legales.
cuando: Cuando hay que justificar decisiones, auditar un modelo o comprobar que no se apoya en patrones espurios.
ej: Antes de usar un modelo que prioriza pacientes en urgencias, el hospital comprueba con SHAP que se apoya en las constantes vitales y los síntomas, y no en el tipo de seguro médico del paciente.
rel: taxonomia-de-metodos-xai, interpretabilidad-vs-explicabilidad, shap, correlacion-espuria, decisiones-automatizadas-y-rgpd
en: explainable AI (XAI)

## caja-negra-vs-caja-blanca
frase: Modelos cuyo funcionamiento se puede seguir frente a modelos opacos que solo se observan desde fuera.
desc: Un modelo de caja blanca, como una regresión lineal, un árbol pequeño o una lista de reglas, permite seguir cómo llega a cada predicción. Uno de caja negra, como una red profunda o un ensemble de cientos de árboles, solo puede explicarse desde fuera, con métodos post hoc. Suele suponerse que la opacidad se paga con más precisión, pero en datos tabulares un modelo interpretable bien construido a menudo rinde casi igual.
ej: Un protocolo de triaje basado en cinco reglas se puede auditar entero en una tarde; uno basado en una red neuronal necesita herramientas como SHAP para saber en qué se apoya.
rel: modelos-interpretables-por-diseno, transparencia, arboles-de-decision, mlp, xai
en: black box vs white box (glass box)

## interpretabilidad-vs-explicabilidad
frase: Entender el modelo directamente frente a necesitar una explicación externa de lo que hace.
desc: Se usan a menudo como sinónimos, pero conviene distinguirlos. Un modelo es interpretable si una persona puede entender su mecanismo directamente, como los coeficientes de una regresión. La explicabilidad consiste en producir explicaciones comprensibles de un modelo, sea interpretable o no, normalmente con métodos post hoc. Esas explicaciones son aproximaciones y pueden no ser fieles al modelo; un modelo interpretable, en cambio, se muestra tal como es.
ej: Los coeficientes de una regresión logística que predice el reingreso hospitalario se interpretan por sí mismos; un gráfico SHAP de una red neuronal entrenada para lo mismo es una explicación.
rel: caja-negra-vs-caja-blanca, transparencia, evaluacion-de-explicaciones, regresion-logistica
en: interpretability vs explainability

## transparencia
frase: Tres niveles: poder simular el modelo de cabeza, entender cada pieza y entender cómo aprende.
desc: La simulabilidad es que una persona pueda reproducir el cálculo completo del modelo en un tiempo razonable. La descomponibilidad, que cada parte (entradas, parámetros y cálculos) tenga un significado intuitivo. La transparencia algorítmica, que se entienda el procedimiento de aprendizaje y sus garantías, como que el ajuste de un modelo lineal converge a una solución única. Un modelo puede cumplir unos niveles y no otros.
ej: Una regresión lineal con tres variables es simulable y descomponible; con 500 términos polinómicos sigue siendo algorítmicamente transparente, pero ya nadie puede calcularla de cabeza.
rel: caja-negra-vs-caja-blanca, interpretabilidad-vs-explicabilidad, modelos-interpretables-por-diseno, regresion-lineal
en: transparency (simulatability, decomposability, algorithmic transparency)

## taxonomia-de-metodos-xai
frase: Los métodos de explicación se clasifican por cuándo actúan, qué alcance tienen y si dependen del modelo.
desc: Pueden ser intrínsecos, si el propio modelo es interpretable, o post hoc, si se aplican a un modelo ya entrenado. Globales, si describen el comportamiento general, o locales, si explican una predicción. Agnósticos, si solo necesitan consultar el modelo, como LIME o SHAP, o específicos, si aprovechan su estructura interna, como los gradientes de una red. También varía el formato: importancias de variables, reglas, ejemplos o contrafactuales.
ej: Grad-CAM es un método post hoc, local y específico de las redes convolucionales; la importancia por permutación es post hoc, global y agnóstica.
rel: explicaciones-locales, explicaciones-globales, modelos-interpretables-por-diseno, explicabilidad-en-deep-learning
en: taxonomy of XAI methods

## modelos-interpretables-por-diseno
frase: Modelos cuya estructura ya es la explicación: reglas, árboles pequeños y modelos lineales o aditivos.
desc: En lugar de explicar una caja negra a posteriori, se entrena directamente un modelo legible: una regresión lineal o logística, un árbol poco profundo, una lista de reglas o un modelo aditivo generalizado. Sus explicaciones son exactas, no aproximaciones, y facilitan detectar errores y variables problemáticas. Propuestas modernas como RuleFit o EBM añaden interacciones y no linealidades sin perder legibilidad.
cuando: En decisiones de alto impacto (salud, justicia, empleo) o que deban auditarse, si un modelo legible alcanza un rendimiento suficiente.
ej: Una lista de reglas para estimar el riesgo de ictus se puede aplicar a mano en la consulta y rinde casi como modelos mucho más complejos.
rel: caja-negra-vs-caja-blanca, arboles-de-decision, gam, regresion-logistica, transparencia
en: interpretable-by-design (glass-box) models

## reglas-de-decision
frase: Aprender una lista corta de reglas si-entonces que describen cuándo un caso pertenece a una clase.
desc: Los algoritmos de cobertura secuencial aprenden una regla que cubre muchos casos de la clase objetivo con pocos errores, apartan los casos cubiertos y repiten. RIPPER es el más conocido: hace crecer cada regla añadiendo condiciones, la poda con datos reservados y optimiza el conjunto final con un criterio de longitud de descripción. El resultado es un modelo que cualquiera puede leer y aplicar a mano.
cuando: Para clasificación tabular cuando el modelo debe poder leerse como reglas, por ejemplo para validarlo con expertos del dominio.
ej: Un filtro de correo aprende reglas como "si el mensaje contiene la palabra gratis y más de tres enlaces, es spam", que el equipo puede revisar una a una.
hp: k: número de pasadas de optimización del conjunto de reglas | prune_size: fracción de datos reservada para podar | max_rules: número máximo de reglas
py: wittgenstein.RIPPER
rel: arboles-de-decision, poda-y-criterios-de-parada, rulefit, anchors, modelos-interpretables-por-diseno
en: decision rules (RIPPER)

## rulefit
frase: Un modelo lineal disperso sobre reglas extraídas de árboles: capta interacciones y sigue siendo legible.
desc: RuleFit entrena un ensemble de árboles, convierte el camino desde la raíz hasta cada nodo en una regla binaria, como edad > 60 y fumador, y ajusta una regresión con penalización Lasso sobre esas reglas y las variables originales. El Lasso deja solo unas pocas reglas con peso distinto de cero, así que el modelo final es una suma legible de efectos lineales y de reglas que capturan interacciones.
cuando: En datos tabulares, cuando un modelo lineal se queda corto pero hace falta poder leer el modelo.
ej: Un modelo de riesgo de reingreso hospitalario queda en 15 reglas con su peso, como "más de tres ingresos previos y más de 75 años", además del efecto lineal de cada variable.
hp: max_rules: número máximo de reglas del modelo final | tree_size: tamaño de los árboles que generan las reglas, que limita su longitud
py: imodels.RuleFitClassifier ; imodels.RuleFitRegressor
rel: ridge-lasso-y-elastic-net, gradient-boosting, reglas-de-decision, ebm, regresion-polinomica-e-interacciones
en: RuleFit

## ebm
frase: Un modelo aditivo entrenado con boosting: una curva por variable y por par de variables.
desc: La Explainable Boosting Machine es un modelo aditivo generalizado con interacciones: la predicción es la suma de una función de cada variable y de unas pocas funciones de pares de variables. Cada función se aprende con boosting de árboles diminutos, recorriendo las variables por turnos con una tasa de aprendizaje muy baja. En datos tabulares suele rendir cerca de los mejores modelos de caja negra, y cada curva se puede inspeccionar y corregir.
cuando: En datos tabulares, cuando se quiere una precisión cercana a la de gradient boosting con un modelo totalmente interpretable.
ej: Un modelo de este tipo que predecía la mortalidad por neumonía mostró en su curva de asma que los asmáticos parecían tener menos riesgo, un artefacto de que recibían atención más intensiva, y los médicos pudieron corregirlo.
hp: interactions: número de interacciones entre pares que se añaden | learning_rate: tasa de aprendizaje del boosting | max_bins: número de intervalos en que se discretiza cada variable
py: interpret.glassbox.ExplainableBoostingClassifier ; interpret.glassbox.ExplainableBoostingRegressor
rel: gam, gradient-boosting, analisis-de-interacciones, modelos-interpretables-por-diseno, pdp
en: Explainable Boosting Machine (EBM)

## explicaciones-locales
frase: Explican una predicción concreta: qué variables la han empujado y qué tendría que cambiar.
desc: Se centran en un caso: por qué este paciente tiene un riesgo alto o por qué esta imagen se clasifica como gato. Pueden dar la contribución de cada variable (SHAP, LIME), una regla que basta para sostener la predicción (Anchors), el cambio mínimo que la invertiría (contrafactuales) o casos parecidos de referencia (prototipos). Son las explicaciones que suelen necesitar las personas afectadas por una decisión.
cuando: Para justificar decisiones individuales, revisar casos dudosos o depurar errores concretos del modelo.
ej: Un sistema de cribado de currículos descarta una candidatura, y la explicación local muestra que pesaron sobre todo los años de experiencia y la falta de una certificación.
rel: explicaciones-globales, shap, lime, explicaciones-contrafactuales, taxonomia-de-metodos-xai
en: local explanations

## perfiles-ceteris-paribus
frase: Cómo cambiaría la predicción de un caso si se moviera una sola variable, con todo lo demás fijo.
desc: Se toma una observación, se recorre una de sus variables por una rejilla de valores manteniendo las demás constantes (ceteris paribus significa "lo demás igual") y se calcula la predicción en cada punto. El perfil responde a preguntas de tipo qué pasaría si para ese caso concreto. Superponer los perfiles de muchas observaciones da las curvas ICE. Si la variable está muy correlacionada con otras, el perfil recorre combinaciones irreales.
cuando: Para explorar cómo respondería el modelo a cambios en una variable para un caso concreto, por ejemplo al revisarlo con un experto.
ej: Para un paciente de 55 años, el perfil muestra que su riesgo cardiovascular predicho bajaría del 18 % al 9 % si su presión sistólica pasara de 160 a 120.
hp: grid_points: número de valores en los que se evalúa cada variable | variable_splits_type: rejilla uniforme o basada en cuantiles
py: dalex.Explainer ; dalex.Explainer.predict_profile
rel: ice, pdp, explicaciones-contrafactuales, correlacion-de-pearson
en: Ceteris Paribus profiles

## ice
frase: Una curva por observación que muestra cómo cambia su predicción al variar una variable.
desc: Las curvas ICE dibujan, para cada observación, la predicción del modelo al recorrer los valores de una variable con el resto fijo. Su media es el PDP. Verlas por separado revela efectos heterogéneos e interacciones que el promedio oculta, como curvas que suben para unos casos y bajan para otros. La versión centrada las alinea en su primer punto para comparar mejor sus formas.
cuando: Junto al PDP, para comprobar si el efecto de una variable es el mismo para todos los casos o depende de otras variables.
ej: En un modelo de precios de vivienda, las curvas ICE de la superficie suben con fuerza para los pisos del centro y apenas para los de las afueras, una interacción que el PDP promedio oculta.
hp: kind: individual dibuja solo las curvas ICE y both las superpone al PDP | centered: alinea las curvas en su primer punto
py: sklearn.inspection.PartialDependenceDisplay.from_estimator
rel: pdp, perfiles-ceteris-paribus, analisis-de-interacciones, ale
en: Individual Conditional Expectation (ICE)

## lime
frase: Explica una predicción ajustando un modelo lineal sencillo alrededor de ese caso con datos perturbados.
desc: LIME genera muchas versiones perturbadas de la observación, por ejemplo cambiando valores, quitando palabras u ocultando zonas de una imagen, obtiene la predicción del modelo para cada una y ajusta un modelo lineal ponderado por su cercanía al caso original. Los coeficientes de ese modelo local son la explicación. Funciona con cualquier modelo y con tablas, texto o imágenes, pero sus resultados dependen del muestreo y de la anchura del vecindario.
cuando: Para explicar predicciones individuales de cualquier modelo, sobre todo de texto o imágenes, cuando basta una aproximación local.
ej: Un clasificador de reseñas marca una como negativa, y LIME señala las palabras decepcionante y nunca como las que más han pesado.
hp: kernel_width: anchura del vecindario que pondera las muestras | num_samples: número de muestras perturbadas | num_features: número de variables que muestra la explicación
py: lime.lime_tabular.LimeTabularExplainer ; lime.lime_tabular.LimeTabularExplainer.explain_instance ; lime.lime_text.LimeTextExplainer
rel: shap, anchors, modelo-subrogado-global, regresion-lineal, analisis-de-sentimiento
en: LIME (Local Interpretable Model-agnostic Explanations)

## anchors
frase: Explica una predicción con una regla que la mantiene casi siempre, cambie lo que cambie el resto.
desc: Un anchor es una regla, como edad > 60 y fumador, que mientras se cumpla sostiene la predicción aunque varíen las demás variables. El método busca reglas que alcancen una precisión mínima, por ejemplo del 95 %, y entre ellas prefiere las de mayor cobertura, es decir, las que se aplican a más casos. A diferencia de LIME, deja claro hasta dónde llega la explicación: fuera de la regla no promete nada.
cuando: Para explicar predicciones individuales con reglas fáciles de comunicar y saber a qué otros casos se aplica la misma explicación.
ej: Un modelo predice que un paciente faltará a su cita, y el anchor es "cita pedida con más de 30 días de antelación y sin recordatorio": cuando se cumple, la predicción se mantiene en el 96 % de los casos.
hp: threshold: precisión mínima que debe tener la regla
py: alibi.explainers.AnchorTabular ; alibi.explainers.AnchorTabular.explain
rel: lime, reglas-de-decision, explicaciones-contrafactuales, explicaciones-locales
en: Anchors (scoped rules)

## valores-de-shapley
frase: Reparto justo de un resultado entre jugadores según su contribución media en todas las coaliciones.
desc: Vienen de la teoría de juegos cooperativos. El valor de Shapley de un jugador es su contribución marginal media al unirse a todas las coaliciones posibles de los demás. Es el único reparto que cumple a la vez eficiencia (las partes suman el total), simetría, jugador nulo y aditividad. En explicabilidad, los jugadores son las variables y el resultado es la predicción menos su valor medio; su cálculo exacto es exponencial en el número de variables.
cuando: Para repartir una predicción entre las variables de cualquier modelo con un criterio con garantías; con muchas variables se estima por muestreo.
ej: Dos voluntarios recogen por separado 10 y 20 kg de basura, y juntos 40 kg; el valor de Shapley reparte a partes iguales los 10 kg extra: 15 para el primero y 25 para el segundo.
hp: nsamples: número de coaliciones evaluadas en cada explicación, que gana precisión a cambio de tiempo
py: shap.KernelExplainer ; shap.ExactExplainer
rel: shap, lime, importancia-por-permutacion, analisis-de-interacciones
en: Shapley values

## shap
frase: Explica cada predicción repartiéndola entre las variables según sus valores de Shapley.
desc: SHAP expresa cada predicción como el valor medio del modelo más una contribución por variable, calculada con valores de Shapley. Tiene algoritmos rápidos para cada familia de modelos: exacto y eficiente para árboles (TreeSHAP), basado en gradientes para redes y basado en muestreo para cualquier modelo (KernelSHAP). Agregando las explicaciones locales se obtienen vistas globales, como la importancia media. Con variables correlacionadas, el reparto depende de cómo se simulan las ausentes.
cuando: Como método de referencia para explicar modelos tabulares, sobre todo ensembles de árboles, tanto caso a caso como en conjunto.
ej: Un modelo de gradient boosting predice 7,2 días de estancia hospitalaria para un paciente cuando la media es 4,5; SHAP atribuye +1,9 días a la edad, +1,2 a una cirugía previa y −0,4 a haber ingresado por la mañana.
py: shap.TreeExplainer ; shap.Explainer ; shap.plots.waterfall ; shap.plots.beeswarm
rel: valores-de-shapley, xgboost-y-lightgbm, lime, importancia-por-permutacion, analisis-de-interacciones
en: SHAP (SHapley Additive exPlanations)

## explicaciones-contrafactuales
frase: El cambio más pequeño en las entradas que haría que el modelo diera otra respuesta.
desc: Un contrafactual responde a qué tendría que haber sido distinto para obtener otro resultado. Se busca un punto cercano al original que cambie la predicción, con pocas variables modificadas, valores realistas y solo variables que la persona pueda cambiar: la edad o el país de nacimiento no deberían aparecer. Suelen generarse varios, diversos, para ofrecer alternativas, y no exigen revelar cómo funciona el modelo por dentro.
cuando: Para dar a una persona afectada por una decisión automática una explicación accionable de cómo obtener otro resultado.
ej: Un sistema de admisión rechaza a un estudiante, y el contrafactual indica que habría sido admitido con un 7,8 en matemáticas en lugar de un 7,1, o con un curso de verano adicional.
hp: method: estrategia de búsqueda (aleatoria, genética o por vecinos cercanos) | total_CFs: número de contrafactuales que se generan | features_to_vary: variables que se permite cambiar
py: dice_ml.Dice ; dice_ml.Dice.generate_counterfactuals
rel: perfiles-ceteris-paribus, anchors, decisiones-automatizadas-y-rgpd, ejemplos-adversarios, causalidad
en: counterfactual explanations

## prototipos-y-excepciones
frase: Explicar unos datos o un modelo con casos reales representativos y con los que no encajan.
desc: Los prototipos son observaciones reales que representan bien una parte de los datos; las excepciones (criticisms) son las que esos prototipos representan mal. Mostrar ambos ayuda a entender los datos y el comportamiento del modelo con ejemplos concretos, que a las personas les resultan más intuitivos que los pesos. Métodos como MMD-critic o ProtoSelect los eligen automáticamente, y una predicción puede justificarse con los casos de entrenamiento más parecidos.
cuando: Para entender un conjunto de datos o un modelo a través de ejemplos, o para justificar una predicción con casos similares ya conocidos.
ej: Un clasificador de razas de perro se resume con tres fotos prototipo por raza, y las excepciones muestran fotos de perros disfrazados o en blanco y negro que el modelo confunde.
hp: eps: radio dentro del cual un prototipo cubre a otros puntos | lambda_penalty: penalización por prototipo, que reduce cuántos se eligen
py: alibi.prototypes.ProtoSelect
rel: k-medoids-y-k-median, knn, explicaciones-locales, calidad-y-etiquetado-de-datos
en: prototypes & criticisms

## explicaciones-globales
frase: Describen el comportamiento del modelo en conjunto: qué variables importan y cómo influyen.
desc: Responden a preguntas sobre el modelo entero: qué variables usa más, qué forma tiene el efecto de cada una (lineal, con un umbral, en forma de U) y qué interacciones ha aprendido. Las herramientas habituales son la importancia por permutación, los gráficos PDP y ALE, el análisis de interacciones y los modelos subrogados. Sirven para validar el modelo con expertos, detectar fugas de datos o relaciones absurdas y documentarlo.
cuando: Para validar un modelo antes de desplegarlo, detectar variables problemáticas o documentar cómo se comporta en general.
ej: El análisis global de un modelo de reingresos hospitalarios revela que su variable más importante es el identificador del hospital, señal de una fuga de datos o de diferencias de registro entre centros.
rel: explicaciones-locales, importancia-por-permutacion, pdp, fuga-de-datos, modelo-subrogado-global
en: global explanations

## pdp
frase: El efecto medio de una variable: la predicción promedio al fijarla en cada uno de sus valores.
desc: Se fija una variable en un valor para todas las observaciones, se promedian las predicciones y se repite a lo largo de una rejilla de valores. La curva resultante muestra el efecto medio de la variable y, con dos variables, su interacción en un mapa de calor. Es intuitivo, pero supone que la variable es independiente de las demás: si no lo es, evalúa combinaciones irreales. Además, el promedio oculta efectos heterogéneos.
cuando: Para ver la forma del efecto medio de una o dos variables en cualquier modelo, si no están muy correlacionadas con el resto.
ej: El PDP de un modelo de alquiler de bicicletas muestra que la demanda sube con la temperatura hasta unos 25 °C y baja a partir de ahí.
hp: grid_resolution: número de valores de la variable en los que se evalúa | percentiles: rango de la variable que se recorre, para evitar extremos con pocos datos
py: sklearn.inspection.PartialDependenceDisplay.from_estimator ; sklearn.inspection.partial_dependence
rel: ice, ale, perfiles-ceteris-paribus, gam, independencia-estadistica
en: partial dependence plot (PDP)

## ale
frase: El efecto de una variable calculado con diferencias locales, fiable aunque haya variables correlacionadas.
desc: El gráfico ALE divide la variable en intervalos y, en cada uno, calcula cuánto cambia la predicción al mover la variable del extremo inferior al superior, usando solo las observaciones de ese intervalo; después acumula esas diferencias. Como trabaja con datos reales y cambios pequeños, evita las combinaciones irreales del PDP cuando hay correlación, y además es más rápido. A cambio, su lectura es algo menos intuitiva.
cuando: En lugar del PDP cuando las variables están correlacionadas, que es lo habitual en datos reales.
ej: En un modelo de precios de vivienda, superficie y número de habitaciones van juntas: el PDP evalúa pisos de 30 m² con cinco habitaciones, mientras que el ALE solo compara pisos parecidos a los reales.
hp: min_bin_points: número mínimo de observaciones por intervalo
py: alibi.explainers.ALE ; alibi.explainers.ALE.explain
rel: pdp, ice, correlacion-de-pearson, explicaciones-globales
en: accumulated local effects (ALE)

## importancia-por-permutacion
frase: Cuánto empeora el modelo al barajar los valores de una variable y romper su relación con el objetivo.
desc: Se mide el rendimiento del modelo, se permutan al azar los valores de una variable en los datos de evaluación y se vuelve a medir: la caída es su importancia. Funciona con cualquier modelo, es fácil de entender y, calculada sobre datos de test, refleja lo que el modelo usa para generalizar. Con variables correlacionadas reparte la importancia entre ellas y puede infravalorarlas, y al permutar crea combinaciones irreales.
cuando: Para saber qué variables usa un modelo cualquiera, midiendo sobre validación o test en lugar de fiarse de la importancia interna de los árboles.
ej: Al barajar la tensión arterial, el AUC de un modelo de riesgo cardiovascular cae de 0,84 a 0,71; al barajar el código postal, no cambia.
hp: n_repeats: número de permutaciones por variable, que se promedian | scoring: métrica cuya caída se mide
py: sklearn.inspection.permutation_importance
rel: seleccion-de-caracteristicas, random-forest, shap, metricas-de-evaluacion
en: permutation feature importance

## modelo-subrogado-global
frase: Un modelo interpretable entrenado para imitar las predicciones de una caja negra.
desc: Se entrena un modelo sencillo, como un árbol poco profundo, usando como objetivo las predicciones del modelo complejo en lugar de las etiquetas reales. Si lo imita bien, sus reglas o coeficientes describen aproximadamente el comportamiento de la caja negra. Su fidelidad, medida por ejemplo como el porcentaje de predicciones en que ambos coinciden, indica cuánto fiarse: un subrogado poco fiel no explica el modelo original.
cuando: Para obtener una visión global aproximada de un modelo complejo, después de comprobar que el subrogado lo imita con suficiente fidelidad.
ej: Un árbol de profundidad 3 entrenado con las predicciones de una red reproduce el 92 % de sus decisiones y muestra que la red se apoya sobre todo en la edad y en dos analíticas.
py: sklearn.tree.DecisionTreeClassifier ; `arbol.fit(X, modelo.predict(X))`
rel: arboles-de-decision, destilacion-de-conocimiento, lime, r2
en: global surrogate model

## analisis-de-interacciones
frase: Medir cuánto del efecto de una variable depende del valor de otras.
desc: Hay interacción cuando el efecto de una variable cambia según el valor de otra, de modo que la predicción no es la suma de efectos separados. El estadístico H de Friedman mide qué parte de la variación se debe a interacciones; los valores de interacción de SHAP reparten cada predicción entre efectos principales y efectos de cada par; los PDP de dos variables las visualizan. Detectarlas ayuda a decidir si bastaría un modelo aditivo.
cuando: Para descubrir qué combinaciones de variables usa un modelo complejo o decidir qué interacciones añadir a un modelo interpretable.
ej: En un modelo de riesgo de diabetes, los valores de interacción de SHAP revelan que el índice de masa corporal pesa mucho más en pacientes con antecedentes familiares.
py: shap.TreeExplainer.shap_interaction_values ; sklearn.inspection.PartialDependenceDisplay.from_estimator
rel: regresion-polinomica-e-interacciones, ebm, shap, pdp, ice
en: feature interaction analysis

## explicabilidad-en-deep-learning
frase: Métodos que aprovechan gradientes, activaciones o atención para explicar redes neuronales.
desc: Las redes tienen millones de parámetros, pero son diferenciables y se pueden inspeccionar por dentro. Los mapas de saliencia y Grad-CAM usan gradientes para señalar qué partes de la entrada influyen en la salida; la visualización de filtros y activaciones muestra qué detecta cada capa; los mapas de atención indican en qué tokens o regiones se fija un transformer. Son rápidos y visuales, pero pueden resultar convincentes sin ser fieles al modelo.
cuando: Para depurar y auditar redes de visión, texto o audio, por ejemplo para comprobar que una CNN mira el objeto y no el fondo.
ej: Un clasificador de lobos y huskies acierta en test, pero el análisis revela que se apoya en la nieve del fondo y no en el animal.
py: captum.attr
rel: cnn, mecanismo-de-atencion, filtros-y-mapas-de-caracteristicas, evaluacion-de-explicaciones
en: explainability in deep learning

## grad-cam
frase: Mapas de calor que señalan qué zonas de la entrada han pesado más en la predicción.
desc: Un mapa de saliencia calcula el gradiente de la salida respecto a cada píxel, es decir, cuánto cambiaría la predicción al alterarlo; suele ser ruidoso, y variantes como SmoothGrad o Integrated Gradients lo estabilizan. Grad-CAM usa en su lugar los mapas de características de la última capa convolucional, ponderados por el gradiente medio de la clase, y produce un mapa más grueso pero más limpio de las regiones relevantes para esa clase.
cuando: Para ver en qué zonas de una imagen se apoya una CNN, depurar errores o detectar atajos como marcas de agua o fondos.
ej: Una red que detecta neumonía en radiografías acierta mucho, pero Grad-CAM revela que se fija en la marca del hospital impresa en una esquina, que delata de qué centro viene la imagen.
hp: target_layers: capas cuyos mapas de características se usan, normalmente la última convolucional
py: pytorch_grad_cam.GradCAM ; captum.attr.Saliency ; captum.attr.IntegratedGradients
rel: cnn, filtros-y-mapas-de-caracteristicas, correlacion-espuria, backpropagation, evaluacion-de-explicaciones
en: saliency maps & Grad-CAM

## visualizacion-de-filtros-y-activaciones
frase: Ver qué detecta cada filtro: bordes en las primeras capas y partes de objetos en las últimas.
desc: Hay dos enfoques principales. Uno muestra las activaciones, los mapas de características que produce una imagen concreta en cada capa, o busca las imágenes reales que más activan cada filtro. El otro sintetiza, por ascenso de gradiente a partir de ruido, la entrada que maximiza un filtro. Ambos revelan la jerarquía que aprenden las CNN: bordes y colores al principio, texturas y patrones después y partes de objetos al final.
cuando: Para entender qué ha aprendido una red de visión, detectar filtros muertos o redundantes, o mostrar cómo funcionan las CNN.
ej: En una CNN entrenada con ImageNet, un filtro de la primera capa responde a bordes diagonales y otro de las últimas capas, a caras de perro.
py: `keras.Model(inputs=modelo.input, outputs=capa.output)`
rel: filtros-y-mapas-de-caracteristicas, cnn, campo-receptivo, inspiracion-biologica-de-las-cnn, grad-cam
en: filter & activation visualization

## mapas-de-atencion
frase: Visualizar los pesos de atención de un transformer para ver qué tokens o regiones relaciona.
desc: Cada cabeza de atención asigna pesos entre pares de tokens, y dibujarlos muestra qué partes de la entrada combina el modelo, por ejemplo un pronombre con su antecedente. En visión, la atención del token de clase sobre los parches da un mapa de la imagen. Son útiles para explorar, pero un peso alto no garantiza que ese token sea importante para la predicción; attention rollout, que combina todas las capas, se aproxima mejor.
cuando: Para explorar qué relaciones ha aprendido un transformer o depurar su comportamiento, sin tomar los pesos como explicación causal.
ej: En la frase "María le prestó el libro a Ana porque ella se lo había pedido", una cabeza de atención conecta "ella" con "Ana".
py: bertviz.head_view ; `AutoModel.from_pretrained(nombre, output_attentions=True)`
rel: mecanismo-de-atencion, atencion-multicabeza, vit, token-cls, evaluacion-de-explicaciones
en: attention maps

## evaluacion-de-explicaciones
frase: Comprobar si una explicación refleja de verdad lo que hace el modelo y si es estable y útil.
desc: Una explicación puede parecer convincente sin ser correcta. Se evalúa su fidelidad, por ejemplo eliminando las variables o píxeles señalados como importantes y comprobando que la predicción cae; su robustez, si cambia poco ante entradas casi iguales; y su complejidad, si es lo bastante concisa. Las pruebas de cordura verifican que la explicación cambia al aleatorizar los pesos del modelo, y los estudios con personas, que de verdad ayuda a decidir.
cuando: Antes de confiar en un método de explicación o al elegir entre varios para un modelo y unos datos concretos.
ej: Al borrar el 10 % de los píxeles que Grad-CAM marca como más importantes, la confianza en la clase cae del 95 % al 30 %; al borrar un 10 % al azar, apenas baja.
py: quantus.FaithfulnessCorrelation ; quantus.MaxSensitivity
rel: interpretabilidad-vs-explicabilidad, grad-cam, shap, lime
en: evaluation of explanations

## equidad
frase: Que un modelo no perjudique injustamente a personas o grupos por su sexo, origen u otros atributos.
desc: Un modelo aprende de datos históricos que pueden reflejar desigualdades, y puede reproducirlas o amplificarlas aunque no use directamente el atributo sensible. La equidad estudia cómo detectar esos sesgos, medirlos con métricas de grupo o individuales y mitigarlos antes, durante o después del entrenamiento. No hay una única definición correcta: varias métricas razonables son incompatibles entre sí, y elegir una es también una decisión ética y legal.
cuando: Siempre que un modelo tome o apoye decisiones sobre personas: contratación, salud, educación, justicia o acceso a servicios.
ej: Una herramienta de selección entrenada con contrataciones pasadas penaliza los currículos que mencionan un club deportivo femenino, porque históricamente se contrataba sobre todo a hombres.
rel: sesgo-algoritmico-y-atributos-sensibles, metricas-de-equidad, mitigacion-del-sesgo, ai-act, calidad-y-etiquetado-de-datos
en: fairness

## sesgo-algoritmico-y-atributos-sensibles
frase: Errores sistemáticos que perjudican a ciertos grupos, a menudo heredados de los datos.
desc: Los atributos sensibles son características protegidas como el sexo, el origen étnico, la edad o la discapacidad. El sesgo puede venir de datos históricos discriminatorios, de grupos poco representados, de etiquetas que miden mal lo que interesa o de variables que actúan como sustituto del atributo, como el código postal. Por eso quitar el atributo del modelo no basta, y además impide medir si hay sesgo.
ej: Un algoritmo para priorizar pacientes en programas de cuidados usaba el gasto sanitario como medida de necesidad e infravaloraba el riesgo de los pacientes negros, que con la misma gravedad habían recibido menos atención y generado menos gasto.
rel: equidad, metricas-de-equidad, calidad-y-etiquetado-de-datos, etiquetas-ruidosas, correlacion-espuria
en: algorithmic bias & sensitive attributes

## metricas-de-equidad
frase: Medir si un modelo trata igual a distintos grupos en sus decisiones, sus errores o su calibración.
desc: La paridad demográfica exige la misma tasa de decisiones positivas en cada grupo. La igualdad de oportunidades, la misma tasa de verdaderos positivos, y las probabilidades igualadas, además, la misma tasa de falsos positivos. La calibración por grupos exige que una misma puntuación signifique lo mismo en todos. Si las tasas base difieren y el modelo no es perfecto, no pueden cumplirse todas a la vez, así que hay que elegir según el contexto.
cuando: Para auditar un modelo que decide sobre personas, comparando sus métricas por grupos antes de desplegarlo y durante su uso.
ej: Un modelo de cribado de cáncer de piel detecta el 90 % de los casos en pacientes de piel clara y el 75 % en los de piel oscura: incumple la igualdad de oportunidades aunque su precisión global sea alta.
py: fairlearn.metrics.MetricFrame ; fairlearn.metrics.demographic_parity_difference ; fairlearn.metrics.equalized_odds_difference
rel: matriz-de-confusion, precision-y-recall, umbral-de-decision, mitigacion-del-sesgo, sesgo-algoritmico-y-atributos-sensibles
en: fairness metrics

## mitigacion-del-sesgo
frase: Reducir el sesgo corrigiendo los datos, restringiendo el entrenamiento o ajustando las decisiones finales.
desc: En el preprocesado se corrigen los datos: reponderar o remuestrear grupos, o transformar variables para eliminar su correlación con el atributo sensible. Durante el entrenamiento se impone una restricción de equidad, como en el método de reducciones con gradiente exponenciado. En el posprocesado se fijan umbrales de decisión distintos por grupo para igualar sus tasas de error. Casi siempre tiene un coste en precisión que conviene medir y justificar.
cuando: Cuando una auditoría detecta diferencias injustificadas entre grupos y hay que corregirlas.
ej: Un sistema de alerta de abandono escolar ajusta sus umbrales para detectar la misma proporción de alumnos en riesgo en zonas rurales y urbanas.
hp: constraints: criterio de equidad que se impone (paridad demográfica, probabilidades igualadas…) | eps: incumplimiento máximo permitido de la restricción
py: fairlearn.reductions.ExponentiatedGradient ; fairlearn.postprocessing.ThresholdOptimizer ; fairlearn.preprocessing.CorrelationRemover
rel: metricas-de-equidad, submuestreo-y-ponderacion-de-clases, umbral-de-decision, regularizacion
en: bias mitigation

## robustez
frase: Que el modelo mantenga su comportamiento ante ruido, datos distintos o ataques deliberados.
desc: Un modelo robusto no cambia de predicción ante variaciones pequeñas o irrelevantes de la entrada. Incluye la robustez frente al ruido y a cambios naturales de los datos, como otra cámara o la deriva con el tiempo, y frente a ataques: ejemplos adversarios, envenenamiento de los datos de entrenamiento o extracción del modelo mediante consultas. Se evalúa con perturbaciones y ataques controlados, y se mejora con entrenamiento robusto y aumento de datos.
cuando: En sistemas críticos o expuestos a quien pueda querer engañarlos: conducción, biometría, ciberseguridad o moderación de contenidos.
ej: Un detector de señales de tráfico deja de reconocer una señal de stop a la que se han pegado unas pocas pegatinas blancas y negras.
rel: ejemplos-adversarios, entrenamiento-robusto, deriva-de-datos, data-augmentation, ood
en: robustness

## ejemplos-adversarios
frase: Entradas con perturbaciones mínimas, a menudo invisibles, diseñadas para que el modelo se equivoque.
desc: Se construyen buscando, normalmente con el gradiente del modelo, una perturbación pequeña que cambie su predicción. En imágenes puede ser imperceptible para una persona y aun así provocar un error con mucha confianza. Los ataques pueden ser de caja blanca, con acceso a los gradientes, o de caja negra, solo con consultas, y dirigidos a una clase concreta o no. Muchos se transfieren entre modelos distintos entrenados para la misma tarea.
cuando: Para evaluar la robustez de un modelo antes de desplegarlo en un entorno donde alguien podría intentar engañarlo.
ej: Unas gafas con un estampado diseñado para ello hacen que un sistema de reconocimiento facial identifique a quien las lleva como otra persona.
hp: eps: tamaño máximo de la perturbación | max_iter: número de pasos del ataque iterativo
py: art.attacks.evasion.ProjectedGradientDescent
rel: fgsm, robustez, explicaciones-contrafactuales, descenso-por-gradiente, entrenamiento-robusto
en: adversarial examples

## fgsm
frase: Ataque de un solo paso: mover cada píxel un poco en la dirección del signo del gradiente.
desc: El Fast Gradient Sign Method calcula el gradiente de la pérdida respecto a la entrada y suma a cada píxel ε veces su signo: según una aproximación lineal, es el cambio que más aumenta la pérdida sin mover ningún píxel más de ε. Es muy rápido y mostró que la vulnerabilidad de las redes se debe en buena parte a su comportamiento casi lineal en alta dimensión. Su versión iterativa, PGD, es más potente.
cuando: Para generar ejemplos adversarios rápidamente, en evaluaciones iniciales de robustez o dentro del entrenamiento adversario.
ej: Con ε = 0,007, FGSM convierte la foto de un panda, clasificada con un 58 % de confianza, en un gibón con un 99 %, sin ningún cambio visible.
hp: eps: tamaño máximo de la perturbación (por píxel con la norma L∞) | norm: norma con la que se mide la perturbación (L∞, L2…)
py: art.attacks.evasion.FastGradientMethod
rel: ejemplos-adversarios, backpropagation, entrenamiento-robusto, diferenciacion-automatica
en: Fast Gradient Sign Method (FGSM)

## ataque-de-un-pixel
frase: Engañar a un clasificador de imágenes cambiando un único píxel, elegido con un algoritmo evolutivo.
desc: Es un ataque de caja negra extremo: modifica uno o muy pocos píxeles, sin límite en cuánto los cambia. Usa evolución diferencial, un algoritmo evolutivo que no necesita gradientes, para buscar la posición y el color del píxel que más reducen la probabilidad de la clase correcta. En el estudio original bastó para engañar a la red en cerca de dos tercios de las imágenes de prueba de CIFAR-10, de 32 × 32 píxeles.
cuando: Para evaluar la robustez de un clasificador de imágenes cuando solo se tiene acceso a sus predicciones.
ej: Cambiar el color de un único píxel de una foto de 32 × 32 hace que la red clasifique un caballo como un coche.
hp: th: número de píxeles que se pueden modificar | es: estrategia evolutiva (CMA-ES o evolución diferencial) | max_iter: número de generaciones de la búsqueda
py: art.attacks.evasion.PixelAttack
rel: ejemplos-adversarios, fgsm, cnn, robustez
en: one-pixel attack

## entrenamiento-robusto
frase: Entrenar el modelo con ejemplos adversarios o perturbados para que aprenda a resistirlos.
desc: El entrenamiento adversario genera ejemplos adversarios durante el entrenamiento, por ejemplo con FGSM o PGD, y los mezcla con los datos limpios; equivale a minimizar la pérdida en el peor caso dentro de un radio de perturbación. Es la defensa empírica más eficaz, pero encarece el entrenamiento y suele restar algo de precisión en datos limpios. Las defensas certificadas, en cambio, garantizan matemáticamente la robustez dentro de un radio.
cuando: Cuando el modelo debe resistir ataques o perturbaciones de un tamaño conocido y se puede asumir un entrenamiento más caro.
ej: En CIFAR-10, entrenar con ejemplos PGD sube la precisión ante ese ataque de casi cero a cerca de la mitad de las imágenes, a cambio de perder unos 8 puntos en imágenes limpias.
hp: ratio: proporción de cada lote que se sustituye por ejemplos adversarios
py: art.defences.trainer.AdversarialTrainer
rel: fgsm, ejemplos-adversarios, data-augmentation, inyeccion-de-ruido, regularizacion-en-redes
en: robust (adversarial) training

## incertidumbre
frase: Cuantificar cuánto fiarse de cada predicción, además de la predicción en sí.
desc: Un modelo útil debería indicar cuándo no está seguro, para que esos casos se revisen o se descarten. La incertidumbre puede deberse al ruido propio de los datos (aleatoria) o a la falta de conocimiento del modelo (epistémica). Se expresa con intervalos de predicción, probabilidades bien calibradas o conjuntos de predicción; la predicción conformal los construye para cualquier modelo con una cobertura garantizada.
cuando: Cuando las predicciones apoyan decisiones de riesgo y hace falta saber en qué casos abstenerse o pedir revisión humana.
ej: Un modelo prevé una demanda eléctrica de 520 MW para mañana, con un intervalo del 90 % entre 480 y 570 MW, y el operador planifica las reservas con el extremo superior.
hp: confidence_level: cobertura deseada de los intervalos
py: mapie.regression.SplitConformalRegressor
rel: intervalo-de-confianza, procesos-gaussianos, human-in-the-loop, ood, incertidumbre-aleatoria-y-epistemica
en: uncertainty quantification

## incertidumbre-aleatoria-y-epistemica
frase: El ruido propio de los datos, que no se reduce, y el desconocimiento, que disminuye con más datos.
desc: La incertidumbre aleatoria procede de la variabilidad intrínseca del fenómeno o del ruido de medida, y seguiría ahí con un modelo perfecto. La epistémica procede de lo que el modelo no sabe, por falta de datos en esa zona o por no saber qué parámetros son los correctos. Con un ensemble, la entropía de la predicción se descompone en la entropía media de cada modelo (aleatoria) y su desacuerdo (epistémica).
ej: El resultado de un dado tiene incertidumbre aleatoria que ningún dato elimina; la altura media de los adultos de un país del que solo se ha medido a diez personas tiene sobre todo incertidumbre epistémica.
rel: incertidumbre, informacion-mutua, probabilidad-frecuentista-vs-bayesiana, ood, compromiso-sesgo-varianza
en: aleatoric & epistemic uncertainty

## propagacion-de-la-incertidumbre
frase: Calcular cómo se transmite la incertidumbre de las entradas a los resultados que dependen de ellas.
desc: Si un resultado se calcula a partir de magnitudes inciertas, su incertidumbre depende de la de cada una y de cómo se combinan. La aproximación lineal usa las derivadas parciales: por ejemplo, en una suma de variables independientes se suman las varianzas. Si la función es muy no lineal o las distribuciones no son normales, se recurre a Monte Carlo: muestrear las entradas, calcular el resultado muchas veces y estudiar su distribución.
cuando: Cuando las predicciones de un modelo alimentan cálculos posteriores, como dosis, pedidos de stock o simulaciones, y hay que saber cuánto fiarse del resultado.
ej: Si la demanda diaria de un producto se predice con una desviación típica de 10 unidades y los errores de cada día son independientes, la de la demanda semanal es 10 × √7 ≈ 26 unidades, y no 70.
py: uncertainties.ufloat ; `y = f(rng.normal(mu, sigma, size=10_000))`
rel: incertidumbre, distribuciones-de-probabilidad, intervalo-de-confianza, estimacion-de-incertidumbre-en-redes
en: uncertainty propagation

## estimacion-de-incertidumbre-en-redes
frase: Hacer que una red indique cuánto se fía de cada predicción, con ensembles, dropout o predicción conformal.
desc: Las probabilidades de la softmax suelen estar mal calibradas y no reflejan lo que la red no sabe: puede dar un 99 % a una entrada absurda. Para estimarlo se entrenan varias redes con distinta inicialización y se mide su desacuerdo (deep ensembles), o se mantiene el dropout activo al predecir y se repite la predicción (MC dropout). El escalado de temperatura mejora la calibración, y la predicción conformal da conjuntos con cobertura garantizada.
cuando: Cuando una predicción equivocada con mucha confianza sale cara, como en diagnóstico médico o conducción autónoma.
ej: Un clasificador de lesiones cutáneas con MC dropout repite cada predicción 30 veces: en una imagen nítida coinciden todas; en una borrosa oscilan entre tres diagnósticos, y el caso se deriva al dermatólogo.
hp: confidence_level: cobertura deseada de los conjuntos de predicción
py: `modelo(x, training=True)` ; mapie.classification.SplitConformalClassifier
rel: dropout, ensembles, incertidumbre-aleatoria-y-epistemica, softmax, ood
en: uncertainty estimation in neural networks

## regulacion-y-gobernanza
frase: Normas, estándares y prácticas para que la IA se desarrolle y se use de forma legal y responsable.
desc: Los sistemas de IA están sujetos a leyes generales, como la de protección de datos, y cada vez más a leyes específicas, como el Reglamento Europeo de IA. A ello se suman estándares técnicos y marcos voluntarios para gestionar sus riesgos, y prácticas de documentación como las model cards. La gobernanza convierte todo ello en procesos dentro de la organización: inventario de modelos, evaluación de riesgos, supervisión humana, registros y auditorías.
cuando: Desde el inicio de cualquier proyecto que diseñe, compre o despliegue un sistema de IA que afecte a personas o a procesos críticos.
ej: Antes de desplegar un asistente conversacional en un hospital, se clasifica su nivel de riesgo, se documentan sus datos y limitaciones y se define quién revisa sus respuestas.
rel: ai-act, decisiones-automatizadas-y-rgpd, gobernanza-ia, model-cards, mlops
en: AI regulation & governance

## decisiones-automatizadas-y-rgpd
frase: El RGPD limita las decisiones totalmente automatizadas que afectan de forma significativa a las personas.
desc: El artículo 22 del Reglamento General de Protección de Datos reconoce el derecho a no ser objeto de decisiones basadas únicamente en tratamiento automatizado, incluida la elaboración de perfiles, con efectos jurídicos o igual de significativos. Se permiten si son necesarias para un contrato, las autoriza la ley o hay consentimiento explícito, con garantías como la intervención humana y el derecho a impugnar. La persona también debe recibir información significativa sobre la lógica aplicada.
ej: Una plataforma de empleo que descarta candidaturas de forma automática, sin revisión humana, debe permitir que un candidato pida que una persona revise su caso y explicarle en qué se basó la decisión.
rel: human-in-the-loop, explicaciones-contrafactuales, xai, ai-act
en: automated decision-making & GDPR (art. 22)

## ai-act
frase: El reglamento europeo que regula la IA según su riesgo, desde usos prohibidos hasta obligaciones de transparencia.
desc: El Reglamento (UE) 2024/1689 clasifica los sistemas según su riesgo. Prohíbe prácticas inaceptables, como la puntuación social o la manipulación que explota vulnerabilidades. Los sistemas de alto riesgo, en ámbitos como el empleo, la educación o la aplicación de la ley, deben cumplir requisitos de gestión de riesgos, calidad de datos, documentación, supervisión humana, precisión y robustez. Los chatbots y los deepfakes tienen obligaciones de transparencia, y los modelos de propósito general, obligaciones propias.
ej: Un sistema que filtra currículos es de alto riesgo: su proveedor debe documentarlo, registrar su funcionamiento y garantizar la supervisión humana. Un chatbot de atención al cliente solo debe dejar claro que se está hablando con una IA.
rel: decisiones-automatizadas-y-rgpd, gobernanza-ia, model-cards, human-in-the-loop, que-es-un-modelo-fundacional
en: EU AI Act

## gobernanza-ia
frase: Marcos y normas técnicas para gestionar de forma sistemática los riesgos de la IA en una organización.
desc: Los estándares convierten principios generales en procesos auditables. ISO/IEC 42001 define un sistema de gestión de la IA certificable, al estilo de ISO/IEC 27001 para la seguridad de la información, e ISO/IEC 23894 trata la gestión de sus riesgos. El marco de gestión de riesgos de IA del NIST organiza el trabajo en cuatro funciones: gobernar, mapear, medir y gestionar. En la práctica implica inventariar modelos, asignar responsables, evaluar riesgos y revisar periódicamente.
cuando: Para organizar cómo se desarrollan, compran y supervisan los sistemas de IA en una organización y prepararse para auditorías o certificaciones.
ej: Una empresa mantiene un registro de todos sus modelos con su responsable, su nivel de riesgo, sus datos de entrenamiento y la fecha de su última evaluación.
rel: ai-act, mlops, model-cards, seguimiento-de-experimentos, regulacion-y-gobernanza
en: AI standards & governance (ISO/IEC 42001, NIST AI RMF)

## model-cards
frase: Fichas que acompañan a un modelo: para qué sirve, cómo se evaluó, sus limitaciones y sus riesgos.
desc: Propuestas en 2019, las model cards resumen los detalles del modelo, su uso previsto y los usos desaconsejados, los datos de entrenamiento y evaluación, las métricas, también desglosadas por grupos, y las limitaciones y consideraciones éticas. Su equivalente para los conjuntos de datos son las datasheets for datasets. Son la práctica habitual en repositorios como Hugging Face y facilitan cumplir las obligaciones de documentación de la regulación.
cuando: Al publicar o desplegar cualquier modelo, sobre todo si otras personas lo van a reutilizar o si interviene en decisiones sobre personas.
ej: La model card de un reconocedor de voz indica que se evaluó con hablantes adultos de inglés y español, que su tasa de error se duplica con voces infantiles y que no debe usarse para vigilancia.
py: huggingface_hub.ModelCard ; huggingface_hub.ModelCardData
rel: hugging-face, gobernanza-ia, metricas-de-equidad, ai-act, reconocimiento-de-voz
en: model cards
