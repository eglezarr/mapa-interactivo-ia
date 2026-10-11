# Fichas: Fundamentos

## ia-ml-dl
frase: La IA contiene al machine learning, que a su vez contiene al deep learning.
desc: La inteligencia artificial es el objetivo general: máquinas capaces de hacer tareas que asociamos con la inteligencia. El machine learning es hoy la vía dominante para lograrlo: en lugar de programar reglas, el sistema las aprende de datos. El deep learning es la parte del machine learning que usa redes neuronales con muchas capas, capaces de aprender directamente de datos crudos como imágenes, texto o audio.
ej: Un filtro antispam con reglas escritas a mano es IA sin aprendizaje. Uno entrenado con correos etiquetados es machine learning, y uno basado en un Transformer es deep learning.
rel: programacion-vs-ml, mlp, breve-historia-de-la-ia
en: AI vs machine learning vs deep learning

## programacion-vs-ml
frase: En la programación clásica escribes las reglas; en el machine learning se aprenden de ejemplos.
desc: Un programa clásico recibe datos y reglas escritas por una persona y produce resultados. En el aprendizaje automático se le dan datos y los resultados esperados, y el algoritmo deduce las reglas, que forman el modelo y luego se aplican a datos nuevos. Compensa cuando las reglas son demasiado complejas para escribirlas o cambian con el tiempo.
ej: Es inviable escribir a mano las reglas para reconocer un gato en una foto, pero un modelo las aprende a partir de miles de fotos etiquetadas.
rel: ia-ml-dl, cuatro-ingredientes, aprendizaje-supervisado
en: traditional programming vs machine learning

## breve-historia-de-la-ia
frase: De la neurona artificial de 1943 a los LLMs, con inviernos de por medio y grandes saltos de escala.
desc: McCulloch y Pitts proponen la neurona artificial en 1943 y el término inteligencia artificial nace con el taller de Dartmouth en 1956. Tras el perceptrón (1958) llegan los inviernos de la IA por expectativas incumplidas. La retropropagación (1986), las LSTM (1997) y, sobre todo, la unión de datos masivos y GPUs desde 2012 impulsan el deep learning, que desemboca en los Transformers (2017) y en LLMs como ChatGPT (2022).
ej: En 2012, AlexNet bajó el error top-5 en ImageNet del 26 % del segundo clasificado a alrededor del 15 %, y marcó el arranque de la era del deep learning.
rel: ia-ml-dl, perceptron, imagenet-y-alexnet, llm
en: brief history of AI

## cuatro-ingredientes
frase: Todo modelo combina datos, una familia de modelos, un objetivo que medir y un método para optimizarlo.
desc: Casi cualquier método de machine learning se reduce a cuatro piezas. Los datos son los ejemplos de los que aprender. El modelo es la familia de funciones candidatas, como una recta, un árbol o una red. El objetivo es la función de pérdida, que mide lo mal que lo hace. La optimización es el procedimiento que busca los parámetros que minimizan esa pérdida. Cambiar cualquiera de las cuatro cambia el resultado.
ej: En una regresión lineal de precios de pisos, los datos son pisos vendidos, el modelo una recta, el objetivo el error cuadrático y la optimización, mínimos cuadrados o descenso por gradiente.
rel: funciones-de-perdida, descenso-por-gradiente, minimizacion-del-riesgo-empirico, parametros-e-hiperparametros
en: data, model, objective & optimization

## parametros-e-hiperparametros
frase: Los parámetros los aprende el modelo; los hiperparámetros los fijas tú antes de entrenar.
desc: Los parámetros son los valores internos que el entrenamiento ajusta a partir de los datos, como los pesos de una red o los coeficientes de una regresión. Los hiperparámetros controlan cómo se aprende y qué forma tiene el modelo, como la tasa de aprendizaje, la profundidad de un árbol o el número de vecinos. Se eligen con datos de validación, nunca con los de test.
ej: En un Random Forest, las divisiones de cada árbol son parámetros; el número de árboles y su profundidad máxima son hiperparámetros.
rel: ajuste-de-hiperparametros-y-automl, entrenamiento-validacion-y-test, learning-rate
py: `modelo.get_params()` ; sklearn.model_selection.GridSearchCV
en: parameters vs hyperparameters

## teoria-de-la-decision
frase: Decidir bien bajo incertidumbre, combinando la probabilidad de cada resultado con lo que cuesta equivocarse.
desc: Marco matemático para elegir la mejor acción cuando el resultado es incierto. Combina la probabilidad de cada posible estado con una función de pérdida que dice cuánto cuesta cada tipo de error, y la decisión óptima es la que minimiza la pérdida esperada. Explica, por ejemplo, por qué el umbral de un clasificador no tiene que ser 0,5 si los errores no cuestan lo mismo.
ej: En un cribado médico, no detectar una enfermedad cuesta mucho más que una falsa alarma, así que conviene marcar como sospechosos casos con probabilidad bastante menor del 50 %.
rel: riesgo-de-bayes, umbral-de-decision, funciones-de-perdida
en: decision theory

## riesgo-de-bayes
frase: El error mínimo que podría lograr cualquier modelo si conociera la distribución real de los datos.
desc: El riesgo es la pérdida esperada de una regla de decisión sobre la distribución real de los datos. El riesgo de Bayes es el menor riesgo alcanzable, el del clasificador que decide con las probabilidades verdaderas. Funciona como cota inferior: si las clases se solapan, ningún modelo puede bajar de ese error, por mucho que se entrene.
ej: Si dos especies de flor tienen exactamente las mismas medidas en una parte de los casos, ningún clasificador basado en esas medidas puede acertarlos todos: ese solapamiento fija un error que no se puede eliminar.
rel: teoria-de-la-decision, minimizacion-del-riesgo-empirico, compromiso-sesgo-varianza
en: Bayes risk

## minimizacion-del-riesgo-empirico
frase: Como no conocemos la distribución real, minimizamos el error medio sobre los datos de entrenamiento.
desc: El riesgo real no se puede calcular porque no conocemos la distribución de los datos. Se aproxima con el riesgo empírico, el error medio sobre la muestra de entrenamiento, y se busca el modelo que lo minimiza. Es el principio detrás de casi todo el entrenamiento supervisado. Su peligro es el sobreajuste: un modelo muy flexible puede tener riesgo empírico casi nulo y generalizar mal, y por eso se añade regularización.
ej: Entrenar una red minimizando la entropía cruzada media sobre las imágenes de entrenamiento es minimización del riesgo empírico.
rel: riesgo-de-bayes, sobreajuste, regularizacion, funciones-de-perdida
en: empirical risk minimization (ERM)

## generalizacion
frase: La capacidad de acertar con datos que el modelo no ha visto durante el entrenamiento.
desc: Es el objetivo real del aprendizaje: que el modelo funcione con casos que no estaban en el entrenamiento. Se mide con datos reservados (validación y test) y depende del equilibrio entre la complejidad del modelo y la cantidad y calidad de los datos. La distancia entre el error de entrenamiento y el de test indica cuánto generaliza.
ej: Un modelo de detección de fraude entrenado con operaciones de 2024 tiene que seguir funcionando con las de 2025.
rel: sobreajuste, infraajuste, entrenamiento-validacion-y-test, validacion-cruzada
en: generalization

## sobreajuste
frase: El modelo memoriza el ruido del entrenamiento y falla con datos nuevos.
desc: Ocurre cuando el modelo es demasiado flexible para los datos disponibles: aprende los patrones reales y también el ruido y las casualidades de la muestra. Se reconoce porque el error de entrenamiento sigue bajando mientras el de validación sube. Se combate con más datos, modelos más simples, regularización, early stopping o data augmentation.
ej: Un árbol de decisión sin límite de profundidad acierta el 100 % de los casos de entrenamiento creando casi una hoja por ejemplo, pero falla con clientes nuevos.
rel: infraajuste, compromiso-sesgo-varianza, regularizacion, curvas-de-aprendizaje
en: overfitting

## infraajuste
frase: El modelo es demasiado simple para captar los patrones de los datos.
desc: El modelo no tiene capacidad suficiente, o no se ha entrenado lo bastante, para representar la relación real. Se nota porque el error es alto tanto en entrenamiento como en validación. Se corrige con modelos más flexibles, mejores variables, menos regularización o más tiempo de entrenamiento.
ej: Ajustar una recta a la relación en forma de U entre temperatura y consumo eléctrico, que sube tanto con el frío como con el calor.
rel: sobreajuste, compromiso-sesgo-varianza, feature-engineering
en: underfitting

## compromiso-sesgo-varianza
frase: Los modelos simples se equivocan de forma sistemática; los complejos, de forma inestable.
desc: El error de un modelo se descompone en sesgo (error por suposiciones demasiado simples), varianza (sensibilidad a la muestra concreta de entrenamiento) y ruido irreducible. Aumentar la complejidad reduce el sesgo pero sube la varianza, y al revés. El mejor modelo es el que equilibra ambos. Los ensembles atacan este dilema: el bagging reduce la varianza y el boosting, el sesgo.
ej: Un k-NN con k = 1 tiene sesgo bajo y varianza alta, porque cambia mucho con cada dato; con k = 100 ocurre lo contrario.
rel: sobreajuste, infraajuste, bagging, boosting
en: bias-variance trade-off

## regularizacion
frase: Penalizar la complejidad del modelo para que generalice mejor.
desc: Conjunto de técnicas que restringen la libertad del modelo para que no se ajuste al ruido. Puede actuar sobre los parámetros (penalizaciones L1 o L2), sobre el entrenamiento (early stopping, dropout), sobre los datos (data augmentation, ruido) o sobre la arquitectura (modelos más pequeños). Su intensidad es un hiperparámetro que se ajusta con validación.
cuando: Cuando el modelo sobreajusta: buen resultado en entrenamiento y claramente peor en validación.
ej: Añadir a la pérdida de una regresión el término λ·Σw² (Ridge) para que ningún coeficiente crezca sin control.
rel: sobreajuste, ridge-lasso-y-elastic-net, regularizacion-en-redes, dropout
en: regularization

## distancia-y-similitud
frase: Formas de medir cuánto se parecen dos datos; la elección cambia el resultado de muchos algoritmos.
desc: Muchos métodos se basan en comparar puntos: k-NN, clustering, detección de anomalías o búsqueda con embeddings. Una distancia mide lo lejos que están dos vectores y una similitud, lo parecidos que son. La elección adecuada depende del tipo de dato y de la escala de las variables, que conviene normalizar antes.
ej: Para comparar textos representados como embeddings se suele usar la similitud coseno; para coordenadas en un mapa, la distancia euclídea.
rel: knn, clustering, deteccion-basada-en-distancia, escalado-de-variables
en: distance & similarity

## distancia-euclidea
frase: La distancia en línea recta entre dos puntos.
desc: Es la raíz cuadrada de la suma de las diferencias al cuadrado en cada dimensión, d(x, y) = √Σ(xᵢ − yᵢ)². Es la distancia por defecto en k-NN y k-means. Es sensible a la escala de las variables y a los valores extremos, y pierde capacidad de discriminar en espacios de muy alta dimensión.
cuando: Con variables numéricas en una escala comparable y no demasiadas dimensiones.
ej: Dos clientes de 30 y 35 años que compran 2 y 4 veces al mes están a √(5² + 2²) ≈ 5,4 unidades; sin escalar las variables, la edad domina la distancia.
rel: distancia-manhattan, escalado-de-variables, k-means, maldicion-de-la-dimensionalidad
py: sklearn.metrics.pairwise.euclidean_distances ; scipy.spatial.distance.euclidean
en: Euclidean distance (L2)

## distancia-manhattan
frase: Suma lo que separa a dos puntos en cada eje, como recorrer calles en cuadrícula.
desc: Suma los valores absolutos de las diferencias, d(x, y) = Σ|xᵢ − yᵢ|, y es menos sensible a valores extremos que la euclídea. Ambas pertenecen a la familia Lp o de Minkowski: p = 1 es Manhattan, p = 2 la euclídea y p → ∞ la de Chebyshev, que solo mira la mayor de las diferencias.
cuando: Con datos que tienen valores extremos o en dimensiones altas, donde suele discriminar mejor que la euclídea.
ej: En una ciudad con calles en cuadrícula, ir 3 manzanas al este y 4 al norte supone recorrer 7 manzanas, aunque en línea recta haya 5.
rel: distancia-euclidea, distancia-y-similitud, knn
py: sklearn.metrics.pairwise.manhattan_distances ; scipy.spatial.distance.minkowski
en: Manhattan distance (L1) & Lp norms

## similitud-coseno
frase: Mide el ángulo entre dos vectores: si apuntan en la misma dirección, se parecen.
desc: Es el producto escalar de dos vectores dividido por el producto de sus normas, y va de −1 (opuestos) a 1 (misma dirección). Ignora la magnitud y solo compara la dirección, lo que la hace ideal para textos y embeddings, donde importa el perfil y no el tamaño. La distancia coseno es 1 menos la similitud.
cuando: Para comparar textos, embeddings o perfiles en los que importa la proporción entre componentes y no su tamaño total.
ej: Dos artículos sobre fútbol, uno corto y otro largo, tienen vectores de palabras de tamaños muy distintos pero una similitud coseno cercana a 1.
rel: embeddings, busqueda-por-similitud, tf-idf, distancia-y-similitud
py: sklearn.metrics.pairwise.cosine_similarity
en: cosine similarity

## distancia-de-mahalanobis
frase: Una distancia que tiene en cuenta la escala y la correlación entre variables.
desc: Mide lo lejos que está un punto de una distribución usando su matriz de covarianza: d = √((x − μ)ᵀ Σ⁻¹ (x − μ)). Equivale a la distancia euclídea después de blanquear los datos, es decir, de quitarles la correlación y normalizar su escala. Es muy útil para detectar anomalías en datos multivariantes con variables correlacionadas.
cuando: Con variables correlacionadas entre sí, sobre todo para detectar casos atípicos multivariantes.
ej: En un control de calidad, una pieza con altura y anchura normales por separado, pero en una combinación inusual, queda lejos según Mahalanobis aunque cada medida parezca correcta.
rel: deteccion-basada-en-distancia, distancia-euclidea, pca
py: scipy.spatial.distance.mahalanobis ; sklearn.covariance.EmpiricalCovariance
en: Mahalanobis distance

## paradigmas-de-aprendizaje
frase: Las maneras de aprender según la información disponible: etiquetas completas, parciales, ninguna o recompensas.
desc: Además de los tres grandes paradigmas, supervisado (datos con etiqueta), no supervisado (sin etiquetas) y por refuerzo (recompensas por acciones), hay situaciones intermedias muy habituales. Se puede aprender con pocas etiquetas, con etiquetas imperfectas, eligiendo qué etiquetar o actualizando el modelo según llegan los datos. El paradigma adecuado depende de qué información tienes y de cuánto cuesta conseguir más.
ej: Una empresa con millones de fotos de producto y solo 500 etiquetadas puede combinar aprendizaje semisupervisado y activo en lugar de etiquetarlo todo.
rel: aprendizaje-supervisado, aprendizaje-no-supervisado, aprendizaje-por-refuerzo, aprendizaje-autosupervisado
en: learning paradigms

## aprendizaje-semisupervisado
frase: Aprender con unos pocos ejemplos etiquetados y muchos sin etiquetar.
desc: Aprovecha la estructura de los datos sin etiquetar para mejorar lo que se aprende con las pocas etiquetas disponibles. Se apoya en suposiciones como que puntos cercanos o del mismo grupo comparten clase. Técnicas típicas son el autoentrenamiento con pseudoetiquetas, la propagación de etiquetas y cluster-then-label.
cuando: Cuando etiquetar es caro (expertos, tiempo) pero tienes abundantes datos sin etiquetar del mismo tipo.
ej: Clasificar radiografías con 200 imágenes revisadas por un radiólogo y 20.000 sin revisar.
hp: threshold: confianza mínima para aceptar una pseudoetiqueta en el autoentrenamiento | max_iter: número máximo de rondas de autoentrenamiento
rel: cluster-then-label, supervision-debil, aprendizaje-activo, aprendizaje-autosupervisado
py: sklearn.semi_supervised.SelfTrainingClassifier ; sklearn.semi_supervised.LabelSpreading
en: semi-supervised learning

## cluster-then-label
frase: Agrupar todos los datos y dar a cada grupo la etiqueta de sus pocos ejemplos conocidos.
desc: Primero se agrupan todos los datos, etiquetados o no, con un algoritmo de clustering, a menudo sobre embeddings. Después cada grupo recibe la etiqueta mayoritaria de los ejemplos etiquetados que contiene, y esa etiqueta se extiende al resto. Funciona bien cuando los grupos naturales de los datos coinciden con las clases.
cuando: Cuando tienes muy pocas etiquetas y los datos forman grupos bien separados que corresponden a las clases.
ej: Agrupar 10.000 fotos de productos en 50 grupos y etiquetar a mano solo 2 o 3 fotos de cada uno para clasificar todo el catálogo.
rel: aprendizaje-semisupervisado, k-means, embeddings
py: sklearn.cluster.KMeans
en: cluster-then-label

## supervision-debil
frase: Aprender con etiquetas imperfectas: ruidosas, incompletas o agregadas.
desc: En muchos problemas reales las etiquetas son imperfectas: contienen errores, son conjuntos de etiquetas posibles o se reducen a proporciones por grupo. La supervisión débil adapta las pérdidas y los modelos a esas situaciones, por ejemplo modelando la probabilidad de que una etiqueta esté mal (matriz de mezcla). También abarca generar etiquetas con reglas o heurísticas baratas y combinarlas.
cuando: Cuando conseguir etiquetas exactas es inviable, pero tienes etiquetas aproximadas, reglas heurísticas o información agregada.
ej: Etiquetar automáticamente reseñas como positivas si tienen 4 o 5 estrellas, sabiendo que parte de esas etiquetas será incorrecta.
rel: etiquetas-ruidosas, etiquetas-parciales, proporciones-de-etiquetas, calidad-y-etiquetado-de-datos
en: weak supervision

## etiquetas-ruidosas
frase: Etiquetas en las que una parte de los ejemplos está mal anotada.
desc: Errores de anotación, criterios cambiantes o el etiquetado automático introducen ruido en las etiquetas, y los modelos flexibles acaban memorizándolo. Se mitiga detectando y limpiando los ejemplos sospechosos, usando pérdidas robustas al ruido o modelando con qué probabilidad se confunde cada clase con otra.
cuando: Cuando sospechas que una parte relevante de las etiquetas es errónea, por ejemplo si vienen de varias personas o de reglas automáticas.
ej: En un conjunto de imágenes médicas anotado por varios técnicos, un 10 % de las etiquetas discrepan; filtrar los casos dudosos mejora el modelo.
rel: supervision-debil, calidad-y-etiquetado-de-datos, sobreajuste
en: noisy labels

## etiquetas-parciales
frase: Para cada ejemplo solo se sabe que su clase está dentro de un conjunto de candidatas.
desc: En lugar de una etiqueta exacta, cada ejemplo trae un conjunto de clases posibles, y una de ellas es la verdadera. El modelo aprende a desambiguar usando la información de todos los ejemplos a la vez. Aparece cuando las anotaciones son ambiguas o se recogen de forma indirecta.
cuando: Cuando los anotadores solo pueden acotar la clase a unas pocas opciones o la etiqueta procede de fuentes ambiguas.
ej: Una foto con el pie "Ana y Luis en la gala": cada cara tiene como candidatos los dos nombres, y viendo muchas fotos el modelo aprende quién es quién.
rel: supervision-debil, etiquetas-ruidosas, proporciones-de-etiquetas
en: partial labels

## proporciones-de-etiquetas
frase: Aprender a clasificar individuos sabiendo solo qué proporción de cada clase hay en cada grupo.
desc: Los datos llegan en bolsas y de cada bolsa solo se conoce la proporción de cada clase, no la etiqueta de cada ejemplo. El modelo se entrena para que sus predicciones, agregadas por bolsa, coincidan con esas proporciones. Es habitual cuando las etiquetas individuales son privadas o solo existen estadísticas agregadas.
cuando: Cuando solo tienes estadísticas agregadas por grupo, como resultados por tienda, zona o lote, y no etiquetas individuales.
ej: Conocer solo el porcentaje de clientes que se dieron de baja en cada tienda y aprender a estimar el riesgo de baja de cada cliente.
rel: supervision-debil, etiquetas-parciales, aprendizaje-semisupervisado
en: learning from label proportions

## aprendizaje-activo
frase: El modelo elige qué ejemplos conviene etiquetar para aprender más con menos etiquetas.
desc: En vez de etiquetar datos al azar, el modelo propone en cada ronda los ejemplos sobre los que tiene más dudas o que más información aportarían; una persona los etiqueta y el modelo se reentrena. Así se alcanza la misma precisión con muchas menos etiquetas. Las estrategias típicas son el muestreo por incertidumbre y por desacuerdo entre modelos.
cuando: Cuando etiquetar es caro, hay muchos datos sin etiquetar y una persona puede ir anotando por rondas.
ej: Un sistema que clasifica documentos legales pide cada semana al abogado que revise solo los 50 documentos más dudosos.
rel: aprendizaje-semisupervisado, human-in-the-loop, incertidumbre
en: active learning

## aprendizaje-online
frase: El modelo se actualiza según llegan los datos, sin reentrenar desde cero.
desc: En el aprendizaje online o incremental, el modelo se actualiza con cada dato o pequeño lote nuevo y no necesita guardar todo el histórico. Es adecuado para flujos continuos de datos y para entornos que cambian, donde el modelo debe adaptarse. El riesgo es que olvide patrones antiguos o reaccione en exceso a los datos recientes.
cuando: Cuando los datos llegan en flujo continuo, no caben en memoria o el comportamiento cambia con el tiempo y el modelo debe adaptarse.
ej: Un filtro de spam que se ajusta cada vez que un usuario marca un correo como spam.
hp: learning_rate: cómo evoluciona el tamaño de paso con las actualizaciones | eta0: tamaño de paso inicial | alpha: fuerza de la regularización
rel: deriva-de-datos, batch-mini-batch-y-sgd, olvido-catastrofico
py: sklearn.linear_model.SGDClassifier ; `modelo.partial_fit(X_lote, y_lote)`
en: online learning
