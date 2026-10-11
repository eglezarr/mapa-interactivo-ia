# Fichas: Detección de anomalías

## anomalia-outlier-ruido-y-novedad
frase: Conceptos cercanos con matices: lo raro que interesa, el valor extremo, la variación sin interés y lo nuevo que aparece.
desc: Una anomalía es una observación que no encaja con el comportamiento normal y suele tener interés, como un fraude o una avería. Outlier es el término estadístico para un valor extremo, que puede ser una anomalía o un error. El ruido es variabilidad aleatoria sin interés que conviene filtrar. Una novedad es un patrón nuevo que no estaba en los datos de entrenamiento: se detecta entrenando solo con datos normales y vigilando lo que llega después.
ej: En los datos de un sensor, un pico aislado por una interferencia es ruido; un patrón de vibración nunca visto antes de una rotura es una novedad que interesa detectar.
rel: outliers, ood, enfoques-deteccion-anomalias, deriva-de-datos
en: anomaly, outlier, noise & novelty

## ood
frase: Detectar entradas muy distintas de los datos de entrenamiento, para las que las predicciones del modelo no son fiables.
desc: Un modelo entrenado con fotos de perros y gatos seguirá respondiendo una de las dos clases, a menudo con mucha confianza, si se le enseña un coche. La detección fuera de distribución identifica esas entradas para rechazarlas o derivarlas a una persona. Se usan la confianza máxima de la softmax, la energía de los logits, la distancia en el espacio de embeddings o modelos de densidad, aunque ningún método es infalible.
cuando: En cualquier sistema desplegado en un entorno abierto, sobre todo en medicina, conducción o seguridad.
ej: Un clasificador de lesiones de la piel rechaza la foto de una uña porque, en su espacio de embeddings, queda lejos de todas las imágenes de entrenamiento.
rel: anomalia-outlier-ruido-y-novedad, deriva-de-datos, incertidumbre, verosimilitud-vs-calidad-de-muestras
en: out-of-distribution (OOD) detection

## enfoques-deteccion-anomalias
frase: Supervisado si hay anomalías etiquetadas, semisupervisado si solo hay datos normales y no supervisado si no hay etiquetas.
desc: Con ejemplos etiquetados de ambos tipos, es un problema de clasificación muy desbalanceado, que solo detecta los tipos de anomalía ya vistos. Con datos solo normales, se modela la normalidad y se marca lo que se aparta de ella (detección de novedades). Sin etiquetas, se supone que las anomalías son pocas y distintas, y se buscan los puntos aislados o de baja densidad. En la práctica se combinan, y las pocas etiquetas disponibles se usan para validar y fijar umbrales.
cuando: Para elegir la estrategia según las etiquetas disponibles y según si las anomalías futuras se parecerán a las pasadas.
ej: Un banco combina un clasificador entrenado con fraudes conocidos y un detector no supervisado que avisa de patrones nunca vistos.
rel: desbalanceo-de-clases, isolation-forest, deteccion-por-reconstruccion-o-prediccion, aprendizaje-semisupervisado
en: supervised, semi-supervised & unsupervised anomaly detection

## deteccion-basada-en-distancia
frase: Un punto es anómalo si está lejos de sus vecinos más cercanos.
desc: Es la idea más directa: calcular para cada punto la distancia a su k-ésimo vecino más cercano, o la media de las distancias a sus k vecinos, y marcar como anómalos los más alejados. Es intuitiva y no supone ninguna distribución, pero depende de la escala de las variables y de la medida de distancia, sufre en alta dimensión y falla cuando hay zonas de densidades muy distintas, algo que resuelve el LOF.
cuando: Con datos numéricos de dimensión moderada y bien escalados.
ej: Marcar como sospechosas las transacciones cuyo importe, hora y comercio las alejan de todas las transacciones habituales del cliente.
hp: n_neighbors: número de vecinos | method: distancia al k-ésimo vecino o media de las distancias
py: pyod.models.knn.KNN ; sklearn.neighbors.NearestNeighbors
rel: knn, lof, maldicion-de-la-dimensionalidad, distancia-y-similitud
en: distance-based anomaly detection

## deteccion-basada-en-probabilidad
frase: Modelar la distribución de los datos normales y marcar como anómalo lo improbable.
desc: Se estima la densidad de los datos con una gaussiana, una mezcla de gaussianas, una KDE u otro modelo, y se consideran anómalos los puntos con densidad muy baja. Con una única gaussiana multivariante equivale a usar la distancia de Mahalanobis. Es interpretable y da una puntuación con sentido probabilístico, pero depende de que el modelo de densidad sea adecuado y de estimarlo bien en alta dimensión.
cuando: Cuando los datos normales siguen una distribución que se puede modelar bien, sobre todo en pocas dimensiones.
ej: Un sistema de control de calidad marca las piezas cuyas medidas son muy improbables según la distribución de las piezas buenas.
hp: contamination: proporción esperada de anomalías, que fija el umbral
py: sklearn.covariance.EllipticEnvelope ; pyod.models.ecod.ECOD
rel: estimacion-de-densidad, gmm, distancia-de-mahalanobis, kde
en: probabilistic anomaly detection

## deteccion-teoria-informacion
frase: Una observación es anómala si aporta mucha información, es decir, si resulta muy sorprendente.
desc: Desde la teoría de la información, lo anómalo que es un punto se mide por su sorpresa, −log p(x): cuanto menos probable, más información aporta. También puede medirse cuánto aumenta la complejidad o la longitud de descripción del conjunto al incluir el punto, o cuánto cambia la entropía de un grupo al quitarlo. Este enfoque unifica la detección por densidad y la detección por compresión.
cuando: Cuando se dispone de un modelo de densidad o de compresión de los datos normales y se quiere una puntuación de rareza con interpretación clara.
ej: En el tráfico de una red, una conexión que añade mucha información a un modelo comprimido del tráfico normal se marca como posible intrusión.
py: `-modelo.score_samples(X)`
rel: informacion-shannon, entropia, deteccion-basada-en-probabilidad, estimacion-de-densidad
en: information-theoretic anomaly detection

## deteccion-por-reconstruccion-o-prediccion
frase: Entrenar un modelo con datos normales y marcar como anómalo lo que no consigue reconstruir o predecir bien.
desc: Un autoencoder o un PCA entrenados solo con datos normales reconstruyen bien lo normal y mal lo anómalo, así que el error de reconstrucción sirve como puntuación. En series temporales, un modelo de predicción aprende la dinámica normal, y un error de predicción grande indica una anomalía. Funciona con datos complejos, como imágenes, señales o registros, sin necesidad de etiquetas de anomalías.
cuando: Con datos complejos (imágenes, señales, series multivariantes) y abundantes ejemplos de funcionamiento normal.
ej: Un autoencoder entrenado con el sonido de motores sanos detecta una avería porque reconstruye mal el ruido de un rodamiento dañado.
hp: contamination: proporción esperada de anomalías | n_components: en PCA, componentes con las que se reconstruye
py: pyod.models.pca.PCA ; pyod.models.auto_encoder.AutoEncoder
rel: autoencoders, pca, prediccion-multipaso, eleccion-del-umbral
en: reconstruction- and prediction-based detection

## lof
frase: Compara la densidad local de cada punto con la de sus vecinos: es anómalo si es mucho menos denso que ellos.
desc: El Local Outlier Factor mide, para cada punto, lo concentrados que están sus vecinos a su alrededor en comparación con lo concentrados que están alrededor de esos mismos vecinos. Un valor cercano a 1 indica un punto normal; uno muy superior, un punto aislado respecto a su zona. Al ser local, detecta anomalías en datos con grupos de densidades distintas, donde un umbral global de distancia fallaría.
cuando: Con datos numéricos de dimensión moderada y grupos de distinta densidad.
ej: En un mapa de precios de alquiler, un piso barato en un barrio caro es anómalo aunque su precio sea normal para el conjunto de la ciudad.
hp: n_neighbors: tamaño de la vecindad | contamination: proporción esperada de anomalías | novelty: permite puntuar datos nuevos tras entrenar
py: sklearn.neighbors.LocalOutlierFactor
rel: deteccion-basada-en-distancia, knn, dbscan, isolation-forest
en: Local Outlier Factor (LOF)

## isolation-forest
frase: Aísla cada punto con cortes aleatorios: las anomalías quedan aisladas con muy pocos cortes.
desc: Construye muchos árboles que dividen los datos con variables y umbrales elegidos al azar hasta aislar cada punto. Como las anomalías son pocas y distintas, quedan aisladas cerca de la raíz, con caminos cortos; los puntos normales, en zonas densas, necesitan muchos cortes. La longitud media del camino da la puntuación. Es rápido, escala a muchos datos y variables y no necesita escalarlas.
cuando: Como primera opción de detección no supervisada en datos tabulares, también con muchos datos y muchas variables.
ej: Detectar transacciones con tarjeta inusuales entre millones de operaciones diarias, sin etiquetas de fraude.
hp: n_estimators: número de árboles | max_samples: muestras con las que se construye cada árbol | contamination: proporción esperada de anomalías
py: sklearn.ensemble.IsolationForest ; pyod.models.iforest.IForest
rel: random-forest, lof, enfoques-deteccion-anomalias, eleccion-del-umbral
en: Isolation Forest

## deteccion-con-modelos-generativos
frase: Usar un modelo generativo de los datos normales para puntuar lo raro que es cada dato nuevo.
desc: Los VAE, los flujos normalizantes y los modelos de difusión aprenden la distribución de los datos normales y permiten puntuar cada dato nuevo por su verosimilitud, por su error de reconstrucción o por la dificultad de eliminarle el ruido. Los flujos y los modelos autorregresivos dan la verosimilitud exacta aunque, paradójicamente, a veces asignan más probabilidad a datos de otro dominio, por lo que se combinan con otras puntuaciones.
cuando: Con datos complejos y de alta dimensión, como imágenes médicas o industriales, y muchos ejemplos normales.
ej: Un modelo de difusión entrenado con resonancias cerebrales sanas "repara" la imagen de un paciente, y la diferencia con el original resalta la lesión.
py: pyod.models.vae.VAE
rel: vae, normalizing-flows, verosimilitud-vs-calidad-de-muestras, deteccion-por-reconstruccion-o-prediccion
en: generative-model-based anomaly detection

## eleccion-del-umbral
frase: Decidir a partir de qué puntuación una observación se marca como anomalía.
desc: Los detectores dan una puntuación continua y hay que fijar un corte. Se puede usar una proporción esperada de anomalías (la contaminación), un percentil de las puntuaciones de los datos normales, reglas estadísticas o, si hay algunas etiquetas, el punto que mejor equilibra detecciones y falsas alarmas en una curva precisión-recall. Conviene revisarlo con el tiempo, porque los datos cambian.
cuando: Siempre que un detector pase a producción y haya que convertir sus puntuaciones en alertas.
ej: Un equipo de seguridad puede revisar 50 alertas al día, así que fija el umbral en el percentil que genera, de media, esa cantidad.
py: numpy.quantile
rel: umbral-de-decision, curva-precision-recall, isolation-forest, deriva-de-datos
en: threshold selection
