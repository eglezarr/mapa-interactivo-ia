# Fichas: Flujo de trabajo en ML

## ciclo-de-vida-ml
frase: Las fases de un proyecto de ML, desde definir el problema hasta mantener el modelo en producción.
desc: Un proyecto típico recorre varias fases: definir el problema y la métrica de éxito, conseguir y preparar los datos, entrenar y validar modelos, desplegarlos y monitorizarlos. No es un proceso lineal: lo que se aprende en cada fase obliga a menudo a volver a las anteriores. La mayor parte del esfuerzo suele ir a los datos, no al algoritmo.
ej: Un modelo de previsión de demanda se reentrena cada mes porque los hábitos de compra cambian; detectar ese cambio también forma parte del ciclo.
rel: linea-base, mlops, deriva-de-datos, eda
en: ML project lifecycle

## linea-base
frase: El modelo más sencillo posible, que marca el resultado mínimo que hay que superar.
desc: Antes de entrenar modelos complejos conviene medir lo que logra una regla trivial: predecir siempre la clase más frecuente, la media o el último valor observado. Si un modelo sofisticado no la supera con claridad, algo falla en los datos, en las variables o en la métrica. También da contexto a los resultados: un 95 % de accuracy no es nada si la línea base ya logra un 94 %.
cuando: Siempre, como primer paso de cualquier proyecto, para saber si el modelo aporta algo.
ej: En una serie de ventas diarias, predecir que mañana se venderá lo mismo que hoy es una línea base difícil de batir.
py: sklearn.dummy.DummyClassifier ; sklearn.dummy.DummyRegressor
rel: ciclo-de-vida-ml, modelos-lineales, metricas-de-evaluacion, accuracy
en: baseline

## datos-y-preprocesado
frase: Preparar los datos para que el modelo pueda aprender de ellos: limpiar, transformar y codificar.
desc: Los modelos esperan números limpios y en escalas comparables, y los datos reales llegan con huecos, errores, categorías de texto y escalas dispares. El preprocesado incluye explorar y auditar los datos, imputar faltantes, tratar atípicos, escalar, codificar categorías y crear variables nuevas. Todo debe ajustarse solo con los datos de entrenamiento para no introducir fugas.
ej: Un dataset de clientes con edades vacías, ingresos en distintas monedas y provincias escritas como texto necesita imputación, conversión y codificación antes de entrenar nada.
py: sklearn.preprocessing
rel: eda, pipelines-de-preprocesado, fuga-de-datos, feature-engineering
en: data preprocessing

## eda
frase: Explorar los datos antes de modelar: distribuciones, relaciones, errores y sorpresas.
desc: El análisis exploratorio resume y visualiza los datos para entenderlos: la distribución de cada variable, las relaciones entre ellas y con el objetivo, los valores faltantes, los duplicados y las anomalías. La auditoría añade comprobaciones de calidad, como rangos imposibles, identificadores repetidos o variables que no estarían disponibles en el momento de predecir. Muchos problemas del modelo se descubren aquí.
cuando: Siempre antes de entrenar, y cada vez que llegan datos nuevos o el modelo se comporta de forma extraña.
ej: Un histograma revela edades de 999 años: un código para "dato desconocido" que habría distorsionado el modelo.
py: pandas.DataFrame.describe ; seaborn.pairplot
rel: calidad-y-etiquetado-de-datos, outliers, valores-faltantes-e-imputacion, correlacion-de-pearson
en: exploratory data analysis (EDA)

## calidad-y-etiquetado-de-datos
frase: Unos datos limpios y bien etiquetados importan más que el algoritmo elegido.
desc: Errores, duplicados, sesgos de muestreo y etiquetas inconsistentes limitan lo que cualquier modelo puede aprender. Etiquetar bien exige instrucciones claras, medir el acuerdo entre anotadores y auditar muestras. Las técnicas de aprendizaje confiado (confident learning) detectan ejemplos probablemente mal etiquetados comparando las etiquetas con las predicciones de un modelo.
cuando: Al construir o heredar un dataset etiquetado, y cuando el modelo se estanca sin razón aparente.
ej: En un dataset de radiografías, dos radiólogos discrepan en el 15 % de los casos: medir y resolver esas discrepancias mejora el modelo más que cambiar de arquitectura.
py: cleanlab.filter.find_label_issues
rel: etiquetas-ruidosas, eda, supervision-debil, aprendizaje-activo
en: data quality & labeling

## valores-faltantes-e-imputacion
frase: Rellenar los huecos de los datos con valores razonables sin distorsionar la información.
desc: Los datos faltantes pueden deberse al azar o a algo informativo, como un campo que solo rellenan ciertos clientes. Las opciones van de eliminar filas o columnas a imputar con la media, la mediana o la moda, con los vecinos más cercanos o con un modelo que predice cada variable a partir de las demás. Añadir una columna indicadora conserva la información de que el dato faltaba.
cuando: Cuando el modelo no admite valores faltantes o hay demasiados huecos para eliminar filas sin perder información.
ej: Imputar la renta que falta de un cliente con la mediana de su provincia y marcar con un indicador que era desconocida.
hp: strategy: cómo se rellena (media, mediana, valor más frecuente o constante) | n_neighbors: en la imputación por vecinos, cuántos se promedian | add_indicator: añade columnas que marcan qué valores faltaban
py: sklearn.impute.SimpleImputer ; sklearn.impute.KNNImputer
rel: eda, pipelines-de-preprocesado, fuga-de-datos, knn
en: missing values & imputation

## outliers
frase: Observaciones muy alejadas del resto: pueden ser errores o los casos más interesantes.
desc: Un valor atípico puede ser un error de medida o de captura, o un caso real y extremo. Se detectan con reglas simples, como quedar a más de 1,5 rangos intercuartílicos de los cuartiles o a más de 3 desviaciones típicas de la media, o con métodos multivariantes. Según su origen se corrigen, se eliminan, se acotan (winsorización) o se mantienen usando modelos y escalados robustos.
cuando: Cuando los extremos pueden distorsionar modelos sensibles, como la regresión lineal o el escalado por media y desviación típica.
ej: Un sueldo de 10 millones en una encuesta salarial puede ser real (un alto directivo) o un error con un cero de más; decidirlo cambia mucho la media y apenas la mediana.
py: scipy.stats.mstats.winsorize ; sklearn.preprocessing.RobustScaler
rel: anomalia-outlier-ruido-y-novedad, eda, escalado-de-variables, correlacion-de-spearman
en: outliers

## escalado-de-variables
frase: Llevar las variables numéricas a rangos comparables para que ninguna domine por su escala.
desc: Muchos algoritmos son sensibles a la escala: los basados en distancias (k-NN, k-means, SVM), los regularizados y las redes neuronales. Si una variable va de 0 a 1 y otra de 0 a 100.000, la segunda dominará. Los métodos habituales son la estandarización, la normalización min-max y el escalado robusto con mediana y rango intercuartílico. Los árboles de decisión no lo necesitan.
cuando: Antes de usar modelos basados en distancias, con regularización o entrenados por descenso por gradiente.
ej: Al agrupar clientes por edad e ingresos anuales sin escalar, k-means solo tendría en cuenta los ingresos.
py: sklearn.preprocessing.StandardScaler ; sklearn.preprocessing.MinMaxScaler ; sklearn.preprocessing.RobustScaler
rel: estandarizacion, normalizacion-min-max, knn, distancia-euclidea
en: feature scaling

## estandarizacion
frase: Restar la media y dividir por la desviación típica: media 0 y desviación 1.
desc: Transforma cada variable con z = (x − μ) / σ, de modo que queda centrada en cero y con desviación típica 1. Es el escalado por defecto para la mayoría de modelos lineales, SVM, PCA y redes neuronales. No acota los valores y es sensible a los atípicos extremos, que inflan la desviación típica.
ej: Una altura de 190 cm en una población con media de 170 cm y desviación de 10 cm se convierte en z = 2.
py: sklearn.preprocessing.StandardScaler
rel: escalado-de-variables, normalizacion-min-max, pca, capas-de-normalizacion
en: standardization (z-score)

## normalizacion-min-max
frase: Reescalar cada variable al intervalo [0, 1] usando su mínimo y su máximo.
desc: Aplica x' = (x − mín) / (máx − mín), con lo que todos los valores quedan entre 0 y 1. Conserva la forma de la distribución y es habitual para los píxeles de las imágenes y las entradas de redes neuronales. Un solo valor extremo comprime al resto en un rango estrecho, y los datos nuevos pueden salirse del intervalo.
ej: Dividir los píxeles de una imagen entre 255 para pasarlos de 0-255 a 0-1 es una normalización min-max.
py: sklearn.preprocessing.MinMaxScaler
rel: escalado-de-variables, estandarizacion, outliers
en: min-max normalization

## transformaciones-de-distribucion
frase: Cambiar la forma de la distribución de una variable, por ejemplo para reducir su asimetría.
desc: Las variables muy asimétricas, como los ingresos o los precios, se comportan mejor tras transformarlas. El logaritmo comprime los valores grandes; Box-Cox y Yeo-Johnson buscan automáticamente la potencia que hace los datos más gaussianos; la transformación por cuantiles fuerza una distribución uniforme o normal. Ayudan a los modelos lineales y reducen el peso de los extremos.
cuando: Con variables muy asimétricas o de colas largas, antes de usar modelos lineales o basados en distancias.
ej: Predecir el logaritmo del precio de una vivienda en lugar del precio equilibra el error entre pisos baratos y caros.
hp: method: en la transformación de potencia, Box-Cox (solo valores positivos) o Yeo-Johnson | output_distribution: en la de cuantiles, la distribución de salida (uniforme o normal)
py: sklearn.preprocessing.PowerTransformer ; sklearn.preprocessing.QuantileTransformer ; numpy.log1p
rel: escalado-de-variables, outliers, regresion-lineal, rbig
en: distribution transforms (log, Box-Cox, quantile)

## codificacion-de-variables-categoricas
frase: Convertir categorías de texto en números que el modelo pueda usar sin inventar un orden falso.
desc: One-hot crea una columna binaria por categoría y es la opción segura para categorías sin orden. La codificación ordinal asigna enteros y solo tiene sentido si hay un orden real o con modelos de árboles. La codificación por objetivo (target encoding) sustituye cada categoría por la media del objetivo en ella; es útil con muchas categorías, pero debe calcularse con validación cruzada para no filtrar información.
cuando: Siempre que haya variables categóricas; target encoding, cuando tienen cientos o miles de categorías.
ej: El color ("rojo", "verde", "azul") se codifica en tres columnas binarias; la talla ("S", "M", "L"), como 0, 1 y 2.
hp: handle_unknown: qué hacer con categorías que no aparecían en entrenamiento | min_frequency: agrupa las categorías poco frecuentes en una sola
py: sklearn.preprocessing.OneHotEncoder ; sklearn.preprocessing.OrdinalEncoder ; sklearn.preprocessing.TargetEncoder
rel: feature-engineering, fuga-de-datos, capa-de-embedding, pipelines-de-preprocesado
en: categorical encoding (one-hot, ordinal, target)

## feature-engineering
frase: Crear variables nuevas a partir de las existentes para que el modelo vea mejor el patrón.
desc: Consiste en transformar y combinar los datos brutos usando conocimiento del dominio: ratios, diferencias, agregados por grupo, variables de calendario, tramos o interacciones. En datos tabulares suele marcar más diferencia que cambiar de modelo. El deep learning automatiza parte de este trabajo al aprender representaciones, sobre todo en imágenes, texto y audio.
cuando: Con datos tabulares, sobre todo cuando conoces el dominio y el modelo no capta relaciones que sabes que existen.
ej: A partir de la fecha de una compra, crear "día de la semana", "es festivo" y "días desde la última compra" para predecir la siguiente.
py: sklearn.preprocessing.FunctionTransformer ; sklearn.preprocessing.KBinsDiscretizer
rel: regresion-polinomica-e-interacciones, codificacion-de-variables-categoricas, seleccion-de-caracteristicas, aprendizaje-de-representaciones
en: feature engineering

## desbalanceo-de-clases
frase: Cuando una clase es mucho más rara que otra y el modelo tiende a ignorarla.
desc: En fraude, averías o enfermedades raras, la clase de interés puede ser el 1 % de los datos o menos, y un modelo puede lograr un 99 % de accuracy sin detectar ni un caso. Las soluciones combinan métricas adecuadas (recall, precisión, curva precisión-recall), el ajuste del umbral de decisión, la ponderación de clases y el remuestreo de los datos de entrenamiento.
ej: En un detector de transacciones fraudulentas con un 0,2 % de fraudes, predecir siempre "legítima" da un 99,8 % de accuracy y no sirve para nada.
rel: sobremuestreo, submuestreo-y-ponderacion-de-clases, curva-precision-recall, umbral-de-decision
en: class imbalance

## sobremuestreo
frase: Aumentar los ejemplos de la clase minoritaria, duplicándolos o creando otros sintéticos.
desc: El sobremuestreo aleatorio duplica ejemplos de la clase rara. SMOTE crea ejemplos sintéticos interpolando entre un ejemplo minoritario y sus vecinos de la misma clase, y ADASYN genera más ejemplos donde la clase es más difícil de aprender. Solo debe aplicarse al conjunto de entrenamiento, nunca a validación ni a test.
cuando: Cuando la clase minoritaria tiene pocos ejemplos y la ponderación de clases no basta.
ej: Con 200 casos de avería frente a 20.000 normales, SMOTE genera averías sintéticas hasta equilibrar el entrenamiento del clasificador.
hp: sampling_strategy: proporción final deseada entre clases | k_neighbors: vecinos entre los que SMOTE interpola
py: imblearn.over_sampling.SMOTE ; imblearn.over_sampling.ADASYN ; imblearn.over_sampling.RandomOverSampler
rel: desbalanceo-de-clases, submuestreo-y-ponderacion-de-clases, fuga-de-datos, data-augmentation
en: oversampling (SMOTE, ADASYN)

## submuestreo-y-ponderacion-de-clases
frase: Equilibrar las clases quitando ejemplos de la mayoritaria o dando más peso a los errores en la minoritaria.
desc: El submuestreo elimina ejemplos de la clase mayoritaria, al azar o de forma informada, por ejemplo los pegados a la frontera entre clases (enlaces de Tomek). Es rápido, pero descarta información. La ponderación de clases no toca los datos: multiplica la pérdida de cada ejemplo por un peso, mayor en la clase rara, y la mayoría de modelos la admiten directamente.
cuando: Ponderar, como primera opción ante clases desbalanceadas; submuestrear, cuando además sobran datos de la clase mayoritaria.
ej: Con pesos balanceados, si el fraude es 500 veces menos frecuente, fallar un fraude cuesta en la pérdida lo mismo que fallar 500 transacciones legítimas.
hp: sampling_strategy: proporción final entre clases tras submuestrear | class_weight: pesos por clase; "balanced" los hace inversamente proporcionales a su frecuencia
py: imblearn.under_sampling.RandomUnderSampler ; imblearn.under_sampling.TomekLinks ; sklearn.utils.class_weight.compute_class_weight
rel: desbalanceo-de-clases, sobremuestreo, funciones-de-perdida, umbral-de-decision
en: undersampling & class weighting

## fuga-de-datos
frase: Cuando el modelo usa al entrenar información que no tendrá al predecir.
desc: Produce resultados de validación excelentes que se desploman en producción. Las fugas típicas son variables que contienen la respuesta o solo se conocen después, escalar o imputar con estadísticas de todo el dataset antes de dividirlo, ejemplos del mismo paciente o cliente repartidos entre entrenamiento y test, y usar datos del futuro en series temporales. Se previene con pipelines y particiones bien diseñadas.
ej: Un modelo que predice si un paciente tiene neumonía usa la variable "recibió antibiótico", que solo se conoce después del diagnóstico.
py: sklearn.pipeline.make_pipeline ; sklearn.model_selection.GroupKFold
rel: pipelines-de-preprocesado, entrenamiento-validacion-y-test, validacion-en-series-temporales, validacion-cruzada
en: data leakage

## data-augmentation
frase: Crear variaciones realistas de los datos de entrenamiento para que el modelo generalice mejor.
desc: Genera ejemplos nuevos aplicando transformaciones que no cambian la etiqueta: girar, recortar o cambiar el brillo de imágenes, añadir ruido a audios o parafrasear textos. Actúa como regularización y enseña al modelo invariancias útiles. Las transformaciones deben ser realistas para el problema: voltear horizontalmente una imagen no sirve si se trata de reconocer letras.
cuando: Con pocos datos, sobre todo en imágenes, audio y texto, o cuando quieres robustez frente a variaciones conocidas.
ej: Para clasificar fotos de hojas enfermas tomadas con el móvil, aplicar giros, recortes y cambios de luz multiplica la variedad de cada foto.
hp: factor: en las capas de aumento de Keras, intensidad máxima de la transformación (rotación, zoom, contraste) | scale: en un recorte aleatorio, fracción del área de la imagen que se conserva
py: keras.layers.RandomFlip ; keras.layers.RandomRotation ; torchvision.transforms.v2.RandomResizedCrop
rel: regularizacion-en-redes, sobremuestreo, equivarianza-e-invarianza, datos-sinteticos
en: data augmentation

## pipelines-de-preprocesado
frase: Encadenar preprocesado y modelo en un único objeto que se ajusta y se aplica de una vez.
desc: Un pipeline agrupa todos los pasos, como imputar, escalar, codificar y modelar, y garantiza que cada transformación se ajusta solo con los datos de entrenamiento de cada partición. Así se evitan fugas en la validación cruzada, se pueden ajustar a la vez los hiperparámetros del preprocesado y del modelo, y se despliega un único objeto. ColumnTransformer aplica pasos distintos a las columnas numéricas y a las categóricas.
cuando: Siempre que haya preprocesado antes del modelo, en particular al validar o al desplegar.
ej: Un pipeline que imputa y escala las columnas numéricas, aplica one-hot a las categóricas y termina en un gradient boosting, guardado como un único archivo.
py: sklearn.pipeline.Pipeline ; sklearn.compose.ColumnTransformer
rel: fuga-de-datos, validacion-cruzada, despliegue-e-inferencia, codificacion-de-variables-categoricas
en: preprocessing pipelines

## particion-y-validacion
frase: Separar los datos para medir con honestidad cómo funcionará el modelo con casos nuevos.
desc: Evaluar un modelo con los mismos datos con los que se entrenó da una idea falsamente optimista. Por eso se reservan datos que el modelo no ve: una partición de validación para elegir modelos e hiperparámetros, otra de test para la estimación final, y técnicas como la validación cruzada para aprovechar mejor los datos escasos. La partición debe imitar cómo llegarán los datos en producción.
ej: Un modelo de crédito que se usará con clientes futuros se valida con los clientes de los últimos meses, no con una muestra aleatoria de todo el histórico.
py: sklearn.model_selection
rel: entrenamiento-validacion-y-test, validacion-cruzada, validacion-en-series-temporales, generalizacion
en: data splitting & validation

## entrenamiento-validacion-y-test
frase: Tres particiones: una para aprender, otra para elegir y otra para la nota final.
desc: Con el conjunto de entrenamiento se ajustan los parámetros; con el de validación se comparan modelos y se ajustan hiperparámetros; el de test se usa una sola vez, al final, para estimar el rendimiento real. Si el test se consulta repetidamente para decidir, deja de ser una estimación honesta. Proporciones típicas son 70/15/15 u 80/10/10, estratificando por clase si hay desbalanceo.
cuando: En cualquier proyecto con datos suficientes; con pocos datos, la partición de validación se sustituye por validación cruzada.
ej: Con 10.000 imágenes: 8.000 para entrenar, 1.000 para elegir la arquitectura y 1.000 guardadas para el resultado final.
py: sklearn.model_selection.train_test_split
rel: validacion-cruzada, sobreajuste, fuga-de-datos, early-stopping
en: train / validation / test split

## validacion-cruzada
frase: Rotar la partición de validación k veces para estimar el rendimiento de forma más estable.
desc: Divide los datos en k bloques, entrena k veces dejando cada vez un bloque fuera para validar y promedia los resultados. Aprovecha todos los datos y da una medida de la variabilidad. La versión estratificada mantiene la proporción de clases en cada bloque, y la agrupada mantiene juntos los ejemplos de una misma entidad, como un paciente.
cuando: Para comparar modelos o ajustar hiperparámetros con datos limitados, cuando una sola partición sería demasiado ruidosa.
ej: Con 5 bloques, un modelo obtiene un AUC de 0,84 ± 0,02: el promedio es más fiable que el de una única partición y la desviación indica su estabilidad.
hp: n_splits: número de bloques (k) | shuffle: barajar los datos antes de dividirlos
py: sklearn.model_selection.cross_val_score ; sklearn.model_selection.KFold ; sklearn.model_selection.StratifiedKFold ; sklearn.model_selection.GroupKFold
rel: entrenamiento-validacion-y-test, validacion-en-series-temporales, busqueda-en-rejilla-y-aleatoria, fuga-de-datos
en: cross-validation (k-fold)

## validacion-en-series-temporales
frase: Validar siempre con datos posteriores a los de entrenamiento, respetando el orden del tiempo.
desc: En series temporales, barajar al azar permite entrenar con el futuro y validar con el pasado, lo que infla los resultados. Se usa una validación con origen móvil: se entrena hasta una fecha, se valida en el periodo siguiente y se avanza. Conviene dejar un hueco entre ambos periodos cuando las variables se calculan con ventanas que se solapan.
cuando: Siempre que los datos tengan orden temporal y el modelo vaya a predecir el futuro.
ej: Para un modelo de demanda, entrenar con 2022 y validar con enero de 2023; después, entrenar hasta enero y validar con febrero, y así sucesivamente.
hp: n_splits: número de cortes temporales | max_train_size: limita la ventana de entrenamiento (ventana deslizante) | gap: observaciones que se descartan entre entrenamiento y validación
py: sklearn.model_selection.TimeSeriesSplit
rel: validacion-cruzada, fuga-de-datos, prediccion-multipaso, deriva-de-datos
en: time series cross-validation (walk-forward)

## metricas-de-evaluacion
frase: Números que resumen lo bien que funciona un modelo según lo que importa en el problema.
desc: La métrica traduce el objetivo del negocio o de la investigación en un número con el que comparar modelos. Cada una mide algo distinto: el error medio, el peor caso, cuántos positivos se detectan o con cuánta fiabilidad. Elegir mal la métrica lleva a optimizar algo que no importa, como la accuracy con clases desbalanceadas.
ej: En un filtro de spam importa no perder correos legítimos (precisión alta); en un cribado de cáncer, no dejar escapar casos (recall alto).
py: sklearn.metrics
rel: funcion-de-perdida-vs-metrica, metricas-de-regresion, metricas-de-clasificacion, linea-base
en: evaluation metrics

## funcion-de-perdida-vs-metrica
frase: La pérdida guía el entrenamiento; la métrica mide lo que de verdad importa.
desc: La función de pérdida es lo que el optimizador minimiza, y debe ser diferenciable y bien comportada, como la entropía cruzada. La métrica es lo que se reporta y se usa para decidir, como el F1 o el error en euros, y no necesita ser diferenciable. No tienen por qué coincidir: se puede entrenar con entropía cruzada y elegir el modelo por su recall.
ej: Un clasificador de fraude se entrena minimizando la entropía cruzada, pero se evalúa por el importe de fraude detectado por cada 100 alertas.
py: `model.compile(loss="binary_crossentropy", metrics=["auc"])`
rel: funciones-de-perdida, metricas-de-evaluacion, entropia-cruzada, umbral-de-decision
en: loss function vs metric

## metricas-de-regresion
frase: Miden lo lejos que quedan las predicciones numéricas de los valores reales.
desc: Las más usadas son el MSE y su raíz (RMSE), que penalizan mucho los errores grandes; el MAE, más robusto y fácil de interpretar; el MAPE, en porcentaje, que falla cuando los valores reales son cercanos a cero; y el R², que compara el modelo con predecir siempre la media. Conviene expresarlas en las unidades del problema.
ej: Un modelo de precios de vivienda con un MAE de 12.000 € se equivoca, de media, en 12.000 € por vivienda.
py: sklearn.metrics.mean_absolute_percentage_error ; sklearn.metrics.median_absolute_error
rel: mse-rmse-y-mae, r2, perdidas-de-regresion, funcion-de-perdida-vs-metrica
en: regression metrics

## mse-rmse-y-mae
frase: Error cuadrático medio, su raíz y error absoluto medio: las métricas básicas de regresión.
desc: El MSE promedia los errores al cuadrado, por lo que castiga mucho los errores grandes; su raíz, el RMSE, vuelve a las unidades de la variable. El MAE promedia los errores en valor absoluto: es más robusto a los atípicos y se interpreta directamente como el error típico. Minimizar el MSE lleva a predecir la media condicional; minimizar el MAE, la mediana.
cuando: RMSE, cuando los errores grandes son especialmente graves; MAE, cuando hay atípicos o se quiere un error fácil de explicar.
ej: Errores de 1, 1, 1 y 9 °C dan un MAE de 3 °C y un RMSE de 4,6 °C: el RMSE delata el error grande.
py: sklearn.metrics.mean_squared_error ; sklearn.metrics.root_mean_squared_error ; sklearn.metrics.mean_absolute_error
rel: metricas-de-regresion, perdidas-de-regresion, r2, outliers
en: MSE, RMSE & MAE

## r2
frase: Qué fracción de la variabilidad de los datos explica el modelo frente a predecir siempre la media.
desc: R² = 1 − SSE/SST compara el error del modelo con el de una predicción constante igual a la media. Vale 1 si el ajuste es perfecto y 0 si el modelo no mejora a la media; con datos nuevos puede ser negativo. Un R² alto no garantiza un buen modelo: puede haber sobreajuste, fuga de datos o errores grandes justo en los casos que importan.
ej: Un R² de 0,75 en un modelo de consumo eléctrico indica que explica el 75 % de la variación del consumo entre hogares.
py: sklearn.metrics.r2_score
rel: metricas-de-regresion, mse-rmse-y-mae, regresion-lineal, linea-base
en: coefficient of determination (R²)

## metricas-de-clasificacion
frase: Miden los aciertos y los distintos tipos de error de un clasificador.
desc: Parten de la matriz de confusión, que cuenta aciertos y errores por clase. De ella salen la exactitud, la precisión, el recall y el F1, que dependen de un umbral de decisión, y las curvas ROC y precisión-recall, que resumen el rendimiento en todos los umbrales. La elección depende del coste de cada tipo de error y del desbalanceo entre clases.
ej: En un cribado médico, un falso negativo (un enfermo no detectado) es mucho más grave que un falso positivo, que solo implica una prueba adicional.
py: sklearn.metrics.classification_report
rel: matriz-de-confusion, precision-y-recall, curva-roc-y-auc, desbalanceo-de-clases
en: classification metrics

## matriz-de-confusion
frase: Tabla que cruza las clases reales con las predichas y muestra cada tipo de acierto y de error.
desc: En clasificación binaria tiene cuatro casillas: verdaderos positivos, falsos positivos, falsos negativos y verdaderos negativos. Con más clases, muestra qué clases se confunden entre sí. Es el punto de partida de casi todas las métricas de clasificación y la forma más directa de ver dónde se equivoca el modelo.
ej: Un clasificador de dígitos manuscritos que confunde a menudo el 4 con el 9 lo muestra como una casilla destacada fuera de la diagonal.
py: sklearn.metrics.confusion_matrix ; sklearn.metrics.ConfusionMatrixDisplay
rel: metricas-de-clasificacion, precision-y-recall, accuracy, umbral-de-decision
en: confusion matrix

## accuracy
frase: Proporción de predicciones correctas; engañosa cuando las clases están desbalanceadas.
desc: La exactitud divide los aciertos entre el total de casos. Es intuitiva, pero con clases desbalanceadas premia ignorar la clase rara. La exactitud balanceada promedia el recall de cada clase, de modo que todas cuentan lo mismo y una predicción constante obtiene 1/K (0,5 en un problema binario).
ej: Si el 95 % de los pacientes están sanos, decir siempre "sano" da un 95 % de accuracy, pero solo un 50 % de exactitud balanceada.
py: sklearn.metrics.accuracy_score ; sklearn.metrics.balanced_accuracy_score
rel: matriz-de-confusion, desbalanceo-de-clases, precision-y-recall, linea-base
en: accuracy & balanced accuracy

## precision-y-recall
frase: Precisión: cuántas alarmas son ciertas. Recall: cuántos casos reales se detectan.
desc: La precisión es TP / (TP + FP): de lo que el modelo marca como positivo, qué fracción lo es. El recall, o sensibilidad, es TP / (TP + FN): de los positivos reales, qué fracción encuentra. Suelen estar en tensión: bajar el umbral detecta más casos (más recall) a costa de más falsas alarmas (menos precisión).
cuando: Precisión, cuando las falsas alarmas son caras; recall, cuando perder un caso positivo es grave.
ej: Un detector de fraude marca 100 operaciones y 80 son fraude: precisión del 80 %. Si había 200 fraudes en total, su recall es del 40 %.
py: sklearn.metrics.precision_score ; sklearn.metrics.recall_score
rel: f1-score, curva-precision-recall, umbral-de-decision, matriz-de-confusion
en: precision & recall

## f1-score
frase: La media armónica de precisión y recall: solo es alta si ambas lo son.
desc: F1 = 2·P·R / (P + R). Al ser una media armónica, se hunde si una de las dos es baja, así que resume en un solo número el equilibrio entre ambas. El Fβ generaliza la idea dando β veces más importancia al recall que a la precisión. Depende del umbral de decisión y no tiene en cuenta los verdaderos negativos.
cuando: Para resumir en un número un problema desbalanceado en el que importan tanto la precisión como el recall.
ej: Con una precisión de 0,9 y un recall de 0,1, la media aritmética sería 0,5, pero el F1 es solo 0,18.
py: sklearn.metrics.f1_score ; sklearn.metrics.fbeta_score
rel: precision-y-recall, promedios-macro-y-micro, curva-precision-recall, umbral-de-decision
en: F1-score

## curva-roc-y-auc
frase: Rendimiento del clasificador en todos los umbrales: tasa de aciertos frente a tasa de falsas alarmas.
desc: La curva ROC representa la tasa de verdaderos positivos frente a la de falsos positivos al mover el umbral de decisión. El área bajo la curva (AUC) resume el rendimiento en un número entre 0,5 (azar) y 1 (perfecto), y equivale a la probabilidad de que un positivo elegido al azar reciba más puntuación que un negativo elegido al azar. Con clases muy desbalanceadas puede resultar demasiado optimista.
cuando: Para comparar clasificadores por su capacidad de ordenar los casos, con independencia del umbral, cuando las clases están razonablemente equilibradas.
ej: Un modelo de riesgo de impago con un AUC de 0,80 ordena correctamente el 80 % de los pares formados por un cliente que impaga y otro que paga.
py: sklearn.metrics.roc_curve ; sklearn.metrics.roc_auc_score ; sklearn.metrics.RocCurveDisplay
rel: curva-precision-recall, umbral-de-decision, metricas-de-clasificacion, desbalanceo-de-clases
en: ROC curve & AUC

## curva-precision-recall
frase: Precisión frente a recall en todos los umbrales: la curva útil cuando los positivos son raros.
desc: Muestra cómo cae la precisión a medida que se exige más recall al bajar el umbral. Su área, la precisión media (AP), tiene como referencia la proporción de positivos y no 0,5, por lo que es mucho más informativa que la ROC cuando la clase positiva es rara. Permite elegir el umbral según el recall mínimo o la precisión que se necesiten.
cuando: Con clases muy desbalanceadas, cuando lo que importa es la calidad de las detecciones positivas.
ej: En detección de fraude con un 0,5 % de positivos, un modelo con un AUC ROC de 0,95 puede tener una precisión media de solo 0,30, lo que revela muchas falsas alarmas.
py: sklearn.metrics.precision_recall_curve ; sklearn.metrics.average_precision_score ; sklearn.metrics.PrecisionRecallDisplay
rel: curva-roc-y-auc, precision-y-recall, desbalanceo-de-clases, umbral-de-decision
en: precision-recall curve (average precision)

## umbral-de-decision
frase: El corte de probabilidad a partir del cual el modelo decide "positivo"; 0,5 rara vez es el óptimo.
desc: Los clasificadores dan una puntuación o una probabilidad, y el umbral la convierte en decisión. Moverlo cambia el equilibrio entre falsos positivos y falsos negativos sin reentrenar. Se elige con datos de validación según el coste de cada error o un objetivo mínimo, como un recall del 95 %. Si las probabilidades deben interpretarse como tales, conviene calibrarlas.
cuando: Siempre que los dos tipos de error tengan costes distintos o las clases estén desbalanceadas.
ej: Si dejar pasar un fraude cuesta 50 veces más que revisar una alerta, con probabilidades bien calibradas el umbral óptimo es 1/51, alrededor de 0,02.
hp: scoring: métrica que se maximiza al buscar el umbral | thresholds: umbrales candidatos que se prueban
py: sklearn.model_selection.TunedThresholdClassifierCV ; sklearn.model_selection.FixedThresholdClassifier
rel: precision-y-recall, curva-roc-y-auc, curva-precision-recall, teoria-de-la-decision
en: decision threshold

## promedios-macro-y-micro
frase: Cómo combinar en un solo número una métrica calculada por clase cuando hay varias clases.
desc: El promedio macro calcula la métrica en cada clase y hace la media simple, así que todas las clases pesan igual, aunque sean raras. El micro suma los aciertos y errores de todas las clases antes de calcular, así que dominan las clases frecuentes. El ponderado hace la media por clase pesando por su número de ejemplos.
cuando: Macro, cuando las clases minoritarias importan tanto como las frecuentes; micro, cuando importa el rendimiento global por ejemplo.
ej: Un clasificador de incidencias acierta casi siempre en "consulta" (el 90 % de los casos) y falla en "urgente": su F1 micro es alto y su F1 macro lo delata.
py: `f1_score(y_true, y_pred, average="macro")`
rel: f1-score, clasificacion-multiclase-y-multietiqueta, desbalanceo-de-clases, metricas-de-clasificacion
en: macro & micro averaging

## ajuste-de-hiperparametros-y-automl
frase: Buscar de forma sistemática la mejor configuración del modelo, o incluso el mejor modelo.
desc: Los hiperparámetros, como la profundidad de un árbol o el learning rate, no se aprenden de los datos: se eligen comparando configuraciones en validación. Las estrategias van de la búsqueda en rejilla y aleatoria a la optimización bayesiana. AutoML automatiza además la elección del preprocesado y del algoritmo, y NAS, el diseño de la arquitectura de una red.
ej: Probar 50 combinaciones de profundidad, número de árboles y learning rate de un gradient boosting y quedarse con la de mejor AUC en validación cruzada.
rel: parametros-e-hiperparametros, busqueda-en-rejilla-y-aleatoria, optimizacion-bayesiana-de-hiperparametros, automl
en: hyperparameter tuning & AutoML

## busqueda-en-rejilla-y-aleatoria
frase: Probar todas las combinaciones de una rejilla, o un número fijo de combinaciones al azar.
desc: La búsqueda en rejilla evalúa todas las combinaciones de unos valores prefijados, y su coste crece exponencialmente con el número de hiperparámetros. La aleatoria muestrea combinaciones de distribuciones definidas y, con el mismo presupuesto, suele encontrar mejores resultados, porque explora más valores de los hiperparámetros que de verdad importan. Ambas evalúan cada combinación con validación cruzada.
cuando: Rejilla, con pocos hiperparámetros y valores discretos; aleatoria, como opción por defecto con un presupuesto limitado.
ej: En lugar de probar las 5 × 5 × 5 = 125 combinaciones de una rejilla, probar 30 al azar suele dar un resultado casi igual de bueno.
hp: n_iter: en la búsqueda aleatoria, número de combinaciones que se prueban | cv: esquema de validación cruzada | scoring: métrica que se optimiza
py: sklearn.model_selection.GridSearchCV ; sklearn.model_selection.RandomizedSearchCV
rel: optimizacion-bayesiana-de-hiperparametros, validacion-cruzada, parametros-e-hiperparametros, automl
en: grid & random search

## optimizacion-bayesiana-de-hiperparametros
frase: Elegir la siguiente configuración que se prueba aprendiendo de los resultados anteriores.
desc: Construye un modelo sustituto, como un proceso gaussiano o un estimador de Parzen en árbol (TPE), que predice el rendimiento de cada configuración a partir de las ya evaluadas. Una función de adquisición decide dónde probar después, equilibrando explorar zonas inciertas y explotar las prometedoras. Necesita muchas menos evaluaciones que la búsqueda aleatoria cuando cada entrenamiento es caro.
cuando: Cuando cada entrenamiento es costoso y hay varios hiperparámetros continuos que ajustar.
ej: Ajustar el learning rate, el dropout y el tamaño de las capas de una red con 40 entrenamientos en lugar de cientos.
hp: direction: si la métrica se maximiza o se minimiza | sampler: algoritmo que propone la siguiente prueba (TPE, procesos gaussianos…) | pruner: corta pronto las pruebas poco prometedoras
py: optuna.create_study ; optuna.samplers.TPESampler
rel: busqueda-en-rejilla-y-aleatoria, procesos-gaussianos, exploracion-vs-explotacion, automl
en: Bayesian hyperparameter optimization

## automl
frase: Automatizar la elección del preprocesado, del algoritmo y de sus hiperparámetros.
desc: Un sistema de AutoML recibe los datos y una métrica y busca por sí solo la mejor combinación de preprocesado, modelo e hiperparámetros dentro de un presupuesto de tiempo, a menudo combinando los mejores modelos en un ensemble. Es útil como punto de partida rápido o como línea base fuerte, pero no sustituye entender los datos ni evitar fugas.
cuando: Para obtener rápidamente una línea base fuerte en datos tabulares o cuando no hay tiempo de ajustar a mano.
ej: Darle a una herramienta de AutoML una tabla de clientes y una hora de cómputo, y obtener un ensemble de modelos de gradient boosting ya ajustado.
hp: time_budget: tiempo máximo de búsqueda en segundos | metric: métrica que se optimiza
py: flaml.AutoML ; autosklearn.classification.AutoSklearnClassifier
rel: ajuste-de-hiperparametros-y-automl, nas, ensembles, linea-base
en: AutoML

## nas
frase: Buscar automáticamente la mejor arquitectura de red neuronal para un problema.
desc: La búsqueda de arquitecturas neuronales explora un espacio de diseños posibles (número y tipo de capas, conexiones, anchuras) con estrategias como algoritmos evolutivos, aprendizaje por refuerzo, optimización bayesiana o métodos diferenciables. Es muy costosa, pero ha producido arquitecturas eficientes como EfficientNet, cuya red base se obtuvo así.
cuando: Cuando el cómputo es asumible y la arquitectura óptima no es obvia, o hay restricciones fuertes de latencia o memoria.
ej: Buscar la red de clasificación de imágenes más precisa que quepa en un móvil y responda en menos de 20 ms.
hp: max_trials: número de arquitecturas distintas que se prueban
py: autokeras.ImageClassifier
rel: automl, mobilenet-y-efficientnet, optimizacion-bayesiana-de-hiperparametros, arquitecturas-cnn-de-referencia
en: neural architecture search (NAS)

## despliegue-y-mlops
frase: Llevar el modelo a producción y mantenerlo funcionando bien con el paso del tiempo.
desc: Un modelo solo aporta valor cuando se usa. Esta parte cubre cómo servir predicciones, por lotes o en tiempo real, versionar datos y modelos, automatizar el reentrenamiento y vigilar que el rendimiento no se degrade. MLOps aplica a los modelos las prácticas de la ingeniería de software: automatización, pruebas y monitorización.
ej: Un modelo de recomendación se reentrena cada noche con los datos del día y solo se publica si supera al anterior en una prueba automática.
rel: mlops, despliegue-e-inferencia, deriva-de-datos, seguimiento-de-experimentos
en: deployment & MLOps

## mlops
frase: Prácticas y herramientas para entrenar, desplegar y mantener modelos de forma fiable y repetible.
desc: Extiende DevOps al aprendizaje automático: versiona código, datos y modelos; automatiza el entrenamiento, la evaluación y el despliegue (CI/CD); registra cada modelo con sus métricas; y monitoriza en producción para reentrenar cuando haga falta. Su objetivo es que llevar un modelo a producción y actualizarlo sea rutinario, trazable y reproducible.
cuando: En cuanto un modelo pasa a producción y hay que actualizarlo, auditarlo o mantener varios a la vez.
ej: Un registro de modelos guarda cada versión del modelo de riesgo con sus datos de entrenamiento, sus métricas y quién aprobó su paso a producción.
py: mlflow.register_model
rel: despliegue-e-inferencia, deriva-de-datos, seguimiento-de-experimentos, reproducibilidad-y-semillas
en: MLOps

## despliegue-e-inferencia
frase: Poner el modelo a disposición de otros sistemas para que haga predicciones reales.
desc: El modelo se guarda (se serializa) y se sirve por lotes, con predicciones programadas sobre muchos datos, o en tiempo real, normalmente detrás de una API. En tiempo real importan la latencia, el coste y la escalabilidad, y se recurre a formatos optimizados como ONNX, a la cuantización o a hardware específico. Hay que garantizar que el preprocesado en producción sea idéntico al del entrenamiento.
cuando: Cuando el modelo validado debe empezar a usarse en una aplicación, un proceso o un informe.
ej: Un modelo de detección de fraude publicado como API que responde en menos de 50 ms a cada pago con tarjeta.
py: joblib.dump ; fastapi.FastAPI ; onnxruntime.InferenceSession
rel: mlops, pipelines-de-preprocesado, cuantizacion, gpu-y-tpu
en: deployment & inference

## deriva-de-datos
frase: Los datos de producción cambian respecto a los de entrenamiento y el modelo se degrada en silencio.
desc: La deriva de datos (data drift) es un cambio en la distribución de las entradas; la deriva de concepto, un cambio en la relación entre las entradas y la respuesta. Ambas degradan el modelo sin que nada falle de forma visible. Se vigilan comparando distribuciones con tests estadísticos o índices como el PSI y, cuando llegan las etiquetas reales, siguiendo las métricas de rendimiento.
cuando: En todo modelo en producción, sobre todo en entornos cambiantes como el consumo, el fraude o los mercados.
ej: Los modelos de demanda entrenados antes de la pandemia fallaron en 2020 porque los hábitos de compra cambiaron de golpe.
py: scipy.stats.ks_2samp ; scipy.spatial.distance.jensenshannon
rel: mlops, ood, validacion-en-series-temporales, aprendizaje-online
en: data drift & concept drift

## herramientas-y-hardware
frase: Las librerías y el hardware con los que se construyen y entrenan los modelos en la práctica.
desc: El ecosistema práctico del ML: frameworks como scikit-learn para modelos clásicos y PyTorch, Keras o JAX para deep learning; plataformas como Hugging Face para compartir modelos y datos; herramientas para registrar experimentos; y hardware acelerado (GPU y TPU), que hace viable entrenar redes grandes.
ej: Un proyecto típico usa pandas y scikit-learn para la línea base, PyTorch en GPU para la red final y una herramienta de seguimiento para comparar las ejecuciones.
rel: frameworks-de-ml, gpu-y-tpu, hugging-face, seguimiento-de-experimentos
en: tools & hardware

## gpu-y-tpu
frase: Procesadores para cálculo masivamente paralelo que hacen viable entrenar redes grandes.
desc: Entrenar redes neuronales consiste sobre todo en multiplicar matrices, una operación muy paralelizable. Las GPU, con miles de núcleos, la aceleran decenas de veces respecto a una CPU. Las TPU de Google son chips diseñados específicamente para estas operaciones. La memoria del acelerador suele ser el límite práctico del tamaño del modelo y del lote.
cuando: Para entrenar o servir redes neuronales medianas o grandes; los modelos clásicos sobre datos tabulares rara vez lo necesitan.
ej: Ajustar por completo un modelo de lenguaje de 7.000 millones de parámetros exige varias GPU con decenas de GB de memoria; con QLoRA cabe en una sola.
py: torch.cuda.is_available ; jax.devices
rel: frameworks-de-ml, epoca-y-tamano-de-lote, qlora, despliegue-e-inferencia
en: GPU & TPU

## frameworks-de-ml
frase: Las librerías con las que se construyen los modelos: scikit-learn, Keras, TensorFlow, PyTorch y JAX.
desc: scikit-learn ofrece modelos clásicos con una interfaz común (fit, predict) y es el estándar para datos tabulares. PyTorch domina la investigación y buena parte de la industria en deep learning. Keras ofrece una interfaz de alto nivel que funciona sobre TensorFlow, PyTorch o JAX. JAX combina una sintaxis similar a NumPy con diferenciación automática y compilación para acelerar el cálculo.
ej: Una línea base con regresión logística en scikit-learn y, después, una red en PyTorch o Keras para comparar.
rel: herramientas-y-hardware, diferenciacion-automatica, api-secuencial, hugging-face
en: ML frameworks

## hugging-face
frase: Plataforma y librerías para descargar, usar y compartir modelos y datasets preentrenados.
desc: El Hub de Hugging Face aloja cientos de miles de modelos y datasets públicos. Sus librerías permiten cargarlos y usarlos en pocas líneas: transformers (modelos de texto, visión y audio), datasets, diffusers (modelos de difusión), peft (ajuste eficiente) y tokenizers, entre otras. Se ha convertido en el punto de partida habitual para trabajar con modelos preentrenados.
cuando: Para partir de un modelo preentrenado en lugar de entrenar desde cero, o para compartir modelos y datos.
ej: Clasificar el sentimiento de reseñas de clientes con un modelo preentrenado del Hub en tres líneas de código.
py: transformers.pipeline ; huggingface_hub.hf_hub_download ; datasets.load_dataset
rel: transfer-learning, modelos-fundacionales, llms-locales-vs-api, peft
en: Hugging Face (Hub & libraries)

## seguimiento-de-experimentos
frase: Registrar parámetros, métricas y artefactos de cada entrenamiento para compararlos y reproducirlos.
desc: Cada ejecución guarda su configuración, sus curvas de entrenamiento, sus métricas finales y el modelo resultante. Así se pueden comparar decenas de pruebas, saber qué cambio produjo una mejora y reproducir cualquier resultado. TensorBoard, MLflow y Weights & Biases son las herramientas más extendidas.
cuando: En cuanto se prueban varias configuraciones o modelos, y siempre que se trabaja en equipo.
ej: Comparar en un panel las curvas de pérdida de 20 entrenamientos con distintos learning rates para elegir el mejor.
py: mlflow.start_run ; wandb.init ; keras.callbacks.TensorBoard
rel: mlops, curvas-de-aprendizaje, reproducibilidad-y-semillas, callbacks
en: experiment tracking
