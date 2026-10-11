# Fichas: Series temporales

## componentes-de-una-serie
frase: Descomponer una serie temporal en tendencia, estacionalidad y resto.
desc: Una serie suele combinar una tendencia (su evolución a largo plazo), una o varias estacionalidades (patrones que se repiten con un periodo fijo: diario, semanal, anual) y un residuo irregular. Separarlas, de forma aditiva o multiplicativa, ayuda a entender la serie, a eliminar efectos conocidos antes de modelar y a detectar anomalías en el residuo. STL es el método robusto más usado.
cuando: Al empezar a analizar cualquier serie, y para desestacionalizar antes de modelar o de comparar periodos.
ej: Las ventas de una heladería muestran una tendencia creciente, un pico cada verano y un patrón semanal con máximos en el fin de semana.
hp: period: longitud del ciclo estacional (7 en datos diarios con patrón semanal, 12 en datos mensuales con patrón anual) | seasonal: suavizado del componente estacional en STL
py: statsmodels.tsa.seasonal.STL ; statsmodels.tsa.seasonal.seasonal_decompose
rel: estacionariedad, autocorrelacion, deteccion-por-reconstruccion-o-prediccion, eda
en: time series components (trend, seasonality)

## estacionariedad
frase: Una serie es estacionaria si sus propiedades estadísticas (media, varianza, autocorrelación) no cambian con el tiempo.
desc: Muchos modelos clásicos, como ARIMA, suponen estacionariedad, porque permite aprender del pasado relaciones que siguen valiendo en el futuro. Las series con tendencia o estacionalidad no lo son, y se transforman diferenciando (restando el valor anterior), eliminando la estacionalidad o tomando logaritmos. Los tests de Dickey-Fuller aumentado y KPSS, con hipótesis nulas opuestas, ayudan a comprobarlo.
cuando: Antes de aplicar modelos clásicos de series, tests de causalidad de Granger o correlaciones entre series.
ej: El precio de una vivienda no es estacionario, pero su variación mensual en porcentaje lo es aproximadamente.
py: statsmodels.tsa.stattools.adfuller ; statsmodels.tsa.stattools.kpss
rel: componentes-de-una-serie, autocorrelacion, correlacion-espuria, causalidad-de-granger
en: stationarity

## autocorrelacion
frase: Cuánto se parece una serie a sí misma desplazada en el tiempo: la memoria de la serie.
desc: La función de autocorrelación (ACF) mide la correlación entre la serie y sus propios valores retrasados k pasos; la autocorrelación parcial (PACF) mide la parte de esa relación que no explican los retardos intermedios. Sus gráficos revelan estacionalidades (picos en el periodo), persistencia y qué retardos conviene usar en un modelo autorregresivo. En los residuos de un buen modelo no debería quedar autocorrelación.
cuando: Para identificar estacionalidades, elegir los retardos de un modelo o comprobar que sus residuos son ruido.
ej: El consumo eléctrico horario tiene picos de autocorrelación en el retardo 24 (la misma hora del día anterior) y en el 168 (la misma hora de la semana anterior).
hp: nlags / lags: número de retardos que se calculan o se representan
py: statsmodels.tsa.stattools.acf ; statsmodels.graphics.tsaplots.plot_acf ; statsmodels.graphics.tsaplots.plot_pacf
rel: estacionariedad, prediccion-autorregresiva, componentes-de-una-serie, correlacion-de-pearson
en: autocorrelation (ACF, PACF)

## enventanado
frase: Convertir una serie en ejemplos de entrenamiento: cada ventana de valores pasados predice el siguiente.
desc: Los modelos de ML esperan pares de entrada y salida. El enventanado recorre la serie con una ventana deslizante: los últimos n valores son la entrada y el valor siguiente, o los h siguientes, la salida. La longitud de la ventana define cuánta historia ve el modelo, y el paso entre ventanas, cuántos ejemplos se generan. Las ventanas deben crearse después de dividir en entrenamiento y test, para no mezclar periodos.
cuando: Siempre que se use un modelo de ML genérico, como árboles o redes, para predecir series.
ej: Con 30 días de historia por ventana, una serie de 1.000 días de ventas genera unos 970 ejemplos para entrenar un modelo que predice el día siguiente.
hp: sequence_length: longitud de cada ventana | sequence_stride: desplazamiento entre ventanas consecutivas
py: keras.utils.timeseries_dataset_from_array ; numpy.lib.stride_tricks.sliding_window_view
rel: prediccion-multipaso, validacion-en-series-temporales, redes-neuronales-para-series, feature-engineering
en: windowing (sliding windows)

## prediccion-multipaso
frase: Predecir no solo el siguiente valor, sino varios pasos del futuro.
desc: Hay tres estrategias principales. La recursiva usa un modelo de un paso y realimenta sus propias predicciones: es sencilla, pero acumula errores. La directa entrena un modelo distinto para cada horizonte. La multisalida predice todo el horizonte de una vez con un único modelo, como una red con h salidas o un modelo de secuencia a secuencia. Cuanto más lejano es el horizonte, mayor es la incertidumbre.
cuando: Cuando hay que planificar con antelación: la demanda de la próxima semana, la producción de los próximos días o la capacidad del mes.
ej: Un operador eléctrico predice el consumo de cada una de las próximas 48 horas para programar sus centrales.
py: sklearn.multioutput.MultiOutputRegressor ; sklearn.multioutput.RegressorChain
rel: prediccion-autorregresiva, enventanado, topologias-de-secuencia, validacion-en-series-temporales
en: one-step & multi-step forecasting

## prediccion-autorregresiva
frase: Predecir cada valor de la serie como una combinación de sus valores anteriores.
desc: Un modelo AR(p) predice el valor actual como una combinación lineal de los p anteriores más un ruido. ARIMA añade diferenciación para la tendencia (I) y medias móviles de los errores pasados (MA), y SARIMA, términos estacionales. Son interpretables, funcionan bien con pocas series y poco ruido, y son la línea base clásica. La idea autorregresiva reaparece en las redes recurrentes y en los modelos de lenguaje.
cuando: Como línea base para series individuales con pocos datos o cuando se necesita un modelo interpretable con intervalos de predicción.
ej: Predecir el número mensual de pasajeros de un aeropuerto con un SARIMA que capta la tendencia y el pico de cada verano.
hp: lags: retardos que incluye el modelo AR | order: órdenes (p, d, q) de ARIMA
py: statsmodels.tsa.ar_model.AutoReg ; statsmodels.tsa.arima.model.ARIMA
rel: autocorrelacion, modelos-autorregresivos, estacionariedad, linea-base
en: autoregressive forecasting (AR, ARIMA)

## redes-neuronales-para-series
frase: RNN, convoluciones 1D y transformers aplicados a predecir o clasificar series temporales.
desc: Las redes aprenden patrones no lineales y combinan muchas series y variables a la vez. Las LSTM y las GRU procesan la secuencia paso a paso; las convoluciones 1D y las redes convolucionales temporales captan patrones locales en paralelo; los transformers, y variantes especializadas como el Temporal Fusion Transformer, relacionan horizontes largos. Brillan con muchas series relacionadas o variables externas; con una sola serie corta, los modelos clásicos suelen ganar.
cuando: Con muchas series relacionadas, abundantes variables externas o patrones complejos y no lineales.
ej: Predecir la demanda diaria de 10.000 productos de una cadena de supermercados con un único modelo que aprende de todos a la vez.
py: keras.layers.LSTM ; keras.layers.Conv1D
rel: lstm, convolucion-1d, enventanado, modelos-fundacionales-de-series
en: neural networks for time series

## modelos-fundacionales-de-series
frase: Modelos preentrenados con enormes cantidades de series que predicen series nuevas sin entrenar (zero-shot).
desc: Siguiendo la receta de los LLM, se preentrenan transformers con miles de millones de puntos de series muy diversas (energía, tráfico, ventas, clima). Chronos convierte los valores en tokens y los predice como un modelo de lenguaje; TimesFM y Moirai trabajan con parches de la serie. Dan predicciones probabilísticas razonables sin entrenamiento y se pueden ajustar con los datos propios.
cuando: Para obtener predicciones rápidas sin entrenar, con series cortas o como línea base fuerte frente a modelos propios.
ej: Predecir las visitas diarias de una web nueva, con solo dos meses de historia, usando Chronos sin ajustar nada.
py: chronos.BaseChronosPipeline ; chronos.Chronos2Pipeline
rel: modelos-fundacionales, redes-neuronales-para-series, prediccion-multipaso, incertidumbre
en: time series foundation models (TimesFM, Chronos)

## dtw
frase: Comparar dos series permitiendo estirar y encoger el tiempo, para alinear patrones desfasados.
desc: La distancia euclídea compara punto a punto y penaliza cualquier desfase. Dynamic Time Warping busca, mediante programación dinámica, el alineamiento entre los puntos de las dos series que minimiza la distancia total, de modo que dos patrones iguales que transcurren a distinta velocidad resultan cercanos. Una banda de Sakoe-Chiba limita cuánto se puede deformar el tiempo y acelera el cálculo.
cuando: Para comparar, agrupar o clasificar series cuyos patrones aparecen desplazados o a distinta velocidad.
ej: Reconocer la misma palabra pronunciada por dos personas, una más deprisa que la otra, comparando sus señales de audio.
hp: global_constraint: restricción del alineamiento (banda de Sakoe-Chiba o paralelogramo de Itakura) | sakoe_chiba_radius: anchura máxima de la deformación temporal
py: tslearn.metrics.dtw ; tslearn.clustering.TimeSeriesKMeans
rel: distancia-euclidea, k-means, distancia-y-similitud, series-como-imagenes
en: Dynamic Time Warping (DTW)

## series-como-imagenes
frase: Transformar una serie en una imagen para aprovechar las CNN y los modelos de visión.
desc: Varias transformaciones convierten una serie 1D en una imagen 2D que conserva su información temporal: los campos angulares de Gram (GAF) codifican la relación entre cada par de instantes; los gráficos de recurrencia marcan cuándo la serie vuelve a estados parecidos; los escalogramas wavelet y los espectrogramas muestran la energía por frecuencia a lo largo del tiempo. Sobre esas imágenes se aplican CNN, a menudo preentrenadas.
cuando: Para clasificar series o detectar patrones con CNN, sobre todo si se quiere aprovechar el transfer learning de visión.
ej: Clasificar el estado de un rodamiento a partir de los espectrogramas de su vibración con una CNN preentrenada con imágenes.
hp: image_size: tamaño de la imagen resultante | method: en GAF, suma o diferencia de ángulos
py: pyts.image.GramianAngularField ; pyts.image.RecurrencePlot
rel: espectrograma-de-mel, wavelets, cnn, transfer-learning
en: time series as images (GAF, wavelets, spectrograms)
