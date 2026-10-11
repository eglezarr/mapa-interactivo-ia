# Fichas: Redes neuronales: cómo aprenden

## neurona-artificial
frase: Unidad que suma sus entradas ponderadas y aplica una activación; inspirada, de lejos, en la neurona biológica.
desc: Una neurona biológica recibe señales por las dendritas y, si su suma supera un umbral, dispara un impulso por el axón. La neurona artificial imita esa idea de forma muy simplificada: calcula una suma ponderada de sus entradas más un sesgo, z = w·x + b, y le aplica una función de activación no lineal. Los pesos, que se aprenden, determinan cuánto influye cada entrada.
ej: Una neurona que decide si un correo es spam da un peso positivo a la palabra "gratis" y uno negativo a que el remitente sea un contacto conocido.
py: keras.layers.Dense ; torch.nn.Linear
rel: perceptron, funciones-de-activacion, mlp, inspiracion-biologica-de-las-cnn
en: artificial neuron

## perceptron
frase: La primera neurona artificial entrenable: un clasificador lineal que corrige sus pesos con cada error.
desc: Propuesto por Rosenblatt en 1958, aplica un escalón a una suma ponderada para decidir entre dos clases. Su regla de aprendizaje ajusta los pesos cada vez que se equivoca y converge si las clases son linealmente separables. No puede resolver problemas como el XOR, una limitación que frenó el campo hasta la llegada de las redes multicapa.
ej: Un perceptrón aprende la función lógica AND, pero ninguna recta separa los casos del XOR, así que esa no la aprende nunca.
py: sklearn.linear_model.Perceptron
rel: neurona-artificial, mlp, breve-historia-de-la-ia, svm
en: perceptron

## mlp
frase: Capas de neuronas conectadas en cadena: con capas ocultas y no linealidades, aproxima casi cualquier función.
desc: El perceptrón multicapa encadena una capa de entrada, una o más capas ocultas y una de salida, con todas las neuronas de una capa conectadas a las de la siguiente (capas densas). Las activaciones no lineales le permiten aprender relaciones no lineales, y el teorema de aproximación universal garantiza que una capa oculta suficientemente ancha puede aproximar cualquier función continua. Se entrena con retropropagación.
cuando: Con datos tabulares o vectores de características cuando la relación es no lineal, y como cabeza de clasificación de otras redes.
ej: Una red con dos capas ocultas de 64 neuronas que predice la probabilidad de impago a partir de 30 variables de cada cliente.
hp: hidden_layer_sizes: número y tamaño de las capas ocultas | alpha: fuerza de la regularización L2 | learning_rate_init: tamaño de paso inicial
py: keras.layers.Dense ; torch.nn.Linear ; sklearn.neural_network.MLPClassifier
rel: neurona-artificial, backpropagation, funciones-de-activacion, api-secuencial
en: multilayer perceptron (MLP)

## funciones-de-activacion
frase: La no linealidad que se aplica a la salida de cada neurona y da a la red su capacidad expresiva.
desc: Sin ellas, cualquier número de capas equivaldría a una sola transformación lineal. Las activaciones ocultas más usadas hoy son ReLU y sus variantes, y GELU en los transformers. En la salida, la activación depende del problema: lineal para regresión, sigmoide para clasificación binaria o multietiqueta y softmax para multiclase. Su forma influye en cómo fluye el gradiente durante el entrenamiento.
ej: Una red que predice precios termina en una neurona lineal; una que clasifica imágenes en diez clases, en una softmax.
py: keras.activations.relu ; torch.nn.functional.relu
rel: sigmoide-y-tanh, relu-y-variantes, gelu, softmax, desvanecimiento-y-explosion-del-gradiente
en: activation functions

## sigmoide-y-tanh
frase: Funciones en forma de S que comprimen la entrada al intervalo (0, 1) o al (−1, 1).
desc: La sigmoide, σ(z) = 1/(1 + e⁻ᶻ), da valores entre 0 y 1 y se usa en la salida para probabilidades binarias o multietiqueta. La tangente hiperbólica da valores entre −1 y 1 y está centrada en cero. Ambas se saturan con entradas grandes, donde su derivada casi se anula, lo que provoca el desvanecimiento del gradiente en redes profundas; por eso en las capas ocultas se prefirió ReLU. Siguen siendo clave en las puertas de LSTM y GRU.
ej: La salida de un detector de spam pasa por una sigmoide: un 0,93 se interpreta como un 93 % de probabilidad de spam.
py: keras.activations.sigmoid ; keras.activations.tanh ; torch.nn.Sigmoid ; torch.nn.Tanh
rel: funciones-de-activacion, regresion-logistica, desvanecimiento-y-explosion-del-gradiente, lstm
en: sigmoid & tanh

## relu-y-variantes
frase: max(0, z): la activación por defecto en capas ocultas, simple y con un gradiente que no se satura.
desc: ReLU deja pasar los valores positivos y anula los negativos. Es barata de calcular y su gradiente vale 1 en la parte positiva, lo que facilitó entrenar redes profundas. Su problema es que una neurona puede quedar "muerta", siempre en la zona negativa y sin gradiente. Leaky ReLU usa una pequeña pendiente para los negativos y ELU una curva exponencial suave, lo que mantiene vivo el gradiente.
cuando: ReLU, como activación por defecto en capas ocultas de redes densas y convolucionales; sus variantes, si muchas neuronas se quedan muertas.
ej: En una CNN que clasifica radiografías, cada convolución va seguida de una ReLU.
hp: negative_slope: en Leaky ReLU, pendiente para las entradas negativas
py: keras.layers.ReLU ; keras.layers.LeakyReLU ; torch.nn.ReLU ; torch.nn.LeakyReLU ; torch.nn.ELU
rel: funciones-de-activacion, gelu, desvanecimiento-y-explosion-del-gradiente, inicializacion-de-pesos
en: ReLU & variants (Leaky ReLU, ELU)

## gelu
frase: Variante suave de ReLU que pondera cada entrada por la probabilidad de que sea positiva; estándar en transformers.
desc: GELU multiplica la entrada por la función de distribución de una normal estándar: z·Φ(z). Se comporta como ReLU para valores grandes, pero es suave y deja pasar algo de señal en los negativos cercanos a cero. Es la activación habitual en BERT, GPT y los vision transformers. Variantes como SwiGLU, con una puerta multiplicativa, dominan en los LLM recientes.
cuando: En las capas ocultas de transformers y de modelos modernos de lenguaje y visión.
ej: Cada bloque feed-forward de BERT expande la representación de 768 a 3.072 dimensiones, aplica GELU y la vuelve a proyectar a 768.
py: keras.activations.gelu ; torch.nn.GELU
rel: relu-y-variantes, bloque-transformer, funciones-de-activacion, bert
en: GELU

## softmax
frase: Convierte un vector de puntuaciones en probabilidades positivas que suman 1.
desc: Aplica la exponencial a cada puntuación (logit) y divide por la suma de todas, de modo que la mayor recibe la mayor probabilidad. Es la salida estándar de la clasificación multiclase y el núcleo del mecanismo de atención. Dividir los logits por una temperatura antes de aplicarla hace la distribución más plana (temperatura alta) o más picuda (temperatura baja).
ej: Los logits 2,0, 1,0 y 0,1 para gato, perro y pájaro se convierten en probabilidades de 0,66, 0,24 y 0,10.
py: keras.activations.softmax ; torch.nn.Softmax
rel: entropia-cruzada, clasificacion-multiclase-y-multietiqueta, mecanismo-de-atencion, decodificacion-y-temperatura
en: softmax

## funciones-de-perdida
frase: La medida del error que la red minimiza durante el entrenamiento.
desc: La función de pérdida compara la predicción con el valor real y devuelve un número que el optimizador intenta reducir. Su elección depende del tipo de salida: errores cuadráticos o absolutos en regresión, entropía cruzada en clasificación, pérdidas perceptuales o adversarias en imágenes generadas. Muchas se derivan del principio de máxima verosimilitud bajo distintos supuestos sobre el ruido.
ej: Un modelo que predice la demanda horaria minimiza el MSE; uno que clasifica tumores, la entropía cruzada.
py: keras.losses ; torch.nn.functional
rel: perdidas-de-regresion, entropia-cruzada, funcion-de-perdida-vs-metrica, maxima-verosimilitud
en: loss functions

## perdidas-de-regresion
frase: MSE, MAE y Huber: tres formas de penalizar el error numérico, con distinta sensibilidad a los atípicos.
desc: El MSE eleva los errores al cuadrado: penaliza mucho los grandes y lleva a predecir la media. El MAE usa el valor absoluto: es robusto a los atípicos y lleva a predecir la mediana, pero su gradiente no disminuye cerca del óptimo. Huber combina ambas: se comporta como el MSE con errores pequeños y como el MAE con los grandes, a partir de un umbral δ.
cuando: MSE, por defecto; MAE o Huber, cuando hay atípicos que no deben dominar el entrenamiento.
ej: Al predecir tiempos de entrega con algunos retrasos extremos por incidencias, Huber evita que esos casos arrastren todo el modelo.
hp: delta: en Huber, error a partir del cual la penalización pasa de cuadrática a lineal
py: keras.losses.MeanSquaredError ; keras.losses.MeanAbsoluteError ; keras.losses.Huber ; torch.nn.HuberLoss
rel: mse-rmse-y-mae, funciones-de-perdida, outliers, maxima-verosimilitud
en: regression losses (MSE, MAE, Huber)

## entropia-cruzada
frase: La pérdida estándar de clasificación: penaliza asignar poca probabilidad a la clase correcta.
desc: Mide −log de la probabilidad que el modelo asigna a la clase verdadera, promediada sobre los ejemplos. La versión binaria acompaña a una salida sigmoide y la categórica, a una softmax; la variante "sparse" recibe la clase como un entero en lugar de en one-hot. Es la NLL de una distribución categórica, y por estabilidad numérica conviene pasarle los logits en lugar de las probabilidades.
ej: Si el modelo asigna un 0,9 a la clase correcta, la pérdida de ese ejemplo es 0,105; si le asigna un 0,1, es 2,3.
hp: label_smoothing: reparte una pequeña parte de la probabilidad entre las clases incorrectas para evitar el exceso de confianza
py: keras.losses.BinaryCrossentropy ; keras.losses.SparseCategoricalCrossentropy ; torch.nn.CrossEntropyLoss ; torch.nn.BCEWithLogitsLoss
rel: nll, softmax, divergencia-de-kullback-leibler, funciones-de-perdida
en: cross-entropy (binary & categorical)

## perdidas-perceptuales
frase: Comparan imágenes por sus características visuales en una red preentrenada, no píxel a píxel.
desc: Dos imágenes desplazadas un píxel tienen un error cuadrático grande aunque parezcan idénticas. Las pérdidas perceptuales comparan las activaciones que ambas imágenes producen en una red preentrenada, como VGG, que capturan bordes, texturas y formas. Se usan en superresolución, transferencia de estilo y modelos generativos para obtener resultados más nítidos y naturales. LPIPS es una medida de este tipo calibrada con juicios humanos.
cuando: Para entrenar o evaluar modelos que generan imágenes cuando importa la calidad visual percibida más que el error por píxel.
ej: Un modelo de superresolución entrenado con MSE produce caras borrosas; con una pérdida perceptual recupera la textura de la piel y del pelo.
py: lpips.LPIPS ; keras.applications.VGG19
rel: transferencia-de-estilo-neuronal, srgan, metricas-perceptuales, superresolucion-inpainting-y-colorizacion
en: perceptual losses

## optimizacion
frase: Cómo se ajustan los pesos de la red, paso a paso, para reducir la pérdida.
desc: Entrenar una red es resolver un problema de optimización: encontrar los pesos que minimizan la pérdida. La retropropagación calcula el gradiente, que indica cómo cambia la pérdida al mover cada peso, y un optimizador da pasos en la dirección que la reduce. Detalles como el learning rate, el tamaño de lote, el momentum o la inicialización deciden si el entrenamiento converge, a qué velocidad y a qué solución.
ej: Entrenar una CNN con Adam y un learning rate de 0,001 que se reduce cuando la pérdida de validación se estanca.
rel: descenso-por-gradiente, backpropagation, adam-y-adamw, learning-rate
en: optimization

## descenso-por-gradiente
frase: Bajar por la superficie de la pérdida dando pasos en la dirección de máxima pendiente descendente.
desc: En cada paso actualiza los pesos con w ← w − η·∇L(w), donde ∇L es el gradiente de la pérdida y η, el learning rate. Si el paso es pequeño, avanza despacio; si es grande, oscila o diverge. En las redes neuronales la superficie de la pérdida no es convexa, pero en la práctica el descenso por gradiente encuentra soluciones que generalizan bien.
ej: Es como bajar una montaña con niebla: en cada paso miras la pendiente bajo tus pies y avanzas cuesta abajo.
py: keras.optimizers.SGD ; torch.optim.SGD
rel: learning-rate, batch-mini-batch-y-sgd, backpropagation, momentum
en: gradient descent

## backpropagation
frase: Calcula de forma eficiente cuánto contribuye cada peso al error, propagándolo de la salida hacia atrás.
desc: Aplica la regla de la cadena capa a capa, desde la pérdida hasta la entrada, reutilizando los cálculos intermedios. Con un pase hacia delante y otro hacia atrás obtiene el gradiente de la pérdida respecto a todos los pesos, con un coste similar al de hacer la propia predicción. Es el algoritmo que hizo práctico entrenar redes con muchas capas.
ej: Si una red clasifica mal la foto de un gato, la retropropagación indica a cada uno de sus millones de pesos cuánto y en qué sentido debe cambiar.
py: torch.Tensor.backward
rel: diferenciacion-automatica, descenso-por-gradiente, desvanecimiento-y-explosion-del-gradiente, bptt
en: backpropagation

## diferenciacion-automatica
frase: El framework calcula las derivadas exactas de cualquier cálculo que escribas, sin derivar a mano.
desc: Los frameworks registran las operaciones que se aplican a los tensores y, recorriendo ese grafo de cálculo con la regla de la cadena, obtienen derivadas exactas, no aproximaciones numéricas. El modo inverso, que es la retropropagación, es eficiente cuando hay muchos parámetros y una sola salida, como la pérdida. Permite entrenar cualquier arquitectura diferenciable sin escribir los gradientes.
cuando: Siempre que se entrena un modelo diferenciable o se necesita el gradiente de una función compleja.
ej: Definir una pérdida nueva que combina tres términos y obtener sus gradientes automáticamente en una línea.
py: torch.autograd.grad ; jax.grad
rel: backpropagation, frameworks-de-ml, capas-y-perdidas-personalizadas, descenso-por-gradiente
en: automatic differentiation

## batch-mini-batch-y-sgd
frase: Calcular el gradiente con todos los datos, con un subconjunto o con un solo ejemplo en cada paso.
desc: El descenso por lotes (batch) usa todo el conjunto en cada paso: es preciso pero lento. El estocástico (SGD) usa un solo ejemplo: es rápido pero muy ruidoso. El mini-batch, con lotes de decenas o cientos de ejemplos, es el estándar: aprovecha la GPU y su ruido moderado ayuda a escapar de malos mínimos. En la práctica, "SGD" suele referirse a esta última variante.
cuando: Mini-batch, siempre para redes neuronales; el lote completo, solo con datasets pequeños y métodos que lo necesitan, como L-BFGS.
ej: Con un millón de imágenes y lotes de 256, cada época tiene unos 3.900 pasos de actualización.
hp: batch_size: número de ejemplos por lote | shuffle: barajar los datos en cada época
py: torch.utils.data.DataLoader ; keras.Model.fit
rel: epoca-y-tamano-de-lote, descenso-por-gradiente, learning-rate, gpu-y-tpu
en: batch, mini-batch & SGD

## epoca-y-tamano-de-lote
frase: Época: una pasada completa por los datos. Lote: cuántos ejemplos se usan en cada actualización.
desc: El entrenamiento recorre los datos varias veces (épocas), divididos en lotes, y cada lote produce una actualización de los pesos. Los lotes grandes aprovechan mejor el hardware y dan gradientes más estables, pero necesitan más memoria y a veces generalizan algo peor; si se aumenta el lote, suele convenir subir también el learning rate. El número de épocas se controla con early stopping.
ej: Entrenar 30 épocas con lotes de 64 imágenes; la pérdida de validación deja de mejorar en la época 18.
hp: epochs: número de pasadas completas por los datos | batch_size: ejemplos por actualización
py: keras.Model.fit
rel: batch-mini-batch-y-sgd, early-stopping, learning-rate, curvas-de-aprendizaje
en: epoch & batch size

## learning-rate
frase: El tamaño de cada paso del optimizador: el hiperparámetro más importante del entrenamiento.
desc: Multiplica el gradiente en cada actualización. Si es demasiado alto, la pérdida oscila o se dispara; si es demasiado bajo, el entrenamiento avanza muy despacio o se queda atascado. Su valor óptimo depende del optimizador, del tamaño de lote y de la arquitectura, y suele buscarse en escala logarítmica, por ejemplo entre 10⁻⁵ y 10⁻¹. Casi siempre conviene variarlo durante el entrenamiento.
cuando: Es lo primero que hay que ajustar al entrenar una red y lo primero que hay que revisar si la pérdida no baja o se dispara.
ej: Con Adam, 0,001 suele funcionar como punto de partida; para ajustar un modelo preentrenado se usan valores entre 10 y 100 veces menores.
hp: learning_rate / lr: tamaño del paso (Keras / PyTorch)
py: keras.optimizers.Adam ; torch.optim.Adam
rel: lr-schedulers, adam-y-adamw, descenso-por-gradiente, learning-rates-discriminativos
en: learning rate

## lr-schedulers
frase: Variar el learning rate durante el entrenamiento: subirlo con cuidado al principio y bajarlo para afinar.
desc: Un programador ajusta el learning rate según avanzan los pasos o las épocas. Los más usados son el decaimiento por escalones o exponencial; el coseno, que lo baja suavemente hasta casi cero; el calentamiento (warm-up), que lo sube poco a poco al principio para estabilizar redes grandes; y la reducción cuando la pérdida de validación se estanca. Suelen mejorar tanto la velocidad como el resultado final.
cuando: Casi siempre en redes profundas; el calentamiento, en transformers y en entrenamientos con lotes grandes.
ej: Calentar durante las primeras 1.000 iteraciones hasta 3·10⁻⁴ y después bajar con un coseno hasta cero al final del entrenamiento.
hp: patience: en la reducción por meseta, épocas sin mejora antes de bajar el learning rate | factor: por cuánto se multiplica el learning rate al reducirlo | T_max: en el coseno de PyTorch, pasos hasta llegar al mínimo
py: keras.optimizers.schedules.CosineDecay ; keras.callbacks.ReduceLROnPlateau ; torch.optim.lr_scheduler.CosineAnnealingLR ; torch.optim.lr_scheduler.OneCycleLR
rel: learning-rate, adam-y-adamw, callbacks, receta-de-entrenamiento
en: learning rate schedules

## momentum
frase: Acumula la dirección de los pasos anteriores para avanzar más rápido y oscilar menos.
desc: En lugar de seguir solo el gradiente actual, mantiene una media móvil de los gradientes pasados, como una bola que coge inercia al rodar cuesta abajo. Acelera el avance en las direcciones consistentes y amortigua las oscilaciones en las que cambian de signo. El momentum de Nesterov evalúa el gradiente en la posición a la que apunta la inercia, lo que lo hace algo más preciso.
cuando: Al usar SGD, prácticamente siempre; Adam ya incorpora una forma de momentum.
ej: En un valle estrecho y alargado, SGD zigzaguea de pared a pared; con un momentum de 0,9 avanza casi en línea recta hacia el fondo.
hp: momentum: fracción de la velocidad anterior que se conserva, típicamente 0,9 | nesterov: activa el momentum de Nesterov
py: keras.optimizers.SGD ; torch.optim.SGD
rel: descenso-por-gradiente, adam-y-adamw, rmsprop-y-adagrad, learning-rate
en: momentum

## rmsprop-y-adagrad
frase: Optimizadores que adaptan el learning rate de cada peso según el tamaño de sus gradientes pasados.
desc: Adagrad divide el paso de cada parámetro por la raíz de la suma de sus gradientes al cuadrado: los parámetros con gradientes grandes o frecuentes avanzan menos, y los raros, más, lo que funciona bien con datos dispersos. Su problema es que el paso decrece sin parar. RMSprop lo corrige usando una media móvil de los gradientes al cuadrado en lugar de la suma, y es la base de Adam.
cuando: Adagrad, con variables dispersas como texto o recomendación; RMSprop, en redes recurrentes o como alternativa a Adam.
ej: En un modelo con embeddings de palabras, Adagrad da pasos grandes a las palabras raras, que reciben pocas actualizaciones.
hp: learning_rate / lr: tamaño de paso base (Keras / PyTorch) | rho / alpha: en RMSprop, factor de olvido de la media móvil (Keras / PyTorch)
py: keras.optimizers.RMSprop ; keras.optimizers.Adagrad ; torch.optim.RMSprop ; torch.optim.Adagrad
rel: adam-y-adamw, momentum, learning-rate, word-embeddings
en: RMSprop & Adagrad

## adam-y-adamw
frase: El optimizador por defecto: combina momentum con un learning rate adaptado a cada peso.
desc: Adam mantiene dos medias móviles por parámetro: la del gradiente (momentum) y la de su cuadrado (como RMSprop), con una corrección del sesgo inicial. Funciona bien con poco ajuste en casi cualquier arquitectura. AdamW separa el weight decay del gradiente y lo aplica directamente a los pesos, lo que regulariza mejor; es el estándar para entrenar transformers.
cuando: Como optimizador por defecto en deep learning; AdamW, cuando además se quiere regularizar con weight decay.
ej: Ajustar un modelo de lenguaje con AdamW, un learning rate de 2·10⁻⁵ y un weight decay de 0,01.
hp: learning_rate / lr: tamaño de paso (Keras / PyTorch) | beta_1 / betas: tasas de olvido de las medias móviles (Keras / PyTorch) | weight_decay: fuerza del decaimiento de pesos en AdamW
py: keras.optimizers.Adam ; keras.optimizers.AdamW ; torch.optim.Adam ; torch.optim.AdamW
rel: momentum, rmsprop-y-adagrad, penalizacion-l1-y-l2, learning-rate
en: Adam & AdamW

## metodos-de-segundo-orden
frase: Usan también la curvatura de la pérdida para dar pasos mejor dirigidos, a un coste mucho mayor.
desc: El método de Newton usa la matriz hessiana de segundas derivadas para saltar directamente al mínimo de una aproximación cuadrática. Converge en muy pocas iteraciones, pero con millones de parámetros calcular e invertir la hessiana es inviable. Los métodos cuasi-Newton, como L-BFGS, la aproximan con los gradientes recientes y se usan en modelos pequeños, en optimización clásica y en ajustes deterministas con el lote completo.
cuando: Con pocos parámetros (miles, no millones) y gradientes de lote completo, como en regresión logística o en el ajuste de modelos estadísticos.
ej: La regresión logística de scikit-learn usa por defecto L-BFGS y converge en unas decenas de iteraciones.
py: torch.optim.LBFGS ; scipy.optimize.minimize
rel: descenso-por-gradiente, optimizacion, regresion-logistica, adam-y-adamw
en: second-order methods (Newton, L-BFGS)

## inicializacion-de-pesos
frase: Elegir los valores iniciales de los pesos para que la señal no se apague ni explote al atravesar la red.
desc: Si los pesos empiezan demasiado pequeños, las activaciones y los gradientes se desvanecen capa a capa; si son demasiado grandes, explotan. Las inicializaciones de Glorot (Xavier) y de He fijan la varianza de los pesos según el número de entradas y salidas de cada capa para mantener estable la señal: Glorot, pensada para tanh y sigmoide; He, para ReLU. Inicializar todos los pesos igual impide que las neuronas aprendan cosas distintas.
ej: Una red profunda con ReLU e inicialización He empieza a aprender desde la primera época; con pesos aleatorios demasiado pequeños, apenas se mueve.
py: keras.initializers.HeNormal ; keras.initializers.GlorotUniform ; torch.nn.init.kaiming_normal_ ; torch.nn.init.xavier_uniform_
rel: desvanecimiento-y-explosion-del-gradiente, relu-y-variantes, capas-de-normalizacion, receta-de-entrenamiento
en: weight initialization (Glorot/Xavier, He)

## desvanecimiento-y-explosion-del-gradiente
frase: En redes profundas, el gradiente puede hacerse diminuto o enorme al propagarse hacia atrás por las capas.
desc: La retropropagación multiplica derivadas capa a capa. Si esos factores son menores que 1, el gradiente se desvanece y las primeras capas apenas aprenden; si son mayores, explota y el entrenamiento diverge. Es especialmente grave en redes muy profundas y en las recurrentes. Las soluciones son activaciones como ReLU, una buena inicialización, capas de normalización, conexiones residuales, LSTM o GRU en secuencias y el recorte del gradiente.
cuando: El recorte del gradiente, al entrenar redes recurrentes o transformers grandes, o cuando la pérdida se dispara de repente.
ej: Una RNN simple no aprende dependencias de más de unos diez pasos porque el gradiente se desvanece; una LSTM sí las aprende.
hp: max_norm: norma máxima del gradiente; si la supera, se reescala (en los optimizadores de Keras, clipnorm)
py: torch.nn.utils.clip_grad_norm_ ; `keras.optimizers.Adam(clipnorm=1.0)`
rel: backpropagation, inicializacion-de-pesos, conexion-residual, lstm, capas-de-normalizacion
en: vanishing & exploding gradients

## regularizacion-en-redes
frase: Técnicas para que una red con millones de parámetros no memorice los datos de entrenamiento.
desc: Las redes neuronales tienen capacidad de sobra para memorizar el conjunto de entrenamiento, así que necesitan regularización para generalizar. Las técnicas más usadas son penalizar los pesos grandes (L2 o weight decay), apagar neuronas al azar (dropout), parar el entrenamiento a tiempo (early stopping), añadir ruido y aumentar los datos. Se suelen combinar varias.
ej: Una red que llega al 100 % de acierto en entrenamiento y se queda en el 80 % en validación mejora con dropout y aumento de datos.
rel: penalizacion-l1-y-l2, dropout, early-stopping, data-augmentation, sobreajuste
en: regularization in neural networks

## penalizacion-l1-y-l2
frase: Añadir a la pérdida un castigo por los pesos grandes para obtener modelos más simples.
desc: La penalización L2 suma a la pérdida el cuadrado de los pesos multiplicado por un factor y los empuja hacia cero sin anularlos. La L1 suma su valor absoluto y tiende a dejar pesos exactamente en cero. En redes, la L2 se aplica casi siempre como weight decay, que reduce un poco los pesos en cada paso; con Adam conviene usar AdamW, que lo aplica de forma desacoplada.
cuando: Casi siempre, como regularización básica; L1, cuando además se busca que muchos pesos sean cero.
ej: Añadir un weight decay de 10⁻⁴ al entrenar una CNN con 5.000 imágenes reduce la diferencia entre entrenamiento y validación.
hp: l2: factor de la penalización L2 en Keras | weight_decay: factor del decaimiento de pesos en el optimizador
py: keras.regularizers.L2 ; keras.regularizers.L1L2 ; torch.optim.AdamW
rel: ridge-lasso-y-elastic-net, regularizacion, adam-y-adamw, regularizacion-en-redes
en: L1 & L2 regularization (weight decay)

## dropout
frase: Apagar al azar una fracción de neuronas en cada paso de entrenamiento para que no dependan unas de otras.
desc: En cada paso de entrenamiento, cada neurona se anula con una probabilidad p y las demás se reescalan. La red no puede apoyarse en neuronas concretas y aprende representaciones redundantes y robustas; equivale a entrenar a la vez muchas subredes que comparten pesos. En inferencia se desactiva; mantenerlo activo al predecir (MC dropout) permite estimar la incertidumbre.
cuando: En capas densas grandes y en transformers cuando hay sobreajuste; en las convolucionales se usa menos, o en su versión espacial.
ej: Un dropout de 0,5 tras cada capa densa de un clasificador de texto reduce la distancia entre el acierto de entrenamiento y el de validación.
hp: rate / p: probabilidad de apagar cada neurona (Keras / PyTorch)
py: keras.layers.Dropout ; torch.nn.Dropout
rel: regularizacion-en-redes, ensembles, estimacion-de-incertidumbre-en-redes, inyeccion-de-ruido
en: dropout

## early-stopping
frase: Parar el entrenamiento cuando el error de validación deja de mejorar.
desc: Mientras el error de entrenamiento sigue bajando, el de validación acaba subiendo cuando la red empieza a memorizar. Early stopping vigila la métrica de validación y detiene el entrenamiento tras varias épocas sin mejora, recuperando los pesos del mejor momento. Es una regularización barata que además ahorra cómputo.
cuando: Prácticamente siempre que se entrena una red durante varias épocas con un conjunto de validación.
ej: Con una paciencia de 5 épocas, el entrenamiento se detiene en la época 23 y se restauran los pesos de la 18, la mejor en validación.
hp: patience: épocas sin mejora antes de parar | monitor: métrica que se vigila | restore_best_weights: recupera los pesos de la mejor época
py: keras.callbacks.EarlyStopping
rel: curvas-de-aprendizaje, callbacks, sobreajuste, entrenamiento-validacion-y-test
en: early stopping

## inyeccion-de-ruido
frase: Añadir ruido aleatorio a las entradas, los pesos o las activaciones para que la red sea más robusta.
desc: Perturbar los datos durante el entrenamiento obliga a la red a ser insensible a pequeñas variaciones, lo que actúa como regularización: añadir ruido gaussiano a las entradas equivale, aproximadamente, a una penalización L2. El ruido también es la base de los autoencoders de eliminación de ruido, del suavizado de etiquetas y del entrenamiento robusto frente a ataques.
cuando: Con pocos datos o entradas ruidosas, o cuando se busca robustez frente a perturbaciones.
ej: Añadir ruido gaussiano a las lecturas de sensores durante el entrenamiento para que el modelo tolere la imprecisión de los sensores reales.
hp: stddev: desviación típica del ruido gaussiano
py: keras.layers.GaussianNoise ; keras.layers.GaussianDropout
rel: regularizacion-en-redes, autoencoder-denoising, dropout, data-augmentation
en: noise injection

## entrenamiento-en-la-practica
frase: Los hábitos que separan un entrenamiento que funciona de uno que falla en silencio.
desc: Entrenar redes tiene mucho de oficio: empezar simple, verificar cada pieza, vigilar las curvas de pérdida, guardar los mejores modelos y poder reproducir cada resultado. Las redes fallan en silencio: un error de preprocesado o en la pérdida no produce ninguna excepción, solo un modelo peor de lo que podría ser.
ej: Antes de entrenar con todo el dataset, comprobar que la red es capaz de memorizar un único lote de 32 ejemplos.
rel: receta-de-entrenamiento, curvas-de-aprendizaje, callbacks, reproducibilidad-y-semillas
en: training in practice

## receta-de-entrenamiento
frase: Un proceso paso a paso para entrenar redes: de lo simple a lo complejo, verificando cada paso.
desc: Una receta habitual es: inspeccionar a fondo los datos; montar el ciclo completo con un modelo sencillo y líneas base; comprobar que la red puede sobreajustar un lote pequeño; aumentar la capacidad hasta sobreajustar el conjunto de entrenamiento; regularizar para generalizar; y, solo entonces, ajustar los hiperparámetros. Cada paso aísla posibles errores antes de añadir complejidad.
cuando: Al empezar cualquier proyecto de deep learning y al depurar un entrenamiento que no funciona.
ej: Si la red no consigue memorizar 10 imágenes, el fallo está en el código o en los datos, no en que le falten capas.
rel: entrenamiento-en-la-practica, linea-base, curvas-de-aprendizaje, sobreajuste
en: neural network training recipe

## curvas-de-aprendizaje
frase: La evolución de la pérdida o la métrica durante el entrenamiento, en entrenamiento y en validación.
desc: Comparar la pérdida de entrenamiento con la de validación es la herramienta de diagnóstico más útil. Si ambas son altas, hay infraajuste; si la de entrenamiento baja y la de validación sube, sobreajuste; si oscilan mucho, el learning rate puede ser excesivo. Otra variante muestra el rendimiento frente al tamaño del conjunto de entrenamiento e indica si conseguir más datos ayudaría.
cuando: Siempre, durante y después de cada entrenamiento, para decidir el siguiente paso.
ej: La pérdida de validación toca fondo en la época 12 y después sube mientras la de entrenamiento sigue bajando: señal de sobreajuste.
py: sklearn.model_selection.learning_curve ; `history.history["val_loss"]`
rel: sobreajuste, infraajuste, early-stopping, seguimiento-de-experimentos
en: learning curves

## callbacks
frase: Funciones que se ejecutan en momentos concretos del entrenamiento para vigilar, guardar o ajustar el modelo.
desc: Un callback es código que el bucle de entrenamiento llama en momentos determinados, como al final de cada época o de cada lote. Se usan para parar a tiempo (early stopping), guardar el mejor modelo, ajustar el learning rate, registrar métricas o detener el entrenamiento si la pérdida da NaN. Permiten personalizar el entrenamiento sin reescribir el bucle.
cuando: En cualquier entrenamiento de más de unos minutos, para no perder el mejor modelo ni malgastar cómputo.
ej: Guardar automáticamente los pesos cada vez que mejora el acierto en validación y parar tras 10 épocas sin mejora.
hp: monitor: métrica que se vigila | save_best_only: guarda solo cuando la métrica mejora
py: keras.callbacks.ModelCheckpoint ; keras.callbacks.Callback
rel: early-stopping, lr-schedulers, seguimiento-de-experimentos, curvas-de-aprendizaje
en: callbacks

## reproducibilidad-y-semillas
frase: Fijar las fuentes de aleatoriedad para poder repetir un entrenamiento y obtener el mismo resultado.
desc: La inicialización de pesos, el barajado de datos, el dropout y el aumento de datos dependen de números aleatorios. Fijar las semillas de Python, NumPy y el framework permite repetir resultados, aunque en GPU algunas operaciones no son deterministas salvo que se fuerce. Como cada semilla da un resultado algo distinto, conviene comparar modelos con varias semillas y no con una sola ejecución.
cuando: Siempre que se comparen modelos, se depure un entrenamiento o se publiquen resultados.
ej: Un modelo parece un 1 % mejor que otro, pero con cinco semillas distintas la diferencia desaparece: era ruido.
py: keras.utils.set_random_seed ; torch.manual_seed ; torch.use_deterministic_algorithms
rel: seguimiento-de-experimentos, mlops, validacion-cruzada, entrenamiento-en-la-practica
en: reproducibility & random seeds

## construccion-de-modelos
frase: Las formas de definir la arquitectura de una red en código, de una pila simple a grafos complejos.
desc: Los frameworks ofrecen varios niveles. La API secuencial apila capas una tras otra. La funcional permite grafos con ramas, varias entradas y salidas y capas compartidas. La subclasificación, heredando de Layer, Model o nn.Module, da libertad total para lógica personalizada. Elegir el nivel adecuado hace el código más simple y más fácil de mantener.
ej: Un clasificador de imágenes simple cabe en una API secuencial; un modelo que combina una foto y su descripción necesita la funcional.
rel: api-secuencial, api-funcional, capas-y-perdidas-personalizadas, multi-entrada-salida
en: building models

## api-secuencial
frase: Definir una red como una pila de capas en la que cada una alimenta a la siguiente.
desc: Es la forma más sencilla de construir un modelo: una lista ordenada de capas en la que la salida de cada una es la entrada de la siguiente. Basta para los MLP y las CNN clásicas. No admite varias entradas o salidas, ramas ni capas compartidas; para eso existe la API funcional.
cuando: Para modelos que son una cadena de capas con una entrada y una salida.
ej: Un clasificador de dígitos: aplanar la imagen, una capa densa de 128 neuronas con ReLU y una salida softmax de 10 clases.
py: keras.Sequential ; torch.nn.Sequential
rel: api-funcional, mlp, construccion-de-modelos, frameworks-de-ml
en: Sequential API

## api-funcional
frase: Definir la red conectando capas como funciones, lo que permite ramas, atajos y varias entradas o salidas.
desc: Cada capa se aplica sobre el tensor de salida de otra, y el modelo se define indicando sus entradas y sus salidas. Así se construyen grafos arbitrarios sin ciclos: conexiones residuales, ramas paralelas que se concatenan, capas compartidas y modelos con varias entradas o salidas. La arquitectura sigue siendo inspeccionable y se puede dibujar.
cuando: Cuando el modelo no es una simple cadena de capas: atajos, ramas o varias entradas o salidas.
ej: Una red que recibe la imagen de un producto y su precio, procesa cada uno por una rama distinta y concatena ambas antes de clasificar.
py: keras.Model ; keras.Input
rel: api-secuencial, multi-entrada-salida, conexion-residual, capas-y-perdidas-personalizadas
en: Functional API

## capas-y-perdidas-personalizadas
frase: Programar tus propias capas, pérdidas o modelos cuando las piezas estándar no bastan.
desc: Heredando de las clases base del framework se define una capa con sus pesos y su cálculo hacia delante, o una pérdida que compara la predicción con el objetivo de forma específica. La diferenciación automática se encarga de los gradientes. Es la vía para implementar ideas nuevas o incorporar restricciones del dominio al entrenamiento.
cuando: Cuando necesitas una operación, una restricción o un objetivo que no existe en la librería.
ej: Una pérdida que penaliza el doble quedarse corto en la previsión de demanda que pasarse, porque la rotura de stock cuesta más que el exceso.
py: keras.layers.Layer ; keras.losses.Loss ; torch.nn.Module
rel: diferenciacion-automatica, construccion-de-modelos, funciones-de-perdida, api-funcional
en: custom layers & losses

## redes-siamesas
frase: Dos o más ramas idénticas, con los mismos pesos, que procesan entradas distintas para compararlas.
desc: Cada entrada pasa por la misma red, que la convierte en un vector, y después se mide la distancia entre los vectores. Se entrenan con pérdidas contrastivas o de triplete, que acercan los pares similares y alejan los diferentes hasta un margen. Así aprenden una medida de similitud y permiten reconocer clases nuevas con pocos ejemplos, solo comparando.
cuando: Para verificar si dos elementos son el mismo (firmas, caras, productos) o buscar similares, sobre todo con muchas clases y pocos ejemplos de cada una.
ej: Un sistema de desbloqueo facial compara el vector de tu cara con el que guardó al registrarte, sin reentrenar la red para cada usuario.
hp: margin: separación mínima exigida entre los pares distintos y los similares
py: torch.nn.TripletMarginLoss ; torch.nn.CosineEmbeddingLoss
rel: aprendizaje-contrastivo, embeddings, busqueda-por-similitud, distancia-y-similitud
en: Siamese networks (shared weights)

## multi-entrada-salida
frase: Modelos que combinan varios tipos de datos de entrada o resuelven varias tareas a la vez.
desc: Con varias entradas, cada una (una imagen, un texto, unas variables tabulares) se procesa por su propia rama y las representaciones se fusionan, normalmente concatenándolas. Con varias salidas, un tronco común alimenta cabezas distintas, cada una con su pérdida, y la pérdida total es una suma ponderada. Compartir la representación entre tareas relacionadas suele mejorar todas ellas.
cuando: Cuando los datos llegan en varias modalidades o hay varias predicciones relacionadas que pueden compartir aprendizaje.
ej: A partir de la foto y la descripción de un anuncio inmobiliario, predecir a la vez el precio (regresión) y el tipo de vivienda (clasificación).
py: keras.Model ; keras.layers.Concatenate ; `model.compile(loss={...}, loss_weights={...})`
rel: api-funcional, fine-tuning-multitarea, aprendizaje-multimodal, capas-y-perdidas-personalizadas
en: multi-input & multi-output models
