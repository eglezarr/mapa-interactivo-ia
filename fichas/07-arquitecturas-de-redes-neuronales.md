# Fichas: Arquitecturas de redes neuronales

## bloques-comunes
frase: Piezas reutilizables que aparecen en casi todas las arquitecturas modernas.
desc: Más allá del tipo de red, ciertas piezas se repiten: las conexiones residuales, que permiten apilar muchas capas; las capas de normalización, que estabilizan el entrenamiento; y las capas de embedding, que convierten símbolos discretos en vectores. Conocerlas ayuda a leer cualquier arquitectura nueva como una combinación de bloques ya conocidos.
ej: Un bloque de transformer combina atención, una red densa, dos conexiones residuales y dos normalizaciones de capa.
rel: conexion-residual, capas-de-normalizacion, capa-de-embedding, bloque-transformer
en: common building blocks

## conexion-residual
frase: Sumar la entrada de un bloque a su salida para que la red solo tenga que aprender la corrección necesaria.
desc: En lugar de aprender una transformación completa H(x), el bloque aprende el residuo F(x) y devuelve x + F(x); si el bloque no aporta nada, basta con que F tienda a cero. El atajo deja pasar el gradiente directamente hacia las primeras capas, lo que permitió entrenar redes de cientos de capas (ResNet), y es una parte esencial de todos los transformers.
cuando: En cualquier red profunda, de más de una decena de capas, sea convolucional o transformer.
ej: ResNet-152 se entrena sin problemas gracias a sus atajos, mientras que una red equivalente sin ellos empeora al añadirle capas.
py: keras.layers.Add ; `x = x + bloque(x)`
rel: resnet, desvanecimiento-y-explosion-del-gradiente, bloque-transformer, api-funcional
en: residual (skip) connection

## capas-de-normalizacion
frase: Reescalan las activaciones durante el entrenamiento para mantenerlas en un rango estable.
desc: Normalizan las activaciones a media cero y varianza uno, y después aplican una escala y un desplazamiento aprendidos. Batch normalization calcula las estadísticas sobre el lote y es habitual en las CNN; depende del tamaño del lote y se comporta distinto en entrenamiento y en inferencia. Layer normalization normaliza cada ejemplo por separado y es la estándar en transformers; group normalization, intermedia entre ambas, funciona bien con lotes pequeños.
cuando: Batch norm, en CNN con lotes de tamaño razonable; layer norm, en transformers y RNN; group norm, con lotes pequeños.
ej: Añadir batch normalization tras cada convolución permite usar learning rates más altos y converger en menos épocas.
hp: momentum: en batch norm, inercia de la media móvil de las estadísticas | num_groups: en group norm, número de grupos de canales
py: keras.layers.BatchNormalization ; keras.layers.LayerNormalization ; torch.nn.BatchNorm2d ; torch.nn.GroupNorm
rel: estandarizacion, bloque-transformer, inicializacion-de-pesos, gdn
en: normalization layers (batch, layer, group)

## capa-de-embedding
frase: Una tabla que asigna a cada símbolo discreto un vector denso que se aprende durante el entrenamiento.
desc: Convierte índices enteros (palabras, tokens, productos, usuarios) en vectores de dimensión fija. Equivale a multiplicar un vector one-hot por una matriz, pero es mucho más eficiente. Los vectores empiezan aleatorios y se ajustan junto con el resto de la red, de modo que los símbolos que se comportan de forma parecida acaban cerca. Es la primera capa de todo modelo de lenguaje.
cuando: Siempre que la entrada sean categorías con muchos valores posibles: palabras, identificadores de usuario o de producto.
ej: En un recomendador, cada uno de los 50.000 productos se representa con un vector de 64 dimensiones aprendido a partir de las compras.
hp: input_dim / num_embeddings: número de símbolos distintos (Keras / PyTorch) | output_dim / embedding_dim: dimensión de cada vector (Keras / PyTorch)
py: keras.layers.Embedding ; torch.nn.Embedding
rel: embeddings, word-embeddings, codificacion-de-variables-categoricas, tokenizacion
en: embedding layer

## cnn
frase: Redes que recorren la entrada con pequeños filtros compartidos para detectar patrones locales.
desc: Las redes convolucionales aplican los mismos filtros en todas las posiciones de una imagen, una señal o una secuencia, lo que reduce enormemente los parámetros y aprovecha que un patrón significa lo mismo esté donde esté. Al apilar convoluciones y pooling, las primeras capas detectan bordes y texturas, y las profundas, partes y objetos. Dominaron la visión por computador a partir de 2012.
cuando: Con datos con estructura espacial o temporal local: imágenes, audio, señales y series.
ej: Una CNN que distingue las radiografías de tórax con neumonía de las normales a partir de los píxeles.
py: keras.layers.Conv2D ; torch.nn.Conv2d
rel: convolucion, pooling, arquitecturas-cnn-de-referencia, clasificacion-de-imagenes, transformers-vs-cnn-y-rnn
en: convolutional neural networks (CNN)

## convolucion
frase: Deslizar un pequeño filtro por la entrada y calcular en cada posición su producto con la zona que cubre.
desc: Un filtro, por ejemplo de 3 × 3 píxeles, recorre la imagen y en cada posición multiplica sus pesos por los valores que cubre y los suma, produciendo un mapa de respuestas. Sus pesos se comparten en todas las posiciones y solo mira una zona local, lo que la hace eficiente y adecuada para detectar patrones que se repiten. En las CNN los filtros no se diseñan a mano: se aprenden.
ej: Un filtro con valores negativos a la izquierda y positivos a la derecha responde con fuerza en los bordes verticales de una imagen.
hp: filters / out_channels: número de filtros (Keras / PyTorch) | kernel_size: tamaño del filtro
py: keras.layers.Conv2D ; torch.nn.Conv2d
rel: filtros-y-mapas-de-caracteristicas, padding-y-stride, campo-receptivo, cnn
en: convolution

## filtros-y-mapas-de-caracteristicas
frase: Cada filtro aprende a detectar un patrón; su salida es un mapa que indica dónde aparece.
desc: Una capa convolucional aplica muchos filtros en paralelo, y cada uno produce un mapa de características (un canal) que se activa donde encuentra su patrón. Los filtros de cada capa operan sobre todos los canales de la anterior, combinando patrones simples en otros cada vez más complejos. El número de filtros por capa suele crecer a medida que disminuye la resolución espacial.
ej: En una red entrenada con fotos de caras, los filtros de las primeras capas detectan bordes y colores, y los de las capas profundas, ojos y narices.
rel: convolucion, visualizacion-de-filtros-y-activaciones, campo-receptivo, cnn
en: filters & feature maps

## padding-y-stride
frase: Padding: bordes añadidos para conservar el tamaño. Stride: cuántos píxeles salta el filtro en cada paso.
desc: Sin relleno, cada convolución recorta los bordes de la entrada; con padding "same" se añaden ceros alrededor para que la salida conserve el tamaño. El stride indica el salto del filtro: con stride 2, la salida se reduce a la mitad en cada dimensión, una alternativa al pooling para bajar la resolución. Ambos determinan el tamaño de salida: ⌊(n + 2p − k)/s⌋ + 1.
ej: Una imagen de 32 × 32 con un filtro de 3 × 3, padding de 1 y stride de 2 da un mapa de 16 × 16.
hp: padding: relleno de los bordes ("valid" o "same" en Keras; número de píxeles en PyTorch) | strides / stride: salto del filtro (Keras / PyTorch)
py: keras.layers.Conv2D ; torch.nn.Conv2d
rel: convolucion, pooling, campo-receptivo, convolucion-transpuesta
en: padding & stride

## pooling
frase: Resumir cada pequeña región de un mapa en un solo valor para reducir su tamaño.
desc: El max pooling se queda con el valor máximo de cada ventana y el average pooling, con la media. Reducen la resolución, el cálculo y los parámetros de las capas siguientes, y dan cierta tolerancia a pequeños desplazamientos. El pooling global promedia cada mapa completo en un único número y suele sustituir a las capas densas finales en las CNN modernas.
cuando: Para reducir la resolución entre bloques convolucionales; el pooling global, al final de la red, antes de clasificar.
ej: Un max pooling de 2 × 2 convierte un mapa de 64 × 64 en uno de 32 × 32 conservando las respuestas más fuertes.
hp: pool_size / kernel_size: tamaño de la ventana (Keras / PyTorch)
py: keras.layers.MaxPooling2D ; keras.layers.GlobalAveragePooling2D ; torch.nn.MaxPool2d ; torch.nn.AdaptiveAvgPool2d
rel: padding-y-stride, equivarianza-e-invarianza, cnn, campo-receptivo
en: pooling

## campo-receptivo
frase: La región de la entrada que puede influir en una neurona concreta de la red.
desc: Una neurona de la primera capa convolucional solo ve el tamaño de su filtro, por ejemplo 3 × 3 píxeles. Cada capa adicional, cada pooling y cada stride amplían esa región, de modo que las neuronas profundas ven zonas grandes de la imagen y pueden reconocer objetos completos. Si el campo receptivo es menor que los objetos de interés, la red no puede verlos enteros; las convoluciones dilatadas lo amplían sin añadir parámetros.
ej: Tras apilar tres convoluciones de 3 × 3, cada neurona ve una región de 7 × 7 píxeles de la imagen original.
rel: convolucion, pooling, convolucion-1d, transformers-vs-cnn-y-rnn
en: receptive field

## equivarianza-e-invarianza
frase: Equivarianza: si la entrada se desplaza, la salida se desplaza igual. Invarianza: la salida no cambia.
desc: La convolución es equivariante a las traslaciones: un patrón desplazado produce la misma respuesta, desplazada. El pooling y la agregación global hacen que la salida final sea aproximadamente invariante: la clase no cambia por mover el objeto. Las CNN no son invariantes por sí solas a rotaciones o cambios de escala; esas invarianzas se enseñan con aumento de datos o con arquitecturas específicas.
ej: Un gato en una esquina de la foto o en el centro activa los mismos filtros en posiciones distintas, y la red lo clasifica igual.
rel: convolucion, pooling, data-augmentation, cnn
en: equivariance & invariance

## convolucion-1d
frase: Convolución que se desliza en una sola dimensión, para secuencias, series temporales y audio.
desc: El filtro recorre la secuencia a lo largo del tiempo detectando patrones locales, como subidas bruscas en una señal o combinaciones de palabras en un texto. Es mucho más rápida y paralelizable que una RNN. Con convoluciones dilatadas, que saltan pasos, el campo receptivo crece exponencialmente con la profundidad y capta dependencias largas, como en WaveNet o en las redes convolucionales temporales.
cuando: Para clasificar o predecir series temporales, señales y audio, sobre todo cuando importan los patrones locales.
ej: Detectar arritmias en un electrocardiograma reconociendo la forma de cada latido.
hp: filters: número de filtros | kernel_size: longitud del filtro | dilation_rate: separación entre los elementos del filtro
py: keras.layers.Conv1D ; torch.nn.Conv1d
rel: convolucion, redes-neuronales-para-series, campo-receptivo, rnn
en: 1D convolution

## convolucion-separable
frase: Dividir una convolución en un filtrado por canal y una mezcla de canales: casi igual de potente y mucho más barata.
desc: Una convolución normal filtra y mezcla canales a la vez. La separable en profundidad lo hace en dos pasos: primero aplica un filtro espacial independiente a cada canal (depthwise) y después una convolución de 1 × 1 que combina los canales (pointwise). Con filtros de 3 × 3 reduce el cálculo y los parámetros entre 8 y 9 veces, con una pérdida de precisión pequeña. Es la base de MobileNet y EfficientNet.
cuando: En modelos que deben ser ligeros: móviles, dispositivos embebidos o inferencia en tiempo real.
ej: MobileNet clasifica imágenes en un móvil con una fracción del cálculo de una CNN clásica gracias a las convoluciones separables.
hp: depth_multiplier: filtros espaciales por cada canal de entrada | groups: en PyTorch, igualarlo al número de canales hace la convolución depthwise
py: keras.layers.SeparableConv2D ; keras.layers.DepthwiseConv2D ; torch.nn.Conv2d
rel: mobilenet-y-efficientnet, convolucion, cnn, despliegue-e-inferencia
en: depthwise separable convolution

## convolucion-transpuesta
frase: La operación que aumenta la resolución de un mapa de características con pesos aprendidos.
desc: Hace el camino inverso, en forma, a una convolución con stride: aumenta el tamaño espacial de un mapa, por ejemplo de 16 × 16 a 32 × 32, con filtros que se aprenden. Se usa en los decodificadores de autoencoders, en los generadores de GAN y en redes de segmentación como U-Net. Puede producir artefactos en forma de tablero de ajedrez; una alternativa es aumentar la resolución por interpolación y aplicar después una convolución normal.
cuando: Cuando la red debe producir una salida con resolución espacial: imágenes generadas, máscaras de segmentación o reconstrucciones.
ej: El decodificador de U-Net recupera paso a paso la resolución completa de la imagen para clasificar cada píxel.
hp: strides / stride: factor de aumento de la resolución (Keras / PyTorch) | kernel_size: tamaño del filtro
py: keras.layers.Conv2DTranspose ; keras.layers.UpSampling2D ; torch.nn.ConvTranspose2d
rel: u-net, autoencoder-convolucional, dcgan, padding-y-stride
en: transposed convolution (upsampling)

## inspiracion-biologica-de-las-cnn
frase: Las CNN imitan, a grandes rasgos, la jerarquía del sistema visual de los mamíferos.
desc: Hubel y Wiesel descubrieron en los años 60 que las neuronas de la corteza visual primaria responden a bordes con cierta orientación dentro de una pequeña zona del campo visual, su campo receptivo, y que etapas posteriores combinan esas respuestas en patrones más complejos. El Neocognitrón de Fukushima (1980) y después LeNet trasladaron esa idea a redes con filtros locales, compartidos y organizados en jerarquía. La analogía es útil, pero aproximada.
ej: Los filtros que aprende la primera capa de una CNN se parecen a los detectores de bordes orientados medidos en la corteza visual.
rel: cnn, campo-receptivo, modelos-bioinspirados-y-multiescala, neurona-artificial
en: biological inspiration of CNNs

## rnn
frase: Redes con memoria que procesan secuencias paso a paso, actualizando un estado interno.
desc: Una red recurrente lee un elemento de la secuencia en cada paso y lo combina con su estado oculto anterior para producir uno nuevo, aplicando los mismos pesos en todos los pasos. Así procesa secuencias de cualquier longitud teniendo en cuenta el contexto pasado. Las RNN simples olvidan rápido, y LSTM y GRU resuelven ese problema. Los transformers las han desplazado en lenguaje, pero siguen siendo útiles en series temporales y en dispositivos con pocos recursos.
cuando: Para secuencias como series temporales, señales o texto, sobre todo si se procesan en tiempo real, paso a paso.
ej: Predecir la siguiente lectura de un sensor a partir de las anteriores, actualizando el estado con cada nueva medida.
py: keras.layers.SimpleRNN ; torch.nn.RNN
rel: estado-oculto-y-memoria, lstm, gru, bptt, transformers-vs-cnn-y-rnn
en: recurrent neural networks (RNN)

## estado-oculto-y-memoria
frase: El vector que la red recurrente arrastra de un paso al siguiente y que resume lo visto hasta entonces.
desc: En cada paso, el estado oculto se calcula a partir de la entrada actual y del estado anterior: hₜ = f(W·xₜ + U·hₜ₋₁ + b). Funciona como una memoria de tamaño fijo que comprime toda la historia relevante. Su capacidad es limitada: en las RNN simples la información antigua se diluye rápidamente, por lo que LSTM y GRU añaden puertas que deciden qué guardar y qué olvidar.
ej: Al leer "las llaves que dejé encima de la mesa ___", el estado oculto debe recordar que el sujeto está en plural para predecir "están".
rel: rnn, lstm, gru, bptt
en: hidden state & memory

## bptt
frase: Entrenar una RNN desplegándola en el tiempo y aplicando retropropagación a lo largo de todos los pasos.
desc: Se desenrolla la red como si cada paso temporal fuera una capa, todas con los mismos pesos, y se aplica retropropagación desde el final de la secuencia hacia el principio, sumando los gradientes de cada paso. En secuencias largas el gradiente se desvanece o explota y el coste crece con la longitud, así que se suele truncar a un número fijo de pasos (BPTT truncada).
ej: Para entrenar con textos de 10.000 caracteres, se retropropaga en tramos de 100 pasos, pasando el estado de cada tramo al siguiente.
rel: backpropagation, rnn, desvanecimiento-y-explosion-del-gradiente, lstm
en: backpropagation through time (BPTT)

## lstm
frase: RNN con una memoria interna y puertas que deciden qué recordar, qué olvidar y qué emitir.
desc: Añade a la RNN una celda de memoria que fluye entre pasos con cambios mínimos, regulada por tres puertas sigmoides: la de olvido decide qué borrar, la de entrada qué añadir y la de salida qué mostrar como estado oculto. Esa vía casi lineal deja pasar el gradiente durante muchos pasos, lo que permite aprender dependencias de cientos de pasos. Fue la arquitectura dominante en lenguaje y voz hasta la llegada de los transformers.
cuando: Para series temporales y secuencias con dependencias largas cuando no hay datos o recursos para un transformer.
ej: Predecir la demanda eléctrica de las próximas 24 horas a partir de las dos semanas anteriores.
hp: units / hidden_size: tamaño del estado (Keras / PyTorch) | return_sequences: devolver el estado de cada paso o solo el último | dropout: dropout sobre las entradas de cada paso
py: keras.layers.LSTM ; torch.nn.LSTM
rel: gru, rnn, desvanecimiento-y-explosion-del-gradiente, redes-neuronales-para-series
en: long short-term memory (LSTM)

## gru
frase: Variante simplificada de la LSTM, con dos puertas y sin celda de memoria separada.
desc: La GRU fusiona la celda y el estado oculto, y usa dos puertas: la de actualización decide cuánto del estado anterior se conserva y la de reinicio, cuánto se usa para calcular el candidato nuevo. Tiene menos parámetros que la LSTM, entrena más rápido y en muchas tareas rinde de forma parecida.
cuando: Como alternativa más ligera a la LSTM, sobre todo con pocos datos o recursos limitados.
ej: El teclado de un móvil sugiere la siguiente palabra con una GRU pequeña que funciona sin conexión.
hp: units / hidden_size: tamaño del estado (Keras / PyTorch) | num_layers: en PyTorch, número de capas apiladas
py: keras.layers.GRU ; torch.nn.GRU
rel: lstm, rnn, estado-oculto-y-memoria
en: gated recurrent unit (GRU)

## rnn-bidireccional
frase: Dos RNN que leen la secuencia en sentidos opuestos para que cada posición tenga contexto pasado y futuro.
desc: Una RNN recorre la secuencia de principio a fin y otra de fin a principio, y en cada posición se combinan sus estados. Así la representación de cada elemento tiene en cuenta lo que viene antes y lo que viene después. Solo sirve cuando la secuencia completa está disponible, no para predecir el futuro en tiempo real.
cuando: Para etiquetar o clasificar secuencias completas, como reconocer entidades en un texto o transcribir audio ya grabado.
ej: Para saber si "banco" se refiere a un asiento o a una entidad financiera, ayuda leer también las palabras que vienen después.
hp: merge_mode: cómo se combinan los dos sentidos (concatenar, sumar…) | bidirectional: en PyTorch, activa la lectura en ambos sentidos
py: keras.layers.Bidirectional ; torch.nn.LSTM
rel: rnn, lstm, bert, topologias-de-secuencia
en: bidirectional RNN

## echo-state-networks
frase: RNN con un depósito aleatorio y fijo de neuronas recurrentes en la que solo se entrena la capa de salida.
desc: Las redes de estado de eco, un tipo de reservoir computing, generan al azar una red recurrente grande, el depósito, y no la entrenan: la entrada lo excita y produce dinámicas ricas, como las ondas en un estanque. Solo se aprende una salida lineal, normalmente con regresión ridge, lo que hace el entrenamiento rapidísimo y sin retropropagación. Su radio espectral debe estar cerca de 1 para que el depósito recuerde sin volverse inestable.
cuando: Para predecir series temporales caóticas o dinámicas con muy poco cómputo de entrenamiento.
ej: Predecir la evolución de un sistema caótico, como el de Lorenz, entrenando solo una regresión lineal sobre los estados del depósito.
hp: units: número de neuronas del depósito | sr: radio espectral de la matriz recurrente | lr: tasa de fuga (leak rate) de las neuronas | ridge: regularización de la salida lineal
py: reservoirpy.nodes.Reservoir ; reservoirpy.nodes.Ridge
rel: rnn, ridge-lasso-y-elastic-net, redes-neuronales-para-series, rnn-clasicas
en: echo state networks (reservoir computing)

## rnn-clasicas
frase: Las primeras redes recurrentes: la memoria asociativa de Hopfield y las redes simples de Elman y Jordan.
desc: La red de Hopfield (1982) conecta todas las neuronas entre sí y funciona como memoria asociativa: almacena patrones como mínimos de una función de energía y recupera un patrón completo a partir de una versión parcial o ruidosa. Las redes de Elman y Jordan, de finales de los 80, añadieron unidades de contexto que copian el estado oculto o la salida del paso anterior, y son la base de la RNN simple actual.
ej: Una red de Hopfield que ha memorizado letras reconstruye una "A" completa a partir de una imagen con la mitad de los píxeles borrados.
py: keras.layers.SimpleRNN
rel: rnn, estado-oculto-y-memoria, echo-state-networks, breve-historia-de-la-ia
en: classic RNNs (Hopfield, Elman, Jordan)

## topologias-de-secuencia
frase: Cómo se relacionan entrada y salida en un modelo de secuencias: una a muchas, muchas a una o muchas a muchas.
desc: En one-to-many, una entrada genera una secuencia, como al describir una imagen con una frase. En many-to-one, una secuencia produce una sola salida, como al clasificar el sentimiento de una reseña. En many-to-many, una secuencia produce otra: alineada paso a paso, como al etiquetar cada palabra, o de distinta longitud mediante un codificador y un decodificador, como en la traducción.
ej: Traducir "how are you" a "¿cómo estás?" es many-to-many con codificador y decodificador; puntuar una reseña de 1 a 5 estrellas es many-to-one.
hp: return_sequences: devolver la salida de cada paso (many-to-many) o solo la última (many-to-one)
py: keras.layers.LSTM ; keras.layers.RepeatVector ; keras.layers.TimeDistributed
rel: rnn, tipos-de-transformer, prediccion-multipaso, analisis-de-sentimiento
en: sequence topologies (one-to-many, many-to-one, many-to-many)

## atencion-y-transformers
frase: La arquitectura que pondera qué partes de la entrada importan para cada elemento y que domina la IA actual.
desc: El mecanismo de atención permite que cada elemento de una secuencia consulte directamente a todos los demás y combine su información según su relevancia. El transformer (2017) construye una red entera con atención, capas densas, normalización y conexiones residuales, sin recurrencia ni convolución. Se paraleliza muy bien y escala con los datos y el cómputo, así que es la base de los LLM, de los modelos de visión modernos y de los multimodales.
ej: GPT, BERT, ViT, CLIP y muchos modelos de texto a imagen se basan en transformers.
rel: mecanismo-de-atencion, bloque-transformer, llm, transformers-vs-cnn-y-rnn
en: attention & Transformers

## mecanismo-de-atencion
frase: Combinar la información de todos los elementos de una secuencia ponderando cada uno por su relevancia.
desc: Para cada elemento, la atención calcula una puntuación de relevancia con cada uno de los demás, la convierte en pesos con una softmax y devuelve la media ponderada de sus representaciones. Así cada posición accede directamente a la información que necesita, por lejos que esté, en lugar de depender de una memoria comprimida como las RNN. Nació para mejorar la traducción automática y acabó dando lugar al transformer.
ej: Al traducir "the red car", para generar "rojo" el modelo presta casi toda su atención a "red".
py: keras.layers.Attention ; torch.nn.functional.scaled_dot_product_attention
rel: query-key-y-value, atencion-por-producto-escalar-escalado, autoatencion, mapas-de-atencion
en: attention mechanism

## query-key-y-value
frase: Cada elemento emite una consulta, una clave y un valor: la consulta busca, la clave se compara y el valor se entrega.
desc: La atención proyecta cada elemento en tres vectores con matrices aprendidas. La consulta (query) expresa qué busca; la clave (key), qué ofrece; el valor (value), qué información aporta si resulta elegido. La relevancia entre dos elementos es el producto escalar de la consulta de uno con la clave del otro, y la salida es la suma de los valores ponderada por esas relevancias.
ej: En "el gato que perseguía al perro maulló", la consulta de "maulló" encaja con la clave de "gato", y su valor aporta quién realiza la acción.
rel: mecanismo-de-atencion, atencion-por-producto-escalar-escalado, autoatencion, atencion-cruzada
en: query, key & value

## atencion-por-producto-escalar-escalado
frase: La fórmula de la atención: softmax(QKᵀ/√d)·V.
desc: Multiplica las consultas por las claves para obtener una matriz de puntuaciones, la divide por la raíz de la dimensión d, aplica una softmax por filas para obtener pesos y los usa para promediar los valores. La división por √d evita que, en dimensiones altas, los productos escalares crezcan tanto que la softmax se sature y anule el gradiente. Su coste crece con el cuadrado de la longitud de la secuencia.
ej: Con secuencias de 1.000 tokens, la matriz de puntuaciones tiene un millón de entradas por cada cabeza y cada capa.
py: torch.nn.functional.scaled_dot_product_attention
rel: query-key-y-value, softmax, atencion-multicabeza, ventana-de-contexto
en: scaled dot-product attention

## autoatencion
frase: Atención de una secuencia sobre sí misma: cada elemento se relaciona con todos los demás de la misma secuencia.
desc: En la autoatención, consultas, claves y valores salen de la misma secuencia, así que cada posición actualiza su representación con información de todas las demás. Capta dependencias de cualquier distancia en un solo paso y en paralelo para todas las posiciones. Es la operación central de los codificadores y decodificadores transformer.
ej: En "el animal no cruzó la calle porque estaba cansado", la autoatención relaciona "estaba cansado" con "animal" y no con "calle".
py: keras.layers.MultiHeadAttention ; torch.nn.MultiheadAttention
rel: mecanismo-de-atencion, atencion-multicabeza, atencion-cruzada, bloque-transformer
en: self-attention

## atencion-multicabeza
frase: Varias atenciones en paralelo, cada una libre de fijarse en un tipo de relación distinto.
desc: En lugar de una sola atención, se aplican varias cabezas en paralelo, cada una con sus propias proyecciones de consultas, claves y valores en un subespacio más pequeño. Cada cabeza puede especializarse en un tipo de relación, como la sintaxis, la correferencia o la posición, y sus salidas se concatenan y se proyectan. Cuesta aproximadamente lo mismo que una única atención de dimensión completa.
ej: En un modelo de lenguaje, una cabeza puede seguir qué adjetivo modifica a qué sustantivo, y otra, a qué nombre se refiere cada pronombre.
hp: num_heads: número de cabezas | key_dim / embed_dim: dimensión de cada cabeza en Keras, dimensión total del modelo en PyTorch | dropout: dropout sobre los pesos de atención
py: keras.layers.MultiHeadAttention ; torch.nn.MultiheadAttention
rel: atencion-por-producto-escalar-escalado, autoatencion, bloque-transformer, mapas-de-atencion
en: multi-head attention

## atencion-cruzada
frase: Atención en la que una secuencia consulta a otra distinta.
desc: Las consultas vienen de una secuencia y las claves y valores, de otra. Es el puente entre el codificador y el decodificador en traducción: cada palabra que se genera consulta la frase original. También conecta modalidades: en los modelos de texto a imagen, la imagen en construcción consulta el texto de la instrucción, y en muchos modelos multimodales el texto consulta la imagen.
ej: En Stable Diffusion, la atención cruzada inyecta la descripción "un astronauta a caballo" en cada paso de la generación de la imagen.
py: keras.layers.MultiHeadAttention ; torch.nn.MultiheadAttention
rel: autoatencion, tipos-de-transformer, stable-diffusion, arquitectura-mllm
en: cross-attention

## mascara-causal
frase: Impide que cada posición vea las posiciones futuras, para que el modelo genere de izquierda a derecha.
desc: En un modelo autorregresivo, la predicción del token siguiente no puede depender de tokens que todavía no existen. La máscara causal anula, antes de la softmax, las puntuaciones de atención hacia posiciones posteriores, de modo que cada token solo ve los anteriores. Así el modelo se entrena en paralelo con toda la secuencia y después genera token a token.
ej: Al entrenar con "el cielo es azul", la posición de "es" solo ve "el cielo", nunca "azul".
hp: is_causal: aplica la máscara causal en el cálculo de la atención
py: torch.nn.functional.scaled_dot_product_attention ; torch.nn.Transformer.generate_square_subsequent_mask
rel: modelos-autorregresivos, gpt, clm, autoatencion
en: causal mask

## codificacion-posicional
frase: Información sobre el orden de los elementos, que la atención por sí sola no tiene.
desc: La atención trata la entrada como un conjunto: sin ayuda, "el perro muerde al hombre" y "el hombre muerde al perro" serían indistinguibles. La codificación posicional añade a cada token una señal de su posición. El transformer original usaba senos y cosenos de distintas frecuencias; otros modelos aprenden un vector por posición, y los LLM actuales suelen usar RoPE, que rota consultas y claves según la posición y generaliza mejor a contextos largos.
ej: Sin codificación posicional, un modelo de traducción no distinguiría "Juan llama a Ana" de "Ana llama a Juan".
py: keras_hub.layers.SinePositionEncoding ; keras_hub.layers.PositionEmbedding ; keras_hub.layers.RotaryEmbedding
rel: autoatencion, bloque-transformer, ventana-de-contexto, transformers-vs-cnn-y-rnn
en: positional encoding

## bloque-transformer
frase: La unidad que se apila para construir un transformer: atención, red densa, residuos y normalización.
desc: Cada bloque aplica autoatención multicabeza seguida de una red feed-forward de dos capas que procesa cada posición por separado, normalmente expandiendo la dimensión cuatro veces. Ambas subcapas van envueltas en conexiones residuales y normalización de capa, que en los modelos modernos se aplica antes de cada subcapa (pre-norm) para estabilizar el entrenamiento. Un modelo completo es una pila de bloques idénticos.
ej: BERT-base apila 12 bloques con 12 cabezas de atención y representaciones de 768 dimensiones.
hp: num_heads / nhead: cabezas de atención (Keras / PyTorch) | intermediate_dim / dim_feedforward: tamaño de la red densa interna (Keras / PyTorch) | dropout: dropout en la atención y en la red densa
py: keras_hub.layers.TransformerEncoder ; torch.nn.TransformerEncoderLayer
rel: atencion-multicabeza, conexion-residual, capas-de-normalizacion, tipos-de-transformer
en: Transformer block

## tipos-de-transformer
frase: Tres configuraciones: solo codificador para entender, solo decodificador para generar y ambos para transformar.
desc: Los modelos solo codificador, como BERT, usan atención bidireccional y producen representaciones de toda la entrada: son ideales para clasificar o extraer información. Los solo decodificador, como GPT, usan máscara causal y generan texto token a token: son la base de los LLM. Los codificador-decodificador, como T5 o el transformer original, leen una entrada con el codificador y generan otra secuencia con el decodificador, unidos por atención cruzada; encajan en traducción y resumen.
ej: BERT para clasificar correos, GPT para redactarlos y T5 para traducirlos.
py: torch.nn.Transformer ; transformers.AutoModelForSeq2SeqLM
rel: bert, gpt, t5, atencion-cruzada, mascara-causal
en: encoder, decoder & encoder-decoder Transformers

## token-cls
frase: Un token especial al principio de la secuencia cuya representación final resume toda la entrada.
desc: Modelos como BERT y ViT añaden al inicio un token [CLS] sin significado propio. Gracias a la autoatención, su representación en la última capa recoge información de toda la secuencia y se usa como entrada de la cabeza de clasificación. Una alternativa habitual es promediar las representaciones de todos los tokens (mean pooling), que a menudo funciona igual o mejor para obtener embeddings de frases.
ej: Para clasificar el sentimiento de una reseña con BERT, el vector final del token [CLS] pasa a una capa densa con softmax.
py: `salida.last_hidden_state[:, 0]`
rel: bert, vit, vocabulario-y-tokens-especiales, modelos-de-embeddings-de-texto
en: [CLS] token

## transformers-vs-cnn-y-rnn
frase: Atención global y paralela frente a convoluciones locales y recurrencia secuencial.
desc: Las RNN procesan paso a paso, se paralelizan mal y olvidan las dependencias lejanas. Las CNN ven zonas locales y necesitan muchas capas para relacionar elementos lejanos, pero incorporan supuestos útiles, como la localidad, que las hacen eficientes con pocos datos. Los transformers relacionan cualquier par de posiciones en una sola capa y se paralelizan muy bien, pero su coste es cuadrático con la longitud y necesitan más datos. Con datos y cómputo abundantes, suelen ganar.
cuando: Transformers, con muchos datos o modelos preentrenados; CNN o RNN, con pocos datos, secuencias muy largas o hardware limitado.
ej: Con 10.000 imágenes, una ResNet suele superar a un ViT entrenado desde cero; con cientos de millones de imágenes, gana el ViT.
rel: cnn, rnn, atencion-y-transformers, vit
en: Transformers vs CNNs & RNNs

## autoencoders
frase: Redes que aprenden a reconstruir su propia entrada pasando por una representación comprimida.
desc: Un codificador transforma la entrada en un código de menor tamaño y un decodificador intenta reconstruir la entrada a partir de él. Como el código es un cuello de botella, la red debe quedarse con lo esencial de los datos. Se entrenan sin etiquetas y sirven para reducir la dimensión, eliminar ruido, detectar anomalías (lo que se reconstruye mal es raro) y como base de modelos generativos como el VAE.
cuando: Para comprimir, eliminar ruido o detectar anomalías en datos complejos, como imágenes o señales, sin necesidad de etiquetas.
ej: Un autoencoder entrenado con las vibraciones de motores sanos reconstruye mal las de un motor averiado, lo que delata la avería.
rel: cuello-de-botella, vae, deteccion-por-reconstruccion-o-prediccion, pca
en: autoencoders

## cuello-de-botella
frase: El codificador comprime, el código es el cuello de botella y el decodificador reconstruye.
desc: El codificador reduce progresivamente la dimensión hasta el código o representación latente, la capa más estrecha, y el decodificador hace el camino inverso. El tamaño del cuello de botella marca el compromiso: si es muy grande, la red puede copiar la entrada sin aprender nada útil; si es muy pequeño, la reconstrucción pierde detalle. Tras el entrenamiento, el codificador se puede usar por separado como extractor de características.
ej: Comprimir imágenes de dígitos manuscritos de 784 píxeles en un código de 32 números y reconstruirlas casi sin pérdida visible.
rel: autoencoders, espacio-latente, autoencoder-undercomplete, extraccion-de-caracteristicas
en: encoder, decoder & bottleneck

## autoencoder-undercomplete
frase: Autoencoder con un código más pequeño que la entrada, lo que le obliga a comprimir.
desc: Es el autoencoder básico: la restricción viene solo del tamaño del cuello de botella. Con capas lineales y error cuadrático aprende el mismo subespacio que PCA; con capas no lineales aprende una versión no lineal y más potente. Si el codificador y el decodificador son demasiado potentes, puede memorizar sin extraer una estructura útil.
cuando: Para reducir la dimensión de forma no lineal o comprimir datos con estructura compleja.
ej: Reducir a 10 dimensiones los espectros de 2.000 longitudes de onda de unas muestras químicas para visualizarlas y agruparlas.
rel: pca, cuello-de-botella, autoencoder-sparse, reduccion-de-la-dimensionalidad
en: undercomplete autoencoder

## autoencoder-sparse
frase: Autoencoder que obliga a que, para cada entrada, solo unas pocas neuronas del código estén activas.
desc: En lugar de estrechar el código, o además de hacerlo, añade a la pérdida una penalización sobre las activaciones, como la norma L1 o una divergencia KL respecto a una activación media pequeña. El código puede ser incluso más grande que la entrada, pero cada neurona se especializa en un rasgo y solo se activa cuando aparece. Produce representaciones más interpretables y hoy se usa para analizar los rasgos internos de los LLM.
cuando: Para aprender rasgos interpretables y especializados, o cuando el código puede ser grande pero se quiere disperso.
ej: Entrenado con parches de imágenes naturales, cada neurona del código acaba respondiendo a un borde con una orientación concreta.
hp: activity_regularizer: penalización sobre las activaciones del código, por ejemplo L1
py: keras.layers.Dense ; keras.regularizers.L1
rel: autoencoders, penalizacion-l1-y-l2, divergencia-de-kullback-leibler, explicabilidad-en-deep-learning
en: sparse autoencoder

## autoencoder-denoising
frase: Autoencoder que recibe la entrada corrompida y aprende a reconstruir la versión limpia.
desc: Se añade ruido a la entrada o se borran partes, y la red debe reconstruir el original. Como no puede limitarse a copiar, aprende la estructura de los datos limpios, lo que da representaciones más robustas. La misma idea, eliminar ruido de forma gradual, está en el corazón de los modelos de difusión, y la de reconstruir lo borrado, en el preentrenamiento de BERT y de MAE.
cuando: Para limpiar señales o imágenes ruidosas, o para aprender representaciones robustas sin etiquetas.
ej: Limpiar documentos escaneados con manchas y arrugas para mejorar el reconocimiento de texto posterior.
hp: stddev: intensidad del ruido que se añade a la entrada durante el entrenamiento
py: keras.layers.GaussianNoise
rel: inyeccion-de-ruido, modelos-de-difusion, modelado-enmascarado, autoencoders
en: denoising autoencoder

## autoencoder-convolucional
frase: Autoencoder con capas convolucionales, adecuado para imágenes y señales.
desc: El codificador usa convoluciones y reducciones de resolución para comprimir la imagen en mapas de características pequeños, y el decodificador usa convoluciones transpuestas o aumento de resolución para recuperarla. Aprovecha la estructura espacial con muchos menos parámetros que un autoencoder de capas densas y reconstruye las imágenes con más detalle.
cuando: Con imágenes, espectrogramas u otros datos con estructura espacial.
ej: Detectar defectos en piezas fabricadas: el autoencoder, entrenado solo con piezas correctas, reconstruye mal las zonas con arañazos.
py: keras.layers.Conv2D ; keras.layers.Conv2DTranspose
rel: autoencoders, convolucion-transpuesta, u-net, deteccion-por-reconstruccion-o-prediccion
en: convolutional autoencoder

## autoencoder-contractivo
frase: Autoencoder que penaliza que su código cambie mucho ante pequeñas variaciones de la entrada.
desc: Añade a la pérdida la norma de la matriz jacobiana del codificador, es decir, de las derivadas del código respecto a la entrada. Así el código es poco sensible a las direcciones de variación irrelevantes y solo sigue las que importan para reconstruir los datos, que suelen coincidir con la variedad sobre la que viven. Es una alternativa analítica al autoencoder de eliminación de ruido.
cuando: Para aprender representaciones estables frente a pequeñas perturbaciones de la entrada.
ej: Con fotos de dígitos, el código cambia poco si el trazo se vuelve algo más grueso y mucho si cambia el dígito.
rel: autoencoder-denoising, manifold-learning, autoencoders, regularizacion-en-redes
en: contractive autoencoder

## modelos-bioinspirados-y-multiescala
frase: Arquitecturas que imitan etapas del sistema visual humano o analizan la imagen a varias escalas.
desc: Incorporan conocimiento de la neurociencia y del procesamiento de señales: no linealidades como la normalización divisiva, observada en la corteza visual, y representaciones multiescala, como las pirámides o las wavelets, que describen la imagen a distintas resoluciones. Dan modelos más pequeños, interpretables y alineados con la percepción humana, útiles en compresión de imagen y en medidas de calidad perceptual.
ej: Una red de pocas capas inspiradas en la retina y la corteza visual predice qué distorsiones de una imagen nota más una persona.
rel: gdn, perceptnet, modelos-multiescala, inspiracion-biologica-de-las-cnn
en: bio-inspired & multiscale models

## gdn
frase: No linealidad que divide cada respuesta por una combinación de las respuestas vecinas, como en la corteza visual.
desc: La normalización divisiva generalizada divide la salida de cada canal por una suma ponderada y aprendida de las magnitudes de los demás canales. Imita la normalización divisiva de las neuronas visuales, que reduce la redundancia entre ellas, y gaussianiza y desacopla las respuestas de forma muy eficaz. Es una pieza clave de la compresión de imagen aprendida con redes neuronales.
cuando: En compresión de imagen aprendida y en modelos que buscan representaciones visuales eficientes o perceptuales.
ej: Los códecs de imagen basados en redes sustituyen las ReLU por capas GDN y obtienen archivos más pequeños con la misma calidad visual.
py: compressai.layers.GDN
rel: capas-de-normalizacion, perceptnet, modelos-bioinspirados-y-multiescala, metricas-perceptuales
en: generalized divisive normalization (GDN)

## perceptnet
frase: Red pequeña, inspirada en las etapas del sistema visual, que mide cuán distintas percibimos dos imágenes.
desc: Encadena capas que imitan la retina, el núcleo geniculado lateral y la corteza visual primaria, con convoluciones y normalizaciones divisivas (GDN). Se entrena para que la distancia entre las representaciones de dos imágenes coincida con la diferencia que perciben las personas. Con muchos menos parámetros que las medidas basadas en grandes redes, predice bien los juicios humanos sobre la calidad de imagen.
cuando: Como distancia perceptual entre imágenes cuando importa la coherencia con la visión humana y el bajo coste de cálculo.
ej: Ordenar varias versiones comprimidas de una foto según cuánto se nota la compresión, como lo haría un grupo de observadores.
rel: gdn, metricas-perceptuales, modelos-bioinspirados-y-multiescala, calidad-de-imagen
en: PerceptNet

## modelos-multiescala
frase: Analizar la entrada a varias resoluciones a la vez, de los detalles finos a la estructura global.
desc: Muchas estructuras de una imagen o una señal solo se ven a cierta escala: los bordes, a resolución alta; las formas generales, a resolución baja. Las representaciones multiescala, como las pirámides gaussianas y laplacianas o las wavelets, descomponen la entrada por escalas. En redes neuronales, las pirámides de características combinan mapas de varias resoluciones para detectar a la vez objetos pequeños y grandes.
cuando: Cuando los patrones de interés aparecen a tamaños muy distintos, como objetos cercanos y lejanos en una misma imagen.
ej: Un detector de peatones con pirámide de características encuentra tanto a las personas cercanas como a las que están al fondo de la escena.
py: torchvision.ops.FeaturePyramidNetwork ; skimage.transform.pyramid_laplacian
rel: piramides-gaussiana-y-laplaciana, wavelets, deteccion-de-objetos, modelos-bioinspirados-y-multiescala
en: multiscale models
