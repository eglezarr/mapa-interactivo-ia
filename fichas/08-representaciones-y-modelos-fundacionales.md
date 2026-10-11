# Fichas: Representaciones y modelos fundacionales

## aprendizaje-de-representaciones
frase: Aprender automáticamente a describir los datos con vectores útiles, en lugar de diseñar las variables a mano.
desc: Una buena representación transforma datos brutos (píxeles, palabras, audio) en vectores en los que lo relevante para muchas tareas es fácil de extraer, por ejemplo con un clasificador lineal. Las capas intermedias de una red ya son representaciones aprendidas. Hoy se aprenden a gran escala con autosupervisión y se reutilizan en muchas tareas mediante transferencia.
ej: Las representaciones de un modelo entrenado con millones de imágenes permiten clasificar especies de aves con solo 20 fotos de cada una.
rel: embeddings, aprendizaje-autosupervisado, transfer-learning, feature-engineering
en: representation learning

## embeddings
frase: Vectores numéricos que representan objetos de forma que los parecidos queden cerca.
desc: Un embedding asigna a cada elemento (una palabra, una frase, una imagen, un usuario, un producto) un vector denso de decenas a miles de dimensiones. La geometría codifica el significado: la distancia o el coseno entre vectores mide la similitud. Se obtienen de capas de embedding, de las capas intermedias de una red o de modelos entrenados específicamente para ello, y son la base de la búsqueda semántica, los recomendadores y el RAG.
cuando: Para comparar, buscar, agrupar o alimentar modelos con datos no estructurados, como texto o imágenes.
ej: "Perro" y "cachorro" tienen embeddings cercanos; "perro" e "hipoteca", lejanos.
py: sentence_transformers.SentenceTransformer
rel: espacio-latente, busqueda-por-similitud, capa-de-embedding, modelos-de-embeddings-de-texto, similitud-coseno
en: embeddings

## espacio-latente
frase: El espacio de representaciones internas de un modelo, donde los datos se describen con menos variables y más significado.
desc: Es el espacio de los códigos o embeddings que un modelo produce internamente, como el cuello de botella de un autoencoder o el ruido de entrada de una GAN. Suele tener menos dimensiones que los datos y una estructura con sentido: direcciones asociadas a atributos y regiones que corresponden a tipos de datos. En los modelos generativos, moverse por él permite interpolar entre ejemplos y editarlos.
ej: En una GAN de caras, avanzar en cierta dirección del espacio latente hace sonreír progresivamente a la persona generada.
rel: embeddings, cuello-de-botella, vae, aritmetica-de-embeddings, stable-diffusion
en: latent space

## aritmetica-de-embeddings
frase: Sumar y restar vectores de embeddings para combinar o trasladar significados.
desc: En un buen espacio de embeddings, muchas relaciones se codifican como direcciones aproximadamente constantes, así que se pueden manipular con operaciones vectoriales. El ejemplo clásico de word2vec es rey − hombre + mujer ≈ reina. La misma idea permite editar imágenes en el espacio latente de un generador. Las analogías no siempre funcionan y reflejan los sesgos de los datos de entrenamiento.
ej: Con vectores de palabras, París − Francia + Italia da un vector cuyo vecino más cercano es Roma.
py: gensim.models.KeyedVectors.most_similar
rel: word2vec, espacio-latente, embeddings, sesgo-algoritmico-y-atributos-sensibles
en: embedding arithmetic

## busqueda-por-similitud
frase: Encontrar los elementos más parecidos a una consulta comparando sus embeddings.
desc: Se convierte la consulta en un embedding y se buscan los vectores más cercanos de una colección, normalmente por similitud coseno o producto escalar. Con miles de elementos basta con compararlos todos; con millones se usan índices de vecinos aproximados, mucho más rápidos. Es la base de la búsqueda semántica, la recomendación de contenidos similares y la recuperación en RAG.
cuando: Para buscar por significado y no por coincidencia exacta de palabras, o para encontrar productos, imágenes o documentos parecidos.
ej: Buscar "zapatillas para correr por montaña" y encontrar productos descritos como "calzado de trail running".
py: sentence_transformers.util.semantic_search ; sklearn.neighbors.NearestNeighbors ; faiss.IndexFlatIP
rel: embeddings, busqueda-aproximada-de-vecinos, recuperacion-densa, similitud-coseno
en: similarity search

## aprendizaje-autosupervisado
frase: Aprender representaciones sin etiquetas, inventando tareas cuya respuesta sale de los propios datos.
desc: En lugar de etiquetas humanas, el modelo resuelve tareas construidas a partir de los datos: predecir la palabra siguiente o una palabra oculta, reconstruir parches tapados de una imagen o reconocer que dos vistas distintas son de la misma foto. Al resolverlas aprende representaciones generales que luego se adaptan a tareas concretas con pocos ejemplos etiquetados. Así se preentrenan los LLM y los grandes modelos de visión.
cuando: Cuando hay muchos datos sin etiquetar y pocas etiquetas, o para preentrenar modelos reutilizables.
ej: Un hospital con un millón de radiografías sin etiquetar preentrena un modelo autosupervisado y después lo ajusta con 2.000 radiografías etiquetadas.
rel: tarea-pretexto, aprendizaje-contrastivo, modelado-enmascarado, paradigmas-de-aprendizaje, modelos-fundacionales
en: self-supervised learning (SSL)

## tarea-pretexto
frase: Una tarea artificial, con la respuesta conocida gratis, que obliga al modelo a aprender algo útil.
desc: No interesa por sí misma, sino por lo que obliga a aprender. Ejemplos clásicos son predecir la rotación aplicada a una imagen, recomponer un puzle de parches, colorear una foto en blanco y negro o adivinar palabras ocultas. Si está bien elegida, resolverla exige entender la estructura de los datos, y esas representaciones sirven después para la tarea real.
ej: Para adivinar si una foto está girada 90°, el modelo necesita reconocer los objetos y saber cuál es su orientación natural.
rel: aprendizaje-autosupervisado, modelado-enmascarado, aprendizaje-contrastivo
en: pretext task

## aprendizaje-contrastivo
frase: Aprender acercando las representaciones de pares relacionados y alejando las de pares no relacionados.
desc: Se construyen pares positivos, como dos versiones aumentadas de la misma imagen o una imagen y su texto, y se tratan como negativos los demás ejemplos del lote. La pérdida empuja a que los positivos tengan embeddings parecidos y los negativos, distintos. Así se aprenden representaciones sin etiquetas que capturan qué hace iguales o diferentes a los datos. Es la base de SimCLR, MoCo y CLIP.
cuando: Para aprender embeddings sin etiquetas o a partir de pares naturales, como imagen y texto o pregunta y respuesta.
ej: Entrenar un buscador con pares de pregunta y respuesta correcta, de forma que cada respuesta quede cerca de su pregunta.
hp: temperature: escala las similitudes; cuanto más baja, más se castigan los negativos difíciles
py: lightly.loss.NTXentLoss ; sentence_transformers.sentence_transformer.losses.MultipleNegativesRankingLoss
rel: perdida-infonce, simclr-moco-byol-y-dino, clip, redes-siamesas
en: contrastive learning

## perdida-infonce
frase: La pérdida contrastiva estándar: una clasificación en la que hay que reconocer el par correcto entre muchos candidatos.
desc: Para cada ejemplo compara la similitud con su positivo frente a las similitudes con todos los negativos del lote, divididas por una temperatura, y aplica una entropía cruzada: es como clasificar cuál de N candidatos es el compañero correcto. Maximiza una cota inferior de la información mutua entre las dos vistas. Funciona mejor con lotes grandes, porque aportan más negativos.
ej: En un lote de 256 pares de imagen y texto, cada imagen debe identificar su descripción entre las 256 del lote.
hp: temperature: escala de las similitudes; valores típicos entre 0,05 y 0,5
py: lightly.loss.NTXentLoss ; `F.cross_entropy(similitudes / tau, etiquetas)`
rel: aprendizaje-contrastivo, entropia-cruzada, informacion-mutua, clip
en: InfoNCE loss

## simclr-moco-byol-y-dino
frase: Cuatro métodos de referencia para aprender representaciones visuales sin etiquetas.
desc: SimCLR aplica dos aumentos a cada imagen y usa aprendizaje contrastivo con los negativos del lote, por lo que necesita lotes enormes. MoCo guarda los negativos en una cola alimentada por un codificador que se actualiza lentamente (momentum), lo que permite lotes pequeños. BYOL y DINO prescinden de negativos: una red alumna imita a una red maestra que es una media móvil de ella misma, con mecanismos para evitar el colapso. DINOv2 da algunas de las mejores representaciones visuales de propósito general.
cuando: Para preentrenar extractores visuales con muchas imágenes sin etiquetar o elegir un modelo preentrenado de propósito general.
ej: Usar tal cual las características de DINOv2 para buscar imágenes similares o separar objetos del fondo, sin ajuste fino.
py: `AutoModel.from_pretrained("facebook/dinov2-base")`
rel: aprendizaje-contrastivo, colapso-de-la-representacion, perdida-infonce, destilacion-de-conocimiento
en: SimCLR, MoCo, BYOL & DINO

## modelado-enmascarado
frase: Ocultar partes de la entrada y entrenar al modelo para reconstruirlas.
desc: Se tapa una parte de los datos y el modelo debe predecirla a partir del resto. En texto, BERT oculta el 15 % de los tokens. En imagen, MAE oculta el 75 % de los parches y reconstruye sus píxeles con un decodificador ligero, y BEiT predice tokens visuales discretos de los parches ocultos. Obliga a aprender cómo se relacionan las partes de los datos y es un preentrenamiento muy eficiente.
cuando: Para preentrenar transformers de texto, imagen o audio con datos sin etiquetar.
ej: MAE ve solo una cuarta parte de los parches de la foto de un perro y reconstruye de forma plausible la cabeza y las patas que faltan.
hp: mask_ratio: fracción de parches ocultos (0,75 en MAE)
py: transformers.ViTMAEConfig ; transformers.ViTMAEForPreTraining
rel: mlm, tarea-pretexto, vit, autoencoder-denoising
en: masked modeling (MAE, BEiT)

## colapso-de-la-representacion
frase: Fallo del aprendizaje autosupervisado en el que el modelo da el mismo vector a todas las entradas.
desc: Si el objetivo solo pide que las vistas de un mismo ejemplo tengan representaciones parecidas, la solución trivial es devolver siempre lo mismo: la pérdida es mínima y la representación, inútil. Los métodos lo evitan con negativos (aprendizaje contrastivo), con asimetrías entre redes (BYOL, DINO) o con términos que fuerzan la varianza y la decorrelación entre dimensiones (VICReg, Barlow Twins).
ej: Durante el entrenamiento, la varianza de los embeddings cae a cero y todas las imágenes acaban en el mismo punto: señal de colapso.
rel: aprendizaje-contrastivo, simclr-moco-byol-y-dino, aprendizaje-autosupervisado, colapso-de-modos
en: representation collapse

## evaluacion-de-representaciones
frase: Medir la calidad de una representación por lo fácil que resulta resolver tareas con ella.
desc: Como no hay una métrica directa, se congela el modelo y se evalúan sus representaciones en tareas etiquetadas. El linear probing entrena solo un clasificador lineal encima; el k-NN clasifica por los vecinos más cercanos sin entrenar nada; el ajuste fino con pocos ejemplos mide la capacidad de adaptación. Una buena representación da buenos resultados con clasificadores simples.
cuando: Para comparar modelos preentrenados, o checkpoints de un entrenamiento autosupervisado.
ej: Con las representaciones congeladas de dos modelos, un clasificador lineal logra un 78 % y un 71 % de acierto en ImageNet: el primero representa mejor.
py: sklearn.linear_model.LogisticRegression ; sklearn.neighbors.KNeighborsClassifier
rel: aprendizaje-autosupervisado, extractor-de-caracteristicas, knn, regresion-logistica
en: representation evaluation (linear probing, k-NN)

## transfer-learning
frase: Reutilizar lo que un modelo aprendió en una tarea grande para resolver otra con menos datos.
desc: Un modelo preentrenado con muchos datos ya ha aprendido rasgos generales, como bordes y formas en imágenes o gramática y significado en texto. En lugar de entrenar desde cero, se parte de él y se adapta a la tarea nueva, ya sea como extractor de características fijo o ajustando sus pesos (fine-tuning). Reduce drásticamente los datos y el cómputo necesarios, y es la forma habitual de trabajar hoy.
cuando: Casi siempre que existe un modelo preentrenado en un dominio parecido y tus datos etiquetados son limitados.
ej: Adaptar una red preentrenada con ImageNet para detectar plagas en fotos de hojas con 1.500 imágenes etiquetadas.
py: keras.applications.ResNet50 ; timm.create_model ; torchvision.models.resnet50
rel: extractor-de-caracteristicas, fine-tuning, modelos-fundacionales, adaptacion-de-dominio
en: transfer learning

## extractor-de-caracteristicas
frase: Usar un modelo preentrenado y congelado para convertir los datos en vectores y entrenar encima un modelo simple.
desc: Se pasa cada dato por el modelo preentrenado sin su cabeza de clasificación y se guardan las activaciones de una capa final como vector de características. Sobre esos vectores se entrena un clasificador ligero, como una regresión logística o una capa densa. Es rápido, barato y robusto con muy pocos datos, aunque rinde menos que el ajuste fino si el dominio es muy distinto.
cuando: Con muy pocos datos, poco cómputo o un dominio parecido al del preentrenamiento.
ej: Extraer con una ResNet50 un vector de 2.048 dimensiones de cada foto de producto y entrenar una regresión logística que las clasifique en 30 categorías.
hp: include_top: si se mantiene la cabeza de clasificación original | pooling: cómo se resume el mapa final en un vector (media o máximo)
py: keras.applications.ResNet50
rel: transfer-learning, fine-tuning, embeddings, evaluacion-de-representaciones
en: pretrained model as feature extractor

## fine-tuning
frase: Continuar el entrenamiento de un modelo preentrenado con los datos de la tarea nueva.
desc: Se sustituye la cabeza de salida por una adecuada a la tarea y se entrenan los pesos, todos o una parte, con un learning rate pequeño para no destruir lo aprendido. Suele dar el mejor rendimiento cuando hay datos suficientes, a costa de más cómputo y riesgo de sobreajuste y de olvido. En modelos grandes se recurre a métodos eficientes como LoRA, que solo entrenan una pequeña fracción de los parámetros.
cuando: Cuando hay suficientes datos etiquetados (de cientos a miles) y el dominio difiere del de preentrenamiento.
ej: Ajustar un modelo tipo BERT en español con 5.000 correos etiquetados para clasificarlos en 8 tipos de incidencia.
hp: learning_rate: tamaño de paso, mucho menor que al entrenar desde cero | num_train_epochs: pocas épocas suelen bastar | warmup_steps: pasos de calentamiento del learning rate
py: transformers.Trainer ; transformers.TrainingArguments
rel: transfer-learning, congelar-y-descongelar-capas, peft, olvido-catastrofico, sft
en: fine-tuning

## congelar-y-descongelar-capas
frase: Decidir qué capas del modelo preentrenado se quedan fijas y cuáles se reentrenan.
desc: Congelar una capa impide que sus pesos cambien. Lo habitual es congelar el modelo base, entrenar primero la cabeza nueva y después descongelar las capas superiores, más específicas, para ajustarlas con un learning rate bajo. Las capas inferiores capturan rasgos genéricos y se suelen mantener. Con batch normalization conviene mantener congeladas sus estadísticas.
cuando: Al ajustar un modelo preentrenado con pocos datos, para no destruir lo aprendido al inicio del entrenamiento.
ej: Entrenar durante 5 épocas solo la cabeza de una EfficientNet y después descongelar sus 20 últimas capas otras 10 épocas con un learning rate diez veces menor.
py: `capa.trainable = False` ; `parametro.requires_grad = False`
rel: fine-tuning, learning-rates-discriminativos, transfer-learning, olvido-catastrofico
en: freezing & unfreezing layers

## learning-rates-discriminativos
frase: Usar learning rates distintos por capa: pequeños en las capas genéricas y mayores en las nuevas.
desc: Las capas inferiores de un modelo preentrenado contienen rasgos generales que conviene tocar poco, mientras que las superiores y la cabeza nueva necesitan adaptarse más. Asignar a cada grupo de capas su propio learning rate, creciente hacia la salida, ajusta el modelo sin destruir su conocimiento previo. Combina bien con descongelar las capas de forma gradual.
cuando: Al ajustar modelos preentrenados de visión o de lenguaje cuando el ajuste uniforme sobreajusta u olvida.
ej: Un learning rate de 10⁻⁵ en el cuerpo de un modelo de lenguaje y de 10⁻³ en su nueva cabeza de clasificación.
py: `torch.optim.AdamW([{"params": cuerpo.parameters(), "lr": 1e-5}, {"params": cabeza.parameters(), "lr": 1e-3}])`
rel: fine-tuning, congelar-y-descongelar-capas, learning-rate, lr-schedulers
en: discriminative learning rates

## adaptacion-de-dominio
frase: Hacer que un modelo entrenado con un tipo de datos funcione en otro distinto, a menudo sin etiquetas del nuevo.
desc: Cuando los datos de uso (dominio destino) difieren de los de entrenamiento (dominio origen), el rendimiento cae aunque la tarea sea la misma. La adaptación de dominio reduce esa diferencia: alineando las representaciones de ambos dominios, por ejemplo con un discriminador que no debe poder distinguirlos (DANN), ajustando con los pocos datos etiquetados disponibles o autoetiquetando los datos del destino.
cuando: Cuando el modelo debe funcionar con datos de otra fuente, sensor, hospital o época y allí hay pocas etiquetas.
ej: Un detector de peatones entrenado con imágenes sintéticas de videojuegos se adapta a las fotos reales de cámaras de tráfico.
rel: transfer-learning, deriva-de-datos, ood, datos-sinteticos
en: domain adaptation

## olvido-catastrofico
frase: Al aprender una tarea nueva, la red pierde lo que sabía hacer en las anteriores.
desc: Ajustar todos los pesos para una tarea nueva sobrescribe los que eran importantes para las anteriores, y el rendimiento en estas se desploma. Se mitiga con learning rates bajos y pocas épocas, mezclando datos antiguos con los nuevos (repetición), penalizando los cambios en los pesos importantes (EWC) o entrenando solo parámetros añadidos (adaptadores, LoRA). Es el principal obstáculo del aprendizaje continuo.
ej: Tras ajustar un LLM solo con textos médicos, empeora en conversación general y en programación.
rel: fine-tuning, peft, aprendizaje-online, congelar-y-descongelar-capas
en: catastrophic forgetting

## destilacion-de-conocimiento
frase: Entrenar un modelo pequeño (alumno) para que imite las salidas de uno grande (profesor).
desc: El alumno aprende a reproducir las probabilidades que da el profesor, no solo las etiquetas. Esas probabilidades, suavizadas con una temperatura, contienen información rica, como qué clases se parecen entre sí. La pérdida combina la divergencia KL con el profesor y la entropía cruzada con las etiquetas reales. Así se obtienen modelos mucho más pequeños y rápidos que conservan buena parte del rendimiento.
cuando: Para desplegar un modelo grande en móviles o con poca latencia, o para abaratar la inferencia.
ej: DistilBERT conserva alrededor del 97 % de la capacidad de comprensión de BERT con un 40 % menos de parámetros y un 60 % más de velocidad.
py: torch.nn.KLDivLoss ; keras.losses.KLDivergence
rel: divergencia-de-kullback-leibler, despliegue-e-inferencia, cuantizacion, destilacion-de-difusion
en: knowledge distillation

## modelos-fundacionales
frase: Modelos enormes, preentrenados con datos masivos, que sirven de base para muchas tareas distintas.
desc: Se entrenan una vez, con enormes cantidades de datos generalmente sin etiquetar, y después se adaptan a multitud de tareas mediante instrucciones, ejemplos o ajuste fino. Incluyen los LLM, los modelos de visión como DINOv2 o SAM, los multimodales como CLIP y los de series temporales. Han cambiado la forma de trabajar: hoy casi siempre se parte de uno de ellos.
ej: El mismo LLM resume contratos, escribe código y clasifica correos sin haberse entrenado específicamente para ninguna de esas tareas.
py: transformers.AutoModel
rel: que-es-un-modelo-fundacional, leyes-de-escalado, llm, transfer-learning
en: foundation models

## que-es-un-modelo-fundacional
frase: Un modelo general, entrenado a gran escala, que se convierte en la base sobre la que se construyen aplicaciones.
desc: El término, acuñado en 2021, destaca tres rasgos: escala (miles de millones de parámetros y de datos), generalidad (sirve para tareas para las que no se entrenó explícitamente) y homogeneización (muchas aplicaciones dependen de unos pocos modelos base, lo que también concentra sus defectos y sesgos). Suelen mostrar capacidades emergentes, como aprender de los ejemplos incluidos en la instrucción.
ej: Una empresa construye su asistente interno, su clasificador de incidencias y su buscador sobre el mismo modelo base de un proveedor.
rel: modelos-fundacionales, leyes-de-escalado, in-context-learning, sesgo-algoritmico-y-atributos-sensibles
en: what is a foundation model

## leyes-de-escalado
frase: Relaciones predecibles entre el error de un modelo y su tamaño, sus datos y su cómputo.
desc: Empíricamente, la pérdida de los modelos de lenguaje disminuye como una ley de potencias al aumentar los parámetros, los datos o el cómputo, de forma muy regular a lo largo de muchos órdenes de magnitud. Esto permite predecir el resultado de un entrenamiento caro a partir de otros pequeños y repartir el presupuesto: el trabajo Chinchilla mostró que, para un cómputo dado, conviene escalar parámetros y datos en proporción, con unos 20 tokens por parámetro.
ej: Con el mismo cómputo, un modelo de 70.000 millones de parámetros entrenado con 1,4 billones de tokens superó a otro de 280.000 millones entrenado con muchos menos datos.
rel: modelos-fundacionales, llm, preentrenamiento-de-llms, gpu-y-tpu
en: scaling laws
