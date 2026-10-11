# Fichas: Lenguaje y LLMs

## nlp
frase: La rama de la IA que permite a las máquinas entender, analizar y generar lenguaje humano.
desc: El procesamiento del lenguaje natural abarca desde tareas clásicas, como clasificar textos, extraer entidades o traducir, hasta la generación de texto libre. Evolucionó de las reglas escritas a mano a los modelos estadísticos, después a las redes recurrentes y, desde 2018, a los transformers preentrenados. Hoy los LLM resuelven la mayoría de tareas con un único modelo.
ej: Un sistema que lee las reclamaciones de los clientes, detecta el producto mencionado, mide el enfado y redacta un borrador de respuesta.
rel: tareas-de-nlp, representacion-del-texto, modelos-de-lenguaje, llm
en: natural language processing (NLP)

## tareas-de-nlp
frase: Los problemas típicos con texto: clasificar, extraer, traducir, resumir, responder y generar.
desc: Las tareas se agrupan por el tipo de salida: clasificar un texto completo (tema, sentimiento, spam), etiquetar cada palabra (entidades, categorías gramaticales), extraer respuestas o relaciones y generar texto nuevo (traducción, resumen, diálogo). Antes cada tarea tenía su propio modelo; hoy un LLM, o un transformer preentrenado y ajustado, resuelve casi todas.
ej: Detectar en una noticia las entidades "Banco de España" (organización), "Madrid" (lugar) y "3 de marzo" (fecha).
py: transformers.pipeline
rel: nlp, analisis-de-sentimiento, benchmarks-de-nlp, topologias-de-secuencia
en: NLP tasks

## analisis-de-sentimiento
frase: Determinar si un texto expresa una opinión positiva, negativa o neutra.
desc: Es la tarea de clasificación de texto más conocida. Puede ser global (el tono de una reseña entera), por aspectos (qué opina el cliente del precio y qué de la atención) o de emociones más finas. Va desde diccionarios de palabras positivas y negativas hasta transformers ajustados, que captan mejor la negación, el contexto y, en parte, la ironía.
cuando: Para medir la opinión de los clientes en reseñas, encuestas, redes sociales o llamadas transcritas.
ej: Clasificar miles de reseñas de un hotel y descubrir que las negativas mencionan sobre todo el ruido nocturno.
py: `pipeline("sentiment-analysis")`
rel: tareas-de-nlp, bert, tf-idf, regresion-logistica
en: sentiment analysis

## benchmarks-de-nlp
frase: Conjuntos de tareas estándar para comparar modelos de lenguaje con las mismas reglas.
desc: GLUE (2018) reunió nueve tareas de comprensión, como inferencia, paráfrasis o sentimiento, con una puntuación global; SuperGLUE lo endureció cuando los modelos superaron el nivel humano. Permitieron medir el progreso de BERT y sus sucesores, pero se saturaron en pocos años. Hoy los LLM se evalúan con benchmarks de conocimiento, razonamiento y programación.
ej: BERT elevó la puntuación de GLUE en más de 7 puntos respecto al mejor modelo anterior.
py: `load_dataset("nyu-mll/glue", "sst2")`
rel: benchmarks-y-saturacion, evaluacion-de-llms, bert, tareas-de-nlp
en: NLP benchmarks (GLUE)

## representacion-del-texto
frase: Cómo convertir el texto en números que un modelo pueda procesar.
desc: Los modelos solo operan con números, así que el texto debe representarse numéricamente. Los métodos clásicos cuentan palabras (bolsa de palabras, TF-IDF) y producen vectores enormes y dispersos. Los modernos dividen el texto en tokens y asignan a cada uno un embedding denso, primero estático (word2vec) y después contextual, que depende de la frase en la que aparece.
ej: Con un modelo contextual, "el banco cerró" se representa con vectores distintos según hable de una entidad financiera o de un asiento.
rel: bolsa-de-palabras, tf-idf, tokenizacion, word-embeddings, embeddings-estaticos-vs-contextuales
en: text representation

## preprocesado-de-texto
frase: Limpiar y normalizar el texto antes de representarlo: minúsculas, tokens, palabras vacías y raíces.
desc: En los métodos clásicos, el preprocesado reduce el ruido y el vocabulario: pasar a minúsculas, quitar signos o etiquetas HTML, separar en palabras, eliminar las palabras vacías (stopwords), como "el" o "de", y reducir las palabras a su raíz (stemming) o a su forma de diccionario (lematización). Con transformers se hace mucho menos, porque el tokenizador y el propio modelo gestionan la variación del texto.
cuando: Con modelos clásicos, como la bolsa de palabras o TF-IDF; con transformers, solo una limpieza básica.
ej: "Corriendo", "corrió" y "correrán" se reducen a la raíz "corr" para contarse como la misma palabra.
py: nltk.tokenize.word_tokenize ; nltk.stem.SnowballStemmer
rel: bolsa-de-palabras, tf-idf, tokenizacion, eda
en: text preprocessing

## bolsa-de-palabras
frase: Representar un texto por cuántas veces aparece cada palabra del vocabulario, ignorando el orden.
desc: One-hot representa cada palabra como un vector con un 1 en su posición del vocabulario. La bolsa de palabras suma esos vectores: cada documento es un vector del tamaño del vocabulario con el recuento de cada palabra. Es simple, interpretable y funciona sorprendentemente bien para clasificar, pero pierde el orden ("el perro mordió al hombre" y "el hombre mordió al perro" coinciden) y no capta sinónimos. Los n-gramas recuperan algo del orden local.
cuando: Como línea base rápida para clasificar textos, sobre todo con pocos datos.
ej: Un filtro de spam que cuenta palabras como "gratis", "premio" o "urgente" y pasa los recuentos a una regresión logística.
hp: ngram_range: tamaño de las secuencias de palabras que se cuentan (1 = palabras sueltas) | min_df: frecuencia mínima para incluir una palabra en el vocabulario | max_features: tamaño máximo del vocabulario
py: sklearn.feature_extraction.text.CountVectorizer
rel: tf-idf, preprocesado-de-texto, codificacion-de-variables-categoricas, word-embeddings
en: one-hot & bag of words

## tf-idf
frase: Pondera cada palabra por su frecuencia en el documento y por lo rara que es en el conjunto.
desc: TF-IDF multiplica la frecuencia de un término en un documento (TF) por el logaritmo inverso de la fracción de documentos que lo contienen (IDF). Así, las palabras presentes en casi todos los documentos, como "de" o "que", pesan poco, y las características de cada documento, mucho. Es la base clásica de los buscadores y sigue siendo una línea base fuerte para clasificar y comparar textos.
cuando: Para clasificar o buscar textos con métodos clásicos, como línea base o con pocos recursos.
ej: En una colección de recetas, "azafrán" tiene un IDF alto y pesa mucho en la paella; "sal" aparece en casi todas y apenas cuenta.
hp: ngram_range: tamaño de las secuencias de palabras | max_df / min_df: descartan las palabras demasiado frecuentes o demasiado raras | sublinear_tf: usa el logaritmo de la frecuencia del término
py: sklearn.feature_extraction.text.TfidfVectorizer
rel: bolsa-de-palabras, busqueda-hibrida, metodos-filtro, analisis-de-sentimiento
en: TF-IDF

## tokenizacion
frase: Dividir el texto en las unidades (tokens) que procesa el modelo: palabras, trozos de palabra o caracteres.
desc: Es el primer paso de cualquier modelo de lenguaje: el texto se corta en tokens y cada uno se sustituye por su número en el vocabulario. La elección es un compromiso: las palabras completas dan vocabularios enormes y no cubren palabras nuevas; los caracteres dan secuencias muy largas; las subpalabras, el estándar actual, equilibran ambas cosas. El número de tokens determina el coste y el límite de contexto de un LLM.
ej: Un tokenizador puede dividir "desafortunadamente" en "des", "afortunada" y "mente".
py: transformers.AutoTokenizer ; tiktoken.get_encoding
rel: tokenizacion-por-subpalabras, vocabulario-y-tokens-especiales, ventana-de-contexto, capa-de-embedding
en: tokenization

## tokenizacion-por-subpalabras
frase: Partir las palabras en fragmentos frecuentes: las comunes quedan enteras y las raras se componen de trozos.
desc: BPE parte de los caracteres y fusiona repetidamente el par de símbolos más frecuente hasta alcanzar el tamaño de vocabulario deseado. WordPiece, usado por BERT, elige las fusiones que más aumentan la verosimilitud. SentencePiece trata el texto como una secuencia bruta, espacios incluidos, lo que lo hace independiente del idioma. Con vocabularios de 30.000 a 200.000 tokens se representa cualquier texto sin palabras desconocidas.
ej: Una palabra inventada como "superchulísimo" se divide en "super", "chul" e "ísimo" en lugar de quedar como desconocida.
hp: vocab_size: tamaño del vocabulario final | min_frequency: frecuencia mínima de un par para fusionarlo
py: tokenizers.models.BPE ; tokenizers.trainers.BpeTrainer ; sentencepiece.SentencePieceProcessor
rel: tokenizacion, vocabulario-y-tokens-especiales, llm
en: subword tokenization (BPE, WordPiece, SentencePiece)

## vocabulario-y-tokens-especiales
frase: La lista fija de tokens que conoce el modelo, más unos tokens reservados con funciones especiales.
desc: El vocabulario asigna un número a cada token. Los tokens especiales marcan la estructura: [CLS] y [SEP] en BERT para clasificar y separar frases, [MASK] para el preentrenamiento enmascarado, [PAD] para igualar longitudes dentro de un lote, y marcadores de inicio y fin de texto o de turnos de conversación en los LLM. Añadir tokens nuevos exige ampliar la capa de embedding.
ej: Una conversación con un LLM se codifica intercalando tokens especiales que indican dónde empieza el mensaje del usuario y dónde la respuesta del asistente.
py: `tokenizer.add_special_tokens({"pad_token": "[PAD]"})`
rel: tokenizacion, token-cls, plantillas-de-chat, capa-de-embedding
en: vocabulary & special tokens

## word-embeddings
frase: Un vector denso por palabra, aprendido de forma que las palabras de significado parecido queden cerca.
desc: Se basan en la hipótesis distribucional: una palabra se caracteriza por las palabras que la rodean. Métodos como word2vec, GloVe o fastText aprenden de grandes corpus un vector de entre 100 y 300 dimensiones por palabra que captura relaciones semánticas y sintácticas. Supusieron un salto respecto a la bolsa de palabras, aunque cada palabra tiene un único vector, sea cual sea su contexto.
cuando: Para representar palabras en modelos ligeros o analizar relaciones semánticas en un corpus.
ej: Los vecinos más cercanos de "Madrid" en un espacio de word2vec son "Barcelona", "Sevilla" y "Valencia".
py: gensim.models.Word2Vec ; gensim.models.FastText
rel: word2vec, glove-y-swivel, embeddings-estaticos-vs-contextuales, aritmetica-de-embeddings
en: word embeddings

## word2vec
frase: Aprende vectores de palabras entrenando una red sencilla a predecir cada palabra a partir de su contexto, o al revés.
desc: Tiene dos variantes. CBOW predice una palabra a partir de las que la rodean; Skip-gram predice las palabras del contexto a partir de la central y funciona mejor con palabras raras. La red tiene una sola capa oculta lineal, cuyos pesos son los embeddings. Gracias a trucos como el negative sampling, se entrena con miles de millones de palabras en horas. Sus vectores reflejan analogías como rey − hombre + mujer ≈ reina.
cuando: Para obtener embeddings de palabras de un dominio específico (médico, legal, de productos) a partir de un corpus propio.
ej: Entrenar word2vec con las descripciones de productos de una tienda para que "zapatillas" y "deportivas" queden juntas.
hp: vector_size: dimensión de los vectores | window: palabras de contexto a cada lado | sg: 1 para Skip-gram, 0 para CBOW | min_count: frecuencia mínima para incluir una palabra
py: gensim.models.Word2Vec
rel: word-embeddings, negative-sampling, aritmetica-de-embeddings, glove-y-swivel
en: Word2Vec (Skip-gram & CBOW)

## negative-sampling
frase: Entrenar comparando cada palabra del contexto real con unas pocas palabras al azar, en lugar de con todo el vocabulario.
desc: Calcular una softmax sobre cientos de miles de palabras en cada paso es carísimo. El negative sampling convierte la tarea en una clasificación binaria: distinguir el par real (palabra, contexto) de unos pocos pares falsos con palabras muestreadas al azar, en proporción a su frecuencia elevada a 0,75. Reduce el coste de cada paso de todo el vocabulario a unas pocas palabras, y la misma idea reaparece en el aprendizaje contrastivo.
ej: Para el par real ("café", "taza"), se generan 5 pares falsos, como ("café", "tractor"), que el modelo debe rechazar.
hp: negative: número de palabras negativas por ejemplo (entre 5 y 20) | ns_exponent: exponente de la distribución de muestreo (0,75)
py: gensim.models.Word2Vec
rel: word2vec, aprendizaje-contrastivo, softmax, perdida-infonce
en: negative sampling

## glove-y-swivel
frase: Embeddings de palabras obtenidos factorizando la matriz de coocurrencias de todo el corpus.
desc: En lugar de recorrer ventanas como word2vec, GloVe construye una matriz global de cuántas veces aparece cada palabra cerca de cada otra y aprende vectores cuyo producto escalar aproxime el logaritmo de esas coocurrencias. Combina las estadísticas globales de los métodos de factorización con la eficacia de los métodos predictivos. Swivel, de Google, sigue la misma idea y escala mejor a matrices enormes, tratando con cuidado los pares que nunca aparecen juntos.
cuando: Para usar vectores de palabras preentrenados y de propósito general sin entrenar nada.
ej: Cargar vectores GloVe preentrenados de 100 dimensiones como capa de embedding inicial de un clasificador de textos.
py: gensim.downloader.load
rel: word2vec, word-embeddings, factorizacion-de-matrices, embeddings-estaticos-vs-contextuales
en: GloVe & Swivel

## embeddings-estaticos-vs-contextuales
frase: Un vector fijo por palabra frente a un vector que cambia según la frase en la que aparece.
desc: Los embeddings estáticos (word2vec, GloVe) dan a "banco" el mismo vector hable de dinero o de un asiento, mezclando sus significados. Los contextuales, producidos por modelos como ELMo, BERT o cualquier LLM, calculan el vector de cada token a partir de toda la frase, así que distinguen los sentidos. Son mucho más potentes, a cambio de necesitar un modelo grande para calcularlos.
ej: En "me senté en el banco del parque" y "pedí un préstamo al banco", un modelo contextual da a "banco" vectores muy distintos.
rel: word-embeddings, bert, modelos-de-embeddings-de-texto, autoatencion
en: static vs contextual embeddings

## modelos-de-embeddings-de-texto
frase: Modelos entrenados para convertir frases o documentos enteros en un vector que capta su significado.
desc: Parten de un transformer y se ajustan con aprendizaje contrastivo sobre pares de textos relacionados (pregunta y respuesta, título y resumen, paráfrasis), de modo que los textos con significado parecido den vectores cercanos. Producen un vector por texto, que se compara con la similitud coseno. Son la pieza de recuperación de los sistemas RAG y de los buscadores semánticos.
cuando: Para búsqueda semántica, RAG, agrupar documentos o detectar duplicados por su significado.
ej: "¿Cómo cambio mi contraseña?" y "He olvidado la clave de acceso" obtienen vectores muy cercanos aunque no compartan palabras.
py: sentence_transformers.SentenceTransformer ; `modelo.encode(frases, normalize_embeddings=True)`
rel: embeddings, recuperacion-densa, aprendizaje-contrastivo, busqueda-por-similitud, token-cls
en: text embedding models

## modelos-de-lenguaje
frase: Modelos que asignan probabilidades al texto y, por tanto, pueden predecirlo y generarlo.
desc: Un modelo de lenguaje aprende la distribución de probabilidad de las secuencias de texto, normalmente prediciendo el siguiente token. De los modelos de n-gramas se pasó a las redes recurrentes y después a los transformers, entrenados con cantidades crecientes de texto. Según su objetivo de preentrenamiento y su arquitectura, se especializan en entender (BERT), generar (GPT) o transformar textos (T5).
ej: El autocompletado del teclado del móvil es un modelo de lenguaje pequeño; un LLM es uno enorme.
rel: modelo-de-lenguaje, objetivos-de-preentrenamiento, llm, tipos-de-transformer
en: language models

## modelo-de-lenguaje
frase: Un modelo que, dado un texto, predice la probabilidad de cada posible token siguiente.
desc: Entrenar a un modelo para predecir la siguiente palabra parece una tarea modesta, pero para hacerlo bien con textos variados debe aprender gramática, hechos sobre el mundo, estilo e incluso cierto razonamiento. Generar texto consiste en predecir el siguiente token, elegir uno y repetir. Su calidad se mide con la perplejidad sobre textos no vistos.
ej: Tras "La capital de Francia es", un buen modelo asigna una probabilidad muy alta a "París".
py: transformers.AutoModelForCausalLM
rel: modelos-autorregresivos, clm, perplejidad, decodificacion-y-temperatura
en: language model (next-token prediction)

## objetivos-de-preentrenamiento
frase: La tarea autosupervisada con la que se preentrena un modelo de lenguaje: adivinar palabras ocultas, la siguiente o tramos borrados.
desc: El objetivo determina qué aprende el modelo y para qué sirve después. El modelado enmascarado (MLM) da representaciones bidireccionales ideales para comprender; el causal (CLM) da modelos que generan texto de forma natural; la corrupción de spans, intermedia, encaja con las arquitecturas codificador-decodificador. Todos se entrenan sin etiquetas, con texto bruto.
ej: Con el mismo texto, BERT aprende a rellenar huecos, GPT a continuar frases y T5 a reconstruir tramos borrados.
rel: mlm, clm, corrupcion-de-spans, aprendizaje-autosupervisado
en: pretraining objectives

## mlm
frase: Ocultar al azar una parte de los tokens y entrenar al modelo para adivinarlos a partir del resto.
desc: El modelado de lenguaje enmascarado, introducido con BERT, sustituye alrededor del 15 % de los tokens por [MASK] (o por otro token, o los deja igual) y entrena al modelo para predecir los originales. Como puede mirar a ambos lados del hueco, aprende representaciones bidireccionales muy ricas, ideales para clasificar y extraer información. No sirve para generar texto de forma natural.
ej: En "el [MASK] ladra por las noches", el modelo debe predecir "perro" usando el contexto anterior y el posterior.
hp: mlm_probability: fracción de tokens enmascarados (0,15 por defecto)
py: transformers.DataCollatorForLanguageModeling ; transformers.AutoModelForMaskedLM
rel: bert, modelado-enmascarado, objetivos-de-preentrenamiento, clm
en: masked language modeling (MLM)

## clm
frase: Predecir cada token a partir solo de los anteriores: el objetivo con el que se entrenan los GPT.
desc: El modelado de lenguaje causal entrena al modelo para predecir el siguiente token en cada posición, con una máscara causal que le impide ver el futuro. Cada posición del texto es un ejemplo de entrenamiento, lo que lo hace muy eficiente. Es el objetivo natural para generar texto y el que, a gran escala, ha producido los LLM actuales.
ej: Con la frase "hoy hace mucho calor", el modelo aprende a la vez a predecir "hace" tras "hoy", "mucho" tras "hoy hace" y "calor" tras "hoy hace mucho".
py: transformers.AutoModelForCausalLM ; transformers.DataCollatorForLanguageModeling
rel: gpt, mascara-causal, modelos-autorregresivos, mlm
en: causal language modeling (CLM)

## corrupcion-de-spans
frase: Borrar tramos contiguos del texto y entrenar al modelo para generarlos: el objetivo de T5.
desc: Se eliminan tramos de varios tokens, cada uno sustituido por un único marcador centinela, y el decodificador debe generar los tramos que faltan, precedidos de su marcador. Combina la comprensión bidireccional del codificador con la capacidad de generar del decodificador, y es más eficiente que reconstruir el texto entero.
ej: "Gracias por [X] a mi fiesta [Y] semana" → el modelo genera "[X] invitarme [Y] la pasada".
py: transformers.T5ForConditionalGeneration
rel: t5, mlm, objetivos-de-preentrenamiento, tipos-de-transformer
en: span corruption

## bert
frase: Transformer codificador preentrenado con texto enmascarado; el referente para entender y clasificar texto.
desc: BERT (2018) apila bloques transformer codificadores con atención bidireccional y se preentrena con modelado enmascarado y, en su versión original, prediciendo si una frase sigue a otra. Después se ajusta con pocos datos para clasificar, extraer entidades o responder preguntas, y en su momento batió el estado del arte en casi todos los benchmarks. Variantes como RoBERTa, DistilBERT o los modelos entrenados en español siguen siendo opciones eficientes para tareas de comprensión.
cuando: Para clasificar textos, extraer entidades o calcular embeddings con un modelo pequeño, rápido y barato de ajustar.
ej: Ajustar un modelo tipo BERT en español para detectar qué correos de clientes son quejas urgentes.
py: transformers.BertModel ; transformers.AutoModelForSequenceClassification
rel: mlm, tipos-de-transformer, token-cls, fine-tuning, benchmarks-de-nlp
en: BERT (encoder-only)

## gpt
frase: Transformer decodificador entrenado para predecir el siguiente token: la familia de la que nacen los LLM generativos.
desc: GPT apila bloques transformer decodificadores con máscara causal y se preentrena prediciendo el siguiente token en enormes cantidades de texto. GPT-2 (2019) mostró que un modelo así genera textos coherentes y resuelve tareas sin ajuste específico, y GPT-3 (2020) demostró el aprendizaje a partir de ejemplos incluidos en la instrucción. Casi todos los LLM actuales siguen esta arquitectura.
cuando: Para generar texto, completar código o, una vez ajustado con instrucciones, actuar como asistente.
ej: Dado el inicio de un correo, un modelo tipo GPT lo continúa con el tono y el tema adecuados.
py: transformers.GPT2LMHeadModel ; transformers.AutoModelForCausalLM
rel: clm, mascara-causal, llm, in-context-learning, tipos-de-transformer
en: GPT (decoder-only)

## t5
frase: Transformer codificador-decodificador que trata cualquier tarea como transformar un texto en otro.
desc: T5 (2019) unificó las tareas de NLP en un formato de texto a texto: la entrada incluye la instrucción, como "traduce al alemán:" o "resume:", y la salida siempre es texto. Se preentrena con corrupción de spans y se ajusta en muchas tareas a la vez. Sus variantes, como mT5 (multilingüe) y Flan-T5 (ajustada con instrucciones), siguen siendo útiles para traducir y resumir con modelos medianos.
cuando: Para traducir, resumir o reformular textos con un modelo mediano y controlable.
ej: "resume: [texto de una noticia]" → el modelo devuelve un resumen de dos frases.
py: transformers.T5ForConditionalGeneration
rel: corrupcion-de-spans, tipos-de-transformer, fine-tuning-multitarea, instruction-tuning
en: T5 (encoder-decoder)

## llm
frase: Un modelo de lenguaje con miles de millones de parámetros, entrenado con una parte enorme del texto disponible.
desc: Un gran modelo de lenguaje es un transformer, casi siempre decodificador, preentrenado con billones de tokens y después ajustado para seguir instrucciones y ser útil y seguro. A gran escala muestra capacidades que no tiene a pequeña escala: seguir instrucciones, aprender de ejemplos incluidos en la petición, razonar paso a paso, programar o usar herramientas. Es la base de los asistentes conversacionales y de los agentes.
cuando: Para tareas abiertas de lenguaje: redactar, resumir, extraer, clasificar sin datos de entrenamiento, programar o razonar sobre documentos.
ej: Un LLM convierte el correo desordenado de un cliente en una ficha estructurada con el producto, el problema y la urgencia.
rel: gpt, preentrenamiento-de-llms, adaptacion-de-llms, leyes-de-escalado, agentes-de-ia
en: large language model (LLM)

## preentrenamiento-de-llms
frase: La fase cara y masiva en la que el LLM aprende de billones de tokens a predecir el siguiente.
desc: Se recopilan y filtran enormes cantidades de texto y código (web, libros, artículos, repositorios), se eliminan duplicados y contenidos de baja calidad, y se entrena el modelo durante semanas o meses en miles de GPU. La calidad y la mezcla de los datos importan tanto como el tamaño. El resultado es un modelo base que completa textos, pero que aún no sigue instrucciones ni conversa: eso llega con el ajuste posterior.
ej: Ante "¿Cuál es la capital de Italia?", un modelo base puede continuar con otra pregunta de examen en lugar de responder.
rel: llm, leyes-de-escalado, clm, sft, gpu-y-tpu
en: LLM pretraining

## ventana-de-contexto
frase: El número máximo de tokens que el modelo puede considerar a la vez, entre la entrada y la respuesta.
desc: Todo lo que el modelo "ve" en una petición (instrucciones, historial, documentos y su propia respuesta) debe caber en la ventana de contexto, que va de unos miles a millones de tokens según el modelo. Lo que queda fuera no existe para él. Los contextos largos son caros, porque el coste de la atención crece con la longitud, y los modelos pueden aprovechar peor la información situada en mitad de un contexto muy largo.
ej: En una ventana de 128.000 tokens cabe un libro de unas 300 páginas; para consultar una biblioteca entera hace falta RAG.
rel: atencion-por-producto-escalar-escalado, rag, codificacion-posicional, compresion-y-reordenacion-del-contexto, memoria-del-agente
en: context window

## modelos-de-razonamiento
frase: LLM entrenados para pensar paso a paso antes de responder, dedicando más cómputo a los problemas difíciles.
desc: En lugar de contestar directamente, generan primero un razonamiento interno largo en el que plantean, prueban y corrigen enfoques, y solo después dan la respuesta. Se entrenan con aprendizaje por refuerzo sobre problemas con respuesta verificable, como matemáticas o programación, lo que les enseña a razonar mejor. Rinden mucho más en tareas complejas a cambio de más tiempo y coste por respuesta: mejoran cuanto más "piensan" (escalado en inferencia).
cuando: Para problemas de matemáticas, programación, planificación o análisis con varios pasos, cuando la precisión importa más que la latencia.
ej: Ante un problema de olimpiada matemática, el modelo escribe varias páginas de razonamiento, descarta dos caminos fallidos y llega a la solución.
rel: chain-of-thought, llm, rlhf, agentes-de-ia
en: reasoning models

## adaptacion-de-llms
frase: Cómo se convierte un modelo base en un asistente útil, o se especializa en una tarea o un dominio.
desc: Tras el preentrenamiento, el modelo se adapta en varias etapas: ajuste supervisado con ejemplos de instrucciones y respuestas, alineamiento con preferencias humanas mediante RLHF o DPO y, si hace falta, especialización en un dominio. Como ajustar todos los parámetros es caro, se usan métodos eficientes (PEFT) como LoRA. Antes de ajustar, conviene comprobar si bastan un buen prompt o RAG.
ej: Un modelo abierto se ajusta con 10.000 conversaciones de soporte técnico para responder con el tono y los procedimientos de la empresa.
rel: sft, peft, rlhf, dpo, prompt-engineering
en: LLM adaptation

## sft
frase: Ajustar el LLM con ejemplos de instrucciones y de las respuestas ideales, escritas o revisadas por personas.
desc: El fine-tuning supervisado entrena el modelo, con la misma pérdida de predicción del siguiente token, sobre pares de instrucción y respuesta de alta calidad, normalmente calculando la pérdida solo sobre la respuesta. Es el paso que convierte un modelo que completa textos en uno que sigue instrucciones y conversa. Pocos miles de ejemplos muy cuidados pueden bastar: la calidad importa más que la cantidad.
cuando: Para enseñar a un modelo un formato, un estilo o una tarea concreta a partir de ejemplos de cómo debe responder.
ej: Ajustar un modelo con 2.000 informes radiológicos y sus resúmenes para que redacte resúmenes con el formato del hospital.
hp: learning_rate: tamaño de paso, típicamente entre 10⁻⁵ y 2·10⁻⁴ (más alto con LoRA) | num_train_epochs: pasadas por los datos, normalmente de 1 a 3 | max_length: longitud máxima de cada ejemplo, en tokens
py: trl.SFTTrainer ; trl.SFTConfig
rel: instruction-tuning, fine-tuning, lora, rlhf, plantillas-de-chat
en: supervised fine-tuning (SFT)

## instruction-tuning
frase: SFT con miles de tareas distintas formuladas como instrucciones, para que el modelo aprenda a seguir instrucciones nuevas.
desc: En lugar de entrenar una tarea concreta, se ajusta el modelo con una colección muy variada de tareas expresadas en lenguaje natural ("resume…", "traduce…", "clasifica…"). Así aprende la habilidad general de seguir instrucciones y generaliza a tareas que nunca vio. Flan-T5 y los primeros asistentes demostraron que este paso mejora drásticamente el rendimiento en tareas nuevas sin ejemplos.
cuando: Para convertir un modelo base en un asistente generalista, más que para enseñarle una tarea concreta.
ej: Un modelo ajustado con instrucciones de unas 1.800 tareas responde bien a "escribe un haiku sobre la lluvia" sin haber visto nunca esa tarea.
py: trl.SFTTrainer
rel: sft, fine-tuning-multitarea, t5, in-context-learning
en: instruction tuning

## fine-tuning-multitarea
frase: Entrenar un mismo modelo en varias tareas a la vez para que compartan lo aprendido.
desc: En lugar de un modelo por tarea, se mezclan los datos de varias tareas y se entrena un único modelo, a menudo indicando la tarea en la entrada. Las tareas relacionadas se ayudan entre sí, sobre todo las que tienen pocos datos, y el modelo resultante es más general. Hay que equilibrar la mezcla para que las tareas grandes no ahoguen a las pequeñas.
cuando: Cuando hay varias tareas relacionadas, algunas con pocos datos, y se quiere un único modelo que las resuelva todas.
ej: Un solo modelo que, según la instrucción, clasifica la intención de un cliente, extrae su número de pedido o resume la conversación.
rel: instruction-tuning, t5, multi-entrada-salida, transfer-learning
en: multitask fine-tuning

## peft
frase: Ajustar un modelo grande entrenando solo una pequeñísima fracción de sus parámetros.
desc: El fine-tuning eficiente en parámetros congela el modelo base y entrena solo unos pocos parámetros añadidos o seleccionados, a menudo menos del 1 %. Reduce drásticamente la memoria y el coste, permite guardar cada adaptación en unos pocos megabytes y limita el olvido catastrófico. Las técnicas principales son LoRA, los adapters y el prompt tuning, y suelen rendir casi como el ajuste completo.
cuando: Para ajustar LLM u otros modelos grandes con hardware limitado, o para mantener muchas versiones especializadas de un mismo modelo.
ej: Una empresa mantiene un único modelo base y 20 adaptaciones LoRA de pocos megabytes, una por departamento, que carga según la petición.
py: peft.get_peft_model ; peft.LoraConfig
rel: lora, qlora, prompt-tuning, adapters, olvido-catastrofico
en: parameter-efficient fine-tuning (PEFT)

## lora
frase: Aprender la corrección de cada matriz de pesos como el producto de dos matrices estrechas.
desc: LoRA (Low-Rank Adaptation) congela los pesos originales W y aprende una corrección de rango bajo, ΔW = B·A, con A y B mucho más pequeñas: con rango 8, una matriz de 4.096 × 4.096 se ajusta con unos 65.000 parámetros en lugar de 16,7 millones. Al terminar, la corrección se puede sumar a W, así que la inferencia no se ralentiza. Es la técnica de ajuste eficiente más usada en LLM y en modelos de difusión.
cuando: Como opción por defecto para ajustar LLM o modelos de difusión con recursos limitados.
ej: Ajustar un LLM de 8.000 millones de parámetros en una sola GPU entrenando menos del 1 % de los parámetros.
hp: r: rango de la corrección; más alto, más capacidad | lora_alpha: factor de escala de la corrección | lora_dropout: dropout aplicado en las capas LoRA | target_modules: capas a las que se aplica (por ejemplo, las proyecciones de la atención)
py: peft.LoraConfig ; peft.get_peft_model
rel: peft, qlora, fine-tuning, dreambooth
en: LoRA (low-rank adaptation)

## qlora
frase: LoRA sobre un modelo base cuantizado a 4 bits: ajustar modelos grandes en una sola GPU.
desc: QLoRA carga el modelo base en 4 bits con un formato pensado para pesos con distribución normal (NF4), lo que reduce su memoria unas cuatro veces respecto a 16 bits, y entrena adaptadores LoRA en 16 bits encima. Añade la doble cuantización de las constantes de escala y la paginación de memoria para absorber picos. Logra resultados casi iguales al ajuste en 16 bits con una fracción del hardware.
cuando: Para ajustar LLM grandes cuando la memoria de la GPU es el factor limitante.
ej: Ajustar un modelo de 65.000 millones de parámetros en una única GPU de 48 GB, algo imposible sin cuantizar.
hp: load_in_4bit: carga los pesos del modelo base en 4 bits | bnb_4bit_quant_type: formato de cuantización ("nf4") | bnb_4bit_compute_dtype: precisión de los cálculos (por ejemplo, bfloat16)
py: transformers.BitsAndBytesConfig ; peft.prepare_model_for_kbit_training
rel: lora, cuantizacion, peft, gpu-y-tpu
en: QLoRA

## prompt-tuning
frase: Aprender unos pocos vectores de "instrucción" que se anteponen a la entrada, sin tocar el modelo.
desc: En lugar de escribir un prompt con palabras, se aprenden directamente unos pocos embeddings virtuales que se colocan al principio de la entrada, con el modelo congelado. Solo se entrenan esos vectores, unos miles de parámetros por tarea. Funciona mejor cuanto más grande es el modelo, y variantes como el prefix tuning añaden vectores entrenables en todas las capas.
cuando: Para adaptar un modelo grande a muchas tareas con un coste mínimo por tarea.
ej: Con 20 tokens virtuales entrenados, un modelo congelado aprende a clasificar las consultas de los clientes en sus categorías.
hp: num_virtual_tokens: número de vectores que se anteponen a la entrada
py: peft.PromptTuningConfig
rel: peft, prompt-engineering, lora, adapters
en: prompt tuning

## adapters
frase: Pequeñas capas que se insertan dentro del modelo congelado y son lo único que se entrena.
desc: Un adapter es un pequeño módulo con forma de cuello de botella (proyección hacia abajo, no linealidad y proyección hacia arriba, con conexión residual) que se inserta tras las subcapas de cada bloque transformer. Solo se entrenan los adapters, en torno al 1-4 % de los parámetros, y se pueden intercambiar por tarea. Añaden algo de latencia en inferencia, a diferencia de LoRA, cuya corrección se puede fusionar con los pesos.
cuando: Para mantener muchas especializaciones intercambiables de un mismo modelo o combinar conocimientos de varias tareas.
ej: Un modelo multilingüe con un adapter por idioma que se activa según la lengua del texto.
py: adapters.SeqBnConfig
rel: peft, lora, conexion-residual, cuello-de-botella
en: adapters

## rlhf
frase: Alinear el LLM con lo que prefieren las personas mediante un modelo de recompensa y aprendizaje por refuerzo.
desc: Primero se recogen comparaciones humanas: dadas dos respuestas a la misma petición, cuál es mejor. Con ellas se entrena un modelo de recompensa que puntúa respuestas. Después se optimiza el LLM con aprendizaje por refuerzo (tradicionalmente PPO; hoy también variantes como GRPO) para maximizar esa recompensa, con una penalización KL que le impide alejarse demasiado del modelo original. Así se vuelve más útil, honesto e inofensivo de lo que logra el SFT por sí solo.
cuando: Para alinear un modelo con preferencias difíciles de especificar con ejemplos, como la utilidad, el tono o la seguridad.
ej: Tras el RLHF, ante una pregunta ambigua el asistente pide una aclaración en lugar de inventarse una respuesta.
py: trl.RewardTrainer ; trl.GRPOTrainer
rel: modelo-de-recompensa, ppo, dpo, reward-hacking, rlaif
en: RLHF (reinforcement learning from human feedback)

## modelo-de-recompensa
frase: Un modelo que puntúa lo buena que es una respuesta, entrenado con preferencias humanas.
desc: Suele ser un LLM con una cabeza que devuelve un número en lugar de texto. Se entrena con pares de respuestas en los que una persona eligió la mejor, de modo que la preferida obtenga más puntuación (modelo de Bradley-Terry). Sustituye a las personas durante el aprendizaje por refuerzo, que necesita millones de evaluaciones. Sus errores y sesgos se transmiten al modelo final, que puede aprender a explotarlos.
ej: Entre dos explicaciones de un mismo concepto, el modelo de recompensa da 2,1 a la clara y concisa y −0,8 a la que divaga.
py: trl.RewardTrainer ; trl.RewardConfig
rel: rlhf, reward-hacking, ppo, llm-as-a-judge
en: reward model

## ppo
frase: Algoritmo de aprendizaje por refuerzo que mejora la política con pasos pequeños y recortados para no desestabilizarla.
desc: Proximal Policy Optimization hace más probables las acciones con ventaja positiva, pero recorta el cociente entre la política nueva y la anterior para limitar cuánto cambia en cada paso. Es estable y relativamente sencillo, por lo que se convirtió en el estándar del aprendizaje por refuerzo profundo y del RLHF. En los LLM exige mantener varios modelos a la vez (política, referencia, recompensa y valor), lo que impulsó alternativas más ligeras, como DPO o GRPO.
cuando: Para entrenar políticas con aprendizaje por refuerzo en entornos simulados o, en RLHF, para optimizar un LLM frente a un modelo de recompensa.
ej: Un robot simulado aprende a caminar con PPO tras millones de intentos en los que cada mejora se aplica con prudencia.
hp: learning_rate: tamaño de paso | clip_range: límite al cambio de la política en cada actualización (0,2 es típico) | n_steps: pasos de interacción entre actualizaciones | gamma: factor de descuento de las recompensas futuras
py: stable_baselines3.PPO ; trl.GRPOTrainer
rel: rlhf, politica-y-funcion-de-valor, dpo, modelo-de-recompensa
en: PPO (proximal policy optimization)

## reward-hacking
frase: El modelo aprende a maximizar la recompensa explotando sus fallos, en lugar de hacer lo que se pretendía.
desc: Cualquier recompensa es una aproximación imperfecta de lo que se quiere, y un optimizador potente encuentra atajos: respuestas más largas porque el modelo de recompensa las prefiere, un tono adulador o código que pasa los tests modificándolos en lugar de resolver el problema. Se mitiga con penalizaciones KL, recompensas mejores y más diversas, y vigilando el comportamiento, no solo la puntuación.
ej: Un modelo optimizado para que los usuarios valoren bien sus respuestas aprende a darles siempre la razón.
rel: rlhf, modelo-de-recompensa, alucinaciones, riesgos-y-permisos
en: reward hacking

## rlaif
frase: Como el RLHF, pero las preferencias las da otro modelo de IA siguiendo unos principios, en lugar de personas.
desc: Recoger preferencias humanas es caro y lento. En el RLAIF, un modelo evalúa o compara respuestas guiado por una lista de principios escritos (una "constitución"), y esas valoraciones entrenan el modelo de recompensa o se usan directamente. Escala mucho mejor y hace explícitos los criterios, aunque hereda las limitaciones del modelo evaluador. Es la base de la IA constitucional.
cuando: Para alinear modelos a gran escala o en dominios en los que las preferencias humanas son caras de obtener.
ej: Un modelo juez compara dos respuestas según el principio "elige la respuesta más útil que no ayude a causar daño", y su veredicto se usa como señal de entrenamiento.
rel: rlhf, llm-as-a-judge, modelo-de-recompensa, dpo
en: RLAIF (reinforcement learning from AI feedback)

## dpo
frase: Alinear el LLM directamente con pares de respuestas preferidas y rechazadas, sin modelo de recompensa ni RL.
desc: Direct Preference Optimization demuestra que el objetivo del RLHF se puede optimizar con una simple pérdida de clasificación sobre pares de preferencias: aumentar la probabilidad relativa de la respuesta preferida frente a la rechazada, respecto a un modelo de referencia. Elimina el modelo de recompensa y el bucle de aprendizaje por refuerzo, así que es mucho más sencillo y estable, y se usa ampliamente para alinear modelos abiertos.
cuando: Para alinear un modelo con preferencias cuando se dispone de pares de respuestas buena y mala.
ej: Con 10.000 pares en los que se prefiere la respuesta concisa a la verbosa, el modelo aprende a ser más directo.
hp: beta: intensidad de la restricción respecto al modelo de referencia; más baja, más se aleja de él | learning_rate: tamaño de paso, normalmente muy pequeño (del orden de 10⁻⁶)
py: trl.DPOTrainer ; trl.DPOConfig
rel: rlhf, sft, modelo-de-recompensa, divergencia-de-kullback-leibler
en: DPO (direct preference optimization)

## inferencia-y-eficiencia
frase: Cómo se generan las respuestas y cómo se hace que los LLM sean más rápidos y baratos de ejecutar.
desc: La inferencia de un LLM genera la respuesta token a token, y cada token exige recorrer el modelo entero. Su coste depende del tamaño del modelo, de la longitud del contexto y de la estrategia de decodificación. Para abaratarla se cuantizan los pesos, se reutilizan cálculos de la atención (caché KV), se agrupan peticiones y se usan modelos más pequeños o destilados, o se elige entre ejecutar el modelo en local o pagar una API.
ej: Cuantizar un modelo de 8.000 millones de parámetros a 4 bits permite ejecutarlo en un portátil con 8 GB de memoria.
rel: decodificacion-y-temperatura, cuantizacion, llms-locales-vs-api, destilacion-de-conocimiento
en: inference & efficiency

## decodificacion-y-temperatura
frase: Cómo se elige cada token a partir de las probabilidades del modelo: el más probable, con temperatura o muestreando entre los mejores.
desc: La decodificación voraz elige siempre el token más probable: es determinista, pero repetitiva. El muestreo elige al azar según las probabilidades, y la temperatura las reajusta: baja (menos de 1) concentra la elección en los tokens más probables; alta (más de 1) la reparte y aumenta la creatividad y los errores. Top-k limita el muestreo a los k tokens más probables, y top-p (nucleus), al menor conjunto que acumula una probabilidad p. La búsqueda en haz explora varias secuencias en paralelo.
cuando: Temperatura baja para extraer datos, clasificar o programar; más alta para escritura creativa o lluvia de ideas.
ej: Con temperatura 0, el modelo da siempre la misma respuesta; con 1,2, cada vez propone un eslogan distinto para la misma campaña.
hp: temperature: aleatoriedad del muestreo | top_p: masa de probabilidad acumulada en la que se muestrea | top_k: número máximo de candidatos | max_new_tokens: longitud máxima de la respuesta
py: transformers.GenerationConfig
rel: softmax, modelo-de-lenguaje, alucinaciones, generacion-con-rnn
en: decoding & temperature

## cuantizacion
frase: Representar los pesos del modelo con menos bits para que ocupe menos memoria y vaya más rápido.
desc: Los pesos se entrenan en 16 o 32 bits, pero pueden guardarse en 8, 4 o incluso menos bits con una pérdida de calidad pequeña. Reduce la memoria entre 2 y 4 veces y acelera la inferencia. GPTQ y AWQ cuantizan los pesos de forma calibrada con unos pocos datos para minimizar el error; GGUF es el formato de llama.cpp para ejecutar modelos cuantizados en CPU y portátiles. Por debajo de 4 bits, la calidad cae más deprisa.
cuando: Para ejecutar LLM en hardware limitado, abaratar la inferencia o ajustar modelos con QLoRA.
ej: Un modelo de 70.000 millones de parámetros ocupa 140 GB en 16 bits y unos 40 GB en 4 bits, lo que le permite caber en una sola GPU grande.
hp: bits: número de bits por peso (habitualmente 4 u 8) | load_in_4bit: cuantización a 4 bits al cargar el modelo | group_size: tamaño de los grupos de pesos que comparten escala
py: transformers.BitsAndBytesConfig ; transformers.GPTQConfig ; transformers.AwqConfig
rel: qlora, llms-locales-vs-api, despliegue-e-inferencia, cuantizacion-vectorial
en: quantization (GPTQ, AWQ, GGUF)

## llms-locales-vs-api
frase: Ejecutar un modelo abierto en tu propia máquina o servidor frente a usar un modelo comercial a través de una API.
desc: Las API de los proveedores dan acceso a los modelos más capaces sin infraestructura, pagando por token, pero los datos salen de tu control y dependes de un tercero. Los modelos abiertos ejecutados en local o en tu nube dan privacidad, control, coste fijo y posibilidad de ajustarlos, a cambio de gestionar el hardware y de usar modelos algo menos capaces. Herramientas como Ollama, llama.cpp o vLLM facilitan servirlos, a menudo con una API compatible con la de OpenAI.
cuando: En local, con datos sensibles, mucho volumen o necesidad de ajustar el modelo; por API, para la máxima capacidad y para empezar rápido.
ej: Un hospital analiza informes clínicos con un modelo abierto en sus propios servidores para que ningún dato de pacientes salga del centro.
py: openai.OpenAI ; transformers.pipeline
rel: cuantizacion, hugging-face, despliegue-e-inferencia, decisiones-automatizadas-y-rgpd
en: local LLMs vs API

## uso-de-llms
frase: Las técnicas para obtener resultados fiables de un LLM sin reentrenarlo.
desc: Gran parte del rendimiento depende de cómo se pide la tarea. Las técnicas incluyen redactar instrucciones claras y con contexto, dar ejemplos, pedir que razone paso a paso, fijar el formato de salida y separar las instrucciones del sistema de las del usuario. También hay que conocer sus límites, como las alucinaciones, para diseñar sistemas que los compensen.
ej: Pedir "clasifica este correo" da resultados irregulares; especificar las categorías, dar tres ejemplos y exigir JSON los vuelve consistentes.
rel: prompt-engineering, in-context-learning, salida-estructurada, alucinaciones
en: using LLMs

## prompt-engineering
frase: Diseñar las instrucciones que se dan al LLM para obtener respuestas precisas y consistentes.
desc: Un buen prompt describe la tarea con claridad, aporta el contexto necesario (quién es el usuario, para qué sirve la respuesta), especifica el formato, da ejemplos y delimita los datos de entrada, por ejemplo con etiquetas. Pedir al modelo que adopte un rol, que razone antes de responder o que reconozca cuándo no sabe algo mejora la calidad. Se itera comparando resultados sobre un conjunto de casos de prueba.
cuando: Siempre, antes de plantearse ajustar un modelo: suele ser la forma más rápida y barata de mejorar los resultados.
ej: Pasar de "resume este contrato" a "resume este contrato de alquiler para el inquilino en 5 puntos, destacando plazos, penalizaciones y gastos" mejora mucho el resultado.
rel: in-context-learning, chain-of-thought, plantillas-de-chat, salida-estructurada
en: prompt engineering

## in-context-learning
frase: El LLM aprende una tarea a partir de las instrucciones o los ejemplos incluidos en la propia petición, sin cambiar sus pesos.
desc: En zero-shot, el modelo resuelve la tarea solo con la instrucción. En few-shot, se incluyen unos pocos ejemplos resueltos y el modelo imita el patrón. Este aprendizaje en contexto apareció en modelos grandes como GPT-3 y permite adaptar el modelo a tareas nuevas al instante. La elección, el orden y el formato de los ejemplos influyen en el resultado.
cuando: Para tareas nuevas o cambiantes sin datos de entrenamiento, o como prueba rápida antes de ajustar un modelo.
ej: Tres ejemplos de reseñas con su categoría ("envío", "calidad", "precio") bastan para que el modelo clasifique bien las reseñas nuevas.
rel: prompt-engineering, gpt, chain-of-thought, que-es-un-modelo-fundacional
en: zero-shot, few-shot & in-context learning

## chain-of-thought
frase: Pedir al modelo que razone paso a paso antes de dar la respuesta final.
desc: Al generar los pasos intermedios, el modelo dedica más cómputo al problema y cada paso condiciona el siguiente, lo que mejora mucho los resultados en aritmética, lógica y problemas de varios pasos. Basta con añadir "piensa paso a paso" o con dar ejemplos que incluyan el razonamiento. Los modelos de razonamiento llevan esta idea al entrenamiento. El razonamiento escrito no siempre refleja fielmente cómo llegó el modelo a la respuesta.
cuando: Para problemas de matemáticas, lógica, planificación o análisis que requieren varios pasos.
ej: "Un tren sale a las 9:40 y tarda 2 h 35 min; ¿a qué hora llega?". Razonando paso a paso, el modelo suma bien las horas y responde 12:15.
rel: modelos-de-razonamiento, prompt-engineering, in-context-learning, alucinaciones
en: chain-of-thought (CoT)

## plantillas-de-chat
frase: El formato con el que se codifica una conversación (sistema, usuario, asistente) para que el modelo la entienda.
desc: Los modelos conversacionales se ajustaron con conversaciones en un formato concreto, con tokens especiales que marcan quién habla. La plantilla de chat convierte una lista de mensajes en ese texto exacto, y usar otra plantilla degrada mucho las respuestas. El prompt de sistema, al inicio, fija el rol, las reglas y el contexto para toda la conversación, y suele tener prioridad sobre las instrucciones del usuario.
cuando: Siempre que se use un modelo conversacional directamente, sin una API que aplique la plantilla por ti.
ej: Un prompt de sistema define: "eres el asistente de una aseguradora; responde solo sobre pólizas y deriva las reclamaciones a una persona".
py: `tokenizer.apply_chat_template(mensajes, add_generation_prompt=True)`
rel: vocabulario-y-tokens-especiales, prompt-engineering, sft, guardrails-y-middleware
en: chat templates & system prompt

## salida-estructurada
frase: Obligar al LLM a responder en un formato estricto, como un JSON que cumpla un esquema.
desc: Para integrar un LLM en un programa, su salida debe poder procesarse automáticamente. Se puede pedir el formato en el prompt, pero es más fiable usar las funciones de salida estructurada de las API o la decodificación restringida, que solo permite generar tokens válidos según un esquema JSON. Definir el esquema con una clase de pydantic añade validación de tipos.
cuando: Siempre que la respuesta del modelo alimente a otro programa, una base de datos o un flujo automatizado.
ej: Extraer de cada factura escaneada un JSON con proveedor, fecha, importe e IVA que se inserta directamente en la contabilidad.
py: pydantic.BaseModel ; `llm.with_structured_output(Factura)`
rel: function-calling, prompt-engineering, agente-llm, alucinaciones
en: structured output

## alucinaciones
frase: Cuando el modelo genera información falsa o inventada con total seguridad.
desc: Un LLM produce el texto más plausible, no necesariamente el verdadero, así que puede inventar datos, citas, referencias o funciones de código inexistentes con la misma fluidez con la que acierta. Ocurre sobre todo con hechos poco frecuentes en sus datos, detalles precisos y preguntas fuera de su conocimiento. Se reduce con RAG, con instrucciones que le permitan decir "no lo sé", con verificación externa y citando fuentes.
cuando: Hay que tenerlas en cuenta en cualquier uso con consecuencias: datos, cifras o citas legales o médicas.
ej: En 2023, un abogado presentó ante un tribunal de Nueva York un escrito con sentencias citadas por un LLM que no existían.
rel: rag, llm-as-a-judge, decodificacion-y-temperatura, incertidumbre, reward-hacking
en: hallucinations

## evaluacion-de-llms
frase: Medir lo bien que funciona un LLM: conocimiento, razonamiento, seguridad y utilidad en tu tarea concreta.
desc: Se combinan benchmarks públicos (conocimiento, matemáticas, programación), evaluaciones con otro LLM como juez, comparaciones de preferencia humana y, sobre todo, conjuntos de prueba propios con los casos reales de la aplicación. Los benchmarks públicos se saturan y pueden estar contaminados, así que lo más fiable para decidir es evaluar con tus propios datos y criterios.
ej: Antes de elegir el modelo de un chatbot de soporte, se comparan tres candidatos con 300 preguntas reales de clientes y sus respuestas de referencia.
rel: benchmarks-y-saturacion, llm-as-a-judge, arenas-de-preferencia-humana, perplejidad
en: LLM evaluation

## benchmarks-y-saturacion
frase: Las pruebas estándar de los LLM se quedan pequeñas en pocos años: los modelos las superan o las han visto al entrenar.
desc: Benchmarks como MMLU (conocimiento), GSM8K (matemáticas) o HumanEval (programación) permitieron comparar modelos, pero los mejores alcanzan cifras cercanas al máximo y dejan de discriminar entre ellos (saturación). Además, sus preguntas pueden haberse filtrado en los datos de entrenamiento (contaminación), lo que infla los resultados. Se responde con pruebas más difíciles, privadas o que se renuevan con el tiempo.
ej: En pocos años, los mejores modelos pasaron de acertar menos de la mitad de las preguntas de MMLU a superar el 90 %, y hubo que crear versiones más difíciles.
rel: evaluacion-de-llms, benchmarks-de-nlp, fuga-de-datos, arenas-de-preferencia-humana
en: benchmarks & saturation

## llm-as-a-judge
frase: Usar un LLM para evaluar las respuestas de otro modelo según unos criterios.
desc: Un modelo potente recibe la pregunta, la respuesta y unas pautas de evaluación (una rúbrica o una respuesta de referencia) y devuelve una puntuación o elige la mejor de dos. Escala mucho mejor que la evaluación humana y se correlaciona bien con ella en muchas tareas, pero tiene sesgos conocidos: prefiere las respuestas largas, las de su misma familia de modelos y la primera opción presentada. Conviene validarlo con una muestra de juicios humanos.
cuando: Para evaluar a gran escala respuestas abiertas, como resúmenes, respuestas de chatbots o salidas de RAG.
ej: Un LLM juez puntúa de 1 a 5 la fidelidad de 2.000 resúmenes generados respecto a sus documentos originales.
rel: evaluacion-de-llms, rlaif, evaluacion-de-rag, arenas-de-preferencia-humana
en: LLM-as-a-judge

## arenas-de-preferencia-humana
frase: Plataformas donde las personas comparan a ciegas las respuestas de dos modelos y votan la mejor.
desc: Los usuarios envían sus propias preguntas, reciben dos respuestas anónimas y eligen la que prefieren. Con millones de votos se calcula una clasificación de tipo Elo (modelo de Bradley-Terry). Capturan la preferencia real de los usuarios en tareas variadas y son difíciles de contaminar, aunque premian el estilo (longitud, formato, tono) además de la corrección.
ej: Una arena pública de chatbots ordena decenas de modelos a partir de cientos de miles de comparaciones hechas por usuarios.
rel: evaluacion-de-llms, llm-as-a-judge, rlhf, evaluacion-cualitativa
en: human preference arenas
