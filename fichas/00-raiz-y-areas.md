# Fichas: raíz y áreas de primer nivel
# Formato por ficha:  "## id"  y líneas  desc / ej / rel (ids separados por comas) / py (separados por " ; ") / en (término en inglés)

## inteligencia-artificial
frase: Sistemas que perciben, razonan, aprenden o generan contenido; hoy casi siempre aprendiendo de datos.
desc: Campo que construye sistemas capaces de realizar tareas que asociamos con la inteligencia: percibir, razonar, aprender, predecir o generar contenido. Casi toda la IA actual se basa en aprendizaje automático, es decir, en modelos que aprenden patrones a partir de datos en lugar de seguir reglas escritas a mano.
ej: Un asistente que entiende tu pregunta, busca en documentos y redacta la respuesta combina varias ramas de este mapa: lenguaje, recuperación de información y agentes.
rel: ia-ml-dl, breve-historia-de-la-ia, cuatro-ingredientes
en: artificial intelligence

## fundamentos
frase: Las ideas comunes a todo modelo: aprender de los datos y generalizar a casos nuevos.
desc: Las ideas que comparten todos los modelos: aprender de los datos optimizando un objetivo y generalizar a casos que el modelo no ha visto. Aquí están también los paradigmas de aprendizaje y las nociones de distancia y similitud que reaparecen por todo el mapa.
ej: Un modelo que acierta el 99 % en entrenamiento y el 70 % con datos nuevos ilustra el problema central de la disciplina: el sobreajuste.
rel: generalizacion, cuatro-ingredientes, paradigmas-de-aprendizaje
en: foundations

## probabilidad-informacion-y-causalidad
frase: Las matemáticas para razonar con incertidumbre, medir información y separar causa de correlación.
desc: El lenguaje matemático de la incertidumbre. La probabilidad permite razonar con datos ruidosos. La teoría de la información mide cuánta información contiene una variable y cuánta comparte con otra. Los modelos gráficos y la causalidad ayudan a distinguir qué influye en qué, más allá de la simple correlación.
cuando: Cuando necesitas cuantificar la incertidumbre, medir cuánto dependen unas variables de otras o razonar sobre causas y efectos.
ej: Que dos variables estén correlacionadas no dice cuál causa la otra. Los métodos de descubrimiento causal intentan orientar esa flecha a partir de los datos.
rel: teorema-de-bayes, entropia, red-bayesiana, correlacion-no-implica-causalidad
en: probability, information theory & causality

## flujo-de-trabajo-en-ml
frase: Todo lo que rodea al modelo: datos, validación, métricas, ajuste y puesta en producción.
desc: Todo lo que rodea al modelo en un proyecto real: preparar los datos, separar entrenamiento y test, elegir métricas, ajustar hiperparámetros, desplegar y vigilar el modelo en producción. Los errores de esta parte, como la fuga de datos o una métrica mal elegida, suelen pesar más que la elección del algoritmo.
ej: Un detector de fraude con un 99,9 % de accuracy puede ser inútil si solo el 0,1 % de las operaciones son fraudes; aquí la métrica que importa es el recall o la curva precisión-recall.
rel: ciclo-de-vida-ml, fuga-de-datos, validacion-cruzada, mlops
en: ML workflow

## aprendizaje-supervisado
frase: Aprender a predecir una respuesta conocida a partir de ejemplos etiquetados.
desc: El modelo aprende, a partir de ejemplos etiquetados (X, y), una función que asigna una salida a cada entrada, y después la aplica para predecir casos nuevos. Si la salida es un número, el problema es de regresión; si es una categoría, de clasificación. Es el paradigma más usado en la industria.
cuando: Cuando tienes ejemplos históricos con la respuesta correcta y quieres predecirla en casos nuevos.
ej: Con miles de correos marcados como spam o no spam, entrenar un modelo que filtre los correos nuevos. Con viviendas ya vendidas, predecir el precio de una nueva.
py: `modelo.fit(X_train, y_train)` ; `modelo.predict(X_test)`
rel: regresion-vs-clasificacion, aprendizaje-no-supervisado, entrenamiento-validacion-y-test, funcion-de-perdida-vs-metrica
en: supervised learning

## aprendizaje-no-supervisado
frase: Descubrir estructura en datos sin etiquetas: grupos, dimensiones ocultas o densidades.
desc: Aprender la estructura de los datos sin etiquetas. Agrupa casos parecidos (clustering), comprime muchas variables en pocas (reducción de dimensionalidad) o estima cómo se distribuyen los datos (densidad). Las etiquetas son caras y los datos sin etiquetar abundan, así que resulta clave para explorar y para preparar otros modelos.
cuando: Cuando no tienes etiquetas y quieres explorar los datos, agruparlos, resumirlos o prepararlos para otro modelo.
ej: Segmentar a los clientes de una tienda online en grupos con hábitos de compra parecidos sin decirle al algoritmo qué grupos buscar.
rel: clustering, reduccion-de-la-dimensionalidad, estimacion-de-densidad, aprendizaje-autosupervisado
en: unsupervised learning

## redes-neuronales
frase: Cómo una red ajusta millones de pesos para reducir su error, paso a paso.
desc: Cómo aprende una red neuronal. Las neuronas combinan sus entradas con pesos y aplican una activación no lineal; una función de pérdida mide el error, y un optimizador ajusta millones de pesos mediante descenso por gradiente y retropropagación. Incluye también la regularización y los trucos prácticos que hacen que el entrenamiento funcione.
cuando: Cuando los datos son complejos (imágenes, texto, audio) o muy abundantes y los modelos clásicos se quedan cortos.
ej: Entrenar un MLP para reconocer dígitos escritos a mano: en cada época la red ve todas las imágenes, calcula su error y corrige un poco los pesos en la dirección que lo reduce.
py: keras.Sequential ; torch.nn.Module
rel: mlp, descenso-por-gradiente, backpropagation, arquitecturas-de-redes-neuronales
en: neural networks: how they learn

## arquitecturas-de-redes-neuronales
frase: Las formas de red adaptadas a cada tipo de dato: imágenes, secuencias o texto.
desc: Las formas que puede tomar una red según el tipo de dato. Las CNN aprovechan la estructura espacial de las imágenes y las RNN la secuencial del texto o las series. Los Transformers relacionan todos los elementos entre sí mediante atención, y los autoencoders comprimen y reconstruyen. Cada arquitectura incorpora un sesgo inductivo distinto.
ej: Para clasificar imágenes se suele usar una CNN o un Vision Transformer; para traducir texto, un Transformer encoder-decoder.
rel: cnn, rnn, atencion-y-transformers, autoencoders
en: neural network architectures

## representaciones-y-modelos-fundacionales
frase: Preentrenar una vez con muchos datos y reutilizar lo aprendido en muchas tareas.
desc: El paradigma de preentrenar y adaptar. Un modelo aprende representaciones generales (embeddings) con enormes cantidades de datos, a menudo sin etiquetas gracias al aprendizaje autosupervisado. Después se reutiliza para muchas tareas mediante transfer learning o fine-tuning. Es la base de los modelos fundacionales.
cuando: Cuando tienes pocos datos etiquetados y existe un modelo preentrenado con datos parecidos a los tuyos.
ej: Reutilizar un modelo preentrenado con millones de imágenes para clasificar radiografías disponiendo solo de unos cientos de ejemplos etiquetados.
rel: embeddings, aprendizaje-autosupervisado, transfer-learning, que-es-un-modelo-fundacional
en: representation learning & foundation models

## ia-generativa
frase: Modelos que aprenden cómo son los datos para crear ejemplos nuevos y realistas.
desc: Modelos que aprenden la distribución de los datos para generar ejemplos nuevos y realistas: imágenes, texto, audio o datos sintéticos. Las grandes familias son los modelos autorregresivos, los VAE, las GAN, los flujos normalizantes y los modelos de difusión. Cada una resuelve a su manera cómo modelar y muestrear una distribución de muy alta dimensión.
cuando: Cuando necesitas crear contenido nuevo (imágenes, texto, audio) o datos sintéticos realistas.
ej: Stable Diffusion genera una imagen a partir de una frase: parte de ruido aleatorio y lo va eliminando paso a paso.
rel: modelo-generativo-vs-discriminativo, modelos-de-difusion, gan, vae
en: generative AI

## lenguaje-y-llms
frase: Cómo las máquinas representan, entienden y generan lenguaje, hasta llegar a los LLMs.
desc: Cómo representan y procesan el lenguaje las máquinas, desde la bolsa de palabras y los word embeddings hasta los grandes modelos de lenguaje (LLMs). Un LLM es un Transformer preentrenado para predecir el siguiente token y adaptado después (instruction tuning, RLHF) para seguir instrucciones y conversar.
cuando: Cuando el problema gira en torno a texto: clasificarlo, extraer información, resumirlo, traducirlo o generarlo.
ej: ChatGPT, Gemini o Claude son LLMs: modelos preentrenados con billones de tokens de texto y ajustados para seguir instrucciones.
rel: tokenizacion, word-embeddings, llm, adaptacion-de-llms
en: natural language processing & LLMs

## rag-y-agentes
frase: Conectar un LLM con documentos y herramientas para que responda con fundamento y actúe.
desc: Cómo convertir un LLM en un sistema útil y fiable. El RAG lo conecta con documentos externos: recupera los fragmentos relevantes y los añade al prompt para que la respuesta se apoye en ellos. Un agente da al LLM herramientas (buscar, ejecutar código, llamar a APIs) y un bucle en el que decide qué hacer en cada paso hasta completar la tarea.
cuando: Cuando un LLM debe responder con información propia o actualizada, o completar tareas de varios pasos usando herramientas.
ej: Un asistente que responde sobre la documentación interna de una empresa citando sus fuentes es un RAG. Si además puede abrir incidencias o enviar correos, es un agente.
rel: rag, agentes-de-ia, alucinaciones, function-calling
en: RAG & AI agents

## vision-por-computador
frase: Máquinas que interpretan imágenes y vídeo: clasifican, detectan, segmentan y reconstruyen.
desc: Enseñar a las máquinas a interpretar imágenes y vídeo: clasificar, detectar objetos, segmentar píxeles, estimar profundidad o reconstruir escenas en 3D. El campo ha pasado de los filtros diseñados a mano a las CNN y los Vision Transformers entrenados con grandes cantidades de datos.
cuando: Cuando la información que necesitas está en imágenes o en vídeo.
ej: Un coche autónomo detecta peatones y señales (detección) y separa la calzada del resto de la escena (segmentación) en tiempo real.
rel: cnn, vit, deteccion-de-objetos, segmentacion
en: computer vision

## ia-multimodal-y-audio
frase: Modelos que combinan texto, imagen, audio y vídeo en un mismo sistema.
desc: Modelos que combinan varias modalidades (texto, imagen, audio y vídeo) en un mismo sistema. Sirven para buscar imágenes con texto, describir una foto, responder preguntas sobre un documento escaneado o transcribir y sintetizar voz. La clave es alinear las modalidades en un espacio de representación común.
cuando: Cuando el problema mezcla modalidades, como buscar imágenes con texto o preguntar sobre un documento escaneado, o cuando trabajas con voz.
ej: CLIP permite buscar "un perro corriendo por la playa" entre millones de fotos aunque ninguna esté etiquetada con esas palabras.
rel: clip, mllm, texto-a-imagen, audio-y-voz
en: multimodal AI & audio

## series-temporales
frase: Datos ordenados en el tiempo: anticipar lo que viene y entender su dinámica.
desc: Datos ordenados en el tiempo, donde cada valor depende de los anteriores: ventas, temperatura, tráfico o señales de sensores. Los problemas típicos son predecir valores futuros (forecasting), clasificar secuencias y detectar comportamientos anómalos. Exigen validar respetando el orden temporal.
cuando: Cuando el orden temporal importa y quieres anticipar valores futuros o entender cómo evoluciona algo.
ej: Predecir la demanda eléctrica de las próximas 24 horas a partir del consumo de las últimas semanas y la previsión meteorológica.
rel: prediccion-multipaso, enventanado, modelos-fundacionales-de-series, validacion-en-series-temporales
en: time series

## sistemas-de-recomendacion
frase: Predecir qué le interesará a cada usuario a partir de su comportamiento y el de otros.
desc: Sistemas que predicen qué ítems (películas, productos, canciones) interesarán a cada usuario a partir de su historial y del de usuarios parecidos. Combinan filtrado basado en contenido, filtrado colaborativo, factorización de matrices y modelos profundos basados en embeddings.
cuando: Cuando hay muchos ítems entre los que elegir y quieres mostrar a cada usuario los más relevantes para él.
ej: Netflix recomienda series según lo que has visto y lo que vieron usuarios con gustos similares, y hasta elige qué portada enseñarte.
rel: filtrado-colaborativo, factorizacion-de-matrices, embeddings, recall-k-y-hitrate-k
en: recommender systems

## deteccion-de-anomalias
frase: Encontrar lo raro: datos que no encajan con el comportamiento normal.
desc: Identificar observaciones que se desvían del comportamiento normal: fraudes, averías, intrusiones o errores de medida. Como las anomalías son raras y casi nunca están etiquetadas, se suele modelar la normalidad (por distancia, probabilidad o reconstrucción) y marcar lo que no encaja.
cuando: Cuando buscas casos raros (fraude, averías, errores) y apenas tienes ejemplos etiquetados de ellos.
ej: Un autoencoder entrenado solo con transacciones normales reconstruye mal una fraudulenta, y ese error de reconstrucción alto la delata.
rel: outliers, deteccion-por-reconstruccion-o-prediccion, isolation-forest, ood
en: anomaly detection

## ia-fiable-y-responsable
frase: Que la IA sea explicable, justa, robusta, consciente de su incertidumbre y legal.
desc: Lo que hace falta para que la IA sea digna de confianza. Hay que entender por qué decide lo que decide (explicabilidad), evitar que discrimine (equidad), que resista manipulaciones (robustez), que sepa cuándo no está segura (incertidumbre) y que cumpla la regulación (RGPD, AI Act).
cuando: Cuando un modelo toma decisiones que afectan a personas o hay que justificarlas, auditarlas o adaptarlas a la regulación.
ej: Si un modelo deniega un préstamo, SHAP muestra qué variables pesaron en la decisión. Una explicación contrafactual indica qué tendría que cambiar para aprobarlo.
rel: xai, equidad, robustez, ai-act
en: trustworthy & responsible AI

## aprendizaje-por-refuerzo
frase: Aprender a decidir por ensayo y error, maximizando una recompensa acumulada.
desc: Un agente aprende a tomar decisiones interactuando con un entorno. Actúa, recibe una recompensa y ajusta su política para maximizar la recompensa acumulada. No hay etiquetas con la respuesta correcta, solo señales de lo bien o mal que ha ido, y el dilema central es explorar acciones nuevas o explotar las que ya funcionan.
cuando: Cuando hay que tomar decisiones en secuencia y aprender de la recompensa que generan, sin respuestas correctas de antemano.
ej: AlphaGo Zero aprendió a jugar al go desde cero jugando millones de partidas contra sí mismo. Un sistema de anuncios elige qué banner mostrar según los clics que va recibiendo.
rel: agente-entorno-y-recompensa, exploracion-vs-explotacion, multi-armed-bandits, rlhf
en: reinforcement learning
