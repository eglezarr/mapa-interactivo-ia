# Fichas: Sistemas de recomendación

## el-problema-de-la-recomendacion
frase: Predecir qué elementos interesarán a cada usuario entre miles o millones de opciones.
desc: Un recomendador estima, para cada par de usuario e ítem, cuánto le gustará o lo probable que es que interactúe, y ordena los ítems para mostrar los mejores. El problema tiene rasgos propios: la mayoría de pares no tiene datos, los usuarios y los ítems nuevos no tienen historial, las preferencias cambian y lo que se muestra condiciona los datos futuros. En la práctica se aborda como un problema de ordenación (ranking), no solo de predicción.
ej: Una plataforma de vídeo elige, entre 50.000 títulos, los 10 que muestra en la portada de cada usuario.
rel: matriz-de-utilidad, filtrado-colaborativo, filtrado-basado-en-contenido, evaluacion-de-recomendadores
en: the recommendation problem

## matriz-de-utilidad
frase: La tabla de usuarios por ítems con las valoraciones o interacciones conocidas, casi toda vacía.
desc: Cada fila es un usuario, cada columna un ítem y cada celda, una valoración o una interacción (compra, clic, tiempo de visionado). Es extremadamente dispersa: un usuario típico interactúa con una fracción ínfima del catálogo, a menudo menos del 1 %. Recomendar equivale a rellenar las celdas vacías con predicciones; se guarda en formato disperso para no almacenar los ceros.
ej: Una tienda con un millón de clientes y 100.000 productos tiene una matriz de 100.000 millones de celdas, de las que solo unos pocos millones tienen datos.
py: scipy.sparse.csr_matrix
rel: el-problema-de-la-recomendacion, factorizacion-de-matrices, feedback-explicito-vs-implicito, filtrado-colaborativo
en: utility matrix (user-item)

## feedback-explicito-vs-implicito
frase: Valoraciones que el usuario da a propósito frente a señales que se deducen de su comportamiento.
desc: El feedback explícito (estrellas, "me gusta") es claro, pero escaso y sesgado: pocos usuarios valoran, y suelen hacerlo los muy satisfechos o los muy enfadados. El implícito (clics, compras, tiempo de visionado, reproducciones) es abundante, pero ambiguo: no ver algo no significa que no guste, y un clic no implica satisfacción. Los modelos para feedback implícito dan menos confianza a la ausencia de interacción que a su presencia.
ej: En una plataforma de música, escuchar una canción entera diez veces es una señal más fiable que la mayoría de las valoraciones con estrellas.
rel: matriz-de-utilidad, factorizacion-de-matrices, filtrado-colaborativo, calidad-y-etiquetado-de-datos
en: explicit vs implicit feedback

## filtrado-basado-en-contenido
frase: Recomendar ítems que, por sus características, se parecen a los que el usuario ya ha valorado bien.
desc: Describe cada ítem con atributos (género, autor, ingredientes, el texto de su descripción o embeddings) y construye un perfil del usuario a partir de los ítems que le gustaron; después recomienda los ítems más parecidos a ese perfil. No necesita datos de otros usuarios, así que funciona con ítems nuevos y es fácil de explicar, pero tiende a recomendar más de lo mismo y depende de la calidad de los atributos.
cuando: Con ítems nuevos, catálogos con buenas descripciones o pocos usuarios.
ej: A un lector de novela negra nórdica se le recomiendan otras novelas con descripciones parecidas, aunque nadie más las haya leído.
py: sklearn.feature_extraction.text.TfidfVectorizer ; sklearn.metrics.pairwise.cosine_similarity
rel: filtrado-colaborativo, sistemas-hibridos, similitud-coseno, arranque-en-frio
en: content-based filtering

## filtrado-colaborativo
frase: Recomendar a partir de lo que les gustó a usuarios con gustos parecidos, sin mirar el contenido.
desc: Se basa solo en la matriz de interacciones. La versión basada en usuarios busca usuarios similares y recomienda lo que les gustó a ellos; la basada en ítems recomienda ítems parecidos a los que el usuario valoró, midiendo la similitud por quién los consume. Los métodos de factorización generalizan la idea. Descubre relaciones que el contenido no muestra, pero sufre con los usuarios y los ítems nuevos.
cuando: Cuando hay abundante historial de interacciones y el contenido de los ítems es difícil de describir.
ej: "Los clientes que compraron esta tienda de campaña también compraron este hornillo".
hp: k: número de vecinos que se consideran | sim_options: medida de similitud y si se compara por usuarios o por ítems
py: surprise.KNNBasic
rel: factorizacion-de-matrices, filtrado-basado-en-contenido, knn, arranque-en-frio
en: collaborative filtering

## factorizacion-de-matrices
frase: Representar usuarios e ítems con vectores latentes cuyo producto escalar predice la afinidad.
desc: Aproxima la matriz de utilidad como el producto de dos matrices estrechas: una con un vector de factores latentes por usuario y otra con uno por ítem. Los factores se aprenden con las celdas conocidas, por descenso por gradiente o por mínimos cuadrados alternos (ALS), con regularización. Capturan dimensiones ocultas de los gustos, como "acción frente a drama", y fueron la técnica ganadora del premio Netflix.
cuando: Como modelo de referencia de filtrado colaborativo con muchas interacciones, explícitas o implícitas.
ej: Con 50 factores latentes por usuario y por película, el modelo predice qué nota pondrá cada usuario a las películas que no ha visto.
hp: n_factors / factors: número de factores latentes | reg_all / regularization: fuerza de la regularización | n_epochs / iterations: pasadas de entrenamiento
py: surprise.SVD ; implicit.als.AlternatingLeastSquares ; sklearn.decomposition.NMF
rel: matriz-de-utilidad, filtrado-colaborativo, embeddings, pca
en: matrix factorization

## sistemas-hibridos
frase: Combinar filtrado colaborativo y basado en contenido para aprovechar las ventajas de ambos.
desc: Los sistemas híbridos mezclan las predicciones de varios recomendadores, usan los atributos del contenido como entradas adicionales de un modelo colaborativo o cambian de estrategia según la situación: contenido para los ítems nuevos y colaborativo cuando ya hay historial. Reducen el problema del arranque en frío y suelen ser más precisos. La mayoría de los recomendadores reales son híbridos.
cuando: Casi siempre en producción, sobre todo si llegan ítems o usuarios nuevos con frecuencia.
ej: Una plataforma de noticias recomienda los artículos recién publicados por su contenido y, cuando acumulan lecturas, también por lo que leen los usuarios parecidos.
hp: no_components: dimensión de los embeddings de usuarios e ítems | loss: función de pérdida (WARP, BPR o logística)
py: lightfm.LightFM
rel: filtrado-basado-en-contenido, filtrado-colaborativo, arranque-en-frio, ensembles
en: hybrid recommender systems

## recomendacion-basada-en-sesion
frase: Recomendar a partir de lo que el usuario está haciendo en la visita actual, aunque sea anónimo.
desc: Muchos usuarios no inician sesión o cambian de intención en cada visita. Los modelos basados en sesión tratan la secuencia de clics de la visita actual como una frase y predicen el siguiente ítem, con redes recurrentes (GRU4Rec), transformers (SASRec, BERT4Rec) o reglas de coocurrencia. Capturan la intención inmediata mejor que los perfiles a largo plazo.
cuando: En comercio electrónico, noticias o música, con muchos usuarios anónimos o intenciones que cambian entre visitas.
ej: Un visitante anónimo mira tres zapatillas de trail en cinco minutos y la tienda le muestra calcetines técnicos y mochilas de hidratación.
rel: modelos-autorregresivos, rnn, el-problema-de-la-recomendacion, arranque-en-frio
en: session-based recommendation

## arranque-en-frio
frase: Cómo recomendar a usuarios nuevos, o recomendar ítems nuevos de los que aún no hay interacciones.
desc: El filtrado colaborativo no sabe nada de quien no tiene historial. Para los usuarios nuevos se recurre a lo más popular, a preguntas iniciales sobre sus gustos o a datos de contexto (ubicación, dispositivo); para los ítems nuevos, a su contenido o a mostrarlos a una pequeña fracción de usuarios para explorar. Los bandits equilibran explotar lo conocido y explorar lo nuevo.
cuando: En cualquier recomendador con altas frecuentes de usuarios o de ítems, desde el primer día.
ej: Una app de música pregunta al registrarse por tres artistas favoritos para tener algo con lo que empezar a recomendar.
rel: filtrado-basado-en-contenido, sistemas-hibridos, multi-armed-bandits, exploracion-vs-explotacion
en: cold start

## recomendacion-con-deep-learning
frase: Redes neuronales que aprenden las interacciones entre usuarios, ítems y su contexto.
desc: Generalizan la factorización de matrices sustituyendo el producto escalar por redes que modelan interacciones no lineales, e incorporan con facilidad texto, imágenes, secuencias y contexto. Los sistemas a gran escala suelen tener dos etapas: un modelo de recuperación, a menudo de dos torres (una para usuarios y otra para ítems), que selecciona cientos de candidatos entre millones, y un modelo de ordenación más rico que los puntúa.
cuando: Con catálogos y volúmenes de datos grandes y señales variadas (texto, imágenes, contexto, secuencias).
ej: Una red social elige entre millones de publicaciones unas 500 candidatas con un modelo de dos torres y las ordena con una red que tiene en cuenta cientos de señales.
rel: autorec, ncf, dcn, scoring-y-re-ranking, embeddings
en: deep learning recommenders

## autorec
frase: Un autoencoder que reconstruye las valoraciones de un ítem, o de un usuario, y al hacerlo predice las que faltan.
desc: Toma como entrada el vector de valoraciones conocidas de un ítem (o de un usuario), lo comprime en un cuello de botella y lo reconstruye; la pérdida solo se calcula sobre las valoraciones observadas. Las salidas en las celdas vacías son las predicciones. Es una generalización no lineal de la factorización de matrices y una de las primeras aplicaciones del deep learning a la recomendación.
cuando: Como modelo colaborativo no lineal sencillo cuando hay valoraciones explícitas.
ej: A partir de las notas que 2.000 usuarios han dado a una película, AutoRec reconstruye las que le pondrían los usuarios que aún no la han visto.
rel: autoencoders, factorizacion-de-matrices, recomendacion-con-deep-learning, cuello-de-botella
en: AutoRec

## ncf
frase: Sustituir el producto escalar de la factorización de matrices por una red neuronal que aprende la interacción.
desc: Neural Collaborative Filtering representa usuarios e ítems con embeddings y combina dos ramas: GMF, una factorización generalizada (producto elemento a elemento con pesos aprendidos), y un MLP sobre los embeddings concatenados, que capta interacciones no lineales. NeuMF fusiona ambas. Se entrena con feedback implícito como una clasificación (interactuó o no), muestreando ejemplos negativos.
cuando: Con abundante feedback implícito, cuando la factorización lineal se queda corta.
ej: Una tienda online predice la probabilidad de que cada cliente compre cada producto combinando las dos ramas de NeuMF.
py: keras.layers.Embedding ; keras.layers.Dot
rel: factorizacion-de-matrices, mlp, capa-de-embedding, negative-sampling
en: Neural Collaborative Filtering (GMF, NeuMF)

## dcn
frase: Red que aprende automáticamente combinaciones explícitas entre variables (cruces) junto a una red profunda.
desc: En recomendación y publicidad, muchas señales son interacciones entre variables, como "usuario joven" y "categoría videojuegos". La Deep & Cross Network combina una red de cruces, que en cada capa multiplica la entrada original por la salida de la capa anterior y genera de forma eficiente interacciones de orden creciente, con una red profunda convencional. Sustituye la creación manual de cruces de variables.
cuando: Para predecir clics o compras con muchas variables categóricas y numéricas cuyas interacciones importan.
ej: Predecir la probabilidad de clic en un anuncio combinando automáticamente el país, la hora, el tipo de dispositivo y la categoría del anuncio.
hp: projection_dim: dimensión de la proyección de rango bajo de la capa de cruce, que reduce los parámetros
py: keras_rs.layers.FeatureCross
rel: recomendacion-con-deep-learning, regresion-polinomica-e-interacciones, feature-engineering, mlp
en: Deep & Cross Network (DCN)

## scoring-y-re-ranking
frase: Puntuar los candidatos con un modelo preciso y reordenarlos según objetivos de negocio y de diversidad.
desc: Tras recuperar los candidatos, un modelo de puntuación más rico estima para cada uno la probabilidad de clic, de compra o de satisfacción. Después, el reordenamiento ajusta la lista final con criterios que el modelo no ve: diversidad, novedad, frescura, reglas de negocio o equidad entre proveedores, y evita mostrar seguidos ítems casi idénticos. La lista final es un compromiso entre la relevancia y otros objetivos.
cuando: En cualquier recomendador en producción con catálogos grandes y objetivos que van más allá de la relevancia inmediata.
ej: Tras puntuar 500 vídeos, el sistema evita mostrar cinco seguidos del mismo creador y reserva un hueco para un contenido nuevo.
py: keras_rs.layers.BruteForceRetrieval
rel: recomendacion-con-deep-learning, re-ranking, diversidad-novedad-y-serendipia, burbuja-de-filtro
en: scoring & re-ranking

## burbuja-de-filtro
frase: El recomendador solo muestra lo que confirma los gustos del usuario y acaba encerrándolo en una burbuja.
desc: Como el sistema aprende de lo que el usuario consume, y el usuario consume lo que el sistema le muestra, se crea un bucle de retroalimentación que estrecha progresivamente la variedad de lo recomendado. Puede empobrecer la experiencia, amplificar la polarización en noticias y redes sociales y favorecer siempre a los ítems populares. Se mitiga con diversidad, exploración y métricas que vayan más allá del clic inmediato.
ej: Un usuario que ve dos vídeos sobre una dieta extrema empieza a recibir casi exclusivamente contenidos cada vez más extremos sobre el tema.
rel: diversidad-novedad-y-serendipia, exploracion-vs-explotacion, sesgo-algoritmico-y-atributos-sensibles, scoring-y-re-ranking
en: filter bubble

## evaluacion-de-recomendadores
frase: Medir si las recomendaciones son relevantes, están bien ordenadas, son variadas y útiles para el negocio.
desc: Offline se mide sobre interacciones históricas reservadas: métricas de error si se predicen valoraciones y, más útiles, métricas de ordenación como Recall@K, NDCG o MAP, que premian colocar arriba los ítems relevantes; también la diversidad, la novedad y la cobertura del catálogo. Como los datos históricos dependen de lo que se mostró en su día, la prueba definitiva es un test A/B online con métricas de negocio.
ej: Un modelo nuevo mejora el NDCG@10 offline, pero el test A/B muestra que no aumenta el tiempo de visionado, así que no se despliega.
py: sklearn.metrics.ndcg_score ; keras_rs.metrics.NDCG
rel: recall-k-y-hitrate-k, diversidad-novedad-y-serendipia, contraste-de-hipotesis, metricas-de-evaluacion
en: recommender evaluation

## recall-k-y-hitrate-k
frase: Qué fracción de los ítems relevantes aparece entre las K primeras recomendaciones, o si aparece al menos uno.
desc: Recall@K divide el número de ítems relevantes que aparecen en el top-K entre el total de ítems relevantes del usuario. HitRate@K vale 1 si al menos uno de los relevantes está en el top-K y 0 si no, y se promedia sobre los usuarios; es habitual cuando se oculta una sola interacción por usuario. Precision@K mide qué fracción del top-K es relevante. Ninguna tiene en cuenta la posición exacta dentro del top-K; NDCG sí.
cuando: Para evaluar recomendadores y sistemas de recuperación, incluido RAG, cuando solo se muestran los K primeros resultados.
ej: Si un usuario compró 4 productos en el periodo de test y 2 aparecen entre sus 10 recomendaciones, su Recall@10 es 0,5 y su HitRate@10, 1.
py: keras_rs.metrics.RecallAtK ; `len(set(top_k) & relevantes) / len(relevantes)`
rel: evaluacion-de-recomendadores, precision-y-recall, evaluacion-de-rag, re-ranking
en: Recall@K & HitRate@K

## diversidad-novedad-y-serendipia
frase: Más allá de acertar: recomendar cosas variadas, poco conocidas y gratamente inesperadas.
desc: La diversidad mide lo distintos que son entre sí los ítems de una lista; la novedad, lo poco populares o desconocidos que son para el usuario; la serendipia, lo sorprendentes y a la vez relevantes que resultan. Un recomendador que solo optimiza la precisión tiende a sugerir lo obvio y lo popular. Estas métricas ayudan a equilibrar la relevancia con el descubrimiento y la cobertura del catálogo.
cuando: Para evaluar y ajustar recomendadores en los que importan el descubrimiento y la satisfacción a largo plazo.
ej: Recomendar a un fan de los Beatles otro disco de los Beatles es preciso pero obvio; recomendarle un grupo poco conocido con un sonido parecido aporta serendipia.
rel: burbuja-de-filtro, scoring-y-re-ranking, evaluacion-de-recomendadores, exploracion-vs-explotacion
en: diversity, novelty & serendipity
