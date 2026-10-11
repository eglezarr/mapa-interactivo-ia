# Fichas: Aprendizaje no supervisado

## reduccion-de-la-dimensionalidad
frase: Representar los datos con menos variables conservando lo esencial de su estructura.
desc: Con cientos o miles de variables, los datos son difíciles de visualizar, almacenar y modelar, y muchas variables son redundantes. La reducción de la dimensionalidad elige un subconjunto de variables (selección) o crea otras nuevas que combinan las originales (extracción), de forma lineal, como PCA, o no lineal, como t-SNE o UMAP. Se usa para visualizar, comprimir, quitar ruido y acelerar otros modelos.
ej: Resumir los 20.000 genes medidos en cada paciente en 50 componentes que conservan casi toda la variación entre pacientes.
rel: maldicion-de-la-dimensionalidad, pca, seleccion-de-caracteristicas, manifold-learning
en: dimensionality reduction

## maldicion-de-la-dimensionalidad
frase: Al crecer el número de dimensiones, los datos se vuelven escasos y las distancias pierden significado.
desc: El volumen del espacio crece exponencialmente con las dimensiones, así que cualquier cantidad de datos se vuelve escasa. En muchas dimensiones las distancias entre puntos tienden a parecerse, lo que perjudica a los métodos basados en vecinos, y crece el riesgo de sobreajuste. Por suerte, los datos reales suelen vivir cerca de estructuras de dimensión mucho menor.
ej: Cubrir un cubo con una rejilla de 10 puntos por eje requiere 100 puntos en 2 dimensiones y 10²⁰ en 20 dimensiones.
rel: reduccion-de-la-dimensionalidad, knn, sobreajuste, manifold-learning, distancia-euclidea
en: curse of dimensionality

## seleccion-de-caracteristicas
frase: Quedarse con las variables útiles y descartar las irrelevantes o redundantes.
desc: A diferencia de la extracción, conserva variables originales, así que el modelo sigue siendo interpretable. Reduce el sobreajuste y el coste de entrenamiento. Hay tres familias: los filtros puntúan cada variable por separado, los envoltorios evalúan subconjuntos entrenando el modelo y los métodos integrados seleccionan durante el propio entrenamiento.
cuando: Cuando hay muchas variables candidatas, algunas probablemente irrelevantes, y quieres un modelo más simple, rápido o interpretable.
ej: De 300 indicadores de un sensor industrial, quedarse con los 15 que de verdad anticipan una avería.
py: sklearn.feature_selection
rel: metodos-filtro, metodos-envoltorio, metodos-integrados, extraccion-de-caracteristicas
en: feature selection

## metodos-filtro
frase: Puntuar cada variable por su relación con el objetivo, sin entrenar el modelo final.
desc: Ordenan las variables según una medida estadística, como la correlación, la información mutua, el test chi-cuadrado o el estadístico F del ANOVA, y se quedan con las mejores. Son rápidos y escalan a miles de variables, pero evalúan cada variable por separado: no detectan redundancias ni variables que solo son útiles en combinación con otras.
cuando: Como primer cribado rápido cuando hay muchísimas variables.
ej: En clasificación de textos con 50.000 palabras posibles, quedarse con las 2.000 de mayor chi-cuadrado respecto a la categoría.
hp: k: número de variables que se conservan | score_func: medida con la que se puntúa cada variable
py: sklearn.feature_selection.SelectKBest ; sklearn.feature_selection.mutual_info_classif
rel: seleccion-de-caracteristicas, informacion-mutua, test-chi-cuadrado, correlacion-de-pearson
en: filter methods

## metodos-envoltorio
frase: Elegir variables probando subconjuntos con el propio modelo y midiendo su rendimiento.
desc: Tratan la selección como una búsqueda: añaden variables una a una (selección hacia delante), las quitan (eliminación hacia atrás) o eliminan recursivamente las menos importantes (RFE), evaluando cada subconjunto con validación cruzada. Tienen en cuenta las interacciones y el modelo concreto, pero son costosos y pueden sobreajustar la propia selección.
cuando: Con un número moderado de variables y un modelo rápido de entrenar, cuando importa encontrar el mejor subconjunto.
ej: Eliminar recursivamente variables de un modelo de abandono de clientes hasta encontrar las 10 que maximizan el AUC en validación cruzada.
hp: n_features_to_select: número de variables que se quieren conservar | direction: en la selección secuencial, hacia delante o hacia atrás
py: sklearn.feature_selection.RFECV ; sklearn.feature_selection.SequentialFeatureSelector
rel: seleccion-de-caracteristicas, validacion-cruzada, metodos-integrados, metodos-filtro
en: wrapper methods

## metodos-integrados
frase: Modelos que seleccionan variables por sí solos mientras se entrenan.
desc: La selección ocurre dentro del entrenamiento. Lasso anula los coeficientes de las variables poco útiles, y los modelos de árboles calculan la importancia de cada variable según cuánto reduce el error. Son más baratos que los envoltorios y tienen en cuenta el modelo, aunque la importancia de los árboles puede estar sesgada hacia variables con muchos valores distintos.
cuando: Cuando ya vas a entrenar un modelo con regularización L1 o de árboles y quieres aprovecharlo para seleccionar.
ej: Entrenar un Lasso sobre 400 variables macroeconómicas y conservar las 25 con coeficiente distinto de cero.
hp: threshold: importancia mínima para conservar una variable | max_features: número máximo de variables seleccionadas
py: sklearn.feature_selection.SelectFromModel ; sklearn.linear_model.LassoCV
rel: ridge-lasso-y-elastic-net, random-forest, importancia-por-permutacion, seleccion-de-caracteristicas
en: embedded methods

## extraccion-de-caracteristicas
frase: Crear variables nuevas que combinan las originales y concentran su información en menos dimensiones.
desc: En lugar de elegir variables, construye otras nuevas como combinaciones o transformaciones de las originales. Puede ser lineal (PCA, ICA, LDA) o no lineal (kernel PCA, autoencoders, métodos de variedades). Comprime mucho mejor que la selección, pero las nuevas variables son más difíciles de interpretar.
cuando: Cuando las variables están muy correlacionadas o son muy numerosas y la interpretabilidad de cada una no es prioritaria.
ej: Convertir imágenes de caras de 64 × 64 píxeles (4.096 valores) en 100 componentes que bastan para reconocerlas.
py: sklearn.decomposition.PCA ; sklearn.decomposition.KernelPCA
rel: pca, autoencoders, seleccion-de-caracteristicas, embeddings
en: feature extraction

## pca
frase: Rota los datos para encontrar las direcciones de máxima varianza y se queda con las principales.
desc: El análisis de componentes principales busca nuevos ejes ortogonales, combinaciones lineales de las variables, ordenados por la varianza que capturan. Quedarse con los primeros comprime los datos con la mínima pérdida en error cuadrático. Se calcula con la descomposición en valores singulares, es sensible a la escala de las variables y solo capta estructura lineal.
cuando: Para comprimir variables numéricas correlacionadas, visualizar en 2D o 3D, eliminar ruido o acelerar otros modelos.
ej: Con medidas corporales muy correlacionadas (altura, envergadura, talla de pie…), la primera componente resume el tamaño general de cada persona.
hp: n_components: número de componentes, o fracción de varianza que se quiere conservar | whiten: escala las componentes a varianza unidad
py: sklearn.decomposition.PCA
rel: estandarizacion, ica, autoencoder-undercomplete, extraccion-de-caracteristicas, reduccion-de-la-dimensionalidad
en: principal component analysis (PCA)

## ica
frase: Separa una mezcla de señales en sus fuentes originales, estadísticamente independientes.
desc: El análisis de componentes independientes supone que los datos son mezclas lineales de fuentes independientes y no gaussianas, y busca la transformación que las recupera. A diferencia de PCA, que solo descorrelaciona, ICA exige independencia, lo que permite separar señales superpuestas. No determina el orden ni la escala de las fuentes.
cuando: Para separar señales mezcladas, como el audio de varios micrófonos o registros fisiológicos con artefactos.
ej: El problema del cóctel: con dos micrófonos en una sala donde hablan dos personas, ICA recupera cada voz por separado.
hp: n_components: número de fuentes que se buscan | fun: función con la que se aproxima la no gaussianidad
py: sklearn.decomposition.FastICA
rel: pca, independencia-estadistica, correlacion-total, extraccion-de-caracteristicas
en: independent component analysis (ICA)

## lda
frase: Proyección supervisada que separa al máximo las clases: lejos entre sí y compactas por dentro.
desc: El análisis discriminante lineal busca las direcciones que maximizan la separación entre las medias de las clases respecto a la dispersión dentro de cada clase. Con K clases da como mucho K − 1 componentes. A diferencia de PCA usa las etiquetas, así que sirve para reducir la dimensión antes de clasificar y también como clasificador. No hay que confundirlo con el LDA de modelado de temas (Latent Dirichlet Allocation).
cuando: Para reducir la dimensión o visualizar datos etiquetados cuando lo que importa es separar las clases.
ej: Proyectar las medidas de flores de tres especies en un plano donde las especies aparecen claramente separadas.
hp: n_components: dimensiones de la proyección (como mucho, número de clases − 1) | solver: método de cálculo (svd, lsqr o eigen) | shrinkage: regularización de la covarianza cuando hay pocos datos
py: sklearn.discriminant_analysis.LinearDiscriminantAnalysis
rel: pca, regresion-logistica, reduccion-de-la-dimensionalidad, distancia-de-mahalanobis
en: linear discriminant analysis (LDA)

## mds
frase: Coloca los puntos en pocas dimensiones respetando lo mejor posible las distancias entre ellos.
desc: El escalado multidimensional recibe una matriz de distancias o disimilitudes y busca coordenadas en 2 o 3 dimensiones cuyas distancias se parezcan lo más posible a las originales. Solo necesita las distancias, no las variables, así que sirve cuando únicamente se sabe cuánto se parecen los objetos. La versión clásica con distancias euclídeas equivale a PCA.
cuando: Para visualizar objetos de los que solo se conoce cuánto se parecen entre sí.
ej: A partir de la tabla de distancias por carretera entre ciudades, reconstruir un mapa aproximado de su posición.
hp: n_components: dimensiones de la representación | metric_mds: respetar las distancias en sí (MDS métrico) o solo su orden (no métrico)
py: sklearn.manifold.MDS
rel: isomap, pca, distancia-y-similitud, t-sne
en: multidimensional scaling (MDS)

## manifold-learning
frase: Métodos que suponen que los datos viven sobre una superficie curva de pocas dimensiones y la despliegan.
desc: Los datos de alta dimensión suelen ocupar una variedad (manifold) de dimensión mucho menor: una superficie curvada dentro del espacio. Estos métodos no lineales tratan de desenrollarla preservando relaciones locales, como los vecinos de cada punto o las distancias medidas sobre la superficie. Son muy útiles para visualizar, pero sensibles a sus hiperparámetros.
cuando: Para visualizar o reducir datos con estructura no lineal que PCA no capta.
ej: Fotos de una cara que gira la cabeza: aunque cada imagen tiene miles de píxeles, solo varían en una dimensión, el ángulo de giro.
py: sklearn.manifold
rel: lle-y-pca-local, isomap, t-sne, umap, maldicion-de-la-dimensionalidad
en: manifold learning

## lle-y-pca-local
frase: Desplegar la variedad aproximándola por trozos lineales alrededor de cada punto.
desc: La idea es que, vista de cerca, una superficie curva parece plana. LLE (Locally Linear Embedding) reconstruye cada punto como combinación lineal de sus vecinos y busca una representación de baja dimensión que conserve esos mismos pesos. El PCA local aplica PCA en pequeñas vecindades para estimar la geometría local. Ambos captan curvaturas que un PCA global no ve.
cuando: Para desplegar variedades suaves y bien muestreadas cuando importa conservar la geometría local.
ej: Desenrollar el clásico "rollo suizo", una hoja enrollada en 3D, sobre un plano.
hp: n_neighbors: tamaño de la vecindad local | n_components: dimensión de la representación
py: sklearn.manifold.LocallyLinearEmbedding
rel: manifold-learning, pca, isomap, knn
en: LLE & local PCA

## isomap
frase: Despliega la variedad conservando las distancias medidas a lo largo de su superficie.
desc: Construye un grafo uniendo cada punto con sus vecinos cercanos y calcula las distancias geodésicas como caminos más cortos en ese grafo, es decir, distancias recorridas sobre la superficie. Después aplica MDS a esas distancias. Así, dos puntos cercanos en línea recta pero lejanos sobre la variedad quedan separados.
cuando: Para desplegar variedades no lineales cuando importa la estructura global y no hay agujeros ni ruido excesivo.
ej: En el rollo suizo, dos puntos de capas contiguas están cerca en 3D, pero Isomap los coloca lejos porque sobre la hoja lo están.
hp: n_neighbors: vecinos con los que se construye el grafo | n_components: dimensión de la representación
py: sklearn.manifold.Isomap
rel: mds, manifold-learning, lle-y-pca-local, knn
en: Isomap

## t-sne
frase: Visualización en 2D que mantiene juntos a los vecinos cercanos y separa los grupos.
desc: Convierte las distancias en probabilidades de ser vecinos, tanto en el espacio original como en el reducido, y minimiza la divergencia de Kullback-Leibler entre ambas. En el espacio reducido usa una distribución t de Student, que separa bien los grupos. Conserva la estructura local, pero no la global: los tamaños de los grupos y las distancias entre ellos no significan nada.
cuando: Para visualizar datos de alta dimensión, como embeddings, e identificar grupos; no para alimentar otros modelos.
ej: Visualizar en 2D los embeddings de 10.000 imágenes de dígitos y ver diez nubes, una por dígito.
hp: perplexity: número efectivo de vecinos considerados; valores típicos entre 5 y 50 | learning_rate: tamaño de paso de la optimización
py: sklearn.manifold.TSNE
rel: umap, divergencia-de-kullback-leibler, embeddings, manifold-learning
en: t-SNE

## umap
frase: Reducción no lineal rápida que conserva la estructura local y más estructura global que t-SNE.
desc: Construye un grafo difuso de vecinos en el espacio original y busca una representación de baja dimensión cuyo grafo se parezca lo más posible. Es más rápido que t-SNE, escala a millones de puntos, conserva mejor la estructura global y puede transformar datos nuevos, así que también sirve como paso previo a otros modelos.
cuando: Para visualizar o reducir grandes conjuntos de datos de alta dimensión, también como preprocesado para clustering.
ej: Visualizar los perfiles de expresión génica de un millón de células y ver aparecer los tipos celulares como grupos.
hp: n_neighbors: tamaño de la vecindad; más alto, más estructura global | min_dist: lo juntos que pueden quedar los puntos en la proyección | metric: distancia usada en el espacio original
py: umap.UMAP
rel: t-sne, manifold-learning, clustering, embeddings
en: UMAP

## curvas-principales
frase: Una curva suave que pasa por el centro de los datos: la versión no lineal de la primera componente principal.
desc: Una curva principal es aquella en la que cada punto es la media de los datos que se proyectan sobre él. Generaliza la primera componente de PCA de una recta a una curva y permite describir datos que siguen una trayectoria curvada con una sola coordenada: la posición a lo largo de la curva. Existen extensiones secuenciales que encadenan varias curvas para obtener más dimensiones.
cuando: Para resumir con una sola coordenada datos que siguen una trayectoria curva, como un proceso de desarrollo o de degradación.
ej: Ordenar células según su avance en un proceso de diferenciación a partir de su expresión génica (pseudotiempo).
rel: pca, manifold-learning, lle-y-pca-local
en: principal curves

## clustering
frase: Agrupar los datos en conjuntos de elementos parecidos, sin etiquetas previas.
desc: El clustering busca grupos tales que sus elementos se parezcan entre sí más que a los de otros grupos. No hay una única definición de grupo: por centroides, por jerarquía, por densidad o por distribuciones de probabilidad, y cada familia descubre formas distintas. El resultado depende de la medida de similitud, del escalado y del número de grupos.
cuando: Para descubrir segmentos o estructura en datos sin etiquetar, como clientes, documentos o patrones de uso.
ej: Segmentar a los clientes por su comportamiento de compra para diseñar una campaña distinta para cada grupo.
py: sklearn.cluster
rel: k-means, clustering-jerarquico, dbscan, gmm, evaluacion-del-clustering
en: clustering

## clustering-basado-en-centroides
frase: Cada grupo se representa por un punto central y cada dato se asigna al centro más cercano.
desc: Definen K centros y asignan cada punto al más cercano, alternando la asignación y la actualización de los centros hasta converger. Son rápidos y escalan a muchos datos, pero hay que fijar K de antemano y tienden a encontrar grupos compactos y de tamaño similar. El centro puede ser la media (k-means), un dato real (k-medoids) o la mediana (k-median).
cuando: Con muchos datos y grupos aproximadamente esféricos y de tamaño parecido.
ej: Elegir la ubicación de 5 almacenes de forma que cada tienda quede cerca de uno.
rel: k-means, k-medoids-y-k-median, cuantizacion-vectorial, metodo-del-codo
en: centroid-based clustering

## k-means
frase: Divide los datos en K grupos minimizando la distancia de cada punto a la media de su grupo.
desc: Empieza con K centros, asigna cada punto al más cercano, recalcula cada centro como la media de sus puntos y repite hasta que nada cambia. Minimiza la suma de distancias al cuadrado dentro de los grupos. Es rápido y simple, pero depende de la inicialización (k-means++ la mejora), supone grupos esféricos de tamaño similar y es sensible a la escala y a los atípicos.
cuando: Como primera opción de clustering con muchos datos numéricos escalados y grupos compactos.
ej: Reducir los millones de colores de una foto a 16 representativos para comprimirla.
hp: n_clusters: número de grupos (K) | init: inicialización de los centros (k-means++ o aleatoria) | n_init: número de inicializaciones distintas; se queda con la mejor
py: sklearn.cluster.KMeans ; sklearn.cluster.MiniBatchKMeans
rel: clustering-basado-en-centroides, metodo-del-codo, gmm, cuantizacion-vectorial, escalado-de-variables
en: k-means

## k-medoids-y-k-median
frase: Variantes de k-means con centros más robustos: un dato real (medoide) o la mediana.
desc: k-medoids usa como centro de cada grupo un dato real, el que minimiza la distancia al resto, lo que permite usar cualquier medida de distancia y da representantes interpretables. k-median usa la mediana en cada dimensión y minimiza la distancia Manhattan. Ambos son más robustos a los atípicos que k-means, a cambio de más cálculo.
cuando: Con atípicos, con distancias no euclídeas o cuando necesitas que cada grupo esté representado por un caso real.
ej: Elegir 10 pacientes reales que representen los perfiles clínicos típicos de un hospital.
hp: n_clusters: número de grupos | metric: distancia entre puntos | method: algoritmo de búsqueda (alternate o pam)
py: sklearn_extra.cluster.KMedoids ; kmedoids.KMedoids
rel: k-means, distancia-manhattan, outliers, clustering-basado-en-centroides
en: k-medoids & k-median

## cuantizacion-vectorial
frase: Representar cada vector por el más parecido de un pequeño diccionario de prototipos.
desc: Aprende un libro de códigos con K vectores prototipo, normalmente con k-means, y sustituye cada dato por el índice de su prototipo más cercano. Comprime mucho y convierte datos continuos en símbolos discretos. Es la base de la compresión clásica de imagen y audio, y reaparece en modelos generativos modernos (VQ-VAE) y en la búsqueda vectorial aproximada (cuantización de producto).
cuando: Para comprimir vectores o convertir señales continuas en símbolos discretos.
ej: Comprimir una imagen guardando, para cada bloque de 4 × 4 píxeles, solo el número del bloque más parecido de un diccionario de 256.
py: scipy.cluster.vq.vq ; sklearn.cluster.KMeans
rel: k-means, busqueda-aproximada-de-vecinos, vae, clustering-basado-en-centroides
en: vector quantization (VQ)

## clustering-jerarquico
frase: Construye una jerarquía de grupos anidados uniendo, paso a paso, los dos más parecidos.
desc: La versión aglomerativa empieza con cada punto como un grupo y fusiona en cada paso los dos grupos más cercanos según un criterio de enlace, hasta tener uno solo. El resultado es un árbol (dendrograma) que se corta a la altura deseada para obtener los grupos, sin fijar K de antemano. Su coste crece rápidamente con el número de datos.
cuando: Con conjuntos de tamaño moderado, cuando interesa ver la estructura a varios niveles o no sabes cuántos grupos hay.
ej: Construir una taxonomía de los productos de un supermercado a partir de qué se compra junto, de las categorías amplias a las subcategorías.
hp: n_clusters: número de grupos al cortar el árbol | linkage: criterio para medir la distancia entre grupos | distance_threshold: altura a la que se corta, en lugar de fijar el número de grupos
py: sklearn.cluster.AgglomerativeClustering ; scipy.cluster.hierarchy.linkage
rel: dendrograma, criterios-de-enlace, clustering, k-means
en: hierarchical clustering

## dendrograma
frase: El árbol que muestra en qué orden y a qué distancia se fusionan los grupos.
desc: Cada hoja es un dato y cada unión indica la fusión de dos grupos, a una altura igual a la distancia entre ellos. Cortar el árbol con una línea horizontal da una partición: cuanto más bajo es el corte, más grupos salen. Los saltos grandes de altura sugieren un número natural de grupos.
ej: En un dendrograma de especies animales construido a partir de su ADN, los mamíferos se unen entre sí mucho antes de unirse con las aves.
py: scipy.cluster.hierarchy.dendrogram
rel: clustering-jerarquico, criterios-de-enlace, metodo-del-codo
en: dendrogram

## criterios-de-enlace
frase: Cómo se mide la distancia entre dos grupos para decidir cuáles se fusionan.
desc: El enlace simple usa la distancia entre los puntos más cercanos de cada grupo y produce cadenas alargadas. El completo usa los más lejanos y produce grupos compactos. El promedio usa la media de todas las distancias entre pares. El de Ward fusiona los grupos que menos aumentan la varianza interna y suele dar los grupos más equilibrados.
cuando: Ward, como opción por defecto con variables numéricas; el simple, cuando los grupos pueden tener formas alargadas.
ej: Con dos grupos unidos por un puente de puntos, el enlace simple los fusiona en uno; Ward los mantiene separados.
py: scipy.cluster.hierarchy.linkage
rel: clustering-jerarquico, dendrograma, distancia-y-similitud
en: linkage criteria (single, complete, average, Ward)

## clustering-basado-en-densidad
frase: Los grupos son regiones densas de puntos separadas por zonas vacías.
desc: En lugar de centros, define los grupos como zonas donde los puntos están muy juntos. Encuentra grupos de cualquier forma, no necesita saber cuántos hay y deja sin asignar los puntos aislados, que trata como ruido. Le cuesta cuando los grupos tienen densidades muy distintas.
cuando: Con grupos de formas irregulares, un número de grupos desconocido o presencia de ruido.
ej: Detectar las zonas de una ciudad con muchos avisos de averías, ignorando los avisos aislados.
rel: dbscan, mean-shift, kde, clustering
en: density-based clustering

## dbscan
frase: Agrupa los puntos con suficientes vecinos cercanos y marca como ruido los aislados.
desc: Un punto es central si tiene al menos un número mínimo de vecinos dentro de un radio ε. Los grupos se forman encadenando puntos centrales alcanzables entre sí, y los puntos que no pertenecen a ninguno quedan como ruido. Encuentra grupos de cualquier forma, pero ε y el mínimo de vecinos son difíciles de fijar si las densidades varían; HDBSCAN resuelve ese problema.
cuando: Para grupos de forma arbitraria con ruido, cuando no conoces el número de grupos.
ej: Agrupar las posiciones GPS de una flota de reparto para descubrir los lugares donde los vehículos se paran con frecuencia.
hp: eps: radio de la vecindad | min_samples: vecinos mínimos para que un punto sea central
py: sklearn.cluster.DBSCAN ; sklearn.cluster.HDBSCAN
rel: clustering-basado-en-densidad, mean-shift, deteccion-basada-en-distancia, knn
en: DBSCAN

## mean-shift
frase: Desplaza cada punto hacia la zona más densa cercana; los que acaban en la misma cima forman un grupo.
desc: Estima la densidad con un kernel y mueve iterativamente cada punto hacia la media de sus vecinos ponderada por el kernel, es decir, cuesta arriba en la densidad. Los puntos que convergen al mismo máximo (moda) forman un grupo. No necesita fijar el número de grupos, solo el ancho de banda, pero es costoso con muchos datos.
cuando: Para encontrar los modos de una distribución con una cantidad moderada de datos, como en segmentación de imágenes o seguimiento de objetos.
ej: Segmentar una imagen agrupando sus píxeles por color y posición en regiones homogéneas.
hp: bandwidth: anchura del kernel; más grande, menos grupos
py: sklearn.cluster.MeanShift ; sklearn.cluster.estimate_bandwidth
rel: kde, dbscan, clustering-basado-en-densidad, seguimiento-de-objetos
en: Mean Shift

## spectral-clustering
frase: Agrupa usando los autovectores de un grafo de similitud entre los puntos.
desc: Construye un grafo en el que los puntos parecidos están conectados y usa los autovectores de su matriz laplaciana para representar cada punto en pocas dimensiones, donde después aplica k-means. Así detecta grupos definidos por la conectividad, como anillos concéntricos, que k-means no separaría. Su coste crece mucho con el número de datos.
cuando: Con grupos no convexos o definidos por la conectividad y conjuntos de tamaño moderado.
ej: Separar dos anillos concéntricos de puntos, que k-means cortaría por la mitad.
hp: n_clusters: número de grupos | affinity: cómo se construye el grafo de similitud (kernel RBF o vecinos más cercanos) | gamma: anchura del kernel RBF
py: sklearn.cluster.SpectralClustering
rel: k-means, truco-del-kernel, clustering, isomap
en: spectral clustering

## gmm
frase: Modela los datos como una mezcla de gaussianas y da a cada punto una probabilidad de pertenecer a cada grupo.
desc: Supone que los datos proceden de K distribuciones normales, cada una con su media, su covarianza y su peso, y las estima con el algoritmo EM. A diferencia de k-means, la asignación es blanda (probabilidades) y los grupos pueden ser elípticos y de tamaños distintos. También sirve para estimar densidades y generar datos nuevos.
cuando: Cuando los grupos pueden ser elípticos o solaparse, o cuando necesitas probabilidades de pertenencia.
ej: Modelar las alturas de una población adulta como la mezcla de dos gaussianas, una por sexo, y obtener para cada persona la probabilidad de pertenecer a cada una.
hp: n_components: número de gaussianas | covariance_type: forma de las covarianzas (full, diag, tied o spherical)
py: sklearn.mixture.GaussianMixture ; sklearn.mixture.BayesianGaussianMixture
rel: k-means, estimacion-de-densidad, maxima-verosimilitud, deteccion-basada-en-probabilidad
en: Gaussian mixture models (GMM)

## metodo-del-codo
frase: Elegir el número de grupos donde añadir uno más deja de mejorar mucho el ajuste.
desc: Se ejecuta el clustering para varios valores de K y se representa la inercia (suma de distancias al cuadrado dentro de los grupos) frente a K. La curva siempre baja, pero a partir de cierto punto lo hace mucho más despacio, y ese codo sugiere un K razonable. Como el codo no siempre es claro, conviene contrastarlo con la silueta u otros índices y con el sentido del problema.
cuando: Para orientar la elección de K en k-means y métodos similares.
ej: La inercia cae mucho de 2 a 4 grupos de clientes y casi nada a partir de 5: se eligen 4 o 5 segmentos.
py: `KMeans(n_clusters=k).fit(X).inertia_`
rel: k-means, coeficiente-de-silueta, evaluacion-del-clustering, dendrograma
en: elbow method

## evaluacion-del-clustering
frase: Medir la calidad de unos grupos cuando no hay etiquetas con las que comparar.
desc: Sin etiquetas se usan índices internos, que premian grupos compactos y bien separados, como la silueta, Calinski-Harabasz o Davies-Bouldin. Si existen etiquetas de referencia, se usan índices externos, como el índice de Rand ajustado o la información mutua normalizada. Ningún índice sustituye la pregunta clave: si los grupos son útiles para el problema.
ej: Comparar segmentaciones de clientes con 3, 4 y 5 grupos usando la silueta y comprobar después con el equipo de marketing si los grupos tienen sentido.
py: sklearn.metrics.adjusted_rand_score ; sklearn.metrics.normalized_mutual_info_score
rel: coeficiente-de-silueta, indice-de-calinski-harabasz, indice-de-davies-bouldin, metodo-del-codo
en: clustering evaluation

## coeficiente-de-silueta
frase: Compara, para cada punto, la distancia a su propio grupo con la distancia al grupo vecino más cercano.
desc: Para cada punto calcula s = (b − a) / máx(a, b), donde a es la distancia media a los puntos de su grupo y b, la distancia media a los del grupo más cercano. Va de −1 a 1: cerca de 1, punto bien asignado; cerca de 0, en la frontera; negativo, probablemente en el grupo equivocado. La media de todos los puntos resume la calidad global y ayuda a elegir K.
cuando: Para elegir el número de grupos y detectar puntos mal asignados.
ej: Con K = 3 la silueta media es 0,62 y con K = 6 baja a 0,35: la partición en tres grupos es más clara.
py: sklearn.metrics.silhouette_score ; sklearn.metrics.silhouette_samples
rel: evaluacion-del-clustering, metodo-del-codo, k-means, indice-de-davies-bouldin
en: silhouette coefficient

## indice-de-calinski-harabasz
frase: Cociente entre la dispersión entre grupos y la dispersión dentro de ellos: cuanto más alto, mejor.
desc: Compara la dispersión de los centros de los grupos con la dispersión dentro de cada grupo, corregidas por el número de grupos y de puntos. Los valores altos indican grupos densos y bien separados. Es muy rápido de calcular, pero favorece los grupos convexos y esféricos, como los de k-means.
cuando: Para comparar rápidamente distintos valores de K con muchos datos.
ej: Calcular el índice para K entre 2 y 10 en una segmentación de un millón de clientes, donde la silueta sería demasiado lenta.
py: sklearn.metrics.calinski_harabasz_score
rel: evaluacion-del-clustering, coeficiente-de-silueta, indice-de-davies-bouldin
en: Calinski-Harabasz index

## indice-de-davies-bouldin
frase: Mide cuánto se parece cada grupo a su vecino más similar: cuanto más bajo, mejor.
desc: Para cada grupo calcula el peor cociente entre la suma de las dispersiones de dos grupos y la distancia entre sus centros, y promedia esos valores. Un valor cercano a cero indica grupos compactos y bien separados. Como Calinski-Harabasz, es rápido y favorece los grupos convexos.
cuando: Como índice complementario a la silueta para comparar particiones.
ej: Una segmentación con un índice de Davies-Bouldin de 0,4 es más clara que otra con 1,2.
py: sklearn.metrics.davies_bouldin_score
rel: evaluacion-del-clustering, coeficiente-de-silueta, indice-de-calinski-harabasz
en: Davies-Bouldin index

## estimacion-de-densidad
frase: Estimar, a partir de una muestra, la distribución de probabilidad de la que proceden los datos.
desc: El objetivo es conocer la densidad p(x): qué valores son frecuentes y cuáles raros. Los métodos paramétricos suponen una forma, como una gaussiana o una mezcla de gaussianas; los no paramétricos, como el histograma o la KDE, dejan que los datos dicten la forma. Una buena estimación permite detectar anomalías, generar datos nuevos y calcular probabilidades.
cuando: Para conocer la distribución de unos datos, detectar valores improbables o generar muestras parecidas.
ej: Estimar la distribución de la duración de las llamadas a un servicio de atención para detectar las anormalmente largas.
rel: histograma, kde, gmm, deteccion-basada-en-probabilidad, modelo-generativo-vs-discriminativo
en: density estimation

## histograma
frase: Cuenta cuántos datos caen en cada intervalo: la estimación de densidad más simple.
desc: Divide el rango de valores en intervalos y cuenta los datos de cada uno; normalizado, aproxima la densidad. Es simple e intuitivo, pero el resultado depende mucho de la anchura y la posición de los intervalos, es escalonado y escala mal a varias dimensiones. Reglas como las de Sturges o Freedman-Diaconis ayudan a elegir el número de intervalos.
cuando: Para una primera visión rápida de la distribución de una variable.
ej: El histograma de los tiempos de entrega de un servicio de reparto revela dos picos: entregas en el mismo día y al día siguiente.
hp: bins: número de intervalos, o regla para elegirlo
py: numpy.histogram ; numpy.histogram_bin_edges
rel: kde, estimacion-de-densidad, eda
en: histogram

## kde
frase: Estima la densidad colocando una pequeña campana sobre cada dato y sumándolas.
desc: La estimación kernel de densidad, o ventanas de Parzen, sitúa una función suave, normalmente una gaussiana, sobre cada observación y las promedia. El resultado es una curva continua y suave que no depende de la posición de unos intervalos. El ancho de banda controla el equilibrio: si es pequeño, la estimación es ruidosa; si es grande, borra la estructura.
cuando: Para estimar densidades suaves en pocas dimensiones, visualizar distribuciones o puntuar lo raro que es un dato.
ej: Estimar sobre un mapa la densidad de accidentes de tráfico para localizar los puntos negros de una carretera.
hp: bandwidth: anchura del kernel; controla la suavidad de la estimación | kernel: forma de la campana (gaussiana, Epanechnikov…)
py: sklearn.neighbors.KernelDensity ; scipy.stats.gaussian_kde
rel: histograma, estimacion-de-densidad, mean-shift, deteccion-basada-en-probabilidad
en: kernel density estimation (KDE, Parzen windows)
