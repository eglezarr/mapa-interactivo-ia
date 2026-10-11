# Fichas: IA generativa

## modelo-generativo-vs-discriminativo
frase: Un modelo discriminativo aprende a separar las clases; uno generativo, cómo se generan los datos.
desc: Un modelo discriminativo estima directamente P(y|x), la frontera entre clases, como hace la regresión logística. Uno generativo modela la distribución de los datos, P(x) o P(x|y), y por eso puede crear ejemplos nuevos, estimar lo probable que es un dato o clasificar aplicando Bayes, como Naive Bayes. Los discriminativos suelen clasificar mejor con muchos datos; los generativos permiten crear contenido y detectar lo inusual.
ej: Un modelo discriminativo distingue fotos de perros y de gatos; uno generativo puede además pintar un perro que no existe.
py: sklearn.naive_bayes.GaussianNB ; sklearn.linear_model.LogisticRegression
rel: teorema-de-bayes, densidad-explicita-vs-implicita, estimacion-de-densidad, regresion-logistica
en: generative vs discriminative models

## densidad-explicita-vs-implicita
frase: Algunos modelos generativos dan la probabilidad de cada dato; otros solo saben generar muestras.
desc: Los modelos de densidad explícita permiten calcular, de forma exacta o aproximada, la probabilidad de un dato: los autorregresivos y los flujos, de forma exacta; los VAE y la difusión, mediante cotas. Los de densidad implícita, como las GAN, solo aprenden a muestrear: generan ejemplos realistas, pero no dicen lo probable que es un dato concreto. Tener la densidad sirve para comparar modelos, detectar anomalías y comprimir.
ej: Un flujo normalizante puede indicar que una imagen es muy improbable y marcarla como anómala; una GAN no puede hacerlo.
rel: modelo-generativo-vs-discriminativo, normalizing-flows, gan, verosimilitud-vs-calidad-de-muestras
en: explicit vs implicit density

## muestreo-por-transformada-inversa
frase: Generar muestras de cualquier distribución transformando números uniformes con la inversa de su función de distribución.
desc: Si U es uniforme entre 0 y 1 y F es la función de distribución acumulada de la variable que se quiere simular, F⁻¹(U) sigue exactamente esa distribución. Es el método más básico para generar números aleatorios no uniformes y la idea de fondo de muchos modelos generativos: transformar ruido simple en datos complejos mediante una función, que en esos modelos se aprende.
cuando: Para simular variables cuya función de distribución se puede invertir, como la exponencial o una distribución empírica.
ej: Para simular tiempos entre llegadas exponenciales con una media de 2 minutos, se calcula −2·ln(1 − U) con U uniforme.
py: scipy.stats.rv_continuous.ppf ; numpy.random.Generator.uniform
rel: distribuciones-de-probabilidad, normalizing-flows, cambio-de-variable-y-jacobiano, transformaciones-de-distribucion
en: inverse transform sampling

## modelos-autorregresivos
frase: Generan los datos elemento a elemento, prediciendo cada uno a partir de los anteriores.
desc: Descomponen la probabilidad de una secuencia en un producto de probabilidades condicionadas y aprenden a predecir el siguiente elemento. Para generar, muestrean un elemento, lo añaden a la entrada y repiten. Dan la verosimilitud exacta y son la base de los LLM, aunque la generación es lenta porque no se puede paralelizar y los errores pueden acumularse.
cuando: Para generar secuencias (texto, audio, código) o modelar datos con un orden natural.
ej: Un LLM escribe su respuesta token a token, eligiendo cada uno en función de los anteriores.
py: transformers.AutoModelForCausalLM
rel: regla-de-la-cadena-probabilidad, gpt, prediccion-autorregresiva, perplejidad
en: autoregressive models

## regla-de-la-cadena-probabilidad
frase: Toda distribución conjunta se puede escribir como un producto de condicionales: P(x₁,…,xₙ) = ∏ P(xᵢ | x₁,…,xᵢ₋₁).
desc: Es una identidad exacta de la teoría de la probabilidad, no una aproximación. Convierte el problema, inabordable, de modelar directamente la distribución de secuencias enteras en una serie de problemas de clasificación: predecir el siguiente elemento dados los anteriores. Los modelos autorregresivos aprenden precisamente esas condicionales con una red compartida.
ej: P("el gato duerme") = P("el") · P("gato" | "el") · P("duerme" | "el gato").
rel: modelos-autorregresivos, probabilidad-condicional, clm, perplejidad
en: chain rule of probability

## generacion-con-rnn
frase: Usar una red recurrente para generar texto, música o señales paso a paso, alimentándola con su propia salida.
desc: La RNN se entrena para predecir el siguiente elemento de una secuencia. Para generar, se le da un comienzo, se muestrea el siguiente elemento de su distribución de salida, se añade a la entrada y se repite. La temperatura controla el equilibrio entre seguridad y creatividad. Fue el enfoque estándar de generación de texto antes de los transformers.
cuando: Para generar secuencias con modelos pequeños o en dispositivos con pocos recursos.
ej: Una LSTM entrenada carácter a carácter con obras de teatro genera diálogos con nombres de personajes y formato teatral convincentes.
py: keras.layers.LSTM ; `np.random.choice(len(probs), p=probs)`
rel: rnn, lstm, modelos-autorregresivos, decodificacion-y-temperatura
en: RNN-based generation

## pixelcnn
frase: Modelo autorregresivo que genera imágenes píxel a píxel con convoluciones enmascaradas.
desc: Ordena los píxeles de izquierda a derecha y de arriba abajo y predice la distribución de cada uno a partir de los anteriores. Para respetar ese orden usa convoluciones enmascaradas, que no ven los píxeles futuros y permiten entrenar en paralelo. Da verosimilitudes exactas y fue un referente en estimación de densidad de imágenes, pero genera muy despacio, píxel a píxel. Su idea reaparece como componente de otros modelos, como el VQ-VAE.
ej: Completar la mitad inferior de la foto de una cara a partir de la mitad superior, píxel a píxel.
rel: modelos-autorregresivos, convolucion, mascara-causal, verosimilitud-vs-calidad-de-muestras
en: PixelCNN

## perplejidad
frase: Entre cuántas opciones equiprobables duda, de media, un modelo de lenguaje en cada paso: cuanto menor, mejor.
desc: Es la exponencial de la entropía cruzada media por token sobre un texto de prueba. Una perplejidad de 20 significa que el modelo está tan inseguro como si eligiera entre 20 opciones igual de probables en cada paso. Sirve para comparar modelos de lenguaje con el mismo tokenizador, pero no mide directamente la utilidad ni la veracidad de lo que generan.
ej: Si la pérdida media de validación es de 2,3 nats por token, la perplejidad es e^2,3 ≈ 10.
py: `math.exp(perdida_media)`
rel: entropia-cruzada, nll, modelos-autorregresivos, evaluacion-de-llms
en: perplexity

## vae
frase: Autoencoder probabilístico cuyo espacio latente está organizado para poder generar datos nuevos muestreando en él.
desc: El codificador no produce un punto, sino una distribución gaussiana (media y varianza) sobre el espacio latente, y el decodificador reconstruye a partir de una muestra de ella. La pérdida combina el error de reconstrucción con una divergencia KL que acerca esas distribuciones a una normal estándar. Así el espacio latente queda continuo y bien rellenado, y basta con muestrear ruido normal y decodificarlo para generar datos nuevos. Sus muestras suelen ser algo borrosas.
cuando: Para generar datos con un espacio latente continuo e interpretable, o como compresor dentro de otros generativos, como la difusión latente.
ej: Un VAE entrenado con moléculas propone candidatos a fármaco moviéndose por el espacio latente hacia regiones con la propiedad deseada.
py: torch.distributions.Normal ; torch.distributions.kl_divergence
rel: elbo, truco-de-reparametrizacion, autoencoders, espacio-latente, stable-diffusion
en: variational autoencoder (VAE)

## elbo
frase: La cota inferior de la verosimilitud que se maximiza al entrenar un VAE: reconstruir bien sin alejarse del prior.
desc: La verosimilitud de un modelo con variables latentes no se puede calcular directamente. El ELBO (evidence lower bound) es una cota inferior manejable: la reconstrucción esperada menos la divergencia KL entre la distribución que da el codificador y el prior. Maximizarlo equilibra la fidelidad de la reconstrucción y la regularidad del espacio latente; dar más peso al término KL (β-VAE) favorece latentes más separables.
ej: Si se elimina el término KL, el VAE se convierte en un autoencoder normal y muestrear del prior deja de generar datos válidos.
py: torch.distributions.kl_divergence
rel: vae, divergencia-de-kullback-leibler, maxima-verosimilitud, truco-de-reparametrizacion
en: ELBO (evidence lower bound)

## truco-de-reparametrizacion
frase: Escribir la muestra aleatoria como media más desviación por ruido para poder retropropagar a través de ella.
desc: Muestrear z de una normal con media μ y desviación σ es una operación aleatoria que no deja pasar el gradiente. El truco la reescribe como z = μ + σ·ε, con ε una normal estándar independiente de los parámetros: el azar queda fuera de la red y el gradiente fluye hacia μ y σ. Es lo que permite entrenar un VAE con descenso por gradiente.
ej: En lugar de "muestrear z", el codificador produce μ y σ, se genera ε aparte y se calcula μ + σ·ε como un cálculo determinista más.
py: torch.distributions.Normal.rsample
rel: vae, elbo, backpropagation
en: reparameterization trick

## gan
frase: Dos redes que compiten: un generador que crea datos falsos y un discriminador que intenta detectarlos.
desc: El generador transforma ruido aleatorio en datos y el discriminador clasifica cada muestra como real o generada. Se entrenan por turnos: el discriminador mejora detectando falsificaciones y el generador mejora engañándolo, hasta que sus muestras son indistinguibles de las reales. Producen imágenes muy nítidas, pero su entrenamiento es inestable y propenso al colapso de modos; la difusión las ha superado en la mayoría de tareas.
cuando: Para generar imágenes realistas en un solo paso, muy rápido, o para traducir imágenes entre dominios.
ej: Generar caras de personas que no existen, indistinguibles de fotos reales a simple vista.
rel: generador-y-discriminador, colapso-de-modos, dcgan, densidad-explicita-vs-implicita, modelos-de-difusion
en: generative adversarial networks (GAN)

## generador-y-discriminador
frase: El falsificador y el detective: un juego minimax en el que cada red mejora a costa de la otra.
desc: El discriminador D maximiza la probabilidad de acertar si una muestra es real o falsa; el generador G minimiza esa misma cantidad, es decir, intenta que D se equivoque. En el equilibrio teórico, G reproduce la distribución de los datos y D no puede hacer nada mejor que responder 0,5. En la práctica, G suele entrenarse con una variante no saturante, maximizar log D(G(z)), porque da gradientes útiles cuando D todavía gana con facilidad.
ej: Al principio, D distingue sin esfuerzo el ruido de las fotos; poco a poco, G aprende bordes, texturas y, finalmente, caras completas.
rel: gan, colapso-de-modos, wasserstein-gan, entropia-cruzada
en: generator & discriminator (minimax game)

## colapso-de-modos
frase: El generador produce solo unos pocos tipos de muestras y deja sin cubrir el resto de la variedad de los datos.
desc: Si el generador encuentra una salida que engaña al discriminador, puede limitarse a variaciones de ella e ignorar el resto de la distribución: las muestras son realistas, pero poco diversas. Es el fallo típico de las GAN. Lo mitigan pérdidas como la de Wasserstein, la discriminación por minilotes, el entrenamiento progresivo y la regularización del discriminador.
ej: Una GAN de dígitos que genera unos y sietes perfectos, pero nunca un ocho.
rel: gan, mode-drop-y-cobertura, wasserstein-gan, colapso-de-la-representacion
en: mode collapse

## dcgan
frase: La primera receta estable de GAN convolucional, que sentó las bases de las GAN de imágenes.
desc: La Deep Convolutional GAN (2015) fijó pautas de arquitectura que hicieron las GAN mucho más estables: convoluciones con stride en lugar de pooling, convoluciones transpuestas en el generador, batch normalization, ReLU en el generador, Leaky ReLU en el discriminador y ninguna capa densa oculta. Mostró además que su espacio latente tiene estructura aritmética.
cuando: Como punto de partida para generar imágenes pequeñas o para aprender cómo se construye una GAN.
ej: Una DCGAN entrenada con fotos de dormitorios genera dormitorios nuevos de 64 × 64 píxeles.
py: keras.layers.Conv2DTranspose ; keras.layers.LeakyReLU ; keras.layers.BatchNormalization
rel: gan, convolucion-transpuesta, capas-de-normalizacion, aritmetica-de-embeddings
en: DCGAN

## wasserstein-gan
frase: GAN que mide la distancia entre distribuciones con la distancia de Wasserstein: más estable de entrenar.
desc: Sustituye la clasificación entre real y falso por una estimación de la distancia de Wasserstein, o del transportista: cuánta masa de probabilidad hay que mover, y a qué distancia, para convertir la distribución generada en la real. El discriminador, llamado crítico, da una puntuación sin sigmoide y debe ser 1-Lipschitz, lo que se impone recortando sus pesos o, mejor, con una penalización del gradiente (WGAN-GP). Da gradientes útiles aunque las distribuciones no se solapen, y su pérdida acompaña a la calidad de las muestras.
cuando: Cuando una GAN estándar es inestable o sufre colapso de modos.
ej: Al pasar a WGAN-GP, una GAN que oscilaba sin converger entrena de forma estable y la pérdida del crítico baja a medida que mejoran las imágenes.
py: torch.autograd.grad
rel: gan, colapso-de-modos, generador-y-discriminador, divergencia-de-kullback-leibler
en: Wasserstein GAN (WGAN)

## otras-variantes-de-gan
frase: Las mejoras que llevaron a las GAN a la alta resolución y al fotorrealismo.
desc: ProGAN entrena aumentando la resolución poco a poco. StyleGAN inyecta el código latente como un estilo en cada capa, lo que controla por separado rasgos gruesos (la pose) y finos (pecas, pelo), y genera caras fotorrealistas. BigGAN escala el tamaño del modelo y del lote para generar por clases. SAGAN añade autoatención, y LSGAN usa una pérdida cuadrática más estable.
ej: StyleGAN genera retratos de 1024 × 1024 indistinguibles de fotos y permite combinar la pose de una cara con la textura de otra.
rel: gan, dcgan, wasserstein-gan, gan-condicional
en: other GAN variants (ProGAN, StyleGAN, BigGAN)

## gan-condicional
frase: GAN que genera según una condición dada, como una clase, un texto o una imagen.
desc: Tanto el generador como el discriminador reciben la condición, por ejemplo una etiqueta convertida en embedding o una imagen de entrada. El generador aprende a producir muestras coherentes con ella y el discriminador, a rechazar las que no lo son. Así se controla qué se genera: un dígito concreto, una clase de objeto o la traducción de una imagen.
cuando: Cuando se quiere controlar lo que se genera, en lugar de obtener muestras al azar.
ej: Pedir a una GAN condicional entrenada con dígitos que genere cien variantes distintas del número 7.
rel: gan, pix2pix, guia-difusion, capa-de-embedding
en: conditional GAN (cGAN)

## pix2pix
frase: GAN condicional que traduce una imagen en otra aprendiendo de pares de ejemplos alineados.
desc: Aprende la correspondencia entre dos dominios a partir de pares de imágenes. El generador es una U-Net y el discriminador, un PatchGAN que juzga si cada parche de la imagen es realista. La pérdida combina el término adversario con un término L1 que mantiene la salida cerca de la imagen objetivo. Sirve para cualquier traducción de la que haya pares disponibles.
cuando: Para traducir imágenes de un dominio a otro cuando se dispone de pares alineados.
ej: Convertir bocetos de bolsos en fotos realistas, o mapas en vistas aéreas.
rel: gan-condicional, cyclegan, u-net, superresolucion-inpainting-y-colorizacion
en: Pix2Pix

## cyclegan
frase: Traduce imágenes entre dos dominios sin necesitar pares, exigiendo que la ida y la vuelta devuelvan la original.
desc: Entrena dos generadores, de A a B y de B a A, cada uno con su discriminador. Como no hay pares, añade una pérdida de consistencia cíclica: traducir una imagen a B y devolverla a A debe recuperar la original, lo que impide que el generador invente contenido sin relación con la entrada. Permite traducir estilos con dos colecciones de fotos sin correspondencia.
cuando: Para traducir entre dominios de imagen cuando no hay pares alineados.
ej: Convertir fotos de caballos en cebras, o paisajes de verano en paisajes nevados.
rel: pix2pix, gan, transferencia-de-estilo-neuronal, adaptacion-de-dominio
en: CycleGAN

## srgan
frase: GAN para superresolución que recupera texturas realistas en lugar de imágenes borrosas.
desc: Las redes de superresolución entrenadas con error cuadrático producen resultados suaves y borrosos, porque promedian todas las texturas posibles. SRGAN combina una pérdida perceptual, calculada con las características de una red VGG, con una pérdida adversaria que empuja hacia imágenes naturales. Sus sucesoras, como ESRGAN, mejoraron la nitidez y redujeron los artefactos.
cuando: Para ampliar imágenes cuando importa la apariencia realista más que la fidelidad exacta píxel a píxel.
ej: Ampliar cuatro veces fotos antiguas de baja resolución y obtener texturas creíbles de piel y de tela.
rel: superresolucion-inpainting-y-colorizacion, perdidas-perceptuales, gan, psnr-y-ssim
en: SRGAN

## normalizing-flows
frase: Transforman una distribución simple en otra compleja con una cadena de funciones invertibles, y dan la densidad exacta.
desc: Parten de una distribución sencilla, normalmente una gaussiana, y le aplican una secuencia de transformaciones invertibles y diferenciables aprendidas. Gracias a la fórmula del cambio de variable, la densidad de cualquier dato se calcula exactamente, y para generar basta con muestrear la gaussiana y aplicar la cadena. La exigencia de invertibilidad limita las arquitecturas y suele dar muestras de menor calidad que la difusión.
cuando: Cuando se necesita la densidad exacta además de generar: detección de anomalías, inferencia bayesiana o modelado de variables físicas.
ej: Un flujo entrenado con los datos de un detector de partículas calcula lo probable que es cada colisión y genera colisiones simuladas.
py: nflows.flows.base.Flow ; torch.distributions.TransformedDistribution
rel: cambio-de-variable-y-jacobiano, flujos-autorregresivos-y-de-acoplamiento, rbig, densidad-explicita-vs-implicita
en: normalizing flows

## cambio-de-variable-y-jacobiano
frase: La fórmula que dice cómo cambia una densidad al transformar sus variables: se corrige con el determinante del jacobiano.
desc: Si x = f(z) con f invertible, la densidad de x es la de z en f⁻¹(x) multiplicada por el valor absoluto del determinante del jacobiano de f⁻¹. Ese determinante mide cuánto estira o comprime la transformación el volumen alrededor de cada punto. Los flujos se diseñan para que sea barato de calcular, por ejemplo con jacobianos triangulares.
ej: Si se multiplican por 2 todos los valores de una variable, su probabilidad se reparte en el doble de espacio y la densidad se divide por 2.
py: torch.autograd.functional.jacobian ; torch.distributions.TransformedDistribution
rel: normalizing-flows, muestreo-por-transformada-inversa, flujos-autorregresivos-y-de-acoplamiento
en: change of variables & Jacobian

## flujos-autorregresivos-y-de-acoplamiento
frase: Transformaciones invertibles con jacobiano triangular, cuyo determinante es barato: MAF, RealNVP y Glow.
desc: En una capa de acoplamiento (RealNVP, Glow), la mitad de las variables pasa sin cambios y determina una transformación afín de la otra mitad: invertirla es inmediato y el determinante es el producto de las escalas. En los flujos autorregresivos (MAF), cada variable se transforma en función de las anteriores, lo que da más flexibilidad pero hace lento uno de los dos sentidos. Glow añade convoluciones de 1 × 1 invertibles para mezclar los canales.
cuando: Para construir flujos normalizantes expresivos sobre datos de alta dimensión.
ej: Glow genera caras de alta resolución y permite modificar atributos como la sonrisa moviéndose por su espacio latente.
hp: hidden_features: tamaño de las redes que calculan cada transformación | num_blocks: número de bloques de cada red interna
py: nflows.transforms.autoregressive.MaskedAffineAutoregressiveTransform ; nflows.transforms.coupling.AffineCouplingTransform
rel: normalizing-flows, cambio-de-variable-y-jacobiano, modelos-autorregresivos, descuantizacion
en: autoregressive & coupling flows (MAF, RealNVP, Glow)

## descuantizacion
frase: Añadir ruido a datos discretos, como píxeles enteros, para poder modelarlos con densidades continuas.
desc: Los píxeles toman valores enteros de 0 a 255, y un modelo de densidad continua podría asignarles una densidad infinita concentrando toda la probabilidad en esos puntos. Para evitarlo se suma a cada valor un ruido uniforme en [0, 1), de modo que la densidad continua aprendida da una cota inferior válida de la verosimilitud discreta. La descuantización variacional aprende un ruido más ajustado y mejora la cota.
cuando: Al entrenar flujos u otros modelos de densidad continua con datos discretos, como imágenes o audio cuantizado.
ej: Un píxel de valor 128 pasa a ser un número entre 128 y 129 antes de entrenar el flujo.
py: `x = (x + torch.rand_like(x)) / 256`
rel: normalizing-flows, flujos-autorregresivos-y-de-acoplamiento, verosimilitud-vs-calidad-de-muestras
en: dequantization

## rbig
frase: Gaussianizar datos aplicando repetidamente una gaussianización por variable seguida de una rotación.
desc: La gaussianización iterativa basada en rotaciones (RBIG) transforma cada variable por separado en una gaussiana y después rota los datos, por ejemplo con PCA, para mezclarlas; repite hasta que el conjunto es una gaussiana multivariante. Es un flujo normalizante que no se entrena por gradiente: da la densidad de los datos y permite estimar medidas de información, como la correlación total o la información mutua, en muchas dimensiones.
cuando: Para estimar densidades y medidas de información (entropía, correlación total, información mutua) en datos multivariantes.
ej: Medir cuánta información comparten las variables climáticas de distintas regiones del planeta gaussianizándolas con RBIG.
rel: normalizing-flows, correlacion-total, informacion-mutua, pca
en: rotation-based iterative Gaussianization (RBIG)

## flow-matching
frase: Entrenar un campo de velocidades que transporta el ruido hasta los datos por trayectorias simples.
desc: En lugar de una cadena de transformaciones discretas, aprende un campo vectorial que, integrado como una ecuación diferencial, lleva muestras de ruido hasta los datos. Se entrena con una simple regresión: en puntos intermedios entre una muestra de ruido y un dato, la red debe predecir la velocidad de la trayectoria que los une. Generaliza la difusión, es estable y sencillo de entrenar, y lo usan modelos de imagen recientes como Stable Diffusion 3 y Flux.
cuando: Para entrenar generativos de imagen, audio o moléculas con la calidad de la difusión y trayectorias de muestreo más rectas.
ej: Un modelo de texto a imagen basado en flow matching genera una imagen de calidad con unas 30 integraciones del campo de velocidades.
py: diffusers.FlowMatchEulerDiscreteScheduler
rel: rectified-flow, modelos-de-difusion, normalizing-flows, stable-diffusion
en: flow matching

## rectified-flow
frase: Variante de flow matching con trayectorias rectas, que pueden enderezarse aún más para generar en muy pocos pasos.
desc: Usa como trayectoria la línea recta entre cada muestra de ruido y cada dato, recorrida a velocidad constante. El reflujo (ReFlow) reentrena el modelo con los pares de ruido e imagen que él mismo genera, lo que hace las trayectorias todavía más rectas, de modo que bastan pocos pasos de integración, o incluso uno, para generar. Es la base de modelos como Stable Diffusion 3.
cuando: Cuando se busca generar con calidad en muy pocos pasos de muestreo.
ej: Tras varias rondas de reflujo, un modelo de imagen pasa de necesitar 50 pasos a generar en 1 o 2.
py: diffusers.StableDiffusion3Pipeline ; diffusers.FlowMatchEulerDiscreteScheduler
rel: flow-matching, destilacion-de-difusion, modelos-de-difusion
en: rectified flow (ReFlow)

## modelos-de-difusion
frase: Aprenden a eliminar ruido paso a paso: generan partiendo de ruido puro y limpiándolo gradualmente.
desc: Durante el entrenamiento se corrompen los datos con ruido creciente y una red aprende a predecir y eliminar ese ruido en cualquier nivel. Para generar, se parte de ruido gaussiano y se aplica la red repetidamente hasta obtener una muestra limpia. Son estables de entrenar, cubren bien la diversidad de los datos y logran la mejor calidad en imagen, vídeo y audio, a costa de un muestreo de muchos pasos.
cuando: Para generar o editar imágenes, vídeo o audio de alta calidad, con condiciones como un texto o una imagen de referencia.
ej: Generar una ilustración a partir de la descripción "un faro en un acantilado al atardecer, estilo acuarela".
py: diffusers.DiffusionPipeline ; diffusers.UNet2DModel
rel: proceso-directo-e-inverso, ddpm, stable-diffusion, el-denoiser-como-prior-implicito, gan
en: diffusion models

## proceso-directo-e-inverso
frase: El directo añade ruido a los datos hasta destruirlos; el inverso, aprendido, recorre el camino de vuelta.
desc: El proceso directo es fijo: en cada paso añade un poco de ruido gaussiano, de modo que tras muchos pasos solo queda ruido puro, y permite saltar directamente a cualquier nivel de ruido. El proceso inverso es el que se aprende: en cada paso, una red estima el ruido presente o la imagen limpia, y con esa estimación se avanza hacia datos menos ruidosos.
ej: La foto de un gato pasa por 1.000 pasos de ruido hasta convertirse en estática; el modelo aprende a deshacer cada uno de esos pasos.
py: diffusers.DDPMScheduler
rel: modelos-de-difusion, ddpm, planificacion-del-ruido, autoencoder-denoising
en: forward & reverse diffusion processes

## el-denoiser-como-prior-implicito
frase: Un buen eliminador de ruido contiene, de forma implícita, todo lo que sabe sobre cómo son las imágenes.
desc: Para eliminar ruido de forma óptima hay que saber qué imágenes son probables. De hecho, el residuo de un eliminador de ruido óptimo es proporcional al gradiente del logaritmo de la densidad de las imágenes ruidosas, la llamada puntuación (score). Por eso un mismo eliminador de ruido, entrenado una sola vez, sirve para generar imágenes y para resolver problemas inversos, como quitar desenfoque, rellenar huecos o aumentar la resolución, sin reentrenar para cada tarea.
ej: Un eliminador de ruido entrenado solo para quitar ruido gaussiano reconstruye una foto a partir del 10 % de sus píxeles, guiado por los valores conocidos.
rel: modelos-de-difusion, autoencoder-denoising, superresolucion-inpainting-y-colorizacion, prior-verosimilitud-y-posterior
en: the denoiser as an implicit prior

## ddpm
frase: El modelo de difusión clásico: una red predice el ruido añadido y se generan muestras en cientos o miles de pasos.
desc: Los Denoising Diffusion Probabilistic Models (2020) definieron la receta moderna: una U-Net aprende a predecir el ruido que se añadió a una imagen en el paso t, con un simple error cuadrático. Para generar se recorren hacia atrás todos los pasos del proceso, típicamente 1.000, añadiendo un poco de ruido nuevo en cada uno. La calidad es excelente, pero el muestreo es lento.
cuando: Como formulación de referencia para entrenar modelos de difusión.
ej: Un DDPM entrenado con fotos de iglesias genera iglesias nuevas recorriendo 1.000 pasos de eliminación de ruido.
hp: num_train_timesteps: número de pasos de ruido del proceso (1.000 por defecto) | beta_schedule: cómo crece el ruido a lo largo de los pasos
py: diffusers.DDPMScheduler ; diffusers.DDPMPipeline
rel: modelos-de-difusion, ddim, planificacion-del-ruido, u-net
en: DDPM (denoising diffusion probabilistic models)

## ddim
frase: Muestreo de difusión más rápido y determinista: genera en decenas de pasos con el mismo modelo entrenado.
desc: Los Denoising Diffusion Implicit Models reformulan el proceso inverso para saltar pasos sin reentrenar: un modelo entrenado como DDPM puede generar en 20 o 50 pasos en lugar de 1.000. Con su parámetro η = 0, el muestreo es determinista: el mismo ruido inicial da siempre la misma imagen, lo que permite interpolar en el ruido y recuperar el ruido de una imagen real (inversión), algo muy útil para editar.
cuando: Para generar más rápido con un modelo de difusión ya entrenado o para editar imágenes reales mediante inversión.
ej: Una imagen que con DDPM requería 1.000 pasos se genera con DDIM en 50, con una calidad similar.
py: diffusers.DDIMScheduler
rel: ddpm, destilacion-de-difusion, modelos-de-difusion, planificacion-del-ruido
en: DDIM (denoising diffusion implicit models)

## planificacion-del-ruido
frase: Cómo se reparte el ruido a lo largo de los pasos de la difusión: un detalle que cambia mucho la calidad.
desc: La planificación fija cuánto ruido hay en cada paso del proceso. La lineal original destruye la información demasiado pronto en imágenes pequeñas; la de coseno la degrada de forma más gradual y mejora los resultados. También se ajusta al muestrear: qué pasos se recorren y con qué espaciado. Debe adaptarse a la resolución, porque las imágenes grandes necesitan más ruido para destruir la misma información.
ej: Cambiar la planificación lineal por la de coseno mejora un modelo de difusión de imágenes de 64 × 64 sin tocar nada más.
hp: beta_schedule: forma de la planificación (lineal, coseno…) | beta_start / beta_end: ruido en el primer y en el último paso
py: diffusers.DDPMScheduler
rel: ddpm, ddim, proceso-directo-e-inverso, lr-schedulers
en: noise schedule

## guia-difusion
frase: Técnicas para que la generación se ciña más a la condición, como el texto, a cambio de algo de diversidad.
desc: La guía por clasificador usa el gradiente de un clasificador entrenado con imágenes ruidosas para empujar la muestra hacia la clase deseada. La guía sin clasificador (classifier-free guidance), la estándar hoy, entrena un mismo modelo con y sin condición y, al generar, extrapola desde la predicción sin condición hacia la condicionada. La escala de guía controla el compromiso: más alta, más fidelidad al texto y menos variedad.
cuando: Siempre que se genera con condiciones, como un texto o una clase, para controlar cuánto se respeta la condición.
ej: Con una escala de guía de 1, las imágenes de "un perro con sombrero" a veces olvidan el sombrero; con 7,5, casi siempre aparece.
py: diffusers.StableDiffusionPipeline ; `pipe(prompt, guidance_scale=7.5)`
rel: stable-diffusion, gan-condicional, modelos-de-difusion, decodificacion-y-temperatura
en: classifier & classifier-free guidance

## stable-diffusion
frase: Difusión en el espacio latente comprimido de un autoencoder, guiada por texto: generación de imágenes al alcance de todos.
desc: En lugar de eliminar ruido sobre los píxeles, un autoencoder comprime la imagen en un espacio latente unas 48 veces más pequeño y la difusión ocurre ahí, lo que abarata enormemente el cálculo. El texto se codifica con un modelo como CLIP y se inyecta mediante atención cruzada en la red que elimina el ruido. Al final, el decodificador devuelve la imagen a píxeles. Su publicación en abierto, en 2022, popularizó la generación de imágenes.
cuando: Para generar o editar imágenes a partir de texto en hardware de consumo, o como base para ajustes y extensiones.
ej: Generar en unos segundos, con una GPU de consumo, una imagen de 512 × 512 a partir de la frase "una ciudad flotante entre las nubes".
py: diffusers.StableDiffusionPipeline ; diffusers.AutoencoderKL
rel: modelos-de-difusion, espacio-latente, atencion-cruzada, clip, texto-a-imagen
en: latent diffusion (Stable Diffusion)

## controlnet
frase: Añade a un modelo de difusión preentrenado un control preciso mediante una imagen guía: bordes, pose o profundidad.
desc: Duplica la parte codificadora de la red de difusión en una copia entrenable que recibe la imagen de control, y conecta ambas mediante convoluciones inicializadas a cero, de modo que al principio no altera el modelo original. Así aprende a respetar la estructura de la guía sin dañar lo que el modelo ya sabe y con relativamente pocos datos. Permite fijar la composición y dejar que el texto decida el contenido y el estilo.
cuando: Cuando se necesita controlar la composición, la pose o la estructura de la imagen generada, no solo su contenido.
ej: A partir del esqueleto de una pose de baile y el texto "astronauta en la Luna", generar un astronauta exactamente en esa postura.
py: diffusers.ControlNetModel ; diffusers.StableDiffusionControlNetPipeline
rel: stable-diffusion, guia-difusion, estimacion-de-pose-y-profundidad, fine-tuning
en: ControlNet

## dreambooth
frase: Ajustar un modelo de texto a imagen con unas pocas fotos para que aprenda un sujeto concreto.
desc: Con entre 3 y 5 imágenes de un sujeto, como una persona, una mascota o un producto, se ajusta el modelo asociándolo a un identificador poco común, por ejemplo "un perro sks". Una pérdida de preservación, que mezcla imágenes genéricas de la clase, evita que el modelo olvide cómo son los perros en general. Después se puede generar el sujeto en escenas nuevas. Suele combinarse con LoRA para ajustar muy pocos parámetros.
cuando: Para personalizar un generador con un sujeto concreto a partir de pocas fotos.
ej: Con 5 fotos de una mochila, generar anuncios con esa misma mochila en la cima de una montaña o en una playa.
rel: stable-diffusion, fine-tuning, lora, olvido-catastrofico
en: DreamBooth

## destilacion-de-difusion
frase: Comprimir los muchos pasos de la difusión en unos pocos, o en uno solo, entrenando un modelo alumno.
desc: El muestreo con decenas de pasos es el gran coste de la difusión. La destilación entrena un modelo que imita en pocos pasos lo que el original hace en muchos. Los modelos de consistencia aprenden a llevar cualquier punto de una trayectoria directamente a su imagen final; la destilación adversaria, usada en SD Turbo, añade un discriminador para mantener la calidad en 1 a 4 pasos. Permite generar en tiempo real.
cuando: Cuando la velocidad de generación es crítica: aplicaciones interactivas, tiempo real o dispositivos con poca potencia.
ej: SD Turbo genera una imagen en un solo paso, lo bastante rápido como para actualizarla mientras el usuario escribe la descripción.
py: diffusers.LCMScheduler ; diffusers.AutoPipelineForText2Image
rel: destilacion-de-conocimiento, ddim, rectified-flow, stable-diffusion
en: diffusion distillation (consistency models, SD Turbo)

## difusion-de-video
frase: Extender la difusión al tiempo para generar vídeos coherentes fotograma a fotograma.
desc: Se añade una dimensión temporal a la red que elimina el ruido, con atención o convoluciones entre fotogramas, para que los objetos se muevan de forma coherente y no parpadeen. Los modelos actuales usan transformers de difusión sobre parches espacio-temporales de un espacio latente comprimido. Pueden generar a partir de texto, de una imagen inicial o de otro vídeo; el gran reto es mantener la coherencia en duraciones largas.
cuando: Para generar clips a partir de texto o imágenes, animar una imagen fija o editar vídeos existentes.
ej: Animar la foto de un paisaje para que el agua de la cascada fluya y las nubes se desplacen.
py: diffusers.CogVideoXPipeline ; diffusers.WanPipeline
rel: modelos-de-difusion, generacion-de-video, stable-diffusion, atencion-y-transformers
en: video diffusion

## transferencia-de-estilo-neuronal
frase: Pintar el contenido de una imagen con el estilo de otra usando las características de una red preentrenada.
desc: Se optimiza una imagen para que sus activaciones profundas en una red como VGG se parezcan a las de la imagen de contenido, y sus correlaciones entre canales (matrices de Gram), a las de la imagen de estilo. Las matrices de Gram capturan texturas y pinceladas independientemente de su posición. El método original optimiza cada imagen por separado; después aparecieron redes que aplican un estilo en un solo paso.
cuando: Para efectos artísticos o para aumentar datos cambiando la apariencia y conservando la estructura.
ej: Convertir una foto de tu ciudad en un cuadro con las pinceladas de "La noche estrellada".
py: keras.applications.VGG19 ; `torch.einsum("bchw,bdhw->bcd", f, f)`
rel: perdidas-perceptuales, cyclegan, filtros-y-mapas-de-caracteristicas, visualizacion-de-filtros-y-activaciones
en: neural style transfer

## superresolucion-inpainting-y-colorizacion
frase: Restaurar imágenes: aumentar su resolución, rellenar las zonas que faltan o dar color a fotos en blanco y negro.
desc: Son problemas inversos: hay muchas soluciones posibles y el modelo debe elegir una plausible usando lo que sabe de cómo son las imágenes. La superresolución inventa detalle fino; el inpainting rellena regiones borradas o dañadas de forma coherente con el resto; la colorización asigna colores creíbles. Se abordan con U-Net, GAN y, sobre todo hoy, con modelos de difusión.
cuando: Para restaurar fotos antiguas o dañadas, ampliar imágenes o eliminar objetos no deseados.
ej: Borrar a un turista del fondo de una foto de vacaciones y rellenar el hueco con el paisaje que había detrás.
py: diffusers.StableDiffusionInpaintPipeline ; diffusers.StableDiffusionUpscalePipeline
rel: srgan, el-denoiser-como-prior-implicito, u-net, pix2pix
en: super-resolution, inpainting & colorization

## datos-sinteticos
frase: Datos artificiales generados por modelos que imitan las propiedades de los reales.
desc: Se usan cuando los datos reales son escasos, caros, desbalanceados o privados. Pueden ser tablas que conservan las distribuciones y correlaciones de las originales sin copiar a ninguna persona, imágenes simuladas o generadas, o textos creados por un LLM para entrenar otros modelos. Hay que validar que sean útiles (que un modelo entrenado con ellos funcione con datos reales) y que no filtren información privada.
cuando: Con datos escasos, sensibles o desbalanceados, para compartir datos sin exponer información personal o para cubrir casos raros.
ej: Un banco genera transacciones sintéticas con las mismas pautas estadísticas que las reales para que un proveedor externo desarrolle su modelo sin acceder a datos de clientes.
py: sdv.single_table.GaussianCopulaSynthesizer ; sdv.single_table.CTGANSynthesizer
rel: data-augmentation, adaptacion-de-dominio, desbalanceo-de-clases, decisiones-automatizadas-y-rgpd
en: synthetic data

## evaluacion-de-modelos-generativos
frase: Medir si lo generado es realista, diverso y fiel a lo pedido: mucho más difícil que medir un acierto.
desc: No hay una respuesta correcta con la que comparar, así que se combinan varias miradas: la inspección humana, métricas que comparan las distribuciones de muestras reales y generadas en el espacio de características de una red (IS, FID, KID), medidas separadas de calidad y diversidad (precisión y cobertura) y, cuando existe, la verosimilitud. Ninguna basta por sí sola.
ej: Dos modelos de imagen con un FID parecido pueden fallar de forma distinta: uno genera manos deformes y el otro repite siempre los mismos fondos.
rel: evaluacion-cualitativa, fid-y-kid, mode-drop-y-cobertura, verosimilitud-vs-calidad-de-muestras
en: evaluating generative models

## evaluacion-cualitativa
frase: Mirar las muestras: el juicio humano sigue siendo imprescindible para evaluar lo generado.
desc: Inspeccionar muestras, compararlas lado a lado y pedir a personas que elijan la mejor, o que distingan lo real de lo generado, revela defectos que las métricas no ven: anatomías imposibles, texto ilegible, incoherencias o falta de fidelidad a la instrucción. Para que sea fiable conviene usar muestras al azar y no las mejores, varios evaluadores y comparaciones a ciegas.
cuando: Siempre, como complemento de las métricas automáticas, y sobre todo antes de decidir entre modelos.
ej: Un panel de evaluadores compara a ciegas pares de imágenes de dos modelos para la misma instrucción y elige la que mejor la cumple.
rel: evaluacion-de-modelos-generativos, arenas-de-preferencia-humana, llm-as-a-judge
en: qualitative evaluation

## inception-score
frase: Puntúa las imágenes generadas según lo claras y variadas que le parecen a un clasificador preentrenado.
desc: Pasa las imágenes por la red Inception: si cada una se clasifica con confianza en una clase (es nítida) y en conjunto aparecen muchas clases distintas (hay diversidad), la puntuación es alta. Formalmente, es la exponencial de la divergencia KL media entre la distribución de clases de cada imagen y la distribución marginal. No se compara con imágenes reales y solo tiene sentido con clases parecidas a las de ImageNet.
cuando: Como métrica histórica de referencia para imágenes naturales; hoy se prefiere el FID.
ej: Un modelo que genera solo perros muy nítidos obtiene un IS bajo por falta de diversidad.
py: torchmetrics.image.inception.InceptionScore
rel: fid-y-kid, divergencia-de-kullback-leibler, evaluacion-de-modelos-generativos
en: Inception Score (IS)

## fid-y-kid
frase: Comparan la distribución de imágenes reales y generadas en el espacio de características de una red: cuanto más bajo, mejor.
desc: El FID (Fréchet Inception Distance) extrae características de Inception de imágenes reales y generadas, ajusta una gaussiana a cada conjunto y mide la distancia entre ambas. Recoge a la vez calidad y diversidad y es la métrica estándar en generación de imágenes, aunque está sesgado cuando hay pocas muestras. El KID usa en su lugar una distancia basada en kernels (MMD), insesgada y más fiable con muestras pequeñas.
cuando: Para comparar modelos de generación de imágenes entre sí o seguir su evolución durante el entrenamiento.
ej: A lo largo del entrenamiento, el FID de un modelo de difusión baja de 25 a 8 a medida que sus imágenes se parecen más a las reales.
hp: feature: capa de Inception cuyas características se comparan (2048 por defecto) | subset_size: en KID, tamaño de los subconjuntos con los que se estima
py: torchmetrics.image.fid.FrechetInceptionDistance ; torchmetrics.image.kid.KernelInceptionDistance
rel: inception-score, mode-drop-y-cobertura, evaluacion-de-modelos-generativos, metricas-perceptuales
en: FID & KID

## mode-drop-y-cobertura
frase: Medir por separado si lo generado es realista (precisión) y si cubre toda la variedad de los datos (cobertura).
desc: El FID mezcla calidad y diversidad en un solo número. Las métricas de precisión y recall para modelos generativos, y sus versiones mejoradas de densidad y cobertura, las separan: la precisión mide qué fracción de las muestras generadas cae en la zona de los datos reales, y la cobertura, qué fracción de los datos reales tiene muestras generadas cerca. Así detectan el mode drop, la pérdida de partes de la distribución.
cuando: Para diagnosticar si un modelo generativo falla por calidad o por falta de diversidad.
ej: Un generador de caras con una precisión de 0,9 y una cobertura de 0,4 hace caras muy realistas, pero casi nunca de personas mayores.
py: prdc.compute_prdc
rel: colapso-de-modos, fid-y-kid, evaluacion-de-modelos-generativos, precision-y-recall
en: mode dropping & coverage

## verosimilitud-vs-calidad-de-muestras
frase: Un modelo puede asignar mucha probabilidad a los datos y generar muestras malas, o al revés.
desc: La verosimilitud mide lo bien que el modelo cubre la distribución de los datos, pero está dominada por detalles poco perceptibles: un modelo puede tener muy buena verosimilitud y generar imágenes mediocres, mientras que una GAN genera imágenes excelentes sin dar ninguna verosimilitud. Además, los modelos de densidad a veces asignan más probabilidad a datos de otro dominio que a los suyos. Por eso ambas cosas se evalúan por separado.
ej: Un modelo de densidad entrenado con fotos de prendas de ropa asigna más verosimilitud a dígitos manuscritos, que nunca vio, que a la propia ropa.
rel: densidad-explicita-vs-implicita, evaluacion-de-modelos-generativos, ood, normalizing-flows
en: likelihood vs sample quality
