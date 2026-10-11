# Fichas: IA multimodal y audio

## aprendizaje-multimodal
frase: Modelos que combinan varios tipos de datos, como imagen, texto y audio, para entenderlos o generarlos juntos.
desc: Las modalidades se complementan: una imagen y su descripción, un vídeo y su audio. Los modelos multimodales aprenden a representarlas en un espacio común o a traducir entre ellas. Las estrategias van de la fusión temprana (combinar las entradas) a la tardía (combinar las predicciones), pasando por la alineación contrastiva (CLIP) y los LLM que reciben imágenes o audio como tokens. Son la base de los asistentes que ven, oyen y hablan.
ej: Un asistente recibe la foto de un electrodoméstico y la pregunta "¿por qué parpadea esta luz?", y responde combinando ambas.
rel: tareas-multimodales, modelos-contrastivos-imagen-texto, mllm, multi-entrada-salida
en: multimodal learning

## tareas-multimodales
frase: Problemas que cruzan modalidades: buscar imágenes con texto, describirlas o responder preguntas sobre ellas.
desc: Las más comunes son la recuperación entre modalidades (buscar imágenes con una frase o al revés), la descripción automática de imágenes (captioning), la respuesta a preguntas visuales (VQA), la comprensión de documentos con texto y gráficos, la localización en la imagen de lo que menciona una frase (grounding) y la generación de una modalidad a partir de otra. Hoy muchas se resuelven con un único modelo multimodal.
ej: Un sistema de accesibilidad describe en voz alta las fotos que recibe una persona ciega: "dos niños jugando al fútbol en una playa".
py: `pipeline("image-text-to-text")`
rel: aprendizaje-multimodal, mllm, clip, datasets-multimodales
en: multimodal tasks (retrieval, captioning, VQA)

## datasets-multimodales
frase: Colecciones de datos de distintas modalidades emparejados, como imágenes con su descripción.
desc: Los modelos multimodales se entrenan con enormes colecciones de pares de imagen y texto recogidas de la web (LAION, con miles de millones de pares), con conjuntos anotados con cuidado (COCO Captions, Visual Genome) y con datos de instrucciones visuales para asistentes. La calidad, la diversidad, los derechos de uso y los sesgos de estos datos condicionan directamente el comportamiento de los modelos.
ej: CLIP se entrenó con 400 millones de pares de imagen y texto obtenidos de internet, sin etiquetas manuales.
py: datasets.load_dataset
rel: aprendizaje-multimodal, clip, calidad-y-etiquetado-de-datos, sesgo-algoritmico-y-atributos-sensibles
en: multimodal datasets

## modelos-contrastivos-imagen-texto
frase: Modelos que sitúan imágenes y textos en un mismo espacio, donde lo que se corresponde queda cerca.
desc: Entrenan dos codificadores, uno de imagen y otro de texto, con aprendizaje contrastivo sobre pares de imagen y descripción: el embedding de cada imagen debe acercarse al de su texto y alejarse de los demás. El espacio común resultante permite buscar imágenes con texto, clasificar sin ejemplos y servir de codificador visual para otros modelos.
cuando: Para buscar entre imágenes y textos, clasificar sin ejemplos o como componente de sistemas multimodales y generativos.
ej: Buscar en un archivo de fotos de prensa "manifestación con paraguas bajo la lluvia" sin que las fotos tengan etiquetas.
py: transformers.CLIPModel
rel: clip, siglip, aprendizaje-contrastivo, embeddings
en: contrastive image-text models

## clip
frase: Modelo de OpenAI que aprendió a relacionar imágenes y textos a partir de 400 millones de pares de internet.
desc: CLIP (2021) entrena un codificador de imagen (ViT o ResNet) y uno de texto con una pérdida contrastiva simétrica sobre lotes de pares de imagen y texto. Sin etiquetas manuales, aprende un abanico amplísimo de conceptos visuales y se transfiere a muchas tareas sin ajuste. Es el codificador de texto de los primeros Stable Diffusion, la base de la clasificación zero-shot y uno de los codificadores visuales más usados en los modelos multimodales.
cuando: Para búsqueda semántica de imágenes, clasificación sin ejemplos, filtrado de contenido o como codificador visual.
ej: Clasificar las fotos de un catálogo en "sofá", "mesa" o "lámpara" sin ninguna foto de entrenamiento, comparando cada imagen con esas tres palabras.
py: transformers.CLIPModel ; transformers.CLIPProcessor ; open_clip.create_model_and_transforms
rel: modelos-contrastivos-imagen-texto, clasificacion-zero-shot, perdida-infonce, stable-diffusion
en: CLIP

## clasificacion-zero-shot
frase: Clasificar en categorías que el modelo nunca vio al entrenar, describiéndolas con texto.
desc: Con un modelo como CLIP, cada clase se convierte en una frase ("una foto de un perro", "una foto de un gato") y se calcula su embedding de texto. La imagen se asigna a la clase cuya frase queda más cerca de su embedding. Cambiar o añadir clases es tan fácil como escribir otra frase, aunque la forma de redactar las plantillas influye en el resultado.
cuando: Cuando no hay datos etiquetados, las clases cambian a menudo o hay muchísimas clases.
ej: Un supermercado clasifica las fotos de sus estanterías en 50 tipos de producto solo con la lista de nombres, sin ninguna imagen etiquetada.
py: `pipeline("zero-shot-image-classification", model="openai/clip-vit-base-patch32")`
rel: clip, in-context-learning, clasificacion-de-imagenes, siglip
en: zero-shot classification

## siglip
frase: Variante de CLIP que sustituye la softmax sobre el lote por una sigmoide por pareja: más eficiente y mejor con lotes pequeños.
desc: CLIP compara cada imagen con todos los textos del lote mediante una softmax, lo que exige lotes enormes y mucha comunicación entre GPU. SigLIP trata cada par de imagen y texto como una clasificación binaria independiente (¿se corresponden o no?) con una pérdida sigmoide. Escala mejor, rinde más con lotes moderados y hoy es un codificador visual muy usado en los modelos multimodales.
cuando: Como alternativa a CLIP para búsqueda entre imágenes y textos, clasificación zero-shot o como codificador visual de un MLLM.
ej: Muchos modelos multimodales abiertos recientes usan un codificador SigLIP para convertir las imágenes en tokens.
py: transformers.SiglipModel
rel: clip, modelos-contrastivos-imagen-texto, arquitectura-mllm, sigmoide-y-tanh
en: SigLIP

## mllm
frase: LLM que también reciben imágenes, audio o vídeo y responden sobre ellos en lenguaje natural.
desc: Los modelos de lenguaje multimodales, también llamados modelos de visión y lenguaje, combinan un codificador de la otra modalidad con un LLM, de modo que pueden describir imágenes, leer documentos y gráficos, responder preguntas sobre fotos o vídeos y razonar sobre ellos. Heredan las capacidades del LLM (seguir instrucciones, razonar, conversar) y las extienden al mundo visual.
cuando: Para entender documentos escaneados, gráficos, capturas de pantalla, fotos o vídeos mediante preguntas en lenguaje natural.
ej: Subir la foto de un tique de compra y preguntar "¿cuánto gasté en fruta?".
py: transformers.AutoModelForImageTextToText
rel: arquitectura-mllm, entrenamiento-mllm, llm, tareas-multimodales
en: multimodal LLMs (MLLM / VLM)

## arquitectura-mllm
frase: Un codificador visual convierte la imagen en tokens, un conector los traduce al espacio del LLM y el LLM genera la respuesta.
desc: El diseño más común tiene tres piezas: un codificador de imagen preentrenado, normalmente un ViT como CLIP o SigLIP, que produce una secuencia de vectores; un conector, desde una capa lineal o un pequeño MLP hasta un módulo con atención, que los proyecta como si fueran tokens de texto; y un LLM que procesa esos tokens visuales junto con el texto. Otras arquitecturas inyectan la imagen mediante atención cruzada dentro del LLM.
ej: En LLaVA, los 576 vectores que el codificador CLIP extrae de una imagen se proyectan con un MLP y entran al LLM como si fueran 576 palabras.
py: transformers.LlavaForConditionalGeneration
rel: mllm, vit, atencion-cruzada, siglip
en: MLLM architecture (encoder, connector & LLM)

## entrenamiento-mllm
frase: Primero se alinea el conector con pares de imagen y texto; después se ajusta todo con instrucciones visuales.
desc: El entrenamiento típico tiene dos fases. En el preentrenamiento de alineación, con el codificador y el LLM congelados, solo se entrena el conector con cientos de miles o millones de pares de imagen y descripción, para que el LLM entienda los tokens visuales. En el ajuste con instrucciones visuales se entrenan el conector y el LLM (a veces también el codificador) con conversaciones sobre imágenes, para que siga instrucciones y razone sobre ellas. Puede añadirse un alineamiento con preferencias.
ej: Un MLLM se alinea primero con unos 600.000 pares de imagen y pie de foto y después se ajusta con unas 150.000 conversaciones sobre imágenes.
py: trl.SFTTrainer
rel: mllm, arquitectura-mllm, instruction-tuning, congelar-y-descongelar-capas
en: MLLM training

## modelos-any-to-any
frase: Modelos que reciben y generan cualquier combinación de modalidades: texto, imagen, audio y vídeo.
desc: Van un paso más allá de los MLLM, que solo generan texto: pueden responder con imágenes, con voz o con vídeo. Lo consiguen representando todas las modalidades como tokens de una única secuencia que procesa un transformer, o acoplando el modelo central a decodificadores especializados, como un modelo de difusión para las imágenes o un sintetizador para la voz. Permiten conversaciones fluidas por voz y edición de imágenes con instrucciones.
ej: Un asistente al que se le habla, que ve por la cámara lo que se le enseña y que responde con voz y con una imagen de ejemplo.
rel: mllm, generacion-multimodal, sintesis-de-voz, texto-a-imagen
en: any-to-any models

## generacion-multimodal
frase: Generar contenido de una modalidad a partir de otra: imágenes desde texto, vídeo desde imágenes, audio desde texto.
desc: Combina un codificador que entiende la condición, normalmente un texto, con un generador de la modalidad de salida, casi siempre un modelo de difusión o autorregresivo. La condición guía la generación mediante atención cruzada o tokens compartidos. Abarca texto a imagen, imagen a imagen, texto o imagen a vídeo, texto a voz y texto a música.
ej: A partir del guion de un anuncio, generar imágenes, una locución y una música de fondo coherentes con el tono.
rel: texto-a-imagen, generacion-de-video, sintesis-de-voz, generacion-de-musica
en: multimodal generation

## texto-a-imagen
frase: Generar imágenes a partir de una descripción en lenguaje natural.
desc: Los modelos actuales codifican el texto con un codificador de lenguaje y lo usan para guiar un modelo de difusión o de flow matching, casi siempre en un espacio latente comprimido. La guía sin clasificador controla cuánto se ciñe la imagen al texto. Admiten estilos, composiciones y edición con instrucciones, y se ajustan con LoRA o DreamBooth para estilos o sujetos concretos. Plantean cuestiones de derechos de autor, consentimiento y desinformación.
cuando: Para ilustraciones, prototipos visuales, publicidad, diseño o datos sintéticos.
ej: "Ilustración minimalista de un faro al atardecer, colores pastel" genera en segundos cuatro propuestas para la portada de un libro.
py: diffusers.StableDiffusionPipeline ; diffusers.FluxPipeline
rel: stable-diffusion, guia-difusion, flow-matching, generacion-multimodal
en: text-to-image

## generacion-de-video
frase: Generar vídeos a partir de un texto o animar una imagen fija.
desc: Extiende la generación de imágenes al tiempo, normalmente con transformers de difusión sobre parches espacio-temporales en un espacio latente. El paso de texto a vídeo genera clips desde una descripción; el de imagen a vídeo anima una imagen dada respetando su contenido. Los retos son la coherencia temporal, una física plausible, la duración y el enorme coste de cálculo.
cuando: Para prototipos audiovisuales, publicidad, animación de imágenes o contenido para redes sociales.
ej: Convertir la foto de un producto en un clip de 5 segundos en el que la cámara gira a su alrededor.
py: diffusers.CogVideoXImageToVideoPipeline ; diffusers.LTXPipeline
rel: difusion-de-video, texto-a-imagen, generacion-multimodal, modelos-de-difusion
en: text-to-video & image-to-video

## audio-y-voz
frase: Procesar y generar sonido con IA: reconocer el habla, sintetizar voz, clasificar sonidos o crear música.
desc: El audio es una señal temporal muy densa, con miles de muestras por segundo, así que suele convertirse en espectrogramas o en tokens discretos mediante códecs neuronales. Sobre esas representaciones se aplican las mismas arquitecturas que en texto e imagen: transformers para reconocer el habla, difusión o modelos autorregresivos para generarla y modelos contrastivos para relacionar audio y texto.
ej: Una aplicación transcribe una reunión, identifica quién habla en cada momento y genera un resumen hablado.
py: librosa.load
rel: espectrograma-de-mel, reconocimiento-de-voz, sintesis-de-voz, convolucion-1d
en: audio & speech

## espectrograma-de-mel
frase: Representar el audio como una imagen: la energía por bandas de frecuencia a lo largo del tiempo, en escala auditiva.
desc: Se divide el audio en ventanas cortas, se calcula el espectro de cada una con la transformada de Fourier y se agrupan las frecuencias en bandas según la escala de Mel, que imita la percepción humana (más resolución en los graves que en los agudos). Suele tomarse el logaritmo de la energía. El resultado es una imagen de tiempo frente a frecuencia que procesan las CNN y los transformers de audio.
cuando: Como representación de entrada de casi cualquier modelo de voz, música o sonido.
ej: Whisper convierte cada fragmento de 30 segundos de audio en un espectrograma de Mel antes de transcribirlo.
hp: n_mels: número de bandas de Mel | n_fft: tamaño de la ventana de Fourier | hop_length: desplazamiento entre ventanas consecutivas
py: librosa.feature.melspectrogram ; torchaudio.transforms.MelSpectrogram
rel: transformada-de-fourier, audio-y-voz, reconocimiento-de-voz, series-como-imagenes
en: Mel spectrogram

## reconocimiento-de-voz
frase: Convertir la voz en texto: la transcripción automática del habla.
desc: Los sistemas modernos son transformers que reciben el espectrograma y generan el texto. Whisper (2022), un modelo codificador-decodificador entrenado con 680.000 horas de audio multilingüe de internet, transcribe y traduce casi 100 idiomas con gran robustez frente al ruido y los acentos. Otros enfoques, como wav2vec 2.0, se preentrenan de forma autosupervisada con audio sin transcribir.
cuando: Para transcribir reuniones, llamadas, entrevistas o vídeos, subtitular o crear interfaces de voz.
ej: Transcribir automáticamente las llamadas a un centro de atención para analizar después los motivos de queja.
py: transformers.WhisperForConditionalGeneration ; `pipeline("automatic-speech-recognition", model="openai/whisper-large-v3")`
rel: espectrograma-de-mel, audio-y-voz, tipos-de-transformer, aprendizaje-autosupervisado
en: automatic speech recognition (Whisper)

## sintesis-de-voz
frase: Generar habla natural a partir de un texto, con una voz y una entonación determinadas.
desc: Los sistemas clásicos encadenaban un modelo que predecía un espectrograma a partir del texto y un vocoder que lo convertía en onda sonora. Los actuales trabajan de extremo a extremo (VITS) o generan tokens de audio con modelos de lenguaje o de difusión, lo que permite clonar una voz a partir de unos segundos de muestra y controlar la emoción. Plantean riesgos de suplantación de identidad.
cuando: Para lectura en voz alta, asistentes de voz, doblaje, accesibilidad o audiolibros.
ej: Un periódico ofrece cada noticia también en audio, leída por una voz sintética natural.
py: transformers.VitsModel ; `pipeline("text-to-speech")`
rel: audio-y-voz, espectrograma-de-mel, modelos-any-to-any, generacion-de-musica
en: text-to-speech (TTS)

## clap
frase: El equivalente de CLIP para el audio: relaciona sonidos y descripciones en un espacio común.
desc: CLAP (Contrastive Language-Audio Pretraining) entrena un codificador de audio y uno de texto con aprendizaje contrastivo sobre pares de sonido y descripción. Permite buscar sonidos con texto, clasificar audio sin ejemplos y servir de guía, o de medida de fidelidad, en los modelos de generación de audio.
cuando: Para buscar sonidos mediante texto, clasificar audio sin ejemplos o evaluar la coherencia entre un audio generado y su descripción.
ej: Buscar en una biblioteca de efectos de sonido "puerta de madera que cruje al abrirse" sin depender de etiquetas manuales.
py: transformers.ClapModel
rel: clip, aprendizaje-contrastivo, audio-y-voz, clasificacion-zero-shot
en: CLAP (contrastive language-audio pretraining)

## generacion-de-musica
frase: Generar música a partir de una descripción, una melodía o un fragmento inicial.
desc: Los modelos actuales representan la música como tokens de un códec neuronal de audio y la generan con transformers autorregresivos, como MusicGen, o con difusión, condicionados por un texto, una melodía o un estilo de referencia. Pueden producir piezas de varios minutos con una instrumentación coherente. Plantean cuestiones abiertas sobre los derechos de autor de los datos de entrenamiento y la autoría.
cuando: Para bandas sonoras de vídeos, prototipos musicales, música adaptativa en videojuegos o exploración creativa.
ej: "Música de piano tranquila con lluvia de fondo para estudiar" genera una pieza original de dos minutos.
py: transformers.MusicgenForConditionalGeneration
rel: modelos-autorregresivos, audio-y-voz, cuantizacion-vectorial, generacion-multimodal
en: music generation
