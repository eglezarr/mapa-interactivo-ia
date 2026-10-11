# Fichas: Visión por computador

## vision-clasica
frase: Técnicas anteriores al deep learning para procesar imágenes: filtros, bordes, frecuencias y escalas.
desc: Antes de las CNN, la visión por computador se basaba en operaciones diseñadas a mano: filtros para suavizar o detectar bordes, transformadas que describen la imagen por frecuencias, pirámides y wavelets que la analizan a varias escalas, y descriptores de puntos característicos como SIFT. Siguen siendo útiles para preprocesar, cuando hay pocos datos o recursos, y ayudan a entender lo que aprenden las redes.
ej: Detectar el contorno de una pieza en una cinta transportadora con el detector de bordes de Canny, sin entrenar ningún modelo.
py: skimage.feature.canny ; skimage.filters.sobel
rel: piramides-gaussiana-y-laplaciana, transformada-de-fourier, wavelets, cnn
en: classical computer vision

## piramides-gaussiana-y-laplaciana
frase: Representar una imagen a varias resoluciones: versiones cada vez más pequeñas y suaves, y los detalles que se pierden entre ellas.
desc: La pirámide gaussiana se obtiene suavizando la imagen y reduciéndola a la mitad repetidamente: cada nivel conserva la estructura más gruesa. La laplaciana guarda en cada nivel la diferencia entre un nivel gaussiano y la versión ampliada del siguiente, es decir, los detalles de cada escala; sumándolos se recupera la imagen original. Se usan para fusionar imágenes, comprimir, buscar objetos de distintos tamaños y en medidas de calidad perceptual.
cuando: Para analizar o procesar una imagen a varias escalas, o para mezclar imágenes sin costuras visibles.
ej: Fusionar media manzana y media naranja de dos fotos distintas mezclando cada nivel de sus pirámides laplacianas, sin que se note la unión.
hp: downscale: factor de reducción entre niveles (2 por defecto) | sigma: suavizado gaussiano antes de reducir | max_layer: número de niveles
py: skimage.transform.pyramid_gaussian ; skimage.transform.pyramid_laplacian
rel: modelos-multiescala, wavelets, metricas-perceptuales, vision-clasica
en: Gaussian & Laplacian pyramids

## transformada-de-fourier
frase: Descompone una señal o una imagen en ondas de distintas frecuencias: lo que cambia despacio y lo que cambia deprisa.
desc: Cualquier imagen puede escribirse como una suma de ondas sinusoidales de distintas frecuencias y orientaciones. Las bajas frecuencias describen las zonas suaves y la forma general; las altas, los bordes, las texturas y el ruido. La transformada rápida (FFT) la calcula de forma eficiente. Permite filtrar ruido o patrones periódicos, comprimir y acelerar las convoluciones, que en frecuencia son simples productos.
cuando: Para filtrar ruido periódico, analizar texturas o patrones repetitivos, o acelerar convoluciones con filtros grandes.
ej: Eliminar las franjas periódicas de una imagen escaneada borrando los picos correspondientes de su espectro de Fourier.
py: numpy.fft.fft2 ; numpy.fft.fftshift
rel: wavelets, convolucion, espectrograma-de-mel, vision-clasica
en: Fourier transform

## wavelets
frase: Descomponen la imagen en detalles localizados a la vez en el espacio y en la escala.
desc: A diferencia de las ondas de Fourier, que se extienden por toda la imagen, las wavelets son ondas cortas y localizadas que se desplazan y se escalan. La transformada wavelet separa la imagen en una versión reducida y en coeficientes de detalle horizontales, verticales y diagonales para cada escala. Como las imágenes naturales tienen pocos coeficientes grandes, son la base de la compresión JPEG 2000 y de métodos eficaces de eliminación de ruido.
cuando: Para comprimir, eliminar ruido o analizar señales e imágenes cuyos patrones cambian a lo largo del espacio o del tiempo.
ej: Limpiar el ruido de una radiografía anulando los coeficientes wavelet pequeños, que corresponden sobre todo a ruido, y reconstruyendo la imagen.
hp: wavelet: familia de wavelet (Haar, Daubechies…) | level: número de niveles de descomposición
py: pywt.wavedec2 ; pywt.dwt2
rel: transformada-de-fourier, piramides-gaussiana-y-laplaciana, modelos-multiescala, vision-clasica
en: wavelets

## arquitecturas-cnn-de-referencia
frase: Las redes convolucionales que marcaron época y se siguen usando como base: de AlexNet a ConvNeXt.
desc: Cada arquitectura aportó una idea que hoy es estándar: AlexNet demostró el poder de las CNN profundas entrenadas en GPU; VGG, la profundidad con filtros pequeños; GoogLeNet, los módulos multiescala eficientes; ResNet, las conexiones residuales; MobileNet y EfficientNet, la eficiencia; ConvNeXt, la modernización de las CNN frente a los transformers. Sus versiones preentrenadas son el punto de partida habitual del transfer learning.
ej: Para clasificar fotos de productos se parte de una ResNet o una EfficientNet preentrenada con ImageNet en lugar de diseñar una red propia.
py: keras.applications.ResNet50 ; timm.create_model
rel: imagenet-y-alexnet, resnet, mobilenet-y-efficientnet, transfer-learning
en: reference CNN architectures

## imagenet-y-alexnet
frase: El gran dataset de imágenes y la red que, en 2012, demostró que el deep learning funcionaba.
desc: ImageNet reunió millones de imágenes etiquetadas en miles de categorías, y su competición anual (ILSVRC) consistía en clasificar 1,2 millones de imágenes en 1.000 clases. En 2012, AlexNet, una CNN de 8 capas entrenada en dos GPU con ReLU y dropout, redujo el error top-5 del 26 % al 15 %, muy por delante del resto. Ese salto desencadenó la adopción masiva del deep learning.
ej: Un modelo preentrenado con ImageNet sigue siendo el punto de partida por defecto de muchos clasificadores de imágenes actuales.
py: torchvision.models.alexnet
rel: arquitecturas-cnn-de-referencia, breve-historia-de-la-ia, clasificacion-de-imagenes, dropout
en: ImageNet & AlexNet

## vgg
frase: CNN muy profunda y regular hecha solo de convoluciones de 3 × 3: la simplicidad como principio.
desc: VGG (2014) apila bloques de convoluciones de 3 × 3 seguidos de max pooling, hasta 16 o 19 capas. Demostró que dos convoluciones de 3 × 3 cubren el mismo campo receptivo que una de 5 × 5 con menos parámetros y más no linealidad. Es pesada, con unos 138 millones de parámetros, pero su estructura limpia la hace muy usada como extractor de características, por ejemplo en las pérdidas perceptuales y en la transferencia de estilo.
ej: Las pérdidas perceptuales de muchos modelos de superresolución comparan activaciones de una VGG preentrenada.
py: keras.applications.VGG16 ; torchvision.models.vgg16
rel: arquitecturas-cnn-de-referencia, perdidas-perceptuales, transferencia-de-estilo-neuronal, campo-receptivo
en: VGG

## googlenet
frase: CNN que analiza cada zona a varias escalas en paralelo con módulos Inception, usando muy pocos parámetros.
desc: GoogLeNet (2014) introdujo el módulo Inception: convoluciones de 1 × 1, 3 × 3 y 5 × 5 y un pooling en paralelo, cuyas salidas se concatenan, de modo que la red elige la escala adecuada en cada capa. Las convoluciones de 1 × 1 reducen los canales antes de las operaciones caras. Con 22 capas y solo unos 7 millones de parámetros ganó ImageNet 2014, y sus sucesoras (Inception v3 y v4) refinaron la idea.
ej: En una misma capa, un módulo Inception detecta a la vez un detalle pequeño, como un ojo, y una forma mayor, como una cara.
py: keras.applications.InceptionV3 ; torchvision.models.googlenet
rel: arquitecturas-cnn-de-referencia, modelos-multiescala, convolucion-separable, resnet
en: GoogLeNet (Inception)

## resnet
frase: CNN con conexiones residuales que permitió entrenar redes de más de cien capas.
desc: ResNet (2015) observó que añadir capas a una CNN profunda empeoraba incluso el error de entrenamiento, y lo resolvió con conexiones residuales: cada bloque aprende una corrección que se suma a su entrada. Así entrenó redes de 152 capas, ganó ImageNet 2015 y bajó el error top-5 al 3,6 %. Las conexiones residuales se convirtieron en un componente universal, y ResNet-50 sigue siendo una línea base estándar.
cuando: Como línea base sólida para clasificar imágenes o como columna vertebral de detectores y segmentadores.
ej: Una ResNet-50 preentrenada, ajustada con 3.000 fotos, clasifica los defectos de piezas de fundición.
py: keras.applications.ResNet50 ; torchvision.models.resnet50
rel: conexion-residual, arquitecturas-cnn-de-referencia, desvanecimiento-y-explosion-del-gradiente, transfer-learning
en: ResNet

## densenet-resnext-y-senet
frase: Tres refinamientos de ResNet: conectar cada capa con todas, agrupar transformaciones y repesar los canales.
desc: DenseNet conecta cada capa con todas las siguientes de su bloque concatenando sus salidas, lo que reutiliza características y reduce parámetros. ResNeXt sustituye cada bloque residual por varias ramas paralelas idénticas (cardinalidad), algo más eficaz que hacer la red más ancha o más profunda. SENet añade bloques de compresión y excitación que aprenden cuánto pesa cada canal en cada imagen, una forma temprana de atención; ganó ImageNet 2017.
ej: Un bloque SE aprende a dar más peso a los canales que detectan pelo cuando la imagen es de un animal.
py: torchvision.models.densenet121 ; torchvision.models.resnext50_32x4d
rel: resnet, arquitecturas-cnn-de-referencia, mecanismo-de-atencion, conexion-residual
en: DenseNet, ResNeXt & SENet

## mobilenet-y-efficientnet
frase: CNN diseñadas para dar la máxima precisión con el mínimo cálculo, pensadas para móviles y producción.
desc: MobileNet construye la red con convoluciones separables en profundidad, que reducen el cálculo entre 8 y 9 veces, y sus versiones posteriores añaden bloques residuales invertidos y diseño automático. EfficientNet propone escalar a la vez la profundidad, la anchura y la resolución con un coeficiente común (escalado compuesto), a partir de una red base encontrada con NAS. Ambas dominan cuando importan la latencia, la memoria o el consumo.
cuando: Para visión en móviles, dispositivos embebidos o servicios en los que importa el coste de cada predicción.
ej: Una app identifica en tiempo real especies de plantas con la cámara del móvil usando una MobileNet, sin conexión a internet.
py: keras.applications.MobileNetV2 ; keras.applications.EfficientNetB0 ; torchvision.models.efficientnet_b0
rel: convolucion-separable, nas, despliegue-e-inferencia, arquitecturas-cnn-de-referencia
en: MobileNet & EfficientNet

## convnext
frase: Una CNN modernizada con las ideas de los transformers que compite de tú a tú con ellos.
desc: ConvNeXt (2022) partió de una ResNet y le aplicó, paso a paso, decisiones de diseño de los vision transformers: núcleos convolucionales grandes (7 × 7) en profundidad, normalización de capa, GELU, menos activaciones y una receta de entrenamiento moderna. El resultado iguala o supera a transformers de tamaño similar y conserva la sencillez y la eficiencia de las convoluciones. Demostró que buena parte de la ventaja de los ViT venía del diseño y del entrenamiento, no solo de la atención.
cuando: Como columna vertebral moderna para clasificación, detección o segmentación cuando se prefiere una CNN.
ej: Sustituir la ResNet de un detector por una ConvNeXt de tamaño similar mejora la precisión sin cambiar el resto del sistema.
py: keras.applications.ConvNeXtTiny ; torchvision.models.convnext_tiny
rel: vit, resnet, transformers-vs-cnn-y-rnn, arquitecturas-cnn-de-referencia
en: ConvNeXt

## vit
frase: Un transformer aplicado a imágenes: la imagen se trocea en parches que se tratan como palabras.
desc: El Vision Transformer (2020) divide la imagen en parches, por ejemplo de 16 × 16 píxeles, los proyecta como vectores, añade codificación posicional y un token [CLS], y los procesa con un transformer codificador estándar. Sin los supuestos de localidad de las CNN necesita muchos datos o un preentrenamiento para brillar, pero escala mejor y relaciona zonas lejanas desde la primera capa. Es la base de CLIP, DINOv2, SAM y los modelos multimodales.
cuando: Cuando hay preentrenamientos grandes o muchos datos, y como codificador visual de sistemas multimodales.
ej: Una imagen de 224 × 224 se convierte en una secuencia de 196 parches de 16 × 16 que el transformer procesa como si fueran palabras.
py: transformers.ViTForImageClassification ; torchvision.models.vit_b_16
rel: transformers-vs-cnn-y-rnn, token-cls, codificacion-posicional, clip, modelado-enmascarado
en: Vision Transformer (ViT)

## clasificacion-de-imagenes
frase: Asignar a cada imagen una o varias etiquetas según lo que contiene.
desc: Es la tarea más básica de la visión: dada una imagen, predecir su clase (un gato, una radiografía con neumonía, una pieza defectuosa). Se resuelve casi siempre ajustando una CNN o un ViT preentrenado, con aumento de datos y, si las clases están desbalanceadas, con métricas adecuadas. Con modelos como CLIP también se puede clasificar sin ejemplos, describiendo las clases con texto.
cuando: Cuando cada imagen tiene una categoría global y no importa dónde está el objeto.
ej: Clasificar fotos de lunares en benignos o sospechosos para priorizar las consultas de dermatología.
py: `pipeline("image-classification")` ; timm.create_model
rel: cnn, transfer-learning, data-augmentation, clasificacion-zero-shot
en: image classification

## deteccion-de-objetos
frase: Encontrar qué objetos hay en una imagen y dónde está cada uno, con una caja a su alrededor.
desc: Además de la clase, el detector predice una caja delimitadora para cada objeto, y puede haber muchos objetos de clases distintas. Los detectores de dos fases proponen regiones y después las clasifican; los de una fase lo predicen todo de golpe y son más rápidos; los basados en transformers (DETR) tratan la detección como la predicción de un conjunto. Se evalúan con la precisión media (mAP) a distintos umbrales de IoU.
cuando: Cuando importa localizar y contar los objetos, no solo saber si aparecen.
ej: Contar los vehículos de cada tipo que pasan por una rotonda a partir del vídeo de una cámara de tráfico.
py: ultralytics.YOLO ; torchvision.models.detection.fasterrcnn_resnet50_fpn
rel: cajas-delimitadoras-e-iou, detectores-de-dos-fases, detectores-de-una-fase, seguimiento-de-objetos
en: object detection

## cajas-delimitadoras-e-iou
frase: El rectángulo que localiza cada objeto y la medida de cuánto se solapa con el real.
desc: Una caja delimitadora se describe con cuatro números: sus esquinas, o su centro, su anchura y su altura. La intersección sobre la unión (IoU) divide el área compartida por dos cajas entre el área total que cubren: vale 1 si coinciden y 0 si no se tocan. Una detección cuenta como acierto si su IoU con la caja real supera un umbral, típicamente 0,5. La supresión de no máximos (NMS) elimina las detecciones duplicadas de un mismo objeto.
ej: Si un detector marca el mismo coche con tres cajas solapadas, NMS conserva la de mayor confianza y descarta las otras dos.
hp: iou_threshold: en NMS, solapamiento a partir del cual dos cajas se consideran duplicadas
py: torchvision.ops.box_iou ; torchvision.ops.nms
rel: deteccion-de-objetos, detectores-de-una-fase, metricas-de-clasificacion, segmentacion-semantica-y-de-instancias
en: bounding boxes & IoU

## detectores-de-dos-fases
frase: Primero proponen regiones donde puede haber un objeto y después clasifican y ajustan cada una.
desc: La familia R-CNN evolucionó de procesar por separado miles de regiones propuestas (R-CNN), a compartir el cálculo de la CNN para toda la imagen (Fast R-CNN) y a generar las propuestas con la propia red mediante una red de propuesta de regiones (Faster R-CNN). Mask R-CNN añade una rama que predice la máscara de cada objeto. Son precisos, sobre todo con objetos pequeños, pero más lentos que los detectores de una fase.
cuando: Cuando la precisión importa más que la velocidad, o cuando hay que segmentar cada objeto.
ej: Localizar con precisión pequeños nódulos en tomografías de pulmón, donde pasar uno por alto es grave.
py: torchvision.models.detection.fasterrcnn_resnet50_fpn ; torchvision.models.detection.maskrcnn_resnet50_fpn
rel: deteccion-de-objetos, detectores-de-una-fase, segmentacion-semantica-y-de-instancias, cajas-delimitadoras-e-iou
en: two-stage detectors (R-CNN family)

## detectores-de-una-fase
frase: Predicen todas las cajas y clases en una sola pasada por la red: rápidos y aptos para tiempo real.
desc: YOLO divide la imagen en una rejilla y predice directamente, para cada celda, cajas, confianzas y clases, lo que permite procesar vídeo en tiempo real; sus sucesivas versiones han mejorado mucho la precisión. RetinaNet introdujo la focal loss, que reduce el peso de los ejemplos fáciles de fondo y resolvió el desequilibrio que penalizaba a estos detectores. Hoy alcanzan una precisión comparable a la de los detectores de dos fases.
cuando: Para detectar en tiempo real o en dispositivos con pocos recursos: vídeo, robótica o conducción.
ej: Un dron detecta personas en tiempo real durante una búsqueda y rescate en montaña.
py: ultralytics.YOLO ; torchvision.models.detection.retinanet_resnet50_fpn
rel: deteccion-de-objetos, detectores-de-dos-fases, desbalanceo-de-clases, seguimiento-de-objetos
en: one-stage detectors (YOLO, RetinaNet)

## seguimiento-de-objetos
frase: Detectar los objetos en cada fotograma de un vídeo y mantener la identidad de cada uno a lo largo del tiempo.
desc: El enfoque habitual, el seguimiento por detección, aplica un detector a cada fotograma y asocia las detecciones nuevas con las trayectorias existentes usando una predicción del movimiento (filtro de Kalman), el solapamiento de las cajas y, a veces, la apariencia. Algoritmos como SORT, DeepSORT o ByteTrack gestionan las oclusiones, los objetos que entran y salen y los cambios de identidad.
cuando: Para contar, medir trayectorias o analizar el comportamiento de objetos en vídeo.
ej: Contar los clientes distintos que entran en una tienda y medir cuánto tiempo pasa cada uno en cada pasillo.
hp: track_activation_threshold: confianza mínima de una detección para iniciar una trayectoria | lost_track_buffer: fotogramas que se mantiene una trayectoria perdida antes de darla por terminada
py: supervision.ByteTrack ; `YOLO("yolo11n.pt").track(source="video.mp4")`
rel: deteccion-de-objetos, detectores-de-una-fase, mean-shift, cajas-delimitadoras-e-iou
en: object tracking

## segmentacion
frase: Clasificar cada píxel de la imagen para delimitar con exactitud objetos y regiones.
desc: La segmentación produce una máscara del tamaño de la imagen en lugar de una caja. La semántica asigna una clase a cada píxel; la de instancias distingue además cada objeto individual; la panóptica combina ambas. Se usa allí donde importa la forma exacta: medicina, conducción, agricultura o edición de imágenes. Modelos fundacionales como SAM permiten segmentar casi cualquier cosa sin entrenar.
ej: Medir el área exacta de un tumor en una resonancia segmentando sus píxeles.
rel: segmentacion-semantica-y-de-instancias, u-net, sam, deteccion-de-objetos
en: image segmentation

## segmentacion-semantica-y-de-instancias
frase: Semántica: a qué clase pertenece cada píxel. De instancias: además, a qué objeto concreto.
desc: En la segmentación semántica todos los píxeles de coche reciben la etiqueta "coche", sin distinguir entre coches. En la de instancias, cada coche tiene su propia máscara, lo que permite contarlos y medirlos por separado. La semántica suele resolverse con redes codificador-decodificador (U-Net, DeepLab) y la de instancias, con detectores con rama de máscara (Mask R-CNN) o con transformers. Se evalúan con el IoU medio por clase o con la precisión media de las máscaras.
cuando: La semántica, para clasificar regiones (carretera, cielo, cultivo); la de instancias, para contar o medir objetos individuales.
ej: En una imagen de satélite, la segmentación semántica marca toda la superficie construida; la de instancias separa cada edificio.
py: torchvision.models.segmentation.deeplabv3_resnet50 ; torchvision.models.detection.maskrcnn_resnet50_fpn
rel: segmentacion, u-net, detectores-de-dos-fases, cajas-delimitadoras-e-iou
en: semantic & instance segmentation

## u-net
frase: Red en forma de U que comprime la imagen y la reconstruye, con atajos que conservan los detalles finos.
desc: Un codificador reduce la resolución mientras extrae características, y un decodificador simétrico la recupera; las conexiones de salto llevan los mapas de cada nivel del codificador al nivel correspondiente del decodificador, de modo que se combinan el contexto global y la localización precisa. Nació en 2015 para segmentar imágenes biomédicas con pocos datos y es también la red que elimina el ruido en muchos modelos de difusión.
cuando: Para segmentar con pocos datos, sobre todo en medicina, y en cualquier tarea que produzca una imagen del mismo tamaño que la entrada.
ej: Segmentar células en imágenes de microscopio entrenando con solo 30 imágenes anotadas y mucho aumento de datos.
hp: encoder_name: red preentrenada que se usa como codificador | classes: número de clases de la máscara de salida
py: segmentation_models_pytorch.Unet
rel: convolucion-transpuesta, segmentacion-semantica-y-de-instancias, ddpm, conexion-residual
en: U-Net

## sam
frase: Modelo fundacional que segmenta cualquier objeto de cualquier imagen a partir de un punto, una caja o un texto.
desc: Segment Anything (2023) se entrenó con más de mil millones de máscaras. Recibe una indicación (un clic, una caja o una máscara aproximada) y devuelve la máscara del objeto, sin entrenarse para ese tipo de objeto. Separa un codificador de imagen pesado, que se ejecuta una sola vez, de un decodificador ligero que responde al instante a cada indicación. SAM 2 lo extiende a vídeo y sigue el objeto entre fotogramas.
cuando: Para segmentar objetos de cualquier tipo sin datos de entrenamiento, o para acelerar el etiquetado de máscaras.
ej: En una herramienta de anotación, un clic sobre cada fruta de una foto genera al instante su máscara completa.
py: transformers.SamModel ; segment_anything.SamPredictor
rel: segmentacion, modelos-fundacionales, vit, calidad-y-etiquetado-de-datos
en: Segment Anything (SAM)

## estimacion-de-pose-y-profundidad
frase: Inferir la postura de las personas (sus articulaciones) y la distancia a la cámara de cada punto de la escena.
desc: La estimación de pose localiza puntos clave del cuerpo (hombros, codos, rodillas) y los une en un esqueleto, en 2D o en 3D; sirve para analizar el movimiento, el deporte o la ergonomía. La estimación de profundidad monocular predice, a partir de una sola foto, la distancia de cada píxel, aprendiendo pistas como el tamaño relativo o la perspectiva. Modelos fundacionales como Depth Anything generalizan a casi cualquier escena.
cuando: Para analizar el movimiento humano, controlar la composición al generar imágenes o dar sensación de 3D con una sola cámara.
ej: Una app de fisioterapia comprueba con la cámara del móvil si el paciente dobla la rodilla con el ángulo correcto.
py: ultralytics.YOLO ; transformers.AutoModelForDepthEstimation
rel: deteccion-de-objetos, controlnet, vision-3d-y-neural-rendering, seguimiento-de-objetos
en: pose & depth estimation

## calidad-de-imagen
frase: Medir automáticamente lo buena o degradada que se ve una imagen, idealmente como la juzgaría una persona.
desc: La evaluación de la calidad de imagen (IQA) cuantifica la degradación por compresión, ruido, desenfoque o generación, comparando o no con una imagen de referencia. Las métricas clásicas por píxel, como el PSNR, se correlacionan mal con la percepción; las estructurales y las perceptuales, basadas en modelos del sistema visual o en redes, se ajustan mucho mejor a los juicios humanos.
ej: Un servicio de vídeo ajusta la compresión de cada escena para mantener constante la calidad percibida con el mínimo de bits.
rel: con-referencia-vs-sin-referencia, psnr-y-ssim, metricas-perceptuales, perceptnet
en: image quality assessment (IQA)

## con-referencia-vs-sin-referencia
frase: Medir la calidad comparando con la imagen original perfecta, o sin disponer de ella.
desc: Las métricas con referencia completa (PSNR, SSIM, LPIPS) comparan la imagen procesada con la original y son las más fiables, pero solo sirven cuando esa original existe, como al evaluar una compresión. Las métricas sin referencia (NIQE, BRISQUE o modelos aprendidos) estiman la calidad de una imagen aislada a partir de las propiedades estadísticas de las imágenes naturales; son necesarias con fotos de usuarios o imágenes generadas.
cuando: Con referencia, para evaluar compresión o restauración con el original disponible; sin referencia, para fotos reales o imágenes generadas.
ej: Una red social descarta automáticamente las fotos subidas que están muy borrosas sin tener ninguna versión buena con la que compararlas.
rel: calidad-de-imagen, psnr-y-ssim, metricas-perceptuales, evaluacion-de-modelos-generativos
en: full-reference vs no-reference IQA

## psnr-y-ssim
frase: Las dos métricas clásicas de calidad con referencia: el error por píxel (PSNR) y la similitud de estructura (SSIM).
desc: El PSNR expresa en decibelios el cociente entre el valor máximo de la señal y el error cuadrático medio: es simple, pero trata igual todos los errores, se noten o no. El SSIM compara localmente la luminancia, el contraste y la estructura de ambas imágenes y se acerca más a la percepción; su versión multiescala (MS-SSIM), todavía más. Ninguna de las dos capta bien la calidad de las texturas generadas por modelos.
cuando: Para evaluar compresión, eliminación de ruido o superresolución cuando se dispone de la imagen original.
ej: Una imagen desplazada un píxel puede obtener un PSNR peor que otra visiblemente borrosa, mientras que el SSIM las ordena mejor.
hp: data_range: rango de valores de la imagen (por ejemplo, 255) | win_size: en SSIM, tamaño de la ventana local
py: skimage.metrics.peak_signal_noise_ratio ; skimage.metrics.structural_similarity
rel: calidad-de-imagen, metricas-perceptuales, mse-rmse-y-mae, srgan
en: PSNR & SSIM

## metricas-perceptuales
frase: Distancias entre imágenes diseñadas para coincidir con lo que perciben las personas.
desc: LPIPS compara las activaciones de una red profunda (AlexNet o VGG) ajustada con juicios humanos de similitud. NLPD, basada en una pirámide laplaciana normalizada, imita las primeras etapas del sistema visual con transformaciones locales y normalización divisiva, con muy pocos parámetros. Ambas predicen las preferencias humanas mucho mejor que el PSNR o el SSIM y se usan para evaluar y entrenar modelos de imagen.
cuando: Para evaluar o entrenar modelos que generan o restauran imágenes cuando importa la calidad percibida.
ej: Entre dos resultados de superresolución, uno algo más nítido y otro más suave, LPIPS elige el mismo que prefiere la mayoría de los observadores.
hp: net: red con la que se extraen las características (alex o vgg)
py: lpips.LPIPS
rel: perdidas-perceptuales, perceptnet, piramides-gaussiana-y-laplaciana, gdn
en: perceptual metrics (LPIPS, NLPD)

## vision-3d-y-neural-rendering
frase: Reconstruir escenas en 3D a partir de fotos y generar vistas nuevas desde cualquier ángulo.
desc: El neural rendering aprende una representación de la escena a partir de fotos tomadas desde varios puntos de vista, con las posiciones de la cámara conocidas o estimadas, y la usa para generar imágenes desde puntos de vista nuevos. NeRF representa la escena con una red neuronal y Gaussian Splatting, con millones de gaussianas 3D. Se usa en mapas inmersivos, efectos visuales, patrimonio, comercio electrónico y robótica.
ej: Con un vídeo grabado alrededor de una estatua, generar una visita virtual que se puede recorrer desde cualquier ángulo.
rel: nerf, gaussian-splatting, estimacion-de-pose-y-profundidad
en: 3D vision & neural rendering

## nerf
frase: Una red neuronal que aprende el color y la densidad de cada punto del espacio para renderizar la escena desde cualquier ángulo.
desc: Un Neural Radiance Field es una red que recibe una posición 3D y una dirección de vista y devuelve un color y una densidad. Para formar cada píxel se lanza un rayo desde la cámara, se consulta la red en puntos a lo largo del rayo y se acumulan los colores según la densidad (renderizado volumétrico). Se entrena comparando las imágenes renderizadas con las fotos. Su calidad es excelente, pero entrenar y renderizar es lento.
cuando: Para reconstruir escenas estáticas con gran calidad visual cuando la velocidad de renderizado no es crítica.
ej: Con 100 fotos de una habitación se entrena un NeRF que genera un vídeo de vuelo por la habitación con reflejos realistas.
py: `ns-train nerfacto --data ./escena`
rel: vision-3d-y-neural-rendering, gaussian-splatting, mlp, codificacion-posicional
en: NeRF (neural radiance fields)

## gaussian-splatting
frase: Representar la escena con millones de gaussianas 3D de colores que se proyectan sobre la imagen en tiempo real.
desc: Cada gaussiana tiene una posición, una forma, una opacidad y un color que depende del punto de vista, y todas se optimizan a partir de las fotos. Para renderizar, se proyectan y se mezclan sobre la imagen (rasterización), sin recorrer rayos con una red, lo que permite trabajar en tiempo real y en alta resolución con una calidad igual o superior a la de NeRF. Se ha convertido en la técnica dominante de reconstrucción 3D fotorrealista.
cuando: Para reconstrucciones 3D fotorrealistas que deban visualizarse en tiempo real, como visitas virtuales o realidad aumentada.
ej: Una inmobiliaria escanea un piso con el móvil y lo publica como una visita 3D navegable desde el navegador.
py: `ns-train splatfacto --data ./escena`
rel: nerf, vision-3d-y-neural-rendering, gmm
en: Gaussian Splatting
