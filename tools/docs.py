"""Enlaces a la documentación oficial de cada referencia de Python.

Genera datos/docs.json  {ruta_python: url}  importando cada objeto (necesita las librerías instaladas),
así el build no depende de tenerlas: solo lee esa caché.
Reglas por librería verificadas con páginas reales; los casos especiales van en MANUAL.
Uso: python3 tools/docs.py
"""
import importlib
import inspect
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arbol import RAIZ  # noqa: E402

# Casos que no siguen un patrón (se amplía según se añaden librerías)
MANUAL = {
    "keras.Sequential": "https://keras.io/api/models/sequential/",
    "torch.nn.Module": "https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html",
    "pygam.LinearGAM": "https://github.com/dswah/pyGAM",
    "interpret.glassbox.ExplainableBoostingClassifier": "https://interpret.ml/docs/ebm.html",
    "interpret.glassbox.ExplainableBoostingRegressor": "https://interpret.ml/docs/ebm.html",
    "scipy.stats": "https://docs.scipy.org/doc/scipy/reference/stats.html",
    "keras.losses.KLDivergence": "https://keras.io/api/losses/probabilistic_losses/#kldivergence-class",
    "pgmpy.models.DiscreteBayesianNetwork":
        "https://pgmpy.org/api/generated/models/pgmpy.models.DiscreteBayesianNetwork.html",
    "pgmpy.inference.VariableElimination":
        "https://pgmpy.org/api/generated/inference/pgmpy.inference.VariableElimination.html",
    "pgmpy.causal_discovery.PC":
        "https://pgmpy.org/api/generated/structure_learning/pgmpy.causal_discovery.PC.html",
    "pgmpy.causal_discovery.HillClimbSearch":
        "https://pgmpy.org/api/generated/structure_learning/pgmpy.causal_discovery.HillClimbSearch.html",
    "causallearn.search.ConstraintBased.PC.pc": "https://causal-learn.readthedocs.io/en/latest/search_methods_index/"
        "Constraint-based%20causal%20discovery%20methods/PC.html",
    "causallearn.search.ScoreBased.GES.ges": "https://causal-learn.readthedocs.io/en/latest/search_methods_index/"
        "Score-based%20causal%20discovery%20methods/GES.html",
    "causallearn.search.FCMBased.lingam.DirectLiNGAM": "https://causal-learn.readthedocs.io/en/latest/search_methods_index/"
        "Causal%20discovery%20methods%20based%20on%20constrained%20functional%20causal%20models/lingam.html",
    "causallearn.search.FCMBased.ANM.ANM.ANM": "https://causal-learn.readthedocs.io/en/latest/search_methods_index/"
        "Causal%20discovery%20methods%20based%20on%20constrained%20functional%20causal%20models/anm.html",
}


def resolver(ruta):
    partes = ruta.split(".")
    for i in range(len(partes), 0, -1):
        try:
            obj = importlib.import_module(".".join(partes[:i]))
        except Exception:
            continue
        for attr in partes[i:]:
            obj = getattr(obj, attr)
        return obj
    raise ImportError(ruta)


# Librerías cuya página de documentación se deduce de la ruta sin importar nada
# (cada patrón comprobado con al menos una página real)
PATRONES = {
    "xgboost": "https://xgboost.readthedocs.io/en/stable/python/python_api.html#{ruta}",
    "lightgbm": "https://lightgbm.readthedocs.io/en/stable/pythonapi/{ruta}.html",
    "torch": "https://docs.pytorch.org/docs/stable/generated/{ruta}.html",
    "scipy": "https://docs.scipy.org/doc/scipy/reference/generated/{ruta}.html",
    "numpy": "https://numpy.org/doc/stable/reference/generated/{ruta}.html",
    "pandas": "https://pandas.pydata.org/docs/reference/api/{ruta}.html",
    "imblearn": "https://imbalanced-learn.org/stable/references/generated/{ruta}.html",
    "hmmlearn": "https://hmmlearn.readthedocs.io/en/latest/api.html#{ruta}",
    "hyppo": "https://hyppo.neurodata.io/api/generated/{ruta}.html",
    "pingouin": "https://pingouin-stats.org/build/html/generated/{ruta}.html",
    "seaborn": "https://seaborn.pydata.org/generated/{ruta}.html",
    "jax": "https://docs.jax.dev/en/latest/_autosummary/{ruta}.html",
    "joblib": "https://joblib.readthedocs.io/en/stable/generated/{ruta}.html",
    "torchvision": "https://docs.pytorch.org/vision/stable/generated/{ruta}.html",
}

# Páginas de documentación sueltas, comprobadas una a una
MANUAL.update({
    "wandb.init": "https://docs.wandb.ai/ref/python/functions/init",
    "transformers.pipeline": "https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.pipeline",
    "huggingface_hub.hf_hub_download":
        "https://huggingface.co/docs/huggingface_hub/package_reference/file_download#huggingface_hub.hf_hub_download",
    "datasets.load_dataset": "https://huggingface.co/docs/datasets/package_reference/loading_methods#datasets.load_dataset",
    "onnxruntime.InferenceSession": "https://onnxruntime.ai/docs/api/python/api_summary.html#onnxruntime.InferenceSession",
    "cleanlab.filter.find_label_issues":
        "https://docs.cleanlab.ai/stable/cleanlab/filter.html#cleanlab.filter.find_label_issues",
    "flaml.AutoML": "https://microsoft.github.io/FLAML/docs/reference/automl/automl/",
    "autokeras.ImageClassifier": "https://autokeras.com/image_classifier/",
    "autosklearn.classification.AutoSklearnClassifier":
        "https://automl.github.io/auto-sklearn/master/api.html#autosklearn.classification.AutoSklearnClassifier",
    "fastapi.FastAPI": "https://fastapi.tiangolo.com/reference/fastapi/",
    "umap.UMAP": "https://umap-learn.readthedocs.io/en/latest/api.html#umap.umap_.UMAP",
    "sklearn_extra.cluster.KMedoids":
        "https://scikit-learn-extra.readthedocs.io/en/stable/generated/sklearn_extra.cluster.KMedoids.html",
    "kmedoids.KMedoids": "https://python-kmedoids.readthedocs.io/en/latest/#kmedoids.KMedoids",
})


# keras.io organiza las páginas por categorías (tomado de los índices de keras.io/api)
KERAS_CAPAS = {
    "core_layers": "Input:input Dense:dense EinsumDense:einsum_dense Activation:activation Embedding:embedding "
                   "Masking:masking Lambda:lambda Identity:identity",
    "convolution_layers": "Conv1D:convolution1d Conv2D:convolution2d Conv3D:convolution3d "
                          "SeparableConv1D:separable_convolution1d SeparableConv2D:separable_convolution2d "
                          "DepthwiseConv1D:depthwise_convolution1d DepthwiseConv2D:depthwise_convolution2d "
                          "Conv1DTranspose:convolution1d_transpose Conv2DTranspose:convolution2d_transpose "
                          "Conv3DTranspose:convolution3d_transpose",
    "pooling_layers": "MaxPooling1D:max_pooling1d MaxPooling2D:max_pooling2d MaxPooling3D:max_pooling3d "
                      "AveragePooling1D:average_pooling1d AveragePooling2D:average_pooling2d "
                      "GlobalMaxPooling1D:global_max_pooling1d GlobalMaxPooling2D:global_max_pooling2d "
                      "GlobalAveragePooling1D:global_average_pooling1d GlobalAveragePooling2D:global_average_pooling2d",
    "recurrent_layers": "LSTM:lstm GRU:gru SimpleRNN:simple_rnn TimeDistributed:time_distributed "
                        "Bidirectional:bidirectional ConvLSTM2D:conv_lstm2d RNN:rnn",
    "normalization_layers": "BatchNormalization:batch_normalization LayerNormalization:layer_normalization "
                            "UnitNormalization:unit_normalization GroupNormalization:group_normalization "
                            "RMSNormalization:rms_normalization",
    "regularization_layers": "Dropout:dropout SpatialDropout1D:spatial_dropout1d SpatialDropout2D:spatial_dropout2d "
                             "GaussianDropout:gaussian_dropout AlphaDropout:alpha_dropout GaussianNoise:gaussian_noise "
                             "ActivityRegularization:activity_regularization",
    "attention_layers": "GroupQueryAttention:group_query_attention MultiHeadAttention:multi_head_attention "
                        "Attention:attention AdditiveAttention:additive_attention",
    "reshaping_layers": "Reshape:reshape Flatten:flatten RepeatVector:repeat_vector Permute:permute "
                        "Cropping2D:cropping2d UpSampling1D:up_sampling1d UpSampling2D:up_sampling2d "
                        "ZeroPadding1D:zero_padding1d ZeroPadding2D:zero_padding2d",
    "merging_layers": "Concatenate:concatenate Average:average Maximum:maximum Minimum:minimum Add:add "
                      "Subtract:subtract Multiply:multiply Dot:dot",
    "activation_layers": "ReLU:relu Softmax:softmax LeakyReLU:leaky_relu PReLU:prelu ELU:elu",
    "preprocessing_layers/image_augmentation": "RandomFlip:random_flip RandomRotation:random_rotation "
                                               "RandomZoom:random_zoom RandomCrop:random_crop RandomContrast:random_contrast",
    "preprocessing_layers/numerical": "Normalization:normalization",
    "preprocessing_layers/text": "TextVectorization:text_vectorization",
}
KERAS = {f"keras.layers.{c}": f"https://keras.io/api/layers/{cat}/{p}/"
         for cat, pares in KERAS_CAPAS.items() for c, p in (x.split(":") for x in pares.split())}
KERAS.update({f"keras.optimizers.{c}": f"https://keras.io/api/optimizers/{p}/" for c, p in (
    x.split(":") for x in "SGD:sgd RMSprop:rmsprop Adam:adam AdamW:adamw Adagrad:adagrad Adadelta:adadelta "
                          "Nadam:Nadam Lion:lion Muon:muon".split())})
KERAS.update({f"keras.callbacks.{c}": f"https://keras.io/api/callbacks/{p}/" for c, p in (
    x.split(":") for x in "ModelCheckpoint:model_checkpoint TensorBoard:tensorboard EarlyStopping:early_stopping "
                          "LearningRateScheduler:learning_rate_scheduler ReduceLROnPlateau:reduce_lr_on_plateau "
                          "CSVLogger:csv_logger LambdaCallback:lambda_callback Callback:base_callback".split())})
for _c in "MeanSquaredError MeanAbsoluteError Huber LogCosh".split():
    KERAS[f"keras.losses.{_c}"] = f"https://keras.io/api/losses/regression_losses/#{_c.lower()}-class"
for _c in "BinaryCrossentropy BinaryFocalCrossentropy CategoricalCrossentropy SparseCategoricalCrossentropy Poisson KLDivergence".split():
    KERAS[f"keras.losses.{_c}"] = f"https://keras.io/api/losses/probabilistic_losses/#{_c.lower()}-class"
for _f in "relu sigmoid tanh gelu softmax leaky_relu elu silu".split():
    KERAS[f"keras.activations.{_f}"] = f"https://keras.io/api/layers/activations/#{_f.replace('_', '')}-function"
for _c in "HeNormal HeUniform GlorotNormal GlorotUniform".split():
    KERAS[f"keras.initializers.{_c}"] = f"https://keras.io/api/layers/initializers/#{_c.lower()}-class"
for _c in "L1 L2 L1L2".split():
    KERAS[f"keras.regularizers.{_c}"] = f"https://keras.io/api/layers/regularizers/#{_c.lower()}-class"
for _c, _p in (("ExponentialDecay", "exponential_decay"), ("CosineDecay", "cosine_decay"),
               ("CosineDecayRestarts", "cosine_decay_restarts"), ("PiecewiseConstantDecay", "piecewise_constant_decay")):
    KERAS[f"keras.optimizers.schedules.{_c}"] = f"https://keras.io/api/optimizers/learning_rate_schedules/{_p}/"
KERAS.update({
    "keras.losses": "https://keras.io/api/losses/",
    "keras.losses.Loss": "https://keras.io/api/losses/#loss-class",
    "keras.Model": "https://keras.io/api/models/model/",
    "keras.Input": "https://keras.io/api/layers/core_layers/input/",
    "keras.Model.fit": "https://keras.io/api/models/model_training_apis/#fit-method",
    "keras.Model.compile": "https://keras.io/api/models/model_training_apis/#compile-method",
    "keras.layers.Layer": "https://keras.io/api/layers/base_layer/#layer-class",
    "keras.utils.set_random_seed": "https://keras.io/api/utils/python_utils/#setrandomseed-function",
    "keras.applications.VGG19": "https://keras.io/api/applications/vgg/",
    "keras.applications.VGG16": "https://keras.io/api/applications/vgg/",
})
MANUAL.update(KERAS)
MANUAL.update({
    "torch.nn.functional": "https://docs.pytorch.org/docs/stable/nn.functional.html",
    "lpips.LPIPS": "https://github.com/richzhang/PerceptualSimilarity",
    "compressai.layers.GDN": "https://interdigitalinc.github.io/CompressAI/layers.html#compressai.layers.GDN",
})
for _c, _p in (("TransformerEncoder", "transformer_encoder"), ("TransformerDecoder", "transformer_decoder"),
               ("SinePositionEncoding", "sine_position_encoding"), ("PositionEmbedding", "position_embedding"),
               ("RotaryEmbedding", "rotary_embedding"), ("TokenAndPositionEmbedding", "token_and_position_embedding")):
    MANUAL[f"keras_hub.layers.{_c}"] = f"https://keras.io/keras_hub/api/modeling_layers/{_p}/"
PATRONES.update({
    "reservoirpy": "https://reservoirpy.readthedocs.io/en/latest/api/generated/{ruta}.html",
})
KERAS_APPS = {"ResNet50": "resnet/resnet_models/#resnet50-function", "VGG16": "vgg/vgg_models/#vgg16-function",
              "VGG19": "vgg/vgg_models/#vgg19-function"}
MANUAL.update({f"keras.applications.{c}": f"https://keras.io/api/applications/{p}" for c, p in KERAS_APPS.items()})

SBERT = "https://sbert.net/docs/package_reference"
DIFFUSERS = "https://huggingface.co/docs/diffusers/api"
TORCHMETRICS = "https://lightning.ai/docs/torchmetrics/stable/image"
MANUAL.update({
    "sentence_transformers.SentenceTransformer": f"{SBERT}/sentence_transformer/model.html",
    "sentence_transformers.CrossEncoder": f"{SBERT}/cross_encoder/model.html",
    "sentence_transformers.util.semantic_search":
        f"{SBERT}/util/retrieval.html#sentence_transformers.util.retrieval.semantic_search",
    "sentence_transformers.sentence_transformer.losses.MultipleNegativesRankingLoss":
        f"{SBERT}/sentence_transformer/losses.html#multiplenegativesrankingloss",
    "gensim.models.KeyedVectors.most_similar": "https://radimrehurek.com/gensim/models/keyedvectors.html"
                                               "#gensim.models.keyedvectors.KeyedVectors.most_similar",
    "gensim.models.Word2Vec": "https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.Word2Vec",
    "lightly.loss.NTXentLoss":
        "https://docs.lightly.ai/self-supervised-learning/lightly.loss.html#lightly.loss.ntx_ent_loss.NTXentLoss",
    "timm.create_model": "https://huggingface.co/docs/timm/reference/models#timm.create_model",
    "transformers.Trainer": "https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Trainer",
    "transformers.TrainingArguments":
        "https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments",
    "transformers.ViTMAEConfig": "https://huggingface.co/docs/transformers/model_doc/vit_mae#transformers.ViTMAEConfig",
    "transformers.ViTMAEForPreTraining":
        "https://huggingface.co/docs/transformers/model_doc/vit_mae#transformers.ViTMAEForPreTraining",
    "numpy.random.Generator.uniform":
        "https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.uniform.html",
    "torch.distributions.Normal": "https://docs.pytorch.org/docs/stable/distributions.html#torch.distributions.normal.Normal",
    "torch.distributions.Normal.rsample":
        "https://docs.pytorch.org/docs/stable/distributions.html#torch.distributions.normal.Normal.rsample",
    "torch.distributions.kl_divergence":
        "https://docs.pytorch.org/docs/stable/distributions.html#torch.distributions.kl.kl_divergence",
    "torch.distributions.TransformedDistribution": "https://docs.pytorch.org/docs/stable/distributions.html"
                                                   "#torch.distributions.transformed_distribution.TransformedDistribution",
    "sdv.single_table.GaussianCopulaSynthesizer":
        "https://docs.sdv.dev/sdv/single-table-data/modeling/synthesizers/gaussiancopulasynthesizer",
    "sdv.single_table.CTGANSynthesizer": "https://docs.sdv.dev/sdv/single-table-data/modeling/synthesizers/ctgansynthesizer",
    "prdc.compute_prdc": "https://github.com/clovaai/generative-evaluation-prdc",
    "torchmetrics.image.inception.InceptionScore": f"{TORCHMETRICS}/inception_score.html",
    "torchmetrics.image.fid.FrechetInceptionDistance": f"{TORCHMETRICS}/frechet_inception_distance.html",
    "torchmetrics.image.kid.KernelInceptionDistance": f"{TORCHMETRICS}/kernel_inception_distance.html",
})
for _c, _p in (("DiffusionPipeline", "pipelines/overview"), ("DDPMPipeline", "pipelines/ddpm"),
               ("StableDiffusionPipeline", "pipelines/stable_diffusion/text2img"),
               ("StableDiffusion3Pipeline", "pipelines/stable_diffusion/stable_diffusion_3"),
               ("StableDiffusionInpaintPipeline", "pipelines/stable_diffusion/inpaint"),
               ("StableDiffusionUpscalePipeline", "pipelines/stable_diffusion/upscale"),
               ("StableDiffusionControlNetPipeline", "pipelines/controlnet"),
               ("AutoPipelineForText2Image", "pipelines/auto_pipeline"), ("CogVideoXPipeline", "pipelines/cogvideox"),
               ("WanPipeline", "pipelines/wan"), ("FluxPipeline", "pipelines/flux"),
               ("UNet2DModel", "models/unet2d"), ("AutoencoderKL", "models/autoencoderkl"),
               ("ControlNetModel", "models/controlnet"), ("DDPMScheduler", "schedulers/ddpm"),
               ("DDIMScheduler", "schedulers/ddim"), ("LCMScheduler", "schedulers/lcm"),
               ("FlowMatchEulerDiscreteScheduler", "schedulers/flow_match_euler_discrete")):
    MANUAL[f"diffusers.{_c}"] = f"{DIFFUSERS}/{_p}#diffusers.{_c}"
for _r in ("nflows.flows.base.Flow", "nflows.transforms.autoregressive.MaskedAffineAutoregressiveTransform",
           "nflows.transforms.coupling.AffineCouplingTransform"):
    MANUAL[_r] = "https://github.com/bayesiains/nflows"
for _r in ("faiss.IndexFlatIP", "faiss.IndexFlatL2", "faiss.IndexHNSWFlat", "faiss.IndexIVFFlat", "faiss.IndexIVFPQ"):
    MANUAL[_r] = "https://github.com/facebookresearch/faiss/wiki/Faiss-indexes"

# Hugging Face: transformers, trl, peft, tokenizers, smolagents (una página por tema, ancla por clase)
HF = "https://huggingface.co/docs"
for _c, _p in (("DataCollatorForLanguageModeling", "main_classes/data_collator"),
               ("GenerationConfig", "main_classes/text_generation"),
               ("BitsAndBytesConfig", "main_classes/quantization"), ("GPTQConfig", "main_classes/quantization"),
               ("AwqConfig", "main_classes/quantization"), ("BertModel", "model_doc/bert"),
               ("GPT2LMHeadModel", "model_doc/gpt2"), ("T5ForConditionalGeneration", "model_doc/t5")):
    MANUAL[f"transformers.{_c}"] = f"{HF}/transformers/{_p}#transformers.{_c}"
for _c, _p in (("SFTTrainer", "sft_trainer"), ("SFTConfig", "sft_trainer"), ("DPOTrainer", "dpo_trainer"),
               ("DPOConfig", "dpo_trainer"), ("RewardTrainer", "reward_trainer"), ("RewardConfig", "reward_trainer"),
               ("GRPOTrainer", "grpo_trainer"), ("GRPOConfig", "grpo_trainer")):
    MANUAL[f"trl.{_c}"] = f"{HF}/trl/{_p}#trl.{_c}"
for _c, _p in (("LoraConfig", "lora"), ("get_peft_model", "peft_model"),
               ("prepare_model_for_kbit_training", "peft_model"), ("PromptTuningConfig", "prompt_tuning")):
    MANUAL[f"peft.{_c}"] = f"{HF}/peft/package_reference/{_p}#peft.{_c}"
MANUAL.update({
    "tokenizers.models.BPE": f"{HF}/tokenizers/api/models#tokenizers.models.BPE",
    "tokenizers.trainers.BpeTrainer": f"{HF}/tokenizers/api/trainers#tokenizers.trainers.BpeTrainer",
    "tokenizers.Tokenizer": f"{HF}/tokenizers/api/tokenizer#tokenizers.Tokenizer",
    "smolagents.CodeAgent": f"{HF}/smolagents/reference/agents#smolagents.CodeAgent",
    "smolagents.ToolCallingAgent": f"{HF}/smolagents/reference/agents#smolagents.ToolCallingAgent",
    "smolagents.tool": f"{HF}/smolagents/reference/tools#smolagents.tool",
    "tiktoken.get_encoding": "https://github.com/openai/tiktoken",
    "sentencepiece.SentencePieceProcessor": "https://github.com/google/sentencepiece",
    "nltk.tokenize.word_tokenize": "https://www.nltk.org/api/nltk.tokenize.word_tokenize.html",
    "nltk.stem.SnowballStemmer": "https://www.nltk.org/api/nltk.stem.snowball.html#nltk.stem.snowball.SnowballStemmer",
    "gensim.models.FastText": "https://radimrehurek.com/gensim/models/fasttext.html#gensim.models.fasttext.FastText",
    "gensim.downloader.load": "https://radimrehurek.com/gensim/downloader.html#gensim.downloader.load",
    "adapters.SeqBnConfig": "https://github.com/adapter-hub/adapters",
    "stable_baselines3.PPO": "https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html",
    "openai.OpenAI": "https://github.com/openai/openai-python",
    "pydantic.BaseModel": "https://docs.pydantic.dev/latest/api/base_model/",
    "mcp.server.mcpserver.MCPServer": "https://py.sdk.modelcontextprotocol.io/v2/",
    "chromadb.PersistentClient": "https://docs.trychroma.com/reference/python/client",
    "rank_bm25.BM25Okapi": "https://github.com/dorianbrown/rank_bm25",
})
# LangChain y LangGraph: URLs tomadas de resultados de búsqueda de su documentación oficial
LC_DOCS, LC_REF = "https://docs.langchain.com/oss", "https://reference.langchain.com/python"
MANUAL.update({
    "langchain.agents.create_agent": f"{LC_DOCS}/langchain/agents",
    "langchain_core.tools.tool": f"{LC_REF}/langchain-core/tools",
    "langchain_text_splitters.RecursiveCharacterTextSplitter":
        f"{LC_DOCS}/python/integrations/splitters/recursive_text_splitter",
    "langchain_core.vectorstores.InMemoryVectorStore":
        f"{LC_REF}/langchain-core/vectorstores/in_memory/InMemoryVectorStore",
    "langgraph.graph.StateGraph": f"{LC_DOCS}/python/langgraph/graph-api",
    "langgraph.checkpoint.memory.InMemorySaver": f"{LC_REF}/langgraph.checkpoint/memory/InMemorySaver",
    "langgraph.types.interrupt": f"{LC_REF}/langgraph/types/interrupt",
    "langchain_classic.retrievers.EnsembleRetriever": f"{LC_REF}/langchain-classic/retrievers/ensemble",
    "langchain_classic.retrievers.MultiQueryRetriever": f"{LC_REF}/langchain_classic/retrievers/multi_query/",
    "langchain_community.document_transformers.LongContextReorder": "https://github.com/langchain-ai/langchain-community",
})
for _c in ("HumanInTheLoopMiddleware", "PIIMiddleware", "SummarizationMiddleware", "ModelCallLimitMiddleware",
           "ToolCallLimitMiddleware"):
    MANUAL[f"langchain.agents.middleware.{_c}"] = f"{LC_DOCS}/python/langchain/middleware/overview"
for _c in ("Faithfulness", "ContextRecall", "ContextPrecision", "AnswerRelevancy", "ToolCallAccuracy",
           "AgentGoalAccuracyWithReference"):
    MANUAL[f"ragas.metrics.collections.{_c}"] = "https://docs.ragas.io/"

# Visión, multimodal y audio
for _c, _p in (("InceptionV3", "inceptionv3/"), ("DenseNet121", "densenet/#densenet121-function"),
               ("MobileNetV2", "mobilenet/#mobilenetv2-function"), ("EfficientNetB0", "efficientnet/#efficientnetb0-function"),
               ("ConvNeXtTiny", "convnext/#convnexttiny-function")):
    MANUAL[f"keras.applications.{_c}"] = f"https://keras.io/api/applications/{_p}"
for _c, _p in (("SamModel", "sam"), ("SiglipModel", "siglip"), ("LlavaForConditionalGeneration", "llava"),
               ("WhisperForConditionalGeneration", "whisper"), ("VitsModel", "vits"), ("ClapModel", "clap"),
               ("MusicgenForConditionalGeneration", "musicgen"), ("ViTForImageClassification", "vit"),
               ("CLIPModel", "clip"), ("CLIPProcessor", "clip")):
    MANUAL[f"transformers.{_c}"] = f"{HF}/transformers/model_doc/{_p}#transformers.{_c}"
for _c, _p in (("LTXPipeline", "pipelines/ltx_video"), ("CogVideoXImageToVideoPipeline", "pipelines/cogvideox")):
    MANUAL[f"diffusers.{_c}"] = f"{DIFFUSERS}/{_p}#diffusers.{_c}"
MANUAL.update({
    "pywt.wavedec2": "https://pywavelets.readthedocs.io/en/latest/ref/2d-dwt-and-idwt.html",
    "pywt.dwt2": "https://pywavelets.readthedocs.io/en/latest/ref/2d-dwt-and-idwt.html",
    "ultralytics.YOLO": "https://docs.ultralytics.com/usage/python",
    "supervision.ByteTrack": "https://supervision.roboflow.com/trackers/",
    "segment_anything.SamPredictor": "https://github.com/facebookresearch/segment-anything",
    "segmentation_models_pytorch.Unet": "https://github.com/qubvel-org/segmentation_models.pytorch",
    "open_clip.create_model_and_transforms": "https://github.com/mlfoundations/open_clip",
})
PATRONES.update({
    "librosa": "https://librosa.org/doc/latest/generated/{ruta}.html",
    "torchaudio": "https://docs.pytorch.org/audio/stable/generated/{ruta}.html",
    "pyts": "https://pyts.readthedocs.io/en/stable/generated/{ruta}.html",
})

# Series, recomendación, anomalías y refuerzo
MANUAL.update({
    "keras.utils.timeseries_dataset_from_array":
        "https://keras.io/api/data_loading/timeseries/#timeseriesdatasetfromarray-function",
    "chronos.BaseChronosPipeline": "https://github.com/amazon-science/chronos-forecasting",
    "chronos.Chronos2Pipeline": "https://github.com/amazon-science/chronos-forecasting",
    "surprise.KNNBasic": "https://surprise.readthedocs.io/en/stable/knn_inspired.html",
    "surprise.SVD": "https://surprise.readthedocs.io/en/stable/matrix_factorization.html",
    "implicit.als.AlternatingLeastSquares": "https://github.com/benfred/implicit",
    "lightfm.LightFM": "https://github.com/lyst/lightfm",
    "keras_rs.layers.FeatureCross": "https://keras.io/keras_rs/api/feature_interaction_layers/feature_cross/",
    "keras_rs.layers.BruteForceRetrieval": "https://keras.io/keras_rs/api/retrieval_layers/brute_force_retrieval/",
    "keras_rs.metrics.NDCG": "https://keras.io/keras_rs/api/metrics/ndcg/",
    "keras_rs.metrics.RecallAtK": "https://keras.io/keras_rs/api/metrics/",
    "gymnasium.make": "https://gymnasium.farama.org/api/registry/#gymnasium.make",
    "gymnasium.Env": "https://gymnasium.farama.org/api/env/",
    "numpy.random.Generator.beta": "https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.beta.html",
})
for _r in ("pyod.models.knn.KNN", "pyod.models.ecod.ECOD", "pyod.models.pca.PCA", "pyod.models.auto_encoder.AutoEncoder",
           "pyod.models.iforest.IForest", "pyod.models.vae.VAE"):
    MANUAL[_r] = "https://pyod.readthedocs.io/"
for _r in ("mabwiser.mab.MAB", "mabwiser.mab.LearningPolicy.EpsilonGreedy", "mabwiser.mab.LearningPolicy.UCB1",
           "mabwiser.mab.LearningPolicy.ThompsonSampling", "mabwiser.mab.LearningPolicy.LinUCB"):
    MANUAL[_r] = "https://github.com/fidelity/mabwiser"


def url_para(ruta):
    if ruta in MANUAL:
        return MANUAL[ruta]
    lib = ruta.split(".")[0]
    if ruta.startswith("torch.nn.init."):            # una sola página con un ancla por función
        return f"https://docs.pytorch.org/docs/stable/nn.init.html#{ruta}"
    if ruta.startswith("torch.utils.data."):
        return f"https://docs.pytorch.org/docs/stable/data.html#{ruta}"
    partes = ruta.split(".")
    if lib == "torch" and not ruta.startswith("torch.Tensor.") and len(partes) > 2 \
            and partes[-2][:1].isupper() and partes[-1][:1].islower():   # método de una clase: ancla en su página
        return f"https://docs.pytorch.org/docs/stable/generated/{'.'.join(partes[:-1])}.html#{ruta}"
    if lib == "skimage":                              # una página por módulo
        return f"https://scikit-image.org/docs/stable/api/{'.'.join(partes[:-1])}.html#{ruta}"
    if ruta.startswith("transformers.Auto"):
        return f"https://huggingface.co/docs/transformers/model_doc/auto#{ruta}"
    if ruta.startswith("torchvision.models.") and partes[2][:1].islower():     # constructores de modelos
        return f"https://docs.pytorch.org/vision/stable/models/generated/{ruta}.html"
    if lib == "tslearn":                              # una carpeta por submódulo
        return f"https://tslearn.readthedocs.io/en/stable/gen_modules/{partes[1]}/{ruta}.html"
    if lib in PATRONES:
        return PATRONES[lib].format(ruta=ruta)
    if lib == "mlflow":          # una página por módulo, con un ancla por función
        return f"https://mlflow.org/docs/latest/api_reference/python_api/{ruta.rsplit('.', 1)[0]}.html#{ruta}"
    if lib == "optuna":
        sub = "samplers/" if ruta.startswith("optuna.samplers.") else ""
        return f"https://optuna.readthedocs.io/en/stable/reference/{sub}generated/{ruta}.html"
    obj = resolver(ruta)
    if lib == "sklearn":
        if inspect.ismodule(obj):
            return f"https://scikit-learn.org/stable/api/{ruta}.html"
        return f"https://scikit-learn.org/stable/modules/generated/{ruta}.html"
    if lib == "statsmodels":
        if ruta.startswith("statsmodels.formula.api."):
            return f"https://www.statsmodels.org/stable/generated/{ruta}.html"
        canon = f"{obj.__module__}.{obj.__qualname__}"          # p. ej. statsmodels.api.OLS → ...linear_model.OLS
        return f"https://www.statsmodels.org/stable/generated/{canon}.html"
    return None


if __name__ == "__main__":
    sys.path.insert(0, str(RAIZ / "tools"))
    from verificar_py import rutas  # noqa: E402
    mapa = json.loads((RAIZ / "datos" / "mapa.json").read_text(encoding="utf-8"))
    cache_f = RAIZ / "datos" / "docs.json"
    cache = json.loads(cache_f.read_text(encoding="utf-8")) if cache_f.exists() else {}
    faltan = []
    for _, ruta in rutas(mapa):
        if ruta in cache:
            continue
        try:
            u = url_para(ruta)
        except Exception as e:  # noqa: BLE001
            u = None
            print(f"  ✗ {ruta}: {type(e).__name__}", file=sys.stderr)
        if u:
            cache[ruta] = u
        else:
            faltan.append(ruta)
    cache_f.write_text(json.dumps(dict(sorted(cache.items())), ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"docs.json: {len(cache)} enlaces · sin regla: {faltan}")
