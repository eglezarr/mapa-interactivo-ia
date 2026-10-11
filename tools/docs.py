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
})


def url_para(ruta):
    if ruta in MANUAL:
        return MANUAL[ruta]
    lib = ruta.split(".")[0]
    if ruta.startswith("torch.nn.init."):            # una sola página con un ancla por función
        return f"https://docs.pytorch.org/docs/stable/nn.init.html#{ruta}"
    if ruta.startswith("torch.utils.data."):
        return f"https://docs.pytorch.org/docs/stable/data.html#{ruta}"
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
