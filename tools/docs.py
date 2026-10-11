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
}


def url_para(ruta):
    if ruta in MANUAL:
        return MANUAL[ruta]
    lib = ruta.split(".")[0]
    if lib in PATRONES:
        return PATRONES[lib].format(ruta=ruta)
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
