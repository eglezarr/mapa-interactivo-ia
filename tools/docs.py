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


def url_para(ruta):
    if ruta in MANUAL:
        return MANUAL[ruta]
    lib = ruta.split(".")[0]
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
    if lib == "xgboost":
        return f"https://xgboost.readthedocs.io/en/stable/python/python_api.html#{ruta}"
    if lib == "lightgbm":
        return f"https://lightgbm.readthedocs.io/en/stable/pythonapi/{ruta}.html"
    if lib == "torch":
        return f"https://docs.pytorch.org/docs/stable/generated/{ruta}.html"
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
