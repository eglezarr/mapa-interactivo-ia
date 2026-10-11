"""Comprueba las referencias de Python y los hiperparámetros de las fichas contra las librerías reales.

1) Cada ruta (p. ej. sklearn.ensemble.RandomForestClassifier) debe importarse y resolverse.
2) Cada hiperparámetro de la ficha debe ser un argumento real de alguna de sus clases de Python
   (se omite si la clase solo acepta **kwargs). "a / b" se comprueba como dos parámetros.
Si una librería no se puede importar (p. ej. PyTorch, que no se instala en este entorno), se recurre
a una comprobación estática: se lee su código fuente sin ejecutarlo y se buscan la clase o función
y sus argumentos. Las rutas de código fuente extra se pasan en la variable FUENTES_PY (separadas por ":").
Los fragmentos entre `comillas invertidas` no se verifican.
Uso: python3 tools/verificar_py.py [paquete_raiz ...]   (sin argumentos: todos)
"""
import ast
import importlib
import inspect
import json
import os
import re
import site
import sys
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "jax")   # Keras 3 necesita un backend; JAX es el más ligero de instalar

RAIZ = Path(__file__).resolve().parent.parent


def _code(p):
    return p["c"] if isinstance(p, dict) else p


def rutas(mapa):
    """Genera (id, ruta) para cada referencia de Python que sea una ruta importable."""
    pila = [mapa]
    while pila:
        n = pila.pop()
        for p in n.get("py", []):
            c = _code(p)
            if c.startswith("`") or (isinstance(p, dict) and p.get("s")):
                continue
            m = re.match(r"^([A-Za-z_][\w]*(?:\.[A-Za-z_]\w*)*)", c)
            if m and "." in m.group(1):
                yield n["id"], m.group(1)
        pila.extend(n.get("children", []))


def resolver(ruta):
    """Resuelve la ruta como lo haría `from paquete import Nombre`: primero el atributo, luego el submódulo.
    (Así pgmpy.causal_discovery.PC es la clase PC y no el submódulo PC.py que la define)."""
    partes = ruta.split(".")
    try:
        obj = importlib.import_module(partes[0])
    except Exception as e:  # noqa: BLE001
        raise ImportError(f"no se pudo importar {partes[0]}: {e}") from e
    for i, attr in enumerate(partes[1:], start=1):
        sub = ".".join(partes[:i + 1])
        try:
            obj = getattr(obj, attr)
        except AttributeError:
            try:
                obj = importlib.import_module(sub)
            except ModuleNotFoundError as e:
                if e.name and (sub == e.name or sub.startswith(e.name + ".")):
                    raise AttributeError(f"{attr} no existe en {'.'.join(partes[:i])}") from e
                raise ImportError(f"falta la dependencia {e.name}") from e   # p. ej. torch
            except Exception as e:  # noqa: BLE001
                raise ImportError(f"error al importar {sub}: {e}") from e
        except Exception as e:  # noqa: BLE001  (módulos perezosos que fallan al cargar un submódulo)
            raise ImportError(f"error al cargar {sub}: {e}") from e
        if getattr(obj, "_backends", None):       # objeto de relleno de transformers/diffusers sin PyTorch
            raise ImportError(f"{sub} necesita {obj._backends}")
    return obj


def parametros(obj):
    """(nombres de argumentos, acepta **kwargs) de un objeto de Python, o None si no hay firma."""
    try:
        sig = inspect.signature(obj)
    except (TypeError, ValueError):
        return None
    nombres = {p.name for p in sig.parameters.values() if p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL)}
    hay_kwargs = any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())
    solo_args = not nombres and any(p.kind == p.VAR_POSITIONAL for p in sig.parameters.values())  # p. ej. clases SWIG de faiss
    return (nombres, hay_kwargs or solo_args)


class Estatico:
    """Índice de clases y funciones leído del código fuente, sin importar la librería."""

    def __init__(self):
        extra = [Path(p) for p in os.environ.get("FUENTES_PY", "").split(":") if p]
        self.raices = extra + [Path(p) for p in site.getsitepackages()] + [Path(site.getusersitepackages())]
        self.cache = {}

    def _modulos(self, paquete):
        if paquete not in self.cache:
            mods = {}
            for raiz in self.raices:
                base = raiz / paquete
                if not base.is_dir():
                    continue
                for f in base.rglob("*.py"):
                    partes = f.relative_to(raiz).with_suffix("").parts
                    es_paquete = partes[-1] == "__init__"
                    mod = ".".join(partes[:-1] if es_paquete else partes)
                    try:
                        arbol = ast.parse(f.read_text(encoding="utf-8", errors="ignore"))
                    except SyntaxError:
                        continue
                    defs = {}
                    for n in arbol.body:
                        if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                            defs[n.name] = n
                        elif isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
                            defs.setdefault(n.targets[0].id, n)    # p. ej. funciones de C con docstring en PyTorch
                        elif isinstance(n, ast.ImportFrom):        # reexportaciones: from .estimators import X
                            base_mod = mod.split(".") if es_paquete else mod.split(".")[:-1]
                            if n.level:
                                base_mod = base_mod[:len(base_mod) - (n.level - 1)]
                                origen = ".".join(base_mod + ([n.module] if n.module else []))
                            else:
                                origen = n.module or ""
                            for a in n.names:
                                if a.name != "*":
                                    defs.setdefault(a.asname or a.name, ("importa", origen, a.name))
                    mods[mod] = defs
                break
            self.cache[paquete] = mods
        return self.cache[paquete]

    def buscar(self, ruta, _profundidad=0):
        partes = ruta.split(".")
        mods = self._modulos(partes[0])
        if not mods:
            raise ImportError(f"sin código fuente de {partes[0]}")
        i = next((i for i in range(len(partes), 0, -1) if ".".join(partes[:i]) in mods), 0)
        if i == 0:
            raise ImportError(f"no existe el módulo {partes[0]}")
        pref, resto = ".".join(partes[:i]), partes[i:]
        if not resto:
            return None                                   # es un módulo
        # la definición puede estar en un submódulo y reexportarse en el paquete (torch.nn.Conv2d → torch.nn.modules.conv)
        def rango(nodo):   # preferir definiciones reales a asignaciones, y estas a reexportaciones
            return 2 if isinstance(nodo, tuple) else 1 if isinstance(nodo, ast.Assign) else 0
        cands = sorted((m for m in mods if (m == pref or m.startswith(pref + ".")) and resto[0] in mods[m]),
                       key=lambda m: (rango(mods[m][resto[0]]), m.count("."), m))
        if not cands:
            raise AttributeError(f"{resto[0]} no está definido en {pref}")
        nodo = mods[cands[0]][resto[0]]
        if isinstance(nodo, tuple):                        # solo hay una reexportación: seguirla
            if _profundidad > 5:
                raise AttributeError(f"demasiadas reexportaciones para {ruta}")
            _, origen, nombre = nodo
            return self.buscar(".".join([origen, nombre] + resto[1:]), _profundidad + 1)
        for attr in resto[1:]:
            sub = [n for n in getattr(nodo, "body", []) if getattr(n, "name", None) == attr]
            if not sub:
                raise AttributeError(f"{attr} no está definido en {resto[0]}")
            nodo = sub[0]
        return nodo

    @staticmethod
    def parametros(nodo):
        if nodo is None:
            return None
        if isinstance(nodo, ast.Assign):
            # nombre = _add_docstr(c_func, "nombre(a, b=1, *, c=None) -> Tensor ...": la firma está en el docstring
            nombre = nodo.targets[0].id
            textos = [c.value for c in ast.walk(nodo.value) if isinstance(c, ast.Constant) and isinstance(c.value, str)]
            firmas = [m.group(1) for t in textos for m in re.finditer(rf"{re.escape(nombre)}\(([^)]*)\)", t)]
            if not firmas:
                return None
            firma = max(firmas, key=lambda f: f.count(","))     # la firma completa, no un ejemplo de uso
            nombres = {p.split("=")[0].strip().lstrip("*") for p in firma.split(",")}
            nombres.discard("")
            return (nombres, "**" in firma)
        if isinstance(nodo, ast.ClassDef):
            init = next((n for n in nodo.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"), None)
            if init is None:   # dataclass o configuración: campos anotados; los heredados no se ven → laxo
                campos = {n.target.id for n in nodo.body if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)}
                return (campos, bool(nodo.bases))
            nodo = init
        a = nodo.args
        nombres = {x.arg for x in a.posonlyargs + a.args + a.kwonlyargs} - {"self", "cls"}
        return (nombres, a.kwarg is not None)


if __name__ == "__main__":
    filtro = set(sys.argv[1:])
    mapa = json.loads((RAIZ / "datos" / "mapa.json").read_text(encoding="utf-8"))
    estatico = Estatico()
    fallos, ok, ok_est, hp_ok, hp_omit = [], 0, 0, 0, 0
    firmas = {}
    for nid, ruta in rutas(mapa):
        if filtro and ruta.split(".")[0] not in filtro:
            continue
        try:
            firmas.setdefault(nid, []).append((ruta, parametros(resolver(ruta))))
            ok += 1
        except ImportError:
            try:
                firmas.setdefault(nid, []).append((ruta, estatico.parametros(estatico.buscar(ruta))))
                ok_est += 1
            except Exception as e:  # noqa: BLE001
                fallos.append(f"{nid}: {ruta} → {type(e).__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            fallos.append(f"{nid}: {ruta} → {type(e).__name__}: {e}")

    pila = [mapa]
    while pila:
        n = pila.pop()
        pila.extend(n.get("children", []))
        if not n.get("hp") or n["id"] not in firmas:
            continue
        validas = [f for _, f in firmas[n["id"]] if f]
        nombres = set().union(*[f[0] for f in validas]) if validas else set()
        laxo = any(f[1] for f in validas) or not validas
        for param, _ in n["hp"]:
            for p in [x.strip() for x in param.split("/")]:
                if p in nombres:
                    hp_ok += 1
                elif laxo:
                    hp_omit += 1
                else:
                    fallos.append(f"{n['id']}: hiperparámetro '{p}' no existe en {[r for r, _ in firmas[n['id']]]}")
    print(f"rutas OK: {ok} importando + {ok_est} leyendo el código fuente · hiperparámetros OK: {hp_ok} · "
          f"no comprobables (**kwargs): {hp_omit} · fallos: {len(fallos)}")
    for f in fallos:
        print("  ✗", f)
