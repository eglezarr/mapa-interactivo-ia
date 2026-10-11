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
    partes = ruta.split(".")
    for i in range(len(partes), 0, -1):
        try:
            obj = importlib.import_module(".".join(partes[:i]))
        except Exception:
            continue
        for attr in partes[i:]:
            obj = getattr(obj, attr)          # AttributeError si no existe
        return obj
    raise ImportError(f"no se pudo importar ningún prefijo de {ruta}")


def parametros(obj):
    """(nombres de argumentos, acepta **kwargs) de un objeto de Python, o None si no hay firma."""
    try:
        sig = inspect.signature(obj)
    except (TypeError, ValueError):
        return None
    nombres = {p.name for p in sig.parameters.values() if p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL)}
    hay_kwargs = any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())
    return (nombres, hay_kwargs)


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
                    mod = ".".join(partes[:-1] if partes[-1] == "__init__" else partes)
                    try:
                        arbol = ast.parse(f.read_text(encoding="utf-8", errors="ignore"))
                    except SyntaxError:
                        continue
                    mods[mod] = {n.name: n for n in arbol.body
                                 if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))}
                break
            self.cache[paquete] = mods
        return self.cache[paquete]

    def buscar(self, ruta):
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
        cands = sorted((m for m in mods if (m == pref or m.startswith(pref + ".")) and resto[0] in mods[m]),
                       key=lambda m: (m.count("."), m))
        if not cands:
            raise AttributeError(f"{resto[0]} no está definido en {pref}")
        nodo = mods[cands[0]][resto[0]]
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
