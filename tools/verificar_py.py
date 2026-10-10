"""Comprueba las referencias de Python y los hiperparámetros de las fichas contra las librerías reales.

1) Cada ruta (p. ej. sklearn.ensemble.RandomForestClassifier) debe importarse y resolverse.
2) Cada hiperparámetro de la ficha debe ser un argumento real de alguna de sus clases de Python
   (se omite si la clase solo acepta **kwargs). "a / b" se comprueba como dos parámetros.
Los fragmentos entre `comillas invertidas` no se verifican.
Uso: python3 tools/verificar_py.py [paquete_raiz ...]   (sin argumentos: todos)
"""
import importlib
import inspect
import json
import re
import sys
from pathlib import Path

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
    """Nombres de argumentos aceptados (None si acepta **kwargs sin nombres explícitos útiles)."""
    try:
        sig = inspect.signature(obj)
    except (TypeError, ValueError):
        return None
    nombres = {p.name for p in sig.parameters.values() if p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL)}
    hay_kwargs = any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())
    return (nombres, hay_kwargs)


if __name__ == "__main__":
    filtro = set(sys.argv[1:])
    mapa = json.loads((RAIZ / "datos" / "mapa.json").read_text(encoding="utf-8"))
    fallos, ok, hp_ok, hp_omit = [], 0, 0, 0
    objetos = {}
    for nid, ruta in rutas(mapa):
        if filtro and ruta.split(".")[0] not in filtro:
            continue
        try:
            objetos.setdefault(nid, []).append(resolver(ruta))
            ok += 1
        except Exception as e:  # noqa: BLE001
            fallos.append(f"{nid}: {ruta} → {type(e).__name__}: {e}")

    pila = [mapa]
    while pila:
        n = pila.pop()
        pila.extend(n.get("children", []))
        if not n.get("hp") or n["id"] not in objetos:
            continue
        firmas = [parametros(o) for o in objetos[n["id"]]]
        firmas = [f for f in firmas if f]
        nombres = set().union(*[f[0] for f in firmas]) if firmas else set()
        laxo = any(f[1] for f in firmas)
        for param, _ in n["hp"]:
            for p in [x.strip() for x in param.split("/")]:
                if p in nombres:
                    hp_ok += 1
                elif laxo:
                    hp_omit += 1
                else:
                    fallos.append(f"{n['id']}: hiperparámetro '{p}' no existe en {[o.__name__ for o in objetos[n['id']]]}")
    print(f"rutas OK: {ok} · hiperparámetros OK: {hp_ok} · no comprobables (**kwargs): {hp_omit} · fallos: {len(fallos)}")
    for f in fallos:
        print("  ✗", f)
