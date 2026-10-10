"""Convierte esqueleto.md en un árbol con identificadores estables.

- Cada línea "- Nombre" del esqueleto es un nodo; la indentación (2 espacios) marca el nivel.
- Se eliminan la numeración de áreas ("10. ") y la marca "(+)" (solo era para la revisión).
- Un nodo puede fijar su id con "{#mi-id}" al final del nombre; si no, se genera un slug.
"""
import json
import re
import sys
import unicodedata
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent


def slugify(nombre: str, max_palabras: int = 5) -> str:
    """Slug corto: sin contenido entre paréntesis, sin tildes, en minúsculas y con guiones."""
    base = re.sub(r"\(.*?\)", "", nombre)            # quita "(overfitting)", "(+)"...
    base = base.split(":")[0] if ":" in base and len(base.split(":")[0]) > 3 else base
    base = unicodedata.normalize("NFKD", base).encode("ascii", "ignore").decode()
    palabras = re.findall(r"[a-z0-9]+", base.lower())
    return "-".join(palabras[:max_palabras]) or "nodo"


def parsear(ruta=RAIZ / "esqueleto.md"):
    texto = ruta.read_text(encoding="utf-8").split("---", 1)[1]
    raiz, pila, usados = None, [], set()
    for linea in texto.splitlines():
        m = re.match(r"^( *)- (.+)$", linea)
        if not m:
            continue
        nivel = len(m.group(1)) // 2
        nombre = m.group(2).strip()
        anadido = nombre.endswith("(+)")
        nombre = nombre.replace("(+)", "").strip()
        nombre = re.sub(r"^\d+\.\s+", "", nombre)      # numeración de áreas
        forzado = re.search(r"\{#([a-z0-9-]+)\}$", nombre)
        if forzado:
            nombre = nombre[: forzado.start()].strip()
        nid = forzado.group(1) if forzado else slugify(nombre)
        if nid in usados:                                 # desambiguar duplicados
            k = 2
            while f"{nid}-{k}" in usados:
                k += 1
            nid = f"{nid}-{k}"
        usados.add(nid)
        nodo = {"id": nid, "name": nombre, "added": anadido, "children": []}
        pila = pila[:nivel]
        if nivel == 0:
            raiz = nodo
        else:
            pila[-1]["children"].append(nodo)
        pila.append(nodo)
    return raiz


def recorrer(nodo, ruta=()):
    """Genera (nodo, ruta_de_nombres, profundidad) en preorden."""
    yield nodo, ruta + (nodo["name"],), len(ruta)
    for h in nodo["children"]:
        yield from recorrer(h, ruta + (nodo["name"],))


if __name__ == "__main__":
    arbol = parsear()
    (RAIZ / "datos").mkdir(exist_ok=True)
    (RAIZ / "datos" / "arbol.json").write_text(json.dumps(arbol, ensure_ascii=False, indent=1), encoding="utf-8")
    filas = [f"{n['id']}\t{d}\t{' › '.join(r[1:])}" for n, r, d in recorrer(arbol)]
    (RAIZ / "datos" / "indice.tsv").write_text("\n".join(filas), encoding="utf-8")
    print(f"{len(filas)} nodos → datos/arbol.json, datos/indice.tsv", file=sys.stderr)
