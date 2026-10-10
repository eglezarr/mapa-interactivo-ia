"""Une el árbol (esqueleto.md) con las fichas (fichas/*.md), valida y genera datos/mapa.json.

Campos de una ficha (una línea cada uno):
  frase   En una frase (se muestra al pasar el ratón por el nodo)
  desc    Explicación
  cuando  Cuándo usarlo (solo herramientas)
  ej      Ejemplo
  hp      Hiperparámetros clave:  "param: efecto | param: efecto"
  py      Referencias de Python, separadas por " ; " (`entre comillas` = fragmento de código)
  rel     Relacionados: ids separados por comas
  en      Término en inglés
Validaciones: fichas sin nodo, relacionados inexistentes o a sí mismos, campos obligatorios.
Uso:  python3 tools/construir.py  [--estricto]   (estricto = error si falta alguna ficha)
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arbol import RAIZ, parsear, recorrer  # noqa: E402

CAMPOS = {"frase", "desc", "cuando", "ej", "hp", "py", "rel", "en"}
OBLIGATORIOS = ("frase", "desc", "ej")


def leer_fichas():
    """Devuelve {id: {campo: valor}} a partir de todos los ficheros de fichas/."""
    fichas, origen = {}, {}
    for f in sorted((RAIZ / "fichas").glob("*.md")):
        bloques = re.split(r"^## ", f.read_text(encoding="utf-8"), flags=re.M)[1:]
        for b in bloques:
            lineas = b.strip().splitlines()
            fid, datos = lineas[0].strip(), {}
            for l in lineas[1:]:
                if not l.strip() or l.startswith("#"):
                    continue
                clave, _, valor = l.partition(": ")
                if clave not in CAMPOS:
                    raise ValueError(f"{f.name} · {fid}: campo desconocido '{clave}'")
                datos[clave] = valor.strip()
            if fid in fichas:
                raise ValueError(f"Ficha duplicada '{fid}' en {f.name} y {origen[fid]}")
            fichas[fid], origen[fid] = datos, f.name
    return fichas


def construir(estricto=False):
    arbol = parsear()
    fichas = leer_fichas()
    ids = {n["id"] for n, _, _ in recorrer(arbol)}
    docs_f = RAIZ / "datos" / "docs.json"
    docs = json.loads(docs_f.read_text(encoding="utf-8")) if docs_f.exists() else {}
    errores, avisos = [], []

    for fid, d in fichas.items():
        if fid not in ids:
            errores.append(f"Ficha sin nodo en el árbol: {fid}")
        for campo in OBLIGATORIOS:
            if not d.get(campo):
                errores.append(f"{fid}: falta '{campo}'")
        rel = [r.strip() for r in d.get("rel", "").split(",") if r.strip()]
        for r in rel:
            if r not in ids:
                errores.append(f"{fid}: relacionado inexistente '{r}'")
            if r == fid:
                errores.append(f"{fid}: se relaciona consigo mismo")
        d["rel"] = rel
        # Python: cada referencia con su enlace a la documentación (si es una ruta, no un fragmento)
        py = []
        for p in [x.strip() for x in d.get("py", "").split(" ; ") if x.strip()]:
            if p.startswith("`"):
                py.append({"c": p.strip("`"), "s": 1})       # fragmento de código: sin enlace ni verificación
                continue
            ruta = re.match(r"^[A-Za-z_][\w.]*", p).group(0)
            item = {"c": p}
            if ruta in docs:
                item["u"] = docs[ruta]
            else:
                avisos.append(f"{fid}: sin enlace de documentación para {ruta}")
            py.append(item)
        d["py"] = py
        hp = []
        for item in [x.strip() for x in d.get("hp", "").split(" | ") if x.strip()]:
            param, _, efecto = item.partition(": ")
            if not efecto:
                errores.append(f"{fid}: hiperparámetro sin efecto '{item}'")
            hp.append([param.strip(), efecto.strip()])
        d["hp"] = hp

    def volcar(nodo):
        d = fichas.get(nodo["id"], {})
        salida = {"id": nodo["id"], "name": nodo["name"]}
        for campo in ("en", "frase", "desc", "cuando", "ej", "hp", "py", "rel"):
            if d.get(campo):
                salida[campo] = d[campo]
        if nodo["children"]:
            salida["children"] = [volcar(h) for h in nodo["children"]]
        return salida

    sin_ficha = [n["id"] for n, _, _ in recorrer(arbol) if n["id"] not in fichas]
    if sin_ficha:
        (errores if estricto else avisos).append(f"{len(sin_ficha)} nodos sin ficha todavía")

    for a in avisos:
        print("AVISO:", a, file=sys.stderr)
    if errores:
        for e in errores:
            print("ERROR:", e, file=sys.stderr)
        sys.exit(1)

    mapa = volcar(arbol)
    (RAIZ / "datos" / "mapa.json").write_text(json.dumps(mapa, ensure_ascii=False), encoding="utf-8")
    print(f"OK · {len(ids)} nodos · {len(fichas)} fichas", file=sys.stderr)
    return mapa


if __name__ == "__main__":
    construir(estricto="--estricto" in sys.argv)
