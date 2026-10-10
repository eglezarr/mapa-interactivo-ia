"""Añade o sustituye campos en fichas existentes y reescribe cada ficha en el orden canónico.
Uso (como módulo): actualizar({id: {campo: valor}})"""
import re
from pathlib import Path
RAIZ = Path(__file__).resolve().parent.parent
ORDEN = ["frase", "desc", "cuando", "ej", "hp", "py", "rel", "en"]

def actualizar(cambios):
    pendientes = dict(cambios)
    for f in sorted((RAIZ / "fichas").glob("*.md")):
        texto = f.read_text(encoding="utf-8")
        cabecera, *bloques = re.split(r"^## ", texto, flags=re.M)
        nuevos = []
        for b in bloques:
            lineas = b.rstrip("\n").splitlines()
            fid = lineas[0].strip()
            datos = {}
            for l in lineas[1:]:
                if l.strip() and not l.startswith("#"):
                    k, _, v = l.partition(": ")
                    datos[k] = v.strip()
            if fid in pendientes:
                datos.update({k: v for k, v in pendientes.pop(fid).items() if v})
            cuerpo = "\n".join(f"{k}: {datos[k]}" for k in ORDEN if datos.get(k))
            nuevos.append(f"{fid}\n{cuerpo}\n")
        f.write_text(cabecera + "".join("## " + n + "\n" for n in nuevos).rstrip("\n") + "\n", encoding="utf-8")
    return pendientes  # ids no encontrados
