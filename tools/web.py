"""Genera las páginas publicables a partir de plantilla.html y los datos.

  salida/mapa-ia.html   versión para la vista previa en Claude (sin <head>: el entorno añade el esqueleto)
  public/index.html     versión web autónoma (documento completo) que sirve Cloudflare Pages
  public/_headers       cabeceras HTTP de Cloudflare Pages (no indexar en buscadores, seguridad básica)
Uso: python3 tools/web.py [--estricto]
"""
import json
import sys
from pathlib import Path
from urllib.parse import quote

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arbol import RAIZ          # noqa: E402
from construir import construir  # noqa: E402

TITULO = "Mapa interactivo de IA"
DESCRIPCION = "Conceptos, herramientas y guía de uso de la inteligencia artificial en un mapa interactivo."

# Icono de pestaña: tres nodos conectados con colores de la escala viridis
FAVICON = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">'
           '<path d="M9 16 C15 16 15 8 22 8 M9 16 C15 16 15 24 22 24" fill="none" stroke="#2a778e" stroke-width="2.4"/>'
           '<circle cx="8" cy="16" r="5" fill="#440154"/><circle cx="23" cy="8" r="4.5" fill="#21a685"/>'
           '<circle cx="23" cy="24" r="4.5" fill="#95d840"/></svg>')

# Lo que el entorno de Claude añade por su cuenta y la web autónoma necesita explícitamente
RESET = (":root{padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}"
         "body{margin:0}img{max-width:100%}[hidden]{display:none!important}")

HEADERS = """/*
  X-Robots-Tag: noindex, nofollow
  X-Content-Type-Options: nosniff
  Referrer-Policy: strict-origin-when-cross-origin
"""


def documento_web(pagina: str) -> str:
    """Envuelve la página en un documento HTML completo con <head> propio."""
    corte = pagina.index('<div class="app">')
    cabeza, cuerpo = pagina[:corte], pagina[corte:]
    meta = "\n".join([
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">',
        '<meta name="robots" content="noindex, nofollow">',
        f'<meta name="description" content="{DESCRIPCION}">',
        f'<meta property="og:title" content="{TITULO}">',
        f'<meta property="og:description" content="{DESCRIPCION}">',
        '<meta property="og:type" content="website">',
        '<meta name="theme-color" content="#eef1f0" media="(prefers-color-scheme: light)">',
        '<meta name="theme-color" content="#0d1312" media="(prefers-color-scheme: dark)">',
        f'<link rel="icon" href="data:image/svg+xml,{quote(FAVICON)}">',
        f"<style>{RESET}</style>",
    ])
    return f'<!doctype html>\n<html lang="es">\n<head>\n{meta}\n{cabeza.strip()}\n</head>\n<body>\n{cuerpo.strip()}\n</body>\n</html>\n'


if __name__ == "__main__":
    mapa = construir(estricto="--estricto" in sys.argv)
    datos = json.dumps(mapa, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    plantilla = (RAIZ / "plantilla.html").read_text(encoding="utf-8")
    assert plantilla.count("/*__DATOS__*/") == 1
    pagina = plantilla.replace("/*__DATOS__*/", datos)

    previa = RAIZ / "salida" / "mapa-ia.html"
    previa.parent.mkdir(exist_ok=True)
    previa.write_text(pagina, encoding="utf-8")

    web = RAIZ / "public" / "index.html"
    web.parent.mkdir(exist_ok=True)
    web.write_text(documento_web(pagina), encoding="utf-8")
    (RAIZ / "public" / "_headers").write_text(HEADERS, encoding="utf-8")

    for f in (previa, web):
        print(f"→ {f.relative_to(RAIZ)} ({f.stat().st_size / 1024:.0f} KB)", file=sys.stderr)
