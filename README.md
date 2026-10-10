# Mapa interactivo de IA

Conceptos, herramientas y guía de uso de la inteligencia artificial en un mapa interactivo.
Cada concepto tiene su idea en una frase (visible al pasar el ratón) y una ficha con explicación,
cuándo usarlo, ejemplo, hiperparámetros clave, referencia en Python con enlace a la documentación
oficial y conceptos relacionados.

## Estructura

| Ruta | Qué contiene |
|---|---|
| `esqueleto.md` | El árbol de conceptos (una línea por concepto; la indentación marca el nivel). `{#id}` fija el identificador. |
| `fichas/*.md` | El contenido de cada ficha, por áreas. Campos: `frase`, `desc`, `cuando`, `ej`, `hp`, `py`, `rel`, `en`. |
| `plantilla.html` | Diseño e interacción de la página (HTML, CSS y JavaScript, sin dependencias). |
| `tools/` | Scripts: `construir.py` (une y valida), `html.py` (genera la web), `docs.py` (enlaces a documentación), `verificar_py.py` (comprueba referencias e hiperparámetros contra las librerías). |
| `datos/docs.json` | Caché de enlaces a la documentación oficial de cada referencia de Python. |
| `public/` | La web generada que publica Cloudflare Pages (`index.html` y `_headers`). |

## Actualizar el contenido

```bash
python3 tools/html.py --estricto   # valida árbol y fichas y regenera public/index.html
git add -A && git commit -m "…" && git push   # Cloudflare Pages publica el cambio automáticamente
```

## Publicación

Cloudflare Pages conectado a este repositorio: rama `main`, sin comando de build, directorio de salida `public`.
La web no se indexa en buscadores (`noindex`): solo llega quien tiene el enlace.
