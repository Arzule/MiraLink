"""Gestor de temas: carga de temas .qss y reemplazo de tokens de fuente, sin importar Qt."""

import os

from app_paths import resource_path

# TODO(etapa 1)


def _get_styles_dir():
    """Devuelve la ruta de la carpeta styles (en el .exe, la empaquetada dentro del ejecutable)."""
    return resource_path("styles")


def list_themes(styles_dir=None):
    """Lista los archivos .qss de la carpeta styles.

    Devuelve una lista de dicts con id (nombre sin extensión) y nombre visible.
    El nombre visible se toma de la primera línea del archivo si tiene el formato
    /* name: Nombre visible */; si no, se deriva del nombre de archivo.
    "default" va primero en el orden devuelto.
    """
    if styles_dir is None:
        styles_dir = _get_styles_dir()

    if not os.path.isdir(styles_dir):
        return []

    themes = []
    for fname in os.listdir(styles_dir):
        if fname.endswith(".qss"):
            theme_id = os.path.splitext(fname)[0]
            path = os.path.join(styles_dir, fname)
            visible_name = None
            try:
                with open(path, "r", encoding="utf-8") as f:
                    first_line = f.readline()
                    if first_line and "/* name:" in first_line:
                        # extraer contenido después de name:
                        start = first_line.find("name:")
                        if start != -1:
                            content = first_line[start + len("name:"):]
                            # quitar */ si está
                            content = content.replace("*/", "")
                            visible_name = content.strip()
            except Exception:
                visible_name = None

            if not visible_name:
                # derivar del nombre
                visible_name = theme_id.replace("_", " ").title()

            themes.append({"id": theme_id, "name": visible_name})

    # ordenar: default primero
    themes_sorted = sorted(themes, key=lambda t: (0 if t["id"] == "default" else 1, t["id"]))
    return themes_sorted


def load_theme(theme_id, font_scale, styles_dir=None):
    """Carga el tema y reemplaza los tokens de fuente.

    Tokens: $FONT_SMALL = round(9*scale)pt, $FONT_BASE = round(11*scale)pt, $FONT_TITLE = round(18*scale)pt
    (en pt, con mínimo 1, para que Qt tenga siempre pointSize válido)
    Si theme_id no existe, usa "default".
    """
    if styles_dir is None:
        styles_dir = _get_styles_dir()

    scale = font_scale
    try:
        scale = float(scale)
    except Exception:
        scale = 1.0

    # determinar archivo a cargar
    themes = list_themes(styles_dir)
    theme_ids = [t["id"] for t in themes]
    if theme_id not in theme_ids:
        theme_id = "default"

    path = os.path.join(styles_dir, f"{theme_id}.qss")
    try:
        with open(path, "r", encoding="utf-8") as f:
            qss = f.read()
    except Exception:
        # intentar cargar default
        path_default = os.path.join(styles_dir, "default.qss")
        with open(path_default, "r", encoding="utf-8") as f:
            qss = f.read()

    font_small = f"{max(1, round(9 * scale))}pt"
    font_base = f"{max(1, round(11 * scale))}pt"
    font_title = f"{max(1, round(18 * scale))}pt"

    qss = qss.replace("$FONT_SMALL", font_small)
    qss = qss.replace("$FONT_BASE", font_base)
    qss = qss.replace("$FONT_TITLE", font_title)

    return qss
