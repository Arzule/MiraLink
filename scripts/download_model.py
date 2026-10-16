"""Descarga el modelo FaceLandmarker de MediaPipe a models/face_landmarker.task.

Uso: python scripts/download_model.py (con el python del venv)
"""

import os
import sys
import urllib.error
import urllib.request

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)


def default_model_path():
    """Ruta por defecto del modelo dentro del proyecto."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, "models", "face_landmarker.task")


def download_model(dest=None, url=MODEL_URL):
    """Descarga el modelo de forma atómica. Lanza RuntimeError con mensaje claro si falla."""
    dest = dest or default_model_path()
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    try:
        with urllib.request.urlopen(url, timeout=60) as resp, open(tmp, "wb") as f:
            while True:
                chunk = resp.read(1 << 16)
                if not chunk:
                    break
                f.write(chunk)
    except (urllib.error.URLError, OSError) as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise RuntimeError(
            f"No se pudo descargar el modelo de MediaPipe ({e}). Revisa tu conexión a internet "
            f"o descarga el archivo manualmente desde {url} y guárdalo en {dest}"
        ) from e
    os.replace(tmp, dest)
    return dest


if __name__ == "__main__":
    try:
        print("Modelo guardado en", download_model())
    except RuntimeError as err:
        print(err)
        sys.exit(1)
