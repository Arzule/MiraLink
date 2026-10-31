r"""Descarga las redes de mirada MobileGaze (ONNX) a models/ (evaluación de la red neuronal, 30/10).

Uso: .\venv\Scripts\python.exe scripts\download_gaze_models.py [mobileone_s0 resnet18 resnet34]
Pesos de github.com/yakhyo/gaze-estimation (licencia MIT; entrenados con Gaze360, datos solo para
investigación). No se sobrescriben los que ya existen.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from neural_gaze import GAZE_MODELS  # noqa: E402
from scripts.download_model import download_model  # noqa: E402


def main(argv=None) -> int:
    names = list(argv if argv is not None else sys.argv[1:]) or list(GAZE_MODELS)
    unknown = [n for n in names if n not in GAZE_MODELS]
    if unknown:
        print(f"Modelos desconocidos: {', '.join(unknown)} (disponibles: {', '.join(GAZE_MODELS)})")
        return 2
    for name in names:
        info = GAZE_MODELS[name]
        if os.path.exists(info.path):
            print(f"{name}: ya existe ({info.path})")
            continue
        try:
            path = download_model(info.path, info.url)
        except RuntimeError as err:
            print(f"{name}: {err}")
            return 1
        print(f"{name}: guardado en {path} ({os.path.getsize(path) / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
