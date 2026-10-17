"""Rutas de la aplicación, iguales en desarrollo y en el ejecutable (PyInstaller, un solo archivo).

- data_dir(): datos del usuario que deben sobrevivir entre ejecuciones (config.json, calibration.json,
  logs/, sessions/). En desarrollo es la raíz del proyecto; en el .exe es la carpeta donde está el .exe.
- resource_path(...): recursos de solo lectura empaquetados (styles/*.qss, models/face_landmarker.task).
  En desarrollo es la raíz del proyecto; en el .exe es la carpeta temporal donde PyInstaller los extrae
  (sys._MEIPASS), que se borra al cerrar: nunca escribir ahí.
"""

from __future__ import annotations

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def is_frozen() -> bool:
    """True si se ejecuta desde el .exe generado con PyInstaller."""
    return bool(getattr(sys, "frozen", False))


def data_dir() -> str:
    if is_frozen():
        return os.path.dirname(os.path.abspath(sys.executable))
    return PROJECT_ROOT


def resource_path(*parts: str) -> str:
    base = getattr(sys, "_MEIPASS", PROJECT_ROOT) if is_frozen() else PROJECT_ROOT
    return os.path.join(base, *parts)
