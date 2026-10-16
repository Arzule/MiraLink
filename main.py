"""Punto de entrada de MiraLink; orquesta los demás módulos."""

import sys


def check_environment():
    """Verifica que todas las dependencias requeridas estén disponibles.

    Imprime por cada librería: "OK nombre versión" o "FALLO nombre: error".
    Imprime la versión de Python.
    Devuelve 0 si todo es correcto, 1 si hay algún fallo.
    """
    all_ok = True

    # cv2 (opencv-python)
    try:
        import cv2

        version = getattr(cv2, "__version__", "desconocida")
        print(f"OK cv2 {version}")
    except Exception as e:
        all_ok = False
        print(f"FALLO cv2: {e}")

    # mediapipe
    try:
        import mediapipe as mp

        version = getattr(mp, "__version__", "desconocida")
        print(f"OK mediapipe {version}")
    except Exception as e:
        all_ok = False
        print(f"FALLO mediapipe: {e}")

    # pyautogui
    try:
        import pyautogui

        version = getattr(pyautogui, "__version__", "desconocida")
        print(f"OK pyautogui {version}")
    except Exception as e:
        all_ok = False
        print(f"FALLO pyautogui: {e}")

    # PySide6
    try:
        import PySide6

        version = getattr(PySide6, "__version__", "desconocida")
        print(f"OK PySide6 {version}")
    except Exception as e:
        all_ok = False
        print(f"FALLO PySide6: {e}")

    # numpy
    try:
        import numpy as np

        version = getattr(np, "__version__", "desconocida")
        print(f"OK numpy {version}")
    except Exception as e:
        all_ok = False
        print(f"FALLO numpy: {e}")

    # pyttsx3
    try:
        import pyttsx3

        version = getattr(pyttsx3, "__version__", "desconocida")
        print(f"OK pyttsx3 {version}")
    except Exception as e:
        all_ok = False
        print(f"FALLO pyttsx3: {e}")

    # win32gui
    try:
        import win32gui

        version = getattr(win32gui, "__version__", "disponible")
        print(f"OK win32gui {version}")
    except Exception as e:
        all_ok = False
        print(f"FALLO win32gui: {e}")

    # Versión de Python
    print(f"Python {sys.version.split()[0]}")

    if all_ok:
        return 0
    return 1


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if "--tracker-debug" in args:
        from config_manager import load_config
        from eye_tracker import run_tracker_debug

        return run_tracker_debug(load_config()["camera"]["device_index"])
    dry_run = "--dry-run" in args
    if "--config" in args:
        from PySide6.QtWidgets import QApplication

        from config_ui import ConfigWindow

        from app_logging import set_error_reporter, setup_logging, show_error_dialog

        app = QApplication([sys.argv[0]])
        log_path = setup_logging()
        set_error_reporter(show_error_dialog)  # excepciones en slots -> diálogo visible + log
        print(f"Registro de depuración: {log_path}")
        win = ConfigWindow(dry_run=dry_run)
        win.show()
        return app.exec()
    if dry_run:
        print("--dry-run solo se usa junto con --config: python main.py --config --dry-run")
        return 2
    return check_environment()


if __name__ == "__main__":
    sys.exit(main())
