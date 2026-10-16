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


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--config":
        from PySide6.QtWidgets import QApplication
        from config_ui import ConfigWindow

        app = QApplication(sys.argv)
        win = ConfigWindow()
        win.show()
        return app.exec()
    else:
        return check_environment()


if __name__ == "__main__":
    sys.exit(main())
