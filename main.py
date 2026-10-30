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


def parse_resolution(args):
    """`--resolution 1280x720` -> (1280, 720); `--resolution native` -> "native"; None si no se pidió.
    Lanza ValueError si es inválida."""
    if "--resolution" not in args:
        return None
    i = args.index("--resolution")
    if i + 1 < len(args) and args[i + 1].lower() in ("native", "nativa"):
        return "native"
    try:
        w, h = args[i + 1].lower().split("x")
        size = (int(w), int(h))
    except (IndexError, ValueError):
        raise ValueError("Uso: --resolution ANCHOxALTO (por ejemplo 1280x720) o --resolution native") from None
    if size[0] <= 0 or size[1] <= 0:
        raise ValueError("La resolución debe ser positiva")
    return size


def run_record_session(dialog_factory=None, record_dir=None, resolution=None):
    """`--record-session`: calibración de 9 puntos + validación con 5 puntos nuevos, grabando cada cuadro.

    Cada intento (completado o cancelado) se guarda en sessions/session_<fecha>.npz. "Aceptar" también
    guarda calibration.json, como en la calibración normal. `resolution` ((ancho, alto) o "native")
    sustituye la resolución de config.json solo para esta grabación (si la cámara no sostiene ~24 fps a
    ese tamaño se usa 640x480 y queda anotado).
    """
    from PySide6.QtWidgets import QApplication

    from app_logging import set_error_reporter, setup_logging, show_error_dialog
    from config_manager import load_config
    from gaze_session import default_sessions_dir

    app = QApplication.instance() or QApplication([sys.argv[0]])
    app.setQuitOnLastWindowClosed(False)  # el modal se oculta mientras se muestra la pantalla completa
    log_path = setup_logging()
    set_error_reporter(show_error_dialog)
    print(f"Registro de depuración: {log_path}")
    record_dir = record_dir or default_sessions_dir()
    if dialog_factory is None:
        from calibration_ui import CalibrationDialog

        dialog_factory = CalibrationDialog
    config = load_config()
    if resolution == "native":
        config.setdefault("camera", {})["resolution"] = "native"
    elif resolution:
        config.setdefault("camera", {}).update(resolution="fixed", width=int(resolution[0]), height=int(resolution[1]))
    dialog = dialog_factory(config, record_dir=record_dir)
    dialog.finished.connect(app.quit)
    dialog.show()
    app.exec()
    mapper = getattr(dialog, "accepted_mapper", None)
    eye = getattr(mapper, "eye", None)
    if eye is not None:  # "Aceptar": el ojo elegido queda en config.json (tracking.selected_eye)
        from config_manager import save_selected_eye

        try:
            save_selected_eye(eye)
            print(f"Ojo usado guardado en config.json: {eye}")
        except (OSError, ValueError) as e:
            print(f"No se pudo guardar el ojo en config.json: {e}")
    saved = list(getattr(dialog, "saved_sessions", []))
    if saved:
        for path in saved:
            print(f"Sesión guardada: {path}")
        from app_paths import is_frozen

        if not is_frozen():
            print("Evalúala con: .\\venv\\Scripts\\python.exe scripts\\replay_session.py " + saved[-1])
    else:
        print("No se grabó ninguna sesión.")
    return 0


def ensure_streams():
    """En el .exe sin consola sys.stdout/stderr son None y print() fallaría: se envían a os.devnull
    (lo importante queda en logs/miralink_debug.log)."""
    import os

    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w", encoding="utf-8")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w", encoding="utf-8")


def ensure_config_file(path=None):
    """Crea config.json con los valores por defecto si no existe (en el .exe, junto al ejecutable).

    Devuelve (ruta, creado). Si ya existe no se toca: se lee en cada arranque como siempre.
    """
    import os

    from config_manager import _get_project_root, load_config, save_config

    path = path or os.path.join(_get_project_root(), "config.json")
    if os.path.exists(path):
        return path, False
    save_config(load_config(path), path)
    return path, True


def run_self_test(out_path=None, show_window=True):
    """`--self-test`: comprueba el empaquetado SIN cámara ni control del cursor y escribe un informe.

    Verifica librerías, config.json junto al ejecutable, temas, el modelo de MediaPipe empaquetado
    (crea el FaceLandmarker, no abre la cámara) y que la ventana de configuración se construye y se
    muestra (se cierra sola). Devuelve 0 si todo está bien.
    """
    import io
    import os
    import traceback
    from contextlib import redirect_stdout

    from app_paths import data_dir, is_frozen, resource_path

    lines, ok = [], True

    def check(name, fn):
        nonlocal ok
        try:
            detail = fn()
            lines.append(f"OK    {name}" + (f": {detail}" if detail else ""))
        except Exception as e:  # noqa: BLE001 - el informe debe recoger cualquier fallo
            ok = False
            lines.append(f"FALLO {name}: {e}")
            lines.append(traceback.format_exc())

    lines.append(f"MiraLink self-test | ejecutable={is_frozen()} | datos={data_dir()} | recursos={resource_path()}")

    def libraries():
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = check_environment()
        if code != 0:
            raise RuntimeError(buf.getvalue().strip().replace("\n", " | "))
        return buf.getvalue().strip().replace("\n", " | ")

    def config():
        path, created = ensure_config_file()
        from config_manager import load_config

        cfg = load_config(path)
        return f"{path} ({'creado' if created else 'existente'}), suavizado {cfg['interaction']['smoothing_percent']} %"

    def themes():
        from theme_manager import list_themes, load_theme

        ids = [t["id"] for t in list_themes()]
        if not ids:
            raise RuntimeError("no hay temas .qss empaquetados")
        for theme_id in ids:
            if "$FONT" in load_theme(theme_id, 1.0):
                raise RuntimeError(f"tokens sin reemplazar en {theme_id}")
        return ", ".join(ids)

    def model():
        from eye_tracker import FaceTracker, default_model_path

        tracker = FaceTracker(auto_download=False)  # debe venir empaquetado; no se abre la cámara
        tracker.close()
        return default_model_path()

    def window():
        from PySide6.QtCore import QTimer
        from PySide6.QtWidgets import QApplication

        from config_ui import ConfigWindow

        app = QApplication.instance() or QApplication([sys.argv[0]])
        win = ConfigWindow()
        if show_window:
            win.show()
            QTimer.singleShot(1500, app.quit)  # se cierra sola
            app.exec()
        size = f"{win.width()}x{win.height()}"
        win.close()
        return f"ventana {size}, tema {win.theme_combo.currentData()}"

    check("librerías", libraries)
    check("config.json", config)
    check("temas", themes)
    check("modelo de MediaPipe", model)
    check("ventana de configuración", window)
    lines.append("RESULTADO: " + ("OK" if ok else "FALLO"))
    out_path = out_path or os.path.join(data_dir(), "miralink_selftest.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0 if ok else 1


EXE_MODES = ("--config", "--record-session", "--tracker-debug", "--self-test", "--check")


def resolve_args(args, frozen):
    """En el .exe, sin un modo explícito (doble clic) se abre la ventana de configuración.

    `--check` hace en el .exe lo que `python main.py` sin argumentos: verificar las librerías.
    """
    args = list(args)
    if frozen and not any(a in args for a in EXE_MODES):
        args.append("--config")
    return args


def main(argv=None):
    ensure_streams()
    from app_paths import is_frozen

    args = resolve_args(sys.argv[1:] if argv is None else argv, is_frozen())
    if "--self-test" in args:
        return run_self_test()
    if any(a in args for a in ("--config", "--record-session", "--tracker-debug")):
        ensure_config_file()  # config.json junto al .exe (o en el proyecto) se crea la primera vez
    if "--record-session" in args:
        try:
            resolution = parse_resolution(args)
        except ValueError as e:
            print(e)
            return 2
        return run_record_session(resolution=resolution)
    if "--tracker-debug" in args:
        from config_manager import camera_size, load_config
        from eye_tracker import run_tracker_debug

        config = load_config()
        return run_tracker_debug(config["camera"]["device_index"], camera_size(config))
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
