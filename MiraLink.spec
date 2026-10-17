# -*- mode: python ; coding: utf-8 -*-
"""Receta de PyInstaller para MiraLink.exe (un solo archivo, sin consola).

Construir con build_exe.ps1 (o: .\\venv\\Scripts\\python.exe -m PyInstaller MiraLink.spec --noconfirm).

Dentro del .exe van el código, las librerías, los temas (styles/*.qss) y el modelo de MediaPipe
(models/face_landmarker.task). Fuera, junto al .exe, la aplicación crea en tiempo de ejecución
config.json (la primera vez, con los valores por defecto), calibration.json, logs/ y sessions/
(ver app_paths.py). config.json NO se empaqueta: se lee de la carpeta del .exe en cada arranque.
"""

import glob

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

datas = [(path, "styles") for path in glob.glob("styles/*.qss")]
datas += [("models/face_landmarker.task", "models")]
datas += collect_data_files("mediapipe")  # recursos internos de MediaPipe (grafos, .binarypb)
binaries = collect_dynamic_libs("mediapipe")  # DLL nativas de MediaPipe Tasks

a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=[
        # módulos que main.py importa dentro de funciones (por si el análisis estático no los ve)
        "config_ui",
        "calibration_ui",
        "control_session",
        "cursor_feedback",
        "gaze_session",
        "app_logging",
        "scripts.download_model",
        "win32gui",
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=["tkinter", "tests"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="MiraLink",
    debug=False,
    strip=False,
    upx=False,  # UPX suele disparar falsos positivos de antivirus y rompe algunas DLL de Qt
    runtime_tmpdir=None,
    console=False,  # aplicación de ventana; los mensajes van a logs\miralink_debug.log
)
