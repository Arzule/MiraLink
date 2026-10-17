"""Registro de depuración a archivo, manejador de excepciones y de mensajes de Qt.

Archivo: logs/miralink_debug.log (en la raíz del proyecto, ignorado por git).
`setup_logging()` solo se llama desde main.py (no en los tests, que instalan su propio manejador).
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
import threading
import traceback

from typing import Callable, Optional

from app_paths import data_dir

LOG_DIR = os.path.join(data_dir(), "logs")  # junto al .exe en el ejecutable
LOG_PATH = os.path.join(LOG_DIR, "miralink_debug.log")

_configured = False
_error_reporter: Optional[Callable[[str, str], None]] = None


def set_error_reporter(reporter: Optional[Callable[[str, str], None]]):
    """Registra la función (título, detalle) que muestra un diálogo de error visible en la UI."""
    global _error_reporter
    _error_reporter = reporter


def report_error(title: str, detail: str):
    """Registra el error y, si hay UI, lo muestra en un diálogo (nunca falla en silencio)."""
    logging.getLogger("miralink.error").error("%s: %s", title, detail)
    if _error_reporter is not None:
        try:
            _error_reporter(title, detail)
        except Exception:  # noqa: BLE001
            logging.getLogger("miralink.error").exception("Falló el diálogo de error")


def _excepthook(exc_type, exc, tb):
    text = "".join(traceback.format_exception(exc_type, exc, tb))
    logging.getLogger("miralink.excepthook").critical("Excepción no controlada:\n%s", text)
    if not issubclass(exc_type, KeyboardInterrupt):
        report_error("Error inesperado", f"{exc_type.__name__}: {exc}\n\nDetalle en {LOG_PATH}")
    sys.__excepthook__(exc_type, exc, tb)


def _thread_excepthook(args):
    _excepthook(args.exc_type, args.exc_value, args.exc_traceback)


def setup_logging(path: str = LOG_PATH, level: int = logging.DEBUG, install_hooks: bool = True) -> str:
    """Configura el log a archivo (rotativo) y los manejadores de errores. Devuelve la ruta."""
    global _configured
    os.makedirs(os.path.dirname(path), exist_ok=True)
    root = logging.getLogger()
    root.setLevel(level)
    if not _configured:
        handler = logging.handlers.RotatingFileHandler(path, maxBytes=1_000_000, backupCount=2, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s.%(msecs)03d %(levelname)-8s [%(threadName)s] %(name)s: %(message)s", "%H:%M:%S"))
        root.addHandler(handler)
        _configured = True
    if install_hooks:
        sys.excepthook = _excepthook
        threading.excepthook = _thread_excepthook
        _install_qt_handler()
    logging.getLogger("miralink").info("=== MiraLink iniciado (log en %s) ===", path)
    return path


def _install_qt_handler():
    from PySide6.QtCore import QtMsgType, qInstallMessageHandler

    levels = {
        QtMsgType.QtDebugMsg: logging.DEBUG,
        QtMsgType.QtInfoMsg: logging.INFO,
        QtMsgType.QtWarningMsg: logging.WARNING,
        QtMsgType.QtCriticalMsg: logging.ERROR,
        QtMsgType.QtFatalMsg: logging.CRITICAL,
    }

    def handler(msg_type, context, message):
        logging.getLogger("miralink.qt").log(levels.get(msg_type, logging.INFO), "%s", message)

    qInstallMessageHandler(handler)


def show_error_dialog(title: str, detail: str):
    """Diálogo de error de Qt (para set_error_reporter)."""
    from PySide6.QtWidgets import QApplication, QMessageBox

    if QApplication.instance() is None:
        return
    box = QMessageBox(QMessageBox.Critical, title, detail)
    box.setStandardButtons(QMessageBox.Ok)
    box.button(QMessageBox.Ok).setText("Aceptar")
    box.button(QMessageBox.Ok).setMinimumHeight(48)
    box.setAccessibleName("Diálogo de error")
    box.exec()
