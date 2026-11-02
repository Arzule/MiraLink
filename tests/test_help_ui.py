"""Ventana de ayuda (help_ui.py) y su botón en la ventana de configuración."""

import os
import tempfile
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QtMsgType, qInstallMessageHandler
from PySide6.QtWidgets import QApplication

import help_ui
from config_manager import DEFAULT_CONFIG


class TestHelpContent(unittest.TestCase):
    def test_sections_cover_the_whole_program(self):
        ids = [sid for sid, _t, _h in help_ui.help_sections()]
        for needed in ("calibrar", "control", "modos", "clic", "reposo", "botones", "ajustes", "atajos", "problemas"):
            self.assertIn(needed, ids)
        self.assertEqual(len(ids), len(set(ids)))
        html = help_ui.help_html()
        for text in ("Ctrl+Alt+P", "Ctrl+Alt+R", "Lupa 200%", "Ir al escritorio", "Recentrar", "Enter", "ESC"):
            self.assertIn(text, html)

    def test_numbers_come_from_the_current_config(self):
        cfg = {"interaction": dict(DEFAULT_CONFIG["interaction"], blink_click_ms=320, sleep_close_ms=3000,
                                   wake_close_ms=1500, mode_blinks=4, head_deadzone_deg=6.5, pointer_mode="hybrid")}
        html = help_ui.help_html(cfg)
        self.assertIn("320 ms", html)
        self.assertIn("3 s", html)
        self.assertIn("1,5 s", html)
        self.assertIn("4 parpadeos normales", html)
        self.assertIn("6.5°", html)
        self.assertIn("modo cabeza</b>", html)  # modo con el que arranca
        self.assertIn("modo mirada</b>", help_ui.help_html())


class TestHelpDialog(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.messages = []

        def handler(mode, _ctx, msg):
            if mode in (QtMsgType.QtWarningMsg, QtMsgType.QtCriticalMsg, QtMsgType.QtFatalMsg):
                self.messages.append(msg)

        self._old = qInstallMessageHandler(handler)

    def tearDown(self):
        qInstallMessageHandler(self._old)

    def test_help_button_opens_accessible_dialog(self):
        from config_ui import ConfigWindow

        path = os.path.join(tempfile.mkdtemp(), "config.json")
        win = ConfigWindow(config_path=path)
        win.show()
        self.app.processEvents()
        btn = win.btn_help
        self.assertGreaterEqual(btn.height(), 48)
        self.assertTrue(btn.accessibleName())
        btn.click()
        dlg = win.help_dialog
        self.assertIsNotNone(dlg)
        self.app.processEvents()
        self.assertTrue(dlg.isVisible())
        self.assertFalse(dlg.isModal())  # se puede leer mientras se usa la aplicación
        self.assertEqual(dlg.index.count(), len(dlg.sections))
        for row in range(dlg.index.count()):
            self.assertGreaterEqual(dlg.index.visualItemRect(dlg.index.item(row)).height(), 48)
        self.assertGreaterEqual(dlg.btn_close.height(), 48)
        for w in (dlg, dlg.index, dlg.text, dlg.btn_close):
            self.assertTrue(w.accessibleName())
        self.assertIn("Calibrar", dlg.text.toPlainText())
        # elegir una sección del índice desplaza el texto
        dlg.index.setCurrentRow(dlg.index.count() - 1)
        self.app.processEvents()
        far = dlg.text.verticalScrollBar().value()
        self.assertGreater(far, 200)
        dlg.index.setCurrentRow(0)
        self.app.processEvents()
        self.assertLess(dlg.text.verticalScrollBar().value(), 20)  # vuelve al inicio (margen del documento)
        # abrirla otra vez reemplaza la anterior (una sola ventana de ayuda)
        win.btn_help.click()
        self.assertIsNot(win.help_dialog, dlg)
        win.help_dialog.btn_close.click()
        self.app.processEvents()
        self.assertFalse(win.help_dialog.isVisible())
        win.close()
        # artefactos conocidos de la plataforma offscreen (no ocurren con la real)
        artifacts = ("setPointSize", "does not support raise", "does not support activateWindow",
                     "Cannot find font directory")
        self.assertEqual([m for m in self.messages if not any(a in m for a in artifacts)], [])


if __name__ == "__main__":
    unittest.main()
