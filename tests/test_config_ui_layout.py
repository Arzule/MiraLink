import os
import tempfile
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QRect, qInstallMessageHandler
from PySide6.QtWidgets import (
    QApplication,
    QAbstractSpinBox,
    QComboBox,
    QPushButton,
    QSlider,
    QSpinBox,
)

from config_ui import ConfigWindow


class TestConfigUILayout(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def _collect_widgets(self, win):
        widgets = []
        for cls in (QSlider, QSpinBox, QComboBox, QPushButton):
            widgets.extend(win.findChildren(cls))
        return widgets

    def _rect_in_window(self, win, widget):
        top_left = widget.mapTo(win, widget.rect().topLeft())
        return QRect(top_left, widget.rect().size())

    def test_layout_messages_and_geometry(self):
        messages = []

        def handler(msg_type, context, message):
            messages.append(str(message))

        qInstallMessageHandler(handler)

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "config.json")
            win = ConfigWindow(config_path=path)
            win.show()
            self.app.processEvents()

            theme_count = win.theme_combo.count()
            self.assertEqual(theme_count, 3)

            # Iterate each theme x font scale
            for theme_idx in range(theme_count):
                win.theme_combo.setCurrentIndex(theme_idx)
                self.app.processEvents()
                for scale_pct in (80, 100, 200):
                    win.font_slider.setValue(scale_pct)
                    self.app.processEvents()

                    widgets = self._collect_widgets(win)
                    for w in widgets:
                        self.assertTrue(
                            w.isVisible(),
                            f"{w.__class__.__name__} {w.objectName()} no visible",
                        )
                        self.assertGreater(w.width(), 0)
                        self.assertGreater(w.height(), 0)
                        if isinstance(w, (QPushButton, QComboBox, QSpinBox)):
                            self.assertGreaterEqual(
                                w.height(),
                                48,
                                f"{w.__class__.__name__} {w.objectName()} alto {w.height()} < 48",
                            )

                    # No intersections between different controls (ignore parent-child)
                    rects = []
                    for w in widgets:
                        if not w.isVisible():
                            continue
                        rects.append((w, self._rect_in_window(win, w)))

                    for i in range(len(rects)):
                        for j in range(i + 1, len(rects)):
                            w1, r1 = rects[i]
                            w2, r2 = rects[j]
                            if w1 is w2:
                                continue
                            if w1.isAncestorOf(w2) or w2.isAncestorOf(w1):
                                continue
                            self.assertFalse(
                                r1.intersects(r2),
                                f"Intersección entre {w1.__class__.__name__} "
                                f"{w1.objectName()} y {w2.__class__.__name__} {w2.objectName()}",
                            )

            # Spin boxes without buttons
            for cls in (QSpinBox,):
                for spin in win.findChildren(cls):
                    self.assertEqual(
                        spin.buttonSymbols(),
                        QAbstractSpinBox.NoButtons,
                        f"{spin.objectName()} tiene botones",
                    )

            # +/- buttons change font size by 10 respecting bounds
            win.font_slider.setValue(100)
            self.app.processEvents()
            win.btn_font_minus.click()
            self.app.processEvents()
            self.assertEqual(win.font_slider.value(), 90)
            self.assertEqual(win.font_spin.value(), 90)

            win.btn_font_plus.click()
            win.btn_font_plus.click()
            self.app.processEvents()
            self.assertEqual(win.font_slider.value(), 110)
            self.assertEqual(win.font_spin.value(), 110)

            win.font_slider.setValue(80)
            self.app.processEvents()
            win.btn_font_minus.click()
            self.app.processEvents()
            self.assertEqual(win.font_slider.value(), 80)

            win.font_slider.setValue(200)
            self.app.processEvents()
            win.btn_font_plus.click()
            self.app.processEvents()
            self.assertEqual(win.font_slider.value(), 200)

            win.close()

        print("\n--- MENSAJES DE Qt CAPTURADOS ---")
        for m in messages:
            print(m)
        print("--- FIN MENSAJES DE Qt ---")

        forbidden = ("QLayout", "Could not parse", "QWidget::", "QStyleSheetStyle")
        for m in messages:
            for token in forbidden:
                self.assertNotIn(token, m, f"Mensaje Qt prohibido: {m}")
