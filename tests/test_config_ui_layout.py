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


class TestConfigUIPartZero(unittest.TestCase):
    """Correcciones U1-U3 de la Parte 0."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def _window(self, d):
        win = ConfigWindow(config_path=os.path.join(d, "config.json"))
        win.resize(1100, 1100)
        win.show()
        self.app.processEvents()
        return win

    def test_appearance_card_single_cell_content_height(self):
        with tempfile.TemporaryDirectory() as d:
            win = self._window(d)
            card = win.card_appearance
            grid = card.parentWidget().layout()
            # La tarjeta está en la última fila, columna izquierda, una sola celda
            layout = win.scroll_layout.itemAt(1).layout()
            idx = layout.indexOf(card)
            row, col, rowspan, colspan = layout.getItemPosition(idx)
            self.assertEqual((row, col, rowspan, colspan), (2, 0, 1, 1))
            self.assertEqual(card.height(), card.sizeHint().height())
            # No se estira como las tarjetas de sliders de otra fila
            self.assertLessEqual(card.height(), win.card_smooth.height())
            # El sobrante queda debajo del pie, no entre tarjetas ni arriba
            footer_bottom = win.btn_save.mapTo(win.content, win.btn_save.rect().bottomLeft()).y()
            self.assertLessEqual(footer_bottom, win.content.height())
            win.close()

    def test_spinboxes_synced_with_sliders_on_start(self):
        with tempfile.TemporaryDirectory() as d:
            win = self._window(d)
            self.assertEqual(win.smooth_spin.value(), win.smooth_slider.value())
            self.assertEqual(win.dwell_spin.value(), win.dwell_slider.value())
            self.assertEqual(win.blink_click_spin.value(), win.blink_click_slider.value())
            self.assertEqual(win.blink_release_spin.value(), win.blink_release_slider.value())
            win.close()

    def test_combo_popup_all_themes_real_platform(self):
        """Abre el popup por tema y escala en un subproceso con la plataforma Qt real.

        En la plataforma 'offscreen' Qt emite siempre 'QFont::setPointSizeF: Point size <= 0
        (-0.75)' al abrir cualquier popup (incluso sin hoja de estilo), por eso este chequeo
        corre con la plataforma real de Windows.
        """
        import json
        import subprocess
        import sys

        probe = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_popup_probe.py")
        env = dict(os.environ)
        env.pop("QT_QPA_PLATFORM", None)
        with tempfile.TemporaryDirectory() as d:
            proc = subprocess.run([sys.executable, probe, d], capture_output=True, text=True, env=env, timeout=120)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            data = json.loads(proc.stdout.strip().splitlines()[-1])
            self.assertEqual(data["qfont"], [], f"Mensajes QFont:: {data['qfont']}")
            self.assertEqual(len(data["results"]), 18)
            for r in data["results"]:
                self.assertLessEqual(
                    abs(r["lum"] - r["declared"]),
                    0.1,
                    f"Fondo del popup distinto del declarado: {r}",
                )
                self.assertGreaterEqual(r["view_h"], 3 * 48 - 10)  # ítems de al menos 48 px
