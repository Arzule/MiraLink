import os
import tempfile
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication
from config_ui import ConfigWindow
from config_manager import load_config


class TestConfigUISmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_ui_smoke_roundtrip_and_signals(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "config.json")
            win = ConfigWindow(config_path=path)

            # change values
            win.smooth_slider.setValue(80)
            win.dwell_slider.setValue(1500)
            win.blink_click_slider.setValue(200)
            win.blink_release_slider.setValue(250)

            # change theme and font
            # find index of alto_contraste_1
            for i in range(win.theme_combo.count()):
                if win.theme_combo.itemData(i) == "alto_contraste_1":
                    win.theme_combo.setCurrentIndex(i)
                    break

            win.font_slider.setValue(120)

            # calibrate signal
            emitted = []

            def on_cal():
                emitted.append(True)

            class StubDialog:
                # Evita abrir el modal real (sondearía la cámara) en las pruebas
                class _Sig:
                    def connect(self, *_a):
                        pass

                camera_selected = _Sig()
                accepted_mapper = None

                def exec(self):
                    return 0

            win.calibration_dialog_factory = StubDialog
            win.calibrate_requested.connect(on_cal)
            win.btn_calibrate.click()
            self.assertTrue(len(emitted) > 0)

            # save
            win.btn_save.click()
            win.close()

            # verify file
            cfg = load_config(path)
            self.assertEqual(cfg["interaction"]["smoothing_percent"], 80)
            self.assertEqual(cfg["interaction"]["dwell_hold_ms"], 1500)
            self.assertEqual(cfg["interaction"]["blink_click_ms"], 200)
            self.assertEqual(cfg["interaction"]["blink_release_ms"], 250)
            self.assertEqual(cfg["ui"]["theme"], "alto_contraste_1")
            self.assertAlmostEqual(cfg["ui"]["font_scale"], 1.2, places=2)
