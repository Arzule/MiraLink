"""Parte D: resolución configurable, MJPG con vuelta a 640x480 y fps medidos con timestamps.

Usa un módulo cv2 falso: nunca se abre la cámara real.
"""

import json
import os
import tempfile
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

import eye_tracker as et
from config_manager import DEFAULT_CONFIG, load_config
from eye_tracker import CameraSource, EyeTrackingPipeline, FpsMeter, TrackingFrame


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class FakeCv2:
    """Imita lo que usa CameraSource de cv2. `modes`: {(ancho, alto, mjpg): fps}."""

    CAP_DSHOW = 700
    CAP_PROP_FRAME_WIDTH = 3
    CAP_PROP_FRAME_HEIGHT = 4
    CAP_PROP_FPS = 5
    CAP_PROP_FOURCC = 6

    def __init__(self, modes, clock, opens=True, frames=True):
        self.modes = modes
        self.clock = clock
        self.opens = opens
        self.frames = frames
        self.captures = []

    @staticmethod
    def VideoWriter_fourcc(*chars):  # noqa: N802
        return sum(ord(c) << (8 * i) for i, c in enumerate(chars))

    def VideoCapture(self, index, api):  # noqa: N802
        cap = FakeCap(self, index)
        self.captures.append(cap)
        return cap


class FakeCap:
    def __init__(self, cv2, index):
        self.cv2 = cv2
        self.index = index
        self.props = {}
        self.released = False

    def isOpened(self):  # noqa: N802
        return self.cv2.opens

    def set(self, prop, value):
        self.props[prop] = value
        return True

    @property
    def mjpg(self):
        return self.props.get(FakeCv2.CAP_PROP_FOURCC) == FakeCv2.VideoWriter_fourcc(*"MJPG")

    def _mode(self):
        w = int(self.props.get(FakeCv2.CAP_PROP_FRAME_WIDTH, 640))
        h = int(self.props.get(FakeCv2.CAP_PROP_FRAME_HEIGHT, 480))
        key = (w, h, self.mjpg)
        if key in self.cv2.modes:
            return w, h, self.cv2.modes[key]
        return 640, 480, self.cv2.modes.get((640, 480, False), 30.0)  # tamaño no soportado: entrega 640x480

    def get(self, prop):
        if prop == FakeCv2.CAP_PROP_FOURCC:
            return FakeCv2.VideoWriter_fourcc(*("MJPG" if self.mjpg else "YUY2"))
        return 0.0

    def read(self):
        if not self.cv2.frames:
            return False, None
        w, h, fps = self._mode()
        self.cv2.clock.t += 1.0 / fps
        return True, np.zeros((h, w, 3), np.uint8)

    def release(self):
        self.released = True


class TestCameraSource(unittest.TestCase):
    def make(self, modes, size=(640, 480), **kw):
        clock = FakeClock()
        cv2 = FakeCv2(modes, clock, **kw)
        cam = CameraSource(0, size[0], size[1], cv2_module=cv2, clock=clock)
        return cam, cv2

    def test_default_640x480_without_mjpg(self):
        cam, cv2 = self.make({(640, 480, False): 30.0})
        cam.open()
        self.assertEqual(cam.actual_size, (640, 480))
        self.assertNotIn(FakeCv2.CAP_PROP_FOURCC, cv2.captures[0].props)
        self.assertAlmostEqual(cam.measured_fps, 30.0, places=3)
        self.assertEqual(cam.fallback_reason, "")
        self.assertEqual(cam.describe(), "640x480 YUY2")
        ok, frame = cam.read()
        self.assertTrue(ok)
        cam.release()
        self.assertTrue(cv2.captures[0].released)

    def test_720p_with_mjpg_when_supported(self):
        cam, cv2 = self.make({(640, 480, False): 30.0, (1280, 720, True): 30.0, (1280, 720, False): 10.0}, (1280, 720))
        cam.open()
        self.assertEqual(cam.actual_size, (1280, 720))
        self.assertEqual(cam.fourcc, "MJPG")
        self.assertEqual(len(cv2.captures), 1)
        self.assertEqual(cam.describe(), "1280x720 MJPG")

    def test_falls_back_when_720p_is_too_slow(self):
        cam, cv2 = self.make({(640, 480, False): 30.0, (1280, 720, True): 12.0}, (1280, 720))
        cam.open()
        self.assertEqual(cam.actual_size, (640, 480))
        self.assertIn("12.0 fps", cam.fallback_reason)
        self.assertTrue(cv2.captures[0].released)
        self.assertEqual(len(cv2.captures), 2)
        self.assertNotIn(FakeCv2.CAP_PROP_FOURCC, cv2.captures[1].props)  # 640x480 sin MJPG
        self.assertIn("sin 1280x720", cam.describe())

    def test_accepts_delivered_size_when_fps_are_enough(self):
        # La cámara no da 1280x720 y entrega otro tamaño a 30 fps: se acepta (antes se descartaba sin medir)
        cam, cv2 = self.make({(640, 480, False): 30.0}, (1280, 720))
        cam.open()
        self.assertEqual(cam.actual_size, (640, 480))
        self.assertEqual(cam.fallback_reason, "")
        self.assertEqual(len(cv2.captures), 1)
        self.assertIn("pedido 1280x720", cam.describe())

    def test_errors_are_clear(self):
        cam, _ = self.make({(640, 480, False): 30.0}, opens=False)
        with self.assertRaises(RuntimeError) as ctx:
            cam.open()
        self.assertIn("No se pudo abrir la cámara 0", str(ctx.exception))
        cam, cv2 = self.make({(640, 480, False): 30.0}, frames=False)
        with self.assertRaises(RuntimeError) as ctx:
            cam.open()
        self.assertIn("no entrega imagen", str(ctx.exception))
        self.assertTrue(cv2.captures[-1].released)

    def test_fps_measured_with_median_interval(self):
        cam, _ = self.make({(640, 480, False): 15.0})
        cam.open()
        self.assertAlmostEqual(cam.measured_fps, 15.0, places=3)


class TestFpsMeter(unittest.TestCase):
    def test_measures_rate_from_timestamps(self):
        m = FpsMeter()
        self.assertEqual(m.update(0.0), 0.0)
        out = 0.0
        for i in range(1, 90):
            out = m.update(i / 30)
        self.assertAlmostEqual(out, 30.0, delta=0.5)
        t = 89 / 30
        for _ in range(60):  # la cámara baja a 15 fps (poca luz)
            t += 1 / 15
            out = m.update(t)
        self.assertAlmostEqual(out, 15.0, delta=0.5)

    def test_pipeline_fills_fps_and_camera_info(self):
        class Cam:
            def read(self):
                return True, np.zeros((480, 640, 3), np.uint8)

            def describe(self):
                return "640x480 YUY2"

        class Tracker:
            def process(self, image, ts):
                return TrackingFrame(timestamp=ts, face_found=False)

        clock = FakeClock()

        def tick():
            clock.t += 1 / 30
            return clock.t

        p = EyeTrackingPipeline(Cam(), Tracker(), clock=tick)
        frame = None
        for _ in range(40):
            frame = p.step()
        self.assertAlmostEqual(frame.fps, 30.0, delta=0.5)
        self.assertEqual(frame.camera_info, "640x480 YUY2")


class TestCameraConfig(unittest.TestCase):
    def test_defaults_and_clamps(self):
        self.assertEqual(
            DEFAULT_CONFIG["camera"], {"device_index": 0, "resolution": "native", "width": 640, "height": 480}
        )
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "config.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"camera": {"width": 99999, "height": "x"}}, f)
            cam = load_config(path)["camera"]
            self.assertEqual((cam["width"], cam["height"]), (1920, 480))
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"camera": {"width": 1280, "height": 720}}, f)
            cam = load_config(path)["camera"]
            self.assertEqual((cam["width"], cam["height"]), (1280, 720))

    def test_tracker_thread_uses_configured_size_without_opening(self):
        from PySide6.QtWidgets import QApplication

        QApplication.instance() or QApplication([])
        t = et.TrackerThread(0, camera_size=(1280, 720))
        cam = t._camera_factory(0)
        self.assertIsInstance(cam, CameraSource)
        self.assertEqual((cam.width, cam.height), (1280, 720))
        self.assertIsNone(cam._cap)  # solo se construyó: no se abrió

    def test_parse_resolution(self):
        import main

        self.assertIsNone(main.parse_resolution(["--record-session"]))
        self.assertEqual(main.parse_resolution(["--record-session", "--resolution", "1280x720"]), (1280, 720))
        for bad in (["--resolution"], ["--resolution", "grande"], ["--resolution", "0x720"]):
            with self.assertRaises(ValueError):
                main.parse_resolution(bad)


class TestCameraStatusInUi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication

        cls.app = QApplication.instance() or QApplication([])

    def test_status_text_warns_below_25_fps(self):
        import calibration_ui as cui

        img = np.zeros((720, 1280, 3), np.uint8)
        ok = cui.camera_status_text(TrackingFrame(timestamp=0, image=img, fps=29.6, camera_info="1280x720 MJPG"))
        self.assertEqual(ok, "Cámara: 1280x720 MJPG · 30 fps")
        slow = cui.camera_status_text(TrackingFrame(timestamp=0, image=img, fps=14.2))
        self.assertIn("1280x720", slow)
        self.assertIn("Menos de 25 fps", slow)
        self.assertIn("midiendo", cui.camera_status_text(TrackingFrame(timestamp=0, image=img)))

    def test_modal_shows_camera_status_and_uses_configured_size(self):
        import calibration_ui as cui
        from PySide6.QtCore import QObject, Signal

        class FakeTracker(QObject):
            frame_ready = Signal(object)
            failed = Signal(str)

            def start(self):
                pass

            def stop(self):
                pass

        tr = FakeTracker()
        dlg = cui.CalibrationDialog({"camera": {"device_index": 0}}, camera_lister=lambda: [0], tracker_factory=lambda i: tr)
        dlg.show()
        self.app.processEvents()
        tr.frame_ready.emit(TrackingFrame(timestamp=1.0, image=np.zeros((480, 640, 3), np.uint8), fps=18.0,
                                          camera_info="640x480 YUY2 (sin 1280x720: 1280x720 solo a 9.8 fps)"))
        self.assertIn("sin 1280x720", dlg.camera_info_label.text())
        self.assertIn("Menos de 25 fps", dlg.camera_info_label.text())
        self.assertTrue(dlg.camera_info_label.accessibleName())
        dlg.reject()
        fixed = {"camera": {"resolution": "fixed", "width": 1280, "height": 720}}
        self.assertEqual(cui.camera_size_from_config(fixed), (1280, 720))
        self.assertIsNone(cui.camera_size_from_config({"camera": {"width": 1280, "height": 720}}))  # nativa
        self.assertIsNone(cui.camera_size_from_config({}))

    def test_recorder_keeps_camera_meta(self):
        import gaze_session as gs
        from calibration import CalibrationSession

        rec = gs.SessionRecorder({"version": 1, "screen": {"x": 0, "y": 0, "width": 100, "height": 100}})
        s = CalibrationSession(100, 100)
        for i in range(5):
            rec.add(TrackingFrame(timestamp=i / 30, image=np.zeros((720, 1280, 3), np.uint8), fps=29.0 + i,
                                  camera_info="1280x720 MJPG"), s)
        meta = rec.camera_meta()
        self.assertEqual(meta, {"info": "1280x720 MJPG", "image_size": [1280, 720], "fps_median": 31.0})
        with tempfile.TemporaryDirectory() as d:
            loaded = gs.load_session(rec.save(d))
        self.assertEqual(loaded.meta["camera"]["info"], "1280x720 MJPG")


if __name__ == "__main__":
    unittest.main()
