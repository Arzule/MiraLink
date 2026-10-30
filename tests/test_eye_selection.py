"""Elección del ojo: seguir la mirada con un ojo o con los dos (tracking.eye de config.json).

En "auto" cada calibración ajusta con cada ojo y con los dos y se queda con el de menor temblor entre los
de error parecido al mejor; el ojo elegido se guarda en calibration.json y en config.json
(tracking.selected_eye) y decide también los parpadeos. Sin hardware: rasgos sintéticos.
"""

import json
import math
import os
import tempfile
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtWidgets import QApplication

import calibration as cal
import config_manager as cm
import eye_tracker as et
import gaze_session as gs
from calibration import CalibrationError, CalibrationSession, GazeMapper, choose_eye, fit_session
from eye_tracker import EyeTrackingPipeline, TrackingFrame
from test_calibration import GEOM, H, W

OPEN_EAR = 0.3


def eye_features(target, rng, noise=(0.002, 0.002), gain=(1.0, 1.0)):
    """10 rasgos: cada iris sigue la mirada con su ganancia (0 = ojo que no sigue) y su ruido."""
    gx = (target[0] / W - 0.5) * 2
    gy = (target[1] / H - 0.5) * 2
    along = 0.18 * gx + 0.02 * gx * gy
    perp = 0.06 * gy + 0.01 * gx**2
    f = np.zeros(10)
    for k, (n, g) in enumerate(zip(noise, gain)):
        f[2 * k] = g * along + rng.normal(0, n)
        f[2 * k + 1] = g * perp + rng.normal(0, n)
    f[4], f[5], f[6] = 0.5, 0.5, 0.24
    return f


def run_session(seed=0, noise=(0.002, 0.002), gain=(1.0, 1.0), right_ear_drop=0.0):
    """Calibración + validación completas a 30 fps. El EAR derecho baja `right_ear_drop` al mirar abajo."""
    rng = np.random.default_rng(seed)
    s = CalibrationSession(W, H, validation=cal.validation_points(W, H), order="balanced", rng=np.random.default_rng(seed))
    t = 0.0
    while not s.finished and t < 200:
        target = s.current_point
        f = eye_features(target, rng, noise, gain)
        right = OPEN_EAR * (1.0 - right_ear_drop * target[1] / H)
        s.update(t, f, True, ear=(right + OPEN_EAR) / 2, ears=(right, OPEN_EAR))
        t += 1 / 30
    assert s.phase == "done", s.message
    return s


class TestEyeFeatures(unittest.TestCase):
    def test_reduce_features_per_eye(self):
        x = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.5, 0.2, 0.0, 0.0, 0.0]])
        np.testing.assert_allclose(cal.reduce_features(x, "eyes"), [[0.2, 0.3]])
        np.testing.assert_allclose(cal.reduce_features(x, "eyes", "right"), [[0.1, 0.2]])
        np.testing.assert_allclose(cal.reduce_features(x, "eyes", "left"), [[0.3, 0.4]])
        np.testing.assert_allclose(cal.reduce_features(x, "compact", "left")[0, :2], [0.3, 0.4])
        np.testing.assert_allclose(cal.iris_height(x, "right"), [0.2])
        np.testing.assert_allclose(cal.iris_height(x), [0.3])
        with self.assertRaises(ValueError):
            cal.reduce_features(x, "eyes", "center")
        with self.assertRaises(ValueError):
            GazeMapper(eye="center")

    def test_gaze_perp_and_frame_ear_per_eye(self):
        f = np.array([0.0, 0.02, 0.0, 0.04])
        self.assertAlmostEqual(et.gaze_perp(f), 0.03)
        self.assertAlmostEqual(et.gaze_perp(f, "right"), 0.02)
        self.assertAlmostEqual(et.gaze_perp(f, "left"), 0.04)
        frame = TrackingFrame(timestamp=0.0, face_found=True, ear_left=0.1, ear_right=0.3, ear_mean=0.2)
        self.assertEqual(et.frame_ear(frame), 0.2)
        self.assertEqual(et.frame_ear(frame, "right"), 0.3)
        self.assertEqual(et.frame_ear(frame, "left"), 0.1)


class TestChooseEye(unittest.TestCase):
    def test_lowest_jitter_among_similar_error(self):
        cands = {
            "both": {"error_px": 100.0, "jitter_px": 60.0},
            "right": {"error_px": 108.0, "jitter_px": 45.0},  # dentro del margen: gana por temblor
            "left": {"error_px": 100.0, "jitter_px": 90.0},
        }
        self.assertEqual(choose_eye(cands), "right")

    def test_eye_that_does_not_follow_is_excluded_despite_low_jitter(self):
        cands = {
            "both": {"error_px": 110.0, "jitter_px": 120.0},
            "right": {"error_px": 100.0, "jitter_px": 90.0},
            "left": {"error_px": 500.0, "jitter_px": 5.0},  # iris quieto: no tiembla pero apunta mal
        }
        self.assertEqual(choose_eye(cands), "right")

    def test_ties_missing_and_non_finite(self):
        same = {e: {"error_px": 80.0, "jitter_px": 40.0} for e in cal.EYES}
        self.assertEqual(choose_eye(same), "both")
        self.assertEqual(choose_eye({}), "both")
        self.assertEqual(choose_eye({"left": {"error_px": math.nan, "jitter_px": 1.0}}), "both")
        limit = 100.0 * (1 + cal.EYE_ERROR_TOLERANCE) + cal.EYE_ERROR_MARGIN_PX
        cands = {"both": {"error_px": 100.0, "jitter_px": 50.0}, "left": {"error_px": limit + 0.1, "jitter_px": 1.0}}
        self.assertEqual(choose_eye(cands), "both")

    def test_prediction_jitter(self):
        mapper = GazeMapper("eyes", alpha=0.01, model="linear", clamp=False)
        rng = np.random.default_rng(0)
        targets = np.array([[x, y] for x in (100, 900, 1700) for y in (100, 500, 900)], dtype=float)
        x = np.vstack([eye_features(t, rng) for t in targets for _ in range(10)])
        y = np.repeat(targets, 10, axis=0)
        g = np.repeat(np.arange(9), 10)
        mapper.fit(x, y, g)
        self.assertGreater(cal.prediction_jitter_px(mapper, x, g), 0)
        flat = np.repeat(x[:9], 1, axis=0)
        self.assertAlmostEqual(cal.prediction_jitter_px(mapper, np.repeat(flat[:1], 5, axis=0), np.zeros(5)), 0.0)
        self.assertTrue(math.isnan(cal.prediction_jitter_px(mapper, x[:2], np.arange(2))))


class TestFitSessionEye(unittest.TestCase):
    def test_auto_picks_the_eye_that_works(self):
        mapper = fit_session(run_session(gain=(1.0, 0.0)))  # el ojo izquierdo no sigue los puntos
        sel = mapper.metrics["eye_selection"]
        self.assertEqual(sel["mode"], "auto")
        self.assertEqual(sel["chosen"], "right")
        self.assertEqual(mapper.eye, "right")
        self.assertEqual(set(sel["candidates"]), set(cal.EYES))
        self.assertGreater(sel["candidates"]["left"]["error_px"], 3 * sel["candidates"]["right"]["error_px"])

    def test_auto_avoids_the_noisy_eye_and_keeps_both_when_equal(self):
        self.assertEqual(fit_session(run_session(noise=(0.008, 0.002))).eye, "left")
        self.assertEqual(fit_session(run_session(noise=(0.004, 0.004))).eye, "both")

    def test_forced_mode_and_invalid_mode(self):
        s = run_session()
        mapper = fit_session(s, eye_mode="left")
        self.assertEqual(mapper.eye, "left")
        self.assertEqual(mapper.metrics["eye_selection"], {
            "mode": "left", "chosen": "left", "candidates": {"left": mapper.metrics["eye_selection"]["candidates"]["left"]},
        })
        with self.assertRaises(ValueError):
            fit_session(s, eye_mode="center")

    def test_blink_model_uses_the_chosen_eye(self):
        # solo el párpado derecho baja al mirar abajo: el modelo del ojo derecho lo refleja
        s = run_session(gain=(1.0, 0.0), right_ear_drop=0.5)
        res = s.result()
        self.assertEqual(set(res.ear_points_by_eye), {"right", "left"})
        mapper = fit_session(s)
        self.assertEqual(mapper.eye, "right")
        model = mapper.blink_model
        self.assertIsNotNone(model)
        self.assertLess(model.c1, 0)  # mirar abajo cierra el ojo derecho
        np.testing.assert_allclose(res.ear_points_for("right"), res.ear_points_by_eye["right"])
        np.testing.assert_allclose(res.ear_points_for("both"), res.ear_points)

    def test_without_per_eye_ear_falls_back_to_both(self):
        res = cal.CalibrationResult(np.zeros((3, 10)), np.zeros((3, 2)), np.arange(3), ear_points=np.ones((3, 2)))
        np.testing.assert_allclose(res.ear_points_for("left"), np.ones((3, 2)))


class TestCalibrationFileEye(unittest.TestCase):
    def test_round_trip_and_old_files(self):
        mapper = fit_session(run_session(gain=(1.0, 0.0)))
        data = mapper.to_dict(GEOM)
        self.assertEqual(data["eye"], "right")
        json.dumps(data)  # la comparación de ojos en las métricas debe ser serializable
        loaded = GazeMapper.from_dict(data, GEOM)
        self.assertEqual(loaded.eye, "right")
        x = eye_features((500.0, 500.0), np.random.default_rng(1))
        np.testing.assert_allclose(loaded.predict(x), mapper.predict(x))
        old = dict(data)
        del old["eye"]
        self.assertEqual(GazeMapper.from_dict(old, GEOM).eye, "both")  # calibraciones anteriores: los dos ojos
        bad = dict(data, eye="center")
        with self.assertRaises(CalibrationError):
            GazeMapper.from_dict(bad, GEOM)


class TestPipelineEye(unittest.TestCase):
    class Detector:
        is_closed = False

        def __init__(self):
            self.calls = []

        def update(self, ts, ear, face, perp):
            self.calls.append((ear, perp))

    class Camera:
        def read(self):
            return True, np.zeros((4, 4, 3), np.uint8)

    class Tracker:
        def process(self, image, ts):
            return TrackingFrame(
                timestamp=ts, face_found=True, ear_left=0.1, ear_right=0.3, ear_mean=0.2,
                features=np.array([0.0, 0.02, 0.0, 0.04, 0.5, 0.5, 0.24, 0, 0, 0]),
            )

    def test_detector_gets_the_chosen_eye(self):
        for eye, expected in (("both", (0.2, 0.03)), ("right", (0.3, 0.02)), ("left", (0.1, 0.04))):
            det = self.Detector()
            EyeTrackingPipeline(self.Camera(), self.Tracker(), det, clock=lambda: 1.0, eye=eye).step()
            self.assertAlmostEqual(det.calls[0][0], expected[0])
            self.assertAlmostEqual(det.calls[0][1], expected[1])
        with self.assertRaises(ValueError):
            EyeTrackingPipeline(self.Camera(), self.Tracker(), eye="center")

    def test_tracker_thread_keeps_the_eye(self):
        QApplication.instance() or QApplication([])
        thread = et.TrackerThread(0, eye="left", camera_factory=lambda i: None, tracker_factory=lambda p: None)
        self.assertEqual(thread.eye, "left")  # no se inicia: nunca abre la cámara


class TestFourcc(unittest.TestCase):
    def test_only_ascii_letters_and_digits(self):
        def code(text):
            return sum(ord(c) << (8 * i) for i, c in enumerate(text))

        self.assertEqual(et._fourcc_text(code("YUY2")), "YUY2")
        self.assertEqual(et._fourcc_text(code("MJPG")), "MJPG")
        self.assertEqual(et._fourcc_text(code("}ë6ä")), "")  # el texto basura de la sesión real del 17/10
        self.assertEqual(et._fourcc_text(0), "")
        self.assertEqual(et._fourcc_text(None), "")


class TestConfigEye(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "config.json")

    def tearDown(self):
        self.tmp.cleanup()

    def test_defaults_and_validation(self):
        cfg = cm.load_config(self.path)
        self.assertEqual(cfg["tracking"], {"eye": "auto", "selected_eye": None})
        self.assertEqual(cm.eye_mode(cfg), "auto")
        bad = cm._validate_config(dict(cfg, tracking={"eye": "center", "selected_eye": "nose"}))
        self.assertEqual(bad["tracking"], {"eye": "auto", "selected_eye": None})
        self.assertEqual(cm._validate_config(dict(cfg, tracking="x"))["tracking"]["eye"], "auto")
        self.assertEqual(cm.eye_mode({"tracking": {"eye": "right"}}), "right")
        self.assertEqual(cm.eye_mode({}), "auto")

    def test_save_selected_eye_writes_only_that_key(self):
        cfg = cm.load_config(self.path)
        cfg["interaction"]["smoothing_percent"] = 77
        cm.save_config(cfg, self.path)
        saved = cm.save_selected_eye("left", self.path)
        self.assertEqual(saved["tracking"]["selected_eye"], "left")
        with open(self.path, encoding="utf-8") as f:
            on_disk = json.load(f)
        self.assertEqual(on_disk["tracking"], {"eye": "auto", "selected_eye": "left"})
        self.assertEqual(on_disk["interaction"]["smoothing_percent"], 77)
        with self.assertRaises(ValueError):
            cm.save_selected_eye("center", self.path)

    def test_corrupt_config_is_not_overwritten(self):
        with open(self.path, "w", encoding="utf-8") as f:
            f.write("{roto")
        with self.assertRaises(ValueError):
            cm.save_selected_eye("right", self.path)
        with open(self.path, encoding="utf-8") as f:
            self.assertEqual(f.read(), "{roto")


class TestEyeUi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_result_text(self):
        import calibration_ui as cui

        auto = {
            "mode": "auto", "chosen": "right",
            "candidates": {"both": {"error_px": 90.0, "jitter_px": 60.0}, "right": {"error_px": 95.0, "jitter_px": 40.0}},
        }
        text = cui.eye_selection_text(auto)
        self.assertIn("tu ojo derecho", text)
        self.assertIn("menor temblor", text)
        self.assertIn("ambos 60 px de temblor / 90 px de error", text)
        self.assertIn("fijado en config.json", cui.eye_selection_text({"mode": "left", "chosen": "left"}))
        self.assertEqual(cui.eye_selection_text(None), "")
        dlg = cui.CalibrationResultDialog(80.0, eye_text=text)
        self.assertEqual(dlg.eye_label.text(), text)
        self.assertFalse(dlg.eye_label.isHidden())
        plain = cui.CalibrationResultDialog(80.0)
        self.assertTrue(plain.eye_label.isHidden())

    def test_config_window_stores_the_accepted_eye(self):
        from config_ui import ConfigWindow

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "config.json")
            win = ConfigWindow(config_path=path)
            win._config["interaction"]["smoothing_percent"] = 13  # cambio sin guardar: no debe escribirse

            class Mapper:
                eye = "left"

            class Dialog:
                accepted_mapper = Mapper()

            win.calibration_dialog = Dialog()
            win._on_calibration_finished()
            with open(path, encoding="utf-8") as f:
                on_disk = json.load(f)
            self.assertEqual(on_disk["tracking"]["selected_eye"], "left")
            self.assertNotEqual(on_disk["interaction"]["smoothing_percent"], 13)
            self.assertEqual(win._config["tracking"]["selected_eye"], "left")
            win.close()
            win.deleteLater()
            self.app.processEvents()


class TestEyeReport(unittest.TestCase):
    def test_report_recommends_the_working_eye(self):
        session = gs.synthesize_session(1, user_kwargs={"eye_gain": (1.0, 0.0)})
        rep = gs.eye_report(session)
        self.assertEqual(rep["chosen"], "right")
        self.assertEqual(set(rep["eyes"]), set(cal.EYES))
        self.assertGreater(rep["eyes"]["left"]["error_px"], 3 * rep["eyes"]["right"]["error_px"])
        text = gs.format_eye_report(rep)
        self.assertIn("elegido", text)
        self.assertIn("derecho", text)

    def test_recorded_per_eye_ear(self):
        session = gs.synthesize_session(0)
        self.assertEqual(session.ear_eyes.shape, (len(session), 2))
        with tempfile.TemporaryDirectory() as d:
            rec = gs.SessionRecorder({"version": gs.SESSION_VERSION, "screen": GEOM})
            for i in range(3):
                rec.add(TrackingFrame(timestamp=float(i), face_found=True, ear_left=0.1, ear_right=0.3, ear_mean=0.2),
                        CalibrationSession(W, H))
            loaded = gs.load_session(rec.save(d))
        np.testing.assert_allclose(loaded.ear_eyes, [[0.3, 0.1]] * 3, rtol=1e-6)
        points = gs.ear_points(session, "right")
        self.assertIsNotNone(points)
        np.testing.assert_allclose(points[:, 1], OPEN_EAR, atol=0.05)


if __name__ == "__main__":
    unittest.main()
