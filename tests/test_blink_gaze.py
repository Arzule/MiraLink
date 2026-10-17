"""Parpadeos y mirada hacia abajo, cámara nativa y corrección automática solo si mejora.

Hallazgo de las sesiones reales del 16/10: al mirar la fila inferior el EAR caía a la mitad y el
BlinkDetector lo tomaba por un cierre (calibraciones abortadas y riesgo de clics falsos).
Sin hardware: cv2 falso, rasgos sintéticos y relojes inyectados.
"""

import json
import os
import tempfile
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

import calibration as cal
import eye_tracker as et
import gaze_session as gs
from calibration import CalibrationSession, GazeEarModel, GazeMapper, local_open_mask, validation_points
from config_manager import DEFAULT_CONFIG, camera_size, load_config
from cursor_control import CursorController
from eye_tracker import BlinkDetector, BlinkEvent, CameraSource, TrackingFrame
from test_calibration import GEOM, H, W, synth_features
from test_cursor_control import Base, FakeMouse, IdentityMapper, frame

# EAR abierto según la altura del iris, como en la sesión real 223137 (EAR = 0.23 - 2.3 x altura)
C0, C1 = 0.23, -2.3


def open_ear(perp):
    return C0 + C1 * perp


def model():
    return GazeEarModel(C0, C1, perp_min=-0.05, perp_max=0.05, perp_ref=0.0)


def feats(perp):
    f = np.zeros(10)
    f[1] = f[3] = perp
    return f


def run_detector(det, seq, t0=0.0, fps=30.0):
    """seq: lista de (segundos, perp, ear). Devuelve (eventos, fracción de cuadros cerrados por tramo)."""
    t, events, closed = t0, [], []
    for seconds, perp, ear in seq:
        n = int(round(seconds * fps))
        c = 0
        for _ in range(n):
            ev = det.update(t, ear, True, perp)
            c += det.is_closed
            if ev is not None:
                events.append(ev)
            t += 1.0 / fps
        closed.append(c / max(n, 1))
    return events, closed, t


class TestGazeEarModel(unittest.TestCase):
    def test_fit_factor_and_guard(self):
        pts = np.array([[p, open_ear(p)] for p in (-0.04, -0.02, 0.0, 0.02, 0.04)])
        m = GazeEarModel.fit(pts)
        self.assertAlmostEqual(m.c0, C0, places=6)
        self.assertAlmostEqual(m.c1, C1, places=6)
        self.assertEqual((m.perp_min, m.perp_max, m.perp_ref), (-0.04, 0.04, 0.0))
        self.assertAlmostEqual(m.factor(0.04), open_ear(0.04) / C0, places=6)
        self.assertAlmostEqual(m.factor(0.5), m.factor(0.04), places=9)  # sin extrapolar fuera del rango
        self.assertEqual(m.low_end, 0.04)
        self.assertTrue(m.beyond_low_end(0.05))
        self.assertFalse(m.beyond_low_end(0.03))
        self.assertTrue(m.guard_active)  # en la fila más baja el ojo se abre un 60 %
        flat = GazeEarModel.fit(np.array([[p, 0.3 - 0.3 * p] for p in (-0.1, 0.0, 0.1)]))
        self.assertFalse(flat.guard_active)  # efecto pequeño (primer monitor): sin seguro de clic

    def test_fit_rejects_bad_data_and_round_trips(self):
        self.assertIsNone(GazeEarModel.fit(None))
        self.assertIsNone(GazeEarModel.fit(np.array([[0.0, 0.3], [0.01, 0.29]])))  # < 3 puntos
        self.assertIsNone(GazeEarModel.fit(np.array([[0.0, 0.3]] * 4)))  # sin rango de altura
        self.assertIsNone(GazeEarModel.fit(np.array([[-0.1, 0.3], [0.0, 0.1], [0.1, -0.1]])))  # EAR negativo
        m = model()
        self.assertEqual(GazeEarModel.from_dict(m.to_dict()), m)
        self.assertIsNone(GazeEarModel.from_dict({"c0": 1}))
        self.assertIsNone(GazeEarModel.from_dict({**m.to_dict(), "c1": float("nan")}))


class TestLocalOpenMask(unittest.TestCase):
    def test_drops_point_blinks_and_150ms_after_but_not_low_lid(self):
        t = np.arange(45) / 30
        ear = np.full(45, 0.12)  # mirando abajo: EAR bajo todo el punto, no es un cierre
        ear[20:24] = 0.03  # parpadeo real
        keep = local_open_mask(t, ear)
        self.assertFalse(keep[20:24].any())
        self.assertFalse(keep[24:28].any())  # 150 ms tras el cierre
        self.assertTrue(keep[28:].all())
        self.assertTrue(keep[:20].all())
        self.assertTrue(local_open_mask(t, np.full(45, np.nan)).all())  # sin EAR no se descarta nada


class TestCalibrationWithLowLid(unittest.TestCase):
    def run_session(self, use_ear):
        rng = np.random.default_rng(0)
        s = CalibrationSession(W, H, validation=validation_points(W, H))
        t = 0.0
        while not s.finished and t < 120:
            target = np.array([s.current_point])
            f = synth_features(target, rng)[0]
            perp = (f[1] + f[3]) / 2
            ear = open_ear(perp) + rng.normal(0, 0.005)
            blinking = (t % 3.0) < 0.15
            if blinking:
                ear = 0.03
            # el detector global marca "cerrado" en la fila de abajo (como en las sesiones reales)
            global_closed = blinking or target[0, 1] > 0.8 * H
            s.update(t, f, True, eyes_closed=global_closed, ear=ear if use_ear else None)
            t += 1 / 30
        return s

    def test_low_lid_rows_no_longer_abort_and_model_is_fitted(self):
        self.assertEqual(self.run_session(use_ear=False).phase, "aborted")  # comportamiento anterior
        s = self.run_session(use_ear=True)
        self.assertEqual(s.phase, "done")
        res = s.result()
        self.assertEqual(len(res.ear_points), 9 + 5)
        mapper = cal.fit_session(s)
        self.assertIsNotNone(mapper.blink_model)
        self.assertLess(mapper.blink_model.c1, 0)  # mirar abajo cierra el párpado
        self.assertTrue(mapper.blink_model.guard_active)


class TestBlinkDetectorWithGaze(unittest.TestCase):
    LOOK_DOWN = 0.045  # fila inferior: EAR esperado 0.127 (frente a 0.23 de frente: 0.55 veces)

    def warmup(self):
        return [(2.0, 0.0, open_ear(0.0))]

    def test_without_model_looking_down_is_a_closure(self):
        _, closed, _ = run_detector(BlinkDetector(), self.warmup() + [(3.0, self.LOOK_DOWN, open_ear(self.LOOK_DOWN))])
        self.assertGreater(closed[1], 0.9)  # el defecto que se corrige

    def test_with_model_looking_down_is_open_and_blinks_still_detected(self):
        det = BlinkDetector(gaze_model=model())
        seq = self.warmup() + [
            (3.0, self.LOOK_DOWN, open_ear(self.LOOK_DOWN)),  # mirar abajo 3 s
            (0.3, self.LOOK_DOWN + 0.02, 0.03),  # parpadeo mirando abajo (el iris "cae" al cerrar)
            (1.0, self.LOOK_DOWN, open_ear(self.LOOK_DOWN)),
            (1.0, 0.0, open_ear(0.0)),
            (0.3, 0.03, 0.03),  # parpadeo mirando al frente
            (1.0, 0.0, open_ear(0.0)),
        ]
        events, closed, _ = run_detector(det, seq)
        self.assertLess(closed[1], 0.05)
        self.assertEqual(len(events), 2)
        for ev in events:
            self.assertAlmostEqual(ev.duration_ms, 300, delta=40)
            self.assertFalse(ev.gaze_suspect)

    def test_blink_below_lowest_row_is_suspect(self):
        det = BlinkDetector(gaze_model=model())
        below = 0.055  # más abajo que la fila más baja calibrada (p. ej. la barra de tareas)
        events, _, _ = run_detector(det, self.warmup() + [(1.0, below, open_ear(below)), (0.3, below, 0.02),
                                                          (0.5, below, open_ear(below))])
        self.assertEqual(len(events), 1)
        self.assertTrue(events[0].gaze_suspect)

    def test_pipeline_passes_iris_height(self):
        class Cam:
            def read(self):
                return True, np.zeros((480, 640, 3), np.uint8)

        class Tracker:
            perp = 0.0

            def process(self, image, ts):
                return TrackingFrame(timestamp=ts, face_found=True, ear_mean=open_ear(self.perp), features=feats(self.perp))

        clock = {"t": 0.0}

        def tick():
            clock["t"] += 1 / 30
            return clock["t"]

        tr = Tracker()
        p = et.EyeTrackingPipeline(Cam(), tr, clock=tick, blink_model=model())
        for _ in range(60):
            p.step()
        tr.perp = self.LOOK_DOWN
        closed = [p.step().eyes_closed for _ in range(60)]
        self.assertLess(sum(closed), 3)
        self.assertEqual(et.gaze_perp(feats(0.02)), 0.02)
        self.assertIsNone(et.gaze_perp(None))


class TestClickGuard(Base):
    def suspect(self, duration_ms):
        return BlinkEvent(self.t - duration_ms / 1000.0, self.t, duration_ms, gaze_suspect=True)

    def test_suspect_blink_does_not_click(self):
        self.hold_gaze(0.3)
        self.ctl.update(frame(self.t, blink=self.suspect(300)))
        self.assertEqual(self.mouse.names(), [])
        self.assertIn("click_blocked", self.events)

    def test_suspect_blink_still_releases_hold(self):
        self.cfg["dwell_hold_ms"] = 800
        self.hold_gaze(1.2)
        self.assertEqual(self.mouse.names(), ["down"])
        self.t += 0.5
        self.ctl.update(frame(self.t, blink=self.suspect(600)))
        self.assertEqual(self.mouse.names(), ["down", "up"])  # soltar es la dirección segura


class TestMapperBlinkModelAndAutoOffset(unittest.TestCase):
    def fitted(self):
        rng = np.random.default_rng(2)
        pts = np.repeat(np.array(cal.grid_points(W, H)), 30, axis=0)
        m = GazeMapper.from_fit_options()
        m.fit(synth_features(pts, rng), pts, np.repeat(np.arange(9), 30))
        return m

    def test_blink_model_round_trip_and_optional(self):
        m = self.fitted()
        m.blink_model = model()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "calibration.json")
            cal.save_calibration(path, m, GEOM)
            self.assertEqual(cal.load_calibration(path, GEOM).blink_model, model())
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            data["blink_model"] = {"c0": "x"}
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f)
            self.assertIsNone(cal.load_calibration(path, GEOM).blink_model)  # inválido: se ignora
            del data["blink_model"]
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f)
            self.assertIsNone(cal.load_calibration(path, GEOM).blink_model)  # calibraciones anteriores

    def test_auto_offset_skipped_without_common_bias_text(self):
        import calibration_ui as cui

        rng = np.random.default_rng(4)
        pts = np.repeat(np.array(cal.grid_points(W, H)), 30, axis=0)
        val_t = np.repeat(np.array(validation_points(W, H)), 30, axis=0)
        m = GazeMapper.from_fit_options()
        metrics = m.fit(synth_features(pts, rng, 0.004), pts, np.repeat(np.arange(9), 30),
                        synth_features(val_t, rng, 0.004), val_t, np.repeat(np.arange(5), 30))
        self.assertTrue(metrics["auto_offset_skipped"])
        self.assertGreaterEqual(metrics["validation_corrected_px"], metrics["validation_px"])
        self.assertEqual(m.error_source, "validation")
        text = cui.error_source_text("validation", 50.0, 5, skipped_px=metrics["validation_corrected_px"])
        self.assertIn("No se aplicó la corrección automática", text)


class TestSessionBlendshapesAndReport(unittest.TestCase):
    def test_blendshapes_recorded_and_old_sessions_load(self):
        rec = gs.SessionRecorder({"version": 1, "screen": {"x": 0, "y": 0, "width": 100, "height": 100}})
        s = CalibrationSession(100, 100)
        shapes = np.arange(len(et.EYE_BLENDSHAPES), dtype=float) / 10
        for i in range(4):
            rec.add(TrackingFrame(timestamp=i / 30, face_found=True, features=feats(0.0),
                                  eye_blendshapes=shapes if i else None), s)
        with tempfile.TemporaryDirectory() as d:
            loaded = gs.load_session(rec.save(d))
        self.assertEqual(loaded.blendshapes.shape, (4, len(et.EYE_BLENDSHAPES)))
        self.assertTrue(np.isnan(loaded.blendshapes[0]).all())
        np.testing.assert_allclose(loaded.blendshapes[1], shapes, rtol=1e-6)
        self.assertEqual(loaded.meta["blendshape_names"], list(et.EYE_BLENDSHAPES))
        arrays = {k: v for k, v in rec.arrays().items() if k != "blendshapes"}
        self.assertIsNone(gs.RecordedSession.from_arrays(arrays, rec.meta).blendshapes)

    def test_blink_report_on_synthetic_session(self):
        session = gs.synthesize_session(0)
        rep = gs.blink_report(session)
        self.assertIn("sin modelo", rep)
        text = gs.format_blink_report(rep)
        self.assertIn("Parpadeos", text)


class NativeCv2:
    """cv2 falso para la resolución nativa: `modes` {(ancho, alto): fps} con MJPG; un tamaño no
    soportado entrega `nearest[(ancho, alto)]` (o 640x480)."""

    CAP_DSHOW = 700
    CAP_PROP_FRAME_WIDTH = 3
    CAP_PROP_FRAME_HEIGHT = 4
    CAP_PROP_FPS = 5
    CAP_PROP_FOURCC = 6

    def __init__(self, modes, nearest=None):
        self.modes = dict(modes)
        self.nearest = dict(nearest or {})
        self.t = 0.0
        self.requests = []

    def clock(self):
        return self.t

    @staticmethod
    def VideoWriter_fourcc(*chars):  # noqa: N802
        return sum(ord(c) << (8 * i) for i, c in enumerate(chars))

    def VideoCapture(self, index, api):  # noqa: N802
        return _NativeCap(self)


class _NativeCap:
    def __init__(self, cv2):
        self.cv2, self.props, self.released = cv2, {}, False

    def isOpened(self):  # noqa: N802
        return True

    def set(self, prop, value):
        self.props[prop] = value
        if prop == NativeCv2.CAP_PROP_FRAME_HEIGHT:
            self.cv2.requests.append((int(self.props[NativeCv2.CAP_PROP_FRAME_WIDTH]), int(value)))
        return True

    def _mode(self):
        size = (int(self.props[NativeCv2.CAP_PROP_FRAME_WIDTH]), int(self.props[NativeCv2.CAP_PROP_FRAME_HEIGHT]))
        size = size if size in self.cv2.modes else self.cv2.nearest.get(size, (640, 480))
        return size, self.cv2.modes.get(size, 30.0)

    def get(self, prop):
        mjpg = self.props.get(NativeCv2.CAP_PROP_FOURCC) == NativeCv2.VideoWriter_fourcc(*"MJPG")
        return NativeCv2.VideoWriter_fourcc(*("MJPG" if mjpg else "YUY2")) if prop == NativeCv2.CAP_PROP_FOURCC else 0

    def read(self):
        (w, h), fps = self._mode()
        self.cv2.t += 1.0 / fps
        return True, np.zeros((h, w, 3), np.uint8)

    def release(self):
        self.released = True


class TestNativeResolution(unittest.TestCase):
    def setUp(self):
        et._NATIVE_CACHE.clear()

    def open(self, cv2, **kw):
        cam = CameraSource(0, None, None, cv2_module=cv2, clock=cv2.clock, **kw)
        cam.open()
        return cam

    def test_takes_the_size_the_camera_delivers(self):
        # Como la cámara del 16/10: al pedir más entrega 1280x960
        cv2 = NativeCv2({(1280, 960): 30.0, (1280, 720): 10.4}, nearest={(1920, 1080): (1280, 960)})
        cam = self.open(cv2)
        self.assertEqual(cam.actual_size, (1280, 960))
        self.assertEqual(cv2.requests, [(1920, 1080)])  # una sola apertura
        self.assertEqual(cam.describe(), "1280x960 MJPG (nativa)")

    def test_steps_down_when_large_modes_are_slow(self):
        cv2 = NativeCv2({(1920, 1080): 15.0, (1280, 960): 15.0, (1280, 720): 30.0}, nearest={(1280, 960): (1280, 960)})
        cam = self.open(cv2)
        self.assertEqual(cam.actual_size, (1280, 720))
        self.assertIn("1920x1080 solo a 15.0 fps", cam.describe())
        self.assertIn("nativa", cam.describe())

    def test_falls_back_to_640x480_without_mjpg(self):
        cv2 = NativeCv2({(1920, 1080): 10.0, (1280, 960): 10.0, (1280, 720): 10.0, (800, 600): 12.0, (640, 480): 30.0})
        cam = self.open(cv2)
        self.assertEqual(cam.actual_size, (640, 480))
        self.assertEqual(cam.fourcc, "YUY2")
        self.assertIn("800x600 solo a 12.0 fps", cam.fallback_reason)

    def test_choice_is_cached_for_the_next_open(self):
        modes = {(1920, 1080): 15.0, (1280, 720): 30.0}
        self.open(NativeCv2(modes))
        cv2 = NativeCv2(modes)
        cam = self.open(cv2)
        self.assertEqual(cam.actual_size, (1280, 720))
        self.assertEqual(cv2.requests, [(1280, 720)])  # sin repetir la búsqueda

    def test_tracker_thread_native_by_default_from_config(self):
        from PySide6.QtWidgets import QApplication

        QApplication.instance() or QApplication([])
        t = et.TrackerThread(0, camera_size=None, blink_model=model())
        cam = t._camera_factory(0)
        self.assertTrue(cam.native)
        self.assertIsNone(cam._cap)
        self.assertEqual(t.blink_model, model())


class TestControlCameraStatus(unittest.TestCase):
    def test_status_signal_with_resolution_and_fps(self):
        from PySide6.QtWidgets import QApplication

        import control_session as cs
        from cursor_control import FakeHotkey

        QApplication.instance() or QApplication([])
        session = cs.ControlSession(config_getter=dict, hotkey_factory=lambda cb: FakeHotkey(cb))
        got = []
        session.camera_status.connect(got.append)
        session._check_camera(TrackingFrame(timestamp=1.0, fps=0.0, camera_info="1280x960 MJPG (nativa)"))
        session._check_camera(TrackingFrame(timestamp=1.1, fps=29.7, camera_info="1280x960 MJPG (nativa)"))
        session._check_camera(TrackingFrame(timestamp=2.0, fps=12.0, camera_info="1280x960 MJPG (nativa)"))  # < 5 s
        session._check_camera(TrackingFrame(timestamp=7.0, fps=12.0, camera_info="1280x960 MJPG (nativa)"))
        self.assertEqual(got[0], "Cámara: 1280x960 MJPG (nativa) · 30 fps")
        self.assertEqual(len(got), 2)
        self.assertIn("menos de 25", got[1])
        session.deleteLater()


class TestCameraConfigModes(unittest.TestCase):
    def test_resolution_mode_validation_and_camera_size(self):
        self.assertEqual(DEFAULT_CONFIG["camera"]["resolution"], "native")
        self.assertIsNone(camera_size(DEFAULT_CONFIG))
        self.assertEqual(camera_size({"camera": {"resolution": "fixed", "width": 1280, "height": 720}}), (1280, 720))
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "config.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"camera": {"resolution": "4k"}}, f)
            self.assertEqual(load_config(path)["camera"]["resolution"], "native")

    def test_parse_resolution_native(self):
        import main

        self.assertEqual(main.parse_resolution(["--record-session", "--resolution", "native"]), "native")
        self.assertEqual(main.parse_resolution(["--resolution", "1280x960"]), (1280, 960))


if __name__ == "__main__":
    unittest.main()
