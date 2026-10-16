import json
import os
import tempfile
import unittest

import numpy as np

import calibration as cal
from calibration import (
    CalibrationError,
    CalibrationSession,
    GazeMapper,
    HoldToExit,
    face_guidance,
    grid_points,
    load_calibration,
    save_calibration,
)

W, H = 1920, 1080
GEOM = {"x": 0, "y": 0, "width": W, "height": H}


def synth_features(targets, rng, noise=0.002):
    """Rasgos de 9 valores generados por una función conocida (mirada -> iris) más ruido."""
    gx = (targets[:, 0] / W - 0.5) * 2
    gy = (targets[:, 1] / H - 0.5) * 2
    n = len(targets)
    along = 0.18 * gx + 0.02 * gx * gy
    perp = 0.06 * gy + 0.01 * gx**2
    feats = np.zeros((n, 9))
    feats[:, 0] = along
    feats[:, 2] = along * 0.98
    feats[:, 1] = perp
    feats[:, 3] = perp * 1.02
    feats[:, 4] = 0.5 + 0.01 * gx
    feats[:, 5] = 0.5 + 0.01 * gy
    feats[:, 6] = 0.2
    feats[:, 7] = 0.05 * gx
    feats[:, 8] = 0.03 * gy
    return feats + rng.normal(0, noise, feats.shape)


def make_dataset(per_point=45, seed=0):
    rng = np.random.default_rng(seed)
    pts = np.array(grid_points(W, H))
    targets = np.repeat(pts, per_point, axis=0)
    groups = np.repeat(np.arange(9), per_point)
    return synth_features(targets, rng), targets, groups, rng


class TestGazeMapper(unittest.TestCase):
    def test_grid_points_order_and_margins(self):
        pts = grid_points(W, H)
        self.assertEqual(len(pts), 9)
        self.assertEqual(pts[0], (W * 0.1, H * 0.1))
        self.assertEqual(pts[1], (W * 0.5, H * 0.1))
        self.assertEqual(pts[3], (W * 0.1, H * 0.5))
        self.assertEqual(pts[8], (W * 0.9, H * 0.9))

    def test_fit_recovers_known_function(self):
        x, y, groups, rng = make_dataset()
        mapper = GazeMapper()
        metrics = mapper.fit(x, y, groups)
        self.assertLess(metrics["rmse_px"], 40)
        self.assertIn("cv_rmse_px", metrics)
        self.assertGreaterEqual(metrics["cv_rmse_px"], metrics["rmse_px"] * 0.5)
        self.assertEqual(metrics["n_points"], 9)
        # Generaliza a puntos intermedios no vistos
        test_t = np.column_stack([rng.uniform(0.2, 0.8, 50) * W, rng.uniform(0.2, 0.8, 50) * H])
        pred = mapper.predict(synth_features(test_t, rng))
        err = np.sqrt(np.mean(np.sum((pred - test_t) ** 2, axis=1)))
        self.assertLess(err, 90)

    def test_full_feature_set_also_fits(self):
        x, y, groups, _ = make_dataset()
        mapper = GazeMapper(feature_set="full")
        self.assertLess(mapper.fit(x, y, groups)["rmse_px"], 80)

    def test_predict_before_fit_fails(self):
        with self.assertRaises(RuntimeError):
            GazeMapper().predict(np.zeros((1, 9)))

    def test_fit_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            GazeMapper().fit(np.zeros((2, 9)), np.zeros((2, 2)))
        x, y, _, _ = make_dataset()
        x[0, 0] = np.nan
        with self.assertRaises(ValueError):
            GazeMapper().fit(x, y)

    def test_constant_features_do_not_blow_up(self):
        x, y, groups, rng = make_dataset()
        x[:, 4:] = 0.5  # cabeza completamente quieta -> std 0
        mapper = GazeMapper()
        mapper.fit(x, y, groups)
        self.assertTrue(np.isfinite(mapper.predict(x)).all())

    def test_serialization_round_trip(self):
        x, y, groups, _ = make_dataset()
        mapper = GazeMapper()
        mapper.fit(x, y, groups)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "calibration.json")
            save_calibration(path, mapper, GEOM, camera_index=1)
            with open(path, encoding="utf-8") as f:
                raw = json.load(f)
            self.assertEqual(raw["camera_index"], 1)
            self.assertEqual(raw["screen"], GEOM)
            self.assertIn("created", raw)
            loaded = load_calibration(path, GEOM)
            np.testing.assert_allclose(loaded.predict(x[:20]), mapper.predict(x[:20]), atol=1e-6)
            self.assertAlmostEqual(loaded.estimated_error_px, mapper.estimated_error_px)

    def test_invalidated_when_screen_geometry_changes(self):
        x, y, groups, _ = make_dataset()
        mapper = GazeMapper()
        mapper.fit(x, y, groups)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "calibration.json")
            save_calibration(path, mapper, GEOM)
            with self.assertRaises(CalibrationError) as ctx:
                load_calibration(path, {"x": 0, "y": 0, "width": 2560, "height": 1440})
            self.assertIn("geometría", str(ctx.exception))
            load_calibration(path, GEOM)  # la original sigue siendo válida

    def test_invalid_files(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "calibration.json")
            with self.assertRaises(CalibrationError):
                load_calibration(path, GEOM)  # no existe
            with open(path, "w") as f:
                f.write("{no es json")
            with self.assertRaises(CalibrationError):
                load_calibration(path, GEOM)
            with open(path, "w") as f:
                json.dump({"version": 1, "screen": GEOM}, f)
            with self.assertRaises(CalibrationError):
                load_calibration(path, GEOM)
            with open(path, "w") as f:
                json.dump([1, 2], f)
            with self.assertRaises(CalibrationError):
                load_calibration(path, GEOM)

    def test_nan_weights_rejected(self):
        x, y, groups, _ = make_dataset()
        mapper = GazeMapper()
        mapper.fit(x, y, groups)
        data = mapper.to_dict(GEOM)
        data["weights"][0][0] = float("nan")
        with self.assertRaises(CalibrationError):
            GazeMapper.from_dict(data, GEOM)


def feed(session, t0, duration, fps, valid=True, features=None):
    """Alimenta la sesión con cuadros a `fps` (solo timestamps, sin reloj real). Devuelve el último ts."""
    feats = np.zeros(9) if features is None else features
    n = int(round(duration * fps)) + 1  # incluye el cuadro del instante final
    t = t0
    for i in range(n):
        t = t0 + i / fps
        session.update(t, feats if valid else None, valid)
    return t0 + duration


class TestCalibrationSession(unittest.TestCase):
    def test_full_run_collects_nine_points(self):
        s = CalibrationSession(W, H)
        t = 0.0
        for p in range(9):
            self.assertEqual(s.point_index, p)
            t = feed(s, t, 1.0, 30)  # asentamiento
            self.assertEqual(s.phase, "capturing")
            t = feed(s, t, 1.5, 30)  # captura
            s.update(t, np.zeros(9), True)  # cierra la captura
            t += 1 / 30
        self.assertEqual(s.phase, "done")
        res = s.result()
        self.assertEqual(set(res.groups.tolist()), set(range(9)))
        self.assertEqual(res.features.shape[1], 9)
        # el objetivo de cada muestra es el punto de su grupo
        for g in range(9):
            tg = res.targets[res.groups == g]
            self.assertTrue(np.allclose(tg, s.points[g]))

    def test_settling_samples_are_discarded(self):
        s = CalibrationSession(W, H)
        s.update(0.0, np.ones(9) * 7, True)
        s.update(0.5, np.ones(9) * 7, True)
        self.assertEqual(s.phase, "settling")
        self.assertEqual(s.sample_count, 0)

    def test_timing_depends_on_timestamps_not_frame_count(self):
        # 10 fps: 15 cuadros de captura >= 20 no se alcanzan -> reintento; con 20 fps sí
        slow = CalibrationSession(W, H)
        t = feed(slow, 0.0, 1.0, 10)
        feed(slow, t, 1.5, 10)
        slow.update(2.6, np.zeros(9), True)
        self.assertEqual(slow.retries, 1)
        fast = CalibrationSession(W, H)
        t = feed(fast, 0.0, 1.0, 25)
        feed(fast, t, 1.5, 25)
        fast.update(2.6, np.zeros(9), True)
        self.assertEqual(fast.point_index, 1)

    def test_invalid_frames_not_counted_and_retry_then_abort(self):
        s = CalibrationSession(W, H)
        t = 0.0
        for attempt in range(3):  # intento inicial + 2 reintentos
            t = feed(s, t, 1.0, 30)
            t = feed(s, t, 1.5, 30, valid=False)
            s.update(t, None, False)
            t += 1 / 30
            if attempt < 2:
                self.assertEqual(s.retries, attempt + 1)
                self.assertEqual(s.phase, "settling")
                self.assertIn("reintentando", s.message)
        self.assertEqual(s.phase, "aborted")
        self.assertIn("punto 1", s.message)
        with self.assertRaises(RuntimeError):
            s.result()

    def test_retry_succeeds_and_resets_counter(self):
        s = CalibrationSession(W, H)
        t = feed(s, 0.0, 1.0, 30)
        t = feed(s, t, 1.5, 30, valid=False)
        s.update(t, None, False)
        t += 1 / 30
        self.assertEqual(s.retries, 1)
        t = feed(s, t, 1.0, 30)
        t = feed(s, t, 1.5, 30)
        s.update(t, np.zeros(9), True)
        self.assertEqual(s.point_index, 1)
        self.assertEqual(s.retries, 0)

    def test_ticks_without_frames_eventually_abort(self):
        s = CalibrationSession(W, H)
        t = 0.0
        while not s.finished and t < 60:
            s.update(t, None, False)
            t += 0.1
        self.assertEqual(s.phase, "aborted")

    def test_fit_session_end_to_end(self):
        rng = np.random.default_rng(1)
        s = CalibrationSession(W, H)
        t = 0.0
        for p in range(9):
            target = np.array([s.points[p]])
            t = feed(s, t, 1.0, 30, valid=False)
            for i in range(45):
                s.update(t + i / 30, synth_features(target, rng)[0], True)
            t += 1.5
            s.update(t, np.zeros(9), True)
            t += 1 / 30
        self.assertEqual(s.phase, "done")
        mapper = cal.fit_session(s)
        self.assertLess(mapper.metrics["rmse_px"], 40)


class TestHelpers(unittest.TestCase):
    def test_hold_to_exit(self):
        h = HoldToExit(1.5)
        self.assertEqual(h.progress(0.0), 0.0)
        h.press(10.0)
        h.press(10.5)  # autorepetición: no reinicia
        self.assertAlmostEqual(h.progress(10.75), 0.5)
        self.assertFalse(h.completed(11.0))
        self.assertTrue(h.completed(11.5))
        h.release()
        self.assertEqual(h.progress(12.0), 0.0)

    def test_face_guidance(self):
        self.assertEqual(face_guidance(False, None, 0.0), (False, "Rostro no detectado"))
        self.assertEqual(face_guidance(True, (0.5, 0.5), 0.05), (True, "Acércate"))
        self.assertEqual(face_guidance(True, (0.5, 0.5), 0.5), (True, "Aléjate"))
        self.assertEqual(face_guidance(True, (0.9, 0.5), 0.24), (True, "Centra tu rostro"))
        self.assertEqual(face_guidance(True, (0.5, 0.5), 0.24), (True, ""))
        # 0.19 (sesiones reales 2 y 3: jitter doble) ya pide acercarse; el umbral es 0.21
        self.assertEqual(face_guidance(True, (0.5, 0.5), 0.19), (True, "Acércate"))
        self.assertEqual(face_guidance(True, (0.5, 0.5), 0.21), (True, ""))


if __name__ == "__main__":
    unittest.main()
