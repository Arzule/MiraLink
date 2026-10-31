"""Parte A: fase de validación, grabación de sesiones, sesión sintética y replay sin cámara."""

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest

import numpy as np

import gaze_session as gs
from calibration import CalibrationSession, grid_points, validation_points
from eye_tracker import FEATURE_DIM, TrackingFrame

W, H = 2560, 1440
QUIET_USER = dict(
    iris_noise=0.0, spike_rate=0.0, fixation_px=0.0, head_sigma_mm=0.0, head_sigma_rad=0.0,
    validation_shift_mm=(0, 0, 0), validation_shift_rad=(0, 0, 0), pose_noise_rad=0.0, blink_period_s=1e9,
)


def feed(session, t0, duration, fps=30.0, valid=True):
    t = t0
    for i in range(int(round(duration * fps)) + 1):
        t = t0 + i / fps
        session.update(t, np.zeros(FEATURE_DIM) if valid else None, valid)
    return t0 + duration


class TestValidationPhase(unittest.TestCase):
    def test_validation_points_form_a_central_rectangle(self):
        from calibration import CALIBRATION_LAYOUT, CALIBRATION_MARGIN, calibration_points

        val = validation_points(W, H)
        self.assertEqual(len(val), 5)
        corners = [val[i] for i in (0, 1, 3, 4)]
        xs, ys = sorted({p[0] for p in corners}), sorted({p[1] for p in corners})
        self.assertEqual((len(xs), len(ys)), (2, 2))  # 4 esquinas de un rectángulo
        self.assertAlmostEqual((xs[0] + xs[1]) / 2, W / 2)
        self.assertAlmostEqual((ys[0] + ys[1]) / 2, H / 2)
        self.assertEqual(val[2], (W / 2, H / 2))  # y su centro
        calib = calibration_points(W, H, CALIBRATION_MARGIN, CALIBRATION_LAYOUT)
        for p in corners:  # las esquinas son puntos nuevos, lejos de la calibración
            self.assertNotIn(p, calib)
            self.assertGreater(min(np.hypot(p[0] - g[0], p[1] - g[1]) for g in calib), 0.1 * H)

    def test_session_runs_calibration_then_validation(self):
        s = CalibrationSession(W, H, validation=validation_points(W, H))
        t = 0.0
        stages = []
        while not s.finished and t < 60:
            stages.append((s.stage, s.active_index, s.current_point))
            t = feed(s, t, 0.1)
        self.assertEqual(s.phase, "done")
        res = s.result()
        self.assertEqual(set(res.groups.tolist()), set(range(9)))
        self.assertTrue(res.has_validation)
        self.assertEqual(set(res.val_groups.tolist()), set(range(5)))
        for g in range(5):
            self.assertTrue(np.allclose(res.val_targets[res.val_groups == g], validation_points(W, H)[g]))
        val_stages = [st for st in stages if st[0] == "validation"]
        self.assertEqual(val_stages[0][2], validation_points(W, H)[0])

    def test_without_validation_is_unchanged(self):
        s = CalibrationSession(W, H)
        t = 0.0
        while not s.finished and t < 60:
            t = feed(s, t, 0.1)
        self.assertEqual(s.phase, "done")
        self.assertFalse(s.result().has_validation)

    def test_failed_validation_point_is_skipped_not_aborted(self):
        s = CalibrationSession(W, H, validation=validation_points(W, H))
        t = 0.0
        while s.stage == "calibration" and t < 60:
            t = feed(s, t, 0.1)
        # el primer punto de validación nunca recibe rostro: 1 intento + 2 reintentos y se omite
        while s.validation_index == 0 and not s.finished:
            t = feed(s, t, 0.1, valid=False)
        self.assertIn("omitió", s.message)
        while not s.finished and t < 200:
            t = feed(s, t, 0.1)
        self.assertEqual(s.phase, "done")
        self.assertEqual(set(s.result().val_groups.tolist()), {1, 2, 3, 4})


class TestRecorderAndSession(unittest.TestCase):
    def test_synthetic_session_reproduces_flow(self):
        sess = gs.synthesize_session(seed=1)
        cal = sess.capture_mask("calibration")
        val = sess.capture_mask("validation")
        self.assertEqual(set(sess.point[cal].tolist()), set(range(9)))
        self.assertEqual(set(sess.point[val].tolist()), set(range(5)))
        shown = [tuple(p) for p in sess.meta["calibration_points"]]  # orden de presentación (balanceado)
        self.assertEqual(sorted(shown), sorted(grid_points(W, H)))
        self.assertEqual(sorted(sess.meta["calibration_order"]), list(range(9)))
        for g in range(9):
            self.assertTrue(np.allclose(sess.target[cal & (sess.point == g)], shown[g]))
        dt = np.diff(sess.t)
        self.assertTrue((dt > 0).all())  # timestamps crecientes
        self.assertAlmostEqual(float(np.median(dt)), 1 / 30, delta=0.005)
        self.assertTrue(sess.closed.any())  # hubo parpadeos
        self.assertEqual(sess.features.shape[1], FEATURE_DIM)

    def test_save_and_load_round_trip(self):
        sess_src = gs.synthesize_session(seed=2)
        rec = gs.SessionRecorder(sess_src.meta, keep_landmarks=True)
        lm = np.random.default_rng(0).random((478, 2))
        frame = TrackingFrame(timestamp=1.0, face_found=True, features=np.arange(FEATURE_DIM, dtype=float),
                              landmarks=lm, matrix=np.eye(4))
        s = CalibrationSession(W, H, validation=validation_points(W, H))
        rec.add(frame, s)
        rec.add(TrackingFrame(timestamp=1.04, face_found=False), s)
        with tempfile.TemporaryDirectory() as d:
            path = rec.save(d)
            path2 = rec.save(d)
            self.assertNotEqual(path, path2)  # nunca sobrescribe
            self.assertTrue(os.path.basename(path).startswith("session_"))
            loaded = gs.load_session(path)
        self.assertEqual(len(loaded), 2)
        np.testing.assert_allclose(loaded.features[0], np.arange(FEATURE_DIM))
        self.assertTrue(np.isnan(loaded.features[1]).all())
        np.testing.assert_allclose(loaded.landmarks[0], lm, atol=1e-6)
        self.assertTrue(np.isnan(loaded.landmarks[1]).all())
        np.testing.assert_allclose(loaded.matrix[0], np.eye(4))
        self.assertEqual(loaded.meta["screen"], {"x": 0, "y": 0, "width": W, "height": H})
        self.assertEqual(loaded.meta["n_frames"], 2)
        self.assertEqual(tuple(loaded.target[0]), grid_points(W, H)[0])
        self.assertFalse(loaded.frame(1).face_found)

    def test_load_rejects_invalid_files(self):
        with tempfile.TemporaryDirectory() as d:
            bad = os.path.join(d, "x.npz")
            with open(bad, "wb") as f:
                f.write(b"no es npz")
            with self.assertRaises(ValueError):
                gs.load_session(bad)
            partial = os.path.join(d, "p.npz")
            np.savez(partial, meta=np.array('{"version": 1}'), t=np.zeros(3))
            with self.assertRaises(ValueError):
                gs.load_session(partial)

    def test_old_feature_width_is_padded(self):
        rec = gs.SessionRecorder({})
        s = CalibrationSession(W, H)
        rec.add(TrackingFrame(timestamp=0.0, face_found=True, features=np.ones(5)), s)
        arr = rec.arrays()
        self.assertEqual(arr["features"].shape, (1, max(5, FEATURE_DIM)))
        self.assertTrue((arr["features"][0, 5:] == 0).all())


class TestEvaluation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sess = gs.synthesize_session(seed=3)

    def test_metrics_are_finite_and_use_validation_points(self):
        rows = gs.evaluate(self.sess, gs.filter_variants())
        self.assertEqual(len(rows), 5)
        for r in rows:
            self.assertEqual(r["stage"], "validation")
            self.assertEqual(r["points"], 5)
            for k in ("bias_px", "bias_deg", "jitter_px", "latency_ms"):
                self.assertTrue(np.isfinite(r[k]), (r["name"], k))
        expo, raw, euro, euro_med, full = rows
        self.assertLess(expo["jitter_px"], raw["jitter_px"])  # el suavizado reduce el jitter
        self.assertEqual(raw["latency_ms"], 0.0)  # sin suavizado no hay retraso
        self.assertGreater(expo["latency_ms"], 0.0)
        # parte B: la mediana y el One Euro reducen el jitter respecto del exponencial anterior
        self.assertLess(euro_med["jitter_px"], expo["jitter_px"])
        self.assertLess(full["jitter_px"], expo["jitter_px"] * 0.9)
        self.assertLessEqual(full["latency_ms"], 150)
        table = gs.format_table(rows)
        self.assertIn("Desvío px", table)
        self.assertIn("validación", table)

    def test_noise_free_user_is_mapped_accurately(self):
        sess = gs.synthesize_session(seed=4, user_kwargs=QUIET_USER)
        rows = gs.evaluate(sess, [gs.Variant("sin filtro", gs.fit_baseline, {"smoothing_percent": 0})])
        self.assertLess(rows[0]["bias_px"], 40)
        self.assertLess(rows[0]["jitter_px"], 2)

    def test_px_to_deg(self):
        deg = gs.px_to_deg(W / 597.0 * 10.47, self.sess, 600.0)  # ~10.5 mm a 60 cm ~ 1 grado
        self.assertAlmostEqual(deg, 1.0, delta=0.01)
        self.assertTrue(np.isnan(gs.px_to_deg(10, gs.RecordedSession.from_arrays(
            {k: getattr(self.sess, k) for k in gs.REQUIRED_ARRAYS}, {"screen": self.sess.meta["screen"]}))))

    def test_session_without_validation_falls_back_to_calibration(self):
        sess = gs.synthesize_session(seed=5, validation=False)
        rows = gs.evaluate(sess, [gs.Variant("x")])
        self.assertEqual(rows[0]["stage"], "calibration")


class TestReplayCli(unittest.TestCase):
    def test_replay_script_prints_table(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.insert(0, os.path.join(root, "scripts"))
        try:
            import replay_session
        finally:
            sys.path.pop(0)
        sess = gs.synthesize_session(seed=6)
        with tempfile.TemporaryDirectory() as d:
            arrays = {k: getattr(sess, k) for k in gs.REQUIRED_ARRAYS}
            path = os.path.join(d, "session_test.npz")
            np.savez_compressed(path, meta=np.array(json.dumps(dict(sess.meta))), **arrays)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = replay_session.main([path])
            self.assertEqual(code, 0, out.getvalue())
            text = out.getvalue()
            self.assertIn("SINTÉTICA", text)
            self.assertIn("anterior", text)
            self.assertIn("Jitter px", text)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(replay_session.main([os.path.join(d, "no_existe.npz")]), 1)


if __name__ == "__main__":
    unittest.main()
