"""Parte C: modelos de mapeo, validación con puntos nuevos, filtrado robusto, offset y "Recentrar"."""

import json
import math
import os
import tempfile
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtCore import QObject, Qt, Signal, qInstallMessageHandler
from PySide6.QtGui import QGuiApplication
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

import calibration as cal
import calibration_ui as cui
import control_session as cs
import eye_tracker as et
from calibration import (
    DEFAULT_FIT,
    CalibrationError,
    CalibrationSession,
    GazeMapper,
    RecenterSession,
    aggregate_point_samples,
    filter_point_samples,
    load_calibration,
    reduce_features,
    save_calibration,
    save_offset,
    screen_geometry_from_qt,
    validation_points,
)
from config_manager import DEFAULT_CONFIG
from cursor_control import FakeHotkey
from eye_tracker import TrackingFrame
from test_calibration import GEOM, H, W, make_dataset, synth_features
from test_cursor_control import FakeMouse

FORBIDDEN = ("QLayout", "Could not parse", "QWidget::", "QStyleSheetStyle", "QFont::", "QPainter::")


class TestFeatureSetsAndModels(unittest.TestCase):
    def test_feature_vector_has_roll(self):
        self.assertEqual(et.FEATURE_DIM, 10)
        self.assertEqual(et.FEATURE_NAMES[9], "roll")
        a = math.radians(15)
        rot_z = np.eye(4)
        rot_z[:2, :2] = [[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]]
        yaw, pitch, roll = et.head_angles_from_matrix(rot_z)
        self.assertAlmostEqual(roll, a, places=6)
        self.assertAlmostEqual(yaw, 0.0, places=6)
        self.assertAlmostEqual(pitch, 0.0, places=6)
        self.assertEqual(et.head_angles_from_matrix(None), (0.0, 0.0, 0.0))

    def test_reduce_features_dimensions_and_padding(self):
        x9 = np.arange(9, dtype=float)[None, :]
        for fs, d in (("eyes", 2), ("compact", 6), ("head", 8), ("full", 10)):
            r = reduce_features(x9, fs)
            self.assertEqual(r.shape, (1, d), fs)
        head = reduce_features(np.arange(10, dtype=float)[None, :], "head")[0]
        np.testing.assert_allclose(head, [1.0, 2.0, 4, 5, 7, 8, 9, 6])  # iris medios, nariz, yaw, pitch, roll, escala
        self.assertEqual(reduce_features(x9, "full")[0, 9], 0.0)  # roll ausente = 0
        with self.assertRaises(ValueError):
            reduce_features(x9, "otro")

    def test_term_counts(self):
        self.assertEqual(cal.n_terms(6, "linear"), 7)
        self.assertEqual(cal.n_terms(6, "quad"), 28)
        self.assertEqual(cal.n_terms(6, "quad_reduced"), 10)
        with self.assertRaises(ValueError):
            GazeMapper(model="cubico")
        with self.assertRaises(ValueError):
            GazeMapper(aggregate="media")

    def test_every_combination_fits_known_function(self):
        x, y, groups, _ = make_dataset()
        for fs in cal.IRIS_FEATURE_SETS:
            for model in cal.MODELS:
                m = GazeMapper(fs, alpha="cv", model=model)
                metrics = m.fit(x, y, groups)
                self.assertTrue(np.isfinite(metrics["cv_rmse_px"]), (fs, model))
                self.assertIn(m.alpha, cal.ALPHA_GRID)
                self.assertLess(metrics["rmse_px"], 150, (fs, model))

    def test_alpha_chosen_by_cross_validation_minimizes_cv(self):
        x, y, groups, _ = make_dataset()
        best = GazeMapper("compact", alpha="cv", model="quad")
        best.fit(x, y, groups)
        for a in cal.ALPHA_GRID:
            fixed = GazeMapper("compact", alpha=a, model="quad")
            fixed.fit(x, y, groups)
            self.assertGreaterEqual(fixed.metrics["cv_rmse_px"] + 1e-9, best.metrics["cv_rmse_px"])
        self.assertEqual(best.metrics["alpha"], best.alpha)

    def test_clamp_avoids_extrapolation(self):
        x, y, groups, _ = make_dataset()
        free = GazeMapper("compact", alpha=0.001, model="quad", clamp=False)
        bounded = GazeMapper("compact", alpha=0.001, model="quad")
        free.fit(x, y, groups)
        bounded.fit(x, y, groups)
        far = x[:1].copy()
        far[0, :4] *= 8.0  # iris muy fuera del rango calibrado
        p_free = free.predict(far)[0]
        p_bounded = bounded.predict(far)[0]
        self.assertGreater(np.abs(p_free).max(), np.abs(p_bounded).max())
        self.assertLess(np.abs(p_bounded - np.array([W / 2, H / 2])).max(), 2.0 * max(W, H))
        # dentro del rango calibrado el recorte no cambia nada
        np.testing.assert_allclose(bounded.predict(x[:50]), free.predict(x[:50]), atol=1e-9)

    def test_median_aggregation(self):
        x, y, groups, _ = make_dataset(per_point=45)
        ax, ay, ag = aggregate_point_samples(x, y, groups)
        self.assertEqual(len(ax), 9 * cal.AGGREGATE_CHUNKS)
        self.assertEqual(sorted(set(ag.tolist())), list(range(9)))
        m = GazeMapper("compact", alpha="cv", model="quad_reduced", aggregate="median")
        metrics = m.fit(x, y, groups)
        self.assertEqual(metrics["n_samples"], len(x))
        self.assertLess(metrics["rmse_px"], 80)


class TestValidationMetrics(unittest.TestCase):
    def test_validation_is_shown_error(self):
        x, y, groups, rng = make_dataset()
        val_t = np.repeat(np.array(validation_points(W, H)), 30, axis=0)
        val_g = np.repeat(np.arange(5), 30)
        val_x = synth_features(val_t, rng)
        m = GazeMapper.from_fit_options({"auto_offset": False})
        metrics = m.fit(x, y, groups, val_x, val_t, val_g)
        self.assertEqual(metrics["validation_points"], 5)
        self.assertEqual(m.error_source, "validation")
        self.assertEqual(m.estimated_error_px, metrics["validation_px"])
        self.assertLess(metrics["validation_px"], 90)
        np.testing.assert_allclose(m.offset, [0.0, 0.0])
        # sin validación: validación cruzada; sin grupos: entrenamiento
        m2 = GazeMapper.from_fit_options()
        m2.fit(x, y, groups)
        self.assertEqual(m2.error_source, "cv")
        m3 = GazeMapper.from_fit_options({"alpha": 0.02})
        m3.fit(x, y)
        self.assertEqual(m3.error_source, "train")

    def test_validation_detects_bias_that_loo_misses(self):
        # la cabeza se desplaza tras calibrar: los rasgos de validación cambian aunque la mirada no
        x, y, groups, rng = make_dataset()
        val_t = np.repeat(np.array(validation_points(W, H)), 30, axis=0)
        val_x = synth_features(val_t, rng)
        val_x[:, 0] += 0.03
        val_x[:, 2] += 0.03
        m = GazeMapper.from_fit_options()
        m.fit(x, y, groups, val_x, val_t, np.repeat(np.arange(5), 30))
        self.assertGreater(m.metrics["validation_px"], m.metrics["cv_rmse_px"])

    def test_fit_session_uses_validation_and_default_fit(self):
        rng = np.random.default_rng(1)
        s = CalibrationSession(W, H, validation=validation_points(W, H))
        t = 0.0
        while not s.finished and t < 100:
            target = np.array([s.current_point])
            s.update(t, synth_features(target, rng)[0], True)
            t += 1 / 30
        self.assertEqual(s.phase, "done")
        mapper = cal.fit_session(s)
        self.assertEqual((mapper.feature_set, mapper.model, mapper.aggregate), (
            DEFAULT_FIT["feature_set"], DEFAULT_FIT["model"], DEFAULT_FIT["aggregate"]))
        # Sin sesgo común la corrección automática no mejora el error: no se aplica
        self.assertEqual(mapper.error_source, "validation")
        self.assertTrue(mapper.metrics["auto_offset_skipped"])
        np.testing.assert_allclose(mapper.offset, [0.0, 0.0])
        self.assertIn("discarded_samples", mapper.metrics)
        self.assertLess(mapper.estimated_error_px, 100)


class TestBalancedOrderAndEyesDefault(unittest.TestCase):
    def test_balanced_order_covers_rows_and_columns_in_each_triplet(self):
        rng = np.random.default_rng(0)
        orders = {tuple(cal.balanced_order(rng)) for _ in range(30)}
        self.assertGreater(len(orders), 5)  # es aleatorio
        for order in orders:
            self.assertEqual(sorted(order), list(range(9)))
            self.assertTrue(cal.is_balanced_order(order))
            rows = [i // 3 for i in order]
            self.assertNotEqual(rows, sorted(rows))  # nunca por filas
        self.assertTrue(cal.is_balanced_order(cal.BALANCED_ORDER))
        self.assertFalse(cal.is_balanced_order(list(range(9))))

    def test_session_presents_balanced_order_consistently(self):
        s = CalibrationSession(W, H, order="balanced", rng=np.random.default_rng(3))
        grid = cal.grid_points(W, H)
        self.assertEqual(s.points, [grid[i] for i in s.order])
        self.assertEqual(sorted(s.points), sorted(grid))
        self.assertEqual(s.current_point, grid[s.order[0]])
        self.assertEqual(CalibrationSession(W, H).points, grid)  # la clase sigue por filas por defecto
        with self.assertRaises(ValueError):
            CalibrationSession(W, H, order="espiral")
        rng = np.random.default_rng(5)
        t = 0.0
        while not s.finished and t < 60:
            s.update(t, synth_features(np.array([s.current_point]), rng)[0], True)
            t += 1 / 30
        res = s.result()
        for g in range(9):
            self.assertTrue(np.allclose(res.targets[res.groups == g], s.points[g]))

    def test_default_fit_is_eyes_only(self):
        self.assertEqual(DEFAULT_FIT["feature_set"], "eyes")
        m = GazeMapper.from_fit_options()
        self.assertEqual((m.feature_set, m.model, m.alpha_mode), ("eyes", "quad_reduced", "cv"))

    def test_eyes_only_ignores_head_that_followed_the_gaze(self):
        # caso de la sesión real: la cabeza bajó en la fila inferior y se quedó baja en la validación
        x, y, groups, rng = make_dataset()
        x[groups >= 6, 5] += 0.02  # nose_y: cabeza más baja en la última fila
        x[groups >= 6, 8] += 0.03  # pitch
        val_t = np.repeat(np.array(validation_points(W, H)), 30, axis=0)
        val_x = synth_features(val_t, rng)
        val_x[:, 5] += 0.02
        val_x[:, 8] += 0.03
        val_g = np.repeat(np.arange(5), 30)
        errors = {}
        for fs in ("compact", "eyes"):
            m = GazeMapper(fs, alpha="cv", model="quad_reduced")
            errors[fs] = m.fit(x, y, groups, val_x, val_t, val_g)["validation_px"]
        self.assertLess(errors["eyes"], errors["compact"])
        self.assertLess(errors["eyes"], 60)


class TestAutoOffsetAndPoseDrift(unittest.TestCase):
    def shifted_validation(self, shift_along=0.03):
        """La cabeza/ojos derivan tras calibrar: sesgo común en todos los puntos de validación."""
        x, y, groups, rng = make_dataset()
        val_t = np.repeat(np.array(validation_points(W, H)), 30, axis=0)
        val_x = synth_features(val_t, rng)
        val_x[:, 0] += shift_along
        val_x[:, 2] += shift_along
        val_x[:, 8] += 0.05  # pitch distinto en la validación
        return x, y, groups, val_x, val_t, np.repeat(np.arange(5), 30)

    def test_common_bias_is_corrected_and_reported_honestly(self):
        x, y, groups, vx, vt, vg = self.shifted_validation()
        m = GazeMapper.from_fit_options()
        metrics = m.fit(x, y, groups, vx, vt, vg)
        self.assertEqual(m.error_source, "validation_corrected")
        self.assertLess(metrics["validation_corrected_px"], metrics["validation_px"] * 0.6)
        np.testing.assert_allclose(m.offset, metrics["auto_offset"])
        # con el offset aplicado, la mediana de cada punto queda cerca de su objetivo
        for p in range(5):
            pred = np.median(m.predict(vx[vg == p]), axis=0)
            self.assertLess(np.linalg.norm(pred - vt[vg == p][0]), metrics["validation_px"])
        # el error informado deja un punto fuera: nunca es mejor que el error en muestra con el offset
        in_sample = np.mean([np.linalg.norm(np.median(m.predict(vx[vg == p]), axis=0) - vt[vg == p][0])
                             for p in range(5)])
        self.assertGreaterEqual(metrics["validation_corrected_px"] + 1e-9, in_sample)
        # la referencia de pose del aviso es la de la validación (la que corrige el offset)
        self.assertAlmostEqual(m.pose_ref[3], float(np.median(vx[:, 8])), places=9)

    def test_absurd_offset_is_rejected(self):
        x, y, groups, vx, vt, vg = self.shifted_validation(shift_along=0.6)
        m = GazeMapper.from_fit_options()
        metrics = m.fit(x, y, groups, vx, vt, vg)
        self.assertTrue(metrics["auto_offset_rejected"])
        np.testing.assert_allclose(m.offset, [0.0, 0.0])
        self.assertEqual(m.error_source, "validation")  # se muestra el error sin corregir

    def test_needs_three_validation_points(self):
        x, y, groups, vx, vt, vg = self.shifted_validation()
        keep = vg < 2
        m = GazeMapper.from_fit_options()
        metrics = m.fit(x, y, groups, vx[keep], vt[keep], vg[keep])
        self.assertNotIn("validation_corrected_px", metrics)
        self.assertEqual(m.error_source, "validation")

    def test_pose_ref_round_trip(self):
        x, y, groups, vx, vt, vg = self.shifted_validation()
        m = GazeMapper.from_fit_options()
        m.fit(x, y, groups, vx, vt, vg)
        loaded = GazeMapper.from_dict(m.to_dict(GEOM), GEOM)
        np.testing.assert_allclose(loaded.pose_ref, m.pose_ref)
        np.testing.assert_allclose(loaded.offset, m.offset)
        data = m.to_dict(GEOM)
        data["pose_ref"] = [float("nan")] * 4
        self.assertIsNone(GazeMapper.from_dict(data, GEOM).pose_ref)  # opcional: sin aviso, pero carga

    def pose_features(self, pitch=0.0, nose_y=0.5):
        f = np.zeros(10)
        f[4], f[5], f[8] = 0.5, nose_y, pitch
        return f

    def test_drift_monitor_needs_sustained_deviation_and_has_hysteresis(self):
        mon = cal.PoseDriftMonitor([0.5, 0.5, 0.0, 0.0])
        t = 0.0
        for _ in range(60):  # quieto
            self.assertFalse(mon.update(t, self.pose_features()))
            t += 1 / 30
        for _ in range(20):  # cabeceo breve (0.66 s): no avisa
            self.assertFalse(mon.update(t, self.pose_features(pitch=0.08)))
            t += 1 / 30
        for _ in range(30):
            mon.update(t, self.pose_features())
            t += 1 / 30
        warned_at = None
        for _ in range(150):  # inclinación sostenida
            if mon.update(t, self.pose_features(pitch=0.05)) and warned_at is None:
                warned_at = t
            t += 1 / 30
        self.assertIsNotNone(warned_at)
        self.assertIn("inclinación", mon.message())
        self.assertIn("Ctrl+Alt+R", mon.message())
        for _ in range(45):  # 0.025 rad = 0.83 x tolerancia: sigue avisando (histéresis)
            mon.update(t, self.pose_features(pitch=0.025))
            t += 1 / 30
        self.assertTrue(mon.drifted)
        for _ in range(45):
            mon.update(t, self.pose_features(pitch=0.0))
            t += 1 / 30
        self.assertFalse(mon.drifted)

    def test_rebase_takes_current_pose(self):
        mon = cal.PoseDriftMonitor([0.5, 0.5, 0.0, 0.0])
        t = 0.0
        for _ in range(120):
            mon.update(t, self.pose_features(nose_y=0.56))
            t += 1 / 30
        self.assertTrue(mon.drifted)
        self.assertIn("posición vertical", mon.message())
        mon.rebase()
        self.assertFalse(mon.drifted)
        self.assertAlmostEqual(mon.reference[1], 0.56)
        for _ in range(120):
            self.assertFalse(mon.update(t, self.pose_features(nose_y=0.56)))
            t += 1 / 30

    def test_real_session_numbers_are_documented_in_tolerances(self):
        self.assertEqual(cal.POSE_TOLERANCES[3], 0.03)
        self.assertEqual(cal.IOD_MIN, 0.21)
        self.assertTrue(DEFAULT_FIT["auto_offset"])


class TestRobustSamples(unittest.TestCase):
    def test_outliers_and_pose_are_removed(self):
        rng = np.random.default_rng(0)
        x = np.zeros((60, 10))
        x[:, :4] = 0.05 + rng.normal(0, 0.004, (60, 4))
        x[:, 7:10] = rng.normal(0, 0.01, (60, 3))
        x[5, 0] += 0.2  # pico del iris
        x[10, 7] += 0.3  # giro de cabeza durante la captura
        x[20, 8] = 0.9  # pose fuera de rango
        keep = filter_point_samples(x)
        self.assertFalse(keep[5])
        self.assertFalse(keep[10])
        self.assertFalse(keep[20])
        self.assertGreater(keep.sum(), 50)

    def test_constant_samples_are_kept(self):
        self.assertTrue(filter_point_samples(np.zeros((30, 9))).all())

    def test_session_discards_post_blink_frames_and_counts_after_filtering(self):
        s = CalibrationSession(W, H)
        t = 0.0
        while s.phase == "settling":
            s.update(t, np.zeros(10), True)
            t += 1 / 30
        n0 = s.sample_count
        s.update(t, None, False, eyes_closed=True)
        t += 0.05
        s.update(t, np.zeros(10), True)  # 50 ms tras el parpadeo: se descarta
        self.assertEqual(s.sample_count, n0)
        t += 0.15
        s.update(t, np.zeros(10), True)
        self.assertEqual(s.sample_count, n0 + 1)

    def test_outliers_count_against_min_samples(self):
        rng = np.random.default_rng(2)
        s = CalibrationSession(W, H, min_samples=20)
        t = 0.0
        while s.phase == "settling":
            s.update(t, np.zeros(10), True)
            t += 1 / 30
        for i in range(25):  # 25 muestras, 10 de ellas atípicas -> 15 < 20 -> reintento
            f = np.zeros(10)
            f[:4] = rng.normal(0, 0.003, 4)
            if i % 5 < 2:
                f[0] = 0.5
            s.update(t, f, True)
            t += 1.5 / 25
        s.update(t + 0.1, np.zeros(10), True)
        self.assertEqual(s.retries, 1)
        self.assertGreaterEqual(s.discarded, 10)


class TestSerializationV2(unittest.TestCase):
    def fitted(self):
        x, y, groups, _ = make_dataset()
        m = GazeMapper.from_fit_options()
        m.fit(x, y, groups)
        return m, x

    def test_round_trip_with_model_clamp_and_offset(self):
        m, x = self.fitted()
        m.offset = np.array([12.0, -7.0])
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "calibration.json")
            save_calibration(path, m, GEOM)
            with open(path, encoding="utf-8") as f:
                raw = json.load(f)
            self.assertEqual(raw["version"], 2)
            self.assertEqual(raw["model"], DEFAULT_FIT["model"])
            self.assertEqual(raw["offset"], [12.0, -7.0])
            loaded = load_calibration(path, GEOM)
        np.testing.assert_allclose(loaded.predict(x[:20]), m.predict(x[:20]), atol=1e-6)
        np.testing.assert_allclose(loaded.predict(x[:3], apply_offset=False) + [12.0, -7.0], loaded.predict(x[:3]))

    def test_version_1_compact_still_loads(self):
        x, y, groups, _ = make_dataset()
        m = GazeMapper("compact", alpha=0.02)
        m.fit(x, y, groups)
        data = m.to_dict(GEOM)
        for k in ("model", "aggregate", "alpha_mode", "z_min", "z_max", "offset"):
            data.pop(k)
        data["version"] = 1
        old = GazeMapper.from_dict(data, GEOM)
        self.assertEqual(old.model, "quad")
        self.assertIsNone(old.z_min)
        np.testing.assert_allclose(old.offset, [0.0, 0.0])
        free = GazeMapper("compact", alpha=0.02, clamp=False)
        free.fit(x, y, groups)
        np.testing.assert_allclose(old.predict(x[:10]), free.predict(x[:10]), atol=1e-6)

    def test_invalid_v2_files_rejected(self):
        m, _ = self.fitted()
        data = m.to_dict(GEOM)
        bad = dict(data, offset=[float("nan"), 0.0])
        with self.assertRaises(CalibrationError):
            GazeMapper.from_dict(bad, GEOM)
        bad = dict(data, model="linear")  # pesos cuadráticos con un modelo lineal
        with self.assertRaises(CalibrationError):
            GazeMapper.from_dict(bad, GEOM)
        bad = dict(data, version=99)
        with self.assertRaises(CalibrationError):
            GazeMapper.from_dict(bad, GEOM)

    def test_save_offset_keeps_the_rest(self):
        m, x = self.fitted()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "calibration.json")
            save_calibration(path, m, GEOM, camera_index=2)
            save_offset(path, (30.0, -5.0))
            loaded = load_calibration(path, GEOM)
            with open(path, encoding="utf-8") as f:
                self.assertEqual(json.load(f)["camera_index"], 2)
            np.testing.assert_allclose(loaded.offset, [30.0, -5.0])
            np.testing.assert_allclose(loaded.predict(x[:2]), m.predict(x[:2]) + [30.0, -5.0], atol=1e-6)
            with self.assertRaises(CalibrationError):
                save_offset(os.path.join(d, "no_existe.json"), (1, 1))


class ShiftMapper:
    """Predice siempre centro + desplazamiento (aplica offset como GazeMapper)."""

    def __init__(self, w, h, shift):
        self.base = np.array([w / 2 + shift[0], h / 2 + shift[1]])
        self.offset = np.zeros(2)

    def predict(self, x, apply_offset=True):
        n = len(np.atleast_2d(x))
        out = np.tile(self.base, (n, 1))
        return out + self.offset if apply_offset else out

    def predict_one(self, f):
        p = self.predict(f)[0]
        return float(p[0]), float(p[1])


class TestRecenterSession(unittest.TestCase):
    def run_session(self, s, mapper, seconds=3.0, valid=True, closed_every=None):
        t = 0.0
        i = 0
        while not s.finished and t < seconds:
            closed = closed_every is not None and i % closed_every == 0
            s.update(t, np.zeros(10), valid and not closed, closed, mapper)
            t += 1 / 30
            i += 1
        return t

    def test_offset_is_center_minus_median_prediction(self):
        s = RecenterSession(1920, 1080)
        t = self.run_session(s, ShiftMapper(1920, 1080, (80, -40)))
        self.assertEqual(s.phase, "done")
        self.assertEqual(s.offset, (-80.0, 40.0))
        self.assertLess(t, 2.0)  # ~0.6 s + 1 s

    def test_existing_offset_is_replaced_not_accumulated(self):
        m = ShiftMapper(1920, 1080, (80, -40))
        m.offset = np.array([500.0, 500.0])
        s = RecenterSession(1920, 1080)
        self.run_session(s, m)
        self.assertEqual(s.offset, (-80.0, 40.0))

    def test_too_large_offset_rejected(self):
        s = RecenterSession(1920, 1080)
        self.run_session(s, ShiftMapper(1920, 1080, (900, 0)))
        self.assertEqual(s.phase, "aborted")
        self.assertIn("calibra", s.message)
        self.assertIsNone(s.offset)

    def test_no_valid_samples_aborts(self):
        s = RecenterSession(1920, 1080)
        self.run_session(s, ShiftMapper(1920, 1080, (0, 0)), seconds=10, valid=False)
        self.assertEqual(s.phase, "aborted")
        self.assertIn("muestras", s.message)


class QtBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.messages = []
        qInstallMessageHandler(lambda t, c, m: self.messages.append(str(m)))
        self.tmp = tempfile.TemporaryDirectory()
        self.cal_path = os.path.join(self.tmp.name, "calibration.json")
        self.screen = QGuiApplication.primaryScreen()
        self.geom = screen_geometry_from_qt(self.screen)

    def tearDown(self):
        qInstallMessageHandler(None)
        self.tmp.cleanup()
        for m in self.messages:
            for token in FORBIDDEN:
                self.assertNotIn(token, m, f"Mensaje Qt prohibido: {m}")


class TestResultDialogSource(QtBase):
    def test_validation_source_is_explained(self):
        d = cui.CalibrationResultDialog(80.4, source="validation", cv_px=61.2, points=5)
        d.show()
        self.assertEqual(d.message_label.text(), "Calibración completada. Error estimado: ~80 px")
        self.assertTrue(d.source_label.isVisible())
        self.assertIn("5 puntos de validación nuevos", d.source_label.text())
        self.assertIn("~61 px", d.source_label.text())
        self.assertFalse(d.warning_label.isVisible())
        d.close()
        d3 = cui.CalibrationResultDialog(74.0, source="validation_corrected", points=5, raw_px=135.0)
        d3.show()
        self.assertIn("corregir automáticamente el desvío común", d3.source_label.text())
        self.assertIn("Sin corregir: ~135 px", d3.source_label.text())
        d3.close()
        d2 = cui.CalibrationResultDialog(200, source="cv")
        d2.show()
        self.assertIn("validación cruzada", d2.source_label.text())
        self.assertTrue(d2.warning_label.isVisible())  # el umbral de 150 px se aplica al error mostrado
        d2.close()


class FakeTracker(QObject):
    frame_ready = Signal(object)
    failed = Signal(str)

    def __init__(self, index=0):
        super().__init__()
        self.started = False
        self.stopped = 0

    def start(self):
        self.started = True

    def stop(self):
        self.stopped += 1


class FakeFeedback:
    def __init__(self, dpr=1.0, dry_run=False):
        self.visible = False
        self.warnings = []

    def set_warning(self, text):
        self.warnings.append(text)

    def show(self):
        self.visible = True

    def hide(self):
        self.visible = False

    def deleteLater(self):
        pass

    def update(self):
        pass

    def update_snapshot(self, snap):
        pass

    def notify(self, name):
        pass


class FakeClock:
    def __init__(self):
        self.t = 50.0

    def __call__(self):
        return self.t


class TestControlSessionRecenter(QtBase):
    def make_session(self, recenter_ok=True, with_recenter_factory=True):
        x, y, groups, _ = make_dataset()
        mapper = GazeMapper.from_fit_options()
        mapper.fit(x, y, groups)
        save_calibration(self.cal_path, mapper, self.geom)
        self.tracker = FakeTracker()
        self.mouse = FakeMouse()
        self.feedback = FakeFeedback()
        self.clock = FakeClock()
        self.pause_keys, self.recenter_keys = [], []

        def pause_hk(cb):
            h = FakeHotkey(cb)
            self.pause_keys.append(h)
            return h

        def recenter_hk(cb):
            h = FakeHotkey(cb, register_ok=recenter_ok, error="El atajo Ctrl+Alt+R ya está en uso (código 1409).")
            self.recenter_keys.append(h)
            return h

        self.cfg = dict(DEFAULT_CONFIG["interaction"])
        s = cs.ControlSession(
            config_getter=lambda: self.cfg,
            screen=self.screen,
            calibration_path=self.cal_path,
            tracker_factory=lambda i: self.tracker,
            mouse_factory=lambda: self.mouse,
            hotkey_factory=pause_hk,
            feedback_factory=lambda **kw: self.feedback,
            screen_size_fn=lambda: (self.geom["width"], self.geom["height"]),
            clock=self.clock,
            recenter_hotkey_factory=recenter_hk if with_recenter_factory else None,
        )
        self.results = []
        s.recentered.connect(self.results.append)
        return s

    def gaze_frame(self, t, target):
        """Rasgos de la función conocida de test_calibration (coordenadas del mapper)."""
        feats = synth_features(np.array([target]), np.random.default_rng(int(t * 1000)), noise=0.0005)[0]
        return TrackingFrame(timestamp=t, face_found=True, features=feats)

    def test_hotkey_opens_point_and_offset_is_applied_and_saved(self):
        s = self.make_session()
        ok, msg = s.start()
        self.assertTrue(ok, msg)
        self.assertEqual(len(self.recenter_keys), 1)
        self.assertTrue(self.recenter_keys[0].registered)
        cx, cy = self.geom["width"] / 2, self.geom["height"] / 2
        looked = (cx + 120.0, cy + 60.0)  # el mapeo deriva: mirando el centro estima +120, +60
        self.recenter_keys[0].press()
        scr = s.recenter_screen
        self.assertIsNotNone(scr)
        self.assertTrue(scr.isVisible() and scr.isFullScreen())
        self.assertFalse(self.feedback.visible)
        n_moves = len(self.mouse.events)
        t = 0.0
        while s.recenter_screen is not None and t < 5:
            self.tracker.frame_ready.emit(self.gaze_frame(t, looked))
            t += 1 / 30
        self.assertIsNone(s.recenter_screen)
        self.assertEqual(len(self.mouse.events), n_moves)  # el cursor no se movió mientras se recentraba
        self.assertTrue(self.feedback.visible)
        self.assertEqual(len(self.results), 1)
        self.assertIn("guardado", self.results[0])
        offset = s.controller.mapper.offset
        self.assertAlmostEqual(offset[0], -120, delta=40)
        self.assertAlmostEqual(offset[1], -60, delta=40)
        saved = load_calibration(self.cal_path, self.geom)
        np.testing.assert_allclose(saved.offset, offset)
        # tras recentrar, mirar "looked" lleva el cursor cerca del centro
        for _ in range(60):
            snap = s.controller.update(self.gaze_frame(t, looked))
            t += 1 / 30
        self.assertLess(math.hypot(snap.x - cx, snap.y - cy), 45)
        s.stop()

    def test_escape_cancels_and_control_continues(self):
        s = self.make_session()
        self.assertTrue(s.start()[0])
        ok, _ = s.recenter()
        self.assertTrue(ok)
        scr = s.recenter_screen
        QTest.keyPress(scr, Qt.Key_Escape)
        self.clock.t += 2.0
        scr.tick()
        self.assertIsNone(s.recenter_screen)
        self.assertIn("cancelado", self.results[-1])
        np.testing.assert_allclose(s.controller.mapper.offset, [0.0, 0.0])
        self.assertTrue(s.running)
        s.stop()

    def test_recenter_requires_running_control(self):
        s = self.make_session()
        ok, msg = s.recenter()
        self.assertFalse(ok)
        self.assertIn("Inicia el control", msg)

    def test_recenter_hotkey_failure_does_not_block_control(self):
        s = self.make_session(recenter_ok=False)
        ok, msg = s.start()
        self.assertTrue(ok)
        self.assertIn("Ctrl+Alt+R", msg)
        self.assertIn("botón", msg)
        self.assertIsNone(s.recenter_hotkey)
        self.assertTrue(self.recenter_keys[0].stopped)
        s.stop()

    def test_stop_closes_recenter_and_releases_hotkeys(self):
        s = self.make_session()
        s.start()
        s.recenter()
        scr = s.recenter_screen
        s.stop()
        self.assertIsNone(s.recenter_screen)
        self.assertFalse(scr.isVisible())
        self.assertTrue(self.recenter_keys[0].stopped)
        self.assertTrue(self.pause_keys[0].stopped)

    def test_head_drift_warns_and_recenter_clears_it(self):
        s = self.make_session()
        warnings = []
        s.pose_warning.connect(warnings.append)
        self.assertTrue(s.start()[0])
        self.assertIsNotNone(s.pose_monitor)
        cx, cy = self.geom["width"] / 2, self.geom["height"] / 2

        def tilted(t, pitch_shift):
            f = self.gaze_frame(t, (cx, cy))
            f.features = f.features.copy()
            f.features[8] += pitch_shift
            return f

        t = 0.0
        for _ in range(60):
            self.tracker.frame_ready.emit(tilted(t, 0.0))
            t += 1 / 30
        self.assertEqual(warnings, [])
        for _ in range(90):  # 3 s con la cabeza inclinada 0.06 rad
            self.tracker.frame_ready.emit(tilted(t, 0.06))
            t += 1 / 30
        self.assertEqual(len(warnings), 1)
        self.assertIn("inclinación", warnings[0])
        self.assertEqual(self.feedback.warnings[-1], "Recentra: Ctrl+Alt+R")
        # Recentrar con la cabeza en la nueva pose quita el aviso y la toma como referencia
        s.recenter()
        while s.recenter_screen is not None and t < 20:
            self.tracker.frame_ready.emit(tilted(t, 0.06))
            t += 1 / 30
        self.assertEqual(warnings[-1], "")
        self.assertIsNone(self.feedback.warnings[-1])
        for _ in range(90):
            self.tracker.frame_ready.emit(tilted(t, 0.06))
            t += 1 / 30
        self.assertEqual(warnings[-1], "")
        s.stop()

    def test_injected_pause_hotkey_never_creates_real_recenter_hotkey(self):
        s = self.make_session(with_recenter_factory=False)
        self.assertTrue(s.start()[0])
        self.assertIsNone(s.recenter_hotkey)
        s.stop()


class TestGuidanceAndFeedbackWarning(QtBase):
    def test_modal_guidance_asks_to_come_closer(self):
        tr = FakeTracker()
        dlg = cui.CalibrationDialog({"camera": {"device_index": 0}}, camera_lister=lambda: [0], tracker_factory=lambda i: tr)
        dlg.show()
        self.app.processEvents()

        def frame(t, iod):
            return TrackingFrame(timestamp=t, face_found=True, features=np.zeros(10), face_center=(0.5, 0.5),
                                 interocular=iod, image=np.zeros((480, 640, 3), np.uint8))

        tr.frame_ready.emit(frame(1.0, 0.188))
        self.assertIn("Acércate", dlg.guidance_label.text())
        self.assertIn("0.19", dlg.guidance_label.text())
        self.assertTrue(dlg.guidance_label.accessibleName())
        tr.frame_ready.emit(frame(2.0, 0.234))
        self.assertEqual(dlg.guidance_label.text(), "")
        tr.frame_ready.emit(TrackingFrame(timestamp=3.0, face_found=False, image=np.zeros((480, 640, 3), np.uint8)))
        self.assertIn("No se detecta tu rostro", dlg.guidance_label.text())
        dlg.reject()

    def test_cursor_feedback_paints_warning(self):
        import cursor_feedback as cf

        w = cf.CursorFeedback(clock=lambda: 0.0)
        w.show()
        empty = w.grab().toImage()
        w.set_warning(cf.DRIFT_WARNING_TEXT)
        img = w.grab().toImage()
        self.assertEqual(w.warning, "Recentra: Ctrl+Alt+R")
        changed = sum(img.pixelColor(x, y) != empty.pixelColor(x, y) for x in range(0, 140, 4) for y in range(0, 26, 2))
        self.assertGreater(changed, 10)  # se dibujó el aviso arriba
        w.set_warning(None)
        self.assertIsNone(w.warning)
        w.close()


class TestConfigWindowRecenter(QtBase):
    def test_button_and_messages(self):
        from config_ui import ConfigWindow

        win = ConfigWindow(config_path=os.path.join(self.tmp.name, "config.json"))
        win.show()
        self.app.processEvents()
        self.assertEqual(win.btn_recenter.text(), "Recentrar")
        self.assertGreaterEqual(win.btn_recenter.height(), 48)
        self.assertIn("Ctrl+Alt+R", win.btn_recenter.accessibleName())
        win.btn_recenter.click()
        self.assertIn("Inicia el control", win.control_label.text())

        class Stub(QObject):
            paused_changed = Signal(bool)
            tracker_active_changed = Signal(bool)
            failed = Signal(str)
            recentered = Signal(str)
            pose_warning = Signal(str)

            def __init__(self):
                super().__init__()
                self.running = False
                self.paused = False
                self.calls = 0

            def start(self):
                self.running = True
                self.paused_changed.emit(False)
                return True, ""

            def recenter(self):
                self.calls += 1
                return True, ""

            def stop(self):
                self.running = False

        stub = Stub()
        win.control_session_factory = lambda: stub
        win.btn_control.click()
        self.assertIn("Ctrl+Alt+R", win.control_label.text())
        win.btn_recenter.click()
        self.assertEqual(stub.calls, 1)
        stub.recentered.emit("Recentrado: desplazamiento (+10, -5) px guardado en la calibración.")
        self.assertIn("Recentrado", win.control_label.text())
        stub.pose_warning.emit("Tu cabeza cambió de inclinación respecto a la calibración: pulsa Ctrl+Alt+R.")
        self.assertIn("Tu cabeza cambió", win.control_label.text())
        stub.pose_warning.emit("")  # al quitarse vuelve el texto normal del control
        self.assertIn("Control activo", win.control_label.text())
        win.close()


if __name__ == "__main__":
    unittest.main()
