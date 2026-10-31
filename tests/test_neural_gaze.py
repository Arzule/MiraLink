"""neural_gaze.py y scripts/neural_gaze_eval.py: recortes de la cara, decodificación de MobileGaze y evaluación.

Sin cámara: imágenes y landmarks sintéticos, una red falsa y sesiones sintéticas. La precisión real de
las redes solo se puede medir con una sesión grabada con recortes (no verificado aquí).
"""

import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import gaze_session as gs  # noqa: E402
import neural_gaze as ng  # noqa: E402
import neural_gaze_eval as nge  # noqa: E402
from calibration import CalibrationSession  # noqa: E402
from eye_tracker import FEATURE_DIM, TrackingFrame  # noqa: E402

W, H = 1920, 1080
IMG_W, IMG_H = 640, 480


def make_image():
    """Cuadro con un patrón suave (para comparar recortes tras el JPEG)."""
    yy, xx = np.mgrid[0:IMG_H, 0:IMG_W]
    img = np.stack([(xx * 255 // IMG_W), (yy * 255 // IMG_H), ((xx + yy) * 255 // (IMG_W + IMG_H))], axis=-1)
    return img.astype(np.uint8)


def make_landmarks(x0=0.35, y0=0.25, x1=0.65, y1=0.75, n=478):
    rng = np.random.default_rng(1)
    lm = np.column_stack([rng.uniform(x0, x1, n), rng.uniform(y0, y1, n)])
    lm[0] = (x0, y0)
    lm[1] = (x1, y1)
    return lm


class TestDecoding(unittest.TestCase):
    def test_one_hot_bin_gives_its_angle(self):
        logits = np.full((2, ng.GAZE360_BINS), -50.0)
        logits[0, 45] = 50.0  # 45 x 4 - 180 = 0 grados
        logits[1, 50] = 50.0  # 20 grados
        np.testing.assert_allclose(ng.decode_bins(logits), np.radians([0.0, 20.0]), atol=1e-6)

    def test_uniform_softmax_is_the_mean_bin(self):
        out = ng.decode_bins(np.zeros((1, ng.GAZE360_BINS)))
        self.assertAlmostEqual(float(np.degrees(out[0])), 44.5 * 4 - 180, places=6)

    def test_preprocess_layout_and_normalization(self):
        face = np.zeros((4, 4, 3), dtype=np.uint8)
        face[..., 2] = 255  # rojo en BGR
        batch = ng.preprocess([face, face])
        self.assertEqual(batch.shape, (2, 3, 4, 4))
        self.assertEqual(batch.dtype, np.float32)
        np.testing.assert_allclose(batch[0, 0], (1.0 - 0.485) / 0.229, rtol=1e-5)  # canal R primero
        np.testing.assert_allclose(batch[0, 2], (0.0 - 0.406) / 0.225, rtol=1e-5)


class TestCrops(unittest.TestCase):
    def test_landmark_box_in_pixels(self):
        box = ng.landmark_box(make_landmarks(), (IMG_W, IMG_H))
        np.testing.assert_allclose(box, (0.35 * IMG_W, 0.25 * IMG_H, 0.65 * IMG_W, 0.75 * IMG_H))
        self.assertIsNone(ng.landmark_box(np.full((478, 2), np.nan), (IMG_W, IMG_H)))

    def test_record_crop_round_trip_matches_direct_crop(self):
        img, lm = make_image(), make_landmarks()
        data, rbox = ng.encode_record_crop(img, lm)
        self.assertLess(len(data), 120_000)
        cx, cy, side = rbox
        self.assertAlmostEqual(side, ng.RECORD_CROP_EXPAND * 0.5 * IMG_H)
        crop = ng.decode_jpeg(np.frombuffer(data, dtype=np.uint8))
        self.assertEqual(crop.shape, (ng.RECORD_CROP_SIZE, ng.RECORD_CROP_SIZE, 3))
        a = ng.face_input_from_record(crop, rbox, lm, (IMG_W, IMG_H), 448).astype(float)
        b = ng.face_input_from_image(img, lm, 448).astype(float)
        self.assertLess(np.abs(a - b)[8:-8, 8:-8].mean(), 3.0)  # mismo contenido salvo JPEG y bordes

    def test_face_near_the_border_is_padded(self):
        lm = make_landmarks(0.0, 0.0, 0.3, 0.5)
        data, _ = ng.encode_record_crop(make_image(), lm)
        crop = ng.decode_jpeg(np.frombuffer(data, dtype=np.uint8))
        self.assertLess(crop[:20, :20].mean(), 10)  # fuera del cuadro: negro


def crop_fn(image, landmarks):
    return b"\xff\xd8fake", (10.0, 20.0, 30.0)


class TestRecorderCrops(unittest.TestCase):
    def test_crops_saved_and_loaded(self):
        sess = gs.synthesize_session(seed=0)
        rec = gs.SessionRecorder(sess.meta, face_crop_fn=crop_fn)
        s = CalibrationSession(W, H)
        img = np.zeros((IMG_H, IMG_W, 3), np.uint8)
        feats = np.zeros(FEATURE_DIM)
        rec.add(TrackingFrame(timestamp=0.0, face_found=True, features=feats, landmarks=make_landmarks(), image=img), s)
        rec.add(TrackingFrame(timestamp=0.03, face_found=False, image=img), s)
        rec.add(TrackingFrame(timestamp=0.06, face_found=True, features=feats, landmarks=make_landmarks(), image=img), s)
        with tempfile.TemporaryDirectory() as d:
            loaded = gs.load_session(rec.save(d))
        self.assertTrue(loaded.has_face_crops)
        self.assertEqual(bytes(loaded.face_crop_bytes(0)), b"\xff\xd8fake")
        self.assertIsNone(loaded.face_crop_bytes(1))
        self.assertEqual(bytes(loaded.face_crop_bytes(2)), b"\xff\xd8fake")
        np.testing.assert_allclose(loaded.face_box[0], (10, 20, 30))
        self.assertTrue(np.isnan(loaded.face_box[1]).all())

    def test_without_crop_fn_nothing_is_stored(self):
        rec = gs.SessionRecorder({})
        rec.add(TrackingFrame(timestamp=0.0, face_found=True, features=np.zeros(FEATURE_DIM)), CalibrationSession(W, H))
        self.assertNotIn("face_jpeg", rec.arrays())
        self.assertFalse(rec.to_session().has_face_crops)


class FakeNet:
    input_size = 448

    def __init__(self):
        self.calls = 0

    def predict(self, faces):
        self.calls += 1
        return np.array([[float(f.mean()), 0.0] for f in faces])


class TestEvaluation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sess = gs.synthesize_session(seed=0, head_motion=True)

    def test_iris_as_network_output_reproduces_the_baseline(self):
        iris = np.column_stack([self.sess.features[:, [0, 2]].mean(axis=1), self.sess.features[:, [1, 3]].mean(axis=1)])
        base = nge.metrics(self.sess, "iris")
        same = nge.metrics(nge.with_gaze(self.sess, iris), "red")
        self.assertAlmostEqual(same["validation_px"], base["validation_px"], delta=0.15 * base["validation_px"] + 5)
        for key in ("head_px", "jitter_px", "snr_h", "snr_v"):
            self.assertTrue(np.isfinite(same[key]), key)

    def test_missing_network_output_invalidates_frames(self):
        gaze = np.zeros((len(self.sess), 2))
        gaze[::2] = np.nan
        sub = nge.with_gaze(self.sess, gaze)
        self.assertFalse(sub.valid_mask()[::2].any())
        self.assertEqual(self.sess.features.shape, sub.features.shape)
        self.assertFalse(np.isnan(self.sess.features[0]).all())  # la original no cambia

    def test_predict_session_uses_crops_and_skips_frames_without_them(self):
        img, lm = make_image(), make_landmarks()
        meta = dict(self.sess.meta, camera={"image_size": [IMG_W, IMG_H]})
        rec = gs.SessionRecorder(meta, face_crop_fn=ng.encode_record_crop)
        s = CalibrationSession(W, H)
        for k in range(5):
            found = k != 2
            rec.add(TrackingFrame(timestamp=k / 30, face_found=found, features=np.zeros(FEATURE_DIM),
                                  landmarks=lm if found else None, image=img), s)
        sess = rec.to_session()
        net = FakeNet()
        pred = nge.predict_session(sess, net, log=lambda *_: None)
        self.assertEqual(pred.shape, (5, 2))
        self.assertTrue(np.isnan(pred[2]).all())
        self.assertTrue(np.isfinite(pred[[0, 1, 3, 4]]).all())
        self.assertEqual(net.calls, 1)  # un solo lote
        self.assertIn("sin recortes", nge.usable(self.sess))

    def test_cached_prediction_is_reused(self):
        net = FakeNet()
        sess = self.sess
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "session_x.npz")
            first = nge.cached_prediction(path, sess, "m", lambda: net)
            self.assertTrue(os.path.exists(os.path.join(d, "session_x.m.gaze.npy")))
            again = nge.cached_prediction(path, sess, "m", lambda: self.fail("no debe recalcular"))
        np.testing.assert_array_equal(np.isnan(first), np.isnan(again))


class SeqNet:
    """Red falsa que devuelve una secuencia fija de (giro, inclinación)."""

    input_size = 32

    def __init__(self, outputs):
        self.outputs = list(outputs)

    def predict(self, faces):
        return np.array([self.outputs.pop(0) for _ in faces], dtype=float)


class TestHybridIntegration(unittest.TestCase):
    def face_frame(self, found=True):
        return TrackingFrame(timestamp=0.0, face_found=found, features=np.arange(FEATURE_DIM, dtype=float) if found else None,
                             landmarks=make_landmarks() if found else None)

    def test_net_stage_appends_running_median(self):
        from eye_tracker import NetGazeStage

        stage = NetGazeStage(SeqNet([(0.1, 0.0), (0.3, 0.2), (0.2, 9.0), (5.0, 5.0)]), window=3)
        img = make_image()
        outs = []
        for _ in range(3):
            f = self.face_frame()
            stage.apply(f, img)
            outs.append(f.features)
        self.assertEqual(outs[0].shape, (FEATURE_DIM + 2,))
        np.testing.assert_allclose(outs[0][:FEATURE_DIM], np.arange(FEATURE_DIM))
        np.testing.assert_allclose(outs[0][FEATURE_DIM:], (0.1, 0.0))
        np.testing.assert_allclose(outs[2][FEATURE_DIM:], (0.2, 0.2))  # mediana de 3: el pico 9.0 no pasa
        lost = self.face_frame(found=False)
        stage.apply(lost, img)  # sin cara: se vacía la mediana
        self.assertIsNone(lost.features)
        f = self.face_frame()
        stage.apply(f, img)
        np.testing.assert_allclose(f.features[FEATURE_DIM:], (5.0, 5.0))

    def test_net_stage_pads_old_feature_vectors_and_drops_bad_output(self):
        from eye_tracker import NetGazeStage

        stage = NetGazeStage(SeqNet([(0.1, 0.2), (np.nan, 0.0)]))
        f = TrackingFrame(timestamp=0.0, face_found=True, features=np.ones(9), landmarks=make_landmarks())
        stage.apply(f, make_image())
        self.assertEqual(f.features.shape, (FEATURE_DIM + 2,))
        self.assertEqual(f.features[9], 0.0)
        bad = self.face_frame()
        stage.apply(bad, make_image())
        self.assertFalse(bad.face_found)
        self.assertIsNone(bad.features)

    def test_pipeline_runs_the_network(self):
        from eye_tracker import EyeTrackingPipeline

        class Cam:
            def read(self):
                return True, make_image()

        class Tracker:
            def process(inner, image, ts):
                return TestHybridIntegration.face_frame(self, True)

        pipe = EyeTrackingPipeline(Cam(), Tracker(), clock=lambda: 1.0, gaze_net=SeqNet([(0.3, -0.1)]))
        frame = pipe.step()
        np.testing.assert_allclose(frame.features[FEATURE_DIM:], (0.3, -0.1))

    def test_default_net_factory_errors(self):
        from eye_tracker import _default_net_factory

        with self.assertRaises(ValueError):
            _default_net_factory("no_existe")
        missing = ng.GazeModelInfo("x", "no_existe_gaze.onnx", "")
        with mock.patch.dict(ng.GAZE_MODELS, {"x": missing}):
            with self.assertRaises(RuntimeError) as ctx:
                _default_net_factory("x")
        self.assertIn("download_gaze_models.py", str(ctx.exception))

    def test_iris_net_features_and_mapper_round_trip(self):
        import calibration as cal

        x = np.random.default_rng(0).normal(0, 0.05, (90, FEATURE_DIM + 2))
        self.assertEqual(cal.reduce_features(x, "iris_net").shape, (90, 4))
        np.testing.assert_allclose(cal.reduce_features(x, "iris_net")[:, 2:], x[:, 10:12])
        with self.assertRaises(ValueError):
            cal.reduce_features(x[:, :FEATURE_DIM], "iris_net")
        with self.assertRaises(ValueError):
            cal.GazeMapper("iris_net")  # sin red
        with self.assertRaises(ValueError):
            cal.GazeMapper("eyes", gaze_model="mobileone_s0")
        self.assertIsNone(cal.gaze_fit_options(None))
        y = np.column_stack([x[:, 10] * 4000 + 960, x[:, 11] * 4000 + 540])
        g = np.repeat(np.arange(9), 10)
        m = cal.GazeMapper.from_fit_options(cal.gaze_fit_options("mobileone_s0"))
        m.fit(x, y, g)
        geom = {"x": 0, "y": 0, "width": 1920, "height": 1080}
        back = cal.GazeMapper.from_dict(m.to_dict(geom), geom)
        self.assertEqual((back.feature_set, back.gaze_model), ("iris_net", "mobileone_s0"))
        np.testing.assert_allclose(back.predict(x[:5]), m.predict(x[:5]))
        old = m.to_dict(geom)
        old.pop("gaze_model")
        with self.assertRaises(cal.CalibrationError):
            cal.GazeMapper.from_dict(old, geom)  # iris_net sin red: recalibrar
        eyes = cal.GazeMapper.from_fit_options()
        eyes.fit(x, y, g)
        legacy = eyes.to_dict(geom)
        legacy.pop("gaze_model")
        self.assertIsNone(cal.GazeMapper.from_dict(legacy, geom).gaze_model)  # calibraciones anteriores

    def test_config_gaze_model(self):
        import config_manager as cm

        self.assertIsNone(cm.gaze_model(cm.DEFAULT_CONFIG))
        self.assertEqual(cm.gaze_model({"tracking": {"gaze_model": "mobileone_s0"}}), "mobileone_s0")
        self.assertIsNone(cm.gaze_model({"tracking": {"gaze_model": "otra"}}))
        bad = cm._validate_config(dict(cm.DEFAULT_CONFIG, tracking={"gaze_model": "otra"}))
        self.assertEqual(bad["tracking"]["gaze_model"], "iris")
        self.assertEqual(set(cm.GAZE_MODEL_CHOICES) - {"iris"}, set(ng.GAZE_MODELS))

    def test_eval_running_median_and_hybrid_columns(self):
        gaze = np.array([[1.0, 0], [3, 0], [2, 0], [np.nan, np.nan], [7, 0]])
        valid = np.array([True, True, True, True, True])
        med = nge.running_median(gaze, valid, window=3)
        np.testing.assert_allclose(med[:3, 0], (1, 2, 2))
        self.assertTrue(np.isnan(med[3]).all())
        self.assertEqual(med[4, 0], 7)  # se vació tras el cuadro sin estimación
        sess = gs.synthesize_session(seed=0)
        hyb = nge.with_net_columns(sess, np.zeros((len(sess), 2)))
        self.assertEqual(hyb.features.shape[1], FEATURE_DIM + 2)
        self.assertEqual(sess.features.shape[1], FEATURE_DIM)


@unittest.skipUnless(os.path.exists(ng.GAZE_MODELS["mobileone_s0"].path), "falta models/mobileone_s0_gaze.onnx")
class TestRealModel(unittest.TestCase):
    def test_mobileone_runs_on_cpu(self):
        net = ng.GazeNet(ng.GAZE_MODELS["mobileone_s0"].path)
        self.assertEqual(net.input_size, 448)
        out = net.predict([ng.face_input_from_image(make_image(), make_landmarks(), 448)] * 2)
        self.assertEqual(out.shape, (2, 2))
        self.assertTrue((np.abs(out) <= np.pi).all())


if __name__ == "__main__":
    unittest.main()


class SlowNet:
    input_size = 32
    name = "lenta"

    def __init__(self, value=(0.2, -0.1), fail=False):
        self.value = value
        self.fail = fail
        self.calls = 0

    def predict(self, faces):
        import time

        time.sleep(0.01)
        self.calls += 1
        if self.fail:
            raise ValueError("fallo simulado")
        return np.array([self.value] * len(faces))


class TestAsyncGazeNet(unittest.TestCase):
    def frame(self):
        return TrackingFrame(timestamp=0.0, face_found=True, features=np.zeros(FEATURE_DIM), landmarks=make_landmarks())

    def test_network_runs_in_its_own_thread(self):
        from eye_tracker import AsyncGazeNet, NetGazeStage

        net = AsyncGazeNet(SlowNet())
        try:
            stage = NetGazeStage(net)
            img = make_image()
            first = self.frame()
            stage.apply(first, img)  # entrega la cara y vuelve sin esperar a la red
            self.assertIsNone(first.features)  # aún sin estimación
            self.assertTrue(net.wait_idle())
            second = self.frame()
            stage.apply(second, img)
            np.testing.assert_allclose(second.features[FEATURE_DIM:], (0.2, -0.1))
            self.assertEqual(net.name, "lenta")
        finally:
            net.close()

    def test_results_from_before_losing_the_face_are_dropped(self):
        from eye_tracker import AsyncGazeNet, NetGazeStage

        net = AsyncGazeNet(SlowNet())
        try:
            stage = NetGazeStage(net)
            img = make_image()
            stage.apply(self.frame(), img)
            stage.apply(TrackingFrame(timestamp=0.0, face_found=False), img)  # se pierde la cara
            self.assertTrue(net.wait_idle())
            f = self.frame()
            stage.apply(f, img)
            self.assertIsNone(f.features)  # el resultado viejo no cuenta
        finally:
            net.close()

    def test_errors_of_the_network_thread_reach_the_tracker(self):
        from eye_tracker import AsyncGazeNet, NetGazeStage

        net = AsyncGazeNet(SlowNet(fail=True))
        try:
            stage = NetGazeStage(net)
            stage.apply(self.frame(), make_image())
            net.wait_idle()
            with self.assertRaises(RuntimeError):
                stage.apply(self.frame(), make_image())
        finally:
            net.close()
