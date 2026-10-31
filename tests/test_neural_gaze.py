"""neural_gaze.py y scripts/neural_gaze_eval.py: recortes de la cara, decodificación de MobileGaze y evaluación.

Sin cámara: imágenes y landmarks sintéticos, una red falsa y sesiones sintéticas. La precisión real de
las redes solo se puede medir con una sesión grabada con recortes (no verificado aquí).
"""

import os
import sys
import tempfile
import unittest

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
