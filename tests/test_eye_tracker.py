import math
import os
import tempfile
import unittest

import numpy as np

import eye_tracker as et
from eye_tracker import (
    BlinkDetector,
    EyeTrackingPipeline,
    ModelNotFoundError,
    TrackingFrame,
    build_tracking_frame,
    eye_aspect_ratio,
    extract_features,
    list_cameras,
)

SIZE = (640, 480)


def make_landmarks(open_ratio=0.3, iris_right=(0.0, 0.0), iris_left=(0.0, 0.0), nose=(0.5, 0.55)):
    """Landmarks sintéticos (478, 3) normalizados. Ojos de 60 px de ancho en un cuadro de 640x480.

    open_ratio = EAR deseado; iris_* = desplazamiento del iris en fracción del ancho del ojo (along, perp).
    """
    lm = np.zeros((et.NUM_LANDMARKS, 3))
    w, h = SIZE
    eye_w = 60.0

    def place(ear_idx, iris_idx, cx, cy, iris_off):
        p1, p2, p3, p4, p5, p6 = ear_idx
        half = eye_w / 2
        vert = open_ratio * eye_w  # |p2-p6| = |p3-p5| = vert -> EAR = 2*vert / (2*eye_w)
        pts = {
            p1: (cx - half, cy),
            p4: (cx + half, cy),
            p2: (cx - 10, cy - vert / 2),
            p6: (cx - 10, cy + vert / 2),
            p3: (cx + 10, cy - vert / 2),
            p5: (cx + 10, cy + vert / 2),
        }
        for i, (x, y) in pts.items():
            lm[i] = (x / w, y / h, 0)
        ix = cx + iris_off[0] * eye_w
        iy = cy + iris_off[1] * eye_w
        lm[iris_idx] = (ix / w, iy / h, 0)

    place(et.RIGHT_EYE_EAR, et.RIGHT_IRIS_CENTER, 250, 200, iris_right)
    place(et.LEFT_EYE_EAR, et.LEFT_IRIS_CENTER, 390, 200, iris_left)
    lm[et.NOSE_TIP] = (nose[0], nose[1], 0)
    return lm


class TestEar(unittest.TestCase):
    def test_ear_known_value(self):
        pts = np.array([[0, 0], [10, -3], [20, -3], [30, 0], [20, 3], [10, 3]], dtype=float)
        # vertical = 6 + 6 = 12; horizontal = 30 -> 12 / 60 = 0.2
        self.assertAlmostEqual(eye_aspect_ratio(pts), 0.2)

    def test_ear_degenerate_returns_zero(self):
        self.assertEqual(eye_aspect_ratio(np.zeros((6, 2))), 0.0)

    def test_ear_from_synthetic_landmarks(self):
        for ratio in (0.05, 0.3):
            frame = build_tracking_frame(0.0, make_landmarks(open_ratio=ratio), SIZE)
            self.assertAlmostEqual(frame.ear_left, ratio, places=6)
            self.assertAlmostEqual(frame.ear_right, ratio, places=6)
            self.assertAlmostEqual(frame.ear_mean, ratio, places=6)

    def test_no_landmarks_means_no_face(self):
        frame = build_tracking_frame(1.0, None, SIZE)
        self.assertFalse(frame.face_found)
        self.assertIsNone(frame.features)


class TestFeatures(unittest.TestCase):
    def test_vector_shape_and_names(self):
        f = extract_features(make_landmarks(), SIZE)
        self.assertEqual(f.shape, (et.FEATURE_DIM,))
        self.assertEqual(len(et.FEATURE_NAMES), et.FEATURE_DIM)

    def test_centered_iris_is_zero(self):
        f = extract_features(make_landmarks(), SIZE)
        np.testing.assert_allclose(f[:4], 0.0, atol=1e-9)

    def test_iris_offsets_in_eye_frame(self):
        f = extract_features(make_landmarks(iris_right=(0.2, -0.1), iris_left=(-0.15, 0.05)), SIZE)
        self.assertAlmostEqual(f[0], 0.2, places=6)  # a la derecha en la imagen
        self.assertAlmostEqual(f[1], -0.1, places=6)  # arriba = negativo
        self.assertAlmostEqual(f[2], -0.15, places=6)
        self.assertAlmostEqual(f[3], 0.05, places=6)

    def test_head_proxies(self):
        f = extract_features(make_landmarks(nose=(0.4, 0.6)), SIZE)
        self.assertAlmostEqual(f[4], 0.4)
        self.assertAlmostEqual(f[5], 0.6)
        self.assertGreater(f[6], 0.0)  # distancia interocular
        self.assertEqual((f[7], f[8]), (0.0, 0.0))  # sin matriz: yaw y pitch en cero

    def test_yaw_pitch_from_matrix(self):
        a = math.radians(20)
        rot_y = np.eye(4)
        rot_y[:3, :3] = [[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]]
        yaw, pitch = et.yaw_pitch_from_matrix(rot_y)
        self.assertAlmostEqual(yaw, a, places=6)
        self.assertAlmostEqual(pitch, 0.0, places=6)
        self.assertEqual(et.yaw_pitch_from_matrix(None), (0.0, 0.0))

    def test_interocular_scales_with_distance(self):
        near = extract_features(make_landmarks(), SIZE)[6]
        lm = make_landmarks()
        lm[:, :2] = 0.5 + (lm[:, :2] - 0.5) * 0.5  # cara a la mitad de tamaño
        far = extract_features(lm, SIZE)[6]
        self.assertAlmostEqual(far, near * 0.5, places=6)


def run_sequence(detector, ears, fps=30.0, start=0.0):
    events = []
    t = start
    for e in ears:
        ev = detector.update(t, e)
        if ev:
            events.append(ev)
        t += 1.0 / fps
    return events, t


class TestBlinkDetector(unittest.TestCase):
    def open_eyes(self, n=45, value=0.30):
        return [value + 0.005 * ((i % 3) - 1) for i in range(n)]

    def test_blink_duration_measured(self):
        d = BlinkDetector()
        seq = self.open_eyes() + [0.05] * 9 + self.open_eyes(10)  # 9 cuadros a 30 fps = 300 ms
        events, _ = run_sequence(d, seq)
        self.assertEqual(len(events), 1)
        self.assertAlmostEqual(events[0].duration_ms, 300, delta=1)
        self.assertAlmostEqual(events[0].end_ts - events[0].start_ts, 0.3, places=3)
        self.assertFalse(d.is_closed)

    def test_two_blinks_two_events(self):
        d = BlinkDetector()
        seq = self.open_eyes() + [0.05] * 6 + self.open_eyes(20) + [0.05] * 12 + self.open_eyes(10)
        events, _ = run_sequence(d, seq)
        self.assertEqual([round(e.duration_ms) for e in events], [200, 400])

    def test_noise_single_frame_discarded(self):
        d = BlinkDetector()
        seq = self.open_eyes() + [0.05] + self.open_eyes(20)  # 33 ms < min_event_ms
        events, _ = run_sequence(d, seq)
        self.assertEqual(events, [])

    def test_small_ear_fluctuation_ignored(self):
        d = BlinkDetector()
        seq = self.open_eyes() + [0.22, 0.24, 0.21, 0.23] * 5  # 0.22 > 0.6 * 0.30
        events, _ = run_sequence(d, seq)
        self.assertEqual(events, [])
        self.assertFalse(d.is_closed)

    def test_hysteresis_requires_reopening_above_threshold(self):
        d = BlinkDetector()
        _, t = run_sequence(d, self.open_eyes() + [0.05] * 4)
        self.assertTrue(d.is_closed)
        # entre 0.6 y 0.75 de la base (0.18..0.225) no se reabre
        events2, t = run_sequence(d, [0.20] * 5, start=t)
        self.assertEqual(events2, [])
        self.assertTrue(d.is_closed)
        events3, _ = run_sequence(d, [0.29] * 3, start=t)
        self.assertEqual(len(events3), 1)

    def test_baseline_adapts_to_different_eye_openness(self):
        d = BlinkDetector()
        wide = [0.40] * 45 + [0.20] * 8 + [0.40] * 5  # 0.20 < 0.6*0.40 -> cierre
        events, _ = run_sequence(d, wide)
        self.assertEqual(len(events), 1)
        d2 = BlinkDetector()
        narrow = [0.20] * 45 + [0.20] * 8  # mismo EAR, ojos naturalmente estrechos: sin cierre
        events2, _ = run_sequence(d2, narrow)
        self.assertEqual(events2, [])

    def test_no_detection_before_warmup(self):
        d = BlinkDetector()
        events, _ = run_sequence(d, [0.05] * 10 + [0.30] * 5)
        self.assertEqual(events, [])

    def test_face_lost_resets_without_event(self):
        d = BlinkDetector()
        _, t = run_sequence(d, self.open_eyes() + [0.05] * 5)
        self.assertTrue(d.is_closed)
        self.assertIsNone(d.update(t, 0.0, face_found=False))
        self.assertFalse(d.is_closed)
        # al volver el rostro con ojos abiertos no aparece un evento fantasma
        events, _ = run_sequence(d, self.open_eyes(10), start=t + 0.1)
        self.assertEqual(events, [])


class FakeCamera:
    def __init__(self, n=3):
        self.n = n

    def read(self):
        if self.n <= 0:
            return False, None
        self.n -= 1
        return True, np.zeros((480, 640, 3), np.uint8)


class FakeTracker:
    def __init__(self, ears):
        self.ears = list(ears)

    def process(self, image, ts):
        ear = self.ears.pop(0)
        return TrackingFrame(timestamp=ts, face_found=True, ear_mean=ear, ear_left=ear, ear_right=ear)


class TestPipelineAndInfra(unittest.TestCase):
    def test_pipeline_with_fakes(self):
        clock_values = iter(i / 30 for i in range(100))
        ears = [0.3] * 45 + [0.05] * 8 + [0.3] * 3
        cam = FakeCamera(len(ears))
        pipe = EyeTrackingPipeline(cam, FakeTracker(ears), BlinkDetector(), clock=lambda: next(clock_values))
        events = []
        for _ in range(len(ears)):
            f = pipe.step()
            self.assertIsNotNone(f.image)
            if f.blink_event:
                events.append(f.blink_event)
        self.assertEqual(len(events), 1)
        self.assertIsNone(pipe.step())  # cámara sin imagen

    def test_list_cameras_with_fake_opener(self):
        class Cap:
            def __init__(self, ok):
                self.ok = ok
                self.released = False

            def isOpened(self):
                return self.ok

            def release(self):
                self.released = True

        opened = []

        def opener(i):
            c = Cap(i in (0, 2))
            opened.append(c)
            return c

        self.assertEqual(list_cameras(max_index=4, opener=opener), [0, 2])
        self.assertEqual(len(opened), 5)
        self.assertTrue(all(c.released for c in opened))

    def test_missing_model_raises_clear_error(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "no_existe.task")
            with self.assertRaises(ModelNotFoundError) as ctx:
                et.FaceTracker(path, auto_download=False)
            self.assertIn(path, str(ctx.exception))
            self.assertIn("download_model.py", str(ctx.exception))

    def test_debug_overlay_draws_without_window(self):
        frame = build_tracking_frame(0.0, make_landmarks(), SIZE)
        frame.eyes_closed = True
        img = np.zeros((480, 640, 3), np.uint8)
        out = et.draw_debug_overlay(img, frame, [210.0, 340.0], 29.7)
        self.assertEqual(out.shape, img.shape)
        self.assertGreater(int(out.sum()), 0)
        self.assertEqual(int(img.sum()), 0)  # no modifica la entrada


if __name__ == "__main__":
    unittest.main()
