"""Parte E: radio magnético (función pura, sin UI todavía)."""

import math
import unittest

from gaze_filters import distance_to_rect, magnetic_snap, snap_strength

BUTTON = (100.0, 100.0, 80.0, 48.0)  # centro (140, 124)


class TestDistance(unittest.TestCase):
    def test_inside_edge_and_outside(self):
        self.assertEqual(distance_to_rect((140, 124), BUTTON), 0.0)
        self.assertEqual(distance_to_rect((100, 100), BUTTON), 0.0)
        self.assertEqual(distance_to_rect((90, 124), BUTTON), 10.0)
        self.assertAlmostEqual(distance_to_rect((97, 96), BUTTON), 5.0)  # esquina: 3-4-5


class TestMagneticSnap(unittest.TestCase):
    def test_pulls_toward_center_within_radius(self):
        x, y = magnetic_snap((80.0, 124.0), [BUTTON], 30, 0.5)  # a 20 px del borde
        self.assertAlmostEqual(x, 110.0)
        self.assertAlmostEqual(y, 124.0)

    def test_inside_the_control_also_pulls(self):
        x, y = magnetic_snap((105.0, 104.0), [BUTTON], 30, 0.25)
        self.assertAlmostEqual(x, 105 + 0.25 * 35)
        self.assertAlmostEqual(y, 104 + 0.25 * 20)

    def test_outside_radius_or_disabled_is_unchanged(self):
        self.assertEqual(magnetic_snap((40.0, 124.0), [BUTTON], 30, 0.5), (40.0, 124.0))
        self.assertEqual(magnetic_snap((90.0, 124.0), [BUTTON], 0, 0.5), (90.0, 124.0))
        self.assertEqual(magnetic_snap((90.0, 124.0), [], 30, 0.5), (90.0, 124.0))
        self.assertEqual(magnetic_snap((90.0, 124.0), [BUTTON], 30, 0.0), (90.0, 124.0))

    def test_nearest_control_wins(self):
        other = (200.0, 100.0, 80.0, 48.0)  # centro (240, 124)
        x, _ = magnetic_snap((191.0, 124.0), [BUTTON, other], 30, 1.0)  # 11 px de BUTTON, 9 px de other
        self.assertEqual(x, 240.0)
        x, _ = magnetic_snap((185.0, 124.0), [BUTTON, other], 30, 1.0)
        self.assertEqual(x, 140.0)
        small = (200.0, 100.0, 40.0, 48.0)  # centro (220, 124)
        x, _ = magnetic_snap((190.0, 124.0), [BUTTON, small], 30, 1.0)  # 10 px de ambos bordes
        self.assertEqual(x, 220.0)  # empate de borde: gana el centro más cercano

    def test_repeated_calls_converge_smoothly(self):
        pos = (85.0, 130.0)
        dists = []
        for _ in range(30):
            pos = magnetic_snap(pos, [BUTTON], 30, snap_strength(1 / 30))
            dists.append(math.hypot(pos[0] - 140, pos[1] - 124))
        self.assertTrue(all(a > b for a, b in zip(dists, dists[1:])))  # sin saltos ni oscilación
        self.assertLess(dists[-1], 1.0)
        self.assertGreater(dists[0], 40.0)  # el primer paso es suave, no un salto al centro

    def test_strength_depends_on_time_not_frames(self):
        def after(fps, seconds=0.2):
            pos = (85.0, 124.0)
            for _ in range(int(round(seconds * fps))):
                pos = magnetic_snap(pos, [BUTTON], 60, snap_strength(1 / fps))
            return pos[0]

        self.assertAlmostEqual(after(15), after(60), places=6)
        self.assertEqual(snap_strength(0.0), 0.0)
        self.assertEqual(snap_strength(0.1, 0.0), 1.0)


if __name__ == "__main__":
    unittest.main()
