"""Parte B: filtros puros (One Euro, mediana, zona muerta) y claves nuevas de configuración."""

import json
import math
import os
import tempfile
import unittest

import numpy as np

from config_manager import DEFAULT_CONFIG, load_config
from gaze_filters import (
    SMOOTHING_MAX_CUTOFF_HZ,
    SMOOTHING_MIN_CUTOFF_HZ,
    MedianFilter,
    OneEuroFilter,
    apply_deadzone,
    smoothing_to_min_cutoff,
)


class TestSmoothingMapping(unittest.TestCase):
    def test_zero_means_no_filter(self):
        self.assertIsNone(smoothing_to_min_cutoff(0))
        self.assertIsNone(smoothing_to_min_cutoff(-5))

    def test_more_percent_lower_cutoff(self):
        values = [smoothing_to_min_cutoff(p) for p in range(1, 101)]
        self.assertTrue(all(a > b for a, b in zip(values, values[1:])))
        self.assertAlmostEqual(smoothing_to_min_cutoff(100), SMOOTHING_MIN_CUTOFF_HZ)
        self.assertAlmostEqual(smoothing_to_min_cutoff(150), SMOOTHING_MIN_CUTOFF_HZ)
        self.assertLess(smoothing_to_min_cutoff(1), SMOOTHING_MAX_CUTOFF_HZ)
        self.assertGreater(smoothing_to_min_cutoff(1), 4.5)
        self.assertAlmostEqual(smoothing_to_min_cutoff(50), math.sqrt(SMOOTHING_MAX_CUTOFF_HZ * SMOOTHING_MIN_CUTOFF_HZ))


class TestOneEuro(unittest.TestCase):
    def test_first_sample_passes_and_constant_is_kept(self):
        f = OneEuroFilter(1.0, 0.0, 1.0)
        np.testing.assert_allclose(f([10, 20], 0.0), [10, 20])
        for i in range(1, 30):
            np.testing.assert_allclose(f([10, 20], i / 30), [10, 20])

    def test_low_cutoff_attenuates_noise(self):
        rng = np.random.default_rng(0)
        noisy = 100 + rng.normal(0, 10, (300, 1))
        out_low = []
        out_high = []
        lo, hi = OneEuroFilter(0.3, 0.0), OneEuroFilter(5.0, 0.0)
        for i, x in enumerate(noisy):
            out_low.append(lo(x, i / 30)[0])
            out_high.append(hi(x, i / 30)[0])
        self.assertLess(np.std(out_low[30:]), np.std(out_high[30:]))
        self.assertLess(np.std(out_low[30:]), 3.0)

    def test_matches_exponential_smoothing_formula(self):
        f = OneEuroFilter(1.0, 0.0, 1.0)
        f([0.0], 0.0)
        out = f([100.0], 0.1)[0]
        tau = 1 / (2 * math.pi)
        self.assertAlmostEqual(out, 100 / (1 + tau / 0.1), places=9)

    def test_beta_reduces_lag_on_fast_motion(self):
        def lag(beta):
            f = OneEuroFilter(0.5, beta, 1.0)
            t = 0.0
            for _ in range(30):
                f([0.0], t)
                t += 1 / 30
            for k in range(1, 300):
                if f([1000.0], t)[0] >= 900:
                    return k
                t += 1 / 30
            return 999

        self.assertLess(lag(0.01), lag(0.0))

    def test_repeated_timestamp_keeps_value_and_reset(self):
        f = OneEuroFilter(1.0, 0.0)
        f([5.0], 1.0)
        np.testing.assert_allclose(f([50.0], 1.0), [5.0])
        np.testing.assert_allclose(f([50.0], 0.5), [5.0])  # hacia atrás: se ignora
        f.reset()
        self.assertIsNone(f.value)
        np.testing.assert_allclose(f([50.0], 2.0), [50.0])

    def test_depends_on_time_not_samples(self):
        # el mismo intervalo total con distinto número de muestras da un resultado parecido
        def run(fps):
            f = OneEuroFilter(1.0, 0.0)
            f([0.0], 0.0)
            t, out = 0.0, None
            while t < 0.5 - 1e-9:
                t += 1 / fps
                out = f([100.0], t)[0]
            return out

        self.assertAlmostEqual(run(15), run(60), delta=6)


class TestMedianAndDeadzone(unittest.TestCase):
    def test_median_rejects_spike(self):
        m = MedianFilter(3)
        outs = [m(np.array([v, -v]))[0] for v in (1.0, 1.0, 50.0, 1.0, 1.0)]
        self.assertEqual(outs, [1.0, 1.0, 1.0, 1.0, 1.0])

    def test_median_window_one_and_partial_buffer(self):
        m = MedianFilter(1)
        self.assertEqual(m(np.array([7.0]))[0], 7.0)
        self.assertEqual(len(m), 0)
        m5 = MedianFilter(5)
        self.assertEqual(m5(np.array([2.0]))[0], 2.0)
        self.assertEqual(m5(np.array([4.0]))[0], 3.0)
        for v in (10.0, 10.0, 10.0, 10.0):
            out = m5(np.array([v]))[0]
        self.assertEqual(out, 10.0)
        self.assertEqual(len(m5), 5)

    def test_median_resets_on_dimension_change(self):
        m = MedianFilter(3)
        m(np.zeros(9))
        out = m(np.ones(10))
        self.assertEqual(out.shape, (10,))

    def test_deadzone(self):
        self.assertEqual(apply_deadzone(None, (5, 5), 10), (5.0, 5.0))
        self.assertEqual(apply_deadzone((0, 0), (3, 4), 10), (0, 0))
        self.assertEqual(apply_deadzone((0, 0), (6, 8), 10), (6.0, 8.0))
        self.assertEqual(apply_deadzone((0, 0), (1, 1), 0), (1.0, 1.0))


class TestConfigKeys(unittest.TestCase):
    def write(self, d, interaction):
        path = os.path.join(d, "config.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"interaction": interaction}, f)
        return load_config(path)["interaction"]

    def test_defaults(self):
        inter = DEFAULT_CONFIG["interaction"]
        self.assertEqual(inter["feature_median_window"], 3)
        self.assertEqual(inter["deadzone_px"], 4)
        self.assertAlmostEqual(inter["one_euro_beta"], 0.002)
        self.assertAlmostEqual(inter["one_euro_d_cutoff"], 1.0)

    def test_clamps_and_invalid_values(self):
        with tempfile.TemporaryDirectory() as d:
            inter = self.write(d, {"one_euro_beta": 5, "one_euro_d_cutoff": 0.0, "feature_median_window": 40,
                                   "deadzone_px": -3})
            self.assertEqual((inter["one_euro_beta"], inter["one_euro_d_cutoff"]), (1.0, 0.1))
            self.assertEqual((inter["feature_median_window"], inter["deadzone_px"]), (9, 0))
            inter = self.write(d, {"one_euro_beta": "x", "one_euro_d_cutoff": None, "feature_median_window": "a",
                                   "deadzone_px": [1]})
            self.assertAlmostEqual(inter["one_euro_beta"], 0.002)
            self.assertAlmostEqual(inter["one_euro_d_cutoff"], 1.0)
            self.assertEqual((inter["feature_median_window"], inter["deadzone_px"]), (3, 4))
            inter = self.write(d, {"one_euro_beta": 0.004, "one_euro_d_cutoff": 2.5})
            self.assertEqual((inter["one_euro_beta"], inter["one_euro_d_cutoff"]), (0.004, 2.5))


if __name__ == "__main__":
    unittest.main()
