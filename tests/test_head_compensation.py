"""scripts/head_compensation_eval.py: compensación geométrica de la cabeza evaluada dejando una sesión fuera.

Sin hardware: sesiones sintéticas con la fase de poses de cabeza (el usuario simulado sigue la geometría,
así que ahí la compensación ajustada SÍ debe reducir el error; con las sesiones reales del 30/10 no lo hizo).
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import gaze_session as gs  # noqa: E402
import head_compensation_eval as hce  # noqa: E402


class TestHeadCompensationEval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prepared = [hce.Prepared(gs.synthesize_session(seed, head_motion=True), f"sint{seed}") for seed in (0, 1)]
        cls.results = hce.evaluate(cls.prepared)

    def test_variants_and_rows(self):
        self.assertEqual(list(self.results), [v[0] for v in hce.VARIANTS])
        for rows in self.results.values():
            self.assertEqual([r["session"] for r in rows], ["sint0", "sint1"])
            for r in rows:
                for key in ("head_px", "validation_px", "jitter_px"):
                    self.assertTrue(np.isfinite(r[key]), key)
        base = self.results["sin compensar"]
        self.assertEqual([r["gains"] for r in base], [(0.0, 0.0, 0.0, 0.0)] * 2)

    def test_fitted_rotation_helps_a_user_that_follows_geometry(self):
        base = {r["session"]: r["head_px"] for r in self.results["sin compensar"]}
        for r in self.results["giro ajustado"]:
            self.assertLess(r["head_px"], base[r["session"]])
            # el usuario simulado usa el giro con el signo opuesto al de las sesiones reales (-2065 px/rad en el
            # informe sintético frente a +955..+1855 reales): se comprueba la magnitud, no el signo
            self.assertGreater(abs(r["gains"][0]), 0.3)
        self.assertEqual(set(hce.verdict(self.results)), {v[0] for v in hce.VARIANTS[1:]})

    def test_single_session_cannot_fit_gains(self):
        results = hce.evaluate(self.prepared[:1])
        self.assertEqual(results["giro ajustado"], [])  # dejando una fuera no queda con qué entrenar
        self.assertEqual(len(results["sin compensar"]), 1)


if __name__ == "__main__":
    unittest.main()
