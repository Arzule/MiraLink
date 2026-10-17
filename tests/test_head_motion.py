"""Fase 1 de la compensación de la cabeza: fase de movimiento de cabeza en --record-session e informe.

Solo mide (todavía no hay compensación). Sin hardware: rasgos sintéticos y relojes inyectados.
"""

import os
import tempfile
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtCore import Qt, qInstallMessageHandler
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

import calibration_ui as cui
import gaze_session as gs
from calibration import HEAD_MOTION_STEPS, HEAD_SETTLE_S, CalibrationSession, validation_points
from test_calibration import H, W, synth_features
from test_calibration_ui import FORBIDDEN, FakeClock, make_frame


def run(session, seed=0, fps=30.0, until=200.0):
    """Recorre la sesión con rasgos sintéticos; devuelve [(t, stage, punto, fase)]."""
    rng = np.random.default_rng(seed)
    t, log = 0.0, []
    while not session.finished and t < until:
        log.append((t, session.stage, session.active_index, session.phase))
        f = synth_features(np.array([session.current_point]), rng)[0]
        session.update(t, f, True, ear=0.3)
        t += 1 / fps
    return log


class TestHeadMotionStage(unittest.TestCase):
    def test_runs_after_validation_without_changing_the_result(self):
        plain = CalibrationSession(W, H, validation=validation_points(W, H))
        run(plain)
        s = CalibrationSession(W, H, validation=validation_points(W, H), head_motion=HEAD_MOTION_STEPS)
        log = run(s)
        self.assertEqual(s.phase, "done")
        stages = [e[1] for e in log]
        self.assertEqual(stages[-1], "head")
        self.assertEqual(stages.index("head"), len(stages) - stages[::-1].index("validation"))  # justo después
        head = [e for e in log if e[1] == "head"]
        self.assertEqual(sorted({e[2] for e in head}), [0, 1, 2, 3])
        expected = sum(HEAD_SETTLE_S + sec for _k, _txt, sec in HEAD_MOTION_STEPS)
        # cada uno de los 8 cambios de fase ocurre en el primer cuadro tras el umbral (hasta 1/30 s)
        self.assertAlmostEqual(head[-1][0] - head[0][0], expected, delta=8 / 30 + 0.05)
        np.testing.assert_array_equal(s.result().features, plain.result().features)
        np.testing.assert_array_equal(s.result().val_features, plain.result().val_features)

    def test_center_target_progress_and_skip(self):
        s = CalibrationSession(W, H, head_motion=HEAD_MOTION_STEPS)
        s._complete(10.0)  # como al terminar calibración y validación
        self.assertTrue(s.in_head)
        self.assertEqual(s.current_point, (W / 2, H / 2))
        self.assertEqual(s.head_step[0], "still")
        s.update(10.0, None, False)
        self.assertAlmostEqual(s.phase_progress(11.0), 1.0 / HEAD_SETTLE_S)
        s.update(10.0 + HEAD_SETTLE_S, None, False)  # sin rostro también avanza: no aborta
        self.assertEqual(s.phase, "capturing")
        self.assertAlmostEqual(s.phase_progress(10.0 + HEAD_SETTLE_S + 1.5), 0.5)
        s.skip_head()
        self.assertEqual(s.phase, "done")

    def test_without_validation_goes_to_head_after_calibration(self):
        s = CalibrationSession(W, H, head_motion=HEAD_MOTION_STEPS[:1])
        log = run(s)
        self.assertEqual(s.phase, "done")
        self.assertEqual({e[1] for e in log}, {"calibration", "head"})


class TestHeadMotionReport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.session = gs.synthesize_session(0, head_motion=True)

    def test_recorded_stage_and_meta(self):
        s = self.session
        self.assertTrue((s.stage == 2).any())
        self.assertEqual([k for k, _ in s.meta["head_motion_steps"]], [k for k, _t, _s in HEAD_MOTION_STEPS])
        self.assertEqual(s.meta["head_settle_s"], HEAD_SETTLE_S)

    def test_report_measures_head_motion_against_geometry(self):
        rep = gs.head_motion_report(self.session)
        steps = {r["key"]: r for r in rep["steps"]}
        self.assertEqual(set(steps), {"still", "yaw", "pitch", "depth"})
        self.assertGreater(steps["yaw"]["yaw_deg"], 15)
        self.assertGreater(steps["pitch"]["pitch_deg"], 10)
        lo, hi = steps["depth"]["distance_ratio"]
        self.assertGreater(hi - lo, 0.2)
        # el usuario sintético contrarrota los ojos: el cursor del ajuste solo iris se va con el giro
        slope, r = steps["yaw"]["slope_x_yaw"]
        self.assertLess(r, -0.9)
        self.assertTrue(0.5 < abs(slope) / rep["geometric_px_per_rad"] < 1.2)
        self.assertGreater(steps["yaw"]["error_median_px"], 2 * steps["still"]["error_median_px"])
        text = gs.format_head_motion_report(rep)
        self.assertIn("Movimiento de cabeza", text)
        self.assertIn("px/rad", text)

    def test_old_sessions_say_there_is_no_head_stage(self):
        with self.assertRaises(ValueError):
            gs.head_motion_report(gs.synthesize_session(1))

    def test_posture_report_ignores_head_stage(self):
        rep = gs.posture_report(self.session)
        plain = gs.posture_report(gs.synthesize_session(0))
        self.assertAlmostEqual(rep["pose_max_deviation"], plain["pose_max_deviation"], places=6)


class TestHeadMotionScreen(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.messages = []
        qInstallMessageHandler(lambda t, c, m: self.messages.append(str(m)))

    def tearDown(self):
        qInstallMessageHandler(None)
        for m in self.messages:
            for token in FORBIDDEN:
                self.assertNotIn(token, m, f"Mensaje Qt prohibido: {m}")

    def make_screen(self):
        clock = FakeClock()
        s = CalibrationSession(1280, 720, head_motion=HEAD_MOTION_STEPS)
        s._complete(0.0)
        scr = cui.CalibrationScreen(s, dpr=1.0, clock=clock)
        scr.resize(1280, 720)
        scr.show()
        self.app.processEvents()
        return scr, s, clock

    def test_paints_instruction_and_regions_do_not_cover_the_center(self):
        scr, s, clock = self.make_screen()
        scr.on_frame(make_frame(0.0, np.zeros(10)))
        scr.on_frame(make_frame(HEAD_SETTLE_S + 0.1, np.zeros(10)))
        self.assertEqual(s.phase, "capturing")
        shots = os.path.join(tempfile.gettempdir(), "miralink_shots")
        os.makedirs(shots, exist_ok=True)
        scr.grab().save(os.path.join(shots, "screen_head_motion.png"))
        keep_out = cui.QRectF(640 - 220, 360 - 60, 440, 120)
        for name, r in scr.layout_regions().items():
            self.assertFalse(r.intersects(keep_out), f"{name} tapa el punto central")
        scr.close()

    def test_escape_skips_head_stage_and_completes(self):
        scr, s, clock = self.make_screen()
        completed, aborted = [], []
        scr.completed.connect(completed.append)
        scr.aborted.connect(aborted.append)
        QTest.keyPress(scr, Qt.Key_Escape)
        clock.t += 1.6
        scr.tick()
        self.assertEqual(s.phase, "done")
        # no se cancela: se intenta ajustar (esta sesión de prueba no tiene puntos, así que el ajuste falla)
        self.assertEqual([m for m in aborted if "cancelada" in m], [])
        self.assertTrue(completed or aborted)
        scr.close()


if __name__ == "__main__":
    unittest.main()
