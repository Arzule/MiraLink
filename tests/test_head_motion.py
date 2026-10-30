"""Fase 1 de la compensación de la cabeza: fase de poses de cabeza en --record-session, indicador en vivo
e informe.

Solo mide (todavía no hay compensación). Sin hardware: rasgos sintéticos y relojes inyectados.
"""

import math
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
from calibration import (
    HEAD_MOTION_STEPS,
    HEAD_POSE_TARGETS,
    HEAD_SETTLE_S,
    HEAD_STEP_AXIS,
    CalibrationSession,
    HeadPoseGuide,
    head_guide_text,
    validation_points,
)
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


def pose(yaw_deg=0.0, pitch_deg=0.0, iod=0.25):
    f = np.zeros(10)
    f[6], f[7], f[8] = iod, math.radians(yaw_deg), math.radians(pitch_deg)
    return f


class TestHeadMotionStage(unittest.TestCase):
    def test_steps_are_small_still_poses(self):
        keys = [k for k, _txt, _sec in HEAD_MOTION_STEPS]
        self.assertEqual(keys, ["still", "left", "right", "up", "down", "near", "far"])
        self.assertEqual(set(HEAD_STEP_AXIS), set(keys) - {"still"})
        for _k, text, sec in HEAD_MOTION_STEPS:
            self.assertIn("quieto" if _k != "still" else "sin mover", text)
            self.assertAlmostEqual(sec, 2.5)

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
        self.assertEqual(sorted({e[2] for e in head}), list(range(len(HEAD_MOTION_STEPS))))
        expected = sum(HEAD_SETTLE_S + sec for _k, _txt, sec in HEAD_MOTION_STEPS)
        changes = 2 * len(HEAD_MOTION_STEPS)  # cada cambio de fase ocurre en el primer cuadro tras el umbral
        self.assertAlmostEqual(head[-1][0] - head[0][0], expected, delta=changes / 30 + 0.05)
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
        hold = HEAD_MOTION_STEPS[0][2]
        self.assertAlmostEqual(s.phase_progress(10.0 + HEAD_SETTLE_S + hold / 2), 0.5)
        s.skip_head()
        self.assertEqual(s.phase, "done")

    def test_without_validation_goes_to_head_after_calibration(self):
        s = CalibrationSession(W, H, head_motion=HEAD_MOTION_STEPS[:1])
        log = run(s)
        self.assertEqual(s.phase, "done")
        self.assertEqual({e[1] for e in log}, {"calibration", "head"})


class TestHeadPoseGuide(unittest.TestCase):
    def test_reference_from_still_step_and_zone(self):
        g = HeadPoseGuide(window=1)
        for _ in range(5):
            self.assertIsNone(g.update("still", "capturing", pose(2.0, -1.0)))  # el paso quieto no tiene eje
        state = g.update("left", "capturing", pose(12.0, -1.0))
        self.assertEqual(state["axis"], "yaw")
        self.assertAlmostEqual(state["value"], 10.0, places=6)
        self.assertTrue(state["in_zone"])
        self.assertIn("bien", head_guide_text(state))
        state = g.update("right", "capturing", pose(-1.0, -1.0))  # 3° desde la referencia: poco
        self.assertFalse(state["in_zone"])
        self.assertIn("un poco más", head_guide_text(state))
        state = g.update("down", "capturing", pose(2.0, 25.0))  # 26° de inclinación: demasiado
        self.assertEqual(state["axis"], "pitch")
        self.assertIn("demasiado", head_guide_text(state))

    def test_depth_uses_eye_separation(self):
        g = HeadPoseGuide(window=1)
        g.update("still", "capturing", pose(iod=0.25))
        state = g.update("near", "capturing", pose(iod=0.25 / 0.85))  # 15 % más cerca
        self.assertEqual(state["axis"], "depth")
        self.assertAlmostEqual(state["value"], 15.0, places=4)
        self.assertTrue(state["in_zone"])
        self.assertIn("%", head_guide_text(state))
        lo, hi = HEAD_POSE_TARGETS["depth"]
        self.assertEqual((state["lo"], state["hi"]), (lo, hi))

    def test_without_still_reference_uses_first_frames_and_ignores_bad_frames(self):
        g = HeadPoseGuide(window=1)
        self.assertIsNone(g.update("left", "settling", None))
        bad = pose(5.0)
        bad[6] = float("nan")
        self.assertIsNone(g.update("left", "settling", bad))
        first = g.update("left", "settling", pose(0.0))
        self.assertAlmostEqual(first["value"], 0.0)
        self.assertAlmostEqual(g.update("left", "capturing", pose(9.0))["value"], 9.0 / 2, places=6)  # ref = mediana(0, 9)
        self.assertEqual(head_guide_text(None), "")


class TestHeadMotionReport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.session = gs.synthesize_session(0, head_motion=True)

    def test_recorded_stage_and_meta(self):
        s = self.session
        self.assertTrue((s.stage == 2).any())
        self.assertEqual([k for k, _ in s.meta["head_motion_steps"]], [k for k, _t, _s in HEAD_MOTION_STEPS])
        self.assertEqual(s.meta["head_settle_s"], HEAD_SETTLE_S)

    def test_report_measures_static_poses_against_geometry(self):
        rep = gs.head_motion_report(self.session)
        steps = {r["key"]: r for r in rep["steps"]}
        self.assertEqual(set(steps), {k for k, _t, _s in HEAD_MOTION_STEPS})
        # se descartan los primeros 0.7 s de cada paso (2.5 s de captura a ~30 fps)
        for r in steps.values():
            self.assertLessEqual(r["frames"], (2.5 - gs.HEAD_TRANSITION_S) * 31 + 2)
        # las poses del usuario sintético: ~10° de giro/inclinación y ~15 % de distancia, quietas
        self.assertGreater(steps["left"]["yaw_offset_deg"] - steps["still"]["yaw_offset_deg"], 6)
        self.assertLess(steps["right"]["yaw_offset_deg"] - steps["still"]["yaw_offset_deg"], -6)
        self.assertLess(steps["up"]["pitch_offset_deg"] - steps["still"]["pitch_offset_deg"], -6)
        self.assertGreater(steps["down"]["pitch_offset_deg"] - steps["still"]["pitch_offset_deg"], 6)
        self.assertLess(steps["near"]["distance_change_pct"], steps["far"]["distance_change_pct"] - 20)
        self.assertLess(steps["left"]["yaw_deg"], 5)  # quieto dentro del paso (rango p5-p95 pequeño)
        # el usuario sintético contrarrota los ojos: el cursor del ajuste solo iris se va con el giro
        slope, r = rep["slope_x_yaw"]
        self.assertLess(r, -0.9)
        self.assertTrue(0.5 < abs(slope) / rep["geometric_px_per_rad"] < 1.2)
        self.assertGreater(steps["left"]["error_median_px"], 2 * steps["still"]["error_median_px"])
        bins = rep["by_angle"]
        self.assertGreaterEqual(len(bins), 2)
        self.assertGreater(bins[-1]["error_median_px"], bins[0]["error_median_px"])
        text = gs.format_head_motion_report(rep)
        for token in ("Movimiento de cabeza", "px/rad", "girado a la izquierda", "error por ángulo", "0-3°"):
            self.assertIn(token, text)

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

    def test_paints_instruction_guide_and_regions_do_not_cover_the_center(self):
        scr, s, clock = self.make_screen()
        t = 0.0
        while s.head_index == 0:  # paso quieto: fija la referencia
            scr.on_frame(make_frame(t, pose(0.0)))
            t += 1 / 30
        self.assertEqual(s.head_step[0], "left")
        for _ in range(5):  # el indicador usa la mediana de los últimos 5 cuadros
            scr.on_frame(make_frame(t, pose(10.0)))
            t += 1 / 30
        self.assertIsNotNone(scr.head_state)
        self.assertTrue(scr.head_state["in_zone"])
        scr.repaint()
        shots = os.path.join(tempfile.gettempdir(), "miralink_shots")
        os.makedirs(shots, exist_ok=True)
        scr.grab().save(os.path.join(shots, "screen_head_motion.png"))
        regions = scr.layout_regions()
        self.assertIn("guide", regions)
        keep_out = cui.QRectF(640 - 220, 360 - 60, 440, 120)
        for name, r in regions.items():
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
