import os
import tempfile
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtCore import QObject, QPointF, QRectF, Qt, Signal, qInstallMessageHandler
from PySide6.QtGui import QGuiApplication
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog, QPushButton

import calibration as cal
import calibration_ui as cui
from calibration import HEAD_MOTION_STEPS, CalibrationSession, load_calibration, screen_geometry_from_qt
from config_manager import DEFAULT_CONFIG
from eye_tracker import TrackingFrame
from test_calibration import synth_features

FORBIDDEN = ("QLayout", "Could not parse", "QWidget::", "QStyleSheetStyle", "QFont::", "QPainter::")


class FakeTracker(QObject):
    """Sustituto del hilo de seguimiento: nunca abre la cámara."""

    frame_ready = Signal(object)
    failed = Signal(str)

    def __init__(self, index):
        super().__init__()
        self.index = index
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True


def make_frame(ts, features=None, face=True, center=(0.5, 0.5), iod=0.24, closed=False):
    return TrackingFrame(
        timestamp=ts,
        face_found=face,
        features=features if face else None,
        face_center=center if face else None,
        interocular=iod if face else 0.0,
        eyes_closed=closed,
        image=np.full((480, 640, 3), 90, np.uint8),
    )


class FakeClock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


class UITestBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.messages = []
        qInstallMessageHandler(lambda t, c, m: self.messages.append(str(m)))
        self.shots = os.path.join(tempfile.gettempdir(), "miralink_shots")
        os.makedirs(self.shots, exist_ok=True)

    def tearDown(self):
        qInstallMessageHandler(None)
        for m in self.messages:
            for token in FORBIDDEN:
                self.assertNotIn(token, m, f"Mensaje Qt prohibido: {m}")

    def pump(self, n=3):
        for _ in range(n):
            self.app.processEvents()


class TestCalibrationDialog(UITestBase):
    def make_dialog(self, config=None, calibration_path=None, **kw):
        self.trackers = []

        def factory(i):
            t = FakeTracker(i)
            self.trackers.append(t)
            return t

        cfg = config or {"camera": {"device_index": 0}}
        return cui.CalibrationDialog(
            cfg,
            camera_lister=kw.pop("camera_lister", lambda: [0, 1]),
            tracker_factory=factory,
            calibration_path=calibration_path,
            **kw,
        )

    def test_texts_layout_and_accessibility(self):
        from theme_manager import list_themes, load_theme

        for theme in list_themes():
            for scale in (0.8, 1.0, 2.0):
                dlg = self.make_dialog()
                dlg.setStyleSheet(load_theme(theme["id"], scale))
                dlg.show()
                self.pump()
                tag = f"{theme['id']}@{scale}"
                self.assertEqual(dlg.windowTitle(), "Calibrar Rastreador Ocular")
                self.assertEqual(dlg.title_label.text(), "Calibrar Rastreador Ocular")
                self.assertEqual(dlg.btn_start.text(), "Iniciar Calibración")
                self.assertEqual(dlg.btn_cancel.text(), "Cancelar")
                self.assertIn("Posiciona tu rostro dentro de la guía", dlg.instructions.text())
                self.assertEqual([dlg.camera_combo.itemText(i) for i in range(2)], ["Cámara 0", "Cámara 1"])
                widgets = (dlg.title_label, dlg.camera_combo, dlg.preview, dlg.instructions, dlg.status_label,
                           dlg.btn_start, dlg.btn_cancel)
                for w in widgets:
                    self.assertTrue(w.isVisible(), tag)
                    self.assertGreater(w.width(), 0)
                    self.assertGreater(w.height(), 0)
                for w in (dlg.btn_start, dlg.btn_cancel, dlg.camera_combo):
                    self.assertGreaterEqual(w.height(), 48, f"{w.accessibleName()} {tag}")
                    self.assertTrue(w.accessibleName())
                pv = dlg.preview
                self.assertAlmostEqual(pv.width() / pv.height(), 16 / 9, delta=0.02)
                rects = [QRectFof(dlg, w) for w in widgets]
                for i in range(len(rects)):
                    for j in range(i + 1, len(rects)):
                        self.assertFalse(rects[i].intersects(rects[j]), f"controles {i} y {j} se superponen ({tag})")
                    self.assertTrue(dlg.rect().contains(rects[i]), f"control {i} fuera del diálogo ({tag})")
                if scale in (1.0, 2.0):
                    dlg.grab().save(os.path.join(self.shots, f"modal_{theme['id']}_{scale}.png"))
                dlg.close()

    def test_respects_device_index_and_changes_camera(self):
        dlg = self.make_dialog(config={"camera": {"device_index": 1}})
        selected = []
        dlg.camera_selected.connect(selected.append)
        dlg.show()
        self.pump()
        self.assertEqual(dlg.current_camera_index(), 1)
        self.assertEqual(len(self.trackers), 1)
        self.assertEqual(self.trackers[0].index, 1)
        self.assertTrue(self.trackers[0].started)
        dlg.camera_combo.setCurrentIndex(0)
        self.pump()
        self.assertEqual(selected, [0])
        self.assertTrue(self.trackers[0].stopped)
        self.assertEqual(self.trackers[1].index, 0)
        dlg.reject()
        self.assertTrue(self.trackers[1].stopped)

    def test_configured_camera_missing_from_probe_is_kept(self):
        dlg = self.make_dialog(config={"camera": {"device_index": 3}}, camera_lister=lambda: [0])
        dlg.show()
        self.pump()
        self.assertEqual(dlg.current_camera_index(), 3)
        self.assertEqual(dlg.camera_combo.itemText(dlg.camera_combo.currentIndex()), "Cámara 3")
        dlg.close()

    def test_badge_and_hints_follow_frames(self):
        dlg = self.make_dialog()
        dlg.show()
        self.pump()
        tr = self.trackers[0]
        tr.frame_ready.emit(make_frame(0.0, face=False))
        self.assertEqual(dlg.preview.badge_text, "Rostro No Detectado")
        tr.frame_ready.emit(make_frame(0.1, np.zeros(9)))
        self.assertEqual(dlg.preview.badge_text, "Rostro Detectado")
        self.assertEqual(dlg.preview.hint_text, "")
        tr.frame_ready.emit(make_frame(0.2, np.zeros(9), iod=0.05))
        self.assertEqual(dlg.preview.hint_text, "Acércate")
        tr.frame_ready.emit(make_frame(0.3, np.zeros(9), iod=0.5))
        self.assertEqual(dlg.preview.hint_text, "Aléjate")
        tr.frame_ready.emit(make_frame(0.4, np.zeros(9), center=(0.9, 0.5)))
        self.assertEqual(dlg.preview.hint_text, "Centra tu rostro")
        self.pump()
        dlg.preview.grab().save(os.path.join(self.shots, "preview_hint.png"))
        dlg.close()

    def test_deferred_tracker_start_after_close_does_not_reopen_camera(self):
        # showEvent programa el arranque con singleShot(0); cerrar antes de que se dispare no debe abrir la cámara
        import sys

        from PySide6.QtCore import QEvent

        errors = []
        old_hook = sys.excepthook
        sys.excepthook = lambda *exc: errors.append(exc[1])
        try:
            dlg = self.make_dialog()
            dlg.show()
            dlg.reject()
            self.pump()
            self.assertEqual(self.trackers, [])
            dlg2 = self.make_dialog()
            dlg2.show()
            dlg2.deleteLater()  # destruido antes del arranque diferido
            QApplication.sendPostedEvents(None, QEvent.DeferredDelete)
            del dlg2
            self.pump()
            self.assertEqual(self.trackers, [])
        finally:
            sys.excepthook = old_hook
        self.assertEqual(errors, [])

    def test_tracker_failure_shows_message(self):
        dlg = self.make_dialog()
        dlg.show()
        self.pump()
        self.trackers[0].failed.emit("No se pudo abrir la cámara 0")
        self.assertIn("No se pudo usar la cámara", dlg.status_label.text())
        dlg.close()

    def test_full_flow_accept_saves_calibration(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "calibration.json")
            clock = FakeClock()
            dlg = self.make_dialog(calibration_path=path, clock=clock, result_dialog_exec=lambda dlg_: True)
            dlg.show()
            self.pump()
            dlg.start_calibration()
            self.assertIsNotNone(dlg.calib_screen)
            scr = dlg.calib_screen
            self.assertFalse(dlg.isVisible())
            rng = np.random.default_rng(3)
            t = 0.0
            tr = self.trackers[0]
            guard = 0
            while dlg.calib_screen is not None and guard < 3000:
                press_enter_if_waiting(scr)
                px, py = scr.session.current_point
                geom = screen_geometry_from_qt(QGuiApplication.primaryScreen())
                # rasgos de una función conocida a partir del objetivo en el espacio de ESTA pantalla
                tgt = np.array([[px / geom["width"] * 1920, py / geom["height"] * 1080]])
                clock.t = 100.0 + t
                tr.frame_ready.emit(make_frame(t, synth_features(tgt, rng)[0]))
                t += 1 / 30
                guard += 1
            self.assertIsNone(dlg.calib_screen)
            self.assertIsNotNone(dlg.accepted_mapper)
            self.assertTrue(os.path.isfile(path))
            geom = screen_geometry_from_qt(QGuiApplication.primaryScreen())
            mapper = load_calibration(path, geom)
            self.assertLess(mapper.metrics["rmse_px"], 60)
            self.assertTrue(tr.stopped)

    def test_repeat_restarts_and_abort_returns_to_modal(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "calibration.json")
            clock = FakeClock()
            answers = iter([False, True])
            dlg = self.make_dialog(calibration_path=path, clock=clock, result_dialog_exec=lambda _d: next(answers))
            dlg.show()
            self.pump()
            tr = self.trackers[0]
            rng = np.random.default_rng(4)
            screens = []
            dlg.start_calibration()
            screens.append(dlg.calib_screen)
            t = 0.0
            guard = 0
            while dlg.accepted_mapper is None and guard < 6000:
                scr = dlg.calib_screen
                if scr is not None:
                    if not screens or scr is not screens[-1]:
                        screens.append(scr)
                    press_enter_if_waiting(scr)
                    geom = screen_geometry_from_qt(QGuiApplication.primaryScreen())
                    px, py = scr.session.current_point
                    tgt = np.array([[px / geom["width"] * 1920, py / geom["height"] * 1080]])
                    clock.t = 100.0 + t
                    tr.frame_ready.emit(make_frame(t, synth_features(tgt, rng)[0]))
                    t += 1 / 30
                guard += 1
            self.assertEqual(len(screens), 2)  # "Repetir" abrió una segunda pantalla
            self.assertIsNotNone(dlg.accepted_mapper)

            # aborto: sin rostro nunca se capturan muestras
            dlg2 = self.make_dialog(calibration_path=path, clock=clock)
            dlg2.show()
            self.pump()
            dlg2.start_calibration()
            scr = dlg2.calib_screen
            press_enter_if_waiting(scr)
            for i in range(400):
                clock.t += 0.1
                scr.tick(clock.t)
                if dlg2.calib_screen is None:
                    break
            self.assertIsNone(dlg2.calib_screen)
            self.assertTrue(dlg2.isVisible())
            self.assertIn("No se pudo capturar el punto", dlg2.status_label.text())
            dlg2.close()


def press_enter_if_waiting(scr):
    """Cada pantalla de la calibración espera Enter (tecla real sobre la pantalla completa)."""
    if scr is not None and scr.session.awaiting_start:
        QTest.keyClick(scr, Qt.Key_Return)


def QRectFof(container, widget):
    from PySide6.QtCore import QRect

    return QRect(widget.mapTo(container, widget.rect().topLeft()), widget.rect().size())


class TestCalibrationScreen(UITestBase):
    def make_screen(self, w=1280, h=720, **session_kw):
        clock = FakeClock()
        session = CalibrationSession(w, h, **session_kw)
        scr = cui.CalibrationScreen(session, dpr=1.0, clock=clock)
        scr.resize(w, h)
        scr.show()
        self.pump()
        return scr, clock

    def test_regions_do_not_overlap_points_or_each_other(self):
        # disposición de la aplicación: 13 puntos con margen del 2 %
        for w, h in ((1280, 720), (1920, 1080), (1366, 768), (2560, 1440)):
            scr, _ = self.make_screen(w, h, margin=cal.CALIBRATION_MARGIN, layout=cal.CALIBRATION_LAYOUT)
            regions = scr.layout_regions()
            panel = regions.pop("panel")  # solo se ve mientras espera Enter (sin puntos ni píldora)
            for name in ("notice", "mini"):
                self.assertFalse(panel.intersects(regions[name]), f"panel y {name} ({w}x{h})")
            names = list(regions)
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    self.assertFalse(
                        regions[names[i]].intersects(regions[names[j]]), f"{names[i]} y {names[j]} ({w}x{h})"
                    )
            self.assertEqual(scr.session.total, 13)
            for k in range(13):
                c = scr.logical_point(k)
                sc = max(0.6, h / 1080.0)
                # anillos (radio 60*sc) y etiqueta "Mira aquí" (hasta 220*sc a un lado)
                keep_out = QRectF(c.x() - 220 * sc, c.y() - 60 * sc, 440 * sc, 120 * sc)
                for name, r in regions.items():
                    self.assertFalse(r.intersects(keep_out), f"{name} tapa el punto {k + 1} ({w}x{h})")
            self.assertTrue(scr.rect().contains(scr.mini.geometry()))
            scr.close()

    def test_points_in_logical_coordinates_with_dpr(self):
        session = CalibrationSession(2560, 1440)
        scr = cui.CalibrationScreen(session, dpr=2.0)
        p = scr.logical_point(0)
        self.assertEqual((p.x(), p.y()), (128.0, 72.0))  # 10 % de 2560x1440 físicos / 2
        p8 = scr.logical_point(8)
        self.assertEqual((p8.x(), p8.y()), (1152.0, 648.0))

    def test_escape_hold_aborts_after_1_5_s_and_early_release_resets(self):
        scr, clock = self.make_screen()
        aborted = []
        scr.aborted.connect(aborted.append)
        QTest.keyPress(scr, Qt.Key_Escape)
        clock.t += 1.0
        scr.tick()
        self.assertAlmostEqual(scr.hold.progress(clock.t), 1.0 / 1.5, places=3)
        QTest.keyRelease(scr, Qt.Key_Escape)
        self.assertEqual(scr.hold.progress(clock.t), 0.0)
        self.assertEqual(aborted, [])
        QTest.keyPress(scr, Qt.Key_Escape)
        clock.t += 1.6
        scr.tick()
        self.assertEqual(len(aborted), 1)
        self.assertIn("cancelada", aborted[0])

    def test_tracker_failure_aborts(self):
        scr, _ = self.make_screen()
        aborted = []
        scr.aborted.connect(aborted.append)
        scr.on_tracker_failed("cámara desconectada")
        self.assertEqual(len(aborted), 1)
        self.assertIn("cámara desconectada", aborted[0])

    def test_painted_frames_saved_for_review(self):
        scr, clock = self.make_screen(1280, 720)
        rng = np.random.default_rng(5)
        # primer punto asentándose
        scr.on_frame(make_frame(0.0, synth_features(np.array([scr.session.points[0]]), rng)[0]))
        clock.t = 100.2
        scr.grab().save(os.path.join(self.shots, "screen_point1_settling.png"))
        # punto 5 en captura con ESC a medias
        scr.session.point_index = 4
        scr.session.phase = "capturing"
        scr.session._phase_start = 0.0
        clock.t = 100.0
        scr.session.update(0.75, np.zeros(9), True)
        QTest.keyPress(scr, Qt.Key_Escape)
        clock.t = 100.6
        scr.on_frame(make_frame(0.8, np.zeros(9)))
        scr.grab().save(os.path.join(self.shots, "screen_point5_capturing_esc.png"))
        self.assertGreater(scr.hold.progress(clock.t), 0.3)
        scr.close()


class TestResultDialog(UITestBase):
    def test_message_and_warning(self):
        ok = cui.CalibrationResultDialog(80.4)
        ok.show()
        self.pump()
        self.assertEqual(ok.message_label.text(), "Calibración completada. Error estimado: ~80 px")
        self.assertFalse(ok.warning_label.isVisible())
        for b in (ok.btn_accept, ok.btn_repeat):
            self.assertGreaterEqual(b.height(), 48)
            self.assertTrue(b.accessibleName())
        self.assertEqual((ok.btn_accept.text(), ok.btn_repeat.text()), ("Aceptar", "Repetir"))
        ok.grab().save(os.path.join(self.shots, "result_ok.png"))
        ok.close()

        bad = cui.CalibrationResultDialog(240.0)
        bad.show()
        self.pump()
        self.assertTrue(bad.warning_label.isVisible())
        self.assertIn("repetir", bad.warning_label.text())
        bad.grab().save(os.path.join(self.shots, "result_warn.png"))
        bad.close()

    def test_accept_and_repeat_results(self):
        d = cui.CalibrationResultDialog(10)
        d.show()
        d.btn_accept.click()
        self.assertEqual(d.result(), QDialog.Accepted)
        d2 = cui.CalibrationResultDialog(10)
        d2.show()
        d2.btn_repeat.click()
        self.assertEqual(d2.result(), QDialog.Rejected)


class TestConfigWindowIntegration(UITestBase):
    def test_calibrate_button_opens_injected_dialog(self):
        from config_ui import ConfigWindow

        with tempfile.TemporaryDirectory() as d:
            win = ConfigWindow(config_path=os.path.join(d, "config.json"))
            opened = []

            class StubDialog(QObject):
                camera_selected = Signal(int)
                finished = Signal(int)
                accepted_mapper = None

                def open(self):
                    opened.append(True)
                    self.camera_selected.emit(2)
                    self.finished.emit(0)

            win.calibration_dialog_factory = StubDialog
            win.show()
            self.pump()
            win.btn_calibrate.click()
            self.assertEqual(opened, [True])
            self.assertEqual(win._config["camera"]["device_index"], 2)
            win.close()


class TestRecordSession(UITestBase):
    """--record-session: 13 puntos + 5 de validación, cada cuadro grabado en sessions/."""

    make_dialog = TestCalibrationDialog.make_dialog

    def run_until_closed(self, dlg, clock, rng, max_frames=4000):
        tr = self.trackers[0]
        geom = screen_geometry_from_qt(QGuiApplication.primaryScreen())
        t, n = 0.0, 0
        while dlg.calib_screen is not None and n < max_frames:
            scr = dlg.calib_screen
            press_enter_if_waiting(scr)
            px, py = scr.session.current_point
            tgt = np.array([[px / geom["width"] * 1920, py / geom["height"] * 1080]])
            clock.t = 100.0 + t
            tr.frame_ready.emit(make_frame(t, synth_features(tgt, rng)[0]))
            t += 1 / 30
            n += 1
        return n

    def test_completed_run_is_saved_with_both_stages(self):
        from gaze_session import load_session

        with tempfile.TemporaryDirectory() as d:
            clock = FakeClock()
            dlg = self.make_dialog(calibration_path=os.path.join(d, "calibration.json"), clock=clock,
                                   result_dialog_exec=lambda _d: True, record_dir=os.path.join(d, "sessions"))
            dlg.show()
            self.pump()
            dlg.start_calibration()
            self.assertEqual(len(dlg.calib_screen.session.validation), 5)
            frames = self.run_until_closed(dlg, clock, np.random.default_rng(8))
            self.assertIsNone(dlg.calib_screen)
            self.assertEqual(len(dlg.saved_sessions), 1)
            sess = load_session(dlg.saved_sessions[0])
            self.assertEqual(len(sess), frames)
            self.assertEqual(sess.meta["outcome"], "completed")
            geom = screen_geometry_from_qt(QGuiApplication.primaryScreen())
            self.assertEqual(sess.meta["screen"], geom)
            self.assertEqual(len(sess.meta["validation_points"]), 5)
            self.assertEqual(set(sess.point[sess.capture_mask("calibration")].tolist()), set(range(13)))
            self.assertEqual((sess.meta["calibration_layout"], sess.meta["wait_for_start"]), ("grid13", True))
            self.assertEqual(set(sess.point[sess.capture_mask("validation")].tolist()), set(range(5)))
            # fase de movimiento de cabeza al final (solo en --record-session), mirando el centro
            head = sess.stage == 2
            self.assertEqual(set(sess.point[head & (sess.phase == 1)].tolist()), set(range(len(HEAD_MOTION_STEPS))))
            np.testing.assert_allclose(sess.target[head][0], [geom["width"] / 2, geom["height"] / 2])
            self.assertEqual([s[0] for s in sess.meta["head_motion_steps"]], [k for k, _t, _s in HEAD_MOTION_STEPS])
            self.assertTrue(os.path.isfile(os.path.join(d, "calibration.json")))  # Aceptar también calibra

    def test_cancelled_run_is_saved_too(self):
        from gaze_session import load_session

        with tempfile.TemporaryDirectory() as d:
            clock = FakeClock()
            dlg = self.make_dialog(clock=clock, record_dir=d)
            dlg.show()
            self.pump()
            dlg.start_calibration()
            tr = self.trackers[0]
            for i in range(10):
                tr.frame_ready.emit(make_frame(i / 30, np.zeros(9)))
            QTest.keyPress(dlg.calib_screen, Qt.Key_Escape)
            clock.t += 2.0
            dlg.calib_screen.tick()
            self.assertIsNone(dlg.calib_screen)
            self.assertIn("Sesión guardada en", dlg.status_label.text())
            sess = load_session(dlg.saved_sessions[0])
            self.assertEqual(sess.meta["outcome"], "aborted")
            self.assertEqual(len(sess), 10)
            dlg.close()

    def test_normal_calibration_does_not_record(self):
        with tempfile.TemporaryDirectory() as d:
            dlg = self.make_dialog(calibration_path=os.path.join(d, "c.json"))
            dlg.show()
            self.pump()
            dlg.start_calibration()
            self.assertIsNone(dlg.calib_screen.recorder)
            self.assertEqual(dlg.calib_screen.session.head_motion, [])  # sin fase de cabeza
            dlg.reject()
            self.assertEqual(dlg.saved_sessions, [])

    def test_validation_layout_keeps_points_clear(self):
        for w, h in ((1280, 720), (1920, 1080), (1366, 768), (2560, 1440)):
            session = CalibrationSession(w, h, validation=cui.validation_points(w, h))
            session.stage = "validation"
            scr = cui.CalibrationScreen(session, dpr=1.0, clock=FakeClock())
            scr.resize(w, h)
            regions = scr.layout_regions()
            sc = max(0.6, h / 1080.0)
            regions.pop("panel")  # solo mientras espera Enter
            for k in range(5):
                c = scr.logical_validation_point(k)
                keep_out = QRectF(c.x() - 220 * sc, c.y() - 60 * sc, 440 * sc, 120 * sc)
                for name, r in regions.items():
                    self.assertFalse(r.intersects(keep_out), f"{name} tapa la validación {k + 1} ({w}x{h})")
            names = list(regions)
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    self.assertFalse(regions[names[i]].intersects(regions[names[j]]))
            scr.show()
            self.pump()
            scr._sync_layout()
            self.assertEqual(scr.mini.geometry().topLeft().y(), int(regions["mini"].y()))
            if (w, h) == (1920, 1080):
                scr.grab().save(os.path.join(self.shots, "screen_validation.png"))
            scr.close()


class TestRecordSessionEntryPoint(UITestBase):
    def test_main_record_session_saves_and_prints_paths(self):
        from unittest import mock

        import main

        created = {}

        class StubDialog(QDialog):
            def __init__(self, config, record_dir=None):
                super().__init__()
                created["record_dir"] = record_dir
                self.saved_sessions = [os.path.join(record_dir, "session_x.npz")]

            def show(self):
                from PySide6.QtCore import QTimer

                QTimer.singleShot(0, lambda: self.done(0))

        out = __import__("io").StringIO()
        with tempfile.TemporaryDirectory() as d, \
                mock.patch("app_logging.setup_logging", return_value="log"), \
                mock.patch("app_logging.set_error_reporter"), \
                __import__("contextlib").redirect_stdout(out):
            code = main.run_record_session(dialog_factory=StubDialog, record_dir=d)
        self.app.setQuitOnLastWindowClosed(True)
        self.assertEqual(code, 0)
        self.assertEqual(created["record_dir"], d)
        self.assertIn("Sesión guardada:", out.getvalue())
        self.assertIn("replay_session.py", out.getvalue())


if __name__ == "__main__":
    unittest.main()
