"""Integración del flujo real: ConfigWindow -> modal -> "Iniciar Calibración" -> pantalla completa
-> 9 puntos -> diálogo de resultado -> calibration.json.

Usa Qt real, el TrackerThread real y piezas falsas solo en el borde del hardware: una cámara falsa
(cuenta aperturas y falla si hay dos simultáneas) y un FaceTracker falso que genera TrackingFrames
coherentes (rostro presente, ojos abiertos, rasgos que dependen del punto mirado). Nunca abre la
cámara real. Los clics se programan con temporizadores para que el test funcione también si el modal
bloquea con exec() (así se demostró que el defecto original lo hace fallar).
"""

import gc
import os
import tempfile
import threading
import time
import unittest
import weakref

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtCore import QObject, Qt, QTimer, Signal, qInstallMessageHandler
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

import calibration_ui as cui
from calibration import load_calibration, screen_geometry_from_qt
from config_ui import ConfigWindow
from eye_tracker import TrackerThread, TrackingFrame
from test_calibration import synth_features

FORBIDDEN = ("QLayout", "Could not parse", "QWidget::", "QStyleSheetStyle", "QFont::", "QPainter::", "QThread")


class CameraRegistry:
    """Cuenta aperturas de la cámara falsa y detecta aperturas simultáneas."""

    def __init__(self):
        self.lock = threading.Lock()
        self.opens = 0
        self.open_now = 0
        self.max_concurrent = 0
        self.fail_open = False


class FakeCamera:
    def __init__(self, registry, index):
        self.registry = registry
        self.index = index
        self.is_open = False

    def open(self):
        r = self.registry
        if r.fail_open:
            raise RuntimeError(f"No se pudo abrir la cámara {self.index}. Revisa que no esté en uso.")
        with r.lock:
            r.opens += 1
            r.open_now += 1
            r.max_concurrent = max(r.max_concurrent, r.open_now)
        self.is_open = True

    def read(self):
        if not self.is_open:
            return False, None
        time.sleep(0.002)  # limita el ritmo del hilo
        return True, np.full((480, 640, 3), 90, np.uint8)

    def release(self):
        if self.is_open:
            with self.registry.lock:
                self.registry.open_now -= 1
            self.is_open = False


class GazeSource:
    """Usuario simulado: mira el punto activo de la pantalla de calibración del modal abierto."""

    def __init__(self):
        self._dialog_ref = None  # referencia DÉBIL: el test no debe mantener vivo el modal
        self.geom = None
        self.rng = np.random.default_rng(7)
        self.points_seen = set()

    @property
    def dialog(self):
        return self._dialog_ref() if self._dialog_ref is not None else None

    @dialog.setter
    def dialog(self, dlg):
        self._dialog_ref = weakref.ref(dlg) if dlg is not None else None

    def features(self):
        dlg = self.dialog
        scr = getattr(dlg, "calib_screen", None) if dlg is not None else None
        if scr is None or self.geom is None:
            target = np.array([[960.0, 540.0]])
        else:
            idx = scr.session.point_index
            self.points_seen.add(idx)
            px, py = scr.session.points[min(idx, 8)]
            target = np.array([[px / self.geom["width"] * 1920, py / self.geom["height"] * 1080]])
        return synth_features(target, self.rng)[0]


class FakeFaceTracker:
    def __init__(self, gaze):
        self.gaze = gaze

    def process(self, image, ts):
        return TrackingFrame(
            timestamp=ts,
            face_found=True,
            ear_left=0.3,
            ear_right=0.3,
            ear_mean=0.3,
            features=self.gaze.features(),
            face_center=(0.5, 0.5),
            interocular=0.2,
        )

    def close(self):
        pass


class VirtualClock:
    """Reloj del hilo: avanza 1/30 s por cuadro (sin depender del reloj real)."""

    def __init__(self):
        self.t = 1000.0
        self.lock = threading.Lock()

    def __call__(self):
        with self.lock:
            self.t += 1.0 / 30.0
            return self.t


class FlowBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.messages = []
        qInstallMessageHandler(lambda t, c, m: self.messages.append(str(m)))
        self.tmp = tempfile.TemporaryDirectory()
        self.cal_path = os.path.join(self.tmp.name, "calibration.json")
        self.registry = CameraRegistry()
        self.gaze = GazeSource()
        self.threads = []
        self.answers = []  # respuestas del diálogo de resultado (True = Aceptar)
        self.result_dialogs = 0

    def tearDown(self):
        for t in self.threads:
            t.stop()
        from PySide6.QtCore import QEvent

        QApplication.sendPostedEvents(None, QEvent.DeferredDelete)  # no dejar ventanas para el siguiente test
        qInstallMessageHandler(None)
        self.tmp.cleanup()
        for m in self.messages:
            for token in FORBIDDEN:
                self.assertNotIn(token, m, f"Mensaje Qt prohibido: {m}")

    def tracker_factory(self, index):
        clock = VirtualClock()
        t = TrackerThread(
            index,
            camera_factory=lambda i: FakeCamera(self.registry, i),
            tracker_factory=lambda _p: FakeFaceTracker(self.gaze),
            clock=clock,
        )
        self.threads.append(t)
        return t

    def result_exec(self, dlg):
        self.result_dialogs += 1
        self.assertIn("Calibración completada. Error estimado:", dlg.message_label.text())
        return self.answers.pop(0) if self.answers else True

    def make_window(self, **kw):
        win = ConfigWindow(config_path=os.path.join(self.tmp.name, "config.json"))
        kwargs = dict(
            camera_lister=lambda: [0],
            tracker_factory=self.tracker_factory,
            calibration_path=self.cal_path,
            result_dialog_exec=self.result_exec,
        )
        kwargs.update(kw)
        win.calibration_dialog_kwargs = kwargs
        factory = win.calibration_dialog_factory

        def tracked_factory():
            dlg = factory()
            self.gaze.dialog = dlg
            return dlg

        win.calibration_dialog_factory = tracked_factory
        win.show()
        self.pump()
        self.gaze.geom = screen_geometry_from_qt(win.screen())
        return win

    def pump(self, ms=0):
        end = time.monotonic() + ms / 1000.0
        while True:
            self.app.processEvents()
            if time.monotonic() >= end:
                break
            time.sleep(0.005)

    def wait_until(self, cond, timeout_s=10.0):
        end = time.monotonic() + timeout_s
        while time.monotonic() < end:
            self.app.processEvents()
            if cond():
                return True
            time.sleep(0.005)
        return cond()

    def find_dialog(self):
        """El modal creado en ESTE test (el usuario simulado lo registra al crearse)."""
        dlg = self.gaze.dialog
        try:
            if dlg is not None and dlg.objectName() is not None:
                return dlg
        except RuntimeError:  # objeto C++ ya destruido
            return None
        return None

    def visible_screens(self):
        out = []
        for w in self.app.topLevelWidgets():
            try:
                if isinstance(w, cui.CalibrationScreen) and w.isVisible():
                    out.append(w)
            except RuntimeError:
                pass
        return out

    def open_and_start(self, win):
        """Clic real en "Calibrar" y luego en "Iniciar Calibración" (programado para soportar exec())."""
        state = {"clicked": False}

        def click_start():
            dlg = self.find_dialog()
            if dlg is None or dlg.preview.badge_text != "Rostro Detectado":
                QTimer.singleShot(20, click_start)  # espera a que la previsualización tenga cuadros
                return
            QTest.mouseClick(dlg.btn_start, Qt.LeftButton)
            state["clicked"] = True

        QTimer.singleShot(20, click_start)
        QTest.mouseClick(win.btn_calibrate, Qt.LeftButton)  # con exec() bloquea hasta que termina
        self.assertTrue(self.wait_until(lambda: state["clicked"]), "No se llegó a pulsar 'Iniciar Calibración'")
        gc.collect()  # el defecto: sin referencia retenida, el modal y la pantalla se destruían aquí
        self.pump(150)
        gc.collect()
        self.pump(50)


class TestCalibrationFlow(FlowBase):
    def test_start_opens_fullscreen_and_completes_nine_points(self):
        win = self.make_window()
        self.open_and_start(win)

        screens = self.visible_screens()
        self.assertEqual(len(screens), 1, "La pantalla completa de calibración no está visible tras 'Iniciar'")
        scr = screens[0]
        self.assertTrue(scr.isFullScreen())
        self.assertEqual(scr.geometry(), win.screen().geometry())
        dlg = self.find_dialog()
        self.assertIsNotNone(dlg, "El modal fue destruido al iniciar la calibración")
        self.assertFalse(dlg.isVisible())
        # un solo dueño de la cámara: se abrió una vez y nunca dos a la vez
        self.assertEqual(self.registry.opens, 1)
        self.assertEqual(self.registry.max_concurrent, 1)

        ok = self.wait_until(lambda: win.calibration_dialog is None, timeout_s=30)
        self.assertTrue(ok, f"La calibración no terminó; puntos vistos: {sorted(self.gaze.points_seen)}")
        self.assertEqual(self.gaze.points_seen, set(range(9)))
        self.assertEqual(self.result_dialogs, 1)
        self.assertTrue(os.path.isfile(self.cal_path))
        mapper = load_calibration(self.cal_path, screen_geometry_from_qt(win.screen()))
        # precisión: la cubre test_calibration (aquí los cuadros encolados del hilo falso llegan con retraso)
        self.assertTrue(np.isfinite(mapper.metrics["rmse_px"]))
        self.assertEqual(win.save_label.text(), "Calibración guardada")
        # al terminar se liberó la cámara y terminó el hilo
        self.assertTrue(self.wait_until(lambda: self.registry.open_now == 0))
        self.assertTrue(all(t.isFinished() for t in self.threads))
        self.assertEqual(self.registry.max_concurrent, 1)
        win.close()

    def test_repeat_restarts_with_new_fullscreen_and_same_camera(self):
        self.answers = [False, True]  # "Repetir" y luego "Aceptar"
        win = self.make_window()
        self.open_and_start(win)
        dlg = self.find_dialog()
        screens = []

        def track():
            scr = dlg.calib_screen if win.calibration_dialog is not None else None
            if scr is not None and (not screens or screens[-1] is not scr):
                screens.append(scr)
                self.assertTrue(scr.isVisible() and scr.isFullScreen())
            return win.calibration_dialog is None

        ok = self.wait_until(track, timeout_s=30)
        scr = dlg.calib_screen
        self.assertTrue(ok, f"pantallas={len(screens)} resultados={self.result_dialogs} fase={scr.session.phase if scr else None} punto={scr.session.point_index if scr else None} msg={scr.session.message if scr else dlg.status_label.text()} visible_modal={dlg.isVisible()} hilos_vivos={[t.isRunning() for t in self.threads]}")
        self.assertEqual(len(screens), 2)
        self.assertEqual(self.result_dialogs, 2)
        self.assertTrue(os.path.isfile(self.cal_path))
        self.assertEqual(self.registry.opens, 1)  # "Repetir" reutiliza la misma cámara
        self.assertEqual(self.registry.max_concurrent, 1)
        win.close()

    def test_holding_escape_returns_to_modal(self):
        win = self.make_window()
        self.open_and_start(win)
        scr = self.visible_screens()[0]
        dlg = self.find_dialog()
        QTest.keyPress(scr, Qt.Key_Escape)
        self.assertTrue(self.wait_until(lambda: win.calibration_dialog is not None and dlg.calib_screen is None, 5))
        self.assertTrue(dlg.isVisible())
        self.assertIn("cancelada", dlg.status_label.text())
        self.assertEqual(self.visible_screens(), [])
        self.assertFalse(os.path.isfile(self.cal_path))
        self.assertEqual(self.registry.open_now, 1)  # la previsualización sigue con la misma cámara
        dlg.reject()
        self.assertTrue(self.wait_until(lambda: self.registry.open_now == 0))
        win.close()

    def test_closing_config_window_during_calibration_releases_camera(self):
        win = self.make_window()
        self.open_and_start(win)
        self.assertEqual(len(self.visible_screens()), 1)
        win.close()
        self.pump(50)
        self.assertEqual(self.visible_screens(), [])
        self.assertEqual(self.registry.open_now, 0)
        self.assertTrue(all(t.isFinished() for t in self.threads))


class TestCalibrationFlowErrors(FlowBase):
    def test_camera_open_failure_is_visible_in_modal(self):
        self.registry.fail_open = True
        win = self.make_window()
        QTest.mouseClick(win.btn_calibrate, Qt.LeftButton)
        dlg = win.calibration_dialog
        self.assertIsNotNone(dlg)
        self.assertTrue(self.wait_until(lambda: "No se pudo usar la cámara" in dlg.status_label.text()))
        self.assertTrue(dlg.isVisible())
        dlg.reject()
        win.close()

    def test_fullscreen_failure_is_visible_and_modal_returns(self):
        win = self.make_window()
        QTest.mouseClick(win.btn_calibrate, Qt.LeftButton)
        dlg = win.calibration_dialog
        original = cui.CalibrationScreen

        class Broken(original):
            def showFullScreen(self):
                raise RuntimeError("fallo simulado de la ventana")

        cui.CalibrationScreen = Broken
        try:
            QTest.mouseClick(dlg.btn_start, Qt.LeftButton)
        finally:
            cui.CalibrationScreen = original
        self.pump(20)
        self.assertTrue(dlg.isVisible())
        self.assertIn("No se pudo abrir la pantalla de calibración", dlg.status_label.text())
        self.assertIn("fallo simulado", dlg.status_label.text())
        self.assertIsNone(dlg.calib_screen)
        self.assertEqual(self.visible_screens(), [])
        dlg.reject()
        win.close()

    def test_second_click_on_calibrate_reuses_open_modal(self):
        win = self.make_window()
        QTest.mouseClick(win.btn_calibrate, Qt.LeftButton)
        first = win.calibration_dialog
        win.calibrate_requested.emit()
        self.assertIs(win.calibration_dialog, first)
        self.wait_until(lambda: self.registry.opens == 1)
        self.pump(50)
        self.assertEqual(self.registry.max_concurrent, 1)
        first.reject()
        self.pump()
        self.assertIsNone(win.calibration_dialog)
        win.close()


class StubSession(QObject):
    """Sesión de control que registra si arrancó mientras la cámara del modal seguía abierta."""

    paused_changed = Signal(bool)
    tracker_active_changed = Signal(bool)
    failed = Signal(str)

    def __init__(self, registry):
        super().__init__()
        self.registry = registry
        self.running = False
        self.camera_busy_at_start = None

    def start(self):
        self.camera_busy_at_start = self.registry.open_now
        self.running = True
        return True, ""

    def toggle_pause(self):
        pass

    def stop(self):
        self.running = False


class TestControlAfterCalibration(FlowBase):
    def test_control_starts_after_modal_closed_without_camera_clash(self):
        win = self.make_window()
        sessions = []
        win.control_session_factory = lambda: sessions.append(StubSession(self.registry)) or sessions[-1]
        self.open_and_start(win)
        self.assertTrue(self.wait_until(lambda: win.calibration_dialog is None, 30))
        self.assertTrue(self.wait_until(lambda: self.registry.open_now == 0))
        QTest.mouseClick(win.btn_control, Qt.LeftButton)
        self.assertEqual(len(sessions), 1)
        self.assertEqual(sessions[0].camera_busy_at_start, 0)  # la cámara del modal ya estaba libre
        win.close()
        self.assertFalse(sessions[0].running)

    def test_opening_calibration_stops_running_control_first(self):
        win = self.make_window()
        sessions = []
        win.control_session_factory = lambda: sessions.append(StubSession(self.registry)) or sessions[-1]
        QTest.mouseClick(win.btn_control, Qt.LeftButton)
        self.assertTrue(sessions[0].running)
        QTest.mouseClick(win.btn_calibrate, Qt.LeftButton)
        self.assertFalse(sessions[0].running)  # se detuvo antes de que el modal abriera la cámara
        win.calibration_dialog.reject()
        win.close()


if __name__ == "__main__":
    unittest.main()
