"""Pruebas de ControlSession, CursorFeedback y el botón de control de ConfigWindow.

Todo con piezas falsas: no se abre la cámara, no se mueve el mouse real ni se registra el atajo global.
"""

import os
import tempfile
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtCore import QObject, Qt, Signal, qInstallMessageHandler
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QApplication

import control_session as cs
import cursor_feedback as cf
from calibration import GazeMapper, save_calibration, screen_geometry_from_qt
from config_manager import DEFAULT_CONFIG
from config_ui import ConfigWindow
from cursor_control import STATE_DWELL, STATE_HOLD, CursorSnapshot, FakeHotkey
from eye_tracker import TrackingFrame
from test_calibration import make_dataset
from test_cursor_control import FakeMouse

FORBIDDEN = ("QLayout", "Could not parse", "QWidget::", "QStyleSheetStyle", "QFont::", "QPainter::")


class FakeTracker(QObject):
    frame_ready = Signal(object)
    failed = Signal(str)

    def __init__(self, index=0):
        super().__init__()
        self.index = index
        self.started = False
        self.stopped = 0

    def start(self):
        self.started = True

    def stop(self):
        self.stopped += 1


class FakeFeedback:
    def __init__(self, dpr=1.0, dry_run=False):
        self.dpr, self.dry_run = dpr, dry_run
        self.shown = False
        self.hidden = False
        self.snapshots = []
        self.events = []

    def show(self):
        self.shown = True

    def hide(self):
        self.hidden = True

    def deleteLater(self):
        pass

    def update(self):
        pass

    def update_snapshot(self, snap):
        self.snapshots.append(snap)

    def notify(self, name):
        self.events.append(name)


class FakeClock:
    def __init__(self, t=0.0):
        self.t = t

    def __call__(self):
        return self.t


def frame(ts, closed=False, blink=None):
    return TrackingFrame(
        timestamp=ts, face_found=True, features=np.zeros(9), eyes_closed=closed, blink_event=blink
    )


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.messages = []
        qInstallMessageHandler(lambda t, c, m: self.messages.append(str(m)))
        self.tmp = tempfile.TemporaryDirectory()
        self.cal_path = os.path.join(self.tmp.name, "calibration.json")
        self.screen = QGuiApplication.primaryScreen()
        self.geom = screen_geometry_from_qt(self.screen)

    def tearDown(self):
        qInstallMessageHandler(None)
        self.tmp.cleanup()
        for m in self.messages:
            for token in FORBIDDEN:
                self.assertNotIn(token, m, f"Mensaje Qt prohibido: {m}")

    def pump(self, n=3):
        for _ in range(n):
            self.app.processEvents()

    def save_valid_calibration(self, geometry=None):
        x, y, groups, _ = make_dataset()
        mapper = GazeMapper()
        mapper.fit(x, y, groups)
        save_calibration(self.cal_path, mapper, geometry or self.geom)

    def make_session(self, dry_run=False, hotkey_ok=True, screen_size=None, cfg=None):
        self.tracker = FakeTracker()
        self.mouse = FakeMouse()
        self.feedback = FakeFeedback()
        self.hotkeys = []
        self.mouse_created = 0
        interaction = cfg if cfg is not None else dict(DEFAULT_CONFIG["interaction"])
        self.cfg = interaction
        self.clock = FakeClock()

        def hk(callback):
            h = FakeHotkey(callback, register_ok=hotkey_ok, error="El atajo Ctrl+Alt+P ya está en uso (código 1409).")
            self.hotkeys.append(h)
            return h

        def mouse():
            self.mouse_created += 1
            return self.mouse

        size = screen_size or (self.geom["width"], self.geom["height"])
        return cs.ControlSession(
            config_getter=lambda: self.cfg,
            dry_run=dry_run,
            screen=self.screen,
            calibration_path=self.cal_path,
            tracker_factory=lambda i: self.tracker,
            mouse_factory=mouse,
            hotkey_factory=hk,
            feedback_factory=lambda **kw: self.feedback,
            screen_size_fn=lambda: size,
            clock=self.clock,
        )


class TestControlSessionPreconditions(Base):
    def test_requires_calibration(self):
        s = self.make_session()
        ok, msg = s.start()
        self.assertFalse(ok)
        self.assertEqual(msg, "Primero calibra el rastreador")
        self.assertEqual(self.hotkeys, [])
        self.assertEqual(self.mouse_created, 0)
        self.assertFalse(self.tracker.started)

    def test_calibration_with_other_geometry_is_rejected(self):
        self.save_valid_calibration({"x": 0, "y": 0, "width": 99999, "height": 99999})
        s = self.make_session()
        ok, msg = s.start()
        self.assertFalse(ok)
        self.assertIn("no es válida", msg)
        self.assertIn("geometría", msg)
        self.assertEqual(self.mouse_created, 0)

    def test_dpi_mismatch_blocks_real_control_but_not_dry_run(self):
        self.save_valid_calibration()
        s = self.make_session(screen_size=(self.geom["width"] + 1, self.geom["height"]))
        ok, msg = s.start()
        self.assertFalse(ok)
        self.assertIn("no coincide", msg)
        self.assertIn("--dry-run", msg)
        self.assertEqual(self.hotkeys, [])
        self.assertEqual(self.mouse_created, 0)
        s2 = self.make_session(dry_run=True, screen_size=(1, 1))
        ok2, _ = s2.start()
        self.assertTrue(ok2)
        s2.stop()

    def test_screen_size_error_blocks_real_control(self):
        self.save_valid_calibration()
        s = self.make_session()

        def boom():
            raise OSError("sin pantalla")

        s.screen_size_fn = boom
        ok, msg = s.start()
        self.assertFalse(ok)
        self.assertIn("sin pantalla", msg)

    def test_hotkey_failure_blocks_real_control(self):
        self.save_valid_calibration()
        s = self.make_session(hotkey_ok=False)
        ok, msg = s.start()
        self.assertFalse(ok)
        self.assertIn("ya está en uso", msg)
        self.assertIn("--dry-run", msg)
        self.assertEqual(self.mouse_created, 0)  # nunca se creó el mouse real
        self.assertFalse(self.tracker.started)
        self.assertTrue(self.hotkeys[0].stopped)

    def test_hotkey_failure_allows_dry_run_with_warning(self):
        self.save_valid_calibration()
        s = self.make_session(dry_run=True, hotkey_ok=False)
        ok, msg = s.start()
        self.assertTrue(ok)
        self.assertIn("solo simulación", msg)
        self.assertIn("ya está en uso", msg)
        s.stop()


class TestControlSessionRunning(Base):
    def start_running(self, **kw):
        self.save_valid_calibration()
        s = self.make_session(**kw)
        self.paused_log = []
        self.active_log = []
        self.failed_log = []
        s.paused_changed.connect(self.paused_log.append)
        s.tracker_active_changed.connect(self.active_log.append)
        s.failed.connect(self.failed_log.append)
        ok, msg = s.start()
        self.assertTrue(ok, msg)
        return s

    def feed(self, s, seconds, t0=0.0, fps=30.0):
        t = t0
        for _ in range(int(seconds * fps)):
            self.tracker.frame_ready.emit(frame(t))
            t += 1 / fps
        return t

    def test_start_wires_everything(self):
        s = self.start_running()
        self.assertTrue(self.tracker.started)
        self.assertTrue(self.feedback.shown)
        self.assertTrue(self.hotkeys[0].registered)
        self.assertEqual(self.active_log, [True])
        self.assertEqual(self.paused_log, [False])
        self.assertTrue(s.running)
        self.feed(s, 0.2)
        self.assertGreater(len(self.feedback.snapshots), 3)
        self.assertTrue(any(e[0] == "move" for e in self.mouse.events))
        s.stop()

    def test_hotkey_pauses_and_releases_hold(self):
        s = self.start_running()
        t = self.feed(s, 1.5)
        self.assertTrue(self.mouse.is_down)
        self.assertEqual(self.feedback.snapshots[-1].state, STATE_HOLD)
        self.hotkeys[0].press()
        self.assertTrue(s.paused)
        self.assertFalse(self.mouse.is_down)
        self.assertEqual(self.paused_log[-1], True)
        n = len(self.mouse.events)
        self.feed(s, 1.0, t0=t)
        self.assertEqual(len(self.mouse.events), n)  # pausado: ignora todo
        self.hotkeys[0].press()
        self.assertFalse(s.paused)
        self.assertEqual(self.paused_log[-1], False)
        s.stop()

    def test_live_config_changes_apply_without_restart(self):
        s = self.start_running()
        self.cfg["dwell_hold_ms"] = 3000
        self.feed(s, 1.6)
        self.assertFalse(self.mouse.is_down)
        self.cfg["dwell_hold_ms"] = 800
        self.feed(s, 0.2, t0=1.6)
        self.assertTrue(self.mouse.is_down)
        s.stop()

    def test_stop_releases_mouse_and_is_idempotent(self):
        s = self.start_running()
        self.feed(s, 1.5)
        self.assertTrue(self.mouse.is_down)
        s.stop()
        s.stop()
        self.assertFalse(self.mouse.is_down)
        self.assertEqual(self.tracker.stopped, 1)
        self.assertTrue(self.hotkeys[0].stopped)
        self.assertTrue(self.feedback.hidden)
        self.assertFalse(s.running)
        self.assertEqual(self.active_log, [True, False])

    def test_tracker_failure_releases_and_reports(self):
        s = self.start_running()
        self.feed(s, 1.5)
        self.assertTrue(self.mouse.is_down)
        self.tracker.failed.emit("la cámara se desconectó")
        self.assertFalse(self.mouse.is_down)
        self.assertFalse(s.running)
        self.assertEqual(len(self.failed_log), 1)
        self.assertIn("la cámara se desconectó", self.failed_log[0])

    def test_exception_in_update_stops_safely(self):
        s = self.start_running()
        self.feed(s, 1.5)
        self.assertTrue(self.mouse.is_down)
        self.tracker.frame_ready.emit(object())  # cuadro corrupto
        self.assertFalse(self.mouse.is_down)
        self.assertFalse(s.running)
        self.assertEqual(len(self.failed_log), 1)

    def test_timer_tick_auto_releases_hold_without_frames(self):
        self.cfg_override = dict(DEFAULT_CONFIG["interaction"], hold_timeout_s=5)
        self.save_valid_calibration()
        s = self.make_session(cfg=self.cfg_override)
        self.assertTrue(s.start()[0])
        t = self.feed(s, 1.5)
        self.assertTrue(self.mouse.is_down)
        self.clock.t = t + 10
        s._on_timer()
        self.assertFalse(self.mouse.is_down)
        s.stop()

    def test_dry_run_uses_dry_mouse_by_default(self):
        self.save_valid_calibration()
        s = self.make_session(dry_run=True)
        s._mouse_factory = None  # backend por defecto del modo simulación
        self.assertTrue(s.start()[0])
        from cursor_control import DryRunMouse

        self.assertIsInstance(s.mouse, DryRunMouse)
        s.mouse.log = None
        self.feed(s, 1.5)
        self.assertEqual([e[0] for e in s.mouse.events], ["mouse_down"])
        s.stop()


class TestCursorFeedback(Base):
    def make(self, **kw):
        self.clock = FakeClock(10.0)
        w = cf.CursorFeedback(clock=self.clock, **kw)
        return w

    def test_window_flags_transparent_for_input(self):
        w = self.make()
        flags = w.windowFlags()
        for flag in (Qt.WindowTransparentForInput, Qt.WindowStaysOnTopHint, Qt.FramelessWindowHint):
            self.assertTrue(flags & flag, flag)
        self.assertTrue(w.testAttribute(Qt.WA_TranslucentBackground))
        self.assertTrue(w.testAttribute(Qt.WA_TransparentForMouseEvents))

    def test_arc_only_after_300ms_of_stillness(self):
        w = self.make()
        w.snapshot = CursorSnapshot(state=STATE_DWELL, dwell_elapsed_ms=250, dwell_total_ms=1200)
        self.assertEqual(w.dwell_progress, 0.0)
        w.snapshot = CursorSnapshot(state=STATE_DWELL, dwell_elapsed_ms=600, dwell_total_ms=1200)
        self.assertAlmostEqual(w.dwell_progress, 0.5)
        w.snapshot = CursorSnapshot(state=STATE_DWELL, dwell_elapsed_ms=5000, dwell_total_ms=1200)
        self.assertEqual(w.dwell_progress, 1.0)

    def test_follows_position_with_dpr(self):
        w = self.make(dpr=2.0)
        w.update_snapshot(CursorSnapshot(x=1000, y=600))
        self.assertEqual((w.x(), w.y()), (500 - 70, 300 - 70))

    def _alpha_at_ring(self, w, dx=30, dy=0):
        img = w.grab().toImage()
        c = w.SIZE // 2
        return img.pixelColor(c + dx, c + dy).alpha(), img

    def test_painting_states(self):
        w = self.make()
        w.show()
        self.pump()
        # reposo: transparente
        w.update_snapshot(CursorSnapshot(state="IDLE"))
        a, _ = self._alpha_at_ring(w)
        self.assertEqual(a, 0)
        # arco de dwell visible a mitad de recorrido (sobre el anillo, en la derecha)
        w.update_snapshot(CursorSnapshot(state=STATE_DWELL, dwell_elapsed_ms=900, dwell_total_ms=1200))
        a, img = self._alpha_at_ring(w)
        self.assertGreater(a, 200)
        # donde el arco aún no llega (izquierda, ángulo 180°-270°... a 75 % solo queda libre el cuadrante superior izquierdo)
        c = w.SIZE // 2
        self.assertEqual(img.pixelColor(c - 21, c - 21).alpha(), 0)
        # HOLD: relleno en el centro
        w.update_snapshot(CursorSnapshot(state=STATE_HOLD))
        a, _ = self._alpha_at_ring(w, dx=0, dy=0)
        self.assertGreater(a, 80)
        # pausa: no dibuja nada
        w.update_snapshot(CursorSnapshot(state=STATE_HOLD, paused=True))
        a, _ = self._alpha_at_ring(w, dx=0, dy=0)
        self.assertEqual(a, 0)
        w.close()

    def test_flash_on_click_then_fades(self):
        w = self.make()
        w.show()
        w.update_snapshot(CursorSnapshot(state="IDLE"))
        self.assertFalse(w.flash_active())
        w.notify("click")
        self.assertTrue(w.flash_active())
        self.clock.t += 0.05
        a, _ = self._alpha_at_ring(w, dx=30 + int(24 * 0.2) + 2)
        self.assertGreater(a, 0)
        self.clock.t += 0.3
        self.assertFalse(w.flash_active())
        w._on_anim()
        a, _ = self._alpha_at_ring(w)
        self.assertEqual(a, 0)
        w.close()

    def test_dry_run_draws_crosshair_and_event_label(self):
        w = self.make(dry_run=True)
        w.show()
        w.update_snapshot(CursorSnapshot(state="IDLE"))
        a, _ = self._alpha_at_ring(w, dx=5, dy=0)  # mira en el centro
        self.assertGreater(a, 0)
        w.notify("hold_start")
        self.assertEqual(w._label, "Mantener")
        self.clock.t += 1.5
        w._on_anim()
        self.assertIsNone(w._label)
        w.close()

    def test_real_mode_has_no_crosshair_or_label(self):
        w = self.make(dry_run=False)
        w.show()
        w.notify("hold_start")
        self.assertIsNone(w._label)
        w.update_snapshot(CursorSnapshot(state="IDLE"))
        a, _ = self._alpha_at_ring(w, dx=5, dy=0)
        self.assertEqual(a, 0)
        w.close()


class StubSession(QObject):
    paused_changed = Signal(bool)
    tracker_active_changed = Signal(bool)
    failed = Signal(str)

    def __init__(self, ok=True, message=""):
        super().__init__()
        self.ok, self.message = ok, message
        self.running = False
        self.toggles = 0
        self.stops = 0

    def start(self):
        if self.ok:
            self.running = True
            self.tracker_active_changed.emit(True)
            self.paused_changed.emit(False)
        return self.ok, self.message

    def toggle_pause(self):
        self.toggles += 1
        self.paused_changed.emit(self.toggles % 2 == 1)

    def stop(self):
        self.stops += 1
        self.running = False


class TestConfigWindowControl(Base):
    def make_window(self, factory, dry_run=False):
        win = ConfigWindow(config_path=os.path.join(self.tmp.name, "config.json"), dry_run=dry_run)
        win.control_session_factory = factory
        win.show()
        self.pump()
        return win

    def test_button_initial_state_and_size(self):
        win = self.make_window(lambda: StubSession())
        self.assertEqual(win.btn_control.text(), "Iniciar Control del Cursor")
        self.assertGreaterEqual(win.btn_control.height(), 48)
        self.assertTrue(win.btn_control.accessibleName())
        self.assertEqual(win.status_badge.text(), "RASTREADOR OCULAR INACTIVO")
        win.close()

    def test_needs_calibration_message_shown(self):
        win = self.make_window(lambda: StubSession(ok=False, message="Primero calibra el rastreador"))
        win.btn_control.click()
        self.assertEqual(win.control_label.text(), "Primero calibra el rastreador")
        self.assertEqual(win.btn_control.text(), "Iniciar Control del Cursor")
        self.assertIsNone(win.control_session)
        win.close()

    def test_start_pause_resume_labels_and_badge(self):
        stub = StubSession()
        win = self.make_window(lambda: stub)
        win.btn_control.click()
        self.assertEqual(win.btn_control.text(), "Pausar Control del Cursor")
        self.assertEqual(win.status_badge.text(), "RASTREADOR OCULAR ACTIVO")
        self.assertIn("Control activo", win.control_label.text())
        win.btn_control.click()
        self.assertEqual(stub.toggles, 1)
        self.assertEqual(win.btn_control.text(), "Iniciar Control del Cursor")
        self.assertIn("pausado", win.control_label.text())
        win.btn_control.click()
        self.assertEqual(win.btn_control.text(), "Pausar Control del Cursor")
        win.close()

    def test_dry_run_label_mentions_simulation(self):
        win = self.make_window(lambda: StubSession(), dry_run=True)
        win.btn_control.click()
        self.assertIn("simulación", win.control_label.text())
        win.close()

    def test_close_stops_session(self):
        stub = StubSession()
        win = self.make_window(lambda: stub)
        win.btn_control.click()
        win.close()
        self.assertEqual(stub.stops, 1)
        self.assertEqual(win.status_badge.text(), "RASTREADOR OCULAR INACTIVO")

    def test_failure_resets_button(self):
        stub = StubSession()
        win = self.make_window(lambda: stub)
        win.btn_control.click()
        stub.failed.emit("Se detuvo el seguimiento: x")
        self.assertEqual(win.btn_control.text(), "Iniciar Control del Cursor")
        self.assertIn("Se detuvo", win.control_label.text())
        self.assertIsNone(win.control_session)
        win.close()

    def test_calibrating_stops_running_control(self):
        stub = StubSession()
        win = self.make_window(lambda: stub)
        win.btn_control.click()

        class Dlg(QObject):
            camera_selected = Signal(int)
            finished = Signal(int)
            accepted_mapper = None

            def open(self):
                self.finished.emit(0)

        win.calibration_dialog_factory = Dlg
        win.btn_calibrate.click()
        self.assertEqual(stub.stops, 1)
        self.assertEqual(win.btn_control.text(), "Iniciar Control del Cursor")
        win.close()

    def test_sliders_apply_live_to_real_session_and_close_releases_mouse(self):
        self.save_valid_calibration()
        holder = {}

        def factory():
            s = self.make_session()
            s.config_getter = lambda: win._config["interaction"]
            holder["s"] = s
            return s

        win = self.make_window(factory)
        win.btn_control.click()
        s = holder["s"]
        self.assertTrue(s.running)
        t = 0.0
        win.dwell_slider.setValue(3000)
        for _ in range(48):
            self.tracker.frame_ready.emit(frame(t))
            t += 1 / 30
        self.assertFalse(self.mouse.is_down)
        win.dwell_slider.setValue(800)  # sin guardar: se aplica en vivo
        for _ in range(4):
            self.tracker.frame_ready.emit(frame(t))
            t += 1 / 30
        self.assertTrue(self.mouse.is_down)
        self.assertEqual(s.controller._cfg("dwell_hold_ms"), 800)
        win.close()  # cerrar suelta el botón del mouse
        self.assertFalse(self.mouse.is_down)
        self.assertEqual(self.tracker.stopped, 1)
        self.assertTrue(self.hotkeys[0].stopped)

    def test_control_row_layout_across_themes_and_scales(self):
        from theme_manager import list_themes

        win = self.make_window(lambda: StubSession())
        for theme in list_themes():
            idx = win.theme_combo.findData(theme["id"])
            win.theme_combo.setCurrentIndex(idx)
            for pct in (80, 100, 200):
                win.font_slider.setValue(pct)
                win.control_label.setText("Control activo. Ctrl+Alt+P pausa en cualquier momento.")
                self.pump()
                for w in (win.btn_calibrate, win.btn_save, win.btn_control, win.btn_recenter, win.control_label,
                          win.save_label):
                    self.assertTrue(w.isVisible())
                    self.assertGreater(w.width(), 0)
                self.assertGreaterEqual(win.btn_recenter.height(), 48)
                rects = {}
                for name in ("btn_calibrate", "btn_save", "btn_control", "btn_recenter", "control_label"):
                    w = getattr(win, name)
                    rects[name] = type(w.rect())(w.mapTo(win, w.rect().topLeft()), w.rect().size())
                names = list(rects)
                for i in range(len(names)):
                    for j in range(i + 1, len(names)):
                        self.assertFalse(
                            rects[names[i]].intersects(rects[names[j]]),
                            f"{names[i]} y {names[j]} se superponen ({theme['id']} {pct}%)",
                        )
                self.assertGreaterEqual(win.btn_control.height(), 48)
        win.close()


if __name__ == "__main__":
    unittest.main()
