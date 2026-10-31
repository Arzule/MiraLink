"""Botones flotantes (overlay_buttons), acciones de Windows (windows_integration), radio magnético en el
CursorController, su ciclo de vida en ControlSession y la rueda del mouse en la ventana de configuración.

Sin hardware ni sistema: las acciones usan teclas, lanzador, ventanas y registro falsos (nunca se abre
la Lupa, el teclado en pantalla ni Configuración) y la sesión de control usa tracker, mouse y atajo falsos.
"""

import os
import tempfile
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtCore import QPoint, QPointF, Qt, qInstallMessageHandler
from PySide6.QtGui import QGuiApplication, QWheelEvent
from PySide6.QtWidgets import QApplication

import config_manager as cm
import overlay_buttons as ob
import windows_integration as wi
from calibration import GazeMapper, save_calibration, screen_geometry_from_qt
from cursor_control import CursorController
from eye_tracker import TrackingFrame
from test_calibration import make_dataset
from test_control_ui import Base, FakeFeedback, FakeTracker
from test_cursor_control import FakeMouse


class Recorder:
    def __init__(self, hwnd=0, hotkey=True):
        self.keys, self.opened, self.closed = [], [], []
        self.hwnd = hwnd
        self.hotkey = hotkey

    def actions(self, dry_run=False):
        return wi.WindowsActions(
            dry_run=dry_run,
            send_keys=lambda vks: self.keys.append(tuple(vks)),
            open_target=self.opened.append,
            find_window_fn=lambda cls: self.hwnd,
            close_window=self.closed.append,
            color_hotkey_enabled=lambda: self.hotkey,
            log=lambda _m: None,
        )


class TestWindowsActions(unittest.TestCase):
    def test_zoom_uses_win_and_numpad_plus_minus(self):
        r = Recorder()
        a = r.actions()
        self.assertEqual(a.run("zoom_in"), "Lupa: acercar")
        self.assertEqual(a.run("zoom_out"), "Lupa: alejar")
        self.assertEqual(r.keys, [(wi.VK_LWIN, wi.VK_ADD), (wi.VK_LWIN, wi.VK_SUBTRACT)])

    def test_keyboard_opens_with_shell_and_closes_existing_window(self):
        r = Recorder()
        self.assertIn("abierto", r.actions().run("keyboard"))
        self.assertTrue(r.opened[0].lower().endswith("osk.exe"))
        r2 = Recorder(hwnd=1234)
        self.assertIn("cerrado", r2.actions().run("keyboard"))
        self.assertEqual((r2.closed, r2.opened), ([1234], []))

    def test_color_filter_hotkey_or_settings(self):
        r = Recorder(hotkey=True)
        r.actions().run("color_filter")
        self.assertEqual(r.keys, [(wi.VK_LWIN, wi.VK_CONTROL, wi.VK_C)])
        r2 = Recorder(hotkey=None)  # el atajo no está activado: se abre Configuración, no se toca el registro
        msg = r2.actions().run("color_filter")
        self.assertEqual((r2.keys, r2.opened), ([], [wi.COLOR_FILTER_SETTINGS]))
        self.assertIn("Configuración", msg)

    def test_dry_run_and_errors_never_touch_the_system(self):
        r = Recorder()
        a = r.actions(dry_run=True)
        for action in wi.ACTIONS:
            self.assertIn("simulación", a.run(action))
        self.assertEqual((r.keys, r.opened, r.closed), ([], [], []))

        def boom(_vks):
            raise OSError("sin permiso")

        b = wi.WindowsActions(send_keys=boom, log=lambda _m: None)
        self.assertIn("No se pudo", b.run("zoom_in"))
        with self.assertRaises(ValueError):
            b.run("otra")


class TestOverlayBar(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_buttons_are_big_accessible_and_emit_actions(self):
        bar = ob.OverlayBar(button_size=180)
        got = []
        bar.action_triggered.connect(got.append)
        self.assertEqual(list(bar.buttons), [a for a, _t, _n in ob.OVERLAY_BUTTONS])
        for btn in bar.buttons.values():
            self.assertGreaterEqual(btn.height(), 48)
            self.assertEqual(btn.width(), 180)
            self.assertTrue(btn.accessibleName())
            self.assertEqual(btn.focusPolicy(), Qt.NoFocus)
        self.assertTrue(bar.windowFlags() & Qt.WindowDoesNotAcceptFocus)
        self.assertTrue(bar.windowFlags() & Qt.WindowStaysOnTopHint)
        bar.buttons["zoom_in"].click()
        bar.buttons["keyboard"].click()
        self.assertEqual(got, ["zoom_in", "keyboard"])
        bar.close()

    def test_placed_at_the_edge_and_rects_in_physical_pixels(self):
        screen = QGuiApplication.primaryScreen()
        bar = ob.OverlayBar(screen, button_size=120, position="right")
        self.assertEqual(bar.target_rects(), [])  # oculta: sin atracción
        bar.show()
        self.app.processEvents()
        area = screen.geometry()
        self.assertEqual(bar.geometry().right(), area.right() - ob.EDGE_GAP)
        rects = bar.target_rects()
        dpr = screen.devicePixelRatio()
        self.assertEqual(len(rects), len(ob.OVERLAY_BUTTONS))
        for x, y, w, h in rects:
            self.assertAlmostEqual(w, 120 * dpr)
            self.assertTrue(0 <= x and x + w <= area.width() * dpr + 1)
        left = ob.OverlayBar(screen, button_size=120, position="left")
        left.show()
        self.app.processEvents()
        self.assertEqual(left.x(), area.left() + ob.EDGE_GAP)
        bar.close()
        left.close()
        with self.assertRaises(ValueError):
            ob.OverlayBar(position="arriba")


class TestMagneticSnapInController(unittest.TestCase):
    def make(self, targets, radius=60):
        class Mapper:
            def __init__(self):
                self.p = (0.0, 0.0)

            def predict_one(self, _f):
                return self.p

        mapper = Mapper()
        cfg = {"smoothing_percent": 0, "deadzone_px": 0, "feature_median_window": 1, "snap_radius_px": radius}
        mouse = FakeMouse()
        c = CursorController(mapper, mouse, 1000, 800, config_getter=lambda: cfg, snap_targets=lambda: targets)
        return c, mapper, mouse

    def feed(self, c, n, t0=0.0):
        for i in range(n):
            c.update(TrackingFrame(timestamp=t0 + i / 30, face_found=True, features=np.zeros(10)))
        return t0 + n / 30

    def test_cursor_is_pulled_to_the_button_center_and_released_outside(self):
        button = (800.0, 300.0, 180.0, 180.0)  # centro (890, 390)
        c, mapper, _mouse = self.make([button])
        mapper.p = (760.0, 330.0)  # a 40 px del borde: dentro del radio
        t = self.feed(c, 60)
        self.assertAlmostEqual(c.display[0], 890.0, delta=2)
        self.assertAlmostEqual(c.display[1], 390.0, delta=2)
        mapper.p = (500.0, 330.0)  # fuera del radio: sigue la mirada
        self.feed(c, 3, t)
        self.assertEqual(c.display, (500.0, 330.0))

    def test_without_targets_or_radius_nothing_changes(self):
        for targets, radius in (([], 60), ([(800.0, 300.0, 180.0, 180.0)], 0)):
            c, mapper, _ = self.make(targets, radius)
            mapper.p = (760.0, 330.0)
            self.feed(c, 10)
            self.assertEqual(c.display, (760.0, 330.0))


class FakeOverlay:
    def __init__(self, screen, cfg, on_action):
        self.cfg = cfg
        self.on_action = on_action
        self.visible = False
        self.deleted = False

    def setVisible(self, v):
        self.visible = bool(v)

    def hide(self):
        self.visible = False

    def raise_(self):
        pass

    def deleteLater(self):
        self.deleted = True

    def target_rects(self):
        return [(0.0, 0.0, 10.0, 10.0)] if self.visible else []


class FakeActions:
    def __init__(self, dry_run):
        self.dry_run = dry_run
        self.done = []

    def run(self, action):
        self.done.append(action)
        return f"hecho {action}"


class TestOverlayInControlSession(Base):
    def make_overlay_session(self, enabled=True):
        import control_session as cs

        x, y, groups, _ = make_dataset()
        mapper = GazeMapper()
        mapper.fit(x, y, groups)
        save_calibration(self.cal_path, mapper, self.geom)
        s = self.make_session(dry_run=True)
        self.overlays, self.action_objs = [], []

        def overlay_factory(screen, cfg, on_action):
            self.overlays.append(FakeOverlay(screen, cfg, on_action))
            return self.overlays[-1]

        def actions_factory(dry):
            self.action_objs.append(FakeActions(dry))
            return self.action_objs[-1]

        s.overlay_config_getter = lambda: {"enabled": enabled, "position": "right", "button_size": 180}
        s._overlay_factory = overlay_factory
        s._actions_factory = actions_factory
        return s

    def test_overlay_only_while_control_is_active(self):
        s = self.make_overlay_session()
        self.assertEqual(self.overlays, [])  # antes de iniciar el control no existe
        ok, msg = s.start()
        self.assertTrue(ok, msg)
        bar = self.overlays[0]
        self.assertTrue(bar.visible)
        self.assertEqual(s.controller.snap_targets(), [(0.0, 0.0, 10.0, 10.0)])
        s.set_paused(True)
        self.assertFalse(bar.visible)  # en pausa no hay botones
        s.set_paused(False)
        self.assertTrue(bar.visible)
        messages = []
        s.overlay_message.connect(messages.append)
        bar.on_action("zoom_in")
        self.assertEqual(self.action_objs[0].done, ["zoom_in"])
        self.assertTrue(self.action_objs[0].dry_run)
        self.assertEqual(messages, ["hecho zoom_in"])
        s.stop()
        self.assertFalse(bar.visible)
        self.assertTrue(bar.deleted)
        self.assertIsNone(s.overlay)
        bar.on_action("zoom_out")  # tras detener, un clic tardío no hace nada
        self.assertEqual(self.action_objs[0].done, ["zoom_in"])

    def test_disabled_overlay_is_not_created(self):
        s = self.make_overlay_session(enabled=False)
        ok, _ = s.start()
        self.assertTrue(ok)
        self.assertEqual(self.overlays, [])
        self.assertEqual(s.controller.snap_targets(), [])
        s.stop()


class TestConfig(unittest.TestCase):
    def test_overlay_and_snap_defaults_and_clamp(self):
        cfg = cm._validate_config(dict(cm.DEFAULT_CONFIG))
        self.assertEqual(cfg["overlay"]["enabled"], True)
        self.assertEqual(cfg["overlay"]["position"], "right")
        self.assertEqual(cfg["interaction"]["snap_radius_px"], 60)
        bad = cm._validate_config(dict(cm.DEFAULT_CONFIG, overlay={"enabled": "si", "position": "arriba", "button_size": 5}))
        self.assertEqual((bad["overlay"]["enabled"], bad["overlay"]["position"], bad["overlay"]["button_size"]), (True, "right", 96))
        self.assertEqual(bad["overlay"]["buttons"], {})
        inter = dict(cm.DEFAULT_CONFIG["interaction"], snap_radius_px=999)
        self.assertEqual(cm._validate_config(dict(cm.DEFAULT_CONFIG, interaction=inter))["interaction"]["snap_radius_px"], 300)


class TestNoWheelInConfigWindow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_wheel_does_not_change_sliders_spinboxes_or_combos(self):
        from PySide6.QtWidgets import QComboBox, QSlider, QSpinBox

        from config_ui import ConfigWindow

        with tempfile.TemporaryDirectory() as d:
            win = ConfigWindow(config_path=os.path.join(d, "config.json"))
            win.show()
            self.app.processEvents()
            widgets = win.findChildren(QSlider) + win.findChildren(QSpinBox) + win.findChildren(QComboBox)
            self.assertTrue(widgets)
            for w in widgets:
                before = w.currentIndex() if isinstance(w, QComboBox) else w.value()
                ev = QWheelEvent(QPointF(5, 5), QPointF(w.mapToGlobal(QPoint(5, 5))), QPoint(0, 0), QPoint(0, -120),
                                 Qt.NoButton, Qt.NoModifier, Qt.NoScrollPhase, False)
                QApplication.sendEvent(w, ev)
                after = w.currentIndex() if isinstance(w, QComboBox) else w.value()
                self.assertEqual(before, after, type(w).__name__)
                self.assertFalse(ev.isAccepted())  # la rueda sigue hacia la ventana (desplazamiento)
            win.close()


if __name__ == "__main__":
    unittest.main()
