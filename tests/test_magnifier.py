"""Lupa propia de MiraLink (magnifier.py) y su uso en ControlSession y en la barra de botones.

Sin tocar la pantalla: la ampliación real (Magnification.dll) se sustituye por FakeMagnification, y la
sesión de control usa tracker, mouse y atajo falsos. La ampliación real no se verificó (la prueba el usuario).
"""

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QApplication

import magnifier as mg
import overlay_buttons as ob
from calibration import GazeMapper, save_calibration
from eye_tracker import TrackingFrame
from test_calibration import make_dataset
from test_control_ui import Base
from test_overlay import FakeActions, FakeOverlay

W, H = 2560, 1440


class TestZoomView(unittest.TestCase):
    def test_display_and_desktop_are_inverse_and_zoom_keeps_center(self):
        v = mg.ZoomView(W, H)
        self.assertFalse(v.active)
        self.assertEqual(v.to_desktop(100, 200), (100, 200))
        self.assertTrue(v.zoom_in())
        self.assertEqual(v.zoom, 1.5)
        self.assertTrue(v.zoom_in())
        self.assertEqual(v.zoom, 2.0)
        self.assertEqual(v.view_rect(), (640.0, 360.0, 1280.0, 720.0))  # centrada
        self.assertEqual(v.to_desktop(0, 0), (640.0, 360.0))
        self.assertEqual(v.to_desktop(W, H), (1920.0, 1080.0))
        for p in ((10.0, 20.0), (2000.0, 1300.0)):
            self.assertEqual(v.to_display(*v.to_desktop(*p)), p)
        self.assertEqual(v.rect_to_display((640.0, 360.0, 100.0, 50.0)), (0.0, 0.0, 200.0, 100.0))
        v.zoom_in()
        self.assertFalse(v.zoom_in())  # 3x es el máximo
        self.assertTrue(v.reset())
        self.assertEqual((v.zoom, v.ox, v.oy), (1.0, 0.0, 0.0))
        self.assertFalse(v.zoom_out())

    def test_pan_after_delay_towards_the_edge_and_clamped(self):
        v = mg.ZoomView(W, H)
        v.set_level(2)  # 2x, vista en (640, 360)
        right_edge = (W - 5.0, H / 2)
        self.assertFalse(v.update_pan(right_edge, 0.2))  # todavía un vistazo
        self.assertTrue(v.update_pan(right_edge, 0.25))
        self.assertGreater(v.ox, 640.0)
        self.assertEqual(v.oy, 360.0)
        for _ in range(200):
            v.update_pan(right_edge, 1 / 30)
        self.assertEqual(v.ox, W - W / 2)  # tope: la vista no sale del escritorio
        before = (v.ox, v.oy)
        self.assertFalse(v.update_pan((W / 2, H / 2), 1.0))  # el centro no desplaza
        self.assertFalse(v.update_pan((5.0, 5.0), 1.0, blocked=True))  # sobre un botón no desplaza
        self.assertFalse(v.update_pan(None, 1.0))
        self.assertEqual((v.ox, v.oy), before)
        v.reset()
        self.assertFalse(v.update_pan((5.0, 5.0), 1.0))  # sin zoom no hay desplazamiento

    def test_zoomed_mouse_moves_to_what_is_seen(self):
        class M:
            def __init__(self):
                self.moves, self.clicks = [], 0

            def move(self, x, y):
                self.moves.append((x, y))

            def click(self):
                self.clicks += 1

        inner = M()
        v = mg.ZoomView(W, H)
        zm = mg.ZoomedMouse(inner, v, origin=(0, 0))
        zm.move(100, 100)
        self.assertEqual(inner.moves[-1], (100, 100))
        v.set_level(2)
        zm.refresh()  # tras el zoom el cursor vuelve a quedar bajo la mirada
        self.assertEqual(inner.moves[-1], (690.0, 410.0))
        zm.click()
        self.assertEqual(inner.clicks, 1)
        self.assertEqual(mg.ZoomedMouse(inner, v, origin=(100, 0)).desktop_point(100, 0), (740.0, 360.0))


class TestOverlayInZoomedView(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_bar_shrinks_and_stays_inside_the_visible_area(self):
        screen = QGuiApplication.primaryScreen()
        dpr = screen.devicePixelRatio()
        bar = ob.OverlayBar(screen, button_size=180)
        bar.show()
        self.app.processEvents()
        full = bar.width()
        view = (400.0 * dpr, 200.0 * dpr, 600.0 * dpr, 500.0 * dpr)
        bar.set_view(view, 2.0)
        self.assertEqual(bar.buttons["zoom_in"].width(), 90)
        self.assertLess(bar.width(), full)
        g = bar.geometry()
        origin = screen.geometry().topLeft()
        self.assertLessEqual(g.right(), origin.x() + 400 + 600)
        self.assertGreaterEqual(g.left(), origin.x() + 400)
        self.assertGreaterEqual(g.top(), origin.y() + 200)
        bar.set_view(None)
        self.assertEqual(bar.buttons["zoom_in"].width(), 180)
        bar.close()


class TestZoomInControlSession(Base):
    def make_zoom_session(self):
        x, y, groups, _ = make_dataset()
        mapper = GazeMapper()
        mapper.fit(x, y, groups)
        save_calibration(self.cal_path, mapper, self.geom)
        s = self.make_session(dry_run=True)
        self.mags, self.overlays = [], []

        def overlay_factory(screen, cfg, on_action):
            self.overlays.append(FakeOverlay(screen, cfg, on_action))
            return self.overlays[-1]

        def mag_factory():
            self.mags.append(mg.FakeMagnification())
            return self.mags[-1]

        s.overlay_config_getter = lambda: {"enabled": True}
        s._overlay_factory = overlay_factory
        s._desktop_button_factory = lambda screen, cfg, on_action: FakeOverlay(screen, cfg, on_action)
        s._actions_factory = FakeActions
        s._foreground_is_desktop = lambda: self.desktop  # ventana activa falsa
        self.desktop = True
        s._magnifier_factory = mag_factory
        return s

    def test_zoom_buttons_use_miralink_magnifier_and_pause_resets(self):
        s = self.make_zoom_session()
        ok, msg = s.start()
        self.assertTrue(ok, msg)
        messages = []
        s.overlay_message.connect(messages.append)
        bar = self.overlays[0]
        bar.on_action("zoom_in")
        bar.on_action("zoom_in")
        self.assertEqual(s.zoom, 2.0)
        mag = self.mags[0]
        w, h = self.geom["width"], self.geom["height"]
        self.assertEqual(mag.calls[-1], (2.0, round(w / 4), round(h / 4)))  # centrada
        self.assertEqual(bar.view[1], 2.0)  # la barra se coloca en la zona visible
        self.assertIn("2x", messages[-1])
        # los botones se ven ampliados: el imán usa sus rectángulos en pantalla
        _rx, _ry, rw, rh = s._snap_targets()[0]
        self.assertEqual((rw, rh), (20.0, 20.0))
        s.set_paused(True)  # Ctrl+Alt+P: vuelve a 1x
        self.assertEqual(s.zoom, 1.0)
        self.assertEqual(mag.calls[-1], (1.0, 0, 0))
        s.set_paused(False)
        bar.on_action("zoom_in")
        s.stop()
        self.assertTrue(mag.closed)  # al detener, la pantalla vuelve siempre a 1x
        self.assertEqual(s.zoom_view.zoom, 1.0)

    def test_edge_gaze_pans_and_cursor_goes_to_what_is_seen(self):
        s = self.make_zoom_session()
        s.start()
        s.change_zoom(+1)
        s.change_zoom(+1)  # 2x
        view = s.zoom_view
        ox0 = view.ox
        ctrl = s.controller
        ctrl.display = (self.geom["width"] - 3.0, self.geom["height"] / 2)  # mirando el borde derecho
        ctrl._frozen = False
        for k in range(30):
            s._update_zoom_pan(TrackingFrame(timestamp=1.0 + k / 30, face_found=True, features=np.zeros(10)))
        self.assertGreater(view.ox, ox0)
        self.assertEqual(s.zoomed_mouse.desktop_point(0.0, 0.0), (view.ox, view.oy))
        s.recenter()  # Recentrar se hace sin ampliar
        self.assertEqual(s.zoom, 1.0)
        s.stop()


if __name__ == "__main__":
    unittest.main()
