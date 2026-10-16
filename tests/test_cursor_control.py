import queue
import time
import unittest

import numpy as np

import cursor_control as cc
from config_manager import DEFAULT_CONFIG
from cursor_control import (
    STATE_DWELL,
    STATE_HOLD,
    STATE_IDLE,
    CursorController,
    DryRunMouse,
    FakeHotkey,
    PyAutoGUIMouse,
    Win32Hotkey,
)
from eye_tracker import BlinkEvent, TrackingFrame

W, H = 1920, 1080


class FakeMouse(cc.MouseBackend):
    def __init__(self):
        self.events = []  # (nombre, x, y)
        self.pos = (0.0, 0.0)
        self.is_down = False

    def move(self, x, y):
        self.pos = (x, y)
        self.events.append(("move", x, y))

    def down(self):
        self.is_down = True
        self.events.append(("down",) + self.pos)

    def up(self):
        self.is_down = False
        self.events.append(("up",) + self.pos)

    def click(self):
        self.events.append(("click",) + self.pos)

    def names(self):
        return [e[0] for e in self.events if e[0] != "move"]


class IdentityMapper:
    """Los dos primeros rasgos son la posición de pantalla (simplifica los tests)."""

    def predict_one(self, features):
        return float(features[0]), float(features[1])


def frame(ts, x=500.0, y=400.0, closed=False, blink=None, face=True):
    return TrackingFrame(
        timestamp=ts,
        face_found=face,
        features=np.array([x, y, 0, 0, 0, 0, 0, 0, 0], dtype=float) if face else None,
        eyes_closed=closed,
        blink_event=blink,
    )


def blink(end_ts, duration_ms):
    return BlinkEvent(start_ts=end_ts - duration_ms / 1000.0, end_ts=end_ts, duration_ms=duration_ms)


class Base(unittest.TestCase):
    def setUp(self):
        self.mouse = FakeMouse()
        self.cfg = dict(DEFAULT_CONFIG["interaction"])
        # la mayoría de las pruebas usa el cursor sin filtros (los filtros se prueban en TestFilters)
        self.cfg["smoothing_percent"] = 0
        self.cfg["feature_median_window"] = 1
        self.cfg["deadzone_px"] = 0
        self.events = []
        self.ctl = CursorController(
            IdentityMapper(), self.mouse, W, H, config_getter=lambda: self.cfg,
            listener=lambda n, x, y: self.events.append(n),
        )
        self.t = 0.0

    def hold_gaze(self, seconds, x=500.0, y=400.0, fps=30.0, **kw):
        """Mantiene la mirada fija `seconds` con cuadros espaciados por timestamps."""
        n = int(round(seconds * fps))
        snap = None
        for _ in range(n):
            snap = self.ctl.update(frame(self.t, x, y, **kw))
            self.t += 1.0 / fps
        return snap

    def do_blink(self, duration_ms, x=500.0, y=400.0, fps=30.0, wander=None):
        """Ojos cerrados `duration_ms` (la mirada 'vaga' si wander) y reapertura con BlinkEvent."""
        n = max(1, int(round(duration_ms / 1000.0 * fps)))
        for i in range(n):
            gx, gy = (wander if wander else (x, y))
            self.ctl.update(frame(self.t, gx, gy, closed=True))
            self.t += 1.0 / fps
        return self.ctl.update(frame(self.t, x, y, blink=blink(self.t, duration_ms)))


class TestBlinks(Base):
    def test_short_blink_ignored(self):
        self.hold_gaze(0.3)
        self.do_blink(150)
        self.assertEqual(self.mouse.names(), [])

    def test_valid_blink_without_hold_is_single_click_at_frozen_position(self):
        self.hold_gaze(0.3)
        self.assertEqual(self.mouse.pos, (500.0, 400.0))
        # los ojos cerrados y reabiertos: la estimación de mirada se desvía a (900, 700), no es fiable
        for _ in range(8):
            self.ctl.update(frame(self.t, 900, 700, closed=True))
            self.t += 1 / 30
        self.ctl.update(frame(self.t, 900, 700, blink=blink(self.t, 270)))
        self.t += 1 / 30
        self.assertEqual(self.mouse.names(), ["click"])
        click = [e for e in self.mouse.events if e[0] == "click"][0]
        self.assertEqual(click[1:], (500.0, 400.0))
        self.assertEqual([e for e in self.mouse.events if e[0] == "move"][-1][1:], (500.0, 400.0))
        self.assertIn("click", self.events)

    def test_freeze_after_reopen_holds_position_then_resumes(self):
        self.cfg["freeze_after_blink_ms"] = 150
        self.hold_gaze(0.3)
        self.do_blink(300, wander=(900, 700))
        snap = self.ctl.update(frame(self.t, 900, 700))  # 33 ms después: congelado
        self.assertTrue(snap.frozen)
        self.assertEqual((snap.x, snap.y), (500.0, 400.0))
        self.t += 0.2
        snap = self.ctl.update(frame(self.t, 900, 700))
        self.assertFalse(snap.frozen)
        self.assertEqual((snap.x, snap.y), (900.0, 700.0))

    def test_blink_longer_than_max_ignored(self):
        self.hold_gaze(0.3)
        self.do_blink(2500)
        self.assertEqual(self.mouse.names(), [])

    def test_ignores_blinks_300ms_after_click(self):
        self.hold_gaze(0.3)
        self.do_blink(300)
        self.assertEqual(self.mouse.names(), ["click"])
        self.t += 0.05
        self.ctl.update(frame(self.t, blink=blink(self.t, 300)))  # 50 ms después del clic
        self.assertEqual(self.mouse.names(), ["click"])
        self.t += 0.5
        self.ctl.update(frame(self.t, blink=blink(self.t, 300)))
        self.assertEqual(self.mouse.names(), ["click", "click"])

    def test_face_lost_freezes_without_clicks(self):
        self.hold_gaze(0.3)
        snap = self.ctl.update(frame(self.t, face=False, blink=blink(self.t, 400)))
        self.t += 0.1
        self.assertTrue(snap.frozen)
        self.assertFalse(snap.face_found)
        self.assertEqual(self.mouse.names(), [])
        snap = self.ctl.update(frame(self.t, face=False))
        self.assertEqual((snap.x, snap.y), (500.0, 400.0))
        self.assertEqual(self.mouse.names(), [])

    def test_live_threshold_change(self):
        self.hold_gaze(0.3)
        self.do_blink(200)  # < 250: ignorado
        self.assertEqual(self.mouse.names(), [])
        self.cfg["blink_click_ms"] = 150
        self.t += 0.5
        self.do_blink(200)
        self.assertEqual(self.mouse.names(), ["click"])


class TestDwellAndHold(Base):
    def test_full_dwell_pressed_then_drag_then_blink_releases(self):
        self.hold_gaze(1.0)  # < 1200 ms
        self.assertEqual(self.mouse.names(), [])
        snap = self.hold_gaze(0.4)
        self.assertEqual(self.mouse.names(), ["down"])
        self.assertEqual(snap.state, STATE_HOLD)
        self.assertTrue(self.mouse.is_down)
        n_moves = len([e for e in self.mouse.events if e[0] == "move"])
        self.hold_gaze(0.3, x=900, y=500)  # mover mantiene el arrastre
        self.assertGreater(len([e for e in self.mouse.events if e[0] == "move"]), n_moves)
        self.assertEqual(self.mouse.names(), ["down"])
        self.do_blink(320, x=900, y=500)  # >= blink_release_ms (300)
        self.assertEqual(self.mouse.names(), ["down", "up"])
        self.assertFalse(self.mouse.is_down)
        self.assertEqual(self.ctl.state, STATE_IDLE)

    def test_short_blink_with_hold_does_not_release(self):
        self.hold_gaze(1.3)
        self.assertTrue(self.mouse.is_down)
        self.do_blink(200)  # < 300
        self.assertTrue(self.mouse.is_down)
        self.assertEqual(self.mouse.names(), ["down"])

    def test_blink_with_hold_never_clicks(self):
        self.hold_gaze(1.3)
        self.do_blink(500)
        self.assertEqual(self.mouse.names(), ["down", "up"])

    def test_leaving_radius_restarts_dwell(self):
        self.cfg["dwell_radius_px"] = 50
        self.hold_gaze(1.0)
        self.hold_gaze(1.0, x=700)  # salió del radio antes de tiempo
        self.assertEqual(self.mouse.names(), [])
        self.hold_gaze(0.5, x=700)  # ahora sí acumula >= 1200 ms desde que llegó
        self.assertEqual(self.mouse.names(), ["down"])

    def test_small_jitter_inside_radius_keeps_dwelling(self):
        rng = np.random.default_rng(0)
        for _ in range(60):
            self.ctl.update(frame(self.t, 500 + rng.uniform(-15, 15), 400 + rng.uniform(-15, 15)))
            self.t += 1 / 30
        self.assertEqual(self.mouse.names(), ["down"])

    def test_states_idle_dwell_hold(self):
        snap = self.ctl.update(frame(self.t))
        self.assertEqual(snap.state, STATE_IDLE)
        self.t += 1 / 30
        snap = self.ctl.update(frame(self.t))
        self.assertEqual(snap.state, STATE_DWELL)
        self.assertGreaterEqual(snap.dwell_total_ms, 800)
        self.assertEqual(self.hold_gaze(1.5).state, STATE_HOLD)

    def test_hold_disabled_never_presses(self):
        self.cfg["hold_enabled"] = False
        self.hold_gaze(3.0)
        self.assertEqual(self.mouse.names(), [])

    def test_dwell_clock_stops_while_eyes_closed(self):
        self.hold_gaze(0.8)
        for _ in range(90):  # 3 s con los ojos cerrados
            self.ctl.update(frame(self.t, closed=True))
            self.t += 1 / 30
        self.assertEqual(self.mouse.names(), [])
        self.hold_gaze(0.2)
        self.assertEqual(self.mouse.names(), [])  # solo lleva ~0.85 s de dwell efectivo
        self.hold_gaze(0.6)
        self.assertEqual(self.mouse.names(), ["down"])

    def test_live_dwell_threshold(self):
        self.cfg["dwell_hold_ms"] = 3000
        self.hold_gaze(1.5)
        self.assertEqual(self.mouse.names(), [])
        self.cfg["dwell_hold_ms"] = 800
        self.hold_gaze(0.1)
        self.assertEqual(self.mouse.names(), ["down"])

    def test_dwell_does_not_depend_on_frame_rate(self):
        for fps in (10.0, 60.0):
            mouse = FakeMouse()
            ctl = CursorController(IdentityMapper(), mouse, W, H, config_getter=lambda: self.cfg)
            t = 0.0
            while t < 1.1:
                ctl.update(frame(t))
                t += 1 / fps
            self.assertEqual(mouse.names(), [], f"{fps} fps")
            while t < 1.4:
                ctl.update(frame(t))
                t += 1 / fps
            self.assertEqual(mouse.names(), ["down"], f"{fps} fps")


class TestPauseAndSafety(Base):
    def test_pause_releases_hold_and_ignores_everything_then_resume_idle(self):
        self.hold_gaze(1.4)
        self.assertTrue(self.mouse.is_down)
        self.ctl.set_paused(True)
        self.assertFalse(self.mouse.is_down)
        self.assertEqual(self.mouse.names(), ["down", "up"])
        n = len(self.mouse.events)
        self.hold_gaze(2.0, x=900)
        self.ctl.update(frame(self.t, blink=blink(self.t, 400)))
        self.assertEqual(len(self.mouse.events), n)  # ni movimiento ni clic
        self.ctl.set_paused(False)
        self.assertEqual(self.ctl.state, STATE_IDLE)
        self.hold_gaze(1.5, x=900)
        self.assertEqual(self.mouse.names(), ["down", "up", "down"])

    def test_toggle_pause(self):
        self.assertTrue(self.ctl.toggle_pause())
        self.assertFalse(self.ctl.toggle_pause())

    def test_auto_release_after_hold_timeout(self):
        self.cfg["hold_timeout_s"] = 5
        self.hold_gaze(1.4)
        self.assertTrue(self.mouse.is_down)
        self.hold_gaze(4.0)  # 5.4 s en total, pero el hold empezó a ~1.2 s -> 4.2 s de hold
        self.assertTrue(self.mouse.is_down)
        self.hold_gaze(1.5)
        self.assertFalse(self.mouse.is_down)
        self.assertIn("hold_timeout", self.events)

    def test_auto_release_by_tick_without_frames(self):
        self.cfg["hold_timeout_s"] = 5
        self.hold_gaze(1.4)
        self.ctl.tick(self.t + 3)
        self.assertTrue(self.mouse.is_down)
        self.ctl.tick(self.t + 6)
        self.assertFalse(self.mouse.is_down)

    def test_release_all_is_idempotent(self):
        self.hold_gaze(1.4)
        self.ctl.release_all()
        self.ctl.release_all()
        self.assertEqual(self.mouse.names(), ["down", "up"])

    def test_release_all_when_mouse_up_raises_still_resets_state(self):
        self.hold_gaze(1.4)

        def boom():
            raise RuntimeError("fallo del mouse")

        self.mouse.up = boom
        with self.assertRaises(RuntimeError):
            self.ctl.release_all()
        self.assertEqual(self.ctl.state, STATE_IDLE)


class TestSmoothing(Base):
    """Suavizado exponencial anterior (filter_kind="exponential", se conserva para comparar)."""

    def setUp(self):
        super().setUp()
        self.ctl = CursorController(IdentityMapper(), self.mouse, W, H, config_getter=lambda: self.cfg,
                                    filter_kind="exponential")

    def test_alpha_formula(self):
        for pct, alpha in ((0, 1.0), (100, 0.05), (40, 0.62)):
            self.cfg["smoothing_percent"] = pct
            self.assertAlmostEqual(self.ctl.smoothing_alpha(), alpha)

    def test_zero_follows_raw_gaze(self):
        for kind in ("exponential", "one_euro"):
            ctl = CursorController(IdentityMapper(), FakeMouse(), W, H, config_getter=lambda: self.cfg, filter_kind=kind)
            self.cfg["smoothing_percent"] = 0
            ctl.update(frame(0.0, 100, 100))
            snap = ctl.update(frame(1 / 30, 1000, 800))
            self.assertEqual((snap.x, snap.y), (1000.0, 800.0), kind)

    def test_hundred_is_very_smooth(self):
        self.cfg["smoothing_percent"] = 100
        self.ctl.update(frame(0.0, 100, 100))
        snap = self.ctl.update(frame(1 / 30, 1100, 100))
        self.assertAlmostEqual(snap.x, 100 + 0.05 * 1000, places=6)
        # es temporal: con el doble de dt avanza 1 - 0.95**2
        ctl2 = CursorController(IdentityMapper(), FakeMouse(), W, H, config_getter=lambda: self.cfg,
                                filter_kind="exponential")
        ctl2.update(frame(0.0, 100, 100))
        snap2 = ctl2.update(frame(2 / 30, 1100, 100))
        self.assertAlmostEqual(snap2.x, 100 + (1 - 0.95**2) * 1000, places=6)

    def test_converges_to_target(self):
        for kind in ("exponential", "one_euro"):
            ctl = CursorController(IdentityMapper(), FakeMouse(), W, H, config_getter=lambda: self.cfg, filter_kind=kind)
            self.cfg["smoothing_percent"] = 100
            ctl.update(frame(0.0, 100, 100))
            t = 0.0
            for _ in range(300):
                t += 1 / 30
                snap = ctl.update(frame(t, 1000, 600))
            self.assertAlmostEqual(snap.x, 1000, delta=1, msg=kind)

    def test_unknown_filter_rejected(self):
        with self.assertRaises(ValueError):
            CursorController(IdentityMapper(), FakeMouse(), W, H, filter_kind="kalman")


class TestFilters(Base):
    """Pipeline nuevo: mediana de rasgos -> mapeo -> One Euro -> zona muerta."""

    def make(self, **cfg):
        self.cfg.update(cfg)
        self.mouse = FakeMouse()
        self.ctl = CursorController(IdentityMapper(), self.mouse, W, H, config_getter=lambda: self.cfg)

    def run_noise(self, seconds=3.0, noise=30.0, seed=0, x=800.0, y=500.0):
        rng = np.random.default_rng(seed)
        xs = []
        t = 0.0
        for _ in range(int(seconds * 30)):
            snap = self.ctl.update(frame(t, x + rng.normal(0, noise), y + rng.normal(0, noise)))
            t += 1 / 30
            xs.append((snap.x, snap.y))
        p = np.array(xs[30:])  # tras 1 s de estabilización
        return float(np.sqrt(np.mean(np.sum((p - p.mean(axis=0)) ** 2, axis=1))))

    def test_more_smoothing_means_less_jitter(self):
        jitters = []
        for pct in (0, 20, 40, 70, 100):
            self.make(smoothing_percent=pct)
            jitters.append(self.run_noise())
        self.assertEqual(jitters, sorted(jitters, reverse=True), jitters)
        self.assertLess(jitters[2], jitters[0] * 0.5)

    def test_one_euro_beats_exponential_on_jitter_at_same_setting(self):
        self.make(smoothing_percent=40)
        euro = self.run_noise()
        self.ctl = CursorController(IdentityMapper(), FakeMouse(), W, H, config_getter=lambda: self.cfg,
                                    filter_kind="exponential")
        expo = self.run_noise()
        self.assertLess(euro, expo)

    def test_fast_movement_has_little_lag_thanks_to_beta(self):
        # salto grande: con beta el corte sube y el cursor llega pronto; sin beta tarda mucho más
        def frames_to_90(beta):
            self.make(smoothing_percent=70, one_euro_beta=beta)
            t = 0.0
            for _ in range(30):
                self.ctl.update(frame(t, 200, 500))
                t += 1 / 30
            for k in range(1, 200):
                snap = self.ctl.update(frame(t, 1200, 500))
                t += 1 / 30
                if snap.x >= 200 + 0.9 * 1000:
                    return k
            return 999

        self.assertLess(frames_to_90(0.01), frames_to_90(0.0))
        self.assertLessEqual(frames_to_90(0.01), 6)

    def test_uses_timestamps_not_frame_count(self):
        # mismo recorrido a 15 y 30 fps: posición parecida al cabo de 0.5 s
        def pos_after(fps):
            self.make(smoothing_percent=60)
            t = 0.0
            self.ctl.update(frame(t, 200, 500))
            snap = None
            while t < 0.5 - 1e-9:
                t += 1 / fps
                snap = self.ctl.update(frame(t, 600, 500))
            return snap.x

        self.assertAlmostEqual(pos_after(15), pos_after(30), delta=25)

    def test_median_rejects_single_frame_spikes(self):
        self.make(feature_median_window=3)
        t = 0.0
        for i in range(30):
            x = 3000.0 if i == 15 else 800.0  # pico aislado del iris
            snap = self.ctl.update(frame(t, x, 500))
            t += 1 / 30
            self.assertEqual(snap.x, 800.0)
        moves = [e for e in self.mouse.events if e[0] == "move"]
        self.assertEqual(len(moves), 1)

    def test_median_window_one_passes_spikes(self):
        self.make(feature_median_window=1)
        self.ctl.update(frame(0.0, 800, 500))
        snap = self.ctl.update(frame(1 / 30, 1500, 500))
        self.assertEqual(snap.x, 1500.0)

    def test_deadzone_blocks_small_moves(self):
        self.make(deadzone_px=10)
        self.ctl.update(frame(0.0, 800, 500))
        snap = self.ctl.update(frame(1 / 30, 806, 504))  # 7.2 px < 10
        self.assertEqual((snap.x, snap.y), (800.0, 500.0))
        snap = self.ctl.update(frame(2 / 30, 812, 500))  # 12 px desde lo mostrado
        self.assertEqual((snap.x, snap.y), (812.0, 500.0))
        self.assertEqual(len([e for e in self.mouse.events if e[0] == "move"]), 2)

    def test_deadzone_keeps_dwell_stable_under_jitter(self):
        self.make(deadzone_px=6, smoothing_percent=40)
        rng = np.random.default_rng(3)
        t = 0.0
        snap = None
        for _ in range(int(1.5 * 30)):
            snap = self.ctl.update(frame(t, 800 + rng.normal(0, 4), 500 + rng.normal(0, 4)))
            t += 1 / 30
        self.assertEqual(snap.state, STATE_HOLD)

    def test_live_filter_settings(self):
        self.make(smoothing_percent=100)
        self.ctl.update(frame(0.0, 100, 100))
        slow = self.ctl.update(frame(1 / 30, 1100, 100)).x
        self.cfg["smoothing_percent"] = 0  # el slider se aplica en vivo
        fast = self.ctl.update(frame(2 / 30, 1100, 100)).x
        self.assertLess(slow, 1100)
        self.assertEqual(fast, 1100)

    def test_face_lost_resets_median(self):
        self.make(feature_median_window=5)
        t = 0.0
        for _ in range(5):
            self.ctl.update(frame(t, 300, 300))
            t += 1 / 30
        self.ctl.update(frame(t, 0, 0, face=False))
        t += 1 / 30
        self.assertEqual(len(self.ctl._median), 0)
        snap = self.ctl.update(frame(t, 900, 700))
        self.assertEqual((snap.x, snap.y), (900.0, 700.0))

    def test_clamped_to_screen_and_origin_applied(self):
        mouse = FakeMouse()
        ctl = CursorController(IdentityMapper(), mouse, W, H, origin=(100, 50), config_getter=lambda: self.cfg)
        snap = ctl.update(frame(0.0, -500, 99999))
        self.assertEqual((snap.x, snap.y), (100.0, 50.0 + H - 1))
        self.assertEqual(mouse.pos, (100.0, 50.0 + H - 1))

    def test_non_finite_estimate_is_skipped(self):
        self.ctl.update(frame(0.0, 500, 400))
        snap = self.ctl.update(frame(0.1, float("nan"), 400))
        self.assertEqual((snap.x, snap.y), (500.0, 400.0))


class TestBackends(unittest.TestCase):
    def test_pyautogui_mouse_configures_and_forwards(self):
        class FakePG:
            FAILSAFE = True
            PAUSE = 0.1

            def __init__(self):
                self.calls = []

            def moveTo(self, x, y, **kw):
                self.calls.append(("moveTo", x, y))

            def mouseDown(self, **kw):
                self.calls.append(("down",))

            def mouseUp(self, **kw):
                self.calls.append(("up",))

            def click(self, **kw):
                self.calls.append(("click",))

        pg = FakePG()
        m = PyAutoGUIMouse(pg)
        self.assertFalse(pg.FAILSAFE)
        self.assertEqual(pg.PAUSE, 0)
        m.move(10.4, 20.6)
        m.down()
        m.safe_release()
        m.safe_release()
        m.click()
        self.assertEqual(pg.calls, [("moveTo", 10, 21), ("down",), ("up",), ("click",)])

    def test_dry_run_mouse_only_records(self):
        lines = []
        m = DryRunMouse(log=lines.append)
        m.move(5, 6)
        m.down()
        m.up()
        m.click()
        self.assertEqual([e[0] for e in m.events], ["mouse_down", "mouse_up", "click"])
        self.assertEqual(len(lines), 3)
        self.assertIn("dry-run", lines[0])

    def test_fake_hotkey_toggles_controller_pause(self):
        ctl = CursorController(IdentityMapper(), FakeMouse(), W, H)
        hk = FakeHotkey(ctl.toggle_pause)
        self.assertTrue(hk.start())
        hk.press()
        self.assertTrue(ctl.paused)
        hk.press()
        self.assertFalse(ctl.paused)

    def test_fake_hotkey_registration_failure(self):
        hk = FakeHotkey(lambda: None, register_ok=False, error="en uso")
        self.assertFalse(hk.start())
        self.assertEqual(hk.error, "en uso")


class FakeWinApi:
    instances = []

    def __init__(self, ok=True, err=""):
        self.ok, self.err = ok, err
        self.q = queue.Queue()
        self.unregistered = False
        FakeWinApi.instances.append(self)

    def register(self):
        return self.ok, self.err

    def get_message(self):
        return self.q.get(timeout=3)

    def unregister(self):
        self.unregistered = True

    def post_quit(self):
        self.q.put("quit")


class TestWin32HotkeyWithFakeApi(unittest.TestCase):
    def test_hotkey_calls_back_and_stops_cleanly(self):
        FakeWinApi.instances.clear()
        hits = []
        hk = Win32Hotkey(lambda: hits.append(1), api_factory=FakeWinApi)
        self.assertTrue(hk.start())
        api = FakeWinApi.instances[0]
        api.q.put("other")
        api.q.put("hotkey")
        api.q.put("hotkey")
        deadline = time.time() + 2
        while len(hits) < 2 and time.time() < deadline:
            time.sleep(0.01)
        self.assertEqual(len(hits), 2)
        hk.stop()
        self.assertTrue(api.unregistered)
        self.assertFalse(hk.registered)

    def test_registration_failure_reports_error(self):
        hk = Win32Hotkey(lambda: None, api_factory=lambda: FakeWinApi(ok=False, err="El atajo Ctrl+Alt+P ya está en uso"))
        self.assertFalse(hk.start())
        self.assertIn("en uso", hk.error)
        hk.stop()

    def test_callback_exception_does_not_kill_thread(self):
        FakeWinApi.instances.clear()
        hits = []

        def cb():
            hits.append(1)
            raise RuntimeError("falla de la GUI")

        hk = Win32Hotkey(cb, api_factory=FakeWinApi)
        hk.start()
        api = FakeWinApi.instances[0]
        api.q.put("hotkey")
        api.q.put("hotkey")
        deadline = time.time() + 2
        while len(hits) < 2 and time.time() < deadline:
            time.sleep(0.01)
        self.assertEqual(len(hits), 2)
        hk.stop()

    def test_api_factory_exception_is_reported(self):
        def bad():
            raise OSError("sin user32")

        hk = Win32Hotkey(lambda: None, api_factory=bad)
        self.assertFalse(hk.start())
        self.assertIn("sin user32", hk.error)


if __name__ == "__main__":
    unittest.main()
