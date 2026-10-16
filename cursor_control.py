"""Control del cursor: suavizado, máquina de estados (IDLE/DWELL/HOLD), backends de mouse y hotkey.

CursorController es lógica pura: usa solo los timestamps de TrackingFrame (sin reloj propio ni
tasa de cuadros fija), un MouseBackend inyectado y los umbrales de la configuración en vivo.
Los backends reales (PyAutoGUI, RegisterHotKey) van detrás de interfaces delgadas con versiones
falsas para los tests.

Suavizado exponencial por tiempo: alpha = 1 - 0.95 * smoothing_percent / 100 (referido a 30 fps);
con intervalo dt, alpha_efectivo = 1 - (1 - alpha) ** (dt * 30). smoothing_percent = 0 -> alpha = 1
(sin suavizado: el cursor sigue la estimación cruda); 100 -> alpha = 0.05 (muy suave).
"""

from __future__ import annotations

import atexit
import math
import queue
import threading
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from config_manager import DEFAULT_CONFIG

STATE_IDLE = "IDLE"
STATE_DWELL = "DWELL"
STATE_HOLD = "HOLD"

POST_ACTION_IGNORE_S = 0.3  # tras un clic o soltar se ignoran parpadeos
REFERENCE_FPS = 30.0
MAX_SMOOTHING_DT_S = 0.25  # un intervalo mayor se trata como este máximo (cámara con tirones)
HOTKEY_LABEL = "Ctrl+Alt+P"


# ----------------------------------------------------------------- backends de mouse
class MouseBackend:
    """Interfaz mínima del mouse. Coordenadas en píxeles físicos de pantalla."""

    def move(self, x: float, y: float) -> None:
        raise NotImplementedError

    def down(self) -> None:
        raise NotImplementedError

    def up(self) -> None:
        raise NotImplementedError

    def click(self) -> None:
        raise NotImplementedError


class PyAutoGUIMouse(MouseBackend):
    """Mouse real con PyAutoGUI (FAILSAFE desactivado: la mirada llega legítimamente a las esquinas)."""

    def __init__(self, pyautogui_module=None):
        if pyautogui_module is None:
            import pyautogui as pyautogui_module  # noqa: PLC0415
        self._pg = pyautogui_module
        self._pg.FAILSAFE = False
        self._pg.PAUSE = 0
        self._is_down = False
        atexit.register(self.safe_release)  # último recurso si el proceso termina con el botón presionado

    def move(self, x, y):
        self._pg.moveTo(int(round(x)), int(round(y)), _pause=False)

    def down(self):
        self._pg.mouseDown(_pause=False)
        self._is_down = True

    def up(self):
        try:
            self._pg.mouseUp(_pause=False)
        finally:
            self._is_down = False

    def click(self):
        self._pg.click(_pause=False)

    def safe_release(self):
        """Suelta el botón si quedó presionado (idempotente, no lanza)."""
        if self._is_down:
            try:
                self.up()
            except Exception:  # noqa: BLE001
                pass


class DryRunMouse(MouseBackend):
    """Backend de simulación: solo registra eventos (consola y lista), no mueve ni hace clic."""

    def __init__(self, log: Optional[Callable[[str], None]] = print):
        self.log = log
        self.events: List[Tuple[str, float, float]] = []
        self.position = (0.0, 0.0)

    def _record(self, name):
        self.events.append((name, self.position[0], self.position[1]))
        if self.log:
            self.log(f"[dry-run] {name} en ({self.position[0]:.0f}, {self.position[1]:.0f})")

    def move(self, x, y):
        self.position = (x, y)

    def down(self):
        self._record("mouse_down")

    def up(self):
        self._record("mouse_up")

    def click(self):
        self._record("click")


# ------------------------------------------------------------------- hotkey global
class _WinHotkeyApi:
    """Llamadas reales a user32 para el atajo global. Se usa solo desde Win32Hotkey."""

    MOD_ALT = 0x0001
    MOD_CONTROL = 0x0002
    MOD_NOREPEAT = 0x4000
    VK_P = 0x50
    WM_HOTKEY = 0x0312
    WM_QUIT = 0x0012
    HOTKEY_ID = 0x4D4C

    def __init__(self):
        import ctypes
        from ctypes import wintypes

        self._ctypes = ctypes
        self._wintypes = wintypes
        self.user32 = ctypes.WinDLL("user32", use_last_error=True)
        self.kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._thread_id = None

    def register(self) -> Tuple[bool, str]:
        self._thread_id = self.kernel32.GetCurrentThreadId()
        ok = self.user32.RegisterHotKey(
            None, self.HOTKEY_ID, self.MOD_CONTROL | self.MOD_ALT | self.MOD_NOREPEAT, self.VK_P
        )
        if ok:
            return True, ""
        code = self._ctypes.get_last_error()
        if code == 1409:
            return False, f"El atajo {HOTKEY_LABEL} ya está en uso por otra aplicación (código 1409)."
        return False, f"No se pudo registrar el atajo {HOTKEY_LABEL} (código de Windows {code})."

    def get_message(self) -> str:
        msg = self._wintypes.MSG()
        r = self.user32.GetMessageW(self._ctypes.byref(msg), None, 0, 0)
        if r <= 0:
            return "quit"
        return "hotkey" if msg.message == self.WM_HOTKEY else "other"

    def unregister(self):
        self.user32.UnregisterHotKey(None, self.HOTKEY_ID)

    def post_quit(self):
        if self._thread_id is not None:
            self.user32.PostThreadMessageW(self._thread_id, self.WM_QUIT, 0, 0)


class HotkeyBackend:
    """Interfaz del atajo global de emergencia."""

    registered = False
    error = ""

    def start(self) -> bool:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError


class Win32Hotkey(HotkeyBackend):
    """Ctrl+Alt+P global con RegisterHotKey en su propio hilo. `callback` se llama desde ese hilo."""

    def __init__(self, callback: Callable[[], None], api_factory: Callable[[], object] = _WinHotkeyApi):
        self.callback = callback
        self._api_factory = api_factory
        self._api = None
        self._thread: Optional[threading.Thread] = None
        self._ready = threading.Event()
        self.registered = False
        self.error = ""

    def start(self, timeout: float = 2.0) -> bool:
        self._thread = threading.Thread(target=self._run, name="miralink-hotkey", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout):
            self.error = f"No se pudo iniciar el atajo {HOTKEY_LABEL} a tiempo."
            return False
        return self.registered

    def _run(self):
        try:
            self._api = self._api_factory()
            ok, err = self._api.register()
        except Exception as e:  # noqa: BLE001
            ok, err = False, f"No se pudo registrar el atajo {HOTKEY_LABEL}: {e}"
        self.registered = bool(ok)
        self.error = "" if ok else err
        self._ready.set()
        if not ok:
            return
        try:
            while True:
                kind = self._api.get_message()
                if kind == "quit":
                    break
                if kind == "hotkey":
                    try:
                        self.callback()
                    except Exception:  # noqa: BLE001 - el hilo del atajo nunca debe morir por la GUI
                        pass
        finally:
            self._api.unregister()
            self.registered = False

    def stop(self):
        if self._thread is not None and self._thread.is_alive() and self._api is not None:
            self._api.post_quit()
            self._thread.join(2.0)
        self._thread = None


class FakeHotkey(HotkeyBackend):
    """Versión falsa para tests: `press()` simula el atajo."""

    def __init__(self, callback: Callable[[], None], register_ok: bool = True, error: str = ""):
        self.callback = callback
        self._ok = register_ok
        self._err = error
        self.registered = False
        self.error = ""
        self.stopped = False

    def start(self):
        self.registered = self._ok
        self.error = "" if self._ok else self._err
        return self.registered

    def stop(self):
        self.stopped = True
        self.registered = False

    def press(self):
        if self.registered:
            self.callback()


# --------------------------------------------------------------- máquina de estados
@dataclass
class CursorSnapshot:
    """Estado observable del controlador tras un cuadro (para la retroalimentación visual)."""

    state: str = STATE_IDLE
    x: float = 0.0  # píxeles físicos de pantalla (incluye el origen del monitor)
    y: float = 0.0
    dwell_elapsed_ms: float = 0.0
    dwell_total_ms: float = 0.0
    frozen: bool = False
    paused: bool = False
    face_found: bool = False
    positioned: bool = False  # ya hay una posición válida de cursor


class CursorController:
    """Pipeline mirada -> cursor y máquina de estados IDLE / DWELL / HOLD (ver docstring del módulo)."""

    def __init__(
        self,
        mapper,
        mouse: MouseBackend,
        screen_width: int,
        screen_height: int,
        origin: Tuple[int, int] = (0, 0),
        config_getter: Callable[[], Dict] = lambda: {},
        listener: Optional[Callable[[str, float, float], None]] = None,
    ):
        self.mapper = mapper
        self.mouse = mouse
        self.width = int(screen_width)
        self.height = int(screen_height)
        self.origin = origin
        self.config_getter = config_getter
        self.listener = listener
        self.state = STATE_IDLE
        self.paused = False
        self.pos: Optional[Tuple[float, float]] = None  # relativa a la pantalla (sin origen)
        self._anchor: Optional[Tuple[float, float]] = None
        self._dwell_start = 0.0
        self._hold_start = 0.0
        self._ignore_until = -1e18
        self._freeze_until = -1e18
        self._prev_closed = False
        self._last_ts: Optional[float] = None
        self._frozen = False
        self._face_found = False

    # ------------------------------------------------------------- configuración
    def _cfg(self, key):
        value = self.config_getter().get(key)
        return DEFAULT_CONFIG["interaction"][key] if value is None else value

    def smoothing_alpha(self) -> float:
        return 1.0 - 0.95 * float(self._cfg("smoothing_percent")) / 100.0

    # ----------------------------------------------------------------- utilidades
    @property
    def holding(self) -> bool:
        return self.state == STATE_HOLD

    def _emit(self, name):
        if self.listener:
            x, y = self.screen_pos()
            self.listener(name, x, y)

    def screen_pos(self) -> Tuple[float, float]:
        if self.pos is None:
            return float(self.origin[0]), float(self.origin[1])
        return self.pos[0] + self.origin[0], self.pos[1] + self.origin[1]

    def snapshot(self, ts: Optional[float] = None) -> CursorSnapshot:
        elapsed = 0.0
        if self.state == STATE_DWELL and ts is not None:
            elapsed = max(0.0, (ts - self._dwell_start) * 1000.0)
        x, y = self.screen_pos()
        return CursorSnapshot(
            state=self.state,
            x=x,
            y=y,
            dwell_elapsed_ms=elapsed,
            dwell_total_ms=float(self._cfg("dwell_hold_ms")),
            frozen=self._frozen,
            paused=self.paused,
            face_found=self._face_found,
            positioned=self.pos is not None,
        )

    # ---------------------------------------------------------------------- pausa
    def set_paused(self, paused: bool):
        """Pausar libera cualquier hold e ignora todo; reanudar vuelve a IDLE."""
        if paused == self.paused:
            return
        if paused:
            self._release("pause")
            self.paused = True
            self._emit("paused")
        else:
            self.paused = False
            self.state = STATE_IDLE
            self._anchor = None
            self._last_ts = None
            self._prev_closed = False
            self._freeze_until = -1e18
            self._emit("resumed")

    def toggle_pause(self) -> bool:
        self.set_paused(not self.paused)
        return self.paused

    def release_all(self):
        """Suelta el botón si estaba presionado (idempotente). Usar al cerrar, fallar o salir."""
        self._release("close")

    def _release(self, reason: str, ts: Optional[float] = None):
        was_holding = self.state == STATE_HOLD
        self.state = STATE_IDLE
        self._anchor = None
        if was_holding:
            try:
                self.mouse.up()
            finally:
                if ts is not None and reason == "blink":
                    self._ignore_until = ts + POST_ACTION_IGNORE_S
            self._emit("hold_end")

    # ---------------------------------------------------------------------- tick
    def tick(self, ts: float):
        """Revisión por tiempo sin cuadro nuevo (auto-liberación del hold)."""
        if not self.paused:
            self._check_hold_timeout(ts)

    def _check_hold_timeout(self, ts: float):
        if self.state == STATE_HOLD and ts - self._hold_start >= float(self._cfg("hold_timeout_s")):
            self._release("timeout", ts)
            self._emit("hold_timeout")

    # ------------------------------------------------------------------- cuadro
    def update(self, frame) -> CursorSnapshot:
        """Procesa un TrackingFrame y devuelve el estado resultante."""
        ts = float(frame.timestamp)
        if self.paused:
            return self.snapshot(ts)
        self._check_hold_timeout(ts)

        dt = 0.0 if self._last_ts is None else max(0.0, ts - self._last_ts)
        self._last_ts = ts
        face = bool(frame.face_found and frame.features is not None)
        closed = bool(frame.eyes_closed) and face
        self._face_found = face
        if self._prev_closed and not closed:
            self._freeze_until = ts + float(self._cfg("freeze_after_blink_ms")) / 1000.0
        self._prev_closed = closed
        frozen = (not face) or closed or ts < self._freeze_until
        self._frozen = frozen

        # 1) parpadeo: ocurre en la posición congelada, antes de mover el cursor
        event = frame.blink_event
        if event is not None and face:
            self._handle_blink(event, ts)

        # 2) movimiento
        if not frozen:
            self._move(frame.features, dt)

        # 3) dwell
        self._update_dwell(ts, dt, frozen)
        return self.snapshot(ts)

    def _move(self, features, dt: float):
        try:
            gx, gy = self.mapper.predict_one(features)
        except Exception:  # noqa: BLE001 - un cuadro corrupto no debe tumbar el control
            return
        if not (math.isfinite(gx) and math.isfinite(gy)):
            return
        gx = min(max(gx, 0.0), self.width - 1.0)
        gy = min(max(gy, 0.0), self.height - 1.0)
        if self.pos is None:
            new = (gx, gy)
        else:
            alpha = self.smoothing_alpha()
            if alpha >= 1.0:
                a = 1.0
            else:
                a = 1.0 - (1.0 - alpha) ** (min(dt, MAX_SMOOTHING_DT_S) * REFERENCE_FPS)
            new = (self.pos[0] + a * (gx - self.pos[0]), self.pos[1] + a * (gy - self.pos[1]))
        moved = self.pos is None or abs(new[0] - self.pos[0]) >= 0.5 or abs(new[1] - self.pos[1]) >= 0.5
        self.pos = new
        if moved:
            self.mouse.move(new[0] + self.origin[0], new[1] + self.origin[1])

    def _handle_blink(self, event, ts: float):
        if ts < self._ignore_until:
            return
        duration = float(event.duration_ms)
        if self.state == STATE_HOLD:
            if duration >= float(self._cfg("blink_release_ms")):
                self._release("blink", ts)
            return
        if float(self._cfg("blink_click_ms")) <= duration <= float(self._cfg("blink_max_ms")):
            self.mouse.click()
            self._ignore_until = ts + POST_ACTION_IGNORE_S
            self._anchor = None  # el clic reinicia el dwell
            self._emit("click")

    def _update_dwell(self, ts: float, dt: float, frozen: bool):
        if self.state == STATE_HOLD or self.pos is None:
            return
        if frozen:
            if self._anchor is not None:
                self._dwell_start += dt  # el reloj del dwell se detiene mientras no hay mirada fiable
            return
        radius = float(self._cfg("dwell_radius_px"))
        if self._anchor is None or math.hypot(self.pos[0] - self._anchor[0], self.pos[1] - self._anchor[1]) > radius:
            self._anchor = self.pos
            self._dwell_start = ts
            self.state = STATE_IDLE
            return
        self.state = STATE_DWELL
        if self._cfg("hold_enabled") and (ts - self._dwell_start) * 1000.0 >= float(self._cfg("dwell_hold_ms")):
            self.mouse.down()
            self.state = STATE_HOLD
            self._hold_start = ts
            self._emit("hold_start")
