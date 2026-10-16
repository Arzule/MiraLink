"""Sesión de control del cursor: une tracker, CursorController, hotkey de emergencia y retroalimentación.

Todas las piezas con hardware (hilo de cámara, mouse, hotkey, ventana de feedback) se crean con
fábricas inyectables para poder probar la orquestación sin hardware.

"Recentrar" (botón o Ctrl+Alt+R) reutiliza el mismo tracker: mientras la pantalla del punto central
está abierta, los cuadros van a ella y el cursor no se mueve.
"""

from __future__ import annotations

import atexit
import logging
import time
from typing import Callable, Optional, Tuple

import numpy as np
from PySide6.QtCore import QObject, QTimer, Signal
from PySide6.QtGui import QGuiApplication

from calibration import (
    CalibrationError,
    RecenterSession,
    default_calibration_path,
    load_calibration,
    save_offset,
    screen_geometry_from_qt,
)
from cursor_control import (
    HOTKEY_LABEL,
    RECENTER_HOTKEY_LABEL,
    CursorController,
    DryRunMouse,
    PyAutoGUIMouse,
    Win32Hotkey,
    _WinHotkeyApi,
)

logger = logging.getLogger(__name__)

MSG_NEEDS_CALIBRATION = "Primero calibra el rastreador"
MSG_RECENTER_NEEDS_CONTROL = f"Inicia el control del cursor para usar Recentrar ({RECENTER_HOTKEY_LABEL})."


def default_recenter_hotkey(callback):
    return Win32Hotkey(
        callback, vk=_WinHotkeyApi.VK_R, label=RECENTER_HOTKEY_LABEL, hotkey_id=_WinHotkeyApi.RECENTER_HOTKEY_ID
    )


def default_screen_size() -> Tuple[int, int]:
    import pyautogui

    size = pyautogui.size()
    return int(size[0]), int(size[1])


class ControlSession(QObject):
    paused_changed = Signal(bool)  # True = control en pausa
    tracker_active_changed = Signal(bool)
    event_occurred = Signal(str)  # click, hold_start, hold_end, ...
    failed = Signal(str)
    recentered = Signal(str)  # resultado de "Recentrar" (aplicado o motivo del fallo)
    _hotkey_pressed = Signal()  # el hilo del atajo emite; se atiende en el hilo de la GUI
    _recenter_pressed = Signal()

    def __init__(
        self,
        config_getter: Callable[[], dict],
        dry_run: bool = False,
        screen=None,
        calibration_path: Optional[str] = None,
        camera_index_getter: Callable[[], int] = lambda: 0,
        tracker_factory: Optional[Callable[[int], object]] = None,
        mouse_factory: Optional[Callable[[], object]] = None,
        hotkey_factory: Optional[Callable[[Callable[[], None]], object]] = None,
        feedback_factory: Optional[Callable[..., object]] = None,
        screen_size_fn: Callable[[], Tuple[int, int]] = default_screen_size,
        clock: Callable[[], float] = time.monotonic,
        parent=None,
        recenter_hotkey_factory: Optional[Callable[[Callable[[], None]], object]] = None,
        recenter_screen_factory: Optional[Callable[..., object]] = None,
        camera_size_getter: Callable[[], Tuple[int, int]] = lambda: (640, 480),
    ):
        super().__init__(parent)
        self.config_getter = config_getter
        self.dry_run = dry_run
        self._screen = screen
        self.calibration_path = calibration_path or default_calibration_path()
        self.camera_index_getter = camera_index_getter
        self._tracker_factory = tracker_factory
        self._mouse_factory = mouse_factory
        self._hotkey_factory = hotkey_factory
        self._feedback_factory = feedback_factory
        self.screen_size_fn = screen_size_fn
        self.clock = clock
        self.camera_size_getter = camera_size_getter
        # Ctrl+Alt+R real solo si el atajo de pausa también es el real (los tests nunca registran el real)
        if recenter_hotkey_factory is None and hotkey_factory is None:
            recenter_hotkey_factory = default_recenter_hotkey
        self._recenter_hotkey_factory = recenter_hotkey_factory
        self._recenter_screen_factory = recenter_screen_factory
        self.tracker = None
        self.controller: Optional[CursorController] = None
        self.hotkey = None
        self.recenter_hotkey = None
        self.recenter_screen = None
        self.feedback = None
        self.mouse = None
        self.running = False
        self.hotkey_warning = ""
        self._atexit_registered = False
        self._timer = QTimer(self)
        self._timer.setInterval(100)
        self._timer.timeout.connect(self._on_timer)
        self._hotkey_pressed.connect(self.toggle_pause)
        self._recenter_pressed.connect(self._on_recenter_hotkey)

    # ----------------------------------------------------------- comprobaciones
    def _target_screen(self):
        return self._screen or QGuiApplication.primaryScreen()

    def check_can_start(self):
        """(ok, mensaje, mapper, geometría). Valida calibración y escala de pantalla."""
        screen = self._target_screen()
        geom = screen_geometry_from_qt(screen)
        try:
            mapper = load_calibration(self.calibration_path, geom)
        except CalibrationError as e:
            if "No hay calibración" in str(e):
                return False, MSG_NEEDS_CALIBRATION, None, geom
            return False, f"La calibración no es válida ({e}). Calibra el rastreador de nuevo.", None, geom
        if not self.dry_run:
            try:
                sw, sh = self.screen_size_fn()
            except Exception as e:  # noqa: BLE001
                return False, f"No se pudo leer la resolución del sistema ({e}). El control real no se inicia.", None, geom
            if (sw, sh) != (geom["width"], geom["height"]):
                return (
                    False,
                    f"La resolución del sistema ({sw}x{sh}) no coincide con la pantalla física que usa la "
                    f"calibración ({geom['width']}x{geom['height']}); por seguridad no se inicia el control real. "
                    "Puedes probar con --dry-run.",
                    None,
                    geom,
                )
        return True, "", mapper, geom

    # ----------------------------------------------------------------- arranque
    def start(self) -> Tuple[bool, str]:
        """Inicia tracker y control. Devuelve (ok, mensaje)."""
        if self.running:
            return True, ""
        ok, msg, mapper, geom = self.check_can_start()
        if not ok:
            return False, msg
        screen = self._target_screen()

        try:
            hotkey = (self._hotkey_factory or (lambda cb: Win32Hotkey(cb)))(self._hotkey_pressed.emit)
            registered = hotkey.start()
            self.hotkey_warning = ""
            if not registered:
                reason = getattr(hotkey, "error", "") or f"No se pudo registrar el atajo {HOTKEY_LABEL}."
                if not self.dry_run:
                    hotkey.stop()
                    return False, f"{reason} Sin el atajo de emergencia no se permite el control real; usa --dry-run."
                self.hotkey_warning = f"{reason} (solo simulación)"
            self.hotkey = hotkey
            self._start_recenter_hotkey()

            self.mouse = (self._mouse_factory or (DryRunMouse if self.dry_run else PyAutoGUIMouse))()
            if self._feedback_factory is None:
                from cursor_feedback import CursorFeedback

                self._feedback_factory = CursorFeedback
            self.feedback = self._feedback_factory(dpr=float(screen.devicePixelRatio()), dry_run=self.dry_run)
            self.controller = CursorController(
                mapper,
                self.mouse,
                geom["width"],
                geom["height"],
                origin=(geom["x"], geom["y"]),
                config_getter=self.config_getter,
                listener=self._on_controller_event,
            )
            if not self._atexit_registered:
                atexit.register(self._atexit_release)
                self._atexit_registered = True

            factory = self._tracker_factory
            if factory is None:
                from eye_tracker import TrackerThread

                size = self.camera_size_getter()
                factory = lambda i: TrackerThread(i, camera_size=size)  # noqa: E731
            self.tracker = factory(self.camera_index_getter())
            self.tracker.frame_ready.connect(self._on_frame)
            self.tracker.failed.connect(self._on_tracker_failed)
            self.running = True
            self.tracker.start()
            self.feedback.show()
            self._timer.start()
            self.tracker_active_changed.emit(True)
            self.paused_changed.emit(False)
            return True, self.hotkey_warning
        except Exception as e:  # noqa: BLE001 - no dejar piezas a medio iniciar
            self.stop()
            return False, f"No se pudo iniciar el control del cursor: {e}"

    def _start_recenter_hotkey(self):
        """Ctrl+Alt+R no es de seguridad: si no se puede registrar, se avisa y queda el botón."""
        if self._recenter_hotkey_factory is None:
            return
        hk = self._recenter_hotkey_factory(self._recenter_pressed.emit)
        if hk.start():
            self.recenter_hotkey = hk
            return
        hk.stop()
        reason = getattr(hk, "error", "") or f"No se pudo registrar el atajo {RECENTER_HOTKEY_LABEL}."
        note = f"{reason} Recentrar sigue disponible con el botón."
        self.hotkey_warning = f"{self.hotkey_warning} {note}".strip()

    # ---------------------------------------------------------------- recentrar
    def _on_recenter_hotkey(self):
        ok, message = self.recenter()
        if not ok and message:
            self.recentered.emit(message)

    def recenter(self) -> Tuple[bool, str]:
        """Muestra un punto central, recoge ~1 s de muestras con el mismo tracker y aplica un offset."""
        if not self.running or self.controller is None:
            return False, MSG_RECENTER_NEEDS_CONTROL
        if self.recenter_screen is not None:
            self.recenter_screen.raise_()
            return True, ""
        screen = self._target_screen()
        geom = screen_geometry_from_qt(screen)
        self.controller.release_all()  # un hold activo se suelta antes de recentrar
        factory = self._recenter_screen_factory
        if factory is None:
            from calibration_ui import RecenterScreen

            factory = RecenterScreen
        scr = factory(RecenterSession(geom["width"], geom["height"]), self.controller.mapper, clock=self.clock)
        scr.completed.connect(self._on_recenter_completed)
        scr.aborted.connect(self._on_recenter_aborted)
        self.recenter_screen = scr
        if self.feedback is not None:
            self.feedback.hide()
        scr.setGeometry(screen.geometry())
        scr.showFullScreen()
        scr.raise_()
        scr.activateWindow()
        scr.start()
        logger.info("Recentrar: pantalla mostrada")
        return True, ""

    def _close_recenter_screen(self):
        scr, self.recenter_screen = self.recenter_screen, None
        if scr is not None:
            scr.stop()
            scr.hide()
            scr.deleteLater()
        if self.running and self.feedback is not None:
            self.feedback.show()

    def _on_recenter_completed(self, offset):
        self._close_recenter_screen()
        if self.controller is None:
            return
        self.controller.mapper.offset = np.asarray(offset, dtype=float)
        message = f"Recentrado: desplazamiento ({offset[0]:+.0f}, {offset[1]:+.0f}) px"
        try:
            save_offset(self.calibration_path, offset)
            message += " guardado en la calibración."
        except (CalibrationError, OSError) as e:
            message += f" aplicado, pero no se pudo guardar: {e}"
        logger.info(message)
        self.recentered.emit(message)

    def _on_recenter_aborted(self, message: str):
        self._close_recenter_screen()
        logger.warning("Recentrar: %s", message)
        self.recentered.emit(message)

    # ------------------------------------------------------------------ eventos
    def _on_frame(self, frame):
        if not self.running or self.controller is None:
            return
        if self.recenter_screen is not None:
            self.recenter_screen.on_frame(frame)  # mientras se recentra, el cursor no se mueve
            return
        try:
            snap = self.controller.update(frame)
        except Exception as e:  # noqa: BLE001 - ante cualquier fallo, soltar y detener
            self._fail(f"Error en el control del cursor: {e}")
            return
        if self.feedback is not None:
            self.feedback.update_snapshot(snap)

    def _on_controller_event(self, name: str, x: float, y: float):
        if self.feedback is not None:
            self.feedback.notify(name)
        self.event_occurred.emit(name)

    def _on_timer(self):
        if self.controller is not None:
            self.controller.tick(self.clock())

    def _on_tracker_failed(self, message: str):
        self._fail(f"Se detuvo el seguimiento: {message}")

    def _fail(self, message: str):
        self.stop()
        self.failed.emit(message)

    # ------------------------------------------------------------------- pausa
    @property
    def paused(self) -> bool:
        return bool(self.controller and self.controller.paused)

    def set_paused(self, paused: bool):
        if self.controller is None:
            return
        self.controller.set_paused(paused)
        if self.feedback is not None:
            self.feedback.update()
        self.paused_changed.emit(self.controller.paused)

    def toggle_pause(self):
        """Alterna la pausa (también lo llama el atajo global Ctrl+Alt+P)."""
        if self.controller is not None:
            self.set_paused(not self.controller.paused)

    # ------------------------------------------------------------------- cierre
    def _atexit_release(self):
        try:
            if self.controller is not None:
                self.controller.release_all()
        except Exception:  # noqa: BLE001
            pass

    def stop(self):
        """Detiene todo y suelta siempre el botón del mouse. Idempotente."""
        was_running = self.running
        self.running = False
        self._timer.stop()
        self._close_recenter_screen()
        try:
            if self.controller is not None:
                self.controller.release_all()
        finally:
            tracker, self.tracker = self.tracker, None
            if tracker is not None:
                try:
                    tracker.frame_ready.disconnect(self._on_frame)
                    tracker.failed.disconnect(self._on_tracker_failed)
                except (RuntimeError, TypeError):
                    pass
                tracker.stop()
            hotkey, self.hotkey = self.hotkey, None
            if hotkey is not None:
                hotkey.stop()
            recenter_hotkey, self.recenter_hotkey = self.recenter_hotkey, None
            if recenter_hotkey is not None:
                recenter_hotkey.stop()
            safe = getattr(self.mouse, "safe_release", None)
            if safe:
                safe()
            feedback, self.feedback = self.feedback, None
            if feedback is not None:
                feedback.hide()
                feedback.deleteLater()
        if was_running:
            self.tracker_active_changed.emit(False)
