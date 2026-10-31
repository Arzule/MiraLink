"""Integración con Windows (osk.exe, atajos de la Lupa, win32gui, ms-settings).

Acciones de los botones flotantes (etapa 5). Todo lo que toca el sistema (teclas, procesos, registro,
ventanas) va detrás de funciones inyectables: los tests usan versiones falsas y nunca abren el teclado,
la Lupa ni Configuración.

- Teclado en pantalla: abre osk.exe con ShellExecute (os.startfile; CreateProcess falla porque osk.exe
  pide uiAccess) o lo cierra si ya está abierto (WM_CLOSE a su ventana "OSKMainClass").
- Lupa de Windows: Win + "+" del teclado numérico acerca (y la abre), Win + "-" aleja (VK_ADD /
  VK_SUBTRACT, sin depender de la distribución). Los botones "Acercar"/"Alejar" YA NO la usan: con ella el
  cursor iba a la pantalla sin ampliar; usan la lupa propia (magnifier.py). Se conservan para comparar.
- Filtros de color: Win + Ctrl + C los alterna, pero Windows solo atiende ese atajo si el usuario lo
  activó en Configuración (HKCU\\Software\\Microsoft\\ColorFiltering, HotkeyEnabled = 1). Si no, se abre
  ms-settings:easeofaccess-colorfilter para que lo active (no se cambia el registro por él).
"""

from __future__ import annotations

import logging
import os
from typing import Callable, Optional, Sequence

logger = logging.getLogger(__name__)

VK_LWIN = 0x5B
VK_CONTROL = 0x11
VK_ADD = 0x6B  # "+" del teclado numérico
VK_SUBTRACT = 0x6D  # "-" del teclado numérico
VK_C = 0x43
KEYEVENTF_KEYUP = 0x0002
WM_CLOSE = 0x0010

OSK_WINDOW_CLASS = "OSKMainClass"
COLOR_FILTER_KEY = r"Software\Microsoft\ColorFiltering"
COLOR_FILTER_SETTINGS = "ms-settings:easeofaccess-colorfilter"

ACTIONS = ("keyboard", "zoom_in", "zoom_out", "color_filter")


# ------------------------------------------------------------ backends reales
def send_key_combo(vks: Sequence[int]) -> None:
    """Pulsa las teclas en orden y las suelta en orden inverso (keybd_event de pywin32)."""
    import win32api

    for vk in vks:
        win32api.keybd_event(vk, 0, 0, 0)
    for vk in reversed(vks):
        win32api.keybd_event(vk, 0, KEYEVENTF_KEYUP, 0)


def shell_open(target: str) -> None:
    os.startfile(target)  # noqa: S606 - ShellExecute: necesario para osk.exe (uiAccess) y ms-settings:


def find_window(window_class: str) -> int:
    import win32gui

    return int(win32gui.FindWindow(window_class, None) or 0)


def post_close(hwnd: int) -> None:
    import win32gui

    win32gui.PostMessage(hwnd, WM_CLOSE, 0, 0)


def read_color_filter_hotkey() -> Optional[bool]:
    """True/False según HotkeyEnabled; None si la clave no existe (nunca se configuró)."""
    import winreg

    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, COLOR_FILTER_KEY) as key:
            value, _type = winreg.QueryValueEx(key, "HotkeyEnabled")
            return bool(value)
    except OSError:
        return None


def osk_path() -> str:
    root = os.environ.get("SystemRoot", r"C:\Windows")
    return os.path.join(root, "System32", "osk.exe")


# ------------------------------------------------------------------ acciones
class WindowsActions:
    """Acciones de los botones flotantes. Cada una devuelve un mensaje breve para la interfaz.

    Con `dry_run` solo registra lo que haría (como el mouse de --dry-run).
    """

    def __init__(
        self,
        dry_run: bool = False,
        send_keys: Callable[[Sequence[int]], None] = send_key_combo,
        open_target: Callable[[str], None] = shell_open,
        find_window_fn: Callable[[str], int] = find_window,
        close_window: Callable[[int], None] = post_close,
        color_hotkey_enabled: Callable[[], Optional[bool]] = read_color_filter_hotkey,
        log: Callable[[str], None] = logger.info,
    ):
        self.dry_run = dry_run
        self.send_keys = send_keys
        self.open_target = open_target
        self.find_window = find_window_fn
        self.close_window = close_window
        self.color_hotkey_enabled = color_hotkey_enabled
        self.log = log

    def run(self, action: str) -> str:
        """Ejecuta una acción de ACTIONS; los errores del sistema se devuelven como mensaje (no se lanzan)."""
        handler = {
            "keyboard": self.toggle_keyboard,
            "zoom_in": self.zoom_in,
            "zoom_out": self.zoom_out,
            "color_filter": self.toggle_color_filter,
        }.get(action)
        if handler is None:
            raise ValueError(f"acción desconocida: {action}")
        try:
            message = handler()
        except Exception as e:  # noqa: BLE001 - un fallo del sistema no debe tumbar el control
            logger.exception("Falló la acción %s", action)
            message = f"No se pudo completar la acción ({e})."
        self.log(f"Botón flotante {action}: {message}")
        return message

    def _keys(self, vks: Sequence[int], label: str) -> str:
        if self.dry_run:
            return f"(simulación) {label}"
        self.send_keys(vks)
        return label

    def zoom_in(self) -> str:
        return self._keys((VK_LWIN, VK_ADD), "Lupa: acercar")

    def zoom_out(self) -> str:
        return self._keys((VK_LWIN, VK_SUBTRACT), "Lupa: alejar")

    def toggle_keyboard(self) -> str:
        hwnd = self.find_window(OSK_WINDOW_CLASS)
        if hwnd:
            if not self.dry_run:
                self.close_window(hwnd)
            return ("(simulación) " if self.dry_run else "") + "Teclado en pantalla cerrado"
        if not self.dry_run:
            self.open_target(osk_path())
        return ("(simulación) " if self.dry_run else "") + "Teclado en pantalla abierto"

    def toggle_color_filter(self) -> str:
        if self.color_hotkey_enabled():
            return self._keys((VK_LWIN, VK_CONTROL, VK_C), "Filtro de color alternado")
        if not self.dry_run:
            self.open_target(COLOR_FILTER_SETTINGS)
        return (
            ("(simulación) " if self.dry_run else "")
            + "Activa \"Permitir que la tecla de método abreviado active o desactive el filtro\" en "
            "Configuración (se abrió) y vuelve a pulsar el botón."
        )
