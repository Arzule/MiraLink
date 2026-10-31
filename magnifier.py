"""Lupa propia de MiraLink: zoom a pantalla completa con la API de ampliación de Windows.

Con la Lupa de Windows (Win + "+") el cursor iba al punto de la pantalla sin ampliar y no a lo que el
usuario ve: la Lupa no publica su zoom ni qué zona muestra (no está en el registro). MiraLink hace aquí
el zoom él mismo (MagSetFullscreenTransform), así que siempre sabe qué zona del escritorio se ve:

- Lo que se ve en el punto P del monitor es el punto D = desplazamiento + P / zoom del escritorio.
  La mirada (calibrada en coordenadas del monitor) da P; el cursor se mueve a D (`ZoomedMouse`) y el
  clic cae justo donde se mira. Todas las coordenadas son píxeles físicos relativos a la pantalla.
- Mirar cerca de un borde de la vista ampliada (PAN_MARGIN) durante PAN_DELAY_S la desplaza hacia ese
  lado, más rápido cuanto más cerca del borde (hasta PAN_SPEED_PX_S px de pantalla por segundo).
- Los cambios de zoom se anclan al centro de la vista (no al botón pulsado, que está en un borde).

`ZoomView` es pura (sin Windows ni reloj: recibe dt). `MagnificationApi` es el backend real (ctypes,
Magnification.dll; no necesita permisos de administrador) y `FakeMagnification` el de tests y --dry-run.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

ZOOM_LEVELS = (1.0, 1.5, 2.0, 3.0)
PAN_MARGIN = 0.12  # fracción del ancho/alto de la pantalla junto a cada borde que desplaza la vista
PAN_SPEED_PX_S = 900.0  # velocidad máxima (px de pantalla por segundo) mirando justo al borde
PAN_DELAY_S = 0.4  # tiempo mirando la franja del borde antes de desplazar (un vistazo no mueve la vista)

Rect = Tuple[float, float, float, float]


class ZoomView:
    """Zoom y zona visible del escritorio (pura). `offset` = esquina superior izquierda de la zona visible."""

    def __init__(self, width: int, height: int, levels: Sequence[float] = ZOOM_LEVELS):
        self.width = float(width)
        self.height = float(height)
        self.levels = tuple(float(v) for v in levels)
        self.level = 0
        self.ox = 0.0
        self.oy = 0.0
        self._edge_time = 0.0

    @property
    def zoom(self) -> float:
        return self.levels[self.level]

    @property
    def active(self) -> bool:
        return self.zoom > 1.0

    @property
    def view_size(self) -> Tuple[float, float]:
        return self.width / self.zoom, self.height / self.zoom

    def view_rect(self) -> Rect:
        """Zona visible del escritorio (x, y, ancho, alto)."""
        w, h = self.view_size
        return self.ox, self.oy, w, h

    def to_desktop(self, x: float, y: float) -> Tuple[float, float]:
        """Punto de la pantalla (lo que se ve) -> punto del escritorio que se muestra ahí."""
        return self.ox + x / self.zoom, self.oy + y / self.zoom

    def to_display(self, x: float, y: float) -> Tuple[float, float]:
        """Punto del escritorio -> dónde se ve en la pantalla."""
        return (x - self.ox) * self.zoom, (y - self.oy) * self.zoom

    def rect_to_display(self, rect: Rect) -> Rect:
        x, y = self.to_display(rect[0], rect[1])
        return x, y, rect[2] * self.zoom, rect[3] * self.zoom

    def _clamp(self):
        w, h = self.view_size
        self.ox = min(max(self.ox, 0.0), self.width - w)
        self.oy = min(max(self.oy, 0.0), self.height - h)

    def set_level(self, level: int) -> bool:
        """Cambia el zoom manteniendo el centro de la vista. True si cambió."""
        level = min(max(int(level), 0), len(self.levels) - 1)
        if level == self.level:
            return False
        cx, cy = self.to_desktop(self.width / 2.0, self.height / 2.0)
        self.level = level
        w, h = self.view_size
        self.ox, self.oy = cx - w / 2.0, cy - h / 2.0
        self._clamp()
        self._edge_time = 0.0
        return True

    def zoom_in(self) -> bool:
        return self.set_level(self.level + 1)

    def zoom_out(self) -> bool:
        return self.set_level(self.level - 1)

    def reset(self) -> bool:
        changed = self.level != 0
        self.level, self.ox, self.oy, self._edge_time = 0, 0.0, 0.0, 0.0
        return changed

    def pan_velocity(self, x: float, y: float) -> Tuple[float, float]:
        """Velocidad (px de pantalla/s) por mirar en (x, y): 0 fuera de las franjas de los bordes."""
        def axis(v, size):
            m = PAN_MARGIN * size
            if v < m:
                return -PAN_SPEED_PX_S * (1.0 - max(v, 0.0) / m)
            if v > size - m:
                return PAN_SPEED_PX_S * (1.0 - max(size - v, 0.0) / m)
            return 0.0

        return axis(x, self.width), axis(y, self.height)

    def update_pan(self, gaze: Optional[Tuple[float, float]], dt: float, blocked: bool = False) -> bool:
        """Desplaza la vista si la mirada (en la pantalla) lleva PAN_DELAY_S en una franja del borde.

        `blocked` (p. ej. la mirada está sobre un botón flotante) o sin mirada: no desplaza y reinicia la
        espera. True si la zona visible cambió."""
        if not self.active or gaze is None or blocked or dt <= 0.0:
            self._edge_time = 0.0
            return False
        vx, vy = self.pan_velocity(*gaze)
        if vx == 0.0 and vy == 0.0:
            self._edge_time = 0.0
            return False
        self._edge_time += dt
        if self._edge_time < PAN_DELAY_S:
            return False
        before = (self.ox, self.oy)
        self.ox += vx * dt / self.zoom  # px de pantalla -> px del escritorio
        self.oy += vy * dt / self.zoom
        self._clamp()
        return (self.ox, self.oy) != before


# ------------------------------------------------------------------ backends
class MagnificationApi:
    """Zoom real a pantalla completa (Magnification.dll). Siempre devolver a 1x con `reset`/`close`."""

    def __init__(self):
        import ctypes

        self._ctypes = ctypes
        self._dll = ctypes.WinDLL("Magnification.dll")
        self._dll.MagInitialize.restype = ctypes.c_bool
        self._dll.MagUninitialize.restype = ctypes.c_bool
        self._dll.MagSetFullscreenTransform.restype = ctypes.c_bool
        self._dll.MagSetFullscreenTransform.argtypes = [ctypes.c_float, ctypes.c_int, ctypes.c_int]
        self._initialized = False

    def apply(self, zoom: float, ox: float, oy: float) -> bool:
        if not self._initialized:
            if not self._dll.MagInitialize():
                raise RuntimeError("No se pudo iniciar la ampliación de Windows (MagInitialize).")
            self._initialized = True
        ok = bool(self._dll.MagSetFullscreenTransform(float(zoom), int(round(ox)), int(round(oy))))
        if not ok:
            raise RuntimeError("Windows rechazó el zoom (MagSetFullscreenTransform).")
        return ok

    def close(self):
        if self._initialized:
            try:
                self._dll.MagSetFullscreenTransform(1.0, 0, 0)
            finally:
                self._dll.MagUninitialize()
                self._initialized = False


class FakeMagnification:
    """Registra las transformaciones sin tocar la pantalla (tests y --dry-run)."""

    def __init__(self):
        self.calls: List[Tuple[float, int, int]] = []
        self.closed = False

    def apply(self, zoom: float, ox: float, oy: float) -> bool:
        self.calls.append((float(zoom), int(round(ox)), int(round(oy))))
        return True

    def close(self):
        self.closed = True


class ZoomedMouse:
    """Envuelve el MouseBackend: la posición que da el CursorController (lo que se ve) pasa al escritorio.

    `origin` es el origen físico de la pantalla (el CursorController suma su origen al mover)."""

    def __init__(self, mouse, view: ZoomView, origin: Tuple[float, float] = (0.0, 0.0)):
        self.inner = mouse
        self.view = view
        self.origin = origin
        self.last: Optional[Tuple[float, float]] = None  # última posición pedida (en la pantalla)

    def move(self, x: float, y: float) -> None:
        self.last = (float(x), float(y))
        self.inner.move(*self.desktop_point(x, y))

    def desktop_point(self, x: float, y: float) -> Tuple[float, float]:
        dx, dy = self.view.to_desktop(x - self.origin[0], y - self.origin[1])
        return dx + self.origin[0], dy + self.origin[1]

    def refresh(self) -> None:
        """Tras cambiar el zoom o desplazar la vista, vuelve a poner el cursor bajo la mirada."""
        if self.last is not None:
            self.inner.move(*self.desktop_point(*self.last))

    def down(self) -> None:
        self.inner.down()

    def up(self) -> None:
        self.inner.up()

    def click(self) -> None:
        self.inner.click()

    def safe_release(self) -> None:
        safe = getattr(self.inner, "safe_release", None)
        if safe is not None:
            safe()

    def __getattr__(self, name):  # events de DryRunMouse, etc.
        return getattr(self.inner, name)
