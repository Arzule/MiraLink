"""Botones flotantes para MiraLink (teclado, zoom y filtros de color).

Barra vertical de botones grandes pegada a un lado de la pantalla, siempre al frente. Solo existe
mientras el control del cursor está activo (la crea y la muestra ControlSession; en pausa se oculta).

- No roba el foco (Qt.WindowDoesNotAcceptFocus = WS_EX_NOACTIVATE en Windows y botones sin foco): el
  teclado en pantalla escribe en la aplicación que tenía el foco y los atajos de la Lupa llegan al sistema.
- Tamaño: el error residual de la mirada es ~2° (~100 px), así que los botones miden por defecto 180 px
  lógicos (overlay.button_size); además el radio magnético (interaction.snap_radius_px) acerca el cursor
  al centro del botón cuando pasa cerca (`target_rects`, en píxeles físicos de la pantalla).
- Se pulsan como cualquier botón: parpadeo intencional (clic) o dwell + parpadeo (hold y soltar).
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

from PySide6.QtCore import QRect, Qt, Signal
from PySide6.QtWidgets import QPushButton, QVBoxLayout, QWidget

# (acción de windows_integration.ACTIONS, texto del botón, nombre accesible)
OVERLAY_BUTTONS = (
    ("zoom_in", "Acercar\n(Lupa)", "Acercar con la Lupa"),
    ("zoom_out", "Alejar\n(Lupa)", "Alejar con la Lupa"),
    ("keyboard", "Teclado", "Abrir o cerrar el teclado en pantalla"),
    ("color_filter", "Filtro\nde color", "Activar o desactivar el filtro de color"),
)
POSITIONS = ("right", "left")
# Con una aplicación delante la barra se oculta (taparía su barra de desplazamiento) y aparece solo este
# botón, separado del borde lo bastante para dejar libre la barra de desplazamiento de la ventana.
DESKTOP_BUTTON = (("show_desktop", "Ir al\nescritorio", "Ir al escritorio"),)
DESKTOP_BUTTON_GAP = 40
EDGE_GAP = 12  # px lógicos entre la barra y el borde de la pantalla
SPACING = 16

STYLE = """
QWidget#overlayBar { background: transparent; }
QPushButton#overlayButton {
    background-color: rgba(24, 24, 41, 230);
    color: #F3F3FA;
    border: 2px solid #2E2E4A;
    border-radius: 20px;
    font-family: "Inter", "Segoe UI";
    font-weight: 600;
    font-size: %(font)dpt;
}
QPushButton#overlayButton:hover { border: 3px solid #8070FF; background-color: rgba(48, 40, 96, 240); }
QPushButton#overlayButton:pressed { background-color: #8070FF; }
"""


class OverlayBar(QWidget):
    """Barra de botones flotantes. Emite `action_triggered(acción)` al pulsar un botón."""

    action_triggered = Signal(str)

    def __init__(
        self,
        screen=None,
        button_size: int = 180,
        position: str = "right",
        parent=None,
        buttons=OVERLAY_BUTTONS,
        edge_gap: int = EDGE_GAP,
    ):
        super().__init__(parent)
        self.edge_gap = int(edge_gap)
        if position not in POSITIONS:
            raise ValueError(f"posición desconocida: {position}")
        self.position = position
        self.button_size = int(button_size)
        self._screen = screen
        self.setObjectName("overlayBar")
        self.setWindowTitle("Botones de MiraLink")
        self.setAccessibleName("Botones flotantes de MiraLink")
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool | Qt.WindowDoesNotAcceptFocus)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)
        self.setFocusPolicy(Qt.NoFocus)
        self.scale = 1.0  # 1 / zoom de la lupa de MiraLink: con zoom la barra se reduce para verse igual
        self._view: Optional[Tuple[float, float, float, float]] = None  # zona visible (px físicos) o None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(SPACING)
        self.buttons = {}
        for action, text, name in buttons:
            btn = QPushButton(text, self)
            btn.setObjectName("overlayButton")
            btn.setAccessibleName(name)
            btn.setFocusPolicy(Qt.NoFocus)
            btn.clicked.connect(lambda _checked=False, a=action: self.action_triggered.emit(a))
            layout.addWidget(btn)
            self.buttons[action] = btn
        self._apply_scale()

    def screen_target(self):
        if self._screen is not None:
            return self._screen
        from PySide6.QtGui import QGuiApplication

        return QGuiApplication.primaryScreen()

    def _apply_scale(self):
        size = max(8, int(round(self.button_size * self.scale)))
        self.setStyleSheet(STYLE % {"font": max(1, round(size / 9))})
        self.layout().setSpacing(max(1, int(round(SPACING * self.scale))))
        for btn in self.buttons.values():
            btn.setFixedSize(size, size)
        self.adjustSize()

    def set_view(self, view_rect: Optional[Tuple[float, float, float, float]], zoom: float = 1.0):
        """Zona visible de la lupa de MiraLink (px físicos relativos a la pantalla; None = sin zoom): la barra
        se coloca dentro de ella y se reduce 1/zoom, así se ve en el mismo sitio y del mismo tamaño."""
        self._view = None if view_rect is None or zoom <= 1.0 else tuple(float(v) for v in view_rect)
        scale = 1.0 if self._view is None else 1.0 / float(zoom)
        if scale != self.scale:
            self.scale = scale
            self._apply_scale()
        self.place()

    def place(self):
        """Pega la barra al borde elegido de la zona visible, centrada en vertical (coordenadas lógicas)."""
        screen = self.screen_target()
        area = screen.geometry()
        left, top, width, height = float(area.left()), float(area.top()), float(area.width()), float(area.height())
        if self._view is not None:
            dpr = float(screen.devicePixelRatio()) or 1.0
            vx, vy, vw, vh = self._view
            left, top, width, height = left + vx / dpr, top + vy / dpr, vw / dpr, vh / dpr
        self.adjustSize()
        gap = self.edge_gap * self.scale
        x = left + width - self.width() - gap if self.position == "right" else left + gap
        y = top + (height - self.height()) / 2.0
        self.move(int(round(x)), int(round(y)))

    def showEvent(self, event):
        self.place()
        super().showEvent(event)

    def target_rects(self) -> List[Tuple[float, float, float, float]]:
        """Rectángulos de los botones en píxeles FÍSICOS relativos a la pantalla (los del CursorController).

        Vacío si la barra no está visible (sin botones a la vista no hay atracción)."""
        if not self.isVisible():
            return []
        screen = self.screen_target()
        dpr = float(screen.devicePixelRatio()) or 1.0
        origin = screen.geometry().topLeft()
        rects = []
        for btn in self.buttons.values():
            top_left = btn.mapToGlobal(btn.rect().topLeft())
            r = QRect(top_left, btn.size())
            rects.append(
                ((r.x() - origin.x()) * dpr, (r.y() - origin.y()) * dpr, r.width() * dpr, r.height() * dpr)
            )
        return rects


def make_overlay(screen=None, config: Optional[dict] = None, on_action: Optional[Callable[[str], None]] = None) -> OverlayBar:
    """Crea la barra con la configuración de config.json (sección overlay)."""
    cfg = config or {}
    bar = OverlayBar(screen, button_size=int(cfg.get("button_size", 180)), position=cfg.get("position", "right"))
    if on_action is not None:
        bar.action_triggered.connect(on_action)
    return bar


def make_desktop_button(screen=None, config: Optional[dict] = None, on_action: Optional[Callable[[str], None]] = None) -> OverlayBar:
    """Botón "Ir al escritorio" que sustituye a la barra cuando hay una aplicación delante."""
    cfg = config or {}
    bar = OverlayBar(
        screen,
        button_size=int(cfg.get("button_size", 180)),
        position=cfg.get("position", "right"),
        buttons=DESKTOP_BUTTON,
        edge_gap=DESKTOP_BUTTON_GAP,
    )
    bar.setAccessibleName("Botón para ir al escritorio")
    if on_action is not None:
        bar.action_triggered.connect(on_action)
    return bar
