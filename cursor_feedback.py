"""Retroalimentación visual junto al cursor: arco de dwell, anillo de HOLD y destello de clic.

Ventana sin marco, siempre al frente, transparente y que no intercepta el mouse
(Qt.WindowTransparentForInput). Funciona también en --dry-run: allí el cursor real no se mueve,
así que dibuja además una mira en la posición virtual y el último evento simulado.
"""

from __future__ import annotations

import time
from typing import Callable, Optional

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QWidget

from cursor_control import STATE_DWELL, STATE_HOLD, CursorSnapshot

ACCENT = QColor("#8070FF")
TEXT = QColor("#F3F3FA")
WARNING = QColor("#FFB020")
DRIFT_WARNING_TEXT = "Recentra: Ctrl+Alt+R"

QUIET_BEFORE_ARC_MS = 300.0  # el arco aparece tras 300 ms de quietud
FLASH_MS = 250.0
DRY_LABEL_S = 1.0
EVENT_LABELS = {
    "click": "Clic",
    "click_blocked": "Clic bloqueado (mirada abajo)",
    "hold_start": "Mantener",
    "hold_end": "Soltar",
    "hold_timeout": "Soltado (tiempo máximo)",
    "paused": "Pausado",
    "resumed": "Reanudado",
}


class CursorFeedback(QWidget):
    SIZE = 140

    def __init__(self, dpr: float = 1.0, dry_run: bool = False, clock: Callable[[], float] = time.monotonic, parent=None):
        super().__init__(parent)
        self.dpr = dpr if dpr > 0 else 1.0
        self.dry_run = dry_run
        self.clock = clock
        self.snapshot = CursorSnapshot()
        self._flash_start: Optional[float] = None
        self._label: Optional[str] = None
        self._label_start = 0.0
        self.warning: Optional[str] = None  # aviso persistente junto al cursor (deriva de la cabeza)
        self.setWindowFlags(
            Qt.FramelessWindowHint
            | Qt.WindowStaysOnTopHint
            | Qt.Tool
            | Qt.WindowTransparentForInput
            | Qt.WindowDoesNotAcceptFocus
        )
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setFocusPolicy(Qt.NoFocus)
        self.setFixedSize(self.SIZE, self.SIZE)
        self.setAccessibleName("Indicador visual del cursor")
        self._anim = QTimer(self)
        self._anim.setInterval(33)
        self._anim.timeout.connect(self._on_anim)

    # --------------------------------------------------------------- estado
    @property
    def dwell_progress(self) -> float:
        """0 hasta 300 ms de quietud; después avance 0..1 hacia dwell_hold_ms."""
        s = self.snapshot
        if s.state != STATE_DWELL or s.dwell_elapsed_ms < QUIET_BEFORE_ARC_MS or s.dwell_total_ms <= 0:
            return 0.0
        return float(min(1.0, s.dwell_elapsed_ms / s.dwell_total_ms))

    @property
    def holding(self) -> bool:
        return self.snapshot.state == STATE_HOLD and not self.snapshot.paused

    def flash_active(self, now: Optional[float] = None) -> bool:
        now = self.clock() if now is None else now
        return self._flash_start is not None and (now - self._flash_start) * 1000.0 < FLASH_MS

    def update_snapshot(self, snap: CursorSnapshot):
        self.snapshot = snap
        cx, cy = snap.x / self.dpr, snap.y / self.dpr
        self.move(int(round(cx - self.SIZE / 2)), int(round(cy - self.SIZE / 2)))
        self.update()

    def notify(self, event_name: str):
        """Evento del controlador: 'click' dispara el destello; en dry-run también una etiqueta."""
        now = self.clock()
        if event_name == "click":
            self._flash_start = now
        if self.dry_run and event_name in EVENT_LABELS:
            self._label = EVENT_LABELS[event_name]
            self._label_start = now
        if self._flash_start is not None or self._label is not None:
            self._anim.start()
        self.update()

    def set_warning(self, text: Optional[str]):
        """Muestra (o quita con None) un aviso corto y persistente en la parte superior del indicador."""
        self.warning = text or None
        self.update()

    def _on_anim(self):
        now = self.clock()
        if self._flash_start is not None and not self.flash_active(now):
            self._flash_start = None
        if self._label is not None and now - self._label_start > DRY_LABEL_S:
            self._label = None
        if self._flash_start is None and self._label is None:
            self._anim.stop()
        self.update()

    # ---------------------------------------------------------------- pintura
    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        c = QPointF(self.SIZE / 2, self.SIZE / 2)
        s = self.snapshot
        if s.paused:
            p.end()
            return
        if self.dry_run:  # mira en la posición virtual
            p.setPen(QPen(TEXT, 2))
            p.drawLine(QPointF(c.x() - 10, c.y()), QPointF(c.x() + 10, c.y()))
            p.drawLine(QPointF(c.x(), c.y() - 10), QPointF(c.x(), c.y() + 10))
        radius = 30.0
        ring = QRectF(c.x() - radius, c.y() - radius, 2 * radius, 2 * radius)
        if self.holding:
            fill = QColor(ACCENT)
            fill.setAlpha(110)
            p.setPen(QPen(ACCENT, 5))
            p.setBrush(fill)
            p.drawEllipse(c, radius, radius)
        progress = self.dwell_progress
        if progress > 0:
            p.setPen(QPen(ACCENT, 5, Qt.SolidLine, Qt.RoundCap))
            p.setBrush(Qt.NoBrush)
            p.drawArc(ring, 90 * 16, -int(360 * 16 * progress))
        now = self.clock()
        if self.flash_active(now):
            t = (now - self._flash_start) * 1000.0 / FLASH_MS
            col = QColor(TEXT)
            col.setAlpha(int(255 * (1.0 - t)))
            p.setPen(QPen(col, 4))
            p.setBrush(Qt.NoBrush)
            r = radius + 24 * t
            p.drawEllipse(c, r, r)
        if self.warning:
            f = QFont()
            f.setFamilies(["Inter", "Segoe UI"])
            f.setPointSizeF(8.0)
            f.setBold(True)
            p.setFont(f)
            box = QRectF(2, 2, self.SIZE - 4, 22)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(16, 16, 25, 210))
            p.drawRoundedRect(box, 8, 8)
            p.setPen(WARNING)
            p.drawText(box, Qt.AlignCenter, self.warning)
        if self._label and self.dry_run:
            f = QFont()
            f.setFamilies(["Inter", "Segoe UI"])
            f.setPointSizeF(10.0)
            f.setBold(True)
            p.setFont(f)
            p.setPen(TEXT)
            p.drawText(QRectF(0, self.SIZE - 30, self.SIZE, 24), Qt.AlignCenter, f"[simulación] {self._label}")
        p.end()
