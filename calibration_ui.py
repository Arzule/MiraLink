"""Interfaz Qt de la calibración: modal de preparación, pantalla completa y diálogo de resultado.

El hilo de seguimiento se inyecta con `tracker_factory` (en tests se usa uno falso, nunca la
cámara). Los timestamps vienen de TrackingFrame; el reloj (`clock`) solo se usa para el
watchdog de cuadros y para mantener ESC.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Callable, Dict, List, Optional

import numpy as np
from PySide6.QtCore import QPointF, QRect, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QGuiApplication, QImage, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from calibration import (
    STAGE_VALIDATION,
    WARN_ERROR_PX,
    CalibrationSession,
    GazeMapper,
    HoldToExit,
    RecenterSession,
    face_guidance,
    fit_session,
    save_calibration,
    screen_geometry_from_qt,
    validation_points,
)

logger = logging.getLogger(__name__)

# Colores del diseño (AGENTS.md)
BG = QColor("#101019")
CARD = QColor("#181829")
BORDER = QColor("#2E2E4A")
ACCENT = QColor("#8070FF")
TEXT = QColor("#F3F3FA")
TEXT_SECONDARY = QColor("#A1A1BE")
OK_GREEN = QColor("#3DDC84")
BAD_RED = QColor("#FF5C5C")

FONT_FAMILIES = ["Inter", "Segoe UI"]


def _font(point_size: float, bold: bool = False) -> QFont:
    f = QFont()
    f.setFamilies(FONT_FAMILIES)
    f.setPointSizeF(max(1.0, point_size))
    f.setBold(bold)
    return f


def frame_to_qimage(bgr: np.ndarray) -> QImage:
    """Cuadro BGR de OpenCV -> QImage RGB (copia independiente del buffer)."""
    rgb = np.ascontiguousarray(bgr[:, :, ::-1])
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()


# ------------------------------------------------------------ previsualización
class CameraPreview(QWidget):
    """Previsualización 16:9 en vivo (espejada) con óvalo guía e insignia de rostro."""

    def __init__(self, parent=None, compact: bool = False):
        super().__init__(parent)
        self.compact = compact
        self._image: Optional[QImage] = None
        self.face_ok = False
        self.badge_text = "Rostro No Detectado"
        self.hint_text = ""
        # Tamaño fijo 16:9 (el layout del diálogo no aplica heightForWidth de forma fiable)
        self.setFixedSize(240, 135) if compact else self.setFixedSize(560, 315)
        self.setAccessibleName("Previsualización de la cámara")

    def set_state(self, image_bgr: Optional[np.ndarray], face_found: bool, face_center, interocular: float):
        """Actualiza imagen e insignia con los datos de un TrackingFrame."""
        if image_bgr is not None:
            self._image = frame_to_qimage(image_bgr)
        self.face_ok, self.hint_text = face_guidance(face_found, face_center, interocular)
        self.badge_text = "Rostro Detectado" if self.face_ok else "Rostro No Detectado"
        self.update()

    def on_frame(self, frame):
        self.set_state(frame.image, frame.face_found, frame.face_center, frame.interocular)

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = self.rect()
        radius = 10 if self.compact else 16
        path = QPainterPath()
        path.addRoundedRect(QRectF(r), radius, radius)
        p.setClipPath(path)
        p.fillRect(r, CARD)
        if self._image is not None:
            img = self._image.flipped(Qt.Horizontal)  # espejo, como un espejo real
            scaled = img.scaled(r.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
            off_x = (scaled.width() - r.width()) // 2
            off_y = (scaled.height() - r.height()) // 2
            p.drawImage(r, scaled, QRect(off_x, off_y, r.width(), r.height()))
        # óvalo guía de encuadre
        pen = QPen(ACCENT, 2 if self.compact else 3, Qt.DashLine)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        c = QPointF(r.center())
        p.drawEllipse(c, r.width() * 0.17, r.height() * 0.40)
        # insignia
        p.setClipping(False)
        badge_font = _font(7 if self.compact else 10, True)
        p.setFont(badge_font)
        fm = p.fontMetrics()
        pad = 6 if self.compact else 10
        bw = fm.horizontalAdvance(self.badge_text) + 2 * pad + 14
        bh = fm.height() + pad
        brect = QRectF(8, 8, bw, bh)
        color = OK_GREEN if self.face_ok else BAD_RED
        p.setPen(Qt.NoPen)
        bg = QColor(0, 0, 0, 160)
        p.setBrush(bg)
        p.drawRoundedRect(brect, bh / 2, bh / 2)
        p.setBrush(color)
        p.drawEllipse(QPointF(brect.left() + pad + 3, brect.center().y()), 4, 4)
        p.setPen(TEXT)
        p.drawText(QRectF(brect.left() + pad + 12, brect.top(), bw, bh), Qt.AlignVCenter | Qt.AlignLeft, self.badge_text)
        # aviso de distancia/centrado
        if self.hint_text and self.face_ok:
            p.setFont(_font(8 if self.compact else 12, True))
            fm = p.fontMetrics()
            hw = fm.horizontalAdvance(self.hint_text) + 24
            hh = fm.height() + 10
            hrect = QRectF((r.width() - hw) / 2, r.height() - hh - 10, hw, hh)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(0, 0, 0, 170))
            p.drawRoundedRect(hrect, hh / 2, hh / 2)
            p.setPen(TEXT)
            p.drawText(hrect, Qt.AlignCenter, self.hint_text)
        p.end()


# ------------------------------------------------------ pantalla completa
class CalibrationScreen(QWidget):
    """Pantalla completa con los 9 puntos. Emite `completed(GazeMapper)` o `aborted(str)`."""

    completed = Signal(object)
    aborted = Signal(str)

    HOLD_EXIT_S = 1.5
    FRAME_WATCHDOG_S = 0.3

    def __init__(
        self,
        session: CalibrationSession,
        dpr: float = 1.0,
        fit_options: Optional[dict] = None,
        clock: Callable[[], float] = time.monotonic,
        parent=None,
        recorder=None,
    ):
        super().__init__(parent)
        self.session = session
        self.dpr = dpr if dpr > 0 else 1.0
        self.fit_options = fit_options  # None = calibration.DEFAULT_FIT
        self.clock = clock
        self.recorder = recorder  # gaze_session.SessionRecorder en --record-session
        self._layout_stage = session.stage
        self.hold = HoldToExit(self.HOLD_EXIT_S)
        self._last_frame_wall = -1e9
        self._finished = False
        self.mini = CameraPreview(self, compact=True)
        self.mini.setAccessibleName("Mini previsualización de la cámara")
        self.setWindowTitle("Calibrando Rastreador Ocular")
        self.setWindowFlag(Qt.FramelessWindowHint, True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setAccessibleName("Pantalla de calibración")
        self.setAutoFillBackground(False)
        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._on_timer)

    # ------------------------------------------------------------- control
    def start(self):
        self._timer.start()
        self.setFocus()

    def stop(self):
        self._timer.stop()

    def on_frame(self, frame):
        """Recibe un TrackingFrame del hilo de seguimiento."""
        if self._finished:
            return
        self._last_frame_wall = self.clock()
        self.mini.on_frame(frame)
        if self.recorder is not None:
            self.recorder.add(frame, self.session)  # con el punto que se mostraba al llegar el cuadro
        valid = bool(frame.face_found and not frame.eyes_closed and frame.features is not None)
        self.session.update(frame.timestamp, frame.features, valid, eyes_closed=bool(frame.eyes_closed))
        self._check_finished()
        self._sync_layout()
        self.update()

    def on_tracker_failed(self, message: str):
        self._abort(f"Se perdió el seguimiento: {message}")

    def tick(self, now: Optional[float] = None):
        """Avance por reloj: mantiene la sesión viva si no llegan cuadros y gestiona ESC."""
        now = self.clock() if now is None else now
        if self._finished:
            return
        if self.hold.completed(now):
            self._abort("Calibración cancelada.")
            return
        if now - self._last_frame_wall > self.FRAME_WATCHDOG_S:
            self.session.update(now, None, False)
            self._check_finished()
        self._sync_layout()
        self.update()

    def _on_timer(self):
        self.tick()

    def _check_finished(self):
        if self._finished or not self.session.finished:
            return
        if self.session.phase == "done":
            self._finished = True
            self._timer.stop()
            try:
                mapper = fit_session(self.session, self.fit_options)
            except Exception as e:  # noqa: BLE001
                self._finished = False
                self._abort(f"No se pudo ajustar la calibración: {e}")
                return
            self.completed.emit(mapper)
        else:
            self._abort(self.session.message or "La calibración se interrumpió.")

    def _abort(self, message: str):
        if self._finished:
            return
        self._finished = True
        self._timer.stop()
        self.aborted.emit(message)

    # --------------------------------------------------------------- teclado
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            if not event.isAutoRepeat():
                self.hold.press(self.clock())
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_Escape:
            if not event.isAutoRepeat():
                self.hold.release()
            event.accept()
            return
        super().keyReleaseEvent(event)

    # ------------------------------------------------------------- geometría
    def logical_point(self, index: int) -> QPointF:
        """Punto de calibración (píxeles físicos) -> coordenadas lógicas del widget."""
        x, y = self.session.points[index]
        return QPointF(x / self.dpr, y / self.dpr)

    def logical_validation_point(self, index: int) -> QPointF:
        x, y = self.session.validation[index]
        return QPointF(x / self.dpr, y / self.dpr)

    def layout_regions(self, stage: Optional[str] = None) -> Dict[str, QRectF]:
        """Zonas de título, aviso ESC y mini previsualización.

        Calibración: entre filas de puntos (10/50/90 %), título al 30 % y aviso al 70 %.
        Validación (puntos al 30/56/70 %; la cuadrícula ya no se dibuja): título arriba (10 %) y
        aviso abajo (92 %).
        """
        stage = stage or self.session.stage
        w, h = self.width(), self.height()
        title_h = min(0.24 * h, 170)
        notice_w = min(w * 0.5, 560)
        if stage == STAGE_VALIDATION:
            title_y, band_y = 0.10, 0.92
        else:
            title_y, band_y = 0.30, 0.70
        title = QRectF(w * 0.15, h * title_y - title_h / 2, w * 0.70, title_h)
        notice = QRectF((w - notice_w) / 2, h * band_y - 32, notice_w, 64)
        mini = QRectF(w - 24 - 240, h * band_y - 135 / 2, 240, 135)
        return {"title": title, "notice": notice, "mini": mini}

    def _place_mini(self):
        m = self.layout_regions()["mini"]
        self.mini.setGeometry(int(m.x()), int(m.y()), 240, 135)

    def _sync_layout(self):
        """Recoloca la mini previsualización al pasar de calibración a validación."""
        if self.session.stage != self._layout_stage:
            self._layout_stage = self.session.stage
            self._place_mini()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._place_mini()

    # ------------------------------------------------------------------ pintura
    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), BG)
        regions = self.layout_regions()
        s = self.session
        scale = max(0.6, self.height() / 1080.0)
        now = self.clock()

        validating = s.in_validation
        # puntos inactivos tenues (en validación no se dibuja la cuadrícula: no distrae)
        for i in range(0 if validating else s.total):
            if i == s.point_index and not s.finished:
                continue
            c = self.logical_point(i)
            done = i < s.point_index
            col = QColor(ACCENT)
            col.setAlpha(150 if done else 70)
            p.setPen(Qt.NoPen)
            p.setBrush(col)
            p.drawEllipse(c, 9 * scale + 3, 9 * scale + 3)

        # punto activo con anillos y etiqueta
        if not s.finished:
            if validating:
                c = self.logical_validation_point(s.validation_index)
            else:
                c = self.logical_point(s.point_index)
            prog = s.phase_progress(now)
            pulse = 0.5 + 0.5 * math.sin(now * 5.0)
            for k, base in enumerate((54, 38, 24)):
                col = QColor(ACCENT)
                col.setAlpha(40 + 30 * k + int(30 * pulse * (k == 0)))
                p.setPen(QPen(col, 3))
                p.setBrush(Qt.NoBrush)
                p.drawEllipse(c, (base + 6 * pulse * (k == 0)) * scale, (base + 6 * pulse * (k == 0)) * scale)
            if s.phase == "capturing":
                p.setPen(QPen(OK_GREEN, 5))
                arc = QRectF(c.x() - 30 * scale, c.y() - 30 * scale, 60 * scale, 60 * scale)
                p.drawArc(arc, 90 * 16, -int(360 * 16 * prog))
            p.setPen(Qt.NoPen)
            p.setBrush(ACCENT)
            p.drawEllipse(c, 12 * scale, 12 * scale)
            p.setPen(TEXT)
            p.setFont(_font(14 * scale, True))
            label_w = 150 * scale
            lx = c.x() + 70 * scale if c.x() + 70 * scale + label_w < self.width() else c.x() - 70 * scale - label_w
            p.drawText(QRectF(lx, c.y() - 20 * scale, label_w, 40 * scale), Qt.AlignVCenter | Qt.AlignLeft, "Mira aquí")

        # título, subtítulo y píldora
        t = regions["title"]
        p.setPen(TEXT)
        p.setFont(_font(26 * scale, True))
        p.drawText(QRectF(t.x(), t.y(), t.width(), t.height() * 0.38), Qt.AlignCenter, "Calibrando Rastreador Ocular")
        p.setPen(TEXT_SECONDARY)
        p.setFont(_font(14 * scale))
        subtitle = (
            "Comprobando la precisión con puntos nuevos. Mira cada punto."
            if validating
            else "Mira cada punto resaltado hasta que se fije."
        )
        p.drawText(QRectF(t.x(), t.y() + t.height() * 0.38, t.width(), t.height() * 0.26), Qt.AlignCenter, subtitle)
        if validating:
            n_val = len(s.validation)
            pill_text = f"Validación {min(s.validation_index + 1, n_val)} de {n_val}"
        else:
            pill_text = f"Punto {min(s.point_index + 1, s.total)} de {s.total}"
        p.setFont(_font(12 * scale, True))
        fm = p.fontMetrics()
        pw = fm.horizontalAdvance(pill_text) + 32
        ph = fm.height() + 14
        pill = QRectF(t.center().x() - pw / 2, t.y() + t.height() * 0.70, pw, ph)
        p.setPen(QPen(BORDER, 1))
        p.setBrush(CARD)
        p.drawRoundedRect(pill, ph / 2, ph / 2)
        p.setPen(TEXT)
        p.drawText(pill, Qt.AlignCenter, pill_text)
        if s.message and not s.finished:
            p.setPen(BAD_RED)
            p.setFont(_font(11 * scale))
            p.drawText(QRectF(t.x(), pill.bottom() + 6, t.width(), 28), Qt.AlignCenter, s.message)

        # aviso ESC semiopaco con progreso
        n = regions["notice"]
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(24, 24, 41, 215))
        p.drawRoundedRect(n, 14, 14)
        chip = QRectF(n.x() + 14, n.center().y() - 16, 56, 32)
        p.setPen(QPen(BORDER, 1))
        p.setBrush(BG)
        p.drawRoundedRect(chip, 8, 8)
        p.setPen(TEXT)
        p.setFont(_font(11, True))
        p.drawText(chip, Qt.AlignCenter, "ESC")
        p.setFont(_font(11))
        p.setPen(TEXT_SECONDARY)
        p.drawText(
            QRectF(chip.right() + 12, n.y() + 6, n.width() - chip.width() - 40, n.height() - 18),
            Qt.AlignVCenter | Qt.AlignLeft | Qt.TextWordWrap,
            "Mantén presionada ESC para salir de la calibración",
        )
        hp = self.hold.progress(now)
        bar = QRectF(n.x() + 14, n.bottom() - 10, n.width() - 28, 5)
        p.setPen(Qt.NoPen)
        p.setBrush(BORDER)
        p.drawRoundedRect(bar, 2.5, 2.5)
        if hp > 0:
            p.setBrush(ACCENT)
            p.drawRoundedRect(QRectF(bar.x(), bar.y(), bar.width() * hp, bar.height()), 2.5, 2.5)
        p.end()


# ------------------------------------------------------------ recentrar
class RecenterScreen(QWidget):
    """Pantalla completa de "Recentrar": un único punto central y ~1 s de muestras.

    Emite `completed((dx, dy))` con el offset o `aborted(str)`. Mantener ESC 1.5 s cancela. Si no
    llegan cuadros, el reloj (watchdog) la cancela tras `max_wait_s` de la sesión más 1 s.
    """

    completed = Signal(object)
    aborted = Signal(str)

    HOLD_EXIT_S = 1.5

    def __init__(self, session: RecenterSession, mapper, clock: Callable[[], float] = time.monotonic, parent=None):
        super().__init__(parent)
        self.session = session
        self.mapper = mapper
        self.clock = clock
        self.hold = HoldToExit(self.HOLD_EXIT_S)
        self._started_wall: Optional[float] = None
        self._now_ts: Optional[float] = None
        self._finished = False
        self.setWindowTitle("Recentrar")
        self.setWindowFlag(Qt.FramelessWindowHint, True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setAccessibleName("Pantalla de recentrado")
        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self.tick)

    def start(self):
        self._started_wall = self.clock()
        self._timer.start()
        self.setFocus()

    def stop(self):
        self._timer.stop()

    def on_frame(self, frame):
        if self._finished:
            return
        self._now_ts = frame.timestamp
        valid = bool(frame.face_found and not frame.eyes_closed and frame.features is not None)
        self.session.update(frame.timestamp, frame.features, valid, bool(frame.eyes_closed), self.mapper)
        self._check_finished()
        self.update()

    def on_tracker_failed(self, message: str):
        self._finish_aborted(f"Se perdió el seguimiento: {message}")

    def tick(self, now: Optional[float] = None):
        now = self.clock() if now is None else now
        if self._finished:
            return
        if self.hold.completed(now):
            self._finish_aborted("Recentrado cancelado.")
            return
        if self._started_wall is not None and now - self._started_wall > self.session.max_wait_s + 1.0:
            self._finish_aborted("No se pudo recentrar: no llegan cuadros de la cámara.")
            return
        self.update()

    def _check_finished(self):
        if not self.session.finished:
            return
        if self.session.phase == "done":
            self._finished = True
            self._timer.stop()
            self.completed.emit(self.session.offset)
        else:
            self._finish_aborted(self.session.message or "No se pudo recentrar.")

    def _finish_aborted(self, message: str):
        if self._finished:
            return
        self._finished = True
        self._timer.stop()
        self.aborted.emit(message)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            if not event.isAutoRepeat():
                self.hold.press(self.clock())
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_Escape:
            if not event.isAutoRepeat():
                self.hold.release()
            event.accept()
            return
        super().keyReleaseEvent(event)

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), BG)
        w, h = self.width(), self.height()
        scale = max(0.6, h / 1080.0)
        now = self.clock()
        c = QPointF(w / 2.0, h / 2.0)
        pulse = 0.5 + 0.5 * math.sin(now * 5.0)
        for k, base in enumerate((54, 38, 24)):
            col = QColor(ACCENT)
            col.setAlpha(40 + 30 * k + int(30 * pulse * (k == 0)))
            p.setPen(QPen(col, 3))
            p.setBrush(Qt.NoBrush)
            r = (base + 6 * pulse * (k == 0)) * scale
            p.drawEllipse(c, r, r)
        prog = self.session.progress(self._now_ts) if self._now_ts is not None else 0.0
        if prog > 0:
            p.setPen(QPen(OK_GREEN, 5))
            p.drawArc(QRectF(c.x() - 30 * scale, c.y() - 30 * scale, 60 * scale, 60 * scale), 90 * 16, -int(360 * 16 * prog))
        p.setPen(Qt.NoPen)
        p.setBrush(ACCENT)
        p.drawEllipse(c, 12 * scale, 12 * scale)
        title_h = min(0.24 * h, 170)
        t = QRectF(w * 0.15, h * 0.30 - title_h / 2, w * 0.70, title_h)
        p.setPen(TEXT)
        p.setFont(_font(26 * scale, True))
        p.drawText(QRectF(t.x(), t.y(), t.width(), t.height() * 0.45), Qt.AlignCenter, "Recentrar")
        p.setPen(TEXT_SECONDARY)
        p.setFont(_font(14 * scale))
        p.drawText(
            QRectF(t.x(), t.y() + t.height() * 0.45, t.width(), t.height() * 0.4),
            Qt.AlignCenter,
            "Mira el punto central hasta que se complete el círculo.",
        )
        notice_w = min(w * 0.5, 560)
        n = QRectF((w - notice_w) / 2, h * 0.70 - 32, notice_w, 64)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(24, 24, 41, 215))
        p.drawRoundedRect(n, 14, 14)
        p.setPen(TEXT_SECONDARY)
        p.setFont(_font(11))
        p.drawText(n.adjusted(16, 0, -16, -10), Qt.AlignCenter | Qt.TextWordWrap, "Mantén presionada ESC para cancelar")
        hp = self.hold.progress(now)
        if hp > 0:
            bar = QRectF(n.x() + 14, n.bottom() - 10, (n.width() - 28) * hp, 5)
            p.setBrush(ACCENT)
            p.drawRoundedRect(bar, 2.5, 2.5)
        p.end()


# ------------------------------------------------------ diálogo de resultado
def error_source_text(source: str, cv_px: Optional[float] = None, points: int = 0) -> str:
    """Explica qué error se muestra al final de la calibración."""
    if source == "validation":
        text = f"Medido en {points} puntos de validación nuevos, no usados para calibrar (error real)."
        if cv_px is not None:
            text += f" Validación cruzada de los 9 puntos: ~{int(round(cv_px))} px."
        return text
    if source == "cv":
        return "Estimado con validación cruzada (dejando un punto fuera); no hubo puntos de validación suficientes."
    return "Error de ajuste sobre los propios puntos de calibración (optimista)."


class CalibrationResultDialog(QDialog):
    """"Calibración completada. Error estimado: ~N px" con Aceptar / Repetir y de dónde sale el error."""

    def __init__(self, error_px: float, parent=None, source: str = "", cv_px: Optional[float] = None, points: int = 0):
        super().__init__(parent)
        self.setWindowTitle("Calibración completada")
        self.setModal(True)
        self.error_px = error_px
        self.source = source
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)
        self.message_label = QLabel(f"Calibración completada. Error estimado: ~{int(round(error_px))} px")
        self.message_label.setObjectName("title")
        self.message_label.setWordWrap(True)
        self.message_label.setAccessibleName("Resultado de la calibración")
        layout.addWidget(self.message_label)
        self.source_label = QLabel(error_source_text(source, cv_px, points) if source else "")
        self.source_label.setObjectName("description")
        self.source_label.setWordWrap(True)
        self.source_label.setAccessibleName("Origen del error estimado")
        self.source_label.setVisible(bool(source))
        layout.addWidget(self.source_label)
        self.warning_label = QLabel(
            f"El error estimado supera {WARN_ERROR_PX} px: el control podría ser impreciso. "
            "Te recomendamos repetir la calibración con mejor iluminación y sin mover la cabeza."
        )
        self.warning_label.setObjectName("description")
        self.warning_label.setWordWrap(True)
        self.warning_label.setAccessibleName("Advertencia de error de calibración")
        self.warning_label.setVisible(error_px > WARN_ERROR_PX)
        layout.addWidget(self.warning_label)
        row = QHBoxLayout()
        row.setSpacing(16)
        self.btn_repeat = QPushButton("Repetir")
        self.btn_repeat.setProperty("variant", "secondary")
        self.btn_repeat.setMinimumHeight(48)
        self.btn_repeat.setAccessibleName("Botón repetir calibración")
        self.btn_accept = QPushButton("Aceptar")
        self.btn_accept.setProperty("variant", "primary")
        self.btn_accept.setMinimumHeight(48)
        self.btn_accept.setDefault(True)
        self.btn_accept.setAccessibleName("Botón aceptar calibración")
        row.addStretch()
        row.addWidget(self.btn_repeat)
        row.addWidget(self.btn_accept)
        layout.addLayout(row)
        self.btn_accept.clicked.connect(self.accept)
        self.btn_repeat.clicked.connect(self.reject)
        self.setMinimumWidth(520)


# ------------------------------------------------------------- modal principal
def default_camera_lister() -> List[int]:
    from eye_tracker import list_cameras

    return list_cameras()


def default_tracker_factory(camera_index: int, camera_size=(640, 480)):
    from eye_tracker import TrackerThread

    return TrackerThread(camera_index, camera_size=camera_size)


def camera_size_from_config(config: dict):
    cam = config.get("camera", {}) if isinstance(config, dict) else {}
    return int(cam.get("width", 640)), int(cam.get("height", 480))


def camera_status_text(frame) -> str:
    """Resolución y fps medidos (con aviso si bajan de 25 fps) para el modal de calibración."""
    from eye_tracker import MIN_CAMERA_FPS

    info = frame.camera_info
    if not info and frame.image is not None:
        info = f"{frame.image.shape[1]}x{frame.image.shape[0]}"
    if not frame.fps:
        return f"Cámara: {info} · midiendo fps…" if info else "Cámara: midiendo fps…"
    text = f"Cámara: {info} · {frame.fps:.0f} fps" if info else f"Cámara: {frame.fps:.0f} fps"
    if frame.fps < MIN_CAMERA_FPS:
        text += f". Menos de {MIN_CAMERA_FPS:.0f} fps: la detección puede fallar; mejora la iluminación."
    return text


class CalibrationDialog(QDialog):
    """Modal "Calibrar Rastreador Ocular": cámara, encuadre en vivo y botón de inicio."""

    camera_selected = Signal(int)

    def __init__(
        self,
        config: dict,
        parent=None,
        camera_lister: Callable[[], List[int]] = default_camera_lister,
        tracker_factory: Callable[[int], object] = default_tracker_factory,
        calibration_path: Optional[str] = None,
        screen=None,
        clock: Callable[[], float] = time.monotonic,
        result_dialog_exec: Optional[Callable[[CalibrationResultDialog], bool]] = None,
        validation: bool = True,
        record_dir: Optional[str] = None,
    ):
        super().__init__(parent)
        from calibration import default_calibration_path

        self.record_dir = record_dir  # --record-session: graba cada intento en esta carpeta
        self.validation = bool(validation or record_dir)
        self.recorder = None
        self.saved_sessions: List[str] = []
        self.config = config
        self.camera_lister = camera_lister
        if tracker_factory is default_tracker_factory:
            size = camera_size_from_config(config)
            tracker_factory = lambda i: default_tracker_factory(i, size)  # noqa: E731
        self.tracker_factory = tracker_factory
        self._last_camera_status = -1e9
        self.calibration_path = calibration_path or default_calibration_path()
        self._screen = screen
        self.clock = clock
        self._result_dialog_exec = result_dialog_exec or (lambda d: d.exec() == QDialog.Accepted)
        self.accepted_mapper: Optional[GazeMapper] = None
        self.tracker = None
        self.calib_screen: Optional[CalibrationScreen] = None
        self._cameras_loaded = False
        self._closing = False

        self.setWindowTitle("Calibrar Rastreador Ocular")
        self.setModal(True)
        self.setMinimumWidth(620)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        self.title_label = QLabel("Calibrar Rastreador Ocular")
        self.title_label.setObjectName("title")
        self.title_label.setAccessibleName("Título: Calibrar Rastreador Ocular")
        layout.addWidget(self.title_label)

        cam_label = QLabel("Dispositivo de Cámara")
        cam_label.setObjectName("description")
        cam_label.setAccessibleName("Etiqueta dispositivo de cámara")
        layout.addWidget(cam_label)
        self.camera_combo = QComboBox()
        self.camera_combo.setMinimumHeight(48)
        self.camera_combo.setAccessibleName("Selector de dispositivo de cámara")
        layout.addWidget(self.camera_combo)

        self.preview = CameraPreview()
        layout.addWidget(self.preview, 0, Qt.AlignHCenter)

        self.camera_info_label = QLabel("")
        self.camera_info_label.setObjectName("description")
        self.camera_info_label.setWordWrap(True)
        self.camera_info_label.setAccessibleName("Resolución y fps de la cámara")
        layout.addWidget(self.camera_info_label)

        self.instructions = QLabel(
            "Posiciona tu rostro dentro de la guía y asegúrate de tener buena iluminación antes de comenzar."
        )
        self.instructions.setObjectName("description")
        self.instructions.setWordWrap(True)
        self.instructions.setAccessibleName("Instrucciones de calibración")
        layout.addWidget(self.instructions)

        self.status_label = QLabel("")
        self.status_label.setObjectName("description")
        self.status_label.setWordWrap(True)
        self.status_label.setAccessibleName("Estado de la calibración")
        layout.addWidget(self.status_label)

        row = QHBoxLayout()
        row.setSpacing(16)
        row.addStretch()
        self.btn_cancel = QPushButton("Cancelar")
        self.btn_cancel.setProperty("variant", "secondary")
        self.btn_cancel.setMinimumHeight(48)
        self.btn_cancel.setAccessibleName("Botón cancelar calibración")
        self.btn_start = QPushButton("Iniciar Calibración")
        self.btn_start.setProperty("variant", "primary")
        self.btn_start.setMinimumHeight(48)
        self.btn_start.setDefault(True)
        self.btn_start.setAccessibleName("Botón iniciar calibración")
        row.addWidget(self.btn_cancel)
        row.addWidget(self.btn_start)
        layout.addLayout(row)

        self.btn_cancel.clicked.connect(self.reject)
        self.btn_start.clicked.connect(self.start_calibration)
        self.camera_combo.currentIndexChanged.connect(self._on_camera_changed)

    # --------------------------------------------------------------- cámaras
    def _populate_cameras(self):
        current = int(self.config.get("camera", {}).get("device_index", 0))
        try:
            cams = list(self.camera_lister())
        except Exception:  # noqa: BLE001
            cams = []
        if current not in cams:
            cams.append(current)  # se respeta camera.device_index aunque no responda al sondeo
        cams = sorted(set(cams))
        self.camera_combo.blockSignals(True)
        self.camera_combo.clear()
        for i in cams:
            self.camera_combo.addItem(f"Cámara {i}", i)
        self.camera_combo.setCurrentIndex(cams.index(current))
        self.camera_combo.blockSignals(False)
        self._cameras_loaded = True

    def current_camera_index(self) -> int:
        data = self.camera_combo.currentData()
        return int(data) if data is not None else 0

    def _on_camera_changed(self, _idx):
        self.camera_selected.emit(self.current_camera_index())
        self._restart_tracker()

    # ---------------------------------------------------------------- tracker
    def _start_tracker(self):
        """El modal es el ÚNICO dueño del tracker (y de la cámara): la pantalla completa lo reutiliza."""
        if self.tracker is not None or self._closing:
            return  # un arranque diferido que llega tras cerrar el modal no debe reabrir la cámara
        index = self.current_camera_index()
        logger.info("Modal: iniciando tracker (cámara %s)", index)
        try:
            tracker = self.tracker_factory(index)
            tracker.frame_ready.connect(self._on_frame)
            tracker.failed.connect(self._on_tracker_failed)
            self.tracker = tracker
            tracker.start()
        except Exception as e:  # noqa: BLE001 - mensaje visible en lugar de fallar en silencio
            logger.exception("Modal: no se pudo iniciar el tracker")
            self.tracker = None
            self.status_label.setText(f"No se pudo usar la cámara: {e}")

    def _stop_tracker(self):
        tracker, self.tracker = self.tracker, None
        if tracker is not None:
            logger.info("Modal: deteniendo tracker")
            try:
                tracker.frame_ready.disconnect(self._on_frame)
                tracker.failed.disconnect(self._on_tracker_failed)
            except (RuntimeError, TypeError):
                pass
            tracker.stop()

    def _restart_tracker(self):
        self._stop_tracker()
        self.preview.set_state(None, False, None, 0.0)
        self._start_tracker()

    def _on_frame(self, frame):
        self.preview.on_frame(frame)
        if abs(frame.timestamp - self._last_camera_status) >= 0.5:  # resolución y fps, 2 veces por segundo
            self._last_camera_status = frame.timestamp
            self.camera_info_label.setText(camera_status_text(frame))
        if self.calib_screen is not None:
            self.calib_screen.on_frame(frame)

    def _on_tracker_failed(self, message: str):
        logger.error("Modal: fallo del tracker: %s", message)
        self.status_label.setText(f"No se pudo usar la cámara: {message}")
        if self.calib_screen is not None:
            self.calib_screen.on_tracker_failed(message)

    def showEvent(self, event):
        super().showEvent(event)
        # Las etiquetas con ajuste de línea necesitan recalcular la altura al conocer el ancho
        self.layout().activate()
        self.setMinimumHeight(self.layout().totalHeightForWidth(self.width()))
        if not self._cameras_loaded:
            self._populate_cameras()
        self._closing = False
        # con el modal como contexto: si se destruye antes, Qt descarta la llamada
        QTimer.singleShot(0, self, self._start_tracker)

    def done(self, result):
        logger.info("Modal: cerrando (resultado %s)", result)
        self._closing = True
        if self.calib_screen is not None:
            self._save_recording("closed")
        self._close_screen()
        self._stop_tracker()
        super().done(result)

    # ------------------------------------------------------------ calibración
    def _target_screen(self):
        """Pantalla inyectada, o la de la ventana que abrió el modal, o la primaria."""
        if self._screen is not None:
            return self._screen
        parent = self.parentWidget()
        if parent is not None and parent.screen() is not None:
            return parent.screen()
        return QGuiApplication.primaryScreen()

    def start_calibration(self):
        """Abre la pantalla completa de calibración (9 puntos) reutilizando el tracker del modal.

        Requiere que el modal NO se haya abierto con exec(): ocultarlo (hide) terminaría el bucle de
        exec() y, sin otra referencia, PySide destruiría el modal y con él la pantalla completa.
        """
        logger.info("Clic en 'Iniciar Calibración'")
        try:
            screen = self._target_screen()
            geom = screen_geometry_from_qt(screen)
            val = validation_points(geom["width"], geom["height"]) if self.validation else None
            session = CalibrationSession(geom["width"], geom["height"], validation=val)
            self.recorder = self._make_recorder(screen, geom, session) if self.record_dir else None
            scr = CalibrationScreen(
                session, dpr=float(screen.devicePixelRatio()), clock=self.clock, recorder=self.recorder
            )
            scr.completed.connect(self._on_completed)
            scr.aborted.connect(self._on_aborted)
            self.calib_screen = scr  # referencia fuerte: vive mientras ConfigWindow retenga el modal
            self._geometry = geom
            self.hide()
            scr.setGeometry(screen.geometry())
            scr.showFullScreen()
            scr.raise_()
            scr.activateWindow()
            scr.start()
            logger.info(
                "Pantalla de calibración mostrada en %s: visible=%s fullscreen=%s geometría=%s física=%s",
                screen.name(), scr.isVisible(), scr.isFullScreen(), scr.geometry().getRect(), geom,
            )
            if self.tracker is None:
                self._start_tracker()
        except Exception as e:  # noqa: BLE001 - nunca fallar en silencio
            logger.exception("No se pudo abrir la pantalla de calibración")
            self._close_screen()
            self.status_label.setText(f"No se pudo abrir la pantalla de calibración: {e}")
            self.show()

    def _close_screen(self):
        scr, self.calib_screen = self.calib_screen, None
        if scr is not None:
            scr.stop()
            scr.hide()
            scr.deleteLater()

    # ------------------------------------------------------------- grabación
    def _make_recorder(self, screen, geom, session):
        from gaze_session import SessionRecorder, make_session_meta

        size = screen.physicalSize()
        physical = (float(size.width()), float(size.height())) if size.width() > 0 else None
        meta = make_session_meta(
            geom,
            session,
            dpr=float(screen.devicePixelRatio()),
            physical_mm=physical,
            camera_index=self.current_camera_index(),
            config=self.config,
            extra={"screen_name": screen.name()},
        )
        return SessionRecorder(meta)

    def _save_recording(self, outcome: str) -> Optional[str]:
        """Guarda la sesión grabada (también si se canceló: sirve para analizar el fallo)."""
        recorder, self.recorder = self.recorder, None
        if recorder is None or len(recorder) == 0:
            return None
        recorder.meta["outcome"] = outcome
        try:
            path = recorder.save(self.record_dir)
        except OSError as e:
            logger.exception("No se pudo guardar la sesión grabada")
            self.status_label.setText(f"No se pudo guardar la sesión grabada: {e}")
            return None
        self.saved_sessions.append(path)
        logger.info("Sesión grabada en %s (%d cuadros)", path, len(recorder))
        return path

    def _on_aborted(self, message: str):
        logger.warning("Calibración interrumpida: %s", message)
        self._close_screen()
        path = self._save_recording("aborted")
        self.status_label.setText(message + (f" Sesión guardada en {path}" if path else ""))
        self.show()

    def _on_completed(self, mapper: GazeMapper):
        self._close_screen()
        self._save_recording("completed")
        error = mapper.estimated_error_px
        logger.info("Calibración completada (error mostrado: %s): %s", mapper.error_source, mapper.metrics)
        dlg = CalibrationResultDialog(
            error,
            self,
            source=mapper.error_source,
            cv_px=mapper.metrics.get("cv_rmse_px"),
            points=int(mapper.metrics.get("validation_points", 0)),
        )
        accepted = self._result_dialog_exec(dlg)
        if accepted:
            try:
                save_calibration(self.calibration_path, mapper, self._geometry, self.current_camera_index())
            except OSError as e:
                logger.exception("No se pudo guardar calibration.json")
                self.status_label.setText(f"No se pudo guardar la calibración: {e}")
                self.show()
                return
            logger.info("calibration.json guardado en %s", self.calibration_path)
            self.accepted_mapper = mapper
            self.accept()
        else:
            logger.info("El usuario eligió Repetir")
            self.start_calibration()
