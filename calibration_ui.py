"""Interfaz Qt de la calibración: modal de preparación, pantalla completa y diálogo de resultado.

El hilo de seguimiento se inyecta con `tracker_factory` (en tests se usa uno falso, nunca la
cámara). Los timestamps vienen de TrackingFrame; el reloj (`clock`) solo se usa para el
watchdog de cuadros y para mantener ESC.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Callable, Dict, List, Optional, Tuple

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
    CALIBRATION_MARGIN,
    HEAD_MOTION_STEPS,
    STAGE_HEAD,
    STAGE_VALIDATION,
    WARN_ERROR_PX,
    CalibrationSession,
    GazeMapper,
    HeadPoseGuide,
    HoldToExit,
    RecenterSession,
    face_guidance,
    fit_session,
    head_guide_text,
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
        eye_mode: str = "auto",
    ):
        super().__init__(parent)
        self.session = session
        self.dpr = dpr if dpr > 0 else 1.0
        self.fit_options = fit_options  # None = calibration.DEFAULT_FIT
        self.eye_mode = eye_mode  # config.json tracking.eye: auto elige el ojo de menor temblor
        self.clock = clock
        self.recorder = recorder  # gaze_session.SessionRecorder en --record-session
        self.head_guide = HeadPoseGuide()  # indicador en vivo de la fase de cabeza
        self.head_state: Optional[dict] = None
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
        # Los ojos cerrados los decide la sesión con el EAR de cada punto (mirar abajo baja el párpado)
        valid = bool(frame.face_found and frame.features is not None)
        ear = float(frame.ear_mean) if frame.face_found else None
        ears = (float(frame.ear_right), float(frame.ear_left)) if frame.face_found else None
        if self.session.in_head:
            step = self.session.head_step
            self.head_state = self.head_guide.update(step[0] if step else None, self.session.phase, frame.features)
        self.session.update(frame.timestamp, frame.features, valid, eyes_closed=bool(frame.eyes_closed), ear=ear, ears=ears)
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
            if self.session.in_head:
                # La calibración ya está completa: ESC solo salta la fase de movimiento de cabeza
                self.hold.release()
                self.session.skip_head()
                self._check_finished()
                return
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
                mapper = fit_session(self.session, self.fit_options, self.eye_mode)
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
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            # cada pantalla (calibración, comprobación, cada paso de cabeza) empieza al pulsar Enter
            if not event.isAutoRepeat() and self.session.start_stage():
                logger.info("Pantalla de calibración iniciada con Enter: %s", self.session.stage)
                self.update()
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
        """Zonas de texto, aviso ESC y mini previsualización.

        Calibración (3x3 con los bordes al 2 %): rótulo al 37.5 % y aviso al 62.5 % (entre las filas). Validación (rectángulo 35-65 %) y cabeza (punto central):
        rótulo arriba (10 %) y aviso abajo (92 %). `panel`: tarjeta grande de "Pulsa Enter" mientras la
        pantalla espera (centrada; en la cabeza, arriba para dejar ver el punto central). `guide`: indicador
        en vivo de la fase de cabeza, bajo el rótulo o bajo el panel.
        """
        stage = stage or self.session.stage
        w, h = self.width(), self.height()
        notice_w = min(w * 0.4, 460)
        if stage in (STAGE_VALIDATION, STAGE_HEAD):
            title_y, band_y, title_h = 0.10, 0.92, 0.20 * h  # arriba no hay puntos: cabe instrucción + píldora
        else:
            title_y, band_y, title_h = 0.375, 0.625, min(0.10 * h, 110)  # solo la píldora, entre filas
        title = QRectF(w * 0.20, h * title_y - title_h / 2, w * 0.60, title_h)
        notice = QRectF((w - notice_w) / 2, h * band_y - 30, notice_w, 60)
        mw, mh = self.mini_size()
        mini = QRectF(w - 24 - mw, h * band_y - mh / 2, mw, mh)
        if stage == STAGE_HEAD:
            panel = QRectF(w * 0.12, h * 0.04, w * 0.76, h * 0.30)
        else:
            panel = QRectF(w * 0.12, h * 0.14, w * 0.76, h * 0.42)
        regions = {"title": title, "notice": notice, "mini": mini, "panel": panel}
        if stage == STAGE_HEAD:  # indicador en vivo del giro (lejos del punto central)
            top = panel.bottom() if self.session.awaiting_start else title.bottom()
            regions["guide"] = QRectF(w * 0.25, top + 8, w * 0.50, 56)
        return regions

    def mini_size(self) -> Tuple[int, int]:
        """240x135 a 1080 px de alto o más; menor en pantallas bajas (no tapa los puntos de los bordes)."""
        k = min(1.0, max(0.5, self.height() / 1080.0))
        return int(240 * k), int(135 * k)

    def _place_mini(self):
        m = self.layout_regions()["mini"]
        self.mini.setFixedSize(int(m.width()), int(m.height()))  # CameraPreview tiene tamaño fijo
        self.mini.move(int(m.x()), int(m.y()))

    def _sync_layout(self):
        """Recoloca la mini previsualización al pasar de calibración a validación."""
        if self.session.stage != self._layout_stage:
            self._layout_stage = self.session.stage
            self._place_mini()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._place_mini()

    # ------------------------------------------------------------------ pintura
    def stage_texts(self) -> Tuple[str, str, str]:
        """(título, instrucción, línea secundaria) breves de la pantalla actual."""
        s = self.session
        if s.in_head:
            step = s.head_step
            n = len(s.head_motion)
            return f"Cabeza · paso {min(s.head_index + 1, n)} de {n}", step[1] if step else "", "Sigue mirando el punto."
        if s.in_validation:
            return "Comprobación", "Mira cada punto nuevo.", ""
        return "Calibración", "Mira cada punto hasta que se cierre el círculo.", ""

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), BG)
        regions = self.layout_regions()
        s = self.session
        scale = max(0.6, self.height() / 1080.0)
        now = self.clock()

        validating = s.in_validation
        head = s.in_head
        waiting = s.awaiting_start and not s.finished
        # puntos inactivos tenues (en validación y en la fase de cabeza no se dibuja la cuadrícula)
        for i in range(0 if (validating or head or waiting) else s.total):
            if i == s.point_index and not s.finished:
                continue
            c = self.logical_point(i)
            done = i < s.point_index
            col = QColor(ACCENT)
            col.setAlpha(150 if done else 70)
            p.setPen(Qt.NoPen)
            p.setBrush(col)
            p.drawEllipse(c, 9 * scale + 3, 9 * scale + 3)

        # punto activo con anillos y etiqueta (mientras espera Enter solo el punto central de la cabeza)
        if not s.finished and (not waiting or head):
            if head:
                c = QPointF(s.center[0] / self.dpr, s.center[1] / self.dpr)
            elif validating:
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
            if not waiting:
                p.setPen(TEXT)
                p.setFont(_font(20 * scale, True))
                label_w = 190 * scale
                lx = c.x() + 70 * scale if c.x() + 70 * scale + label_w < self.width() else c.x() - 70 * scale - label_w
                ly = min(max(c.y() - 24 * scale, 0.0), self.height() - 48 * scale)
                p.drawText(QRectF(lx, ly, label_w, 48 * scale), Qt.AlignVCenter | Qt.AlignLeft, "Mira aquí")

        title, instruction, secondary = self.stage_texts()
        if waiting:
            self._paint_start_panel(p, regions["panel"], scale, title, instruction, secondary)
        else:
            self._paint_running_label(p, regions["title"], scale, instruction if head else "")
        if head and self.head_state and "guide" in regions:
            self._paint_head_guide(p, regions["guide"], self.head_state, scale)

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
        p.setFont(_font(12, True))
        p.drawText(chip, Qt.AlignCenter, "ESC")
        p.setFont(_font(13))
        p.setPen(TEXT_SECONDARY)
        p.drawText(
            QRectF(chip.right() + 12, n.y() + 6, n.width() - chip.width() - 40, n.height() - 18),
            Qt.AlignVCenter | Qt.AlignLeft | Qt.TextWordWrap,
            "Mantén ESC para saltar este paso" if head else "Mantén ESC para salir",
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

    def _paint_start_panel(self, p: QPainter, r: QRectF, scale: float, title: str, instruction: str, secondary: str):
        """Tarjeta grande antes de cada pantalla: título, instrucción breve y "Pulsa Enter para empezar"."""
        p.setPen(QPen(BORDER, 1))
        p.setBrush(CARD)
        p.drawRoundedRect(r, 24, 24)
        rows = [(title, _font(34 * scale, True), TEXT, 1.0), (instruction, _font(30 * scale, True), TEXT, 1.3)]
        if secondary:
            rows.append((secondary, _font(22 * scale), TEXT_SECONDARY, 0.8))
        rows.append(("Pulsa Enter para empezar", _font(26 * scale, True), ACCENT, 1.0))
        total = sum(weight for *_rest, weight in rows)
        y = r.y() + 0.06 * r.height()
        usable = 0.88 * r.height()
        for text, font, color, weight in rows:
            hh = usable * weight / total
            p.setFont(font)
            p.setPen(color)
            p.drawText(QRectF(r.x() + 24, y, r.width() - 48, hh), Qt.AlignCenter | Qt.TextWordWrap, text)
            y += hh
        if self.session.message:
            p.setPen(BAD_RED)
            p.setFont(_font(16 * scale))
            p.drawText(QRectF(r.x(), r.bottom() + 6, r.width(), 40 * scale), Qt.AlignCenter | Qt.TextWordWrap, self.session.message)

    def _paint_running_label(self, p: QPainter, t: QRectF, scale: float, instruction: str):
        """Durante la captura: instrucción breve (solo en la cabeza) y píldora grande con el avance."""
        s = self.session
        if s.in_head:
            n_head = len(s.head_motion)
            action = "Quédate quieto" if s.phase == "capturing" else "Prepárate"
            pill_text = f"Paso {min(s.head_index + 1, n_head)} de {n_head} · {action}"
        elif s.in_validation:
            n_val = len(s.validation)
            pill_text = f"Comprobación {min(s.validation_index + 1, n_val)} de {n_val}"
        else:
            pill_text = f"Punto {min(s.point_index + 1, s.total)} de {s.total}"
        top = t.y()
        if instruction:
            p.setPen(TEXT)
            p.setFont(_font(28 * scale, True))
            p.drawText(QRectF(t.x(), top, t.width(), t.height() * 0.5), Qt.AlignCenter | Qt.TextWordWrap, instruction)
            top += t.height() * 0.5
        p.setFont(_font(22 * scale, True))
        fm = p.fontMetrics()
        pw = fm.horizontalAdvance(pill_text) + 40
        ph = fm.height() + 18
        if not instruction:  # solo la píldora: centrada en su franja
            top = t.center().y() - ph / 2 - 4
        pill = QRectF(t.center().x() - pw / 2, top + 4, pw, ph)
        p.setPen(QPen(BORDER, 1))
        p.setBrush(CARD)
        p.drawRoundedRect(pill, ph / 2, ph / 2)
        p.setPen(TEXT)
        p.drawText(pill, Qt.AlignCenter, pill_text)
        if s.message and not s.finished:
            p.setPen(BAD_RED)
            p.setFont(_font(16 * scale))
            p.drawText(QRectF(t.x(), pill.bottom() + 6, t.width(), 40 * scale), Qt.AlignCenter | Qt.TextWordWrap, s.message)

    @staticmethod
    def _paint_head_guide(p: QPainter, r: QRectF, state: dict, scale: float = 1.0):
        """Texto del giro actual y barra con la zona objetivo (verde dentro, ámbar fuera)."""
        color = OK_GREEN if state["in_zone"] else QColor("#FFB020")
        p.setPen(color)
        p.setFont(_font(16 * scale, True))
        p.drawText(QRectF(r.x(), r.y(), r.width(), 34), Qt.AlignCenter, head_guide_text(state))
        bar = QRectF(r.x(), r.y() + 42, r.width(), 8)
        span = max(state["hi"] * 1.5, 1.0)  # la barra llega a 1.5 x el máximo de la zona
        p.setPen(Qt.NoPen)
        p.setBrush(BORDER)
        p.drawRoundedRect(bar, 4, 4)
        zone_col = QColor(OK_GREEN)
        zone_col.setAlpha(110)
        p.setBrush(zone_col)
        p.drawRect(QRectF(bar.x() + bar.width() * state["lo"] / span, bar.y(), bar.width() * (state["hi"] - state["lo"]) / span, bar.height()))
        x = bar.x() + bar.width() * min(state["value"], span) / span
        p.setBrush(color)
        p.drawEllipse(QPointF(x, bar.center().y()), 7, 7)


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
def error_source_text(
    source: str,
    cv_px: Optional[float] = None,
    points: int = 0,
    raw_px: Optional[float] = None,
    skipped_px: Optional[float] = None,
) -> str:
    """Explica qué error se muestra al final de la calibración. `skipped_px`: error que habría dado la
    corrección automática cuando no se aplicó porque no mejoraba."""
    if source == "validation_corrected":
        text = (
            f"Medido en {points} puntos de validación nuevos tras corregir automáticamente el desvío común "
            "(cada punto se evalúa con la corrección calculada con los demás)."
        )
        if raw_px is not None:
            text += f" Sin corregir: ~{int(round(raw_px))} px."
        return text
    if source == "validation":
        text = f"Medido en {points} puntos de validación nuevos, no usados para calibrar (error real)."
        if skipped_px is not None:
            text += (
                f" No se aplicó la corrección automática del desvío común: no mejoraba (~{int(round(skipped_px))} px)."
            )
        if cv_px is not None:
            text += f" Validación cruzada de los puntos de calibración: ~{int(round(cv_px))} px."
        return text
    if source == "cv":
        return "Estimado con validación cruzada (dejando un punto fuera); no hubo puntos de validación suficientes."
    return "Error de ajuste sobre los propios puntos de calibración (optimista)."


EYE_LABELS = {"both": "los dos ojos", "right": "tu ojo derecho", "left": "tu ojo izquierdo"}
EYE_SHORT = {"both": "ambos", "right": "derecho", "left": "izquierdo"}


def eye_selection_text(selection: Optional[dict]) -> str:
    """Qué ojo se usa y por qué (metrics["eye_selection"] de fit_session); vacío sin datos."""
    if not selection or selection.get("chosen") not in EYE_LABELS:
        return ""
    chosen = selection["chosen"]
    if selection.get("mode") != "auto":
        return f"Ojo usado: {EYE_LABELS[chosen]} (fijado en config.json, tracking.eye)."
    cands = selection.get("candidates") or {}
    parts = [
        f"{EYE_SHORT[e]} {int(round(c['jitter_px']))} px de temblor / {int(round(c['error_px']))} px de error"
        for e, c in cands.items()
        if e in EYE_SHORT and math.isfinite(c.get("jitter_px", math.nan)) and math.isfinite(c.get("error_px", math.nan))
    ]
    text = f"Ojo usado: {EYE_LABELS[chosen]}, elegido por tener el menor temblor entre los de error parecido."
    if parts:
        text += " Comparación: " + "; ".join(parts) + "."
    return text


class CalibrationResultDialog(QDialog):
    """"Calibración completada. Error estimado: ~N px" con Aceptar / Repetir, de dónde sale el error y el ojo usado."""

    def __init__(
        self,
        error_px: float,
        parent=None,
        source: str = "",
        cv_px: Optional[float] = None,
        points: int = 0,
        raw_px: Optional[float] = None,
        skipped_px: Optional[float] = None,
        eye_text: str = "",
    ):
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
        self.source_label = QLabel(error_source_text(source, cv_px, points, raw_px, skipped_px) if source else "")
        self.source_label.setObjectName("description")
        self.source_label.setWordWrap(True)
        self.source_label.setAccessibleName("Origen del error estimado")
        self.source_label.setVisible(bool(source))
        layout.addWidget(self.source_label)
        self.eye_label = QLabel(eye_text)
        self.eye_label.setObjectName("description")
        self.eye_label.setWordWrap(True)
        self.eye_label.setAccessibleName("Ojo usado para seguir la mirada")
        self.eye_label.setVisible(bool(eye_text))
        layout.addWidget(self.eye_label)
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


def default_tracker_factory(camera_index: int, camera_size=None, camera_exposure: str = "auto", gaze_model=None):
    from eye_tracker import TrackerThread

    # camera_size None = resolución nativa; camera_exposure "auto" o "fast" (config.json camera.exposure)
    return TrackerThread(camera_index, camera_size=camera_size, camera_exposure=camera_exposure, gaze_model=gaze_model)


def camera_size_from_config(config: dict):
    """None = resolución nativa (por defecto); (ancho, alto) si camera.resolution es "fixed"."""
    from config_manager import camera_size

    return camera_size(config)


def guidance_text(face_ok: bool, hint: str, interocular: float) -> str:
    """Aviso destacado del modal antes de calibrar (vacío si la posición es buena)."""
    from calibration import IOD_MIN

    if not face_ok:
        return "No se detecta tu rostro: colócate frente a la cámara con buena luz."
    if hint == "Acércate":
        return (
            f"Acércate a la cámara: tu rostro se ve pequeño (distancia entre ojos {interocular:.2f}, "
            f"se recomienda {IOD_MIN:.2f} o más) y el cursor temblará más."
        )
    if hint == "Aléjate":
        return "Aléjate un poco de la cámara: tu rostro no cabe bien en la guía."
    if hint:
        return "Centra tu rostro dentro de la guía."
    return ""


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
        text += f". Menos de {MIN_CAMERA_FPS:.0f} fps: la detección puede fallar; mejora la iluminación"
        if "exposición rápida" not in info:
            # algunas cámaras alargan la exposición aunque haya luz (TE-9072: ~21 fps con el cuarto iluminado)
            text += ' o pon "exposure": "fast" en la sección camera de config.json (imagen más oscura, 30 fps)'
        text += "."
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
        from config_manager import gaze_model

        self.gaze_model = gaze_model(config)  # tracking.gaze_model: red de mirada del híbrido (None = solo iris)
        if tracker_factory is default_tracker_factory:
            from config_manager import camera_exposure

            size = camera_size_from_config(config)
            exposure = camera_exposure(config)
            net = self.gaze_model
            tracker_factory = lambda i: default_tracker_factory(i, size, exposure, net)  # noqa: E731
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

        self.guidance_label = QLabel("")
        self.guidance_label.setObjectName("guidance")
        self.guidance_label.setWordWrap(True)
        self.guidance_label.setStyleSheet("color: #FFB020; font-weight: bold; background: transparent;")
        self.guidance_label.setAccessibleName("Aviso de posición frente a la cámara")
        layout.addWidget(self.guidance_label)

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
            self.guidance_label.setText(guidance_text(self.preview.face_ok, self.preview.hint_text, frame.interocular))
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
            # orden balanceado: un movimiento de cabeza no queda ligado a la fila/columna del punto
            # --record-session añade la fase de movimiento de cabeza (datos para compensarla más adelante)
            head = HEAD_MOTION_STEPS if self.record_dir else None
            session = CalibrationSession(
                geom["width"],
                geom["height"],
                margin=CALIBRATION_MARGIN,
                validation=val,
                order="balanced",
                head_motion=head,
                wait_for_start=True,  # Enter antes de cada pantalla: tiempo para leer la instrucción
            )
            logger.info("Orden de los puntos de calibración: %s", session.order)
            self.recorder = self._make_recorder(screen, geom, session) if self.record_dir else None
            from calibration import gaze_fit_options
            from config_manager import eye_mode

            scr = CalibrationScreen(
                session,
                dpr=float(screen.devicePixelRatio()),
                fit_options=gaze_fit_options(self.gaze_model),
                clock=self.clock,
                recorder=self.recorder,
                eye_mode=eye_mode(self.config),
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
        from neural_gaze import encode_record_crop

        return SessionRecorder(meta, face_crop_fn=encode_record_crop)  # recortes de la cara para evaluar redes

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
            raw_px=mapper.metrics.get("validation_px"),
            skipped_px=mapper.metrics.get("validation_corrected_px") if mapper.metrics.get("auto_offset_skipped") else None,
            eye_text=eye_selection_text(mapper.metrics.get("eye_selection")),
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
