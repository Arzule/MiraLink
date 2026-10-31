"""Ventana de configuración de MiraLink en PySide6."""

from __future__ import annotations

import logging
import sys
from typing import Optional

from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtWidgets import (
    QApplication,
    QAbstractSpinBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from config_manager import camera_exposure, camera_size, load_config, save_config, save_selected_eye
from theme_manager import list_themes, load_theme

logger = logging.getLogger(__name__)


class NoWheelSlider(QSlider):
    """Slider que ignora la rueda del mouse: la rueda desplaza la ventana y no cambia el valor por accidente."""

    def wheelEvent(self, event):
        event.ignore()  # pasa al QScrollArea


class NoWheelSpinBox(QSpinBox):
    def wheelEvent(self, event):
        event.ignore()


class NoWheelComboBox(QComboBox):
    def wheelEvent(self, event):
        event.ignore()


class ConfigWindow(QMainWindow):
    calibrate_requested = Signal()
    config_saved = Signal(dict)
    config_changed = Signal(dict)

    def __init__(self, config_path: Optional[str] = None, dry_run: bool = False):
        super().__init__()
        self.config_path = config_path
        self.dry_run = dry_run
        self.control_session = None
        # Fábrica de la sesión de control (los tests inyectan una con piezas falsas)
        self.control_session_factory = self._default_control_session
        self._config = load_config(config_path)
        self._save_label_timer = QTimer(self)
        self._save_label_timer.setSingleShot(True)
        self._save_label_timer.timeout.connect(self._clear_save_label)
        # Fábrica del modal de calibración (los tests inyectan una con tracker falso)
        self.calibration_dialog_factory = self._default_calibration_dialog
        self.calibration_dialog_kwargs = {}  # inyección de piezas falsas en tests de integración
        self.calibration_dialog = None  # referencia fuerte al modal abierto

        self.setWindowTitle("MiraLink - Configuración")
        self.setMinimumSize(960, 720)

        central = QWidget(self)
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)

        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.NoFrame)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        root_layout.addWidget(self.scroll_area)

        self.content = QWidget(self.scroll_area)
        self.scroll_layout = QVBoxLayout(self.content)
        self.scroll_layout.setContentsMargins(24, 24, 24, 24)
        self.scroll_layout.setSpacing(24)
        self.scroll_area.setWidget(self.content)

        self._build_ui()
        self._apply_config()
        self._apply_theme()

        # Signals
        self.smooth_slider.valueChanged.connect(self._on_smooth_changed)
        self.smooth_spin.valueChanged.connect(self._on_smooth_spin_changed)
        self.dwell_slider.valueChanged.connect(self._on_dwell_changed)
        self.dwell_spin.valueChanged.connect(self._on_dwell_spin_changed)
        self.blink_click_slider.valueChanged.connect(self._on_blink_click_changed)
        self.blink_click_spin.valueChanged.connect(self._on_blink_click_spin_changed)
        self.blink_release_slider.valueChanged.connect(self._on_blink_release_changed)
        self.blink_release_spin.valueChanged.connect(self._on_blink_release_spin_changed)
        self.theme_combo.currentIndexChanged.connect(self._on_theme_changed)
        self.font_slider.valueChanged.connect(self._on_font_changed)
        self.font_spin.valueChanged.connect(self._on_font_spin_changed)
        self.btn_font_minus.clicked.connect(self._on_font_minus)
        self.btn_font_plus.clicked.connect(self._on_font_plus)
        self.btn_calibrate.clicked.connect(self.calibrate_requested.emit)
        self.calibrate_requested.connect(self._open_calibration)
        self.btn_control.clicked.connect(self._on_control_clicked)
        self.btn_recenter.clicked.connect(self._on_recenter_clicked)
        self.btn_save.clicked.connect(self._on_save_clicked)

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        self._build_header()
        self._build_cards()
        self._build_footer()

    def _build_header(self):
        header_frame = QFrame(self.content)
        header_layout = QHBoxLayout(header_frame)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(16)

        header_left = QWidget(header_frame)
        header_left_layout = QVBoxLayout(header_left)
        header_left_layout.setContentsMargins(0, 0, 0, 0)
        header_left_layout.setSpacing(8)

        self.title_label = QLabel("Configuración de Control Ocular", header_left)
        self.title_label.setObjectName("title")
        self.title_label.setAccessibleName("Título: Configuración de Control Ocular")
        header_left_layout.addWidget(self.title_label)

        self.subtitle_label = QLabel(
            "Configura los parámetros de mirada y parpadeo para el control del cursor en todo el sistema.",
            header_left,
        )
        self.subtitle_label.setObjectName("subtitle")
        self.subtitle_label.setWordWrap(True)
        self.subtitle_label.setAccessibleName("Subtítulo de configuración")
        header_left_layout.addWidget(self.subtitle_label)
        header_layout.addWidget(header_left, 1)

        self.status_badge = QLabel("RASTREADOR OCULAR INACTIVO", header_frame)
        self.status_badge.setObjectName("statusBadge")
        self.status_badge.setAccessibleName("Estado del rastreador ocular")
        self.status_badge.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.status_badge.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)
        header_layout.addWidget(self.status_badge, 0, Qt.AlignVCenter)

        self.scroll_layout.addWidget(header_frame)

    def _build_cards(self):
        grid = QGridLayout()
        grid.setSpacing(24)
        self.scroll_layout.addLayout(grid)

        self.card_smooth = self._create_card(
            "Suavizado del Cursor",
            "Reduce las vibraciones del movimiento natural del ojo. Valores más altos dan un cursor más estable pero con algo más de retraso.",
        )
        self.smooth_slider = self._create_slider(0, 100, 1, self.card_smooth)
        self.smooth_spin = self._create_spinbox(0, 100, 1, suffix=" %", parent=self.card_smooth)
        self._add_slider_spin(self.card_smooth, self.smooth_slider, self.smooth_spin, "0% - Ágil", "100% - Estable")
        grid.addWidget(self.card_smooth, 0, 0)

        self.card_dwell = self._create_card(
            "Tiempo de Fijación para Mantener",
            "Cuánto tiempo debe permanecer fija tu mirada para activar un arrastrar y soltar intencional o una pulsación larga secundaria.",
        )
        self.dwell_slider = self._create_slider(800, 3000, 50, self.card_dwell)
        self.dwell_spin = self._create_spinbox(800, 3000, 50, suffix=" ms", parent=self.card_dwell)
        self._add_slider_spin(self.card_dwell, self.dwell_slider, self.dwell_spin, "800 ms", "3000 ms")
        grid.addWidget(self.card_dwell, 0, 1)

        self.card_blink_click = self._create_card(
            "Duración Mínima de Parpadeo para Clic",
            "El intervalo mínimo con los ojos cerrados necesario para registrar un clic deliberado en vez de un parpadeo natural involuntario.",
        )
        self.blink_click_slider = self._create_slider(150, 500, 10, self.card_blink_click)
        self.blink_click_spin = self._create_spinbox(150, 500, 10, suffix=" ms", parent=self.card_blink_click)
        self._add_slider_spin(self.card_blink_click, self.blink_click_slider, self.blink_click_spin, "150 ms", "500 ms")
        grid.addWidget(self.card_blink_click, 1, 0)

        self.card_blink_release = self._create_card(
            "Duración Mínima de Parpadeo para Soltar",
            "Mantén los ojos cerrados durante este tiempo para desbloquear/soltar una acción de arrastre activa o un anclaje de ventana fijo.",
        )
        self.blink_release_slider = self._create_slider(150, 500, 10, self.card_blink_release)
        self.blink_release_spin = self._create_spinbox(150, 500, 10, suffix=" ms", parent=self.card_blink_release)
        self._add_slider_spin(self.card_blink_release, self.blink_release_slider, self.blink_release_spin, "150 ms", "500 ms")
        grid.addWidget(self.card_blink_release, 1, 1)

        # Apariencia: solo selector de tema (reutiliza el layout de la tarjeta)
        self.card_appearance = self._create_card(
            "Apariencia",
            "Elige el tema visual de la aplicación.",
        )
        app_layout = self.card_appearance.layout()
        theme_row = QHBoxLayout()
        theme_label = QLabel("Tema", self.card_appearance)
        theme_label.setObjectName("description")
        theme_label.setAccessibleName("Etiqueta tema")
        theme_row.addWidget(theme_label)
        self.theme_combo = NoWheelComboBox(self.card_appearance)
        self.theme_combo.setAccessibleName("Selector de tema")
        self.theme_combo.setMinimumHeight(48)
        self.theme_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        theme_row.addWidget(self.theme_combo, 1)
        app_layout.addLayout(theme_row)
        # Una sola celda (columna izquierda de la última fila) con la altura de su contenido
        self.card_appearance.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        grid.addWidget(self.card_appearance, 2, 0, Qt.AlignTop)

        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)

    def _build_footer(self):
        footer = QFrame(self.content)
        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(0, 0, 0, 0)
        footer_layout.setSpacing(16)

        # Left: dos filas de botones (no caben tres en 960 px junto al bloque de letra)
        left = QWidget(footer)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(12)

        row1 = QWidget(left)
        row1_layout = QHBoxLayout(row1)
        row1_layout.setContentsMargins(0, 0, 0, 0)
        row1_layout.setSpacing(16)

        self.btn_calibrate = QPushButton("Calibrar Rastreador Ocular", row1)
        self.btn_calibrate.setObjectName("btn_calibrate")
        self.btn_calibrate.setProperty("variant", "primary")
        self.btn_calibrate.setAccessibleName("Botón calibrar rastreador ocular")
        self.btn_calibrate.setMinimumHeight(48)
        row1_layout.addWidget(self.btn_calibrate)

        self.btn_save = QPushButton("Guardar Configuración", row1)
        self.btn_save.setObjectName("btn_save")
        self.btn_save.setProperty("variant", "secondary")
        self.btn_save.setAccessibleName("Botón guardar configuración")
        self.btn_save.setMinimumHeight(48)
        row1_layout.addWidget(self.btn_save)

        self.save_label = QLabel("", row1)
        self.save_label.setObjectName("description")
        self.save_label.setAccessibleName("Mensaje de guardado")
        row1_layout.addWidget(self.save_label)
        row1_layout.addStretch()
        left_layout.addWidget(row1)

        row2 = QWidget(left)
        row2_layout = QHBoxLayout(row2)
        row2_layout.setContentsMargins(0, 0, 0, 0)
        row2_layout.setSpacing(16)

        self.btn_control = QPushButton("Iniciar Control del Cursor", row2)
        self.btn_control.setObjectName("btn_control")
        self.btn_control.setProperty("variant", "primary")
        self.btn_control.setAccessibleName("Botón iniciar o pausar el control del cursor")
        self.btn_control.setMinimumHeight(48)
        row2_layout.addWidget(self.btn_control)

        self.btn_recenter = QPushButton("Recentrar", row2)
        self.btn_recenter.setObjectName("btn_recenter")
        self.btn_recenter.setProperty("variant", "secondary")
        self.btn_recenter.setAccessibleName("Botón recentrar la mirada (Ctrl+Alt+R)")
        self.btn_recenter.setToolTip("Corrige la deriva mirando un punto central (Ctrl+Alt+R)")
        self.btn_recenter.setMinimumHeight(48)
        row2_layout.addWidget(self.btn_recenter)

        self.control_label = QLabel("", row2)
        self.control_label.setObjectName("description")
        self.control_label.setWordWrap(True)
        self.control_label.setAccessibleName("Estado del control del cursor")
        row2_layout.addWidget(self.control_label, 1)
        left_layout.addWidget(row2)
        footer_layout.addWidget(left, 1)

        # Right: font size block
        font_block = QWidget(footer)
        font_block_layout = QVBoxLayout(font_block)
        font_block_layout.setContentsMargins(0, 0, 0, 0)
        font_block_layout.setSpacing(8)

        font_label = QLabel("Tamaño de letra", font_block)
        font_label.setObjectName("description")
        font_label.setAccessibleName("Etiqueta tamaño de letra")
        font_block_layout.addWidget(font_label)

        self.font_slider = self._create_slider(80, 200, 10, font_block)
        self.font_slider.setAccessibleName("Slider tamaño de letra")
        self.font_slider.setMinimumWidth(260)
        font_block_layout.addWidget(self.font_slider)

        font_controls = QHBoxLayout()
        font_controls.setSpacing(8)
        self.btn_font_minus = QPushButton("−", font_block)
        self.btn_font_minus.setObjectName("btn_font_minus")
        self.btn_font_minus.setProperty("variant", "secondary")
        self.btn_font_minus.setAccessibleName("Reducir tamaño de letra")
        self.btn_font_minus.setFixedSize(48, 48)
        font_controls.addWidget(self.btn_font_minus)

        self.font_spin = self._create_spinbox(80, 200, 10, suffix=" %", parent=font_block)
        self.font_spin.setAccessibleName("Caja numérica tamaño de letra")
        self.font_spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
        font_controls.addWidget(self.font_spin)

        self.btn_font_plus = QPushButton("+", font_block)
        self.btn_font_plus.setObjectName("btn_font_plus")
        self.btn_font_plus.setProperty("variant", "secondary")
        self.btn_font_plus.setAccessibleName("Aumentar tamaño de letra")
        self.btn_font_plus.setFixedSize(48, 48)
        font_controls.addWidget(self.btn_font_plus)
        font_controls.addStretch()
        font_block_layout.addLayout(font_controls)

        footer_layout.addWidget(font_block, 0, Qt.AlignTop | Qt.AlignRight)
        self.scroll_layout.addWidget(footer)
        # El espacio sobrante de la ventana queda vacío debajo del pie
        self.scroll_layout.addStretch(1)

    def _create_card(self, title: str, description: str) -> QFrame:
        card = QFrame(self.content)
        card.setObjectName("card")
        layout = QVBoxLayout(card)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 16)
        title_label = QLabel(title, card)
        title_label.setObjectName("title")
        title_label.setWordWrap(True)
        layout.addWidget(title_label)
        desc_label = QLabel(description, card)
        desc_label.setObjectName("description")
        desc_label.setWordWrap(True)
        layout.addWidget(desc_label)
        return card

    def _create_slider(self, minv, maxv, step, parent):
        slider = NoWheelSlider(Qt.Horizontal, parent)
        slider.setRange(minv, maxv)
        slider.setSingleStep(step)
        slider.setPageStep(step * 5)
        slider.setTickInterval(step * 10)
        slider.setTickPosition(QSlider.TicksBelow)
        slider.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        slider.setMinimumHeight(48)
        return slider

    def _create_spinbox(self, minv, maxv, step, suffix="", parent=None):
        spin = NoWheelSpinBox(parent)
        spin.setRange(minv, maxv)
        spin.setSingleStep(step)
        spin.setSuffix(suffix)
        spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
        spin.setMinimumHeight(48)
        return spin

    def _add_slider_spin(self, card, slider, spin, label_min, label_max):
        layout = card.layout()
        row = QHBoxLayout()
        lbl_min = QLabel(label_min, card)
        lbl_min.setObjectName("description")
        row.addWidget(lbl_min)
        row.addStretch()
        lbl_max = QLabel(label_max, card)
        lbl_max.setObjectName("description")
        row.addWidget(lbl_max)
        layout.addWidget(slider)
        layout.addWidget(spin)
        layout.addLayout(row)

    # ---------------------------------------------------------------- config
    def _apply_config(self):
        inter = self._config.get("interaction", {})
        ui = self._config.get("ui", {})

        self.smooth_slider.setValue(int(inter.get("smoothing_percent", 40)))
        self.dwell_slider.setValue(int(inter.get("dwell_hold_ms", 1200)))
        self.blink_click_slider.setValue(int(inter.get("blink_click_ms", 250)))
        self.blink_release_slider.setValue(int(inter.get("blink_release_ms", 300)))
        # Las señales aún no están conectadas: sincronizar las cajas numéricas con los sliders
        self.smooth_spin.setValue(self.smooth_slider.value())
        self.dwell_spin.setValue(self.dwell_slider.value())
        self.blink_click_spin.setValue(self.blink_click_slider.value())
        self.blink_release_spin.setValue(self.blink_release_slider.value())

        self.theme_combo.blockSignals(True)
        self.theme_combo.clear()
        themes = list_themes()
        current_theme = ui.get("theme", "default")
        idx = 0
        for i, t in enumerate(themes):
            self.theme_combo.addItem(t["name"], t["id"])
            if t["id"] == current_theme:
                idx = i
        self.theme_combo.setCurrentIndex(idx)
        self.theme_combo.blockSignals(False)

        font_scale = float(ui.get("font_scale", 1.0))
        font_pct = int(round(font_scale * 100))
        self.font_slider.setValue(font_pct)
        self.font_spin.setValue(font_pct)

    def _apply_theme(self):
        ui = self._config.get("ui", {})
        theme_id = ui.get("theme", "default")
        font_scale = float(ui.get("font_scale", 1.0))
        qss = load_theme(theme_id, font_scale)
        self.setStyleSheet(qss)

    def set_tracker_active(self, active: bool):
        if active:
            self.status_badge.setText("RASTREADOR OCULAR ACTIVO")
        else:
            self.status_badge.setText("RASTREADOR OCULAR INACTIVO")

    def _default_calibration_dialog(self):
        from calibration_ui import CalibrationDialog

        return CalibrationDialog(self._config, parent=self, **self.calibration_dialog_kwargs)

    def _open_calibration(self):
        """Abre el modal de calibración SIN exec(), reteniendo la referencia hasta `finished`.

        Con exec(), el hide() del modal al iniciar la calibración terminaba exec(), la variable local
        se soltaba y PySide destruía el modal y la pantalla completa (ver AGENTS.md).
        """
        logger.info("Clic en 'Calibrar Rastreador Ocular'")
        if self.calibration_dialog is not None:
            self.calibration_dialog.raise_()
            return
        self._stop_control()  # la cámara no puede compartirse con la sesión de control
        dialog = self.calibration_dialog_factory()
        dialog.camera_selected.connect(self._on_camera_selected)
        dialog.finished.connect(self._on_calibration_finished)
        self.calibration_dialog = dialog
        dialog.open()

    def _on_calibration_finished(self, _result=None):
        dialog, self.calibration_dialog = self.calibration_dialog, None
        if dialog is None:
            return
        logger.info("Modal de calibración cerrado")
        mapper = getattr(dialog, "accepted_mapper", None)
        if mapper is not None:
            self.save_label.setText("Calibración guardada")
            self._save_label_timer.start(2500)
            self._store_selected_eye(getattr(mapper, "eye", None))
        if hasattr(dialog, "deleteLater"):
            dialog.deleteLater()

    def _store_selected_eye(self, eye):
        """Guarda en config.json (tracking.selected_eye) el ojo de la calibración aceptada, sin guardar de
        paso otros cambios de la ventana; también lo refleja en la configuración en memoria."""
        if eye is None:
            return
        try:
            save_selected_eye(eye, self.config_path)
        except (OSError, ValueError) as e:
            logger.exception("No se pudo guardar el ojo elegido en config.json")
            self.save_label.setText(f"Calibración guardada, pero no se pudo escribir el ojo en config.json: {e}")
            self._save_label_timer.start(4000)
            return
        self._config.setdefault("tracking", {})["selected_eye"] = eye
        logger.info("Ojo elegido guardado en config.json: %s", eye)

    def _on_camera_selected(self, index):
        self._config.setdefault("camera", {})["device_index"] = int(index)
        self._emit_config_changed()

    # ------------------------------------------------------------ control del cursor
    def _default_control_session(self):
        from control_session import ControlSession

        return ControlSession(
            config_getter=lambda: self._config.get("interaction", {}),
            dry_run=self.dry_run,
            camera_index_getter=lambda: int(self._config.get("camera", {}).get("device_index", 0)),
            camera_size_getter=lambda: camera_size(self._config),
            camera_exposure_getter=lambda: camera_exposure(self._config),
            overlay_config_getter=lambda: self._config.get("overlay", {}),
        )

    def _on_control_clicked(self):
        """Iniciar el control (valida calibración, escala y atajo) o alternar la pausa."""
        session = self.control_session
        if session is not None and session.running:
            session.toggle_pause()
            return
        session = self.control_session_factory()
        session.paused_changed.connect(self._on_paused_changed)
        session.tracker_active_changed.connect(self.set_tracker_active)
        session.failed.connect(self._on_control_failed)
        if hasattr(session, "recentered"):
            session.recentered.connect(self.control_label.setText)
        if hasattr(session, "pose_warning"):
            session.pose_warning.connect(self._on_pose_warning)
        if hasattr(session, "camera_status"):
            session.camera_status.connect(self._on_camera_status)
        if hasattr(session, "overlay_message"):
            session.overlay_message.connect(self.control_label.setText)
        ok, message = session.start()
        if not ok:
            self.control_label.setText(message)
            session.deleteLater() if hasattr(session, "deleteLater") else None
            return
        self.control_session = session
        if message:
            self.control_label.setText(message)

    def _on_recenter_clicked(self):
        """Recentrar usa la sesión de control en marcha (un solo dueño de la cámara)."""
        logger.info("Clic en 'Recentrar'")
        session = self.control_session
        if session is None or not session.running or not hasattr(session, "recenter"):
            self.control_label.setText("Inicia el control del cursor para usar Recentrar (Ctrl+Alt+R).")
            return
        _ok, message = session.recenter()
        if message:
            self.control_label.setText(message)

    def _on_pose_warning(self, message: str):
        """Deriva de la cabeza: aviso en la etiqueta de estado; al quitarse vuelve el texto normal."""
        if message:
            self.control_label.setText(message)
        elif self.control_session is not None:
            self._on_paused_changed(bool(getattr(self.control_session, "paused", False)))

    def _on_camera_status(self, text: str):
        """Resolución y fps reales de la cámara: se añaden al texto de estado del control."""
        self._camera_status = text
        session = self.control_session
        if session is not None and not getattr(session, "_pose_warning_shown", False):
            self._on_paused_changed(bool(getattr(session, "paused", False)))

    def _on_paused_changed(self, paused: bool):
        self.btn_control.setText("Iniciar Control del Cursor" if paused else "Pausar Control del Cursor")
        suffix = " (modo simulación: no se mueve el cursor real)" if self.dry_run else ""
        camera = getattr(self, "_camera_status", "")
        if camera:
            suffix += f" {camera}."
        if paused:
            self.control_label.setText(f"Control pausado. Ctrl+Alt+P o el botón lo reanudan.{suffix}")
        else:
            self.control_label.setText(
                f"Control activo. Ctrl+Alt+P pausa en cualquier momento; Ctrl+Alt+R recentra.{suffix}"
            )

    def _on_control_failed(self, message: str):
        self.btn_control.setText("Iniciar Control del Cursor")
        self.control_label.setText(message)
        self.control_session = None

    def _stop_control(self):
        self._camera_status = ""
        session, self.control_session = self.control_session, None
        if session is not None:
            session.stop()
        self.btn_control.setText("Iniciar Control del Cursor")
        self.set_tracker_active(False)

    def closeEvent(self, event):
        """Al cerrar se detienen el tracker y el control y se libera el mouse."""
        if self.calibration_dialog is not None:
            self.calibration_dialog.reject()  # detiene su tracker y libera la cámara
        self._stop_control()
        super().closeEvent(event)

    def _clear_save_label(self):
        self.save_label.setText("")

    def _emit_config_changed(self):
        self.config_changed.emit(self._config)

    # --------------------------------------------------------------- handlers
    def _on_smooth_changed(self, v):
        self.smooth_spin.blockSignals(True)
        self.smooth_spin.setValue(v)
        self.smooth_spin.blockSignals(False)
        self._config.setdefault("interaction", {})["smoothing_percent"] = int(v)
        self._emit_config_changed()

    def _on_smooth_spin_changed(self, v):
        self.smooth_slider.blockSignals(True)
        self.smooth_slider.setValue(v)
        self.smooth_slider.blockSignals(False)
        self._config.setdefault("interaction", {})["smoothing_percent"] = int(v)
        self._emit_config_changed()

    def _on_dwell_changed(self, v):
        self.dwell_spin.blockSignals(True)
        self.dwell_spin.setValue(v)
        self.dwell_spin.blockSignals(False)
        self._config.setdefault("interaction", {})["dwell_hold_ms"] = int(v)
        self._emit_config_changed()

    def _on_dwell_spin_changed(self, v):
        self.dwell_slider.blockSignals(True)
        self.dwell_slider.setValue(v)
        self.dwell_slider.blockSignals(False)
        self._config.setdefault("interaction", {})["dwell_hold_ms"] = int(v)
        self._emit_config_changed()

    def _on_blink_click_changed(self, v):
        self.blink_click_spin.blockSignals(True)
        self.blink_click_spin.setValue(v)
        self.blink_click_spin.blockSignals(False)
        self._config.setdefault("interaction", {})["blink_click_ms"] = int(v)
        self._emit_config_changed()

    def _on_blink_click_spin_changed(self, v):
        self.blink_click_slider.blockSignals(True)
        self.blink_click_slider.setValue(v)
        self.blink_click_slider.blockSignals(False)
        self._config.setdefault("interaction", {})["blink_click_ms"] = int(v)
        self._emit_config_changed()

    def _on_blink_release_changed(self, v):
        self.blink_release_spin.blockSignals(True)
        self.blink_release_spin.setValue(v)
        self.blink_release_spin.blockSignals(False)
        self._config.setdefault("interaction", {})["blink_release_ms"] = int(v)
        self._emit_config_changed()

    def _on_blink_release_spin_changed(self, v):
        self.blink_release_slider.blockSignals(True)
        self.blink_release_slider.setValue(v)
        self.blink_release_slider.blockSignals(False)
        self._config.setdefault("interaction", {})["blink_release_ms"] = int(v)
        self._emit_config_changed()

    def _on_theme_changed(self, _idx):
        theme_id = self.theme_combo.currentData()
        if theme_id:
            self._config.setdefault("ui", {})["theme"] = theme_id
        self._apply_theme()
        self._emit_config_changed()

    def _on_font_changed(self, v):
        self.font_spin.blockSignals(True)
        self.font_spin.setValue(v)
        self.font_spin.blockSignals(False)
        self._config.setdefault("ui", {})["font_scale"] = float(v / 100.0)
        self._apply_theme()
        self._emit_config_changed()

    def _on_font_spin_changed(self, v):
        self.font_slider.blockSignals(True)
        self.font_slider.setValue(v)
        self.font_slider.blockSignals(False)
        self._config.setdefault("ui", {})["font_scale"] = float(v / 100.0)
        self._apply_theme()
        self._emit_config_changed()

    def _on_font_minus(self):
        v = self.font_slider.value()
        nv = max(80, v - 10)
        if nv != v:
            self.font_slider.setValue(nv)

    def _on_font_plus(self):
        v = self.font_slider.value()
        nv = min(200, v + 10)
        if nv != v:
            self.font_slider.setValue(nv)

    def _on_save_clicked(self):
        try:
            saved = save_config(self._config, self.config_path)
            self._config = saved
            self.save_label.setText("Configuración guardada")
            self._save_label_timer.start(2500)
            self.config_saved.emit(saved)
        except Exception as e:
            self.save_label.setText(f"Error al guardar: {e}")
            self._save_label_timer.start(4000)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = ConfigWindow()
    win.show()
    sys.exit(app.exec())
