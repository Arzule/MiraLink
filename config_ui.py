"""Ventana de configuración de MiraLink en PySide6."""

from __future__ import annotations

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

from config_manager import load_config, save_config
from theme_manager import list_themes, load_theme

# TODO(etapa 1)


class ConfigWindow(QMainWindow):
    calibrate_requested = Signal()
    config_saved = Signal(dict)
    config_changed = Signal(dict)

    def __init__(self, config_path: Optional[str] = None):
        super().__init__()
        self.config_path = config_path
        self._config = load_config(config_path)
        self._save_label_timer = QTimer(self)
        self._save_label_timer.setSingleShot(True)
        self._save_label_timer.timeout.connect(self._clear_save_label)

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
        self.theme_combo = QComboBox(self.card_appearance)
        self.theme_combo.setAccessibleName("Selector de tema")
        self.theme_combo.setMinimumHeight(48)
        self.theme_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        theme_row.addWidget(self.theme_combo, 1)
        app_layout.addLayout(theme_row)
        app_layout.addStretch()
        grid.addWidget(self.card_appearance, 2, 0)

        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)

    def _build_footer(self):
        footer = QFrame(self.content)
        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(0, 0, 0, 0)
        footer_layout.setSpacing(16)

        # Left: buttons + save status
        left = QWidget(footer)
        left_layout = QHBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(16)

        self.btn_calibrate = QPushButton("Calibrar Rastreador Ocular", left)
        self.btn_calibrate.setObjectName("btn_calibrate")
        self.btn_calibrate.setProperty("variant", "primary")
        self.btn_calibrate.setAccessibleName("Botón calibrar rastreador ocular")
        self.btn_calibrate.setMinimumHeight(48)
        left_layout.addWidget(self.btn_calibrate)

        self.btn_save = QPushButton("Guardar Configuración", left)
        self.btn_save.setObjectName("btn_save")
        self.btn_save.setProperty("variant", "secondary")
        self.btn_save.setAccessibleName("Botón guardar configuración")
        self.btn_save.setMinimumHeight(48)
        left_layout.addWidget(self.btn_save)

        self.save_label = QLabel("", left)
        self.save_label.setObjectName("description")
        self.save_label.setAccessibleName("Mensaje de guardado")
        left_layout.addWidget(self.save_label)
        left_layout.addStretch()
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
        slider = QSlider(Qt.Horizontal, parent)
        slider.setRange(minv, maxv)
        slider.setSingleStep(step)
        slider.setPageStep(step * 5)
        slider.setTickInterval(step * 10)
        slider.setTickPosition(QSlider.TicksBelow)
        slider.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        slider.setMinimumHeight(48)
        return slider

    def _create_spinbox(self, minv, maxv, step, suffix="", parent=None):
        spin = QSpinBox(parent)
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
