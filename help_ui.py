"""Ayuda de MiraLink: guía completa de uso (ventana "Ayuda" de la configuración).

Complementa los avisos que la aplicación ya muestra durante el uso. El texto se arma con la
configuración actual (umbrales, atajos y modos), así los números coinciden con lo que hará el programa.
Ventana no modal con un índice de secciones (ítems de 48 px, se puede usar con la mirada) y el texto en
un QTextBrowser; hereda el tema de la ventana de configuración (los colores vienen del .qss).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSizePolicy,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from config_manager import DEFAULT_CONFIG

HELP_TITLE = "Ayuda de MiraLink"


def _inter(config: Optional[dict]) -> Dict:
    inter = dict(DEFAULT_CONFIG["interaction"])
    if isinstance(config, dict) and isinstance(config.get("interaction"), dict):
        inter.update({k: v for k, v in config["interaction"].items() if v is not None})
    return inter


def _fmt_s(ms) -> str:
    """1500 -> "1,5 s"; 2000 -> "2 s"."""
    s = float(ms) / 1000.0
    return (f"{s:.1f}".rstrip("0").rstrip(".") + " s").replace(".", ",")


def help_sections(config: Optional[dict] = None) -> List[Tuple[str, str, str]]:
    """(id, título, HTML) de cada sección, con los valores de la configuración actual."""
    from cursor_control import HOTKEY_LABEL, RECENTER_HOTKEY_LABEL

    i = _inter(config)
    pause, recenter = HOTKEY_LABEL, RECENTER_HOTKEY_LABEL
    click = int(i["blink_click_ms"])
    release = int(i["blink_release_ms"])
    blink_max = int(i["blink_max_ms"])
    dwell = int(i["dwell_hold_ms"])
    mode_n = int(i["mode_blinks"])
    mode_win = _fmt_s(i["mode_blink_window_ms"])
    sleep = _fmt_s(i["sleep_close_ms"])
    wake = _fmt_s(i["wake_close_ms"])
    dz = f"{float(i['head_deadzone_deg']):g}"
    start_mode = "cabeza" if i.get("pointer_mode") == "hybrid" else "mirada"
    hold_timeout = int(i["hold_timeout_s"])

    return [
        ("inicio", "Qué es MiraLink", f"""
<p>MiraLink mueve el cursor con la <b>mirada</b> y hace clic con un <b>parpadeo</b>, usando solo la webcam.</p>
<ol>
<li><b>Calibra</b>: mira unos puntos en la pantalla.</li>
<li><b>Inicia el control del cursor</b>.</li>
<li>Usa el ordenador con la mirada, los parpadeos y la cabeza.</li>
</ol>
<p><b>{pause}</b> pausa o reanuda el control en cualquier momento.</p>
"""),
        ("preparar", "Antes de empezar", """
<ul>
<li>Cierra otras apps que usen la cámara (por ejemplo, la app Cámara).</li>
<li>Ilumina tu cara de frente; evita tener una ventana o lámpara detrás.</li>
<li>Siéntate cómodo, con la cara centrada frente a la cámara.</li>
<li>Mantén la misma postura después de calibrar.</li>
</ul>
"""),
        ("calibrar", "Calibrar", """
<ol>
<li>Pulsa <b>Calibrar Rastreador Ocular</b>. Colócate hasta que la vista previa no muestre avisos y pulsa
<b>Iniciar Calibración</b>.</li>
<li>Pulsa <b>Enter</b> y mira cada punto, sin mover la cabeza, hasta que su círculo se cierre.</li>
<li>Pulsa <b>Enter</b> otra vez y mira los puntos de comprobación.</li>
<li>Pulsa <b>Aceptar</b>, o <b>Repetir</b> si el error sale alto.</li>
</ol>
<p>Para salir, mantén <b>ESC</b>. Vuelve a calibrar si cambias de lugar, de pantalla o de luz.</p>
"""),
        ("control", "Iniciar, pausar y recentrar", f"""
<ul>
<li><b>Iniciar Control del Cursor</b>: el cursor sigue tu mirada. El mismo botón lo pausa.</li>
<li><b>{pause}</b>: pausa o reanuda desde cualquier programa.</li>
<li><b>Recentrar</b> o <b>{recenter}</b>: si el cursor queda siempre corrido, mira el punto central un
segundo.</li>
</ul>
"""),
        ("modos", "Modo mirada y modo cabeza", f"""
<p>El control empieza en <b>modo {start_mode}</b>.</p>
<ul>
<li><b>Modo mirada:</b> el cursor va a donde miras. Clic con parpadeo.</li>
<li><b>Modo cabeza:</b> el cursor salta a donde miras y se queda quieto. Para afinar, gira un poco la
cabeza (más de {dz}°): el cursor se desliza hacia ese lado. El centro es la posición de tu cabeza al entrar
en este modo: vuelve a ella para detenerlo.</li>
<li><b>Cambiar de modo:</b> {mode_n} parpadeos normales en menos de {mode_win}.</li>
</ul>
"""),
        ("clic", "Clic y arrastrar", f"""
<ul>
<li><b>Clic:</b> cierra los ojos al menos <b>{click} ms</b> (un parpadeo algo largo). Los parpadeos normales se
ignoran.</li>
<li><b>Mantener (modo cabeza):</b> deja el cursor totalmente quieto <b>{_fmt_s(dwell)}</b>. Luego mueve el
cursor para arrastrar.</li>
<li><b>Soltar:</b> cierra los ojos al menos <b>{release} ms</b>. Se suelta solo a los {hold_timeout} s.</li>
</ul>
"""),
        ("reposo", "Reposo", f"""
<ul>
<li><b>Ojos cerrados {sleep}:</b> el cursor se queda quieto, para leer o ver un video.</li>
<li><b>Ojos cerrados {wake}:</b> el control vuelve.</li>
</ul>
<p>Mirar mucho rato hacia abajo (al teclado) también puede activar el reposo.</p>
"""),
        ("botones", "Botones flotantes", """
<p>En el escritorio aparecen botones grandes a la derecha:</p>
<ul>
<li><b>Lupa 200%:</b> amplía la pantalla. Mira un borde para moverte. Púlsalo otra vez para quitarla.</li>
<li><b>Teclado:</b> abre o cierra el teclado en pantalla.</li>
<li><b>Filtro de color:</b> activa o quita el filtro de color de Windows.</li>
</ul>
<p>Sobre otras apps solo queda <b>Ir al escritorio</b>.</p>
"""),
        ("ajustes", "Ajustes", """
<ul>
<li><b>Suavizado:</b> más alto, cursor más estable pero más lento.</li>
<li><b>Tiempo de fijación:</b> tiempo quieto para mantener el clic.</li>
<li><b>Parpadeo para clic:</b> súbelo si hay clics sin querer.</li>
<li><b>Parpadeo para soltar:</b> tiempo para soltar un clic mantenido.</li>
<li><b>Tema</b> y <b>tamaño de letra</b>.</li>
</ul>
<p>Los cambios se aplican al momento. Pulsa <b>Guardar Configuración</b> para conservarlos.</p>
"""),
        ("avanzado", "Ajustes avanzados", """
<p>En <b>config.json</b>, junto al programa, dentro de <b>"interaction"</b>:</p>
<ul>
<li><b>pointer_mode:</b> modo al iniciar ("absolute" mirada, "hybrid" cabeza).</li>
<li><b>head_speed_px_s_per_deg:</b> velocidad del cursor con la cabeza.</li>
<li><b>head_deadzone_deg:</b> giro de cabeza que se ignora.</li>
<li><b>sleep_close_ms</b> y <b>wake_close_ms:</b> tiempos del reposo.</li>
</ul>
"""),
        ("atajos", "Atajos y gestos", f"""
<table cellpadding="6">
<tr><td><b>{pause}</b></td><td>Pausar o reanudar</td></tr>
<tr><td><b>{recenter}</b></td><td>Recentrar</td></tr>
<tr><td><b>Parpadeo de {click} ms</b></td><td>Clic</td></tr>
<tr><td><b>Cursor quieto {_fmt_s(dwell)}</b></td><td>Mantener (modo cabeza)</td></tr>
<tr><td><b>Parpadeo de {release} ms</b></td><td>Soltar</td></tr>
<tr><td><b>{mode_n} parpadeos normales en {mode_win}</b></td><td>Cambiar de modo</td></tr>
<tr><td><b>Ojos cerrados {sleep}</b></td><td>Reposo</td></tr>
<tr><td><b>Ojos cerrados {wake}</b></td><td>Salir del reposo</td></tr>
<tr><td><b>Enter / mantener ESC</b></td><td>Empezar / salir de la calibración</td></tr>
</table>
"""),
        ("problemas", "Problemas frecuentes", f"""
<ul>
<li><b>Cursor corrido:</b> Recentrar ({recenter}). Si sigue, calibra otra vez.</li>
<li><b>Cursor tembloroso:</b> acércate, mejora la luz o usa el modo cabeza.</li>
<li><b>Clics sin querer:</b> sube el parpadeo para clic.</li>
<li><b>El cursor no se mueve:</b> puede estar en pausa ({pause}) o en reposo (ojos cerrados {wake}).</li>
<li><b>La cámara no abre:</b> cierra otras apps que la usen.</li>
<li><b>No veo los botones:</b> solo salen en el escritorio; usa Ir al escritorio.</li>
</ul>
"""),
    ]


def help_html(config: Optional[dict] = None) -> str:
    """Toda la guía en un documento con anclas por sección."""
    parts = []
    for sid, title, body in help_sections(config):
        parts.append(f'<a name="{sid}"></a><h2>{title}</h2>{body}')
    return "<html><body>" + "\n".join(parts) + "</body></html>"


DIALOG_STYLE = """
QListWidget#helpIndex { border: none; outline: 0; }
QListWidget#helpIndex::item { min-height: 48px; padding: 4px 12px; border-radius: 8px; border: 2px solid transparent; }
QListWidget#helpIndex::item:selected { border: 2px solid palette(text); }
QTextBrowser#helpText { border: none; padding: 8px; }
"""


class HelpDialog(QDialog):
    """Ventana de ayuda no modal: índice a la izquierda y guía a la derecha."""

    def __init__(self, config: Optional[dict] = None, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle(HELP_TITLE)
        self.setAccessibleName("Ventana de ayuda de MiraLink")
        self.setModal(False)
        self.resize(980, 720)
        self.setMinimumSize(720, 520)
        self.setStyleSheet(DIALOG_STYLE)
        self.sections = help_sections(config)

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 24, 24, 24)
        root.setSpacing(16)

        title = QLabel(HELP_TITLE, self)
        title.setObjectName("title")
        title.setAccessibleName("Título: Ayuda de MiraLink")
        root.addWidget(title)
        subtitle = QLabel("Guía completa de uso. Elige una sección o desplázate por el texto.", self)
        subtitle.setObjectName("subtitle")
        subtitle.setWordWrap(True)
        subtitle.setAccessibleName("Subtítulo de la ayuda")
        root.addWidget(subtitle)

        body = QWidget(self)
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(16)
        self.index = QListWidget(body)
        self.index.setObjectName("helpIndex")
        self.index.setAccessibleName("Índice de secciones de la ayuda")
        self.index.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.index.setFixedWidth(260)
        for sid, name, _html in self.sections:
            item = QListWidgetItem(name, self.index)
            item.setData(Qt.UserRole, sid)
        body_layout.addWidget(self.index)
        self.text = QTextBrowser(body)
        self.text.setObjectName("helpText")
        self.text.setAccessibleName("Texto de la ayuda")
        self.text.setOpenLinks(False)
        self.text.setHtml(help_html(config))
        body_layout.addWidget(self.text, 1)
        root.addWidget(body, 1)

        self.btn_close = QPushButton("Cerrar", self)
        self.btn_close.setProperty("variant", "primary")
        self.btn_close.setAccessibleName("Cerrar la ayuda")
        self.btn_close.setMinimumHeight(48)
        root.addWidget(self.btn_close, 0, Qt.AlignRight)

        self.index.currentRowChanged.connect(self.show_section)
        self.btn_close.clicked.connect(self.close)
        self.index.setCurrentRow(0)

    def show_section(self, row: int):
        """Lleva el texto al inicio de la sección `row` del índice."""
        if 0 <= row < len(self.sections):
            self.text.scrollToAnchor(self.sections[row][0])
