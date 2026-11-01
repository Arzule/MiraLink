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
<p>MiraLink mueve el cursor del sistema con la <b>mirada</b> y hace clic con un <b>parpadeo</b>, usando solo
la webcam. Está pensado para personas que no pueden usar bien el mouse.</p>
<p>El uso normal tiene tres pasos:</p>
<ol>
<li><b>Calibrar</b> (una vez por sesión o cuando cambies de sitio): miras unos puntos en la pantalla.</li>
<li><b>Iniciar el control del cursor</b>.</li>
<li>Usar el ordenador con la mirada, los parpadeos y, si quieres, pequeños giros de la cabeza.</li>
</ol>
<p><b>Seguridad:</b> <b>{pause}</b> pausa o reanuda el control en cualquier momento, desde cualquier
programa.</p>
"""),
        ("preparar", "Antes de empezar", """
<ul>
<li><b>Cámara:</b> cierra la app Cámara de Windows y otros programas que la usen (solo uno puede usarla a la vez).</li>
<li><b>Luz:</b> que la cara esté bien iluminada de frente; evita una ventana o una lámpara detrás de ti.
Con poca luz la cámara entrega menos imágenes por segundo y el cursor tiembla más.</li>
<li><b>Distancia:</b> siéntate a una distancia cómoda con la cara centrada. Al calibrar, la vista previa
dice <i>Acércate</i>, <i>Aléjate</i> o <i>Centra tu rostro</i> si hace falta.</li>
<li><b>Postura:</b> busca una postura que puedas mantener. Mover la cabeza mucho después de calibrar desvía el
cursor (se corrige con <i>Recentrar</i>).</li>
<li><b>Gafas:</b> funcionan, pero los reflejos fuertes en los cristales pueden empeorar la precisión.</li>
</ul>
"""),
        ("calibrar", "Calibrar", """
<p>Pulsa <b>Calibrar Rastreador Ocular</b>. Se abre una ventana con la vista previa de la cámara: colócate
hasta que no haya avisos y pulsa <b>Iniciar Calibración</b>.</p>
<ol>
<li><b>Calibración (9 puntos):</b> aparecen puntos en el centro, los bordes y las esquinas. Pulsa
<b>Enter</b> cuando estés listo y mira cada punto <b>sin mover la cabeza</b> hasta que su círculo se cierre.</li>
<li><b>Comprobación (5 puntos):</b> pulsa <b>Enter</b> otra vez y mira cada punto nuevo. Sirve para medir
el error real.</li>
</ol>
<p>Al terminar verás el <b>error estimado</b> en píxeles y qué ojo se eligió para seguir la mirada (el
programa elige solo el que tiembla menos). Pulsa <b>Aceptar</b> para guardarla o <b>Repetir</b> si el
error es grande (aviso en ámbar).</p>
<p>Para salir a mitad, <b>mantén ESC</b> pulsado un momento.</p>
<p><b>Vuelve a calibrar</b> si cambias de pantalla, de silla o de lugar, si cambia mucho la luz o si
<i>Recentrar</i> ya no basta.</p>
"""),
        ("control", "Iniciar, pausar y recentrar", f"""
<ul>
<li><b>Iniciar Control del Cursor:</b> el cursor empieza a seguir tu mirada (hace falta haber calibrado).
El mismo botón lo pausa.</li>
<li><b>{pause}</b> (atajo global de seguridad): pausa o reanuda desde cualquier programa. Al pausar se
suelta cualquier clic mantenido y se quita el zoom de la lupa.</li>
<li><b>Recentrar</b> (botón o <b>{recenter}</b>): si el cursor queda siempre desplazado hacia un lado, mira
el punto que aparece en el centro durante un segundo y el desplazamiento se corrige.</li>
<li>Si ves <b>Recentra: {recenter}</b> junto al cursor, tu cabeza se movió respecto a la calibración:
recentra mirando el punto central.</li>
</ul>
<p>La etiqueta de estado (bajo los botones) muestra la cámara, los fps y los mensajes del control.</p>
"""),
        ("modos", "Modo mirada y modo cabeza", f"""
<p>Hay dos formas de mover el cursor. El control empieza en <b>modo {start_mode}</b>.</p>
<h3>Modo mirada</h3>
<p>El cursor va a donde miras. Es el más directo, pero la mirada tiembla y tiene un pequeño error, así que
apuntar a cosas pequeñas cuesta. En este modo no hay clic mantenido (dwell).</p>
<h3>Modo cabeza (mirada + cabeza)</h3>
<ul>
<li><b>La mirada hace lo grueso:</b> cuando miras a otro sitio de la pantalla, el cursor salta allí.</li>
<li><b>Después el cursor se queda quieto:</b> el temblor de la mirada ya no lo mueve.</li>
<li><b>La cabeza afina, como un joystick:</b> gira la cabeza un poco más de {dz}° hacia un lado y el cursor
se desliza despacio en esa dirección; cuanto más la giras, más rápido. Girar hacia tu izquierda lo lleva a
la izquierda; subir la barbilla, hacia arriba. Movimientos pequeños (menos de {dz}°) no lo mueven.</li>
<li>Vuelve la cabeza a su posición normal para que el cursor se detenga.</li>
</ul>
<h3>Cambiar de modo</h3>
<p>Haz <b>{mode_n} parpadeos normales</b> (rápidos, más cortos que un clic) en menos de {mode_win}. Junto al
cursor aparece <i>Modo cabeza</i> o <i>Modo mirada</i>.</p>
"""),
        ("clic", "Clic, clic mantenido y arrastrar", f"""
<ul>
<li><b>Clic:</b> cierra los ojos a propósito al menos <b>{click} ms</b> (un parpadeo un poco más largo de lo
normal) y ábrelos. El clic se hace donde estaba el cursor al cerrar los ojos. Los parpadeos naturales, más
cortos, se ignoran. Cierres de más de {_fmt_s(blink_max)} tampoco hacen clic.</li>
<li><b>Clic mantenido (solo en modo cabeza):</b> deja el cursor quieto <b>{_fmt_s(dwell)}</b>; un arco alrededor del
cursor muestra el avance y al completarse se mantiene pulsado el botón (anillo relleno). Mueve el cursor
para arrastrar.</li>
<li><b>Soltar:</b> con el botón mantenido, cierra los ojos al menos <b>{release} ms</b>. Por seguridad se
suelta solo a los {hold_timeout} s.</li>
<li>Si aparece <i>Clic bloqueado</i>, el parpadeo empezó mirando por debajo de la zona calibrada (al mirar
abajo el párpado baja solo): mira un poco más arriba.</li>
</ul>
<p>Los umbrales se ajustan en las tarjetas de la ventana principal (ver <i>Ajustes</i>).</p>
"""),
        ("reposo", "Reposo (leer o ver un video)", f"""
<p>Para leer o ver un video sin que el cursor estorbe:</p>
<ul>
<li><b>Cierra los ojos {sleep} o más:</b> el control queda <b>en reposo</b>. El cursor no se mueve ni hace
clic, y los botones flotantes se ocultan. La cámara y el programa siguen funcionando y el zoom de la lupa
se mantiene.</li>
<li><b>Cierra los ojos {wake} o más</b> estando en reposo: el control vuelve, con el cursor donde miras.</li>
</ul>
<p>Mirar hacia abajo (por ejemplo al teclado) puede parecer ojos cerrados: si el control entra en reposo
sin querer, cierra los ojos {wake} para volver.</p>
"""),
        ("botones", "Botones flotantes y lupa", """
<p>Con el control en marcha y el <b>escritorio</b> delante aparece una barra de botones grandes en el borde
derecho. Al acercar el cursor, un imán lo lleva al centro del botón; púlsalo con un clic (parpadeo).</p>
<ul>
<li><b>Lupa 200%:</b> amplía toda la pantalla al doble; púlsalo otra vez para quitarlo. Con zoom, el cursor
va a lo que ves ampliado, y mirar cerca de un borde un momento desplaza la vista hacia ese lado.</li>
<li><b>Teclado:</b> abre o cierra el teclado en pantalla de Windows para escribir con clics.</li>
<li><b>Filtro de color:</b> activa o desactiva el filtro de color de Windows. Si el atajo está
desactivado, se abre la configuración para activarlo.</li>
</ul>
<p>Con una <b>aplicación</b> delante la barra se oculta (no tapa su barra de desplazamiento) y queda un solo
botón, <b>Ir al escritorio</b>, que minimiza las ventanas y vuelve a mostrar la barra.</p>
<p>En pausa, en reposo y al recentrar no hay botones.</p>
"""),
        ("ajustes", "Ajustes de la ventana", """
<ul>
<li><b>Suavizado del cursor:</b> más alto = cursor más estable pero con más retraso.</li>
<li><b>Tiempo de fijación para mantener:</b> cuánto tiempo quieto hasta el clic mantenido (modo cabeza).</li>
<li><b>Duración mínima de parpadeo para clic:</b> súbela si hay clics sin querer; bájala si cuesta hacer clic.</li>
<li><b>Duración mínima de parpadeo para soltar:</b> cuánto cerrar los ojos para soltar un clic mantenido.</li>
<li><b>Tema</b> y <b>tamaño de letra</b> (abajo a la derecha): temas oscuros y de alto contraste.</li>
</ul>
<p>Los cambios se aplican al momento, incluso con el control en marcha. Pulsa <b>Guardar Configuración</b>
para conservarlos la próxima vez.</p>
"""),
        ("avanzado", "Ajustes avanzados (config.json)", """
<p>Algunos ajustes no tienen control en la ventana. Están en el archivo <b>config.json</b>, junto al
programa (edítalo con el programa cerrado o reinicia el control). Valores útiles en <b>"interaction"</b>:</p>
<ul>
<li><b>pointer_mode</b>: modo con el que arranca el control, "absolute" (mirada) o "hybrid" (cabeza).</li>
<li><b>head_deadzone_deg</b>: grados de giro de la cabeza que no mueven el cursor (zona muerta).</li>
<li><b>head_speed_px_s_per_deg</b>: velocidad del joystick de la cabeza (más alto = más rápido).</li>
<li><b>hybrid_jump_px</b>: cuánto tiene que alejarse la mirada para que el cursor salte.</li>
<li><b>mode_blinks</b> y <b>mode_blink_window_ms</b>: parpadeos para cambiar de modo y en cuánto tiempo.</li>
<li><b>sleep_close_ms</b> y <b>wake_close_ms</b>: ojos cerrados para el reposo y para volver.</li>
<li><b>snap_radius_px</b>: alcance del imán de los botones (0 = sin imán).</li>
</ul>
<p>Otros: <b>"overlay"</b> (<b>enabled</b>, <b>position</b> "right"/"left", <b>button_size</b>),
<b>"camera"</b> (<b>exposure</b> "fast" si la cámara da pocos fps) y <b>"tracking"</b> (<b>eye</b>: "auto",
"both", "right" o "left"; <b>gaze_model</b>: "mobileone_s0" usa además una red neuronal; tras cambiarlo
hay que volver a calibrar).</p>
"""),
        ("atajos", "Atajos y gestos", f"""
<table cellpadding="6">
<tr><td><b>{pause}</b></td><td>Pausar o reanudar el control (seguridad)</td></tr>
<tr><td><b>{recenter}</b></td><td>Recentrar la mirada</td></tr>
<tr><td><b>Parpadeo de {click} ms o más</b></td><td>Clic</td></tr>
<tr><td><b>Cursor quieto {_fmt_s(dwell)}</b></td><td>Clic mantenido (modo cabeza)</td></tr>
<tr><td><b>Parpadeo de {release} ms o más</b></td><td>Soltar el clic mantenido</td></tr>
<tr><td><b>{mode_n} parpadeos normales en {mode_win}</b></td><td>Cambiar entre modo mirada y modo cabeza</td></tr>
<tr><td><b>Ojos cerrados {sleep}</b></td><td>Reposo</td></tr>
<tr><td><b>Ojos cerrados {wake} (en reposo)</b></td><td>Volver del reposo</td></tr>
<tr><td><b>Enter</b></td><td>Empezar cada pantalla de la calibración</td></tr>
<tr><td><b>Mantener ESC</b></td><td>Salir de la calibración</td></tr>
</table>
"""),
        ("problemas", "Problemas frecuentes", f"""
<ul>
<li><b>El cursor queda desplazado:</b> usa Recentrar ({recenter}). Si sigue, calibra de nuevo.</li>
<li><b>El cursor tiembla:</b> acércate un poco, mejora la luz, sube el suavizado o usa el modo cabeza.</li>
<li><b>Hay clics sin querer:</b> sube la duración mínima de parpadeo para clic.</li>
<li><b>Cuesta hacer clic:</b> bájala, o cierra los ojos un poco más largo.</li>
<li><b>El cursor no se mueve:</b> mira si está en pausa ({pause}) o en reposo (cierra los ojos {wake}).</li>
<li><b>La cámara no abre:</b> cierra la app Cámara u otros programas que la usen.</li>
<li><b>Pocos fps (aviso en la etiqueta de estado):</b> mejora la iluminación o pon "exposure": "fast" en
config.json.</li>
<li><b>El modo cambió solo:</b> pasa si parpadeas {mode_n} veces seguidas rápido; repite los {mode_n}
parpadeos para volver.</li>
<li><b>Los botones flotantes no aparecen:</b> solo se ven con el escritorio delante; usa <i>Ir al
escritorio</i>.</li>
</ul>
<p>Si algo falla, el registro <b>logs\\miralink_debug.log</b> (junto al programa) ayuda a encontrar la causa.</p>
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
