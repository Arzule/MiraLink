"""Sonda de popup del combo de temas (se ejecuta en un subproceso con la plataforma Qt real).

Uso: python tests/_popup_probe.py <carpeta_capturas>
Imprime un JSON con, por tema y escala: mensajes de Qt con "QFont::" y la luminancia del
fondo del popup frente a la declarada en el tema. No usa cámara, mouse real ni hotkeys.
"""

import json
import os
import re
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PySide6.QtCore import qInstallMessageHandler
from PySide6.QtWidgets import QApplication

from theme_manager import load_theme


def luminance(r, g, b):
    return (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0


def declared_popup_bg(theme_id):
    qss = load_theme(theme_id, 1.0)
    m = re.search(r"QComboBox QAbstractItemView[^{]*\{[^}]*?background-color:\s*(#[0-9A-Fa-f]{6})", qss)
    h = m.group(1).lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def main():
    out_dir = sys.argv[1]
    os.makedirs(out_dir, exist_ok=True)
    messages = []
    qInstallMessageHandler(lambda t, c, m: messages.append(str(m)))
    from config_ui import ConfigWindow

    app = QApplication([])
    win = ConfigWindow(config_path=os.path.join(tempfile.mkdtemp(), "c.json"))
    win.resize(1100, 1100)
    win.show()
    app.processEvents()
    results = []
    for idx in range(win.theme_combo.count()):
        theme_id = win.theme_combo.itemData(idx)
        win.theme_combo.setCurrentIndex(idx)
        app.processEvents()
        for scale in (80, 100, 200):
            win.font_slider.setValue(scale)
            app.processEvents()
            for reopen in (0, 1):  # abrir y reabrir (defecto U3 aparecía al reabrir)
                win.theme_combo.showPopup()
                app.processEvents()
                view = win.theme_combo.view()
                img = view.grab().toImage()
                px = img.pixelColor(img.width() - 4, img.height() - 4)  # esquina: fondo, no ítem seleccionado
                lum = luminance(px.red(), px.green(), px.blue())
                dec = luminance(*declared_popup_bg(theme_id))
                if scale == 100 and reopen == 1:
                    img.save(os.path.join(out_dir, f"popup_{theme_id}.png"))
                    win.grab().save(os.path.join(out_dir, f"window_{theme_id}.png"))
                    win.content.grab().save(os.path.join(out_dir, f"content_{theme_id}.png"))
                results.append({"theme": theme_id, "scale": scale, "reopen": reopen,
                                "lum": lum, "declared": dec, "view_h": view.height()})
                win.theme_combo.hidePopup()
                app.processEvents()
    win.close()
    print(json.dumps({"results": results, "qfont": [m for m in messages if "QFont::" in m],
                      "platform": app.platformName()}))


if __name__ == "__main__":
    main()
