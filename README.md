# MiraLink

MiraLink es una aplicación de escritorio para Windows que permite **mover el cursor con la mirada y hacer clic parpadeando**, usando solo una webcam común. No necesita hardware especial: ni rastreadores oculares, ni cámaras infrarrojas, ni sensores.

Está pensada para personas con **discapacidad motriz** que no pueden usar el mouse con comodidad, y para **adultos mayores** con poca experiencia con la tecnología.

> Es un proyecto académico (curso de Interacción Hombre-Máquina, UTP 2026-2) en fase de prototipo (MVP). Funciona, pero su precisión depende mucho de la cámara, la luz y la postura. Lee la sección [Limitaciones](#limitaciones) antes de usarlo.

---

## Contenido

- [Qué hace](#qué-hace)
- [Requisitos](#requisitos)
- [Instalación](#instalación)
- [Primeros pasos](#primeros-pasos)
- [Cómo se usa](#cómo-se-usa)
- [Atajos y gestos](#atajos-y-gestos)
- [Configuración](#configuración)
- [Cómo funciona por dentro](#cómo-funciona-por-dentro)
- [Estructura del proyecto](#estructura-del-proyecto)
- [Para desarrolladores](#para-desarrolladores)
- [Limitaciones](#limitaciones)
- [Privacidad](#privacidad)

---

## Qué hace

- **Cursor con la mirada:** el cursor del sistema va adonde miras, después de una calibración de 9 puntos.
- **Clic con parpadeo:** un parpadeo voluntario, algo más largo que uno natural, hace clic. Los parpadeos naturales se ignoran.
- **Dos modos de puntero:**
  - **Modo mirada:** el cursor sigue a los ojos.
  - **Modo cabeza (híbrido):** la mirada lleva el cursor cerca del objetivo y la cabeza lo afina, como un joystick. Sirve para objetivos pequeños. En este modo, mantener el cursor quieto un momento mantiene el clic pulsado, para arrastrar.
- **Reposo:** cierra los ojos 2 segundos y el control se detiene (por ejemplo, para leer o descansar). Ciérralos 1 segundo para reactivarlo.
- **Botones flotantes:** sobre el escritorio aparece una barra grande con:
  - Lupa al 200 %, que se desplaza mirando los bordes.
  - Teclado en pantalla de Windows.
  - Filtro de color de Windows.
- **Botón "Ir al escritorio":** aparece cuando hay una aplicación delante.
- **Atracción magnética:** el cursor se "pega" a los botones de MiraLink para que pulsarlos sea fácil.
- **Recentrar:** corrige en segundos el desvío que aparece si te mueves, sin calibrar de nuevo.
- **Atajo de emergencia:** **Ctrl+Alt+P** pausa el control en cualquier momento.
- **Accesibilidad de la propia ventana:**
  - Botones de al menos 48 px.
  - Temas oscuro y de alto contraste.
  - Tamaño de letra ajustable.
  - Ayuda integrada.

## Requisitos

- **Windows 10 u 11.**
- **Una webcam.** La integrada del portátil sirve. Lo ideal es que dé unos 30 fps.
- **Buena iluminación frontal:** la cara bien iluminada y sin una ventana brillante detrás.
- Sentarse a unos **50-70 cm** de la pantalla, con la cara centrada en la cámara.
- Solo si usas el código fuente: **Python 3.10 a 3.13**.

## Instalación

### Opción 1: ejecutable (recomendada)

1. Ve a [Releases](https://github.com/Arzule/MiraLink/releases) y descarga `MiraLink.exe` de la versión más reciente.
2. Ponlo en una carpeta propia, por ejemplo `Documentos\MiraLink`. Ahí se guardarán la configuración y la calibración.
3. Ábrelo con doble clic.
   - La primera vez Windows puede mostrar un aviso de SmartScreen, porque el ejecutable no está firmado. Pulsa "Más información" → "Ejecutar de todas formas".
   - El arranque tarda unos segundos, porque el .exe se descomprime al abrirse.

### Opción 2: desde el código fuente

```powershell
git clone https://github.com/Arzule/MiraLink.git
cd MiraLink
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe main.py --config
```

- El modelo de MediaPipe (`models/face_landmarker.task`) se descarga solo la primera vez. También puedes bajarlo antes con `scripts\download_model.py`.
- Para el modo con red neuronal, descarga también `scripts\download_gaze_models.py` (ver [Configuración](#configuración)).

## Primeros pasos

1. **Abre MiraLink.** Se muestra la ventana de configuración.
2. **Pulsa "Calibrar Rastreador Ocular".**
   - En la vista previa, comprueba que tu cara se ve bien. Si sale "Acércate", acércate a la cámara.
   - Pulsa **Iniciar** y luego **Enter**.
   - Mira cada punto hasta que se cierre su círculo. Son 9 puntos y después 5 de comprobación.
   - **Mantén la cabeza quieta** y mueve solo los ojos.
   - Al final verás el error estimado en píxeles. Si te parece bien, pulsa **Aceptar**.
3. **Pulsa "Iniciar Control del Cursor".** Desde ese momento el cursor sigue tu mirada.
4. Si el cursor se va desviando, pulsa **Ctrl+Alt+R** (Recentrar) y mira el punto central.
5. Para detener el control: **Ctrl+Alt+P**.

Hay que volver a calibrar si cambias de pantalla, de resolución o de cámara, o si te sientas en otra posición.

## Cómo se usa

### Clic, mantener y arrastrar

| Acción | Cómo |
|---|---|
| Clic | Parpadeo voluntario de unos 0,25-2 s. |
| Mantener pulsado (solo modo cabeza) | Deja el cursor completamente quieto 1,2 s. |
| Soltar | Parpadea mientras está pulsado. |

Si el clic se queda pulsado demasiado tiempo, se suelta solo a los 30 s.

### Modo mirada y modo cabeza

- Para cambiar de modo, haz **3 parpadeos normales y rápidos** (en menos de 2 s).
- En el **modo cabeza**:
  - Si miras lejos, el cursor salta adonde miras.
  - Después, gira un poco la cabeza para moverlo despacio en esa dirección.
  - Los giros pequeños (menos de 5°) no lo mueven, para ignorar movimientos involuntarios.

### Reposo

- Cierra los ojos **2 s**: MiraLink deja de mover el cursor y de hacer clic.
- Ciérralos **1 s** para volver.

Cuidado: mirar el teclado durante 2 s también puede activar el reposo.

### Botones flotantes y lupa

- La barra aparece a la derecha cuando el escritorio está delante.
- Con una aplicación abierta, aparece solo un botón "Ir al escritorio", que hace Win+D.
- **Lupa 200 %:** amplía la pantalla. Mira hacia un borde para desplazar la vista. El cursor apunta a lo que ves ampliado.

## Atajos y gestos

| Atajo / gesto | Qué hace |
|---|---|
| **Ctrl+Alt+P** | Pausa o reanuda el control (atajo de emergencia). |
| **Ctrl+Alt+R** | Recentrar: corrige el desvío mirando un punto central. |
| Parpadeo largo | Clic. |
| 3 parpadeos rápidos | Cambia entre modo mirada y modo cabeza. |
| Ojos cerrados 2 s / 1 s | Entrar en reposo / salir del reposo. |
| ESC mantenido (calibración) | Salir de la calibración. |

## Configuración

Los ajustes principales están en la ventana:

- Tiempos de parpadeo y de mantener pulsado.
- Suavizado del cursor.
- Cámara.
- Tema y tamaño de letra.

Todo se guarda en `config.json`, junto al ejecutable o en la raíz del proyecto. Algunos ajustes avanzados solo se cambian editando ese archivo:

| Clave | Por defecto | Para qué sirve |
|---|---|---|
| `interaction.pointer_mode` | `"absolute"` | Modo con el que arranca: `"absolute"` (mirada) o `"hybrid"` (cabeza). |
| `interaction.head_speed_px_s_per_deg` | `22.5` | Velocidad del joystick de la cabeza. |
| `interaction.head_deadzone_deg` | `5.0` | Giro de cabeza que se ignora. |
| `interaction.sleep_close_ms` / `wake_close_ms` | `2000` / `1000` | Tiempo con los ojos cerrados para el reposo. |
| `interaction.snap_radius_px` | `60` | Radio de atracción de los botones (0 = desactivado). |
| `overlay.position` / `button_size` | `"right"` / `180` | Lado y tamaño de la barra de botones. |
| `tracking.eye` | `"auto"` | Ojo a seguir: `auto`, `both`, `right` o `left`. Útil si solo un ojo funciona bien. |
| `tracking.gaze_model` | `"iris"` | `"mobileone_s0"` añade una red neuronal que mejora la precisión con la cabeza en movimiento. Exige recalibrar. |
| `camera.resolution` | `"native"` | `"native"` usa la mayor resolución que la cámara sostiene a ~30 fps. |
| `camera.exposure` | `"auto"` | `"fast"` fuerza 30 fps en cámaras que bajan los fps con poca luz, con una imagen más oscura. |

La ventana **Ayuda** de la aplicación explica todo esto con los valores de tu configuración actual.

## Cómo funciona por dentro

Por cada cuadro de la cámara:

```
Webcam (OpenCV)
  → MediaPipe Face Landmarker (478 puntos de la cara, iris y pose de la cabeza)
  → Rasgos: posición del iris dentro del ojo (+ red neuronal opcional)
  → Detector de parpadeos (EAR: apertura del ojo, corregida según la altura de la mirada)
  → Mapeo mirada → pantalla (regresión polinómica ajustada en la calibración)
  → Filtros: mediana + One Euro + zona muerta (reducen el temblor sin mucho retraso)
  → Máquina de estados: clic, mantener, modos, reposo
  → Cursor del sistema (PyAutoGUI)
```

- **Calibración:**
  - 9 puntos en orden aleatorio, para que un movimiento de cabeza no se confunda con mirar a una fila.
  - Se descartan muestras tomadas con los ojos cerrados y valores atípicos.
  - Después, 5 puntos nuevos miden el error real.
  - Si todos los puntos se desvían hacia el mismo lado, ese desvío se corrige automáticamente.
- **Elección del ojo:** compara el ojo derecho, el izquierdo y los dos juntos, y se queda con el que tiembla menos entre los que tienen un error parecido.
- **Hilos:**
  - La captura y la inferencia corren en un hilo aparte.
  - La red neuronal, si se usa, en otro.
  - La interfaz no se bloquea.

## Estructura del proyecto

| Archivo | Responsabilidad |
|---|---|
| `main.py` | Punto de entrada y modos de arranque. |
| `config_ui.py` | Ventana de configuración (PySide6). |
| `help_ui.py` | Ventana de Ayuda. |
| `eye_tracker.py` | Cámara, MediaPipe, rasgos del iris, parpadeos e hilo de seguimiento. |
| `neural_gaze.py` | Red neuronal de mirada opcional (ONNX). |
| `calibration.py` / `calibration_ui.py` | Calibración y mapeo mirada-pantalla / sus pantallas. |
| `cursor_control.py` | Filtros, máquina de estados y movimiento del mouse. |
| `gaze_filters.py` | Filtro One Euro, mediana y atracción magnética. |
| `control_session.py` | Une rastreador, cursor, atajos, botones y lupa durante el control. |
| `overlay_buttons.py` | Barra de botones flotantes. |
| `magnifier.py` | Lupa propia (API de ampliación de Windows). |
| `windows_integration.py` | Teclado en pantalla, filtro de color, Win+D, ventana activa. |
| `cursor_feedback.py` | Indicador visual junto al cursor. |
| `config_manager.py` / `theme_manager.py` | `config.json` con validación / temas `.qss`. |
| `styles/` | Temas: oscuro y dos de alto contraste. |
| `scripts/` | Descarga de modelos y análisis de sesiones grabadas. |
| `tests/` | Pruebas unitarias y de flujo (unittest). |

## Para desarrolladores

```powershell
# Pruebas (no usan la cámara ni mueven el mouse)
.\venv\Scripts\python.exe -m unittest discover -s tests

# Ventana sin mover el mouse real: solo registra los eventos
.\venv\Scripts\python.exe main.py --config --dry-run

# Ver la detección de la cara y los ojos en una ventana de OpenCV
.\venv\Scripts\python.exe main.py --tracker-debug

# Grabar una sesión de calibración para analizarla sin cámara
.\venv\Scripts\python.exe main.py --record-session
.\venv\Scripts\python.exe scripts\replay_session.py sessions\session_<fecha>.npz

# Construir MiraLink.exe (PyInstaller; corre los tests y un autotest)
powershell -ExecutionPolicy Bypass -File .\build_exe.ps1
```

`MiraLink.exe --self-test` comprueba las librerías, los temas y los modelos sin abrir la cámara. Escribe el resultado en `miralink_selftest.txt`.

Si algo falla, el registro está en `logs\miralink_debug.log`, junto al ejecutable. Adjúntalo al reportar un problema.

## Limitaciones

- **Precisión:**
  - Con una webcam el error típico es de **~1-2° (unos 50-100 px)** a 60 cm.
  - Alcanza para botones grandes, pero no para objetivos pequeños.
  - Para esos casos usa el modo cabeza, la lupa o la atracción de los botones.
- **La cabeza debe estar quieta en el modo mirada.** Moverla desplaza el cursor; usa Recentrar (Ctrl+Alt+R) cuando pase.
- **Clics no deseados:** los parpadeos largos involuntarios o mirar hacia abajo pueden provocar clics o activar el reposo.
- **Solo Windows** y solo la pantalla principal para la lupa.
- **Cámara ocupada:** si otra aplicación usa la cámara (Cámara de Windows, Teams, Zoom…), MiraLink no podrá abrirla.
- **Prueba con usuarios:** una encuesta con 3 personas dio una precisión de 3/5, ningún cansancio y varias mejoras pendientes. Entre ellas:
  - Una pantalla inicial que explique los gestos.
  - Menos clics accidentales.
  - Clic derecho con un guiño.

## Privacidad

- Todo se procesa **en tu computadora**. MiraLink no envía imágenes ni datos a internet.
- Solo se conecta para descargar los modelos la primera vez, si no vienen incluidos.
- La calibración (`calibration.json`) guarda números del mapeo, no imágenes.
- Solo `--record-session`, una herramienta de análisis que hay que pedir explícitamente, guarda datos de la cara en la carpeta `sessions/`.
