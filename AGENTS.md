# MiraLink - reglas del proyecto
Producto: app de escritorio (Windows 10/11, Python 3.10-3.13 (venv actual: 3.13.7)) que mueve el cursor del sistema con la mirada y hace clic con parpadeo, usando solo una webcam. Público: personas con discapacidad motriz (principal) y adultos mayores con limitaciones tecnológicas (secundario).

## Hoja de ruta
1. Etapa 0 - Andamiaje (hecha): estructura base, dependencias, archivos vacíos con docstrings, estilos mínimos y main.py con verificación de entorno.
2. Etapa 1 - Configuración y estilos: config_manager.py, theme_manager.py, estilos .qss con tokens de fuente, config_ui.py y actualización de main.py.
3. Etapa 2 - Seguimiento ocular (cámara, MediaPipe, iris, EAR, parpadeos): captura de cámara, MediaPipe Face Mesh/Iris (API Tasks según el paquete instalado), detección de parpadeos (EAR) y salida de coordenadas de mirada.
4. Etapa 3 - Calibración de 9 puntos y mapeo mirada-pantalla: flujo de calibración (9 puntos), transformación mirada-pantalla, guardado en calibration.json (ignorado por git).
5. Etapa 4 - Control del cursor (MVP): suavizado, máquina de estados (reposo/mirada fija/hold/clic), movimiento con PyAutoGUI, atajo de emergencia para pausar el control y main.py mínimo de punta a punta.
6. Etapa 5 - Botones de overlay y windows_integration: botones flotantes (teclado osk.exe, zoom in/out de la Lupa, filtros de color), integración con Windows (osk.exe, atajos de la Lupa, win32gui, ms-settings).
7. Etapa 6 - TTS: gestor de texto a voz con pyttsx3.
8. Etapa 7 - Integración y cierre: verificación final, pruebas básicas y entrega.

## Notas importantes
- mediapipe 1.0.1 instalado expone mp.tasks y NO mp.solutions. El seguimiento ocular (etapa 2) debe usar la API Tasks (FaceLandmarker); verificar la API real del paquete instalado antes de escribir código.

## Reglas de trabajo
- Terminal: PowerShell, Windows nativo (no WSL). Usar siempre .\venv\Scripts\python.exe con ruta explícita.
- Git: el usuario autorizó (prompt de las etapas 2-4) commits locales por etapa con mensaje "feat: etapa N (resumen)"; nunca push ni comandos destructivos. Fuera de esa autorización, no ejecutar git.
- Solo dependencias de requirements.txt. No agregar paquetes sin que se pida.
- Trabajar solo las etapas indicadas en el prompt vigente (hoy: Parte 0 y etapas 2-4); no adelantar funcionalidad de otras etapas.
- Identificadores en inglés; comentarios, docstrings y textos de UI en español.
- Si algo falla o es ambiguo: reportar el error exacto y detenerse, sin inventar workarounds.
- Al terminar cada tarea, entregar un reporte con: archivos creados/modificados, comandos ejecutados con su salida relevante, cada condición de cierre marcada cumplida o no, y cualquier desviación del prompt.

## Reglas de UI
- Todo elemento clicable de al menos 48 px de alto (se usará con la mirada).
- Todos los controles accesibles por teclado con accessibleName.
- Los textos visibles en español.

## Tokens de fuente para .qss
Los archivos .qss usan estos tokens, reemplazados por theme_manager.load_theme según font_scale (en pt, mínimo 1pt; los px dejaban pointSize() = -1 y el popup del combo emitía "QFont::setPointSize: Point size <= 0"):
- $FONT_SMALL = max(1, round(9*scale))pt
- $FONT_BASE = max(1, round(11*scale))pt
- $FONT_TITLE = max(1, round(18*scale))pt
Nunca deben quedar tokens sin reemplazar. No usar tamaños de fuente fijos en px; siempre estos tokens. Las alturas mínimas clicables (48px) sí van en px.

## Módulos (responsabilidades)
- main.py: punto de entrada; orquesta los demás módulos.
- config_manager.py: carga/guarda config.json (valores por defecto y validación de rangos).
- theme_manager.py: carga de temas .qss y tokens de fuente, sin importar Qt.
- eye_tracker.py: captura de cámara, MediaPipe Face Mesh, iris, EAR y parpadeos (etapa 2, usar API Tasks).
- calibration (dentro de eye_tracker.py o módulo propio según la etapa 3): calibración de 9 puntos y mapeo mirada-pantalla; se guarda en calibration.json (ignorado por git).
- cursor_control.py: suavizado, movimiento del cursor con PyAutoGUI y máquina de estados.
- config_ui.py: ventana de configuración en PySide6.
- overlay_buttons.py: botones flotantes (teclado, zoom in/out, filtros de color).
- tts_manager.py: texto a voz con pyttsx3.
- windows_integration.py: osk.exe, atajos de la Lupa, win32gui, ms-settings.
- styles/*.qss: temas; la UI debe cargar dinámicamente todos los .qss de esta carpeta.
- tests/: pruebas unittest, se ejecutan con: .\venv\Scripts\python.exe -m unittest discover -s tests -v.

## Interacción (máquina de estados)
- Mirada controla el cursor. Dwell (mirada fija) sostenido activa modo HOLD (clic mantenido) tras dwell_hold_ms.
- Parpadeo intencional >= blink_click_ms sin hold activo = clic simple.
- Con hold activo, parpadeo >= blink_release_ms = soltar. Parpadeos más cortos que el umbral se ignoran (parpadeo natural).
- Todos los umbrales vienen de config.json (rangos: dwell_hold_ms 800-3000, blink_click_ms 150-500, blink_release_ms 150-500, smoothing_percent 0-100).

## Estilo visual (para la UI, tomado del diseño en Figma)
Tema oscuro, flat, bordes redondeados, un solo acento. Colores: fondo #101019, tarjetas #181829, borde #2E2E4A, acento #8070FF, texto principal #F3F3FA, texto secundario #A1A1BE. Tipografía: Inter con fallback Segoe UI.

## Seguridad
El control del cursor debe tener siempre un atajo global de emergencia que lo pause (se implementa en la etapa 4).

CONDICIONES DE CIERRE (confírmalas una por una en el reporte)
[ ] La versión de Python del venv está entre 3.10 y 3.13 (indica cuál).
[ ] Las siete dependencias instaladas sin error.
[ ] requirements.txt contiene exactamente las siete dependencias con versión.
[ ] AGENTS.md, .gitignore, config.json, styles/ (3 archivos) y los 8 módulos .py existen en la raíz.
[ ] "python main.py" (con el venv) imprime OK para las siete librerías y termina con código 0. Pega la salida completa.
[ ] No se ejecutó ningún comando de git ni se tocó el contenido de venv/ salvo por pip.
[ ] Lista de desviaciones o problemas (si no hay, escribe "ninguna").

Al terminar, DETENTE y entrega el reporte. No empieces la etapa 1.


## Reglas de UI adicionales
- Un solo layout por contenedor; no agregar un segundo layout a un contenedor que ya lo tiene.
- QSpinBox siempre con NoButtons (sin botones de subir/bajar).
- Fondos transparentes para QLabel, QSlider y elementos dentro de #card.
- El control de tamaño de letra vive en el pie de la ventana (no en la tarjeta Apariencia).
- Todo cambio de UI debe verificarse con las pruebas de layout; no afirmar verificaciones visuales no realizadas (usar 'no verificado visualmente' cuando corresponda).
- Antes de afirmar ausencia de errores de Qt, capturar sus mensajes (qInstallMessageHandler) incluso en modo offscreen.
- Los popups de QComboBox tienen reglas opacas propias en cada tema (`QComboBox QAbstractItemView`, ítems de 48 px); `#card QWidget` no debe volver a alcanzarlos. En la plataforma `offscreen` Qt emite siempre "QFont::setPointSizeF ... (-0.75)" al abrir un popup (artefacto, también sin hoja de estilo), así que el test del popup corre en un subproceso con la plataforma real (tests/_popup_probe.py).
- La tarjeta Apariencia ocupa solo una celda del grid (fila 2, columna 0) con la altura de su contenido; el espacio sobrante queda vacío bajo el pie (stretch final).

## Decisiones técnicas transversales (etapas 2-4)
- Lógica pura y testeable: BlinkDetector, extractor de rasgos, GazeMapper, máquina de estados del cursor y filtros no tocan hardware ni relojes reales; reciben timestamps inyectados. Cámara, mouse y hotkey van detrás de interfaces delgadas con versiones falsas para los tests.
- Hilos: captura e inferencia en `eye_tracker.TrackerThread` (QThread, fábrica para no importar Qt al importar el módulo) que emite `frame_ready(TrackingFrame)` y `failed(str)`.
- Coordenadas: mapeo mirada-pantalla en píxeles físicos (geometría del QScreen x devicePixelRatio). NO llamar SetProcessDpiAwareness tras crear QApplication (Qt 6 ya fija PerMonitorV2). Al iniciar el control comparar pyautogui.size() con la geometría física; si no coinciden, advertir y no iniciar el control real. calibration.json guarda la geometría y se invalida si cambia.
- PyAutoGUI: FAILSAFE = False y PAUSE = 0.
- Seguridad: atajo global Ctrl+Alt+P alterna la pausa del control (RegisterHotKey en su propio hilo). Si RegisterHotKey falla NO se permite el control real, solo --dry-run, con mensaje claro en la UI. Al pausar/cerrar/fallar se suelta siempre el botón del mouse (try/finally y atexit); auto-liberación del hold tras interaction.hold_timeout_s.
- Flag --dry-run: el backend de mouse solo registra eventos; no mueve ni hace clic.
- Verificación del agente: no abre la cámara, no inicia el control del cursor (ni en --dry-run), no registra el hotkey real ni ejecuta --tracker-debug; eso lo prueba el usuario.

## Claves de interaction añadidas (validadas con clamp, sin UI)
dwell_radius_px=50 (20-150), hold_enabled=true, blink_max_ms=2000 (500-5000), freeze_after_blink_ms=150 (0-500), hold_timeout_s=30 (5-120).

## Etapa 2 - eye_tracker.py
- Modelo: models/face_landmarker.task (ignorado por git), descargado con scripts/download_model.py (URL oficial float16/latest de storage.googleapis.com/mediapipe-models) o automáticamente al crear FaceTracker; sin internet lanza ModelNotFoundError con instrucciones.
- API usada: mediapipe.tasks.python.vision.FaceLandmarker en modo VIDEO (timestamps crecientes en ms), output_face_blendshapes y output_facial_transformation_matrixes activados.
- Vector de rasgos (FEATURE_NAMES, 9 valores): [0] right_iris_along, [1] right_iris_perp, [2] left_iris_along, [3] left_iris_perp (iris en el marco local del ojo: componente a lo largo de la línea entre esquinas, de izquierda a derecha en la imagen, y perpendicular positivo hacia abajo, ambas relativas al centro de las esquinas y divididas por el ancho del ojo), [4] nose_x, [5] nose_y (landmark 1 normalizado en el cuadro), [6] interocular (distancia entre esquinas externas 33 y 263 / ancho del cuadro; proxy de profundidad), [7] yaw, [8] pitch (radianes, de la matriz facial; 0 si no hay).
- EAR: 6 puntos por ojo (derecho 33,160,158,133,153,144; izquierdo 362,385,387,263,373,380) en píxeles.
- BlinkDetector: línea base = percentil 90 del EAR abierto en ventana de 5 s (mín. 15 muestras, base mínima 0.12), cierre < 0.6 x base, reapertura > 0.75 x base, usa el EAR promedio; descarta eventos < 60 ms; al perder el rostro reinicia el cierre sin evento. Los blendshapes eyeBlink se exponen en TrackingFrame.blink_score solo como dato informativo (el EAR resultó suficiente en teoría; no verificado con cámara real).
- EyeTrackingPipeline(camera, tracker, detector, clock) une cámara -> FaceTracker -> BlinkDetector y rellena eyes_closed/blink_event/image en el TrackingFrame.
- Depuración: `python main.py --tracker-debug` (ventana OpenCV; ESC o Q para salir).

## Mejoras futuras
- Hallazgo de la prueba manual de --tracker-debug: la detección ocular solo funciona bien si la cámara entrega ~30 fps; si los fps bajan, la detección falla.
  - Medir los fps reales y avisar en la UI si bajan de ~25.
  - Pedir 30 fps con CAP_PROP_FPS y revisar la exposición automática (las webcams bajan los fps con poca luz).
  - Hacer que la detección de parpadeos y la calibración dependan de timestamps y no de conteo de cuadros (el BlinkDetector ya usa timestamps; los umbrales de CalibrationSession son en segundos salvo min_samples=20).
- Regla para el código nuevo: no asumir una tasa de cuadros fija; usar siempre TrackingFrame.timestamp.

## Etapa 3 - calibración (calibration.py, calibration_ui.py)
- Módulos nuevos: `calibration.py` (puro, solo numpy: GazeMapper, CalibrationSession, HoldToExit, face_guidance, save/load_calibration) y `calibration_ui.py` (Qt: CameraPreview, CalibrationScreen, CalibrationResultDialog, CalibrationDialog). `ConfigWindow.calibrate_requested` abre el modal vía `ConfigWindow.calibration_dialog_factory` (los tests inyectan un falso: NUNCA hacer clic en "Calibrar" en tests con la fábrica por defecto, abre y sondea la cámara).
- GazeMapper: reduce el vector de 9 rasgos a 6 (`compact`: iris_along medio, iris_perp medio, nose_x, nose_y, yaw, pitch; sin interocular) o usa los 9 (`full`); normaliza (media/desv. con piso 0.02), polinomio de grado 2 (28 términos en compact) y ridge con penalización alpha*n_muestras (alpha=0.02, constante sin penalizar). Métricas: `rmse_px` (entrenamiento) y `cv_rmse_px` (dejando un punto de calibración fuera). El "error estimado" mostrado al usuario es `cv_rmse_px` (pesimista en las esquinas); advierte si > 150 px (`WARN_ERROR_PX`).
- calibration.json (ignorado por git): version, created, camera_index, screen {x,y,width,height en píxeles físicos}, feature_set, alpha, mean, scale, weights, metrics. `load_calibration(path, geometry)` lanza CalibrationError si falta, está corrupto, tiene valores no finitos o la geometría cambió. Se guarda solo al pulsar "Aceptar".
- CalibrationSession: cuadrícula 3x3 con márgenes 10 % (izquierda a derecha, arriba abajo), 1.0 s de asentamiento + 1.5 s de captura medidos con timestamps de TrackingFrame, descarta cuadros sin rostro u ojos cerrados, mínimo 20 muestras (único criterio por conteo), hasta 2 reintentos por punto y luego aborta con mensaje. Sin cuadros, el reloj de la pantalla la hace avanzar (watchdog 0.3 s) hasta abortar.
- Pantalla completa: título/subtítulo/píldora centrados al 30 % de la altura, aviso ESC (mantener 1.5 s) y mini previsualización en la banda del 70 % (entre las filas de puntos 50 % y 90 %), para no tapar puntos. Los puntos se dibujan en coordenadas lógicas = físicas / devicePixelRatio.
- Verificación visual: las capturas con la plataforma `offscreen` no tienen fuentes (texto como cuadros); para revisar texto usar un script desechable con la plataforma real.
