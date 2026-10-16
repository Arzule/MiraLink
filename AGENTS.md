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
- No ejecutar comandos de git; el usuario hace commits y pushes.
- Solo dependencias de requirements.txt. No agregar paquetes sin que se pida.
- Trabajar solo la etapa indicada en el prompt; no adelantar funcionalidad de otras etapas.
- Identificadores en inglés; comentarios, docstrings y textos de UI en español.
- Si algo falla o es ambiguo: reportar el error exacto y detenerse, sin inventar workarounds.
- Al terminar cada tarea, entregar un reporte con: archivos creados/modificados, comandos ejecutados con su salida relevante, cada condición de cierre marcada cumplida o no, y cualquier desviación del prompt.

## Reglas de UI
- Todo elemento clicable de al menos 48 px de alto (se usará con la mirada).
- Todos los controles accesibles por teclado con accessibleName.
- Los textos visibles en español.

## Tokens de fuente para .qss
Los archivos .qss deben usar los siguientes tokens (reemplazados por theme_manager.load_theme con font_scale):
-  = round(12*scale)px
-   = round(14*scale)px
-  = round(24*scale)px
Nunca deben quedar tokens  sin reemplazar. No usar tamaños fijos; siempre usar estos tokens.

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
