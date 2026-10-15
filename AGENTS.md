# MiraLink - reglas del proyecto
Producto: app de escritorio (Windows 10/11, Python 3.10-3.13 (venv actual: 3.13.7)) que mueve el cursor del sistema con la mirada y hace clic con parpadeo, usando solo una webcam. Público: personas con discapacidad motriz (principal) y adultos mayores con limitaciones tecnológicas (secundario).

## Reglas de trabajo
- Terminal: PowerShell, Windows nativo (no WSL). Usar siempre .\venv\Scripts\python.exe con ruta explícita.
- No ejecutar comandos de git; el usuario hace commits y pushes.
- Solo dependencias de requirements.txt. No agregar paquetes sin que se pida.
- Trabajar solo la etapa indicada en el prompt; no adelantar funcionalidad de otras etapas.
- Identificadores en inglés; comentarios, docstrings y textos de UI en español.
- Si algo falla o es ambiguo: reportar el error exacto y detenerse, sin inventar workarounds.
- Al terminar cada tarea, entregar un reporte con: archivos creados/modificados, comandos ejecutados con su salida relevante, cada condición de cierre marcada cumplida o no, y cualquier desviación del prompt.

## Módulos (responsabilidades)
- main.py: punto de entrada; orquesta los demás módulos.
- config_manager.py: carga/guarda config.json (valores por defecto y validación de rangos).
- eye_tracker.py: captura de cámara, MediaPipe Face Mesh, iris, EAR y parpadeos.
- calibration (dentro de eye_tracker.py o módulo propio según la etapa 3): calibración de 9 puntos y mapeo mirada-pantalla; se guarda en calibration.json (ignorado por git).
- cursor_control.py: suavizado, movimiento del cursor con PyAutoGUI y máquina de estados.
- config_ui.py: ventana de configuración en PySide6.
- overlay_buttons.py: botones flotantes (teclado, zoom in/out, filtros de color).
- tts_manager.py: texto a voz con pyttsx3.
- windows_integration.py: osk.exe, atajos de la Lupa, win32gui, ms-settings.
- styles/*.qss: temas; la UI debe cargar dinámicamente todos los .qss de esta carpeta.

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