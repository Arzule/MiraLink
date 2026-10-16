"""Seguimiento ocular: cámara, MediaPipe FaceLandmarker (API Tasks), iris, EAR y parpadeos.

La lógica (EAR, rasgos, BlinkDetector) es pura: recibe landmarks y timestamps inyectados y
no toca hardware ni relojes. La cámara y el modelo van detrás de clases delgadas
(CameraSource, FaceTracker) y el hilo de captura (TrackerThread) solo las orquesta.
"""

from __future__ import annotations

import logging
import math
import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# --------------------------------------------------------------- landmarks
# Índices del modelo de 478 puntos (468 de malla + 10 de iris).
# "Derecho"/"izquierdo" se refieren al ojo del sujeto; en la imagen sin espejo el derecho queda a la izquierda.
RIGHT_EYE_EAR = (33, 160, 158, 133, 153, 144)  # p1..p6 del EAR
LEFT_EYE_EAR = (362, 385, 387, 263, 373, 380)
RIGHT_EYE_CORNERS = (33, 133)
LEFT_EYE_CORNERS = (362, 263)
RIGHT_IRIS_CENTER = 468
LEFT_IRIS_CENTER = 473
RIGHT_IRIS_RING = (468, 469, 470, 471, 472)
LEFT_IRIS_RING = (473, 474, 475, 476, 477)
NOSE_TIP = 1
OUTER_EYE_CORNERS = (33, 263)  # para la distancia interocular
NUM_LANDMARKS = 478

# Vector de rasgos (documentado en AGENTS.md)
FEATURE_NAMES = (
    "right_iris_along",  # posición del iris a lo largo de la línea de las esquinas, centrada (0 = centro)
    "right_iris_perp",  # posición perpendicular (positivo = hacia abajo), dividida por el ancho del ojo
    "left_iris_along",
    "left_iris_perp",
    "nose_x",  # nariz normalizada en el cuadro (0..1)
    "nose_y",
    "interocular",  # distancia entre esquinas externas / ancho del cuadro (proxy de profundidad)
    "yaw",  # radianes, desde la matriz facial (0 si no hay)
    "pitch",  # radianes
    "roll",  # radianes (inclinación lateral de la cabeza)
)
FEATURE_DIM = len(FEATURE_NAMES)

DEFAULT_MODEL_FILENAME = "face_landmarker.task"


class ModelNotFoundError(RuntimeError):
    """El modelo de MediaPipe no existe y no se pudo descargar."""


# ------------------------------------------------------------------ datos
@dataclass
class BlinkEvent:
    """Parpadeo completo: cierre y reapertura de los ojos."""

    start_ts: float
    end_ts: float
    duration_ms: float


@dataclass
class TrackingFrame:
    """Resultado del seguimiento de un cuadro."""

    timestamp: float
    face_found: bool = False
    ear_left: float = 0.0
    ear_right: float = 0.0
    ear_mean: float = 0.0
    features: Optional[np.ndarray] = None
    face_center: Optional[Tuple[float, float]] = None  # normalizado (0..1) en el cuadro
    interocular: float = 0.0  # fracción del ancho del cuadro
    landmarks: Optional[np.ndarray] = None  # (478, 2) normalizados, para dibujar
    matrix: Optional[np.ndarray] = None  # matriz de transformación facial 4x4 (se graba en --record-session)
    blink_score: Optional[float] = None  # promedio eyeBlinkLeft/Right si el modelo lo entrega (informativo)
    eyes_closed: bool = False  # estado del BlinkDetector
    blink_event: Optional[BlinkEvent] = None  # parpadeo que terminó en este cuadro
    image: Optional[np.ndarray] = None  # cuadro BGR sin espejo (para previsualizaciones)
    fps: float = 0.0  # cuadros por segundo medidos con los timestamps (FpsMeter); 0 = aún sin medir
    camera_info: str = ""  # p. ej. "1280x720 MJPG" (CameraSource.describe)


# --------------------------------------------------------- funciones puras
def eye_aspect_ratio(pts: np.ndarray) -> float:
    """EAR de 6 puntos p1..p6 (arreglo (6, 2)): (|p2-p6| + |p3-p5|) / (2 |p1-p4|)."""
    pts = np.asarray(pts, dtype=float)
    horizontal = np.linalg.norm(pts[0] - pts[3])
    if horizontal < 1e-9:
        return 0.0
    vertical = np.linalg.norm(pts[1] - pts[5]) + np.linalg.norm(pts[2] - pts[4])
    return float(vertical / (2.0 * horizontal))


def _px(landmarks: np.ndarray, idx, size: Tuple[int, int]) -> np.ndarray:
    """Puntos normalizados -> píxeles (para respetar la relación de aspecto)."""
    w, h = size
    pts = np.asarray(landmarks, dtype=float)[list(idx) if not isinstance(idx, int) else idx, :2]
    return pts * np.array([w, h], dtype=float)


def compute_ears(landmarks: np.ndarray, size: Tuple[int, int]) -> Tuple[float, float]:
    """(EAR izquierdo, EAR derecho) a partir de landmarks normalizados y tamaño (ancho, alto)."""
    left = eye_aspect_ratio(_px(landmarks, LEFT_EYE_EAR, size))
    right = eye_aspect_ratio(_px(landmarks, RIGHT_EYE_EAR, size))
    return left, right


def _iris_in_eye_frame(landmarks, corners, iris_idx, size) -> Tuple[float, float]:
    """Iris en el marco local del ojo: (a lo largo de las esquinas, perpendicular) / ancho del ojo."""
    a, b = _px(landmarks, corners, size)
    if a[0] > b[0]:  # eje siempre de izquierda a derecha en la imagen
        a, b = b, a
    axis = b - a
    width = float(np.linalg.norm(axis))
    if width < 1e-9:
        return 0.0, 0.0
    u = axis / width
    n = np.array([-u[1], u[0]])  # perpendicular; con y hacia abajo, positivo = abajo
    iris = _px(landmarks, iris_idx, size)
    rel = iris - (a + b) / 2.0
    return float(np.dot(rel, u) / width), float(np.dot(rel, n) / width)


def head_angles_from_matrix(matrix: Optional[np.ndarray]) -> Tuple[float, float, float]:
    """Yaw, pitch y roll aproximados (radianes) desde la matriz de transformación facial 4x4."""
    if matrix is None:
        return 0.0, 0.0, 0.0
    m = np.asarray(matrix, dtype=float)
    if m.ndim != 2 or m.shape[0] < 3 or m.shape[1] < 3:
        return 0.0, 0.0, 0.0
    r = m[:3, :3]
    pitch = math.atan2(r[2, 1], r[2, 2])
    yaw = math.atan2(-r[2, 0], math.hypot(r[2, 1], r[2, 2]))
    roll = math.atan2(r[1, 0], r[0, 0])
    return float(yaw), float(pitch), float(roll)


def yaw_pitch_from_matrix(matrix: Optional[np.ndarray]) -> Tuple[float, float]:
    """Yaw y pitch aproximados (radianes) desde la matriz de transformación facial 4x4."""
    yaw, pitch, _ = head_angles_from_matrix(matrix)
    return yaw, pitch


def extract_features(landmarks: np.ndarray, size: Tuple[int, int], matrix: Optional[np.ndarray] = None) -> np.ndarray:
    """Vector de rasgos de mirada (ver FEATURE_NAMES), dtype float64 de longitud FEATURE_DIM."""
    r_along, r_perp = _iris_in_eye_frame(landmarks, RIGHT_EYE_CORNERS, RIGHT_IRIS_CENTER, size)
    l_along, l_perp = _iris_in_eye_frame(landmarks, LEFT_EYE_CORNERS, LEFT_IRIS_CENTER, size)
    lm = np.asarray(landmarks, dtype=float)
    nose_x, nose_y = float(lm[NOSE_TIP, 0]), float(lm[NOSE_TIP, 1])
    c1, c2 = _px(landmarks, OUTER_EYE_CORNERS, size)
    interocular = float(np.linalg.norm(c1 - c2) / size[0])
    yaw, pitch, roll = head_angles_from_matrix(matrix)
    return np.array(
        [r_along, r_perp, l_along, l_perp, nose_x, nose_y, interocular, yaw, pitch, roll], dtype=float
    )


def build_tracking_frame(
    timestamp: float,
    landmarks: Optional[np.ndarray],
    size: Tuple[int, int],
    matrix: Optional[np.ndarray] = None,
    blink_score: Optional[float] = None,
) -> TrackingFrame:
    """Construye un TrackingFrame a partir de landmarks (pura, sin MediaPipe)."""
    if landmarks is None or len(landmarks) < NUM_LANDMARKS:
        return TrackingFrame(timestamp=timestamp, face_found=False)
    lm = np.asarray(landmarks, dtype=float)
    ear_l, ear_r = compute_ears(lm, size)
    feats = extract_features(lm, size, matrix)
    return TrackingFrame(
        timestamp=timestamp,
        face_found=True,
        ear_left=ear_l,
        ear_right=ear_r,
        ear_mean=(ear_l + ear_r) / 2.0,
        features=feats,
        face_center=(float(lm[:, 0].mean()), float(lm[:, 1].mean())),
        interocular=float(feats[6]),
        landmarks=lm[:, :2].copy(),
        matrix=None if matrix is None else np.asarray(matrix, dtype=float),
        blink_score=blink_score,
    )


# ---------------------------------------------------------- BlinkDetector
class BlinkDetector:
    """Detecta parpadeos desde el EAR promedio con línea base adaptativa.

    - Línea base: percentil `baseline_percentile` del EAR reciente con los ojos abiertos
      (ventana móvil de `window_s`); no se actualiza mientras los ojos están cerrados.
    - Cierre cuando EAR < close_ratio * base; reapertura cuando EAR > reopen_ratio * base
      (histéresis). Hasta tener `min_samples` muestras abiertas no se detecta nada.
    - Los eventos más cortos que `min_event_ms` se descartan (ruido de un solo cuadro).
    - Si se pierde el rostro se reinicia el estado sin emitir evento.
    Usa solo timestamps inyectados (segundos).
    """

    def __init__(
        self,
        close_ratio: float = 0.6,
        reopen_ratio: float = 0.75,
        window_s: float = 5.0,
        baseline_percentile: float = 90.0,
        min_samples: int = 15,
        min_baseline: float = 0.12,
        min_event_ms: float = 60.0,
    ):
        self.close_ratio = close_ratio
        self.reopen_ratio = reopen_ratio
        self.window_s = window_s
        self.baseline_percentile = baseline_percentile
        self.min_samples = min_samples
        self.min_baseline = min_baseline
        self.min_event_ms = min_event_ms
        self._samples: deque = deque()
        self._closed = False
        self._closed_since: Optional[float] = None
        self._baseline: Optional[float] = None

    @property
    def is_closed(self) -> bool:
        return self._closed

    @property
    def baseline(self) -> Optional[float]:
        """Línea base actual del EAR abierto (None mientras no haya muestras suficientes)."""
        return self._baseline

    def reset(self):
        """Reinicia estado y línea base."""
        self._samples.clear()
        self._closed = False
        self._closed_since = None
        self._baseline = None

    def _update_baseline(self):
        if len(self._samples) >= self.min_samples:
            values = [v for _, v in self._samples]
            self._baseline = float(np.percentile(values, self.baseline_percentile))

    def update(self, timestamp: float, ear: float, face_found: bool = True) -> Optional[BlinkEvent]:
        """Procesa un cuadro. Devuelve BlinkEvent cuando el ojo se reabre tras un parpadeo válido."""
        if not face_found:
            # Se conserva la línea base (la persona sigue siendo la misma) pero no el cierre
            self._closed = False
            self._closed_since = None
            return None

        if not self._closed:
            self._samples.append((timestamp, ear))
            while self._samples and timestamp - self._samples[0][0] > self.window_s:
                self._samples.popleft()
            self._update_baseline()
            base = self._baseline
            if base is None or base < self.min_baseline:
                return None
            if ear < self.close_ratio * base:
                self._closed = True
                self._closed_since = timestamp
                # La muestra que cerró no debe contaminar la línea base
                self._samples.pop()
                self._update_baseline()
            return None

        base = self._baseline if self._baseline is not None else 0.0
        if ear > self.reopen_ratio * base:
            start = self._closed_since if self._closed_since is not None else timestamp
            self._closed = False
            self._closed_since = None
            duration_ms = (timestamp - start) * 1000.0
            if duration_ms < self.min_event_ms:
                return None
            return BlinkEvent(start_ts=start, end_ts=timestamp, duration_ms=duration_ms)
        return None


# ------------------------------------------------------------- hardware
FALLBACK_SIZE = (640, 480)
MIN_CAMERA_FPS = 25.0  # por debajo la detección ocular se degrada (hallazgo de la prueba manual)
MIN_PROBE_FPS = 24.0  # una resolución alta que no llega a esto al abrir se descarta


def _fourcc_text(value) -> str:
    try:
        code = int(value)
    except (TypeError, ValueError):
        return ""
    text = "".join(chr((code >> (8 * i)) & 0xFF) for i in range(4))
    return text if text.isprintable() and text.strip() else ""


class CameraSource:
    """Captura de cámara con cv2.VideoCapture (DirectShow) a 30 fps.

    640x480 por defecto. Con otra resolución (p. ej. 1280x720) pide MJPG (muchas webcams solo dan
    30 fps a 720p comprimiendo) y sondea `probe_frames` cuadros midiendo los fps con el reloj: si la
    cámara no entrega el tamaño pedido o baja de MIN_PROBE_FPS, vuelve a 640x480 sin MJPG y lo anota
    en `fallback_reason`. Más resolución mejora la precisión del iris pero cuesta CPU (MediaPipe).
    `cv2_module` y `clock` se inyectan en los tests (nunca se abre la cámara real en ellos).
    """

    def __init__(
        self,
        index: int = 0,
        width: int = FALLBACK_SIZE[0],
        height: int = FALLBACK_SIZE[1],
        fps: int = 30,
        cv2_module=None,
        clock: Callable[[], float] = time.monotonic,
        probe_frames: int = 12,
    ):
        self.index = index
        self.width = int(width)
        self.height = int(height)
        self.fps = fps
        self._cv2 = cv2_module
        self._clock = clock
        self.probe_frames = max(3, int(probe_frames))
        self._cap = None
        self.actual_size: Optional[Tuple[int, int]] = None
        self.fourcc = ""
        self.measured_fps = 0.0
        self.fallback_reason = ""

    def _open_capture(self, cv2, size: Tuple[int, int], mjpg: bool):
        cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            return None
        if mjpg:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # antes del tamaño
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        return cap

    def _probe(self, cap) -> Tuple[bool, Optional[Tuple[int, int]], float]:
        """Lee unos cuadros: (ok, tamaño entregado, fps medidos con la mediana de los intervalos)."""
        times, size = [], None
        for _ in range(self.probe_frames):
            ok, frame = cap.read()
            if not ok or frame is None:
                return False, size, 0.0
            size = (int(frame.shape[1]), int(frame.shape[0]))
            times.append(self._clock())
        gaps = np.diff(times[1:])  # el primer cuadro suele tardar más (exposición automática)
        gaps = gaps[gaps > 0]
        fps = float(1.0 / np.median(gaps)) if len(gaps) else 0.0
        return True, size, fps

    def open(self):
        if self._cv2 is None:
            import cv2

            self._cv2 = cv2
        cv2 = self._cv2
        requested = (self.width, self.height)
        high = requested != FALLBACK_SIZE
        cap = self._open_capture(cv2, requested, mjpg=high)
        if cap is None:
            raise RuntimeError(f"No se pudo abrir la cámara {self.index}. Revisa que no esté en uso por otra aplicación.")
        ok, size, fps = self._probe(cap)
        if high and (not ok or size != requested or fps < MIN_PROBE_FPS):
            if not ok:
                reason = f"la cámara no entregó imagen a {requested[0]}x{requested[1]}"
            elif size != requested:
                reason = f"pedido {requested[0]}x{requested[1]}, la cámara entrega {size[0]}x{size[1]}"
            else:
                reason = f"{requested[0]}x{requested[1]} solo a {fps:.1f} fps"
            logger.warning("Cámara %s: %s; se usa %dx%d", self.index, reason, *FALLBACK_SIZE)
            cap.release()
            cap = self._open_capture(cv2, FALLBACK_SIZE, mjpg=False)
            if cap is None:
                raise RuntimeError(f"No se pudo reabrir la cámara {self.index} a {FALLBACK_SIZE[0]}x{FALLBACK_SIZE[1]}.")
            ok, size, fps = self._probe(cap)
            self.fallback_reason = reason
        if not ok:
            cap.release()
            raise RuntimeError(f"La cámara {self.index} se abrió pero no entrega imagen.")
        self.actual_size, self.measured_fps = size, fps
        self.fourcc = _fourcc_text(cap.get(cv2.CAP_PROP_FOURCC))
        self._cap = cap
        logger.info(
            "Cámara %s: %dx%d %s a %.1f fps medidos (pedido %dx%d)",
            self.index, size[0], size[1], self.fourcc or "?", fps, requested[0], requested[1],
        )

    def describe(self) -> str:
        """Resumen legible de lo que entrega la cámara (para la UI y las sesiones grabadas)."""
        if self.actual_size is None:
            return ""
        text = f"{self.actual_size[0]}x{self.actual_size[1]}"
        if self.fourcc:
            text += f" {self.fourcc}"
        if self.fallback_reason:
            text += f" (sin {self.width}x{self.height}: {self.fallback_reason})"
        return text

    def read(self):
        """Devuelve (ok, cuadro BGR)."""
        if self._cap is None:
            return False, None
        return self._cap.read()

    def release(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None


def list_cameras(max_index: int = 5, opener: Optional[Callable] = None) -> List[int]:
    """Sondea los índices 0..max_index y devuelve los que abren una cámara.

    `opener(index)` debe devolver un objeto con isOpened() y release(); por defecto usa OpenCV.
    """
    if opener is None:
        import cv2

        def opener(i):
            return cv2.VideoCapture(i, cv2.CAP_DSHOW)

    found = []
    for i in range(max_index + 1):
        cap = None
        try:
            cap = opener(i)
            if cap is not None and cap.isOpened():
                found.append(i)
        except Exception:
            pass
        finally:
            if cap is not None:
                cap.release()
    return found


def default_model_path() -> str:
    root = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(root, "models", DEFAULT_MODEL_FILENAME)


def ensure_model(model_path: Optional[str] = None, auto_download: bool = True) -> str:
    """Devuelve la ruta del modelo; lo descarga la primera vez si falta. Lanza ModelNotFoundError."""
    path = model_path or default_model_path()
    if os.path.isfile(path):
        return path
    if auto_download:
        try:
            from scripts.download_model import download_model

            return download_model(path)
        except RuntimeError as e:
            raise ModelNotFoundError(str(e)) from e
    raise ModelNotFoundError(
        f"Falta el modelo de MediaPipe en {path}. Ejecuta: .\\venv\\Scripts\\python.exe scripts\\download_model.py"
    )


class FaceTracker:
    """Envoltura del FaceLandmarker (API Tasks de mediapipe 1.0.1, modo VIDEO)."""

    def __init__(self, model_path: Optional[str] = None, auto_download: bool = True):
        path = ensure_model(model_path, auto_download)
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self._mp = mp
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=path),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._last_ms = -1

    def process(self, bgr_frame: np.ndarray, timestamp: float) -> TrackingFrame:
        """Analiza un cuadro BGR; `timestamp` en segundos."""
        h, w = bgr_frame.shape[:2]
        rgb = np.ascontiguousarray(bgr_frame[:, :, ::-1])
        ts_ms = max(int(timestamp * 1000), self._last_ms + 1)  # MediaPipe exige timestamps crecientes
        self._last_ms = ts_ms
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(image, ts_ms)
        if not result.face_landmarks:
            return TrackingFrame(timestamp=timestamp, face_found=False)
        landmarks = np.array([[p.x, p.y, p.z] for p in result.face_landmarks[0]], dtype=float)
        matrix = None
        if result.facial_transformation_matrixes:
            matrix = np.asarray(result.facial_transformation_matrixes[0], dtype=float)
        blink_score = None
        if result.face_blendshapes:
            scores = {c.category_name: c.score for c in result.face_blendshapes[0]}
            if "eyeBlinkLeft" in scores and "eyeBlinkRight" in scores:
                blink_score = (scores["eyeBlinkLeft"] + scores["eyeBlinkRight"]) / 2.0
        return build_tracking_frame(timestamp, landmarks, (w, h), matrix, blink_score)

    def close(self):
        self._landmarker.close()


class FpsMeter:
    """Cuadros por segundo en una ventana móvil de timestamps (no asume una tasa fija)."""

    def __init__(self, window_s: float = 2.0):
        self.window_s = window_s
        self._ts: deque = deque()

    def update(self, timestamp: float) -> float:
        self._ts.append(float(timestamp))
        while len(self._ts) > 2 and self._ts[-1] - self._ts[0] > self.window_s:
            self._ts.popleft()
        span = self._ts[-1] - self._ts[0]
        return (len(self._ts) - 1) / span if len(self._ts) >= 5 and span > 0 else 0.0


class EyeTrackingPipeline:
    """Cámara -> FaceTracker -> BlinkDetector. Los tres se inyectan (versiones falsas en tests)."""

    def __init__(self, camera, tracker, detector: Optional[BlinkDetector] = None, clock: Callable[[], float] = time.monotonic):
        self.camera = camera
        self.tracker = tracker
        self.detector = detector or BlinkDetector()
        self.clock = clock
        self.fps_meter = FpsMeter()
        describe = getattr(camera, "describe", None)
        self.camera_info = describe() if callable(describe) else ""

    def step(self) -> Optional[TrackingFrame]:
        """Procesa un cuadro. Devuelve None si la cámara no entregó imagen."""
        ok, image = self.camera.read()
        if not ok or image is None:
            return None
        ts = self.clock()
        frame = self.tracker.process(image, ts)
        frame.image = image
        event = self.detector.update(ts, frame.ear_mean, frame.face_found)
        frame.eyes_closed = self.detector.is_closed
        frame.blink_event = event
        frame.fps = self.fps_meter.update(ts)
        frame.camera_info = self.camera_info
        return frame


# ------------------------------------------------------------------ hilo Qt
def _make_tracker_thread_class():
    from PySide6.QtCore import QThread, Signal

    class TrackerThread(QThread):
        """Captura e inferencia en un hilo propio; emite TrackingFrame sin bloquear la GUI.

        Dueño único de la cámara: abre CameraSource al iniciar el hilo y la libera al terminar.
        `camera_factory`, `tracker_factory` y `clock` son inyectables (versiones falsas en tests).
        """

        frame_ready = Signal(object)  # TrackingFrame
        failed = Signal(str)  # mensaje de error legible

        def __init__(
            self,
            camera_index: int = 0,
            model_path: Optional[str] = None,
            parent=None,
            camera_factory: Optional[Callable[[int], object]] = None,
            tracker_factory: Optional[Callable[[Optional[str]], object]] = None,
            clock: Callable[[], float] = time.monotonic,
            camera_size: Tuple[int, int] = FALLBACK_SIZE,
        ):
            super().__init__(parent)
            self.camera_index = camera_index
            self.model_path = model_path
            w, h = int(camera_size[0]), int(camera_size[1])
            self._camera_factory = camera_factory or (lambda i: CameraSource(i, w, h))
            self._tracker_factory = tracker_factory or FaceTracker
            self._clock = clock
            self._stop_requested = False

        def start(self, *args):
            self._stop_requested = False  # antes de arrancar, para que un stop() temprano no se pierda
            super().start(*args)

        def run(self):
            logger.info("Hilo de seguimiento: inicio (cámara %s)", self.camera_index)
            camera = self._camera_factory(self.camera_index)
            tracker = None
            try:
                tracker = self._tracker_factory(self.model_path)
                camera.open()
                logger.info("Cámara %s abierta", self.camera_index)
                pipeline = EyeTrackingPipeline(camera, tracker, clock=self._clock)
                misses = 0
                last_fps_log = None
                while not self._stop_requested:
                    frame = pipeline.step()
                    if frame is None:
                        misses += 1
                        if misses > 60:
                            raise RuntimeError("La cámara dejó de entregar imagen.")
                        self.msleep(10)
                        continue
                    misses = 0
                    if frame.fps and (last_fps_log is None or frame.timestamp - last_fps_log >= 10.0):
                        last_fps_log = frame.timestamp
                        log = logger.warning if frame.fps < MIN_CAMERA_FPS else logger.info
                        log("Cámara %s: %.1f fps medidos con timestamps", self.camera_index, frame.fps)
                    self.frame_ready.emit(frame)
            except Exception as e:  # noqa: BLE001 - se informa a la GUI
                logger.exception("Error en el hilo de seguimiento")
                self.failed.emit(str(e))
            finally:
                camera.release()
                logger.info("Cámara %s liberada", self.camera_index)
                if tracker is not None:
                    tracker.close()
                logger.info("Hilo de seguimiento: fin")

        def stop(self, timeout_ms: int = 5000) -> bool:
            """Pide detener el hilo y espera a que termine (y libere la cámara). True si terminó."""
            self._stop_requested = True
            finished = self.wait(timeout_ms)
            if not finished:
                logger.error("El hilo de seguimiento no terminó en %d ms", timeout_ms)
            return finished

    return TrackerThread


_TRACKER_THREAD_CLASS = None


def TrackerThread(*args, **kwargs):  # noqa: N802 - fábrica para no importar Qt al importar el módulo
    global _TRACKER_THREAD_CLASS
    if _TRACKER_THREAD_CLASS is None:
        _TRACKER_THREAD_CLASS = _make_tracker_thread_class()
    return _TRACKER_THREAD_CLASS(*args, **kwargs)


# ------------------------------------------------------- modo de depuración
def draw_debug_overlay(image: np.ndarray, frame: TrackingFrame, blink_durations_ms, fps: float) -> np.ndarray:
    """Dibuja sobre una copia del cuadro: ojos/iris, EAR, estado, parpadeos recientes y FPS."""
    import cv2

    out = image.copy()
    h, w = out.shape[:2]
    if frame.face_found and frame.landmarks is not None:
        lm = frame.landmarks
        for idx in RIGHT_EYE_EAR + LEFT_EYE_EAR:
            cv2.circle(out, (int(lm[idx, 0] * w), int(lm[idx, 1] * h)), 2, (0, 255, 255), -1)
        for idx in RIGHT_IRIS_RING + LEFT_IRIS_RING:
            cv2.circle(out, (int(lm[idx, 0] * w), int(lm[idx, 1] * h)), 1, (0, 255, 0), -1)
    lines = [
        f"FPS: {fps:.1f}",
        "Rostro: " + ("detectado" if frame.face_found else "NO detectado"),
        f"EAR izq {frame.ear_left:.3f}  der {frame.ear_right:.3f}  prom {frame.ear_mean:.3f}",
        "Ojos: " + ("CERRADOS" if frame.eyes_closed else "abiertos"),
        "Parpadeos (ms): " + ", ".join(f"{d:.0f}" for d in list(blink_durations_ms)[-5:]),
        "ESC o Q para salir",
    ]
    for i, text in enumerate(lines):
        cv2.putText(out, text, (10, 22 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, text, (10, 22 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def run_tracker_debug(camera_index: int = 0, camera_size: Tuple[int, int] = FALLBACK_SIZE) -> int:
    """Ventana de OpenCV con la cámara y el seguimiento. Prueba manual de la etapa 2."""
    import cv2

    try:
        tracker = FaceTracker()
    except ModelNotFoundError as e:
        print(e)
        return 1
    camera = CameraSource(camera_index, camera_size[0], camera_size[1])
    try:
        camera.open()
    except RuntimeError as e:
        tracker.close()
        print(e)
        return 1
    pipeline = EyeTrackingPipeline(camera, tracker)
    durations: deque = deque(maxlen=10)
    last = time.monotonic()
    fps = 0.0
    try:
        while True:
            frame = pipeline.step()
            if frame is None:
                if cv2.waitKey(10) & 0xFF in (27, ord("q")):
                    break
                continue
            now = time.monotonic()
            dt = now - last
            last = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps else 1.0 / dt
            if frame.blink_event is not None:
                durations.append(frame.blink_event.duration_ms)
            shown = draw_debug_overlay(frame.image, frame, durations, fps)
            cv2.imshow("MiraLink - depuración del seguimiento", shown)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
    finally:
        camera.release()
        tracker.close()
        cv2.destroyAllWindows()
    return 0
