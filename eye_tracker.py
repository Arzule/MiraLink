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

# Blendshapes de MediaPipe que se guardan por cuadro (TrackingFrame.eye_blendshapes y las sesiones):
# sirven para comprobar con datos reales si distinguen un parpadeo de mirar hacia abajo.
EYE_BLENDSHAPES = (
    "eyeBlinkLeft",
    "eyeBlinkRight",
    "eyeLookDownLeft",
    "eyeLookDownRight",
    "eyeLookUpLeft",
    "eyeLookUpRight",
    "eyeSquintLeft",
    "eyeSquintRight",
)


class ModelNotFoundError(RuntimeError):
    """El modelo de MediaPipe no existe y no se pudo descargar."""


# ------------------------------------------------------------------ datos
@dataclass
class BlinkEvent:
    """Parpadeo completo: cierre y reapertura de los ojos."""

    start_ts: float
    end_ts: float
    duration_ms: float
    # El cierre empezó mirando más abajo que la fila más baja calibrada, donde el párpado baja solo
    # (GazeEarModel.guard_active): el CursorController no hace clic con él.
    gaze_suspect: bool = False


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
    eye_blendshapes: Optional[np.ndarray] = None  # EYE_BLENDSHAPES (NaN si falta alguno); se graba en las sesiones
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


EYES = ("both", "right", "left")  # como calibration.EYES (ojo con el que se sigue la mirada)


def gaze_perp(features: Optional[np.ndarray], eye: str = "both") -> Optional[float]:
    """Altura del iris en el ojo (iris_perp del ojo, o media de los dos; positivo = abajo) o None."""
    if features is None or len(features) < 4:
        return None
    if eye == "right":
        value = float(features[1])
    elif eye == "left":
        value = float(features[3])
    else:
        value = (float(features[1]) + float(features[3])) / 2.0
    return value if math.isfinite(value) else None


def frame_ear(frame: "TrackingFrame", eye: str = "both") -> float:
    """EAR del ojo con el que se sigue la mirada (el promedio con "both"). Con un solo ojo funcional, el
    promedio mezclaría el otro ojo (cerrado o sin parpadeo normal) y daría falsos clics o ningún clic."""
    if eye == "right":
        return frame.ear_right
    if eye == "left":
        return frame.ear_left
    return frame.ear_mean


def build_tracking_frame(
    timestamp: float,
    landmarks: Optional[np.ndarray],
    size: Tuple[int, int],
    matrix: Optional[np.ndarray] = None,
    blink_score: Optional[float] = None,
    eye_blendshapes: Optional[np.ndarray] = None,
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
        eye_blendshapes=eye_blendshapes,
    )


# ---------------------------------------------------------- BlinkDetector
class BlinkDetector:
    """Detecta parpadeos desde el EAR (el promedio, o el del ojo elegido: frame_ear) con línea base adaptativa.

    - Línea base: percentil `baseline_percentile` del EAR reciente con los ojos abiertos
      (ventana móvil de `window_s`); no se actualiza mientras los ojos están cerrados.
    - Cierre cuando EAR < close_ratio * base; reapertura cuando EAR > reopen_ratio * base
      (histéresis). Hasta tener `min_samples` muestras abiertas no se detecta nada.
    - Los eventos más cortos que `min_event_ms` se descartan (ruido de un solo cuadro).
    - Si se pierde el rostro se reinicia el estado sin emitir evento.
    - Con `gaze_model` (GazeEarModel de la calibración) y `gaze_perp` en cada cuadro, el EAR se divide
      por la apertura esperada a esa altura de la mirada: al mirar abajo el párpado baja (en sesiones
      reales el EAR caía a la mitad) y sin esto se tomaba por un cierre. El cierre se decide con la
      apertura esperada mayor entre la altura actual y la mediana de los últimos `gaze_window` cuadros
      (un parpadeo desplaza el iris hacia abajo en el mismo cuadro); la reapertura, con la actual.
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
        gaze_model=None,
        gaze_window: int = 3,
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
        self.gaze_model = gaze_model
        self._perps: deque = deque(maxlen=max(1, int(gaze_window)))
        self._suspect = False

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
        self._perps.clear()
        self._suspect = False

    def _update_baseline(self):
        if len(self._samples) >= self.min_samples:
            values = [v for _, v in self._samples]
            self._baseline = float(np.percentile(values, self.baseline_percentile))

    def _gaze_factors(self, gaze_perp: Optional[float]) -> Tuple[float, float, Optional[float]]:
        """(factor para cerrar, factor para reabrir, mediana reciente de la altura del iris)."""
        model = self.gaze_model
        if model is None or gaze_perp is None or not math.isfinite(gaze_perp):
            return 1.0, 1.0, None
        self._perps.append(float(gaze_perp))
        recent = float(np.median(self._perps))
        now = float(model.factor(gaze_perp))
        return max(now, float(model.factor(recent))), now, recent

    def update(
        self, timestamp: float, ear: float, face_found: bool = True, gaze_perp: Optional[float] = None
    ) -> Optional[BlinkEvent]:
        """Procesa un cuadro. Devuelve BlinkEvent cuando el ojo se reabre tras un parpadeo válido.
        `gaze_perp` (eye_tracker.gaze_perp de los rasgos) solo se usa con `gaze_model`."""
        if not face_found:
            # Se conserva la línea base (la persona sigue siendo la misma) pero no el cierre
            self._closed = False
            self._closed_since = None
            self._perps.clear()
            return None

        close_factor, open_factor, recent = self._gaze_factors(gaze_perp)
        ear = ear / open_factor  # EAR normalizado a la apertura de la mirada de referencia
        if not self._closed:
            self._samples.append((timestamp, ear))
            while self._samples and timestamp - self._samples[0][0] > self.window_s:
                self._samples.popleft()
            self._update_baseline()
            base = self._baseline
            if base is None or base < self.min_baseline:
                return None
            if ear * open_factor / close_factor < self.close_ratio * base:
                self._closed = True
                self._closed_since = timestamp
                model = self.gaze_model
                self._suspect = bool(
                    model is not None and recent is not None and model.guard_active and model.beyond_low_end(recent)
                )
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
            return BlinkEvent(start_ts=start, end_ts=timestamp, duration_ms=duration_ms, gaze_suspect=self._suspect)
        return None


# ------------------------------------------------------------- hardware
FALLBACK_SIZE = (640, 480)
MIN_CAMERA_FPS = 25.0  # por debajo la detección ocular se degrada (hallazgo de la prueba manual)
MIN_PROBE_FPS = 24.0  # un modo que no llega a esto al abrir se descarta (salvo NATIVE_RELATIVE_FPS)
# Si un tamaño mayor da al menos este 90 % de los fps de 640x480 con la misma cámara y la misma luz, se usa
# aunque no llegue a MIN_PROBE_FPS: lo que limita es la exposición, no la resolución, y bajar no gana nada
# (TE-9072 del usuario, 30/10: exposición automática ~21 fps tanto a 1080p MJPG como a 480p).
NATIVE_RELATIVE_FPS = 0.9
# Resolución nativa: se piden estos tamaños de mayor a menor (con MJPG) y se queda el mayor que la cámara
# entrega a >= MIN_PROBE_FPS. Tope 1920x1080: más píxeles cuestan CPU (decodificar y MediaPipe) y el
# modelo de landmarks trabaja con un recorte de 256x256 de la cara, así que la ganancia se agota antes.
NATIVE_CANDIDATES = ((1920, 1080), (1280, 960), (1280, 720), (800, 600), FALLBACK_SIZE)
# (índice, exposición) -> (tamaño elegido, fps de 640x480 si se eligió por NATIVE_RELATIVE_FPS o None):
# las aperturas siguientes no repiten la búsqueda ni la medición de referencia
_NATIVE_CACHE: dict = {}
# Exposición: "auto" (la de la cámara) o "fast": EXPOSURE_FAST (log2 de segundos en Media Foundation:
# -5 = 1/32 s), la más larga que permite 30 fps. Con la TE-9072: 31 fps frente a ~21-25 en automática, sin
# diferencia medible en el ruido del iris, pero la imagen queda muy oscura (brillo ~25/255).
CAMERA_EXPOSURES = ("auto", "fast")
EXPOSURE_FAST = -5.0
MAX_START_MISSES = 10  # lecturas fallidas toleradas al arrancar (Media Foundation puede tardar en dar el primer cuadro)
CAMERA_BUSY_HINT = "Cierra la app Cámara u otras aplicaciones que la estén usando."

# Media Foundation: tarda menos en abrir sin las transformaciones por hardware
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")


def _fourcc_text(value) -> str:
    try:
        code = int(value)
    except (TypeError, ValueError):
        return ""
    text = "".join(chr((code >> (8 * i)) & 0xFF) for i in range(4))
    # solo letras y números ASCII (YUY2, MJPG, NV12): un código sin sentido salía como "}ë6ä" en la UI
    if not text.strip() or not all(c == " " or (c.isascii() and c.isalnum()) for c in text):
        return ""
    return text.strip()


def _area(size) -> int:
    return int(size[0]) * int(size[1])


class CameraSource:
    """Captura de cámara con cv2.VideoCapture a 30 fps: Media Foundation y, si no abre, DirectShow.

    - Backend (diagnóstico con la TE-9072 del usuario, 30/10): con DirectShow la cámara ignora el pedido
      de MJPG y entrega YUY2 sin comprimir (1080p a 2.2 fps); pedir MJPG antes del tamaño a 720p llegó a
      colgar el driver hasta reconectarla. Con Media Foundation y MJPG pedido DESPUÉS del tamaño entrega
      1080p a la misma tasa que 480p. Por eso: Media Foundation (MJPG tras el tamaño) y DirectShow solo de
      respaldo y nunca con MJPG.
    - Resolución nativa (`width`/`height` = None, lo que usa la aplicación por defecto): prueba
      NATIVE_CANDIDATES de mayor a menor, midiendo los fps con el reloj en `probe_frames` cuadros, y se
      queda con el mayor tamaño ENTREGADO que sostiene MIN_PROBE_FPS o, si ninguno llega, el mayor que da
      NATIVE_RELATIVE_FPS de los fps de 640x480 (la exposición limita igual). Si no, 640x480 sin MJPG. La
      elección se recuerda por índice y exposición durante la ejecución.
    - Tamaño fijo: 640x480 sin MJPG; otro tamaño con MJPG y se acepta lo que entregue la cámara si sostiene
      MIN_PROBE_FPS (o NATIVE_RELATIVE_FPS de 640x480); si no, vuelve a 640x480 y lo anota en `fallback_reason`.
    - `exposure`: "auto" o "fast" (EXPOSURE_FAST, ver CAMERA_EXPOSURES).
    `cv2_module` y `clock` se inyectan en los tests (nunca se abre la cámara real en ellos).
    """

    def __init__(
        self,
        index: int = 0,
        width: Optional[int] = FALLBACK_SIZE[0],
        height: Optional[int] = FALLBACK_SIZE[1],
        fps: int = 30,
        cv2_module=None,
        clock: Callable[[], float] = time.monotonic,
        probe_frames: int = 12,
        use_cache: bool = True,
        exposure: str = "auto",
    ):
        if exposure not in CAMERA_EXPOSURES:
            raise ValueError(f"exposición desconocida: {exposure}")
        self.index = index
        self.native = width is None or height is None
        self.width = None if self.native else int(width)
        self.height = None if self.native else int(height)
        self.fps = fps
        self.exposure = exposure
        self._cv2 = cv2_module
        self._clock = clock
        self.probe_frames = max(3, int(probe_frames))
        self.use_cache = use_cache
        self._cap = None
        self.actual_size: Optional[Tuple[int, int]] = None
        self.fourcc = ""
        self.backend = ""
        self.measured_fps = 0.0
        self.fallback_reason = ""
        self.exposure_limited = False  # se aceptó un tamaño por NATIVE_RELATIVE_FPS
        self._reference_fps: Optional[float] = None

    @staticmethod
    def _apis(cv2):
        """(nombre, API) en orden de preferencia; un cv2 sin Media Foundation solo usa DirectShow."""
        apis = []
        if hasattr(cv2, "CAP_MSMF"):
            apis.append(("MSMF", cv2.CAP_MSMF))
        apis.append(("DSHOW", cv2.CAP_DSHOW))
        return apis

    def _open_capture(self, cv2, size: Tuple[int, int], mjpg: bool):
        for name, api in self._apis(cv2):
            cap = cv2.VideoCapture(self.index, api)
            if not cap.isOpened():
                cap.release()
                continue
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
            if mjpg and name != "DSHOW":  # después del tamaño; con DirectShow nunca (ver docstring)
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FPS, self.fps)
            if hasattr(cv2, "CAP_PROP_AUTO_EXPOSURE"):
                if self.exposure == "fast":
                    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # manual (convención de OpenCV)
                    cap.set(cv2.CAP_PROP_EXPOSURE, EXPOSURE_FAST)
                else:
                    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)  # automática (la cámara puede recordar la manual)
            self.backend = name
            return cap
        return None

    def _probe(self, cap) -> Tuple[bool, Optional[Tuple[int, int]], float]:
        """Lee unos cuadros: (ok, tamaño entregado, fps medidos con la mediana de los intervalos)."""
        times, size, misses = [], None, 0
        while len(times) < self.probe_frames:
            ok, frame = cap.read()
            if not ok or frame is None:
                misses += 1
                if misses > MAX_START_MISSES:
                    return False, size, 0.0
                continue
            size = (int(frame.shape[1]), int(frame.shape[0]))
            times.append(self._clock())
        gaps = np.diff(times[1:])  # el primer cuadro suele tardar más (exposición automática)
        gaps = gaps[gaps > 0]
        fps = float(1.0 / np.median(gaps)) if len(gaps) else 0.0
        return True, size, fps

    def _try(self, cv2, size: Tuple[int, int]):
        """Abre a `size` (MJPG salvo 640x480) y sondea: (cap o None, ok, tamaño entregado, fps)."""
        cap = self._open_capture(cv2, size, mjpg=tuple(size) != FALLBACK_SIZE)
        if cap is None:
            return None, False, None, 0.0
        ok, delivered, fps = self._probe(cap)
        return cap, ok, delivered, fps

    def _open_fallback(self, cv2):
        cap, ok, size, fps = self._try(cv2, FALLBACK_SIZE)
        if cap is None:
            raise RuntimeError(f"No se pudo abrir la cámara {self.index}. {CAMERA_BUSY_HINT}")
        return cap, ok, size, fps

    def _reference(self, cv2) -> float:
        """fps a 640x480 sin MJPG con la misma luz (se abre y se cierra; se mide una vez por apertura)."""
        if self._reference_fps is None:
            cap, ok, _size, fps = self._try(cv2, FALLBACK_SIZE)
            if cap is not None:
                cap.release()
            self._reference_fps = fps if ok else 0.0
        return self._reference_fps

    def _relative_known_ok(self, delivered, fps: float) -> bool:
        """Con la referencia de 640x480 ya medida (o recordada), ¿se acepta este tamaño sin reabrir?"""
        ref = self._reference_fps
        return bool(
            ref and delivered is not None and _area(delivered) > _area(FALLBACK_SIZE) and fps >= NATIVE_RELATIVE_FPS * ref
        )

    def _relative_retry(self, cv2, size: Tuple[int, int], delivered, fps: float):
        """Un tamaño mayor que 640x480 que no llega a MIN_PROBE_FPS se acepta si da NATIVE_RELATIVE_FPS de
        los fps de 640x480: se reabre (la cámara ya se cerró) y se comprueba. Devuelve el resultado o None."""
        if delivered is None or _area(delivered) <= _area(FALLBACK_SIZE) or fps <= 0:
            return None
        ref = self._reference(cv2)
        if ref <= 0 or fps < NATIVE_RELATIVE_FPS * ref:
            return None
        cap, ok, again, fps2 = self._try(cv2, size)
        if cap is not None and ok and tuple(again) == tuple(delivered) and fps2 >= NATIVE_RELATIVE_FPS * ref:
            self.exposure_limited = True
            logger.info("Cámara %s: %dx%d a %.1f fps (640x480: %.1f): limita la exposición, no la resolución",
                        self.index, again[0], again[1], fps2, ref)
            return cap, ok, again, fps2
        if cap is not None:
            cap.release()
        return None

    def _open_fixed(self, cv2):
        requested = (self.width, self.height)
        if requested == FALLBACK_SIZE:
            return self._open_fallback(cv2)
        cap, ok, size, fps = self._try(cv2, requested)
        if cap is None:
            raise RuntimeError(f"No se pudo abrir la cámara {self.index}. {CAMERA_BUSY_HINT}")
        if ok and fps >= MIN_PROBE_FPS:
            return cap, ok, size, fps  # se acepta el tamaño entregado aunque no sea el pedido
        cap.release()
        if ok:
            retried = self._relative_retry(cv2, requested, size, fps)
            if retried is not None:
                return retried
        if not ok:
            reason = f"la cámara no entregó imagen a {requested[0]}x{requested[1]}"
        else:
            reason = f"{size[0]}x{size[1]} solo a {fps:.1f} fps"
        logger.warning("Cámara %s: %s; se usa %dx%d", self.index, reason, *FALLBACK_SIZE)
        self.fallback_reason = reason
        return self._open_fallback(cv2)

    def _open_native(self, cv2):
        """Busca el mayor tamaño entregado que sostiene MIN_PROBE_FPS (ver docstring de la clase)."""
        candidates = list(NATIVE_CANDIDATES)
        key = (self.index, self.exposure)
        cached = _NATIVE_CACHE.get(key) if self.use_cache else None
        if cached is not None:
            cached, self._reference_fps = cached
            candidates = [cached] + [c for c in candidates if _area(c) < _area(cached)]
        best = None  # (tamaño pedido, tamaño entregado, fps)
        best_cap = None
        failed_area = None  # el menor tamaño entregado que no llegó a los fps
        notes = []
        opened_any = False
        for size in candidates:
            if tuple(size) == FALLBACK_SIZE:
                break  # 640x480 sin MJPG queda como último recurso
            if best is not None and _area(size) <= _area(best[1]):
                break  # ya no se puede mejorar
            if failed_area is not None and _area(size) >= failed_area:
                continue  # la cámara volvería a entregar un modo demasiado lento
            if best_cap is not None:
                best_cap.release()  # la misma cámara no se abre dos veces a la vez
                best_cap = None
            cap, ok, delivered, fps = self._try(cv2, size)
            if cap is None:
                if not opened_any:
                    raise RuntimeError(f"No se pudo abrir la cámara {self.index}. {CAMERA_BUSY_HINT}")
                continue
            opened_any = True
            if ok and fps >= MIN_PROBE_FPS and (best is None or _area(delivered) > _area(best[1])):
                best, best_cap = (tuple(size), delivered, fps), cap
                if tuple(delivered) == tuple(size):
                    break  # la cámara dio exactamente lo pedido: es el mayor que queda
                continue
            if ok and best is None and self._relative_known_ok(delivered, fps):
                self.exposure_limited = True  # referencia recordada: sin volver a medir 640x480 ni reabrir
                best, best_cap = (tuple(size), tuple(delivered), fps), cap
                break
            cap.release()
            if ok and best is None:
                retried = self._relative_retry(cv2, size, delivered, fps)
                if retried is not None:
                    best_cap, _ok, delivered, fps = retried
                    best = (tuple(size), tuple(delivered), fps)
                    break  # con la exposición como límite, los tamaños menores no darían más fps
            if not ok:
                notes.append(f"sin imagen a {size[0]}x{size[1]}")
                failed_area = _area(size) if failed_area is None else min(failed_area, _area(size))
            elif fps < MIN_PROBE_FPS:
                notes.append(f"{delivered[0]}x{delivered[1]} solo a {fps:.1f} fps")
                area = _area(delivered)
                failed_area = area if failed_area is None else min(failed_area, area)
        if best is not None:
            if best_cap is None:  # se probó algo después y hay que volver al mejor
                best_cap, ok, delivered, fps = self._try(cv2, best[0])
                if best_cap is not None and ok and tuple(delivered) == tuple(best[1]) and fps >= MIN_PROBE_FPS:
                    result = (best_cap, ok, delivered, fps)
                else:
                    if best_cap is not None:
                        best_cap.release()
                    result = None
            else:
                result = (best_cap, True, best[1], best[2])
            if result is not None:
                if self.use_cache:
                    _NATIVE_CACHE[key] = (best[0], self._reference_fps if self.exposure_limited else None)
                self.fallback_reason = "" if self.exposure_limited else "; ".join(notes)
                return result
        reason = "; ".join(notes) or "ningún modo mayor sostiene 24 fps"
        logger.warning("Cámara %s: resolución nativa: %s; se usa %dx%d", self.index, reason, *FALLBACK_SIZE)
        self.fallback_reason = reason
        if self.use_cache:
            _NATIVE_CACHE[key] = (FALLBACK_SIZE, None)
        return self._open_fallback(cv2)

    def open(self):
        if self._cv2 is None:
            import cv2

            self._cv2 = cv2
        cv2 = self._cv2
        self.fallback_reason = ""
        self.exposure_limited = False
        self._reference_fps = None
        cap, ok, size, fps = self._open_native(cv2) if self.native else self._open_fixed(cv2)
        if not ok:
            cap.release()
            raise RuntimeError(f"La cámara {self.index} se abrió pero no entrega imagen. {CAMERA_BUSY_HINT}")
        self.actual_size, self.measured_fps = size, fps
        self.fourcc = _fourcc_text(cap.get(cv2.CAP_PROP_FOURCC))
        self._cap = cap
        logger.info(
            "Cámara %s: %dx%d %s a %.1f fps medidos (%s, %s, exposición %s)",
            self.index, size[0], size[1], self.fourcc or "?", fps, self.backend,
            "resolución nativa" if self.native else f"pedido {self.width}x{self.height}", self.exposure,
        )

    def describe(self) -> str:
        """Resumen legible de lo que entrega la cámara (para la UI y las sesiones grabadas)."""
        if self.actual_size is None:
            return ""
        text = f"{self.actual_size[0]}x{self.actual_size[1]}"
        if self.fourcc:
            text += f" {self.fourcc}"
        if self.native:
            text += " (nativa"
            text += f"; descartado: {self.fallback_reason})" if self.fallback_reason else ")"
        elif self.fallback_reason:
            text += f" (sin {self.width}x{self.height}: {self.fallback_reason})"
        elif self.actual_size != (self.width, self.height):
            text += f" (pedido {self.width}x{self.height})"
        if self.exposure == "fast":
            text += ", exposición rápida"
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
    from app_paths import resource_path

    return resource_path("models", DEFAULT_MODEL_FILENAME)  # en el .exe: empaquetado dentro del ejecutable


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
        eye_shapes = None
        if result.face_blendshapes:
            scores = {c.category_name: c.score for c in result.face_blendshapes[0]}
            if "eyeBlinkLeft" in scores and "eyeBlinkRight" in scores:
                blink_score = (scores["eyeBlinkLeft"] + scores["eyeBlinkRight"]) / 2.0
            eye_shapes = np.array([scores.get(n, np.nan) for n in EYE_BLENDSHAPES], dtype=float)
        return build_tracking_frame(timestamp, landmarks, (w, h), matrix, blink_score, eye_shapes)

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


# Red de mirada (híbrido iris + red, config.json tracking.gaze_model): mediana de las últimas estimaciones
# (la red tiembla ~3° por cuadro; con la mediana de 5 la validación de la sesión 230535 bajó de 45 a 33 px).
NET_MEDIAN_WINDOW = 5
NET_THREADS = 4  # hilos de ONNX Runtime: MediaPipe corre en el mismo hilo de seguimiento


class AsyncGazeNet:
    """Ejecuta la red de mirada en un hilo propio para no sumar su tiempo al de MediaPipe.

    Con la red en el mismo hilo la cámara bajó de 28.8 a 21 fps (sesión 000755): cada cuadro esperaba a
    MediaPipe y después a la red. Aquí `submit` entrega una cara si el hilo está libre (si no, ese cuadro
    no se evalúa) y `take` devuelve los resultados terminados. ONNX Runtime suelta el GIL al calcular, así
    que la red corre en paralelo con el siguiente cuadro de MediaPipe (llega ~1 cuadro tarde).
    """

    def __init__(self, net):
        import threading

        self.net = net
        self.input_size = net.input_size
        self.name = getattr(net, "name", "")
        self._cv = threading.Condition()
        self._pending = None  # (generación, cara)
        self._busy = False
        self._results: List[Tuple[int, np.ndarray]] = []
        self._error: Optional[BaseException] = None
        self._stop = False
        self._thread = threading.Thread(target=self._run, name="red-de-mirada", daemon=True)
        self._thread.start()

    @property
    def idle(self) -> bool:
        with self._cv:
            return not self._busy and self._pending is None

    def submit(self, generation: int, face: np.ndarray) -> bool:
        with self._cv:
            if self._busy or self._pending is not None or self._stop:
                return False
            self._pending = (generation, face)
            self._cv.notify()
            return True

    def take(self) -> List[Tuple[int, np.ndarray]]:
        """Resultados terminados desde la última llamada; relanza aquí un error del hilo de la red."""
        with self._cv:
            if self._error is not None:
                raise RuntimeError(f"Falló la red de mirada: {self._error}") from self._error
            out, self._results = self._results, []
            return out

    def wait_idle(self, timeout: float = 2.0) -> bool:
        """Espera a que termine la cara en curso (para tests deterministas)."""
        with self._cv:
            return self._cv.wait_for(lambda: not self._busy and self._pending is None, timeout)

    def close(self):
        with self._cv:
            self._stop = True
            self._cv.notify_all()
        self._thread.join(timeout=2.0)

    def _run(self):
        while True:
            with self._cv:
                self._cv.wait_for(lambda: self._pending is not None or self._stop)
                if self._stop:
                    return
                generation, face = self._pending
                self._pending = None
                self._busy = True
            try:
                out = np.asarray(self.net.predict([face])[0], dtype=float)
            except Exception as e:  # noqa: BLE001 - se relanza en take()
                with self._cv:
                    self._error = e
                    self._busy = False
                    self._cv.notify_all()
                return
            with self._cv:
                self._results.append((generation, out))
                self._busy = False
                self._cv.notify_all()


class NetGazeStage:
    """Añade al vector de rasgos [giro, inclinación] de una red de mirada (calibration.NET_COLUMNS).

    `net` tiene predict(caras BGR) -> (n, 2) e input_size (neural_gaze.GazeNet; falsa en tests). La cara
    sale de la caja de los landmarks del cuadro. Sin cara o si no se puede recortar, el cuadro queda sin
    rasgos (como sin rostro) y la mediana se vacía. Con una AsyncGazeNet (tiene `submit`) la red corre en
    otro hilo: cada cuadro recoge los resultados terminados, entrega su cara si la red está libre y usa la
    mediana de las últimas estimaciones; los resultados de antes de perder la cara se descartan
    (generación) y, hasta el primer resultado, el cuadro queda sin rasgos.
    """

    def __init__(self, net, window: int = NET_MEDIAN_WINDOW):
        self.net = net
        self.is_async = hasattr(net, "submit")
        self._recent: deque = deque(maxlen=max(1, int(window)))
        self._generation = 0

    def _lose(self, frame: Optional["TrackingFrame"] = None):
        self._recent.clear()
        self._generation += 1
        if frame is not None:
            frame.face_found = False
            frame.features = None

    def apply(self, frame: "TrackingFrame", image: np.ndarray) -> None:
        if not frame.face_found or frame.features is None or frame.landmarks is None:
            self._lose()
            return
        from neural_gaze import face_input_from_image

        if self.is_async:
            for generation, out in self.net.take():
                if generation == self._generation and np.isfinite(out).all():
                    self._recent.append(out)
            if self.net.idle:
                face = face_input_from_image(image, frame.landmarks, self.net.input_size)
                if face is not None:
                    self.net.submit(self._generation, face)
            if not self._recent:
                frame.features = None  # aún sin estimación de la red para esta cara
                frame.face_found = False
                return
        else:
            face = face_input_from_image(image, frame.landmarks, self.net.input_size)
            out = self.net.predict([face])[0] if face is not None else None
            if out is None or not np.isfinite(out).all():
                self._lose(frame)
                return
            self._recent.append(np.asarray(out, dtype=float))
        feats = np.zeros(FEATURE_DIM)  # la red va siempre en las columnas FEATURE_DIM y FEATURE_DIM + 1
        base = np.asarray(frame.features, dtype=float)[:FEATURE_DIM]
        feats[: len(base)] = base
        frame.features = np.concatenate([feats, np.median(self._recent, axis=0)])


class EyeTrackingPipeline:
    """Cámara -> FaceTracker -> BlinkDetector. Los tres se inyectan (versiones falsas en tests).
    `eye` (EYES): ojo cuyo EAR y altura del iris usa el detector de parpadeos (el de la calibración)."""

    def __init__(
        self,
        camera,
        tracker,
        detector: Optional[BlinkDetector] = None,
        clock: Callable[[], float] = time.monotonic,
        blink_model=None,
        eye: str = "both",
        gaze_net=None,
    ):
        if eye not in EYES:
            raise ValueError(f"ojo desconocido: {eye}")
        self.camera = camera
        self.tracker = tracker
        self.net_stage = NetGazeStage(gaze_net) if gaze_net is not None else None  # híbrido iris + red
        self.eye = eye
        self.detector = detector or BlinkDetector(gaze_model=blink_model)
        self.clock = clock
        self.fps_meter = FpsMeter()
        describe = getattr(camera, "describe", None)
        self.camera_info = describe() if callable(describe) else ""
        if gaze_net is not None:  # se ve en "Cámara: ..." del modal y en la etiqueta de estado del control
            self.camera_info += f" · iris + red {getattr(gaze_net, 'name', '')}".rstrip()

    def step(self) -> Optional[TrackingFrame]:
        """Procesa un cuadro. Devuelve None si la cámara no entregó imagen."""
        ok, image = self.camera.read()
        if not ok or image is None:
            return None
        ts = self.clock()
        frame = self.tracker.process(image, ts)
        frame.image = image
        if self.net_stage is not None:
            self.net_stage.apply(frame, image)
        event = self.detector.update(ts, frame_ear(frame, self.eye), frame.face_found, gaze_perp(frame.features, self.eye))
        frame.eyes_closed = self.detector.is_closed
        frame.blink_event = event
        frame.fps = self.fps_meter.update(ts)
        frame.camera_info = self.camera_info
        return frame


# ------------------------------------------------------------------ hilo Qt
def _default_net_factory(name: str):
    """Carga la red ONNX de neural_gaze.GAZE_MODELS (error legible si no existe o falta el archivo)."""
    from neural_gaze import GAZE_MODELS, GazeNet

    if name not in GAZE_MODELS:
        raise ValueError(f"Red de mirada desconocida: {name}")
    try:
        return GazeNet(GAZE_MODELS[name].path, threads=NET_THREADS)
    except FileNotFoundError as e:
        raise RuntimeError(
            f"Falta el modelo de la red de mirada {name}. Descárgalo con: "
            r".\venv\Scripts\python.exe scripts\download_gaze_models.py "
            "o pon tracking.gaze_model en \"iris\" en config.json."
        ) from e


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
            camera_size: Optional[Tuple[int, int]] = FALLBACK_SIZE,
            blink_model=None,
            eye: str = "both",
            camera_exposure: str = "auto",
            gaze_model: Optional[str] = None,
            net_factory: Optional[Callable[[str], object]] = None,
        ):
            super().__init__(parent)
            self.gaze_model = gaze_model  # red de mirada (neural_gaze.GAZE_MODELS) o None = solo iris
            self._net_factory = net_factory or _default_net_factory
            self.camera_index = camera_index
            self.model_path = model_path
            self.blink_model = blink_model  # GazeEarModel de la calibración (o None)
            self.eye = eye  # ojo de la calibración: su EAR decide los parpadeos
            # camera_size None = resolución nativa (CameraSource elige la mayor que sostiene ~30 fps)
            w, h = (None, None) if camera_size is None else (int(camera_size[0]), int(camera_size[1]))
            self._camera_factory = camera_factory or (lambda i: CameraSource(i, w, h, exposure=camera_exposure))
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
            net = None
            try:
                tracker = self._tracker_factory(self.model_path)
                if self.gaze_model:
                    net = AsyncGazeNet(self._net_factory(self.gaze_model))  # en su hilo: no baja los fps
                    logger.info("Red de mirada cargada: %s", self.gaze_model)
                camera.open()
                logger.info("Cámara %s abierta", self.camera_index)
                pipeline = EyeTrackingPipeline(
                    camera, tracker, clock=self._clock, blink_model=self.blink_model, eye=self.eye, gaze_net=net
                )
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
                if net is not None:
                    net.close()
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


def run_tracker_debug(
    camera_index: int = 0, camera_size: Optional[Tuple[int, int]] = None, camera_exposure: str = "auto"
) -> int:
    """Ventana de OpenCV con la cámara y el seguimiento. Prueba manual de la etapa 2."""
    import cv2

    try:
        tracker = FaceTracker()
    except ModelNotFoundError as e:
        print(e)
        return 1
    size = (None, None) if camera_size is None else camera_size
    camera = CameraSource(camera_index, *size, exposure=camera_exposure)
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
