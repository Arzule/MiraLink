"""Mirada por red neuronal (MobileGaze, ONNX) y recortes de la cara para evaluarla SIN cámara.

Fase de evaluación (30/10): todavía no se usa en la aplicación. `--record-session` guarda por cuadro un
recorte cuadrado de la cara (JPEG) para que `scripts/neural_gaze_eval.py` compare, con la misma sesión,
el método actual (iris de MediaPipe) frente a las redes. Solo se integrará la que mejore el error al
mover la cabeza sin empeorar la validación ni bajar de ~25 fps.

- Recorte de grabación: cuadrado de lado `RECORD_CROP_EXPAND` x el lado mayor de la caja de los
  landmarks, centrado en ella, a `RECORD_CROP_SIZE` px (lo que cae fuera del cuadro se rellena de negro).
  Se guarda también la caja (centro x, centro y, lado) en píxeles de la imagen para recortar después
  la caja ajustada que espera cada red.
- MobileGaze (yakhyo/gaze-estimation, MIT; entrenado con Gaze360, datos solo para investigación):
  entrada = caja de la cara (como la de RetinaFace) redimensionada a 448x448, RGB, /255 y normalización
  de ImageNet; salidas = logits de giro y de inclinación en 90 intervalos de 4 grados desde -180.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from app_paths import resource_path

RECORD_CROP_SIZE = 448
RECORD_CROP_EXPAND = 1.5
RECORD_JPEG_QUALITY = 90
FACE_BOX_EXPAND = 1.0  # caja de la red = caja de los landmarks (parecida a la de RetinaFace)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

GAZE360_BINS = 90
GAZE360_BIN_DEG = 4.0
GAZE360_OFFSET_DEG = -180.0

WEIGHTS_URL = "https://github.com/yakhyo/gaze-estimation/releases/download/weights/"


@dataclass(frozen=True)
class GazeModelInfo:
    name: str
    filename: str
    note: str

    @property
    def url(self) -> str:
        return WEIGHTS_URL + self.filename

    @property
    def path(self) -> str:
        return resource_path(os.path.join("models", self.filename))


GAZE_MODELS: Dict[str, GazeModelInfo] = {
    "mobileone_s0": GazeModelInfo("mobileone_s0", "mobileone_s0_gaze.onnx", "MobileOne S0, 4.8 MB, el más ligero"),
    "resnet18": GazeModelInfo("resnet18", "resnet18_gaze.onnx", "ResNet-18"),
    "resnet34": GazeModelInfo("resnet34", "resnet34_gaze.onnx", "ResNet-34, el más preciso en Gaze360"),
}


# ------------------------------------------------------------------ recortes
def landmark_box(landmarks: np.ndarray, image_size: Tuple[int, int]) -> Optional[Tuple[float, float, float, float]]:
    """Caja (x0, y0, x1, y1) en píxeles de los landmarks normalizados; None si no son finitos."""
    lm = np.asarray(landmarks, dtype=float)[:, :2]
    if lm.size == 0 or not np.isfinite(lm).all():
        return None
    w, h = image_size
    x0, y0 = lm.min(axis=0) * (w, h)
    x1, y1 = lm.max(axis=0) * (w, h)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    return float(x0), float(y0), float(x1), float(y1)


def crop_affine(image: np.ndarray, box: Tuple[float, float, float, float], out_w: int, out_h: int) -> np.ndarray:
    """Recorta la caja (x0, y0, x1, y1) redimensionada a out_w x out_h; fuera del cuadro queda negro."""
    import cv2

    x0, y0, x1, y1 = box
    sx = out_w / max(x1 - x0, 1e-6)
    sy = out_h / max(y1 - y0, 1e-6)
    m = np.array([[sx, 0.0, -x0 * sx], [0.0, sy, -y0 * sy]], dtype=np.float64)
    interp = cv2.INTER_AREA if sx < 1.0 else cv2.INTER_LINEAR
    return cv2.warpAffine(image, m, (out_w, out_h), flags=interp, borderMode=cv2.BORDER_CONSTANT)


def record_box(landmarks: np.ndarray, image_size: Tuple[int, int]) -> Optional[Tuple[float, float, float]]:
    """(centro x, centro y, lado) del recorte de grabación en píxeles de la imagen."""
    box = landmark_box(landmarks, image_size)
    if box is None:
        return None
    x0, y0, x1, y1 = box
    side = RECORD_CROP_EXPAND * max(x1 - x0, y1 - y0)
    return (x0 + x1) / 2.0, (y0 + y1) / 2.0, side


def encode_record_crop(image: np.ndarray, landmarks: np.ndarray) -> Optional[Tuple[bytes, Tuple[float, float, float]]]:
    """JPEG del recorte de grabación y su caja (cx, cy, lado); None sin cara o si falla la codificación."""
    import cv2

    if image is None or landmarks is None:
        return None
    h, w = image.shape[:2]
    rb = record_box(landmarks, (w, h))
    if rb is None:
        return None
    cx, cy, side = rb
    crop = crop_affine(image, (cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2), RECORD_CROP_SIZE, RECORD_CROP_SIZE)
    ok, buf = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), RECORD_JPEG_QUALITY])
    if not ok:
        return None
    return buf.tobytes(), rb


def decode_jpeg(data: np.ndarray) -> Optional[np.ndarray]:
    import cv2

    if data is None or len(data) == 0:
        return None
    return cv2.imdecode(np.frombuffer(np.asarray(data, dtype=np.uint8).tobytes(), dtype=np.uint8), cv2.IMREAD_COLOR)


def face_input_from_record(
    crop: np.ndarray,
    rbox: Sequence[float],
    landmarks: np.ndarray,
    image_size: Tuple[int, int],
    size: int,
    expand: float = FACE_BOX_EXPAND,
) -> Optional[np.ndarray]:
    """Caja de la cara que espera la red, sacada del recorte grabado (coordenadas del recorte)."""
    box = landmark_box(landmarks, image_size)
    if box is None or crop is None:
        return None
    cx, cy, side = (float(v) for v in rbox)
    scale = crop.shape[1] / side  # píxeles del recorte por píxel de la imagen
    ox, oy = cx - side / 2.0, cy - side / 2.0
    x0, y0, x1, y1 = box
    mx, my = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    hw, hh = expand * (x1 - x0) / 2.0, expand * (y1 - y0) / 2.0
    sub = ((mx - hw - ox) * scale, (my - hh - oy) * scale, (mx + hw - ox) * scale, (my + hh - oy) * scale)
    return crop_affine(crop, sub, size, size)


def face_input_from_image(image: np.ndarray, landmarks: np.ndarray, size: int, expand: float = FACE_BOX_EXPAND) -> Optional[np.ndarray]:
    """Caja de la cara directamente del cuadro de la cámara (uso en vivo)."""
    h, w = image.shape[:2]
    box = landmark_box(landmarks, (w, h))
    if box is None:
        return None
    x0, y0, x1, y1 = box
    mx, my = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    hw, hh = expand * (x1 - x0) / 2.0, expand * (y1 - y0) / 2.0
    return crop_affine(image, (mx - hw, my - hh, mx + hw, my + hh), size, size)


# ------------------------------------------------------------------ red
def preprocess(faces_bgr: Sequence[np.ndarray]) -> np.ndarray:
    """Lote NCHW float32: BGR -> RGB, /255, normalización de ImageNet."""
    batch = np.stack([np.asarray(f)[:, :, ::-1] for f in faces_bgr]).astype(np.float32) / 255.0
    batch = (batch - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(batch.transpose(0, 3, 1, 2))


def decode_bins(logits: np.ndarray) -> np.ndarray:
    """Logits (n, 90) -> ángulo en radianes = esperanza de la softmax x 4 grados - 180."""
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=1, keepdims=True)
    idx = np.arange(z.shape[1], dtype=np.float64)
    deg = (p * idx).sum(axis=1) * GAZE360_BIN_DEG + GAZE360_OFFSET_DEG
    return np.radians(deg)


class GazeNet:
    """Red de mirada ONNX en CPU. predict(caras BGR) -> (n, 2) [giro, inclinación] en radianes."""

    def __init__(self, path: str, threads: int = 0):
        try:
            import onnxruntime as ort
        except ImportError as e:  # pragma: no cover - depende del entorno
            raise RuntimeError("Falta onnxruntime: instala requirements.txt") from e
        if not os.path.exists(path):
            raise FileNotFoundError(f"No está el modelo {path}. Descárgalo con scripts/download_gaze_models.py")
        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        self.session = ort.InferenceSession(path, sess_options=opts, providers=["CPUExecutionProvider"])
        self.name = os.path.basename(path).replace("_gaze.onnx", "")
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        shape = inp.shape
        self.input_size = int(shape[2]) if isinstance(shape[2], int) else 448
        self.dynamic_batch = not isinstance(shape[0], int)

    def predict(self, faces_bgr: Sequence[np.ndarray]) -> np.ndarray:
        if len(faces_bgr) == 0:
            return np.zeros((0, 2))
        if not self.dynamic_batch:
            return np.vstack([self._run([f]) for f in faces_bgr])
        return self._run(faces_bgr)

    def _run(self, faces) -> np.ndarray:
        outputs = self.session.run(None, {self.input_name: preprocess(faces)})
        return np.column_stack([decode_bins(outputs[0]), decode_bins(outputs[1])])

    def time_per_frame_ms(self, face: np.ndarray, repeats: int = 30) -> float:
        """Mediana del tiempo de una inferencia de un cuadro (como en vivo)."""
        self._run([face])  # calentamiento
        times: List[float] = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            self._run([face])
            times.append((time.perf_counter() - t0) * 1000.0)
        return float(np.median(times))
