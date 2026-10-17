"""Sesiones de mirada: grabación, carga, síntesis y evaluación sin cámara (lógica pura, solo numpy).

- SessionRecorder: durante `main.py --record-session` guarda por cuadro los rasgos crudos, el
  timestamp, el punto objetivo activo, la fase y la geometría de pantalla en
  `sessions/session_<fecha>.npz` (carpeta ignorada por git).
- synthesize_session: recorre la CalibrationSession real con un usuario simulado (ruido del iris,
  picos, parpadeos, movimiento de cabeza). Sirve para tests y para comparar variantes cuando no hay
  una sesión real; sus números dependen del modelo simulado y NO sustituyen a datos reales.
- evaluate / format_table: comparan variantes de mapeo y filtro con la sesión grabada
  (scripts/replay_session.py). Métricas: desvío en los puntos de validación (contra puntos nuevos,
  no leave-one-out), jitter del cursor durante la fijación y latencia aproximada del filtro.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from app_paths import data_dir

from calibration import (
    AGGREGATES,
    DEFAULT_FIT,
    FEATURE_SETS,
    MODELS,
    PHASE_CAPTURING,
    PHASE_SETTLING,
    STAGE_VALIDATION,
    HEAD_MOTION_STEPS,
    CalibrationSession,
    GazeEarModel,
    GazeMapper,
    filter_point_samples,
    iris_height,
    local_open_mask,
    validation_points,
)
from config_manager import DEFAULT_CONFIG
from eye_tracker import (
    EYE_BLENDSHAPES,
    FEATURE_DIM,
    FEATURE_NAMES,
    NUM_LANDMARKS,
    BlinkDetector,
    BlinkEvent,
    TrackingFrame,
    gaze_perp,
)

SESSION_VERSION = 1
SESSIONS_DIRNAME = "sessions"
DEFAULT_DISTANCE_MM = 600.0  # distancia de visión supuesta para convertir px a grados

STAGE_CODES = {"calibration": 0, "validation": 1, "head": 2}
PHASE_CODES = {PHASE_SETTLING: 0, PHASE_CAPTURING: 1}  # cualquier otra fase -> 2


def default_sessions_dir() -> str:
    return os.path.join(data_dir(), SESSIONS_DIRNAME)


def make_session_meta(
    geometry: Dict[str, int],
    session: CalibrationSession,
    dpr: float = 1.0,
    physical_mm: Optional[Tuple[float, float]] = None,
    camera_index: int = 0,
    config: Optional[dict] = None,
    extra: Optional[dict] = None,
) -> dict:
    """Metadatos de la sesión: geometría física, puntos, ritmo de la calibración y configuración."""
    meta = {
        "version": SESSION_VERSION,
        "created": datetime.now().isoformat(timespec="seconds"),
        "screen": {k: int(geometry[k]) for k in ("x", "y", "width", "height")},
        "dpr": float(dpr),
        "physical_mm": [float(physical_mm[0]), float(physical_mm[1])] if physical_mm else None,
        "camera_index": int(camera_index),
        "feature_names": list(FEATURE_NAMES),
        "calibration_points": [list(map(float, p)) for p in session.points],  # en orden de presentación
        "calibration_order": list(getattr(session, "order", range(len(session.points)))),
        "validation_points": [list(map(float, p)) for p in session.validation],
        "settle_s": session.settle_s,
        "capture_s": session.capture_s,
        # fase de movimiento de cabeza (stage 2, point = paso): [clave, segundos de movimiento]
        "head_motion_steps": [[str(k), float(sec)] for k, _text, sec in getattr(session, "head_motion", [])],
        "head_settle_s": float(getattr(session, "head_settle_s", 0.0)),
        "config": config or {},
    }
    if extra:
        meta.update(extra)
    return meta


# ------------------------------------------------------------------ grabación
class SessionRecorder:
    """Acumula cuadros con el estado de la sesión de calibración y los guarda en .npz."""

    def __init__(self, meta: Optional[dict] = None, keep_landmarks: bool = True):
        self.meta = dict(meta or {})
        self.keep_landmarks = keep_landmarks
        self._t: List[float] = []
        self._features: List[Optional[np.ndarray]] = []
        self._face: List[bool] = []
        self._closed: List[bool] = []
        self._blink_ms: List[float] = []
        self._ear: List[float] = []
        self._stage: List[int] = []
        self._point: List[int] = []
        self._phase: List[int] = []
        self._target: List[Tuple[float, float]] = []
        self._landmarks: List[Optional[np.ndarray]] = []
        self._matrix: List[Optional[np.ndarray]] = []
        self._fps: List[float] = []
        self._blendshapes: List[Optional[np.ndarray]] = []
        self._camera_info = ""
        self._image_size: Optional[Tuple[int, int]] = None

    def __len__(self) -> int:
        return len(self._t)

    def add(self, frame, session: CalibrationSession):
        """Registra un cuadro con el punto que se mostraba al llegar (antes de procesarlo)."""
        feats = getattr(frame, "features", None)
        self._t.append(float(frame.timestamp))
        self._features.append(None if feats is None else np.asarray(feats, dtype=float).copy())
        self._face.append(bool(frame.face_found))
        self._closed.append(bool(frame.eyes_closed))
        event = getattr(frame, "blink_event", None)
        self._blink_ms.append(float(event.duration_ms) if event is not None else 0.0)
        self._ear.append(float(getattr(frame, "ear_mean", 0.0)))
        self._stage.append(STAGE_CODES.get(session.stage, 0))
        self._point.append(int(session.active_index))
        self._phase.append(PHASE_CODES.get(session.phase, 2) if not session.finished else 2)
        tx, ty = session.current_point
        self._target.append((float(tx), float(ty)))
        if self.keep_landmarks:
            lm = getattr(frame, "landmarks", None)
            self._landmarks.append(None if lm is None else np.asarray(lm, dtype=np.float32)[:, :2])
        mx = getattr(frame, "matrix", None)
        self._matrix.append(None if mx is None else np.asarray(mx, dtype=np.float32))
        self._fps.append(float(getattr(frame, "fps", 0.0) or 0.0))
        shapes = getattr(frame, "eye_blendshapes", None)
        self._blendshapes.append(None if shapes is None else np.asarray(shapes, dtype=np.float32))
        self._camera_info = getattr(frame, "camera_info", "") or self._camera_info
        image = getattr(frame, "image", None)
        if image is not None:
            self._image_size = (int(image.shape[1]), int(image.shape[0]))

    def camera_meta(self) -> dict:
        """Lo que entregó la cámara: resolución real, formato y fps medidos con timestamps."""
        fps = [f for f in self._fps if f > 0]
        return {
            "info": self._camera_info,
            "image_size": list(self._image_size) if self._image_size else None,
            "fps_median": float(np.median(fps)) if fps else None,
        }

    def arrays(self) -> Dict[str, np.ndarray]:
        n = len(self._t)
        width = max([len(f) for f in self._features if f is not None] + [FEATURE_DIM])
        feats = np.full((n, width), np.nan)
        for i, f in enumerate(self._features):
            if f is not None:
                feats[i, :] = 0.0  # rasgos que una versión vieja no tenía (p. ej. roll) quedan en 0
                feats[i, : len(f)] = f
        out = {
            "t": np.asarray(self._t, dtype=float),
            "features": feats,
            "face": np.asarray(self._face, dtype=bool),
            "closed": np.asarray(self._closed, dtype=bool),
            "blink_ms": np.asarray(self._blink_ms, dtype=np.float32),
            "ear": np.asarray(self._ear, dtype=np.float32),
            "stage": np.asarray(self._stage, dtype=np.int8),
            "point": np.asarray(self._point, dtype=np.int16),
            "phase": np.asarray(self._phase, dtype=np.int8),
            "target": np.asarray(self._target, dtype=float).reshape(n, 2),
            "fps": np.asarray(self._fps, dtype=np.float32),
        }
        matrix = np.full((n, 4, 4), np.nan, dtype=np.float32)
        for i, m in enumerate(self._matrix):
            if m is not None and m.shape[0] >= 4 and m.shape[1] >= 4:
                matrix[i] = m[:4, :4]
        out["matrix"] = matrix
        shapes = np.full((n, len(EYE_BLENDSHAPES)), np.nan, dtype=np.float32)
        for i, b in enumerate(self._blendshapes):
            if b is not None and b.shape == (len(EYE_BLENDSHAPES),):
                shapes[i] = b
        out["blendshapes"] = shapes  # columnas = eye_tracker.EYE_BLENDSHAPES (meta["blendshape_names"])
        if self.keep_landmarks:
            lms = np.full((n, NUM_LANDMARKS, 2), np.nan, dtype=np.float32)
            for i, lm in enumerate(self._landmarks):
                if lm is not None and lm.shape[0] >= NUM_LANDMARKS:
                    lms[i] = lm[:NUM_LANDMARKS]
            out["landmarks"] = lms
        return out

    def to_session(self) -> "RecordedSession":
        return RecordedSession.from_arrays(self.arrays(), self.meta)

    def save(self, directory: Optional[str] = None, now: Optional[datetime] = None) -> str:
        """Guarda en `directory/session_<fecha>.npz` (sin sobrescribir) y devuelve la ruta."""
        directory = directory or default_sessions_dir()
        os.makedirs(directory, exist_ok=True)
        stamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
        path = os.path.join(directory, f"session_{stamp}.npz")
        k = 2
        while os.path.exists(path):
            path = os.path.join(directory, f"session_{stamp}_{k}.npz")
            k += 1
        arrays = self.arrays()
        meta = dict(self.meta, n_frames=len(self), camera=self.camera_meta(), blendshape_names=list(EYE_BLENDSHAPES))
        tmp = path + ".tmp.npz"
        np.savez_compressed(tmp, meta=np.array(json.dumps(meta, ensure_ascii=False)), **arrays)
        os.replace(tmp, path)
        return path


# ----------------------------------------------------------------- sesión cargada
REQUIRED_ARRAYS = ("t", "features", "face", "closed", "stage", "point", "phase", "target")


@dataclass
class RecordedSession:
    meta: dict
    t: np.ndarray
    features: np.ndarray
    face: np.ndarray
    closed: np.ndarray
    stage: np.ndarray
    point: np.ndarray
    phase: np.ndarray
    target: np.ndarray
    blink_ms: Optional[np.ndarray] = None
    ear: Optional[np.ndarray] = None
    landmarks: Optional[np.ndarray] = None
    matrix: Optional[np.ndarray] = None
    blendshapes: Optional[np.ndarray] = None  # (n, len(EYE_BLENDSHAPES)); no existe en sesiones anteriores

    @classmethod
    def from_arrays(cls, arrays: Dict[str, np.ndarray], meta: dict) -> "RecordedSession":
        missing = [k for k in REQUIRED_ARRAYS if k not in arrays]
        if missing:
            raise ValueError(f"Sesión inválida: faltan {', '.join(missing)}")
        n = len(arrays["t"])
        for k in REQUIRED_ARRAYS:
            if len(arrays[k]) != n:
                raise ValueError(f"Sesión inválida: '{k}' no tiene {n} cuadros")
        return cls(
            meta=dict(meta),
            **{k: np.asarray(arrays[k]) for k in REQUIRED_ARRAYS},
            blink_ms=arrays.get("blink_ms"),
            ear=arrays.get("ear"),
            landmarks=arrays.get("landmarks"),
            matrix=arrays.get("matrix"),
            blendshapes=arrays.get("blendshapes"),
        )

    def __len__(self) -> int:
        return len(self.t)

    @property
    def width(self) -> int:
        return int(self.meta["screen"]["width"])

    @property
    def height(self) -> int:
        return int(self.meta["screen"]["height"])

    def valid_mask(self) -> np.ndarray:
        return self.face & ~self.closed & np.isfinite(self.features).all(axis=1)

    def capture_mask(self, stage: str) -> np.ndarray:
        """Cuadros válidos en fase de captura de la etapa indicada (calibration/validation)."""
        return self.valid_mask() & (self.phase == 1) & (self.stage == STAGE_CODES[stage])

    def samples(self, stage: str = "calibration") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(rasgos, objetivos, grupo=punto) de los cuadros de captura válidos."""
        m = self.capture_mask(stage)
        return self.features[m], self.target[m], self.point[m].astype(int)

    @property
    def has_validation(self) -> bool:
        return bool(self.capture_mask("validation").any())

    def frame(self, i: int) -> TrackingFrame:
        """Reconstruye el TrackingFrame i (sin imagen ni landmarks)."""
        face = bool(self.face[i]) and bool(np.isfinite(self.features[i]).all())
        blink = None
        if self.blink_ms is not None and self.blink_ms[i] > 0:
            d = float(self.blink_ms[i])
            blink = BlinkEvent(start_ts=float(self.t[i]) - d / 1000.0, end_ts=float(self.t[i]), duration_ms=d)
        return TrackingFrame(
            timestamp=float(self.t[i]),
            face_found=face,
            features=self.features[i].copy() if face else None,
            eyes_closed=bool(self.closed[i]) and face,
            blink_event=blink,
        )


def load_session(path: str) -> RecordedSession:
    """Carga una sesión .npz. Lanza ValueError con el motivo si no es válida."""
    try:
        with np.load(path, allow_pickle=False) as data:
            arrays = {k: data[k] for k in data.files}
    except (OSError, ValueError) as e:
        raise ValueError(f"No se pudo leer la sesión {path}: {e}") from e
    if "meta" not in arrays:
        raise ValueError("Sesión inválida: faltan los metadatos")
    meta = json.loads(str(arrays.pop("meta")))
    if meta.get("version") != SESSION_VERSION:
        raise ValueError(f"Versión de sesión no compatible: {meta.get('version')}")
    return RecordedSession.from_arrays(arrays, meta)


# ------------------------------------------------------------------- síntesis
class SyntheticUser:
    """Usuario simulado con un modelo geométrico simple de ojos y cabeza (solo para tests y comparativas).

    Coordenadas en mm centradas en la pantalla (x a la derecha, y hacia abajo, z hacia el usuario).
    La cabeza deriva lentamente (Ornstein-Uhlenbeck) y se desplaza al empezar la validación; el iris
    tiene ruido gaussiano, picos aislados y se distorsiona durante y justo después de un parpadeo.
    """

    def __init__(
        self,
        rng: np.random.Generator,
        width: int,
        height: int,
        physical_mm: Tuple[float, float] = (597.0, 336.0),
        distance_mm: float = DEFAULT_DISTANCE_MM,
        iris_noise: float = 0.008,
        spike_rate: float = 0.03,
        spike_size: float = 0.06,
        fixation_px: float = 8.0,
        reaction_s: float = 0.25,
        head_sigma_mm: float = 8.0,
        head_sigma_rad: float = 0.025,
        head_tau_s: float = 3.0,
        validation_shift_mm: Tuple[float, float, float] = (12.0, -6.0, 25.0),
        validation_shift_rad: Tuple[float, float, float] = (0.03, 0.02, 0.01),
        blink_period_s: float = 4.0,
        pose_noise_rad: float = 0.008,
    ):
        self.rng = rng
        self.width, self.height = width, height
        self.phys = physical_mm
        self.distance = distance_mm
        self.iris_noise = iris_noise
        self.spike_rate = spike_rate
        self.spike_size = spike_size
        self.fixation_px = fixation_px
        self.reaction_s = reaction_s
        self.head_sigma_mm = head_sigma_mm
        self.head_sigma_rad = head_sigma_rad
        self.head_tau = head_tau_s
        self.val_shift_mm = np.asarray(validation_shift_mm, dtype=float)
        self.val_shift_rad = np.asarray(validation_shift_rad, dtype=float)
        self.blink_period = blink_period_s
        self.pose_noise = pose_noise_rad
        self.pos = np.array([0.0, 0.0, distance_mm])
        self.rot = np.array([0.0, -0.05, 0.0])  # yaw, pitch, roll
        self._t: Optional[float] = None
        self._target: Optional[Tuple[float, float]] = None
        self._pending: Optional[Tuple[float, float]] = None
        self._switch_at: Optional[float] = None
        self._fix = np.zeros(2)
        self._next_blink = blink_period_s * (0.5 + rng.random())
        self._blink_end = -1.0

    def _ou(self, x, mu, sigma, dt):
        k = min(1.0, dt / self.head_tau)
        return x + (mu - x) * k + sigma * math.sqrt(2.0 * k) * self.rng.normal(size=np.shape(x))

    # Movimiento guionado de la fase de cabeza: amplitud de cada paso (2 ciclos por paso)
    HEAD_MOTION_AMPLITUDE = {"yaw": 0.25, "pitch": 0.15, "depth": 100.0}  # rad, rad, mm

    def frame(
        self, t: float, target: Tuple[float, float], stage: str = "calibration", motion: Optional[Tuple[str, float]] = None
    ) -> TrackingFrame:
        """`motion` = (clave del paso de HEAD_MOTION_STEPS, avance 0..1) mientras se mueve la cabeza."""
        dt = 1.0 / 30.0 if self._t is None else max(1e-3, t - self._t)
        self._t = t
        shift = stage in (STAGE_VALIDATION, "head")
        mu_pos = np.array([0.0, 0.0, self.distance]) + (self.val_shift_mm if shift else 0.0)
        mu_rot = np.array([0.0, -0.05, 0.0]) + (self.val_shift_rad if shift else 0.0)
        self.pos = self._ou(self.pos, mu_pos, self.head_sigma_mm, dt)
        self.rot = self._ou(self.rot, mu_rot, self.head_sigma_rad, dt)
        pos, rot = self.pos, self.rot
        if motion is not None and motion[0] in self.HEAD_MOTION_AMPLITUDE:
            wave = self.HEAD_MOTION_AMPLITUDE[motion[0]] * math.sin(4.0 * math.pi * motion[1])
            pos, rot = pos.copy(), rot.copy()
            if motion[0] == "yaw":
                rot[0] += wave
            elif motion[0] == "pitch":
                rot[1] += wave
            else:
                pos[2] += wave

        # la mirada cambia de objetivo con una latencia de reacción
        target = (float(target[0]), float(target[1]))
        if self._target is None:
            self._target = target
        if target != self._pending:
            self._pending = target
            self._switch_at = t + self.reaction_s
        if self._switch_at is not None and t >= self._switch_at:
            self._target = self._pending
            self._switch_at = None
        k = min(1.0, dt / 0.5)
        self._fix = self._fix * (1 - k) + self.fixation_px * math.sqrt(2 * k) * self.rng.normal(size=2)
        gx, gy = self._target[0] + self._fix[0], self._target[1] + self._fix[1]

        # parpadeos
        if t >= self._next_blink:
            self._blink_end = t + self.rng.uniform(0.12, 0.2)
            self._next_blink = t + self.blink_period * (0.6 + 0.8 * self.rng.random())
        closed = t < self._blink_end
        reopening = (not closed) and t < self._blink_end + 0.1

        feats = self._features(gx, gy, closed, reopening, pos, rot)
        return TrackingFrame(
            timestamp=t,
            face_found=True,
            ear_left=0.08 if closed else 0.3,
            ear_right=0.08 if closed else 0.3,
            ear_mean=0.08 if closed else 0.3,
            features=feats[:FEATURE_DIM],
            face_center=(float(feats[4]), float(feats[5])),
            interocular=float(feats[6]),
            eyes_closed=closed,
        )

    def _features(self, gx, gy, closed, reopening, pos=None, rot=None) -> np.ndarray:
        """Vector completo (10 rasgos, incluye roll); se recorta a FEATURE_DIM."""
        rng = self.rng
        hx, hy, hz = self.pos if pos is None else pos
        yaw, pitch, roll = self.rot if rot is None else rot
        tx = (gx / self.width - 0.5) * self.phys[0]
        ty = (gy / self.height - 0.5) * self.phys[1]
        ex = math.atan2(tx - hx, hz) - yaw
        ey = math.atan2(ty - hy, hz) - pitch
        along = 0.40 * math.sin(ex) + 0.10 * yaw  # la perspectiva acopla el giro de la cabeza
        perp = 0.28 * math.sin(ey) + 0.06 * ex * ex + 0.08 * pitch
        eyes = []
        for vergence in (0.01, -0.01):
            a = along + vergence + rng.normal(0, self.iris_noise)
            p = perp + rng.normal(0, self.iris_noise)
            if rng.random() < self.spike_rate:
                a += rng.normal(0, self.spike_size)
                p += rng.normal(0, self.spike_size)
            if closed:
                a = 0.3 * a + rng.normal(0, 0.03)
                p += 0.15
            elif reopening:
                p += 0.05
            eyes.append((a, p))
        nose_x = 0.5 + 0.9 * hx / hz + 0.35 * math.sin(yaw)
        nose_y = 0.5 + 0.9 * hy / hz + 0.35 * math.sin(pitch)
        interocular = 100.0 * 0.866 / hz
        pn = self.pose_noise
        return np.array(
            [
                eyes[0][0], eyes[0][1], eyes[1][0], eyes[1][1],
                nose_x + rng.normal(0, 0.001), nose_y + rng.normal(0, 0.001),
                interocular + rng.normal(0, 0.0005),
                yaw + rng.normal(0, pn), pitch + rng.normal(0, pn), roll + rng.normal(0, pn),
            ],
            dtype=float,
        )


def synthesize_session(
    seed: int = 0,
    width: int = 2560,
    height: int = 1440,
    physical_mm: Tuple[float, float] = (597.0, 336.0),
    fps: float = 30.0,
    validation: bool = True,
    user_kwargs: Optional[dict] = None,
    session_kwargs: Optional[dict] = None,
    max_seconds: float = 180.0,
    order: str = "balanced",
    head_motion: bool = False,
) -> RecordedSession:
    """Sesión sintética que reproduce el flujo real: la CalibrationSession decide puntos, orden y fases.
    `head_motion` añade la fase de movimiento de cabeza de --record-session (movimiento guionado)."""
    rng = np.random.default_rng(seed)
    user = SyntheticUser(rng, width, height, physical_mm, **(user_kwargs or {}))
    session = CalibrationSession(
        width,
        height,
        validation=validation_points(width, height) if validation else None,
        order=order,
        rng=np.random.default_rng(seed + 1000),
        head_motion=HEAD_MOTION_STEPS if head_motion else None,
        **(session_kwargs or {}),
    )
    geometry = {"x": 0, "y": 0, "width": width, "height": height}
    rec = SessionRecorder(
        make_session_meta(geometry, session, physical_mm=physical_mm, extra={"synthetic": True, "seed": seed}),
        keep_landmarks=False,
    )
    t = 0.0
    while not session.finished and t < max_seconds:
        dt = 1.0 / fps + rng.normal(0, 0.003)
        if rng.random() < 0.01:
            dt += 1.0 / fps  # cuadro perdido
        t += max(0.005, dt)
        motion = None
        if session.in_head and session.phase == PHASE_CAPTURING:
            motion = (session.head_step[0], session.phase_progress(t))
        frame = user.frame(t, session.current_point, session.stage, motion)
        rec.add(frame, session)
        valid = frame.face_found and not frame.eyes_closed and frame.features is not None
        session.update(t, frame.features, valid)
    return rec.to_session()


# ------------------------------------------------------------------- evaluación
def fit_baseline(x, y, groups):
    """Mapeo anterior a esta mejora: compacto, cuadrático completo, ridge alpha=0.02, todas las muestras."""
    mapper = GazeMapper("compact", alpha=0.02, clamp=False)
    mapper.fit(x, y, groups)
    return mapper


def make_fit(options: Optional[dict] = None) -> Callable:
    """Ajuste con las opciones de GazeMapper.from_fit_options (DEFAULT_FIT por defecto)."""

    def fit(x, y, groups):
        mapper = GazeMapper.from_fit_options(options)
        mapper.fit(x, y, groups)
        return mapper

    return fit


def post_blink_mask(t: np.ndarray, closed: np.ndarray, window_s: float = 0.15) -> np.ndarray:
    """True en los cuadros con ojos cerrados o a menos de `window_s` de un cierre (como CalibrationSession)."""
    out = np.zeros(len(t), dtype=bool)
    last = None
    for i in range(len(t)):
        if closed[i]:
            last = t[i]
        out[i] = last is not None and t[i] - last < window_s
    return out


@dataclass
class Variant:
    """Una combinación de mapeo (cómo se ajusta) y filtro (claves de interaction y opciones del controlador).

    `robust`: las muestras de calibración pasan por el mismo filtrado que CalibrationSession(robust=True)
    (tras parpadeo, pose fuera de rango y atípicos del iris). El mapeo anterior no lo tenía.
    """

    name: str
    fit: Callable[[np.ndarray, np.ndarray, np.ndarray], object] = fit_baseline
    interaction: Dict = field(default_factory=dict)
    controller_kwargs: Dict = field(default_factory=dict)
    robust: bool = False


def robust_samples(session: "RecordedSession", stage: str = "calibration"):
    """Muestras de captura con el filtrado robusto de la calibración (por punto).

    Como la aplicación: con EAR grabado, los ojos cerrados se deciden con el EAR del propio punto
    (local_open_mask); sin él, con los cierres del BlinkDetector grabados y los 150 ms siguientes.
    """
    has_ear = session.ear is not None
    m = session.capture_mask(stage)
    if not has_ear:
        m &= ~post_blink_mask(session.t, session.closed)
    x, y, g, t = session.features[m], session.target[m], session.point[m].astype(int), session.t[m]
    ear = session.ear[m] if has_ear else None
    keep = np.zeros(len(x), dtype=bool)
    for p in np.unique(g):
        idx = np.flatnonzero(g == p)
        k = local_open_mask(t[idx], ear[idx]) if has_ear else np.ones(len(idx), dtype=bool)
        if k.any():
            k[k] = filter_point_samples(x[idx][k])
        keep[idx] = k
    return x[keep], y[keep], g[keep]


def ear_points(session: "RecordedSession") -> Optional[np.ndarray]:
    """[altura del iris, EAR] medianos por punto (calibración y validación), como CalibrationSession."""
    if session.ear is None:
        return None
    rows = []
    for stage in ("calibration", "validation"):
        m = session.capture_mask(stage)
        t, x, ear, g = session.t[m], session.features[m], session.ear[m], session.point[m]
        for p in np.unique(g):
            idx = np.flatnonzero(g == p)
            k = local_open_mask(t[idx], ear[idx])
            if k.any():
                k[k] = filter_point_samples(x[idx][k])
            if k.sum() >= 3:
                rows.append((float(np.median(iris_height(x[idx][k]))), float(np.median(ear[idx][k]))))
    return np.asarray(rows, dtype=float) if rows else None


def blink_report(session: "RecordedSession", click_range_ms: Tuple[float, float] = (150.0, 2000.0)) -> Dict[str, object]:
    """Cierres por fila de la pantalla (arriba / medio / abajo) con el detector sin y con GazeEarModel.

    "grabado" son los cierres que marcó la aplicación al grabar. Con blendshapes grabados se añaden sus
    medianas por fila con los ojos abiertos y en los cierres detectados (para decidir si sirven).
    """
    if session.ear is None:
        raise ValueError("La sesión no tiene EAR grabado")
    row = np.digitize(session.target[:, 1] / max(session.height, 1), [0.35, 0.65])
    face = session.face.astype(bool)
    model = GazeEarModel.fit(ear_points(session))
    out: Dict[str, object] = {"model": model, "rows": ("arriba", "medio", "abajo")}

    def by_row(closed):
        return [float(closed[face & (row == r)].mean() * 100) if (face & (row == r)).any() else None for r in range(3)]

    out["recorded"] = {"closed_pct": by_row(session.closed.astype(bool) & face)}
    variants = [("sin modelo", None)] + ([("con modelo", model)] if model is not None else [])
    for label, gm in variants:
        det = BlinkDetector(gaze_model=gm)
        closed = np.zeros(len(session), dtype=bool)
        events = []
        for i in range(len(session)):
            ev = det.update(float(session.t[i]), float(session.ear[i]), bool(face[i]), gaze_perp(session.features[i]))
            closed[i] = det.is_closed
            if ev is not None:
                events.append(ev)
        lo, hi = click_range_ms
        clicks = [e for e in events if lo <= e.duration_ms <= hi]
        out[label] = {
            "closed_pct": by_row(closed),
            "events": len(events),
            "click_range": len(clicks),
            "suspect": sum(1 for e in clicks if e.gaze_suspect),
            "closed_mask": closed,
        }
    shapes = session.blendshapes
    if shapes is not None and np.isfinite(shapes).any():
        names = list(session.meta.get("blendshape_names") or EYE_BLENDSHAPES)
        blink = shapes[:, [names.index("eyeBlinkLeft"), names.index("eyeBlinkRight")]].mean(axis=1)
        down = shapes[:, [names.index("eyeLookDownLeft"), names.index("eyeLookDownRight")]].mean(axis=1)
        closed = out[variants[-1][0]]["closed_mask"]
        table = {}
        for r in range(3):
            for state, mask in (("abiertos", face & (row == r) & ~closed), ("cerrados", face & (row == r) & closed)):
                if mask.sum() >= 3:
                    table[(r, state)] = (float(np.nanmedian(blink[mask])), float(np.nanmedian(down[mask])), int(mask.sum()))
        out["blendshapes"] = table
    return out


def format_blink_report(rep: Dict[str, object]) -> str:
    def pct(values):
        return " / ".join("-" if v is None else f"{v:.0f} %" for v in values)

    lines = ["Parpadeos: cuadros con ojos 'cerrados' por fila (arriba / medio / abajo)"]
    lines.append(f"  grabado por la aplicación: {pct(rep['recorded']['closed_pct'])}")
    for label in ("sin modelo", "con modelo"):
        if label in rep:
            r = rep[label]
            lines.append(f"  detector {label}: {pct(r['closed_pct'])}; {r['events']} parpadeos, {r['click_range']} en el "
                         f"rango de clic" + (f" ({r['suspect']} bloqueados por el seguro)" if label == "con modelo" else ""))
    model = rep.get("model")
    if model is not None:
        lines.append(f"  modelo EAR-mirada: EAR = {model.c0:.3f} {model.c1:+.3f} x altura del iris; en la fila más baja el"
                     f" ojo se abre un {model.factor(model.low_end) * 100:.0f} % de lo normal; seguro de clic "
                     + ("activo" if model.guard_active else "inactivo"))
    else:
        lines.append("  sin modelo EAR-mirada (faltan puntos completos)")
    table = rep.get("blendshapes")
    if table:
        lines.append("  blendshapes (medianas eyeBlink / eyeLookDown, cuadros):")
        for (r, state), (b, d, n) in sorted(table.items()):
            lines.append(f"    {rep['rows'][r]:6s} ojos {state}: {b:.2f} / {d:.2f} ({n})")
    elif "blendshapes" not in rep:
        lines.append("  (la sesión no tiene blendshapes: se graban desde esta versión)")
    return "\n".join(lines)


class _NullMouse:
    def move(self, x, y):
        pass

    def down(self):
        pass

    def up(self):
        pass

    def click(self):
        pass


class _IdentityMapper:
    def predict_one(self, features):
        return float(features[0]), float(features[1])


def _interaction(overrides: Dict) -> Dict:
    cfg = dict(DEFAULT_CONFIG["interaction"])
    cfg.update(overrides)
    cfg["hold_enabled"] = False  # la simulación solo mide posición
    return cfg


def simulate_cursor(session: RecordedSession, mapper, interaction: Optional[Dict] = None, controller_kwargs=None) -> np.ndarray:
    """Pasa la sesión completa por el CursorController real y devuelve la posición mostrada por cuadro."""
    from cursor_control import CursorController

    cfg = _interaction(interaction or {})
    ctl = CursorController(
        mapper, _NullMouse(), session.width, session.height, config_getter=lambda: cfg, **(controller_kwargs or {})
    )
    pos = np.full((len(session), 2), np.nan)
    for i in range(len(session)):
        snap = ctl.update(session.frame(i))
        if snap.positioned:
            pos[i] = (snap.x, snap.y)
    return pos


def fixation_metrics(session: RecordedSession, positions: np.ndarray, stage: str = "validation") -> Dict[str, float]:
    """Por punto de la etapa: desvío = |media del cursor - objetivo| y jitter = desviación típica 2D."""
    biases, jitters = [], []
    sel = (session.phase == 1) & (session.stage == STAGE_CODES[stage]) & np.isfinite(positions).all(axis=1)
    for g in np.unique(session.point[sel]):
        m = sel & (session.point == g)
        if m.sum() < 5:
            continue
        p = positions[m]
        mean = p.mean(axis=0)
        biases.append(float(np.linalg.norm(mean - session.target[m][0])))
        jitters.append(float(np.sqrt(np.mean(np.sum((p - mean) ** 2, axis=1)))))
    nan = float("nan")
    return {
        "bias_px": float(np.mean(biases)) if biases else nan,
        "jitter_px": float(np.mean(jitters)) if jitters else nan,
        "points": len(biases),
    }


def filter_latency_ms(interaction: Optional[Dict] = None, controller_kwargs=None, fps: float = 30.0, step_px: float = 400.0) -> float:
    """Tiempo hasta recorrer el 90 % de un salto de `step_px` (entrada sin ruido a `fps`)."""
    from cursor_control import CursorController

    cfg = _interaction(interaction or {})
    ctl = CursorController(_IdentityMapper(), _NullMouse(), 4000, 4000, config_getter=lambda: cfg, **(controller_kwargs or {}))

    def feats(x):
        f = np.zeros(FEATURE_DIM)
        f[0], f[1] = x, 1000.0
        return f

    t = 0.0
    for _ in range(int(fps)):
        ctl.update(TrackingFrame(timestamp=t, face_found=True, features=feats(1000.0)))
        t += 1.0 / fps
    t_step = t
    goal = 1000.0 + 0.9 * step_px
    for _ in range(int(fps * 5)):
        snap = ctl.update(TrackingFrame(timestamp=t, face_found=True, features=feats(1000.0 + step_px)))
        if snap.x >= goal:
            return (t - t_step) * 1000.0
        t += 1.0 / fps
    return float("inf")


def px_to_deg(px: float, session: RecordedSession, distance_mm: float = DEFAULT_DISTANCE_MM) -> float:
    """Ángulo visual aproximado de `px` píxeles físicos (necesita physical_mm en los metadatos)."""
    phys = session.meta.get("physical_mm")
    if not phys or not phys[0] or not math.isfinite(px):
        return float("nan")
    mm = px * float(phys[0]) / session.width
    return math.degrees(math.atan2(mm, distance_mm))


def evaluate(
    session: RecordedSession, variants: Sequence[Variant], distance_mm: float = DEFAULT_DISTANCE_MM
) -> List[Dict[str, object]]:
    """Evalúa cada variante: ajusta con la etapa de calibración y mide en la de validación."""
    plain = session.samples("calibration")
    if len(plain[0]) < 3:
        raise ValueError("La sesión no tiene muestras de calibración suficientes")
    robust = robust_samples(session, "calibration")
    stage = "validation" if session.has_validation else "calibration"
    rows = []
    for v in variants:
        x, y, g = robust if v.robust else plain
        mapper = v.fit(x, y, g)
        pos = simulate_cursor(session, mapper, v.interaction, v.controller_kwargs)
        fm = fixation_metrics(session, pos, stage)
        metrics = getattr(mapper, "metrics", {}) or {}
        rows.append(
            {
                "name": v.name,
                "stage": stage,
                "bias_px": fm["bias_px"],
                "bias_deg": px_to_deg(fm["bias_px"], session, distance_mm),
                "jitter_px": fm["jitter_px"],
                "latency_ms": filter_latency_ms(v.interaction, v.controller_kwargs),
                "loo_px": float(metrics.get("cv_rmse_px", float("nan"))),
                "points": fm["points"],
            }
        )
    return rows


OLD_FILTER = {"feature_median_window": 1, "deadzone_px": 0}  # el pipeline anterior no tenía mediana ni zona muerta
EXPONENTIAL = {"filter_kind": "exponential"}


def filter_variants(fit=fit_baseline, prefix: str = "", robust: bool = False) -> List[Variant]:
    """Mismo mapeo con los distintos filtros (parte B)."""
    return [
        Variant(f"{prefix}exponencial (anterior)", fit, dict(OLD_FILTER), dict(EXPONENTIAL), robust),
        Variant(f"{prefix}sin suavizado (referencia)", fit, dict(OLD_FILTER, smoothing_percent=0), dict(EXPONENTIAL), robust),
        Variant(f"{prefix}One Euro", fit, dict(OLD_FILTER), {}, robust),
        Variant(f"{prefix}One Euro + mediana", fit, {"deadzone_px": 0}, {}, robust),
        Variant(f"{prefix}One Euro + mediana + zona muerta", fit, {}, {}, robust),
    ]


FIT_LABELS = {"eyes": "solo ojos", "compact": "ojos+yaw/pitch/nariz", "head": "cabeza completa", "full": "10 rasgos"}
MODEL_LABELS = {"linear": "lineal", "quad": "cuadr. completo", "quad_reduced": "cuadr. reducido"}


def fit_label(options: dict) -> str:
    o = dict(DEFAULT_FIT, **options)
    alpha = "a=CV" if o["alpha"] == "cv" else f"a={o['alpha']}"
    agg = "mediana" if o["aggregate"] == "median" else "muestras"
    return f"{FIT_LABELS[o['feature_set']]}, {MODEL_LABELS[o['model']]}, {alpha}, {agg}"


def before_after_variants() -> List[Variant]:
    """Antes (mapeo y filtro anteriores) frente a después (DEFAULT_FIT + filtrado robusto + filtro nuevo)."""
    return [
        Variant("ANTES: mapeo anterior + exponencial", fit_baseline, dict(OLD_FILTER), dict(EXPONENTIAL)),
        Variant("DESPUÉS: " + fit_label({}) + " + One Euro", make_fit(), {}, {}, robust=True),
    ]


def model_variants() -> List[Variant]:
    """Parte C: conjuntos de rasgos x modelos x agregación, todos con alpha por CV y el filtro nuevo."""
    out = []
    for fs in FEATURE_SETS:
        for model in MODELS:
            for agg in AGGREGATES:
                opts = {"feature_set": fs, "model": model, "alpha": "cv", "aggregate": agg}
                out.append(Variant(fit_label(opts), make_fit(opts), {}, {}, robust=True))
    opts = dict(DEFAULT_FIT, alpha=0.02)
    out.append(Variant(fit_label(opts) + " (alpha fijo)", make_fit(opts), {}, {}, robust=True))
    return out


def posture_report(session: "RecordedSession") -> Dict[str, object]:
    """Distancia a la cámara, corrección automática del sesgo y deriva de la cabeza con DEFAULT_FIT."""
    from calibration import IOD_MIN, PoseDriftMonitor

    valid = session.valid_mask()
    iod = float(np.median(session.features[valid, 6])) if valid.any() else float("nan")
    out: Dict[str, object] = {"interocular": iod, "too_far": bool(iod < IOD_MIN), "iod_min": IOD_MIN}
    x, y, g = robust_samples(session, "calibration")
    if len(x) < 3:
        return out
    mapper = GazeMapper.from_fit_options()
    if session.has_validation:
        mapper.fit(x, y, g, *robust_samples(session, "validation"))
        out["validation_px"] = mapper.metrics.get("validation_px")
        out["validation_corrected_px"] = mapper.metrics.get("validation_corrected_px")
        out["auto_offset"] = mapper.metrics.get("auto_offset")
        out["auto_offset_rejected"] = bool(mapper.metrics.get("auto_offset_rejected"))
        out["auto_offset_skipped"] = bool(mapper.metrics.get("auto_offset_skipped"))
    else:
        mapper.fit(x, y, g)
    # cuánto se movió la cabeza en la validación respecto a la calibración (la sesión no tiene fase de uso)
    from calibration import POSE_COLS, pad_features

    monitor = PoseDriftMonitor(np.median(pad_features(x)[:, list(POSE_COLS)], axis=0))
    first_val = np.flatnonzero(session.stage == STAGE_CODES["validation"])
    start = int(first_val[0]) if len(first_val) else len(session)
    max_dev = 0.0
    warned_at = None
    for i in range(start, len(session)):
        if valid[i] and session.stage[i] == STAGE_CODES["validation"]:  # sin la fase de movimiento de cabeza
            if monitor.update(float(session.t[i]), session.features[i]) and warned_at is None:
                warned_at = float(session.t[i] - session.t[start])
            dev = monitor.deviation()
            if dev is not None:
                max_dev = max(max_dev, float(dev.max()))
    out["pose_max_deviation"] = max_dev  # en múltiplos de la tolerancia
    out["drift_warning_after_s"] = warned_at
    return out


def format_posture_report(rep: Dict[str, object]) -> str:
    lines = [f"Distancia: separación de ojos en la imagen {rep['interocular']:.3f} (recomendado >= {rep['iod_min']:.2f})"
             + (" -> DEMASIADO LEJOS: el iris tiene menos píxeles y el cursor tiembla más" if rep["too_far"] else "")]
    if rep.get("validation_px") is not None:
        line = f"Ajuste por defecto: desvío en validación {rep['validation_px']:.0f} px sin corregir"
        if rep.get("validation_corrected_px") is not None:
            off = rep.get("auto_offset") or [0, 0]
            line += (f", {rep['validation_corrected_px']:.0f} px con la corrección automática (dejando un punto fuera;"
                     f" offset {off[0]:+.0f}, {off[1]:+.0f} px")
            if rep.get("auto_offset_rejected"):
                line += ", RECHAZADO por grande)"
            elif rep.get("auto_offset_skipped"):
                line += ", NO SE APLICA: no mejora el error sin corregir)"
            else:
                line += ")"
        lines.append(line)
    if "pose_max_deviation" in rep:
        warned = rep["drift_warning_after_s"]
        lines.append(
            f"Cabeza en la validación respecto a la calibración: desviación máxima {rep['pose_max_deviation']:.1f}x "
            "la tolerancia; "
            + (f"el aviso de deriva habría saltado a los {warned:.1f} s" if warned is not None else "sin aviso de deriva")
            + " (en uso real el aviso se mide desde la pose de la validación, que es la que corrige el offset)"
        )
    return "\n".join(lines)


TYPICAL_EYE_SPAN_MM = 93.0  # separación media entre las esquinas externas de los ojos (landmarks 33 y 263)
HEAD_POSE_COLS = (4, 5, 6, 7, 8)  # nose_x, nose_y, interocular, yaw, pitch


def head_motion_report(session: "RecordedSession", distance_mm: float = DEFAULT_DISTANCE_MM) -> Dict[str, object]:
    """Fase de movimiento de cabeza (stage 2): cuánto se mueve el cursor del ajuste por defecto (solo iris)
    mientras se mira el punto central moviendo la cabeza, frente a la pose de la calibración.

    Por paso: rango de la pose (p5-p95), dispersión del error y la pendiente del error frente al giro
    (px/rad) o al desplazamiento, para compararla con la geometría: girar la cabeza un ángulo a desplaza el
    punto mirado ~distancia x tan(a); en px, x los px/mm del monitor. Es la medida de la fase 1: todavía no
    hay compensación.
    """
    stage = session.stage == STAGE_CODES["head"]
    if not stage.any():
        raise ValueError("La sesión no tiene fase de movimiento de cabeza (se graba desde esta versión)")
    x, y, g = robust_samples(session, "calibration")
    if len(x) < 3:
        raise ValueError("No hay muestras de calibración suficientes")
    mapper = GazeMapper.from_fit_options()
    if session.has_validation:
        mapper.fit(x, y, g, *robust_samples(session, "validation"))
    else:
        mapper.fit(x, y, g)
    ref = np.median(x[:, list(HEAD_POSE_COLS)], axis=0)
    mm = session.meta.get("physical_mm")
    px_per_mm = session.width / float(mm[0]) if mm else None
    steps = [s[0] for s in session.meta.get("head_motion_steps") or []]
    center = np.array([session.width / 2.0, session.height / 2.0])
    out: Dict[str, object] = {
        "distance_mm": distance_mm,
        "px_per_mm": px_per_mm,
        "geometric_px_per_rad": distance_mm * px_per_mm if px_per_mm else None,
        "reference": ref,
        "steps": [],
    }
    for k in np.unique(session.point[stage]):
        m = stage & (session.point == k) & (session.phase == 1) & session.valid_mask()
        idx = np.flatnonzero(m)
        if len(idx) < 10:
            continue
        keep = local_open_mask(session.t[idx], session.ear[idx]) if session.ear is not None else np.ones(len(idx), bool)
        idx = idx[keep]
        if len(idx) < 10:
            continue
        feats = session.features[idx]
        err = mapper.predict(feats) - center
        d = feats[:, list(HEAD_POSE_COLS)] - ref
        iod = feats[:, 6]
        mm_per_unit = TYPICAL_EYE_SPAN_MM / np.maximum(iod, 1e-6)  # mm por unidad normalizada del cuadro

        def span(v):
            return float(np.percentile(v, 95) - np.percentile(v, 5))

        def slope(v, e):
            if np.ptp(v) < 1e-6:
                return None, None
            s = float(np.polyfit(v, e, 1)[0])
            r = float(np.corrcoef(v, e)[0, 1]) if np.std(e) > 0 else 0.0
            return s, r

        key = steps[int(k)] if int(k) < len(steps) else str(int(k))
        norm = np.linalg.norm(err, axis=1)
        row = {
            "key": key,
            "frames": int(len(idx)),
            "yaw_deg": span(np.degrees(d[:, 3])),
            "pitch_deg": span(np.degrees(d[:, 4])),
            "shift_x_mm": span(d[:, 0] * mm_per_unit),
            "shift_y_mm": span(d[:, 1] * mm_per_unit),
            "distance_ratio": (float(np.percentile(ref[2] / iod, 5)), float(np.percentile(ref[2] / iod, 95))),
            "bias_px": (float(np.median(err[:, 0])), float(np.median(err[:, 1]))),
            "error_median_px": float(np.median(norm)),
            "error_p90_px": float(np.percentile(norm, 90)),
        }
        row["slope_x_yaw"] = slope(d[:, 3], err[:, 0])
        row["slope_y_pitch"] = slope(d[:, 4], err[:, 1])
        row["slope_x_shift"] = slope(d[:, 0] * mm_per_unit, err[:, 0])  # px por mm de desplazamiento
        out["steps"].append(row)
    return out


def format_head_motion_report(rep: Dict[str, object]) -> str:
    names = {"still": "quieto", "yaw": "girar izq./der.", "pitch": "subir/bajar barbilla", "depth": "acercarse/alejarse"}
    lines = ["Movimiento de cabeza (mirando el centro; ajuste por defecto, solo iris, sin compensar la cabeza)"]
    geo = rep.get("geometric_px_per_rad")
    if geo:
        lines.append(f"  geometría a {rep['distance_mm'] / 10:.0f} cm: girar la cabeza 1 rad desplaza el punto mirado ~{geo:.0f} px"
                     f" ({geo * np.pi / 180:.0f} px/°); 1 mm de desplazamiento = {rep['px_per_mm']:.2f} px")

    def fmt_slope(s, unit):
        value, r = s
        return "-" if value is None else f"{value:+.0f} {unit} (r={r:+.2f})"

    for row in rep["steps"]:
        lo, hi = row["distance_ratio"]
        lines.append(
            f"  {names.get(row['key'], row['key'])}: {row['frames']} cuadros; rango giro {row['yaw_deg']:.0f}°, inclinación"
            f" {row['pitch_deg']:.0f}°, desplazamiento {row['shift_x_mm']:.0f}x{row['shift_y_mm']:.0f} mm, distancia"
            f" x{lo:.2f}-x{hi:.2f}"
        )
        bx, by = row["bias_px"]
        lines.append(
            f"      error del cursor: mediana {row['error_median_px']:.0f} px, p90 {row['error_p90_px']:.0f} px,"
            f" sesgo ({bx:+.0f}, {by:+.0f}) px"
        )
        if row["key"] == "yaw":
            lines.append(f"      pendiente error x / giro: {fmt_slope(row['slope_x_yaw'], 'px/rad')}")
        elif row["key"] == "pitch":
            lines.append(f"      pendiente error y / inclinación: {fmt_slope(row['slope_y_pitch'], 'px/rad')}")
        if row["key"] in ("yaw", "depth"):
            lines.append(f"      pendiente error x / desplazamiento: {fmt_slope(row['slope_x_shift'], 'px/mm')}")
    if not rep["steps"]:
        lines.append("  sin cuadros válidos en la fase de cabeza")
    return "\n".join(lines)


def default_variants() -> List[Variant]:
    """Variantes que compara scripts/replay_session.py (la primera es la referencia anterior)."""
    return before_after_variants() + filter_variants(make_fit(), "mapeo nuevo + ", robust=True)


def format_table(rows: Sequence[Dict[str, object]]) -> str:
    """Tabla de texto con las métricas de evaluate()."""
    stage = rows[0]["stage"] if rows else "validation"
    head = ("Variante", "Desvío px", "Desvío °", "Jitter px", "Latencia ms", "LOO px")
    width = max([len(head[0])] + [len(str(r["name"])) for r in rows])

    def num(v, d=1):
        return "-" if v is None or not math.isfinite(float(v)) else f"{float(v):.{d}f}"

    lines = [
        f"Métricas en los puntos de {'validación (nuevos)' if stage == 'validation' else 'calibración (sin validación)'}",
        f"{head[0]:<{width}}  " + "  ".join(f"{h:>11}" for h in head[1:]),
    ]
    lines.append("-" * len(lines[1]))
    for r in rows:
        cells = (num(r["bias_px"]), num(r["bias_deg"], 2), num(r["jitter_px"]), num(r["latency_ms"], 0), num(r["loo_px"]))
        lines.append(f"{str(r['name']):<{width}}  " + "  ".join(f"{c:>11}" for c in cells))
    return "\n".join(lines)
