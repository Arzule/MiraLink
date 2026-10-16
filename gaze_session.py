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

from calibration import (
    PHASE_CAPTURING,
    PHASE_SETTLING,
    STAGE_VALIDATION,
    CalibrationSession,
    GazeMapper,
    validation_points,
)
from config_manager import DEFAULT_CONFIG
from eye_tracker import FEATURE_DIM, FEATURE_NAMES, NUM_LANDMARKS, BlinkEvent, TrackingFrame

SESSION_VERSION = 1
SESSIONS_DIRNAME = "sessions"
DEFAULT_DISTANCE_MM = 600.0  # distancia de visión supuesta para convertir px a grados

STAGE_CODES = {"calibration": 0, "validation": 1}
PHASE_CODES = {PHASE_SETTLING: 0, PHASE_CAPTURING: 1}  # cualquier otra fase -> 2


def default_sessions_dir() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), SESSIONS_DIRNAME)


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
        "calibration_points": [list(map(float, p)) for p in session.points],
        "validation_points": [list(map(float, p)) for p in session.validation],
        "settle_s": session.settle_s,
        "capture_s": session.capture_s,
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
        }
        matrix = np.full((n, 4, 4), np.nan, dtype=np.float32)
        for i, m in enumerate(self._matrix):
            if m is not None and m.shape[0] >= 4 and m.shape[1] >= 4:
                matrix[i] = m[:4, :4]
        out["matrix"] = matrix
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
        meta = dict(self.meta, n_frames=len(self))
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

    def frame(self, t: float, target: Tuple[float, float], stage: str = "calibration") -> TrackingFrame:
        dt = 1.0 / 30.0 if self._t is None else max(1e-3, t - self._t)
        self._t = t
        shift = stage == STAGE_VALIDATION
        mu_pos = np.array([0.0, 0.0, self.distance]) + (self.val_shift_mm if shift else 0.0)
        mu_rot = np.array([0.0, -0.05, 0.0]) + (self.val_shift_rad if shift else 0.0)
        self.pos = self._ou(self.pos, mu_pos, self.head_sigma_mm, dt)
        self.rot = self._ou(self.rot, mu_rot, self.head_sigma_rad, dt)

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

        feats = self._features(gx, gy, closed, reopening)
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

    def _features(self, gx, gy, closed, reopening) -> np.ndarray:
        """Vector completo (10 rasgos, incluye roll); se recorta a FEATURE_DIM."""
        rng = self.rng
        hx, hy, hz = self.pos
        yaw, pitch, roll = self.rot
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
) -> RecordedSession:
    """Sesión sintética que reproduce el flujo real: la CalibrationSession decide puntos y fases."""
    rng = np.random.default_rng(seed)
    user = SyntheticUser(rng, width, height, physical_mm, **(user_kwargs or {}))
    session = CalibrationSession(
        width, height, validation=validation_points(width, height) if validation else None, **(session_kwargs or {})
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
        frame = user.frame(t, session.current_point, session.stage)
        rec.add(frame, session)
        valid = frame.face_found and not frame.eyes_closed and frame.features is not None
        session.update(t, frame.features, valid)
    return rec.to_session()


# ------------------------------------------------------------------- evaluación
def fit_baseline(x, y, groups):
    """Mapeo anterior a esta mejora: compacto, cuadrático completo, ridge alpha=0.02, todas las muestras."""
    mapper = GazeMapper("compact", alpha=0.02)
    mapper.fit(x, y, groups)
    return mapper


@dataclass
class Variant:
    """Una combinación de mapeo (cómo se ajusta) y filtro (claves de interaction y opciones del controlador)."""

    name: str
    fit: Callable[[np.ndarray, np.ndarray, np.ndarray], object] = fit_baseline
    interaction: Dict = field(default_factory=dict)
    controller_kwargs: Dict = field(default_factory=dict)


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
    x, y, g = session.samples("calibration")
    if len(x) < 3:
        raise ValueError("La sesión no tiene muestras de calibración suficientes")
    stage = "validation" if session.has_validation else "calibration"
    rows = []
    for v in variants:
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


def default_variants() -> List[Variant]:
    """Variantes que compara scripts/replay_session.py (la primera es la referencia anterior)."""
    return [
        Variant("anterior: compacto cuadr. a=0.02 + exponencial", fit_baseline),
        Variant("anterior sin suavizado (referencia)", fit_baseline, {"smoothing_percent": 0}),
    ]


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
