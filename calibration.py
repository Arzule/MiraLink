"""Calibración de 9 puntos y mapeo mirada-pantalla (lógica pura, solo numpy).

Nada aquí toca hardware ni relojes: la sesión recibe timestamps (los de TrackingFrame) y
el mapper trabaja con arreglos. Las coordenadas de pantalla son píxeles físicos.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

CALIBRATION_VERSION = 1
CALIBRATION_FILENAME = "calibration.json"
WARN_ERROR_PX = 150  # por encima de este error estimado se sugiere repetir
FEATURE_SETS = ("compact", "full")


class CalibrationError(Exception):
    """calibration.json inexistente, corrupto o inválido para la pantalla actual."""


def default_calibration_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), CALIBRATION_FILENAME)


def reduce_features(x: np.ndarray, feature_set: str = "compact") -> np.ndarray:
    """Reduce el vector de 9 rasgos de eye_tracker.

    compact (6): [iris_along_medio, iris_perp_medio, nose_x, nose_y, yaw, pitch]
    full (9): sin cambios. Se excluye la distancia interocular (proxy de profundidad) en compact.
    """
    x = np.atleast_2d(np.asarray(x, dtype=float))
    if feature_set == "full":
        return x
    if feature_set != "compact":
        raise ValueError(f"feature_set desconocido: {feature_set}")
    return np.column_stack(
        [(x[:, 0] + x[:, 2]) / 2.0, (x[:, 1] + x[:, 3]) / 2.0, x[:, 4], x[:, 5], x[:, 7], x[:, 8]]
    )


def _poly2(z: np.ndarray) -> np.ndarray:
    """Rasgos polinomiales de grado 2: [1, z_i, z_i*z_j (i<=j)]."""
    n, d = z.shape
    cols = [np.ones(n)]
    cols.extend(z[:, i] for i in range(d))
    for i in range(d):
        for j in range(i, d):
            cols.append(z[:, i] * z[:, j])
    return np.column_stack(cols)


class GazeMapper:
    """Regresión ridge con rasgos polinomiales de grado 2 (normalización + ridge, solo numpy)."""

    def __init__(self, feature_set: str = "compact", alpha: float = 0.02, scale_floor: float = 0.02):
        if feature_set not in FEATURE_SETS:
            raise ValueError(f"feature_set desconocido: {feature_set}")
        self.feature_set = feature_set
        self.alpha = float(alpha)  # penalización relativa al número de muestras
        self.scale_floor = float(scale_floor)  # evita amplificar ruido de rasgos casi constantes
        self.mean: Optional[np.ndarray] = None
        self.scale: Optional[np.ndarray] = None
        self.weights: Optional[np.ndarray] = None  # (m, 2)
        self.metrics: Dict[str, float] = {}

    @property
    def is_fitted(self) -> bool:
        return self.weights is not None

    def _design(self, x: np.ndarray) -> np.ndarray:
        r = reduce_features(x, self.feature_set)
        return _poly2((r - self.mean) / self.scale)

    def _solve(self, r: np.ndarray, y: np.ndarray, mean, scale) -> np.ndarray:
        p = _poly2((r - mean) / scale)
        penalty = self.alpha * len(r) * np.eye(p.shape[1])
        penalty[0, 0] = 0.0  # el término constante no se penaliza
        return np.linalg.solve(p.T @ p + penalty, p.T @ y)

    def _normalization(self, r: np.ndarray):
        return r.mean(axis=0), np.maximum(r.std(axis=0), self.scale_floor)

    def fit(self, x: np.ndarray, y: np.ndarray, groups: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Ajusta el mapeo. `groups` (punto de calibración de cada muestra) activa la validación
        cruzada dejando un punto fuera. Devuelve y guarda métricas (rmse_px, cv_rmse_px)."""
        x = np.atleast_2d(np.asarray(x, dtype=float))
        y = np.atleast_2d(np.asarray(y, dtype=float))
        if len(x) != len(y) or len(x) < 3:
            raise ValueError("Se necesitan al menos 3 muestras con objetivos del mismo tamaño")
        if not (np.isfinite(x).all() and np.isfinite(y).all()):
            raise ValueError("Las muestras contienen valores no finitos")
        r = reduce_features(x, self.feature_set)
        self.mean, self.scale = self._normalization(r)
        self.weights = self._solve(r, y, self.mean, self.scale)
        pred = self.predict(x)
        metrics = {
            "rmse_px": float(np.sqrt(np.mean(np.sum((pred - y) ** 2, axis=1)))),
            "n_samples": int(len(x)),
        }
        if groups is not None:
            groups = np.asarray(groups)
            uniq = np.unique(groups)
            if len(uniq) >= 3:
                sq = []
                for g in uniq:
                    test = groups == g
                    train = ~test
                    mean, scale = self._normalization(r[train])
                    w = self._solve(r[train], y[train], mean, scale)
                    p = _poly2((r[test] - mean) / scale) @ w
                    sq.extend(np.sum((p - y[test]) ** 2, axis=1))
                metrics["cv_rmse_px"] = float(np.sqrt(np.mean(sq)))
                metrics["n_points"] = int(len(uniq))
        self.metrics = metrics
        return metrics

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Devuelve (n, 2) en píxeles físicos (sin recortar a la pantalla)."""
        if not self.is_fitted:
            raise RuntimeError("El mapper no está ajustado")
        return self._design(np.atleast_2d(np.asarray(x, dtype=float))) @ self.weights

    def predict_one(self, features: np.ndarray) -> Tuple[float, float]:
        out = self.predict(features)[0]
        return float(out[0]), float(out[1])

    @property
    def estimated_error_px(self) -> float:
        """Error estimado: validación cruzada si existe; si no, el RMSE de entrenamiento."""
        return float(self.metrics.get("cv_rmse_px", self.metrics.get("rmse_px", float("inf"))))

    # ----------------------------------------------------- serialización
    def to_dict(self, geometry: Dict[str, int], camera_index: int = 0) -> dict:
        if not self.is_fitted:
            raise RuntimeError("El mapper no está ajustado")
        return {
            "version": CALIBRATION_VERSION,
            "created": datetime.now().isoformat(timespec="seconds"),
            "camera_index": int(camera_index),
            "screen": {k: int(geometry[k]) for k in ("x", "y", "width", "height")},
            "feature_set": self.feature_set,
            "degree": 2,
            "alpha": self.alpha,
            "scale_floor": self.scale_floor,
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "weights": self.weights.tolist(),
            "metrics": self.metrics,
        }

    @classmethod
    def from_dict(cls, data: dict, geometry: Optional[Dict[str, int]] = None) -> "GazeMapper":
        """Valida y reconstruye. Lanza CalibrationError si es inválido o la geometría cambió."""
        try:
            if data.get("version") != CALIBRATION_VERSION:
                raise CalibrationError("Versión de calibración no compatible")
            if geometry is not None:
                saved = data["screen"]
                for k in ("x", "y", "width", "height"):
                    if int(saved[k]) != int(geometry[k]):
                        raise CalibrationError(
                            "La geometría de pantalla cambió desde la calibración "
                            f"({saved['width']}x{saved['height']} -> {geometry['width']}x{geometry['height']})"
                        )
            mapper = cls(data["feature_set"], float(data["alpha"]), float(data.get("scale_floor", 0.02)))
            mapper.mean = np.asarray(data["mean"], dtype=float)
            mapper.scale = np.asarray(data["scale"], dtype=float)
            mapper.weights = np.asarray(data["weights"], dtype=float)
            mapper.metrics = dict(data.get("metrics", {}))
        except CalibrationError:
            raise
        except (KeyError, TypeError, ValueError) as e:
            raise CalibrationError(f"calibration.json inválido: {e}") from e
        d = 6 if mapper.feature_set == "compact" else 9
        m = 1 + d + d * (d + 1) // 2
        if mapper.mean.shape != (d,) or mapper.scale.shape != (d,) or mapper.weights.shape != (m, 2):
            raise CalibrationError("calibration.json inválido: dimensiones inesperadas")
        for arr in (mapper.mean, mapper.scale, mapper.weights):
            if not np.isfinite(arr).all():
                raise CalibrationError("calibration.json inválido: valores no finitos")
        if (mapper.scale <= 0).any():
            raise CalibrationError("calibration.json inválido: escala no positiva")
        return mapper


def save_calibration(path: str, mapper: GazeMapper, geometry: Dict[str, int], camera_index: int = 0) -> None:
    """Guarda calibration.json de forma atómica."""
    data = mapper.to_dict(geometry, camera_index)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def load_calibration(path: str, geometry: Optional[Dict[str, int]] = None) -> GazeMapper:
    """Carga y valida calibration.json. Lanza CalibrationError con el motivo si no sirve."""
    if not os.path.isfile(path):
        raise CalibrationError("No hay calibración guardada")
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        raise CalibrationError(f"calibration.json ilegible: {e}") from e
    if not isinstance(data, dict):
        raise CalibrationError("calibration.json inválido")
    return GazeMapper.from_dict(data, geometry)


# ---------------------------------------------------------------- sesión
PHASE_SETTLING = "settling"
PHASE_CAPTURING = "capturing"
PHASE_DONE = "done"
PHASE_ABORTED = "aborted"

STAGE_CALIBRATION = "calibration"
STAGE_VALIDATION = "validation"

# Puntos de validación (fracción de la pantalla): centro desplazado y cuatro intermedios, todos
# distintos de la cuadrícula 3x3 de calibración. Miden el error real con puntos no usados al ajustar.
VALIDATION_LAYOUT = ((0.30, 0.30), (0.70, 0.30), (0.42, 0.56), (0.30, 0.70), (0.70, 0.70))


def grid_points(width: int, height: int, margin: float = 0.10) -> List[Tuple[float, float]]:
    """Cuadrícula 3x3 con márgenes proporcionales, de izquierda a derecha y de arriba abajo."""
    xs = [width * margin, width * 0.5, width * (1.0 - margin)]
    ys = [height * margin, height * 0.5, height * (1.0 - margin)]
    return [(x, y) for y in ys for x in xs]


def validation_points(width: int, height: int) -> List[Tuple[float, float]]:
    """Los 5 puntos de validación en píxeles físicos (ver VALIDATION_LAYOUT)."""
    return [(width * fx, height * fy) for fx, fy in VALIDATION_LAYOUT]


@dataclass
class CalibrationResult:
    features: np.ndarray  # (n, d)
    targets: np.ndarray  # (n, 2) píxeles físicos
    groups: np.ndarray  # (n,) índice del punto
    val_features: Optional[np.ndarray] = None  # (m, d) muestras de los puntos de validación
    val_targets: Optional[np.ndarray] = None
    val_groups: Optional[np.ndarray] = None

    @property
    def has_validation(self) -> bool:
        return self.val_features is not None and len(self.val_features) > 0


class CalibrationSession:
    """Flujo de calibración de 9 puntos con timestamps inyectados (sin reloj propio).

    Por punto: `settle_s` de asentamiento (se descartan los cuadros) y `capture_s` de captura.
    Se descartan cuadros sin rostro u ojos cerrados. Si al terminar la captura hay menos de
    `min_samples` muestras válidas se reintenta el punto hasta `max_retries` veces y luego se aborta.

    Con `validation` (lista de puntos) la sesión sigue con una fase de validación con el mismo
    ritmo; un punto de validación sin muestras suficientes tras los reintentos se omite (no aborta).
    """

    def __init__(
        self,
        width: int,
        height: int,
        margin: float = 0.10,
        settle_s: float = 1.0,
        capture_s: float = 1.5,
        min_samples: int = 20,
        max_retries: int = 2,
        validation: Optional[List[Tuple[float, float]]] = None,
    ):
        self.points = grid_points(width, height, margin)
        self.validation = list(validation or [])
        self.settle_s = settle_s
        self.capture_s = capture_s
        self.min_samples = min_samples
        self.max_retries = max_retries
        self.stage = STAGE_CALIBRATION
        self.point_index = 0
        self.validation_index = 0
        self.retries = 0
        self.phase = PHASE_SETTLING
        self.message = ""
        self._phase_start: Optional[float] = None
        self._samples: List[np.ndarray] = []
        self._done_features: List[np.ndarray] = []
        self._done_targets: List[np.ndarray] = []
        self._done_groups: List[np.ndarray] = []
        self._val_features: List[np.ndarray] = []
        self._val_targets: List[np.ndarray] = []
        self._val_groups: List[np.ndarray] = []

    @property
    def total(self) -> int:
        return len(self.points)

    @property
    def in_validation(self) -> bool:
        return self.stage == STAGE_VALIDATION

    @property
    def active_index(self) -> int:
        """Índice del punto activo dentro de su fase (calibración o validación)."""
        return self.validation_index if self.in_validation else self.point_index

    @property
    def current_point(self) -> Tuple[float, float]:
        if self.in_validation:
            return self.validation[min(self.validation_index, len(self.validation) - 1)]
        return self.points[min(self.point_index, self.total - 1)]

    @property
    def finished(self) -> bool:
        return self.phase in (PHASE_DONE, PHASE_ABORTED)

    @property
    def sample_count(self) -> int:
        return len(self._samples)

    def phase_progress(self, now: float) -> float:
        """Avance 0..1 de la fase actual."""
        if self._phase_start is None or self.finished:
            return 0.0
        span = self.settle_s if self.phase == PHASE_SETTLING else self.capture_s
        return float(min(1.0, max(0.0, (now - self._phase_start) / span)))

    def update(self, timestamp: float, features: Optional[np.ndarray], valid: bool) -> str:
        """Procesa un cuadro (o un tick sin rostro con valid=False). Devuelve la fase actual."""
        if self.finished:
            return self.phase
        if self._phase_start is None:
            self._phase_start = timestamp
        elapsed = timestamp - self._phase_start
        if self.phase == PHASE_SETTLING:
            if elapsed < self.settle_s:
                return self.phase
            self.phase = PHASE_CAPTURING
            self._phase_start = timestamp
            elapsed = 0.0
        if self.phase == PHASE_CAPTURING:
            if elapsed >= self.capture_s:
                self._finish_attempt(timestamp)
                return self.phase
            if valid and features is not None:
                self._samples.append(np.asarray(features, dtype=float).copy())
        return self.phase

    def _finish_attempt(self, timestamp: float):
        if self.in_validation:
            self._finish_validation_attempt(timestamp)
            return
        n = len(self._samples)
        idx = self.point_index
        if n >= self.min_samples:
            tx, ty = self.points[idx]
            self._done_features.append(np.vstack(self._samples))
            self._done_targets.append(np.tile([tx, ty], (n, 1)))
            self._done_groups.append(np.full(n, idx))
            self._samples = []
            self.retries = 0
            self.message = ""
            if idx + 1 >= self.total:
                if not self.validation:
                    self.phase = PHASE_DONE
                    return
                self.stage = STAGE_VALIDATION
                self.validation_index = 0
            else:
                self.point_index = idx + 1
        else:
            if self.retries >= self.max_retries:
                self.phase = PHASE_ABORTED
                self.message = (
                    f"No se pudo capturar el punto {idx + 1} de {self.total}: solo {n} muestras válidas "
                    f"(se necesitan {self.min_samples}). Revisa la iluminación y que tu rostro esté visible."
                )
                return
            self.retries += 1
            self.message = (
                f"Punto {idx + 1}: pocas muestras válidas ({n}); reintentando ({self.retries}/{self.max_retries})."
            )
            self._samples = []
        self.phase = PHASE_SETTLING
        self._phase_start = timestamp

    def _finish_validation_attempt(self, timestamp: float):
        n = len(self._samples)
        idx = self.validation_index
        if n >= self.min_samples:
            tx, ty = self.validation[idx]
            self._val_features.append(np.vstack(self._samples))
            self._val_targets.append(np.tile([tx, ty], (n, 1)))
            self._val_groups.append(np.full(n, idx))
            self.message = ""
        elif self.retries < self.max_retries:
            self.retries += 1
            self.message = (
                f"Validación {idx + 1}: pocas muestras válidas ({n}); reintentando ({self.retries}/{self.max_retries})."
            )
            self._samples = []
            self.phase = PHASE_SETTLING
            self._phase_start = timestamp
            return
        else:
            self.message = f"Se omitió el punto de validación {idx + 1}: solo {n} muestras válidas."
        self._samples = []
        self.retries = 0
        if idx + 1 >= len(self.validation):
            self.phase = PHASE_DONE
            return
        self.validation_index = idx + 1
        self.phase = PHASE_SETTLING
        self._phase_start = timestamp

    def result(self) -> CalibrationResult:
        if self.phase != PHASE_DONE:
            raise RuntimeError("La calibración no terminó")
        val = (None, None, None)
        if self._val_features:
            val = (np.vstack(self._val_features), np.vstack(self._val_targets), np.concatenate(self._val_groups))
        return CalibrationResult(
            np.vstack(self._done_features), np.vstack(self._done_targets), np.concatenate(self._done_groups), *val
        )


def fit_session(session: CalibrationSession, feature_set: str = "compact") -> GazeMapper:
    """Ajusta un GazeMapper con el resultado de una sesión terminada."""
    res = session.result()
    mapper = GazeMapper(feature_set)
    mapper.fit(res.features, res.targets, res.groups)
    return mapper


# ------------------------------------------------- utilidades de interfaz
class HoldToExit:
    """Mantener una tecla `duration_s` segundos para salir (timestamps inyectados)."""

    def __init__(self, duration_s: float = 1.5):
        self.duration_s = duration_s
        self._start: Optional[float] = None

    def press(self, now: float):
        if self._start is None:
            self._start = now

    def release(self):
        self._start = None

    def progress(self, now: float) -> float:
        if self._start is None:
            return 0.0
        return float(min(1.0, max(0.0, (now - self._start) / self.duration_s)))

    def completed(self, now: float) -> bool:
        return self.progress(now) >= 1.0


# Distancia interocular (fracción del ancho del cuadro) aceptable y tolerancia de centrado
IOD_MIN = 0.14
IOD_MAX = 0.32
CENTER_TOL_X = 0.15
CENTER_TOL_Y = 0.18


def face_guidance(face_found: bool, face_center, interocular: float) -> Tuple[bool, str]:
    """(rostro_ok, aviso) para la insignia del encuadre: Acércate / Aléjate / Centra tu rostro."""
    if not face_found or face_center is None:
        return False, "Rostro no detectado"
    if interocular < IOD_MIN:
        return True, "Acércate"
    if interocular > IOD_MAX:
        return True, "Aléjate"
    if abs(face_center[0] - 0.5) > CENTER_TOL_X or abs(face_center[1] - 0.5) > CENTER_TOL_Y:
        return True, "Centra tu rostro"
    return True, ""


def screen_geometry_from_qt(screen) -> Dict[str, int]:
    """Geometría física (píxeles) de un QScreen: geometry() x devicePixelRatio()."""
    g = screen.geometry()
    dpr = float(screen.devicePixelRatio())
    return {
        "x": int(round(g.x() * dpr)),
        "y": int(round(g.y() * dpr)),
        "width": int(round(g.width() * dpr)),
        "height": int(round(g.height() * dpr)),
    }
