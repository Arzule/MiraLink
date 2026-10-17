"""Calibración de 9 puntos y mapeo mirada-pantalla (lógica pura, solo numpy).

Nada aquí toca hardware ni relojes: la sesión recibe timestamps (los de TrackingFrame) y
el mapper trabaja con arreglos. Las coordenadas de pantalla son píxeles físicos.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from app_paths import data_dir

CALIBRATION_VERSION = 2  # 2: modelo, alpha por validación cruzada, recorte de extrapolación y offset
SUPPORTED_VERSIONS = (1, 2)
CALIBRATION_FILENAME = "calibration.json"
WARN_ERROR_PX = 150  # por encima de este error estimado se sugiere repetir
RAW_FEATURE_DIM = 10  # eye_tracker.FEATURE_NAMES (los vectores viejos de 9 se completan con roll = 0)
FEATURE_SETS = ("eyes", "compact", "head", "full")
MODELS = ("linear", "quad", "quad_reduced")
AGGREGATES = ("samples", "median")
ALPHA_GRID = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
CLAMP_MARGIN = 0.25  # fracción del rango calibrado que se permite extrapolar (en rasgos normalizados)
AGGREGATE_CHUNKS = 5  # "median": medianas de 5 tramos consecutivos por punto
MAX_RECENTER_FRACTION = 0.25  # un offset mayor que esta fracción de la pantalla se rechaza

# Ajuste usado por la aplicación: el de menor error de VALIDACIÓN en la primera sesión REAL
# (sessions/session_20261016_014305.npz: 118 px frente a 185 px de "compact"; comparativa en AGENTS.md).
# Solo iris: no compensa movimientos de cabeza (Recentrar corrige la deriva); confirmar con más sesiones.
DEFAULT_FIT = {
    "feature_set": "eyes",
    "model": "quad_reduced",
    "alpha": "cv",
    "aggregate": "samples",
    "auto_offset": True,  # corrige el sesgo común con los 5 puntos de validación (3 sesiones reales: 135 -> 101 px)
}

# Pose de la cabeza vigilada para el aviso de deriva: nose_x, nose_y, yaw, pitch (índices de FEATURE_NAMES)
POSE_COLS = (4, 5, 7, 8)
POSE_NAMES = ("posición horizontal", "posición vertical", "giro", "inclinación")
# Tolerancias (unidades de cada rasgo). En la sesión real 3 el pitch derivó ~0.07 rad y dio ~140 px de
# sesgo (~2000 px/rad): 0.03 rad ~ 60 px, del orden del error residual. La nariz va normalizada en el cuadro.
POSE_TOLERANCES = (0.025, 0.025, 0.03, 0.03)


class CalibrationError(Exception):
    """calibration.json inexistente, corrupto o inválido para la pantalla actual."""


def default_calibration_path() -> str:
    return os.path.join(data_dir(), CALIBRATION_FILENAME)


def pad_features(x: np.ndarray) -> np.ndarray:
    """(n, d) con d >= RAW_FEATURE_DIM; los rasgos que faltan (vectores viejos sin roll) valen 0."""
    x = np.atleast_2d(np.asarray(x, dtype=float))
    if x.shape[1] < RAW_FEATURE_DIM:
        x = np.hstack([x, np.zeros((x.shape[0], RAW_FEATURE_DIM - x.shape[1]))])
    return x


def reduce_features(x: np.ndarray, feature_set: str = "compact") -> np.ndarray:
    """Reduce el vector de rasgos de eye_tracker (FEATURE_NAMES) al conjunto pedido.

    eyes (2): [iris_along_medio, iris_perp_medio]
    compact (6): + [nose_x, nose_y, yaw, pitch]
    head (8): compact + [roll, interocular] (compensación completa de la cabeza: giro, posición y escala)
    full (10): los 10 rasgos sin promediar los ojos.
    """
    x = pad_features(x)
    along = (x[:, 0] + x[:, 2]) / 2.0
    perp = (x[:, 1] + x[:, 3]) / 2.0
    if feature_set == "eyes":
        return np.column_stack([along, perp])
    if feature_set == "compact":
        return np.column_stack([along, perp, x[:, 4], x[:, 5], x[:, 7], x[:, 8]])
    if feature_set == "head":
        return np.column_stack([along, perp, x[:, 4], x[:, 5], x[:, 7], x[:, 8], x[:, 9], x[:, 6]])
    if feature_set == "full":
        return x[:, :RAW_FEATURE_DIM]
    raise ValueError(f"feature_set desconocido: {feature_set}")


def feature_set_dim(feature_set: str) -> int:
    return {"eyes": 2, "compact": 6, "head": 8, "full": RAW_FEATURE_DIM}[feature_set]


def _design(z: np.ndarray, model: str) -> np.ndarray:
    """Términos del modelo sobre rasgos normalizados z (los dos primeros son los del iris).

    linear: [1, z]; quad: [1, z, z_i*z_j (i<=j)]; quad_reduced: [1, z, iris_a^2, iris_a*iris_p, iris_p^2]
    (curvatura solo en el iris; la cabeza entra lineal: menos parámetros con solo 9 puntos).
    """
    n, d = z.shape
    cols = [np.ones(n)]
    cols.extend(z[:, i] for i in range(d))
    if model == "quad":
        pairs = [(i, j) for i in range(d) for j in range(i, d)]
    elif model == "quad_reduced":
        pairs = [(0, 0), (0, 1), (1, 1)] if d >= 2 else [(0, 0)]
    elif model == "linear":
        pairs = []
    else:
        raise ValueError(f"modelo desconocido: {model}")
    cols.extend(z[:, i] * z[:, j] for i, j in pairs)
    return np.column_stack(cols)


def n_terms(d: int, model: str) -> int:
    return _design(np.zeros((1, d)), model).shape[1]


def _poly2(z: np.ndarray) -> np.ndarray:
    """Rasgos polinomiales de grado 2: [1, z_i, z_i*z_j (i<=j)] (compatibilidad)."""
    return _design(z, "quad")


def aggregate_point_samples(x, y, groups, chunks: int = AGGREGATE_CHUNKS):
    """Mediana por tramos: cada punto se parte en `chunks` tramos consecutivos y cada tramo aporta la
    mediana de sus muestras. Reduce el ruido del iris (y la atenuación que provoca en la regresión)
    sin perder del todo la variación de la cabeza dentro del punto."""
    x = np.atleast_2d(np.asarray(x, dtype=float))
    y = np.atleast_2d(np.asarray(y, dtype=float))
    groups = np.asarray(groups)
    xs, ys, gs = [], [], []
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        for part in np.array_split(idx, min(chunks, len(idx))):
            if len(part):
                xs.append(np.median(x[part], axis=0))
                ys.append(np.median(y[part], axis=0))
                gs.append(g)
    return np.vstack(xs), np.vstack(ys), np.asarray(gs)


def filter_point_samples(
    samples: np.ndarray,
    z_max: float = 2.5,
    pose_tol: float = 0.15,
    pose_max: float = 0.7,
    iris_floor: float = 0.003,
) -> np.ndarray:
    """Máscara de muestras válidas de UN punto de calibración.

    - Pose fuera de rango: |yaw|, |pitch| o |roll| > pose_max rad, o a más de pose_tol rad de la
      mediana del punto (la cabeza se movió durante la captura).
    - Atípicos del iris: desviación robusta (mediana y MAD x 1.4826, con piso `iris_floor`) mayor que
      `z_max` en cualquiera de los 4 rasgos del iris.
    """
    x = pad_features(samples)
    keep = np.isfinite(x).all(axis=1)
    pose = x[:, 7:10]
    keep &= (np.abs(pose) <= pose_max).all(axis=1)
    if keep.any():
        med = np.median(pose[keep], axis=0)
        keep &= (np.abs(pose - med) <= pose_tol).all(axis=1)
    if keep.any():
        eyes = x[:, :4]
        med = np.median(eyes[keep], axis=0)
        sigma = np.maximum(np.median(np.abs(eyes[keep] - med), axis=0) * 1.4826, iris_floor)
        keep &= (np.abs(eyes - med) / sigma <= z_max).all(axis=1)
    return keep


class GazeMapper:
    """Regresión ridge sobre rasgos normalizados (solo numpy).

    - feature_set: eyes / compact / head / full (reduce_features).
    - model: linear / quad / quad_reduced (_design).
    - alpha: penalización relativa al número de muestras, o "cv" para elegirla de ALPHA_GRID con
      validación cruzada dejando un punto fuera.
    - aggregate: "samples" (todas las muestras) o "median" (aggregate_point_samples).
    - Predicción: los rasgos normalizados se recortan al rango calibrado +/- CLAMP_MARGIN para no
      extrapolar (el polinomio se dispara fuera del rango); después se suma el offset de "Recentrar".
    Los valores por defecto del constructor son los del mapeo anterior; la aplicación usa DEFAULT_FIT.
    """

    def __init__(
        self,
        feature_set: str = "compact",
        alpha: Union[float, str] = 0.02,
        scale_floor: float = 0.02,
        model: str = "quad",
        aggregate: str = "samples",
        clamp: bool = True,
        auto_offset: bool = False,
    ):
        if feature_set not in FEATURE_SETS:
            raise ValueError(f"feature_set desconocido: {feature_set}")
        if model not in MODELS:
            raise ValueError(f"modelo desconocido: {model}")
        if aggregate not in AGGREGATES:
            raise ValueError(f"agregación desconocida: {aggregate}")
        self.feature_set = feature_set
        self.model = model
        self.aggregate = aggregate
        self.alpha_mode = "cv" if alpha == "cv" else "fixed"
        self.alpha = 0.02 if alpha == "cv" else float(alpha)  # en "cv" se reemplaza al ajustar
        self.scale_floor = float(scale_floor)  # evita amplificar ruido de rasgos casi constantes
        self.clamp = clamp
        self.auto_offset = auto_offset  # corrige el sesgo común con los puntos de validación
        self.pose_ref: Optional[np.ndarray] = None  # pose media de la calibración (POSE_COLS)
        self.mean: Optional[np.ndarray] = None
        self.scale: Optional[np.ndarray] = None
        self.weights: Optional[np.ndarray] = None  # (m, 2)
        self.z_min: Optional[np.ndarray] = None
        self.z_max: Optional[np.ndarray] = None
        self.offset = np.zeros(2)
        self.metrics: Dict[str, float] = {}

    @classmethod
    def from_fit_options(cls, options: Optional[dict] = None) -> "GazeMapper":
        opts = dict(DEFAULT_FIT)
        opts.update(options or {})
        return cls(
            opts["feature_set"],
            opts["alpha"],
            model=opts["model"],
            aggregate=opts["aggregate"],
            auto_offset=bool(opts.get("auto_offset", False)),
        )

    @property
    def is_fitted(self) -> bool:
        return self.weights is not None

    # ------------------------------------------------------------------ ajuste
    def _normalization(self, r: np.ndarray):
        return r.mean(axis=0), np.maximum(r.std(axis=0), self.scale_floor)

    def _solve(self, r, y, mean, scale, alpha) -> np.ndarray:
        p = _design((r - mean) / scale, self.model)
        penalty = alpha * len(r) * np.eye(p.shape[1])
        penalty[0, 0] = 0.0  # el término constante no se penaliza
        return np.linalg.solve(p.T @ p + penalty, p.T @ y)

    def _bounded(self, z, z_min, z_max):
        if not self.clamp or z_min is None:
            return z
        span = np.maximum(z_max - z_min, 1e-9)
        return np.clip(z, z_min - CLAMP_MARGIN * span, z_max + CLAMP_MARGIN * span)

    def _cv_errors(self, r, y, groups, alpha) -> np.ndarray:
        """Errores cuadrados al predecir cada punto con un ajuste sin ese punto."""
        sq = []
        for g in np.unique(groups):
            test = groups == g
            train = ~test
            mean, scale = self._normalization(r[train])
            w = self._solve(r[train], y[train], mean, scale, alpha)
            zt = (r[train] - mean) / scale
            z = self._bounded((r[test] - mean) / scale, zt.min(axis=0), zt.max(axis=0))
            sq.extend(np.sum((_design(z, self.model) @ w - y[test]) ** 2, axis=1))
        return np.asarray(sq)

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        groups: Optional[np.ndarray] = None,
        val_x: Optional[np.ndarray] = None,
        val_y: Optional[np.ndarray] = None,
        val_groups: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Ajusta el mapeo. `groups` (punto de cada muestra) activa la validación cruzada dejando un
        punto fuera; `val_*` (puntos de validación nuevos) dan el error real. Devuelve las métricas."""
        x = pad_features(x)
        y = np.atleast_2d(np.asarray(y, dtype=float))
        if len(x) != len(y) or len(x) < 3:
            raise ValueError("Se necesitan al menos 3 muestras con objetivos del mismo tamaño")
        if not (np.isfinite(x).all() and np.isfinite(y).all()):
            raise ValueError("Las muestras contienen valores no finitos")
        n_raw = len(x)
        self.pose_ref = np.median(x[:, list(POSE_COLS)], axis=0)
        if groups is not None:
            groups = np.asarray(groups)
            if self.aggregate == "median":
                x, y, groups = aggregate_point_samples(x, y, groups)
        r = reduce_features(x, self.feature_set)
        uniq = np.unique(groups) if groups is not None else np.array([])
        cv_by_alpha = {}
        if groups is not None and len(uniq) >= 3:
            alphas = ALPHA_GRID if self.alpha_mode == "cv" else (self.alpha,)
            for a in alphas:
                cv_by_alpha[a] = float(np.sqrt(np.mean(self._cv_errors(r, y, groups, a))))
            if self.alpha_mode == "cv":
                self.alpha = min(cv_by_alpha, key=cv_by_alpha.get)
        self.mean, self.scale = self._normalization(r)
        self.weights = self._solve(r, y, self.mean, self.scale, self.alpha)
        z = (r - self.mean) / self.scale
        self.z_min, self.z_max = z.min(axis=0), z.max(axis=0)
        self.offset = np.zeros(2)
        pred = self.predict(x)
        metrics = {
            "rmse_px": float(np.sqrt(np.mean(np.sum((pred - y) ** 2, axis=1)))),
            "n_samples": int(n_raw),
            "alpha": float(self.alpha),
        }
        if cv_by_alpha:
            metrics["cv_rmse_px"] = cv_by_alpha[self.alpha]
            metrics["n_points"] = int(len(uniq))
        if val_x is not None and val_y is not None and len(val_x):
            metrics.update(self.validation_metrics(val_x, val_y, val_groups))
            if self.auto_offset:
                metrics.update(self._apply_validation_offset(val_x, val_y, val_groups, y))
        self.metrics = metrics
        return metrics

    def _validation_errors(self, val_x, val_y, val_groups=None) -> np.ndarray:
        """Por punto de validación: mediana de las predicciones (sin offset) - objetivo. (k, 2)."""
        val_x = pad_features(val_x)
        val_y = np.atleast_2d(np.asarray(val_y, dtype=float))
        groups = np.zeros(len(val_x), dtype=int) if val_groups is None else np.asarray(val_groups)
        pred = self.predict(val_x, apply_offset=False)
        return np.array([np.median(pred[groups == g], axis=0) - val_y[groups == g][0] for g in np.unique(groups)])

    def validation_metrics(self, val_x, val_y, val_groups=None) -> Dict[str, float]:
        """Error con puntos NO usados al ajustar: por punto, |mediana de las predicciones - objetivo|."""
        val_x = pad_features(val_x)
        val_y = np.atleast_2d(np.asarray(val_y, dtype=float))
        pred = self.predict(val_x, apply_offset=False)
        per_point = np.linalg.norm(self._validation_errors(val_x, val_y, val_groups), axis=1)
        return {
            "validation_px": float(np.mean(per_point)),
            "validation_max_px": float(np.max(per_point)),
            "validation_rmse_px": float(np.sqrt(np.mean(np.sum((pred - val_y) ** 2, axis=1)))),
            "validation_points": int(len(per_point)),
        }

    def _apply_validation_offset(self, val_x, val_y, val_groups, cal_targets) -> Dict[str, object]:
        """Corrección automática del sesgo común (como un "Recentrar" con los puntos de validación).

        offset = -media de los errores por punto. El error que se informa es honesto: cada punto se
        evalúa con el offset calculado con los OTROS puntos (dejando uno fuera). Necesita >= 3 puntos y
        se rechaza si el offset supera MAX_RECENTER_FRACTION del área calibrada (algo falló).
        """
        errs = self._validation_errors(val_x, val_y, val_groups)
        if len(errs) < 3:
            return {}
        offset = -errs.mean(axis=0)
        loo = [np.linalg.norm(errs[i] - np.delete(errs, i, axis=0).mean(axis=0)) for i in range(len(errs))]
        span = np.ptp(np.asarray(cal_targets, dtype=float), axis=0) / 0.8  # los puntos cubren el 80 %
        out = {"validation_corrected_px": float(np.mean(loo)), "auto_offset": [float(offset[0]), float(offset[1])]}
        if (np.abs(offset) > MAX_RECENTER_FRACTION * np.maximum(span, 1.0)).any():
            out["auto_offset_rejected"] = True
            return out
        self.offset = offset
        # el offset corrige la pose de la validación: el aviso de deriva se mide desde esa pose
        self.pose_ref = np.median(pad_features(val_x)[:, list(POSE_COLS)], axis=0)
        return out

    # -------------------------------------------------------------- predicción
    def predict(self, x: np.ndarray, apply_offset: bool = True) -> np.ndarray:
        """Devuelve (n, 2) en píxeles físicos (sin recortar a la pantalla)."""
        if not self.is_fitted:
            raise RuntimeError("El mapper no está ajustado")
        z = (reduce_features(x, self.feature_set) - self.mean) / self.scale
        out = _design(self._bounded(z, self.z_min, self.z_max), self.model) @ self.weights
        return out + self.offset if apply_offset else out

    def predict_one(self, features: np.ndarray) -> Tuple[float, float]:
        out = self.predict(features)[0]
        return float(out[0]), float(out[1])

    @property
    def offset_from_validation(self) -> bool:
        return "validation_corrected_px" in self.metrics and not self.metrics.get("auto_offset_rejected")

    @property
    def error_source(self) -> str:
        """De dónde sale estimated_error_px: validation_corrected (puntos nuevos tras corregir el sesgo
        común, dejando uno fuera), validation (puntos nuevos), cv (dejando un punto fuera) o train."""
        if self.offset_from_validation:
            return "validation_corrected"
        if "validation_px" in self.metrics:
            return "validation"
        if "cv_rmse_px" in self.metrics:
            return "cv"
        return "train"

    @property
    def estimated_error_px(self) -> float:
        """Error mostrado según error_source (validación corregida > validación > cruzada > entrenamiento)."""
        key = {
            "validation_corrected": "validation_corrected_px",
            "validation": "validation_px",
            "cv": "cv_rmse_px",
            "train": "rmse_px",
        }[self.error_source]
        return float(self.metrics.get(key, float("inf")))

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
            "model": self.model,
            "aggregate": self.aggregate,
            "alpha": self.alpha,
            "alpha_mode": self.alpha_mode,
            "scale_floor": self.scale_floor,
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "weights": self.weights.tolist(),
            "z_min": None if self.z_min is None else self.z_min.tolist(),
            "z_max": None if self.z_max is None else self.z_max.tolist(),
            "offset": [float(self.offset[0]), float(self.offset[1])],
            "pose_ref": None if self.pose_ref is None else [float(v) for v in self.pose_ref],
            "metrics": self.metrics,
        }

    @classmethod
    def from_dict(cls, data: dict, geometry: Optional[Dict[str, int]] = None) -> "GazeMapper":
        """Valida y reconstruye. Lanza CalibrationError si es inválido o la geometría cambió."""
        try:
            version = data.get("version")
            if version not in SUPPORTED_VERSIONS:
                raise CalibrationError("Versión de calibración no compatible")
            if geometry is not None:
                saved = data["screen"]
                for k in ("x", "y", "width", "height"):
                    if int(saved[k]) != int(geometry[k]):
                        raise CalibrationError(
                            "La geometría de pantalla cambió desde la calibración "
                            f"({saved['width']}x{saved['height']} -> {geometry['width']}x{geometry['height']})"
                        )
            mapper = cls(
                data["feature_set"],
                float(data["alpha"]),
                float(data.get("scale_floor", 0.02)),
                model=data.get("model", "quad"),
                aggregate=data.get("aggregate", "samples"),
            )
            mapper.alpha_mode = data.get("alpha_mode", "fixed")
            mapper.mean = np.asarray(data["mean"], dtype=float)
            mapper.scale = np.asarray(data["scale"], dtype=float)
            mapper.weights = np.asarray(data["weights"], dtype=float)
            if data.get("z_min") is not None and data.get("z_max") is not None:
                mapper.z_min = np.asarray(data["z_min"], dtype=float)
                mapper.z_max = np.asarray(data["z_max"], dtype=float)
            mapper.offset = np.asarray(data.get("offset", [0.0, 0.0]), dtype=float)
            if data.get("pose_ref") is not None:
                mapper.pose_ref = np.asarray(data["pose_ref"], dtype=float)
                if mapper.pose_ref.shape != (len(POSE_COLS),) or not np.isfinite(mapper.pose_ref).all():
                    mapper.pose_ref = None  # opcional: sin referencia solo se pierde el aviso de deriva
            mapper.metrics = dict(data.get("metrics", {}))
        except CalibrationError:
            raise
        except (KeyError, TypeError, ValueError) as e:
            raise CalibrationError(f"calibration.json inválido: {e}") from e
        d = feature_set_dim(mapper.feature_set)
        if version == 1 and mapper.feature_set == "full":
            d = 9  # la versión 1 no tenía roll
        shapes_ok = (
            mapper.mean.shape == (d,)
            and mapper.scale.shape == (d,)
            and mapper.weights.shape == (n_terms(d, mapper.model), 2)
            and mapper.offset.shape == (2,)
            and (mapper.z_min is None or (mapper.z_min.shape == (d,) and mapper.z_max.shape == (d,)))
        )
        if not shapes_ok or (version == 1 and mapper.feature_set == "full"):
            raise CalibrationError("calibration.json inválido: dimensiones inesperadas; calibra de nuevo")
        for arr in (mapper.mean, mapper.scale, mapper.weights, mapper.offset):
            if not np.isfinite(arr).all():
                raise CalibrationError("calibration.json inválido: valores no finitos")
        if mapper.z_min is not None and not (np.isfinite(mapper.z_min).all() and np.isfinite(mapper.z_max).all()):
            raise CalibrationError("calibration.json inválido: valores no finitos")
        if (mapper.scale <= 0).any():
            raise CalibrationError("calibration.json inválido: escala no positiva")
        return mapper


def save_calibration(path: str, mapper: GazeMapper, geometry: Dict[str, int], camera_index: int = 0) -> None:
    """Guarda calibration.json de forma atómica."""
    _write_json_atomic(path, mapper.to_dict(geometry, camera_index))


def _write_json_atomic(path: str, data: dict) -> None:
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


def save_offset(path: str, offset: Tuple[float, float]) -> None:
    """Guarda el offset de "Recentrar" en calibration.json sin tocar el resto del ajuste."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        raise CalibrationError(f"calibration.json ilegible: {e}") from e
    if not isinstance(data, dict):
        raise CalibrationError("calibration.json inválido")
    data["offset"] = [float(offset[0]), float(offset[1])]
    _write_json_atomic(path, data)


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


# Orden de respaldo: cada tramo de 3 puntos consecutivos cubre las 3 filas y las 3 columnas
BALANCED_ORDER = (0, 4, 8, 5, 6, 1, 7, 2, 3)


def is_balanced_order(order) -> bool:
    """True si cada tramo de 3 puntos consecutivos (índices de grid_points) tiene una fila y una columna distintas."""
    for k in range(0, 9, 3):
        chunk = order[k : k + 3]
        if len({i // 3 for i in chunk}) != 3 or len({i % 3 for i in chunk}) != 3:
            return False
    return True


def balanced_order(rng: Optional[np.random.Generator] = None) -> List[int]:
    """Orden aleatorio de los 9 puntos con cada tramo de 3 cubriendo todas las filas y columnas.

    Con el orden por filas (arriba -> abajo) un movimiento de cabeza durante la calibración quedaba
    ligado a la altura del punto, y el mapeo aprendía "cabeza baja = mirar abajo" (sesión real del
    2026-10-16: sesgo común de +169 px en la validación). Así la posición en el tiempo no dice la fila
    ni la columna.
    """
    rng = rng if rng is not None else np.random.default_rng()
    for _ in range(2000):
        order = [int(i) for i in rng.permutation(9)]
        if is_balanced_order(order):
            return order
    return list(BALANCED_ORDER)


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

    `order="balanced"` (lo usa la aplicación) presenta los 9 puntos en un orden aleatorio balanceado
    (balanced_order); "grid" (por defecto de la clase) los presenta por filas. `points` está en orden
    de presentación y los grupos del resultado son índices de presentación.

    Con `robust` (por defecto) se descartan además los cuadros de los `post_blink_s` siguientes a un
    parpadeo y, al cerrar cada punto, los atípicos y las poses de cabeza fuera de rango
    (filter_point_samples); `min_samples` se cuenta después de filtrar.
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
        robust: bool = True,
        post_blink_s: float = 0.15,
        order: str = "grid",
        rng: Optional[np.random.Generator] = None,
    ):
        grid = grid_points(width, height, margin)
        if order == "balanced":
            self.order = balanced_order(rng)
        elif order == "grid":
            self.order = list(range(len(grid)))
        else:
            raise ValueError(f"orden desconocido: {order}")
        self.points = [grid[i] for i in self.order]  # en orden de presentación
        self.validation = list(validation or [])
        self.settle_s = settle_s
        self.capture_s = capture_s
        self.min_samples = min_samples
        self.max_retries = max_retries
        self.robust = robust
        self.post_blink_s = post_blink_s
        self.discarded = 0  # muestras descartadas por atípicas o por pose (diagnóstico)
        self._last_closed: Optional[float] = None
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

    def update(self, timestamp: float, features: Optional[np.ndarray], valid: bool, eyes_closed: bool = False) -> str:
        """Procesa un cuadro (o un tick sin rostro con valid=False). Devuelve la fase actual."""
        if self.finished:
            return self.phase
        if eyes_closed:
            self._last_closed = timestamp
        if valid and self.robust and self._last_closed is not None and timestamp - self._last_closed < self.post_blink_s:
            valid = False  # justo tras un parpadeo el iris aún no es fiable
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

    def _clean_samples(self):
        if self.robust and self._samples:
            keep = filter_point_samples(np.vstack(self._samples))
            self.discarded += int((~keep).sum())
            self._samples = [s for s, k in zip(self._samples, keep) if k]

    def _finish_attempt(self, timestamp: float):
        self._clean_samples()
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


def fit_session(session: CalibrationSession, options: Optional[dict] = None) -> GazeMapper:
    """Ajusta un GazeMapper (DEFAULT_FIT salvo `options`) con una sesión terminada; si hubo fase de
    validación, sus puntos nuevos dan el error mostrado (metrics["validation_px"])."""
    res = session.result()
    mapper = GazeMapper.from_fit_options(options)
    if res.has_validation:
        mapper.fit(res.features, res.targets, res.groups, res.val_features, res.val_targets, res.val_groups)
    else:
        mapper.fit(res.features, res.targets, res.groups)
    mapper.metrics["discarded_samples"] = int(session.discarded)
    return mapper


# ------------------------------------------------------------------ recentrar
class RecenterSession:
    """"Recentrar": un único punto (el centro), ~1 s de muestras y un offset constante al mapeo.

    Usa timestamps inyectados. Tras `settle_s` captura `capture_s`; con al menos `min_samples`
    muestras válidas calcula offset = objetivo - mediana(predicción sin offset). Si no las consigue
    en `max_wait_s`, o el offset supera MAX_RECENTER_FRACTION de la pantalla, falla con un mensaje.
    """

    def __init__(
        self,
        width: int,
        height: int,
        settle_s: float = 0.6,
        capture_s: float = 1.0,
        min_samples: int = 15,
        max_wait_s: float = 6.0,
        post_blink_s: float = 0.15,
    ):
        self.width, self.height = int(width), int(height)
        self.target = (self.width / 2.0, self.height / 2.0)
        self.settle_s = settle_s
        self.capture_s = capture_s
        self.min_samples = min_samples
        self.max_wait_s = max_wait_s
        self.post_blink_s = post_blink_s
        self.phase = PHASE_SETTLING
        self.message = ""
        self.offset: Optional[Tuple[float, float]] = None
        self._start: Optional[float] = None
        self._capture_start: Optional[float] = None
        self._last_closed: Optional[float] = None
        self._samples: List[np.ndarray] = []

    @property
    def finished(self) -> bool:
        return self.phase in (PHASE_DONE, PHASE_ABORTED)

    def progress(self, now: float) -> float:
        if self._capture_start is None:
            return 0.0
        return float(min(1.0, max(0.0, (now - self._capture_start) / self.capture_s)))

    def update(self, timestamp: float, features, valid: bool, eyes_closed: bool, mapper) -> str:
        if self.finished:
            return self.phase
        if self._start is None:
            self._start = timestamp
        if eyes_closed:
            self._last_closed = timestamp
        if self._last_closed is not None and timestamp - self._last_closed < self.post_blink_s:
            valid = False
        if self.phase == PHASE_SETTLING and timestamp - self._start >= self.settle_s:
            self.phase = PHASE_CAPTURING
            self._capture_start = timestamp
        if self.phase == PHASE_CAPTURING:
            if valid and features is not None:
                self._samples.append(np.asarray(features, dtype=float).copy())
            if timestamp - self._capture_start >= self.capture_s and len(self._samples) >= self.min_samples:
                self._finish(mapper)
                return self.phase
        if timestamp - self._start >= self.max_wait_s:
            self.phase = PHASE_ABORTED
            self.message = (
                f"No se pudo recentrar: solo {len(self._samples)} muestras válidas. "
                "Mira el punto central con el rostro visible."
            )
        return self.phase

    def _finish(self, mapper):
        x = np.vstack(self._samples)
        x = x[filter_point_samples(x)]
        if len(x) < max(3, self.min_samples // 2):
            self.phase = PHASE_ABORTED
            self.message = "No se pudo recentrar: demasiadas muestras inestables."
            return
        pred = np.median(mapper.predict(x, apply_offset=False), axis=0)
        dx, dy = self.target[0] - pred[0], self.target[1] - pred[1]
        if abs(dx) > MAX_RECENTER_FRACTION * self.width or abs(dy) > MAX_RECENTER_FRACTION * self.height:
            self.phase = PHASE_ABORTED
            self.message = (
                f"El desvío medido ({dx:+.0f}, {dy:+.0f}) px es demasiado grande para recentrar; "
                "calibra el rastreador de nuevo."
            )
            return
        self.offset = (float(dx), float(dy))
        self.phase = PHASE_DONE


# ------------------------------------------------------------ deriva de la pose
class PoseDriftMonitor:
    """Avisa cuando la cabeza se aleja de la pose de la calibración (con solo iris, eso desvía el cursor).

    Compara la mediana de la última `window_s` de POSE_COLS con la referencia; si alguna componente
    supera su tolerancia durante `hold_s` seguidos pasa a "deriva" y vuelve a normal cuando todas
    bajan del 70 % de su tolerancia (histéresis). Timestamps inyectados, sin reloj propio.
    """

    def __init__(self, reference, tolerances=POSE_TOLERANCES, window_s: float = 1.0, hold_s: float = 2.0):
        self.reference = np.asarray(reference, dtype=float)
        self.tolerances = np.asarray(tolerances, dtype=float)
        self.window_s = window_s
        self.hold_s = hold_s
        self.drifted = False
        self._buf: List[Tuple[float, np.ndarray]] = []
        self._since: Optional[float] = None

    def current_pose(self) -> Optional[np.ndarray]:
        if not self._buf:
            return None
        return np.median(np.vstack([p for _, p in self._buf]), axis=0)

    def deviation(self) -> Optional[np.ndarray]:
        """Desviación de la pose actual respecto a la referencia, en múltiplos de la tolerancia."""
        pose = self.current_pose()
        return None if pose is None else np.abs(pose - self.reference) / self.tolerances

    def update(self, timestamp: float, features) -> bool:
        """Procesa los rasgos de un cuadro con rostro. Devuelve si hay deriva."""
        x = pad_features(features)[0]
        pose = x[list(POSE_COLS)]
        if not np.isfinite(pose).all():
            return self.drifted
        self._buf.append((float(timestamp), pose))
        while self._buf and timestamp - self._buf[0][0] > self.window_s:
            self._buf.pop(0)
        dev = self.deviation()
        if self.drifted:
            if (dev < 0.7).all():
                self.drifted = False
                self._since = None
            return self.drifted
        if (dev > 1.0).any():
            if self._since is None:
                self._since = timestamp
            if timestamp - self._since >= self.hold_s:
                self.drifted = True
        else:
            self._since = None
        return self.drifted

    def rebase(self):
        """Tras "Recentrar": la pose actual pasa a ser la referencia."""
        pose = self.current_pose()
        if pose is not None:
            self.reference = pose
        self.drifted = False
        self._since = None

    def message(self) -> str:
        dev = self.deviation()
        if dev is None:
            return ""
        parts = [name for name, d in zip(POSE_NAMES, dev) if d > 1.0] or [POSE_NAMES[int(np.argmax(dev))]]
        return (
            f"Tu cabeza cambió de {' y '.join(parts)} respecto a la calibración y el cursor puede desviarse: "
            "pulsa Ctrl+Alt+R (o «Recentrar») mirando el punto central."
        )


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
# 0.21: en las sesiones reales, con 0.19 (más lejos) la señal del iris / ruido bajó de ~70 a ~33 y el
# jitter se duplicó (26 -> 52-61 px) frente a 0.23. Por debajo se pide acercarse.
IOD_MIN = 0.21
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
