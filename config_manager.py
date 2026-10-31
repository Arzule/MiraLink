"""Módulo para cargar y guardar config.json (valores por defecto y validación de rangos)."""

import json
import logging
import os
import tempfile
from typing import Optional, Tuple

from app_paths import data_dir

# TODO(etapa 1)

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = {
    "version": 1,
    # resolution "native": la mayor resolución que la cámara sostiene a ~30 fps (hasta 1920x1080);
    # "fixed": width x height (si no llega a ~24 fps se usa 640x480)
    # exposure "auto": la de la cámara; "fast": exposición fija corta (~31 ms) para 30 fps, imagen más oscura
    "camera": {"device_index": 0, "resolution": "native", "width": 640, "height": 480, "exposure": "auto"},
    "interaction": {
        "dwell_hold_ms": 1200,
        "blink_click_ms": 250,
        "blink_release_ms": 300,
        "smoothing_percent": 40,
        "dwell_radius_px": 50,
        "hold_enabled": True,
        "blink_max_ms": 2000,
        "freeze_after_blink_ms": 150,
        "hold_timeout_s": 30,
        "one_euro_beta": 0.002,
        "one_euro_d_cutoff": 1.0,
        "feature_median_window": 3,
        "deadzone_px": 4,
        # radio magnético de los botones flotantes (px físicos desde el borde del botón; 0 = desactivado)
        "snap_radius_px": 60,
    },
    # eye: con qué ojo se sigue la mirada y se detectan los parpadeos. "auto" = en cada calibración se elige
    # el de menor temblor (entre los de error parecido); "right"/"left"/"both" lo fuerzan.
    # selected_eye: el ojo de la última calibración aceptada (lo escribe la aplicación; null = ninguna)
    "tracking": {"eye": "auto", "selected_eye": None, "gaze_model": "iris"},
    "ui": {"theme": "default", "font_scale": 1.0},
    # botones flotantes (solo con el control del cursor activo): lado de la pantalla y tamaño en px lógicos
    "overlay": {"enabled": True, "position": "right", "button_size": 180, "buttons": {}},
    "tts": {"enabled": False},
}

EYE_MODES = ("auto", "both", "right", "left")  # como calibration.EYE_MODES (sin importar numpy aquí)
SELECTED_EYES = ("both", "right", "left")
# Mirada: "iris" (solo MediaPipe) o una red de neural_gaze.GAZE_MODELS combinada con el iris (híbrido)
GAZE_MODEL_CHOICES = ("iris", "mobileone_s0", "resnet18", "resnet34")


def _get_project_root():
    """Devuelve la ruta de la carpeta del proyecto (donde está este archivo)."""
    return data_dir()  # en el .exe: la carpeta del .exe (config.json vive ahí)


def _deep_merge(base, updates):
    """Mezcla en profundidad updates sobre base (retorna nuevo dict)."""
    result = base.copy()
    for key, value in updates.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _clamp(value, min_val, max_val):
    """Restringe un valor numérico al rango [min_val, max_val]."""
    if value < min_val:
        return min_val
    if value > max_val:
        return max_val
    return value


CAMERA_RESOLUTION_MODES = ("native", "fixed")
CAMERA_EXPOSURE_MODES = ("auto", "fast")  # como eye_tracker.CAMERA_EXPOSURES


def camera_exposure(config) -> str:
    """Exposición de la cámara de config.json (camera.exposure): "auto" o "fast"."""
    cam = config.get("camera", {}) if isinstance(config, dict) else {}
    value = cam.get("exposure", "auto") if isinstance(cam, dict) else "auto"
    return value if value in CAMERA_EXPOSURE_MODES else "auto"


def camera_size(config) -> Optional[Tuple[int, int]]:
    """Tamaño que se pide a la cámara: None = resolución nativa; (ancho, alto) con resolution "fixed"."""
    cam = config.get("camera", {}) if isinstance(config, dict) else {}
    if not isinstance(cam, dict) or cam.get("resolution", "native") != "fixed":
        return None
    return int(cam.get("width", 640)), int(cam.get("height", 480))


def _validate_config(config):
    """Valida y corrige valores según rangos especificados."""
    validated = json.loads(json.dumps(config))  # copia profunda

    # camera.device_index
    if not isinstance(validated.get("camera", {}), dict):
        validated["camera"] = {}
    try:
        device_index = int(validated["camera"].get("device_index", 0))
        if device_index < 0:
            device_index = 0
        validated["camera"]["device_index"] = device_index
    except Exception:
        validated["camera"]["device_index"] = DEFAULT_CONFIG["camera"]["device_index"]
    # resolución: nativa (por defecto) o fija width x height (si no la entrega a ~24 fps, eye_tracker cae a 640x480)
    if validated["camera"].get("resolution") not in CAMERA_RESOLUTION_MODES:
        validated["camera"]["resolution"] = DEFAULT_CONFIG["camera"]["resolution"]
    if validated["camera"].get("exposure") not in CAMERA_EXPOSURE_MODES:
        validated["camera"]["exposure"] = DEFAULT_CONFIG["camera"]["exposure"]
    for key, lo, hi in (("width", 160, 1920), ("height", 120, 1080)):
        try:
            validated["camera"][key] = _clamp(int(validated["camera"].get(key, DEFAULT_CONFIG["camera"][key])), lo, hi)
        except Exception:
            validated["camera"][key] = DEFAULT_CONFIG["camera"][key]

    # interaction
    if not isinstance(validated.get("interaction", {}), dict):
        validated["interaction"] = {}
    inter = validated["interaction"]

    # dwell_hold_ms 800-3000
    try:
        inter["dwell_hold_ms"] = _clamp(int(inter.get("dwell_hold_ms", DEFAULT_CONFIG["interaction"]["dwell_hold_ms"])), 800, 3000)
    except Exception:
        inter["dwell_hold_ms"] = DEFAULT_CONFIG["interaction"]["dwell_hold_ms"]

    # blink_click_ms 150-500
    try:
        inter["blink_click_ms"] = _clamp(int(inter.get("blink_click_ms", DEFAULT_CONFIG["interaction"]["blink_click_ms"])), 150, 500)
    except Exception:
        inter["blink_click_ms"] = DEFAULT_CONFIG["interaction"]["blink_click_ms"]

    # blink_release_ms 150-500
    try:
        inter["blink_release_ms"] = _clamp(int(inter.get("blink_release_ms", DEFAULT_CONFIG["interaction"]["blink_release_ms"])), 150, 500)
    except Exception:
        inter["blink_release_ms"] = DEFAULT_CONFIG["interaction"]["blink_release_ms"]

    # smoothing_percent 0-100
    try:
        inter["smoothing_percent"] = _clamp(int(inter.get("smoothing_percent", DEFAULT_CONFIG["interaction"]["smoothing_percent"])), 0, 100)
    except Exception:
        inter["smoothing_percent"] = DEFAULT_CONFIG["interaction"]["smoothing_percent"]

    # claves avanzadas (sin UI): enteros con clamp
    for key, lo, hi in (
        ("dwell_radius_px", 20, 150),
        ("blink_max_ms", 500, 5000),
        ("freeze_after_blink_ms", 0, 500),
        ("hold_timeout_s", 5, 120),
        ("feature_median_window", 1, 9),
        ("deadzone_px", 0, 50),
        ("snap_radius_px", 0, 300),
    ):
        try:
            inter[key] = _clamp(int(inter.get(key, DEFAULT_CONFIG["interaction"][key])), lo, hi)
        except Exception:
            inter[key] = DEFAULT_CONFIG["interaction"][key]
    # filtro One Euro (flotantes con clamp; no finitos -> valor por defecto)
    for key, lo, hi in (("one_euro_beta", 0.0, 1.0), ("one_euro_d_cutoff", 0.1, 10.0)):
        try:
            value = float(inter.get(key, DEFAULT_CONFIG["interaction"][key]))
            if value != value or value in (float("inf"), float("-inf")):
                raise ValueError(key)
            inter[key] = _clamp(value, lo, hi)
        except Exception:
            inter[key] = DEFAULT_CONFIG["interaction"][key]
    if not isinstance(inter.get("hold_enabled"), bool):
        inter["hold_enabled"] = bool(DEFAULT_CONFIG["interaction"]["hold_enabled"])

    validated["interaction"] = inter

    # tracking: modo del ojo y ojo de la última calibración
    if not isinstance(validated.get("tracking", {}), dict):
        validated["tracking"] = {}
    tracking = validated.setdefault("tracking", {})
    if tracking.get("eye") not in EYE_MODES:
        tracking["eye"] = DEFAULT_CONFIG["tracking"]["eye"]
    if tracking.get("selected_eye") not in SELECTED_EYES:
        tracking["selected_eye"] = None
    if tracking.get("gaze_model") not in GAZE_MODEL_CHOICES:
        tracking["gaze_model"] = DEFAULT_CONFIG["tracking"]["gaze_model"]

    # ui
    if not isinstance(validated.get("ui", {}), dict):
        validated["ui"] = {}
    ui = validated["ui"]
    if not isinstance(ui.get("theme"), str) or not ui.get("theme"):
        ui["theme"] = DEFAULT_CONFIG["ui"]["theme"]
    # font_scale 0.8-2.0
    try:
        ui["font_scale"] = _clamp(float(ui.get("font_scale", DEFAULT_CONFIG["ui"]["font_scale"])), 0.8, 2.0)
    except Exception:
        ui["font_scale"] = DEFAULT_CONFIG["ui"]["font_scale"]
    validated["ui"] = ui

    # overlay
    if not isinstance(validated.get("overlay", {}), dict):
        validated["overlay"] = {}
    overlay = validated["overlay"]
    if not isinstance(overlay.get("buttons"), dict):
        overlay["buttons"] = {}
    if not isinstance(overlay.get("enabled"), bool):
        overlay["enabled"] = DEFAULT_CONFIG["overlay"]["enabled"]
    if overlay.get("position") not in ("right", "left"):
        overlay["position"] = DEFAULT_CONFIG["overlay"]["position"]
    try:
        overlay["button_size"] = _clamp(int(overlay.get("button_size", DEFAULT_CONFIG["overlay"]["button_size"])), 96, 320)
    except Exception:
        overlay["button_size"] = DEFAULT_CONFIG["overlay"]["button_size"]

    # tts
    if not isinstance(validated.get("tts", {}), dict):
        validated["tts"] = {}
    if not isinstance(validated["tts"].get("enabled"), bool):
        validated["tts"]["enabled"] = bool(DEFAULT_CONFIG["tts"]["enabled"])

    validated["version"] = DEFAULT_CONFIG["version"]
    return validated


def load_config(path=None):
    """Carga la configuración desde path o config.json del proyecto.

    Si el archivo no existe o es inválido, devuelve valores por defecto
    y registra un aviso con logging (no lanza excepción).
    """
    if path is None:
        path = os.path.join(_get_project_root(), "config.json")

    try:
        if not os.path.exists(path):
            logger.warning("No se encontró config.json en %s; usando valores por defecto", path)
            return json.loads(json.dumps(DEFAULT_CONFIG))

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        merged = _deep_merge(json.loads(json.dumps(DEFAULT_CONFIG)), data)
        validated = _validate_config(merged)
        return validated
    except json.JSONDecodeError as e:
        logger.warning("JSON inválido en %s (%s); usando valores por defecto", path, e)
        return json.loads(json.dumps(DEFAULT_CONFIG))
    except Exception as e:
        logger.warning("Error al cargar config de %s (%s); usando valores por defecto", path, e)
        return json.loads(json.dumps(DEFAULT_CONFIG))


def save_config(config, path=None):
    """Valida y guarda la configuración de forma atómica."""
    if path is None:
        path = os.path.join(_get_project_root(), "config.json")

    validated = _validate_config(config)

    dir_path = os.path.dirname(path) or "."
    fd, tmp_path = tempfile.mkstemp(prefix=".config_tmp_", dir=dir_path)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(validated, f, indent=2, ensure_ascii=False)
    except Exception:
        os.close(fd)
        raise

    os.replace(tmp_path, path)
    return validated


def eye_mode(config) -> str:
    """Modo del ojo de config.json (tracking.eye): auto, both, right o left."""
    tracking = config.get("tracking", {}) if isinstance(config, dict) else {}
    mode = tracking.get("eye", "auto") if isinstance(tracking, dict) else "auto"
    return mode if mode in EYE_MODES else "auto"


def gaze_model(config):
    """Red de mirada de config.json (tracking.gaze_model) o None si es "iris" (solo MediaPipe)."""
    tracking = config.get("tracking", {}) if isinstance(config, dict) else {}
    value = tracking.get("gaze_model", "iris") if isinstance(tracking, dict) else "iris"
    return value if value in GAZE_MODEL_CHOICES and value != "iris" else None


def save_selected_eye(eye, path=None):
    """Escribe tracking.selected_eye en config.json leyendo el archivo actual, para no guardar de paso otros
    cambios de la ventana que el usuario no haya guardado. Devuelve la configuración guardada."""
    if eye not in SELECTED_EYES:
        raise ValueError(f"ojo desconocido: {eye}")
    if path is None:
        path = os.path.join(_get_project_root(), "config.json")
    if os.path.exists(path):  # un config.json ilegible no se pisa con los valores por defecto
        with open(path, "r", encoding="utf-8") as f:
            json.load(f)
    config = load_config(path)
    config.setdefault("tracking", {})["selected_eye"] = eye
    return save_config(config, path)
