"""Módulo para cargar y guardar config.json (valores por defecto y validación de rangos)."""

import json
import logging
import os
import tempfile

# TODO(etapa 1)

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = {
    "version": 1,
    "camera": {"device_index": 0},
    "interaction": {
        "dwell_hold_ms": 1200,
        "blink_click_ms": 250,
        "blink_release_ms": 300,
        "smoothing_percent": 40,
    },
    "ui": {"theme": "default", "font_scale": 1.0},
    "overlay": {"buttons": {}},
    "tts": {"enabled": False},
}


def _get_project_root():
    """Devuelve la ruta de la carpeta del proyecto (donde está este archivo)."""
    return os.path.dirname(os.path.abspath(__file__))


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

    validated["interaction"] = inter

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
    if not isinstance(validated["overlay"].get("buttons"), dict):
        validated["overlay"]["buttons"] = {}

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
