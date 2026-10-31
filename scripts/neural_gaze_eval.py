r"""Compara, SIN cámara, el método actual (iris de MediaPipe) con las redes de mirada MobileGaze.

Uso:
  .\venv\Scripts\python.exe scripts\neural_gaze_eval.py [sesión.npz ...] [--models mobileone_s0 resnet18 resnet34]
  (por defecto, todas las sesiones de sessions/ grabadas con recortes de la cara)

Cada red estima (giro, inclinación) de la mirada en cada cuadro con el recorte grabado. Esos dos valores
sustituyen a los del iris (columnas iris_along/iris_perp de los dos ojos) y pasan por el MISMO mapeo de la
aplicación (DEFAULT_FIT, calibración de 9 puntos) y el mismo filtro del cursor. Métricas por variante:
- validación: desvío medio en los 5 puntos nuevos (px);
- cabeza: mediana del error en la fase de movimiento de cabeza (px), lo que se quiere reducir;
- jitter: jitter del cursor en la validación (px);
- S/R h y v: rango entre puntos de calibración / ruido dentro de cada punto (más es mejor);
- ms/cuadro: inferencia de la red en esta CPU (además de MediaPipe; para 30 fps hay ~33 ms por cuadro).
Criterio para integrar una red (AGENTS.md): reducir el error de cabeza sin empeorar la validación ni bajar
de ~25 fps. Las predicciones se guardan junto a la sesión (<sesión>.<modelo>.gaze.npy) para no repetirlas.
"""

from __future__ import annotations

import argparse
import dataclasses
import glob
import os
import sys
from typing import Dict, List, Optional, Sequence

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import gaze_session as gs  # noqa: E402
import neural_gaze as ng  # noqa: E402
from head_compensation_eval import Prepared  # noqa: E402

NO_COMPENSATION = (0.0, 0.0, 0.0, 0.0)
BATCH = 32


def image_size(session) -> Optional[tuple]:
    size = (session.meta.get("camera") or {}).get("image_size")
    return (int(size[0]), int(size[1])) if size else None


def predict_session(session, net, log=print) -> np.ndarray:
    """(n, 2) giro e inclinación de la red por cuadro; NaN sin recorte o sin landmarks."""
    n = len(session)
    out = np.full((n, 2), np.nan)
    size = image_size(session)
    if not session.has_face_crops or session.landmarks is None or size is None:
        return out
    idx: List[int] = []
    faces: List[np.ndarray] = []

    def flush():
        if faces:
            out[idx] = net.predict(faces)
            idx.clear()
            faces.clear()

    for i in range(n):
        data = session.face_crop_bytes(i)
        box = session.face_box[i]
        if data is None or not np.isfinite(box).all():
            continue
        face = ng.face_input_from_record(ng.decode_jpeg(data), box, session.landmarks[i], size, net.input_size)
        if face is None:
            continue
        idx.append(i)
        faces.append(face)
        if len(faces) >= BATCH:
            flush()
            if i % (BATCH * 20) < BATCH:
                log(f"    {i}/{n} cuadros")
    flush()
    return out


def cached_prediction(path: str, session, model: str, net_factory, log=print) -> np.ndarray:
    cache = f"{os.path.splitext(path)[0]}.{model}.gaze.npy"
    if os.path.exists(cache):
        pred = np.load(cache)
        if pred.shape == (len(session), 2):
            return pred
    pred = predict_session(session, net_factory(), log)
    np.save(cache, pred)
    return pred


def with_gaze(session, gaze: np.ndarray):
    """Copia de la sesión con (giro, inclinación) de la red en lugar del iris de los dos ojos."""
    f = session.features.copy()
    ok = np.isfinite(gaze).all(axis=1)
    f[:, 0] = f[:, 2] = gaze[:, 0]
    f[:, 1] = f[:, 3] = gaze[:, 1]
    f[~ok] = np.nan  # sin estimación de la red el cuadro no es válido
    return dataclasses.replace(session, features=f)


def signal_to_noise(session) -> tuple:
    """Por eje: rango de las medianas de los puntos de calibración / mediana de la desviación dentro del punto."""
    x, _y, g = gs.robust_samples(session, "calibration")
    res = []
    for cols in ((0, 2), (1, 3)):
        v = x[:, cols].mean(axis=1)
        meds = [np.median(v[g == k]) for k in np.unique(g)]
        noise = np.median([np.std(v[g == k]) for k in np.unique(g)])
        res.append(float((max(meds) - min(meds)) / max(noise, 1e-9)))
    return tuple(res)


def metrics(session, name: str) -> Dict[str, object]:
    prep = Prepared(session, name)
    has_head = bool((session.stage == gs.STAGE_CODES["head"]).any()) and len(prep.head_frames()) > 0
    snr_h, snr_v = signal_to_noise(session)
    return {
        "validation_px": prep.validation_error(NO_COMPENSATION),
        "head_px": prep.head_error(NO_COMPENSATION) if has_head else float("nan"),
        "jitter_px": prep.cursor_jitter(NO_COMPENSATION),
        "snr_h": snr_h,
        "snr_v": snr_v,
        "valid": float(session.valid_mask().mean()),
    }


def usable(session) -> Optional[str]:
    if not session.has_face_crops:
        return "sin recortes de la cara (grabada antes de la red neuronal)"
    if session.landmarks is None or image_size(session) is None:
        return "sin landmarks o sin tamaño de imagen"
    if not session.has_validation or not session.meta.get("physical_mm"):
        return "sin validación o sin tamaño del monitor"
    return None


def evaluate_session(path: str, models: Sequence[str], log=print) -> Dict[str, Dict[str, object]]:
    s = gs.load_session(path)
    rows = {"iris (actual)": metrics(s, "iris")}
    for model in models:
        info = ng.GAZE_MODELS[model]
        log(f"  {model}: estimando la mirada...")
        pred = cached_prediction(path, s, model, lambda: ng.GazeNet(info.path), log)
        rows[model] = metrics(with_gaze(s, pred), model)
    return rows


def model_timings(models: Sequence[str]) -> Dict[str, float]:
    face = np.random.default_rng(0).integers(0, 255, (448, 448, 3), dtype=np.uint8)
    return {m: ng.GazeNet(ng.GAZE_MODELS[m].path).time_per_frame_ms(face) for m in models}


def format_rows(rows: Dict[str, Dict[str, object]], timings: Dict[str, float]) -> str:
    lines = [f"  {'variante':<15}{'validación':>11}{'cabeza':>9}{'jitter':>9}{'S/R h':>8}{'S/R v':>8}{'válidos':>9}{'ms/cuadro':>11}"]
    for name, r in rows.items():
        ms = timings.get(name)
        lines.append(
            f"  {name:<15}{r['validation_px']:>9.0f}px{r['head_px']:>7.0f}px{r['jitter_px']:>7.1f}px"
            f"{r['snr_h']:>8.1f}{r['snr_v']:>8.1f}{100 * r['valid']:>8.0f}%{(f'{ms:.0f}' if ms else '-'):>11}"
        )
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Compara el iris de MediaPipe con redes de mirada en sesiones grabadas.")
    parser.add_argument("sessions", nargs="*")
    parser.add_argument("--models", nargs="+", default=list(ng.GAZE_MODELS), choices=list(ng.GAZE_MODELS))
    args = parser.parse_args(argv)
    missing = [m for m in args.models if not os.path.exists(ng.GAZE_MODELS[m].path)]
    if missing:
        print(f"Faltan modelos ({', '.join(missing)}): .\\venv\\Scripts\\python.exe scripts\\download_gaze_models.py")
        return 1
    paths = args.sessions or sorted(glob.glob(os.path.join(gs.default_sessions_dir(), "*.npz")))
    timings = None
    done = 0
    for path in paths:
        reason = usable(gs.load_session(path))
        if reason:
            print(f"(se omite {os.path.basename(path)}: {reason})")
            continue
        if timings is None:
            timings = model_timings(args.models)
        print(f"\n== {os.path.basename(path)}")
        rows = evaluate_session(path, args.models)
        print(format_rows(rows, timings))
        done += 1
    if not done:
        print("No hay sesiones con recortes de la cara. Graba una con: .\\venv\\Scripts\\python.exe main.py --record-session")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
