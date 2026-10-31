"""Evalúa compensaciones geométricas del movimiento de la cabeza con sesiones grabadas, SIN cámara.

Uso:
  .\\venv\\Scripts\\python.exe scripts\\head_compensation_eval.py [sesión.npz ...]   (por defecto, todas las de sessions/)

Corrección sobre el mapeo de la aplicación (DEFAULT_FIT, solo iris) relativa a la pose de la calibración:
  dx = -cx * G * Δgiro * r - tx * px/mm * desplazamiento_x_mm
  dy = -cy * G * Δinclinación * r - ty * px/mm * desplazamiento_y_mm
  y escala de profundidad opcional p = C + (p0 - C) * r^kd, con r = distancia relativa (separación de ojos)
G = distancia supuesta (60 cm) x px/mm del monitor. La traslación de la matriz de MediaPipe no es métrica
(dice ~20 cm a ~60 cm y casi no cambia al acercarse), así que distancia y desplazamiento salen de la
separación de los ojos en la imagen. Las ganancias ajustadas se entrenan con la fase de cabeza de las OTRAS
sesiones (dejando una fuera) y se miden en la que queda: error en la fase de cabeza, validación y jitter.
Criterio para activarla en la aplicación (AGENTS.md): reducir a la mitad el error de la fase de cabeza sin
empeorar la validación ni subir el jitter más de un 10 %.
"""

from __future__ import annotations

import glob
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import gaze_session as gs  # noqa: E402
from calibration import GazeMapper, pad_features  # noqa: E402

DISTANCE_MM = 600.0
TRANSITION_S = gs.HEAD_TRANSITION_S


class Prepared:
    """Sesión con el mapeo por defecto ajustado y la pose de referencia de la calibración."""

    def __init__(self, session, name: str = "", fit_options=None):
        self.s = session
        self.name = name
        x, y, g = gs.robust_samples(session, "calibration")
        self.val = gs.robust_samples(session, "validation")
        self.mapper = GazeMapper.from_fit_options(fit_options)
        self.mapper.fit(x, y, g, *self.val)
        self.ref = np.median(pad_features(x)[:, [4, 5, 6, 7, 8]], axis=0)  # nose_x, nose_y, iod, yaw, pitch
        self.ppmm = session.width / float(session.meta["physical_mm"][0])
        self.G = DISTANCE_MM * self.ppmm
        self.center = np.array([session.width / 2.0, session.height / 2.0])

    def terms(self, f) -> Tuple[np.ndarray, np.ndarray]:
        """Columnas por cuadro: [giro, inclinación, desplazamiento x, desplazamiento y] en px, y r."""
        f = pad_features(f)
        r = self.ref[2] / f[:, 6]
        mm = gs.TYPICAL_EYE_SPAN_MM / f[:, 6]
        cols = np.column_stack([
            self.G * (f[:, 7] - self.ref[3]) * r,
            self.G * (f[:, 8] - self.ref[4]) * r,
            self.ppmm * (f[:, 4] - self.ref[0]) * mm,
            self.ppmm * (f[:, 5] - self.ref[1]) * mm,
        ])
        return cols, r

    def base(self, f, depth: float) -> np.ndarray:
        _t, r = self.terms(f)
        return self.center + (self.mapper.predict(f) - self.center) * (r[:, None] ** depth)

    def predict(self, f, gains, depth: float = 0.0) -> np.ndarray:
        cx, cy, tx, ty = gains
        t, _r = self.terms(f)
        p = self.base(f, depth)
        p[:, 0] -= cx * t[:, 0] + tx * t[:, 2]
        p[:, 1] -= cy * t[:, 1] + ty * t[:, 3]
        return p

    def head_frames(self) -> np.ndarray:
        s = self.s
        stage = s.stage == gs.STAGE_CODES["head"]
        idx: List[int] = []
        for k in np.unique(s.point[stage]):
            cap = stage & (s.point == k) & (s.phase == 1)
            if cap.any():
                t0 = s.t[np.flatnonzero(cap)[0]]
                idx.extend(np.flatnonzero(cap & s.valid_mask() & (s.t - t0 >= TRANSITION_S)))
        return s.features[np.asarray(idx, dtype=int)]

    def head_error(self, gains, depth: float = 0.0) -> float:
        f = self.head_frames()
        return float(np.median(np.linalg.norm(self.predict(f, gains, depth) - self.center, axis=1)))

    def validation_error(self, gains, depth: float = 0.0) -> float:
        vx, vy, vg = self.val
        p = self.predict(vx, gains, depth)
        return float(np.mean([np.linalg.norm(np.median(p[vg == k], axis=0) - vy[vg == k][0]) for k in np.unique(vg)]))

    def cursor_jitter(self, gains, depth: float = 0.0) -> float:
        prep = self

        class Wrapped:
            def predict_one(self, f):
                p = prep.predict(np.atleast_2d(f), gains, depth)[0]
                return float(p[0]), float(p[1])

        pos = gs.simulate_cursor(self.s, Wrapped())
        return gs.fixation_metrics(self.s, pos, "validation")["jitter_px"]


def fit_gains(train: Sequence[Prepared], use_shift: bool, depth: float) -> Tuple[float, float, float, float]:
    """Mínimos cuadrados de las ganancias con la fase de cabeza de las sesiones de entrenamiento."""
    ax, ay, bx, by = [], [], [], []
    for d in train:
        f = d.head_frames()
        t, _r = d.terms(f)
        err = d.base(f, depth) - d.center
        ax.append(t[:, [0, 2]] if use_shift else t[:, [0]])
        ay.append(t[:, [1, 3]] if use_shift else t[:, [1]])
        bx.append(err[:, 0])
        by.append(err[:, 1])
    gx = np.linalg.lstsq(np.vstack(ax), np.concatenate(bx), rcond=None)[0]
    gy = np.linalg.lstsq(np.vstack(ay), np.concatenate(by), rcond=None)[0]
    return float(gx[0]), float(gy[0]), float(gx[1]) if use_shift else 0.0, float(gy[1]) if use_shift else 0.0


VARIANTS = (
    ("sin compensar", (0.0, 0.0, 0.0, 0.0), False, 0.0),
    ("geometría pura", (1.0, 1.0, 0.0, 0.0), False, 0.0),
    ("giro ajustado", "fit", False, 0.0),
    ("giro + desplazamiento", "fit", True, 0.0),
    ("giro + desplazamiento + profundidad", "fit", True, 1.0),
)


def evaluate(prepared: Sequence[Prepared]) -> Dict[str, List[Dict[str, object]]]:
    """Por variante y sesión: error en la fase de cabeza, validación, jitter y ganancias (dejando una fuera)."""
    out: Dict[str, List[Dict[str, object]]] = {}
    for name, gains, shift, depth in VARIANTS:
        rows = []
        for i, d in enumerate(prepared):
            if gains == "fit":
                others = [e for j, e in enumerate(prepared) if j != i]
                if not others:
                    continue
                g = fit_gains(others, shift, depth)
            else:
                g = gains
            rows.append({
                "session": d.name,
                "head_px": d.head_error(g, depth),
                "validation_px": d.validation_error(g, depth),
                "jitter_px": d.cursor_jitter(g, depth),
                "gains": g,
            })
        out[name] = rows
    return out


def verdict(results: Dict[str, List[Dict[str, object]]]) -> Dict[str, bool]:
    """¿Cumple cada variante el criterio en TODAS las sesiones? (mitad de error de cabeza, validación y jitter)."""
    base = {r["session"]: r for r in results["sin compensar"]}
    ok = {}
    for name, rows in results.items():
        if name == "sin compensar" or not rows:
            continue
        ok[name] = all(
            r["head_px"] <= 0.5 * base[r["session"]]["head_px"]
            and r["validation_px"] <= base[r["session"]]["validation_px"]
            and r["jitter_px"] <= 1.1 * base[r["session"]]["jitter_px"]
            for r in rows
        )
    return ok


def load(paths: Sequence[str]) -> List[Prepared]:
    prepared = []
    for p in paths:
        s = gs.load_session(p)
        if not (s.stage == gs.STAGE_CODES["head"]).any() or not s.has_validation or not s.meta.get("physical_mm"):
            print(f"(se omite {os.path.basename(p)}: sin fase de cabeza, validación o tamaño del monitor)")
            continue
        prepared.append(Prepared(s, os.path.basename(p)))
    return prepared


def main(argv: Optional[Sequence[str]] = None) -> int:
    paths = list(argv if argv is not None else sys.argv[1:]) or sorted(glob.glob(os.path.join(gs.default_sessions_dir(), "*.npz")))
    prepared = load(paths)
    if not prepared:
        print("No hay sesiones con fase de cabeza para evaluar.")
        return 1
    results = evaluate(prepared)
    for name, rows in results.items():
        print(f"\n== {name}")
        for r in rows:
            g = ", ".join(f"{v:+.2f}" for v in r["gains"])
            print(f"  {r['session']}: cabeza {r['head_px']:5.0f} px | validación {r['validation_px']:4.0f} px | "
                  f"jitter {r['jitter_px']:5.1f} px | ganancias ({g})")
    print("\nCumple el criterio en todas las sesiones (mitad de error de cabeza, sin empeorar validación ni jitter > 10 %):")
    for name, ok in verdict(results).items():
        print(f"  {name}: {'SÍ' if ok else 'no'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
