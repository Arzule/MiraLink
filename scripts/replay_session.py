"""Evalúa algoritmos de mirada con una sesión grabada, SIN cámara.

Uso:
  .\\venv\\Scripts\\python.exe scripts\\replay_session.py sessions\\session_<fecha>.npz
  .\\venv\\Scripts\\python.exe scripts\\replay_session.py --synthetic [--seed 0]

Opciones: --distance-cm 60 (distancia de visión para convertir px a grados).
La sesión se graba con: .\\venv\\Scripts\\python.exe main.py --record-session
"""

from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from gaze_session import (  # noqa: E402
    default_variants,
    evaluate,
    fit_label,
    blink_report,
    eye_report,
    format_blink_report,
    format_eye_report,
    format_head_motion_report,
    head_motion_report,
    format_posture_report,
    format_table,
    posture_report,
    load_session,
    model_variants,
    synthesize_session,
)


def describe(session) -> str:
    meta = session.meta
    duration = float(session.t[-1] - session.t[0]) if len(session) > 1 else 0.0
    fps = (len(session) - 1) / duration if duration > 0 else 0.0
    valid = float(session.valid_mask().mean()) * 100 if len(session) else 0.0
    parts = [
        f"Sesión: {len(session)} cuadros, {duration:.1f} s, {fps:.1f} fps medidos con timestamps, "
        f"{valid:.0f} % cuadros válidos",
        f"Pantalla: {session.width}x{session.height} px físicos"
        + (f", {meta['physical_mm'][0]:.0f}x{meta['physical_mm'][1]:.0f} mm" if meta.get("physical_mm") else ""),
    ]
    cam = meta.get("camera") or {}
    if cam.get("info") or cam.get("image_size"):
        size = cam.get("image_size")
        text = cam.get("info") or (f"{size[0]}x{size[1]}" if size else "?")
        requested = (meta.get("config") or {}).get("camera", {})
        if requested.get("width"):
            text += f" (pedido {requested['width']}x{requested['height']})"
        if cam.get("fps_median"):
            text += f", {cam['fps_median']:.1f} fps (mediana medida en el tracker)"
        parts.append(f"Cámara: {text}")
    if meta.get("synthetic"):
        parts.append("ATENCIÓN: sesión SINTÉTICA (modelo simulado), no son datos reales.")
    return "\n".join(parts)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Compara variantes de mapeo y filtro con una sesión grabada.")
    parser.add_argument("session", nargs="?", help="archivo sessions/session_<fecha>.npz")
    parser.add_argument("--synthetic", action="store_true", help="usar una sesión sintética")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--distance-cm", type=float, default=60.0)
    parser.add_argument("--quick", action="store_true", help="omitir la comparativa de modelos")
    parser.add_argument("--head-motion", action="store_true", help="con --synthetic: incluir la fase de cabeza")
    args = parser.parse_args(argv)
    if not args.session and not args.synthetic:
        parser.error("indica un archivo de sesión o --synthetic")
    try:
        session = synthesize_session(args.seed, head_motion=args.head_motion) if args.synthetic else load_session(args.session)
    except ValueError as e:
        print(e)
        return 1
    print(describe(session))
    try:
        print(format_posture_report(posture_report(session)))
    except ValueError as e:
        print(e)
    try:
        print(format_eye_report(eye_report(session)))
    except ValueError as e:
        print(e)
    try:
        print(format_blink_report(blink_report(session)))
    except ValueError as e:
        print(e)
    try:
        print(format_head_motion_report(head_motion_report(session, args.distance_cm * 10.0)))
    except ValueError as e:
        print(e)
    print()
    distance_mm = args.distance_cm * 10.0
    try:
        rows = evaluate(session, default_variants(), distance_mm=distance_mm)
        model_rows = [] if args.quick else evaluate(session, model_variants(), distance_mm=distance_mm)
    except ValueError as e:
        print(e)
        return 1
    print("Antes / después y filtros (parte B)")
    print(format_table(rows))
    if model_rows:
        print()
        print("Modelos (parte C), ordenados por desvío en validación; todos con el filtro nuevo")
        model_rows.sort(key=lambda r: (r["bias_px"] != r["bias_px"], r["bias_px"]))
        print(format_table(model_rows))
        print(f"Ajuste por defecto de la aplicación: {fit_label({})}")
    print()
    print("Desvío: |cursor medio - objetivo| por punto; jitter: desviación típica 2D del cursor en la fijación;")
    print("latencia: tiempo hasta el 90 % de un salto de 400 px sin ruido; LOO: validación cruzada en calibración.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
