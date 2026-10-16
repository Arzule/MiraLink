"""Filtros puros para la mirada: One Euro, mediana corta de rasgos, zona muerta y radio magnético.

Sin reloj propio: reciben los timestamps (segundos) de TrackingFrame, nunca asumen una tasa de
cuadros fija. Solo numpy.

One Euro (Casiez, Roussel y Vogel, CHI 2012): paso bajo exponencial cuya frecuencia de corte crece
con la velocidad: cutoff = min_cutoff + beta * |velocidad filtrada|. Quieto -> corte bajo (estable);
en un movimiento rápido -> corte alto (poco retraso). La velocidad (px/s) se filtra con d_cutoff.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Optional, Sequence, Tuple

import numpy as np

# smoothing_percent (slider "0 % Ágil - 100 % Estable") -> min_cutoff del One Euro, escala logarítmica
SMOOTHING_MAX_CUTOFF_HZ = 5.0  # 1 %: casi sin suavizado
SMOOTHING_MIN_CUTOFF_HZ = 0.1  # 100 %: muy estable cuando la mirada está quieta


def smoothing_to_min_cutoff(percent: float) -> Optional[float]:
    """min_cutoff (Hz) para un smoothing_percent. 0 % -> None (sin filtro: el cursor sigue la mirada cruda).

    1..100 % recorre 5 Hz -> 0.1 Hz en escala logarítmica: más % = corte más bajo = más estable.
    """
    p = min(max(float(percent), 0.0), 100.0)
    if p <= 0.0:
        return None
    ratio = SMOOTHING_MIN_CUTOFF_HZ / SMOOTHING_MAX_CUTOFF_HZ
    return SMOOTHING_MAX_CUTOFF_HZ * ratio ** (p / 100.0)


def _alpha(cutoff: float, dt: float) -> float:
    """Factor del paso bajo exponencial para una frecuencia de corte y un intervalo."""
    tau = 1.0 / (2.0 * math.pi * max(cutoff, 1e-6))
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    """One Euro sobre un vector (p. ej. x, y del cursor). La velocidad usa la norma del vector, así
    los dos ejes comparten la frecuencia de corte y el cursor no se deforma en diagonal."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.002, d_cutoff: float = 1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.reset()

    def reset(self):
        self._x: Optional[np.ndarray] = None
        self._dx: Optional[np.ndarray] = None
        self._t: Optional[float] = None

    @property
    def value(self) -> Optional[np.ndarray]:
        return None if self._x is None else self._x.copy()

    def __call__(self, x, t: float) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if self._x is None or self._t is None:
            self._x, self._dx, self._t = x.copy(), np.zeros_like(x), float(t)
            return x.copy()
        dt = float(t) - self._t
        if dt <= 0.0:
            return self._x.copy()  # timestamp repetido o hacia atrás: se conserva el valor
        self._t = float(t)
        dx = (x - self._x) / dt
        self._dx = self._dx + _alpha(self.d_cutoff, dt) * (dx - self._dx)
        cutoff = self.min_cutoff + self.beta * float(np.linalg.norm(self._dx))
        self._x = self._x + _alpha(cutoff, dt) * (x - self._x)
        return self._x.copy()


class MedianFilter:
    """Mediana por componente de las últimas `window` muestras (rechaza picos aislados del iris).

    Con menos muestras que la ventana usa las que hay. window=1 la desactiva.
    """

    def __init__(self, window: int = 3):
        self.window = max(1, int(window))
        self._buf: deque = deque()

    def reset(self):
        self._buf.clear()

    def __len__(self) -> int:
        return len(self._buf)

    def __call__(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if self.window <= 1:
            self._buf.clear()
            return x.copy()
        if self._buf and np.shape(self._buf[-1]) != x.shape:
            self._buf.clear()  # cambió la dimensión del vector
        self._buf.append(x.copy())
        while len(self._buf) > self.window:
            self._buf.popleft()
        if len(self._buf) == 1:
            return x.copy()
        return np.median(np.vstack(self._buf), axis=0)


Rect = Tuple[float, float, float, float]  # (x, y, ancho, alto) en las mismas coordenadas que el cursor


def distance_to_rect(pos: Tuple[float, float], rect: Rect) -> float:
    """Distancia de un punto al borde de un rectángulo (0 si está dentro)."""
    x, y, w, h = rect
    dx = max(x - pos[0], 0.0, pos[0] - (x + w))
    dy = max(y - pos[1], 0.0, pos[1] - (y + h))
    return math.hypot(dx, dy)


def snap_strength(dt: float, time_constant_s: float = 0.15) -> float:
    """Fracción de atracción para un intervalo dt (s): 1 - exp(-dt/tau). Independiente de los fps."""
    if dt <= 0.0:
        return 0.0
    if time_constant_s <= 0.0:
        return 1.0
    return 1.0 - math.exp(-dt / time_constant_s)


def magnetic_snap(
    pos: Tuple[float, float], targets: Sequence[Rect], snap_radius_px: float, strength: float
) -> Tuple[float, float]:
    """Radio magnético: si el cursor está a menos de `snap_radius_px` de un control (incluido dentro de
    él), lo acerca una fracción `strength` (0..1) hacia el centro del control más cercano.

    Pura y sin estado: el llamador decide `strength` por cuadro (p. ej. con snap_strength(dt)) para que
    la atracción sea suave y no dependa de la tasa de cuadros. Sin controles, radio <= 0 o fuera del
    radio devuelve la posición sin cambios. Pensada para los botones del overlay (etapa 5).
    """
    if snap_radius_px <= 0.0 or strength <= 0.0 or not targets:
        return float(pos[0]), float(pos[1])
    best = None
    for rect in targets:
        d = distance_to_rect(pos, rect)
        if d <= snap_radius_px:
            cx, cy = rect[0] + rect[2] / 2.0, rect[1] + rect[3] / 2.0
            center_d = math.hypot(cx - pos[0], cy - pos[1])
            if best is None or (d, center_d) < best[0]:
                best = ((d, center_d), (cx, cy))
    if best is None:
        return float(pos[0]), float(pos[1])
    k = min(1.0, float(strength))
    cx, cy = best[1]
    return pos[0] + k * (cx - pos[0]), pos[1] + k * (cy - pos[1])


def apply_deadzone(shown: Optional[Tuple[float, float]], target: Tuple[float, float], radius: float) -> Tuple[float, float]:
    """Zona muerta: si el destino está a menos de `radius` px de la posición mostrada, no se mueve."""
    if shown is None or radius <= 0.0:
        return float(target[0]), float(target[1])
    if math.hypot(target[0] - shown[0], target[1] - shown[1]) < radius:
        return shown
    return float(target[0]), float(target[1])
