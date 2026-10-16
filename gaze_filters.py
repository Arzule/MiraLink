"""Filtros puros para la mirada: One Euro, mediana corta de rasgos y zona muerta.

Sin reloj propio: reciben los timestamps (segundos) de TrackingFrame, nunca asumen una tasa de
cuadros fija. Solo numpy.

One Euro (Casiez, Roussel y Vogel, CHI 2012): paso bajo exponencial cuya frecuencia de corte crece
con la velocidad: cutoff = min_cutoff + beta * |velocidad filtrada|. Quieto -> corte bajo (estable);
en un movimiento rápido -> corte alto (poco retraso). La velocidad (px/s) se filtra con d_cutoff.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Optional, Tuple

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


def apply_deadzone(shown: Optional[Tuple[float, float]], target: Tuple[float, float], radius: float) -> Tuple[float, float]:
    """Zona muerta: si el destino está a menos de `radius` px de la posición mostrada, no se mueve."""
    if shown is None or radius <= 0.0:
        return float(target[0]), float(target[1])
    if math.hypot(target[0] - shown[0], target[1] - shown[1]) < radius:
        return shown
    return float(target[0]), float(target[1])
