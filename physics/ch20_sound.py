"""Capitulo 20. Ondas sonoras."""
from __future__ import annotations

import numpy as np

C = 343.0
F0 = 440.0
VS = 120.0


def doppler_analytic(f0: float = F0, vs: float = VS, vo: float = 0.0, c: float = C) -> float:
    """Fuente que se acerca (vs > 0) y/o observador que se acerca (vo > 0)."""
    return f0 * (c + vo) / (c - vs)


def doppler_numeric(f0: float = F0, vs: float = VS, vo: float = 0.0, c: float = C, n_pulses: int = 400) -> float:
    """Emite pulsos cada 1/f0 desde una fuente en movimiento y mide el intervalo entre llegadas al observador.

    El observador esta en x = 0 (fijo o en movimiento con velocidad vo hacia la fuente) y la fuente parte de
    x0 = 1000 m moviendose hacia el con velocidad vs. El tiempo de llegada resuelve x_f(te) - c (t - te) = x_o(t).
    """
    x0 = 1000.0
    te = np.arange(n_pulses) / f0
    xs = x0 - vs * te
    # observador: x_o(t) = vo * t (se mueve hacia +x, hacia la fuente); llegada: x_s - c (t - te) = vo t
    # => t = (xs + c te) / (c + vo)
    t_arr = (xs + c * te) / (c + vo)
    return float(1.0 / np.mean(np.diff(t_arr)))


# ----------------------------------------------------------------------- pulsaciones
F1, F2 = 440.0, 444.0


def beat_frequency_analytic(f1: float = F1, f2: float = F2) -> float:
    return abs(f2 - f1)


def beat_signal(fs: float = 16000.0, duration: float = 2.0, f1: float = F1, f2: float = F2):
    t = np.arange(0, duration, 1 / fs)
    return t, np.sin(2 * np.pi * f1 * t) + np.sin(2 * np.pi * f2 * t)


def beat_frequency_numeric(fs: float = 16000.0, duration: float = 4.0) -> float:
    """Frecuencia de pulsacion: separacion de las dos frecuencias dominantes del espectro (FFT con interpolacion parabolica)."""
    t, s = beat_signal(fs, duration)
    spec = np.abs(np.fft.rfft(s * np.hanning(len(s))))
    freqs = np.fft.rfftfreq(len(s), 1 / fs)
    peaks = [i for i in range(1, len(spec) - 1) if spec[i] > spec[i - 1] and spec[i] >= spec[i + 1] and spec[i] > 0.2 * spec.max()]
    def refine(i):
        a, b, c = np.log(spec[i - 1]), np.log(spec[i]), np.log(spec[i + 1])
        return freqs[i] + 0.5 * (a - c) / (a - 2 * b + c) * (freqs[1] - freqs[0])
    f = sorted(refine(i) for i in peaks)[:2]
    return float(f[1] - f[0])


def envelope(signal: np.ndarray) -> np.ndarray:
    """Envolvente |señal analitica| (transformada de Hilbert por FFT)."""
    n = len(signal)
    spec = np.fft.fft(signal)
    h = np.zeros(n)
    h[0] = 1.0
    h[1:n // 2] = 2.0
    h[n // 2] = 1.0
    return np.abs(np.fft.ifft(spec * h))


def beat_envelope_frequency_numeric(fs: float = 16000.0, duration: float = 2.0) -> float:
    """Frecuencia de pulsacion en el dominio del tiempo: ritmo de los maximos de la envolvente."""
    t, s = beat_signal(fs, duration)
    env = envelope(s)
    peaks = [i for i in range(1, len(env) - 1) if env[i] > env[i - 1] and env[i] >= env[i + 1] and env[i] > 1.8]
    return float(1.0 / np.mean(np.diff(t[peaks])))


# ------------------------------------------------------------------------- tubos
LEN = 0.75


def pipe_frequencies_analytic(kind: str, count: int = 4, length: float = LEN, c: float = C):
    if kind == "open":
        return np.array([n * c / (2 * length) for n in range(1, count + 1)])
    return np.array([(2 * n - 1) * c / (4 * length) for n in range(1, count + 1)])


def pipe_frequencies_numeric(kind: str, count: int = 4, length: float = LEN, c: float = C, n_points: int = 600):
    """Autovalores de p'' = -k^2 p con condiciones: abierto-abierto (p = 0 en ambos) o cerrado-abierto (p' = 0 | p = 0)."""
    dx = length / (n_points + 0.5 if kind == "closed" else n_points + 1)
    n = n_points
    a = (np.diag(np.full(n, 2.0)) + np.diag(np.full(n - 1, -1.0), 1) + np.diag(np.full(n - 1, -1.0), -1)) / dx**2
    if kind == "closed":
        a[0, 0] = 1.0 / dx**2  # extremo cerrado: derivada de presion nula (condicion de Neumann de segundo orden)
    w2 = np.linalg.eigvalsh(a)
    return c * np.sqrt(w2[:count]) / (2 * np.pi)
