"""Lighting corruptions: Brightness, Contrast, and Gamma."""

import numpy as np


def brightness(x: np.ndarray, severity: int = 1) -> np.ndarray:
    """
    Shifts image brightness.
    severity: 1 to 5
    """
    deltas = [0.08, 0.16, 0.24, 0.32, 0.42]
    delta = deltas[severity - 1]
    return np.clip(x + delta, 0.0, 1.0).astype(np.float32)


def contrast(x: np.ndarray, severity: int = 1) -> np.ndarray:
    """
    Scales image contrast relative to mean intensity.
    severity: 1 to 5
    """
    factors = [0.75, 0.55, 0.40, 0.25, 0.12]
    factor = factors[severity - 1]
    mean = np.mean(x, axis=(0, 1), keepdims=True)
    return np.clip((x - mean) * factor + mean, 0.0, 1.0).astype(np.float32)


def gamma(x: np.ndarray, severity: int = 1) -> np.ndarray:
    """
    Applies non-linear gamma adjustment: x^(gamma).
    severity: 1 to 5
    """
    gammas = [1.4, 1.8, 2.2, 2.8, 3.5]
    g = gammas[severity - 1]
    return np.clip(np.power(x, g), 0.0, 1.0).astype(np.float32)
