"""Noise corruptions: Gaussian noise and Salt-and-Pepper noise."""

import numpy as np


def gaussian_noise(x: np.ndarray, severity: int = 1) -> np.ndarray:
    """
    Adds Gaussian noise to image x in [0, 1].
    severity: 1 to 5
    """
    scale = [0.04, 0.08, 0.12, 0.18, 0.26][severity - 1]
    noise = np.random.normal(0, scale, x.shape)
    return np.clip(x + noise, 0.0, 1.0).astype(np.float32)


def salt_and_pepper_noise(x: np.ndarray, severity: int = 1) -> np.ndarray:
    """
    Adds Salt-and-Pepper impulse noise to image x in [0, 1].
    Operates per-pixel across all channels.
    severity: 1 to 5
    """
    amount = [0.02, 0.05, 0.09, 0.15, 0.23][severity - 1]
    out = x.copy()
    h, w = x.shape[:2]
    num_pixels = int(amount * h * w)

    # Salt (1.0)
    salt_count = int(num_pixels * 0.5)
    if salt_count > 0:
        rows_s = np.random.randint(0, h, salt_count)
        cols_s = np.random.randint(0, w, salt_count)
        out[rows_s, cols_s, :] = 1.0

    # Pepper (0.0)
    pepper_count = int(num_pixels * 0.5)
    if pepper_count > 0:
        rows_p = np.random.randint(0, h, pepper_count)
        cols_p = np.random.randint(0, w, pepper_count)
        out[rows_p, cols_p, :] = 0.0

    return out.astype(np.float32)
