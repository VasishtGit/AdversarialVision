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
    severity: 1 to 5
    """
    amount = [0.02, 0.05, 0.09, 0.15, 0.23][severity - 1]
    out = x.copy()
    num_salt = np.ceil(amount * x.size * 0.5)
    num_pepper = np.ceil(amount * x.size * 0.5)

    # Salt (1.0)
    coords = [np.random.randint(0, i, int(num_salt)) for i in x.shape]
    out[tuple(coords)] = 1.0

    # Pepper (0.0)
    coords = [np.random.randint(0, i, int(num_pepper)) for i in x.shape]
    out[tuple(coords)] = 0.0

    return out.astype(np.float32)
