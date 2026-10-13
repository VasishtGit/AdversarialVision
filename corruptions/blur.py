"""Blur corruptions: Gaussian blur and Motion blur."""

import numpy as np
import cv2


def gaussian_blur(x: np.ndarray, severity: int = 1) -> np.ndarray:
    """
    Applies Gaussian blur to image x in [0, 1] with shape (H, W, C).
    severity: 1 to 5
    """
    # Kernel sizes (must be odd) and sigmas
    params = [(3, 0.5), (3, 1.0), (5, 1.2), (5, 1.8), (7, 2.5)]
    ksize, sigma = params[severity - 1]
    blurred = cv2.GaussianBlur(x, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)
    return np.clip(blurred, 0.0, 1.0).astype(np.float32)


def motion_blur(x: np.ndarray, severity: int = 1) -> np.ndarray:
    """
    Applies linear motion blur to image x in [0, 1] with shape (H, W, C).
    severity: 1 to 5
    """
    sizes = [3, 5, 7, 9, 11]
    size = sizes[severity - 1]

    # Generating random angle motion kernel
    kernel = np.zeros((size, size))
    kernel[int((size - 1) / 2), :] = np.ones(size)
    kernel = kernel / size

    blurred = cv2.filter2D(x, -1, kernel)
    return np.clip(blurred, 0.0, 1.0).astype(np.float32)
