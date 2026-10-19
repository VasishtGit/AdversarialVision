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
    angle = np.random.uniform(0.0, 360.0)
    kernel = np.zeros((size, size), dtype=np.float32)
    kernel[int((size - 1) / 2), :] = 1.0
    # Rotate kernel
    M = cv2.getRotationMatrix2D((size / 2.0 - 0.5, size / 2.0 - 0.5), angle, 1.0)
    kernel = cv2.warpAffine(kernel, M, (size, size))
    ksum = kernel.sum()
    if ksum > 0:
        kernel = kernel / ksum

    blurred = cv2.filter2D(x, -1, kernel)
    return np.clip(blurred, 0.0, 1.0).astype(np.float32)
