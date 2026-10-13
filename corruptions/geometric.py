"""Geometric corruptions: Rotation and Perspective transformation."""

import numpy as np
import cv2


def rotation(x: np.ndarray, severity: int = 1) -> np.ndarray:
    """
    Rotates image by a small random angle.
    severity: 1 to 5
    """
    angles = [4.0, 8.0, 12.0, 18.0, 25.0]
    angle = angles[severity - 1]
    # Random direction (+ or -)
    sign = np.random.choice([-1.0, 1.0])
    angle = angle * sign

    h, w = x.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(x, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
    return np.clip(rotated, 0.0, 1.0).astype(np.float32)


def perspective(x: np.ndarray, severity: int = 1) -> np.ndarray:
    """
    Applies small perspective distortion.
    severity: 1 to 5
    """
    strengths = [0.03, 0.06, 0.09, 0.13, 0.18]
    s = strengths[severity - 1]

    h, w = x.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

    # Perturb corner points
    dx = np.random.uniform(-s * w, s * w, 4)
    dy = np.random.uniform(-s * h, s * h, 4)
    dst = np.float32([
        [0 + dx[0], 0 + dy[0]],
        [w + dx[1], 0 + dy[1]],
        [w + dx[2], h + dy[2]],
        [0 + dx[3], h + dy[3]],
    ])

    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(x, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
    return np.clip(warped, 0.0, 1.0).astype(np.float32)
