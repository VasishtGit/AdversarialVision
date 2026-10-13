"""Image quality corruptions: JPEG compression and downsampling."""

import numpy as np
import cv2


def jpeg_compression(x: np.ndarray, severity: int = 1) -> np.ndarray:
    """
    Applies JPEG compression artifacts to image x in [0, 1] with shape (H, W, C).
    severity: 1 to 5 (maps to quality factor 75 down to 10)
    """
    qualities = [75, 55, 35, 20, 10]
    quality = qualities[severity - 1]

    # Convert to uint8 [0, 255] for OpenCV encode
    img_uint8 = np.clip(x * 255.0, 0, 255).astype(np.uint8)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]

    _, encimg = cv2.imencode(".jpg", img_uint8, encode_param)
    decimg = cv2.imdecode(encimg, 1)

    return (decimg / 255.0).astype(np.float32)


def downsampling(x: np.ndarray, severity: int = 1) -> np.ndarray:
    """
    Downsamples the image then upsamples back to original resolution.
    severity: 1 to 5
    """
    factors = [1.25, 1.5, 2.0, 3.0, 4.0]
    factor = factors[severity - 1]
    h, w = x.shape[:2]

    new_h, new_w = max(4, int(h / factor)), max(4, int(w / factor))
    down = cv2.resize(x, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    up = cv2.resize(down, (w, h), interpolation=cv2.INTER_NEAREST)

    return np.clip(up, 0.0, 1.0).astype(np.float32)
