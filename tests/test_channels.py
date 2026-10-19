"""Unit tests verifying color space consistency (RGB preservation)."""

import pytest
import numpy as np
import torch
from corruptions.compression import jpeg_compression
from defenses.preprocessing import def_jpeg_recompression, def_bilateral_filter


def test_jpeg_color_channel_order():
    """
    Constructs an image with pure Red color in RGB: [1.0, 0.0, 0.0].
    Verifies that after JPEG compression and JPEG defense, Red channel remains dominant.
    If decoded as BGR, Red and Blue would be flipped and Blue would be dominant!
    """
    # Pure red image in RGB (H, W, C)
    red_image = np.zeros((32, 32, 3), dtype=np.float32)
    red_image[:, :, 0] = 1.0  # R=1, G=0, B=0

    # 1. Test corruptions/compression.py
    jpeg_corrupt = jpeg_compression(red_image, severity=1)
    mean_r = jpeg_corrupt[:, :, 0].mean()
    mean_b = jpeg_corrupt[:, :, 2].mean()
    assert mean_r > 0.8, f"Expected dominant red, got mean R={mean_r}"
    assert mean_b < 0.2, f"Expected low blue in red image, got mean B={mean_b} (BGR swap detected!)"

    # 2. Test defenses/preprocessing.py
    jpeg_def = def_jpeg_recompression(red_image, quality=90)
    def_r = jpeg_def[:, :, 0].mean()
    def_b = jpeg_def[:, :, 2].mean()
    assert def_r > 0.8, f"Defense: Expected dominant red, got mean R={def_r}"
    assert def_b < 0.2, f"Defense: Expected low blue, got mean B={def_b} (BGR swap detected!)"
