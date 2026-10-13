"""Frequency analysis utilities (2D FFT and 2D DCT)."""

from typing import Dict, Union
import numpy as np
import torch

from .fft import compute_fft2d, extract_fft_features
from .dct import compute_dct2d, extract_dct_features


def analyze_spectral_profile(image: Union[np.ndarray, torch.Tensor]) -> Dict[str, float]:
    """Combines FFT and DCT spectral metrics into a single dictionary."""
    fft_feats = extract_fft_features(image)
    dct_feats = extract_dct_features(image)
    return {**fft_feats, **dct_feats}


__all__ = [
    "compute_fft2d",
    "extract_fft_features",
    "compute_dct2d",
    "extract_dct_features",
    "analyze_spectral_profile",
]
