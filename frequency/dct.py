"""2D Discrete Cosine Transform (DCT) spectral analysis."""

from typing import Dict, Union, Tuple
import numpy as np
import torch
import cv2


def compute_dct2d(image: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """
    Computes 2D DCT for image in [0, 1] using OpenCV's fast float32 DCT.
    """
    if isinstance(image, torch.Tensor):
        if image.dim() == 3:
            img_np = image.detach().cpu().mean(dim=0).numpy()
        else:
            img_np = image.detach().cpu().numpy()
    else:
        if image.ndim == 3:
            img_np = np.mean(image, axis=2)
        else:
            img_np = image

    img_float = img_np.astype(np.float32)
    dct = cv2.dct(img_float)
    return dct


def extract_dct_features(image: Union[np.ndarray, torch.Tensor], cutoff_ratio: float = 0.25) -> Dict[str, float]:
    """
    Extracts frequency-domain features from 2D DCT:
    - low_freq_energy: energy in top-left low-frequency corner
    - high_freq_energy: energy outside low-frequency corner
    - energy_ratio: high / low
    """
    dct = compute_dct2d(image)
    power = dct ** 2
    total_energy = float(np.sum(power)) + 1e-12

    h, w = power.shape
    cutoff_h = max(1, int(h * cutoff_ratio))
    cutoff_w = max(1, int(w * cutoff_ratio))

    low_mask = np.zeros_like(power, dtype=bool)
    low_mask[:cutoff_h, :cutoff_w] = True
    high_mask = ~low_mask

    low_energy = float(np.sum(power[low_mask]))
    high_energy = float(np.sum(power[high_mask]))
    ratio = float(high_energy / (low_energy + 1e-12))

    prob = power / total_energy
    entropy = -float(np.sum(prob * np.log(prob + 1e-12)))

    return {
        "dct_low_freq_energy": low_energy,
        "dct_high_freq_energy": high_energy,
        "dct_high_low_ratio": ratio,
        "dct_spectral_entropy": entropy,
    }
