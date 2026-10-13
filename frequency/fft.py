"""2D Fast Fourier Transform (FFT) spectral domain analysis."""

from typing import Dict, Union, Tuple
import numpy as np
import torch


def compute_fft2d(image: Union[np.ndarray, torch.Tensor]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes shifted 2D FFT magnitude spectrum for an image in [0, 1].
    Returns (shifted_fft, magnitude_spectrum).
    """
    if isinstance(image, torch.Tensor):
        if image.dim() == 3:  # (C, H, W) -> grayscale or mean across channels
            img_np = image.detach().cpu().mean(dim=0).numpy()
        elif image.dim() == 2:
            img_np = image.detach().cpu().numpy()
        else:
            raise ValueError(f"Expected 2D or 3D tensor, got {image.shape}")
    else:
        if image.ndim == 3:
            img_np = np.mean(image, axis=2)
        else:
            img_np = image

    # Compute 2D FFT and center zero frequency
    f = np.fft.fft2(img_np)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)

    return fshift, magnitude


def extract_fft_features(image: Union[np.ndarray, torch.Tensor], cutoff_radius_ratio: float = 0.25) -> Dict[str, float]:
    """
    Extracts frequency-domain features from 2D FFT:
    - low_freq_energy: energy inside radius cutoff
    - high_freq_energy: energy outside radius cutoff
    - energy_ratio: high_freq_energy / low_freq_energy
    - spectral_entropy: Shannon entropy of normalized power spectrum
    """
    _, magnitude = compute_fft2d(image)
    power_spectrum = magnitude ** 2
    total_energy = float(np.sum(power_spectrum)) + 1e-12

    h, w = power_spectrum.shape
    cy, cx = h // 2, w // 2
    max_radius = min(cy, cx)
    cutoff = max_radius * cutoff_radius_ratio

    # Circular mask for low frequency
    y, x = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    low_mask = dist_from_center <= cutoff
    high_mask = ~low_mask

    low_energy = float(np.sum(power_spectrum[low_mask]))
    high_energy = float(np.sum(power_spectrum[high_mask]))
    ratio = float(high_energy / (low_energy + 1e-12))

    # Normalized spectral probability distribution
    prob = power_spectrum / total_energy
    entropy = -float(np.sum(prob * np.log(prob + 1e-12)))

    return {
        "fft_low_freq_energy": low_energy,
        "fft_high_freq_energy": high_energy,
        "fft_high_low_ratio": ratio,
        "fft_spectral_entropy": entropy,
    }
