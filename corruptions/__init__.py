"""OpenCV natural image corruption engine.

Provides corrupt(image, corruption_type, severity) across 10 distinct corruptions:
- Noise: gaussian_noise, salt_and_pepper_noise
- Blur: gaussian_blur, motion_blur
- Compression: jpeg_compression, downsampling
- Lighting: brightness, contrast, gamma
- Geometry: rotation, perspective
"""

from typing import Union, List
import numpy as np
import torch

from .noise import gaussian_noise, salt_and_pepper_noise
from .blur import gaussian_blur, motion_blur
from .compression import jpeg_compression, downsampling
from .lighting import brightness, contrast, gamma
from .geometric import rotation, perspective

CORRUPTIONS_REGISTRY = {
    "gaussian_noise": gaussian_noise,
    "salt_and_pepper_noise": salt_and_pepper_noise,
    "gaussian_blur": gaussian_blur,
    "motion_blur": motion_blur,
    "jpeg_compression": jpeg_compression,
    "downsampling": downsampling,
    "brightness": brightness,
    "contrast": contrast,
    "gamma": gamma,
    "rotation": rotation,
    "perspective": perspective,
}

CORRUPTIONS_LIST: List[str] = list(CORRUPTIONS_REGISTRY.keys())


def corrupt(
    image: Union[np.ndarray, torch.Tensor],
    corruption_type: str,
    severity: int = 1,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Applies the specified corruption to an image in [0, 1] range.
    Supports:
    - np.ndarray of shape (H, W, C)
    - torch.Tensor of shape (C, H, W)
    - torch.Tensor of shape (B, C, H, W)
    """
    if corruption_type not in CORRUPTIONS_REGISTRY:
        raise ValueError(f"Unknown corruption '{corruption_type}'. Available: {CORRUPTIONS_LIST}")
    if not (1 <= severity <= 5):
        raise ValueError(f"Severity must be between 1 and 5 (got {severity}).")

    fn = CORRUPTIONS_REGISTRY[corruption_type]

    # Handle torch.Tensor
    if isinstance(image, torch.Tensor):
        is_batch = image.dim() == 4
        device = image.device
        dtype = image.dtype

        if is_batch:
            # (B, C, H, W) -> process each item
            batch_list = []
            for i in range(image.size(0)):
                item_np = image[i].detach().cpu().permute(1, 2, 0).numpy()
                corrupted_np = fn(item_np, severity=severity)
                batch_list.append(torch.from_numpy(corrupted_np).permute(2, 0, 1))
            return torch.stack(batch_list).to(device=device, dtype=dtype)
        else:
            # (C, H, W)
            item_np = image.detach().cpu().permute(1, 2, 0).numpy()
            corrupted_np = fn(item_np, severity=severity)
            return torch.from_numpy(corrupted_np).permute(2, 0, 1).to(device=device, dtype=dtype)

    # Handle np.ndarray (H, W, C)
    return fn(image, severity=severity)


__all__ = [
    "corrupt",
    "CORRUPTIONS_REGISTRY",
    "CORRUPTIONS_LIST",
    "gaussian_noise",
    "salt_and_pepper_noise",
    "gaussian_blur",
    "motion_blur",
    "jpeg_compression",
    "downsampling",
    "brightness",
    "contrast",
    "gamma",
    "rotation",
    "perspective",
]
