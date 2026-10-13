"""OpenCV-based image preprocessing defenses against adversarial perturbations."""

from typing import Union, Dict, Callable
import numpy as np
import torch
import cv2


def def_gaussian_filter(x: np.ndarray, ksize: int = 3, sigma: float = 0.5) -> np.ndarray:
    """Gaussian smoothing filter."""
    return cv2.GaussianBlur(x, (ksize, ksize), sigmaX=sigma, sigmaY=sigma).astype(np.float32)


def def_median_filter(x: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Median smoothing filter (effective against impulse and high-frequency noise)."""
    # OpenCV medianBlur expects uint8 or float32 (single channel), uint8 is most standard
    img_u8 = np.clip(x * 255.0, 0, 255).astype(np.uint8)
    med = cv2.medianBlur(img_u8, ksize)
    return (med / 255.0).astype(np.float32)


def def_bilateral_filter(x: np.ndarray, d: int = 3, sigma_color: float = 0.1, sigma_space: float = 3.0) -> np.ndarray:
    """Bilateral filter (edge-preserving smoothing)."""
    img_u8 = np.clip(x * 255.0, 0, 255).astype(np.uint8)
    # sigma_color in [0, 255]
    sc = int(sigma_color * 255.0)
    filtered = cv2.bilateralFilter(img_u8, d=d, sigmaColor=sc, sigmaSpace=sigma_space)
    return (filtered / 255.0).astype(np.float32)


def def_jpeg_recompression(x: np.ndarray, quality: int = 75) -> np.ndarray:
    """JPEG recompression defense (removes high-frequency adversarial perturbations)."""
    img_u8 = np.clip(x * 255.0, 0, 255).astype(np.uint8)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, enc = cv2.imencode(".jpg", img_u8, encode_param)
    dec = cv2.imdecode(enc, 1)
    return (dec / 255.0).astype(np.float32)


DEFENSES_REGISTRY: Dict[str, Callable] = {
    "gaussian": def_gaussian_filter,
    "median": def_median_filter,
    "bilateral": def_bilateral_filter,
    "jpeg": def_jpeg_recompression,
}


class PreprocessDefense:
    """
    Applies an OpenCV preprocessing defense to an image or batch of images.
    Operates strictly in [0.0, 1.0].
    """
    def __init__(self, defense_type: str = "gaussian", **kwargs):
        if defense_type not in DEFENSES_REGISTRY:
            raise ValueError(f"Unknown defense '{defense_type}'. Available: {list(DEFENSES_REGISTRY.keys())}")
        self.defense_type = defense_type
        self.fn = DEFENSES_REGISTRY[defense_type]
        self.kwargs = kwargs

    def __call__(self, image: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        return self.preprocess(image)

    def preprocess(self, image: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        if isinstance(image, torch.Tensor):
            is_batch = image.dim() == 4
            device = image.device
            dtype = image.dtype

            if is_batch:
                cleaned = []
                for i in range(image.size(0)):
                    item_np = image[i].detach().cpu().permute(1, 2, 0).numpy()
                    filtered_np = self.fn(item_np, **self.kwargs)
                    cleaned.append(torch.from_numpy(filtered_np).permute(2, 0, 1))
                return torch.stack(cleaned).to(device=device, dtype=dtype)
            else:
                item_np = image.detach().cpu().permute(1, 2, 0).numpy()
                filtered_np = self.fn(item_np, **self.kwargs)
                return torch.from_numpy(filtered_np).permute(2, 0, 1).to(device=device, dtype=dtype)

        return self.fn(image, **self.kwargs)
