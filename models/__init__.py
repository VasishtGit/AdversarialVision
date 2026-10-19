"""Models package for adversarial vision architectures with model registry."""

from typing import Dict, Callable
import torch.nn as nn

from .resnet_cifar import resnet18_cifar, ResNet_CIFAR, NormalizeLayer
from .wideresnet import WideResNet, wideresnet28_10, wideresnet34_10

MODEL_REGISTRY: Dict[str, Callable[..., nn.Module]] = {
    "resnet18": resnet18_cifar,
    "resnet18_cifar": resnet18_cifar,
    "wideresnet28_10": wideresnet28_10,
    "wideresnet34_10": wideresnet34_10,
}


def build_model(architecture: str, num_classes: int = 100, normalize: bool = True, **kwargs) -> nn.Module:
    """Builds a model from the registry."""
    arch = architecture.lower()
    if arch not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model architecture '{architecture}'. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[arch](num_classes=num_classes, normalize=normalize, **kwargs)


__all__ = [
    "resnet18_cifar",
    "ResNet_CIFAR",
    "NormalizeLayer",
    "WideResNet",
    "wideresnet28_10",
    "wideresnet34_10",
    "MODEL_REGISTRY",
    "build_model",
]
