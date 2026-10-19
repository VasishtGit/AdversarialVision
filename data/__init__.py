"""Data loading and preprocessing pipelines for CIFAR-10 and CIFAR-100."""

from .cifar100 import get_cifar100_loaders, get_cifar100_test_loader, CIFAR100_MEAN, CIFAR100_STD
from .cifar10 import get_cifar10_loaders, CIFAR10_MEAN, CIFAR10_STD

__all__ = [
    "get_cifar100_loaders",
    "get_cifar100_test_loader",
    "CIFAR100_MEAN",
    "CIFAR100_STD",
    "get_cifar10_loaders",
    "CIFAR10_MEAN",
    "CIFAR10_STD",
]
