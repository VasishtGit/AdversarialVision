"""CIFAR-10 dataset pipeline matching CIFAR-100 contract for multi-dataset benchmarking."""

from typing import Tuple
import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import datasets, transforms
from data.cifar100 import SubsetWithTransform

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def get_cifar10_transforms(normalize_in_transform: bool = False) -> Tuple[transforms.Compose, transforms.Compose]:
    train_ops = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
    eval_ops = [
        transforms.ToTensor(),
    ]

    if normalize_in_transform:
        norm = transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD)
        train_ops.append(norm)
        eval_ops.append(norm)

    return transforms.Compose(train_ops), transforms.Compose(eval_ops)


def get_cifar10_loaders(
    data_dir: str = "./data",
    batch_size: int = 128,
    eval_batch_size: int = 256,
    num_workers: int = 2,
    pin_memory: bool = True,
    download: bool = True,
    train_size: int = 45000,
    val_size: int = 5000,
    seed: int = 42,
    normalize_in_transform: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_transform, eval_transform = get_cifar10_transforms(normalize_in_transform)

    raw_train = datasets.CIFAR10(root=data_dir, train=True, download=download, transform=None)
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(raw_train, [train_size, val_size], generator=generator)

    train_dataset = SubsetWithTransform(train_subset, transform=train_transform)
    val_dataset = SubsetWithTransform(val_subset, transform=eval_transform)
    raw_test = datasets.CIFAR10(root=data_dir, train=False, download=download, transform=eval_transform)

    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory),
        DataLoader(val_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory),
        DataLoader(raw_test, batch_size=eval_batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory),
    )
