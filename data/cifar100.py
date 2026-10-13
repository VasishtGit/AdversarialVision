"""CIFAR-100 dataset pipeline with deterministic train/val/test splits."""

from typing import Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import datasets, transforms

# Official CIFAR-100 per-channel mean and standard deviation
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


class SubsetWithTransform(Dataset):
    """
    Wraps a torch.utils.data.Subset (or dataset) to apply a specific transform.
    Allows splitting CIFAR-100 once while applying different transforms to train and val.
    """
    def __init__(self, subset: Subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx: int):
        img, target = self.subset[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, target


def get_cifar100_transforms(normalize_in_transform: bool = False) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Creates train and validation/test transforms.

    By default, `normalize_in_transform=False` keeps tensors bounded in [0.0, 1.0],
    allowing attacks and corruptions to operate on physical pixel ranges.
    Channel normalization is applied inside the model's differentiable input layer.
    """
    train_ops = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
    eval_ops = [
        transforms.ToTensor(),
    ]

    if normalize_in_transform:
        train_ops.append(transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD))
        eval_ops.append(transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD))

    return transforms.Compose(train_ops), transforms.Compose(eval_ops)


def get_cifar100_loaders(
    data_dir: str = "./data",
    batch_size: int = 128,
    eval_batch_size: int = 256,
    num_workers: int = 2,
    pin_memory: bool = True,
    train_size: int = 45000,
    val_size: int = 5000,
    seed: int = 42,
    normalize_in_transform: bool = False,
    download: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Downloads and partitions CIFAR-100 into:
    - Train: 45,000 samples (with RandomCrop + RandomHorizontalFlip)
    - Validation: 5,000 samples (no augmentations)
    - Test: 10,000 samples (no augmentations)

    Returns:
        (train_loader, val_loader, test_loader)
    """
    assert train_size + val_size == 50000, f"train_size ({train_size}) + val_size ({val_size}) must equal 50000"

    train_transform, eval_transform = get_cifar100_transforms(normalize_in_transform=normalize_in_transform)

    # Base raw dataset without transforms (returns PIL.Image)
    raw_train_dataset = datasets.CIFAR100(
        root=data_dir,
        train=True,
        download=download,
        transform=None,
    )

    # Deterministic split into train and val
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(
        raw_train_dataset,
        [train_size, val_size],
        generator=generator,
    )

    train_dataset = SubsetWithTransform(train_subset, transform=train_transform)
    val_dataset = SubsetWithTransform(val_subset, transform=eval_transform)

    raw_test_dataset = datasets.CIFAR100(
        root=data_dir,
        train=False,
        download=download,
        transform=eval_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        raw_test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    print("Testing CIFAR-100 pipeline...")
    tr_loader, v_loader, te_loader = get_cifar100_loaders(data_dir="./data")
    print(f"Train samples: {len(tr_loader.dataset)} ({len(tr_loader)} batches)")
    print(f"Val samples:   {len(v_loader.dataset)} ({len(v_loader)} batches)")
    print(f"Test samples:  {len(te_loader.dataset)} ({len(te_loader)} batches)")

    images, labels = next(iter(tr_loader))
    print(f"Batch shape:   {images.shape}")
    print(f"Labels shape:  {labels.shape}")
    print(f"Value range:   [{images.min().item():.3f}, {images.max().item():.3f}]")
    print("CIFAR-100 pipeline verified successfully!")
