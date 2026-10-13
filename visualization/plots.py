"""Visualization and plotting suite for training, robustness benchmarking, and sample inspection."""

from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import matplotlib.pyplot as plt


def plot_training_curves(log_csv_path: str = "experiments/baseline_training_log.csv", save_path: str = "experiments/training_curves.png"):
    """Plots training and validation loss & accuracy curves over epochs."""
    import pandas as pd
    df = pd.read_csv(log_csv_path)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss plot
    ax1.plot(df["epoch"], df["train_loss"], label="Train Loss", color="royalblue", lw=2)
    ax1.plot(df["epoch"], df["val_loss"], label="Val Loss", color="crimson", lw=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross Entropy Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()

    # Accuracy plot
    ax1_twin = ax2
    ax1_twin.plot(df["epoch"], df["train_acc"], label="Train Acc (Top-1)", color="royalblue", lw=2)
    ax1_twin.plot(df["epoch"], df["val_acc1"], label="Val Acc (Top-1)", color="crimson", lw=2)
    if "val_acc5" in df.columns:
        ax1_twin.plot(df["epoch"], df["val_acc5"], label="Val Acc (Top-5)", color="forestgreen", lw=1.5, linestyle="--")
    ax1_twin.set_xlabel("Epoch")
    ax1_twin.set_ylabel("Accuracy (%)")
    ax1_twin.set_title("Training & Validation Accuracy")
    ax1_twin.grid(True, linestyle="--", alpha=0.6)
    ax1_twin.legend()

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Training curves saved to {save_path}")


def plot_confusion_matrix(conf_mat: np.ndarray, save_path: str = "experiments/confusion_matrix.png"):
    """Plots a 100x100 confusion matrix heatmap."""
    plt.figure(figsize=(10, 8))
    plt.imshow(conf_mat, interpolation="nearest", cmap="Blues")
    plt.title("CIFAR-100 Confusion Matrix (100 Classes)")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def plot_class_accuracy(class_acc: Dict[int, float], save_path: str = "experiments/class_accuracy.png"):
    """Plots class-wise accuracy distribution sorted from worst to best."""
    sorted_items = sorted(class_acc.items(), key=lambda x: x[1])
    classes = [str(x[0]) for x in sorted_items]
    accuracies = [x[1] for x in sorted_items]

    plt.figure(figsize=(16, 5))
    bars = plt.bar(range(len(classes)), accuracies, color="steelblue", width=0.8)
    plt.xlabel("Class Index (sorted by accuracy)")
    plt.ylabel("Accuracy (%)")
    plt.title("CIFAR-100 Per-Class Accuracy Distribution")
    plt.axhline(np.mean(accuracies), color="crimson", linestyle="--", label=f"Mean Acc: {np.mean(accuracies):.2f}%")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Class accuracy plot saved to {save_path}")


def plot_adversarial_triplet(
    clean_img: np.ndarray,
    pert_img: np.ndarray,
    adv_img: np.ndarray,
    clean_label: str,
    adv_label: str,
    clean_conf: float,
    adv_conf: float,
    save_path: str = "experiments/adversarial_sample.png",
):
    """
    Renders an original image, amplified perturbation, and adversarial image.
    clean_img and adv_img should be in [0, 1] range (H, W, C).
    pert_img is difference (adv - clean).
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # 1. Clean
    axes[0].imshow(np.clip(clean_img, 0, 1))
    axes[0].set_title(f"Clean Input\nPred: {clean_label} ({clean_conf*100:.1f}%)")
    axes[0].axis("off")

    # 2. Perturbation (amplified 10x and shifted by 0.5 for visualization)
    amp_pert = np.clip((pert_img * 10.0) + 0.5, 0, 1)
    axes[1].imshow(amp_pert)
    axes[1].set_title("Perturbation $\Delta$\n(Amplified 10x)")
    axes[1].axis("off")

    # 3. Adversarial
    axes[2].imshow(np.clip(adv_img, 0, 1))
    axes[2].set_title(f"Adversarial Input\nPred: {adv_label} ({adv_conf*100:.1f}%)")
    axes[2].axis("off")

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Adversarial sample saved to {save_path}")
