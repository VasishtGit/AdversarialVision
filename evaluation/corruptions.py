"""Natural image corruption benchmarking across 10 corruptions and 5 severity levels."""

import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from corruptions import corrupt, CORRUPTIONS_LIST
from evaluation.metrics import compute_confidence_and_entropy


@torch.no_grad()
def evaluate_corrupted(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    corruption_type: str,
    severity: int,
    device: torch.device,
    amp_enabled: bool = True,
    criterion: Optional[nn.Module] = None,
) -> Dict[str, float]:
    """
    Evaluates the model on the full test set corrupted by a specific type & severity.
    """
    model.eval()
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    running_loss = 0.0
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    all_confs = []
    all_entropies = []

    for images, targets in loader:
        # 1. Apply corruption in [0, 1] input space
        corrupted_images = corrupt(images, corruption_type, severity=severity)

        corrupted_images = corrupted_images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda", enabled=amp_enabled and device.type == "cuda"):
            logits = model(corrupted_images)
            loss = criterion(logits, targets)

        running_loss += loss.item() * targets.size(0)
        total += targets.size(0)

        # Accuracy
        _, preds_top5 = logits.topk(5, 1, largest=True, sorted=True)
        targets_expanded = targets.view(-1, 1).expand_as(preds_top5)
        correct_in_top5 = preds_top5.eq(targets_expanded)
        correct_top1 += correct_in_top5[:, 0].sum().item()
        correct_top5 += correct_in_top5.any(dim=1).sum().item()

        # Confidence & Entropy
        confs, entropies = compute_confidence_and_entropy(logits)
        all_confs.append(confs.cpu().numpy())
        all_entropies.append(entropies.cpu().numpy())

    confs_arr = np.concatenate(all_confs)
    entropies_arr = np.concatenate(all_entropies)

    return {
        "accuracy": float((correct_top1 / total) * 100.0),
        "top5_accuracy": float((correct_top5 / total) * 100.0),
        "loss": float(running_loss / total),
        "mean_confidence": float(np.mean(confs_arr)),
        "mean_entropy": float(np.mean(entropies_arr)),
    }


def run_natural_corruption_benchmark(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    clean_accuracy: float = 77.84,
    output_csv: str = "experiments/natural_corruption_results.csv",
    curves_plot_path: str = "experiments/natural_corruption_curves.png",
    ranking_plot_path: str = "experiments/natural_corruption_ranking.png",
    amp_enabled: bool = True,
) -> pd.DataFrame:
    """
    Runs full Phase 9 natural robustness experiment:
    10 corruptions x 5 severities = 50 evaluation passes.
    """
    results = []
    print("=" * 70)
    print(" PHASE 9: NATURAL IMAGE CORRUPTIONS ROBUSTNESS BENCHMARK")
    print(f" Corruptions ({len(CORRUPTIONS_LIST)}): {', '.join(CORRUPTIONS_LIST)}")
    print(" Severities: 1, 2, 3, 4, 5")
    print(f" Clean Baseline Top-1 Acc: {clean_accuracy:.2f}%")
    print("=" * 70)

    # Add clean baseline row (severity 0)
    for c_type in CORRUPTIONS_LIST:
        results.append({
            "corruption": c_type,
            "severity": 0,
            "accuracy": clean_accuracy,
            "top5_accuracy": 93.92,
            "loss": 0.9134,
            "mean_confidence": 0.7983,
            "mean_entropy": 0.9288,
            "relative_retention": 100.0,
        })

    for c_idx, c_type in enumerate(CORRUPTIONS_LIST, 1):
        print(f"\n[{c_idx}/{len(CORRUPTIONS_LIST)}] Benchmarking '{c_type}'...")
        for severity in range(1, 6):
            start = time.perf_counter()
            metrics = evaluate_corrupted(
                model=model,
                loader=loader,
                corruption_type=c_type,
                severity=severity,
                device=device,
                amp_enabled=amp_enabled,
            )
            elapsed = time.perf_counter() - start

            retention = (metrics["accuracy"] / clean_accuracy) * 100.0 if clean_accuracy > 0 else 0.0

            results.append({
                "corruption": c_type,
                "severity": severity,
                "accuracy": metrics["accuracy"],
                "top5_accuracy": metrics["top5_accuracy"],
                "loss": metrics["loss"],
                "mean_confidence": metrics["mean_confidence"],
                "mean_entropy": metrics["mean_entropy"],
                "relative_retention": retention,
            })

            print(
                f"  - Severity {severity}: Acc: {metrics['accuracy']:.2f}% (Retention: {retention:.1f}%) | "
                f"Top-5: {metrics['top5_accuracy']:.2f}% | Loss: {metrics['loss']:.3f} | "
                f"Conf: {metrics['mean_confidence']*100:.1f}% | Time: {elapsed:.1f}s"
            )

    df = pd.DataFrame(results)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\nSaved results to {output_csv}")

    # Generate plots
    plot_corruption_degradation(df, clean_accuracy, curves_plot_path)
    plot_corruption_ranking(df, ranking_plot_path)

    return df


def plot_corruption_degradation(df: pd.DataFrame, clean_accuracy: float, save_path: str):
    """Plots accuracy vs severity curves for all corruptions."""
    plt.figure(figsize=(12, 7))

    # Color palette
    colors = plt.cm.tab10(np.linspace(0, 1, len(CORRUPTIONS_LIST)))

    for idx, c_type in enumerate(CORRUPTIONS_LIST):
        sub_df = df[df["corruption"] == c_type].sort_values("severity")
        plt.plot(
            sub_df["severity"],
            sub_df["accuracy"],
            marker="o",
            linewidth=2,
            label=c_type.replace("_", " ").title(),
            color=colors[idx],
        )

    plt.axhline(clean_accuracy, color="black", linestyle="--", alpha=0.7, label=f"Clean Test Acc ({clean_accuracy:.1f}%)")
    plt.xlabel("Corruption Severity (0 = Clean)", fontsize=12)
    plt.ylabel("Top-1 Accuracy (%)", fontsize=12)
    plt.title("Model Robustness Under Natural Image Corruptions", fontsize=14, fontweight="bold")
    plt.xticks([0, 1, 2, 3, 4, 5])
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", frameon=True)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Degradation curves saved to {save_path}")


def plot_corruption_ranking(df: pd.DataFrame, save_path: str):
    """Plots average severity 1-5 accuracy ranking showing which corruption hurts performance most."""
    # Only severities 1 to 5
    corrupted_df = df[df["severity"] > 0]
    avg_acc = corrupted_df.groupby("corruption")["accuracy"].mean().sort_values()

    plt.figure(figsize=(11, 6))
    bars = plt.barh([c.replace("_", " ").title() for c in avg_acc.index], avg_acc.values, color="tomato", edgecolor="black", alpha=0.85)

    plt.xlabel("Mean Accuracy Across Severities 1-5 (%)", fontsize=12)
    plt.title("Vulnerability Ranking: Most Destructive Natural Corruptions", fontsize=14, fontweight="bold")
    plt.grid(axis="x", linestyle="--", alpha=0.6)

    # Value labels on bars
    for bar in bars:
        w = bar.get_width()
        plt.text(w + 0.8, bar.get_y() + bar.get_height() / 2, f"{w:.1f}%", va="center", ha="left", fontsize=10, fontweight="bold")

    plt.xlim(0, max(avg_acc.values) + 8)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Vulnerability ranking plot saved to {save_path}")
