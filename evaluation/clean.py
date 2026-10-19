"""Evaluation module for untouched clean test set."""

import time
import json
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import numpy as np

from evaluation.metrics import (
    compute_topk_accuracies,
    compute_confidence_and_entropy,
    compute_class_wise_accuracy,
    compute_confusion_matrix,
)


@torch.no_grad()
def evaluate_clean(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    amp_enabled: bool = True,
    criterion: Optional[nn.Module] = None,
) -> Dict[str, Any]:
    """
    Evaluates the model on clean data and computes comprehensive metrics:
    - Top-1 and Top-5 accuracy
    - Cross-Entropy Loss
    - Mean Confidence and Mean Entropy
    - Inference Latency per sample / batch
    """
    model.eval()
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    running_loss = 0.0
    all_preds = []
    all_targets = []
    all_confs = []
    all_entropies = []
    total_samples = 0

    # Warm-up GPU
    if device.type == "cuda":
        dummy = torch.zeros(1, 3, 32, 32, device=device)
        _ = model(dummy)
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda", enabled=amp_enabled and device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, targets)

        confs, entropies = compute_confidence_and_entropy(logits)
        _, preds = logits.max(1)

        running_loss += loss.item() * targets.size(0)
        total_samples += targets.size(0)

        all_preds.append(preds.cpu().numpy())
        all_targets.append(targets.cpu().numpy())
        all_confs.append(confs.cpu().numpy())
        all_entropies.append(entropies.cpu().numpy())

        # Top-5 accuracy tracking
        _, preds_top5 = logits.topk(5, 1, largest=True, sorted=True)
        targets_expanded = targets.view(-1, 1).expand_as(preds_top5)
        correct_top5 = preds_top5.eq(targets_expanded).any(dim=1).sum().item()
        if "total_top5_correct" not in locals():
            total_top5_correct = 0
        total_top5_correct += correct_top5

    if device.type == "cuda":
        torch.cuda.synchronize()
    total_time = time.perf_counter() - start_time

    all_preds_arr = np.concatenate(all_preds)
    all_targets_arr = np.concatenate(all_targets)
    all_confs_arr = np.concatenate(all_confs)
    all_entropies_arr = np.concatenate(all_entropies)

    correct_top1 = (all_preds_arr == all_targets_arr).sum()

    # Class-wise accuracy and confusion matrix
    class_acc = compute_class_wise_accuracy(all_preds_arr, all_targets_arr, num_classes=100)
    conf_mat = compute_confusion_matrix(all_preds_arr, all_targets_arr, num_classes=100)

    results = {
        "accuracy": float((correct_top1 / total_samples) * 100.0),
        "top5_accuracy": float((total_top5_correct / total_samples) * 100.0),
        "loss": float(running_loss / total_samples),
        "mean_confidence": float(np.mean(all_confs_arr)),
        "mean_entropy": float(np.mean(all_entropies_arr)),
        "total_samples": int(total_samples),
        "inference_latency_ms_per_sample": float((total_time / total_samples) * 1000.0),
        "total_inference_time_sec": float(total_time),
        "class_wise_accuracy": class_acc,
        "confusion_matrix": conf_mat.tolist(),
    }

    return results


def save_baseline_results(results: Dict[str, Any], output_path: str = "experiments/baseline_results.json"):
    """Saves baseline evaluation results to JSON."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Baseline results saved to {output_path}")
