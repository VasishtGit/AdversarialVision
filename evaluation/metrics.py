"""Metrics calculation utilities for robustness benchmarking."""

from typing import Dict, Any, Tuple
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import confusion_matrix


def compute_confidence_and_entropy(logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes top-1 prediction confidence and Shannon entropy from raw logits.
    Uses FP32 log_softmax for strict numerical stability.
    """
    logits_f32 = logits.float()
    probs = F.softmax(logits_f32, dim=1)
    log_probs = F.log_softmax(logits_f32, dim=1)
    confs, _ = probs.max(dim=1)
    entropy = -torch.sum(probs * log_probs, dim=1)
    return confs, entropy


def compute_topk_accuracies(logits: torch.Tensor, targets: torch.Tensor, topk: Tuple[int, ...] = (1, 5)) -> Dict[str, float]:
    """Computes top-k accuracy percentages."""
    maxk = max(topk)
    batch_size = targets.size(0)

    _, pred = logits.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))

    res = {}
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True).item()
        res[f"top{k}_accuracy"] = (correct_k / batch_size) * 100.0
    return res


def compute_class_wise_accuracy(
    all_preds: np.ndarray,
    all_targets: np.ndarray,
    num_classes: int = 100,
) -> Dict[int, float]:
    """Computes per-class accuracy percentages."""
    class_acc = {}
    for c in range(num_classes):
        mask = (all_targets == c)
        total_c = np.sum(mask)
        if total_c > 0:
            correct_c = np.sum(all_preds[mask] == c)
            class_acc[c] = float((correct_c / total_c) * 100.0)
        else:
            class_acc[c] = 0.0
    return class_acc


def compute_confusion_matrix(
    all_preds: np.ndarray,
    all_targets: np.ndarray,
    num_classes: int = 100,
) -> np.ndarray:
    """Computes square confusion matrix of size (num_classes, num_classes)."""
    return confusion_matrix(all_targets, all_preds, labels=list(range(num_classes)))
