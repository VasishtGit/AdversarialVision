"""Evaluation package for adversarial vision framework."""

from .metrics import (
    compute_confidence_and_entropy,
    compute_topk_accuracies,
    compute_class_wise_accuracy,
    compute_confusion_matrix,
)
from .clean import evaluate_clean, save_baseline_results

__all__ = [
    "compute_confidence_and_entropy",
    "compute_topk_accuracies",
    "compute_class_wise_accuracy",
    "compute_confusion_matrix",
    "evaluate_clean",
    "save_baseline_results",
]
