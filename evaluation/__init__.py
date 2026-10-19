"""Evaluation package for adversarial vision framework."""

from .metrics import (
    compute_confidence_and_entropy,
    compute_topk_accuracies,
    compute_class_wise_accuracy,
    compute_confusion_matrix,
)
from .clean import evaluate_clean, save_baseline_results
from .corruptions import run_natural_corruption_benchmark
from .attacks import run_adversarial_benchmark
from .defenses import run_defense_benchmark
from .transfer import evaluate_transferability

__all__ = [
    "compute_confidence_and_entropy",
    "compute_topk_accuracies",
    "compute_class_wise_accuracy",
    "compute_confusion_matrix",
    "evaluate_clean",
    "save_baseline_results",
    "run_natural_corruption_benchmark",
    "run_adversarial_benchmark",
    "run_defense_benchmark",
    "evaluate_transferability",
]
