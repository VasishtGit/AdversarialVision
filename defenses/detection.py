"""Adversarial perturbation detection module.
Implements Local Intrinsic Dimensionality (LID) and Reconstruction-based anomaly detection.
"""

from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.distance import cdist


def compute_lid(reference_features: np.ndarray, query_features: np.ndarray, k: int = 20) -> np.ndarray:
    """
    Computes Local Intrinsic Dimensionality (LID) score for query features
    relative to reference training feature distribution (Ma et al., ICLR 2018).
    High LID often signifies out-of-manifold adversarial points.
    """
    dists = cdist(query_features, reference_features)
    # Sort distances to get k nearest neighbors
    sorted_dists = np.sort(dists, axis=1)[:, 1 : k + 1]
    r_k = sorted_dists[:, -1:]  # max distance among k neighbors
    # Avoid zero division
    r_k = np.maximum(r_k, 1e-12)
    log_ratios = np.log(sorted_dists / r_k)
    lid = -k / (np.sum(log_ratios, axis=1) + 1e-12)
    return lid


class EnergyBasedDetector:
    """
    Detects adversarial and out-of-distribution inputs via Energy scores (Liu et al., NeurIPS 2020).
    E(x; T) = -T * log(sum(exp(f(x) / T))).
    Adversarial attacks often distort energy dynamics compared to in-distribution clean inputs.
    """

    def __init__(self, temperature: float = 1.0, threshold: float = -10.0):
        self.temperature = temperature
        self.threshold = threshold

    @torch.no_grad()
    def compute_energy(self, logits: torch.Tensor) -> torch.Tensor:
        """Returns free energy values for batch logits."""
        return -self.temperature * torch.logsumexp(logits / self.temperature, dim=1)

    @torch.no_grad()
    def is_adversarial(self, logits: torch.Tensor) -> torch.Tensor:
        """Returns boolean mask indicating detected anomalous inputs."""
        energies = self.compute_energy(logits)
        return energies > self.threshold
