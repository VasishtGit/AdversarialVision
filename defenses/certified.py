"""Certified defense via Randomized Smoothing (Cohen et al., ICML 2019).
Certifies an L2 radius around inputs where prediction is provably invariant.
"""

from typing import Tuple, Optional
import math
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import norm


class RandomizedSmoothing(nn.Module):
    """
    Randomized Smoothing classifier wrapper.
    g(x) = argmax_c P(f(x + epsilon) = c), where epsilon ~ N(0, sigma^2 * I).
    """

    def __init__(self, base_classifier: nn.Module, num_classes: int = 100, sigma: float = 0.25):
        super().__init__()
        self.base_classifier = base_classifier
        self.num_classes = num_classes
        self.sigma = sigma

    @torch.no_grad()
    def predict(self, x: torch.Tensor, n_samples: int = 64, batch_size: int = 32) -> torch.Tensor:
        """Monte Carlo prediction via majority vote over noisy perturbations."""
        self.base_classifier.eval()
        counts = torch.zeros(x.size(0), self.num_classes, device=x.device)

        for _ in range(0, n_samples, batch_size):
            cur_n = min(batch_size, n_samples)
            # Repeat batch cur_n times
            x_rep = x.repeat_interleave(cur_n, dim=0)
            noise = torch.randn_like(x_rep) * self.sigma
            noisy_x = torch.clamp(x_rep + noise, 0.0, 1.0)

            logits = self.base_classifier(noisy_x)
            preds = logits.argmax(dim=1)

            for i in range(x.size(0)):
                chunk_preds = preds[i * cur_n : (i + 1) * cur_n]
                for p in chunk_preds:
                    counts[i, p] += 1

        return counts.argmax(dim=1)

    @torch.no_grad()
    def certify(
        self,
        x: torch.Tensor,
        n0: int = 100,
        n: int = 1000,
        alpha: float = 0.001,
        batch_size: int = 100,
    ) -> Tuple[int, float]:
        """
        Certifies a single input image x.
        Returns:
            (top_class, certified_radius_l2)
            certified_radius is 0.0 if certification test fails.
        """
        self.base_classifier.eval()
        # 1. Selection pass with n0 samples
        counts0 = np.zeros(self.num_classes, dtype=int)
        for _ in range(0, n0, batch_size):
            cur_n = min(batch_size, n0)
            x_rep = x.repeat(cur_n, 1, 1, 1)
            noise = torch.randn_like(x_rep) * self.sigma
            preds = self.base_classifier(torch.clamp(x_rep + noise, 0.0, 1.0)).argmax(dim=1).cpu().numpy()
            for p in preds:
                counts0[p] += 1

        c_hat = int(counts0.argmax())

        # 2. Estimation pass with n samples
        counts = np.zeros(self.num_classes, dtype=int)
        for _ in range(0, n, batch_size):
            cur_n = min(batch_size, n)
            x_rep = x.repeat(cur_n, 1, 1, 1)
            noise = torch.randn_like(x_rep) * self.sigma
            preds = self.base_classifier(torch.clamp(x_rep + noise, 0.0, 1.0)).argmax(dim=1).cpu().numpy()
            for p in preds:
                counts[p] += 1

        nA = counts[c_hat]

        # Clopper-Pearson lower bound on pA
        from scipy.stats import beta
        pA_lower = beta.ppf(alpha, nA, n - nA + 1) if nA > 0 else 0.0

        if pA_lower > 0.5:
            radius = self.sigma * norm.ppf(pA_lower)
            return c_hat, float(radius)
        else:
            return -1, 0.0
