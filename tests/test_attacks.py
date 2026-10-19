"""Unit tests for adversarial attacks constraints and validity."""

import pytest
import torch
import torch.nn as nn
from attacks.fgsm import FGSMAttack, fgsm_attack
from attacks.pgd import PGDAttack, pgd_attack


class DummyModel(nn.Module):
    def __init__(self, in_features=3 * 32 * 32, num_classes=10):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        flat = x.view(x.size(0), -1)
        return self.fc(flat)


def test_fgsm_linf_bound():
    model = DummyModel()
    eps = 8.0 / 255.0
    attacker = FGSMAttack(model=model, eps=eps)

    x = torch.rand(4, 3, 32, 32)
    y = torch.tensor([0, 1, 2, 3])

    x_adv = attacker.generate(x, y)

    # 1. Output range [0, 1]
    assert x_adv.min() >= 0.0, "Adversarial example below 0.0"
    assert x_adv.max() <= 1.0, "Adversarial example above 1.0"

    # 2. Perturbation strictly within epsilon bound
    perturbation = (x_adv - x).abs()
    max_linf = perturbation.max().item()
    assert max_linf <= eps + 1e-6, f"L_inf perturbation {max_linf} exceeded eps {eps}"


def test_pgd_linf_bound_and_steps():
    model = DummyModel()
    eps = 8.0 / 255.0
    alpha = 2.0 / 255.0
    steps = 5
    attacker = PGDAttack(model=model, eps=eps, alpha=alpha, steps=steps, random_start=True)

    x = torch.rand(4, 3, 32, 32)
    y = torch.tensor([1, 2, 3, 4])

    x_adv = attacker.generate(x, y)

    assert x_adv.min() >= 0.0
    assert x_adv.max() <= 1.0
    max_linf = (x_adv - x).abs().max().item()
    assert max_linf <= eps + 1e-6, f"PGD L_inf perturbation {max_linf} exceeded eps {eps}"
