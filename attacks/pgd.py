"""Projected Gradient Descent (PGD) iterative adversarial attack."""

from typing import Optional
import torch
import torch.nn as nn


class PGDAttack:
    """
    Projected Gradient Descent L_infinity attack (Madry et al., 2018).
    x^(t+1) = Proj_{x + S} ( x^(t) + alpha * sign(grad_x(L(f(x), y))) )
    Operates strictly in [0.0, 1.0] input space.
    """
    def __init__(
        self,
        model: nn.Module,
        eps: float = 4.0 / 255.0,
        alpha: float = 1.0 / 255.0,
        steps: int = 10,
        random_start: bool = True,
        targeted: bool = False,
    ):
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.targeted = targeted
        self.criterion = nn.CrossEntropyLoss()

    def generate(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Generates iterative PGD adversarial examples.
        """
        x_adv = x.clone().detach()

        if self.random_start:
            # Random uniform perturbation within [-eps, eps]
            noise = torch.empty_like(x_adv).uniform_(-self.eps, self.eps)
            x_adv = torch.clamp(x_adv + noise, 0.0, 1.0)

        for _ in range(self.steps):
            x_adv.requires_grad_()

            outputs = self.model(x_adv)
            loss = self.criterion(outputs, y)

            if self.targeted:
                loss = -loss

            loss.backward()

            grad = x_adv.grad.data

            # Gradient step
            x_adv = x_adv.detach() + self.alpha * torch.sign(grad)

            # Projection back into L_inf ball [x - eps, x + eps]
            eta = torch.clamp(x_adv - x, min=-self.eps, max=self.eps)
            x_adv = torch.clamp(x + eta, min=0.0, max=1.0)

        return x_adv.detach()


def pgd_attack(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float = 4.0 / 255.0,
    alpha: float = 1.0 / 255.0,
    steps: int = 10,
    random_start: bool = True,
    targeted: bool = False,
) -> torch.Tensor:
    """Functional interface for PGD."""
    attacker = PGDAttack(
        model=model,
        eps=eps,
        alpha=alpha,
        steps=steps,
        random_start=random_start,
        targeted=targeted,
    )
    return attacker.generate(x, y)
