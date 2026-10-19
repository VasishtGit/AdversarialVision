"""Fast Gradient Sign Method (FGSM) adversarial attack."""

from typing import Optional
import torch
import torch.nn as nn


class FGSMAttack:
    """
    Fast Gradient Sign Method (Goodfellow et al., 2014).
    x_adv = clamp(x + eps * sign(grad_x(L(f(x), y))), 0, 1)
    Operates strictly in [0.0, 1.0] input space.
    """
    def __init__(self, model: nn.Module, eps: float = 4.0 / 255.0, targeted: bool = False):
        self.model = model
        self.eps = eps
        self.targeted = targeted
        self.criterion = nn.CrossEntropyLoss()

    def generate(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Generates adversarial examples for batch x with true/target labels y.
        """
        self.model.zero_grad(set_to_none=True)
        x_adv = x.clone().detach().requires_grad_(True)

        outputs = self.model(x_adv)
        loss = self.criterion(outputs, y)

        if self.targeted:
            loss = -loss

        loss.backward()

        # FGSM step
        grad = x_adv.grad.data
        x_adv = x_adv + self.eps * torch.sign(grad)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

        self.model.zero_grad(set_to_none=True)
        return x_adv.detach()


def fgsm_attack(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float = 4.0 / 255.0,
    targeted: bool = False,
) -> torch.Tensor:
    """Functional interface for FGSM."""
    attacker = FGSMAttack(model=model, eps=eps, targeted=targeted)
    return attacker.generate(x, y)
