"""TRADES Adversarial Trainer (Zhang et al., ICML 2019).

Optimizes:
    L_TRADES = CE(f(x), y) + beta * KL( f(x) || f(x_adv) )
Balances clean accuracy and adversarial robustness trade-off.
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from tqdm import tqdm
from .base import BaseTrainer


class TRADESTrainer(BaseTrainer):
    """
    TRADES (Theoretically Principled Trade-off between Robustness and Accuracy).
    Minimizes classification error on natural examples while penalizing divergence
    between predictions on natural and adversarially perturbed inputs.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler: torch.amp.GradScaler,
        amp_enabled: bool = True,
        eps: float = 8.0 / 255.0,
        alpha: float = 2.0 / 255.0,
        steps: int = 7,
        beta: float = 6.0,
        **kwargs,
    ):
        super().__init__(model, device, criterion, optimizer, scaler, amp_enabled, **kwargs)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.beta = beta
        self.kl_criterion = nn.KLDivLoss(reduction="batchmean")

    def _generate_trades_adv(self, x: torch.Tensor) -> torch.Tensor:
        """Find perturbation maximizing KL divergence with respect to model(x)."""
        x_adv = x.clone().detach() + 0.001 * torch.randn_like(x)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

        with torch.no_grad():
            natural_logits = self.model(x)
            natural_probs = F.softmax(natural_logits, dim=1)

        for _ in range(self.steps):
            x_adv.requires_grad_()
            self.model.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=self.amp_enabled and self.device.type == "cuda"):
                adv_logits = self.model(x_adv)
                adv_log_probs = F.log_softmax(adv_logits, dim=1)
                kl_loss = self.kl_criterion(adv_log_probs, natural_probs)
            kl_loss.backward()

            grad = x_adv.grad.data
            x_adv = x_adv.detach() + self.alpha * torch.sign(grad)
            eta = torch.clamp(x_adv - x, min=-self.eps, max=self.eps)
            x_adv = torch.clamp(x + eta, min=0.0, max=1.0)

        self.model.zero_grad(set_to_none=True)
        return x_adv.detach()

    def train_epoch(self, loader: torch.utils.data.DataLoader) -> Tuple[float, float]:
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(loader, desc=f"TRADES (beta={self.beta:.1f}, eps={self.eps*255:.1f}/255)", leave=False)
        for images, targets in pbar:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.model.eval()
            x_adv = self._generate_trades_adv(images)
            self.model.train()

            self.optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=self.amp_enabled and self.device.type == "cuda"):
                natural_logits = self.model(images)
                adv_logits = self.model(x_adv)

                ce_loss = self.criterion(natural_logits, targets)
                natural_probs = F.softmax(natural_logits, dim=1)
                adv_log_probs = F.log_softmax(adv_logits, dim=1)
                kl_loss = self.kl_criterion(adv_log_probs, natural_probs)

                total_loss = ce_loss + self.beta * kl_loss

            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += total_loss.item() * targets.size(0)
            _, preds = natural_logits.max(1)
            correct += preds.eq(targets).sum().item()
            total += targets.size(0)

            pbar.set_postfix({"ce": f"{ce_loss.item():.3f}", "kl": f"{kl_loss.item():.3f}", "acc": f"{100.0 * correct / total:.2f}%"})

        return running_loss / total, 100.0 * correct / total
