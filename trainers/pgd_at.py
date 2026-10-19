"""Projected Gradient Descent Adversarial Trainer (Madry et al., 2018)."""

from typing import Tuple
import torch
import torch.nn as nn
from torch.amp import autocast
from tqdm import tqdm
from .base import BaseTrainer


class PGDAdversarialTrainer(BaseTrainer):
    """
    Min-Max robust optimization via multi-step PGD.
    Generates online adversarial perturbations during the training loop.
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
        **kwargs,
    ):
        super().__init__(model, device, criterion, optimizer, scaler, amp_enabled, **kwargs)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps

    def _generate_pgd(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_adv = x.clone().detach()
        noise = torch.empty_like(x_adv).uniform_(-self.eps, self.eps)
        x_adv = torch.clamp(x_adv + noise, 0.0, 1.0)

        for _ in range(self.steps):
            x_adv.requires_grad_()
            self.model.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=self.amp_enabled and self.device.type == "cuda"):
                outputs = self.model(x_adv)
                loss = self.criterion(outputs, y)
            loss.backward()

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

        pbar = tqdm(loader, desc=f"PGD-AT (eps={self.eps*255:.1f}/255)", leave=False)
        for images, targets in pbar:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            # Prevent BN statistics contamination during inner attack loop
            self.model.eval()
            x_adv = self._generate_pgd(images, targets)
            self.model.train()

            self.optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=self.amp_enabled and self.device.type == "cuda"):
                outputs = self.model(x_adv)
                loss = self.criterion(outputs, targets)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss.item() * targets.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(targets).sum().item()
            total += targets.size(0)

            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{100.0 * correct / total:.2f}%"})

        return running_loss / total, 100.0 * correct / total
