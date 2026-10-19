"""Fast Adversarial Training with Random Start (Wong et al., ICLR 2020)."""

from typing import Tuple
import torch
import torch.nn as nn
from torch.amp import autocast
from tqdm import tqdm
from .base import BaseTrainer


class FastFGSMTrainer(BaseTrainer):
    """
    Fast adversarial training using FGSM with uniform random initialization.
    Reduces compute cost while guarding against catastrophic overfitting.
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
        alpha: float = 1.25 * 8.0 / 255.0,
        **kwargs,
    ):
        super().__init__(model, device, criterion, optimizer, scaler, amp_enabled, **kwargs)
        self.eps = eps
        self.alpha = alpha

    def _generate_fast_fgsm(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_adv = x.clone().detach()
        noise = torch.empty_like(x_adv).uniform_(-self.eps, self.eps)
        x_adv = torch.clamp(x_adv + noise, 0.0, 1.0).requires_grad_(True)

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

        pbar = tqdm(loader, desc=f"Fast-FGSM-AT (eps={self.eps*255:.1f}/255)", leave=False)
        for images, targets in pbar:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.model.eval()
            x_adv = self._generate_fast_fgsm(images, targets)
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
