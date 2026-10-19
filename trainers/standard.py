"""Standard clean Empirical Risk Minimization (ERM) trainer."""

from typing import Tuple
import torch
from torch.amp import autocast
from tqdm import tqdm
from .base import BaseTrainer


class StandardTrainer(BaseTrainer):
    """Standard supervised cross-entropy training."""

    def train_epoch(self, loader: torch.utils.data.DataLoader) -> Tuple[float, float]:
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(loader, desc="Standard Training", leave=False)
        for images, targets in pbar:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast("cuda", enabled=self.amp_enabled and self.device.type == "cuda"):
                outputs = self.model(images)
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
