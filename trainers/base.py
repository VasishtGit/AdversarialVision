"""Base interface for all training strategies in AdversarialVision."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
from torch.amp import GradScaler


class BaseTrainer(ABC):
    """Abstract base class for standard and adversarial trainers."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler: GradScaler,
        amp_enabled: bool = True,
        **kwargs: Any,
    ):
        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.scaler = scaler
        self.amp_enabled = amp_enabled
        self.kwargs = kwargs

    @abstractmethod
    def train_epoch(self, loader: torch.utils.data.DataLoader) -> Tuple[float, float]:
        """
        Executes one training epoch.
        Returns:
            (epoch_loss, epoch_accuracy)
        """
        pass
