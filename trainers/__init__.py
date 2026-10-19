"""Training strategies registry for AdversarialVision."""

from typing import Dict, Type
from .base import BaseTrainer
from .standard import StandardTrainer
from .pgd_at import PGDAdversarialTrainer
from .fast_fgsm import FastFGSMTrainer
from .trades import TRADESTrainer

TRAINING_REGISTRY: Dict[str, Type[BaseTrainer]] = {
    "standard": StandardTrainer,
    "pgd": PGDAdversarialTrainer,
    "pgd_at": PGDAdversarialTrainer,
    "fast_fgsm": FastFGSMTrainer,
    "trades": TRADESTrainer,
}

__all__ = [
    "BaseTrainer",
    "StandardTrainer",
    "PGDAdversarialTrainer",
    "FastFGSMTrainer",
    "TRADESTrainer",
    "TRAINING_REGISTRY",
]
