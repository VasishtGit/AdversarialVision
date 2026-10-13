"""Adversarial attacks implementation (FGSM, PGD)."""

from .fgsm import FGSMAttack, fgsm_attack
from .pgd import PGDAttack, pgd_attack

__all__ = ["FGSMAttack", "fgsm_attack", "PGDAttack", "pgd_attack"]
