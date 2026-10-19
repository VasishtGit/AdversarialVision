"""AdversarialVision: Vision Model Robustness & Defense Framework.

Provides modular packages for:
- attacks: FGSM, PGD, AutoAttack
- corruptions: 10 natural image corruptions across 5 severities
- defenses: Preprocessing filters, certified randomized smoothing, adversarial detection
- evaluation: Clean, corrupted, adversarial, and transferability benchmarks
- frequency: 2D FFT & 2D DCT spectral decompositions and Fourier probes
- models: CIFAR-adapted ResNet and WideResNet with on-GPU NormalizeLayer
- trainers: Standard ERM, PGD-AT, Fast-FGSM, and TRADES robust training
- data: CIFAR-100 and CIFAR-10 data loaders
"""

import attacks
import corruptions
import defenses
import evaluation
import frequency
import models
import trainers
import data

__version__ = "0.1.0"

__all__ = [
    "attacks",
    "corruptions",
    "defenses",
    "evaluation",
    "frequency",
    "models",
    "trainers",
    "data",
    "__version__",
]
