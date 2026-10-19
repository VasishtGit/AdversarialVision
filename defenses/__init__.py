"""Defenses package for AdversarialVision.
Provides input preprocessing, certified randomized smoothing, detection, and composite defenses.
"""

from typing import List, Union
import torch
import numpy as np

from .preprocessing import (
    DEFENSES_REGISTRY,
    PreprocessDefense,
    def_gaussian_filter,
    def_median_filter,
    def_bilateral_filter,
    def_jpeg_recompression,
)
from .certified import RandomizedSmoothing
from .detection import EnergyBasedDetector, compute_lid


class CompositeDefense:
    """
    Chains multiple preprocessing defense transformations sequentially.
    Operates strictly in physical pixel range [0.0, 1.0].
    """

    def __init__(self, defenses: List[Union[str, PreprocessDefense]]):
        self.pipeline: List[PreprocessDefense] = []
        for d in defenses:
            if isinstance(d, str):
                self.pipeline.append(PreprocessDefense(defense_type=d))
            elif isinstance(d, PreprocessDefense):
                self.pipeline.append(d)
            else:
                raise TypeError(f"Expected str or PreprocessDefense, got {type(d)}")

    def __call__(self, image: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        out = image
        for defense in self.pipeline:
            out = defense(out)
        return out


__all__ = [
    "DEFENSES_REGISTRY",
    "PreprocessDefense",
    "def_gaussian_filter",
    "def_median_filter",
    "def_bilateral_filter",
    "def_jpeg_recompression",
    "RandomizedSmoothing",
    "EnergyBasedDetector",
    "compute_lid",
    "CompositeDefense",
]
