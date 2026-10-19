"""Unit tests for natural image corruptions bounding and shapes."""

import pytest
import numpy as np
import torch
from corruptions import corrupt, CORRUPTIONS_LIST


def test_all_corruptions_numpy_output_bounds():
    img_np = np.random.uniform(0.0, 1.0, (32, 32, 3)).astype(np.float32)

    for c_type in CORRUPTIONS_LIST:
        corrupted = corrupt(img_np, corruption_type=c_type, severity=3)
        assert isinstance(corrupted, np.ndarray)
        assert corrupted.shape == (32, 32, 3), f"{c_type} changed spatial dimensions"
        assert corrupted.min() >= 0.0 - 1e-5, f"{c_type} min value below 0.0: {corrupted.min()}"
        assert corrupted.max() <= 1.0 + 1e-5, f"{c_type} max value above 1.0: {corrupted.max()}"


def test_corruptions_torch_batch():
    batch = torch.rand(4, 3, 32, 32)
    corrupted_batch = corrupt(batch, corruption_type="gaussian_blur", severity=2)

    assert isinstance(corrupted_batch, torch.Tensor)
    assert corrupted_batch.shape == (4, 3, 32, 32)
    assert corrupted_batch.min() >= 0.0
    assert corrupted_batch.max() <= 1.0
