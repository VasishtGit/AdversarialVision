"""WideResNet for CIFAR (Zagoruyko & Komodakis, BMVC 2016).
Standard architecture for adversarial robustness benchmarking on CIFAR datasets.
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.cifar100 import CIFAR100_MEAN, CIFAR100_STD
from .resnet_cifar import NormalizeLayer


class WideBasicBlock(nn.Module):
    def __init__(self, in_planes: int, planes: int, dropout_rate: float, stride: int = 1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=False)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(F.relu(self.bn1(x), inplace=True))
        out = self.dropout(out)
        out = self.conv2(F.relu(self.bn2(out), inplace=True))
        out += self.shortcut(x)
        return out


class WideResNet(nn.Module):
    """
    WideResNet-28-10 / WideResNet-34-10 architecture for CIFAR.
    Incorporates differentiable on-GPU normalization.
    """

    def __init__(
        self,
        depth: int = 28,
        widen_factor: int = 10,
        dropout_rate: float = 0.0,
        num_classes: int = 100,
        normalize: bool = True,
        mean: Tuple[float, float, float] = CIFAR100_MEAN,
        std: Tuple[float, float, float] = CIFAR100_STD,
    ):
        super().__init__()
        assert (depth - 4) % 6 == 0, "Depth should be 6n + 4"
        n = (depth - 4) // 6

        self.normalize_layer = NormalizeLayer(mean, std) if normalize else nn.Identity()

        k = widen_factor
        n_stages = [16, 16 * k, 32 * k, 64 * k]

        self.conv1 = nn.Conv2d(3, n_stages[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._wide_layer(WideBasicBlock, n_stages[1], n, dropout_rate, stride=1, in_planes=n_stages[0])
        self.layer2 = self._wide_layer(WideBasicBlock, n_stages[2], n, dropout_rate, stride=2, in_planes=n_stages[1])
        self.layer3 = self._wide_layer(WideBasicBlock, n_stages[3], n, dropout_rate, stride=2, in_planes=n_stages[2])
        self.bn1 = nn.BatchNorm2d(n_stages[3], momentum=0.9)
        self.linear = nn.Linear(n_stages[3], num_classes)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                m.bias.data.zero_()

    def _wide_layer(self, block, planes: int, num_blocks: int, dropout_rate: float, stride: int, in_planes: int):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        curr_in = in_planes
        for s in strides:
            layers.append(block(curr_in, planes, dropout_rate, s))
            curr_in = planes
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.normalize_layer(x)
        out = self.conv1(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out), inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        return self.linear(out)


def wideresnet28_10(num_classes: int = 100, normalize: bool = True, **kwargs) -> WideResNet:
    """WideResNet-28-10 (standard adversarial robustness benchmark model)."""
    return WideResNet(depth=28, widen_factor=10, num_classes=num_classes, normalize=normalize, **kwargs)


def wideresnet34_10(num_classes: int = 100, normalize: bool = True, **kwargs) -> WideResNet:
    """WideResNet-34-10."""
    return WideResNet(depth=34, widen_factor=10, num_classes=num_classes, normalize=normalize, **kwargs)
