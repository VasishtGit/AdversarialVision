"""CIFAR-tailored ResNet-18 implementation with differentiable input normalization."""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.cifar100 import CIFAR100_MEAN, CIFAR100_STD


class NormalizeLayer(nn.Module):
    """
    Differentiable on-GPU normalization layer.
    Transforms input tensors x in [0.0, 1.0] to normalized space (x - mean) / std.
    Ensures attacks compute gradients with respect to physical pixel bounds [0, 1].
    """
    def __init__(self, mean: Tuple[float, float, float] = CIFAR100_MEAN, std: Tuple[float, float, float] = CIFAR100_STD):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out, inplace=True)
        return out


class ResNet_CIFAR(nn.Module):
    """
    CIFAR-adapted ResNet architecture.
    Key differences from standard ImageNet ResNet:
    - 3x3 conv1 with stride 1 and padding 1 (instead of 7x7 stride 2)
    - No initial max pooling (preserves 32x32 spatial resolution into block 1)
    - Integrated on-GPU differentiable normalization
    """
    def __init__(
        self,
        block: type = BasicBlock,
        num_blocks: Tuple[int, int, int, int] = (2, 2, 2, 2),
        num_classes: int = 100,
        normalize: bool = True,
        mean: Tuple[float, float, float] = CIFAR100_MEAN,
        std: Tuple[float, float, float] = CIFAR100_STD,
    ):
        super().__init__()
        self.in_planes = 64
        self.normalize = NormalizeLayer(mean, std) if normalize else nn.Identity()

        # Initial conv: 32x32x3 -> 32x32x64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Residual stages
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)   # 32x32
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)  # 16x16
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)  # 8x8
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)  # 4x4

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self._init_weights()

    def _make_layer(self, block: type, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0.0)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts penultimate 512-dim embedding (for detection/latent feature analysis)."""
        x = self.normalize(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        return torch.flatten(out, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(x)
        logits = self.fc(features)
        return logits


def resnet18_cifar(num_classes: int = 100, normalize: bool = True, **kwargs) -> ResNet_CIFAR:
    """Constructs a CIFAR-style ResNet-18 model."""
    return ResNet_CIFAR(
        block=BasicBlock,
        num_blocks=(2, 2, 2, 2),
        num_classes=num_classes,
        normalize=normalize,
        **kwargs,
    )


if __name__ == "__main__":
    model = resnet18_cifar(num_classes=100)
    dummy_input = torch.rand(4, 3, 32, 32)
    output = model(dummy_input)
    features = model.forward_features(dummy_input)
    total_params = sum(p.numel() for p in model.parameters())

    print(f"ResNet-18 CIFAR Architecture Initialized Successfully.")
    print(f"Total Parameters:    {total_params:,}")
    print(f"Input Shape:         {tuple(dummy_input.shape)}")
    print(f"Features Shape:      {tuple(features.shape)}")
    print(f"Logits Output Shape: {tuple(output.shape)}")
