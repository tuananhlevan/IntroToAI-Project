import torch
import torch.nn as nn
import torch.nn.functional as F


class WideBasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1, dropout_rate=0.0):
        super(WideBasicBlock, self).__init__()

        # 1. First set of BN -> ReLU -> Conv
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=False)

        # 2. Dropout (applied between the two convolutions)
        self.dropout = nn.Dropout(p=dropout_rate)

        # 3. Second set of BN -> ReLU -> Conv
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)

        # 4. Shortcut connection
        # If input shape != output shape (stride != 1 or channel count changes),
        # we need a 1x1 convolution to match dimensions for the addition.
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        # Pre-activation: BN -> ReLU
        out = F.relu(self.bn1(x))
        # First Conv
        out = self.conv1(out)

        # Pre-activation 2
        out = F.relu(self.bn2(out))
        # Dropout
        out = self.dropout(out)
        # Second Conv
        out = self.conv2(out)

        # Add Shortcut
        out += self.shortcut(x)
        return out