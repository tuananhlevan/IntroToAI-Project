import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Bottleneck(nn.Module):
    """
    The 'B' in DenseNet-BC.
    Reduces channels to 4*growth_rate before the expensive 3x3 convolution.
    """

    def __init__(self, in_planes, growth_rate):
        super(Bottleneck, self).__init__()
        # Bottleneck (1x1 conv) -> 4*k output channels
        inter_planes = 4 * growth_rate

        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, bias=False)

        # 3x3 conv -> k output channels
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        # DenseNet Feature: Concatenate input with output
        return torch.cat([x, out], 1)