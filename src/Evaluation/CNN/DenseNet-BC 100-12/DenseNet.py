import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from BottleNeck import Bottleneck
from Transition import Transition

class DenseNet(nn.Module):
    def __init__(self, depth=100, growth_rate=12, reduction=0.5, num_classes=10):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate

        # Calculate number of layers per block
        # Formula: (Depth - 4) / 6
        # The '6' comes from: 3 blocks * 2 convs per bottleneck layer
        num_blocks = (depth - 4) // 6

        # Initial Convolution
        # For BC variants, input planes usually start at 2 * growth_rate
        num_planes = 2 * growth_rate
        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=False)

        # --- Block 1 ---
        self.dense1 = self._make_dense_layers(num_planes, num_blocks)
        num_planes += num_blocks * growth_rate
        # Transition 1
        out_planes = int(math.floor(num_planes * reduction))
        self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes

        # --- Block 2 ---
        self.dense2 = self._make_dense_layers(num_planes, num_blocks)
        num_planes += num_blocks * growth_rate
        # Transition 2
        out_planes = int(math.floor(num_planes * reduction))
        self.trans2 = Transition(num_planes, out_planes)
        num_planes = out_planes

        # --- Block 3 ---
        self.dense3 = self._make_dense_layers(num_planes, num_blocks)
        num_planes += num_blocks * growth_rate

        # Final Batch Norm
        self.bn = nn.BatchNorm2d(num_planes)

        # Linear Layer
        self.linear = nn.Linear(num_planes, num_classes)

        # Weight Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def _make_dense_layers(self, in_planes, nblock):
        layers = []
        for i in range(nblock):
            # Each layer takes 'in_planes + i * growth_rate' channels
            # and outputs 'growth_rate' channels, which are concatenated.
            layers.append(Bottleneck(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)

        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.dense3(out)

        out = F.relu(self.bn(out))
        out = F.avg_pool2d(out, 8)  # Assumes 32x32 input

        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out