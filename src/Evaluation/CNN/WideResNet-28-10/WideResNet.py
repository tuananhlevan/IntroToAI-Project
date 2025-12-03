import torch
from torch import nn
import torch.nn.functional as F
from WideBasicBlock import WideBasicBlock

class WideResNet(nn.Module):
    def __init__(self, depth=28, widen_factor=10, dropout_rate=0.3, num_classes=10):
        super(WideResNet, self).__init__()
        self.in_planes = 16

        # Calculate 'n' (number of blocks per group)
        # Depth should be 6n + 4
        assert ((depth - 4) % 6 == 0), "Depth must be 6n + 4"
        n = (depth - 4) // 6

        # Standard initial convolution
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)

        # The three main groups of blocks
        # 16 -> 160 channels
        self.layer1 = self._make_layer(16 * widen_factor, n, stride=1, dropout_rate=dropout_rate)
        # 160 -> 320 channels
        self.layer2 = self._make_layer(32 * widen_factor, n, stride=2, dropout_rate=dropout_rate)
        # 320 -> 640 channels
        self.layer3 = self._make_layer(64 * widen_factor, n, stride=2, dropout_rate=dropout_rate)

        # Final Batch Norm (needed because the blocks end with conv, not BN)
        self.bn1 = nn.BatchNorm2d(64 * widen_factor)

        # Final Linear Layer
        self.linear = nn.Linear(64 * widen_factor, num_classes)

        # Initialize weights (Optional but recommended)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes, num_blocks, stride, dropout_rate):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(WideBasicBlock(self.in_planes, planes, stride, dropout_rate))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        # Initial Conv
        out = self.conv1(x)

        # Residual Groups
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        # Final Activation and Pooling
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)  # Assumes 32x32 input becomes 8x8 at this stage

        # Flatten and FC
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out