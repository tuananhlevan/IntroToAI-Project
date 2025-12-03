import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Transition(nn.Module):
    """
    The 'C' in DenseNet-BC.
    Reduces the number of feature maps (compression) and spatial dimensions.
    """
    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = F.avg_pool2d(out, 2)
        return out