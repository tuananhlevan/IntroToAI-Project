import torch
from torch import nn
from torch.nn import functional as F

from blitz.modules import BayesianConv2d, BayesianLinear
from blitz.utils import variational_estimator

@variational_estimator
class CancerPredictionNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Block 1
        # Input: 1 x 256 x 256
        self.bconv1 = BayesianConv2d(1, 16, kernel_size=(5, 5), padding=2) # 16x256x256
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)            # 16x128x128
        
        # Block 2
        self.bconv2 = BayesianConv2d(16, 32, kernel_size=(5, 5), padding=2) # 32x128x128
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)             # 32x64x64
        
        # Block 3
        self.bconv3 = BayesianConv2d(32, 64, kernel_size=(5, 5), padding=2) # 64x64x64
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)             # 64x32x32

        # --- This is the new flattened size ---
        # 64 channels * 32 * 32 features
        self.flat_size = 64 * 32 * 32

        self.bfc1 = BayesianLinear(self.flat_size, 256)
        self.bfc2 = BayesianLinear(256, 2)

    def forward(self, x):
        # We add F.relu after each conv/linear layer
        x = self.pool1(F.relu(self.bconv1(x)))
        x = self.pool2(F.relu(self.bconv2(x)))
        x = self.pool3(F.relu(self.bconv3(x)))
        
        x = torch.flatten(x, 1) # Flatten all dimensions except batch
        
        x = F.relu(self.bfc1(x))
        x = self.bfc2(x) # No relu on the final output
        return x