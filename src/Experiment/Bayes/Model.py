import torch
from blitz.modules import BayesianConv2d, BayesianLinear
from blitz.utils import variational_estimator
from torch import nn
from torch.nn import functional as F
from torch.nn.modules import Conv2d, Linear, MaxPool2d

@variational_estimator
class BayesNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        # Block 1
        # Input: 3 x 32 x 32
        self.conv1 = Conv2d(3, 16, kernel_size=(3, 3), padding=1)  # 16x32x32
        self.pool1 = MaxPool2d(kernel_size=(2, 2), stride=2)  # 16x16x16

        # Block 2
        self.conv2 = Conv2d(16, 32, kernel_size=(3, 3), padding=1)  # 32x16x16
        self.pool2 = MaxPool2d(kernel_size=(2, 2), stride=2)  # 32x8x8

        # --- This is the new flattened size ---
        # 32 channels * 8 * 8 features
        self.flat_size = 32 * 8 * 8

        self.fc = Linear(self.flat_size, 256)
        self.bfc = BayesianLinear(
            in_features=256,
            out_features=num_classes,
            prior_sigma_1=1.0,
            prior_sigma_2=0.002,
            prior_pi=0.5,
            posterior_mu_init=0.0,
            posterior_rho_init=-3.0,
        )

    def forward(self, x):
        # We add F.relu after each conv/linear layer
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))

        x = torch.flatten(x, 1)  # Flatten all dimensions except batch

        x = F.relu(self.fc(x))
        x = self.bfc(x)  # No relu on the final output
        return x