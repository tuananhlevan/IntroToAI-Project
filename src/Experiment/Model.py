import torch
from blitz.modules import BayesianConv2d, BayesianLinear
from blitz.utils import variational_estimator
from torch import nn
from torch.nn import functional as F
from torch.nn.modules import Conv2d, Linear, MaxPool2d


@variational_estimator
class BayesNet(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        # Block 1
        # Input: 1 x 128 x 128
        self.conv1 = Conv2d(1, 16, kernel_size=(3, 3), padding=1)  # 16 x 128 x 128
        self.pool1 = MaxPool2d(kernel_size=(2, 2), stride=2)  # 16 x 64 x 64

        # Block 2
        self.conv2 = Conv2d(16, 32, kernel_size=(3, 3), padding=1)  # 32 x 64 x 64
        self.pool2 = MaxPool2d(kernel_size=(2, 2), stride=2)  # 32 x 32 x 32

        # Block 3
        self.conv3 = Conv2d(32, 64, kernel_size=(3, 3), padding=1)  # 64 x 32 x 32
        self.pool3 = MaxPool2d(kernel_size=(2, 2), stride=2)  # 64 x 16 x 16

        # Block 4
        self.conv4 = Conv2d(64, 128, kernel_size=(3, 3), padding=1)  # 128 x 16 x 16
        self.pool4 = MaxPool2d(kernel_size=(2, 2), stride=2)  # 128 x 8 x 8

        # --- This is the new flattened size ---
        # 128 channels * 8 * 8 features
        self.flat_size = 128 * 8 * 8

        self.fc = Linear(self.flat_size, 512)
        self.bfc = BayesianLinear(
            in_features=512,
            out_features=num_classes,
            prior_sigma_1=1.5,
            prior_sigma_2=0.002,
            prior_pi=0.5,
            posterior_mu_init=0.0,
            posterior_rho_init=-3.0,
        )

    def forward(self, x):
        # We add F.relu after each conv/linear layer
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.pool4(F.relu(self.conv4(x)))

        x = torch.flatten(x, 1)  # Flatten all dimensions except batch

        x = F.relu(self.fc(x))
        x = self.bfc(x)  # No relu on the final output
        return x