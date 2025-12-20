import torch
from blitz.modules import BayesianConv2d, BayesianLinear
from blitz.utils import variational_estimator
from torch import nn
from torch.nn import functional as F
from torch.nn.modules import Conv2d, Linear, MaxPool2d


@variational_estimator
class BayesNet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        # Block 1
        # Input: 3 x 224 x 224
        self.conv1 = Conv2d(3, 16, kernel_size=(3, 3), padding=1)  # 16 x 224 x 224
        self.pool1 = MaxPool2d(kernel_size=(2, 2), stride=2)  # 16 x 112 x 112

        # Block 2
        self.conv2 = Conv2d(16, 32, kernel_size=(3, 3), padding=1)  # 32 x 112 x 112
        self.pool2 = MaxPool2d(kernel_size=(2, 2), stride=2)  # 32 x 56 x 56

        # Block 3
        self.conv3 = Conv2d(32, 64, kernel_size=(3, 3), padding=1)  # 64 x 56 x 56
        self.pool3 = MaxPool2d(kernel_size=(2, 2), stride=2)  # 64 x 28 x 28

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.bfc = BayesianLinear(
            in_features=64,
            out_features=num_classes,
            prior_sigma_1=1.,
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

        x = self.gap(x)
        x = x.view(x.size(0), -1)

        x = self.bfc(x)  # No relu on the final output
        return x