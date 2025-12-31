from blitz.modules import BayesianLinear
from blitz.utils import variational_estimator
from torch import nn
from torch.nn import MultiheadAttention
from torch.nn import functional as F
from torchvision import models

@variational_estimator
class BayesMobileNet(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        base_model = models.mobilenet_v2(weights="DEFAULT")
        self.features = base_model.features
        self.features.requires_grad_(False)

        height, width, channels = 7, 7, 1280

        self.multihead_attn = MultiheadAttention(num_heads=8, embed_dim=channels, batch_first=True)
        self.dropout = nn.Dropout(0.25)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_features=channels, out_features=512)
        self.batch_norm = nn.BatchNorm1d(num_features=512)

        self.bfc = BayesianLinear(
            in_features=512,
            out_features=num_classes,
            prior_sigma_1=1.,
            prior_sigma_2=0.002,
            prior_pi=0.5,
            posterior_mu_init=0.0,
            posterior_rho_init=-3.0,
        )

    def forward(self, x):
        x = self.features(x)

        B, C, H, W = x.shape
        x = x.view(B, C, H * W).permute(0, 2, 1)

        x, _ = self.multihead_attn(x, x, x)
        x = x.permute(0, 2, 1).view(B, C, H, W)
        x = self.dropout(x)

        x = self.pool(x).flatten(1)
        x = F.relu(self.fc(x))
        x = self.batch_norm(x)

        x = self.bfc(x)  # No relu on the final output

        return x