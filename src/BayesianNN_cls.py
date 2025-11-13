import torch
from torch import nn
import torch.nn.functional as F
from BayesianLinear import BayesianLinear

class BayesianNN_cls(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = BayesianLinear(2, 20)
        self.layer_2 = BayesianLinear(20, 20)
        self.layer_3 = BayesianLinear(20, 1)
        
    def forward(self, x, sample=True):
        h = F.softplus(self.layer_1(x, sample=sample))
        h = F.softplus(self.layer_2(h, sample=sample))
        out = self.layer_3(h, sample=sample)
        return torch.sigmoid(out)

    def kl_loss(self):
        return self.layer_1.kl_loss() + self.layer_2.kl_loss() + self.layer_3.kl_loss()