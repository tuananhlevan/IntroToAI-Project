import torch
from torch import nn

from src.UncertaintyVisualization.BayesianLinear.BayesianLinear import BayesianLinear


class BayesianNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = BayesianLinear(1, 20)
        self.layer_2 = BayesianLinear(20, 20)
        self.layer_3 = BayesianLinear(20, 1)
    
    def forward(self, x, sample=True):
        h = torch.tanh(self.layer_1(x, sample=sample))
        h = torch.tanh(self.layer_2(h, sample=sample))
        out = self.layer_3(h, sample=sample)
        return out
    
    def kl_loss(self):
        return self.layer_1.kl_loss() + self.layer_2.kl_loss() + self.layer_3.kl_loss()