import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # Variational parameters
        self.mu_w = nn.Parameter(torch.zeros(out_features, in_features))
        self.rho_w = nn.Parameter(torch.ones(out_features, in_features) * -3)
        self.mu_b = nn.Parameter(torch.zeros(out_features))
        self.rho_b = nn.Parameter(torch.ones(out_features) * -3)

        # Prior (standard normal)
        self.prior_mu = 0.
        self.prior_sigma = 1.
        
    def sample_weights(self):
        epsilon_w = torch.randn_like(self.mu_w)
        epsilon_b = torch.randn_like(self.mu_b)
        sigma_w = torch.log1p(torch.exp(self.rho_w))
        sigma_b = torch.log1p(torch.exp(self.rho_b))
        w = self.mu_w + sigma_w * epsilon_w
        b = self.mu_b + sigma_b * epsilon_b
        return w, b
    
    def forward(self, x, sample=True):
        if sample:
            w, b = self.sample_weights()
        else:
            w, b = self.mu_w, self.mu_b
        return F.linear(x, w, b)
    
    def kl_loss(self):
        # KL(q||p) for Gaussian distributions
        sigma_w = torch.log1p(torch.exp(self.rho_w))
        sigma_b = torch.log1p(torch.exp(self.rho_b))

        kl_w = (
            torch.log(self.prior_sigma / sigma_w)
            + (sigma_w ** 2 + (self.mu_w - self.prior_mu) ** 2) / (2 * self.prior_sigma ** 2)
            - 0.5
        ).sum()

        kl_b = (
            torch.log(self.prior_sigma / sigma_b)
            + (sigma_b ** 2 + (self.mu_b - self.prior_mu) ** 2) / (2 * self.prior_sigma ** 2)
            - 0.5
        ).sum()

        return kl_w + kl_b

class BayesianNN_reg(nn.Module):
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