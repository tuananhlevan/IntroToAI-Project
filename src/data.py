import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.datasets import make_moons

torch.manual_seed(42)

def get_data_reg(DEVICE):
    X_train = torch.linspace(-3, 3, 100).unsqueeze(1)
    y_true = torch.sin(X_train) + 0.3 * X_train
    y_train = y_true + 0.1 * torch.randn_like(y_true)
    
    return X_train.to(DEVICE), y_train.to(DEVICE)

def get_data_cls(DEVICE):
    X_train, y_train = make_moons(n_samples=200, noise=0.2, random_state=42)
    
    return torch.tensor(X_train, dtype=torch.float32).to(DEVICE), torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1).to(DEVICE)