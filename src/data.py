import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def get_data(DEVICE):
    torch.manual_seed(42)
    X_train = torch.linspace(-3, 3, 100).unsqueeze(1)
    y_true = torch.sin(X_train) + 0.3 * X_train
    y_train = y_true + 0.1 * torch.randn_like(y_true)
    
    X_test = torch.linspace(-3, 3, 200).unsqueeze(1)

    return X_train.to(DEVICE), y_train.to(DEVICE), X_test.to(DEVICE)