import torch

torch.manual_seed(42)

def get_data(DEVICE):
    X_train = torch.linspace(-3, 3, 100).unsqueeze(1)
    y_true = torch.sin(X_train) + 0.3 * X_train
    y_train = y_true + 0.1 * torch.randn_like(y_true)
    
    return X_train.to(DEVICE), y_train.to(DEVICE)

