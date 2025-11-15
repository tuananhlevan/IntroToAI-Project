import torch
import torch.optim as optim

from BayesianNN_cls import BayesianNN_cls
from train import train_cls
from data import get_data_cls
from visualize import visualize_prediction_cls, visualize_uncertainty_cls

# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")

X_train, y_train = get_data_cls(DEVICE=DEVICE)

model_cls = BayesianNN_cls().to(DEVICE)
optimizer = optim.Adam(model_cls.parameters(), lr=1e-3)
train_cls(model_cls, optimizer, X_train=X_train, y_train=y_train, epochs=5000)

visualize_prediction_cls(model_cls, X_train=X_train, y_train=y_train)
visualize_uncertainty_cls(model_cls, X_train=X_train, y_train=y_train)
