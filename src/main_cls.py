import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

from model import BayesianLinear, BayesianNN_reg, BayesianNN_cls
from train import train_cls
from data import get_data_reg, get_data_cls
from helper import visualize_prediction_cls, visualize_uncertainty

# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")

X_train, y_train = get_data_cls(DEVICE=DEVICE)

model_cls = BayesianNN_cls().to(DEVICE)
optimizer = optim.Adam(model_cls.parameters(), lr=1e-3)
train_cls(model_cls, optimizer, X_train=X_train, y_train=y_train, epochs=1000, num_samples=5)
print(torch.tensor(model_cls(X_train)))

visualize_prediction_cls(model_cls, X_train=X_train, y_train=y_train)
