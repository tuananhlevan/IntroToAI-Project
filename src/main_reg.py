import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

from model import BayesianLinear, BayesianNN_reg, BayesianNN_cls
from train import train_reg
from data import get_data_reg, get_data_cls
from helper import visualize_prediction_reg, visualize_uncertainty

# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")

X_train, y_train = get_data_reg(DEVICE=DEVICE)

model_reg = BayesianNN_reg().to(DEVICE)
optimizer = optim.Adam(model_reg.parameters(), lr=1e-3)
train_reg(model_reg, optimizer, X_train=X_train, y_train=y_train, epochs=1000, num_samples=5)

visualize_prediction_reg(model_reg, X_train=X_train, y_train=y_train)
