import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

from model import BayesianLinear, BayesianNN
from train import train
from data import get_data
from helper import visualize_prediction, visualize_uncertainty

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

X_train, y_train, X_test = get_data(DEVICE=DEVICE)

model = BayesianNN().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
train(model, optimizer, X_train=X_train, y_train=y_train, epochs=1000, num_samples=5)

visualize_prediction(model, X_test=X_test, X_train=X_train, y_train=y_train)
