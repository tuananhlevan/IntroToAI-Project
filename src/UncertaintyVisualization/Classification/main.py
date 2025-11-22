import torch
import torch.optim as optim

from BayesianNN import BayesianNN
from data import get_data
from train import train
from visualize import visualize_prediction, visualize_uncertainty

# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")

X_train, y_train = get_data(DEVICE=DEVICE)

model = BayesianNN().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
train(model, optimizer, X_train=X_train, y_train=y_train, epochs=5000)

visualize_prediction(model, X_train=X_train, y_train=y_train)
visualize_uncertainty(model, X_train=X_train, y_train=y_train)
