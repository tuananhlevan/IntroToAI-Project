import torch
import torch.optim as optim

from data import get_data
from predict import predict
from src.UncertaintyVisualization.Regression.BayesianNN import BayesianNN
from train import train
from visualize import visualize_prediction

# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")

X_train, y_train = get_data(DEVICE=DEVICE)

model = BayesianNN().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
train(model, optimizer, X_train=X_train, y_train=y_train, epochs=5000, num_samples=5)
mean_pred, std_pred = predict(model=model, X_train=X_train)

visualize_prediction(model, X_train=X_train, y_train=y_train, mean_pred=mean_pred, std_pred=std_pred)