import torch
from torch import optim, nn
from torch.nn import functional as F

from BayesModel import CancerPredictionNet
from BayesTrain import Bayes_train
from CNNTrain import CNN_train

torch.manual_seed(42)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = CancerPredictionNet().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
EPOCHS = 10

Bayes_train(model=model, optimizer=optimizer, criterion=criterion, epochs=EPOCHS, device=DEVICE)
