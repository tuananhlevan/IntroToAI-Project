import torch
from torch import optim, nn
from torch.nn import functional as F

from BayesModel import BayesNet
from BayesTrain import Bayes_train

from data import c100_train_loader, c100_val_loader

torch.manual_seed(42)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

c100_model = BayesNet(num_classes=100).to(DEVICE)
c100_optimizer = optim.Adam(c100_model.parameters(), lr=1e-3)
c100_criterion = nn.CrossEntropyLoss()
EPOCHS = 50

Bayes_train(model=c100_model, optimizer=c100_optimizer, criterion=c100_criterion, epochs=EPOCHS, device=DEVICE, path_to_model="IntroToAI-Project/model/CIFAR-100-Bayes.pth", train_loader=c100_train_loader, val_loader=c100_val_loader, data_size=40000)
