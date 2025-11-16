import torch
from torch import optim, nn
from torch.nn import functional as F

from CNNModel import CNN
from CNNTrain import CNN_train

from data import c100_train_loader, c100_val_loader

torch.manual_seed(42)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

c100_model = CNN(num_classes=100).to(DEVICE)
c100_optimizer = optim.Adam(c100_model.parameters(), lr=1e-3)
c100_criterion = nn.CrossEntropyLoss()
EPOCHS = 50

CNN_train(model=c100_model, optimizer=c100_optimizer, criterion=c100_criterion, epochs=EPOCHS, device=DEVICE, train_loader=c100_train_loader, val_loader=c100_val_loader, path_to_model="IntroToAI-Project/model/CIFAR-100-CNN.pth")