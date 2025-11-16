import torch
from torch import optim, nn
from torch.nn import functional as F

from CNNModel import CNN
from CNNTrain import CNN_train

from data import c10_train_loader, c10_val_loader

torch.manual_seed(42)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

c10_model = CNN(num_classes=10).to(DEVICE)
c10_optimizer = optim.Adam(c10_model.parameters(), lr=1e-3)
c10_criterion = nn.CrossEntropyLoss()
EPOCHS = 50

CNN_train(model=c10_model, optimizer=c10_optimizer, criterion=c10_criterion, epochs=EPOCHS, device=DEVICE, train_loader=c10_train_loader, val_loader=c10_val_loader, path_to_model="IntroToAI-Project/model/CIFAR-10-CNN.pth")