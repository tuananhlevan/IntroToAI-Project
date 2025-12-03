import torch
from torch import nn, optim

from DenseNet import DenseNet
from src.Evaluation.CNN.Train import train
from src.Evaluation.DownloadData import c100_train_loader, c100_val_loader

torch.manual_seed(42)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = DenseNet(num_classes=100).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
EPOCHS = 50

train(model=model, optimizer=optimizer, criterion=criterion, epochs=EPOCHS, device=DEVICE, train_loader=c100_train_loader, val_loader=c100_val_loader, path_to_model="../model/CIFAR-100-DenseNet.pth")