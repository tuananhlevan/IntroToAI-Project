import torch
from torch import optim, nn

from src.Evaluation.CNN.Model import CNN
from src.Evaluation.CNN.Train import train
from src.Evaluation.DownloadData import c10_train_loader, c10_val_loader

torch.manual_seed(42)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = CNN(num_classes=10).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
EPOCHS = 50

train(model=model, optimizer=optimizer, criterion=criterion, epochs=EPOCHS, device=DEVICE, train_loader=c10_train_loader, val_loader=c10_val_loader, path_to_model="model/CIFAR-10-CNN.pth")