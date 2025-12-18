import torch
from torch import optim, nn

from src.Evaluation.CNN.BasicCNN.Model import CNN
from src.Evaluation.CNN.Train import train
from src.Evaluation.DownloadData import c100_train_loader, c100_val_loader

torch.manual_seed(42)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = CNN(num_classes=100).to(DEVICE)

train(model=model, device=DEVICE, train_loader=c100_train_loader, val_loader=c100_val_loader, path_to_model="../model/CIFAR-100-CNN.pth", log_path="../../log/CIFAR-100-CNN.csv")