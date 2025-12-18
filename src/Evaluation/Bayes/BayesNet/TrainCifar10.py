import torch

from src.Evaluation.Bayes.BayesNet.Model import BayesNet
from src.Evaluation.Bayes.Train import train
from src.Evaluation.DownloadData import c10_train_loader, c10_val_loader

torch.manual_seed(42)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = BayesNet(num_classes=10).to(DEVICE)

train(model=model, device=DEVICE, path_to_model="../model/CIFAR-10-Bayes.pth", log_path="../../log/CIFAR-10-Bayes.csv", train_loader=c10_train_loader, val_loader=c10_val_loader)