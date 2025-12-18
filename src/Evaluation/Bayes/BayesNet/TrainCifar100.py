import torch

from src.Evaluation.Bayes.BayesNet.Model import BayesNet
from src.Evaluation.Bayes.Train import train
from src.Evaluation.DownloadData import c100_train_loader, c100_val_loader

torch.manual_seed(42)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = BayesNet(num_classes=100).to(DEVICE)

train(model=model, device=DEVICE, path_to_model="../model/CIFAR-100-Bayes.pth", log_path="../../log/CIFAR-100-Bayes.csv", train_loader=c100_train_loader, val_loader=c100_val_loader)
