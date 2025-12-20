import torch

from src.Experiment.Model import BayesNet
from src.Experiment.TrainFunction import train
from src.Experiment.GetDataLoader import train_loader, val_loader

torch.manual_seed(42)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = BayesNet(num_classes=2).to(DEVICE)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")

train(model=model, device=DEVICE, path_to_model="model/model.pth", log_path="log/log.csv", train_loader=train_loader, val_loader=val_loader)