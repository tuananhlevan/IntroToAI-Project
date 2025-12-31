import torch

from src.Experiment.Bayes.Model import BayesMobileNet
from src.Experiment.Bayes.TrainFunction import train
from src.Experiment.GetDataLoader import train_loader, val_loader

torch.manual_seed(42)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = BayesMobileNet(num_classes=4).to(DEVICE)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total Trainable Parameters: {trainable_params:,}")

train(model=model, device=DEVICE, path_to_model="model.pth", log_path="../log/bayes_log.csv", train_loader=train_loader, val_loader=val_loader)