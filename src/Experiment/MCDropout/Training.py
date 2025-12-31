import torch

from src.Experiment.MCDropout.Model import MCDropoutMobileNet
from src.Experiment.MCDropout.TrainFunction import train
from src.Experiment.GetDataLoader import train_loader, val_loader

torch.manual_seed(42)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = MCDropoutMobileNet(num_classes=4).to(DEVICE)
# model = torch.compile(model)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total Trainable Parameters: {trainable_params:,}")

train(model=model, device=DEVICE, path_to_model="model.pth", log_path="../log/mc_dropout_log.csv", train_loader=train_loader, val_loader=val_loader)