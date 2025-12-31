import torch

from src.Experiment.CNN.Model import CNN
from src.Experiment.CNN.TrainFunction import train
from src.Experiment.GetDataLoader import train_loader, val_loader

torch.manual_seed(42)
torch.set_float32_matmul_precision('high')
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = CNN(num_classes=4).to(DEVICE)
model = torch.compile(model)

train(model=model, device=DEVICE, path_to_model="model.pth", log_path="../log/cnn_log.csv", train_loader=train_loader, val_loader=val_loader)