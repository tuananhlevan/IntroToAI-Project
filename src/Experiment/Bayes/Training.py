import torch

from src.Experiment.Bayes.Model import BayesNet
from src.Experiment.Bayes.TrainFunction import train
from src.Experiment.GetDataLoader import train_loader, val_loader

torch.manual_seed(42)
torch.set_float32_matmul_precision('high')
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = BayesNet(num_classes=4).to(DEVICE)
model = torch.compile(model)

train(model=model, device=DEVICE, path_to_model="model.pth", log_path="../log/bayes_log.csv", train_loader=train_loader, val_loader=val_loader)