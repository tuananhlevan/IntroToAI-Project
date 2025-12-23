import torch

from src.Experiment.HelperFunctions import evaluate_model, visualize_results
from src.Experiment.Model import BayesMobileNet
from src.Experiment.GetDataLoader import test_loader

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = BayesMobileNet(num_classes=4)
model.load_state_dict(torch.load('model/model.pth'))
model.to(DEVICE)
model.eval()

modelMetrics = evaluate_model(model=model, device=DEVICE, num_classes=4, loader=test_loader, is_bayesian=True, num_samples=100)
visualize_results(modelMetrics)

