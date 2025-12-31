import torch

from src.Experiment.HelperFunctions import evaluate_model, visualize_results, reliability_diagram, get_metrics
from src.Experiment.Bayes.Model import BayesMobileNet
from src.Experiment.GetDataLoader import test_loader

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
N_BINS = 15
NUM_SAMPLES = 100

model = BayesMobileNet(num_classes=4)
model.load_state_dict(torch.load('Bayes/model.pth'))
model.to(DEVICE)
model.eval()

probs, labels = evaluate_model(model=model, device=DEVICE, loader=test_loader, num_samples=NUM_SAMPLES)
modelMetrics = get_metrics(probs, labels, num_classes=4, device=DEVICE, n_bins=N_BINS)
visualize_results(modelMetrics, path="metrics_experiment")
reliability_diagram(probs.detach().cpu().numpy(), labels.detach().cpu().numpy(), path="reliability_experiment", n_bins=N_BINS)