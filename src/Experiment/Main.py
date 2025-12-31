import torch

from Experiment.CNN.Model import CNN
from src.Experiment.HelperFunctions import evaluate_model, visualize_results, reliability_diagram, get_metrics
from src.Experiment.Bayes.Model import BayesNet
from src.Experiment.MCDropout.Model import MCDropout
from src.Experiment.GetDataLoader import test_loader

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
N_BINS = 15
NUM_SAMPLES = 50

bayes = BayesNet(num_classes=4)
bayes.load_state_dict(torch.load('Bayes/model.pth'))
bayes.to(DEVICE)

mcdropout = MCDropout(num_classes=4)
mcdropout.load_state_dict(torch.load('MCDropout/model.pth'))
mcdropout.to(DEVICE)

cnn = CNN(num_classes=4)
cnn.load_state_dict(torch.load('CNN/model.pth'))
cnn.to(DEVICE)

probsBayes, labels = evaluate_model(model=bayes, device=DEVICE, loader=test_loader, num_samples=NUM_SAMPLES)
metricsBayes = get_metrics(probsBayes, labels, num_classes=4, device=DEVICE, n_bins=N_BINS)
probsMcDropout, _ = evaluate_model(model=mcdropout, device=DEVICE, loader=test_loader, num_samples=NUM_SAMPLES)
metricsMcDropout = get_metrics(probsMcDropout, labels, num_classes=4, device=DEVICE, n_bins=N_BINS)
probsCNN, _ = evaluate_model(model=cnn, device=DEVICE, loader=test_loader, num_samples=NUM_SAMPLES)
metricsCNN = get_metrics(probsCNN, labels, num_classes=4, device=DEVICE, n_bins=N_BINS)

result_dict = {
    "Bayes": metricsBayes,
    "MCDropout": metricsMcDropout,
    "CNN": metricsCNN,
}

reliability_diagram(labels.detach().cpu().numpy(), [probsBayes.detach().cpu().numpy(), probsCNN.detach().cpu().numpy(), probsMcDropout.detach().cpu().numpy()], path="reliability_experiment", model_names=["Bayes", "CNN", "MCDropout"], n_bins=N_BINS)
visualize_results(result_dict, path="metrics_experiment", name="Brain Tumor")