import torch

from src.Evaluation.Bayes.BayesNet.Model import BayesNet
from src.Evaluation.CNN.BasicCNN.Model import CNN

from src.Evaluation.EvaluationFunction import evaluate_model, visualize_results
from src.Evaluation.DownloadData import c10_test_loader, c100_test_loader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BayesCifar10 = BayesNet(num_classes=10)
BayesCifar10.load_state_dict(torch.load("Bayes/model/CIFAR-10-Bayes.pth", weights_only=True))
BayesCifar10.to(DEVICE)
BayesCifar10.eval()

BayesCifar100 = BayesNet(num_classes=100)
BayesCifar100.load_state_dict(torch.load("Bayes/model/CIFAR-100-Bayes.pth", weights_only=True))
BayesCifar100.to(DEVICE)
BayesCifar100.eval()

CNNCifar10 = CNN(num_classes=10)
CNNCifar10.load_state_dict(torch.load("CNN/model/CIFAR-10-CNN.pth", weights_only=True))
CNNCifar10.to(DEVICE)
CNNCifar10.eval()

CNNCifar100 = CNN(num_classes=100)
CNNCifar100.load_state_dict(torch.load("CNN/model/CIFAR-100-CNN.pth", weights_only=True))
CNNCifar100.to(DEVICE)
CNNCifar100.eval()

cnnMetrics10 = evaluate_model(CNNCifar10, DEVICE, 10, c10_test_loader, is_bayesian=False, num_samples=100)
cnnMetrics100 = evaluate_model(CNNCifar100, DEVICE, 100, c100_test_loader, is_bayesian=False, num_samples=100)
bayesMetrics10 = evaluate_model(BayesCifar10, DEVICE, 10, c10_test_loader, is_bayesian=True, num_samples=100)
bayesMetrics100 = evaluate_model(BayesCifar100, DEVICE, 100, c100_test_loader, is_bayesian=True, num_samples=100)

results_dict = {
    "Bayesian10": bayesMetrics10,
    "CNN10": cnnMetrics10,
    "Bayes100": bayesMetrics100,
    "CNN100": cnnMetrics100
}

visualize_results(results_dict)