import torch

from src.Evaluation.Bayes.BayesNet.Model import BayesNet
from src.Evaluation.CNN.BasicCNN.Model import CNN

from src.Evaluation.EvaluationFunction import evaluate_model, visualize_results, reliability_diagram
from src.Evaluation.DownloadData import c10_test_loader, c100_test_loader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BayesCifar10 = BayesNet(num_classes=10)
BayesCifar10.load_state_dict(torch.load("Bayes/model/CIFAR-10-Bayes.pth", weights_only=True))
BayesCifar10.to(DEVICE)
BayesCifar10.eval()

CNNCifar10 = CNN(num_classes=10)
CNNCifar10.load_state_dict(torch.load("CNN/model/CIFAR-10-CNN.pth", weights_only=True))
CNNCifar10.to(DEVICE)
CNNCifar10.eval()

BayesCifar100 = BayesNet(num_classes=100)
BayesCifar100.load_state_dict(torch.load("Bayes/model/CIFAR-100-Bayes.pth", weights_only=True))
BayesCifar100.to(DEVICE)
BayesCifar100.eval()

CNNCifar100 = CNN(num_classes=100)
CNNCifar100.load_state_dict(torch.load("CNN/model/CIFAR-100-CNN.pth", weights_only=True))
CNNCifar100.to(DEVICE)
CNNCifar100.eval()

cnnMetrics10, probCnn10, label10 = evaluate_model(CNNCifar10, DEVICE, 10, c10_test_loader, is_bayesian=False, num_samples=20)
bayesMetrics10, probBayes10, _ = evaluate_model(BayesCifar10, DEVICE, 10, c10_test_loader, is_bayesian=True, num_samples=20)
cnnMetrics100, probCnn100, label100 = evaluate_model(CNNCifar100, DEVICE, 100, c100_test_loader, is_bayesian=False, num_samples=20)
bayesMetrics100, probBayes100, _ = evaluate_model(BayesCifar100, DEVICE, 100, c100_test_loader, is_bayesian=True, num_samples=20)

results_dict_cifar10 = {
    "Bayesian Hybrid": bayesMetrics10,
    "CNN": cnnMetrics10,
}

results_dict_cifar100 = {
    "Bayesian Hybrid": bayesMetrics100,
    "CNN": cnnMetrics100
}

reliability_diagram(label10, [probBayes10, probCnn10], ["Bayes", "CNN"], n_bins=15, path="calibration_cifar10")
reliability_diagram(label100, [probBayes100, probCnn100], ["Bayes", "CNN"], n_bins=15, path="calibration_cifar100")

visualize_results(results_dict_cifar10, "evaluate_cifar10", name="CIFAR-10")
visualize_results(results_dict_cifar100, "evaluate_cifar100", name="CIFAR-100")