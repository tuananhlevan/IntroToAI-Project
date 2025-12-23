import torch
import torch.nn.functional as F
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassCalibrationError, MulticlassAccuracy
from tqdm import tqdm
import matplotlib.pyplot as plt

def evaluate_model(model, device, num_classes, loader, is_bayesian=True, num_samples=100):
    model.eval()

    metrics = MetricCollection({
        "acc": MulticlassAccuracy(num_classes=num_classes),
        "ece": MulticlassCalibrationError(num_classes=num_classes, n_bins=15, norm='l1'),
        "mce": MulticlassCalibrationError(num_classes=num_classes, n_bins=15, norm='max'),
    })
    metrics.to(device)

    all_mean_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)

            if is_bayesian:
                batch_preds_samples = []
                for _ in range(num_samples):
                    outputs = model(images)
                    batch_preds_samples.append(outputs)

                all_probs = F.softmax(torch.stack(batch_preds_samples), dim=2)

                mean_probs = torch.mean(all_probs, dim=0)

            else:
                outputs = model(images)
                mean_probs = F.softmax(outputs, dim=1)

            all_mean_probs.append(mean_probs)
            all_labels.append(labels)

    all_mean_probs = torch.cat(all_mean_probs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    metrics.update(preds=all_mean_probs, target=all_labels)

    return metrics.compute()

def visualize_results(metrics):
    plt.rcParams['axes.axisbelow'] = True
    plt.figure(figsize=(8, 5))
    colors = ['#3498db', '#e74c3c', '#2ecc71']

    plt.grid(True, alpha=0.3)

    evaluate_models = list(metrics.keys())
    evaluate_values = list([value.cpu().numpy() for value in metrics.values()])

    for i in range(len(metrics)):
        plt.bar(evaluate_models[i], evaluate_values[i], color=colors[i % len(colors)])
        plt.title(f'{evaluate_models[i].upper()}')

    plt.ylim(0, max(evaluate_values) * 1.2)  # Add some headspace

    # Add value labels on top of bars
    for j, v in enumerate(evaluate_values):
        plt.text(j, v + (max(evaluate_values) * 0.02), f'{v:.3f}', ha='center')

    plt.tight_layout()
    plt.show()