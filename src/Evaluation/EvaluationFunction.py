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
        "ece": MulticlassCalibrationError(num_classes=num_classes, n_bins=20, norm='l1'),
        "mce": MulticlassCalibrationError(num_classes=num_classes, n_bins=20, norm='max'),
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

def visualize_results(results_dict):
    models = list(results_dict.keys())
    metrics = list(results_dict[models[0]].keys())

    # Create subplots: 1 row, N columns
    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 5))
    colors = ['#3498db', '#e74c3c', '#2ecc71']

    for i, metric in enumerate(metrics):
        axes[i].grid(True, alpha=0.3, axis="y")
        axes[i].set_axisbelow(True)

        # Extract values for this specific metric across all models
        values = [results_dict[m][metric] for m in models]

        # Convert tensors to floats if they are still torch tensors
        values = [v.item() if hasattr(v, 'item') else v for v in values]

        axes[i].bar(models, values, color=colors[i % len(colors)])
        axes[i].set_title(f'{metric.upper()}')
        axes[i].set_ylim(0, max(values) * 1.2)  # Add some headspace

        # Add value labels on top of bars
        for j, v in enumerate(values):
            axes[i].text(j, v + (max(values) * 0.02), f'{v:.3f}', ha='center')

    plt.tight_layout()
    plt.show()