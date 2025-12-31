import torch
import torch.nn.functional as F
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassCalibrationError, MulticlassAccuracy
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

def evaluate_model(model, device, num_classes, loader, is_bayesian=True, num_samples=100):
    model.eval()
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()
    
    metrics = MetricCollection({
        "ACC ↑": MulticlassAccuracy(num_classes=num_classes),
        "ECE ↓": MulticlassCalibrationError(num_classes=num_classes, n_bins=15, norm='l1'),
        "MCE ↓": MulticlassCalibrationError(num_classes=num_classes, n_bins=15, norm='max'),
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

    return {k: v.item() for k, v in metrics.compute().items()}, all_mean_probs.detach().cpu().numpy(), all_labels.detach().cpu().numpy()

def visualize_results(results_dict, path, name):
    models = list(results_dict.keys())
    metrics = list(results_dict[models[0]].keys())

    # Create subplots: 1 row, N columns
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].set_ylabel(name)
    colors = ['#3498db', '#e74c3c', '#50C878']

    for i, metric in enumerate(metrics):
        axes[i].grid(True, alpha=0.3)
        axes[i].set_axisbelow(True)

        # Extract values for this specific metric across all models
        values = [results_dict[m][metric] for m in models]

        # Convert tensors to floats if they are still torch tensors
        values = [v.item() if hasattr(v, 'item') else v for v in values]

        axes[i].bar(models, values, color=colors, label=models if i == 0 else None)
        if name == "CIFAR-100":
            axes[i].set_title(f'{metric}', y = -0.13)
        axes[i].set_ylim(0, max(values) * 1.2)  # Add some headspace

        # Add value labels on top of bars
        for j, v in enumerate(values):
            axes[i].text(j, v + (max(values) * 0.02), f'{v:.3f}', ha='center')

    if name == "CIFAR-10":
        fig.legend(loc="upper center", ncols=2, bbox_to_anchor=(0.5, 0.98))
    plt.savefig(path, bbox_inches='tight', dpi=600)

def reliability_diagram(y_true, y_prob_list, model_names, path, n_bins=15):
    assert len(y_prob_list) == len(model_names)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    width = 1.0 / n_bins
    x = bins[:-1]

    def compute_stats(y_prob):
        y_pred = np.argmax(y_prob, axis=1)
        y_conf = np.max(y_prob, axis=1)
        correct = (y_pred == y_true).astype(float)

        acc = np.zeros(n_bins)
        conf = np.zeros(n_bins)
        count = np.zeros(n_bins)

        ids = np.digitize(y_conf, bins) - 1
        ids = np.clip(ids, 0, n_bins - 1)

        for b in range(n_bins):
            m = ids == b
            if np.any(m):
                acc[b] = correct[m].mean()
                conf[b] = y_conf[m].mean()
                count[b] = m.sum()

        ece = np.sum((count / len(y_true)) * np.abs(acc - conf))
        return acc, conf, ece

    fig, axes = plt.subplots(
        1, len(y_prob_list),
        figsize=(6 * len(y_prob_list), 6),
        sharey=True,
    )

    if len(y_prob_list) == 1:
        axes = [axes]

    for i, (ax, y_prob, name) in enumerate(zip(axes, y_prob_list, model_names)):
        acc, conf, ece = compute_stats(y_prob)

        # Base accuracy
        ax.bar(
            x,
            acc,
            width=width,
            align="edge",
            color="#cfd6cc",
            edgecolor="black",
            label="Accuracy" if i == 0 else None
        )

        # Over-confidence
        ax.bar(
            x,
            np.maximum(conf - acc, 0),
            bottom=acc,
            width=width,
            align="edge",
            edgecolor="#4C9AFF",
            hatch="\\\\",
            fill=False,
            label="Over Confident" if i == 0 else None
        )

        # Under-confidence
        ax.bar(
            x,
            np.maximum(acc - conf, 0),
            bottom=conf,
            width=width,
            align="edge",
            color="#cfd6cc",
            alpha=0.7,
            edgecolor="#4C9AFF",
            hatch="//",
            label="Under Confident" if i == 0 else None
        )

        # Perfect calibration
        ax.plot([0, 1], [0, 1], "k--", linewidth=2, label="Perfect Calibration" if i == 0 else None)

        ax.set_title(f"{name} - ECE = {ece:.3f}")
        ax.set_xlabel(f"Confident", fontsize=12)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    axes[0].set_ylabel("Accuracy")
    if path == "calibration_cifar10":
        fig.legend(bbox_to_anchor=(0.6, 0.87), loc="upper center")
    plt.savefig(path, bbox_inches='tight', dpi=600)