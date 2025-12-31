import torch
import torch.nn.functional as F
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassCalibrationError, MulticlassAccuracy
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

def evaluate_model(model, device, loader, num_samples=100):
    model.eval()
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()
    all_mean_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)

            batch_preds_samples = []
            for _ in range(num_samples):
                outputs = model(images)
                batch_preds_samples.append(outputs)

            all_probs = F.softmax(torch.stack(batch_preds_samples), dim=2)
            mean_probs = torch.mean(all_probs, dim=0)

            all_mean_probs.append(mean_probs)
            all_labels.append(labels)

    all_mean_probs = torch.cat(all_mean_probs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    return all_mean_probs, all_labels

def get_metrics(all_mean_probs, all_labels, num_classes, device, n_bins=15):
    metrics = MetricCollection({
        "ACC ↑": MulticlassAccuracy(num_classes=num_classes),
        "ECE ↓": MulticlassCalibrationError(num_classes=num_classes, n_bins=n_bins, norm='l1'),
        "MCE ↓": MulticlassCalibrationError(num_classes=num_classes, n_bins=n_bins, norm='max'),
    })
    metrics.to(device)

    metrics.update(preds=all_mean_probs, target=all_labels)

    return {k: v.item() for k, v in metrics.compute().items()}

def visualize_results(metrics, path):
    plt.rcParams['axes.axisbelow'] = True
    plt.figure(figsize=(8, 5))
    colors = ['#3498db', '#e74c3c', '#2ecc71']

    plt.grid(True, alpha=0.3)

    evaluate_models = list(metrics.keys())
    evaluate_values = list(metrics.values())

    for i in range(len(metrics)):
        plt.bar(evaluate_models[i], evaluate_values[i], color=colors[i % len(colors)])

    plt.ylim(0, max(evaluate_values) * 1.2)  # Add some headspace

    # Add value labels on top of bars
    for j, v in enumerate(evaluate_values):
        plt.text(j, v + (max(evaluate_values) * 0.02), f'{v:.3f}', ha='center')

    # plt.savefig(path, bbox_inches='tight', dpi=600)
    plt.show()

def reliability_diagram(y_prob, y_true, path, n_bins=15):

    # Top-1 confidence + correctness
    y_pred = np.argmax(y_prob, axis=1)
    y_conf = np.max(y_prob, axis=1)
    correct = (y_pred == y_true).astype(float)

    # Fixed bins
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_conf, bins) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    acc = np.zeros(n_bins)
    conf = np.zeros(n_bins)
    count = np.zeros(n_bins)

    for b in range(n_bins):
        mask = bin_ids == b
        if np.any(mask):
            acc[b] = correct[mask].mean()
            conf[b] = y_conf[mask].mean()
            count[b] = mask.sum()

    # ECE (classic definition)
    ece = np.sum((count / len(y_true)) * np.abs(acc - conf))

    fig, ax = plt.subplots(figsize=(6, 6))

    width = 1.0 / n_bins
    x = bins[:-1]

    # Base accuracy bars (gray)
    ax.bar(
        x,
        acc,
        width=width,
        align="edge",
        color="#cfd6cc",
        edgecolor="black",
        label="Accuracy",
    )

    # Over-confidence: confidence > accuracy
    over = np.maximum(conf - acc, 0)
    ax.bar(
        x,
        over,
        bottom=acc,
        width=width,
        align="edge",
        color="none",
        edgecolor="#4C9AFF",
        hatch="\\\\",
        label="Over Confident",
    )

    # Under-confidence: accuracy > confidence
    under = np.maximum(acc - conf, 0)
    ax.bar(
        x,
        under,
        bottom=conf,
        width=width,
        align="edge",
        color="#cfd6cc",
        alpha=0.7,
        edgecolor="#4C9AFF",
        hatch="//",
        label="Under Confident",
    )

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], "k--", linewidth=2, label="Perfect Calibration")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"ECE = {ece:.3f}")
    ax.legend(frameon=True)
    # plt.savefig(path, bbox_inches='tight', dpi=600)
    plt.show()