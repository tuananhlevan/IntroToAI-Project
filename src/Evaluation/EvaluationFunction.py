import torch
import torch.nn.functional as F
from torchmetrics.classification import MulticlassCalibrationError
from tqdm.auto import tqdm

def evaluate_model(model, device, num_classes, loader, is_bayesian=True, num_samples=100):
    model.eval()
    
    ece_metric = MulticlassCalibrationError(num_classes=num_classes, n_bins=20, norm='l1')
    ece_metric = ece_metric.to(device)

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

    predicted_classes = torch.argmax(all_mean_probs, dim=1)
    correct = (predicted_classes == all_labels).sum().item()
    accuracy = correct / len(all_labels)

    ece = ece_metric(all_mean_probs, all_labels)

    return accuracy, ece.item()