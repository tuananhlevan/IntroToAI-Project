import numpy as np
import torch


def predict(model, X_train):
    model.eval()
    predictions = []
    
    for _ in range(100):
        with torch.no_grad():
            y_pred = model(X_train, sample=True)
            predictions.append(y_pred.cpu().numpy())
    
    predictions = np.stack(predictions, axis=0)
    mean_pred = predictions.mean(0)
    std_pred = predictions.std(0)
    
    return mean_pred, std_pred