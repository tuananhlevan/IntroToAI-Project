import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

def visualize_prediction_reg(model, X_train, y_train):
    model.eval()
    predictions = []
    
    for _ in range(100):
        with torch.no_grad():
            y_pred = model(X_train, sample=True)
            predictions.append(y_pred.cpu().numpy())
    
    predictions = np.stack(predictions, axis=0)
    mean_pred = predictions.mean(0)
    std_pred = predictions.std(0)
    
    plt.figure(figsize=(8, 5))
    plt.scatter(X_train.cpu(), y_train.cpu(), s=15, color='k', label='Data')
    plt.plot(X_train.cpu(), torch.sin(X_train.cpu()) + 0.3 * X_train.cpu(), 'g--', label='True Function')
    plt.plot(X_train.cpu(), mean_pred, 'r', label='Predictive Mean')
    plt.fill_between(
        X_train.cpu().ravel(),
        (mean_pred - 2*std_pred).ravel(),
        (mean_pred + 2*std_pred).ravel(),
        color='r',
        alpha=0.2,
        label='~95% predictive interval'
    )
    plt.legend()
    plt.title("Bayesian Neural Network (Variational Inference)")
    plt.show()
    
def visualize_prediction_cls(model, X_train, y_train):
    model.eval()
    predictions = []
    
    for _ in range(100):
        with torch.no_grad():
            y_pred = model(X_train, sample=True)
            predictions.append(y_pred.cpu().numpy())
    
    predictions = np.stack(predictions, axis=0)
    mean_pred = predictions.mean(0)
    MLE_pred = torch.tensor(mean_pred > 0.5, dtype=torch.float32)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
    ax1.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis', s=50, edgecolors='k', alpha=0.8)
    ax1.set_title("True Labels", fontsize=16)
    ax1.set_xlabel("Feature 1 (X1)", fontsize=12)
    ax1.set_ylabel("Feature 2 (X2)", fontsize=12)
    ax1.set_aspect('equal', adjustable='box')
    
    ax2.scatter(X_train[:, 0], X_train[:, 1], c=MLE_pred, cmap='viridis', s=50, edgecolors='k', alpha=0.8)
    ax2.set_title("Predicted Labels", fontsize=16)
    ax2.set_xlabel("Feature 1 (X1)", fontsize=12)
    ax2.set_aspect('equal', adjustable='box')
    
    fig.suptitle("Model Classification vs. True Labels", fontsize=20, y=1.03)
    plt.tight_layout()
    plt.show()
    
    
def visualize_uncertainty(model, X_test, X_train, y_train):
    pass