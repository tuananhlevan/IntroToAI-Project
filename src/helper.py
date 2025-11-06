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
    
    plt.figure(figsize=(8,5))
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
    # model.eval()
    # predictions = []
    
    # for _ in range(100):
    #     with torch.no_grad():
    #         y_pred = model(X_train, sample=True)
    #         predictions.append(y_pred)
    pass
    
def visualize_uncertainty(model, X_test, X_train, y_train):
    pass