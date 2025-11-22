import matplotlib.pyplot as plt
import torch


def visualize_prediction(model, X_train, y_train, mean_pred, std_pred):
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