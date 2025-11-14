import torch
import numpy as np

import matplotlib.pyplot as plt
from csaps import csaps

from helper_cls import cls_alea, cls_total

def visualize_prediction_reg(model, X_train, y_train, mean_pred, std_pred):
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
    
def visualize_uncertainty_reg(model, X_train, y_train):
    pass
    
def visualize_prediction_cls(model, X_train, y_train):
    plt.figure(figsize=(8, 6))
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.scatter(X_train[:, 0], y_train, color='blue', label='True value')
    x_loc = torch.linspace(-25, 25, 100).unsqueeze(1)
    plt.plot(x_loc.cpu().ravel(), model(x_loc).detach().numpy(), color='red', linewidth=2, label="Predicted value")
    
    plt.axhline(y=0.5, color='green', linestyle='--', label='0.5 Probability Threshold')
    plt.tight_layout()
    plt.legend()
    plt.show()
        
def visualize_uncertainty_cls(model, X_train, y_train):
    plt.figure(figsize=(8, 6))
    plt.grid(True, linestyle='--', alpha=0.7)
    
    seen_y0 = False
    seen_y1 = False
    for (x, y) in zip(X_train, y_train):
        if -20 < x < -3 or 20 > x > 3:
            continue
        
        if y == 0:
            line_color = 'cyan'
            label = 'ICL Data: y=0'
            
            # Only add the label to the legend once
            if not seen_y0:
                plt.axvline(x=x, color=line_color, linestyle='--', linewidth=2, label=label)
                seen_y0 = True
            else:
                plt.axvline(x=x, color=line_color, linestyle='--', linewidth=2)
        else:
            line_color = 'gray'
            label = 'ICL Data: y=1'
            
            # Only add the label to the legend once
            if not seen_y1:
                plt.axvline(x=x, color=line_color, linestyle='--', linewidth=2, label=label)
                seen_y1 = True
            else:
                plt.axvline(x=x, color=line_color, linestyle='--', linewidth=2)            
    
    x_loc = torch.linspace(-25, 25, 100).unsqueeze(1)
    alea = cls_alea(model=model, x=x_loc, num_sample=10)
    total = cls_total(model=model, x=x_loc)
    plt.scatter(x_loc, alea.detach().numpy(), color="C0", alpha=0.4)
    plt.scatter(x_loc, total.detach().numpy(), color="C1", alpha=0.4)
    
    x_loc_1d = x_loc.squeeze().detach().numpy()
    alea_line = csaps(x_loc_1d, alea.squeeze().detach().numpy(), smooth=0.35)
    total_line = csaps(x_loc_1d, total.squeeze().detach().numpy(), smooth=0.35)
    plt.plot(x_loc, alea_line(x_loc), color='C0', linewidth=3, label='Aleatoric Uncertainty')
    plt.plot(x_loc, total_line(x_loc), color="C1", linewidth=3, label="Total Uncertainty")
    
    plt.tight_layout()
    plt.legend()
    plt.show()