import torch
    
def entropy(model, x):
    model.eval()
    prob_out = model(x, sample=True)
    return - prob_out * torch.log(prob_out + 1e-10) - (1 - prob_out) * torch.log(1 - prob_out + 1e-10)

def AleaUnc(model, x, num_sample=100):
    model.eval()
    total_entropy = 0
    for _ in range(num_sample):
        total_entropy += entropy(model, x=x)
    return total_entropy / num_sample

def TotalUnc(model, x, num_samples=100):
    model.eval()
    prob = 0
    for _ in range(num_samples):
        prob += model(x, sample=True)
    prob_out = prob / num_samples
    return - prob_out * torch.log(prob_out + 1e-10) - (1 - prob_out) * torch.log(1 - prob_out + 1e-10)

def EpiUnc(model, x, num_sample=100):
    model.eval()
    return TotalUnc(model=model, x=x) - AleaUnc(model=model, x=x, num_sample=num_sample)