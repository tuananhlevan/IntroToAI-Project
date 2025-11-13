import torch
    
def cls_entropy(model, x):
    prob_out = model(x, sample=True)
    return - prob_out * torch.log2(prob_out + 1e-10) - (1 - prob_out) * torch.log2(1 - prob_out + 1e-10)

def cls_alea(model, x, num_sample=100):
    total_entropy = 0
    for _ in range(num_sample):
        total_entropy += cls_entropy(model, x=x)
    return total_entropy / num_sample

def cls_total(model, x):
    prob_out = model(x, sample=False)
    return - prob_out * torch.log2(prob_out + 1e-10) - (1 - prob_out) * torch.log2(1 - prob_out + 1e-10)

def cls_epi(model, x, num_sample=100):
    return cls_total(model=model, x=x) - cls_alea(model=model, x=x, num_sample=num_sample)