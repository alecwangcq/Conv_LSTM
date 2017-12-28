import torch
import torch.nn as nn

def cross_entropy(x, target):
    eps = 0
    x = nn.Sigmoid()(-x)
    n_samples = x.size(0)
    loss = -torch.sum( target * torch.log(x) + (1-target)*torch.log(1-x + eps) )/n_samples
    return loss