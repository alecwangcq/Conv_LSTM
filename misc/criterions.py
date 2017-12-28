import torch
import torch.nn as nn

def cross_entropy(x, target):
    x = nn.Sigmoid()(x)
    n_samples = x.size(0)
    loss = -torch.sum( target * torch.log(x) + (1-target)*torch.log(1-x) )/n_samples
    return loss