import torch
import torch.nn as nn


class SLAF(nn.Module):
    def __init__(self, k=2):
        super().__init__()
        self.k = k
        self.coeff = nn.ParameterList(
            [nn.Parameter(torch.tensor(1.0)) for i in range(k)])

    def forward(self, x):
        out = sum([self.coeff[k] * torch.pow(x, k) for k in range(self.k)])
        return out
