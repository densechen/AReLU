import torch
import torch.nn as nn


class Mixture(nn.Module):
    def __init__(self,):
        super().__init__()
        self.p = nn.Parameter(torch.tensor([0.0]))

    def forward(self, x):
        return self.p * x + (1-self.p) * torch.relu(x)
