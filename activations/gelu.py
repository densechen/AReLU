import torch
import torch.nn as nn


class GELU(nn.Module):
    """https://mlfromscratch.com/activation-functions-explained/#/
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(0.7979946 * (x + 0.044715 * torch.pow(x, 3))))
