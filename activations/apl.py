import torch
import torch.nn as nn


class APL(nn.Module):
    def __init__(self, s=1):
        super().__init__()

        self.a = nn.ParameterList(
            [nn.Parameter(torch.tensor(0.2)) for _ in range(s)])
        self.b = nn.ParameterList(
            [nn.Parameter(torch.tensor(0.5)) for _ in range(s)])
        self.s = s

    def forward(self, x):
        part_1 = torch.clamp_min(x, min=0.0)
        part_2 = 0
        for i in range(self.s):
            part_2 += self.a[i] * torch.clamp_min(-x+self.b[i], min=0)

        return part_1 + part_2
