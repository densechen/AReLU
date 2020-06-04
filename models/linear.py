from models import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearMNIST(BaseModel):
    def __init__(self, activation: nn.Module, in_ch: int=1):
        super().__init__(activation)

        self.linear1 = nn.Sequential(
            nn.Linear(in_ch * 28 * 28, 512),
            activation(),
        )

        self.linear2 = nn.Sequential(
            nn.Linear(512, 10),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, x):
        x = x.view(-1, 28 * 28)

        x = self.linear1(x)

        x = self.linear2(x)

        return x
