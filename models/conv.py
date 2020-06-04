import torch
import torch.nn as nn
import torch.nn.functional as F
from models import BaseModel


def conv_block(in_plane, out_plane, kernel_size, activation):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_plane, out_channels=out_plane,
                  kernel_size=kernel_size),
        nn.MaxPool2d(2),
        activation(),
    )


class ConvMNIST(BaseModel):
    def __init__(self, activation: nn.Module, in_ch: int=1):
        super().__init__(activation)

        self.conv_block1 = conv_block(
            in_ch, 10, kernel_size=5, activation=activation)
        self.conv_block2 = conv_block(
            10, 20, kernel_size=5, activation=activation)
        self.conv_block3 = conv_block(
            20, 40, kernel_size=3, activation=activation)

        self.fc = nn.Sequential(
            nn.Linear(40, 10),
            nn.LogSoftmax(dim=-1),
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)

        x = x.view(-1, 40)

        x = self.fc(x)

        return x
