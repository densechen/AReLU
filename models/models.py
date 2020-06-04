import torch
import torch.nn as nn


class BaseModel(nn.Module):
    """ Base Model for implementation different models.
    """

    def __init__(self, activation: nn.Module, in_ch: int=1):
        super().__init__()
