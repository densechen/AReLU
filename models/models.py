import torch
import torch.nn as nn


class BaseModel(nn.Module):
    """ Base Model for implementation different models.
    """

    def __init__(self, activation: nn.Module):
        super().__init__()
