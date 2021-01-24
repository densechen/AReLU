import activations
import torch
import torch.nn as nn
import torch.nn.functional as F


class ELSA(nn.Module):
    def __init__(self, activation: str = "ReLU", with_elsa: bool = False, **kwargs):
        super().__init__()
        self.activation = activations.__class_dict__[activation](**kwargs)
        self.with_elsa = with_elsa

        if self.with_elsa:
            self.alpha = nn.Parameter(
                torch.tensor([kwargs.get("alpha", 0.90)]))
            self.beta = nn.Parameter(torch.tensor([kwargs.get("beta", 2.0)]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.with_elsa:
            alpha = torch.clamp(self.alpha, min=0.01, max=0.99)
            beta = torch.sigmoid(self.beta)

            return self.activation(x) + torch.where(x > 0, x * self.beta, x * self.alpha)
        else:
            return self.activation(x)
