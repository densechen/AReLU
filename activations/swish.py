import torch
import torch.nn as nn


class Swish_fun(torch.autograd.Function):

    @staticmethod
    def forward(ctx, i):
        result = i * i.sigmoid()
        ctx.save_for_backward(result, i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        result, i = ctx.saved_variables
        sigmoid_x = i.sigmoid()
        return grad_output * (result + sigmoid_x * (1 - result))


swish = Swish_fun.apply


class Swish(nn.Module):
    def forward(self, x):
        return swish(x)
