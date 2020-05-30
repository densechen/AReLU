import torch
import torch.nn as nn


class Maxout(nn.Module):
    """https://github.com/pytorch/pytorch/issues/805#issuecomment-460385007
    """

    def __init__(self, pool_size=1):
        super().__init__()
        self._pool_size = pool_size

    def forward(self, x):
        assert x.shape[1] % self._pool_size == 0, \
            'Wrong input last dim size ({}) for Maxout({})'.format(
                x.shape[1], self._pool_size)
        m, i = x.view(*x.shape[:1], x.shape[1] // self._pool_size,
                      self._pool_size, *x.shape[2:]).max(2)
        return m
