"""Shared neural-network layers."""

import jittor as jt
from jittor import nn


class ChannelLayerNorm(nn.Module):
    """Normalize (B, C, ...) over channels, with optional channel-wise affine."""

    def __init__(self, channels: int = None, eps: float = 1e-6, affine: bool = False):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            if channels is None:
                raise ValueError("affine ChannelLayerNorm requires channels")
            self.weight = jt.ones((channels,))
            self.bias = jt.zeros((channels,))

    def execute(self, x):
        mean = x.mean(dim=1, keepdims=True)
        var = ((x - mean) ** 2).mean(dim=1, keepdims=True)
        x = (x - mean) / jt.sqrt(var + self.eps)
        if self.affine:
            shape = (1, -1) + (1,) * (x.ndim - 2)
            x = x * self.weight.reshape(shape) + self.bias.reshape(shape)
        return x
