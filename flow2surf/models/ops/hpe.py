"""Optimized hidden path for High-dimensional Positional Encoding."""

import jittor as jt
from jittor import nn

from ._sources import load_cuda

_CUDA_WIDTH_LIMIT = 256
_CUDA_BLOCK_REDUCE = load_cuda("block_reduce.cuh")
_CUDA_FORWARD = load_cuda("hpe_hidden_forward.cu")
_CUDA_BACKWARD = load_cuda("hpe_activation_backward.cu")


def project_hpe_input(geometry, weight, bias):
    hidden_channels, input_channels = weight.shape
    return nn.linear(geometry.reshape(-1, input_channels), weight, bias).reshape(
        geometry.shape[:3] + (hidden_channels,)
    )


def _activate_hpe(preactivation, weight, bias):
    mean = preactivation.mean(dim=-1, keepdims=True)
    second_moment = (preactivation * preactivation).mean(dim=-1, keepdims=True)
    variance = jt.maximum(second_moment - mean * mean, 0.0)
    scale = weight / jt.sqrt(variance + 1e-5)
    shift = bias - mean * scale
    return nn.gelu(preactivation * scale + shift)


def hpe_hidden_jittor(geometry, input_weight, input_bias, norm_weight, norm_bias):
    preactivation = project_hpe_input(geometry, input_weight, input_bias)
    return _activate_hpe(preactivation, norm_weight, norm_bias)


def hpe_hidden_cuda(geometry, input_weight, input_bias, norm_weight, norm_bias):
    shape = geometry.shape[:3] + (input_weight.shape[0],)
    return jt.code(
        shape,
        geometry.dtype,
        [geometry, input_weight, input_bias, norm_weight, norm_bias],
        cuda_header=_CUDA_BLOCK_REDUCE,
        cuda_src=_CUDA_FORWARD,
    )


def project_hpe_output(hidden, weight, bias):
    channels, hidden_channels = weight.shape
    return nn.linear(hidden.reshape(-1, hidden_channels), weight, bias).reshape(
        hidden.shape[:3] + (channels,)
    )


def _hpe_activation_grad_jittor(preactivation, weight, bias, grad_hidden):
    mean = preactivation.mean(dim=-1, keepdims=True)
    second_moment = (preactivation * preactivation).mean(dim=-1, keepdims=True)
    raw_variance = second_moment - mean * mean
    inv_std = jt.rsqrt(jt.maximum(raw_variance, 0.0) + 1e-5)
    normalized = (preactivation - mean) * inv_std
    affine = normalized * weight + bias

    sqrt_two = 1.4142135623730951
    inv_sqrt_two_pi = 0.3989422804014327
    cdf = 0.5 * (jt.erf(affine / sqrt_two) + 1.0)
    gelu_grad = cdf + affine * jt.exp(-0.5 * affine * affine) * inv_sqrt_two_pi
    grad_affine = grad_hidden * gelu_grad

    flat_grad = grad_affine.reshape(-1, grad_affine.shape[-1])
    flat_normalized = normalized.reshape(-1, normalized.shape[-1])
    grad_weight = (flat_grad * flat_normalized).sum(dim=0)
    grad_bias = flat_grad.sum(dim=0)

    grad_normalized = grad_affine * weight
    mean_grad = grad_normalized.mean(dim=-1, keepdims=True)
    mean_projection = (grad_normalized * normalized).mean(dim=-1, keepdims=True)
    variance_active = (raw_variance > 0.0).float32()
    grad_preactivation = inv_std * (
        grad_normalized - mean_grad - variance_active * normalized * mean_projection
    )
    return grad_preactivation, grad_weight, grad_bias


def hpe_activation_grad(preactivation, weight, bias, grad_hidden):
    if (
        not jt.flags.use_cuda
        or preactivation.dtype != "float32"
        or len(preactivation.shape) != 4
        or preactivation.shape[-1] > _CUDA_WIDTH_LIMIT
    ):
        return _hpe_activation_grad_jittor(preactivation, weight, bias, grad_hidden)
    return jt.code(
        [preactivation.shape, weight.shape, bias.shape],
        [preactivation.dtype] * 3,
        [preactivation, weight, bias, grad_hidden],
        cuda_header=_CUDA_BLOCK_REDUCE,
        cuda_src=_CUDA_BACKWARD,
    )
