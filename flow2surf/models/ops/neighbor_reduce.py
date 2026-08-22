"""Normalized neighbor-message reduction with HPE geometry."""

import jittor as jt
from jittor import Function, nn

from ._sources import load_cuda
from .hpe import (
    hpe_activation_grad,
    hpe_hidden_cuda,
    hpe_hidden_jittor,
    project_hpe_input,
    project_hpe_output,
)

_CUDA_WIDTH_LIMIT = 256
_CUDA_BLOCK_REDUCE = load_cuda("block_reduce.cuh")
_CUDA_NEIGHBOR_FORWARD = load_cuda("neighbor_reduce_forward.cu")
_CUDA_NEIGHBOR_BACKWARD = load_cuda("neighbor_reduce_backward.cu")


def _gather_neighbor_terms(content, indices, channels):
    batch, _, points = content.shape
    neighbors = indices.shape[-1]
    values = content[:, :channels].permute(0, 2, 1)
    offsets = (jt.arange(batch) * points).reshape(batch, 1, 1)
    flat_indices = (indices + offsets).reshape(-1)
    return values.reshape(batch * points, channels)[flat_indices].reshape(
        batch, points, neighbors, channels
    )


def _reduce_messages_jittor(content, contribution, indices, weight, bias):
    channels = contribution.shape[-1]
    neighbor = _gather_neighbor_terms(content, indices, channels)
    center = content[:, channels:].permute(0, 2, 1).unsqueeze(2)
    message = (neighbor + center + contribution).permute(0, 3, 1, 2)
    mean = message.mean(dim=1, keepdims=True)
    variance = ((message - mean) ** 2).mean(dim=1, keepdims=True)
    message = (message - mean) / jt.sqrt(variance + 1e-6)
    message = message * weight.reshape(1, -1, 1, 1)
    message = message + bias.reshape(1, -1, 1, 1)
    return nn.leaky_relu(message, scale=0.2).max(dim=-1)


def _reduce_messages_cuda(content, contribution, indices, weight, bias):
    batch, _, points = content.shape
    channels = contribution.shape[-1]
    return jt.code(
        (batch, channels, points),
        content.dtype,
        [content, contribution, indices, weight, bias],
        cuda_header=_CUDA_BLOCK_REDUCE,
        cuda_src=_CUDA_NEIGHBOR_FORWARD,
    )


def _reduce_messages_grad(
    content,
    contribution,
    indices,
    weight,
    bias,
    output,
    grad_output,
):
    return jt.code(
        [content.shape, contribution.shape, weight.shape, bias.shape],
        [content.dtype] * 4,
        [content, contribution, indices, weight, bias, output, grad_output],
        cuda_header=_CUDA_BLOCK_REDUCE,
        cuda_src=_CUDA_NEIGHBOR_BACKWARD,
    )


def _reduce_messages_reference(
    content,
    geometry,
    indices,
    hpe_input_weight,
    hpe_input_bias,
    hpe_output_weight,
    hpe_output_bias,
    hpe_norm_weight,
    hpe_norm_bias,
    message_norm_weight,
    message_norm_bias,
):
    hidden = hpe_hidden_jittor(
        geometry,
        hpe_input_weight,
        hpe_input_bias,
        hpe_norm_weight,
        hpe_norm_bias,
    )
    contribution = project_hpe_output(hidden, hpe_output_weight, hpe_output_bias)
    return _reduce_messages_jittor(
        content,
        contribution,
        indices,
        message_norm_weight,
        message_norm_bias,
    )


class _NeighborMessageFunction(Function):
    """Reduce messages while recomputing HPE intermediates in backward."""

    def execute(
        self,
        content,
        geometry,
        indices,
        hpe_input_weight,
        hpe_input_bias,
        hpe_output_weight,
        hpe_output_bias,
        hpe_norm_weight,
        hpe_norm_bias,
        message_norm_weight,
        message_norm_bias,
    ):
        with jt.no_grad():
            hidden = hpe_hidden_cuda(
                geometry,
                hpe_input_weight,
                hpe_input_bias,
                hpe_norm_weight,
                hpe_norm_bias,
            )
            contribution = project_hpe_output(
                hidden, hpe_output_weight, hpe_output_bias
            )
        output = _reduce_messages_cuda(
            content,
            contribution,
            indices,
            message_norm_weight,
            message_norm_bias,
        )
        self.saved_tensors = (
            content,
            geometry,
            indices,
            hpe_input_weight,
            hpe_input_bias,
            hpe_output_weight,
            hpe_output_bias,
            hpe_norm_weight,
            hpe_norm_bias,
            message_norm_weight,
            message_norm_bias,
            output,
        )
        return output

    def grad(self, grad_output):
        (
            content,
            geometry,
            indices,
            hpe_input_weight,
            hpe_input_bias,
            hpe_output_weight,
            hpe_output_bias,
            hpe_norm_weight,
            hpe_norm_bias,
            message_norm_weight,
            message_norm_bias,
            output,
        ) = self.saved_tensors
        channels, hidden_channels = hpe_output_weight.shape

        with jt.no_grad():
            hidden = hpe_hidden_cuda(
                geometry,
                hpe_input_weight,
                hpe_input_bias,
                hpe_norm_weight,
                hpe_norm_bias,
            )
            contribution = project_hpe_output(
                hidden, hpe_output_weight, hpe_output_bias
            )

        (
            grad_content,
            grad_contribution,
            grad_message_norm_weight,
            grad_message_norm_bias,
        ) = _reduce_messages_grad(
            content,
            contribution,
            indices,
            message_norm_weight,
            message_norm_bias,
            output,
            grad_output,
        )

        flat_contribution_grad = grad_contribution.reshape(-1, channels)
        flat_hidden = hidden.reshape(-1, hidden_channels)
        grad_hidden = jt.matmul(flat_contribution_grad, hpe_output_weight)
        grad_hpe_output_weight = jt.matmul(
            flat_contribution_grad.transpose(0, 1), flat_hidden
        )
        grad_hpe_output_bias = flat_contribution_grad.sum(dim=0)

        preactivation = project_hpe_input(geometry, hpe_input_weight, hpe_input_bias)
        (
            grad_preactivation,
            grad_hpe_norm_weight,
            grad_hpe_norm_bias,
        ) = hpe_activation_grad(
            preactivation,
            hpe_norm_weight,
            hpe_norm_bias,
            grad_hidden.reshape(preactivation.shape),
        )
        flat_preactivation_grad = grad_preactivation.reshape(-1, hidden_channels)
        flat_geometry = geometry.reshape(-1, geometry.shape[-1])
        grad_geometry = jt.matmul(flat_preactivation_grad, hpe_input_weight).reshape(
            geometry.shape
        )
        grad_hpe_input_weight = jt.matmul(
            flat_preactivation_grad.transpose(0, 1), flat_geometry
        )
        grad_hpe_input_bias = flat_preactivation_grad.sum(dim=0)

        return (
            grad_content,
            grad_geometry,
            None,
            grad_hpe_input_weight,
            grad_hpe_input_bias,
            grad_hpe_output_weight,
            grad_hpe_output_bias,
            grad_hpe_norm_weight,
            grad_hpe_norm_bias,
            grad_message_norm_weight,
            grad_message_norm_bias,
        )


def reduce_neighbor_messages(
    content,
    geometry,
    indices,
    hpe_input_weight,
    hpe_input_bias,
    hpe_output_weight,
    hpe_output_bias,
    hpe_norm_weight,
    hpe_norm_bias,
    message_norm_weight,
    message_norm_bias,
):
    """Encode relative geometry and max-reduce normalized neighbor messages.

    content  : (B, 2C, N), projected neighbor and center terms
    geometry : (B, N, K, D), compact relative coordinates
    indices  : (B, N, K)
    output   : (B, C, N)
    """
    channels, hidden_channels = hpe_output_weight.shape
    input_hidden_channels, input_channels = hpe_input_weight.shape
    if content.shape[1] != 2 * channels:
        raise ValueError(
            f"Expected {2 * channels} content channels, got {content.shape[1]}"
        )
    if input_hidden_channels != hidden_channels:
        raise ValueError(
            "HPE projections disagree on hidden width: "
            f"{input_hidden_channels} != {hidden_channels}"
        )
    if geometry.shape[-1] != input_channels:
        raise ValueError(
            f"Expected {input_channels} geometry channels, got {geometry.shape[-1]}"
        )
    if tuple(geometry.shape[:3]) != tuple(indices.shape):
        raise ValueError(
            f"Geometry shape {geometry.shape[:3]} does not match indices "
            f"{indices.shape}"
        )
    if content.shape[0] != indices.shape[0] or content.shape[2] != indices.shape[1]:
        raise ValueError(
            f"Content shape {content.shape} does not match indices {indices.shape}"
        )

    float_inputs = (
        content,
        geometry,
        hpe_input_weight,
        hpe_input_bias,
        hpe_output_weight,
        hpe_output_bias,
        hpe_norm_weight,
        hpe_norm_bias,
        message_norm_weight,
        message_norm_bias,
    )
    use_cuda = (
        jt.flags.use_cuda
        and all(value.dtype == "float32" for value in float_inputs)
        and indices.dtype == "int32"
        and channels <= _CUDA_WIDTH_LIMIT
        and hidden_channels <= _CUDA_WIDTH_LIMIT
    )
    if not use_cuda:
        return _reduce_messages_reference(
            content,
            geometry,
            indices,
            hpe_input_weight,
            hpe_input_bias,
            hpe_output_weight,
            hpe_output_bias,
            hpe_norm_weight,
            hpe_norm_bias,
            message_norm_weight,
            message_norm_bias,
        )
    if jt.flags.no_grad:
        hidden = hpe_hidden_cuda(
            geometry,
            hpe_input_weight,
            hpe_input_bias,
            hpe_norm_weight,
            hpe_norm_bias,
        )
        contribution = project_hpe_output(hidden, hpe_output_weight, hpe_output_bias)
        return _reduce_messages_cuda(
            content,
            contribution,
            indices,
            message_norm_weight,
            message_norm_bias,
        )
    return _NeighborMessageFunction()(
        content,
        geometry,
        indices,
        hpe_input_weight,
        hpe_input_bias,
        hpe_output_weight,
        hpe_output_bias,
        hpe_norm_weight,
        hpe_norm_bias,
        message_norm_weight,
        message_norm_bias,
    )
