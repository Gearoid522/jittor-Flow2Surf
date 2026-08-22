"""Offset attention with an exact bounded-memory training path."""

import jittor as jt
from jittor import Function, nn

from ._sources import load_cuda

_ATTENTION_TILE = 256
_CUDA_HEADER = load_cuda("offset_attention.cuh")
_CUDA_FORWARD = load_cuda("offset_attention_forward.cu")
_CUDA_BACKWARD = load_cuda("offset_attention_backward.cu")
_TILE_ONES = None


def _aggregate_jittor(query, key, value):
    scores = jt.nn.bmm(query, key)
    column_weights = nn.softmax(scores, dim=1)
    weights = column_weights / (column_weights.sum(dim=-1, keepdims=True) + 1e-9)
    return jt.nn.bmm(weights, value)


def _tile_ones():
    global _TILE_ONES
    if _TILE_ONES is None:
        _TILE_ONES = jt.ones((_ATTENTION_TILE, 1), dtype="float32")
    return _TILE_ONES


def _aggregate_cuda(query, key, value):
    batch, points, _ = query.shape
    channels = value.shape[2]
    output, row_sums, _ = jt.code(
        [
            (batch, points, channels),
            (batch, points),
            (batch, _ATTENTION_TILE, points),
        ],
        [query.dtype] * 3,
        [query, key, value, _tile_ones()],
        cuda_header=_CUDA_HEADER,
        cuda_src=_CUDA_FORWARD,
    )
    return output, row_sums


def _aggregate_cuda_grad(query, key, value, output, row_sums, grad_output):
    batch, points, _ = query.shape
    channels = value.shape[2]
    grad_query, grad_key, grad_value, _, _, _, _ = jt.code(
        [
            query.shape,
            key.shape,
            value.shape,
            (batch, _ATTENTION_TILE, points),
            (batch, _ATTENTION_TILE, points),
            (batch, points),
            (batch, points, channels),
        ],
        [query.dtype] * 7,
        [query, key, value, output, row_sums, grad_output],
        cuda_header=_CUDA_HEADER,
        cuda_src=_CUDA_BACKWARD,
    )
    return grad_query, grad_key, grad_value


class _OffsetAttentionFunction(Function):
    """Evaluate exact offset attention in bounded-memory key tiles."""

    def execute(self, query, key, value):
        with jt.no_grad():
            output, row_sums = _aggregate_cuda(query, key, value)
        self.saved_tensors = query, key, value, output, row_sums
        return output

    def grad(self, grad_output):
        query, key, value, output, row_sums = self.saved_tensors
        with jt.no_grad():
            return _aggregate_cuda_grad(
                query, key, value, output, row_sums, grad_output
            )


def offset_attention_aggregate(query, key, value):
    """Aggregate values using normalized offset-attention weights."""
    if len(query.shape) != 3 or len(key.shape) != 3 or len(value.shape) != 3:
        raise ValueError("query, key, and value must be rank-3 tensors")
    if (
        query.shape[0] != key.shape[0]
        or query.shape[0] != value.shape[0]
        or query.shape[1] != key.shape[2]
        or query.shape[1] != value.shape[1]
        or query.shape[2] != key.shape[1]
    ):
        raise ValueError(
            f"Incompatible attention shapes: {query.shape}, {key.shape}, "
            f"{value.shape}"
        )
    use_tiled = (
        jt.flags.use_cuda
        and not jt.flags.no_grad
        and all(tensor.dtype == "float32" for tensor in (query, key, value))
    )
    if not use_tiled:
        return _aggregate_jittor(query, key, value)
    return _OffsetAttentionFunction()(query, key, value)
