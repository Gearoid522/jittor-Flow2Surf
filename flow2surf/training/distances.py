"""Bidirectional nearest-distance computation for point-set losses."""

from pathlib import Path

import jittor as jt
from jittor import Function

_CUDA_DIR = Path(__file__).parent / "cuda"


def _load_cuda(name):
    return (_CUDA_DIR / name).read_text(encoding="utf-8")


_CUDA_NEAREST_FORWARD = _load_cuda("nearest_distance_forward.cu")
_CUDA_NEAREST_BACKWARD = _load_cuda("nearest_distance_backward.cu")


class _NearestDistanceFunction(Function):
    """Bidirectional nearest squared distances with an exact coordinate gradient."""

    def execute(self, first, second):
        batch, first_count, _ = first.shape
        second_count = second.shape[1]
        first_distance, first_index, second_distance, second_index = jt.code(
            [
                (batch, first_count),
                (batch, first_count),
                (batch, second_count),
                (batch, second_count),
            ],
            [first.dtype, "int32", second.dtype, "int32"],
            [first, second],
            cuda_header="#include <cfloat>",
            cuda_src=_CUDA_NEAREST_FORWARD,
        )
        first_index.stop_grad()
        second_index.stop_grad()
        self.saved_tensors = first, second, first_index, second_index
        return first_distance, first_index, second_distance, second_index

    def grad(
        self,
        grad_first_distance,
        grad_first_index,
        grad_second_distance,
        grad_second_index,
    ):
        first, second, first_index, second_index = self.saved_tensors
        if grad_first_distance is None:
            grad_first_distance = jt.zeros(first_index.shape, dtype=first.dtype)
        if grad_second_distance is None:
            grad_second_distance = jt.zeros(second_index.shape, dtype=second.dtype)
        grad_first, grad_second = jt.code(
            [first.shape, second.shape],
            [first.dtype, second.dtype],
            [
                first,
                second,
                first_index,
                second_index,
                grad_first_distance,
                grad_second_distance,
            ],
            cuda_src=_CUDA_NEAREST_BACKWARD,
        )
        return grad_first, grad_second


def nearest_squared_distances(first, second):
    """Return distances and indices from each 3D point set to the other.

    Inputs are (B, M, 3) and (B, N, 3). Outputs are first distances and second
    indices shaped (B, M), then second distances and first indices shaped (B, N).
    """
    if len(first.shape) != 3 or first.shape[2] != 3:
        raise ValueError(f"Expected first shaped (B, M, 3), got {first.shape}")
    if len(second.shape) != 3 or second.shape[2] != 3:
        raise ValueError(f"Expected second shaped (B, N, 3), got {second.shape}")
    if first.shape[0] != second.shape[0]:
        raise ValueError(f"Batch sizes differ: {first.shape[0]} and {second.shape[0]}")
    if first.shape[1] < 1 or second.shape[1] < 1:
        raise ValueError("Point sets must be nonempty")
    if first.dtype != second.dtype:
        raise ValueError(f"Dtypes differ: {first.dtype} and {second.dtype}")
    if jt.flags.use_cuda and first.dtype == "float32" and second.dtype == "float32":
        return _NearestDistanceFunction()(first, second)

    first_norm = (first * first).sum(dim=-1)
    second_norm = (second * second).sum(dim=-1)
    inner = jt.nn.bmm(first, second.transpose(0, 2, 1))
    distance = jt.maximum(
        first_norm.unsqueeze(2) + second_norm.unsqueeze(1) - 2.0 * inner,
        0.0,
    )
    second_index, first_distance = jt.argmin(distance, dim=2)
    first_index, second_distance = jt.argmin(distance, dim=1)
    return first_distance, second_index, second_distance, first_index
