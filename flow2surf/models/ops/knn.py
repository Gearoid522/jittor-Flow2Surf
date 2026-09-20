"""Global k-nearest-neighbor search for point features."""

import jittor as jt

from ._sources import load_cuda

_CUDA_K_LIMIT = 32
_CUDA_POINT_LIMIT = 1024
_CUDA_HEADER = load_cuda("knn.cuh")
_CUDA_GLOBAL = load_cuda("knn_global.cu")


def _knn_jittor(features, k):
    batch, _, points = features.shape
    if points <= 1:
        return jt.zeros((batch, points, k), dtype="int32")

    inner = jt.nn.bmm(features.permute(0, 2, 1), features)
    squared_norm = (features * features).sum(dim=1)
    distance = squared_norm.unsqueeze(2) + squared_norm.unsqueeze(1) - 2.0 * inner
    point_index = jt.arange(points)
    self_mask = (
        point_index.reshape(1, points, 1) == point_index.reshape(1, 1, points)
    ).float32()
    distance = distance + self_mask * 1e10
    actual_k = min(k, points - 1)

    indices = (-distance).topk(k=actual_k, dim=-1)[1]
    if actual_k < k:
        padding = indices[:, :, -1:].expand(batch, points, k - actual_k)
        indices = jt.concat([indices, padding], dim=-1)
    return indices


def knn_indices(features, k):
    """Find k nearest non-self neighbors in feature space.

    features : (B, C, N)
    output   : (B, N, k)
    """
    batch, _, points = features.shape
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if points <= 1:
        return jt.zeros((batch, points, k), dtype="int32")

    if (
        not jt.flags.use_cuda
        or features.dtype != "float32"
        or points > _CUDA_POINT_LIMIT
        or k > _CUDA_K_LIMIT
    ):
        return _knn_jittor(features, k)

    actual_k = min(k, points - 1)
    indices = jt.code(
        (batch, points, actual_k),
        "int32",
        [features],
        cuda_header=_CUDA_HEADER,
        cuda_src=_CUDA_GLOBAL,
    ).stop_grad()
    if actual_k < k:
        padding = indices[:, :, -1:].expand(batch, points, k - actual_k)
        indices = jt.concat([indices, padding], dim=-1)
    return indices
