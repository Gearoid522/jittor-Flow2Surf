"""Global and candidate-restricted k-nearest-neighbor search."""

import jittor as jt

from ._sources import load_cuda

_CUDA_K_LIMIT = 32
_CUDA_POINT_LIMIT = 1024
_CUDA_CANDIDATE_LIMIT = 256
_CUDA_HEADER = load_cuda("knn.cuh")
_CUDA_GLOBAL = load_cuda("knn_global.cu")
_CUDA_RESTRICTED = load_cuda("knn_restricted.cu")


def _knn_jittor(features, k, candidates=None):
    batch, _, points = features.shape
    if points <= 1:
        return jt.zeros((batch, points, k), dtype="int32")

    inner = jt.nn.bmm(features.permute(0, 2, 1), features)
    squared_norm = (features * features).sum(dim=1)
    if candidates is None:
        distance = squared_norm.unsqueeze(2) + squared_norm.unsqueeze(1) - 2.0 * inner
        point_index = jt.arange(points)
        self_mask = (
            point_index.reshape(1, points, 1) == point_index.reshape(1, 1, points)
        ).float32()
        distance = distance + self_mask * 1e10
        actual_k = min(k, points - 1)
    else:
        candidate_inner = jt.gather(inner, 2, candidates)
        batch_offset = (jt.arange(batch) * points).reshape(batch, 1, 1)
        candidate_norm = squared_norm.reshape(-1)[
            (candidates + batch_offset).reshape(-1)
        ].reshape(candidates.shape)
        distance = squared_norm.unsqueeze(2) + candidate_norm - 2.0 * candidate_inner
        actual_k = k

    indices = (-distance).topk(k=actual_k, dim=-1)[1]
    if candidates is not None:
        return jt.gather(candidates, 2, indices)
    if actual_k < k:
        padding = indices[:, :, -1:].expand(batch, points, k - actual_k)
        indices = jt.concat([indices, padding], dim=-1)
    return indices


def knn_indices(features, k, candidates=None):
    """Find global kNN or rerank a supplied candidate neighborhood.

    features   : (B, C, N)
    candidates : optional (B, N, P), with P >= k
    output     : (B, N, k)
    """
    batch, _, points = features.shape
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if points <= 1:
        return jt.zeros((batch, points, k), dtype="int32")

    if candidates is not None:
        if tuple(candidates.shape[:2]) != (batch, points):
            raise ValueError(
                f"Candidates {candidates.shape} do not match features {features.shape}"
            )
        candidate_count = candidates.shape[2]
        if candidate_count < k:
            raise ValueError(f"Expected at least {k} candidates, got {candidate_count}")
        if candidate_count == k:
            return candidates
        if (
            not jt.flags.use_cuda
            or features.dtype != "float32"
            or candidate_count > _CUDA_CANDIDATE_LIMIT
            or k > _CUDA_K_LIMIT
        ):
            return _knn_jittor(features, k, candidates)
        return jt.code(
            (batch, points, k),
            "int32",
            [features, candidates],
            cuda_header=_CUDA_HEADER,
            cuda_src=_CUDA_RESTRICTED,
        ).stop_grad()

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
