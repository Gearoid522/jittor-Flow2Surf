"""Deterministic patch-wise reverse-flow inference."""

import hashlib

import jittor as jt
import numpy as np
from scipy.spatial import cKDTree

from .flow import alpha_schedule


def make_rng(seed, *parts):
    """Create a deterministic NumPy generator from a seed and namespace parts."""
    key = "::".join([str(seed), *map(str, parts)]).encode("utf-8")
    digest = hashlib.blake2b(key, digest_size=8).digest()
    return np.random.default_rng(int.from_bytes(digest, "little") % (2**32))


def _farthest_point_indices(points, count, rng):
    """Select count farthest-point indices from points shaped (N, 3)."""
    num_points = points.shape[0]
    selected = np.empty(count, dtype=np.int64)
    selected[0] = rng.integers(num_points)
    min_distance = np.full(num_points, np.inf)
    distance = np.empty(num_points, dtype=points.dtype)
    workspace = np.empty_like(distance)

    for i in range(1, count):
        center = points[selected[i - 1]]
        np.subtract(points[:, 0], center[0], out=distance)
        np.square(distance, out=distance)
        np.subtract(points[:, 1], center[1], out=workspace)
        np.square(workspace, out=workspace)
        np.add(distance, workspace, out=distance)
        np.subtract(points[:, 2], center[2], out=workspace)
        np.square(workspace, out=workspace)
        np.add(distance, workspace, out=distance)
        np.minimum(min_distance, distance, out=min_distance)
        selected[i] = min_distance.argmax()

    return selected


def patch_seed_count(num_points, patch_size, seed_k):
    """Return the patch-center count for one point cloud."""
    num_points = int(num_points)
    patch_size = int(patch_size)
    seed_k = int(seed_k)
    if num_points < 1:
        raise ValueError(f"num_points must be >= 1, got {num_points}")
    if patch_size < 1:
        raise ValueError(f"patch_size must be >= 1, got {patch_size}")
    if seed_k < 1:
        raise ValueError(f"seed_k must be >= 1, got {seed_k}")
    effective_patch_size = min(patch_size, num_points)
    return min(
        num_points,
        max(1, int(seed_k * num_points / effective_patch_size)),
    )


def patch_seed_indices(points, patch_size, seed_k, rng):
    """Select deterministic farthest-point patch centers from an (N, 3) cloud."""
    num_patches = patch_seed_count(len(points), patch_size, seed_k)
    return _farthest_point_indices(points, num_patches, rng)


def patch_denoise(
    model,
    pc_noisy,
    patch_size,
    seed_k,
    num_steps,
    num_chunks,
    rng=None,
    patch_seed_idx=None,
    patch_agg_beta=4.0,
    alpha_clean_power=1.0,
    alpha_noisy_power=1.0,
):
    """Denoise one normalized point cloud by patch-wise reverse integration.

    Euler integration traverses alpha from 1 to 0. At each level, overlapping
    patch predictions form one center-weighted full-cloud velocity field.
    Patch membership and weights are fixed from the noisy input. num_chunks
    bounds patch-forward batches per integration step. patch_seed_idx optionally
    supplies fixed farthest-point center indices shaped (P,).
    """
    if rng is None:
        rng = np.random.default_rng(0)

    num_points = pc_noisy.shape[0]
    if num_chunks < 1:
        raise ValueError(f"num_chunks must be >= 1, got {num_chunks}")
    patch_size = min(patch_size, num_points)
    num_patches = patch_seed_count(num_points, patch_size, seed_k)

    if patch_seed_idx is None:
        seed_idx = patch_seed_indices(pc_noisy, patch_size, seed_k, rng)
    else:
        seed_idx = np.asarray(patch_seed_idx, dtype=np.int64)
        if seed_idx.shape != (num_patches,):
            raise ValueError(
                f"patch_seed_idx must have shape ({num_patches},), "
                f"got {seed_idx.shape}"
            )
        if np.any((seed_idx < 0) | (seed_idx >= num_points)):
            raise ValueError("patch_seed_idx contains an out-of-range point index")

    def build_patch_index(points):
        tree = cKDTree(points)
        seed_points = points[seed_idx]
        patch_distances, neighbor_idx = tree.query(
            seed_points,
            k=patch_size,
            workers=-1,
        )
        if patch_size == 1:
            patch_distances = patch_distances[:, None]
            neighbor_idx = neighbor_idx[:, None]
        patch_distances = patch_distances / (patch_distances[:, -1:] + 1e-8)
        return neighbor_idx, patch_distances

    def build_aggregation(neighbor_idx, patch_distances):
        point_idx = neighbor_idx.reshape(-1)
        radial_position = patch_distances.reshape(-1)
        weights = np.exp(-patch_agg_beta * radial_position * radial_position)
        weight_sum = np.bincount(
            point_idx,
            weights=weights,
            minlength=num_points,
        )
        uncovered = weight_sum == 0
        return point_idx, weights, weight_sum, uncovered

    chunk_size = (num_patches + num_chunks - 1) // num_chunks

    neighbor_idx, patch_distances = build_patch_index(pc_noisy)
    aggregation = build_aggregation(neighbor_idx, patch_distances)

    def predict_velocity(points, alpha_value):
        patches = points[neighbor_idx]
        patch_velocity = np.zeros_like(patches)

        with jt.no_grad():
            for start in range(0, num_patches, chunk_size):
                batch = patches[start : start + chunk_size]
                batch_size = batch.shape[0]
                centers = points[seed_idx[start : start + batch_size]][:, None, :]
                model_input = jt.array(batch - centers).permute(0, 2, 1)
                alpha_input = jt.full((batch_size,), alpha_value)
                velocity = model(model_input, alpha_input).numpy()
                patch_velocity[start : start + batch_size] = velocity

        point_idx, weights, weight_sum, uncovered = aggregation
        return patch_velocity, point_idx, weights, weight_sum, uncovered

    def aggregate_votes(votes, point_idx, weights, weight_sum):
        flat_votes = votes.reshape(-1, 3)
        velocity_sum = np.stack(
            [
                np.bincount(
                    point_idx,
                    weights=weights * flat_votes[:, channel],
                    minlength=num_points,
                )
                for channel in range(3)
            ],
            axis=1,
        )
        return (velocity_sum / (weight_sum[:, None] + 1e-8)).astype(np.float32)

    alphas = alpha_schedule(
        num_steps,
        alpha_clean_power,
        alpha_noisy_power,
    )
    current = pc_noisy.copy()
    for step in range(num_steps - 1, -1, -1):
        alpha_current = float(alphas[step])
        alpha_previous = float(alphas[step - 1]) if step > 0 else 0.0
        alpha_step = alpha_current - alpha_previous

        patch_velocity, point_idx, weights, weight_sum, uncovered = predict_velocity(
            current, alpha_current
        )
        velocity = aggregate_votes(
            patch_velocity,
            point_idx,
            weights,
            weight_sum,
        )
        updated = current - alpha_step * velocity
        if uncovered.any():
            updated[uncovered] = current[uncovered]
        current = updated

    return current.astype(np.float32)
