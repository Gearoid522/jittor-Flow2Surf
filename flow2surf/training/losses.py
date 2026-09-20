"""Patch-level objectives for conditional-flow training."""

import jittor as jt

from .distances import nearest_squared_distances


def dcd_loss(pred, target, alpha):
    """Return density-aware Chamfer loss per point set.

    Reciprocal nearest-neighbor multiplicity penalizes many-to-one matches; an
    exponential kernel maps squared distances to a loss in [0, 1].
    """
    batch_size, pred_count, _ = pred.shape
    target_count = target.shape[1]
    pred_distance, target_idx, target_distance, pred_idx = nearest_squared_distances(
        pred, target
    )
    target_multiplicity = jt.zeros((batch_size, target_count)).scatter(
        1,
        target_idx,
        jt.ones((batch_size, pred_count)),
        reduce="add",
    )
    pred_multiplicity = jt.zeros((batch_size, pred_count)).scatter(
        1,
        pred_idx,
        jt.ones((batch_size, target_count)),
        reduce="add",
    )
    pred_weight = 1.0 / jt.maximum(
        jt.gather(target_multiplicity, 1, target_idx),
        1.0,
    )
    target_weight = 1.0 / jt.maximum(
        jt.gather(pred_multiplicity, 1, pred_idx),
        1.0,
    )
    pred_term = (1.0 - pred_weight * jt.exp(-alpha * pred_distance)).mean(dim=1)
    target_term = (1.0 - target_weight * jt.exp(-alpha * target_distance)).mean(dim=1)
    return 0.5 * (pred_term + target_term)


def velocity_loss(pred_velocity, target_velocity):
    """Return mean squared L2 velocity error for each patch in the batch."""
    error = pred_velocity - target_velocity
    return (error * error).sum(dim=-1).mean(dim=1)
