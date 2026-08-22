"""Conditional-flow mathematics shared by training and inference."""

import numpy as np


def flow_state(clean, noisy, alpha):
    """Interpolate from clean alpha=0 to noisy alpha=1.

    alpha must be broadcast-compatible with clean and noisy.
    """
    return clean + alpha * (noisy - clean)


def flow_velocity(clean, noisy):
    """Return dX_alpha/dalpha for the linear clean-to-noisy path."""
    return noisy - clean


def reconstruct_clean(x_alpha, velocity, alpha):
    """Return x_alpha - alpha * velocity, the implied clean endpoint."""
    return x_alpha - alpha * velocity


def alpha_schedule(num_steps, clean_power, noisy_power):
    """Return endpoint-controlled integration levels from near-clean to noisy.

    For s_i = i / num_steps,
    alpha_i = s_i^clean_power / (s_i^clean_power + (1-s_i)^noisy_power).
    Reverse inference traverses these levels from alpha=1 toward alpha=0.
    """
    num_steps = int(num_steps)
    if num_steps < 1:
        raise ValueError("num_steps must be >= 1")
    clean_power = float(clean_power)
    noisy_power = float(noisy_power)
    for name, value in (
        ("clean_power", clean_power),
        ("noisy_power", noisy_power),
    ):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive, got {value}")

    s = np.arange(1, num_steps + 1, dtype=np.float64) / num_steps
    clean_term = s**clean_power
    noisy_term = (1.0 - s) ** noisy_power
    alpha = (clean_term / (clean_term + noisy_term)).astype(np.float32)
    if not np.isfinite(alpha).all() or alpha[0] <= 0 or np.any(np.diff(alpha) <= 0):
        raise ValueError(
            "schedule parameters must produce strictly increasing positive "
            "float32 levels"
        )
    return alpha
