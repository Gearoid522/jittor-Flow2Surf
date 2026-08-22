"""Full-cloud metrics and deterministic mini-validation."""

import os

import jittor as jt
import numpy as np
import point_cloud_utils as pcu
from scipy.spatial import cKDTree

from .dataset import (
    add_noise,
    load_mesh,
    normalize_points,
    sample_noise_std,
    sample_surface,
)
from .inference import (
    make_rng,
    patch_denoise,
    patch_seed_indices,
)


def chamfer_distance(pred, target):
    """Compute bidirectional squared Chamfer distance between NumPy point sets."""
    pred_to_target, _ = cKDTree(target).query(pred, workers=-1)
    target_to_pred, _ = cKDTree(pred).query(target, workers=-1)
    return float((pred_to_target**2).mean() + (target_to_pred**2).mean())


def point_to_surface(points, vertices, faces):
    """Compute mean squared point-to-triangle-mesh distance."""
    distances, _, _ = pcu.closest_points_on_mesh(points, vertices, faces)
    return float((distances**2).mean())


def improvement_score(pred_value, noisy_value):
    """Return clipped percentage reduction relative to a noisy metric value."""
    score = 100.0 * (1.0 - pred_value / (noisy_value + 1e-12))
    return float(np.clip(score, 0.0, 100.0))


def _describe(values):
    """Return compact statistics for a non-empty scalar sequence."""
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "median": float(np.median(values)),
        "max": float(values.max()),
    }


def summarize_noise_grid(mesh_results, noise_types, noise_std_levels):
    """Aggregate per-mesh noise-grid metrics by family and scale."""
    if not mesh_results:
        raise ValueError("mesh_results must not be empty")
    noise_types = tuple(noise_types)
    levels = tuple(f"{float(value):.3f}" for value in noise_std_levels)

    def summarize(conditions):
        values = {
            metric: [
                mesh_result[noise_type][level][metric]
                for mesh_result in mesh_results
                for noise_type, level in conditions
            ]
            for metric in (
                "score",
                "cd",
                "p2s",
                "cd_pred",
                "cd_noisy",
                "p2s_pred",
                "p2s_noisy",
            )
        }
        return {
            "score": _describe(values["score"]),
            "cd": {
                "score": _describe(values["cd"]),
                "pred": float(np.mean(values["cd_pred"])),
                "noisy": float(np.mean(values["cd_noisy"])),
            },
            "p2s": {
                "score": _describe(values["p2s"]),
                "pred": float(np.mean(values["p2s_pred"])),
                "noisy": float(np.mean(values["p2s_noisy"])),
            },
        }

    by_family = {
        noise_type: {
            "overall": summarize([(noise_type, level) for level in levels]),
            "by_level": {
                level: summarize([(noise_type, level)]) for level in levels
            },
        }
        for noise_type in noise_types
    }
    by_level = {
        level: summarize([(noise_type, level) for noise_type in noise_types])
        for level in levels
    }
    return {
        "overall": summarize(
            [
                (noise_type, level)
                for noise_type in noise_types
                for level in levels
            ]
        ),
        "by_family": by_family,
        "by_level": by_level,
    }


def _mesh_identity(path):
    """Return ShapeNet synset and model identifiers for one mesh path."""
    synset = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(path))))
    model_id = os.path.basename(os.path.dirname(os.path.dirname(path)))
    return synset, model_id


def _normalized_mesh(path, sample_seed, num_samples):
    """Sample one mesh and return normalized points, vertices, and faces."""
    vertices, faces = load_mesh(path)
    raw_points = sample_surface(
        vertices,
        faces,
        num_samples,
        np.random.default_rng(sample_seed),
    )
    clean, center, scale = normalize_points(raw_points)
    normalized_vertices = (np.asarray(vertices, dtype=np.float32) - center) / scale
    return clean, normalized_vertices, faces


def _mini_baseline(path, sample_seed, noise_type, cfg, seed, cache=None):
    """Build or retrieve deterministic mini-validation inputs for one mesh."""
    key = (
        path,
        sample_seed,
        noise_type,
        int(cfg["num_samples"]),
        tuple(map(float, cfg["noise_std_range"])),
        cfg["noise_std_sampling"],
        int(cfg["patch_size"]),
        int(cfg["seed_k"]),
        int(seed),
    )
    if cache is not None and key in cache:
        return cache[key]

    clean, normalized_vertices, faces = _normalized_mesh(
        path,
        sample_seed,
        cfg["num_samples"],
    )
    synset, model_id = _mesh_identity(path)
    noise_rng = make_rng(seed, "mini_noise", noise_type, synset, model_id)
    noise_std = sample_noise_std(cfg, noise_rng)
    noisy = add_noise(clean, noise_std, noise_type, noise_rng)
    seed_indices = patch_seed_indices(
        noisy,
        cfg["patch_size"],
        cfg["seed_k"],
        make_rng(seed, "mini_val", noise_type, synset, model_id),
    )

    baseline = {
        "clean": clean,
        "faces": faces,
        "normalized_vertices": normalized_vertices,
        "noisy": noisy,
        "seed_indices": seed_indices,
        "noisy_cd": chamfer_distance(noisy, clean),
        "noisy_p2s": point_to_surface(
            noisy,
            normalized_vertices,
            faces,
        ),
    }
    if cache is not None:
        cache[key] = baseline
    return baseline


@jt.no_grad()
def mini_val_score(model, paths, cfg, num_chunks, baseline_cache=None, sample_offset=0):
    """Return scores on deterministic noise from the training family."""
    model.eval()
    seed = cfg["inference_seed"]
    num_chunks = int(num_chunks)
    noise_type = cfg["noise_type"]
    scores = []
    cd_scores = []
    p2s_scores = []

    for index, path in enumerate(paths):
        sample_seed = int(cfg["seed"]) + sample_offset + index
        baseline = _mini_baseline(
            path,
            sample_seed,
            noise_type,
            cfg,
            seed,
            baseline_cache,
        )
        clean = baseline["clean"]
        denoised = patch_denoise(
            model,
            baseline["noisy"],
            patch_size=cfg["patch_size"],
            seed_k=cfg["seed_k"],
            num_steps=cfg["num_steps"],
            patch_seed_idx=baseline["seed_indices"],
            patch_agg_beta=cfg["patch_agg_beta"],
            num_chunks=num_chunks,
            alpha_clean_power=cfg["alpha_clean_power"],
            alpha_noisy_power=cfg["alpha_noisy_power"],
        )
        pred_cd = chamfer_distance(denoised, clean)
        pred_p2s = point_to_surface(
            denoised,
            baseline["normalized_vertices"],
            baseline["faces"],
        )
        cd_score = improvement_score(pred_cd, baseline["noisy_cd"])
        p2s_score = improvement_score(pred_p2s, baseline["noisy_p2s"])
        scores.append(0.5 * (cd_score + p2s_score))
        cd_scores.append(cd_score)
        p2s_scores.append(p2s_score)

    def mean(values):
        return float(np.mean(values)) if values else 0.0

    return {
        "score": mean(scores),
        "cd_score": mean(cd_scores),
        "p2s_score": mean(p2s_scores),
        "num_meshes": len(paths),
    }


def prepare_noise_grid_mesh(
    path,
    sample_seed,
    cfg,
    noise_types,
    noise_std_levels,
):
    """Prepare one mesh and its deterministic noisy evaluation conditions."""
    clean, normalized_vertices, faces = _normalized_mesh(
        path,
        sample_seed,
        cfg["num_samples"],
    )
    synset, model_id = _mesh_identity(path)
    seed = int(cfg["inference_seed"])
    by_noise = {}

    for noise_type in noise_types:
        by_level = {}
        for noise_std in noise_std_levels:
            noise_std = float(noise_std)
            level = f"{noise_std:.3f}"
            noise_rng = make_rng(
                seed,
                "eval_noise",
                noise_type,
                level,
                synset,
                model_id,
            )
            noisy = add_noise(clean, noise_std, noise_type, noise_rng)
            seed_indices = patch_seed_indices(
                noisy,
                cfg["patch_size"],
                cfg["seed_k"],
                make_rng(
                    seed,
                    "eval_patches",
                    noise_type,
                    level,
                    synset,
                    model_id,
                ),
            )
            by_level[level] = {
                "noisy": noisy,
                "seed_indices": seed_indices,
                "cd_noisy": chamfer_distance(noisy, clean),
                "p2s_noisy": point_to_surface(
                    noisy,
                    normalized_vertices,
                    faces,
                ),
            }
        by_noise[noise_type] = by_level

    return {
        "clean": clean,
        "normalized_vertices": normalized_vertices,
        "faces": faces,
        "conditions": by_noise,
    }


@jt.no_grad()
def score_noise_grid_mesh(model, prepared, cfg, num_chunks):
    """Score one prepared mesh with the configured reverse integration."""
    model.eval()
    clean = prepared["clean"]
    normalized_vertices = prepared["normalized_vertices"]
    faces = prepared["faces"]
    by_noise = {}

    for noise_type, conditions in prepared["conditions"].items():
        by_level = {}
        for level, condition in conditions.items():
            denoised = patch_denoise(
                model,
                condition["noisy"],
                patch_size=cfg["patch_size"],
                seed_k=cfg["seed_k"],
                num_steps=cfg["num_steps"],
                patch_seed_idx=condition["seed_indices"],
                patch_agg_beta=cfg["patch_agg_beta"],
                num_chunks=num_chunks,
                alpha_clean_power=cfg["alpha_clean_power"],
                alpha_noisy_power=cfg["alpha_noisy_power"],
            )
            pred_cd = chamfer_distance(denoised, clean)
            pred_p2s = point_to_surface(denoised, normalized_vertices, faces)
            cd_score = improvement_score(pred_cd, condition["cd_noisy"])
            p2s_score = improvement_score(pred_p2s, condition["p2s_noisy"])
            by_level[level] = {
                "score": 0.5 * (cd_score + p2s_score),
                "cd": cd_score,
                "p2s": p2s_score,
                "cd_pred": pred_cd,
                "cd_noisy": condition["cd_noisy"],
                "p2s_pred": pred_p2s,
                "p2s_noisy": condition["p2s_noisy"],
            }
        by_noise[noise_type] = by_level

    return by_noise


def noise_grid_mesh_score(
    model,
    path,
    sample_seed,
    cfg,
    noise_types,
    noise_std_levels,
    num_chunks,
):
    """Prepare and score one mesh under a fixed family-scale noise grid."""
    prepared = prepare_noise_grid_mesh(
        path,
        sample_seed,
        cfg,
        noise_types,
        noise_std_levels,
    )
    return score_noise_grid_mesh(model, prepared, cfg, num_chunks)
