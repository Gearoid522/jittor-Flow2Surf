"""Score a Flow2Surf checkpoint on a fixed synthetic-noise grid.

Usage:
    python eval.py --config configs/default.yaml --dataset datasets/A.yaml \
        --model <ckpt.pkl> --meshes 50

Each family-scale condition uses the same validation meshes. The output JSON
contains overall, per-family, and per-scale summaries.
"""

import argparse
import json
import logging
import os

import jittor as jt
import yaml

from flow2surf.dataset import load_dataset_spec, mesh_split_paths
from flow2surf.evaluation import noise_grid_mesh_score, summarize_noise_grid
from flow2surf.models import build_model
from flow2surf.runtime import configure_logging, run_stamp, seed_runtime

log = logging.getLogger("flow2surf.eval")

EVAL_NOISE_TYPES = ("gaussian", "laplace")
EVAL_NOISE_STD_LEVELS = (0.005, 0.010, 0.020, 0.030, 0.040)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--dataset", default="datasets/A.yaml")
    ap.add_argument("--model", required=True, help="Checkpoint .pkl to score")
    ap.add_argument(
        "--meshes", type=int, default=50, help="Validation meshes per condition"
    )
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    dataset = load_dataset_spec(args.dataset)

    configure_logging()

    # Inference settings come from config; mesh count is an eval flag.
    seed = int(cfg["inference_seed"])
    n_meshes = int(args.meshes)
    if n_meshes < 1:
        raise ValueError(f"--meshes must be >= 1, got {n_meshes}")

    # Deterministic eval for a fixed config, checkpoint, and mesh count.
    seed_runtime(seed)
    jt.flags.use_cuda = 1

    model = build_model(cfg)
    model.load(args.model)
    model.eval()

    _, val_paths = mesh_split_paths(dataset)
    paths = val_paths[:n_meshes]
    if not paths:
        raise ValueError("Validation split is empty; nothing to score")

    architecture = {
        "feat_dim": cfg["feat_dim"],
        "encoder_dims": cfg["encoder_dims"],
        "k_neighbors": cfg["k_neighbors"],
        "num_blocks": cfg["num_blocks"],
        "alpha_dim": cfg["alpha_dim"],
        "ffn_ratio": cfg["ffn_ratio"],
        "decoder_dims": cfg["decoder_dims"],
    }
    inference = {
        "num_steps": cfg["num_steps"],
        "alpha_clean_power": cfg["alpha_clean_power"],
        "alpha_noisy_power": cfg["alpha_noisy_power"],
        "seed_k": cfg["seed_k"],
        "inference_chunks": cfg["inference_chunks"],
        "patch_agg_beta": cfg["patch_agg_beta"],
        "inference_seed": seed,
    }
    log.info("Model: %s", args.model)
    log.info("Config: %s", args.config)
    log.info("Dataset: %s (%s)", args.dataset, dataset.name)
    log.info(
        f"Scoring: {len(paths)} meshes * {len(EVAL_NOISE_TYPES)} families * "
        f"{len(EVAL_NOISE_STD_LEVELS)} levels"
    )

    mesh_results = []
    for i, path in enumerate(paths):
        mesh_result = noise_grid_mesh_score(
            model,
            path,
            int(cfg["seed"]) + i,
            cfg,
            EVAL_NOISE_TYPES,
            EVAL_NOISE_STD_LEVELS,
            num_chunks=cfg["inference_chunks"],
        )
        mesh_results.append(mesh_result)
        if (i + 1) % 10 == 0 or (i + 1) == len(paths):
            log.info("  [%2d/%d] meshes", i + 1, len(paths))

    results = summarize_noise_grid(
        mesh_results,
        EVAL_NOISE_TYPES,
        EVAL_NOISE_STD_LEVELS,
    )
    levels = tuple(f"{value:.3f}" for value in EVAL_NOISE_STD_LEVELS)
    overall = results["overall"]
    by_family = results["by_family"]
    stamp = run_stamp()
    result = {
        "timestamp": stamp,
        "model": args.model,
        "config": args.config,
        "dataset": args.dataset,
        "protocol": {
            "meshes_per_condition": len(mesh_results),
            "num_samples": cfg["num_samples"],
            "noise_families": list(EVAL_NOISE_TYPES),
            "noise_std_levels": list(EVAL_NOISE_STD_LEVELS),
        },
        "architecture": architecture,
        "inference": inference,
        "results": results,
    }

    log.info("-" * 76)
    for noise_type in EVAL_NOISE_TYPES:
        for level in levels:
            condition = by_family[noise_type]["by_level"][level]
            log.info(
                f"  {noise_type:<8} {level} | "
                f"score {condition['score']['mean']:.3f} "
                f"(cd {condition['cd']['score']['mean']:.3f}, "
                f"p2s {condition['p2s']['score']['mean']:.3f})"
            )
    log.info(
        f"  overall          | score {overall['score']['mean']:.3f} "
        f"(cd {overall['cd']['score']['mean']:.3f}, "
        f"p2s {overall['p2s']['score']['mean']:.3f})"
    )

    # Persistent JSON, auto-named by config and timestamp.
    cfg_name = os.path.splitext(os.path.basename(args.config))[0]
    out = f"eval_{cfg_name}_{stamp}.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    log.info("-" * 76)
    log.info("Wrote %s (%d meshes)", out, len(mesh_results))


if __name__ == "__main__":
    main()
