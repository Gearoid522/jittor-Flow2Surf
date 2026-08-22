"""Denoise test point clouds with a trained Flow2Surf model."""

import argparse
import logging
import os

import jittor as jt
import numpy as np
import yaml

from flow2surf.dataset import load_dataset_spec, test_cloud_paths
from flow2surf.inference import make_rng, patch_denoise
from flow2surf.models import build_model
from flow2surf.runtime import configure_logging, seed_runtime

log = logging.getLogger("flow2surf.predict")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/default.yaml", help="Model and inference config"
    )
    parser.add_argument("--dataset", default="datasets/A.yaml")
    parser.add_argument("--model", required=True, help="Checkpoint path (.pkl)")
    parser.add_argument("--out", default="results", help="Output directory")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    dataset = load_dataset_spec(args.dataset)

    configure_logging()

    model_path = args.model

    seed = cfg["inference_seed"]
    num_steps = cfg["num_steps"]

    if os.path.exists(args.out):
        raise SystemExit(f"Output already exists: {args.out}")

    seed_runtime(seed)
    jt.flags.use_cuda = 1
    model = build_model(cfg)
    model.load(model_path)
    model.eval()
    log.info(f"Loaded model: {model_path}")
    log.info(f"Dataset: {args.dataset} ({dataset.name})")
    log.info(
        f"Inference: config={args.config} steps={num_steps} seed_k={cfg['seed_k']}"
    )

    total = 0

    for synset, model_id, noisy_path in test_cloud_paths(dataset):
        pc_noisy = np.load(noisy_path)  # (N, 3) float32

        denoised = patch_denoise(
            model,
            pc_noisy,
            patch_size=cfg["patch_size"],
            seed_k=cfg["seed_k"],
            num_steps=num_steps,
            rng=make_rng(seed, synset, model_id),
            num_chunks=cfg["inference_chunks"],
            patch_agg_beta=cfg["patch_agg_beta"],
            alpha_clean_power=cfg["alpha_clean_power"],
            alpha_noisy_power=cfg["alpha_noisy_power"],
        )

        out_dir = os.path.join(args.out, "shapenet", synset, model_id)
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, "denoised.npy"), denoised)
        total += 1

        if total % 10 == 0:
            log.info(f"Processed {total} samples")

    log.info(f"Saved {total} samples -> {args.out}/")


if __name__ == "__main__":
    main()
