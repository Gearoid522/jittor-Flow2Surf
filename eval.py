"""Score a checkpoint on the held-out val split (config-driven, reproducible).

Usage:
    python eval.py --config configs/default.yaml --model <ckpt.pkl>

Writes eval_<config>_<timestamp>.json with the setting and per-mesh scores.
"""

import argparse
import json
import os
import random
from datetime import datetime

import yaml
import numpy as np
import jittor as jt

from src.dataset import build_datasets
from predict import build_model
from train import mini_val_score


def describe(values):
    a = np.asarray(values, dtype=np.float64)
    q = np.percentile(a, [0, 25, 50, 75, 100])
    return {
        "mean": float(a.mean()),
        "std": float(a.std()),
        "min": float(q[0]),
        "p25": float(q[1]),
        "median": float(q[2]),
        "p75": float(q[3]),
        "max": float(q[4]),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--model", required=True, help="Checkpoint .pkl to score")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # everything that defines the run comes from the config
    seed = int(cfg.get("inference_seed", cfg.get("seed", 42)))
    n_meshes = int(cfg.get("eval_meshes", 200))

    # full determinism: identical inputs across runs, only the config differs
    random.seed(seed)
    np.random.seed(seed)
    jt.set_global_seed(seed)
    jt.flags.use_cuda = 1

    model = build_model(cfg)
    model.load(args.model)
    model.eval()

    _, val_ds = build_datasets(cfg)
    paths = val_ds.paths[:n_meshes]

    inference = {
        "num_steps": cfg["num_steps"],
        "seed_k": cfg.get("seed_k"),
        "patch_frame": cfg.get("patch_frame", "local"),
        "patch_aggregation": cfg.get("patch_aggregation", "residual"),
        "patch_agg_beta": cfg.get("patch_agg_beta", 4),
        "recompute_neighbors": cfg.get("recompute_neighbors", False),
        "seed": seed,
    }
    print(
        f"scoring {len(paths)} meshes | "
        + "  ".join(f"{k}={v}" for k, v in inference.items()),
        flush=True,
    )

    # score one mesh at a time to keep the full per-mesh distribution (the denoise
    # cost is identical to a batched call; only bookkeeping differs)
    rows = []
    for i, path in enumerate(paths):
        r = mini_val_score(model, [path], cfg)
        synset = os.path.basename(
            os.path.dirname(os.path.dirname(os.path.dirname(path)))
        )
        model_id = os.path.basename(os.path.dirname(os.path.dirname(path)))
        rows.append(
            {
                "synset": synset,
                "model_id": model_id,
                "score": r["score"],
                "cd": r["cd_score"],
                "p2s": r["p2s_score"],
            }
        )
        if (i + 1) % 10 == 0 or (i + 1) == len(paths):
            run_mean = float(np.mean([row["score"] for row in rows]))
            print(
                f"  [{i + 1:>4}/{len(paths)}] running score={run_mean:.3f}", flush=True
            )

    summary = {
        "score": describe([r["score"] for r in rows]),
        "cd": describe([r["cd"] for r in rows]),
        "p2s": describe([r["p2s"] for r in rows]),
    }
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result = {
        "timestamp": stamp,
        "model": args.model,
        "config": args.config,
        "num_samples": cfg["num_samples"],
        "meshes": len(rows),
        "inference": inference,
        "summary": summary,
        "per_mesh": rows,
    }

    # console summary
    print(f"model      : {args.model}")
    print(f"config     : {args.config}")
    print(f"num_samples: {cfg['num_samples']}  meshes: {len(rows)}")
    print(f"inference  : {inference}")
    print("-" * 84)
    for name in ("score", "cd", "p2s"):
        s = summary[name]
        print(
            f"  {name:5}: mean={s['mean']:7.3f}  std={s['std']:6.3f}  "
            f"min={s['min']:7.3f}  med={s['median']:7.3f}  max={s['max']:7.3f}"
        )

    # persistent JSON, auto-named by config + setting (never overwrites across settings)
    cfg_name = os.path.splitext(os.path.basename(args.config))[0]
    out = f"eval_{cfg_name}_{stamp}.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print("-" * 84)
    print(f"written {out}  ({len(rows)} meshes)")


if __name__ == "__main__":
    main()
