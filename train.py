#!/usr/bin/env python
"""Train the Flow2Surf conditional-flow point-cloud denoiser.

Usage:
    python train.py --config configs/default.yaml
    python train.py --config configs/default.yaml --resume checkpoints/epoch_010.pkl
"""

import argparse
import logging
import math
import os
import time

import jittor as jt
import numpy as np
import yaml
from jittor import nn

from flow2surf.dataset import build_datasets, load_dataset_spec
from flow2surf.evaluation import mini_val_score
from flow2surf.flow import flow_velocity, reconstruct_clean
from flow2surf.models import build_model
from flow2surf.runtime import configure_logging, git_revision, run_stamp, seed_runtime
from flow2surf.training.checkpoint import CheckpointManager
from flow2surf.training.losses import geometric_loss, velocity_loss
from flow2surf.training.optimization import ModelEMA, clip_grad_norm

# -- Logging --------------------------------------------------------------------

log = logging.getLogger("flow2surf.train")


def setup_run_paths(cfg, stamp):
    checkpoint_dir = cfg["checkpoint_dir"].rstrip(os.sep)
    cfg["checkpoint_dir"] = f"{checkpoint_dir}_{stamp}"


# -- Train / val ----------------------------------------------------------------


def _patch_loss(model, pc_noisy, x_alpha, pc_clean, alpha, cfg):
    """Return total, velocity, and geometry loss vectors for a patch batch."""
    pred_velocity = model(x_alpha.permute(0, 2, 1), alpha)
    vel = velocity_loss(pred_velocity, flow_velocity(pc_clean, pc_noisy))
    if cfg["cd_weight"] <= 0:
        return vel, vel, None

    alpha = alpha.reshape(-1, 1, 1)
    pred_clean = reconstruct_clean(x_alpha, pred_velocity, alpha)
    cd = geometric_loss(
        pred_clean,
        pc_clean,
        cfg["cd_type"],
        cfg["dcd_alpha"],
    )
    return vel + cfg["cd_weight"] * cd, vel, cd


def train_epoch(model, loader, optimizer, ema, cfg, epoch):
    """Optimize one epoch over full patch batches."""
    frozen = [name for name, param in model.named_parameters() if param.is_stop_grad()]
    if frozen:
        names = ", ".join(frozen[:3])
        raise RuntimeError(f"Model parameters are frozen before training: {names}")
    model.train()
    total_loss = total_n = 0
    step_loss = step_vel = step_cd = 0.0
    cd_weight = cfg["cd_weight"]
    grad_clip = cfg["grad_clip"]
    clipping = grad_clip is not None and grad_clip > 0
    max_grad_norm = jt.float32(0.0).stop_grad() if clipping else None
    t0 = step_t0 = time.time()
    total_steps = len(loader)

    for i, (
        pat_noisy,
        pat_x_alpha,
        pat_clean,
        pat_alpha,
    ) in enumerate(loader):
        B, P, M, _ = pat_noisy.shape
        n = B * P
        pc_noisy = jt.array(pat_noisy).reshape(n, M, 3)
        x_alpha = jt.array(pat_x_alpha).reshape(n, M, 3)
        pc_clean = jt.array(pat_clean).reshape(n, M, 3)
        alpha = jt.array(pat_alpha).reshape(B * P)
        loss, vel, cd = _patch_loss(
            model,
            pc_noisy,
            x_alpha,
            pc_clean,
            alpha,
            cfg,
        )
        loss_v = np.asarray(loss.numpy())
        vel_v = np.asarray(vel.numpy())
        cd_v = np.asarray(cd.numpy()) if cd is not None else np.zeros_like(loss_v)
        batch_loss = float(loss_v.mean())
        batch_vel = float(vel_v.mean())
        batch_cd = float(cd_v.mean())
        optimizer.backward(loss.sum() / n)

        if clipping:
            grad_norm = clip_grad_norm(optimizer, grad_clip)
            max_grad_norm.update(jt.maximum(max_grad_norm, grad_norm))
        optimizer.step()
        if ema is not None:
            ema.update()

        total_loss += batch_loss * n
        total_n += n
        step_loss += batch_loss
        step_vel += batch_vel
        step_cd += batch_cd

        if (i + 1) % cfg["log_interval"] == 0:
            inv = 1.0 / cfg["log_interval"]
            avg_loss = step_loss * inv
            avg_vel = step_vel * inv
            avg_cd = step_cd * inv
            step_loss = step_vel = step_cd = 0.0
            elapsed = time.time() - t0
            eta_s = (
                (time.time() - step_t0) / cfg["log_interval"] * (total_steps - i - 1)
            )
            step_t0 = time.time()

            gn = f" | grad_norm_max {float(max_grad_norm):.4f}" if clipping else ""
            if clipping:
                max_grad_norm.update(0.0)
            if cd_weight > 0:
                log.info(
                    f"  [{epoch}] step {i+1:4d}/{total_steps} | "
                    f"loss {avg_loss:.6f} "
                    f"(vel {avg_vel:.6f} + cd {avg_cd:.6f}*{cd_weight}) | "
                    f"elapsed {elapsed:.0f}s | eta {eta_s:.0f}s{gn}"
                )
            else:
                log.info(
                    f"  [{epoch}] step {i+1:4d}/{total_steps} | "
                    f"loss {avg_loss:.6f} | avg {total_loss/total_n:.6f} | "
                    f"elapsed {elapsed:.0f}s | eta {eta_s:.0f}s{gn}"
                )

    return total_loss / max(total_n, 1)


@jt.no_grad()
def val_epoch(model, loader, cfg):
    """Return mesh-averaged validation loss components."""
    model.eval()
    metric_sums = np.zeros(3, dtype=np.float64)
    total_meshes = 0

    for pat_noisy, pat_x_alpha, pat_clean, pat_alpha in loader:
        B, P, M, _ = pat_noisy.shape
        n = B * P
        loss, vel, cd = _patch_loss(
            model,
            jt.array(pat_noisy).reshape(n, M, 3),
            jt.array(pat_x_alpha).reshape(n, M, 3),
            jt.array(pat_clean).reshape(n, M, 3),
            jt.array(pat_alpha).reshape(n),
            cfg,
        )
        loss_v = np.asarray(loss.numpy())
        metric_values = (
            loss_v,
            np.asarray(vel.numpy()),
            np.asarray(cd.numpy()) if cd is not None else np.zeros_like(loss_v),
        )
        metric_sums += [
            values.reshape(B, P).mean(axis=1).sum() for values in metric_values
        ]
        total_meshes += B

    if not total_meshes:
        raise ValueError("Validation dataset is empty")
    return tuple(float(value) for value in metric_sums / total_meshes)


# -- Main -----------------------------------------------------------------------


def get_lr(cfg, epoch):
    """Return the warmup and cosine-decay learning rate for an epoch."""
    lr_max = cfg["lr"]
    lr_min = cfg["lr_min"]
    warmup = cfg["warmup_epochs"]
    if warmup > 0 and epoch <= warmup:
        return lr_max * epoch / warmup
    progress = (epoch - warmup - 1) / max(cfg["epochs"] - warmup - 1, 1)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))


def fmt_eta(seconds):
    """Format a duration as compact hours and minutes."""
    h, m = divmod(int(seconds), 3600)
    m //= 60
    return f"{h}h{m:02d}m" if h else f"{m}m"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dataset", default="datasets/A.yaml")
    parser.add_argument(
        "--resume", default=None, help="Path to checkpoint .pkl to resume training from"
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    dataset = load_dataset_spec(args.dataset)

    stamp = run_stamp()
    setup_run_paths(cfg, stamp)
    log_file = os.path.join("logs", f"train_{stamp}.log")
    configure_logging(log_file)
    log.info("Log file: %s", log_file)

    seed_runtime(cfg["seed"])
    jt.flags.use_cuda = 1

    log.info("=" * 64)
    log.info("Flow2Surf - Point Cloud Denoising")
    log.info("=" * 64)
    repo = os.path.dirname(os.path.abspath(__file__))
    log.info(f"  {'git_commit':<24}: {git_revision(repo)}")
    log.info(f"  {'config':<24}: {args.config}")
    log.info(f"  {'dataset':<24}: {args.dataset} ({dataset.name})")
    for k, v in cfg.items():
        log.info(f"  {k:<24}: {v}")
    log.info("-" * 64)

    train_ds, val_ds = build_datasets(cfg, dataset)
    log.info(f"  Train meshes : {len(train_ds.paths)}")
    log.info(f"  Val meshes   : {len(val_ds.paths)}")
    log.info(f"  Steps/epoch  : {len(train_ds)}")

    mini_interval = int(cfg["mini_val_interval"])
    mini_size = int(cfg["mini_val_size"])
    checkpoint_interval = int(cfg["checkpoint_interval"])
    if checkpoint_interval < 1:
        raise ValueError(f"checkpoint_interval must be >= 1, got {checkpoint_interval}")
    mini_paths = val_ds.paths[: min(mini_size, len(val_ds.paths))]
    # Mini inputs, noisy baselines, and FPS centers are fixed across epochs.
    # Cache them while recomputing every model-dependent output and score.
    mini_baseline_cache = {}
    if mini_interval > 0:
        log.info(
            f"  Mini score   : {len(mini_paths)} fixed val meshes every "
            f"{mini_interval} epochs"
        )

    model = build_model(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"  Parameters   : {n_params/1e6:.3f}M")
    log.info("-" * 64)

    optimizer = nn.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    ema_decay = float(cfg["ema_decay"])
    if not 0.0 <= ema_decay < 1.0:
        raise ValueError(f"ema_decay must be in [0, 1), got {ema_decay}")
    ema = ModelEMA(model, ema_decay) if ema_decay > 0 else None
    eval_model = ema.model if ema else model

    ckpt = CheckpointManager(cfg["checkpoint_dir"])
    os.makedirs(ckpt.dir, exist_ok=True)

    start_epoch = 1
    best_loss = float("inf")
    best_epoch = 0
    best_mini_score = -float("inf")
    best_mini_epoch = 0
    history = []
    if args.resume:
        log.info(f"Resume: {args.resume}")
        (
            start_epoch,
            best_loss,
            best_epoch,
            best_mini_score,
            best_mini_epoch,
            history,
        ) = ckpt.load(model, optimizer, ema, args.resume)
        log.info("-" * 64)

    if best_mini_epoch:
        log.info(f"  Resume mini best: {best_mini_score:.2f}@ep{best_mini_epoch}")

    epoch_times = []

    for epoch in range(start_epoch, cfg["epochs"] + 1):
        lr = get_lr(cfg, epoch)
        for g in optimizer.param_groups:
            g["lr"] = lr

        log.info(
            f"-- Epoch {epoch}/{cfg['epochs']}  lr={lr:.2e} --------------------------"
        )
        t0 = time.time()

        train_loss = train_epoch(
            model,
            train_ds,
            optimizer,
            ema,
            cfg,
            epoch,
        )
        val_loss, val_vel, val_cd = val_epoch(eval_model, val_ds, cfg)
        mini = None
        mini_improved = False
        if mini_interval > 0 and epoch % mini_interval == 0:
            mini = mini_val_score(
                eval_model,
                mini_paths,
                cfg,
                num_chunks=cfg["mini_val_chunks"],
                baseline_cache=mini_baseline_cache,
            )
            if mini["score"] > best_mini_score:
                best_mini_score = mini["score"]
                best_mini_epoch = epoch
                mini_improved = True

        is_best = val_loss < best_loss
        if is_best:
            best_loss = val_loss
            best_epoch = epoch
            ckpt.save_best(eval_model, epoch, val_loss)

        if mini_improved:
            ckpt.save_best_mini(eval_model, epoch, mini["score"])

        elapsed = time.time() - t0
        epoch_times.append(elapsed)

        eta = (sum(epoch_times[-10:]) / len(epoch_times[-10:])) * (
            cfg["epochs"] - epoch
        )

        hist_item = {
            "epoch": epoch,
            "lr": float(lr),
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "val_vel": round(val_vel, 6),
            "val_cd": round(val_cd, 6),
        }
        if mini is not None:
            hist_item.update(
                {
                    "mini_val_score": round(mini["score"], 4),
                    "mini_val_cd_score": round(mini["cd_score"], 4),
                    "mini_val_p2s_score": round(mini["p2s_score"], 4),
                    "mini_val_meshes": mini["num_meshes"],
                }
            )
        history.append(hist_item)

        if epoch % checkpoint_interval == 0 or epoch == cfg["epochs"]:
            ckpt.save_periodic(
                model,
                optimizer,
                ema,
                epoch,
                best_loss,
                best_epoch,
                best_mini_score,
                best_mini_epoch,
                history,
            )

        cd_w = cfg["cd_weight"]
        log.info(
            f"  train | loss {train_loss:.6f} | epoch {elapsed:.0f}s | "
            f"eta {fmt_eta(eta)}"
        )
        if cd_w > 0:
            log.info(
                f"  val   | loss {val_loss:.6f} "
                f"(vel {val_vel:.6f}, cd {val_cd:.6f}*{cd_w}) | "
                f"best {best_loss:.6f}@ep{best_epoch}"
                + (" | NEW BEST" if is_best else "")
            )
        else:
            log.info(
                f"  val   | loss {val_loss:.6f} (vel {val_vel:.6f}) | "
                f"best {best_loss:.6f}@ep{best_epoch}"
                + (" | NEW BEST" if is_best else "")
            )
        if mini is not None:
            log.info(
                f"  mini  | score {mini['score']:.2f} "
                f"(cd {mini['cd_score']:.2f}, p2s {mini['p2s_score']:.2f}) | "
                f"best {best_mini_score:.2f}@ep{best_mini_epoch}"
                + (" | NEW MINI BEST" if mini_improved else "")
            )

    log.info("=" * 64)
    log.info("Training complete")
    log.info(f"  Best val loss        : {best_loss:.6f}@ep{best_epoch}")
    if best_mini_epoch:
        log.info(f"  Best mini score      : {best_mini_score:.2f}@ep{best_mini_epoch}")
    best_paths = ckpt.model_paths("best_ep*.pkl")
    best_mini_paths = ckpt.model_paths("best_mini_ep*.pkl")
    if best_paths:
        log.info(f"  Best val checkpoint  : {best_paths[-1]}")
    if best_mini_paths:
        log.info(f"  Best mini checkpoint : {best_mini_paths[-1]}")


if __name__ == "__main__":
    main()
