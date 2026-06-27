#!/usr/bin/env python
"""Train Flow2Surf denoiser.

Usage:
    python train.py --config configs/default.yaml
    python train.py --config configs/default.yaml --resume checkpoints/epoch_010.pkl
"""

import os, json, time, argparse, logging, glob, math, pickle, random, subprocess
from datetime import datetime

import yaml
import numpy as np
import jittor as jt
from jittor import nn
import point_cloud_utils as pcu
from scipy.spatial import cKDTree

from src.dataset import (
    alpha_schedule,
    build_datasets,
    load_mesh,
    sample_surface,
    add_noise,
)
from src.model import Flow2SurfNet
from predict import make_rng, patch_denoise

# -- Logger ---------------------------------------------------------------------
# Defined at module level so all helpers can safely call log.info() without
# worrying about call order relative to main().

log = logging.getLogger("flow2surf")


def alpha_from_step(t_idx, cfg):
    if hasattr(t_idx, "numpy"):
        t_idx = t_idx.numpy()
    idx = np.asarray(t_idx).reshape(-1).astype(np.int64)
    alphas = alpha_schedule(cfg.get("num_steps", 30))
    return jt.array(alphas[idx]).reshape(-1)


def git_commit():
    """Short git commit (with -dirty flag) for run provenance; 'unknown' if unavailable.

    Logged at startup so every run records the exact code state it ran on - makes
    cross-run comparisons (e.g. metric/behaviour changes) traceable to a commit.
    """
    repo = os.path.dirname(os.path.abspath(__file__))
    try:
        commit = (
            subprocess.check_output(
                ["git", "-C", repo, "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
        dirty = subprocess.run(["git", "-C", repo, "diff", "--quiet"]).returncode != 0
        return f"{commit}{'-dirty' if dirty else ''}"
    except Exception:
        return "unknown"


def make_run_stamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def setup_run_paths(cfg, stamp):
    checkpoint_dir = cfg["checkpoint_dir"].rstrip(os.sep)
    cfg["checkpoint_dir"] = f"{checkpoint_dir}_{stamp}"

    root, ext = os.path.splitext(cfg["history_path"])
    cfg["history_path"] = f"{root}_{stamp}{ext or '.json'}"


def setup_logger(log_dir="logs", stamp=None):
    os.makedirs(log_dir, exist_ok=True)
    stamp = stamp or make_run_stamp()
    log_file = os.path.join(log_dir, f"train_{stamp}.log")

    log.setLevel(logging.INFO)
    log.propagate = False
    log.handlers.clear()
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S"
    )
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    log.addHandler(ch)
    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)
    log.addHandler(fh)
    log.info(f"Log: {log_file}")


# -- Loss -----------------------------------------------------------------------


def pairwise_sq(a, b):
    """Squared pairwise distances (B, M, N) via matmul: |a|^2 + |b|^2 - 2 a.b.
    Avoids the (B, M, N, 3) difference tensor."""
    aa = (a * a).sum(-1)  # (B, M)
    bb = (b * b).sum(-1)  # (B, N)
    inner = jt.nn.bmm(a, b.transpose(0, 2, 1))  # (B, M, N)
    return jt.maximum(aa.unsqueeze(2) + bb.unsqueeze(1) - 2 * inner, 0.0)


def patch_chamfer(pred, gt):
    """
    Bidirectional Chamfer Distance between two batches of patches.
    pred, gt : (B, M, 3)

    pred->gt term: each predicted point must be close to some GT surface point
                  (approximates P2S when GT is a dense surface sample).
    gt->pred term: every GT point must have a nearby prediction (no holes).
    """
    sq = pairwise_sq(pred, gt)  # (B, M, N)
    return sq.min(dim=2).mean() + sq.min(dim=1).mean()


def density_aware_chamfer(pred, gt, alpha):
    """Chamfer that downweights many-to-one matches by 1/n (n = sources sharing a
    nearest target), so clustering is penalized. Gaussian kernel on squared
    distance; returns [0, 1], lower = better-distributed."""
    d2 = pairwise_sq(pred, gt)  # (B, M, N)
    B, M, N = d2.shape
    idx_p, d_p = jt.argmin(d2, dim=2)  # nearest gt per pred, (B, M)
    idx_g, d_g = jt.argmin(d2, dim=1)  # nearest pred per gt, (B, N)
    n_g = jt.zeros((B, N)).scatter(
        1, idx_p, jt.ones((B, M)), reduce="add"
    )  # pred per gt
    n_p = jt.zeros((B, M)).scatter(
        1, idx_g, jt.ones((B, N)), reduce="add"
    )  # gt per pred
    w_p = 1.0 / jt.maximum(jt.gather(n_g, 1, idx_p), 1.0)
    w_g = 1.0 / jt.maximum(jt.gather(n_p, 1, idx_g), 1.0)
    term_p = (1.0 - w_p * jt.exp(-alpha * d_p)).mean()
    term_g = (1.0 - w_g * jt.exp(-alpha * d_g)).mean()
    return 0.5 * (term_p + term_g)


def chamfer_term(pred, gt, cfg):
    """Distribution loss: density-aware (cd_type=dcd) or plain Chamfer (default)."""
    if cfg.get("cd_type", "chamfer") == "dcd":
        return density_aware_chamfer(pred, gt, cfg.get("dcd_alpha", 8000.0))
    return patch_chamfer(pred, gt)


# -- Full-cloud mini score ------------------------------------------------------


def normalize_with_params(pts):
    lo, hi = pts.min(0), pts.max(0)
    center = (lo + hi) / 2
    shifted = pts - center
    scale = float(np.sqrt((shifted**2).sum(1)).max()) + 1e-8
    return shifted / scale, center, scale


def chamfer_distance(pred, gt):
    d1, _ = cKDTree(gt).query(pred)
    d2, _ = cKDTree(pred).query(gt)
    return float((d1**2).mean() + (d2**2).mean())


def point_to_surface(pts, verts, faces):
    dists, _, _ = pcu.closest_points_on_mesh(pts, verts, faces)
    return float((dists**2).mean())


def improvement_score(pred_val, noisy_val):
    return float(np.clip(100.0 * (1.0 - pred_val / (noisy_val + 1e-12)), 0.0, 100.0))


@jt.no_grad()
def mini_val_score(model, paths, cfg):
    model.eval()
    seed = cfg.get("inference_seed", cfg.get("seed", 42))
    # Score each mesh under one or more noise types. Defaults to the training
    # noise_type (single, back-compatible); set mini_noise_types: [laplace, gaussian]
    # to also see / select on gaussian robustness.
    noise_types = cfg.get("mini_noise_types") or [cfg["noise_type"]]
    per_type = {nt: {"score": [], "cd": [], "p2s": []} for nt in noise_types}

    for i, path in enumerate(paths):
        sample_seed = int(cfg["seed"]) + i

        verts, faces = load_mesh(path)
        raw_pts, _ = sample_surface(verts, faces, cfg["num_samples"], seed=sample_seed)
        clean, center, scale = normalize_with_params(raw_pts)
        verts_norm = (np.array(verts, dtype=np.float32) - center) / scale

        synset = os.path.basename(
            os.path.dirname(os.path.dirname(os.path.dirname(path)))
        )
        model_id = os.path.basename(os.path.dirname(os.path.dirname(path)))

        for nt in noise_types:
            rng = make_rng(seed, "mini_noise", nt, synset, model_id)
            noisy = add_noise(
                clean, cfg["noise_std_min"], cfg["noise_std_max"], nt, rng=rng
            )
            denoised = patch_denoise(
                model,
                noisy,
                patch_size=cfg["patch_size"],
                seed_k=cfg["seed_k"],
                steps=cfg["num_steps"],
                rng=make_rng(seed, "mini_val", nt, synset, model_id),
                recompute_neighbors=cfg.get("recompute_neighbors", False),
                patch_frame=cfg.get("patch_frame", "local"),
                patch_aggregation=cfg.get("patch_aggregation", "residual"),
                patch_agg_beta=cfg.get("patch_agg_beta", 4),
            )

            cd_noisy = chamfer_distance(noisy, clean)
            cd_pred = chamfer_distance(denoised, clean)
            p2s_noisy = point_to_surface(noisy, verts_norm, faces)
            p2s_pred = point_to_surface(denoised, verts_norm, faces)

            cd_s = improvement_score(cd_pred, cd_noisy)
            p2s_s = improvement_score(p2s_pred, p2s_noisy)
            per_type[nt]["score"].append(0.5 * cd_s + 0.5 * p2s_s)
            per_type[nt]["cd"].append(cd_s)
            per_type[nt]["p2s"].append(p2s_s)

    mean = lambda xs: float(np.mean(xs)) if xs else 0.0
    flat = lambda key: [v for nt in noise_types for v in per_type[nt][key]]
    by_noise = {
        nt: {
            "score": round(mean(per_type[nt]["score"]), 4),
            "cd_score": round(mean(per_type[nt]["cd"]), 4),
            "p2s_score": round(mean(per_type[nt]["p2s"]), 4),
            "n": len(per_type[nt]["score"]),
        }
        for nt in noise_types
    }
    return {
        "score": mean(flat("score")),
        "cd_score": mean(flat("cd")),
        "p2s_score": mean(flat("p2s")),
        "n": len(flat("score")),
        "by_noise": by_noise,
    }


def best_mini_from_history(history):
    best_score = -float("inf")
    best_epoch = 0
    for item in history:
        score = item.get("mini_val_score")
        if score is not None and score > best_score:
            best_score = score
            best_epoch = item["epoch"]
    return best_score, best_epoch


# -- Checkpoint helpers ---------------------------------------------------------


def periodic_ckpt_path(cfg, epoch):
    return os.path.join(cfg["checkpoint_dir"], f"epoch_{epoch:03d}.pkl")


def best_ckpt_path(cfg, epoch, val_loss):
    """Encodes epoch and val loss in the filename for easy identification."""
    return os.path.join(
        cfg["checkpoint_dir"], f"best_ep{epoch:03d}_val{val_loss:.6f}.pkl"
    )


def best_mini_ckpt_path(cfg, epoch, mini_score):
    """Encodes epoch and mini score in the filename for easy identification."""
    return os.path.join(
        cfg["checkpoint_dir"], f"best_mini_ep{epoch:03d}_score{mini_score:.4f}.pkl"
    )


def best_info_path(cfg):
    return os.path.join(cfg["checkpoint_dir"], "best_info.json")


def best_mini_info_path(cfg):
    return os.path.join(cfg["checkpoint_dir"], "best_mini_info.json")


def optimizer_state_path(model_path):
    root, ext = os.path.splitext(model_path)
    return f"{root}_optim{ext}"


def checkpoint_state_path(model_path):
    return model_path.replace(".pkl", "_state.json")


def write_json_atomic(path, data):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_path, path)


def remove_if_exists(path):
    try:
        os.remove(path)
    except FileNotFoundError:
        return


def model_checkpoint_paths(cfg, pattern):
    paths = glob.glob(os.path.join(cfg["checkpoint_dir"], pattern))
    return sorted(p for p in paths if not p.endswith("_optim.pkl"))


def save_optimizer_state(optimizer, model_path):
    if not hasattr(optimizer, "state_dict"):
        log.warning("  Optimizer does not expose state_dict(); resume is weights-only")
        return
    try:
        os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
        with open(optimizer_state_path(model_path), "wb") as f:
            pickle.dump(optimizer.state_dict(), f)
    except Exception as e:
        log.warning("  Failed to save optimizer state (%s); resume is weights-only", e)


def load_optimizer_state(optimizer, model_path):
    path = optimizer_state_path(model_path)
    if not os.path.exists(path):
        log.warning("  Optimizer state missing - resume is weights-only")
        return
    if not hasattr(optimizer, "load_state_dict"):
        log.warning(
            "  Optimizer does not expose load_state_dict(); resume is weights-only"
        )
        return
    try:
        with open(path, "rb") as f:
            optimizer.load_state_dict(pickle.load(f))
        log.info(f"  Loaded optimizer state {path}")
    except Exception as e:
        log.warning("  Failed to load optimizer state (%s); resume is weights-only", e)


def save_periodic_checkpoint(model, optimizer, cfg, epoch, best_loss, history):
    path = periodic_ckpt_path(cfg, epoch)
    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)
    model.save(path)
    save_optimizer_state(optimizer, path)
    state = {"epoch": epoch, "best_loss": best_loss, "history": history}
    write_json_atomic(checkpoint_state_path(path), state)
    log.info(f"  Periodic checkpoint saved -> {path}")


def save_best_checkpoint(model, optimizer, cfg, epoch, val_loss, history):
    """Save a new best checkpoint, then clean older best checkpoint files."""
    path = best_ckpt_path(cfg, epoch, val_loss)
    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)
    model.save(path)
    save_optimizer_state(optimizer, path)

    state = {"epoch": epoch, "best_loss": val_loss, "history": history}
    write_json_atomic(checkpoint_state_path(path), state)

    info = {"path": path, "epoch": epoch, "val_loss": val_loss}
    write_json_atomic(best_info_path(cfg), info)

    for old in model_checkpoint_paths(cfg, "best_ep*.pkl"):
        if old == path:
            continue
        remove_if_exists(old)
        remove_if_exists(checkpoint_state_path(old))
        remove_if_exists(optimizer_state_path(old))

    log.info(f"  Best checkpoint saved -> {path}")


def save_best_mini_checkpoint(model, optimizer, cfg, epoch, mini, history):
    """Save a new mini-score best checkpoint, then clean older mini best files."""
    path = best_mini_ckpt_path(cfg, epoch, mini["score"])
    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)
    model.save(path)
    save_optimizer_state(optimizer, path)

    state = {
        "epoch": epoch,
        "best_mini_score": mini["score"],
        "best_mini_cd_score": mini["cd_score"],
        "best_mini_p2s_score": mini["p2s_score"],
        "history": history,
    }
    write_json_atomic(checkpoint_state_path(path), state)

    info = {
        "path": path,
        "epoch": epoch,
        "mini_val_score": mini["score"],
        "mini_val_cd_score": mini["cd_score"],
        "mini_val_p2s_score": mini["p2s_score"],
    }
    write_json_atomic(best_mini_info_path(cfg), info)

    for old in model_checkpoint_paths(cfg, "best_mini_ep*.pkl"):
        if old == path:
            continue
        remove_if_exists(old)
        remove_if_exists(checkpoint_state_path(old))
        remove_if_exists(optimizer_state_path(old))

    log.info(f"  Best mini checkpoint saved -> {path}")


def load_checkpoint(model, optimizer, path):
    assert os.path.exists(path), f"Checkpoint not found: {path}"
    model.load(path)
    load_optimizer_state(optimizer, path)
    state_path = checkpoint_state_path(path)
    if not os.path.exists(state_path):
        log.warning("  State file missing - epoch counter resets to 1")
        return 1, float("inf"), []
    with open(state_path) as f:
        s = json.load(f)
    history = s.get("history", [])
    # Derive best val loss from history (the source of truth) so resume works for any
    # checkpoint type, incl. best_mini_* whose state has no "best_loss" key.
    best_loss = min(
        (h["val_loss"] for h in history if "val_loss" in h), default=float("inf")
    )
    log.info(f"  Loaded epoch {s['epoch']}, best_val_loss={best_loss:.6f}")
    return s["epoch"] + 1, best_loss, history


# -- Train / val ----------------------------------------------------------------


def residual_loss(pred_res, target, normals, w_perp, w_tang):
    """Residual reconstruction loss. With w_perp == w_tang this is plain MSE.
    Otherwise the per-point error is split via the clean normal into perpendicular
    (off-surface) and tangential (along-surface) parts, weighted separately -
    penalize leaving the surface, forgive sliding along it (P2S-aligned)."""
    err = pred_res - target
    if w_perp == w_tang:
        return w_perp * (err * err).sum(dim=-1).mean()
    perp = (err * normals).sum(dim=-1)
    sq = (err * err).sum(dim=-1)
    tang_sq = jt.maximum(sq - perp * perp, 0.0)
    return (w_perp * perp * perp + w_tang * tang_sq).mean()


def train_epoch(model, loader, optimizer, cfg, epoch):
    model.train()
    total_loss = total_n = 0
    step_loss = step_res = step_cd = 0.0
    cd_weight = cfg.get("cd_weight", 0.2)
    w_perp = cfg.get("loss_perp_weight", 1.0)
    w_tang = cfg.get("loss_tang_weight", 1.0)
    use_normals = w_perp != w_tang
    grad_clip = cfg.get("grad_clip", 1.0)
    max_grad_norm = 0.0
    t0 = step_t0 = time.time()
    total_steps = len(loader)

    for i, (pat_noisy, pat_mix, pat_clean, pat_t, full_noisy, pat_normals) in enumerate(
        loader
    ):
        B, P, M, _ = pat_noisy.shape

        pc_noisy = jt.array(pat_noisy).reshape(B * P, M, 3)
        pc_mix = jt.array(pat_mix).reshape(B * P, M, 3)
        pc_clean = jt.array(pat_clean).reshape(B * P, M, 3)
        t_idx = jt.array(pat_t).reshape(B * P)
        normals = jt.array(pat_normals).reshape(B * P, M, 3) if use_normals else None

        pred_res = model(pc_mix.permute(0, 2, 1), t_idx)
        target = pc_noisy - pc_clean  # Residual: noisy - clean

        res = residual_loss(pred_res, target, normals, w_perp, w_tang)

        if cd_weight > 0:
            alpha = alpha_from_step(pat_t, cfg)
            pred_clean = pc_mix - alpha.reshape(-1, 1, 1) * pred_res
            cd = chamfer_term(pred_clean, pc_clean, cfg)
            loss = res + cd_weight * cd
            cd_v = cd.item()
        else:
            loss = res
            cd_v = 0.0

        if grad_clip and grad_clip > 0:
            optimizer.pre_step(loss)
            sq = [
                (g * g).sum()
                for pg in optimizer.param_groups
                for p, g in zip(pg["params"], pg["grads"])
                if not p.is_stop_grad()
            ]
            total_norm = jt.sqrt(sum(sq)) if sq else jt.float32(0.0)
            max_grad_norm = max(max_grad_norm, float(total_norm))
            clip_coef = jt.minimum(grad_clip / (total_norm + 1e-6), 1.0)
            for pg in optimizer.param_groups:
                for p, g in zip(pg["params"], pg["grads"]):
                    if not p.is_stop_grad():
                        g.update(g * clip_coef)
            optimizer.step()
        else:
            optimizer.step(loss)

        n = B * P
        loss_v = loss.item()
        total_loss += loss_v * n
        total_n += n
        step_loss += loss_v
        step_res += res.item()
        step_cd += cd_v

        if (i + 1) % cfg["log_interval"] == 0:
            inv = 1.0 / cfg["log_interval"]
            avg_loss = step_loss * inv
            avg_res = step_res * inv
            avg_cd = step_cd * inv
            step_loss = step_res = step_cd = 0.0
            elapsed = time.time() - t0
            eta_s = (
                (time.time() - step_t0) / cfg["log_interval"] * (total_steps - i - 1)
            )
            step_t0 = time.time()

            gn = (
                f" | max_gnorm {max_grad_norm:.2f}"
                if grad_clip and grad_clip > 0
                else ""
            )
            max_grad_norm = 0.0
            if cd_weight > 0:
                log.info(
                    f"  [{epoch}] step {i+1:4d}/{total_steps} | "
                    f"loss {avg_loss:.6f} "
                    f"(res {avg_res:.6f} + cd {avg_cd:.6f}*{cd_weight}) | "
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
    model.eval()
    total_loss = total_res = total_cd = total_n = 0
    cd_weight = cfg.get("cd_weight", 0.2)
    w_perp = cfg.get("loss_perp_weight", 1.0)
    w_tang = cfg.get("loss_tang_weight", 1.0)
    use_normals = w_perp != w_tang

    for pat_noisy, pat_mix, pat_clean, pat_t, full_noisy, pat_normals in loader:
        B, P, M, _ = pat_noisy.shape
        pc_noisy = jt.array(pat_noisy).reshape(B * P, M, 3)
        pc_mix = jt.array(pat_mix).reshape(B * P, M, 3)
        pc_clean = jt.array(pat_clean).reshape(B * P, M, 3)
        t_idx = jt.array(pat_t).reshape(B * P)
        normals = jt.array(pat_normals).reshape(B * P, M, 3) if use_normals else None

        pred_res = model(pc_mix.permute(0, 2, 1), t_idx)
        target = pc_noisy - pc_clean
        res = residual_loss(pred_res, target, normals, w_perp, w_tang)

        if cd_weight > 0:
            alpha = alpha_from_step(pat_t, cfg)
            pred_clean = pc_mix - alpha.reshape(-1, 1, 1) * pred_res
            cd = chamfer_term(pred_clean, pc_clean, cfg)
            loss = res + cd_weight * cd
            total_cd += cd.item() * (B * P)
        else:
            loss = res

        n = B * P
        total_loss += loss.item() * n
        total_res += res.item() * n
        total_n += n

    return (
        total_loss / max(total_n, 1),
        total_res / max(total_n, 1),
        total_cd / max(total_n, 1),
    )


# -- Main -----------------------------------------------------------------------


def get_lr(cfg, epoch):
    lr_max = cfg["lr"]
    lr_min = cfg.get("lr_min", 1e-5)
    warmup = cfg.get("warmup_epochs", 0)
    if warmup > 0 and epoch <= warmup:
        return lr_max * epoch / warmup
    t = (epoch - warmup - 1) / max(cfg["epochs"] - warmup - 1, 1)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * t))


def fmt_eta(seconds):
    h, m = divmod(int(seconds), 3600)
    m //= 60
    return f"{h}h{m:02d}m" if h else f"{m}m"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--resume", default=None, help="Path to checkpoint .pkl to resume training from"
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    stamp = make_run_stamp()
    setup_run_paths(cfg, stamp)
    setup_logger(stamp=stamp)

    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    jt.set_global_seed(cfg["seed"])
    jt.flags.use_cuda = 1

    log.info("=" * 64)
    log.info("Flow2Surf - Point Cloud Denoising")
    log.info("=" * 64)
    log.info(f"  {'git_commit':<24}: {git_commit()}")
    log.info(f"  {'config':<24}: {args.config}")
    for k, v in cfg.items():
        log.info(f"  {k:<24}: {v}")
    log.info("-" * 64)

    train_ds, val_ds = build_datasets(cfg)
    log.info(f"  Train meshes : {len(train_ds.paths)}")
    log.info(f"  Val meshes   : {len(val_ds.paths)}")
    log.info(f"  Steps/epoch  : {len(train_ds)}")

    mini_interval = int(cfg.get("mini_val_interval", 2))
    mini_size = int(cfg.get("mini_val_size", 20))
    mini_paths = val_ds.paths[: min(mini_size, len(val_ds.paths))]
    if mini_interval > 0:
        log.info(
            f"  Mini score   : every {mini_interval} epochs on {len(mini_paths)} fixed val meshes"
        )
        log.info("  Mini score   : saves its own best checkpoint")

    model = Flow2SurfNet(
        k=cfg["k_neighbors"],
        feat_dim=cfg["feat_dim"],
        num_attn=cfg["num_attn"],
        emb_dims=cfg.get("emb_dims", [64, 64]),
        emb_concat=cfg.get("emb_concat", True),
        emb_fusion=cfg.get("emb_fusion", "concat"),
        decoder_dims=cfg.get("decoder_dims", [128, 64]),
        attn_type=cfg.get("attn_type", "offset"),
        attn_heads=cfg.get("attn_heads", 1),
        attn_softmax=cfg.get("attn_softmax", "pct"),
        ffn_ratio=cfg.get("ffn_ratio", 4),
        ffn_type=cfg.get("ffn_type", "swiglu"),
    )
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"  Model params : {n_params/1e6:.3f}M")
    log.info("-" * 64)

    opt_name = cfg["optimizer"].lower()
    if opt_name == "adamw":
        optimizer = nn.AdamW(
            model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
        )
    elif opt_name == "adam":
        optimizer = nn.Adam(
            model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
        )
    elif opt_name == "sgd":
        optimizer = nn.SGD(
            model.parameters(),
            lr=cfg["lr"],
            momentum=0.9,
            weight_decay=cfg["weight_decay"],
        )
    else:
        raise ValueError(f"Unknown optimizer: {cfg['optimizer']}")

    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)

    start_epoch, best_loss, history = 1, float("inf"), []
    if args.resume:
        log.info(f"Resuming from: {args.resume}")
        start_epoch, best_loss, history = load_checkpoint(model, optimizer, args.resume)
        log.info("-" * 64)

    best_mini_score, best_mini_epoch = best_mini_from_history(history)
    if best_mini_epoch:
        log.info(
            f"  Resume mini best : score={best_mini_score:.2f} at epoch {best_mini_epoch}"
        )

    epoch_times = []

    for epoch in range(start_epoch, cfg["epochs"] + 1):
        lr = get_lr(cfg, epoch)
        for g in optimizer.param_groups:
            g["lr"] = lr

        log.info(
            f"-- Epoch {epoch}/{cfg['epochs']}  lr={lr:.2e} --------------------------"
        )
        t0 = time.time()

        train_loss = train_epoch(model, train_ds, optimizer, cfg, epoch)
        val_loss, val_res, val_cd = val_epoch(model, val_ds, cfg)
        mini = None
        mini_improved = False
        if mini_interval > 0 and epoch % mini_interval == 0:
            score_t0 = time.time()
            mini = mini_val_score(model, mini_paths, cfg)
            mini["elapsed"] = round(time.time() - score_t0, 1)
            min_delta = float(cfg.get("mini_val_min_delta", 0.0))
            if mini["score"] > best_mini_score + min_delta:
                best_mini_score = mini["score"]
                best_mini_epoch = epoch
                mini_improved = True

        elapsed = time.time() - t0
        epoch_times.append(elapsed)

        eta = (sum(epoch_times[-10:]) / len(epoch_times[-10:])) * (
            cfg["epochs"] - epoch
        )

        hist_item = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "val_res": round(val_res, 6),
            "val_cd": round(val_cd, 6),
            "elapsed": round(elapsed, 1),
        }
        if mini is not None:
            hist_item.update(
                {
                    "mini_val_score": round(mini["score"], 4),
                    "mini_val_cd_score": round(mini["cd_score"], 4),
                    "mini_val_p2s_score": round(mini["p2s_score"], 4),
                    "mini_val_by_noise": mini.get("by_noise", {}),
                    "mini_val_elapsed": mini["elapsed"],
                }
            )
        history.append(hist_item)
        write_json_atomic(cfg["history_path"], history)

        if epoch % cfg["checkpoint_interval"] == 0:
            save_periodic_checkpoint(model, optimizer, cfg, epoch, best_loss, history)

        is_best = val_loss < best_loss
        if is_best:
            best_loss = val_loss
            save_best_checkpoint(model, optimizer, cfg, epoch, val_loss, history)

        if mini_improved:
            save_best_mini_checkpoint(model, optimizer, cfg, epoch, mini, history)

        cd_w = cfg.get("cd_weight", 0.2)
        val_detail = (
            f"res={val_res:.6f} cd={val_cd:.6f}*{cd_w}"
            if cd_w > 0
            else f"res={val_res:.6f}"
        )
        log.info(
            f"  train_loss={train_loss:.6f} | "
            f"val_loss={val_loss:.6f} ({val_detail}) | "
            f"best={best_loss:.6f} | "
            f"{elapsed:.0f}s/epoch | eta={fmt_eta(eta)}"
            + (" <- NEW BEST" if is_best else "")
        )
        if mini is not None:
            stale_epochs = epoch - best_mini_epoch
            log.info(
                f"  mini_score={mini['score']:.2f} "
                f"(cd={mini['cd_score']:.2f} p2s={mini['p2s_score']:.2f}, "
                f"n={mini['n']}, {mini['elapsed']:.0f}s) | "
                f"best_mini={best_mini_score:.2f}@ep{best_mini_epoch}"
                + (
                    " <- NEW MINI BEST"
                    if mini_improved
                    else f" | stale={stale_epochs}ep"
                )
            )
            by_noise = mini.get("by_noise", {})
            if len(by_noise) > 1:
                split = "  ".join(
                    f"{nt}={d['score']:.2f}(cd={d['cd_score']:.2f} p2s={d['p2s_score']:.2f})"
                    for nt, d in by_noise.items()
                )
                log.info(f"  mini by noise: {split}")
        log.info("")

    log.info("=" * 64)
    log.info("Training complete")
    log.info(f"  Best val loss : {best_loss:.6f}")
    if best_mini_epoch:
        log.info(
            f"  Best mini score : {best_mini_score:.2f} at epoch {best_mini_epoch}"
        )
    log.info(f"  Best val checkpoint  : {best_info_path(cfg)}")
    if best_mini_epoch:
        log.info(f"  Best mini checkpoint : {best_mini_info_path(cfg)}")


if __name__ == "__main__":
    main()
