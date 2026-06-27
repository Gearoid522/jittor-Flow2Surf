import os, argparse, logging, hashlib, random
import yaml
import numpy as np
import jittor as jt
from scipy.spatial import cKDTree

from src.dataset import alpha_schedule, normalize_patch_frame
from src.model import Flow2SurfNet

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("predict")

PATCH_AGGREGATION_MODES = {"position", "residual"}


def normalize_patch_aggregation(mode):
    if mode not in PATCH_AGGREGATION_MODES:
        valid = ", ".join(sorted(PATCH_AGGREGATION_MODES))
        raise ValueError(f"Unknown patch_aggregation: {mode}. Valid values: {valid}")
    return mode


# -- FPS ------------------------------------------------------------------------


def make_rng(seed, *parts):
    key = "::".join([str(seed), *map(str, parts)]).encode("utf-8")
    digest = hashlib.blake2b(key, digest_size=8).digest()
    return np.random.default_rng(int.from_bytes(digest, "little") % (2**32))


def fps(pts, n, rng):
    """Farthest-point sampling. pts: (N, 3) -> indices (n,)."""
    N = pts.shape[0]
    selected = [int(rng.integers(N))]
    dists = np.full(N, np.inf)
    for _ in range(n - 1):
        d = ((pts - pts[selected[-1]]) ** 2).sum(1)
        dists = np.minimum(dists, d)
        selected.append(int(dists.argmax()))
    return np.array(selected)


# -- Patch denoising ------------------------------------------------------------


def patch_denoise(
    model,
    pc_noisy,
    patch_size,
    seed_k,
    steps,
    chunk=8,
    rng=None,
    recompute_neighbors=False,
    patch_frame="local",
    patch_aggregation="residual",
    patch_agg_beta=4.0,
):
    """
    Denoise a full point cloud patch-by-patch.

    pc_noisy : (N, 3) float32, already normalised to unit sphere
    patch_size: points per patch (from config)
    seed_k   : num_patches = seed_k * N / patch_size
    steps    : steps T using the decreasing alpha schedule
    recompute_neighbors: rebuild kNN patches from current geometry at each step
    patch_frame: global or local. Local feeds seed-centered patch coordinates
                 to the model but assembles denoised global coordinates.
    patch_aggregation:
        position -> average denoised point positions from overlapping patches.
        residual -> average predicted residual vectors, then move each point once.
    patch_agg_beta: weight overlapping patches by exp(-beta * dist^2)
        (center-dominant); larger beta suppresses edge-patch votes more.

    Iterative schedule for T > 1
    -----------------------------
    Start from X_noisy (alpha=1). At each step:
        alpha_cur  = current cumulative alpha
        alpha_prev = previous cumulative alpha, or 0 for final clean

        model predicts residual R = X_noisy - X_clean
        X_next = X_current - (alpha_cur - alpha_prev) * R

    This applies only the step-sized residual instead of repeatedly applying
    the full noisy-to-clean displacement.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    patch_frame = normalize_patch_frame(patch_frame)
    patch_aggregation = normalize_patch_aggregation(patch_aggregation)

    N = pc_noisy.shape[0]
    num_patches = max(1, int(seed_k * N / patch_size))

    seed_idx = fps(pc_noisy, num_patches, rng)

    def build_patch_index(points):
        tree = cKDTree(points)
        seed_pts = points[seed_idx]  # fixed seed point identities, current coordinates
        patch_dists, nn_idx = tree.query(seed_pts, k=patch_size)
        patch_dists = patch_dists / (patch_dists[:, -1:] + 1e-8)  # (P, M) in [0,1]
        return nn_idx, patch_dists

    if not recompute_neighbors:
        nn_idx, patch_dists = build_patch_index(pc_noisy)

    alphas = alpha_schedule(steps)
    X_current = pc_noisy.copy()

    for step in range(steps - 1, -1, -1):
        if recompute_neighbors:
            nn_idx, patch_dists = build_patch_index(X_current)

        alpha_cur = float(alphas[step])
        alpha_prev = float(alphas[step - 1]) if step > 0 else 0.0
        alpha_step = alpha_cur - alpha_prev

        patches = X_current[nn_idx]  # (P, M, 3)

        patches_residual = np.zeros_like(patches)
        chunk_size = max(1, num_patches // chunk)

        with jt.no_grad():
            for i in range(0, num_patches, chunk_size):
                b = patches[i : i + chunk_size]
                B = b.shape[0]
                if patch_frame == "local":
                    centers = X_current[seed_idx[i : i + B]][:, None, :]
                    b_in = b - centers
                else:
                    b_in = b
                pc_in = jt.array(b_in).permute(0, 2, 1)  # (B, 3, M)
                t_in = jt.full((B,), float(step))
                res = model(pc_in, t_in).numpy()  # (B, M, 3)
                patches_residual[i : i + chunk_size] = res

        # Assemble: center-weighted mean of the overlapping per-patch votes.
        # Flatten the (P, M) patch votes and scatter-sum with bincount (vectorised
        # C, no Python loop over patches) -> weighted mean per point.
        residual_mode = patch_aggregation == "residual"
        idx = nn_idx.reshape(-1)  # (P*M,) target point of each vote
        d = patch_dists.reshape(-1)  # (P*M,) radial position, 0 at patch centre
        w = np.exp(-patch_agg_beta * d * d)  # center-dominant weight
        if residual_mode:
            votes = patches_residual.reshape(-1, 3)
        else:
            votes = (patches - alpha_step * patches_residual).reshape(-1, 3)
        weight_sum = np.bincount(idx, weights=w, minlength=N)
        pred_sum = np.stack(
            [np.bincount(idx, weights=w * votes[:, c], minlength=N) for c in range(3)],
            axis=1,
        )
        avg = (pred_sum / (weight_sum[:, None] + 1e-8)).astype(np.float32)
        X_step = X_current - alpha_step * avg if residual_mode else avg
        # points covered by no patch keep their position, not the origin
        uncovered = weight_sum == 0
        if uncovered.any():
            X_step[uncovered] = X_current[uncovered]

        X_current = X_step

    return X_current.astype(np.float32)


# -- Main -----------------------------------------------------------------------


def build_model(cfg):
    """Construct Flow2SurfNet from a config dict (shared by predict and eval)."""
    return Flow2SurfNet(
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model", required=True, help="Checkpoint path (.pkl)")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_path = args.model

    seed = cfg.get("inference_seed", cfg.get("seed", 42))
    steps = cfg["num_steps"]

    if os.path.exists(args.out):
        raise SystemExit(f"Output already exists: {args.out}")

    random.seed(seed)
    np.random.seed(seed)
    jt.set_global_seed(seed)
    jt.flags.use_cuda = 1
    model = build_model(cfg)
    model.load(model_path)
    model.eval()
    log.info(f"Loaded {model_path}  (steps={steps})")

    test_root = os.path.join(cfg["test_dir"], "shapenet")
    total = 0

    for synset in sorted(os.listdir(test_root)):
        for model_id in sorted(os.listdir(os.path.join(test_root, synset))):
            noisy_path = os.path.join(test_root, synset, model_id, "noisy.npy")
            if not os.path.exists(noisy_path):
                continue

            pc_noisy = np.load(noisy_path)  # (N, 3) float32

            denoised = patch_denoise(
                model,
                pc_noisy,
                patch_size=cfg["patch_size"],
                seed_k=cfg["seed_k"],
                steps=steps,
                rng=make_rng(seed, synset, model_id),
                recompute_neighbors=cfg.get("recompute_neighbors", False),
                patch_frame=cfg.get("patch_frame", "local"),
                patch_aggregation=cfg.get("patch_aggregation", "residual"),
                patch_agg_beta=cfg.get("patch_agg_beta", 4),
            )

            out_dir = os.path.join(args.out, "shapenet", synset, model_id)
            os.makedirs(out_dir, exist_ok=True)
            np.save(os.path.join(out_dir, "denoised.npy"), denoised)
            total += 1

            if total % 10 == 0:
                log.info(f"Processed {total} samples")

    log.info(f"Done - {total} samples saved to {args.out}/")


if __name__ == "__main__":
    main()
