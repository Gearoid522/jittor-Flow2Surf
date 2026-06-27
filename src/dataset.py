import os
import logging
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from jittor.dataset import Dataset

log = logging.getLogger("flow2surf.dataset")


def normalize_patch_frame(patch_frame):
    if patch_frame not in {"global", "local"}:
        raise ValueError(
            f"Unknown patch_frame: {patch_frame}. Valid values: global, local"
        )
    return patch_frame


def load_obj_paths(data_dir):
    paths = []
    shapenet_dir = os.path.join(data_dir, "shapenet")
    for synset in sorted(os.listdir(shapenet_dir)):
        synset_dir = os.path.join(shapenet_dir, synset)
        if not os.path.isdir(synset_dir):
            continue
        for model in sorted(os.listdir(synset_dir)):
            obj = os.path.join(synset_dir, model, "models", "model_normalized.obj")
            if os.path.exists(obj):
                paths.append(obj)
    return paths


def load_mesh(path):
    mesh = trimesh.load(path, process=False, force="mesh")
    return np.array(mesh.vertices, dtype=np.float32), np.array(
        mesh.faces, dtype=np.int32
    )


def sample_surface(vertices, faces, n, seed=None):
    """Sample n surface points with their exact face normals -> (pts, normals)."""
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    pts, face_idx = trimesh.sample.sample_surface(mesh, n, seed=seed)
    normals = mesh.face_normals[face_idx]
    return np.array(pts, dtype=np.float32), np.array(normals, dtype=np.float32)


def normalize(pts):
    lo, hi = pts.min(0), pts.max(0)
    center = (lo + hi) / 2
    shifted = pts - center
    scale = np.sqrt((shifted**2).sum(1)).max()
    return shifted / (scale + 1e-8)


def add_noise(pts, std_min, std_max, noise_type="laplace", rng=None):
    rng = rng or np.random
    std = rng.uniform(std_min, std_max)
    if noise_type == "mixed":
        noise_type = rng.choice(["laplace", "gaussian"])
    if noise_type == "gaussian":
        noise = rng.normal(0, std, pts.shape).astype(np.float32)
    else:
        noise = rng.laplace(0, std, pts.shape).astype(np.float32)
    return pts + noise


def rng_integers(rng, low, high=None, size=None):
    if hasattr(rng, "integers"):
        return rng.integers(low, high, size=size)
    return rng.randint(low, high, size=size)


def alpha_schedule(num_steps):
    """
    Decreasing cumulative alpha schedule.

    Returns cumulative alpha_bar values for t = 0 ... T-1:
        alpha_bar_t = (t + 1) * (2T - t) / (T * (T + 1))

    alpha_bar_0 is near clean, alpha_bar_{T-1} is exactly noisy.
    """
    T = int(num_steps)
    if T < 1:
        raise ValueError("num_steps must be >= 1")
    t = np.arange(T, dtype=np.float32)
    return ((t + 1.0) * (2.0 * T - t) / (T * (T + 1.0))).astype(np.float32)


def _random_rotations(rng, n):
    """`n` uniform-random SO(3) rotation matrices (n, 3, 3) from unit quaternions."""
    q = rng.standard_normal((n, 4))
    q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ],
        axis=1,
    ).reshape(n, 3, 3)
    return R.astype(np.float32)


def augment_patches(
    pat_clean, pat_noisy, pat_mix, pat_normals, rng, rotate, scale, local
):
    """Per-patch SO(3) rotation + isotropic scale, applied identically to clean,
    noisy and mix; the shared transform cancels in the residual (noisy-clean) and
    preserves the mix interpolation. Normals get the rotation only (they are
    directions - no pivot, no scale). Pivot: patch origin (local) or centroid."""
    P = pat_mix.shape[0]
    pivot = 0.0 if local else pat_mix.mean(axis=1, keepdims=True)
    R = _random_rotations(rng, P) if rotate else None
    s = (
        (1.0 + rng.uniform(-scale, scale, size=(P, 1, 1))).astype(np.float32)
        if scale > 0
        else None
    )

    def xf(p):
        p = p - pivot
        if R is not None:
            p = np.einsum("pij,pmj->pmi", R, p)
        if s is not None:
            p = p * s
        return (p + pivot).astype(np.float32)

    nrm = pat_normals if R is None else np.einsum("pij,pmj->pmi", R, pat_normals)
    return xf(pat_clean), xf(pat_noisy), xf(pat_mix), nrm.astype(np.float32)


def extract_patches(
    clean,
    noisy,
    patch_size,
    num_patches,
    num_steps=30,
    rng=None,
    patch_frame="local",
    clean_normals=None,
    augment_rotation=False,
    augment_scale=0.0,
):
    """
    Sample `num_patches` patches and their step indices.

    Convention  alpha = 0 -> fully clean,  alpha = 1 -> fully noisy.
    Mix formula  X_alpha = X_clean + alpha * (X_noisy - X_clean).
    Each patch samples one discrete step index and uses its cumulative
    alpha for interpolation.

    patch_frame:
        global -> patches keep global object-normalized coordinates.
        local  -> subtract each patch seed coordinate from clean/noisy/mix.
                  The residual target is unchanged because the same center
                  is subtracted from both noisy and clean.

    augment_rotation / augment_scale:
        per-patch SO(3) rotation and +/-fraction isotropic scale (training only),
        applied identically to clean/noisy/mix so the residual and mix are kept.

    Returns
    -------
    pat_noisy  : (P, M, 3)
    pat_mix    : (P, M, 3)
    pat_clean  : (P, M, 3)
    pat_t      : (P,)       step index for each patch
    full_noisy : (N, 3)     full noisy cloud
    pat_normals: (P, M, 3)  exact clean surface normals (rotated with the patch)
    """
    rng = rng or np.random
    patch_frame = normalize_patch_frame(patch_frame)

    N = noisy.shape[0]
    seed_idx = rng.choice(N, num_patches, replace=False)

    tree = cKDTree(noisy)
    _, nn_idx = tree.query(noisy[seed_idx], k=patch_size)  # (P, M)

    pat_clean = clean[nn_idx]  # (P, M, 3)
    pat_noisy = noisy[nn_idx]  # (P, M, 3)

    alphas = alpha_schedule(num_steps)
    t_idx = rng_integers(rng, 0, len(alphas), size=num_patches)
    t = alphas[t_idx].astype(np.float32)
    t_bc = t[:, None, None]

    pat_mix = pat_clean + t_bc * (pat_noisy - pat_clean)  # (P, M, 3)

    if patch_frame == "local":
        seed_clean = clean[seed_idx]
        seed_noisy = noisy[seed_idx]
        patch_center = (seed_clean + t[:, None] * (seed_noisy - seed_clean))[:, None, :]
        patch_center = patch_center.astype(np.float32)
        pat_clean = pat_clean - patch_center
        pat_noisy = pat_noisy - patch_center
        pat_mix = pat_mix - patch_center

    pat_normals = (
        clean_normals[nn_idx] if clean_normals is not None else np.zeros_like(pat_clean)
    )

    if augment_rotation or augment_scale > 0:
        pat_clean, pat_noisy, pat_mix, pat_normals = augment_patches(
            pat_clean,
            pat_noisy,
            pat_mix,
            pat_normals,
            rng,
            rotate=augment_rotation,
            scale=augment_scale,
            local=(patch_frame == "local"),
        )

    return (
        pat_noisy.astype(np.float32),
        pat_mix.astype(np.float32),
        pat_clean.astype(np.float32),
        t_idx.astype(np.float32),
        noisy.astype(np.float32),
        pat_normals.astype(np.float32),
    )


class PointCloudDataset(Dataset):
    def __init__(self, paths, cfg, shuffle=True, deterministic=False):
        super().__init__()
        self.paths = paths
        self.cfg = cfg
        self.deterministic = deterministic
        self.set_attrs(
            total_len=len(paths),
            batch_size=cfg["batch_size"],
            shuffle=shuffle,
            num_workers=cfg["num_workers"],
        )

    def __getitem__(self, idx):
        last_exc = None
        n = len(self.paths)
        for attempt in range(min(n, 8)):
            try_idx = (idx + attempt) % n
            path = self.paths[try_idx]
            try:
                rng = None
                sample_seed = None
                if self.deterministic:
                    sample_seed = int(self.cfg["seed"]) + int(try_idx)
                    rng = np.random.default_rng(sample_seed)

                verts, faces = load_mesh(path)
                clean, clean_normals = sample_surface(
                    verts, faces, self.cfg["num_samples"], seed=sample_seed
                )
                clean = normalize(
                    clean
                )  # uniform scale + shift; normal directions unchanged
                noisy = add_noise(
                    clean,
                    self.cfg["noise_std_min"],
                    self.cfg["noise_std_max"],
                    self.cfg["noise_type"],
                    rng=rng,
                )
                augment = not self.deterministic  # never augment the validation split
                return extract_patches(
                    clean,
                    noisy,
                    self.cfg["patch_size"],
                    self.cfg["num_patches"],
                    self.cfg.get("num_steps", 30),
                    rng=rng,
                    patch_frame=self.cfg.get("patch_frame", "local"),
                    clean_normals=clean_normals,
                    augment_rotation=augment
                    and self.cfg.get("augment_rotation", False),
                    augment_scale=(
                        self.cfg.get("augment_scale", 0.0) if augment else 0.0
                    ),
                )
            except Exception as e:
                split = "validation" if self.deterministic else "training"
                log.warning(
                    "Failed to load %s sample %s (attempt %d): %s",
                    split,
                    path,
                    attempt + 1,
                    e,
                )
                last_exc = e
        raise RuntimeError(
            f"Could not load any sample after 8 attempts starting at index {idx}"
        ) from last_exc


def build_datasets(cfg):
    paths = load_obj_paths(cfg["data_dir"])
    np.random.seed(cfg["seed"])
    np.random.shuffle(paths)
    n_val = max(1, int(len(paths) * cfg["val_split"]))
    val_paths = paths[:n_val]
    train_paths = paths[n_val:]
    train_ds = PointCloudDataset(train_paths, cfg, shuffle=True, deterministic=False)
    val_ds = PointCloudDataset(val_paths, cfg, shuffle=False, deterministic=True)
    return train_ds, val_ds
