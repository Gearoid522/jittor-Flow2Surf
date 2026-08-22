"""Mesh sampling, synthetic noise, flow patches, and dataset construction."""

from dataclasses import dataclass
import logging
import os

from jittor.dataset import Dataset
import numpy as np
from scipy.spatial.distance import cdist
import trimesh
import yaml

from .flow import flow_state

log = logging.getLogger(__name__)


_NOISE_TYPES = {"gaussian", "laplace"}


@dataclass(frozen=True)
class DatasetSpec:
    """Resolved roots and exact sample manifests for one dataset."""

    name: str
    train_root: str
    test_root: str
    train_manifest: str
    val_manifest: str
    test_manifest: str


def load_dataset_spec(path):
    """Load and resolve one dataset descriptor relative to its own directory."""
    path = os.path.abspath(path)
    with open(path) as file:
        values = yaml.safe_load(file)
    required = {
        "name",
        "train_root",
        "test_root",
        "train_manifest",
        "val_manifest",
        "test_manifest",
    }
    if not isinstance(values, dict):
        raise ValueError(f"Dataset descriptor must be a mapping: {path}")
    missing = required - values.keys()
    extra = values.keys() - required
    if missing or extra:
        raise ValueError(
            f"Invalid dataset descriptor {path}: "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )
    if not isinstance(values["name"], str) or not values["name"].strip():
        raise ValueError(f"Dataset name must be a non-empty string: {path}")

    base = os.path.dirname(path)

    def resolve(key):
        value = values[key]
        if not isinstance(value, str) or not value:
            raise ValueError(f"Dataset field {key} must be a non-empty path: {path}")
        return os.path.abspath(os.path.join(base, value))

    return DatasetSpec(
        name=values["name"],
        train_root=resolve("train_root"),
        test_root=resolve("test_root"),
        train_manifest=resolve("train_manifest"),
        val_manifest=resolve("val_manifest"),
        test_manifest=resolve("test_manifest"),
    )


def _manifest_entries(path):
    """Read unique, relative model-directory paths from a manifest."""
    entries = []
    seen = set()
    with open(path) as file:
        for line_number, line in enumerate(file, 1):
            entry = line.strip()
            if not entry:
                continue
            normalized = os.path.normpath(entry)
            unsafe = (
                os.path.isabs(entry)
                or normalized == ".."
                or normalized.startswith("../")
            )
            if unsafe:
                raise ValueError(f"Unsafe path at {path}:{line_number}: {entry}")
            if normalized in seen:
                raise ValueError(f"Duplicate path at {path}:{line_number}: {entry}")
            seen.add(normalized)
            entries.append(normalized)
    if not entries:
        raise ValueError(f"Manifest is empty: {path}")
    return entries


def _listed_paths(root, entries, suffix):
    """Resolve manifest model directories to existing sample files."""
    paths = []
    for entry in entries:
        path = os.path.join(root, entry, suffix)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Listed sample not found: {path}")
        paths.append(path)
    return paths


def mesh_split_paths(dataset):
    """Return the exact training and validation mesh paths from manifests."""
    train_entries = _manifest_entries(dataset.train_manifest)
    val_entries = _manifest_entries(dataset.val_manifest)
    train_paths = _listed_paths(
        dataset.train_root,
        train_entries,
        os.path.join("models", "model_normalized.obj"),
    )
    val_paths = _listed_paths(
        dataset.train_root,
        val_entries,
        os.path.join("models", "model_normalized.obj"),
    )
    overlap = set(train_paths) & set(val_paths)
    if overlap:
        raise ValueError(
            f"Train and validation manifests overlap at {len(overlap)} samples"
        )
    return train_paths, val_paths


def test_cloud_paths(dataset):
    """Return (synset, model, noisy path) records in manifest order."""
    entries = _manifest_entries(dataset.test_manifest)
    paths = _listed_paths(dataset.test_root, entries, "noisy.npy")
    records = []
    for entry, path in zip(entries, paths):
        parts = entry.split(os.sep)
        if len(parts) < 2:
            raise ValueError(f"Test manifest entry lacks synset/model IDs: {entry}")
        records.append((parts[-2], parts[-1], path))
    return records


def load_mesh(path):
    """Load mesh vertices and triangular faces as NumPy arrays."""
    mesh = trimesh.load(
        path,
        process=False,
        force="mesh",
        skip_materials=True,
    )
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices):
        raise ValueError(f"Mesh has invalid vertices: {path}")
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces):
        raise ValueError(f"Mesh has no triangular faces: {path}")
    if not np.isfinite(vertices).all():
        raise ValueError(f"Mesh has non-finite vertices: {path}")
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError(f"Mesh has out-of-range face indices: {path}")
    return vertices, faces


def sample_surface(vertices, faces, num_points, rng):
    """Sample (num_points, 3) points with probability uniform over mesh area."""
    num_points = int(num_points)
    if num_points < 1:
        raise ValueError(f"num_points must be >= 1, got {num_points}")

    triangles = np.asarray(vertices, dtype=np.float64)[faces]
    cumulative_area = np.cumsum(trimesh.triangles.area(triangles))
    if not len(cumulative_area):
        raise ValueError("Mesh must contain at least one triangle")
    total_area = cumulative_area[-1]
    if not np.isfinite(total_area) or total_area <= 0:
        raise ValueError("Mesh must have finite positive surface area")

    face_idx = np.searchsorted(
        cumulative_area,
        rng.random(num_points) * total_area,
    )

    origins = triangles[:, 0]
    vectors = triangles[:, 1:] - origins[:, None, :]
    origins = origins[face_idx]
    vectors = vectors[face_idx]

    edge_weights = rng.random((num_points, 2, 1))
    reflected = edge_weights.sum(axis=1).reshape(-1) > 1.0
    edge_weights[reflected] = 1.0 - edge_weights[reflected]
    return ((vectors * edge_weights).sum(axis=1) + origins).astype(np.float32)


def normalize_points(points):
    """Return (points - center) / radius, center, and maximum Euclidean radius."""
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points):
        raise ValueError(
            f"Expected non-empty points with shape (N, 3), got {points.shape}"
        )
    if not np.isfinite(points).all():
        raise ValueError("Points must be finite")

    lo, hi = points.min(0), points.max(0)
    center = (lo + hi) / 2
    centered = points - center
    radius = float(np.linalg.norm(centered, axis=1).max())
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("Points must have positive finite radius")
    return centered / radius, center, radius


def sample_noise_std(cfg, rng):
    """Sample a coordinate standard deviation under the configured measure."""
    std_min, std_max = cfg["noise_std_range"]
    sampling = cfg["noise_std_sampling"]
    if sampling == "linear":
        return float(rng.uniform(std_min, std_max))
    if sampling == "log":
        return float(np.exp(rng.uniform(np.log(std_min), np.log(std_max))))
    raise ValueError(f"Unknown noise_std_sampling: {sampling}")


def _sample_unit_directions(count, rng):
    """Sample `count` independent directions uniformly on the unit sphere."""
    directions = rng.normal(size=(count, 3))
    lengths = np.linalg.norm(directions, axis=1, keepdims=True)
    zero = lengths[:, 0] == 0
    while zero.any():
        directions[zero] = rng.normal(size=(int(zero.sum()), 3))
        lengths[zero] = np.linalg.norm(directions[zero], axis=1, keepdims=True)
        zero = lengths[:, 0] == 0
    return directions / lengths


def add_noise(points, noise_std, noise_type, rng):
    """Add pointwise spherical noise with covariance `noise_std^2 I`."""
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected points with shape (N, 3), got {points.shape}")
    if not np.isfinite(points).all():
        raise ValueError("Points must be finite")
    if noise_type not in _NOISE_TYPES:
        valid = ", ".join(sorted(_NOISE_TYPES))
        raise ValueError(f"Unknown noise_type: {noise_type}. Valid values: {valid}")
    noise_std = float(noise_std)
    if not np.isfinite(noise_std) or noise_std < 0:
        raise ValueError(f"noise_std must be finite and non-negative, got {noise_std}")
    if noise_type == "gaussian":
        noise = rng.normal(0, noise_std, points.shape).astype(np.float32)
    elif noise_type == "laplace":
        direction = _sample_unit_directions(len(points), rng)
        radius = rng.gamma(3.0, noise_std / 2.0, size=(len(points), 1))
        noise = (radius * direction).astype(np.float32)
    return points + noise


def _nearest_indices(points, queries, k):
    """Return k nearest point indices in ascending squared-distance order."""
    points = np.asarray(points)
    queries = np.asarray(queries)
    if points.ndim != 2 or queries.ndim != 2 or points.shape[1:] != queries.shape[1:]:
        raise ValueError(
            f"Expected points and queries with matching (N, D) shapes, got "
            f"{points.shape} and {queries.shape}"
        )
    if not 1 <= k <= len(points):
        raise ValueError(f"k must be in [1, {len(points)}], got {k}")
    sq_dist = cdist(queries, points, metric="sqeuclidean")
    indices = np.argpartition(sq_dist, k - 1, axis=1)[:, :k]
    selected = np.take_along_axis(sq_dist, indices, axis=1)
    order = np.argsort(selected, axis=1, kind="stable")
    return np.take_along_axis(indices, order, axis=1)


def _extract_patches(clean, noisy, patch_size, seed_idx, alpha):
    """Extract seed-relative patches at supplied continuous flow levels.

    alpha = 0 is clean and alpha = 1 is noisy:

        X_alpha = X_clean + alpha * (X_noisy - X_clean)

    Neighborhoods are selected in the noisy cloud. All three states subtract
    the seed at X_alpha, preserving the velocity X_noisy - X_clean exactly.

    Returns
    -------
    patch_noisy : (P, M, 3)
    patch_alpha : (P, M, 3)
    patch_clean : (P, M, 3)
    alpha       : (P,)
    """
    clean = np.asarray(clean, dtype=np.float32)
    noisy = np.asarray(noisy, dtype=np.float32)
    if clean.shape != noisy.shape or clean.ndim != 2 or clean.shape[1] != 3:
        raise ValueError(
            f"Expected matching clean/noisy shapes (N, 3), got "
            f"{clean.shape} and {noisy.shape}"
        )
    num_points = len(noisy)
    seed_idx = np.asarray(seed_idx, dtype=np.int64)
    alpha = np.asarray(alpha, dtype=np.float32)
    if seed_idx.ndim != 1 or not len(seed_idx):
        raise ValueError(
            f"Expected non-empty seed_idx with shape (P,), got {seed_idx.shape}"
        )
    if seed_idx.min() < 0 or seed_idx.max() >= num_points:
        raise ValueError("seed_idx contains an out-of-range point index")
    if alpha.shape != seed_idx.shape:
        raise ValueError(f"Expected alpha shape {seed_idx.shape}, got {alpha.shape}")
    if not np.isfinite(alpha).all() or (alpha < 0).any() or (alpha > 1).any():
        raise ValueError("alpha values must be finite and in [0, 1]")

    nn_idx = _nearest_indices(noisy, noisy[seed_idx], patch_size)  # (P, M)

    patch_clean = clean[nn_idx]
    patch_noisy = noisy[nn_idx]

    alpha_view = alpha[:, None, None]

    patch_alpha = flow_state(patch_clean, patch_noisy, alpha_view)

    seed_clean = clean[seed_idx]
    seed_noisy = noisy[seed_idx]
    seed_alpha = flow_state(seed_clean, seed_noisy, alpha[:, None])[:, None, :]
    seed_alpha = seed_alpha.astype(np.float32)
    patch_clean = patch_clean - seed_alpha
    patch_noisy = patch_noisy - seed_alpha
    patch_alpha = patch_alpha - seed_alpha

    return (
        patch_noisy.astype(np.float32),
        patch_alpha.astype(np.float32),
        patch_clean.astype(np.float32),
        alpha,
    )


class FlowPatchDataset(Dataset):
    """Sample flow patches for the configured noise family.

    Each item returns noisy, interpolated, and clean arrays with shape
    (P, M, 3), plus an alpha vector with shape (P,).
    """

    def __init__(self, paths, cfg, deterministic):
        super().__init__()
        self.paths = paths
        self.cfg = cfg
        self.deterministic = deterministic
        self._sample_cache = {} if deterministic else None
        self.set_attrs(
            total_len=len(paths),
            batch_size=cfg["batch_size"],
            shuffle=not deterministic,
            num_workers=cfg["num_workers"],
        )

    def __getitem__(self, idx):
        idx = int(idx)
        if self._sample_cache is not None:
            cached = self._sample_cache.get(idx)
            if cached is not None:
                return cached

        last_exc = None
        num_meshes = len(self.paths)
        max_attempts = min(num_meshes, 8)
        for attempt in range(max_attempts):
            path_idx = (idx + attempt) % num_meshes
            path = self.paths[path_idx]
            try:
                rng = np.random
                if self.deterministic:
                    sample_seed = int(self.cfg["seed"]) + int(path_idx)
                    rng = np.random.default_rng(sample_seed)

                vertices, faces = load_mesh(path)
                clean = sample_surface(
                    vertices,
                    faces,
                    self.cfg["num_samples"],
                    rng,
                )
                clean, _, _ = normalize_points(clean)
                noise_std = sample_noise_std(self.cfg, rng)
                noisy = add_noise(clean, noise_std, self.cfg["noise_type"], rng)
                seed_idx = rng.choice(
                    len(clean), self.cfg["num_patches"], replace=False
                )
                alpha = rng.uniform(
                    self.cfg["train_alpha_min"], 1.0, size=self.cfg["num_patches"]
                ).astype(np.float32)
                sample = _extract_patches(
                    clean,
                    noisy,
                    self.cfg["patch_size"],
                    seed_idx,
                    alpha,
                )
                if self._sample_cache is not None:
                    self._sample_cache[idx] = sample
                return sample
            except Exception as e:
                split = "validation" if self.deterministic else "training"
                log.warning(
                    "Failed to prepare %s sample %s (attempt %d): %s",
                    split,
                    path,
                    attempt + 1,
                    e,
                )
                last_exc = e
        raise RuntimeError(
            f"Could not prepare any sample after {max_attempts} attempts "
            f"starting at index {idx}"
        ) from last_exc


def _validate_noise_config(cfg):
    """Validate the training noise family and scale distribution."""
    noise_type = cfg["noise_type"]
    if not isinstance(noise_type, str) or noise_type not in _NOISE_TYPES:
        valid = ", ".join(sorted(_NOISE_TYPES))
        raise ValueError(f"Unknown noise_type: {noise_type}. Valid values: {valid}")

    values = np.asarray(cfg["noise_std_range"], dtype=np.float64)
    if values.shape != (2,) or not np.isfinite(values).all():
        raise ValueError("noise_std_range must contain two finite values")
    if values[0] < 0 or values[1] < values[0]:
        raise ValueError(f"Invalid noise_std_range: {values.tolist()}")
    sampling = cfg["noise_std_sampling"]
    if sampling not in ("linear", "log"):
        raise ValueError(
            f"noise_std_sampling must be 'linear' or 'log', got {sampling}"
        )
    if sampling == "log" and values[0] <= 0:
        raise ValueError("log noise_std_sampling requires a positive lower bound")


def build_datasets(cfg, dataset):
    """Build manifest-defined stochastic training and deterministic validation."""
    num_samples = int(cfg["num_samples"])
    patch_size = int(cfg["patch_size"])
    num_patches = int(cfg["num_patches"])
    if num_samples < 1:
        raise ValueError(f"num_samples must be >= 1, got {num_samples}")
    if not 1 <= patch_size <= num_samples:
        raise ValueError(f"patch_size must be in [1, {num_samples}], got {patch_size}")
    if not 1 <= num_patches <= num_samples:
        raise ValueError(
            f"num_patches must be in [1, {num_samples}], got {num_patches}"
        )
    _validate_noise_config(cfg)
    train_alpha_min = float(cfg["train_alpha_min"])
    if not np.isfinite(train_alpha_min) or not 0.0 <= train_alpha_min < 1.0:
        raise ValueError(f"train_alpha_min must be in [0, 1), got {train_alpha_min}")

    train_paths, val_paths = mesh_split_paths(dataset)
    train_ds = FlowPatchDataset(train_paths, cfg, deterministic=False)
    val_ds = FlowPatchDataset(val_paths, cfg, deterministic=True)
    return train_ds, val_ds
