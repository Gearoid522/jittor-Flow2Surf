import jittor as jt
from jittor import nn


def knn_indices(x: jt.Var, k: int) -> jt.Var:
    """
    x:   (B, C, N)
    out: (B, N, k)
    """
    B, _, N = x.shape
    if N <= 1:
        return jt.zeros((B, N, k), dtype="int32")

    x_t = x.permute(0, 2, 1)
    inner = jt.nn.bmm(x_t, x)
    sq = (x * x).sum(dim=1)
    dist = sq.unsqueeze(2) + sq.unsqueeze(1) - 2 * inner
    ar = jt.arange(N)
    self_mask = (ar.reshape(1, N, 1) == ar.reshape(1, 1, N)).float32()
    dist = dist + self_mask * 1e10
    idx = (-dist).topk(k=min(k, N - 1), dim=-1)[1]
    if idx.shape[-1] < k:
        pad = idx[:, :, -1:].expand(B, N, k - idx.shape[-1])
        idx = jt.concat([idx, pad], dim=-1)
    return idx


def gather_neighbors(x: jt.Var, idx: jt.Var) -> jt.Var:
    """
    Gather neighbor features (no centre concat).
    x:   (B, C, N)
    idx: (B, N, k)
    out: (B, C, N, k)
    """
    B, C, N = x.shape
    k = idx.shape[-1]
    x_t = x.permute(0, 2, 1)
    batch_offset = (jt.arange(B) * N).reshape(B, 1, 1)
    idx_flat = (idx + batch_offset).reshape(-1)
    neighbors = x_t.reshape(B * N, C)[idx_flat].reshape(B, N, k, C)
    return neighbors.permute(0, 3, 1, 2)


def edge_feature_from_idx(x: jt.Var, idx: jt.Var) -> jt.Var:
    """
    x:   (B, C, N)
    idx: (B, N, k)
    out: (B, 2C, N, k), [neighbor - centre, centre]
    """
    B, C, N = x.shape
    k = idx.shape[-1]
    neighbors = gather_neighbors(x, idx)  # (B, C, N, k)
    center = x.unsqueeze(-1).expand(B, C, N, k)
    return jt.concat([neighbors - center, center], dim=1)


def grid_mlp(mlp: nn.Module, feat: jt.Var) -> jt.Var:
    """Apply a per-point MLP across an (N, k) edge grid. feat (B, Cin, N, k) -> (B, Cout, N, k)."""
    B, Cin, N, k = feat.shape
    flat = feat.permute(0, 2, 3, 1).reshape(-1, Cin)
    return mlp(flat).reshape(B, N, k, -1).permute(0, 3, 1, 2)


def point_mlp(mlp: nn.Module, x: jt.Var) -> jt.Var:
    """Apply a per-point MLP over points. x (B, Cin, N) -> (B, Cout, N)."""
    return mlp(x.permute(0, 2, 1)).permute(0, 2, 1)


class HPE(nn.Module):
    """Project an xyz position component (in_dim channels) into a stage-width space."""

    def __init__(self, width: int, in_dim: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, width),
            nn.LayerNorm(width),
            nn.GELU(),
            nn.Linear(width, width),
        )

    def execute(self, x):
        return self.net(x)


def _conv_stage(in_c: int, out_c: int) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.LeakyReLU(scale=0.2),
    )


def _proj_head(in_c: int, out_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.Conv1d(in_c, out_dim, 1, bias=False),
        nn.BatchNorm1d(out_dim),
        nn.LeakyReLU(scale=0.2),
    )


class NeighborEmbedding(nn.Module):
    """
    Multi-stage EdgeConv embedder with concat-fused positional encoding.

    xyz is embedded once into features; then every stage runs a uniform EdgeConv
    whose content edge is concatenated with a positional feature built from the
    full xyz edge [delta_xyz, centre_xyz]:

        f0   = MLP(xyz)                        # embed position once, per point
        edge = [f_j - f_i, f_i]                # EdgeConv content edge
        pos  = MLP([xyz_j - xyz_i, xyz_i])     # relative + absolute position
        f    = Conv2d( fuse(edge, pos) ).max(k)

    fusion controls how pos is combined with the edge:
      "concat" -> Conv2d(concat[edge, pos])   (pos in its own channels)
      "add"    -> Conv2d(edge + pos)          (pos summed into the edge channels)
    Concat gives the conv independent weights for content vs position. Neighbors
    come from the dynamic feature-space graph (DGCNN-style), rebuilt each stage.
    concat_all fuses all stage outputs before the final projection. dims are the
    intermediate stage widths; the final stage width is out_dim.
    """

    def __init__(
        self, k=32, out_dim=128, dims=(64, 64), concat_all=True, fusion="concat"
    ):
        super().__init__()
        if fusion not in ("add", "concat"):
            raise ValueError(f"fusion must be 'add' or 'concat', got {fusion}")
        self.k = k
        self.concat_all = concat_all
        self.fusion = fusion
        self.all_dims = list(dims) + [out_dim]

        w0 = self.all_dims[0]
        prev = [w0] + self.all_dims[:-1]  # input feature width to each stage's edge
        self.input_embed = HPE(w0, in_dim=3)  # xyz -> w0, once per point

        # add: pos matches the edge channels (2*prev) and is summed in.
        # concat: pos gets its own prev channels alongside the 2*prev edge.
        pos_widths, conv_ins = [], []
        for i in range(len(self.all_dims)):
            edge_c = 2 * prev[i]
            if fusion == "add":
                pos_widths.append(edge_c)
                conv_ins.append(edge_c)
            else:
                pos_widths.append(prev[i])
                conv_ins.append(edge_c + prev[i])
        self.pos = nn.ModuleList(
            [HPE(w, in_dim=6) for w in pos_widths]
        )  # [delta, centre]
        self.stages = nn.ModuleList(
            [_conv_stage(conv_ins[i], d) for i, d in enumerate(self.all_dims)]
        )
        if concat_all:
            self.proj = _proj_head(sum(self.all_dims), out_dim)

    def execute(self, x):
        xyz = x
        f = point_mlp(self.input_embed, xyz)  # (B, w0, N)
        outputs = []
        for i, stage in enumerate(self.stages):
            idx = knn_indices(f, self.k)  # dynamic feature-space graph
            edge = edge_feature_from_idx(f, idx)  # (B, 2*prev, N, k) [f_j-f_i, f_i]
            pos_in = edge_feature_from_idx(
                xyz, idx
            )  # (B, 6, N, k) [delta_xyz, centre_xyz]
            pos = grid_mlp(self.pos[i], pos_in)
            if self.fusion == "add":
                f = stage(edge + pos).max(dim=-1)
            else:
                f = stage(jt.concat([edge, pos], dim=1)).max(dim=-1)
            outputs.append(f)
        if self.concat_all:
            return self.proj(jt.concat(outputs, dim=1))
        return outputs[-1]
