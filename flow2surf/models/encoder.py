"""Local neighbor encoding for Flow2Surf point patches."""

import jittor as jt
from jittor import nn

from .layers import ChannelLayerNorm
from .ops import knn_indices, reduce_neighbor_messages


def _gather_neighbors(x: jt.Var, indices: jt.Var) -> jt.Var:
    """Gather indexed neighbors from a batched point tensor.

    x       : (B, C, N)
    indices : (B, N, K)
    output  : (B, C, N, K)
    """
    batch, channels, points = x.shape
    neighbors = indices.shape[-1]
    x_t = x.permute(0, 2, 1)
    batch_offset = (jt.arange(batch) * points).reshape(batch, 1, 1)
    flat_indices = (indices + batch_offset).reshape(-1)
    gathered = x_t.reshape(batch * points, channels)[flat_indices]
    return gathered.reshape(batch, points, neighbors, channels).permute(0, 3, 1, 2)


def _relative_neighbor_features(x: jt.Var, indices: jt.Var) -> jt.Var:
    """Build [neighbor - center, center] features for fixed neighbors.

    x       : (B, C, N)
    indices : (B, N, K)
    output  : (B, 2C, N, K)
    """
    batch, channels, points = x.shape
    neighbor_count = indices.shape[-1]
    neighbors = _gather_neighbors(x, indices)
    center = x.unsqueeze(-1).expand(batch, channels, points, neighbor_count)
    return jt.concat([neighbors - center, center], dim=1)


def _point_mlp(mlp: nn.Module, x: jt.Var) -> jt.Var:
    """Apply an MLP independently to each point."""
    return mlp(x.permute(0, 2, 1)).permute(0, 2, 1)


class HPE(nn.Module):
    """High-dimensional Positional Encoding for coordinate features."""

    def __init__(self, width: int, in_dim: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, width),
            nn.LayerNorm(width),
            nn.GELU(),
            nn.Linear(width, width),
        )

    @property
    def input_projection(self):
        return self.net[0]

    @property
    def hidden_norm(self):
        return self.net[1]

    @property
    def output_projection(self):
        return self.net[3]

    def execute(self, x):
        hidden = self.net[2](self.hidden_norm(self.input_projection(x)))
        return self.output_projection(hidden)


class _NeighborStage(nn.Module):
    """Encode local geometry and reduce neighbor messages."""

    def __init__(self, in_channels: int, out_channels: int, position_hpe: HPE):
        super().__init__()
        self.position_embed = position_hpe
        self.projection = nn.Conv2d(3 * in_channels, out_channels, 1, bias=False)
        self.norm = ChannelLayerNorm(out_channels, affine=True)

    def execute(self, features, xyz, indices):
        channels = features.shape[1]
        geometry = _relative_neighbor_features(xyz, indices).permute(0, 2, 3, 1)

        # Projection blocks act on delta, center, and positional HPE features.
        weight = self.projection.weight[:, :, 0, 0]
        delta_weight = weight[:, :channels]
        center_weight = weight[:, channels : 2 * channels]
        position_projection = weight[:, 2 * channels :]

        content_weight = (
            jt.concat([delta_weight, center_weight - delta_weight], dim=0)
            .unsqueeze(-1)
            .unsqueeze(-1)
        )
        content = nn.conv2d(features.unsqueeze(-1), content_weight).squeeze(-1)
        hpe_output = self.position_embed.output_projection
        hpe_output_weight = jt.matmul(position_projection, hpe_output.weight)
        hpe_output_bias = jt.matmul(position_projection, hpe_output.bias)
        hpe_input = self.position_embed.input_projection
        return reduce_neighbor_messages(
            content,
            geometry,
            indices,
            hpe_input.weight,
            hpe_input.bias,
            hpe_output_weight,
            hpe_output_bias,
            self.position_embed.hidden_norm.weight,
            self.position_embed.hidden_norm.bias,
            self.norm.weight,
            self.norm.bias,
        )


def _projection_head(in_channels: int, out_channels: int) -> nn.Module:
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, 1, bias=False),
        ChannelLayerNorm(out_channels, affine=True),
        nn.LeakyReLU(scale=0.2),
    )


class NeighborEncoder(nn.Module):
    """
    Encode local content and geometry with neighbor-order-invariant stages.

    Input xyz is projected once. Every stage reuses one xyz kNN graph and
    combines content with geometry:

        f0         = HPE(xyz)
        idx        = kNN(xyz)
        content    = [f_j - f_i, f_i]
        positional = HPE([xyz_j - xyz_i, xyz_i])
        message    = Linear([content, positional])
        f          = max_k(message)

    Content projection factorizes as Wd(f_j - f_i) + Wc f_i = Wd f_j +
    (Wc - Wd)f_i. Composing the HPE output projection with its message-projection
    block is also exact. Both factorizations avoid storing the full
    neighbor-message tensor.

    Reusing fixed spatial neighbors keeps local geometry semantics stable and
    avoids repeated graph searches. All stage outputs are concatenated and
    projected to the requested output width.
    """

    def __init__(
        self,
        k=32,
        out_dim=128,
        dims=(64, 64),
    ):
        super().__init__()
        self.k = k
        self.stage_dims = list(dims) + [out_dim]

        input_width = self.stage_dims[0]
        stage_inputs = [input_width] + self.stage_dims[:-1]
        self.input_embed = HPE(input_width, in_dim=3)

        position_encodings = [HPE(width, in_dim=6) for width in stage_inputs]
        self.stages = nn.ModuleList(
            [
                _NeighborStage(in_channels, out_channels, position_hpe)
                for in_channels, out_channels, position_hpe in zip(
                    stage_inputs, self.stage_dims, position_encodings
                )
            ]
        )
        self.proj = _projection_head(sum(self.stage_dims), out_dim)

    def execute(self, x):
        xyz = x
        features = _point_mlp(self.input_embed, xyz)
        indices = knn_indices(xyz, self.k)

        outputs = []
        for stage in self.stages:
            features = stage(features, xyz, indices)
            outputs.append(features)
        return self.proj(jt.concat(outputs, dim=1))
