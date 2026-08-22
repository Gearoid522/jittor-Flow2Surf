"""Point-wise velocity decoding for Flow2Surf."""

from jittor import init, nn

from .layers import ChannelLayerNorm


class VelocityDecoder(nn.Module):
    """Decode point features (B, C, N) into xyz velocities (B, N, 3)."""

    def __init__(self, feat_dim: int, hidden_dims=(128, 64)):
        super().__init__()
        layers = []
        in_dim = feat_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Conv1d(in_dim, hidden_dim, 1, bias=False),
                    ChannelLayerNorm(hidden_dim, affine=True),
                    nn.LeakyReLU(scale=0.2),
                ]
            )
            in_dim = hidden_dim
        # Zero-init, no bias: the trunk starts as identity, so the model starts
        # as a zero-velocity identity flow; bias off since the target is zero-mean.
        output = nn.Conv1d(in_dim, 3, 1, bias=False)
        init.zero_(output.weight)
        layers.append(output)
        self.net = nn.Sequential(*layers)

    def execute(self, features):
        return self.net(features).permute(0, 2, 1)
