"""Flow2Surf model composition."""

from jittor import nn

from .encoder import NeighborEncoder
from .decoder import VelocityDecoder
from .transformer import AdaLNTransformerBlock


class AlphaEmbed(nn.Module):
    """Map normalized continuous alpha to shared flow conditions."""

    def __init__(self, output_dim: int):
        super().__init__()
        if output_dim < 1:
            raise ValueError(f"output_dim must be >= 1, got {output_dim}")
        hidden_dim = output_dim * 4
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def execute(self, alpha):
        embedding = 2.0 * alpha.unsqueeze(1) - 1.0
        return self.mlp(embedding)


class Flow2SurfNet(nn.Module):
    """
    Predict patch velocities for the clean-to-noisy conditional flow.

    Flow   : X_alpha = X_clean + alpha * (X_noisy - X_clean)
    Inputs : x_alpha (B, 3, M), alpha (B,) in [0, 1]
    Output : velocity (B, M, 3), dX_alpha/dalpha = X_noisy - X_clean

    NeighborEncoder -> alpha-conditioned transformer trunk -> VelocityDecoder
    """

    def __init__(
        self,
        k: int = 32,
        feat_dim: int = 128,
        num_blocks: int = 3,
        encoder_dims=(64, 64),
        alpha_dim: int = 192,
        ffn_ratio: int = 4,
        decoder_dims=(128, 64),
    ):
        super().__init__()
        if num_blocks < 0:
            raise ValueError(f"num_blocks must be >= 0, got {num_blocks}")
        if ffn_ratio < 1:
            raise ValueError(f"ffn_ratio must be >= 1, got {ffn_ratio}")
        if alpha_dim < 1:
            raise ValueError(f"alpha_dim must be >= 1, got {alpha_dim}")
        self.local_encoder = NeighborEncoder(
            k=k,
            out_dim=feat_dim,
            dims=encoder_dims,
        )
        self.decoder = VelocityDecoder(feat_dim, decoder_dims)
        self.alpha_emb = AlphaEmbed(alpha_dim)
        self.blocks = nn.ModuleList(
            [
                AdaLNTransformerBlock(
                    feat_dim,
                    alpha_dim,
                    ffn_ratio=ffn_ratio,
                )
                for _ in range(num_blocks)
            ]
        )

    def execute(self, x_alpha, alpha):
        alpha_cond = self.alpha_emb(alpha)  # (B, C)
        x = self.local_encoder(x_alpha)

        for block in self.blocks:
            x = block(x, alpha_cond)

        return self.decoder(x)  # (B, M, 3)


def build_model(cfg):
    """Construct Flow2SurfNet from the current config schema."""
    return Flow2SurfNet(
        k=cfg["k_neighbors"],
        feat_dim=cfg["feat_dim"],
        num_blocks=cfg["num_blocks"],
        encoder_dims=cfg["encoder_dims"],
        alpha_dim=cfg["alpha_dim"],
        ffn_ratio=cfg["ffn_ratio"],
        decoder_dims=cfg["decoder_dims"],
    )
