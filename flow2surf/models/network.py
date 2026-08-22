"""Flow2Surf model composition."""

import math

import jittor as jt
from jittor import nn

from .encoder import NeighborEncoder
from .decoder import VelocityDecoder
from .transformer import AdaLNTransformerBlock


class AlphaEmbed(nn.Module):
    """Embed continuous alpha values as shared flow conditions.

    Fourier uses an output-width sinusoidal basis and a 2x hidden MLP. Scalar
    maps normalized alpha through a 4x hidden MLP. Both return (B, output_dim).
    """

    def __init__(
        self,
        output_dim: int,
        encoding: str = "fourier",
        max_freq: float = 64.0,
    ):
        super().__init__()
        if output_dim < 1:
            raise ValueError(f"output_dim must be >= 1, got {output_dim}")
        if encoding not in ("fourier", "scalar"):
            raise ValueError(
                f"alpha encoding must be 'fourier' or 'scalar', got {encoding}"
            )

        self.encoding = encoding
        if encoding == "fourier":
            if output_dim < 2:
                raise ValueError(
                    f"output_dim must be >= 2 in Fourier mode, got {output_dim}"
                )
            self.input_dim = output_dim
            half = output_dim // 2
            denominator = max(half - 1, 1)
            self.freqs = tuple(
                math.exp(math.log(max_freq) * index / denominator)
                for index in range(half)
            )
        else:
            self.input_dim = 1
            self.freqs = ()

        hidden_ratio = 2 if encoding == "fourier" else 4
        hidden_dim = output_dim * hidden_ratio
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def execute(self, alpha):
        if self.encoding == "fourier":
            frequencies = jt.array(self.freqs)
            phase = alpha.unsqueeze(1) * frequencies.unsqueeze(0) * (2.0 * math.pi)
            embedding = jt.concat([jt.sin(phase), jt.cos(phase)], dim=1)
            if self.input_dim % 2:
                embedding = jt.concat(
                    [embedding, jt.zeros_like(embedding[:, :1])], dim=1
                )
        else:
            embedding = 2.0 * alpha.unsqueeze(1) - 1.0
        return self.mlp(embedding)


class Flow2SurfNet(nn.Module):
    """
    Predict patch velocities for the clean-to-noisy conditional flow.

    Flow   : X_alpha = X_clean + alpha * (X_noisy - X_clean)
    Inputs : x_alpha (B, 3, M), alpha (B,) in [0, 1]
    Output : velocity (B, M, 3), dX_alpha/dalpha = X_noisy - X_clean

    NeighborEncoder -> alpha-conditioned transformer trunk -> VelocityDecoder

    ffn_mode selects how each block FFN uses its offset-attention residual.
    """

    def __init__(
        self,
        k: int = 32,
        feat_dim: int = 128,
        num_blocks: int = 3,
        encoder_dims=(64, 64),
        encoder_concat: bool = True,
        graph_gate_mult: int = 0,
        alpha_dim: int = 192,
        alpha_encoding: str = "fourier",
        ffn_ratio: int = 4,
        ffn_mode: str = "plain",
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
            concat_all=encoder_concat,
            graph_gate_mult=graph_gate_mult,
        )
        self.decoder = VelocityDecoder(feat_dim, decoder_dims)
        self.alpha_emb = AlphaEmbed(
            alpha_dim,
            encoding=alpha_encoding,
        )
        self.blocks = nn.ModuleList(
            [
                AdaLNTransformerBlock(
                    feat_dim,
                    alpha_dim,
                    ffn_ratio=ffn_ratio,
                    ffn_mode=ffn_mode,
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
        encoder_concat=cfg["encoder_concat"],
        graph_gate_mult=cfg["graph_gate_mult"],
        alpha_dim=cfg["alpha_dim"],
        alpha_encoding=cfg["alpha_encoding"],
        ffn_ratio=cfg["ffn_ratio"],
        ffn_mode=cfg["ffn_mode"],
        decoder_dims=cfg["decoder_dims"],
    )
