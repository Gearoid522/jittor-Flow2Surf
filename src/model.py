import math
import jittor as jt
from jittor import init, nn
from .embedding import NeighborEmbedding


class ChannelLayerNorm(nn.Module):
    """
    Parameter-free LayerNorm over channels for tensors shaped (B, C, N).

    Carries no learnable affine; the AdaLN-Zero modulation supplies the
    per-block scale and shift.
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def execute(self, x):
        mean = x.mean(dim=1, keepdims=True)
        var = ((x - mean) ** 2).mean(dim=1, keepdims=True)
        return (x - mean) / jt.sqrt(var + self.eps)


def modulate(x, shift, scale):
    """AdaLN modulation for (B, C, N) features; shift/scale are (B, C, 1)."""
    return x * (1.0 + scale) + shift


class AdaLNBlock(nn.Module):
    """
    AdaLN-Zero block over point features (B, C, N): two AdaLN-Zero modulated residual
    sublayers - offset/standard attention, then a gated FFN.

        x = x + gate1 * attn(modulate(norm1(x), shift1, scale1))
        x = x + gate2 * ffn (modulate(norm2(x), shift2, scale2))

    A SiLU + zero-init Linear maps the time embedding to all six
    scale/shift/gate vectors, so both gates are identity at init.

    Knobs
    -----
    attn_type    : "offset" -> core = h - agg (PCT/Laplacian) | "standard" -> core = agg
    attn_heads   : number of attention heads. 1 keeps the PCT reduced-dim
                   single head and exactly reproduces the baseline.
    attn_softmax : "pct"    -> softmax over queries + L1 renorm, no 1/sqrt(d) (baseline)
                   "scaled" -> standard 1/sqrt(d_head) scaled softmax over keys
    ffn_ratio    : FFN hidden expansion factor.
    ffn_type     : gated GLU-family FFN.
                   "geglu" -> GELU(gate) * value -> Conv1d;
                   "swiglu" -> SiLU(gate) * value -> Conv1d.
    """

    def __init__(
        self,
        channels: int,
        t_dim: int,
        attn_type: str = "offset",
        attn_heads: int = 1,
        attn_softmax: str = "pct",
        ffn_ratio: int = 4,
        ffn_type: str = "swiglu",
    ):
        super().__init__()
        if attn_type not in ("offset", "standard"):
            raise ValueError(
                f"attn_type must be 'offset' or 'standard', got {attn_type}"
            )
        if attn_softmax not in ("pct", "scaled"):
            raise ValueError(
                f"attn_softmax must be 'pct' or 'scaled', got {attn_softmax}"
            )
        if ffn_type not in ("geglu", "swiglu"):
            raise ValueError(f"ffn_type must be 'geglu' or 'swiglu', got {ffn_type}")
        self.attn_type = attn_type
        self.attn_softmax = attn_softmax
        self.heads = attn_heads

        # qk_dim: keep the PCT reduced dim for the single-head baseline; use full
        # width split across heads for true multi-head attention.
        qk_dim = channels // 4 if attn_heads == 1 else channels
        if qk_dim % attn_heads or channels % attn_heads:
            raise ValueError(
                f"channels={channels}/qk_dim={qk_dim} not divisible by heads={attn_heads}"
            )
        self.q_head = qk_dim // attn_heads
        self.v_head = channels // attn_heads

        # attention sublayer
        self.norm1 = ChannelLayerNorm()
        self.q_conv = nn.Conv1d(channels, qk_dim, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, qk_dim, 1, bias=False)
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.proj = nn.Conv1d(channels, channels, 1)
        self.qk_norm = ChannelLayerNorm()  # QK-LayerNorm: bounds q*k logits

        # gated FFN sublayer
        hidden = channels * ffn_ratio
        self.norm2 = ChannelLayerNorm()
        self.ff_gate = nn.Conv1d(channels, hidden, 1)
        self.ff1 = nn.Conv1d(channels, hidden, 1)
        self.ff2 = nn.Conv1d(hidden, channels, 1)
        self.ff_act = nn.SiLU() if ffn_type == "swiglu" else nn.GELU()

        # shared AdaLN-Zero modulation: SiLU then a 6-chunk projection
        # (scale/shift/gate per sublayer). Linear zero-init -> identity at start.
        mod_lin = nn.Linear(t_dim, channels * 6)
        init.zero_(mod_lin.weight)
        if mod_lin.bias is not None:
            init.zero_(mod_lin.bias)
        self.mod = nn.Sequential(nn.SiLU(), mod_lin)

    def _attn(self, h):
        """Attention sublayer transform on already-modulated h -> (B, C, N)."""
        B, C, N = h.shape
        H = self.heads
        # (B, *, N) -> (B*H, N, head) for q/v, (B*H, head, N) for k
        q = (
            self.qk_norm(self.q_conv(h))
            .reshape(B, H, self.q_head, N)
            .permute(0, 1, 3, 2)
            .reshape(B * H, N, self.q_head)
        )
        k = (
            self.qk_norm(self.k_conv(h))
            .reshape(B, H, self.q_head, N)
            .reshape(B * H, self.q_head, N)
        )
        v = (
            self.v_conv(h)
            .reshape(B, H, self.v_head, N)
            .permute(0, 1, 3, 2)
            .reshape(B * H, N, self.v_head)
        )

        scores = jt.nn.bmm(q, k)  # (B*H, N_q, N_k)
        if self.attn_softmax == "scaled":
            attn = nn.softmax(scores / math.sqrt(self.q_head), dim=-1)
        else:
            attn = nn.softmax(scores, dim=1)
            attn = attn / (attn.sum(dim=-1, keepdims=True) + 1e-9)

        agg = jt.nn.bmm(attn, v)  # (B*H, N, v_head)
        agg = agg.reshape(B, H, N, self.v_head).permute(0, 1, 3, 2).reshape(B, C, N)
        core = h - agg if self.attn_type == "offset" else agg
        return self.proj(core)  # linear sublayer output (standard residual)

    def _ffn(self, h):
        """Gated GLU-family FFN sublayer on already-modulated h -> (B, C, N)."""
        return self.ff2(self.ff_act(self.ff_gate(h)) * self.ff1(h))

    def execute(self, x, te):
        C = x.shape[1]
        m = self.mod(te)
        scale1 = m[:, 0 * C : 1 * C].unsqueeze(2)
        shift1 = m[:, 1 * C : 2 * C].unsqueeze(2)
        gate1 = m[:, 2 * C : 3 * C].unsqueeze(2)
        scale2 = m[:, 3 * C : 4 * C].unsqueeze(2)
        shift2 = m[:, 4 * C : 5 * C].unsqueeze(2)
        gate2 = m[:, 5 * C : 6 * C].unsqueeze(2)

        x = x + gate1 * self._attn(modulate(self.norm1(x), shift1, scale1))
        x = x + gate2 * self._ffn(modulate(self.norm2(x), shift2, scale2))
        return x


class SinusoidalTimeEmbed(nn.Module):
    """Discrete step index -> dense (B, dim) via log-spaced sinusoids + MLP."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim),
        )

    def execute(self, t):
        half = self.dim // 2
        denom = max(half - 1, 1)
        freqs = jt.exp(-math.log(10000) * jt.arange(half).float32() / denom)
        args = t.unsqueeze(1) * freqs.unsqueeze(0)
        emb = jt.concat([jt.sin(args), jt.cos(args)], dim=1)
        if self.dim % 2:
            emb = jt.concat([emb, jt.zeros_like(emb[:, :1])], dim=1)
        return self.mlp(emb)


# -- Decoder --------------------------------------------------------------------


class Decoder(nn.Module):
    """Shared MLP: (B, C, N) -> (B, N, 3)"""

    def __init__(self, feat_dim: int, hidden_dims=(128, 64)):
        super().__init__()
        hidden_dims = list(hidden_dims)
        layers = []
        in_dim = feat_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Conv1d(in_dim, hidden_dim, 1, bias=False),
                    nn.BatchNorm1d(hidden_dim),
                    nn.LeakyReLU(scale=0.2),
                ]
            )
            in_dim = hidden_dim
        layers.append(nn.Conv1d(in_dim, 3, 1))
        self.net = nn.Sequential(*layers)

    def execute(self, x):
        return self.net(x).permute(0, 2, 1)


# -- Flow2SurfNet ---------------------------------------------------------------


class Flow2SurfNet(nn.Module):
    """
    Patch-based point cloud denoiser.

    Inputs  : pc_mix (B, 3, M)  patch at noise level alpha
              t      (B,)       discrete step index
    Output  : disp   (B, M, 3)  predicted residual  X_noisy - X_clean

    Architecture: NeighborEmbedding -> time-conditioned attention x num_attn -> Decoder
    """

    def __init__(
        self,
        k: int = 32,
        feat_dim: int = 128,
        num_attn: int = 3,
        emb_dims=(64, 64),
        emb_concat: bool = True,
        emb_fusion: str = "concat",
        decoder_dims=(128, 64),
        attn_type: str = "offset",
        attn_heads: int = 1,
        attn_softmax: str = "pct",
        ffn_ratio: int = 4,
        ffn_type: str = "swiglu",
    ):
        super().__init__()
        self.local_enc = NeighborEmbedding(
            k=k,
            out_dim=feat_dim,
            dims=emb_dims,
            concat_all=emb_concat,
            fusion=emb_fusion,
        )
        self.t_emb = SinusoidalTimeEmbed(feat_dim)
        self.blocks = nn.ModuleList(
            [
                AdaLNBlock(
                    feat_dim,
                    feat_dim,
                    attn_type=attn_type,
                    attn_heads=attn_heads,
                    attn_softmax=attn_softmax,
                    ffn_ratio=ffn_ratio,
                    ffn_type=ffn_type,
                )
                for _ in range(num_attn)
            ]
        )
        self.decoder = Decoder(feat_dim, decoder_dims)

    def execute(self, pc_mix, t):
        te = self.t_emb(t)  # (B, C)
        x = self.local_enc(pc_mix)  # (B, C, M)

        for block in self.blocks:
            x = block(x, te)

        return self.decoder(x)  # (B, M, 3)
