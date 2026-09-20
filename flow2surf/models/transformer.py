"""Alpha-conditioned transformer blocks."""

import jittor as jt
from jittor import init, nn

from .layers import ChannelLayerNorm
from .ops import offset_attention_aggregate


def modulate(x, shift, scale):
    """Apply AdaLN modulation to (B, C, N) features."""
    return x * (1.0 + scale) + shift


class AdaLNTransformerBlock(nn.Module):
    """Apply alpha-conditioned offset attention followed by a SwiGLU FFN.

    Both residual branches use zero-initialized AdaLN gates. Attention packs
    Q/K/V into one projection and uses a compact C // 4 query/key space with
    query-column softmax followed by key-axis normalization.

    Attention forms R = H - V A^T. The pointwise FFN is a single SwiGLU stage.
    """

    def __init__(
        self,
        channels: int,
        cond_dim: int,
        ffn_ratio: int = 4,
    ):
        super().__init__()
        qk_dim = channels // 4
        if qk_dim < 1:
            raise ValueError(f"channels must be >= 4, got {channels}")
        self.qk_dim = qk_dim
        self.norm1 = ChannelLayerNorm()
        self.qkv_conv = nn.Conv1d(channels, channels + 2 * qk_dim, 1, bias=False)
        qk_weight = self.qkv_conv.weight[: 2 * qk_dim]
        value_weight = init.eye(channels).reshape(channels, channels, 1)
        self.qkv_conv.weight.assign(jt.concat([qk_weight, value_weight], dim=0))
        self.proj = nn.Conv1d(channels, channels, 1, bias=False)
        self.qk_norm = ChannelLayerNorm()

        hidden = channels * ffn_ratio
        self.norm2 = ChannelLayerNorm()
        self.ff_gate = nn.Conv1d(channels, hidden, 1)
        self.ff1 = nn.Conv1d(channels, hidden, 1)
        self.ff2 = nn.Conv1d(hidden, channels, 1)
        self.ff_act = nn.SiLU()

        modulation = nn.Linear(cond_dim, channels * 6)
        init.zero_(modulation.weight)
        if modulation.bias is not None:
            init.zero_(modulation.bias)
        self.mod = nn.Sequential(nn.SiLU(), modulation)

    def _offset(self, features):
        """Return the offset-attention residual R = H - V A^T, shaped (B, C, N)."""
        qkv = self.qkv_conv(features)
        batch, _, points = qkv.shape
        # Normalize Q and K independently in one batched operation.
        qk = qkv[:, : 2 * self.qk_dim].reshape(batch * 2, self.qk_dim, points)
        qk = self.qk_norm(qk).reshape(batch, 2, self.qk_dim, points)
        query = qk[:, 0].permute(0, 2, 1)
        key = qk[:, 1]
        value = qkv[:, 2 * self.qk_dim :].permute(0, 2, 1)
        aggregate = offset_attention_aggregate(query, key, value).permute(0, 2, 1)
        return features - aggregate

    def _swiglu(self, features):
        gate = self.ff_gate(features)
        value = self.ff1(features)
        return self.ff2(self.ff_act(gate) * value)

    def execute(self, x, alpha_cond):
        channels = x.shape[1]
        modulation = self.mod(alpha_cond)
        scale1 = modulation[:, 0 * channels : 1 * channels].unsqueeze(2)
        shift1 = modulation[:, 1 * channels : 2 * channels].unsqueeze(2)
        gate1 = modulation[:, 2 * channels : 3 * channels].unsqueeze(2)
        scale2 = modulation[:, 3 * channels : 4 * channels].unsqueeze(2)
        shift2 = modulation[:, 4 * channels : 5 * channels].unsqueeze(2)
        gate2 = modulation[:, 5 * channels : 6 * channels].unsqueeze(2)

        hidden = modulate(self.norm1(x), shift1, scale1)
        offset = self._offset(hidden)
        x = x + gate1 * self.proj(offset)
        hidden = modulate(self.norm2(x), shift2, scale2)
        return x + gate2 * self._swiglu(hidden)
