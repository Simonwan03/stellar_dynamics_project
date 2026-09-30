"""Mask-aware sequence classifiers for Task F."""
from __future__ import annotations

import torch
from torch import nn

from tasks.common.sequence_layers import PositionalEncoding, TCNBlock


class MaskedTCNClassifier(nn.Module):
    """TCN with observed-token pooling.

    When ``append_mask_channel`` is false, all-zero feature vectors are treated
    as hidden positions; an explicit channel is preferable when exact masks are
    available.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden: int = 160,
        levels: int = 5,
        dropout: float = 0.2,
        append_mask_channel: bool = True,
    ) -> None:
        super().__init__()
        self.append_mask_channel = bool(append_mask_channel)
        blocks: list[nn.Module] = []
        input_dim = feature_dim + int(self.append_mask_channel)
        for level in range(levels):
            blocks.append(TCNBlock(input_dim, hidden, dilation=2**level, dropout=dropout))
            input_dim = hidden
        self.backbone = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.append_mask_channel:
            observed = x[..., -1:].clamp(0.0, 1.0)
        else:
            observed = (x.abs().sum(dim=-1, keepdim=True) > 0.0).to(dtype=x.dtype)
        h = self.backbone(x.transpose(1, 2)).transpose(1, 2)
        pooled = (h * observed).sum(dim=1) / observed.sum(dim=1).clamp_min(1.0)
        return self.head(pooled)


class MaskedAttentionClassifier(nn.Module):
    """Transformer that excludes hidden positions from attention.

    Without a channel, all-zero feature vectors are used as the hidden sentinel.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden: int = 128,
        heads: int = 4,
        layers: int = 3,
        dropout: float = 0.2,
        max_len: int = 4096,
        append_mask_channel: bool = True,
    ) -> None:
        super().__init__()
        self.append_mask_channel = bool(append_mask_channel)
        if hidden % heads:
            raise ValueError("hidden must be divisible by heads.")
        self.input_proj = nn.Linear(feature_dim + int(self.append_mask_channel), hidden)
        self.cls = nn.Parameter(torch.zeros(1, 1, hidden))
        self.pos = PositionalEncoding(hidden, max_len=max_len + 1)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=hidden * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.append_mask_channel:
            observed = x[..., -1] > 0.5
        else:
            observed = x.abs().sum(dim=-1) > 0.0
        h = self.input_proj(x)
        cls = self.cls.expand(h.size(0), -1, -1)
        h = torch.cat([cls, h], dim=1)
        padding = torch.cat(
            [torch.zeros((x.size(0), 1), dtype=torch.bool, device=x.device), ~observed],
            dim=1,
        )
        encoded = self.encoder(self.pos(h), src_key_padding_mask=padding)
        return self.head(encoded[:, 0])


def build_model(
    name: str,
    feature_dim: int,
    *,
    dropout: float,
    tcn_hidden: int,
    tcn_levels: int,
    attention_hidden: int,
    attention_heads: int,
    attention_layers: int,
    max_len: int,
    append_mask_channel: bool = True,
) -> nn.Module:
    if name == "tcn":
        return MaskedTCNClassifier(feature_dim, tcn_hidden, tcn_levels, dropout, append_mask_channel)
    if name == "attention":
        return MaskedAttentionClassifier(
            feature_dim, attention_hidden, attention_heads, attention_layers, dropout, max_len, append_mask_channel
        )
    raise ValueError(f"Unknown Task F model {name!r}.")
