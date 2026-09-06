"""Mask-aware sequence classifiers for Task F."""
from __future__ import annotations

import torch
from torch import nn

from tasks.task_b.main import PositionalEncoding, TCNBlock


class MaskedTCNClassifier(nn.Module):
    """TCN with an explicit observation channel and observed-token pooling."""

    def __init__(
        self,
        feature_dim: int,
        hidden: int = 160,
        levels: int = 5,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        input_dim = feature_dim + 1
        for level in range(levels):
            blocks.append(TCNBlock(input_dim, hidden, dilation=2**level, dropout=dropout))
            input_dim = hidden
        self.backbone = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        observed = x[..., -1:].clamp(0.0, 1.0)
        h = self.backbone(x.transpose(1, 2)).transpose(1, 2)
        pooled = (h * observed).sum(dim=1) / observed.sum(dim=1).clamp_min(1.0)
        return self.head(pooled)


class MaskedAttentionClassifier(nn.Module):
    """Transformer that excludes randomly hidden positions from attention."""

    def __init__(
        self,
        feature_dim: int,
        hidden: int = 128,
        heads: int = 4,
        layers: int = 3,
        dropout: float = 0.2,
        max_len: int = 4096,
    ) -> None:
        super().__init__()
        if hidden % heads:
            raise ValueError("hidden must be divisible by heads.")
        self.input_proj = nn.Linear(feature_dim + 1, hidden)
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
        observed = x[..., -1] > 0.5
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
) -> nn.Module:
    if name == "tcn":
        return MaskedTCNClassifier(feature_dim, tcn_hidden, tcn_levels, dropout)
    if name == "attention":
        return MaskedAttentionClassifier(
            feature_dim, attention_hidden, attention_heads, attention_layers, dropout, max_len
        )
    raise ValueError(f"Unknown Task F model {name!r}.")
