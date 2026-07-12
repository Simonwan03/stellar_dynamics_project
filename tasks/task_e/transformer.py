from __future__ import annotations

import math

import torch
from torch import nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 8192) -> None:
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
            raise ValueError(
                f"Sequence length {seq_len} exceeds positional encoding capacity {self.pe.size(1)}."
            )
        return x + self.pe[:, :seq_len]


class TransformerForecaster(nn.Module):
    def __init__(
        self,
        num_features: int,
        context_len: int,
        pred_len: int,
        d_model: int = 256,
        n_heads: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model must be divisible by n_heads, got {d_model=} {n_heads=}")

        self.num_features = int(num_features)
        self.context_len = int(context_len)
        self.pred_len = int(pred_len)
        self.d_model = int(d_model)

        self.input_projection = nn.Linear(num_features, d_model)
        self.positional_encoding = SinusoidalPositionalEncoding(d_model=d_model, max_len=context_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.forecast_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, pred_len * num_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected input shape [B, context_len, num_features], got {tuple(x.shape)}")
        if x.size(1) != self.context_len:
            raise ValueError(f"Expected context length {self.context_len}, got {x.size(1)}")
        if x.size(2) != self.num_features:
            raise ValueError(f"Expected feature dimension {self.num_features}, got {x.size(2)}")

        encoded = self.input_projection(x)
        encoded = self.positional_encoding(encoded)
        encoded = self.dropout(encoded)
        encoded = self.encoder(encoded)
        final_state = encoded[:, -1, :]
        forecast = self.forecast_head(final_state)
        return forecast.view(x.size(0), self.pred_len, self.num_features)
