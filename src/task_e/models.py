from __future__ import annotations

import math

import torch
from torch import nn

from src.models.transformer_forecaster import TransformerForecaster


class Chomp1d(nn.Module):
    def __init__(self, size: int) -> None:
        super().__init__()
        self.size = int(size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, : -self.size] if self.size else x


class TemporalBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.norm = nn.GroupNorm(1, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.net(x) + self.residual(x))


class TCNForecaster(nn.Module):
    def __init__(
        self,
        num_features: int,
        pred_len: int,
        hidden_dim: int,
        num_layers: int,
        kernel_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        blocks = []
        in_channels = num_features
        for layer in range(num_layers):
            blocks.append(TemporalBlock(in_channels, hidden_dim, kernel_size, 2**layer, dropout))
            in_channels = hidden_dim
        self.network = nn.Sequential(*blocks)
        self.head = nn.Linear(hidden_dim, pred_len * num_features)
        self.pred_len = pred_len
        self.num_features = num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.network(x.transpose(1, 2))[:, :, -1]
        return self.head(encoded).view(x.shape[0], self.pred_len, self.num_features)


class PatchTSTForecaster(nn.Module):
    def __init__(
        self,
        num_features: int,
        context_len: int,
        pred_len: int,
        patch_len: int,
        patch_stride: int,
        d_model: int,
        n_heads: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if patch_len > context_len:
            raise ValueError("patch_len cannot exceed context_len.")
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads.")
        self.num_features = num_features
        self.pred_len = pred_len
        self.patch_len = patch_len
        self.patch_stride = patch_stride
        self.num_patches = 1 + (context_len - patch_len) // patch_stride
        self.patch_embedding = nn.Linear(patch_len, d_model)
        self.position = nn.Parameter(torch.zeros(1, self.num_patches, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Linear(self.num_patches * d_model, pred_len)
        nn.init.normal_(self.position, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, channels = x.shape
        patches = x.transpose(1, 2).unfold(-1, self.patch_len, self.patch_stride)
        tokens = patches.reshape(batch * channels, self.num_patches, self.patch_len)
        tokens = self.patch_embedding(tokens) + self.position
        encoded = self.encoder(tokens).reshape(batch * channels, -1)
        prediction = self.head(encoded).view(batch, channels, self.pred_len)
        return prediction.transpose(1, 2)


class ITransformerSimple(nn.Module):
    def __init__(
        self,
        num_features: int,
        context_len: int,
        pred_len: int,
        d_model: int,
        n_heads: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads.")
        self.embedding = nn.Linear(context_len, d_model)
        self.channel_embedding = nn.Parameter(torch.zeros(1, num_features, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, pred_len)
        nn.init.normal_(self.channel_embedding, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.embedding(x.transpose(1, 2)) + self.channel_embedding
        return self.head(self.encoder(tokens)).transpose(1, 2)


def build_supervised_model(
    model_name: str,
    num_features: int,
    context_len: int,
    pred_len: int,
    hidden_dim: int,
    num_layers: int,
    kernel_size: int,
    dropout: float,
    patch_len: int,
    patch_stride: int,
    d_model: int,
    n_heads: int,
) -> nn.Module:
    if model_name == "tcn":
        return TCNForecaster(num_features, pred_len, hidden_dim, num_layers, kernel_size, dropout)
    if model_name == "patchtst":
        return PatchTSTForecaster(
            num_features, context_len, pred_len, patch_len, patch_stride,
            d_model, n_heads, num_layers, dropout,
        )
    if model_name in {"itransformer", "itransformer_simple"}:
        return ITransformerSimple(
            num_features, context_len, pred_len, d_model, n_heads, num_layers, dropout
        )
    if model_name == "transformer":
        return TransformerForecaster(
            num_features=num_features,
            context_len=context_len,
            pred_len=pred_len,
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_layers,
            dim_feedforward=4 * d_model,
            dropout=dropout,
        )
    raise ValueError(f"Unsupported supervised model: {model_name}")
