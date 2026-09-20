from __future__ import annotations

import torch
from torch import nn

from tasks.task_e.transformer import TransformerForecaster


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
        output_features: int | None = None,
    ) -> None:
        super().__init__()
        output_features = num_features if output_features is None else int(output_features)
        blocks = []
        in_channels = num_features
        for layer in range(num_layers):
            blocks.append(TemporalBlock(in_channels, hidden_dim, kernel_size, 2**layer, dropout))
            in_channels = hidden_dim
        self.network = nn.Sequential(*blocks)
        self.head = nn.Linear(hidden_dim, pred_len * output_features)
        self.pred_len = pred_len
        self.num_features = num_features
        self.output_features = output_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.network(x.transpose(1, 2))[:, :, -1]
        return self.head(encoded).view(x.shape[0], self.pred_len, self.output_features)


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
        output_features: int | None = None,
    ) -> None:
        super().__init__()
        if patch_len > context_len:
            raise ValueError("patch_len cannot exceed context_len.")
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads.")
        self.num_features = num_features
        self.output_features = num_features if output_features is None else int(output_features)
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
        return prediction[:, : self.output_features, :].transpose(1, 2)


class ITransformerForecaster(nn.Module):
    """Task E adapter for the official iTransformer forecasting architecture.

    The implementation follows the official variates-as-tokens path:
    per-sample temporal normalization, inverted value embedding, Transformer
    encoding across variates, direct projection to the forecast horizon, and
    de-normalization.  Task E does not provide timestamp covariates, so the
    optional official ``x_mark`` token path is deliberately omitted.

    Reference: https://github.com/thuml/iTransformer/blob/main/model/iTransformer.py
    """

    def __init__(
        self,
        num_features: int,
        context_len: int,
        pred_len: int,
        d_model: int,
        n_heads: int,
        num_layers: int,
        dropout: float,
        output_features: int | None = None,
    ) -> None:
        super().__init__()
        output_features = num_features if output_features is None else int(output_features)
        if output_features <= 0 or output_features > num_features:
            raise ValueError("output_features must be in [1, num_features].")
        if num_features <= 0 or context_len <= 0 or pred_len <= 0:
            raise ValueError("num_features, context_len, and pred_len must be positive.")
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads.")
        self.num_features = int(num_features)
        self.output_features = int(output_features)
        self.context_len = int(context_len)
        self.pred_len = int(pred_len)
        self.normalization_epsilon = 1e-5

        # Official DataEmbedding_inverted without optional timestamp covariates.
        self.embedding = nn.Linear(context_len, d_model)
        self.embedding_dropout = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )
        self.projector = nn.Linear(d_model, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"iTransformer expects [batch,time,channels], got {tuple(x.shape)}.")
        if x.shape[1:] != (self.context_len, self.num_features):
            raise ValueError(
                "iTransformer input has shape "
                f"{tuple(x.shape)}; expected [batch,{self.context_len},{self.num_features}]."
            )

        # Non-stationary normalization used by the official implementation.
        means = x.mean(dim=1, keepdim=True).detach()
        centered = x - means
        stdev = torch.sqrt(
            torch.var(centered, dim=1, keepdim=True, unbiased=False)
            + self.normalization_epsilon
        )
        normalized = centered / stdev

        # [B,L,N] -> [B,N,E]: each physical channel is one variate token.
        tokens = self.embedding_dropout(self.embedding(normalized.transpose(1, 2)))
        encoded = self.encoder(tokens)
        forecast = self.projector(encoded).transpose(1, 2)

        # Return to the same coordinates as the Task E input/scaler.
        return (forecast * stdev + means)[..., : self.output_features]


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
    input_features: int | None = None,
    output_features: int | None = None,
) -> nn.Module:
    input_features = num_features if input_features is None else int(input_features)
    output_features = num_features if output_features is None else int(output_features)
    if input_features <= 0 or output_features <= 0 or output_features > input_features:
        raise ValueError("Expected 1 <= output_features <= input_features.")
    if model_name == "tcn":
        return TCNForecaster(input_features, pred_len, hidden_dim, num_layers, kernel_size, dropout, output_features=output_features)
    if model_name == "patchtst":
        return PatchTSTForecaster(
            input_features, context_len, pred_len, patch_len, patch_stride,
            d_model, n_heads, num_layers, dropout, output_features=output_features,
        )
    if model_name == "itransformer":
        return ITransformerForecaster(
            input_features, context_len, pred_len, d_model, n_heads, num_layers, dropout,
            output_features=output_features
        )
    if model_name == "transformer":
        return TransformerForecaster(
            num_features=input_features,
            context_len=context_len,
            pred_len=pred_len,
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_layers,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            output_features=output_features,
        )
    raise ValueError(f"Unsupported supervised model: {model_name}")
