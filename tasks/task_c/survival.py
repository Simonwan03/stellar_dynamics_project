"""Discrete-time survival modelling for Task C.

This module is the event-time/censoring version of Task C.  The original
``tasks.task_c.main`` entry point is intentionally left backward compatible:
it trains on the historical seven survival fractions.  This entry point uses
the same case split and prefix loading code, but builds proper interval
hazard targets from ``time_unbound`` and trains an independent variation
encoder whose system curve is the mean of the 32 variation curves.

Run with::

    python -m tasks.task_c.survival --prefix-cache results/task_c_ablation/_cache \
        --prefix-len 100 --models baseline,stats,cnn --epochs 40

For a precomputed frozen foundation-model cache, use
``--embedding-cache ... --models baseline,embedding_direct,embedding_attention``.
The cache is
checked against its aligned labels before complete 32-variation systems are
formed.

The implementation is deliberately small and composable.  In particular,
``build_discrete_hazard_targets`` and ``MaskedHazardNLL`` can be reused by
the later Panda/Chronos and LoRA experiments without changing their target
semantics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tasks.common.experiment_utils import (  # noqa: E402
    ensure_dir,
    parse_name_list,
    save_csv,
    save_json,
    select_device,
    set_random_seed,
)
from tasks.common.case_split import (  # noqa: E402
    case_split_indices,
    default_case_split_manifest_path,
    ensure_case_split_manifest,
    missing_case_ids,
)
from tasks.task_b.main import (  # noqa: E402
    FULL_RESULT_KEYS,
    ORBITAL15_RESULT_KEYS,
    build_prefix_features,
    discover_result_paths,
    filter_valid_samples,
    load_aligned_labels,
    load_result_prefixes,
)
from tasks.task_c.cache import load_cached_prefixes  # noqa: E402
from tasks.task_c.main import (  # noqa: E402
    ConvTrajectoryEncoder,
    GRUTrajectoryEncoder,
    TemporalAttentionEncoder,
    build_variation_subset_indices,
    standardize_case_train,
)


def make_log_bin_edges(t_min: float, t_max: float, num_bins: int = 20) -> np.ndarray:
    """Return ``num_bins + 1`` strictly increasing logarithmic edges."""

    t_min = float(t_min)
    t_max = float(t_max)
    num_bins = int(num_bins)
    if not np.isfinite(t_min) or not np.isfinite(t_max):
        raise ValueError("Hazard bin limits must be finite.")
    if t_min <= 0.0 or t_max <= t_min:
        raise ValueError(f"Require 0 < t_min < t_max, got {t_min:g}, {t_max:g}.")
    if num_bins < 1:
        raise ValueError(f"num_bins must be positive, got {num_bins}.")
    edges = np.geomspace(t_min, t_max, num_bins + 1).astype(np.float32)
    if not np.all(np.diff(edges) > 0.0):
        raise ValueError("Log-spaced hazard edges are not strictly increasing.")
    return edges


def build_discrete_hazard_targets(
    instability_times: np.ndarray,
    bin_edges: np.ndarray,
    censored: np.ndarray | None = None,
    *,
    return_metadata: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Build event targets and at-risk masks for discrete-time hazards.

    Intervals use ``(t[k], t[k+1]]``.  An observed event contributes zeros
    before its interval, one in its interval, and is masked after it.  A
    right-censored sample contributes target zero and at-risk one in every
    interval.  Events at or before the left edge are represented as an event
    in the first interval; this is a useful, explicit left-boundary policy
    when a prefix starts after the earliest event in a dataset.

    By default values ``>= bin_edges[-1]`` are treated as right-censored.  In
    this dataset that includes the sentinel ``time_unbound == 10000``.
    ``censored`` may be supplied to override the censoring flag, but the
    right boundary, negative values, and non-finite values remain censored for
    safety.
    """

    times = np.asarray(instability_times, dtype=np.float64)
    edges = np.asarray(bin_edges, dtype=np.float64).reshape(-1)
    if edges.size < 2 or not np.all(np.diff(edges) > 0.0):
        raise ValueError("bin_edges must contain at least two strictly increasing values.")
    if not np.all(np.isfinite(edges)):
        raise ValueError("bin_edges must be finite.")

    horizon = edges.size - 1
    if censored is None:
        censored_arr = np.zeros(times.shape, dtype=bool)
    else:
        censored_arr = np.asarray(censored, dtype=bool)
        if censored_arr.shape != times.shape:
            censored_arr = np.broadcast_to(censored_arr, times.shape).copy()
    finite = np.isfinite(times)
    nonnegative = times >= 0.0
    censored_arr = censored_arr | ~finite | ~nonnegative | (times >= edges[-1])
    observed_event = finite & ~censored_arr

    # ``searchsorted(..., side='left') - 1`` maps an event at an upper edge
    # to the interval ending at that edge.  Values below/equal to t_min are
    # explicit left-boundary events in bin zero.
    event_bin = np.searchsorted(edges, np.where(finite, times, edges[-1]), side="left") - 1
    event_bin = np.where(times <= edges[0], 0, event_bin)
    event_bin = np.clip(event_bin, 0, horizon - 1).astype(np.int64)
    columns = np.arange(horizon, dtype=np.int64)
    at_risk = (observed_event[..., None] & (columns <= event_bin[..., None])) | censored_arr[..., None]
    target = observed_event[..., None] & (columns == event_bin[..., None])
    target = target.astype(np.float32)
    at_risk = at_risk.astype(np.float32)

    if return_metadata:
        return target, at_risk, {
            "event_bin": event_bin,
            "observed_event": observed_event,
            "censored": censored_arr,
        }
    return target, at_risk


def survival_targets_from_times(instability_times: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    """Return empirical survival indicators at the upper edge of every bin."""

    times = np.asarray(instability_times, dtype=np.float32)
    upper_edges = np.asarray(bin_edges, dtype=np.float32)[1:]
    return (times[..., None] >= upper_edges).astype(np.float32)


def hazard_logits_to_survival(logits: torch.Tensor) -> torch.Tensor:
    """Convert conditional hazard logits to a monotone survival curve."""

    hazard = torch.sigmoid(logits)
    return torch.cumprod(1.0 - hazard, dim=-1)


def masked_hazard_bce(
    logits: torch.Tensor,
    target: torch.Tensor,
    at_risk: torch.Tensor,
    interval_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Masked BCE over only intervals in which a sample is at risk."""

    if logits.shape != target.shape or logits.shape != at_risk.shape:
        raise ValueError(
            f"Hazard logits/target/mask must have identical shapes, got "
            f"{tuple(logits.shape)}, {tuple(target.shape)}, {tuple(at_risk.shape)}."
        )
    loss = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
    weights = at_risk
    if interval_weights is not None:
        weights = weights * interval_weights.to(device=weights.device, dtype=weights.dtype)
    denominator = weights.sum().clamp_min(1.0)
    return (loss * weights).sum() / denominator


class MaskedHazardNLL(nn.Module):
    """Event-time hazard NLL with an explicit at-risk mask."""

    def __init__(self, interval_weights: torch.Tensor | None = None):
        super().__init__()
        if interval_weights is not None:
            self.register_buffer("interval_weights", interval_weights.float().reshape(-1))
        else:
            self.interval_weights = None

    def forward(self, logits: torch.Tensor, target: torch.Tensor, at_risk: torch.Tensor) -> torch.Tensor:
        weights = self.interval_weights
        if weights is not None and weights.numel() != logits.shape[-1]:
            raise ValueError("interval_weights length must match the hazard horizon.")
        return masked_hazard_bce(logits, target, at_risk, weights)


def interpolate_survival(
    survival: np.ndarray,
    bin_edges: np.ndarray,
    query_times: np.ndarray,
) -> np.ndarray:
    """Query a bin survival curve, linearly interpolated in log time.

    The value at ``t_min`` is anchored at one and the values at the remaining
    grid points are the cumulative survival values.  This makes the legacy
    seven-horizon comparison well-defined even when those horizons do not
    coincide with a 20-bin edge.
    """

    values = np.asarray(survival, dtype=np.float64)
    edges = np.asarray(bin_edges, dtype=np.float64).reshape(-1)
    queries = np.asarray(query_times, dtype=np.float64).reshape(-1)
    if values.shape[-1] != edges.size - 1:
        raise ValueError("The last survival dimension must be len(bin_edges)-1.")
    if np.any(queries <= 0.0):
        raise ValueError("query_times must be positive for log-time interpolation.")
    x = np.log(edges)
    qx = np.log(queries)
    anchors = np.concatenate(
        [np.ones(values.shape[:-1] + (1,), dtype=np.float64), values.astype(np.float64)], axis=-1
    )
    flat = anchors.reshape(-1, anchors.shape[-1])
    out = np.stack([np.interp(qx, x, row) for row in flat], axis=0)
    return out.reshape(values.shape[:-1] + (queries.size,)).astype(np.float32)


def task_b_spock_summary(x: torch.Tensor) -> torch.Tensor:
    """Return the eight Task-B SPoCK-inspired summaries per channel.

    The output uses the same statistic order and channel-major flattening as
    ``train_spock_inspired_task_b_xgboost.py``:
    ``initial, final, mean, std, slope, min, max, delta``.
    """

    if x.ndim != 3:
        raise ValueError(f"Expected [batch,time,channels], got {tuple(x.shape)}.")
    if x.size(1) <= 1:
        raise ValueError("The prefix must contain at least two time steps.")
    mean = x.mean(dim=1)
    std = x.std(dim=1, unbiased=False)
    minimum = x.amin(dim=1)
    maximum = x.amax(dim=1)
    initial = x[:, 0]
    final = x[:, -1]
    time = torch.linspace(0.0, 1.0, x.size(1), device=x.device, dtype=x.dtype)
    centered = time - time.mean()
    denom = centered.square().mean().clamp_min(torch.finfo(x.dtype).eps)
    slope = ((x - mean[:, None, :]) * centered[None, :, None]).mean(dim=1) / denom
    delta = final - initial
    # [batch, channel, statistic] -> channel-major flattening, matching
    # train_spock_inspired_task_b_xgboost.py exactly.
    return torch.stack(
        [initial, final, mean, std, slope, minimum, maximum, delta], dim=-1
    ).reshape(x.size(0), -1)


class StatsTrajectoryEncoder(nn.Module):
    """Per-variation summary encoder used as the physics/statistics baseline."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        # Keep the same eight summaries and ordering as the Task B
        # SPoCK-inspired baseline: initial, final, mean, std, slope, min, max,
        # and delta.  The input trajectories are still train-only
        # standardised by the Task C data pipeline before reaching this
        # encoder; consequently the feature definitions/order match Task B,
        # while their numerical scale follows the survival experiment.
        summary_dim = 8 * input_dim
        bottleneck = max(hidden_dim, 32)
        self.net = nn.Sequential(
            nn.LayerNorm(summary_dim),
            nn.Linear(summary_dim, bottleneck),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck, hidden_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        summary = task_b_spock_summary(x)
        return self.net(summary)


class TCNBlock(nn.Module):
    """A small residual temporal-convolution block with same-length padding."""

    def __init__(self, input_dim: int, output_dim: int, dilation: int, dropout: float):
        super().__init__()
        padding = 2 * dilation
        self.conv1 = nn.Conv1d(input_dim, output_dim, kernel_size=5, dilation=dilation, padding=padding)
        self.conv2 = nn.Conv1d(output_dim, output_dim, kernel_size=5, dilation=dilation, padding=padding)
        self.norm1 = nn.BatchNorm1d(output_dim)
        self.norm2 = nn.BatchNorm1d(output_dim)
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Conv1d(input_dim, output_dim, kernel_size=1) if input_dim != output_dim else nn.Identity()

    @staticmethod
    def _trim(x: torch.Tensor, length: int) -> torch.Tensor:
        # Conv1d with symmetric dilation padding can be one or more samples
        # longer than the input; trim symmetrically to keep residual addition
        # and the temporal receptive field deterministic.
        if x.size(-1) == length:
            return x
        start = max((x.size(-1) - length) // 2, 0)
        return x[..., start : start + length]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        length = residual.size(-1)
        h = self._trim(self.conv1(x), length)
        h = self.dropout(nn.functional.gelu(self.norm1(h)))
        h = self._trim(self.conv2(h), length)
        h = self.dropout(nn.functional.gelu(self.norm2(h)))
        return nn.functional.gelu(h + residual)


class TCNTrajectoryEncoder(nn.Module):
    """Dilated TCN baseline for the event-time head."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        dilations = (1, 2, 4)
        blocks: list[nn.Module] = []
        channels = input_dim
        for dilation in dilations:
            blocks.append(TCNBlock(channels, hidden_dim, dilation, dropout))
            channels = hidden_dim
        self.net = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x.transpose(1, 2))
        return h.mean(dim=-1)


class HazardHead(nn.Module):
    """LayerNorm -> Linear(128) -> GELU -> Dropout -> Linear(H)."""

    def __init__(self, input_dim: int, horizon: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class IndependentHazardModel(nn.Module):
    """Encode each variation independently and predict its hazard logits."""

    def __init__(self, input_dim: int, hidden_dim: int, horizon: int, encoder_type: str, dropout: float):
        super().__init__()
        if encoder_type == "stats":
            self.encoder = StatsTrajectoryEncoder(input_dim, hidden_dim, dropout)
        elif encoder_type == "cnn":
            self.encoder = ConvTrajectoryEncoder(input_dim, hidden_dim, dropout)
        elif encoder_type == "tcn":
            self.encoder = TCNTrajectoryEncoder(input_dim, hidden_dim, dropout)
        elif encoder_type == "gru":
            self.encoder = GRUTrajectoryEncoder(input_dim, hidden_dim, dropout)
        elif encoder_type == "attention":
            self.encoder = TemporalAttentionEncoder(input_dim, hidden_dim, heads=4, layers=1, dropout=dropout)
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")
        self.head = HazardHead(hidden_dim, horizon, hidden_dim, dropout=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected [batch,variation,time,channels], got {tuple(x.shape)}.")
        batch, variations, time, channels = x.shape
        z = self.encoder(x.reshape(batch * variations, time, channels))
        return self.head(z).reshape(batch, variations, -1)


class FrozenChannelAttentionPooling(nn.Module):
    """Pool frozen channel embeddings with learned attention weights."""

    def __init__(
        self,
        channels: int,
        embedding_hidden: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if channels <= 0 or embedding_hidden <= 0 or hidden_dim <= 0:
            raise ValueError("channels, embedding_hidden, and hidden_dim must be positive.")
        self.channels = int(channels)
        self.embedding_hidden = int(embedding_hidden)
        self.hidden_dim = int(hidden_dim)
        self.input_norm = nn.LayerNorm(self.embedding_hidden)
        self.project = nn.Linear(self.embedding_hidden, self.hidden_dim)
        self.channel_embedding = nn.Parameter(torch.empty(self.channels, self.hidden_dim))
        score_hidden = max(self.hidden_dim // 2, 32)
        self.score = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, score_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(score_hidden, 1),
        )
        nn.init.normal_(self.channel_embedding, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected [batch,variation,embedding], got {tuple(x.shape)}.")
        batch, variations, embedding_dim = x.shape
        expected = self.channels * self.embedding_hidden
        if embedding_dim != expected:
            raise ValueError(f"Expected embedding_dim={expected}, got {embedding_dim}.")
        tokens = x.reshape(batch * variations, self.channels, self.embedding_hidden)
        tokens = self.project(self.input_norm(tokens)) + self.channel_embedding.unsqueeze(0)
        scores = self.score(tokens).squeeze(-1)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        pooled = (weights * tokens).sum(dim=1)
        return pooled.reshape(batch, variations, self.hidden_dim)


class FrozenEmbeddingHazardModel(nn.Module):
    """Hazard head for already-computed Panda/Chronos representations.

    ``direct`` keeps the flattened embedding (15 x 768 = 11520 for the
    current caches) and lets the HazardHead's first linear layer consume it
    directly.  ``meanstd`` reshapes it to [15, 768], pools over the 15 input
    variates, and feeds [mean, std] (1536 dimensions) to the same type of
    head.  ``attention`` reshapes the embedding to 15 channel tokens and uses
    a trainable learned-weight attention pool before the hazard head.  No
    Panda/Chronos parameters are present here: the backbone is frozen by
    construction because only the cached features are loaded.
    """

    def __init__(
        self,
        input_dim: int,
        horizon: int,
        hidden_dim: int,
        dropout: float,
        pooling: str = "direct",
        embedding_channels: int = 15,
        embedding_hidden: int = 768,
    ):
        super().__init__()
        if pooling not in {"direct", "meanstd", "attention"}:
            raise ValueError(f"Unknown frozen embedding pooling: {pooling}")
        self.input_dim = int(input_dim)
        self.pooling = pooling
        self.embedding_channels = int(embedding_channels)
        self.embedding_hidden = int(embedding_hidden)
        if pooling in {"meanstd", "attention"}:
            expected = self.embedding_channels * self.embedding_hidden
            if self.input_dim != expected:
                raise ValueError(
                    f"{pooling} pooling expects input_dim={expected} (= channels*hidden), "
                    f"got {self.input_dim}."
                )
            if pooling == "meanstd":
                head_dim = 2 * self.embedding_hidden
            else:
                self.attention_pool = FrozenChannelAttentionPooling(
                    channels=self.embedding_channels,
                    embedding_hidden=self.embedding_hidden,
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                )
                head_dim = hidden_dim
        else:
            head_dim = self.input_dim
        self.head = HazardHead(head_dim, horizon, hidden_dim, dropout=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected [batch,variation,embedding], got {tuple(x.shape)}.")
        if self.pooling == "direct":
            features = x
        elif self.pooling == "meanstd":
            bsz, variations, _ = x.shape
            channels = x.reshape(bsz, variations, self.embedding_channels, self.embedding_hidden)
            features = torch.cat([channels.mean(dim=2), channels.std(dim=2, unbiased=False)], dim=-1)
        else:
            features = self.attention_pool(x)
        return self.head(features)


class EventSurvivalDataset(Dataset):
    def __init__(
        self,
        X: np.ndarray,
        hazard_target: np.ndarray,
        at_risk: np.ndarray,
        variation_survival_target: np.ndarray,
        system_target: np.ndarray,
        old_system_target: np.ndarray,
    ):
        arrays = [X, hazard_target, at_risk, variation_survival_target, system_target, old_system_target]
        n = int(np.asarray(X).shape[0])
        if any(np.asarray(arr).shape[0] != n for arr in arrays):
            raise ValueError("All survival dataset arrays must share the case dimension.")
        self.X = torch.from_numpy(np.asarray(X, dtype=np.float32))
        self.hazard_target = torch.from_numpy(np.asarray(hazard_target, dtype=np.float32))
        self.at_risk = torch.from_numpy(np.asarray(at_risk, dtype=np.float32))
        self.variation_survival_target = torch.from_numpy(np.asarray(variation_survival_target, dtype=np.float32))
        self.system_target = torch.from_numpy(np.asarray(system_target, dtype=np.float32))
        self.old_system_target = torch.from_numpy(np.asarray(old_system_target, dtype=np.float32))

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        return (
            self.X[index],
            self.hazard_target[index],
            self.at_risk[index],
            self.variation_survival_target[index],
            self.system_target[index],
            self.old_system_target[index],
        )


class EventSurvivalObjective(nn.Module):
    """Variation hazard NLL plus the requested system curve MAE."""

    def __init__(self, curve_weight: float = 0.2):
        super().__init__()
        if curve_weight < 0.0:
            raise ValueError("curve_weight must be non-negative.")
        self.curve_weight = float(curve_weight)
        self.variation_criterion = MaskedHazardNLL()

    def components(
        self,
        logits: torch.Tensor,
        hazard_target: torch.Tensor,
        at_risk: torch.Tensor,
        system_target: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        variation_loss = self.variation_criterion(logits, hazard_target, at_risk)
        variation_survival = hazard_logits_to_survival(logits)
        system_prediction = variation_survival.mean(dim=1)
        curve_loss = nn.functional.l1_loss(system_prediction, system_target)
        total = variation_loss + self.curve_weight * curve_loss
        return {
            "loss": total,
            "variation_nll": variation_loss,
            "curve_mae_loss": curve_loss,
        }

    def forward(
        self,
        logits: torch.Tensor,
        hazard_target: torch.Tensor,
        at_risk: torch.Tensor,
        system_target: torch.Tensor,
    ) -> torch.Tensor:
        return self.components(logits, hazard_target, at_risk, system_target)["loss"]


def _case_arrays(
    X: np.ndarray,
    labels,
    group_size: int,
    max_cases: int | None,
    output_dtype: np.dtype | type = np.float32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = labels.copy()
    labels["case_id"] = labels["index"] // group_size
    labels["variation_id"] = labels["index"] % group_size
    labels["_row"] = np.arange(len(labels), dtype=np.int64)
    expected = np.arange(group_size, dtype=np.int64)
    complete: list[int] = []
    for case_id, frame in labels.groupby("case_id", sort=True):
        variation_ids = np.sort(frame["variation_id"].to_numpy(dtype=np.int64))
        if len(frame) == group_size and np.array_equal(variation_ids, expected):
            rows = frame.sort_values("variation_id")["_row"].to_numpy(dtype=np.int64)
            if np.isfinite(X[rows]).all():
                complete.append(int(case_id))
    if max_cases is not None:
        complete = complete[: int(max_cases)]
    if not complete:
        raise ValueError("No complete finite case groups survived filtering.")
    selected = labels[labels["case_id"].isin(complete)].sort_values(["case_id", "variation_id"])
    rows = selected["_row"].to_numpy(dtype=np.int64).reshape(len(complete), group_size)
    times = selected["time_unbound"].to_numpy(dtype=np.float32).reshape(len(complete), group_size)
    return X[rows].astype(output_dtype, copy=False), times, np.asarray(complete, dtype=np.int64)


def parse_future_times(value: str) -> np.ndarray:
    result = np.asarray([float(item.strip()) for item in value.split(",") if item.strip()], dtype=np.float32)
    if result.size == 0 or not np.all(np.diff(result) > 0.0) or np.any(result <= 0.0):
        raise ValueError("future-times must be positive and strictly increasing.")
    return result


def _embedding_label_fingerprint(labels: pd.DataFrame) -> str:
    """Hash the row identity used by the Task B embedding caches."""

    columns = ["index", "num_batch"]
    if "row_in_batch" in labels.columns:
        columns.append("row_in_batch")
    if any(column not in labels.columns for column in columns):
        raise ValueError(f"Embedding labels need columns {columns}, got {list(labels.columns)}.")
    identity = labels[columns].to_numpy(dtype=np.int64, copy=True)
    return hashlib.sha256(identity.tobytes()).hexdigest()


def _load_embedding_cases(args: argparse.Namespace) -> tuple[np.ndarray, Any, list[Path], list[str], dict[str, Any]]:
    """Load and group a frozen Panda/Chronos feature cache into complete systems."""

    cache_path = Path(args.embedding_cache)
    if not cache_path.exists():
        raise FileNotFoundError(f"Embedding cache does not exist: {cache_path}")
    features = np.load(cache_path, mmap_mode="r")
    if features.ndim != 2:
        raise ValueError(f"Expected [trajectory,embedding_dim] cache, got {features.shape}.")

    metadata_path = cache_path.with_suffix(".json")
    metadata: dict[str, Any] = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
    expected_rows = int(features.shape[0])

    label_candidates: list[Path] = []
    if args.embedding_labels is not None:
        label_candidates.append(Path(args.embedding_labels))
    label_candidates.append(Path("results/task_b_prefix_ablation") / f"prefix_{args.prefix_len}" / "labels_filtered.csv")
    label_candidates.append(args.dataset_dir / f"completed_systems_sampling_{args.sampling_id}.txt")
    labels_path = next((path for path in label_candidates if path.exists()), None)
    if labels_path is None:
        raise FileNotFoundError(
            "Could not locate labels for the embedding cache. Pass --embedding-labels "
            "with the labels file used during feature extraction."
        )

    if labels_path.suffix.lower() == ".csv":
        labels = pd.read_csv(labels_path)
    else:
        labels = pd.read_csv(
            labels_path,
            sep=r"\s+",
            header=None,
            names=["index", "num_batch", "time_unbound", "wall_time", "diagnosis"],
        )
        labels = labels.astype(
            {"index": "int64", "num_batch": "int64", "time_unbound": "float32", "diagnosis": "int64"}
        )
        # The historical sampling_2 shards contain 8000 rows, while the
        # high-resolution sampling_3 shards contain 4000.  Use the explicit
        # completion-table batch id and derive the row order within each batch
        # instead of assuming a fixed shard size.
        labels["batch_id"] = labels["num_batch"].astype("int64")
        labels = labels.sort_values(["batch_id", "index"]).reset_index(drop=True)
        labels["row_in_batch"] = labels.groupby("batch_id", sort=False).cumcount()

    if len(labels) != expected_rows:
        # A complete 160000-row completion table can be reduced to the valid
        # rows used by the Task B caches.  Never silently truncate/reorder a
        # cache with an incompatible number of rows.
        valid = (labels["time_unbound"].to_numpy(dtype=np.float32) >= 0.0) & (
            labels["diagnosis"].to_numpy(dtype=np.int64) == 0
        )
        if int(valid.sum()) == expected_rows:
            labels = labels.loc[valid].reset_index(drop=True)
        else:
            raise ValueError(
                f"Embedding rows ({expected_rows}) do not match labels ({len(labels)}) and "
                f"valid labels ({int(valid.sum())}). Pass the exact aligned label file."
            )
    labels = labels.reset_index(drop=True)
    expected_fingerprint = metadata.get("label_fingerprint")
    if expected_fingerprint:
        actual_fingerprint = _embedding_label_fingerprint(labels)
        if actual_fingerprint != expected_fingerprint:
            raise ValueError(
                "Embedding/label identity mismatch: "
                f"cache fingerprint={expected_fingerprint}, labels fingerprint={actual_fingerprint}. "
                "Use the aligned labels file produced with this cache."
            )

    # Keep float16 until the train-only standardisation step to avoid making a
    # second 7+ GB float32 copy of the 11520-dimensional cache unnecessarily.
    X_cases, times_cases, case_ids = _case_arrays(
        features,
        labels,
        args.group_size,
        args.max_cases,
        output_dtype=np.float16 if features.dtype == np.float16 else np.float32,
    )
    feature_names = [f"embedding_{i}" for i in range(int(features.shape[1]))]
    return X_cases, times_cases, case_ids, feature_names, metadata


def _load_prefix_data(args: argparse.Namespace) -> tuple[np.ndarray, Any, list[Path], list[str]]:
    if args.prefix_representation == "orbital15":
        result_keys = ORBITAL15_RESULT_KEYS
        include_geom = False
        add_time = False
    else:
        result_keys = FULL_RESULT_KEYS
        include_geom = not args.no_geom
        add_time = not args.no_time
    if args.prefix_cache is not None:
        data, labels, paths, X, feature_names = load_cached_prefixes(
            args.prefix_cache,
            prefix_len=args.prefix_len,
            time_stride=args.time_stride,
            add_time=add_time,
            result_keys=result_keys,
            include_geom=include_geom,
        )
    else:
        paths = discover_result_paths(args.dataset_dir, args.sampling_id, args.max_shards)
        data, batch_ids, rows_loaded = load_result_prefixes(
            paths,
            result_keys=result_keys,
            prefix_len=args.prefix_len,
            time_stride=args.time_stride,
            rows_per_shard=args.rows_per_shard,
            include_geom=include_geom,
            add_time=add_time,
        )
        labels = load_aligned_labels(
            args.dataset_dir / f"completed_systems_sampling_{args.sampling_id}.txt",
            batch_ids,
            rows_loaded,
        )
        X, feature_names = build_prefix_features(
            data,
            result_keys=result_keys,
            add_time=add_time,
            include_geom=include_geom,
        )
    X, _, labels, _ = filter_valid_samples(X, labels, data, time_horizon=None)
    X_cases, times_cases, case_ids = _case_arrays(X, labels, args.group_size, args.max_cases)
    return X_cases, times_cases, case_ids, feature_names


def _metrics_from_predictions(
    variation_survival: np.ndarray,
    hazard_target: np.ndarray,
    at_risk: np.ndarray,
    variation_survival_target: np.ndarray,
    system_target: np.ndarray,
    old_system_target: np.ndarray,
    edges: np.ndarray,
    future_times: np.ndarray,
    loss_components: dict[str, float] | None = None,
) -> dict[str, Any]:
    system_prediction = variation_survival.mean(axis=1)
    system_error = system_prediction - system_target
    variation_error = variation_survival - variation_survival_target
    # The event/censoring boundary is at t_max.  The empirical indicator is
    # known at all earlier grid points and is conservatively one for a sample
    # censored at the observation horizon.
    brier_by_bin = np.mean(variation_error**2, axis=(0, 1))
    old_prediction = interpolate_survival(system_prediction, edges, future_times)
    old_error = old_prediction - old_system_target
    result: dict[str, Any] = {
        **(loss_components or {}),
        "overall_mae": float(np.abs(system_error).mean()),
        "final_mae": float(np.abs(system_error[:, -1]).mean()),
        "horizon_mae": np.abs(system_error).mean(axis=0),
        "overall_rmse": float(np.sqrt(np.mean(system_error**2))),
        "brier_by_bin": brier_by_bin,
        "ibs": float(brier_by_bin.mean()),
        "variation_overall_mae": float(np.abs(variation_error).mean()),
        "variation_final_mae": float(np.abs(variation_error[:, :, -1]).mean()),
        "variation_brier": float(np.mean(variation_error**2)),
        "monotonicity_violation_max": float(np.maximum(np.diff(system_prediction, axis=-1), 0.0).max(initial=0.0)),
        "old_horizon_mae": np.abs(old_error).mean(axis=0),
        "old_overall_mae": float(np.abs(old_error).mean()),
        "old_final_mae": float(np.abs(old_error[:, -1]).mean()),
        "old_horizon_rmse": np.sqrt(np.mean(old_error**2, axis=0)),
        "preds": system_prediction,
        "targets": system_target,
        "variation_preds": variation_survival,
        "variation_targets": variation_survival_target,
        "old_preds": old_prediction,
        "old_targets": old_system_target,
        "hazard_targets": hazard_target,
        "at_risk": at_risk,
        "bin_edges": edges,
        "future_times": future_times,
    }
    return result


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    objective: EventSurvivalObjective,
    device: torch.device,
    edges: np.ndarray,
    future_times: np.ndarray,
) -> dict[str, Any]:
    model.eval()
    loss_sums = {"loss": 0.0, "variation_nll": 0.0, "curve_mae_loss": 0.0}
    logits_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    risk_parts: list[np.ndarray] = []
    survival_target_parts: list[np.ndarray] = []
    system_target_parts: list[np.ndarray] = []
    old_target_parts: list[np.ndarray] = []
    for xb, target, risk, var_surv_target, system_target, old_target in loader:
        xb = xb.to(device)
        target = target.to(device)
        risk = risk.to(device)
        system_target = system_target.to(device)
        logits = model(xb)
        components = objective.components(logits, target, risk, system_target)
        for name in loss_sums:
            loss_sums[name] += float(components[name].item()) * xb.size(0)
        logits_parts.append(logits.cpu().numpy())
        target_parts.append(target.cpu().numpy())
        risk_parts.append(risk.cpu().numpy())
        survival_target_parts.append(var_surv_target.numpy())
        system_target_parts.append(system_target.cpu().numpy())
        old_target_parts.append(old_target.numpy())
    logits = np.concatenate(logits_parts, axis=0)
    variation_survival = 1.0 / (1.0 + np.exp(-logits))
    variation_survival = np.cumprod(1.0 - variation_survival, axis=-1)
    n = len(loader.dataset)
    loss_components = {name: value / n for name, value in loss_sums.items()}
    return _metrics_from_predictions(
        variation_survival,
        np.concatenate(target_parts, axis=0),
        np.concatenate(risk_parts, axis=0),
        np.concatenate(survival_target_parts, axis=0),
        np.concatenate(system_target_parts, axis=0),
        np.concatenate(old_target_parts, axis=0),
        edges,
        future_times,
        loss_components,
    )


def _train_one(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    objective: EventSurvivalObjective,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    edges: np.ndarray,
    future_times: np.ndarray,
) -> tuple[nn.Module, list[dict[str, float]]]:
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_state: dict[str, torch.Tensor] | None = None
    best_value = float("inf")
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        sums = {"loss": 0.0, "variation_nll": 0.0, "curve_mae_loss": 0.0}
        for xb, target, risk, _var_target, system_target, _old_target in train_loader:
            xb = xb.to(device)
            target = target.to(device)
            risk = risk.to(device)
            system_target = system_target.to(device)
            optimizer.zero_grad(set_to_none=True)
            components = objective.components(model(xb), target, risk, system_target)
            components["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            for name in sums:
                sums[name] += float(components[name].item()) * xb.size(0)
        val = evaluate_model(model, val_loader, objective, device, edges, future_times)
        train_n = len(train_loader.dataset)
        row = {
            "epoch": float(epoch),
            "train_loss": sums["loss"] / train_n,
            "train_variation_nll": sums["variation_nll"] / train_n,
            "train_curve_mae_loss": sums["curve_mae_loss"] / train_n,
            "val_loss": float(val["loss"]),
            "val_variation_nll": float(val["variation_nll"]),
            "val_curve_mae_loss": float(val["curve_mae_loss"]),
            "val_overall_mae": float(val["overall_mae"]),
        }
        history.append(row)
        if row["val_loss"] < best_value:
            best_value = row["val_loss"]
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        print(
            f"epoch={epoch:03d} train_loss={row['train_loss']:.5f} "
            f"val_loss={row['val_loss']:.5f} val_system_mae={row['val_overall_mae']:.5f}"
        )
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def _baseline_metrics(
    train_target: np.ndarray,
    test_target: np.ndarray,
    old_train_target: np.ndarray,
    old_test_target: np.ndarray,
    edges: np.ndarray,
    future_times: np.ndarray,
) -> dict[str, Any]:
    prediction = np.broadcast_to(train_target.mean(axis=0, keepdims=True), test_target.shape).copy()
    # A system baseline has no variation hazard logits.  Use the mean system
    # curve as a one-variation proxy solely for the common metric formatter.
    variation_prediction = np.broadcast_to(prediction[:, None, :], (prediction.shape[0], 1, prediction.shape[1]))
    variation_target = test_target[:, None, :]
    return _metrics_from_predictions(
        variation_prediction,
        np.zeros_like(variation_prediction),
        np.ones_like(variation_prediction),
        variation_target,
        test_target,
        old_test_target,
        edges,
        future_times,
        {"loss": float("nan"), "variation_nll": float("nan"), "curve_mae_loss": float("nan")},
    )


def _compact_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    excluded = {
        "preds",
        "targets",
        "variation_preds",
        "variation_targets",
        "old_preds",
        "old_targets",
        "hazard_targets",
        "at_risk",
    }
    return {key: value for key, value in payload.items() if key not in excluded}


def _save_plots(output_dir: Path, edges: np.ndarray, future_times: np.ndarray, metrics: dict[str, dict[str, Any]]) -> None:
    names = [name for name, payload in metrics.items() if "preds" in payload]
    if not names:
        return
    first = metrics[names[0]]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(edges[1:], first["targets"].mean(axis=0), color="black", linewidth=2.5, label="true")
    for name in names:
        ax.plot(edges[1:], metrics[name]["preds"].mean(axis=0), linewidth=1.8, label=name)
    ax.set_xscale("log")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("absolute time_unbound")
    ax.set_ylabel("system survival")
    ax.set_title("20-bin event-time survival curve")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "figure_1_20bin_survival_curve.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(future_times, first["old_targets"].mean(axis=0), color="black", linewidth=2.5, marker="o", label="true")
    for name in names:
        ax.plot(future_times, metrics[name]["old_preds"].mean(axis=0), marker="o", linewidth=1.8, label=name)
    ax.set_xscale("log")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("legacy horizon")
    ax.set_ylabel("system survival")
    ax.set_title("Legacy seven-horizon readout")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "figure_2_legacy_horizons.png", dpi=180)
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Task C event-time discrete survival model.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset_v3"))
    parser.add_argument("--sampling-id", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=Path("results/task_c_survival20"))
    parser.add_argument("--prefix-cache", type=Path, default=None)
    parser.add_argument(
        "--embedding-cache",
        type=Path,
        default=None,
        help="Precomputed frozen Panda/Chronos cache with shape [trajectory, embedding_dim].",
    )
    parser.add_argument(
        "--embedding-labels",
        type=Path,
        default=None,
        help="Labels aligned to --embedding-cache; defaults to the Task B prefix labels for --prefix-len.",
    )
    parser.add_argument("--embedding-channels", type=int, default=15)
    parser.add_argument("--embedding-hidden", type=int, default=768)
    parser.add_argument("--prefix-len", type=int, default=100)
    parser.add_argument("--time-stride", type=int, default=1)
    parser.add_argument("--prefix-representation", choices=["orbital15", "full"], default="orbital15")
    parser.add_argument("--no-geom", action="store_true")
    parser.add_argument("--no-time", action="store_true")
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--rows-per-shard", type=int, default=None)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--variation-count", type=int, default=None)
    parser.add_argument(
        "--variation-target",
        choices=["subset", "full32"],
        default="full32",
        help="Supervise the selected variation fraction or the full 32-variation system curve.",
    )
    parser.add_argument("--variation-subset-mode", choices=["balanced", "random"], default="balanced")
    parser.add_argument("--variation-subset-seed", type=int, default=42)
    parser.add_argument("--future-times", type=str, default="500,800,1500,2500,4000,6500,10000")
    parser.add_argument("--hazard-bins", type=int, default=20)
    parser.add_argument("--hazard-t-min", type=float, default=None)
    parser.add_argument("--hazard-t-max", type=float, default=10000.0)
    parser.add_argument("--models", type=str, default="baseline,stats")
    parser.add_argument("--encoder-type", choices=["stats", "cnn", "tcn", "gru", "attention"], default="stats")
    parser.add_argument("--curve-loss-weight", type=float, default=0.2)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument(
        "--split-horizon",
        type=float,
        default=10000.0,
        help="Fixed horizon used only to stratify the shared A/B/C system split.",
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=None,
        help="Shared raw_case_id split CSV; created from the completion table when absent.",
    )
    parser.add_argument("--overwrite-split-manifest", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    output_dir = ensure_dir(args.output_dir)
    set_random_seed(args.seed)
    device = select_device(args.device)
    future_times = parse_future_times(args.future_times)
    requested_models = set(parse_name_list(args.models))
    allowed_models = {
        "baseline",
        "stats",
        "cnn",
        "tcn",
        "gru",
        "attention",
        "independent",
        "embedding_direct",
        "embedding_meanstd",
        "embedding_attention",
    }
    unsupported = sorted(requested_models - allowed_models)
    if unsupported:
        raise ValueError(f"Unsupported event-survival models: {unsupported}")
    if args.group_size < 1:
        raise ValueError("group-size must be positive.")

    if args.embedding_cache is not None:
        raw_model_names = requested_models & {"stats", "cnn", "tcn", "gru", "attention", "independent"}
        if raw_model_names:
            raise ValueError(
                "--embedding-cache is for embedding_direct/embedding_meanstd/embedding_attention; "
                f"raw trajectory models were also requested: {sorted(raw_model_names)}"
            )
        X_cases, times_cases, case_ids, feature_names, embedding_metadata = _load_embedding_cases(args)
        input_kind = "frozen_embedding"
    else:
        embedding_metadata = {}
        X_cases, times_cases, case_ids, feature_names = _load_prefix_data(args)
        input_kind = "raw_trajectory"
    if args.variation_count is not None and args.variation_count != args.group_size:
        indices = build_variation_subset_indices(
            args.group_size,
            args.variation_count,
            mode=args.variation_subset_mode,
            seed=args.variation_subset_seed,
        )
        X_model = X_cases[:, indices]
        times_model = times_cases[:, indices]
    else:
        indices = np.arange(args.group_size, dtype=np.int64)
        X_model = X_cases
        times_model = times_cases
    if args.hazard_t_min is None:
        candidate = times_cases[np.isfinite(times_cases) & (times_cases > 0.0) & (times_cases < args.hazard_t_max)]
        if candidate.size == 0:
            raise ValueError("Could not infer a positive event time below hazard-t-max.")
        hazard_t_min = float(candidate.min())
    else:
        hazard_t_min = float(args.hazard_t_min)
    edges = make_log_bin_edges(hazard_t_min, args.hazard_t_max, args.hazard_bins)

    hazard_target, at_risk, target_meta = build_discrete_hazard_targets(
        times_model,
        edges,
        return_metadata=True,
    )
    variation_survival_target = survival_targets_from_times(times_model, edges)
    full_system_target = survival_targets_from_times(times_cases, edges).mean(axis=1)
    model_system_target = variation_survival_target.mean(axis=1)
    # Keep the system target on all 32 variations by default, matching the
    # Task C definition even when a reduced variation subset is input.
    if args.variation_target == "full32":
        system_target = full_system_target
    else:
        system_target = model_system_target
    old_full_target = (times_cases[..., None] >= future_times[None, None, :]).mean(axis=1)
    old_model_target = (times_model[..., None] >= future_times[None, None, :]).mean(axis=1)
    old_system_target = old_full_target if args.variation_target == "full32" else old_model_target

    labels_path = args.dataset_dir / f"completed_systems_sampling_{args.sampling_id}.txt"
    split_manifest = args.split_manifest or default_case_split_manifest_path(
        args.dataset_dir,
        args.sampling_id,
        args.group_size,
        args.split_horizon,
        args.seed,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
    )
    split_manifest_df = ensure_case_split_manifest(
        labels_path,
        split_manifest,
        group_size=args.group_size,
        time_horizon=args.split_horizon,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        overwrite=args.overwrite_split_manifest,
    )
    if args.max_shards is None and args.rows_per_shard is None and args.max_cases is None:
        missing = missing_case_ids(case_ids, split_manifest_df)
        if missing.size:
            raise ValueError(
                "Task C survival does not contain the full common system universe after input filtering; "
                f"missing {missing.size} raw_case_id values (first: {missing[:10].tolist()})."
            )
    split_indices = case_split_indices(case_ids, split_manifest_df)
    train_idx, val_idx, test_idx = (
        split_indices["train"],
        split_indices["val"],
        split_indices["test"],
    )
    X_train, X_val, X_test = X_model[train_idx], X_model[val_idx], X_model[test_idx]
    target_train, target_val, target_test = hazard_target[train_idx], hazard_target[val_idx], hazard_target[test_idx]
    risk_train, risk_val, risk_test = at_risk[train_idx], at_risk[val_idx], at_risk[test_idx]
    var_surv_train, var_surv_val, var_surv_test = (
        variation_survival_target[train_idx],
        variation_survival_target[val_idx],
        variation_survival_target[test_idx],
    )
    system_train, system_val, system_test = system_target[train_idx], system_target[val_idx], system_target[test_idx]
    old_train, old_val, old_test = old_system_target[train_idx], old_system_target[val_idx], old_system_target[test_idx]
    # Embedding caches are float16 on disk.  Compute train-only statistics in
    # float32 before the DataLoader converts batches to torch tensors.
    if X_train.dtype != np.float32:
        X_train = X_train.astype(np.float32)
        X_val = X_val.astype(np.float32)
        X_test = X_test.astype(np.float32)
    X_train, [X_val, X_test], norm = standardize_case_train(X_train, X_val, X_test)

    train_ds = EventSurvivalDataset(X_train, target_train, risk_train, var_surv_train, system_train, old_train)
    val_ds = EventSurvivalDataset(X_val, target_val, risk_val, var_surv_val, system_val, old_val)
    test_ds = EventSurvivalDataset(X_test, target_test, risk_test, var_surv_test, system_test, old_test)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    np.savez(output_dir / "split_indices.npz", train_idx=train_idx, val_idx=val_idx, test_idx=test_idx, case_ids=case_ids)
    np.savez(output_dir / "normalization_stats.npz", mean=norm["mean"], std=norm["std"])
    np.savez(
        output_dir / "target_summary.npz",
        bin_edges=edges,
        full_system_target=full_system_target[test_idx],
        model_system_target=system_test,
        event_bin=target_meta["event_bin"][test_idx],
        censored=target_meta["censored"][test_idx],
    )
    save_json(
        output_dir / "config.json",
        {
            **vars(args),
            "device": str(device),
            "input_kind": input_kind,
            "embedding_metadata": embedding_metadata,
            "hazard_t_min": hazard_t_min,
            "hazard_t_max": float(args.hazard_t_max),
            "bin_edges": edges,
            "num_cases": int(X_model.shape[0]),
            "input_shape": tuple(X_model.shape[1:]),
            "feature_names": feature_names,
            "train_cases": int(train_idx.size),
            "val_cases": int(val_idx.size),
            "test_cases": int(test_idx.size),
            "variation_indices": indices,
            "variation_target": args.variation_target,
            "censoring_fraction": float(target_meta["censored"].mean()),
            "left_boundary_event_fraction": float((times_model <= hazard_t_min).mean()),
            "full32_system_target_saved": True,
            "split_manifest": split_manifest,
            "split_unit": "complete raw_case_id system",
            "split_horizon": float(args.split_horizon),
            "split_ratios": {
                "train": float(args.train_ratio),
                "val": float(args.val_ratio),
                "test": float(args.test_ratio),
            },
            "split_counts": {name: int(rows.size) for name, rows in split_indices.items()},
        },
    )

    metrics: dict[str, dict[str, Any]] = {}
    if "baseline" in requested_models:
        metrics["baseline"] = _baseline_metrics(system_train, system_test, old_train, old_test, edges, future_times)

    objective = EventSurvivalObjective(curve_weight=args.curve_loss_weight)
    for name in sorted(requested_models - {"baseline"}):
        if name in {"embedding_direct", "embedding_meanstd", "embedding_attention"}:
            if args.embedding_cache is None:
                raise ValueError(f"{name} requires --embedding-cache.")
            pooling = {
                "embedding_direct": "direct",
                "embedding_meanstd": "meanstd",
                "embedding_attention": "attention",
            }[name]
            model = FrozenEmbeddingHazardModel(
                input_dim=X_train.shape[-1],
                horizon=args.hazard_bins,
                hidden_dim=args.hidden_dim,
                dropout=args.dropout,
                pooling=pooling,
                embedding_channels=args.embedding_channels,
                embedding_hidden=args.embedding_hidden,
            )
            encoder_type = f"frozen_embedding_{pooling}"
        else:
            if args.embedding_cache is not None:
                raise ValueError(f"Raw trajectory model {name} cannot consume --embedding-cache.")
            encoder_type = args.encoder_type if name == "independent" else name
            model = IndependentHazardModel(
                input_dim=X_train.shape[-1],
                hidden_dim=args.hidden_dim,
                horizon=args.hazard_bins,
                encoder_type=encoder_type,
                dropout=args.dropout,
            )
        model, history = _train_one(
            model,
            train_loader,
            val_loader,
            objective,
            args.epochs,
            args.lr,
            args.weight_decay,
            device,
            edges,
            future_times,
        )
        metrics[name] = evaluate_model(model, test_loader, objective, device, edges, future_times)
        save_csv(output_dir / f"{name}_train_log.csv", history)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "input_dim": int(X_train.shape[-1]),
                "hidden_dim": int(args.hidden_dim),
                "hazard_bins": int(args.hazard_bins),
                "encoder_type": encoder_type,
                "bin_edges": edges,
            },
            output_dir / f"{name}_model.pt",
        )

    _save_plots(output_dir, edges, future_times, metrics)
    prediction_arrays: dict[str, np.ndarray] = {"bin_edges": edges, "future_times": future_times}
    for name, payload in metrics.items():
        for key in ("preds", "targets", "variation_preds", "variation_targets", "old_preds", "old_targets"):
            if key in payload:
                prediction_arrays[f"{name}_{key}"] = payload[key]
    np.savez(output_dir / "test_predictions.npz", **prediction_arrays)
    save_json(output_dir / "metrics.json", {name: _compact_metrics(payload) for name, payload in metrics.items()})
    print(
        f"Task C event-survival complete: {output_dir} | cases={X_model.shape[0]} "
        f"bins={args.hazard_bins} edges={hazard_t_min:g}->{args.hazard_t_max:g}"
    )
    for name, payload in metrics.items():
        print(
            f"{name}: overall_mae={payload.get('overall_mae', float('nan')):.6f} "
            f"final_mae={payload.get('final_mae', float('nan')):.6f} "
            f"old_overall_mae={payload.get('old_overall_mae', float('nan')):.6f} "
            f"ibs={payload.get('ibs', float('nan')):.6f}"
        )


if __name__ == "__main__":
    main()
