"""Models and deterministic sampling helpers for phase-ensemble experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class BudgetCondition:
    """One sampled-unit / prefix-length allocation under a fixed point budget."""

    k: float
    t: int
    unit_size: int = 1

    @property
    def budget(self) -> int:
        return int(round(self.k * self.t * self.unit_size))


def budget_conditions(
    budget: int = 512,
    phase_counts: tuple[int | float, ...] = (1, 2, 4, 8, 16),
    max_prefix: int = 1024,
    phases_per_unit: int = 1,
) -> tuple[BudgetCondition, ...]:
    if budget <= 0 or max_prefix <= 0 or phases_per_unit <= 0:
        raise ValueError("budget, max_prefix, and phases_per_unit must be positive.")
    conditions: list[BudgetCondition] = []
    for k in phase_counts:
        divisor = float(k) * phases_per_unit
        if divisor <= 0 or not np.isclose(budget / divisor, round(budget / divisor)):
            raise ValueError(
                f"Budget {budget} must be divisible by K={k} sampled units "
                f"times {phases_per_unit} phases per unit."
            )
        t = int(round(budget / divisor))
        if t > max_prefix:
            raise ValueError(f"Prefix T={t} exceeds the available {max_prefix} stored points.")
        numeric_k = float(k)
        stored_k: int | float = int(numeric_k) if numeric_k.is_integer() else numeric_k
        conditions.append(BudgetCondition(k=stored_k, t=int(t), unit_size=int(phases_per_unit)))
    if not conditions:
        raise ValueError("At least one budget condition is required.")
    return tuple(conditions)

def build_pair_partition(
    system_ids: np.ndarray,
    pair_count: int = 16,
    heldout_pairs: int = 8,
    seed: int = 42,
) -> np.ndarray:
    """Return a reproducible per-system permutation of matched-pair IDs."""

    ids = np.asarray(system_ids, dtype=np.int64).reshape(-1)
    if np.unique(ids).size != ids.size:
        raise ValueError("system_ids must be unique.")
    if not 0 < heldout_pairs < pair_count:
        raise ValueError("heldout_pairs must be between zero and pair_count.")
    result = np.empty((ids.size, pair_count), dtype=np.int16)
    for row, system_id in enumerate(ids):
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(system_id), 73021]))
        result[row] = rng.permutation(pair_count)
    return result


def pair_ids_to_variations(pair_ids: np.ndarray, pair_members: np.ndarray) -> np.ndarray:
    """Expand sampled pair IDs to their two matched variation IDs."""

    ids = np.asarray(pair_ids, dtype=np.int64)
    members = np.asarray(pair_members, dtype=np.int64)
    if members.ndim != 2 or members.shape[1] != 2:
        raise ValueError("pair_members must have shape [pairs, 2].")
    if np.any(ids < 0) or np.any(ids >= members.shape[0]):
        raise ValueError("pair_ids contains an ID outside pair_members.")
    return members[ids]


def build_phase_partition(
    system_ids: np.ndarray,
    group_size: int = 32,
    heldout_size: int = 16,
    seed: int = 42,
) -> np.ndarray:
    """Return [systems, variations] with candidate slots first and heldout last.

    A system-specific stream makes the partition independent of manifest row
    order while still producing a reproducible, balanced 16/16 split.
    """

    ids = np.asarray(system_ids, dtype=np.int64).reshape(-1)
    if np.unique(ids).size != ids.size:
        raise ValueError("system_ids must be unique.")
    if not 0 < heldout_size < group_size:
        raise ValueError("heldout_size must be between zero and group_size.")
    result = np.empty((ids.size, group_size), dtype=np.int16)
    for row, system_id in enumerate(ids):
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(system_id), 73021]))
        result[row] = rng.permutation(group_size)
    return result


def nested_phase_draws(
    phase_pool: np.ndarray,
    n_draws: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate without-replacement phase permutations for nested K subsets.

    The returned shape is [draw, system, pool slot]. For a given draw and
    system, the first K entries are the subset used at phase count K.
    """

    pool = np.asarray(phase_pool, dtype=np.int16)
    if pool.ndim != 2 or pool.shape[1] == 0:
        raise ValueError("phase_pool must have shape [systems, nonzero phases].")
    if n_draws <= 0:
        raise ValueError("n_draws must be positive.")
    draws = np.empty((n_draws, *pool.shape), dtype=np.int16)
    for draw in range(n_draws):
        for system in range(pool.shape[0]):
            draws[draw, system] = rng.permutation(pool[system])
    return draws


def prefix_endpoints_in_label_time(
    time_grid: np.ndarray,
    prefix_lengths: tuple[int, ...] | list[int],
    source_time_steps: int,
    label_horizon: float = 10000.0,
    source_time_end: float | None = None,
) -> np.ndarray:
    """Map stored prefix endpoints onto the completion-table time axis.

    Sampling3 stores a prefix of a longer, regularly sampled source axis. The
    completion table is expressed on ``label_horizon``; this scales the actual
    stored time coordinates by the inferred full-source duration.
    """

    grid = np.asarray(time_grid, dtype=np.float64).reshape(-1)
    finite = grid[np.isfinite(grid)]
    if finite.size == 0:
        raise ValueError("time_grid contains no finite values.")
    if source_time_steps < finite.size or source_time_steps < 1:
        raise ValueError("source_time_steps must cover the stored time grid.")
    deltas = np.diff(finite)
    positive = deltas[np.isfinite(deltas) & (deltas > 0.0)]
    start = float(finite[0])
    if positive.size:
        interval = float(np.median(positive))
        inferred_end = start + interval * (int(source_time_steps) - 1)
    else:
        inferred_end = float(finite[-1])
    end = inferred_end
    if source_time_end is not None and np.isfinite(source_time_end) and source_time_end >= finite[-1]:
        end = float(source_time_end)
    if end <= start:
        raise ValueError("The inferred source time span must be positive.")
    lengths = np.asarray(prefix_lengths, dtype=np.int64)
    if np.any(lengths < 1) or np.any(lengths > grid.size):
        raise ValueError("Every requested prefix length must be within the stored grid.")
    endpoints = grid[lengths - 1]
    return ((endpoints - start) / (end - start) * float(label_horizon)).astype(np.float64)


def common_preinstability_mask(
    failure_times: np.ndarray,
    phase_draws: np.ndarray,
    conditions: tuple[BudgetCondition, ...],
    endpoints: np.ndarray,
    pair_members: np.ndarray | None = None,
    singleton_variations: np.ndarray | None = None,
) -> np.ndarray:
    """Mark draw/system pairs event-free at every fixed-budget endpoint.

    ``failure_times`` has shape [systems, group_size]; right-censored
    trajectories should be represented as ``np.inf``. ``phase_draws`` has
    shape [draws, systems, pool_size]. An event at the endpoint is excluded.
    """

    failures = np.asarray(failure_times, dtype=np.float64)
    draws = np.asarray(phase_draws, dtype=np.int64)
    endpoint_values = np.asarray(endpoints, dtype=np.float64).reshape(-1)
    if failures.ndim != 2 or draws.ndim != 3:
        raise ValueError("failure_times and phase_draws must be rank 2 and rank 3.")
    if draws.shape[1] != failures.shape[0] or endpoint_values.size != len(conditions):
        raise ValueError("System or condition dimensions do not match.")
    singleton_ids = None if singleton_variations is None else np.asarray(singleton_variations, dtype=np.int64)
    if singleton_ids is not None and singleton_ids.shape != draws.shape[:2]:
        raise ValueError("singleton_variations must have shape [draws, systems].")
    if pair_members is None:
        if np.any(draws < 0) or np.any(draws >= failures.shape[1]):
            raise ValueError("phase_draws contains a variation index outside failure_times.")
        members = None
    else:
        members = np.asarray(pair_members, dtype=np.int64)
        if members.ndim != 2 or members.shape[1] != 2:
            raise ValueError("pair_members must have shape [pairs, 2].")
        if members.size and (members.min() < 0 or members.max() >= failures.shape[1]):
            raise ValueError("pair_members contains a variation ID outside failure_times.")
        if np.any(draws < 0) or np.any(draws >= members.shape[0]):
            raise ValueError("phase_draws contains a pair ID outside pair_members.")
    eligible = np.ones(draws.shape[:2], dtype=bool)
    system_index = np.arange(failures.shape[0])[None, :, None]
    for condition, endpoint in zip(conditions, endpoint_values):
        if condition.k > draws.shape[2]:
            raise ValueError("A condition requests more sampled units than present in phase_draws.")
        if members is not None and condition.k == 0.5:
            if singleton_ids is None:
                raise ValueError("A K=0.5 condition requires the sampled singleton variation IDs.")
            selected = failures[system_index[:, :, 0], singleton_ids]
            eligible &= selected > endpoint
            continue
        selected_ids = draws[:, :, : int(condition.k)]
        if members is None:
            selected = failures[system_index, selected_ids]
        else:
            variation_ids = members[selected_ids].reshape(*selected_ids.shape[:2], -1)
            selected = failures[system_index, variation_ids]
        eligible &= np.all(selected > endpoint, axis=-1)
    return eligible


class TCNBlock(nn.Module):
    def __init__(self, in_channels: int, hidden: int, dilation: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=3, padding=dilation, dilation=dilation),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=dilation, dilation=dilation),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
        )
        self.skip = nn.Identity() if in_channels == hidden else nn.Conv1d(in_channels, hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x)
        if y.shape[-1] != x.shape[-1]:
            left = (y.shape[-1] - x.shape[-1]) // 2
            y = y[..., left : left + x.shape[-1]]
        return y + self.skip(x)


class TCNEncoder(nn.Module):
    """The sampling3 high-resolution TCN backbone with temporal mean pooling."""

    def __init__(
        self,
        input_dim: int = 15,
        hidden: int = 160,
        levels: int = 5,
        dropout: float = 0.2,
    ):
        super().__init__()
        blocks: list[nn.Module] = []
        in_channels = input_dim
        for level in range(levels):
            blocks.append(TCNBlock(in_channels, hidden, 2**level, dropout))
            in_channels = hidden
        self.backbone = nn.Sequential(*blocks)
        self.hidden = int(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"TCNEncoder expects [batch, time, channels], got {tuple(x.shape)}")
        return self.backbone(x.transpose(1, 2)).mean(dim=-1)


class DeepSetsRegressor(nn.Module):
    """Shared trajectory TCN followed by permutation-invariant DeepSets pooling."""

    def __init__(
        self,
        input_dim: int = 15,
        tcn_hidden: int = 160,
        tcn_levels: int = 5,
        set_hidden: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.encoder = TCNEncoder(input_dim, tcn_hidden, tcn_levels, dropout)
        self.phi = nn.Sequential(
            nn.Linear(tcn_hidden, set_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(set_hidden, set_hidden),
            nn.GELU(),
        )
        self.rho = nn.Sequential(
            nn.Linear(set_hidden, set_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(set_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"DeepSetsRegressor expects [systems, K, T, C], got {tuple(x.shape)}")
        batch, phases, steps, channels = x.shape
        encoded = self.encoder(x.reshape(batch * phases, steps, channels))
        encoded = encoded.reshape(batch, phases, -1)
        system_repr = self.phi(encoded).mean(dim=1)
        return torch.sigmoid(self.rho(system_repr))


class PhaseTCNClassifier(nn.Module):
    """Per-trajectory stable-probability baseline; phase aggregation is external."""

    def __init__(
        self,
        input_dim: int = 15,
        hidden: int = 160,
        levels: int = 5,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.encoder = TCNEncoder(input_dim, hidden, levels, dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))


class PairDeepSetsRegressor(nn.Module):
    """TCN encoder with permutation-invariant pooling over matched phase pairs.

    Pair members are encoded independently with shared weights. Their mean and
    absolute difference form a symmetric pair embedding; DeepSets then pools
    those embeddings across the observed pairs.
    """

    def __init__(
        self,
        input_dim: int = 15,
        tcn_hidden: int = 160,
        tcn_levels: int = 5,
        set_hidden: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.encoder = TCNEncoder(input_dim, tcn_hidden, tcn_levels, dropout)
        self.phi = nn.Sequential(
            nn.Linear(2 * tcn_hidden + 1, set_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(set_hidden, set_hidden),
            nn.GELU(),
        )
        self.rho = nn.Sequential(
            nn.Linear(set_hidden, set_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(set_hidden, 1),
        )

    def forward(self, x: torch.Tensor, member_mask: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim != 5 or x.shape[2] != 2:
            raise ValueError(
                "PairDeepSetsRegressor expects [systems, pairs, 2, T, C], "
                f"got {tuple(x.shape)}"
            )
        batch, pairs, members, steps, channels = x.shape
        if member_mask is None:
            mask = torch.ones((batch, pairs, 2), dtype=x.dtype, device=x.device)
        else:
            if member_mask.shape != (batch, pairs, 2):
                raise ValueError("member_mask must have shape [systems, pairs, 2].")
            mask = member_mask.to(dtype=x.dtype, device=x.device)
            if torch.any((mask != 0) & (mask != 1)) or torch.any(mask.sum(dim=2) == 0):
                raise ValueError("Each represented pair must observe one or two members.")
        # Do not encode an unobserved partner: the TCN contains BatchNorm, so
        # even a masked embedding would otherwise affect the active member.
        flat_x = x.reshape(batch * pairs * members, steps, channels)
        observed = mask.reshape(-1).bool()
        encoded = x.new_zeros((batch * pairs * members, self.encoder.hidden))
        encoded[observed] = self.encoder(flat_x[observed]).to(dtype=encoded.dtype)
        encoded = encoded.reshape(batch, pairs, members, -1)
        count = mask.sum(dim=2, keepdim=True).clamp_min(1.0)
        mean = (encoded * mask.unsqueeze(-1)).sum(dim=2) / count
        first, second = encoded[:, :, 0], encoded[:, :, 1]
        complete_pair = (count == 2.0).to(dtype=encoded.dtype)
        difference = torch.abs(first - second) * complete_pair
        observed_fraction = count / 2.0
        pair_repr = torch.cat((mean, difference, observed_fraction), dim=-1)
        system_repr = self.phi(pair_repr).mean(dim=1)
        return torch.sigmoid(self.rho(system_repr))
