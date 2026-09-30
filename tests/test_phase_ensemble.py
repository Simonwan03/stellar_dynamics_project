from __future__ import annotations

import numpy as np
import torch

from tasks.common.case_split import create_case_split_manifest
from scripts.run_fixed_budget_sampling3 import bootstrap_delta_mae
from tasks.task_b.phase_ensemble import (
    DeepSetsRegressor,
    PairDeepSetsRegressor,
    BudgetCondition,
    budget_conditions,
    build_pair_partition,
    build_phase_partition,
    common_preinstability_mask,
    nested_phase_draws,
    pair_ids_to_variations,
    prefix_endpoints_in_label_time,
)


def test_fixed_budget_conditions_and_prefix_bounds() -> None:
    conditions = budget_conditions(512, (1, 2, 4, 8, 16), max_prefix=1024)
    assert [(item.k, item.t, item.budget) for item in conditions] == [
        (1, 512, 512), (2, 256, 512), (4, 128, 512), (8, 64, 512), (16, 32, 512)
    ]
    try:
        budget_conditions(2048, (1,), max_prefix=1024)
    except ValueError:
        pass
    else:
        raise AssertionError("a prefix longer than the stored history must be rejected")


def test_split_manifest_keeps_each_system_in_one_split() -> None:
    catalog = __import__("pandas").DataFrame(
        {
            "raw_case_id": np.arange(60),
            "stable_count": np.tile(np.arange(6), 10),
            "stable_fraction": np.tile(np.arange(6) / 5, 10),
            "stable_fraction_bin": np.tile(np.arange(6), 10),
        }
    )
    manifest = create_case_split_manifest(catalog, seed=42)
    assert manifest["raw_case_id"].is_unique
    assert manifest.groupby("raw_case_id")["split"].nunique().max() == 1
    assert set(manifest["split"]) == {"train", "val", "test"}


def test_fixed_phase_partition_is_balanced_disjoint_and_reproducible() -> None:
    ids = np.array([3, 20, 41], dtype=np.int64)
    first = build_phase_partition(ids, seed=42)
    second = build_phase_partition(ids, seed=42)
    assert np.array_equal(first, second)
    for phases in first:
        candidate = phases[:16]
        heldout = phases[16:]
        assert len(np.unique(candidate)) == 16
        assert len(np.unique(heldout)) == 16
        assert set(candidate).isdisjoint(set(heldout))
        assert set(candidate) | set(heldout) == set(range(32))


def test_nested_phase_draws_sample_without_replacement() -> None:
    pool = np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int16)
    draws = nested_phase_draws(pool, 7, np.random.default_rng(123))
    assert draws.shape == (7, 2, 4)
    for draw in draws:
        for system_draw, system_pool in zip(draw, pool):
            assert len(np.unique(system_draw)) == len(system_pool)
            assert set(system_draw) == set(system_pool)
            assert set(system_draw[:2]).issubset(set(system_draw[:3]))


def test_deepsets_is_invariant_to_phase_permutation() -> None:
    torch.manual_seed(5)
    model = DeepSetsRegressor(input_dim=3, tcn_hidden=8, tcn_levels=2, set_hidden=12, dropout=0.0)
    model.eval()
    x = torch.randn(2, 4, 32, 3)
    permutation = torch.tensor([2, 0, 3, 1])
    with torch.no_grad():
        first = model(x)
        second = model(x[:, permutation])
    assert torch.allclose(first, second, atol=1e-6, rtol=1e-6)
    assert torch.all((first >= 0.0) & (first <= 1.0))


def test_sampling3_prefix_endpoints_map_to_completion_time_axis() -> None:
    grid = np.arange(1024, dtype=np.float64) * 10000.0
    values = prefix_endpoints_in_label_time(grid, [32, 64, 128, 256, 512], 10001)
    assert np.allclose(values, [31, 63, 127, 255, 511])


def test_preinstability_uses_strict_event_time_and_handles_censoring() -> None:
    conditions = (BudgetCondition(1, 512), BudgetCondition(2, 256))
    failures = np.array([[np.inf, 300.0, 700.0], [255.0, np.inf, np.inf]])
    draws = np.array(
        [
            [[0, 1, 2], [1, 0, 2]],
            [[2, 0, 1], [0, 1, 2]],
        ],
        dtype=np.int16,
    )
    eligible = common_preinstability_mask(failures, draws, conditions, np.array([100.0, 256.0]))
    assert eligible.shape == (2, 2)
    # System 1's event at 255 is after T=100 but not after T=256 for its K=2 draw.
    assert not eligible[0, 1]
    # A phase censored at the 10,000 horizon is represented as no observed event.
    assert eligible[1, 0]



def test_paired_bootstrap_reports_small_two_sided_p_for_clear_gain() -> None:
    target = np.zeros(64, dtype=np.float64)
    prediction_k1 = np.full(64, 0.5, dtype=np.float64)
    prediction_k = np.full(64, 0.1, dtype=np.float64)
    estimate, low, high, p_value = bootstrap_delta_mae(
        target,
        prediction_k1,
        prediction_k,
        2000,
        np.random.default_rng(42),
    )
    assert np.isclose(estimate, 0.4)
    assert low > 0.0 and high > 0.0
    assert p_value <= 0.001


def test_pair_budget_conditions_count_both_trajectory_members() -> None:
    conditions = budget_conditions(512, (1, 2, 4, 8), max_prefix=1024, phases_per_unit=2)
    assert [(item.k, item.t, item.budget) for item in conditions] == [
        (1, 256, 512), (2, 128, 512), (4, 64, 512), (8, 32, 512)
    ]


def test_pair_partition_and_variation_mapping_are_balanced() -> None:
    ids = np.array([3, 20, 41], dtype=np.int64)
    first = build_pair_partition(ids, seed=42)
    second = build_pair_partition(ids, seed=42)
    members = np.column_stack((np.arange(16), np.arange(16) + 16))
    assert np.array_equal(first, second)
    for pairs in first:
        candidate = pairs[:8]
        heldout = pairs[8:]
        assert set(candidate).isdisjoint(set(heldout))
        assert set(candidate) | set(heldout) == set(range(16))
        variations = pair_ids_to_variations(pairs, members)
        assert variations.shape == (16, 2)
        assert np.all(variations[:, 1] - variations[:, 0] == 16)


def test_preinstability_checks_both_members_of_each_pair() -> None:
    pair_members = np.array([[0, 16], [1, 17]], dtype=np.int64)
    failures = np.full((1, 32), np.inf)
    failures[0, 16] = 100.0
    draws = np.array([[[0, 1]]], dtype=np.int16)
    conditions = (BudgetCondition(1, 64, unit_size=2), BudgetCondition(2, 32, unit_size=2))
    eligible = common_preinstability_mask(
        failures, draws, conditions, np.array([50.0, 50.0]), pair_members=pair_members
    )
    assert eligible.tolist() == [[True]]
    eligible = common_preinstability_mask(
        failures, draws, conditions, np.array([100.0, 100.0]), pair_members=pair_members
    )
    assert eligible.tolist() == [[False]]


def test_pair_deepsets_is_invariant_to_pair_and_member_order() -> None:
    torch.manual_seed(6)
    model = PairDeepSetsRegressor(input_dim=3, tcn_hidden=8, tcn_levels=2, set_hidden=12, dropout=0.0)
    model.eval()
    x = torch.randn(2, 3, 2, 32, 3)
    pair_permutation = torch.tensor([2, 0, 1])
    with torch.no_grad():
        first = model(x)
        permuted_pairs = model(x[:, pair_permutation])
        swapped_members = model(x[:, :, [1, 0]])
    assert torch.allclose(first, permuted_pairs, atol=1e-6, rtol=1e-6)
    assert torch.allclose(first, swapped_members, atol=1e-6, rtol=1e-6)
    assert torch.all((first >= 0.0) & (first <= 1.0))


def test_half_pair_budget_is_one_trajectory():
    conditions = budget_conditions(256, (0.5, 1), max_prefix=512, phases_per_unit=2)
    assert [(item.k, item.t, item.budget) for item in conditions] == [
        (0.5, 256, 256), (1, 128, 256)
    ]


def test_half_pair_preinstability_checks_only_selected_member():
    pair_members = np.array([[0, 16]], dtype=np.int64)
    failures = np.full((1, 32), np.inf)
    failures[0, 16] = 40.0
    draws = np.array([[[0]]], dtype=np.int16)
    condition = (BudgetCondition(0.5, 32, unit_size=2),)
    left = common_preinstability_mask(
        failures, draws, condition, np.array([50.0]), pair_members=pair_members,
        singleton_variations=np.array([[0]], dtype=np.int64),
    )
    right = common_preinstability_mask(
        failures, draws, condition, np.array([50.0]), pair_members=pair_members,
        singleton_variations=np.array([[16]], dtype=np.int64),
    )
    assert left.tolist() == [[True]]
    assert right.tolist() == [[False]]


def test_pair_deepsets_single_member_mask_ignores_unobserved_partner():
    torch.manual_seed(17)
    model = PairDeepSetsRegressor(input_dim=3, tcn_hidden=8, tcn_levels=2, set_hidden=12, dropout=0.0)
    model.eval()
    x = torch.randn(3, 1, 2, 32, 3)
    mask = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]], [[1.0, 0.0]]])
    changed = x.clone()
    changed[0, 0, 1] += 1000.0
    changed[1, 0, 0] -= 1000.0
    changed[2, 0, 1] *= -1000.0
    with torch.no_grad():
        first = model(x, mask)
        second = model(changed, mask)
    assert torch.allclose(first, second, atol=1e-6, rtol=1e-6)
