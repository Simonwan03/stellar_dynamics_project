"""Check system separation for the forecasting split used in the paper."""

import numpy as np

from tasks.task_e.data import split_grouped_trajectory_indices


def test_task_e_split_keeps_all_variations_of_each_system_together():
    # Omit some variations to exercise the completion/capping path as well.
    eligible = np.array([system * 32 + variation for system in range(12) for variation in (0, 1, 16, 31)])
    splits = split_grouped_trajectory_indices(eligible, group_size=32, seed=42, train_frac=0.7, val_frac=0.15)
    assert np.array_equal(np.sort(np.concatenate(list(splits.values()))), eligible)
    system_sets = {name: set(indices // 32) for name, indices in splits.items()}
    assert all(system_sets[name] for name in system_sets)
    assert not (system_sets["train"] & system_sets["val"])
    assert not (system_sets["train"] & system_sets["test"])
    assert not (system_sets["val"] & system_sets["test"])
