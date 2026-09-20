from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from tasks.task_e.cache import (
    build_preparation_cache_signature,
    load_prepared_cache,
    save_prepared_cache,
)
from tasks.task_e.data import SequenceSource, load_completion_metadata, prepare_forecasting_data


class PreparedCacheTest(unittest.TestCase):
    def test_round_trip_and_signature_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_path = root / "results_batch_0_sampling_2.hdf5"
            completion_path = root / "completed.txt"
            cache_path = root / "cache"
            rows, time_steps = 8, 10
            with h5py.File(data_path, "w") as handle:
                values = np.arange(rows * time_steps, dtype=np.float32).reshape(rows, time_steps, 1) / 10.0
                for offset, name in enumerate(("aratios", "mratios", "eccs", "incs", "APs", "LANs")):
                    handle.create_dataset(name, data=values + offset)
                handle.create_dataset("times_complete", data=np.arange(time_steps, dtype=np.float64)[None, :])
            np.savetxt(
                completion_path,
                np.column_stack(
                    [
                        np.arange(rows),
                        np.zeros(rows),
                        np.full(rows, 9.0),
                        np.zeros(rows),
                        np.zeros(rows),
                    ]
                ),
                fmt="%.1f",
            )

            settings = dict(
                data_glob=str(root / "results_batch_*_sampling_2.hdf5"),
                representation="orbital15",
                context_len=2,
                pred_len=2,
                stride=2,
                split_seed=42,
                train_frac=0.7,
                val_frac=0.15,
                group_size=1,
                completion_table=completion_path,
                require_complete_systems=True,
                max_trajectories=0,
                max_variations_per_system=0,
                max_train_variations_per_system=0,
                max_windows_per_trajectory=0,
                window_sampling="sequential",
                drop_nan_windows=True,
                max_scaler_trajectories=0,
            )
            signature = build_preparation_cache_signature(**settings)
            prepared = prepare_forecasting_data(verbose=False, **settings)
            try:
                save_prepared_cache(prepared, cache_path, signature)
            finally:
                for dataset in (prepared.train_dataset, prepared.val_dataset, prepared.test_dataset):
                    dataset.close()

            loaded = load_prepared_cache(cache_path, signature, verbose=False)
            try:
                self.assertEqual(len(loaded.train_dataset), 20)
                self.assertEqual(len(loaded.val_dataset), 4)
                self.assertEqual(len(loaded.test_dataset), 8)
                np.testing.assert_allclose(loaded.scaler.mean, prepared.scaler.mean)
                np.testing.assert_array_equal(
                    loaded.test_dataset.record_starts,
                    np.load(cache_path / "records.npz")["test_record_starts"],
                )
            finally:
                for dataset in (loaded.train_dataset, loaded.val_dataset, loaded.test_dataset):
                    dataset.close()

            mismatched = dict(signature)
            mismatched["pred_len"] = 3
            with self.assertRaises(ValueError):
                load_prepared_cache(cache_path, mismatched, verbose=False)

    def test_prefix_time_metadata_preserves_full_completion_scale(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result_path = root / "results_batch_00_sampling_3.hdf5"
            with h5py.File(result_path, "w") as handle:
                handle.create_dataset(
                    "times_complete",
                    data=np.tile(np.asarray([0.0, 10.0, 20.0, 30.0]), (4, 1)),
                )
                handle.attrs["source_time_steps"] = 11
                handle.attrs["source_time_end"] = 100.0
            completion_path = root / "completed.txt"
            np.savetxt(
                completion_path,
                np.asarray(
                    [
                        [0, 0, 1, 0, 0],
                        [1, 0, 2, 0, 0],
                        [2, 0, 10, 0, 0],
                        [3, 0, 10, 0, 0],
                    ],
                    dtype=np.float64,
                ),
            )
            source = SequenceSource(
                path=result_path,
                row_count=4,
                time_steps=4,
                resolved_keys={},
                step_shapes={},
            )

            valid, lengths, _ = load_completion_metadata(
                completion_path,
                [source],
                group_size=1,
                require_complete_systems=False,
            )

            np.testing.assert_array_equal(valid, np.ones(4, dtype=bool))
            np.testing.assert_array_equal(lengths, np.asarray([2, 3, 4, 4], dtype=np.int64))


    def test_corrupt_prefix_time_metadata_falls_back_to_reconstructed_end(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result_path = root / "results_batch_00_sampling_3.hdf5"
            with h5py.File(result_path, "w") as handle:
                handle.create_dataset(
                    "times_complete",
                    data=np.tile(np.asarray([0.0, 10.0, 20.0, 30.0]), (4, 1)),
                )
                handle.attrs["source_time_steps"] = 11
                handle.attrs["source_time_end"] = 1.5
            completion_path = root / "completed.txt"
            np.savetxt(
                completion_path,
                np.asarray(
                    [
                        [0, 0, 1, 0, 0],
                        [1, 0, 2, 0, 0],
                        [2, 0, 10, 0, 0],
                        [3, 0, 10, 0, 0],
                    ],
                    dtype=np.float64,
                ),
            )
            source = SequenceSource(
                path=result_path,
                row_count=4,
                time_steps=4,
                resolved_keys={},
                step_shapes={},
            )

            valid, lengths, _ = load_completion_metadata(
                completion_path,
                [source],
                group_size=1,
                require_complete_systems=False,
            )

            np.testing.assert_array_equal(valid, np.ones(4, dtype=bool))
            np.testing.assert_array_equal(lengths, np.asarray([2, 3, 4, 4], dtype=np.int64))


if __name__ == "__main__":
    unittest.main()
