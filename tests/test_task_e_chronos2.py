from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from tasks.task_e import main as task_e_main


class FakeChronos2Pipeline:
    last_inputs: np.ndarray | None = None

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs: object) -> "FakeChronos2Pipeline":
        del model_id, kwargs
        return cls()

    def predict(
        self,
        inputs: np.ndarray,
        *,
        prediction_length: int,
        batch_size: int,
        context_length: int,
    ) -> list[torch.Tensor]:
        del batch_size, context_length
        FakeChronos2Pipeline.last_inputs = np.asarray(inputs)
        batch, channels, _ = inputs.shape
        return [
            torch.arange(channels * 4 * prediction_length, dtype=torch.float32).reshape(
                channels, 4, prediction_length
            )
            for _ in range(batch)
        ]


def chronos2_args() -> SimpleNamespace:
    return SimpleNamespace(
        model="chronos",
        representation="orbital15",
        context_len=5,
        pred_len=4,
        foundation_batch_size=2,
        hf_local_files_only=True,
        chronos_model_id="amazon/chronos-2",
        chronos_variant="chronos2",
        chronos_num_samples=2,
        timesfm_model_id="unused",
        panda_model_id="unused",
        torch_dtype="float32",
    )


class Chronos2PipelineTest(unittest.TestCase):
    def test_unified_entry_uses_native_multivariate_path(self) -> None:
        args = chronos2_args()
        with patch("chronos.Chronos2Pipeline", FakeChronos2Pipeline):
            predictor, multivariate = task_e_main.build_foundation_predictor(args, torch.device("cpu"))
            prediction = predictor(np.zeros((2, 5, 3), dtype=np.float32), 4)

        self.assertTrue(multivariate)
        self.assertEqual(prediction.shape, (2, 4, 3))
        np.testing.assert_array_equal(FakeChronos2Pipeline.last_inputs.shape, (2, 3, 5))


if __name__ == "__main__":
    unittest.main()
