from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from tasks.task_e import main as task_e_main


class FakePanda(nn.Module):
    def __init__(self, configured_channels: int = 1) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            num_input_channels=configured_channels,
            context_length=8,
            prediction_length=4,
        )

    def forward(
        self,
        *,
        past_values: torch.Tensor,
        past_observed_mask: torch.Tensor,
        future_values: torch.Tensor | None,
    ) -> SimpleNamespace:
        del past_observed_mask, future_values
        batch_size, _, channels = past_values.shape
        prediction = torch.zeros(
            batch_size,
            self.config.prediction_length,
            channels,
            dtype=past_values.dtype,
            device=past_values.device,
        )
        return SimpleNamespace(prediction_outputs=prediction)


def foundation_args(representation: str) -> SimpleNamespace:
    return SimpleNamespace(
        model="panda_zero_shot",
        representation=representation,
        timesfm_model_id="unused",
        context_len=8,
        pred_len=4,
        foundation_batch_size=2,
        hf_local_files_only=True,
        chronos_model_id="unused",
        chronos_num_samples=1,
        panda_model_id="fake-panda",
        torch_dtype="float32",
    )


def adapter_args(representation: str) -> SimpleNamespace:
    return SimpleNamespace(
        representation=representation,
        panda_model_id="fake-panda",
        hf_local_files_only=True,
        context_len=6,
        pred_len=3,
        adapter_hidden_dim=None,
        adapter_dropout=0.0,
        freeze_panda=True,
    )


class PandaChannelSupportTest(unittest.TestCase):
    def test_panda_zero_shot_accepts_all_task_e_representations(self) -> None:
        for representation in ("orbital15", "raw_state18", "trajectory"):
            with self.subTest(representation=representation):
                sentinel = object()
                with patch.object(
                    task_e_main,
                    "build_panda_predictor",
                    return_value=sentinel,
                ):
                    predictor, multivariate = task_e_main.build_foundation_predictor(
                        foundation_args(representation),
                        torch.device("cpu"),
                    )

                self.assertIs(predictor, sentinel)
                self.assertTrue(multivariate)

    def test_panda_adapter_preserves_raw_state18_channels(self) -> None:
        with patch("tasks.task_e.foundation.load_panda_model", return_value=FakePanda()):
            model = task_e_main.build_panda_adapter(
                adapter_args("raw_state18"),
                num_features=18,
                device=torch.device("cpu"),
            )

        output = model(torch.randn(2, 6, 18))
        self.assertEqual(output.shape, (2, 3, 18))

    def test_panda_gru_decoder_supports_another_output_length(self) -> None:
        args = adapter_args("orbital15")
        args.adapter_decoder = "gru"
        args.adapter_gru_layers = 2
        with patch("tasks.task_e.foundation.load_panda_model", return_value=FakePanda()):
            model = task_e_main.build_panda_adapter(
                args,
                num_features=15,
                device=torch.device("cpu"),
            )

        context = torch.randn(2, 6, 15)
        self.assertEqual(model(context).shape, (2, 3, 15))
        self.assertEqual(model(context, output_len=7).shape, (2, 7, 15))

    def test_panda_adapter_rejects_a_genuinely_fixed_channel_mismatch(self) -> None:
        with (
            patch(
                "tasks.task_e.foundation.load_panda_model",
                return_value=FakePanda(configured_channels=15),
            ),
            self.assertRaisesRegex(ValueError, "raw_state18 produced 18"),
        ):
            task_e_main.build_panda_adapter(
                adapter_args("raw_state18"),
                num_features=18,
                device=torch.device("cpu"),
            )


if __name__ == "__main__":
    unittest.main()
