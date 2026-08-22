from __future__ import annotations

import unittest

import torch

from tasks.task_e.models import ITransformerForecaster, build_supervised_model
from tasks.task_e.main import build_arg_parser


class ITransformerForecasterTest(unittest.TestCase):
    def _model(self) -> ITransformerForecaster:
        model = build_supervised_model(
            "itransformer",
            num_features=5,
            context_len=24,
            pred_len=8,
            hidden_dim=16,
            num_layers=2,
            kernel_size=3,
            dropout=0.0,
            patch_len=4,
            patch_stride=2,
            d_model=16,
            n_heads=4,
        )
        self.assertIsInstance(model, ITransformerForecaster)
        return model

    def test_forward_shape_and_backward(self) -> None:
        model = self._model()
        context = torch.randn(3, 24, 5, requires_grad=True)
        prediction = model(context)

        self.assertEqual(prediction.shape, (3, 8, 5))
        self.assertTrue(torch.isfinite(prediction).all())
        prediction.square().mean().backward()
        self.assertIsNotNone(context.grad)
        self.assertTrue(torch.isfinite(context.grad).all())

    def test_instance_normalization_is_affine_equivariant(self) -> None:
        model = self._model().eval()
        context = torch.randn(2, 24, 5)
        scale = torch.tensor([0.5, 1.5, 2.0, 3.0, 4.0]).view(1, 1, 5)
        offset = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]).view(1, 1, 5)

        with torch.no_grad():
            base = model(context)
            transformed = model(context * scale + offset)

        torch.testing.assert_close(
            transformed,
            base * scale + offset,
            rtol=2e-4,
            atol=2e-4,
        )

    def test_rejects_wrong_input_shape(self) -> None:
        model = self._model()
        with self.assertRaisesRegex(ValueError, "expected"):
            model(torch.randn(2, 23, 5))

    def test_server_training_defaults(self) -> None:
        args = build_arg_parser().parse_args(["--output-dir", "/tmp/task_e_test"])
        self.assertEqual(args.context_len, 512)
        self.assertEqual(args.pred_len, 128)
        self.assertEqual(args.d_model, 256)
        self.assertEqual(args.n_heads, 8)
        self.assertEqual(args.num_layers, 2)
        self.assertEqual(args.batch_size, 512)
        self.assertEqual(args.num_workers, 8)
        self.assertEqual(args.epochs, 30)
        self.assertAlmostEqual(args.lr, 3e-4)
        self.assertAlmostEqual(args.weight_decay, 1e-4)
        self.assertEqual(args.patience, 7)
        self.assertAlmostEqual(args.grad_clip, 1.0)
        self.assertAlmostEqual(args.dropout, 0.1)
        self.assertTrue(args.amp)

        args = build_arg_parser().parse_args(
            ["--output-dir", "/tmp/task_e_test", "--no-amp"]
        )
        self.assertFalse(args.amp)


if __name__ == "__main__":
    unittest.main()
