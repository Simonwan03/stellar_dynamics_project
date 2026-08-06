from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import h5py
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tasks.task_e.panda_dependency import load_panda_pretrained  # noqa: E402

DEFAULT_TRAJECTORY_KEYS = ("positions_complete", "velocities_complete")
DEFAULT_ORBITAL_FEATURES = ("aratios", "mratios", "eccs", "incs", "APs", "LANs")
LOG1P_FEATURES = {"aratios", "mratios"}
ORBITAL15_ANGLE_FEATURES = {"APs", "LANs"}


@dataclass(frozen=True)
class Source:
    path: Path
    row_count: int
    time_steps: int
    shapes: dict[str, tuple[int, ...]]


@dataclass
class EvalData:
    context: np.ndarray
    target: np.ndarray
    channel_names: list[str]
    window_metadata: list[dict[str, Any]]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate pretrained TimesFM, Chronos, Panda, and PatchTST on direct three-body trajectory forecasting. "
            "Univariate models treat each channel independently; Panda consumes the full multivariate window."
        )
    )
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "dataset_v3")
    parser.add_argument("--sampling-id", type=int, default=2)
    parser.add_argument("--representation", choices=["trajectory", "orbital15"], default="trajectory")
    parser.add_argument("--trajectory-keys", type=str, default=",".join(DEFAULT_TRAJECTORY_KEYS))
    parser.add_argument("--orbital-features", type=str, default=",".join(DEFAULT_ORBITAL_FEATURES))
    parser.add_argument("--context-len", type=int, default=512)
    parser.add_argument("--pred-len", type=int, default=128)
    parser.add_argument("--num-windows", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--models", type=str, default="naive,timesfm,chronos,patchtst")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs" / "foundation_trajectory_baselines")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--torch-dtype", choices=["auto", "float32", "bfloat16", "float16"], default="auto")
    parser.add_argument(
        "--hf-local-files-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Load Hugging Face checkpoints from the local cache only. This avoids long network "
            "retry delays on offline machines; pass --no-hf-local-files-only to allow downloads."
        ),
    )
    parser.add_argument("--freq", type=int, default=0, help="TimesFM frequency bucket. 0 is high-frequency/default.")
    parser.add_argument("--chronos-model-id", type=str, default="amazon/chronos-t5-small")
    parser.add_argument("--chronos-num-samples", type=int, default=20)
    parser.add_argument("--panda-model-id", type=str, default="GilpinLab/panda-72M")
    parser.add_argument(
        "--timesfm-model-id",
        type=str,
        default="google/timesfm-2.5-200m-pytorch",
        help=(
            "TimesFM checkpoint. The current official PyTorch API exposed by recent timesfm installs "
            "expects a 2.5 checkpoint such as google/timesfm-2.5-200m-pytorch."
        ),
    )
    parser.add_argument("--patchtst-model-id", type=str, default="ibm/patchtst-etth1-forecasting")
    parser.add_argument(
        "--patchtst-fallback-model-ids",
        type=str,
        default="ibm-granite/granite-timeseries-patchtst",
        help="Comma-separated fallback PatchTST checkpoints to try if the primary checkpoint is gated or unavailable.",
    )
    parser.add_argument("--log-every-batches", type=int, default=10)
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    trajectory_keys = [item.strip() for item in args.trajectory_keys.split(",") if item.strip()]
    orbital_features = [item.strip() for item in args.orbital_features.split(",") if item.strip()]
    if args.representation == "trajectory" and not trajectory_keys:
        raise ValueError("No trajectory keys were selected.")
    if args.representation == "orbital15" and not orbital_features:
        raise ValueError("No orbital features were selected.")

    data = load_eval_data(
        dataset_dir=args.dataset_dir,
        sampling_id=args.sampling_id,
        representation=args.representation,
        trajectory_keys=trajectory_keys,
        orbital_features=orbital_features,
        context_len=args.context_len,
        pred_len=args.pred_len,
        num_windows=args.num_windows,
        seed=args.seed,
    )
    save_json(
        args.output_dir / "config.json",
        {
            "dataset_dir": str(args.dataset_dir),
            "sampling_id": args.sampling_id,
            "representation": args.representation,
            "trajectory_keys": trajectory_keys,
            "orbital_features": orbital_features,
            "context_len": args.context_len,
            "pred_len": args.pred_len,
            "num_windows": args.num_windows,
            "num_channels": len(data.channel_names),
            "channel_names": data.channel_names,
            "device": device,
            "models": parse_name_list(args.models),
            "hf_local_files_only": args.hf_local_files_only,
            "timesfm_model_id": args.timesfm_model_id,
            "chronos_model_id": args.chronos_model_id,
            "panda_model_id": args.panda_model_id,
            "patchtst_model_id": args.patchtst_model_id,
        },
    )

    series_context, series_target = flatten_windows_to_series(data.context, data.target)
    metric_scale = compute_metric_scale(data.context)
    all_rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    model_names = parse_name_list(args.models)
    cleanup_previous_outputs(args.output_dir, model_names)

    for model_name in model_names:
        started = time.perf_counter()
        try:
            predictor = build_predictor(model_name, args, device)
            if uses_multivariate_context(model_name, args):
                print(
                    f"Running {model_name} on {data.context.shape[0]} multivariate windows "
                    f"with {len(data.channel_names)} channels...",
                    flush=True,
                )
                prediction_windows = run_batched_prediction_multivariate(
                    predictor=predictor,
                    context=data.context,
                    pred_len=args.pred_len,
                    batch_size=args.batch_size,
                    log_every_batches=args.log_every_batches,
                    label=model_name,
                )
            else:
                print(f"Running {model_name} on {series_context.shape[0]} univariate series...", flush=True)
                prediction = run_batched_prediction(
                    predictor=predictor,
                    context=series_context,
                    pred_len=args.pred_len,
                    batch_size=args.batch_size,
                    log_every_batches=args.log_every_batches,
                    label=model_name,
                )
                prediction_windows = unflatten_series_to_windows(prediction, data.context.shape[0], len(data.channel_names))
            metrics = compute_metrics(prediction_windows, data.target, metric_scale)
            metrics["model"] = model_name
            metrics["runtime_sec"] = time.perf_counter() - started
            all_rows.append(metrics)
            save_json(args.output_dir / f"{model_name}_metrics.json", metrics)
            save_horizon_csv(args.output_dir / f"{model_name}_horizon_metrics.csv", metrics)
            save_example_csv(
                args.output_dir / f"{model_name}_example_prediction.csv",
                prediction=prediction_windows,
                target=data.target,
                channel_names=data.channel_names,
                window_metadata=data.window_metadata,
            )
            print(
                f"{model_name}: raw_mse={metrics['raw_mse']:.6g} raw_mae={metrics['raw_mae']:.6g} "
                f"z_mse={metrics['z_mse']:.6g} runtime={metrics['runtime_sec']:.1f}s",
                flush=True,
            )
        except Exception as exc:
            failure = {
                "model": model_name,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            failures.append(failure)
            save_json(args.output_dir / f"{model_name}_error.json", failure)
            print(f"{model_name} failed: {exc}", file=sys.stderr, flush=True)
            if args.fail_fast:
                raise

    save_csv(args.output_dir / "summary_metrics.csv", all_rows)
    save_json(args.output_dir / "failures.json", failures)
    print(f"Wrote foundation-model evaluation outputs to {args.output_dir}", flush=True)


def load_eval_data(
    dataset_dir: Path,
    sampling_id: int,
    representation: str,
    trajectory_keys: list[str],
    orbital_features: list[str],
    context_len: int,
    pred_len: int,
    num_windows: int,
    seed: int,
) -> EvalData:
    selected_keys = resolve_selected_keys(representation, trajectory_keys, orbital_features)
    sources = inspect_sources(dataset_dir, sampling_id, selected_keys)
    rng = np.random.default_rng(seed)
    total_trajectories = sum(source.row_count for source in sources)
    total_window = context_len + pred_len
    starts_per_trajectory = sources[0].time_steps - total_window + 1
    if starts_per_trajectory <= 0:
        raise ValueError(f"Trajectory length is shorter than context_len + pred_len = {total_window}.")

    channel_names = build_channel_names(sources[0], selected_keys, representation)
    num_channels = len(channel_names)
    context = np.empty((num_windows, context_len, num_channels), dtype=np.float32)
    target = np.empty((num_windows, pred_len, num_channels), dtype=np.float32)
    metadata: list[dict[str, Any]] = []

    source_offsets = np.cumsum([0] + [source.row_count for source in sources])
    handles: dict[int, h5py.File] = {}
    try:
        for sample_idx in range(num_windows):
            global_idx = int(rng.integers(0, total_trajectories))
            source_id = int(np.searchsorted(source_offsets, global_idx, side="right") - 1)
            local_idx = int(global_idx - source_offsets[source_id])
            start = int(rng.integers(0, starts_per_trajectory))
            stop = start + total_window
            source = sources[source_id]

            blocks = []
            if source_id not in handles:
                handles[source_id] = h5py.File(source.path, "r")
            handle = handles[source_id]
            for key in selected_keys:
                raw = np.asarray(handle[key][local_idx, start:stop], dtype=np.float32)
                transformed = transform_feature_array(key, raw, representation)
                blocks.append(transformed.reshape(total_window, -1))
            window = np.concatenate(blocks, axis=-1)
            window = np.nan_to_num(window, nan=0.0, posinf=0.0, neginf=0.0)
            context[sample_idx] = window[:context_len]
            target[sample_idx] = window[context_len:]
            metadata.append(
                {
                    "source_path": str(source.path),
                    "source_row": local_idx,
                    "global_row": global_idx,
                    "start": start,
                    "stop": stop,
                }
            )
    finally:
        for handle in handles.values():
            handle.close()

    return EvalData(context=context, target=target, channel_names=channel_names, window_metadata=metadata)


def resolve_selected_keys(representation: str, trajectory_keys: list[str], orbital_features: list[str]) -> list[str]:
    if representation == "trajectory":
        return trajectory_keys
    return [feature if feature.endswith("_complete") else f"{feature}_complete" for feature in orbital_features]


def inspect_sources(dataset_dir: Path, sampling_id: int, selected_keys: list[str]) -> list[Source]:
    paths = sorted(dataset_dir.glob(f"results_batch_*_sampling_{sampling_id}.hdf5"))
    if not paths:
        raise FileNotFoundError(f"No result shards found in {dataset_dir} for sampling id {sampling_id}.")

    sources = []
    for path in paths:
        with h5py.File(path, "r") as handle:
            if not all(key in handle for key in selected_keys):
                continue
            row_count = None
            time_steps = None
            shapes: dict[str, tuple[int, ...]] = {}
            for key in selected_keys:
                dataset = handle[key]
                if dataset.ndim < 3:
                    raise ValueError(f"Expected {key} in {path} to have shape [N, T, ...], got {dataset.shape}.")
                if row_count is None:
                    row_count = int(dataset.shape[0])
                    time_steps = int(dataset.shape[1])
                elif row_count != int(dataset.shape[0]) or time_steps != int(dataset.shape[1]):
                    raise ValueError(f"Inconsistent shape for {key} in {path}: {dataset.shape}.")
                shapes[key] = tuple(int(dim) for dim in dataset.shape[2:])
            assert row_count is not None and time_steps is not None
            sources.append(Source(path=path, row_count=row_count, time_steps=time_steps, shapes=shapes))
    if not sources:
        raise FileNotFoundError(f"No result shards contain all keys {selected_keys}.")
    return sources


def build_channel_names(source: Source, selected_keys: list[str], representation: str) -> list[str]:
    coord_labels = ["x", "y", "z"]
    names = []
    for key in selected_keys:
        feature_name = key.removesuffix("_complete")
        prefix = "pos" if key == "positions_complete" else "vel" if key == "velocities_complete" else key
        shape = source.shapes[key]
        if representation == "orbital15":
            raw_dim = int(np.prod(shape))
            if feature_name in ORBITAL15_ANGLE_FEATURES:
                names.extend([f"{feature_name}_sin_{idx}" for idx in range(raw_dim)])
                names.extend([f"{feature_name}_cos_{idx}" for idx in range(raw_dim)])
            else:
                names.extend([f"{feature_name}_{idx}" for idx in range(raw_dim)])
            continue
        body_count = shape[0] if len(shape) >= 1 else 1
        coord_count = shape[1] if len(shape) >= 2 else 1
        for body_idx in range(body_count):
            for coord_idx in range(coord_count):
                suffix = coord_labels[coord_idx] if coord_idx < len(coord_labels) else str(coord_idx)
                names.append(f"{prefix}_body{body_idx}_{suffix}")
    return names


def transform_feature_array(key: str, raw: np.ndarray, representation: str) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.float32)
    if representation == "trajectory":
        return raw
    feature_name = key.removesuffix("_complete")
    if representation == "orbital15":
        if feature_name in ORBITAL15_ANGLE_FEATURES:
            flat = raw.reshape(raw.shape[0], -1)
            return np.concatenate([np.sin(flat), np.cos(flat)], axis=-1).astype(np.float32)
        return raw
    if feature_name in LOG1P_FEATURES:
        with np.errstate(invalid="ignore"):
            return np.log1p(raw).astype(np.float32)
    return raw


def flatten_windows_to_series(context: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    num_windows, context_len, num_channels = context.shape
    pred_len = target.shape[1]
    return (
        context.transpose(0, 2, 1).reshape(num_windows * num_channels, context_len),
        target.transpose(0, 2, 1).reshape(num_windows * num_channels, pred_len),
    )


def unflatten_series_to_windows(prediction: np.ndarray, num_windows: int, num_channels: int) -> np.ndarray:
    pred_len = prediction.shape[1]
    return prediction.reshape(num_windows, num_channels, pred_len).transpose(0, 2, 1)


def compute_metric_scale(context: np.ndarray) -> dict[str, np.ndarray]:
    mean = context.mean(axis=(0, 1), keepdims=True)
    std = context.std(axis=(0, 1), keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}


def build_predictor(model_name: str, args: argparse.Namespace, device: str) -> Callable[[np.ndarray, int], np.ndarray]:
    if model_name == "naive":
        return lambda batch, pred_len: np.repeat(batch[:, -1:], pred_len, axis=1)
    if model_name == "timesfm":
        return build_timesfm_predictor(args, device)
    if model_name == "chronos":
        if is_chronos2_model(args.chronos_model_id):
            return build_chronos2_predictor(args, device)
        return build_chronos_predictor(args, device)
    if model_name == "panda":
        return build_panda_predictor(args, device)
    if model_name == "patchtst":
        return build_patchtst_predictor(args, device)
    raise ValueError(f"Unknown model '{model_name}'. Supported: naive,timesfm,chronos,panda,patchtst")


def uses_multivariate_context(model_name: str, args: argparse.Namespace | None = None) -> bool:
    if model_name == "panda":
        return True
    return model_name == "chronos" and args is not None and is_chronos2_model(args.chronos_model_id)


def build_timesfm_predictor(args: argparse.Namespace, device: str) -> Callable[[np.ndarray, int], np.ndarray]:
    try:
        import timesfm  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "TimesFM is not installed. Install the official package, for example: "
            "pip install git+https://github.com/google-research/timesfm.git"
        ) from exc

    if hasattr(timesfm, "TimesFM_2p5_200M_torch"):
        if is_legacy_timesfm_checkpoint(args.timesfm_model_id):
            raise ValueError(
                "The installed timesfm package exposes the TimesFM 2.5 PyTorch API, but "
                f"--timesfm-model-id={args.timesfm_model_id!r} is a legacy 2.0 checkpoint. "
                "Use google/timesfm-2.5-200m-pytorch, or install a timesfm version that still "
                "exposes the legacy TimesFm API."
            )
        _ModelCls = timesfm.TimesFM_2p5_200M_torch

        patch_timesfm_2p5_init(_ModelCls)

        model = _ModelCls.from_pretrained(args.timesfm_model_id, local_files_only=args.hf_local_files_only)
        if hasattr(model, "to"):
            model = model.to(device)
        if hasattr(model, "compile") and hasattr(timesfm, "ForecastConfig"):
            model.compile(
                timesfm.ForecastConfig(
                    max_context=args.context_len,
                    max_horizon=args.pred_len,
                    per_core_batch_size=max(1, int(args.batch_size)),
                )
            )

        def predict(batch: np.ndarray, pred_len: int) -> np.ndarray:
            point_forecast, _ = model.forecast(
                horizon=pred_len,
                inputs=[row.astype(np.float32) for row in batch],
            )
            return np.asarray(point_forecast, dtype=np.float32)[:, :pred_len]

        return predict

    if hasattr(timesfm, "TimesFm"):
        hparams = timesfm.TimesFmHparams(
            backend="gpu" if device.startswith("cuda") else "cpu",
            per_core_batch_size=args.batch_size,
            horizon_len=args.pred_len,
            context_len=args.context_len,
        )
        checkpoint = timesfm.TimesFmCheckpoint(huggingface_repo_id=args.timesfm_model_id)
        model = timesfm.TimesFm(hparams=hparams, checkpoint=checkpoint)

        def predict(batch: np.ndarray, pred_len: int) -> np.ndarray:
            point_forecast, _ = model.forecast([row.astype(np.float32) for row in batch], freq=[args.freq] * len(batch))
            return np.asarray(point_forecast, dtype=np.float32)[:, :pred_len]

        return predict

    raise RuntimeError("Unsupported installed timesfm package API.")


def is_legacy_timesfm_checkpoint(model_id: str) -> bool:
    normalized = model_id.lower()
    return "timesfm-2.0" in normalized or "timesfm_2.0" in normalized or "500m" in normalized


def patch_timesfm_2p5_init(model_cls: type[Any]) -> None:
    if getattr(model_cls, "_stellar_init_patch_applied", False):
        return

    # Newer huggingface_hub forwards loader kwargs such as proxies and
    # resume_download into the model constructor.  The official TimesFM class
    # does not accept them, so keep only constructor-supported kwargs.
    orig_init = model_cls.__init__
    init_params = set(inspect.signature(orig_init).parameters)

    def patched_init(self, *init_args, **init_kwargs):
        filtered = {key: value for key, value in init_kwargs.items() if key in init_params}
        orig_init(self, *init_args, **filtered)

    model_cls.__init__ = patched_init
    setattr(model_cls, "_stellar_init_patch_applied", True)


def build_chronos_predictor(args: argparse.Namespace, device: str) -> Callable[[np.ndarray, int], np.ndarray]:
    try:
        try:
            from chronos import BaseChronosPipeline as ChronosPipeline  # type: ignore
        except ImportError:
            from chronos import ChronosPipeline  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Chronos is not installed. Install it with: "
            "pip install git+https://github.com/amazon-science/chronos-forecasting.git"
        ) from exc

    pipeline_kwargs: dict[str, Any] = {"device_map": device}
    dtype = resolve_torch_dtype(args.torch_dtype, device)
    if dtype is not None:
        pipeline_kwargs["dtype"] = dtype
    try:
        pipeline = ChronosPipeline.from_pretrained(
            args.chronos_model_id,
            local_files_only=args.hf_local_files_only,
            **pipeline_kwargs,
        )
    except TypeError as exc:
        # Chronos 1.x forwards loader kwargs to Transformers, which expects
        # ``torch_dtype`` rather than the newer Chronos ``dtype`` spelling.
        if "unexpected keyword argument 'dtype'" not in str(exc):
            raise
        legacy_kwargs = dict(pipeline_kwargs)
        dtype = legacy_kwargs.pop("dtype", None)
        if dtype is not None:
            legacy_kwargs["torch_dtype"] = dtype
        pipeline = ChronosPipeline.from_pretrained(
            args.chronos_model_id,
            local_files_only=args.hf_local_files_only,
            **legacy_kwargs,
        )
    predict_parameters = inspect.signature(pipeline.predict).parameters

    def predict(batch: np.ndarray, pred_len: int) -> np.ndarray:
        context = torch.as_tensor(batch, dtype=torch.float32)
        predict_kwargs: dict[str, Any] = {}
        if "num_samples" in predict_parameters:
            predict_kwargs["num_samples"] = args.chronos_num_samples
        forecast = pipeline.predict(context, pred_len, **predict_kwargs)
        if isinstance(forecast, torch.Tensor):
            arr = forecast.detach().cpu().float().numpy()
        else:
            arr = np.asarray(forecast, dtype=np.float32)
        return reduce_forecast_array(arr, pred_len)

    return predict


def is_chronos2_model(model_id: str) -> bool:
    """Return whether a checkpoint should be loaded through Chronos2Pipeline."""
    return "chronos-2" in str(model_id).lower()


def build_chronos2_predictor(args: argparse.Namespace, device: str) -> Callable[[np.ndarray, int], np.ndarray]:
    """Build a native multivariate Chronos-2 predictor.

    The unified Task E data layout is ``[batch, time, channels]`` while
    Chronos-2 expects a batch of ``[channels, time]`` targets.  Chronos-2
    returns one tensor per target with shape ``[channels, quantiles, horizon]``;
    this adapter selects the median quantile and converts it back to
    ``[batch, horizon, channels]``.
    """
    try:
        from chronos import Chronos2Pipeline  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Chronos-2 requires a recent chronos-forecasting installation with "
            "Chronos2Pipeline. Upgrade with: pip install -U chronos-forecasting"
        ) from exc

    pipeline_kwargs: dict[str, Any] = {"device_map": device}
    dtype = resolve_torch_dtype(args.torch_dtype, device)
    if dtype is not None:
        pipeline_kwargs["dtype"] = dtype
    try:
        pipeline = Chronos2Pipeline.from_pretrained(
            args.chronos_model_id,
            local_files_only=args.hf_local_files_only,
            **pipeline_kwargs,
        )
    except TypeError as exc:
        if "unexpected keyword argument 'dtype'" not in str(exc):
            raise
        legacy_kwargs = dict(pipeline_kwargs)
        dtype = legacy_kwargs.pop("dtype", None)
        if dtype is not None:
            legacy_kwargs["torch_dtype"] = dtype
        pipeline = Chronos2Pipeline.from_pretrained(
            args.chronos_model_id,
            local_files_only=args.hf_local_files_only,
            **legacy_kwargs,
        )

    def predict(batch: np.ndarray, pred_len: int) -> np.ndarray:
        batch = np.asarray(batch, dtype=np.float32)
        if batch.ndim != 3:
            raise ValueError(f"Chronos-2 expects [B,T,C] Task E input, got {batch.shape}.")
        batch_size, context_len, channels = batch.shape
        chronos_inputs = np.transpose(batch, (0, 2, 1))
        forecasts = pipeline.predict(
            chronos_inputs,
            prediction_length=pred_len,
            batch_size=max(1, int(getattr(args, "batch_size", batch_size))),
            context_length=context_len,
        )
        if len(forecasts) != batch_size:
            raise ValueError(
                f"Chronos-2 returned {len(forecasts)} forecasts for a batch of {batch_size} windows."
            )
        predictions = [
            reduce_chronos2_forecast(forecast, pred_len, channels)
            for forecast in forecasts
        ]
        return np.stack(predictions, axis=0).astype(np.float32)

    return predict


def reduce_chronos2_forecast(forecast: Any, pred_len: int, num_channels: int) -> np.ndarray:
    """Convert one Chronos-2 ``[C,Q,H]`` forecast to Task E ``[H,C]``."""
    if isinstance(forecast, torch.Tensor):
        array = forecast.detach().cpu().float().numpy()
    else:
        array = np.asarray(forecast, dtype=np.float32)
    if array.ndim != 3:
        raise ValueError(
            f"Chronos-2 returned a forecast with shape {array.shape}; expected [channels, quantiles, horizon]."
        )
    if array.shape[0] != num_channels or array.shape[2] < pred_len:
        raise ValueError(
            f"Chronos-2 returned shape {array.shape}; expected [{num_channels}, quantiles, >= {pred_len}]."
        )
    median_forecast = np.median(array[:, :, :pred_len], axis=1)
    return median_forecast.transpose(1, 0).astype(np.float32)


def build_panda_predictor(args: argparse.Namespace, device: str) -> Callable[[np.ndarray, int], np.ndarray]:
    model = load_panda_model(args, device)
    model_context_len = int(getattr(model.config, "context_length", args.context_len))

    def predict(batch: np.ndarray, pred_len: int) -> np.ndarray:
        adapted = adapt_multivariate_context_length(batch, model_context_len)
        context_tensor = torch.as_tensor(adapted, dtype=torch.float32, device=device)
        predictions: list[np.ndarray] = []
        remaining = pred_len

        while remaining > 0:
            observed = torch.ones_like(context_tensor, dtype=torch.bool)
            with torch.no_grad():
                outputs = model.generate(
                    past_values=context_tensor,
                    past_observed_mask=observed,
                )
            sample_forecast = outputs.sequences
            step_forecast = sample_forecast.median(dim=1).values
            step_count = min(remaining, int(step_forecast.shape[1]))
            predictions.append(step_forecast[:, :step_count, :].detach().cpu().float().numpy())
            remaining -= step_count
            if remaining <= 0:
                break
            context_tensor = torch.cat([context_tensor, step_forecast], dim=1)
            context_tensor = context_tensor[:, -model_context_len:, :]

        return np.concatenate(predictions, axis=1).astype(np.float32)

    return predict


def load_panda_model(args: argparse.Namespace, device: str) -> torch.nn.Module:
    return load_panda_pretrained(
        args.panda_model_id,
        device,
        local_files_only=args.hf_local_files_only,
    )


def build_patchtst_predictor(args: argparse.Namespace, device: str) -> Callable[[np.ndarray, int], np.ndarray]:
    try:
        from transformers import PatchTSTForPrediction  # type: ignore
    except ImportError as exc:
        raise ImportError("Transformers with PatchTST support is not installed.") from exc

    model_ids = [args.patchtst_model_id] + parse_name_list(args.patchtst_fallback_model_ids)
    errors = []
    model = None
    loaded_model_id = None
    for model_id in model_ids:
        try:
            model = PatchTSTForPrediction.from_pretrained(
                model_id,
                local_files_only=args.hf_local_files_only,
            ).to(device)
            loaded_model_id = model_id
            break
        except Exception as exc:
            errors.append(f"{model_id}: {exc}")
            print(f"PatchTST checkpoint failed, trying next if available: {model_id}: {exc}", file=sys.stderr, flush=True)
    if model is None:
        raise RuntimeError("Could not load any PatchTST checkpoint:\n" + "\n".join(errors))
    if loaded_model_id != args.patchtst_model_id:
        print(f"Loaded fallback PatchTST checkpoint: {loaded_model_id}", flush=True)
    model.eval()
    model_context_len = int(getattr(model.config, "context_length", args.context_len))
    model_pred_len = int(getattr(model.config, "prediction_length", args.pred_len))
    model_channels = int(getattr(model.config, "num_input_channels", 1))

    def predict(batch: np.ndarray, pred_len: int) -> np.ndarray:
        adapted = adapt_context_length(batch, model_context_len)
        if model_channels == 1:
            model_input = adapted[:, :, None]
            original_count = adapted.shape[0]
        else:
            model_input, original_count = pack_univariate_series(adapted, model_channels)
        with torch.no_grad():
            out = model(past_values=torch.as_tensor(model_input, dtype=torch.float32, device=device))
        prediction = out.prediction_outputs.detach().cpu().float().numpy()
        if model_channels == 1:
            prediction = prediction[:, :, 0]
        else:
            prediction = unpack_univariate_series(prediction, original_count)
        if model_pred_len < pred_len:
            pad = np.repeat(prediction[:, -1:], pred_len - model_pred_len, axis=1)
            prediction = np.concatenate([prediction, pad], axis=1)
        return prediction[:, :pred_len]

    return predict


def pack_univariate_series(batch: np.ndarray, model_channels: int) -> tuple[np.ndarray, int]:
    original_count, context_len = batch.shape
    pad_count = (-original_count) % model_channels
    if pad_count:
        pad = np.repeat(batch[-1:, :], pad_count, axis=0)
        batch = np.concatenate([batch, pad], axis=0)
    packed = batch.reshape(-1, model_channels, context_len).transpose(0, 2, 1)
    return packed, original_count


def unpack_univariate_series(prediction: np.ndarray, original_count: int) -> np.ndarray:
    unpacked = prediction.transpose(0, 2, 1).reshape(-1, prediction.shape[1])
    return unpacked[:original_count]


def adapt_context_length(batch: np.ndarray, context_len: int) -> np.ndarray:
    if batch.shape[1] == context_len:
        return batch
    if batch.shape[1] > context_len:
        return batch[:, -context_len:]
    pad_len = context_len - batch.shape[1]
    pad = np.repeat(batch[:, :1], pad_len, axis=1)
    return np.concatenate([pad, batch], axis=1)


def adapt_multivariate_context_length(batch: np.ndarray, context_len: int) -> np.ndarray:
    if batch.shape[1] == context_len:
        return batch
    if batch.shape[1] > context_len:
        return batch[:, -context_len:, :]
    pad_len = context_len - batch.shape[1]
    pad = np.repeat(batch[:, :1, :], pad_len, axis=1)
    return np.concatenate([pad, batch], axis=1)


def run_batched_prediction(
    predictor: Callable[[np.ndarray, int], np.ndarray],
    context: np.ndarray,
    pred_len: int,
    batch_size: int,
    log_every_batches: int,
    label: str,
) -> np.ndarray:
    outputs = []
    total_batches = math.ceil(context.shape[0] / batch_size)
    started = time.perf_counter()
    for batch_idx, start in enumerate(range(0, context.shape[0], batch_size), start=1):
        batch = context[start : start + batch_size]
        outputs.append(predictor(batch, pred_len))
        if log_every_batches > 0 and (batch_idx == 1 or batch_idx == total_batches or batch_idx % log_every_batches == 0):
            elapsed = max(time.perf_counter() - started, 1e-9)
            rate = batch_idx / elapsed
            eta = (total_batches - batch_idx) / rate if rate > 0 else 0.0
            print(f"{label} [{batch_idx}/{total_batches}] {rate:.2f} batch/s eta={format_duration(eta)}", flush=True)
    return np.concatenate(outputs, axis=0).astype(np.float32)


def run_batched_prediction_multivariate(
    predictor: Callable[[np.ndarray, int], np.ndarray],
    context: np.ndarray,
    pred_len: int,
    batch_size: int,
    log_every_batches: int,
    label: str,
) -> np.ndarray:
    outputs = []
    total_batches = math.ceil(context.shape[0] / batch_size)
    started = time.perf_counter()
    for batch_idx, start in enumerate(range(0, context.shape[0], batch_size), start=1):
        batch = context[start : start + batch_size]
        outputs.append(predictor(batch, pred_len))
        if log_every_batches > 0 and (batch_idx == 1 or batch_idx == total_batches or batch_idx % log_every_batches == 0):
            elapsed = max(time.perf_counter() - started, 1e-9)
            rate = batch_idx / elapsed
            eta = (total_batches - batch_idx) / rate if rate > 0 else 0.0
            print(f"{label} [{batch_idx}/{total_batches}] {rate:.2f} batch/s eta={format_duration(eta)}", flush=True)
    return np.concatenate(outputs, axis=0).astype(np.float32)


def format_checkpoint_load_error(model_family: str, model_id: str, local_files_only: bool, exc: Exception) -> str:
    if local_files_only:
        return (
            f"{model_family} checkpoint {model_id!r} was not available in the local Hugging Face cache. "
            "Either warm the cache first, or rerun with --no-hf-local-files-only where network access is available. "
            f"Original error: {exc}"
        )
    return f"Failed to load {model_family} checkpoint {model_id!r}: {exc}"


def reduce_forecast_array(arr: np.ndarray, pred_len: int) -> np.ndarray:
    if arr.ndim == 2:
        return arr[:, :pred_len].astype(np.float32)
    if arr.ndim == 3:
        if arr.shape[-1] == pred_len:
            return np.median(arr, axis=1).astype(np.float32)
        if arr.shape[1] == pred_len:
            return np.median(arr, axis=2).astype(np.float32)
    raise ValueError(f"Could not interpret forecast array with shape {arr.shape}.")


def compute_metrics(prediction: np.ndarray, target: np.ndarray, scale: dict[str, np.ndarray]) -> dict[str, Any]:
    error = prediction - target
    raw_mse = np.square(error).mean()
    raw_mae = np.abs(error).mean()
    z_error = (prediction - scale["mean"]) / scale["std"] - (target - scale["mean"]) / scale["std"]
    horizon_raw_mse = np.square(error).mean(axis=(0, 2))
    horizon_raw_mae = np.abs(error).mean(axis=(0, 2))
    horizon_z_mse = np.square(z_error).mean(axis=(0, 2))
    return {
        "raw_mse": float(raw_mse),
        "raw_mae": float(raw_mae),
        "z_mse": float(np.square(z_error).mean()),
        "z_mae": float(np.abs(z_error).mean()),
        "horizon_raw_mse": horizon_raw_mse.tolist(),
        "horizon_raw_mae": horizon_raw_mae.tolist(),
        "horizon_z_mse": horizon_z_mse.tolist(),
    }


def save_horizon_csv(path: Path, metrics: dict[str, Any]) -> None:
    rows = []
    for idx, (raw_mse, raw_mae, z_mse) in enumerate(
        zip(metrics["horizon_raw_mse"], metrics["horizon_raw_mae"], metrics["horizon_z_mse"]), start=1
    ):
        rows.append({"horizon_step": idx, "raw_mse": raw_mse, "raw_mae": raw_mae, "z_mse": z_mse})
    save_csv(path, rows)


def save_example_csv(
    path: Path,
    prediction: np.ndarray,
    target: np.ndarray,
    channel_names: list[str],
    window_metadata: list[dict[str, Any]],
    example_index: int = 0,
) -> None:
    rows = []
    metadata = window_metadata[example_index]
    for step in range(target.shape[1]):
        for channel_idx, channel_name in enumerate(channel_names):
            rows.append(
                {
                    **metadata,
                    "forecast_step": step + 1,
                    "channel": channel_name,
                    "target": float(target[example_index, step, channel_idx]),
                    "prediction": float(prediction[example_index, step, channel_idx]),
                }
            )
    save_csv(path, rows)


def resolve_device(device_name: str | None) -> str:
    if device_name:
        return device_name
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_torch_dtype(dtype_name: str, device: str) -> torch.dtype | None:
    if dtype_name == "auto":
        return torch.bfloat16 if device.startswith("cuda") else None
    if dtype_name == "float32":
        return torch.float32
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float16":
        return torch.float16
    return None


def parse_name_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def cleanup_previous_outputs(output_dir: Path, model_names: list[str]) -> None:
    for name in model_names:
        for suffix in ("_metrics.json", "_horizon_metrics.csv", "_example_prediction.csv", "_error.json"):
            path = output_dir / f"{name}{suffix}"
            if path.exists():
                path.unlink()
    for name in ("summary_metrics.csv", "failures.json"):
        path = output_dir / name
        if path.exists():
            path.unlink()


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(to_jsonable(payload), handle, indent=2)


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def format_duration(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


if __name__ == "__main__":
    main()
