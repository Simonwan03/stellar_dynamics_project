from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import EXPERIMENT_ROOT, VENDOR_PANDA, hydra_list, python_cmd, stream_subprocess, write_json
from adapters.project_data import get_export_root, load_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_yaml(EXPERIMENT_ROOT / args.config)
    export_root = get_export_root(cfg, representation=cfg.get("representation", "orbital15"))
    train_root = export_root / cfg.get("train_split", "train")
    if not train_root.exists():
        raise FileNotFoundError(f"Training split not found: {train_root}")

    checkpoint_root = EXPERIMENT_ROOT / "outputs" / "checkpoints"
    log_path = EXPERIMENT_ROOT / "outputs" / "logs" / f"{cfg['run_name']}.log"
    command = [
        *python_cmd(),
        str(VENDOR_PANDA / "scripts" / "panda" / "train.py"),
        f"hydra.run.dir={EXPERIMENT_ROOT / 'outputs' / 'logs' / 'hydra'}",
        f"run_name={cfg['run_name']}",
        f"train_data_dirs={hydra_list([train_root])}",
        f"train.output_dir={checkpoint_root}",
        f"train.max_steps={cfg['max_steps']}",
        f"train.save_steps={cfg['save_steps']}",
        f"train.log_steps={cfg['log_steps']}",
        f"train.per_device_train_batch_size={cfg['batch_size']}",
        f"train.learning_rate={cfg['learning_rate']}",
        f"train.dataloader_num_workers={cfg['dataloader_num_workers']}",
        f"train.dataloader_prefetch_factor={'2' if int(cfg['dataloader_num_workers']) > 0 else 'null'}",
        "train.dataloader_persistent_workers=false",
        f"train.tf32={str(cfg.get('tf32', False)).lower()}",
        f"train.torch_compile={str(cfg.get('torch_compile', False)).lower()}",
        "train.ddp_backend=null",
        "wandb.log=false",
        "scheduler.enabled=false",
        "shuffle_buffer_length=10000",
        f"fixed_dim={cfg['fixed_dim']}",
        f"min_past={cfg['context_length']}",
        "augmentations.augmentation_rate=0.0",
        "patchtst.mode=predict",
        f"patchtst.context_length={cfg['context_length']}",
        f"patchtst.prediction_length={cfg['prediction_length']}",
        f"patchtst.patch_length={cfg['patch_length']}",
        f"patchtst.patch_stride={cfg['patch_stride']}",
        f"patchtst.d_model={cfg['d_model']}",
        f"patchtst.ffn_dim={cfg['ffn_dim']}",
        f"patchtst.num_hidden_layers={cfg['num_hidden_layers']}",
        f"patchtst.num_attention_heads={cfg['num_attention_heads']}",
        "patchtst.channel_attention=true",
    ]
    for key in [
        "pooling_type",
        "use_dynamics_embedding",
        "num_poly_feats",
        "num_rff",
        "poly_degrees",
    ]:
        if key in cfg:
            value = cfg[key]
            if isinstance(value, bool):
                value = str(value).lower()
            command.append(f"patchtst.{key}={value}")
    if cfg.get("optim"):
        command.append(f"train.optim={cfg['optim']}")
    if cfg.get("pretrained_pft_path"):
        command.append(f"patchtst.pretrained_pft_path={cfg['pretrained_pft_path']}")
    if cfg.get("pretrained_encoder_path"):
        command.append(f"patchtst.pretrained_encoder_path={cfg['pretrained_encoder_path']}")

    print("Running:", " ".join(str(x) for x in command))
    stream_subprocess(command, log_path=log_path, cwd=EXPERIMENT_ROOT)

    run_dirs = sorted(checkpoint_root.glob(f"{cfg['run_name']}-*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not run_dirs:
        raise FileNotFoundError(f"No run directory produced under {checkpoint_root}")
    checkpoint = run_dirs[0] / "checkpoint-final"
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint-final missing under {run_dirs[0]}")

    metadata = {
        "config": str(EXPERIMENT_ROOT / args.config),
        "run_dir": str(run_dirs[0]),
        "checkpoint_final": str(checkpoint),
        "train_root": str(train_root),
        "log_path": str(log_path),
    }
    write_json(checkpoint_root / f"{cfg['run_name']}_latest.json", metadata)
    print("Training finished")
    print(metadata)


if __name__ == "__main__":
    main()
