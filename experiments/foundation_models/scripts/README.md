# Compatibility and pilot scripts

The short Python files named `train_panda_*`, `run_chronos2_*`, `train_chronos2_*`, `train_spock_*`, `train_task_b_*`, and `recompute_task_b_base_logits.py` forward to implementations in `../panda/`, `../chronos2/`, or `../shared/`. They preserve old command and import paths for notebooks and saved workflows.

The remaining launchers (`run_all.sh`, `run_zero_shot.py`, finetuning/scratch/evaluation and unbound-dataset scripts) belong to the earlier Panda pilot. They are not the current sampling 3 Task B/C/E manuscript reproduction entry points.
