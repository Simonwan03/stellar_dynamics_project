# Chronos-2 Task B

`run_chronos2_task_b_cache.py` extracts frozen REG-token features in Task B label order. `train_chronos2_task_b_linear.py` trains the linear probe on that cache. `run_chronos2_prefix96_example.py` is a small historical example. Supply the dataset, labels, model checkpoint, and output paths explicitly; script defaults refer to earlier sampling 2 runs. The paper protocol and saved configurations are in [docs/reproduction.md](../../../docs/reproduction.md).
