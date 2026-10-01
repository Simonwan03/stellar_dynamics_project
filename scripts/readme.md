# scripts/ 任务入口

从仓库根目录运行。这里仅放 Task A–F 的命令行入口；数据处理、模型、训练、评估和辅助实验的实现在 [`tasks/`](../tasks/README.md)。论文结果对应的参数和配置见 [复现映射](../docs/reproduction.md)。

| 任务 | 入口 | 默认实现 |
| --- | --- | --- |
| A：初始条件稳定性 | `python scripts/task_a.py --resolution low --help` | [`tasks/task_a/main.py`](../tasks/task_a/main.py) |
| B：短前缀稳定性 | `python scripts/task_b.py --resolution low --help`（high 同理） | low：[main.py](../tasks/task_b/main.py)；high：[highres.py](../tasks/task_b/highres.py) |
| C：survival curve | `python scripts/task_c.py --resolution high --help`（low 同理） | [`tasks/task_c/survival.py`](../tasks/task_c/survival.py) |
| D：轨道去相关时间 | `python scripts/task_d.py --resolution high --help` | [`tasks/task_d/main.py`](../tasks/task_d/main.py) |
| E：轨道预测 | `python scripts/task_e.py --resolution high --help`（low 同理） | [`tasks/task_e/main.py`](../tasks/task_e/main.py) |
| F：掩码前缀稳定性 | `python scripts/task_f.py --resolution high --help` | [`tasks/task_f/main.py`](../tasks/task_f/main.py) |

`--resolution low` 对应 sampling 2 / `dataset_v3_low_resolution`；`--resolution high` 对应 sampling 3 / `dataset_v3_high_resolution`。A 仅支持 low，D/F 仅支持 high；不支持的组合会直接报错。C/E 支持两版数据，C/E 省略 `--resolution` 时沿用原入口的 low 默认值；C 选择 high 时使用论文主实验的 128 步前缀、最小 survival horizon 1 和已提交的 system split。C 的 low 保留原始 CLI 默认值，复现某项历史结果时仍应参照该结果的 `config.json` 显式传入参数。F 默认使用已提交的 sampling 3 system split。各入口允许传入任务自身的其他参数覆盖所选 profile，显式 `--sampling-id` 必须与 resolution 一致。

Task E 的 `scripts/task_e.py` 负责单次训练/评估，必须指定 `--output_dir`；切换 resolution 时会同时切换 shard glob 和 completion table。完整 sampling 3 E1/E2 论文复现使用 `bash scripts/task_e.sh <model> [dataset_dir]`，该脚本会准备缓存并按保存的训练与继续训练顺序运行。只运行 Python 入口并不能替代这个完整流程。

辅助模块按任务存放：B 的标签对齐、embedding 筛选和两层评估在 `tasks/task_b/`；D 的阈值分析在 `tasks/task_d/`；F 的块掩码、混合掩码和 Summary+XGBoost 比较在 `tasks/task_f/`；跨 Task A/B/C 的比较和观测预算实验在 [`tasks/comparisons/`](../tasks/comparisons/README.md)；系统划分生成及复现前检查在 `tasks/common/`。带命令行参数的模块可用 `python -m tasks.<包>.<模块> --help` 查看选项；无参数的评估模块从仓库根目录按模块运行。

`scripts/task_b.py --resolution low` 使用 sampling 2 的历史实现，`--resolution high` 使用 sampling 3 实现；省略时沿用 high 默认值。`scripts/task_c.py` 对应 survival 主实验；旧 fraction 目标保留为 `python -m tasks.task_c`。Task D 的 sampling 2 历史分位数实验与当前中位去相关时间目标不同，未接入当前入口。
