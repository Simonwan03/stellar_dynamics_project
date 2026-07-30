# Stellar Dynamics Research Repo

这个仓库围绕 hierarchical three-body stability 数据集做 6 类任务：

- `Task A`: 只用初始条件做 system-level 稳定性预测
- `Task B`: 用早期轨道前缀做 instability 分类或 group stable fraction 预测
- `Task C`: 对完整 32-variation 系统做 multi-horizon survival 预测
- `Task D`: 用半长轴发散时间定义 Lyapunov-like 回归目标
- `Task E`: 直接做轨道/状态序列 forecasting，包括 Transformer 和 foundation baselines
- `Task F`: 用成对扰动初值预测 stability-boundary sensitivity

当前主线代码在 [scripts](</home/wan/stellar_dynamics/scripts>) 和 [src](</home/wan/stellar_dynamics/src>)。Panda 论文 pretrained 参数通过外部 submodule 加载，主项目只维护三体数据适配、adapter 和评估入口；细节见 [docs/panda_pretrained.md](/home/wan/stellar_dynamics/docs/panda_pretrained.md)。

## Repository Layout

- [dataset_v3](</home/wan/stellar_dynamics/dataset_v3>): 当前主数据集，`Task A-E` 都围绕它展开
- [scripts](</home/wan/stellar_dynamics/scripts>): 任务级入口脚本
- [src](</home/wan/stellar_dynamics/src>): 通用数据处理、模型、训练工具
- [results](</home/wan/stellar_dynamics/results>) / [runs](</home/wan/stellar_dynamics/runs>) / [outputs](</home/wan/stellar_dynamics/outputs>): 各任务产物
- [experiments/panda_threebody](</home/wan/stellar_dynamics/experiments/panda_threebody>): Panda submodule 与历史隔离实验区；新 Task E Panda 对比优先走主项目脚本

## `dataset_v3`

### Core Files

| File | Role |
| --- | --- |
| `dataset_v3/ICs_sampling_2.hdf5` | 16 万条 trajectory 的静态初始条件 |
| `dataset_v3/completed_systems_sampling_2.txt` | 每条 trajectory 的完成状态、`time_unbound`、诊断标记 |
| `dataset_v3/results_batch_00_sampling_2.hdf5` ... `results_batch_19_sampling_2.hdf5` | 20 个时间序列结果分片 |

### Scale

- `160000` trajectories
- `5000` base systems
- 每个 system 有 `32` 个 perturbation variations
- 每条 trajectory 有 `2001` 个时间步
- 每个 result shard 有 `8000` trajectories
- completion table 中有 `159953` 条有效 trajectory
- 有 `4998` 个 system 保留了完整且有效的 `32` 条 variations

### Indexing Convention

- `case_id = run_id // 32`
- `variation_id = run_id % 32`
- `batch_id = run_id // 8000`
- `row_in_batch = run_id % 8000`

`32` 条 variations 可以看成两组 `16`。`Task D` 默认用 `(i, i+16)` 这种配对方式，比较相同 phase grid 下、不同内轨半长轴扰动的响应差异。

### Static Initial Conditions

`ICs_sampling_2.hdf5` 的常用字段：

| Key | Shape | Meaning |
| --- | --- | --- |
| `aratios` | `(160000, 1)` | 半长轴比 |
| `mratios` | `(160000, 2)` | 质量比 |
| `eccs` | `(160000, 2)` | 偏心率 |
| `incs` | `(160000, 2)` | 倾角 |
| `APs` | `(160000, 2)` | 近地点角 |
| `LANs` | `(160000, 2)` | 升交点经度 |
| `positions` | `(160000, 3, 3)` | 3 个天体的初始位置 |
| `velocities` | `(160000, 3, 3)` | 3 个天体的初始速度 |

此外还有 `masses`, `smaxes`, `periods`, `tAnos`, `rel_positions`, `rel_velocities` 等字段，但主线任务最常用的是上表这些 orbital / state features。

### Time-Series Result Shards

每个 `results_batch_XX_sampling_2.hdf5` 的常用字段：

| Key | Shape per shard | Meaning |
| --- | --- | --- |
| `times_complete` | `(8000, 2001)` | 时间网格 |
| `aratios_complete` | `(8000, 2001, 1)` | 半长轴比随时间变化 |
| `mratios_complete` | `(8000, 2001, 2)` | 质量比随时间变化 |
| `eccs_complete` | `(8000, 2001, 2)` | 偏心率轨道序列 |
| `incs_complete` | `(8000, 2001, 2)` | 倾角序列 |
| `APs_complete` | `(8000, 2001, 2)` | 角变量序列 |
| `LANs_complete` | `(8000, 2001, 2)` | 角变量序列 |
| `smaxes_complete` | `(8000, 2001, 2)` | 半长轴序列 |
| `positions_complete` | `(8000, 2001, 3, 3)` | 位置序列 |
| `velocities_complete` | `(8000, 2001, 3, 3)` | 速度序列 |
| `energy_complete` | `(8000, 2001, 1)` | 能量 |
| `angMomentum_complete` | `(8000, 2001, 3)` | 角动量 |

### Data Quality Notes

- 通常先用 `completed_systems_sampling_2.txt` 过滤 `time_unbound < 0` 或 `diagnosis != 0` 的样本。
- 个别 `*_complete` 通道即使在有效样本里也会有 NaN / inf。
- 所以主线脚本一般会做 finite-value screening，或者退回到更稳健的特征子集。

## Common Input Representations

不同任务虽然目标不同，但输入基本落在下面 4 类表示里。

### 1. IC token representation

给 `Task A` 用。

- 原始字段：`aratios`, `mratios`, `eccs`, `incs`, `APs`, `LANs`
- `Task A` 当前默认只取每组 `32` 条 variations 里的第一个初始条件
- `Task A` 默认对 `APs` / `LANs` 做 `sin/cos`，保留原始 `incs`
- 因而 `Task A` 当前默认输入与 `orbital15` 的静态初值版本一致，总计 `15` 维

### 2. Prefix trajectory representation

给 `Task B` 和 `Task C` 用。

- 从 result shards 中截取前 `prefix_len` 个时间步
- 默认基础通道来自 orbital 序列：
  - `aratios_complete`
  - `mratios_complete`
  - `masses_complete`
  - `eccs_complete`
  - `smaxes_complete`
  - `energy_complete`
  - `angMomentum_complete`
  - `APs_complete`
  - `LANs_complete`
  - `incs_complete`
- 仅 `APs_complete` / `LANs_complete` 做 `sin/cos`，`incs_complete` 保留原值
- 可追加：
  - pairwise geometry features
  - 归一化时间通道

### 3. `orbital15`

这是 `Task E` 里 Panda 相关分支最重要的表示。

- `aratios`: `1`
- `mratios`: `2`
- `eccs`: `2`
- `incs`: `2`
- `sin/cos(APs)`: `4`
- `sin/cos(LANs)`: `4`
- 总计 `15` channels

`orbital15` 和通用 orbital stack 的区别在于：它保留 `incs` 原值，不对 `incs` 做 `sin/cos` 展开，所以是 `15` 维而不是 `17` 维。

### 4. `trajectory/raw_state18`

这是直接状态预测的默认表示。

- `positions_complete`: `3 bodies x 3 coords = 9`
- `velocities_complete`: `3 bodies x 3 coords = 9`
- 总计 `18` channels

## Tasks

### Task A: Group Stability from Initial Conditions

入口脚本：[tasks/task_a/main.py](/home/wan/stellar_dynamics/tasks/task_a/main.py)

**任务定义**

- 输入：一个 system 的 `32` 条 variation 的初始条件
- 默认目标：`stable_fraction`
  - 即这 `32` 条里有多少比例在 `time_horizon` 内保持稳定

**输入形式**

- 数据源：`ICs_sampling_2.hdf5` + `completed_systems_sampling_2.txt`
- 先把初始条件重排成 `group_size=32`
- 但建模时默认只取每组 `32` 条 variations 里的第一个初始条件
- 特征定义与 `orbital15` 对齐：
  - `aratios`: `1`
  - `mratios`: `2`
  - `eccs`: `2`
  - `incs`: `2`
  - `sin/cos(APs)`: `4`
  - `sin/cos(LANs)`: `4`
  - 总计 `15` 维
- 这样做的原因是：同组 `32` 条 variations 的初始条件几乎完全相同，保留整组会引入大量重复输入

**模型族**

- `mlp`
- `xgboost`

这项任务强调的是：不看时间演化，只凭初值能预测多少 system-level 稳定性信息。

### Task B: Prefix-Based Instability Prediction

入口脚本：[tasks/task_b/main.py](/home/wan/stellar_dynamics/tasks/task_b/main.py)

**任务定义**

Task B 有两个模式：

- `trajectory_binary`
  - 输入单条 trajectory 的早期 prefix
  - 目标是预测它在 horizon 内是否仍然稳定
- `group_fraction`
  - 输入一个 system 的 `32` 条 trajectory prefixes
  - 目标是预测整个 system 的 stable fraction

**输入形式**

- 数据源：`results_batch_*_sampling_2.hdf5`
- 默认只截取前 `prefix_len=100` 个时间步
- 默认标签 horizon 是 `time_horizon = 10000`
- 默认表示是 `orbital15` prefix：`aratios_complete(1) + mratios_complete(2) + eccs_complete(2) + incs_complete(2) + sin/cos(APs_complete)(4) + sin/cos(LANs_complete)(4) = 15` 维
- 常见输入张量：
  - trajectory 模式：`[N, T, C]`
  - group 模式：
    - `concat`: `[G, T, 32*C]`
    - `set_attention`: `[G, T, 32, C]`
- `trajectory_binary` 默认按 `raw_case_id` 做 group split，避免同一个 base system 的 32 条 perturbations 泄漏到不同数据集
- 如需旧版宽特征栈，可切到 `--representation full`

**模型族**

- `svm`: prefix flatten 后的线性 baseline
- `cnn`: 1D temporal CNN
- `tcn`: dilated temporal convolution
- `attention`: Transformer encoder over time
- `group_attention`: 先在 variation 维做 attention，再在 time 维做 attention

**默认参数规模（`orbital15` 表示）**

- `SVM`
  - 参数量自动由 `T * C` 决定；当前 `C = 15`
- `T = 100`
  - `cnn`: ~`92K`
  - `tcn`: ~`355K`
  - `attention`: ~`399K`
- `T = 200`
  - `cnn`: ~`141K`
  - `tcn`: ~`454K`
  - `attention`: ~`597K`
- `T = 500`
  - `cnn`: ~`200K`
  - `tcn`: ~`706K`
  - `attention`: ~`931K`
- 这些规模由脚本按实际输入时间步数自动选择；若显式设置 `--hidden-dim`，则 `cnn/tcn/attention` 统一改用该 hidden width；`--tcn-levels`、`--attention-layers`、`--attention-heads` 也可单独覆盖
- 默认最大训练轮数是 `50`，并启用 early stopping：`patience = 8`

这项任务是仓库里最直接的 early-warning 分类线。

### Task C: Multi-Horizon Survival for 32-Variation Systems

入口脚本：[tasks/task_c/main.py](/home/wan/stellar_dynamics/tasks/task_c/main.py)

**任务定义**

- 输入：一个完整 system 的 `32` 条 trajectory prefixes
- 输出：多个 future horizons 上的 survival curve
- 标签是 `y_cases[variation, horizon]`
  - 某条 variation 在给定未来时刻是否仍然稳定
- 评估时还会对 `32` 条 variations 求均值，得到 system-level survival fraction 曲线

**输入形式**

- 数据源：`results_batch_*_sampling_2.hdf5`
- 默认先做 prefix truncation，再按 `case_id` 组回 `32` 条 variations
- 输入张量是 `[num_cases, 32, time, channels]`
- 默认 future horizons 是：
  - `500,800,1500,2500,4000,6500,10000`

**模型族**

- `baseline`: 直接用训练集平均 survival curve
- `direct`: aggregation-before-encoding weak baseline；先把 32 条 variation 做均值，再直接回归整条 survival curve
- `direct_mean_std`: 对 32 条 variation 分别计算逐时间点的 `mean / std / min / max`，拼接后直接回归 survival curve
- `traj`: 每条 variation 先做 temporal encoding，再与 group context 融合
- `dual`: 同时建模时间维和 variation 集合结构的 dual-attention 模型

所有 Task C 神经网络统一使用 discrete-time hazard head：

- 输出每个未来区间的条件失稳 hazard
- survival curve 通过 `cumprod(1 - sigmoid(hazard_logits))` 构造，因此严格单调不增
- `traj` / `dual` 使用逐 variation、仅在 at-risk 区间计算的 hazard NLL
- `direct` / `direct_mean_std` 使用由 32 条 variation 的区间存活数和失稳数得到的 binomial hazard NLL

运行后会生成平均 survival curve、代表性 case、per-horizon MAE、`q_pred` vs `q_true` scatter，以及 true/pred/error heatmap 五张结果图。

这是最强调 perturbation-set 结构的任务。

### Task D: Finite-Time Lyapunov-Time Proxy Regression

入口脚本：[tasks/task_d/main.py](/home/wan/stellar_dynamics/tasks/task_d/main.py)

底层实现：[tasks/task_d/pipeline.py](/home/wan/stellar_dynamics/tasks/task_d/pipeline.py)

**任务定义**

`Task D` 在质心系中用 position + velocity 定义无量纲 phase-space separation：

- 每条 trajectory 逐时刻减去自身的质量加权质心位置和质心速度
- position 和 velocity 分别用 pair 在 `t=0` 的 characteristic scale 归一化
- 合并得到 `delta_x(t)`，并计算 `R_x(t) = delta_x(t) / delta_x(0)`
- 对 16 个 `(i, i+16)` perturbation pairs，在 `log R_x(t)` 的早期增长段拟合 slope `lambda_pair`
- 只保留正 slope、`R² >= 0.8`、至少 `5` 个拟合点的 pair
- 默认至少要求 `8/16` 个有效 pair，并要求 pair 间 `log(lambda)` MAD 不超过 `0.75`
- 对合格 pair 的 slope 取 median，得到 `lambda_phase_space`
- 定义 `lyapunov_time_proxy = 1 / lambda_phase_space`
- 主训练目标为 `log_lyapunov_time_proxy = log(1 / lambda_phase_space)`

脚本同时计算 `1 < R < 10`、`1 < R < 30`、`1 < R < 100` 三个区间的
slope sensitivity，并保留 `R > 1000` 的首次 crossing time `tau_cross` 作为辅助诊断，
不再将 `tau_cross` 作为训练 target。

**Task D-IC：initial-condition prediction**

- 默认主实验：`--task-setting initial`
- 输入只使用 `t=0` 的 masses、质量比、轨道尺度、偏心率、倾角、能量、角动量和周期等初始条件
- 目标：`log(finite_time_lyapunov_time_proxy)`
- 模型：`constant_median`、`random_forest`、`mlp`

**Task D-Prefix：early-window prediction**

- 显式使用：`--task-setting prefix`
- 目标与 initial setting 相同，仍为 `log(finite_time_lyapunov_time_proxy)`
- 模型：`cnn_orbital`、`cnn_phase_space`、`set_cnn`
- 当前 proxy 使用整条保存轨迹拟合，包含 prefix 区间；若要把 prefix 版本解释为严格的未来预测，
  应改为只在 `t > t_obs` 的非重叠未来窗口拟合 label

神经网络默认使用 `SmoothL1Loss(beta=0.5)`。评价报告 log-time MAE/RMSE、
原始 Lyapunov-time proxy MAE、Spearman、Pearson、factor-2/factor-3 accuracy。

**数据分辨率限制**

当前 `dataset_v3` 的时间采样间隔约为 `50000`。抽样检查中，不少 pair 从
`R(0)=1` 在第一个非零采样点直接跳到 `R≈400–1500`，使 `1–10`、`1–30`
甚至 `1–100` 区间没有足够的观测点。严格质量筛选会正确拒绝这些标签；
要得到可用于正式训练的 Lyapunov-time proxy，需要重新输出更高时间分辨率的早期轨迹，
或直接积分 variational/tangent equations 并周期 renormalize deviation vector。
因此该 target 应称为 `finite-time Lyapunov-time proxy`，不应写成严格的
asymptotic Lyapunov time。

### Task E: Forecasting Baselines

相关入口：

- [tasks/task_e/main.py](/home/wan/stellar_dynamics/tasks/task_e/main.py)
推荐优先使用统一入口 `tasks/task_e/main.py`。它按 32-variation base system
完成 train/val/test group split，使用 completion-aware train-window scaler，并统一输出 transformed/z
scale 的 overall、horizon-wise、channel-wise metrics。支持
`persistence`, `ridge`, `tcn`, `patchtst`, `transformer`, 简化版
`itransformer`, `timesfm`, `chronos`, `panda_zero_shot` 和
`panda_adapter`。

#### E.0 Baseline Results

正式基线使用全部 `20` 个 sampling-2 shards（`160000` 条 trajectories），
输入表示为 `orbital15`，预测设置为
`context_len=200`, `pred_len=50`, `stride=20`。Scaler 仅使用训练 split
中的 `8192` 条 trajectories；验证集和测试集分别随机评估 `50000` 个
windows。

| Model | Val MSE-Z | Test MSE-Z | Test MAE-Z | Test MSE-Raw |
|---|---:|---:|---:|---:|
| Persistence | 0.808118 | 0.881501 | 0.395696 | 8.62724e8 |
| Ridge (`alpha=1.0`) | 3.162180 | 1.958371 | 0.704020 | 1.18044e8 |

Persistence 直接将 context 最后一个时刻复制到全部未来步，不进行训练。
该结果作为 Task E 的最低比较基线：其他模型应主要在相同 split 和
evaluation windows 下比较 `test_mse_z`。Raw-scale MSE 容易被高方差
channel 主导，不适合脱离 channel-wise metrics 单独比较。

Ridge 使用同样的 `orbital15`, `context_len=200`, `pred_len=50`,
`stride=20` 和 `split_seed=42`，但由于需要在内存中展开窗口，仅使用
`5000` 个训练 windows、`2000` 个验证 windows 和 `2000` 个测试
windows。其 `test_mse_z=1.958371`，没有超过 Persistence。Ridge 的
raw MSE 虽然更低，但不能据此判定整体效果更好。

注意：Persistence 和 Ridge 当前使用的 evaluation window 数量及 scaler
统计范围不同，因此表格只记录已有结果，不属于严格的同条件模型排名。

#### E.0.1 `orbital15` 与 `raw_state18` 表示对比

当前保存的 matched preliminary benchmark 使用相同的 `context_len=200`、
`pred_len=50`、`split_seed=42`、32-variation group split、顺序采样和
completion filtering。`orbital15` 测试集包含 `305` 个窗口，
`raw_state18` 测试集包含 `306` 个窗口，因此两批结果基本匹配，但不是
逐窗口完全相同。由于两种表示的物理量和原始尺度不同，比较应使用共同的
标准化指标 `MSE-Z`、`MAE-Z` 和 `Spearman-Z`，不应直接比较 raw-scale
MSE。

| Model | orbital15 MSE-Z | raw_state18 MSE-Z | orbital15 Spearman-Z | raw_state18 Spearman-Z |
|---|---:|---:|---:|---:|
| TimesFM | **0.2516** | 0.5252 | **0.8567** | 0.4242 |
| Chronos | **0.3044** | 0.5805 | **0.8388** | 0.4283 |
| TCN | **0.4261** | 0.6233 | **0.7289** | 0.2208 |
| iTransformer | **0.3854** | 0.6237 | **0.7530** | 0.1938 |
| PatchTST | **0.3725** | 0.6265 | **0.7671** | 0.1972 |
| Persistence | **0.6938** | 1.0541 | **0.6372** | 0.2043 |
| Ridge | **2.0256** | 2.5951 | **0.4742** | 0.0994 |

在所有具有共同运行结果的模型上，`orbital15` 都取得更低的标准化 MSE
和更高的 Spearman 相关性。TimesFM 的 Spearman-Z 从 `raw_state18`
的 `0.4242` 提升到 `orbital15` 的 `0.8567`；PatchTST 则从
`0.1972` 提升到 `0.7671`。标准化 MAE 的趋势整体一致，但 TCN 在
`raw_state18` 上略低（`0.4312` vs. `0.4628`），iTransformer 两者几乎
相同（均约 `0.4191`）。

`orbital15` 独有的 Panda Adapter 当前进一步达到
`MSE-Z=0.2115`、`MAE-Z=0.2547`、`Spearman-Z=0.8810`。这些结果表明，
在当前三体 forecasting 设置与训练预算下，预测经过物理结构化处理的
`orbital15`，整体上明显比直接预测位置和速度组成的 `raw_state18`
更容易。完整运行口径和原始结果见
[`docs/task_e_experiment_summary.md`](docs/task_e_experiment_summary.md)。

#### E.1 Foundation Forecasting

- 目标：在同样的 forecasting 任务上评估 pretrained baselines
- 支持表示：
  - `trajectory`
  - `orbital15`
- 支持模型：
  - `naive`
  - `timesfm`
  - `chronos`
  - `patchtst`
  - `panda`

**模型差异**

- `timesfm`, `chronos`, `patchtst`
  - 在当前脚本里按 univariate baseline 处理
  - 一个 `C` 通道窗口会拆成 `C` 条独立序列
- `panda`
  - 直接消费 multivariate window
  - 默认 checkpoint `GilpinLab/panda-72M` 期望 `orbital15`
  - Panda 源码来自 `experiments/panda_threebody/vendor/panda` submodule
  - pretrained 权重使用 Hugging Face cache 或本地 `checkpoints/panda/`，不提交到 Git

### Task F: Boundary Sensitivity

入口脚本：[tasks/task_f/main.py](/home/wan/stellar_dynamics/tasks/task_f/main.py)

**任务定义**

- 输入：同一 base system 内成对 perturbation 的初始条件差异
- 输出：多个 future horizons 上，pair 内两条 trajectory 的稳定性是否出现分歧
- 默认 horizon 与 Task C 对齐：
  - `500,800,1500,2500,4000,6500,10000`

**输入形式**

- 数据源：`dataset_v3/results_batch_*_sampling_2.hdf5` + `completed_systems_sampling_2.txt`
- 默认表示：`pair_ic54`
- 可选表示：`pair_ic72`
- 默认每个 system 使用 `group_size=32`

**模型族**

- `constant`
- `logistic`
- `random_forest`
- `xgboost`
- `mlp`

这项任务关注的是：非常接近的初始扰动对是否会跨过稳定性边界，因而与 Task A 的 system-level stable fraction 互补。

## Script Entry Points

- `Task A`: [tasks/task_a/main.py](/home/wan/stellar_dynamics/tasks/task_a/main.py)
- `Task B`: [tasks/task_b/main.py](/home/wan/stellar_dynamics/tasks/task_b/main.py)
- `Task C`: [tasks/task_c/main.py](/home/wan/stellar_dynamics/tasks/task_c/main.py)
- `Task D`: [tasks/task_d/main.py](/home/wan/stellar_dynamics/tasks/task_d/main.py)
- `Task E`: [tasks/task_e/main.py](/home/wan/stellar_dynamics/tasks/task_e/main.py)
- `Task F`: [tasks/task_f/main.py](/home/wan/stellar_dynamics/tasks/task_f/main.py)

## Reproducible Commands

下面命令从仓库根目录运行。默认环境是 `stellar` conda 环境；如在别的机器上运行，需要先安装 [environment.yml](/home/wan/stellar_dynamics/environment.yml)、[requirement.txt](/home/wan/stellar_dynamics/requirement.txt) 和可选的 [requirements-foundation.txt](/home/wan/stellar_dynamics/requirements-foundation.txt)。

### Task A

当前 `results/task_a/config.json` 对应的主实验：

```bash
conda run -n stellar python -m tasks.task_a \
  --dataset-dir dataset_v3 \
  --sampling-id 2 \
  --group-size 32 \
  --time-horizon 10000 \
  --output-dir results/task_a \
  --features aratios,mratios,eccs,incs,APs,LANs \
  --models mlp,xgboost \
  --mlp-size large \
  --xgb-size strong \
  --test-size 0.2 \
  --val-size 0.1 \
  --seed 42 \
  --batch-size 128 \
  --epochs 50 \
  --lr 0.001 \
  --weight-decay 0.0001 \
  --early-stopping-patience 15 \
  --early-stopping-min-delta 1e-4 \
  --mlp-loss mse \
  --device cuda
```

常用参数：

- `--regroup --overwrite-tokenized`: 重新生成 `dataset_v3/ICs_sampling_2_groupof32_taska_orbital15.hdf5`
- `--max-cases N`: smoke test 只跑前 `N` 个 systems
- `--mlp-size small|base|large`: MLP hidden preset
- `--xgb-size small|base|strong`: XGBoost preset
- `--hidden 256,256,128` / `--dropout 0.2`: 手动覆盖 MLP 结构

### Task B

当前 `results/task_b/config.json` 对应的 trajectory binary 主实验：

```bash
conda run -n stellar python -m tasks.task_b \
  --dataset-dir dataset_v3 \
  --sampling-id 2 \
  --output-dir results/task_b \
  --representation orbital15 \
  --target-mode trajectory_binary \
  --group-size 32 \
  --group-aggregation concat \
  --prefix-len 200 \
  --time-stride 1 \
  --time-horizon 10000 \
  --models svm,cnn,tcn,attention \
  --test-size 0.125 \
  --val-size 0.1 \
  --seed 42 \
  --batch-size 256 \
  --epochs 50 \
  --early-stopping-patience 8 \
  --lr 0.001 \
  --weight-decay 0.0001 \
  --attention-heads 4 \
  --dropout 0.2 \
  --num-workers 0 \
  --device cuda
```

Group fraction / set-attention 版本：

```bash
conda run -n stellar python -m tasks.task_b \
  --dataset-dir dataset_v3 \
  --sampling-id 2 \
  --output-dir results/task_b_group_attention \
  --target-mode group_fraction \
  --group-size 32 \
  --group-aggregation set_attention \
  --prefix-len 100 \
  --time-stride 1 \
  --models group_attention \
  --test-size 0.125 \
  --val-size 0.1 \
  --seed 42 \
  --batch-size 256 \
  --epochs 20 \
  --lr 0.001 \
  --weight-decay 0.0001 \
  --hidden-dim 96 \
  --dropout 0.2 \
  --num-workers 0 \
  --device cuda
```

常用参数：

- `--representation orbital15|full`: `orbital15` 是 15 通道；`full` 会使用更宽的 orbital + geometry/time stack
- `--target-mode trajectory_binary|group_fraction`: 单 trajectory 分类或 system stable fraction 回归
- `--group-aggregation concat|set_attention`: 只影响 `group_fraction`
- `--max-shards N --rows-per-shard M`: smoke test / 限制数据量
- `--hidden-dim`, `--tcn-levels`, `--attention-layers`, `--attention-heads`: 覆盖自动模型规模
- `--no-geom --no-time`: 仅对 `--representation full` 生效

### Task C

当前 `results/task_c/config.json` 对应的主实验：

```bash
conda run -n stellar python -m tasks.task_c \
  --dataset-dir dataset_v3 \
  --sampling-id 2 \
  --output-dir results/task_c \
  --group-size 32 \
  --prefix-len 100 \
  --time-stride 1 \
  --future-times 500,800,1500,2500,4000,6500,10000 \
  --models baseline,direct,direct_mean_std,traj,dual \
  --encoder-type cnn \
  --test-ratio 0.2 \
  --val-ratio 0.1 \
  --seed 42 \
  --batch-size 32 \
  --epochs 40 \
  --lr 0.001 \
  --weight-decay 0.0001 \
  --hidden-dim 128 \
  --dropout 0.2 \
  --num-workers 0 \
  --device cuda
```

常用参数：

- `--max-shards`, `--rows-per-shard`, `--max-cases`: 限制数据量
- `--future-times`: 修改 survival curve 的评估 horizons
- `--models baseline,direct,direct_mean_std,traj,dual`: 选择要跑的模型
- `--encoder-type cnn|gru|attention`: 神经网络 encoder
- `--no-geom --no-time`: 移除 geometry / normalized time features

### Task D

当前 `results/task_d_initial/metrics.json` 对应的 initial-condition 主实验：

```bash
conda run -n stellar python -m tasks.task_d \
  --task-setting initial \
  --dataset-dir dataset_v3 \
  --sampling-id 2 \
  --group-size 32 \
  --threshold 1000 \
  --growth-lower 1 \
  --growth-upper 100 \
  --growth-sensitivity-uppers 10 30 100 \
  --min-growth-points 5 \
  --lambda-r2-min 0.8 \
  --min-valid-lambda-pairs 8 \
  --max-pair-log-mad 0.75 \
  --aggregation median \
  --censoring-rule median_pair \
  --feature-aggregation first \
  --test-size 0.2 \
  --seed 42 \
  --output-dir results/task_d_initial \
  --completion-table-name completed_systems_sampling_2.txt \
  --prefix-len 200 \
  --prefix-time-stride 1 \
  --sequence-aggregation mean_std \
  --mlp-epochs 80 \
  --cnn-epochs 80 \
  --batch-size 128 \
  --lr 0.001 \
  --weight-decay 0.0001 \
  --dropout 0.1 \
  --val-fraction 0.1 \
  --patience 12 \
  --loss huber \
  --huber-beta 0.5
```

Prefix setting：

```bash
conda run -n stellar python -m tasks.task_d \
  --task-setting prefix \
  --dataset-dir dataset_v3 \
  --sampling-id 2 \
  --group-size 32 \
  --threshold 1000 \
  --growth-lower 1 \
  --growth-upper 100 \
  --growth-sensitivity-uppers 10 30 100 \
  --lambda-r2-min 0.8 \
  --min-valid-lambda-pairs 8 \
  --max-pair-log-mad 0.75 \
  --prefix-len 200 \
  --prefix-time-stride 1 \
  --sequence-aggregation mean_std \
  --output-dir results/task_d_prefix_tobs_200 \
  --mlp-epochs 80 \
  --cnn-epochs 80 \
  --batch-size 128 \
  --loss huber \
  --huber-beta 0.5
```

常用参数：

- `--force-recompute`: 忽略缓存标签，重新计算 slope、Lyapunov proxy 和 crossing-time 诊断
- `--lambda-r2-min`, `--min-valid-lambda-pairs`, `--max-pair-log-mad`: 调整 slope 与 pair consistency 质量筛选
- `--feature-aggregation first|mean_std`: initial setting 的 feature 聚合方式
- `--no-mlp`, `--no-cnn-orbital`, `--no-cnn-phase-space`, `--no-set-cnn`: 关闭对应神经模型
- `--loss mse|huber`: 神经网络回归损失
- `--allow-incomplete-valid-group`: 放宽完整 `32` 条有效 variation 的要求

### Task E

统一 forecasting 入口。当前 `results/task_e_local_timesfm/config.json` 对应的 local TimesFM zero-shot 实验：

```bash
conda run -n stellar python -m tasks.task_e \
  --data_glob "dataset_v3/results_batch_*_sampling_2.hdf5" \
  --representation orbital15 \
  --model timesfm \
  --context_len 200 \
  --pred_len 50 \
  --stride 20 \
  --split_seed 42 \
  --seed 42 \
  --train_frac 0.7 \
  --val_frac 0.15 \
  --batch_size 128 \
  --num_workers 0 \
  --epochs 50 \
  --lr 0.001 \
  --weight_decay 0.0001 \
  --patience 10 \
  --grad_clip 1.0 \
  --hidden_dim 128 \
  --num_layers 4 \
  --kernel_size 3 \
  --dropout 0.1 \
  --patch_len 16 \
  --patch_stride 8 \
  --d_model 128 \
  --n_heads 4 \
  --ridge_alpha 1.0 \
  --num_eval_windows 128 \
  --foundation_batch_size 32 \
  --timesfm_model_id google/timesfm-2.5-200m-pytorch \
  --timesfm_backend auto \
  --chronos_model_id amazon/chronos-bolt-small \
  --chronos_variant bolt \
  --chronos_num_samples 20 \
  --panda_model_id GilpinLab/panda-72M \
  --freeze_panda \
  --adapter_lr 0.0001 \
  --adapter_chunk_size 4096 \
  --chronos_adapter_context_len 512 \
  --timesfm_adapter_context_len 512 \
  --timesfm_adapter_pred_len 64 \
  --num_train_windows_for_adapter 4096 \
  --hf-local-files-only \
  --torch_dtype auto \
  --output_dir results/task_e_local_timesfm \
  --prediction_samples 16 \
  --log_every_batches 100
```

更轻量的 supervised smoke test：

```bash
conda run -n stellar python -m tasks.task_e \
  --data_glob "dataset_v3/results_batch_00_sampling_2.hdf5" \
  --representation orbital15 \
  --model tcn \
  --context_len 32 \
  --pred_len 8 \
  --stride 20 \
  --max_trajectories 100 \
  --max_scaler_trajectories 70 \
  --max_windows_per_trajectory 5 \
  --epochs 2 \
  --output_dir runs/task_e_tcn_smoke
```

常用参数：

- `--model persistence|ridge|tcn|patchtst|transformer|itransformer|timesfm|chronos|panda_zero_shot|panda_adapter`
- `--representation orbital15|raw_state18|trajectory`
- `--max_trajectories`, `--max_windows_per_trajectory`, `--max_train_windows`, `--max_val_windows`, `--max_test_windows`: 控制窗口数量
- `--drop_nan_windows` / `--no-drop_nan_windows`: 默认拒绝 completion-filtered 后仍含非有限值的窗口
- `--completion_table`: completion metadata 路径，用于过滤无效 trajectory 和事件后窗口
- `--require_complete_systems` / `--no-require_complete_systems`: 是否只保留完整有效的 32-variation system
- `--amp`: 启用 mixed precision
- `--hf-local-files-only` / `--no-hf-local-files-only`: foundation checkpoint 是否只从本地 cache 读取
- Panda 依赖和 pretrained 权重约定见 [docs/panda_pretrained.md](/home/wan/stellar_dynamics/docs/panda_pretrained.md)

### Task F

当前 `results/task_f_boundary_sensitivity/config.json` 对应的主实验：

```bash
conda run -n stellar python -m tasks.task_f \
  --data-glob "dataset_v3/results_batch_*_sampling_2.hdf5" \
  --completed-systems dataset_v3/completed_systems_sampling_2.txt \
  --output-dir results/task_f_boundary_sensitivity \
  --representation pair_ic54 \
  --models constant logistic random_forest xgboost mlp \
  --horizons 500 800 1500 2500 4000 6500 10000 \
  --group-size 32 \
  --seed 42 \
  --rf-estimators 500 \
  --rf-min-samples-leaf 2 \
  --xgb-estimators 500 \
  --xgb-learning-rate 0.05 \
  --xgb-max-depth 4 \
  --batch-size 512 \
  --epochs 30 \
  --lr 0.001 \
  --weight-decay 0.0001 \
  --dropout 0.2 \
  --patience 8
```

常用参数：

- `--representation pair_ic54|pair_ic72`: 选择 pair 初值特征维度
- `--models constant logistic random_forest xgboost mlp`: 空格分隔的模型列表
- `--horizons`: 空格分隔的评估时间点
- `--max-systems N`: 限制 systems 数量做 smoke test
- `--device cuda`: 显式指定 MLP 训练设备

## Environment

大多数脚本默认在 `stellar` conda 环境下运行。基础依赖可以从 [environment.yml](/home/wan/stellar_dynamics/environment.yml) 和 [requirement.txt](/home/wan/stellar_dynamics/requirement.txt) 查看；foundation-model 额外依赖见 [requirements-foundation.txt](/home/wan/stellar_dynamics/requirements-foundation.txt)。

## Model Settings

### Task A
- **XGBoost**
- **MLP-flatten-base**: `544 → 256 → 128 → 64 → K` — params ~180K

### Task B
- default representation: `orbital15` prefix (`15` channels per step)
- **SVM**: params determined by `T * 15`
- **CNN**
  - `T=100`: ~`92K`
  - `T=200`: ~`141K`
  - `T=500`: ~`200K`
- **TCN**
  - `T=100`: ~`355K`
  - `T=200`: ~`454K`
  - `T=500`: ~`706K`
- **Transformer**
  - `T=100`: ~`399K`
  - `T=200`: ~`597K`
  - `T=500`: ~`931K`

### Task C
- **Shared CNN-DeepSets-Base** — params ~300K
- **Dual Attention**: `d=128, time_layers=2, set_layers=2` — params ~1M–1.5M

### Task D
- **XGBoost**
- **MLP-Base**: `summary → 256 → 128 → 64 → 1` — params ~100K
- **CNN-Base prefix** — params ~100K–150K

### Task E
- **TCN-Forecast-Base** — params ~0.5M–1M
- **Orbital Transformer-Base**: `context=256, pred=64, d_model=256, layers=4` — params ~3M–4M
- **Raw Transformer-Base**: `context=128, pred=16, d_model=192, layers=3` — params ~1M–2M
- **Panda / Chronos / TimesFM / PatchTST** as foundation baselines
