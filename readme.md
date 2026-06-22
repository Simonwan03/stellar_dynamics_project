# Stellar Dynamics Research Repo

这个仓库围绕 hierarchical three-body stability 数据集做 5 类任务：

- `Task A`: 只用初始条件做 system-level 稳定性预测
- `Task B`: 用早期轨道前缀做 instability 分类或 group stable fraction 预测
- `Task C`: 对完整 32-variation 系统做 multi-horizon survival 预测
- `Task D`: 用半长轴发散时间定义 Lyapunov-like 回归目标
- `Task E`: 直接做轨道/状态序列 forecasting，包括 Transformer 和 foundation baselines

当前主线代码在 [scripts](</home/wan/stellar_dynamics/scripts>) 和 [src](</home/wan/stellar_dynamics/src>)；Panda 相关实验单独放在 [experiments/panda_threebody](</home/wan/stellar_dynamics/experiments/panda_threebody>)。

## Repository Layout

- [dataset_v3](</home/wan/stellar_dynamics/dataset_v3>): 当前主数据集，`Task A-E` 都围绕它展开
- [scripts](</home/wan/stellar_dynamics/scripts>): 任务级入口脚本
- [src](</home/wan/stellar_dynamics/src>): 通用数据处理、模型、训练工具
- [results](</home/wan/stellar_dynamics/results>) / [runs](</home/wan/stellar_dynamics/runs>) / [outputs](</home/wan/stellar_dynamics/outputs>): 各任务产物
- [experiments/panda_threebody](</home/wan/stellar_dynamics/experiments/panda_threebody>): Panda zero-shot / adapter / classifier 子项目

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

入口脚本：[scripts/task_a_stability_classifier.py](/home/wan/stellar_dynamics/scripts/task_a_stability_classifier.py)

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

入口脚本：[scripts/task_b_prefix_classifiers.py](/home/wan/stellar_dynamics/scripts/task_b_prefix_classifiers.py)

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

入口脚本：[scripts/task_c_survival.py](/home/wan/stellar_dynamics/scripts/task_c_survival.py)

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

### Task D: Lyapunov-Like Divergence Time Regression

入口脚本：[scripts/task_d_lyapunov_regression.py](/home/wan/stellar_dynamics/scripts/task_d_lyapunov_regression.py)

底层实现：[src/task_d_lyapunov.py](/home/wan/stellar_dynamics/src/task_d_lyapunov.py)

**任务定义**

`Task D` 不直接预测 stable / unstable，而是在质心系中用 position + velocity 定义无量纲 phase-space separation：

- 每条 trajectory 逐时刻减去自身的质量加权质心位置和质心速度
- position 和 velocity 分别用 pair 在 `t=0` 的 characteristic scale 归一化
- 合并得到 `delta_x(t)`，并计算 `R_x(t) = delta_x(t) / delta_x(0)`
- 当 `R_x(t) > threshold` 时记录首次越阈时间 `tau_system`
- 在默认 `1 < R_x(t) < 100` 区间拟合 `log R_x(t)` 的斜率，得到 `lambda_phase_space`
- 多个 pair 默认用 median 聚合为 system-level `tau_system` 和 `lambda_phase_space`
- 当前主训练目标仍是 `log_tau_system = log1p(tau_system)`；`lambda_phase_space` 同时写入标签用于验证和后续回归

脚本还会检查近似关系 `lambda_phase_space * tau_system / log(threshold)`，理想指数增长情况下该比值接近 `1`。

Task D 分成两个 setting。

**Task D-IC：initial-condition prediction**

- 默认主实验：`--task-setting initial`
- 输入只使用 `t=0` 的 masses、质量比、轨道尺度、偏心率、倾角、能量、角动量和周期等初始条件
- 目标：`log1p(phase_space_divergence_time)`
- 模型：`constant_median`、`random_forest`、`mlp`
- 回答：仅根据初始轨道配置，能否推断未来的 Lyapunov-like divergence timescale

**Task D-Prefix：early-window conditional prediction**

- 显式使用：`--task-setting prefix`
- 从实际 `times_complete` 计算 prefix 结束时间 `t_obs`
- 只保留 `tau_system > t_obs` 的系统，避免输入窗口已经包含 divergence event
- 目标改为 `log1p(tau_system - t_obs)`，即 remaining divergence time
- 模型：
  - `cnn_orbital`：32 条 variations 的逐时刻 orbital mean/std
  - `cnn_phase_space`：在 orbital 基础上加入质心相对 position/velocity
  - `set_cnn`：保留完整 `[B,32,T,C]` orbital ensemble，shared temporal CNN 后做 set mean pooling

神经网络默认使用 `SmoothL1Loss(beta=0.5)`，可通过 `--loss mse` 切回 MSE。评价统一报告 log-time MAE/RMSE、Spearman、Pearson、factor-2/factor-3 accuracy 和 median absolute log error。

输出按 setting 分到 `results/task_d_initial/` 或 `results/task_d_prefix_tobs_<prefix_len>/`，包含 `metrics.json`、`diagnostics.json`、`predictions.csv`、标签、feature importance 和 `plots/`。

这个 target 是由有限扰动在 phase space 中增长得到的 empirical Lyapunov-like divergence timescale，不应解释成严格的 infinitesimal Lyapunov time。

### Task E: Forecasting Baselines

相关入口：

- [scripts/task_e_transformer_orbital.py](/home/wan/stellar_dynamics/scripts/task_e_transformer_orbital.py)
- [scripts/task_e_transformer_trajectory.py](/home/wan/stellar_dynamics/scripts/task_e_transformer_trajectory.py)
- [scripts/task_e_foundation_forecasters.py](/home/wan/stellar_dynamics/scripts/task_e_foundation_forecasters.py)

`Task E` 分成 3 条线。

#### E.1 Orbital Transformer

- 目标：从早期 orbital window 预测未来 orbital window
- 实际训练入口：[train_transformer_orbital.py](/home/wan/stellar_dynamics/train_transformer_orbital.py)
- 默认特征是通用 orbital stack：
  - `aratios`, `mratios`, `eccs`, `incs`, `APs`, `LANs`
  - 其中 `aratios`, `mratios` 做 `log1p`
  - `incs`, `APs`, `LANs` 做 `sin/cos`
  - 总计 `17` channels
- 默认 full run：
  - `context_len=256`
  - `pred_len=64`
  - `stride=32`
  - `d_model=256`
  - `4` 层 Transformer

#### E.2 Trajectory Transformer

- 目标：从早期 `positions_complete + velocities_complete` 预测未来状态窗口
- 实际训练入口：[train_transformer_trajectory.py](/home/wan/stellar_dynamics/train_transformer_trajectory.py)
- 默认表示是 `trajectory/raw_state18`
- 默认 full run：
  - `context_len=64`
  - `pred_len=8`
  - `stride=128`
  - `d_model=128`
  - `2` 层 Transformer

这个分支刻意做得更轻，方便快速扫实验。

#### E.3 Foundation Forecasting

- 目标：在同样的 forecasting 任务上评估 pretrained baselines
- 支持表示：
  - `trajectory`
  - `orbital`
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

## Script Entry Points

- `Task A`: [scripts/task_a_stability_classifier.py](/home/wan/stellar_dynamics/scripts/task_a_stability_classifier.py)
- `Task B`: [scripts/task_b_prefix_classifiers.py](/home/wan/stellar_dynamics/scripts/task_b_prefix_classifiers.py)
- `Task C`: [scripts/task_c_survival.py](/home/wan/stellar_dynamics/scripts/task_c_survival.py)
- `Task D`: [scripts/task_d_lyapunov_regression.py](/home/wan/stellar_dynamics/scripts/task_d_lyapunov_regression.py)
- `Task E`: [scripts/task_e_transformer_orbital.py](/home/wan/stellar_dynamics/scripts/task_e_transformer_orbital.py), [scripts/task_e_transformer_trajectory.py](/home/wan/stellar_dynamics/scripts/task_e_transformer_trajectory.py), [scripts/task_e_foundation_forecasters.py](/home/wan/stellar_dynamics/scripts/task_e_foundation_forecasters.py)

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
