# Stellar Dynamics Research Repo

这个仓库围绕 hierarchical three-body stability 数据集做 6 类任务：

- `Task A`: 只用初始条件做 system-level 稳定性预测
- `Task B`: 用早期轨道前缀做 instability 分类或 group stable fraction 预测
- `Task C`: 对完整 32-variation 系统做 multi-horizon survival 预测
- `Task D`: 用半长轴发散时间定义 Lyapunov-like 回归目标
- `Task E`: 直接做轨道/状态序列 forecasting，包括 Transformer 和 foundation baselines
- `Task F`: 用成对扰动初值预测 stability-boundary sensitivity

完整结果索引见[实验结果总汇](#实验结果总汇磁盘产物截至-2026-08-16)；其中会区分正式
主实验、ablation、历史重跑和没有最终指标的目录。

当前主线代码在 [scripts](</home/wan/stellar_dynamics/scripts>) 和 [tasks](</home/wan/stellar_dynamics/tasks>)。Panda 论文 pretrained 参数通过外部 submodule 加载，主项目只维护三体数据适配、adapter 和评估入口；细节见 [docs/panda_pretrained.md](/home/wan/stellar_dynamics/docs/panda_pretrained.md)。

## Repository Layout

- [dataset_v3](</home/wan/stellar_dynamics/dataset_v3>): 当前主数据集，`Task A-E` 都围绕它展开
- [scripts](</home/wan/stellar_dynamics/scripts>): 任务级入口脚本
- [tasks](</home/wan/stellar_dynamics/tasks>): 各任务入口、数据处理、模型和训练工具
- [results](</home/wan/stellar_dynamics/results>) / [runs](</home/wan/stellar_dynamics/runs>) / [Panda outputs](</home/wan/stellar_dynamics/experiments/panda_threebody/outputs>): 各任务产物
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

### Unstable Variation Distribution

以 `time_unbound < 10000` 判定单条 variation 为 unstable；仅统计
`time_unbound >= 0` 且 `diagnosis == 0` 的有效记录。5000 个原始 system 中有
4998 个 system 具备完整且有效的 32 条 variations；system `182` 和 `365` 因无效
诊断记录被排除。

在这 4998 个有效 system 中，共有 `109536 / 159936 = 68.49%` 条 variation
unstable。每个 system 内 unstable variation 数量的分布如下：

| Unstable 数量 | Systems | 占比 | Unstable 数量 | Systems | 占比 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 790 | 15.81% | 17 | 33 | 0.66% |
| 1 | 101 | 2.02% | 18 | 48 | 0.96% |
| 2 | 64 | 1.28% | 19 | 34 | 0.68% |
| 3 | 48 | 0.96% | 20 | 36 | 0.72% |
| 4 | 34 | 0.68% | 21 | 51 | 1.02% |
| 5 | 38 | 0.76% | 22 | 45 | 0.90% |
| 6 | 38 | 0.76% | 23 | 58 | 1.16% |
| 7 | 40 | 0.80% | 24 | 66 | 1.32% |
| 8 | 40 | 0.80% | 25 | 86 | 1.72% |
| 9 | 33 | 0.66% | 26 | 86 | 1.72% |
| 10 | 32 | 0.64% | 27 | 103 | 2.06% |
| 11 | 36 | 0.72% | 28 | 138 | 2.76% |
| 12 | 36 | 0.72% | 29 | 178 | 3.56% |
| 13 | 40 | 0.80% | 30 | 294 | 5.88% |
| 14 | 37 | 0.74% | 31 | 557 | 11.14% |
| 15 | 44 | 0.88% | 32 | 1,692 | 33.85% |
| 16 | 42 | 0.84% |  |  |  |

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
- 可选目标：`binary_any`
  - 只有 `32` 条全部在 `time_horizon` 内保持稳定时标签才是 `1`
  - 任意一条 variation unbound 时标签就是 `0`

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

`stable_fraction` 是连续值回归；`binary_any` 是 system-level 的 0/1 分类。
在 `binary_any` 模式下，MLP 默认使用 `BCEWithLogitsLoss`，XGBoost 使用
`binary:logistic`，并报告 accuracy、precision、recall、F1、AUROC 和 AUPRC。
此外，`metrics.json` 还会保存 balanced accuracy、specificity、FPR/FNR、MCC、
混淆矩阵（TN/FP/FN/TP）、Brier score、log loss，以及 10-bin 的 ECE/MCE，
便于同时检查类别不平衡、阈值表现和概率校准。`stable_fraction` 回归会额外报告
median absolute error、explained variance、Pearson/Spearman、平均偏差、误差容忍率、
按目标稳定比例分箱的误差，以及相对于 32 次扰动二项采样噪声的 MSE 比值。

这项任务强调的是：不看时间演化，只凭初值能预测多少 system-level 稳定性信息。

**实验结果**

观测稳定比例来自每个 system 的 `32` 次扰动试验，因此即使模型准确预测了
潜在稳定概率 \(p\)，标签本身仍有二项采样噪声：

\[
\operatorname{MSE}_{\mathrm{binomial}}=\frac{p(1-p)}{32}.
\]

在噪声最大的 \(p=0.5\) 附近，
\(\operatorname{MSE}=0.0078\)、\(\operatorname{RMSE}\approx0.088\)，
正态近似下 \(\operatorname{MAE}\approx0.07\)。这是中等稳定概率 system
附近的近似不可约误差尺度。

| Model | MSE | RMSE | MAE | \(R^2\) |
| --- | ---: | ---: | ---: | ---: |
| MLP-base | 0.0916 | 0.3026 | 0.2149 | 0.3876 |
| XGBoost-base | 0.0862 | 0.2937 | 0.2239 | 0.4233 |
| MLP-large | 0.0938 | 0.3062 | **0.2033** | 0.3728 |
| XGBoost-strong | **0.0834** | **0.2887** | 0.2176 | **0.4425** |

`XGBoost-strong` 取得最低 MSE/RMSE 和最高 \(R^2\)，整体表现最佳；
`MLP-large` 的 MAE 最低。所有模型仍明显高于上述二项采样噪声尺度，
说明仅凭初始条件预测长期稳定比例仍有较大难度。

要运行“只要一个 variation unbound 就判为不稳定”的二分类版本：

```bash
conda run -n stellar python -m tasks.task_a \
  --dataset-dir dataset_v3 \
  --sampling-id 2 \
  --group-size 32 \
  --time-horizon 10000 \
  --target-mode binary_any \
  --models mlp,xgboost \
  --mlp-size base \
  --xgb-size base \
  --test-size 0.2 \
  --val-size 0.1 \
  --seed 42 \
  --epochs 50 \
  --device cuda \
  --output-dir results/task_a_binary_any
```

`--time-horizon` 可以直接改成其他判定时刻，例如 `5000`；当前
`completed_systems_sampling_2.txt` 的 `time_unbound` 最大值是 `10000`，因此不要设置超过
`10000` 的 horizon。
二分类输出会保存 `metrics.json`、MLP/XGBoost 模型文件、测试集预测以及
MLP 的 ROC/PR 曲线。

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
- `gru`: 两层单向 GRU temporal encoder
- `attention`: Transformer encoder over time
- `group_attention`: 先在 variation 维做 attention，再在 time 维做 attention

**默认参数规模（`orbital15` 表示）**

- `SVM`
  - 参数量自动由 `T * C` 决定；当前 `C = 15`
- `T = 100`
  - `cnn`: ~`92K`
  - `tcn`: ~`355K`
  - `gru`: ~`155K`
  - `attention`: ~`399K`
- `T = 200`
  - `cnn`: ~`141K`
  - `tcn`: ~`454K`
  - `gru`: ~`155K`
  - `attention`: ~`597K`
- `T = 500`
  - `cnn`: ~`200K`
  - `tcn`: ~`706K`
  - `gru`: ~`240K`
  - `attention`: ~`931K`
- 这些规模由脚本按实际输入时间步数自动选择；若显式设置 `--hidden-dim`，则 `cnn/tcn/gru/attention` 统一改用该 hidden width；`--tcn-levels`、`--gru-layers`、`--attention-layers`、`--attention-heads` 也可单独覆盖
- 默认最大训练轮数是 `50`，并启用 early stopping：`patience = 8`

这项任务是仓库里最直接的 early-warning 分类线。

**实验结果**

以下 `trajectory_binary` 测试集中的不稳定类别占比为 `31.6%`：

| Prefix | Model | Accuracy | Precision | Recall | F1 | AUROC | AUPRC |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | SVM | 0.7687 | 0.6659 | 0.5385 | 0.5955 | 0.8361 | 0.6387 |
| 100 | CNN | **0.8135** | 0.6912 | **0.7414** | **0.7154** | 0.8891 | 0.7509 |
| 100 | TCN | 0.8042 | 0.7267 | 0.6101 | 0.6633 | 0.8922 | **0.7642** |
| 100 | Attention | 0.8053 | **0.7270** | 0.6150 | 0.6663 | **0.8930** | 0.7467 |
| 200 | SVM | 0.7723 | 0.6611 | 0.5743 | 0.6147 | 0.8418 | 0.6476 |
| 200 | CNN | 0.8391 | 0.7275 | 0.7851 | 0.7552 | 0.9132 | 0.7943 |
| 200 | TCN | **0.8622** | **0.7575** | **0.8297** | **0.7920** | **0.9321** | **0.8293** |
| 200 | Attention | 0.8456 | 0.7277 | 0.8179 | 0.7701 | 0.9167 | 0.7709 |

在 `100` 步前缀上，所有时序神经网络都优于展平 SVM；CNN 的 F1 最高，
而 TCN/Attention 在排序指标上更强。前缀增加到 `200` 步后，TCN 的
F1、AUROC 和 AUPRC 分别达到 `0.7920`、`0.9321` 和 `0.8293`，
召回率也由 `0.6101` 提升到 `0.8297`。SVM 的提升较小，说明更长观测
需要合适的时序归纳偏置才能转化为有效的早期预警信号。

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
- `stats`: 对每个 `orbital15` 通道提取 Task-B 对齐的八统计量，再预测逐 variation hazard
- `embedding_direct`: 直接使用冻结 Panda/Chronos 的 `15×768=11520` 维 embedding
- `embedding_meanstd`: 对 15 个通道做 mean/std pooling 后使用 1536 维表示
- `embedding_attention`: 对 15 个通道 embedding 学习 softmax attention 权重后池化

所有 Task C 神经网络统一使用 discrete-time hazard head：

- 输出每个未来区间的条件失稳 hazard
- survival curve 通过 `cumprod(1 - sigmoid(hazard_logits))` 构造，因此严格单调不增
- `traj` / `dual` 同时输出逐 variation survival curve 和 system-level survival curve
- `traj` / `dual` 的 variation head 使用逐 variation、仅在 at-risk 区间计算的 hazard NLL；system head 使用 binomial hazard NLL
- 训练时额外使用 system head 与 variation survival 均值之间的一致性损失
- `direct` / `direct_mean_std` 使用由 32 条 variation 的区间存活数和失稳数得到的 binomial hazard NLL

#### Event-time/censoring 版本（20 个 log-spaced bins）

为避免把 7 个 survival fraction 反推成伪 hazard，新增了独立入口
[`tasks/task_c/survival.py`](/home/wan/stellar_dynamics/tasks/task_c/survival.py)：

```bash
conda run -n stellar python -m tasks.task_c.survival \
  --prefix-cache results/task_c_ablation/_cache \
  --prefix-len 100 \
  --prefix-representation orbital15 \
  --models baseline,stats,cnn \
  --hazard-bins 20 \
  --hazard-t-max 10000 \
  --variation-target full32 \
  --epochs 40 \
  --device cuda
```

它直接使用每条 variation 的 `time_unbound` 构造事件时间标签：事件 bin 之前
target 为 0、事件 bin 为 1，之后由 at-risk mask 屏蔽；`time_unbound == 10000`
按观测上限处理为 right-censored（所有 bin target=0、mask=1）。模型为每条
variation 输出 20 个 conditional hazards，再用 `cumprod(1-hazard)` 得到严格
单调的 survival curve；训练目标为 variation masked hazard NLL 加上权重
`0.2` 的 system curve MAE（32 条 variation survival 的均值对真实 system curve）。
默认 `t_min` 从有效数据最早正事件时间推断（sampling_2 为 5），也可通过
`--hazard-t-min` 显式指定。输出同时保存 20-bin 的 MAE/RMSE、variation NLL、
Brier/IBS、删失比例和 monotonicity 检查，并把曲线插值到原有 7 个 horizons，
对应指标位于 `old_*` 字段。原 `python -m tasks.task_c` 入口仍保留，便于和旧
的 7-horizon fraction 训练结果直接对比。

在相同的 4998-system case split、orbital15 prefix=100 上，Stats 和 7/20 bins
均训练 20 epochs；Stats 编码器现在与 Task B 类 SPoCK baseline 对齐，按每个
orbital15 通道提取 `initial, final, mean, std, slope, min, max, delta` 八个
统计量（共 120 维/variation），TCN 为 5-epoch pilot：

| event model | bins | 20-bin MAE | final-bin MAE | old 7-horizon MAE | IBS |
| --- | ---: | ---: | ---: | ---: | ---: |
| mean baseline | 20 | 0.3093 | 0.3478 | 0.3720 | 0.1309 |
| Stats | 7 | 0.0723 | 0.1493 | 0.1057 | 0.0669 |
| Stats | 20 | **0.0671** | **0.1499** | **0.1029** | **0.0636** |
| TCN (pilot) | 20 | 0.1160 | 0.2072 | 0.1625 | 0.0915 |

八统计量重跑结果分别保存在
`results/task_c_survival7_stats_full_taskb8/` 和
`results/task_c_survival20_stats_full_taskb8/`；旧的 `stats_full` 目录保留作
7-stat 历史结果参考。

这组数字是正确 event-time/censoring 目标的第一版参考，不应直接与旧脚本中
由 7 个 fraction 反推 pseudo-hazard 的 dual 数字混为同一实验；后续 Panda、
Chronos 和 LoRA 也会复用这套 target、mask、head 和评估协议。

已有的 prefix=96 frozen embedding 也可以直接接入同一套 hazard head：

```bash
conda run -n stellar python -m tasks.task_c.survival \
  --embedding-cache experiments/panda_threebody/outputs/panda_unbound_dataset/panda_task_b_prefix_96_features_aligned.float16.npy \
  --embedding-labels results/task_b_prefix_ablation/prefix_100/labels_filtered.csv \
  --prefix-len 96 \
  --models baseline,embedding_direct,embedding_meanstd,embedding_attention \
  --hazard-bins 20 --epochs 10 --device cuda \
  --output-dir results/task_c_survival20_panda
```

`embedding_direct` 不增加额外 projection，直接让 HazardHead 的第一层处理
`15×768=11520` 维；`embedding_meanstd` 先变为 1536 维。相同 split、20 bins、
10 epochs 的结果为（Stats 使用上述八统计量）：

| encoder | pooling | 20-bin MAE | final-bin MAE | old 7-horizon MAE | IBS |
| --- | --- | ---: | ---: | ---: | ---: |
| Stats (prefix=96, 10 epochs) | Task-B eight statistics | 0.0747 | 0.1605 | 0.1138 | 0.0670 |
| Panda | direct | **0.0449** | 0.0923 | 0.0658 | 0.0473 |
| Panda | mean+std | 0.0784 | 0.1272 | 0.0980 | 0.0651 |
| Panda | attention pooling | 0.0527 | 0.1057 | 0.0760 | 0.0523 |
| Chronos-2 | direct | **0.0388** | 0.0887 | **0.0616** | **0.0426** |
| Chronos-2 | mean+std | 0.0582 | 0.1079 | 0.0787 | 0.0542 |
| Chronos-2 | attention pooling | 0.0512 | 0.1068 | 0.0749 | 0.0501 |

prefix=96 的八统计量重跑结果位于
`results/task_c_survival20_stats_prefix96_taskb8/`。

结果目录分别为
`results/task_c_survival20_panda_direct/`、
`results/task_c_survival20_panda_meanstd/` 和
`results/task_c_survival20_panda_attention/`、
`results/task_c_survival20_chronos_direct/`、
`results/task_c_survival20_chronos_attention/`。attention pooling 在现有
`[15,768]` 通道 embedding 上学习 softmax 权重并加权求和，不重新运行
Panda/Chronos-2 backbone；这些 embedding cache 已先通过
label fingerprint 检查，再按完整 system 重组为 `[4998,32,embedding_dim]`。
Chronos-2 的 direct 和 mean+std 指标共同保存在
`results/task_c_survival20_chronos_direct/metrics.json` 中。

#### 20-bin 逐时间区间 MAE

下面是同一 prefix=96、同一 case split、20-bin、10-epoch 协议下的
`horizon_mae`。每个数表示该时间 bin 上的 system-level 平均绝对误差，时间边界
为从 `5` 到 `10000` 的 log-spaced edges：

| Bin | 上界时间 | Stats | Panda direct | Panda mean+std | Panda attention | Chronos direct | Chronos mean+std | Chronos attention |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 7.31 | 0.04197 | 0.03507 | 0.04858 | 0.04196 | 0.02962 | 0.04688 | 0.04452 |
| 2 | 10.69 | 0.05077 | 0.04093 | 0.06731 | 0.04862 | 0.03117 | 0.05503 | 0.04885 |
| 3 | 15.64 | 0.05426 | 0.04262 | 0.07390 | 0.05039 | 0.03060 | 0.05765 | 0.04887 |
| 4 | 22.87 | 0.05428 | 0.04069 | 0.07605 | 0.04698 | 0.02760 | 0.05568 | 0.04477 |
| 5 | 33.44 | 0.05168 | 0.03515 | 0.07667 | 0.04218 | 0.02374 | 0.05203 | 0.03892 |
| 6 | 48.90 | 0.04805 | 0.02889 | 0.07377 | 0.03620 | 0.02001 | 0.04814 | 0.03273 |
| 7 | 71.50 | 0.04255 | 0.02163 | 0.06483 | 0.02793 | 0.01542 | 0.03985 | 0.02523 |
| 8 | 104.56 | 0.03973 | 0.02108 | 0.06043 | 0.02529 | 0.01593 | 0.03395 | 0.02182 |
| 9 | 152.91 | 0.03620 | 0.01805 | 0.05699 | 0.02209 | 0.01480 | 0.02824 | 0.01938 |
| 10 | 223.61 | 0.03544 | 0.01637 | 0.05140 | 0.02036 | 0.01217 | 0.02495 | 0.01882 |
| 11 | 326.99 | 0.03451 | 0.01537 | 0.04850 | 0.01933 | 0.01319 | 0.02470 | 0.01998 |
| 12 | 478.18 | 0.03429 | 0.01614 | 0.04830 | 0.02125 | 0.01340 | 0.02479 | 0.01930 |
| 13 | 699.26 | 0.05861 | 0.03262 | 0.06274 | 0.03937 | 0.02954 | 0.04327 | 0.03539 |
| 14 | 1022.57 | 0.08302 | 0.04844 | 0.07774 | 0.05599 | 0.04388 | 0.05879 | 0.05305 |
| 15 | 1495.35 | 0.10538 | 0.06193 | 0.09334 | 0.07151 | 0.05728 | 0.07468 | 0.06940 |
| 16 | 2186.72 | 0.12301 | 0.07222 | 0.10369 | 0.08356 | 0.06665 | 0.08519 | 0.08184 |
| 17 | 3197.76 | 0.13642 | 0.08081 | 0.11264 | 0.09196 | 0.07459 | 0.09412 | 0.09071 |
| 18 | 4676.24 | 0.14747 | 0.08704 | 0.12012 | 0.09957 | 0.08149 | 0.10228 | 0.09965 |
| 19 | 6838.31 | 0.15547 | 0.08979 | 0.12444 | 0.10282 | 0.08620 | 0.10506 | 0.10386 |
| 20 | 10000 | 0.16053 | 0.09229 | 0.12715 | 0.10567 | 0.08866 | 0.10795 | 0.10682 |

逐 bin MAE 的均值、极小值和极大值如下：

| Model | Mean over 20 bins | Minimum (bin) | Maximum (bin) |
| --- | ---: | ---: | ---: |
| Stats | 0.07468 | 0.03429 (12) | 0.16053 (20) |
| Panda direct | 0.04486 | 0.01537 (11) | 0.09229 (20) |
| Panda mean+std | 0.07843 | 0.04830 (12) | 0.12715 (20) |
| Panda attention | 0.05265 | 0.01933 (11) | 0.10567 (20) |
| Chronos direct | **0.03880** | **0.01217 (10)** | **0.08866 (20)** |
| Chronos mean+std | 0.05816 | 0.02470 (11) | 0.10795 (20) |
| Chronos attention | 0.05120 | 0.01882 (10) | 0.10682 (20) |

共同趋势是 Bin 1–12 的误差逐渐下降，Bin 13 之后快速上升，说明长期 survival
预测明显更困难。当前结果中 direct 在 Panda 和 Chronos-2 中都优于对应的
mean+std 与 attention pooling；attention pooling 又稳定优于 mean+std。

#### IBS 定义

IBS（Integrated Brier Score）是所有时间 bin 上 Brier score 的平均。当前实现为：

```python
variation_error = variation_survival - variation_survival_target
brier_by_bin = np.mean(variation_error**2, axis=(0, 1))
ibs = brier_by_bin.mean()
```

它衡量完整 survival curve 的平方误差，越低越好；当前是离散 20-bin IBS，并且
对 system、32 个 variation 和时间 bin 做等权平均。相同协议下的总体 IBS 为：
这里采用当前 event-time target 的离散评估实现，不是额外使用 IPCW 的连续时间 IBS。

| Model | IBS |
| --- | ---: |
| Stats | 0.0670 |
| Panda direct | 0.0473 |
| Panda mean+std | 0.0651 |
| Panda attention | 0.0523 |
| Chronos direct | **0.0426** |
| Chronos mean+std | 0.0542 |
| Chronos attention | 0.0501 |

因此，在 Task C 当前的完整 survival-curve 任务上，预训练 Panda/Chronos 表示整体
优于手工 Stats 表示；而 Task B 是单一稳定性分类，类 SPoCK + XGBoost 仍然可以
凭借直接的轨道趋势特征取得更高 AUROC/F1。两类任务的指标和目标不同，不能直接
把 AUROC/F1 与 survival MAE/IBS 混为一个排名。

运行后会生成平均 survival curve、代表性 case、per-horizon MAE、`q_pred` vs `q_true` scatter，以及 true/pred/error heatmap 五张结果图。

这是最强调 perturbation-set 结构的任务。

**实验结果**

下面的 prefix 表格来自双头改造之前的 single-head 运行；代码结构更新后需要重新运行，不能直接把这些数值当作双头模型结果。

总体 MAE 对所有预测时间点取平均；Final MAE 对应 \(t=10000\)。
Improvement 是相对平均 survival-curve baseline 的总体 MAE 降幅。

| Model | Loss | Overall MAE | Final MAE | Improvement |
| --- | ---: | ---: | ---: | ---: |
| Baseline (average) | — | 0.3720 | 0.3478 | — |
| Direct | 0.2742 | 0.0830 | 0.1101 | 77.7% |
| Direct mean/std | 0.2640 | 0.0710 | 0.0912 | 80.9% |
| Trajectory-context | 0.2067 | 0.0657 | 0.1046 | 82.3% |
| Dual-attention | **0.1929** | **0.0529** | **0.0817** | **85.8%** |

| Model | \(t=500\) | \(t=800\) | \(t=1500\) | \(t=2500\) | \(t=4000\) | \(t=6500\) | \(t=10000\) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline (average) | 0.3776 | 0.3813 | 0.3843 | 0.3788 | 0.3726 | 0.3617 | 0.3478 |
| Direct | 0.0464 | 0.0584 | 0.0752 | 0.0875 | 0.0984 | 0.1048 | 0.1101 |
| Direct mean/std | 0.0428 | 0.0522 | 0.0652 | 0.0750 | 0.0832 | 0.0877 | 0.0912 |
| Trajectory-context | 0.0140 | 0.0347 | 0.0544 | 0.0712 | 0.0850 | 0.0962 | 0.1046 |
| Dual-attention | **0.0113** | **0.0302** | **0.0472** | **0.0578** | **0.0672** | **0.0748** | **0.0817** |

所有学习模型都显著优于平均值基线。Dual-attention 在全部 horizon 上取得
最低 MAE，总体误差相对 baseline 降低 `85.8%`，说明联合建模时间维与
32 条扰动 variation 的集合结构最有效。各模型误差总体随预测 horizon
增长，符合混沌系统长期可预测性逐渐下降的预期。

### Task D: Lyapunov-Proxy Regression

入口脚本：[tasks/task_d/main.py](/home/wan/stellar_dynamics/tasks/task_d/main.py)

底层实现：[tasks/task_d/pipeline.py](/home/wan/stellar_dynamics/tasks/task_d/pipeline.py)

**任务定义**

`Task D` 默认读取 `smaxes_complete[..., 0]` 的内轨道半长轴，构造发散 proxy：

- 对 16 个 `(i, i+16)` perturbation pairs，读取两条轨迹的内轨道半长轴
- 计算 `R_ij(t) = |a_i(t)-a_j(t)| / max(|a_i(0)-a_j(0)|, eps)`
- 定义 pair-level proxy：`tau_ij = min{t: R_ij(t) >= threshold}`；主线默认 `threshold=10000`
- 对 16 个 pair 取最早 crossing：`tau_proxy = min_ij tau_ij`
- 主训练目标为 `log(tau_proxy)`

这表示同组 16 个扰动方向中，只要一个方向达到初始差异的 `threshold` 倍，就把首次
采样到的 crossing time 作为 system-level proxy。无 crossing 的 system 是
right-censored 样本，不进入 point-regression 训练，但会保留在诊断统计中。

这是易计算的观测量 proxy，并不是由变分方程得到的严格最大 Lyapunov time。
首次离散采样 crossing 还是连续 crossing time 的上界，其时间精度受存储采样网格限制。

主实验命令：

```bash
conda run -n stellar python -m tasks.task_d \
  --task-setting initial \
  --target-mode lyapunov_proxy \
  --label-observable smax \
  --threshold 10000 \
  --aggregation min \
  --censoring-rule all_pairs \
  --dataset-dir dataset_v3 \
  --sampling-id 2 \
  --group-size 32 \
  --output-dir results/task_d_initial_lyapunov_proxy \
  --device cuda
```

`crossing_upper` 保留为 `lyapunov_proxy` 的兼容别名。原先通过增长段 slope 倒数
构造的 target 改名为 `--target-mode fitted_lyapunov_time`。

Threshold 与 pair aggregation 的完整敏感性实验：

```bash
conda run -n stellar python -m tasks.task_d.sensitivity \
  --thresholds 1000 3000 10000 30000 \
  --aggregations min median \
  --output-dir results/task_d_proxy_sensitivity \
  --device cuda
```

每个 cell 使用相同的 system split；汇总结果写入 `summary.csv`。主实验默认
threshold 已设为 `10000`，`1000/3000/30000` 用于检验结论稳定性。

```bash
conda run -n stellar python -m tasks.task_d \
  --task-setting initial \
  --label-observable smax \
  --target-mode lyapunov_proxy \
  --dataset-dir dataset_v3 \
  --sampling-id 2 \
  --group-size 32 \
  --threshold 10000 \
  --aggregation min \
  --test-size 0.2 \
  --seed 42 \
  --output-dir results/task_d_initial_lyapunov_proxy \
  --completion-table-name completed_systems_sampling_2.txt \
  --mlp-epochs 80 \
  --batch-size 128 \
  --device cuda
```

**Task D-IC：initial-condition prediction**

- 默认主实验：`--task-setting initial`
- 输入只使用 `t=0` 的 masses、质量比、轨道尺度、偏心率、倾角、能量、角动量和周期等初始条件
- 目标：`log(tau_proxy)`
- 模型：`constant_median`、`random_forest`、`mlp`

**Task D-Prefix：early-window prediction**

- 显式使用：`--task-setting prefix`
- 目标与 initial setting 相同，仍为 `log(tau_proxy)`
- 模型：`cnn_orbital`、`cnn_phase_space`、`set_cnn`
- prefix 版本应确保 crossing 发生在 `t_obs` 之后，才能解释为严格的未来预测

神经网络默认使用 `SmoothL1Loss(beta=0.5)`。评价报告 log-time MAE/RMSE、
原始 proxy-time MAE、Spearman、Pearson、factor-2/factor-3 accuracy。

**数据分辨率限制**

`dataset_v3` 的时间采样间隔约为 `50000`。不少 pair 从 `R(0)=1` 在第一个
非零采样点直接跳到 `R≈400–1500`，因此 proxy 的时间精度由首次 crossing 的
采样网格决定。

Task D 的已保存结果已经汇总在本文后面的[完整实验结果总汇](#实验结果总汇磁盘产物截至-2026-08-16)中。
需要注意：`results/task_d_initial_lyapunov_proxy/metrics.json` 是早期
`threshold=1000, aggregation=min` 运行留下的产物，而当前代码和推荐命令的主线默认值是
`threshold=10000, aggregation=min`；两者不可直接混作同一个实验。所有 proxy 仍是离散
采样 crossing time，不能等同于由 variational equations 得到的严格 Lyapunov time。

### Task E: Forecasting Baselines

相关入口：

- [tasks/task_e/main.py](/home/wan/stellar_dynamics/tasks/task_e/main.py)
推荐优先使用统一入口 `tasks/task_e/main.py`。它按 32-variation base system
完成 train/val/test group split，使用 completion-aware train-window scaler，并统一输出 transformed/z
scale 的 overall、horizon-wise、channel-wise metrics。支持
`ridge`, `tcn`, `patchtst`, `transformer`, 简化版
`itransformer`, `timesfm`, `chronos`, `chronos2` 和
`panda_zero_shot`。

当前主协议见 [`docs/task_e_pipeline.md`](/home/wan/stellar_dynamics/docs/task_e_pipeline.md)：
E1 是 one-shot `512 -> 128`，E2 是复用同一模型的四段 recursive rolling，最长
预测 horizon 为 `512`。下面的 E.0/E.1 数值是旧的 `200 -> 50` preliminary 结果，
仅保留作历史记录。

#### E.0 统一 preliminary benchmark

这组完全可比的 `orbital15` 实验使用 `512` 条轨迹、全部 `20` 个数据文件、
`context_len=200`、`pred_len=50`、`split_seed=42`、顺序采样和相同的
`1398/292/305` 个 train/validation/test windows。下表报告保存结果中的
transformed-feature scale 指标（历史字段名为 `*_raw`）；Panda Adapter
虽然在相同 validation/test windows 上评估，但只使用 `256` 个训练窗口。

| Model | Training method | Epochs | Validation MSE | Test MSE | Test MAE | Test Spearman |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Panda Adapter† | Frozen-backbone adapter | 10 | **0.098100** | **0.094817** | **0.161361** | **0.897936** |
| TimesFM zero-shot | Zero-shot | — | 0.122229 | 0.115029 | 0.177769 | 0.876202 |
| Panda zero-shot | Zero-shot | — | 0.136210 | 0.124031 | 0.198511 | 0.860429 |
| Chronos T5-small | Zero-shot | — | 0.153873 | 0.138688 | 0.165562 | 0.865887 |
| PatchTST | Fully supervised | 4 | 0.189831 | 0.172052 | 0.252842 | 0.797724 |
| iTransformer | Fully supervised | 4 | 0.193811 | 0.177432 | 0.266774 | 0.781257 |
| TCN | Fully supervised | 4 | 0.218606 | 0.199553 | 0.298014 | 0.755576 |
| Persistence | Last-value baseline | — | 0.355868 | 0.325589 | 0.309381 | 0.689615 |
| Ridge | Linear baseline | — | 0.998017 | 0.949673 | 0.583040 | 0.537908 |

† Panda Adapter 的评估窗口与表中其他模型一致，但训练数据预算更小。
表中的 TCN 是附件所汇总的 `4`-epoch 运行；仓库还保存了后续
`10`-epoch TCN，其 test MSE/MAE/Spearman 为
`0.195304/0.292288/0.761872`，详见下方链接的完整实验汇总。

Panda Adapter 在所有测试指标上最好。TimesFM 是最强 zero-shot MSE
baseline，并优于全部从零训练的监督模型；在监督模型中 PatchTST 最好。
Persistence 明显优于 Ridge，说明局部时间连续性比单个全局线性映射更适合
这组短期轨道预测。详细运行目录和指标口径见
[Task E 实验汇总](docs/task_e_experiment_summary.md)。

#### E.1 `orbital15` 与 `raw_state18` 表示对比

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

当前仅在 `orbital15` 上已有运行结果的 Panda Adapter 进一步达到
`MSE-Z=0.2115`、`MAE-Z=0.2547`、`Spearman-Z=0.8810`。这些结果表明，
在当前三体 forecasting 设置与训练预算下，预测经过物理结构化处理的
`orbital15`，整体上明显比直接预测位置和速度组成的 `raw_state18`
更容易。完整运行口径和原始结果见
[`docs/task_e_experiment_summary.md`](docs/task_e_experiment_summary.md)。

#### E.2 Foundation Forecasting

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
  - 默认 checkpoint `GilpinLab/panda-72M` 使用共享的通道 embedding 和
    forecast head，可接收 `orbital15`（15 通道）或 `raw_state18`（18 通道）
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

**实验结果**

已保存的 `pair_ic54` 主实验包含 `4998` 个 system、`79968` 个 perturbation
pair 和 `54` 个初始条件特征。下表为 7 个 horizon 上测试指标的平均值；
F1 使用 validation set 调优的逐 horizon threshold：

| Model | Mean AUPRC | Mean AUROC | Mean F1 |
| --- | ---: | ---: | ---: |
| Constant | 0.1184 | 0.5000 | 0.2114 |
| Logistic | 0.1635 | 0.6303 | 0.2555 |
| Random Forest | 0.1935 | 0.6656 | 0.2671 |
| MLP | **0.2103** | **0.7005** | **0.2982** |

MLP 在 AUPRC、AUROC 和 F1 上均为当前最佳模型，但绝对数值表明，仅凭
pair 初值判断两条扰动轨道未来是否落到稳定边界两侧仍然较难。配置中请求了
XGBoost，但该次运行没有保存 XGBoost 预测或指标，因此不列入结果表。

边界分歧本身也随 horizon 改变：

| Horizon | Pair disagreement rate | Systems with any disagreement |
| ---: | ---: | ---: |
| 500 | 0.1276 | 0.5298 |
| 800 | **0.1317** | 0.5488 |
| 1500 | 0.1303 | **0.5540** |
| 2500 | 0.1249 | 0.5432 |
| 4000 | 0.1170 | 0.5300 |
| 6500 | 0.1078 | 0.5084 |
| 10000 | 0.0987 | 0.4850 |

pair disagreement rate 在 \(t=800\) 附近最高，之后逐渐下降；至少包含一对
分歧轨道的 system 比例在 \(t=1500\) 达到峰值。原始汇总见
[Task F summary](results/task_f_boundary_sensitivity/summary.md) 和
[dataset summary](results/task_f_boundary_sensitivity/dataset_summary.json)。

## 实验结果总汇（磁盘产物，截至 2026-08-16）

本节是仓库现有实验的结果审计索引。指标以各实验目录中实际保存的
<code>metrics.json</code>、<code>summary.csv</code>、<code>test_metrics.json</code> 和
汇总 markdown 为准；训练日志、模型权重和曲线文件不逐个复制，但都保留在对应目录。
不同的标签定义、split、horizon 或输入预算不会合并到同一张排行榜。主线任务的定义和
复现实验命令仍见上文各 Task 小节。

### 0. 数据与非学习基线

主数据集规模为 160000 条 trajectory、5000 个 base systems；过滤后有 159953 条有效
trajectory 和 4998 个完整的 32-variation systems。

| 实验 | 关键设置 | 结果 |
| --- | --- | --- |
| 发散统计 | censor=10000，按 pair 取最早 divergence | raw rows/groups = 160000/5000，invalid rows = 47，排除 2 个不完整/无效 group；trajectory divergent 109553/159953 = **0.6849**；system divergent 4208/4998 = **0.8419**；trajectory divergence-time P50/P90/P99 = 255/4050/9062.4；system P50/P90/P99 = 70/2048/8255.45 |
| 传统稳定性判据 | paper Eq.(2)，group prediction 取 pair margin 最小值 | individual accuracy 0.6806，balanced accuracy 0.7099，sensitivity 0.7891，specificity 0.6307；group accuracy 0.6232，balanced accuracy 0.7336，sensitivity 0.8949，specificity 0.5722 |
| 轨道参数示例 | 6 个代表性 system，rows 0/800/1600/2400/3200/4000 | 生成 6 张 orbital-parameter 诊断图；大多数通道 finite fraction 为 1，periods_complete 在部分代表样本中存在缺失 |

原始文件：[divergence_summary.json](results/divergence_statistics/divergence_summary.json)、
[traditional_stability_summary.json](results/traditional_stability/traditional_stability_summary.json)、
[orbital examples summary](results/dataset_v3_orbital_examples/summary.json)。

### 1. Task A：初始条件到 system stability

最新 expanded rerun 使用 4998 systems、70/10/20 split、seed=42、horizon=10000、
orbital15 初始条件。stable_fraction 是 32 条 variation 的观测稳定比例；
binary_any 的正类是 32 条 variation 全部稳定。

#### 1.1 stable fraction 主结果

| Model | MSE | MAE | R2 | Spearman | MSE / binomial noise |
| --- | ---: | ---: | ---: | ---: | ---: |
| MLP-small | 0.08553 | 0.20680 | 0.42798 | 0.64393 | 46.05 |
| MLP-base | 0.09157 | 0.21485 | 0.38760 | 0.62934 | 49.30 |
| MLP-large | 0.09377 | **0.20326** | 0.37285 | 0.63198 | 50.49 |
| XGB-base | 0.08624 | 0.22388 | 0.42327 | 0.65034 | 46.43 |
| XGB-strong | **0.08336** | 0.21765 | **0.44251** | **0.66239** | **44.88** |

XGB-strong 的 MSE、R2 和 Spearman 最好；MLP-large 的 MAE 最低。所有模型误差仍显著
高于 32 次 Bernoulli 观测的采样噪声尺度，因此初始条件对长期稳定比例的可预测性仍有限。

#### 1.2 binary_any 主结果

| Model | F1 | AUROC | AUPRC | Balanced Acc | MCC | Brier | ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MLP-small | 0.43137 | 0.82748 | 0.47315 | 0.64911 | 0.36753 | 0.10756 | 0.04670 |
| MLP-base | **0.45890** | 0.82711 | 0.47549 | **0.67224** | **0.36884** | 0.10977 | 0.05861 |
| MLP-large | 0.42636 | 0.82360 | 0.45381 | 0.64733 | 0.35825 | 0.10928 | 0.05056 |
| XGB-small | 0.21053 | 0.82426 | 0.46076 | 0.55617 | 0.23279 | 0.10652 | **0.01183** |
| XGB-base | 0.29108 | **0.84806** | 0.48913 | 0.58385 | 0.26830 | 0.10256 | 0.01874 |
| XGB-strong | 0.35242 | **0.85299** | **0.50816** | 0.60936 | 0.31476 | **0.10120** | 0.01993 |

MLP-base 的阈值 F1/平衡准确率最好；XGB-strong 的 AUROC/AUPRC 和 Brier 更好。

#### 1.3 horizon grid

下表是 expanded rerun 的全部 horizon grid；stable fraction 的列为 MLP/XGB 的
MSE、MAE、R2，后两列为 binary_any 的 F1。

| Horizon | Stable frac. | MLP MSE/MAE/R2 | XGB MSE/MAE/R2 | MLP F1 | XGB F1 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1000 | 0.4888 | 0.08518/0.21465/0.50121 | 0.08666/0.23410/0.49256 | **0.59696** | 0.58614 |
| 2000 | 0.4233 | 0.09338/0.23548/0.45006 | 0.08936/0.23724/0.47374 | **0.58093** | 0.53944 |
| 3000 | 0.3878 | 0.09136/0.22603/0.45177 | 0.08911/0.23702/0.46528 | **0.49468** | 0.44575 |
| 4000 | 0.3653 | 0.09252/0.22249/0.43448 | 0.08874/0.23494/0.45759 | **0.49102** | 0.45847 |
| 5000 | 0.3474 | 0.09264/0.22798/0.42540 | 0.08906/0.23399/0.44758 | **0.46207** | 0.38849 |
| 6000 | 0.3331 | 0.09090/0.22298/0.42599 | 0.08838/0.23155/0.44193 | **0.52514** | 0.43011 |
| 7000 | 0.3218 | 0.09174/0.22251/0.41207 | 0.08766/0.22889/0.43818 | **0.34959** | 0.29661 |
| 8000 | 0.3130 | 0.09395/0.22843/0.38945 | 0.08798/0.22820/0.42824 | **0.43636** | 0.30508 |
| 9000 | 0.3046 | 0.09074/0.21862/0.40150 | 0.08719/0.22612/0.42492 | **0.45038** | 0.32743 |
| 10000 | 0.2974 | 0.09157/0.21485/0.38760 | 0.08624/0.22388/0.42327 | **0.45890** | 0.29108 |

短 horizon 独立 grid（同一实验的 short rerun）得到 stable fraction mean：
100/200/500/1000/2000 = 0.745861/0.682454/0.587529/0.513718/0.445616；
对应 MLP MSE = 0.051570/0.063825/0.076093/0.088373/0.090029，
XGB MSE = 0.049802/0.059510/0.074599/0.086658/0.089360。

#### 1.4 loss 与 feature ablation

| Ablation | Regression MSE/MAE/R2 | Binary F1/AUROC/AUPRC |
| --- | --- | --- |
| BCE, all orbital15 | 0.09157/0.21485/0.38760 | 0.45890/0.82711/0.47549 |
| MSE-sigmoid, all orbital15 | not saved / not saved / not saved | 0.42969/0.81445/0.47010 |
| no AP/LAN angles | 0.08787/0.21887/0.41233 | 0.30189/0.85104/0.50852 |
| no eccentricity/inclination | 0.11714/0.26340/0.21656 | 0.19718/0.71838/0.30933 |
| aratio+mratio+ecc | 0.08751/0.22204/0.41476 | 0.24510/0.84400/0.48692 |
| aratio+mratio | 0.11588/0.26706/0.22499 | 0.00000/0.74106/0.33754 |

MSE-sigmoid 行的回归指标未在该 ablation 汇总中保存，不能用 binary 指标反推。
作为历史 CPU reference，旧版 summary 的 stable_fraction 指标为：

| Model | MSE | RMSE | MAE | R2 |
| --- | ---: | ---: | ---: | ---: |
| MLP-small | 0.08489 | 0.29135 | 0.20280 | 0.43229 |
| MLP-base | 0.09184 | 0.30305 | 0.21867 | 0.38579 |
| MLP-large | 0.08799 | 0.29662 | 0.20930 | 0.41156 |
| XGB-base | 0.08624 | 0.29366 | 0.22388 | 0.42327 |
| XGB-strong | 0.08336 | 0.28872 | 0.21765 | 0.44251 |

同一旧版 run 的 binary_any 指标（Acc/Prec/Recall/F1/AUROC/AP）为：

| Model | Acc | Prec | Recall | F1 | AUROC | AP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MLP-small | 0.855 | 0.562 | 0.373 | 0.449 | 0.836 | 0.481 |
| MLP-base | 0.852 | 0.550 | 0.348 | 0.426 | 0.825 | 0.481 |
| MLP-large | 0.851 | 0.545 | 0.342 | 0.420 | 0.823 | 0.467 |
| XGB-small | 0.850 | 0.625 | 0.127 | 0.211 | 0.824 | 0.461 |
| XGB-base | 0.849 | 0.564 | 0.196 | 0.291 | 0.848 | 0.489 |
| XGB-strong | 0.853 | 0.580 | 0.253 | 0.352 | 0.853 | 0.508 |

完整文件：[Task A legacy summary](results/task_a_experiments/summary.md)、
[expanded summary](results/task_a_experiments/metrics_augmented/summary.md)、
[horizon grid](results/task_a_experiments/metrics_augmented/horizon_grid/summary.md)、
[short horizon grid](results/task_a_experiments/metrics_augmented/horizon_grid_short/summary.md)。
其中 <code>h=20000</code> 的边界测试已经超过数据中 <code>time_unbound</code> 最大值，
不纳入正式比较；若目录中只有 config 或 smoke log，也不伪造最终指标。

### 2. Task B：trajectory prefix instability classification

标准 prefix ablation 使用 dataset v3、orbital15、horizon=10000、trajectory_binary、
group split、测试集不稳定比例约 31.6%。以下是
[results/task_b_prefix_ablation/readme.md](results/task_b_prefix_ablation/readme.md) 中
<code>metrics.json</code> 的完整结果；表中 prefix=512 是后来补跑的部分模型。

| Prefix | Model | Acc | Prec | Recall | F1 | AUROC | AUPRC |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | SVM | 0.7765 | 0.6964 | 0.5195 | 0.5951 | 0.8256 | 0.6657 |
| 10 | CNN | 0.7522 | 0.6210 | 0.5548 | 0.5860 | 0.8179 | 0.6539 |
| 10 | TCN | 0.7536 | 0.6840 | 0.4100 | 0.5127 | 0.8127 | 0.6514 |
| 10 | GRU | 0.7623 | 0.6552 | 0.5237 | 0.5821 | 0.8176 | 0.6573 |
| 10 | Attention | 0.7543 | 0.6326 | 0.5317 | 0.5778 | 0.8039 | 0.6491 |
| 20 | SVM | 0.7749 | 0.6930 | 0.5170 | 0.5922 | 0.8223 | 0.6552 |
| 20 | CNN | 0.7771 | 0.6491 | 0.6425 | 0.6458 | 0.8374 | 0.6849 |
| 20 | TCN | 0.7842 | 0.6971 | 0.5613 | 0.6219 | 0.8397 | 0.6819 |
| 20 | GRU | 0.7660 | 0.6639 | 0.5263 | 0.5872 | 0.8256 | 0.6544 |
| 20 | Attention | 0.7756 | 0.6449 | 0.6463 | 0.6456 | 0.8378 | 0.6493 |
| 50 | SVM | 0.7649 | 0.6658 | 0.5150 | 0.5808 | 0.8232 | 0.6387 |
| 50 | CNN | 0.7935 | 0.6750 | 0.6693 | 0.6721 | 0.8671 | 0.7188 |
| 50 | TCN | 0.7897 | 0.7115 | 0.5632 | 0.6287 | 0.8697 | 0.7158 |
| 50 | GRU | 0.7703 | 0.6828 | 0.5107 | 0.5843 | 0.8523 | 0.7129 |
| 50 | Attention | 0.7946 | 0.6940 | 0.6270 | 0.6588 | 0.8673 | 0.7121 |
| 100 | SVM | 0.7687 | 0.6659 | 0.5385 | 0.5955 | 0.8361 | 0.6387 |
| 100 | CNN | 0.8130 | 0.6895 | 0.7435 | 0.7154 | 0.8890 | 0.7503 |
| 100 | TCN | 0.8045 | 0.7273 | 0.6104 | 0.6637 | 0.8922 | **0.7642** |
| 100 | GRU | 0.7997 | **0.7532** | 0.5453 | 0.6326 | 0.8908 | 0.7420 |
| 100 | Attention | 0.7904 | 0.7156 | 0.5597 | 0.6281 | 0.8890 | 0.7322 |
| 200 | SVM | 0.7723 | 0.6611 | 0.5743 | 0.6147 | 0.8418 | 0.6476 |
| 200 | CNN | 0.8370 | 0.7268 | 0.7761 | 0.7507 | 0.9133 | 0.7947 |
| 200 | GRU | 0.8333 | 0.7535 | 0.7028 | 0.7273 | 0.9186 | 0.7848 |
| 200 | TCN | **0.8622** | 0.7575 | **0.8297** | **0.7920** | 0.9321 | 0.8293 |
| 200 | Attention | 0.8420 | 0.7507 | 0.7490 | 0.7498 | 0.9208 | 0.7845 |
| 300 | SVM | 0.7868 | 0.6681 | 0.6474 | 0.6576 | 0.8503 | 0.6634 |
| 300 | CNN | 0.8619 | 0.7594 | 0.8245 | 0.7906 | 0.9282 | 0.8145 |
| 300 | GRU | 0.8605 | 0.7990 | 0.7466 | 0.7719 | 0.9366 | 0.8219 |
| 300 | TCN | 0.8697 | 0.8123 | 0.7644 | 0.7876 | 0.9440 | **0.8654** |
| 300 | Attention | **0.8737** | 0.7546 | **0.8897** | **0.8166** | 0.9350 | 0.8158 |
| 512 | SVM | 0.7849 | 0.6815 | 0.6006 | 0.6385 | 0.8442 | 0.6684 |
| 512 | CNN | 0.8934 | 0.8153 | 0.8571 | 0.8357 | 0.9508 | 0.8654 |
| 512 | TCN | **0.9039** | **0.8425** | 0.8560 | **0.8492** | **0.9599** | **0.8919** |

结论：在 prefix 10–300 的完整网格中，长 prefix 明显有利；prefix=200 的 TCN
达到 F1=0.7920，prefix=300 的 Attention 达到 Recall=0.8897，prefix=512 的
补跑 TCN 达到 F1=0.8492、AUROC=0.9599。prefix=512 没有 GRU/Attention 的最终
<code>metrics.json</code>，因此不把缺失结果写成 0。

上表与本文前面 Task B 的旧版 100/200 摘要存在小数点末位差异，原因是历史运行的
训练/保存时点不同；这里以当前 prefix 目录中的指标文件为准。

#### Task B：类 SPoCK 统计特征与 frozen encoder 对比

为了复现“类 SPoCK”思路，prefix=96 的 `orbital15` 每个通道提取
`initial, final, mean, std, slope, min, max, delta` 八个统计量，共 `15×8=120`
维，再使用 XGBoost 分类。它是 prefix-only 的 SPoCK-inspired baseline，不运行额外
N-body integration，也不直接计算原始 SPoCK 的 MEGNO 特征。

| Representation / head | AUROC | Unbound F1 | Stable F1 |
| --- | ---: | ---: | ---: |
| Panda frozen embedding + linear head | 0.94545 | 0.89840 | 0.80812 |
| Chronos-2 frozen embedding + linear head | 0.95076 | 0.89689 | 0.81051 |
| SPoCK-inspired statistics + XGBoost | **0.96253** | **0.92525** | **0.83479** |

这说明类 SPoCK 的手工统计特征在 Task B 的单一稳定性分类目标上很有竞争力；它与
Task C 的完整 survival-curve 预测不是同一个目标，不能直接用 AUROC/F1 与
20-bin MAE/IBS 做数值比较。

### 3. Task C：multi-horizon survival 与 ablation

Task C ablation 使用 4998 systems、prefix lengths 10/20/50/100、variation counts
1/2/4/8/16/32、future times
500/800/1500/2500/4000/6500/10000。每个 cell 都是嵌套 variation 子集，并预测完整
32 条 variation 的 system survival fraction。表中数字是测试集 overall MAE，列顺序为
baseline/direct/traj/dual。

| Prefix | Variations | Baseline | Direct | Traj | Dual |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 1 | 0.3720 | 0.2216 | 0.2377 | 0.2381 |
| 10 | 2 | 0.3720 | 0.2169 | 0.2272 | 0.2428 |
| 10 | 4 | 0.3720 | 0.2024 | 0.2158 | 0.2277 |
| 10 | 8 | 0.3720 | 0.1868 | 0.2107 | 0.2026 |
| 10 | 16 | 0.3720 | 0.1875 | 0.2175 | 0.1903 |
| 10 | 32 | 0.3720 | 0.1805 | 0.1880 | 0.1766 |
| 20 | 1 | 0.3720 | 0.2159 | 0.2374 | 0.2484 |
| 20 | 2 | 0.3720 | 0.2075 | 0.2166 | 0.2301 |
| 20 | 4 | 0.3720 | 0.1827 | 0.1967 | 0.2085 |
| 20 | 8 | 0.3720 | 0.1647 | 0.1992 | 0.1808 |
| 20 | 16 | 0.3720 | 0.1644 | 0.1795 | 0.1611 |
| 20 | 32 | 0.3720 | 0.1548 | 0.1747 | 0.1512 |
| 50 | 1 | 0.3720 | 0.2141 | 0.2249 | 0.2256 |
| 50 | 2 | 0.3720 | 0.1802 | 0.1954 | 0.1989 |
| 50 | 4 | 0.3720 | 0.1544 | 0.1682 | 0.1620 |
| 50 | 8 | 0.3720 | 0.1426 | 0.1556 | 0.1295 |
| 50 | 16 | 0.3720 | 0.1356 | 0.1437 | 0.1175 |
| 50 | 32 | 0.3720 | 0.1305 | 0.1279 | 0.1057 |
| 100 | 1 | 0.3720 | 0.1938 | 0.1995 | 0.2104 |
| 100 | 2 | 0.3720 | 0.1505 | 0.1741 | 0.1680 |
| 100 | 4 | 0.3720 | 0.1305 | 0.1443 | 0.1316 |
| 100 | 8 | 0.3720 | 0.1212 | 0.1356 | 0.1090 |
| 100 | 16 | 0.3720 | 0.1140 | 0.1171 | 0.0946 |
| 100 | 32 | 0.3720 | **0.1061** | **0.0971** | **0.0748** |

各模型在最佳输入配置 prefix=100、variations=32 下的 horizon-wise MAE：

| Model | 500 | 800 | 1500 | 2500 | 4000 | 6500 | 10000 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 0.377554 | 0.381327 | 0.384258 | 0.378834 | 0.372599 | 0.361742 | 0.347804 |
| Direct | 0.083846 | 0.091997 | 0.102430 | 0.110449 | 0.115667 | 0.118593 | 0.119743 |
| Traj | 0.064176 | 0.073784 | 0.086962 | 0.099471 | 0.109715 | 0.118828 | 0.126826 |
| Dual | **0.047085** | **0.055776** | **0.067096** | **0.075407** | **0.085552** | **0.093667** | **0.098697** |

整体最优为 dual（overall MAE=0.0748），其次是 traj（0.0971）和 direct（0.1061）；
baseline 的 overall MAE 为 0.3720。另有 initial-condition MLP 对照：
baseline overall/final = 0.3720169/0.3478040，IC-MLP =
0.2451511/0.2435258，IC-MLP 各 horizon MAE =
0.229286/0.239105/0.250980/0.251613/0.252635/0.248913/0.243526。

另一个已保存的 single-head prefix run（与当前 96-cell ablation 不是同一组实现）结果为：

| Model | Overall MAE | Final MAE | Improvement |
| --- | ---: | ---: | ---: |
| Baseline | 0.3720 | 0.3478 | — |
| Direct | 0.0830 | 0.1101 | 77.7% |
| Direct mean/std | 0.0710 | 0.0912 | 80.9% |
| Trajectory-context | 0.0657 | 0.1046 | 82.3% |
| Dual-attention | **0.0529** | **0.0817** | **85.8%** |

该 single-head run 的逐 horizon MAE 为：

| Model | 500 | 800 | 1500 | 2500 | 4000 | 6500 | 10000 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 0.3776 | 0.3813 | 0.3843 | 0.3788 | 0.3726 | 0.3617 | 0.3478 |
| Direct | 0.0464 | 0.0584 | 0.0752 | 0.0875 | 0.0984 | 0.1048 | 0.1101 |
| Direct mean/std | 0.0428 | 0.0522 | 0.0652 | 0.0750 | 0.0832 | 0.0877 | 0.0912 |
| Trajectory-context | 0.0140 | 0.0347 | 0.0544 | 0.0712 | 0.0850 | 0.0962 | 0.1046 |
| Dual-attention | **0.0113** | **0.0302** | **0.0472** | **0.0578** | **0.0672** | **0.0748** | **0.0817** |

完整 96-cell 结果、输入预算表和每个 horizon 的原始值见
[Task C ablation README](results/task_c_ablation/readme.md)、
[Task C ablation summary.csv](results/task_c_ablation/summary.csv) 和
[initial-condition MLP metrics](results/task_c_initial_condition_mlp/metrics.json)。

### 4. Task D：Lyapunov-proxy / lambda-fast 回归

Task D 的 proxy 标签由 16 个 pair 的离散 crossing time 聚合得到；数据采样间隔约为
50000，因此这里的时间是“首次存储采样点”而不是连续 crossing time。下面先列出已经
保存的 initial/prefix 结果，再列出 threshold × aggregation 敏感性网格。

#### 4.1 已保存的 initial/prefix 结果

表中的 MAE/RMSE 都是在 log-time 上计算；F2/F3 表示预测落在真实值 2 倍/3 倍范围内
的比例。lambda_fast 与 lyapunov_proxy 是两种不同 target，不能直接横向混排。

| Run | Target / setting | Representation | Labels | Censoring | Model | MAE | RMSE | Spearman | F2 / F3 |
| --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| D-lambda-raw | lambda_fast / initial | raw IC | 3199 | 0.0004 before quality filter | constant median | 0.958787 | 1.757718 | — | 0.6375/0.8094 |
| D-lambda-raw | lambda_fast / initial | raw IC | 3199 | 0.0004 before quality filter | random forest | 1.074405 | 1.622455 | 0.208866 | 0.4688/0.6875 |
| D-lambda-raw | lambda_fast / initial | raw IC | 3199 | 0.0004 before quality filter | MLP | 0.962735 | 1.645267 | 0.235682 | 0.5734/0.7625 |
| D-lambda-orbital15 | lambda_fast / initial | orbital15 | 3199 | 0.0004 before quality filter | constant median | 0.958787 | 1.757718 | — | 0.6375/0.8094 |
| D-lambda-orbital15 | lambda_fast / initial | orbital15 | 3199 | 0.0004 before quality filter | random forest | 1.092392 | 1.654524 | 0.188678 | 0.4469/0.6500 |
| D-lambda-orbital15 | lambda_fast / initial | orbital15 | 3199 | 0.0004 before quality filter | MLP | 1.021682 | 1.679629 | 0.197611 | 0.5266/0.7297 |
| D-proxy-initial | lyapunov_proxy, threshold=1000, min | orbital15 | 4997 | 0.0002 | constant median | 1.000961 | 1.278772 | — | 0.3590/0.7680 |
| D-proxy-initial | lyapunov_proxy, threshold=1000, min | orbital15 | 4997 | 0.0002 | random forest | 0.748865 | 0.984519 | 0.667342 | 0.5630/0.7560 |
| D-proxy-initial | lyapunov_proxy, threshold=1000, min | orbital15 | 4997 | 0.0002 | MLP | **0.652977** | **0.878598** | **0.721417** | **0.6300/0.8200** |
| D-prefix | lambda_fast / initial_prefix, prefix=100 | orbital15 | 4998 | 0.886154 | constant median | 1.642400 | 2.607517 | — | 0.3610/0.5070 |
| D-prefix | lambda_fast / initial_prefix, prefix=100 | orbital15 | 4998 | 0.886154 | random forest | **1.237786** | **2.084188** | **0.657684** | **0.4710/0.6240** |
| D-prefix | lambda_fast / initial_prefix, prefix=100 | orbital15 | 4998 | 0.886154 | MLP | 1.647709 | 3.003794 | 0.495328 | 0.3470/0.5320 |

当前代码推荐的 threshold=10000 版本不是上面 D-proxy-initial 目录中的历史
threshold=1000 结果；推荐使用后面的 sensitivity cell 或重新运行主命令后再作正式比较。

#### 4.2 threshold / aggregation sensitivity

标签分布（每个 cell 都有 4998 个 systems）：

| Threshold | Aggregation | Uncensored | Censored fraction | Median tau | P90 tau |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1000 | median | 4992 | 0.001200 | 300000 | 3957500 |
| 1000 | min | 4997 | 0.000200 | 150000 | 1100000 |
| 3000 | median | 4974 | 0.004802 | 350000 | 5550000 |
| 3000 | min | 4989 | 0.001801 | 150000 | 1300000 |
| 10000 | median | 4921 | 0.015406 | 387500 | 6590000 |
| 10000 | min | 4968 | 0.006002 | 150000 | 1565000 |
| 30000 | median | 4762 | 0.047219 | 425000 | 7615000 |
| 30000 | min | 4934 | 0.012805 | 200000 | 1800000 |

每个 cell 的所有模型指标：

| Threshold | Agg. | Model | MAE(log tau) | RMSE | Spearman | F2 | F3 |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1000 | median | constant | 1.2960 | 1.5767 | — | 0.2442 | 0.5385 |
| 1000 | median | MLP | **0.7882** | **1.0205** | **0.7957** | **0.5405** | **0.7558** |
| 1000 | median | RF | 0.9099 | 1.1499 | 0.7005 | 0.4725 | 0.6757 |
| 1000 | min | constant | 1.0010 | 1.2788 | — | 0.3590 | 0.7680 |
| 1000 | min | MLP | **0.6794** | **0.9105** | **0.7158** | **0.6230** | **0.8050** |
| 1000 | min | RF | 0.7489 | 0.9845 | 0.6673 | 0.5630 | 0.7560 |
| 3000 | median | constant | 1.3552 | 1.6973 | — | 0.2673 | 0.4523 |
| 3000 | median | MLP | **0.8769** | **1.1768** | **0.7579** | **0.5206** | **0.6965** |
| 3000 | median | RF | 0.9918 | 1.2911 | 0.6812 | 0.4503 | 0.6332 |
| 3000 | min | constant | 1.0379 | 1.3253 | — | 0.3417 | 0.7365 |
| 3000 | min | MLP | **0.6798** | **0.9164** | **0.7354** | **0.6182** | **0.8116** |
| 3000 | min | RF | 0.7665 | 1.0148 | 0.6527 | 0.5521 | 0.7595 |
| 10000 | median | constant | 1.3625 | 1.7322 | — | 0.2853 | 0.4426 |
| 10000 | median | MLP | **0.8712** | **1.2209** | **0.7621** | **0.5350** | **0.7411** |
| 10000 | median | RF | 1.0037 | 1.3205 | 0.6831 | 0.4508 | 0.6487 |
| 10000 | min | constant | 1.1449 | 1.5105 | — | 0.3239 | 0.6992 |
| 10000 | min | MLP | **0.7297** | **0.9992** | **0.7891** | **0.6137** | **0.7907** |
| 10000 | min | RF | 0.8304 | 1.1433 | 0.6734 | 0.5151 | 0.7334 |
| 30000 | median | constant | 1.3171 | 1.6694 | — | 0.2844 | 0.4533 |
| 30000 | median | MLP | **0.8599** | **1.1857** | **0.7515** | **0.5331** | **0.7272** |
| 30000 | median | RF | 0.9898 | 1.3002 | 0.6574 | 0.4407 | 0.6537 |
| 30000 | min | constant | 1.1276 | 1.4417 | — | 0.2523 | 0.5137 |
| 30000 | min | MLP | **0.7644** | **1.0704** | **0.7383** | **0.5735** | **0.7771** |
| 30000 | min | RF | 0.8504 | 1.1654 | 0.6490 | 0.5360 | 0.7224 |

在当前 sensitivity 网格里，min aggregation 通常保留更多 uncensored labels，MLP 的
log-time MAE 约为 0.679–0.764；median aggregation 的 censoring 更高，MLP MAE
约为 0.788–0.877。完整原始表见
[task_d_proxy_sensitivity/summary.csv](results/task_d_proxy_sensitivity/summary.csv)。
各 initial/prefix run 的 config、feature importance 和 predictions 分别保存在：
[lambda-fast raw](results/task_d_initial_lambda_fast/metrics.json)、
[lambda-fast orbital15](results/task_d_initial_lambda_fast_orbital15/metrics.json)、
[lyapunov proxy](results/task_d_initial_lyapunov_proxy/metrics.json)、
[prefix](results/task_d_initial_prefix_lambda_fast_orbital15/metrics.json)。

#### 4.3 Task D feature importance

已保存 feature importance 中的第一重要特征为：

| Run | Top feature | Importance |
| --- | --- | ---: |
| lambda_fast raw | periods_complete_0 | 0.132797 |
| lambda_fast orbital15 | eccs_complete_1 | 0.143158 |
| lyapunov proxy | incs_complete_0 | 0.256413 |
| prefix lambda_fast | early_log_R_q90_1 | 0.137737 |

完整排名见各目录中的 <code>feature_importance.csv</code>；它们说明不同 target/输入设定
依赖的物理摘要并不相同。

### 5. Task E：forecasting 与 foundation-model 对比

#### 5.1 统一 benchmark

统一 benchmark 使用 orbital15、context=200、prediction=50、512 tracks、20 shards、
group split seed=42，train/validation/test windows=1398/292/305。指标是 transformed
feature scale；历史文件中的 <code>*_raw</code> 字段对应此 scale。

| Model | Training | Val MSE | Test MSE | Test MAE | Test Spearman |
| --- | --- | ---: | ---: | ---: | ---: |
| Panda Adapter | frozen-backbone adapter | **0.098100** | **0.094817** | 0.161361 | **0.897936** |
| TimesFM zero-shot | zero-shot | 0.122229 | 0.115029 | 0.177769 | 0.876202 |
| Panda zero-shot | zero-shot | 0.136210 | 0.124031 | 0.198511 | 0.860429 |
| Chronos T5-small | zero-shot | 0.153873 | 0.138688 | **0.165562** | 0.865887 |
| PatchTST | supervised | 0.189831 | 0.172052 | 0.252842 | 0.797724 |
| iTransformer | supervised | 0.193811 | 0.177432 | 0.266774 | 0.781257 |
| TCN, 4 epochs | supervised | 0.218606 | 0.199553 | 0.298014 | 0.755576 |
| Persistence | last value | 0.355868 | 0.325589 | 0.309381 | 0.689615 |
| Ridge | linear | 0.998017 | 0.949673 | 0.583040 | 0.537908 |

另一个 10-epoch TCN run 的 test MSE/MAE/Spearman =
0.195304/0.292288/0.761872。Panda Adapter 的 test window 与其他模型相同，
但只使用 256 个训练窗口，不能把它解释为相同训练预算下的公平比较。

#### 5.2 raw_state18 matched benchmark

raw_state18 对照使用 512 tracks、context=200、prediction=50，train/validation/test =
1400/292/306；以下指标是在 standardized z scale 上。

| Model | MSE-z | MAE-z | Spearman-z |
| --- | ---: | ---: | ---: |
| TimesFM | **0.525152** | **0.349141** | 0.424185 |
| Chronos T5-small | 0.580466 | 0.355880 | **0.428264** |
| TCN | 0.623281 | 0.431166 | 0.220818 |
| iTransformer | 0.623715 | 0.419106 | 0.193764 |
| PatchTST | 0.626541 | 0.419600 | 0.197222 |
| Transformer | 0.906040 | 0.481039 | 0.155406 |
| Persistence | 1.054121 | 0.525768 | 0.204258 |
| Ridge | 2.595065 | 0.965088 | 0.099415 |

#### 5.3 context=64 / prediction=32 horizon 对照

这一组固定 orbital15、context=64、pred=32、stride=20、每 system 最多 2 条
variation 和每条轨迹最多 2 个 sequential windows，test 共 2556 windows。

| Model | MSE-z | MAE-z | Spearman-z | MSE transformed | MAE transformed | Spearman transformed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Chronos T5-small | 0.3984 | 0.3180 | 0.7893 | 0.1800 | 0.2030 | 0.8244 |
| Panda GRU | 0.3717 | 0.4108 | 0.7726 | 0.1671 | 0.2589 | 0.8004 |
| Panda GRU early-stop | 0.3417 | 0.3787 | 0.7949 | 0.1530 | 0.2387 | 0.8223 |
| Panda MLP, epoch 15 | **0.2904** | 0.3227 | **0.8308** | **0.1290** | **0.2038** | **0.8531** |
| Panda zero-shot | 0.3847 | 0.3688 | 0.7755 | 0.1665 | 0.2320 | 0.8146 |
| TimesFM | 0.3305 | **0.3208** | 0.8132 | 0.1482 | **0.2052** | 0.8405 |

同一 horizon run 的 GRU early-stop test overall 为
MSE-z/MAE-z/RMSE-z = 0.341675/0.378713/0.584530，transformed
MSE/MAE/Spearman = 0.152961/0.238729/0.822295；step 1 与 step 32 的 MSE-z
为 0.2187/0.4128。MLP epoch 15 为
0.290387/0.322718/0.538875 和 0.128972/0.203784/0.853114，step 1 与
step 32 的 MSE-z 为 0.0636/0.4214。

#### 5.4 其他统一入口运行

| Run | Test result |
| --- | --- |
| Panda Adapter small-window | MSE 0.098633，MAE 0.169172，Spearman 0.891364；Val MSE 0.100773，best epoch 4 |
| Panda Adapter resume | MSE 0.098756，MAE 0.168499，Spearman 0.891965；Val MSE 0.098271，best epoch 8 |
| Panda zero-shot non-comparable | MSE 0.122947，MAE 0.198808，Spearman 0.861434 |
| Preliminary persistence | MSE 0.336771，MAE 0.317444，Spearman 0.675598 |
| Historical independent Panda adapter | z-MSE 0.173401，z-MAE 0.191221，raw-MSE 1992.260254，raw-MAE 2.509673；epoch 10 val MSE 1788.239416 |

完整说明见 [docs/task_e_experiment_summary.md](docs/task_e_experiment_summary.md) 和
[README_task_e_horizon_results.md](README_task_e_horizon_results.md)。只有 config/feature
metadata 而没有最终 test metrics 的目录（例如
<code>runs/task_e_orbital15_ctx64_pred32_panda_gru</code>、
<code>runs/task_e_orbital15_ctx64_pred32_panda_mlp_2var2win</code>）标记为未完成，不列入排名。

### 6. Task F：stability-boundary sensitivity

Task F 的主实验是 pair_ic54、4998 systems、7 个 horizons；下表是保存的 pair disagreement
统计，MLP 的 mean AUPRC/AUROC/F1 = 0.2103/0.7005/0.2982，为四个已保存模型中最高。

| Model | Mean AUPRC | Mean AUROC | Mean F1 |
| --- | ---: | ---: | ---: |
| Constant | 0.1184 | 0.5000 | 0.2114 |
| Logistic | 0.1635 | 0.6303 | 0.2555 |
| Random Forest | 0.1935 | 0.6656 | 0.2671 |
| MLP | **0.2103** | **0.7005** | **0.2982** |

| Horizon | Pair disagreement | Systems with any disagreement |
| ---: | ---: | ---: |
| 500 | 0.1276 | 0.5298 |
| 800 | **0.1317** | 0.5488 |
| 1500 | 0.1303 | **0.5540** |
| 2500 | 0.1249 | 0.5432 |
| 4000 | 0.1170 | 0.5300 |
| 6500 | 0.1078 | 0.5084 |
| 10000 | 0.0987 | 0.4850 |

pair disagreement 在 horizon=800 附近最高，system-level any-disagreement 在 1500 达到
峰值。配置请求了 XGBoost，但该次运行没有保存 XGBoost prediction/metrics，因此不能
把它加入比较。原始文件：[Task F summary](results/task_f_boundary_sensitivity/summary.md)、
[dataset summary](results/task_f_boundary_sensitivity/dataset_summary.json)。

### 7. Panda 独立实验区

以下结果来自 <code>experiments/panda_threebody</code> 的历史/隔离 pipeline，和主项目
Task E 的统一窗口 benchmark 分开记录。

#### 7.1 unbound classifier

| Run | Test n | Positive rate | Acc | Precision | Recall | F1 | AUROC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| legacy orbital15 dataset | 8000 | 0.642625 | 0.884375 | 0.968861 | 0.847306 | 0.904016 | 0.950854 |
| Panda orbital15 dataset | 8000 | 0.642625 | **0.939250** | 0.966153 | **0.938339** | **0.952043** | **0.978368** |
| Panda patch-mean keep-channels | 32032 | 0.684253 | 0.922203 | **0.969908** | 0.914682 | 0.941486 | 0.977106 |

对应的 Task-B split（test 指标；文件中同时保存 unbound-positive 和互补的
stable-positive）如下。这里的 head 以各目录的 <code>run_config.json</code> 为准：

| Run / head | Unbound Acc | Unbound Prec | Unbound Recall | Unbound F1 | AUROC | Unbound AP | Stable F1 | Stable AP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| prefix96 / linear | 0.508281 | 0.684763 | 0.520562 | 0.591478 | 0.500254 | 0.684630 | 0.382532 | 0.316294 |
| prefix96 aligned / linear | 0.867150 | 0.941606 | 0.858993 | 0.898404 | 0.945449 | 0.974732 | 0.808123 | 0.883745 |
| prefix512 / main | 0.928897 | 0.962739 | 0.932094 | **0.947169** | 0.978716 | 0.990267 | 0.891303 | 0.954518 |
| prefix512 attention / attention | 0.921541 | **0.978409** | 0.905239 | 0.940403 | 0.979344 | 0.990641 | 0.885212 | 0.956522 |
| prefix512 DeepSets / deepsets | 0.923793 | 0.974077 | **0.912849** | 0.942470 | **0.979450** | **0.990742** | 0.887160 | 0.956113 |

#### 7.2 小样本 forecasting

| Training mode | Prediction length | MSE | MAE | sMAPE |
| --- | ---: | ---: | ---: | ---: |
| fine-tune orbital15 | 64 | 4.33478e9 | 4043.69 | 57.3918 |
| fine-tune orbital15 | 128 | 5.60501e9 | 4347.59 | 61.6773 |
| scratch orbital15 | 64 | 2.09162e10 | 16308.8 | 96.4610 |
| scratch orbital15 | 128 | 2.20187e10 | 16967.0 | 96.5859 |
| zero-shot orbital15 | 64 | 4.17944e9 | 3950.24 | 55.5869 |
| zero-shot orbital15 | 128 | 6.29700e9 | 4515.44 | 60.3770 |

详细路径和训练配置见 [experiments/panda_threebody/reports/results_summary.md](experiments/panda_threebody/reports/results_summary.md)、
[Panda classifier metrics](experiments/panda_threebody/outputs/panda_unbound_dataset/panda_unbound_orbital15_dataset/metrics.json)
及各 <code>panda_task_b_split_*</code> 子目录。

### 8. Historical notebook-only 结果

这些结果来自 notebook 的已保存输出，标签和 split 与当前 production runner 不完全相同，
所以只作为历史记录，不与上面的正式表直接排序。

| Notebook | 设置 | 已保存结果 |
| --- | --- | --- |
| task_a.ipynb | system label = “32 条 variation 中任意一条稳定”，train/test=3998/1000 | MLP AUC/AP/F1/Acc/Prec/Recall = 0.8489/0.9029/0.8434/0.7890/0.8280/0.8593；XGB = 0.8458/0.9045/0.8478/0.7900/0.8136/0.8850 |
| task_b.ipynb | soft-time / early-prefix，64 bins，train/val/test=125963/13996/19994 | notebook 最终输出 AUC=0.9689，F1=0.8548，Acc=0.9039；模型名未在保存输出中可靠记录 |
| task_c.ipynb | 9 horizons = 1100/1500/2000/3000/4000/5000/7000/8500/10000，prefix=200 | baseline overall/final MAE=0.3735/0.3480；direct=0.0768/0.0857；trajectory-context=0.0748/0.1076；dual=0.0627/0.0947 |
| task_d.ipynb | notebook 说明 threshold=10000、lyapunov proxy | 没有保存可复核的 stream metrics；对应目录中的旧 metrics 实际是 threshold=1000 |

Notebook 文件：[task_a.ipynb](jupyter_notebook/task_a.ipynb)、
[task_b.ipynb](jupyter_notebook/task_b.ipynb)、
[task_c.ipynb](jupyter_notebook/task_c.ipynb)、
[task_d.ipynb](jupyter_notebook/task_d.ipynb)。

### 9. 未完成、部分完成与不纳入正式排名的运行

- Task A 的 <code>fraction_mlp_base_cuda</code>、<code>fraction_mlp_cuda_smoke</code> 没有
  可用最终 metrics；<code>h=20000</code> 只属于超出数据有效范围的边界测试。
- Task B prefix=512 当前只有 SVM/CNN/TCN 的 metrics，GRU/Attention 未完成。
- Task B 的 group_fraction / group_attention 入口在代码中存在，但本仓库当前没有可复核的
  最终结果文件，因此不虚构其测试指标。
- Task D prefix 运行的 censoring fraction 为 0.886154，说明它更像高 censoring
  的诊断实验，不能和 initial run 当作同一个预测问题。
- Task E 有若干只保存 config/feature metadata 的目录；没有 test metrics 的运行不纳入
  统一 benchmark。
- <code>runs/panda_temporal_adapter_full</code> 只有配置，没有最终 metrics；只有
  <code>runs/panda_temporal_adapter_full_20k</code> 的指标列入上面的历史 Panda adapter 记录。
- Task F 的 XGBoost 被请求但没有保存预测/指标。
- 训练日志中的中间 epoch 指标和重复重跑结果都保留在产物目录，但本节只把有明确最终
  test/validation 汇总的运行作为结果。

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
  --target-mode stable_fraction \
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

如需把新增的 GRU 一起加入 trajectory binary 对比，将上面的模型列表改为：

```bash
--models svm,cnn,gru,tcn,attention
```

GRU 默认使用两层单向结构；可通过 `--gru-layers` 和 `--hidden-dim` 调整层数与 hidden width。

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
- `--gru-layers`: 覆盖 GRU 层数，默认 `2`
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

Task C 的 initial-condition 子实验只使用每个 system 在 `t=0` 的一份
`orbital15` 特征，不读取后续轨迹；由于 32 条 variation 的 IC 相同，使用
一个简单的 `15 → 128 → 128 → 64 → 7` MLP 预测 7 个 future horizons：

```bash
conda run -n stellar python -m tasks.task_c \
  --dataset-dir dataset_v3 \
  --sampling-id 2 \
  --output-dir results/task_c_initial_condition_mlp \
  --input-mode initial \
  --group-size 32 \
  --future-times 500,800,1500,2500,4000,6500,10000 \
  --models baseline,ic_mlp \
  --test-ratio 0.2 \
  --val-ratio 0.1 \
  --seed 42 \
  --batch-size 128 \
  --epochs 40 \
  --lr 0.001 \
  --weight-decay 0.0001 \
  --hidden-dim 128 \
  --dropout 0.2 \
  --num-workers 0 \
  --device cuda
```

`initial` 模式固定读取 `orbital15` 的 `t=0` 特征，并将 32 条轨迹的
survival labels 汇总成 system-level survival curve；它与 prefix 模式共享
future horizons、system split、hazard head 和评估指标。

常用参数：

- `--max-shards`, `--rows-per-shard`, `--max-cases`: 限制数据量
- `--future-times`: 修改 survival curve 的评估 horizons
- `--models baseline,direct,direct_mean_std,traj,dual`: 选择要跑的模型
- `--encoder-type cnn|gru|attention`: 神经网络 encoder
- `--system-loss-weight`, `--consistency-weight`: 控制 `traj`/`dual` 双头 loss 中 system head 和一致性项的权重
- `--no-geom --no-time`: 移除 geometry / normalized time features

Prefix 长度与 variation 数量的二维 ablation 可使用：

```bash
conda run -n stellar python -m tasks.task_c.ablation \
  --output-dir results/task_c_ablation \
  --prefix-lens 10,20,50,100 \
  --variation-counts 2,4,8,16,32 \
  --models baseline,direct,traj,dual \
  --prefix-representation orbital15 \
  --variation-target full32 \
  --device cuda
```

该 runner 使用平衡、嵌套的 variation 子集；减少输入 variation 时仍预测完整
32 条 variation 的 system-level survival fraction，并在 `summary.csv` 中汇总每个 cell。
future horizons 沿用 Task C 原始的 `time_unbound` absolute 时间轴；不要用轨迹
采样时间去平移它，因为两者单位不同。
它会自动跳过已有 `metrics.json` 的 cell，适合中断后继续；需要重跑时加
`--force`。runner 默认先在 `results/task_c_ablation/_cache` 构建一次共享 prefix
cache；如果缓存放在更快的本地磁盘，可用 `--cache-dir PATH` 指定，并用
`--rebuild-cache` 强制重建。

### Task D

当前 Task D 的 initial-condition 主实验预测半长轴 Lyapunov proxy：

```bash
conda run -n stellar python -m tasks.task_d \
  --task-setting initial \
  --dataset-dir dataset_v3 \
  --sampling-id 2 \
  --group-size 32 \
  --threshold 10000 \
  --target-mode lyapunov_proxy \
  --label-observable smax \
  --aggregation min \
  --censoring-rule all_pairs \
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
  --target-mode lyapunov_proxy \
  --label-observable smax \
  --dataset-dir dataset_v3 \
  --sampling-id 2 \
  --group-size 32 \
  --threshold 1000 \
  --aggregation min \
  --censoring-rule all_pairs \
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

- `--force-recompute`: 忽略缓存标签并重新计算 proxy
- `--target-mode lyapunov_proxy`: 预测 `R_ij(t) >= threshold` 的首次采样时间；这是默认模式
- `--threshold`: 主实验默认 `10000`；建议用 sensitivity runner 同时报告 `1000/3000/10000/30000`
- `--aggregation min|median|mean`: 聚合 16 个 pair 的 proxy time；默认 `min`
- `--target-mode fitted_lyapunov_time`: 使用增长 slope 倒数的旧目标
- `--lyapunov-aggregation max|median|mean`: 控制 Lyapunov pair slope 的聚合方式
- `--require-pair-consistency`: 恢复 pair 间 MAD 一致性筛选
- `--lambda-r2-min`, `--min-valid-lambda-pairs`, `--max-pair-log-mad`: 调整 slope 与 pair consistency 质量筛选
- `--feature-aggregation first|mean_std`: initial setting 的 feature 聚合方式
- `--no-mlp`, `--no-cnn-orbital`, `--no-cnn-phase-space`, `--no-set-cnn`: 关闭对应神经模型
- `--loss mse|huber`: 神经网络回归损失
- `--allow-incomplete-valid-group`: 放宽完整 `32` 条有效 variation 的要求

### Task E

统一 forecasting 入口。当前主协议是 E1 `512 -> 128`；追加
`--run-rolling --rolling-blocks 4` 可在同一次运行中执行 E2：

```bash
conda run -n stellar python -m tasks.task_e \
  --data_glob "dataset_v3/results_batch_*_sampling_2.hdf5" \
  --representation orbital15 \
  --model timesfm \
  --context_len 512 \
  --pred_len 128 \
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
  --num_eval_windows 0 \
  --train-windows-per-trajectory 8 \
  --val-windows-per-trajectory 4 \
  --test-windows-per-trajectory 4 \
  --rolling-windows-per-trajectory 2 \
  --bootstrap-samples 1000 \
  --foundation_batch_size 32 \
  --timesfm_model_id google/timesfm-2.5-200m-pytorch \
  --chronos_model_id amazon/chronos-bolt-small \
  --chronos_variant bolt \
  --chronos_num_samples 20 \
  --panda_model_id GilpinLab/panda-72M \
  --freeze_panda \
  --hf-local-files-only \
  --torch_dtype auto \
  --output_dir results/task_e_local_timesfm \
  --prediction_samples 16 \
  --log_every_batches 100
```

旧的 `64 -> 32` Panda adapter 结果仍保留在历史报告中，但 `panda_adapter` 已从当前
CLI 模型列表停用；当前 Panda 基线使用 `panda_zero_shot` 的原生 `512 -> 128`。

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

- `--model ridge|tcn|patchtst|transformer|itransformer|timesfm|chronos|chronos2|panda_zero_shot`
- `--representation orbital15|raw_state18|trajectory`
- `chronos` 默认使用 `amazon/chronos-bolt-small`；使用
  `--model chronos2` 可切换到 Chronos-2 原生 multivariate pipeline。
- `--context-len 512 --pred-len 128 --run-rolling --rolling-blocks 4`: 运行当前 E1/E2 主协议。
- `--train-windows-per-trajectory 8`: 每个 epoch 每条 train trajectory 动态随机采样最多 8 个 candidate windows。
- `--val-windows-per-trajectory 4`, `--test-windows-per-trajectory 4`: 固定选择约 10/35/65/90% 的 validation/test windows。
- `--rolling-windows-per-trajectory 2`: E2 每条 trajectory 固定选择约 25/75% 的 rolling starts。
- `--bootstrap-samples 1000`: 以 system 为 bootstrap unit 计算 95% CI。
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
