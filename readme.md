# Stellar Dynamics Stability Prediction
## **English version below** 
本项目聚焦于动力系统长期稳定性预测，支持两类任务：

1. `Task A`: 同一初始条件下的扰动集合概率预测（输出稳定/发散概率分布）。
2. `Task B`: 基于初始条件 + 早期时间序列前缀的二分类预测（输出单一标签）。

## 1. 问题定义

### Task A: 扰动集合概率预测（Distribution Prediction）

- 输入：一个 `batch`，对应同一个 `initial condition` 下的多个扰动轨迹（例如 8 条）。
- 输出：该初始条件未来演化结果的概率分布：
  - `P(stable)`
  - `P(diverge)`（或 `P(unstable)`）
- Ground truth：由同一初始条件下 8 条扰动轨迹的最终结果统计得到（经验分布）。
  - 例：8 条里 6 条稳定，2 条发散，则标签分布为 `P(stable)=0.75, P(diverge)=0.25`。

这个任务本质上是从“单个系统轨迹分类”提升到“同一初始条件下不确定性建模”。

### Task B: 前缀时间序列二分类（Prefix-to-Label Classification）

- 输入：
  - 初始条件（initial condition）；
  - 该 case 的早期时间序列前缀（例如前 `T=700` 个时间步）。
- 输出：单一二分类标签（single-label classification）：
  - `stable`（在设定观察窗口内保持稳定）
  - `unstable`（后续会发散/失稳）
- 标签定义可按已有流程，例如使用 `time_unbound`：
  - 若 `time_unbound == 5000`，记为稳定类（`1`）；
  - 否则记为不稳定类（`0`）。

该任务与当前已有分类流程一致，只是明确强调“只喂前期片段，预测长期结局”。

## 2. 数据理解与组织建议

根据当前描述，数据可以理解为两层结构：

- 层 1（case 级）：约 `5000` 个系统样本（或等价规模）。
- 层 2（perturbation 级）：每个初始条件有 `8` 个扰动版本，用于反映该初始条件下结局的不确定性。

建议统一数据索引字段（示例）：

- `case_id`: 初始条件 ID（同一 `case_id` 下有 8 条扰动）。
- `perturb_id`: 扰动 ID（0~7）。
- `x0`: 初始条件向量。
- `trajectory`: 时间序列观测（形如 `[C, L]` 或 `[L, C]`）。
- `time_unbound` / `diagnosis`: 结局标签字段。

## 3. 标签构建

### Task A 标签（分布标签）

对每个 `case_id` 聚合 8 条扰动：

- `n_stable = sum(y_i == stable)`
- `n_total = 8`
- `p_stable = n_stable / n_total`
- `p_diverge = 1 - p_stable`

训练目标可选：

- KL / Cross-Entropy（模型输出分布 vs 经验分布）
- 或将 `p_stable` 当回归目标（MSE/BCE with soft target）

### Task B 标签（单标签）

每条样本独立构造：

- `y = 1`（stable） if `time_unbound == 5000`
- `y = 0`（unstable） otherwise

输入仅使用前缀 `trajectory[:, :T]`（例如 `T=700`）。

## 4. 评估指标

### Task A

- `Brier Score`（概率校准）
- `NLL / Cross-Entropy`
- `ECE`（Expected Calibration Error，可选）

### Task B

- `AUC`
- `F1`
- `Accuracy`

> 建议同时报告不同 `T`（如 `100/200/500/700`）下的结果，观察“可观测前缀长度”对预测能力的影响。

## 5. 数据切分原则（重要）

为避免信息泄漏，优先按 `case_id` 切分训练/验证/测试集：

- 同一 `case_id` 的 8 条扰动不能跨集合出现。
- Task A 天然按 `case_id` 建样本；
- Task B 也建议保持同样切分策略，避免模型记住某个初始条件族。

## 6. 基线实现建议

### Task A 基线

- 输入 8 条扰动轨迹后做集合建模（Deep Sets / Mean Pooling / Attention Pooling）。
- 输出 `p_stable`。

### Task B 基线

- 直接复用现有 1D-CNN 或 Transformer 编码器；
- 将输入限制为前缀长度 `T`；
- 输出二分类 logit。

## 7. 结果记录模板

建议在 `result.md` 中统一记录：

- Task A：`Brier / NLL / ECE`
- Task B：`AUC / F1 / Acc`（按不同 `T`）
- 数据版本、切分方式（是否按 `case_id`）、随机种子

