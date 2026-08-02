# `visualize_dataset_v3_trajectories.py` 使用说明

这个脚本用于读取 `dataset_v3` 的三体轨道结果，并生成：

- 三维轨道图（PNG）；
- `x/y/z` 坐标随时间变化图（PNG）；
- 可选的 GIF 或 MP4 轨道动画；
- 一个记录输入和输出文件的 `summary.json`。

脚本默认读取：

```text
dataset_v3/results_batch_00_sampling_2.hdf5
```

当前本地环境中，`dataset_v3` 已链接到 `/mnt/d/stellar_dynamics_data/sampling_2`；服务器环境可以继续链接到 `/data/sampling_2`。因此通常不需要在命令中写机器相关的绝对路径。

## 基本用法

在仓库根目录 `/home/wan/stellar_dynamics` 下运行：

```bash
conda run -n stellar python tools/visualize_dataset_v3_trajectories.py \
  --shard-index 0 \
  --row 0 \
  --time-stride 20
```

默认输出目录是：

```text
results/dataset_v3_trajectory_viz/
```

对于 `row=0`，通常会得到：

```text
sampling2_shard00_row0_3d.png
sampling2_shard00_row0_coords.png
summary.json
```

脚本使用无界面的 Matplotlib 后端，适合在服务器或 WSL 中运行，不会弹出图形窗口。

## 常用示例

### 随机抽取 5 条轨道

不指定 `--row` 时，脚本会按照 `--seed` 固定随机抽样：

```bash
conda run -n stellar python tools/visualize_dataset_v3_trajectories.py \
  --shard-index 3 \
  --num-examples 5 \
  --seed 42 \
  --time-stride 20 \
  --center com
```

### 截取一段时间序列

`--time-stop` 是 Python 切片的结束位置，不包含该位置：

```bash
conda run -n stellar python tools/visualize_dataset_v3_trajectories.py \
  --shard-index 0 \
  --row 12 \
  --time-start 0 \
  --time-stop 1000 \
  --time-stride 8
```

### 生成 GIF 动画

```bash
conda run -n stellar python tools/visualize_dataset_v3_trajectories.py \
  --shard-index 0 \
  --row 0 \
  --time-stride 8 \
  --animation \
  --animation-format gif \
  --fps 24 \
  --trail 120
```

其中 `--trail` 表示动画中每个物体最多保留多少个历史采样点形成轨迹尾迹。

### 生成 MP4 动画

```bash
conda run -n stellar python tools/visualize_dataset_v3_trajectories.py \
  --shard-index 0 \
  --row 0 \
  --time-stride 8 \
  --animation \
  --animation-format mp4 \
  --fps 24
```

生成 MP4 需要系统中可用的 FFmpeg；如果失败，可以改用 GIF。

## 参数说明

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--dataset-dir` | `dataset_v3` | 数据集目录 |
| `--sampling-id` | `2` | 文件名中的 sampling 编号 |
| `--shard-index` | `0` | 分片编号，例如 `0` 对应 `results_batch_00` |
| `--row` | 随机 | 分片内部的行号；不是整个数据集的全局 `run_id` |
| `--num-examples` | `1` | 抽取数量；指定 `--row` 时取从该行开始的连续行 |
| `--seed` | `42` | 随机抽样种子 |
| `--time-start` | `0` | 起始时间步 |
| `--time-stop` | 全部 | 结束时间步，不包含该位置 |
| `--time-stride` | `4` | 时间步采样间隔 |
| `--center` | `com` | `none`、`body0` 或 `com` |
| `--output-dir` | `results/dataset_v3_trajectory_viz` | 输出目录 |
| `--dpi` | `160` | PNG/动画分辨率参数 |
| `--animation` | 关闭 | 同时生成动画 |
| `--animation-format` | `gif` | `gif` 或 `mp4` |
| `--fps` | `24` | 动画帧率 |
| `--trail` | `120` | 动画轨迹尾迹长度 |

## 坐标中心化方式

- `none`：保留原始坐标；
- `body0`：将 body 0 放在原点；
- `com`：将三个物体位置的简单平均值放在原点。

这里的 `com` 是位置的算术平均，不是根据物体质量计算的质量加权质心。

## 输入数据要求

目标 HDF5 文件必须包含：

```text
positions_complete
```

脚本会优先读取：

```text
times_complete
```

如果不存在，则使用采样步编号作为横轴。轨道数组应表示为时间步、物体和三维坐标，例如 `(time, body, xyz)`。

## 注意事项

- `--shard-index` 只选择一个分片，脚本不会自动遍历 20 个分片；
- `--row` 是所选分片内部的行号；
- 缺失值、NaN 和无穷值会被替换为 `0`；
- 三维图会自动设置相同的 x/y/z 显示尺度；
- 该脚本主要使用 CPU 绘图，不会使用 NVIDIA GPU；
- 运行前可以用下面的命令检查输出：

```bash
find results/dataset_v3_trajectory_viz -maxdepth 1 -type f | sort
```

如果出现 `Missing result shard`，请先检查 `dataset_v3` 链接和目标分片文件是否存在：

```bash
readlink -f dataset_v3
ls -lh dataset_v3/results_batch_00_sampling_2.hdf5
```
