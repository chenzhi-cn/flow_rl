# Manifold adherence：固定样本的独立评估工具

入口：`python -m flow_rl.manifold`。工具分为样本生成、模型推理和几何统计三个部分，不依赖训练循环。内置适配 OGBench single-cube 与本项目 MLP/diffusion checkpoint，也可对外部模型导出的 action chunks 评分。

## 1. 指标与论文、代码的对应关系

[Much Ado About Noising，ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/file/9299ca9ed58731945e934adb5b71728c-Paper-Conference.pdf) 的 **E.1** 定义 action 扰动后仿真生成可行 observation 的方法；**G.2** 定义邻近训练 states 对应 actions 的线性张成空间。你提供的 [toyexp 代码](https://github.com/simchowitzlabpublic/much-ado-about-noising/tree/9f28318b6f913dda9a63aa65cdab2f1bf4c2a0e4/toyexp/toyexp) 对应附录 L 的解析几何实验。

给定扰动后的 observation，寻找训练集的 k 个最近邻，将对应完整动作块展平后作为矩阵 A 的列。对 A 做 SVD，取数值非零奇异值对应的正交基 U，令 P=UUᵀ。模型输出也展平为 a：

| 指标 | 定义 | 含义 |
| --- | --- | --- |
| off-manifold norm | ‖a−Pa‖₂ | 输出在邻域 action span 外的分量 |
| on-manifold norm | ‖Pa‖₂ | 输出在该空间内的分量大小 |
| total norm | ‖a‖₂ | 完整输出的大小 |

逐样本满足 `total² = off² + on²`。on 和 total 是对论文 off 指标的互补分解，不是相对某个 expert action 的误差；报告中明确记录公式。

[train_proj.py](https://github.com/simchowitzlabpublic/much-ado-about-noising/blob/9f28318b6f913dda9a63aa65cdab2f1bf4c2a0e4/toyexp/toyexp/toyexp/scripts/train_proj.py#L145) 区分输出的垂直分量、归一化比例，以及 reconstruction residual 的垂直分量。本工具采用 G.2 的**输出本身的 norm**。不对邻居 actions 做局部均值中心化，不将它替换成 PCA 切空间，也不除以 total norm。

## 2. 固定样本的生成规则

1. **划分数据**：validation 用于筛选 critical states；train 用于计算归一化统计、查找近邻和构建 action span。OGBench 使用官方 noncompact transition 索引，qpos/qvel 与 observations/actions 对齐。
2. **筛选**：对每个 validation state i，取同一轨迹中的 `[i-window_before, i+window_after]`，包含两个端点和当前 action；轨迹两端裁剪窗口。按各 action 维度的总体方差之和排序，取前 n 个。这个明确的方差规则来自本次需求；E.1 本身仅描述选择高歧义 states。并列分数按原始索引升序，不额外去重或施加间距。
3. **扰动**：固定 seed，为每个 state 生成 m 个 `η~N(0, noise_scale² I)`。使用 train action 的逐维均值、标准差做 z-score，反归一化后等价于 `a_perturbed = a_original + action_std * η`。标准差下限 `std_floor` 防止常量维除零。将扰动动作显式裁剪到物理 action bounds，记录原始请求、实际执行值及裁剪数量。
4. **仿真**：每一个扰动都独立恢复同一个原始 critical state，执行一次环境 `step`。得到的 observation 是新的评估输入。另存一次未扰动 rollout 的 nominal next observation。没有直接往 observation 上加噪声。
5. **几何**：针对这 n×m 个新 observations，从 train 中检索 k 个近邻；每个 reference state 必须有完整、同一轨迹内的 H 步 action chunk。SVD 保留 `s > svd_rcond * s_max` 的方向，仅去掉数值零方向。
6. **保存**：将样本、噪声、动作、critical indices/scores/windows、近邻原始索引/距离、投影基/秩/奇异值、训练统计、推理 seeds、配置和数据指纹存入一个 NPZ。文件含内容 SHA-256；生成使用原子排他写入，已有文件不会被覆盖。

默认 `observation_metric: raw` 使用原始 state 欧氏距离，`action_metric: raw` 使用物理 action 单位。可显式配置 `standardized` 使用 train 的统计；该选择改变几何坐标，action 标准化还会改变原点，因此必须为所有模型固定。模型内部的 observation normalization 仍来自各自 checkpoint。

### 动作块和秩

默认 H=4、single-cube action_dim=5，因此向量维数为 20，k=8。比较要求所有模型具有相同 observation/action 维数及 H，维度不一致直接报错，不截断模型输出。

若 k 足够大，邻居 actions 可能张成整个输出空间，此时所有模型的 off norm 都接近零。报告包含最小/最大/平均 rank 和 full-rank fraction，方便识别这种情况。不要为了得到差异而事后修改同一实验的投影空间。

## 3. 使用

先安装项目依赖，再从仓库根目录运行。配置见 [`configs/manifold/cube.yaml`](../../configs/manifold/cube.yaml)。默认 n=100、m=64，样本文件为 `runs/manifold_cube/samples.npz`。

```bash
# 仅首次需要数据集和 MuJoCo；相同配置再次运行会校验并复用样本
uv run python -m flow_rl.manifold prepare --config configs/manifold/cube.yaml

# 使用配置中的 MLP/diffusion checkpoint 和输出目录
uv run python -m flow_rl.manifold compare --config configs/manifold/cube.yaml

# 查看样本指纹、参数、rank 和恢复质量
uv run python -m flow_rl.manifold inspect --samples runs/manifold_cube/samples.npz
```

配置默认使用本机已有的 30 次更新 MLP/diffusion 验证 checkpoint，只用于检查工具流程。正式实验请替换成充分训练的模型，例如 `cube_mlp_matched.yaml` 和 `cube_diffusion_matched.yaml` 的训练输出。也可在命令行指定 checkpoint：

```bash
uv run python -m flow_rl.manifold compare \
  --samples runs/manifold_cube/samples.npz \
  --checkpoint mlp=runs/verify_mlp_matched_20261015/checkpoint.pt \
  --checkpoint diffusion=runs/verify_diffusion_matched_20261015/checkpoint.pt \
  --output-dir runs/manifold_cube/comparison_existing
```

小规模链路验证：

```bash
uv run python -m flow_rl.manifold prepare --config configs/manifold/cube.yaml \
  --set samples=runs/manifold_cube_smoke/samples.npz \
  --set preparation.n_critical=8 \
  --set preparation.perturbations_per_state=8
```

`prepare` 发现已有样本且生成配置不一致时会报错。新的 n、m、seed、窗口、k、H、尺度或 simulator 设置应使用新样本路径。复用只校验已保存文件及配置，不重新加载当前数据集；原始数据内容指纹已写入文件。

`compare` 完全不启动 simulator，也不读取数据集。默认调用模型的确定性接口；加 `--stochastic --seed 0` 时使用文件内固定的逐样本 seed，单样本推理保证结果不受 batch_size 或模型评估顺序影响。不同架构的采样机制仍可能不同；报告记录采样方式、设备、版本、checkpoint SHA-256。跨硬件浮点运算不保证位级一致。

每次比较使用新的输出目录，包含：

- `report.json`：每个 norm 的 mean、std、RMS、median、p95、min、max；共享 sample_id、配置和 rank 诊断。
- `per_sample.csv`：每个 model × critical state × perturbation 的三个 norm。
- `per_state.csv`：每个 critical state 在 m 次扰动上的均值。
- `predictions.npz`：物理单位及 metric 坐标下的完整输出，包含 sample_id。

mean/std 对 n×m 个逐样本 L2 norm 聚合。RMS 单独报告，不将 L2 norm 的均值混作 MSE。

## 4. 外部模型与新 simulator

### 外部推理结果

读取样本文件的 `observations` 数组，严格按存储顺序推理，导出 `[n*m,H,A]` 的 `.npy`：

```bash
uv run python -m flow_rl.manifold score \
  --samples runs/manifold_cube/samples.npz \
  --predictions my_model_actions.npy --label my_model \
  --action-space physical --output-dir runs/manifold_cube/my_model
```

`normalized` 表示本框架策略输出的 `[-1,1]` actuator 坐标，工具会按已存 bounds 转为物理单位；它不是数据集 z-score。外部模型输出已经是环境 action 单位时用 `physical`。

也可直接调用：

```python
from flow_rl.manifold import load_bundle, evaluate_predictions

bundle = load_bundle("runs/manifold_cube/samples.npz")
summary, details = evaluate_predictions(actions, bundle, action_space="physical")
```

### Simulator 扩展

在 YAML 的 `source.target` 配置自己的 `module:factory`。工厂返回符合 `RolloutAdapter` 协议的对象：

- `load_datasets()` 返回 `(train, validation)`，至少包含 observations、actions、terminals。
- `action_low/action_high` 是物理控制界限。
- `rollout(validation_index, physical_action)` 每次独立恢复该行保存的完整状态，执行一步，返回 observation。
- `restore_observation`、`metadata`、`restoration_statistics` 提供恢复质量与来源；`close()` 释放资源。

内置 adapter 目前仅验证 state-based OGBench single-cube。其他 simulator 需要保存并恢复其控制器、执行器、物理引擎状态；不能仅凭一般 observation 向量假定可以恢复状态。

## 5. 解释边界

- 官方 OGBench NPZ 提供 qpos/qvel，但不提供历史 solver warm start、接触力、上一物理子步缓存的末端位姿等所有内部状态。内置 adapter 每个分支清空 MuJoCo 动态数据后恢复 qpos/qvel，记录恢复 observation、nominal next observation 与原始数据的差异。重算运动学也可能改变控制器相对历史轨迹的输入。这是确定的反事实 rollout，不能声称逐位复原了历史轨迹。
- OGBench play 数据描述行为数据的几何结构，并非每一步都是专家最优动作。这个工具测量对参考数据 action span 的遵循程度。
- metric 坐标下的零向量，其 off norm 也是零（默认 raw 配置下即零物理 action），因此 off 更低不能单独证明任务表现更好；需结合 total norm、闭环成功率及训练充分程度判断。
- 模型的 chunk 拼接、平滑和推理时延属于控制器行为；这里直接评价模型输出，避免这些后处理改变架构比较。

## 6. 文件职责

| 文件 | 职责 |
| --- | --- |
| `config.py` | 独立 YAML 配置、校验和覆盖 |
| `geometry.py` | 窗口方差、精确分块 KNN、SVD 与三个 norm |
| `rollout.py` | simulator 协议与 OGBench 恢复/推进 |
| `prepare.py` | model-independent 样本与几何生成 |
| `artifact.py` | NPZ 原子保存、内容指纹与校验 |
| `evaluate.py` | 模型推理、外部预测评分、结果统计 |
| `cli.py` | prepare / compare / inspect / score |

## 7. 已完成的运行验证（2026-10-16）

使用完整官方数据：train 1,000,000 transitions，validation 100,000 transitions，997,000 个合法 H=4 reference chunks。默认 n=100、m=64 的 **6,400 个样本已生成并保存**于 `runs/manifold_cube/samples.npz`；再次 prepare 已验证直接复用，sample_id 不变。

两份现有 checkpoint 都只有 30 次训练更新；下面的确定性推理结果验证工具流程，不构成架构优劣结论：

| 模型 | off mean | on mean | total mean |
| --- | --- | --- | --- |
| MLP | 0.456581 | 0.413182 | 0.621846 |
| Diffusion | 2.068958 | 2.164723 | 3.018740 |

全部样本的 action span rank=8，向量维度=20；未出现满秩退化。逐样本平方分解最大数值误差低于 `2e-14`。1384/6400 个扰动动作触及 bounds 并被裁剪；恢复 observation 的最大绝对误差为 0.007621，完整恢复诊断保存在文件中。

测试共 **250 passed**（含 2 项真实 MuJoCo 集成测试，其中 68 项为本工具新增测试），Ruff lint/format 与 diff whitespace 检查通过。重复 simulator 分支的一致性、样本校验和、模型维度检查、随机推理 seed 和批次不变性均已验证。

完整逐样本结果位于 `runs/manifold_cube/comparison/`。可版本管理的运行记录见 [manifold_validation_20261016.json](manifold_validation_20261016.json)。后续比较使用新的 `--output-dir`，保留已有报告。
