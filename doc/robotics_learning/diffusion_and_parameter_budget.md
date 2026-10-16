# Diffusion action model 与精确参数预算

## 1. 与 MLP 并列的最小动作模型

实现位于 `src/flow_rl/models/diffusion.py`，导出名称为 `flow_rl.models:DiffusionChunkPolicy`。原来的 `GaussianChunkPolicy` 继续可用。

| 接口 | 两种模型的约定 |
| --- | --- |
| observation | 原始状态向量 `[B, O]`，模型内部使用 checkpoint 中的统计量标准化 |
| `sample(observations)` | 随机动作 `[B, H, A]`，最终范围 `[-1, 1]` |
| `sample(..., deterministic=True)` / `forward` | 可复现动作 `[B, H, A]` |
| `predict(observation)` | NumPy 状态 `[O]` → NumPy action chunk `[H, A]` |
| `per_sample_loss(observations, actions)` | 每条样本的训练损失 `[B]`，由算法决定如何加权 |

这两个模型可以使用相同的环境、离线 batch、replay、Q 模型、chunk controller 和 checkpoint 入口。

### 动作头结构

状态线性投影形成一个 token，H 个带噪动作分别形成 H 个 token。固定正弦位置编码区分 chunk 中的位置。小型 Transformer 在这些 token 间做双向 attention，时间正弦特征通过 MLP 和 adaptive RMSNorm 条件化每个 block。最终将 H 个动作 token 映射回 A 维速度场。

默认 `hidden_dim=64, num_layers=2, num_heads=4, ff_multiplier=2, time_embed_dim=32, num_inference_steps=10`。没有 dropout。该实现没有视觉/语言编码器，也不能加载 PI0.5 权重。

参考 [OpenPI README](https://github.com/Physical-Intelligence/openpi/blob/main/README.md) 与 [动作头实现](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/pi0.py)：保留 flow-matching 动作生成、时间条件和多步积分，将状态作为轻量条件输入。

### 训练与采样

采用 OpenPI 代码的时间方向：`t=1` 为噪声、`t=0` 为动作。对于数据动作 `a` 和标准高斯噪声 `ε`：

\[
x_t=t\epsilon+(1-t)a,\qquad u_t=\epsilon-a,
\]

\[
\ell_i=\operatorname{mean}_{H,A}\left(v_\theta(s_i,x_t,t)-u_t\right)^2.
\]

训练时间按 `Beta(1.5, 1) * 0.999 + 0.001` 采样。`per_sample_loss` 可显式传入 `noise` 和 `time`，用于检查公式和固定样本测试。

推理从高斯 chunk 开始，以显式 Euler 法迭代 N 次，`dt=-1/N`、`x ← x + dt*vθ(s,x,t)`，最后裁剪到动作范围。确定性推理使用随 checkpoint 保存的固定高斯初始 chunk；它不改变全局随机状态，也不等于分布均值。随机推理为每条 observation 采样新噪声。

### 与 BC / AWAC 的关系

`algorithms/base.py` 的公共损失函数读取策略提供的每样本损失：

| 策略 | `training_objective` | BC actor 目标 | AWAC actor 目标 |
| --- | --- | --- | --- |
| Gaussian MLP | `negative_log_likelihood` | `mean(-log π(a|s))` | `mean(w * -log π(a|s))` |
| Diffusion | `flow_matching` | `mean(ℓ)` | `mean(w * ℓ)` |

AWAC 的 Q target、采样策略 baseline 和 `w=clip(exp(A/temperature))` 沿用现有实现。权重停止梯度。Diffusion 的这一 actor 更新是优势加权 flow-matching 替代目标，不能解释为精确动作似然或声称复现标准 AWAC。该模型故意不提供伪 `log_prob`。

Gaussian 的损失和梯度保留原有数学行为。旧插件只有 `log_prob` 时，仍可走兼容路径；新策略应提供 `per_sample_loss` 和目标名称。日志分别输出 `log_prob_mean`、`flow_matching_loss_mean`。Checkpoint 记录 `actor_objective`，不允许加载到目标类型不一致的策略。

## 2. 精确参数量配置

```yaml
model:
  parameter_budget: 100000
value:
  parameter_budget: 200000
```

- `model.parameter_budget`：整个 actor 的参数总数。
- `value.parameter_budget`：整个 critic ensemble 的参数总数，包括所有 Q 网络。
- 默认为 `null`，按原 `kwargs` 构造，不改变已有网络和 checkpoint 的布局。
- 必须为正整数。每个权重/bias 元素计为一个参数；固定噪声、时间/位置编码、标准化统计和稀疏索引等 buffer 不计。
- 自动预算当前要求模型全部可训练；包含冻结参数的自定义模型会明确报错。

### 构建过程

1. 保留深度和其他结构参数，通过 meta tensor 搜索统一隐藏宽度，无需为候选模型分配完整权重，也不消耗实验 RNG。
2. Gaussian / critic 保留 `hidden_dims` 的层数，将每层宽度设为搜索结果；设置预算时，手写宽度值仅决定层数。Diffusion 的宽度必须满足 attention head 和偶数约束。
3. 找到第一个参数量不小于预算的密集结构。对于超额部分，按容量比例从允许的线性层移除连接，精确达到预算。
4. `BudgetedLinear` 只存储被保留连接的可训练向量，前向 scatter 成权重矩阵后执行线性映射。每个输入和输出至少保留一个连接；不存在不参与前向的凑数参数。
5. Diffusion 只允许裁剪 FFN 连接，attention、时间条件、输入和输出投影保持密集。模型可以通过 `budget_layer_names()` 声明允许修改的线性层。
6. 实际参数总数与预算不相等即报错。预算过小或结构约束无法满足时也报错，不自动退回近似参数量。

这是精确的**连接数匹配**，模型在预算模式下可能包含固定稀疏层。原生密集结构之间仅靠整数宽度通常无法精确匹配任意总数。稀疏矩阵当前通过密集矩阵计算，因此不能把参数减少量当作 FLOPs 减少量；稀疏结构、原始参数量和最终参数量均可检查。

### 相同预算的对照

`cube_mlp_matched.yaml` 与 `cube_diffusion_matched.yaml` 均使用 O=28、A=5、H=4、同一 critic 配置。预算约定为：

| 统计项 | MLP | Diffusion |
| --- | ---: | ---: |
| actor | 100,000 | 100,000 |
| critic ensemble | 200,000 | 200,000 |
| 可训练总量 | 300,000 | 300,000 |
| 包括冻结 target critic 的模型总量 | 500,000 | 500,000 |

BC 不构造 critic，因此 `cube_diffusion_bc.yaml` 的可训练总量是 actor 的 100,000。

```bash
uv run flow-rl inspect-model --config configs/cube_mlp_matched.yaml \
  --observation-dim 28 --action-dim 5
uv run flow-rl inspect-model --config configs/cube_diffusion_matched.yaml \
  --observation-dim 28 --action-dim 5

# 修改预算，保留深度、head 数和求解步数
uv run flow-rl inspect-model --config configs/cube_diffusion_matched.yaml \
  --observation-dim 28 --action-dim 5 --set model.parameter_budget=50000
```

`inspect-model` 不创建环境、不读取数据。它报告实际计数、预算、解析后的宽度、各稀疏层的连接数，以及 actor 目标名称。输入输出维度变化后预算会重新求解；必须在相同 O/A/H 下比较模型。

`metadata.json` 和 `summary.json` 包含同样的结构报告。Checkpoint 保存解析后的 actor/critic 结构，以及每层连接索引和权重。加载推理、断点续训时复用保存的结构，不依赖重新搜索宽度；此前未带这些字段的 checkpoint 继续受支持。

## 3. 运行

```bash
# 快速真实仿真链路验证，使用随机数据
uv run flow-rl train --config configs/cube_diffusion_smoke.yaml

# 官方离线数据：同参数量的两个模型
uv run flow-rl train --config configs/cube_mlp_matched.yaml
uv run flow-rl train --config configs/cube_diffusion_matched.yaml

# Flow-matching 监督训练
uv run flow-rl train --config configs/cube_diffusion_bc.yaml

uv run flow-rl evaluate --checkpoint runs/cube_diffusion_matched/checkpoint.pt
```

每组实验使用独立输出目录。参数量相同并不代表相同推理预算：MLP 单次前向生成 chunk，Diffusion 默认调用速度场 10 次；可用 `model.kwargs.num_inference_steps` 调整这一成本。任务成功率、时延和不同 actor 目标都应在对比中单独报告。

## 4. 实测验证（2026-10-15）

- 完整测试 **124 passed**，包含真实 MuJoCo 集成；lint、format 与 diff 检查通过。
- 两种匹配预算模型都使用官方 1,000,000 条 transition 完成 30 步训练、checkpoint 保存和真实仿真推理。实际 actor / critic / 可训练总量分别为 100,000 / 200,000 / 300,000。
- Diffusion 的 BC、Offline RL、Online RL 路径均通过完整训练测试；带预算的 diffusion 在 CPU 上断点续训与连续训练结果一致。
- 独立进程加载 diffusion checkpoint 成功；旧 Gaussian checkpoint 加载兼容。
- 这次只做短程功能验证，两种模型的 30 步 rollout 都未完成任务，不能据此判断学习性能。

完整结构与指标见 [实测记录](diffusion_validation_20261015.json)。本机已有 checkpoint：`runs/verify_diffusion_matched_20261015/checkpoint.pt` 和 `runs/verify_mlp_matched_20261015/checkpoint.pt`。
