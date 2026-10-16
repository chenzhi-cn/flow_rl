# Flow RL

轻量、模块化的 robotics 训练框架。第一条完整路径是 **OGBench single-cube → 官方离线数据 → AWAC → checkpoint → action chunk 推理**。

当前内置 PyTorch 状态 MLP、π0.5 风格的最小 flow-matching action expert、AWAC、BC、在线 replay、OGBench/Gymnasium 环境，以及可替换的 chunk 拼接和平滑。模型、value、算法和环境通过 `module:factory` 配置扩展。

## 快速开始

Python 3.10.6+，使用仓库中的 `uv.lock` 固定依赖版本：

```bash
uv sync --locked

# 正式最小例子：首次自动下载官方训练/验证数据到 data/ogbench
uv run flow-rl train --config configs/cube_awac.yaml

# 加载 checkpoint，在同一固定任务上推理，无需再次加载数据集
uv run flow-rl evaluate --checkpoint runs/cube_awac/checkpoint.pt
```

默认任务为 `cube-single-play-singletask-task2-v0`：28 维 observation，5 维连续 action，20 Hz，最多 200 步。使用官方单任务 reward/mask 重标注；默认配置为 10,000 次 AWAC 更新、batch size 256、CPU 单线程。

Actor 和双 Q critic 使用 **256×256 GELU MLP**，结构依据 [OGBench 官方 SAC 小型状态网络](https://github.com/seohongpark/ogbench/blob/master/impls/agents/sac.py)。这里复用其网络结构，用 PyTorch 实现并训练 AWAC；没有使用预训练权重，也不宣称复现官方算法成绩。OGBench 本身没有标准 AWAC 基准实现。

只验证安装和训练链路，可运行：

```bash
uv run flow-rl train --config configs/cube_smoke.yaml
uv run flow-rl evaluate --checkpoint runs/cube_smoke/checkpoint.pt
```

该配置在**真实 single-cube 仿真**中采集 240 条随机 transition，使用 64×64 MLP 训练 20 步、输出 4 步 chunk。它只用于功能验证；正式离线实验使用官方 play 数据。已存在 checkpoint 的输出目录会阻止新训练覆盖，请指定新的 `experiment.output_dir` 或使用 `--resume`。

## 实验配置

| 配置 | 模式 | 说明 |
| --- | --- | --- |
| `configs/cube_awac.yaml` | Offline RL | 官方 play 数据，AWAC，H=1，标准单步 baseline |
| `configs/cube_awac_chunk.yaml` | Offline RL | H=4，Q 对整段 action 建模，折扣多步回报，重叠融合和 EMA |
| `configs/cube_bc.yaml` | Supervised | 相同 chunk actor，BC 最大似然训练，无 critic |
| `configs/cube_online.yaml` | Online RL | 环境采样、环形 replay，默认混入 50% 官方离线数据 |
| `configs/cube_smoke.yaml` | Offline RL | 随机仿真数据，快速检查执行链路 |
| `configs/cube_mlp_matched.yaml` | Offline RL | MLP，actor 精确 100,000 参数、critic 精确 200,000 参数 |
| `configs/cube_diffusion_matched.yaml` | Offline RL | 相同参数预算的 Transformer flow policy，优势加权 flow matching |
| `configs/cube_diffusion_bc.yaml` | Supervised | Flow-matching 监督训练，无 critic |
| `configs/cube_diffusion_smoke.yaml` | Offline RL | 真实仿真随机数据，快速验证 diffusion 训练与推理 |

配置支持相对路径 `extends` 继承与 `--set section.key=value` 覆盖。实验、数据和输出路径相对当前工作目录；从项目根目录运行上述命令。

```bash
# 查看最终解析的配置，不启动仿真或下载
uv run flow-rl check-config --config configs/cube_awac_chunk.yaml

# 先做短程官方数据验证
uv run flow-rl train --config configs/cube_awac.yaml \
  --set training.steps=100 --set experiment.output_dir=runs/quick_awac

# 多步 chunk / 监督训练
uv run flow-rl train --config configs/cube_awac_chunk.yaml
uv run flow-rl train --config configs/cube_bc.yaml

# 离线训练后在线微调，保留网络和优化器状态，重新收集 replay
uv run flow-rl train --config configs/cube_online.yaml \
  --init-from runs/cube_awac/checkpoint.pt

# 从零开始在线训练，不加载离线数据
uv run flow-rl train --config configs/cube_online.yaml \
  --set data.source=none --set experiment.output_dir=runs/cube_online_scratch

# 离线断点续训，steps 是新的总步数
uv run flow-rl train --config configs/cube_awac.yaml \
  --resume runs/cube_awac/checkpoint.pt --set training.steps=20000
```

`--resume` 恢复 actor、critic、target critic、optimizer、Torch 和数据采样 RNG。它要求模型/算法/环境/数据及 seed 一致。在线任务使用 `--init-from`，从新的环境 episode 和空 replay 开始，不把真机或仿真现场当作可精确恢复的状态。

每次实验输出：

```text
runs/<experiment>/
  config.yaml       # 完整配置
  metadata.json     # 数据来源、维度、窗口数、参数量
  metrics.jsonl     # 训练 loss、Q、advantage、权重等
  evaluation.jsonl  # return、success rate、控制调用耗时
  checkpoint.pt     # 可恢复的训练状态和推理参数
  summary.json      # 最终结果
```

检查点保存了 observation 标准化统计。所有内置策略在 `[-1, 1]` 动作空间训练，执行前按环境 action bounds 转换。官方训练数据约 245 MiB，加上验证集；加载及 observation 标准化时还需要额外内存。`data.max_transitions` 可以限制参与训练的数据量，但官方 loader 仍先读取完整数据集。

## Chunk 推理与时延

### MLP / Diffusion 切换与参数量对齐

两种策略均输入 `[B, observation_dim]`，输出 `[B, H, action_dim]`；单条 `predict` 输出 `[H, action_dim]`。single-cube 对照配置均为 `28 → [4, 5]`。

`DiffusionChunkPolicy` 使用小型 Transformer、时间条件 RMSNorm 和 flow matching，通过多次 Euler 更新将高斯噪声变为动作。设计参考 [OpenPI 的 π0.5 flow head](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/pi0.py)；这里是从零训练的状态版本，未包含视觉语言骨干或官方权重。

```yaml
model:
  target: flow_rl.models:DiffusionChunkPolicy  # 或 GaussianChunkPolicy
  chunk_size: 4
  parameter_budget: 100000
value:
  parameter_budget: 200000
```

参数预算默认 `null`，原有配置的网络结构保持原样。启用后自动选择宽度，并用参与前向计算的固定稀疏连接精确达到预算；不存在用于凑数的闲置 Parameter。统计包括权重与 bias，排除归一化统计量等 buffer。无法满足的预算会报错。对照配置的可训练参数均为 **300,000**；加上冻结的 target critic 均为 **500,000**。

```bash
# 无需仿真、数据下载即可检查结构和精确参数量
uv run flow-rl inspect-model --config configs/cube_mlp_matched.yaml \
  --observation-dim 28 --action-dim 5
uv run flow-rl inspect-model --config configs/cube_diffusion_matched.yaml \
  --observation-dim 28 --action-dim 5

# 训练与推理复用相同入口
uv run flow-rl train --config configs/cube_diffusion_smoke.yaml
uv run flow-rl train --config configs/cube_diffusion_matched.yaml
uv run flow-rl evaluate --checkpoint runs/cube_diffusion_matched/checkpoint.pt
```

Gaussian 使用真实 NLL；diffusion 使用 `per_sample_loss` 提供的 flow MSE。选择 AWAC 时，diffusion 的 actor 采用 **advantage-weighted flow-matching surrogate**，不是标准 AWAC 的精确 log-likelihood 目标。日志和 checkpoint 会记录这个区别。等参数量不代表等 FLOPs 或推理时延；diffusion 默认做 10 次模型调用。

完整公式、预算规则与扩展接口见 [Diffusion 与参数预算说明](doc/robotics_learning/diffusion_and_parameter_budget.md)。

### 执行控制

策略统一输出 `[H, action_dim]`。`ChunkController` 与模型解耦：

- `replace` 使用最新有效预测；`ensemble` 融合同一执行时间上的重叠预测，按预测年龄指数加权。
- `none` 或 `ema` 控制平滑；自定义 `ChunkStitcher` / `ActionSmoother` 可直接替换。
- 根据 observation 采样时间与预测收到时间，丢弃 `ceil(latency / control_dt)` 个已过期动作；耗尽后使用 `hold` 或 `zeros`。
- 同步虚拟时间用于快速仿真：仿真在计算期间暂停，H=1 的动作不会因为 CPU 耗时被误丢弃。
- 异步模式单独运行预测线程，控制循环继续消费已有 chunk，reset 后不会使用上一 episode 的预测。

```bash
# 使用训练过的 H=4 checkpoint，按实际时间执行，控制循环 20 Hz
uv run flow-rl evaluate --checkpoint runs/cube_awac_chunk/checkpoint.pt \
  --set inference.asynchronous=true --set inference.realtime=true
```

实时模式需要 `H >= 2`，并确保 chunk 总时长足以覆盖推理延迟和刷新间隔。H=1 与延迟向上取整不兼容，配置会直接报错。`control_call_ms_*` 是调用控制器获得动作的耗时；异步模式下它不等于后台模型完整推理耗时。当前执行器适合仿真和接口验证，没有硬实时调度保证。

## 可塑性检测

用一行 hook 选择被测层，独立统计死亡比例、休眠比例（默认激活频率低于 1%）、全模型平均权重绝对值、常规 stable rank 和谱熵 effective rank：

```python
from flow_rl.diagnostics import PlasticityMonitor

probe = PlasticityMonitor(model, layer="net.3")
# 正常执行模型；train 模式仅保留当前 batch，eval 模式累计所有 batch
model(observations)
print(probe.summary())  # 全部样本结果，以及按样本数加权的 batch 均值
probe.close()
```

也支持在 forward 中手动插入 `h = probe(h)`。每轮评估前调用 `probe.reset()`；Transformer/CNN 可指定特征维，权重范围与激活判据可配置。

```bash
uv run python examples/plasticity_diagnostics.py --output runs/plasticity_demo.json
```

该示例运行 MLP/diffusion 的训练与全数据集检测，无需仿真。完整公式、参数筛选、聚合语义与接入方式见 [可塑性检测说明](doc/robotics_learning/plasticity_diagnostics.md)。

## Manifold adherence 评估

独立工具按 validation 轨迹窗口内的 action 方差选择 critical states，扰动其 action 后通过仿真生成新 states，固定保存样本、训练集近邻及投影基。之后可复用同一文件比较 MLP、diffusion 或外部模型的 off-manifold、on-manifold、total norm：

```bash
uv run python -m flow_rl.manifold prepare --config configs/manifold/cube.yaml
uv run python -m flow_rl.manifold compare --config configs/manifold/cube.yaml
```

配置中的 checkpoint 路径需要对应已完成的训练。生成配置改变时使用新样本文件；比较阶段无需 simulator。指标公式、论文/代码对应关系、外部模型接入和恢复误差说明见 [Manifold adherence 工具](doc/robotics_learning/manifold_adherence.md)。

## 模块与扩展

```text
src/flow_rl/
  config.py             # YAML、继承、校验、覆盖
  factory.py            # module:factory 插件加载
  models/               # MLP / diffusion actor、精确参数预算、Q ensemble、模型协议
  algorithms/           # 独立 AWAC / BC 更新逻辑
  envs/                 # 统一协议、OGBench、Gym、RobotTransport
  data/                 # 离线 chunk 采样、NPZ、在线 replay
  inference/            # 时间对齐、拼接、平滑、同步/异步执行
  training/             # 构建、训练循环、checkpoint、指标
  diagnostics/          # 独立可塑性检测、流式全数据集统计
  manifold/             # 固定扰动样本、近邻 action span、模型几何评估
  evaluation.py         # 统一 rollout
  cli.py                # train / evaluate / collect / check-config
```

扩展实例及协议见 [架构与开发指南](doc/robotics_learning/architecture.md)。

- **VLA / π0.5**：内置最小状态 flow-matching 动作头；完整 VLA 仍需模型 adapter、权重和图像/语言预处理。`ChunkPolicy.predict(observation)` 与推理 runner 可接收图像、状态、指令组成的字典。
- **世界模型**：已有 `WorldModel.rollout` 协议；模型实现和基于模型的 RL 算法尚未接入。
- **其他仿真器**：可用 `make_gym_env`，或提供统一 reset/step 工厂。
- **Franka / 方舟无限等真机**：已有 `RobotTransport` / `RobotEnv` 边界，需要针对实际 SDK 实现 transport、任务 reward、控制频率与硬件保护。当前没有厂商驱动或真机验证。
- **训练扩展**：当前内置 trainer 处理向量 observation 与连续 action；多模态 VLA 训练需扩展数据 batch 和训练算法。AWAC 使用 Q 模型，通过采样当前策略估计 V，无独立 V 网络；其他算法可自行声明对应 value 模型。

## 测试

```bash
uv run pytest -q -m 'not integration'
uv run pytest -q -m integration
uv run ruff check src tests
uv run ruff format --check src tests
```

测试覆盖 TD target、advantage 权重、优化器恢复、trajectory 边界、timeout bootstrap、replay 回绕、多步训练、推理延迟/异常/reset，以及各训练模式的完整流程。`integration` 会启动真实 MuJoCo。macOS 普通终端可直接运行；Codex 受限沙箱中可能需要允许仿真进程访问 macOS 服务。Linux 无显示渲染可按 [OGBench 官方说明](https://github.com/seohongpark/ogbench#reference-implementations) 配置 EGL。

任务成功率属于学习效果，必须通过足够训练与评估测量。短程链路验证、随机数据测试不代表机器人已经学会搬运方块。实际执行结果记录在 `runs/*/summary.json`。

## 算法依据

- [OGBench 接口和数据语义](https://github.com/seohongpark/ogbench#usage-for-standard-non-goal-conditioned-offline-rl)：`terminals` 标记轨迹边界；`masks` 控制 Bellman bootstrap，二者独立。
- [AWAC 论文与官方项目](https://awacrl.github.io/)：critic 做 TD 学习，actor 做 advantage 加权最大似然；本实现没有 SAC entropy 项或 IQL expectile V。

## 本次运行验证（2026-09-30）

已在当前 macOS ARM64 / CPU 环境验证：完整测试 **50 passed**，Ruff lint/format 通过，`uv sync --locked --offline` 通过。

| 路径 | 数据 / 更新 | 结果 |
| --- | --- | --- |
| `runs/cube_awac` | 官方 1,000,000 transitions；H=1；10,000 次 AWAC 更新 | 训练、保存、5 回合评估完成 |
| `runs/verify_official_chunk` | 官方数据；H=4；1,000 次 AWAC 更新 | chunk 训练、融合、EMA 推理完成 |
| `runs/cube_smoke` | 240 条真实仿真随机数据；20 次更新 | 训练和独立 checkpoint 推理完成 |
| `runs/verify_bc` | 100 条仿真随机数据；10 次 BC 更新 | 监督模式与 H=4 推理完成 |
| `runs/verify_online` | 无离线数据；60 个环境步 | 采集、replay、AWAC 更新完成 |

两个官方数据实验的 5 回合评估均为 **0% 成功率、平均回报 −200**。这些结果确认实现可运行，尚未达到任务学习目标，需要更长训练和超参数实验。详细版本与实测指标见 [validation.json](doc/robotics_learning/validation.json)。数据、日志和 checkpoint 留在本机的 `data/`、`runs/`，不纳入源码版本管理。
