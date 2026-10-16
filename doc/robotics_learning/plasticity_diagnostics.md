# 模型可塑性检测工具

`flow_rl.diagnostics` 只依赖 PyTorch 和 Python 标准库，可用于普通 `nn.Module`、现有 MLP、diffusion、actor 或 critic，无需启动环境、训练器或数据下载。默认测量指定层的输出；权重幅度统计传入的整个模型。

## 一行接入

### 不改 forward：注册到指定层

```python
from flow_rl.diagnostics import PlasticityMonitor

probe = PlasticityMonitor(model, layer="net.3", dormant_threshold=0.01)
```

`layer` 可以是 `named_modules()` 中的路径，也可以直接传入模块对象，例如 `layer=model.net[3]`。现有 MLP 的 `net.3` 是第二个 GELU 激活；不同配置应先检查 `dict(model.named_modules())`。同一模型可创建多个 probe，分别测量不同层。

### 在 forward 中插入一行

```python
probe = PlasticityMonitor(model)  # 构造一次，持有待测 model

# 在 forward 的目标层输出之后：
h = probe(h)
```

工具返回原来的 Tensor，保留其梯度。probe 可作为模型的普通属性保存；它不是 `nn.Module`，也不将 model 注册成子模块，因此不会引入递归模型结构或改变 `state_dict`。自动 hook 和手动插入二选一，避免重复统计。

随整个模型一起 `copy.deepcopy(model)` 时，probe 会绑定到复制后的模型，统计与 hook 相互独立。若单独复制 probe，请改为构造一个新的 `PlasticityMonitor`。检查点仍使用模型的 `state_dict`，恢复后重新注册需要的 probe。

### 训练：当前 batch

```python
model.train()
for observations, targets in train_loader:
    optimizer.zero_grad()
    prediction = model(observations)  # 自动 hook 记录激活和当时的模型权重
    loss = loss_fn(prediction, targets)
    loss.backward()
    optimizer.step()
    print(probe.compute())  # 恰好五个标量；不会混入之前的 batch
```

权重幅度对应被测层执行时的参数，即通常是本步 optimizer 更新之前的值。训练模式下，每次被测层调用替换上一批统计。`probe.latest` 返回最后一次调用的五指标。

### 评估：全数据集与 batch 均值

```python
import json
import torch

model.eval()
probe.reset()  # 每轮评估开始调用；重复 model.eval() 不代表新一轮
with torch.no_grad():
    for observations, _ in eval_loader:
        model(observations)

print(json.dumps(probe.summary(), indent=2))
dataset_metrics = probe.compute()              # 合并所有样本后计算
average_metrics = probe.compute("batch_mean")  # 每批结果按样本行数加权平均
probe.close()                                 # 移除 hook，结果仍可读取
```

`summary()` 包含 `mode`、`layer`、`num_samples`、`num_batches`、`num_features`、`dataset` 和 `batch_mean`。切换 train/eval 后，下一次被测层调用会自动开始新的统计窗口；仍建议每轮评估显式 `reset()`。评估期间应固定模型参数。

`batch_mean = sum(batch_rows × batch_metric) / sum(batch_rows)`，因此最后不足一批的数据不会获得过高权重。它与全数据集指标不同。例如两批分别只激活两个神经元中的一个：每批死亡比例 50%，合并后为 0%；每批 rank 为 1，合并后可以为 2。工具同时输出两个结果，保留两种统计含义。

也可以使用 `with PlasticityMonitor(model, layer=...) as probe:` 自动移除 hook。设置 `probe.enabled = False` 暂停测量；再次启用会继续原窗口，需要新窗口时调用 `reset()`。

## 五个指标的定义

将指定层的输出整理为未中心化的矩阵 `X ∈ R^(N×D)`，每列为一个神经元，每行为一次激活观测；`σ` 是 `X` 的奇异值。

| 返回键 | 定义 |
| --- | --- |
| `dead_neuron_ratio` | `mean_j[all_i(X[i,j] == 0)]`。死亡判据始终为严格零值。 |
| `dormant_neuron_ratio` | `mean_j[activation_frequency[j] < dormant_threshold]`，默认阈值 `0.01`。 |
| `mean_absolute_weight` | 所选整个模型权重的 `sum(abs(weight)) / 总元素数`。 |
| `stable_rank` | `sum(σ²) / max(σ)²`，常规 stable rank。 |
| `effective_rank` | `exp(-sum(p × log(p)))`，其中 `p = σ / sum(σ)`，忽略零概率项。 |

比例范围为 `[0,1]`，例如 `0.25` 表示 25%。休眠阈值使用严格小于：激活频率恰好 1% 时不计入默认休眠比例。`dormant_threshold` 可取 `[0,1]`。

默认 `activity_mode="nonzero"`、`activation_threshold=0.0`，激活频率为 `mean_i(abs(X[i,j]) > activation_threshold)`，负值也算活跃。这与作者 CIFAR 实现的非零判据一致，适用于本仓库 GELU/SiLU 输出。可设置 `activity_mode="positive"`，改为 `X > activation_threshold`；对 ReLU 的非负输出，两者在阈值 0 时等价。`activation_threshold` 只改变休眠统计，死亡比例仍要求严格为零。

两个秩依赖目标层、观测样本及特征布局。不做中心化、标准化或奇异值截断；effective rank 使用 **奇异值本身**归一化。全零矩阵显式约定两个秩为 `0.0`。空输入、NaN/Inf、非法维度会报错；尚未记录数据时 `compute()` 也会报错。

### 权重范围

默认计入模型所有层中名字为 `weight`、以 `_weight` 结尾以及 `values` 的参数，涵盖 Linear、Conv、Embedding、归一化 scale、attention 的 `in_proj_weight`，以及本仓库 `BudgetedLinear` 的实际连接权重。共享 Parameter 只计一次，冻结权重也计入。buffer、bias、`log_std` 等其他参数默认不计。

- `include_bias=True`：额外计入 `bias` 和 `*_bias`。
- `parameter_filter=lambda name, parameter: True`：计入所有 Parameter，与作者 CIFAR 统计口径一致。
- 自定义 `parameter_filter`：完全替代默认名字筛选，用于特殊参数命名、仅检查部分模块等。
- 稀疏模型的均值分母为实际存储的连接权重数，省略的结构性零连接不计入。
- 无符合规则的权重时返回 `None`，JSON 中为 `null`。

作者各实验的权重范围并不完全相同：CIFAR 实现遍历全部参数，MNIST 实现选取 Linear.weight。本工具默认按“权重”统计，比较不同模型时应保持筛选规则一致。

### 特征维和重复调用

`feature_dim=-1` 是默认值，其他所有维展平成样本行：

| 输出布局 | 参数 | 表征矩阵 |
| --- | --- | --- |
| MLP `[B,D]` | 默认 | `[B,D]` |
| Transformer `[B,T,D]` | 默认 | `[B×T,D]`，一个 channel 为一个神经元 |
| Action chunk `[B,H,A]` | 默认 | `[B×H,A]` |
| CNN `[B,C,H,W]` | `feature_dim=1` | `[B×H×W,C]` |

如果要把整个 action chunk 作为每个输入的一行，可手动 `probe(actions.flatten(1))`。多输出层可以用 `output_selector=lambda output: output[0]` 或字典索引选取 Tensor；hook 仍保留完整原输出。

`num_samples` 表示上述矩阵的行数，`num_batches` 表示被测层调用次数。diffusion 多次去噪、共享层复用、梯度 checkpoint 重算都可能让一个数据 batch 对应多次调用。训练统计最后一次调用，评估累计所有启用期间的调用。若要固定 flow time/noise 测量，请直接用固定 time/noise 调用一次 `velocity` 或 `per_sample_loss`，或手动在目标 Tensor 处测量；正式对照实验应保持这些条件一致。工具不自动移除 padding、筛掉 token 或跨 DDP 进程聚合；需要在输入端选择有效激活，分布式结果默认仅覆盖当前进程。

## 独立运行示例

```bash
uv run python examples/plasticity_diagnostics.py --output runs/plasticity_demo.json
# 只检测其中一种模型
uv run python examples/plasticity_diagnostics.py --model diffusion
```

示例使用固定种子的 129 条合成状态/动作、batch size 32，实际执行训练反向传播，并在评估中覆盖全部 5 个 batch（包含末尾 1 条样本）。同时输出 MLP 与 diffusion 的五指标。diffusion 评估固定 `time=0.5` 和零噪声，选取 SiLU 输出；token 数为 5，因此评估 `num_samples=645`。示例用于检查检测链路，不表示模型学习效果或可塑性优劣。

## 计算成本与数值实现

测量过程将 detached 激活与所选权重转到 CPU float64；不保存 autograd 图，不修改模型参数、输出或随机数状态。GPU 模型测量会发生设备同步。精确谱计算有额外成本，可只在指定测量步骤启用。

训练每批仅做一次 SVD，`compute()` 直接读取已计算的结果。评估的全数据集结果也会缓存到下一次观测，重复读取不会重复分解。

评估用流式 QR 保存至多 `D×D` 的因子；其奇异值与全部激活拼接后的矩阵相同（浮点精度内），保留内存为 `O(D²)`，不随数据集样本总数增长。单次更新还有当前 batch 和分解临时内存。不会通过构造 Gram 矩阵平方条件数；batch 均值仍需要每批的奇异值计算。宽层可选择更靠后的低维层或预先定义的降维表征，并记录这种选择。

## 依据

- [仓库研究笔记](../flow_manifold/loss_of_plasticity_codex_research_notes_zh.md)：第 8、10 节。
- [论文及 Methods](https://www.nature.com/articles/s41586-024-07711-7)：effective rank 的熵定义。
- [作者仓库 README](https://github.com/shibhansh/loss-of-plasticity/blob/main/README.md)。
- [CIFAR 分析源码](https://github.com/shibhansh/loss-of-plasticity/blob/main/lop/incremental_cifar/post_run_analysis.py)：非零频率、默认 1% 阈值及权重统计。
- [作者 rank 工具](https://github.com/shibhansh/loss-of-plasticity/blob/main/lop/utils/miscellaneous.py)：effective rank。
- [MNIST 测量代码](https://github.com/shibhansh/loss-of-plasticity/blob/main/lop/permuted_mnist/online_expr.py)：死亡单元及 Linear 权重统计。

五项指标是内部诊断量；判断可塑性变化仍需结合新任务学习表现。这里按需求使用常规 stable rank，作者另一个累计 99% 奇异值指标未作为本工具的 stable rank。
