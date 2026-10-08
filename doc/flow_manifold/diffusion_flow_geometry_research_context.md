---
title: "从生成策略机制到无条件动作几何约束：论文与研究对话整理"
document_id: diffusion_flow_action_geometry_research_context
project: Robotic Learning
version: "1.0"
created_date: "2026-09-30"
language: zh-CN
audience: [Codex, robotics_researcher]
status: research_proposal_not_implemented
experiments_run: false
primary_paper: "Much Ado About Noising: Dispelling the Myths of Generative Robotic Control"
primary_paper_url: "https://proceedings.iclr.cc/paper_files/paper/2026/file/9299ca9ed58731945e934adb5b71728c-Paper-Conference.pdf"
benchmark: OGBench
recommended_first_stage: paired_rollout_diagnostic
---

# 从生成策略机制到无条件动作几何约束

## 论文核心论点、研究对话与 Codex 实验交接

**摘要。**主论文将生成式行为克隆的优势与“训练噪声＋有中间监督的迭代计算”联系起来，并把动作流形贴合性作为候选解释。本次对话将问题推进到 off-policy RL：能否利用一份冻结的无条件动作先验，约束 Q 引导的局部更新方向，再通过条件 flow matching 保留改进？当前拟议实验采用 OGBench Cube-Single，先做同状态配对 rollout，后做小规模在线微调。本文保留概念边界、关键推导、算法演化、实验对照和待确认项；没有实验结果。

## 目录

- [00. 阅读入口与当前研究目标](#overview)
- [01. 主论文核心观点与证据](#paper)
- [02. 当前对话的结论与推导](#dialogue)
- [03. 无条件动作几何候选算法](#algorithm)
- [04. OGBench 最小验证实验](#experiment)
- [05. Codex 交接与待决事项](#handoff)
- [06. 来源与证据边界](#sources)

<a id="overview"></a>
## 00. 阅读入口与当前研究目标

### 00.1 文档用途与状态

本知识文档整理了《Much Ado About Noising: Dispelling the Myths of Generative Robotic Control》的核心论点，以及本次对话从论文解读到 OGBench 最小实验的研究设计。目标读者是接手实现的 Codex 和研究人员。

**当前状态：研究方案已形成，实验尚未执行。本文不是实验报告，也不是已有算法的官方实现规范。**

整理范围是当前对话，不是将项目内全部 Sergey Levine 讲座文档重新汇总。中英文重复回答已合并；保留核心公式、论证边界、方案演化和未确定的实现细节。

| 标签 | 含义 | 使用限制 |
|---|---|---|
| `PAPER` | 主论文的观点、方法和实验观察 | 不自动推广到所有 diffusion、flow、RL 或机器人任务 |
| `SOURCE` | 所列官方源码/文档支持的接口信息 | 需要在实际实现时固定版本并运行核验 |
| `DISCUSSION` | 当前对话形成的概念区分与方法梳理 | 不等于该论文提出的结论 |
| `DERIVATION` | 对话中的数学推导或说明性反例 | 只在明示假设下成立，不是已验证的硬件保证 |
| `PROPOSAL` | 对话提出的算法或实验设计 | 未运行、未验证，不声称新颖性已确认 |
| `OPEN` | 对话尚未闭合或需要代码核验的事项 | Codex 不得将其悄悄补成既定事实 |

来源审阅说明：整理时重新读取了主论文相关章节、公式和表格，并查看 OGBench 官方 README、Cube 与 ManipSpace 相关源码。没有运行 OGBench、训练网络或进行 rollout；其他相关工作主要作为此前对话的阅读索引保留，未在本次整理中逐篇重新核查。[P1] [S-OGB] [S-CUBE] [S-MANIP] [S-REGISTER]

### 00.2 一句话研究目标

> 学习一个跨任务复用的无条件动作结构先验，用它约束 off-policy RL 的动作改进方向，在保留条件 flow policy 对状态的响应能力和已有控制能力的同时，减少在线训练中的性能退化。

最终优化对象是 **closed-loop performance（闭环任务表现）**，不是动作似然、去噪误差或流形距离本身。

### 00.3 当前最小算法

```text
相容的历史动作块 → 无条件 flow / denoiser → 冻结的局部几何算子
                                                 ↓
当前状态 → 条件 flow policy → 参考动作块 → Q 梯度的几何预条件
                                                 ↓
                                        改进后的干净动作块
                                                 ↓
                              条件 flow-matching 监督更新
                                                 ↓
                            真实环境交互 → replay → critic 更新
```

**分工固定：**无条件模型描述动作结构；条件 actor 决定当前状态下的动作；critic 学习实际动作的长期结果。三者不可互相替代。

### 00.4 接手实现时的优先级

1. 先读第 01–02 节，理解为什么“流形贴合”不等于“稳定控制”。
2. 读第 03 节，理解局部几何预条件器及完整候选方案。
3. 以第 04 节的最小实验为首版范围，不把完整方案中的额外模块全部加入。
4. 按第 05 节先关闭接口与数据语义问题，再运行局部配对 rollout。
5. 局部诊断有正面信号后，才运行三种学习方法、三个种子的在线微调。

模块化版本按 `00_README.md` 到 `06_sources.md` 顺序阅读。单文件整合版已经包含相同内容；知识库选择其中一种导入方式即可，避免重复索引。

### 00.5 当前决定与尚未决定的事项

| 类型 | 当前记录 |
|---|---|
| 用户明确提出的方向 | 使用全部相容数据预训练无条件去噪生成器，为条件 flow policy 的优化提供动作几何 |
| 用户指定的实验平台 | OGBench |
| 对话建议的首版任务 | `cube-single-play-singletask-task2-v0` |
| 对话建议的首版改进方式 | 对干净动作块做一次几何预条件的 Q 梯度改进，再做条件 FM 拟合 |
| 对话建议的对照 | REF、EUCLIDEAN、GEOMETRY、ROTATED |
| 未确定 | 实际成功的初始化 checkpoint、仓库 commit、完整数据处理细节、数值阈值、最终超参数 |
| 尚无 | 实验成功率、学习曲线、几何收益证明、在线稳定性保证、运行时间估计 |

### 00.6 统一符号

| 符号 | 含义 |
|---|---|
| $s,o$ | 环境状态、策略观测；首版采用 state-based 观测，但二者概念仍需区分 |
| $a_t\in\mathbb R^d$ | 单个原始控制动作 |
| $A\in\mathbb R^{D_A}$ | 动作块，$D_A=Hd$，按时间顺序展平 |
| $H,h,\ell$ | 预测长度、承诺执行长度、提前终止时的实际执行长度 |
| $\pi_0,\pi_k,\pi_\theta$ | 固定初始参考、当前参考快照、正在学习的条件策略 |
| $G_0(s,z)$ | 冻结的条件动作生成器 |
| $v_\psi(x,t)$ | 无条件 flow 速度场 |
| $D_{\psi,\sigma}(y)$ | 无条件模型导出的最终动作均值去噪器 |
| $Q_\phi(s,A)$ | 给定执行规则下的动作块 critic |
| $\mathcal M_s,\mathcal M_u$ | 条件动作几何、跨状态边缘动作几何；二者不等价 |
| $P_T,P_N$ | 理想切向与法向正交投影 |
| $J_D,J_G,J_\theta$ | 去噪器对输入、生成器对潜变量、actor 对参数的 Jacobian |
| $\sigma$ | 加性噪声尺度；不等于 flow 时间 |
| $\eta,\lambda,\kappa,\rho$ | 局部步长、几何代价系数、首版预条件强度、实际动作 RMS 预算 |
| $e_u,B_u,M_u$ | 去噪残差、残差 Jacobian、半正定几何代价矩阵 |

全文采用 **$t=0$ 为噪声、$t=1$ 为数据** 的 flow 时间约定。


---

<a id="paper"></a>
## 01. 主论文：Motivation、核心观点与证据

### 01.1 论文身份与研究范围

- 标题：**Much Ado About Noising: Dispelling the Myths of Generative Robotic Control**。
- 版本：用户提供的 ICLR 2026 proceedings PDF。
- 研究对象：主要是 flow-based generative control policies 在行为克隆基准上的表现，不是 off-policy RL 算法。
- 论文位置索引：动机见 §1；架构、多峰与表达能力见 §3；MIP 见 §4；流形贴合性与迭代训练解释见 §5；指标细节见 Appendix G。

来源：[P1]。以下按此前对话的“动机—观点—论据—边界”结构整理。

### 01.2 Motivation：为什么要重新解释生成策略的优势？

`PAPER`：生成建模追求复现数据分布；控制追求能带来任务成功的动作。多个动作都可能成功，但控制器不一定需要重现它们的全部概率。

论文据此检验三个问题：生成策略的优势是否被网络架构与 action chunking 混淆？是否真的依赖条件动作多峰？是否主要来自迭代计算而非完整分布学习？[P1]（§1、§3）

### 01.3 核心观点—证据矩阵

| 编号 | 作者观点 | 主要证据 | 解读边界 |
|---|---|---|---|
| P-C1 | 强回归基线比通常认为的更有竞争力 | 用相同现代骨干比较 regression 与 flow；多数设定接近 | Figure 2 对两种方法分别选最佳架构，不是任意回归配置均可替代 flow |
| P-C2 | 多峰建模不足以解释观察到的全部差距 | 固定观测的采样可视化、动作均值测试、确定性专家重采数据 | 没观察到明显多峰，不等于所有机器人数据都没有多峰 |
| P-C3 | 更多积分步不自动提供更复杂的观测—动作映射 | 条件假设下的 Lipschitz 上界及经验敏感度测量 | 不是所有表达能力意义上的等价定理 |
| P-C4 | 训练噪声与有中间监督的迭代计算的组合很重要 | MIP 与只保留其中一个组件的消融 | 不能简化成“加噪即可”或“多调用网络即可” |
| P-C5 | Manifold adherence 是更相关的候选机制 | OOD 观测附近的动作投影残差与性能比较 | 是代理指标和机制假说，不是闭环稳定性证书 |

来源：[P1]（§3–5，Figure 2，Tables 1–4）。

### 01.4 论文使用的三个组件

| 组件 | 原文术语 | 含义 |
|---|---|---|
| C1 | Distributional Learning | 拟合给定观测下的动作分布 |
| C2 | Stochasticity Injection | 训练时注入随机扰动 |
| C3 | Supervised Iterative Computation | 迭代生成动作，并对中间阶段提供监督 |

| 方法 | C1 | C2 | C3 |
|---|---:|---:|---:|
| Regression | 否 | 否 | 否 |
| Straight Flow（SF） | 否 | 是 | 否 |
| Residual Regression（RR） | 否 | 否 | 是 |
| Minimal Iterative Policy（MIP） | 否 | 是 | 是 |
| Flow | 是 | 是 | 是 |

这里 C1 是**训练目标**属性，不由推理是否随机决定。将一个经 flow matching 训练的模型固定为零噪声推理，不等于移除了分布学习。[P1]（§4、p.8 脚注）

### 01.5 MIP：对话中需要保留的具体公式

令 $z\sim\mathcal N(0,I)$、$t_\star=0.9$，网络 $f_\theta(o,x,t)$ 预测最终动作而不是速度。

$$
\mathcal L_{\mathrm{MIP}}
=\mathbb E\left[
\|f_\theta(o,0,0)-a\|^2
+
\|f_\theta(o,t_\star a+(1-t_\star)z,t_\star)-a\|^2
\right].
$$

推理为：

$$
\hat a_0=f_\theta(o,0,0),\qquad
\hat a=f_\theta(o,t_\star\hat a_0,t_\star).
$$

训练第二阶段看的是带扰动的专家动作；推理第二阶段看的是第一阶段预测。推理没有随机噪声。它不是把完整 flow 直接截断为两步。[P1]（Eqs. 4.2–4.3）

### 01.6 当前对话引用的关键实验数字

以下是论文报告的数字，**不是本研究方案的实验结果**。不同表格使用的任务、数据与聚合方法不同，不可混为同一组实验。

**Table 1：采样方式。**

| 任务 | 零噪声 | 随机采样 | 64 个生成动作的均值 |
|---|---:|---:|---:|
| Push-T | 0.97 | 0.97 | 0.95 |
| Kitchen | 0.99 | 0.99 | 0.97 |
| Tool-Hang | 0.78 | 0.80 | 0.76 |

“均值”指输出动作的均值，不是先把输入噪声平均成零再生成。[P1]（Table 1）

**Table 2：去除专家随机性后仍有差距。**

| 数据 | Flow | Regression |
|---|---:|---:|
| 原始示教 | 0.78 | 0.58 |
| 确定性策略重新收集 | 0.72 | 0.64 |

差距缩小但没有消失。支持“多峰不足以解释全部差异”，不支持“多峰完全无贡献”。[P1]（Table 2）

**Table 4：流形与验证误差。**

| 指标 | Regression | SF | RR | MIP | Flow |
|---|---:|---:|---:|---:|---:|
| Off-manifold $L_2$ | 0.067 | 0.063 | 0.062 | 0.054 | 0.042 |
| Validation $L_2$ | 0.290 | 0.234 | 0.224 | 0.195 | 0.217 |

该表对应 state-based Tool-Hang 的确定性数据设置，聚合 3 种架构和 32 个状态。正文个别表述与表中数值不完全一致；这里保留表中数字，不将“验证误差完全相同”当作事实。[P1]（Table 4）

**Table 6：中间监督消融。**MIP 为 0.80，MIP-NoSupervision 为 0.42，RR 为 0.54，单步 SF 为 0.54。[P1]（Appendix C.3，Table 6）

**Table 9：LIBERO 微调。**LIBERO-10 上 MIP/Flow/$L_2$ regression 为 82.2/81.6/78.0；Object 上 MIP/Flow 为 95.8/97.4。因此是整体竞争力，不是 MIP 全面获胜。这是已有 $\pi_0$ 的微调对比，不能否定生成式预训练的潜在价值。[P1]（Table 9）

**Table 18：验证误差与成功率可以不同向。**Sudeep-DiT regression 的成功率/$L_2$ 验证误差为 0.20/0.063，9 步 flow 为 0.76/0.080。[P1]（Table 18）

### 01.7 Manifold adherence 的实际指标

论文在状态的近邻训练样本中收集对应专家动作，构造列矩阵 $\mathbf A_o$，计算：

$$
e_\perp(o)
=\min_c\|\pi(o)-\mathbf A_oc\|_2.
$$

它是到邻近专家动作**线性张成空间**的距离，不是精确非线性流形距离，也不是下一状态误差、成功率或稳定性指标。[P1]（§5.1、Appendix G.2）

`DERIVATION`：线性空间包含 $a$ 时也包含 $-a$、$0$、$100a$，所以该指标本身不确定正确动作的符号与幅度。

### 01.8 理论与机制的证据边界

论文在 $p(a\mid o)$ 满足 $\kappa$ 强对数凹等假设，且真实流场满足 $\|\nabla_o b_t^\star\|\le L$ 时，给出理想积分策略的敏感度上界：

$$
\|\nabla_o\pi^\star(z,o)\|
\le L\sqrt{1+\kappa^{-1}}.
$$

这里限制的是特定假设下的输入敏感度，不是证明所有 flow 与 regression 等价。[P1]（Theorem 1、Appendix J）

作者将 C2 的作用解释为增加**内部生成过程**的覆盖，使后续迭代能处理前一步的不完美结果；不是环境探索或任务级 DAgger 的等价替代。[P1]（§5.2）

主论文没有建立：为何这种几何偏置一定优于训练良好的 regression；为何 off-manifold 指标降低必然带来闭环收益；以及在 RL 更新中维持该性质的保证。[P1]（§5.1–5.2）

### 01.9 论文与后续研究方案的分界

**主论文没有提出本对话的“无条件先验＋Q 梯度预条件＋条件 FM 提取＋OGBench”算法。**后者是从论文机制假说出发提出的研究方案，必须独立验证。


---

<a id="dialogue"></a>
## 02. 当前对话：问题演化与保留结论

### 02.1 对话演化记录

| 阶段 | 用户问题 | 对话形成的结论 | 后续影响 |
|---|---|---|---|
| D01 | 论文的 motivation、观点和论据是什么？ | 从分布拟合转向控制表现；关注噪声与监督式迭代 | 提出动作几何的研究切入点 |
| D02 | 流形贴合性是否等于闭环稳定性？ | 不等于；一个是输出几何，一个是反馈动力学 | 不能把几何正则包装成稳定性证书 |
| D03 | DAgger、online RL 等是否也是工具？ | DAgger 改变监督所覆盖的状态；RL 优化后果 | 几何约束是可组合的组件，而非替代所有数据与优化方法 |
| D04 | Q 是否也需要 diffusion？如何防止精细动作退化？ | Q 不必生成式；要控制价值误差、梯度误差与提取误差 | 保留普通 critic，研究受约束的策略提取 |
| D05 | 如何让 Q 梯度留在隐式动作流形上？ | 切向投影＋回缩；可由 denoiser 或固定生成器近似提供 | 分离真实环境 Q 与用于 actor 的几何引导 |
| D06 | 能否预训练无条件动作生成器作为共享流形？ | 可作为共享动作结构先验，但会丢失状态相关约束 | 不直接让条件策略匹配无条件概率；使用局部更新几何 |
| D07 | 如何设计优化算法？ | 干净动作空间局部改进，再通过 FM 拟合回条件策略 | 避免直接让 Q 梯度穿过全部 flow 参数 |
| D08 | 用 OGBench 做最小验证 | 单任务、四组对照、先配对 rollout 再在线学习 | 先证明学到的方向有控制价值，再研究提取与规模化 |

中文翻译回合没有引入单独算法变更，已与对应英文回答合并。

### 02.2 流形贴合不等于闭环稳定

`DISCUSSION`：应区分三种“稳定”：

| 对象 | 问题 |
|---|---|
| 内部生成迭代 | 动作候选是否在迭代中数值稳定、逐步修正？ |
| 物理闭环 | 动作影响环境后，状态扰动会增长、保持有界还是衰减？ |
| 在线学习过程 | 随 actor/critic 更新，已有能力是否出现暂时或持续退化？ |

连续时间闭环为：

$$
\dot s=f(s,\pi(h(s))).
$$

动作几何中没有自动包含 $f$，所以不能仅由几何距离推断物理稳定。

**对话中的反例，非论文实验：**

$$
\dot x=u_x,\qquad \dot y=y+u_y.
$$

示教仅在 $y=0$，动作为 $(-x,0)$，记录的动作流形为 $\mathcal M_a=\{(u_x,0)\}$。

- 策略 $(-x,0)$ 始终在该流形上，但 $y(t)=e^t y(0)$，微小扰动发散。
- 策略 $(-x,-2y)$ 在 $y\ne0$ 时离开记录流形，但闭环为 $\dot x=-x,\dot y=-y$，状态收敛。

**含义：恢复所需动作可能恰好是示教流形没有覆盖的方向。**

若要给出稳定性证据，还需要类似 Lyapunov 条件：

$$
\nabla V(s)^\top f(s,\pi(h(s)))<0
$$

在明确区域内成立，并满足其他相关条件。当前方案没有建立这样的证书。

### 02.3 除动作几何之外的工具

以下是对话中的方法地图，不是主论文的贡献，也不是首版实验必须实现的组件。

| 工具 | 主要作用 | 本研究中的位置 |
|---|---|---|
| DAgger / HG-DAgger | 在学习者真正到达的状态收集专家纠正 | 补足恢复监督；需要可用专家或人工介入 |
| DART | 在示教过程中引入执行扰动，诱发纠正行为 | 与内部去噪噪声不同，是环境轨迹级覆盖 |
| CCIL | 利用局部动力学结构构造纠正样本 | 依赖局部模型准确性 |
| Online RL / RLPD | 用真实任务结果与 replay 改进策略 | 当前主路线；不能假定在线交互天然安全 |
| HIL-SERL | 将示教、人工干预与 reward-driven RL 结合 | 可以补充高质量恢复数据 |
| AggreVaTe | 通过 cost-to-go 进行代价敏感模仿 | 区分动作偏差的不同后果 |
| IQL / offline RL | 从固定、带奖励的数据提取高价值行为 | 用于共同初始化；受覆盖限制 |
| Guided policy search | 先在行为/轨迹空间改进，再监督拟合全局策略 | 为当前“局部改进—FM 提取”提供结构类比 |
| Stable-BC、MPC、扰动训练 | 分别针对误差动力学、未来后果、扰动覆盖 | 后续对照或扩展，不放入首版 |
| 历史/触觉/状态估计 | 提供正确决策所必需的信息 | 无法由改进梯度代替 |
| 低延迟执行、安全过滤与备用控制 | 改善响应与约束执行 | 与学习稳定性分开评估 |

相关阅读入口见第 06 节。这里只保留对话中的功能区分，不作最新方法排名。

### 02.4 为什么 Q 不需要 diffusion？

生成器学习动作分布；普通 critic 学习给定动作的标量价值：

$$
Q^\pi(s,A)=\mathbb E[G_t\mid s_t=s,A_t=A,\text{之后遵循 }\pi].
$$

标量输出不妨碍多个高价值模式。例如 $Q(a)=-(a^2-1)^2$ 同时在 $a=\pm1$ 取最大值，不会因为使用回归就自动把动作平均为零。

**回归一个动作的回报，不等于回归动作标签的条件均值。**

主要风险是数据与梯度支持，而非 critic 是否生成式。固定状态、动作 $a=(u,v)$、数据仅覆盖 $v=0$ 时：

$$
\widehat Q_c(u,v)=g(u)+cv
$$

对任意 $c$ 都与流形上数据一致，但法向梯度为 $c$，完全无法由这些数据确定。

因此：

- 候选筛选需要可靠的相对价值；
- 动作梯度优化还需要可靠的局部导数；
- 普通 TD 验证误差较低，不能单独建立以上性质；
- 将 critic 换成 diffusion，不能自动补充未观测动作的反事实信息。

### 02.5 critic 的评价对象必须与策略优化空间一致

可选择 $Q(s,A)$ 评价物理动作块，或在固定生成器下定义 $Q_z(s,z)$ 评价潜变量。若评价中间带噪动作，则必须另行定义其条件语义。

当前选择是 **最终干净动作块上的普通双 critic**。

若预测 $H$ 步、只执行 $h$ 步，则 $Q$ 必须对应真实执行前缀及后续重规划策略。不能把完整块执行价值与前缀执行价值混用。

也不能给记录动作加噪后，继续沿用原奖励与下一状态，制造一个从未发生过的转移。

### 02.6 如何保留已有精细动作能力：四类策略提取

| 路线 | 保留什么 | 仍然可能出错的地方 |
|---|---|---|
| 从冻结生成器样本中筛选（IDQL 式思路） | 原生成机制及完整候选结构 | critic 选错；候选更多时更容易找到高估；跨 chunk 切换不连贯 |
| 冻结生成器，学习潜空间选择（DSRL） | 去噪/生成网络参数 | 潜输入偏移；新选择器改变反馈律；不能发现生成器之外的新技能 |
| 优势加权去噪或 QAM/TRQAM 式更新 | 逐步生成形式及显式先验约束 | 有限 critic 误差、分布拟合误差、内部 KL 不等于物理轨迹安全 |
| 有界残差或动作编辑（EXPO 式思路） | 基础动作作为参考 | 欧氏距离小不等于控制敏感方向变化小 |

这些名称是对话中的关联工作，不表示已完成逐一复现或验证。

**固定生成器不等于固定反馈律。**若：

$$
\pi_\psi(s)=G_0(s,\mu_\psi(s)),
$$

则：

$$
\frac{\partial\pi_\psi}{\partial s}
=\frac{\partial G_0}{\partial s}
+\frac{\partial G_0}{\partial z}\frac{\partial\mu_\psi}{\partial s}.
$$

第二项可以改变扰动响应。共享观测编码器更新也会改变有效生成器映射。

### 02.7 性能保持需要同时控制 critic 误差和提取误差

`DERIVATION`：设 critic 评价参考策略 $\pi_k$，且在相关区域满足：

$$
|\widehat Q_k(s,A)-Q^{\pi_k}(s,A)|\le\epsilon_Q.
$$

设理想改进分布为 $q_k$，估计改善为：

$$
g_k(s)=\mathbb E_{q_k}\widehat Q_k(s,A)-\mathbb E_{\pi_k}\widehat Q_k(s,A).
$$

若实际提取策略满足：

$$
\mathbb E_{\pi_{k+1}}\widehat Q_k
\ge\mathbb E_{q_k}\widehat Q_k-\epsilon_{\mathrm{ext}},
$$

则：

$$
\mathbb E_{\pi_{k+1}}A^{\pi_k}(s,A)
\ge g_k(s)-2\epsilon_Q-\epsilon_{\mathrm{ext}}.
$$

在相关状态上右侧非负，并满足通常策略改进条件时，可建立回报不下降。**本方案没有已校准的 $\epsilon_Q$ 或 $\epsilon_{\mathrm{ext}}$，所以不能宣称该保证实际成立。**

此外，总回报提高也可能以精度下降换取速度。因此，保护精度需要独立指标或约束，例如：

$$
\max_\pi J_{\mathrm{throughput}}(\pi)
\quad\text{s.t.}\quad
J_{\mathrm{precision}}(\pi)\ge J_{\mathrm{precision}}(\pi_0)-\varepsilon.
$$

### 02.8 “流形内梯度”的理想形式

若真实动作流形 $\mathcal M_s$ 已知，则用：

$$
g_T=P_{T_A\mathcal M_s}\nabla_AQ(s,A),
\qquad A^+=R_{s,A}(\eta g_T).
$$

$R$ 是回缩。仅执行 $A+\eta g_T$ 不保证有限步后仍在流形上。圆上的切线位移就是直接反例。

### 02.9 去噪器与固定生成器提供的是几何近似

理想加性高斯去噪器：

$$
D_\sigma(y)=\mathbb E[A\mid y],\qquad y=A+\sigma z,
$$

满足：

$$
J_D=I+\sigma^2\nabla^2\log p_\sigma(y)
=\sigma^{-2}\operatorname{Cov}[A\mid y].
$$

在光滑低维数据几何与充分精确估计等条件下，$J_D$ 可近似切向投影。对复合目标：

$$
\nabla_yQ(s,D_\sigma(y))
=J_D^\top\nabla_AQ(s,D_\sigma(y)),
$$

输出动作一阶变化为 $\eta J_DJ_D^\top\nabla_AQ$。

若冻结生成器 $A=G_0(s,z)$，优化潜变量的一阶动作变化则是：

$$
\Delta A\approx\eta J_GJ_G^\top\nabla_AQ.
$$

两者都是几何预条件，不自动等于精确正交投影。

**满秩薄分布反例：**$G(z_1,z_2)=(z_1,\epsilon z_2)$ 的输出集中在横轴附近，但映射仍满秩。$J_GJ_G^\top=\operatorname{diag}(1,\epsilon^2)$ 会抑制纵向变化；完整列空间的伪逆投影却是 $I$。因此盲目求“严格投影”可能消除有用的各向异性。

### 02.10 不要让环境 critic 假装所有投影等价动作都有相同价值

可为策略引导构造：

$$
Q_{\mathrm{guide}}(s,A)=Q_\phi(s,\Pi_s(A)),
$$

使流形附近的引导具有切向性质。但两个投影相同的真实动作仍可能导致完全不同的接触和回报。

保留分工：

```text
Q_env：学习实际动作的后果，包括流形外的风险。
几何算子：限制 actor 当前允许采用的改进方向。
```

### 02.11 投影动作梯度，不等于更新参数后仍在流形上

即使 $g_T=P_T\nabla_AQ$，普通参数更新产生：

$$
\Delta\theta=\eta J_\theta^\top g_T,
\qquad
\Delta A\approx\eta J_\theta J_\theta^\top g_T.
$$

$J_\theta J_\theta^\top$ 可以再次混合出法向分量。共享参数还会跨状态干扰；动量或 Adam 的预条件也可能改变最终方向。

这正是当前方案选择“先显式改进干净动作，再监督拟合”的理由之一。但监督拟合同样有提取误差，不能据此宣称已经消除了问题。

### 02.12 本对话的核心研究假设

> 无条件模型能否从多任务动作数据中学到有价值的局部方向性，使得在相同实际更新幅度下，几何预条件比普通 Q 梯度更少产生真实有害动作，并且该优势能够通过条件 FM 训练保留到部署策略中？

分成两个必须独立检验的问题：

- **几何改进是否有效？**固定策略、critic、状态，直接执行候选比较。
- **策略提取是否保留收益？**把改进动作拟合进条件 actor 后，再看真实在线学习曲线。


---

<a id="algorithm"></a>
## 03. 无条件动作几何先验：候选优化算法

本节为 `PROPOSAL` 与 `DERIVATION`。这是对话中的完整候选设计；第 04 节会明确缩减为首版实验，不能把本节所有机制一并加入首版。

### 03.1 无条件模型学习什么？

从相容动作数据边缘化状态：

$$
p_u(A)=\int p_{\mathcal D}(A\mid s)\,d_{\mathcal D}(s)\,\mathrm ds.
$$

它可能提取跨关节、跨时间和跨任务共享的协调结构，但丢失“当前状态该做什么”的对应关系。

例如专家 $a=-ks$ 在每个状态上的动作非常精确，边缘动作分布却可能覆盖一个宽区间。因此 **$\mathcal M_u$ 不是 $\mathcal M_s$ 的替代品**。

首版“所有数据”限定为同一 OGBench 操作系统的相容训练动作块，不包括其他机器人、其他动作单位、测试集或验证集。

### 03.2 为什么不直接把条件策略 KL 拉向无条件先验？

对固定状态分布 $\mu(s)$，定义策略的边缘动作分布：

$$
p_\theta(A)=\int\mu(s)\pi_\theta(A\mid s)\,\mathrm ds.
$$

在相关密度和 KL 有限时：

$$
\mathbb E_s D_{\mathrm{KL}}(\pi_\theta(A\mid s)\|p_u(A))
=I_\theta(S;A)+D_{\mathrm{KL}}(p_\theta(A)\|p_u(A)).
$$

强力最小化该量会同时惩罚状态—动作依赖与边缘分布偏移。它可能抑制精细反馈所需要的状态敏感性。

因此，对话优先选择 **使用先验的局部方向性，而不是保持全部动作频率**。这不排除把完整先验 KL 作为后续对照。

### 03.3 预训练无条件 flow

输入为展平动作块 $A\in\mathbb R^{D_A}$。噪声 $z\sim\mathcal N(0,I)$ 与 $A$ 独立，时间 $t$ 从训练时间分布采样，采用：

$$
x_t=(1-t)z+tA,
$$

$$
\mathcal L_u(\psi)
=\mathbb E\|v_\psi(x_t,t)-(A-z)\|^2.
$$

给定理想最小二乘速度场 $v^\star(x_t,t)=\mathbb E[A-z\mid x_t]$：

$$
\widehat A(x_t,t)=x_t+(1-t)v^\star(x_t,t)
=\mathbb E[A\mid x_t].
$$

若使用加性噪声坐标 $y=A+\sigma z$，令：

$$
t_\sigma=\frac{1}{1+\sigma},\qquad x_{t_\sigma}=t_\sigma y,
$$

则定义：

$$
D_{\psi,\sigma}(y)
=t_\sigma y+(1-t_\sigma)v_\psi(t_\sigma y,t_\sigma).
$$

**自动微分应当对完整的 $y\mapsto D_{\psi,\sigma}(y)$ 求导，不能遗漏输入乘 $t_\sigma$ 的链式因子。**

预训练后冻结 $\psi$；只保留对输入的导数计算。预训练优化过程不得使用验证动作块。

### 03.4 软几何代价

在选定噪声尺度下定义：

$$
e_u(A)=A-D_\psi(A),
$$

$$
B_u(A)=I-J_{D_\psi}(A),
$$

$$
M_u(A)=B_u(A)^\top B_u(A)\succeq0.
$$

于是 $\|B_u\Delta A\|^2$ 衡量局部变化在去噪残差方向上的大小。

理想局部投影条件下 $J_D\approx P_T$，故 $B_u\approx P_N$。**在学习到的网络上，“法向代价”只是解释，确定成立的是 $M_u$ 半正定。**

### 03.5 完整候选方案：局部二次动作改进

从当前条件参考策略采样：

$$
A_0\sim\pi_k(\cdot\mid s),\qquad
 g=\nabla_A Q_{\mathrm{rob}}(s,A_0).
$$

$Q_{\mathrm{rob}}$ 可以先用两个 critic 的较小值；这是一种保守启发式，不是置信下界。

解：

$$
\max_{\Delta A}
\left[
 g^\top\Delta A-
 \frac{1}{2\eta}\|\Delta A\|^2-
 \frac{\lambda}{2}\|B_u(A_0)\Delta A\|^2
\right].
$$

得到：

$$
\boxed{
\Delta A^\star=(\eta^{-1}I+\lambda M_u(A_0))^{-1}g.
}
$$

若 $J_D=P_T$、$M_u=P_N$，则：

$$
\Delta A^\star
=\eta P_Tg+\frac{\eta}{1+\eta\lambda}P_Ng.
$$

切向分量保留，法向分量缩小。软约束保留了一定的行为扩展空间。

高维实现可用 JVP/VJP 计算 $M_uv=(I-J_D)^\top(I-J_D)v$，再做迭代求解。首版 $D_A=20$ 时直接构造 Jacobian 与小矩阵即可。

### 03.6 完整候选方案：有限步的相对几何约束

直接执行 $D_\psi(A_0+\Delta A)$ 可能在 $\Delta A=0$ 时也改变原动作。因此对话提出相对参考残差：

$$
C_u(A;A_0)=\|e_u(A)-e_u(A_0)\|^2.
$$

它满足：

$$
C_u(A_0;A_0)=0,\qquad
C_u(A_0+\Delta A;A_0)\approx\|B_u(A_0)\Delta A\|^2.
$$

对 $A_\alpha=A_0+\alpha\Delta A^\star$ 回溯，候选检查为：

$$
C_u(A_\alpha;A_0)\le\varepsilon_{\mathrm{geo}}^2,
\quad \|A_\alpha-A_0\|\le\delta,
\quad Q_{\mathrm{rob}}(s,A_\alpha)\ge Q_{\mathrm{rob}}(s,A_0).
$$

无合格候选时保留 $A_0$。这些条件限制几何变化和估计价值下降，不是物理性能保证。

**首版 OGBench 实验不启用这套独有回溯/拒绝机制**，以免 GEOMETRY 比较的其实是更多保护措施，而非方向性。

### 03.7 条件几何界

`DERIVATION`：若局部真实最近点投影 $\Pi_{\mathcal M}$ 存在，且对参考和候选都满足：

$$
\|D_\psi(A)-\Pi_{\mathcal M}(A)\|\le\epsilon_D,
$$

则 $\sqrt{C_u(A;A_0)}\le\varepsilon_{\mathrm{geo}}$ 蕴含：

$$
\operatorname{dist}(A,\mathcal M)
\le\operatorname{dist}(A_0,\mathcal M)
+\varepsilon_{\mathrm{geo}}+2\epsilon_D.
$$

若实际提取动作还满足 $\|\widehat A_\theta-A^\star\|\le\epsilon_{\mathrm{ext}}$，再加上 $\epsilon_{\mathrm{ext}}$。

当前没有证据表明学习到的去噪器具有该一致误差界，也没有从 FM loss 到逐样本 $\epsilon_{\mathrm{ext}}$ 的保证。这条式子用于分解误差，不可写成已证明的算法性能。

### 03.8 改进动作分布的条件 FM 提取

对状态生成一个或多个参考候选，得到改进目标 $A^\star$。停止其梯度，采样与目标独立的新噪声 $\xi$：

$$
\widetilde x_t=(1-t)\xi+tA^\star,
$$

$$
\mathcal L_{\mathrm{improve}}(\theta)
=\mathbb E\|v_\theta(s,\widetilde x_t,t)-(A^\star-\xi)\|^2.
$$

多个模式作为多个目标样本保留，不能先平均成一个动作块。

该训练学习改进目标的条件分布；它不保证相同初始噪声下的学生输出逐点等于对应教师目标。评估提取误差时必须说明比较的是分布、某种配对还是控制结果。

加入能力保留：

$$
\mathcal L_{\mathrm{actor}}
=\mathcal L_{\mathrm{improve}}
+\beta_{\mathrm{keep}}\mathcal L_{\mathrm{keep}}.
$$

保留任务相关、高质量与恢复数据；不要把 mixed play 的无权 BC 当作等价保留项。固定 $\pi_0$ 检测长期漂移，滚动 $\pi_k$ 控制局部更新。

### 03.9 Off-policy critic 的数据契约

真实执行 $\ell$ 个原始动作后的 target 为：

$$
y_t=\sum_{j=0}^{\ell-1}\gamma^jr_{t+j}
+\gamma^\ell b_t\min_iQ_{\bar\phi_i}(s_{t+\ell},A'),
$$

其中 $b_t$ 是与任务和数据语义相符的 bootstrap mask，$A'$ 来自目标版本的真实部署策略。若部署包含筛选或回退，target 也应对应这套规则。

- 更新 critic 的是**实际发生过的转移**。
- $A^\star$ 仅用于 actor 监督，除非它真的被执行并得到对应结果。
- 奖励、终止、时间截断和执行长度需要分别记录。
- 干净动作上的 critic 不直接评价中间带噪动作。
- 最终干净动作的几何矩阵不直接施加到全部中间 flow velocity 上。

### 03.10 完整设计与首版实验的差异

| 组件 | 完整候选设计 | 首版 OGBench 最小实验 |
|---|---|---|
| 无条件先验 | 冻结，后续可考虑慢更新 | 冻结 |
| 动作改进 | 几何二次步，可回溯 | 一次预条件方向 |
| 改动幅度 | 局部信赖域 | 所有学习方法匹配裁剪后 RMS |
| 有限步几何回缩/检查 | 可加入 | 不加入方法专属版本 |
| 不确定性接受规则 | 待校准 | 不作为首版独有模块 |
| checkpoint 回退 | 实际部署可考虑 | 不启用，以记录原始训练退化 |
| actor 提取 | 多步条件 FM＋保留项 | 保持相同形式 |
| 核心对照 | 概念设计 | EUCLIDEAN、GEOMETRY、ROTATED；另有 REF |

**不要把完整方案中的界、保护措施或直觉，误当成最小实验中已经实现的性质。**

### 03.11 已知风险

无条件动作混合可能接近满维，导致 $J_D\approx I$、$M_u\approx0$；也可能只学到各向同性收缩，没有控制相关方向。

多模式交叉附近的局部几何不一定唯一；极小噪声并不总能改进估计。若数据本身含满维噪声，去噪器在小尺度可能接近恒等映射。

无条件先验允许某个动作方向，不表示该方向在当前接触状态可用。几何约束也不会修复流形内部的 critic 高估。

严格冻结支持集最终可能限制新恢复策略。是否慢更新先验、加入状态相关修正或扩大邻域，属于后续研究，不进入首版。


---

<a id="experiment"></a>
## 04. OGBench 最小验证实验

状态：`PROPOSAL / NOT RUN`。本节是首版范围的主要依据。目标是最小成本地判别方向性几何是否有用，不是直接复现主论文，也不是全面评测 OGBench。

### 04.1 可证伪假设

**H-GEOMETRY：在相同状态、参考动作和实际改动幅度下，无条件动作先验的几何方向约束，比普通 Q 梯度更少产生真实有害更新。**

**H-EXTRACTION：上述局部收益能够通过条件 flow-matching 拟合，转化为更小的在线能力退化及更好的学习曲线。**

H-GEOMETRY 不成立时，先不要用更复杂的 actor 训练来掩盖问题。H-GEOMETRY 成立但 H-EXTRACTION 不成立时，优先研究提取误差和跨状态干扰。

### 04.2 环境与范围

| 项目 | 当前方案 | 状态/来源 |
|---|---|---|
| 数据集/环境名称 | `cube-single-play-singletask-task2-v0` | 对话选定的首版候选；按官方命名接口使用 [S-OGB] |
| 观测 | State-based，不用图像 | `PROPOSAL` |
| 任务 | 单方块的指定目标移动任务 | 官方 Cube 任务定义 [S-CUBE] |
| 单步动作 | 末端相对 xyz、相对 yaw、夹爪开合，共 5 维 | [S-MANIP] |
| 归一化动作范围 | 每维 `[-1,1]` | [S-MANIP] |
| 默认原始控制间隔 | 0.05 秒 | [S-MANIP] |
| 预测/执行长度 | $H=h=4$ | `PROPOSAL` |
| 动作块形状 | `(4,5)`，展平为 20 维 | 由上述设定推得 |
| 决策间隔 | 正常执行完整块时为 0.2 秒 | 由上述设定推得 |
| 条件 flow 采样 | 10 步 Euler，组间一致 | `PROPOSAL` |
| 每学习方法每种子在线预算 | 100,000 原始环境步 | `PROPOSAL` |

**任务解释边界：**官方 Cube 成功判定使用目标位置距离不超过 0.04 米。因此本首版只能为接触操作中的保持与改进提供证据，不能据此声称已验证毫米级精细插入。[S-CUBE]（`_compute_successes`）

OGBench 是基准/环境与数据接口，物理模拟由 MuJoCo 提供；实际部署依赖环境低层控制。首版不是纯关节力矩控制实验。[S-MANIP]

### 04.3 环境加载检查

下列是接口检查片段，不是完整训练程序，也未在本次整理中运行：

```python
import ogbench

name = "cube-single-play-singletask-task2-v0"
env, train_data, val_data = ogbench.make_env_and_datasets(
    name, compact_dataset=False
)
eval_env = ogbench.make_env_and_datasets(name, env_only=True)

print(env.observation_space)
print(env.action_space)
print(train_data.keys())
```

使用官方 `make_env_and_datasets`；不要将带 `play` 的数据集名称直接作为普通环境 ID 传给 `gymnasium.make`。[S-OGB]

记录实际软件版本、仓库 commit、任务 ID、观测维度、动作边界、时间限制、reward 与 terminal 语义。源码阅读不是运行成功的证明。

### 04.4 四组对照

| 方法 | 动作改进 | 是否在线训练 | 解释用途 |
|---|---|---:|---|
| REF | 不改进，使用初始策略 $\pi_0$ | 否 | 初始能力基准 |
| EUCLIDEAN | 普通 Q 梯度＋共同动作幅度约束 | 是 | 主基线 |
| GEOMETRY | 真实无条件几何预条件器 | 是 | 待检验方法 |
| ROTATED | 相同谱、随机旋转方向的预条件器 | 是 | 排除一般缩放/正则化解释 |

相同种子内复制相同的预训练 actor、critic、优化器状态和无条件先验。网络、训练次数、执行规则、数据处理和评估预算都保持一致。

### 04.5 实验专用更新算子

给定相同 $s,A_0$：

$$
g=\nabla_A\min(Q_1(s,A_0),Q_2(s,A_0)),
$$

$$
B=I-J_D(A_0,\sigma),\quad M=B^\top B,
\quad\bar M=\frac{M}{\operatorname{tr}(M)/D_A+\epsilon}.
$$

对一个每种子固定的随机正交矩阵 $O$：

$$
\begin{aligned}
 d_E&=g,\\
 d_G&=(I+\kappa\bar M)^{-1}g,\\
 d_R&=(I+\kappa O\bar M O^\top)^{-1}g.
\end{aligned}
$$

然后让三组候选的实际动作改动满足相同 RMS 预算：

$$
\operatorname{RMS}(A^\star-A_0)
=\sqrt{\|A^\star-A_0\|^2/D_A}\approx\rho.
$$

诊断阶段尝试 $\rho\in\{0.02,0.05\}$，$\kappa=4$，$\sigma=0.1$。这些均为建议起点，不是最优超参数。

关键约束：

- RMS 在归一化动作坐标中计算，不是米或弧度。
- 匹配**裁剪后的实际更新**，不能只匹配裁剪前的向量。
- 不可行动作、零梯度、数值异常和未达到预算的情况必须记录。
- 坐标裁剪可能改变方向，应记录饱和比例与实际方向变化。
- 首版不加入额外去噪投影、方法专属 line search、不确定性回退或 checkpoint 回退。
- 用 `solve` 表达线性系统；数学写作中的逆矩阵不要求实现显式求逆。

`OPEN`：对话尚未指定裁剪后匹配 RMS 的精确数值算法与容差，必须在正式实验前固定并单测，见第 05 节。

### 04.6 数据预处理

用原始轨迹切动作块：

$$
A_t=(a_t,a_{t+1},a_{t+2},a_{t+3}).
$$

动作块不能跨原始记录轨迹边界。重叠窗口保持原训练/验证划分，不对窗口随机拆分，以避免同一轨迹泄漏。

无条件模型使用相容的完整训练动作块。critic 则需要块级累计奖励、下一状态和正确的 bootstrap 信息。

**终止语义必须分开：**

- `terminals` 表示数据轨迹结束；
- `masks` 表示是否允许从下一状态 bootstrap；
- play 数据在任务已经达成后也可能继续记录；
- 环境在线运行还有 `terminated` 与 `truncated` 的区别。

这些字段不能互相替换。[S-OGB]

Cube 单任务注册使用 `success_timing='pre'`。奖励与终止应沿用同一实现逻辑，不得在离线数据侧和在线执行侧分别改成不同时间语义，也不要将单任务奖励擅自改写为另一种“成功给 1”规则。[S-REGISTER] [S-OGB]

`OPEN`：轨迹尾部不足 4 步、块内任务成功、提前终止时的未执行后缀、padding 与有效长度尚无完整约定。实现前必须明确；不能简单丢弃全部成功附近样本，也不能拼接下一条轨迹。

### 04.7 预训练共同初始化

#### 无条件几何模型

用全部相容训练动作块训练第 03.3 节的无条件 flow，随后冻结。

建议：3×256 SiLU MLP、batch 256、学习率 $3\times10^{-4}$、100k gradient updates。

验证模型不仅看 denoising loss，还看 $J_D$ 与 $M$ 的谱结构、稳定性及方向性。若 $M\approx0$ 或接近各向同性，先暂停在线实验。不能靠 trace 归一化把数值噪声放大后称为流形信息。

#### 条件 actor 与 critic

对话建议：用 IQL 的价值学习与优势加权条件 flow matching 得到共同的 $\pi_0$。

$$
\mathcal L_{\pi_0}
=\mathbb E\left[w(s,A)
\|v_\theta(s,x_t,t)-(A-z)\|^2\right],
$$

$$
w(s,A)=\operatorname{clip}\left(
\exp\{\beta(\min_iQ_i(s,A)-V(s))\},0,w_{\max}\right).
$$

建议：actor 为 3×256 MLP，两个 critic 为独立的 3×256 MLP；初始化训练 300k updates；batch 256；学习率 $3\times10^{-4}$；IQL expectile 0.7；$\beta=3$、$w_{\max}=100$。

这是一项实验组合，不是声称官方 IFQL 原样实现了这个加权 actor。FQL 仓库只作为可复用代码入口；动作块支持与本方法训练目标需要新增。[R-FQL-CODE]

**不能对全部 mixed play 数据做无权 BC 后，假定策略已经学会当前 task2。**进入局部实验前要确认初始策略具有可重复的非零成功率。初始策略不会任务时，先修复初始化，不能解释为几何方法没有保护住能力。

保留多步 flow actor；不要悄悄换成 FQL 的单步提取 actor，再把结果解读成对迭代精细动作能力的检验。

### 04.8 阶段 A：局部配对 rollout

**目的：排除数据变化与 actor 提取误差，直接检验更新方向。**

1. 固定 $\pi_0$、critic 和无条件模型。
2. 用 $\pi_0$ 运行任务，按预先规则选约 32 个状态快照，覆盖接近、接触/搬运与目标邻域，不仅选成功回合。
3. 每个快照采样固定参考动作 $A_0$，构造四组候选。
4. 对每个候选恢复完全相同的快照，执行该候选块，后续全部使用相同 $\pi_0$。
5. 每个快照使用 4 个配对 continuation 随机种子，直到回合结束。
6. 对两个 RMS 候选预算分别报告结果。

快照包括物理状态、warm-start 所需状态、环境任务变量、时间限制计数、随机数状态、必要的控制器/动作队列状态。仅保存 observation 或 qpos 不够。MuJoCo 状态保存入口见 [R-MUJOCO]，完整环境层快照需自行实现与验证。

在正式比较前执行：**恢复同一快照＋相同动作/随机数序列，两次 rollout 的状态和奖励一致。**不通过就停止配对实验。

#### 真实后果指标

$$
\widehat{\Delta G}_{\mathrm{MC}}
=\widehat G(s,A_{\mathrm{new}};\pi_0)-\widehat G(s,A_0;\pi_0).
$$

报告：

| 指标 | 用途 |
|---|---|
| 平均 $\widehat{\Delta G}_{\mathrm{MC}}$ | 实际修改是否改善后果 |
| $\widehat{\Delta G}_{\mathrm{MC}}<0$ 的比例 | 有害更新比例 |
| 预测改善但实际退化的比例 | critic 引导可靠性诊断 |
| 实际 RMS、饱和率、方向夹角 | 确认对照真的幅度匹配 |
| 几何残差变化 | 解释机制，不作为主成功指标 |

Monte Carlo 回报仍是有限样本估计。多个快照若来自同一轨迹也可能相关，不能当作独立训练种子。

**进入下一阶段的定性依据：GEOMETRY 在相同实际幅度下，相比 EUCLIDEAN 更少产生有害更新，并优于 ROTATED；不是仅几何距离下降。**具体 go/no-go 数值门槛尚未指定。

### 04.9 阶段 B：offline-to-online 学习

运行 3 个学习方法×3 个种子×100k 原始环境步，共 9 条训练曲线。REF 作为初始能力评估，不额外训练。

每次从参考/目标 actor 得到 $A_0$，按对应方法生成 $A^\star$，然后用相同条件 FM 目标训练：

$$
\mathcal L_{\mathrm{actor}}
=\mathcal L_{\mathrm{improve}}+\beta_{\mathrm{keep}}\mathcal L_{\mathrm{keep}}.
$$

保留项采用任务相关的原始优势加权 FM 目标，冻结用于其权重的初始评价器；不要换成全部 play 的无权 BC。$\beta_{\mathrm{keep}}$ 的数值尚未在对话中确定。

正常执行完整块时 critic target 为：

$$
y_t=\sum_{j=0}^{3}\gamma^jr_{t+j}
+\gamma^4b_t\min_iQ_{\bar\phi_i}(s_{t+4},A'),
\quad A'\sim\pi_{\bar\theta}(\cdot\mid s_{t+4}).
$$

提前结束使用真实 $\ell$ 与 mask。$A^\star$ 不伪造 TD 转移。

| 配置 | 建议起点 |
|---|---:|
| 原始环境步数/方法/种子 | 100,000 |
| 每个完整动作块的 critic updates | 4 |
| 每个完整动作块的 actor updates | 1 |
| critic batch | 256，半离线半在线 |
| actor batch | 64 |
| critic learning rate | $3\times10^{-4}$ |
| actor learning rate | $10^{-4}$ |
| 原始步折扣 | 0.99；块级用 $\gamma^\ell$ |
| 几何强度 $\kappa$ | 4 |
| 去噪尺度 $\sigma$ | 0.1 |
| 在线 RMS | 从开发阶段选择，正式实验固定 |
| 评估时间点 | 0、20k、40k、60k、80k、100k |
| 每次评估回合 | 50 |

不同方法随学习产生不同 replay 是在线系统效果的一部分；固定数据的因果隔离由阶段 A 提供。所有组记录相同资源口径。

### 04.10 主评估与结论规则

用 $S(\pi_k)$ 表示评估成功率，记录最大退化：

$$
D_{\max}=\max_k[S(\pi_0)-S(\pi_k)]_+.
$$

同时报告成功率学习曲线 AUC、末期平均成功率、回报、交互数和计算成本。

| 观察 | 可支持的解释 |
|---|---|
| GEOMETRY 最大退化较小，AUC 或末期表现也更好 | 对稳定改进有初步支持 |
| GEOMETRY≈ROTATED，二者都比 EUCLIDEAN 好 | 可能是一般预条件/正则化，不足以证明学到的方向有价值 |
| 局部 rollout 更好，更新后的 actor 无收益 | 优先诊断提取误差、共享参数干扰与数据分布变化 |
| 几何残差降低，真实表现下降 | 先验结构与当前控制目标不一致 |
| 所有方法接近满分且无退化 | 当前任务区分能力不足 |
| 初始策略几乎全失败 | “保持已有能力”的实验前提未成立 |

不能只展示最佳 checkpoint；首版不启用方法专属回退来隐藏退化。三个种子仅用于初筛，展示逐种子结果，不以回合数量替代训练种子数。

训练、评估、配对分支 rollout 的环境交互分别计数；几何 Jacobian 的额外计算、wall-clock 时间与推理延迟分别记录。当前没有经过实测的时间或 GPU 预算。

### 04.11 机器可读草案配置

以下 YAML 是**研究配置草案**，不是现成仓库可直接接受的 CLI 参数。`null` 表示对话未确定，不得静默替换成默认值后开始正式实验。

```yaml
study:
  id: unconditional_action_geometry_ogbench_mve
  status: proposed_not_run
  independent_training_seeds: 3
  seed_values: null
  methods: [REF, EUCLIDEAN, GEOMETRY, ROTATED]
  priority: paired_rollout_before_online_training

environment:
  dataset_name: cube-single-play-singletask-task2-v0
  observation_type: states
  observation_dim: null
  compact_dataset: false
  action_dim_expected: 5
  normalized_action_bounds: [-1.0, 1.0]
  primitive_control_dt_expected: 0.05
  success_timing_expected: pre
  repository_commit: null
  installed_package_versions: null
  runtime_max_episode_steps: null

chunking:
  prediction_horizon: 4
  execution_horizon: 4
  flattened_action_dim: 20
  cross_trajectory_windows: false
  short_chunk_and_padding_contract: null
  within_chunk_success_contract: null

models:
  unconditional_flow:
    hidden_dims: [256, 256, 256]
    activation: silu
    frozen_during_online_rl: true
    time_convention: noise_0_data_1
  conditional_flow:
    hidden_dims: [256, 256, 256]
    solver: euler
    sampling_steps: 10
    deployment_policy: multistep_flow
  critics:
    count: 2
    hidden_dims: [256, 256, 256]
    action_guidance: minimum_of_two
  iql_value_network:
    architecture: null

pretraining:
  unconditional_updates: 100000
  actor_critic_updates: 300000
  batch_size: 256
  learning_rate: 0.0003
  iql_expectile: 0.7
  advantage_beta: 3.0
  advantage_weight_cap: 100.0
  actor_loss: advantage_weighted_flow_matching
  initial_policy_acceptance_threshold: null

geometry:
  denoising_sigma: 0.1
  kappa: 4.0
  metric: transpose_I_minus_JD_times_I_minus_JD
  trace_normalization: true
  numerical_epsilon: null
  jacobian_mode: explicit_for_20_dimensions
  rotation: fixed_orthogonal_matrix_per_seed
  local_update_steps: 1
  nonlinear_projection: false
  method_specific_accept_reject: false
  clipping_and_actual_rms_matching_algorithm: null
  rms_matching_tolerance: null

paired_rollout:
  snapshots: 32
  continuation_seeds_per_snapshot: 4
  rms_budgets: [0.02, 0.05]
  continuation_policy: frozen_pi0
  include_unsuccessful_reference_episodes: true
  snapshot_fields_and_restore_test: null
  critic_reference_continuation_alignment: null
  go_no_go_threshold: null

online:
  primitive_environment_steps: 100000
  critic_updates_per_complete_chunk: 4
  actor_updates_per_complete_chunk: 1
  critic_batch_size: 256
  actor_batch_size: 64
  offline_fraction_in_critic_batch: 0.5
  critic_learning_rate: 0.0003
  actor_learning_rate: 0.0001
  primitive_discount: 0.99
  gamma_exponent: actual_executed_length
  actor_keep_loss: initial_advantage_weighted_flow_matching
  keep_loss_weight: null
  rms_budget_selected_before_final_runs: null
  target_network_update_rule: null
  replay_capacity_and_warmup: null
  checkpoint_rollback: false

evaluation:
  checkpoints_primitive_steps: [0, 20000, 40000, 60000, 80000, 100000]
  episodes_per_checkpoint: 50
  primary_metrics: [maximum_success_drop, success_auc, late_success]
  exact_late_checkpoint_aggregation: null
  separate_training_and_evaluation_interaction_counts: true
  report_compute_and_wall_clock: true
```


---

<a id="handoff"></a>
## 05. Codex 交接：实现边界、待决事项与验收

本节把此前讨论转成实现检查项。接口建议与未闭合项属于 `PROPOSAL / OPEN`，不是原论文内容，也不是已存在代码。

### 05.1 首版明确不做什么

不引入图像编码器、VLA、多机器人动作混合、world model、diffusion critic、全 denoising-path 反向传播、额外去噪投影、在线先验更新或方法专属安全回退。

不把最小实验扩成完整 OGBench benchmark。没有局部正面信号前，不扩大任务数、模型规模或在线预算。

不宣称算法稳定、不宣称几何准确，也不填入任何假想成功率。

### 05.2 最小新增模块

```text
chunk_data.py
    原始轨迹与动作块转换、累计奖励、有效长度、bootstrap 信息、执行 wrapper。

uncond_flow.py
    无条件 flow 训练、最终动作去噪器接口、checkpoint 冻结。

geometry_update.py
    去噪器 Jacobian、M 矩阵、线性系统求解、旋转对照、真实 RMS 匹配。

paired_rollout.py
    完整模拟器/环境快照、恢复单测、配对候选执行与 Monte Carlo 结果。

conditional_actor_update
    在现有 actor 中新增“改进动作目标的条件 FM”训练分支。
```

这些是建议模块名，不是声称仓库里已经存在这些文件。

建议在现有 FQL/IQL/flow 代码中复用可用组件，而不是默认重写整个 RL 框架。若使用 FQL 仓库，先核对当前 agent 的 actor 表达方式，避免无意间保留了单步提取策略。

### 05.3 函数级数据契约

以下为接口草图，不是可直接运行的完整程序：

```text
build_chunks(raw_dataset, H, boundary_contract)
    -> observations, actions, returns, next_observations,
       bootstrap_masks, executed_lengths, action_valid_masks,
       episode_ids, primitive_indices

conditional_sample(policy, observation, rng, H, solver_steps)
    -> action_chunk[H, d]

denoise(unconditional_model, noisy_action_flat, sigma)
    -> clean_action_estimate[D_A]

geometry_metric(unconditional_model, action_flat, sigma)
    -> M[D_A, D_A], diagnostic_spectrum, residual

improve_action(critic, observation, base_chunk, method, geometry, budget)
    -> teacher_chunk, actual_delta, diagnostics

snapshot_environment(env, wrapper_state, policy_execution_state, rng_state)
    -> complete_snapshot

restore_environment(env, complete_snapshot)
    -> observation, restored_execution_state

evaluate_paired(snapshot, candidate_chunk, continuation_policy, continuation_rng)
    -> return, success, steps, trajectory_diagnostics
```

`executed_lengths` 与 `action_valid_masks` 的具体尾部约定尚待确定，不可从字段名字自行猜测语义。

### 05.4 先做的单元测试

| 测试 | 通过要求 |
|---|---|
| 动作展平/恢复 | `(H,d) ↔ (H*d,)` 保持时间和坐标顺序 |
| 数据边界 | 任何动作块都不跨原始 episode；训练/验证轨迹不混用 |
| 块级回报 | 与按原始 step 逐项计算的折扣回报一致 |
| 成功与截断 | `masks`、`terminals`、`terminated`、`truncated` 不混淆 |
| FM 时间约定 | `t=0` 是噪声、`t=1` 是数据，target 为 `A-z` |
| 去噪转换 | 输入缩放与时间转换正确，Jacobian 含完整链式因子 |
| 输入导数 | 冻结模型参数后仍能得到非空输入导数；不要误用整体 `no_grad` |
| Jacobian 核验 | 小样本上自动微分与有限差分方向导数相符 |
| 几何代价 | $M=B^\top B$ 对称、半正定至数值精度；线性求解无 NaN |
| 零几何强度 | $\kappa=0$ 时，GEOMETRY 与 EUCLIDEAN 在相同幅度处理后相同 |
| 零梯度 | 不因归一化产生 NaN 或强行非零动作 |
| 旋转对照 | $O^\top O=I$；旋转前后谱一致；O 在每种子中固定 |
| 动作幅度 | 记录并匹配裁剪后的 RMS，而不是只看原始梯度范数 |
| TD 数据真实性 | teacher action 不与未经其执行的旧 reward/next_state 配对 |
| 快照恢复 | 相同完整快照＋动作/随机数序列，轨迹与奖励复现 |
| actor 梯度隔离 | FM 更新不反向修改 critic、teacher target 或无条件模型 |

这些检查项是实现验收，不意味着本次已经运行通过。

### 05.5 尚未闭合的设计项

以下事项是在整理已有方案时显式标出的缺口；没有悄悄替换之前的设计。

| 编号 | 未闭合项 | 为什么重要 | 处理要求 |
|---|---|---|---|
| O01 | 仓库版本与运行环境 | `master`、包版本、成功时间语义可能不同 | 固定 commit、依赖版本，运行 smoke test |
| O02 | 初始条件策略是否会任务 | 没有已有能力就无法测保持 | 保存独立初始评估，不编造接受阈值 |
| O03 | 初始 critic 的 continuation 语义 | IQL 初始化的 Q 不自动等于 $Q^{\pi_0}$ | 明确局部 rollout 是动作比较还是严格 critic 校准；需要时先做明确的参考策略评价 |
| O04 | 块内成功、轨迹尾部和后缀 | 可能污染 TD target 或丢弃关键成功样本 | 先定义长度/padding/mask 契约，再写数据转换 |
| O05 | RMS 匹配与 box clipping | 裁剪改变步长与方向，可能破坏对照 | 固定实现与容差，记录不可达预算情况 |
| O06 | trace 归一化与近零 M | 放大估计噪声可能被误称为几何 | 记录原始谱与 trace；退化阈值需在开发阶段确定 |
| O07 | 去噪器在干净动作上的 Jacobian | 训练噪声尺度与实际查询位置可能不匹配 | 在固定验证集检查局部导数，不仅检查生成质量 |
| O08 | 去噪尺度、步长与强度 | 0.1、0.02/0.05、4 都只是起点 | 用独立开发数据选择，正式实验前固定 |
| O09 | 提取误差的评估口径 | FM 不自动保留初始噪声到目标的逐点配对 | 明确分布级、配对级与闭环评价的区别 |
| O10 | actor/critic target 更新 | EMA 系数和调度未指定 | 从代码与协议中明确，不冒充对话已有设定 |
| O11 | 保留项权重 | 过大阻止改进，过小可能遗忘 | $\beta_{keep}$ 在本对话中未赋值 |
| O12 | replay warm-up 与容量 | 在线初始空 buffer 与采样比例需要处理 | 在共同基础实现中统一 |
| O13 | 统计单位 | 同轨迹快照、同种子回合不独立 | 报告逐种子结果与配对组织方式 |
| O14 | 接触阶段划分与快照抽样 | 人工挑选有利状态可能产生偏差 | 预先固定规则，包含失败参考轨迹 |
| O15 | 首版成功率不等于精密控制 | Cube 容差较宽 | 首轮结论限制为该任务；后续另测精度与恢复 |
| O16 | 新颖性是否成立 | 多个相关方法已有生成先验引导优化 | 当前不宣称新算法首次提出，后续做系统 related-work 比较 |

对 O03 的具体提醒：若 critic 的 target continuation 与配对 rollout 的 $\pi_0$ 不同，“预测改善但 rollout 退化”可能混合了价值误差与后续策略不匹配。首版至少要记录这种区别，而不是将所有差异都叫作 critic 的乐观错误。

### 05.6 建议日志字段

日志格式尚未指定；以下是本研究问题需要保存的最小信息，而不是已经实现的 schema。

```text
run_metadata:
  method, train_seed, software_versions, commits, data_hashes,
  actor_init_hash, critic_init_hash, prior_hash, resolved_config

local_update:
  snapshot_id, episode_id, primitive_index, candidate_id,
  q_before, q_after, gradient_norm, actual_delta_rms,
  box_saturation_fraction, geometry_trace, geometry_spectrum,
  residual_before, residual_after, step_failure_reason

paired_rollout:
  continuation_seed, return, success, rollout_steps,
  paired_reference_return, monte_carlo_delta

online_evaluation:
  primitive_train_steps, critic_updates, actor_updates,
  success_rate, mean_return, evaluation_episode_count,
  train_interactions, evaluation_interactions, diagnostic_interactions,
  wall_clock, geometry_compute_time, inference_latency
```

统计图只展示真实测量，缺少的数据标记缺失；不得填平失败运行或仅输出最佳 checkpoint。

### 05.7 里程碑与交付物

| 里程碑 | 必需交付物 | 进入下一步的条件 |
|---|---|---|
| M0：接口和数据 | 版本记录、数据字段报告、边界/回报测试 | 基础契约清楚 |
| M1：预训练 | 无条件模型与共同 actor/critic checkpoint、初始真实评估 | 初始策略有任务能力，几何算子有稳定方向性 |
| M2：局部诊断 | 配对 rollout 原始记录、真实收益/有害更新图 | GEOMETRY 的收益不是仅由更小步长解释 |
| M3：在线验证 | 9 条训练曲线、逐种子最大退化/AUC/末期表现 | 能分析保持与改善，而非只看最终最好值 |
| M4：后续扩展 | 新任务/条件先验/完整约束方案的独立消融 | 根据首版证据决定，不预先承诺扩大规模 |

### 05.8 可直接交给 Codex 的任务说明

```text
阅读本研究上下文。实现范围以第04节最小实验为准，而不是第03节完整候选算法。

先检查工作区中是否已有 OGBench/FQL/IQL/flow 代码，记录版本与可复用组件。
不要开始完整训练；先给出第05.5节 OPEN 项的处理计划，完成环境加载、
动作块数据语义、Jacobian、RMS匹配和快照恢复测试。

保留普通双 critic 和多步条件 flow actor。
无条件模型只接收相容动作块，不接收状态或任务条件，并在在线阶段冻结。
三种学习方法只在局部动作改进方向上不同：EUCLIDEAN、GEOMETRY、ROTATED。
REF 是冻结初始策略评估基线。

先实现固定策略/critic的同状态配对 rollout，评估真实环境后果。
没有完成局部验证前，不扩大任务数或引入额外保护模块。
teacher actions 只能进入 actor FM 监督，不能伪造成真实 TD 转移。
所有新增默认值都记录在配置和决策日志；尚未确定的项不得静默填充。

不要宣称实验成功或稳定性保证。运行报告区分代码完成、单测通过、
训练完成和研究假设得到支持这四个不同状态。
```


---

<a id="sources"></a>
## 06. 来源、阅读索引与证据边界

### 06.1 主来源

**[P1] 主论文**

*Much Ado About Noising: Dispelling the Myths of Generative Robotic Control*，ICLR 2026 proceedings PDF。

- 来源是用户在本对话给出的链接。
- 本次整理重新读取了相关章节与表格。
- 关键定位：§1、§3.1–3.3、§4.1–4.2、§5.1–5.3；Eqs. 4.2–4.3；Tables 1、2、4、6、9、18；Appendix G.2。
- 第 01 节是该论文及此前对话解读的整理；第 02–05 节的优化算法与实验是后续讨论，不属于主论文贡献。

**[C0] 当前对话**

从用户要求梳理论文动机开始，到无条件动作几何模型、优化算法和 OGBench 最小实验为止的可见问答。没有公共 URL。相关推导按 `DERIVATION` 记录；实验按 `PROPOSAL` 记录；翻译回合已合并。

### 06.2 本次查看的 OGBench 官方材料

| 标识 | 材料 | 支持的内容 | 未完成事项 |
|---|---|---|---|
| [S-OGB] | 官方 README | 加载接口、单任务命名、compact_dataset、masks/terminals、成功时间语义提示 | 未运行下载和环境加载 |
| [S-CUBE] | `cube_env.py` | Cube 任务与成功阈值 | 未验证目标任务实际 rollout |
| [S-MANIP] | `manipspace_env.py` | 5维归一化动作、机器人/夹爪、控制间隔 | 未测运行时维度/性能 |
| [S-REGISTER] | `manipspace/__init__.py` | 单任务注册和 `success_timing='pre'` | 未固定本地安装版本 |

这些链接指向可变分支。实现时应保存 commit/permalink；本文不编造未读取的 commit hash。

### 06.3 与当前方法最相关的阅读入口

以下是**对话中引用的相关工作索引**，本次没有逐篇重新核查。它们提供比较与实现线索，不为当前候选算法背书。2026 年工作尤其需要在正式相关工作部分重读其原文、版本与实验设置。

| 标识 | 对话中的名称/入口 | 应重点核查的问题 |
|---|---|---|
| [R-LANDING] | Landing with the Score: Riemannian Optimization through Denoising | 去噪器、切空间和回缩的近似条件与 Jacobian 要求 |
| [R-MCG] | Improving Diffusion Models for Inverse Problems using Manifold Constraints | 通过去噪器反传任务损失的几何解释；与 RL 的差别 |
| [R-DFLOW] | D-Flow | 固定 flow/diffusion 生成器的潜变量优化及各向异性 |
| [R-DENOISER] | 对话用于后验去噪器导数关系的 arXiv:2410.11149 | 精确后验均值恒等式与学习网络近似的区别 |
| [R-DSRL] | Steering Your Diffusion Policy with Latent Space Reinforcement Learning | 潜空间 RL、noise aliasing、固定生成器的限制 |
| [R-IDQL] | IDQL: Implicit Q-Learning as an Actor-Critic Method with Diffusion Policies | 通过候选生成和价值重加权做 policy extraction |
| [R-DQL] | Diffusion-QL | diffusion actor 与普通 critic 的组合 |
| [R-FQL] | Flow Q-Learning | FM 与单步提取的分工；不可误当多步 actor 的同等对照 |
| [R-QAM] | Q-learning with Adjoint Matching | 最终动作 Q 梯度到中间训练目标的构造 |
| [R-TRQAM] | 对话称为 Trust Region Q Adjoint Matching | denoising-path KL 与最终分布 KL；不能直接推广为物理安全 |
| [R-EXPO] | EXPO: Stable Reinforcement Learning with Expressive Policies | 基础策略、局部动作修正、候选筛选 |
| [R-QCHUNK] | Q-Chunking | 动作序列的价值学习与执行语义 |
| [R-DQCHUNK] | Decoupled Q-Chunking | 评价时域与反应时域如何解耦 |
| [R-DAP] | 对话称为 Denoising as Projection | 去噪作为近似投影；对梯度查询点的假设 |
| [R-TRPO] | Trust Region Policy Optimization | 性能差分关系与信赖域假设 |
| [R-SPIBB] | Safe Policy Improvement with Baseline Bootstrapping | 证据不足时保留基线的形式化条件 |

该表不声称当前方案具备相关论文的定理保证，也不声称以上条目已经完整覆盖最新文献。

### 06.4 其他工具与代码入口

| 标识 | 用途 |
|---|---|
| [R-DAGGER] | 交互式数据聚合与学习者状态分布 |
| [R-HGDAGGER] | 人工门控纠正 |
| [R-DART] | 在示教执行中引入扰动、获得纠正覆盖 |
| [R-CCIL] | 模型辅助的局部纠正样本 |
| [R-AGGREVATE] | 代价敏感的模仿学习 |
| [R-IQL] | 离线价值学习与优势加权提取 |
| [R-RLPD] | 先验数据与在线 replay 的结合 |
| [R-HILSERL] | 人工介入与 off-policy RL |
| [R-DPPO] | diffusion policy 的直接 RL 微调 |
| [R-RECAP] | advantage-conditioned VLA 的经验改进 |
| [R-STABLEBC] | 与误差动力学相关的模仿目标 |
| [R-RTC] | 实时动作块执行与反馈延迟 |
| [R-FQL-CODE] | 建议复用的 FQL/IQL/flow 代码仓库 |
| [R-FQL-IFQL] | 需审查 actor 训练方式的实现文件 |
| [R-FQL-MAIN] | 需审查 offline-to-online/replay 调度的主循环 |
| [R-MUJOCO] | 状态保存/恢复与复现检查的文档入口 |

### 06.5 项目背景资料的定位

此前对话曾使用项目中的 Sergey Levine 资料，主要关联三种思想：

- `2017_Apr_Sergey_Summary.docx`：先在更适合的行为空间改进，再监督投影到神经策略；可见资料 §3.1，原资料行 106–119。
- `2021_Sep_Sergey_Summary.docx`：固定数据的反事实信息与覆盖边界；可见资料 §2.1，原资料行 83–92。
- `2026_Aug_Sergey_Summary.docx`：critic 被优化器主动利用、offline-to-online 退化、固定生成器的限制；可见资料行 171–189、214–224。
- `2019_Jun_Sergey_Summary.docx`：先验作为启动工具，但应保留由真实经验纠正的通道；可见资料行 190–198。

这些是项目背景，不是上述主论文，也不是本实验已经得到结果的证据。原 DOCX 不包含在本 Markdown 包内；这里保留文件名和定位，避免使用离开聊天后不可解析的临时附件标识。

### 06.6 结论与后续使用边界

论文支持重新审视生成策略优势的来源；对话进一步提出将动作几何用于 off-policy 策略改进。但：

```text
论文观察 ≠ 完整因果机制
动作几何 ≠ 状态相关可行性
几何约束 ≠ 物理闭环稳定
双 critic 的 min ≠ 校准的置信下界
教师动作满足约束 ≠ 学生 actor 满足约束
局部更新更好 ≠ 整体在线训练更好
Cube 任务改进 ≠ 毫米级精密操作已验证
研究方案与配置 ≠ 代码已实现或实验已运行
```

当前应首先验证的，是：**相同幅度下，真实学到的几何方向能否比随机旋转方向和普通梯度更可靠地改善物理执行。**


<!-- Portable source references -->
[P1]: https://proceedings.iclr.cc/paper_files/paper/2026/file/9299ca9ed58731945e934adb5b71728c-Paper-Conference.pdf
[S-OGB]: https://github.com/seohongpark/ogbench
[S-CUBE]: https://github.com/seohongpark/ogbench/blob/master/ogbench/manipspace/envs/cube_env.py
[S-MANIP]: https://github.com/seohongpark/ogbench/blob/master/ogbench/manipspace/envs/manipspace_env.py
[S-REGISTER]: https://github.com/seohongpark/ogbench/blob/master/ogbench/manipspace/__init__.py
[R-LANDING]: https://arxiv.org/abs/2509.23357
[R-MCG]: https://arxiv.org/abs/2206.00941
[R-DFLOW]: https://proceedings.mlr.press/v235/ben-hamu24a.html
[R-DENOISER]: https://arxiv.org/abs/2410.11149
[R-DSRL]: https://arxiv.org/abs/2506.15799
[R-IDQL]: https://arxiv.org/abs/2304.10573
[R-DQL]: https://arxiv.org/abs/2208.06193
[R-FQL]: https://arxiv.org/abs/2502.02538
[R-QAM]: https://arxiv.org/abs/2601.14234
[R-TRQAM]: https://arxiv.org/abs/2605.27079
[R-EXPO]: https://arxiv.org/abs/2507.07986
[R-QCHUNK]: https://arxiv.org/abs/2507.07969
[R-DQCHUNK]: https://arxiv.org/abs/2512.10926
[R-DAP]: https://arxiv.org/abs/2608.29507
[R-TRPO]: https://proceedings.mlr.press/v37/schulman15.html
[R-SPIBB]: https://proceedings.mlr.press/v97/laroche19a.html
[R-DAGGER]: https://proceedings.mlr.press/v15/ross11a.html
[R-HGDAGGER]: https://arxiv.org/abs/1810.02890
[R-DART]: https://proceedings.mlr.press/v78/laskey17a.html
[R-CCIL]: https://arxiv.org/abs/2310.12972
[R-AGGREVATE]: https://arxiv.org/abs/1406.5979
[R-IQL]: https://arxiv.org/abs/2110.06169
[R-RLPD]: https://proceedings.mlr.press/v202/ball23a.html
[R-HILSERL]: https://arxiv.org/abs/2410.21845
[R-DPPO]: https://arxiv.org/abs/2409.00588
[R-RECAP]: https://arxiv.org/abs/2511.14759
[R-STABLEBC]: https://arxiv.org/abs/2408.06246
[R-RTC]: https://arxiv.org/abs/2506.07339
[R-FQL-CODE]: https://github.com/seohongpark/fql
[R-FQL-IFQL]: https://github.com/seohongpark/fql/blob/master/agents/ifql.py
[R-FQL-MAIN]: https://github.com/seohongpark/fql/blob/master/main.py
[R-MUJOCO]: https://mujoco.readthedocs.io/en/latest/programming/simulation.html
