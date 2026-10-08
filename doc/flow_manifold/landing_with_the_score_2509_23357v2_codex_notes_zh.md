---
title: "Landing with the Score：核心论点、方法与实现边界"
paper_title: "Landing with the Score: Riemannian Optimization through Denoising"
arxiv_id: "2509.23357"
paper_version: "v2"
paper_version_date: "2026-02-28"
note_date: "2026-09-30"
language: "zh-CN"
document_type: "paper_knowledge_note"
intended_use: "供 Codex 检索、算法设计与实现时参考"
source_html: "https://arxiv.org/html/2509.23357v2"
source_pdf: "https://arxiv.org/pdf/2509.23357v2"
tags:
  - riemannian-optimization
  - diffusion-models
  - denoising
  - score-function
  - manifold-adherence
  - trajectory-optimization
  - diffusion-policy
---

# Landing with the Score：核心论点、方法与实现边界

本文档整理论文 v2 的核心内容，并保留当前讨论中与 diffusion/flow policy、Q 梯度有关的联系。中文解释与英文术语并用，数学符号在全文保持一致。本文档是研究知识笔记，不是官方实现或已验证的 RL 算法。

证据类型：

- **[已知恒等式]**：论文引用并使用的已有数学恒等式。
- **[论文结论]**：论文陈述的算法、定理或实验结果；不表示本文档独立验证了全部证明。
- **[解释]**：为理解论文而给出的推导、直觉和判断。
- **[独立审阅]**：本次讨论发现的技术问题，不是作者声明或官方勘误。
- **[研究扩展]**：将论文用于 RL/策略优化的建议，未经本文实验验证。

## 1. 核心结论速览

**[论文结论]** 去噪模型可以提供隐式数据流形上的两类几何操作：

1. 去噪输出近似最近点投影，把点拉回流形附近。
2. 去噪映射的 Jacobian 近似最近点投影的导数；在流形上，它进一步近似切空间正交投影。

设数据流形为 $\mathcal M$，高斯加噪密度为 $p_\sigma$，则：

$$
D_\sigma(x)=x+\sigma^2\nabla\log p_\sigma(x)
\approx\pi_{\mathcal M}(x),
$$

$$
J_{D_\sigma}(x)=I+\sigma^2\nabla^2\log p_\sigma(x)
\approx J_{\pi_{\mathcal M}}(x).
$$

当 $x\in\mathcal M$ 时：

$$
J_{\pi_{\mathcal M}}(x)=P_{T_x\mathcal M}.
$$

由此构造两种优化算法：

- **DRGD**：用 Jacobian 筛选目标梯度，更新后再次去噪。
- **DLF**：将近似切向目标更新与返回流形的着陆项相加。

论文的主要价值是把去噪统计量与非线性流形几何联系起来，并给出近似可行性和一阶驻点分析。实验是正交矩阵优化和仿真轨迹优化，不是在线 RL、真实机器人精细操作或闭环稳定性验证。

## 2. 研究目标与动机

**[论文结论；正文 §1–2]** 研究问题为：

$$
\min_{x\in\mathcal M} f(x),
$$

其中 $f:\mathbb R^d\to\mathbb R$ 可微，$\mathcal M\subset\mathbb R^d$ 是低维数据流形。只能观察样本 $x_i\sim\mu$，不知道显式约束方程或流形参数化。

经典黎曼优化需要切空间投影、回缩等几何操作；本文试图从数据学习这些操作。

| 场景 | 优化变量 | 优化目标 | 流形的含义 |
| --- | --- | --- | --- |
| 论文控制实验 | 输入输出联合轨迹 $(\mathbf u,\mathbf y)$ | 跟踪误差与控制代价 | 满足动力学的轨迹 |
| 论文矩阵实验 | 矩阵 $X$ | Brockett 代价 | 正交矩阵流形 |
| RL 扩展 | 动作或动作块 $A$ | $-Q(s,A)$ | 数据体现的动作结构；是否等于物理可行集需另行验证 |

**[解释]** 学习行为分布和优化任务目标是不同问题。数据可以只覆盖可行但并不优秀的行为；如果学到了有用的几何，就可能沿可行方向找到比训练样本更好的解。

## 3. 符号与几何基础

| 符号 | 含义 |
| --- | --- |
| $d,k$ | 环境空间维度、流形内在维度 |
| $\mathcal M$ | 数据流形 |
| $\mu$ | 支撑于 $\mathcal M$ 的总体数据分布 |
| $\rho=d\mu/d\operatorname{Vol}_{\mathcal M}$ | 相对于流形体积测度的密度 |
| $\sigma$ | 高斯噪声标准差，不是方差 |
| $p_\sigma$ | 数据分布与 $\mathcal N(0,\sigma^2I)$ 的卷积 |
| $s_\sigma(x)$ | 精确 score：$\nabla_x\log p_\sigma(x)$ |
| $s_\theta(x,\sigma)$ | 学到的 score 网络 |
| $D_\sigma(x)$ | 精确后验均值去噪器 |
| $D_\theta(x)$ | 学习去噪器；对应论文中的 $v(x)$ |
| $J_D(x)$ | 去噪器对输入的 Jacobian |
| $\pi(x)$ | $x$ 到 $\mathcal M$ 的最近点投影 |
| $P_{T_x\mathcal M}$ | 在 $x\in\mathcal M$ 处的切空间正交投影 |
| $\mathcal T(\tau)$ | 距离流形小于 $\tau$ 的管状邻域 |
| $\eta$ | DLF 的着陆系数 |
| $\alpha_k$ | 离散优化步长；论文 DRGD 中记为 $\gamma_k$ |

注意：score $s_\theta$ 与 RL 状态 $s$ 是不同对象；生成过程的噪声时间与 DLF 的优化时间也不是同一变量。

在流形上，普通梯度分解为切向和法向：

$$
\nabla f(x)=P_{T_x\mathcal M}\nabla f(x)+P_{N_x\mathcal M}\nabla f(x).
$$

黎曼梯度为：

$$
\operatorname{grad}_{\mathcal M}f(x)=P_{T_x\mathcal M}\nabla f(x).
$$

理想的最近点回缩更新：

$$
x_{k+1}=\pi\!\left(x_k-\alpha_kP_{T_{x_k}\mathcal M}\nabla f(x_k)\right).
$$

切向更新和回缩承担不同职责。曲面切平面上的有限步长仍会离开曲面，例如单位圆上 $(1,0)$ 的切向更新 $(1,-\alpha)$ 已不在圆上。

## 4. 从去噪到几何：关键公式

### 4.1 Gaussian corruption 与 link function

令：

$$
Y\sim\mu,\qquad X=Y+\sigma Z,\qquad Z\sim\mathcal N(0,I).
$$

于是：

$$
p_\sigma=\mu*\mathcal N(0,\sigma^2I),\qquad
\ell_\sigma(x)=\tfrac12\|x\|^2+\sigma^2\log p_\sigma(x).
$$

### 4.2 后验均值与协方差恒等式

**[已知恒等式；正文 Eq. (4)]**

$$
D_\sigma(x)=\nabla\ell_\sigma(x)
=x+\sigma^2\nabla\log p_\sigma(x)
=\mathbb E[Y\mid X=x].
$$

$$
J_{D_\sigma}(x)=\nabla^2\ell_\sigma(x)
=I+\sigma^2\nabla^2\log p_\sigma(x)
=\sigma^{-2}\operatorname{Cov}(Y\mid X=x).
$$

精确 $J_{D_\sigma}$ 对称且半正定，但有限噪声下不一定满足 $J_D^2=J_D$，因此不能直接当成严格投影矩阵。

### 4.3 小噪声几何极限

**[论文结论；定理 1、推论 2]** 在适当的几何和分布条件下，对一个固定的有效管状邻域，论文陈述：

$$
\sup_{x\in\mathcal T(\tau)}\|D_\sigma(x)-\pi(x)\|
\le K\sigma|\log\sigma|^3,
$$

$$
\sup_{x\in\mathcal T(\tau)}\|J_{D_\sigma}(x)-J_\pi(x)\|
\le K\sigma|\log\sigma|^3.
$$

关于论文条件中缺失的正密度假设，见第 9 节。不能省略该审阅意见后把上述结果视为无条件成立。

**[解释]** 在最近点附近用局部坐标展开后验，后验均值恢复最近点，后验协方差恢复局部几何。证明使用非渐近 Laplace 展开；统一误差界还需要控制流形几何和数据密度。

### 4.4 流形外不能直接替换成正交切投影

**[论文结论；附录 B]** 当 $x\notin\mathcal M$，$J_\pi(x)$ 包含曲率引起的切向缩放；只有 $x\in\mathcal M$ 时它才等于 $P_{T_x\mathcal M}$。

单位球面例子：

$$
\pi(x)=x/\|x\|,\qquad
J_\pi(x)=\|x\|^{-1}(I-uu^\top),\quad u=x/\|x\|.
$$

它去掉法向分量，但当 $\|x\|\ne1$ 时还会缩放切向分量。

## 5. 两种核心算法

以下固定噪声水平 $\sigma>0$，冻结 score 网络参数，定义：

$$
D_\theta(x)=x+\sigma^2s_\theta(x,\sigma).
$$

优化只更新输入 $x$，不要求更新网络参数。若已有合适的 score 网络，可以将其作为推理期几何模型使用；已有模型是否满足所需精度，需要实际检查。

### 5.1 DRGD：Denoising Riemannian Gradient Descent

**[论文结论；Eq. (12)]**

$$
x_{k+1}=D_\theta\!\left(x_k-\alpha_kJ_{D_\theta}(x_k)\nabla f(x_k)\right).
$$

算法分解：

1. 在当前点计算目标梯度 $g_k=\nabla f(x_k)$。
2. 计算 $h_k=J_{D_\theta}(x_k)g_k$，得到近似切向方向。
3. 梯度下降提案 $z_k=x_k-\alpha_kh_k$。
4. 去噪回缩 $x_{k+1}=D_\theta(z_k)$。

有限噪声下最后一步只保证近似回缩，不能声称每一步精确位于真实流形。

### 5.2 DLF：Denoising Landing Flow

**[论文结论；Eq. (8)]**

$$
\dot x=-J_{D_\theta}(x)\nabla f(D_\theta(x))+\eta(D_\theta(x)-x).
$$

- 第一项：在去噪结果上评估目标，用几何导数筛选更新方向。
- 第二项：把当前点拉回流形附近。
- 从流形外实现着陆需要 $\eta>0$。

对于精确去噪器，其 Jacobian 对称，上式为以下函数的梯度流：

$$
F_\sigma^\eta(x)=f(D_\sigma(x))-\eta\sigma^2\log p_\sigma(x).
$$

零噪声且几何精确时，目标项与着陆项分别位于最近点的切空间和法空间，因而：

$$
\frac{d}{dt}\tfrac12\operatorname{dist}(x(t),\mathcal M)^2
=-\eta\operatorname{dist}(x(t),\mathcal M)^2,
$$

$$
\operatorname{dist}(x(t),\mathcal M)
=e^{-\eta t}\operatorname{dist}(x(0),\mathcal M).
$$

该精确连续时间关系不能直接推广成学习模型、有限噪声和任意离散步长下的逐步保证。

### 5.3 两者的区别

| 方面 | DRGD | DLF |
| --- | --- | --- |
| 目标梯度位置 | $x_k$ | $D_\theta(x)$ |
| 方向计算 | $J_D(x_k)\nabla f(x_k)$ | $J_D(x)\nabla f(D(x))$ |
| 可行性修正 | 更新后外层去噪 | 着陆项 $\eta(D(x)-x)$ |
| 着陆系数 | 无单独的 $\eta$ | 需要选择 $\eta$ |
| 数值形式 | 直接离散迭代 | 需要积分或离散化 |

## 6. 理论保证及其适用范围

### 6.1 假设

**[论文结论；§2–5]** 主要条件包括：

- 光滑、嵌入、紧致且无边界的流形；主几何结果要求 $C^3$ 正则性。
- 数据相对于流形体积测度具有 $C^3$ 密度，且支撑集为整个流形。
- 必须注意第 9 节指出的正密度问题。
- 初始化位于最近点唯一、模型近似准确的管状邻域。
- 目标梯度满足所需有界性和 Lipschitz 条件。
- 学习模型同时满足值和 Jacobian 的一致误差界：

$$
\sup_{x\in\mathcal T(\tau)}\|D_\theta(x)-D_\sigma(x)\|\le\epsilon,
$$

$$
\sup_{x\in\mathcal T(\tau)}\|J_{D_\theta}(x)-J_{D_\sigma}(x)\|\le\epsilon.
$$

这不是普通 denoising MSE 小就自动满足的条件。

### 6.2 收敛结论的量级

定义：

$$
\delta=\epsilon+K\sigma|\log\sigma|^3.
$$

| 方法 | 论文报告的主要结论 | 解读限制 |
| --- | --- | --- |
| DLF，定理 3 | 保持在有效邻域；聚点距离流形 $O(\delta)$；投影点的黎曼梯度范数 $O(\delta)$ | 渐近聚点结论；不等于有限时间复杂度或全局最优 |
| DRGD，定理 5 | 回缩后迭代点距离流形 $O(\delta)$；平均梯度范数平方 $O(1/N)+O(\delta)$ | 梯度范数的误差下限通常是 $O(\sqrt\delta)$ |

这里隐藏了与目标平滑性、几何、着陆系数和步长有关的常数。精确的步长条件以论文定理为准，不能把任意学习率视为满足理论。

当 $\epsilon=\sigma=0$ 时，DRGD 恢复经典非凸黎曼梯度下降的平均梯度范数平方 $O(1/N)$ 结论。

## 7. 实验及其证据边界

**[论文结论；§6、附录 H]**

| 实验 | 数据与设置 | 算法/对照 | 观察 |
| --- | --- | --- | --- |
| 正交矩阵 | $O(10)$、$O(20)$；20,000 个矩阵；$f(X)=\operatorname{tr}(AXQX^\top)$ | 学习 DLF、精确几何着陆流、训练集最佳代价 | 学习方法优于训练集最佳值；更小噪声改善几何近似 |
| 双摆 | 50,000 条轨迹；长度 100；固定初态；随机输入 | DRGD，从训练集最佳轨迹初始化 | 在实际仿真回放中改进参考跟踪 |
| 单轮车 | 20,000 条轨迹；长度 100；固定初态；随机输入 | DRGD，从训练集最佳轨迹初始化 | 实际仿真轨迹比训练集最佳轨迹更接近参考 |

控制问题学习的是输入输出联合轨迹流形：

$$
\mathcal M_{\mathrm{IO}}=
\{(\mathbf u,\mathbf y):\mathbf y\text{ 由系统在输入 }\mathbf u\text{ 下生成}\}.
$$

需区分：

$$
f(\mathbf u,\mathbf y)
\quad\text{与}\quad
f(\mathbf u,\mathbf y^{\mathrm{true}}).
$$

$\mathbf y$ 是优化出的输出，$\mathbf y^{\mathrm{true}}$ 是将输入送入真实仿真动力学后得到的输出。两者的差异揭示动力学不一致；双摆实验中途存在明显差距。

实现信息：控制 score 使用时间维度上的 1D U-Net；DRGD 使用固定步长 $0.001$。双摆和主要单轮车实验分别设置 3,000 与 2,500 次迭代，额外单轮车例子使用 4,000 次；停止时目标仍在下降。

**[解释]** 证据尚不覆盖：

- 在线 RL 的 critic 学习、探索或策略蒸馏。
- 真实机器人接触操作与精细动作保留。
- 闭环稳定性、扰动恢复和安全探索。
- 相对于强 learned-dynamics MPC、guided diffusion 等方法的全面比较。
- 实时推理延迟；优化流的时间变量不能当作实际运行时间。

控制轨迹流形在正文被称为非紧致流形，主理论却假设紧致；需要局部化或限制研究区域后才能衔接。

## 8. 实现时最容易出错的地方

### 8.1 JVP 与 VJP 不相同

**[独立审阅；针对 Remark 4]** 论文算法写的是 $J_D(x)g$，而对

$$
\langle D(x),\operatorname{stopgrad}(g)\rangle
$$

进行反向传播得到的是 $J_D(x)^\top g$。

- JVP：$J_Dg$，实现论文展示的更新公式。
- VJP：$J_D^\top g$，实现对复合目标反向传播时的梯度。
- 精确去噪器的 Jacobian 对称，二者相等。
- 普通学习 score 的 Jacobian 不保证对称；不能默认为相等。
- 在论文的一致 Jacobian 近似假设下，两者都逼近同一个对称几何算子，但有限模型下仍不同。

### 8.2 不需要显式构造完整 Hessian/Jacobian

所需量是一个 Jacobian-vector product。可以用自动微分直接计算，不必构造 $d\times d$ 矩阵。这里的微分变量是输入 $x$；冻结模型参数不等于禁止输入梯度。

### 8.3 去噪误差与导数误差不同

**[解释]** 若 $\widehat D(x)=D(x)+\varepsilon\sin(\omega x)$，值误差最多为 $\varepsilon$，导数误差却可达 $\varepsilon\omega$。普通训练损失小不足以认证切空间估计质量。

论文明确将基于非零 $L^2$ score 误差的保证列为未来工作。应把 Jacobian 质量当成需要诊断的独立问题。

### 8.4 噪声越小不一定越好

- **[论文结论]** 对理想总体 score，小噪声改善几何近似。
- **[解释]** 对有限数据/有限模型，噪声过小可能暴露数据稀疏与导数误差。
- **[解释]** 固定经验分布在 $\sigma\to0$ 时，去噪通常趋向最近训练样本；在唯一最近样本区域，Jacobian 趋近零，而不是恢复一个连续流形的切空间。
- **[解释]** 噪声较大时，后验均值可能混合不同模式，使输出偏离真实流形。

因此，实际噪声尺度需要选择，不能仅凭渐近定理设为接近零。

### 8.5 不要混淆流形、密度与稳定性

- 密度高不等于任务价值高。
- 数据流形不自动等于物理可行集或安全集。
- 流形外的 $J_D$ 不是严格正交投影。
- 去噪回缩只是近似回缩。
- 流形贴合性不等于闭环稳定性。
- 得到一阶驻点不等于得到全局最优动作。

## 9. 定理 1 的正密度条件：独立审阅意见

**[独立审阅；定理 1 与附录 E.4]** 论文由“$\rho\in C^3$ 且完整支撑”推断密度具有严格正的下界。这一步不成立：光滑且具有完整支撑集的密度，可以在孤立点为零。

一个可直接检查的例子是单位圆 $\mathcal M=S^1$，令：

$$
Y=(\cos\theta,\sin\theta),\qquad
\rho(\theta)=\frac{1-\cos\theta}{2\pi}.
$$

该密度光滑、支撑整个圆，但在 $\theta=0$ 处为零。对观测点 $x=(1,0)$，角度后验正比于：

$$
(1-\cos\theta)\exp[-(1-\cos\theta)/\sigma^2].
$$

令 $u=\theta/\sigma$，极限分布正比于 $u^2e^{-u^2/2}$，其二阶矩为 3。因此：

$$
\sigma^{-2}\operatorname{Cov}(Y\mid X=x)
\longrightarrow\operatorname{diag}(0,3),
$$

而真实切投影为 $\operatorname{diag}(0,1)$。这说明定理按当前字面条件，Jacobian 极限可以失败。

在紧致流形上，增加 $\rho(p)>0$ 对所有 $p\in\mathcal M$ 成立的条件，可由连续性得到统一正下界。引用论文时应说明这个条件问题；本文档不把该修正视为作者已发布的正式更正，也不声称独立验证了后续全部证明。

## 10. 可供 Codex 实现的算法伪代码

以下是数学伪代码，不是论文官方代码，也未绑定某个自动微分框架。它实现显示公式中的 JVP 版本。所有网络参数冻结，仅保留输入微分能力。

```text
Inputs:
  differentiable scalar objective f(x)
  frozen score network score(x, sigma)
  fixed noise standard deviation sigma > 0
  initial x near the data manifold
  iteration count K and suitable step sizes

D(x) := x + sigma^2 * score(x, sigma)

DRGD:
  for k = 0, ..., K-1:
    g := gradient_x f(x)
    h := JVP(D, input=x, direction=stop_gradient(g))
    z := x - alpha[k] * h
    x := stop_gradient(D(z))
  return x

DLF with explicit Euler discretization:
  for k = 0, ..., K-1:
    p := D(x)
    g := gradient_p f(p)  # gradient with respect to p, not x
    h := JVP(D, input=x, direction=stop_gradient(g))
    dx := -h + eta * (p - x)
    x := stop_gradient(x + dt[k] * dx)
  return x
```

注意事项：

1. `gradient_p f(p)` 不能误实现为已经通过 $D$ 回传的 `gradient_x f(D(x))`，否则会多乘一次 Jacobian。
2. `stop_gradient(g)` 表示把目标梯度作为方向向量，不引入目标 Hessian。
3. 迭代末尾截断计算图适用于推理期求解器；若研究可微优化层，需要另行设计跨迭代求导。
4. 最小化使用负号；最大化 Q 时改为正号。
5. DRGD 需要更新后的外层 $D(z)$，不能只做 Jacobian 筛选。
6. Euler 离散化的步长需要控制；连续流结论不自动覆盖任意 `dt`。
7. 若使用 VJP 版本，必须在算法说明中记录与显示公式的区别，不能静默替换。

## 11. 与 diffusion/flow policy 的关系：研究扩展

本节均为 **[研究扩展]**，不是论文已经验证的 RL 结果。

### 11.1 用几何筛选 Q 梯度

固定状态或历史 $s$，令 $A$ 为动作或动作块，学习条件去噪器：

$$
D_{\psi,\sigma}(A;s)=A+\sigma^2s_\psi(A,\sigma\mid s).
$$

对 $f_s(A)=-Q_\phi(s,A)$，DRGD 的直接改写为：

$$
A_{k+1}=D_{\psi,\sigma}\!\left(
A_k+\alpha_kJ_A D_{\psi,\sigma}(A_k;s)\nabla_AQ_\phi(s,A_k)
\,;s\right).
$$

DLF 的改写为：

$$
\dot A=J_A D_{\psi,\sigma}(A;s)
\nabla_{A'}Q_\phi(s,A')\big|_{A'=D_{\psi,\sigma}(A;s)}
+\eta(D_{\psi,\sigma}(A;s)-A).
$$

Q 可以是普通可微标量 critic，不需要也是 diffusion model。去噪器提供几何，critic 提供改进目标。

精确切投影可去掉 Q 梯度的法向分量，但无法修复流形内部的价值误差、过估计或数据覆盖不足。如果 $A$ 是动作块，critic 必须匹配实际执行的动作块时域和后续价值定义。

### 11.2 无条件动作流形的边界

共享无条件去噪器 $D(A)$ 可表示总体动作先验，但：

$$
A\in\operatorname{supp}p(A)
\not\Rightarrow A\text{ 在当前状态 }s\text{ 下可行或有用}.
$$

跨状态混合可能使动作支撑集接近全维，切向筛选的约束作用因此变弱。状态/历史条件去噪器，或联合状态动作/输入输出轨迹流形，更接近本文控制实验表达的动力学结构。

接触切换、流形交叉、边界、离散模式和厚噪声数据不一定满足光滑单一流形假设。

### 11.3 推理期动作优化与策略参数优化要区分

论文直接支持的是固定去噪器下的推理期优化。可以从现有 diffusion/flow policy 采样动作候选，再做若干次几何约束的 Q 改进。

把改进后的动作作为训练目标蒸馏进 actor，是额外研究步骤。对动作点的可行性分析不自动保证共享参数更新后的整个策略保持可行，蒸馏也会引入误差和模式覆盖变化。

### 11.4 Flow velocity 不能直接当作 score

一般 flow-matching velocity 不是 score，也不是最近点投影。必须依据具体的噪声/插值路径转换成后验均值去噪器，并正确处理坐标缩放，才可使用其输入 Jacobian。本文没有验证一般 flow policy 的直接替换方案。

### 11.5 最小机制对照

建议先固定 critic、数据、去噪模型、初始化分布和总计算预算，比较：

| 方法 | 动作更新 |
| --- | --- |
| 普通 Q 改进 | $A\leftarrow A+\alpha\nabla_AQ$ |
| Q 改进后去噪 | $A\leftarrow D(A+\alpha\nabla_AQ)$ |
| Jacobian 筛选并去噪 | $A\leftarrow D(A+\alpha J_D(A)\nabla_AQ)$ |

这只是研究建议，未选择或声称运行任何具体仿真器。应分别测量真实回报、可执行性/动力学误差、精细任务表现、critic 误差利用程度和延迟。更小的去噪残差不能单独充当真实可行性或闭环稳定性的证明。

## 12. 引用与回查位置

主来源：

- Kharitenko et al., *Landing with the Score: Riemannian Optimization through Denoising*, arXiv:2509.23357v2, 2026-02-28。
- [HTML 正文与附录](https://arxiv.org/html/2509.23357v2)
- [PDF](https://arxiv.org/pdf/2509.23357v2)
- [版本记录](https://arxiv.org/abs/2509.23357v2)

| 内容 | 原文回查位置 |
| --- | --- |
| 问题、动机、与采样的区别 | §1、§1.2 |
| 后验均值与协方差恒等式 | §2.2，Eq. (4) |
| 投影与导数的几何近似 | §3，Theorem 1、Corollary 2 |
| 网络近似假设 | §4，Eq. (7) |
| DLF 与精确目标函数 | §4，Eq. (8)–(11)、Theorem 3 |
| 反向传播实现说明 | §4，Remark 4 |
| DRGD 与收敛界 | §5，Eq. (12)、Theorem 5 |
| 实验 | §6、Appendix H |
| 与 denoising autoencoder 的联系 | Appendix A.1 |
| 与 latent-space optimization 的比较 | Appendix A.2 |
| 流形外的投影导数 | Appendix B |
| 密度正下界的推断 | Appendix E.4 |
| 普通训练误差与理论误差的差距 | §7 的 future work，以及相关讨论 |

## 13. 下游使用时必须保留的事实边界

- 方法的核心是“Jacobian 筛选 + 去噪回缩”，不是单纯添加一个 score 正则项。
- $J_D\to J_\pi$ 是邻域内的结论；$J_\pi=P_T$ 需要点在流形上。
- 训练分布上的平均误差不等于邻域内的一致 Jacobian 误差。
- 论文的几何定理有上述正密度条件问题，不应省略。
- DLF 与 DRGD 的目标梯度位置、误差率含义不同。
- 理论只给近似可行性和一阶驻点保证，不给全局最优或闭环稳定性。
- Q 无需采用生成模型结构；几何筛选不解决全部 critic 误差。
- RL、flow-policy 优化、无条件动作先验及蒸馏方案属于扩展设想，不能写成作者实验证明。
- 本文档整理既有讨论；没有实现或运行论文复现实验、RL 训练或机器人评测。
