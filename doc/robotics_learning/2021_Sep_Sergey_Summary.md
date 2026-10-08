---
id: sergey-2021-09-summary
title: Generalization and the Role of Data in Reinforcement Learning
speaker: Sergey Levine
language: en
document_type: research_talk_summary
source_file: 2021_Sep_Sergey_Summary.docx
source_sha256: 7e8a1909af1fd9138960e2ee56cc9d537592cbf4f8391c3a23b6b6eba343ae2f
period_from_filename: 2021-09
talk_date: null
recording_date: null
video_publication_date: '2021-09-12'
video_url: https://www.youtube.com/watch?v=0_tir2t-mbU
keywords:
- offline reinforcement learning
- data reuse
- distribution shift
- pessimism
- AWAC
- CQL
- MOPO
- COMBO
- D4RL
- offline policy selection
conversion_date: '2026-09-30'
content_scope: Full source notes, with source-derived retrieval overview and normalized Markdown formatting.
verification_scope: Converted against the supplied DOCX only. Original attribution, caveats, and references are
  retained; external sources were not re-verified during conversion.
date_note: 12 September 2021 is identified as the recording upload date, not necessarily the talk delivery date.
---

# Generalization and the Role of Data in Reinforcement Learning

## Retrieval overview

This overview is derived from the supplied summary. The full source notes follow, including their evidence qualifications and editorial interpretations.

**Research goal:** Make past interaction data reusable at scale so reinforcement learning can support broadly useful decision systems.

**Motivation:** Fresh interaction after every policy change is expensive, while policy optimization on fixed data can exploit unsupported, overestimated actions.

**Approach:** Control unsupported optimism through policy constraints, advantage weighting, conservative value learning, and pessimistic model-based learning; evaluate improvement and trajectory stitching.

## Source notes

RESEARCH TALK NOTES

Why reusable experience, distribution shift, and pessimism define the offline-RL scaling problem

**SPEAKER**  Sergey Levine · UC Berkeley

**SOURCE**  ShaRing Colloquium YouTube recording · uploaded 12 September 2021

**DURATION**  1:10:29 including audience Q&amp;A

**FOCUS**  Motivation · offline RL · data reuse · pessimism · evaluation

**VIDEO**  [Watch the talk on YouTube](<https://www.youtube.com/watch?v=0_tir2t-mbU>)

**CENTRAL MOTIVATION.** Modern prediction systems scale by training large models on large, reusable datasets. Decision systems need the same data advantage, but conventional RL repeatedly depends on fresh interaction after each policy change—an unacceptable loop for robots, healthcare, driving, and other costly or risky domains.

**ONE-SENTENCE THESIS.** Offline RL can turn historical interaction into a reusable decision-making resource only if learning is conservative where the dataset lacks support and compositional where reliable trajectory fragments can be stitched together.

### Executive takeaway

**Research goal.** Move reinforcement learning from narrow, closed-world policies toward broadly useful decision systems by making past interaction data reusable at scale.

**Technical obstacle.** A policy does not passively consume predictions: it searches for actions that maximize its learned value. With a fixed dataset, that search can select out-of-distribution actions whose values are wrong, and Bellman bootstrapping can amplify the error without any new experience to correct it.

**Methodological answer.** The talk presents a family of solutions rather than one new algorithm: constrain the policy (AWAC), make the critic pessimistic (CQL), penalize uncertain model rollouts (MOPO), or conservatively value model-generated state-action pairs (COMBO).

**Most important boundary.** No method can recover information absent from the data. Coverage, offline policy selection, and reliable evaluation remain the limits that separate a promising scaling recipe from a dependable real-world workflow.

**EVIDENCE CONVENTION.** Timestamps mark claims and examples stated in the talk. Paper-backed context clarifies formal guarantees or reported results. Editorial synthesis identifies implications without attributing them to the speaker.

### 1. Motivation: give decision learning a reusable data engine

#### 1.1 The scaling recipe from supervised learning

**Talk-backed observation.** Levine begins with an intentionally simple account of modern machine learning: high-capacity models become useful when they can exploit very large datasets. Image recognition, translation, and speech recognition all benefit because a collected dataset can be reused across model revisions and experiments. [\[02:04–02:49\]](<https://www.youtube.com/watch?v=0_tir2t-mbU&t=124s>)

**Editorial synthesis.** The central systems question is therefore not merely whether RL can represent a complex policy. It is whether the whole learning process accumulates knowledge instead of resetting its evidence after every policy update.

#### 1.2 Decision problems are not ordinary prediction problems

**Prediction setting.** A supervised learner receives inputs and labels drawn from a data-generating process. Its prediction normally does not change the label distribution inside the training example.

**Decision setting.** An action changes the next state, the observations that follow, and the future rewards. Supervision is often a delayed objective rather than a correct action for each state, so the learner must reason about counterfactual consequences and long horizons. [\[02:49–05:12\]](<https://www.youtube.com/watch?v=0_tir2t-mbU&t=169s>)

**Illustration.** A traffic predictor becomes a decision system when its route recommendations alter traffic flow. Once predictions influence the world, the learner helps create the distribution on which it will later be judged.

#### 1.3 The gap is breadth, not strategic complexity

**Speaker's contrast.** AlphaGo and Atari agents demonstrate sophisticated strategies inside known rules, but their competence is narrow. The memorable coffee-on-the-Go-board example isolates the missing property: handling events outside the closed-world specification. [\[05:12–06:52\]](<https://www.youtube.com/watch?v=0_tir2t-mbU&t=312s>)

**GENERALIZATION GAP.** The target is not simply a smarter policy for one benchmark. It is breadth across objects, tasks, environments, histories, and exceptional cases—the type of diversity large datasets provide to perception systems.

#### 1.4 Why offline RL is the proposed scaling path

**On-policy bottleneck.** If each policy revision requires a new ImageNet-scale interaction dataset, open-world generalization becomes economically and physically implausible. Ordinary off-policy RL reuses a replay buffer, but still relies on continued collection to repair the current policy's errors. [\[06:52–08:04\]](<https://www.youtube.com/watch?v=0_tir2t-mbU&t=412s>)

**Offline-RL vision.** Collect a fixed dataset once—from people, scripts, prior controllers, random behavior, or mixtures—then train and compare many policies without additional environment access. A small online fine-tuning phase may follow, but prior data should supply most of the coverage and reusable knowledge. [\[08:04–10:44\]](<https://www.youtube.com/watch?v=0_tir2t-mbU&t=484s>)

1. **Collect.** Acquire heterogeneous trajectories once and retain them as a persistent dataset.

2. **Learn.** Optimize a policy from that fixed evidence without querying the real environment during training.

3. **Deploy.** Evaluate or deploy; if the algorithm changes, return to the same dataset rather than recollecting everything.

4. **Optionally refine.** Use limited online data when the application permits it, while preserving the offline dataset as the main knowledge base.

### 2. The problem: policy improvement under fixed evidence

#### 2.1 Formal objective and information boundary

Given a Markov decision process and a fixed dataset collected by an unknown behavior process—possibly a mixture of humans, hand-engineered controllers, and historical policies—learn a policy that maximizes expected discounted return without further interaction during training.

```text
D = {(sᵢ, aᵢ, rᵢ, s′ᵢ)}     and     J(π) = Eπ[Σₜ γᵗ rₜ]
```

**Counterfactual requirement.** The learner must compare actions, including actions that occur rarely or not at all in a particular state. This is exactly where the fixed dataset provides the least direct evidence.

**Coverage boundary.** If the data never contains a successful outcome or any trajectory that reaches a necessary region, no algorithm can infer the missing semantics from reward-free evidence alone. Structural generalization may help, but it is an assumption—not free information.

#### 2.2 Offline RL is not imitation learning

**Imitation learning.** Fits the behavior represented in the dataset. With near-expert demonstrations this can be the right objective, but it usually cannot select better actions from mixed-quality data without additional structure.

**Offline reinforcement learning.** Uses reward and dynamic programming to prefer useful actions, reject poor ones, and potentially combine strong fragments from different trajectories. The desired policy may never have appeared as one complete demonstration.

**STITCHING.** If the dataset contains experience from A→B and B→C, an offline-RL method should be able to learn A→C. This ability to compose reliable fragments is the main reason to seek more than behavior cloning.

#### 2.3 Why standard Q-learning fails offline

```text
Q(s,a) ← r + γ maxₐ′ Q(s′,a′)
```

The Bellman update appears attractive because it can propagate information across trajectories. The failure emerges from the interaction between function approximation, maximization, and a fixed behavior distribution.

1. **Fit on behavior data.** The critic is supervised mainly on state-action pairs sampled from the dataset.

2. **Search for the maximum.** Policy improvement deliberately selects the action with the largest predicted value, not a typical data action.

3. **Select positive error.** Out-of-distribution actions have unconstrained approximation error; maximization systematically favors the most optimistic mistakes.

4. **Bootstrap the mistake.** The inflated value becomes a target for earlier states, so error can accumulate through repeated Bellman backups.

5. **Lose the correction channel.** Unlike online RL, the offline learner cannot execute the suspect action and add corrective evidence to the dataset. [\[18:03–24:39\]](<https://www.youtube.com/watch?v=0_tir2t-mbU&t=1083s>)

#### 2.4 Diagnostic evidence

| **Case** | **Observation** | **Interpretation** |
| --- | --- | --- |
| QT-Opt grasping | ≈580k offline grasp attempts: 87% success; +≈28k on-policy attempts: 96% | A ≈5% data increase cut failure from 13% to 4%; distribution relevance mattered more than raw count. |
| HalfCheetah | 1k and 1M fixed samples both produced return ≈−250 | Adding orders of magnitude more data did not remove the structural distribution-shift failure. |
| Q calibration | Predicted return approached 10⁷ while actual return stayed ≈−250 | The learned critic was not merely noisy; it was catastrophically optimistic on the selected policy. |

Talk examples: [QT-Opt comparison \[15:59\]](<https://www.youtube.com/watch?v=0_tir2t-mbU&t=959s>); [HalfCheetah failure \[18:03\]](<https://www.youtube.com/watch?v=0_tir2t-mbU&t=1083s>). QT-Opt terminology and grasp-attempt count are cross-checked against the primary paper.

**CORE TECHNICAL PROBLEM.** Offline RL must improve beyond the behavior data without trusting unsupported counterfactuals. Too little constraint causes exploitation of model error; too much constraint collapses the method toward imitation.

### 3. Methods: four ways to control unsupported optimism

#### 3.1 Policy constraints: stay near the data support

```text
maximizeπ  E[Q(s,a)]     subject to     D(π(·|s), β(·|s)) ≤ ε
```

**Mechanism.** Restrict the learned policy so that its action distribution remains close to the unknown behavior distribution β. This prevents the critic from being queried arbitrarily far outside its training support. [\[24:39–27:39\]](<https://www.youtube.com/watch?v=0_tir2t-mbU&t=1479s>)

**Limit.** Real datasets may mix humans, controllers, and several historical policies. Explicitly fitting one behavior model can be difficult, and matching the full mixture may preserve bad actions that the learner should discard.

#### 3.2 AWAC: implicit constraint through advantage weighting

```text
π*(a|s) ∝ β(a|s) exp(A(s,a)/λ)
```

```text
maximizeπ  E(s,a)~D [ exp(A(s,a)/λ) · log π(a|s) ]
```

**Method.** AWAC samples actions directly from the dataset and performs weighted maximum-likelihood policy learning. High-advantage actions receive larger weights; low-advantage actions are downweighted. The critic still uses off-policy temporal-difference learning. [\[27:39–31:53\]](<https://www.youtube.com/watch?v=0_tir2t-mbU&t=1659s>)

**Why it helps.** The actor never needs an explicit density model for the mixed behavior policy. The update is conservative by construction because it regresses only toward observed actions, yet it selects among them using long-horizon value.

**Best fit.** Offline pretraining followed by online fine-tuning. The paper demonstrates real valve rotation, drawer opening, and dexterous manipulation, and the talk shows a case improving from roughly 24% after offline learning to 88% after fine-tuning.

**Boundary.** AWAC cannot invent actions absent from the data support; λ controls how selective the weighting is; and the talk explicitly frames it as particularly useful for offline-to-online learning, not as the strongest pure-offline method.

#### 3.3 CQL: make the critic pessimistic

```text
Bellman loss + α( Eₛ,ₐ~μ[Q(s,a)] − E(s,a)~D[Q(s,a)] )
```

**Method.** Conservative Q-Learning adds a regularizer that lowers values for actions the optimizer might select and raises values for actions actually present in the dataset, while retaining the Bellman objective. [\[31:53–41:47\]](<https://www.youtube.com/watch?v=0_tir2t-mbU&t=1913s>)

**Interpretation.** Supported high-value actions receive competing upward and downward forces; unsupported high-value actions receive mainly the downward penalty. The critic therefore stops advertising unknown actions as free improvements.

**Guarantee, stated carefully.** Under the paper's assumptions, practical CQL variants can produce a conservative estimate of policy value in expectation. This should not be rewritten as a universal pointwise lower bound for every state-action pair in arbitrary deep-network settings.

**Boundary.** The regularization coefficient determines how pessimistic the method becomes. Excess conservatism can suppress useful generalization, and the optimism that is harmful offline can be helpful for exploration online. Levine also notes that CQL illustrates the principle rather than representing a permanent final state of the fast-moving field.

#### 3.4 MOPO: pessimism through model uncertainty

```text
r̃(s,a) = r̂(s,a) − λ u(s,a)
```

**Method.** Learn a dynamics model from the fixed dataset, start short synthetic rollouts from real data states, and subtract an uncertainty penalty from predicted reward. Policy optimization then prefers regions where the learned model is trusted. [\[41:47–49:37\]](<https://www.youtube.com/watch?v=0_tir2t-mbU&t=2507s>)

**Why model-based methods are attractive.** A model can answer local what-if questions and bridge trajectory fragments, potentially supporting stronger stitching than a policy constrained entirely to observed actions.

**Boundary.** The guarantee depends on u(s,a) behaving like an upper bound on model error. In practice, ensemble disagreement is a useful proxy, not a universal certificate. High-dimensional visual dynamics and uncalibrated uncertainty remain difficult.

#### 3.5 COMBO: conservative values on model-generated data

**Method.** COMBO learns a model and generates rollouts, but it applies a CQL-like value penalty to state-action pairs from the model-rollout distribution while comparatively increasing values on real dataset pairs. It avoids requiring an explicit model-uncertainty score. [\[49:37–50:57\]](<https://www.youtube.com/watch?v=0_tir2t-mbU&t=2977s>)

**Interpretation.** When model rollouts remain close to real data, the conservative penalty can be mild. When the model generates implausible states, those state-action pairs receive low value. The talk uses a GAN-like intuition, but COMBO is not literally trained with a discriminator.

**Boundary.** COMBO still depends on the learned dynamics model and on several conservatism and data-mixture choices. Avoiding explicit uncertainty estimation does not eliminate model bias.

#### 3.6 Unified method map

| **Family** | **Primary target** | **Core mechanism** | **Main limitation** |
| --- | --- | --- | --- |
| Policy constraint | Actor distribution | Keep π close to β or to dataset support | Can preserve bad behavior; β may be unknown/multimodal |
| AWAC | Actor update | Advantage-weighted regression on observed actions | Best suited to offline-to-online; support-limited |
| CQL | Critic values | Penalize high values on unsupported candidate actions | α can cause excessive pessimism; weak for exploration |
| MOPO | Model rollouts | Subtract uncertainty penalty from model reward | Requires reliable model-error proxy |
| COMBO | Model-rollout values | Lower Q on generated distribution vs real data | Still model-dependent; several trade-off coefficients |

Editorial synthesis of the method spectrum presented in the talk; formal details are cross-checked against the AWAC, CQL, MOPO, and COMBO papers listed in Section 9.

**UNIFYING PRINCIPLE.** These approaches negotiate the same dilemma: improve beyond the behavior policy, but attach a cost to action-state regions where the fixed dataset cannot support a trustworthy counterfactual estimate.

### 4. Evidence: what the demonstrations establish—and what they do not

| **System** | **Talk/paper evidence** | **Correct reading** |
| --- | --- | --- |
| QT-Opt | Large-scale real-world grasping; offline 87%, after small online addition 96% | Shows value of data reuse and the remaining importance of on-policy correction; not proof of general manipulation. |
| AWAC | Offline initialization plus rapid online fine-tuning on simulated and real robot tasks | Supports the workflow it targets; results are task- and dataset-specific. |
| CQL | Paper reports strong D4RL and Atari results, often 2–5× prior return on selected settings | Benchmark-era evidence for critic pessimism, not a universal performance guarantee. |
| MOPO | Benchmark gains and cross-task continuous-control problems requiring model generalization | Supports pessimistic model rollouts when uncertainty is informative. |
| COMBO | Paper reports top scores on 9 of 12 D4RL MuJoCo datasets at publication | Evidence that model-based pessimism can work without explicit uncertainty estimation. |

The talk's scientific contribution is primarily explanatory and agenda-setting. It does not introduce all of these algorithms for the first time, and it does not claim that one method has solved offline RL. QT-Opt dates to 2018; BEAR to 2019; D4RL, AWAC, CQL, and MOPO to 2020; COMBO was the most recent method discussed, appearing in 2021.

**WHAT IS NOVEL IN THE TALK.** The synthesis recasts generalization as a data-reuse problem and organizes several algorithms by how they impose pessimism. The new value is the scaling argument and method taxonomy, not a single newly announced algorithm.

### 5. Evaluation: benchmarks must reward improvement, not imitation

#### 5.1 Why common offline benchmarks can be misleading

**Weak protocol.** Train a nearly optimal online policy, record its data, and ask offline RL to recover the same policy. The dataset has little room for improvement and does not resemble mixed historical logs from real systems. [\[51:00–52:33\]](<https://www.youtube.com/watch?v=0_tir2t-mbU&t=3060s>)

**Better protocol.** Use heterogeneous, suboptimal, multimodal data from people, controllers, and policy mixtures. Ensure that the task rewards selection among good and bad actions and composition across partial trajectories.

#### 5.2 D4RL's intended stress tests

- **Heterogeneity.** Mixtures of policies and collection processes rather than one clean behavior policy.

- **Human and scripted data.** Non-Markov behavior, hand-designed controllers, and demonstrations that resemble operational logs.

- **Suboptimality.** Enough poor behavior to require value-based selection instead of pure cloning.

- **Sparse or delayed goals.** Tasks where reward propagation and long-horizon reasoning matter.

- **Stitching.** Partial paths must be composed into a policy better than any complete recorded trajectory.

- **Distribution shift.** Evaluation should reveal whether the learned policy exploits unsupported estimates. [\[52:33–54:05\]](<https://www.youtube.com/watch?v=0_tir2t-mbU&t=3153s>)

**Practical criterion.** A convincing result should compare the learned policy not only with prior algorithms but also with behavior cloning, the behavior policy, and the best trajectory in the dataset. Otherwise the method may simply reproduce logged behavior.

### 6. Open problems and limits

**LARGEST PRACTICAL BOTTLENECK.** Offline policy selection remains unreliable. Without real rollouts, practitioners lack a supervised-learning-style train/validation/test workflow for choosing algorithms, hyperparameters, pessimism strength, or stopping time. [\[54:20–55:43\]](<https://www.youtube.com/watch?v=0_tir2t-mbU&t=3260s>)

| **Open problem** | **Question** | **Why it matters** |
| --- | --- | --- |
| Offline model selection | Which learned policy is actually best when no environment rollout is available? | Needed for safe tuning, early stopping, and reproducible deployment. |
| Dataset composition | What ratio of expert, non-expert, failures, and diverse controllers is best? | Coverage and comparative actions determine what value learning can identify. |
| Minimum success coverage | How many successful or near-success trajectories are required under sparse reward? | If success never appears, the task may be unidentifiable from the log. |
| Data prioritization | Should training emphasize hard negatives, rare transitions, or high-uncertainty regions? | The talk calls this promising but provides no established method. |
| Calibrated pessimism | How conservative should values or model rollouts be? | Too little causes exploitation; too much prevents improvement. |
| Model reliability | Can uncertainty estimates detect model hallucinations in high-dimensional observations? | MOPO-style guarantees depend on error bounds that practical ensembles only approximate. |
| Offline-to-online transition | How should a pessimistic offline policy become exploratory once interaction is allowed? | The objectives that stabilize offline learning can suppress online exploration. |
| Safety | Does conservative value estimation imply safe behavior under deployment shift? | No: reward misspecification, rare hazards, and partial observability remain separate risks. |

Talk Q&amp;A anchors: data prioritization \[59:23\], dataset composition \[60:53\], offline RL vs imitation \[61:39\], data scale \[62:26\], model-free vs model-based \[67:17\].

#### 6.1 What the talk does not establish

- **Any static dataset is sufficient.** Offline RL still needs reward information, coverage, and assumptions that support generalization.

- **Pessimism equals safety.** A lower value estimate does not certify physical safety, ethical acceptability, or robustness to unmodeled hazards.

- **More data automatically fixes Q-learning.** The HalfCheetah example shows that support mismatch can persist even after a thousand-fold increase in fixed data.

- **Model-based is always superior.** Levine's Q&amp;A makes the choice task-dependent: predicting high-dimensional visual dynamics may be harder than learning good actions.

- **Offline RL always beats imitation.** For near-expert data, the error trade-off is not obvious and was identified as an open research question.

### 7. Research synthesis for general robot learning

#### 7.1 The research objective

**GOAL.** Build decision-learning systems whose competence grows with accumulated experience across policies, tasks, and deployments—so each new robot or objective can exploit a shared history instead of beginning with fresh exploration.

Offline RL is the data-reuse layer of this agenda. It explains how reward-labeled interaction might become a persistent asset, much as web-scale corpora became persistent assets for language and vision. It is necessary for a generalist scaling path, but not sufficient: task semantics, representation learning, cross-embodiment transfer, data governance, and safe deployment are additional system problems.

#### 7.2 What is solved versus still missing

| **Capability** | **State by the 2021 talk** | **Remaining boundary** |
| --- | --- | --- |
| Reusable fixed datasets | Established problem setting, benchmarks, and practical algorithms | Logs can train policies without immediate interaction. |
| Control of OOD optimism | Several effective families: support constraints, conservative critics, pessimistic models | Failure is mitigated, not universally eliminated. |
| Trajectory stitching | Supported in principle by dynamic programming and model-based rollouts | Depends on coverage and reliable value/model generalization. |
| Offline-to-online improvement | AWAC and related methods demonstrate practical fine-tuning | Exploration and stability across deployment shifts remain difficult. |
| Policy selection without rollout | Unsolved general workflow | Blocks trustworthy tuning in costly or safety-critical environments. |
| Dataset design | Useful intuitions—diversity, good and bad actions, broad coverage | No universal rule for scale, expert ratio, or prioritized sampling. |

#### 7.3 Highest-value research opportunity implied by the talk

**RESEARCH OPPORTUNITY.** Develop deployment-free offline policy selection with calibrated uncertainty: rank candidate policies and hyperparameters using only logged data, identify when the ranking is unreliable, and connect that signal to a minimal, risk-aware online data request.

This problem is high leverage because every offline-RL algorithm needs it, success can be measured independently of inventing a new policy optimizer, and it directly closes the workflow gap Levine identifies. A strong solution would combine off-policy evaluation, uncertainty calibration, dataset-coverage diagnostics, and a protocol for abstaining when the log cannot support a reliable decision.

#### 7.4 Bottom line

**BOTTOM LINE.** Levine is not arguing that an algorithm can find an optimal policy from arbitrary static data. He is arguing that scalable decision learning requires persistent datasets—and that the defining technical problem is controlling optimistic counterfactual error while still extracting and composing every reliable fragment the dataset contains.

### 8. Timestamped map of the talk

| **Time** | **Block** | **Why it matters** |
| --- | --- | --- |
| 00:00–01:37 | Host introduction | Speaker and topic framing. |
| 01:37–02:04 | Opening | Learning algorithms viewed through data utilization. |
| 02:04–05:12 | Modern ML and decisions | Large data + large models; actions change future inputs. |
| 05:12–06:52 | Closed-world RL | Strategic complexity is not open-world breadth. |
| 06:52–09:35 | Offline-RL motivation | Why repeated on-policy data collection does not scale. |
| 09:35–11:35 | Ideal workflow | Collect once, train repeatedly, deploy, optionally fine-tune. |
| 11:35–15:59 | Q-learning background | Bellman backup and off-policy training setup. |
| 15:59–18:03 | QT-Opt evidence | Offline 87% vs offline + small online phase 96%. |
| 18:03–24:39 | Failure diagnosis | Distribution shift, OOD actions, and bootstrapping error. |
| 24:39–27:39 | Policy constraints | Stay near data support; difficulty of behavior modeling. |
| 27:39–31:53 | AWAC | Advantage-weighted regression and offline-to-online learning. |
| 31:53–41:47 | CQL | Conservative critic and lower-bound intuition. |
| 41:47–49:37 | MOPO | Model rollouts with uncertainty-penalized reward. |
| 49:37–50:57 | COMBO | Conservative Q-values for model-generated distributions. |
| 51:00–54:05 | Benchmarks and D4RL | Heterogeneous data, improvement room, and stitching. |
| 54:05–55:43 | Conclusion | Open workflow and evaluation problems. |
| 55:43–70:29 | Audience Q&amp;A | Sparse rewards, data design, imitation, model choice, and stitching. |

### 9. Selected primary sources

The talk is the primary source for narrative structure and timestamps. The papers below are used to correct caption terminology, state guarantees carefully, and distinguish talk examples from paper-reported evidence.

- [**Original YouTube talk**](<https://www.youtube.com/watch?v=0_tir2t-mbU>) — Sergey Levine, 2021; 1:10:29 including Q&amp;A.

- [**Offline Reinforcement Learning: Tutorial, Review, and Perspectives on Open Problems**](<https://arxiv.org/abs/2005.01643>) — Levine, Kumar, Tucker, and Fu; conceptual overview and open problems.

- [**Stabilizing Off-Policy Q-Learning via Bootstrapping Error Reduction (BEAR)**](<https://arxiv.org/abs/1906.00949>) — Bootstrapping-error diagnosis and support-constrained action selection.

- [**AWAC: Accelerating Online Reinforcement Learning with Offline Datasets**](<https://arxiv.org/abs/2006.09359>) — Advantage-weighted actor update for offline-to-online learning.

- [**Conservative Q-Learning for Offline Reinforcement Learning**](<https://proceedings.neurips.cc/paper/2020/hash/0d2b2061826a5df3221116a5085a6052-Abstract.html>) — Critic pessimism and conservative policy-value estimates.

- [**MOPO: Model-Based Offline Policy Optimization**](<https://proceedings.neurips.cc/paper/2020/hash/a322852ce0df73e204b7e67cbbef0d0a-Abstract.html>) — Uncertainty-penalized model rollouts.

- [**COMBO: Conservative Offline Model-Based Policy Optimization**](<https://arxiv.org/abs/2102.08363>) — Conservative value learning over model-generated state-action pairs.

- [**D4RL: Datasets for Deep Data-Driven Reinforcement Learning**](<https://arxiv.org/abs/2004.07219>) — Offline benchmarks emphasizing heterogeneous and realistic collection processes.

- [**QT-Opt: Scalable Deep Reinforcement Learning for Vision-Based Robotic Manipulation**](<https://arxiv.org/abs/1806.10293>) — Large-scale real-world grasp data and closed-loop grasping evidence.

#### Editorial note

These are independent analytical notes, not a verbatim transcript. The complete public talk was reviewed using its English auto-generated caption track; obvious caption errors in names and method acronyms were corrected against the primary papers. Numerical claims are labeled as talk examples or paper-reported results and should be interpreted in their 2021 benchmark context.
