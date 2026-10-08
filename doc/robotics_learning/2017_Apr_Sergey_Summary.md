---
id: sergey-2017-04-summary
title: Deep Robotic Learning
speaker: Sergey Levine
language: en
document_type: research_talk_summary
source_file: 2017_Apr_Sergey_Summary.docx
source_sha256: a8b71413038170e2edcc8d0d9aecde2aa056dae25a5603458d7e7e4f27613cc6
period_from_filename: 2017-04
talk_date: '2017-04-07'
recording_date: null
video_publication_date: null
video_url: https://www.youtube.com/watch?v=eKaYnXQUb2g
keywords:
- robot learning
- MDGPS
- guided policy search
- sample efficiency
- self-supervised grasping
- visual prediction
- cross-embodiment transfer
- meta-learning
conversion_date: '2026-09-30'
content_scope: Full source notes, with source-derived retrieval overview and normalized Markdown formatting.
verification_scope: Converted against the supplied DOCX only. Original attribution, caveats, and references are
  retained; external sources were not re-verified during conversion.
---

# Deep Robotic Learning

## Retrieval overview

This overview is derived from the supplied summary. The full source notes follow, including their evidence qualifications and editorial interpretations.

**Research goal:** Build robot-learning systems that learn efficiently from interaction, reuse experience across tasks and bodies, accept flexible goals, and adapt rapidly.

**Motivation:** Task-specific mastery and locally accurate pipeline components do not ensure robust task execution or generalization to new conditions.

**Approach:** Combine task-aligned end-to-end learning, mirror-descent guided policy search, hybrid model-based and model-free optimization, diverse experience, predictive representations, and rapid adaptation.

## Source notes

**RESEARCH SEMINAR NOTES**

From task-specific mastery to robots that generalize and adapt

**Speaker:** Sergey Levine

**Venue:** Robotics Institute Seminar, Carnegie Mellon University

**Date:** 7 April 2017

**Video:** Deep Robotic Learning (YouTube ID: eKaYnXQUb2g)

**Focus:** Research motivation, system-level argument, and broadly transferable insights

**RESEARCH GOAL**  Build robot-learning systems that do more than master a single, carefully instrumented task: they should learn efficiently from interaction, reuse experience across tasks and bodies, accept flexible goals, and adapt rapidly to new conditions.

**CENTRAL THESIS**  General robot learning is not achieved by making one policy larger. It requires an entire learning system: task-aligned objectives, efficient optimization, diverse experience, reusable predictive or behavioral structure, and evaluation on held-out adaptation rather than training-task performance alone.

### **Executive summary**

1. The motivating failure is brittleness. Classical pipelines decompose perception, estimation, planning, and control behind narrow interfaces. Each local approximation can look reasonable, yet errors compound at execution time.

2. End-to-end learning is valuable because representations and actions can co-adapt to the downstream task. The point is not to eliminate structure; it is to let the final objective shape the interfaces that matter.

3. Deep reinforcement learning supplies a framework for learning without direct action labels and for assigning value to actions with delayed consequences. Its 2017 bottlenecks are sample complexity and a culture of evaluating mastery rather than generalization.

4. Levine's response has two tracks: make policy improvement dramatically more data-efficient through guided, hybrid optimization; and scale the diversity of experience through parallel robots, self-supervision, predictive models, flexible goals, and transfer across embodiments.

5. The longer-term idea is fast adaptation. Instead of seeking one fixed policy for every situation, train parameters that can become useful for a new task after one or a few updates.

**EVIDENCE CONVENTION**  Sections labeled Talk-backed observation summarize claims, examples, or caveats stated in the seminar. Sections labeled Editorial synthesis generalize those claims beyond the specific experiments. Numerical results are reported as presented in the talk and cross-checked against the associated papers where possible.

### **1. Research goal and motivation**

#### **1.1 The gap: impressive robots, fragile intelligence**

Levine opens with a contrast from the DARPA Robotics Challenge: a winning robot moves through a door slowly and deliberately, while a human can perform the same basic operation fluidly even with severely reduced vision. The comparison is not about hardware speed. It exposes the cost of brittle internal assumptions: when perception, modeling, and control are only loosely compatible, the system must move slowly enough that every module remains inside its comfort zone.

**TALK-BACKED OBSERVATION**  The robotics pipeline is often built as pixels → state estimate → model/prediction → plan → low-level control. Its interfaces are “leaky abstractions”: upstream uncertainty and downstream contact dynamics do not fit neatly into the variables exchanged between modules.

#### **1.2 Why end-to-end training is attractive**

The successful shift in computer vision was not merely from shallow to deep models. It was from hand-designed intermediate features to representations optimized jointly for the final task. In robotics, the analogous opportunity is to let the task loss shape perception and action together. This can reveal control-relevant shortcuts that a proxy objective—such as globally accurate pose estimation—would suppress.

A useful intuition is the gaze heuristic for catching a ball: a person need not reconstruct the full trajectory in world coordinates. Keeping the ball at a stable visual angle can directly couple observation and action. Such shortcuts are not inferior intelligence; they are often exactly what robust control requires.

#### **1.3 Why robotics is harder than passive perception**

- There is no natural label for the correct motor command at every image.

- Actions change future observations, so data are interactive rather than independently sampled.

- Consequences can be delayed; a locally plausible action may create failure much later.

- The policy changes the distribution of states it encounters, producing a moving train/test boundary.

Reinforcement learning addresses these issues by evaluating behavior through outcomes rather than direct action labels. But this creates a new research problem: how to make outcome-driven learning efficient enough for real robots and broad enough to generalize.

#### **1.4 The insertion experiment: optimize the real objective**

A shape-insertion experiment makes the argument concrete. A modular system first estimated the hole location and then executed a controller. Its visual estimate was accurate to roughly 1.5 cm—respectable for localization, but useless when insertion clearance was only a few millimeters. The end-to-end policy achieved 96.3% success, while the modular baseline achieved 0% in the reported setup.

| **Design** | **Training objective** | **Reported outcome** | **What the task needed** |
| --- | --- | --- | --- |
| Modular baseline | Estimate hole pose, then control | 0% insertion success | Millimeter-level, control-relevant precision |
| End-to-end policy | Maximize insertion success | 96.3% success | Ignore irrelevant vertical error; exploit sliding/contact |

**EDITORIAL SYNTHESIS**  Intermediate accuracy is not a reliable proxy for system utility. A representation may be globally imprecise yet excellent for the decisions the controller must make; conversely, a strong benchmark score can be operationally useless if it is accurate in the wrong directions.

**EVIDENCE BOUNDARY**  The 96.3% versus 0% result compares the learned policy with a particular pose-prediction pipeline, not with every modular architecture. Training also used guided policy search, task-specific costs, privileged state for local controllers, pose pretraining, and a hand-chosen spatial architecture. “End-to-end” describes task-aligned co-training and the deployed perception-to-action mapping; it does not mean assumption-free learning.

### **2. The talk's research map**

The seminar moves from a diagnosis of brittle pipelines to two concrete bottlenecks—sample efficiency and generalization—and then to a proposal for fast adaptation. The Q&amp;A broadens the frame to hierarchy, simulation, active data collection, safety, and infrastructure.

| **Time** | **Block** | **Research question** |
| --- | --- | --- |
| 00:31–14:58 | Motivation and end-to-end control | Why do modular systems fail, and what does task-level co-adaptation buy? |
| 14:58–31:20 | Sample-efficient deep RL | How can real robots improve policies with orders of magnitude fewer interactions? |
| 31:20–42:25 | Data and generalization | What kind of diverse, self-supervised experience can support reuse? |
| 42:25–49:02 | Fast adaptation / MAML | Can a learned initialization become useful after one or a few updates? |
| 49:16–60:43 | Q&amp;A and open problems | How should structure, hierarchy, simulation, safety, and data systems fit together? |

### **3. Sample efficiency: guided improvement, then projection**

#### **3.1 The geometric idea behind MDGPS**

A neural policy is a parameter vector, but it also defines a probability distribution over trajectories. Levine's key move is to optimize in the richer space of trajectory distributions, where local control tools are effective, and then project the improved behavior back into the neural policy through supervised learning. The resulting method is mirror-descent guided policy search (MDGPS).

The policy-improvement step is constrained by a KL-divergence trust region, so each update remains close enough to the current behavior for local approximations to be informative. Multiple time-varying linear-Gaussian controllers are trained from sampled initial conditions; their improved actions become targets for the global policy network.

**GENERAL PRINCIPLE**  When direct optimization in parameter space is difficult, optimize behavior in a space with better local structure, then distill or project the result into the deployable model. The projection is not an afterthought—it is the bridge between specialized local improvement and reusable global behavior.

#### **3.2 Why hybrid model-based/model-free updates help**

Locally fitted dynamics models are data-efficient in smooth regions, but contact and discontinuities produce systematic model error. Path-integral policy improvement is more robust to those discontinuities, but wastes samples when an accurate local model is available. PILQR combines the two: make most of the update with the local model, then use a model-free residual correction to emphasize trajectories that perform better than the model predicted.

| **Update** | **Strength** | **Characteristic failure** | **Hybrid role** |
| --- | --- | --- | --- |
| Model-based | Fast local improvement | Breaks at contact/discontinuity | Handle smooth, predictable motion |
| Model-free | Robust to model mismatch | High variance and more samples | Correct the residual where the model is wrong |
| PILQR hybrid | Complementary error modes | Still local and task-specific | Allocate each method to the regime it handles best |

#### **3.3 Evidence presented in the seminar**

- In a simulated door-opening example, a model-free update reached toward the door but struggled to acquire the handle; a model-based update reached contact but failed around the discontinuity; the hybrid completed the task.

- A real hockey-striking policy was learned from scratch with roughly 40 minutes of actual robot motion, excluding computation time.

- On a simulated reacher benchmark, MDGPS was reported at roughly 10× the sample efficiency of DDPG and roughly 100× that of policy-gradient/TRPO-style baselines; the comparison is benchmark-specific and shown on a log scale.

- An asynchronous off-policy continuous Q-learning system trained a door-opening behavior on two robots in about 2.5 hours wall-clock time, corresponding to five robot-hours.

- In a related four-robot collective-learning study, robots improved local controllers in parallel, pooled supervision, and trained one global visual policy that averaged about 90% success on a held-out test door under the reported pose grid.

**EVIDENCE BOUNDARY**  These results establish feasibility and large gains on the reported tasks; they do not prove universal sample-efficiency rankings. The 2.5-hour experiment used a fixed door, joint/end-effector state plus a door-mounted IMU, a dense privileged reward, and engineered safety constraints. The visual multi-door study used kinesthetic initialization, IMU-defined costs, pose pretraining, and frozen visual layers; camera shifts reduced success sharply. Parallel robots improve wall-clock utilization and coverage, not automatically the number of interactions required. The strongest transferable claim is the design pattern: exploit trustworthy local structure, correct residual failures from data, and parallelize collection when hardware permits.

### **4. Generalization requires diverse experience**

#### **4.1 The missing ingredient is not only an algorithm**

Levine frames supervised learning's progress as the conjunction of compute, algorithms, and large varied datasets. Robotics has access to compute and improving models, but cannot avoid the experience problem. If a system is trained on one narrow task in one narrow environment, there is no reason to expect reusable behavior—regardless of model size.

#### **4.2 Self-supervised grasping at scale**

The grasping system used six to fourteen robots over several months to collect roughly 800,000 grasp attempts. A monocular RGB image and a candidate action were scored by a learned success predictor; cross-entropy search selected promising displacements and the controller closed the loop at 2–5 Hz. Cameras were intentionally uncalibrated and placed differently, forcing the model to tolerate nuisance variation. The principal hand-coded supervision was the gripper-closure heuristic used to decide whether a grasp succeeded.

The learned behavior displayed task-relevant regularities that were not specified directly: pinching soft material, favoring the center of mass for heavy objects, handling flat or transparent objects, and repositioning after imperfect contacts. Training jointly or fine-tuning across two KUKA robot platforms improved performance, especially in the low-data regime and still noticeably near one million examples.

**TALK-BACKED CAVEAT**  The dataset was large but not automatically diverse. Robots preferred easy objects; later human inspection suggested that roughly half of the attempts involved erasers. Raw interaction count therefore overstated effective coverage. The setup also remained a narrow tabletop grasping task with human maintenance and object replacement, not an unrestricted autonomous manipulation learner.

#### **4.3 Predictive models as reusable experience**

Action-conditioned video prediction offers a route from unlabeled interaction to reusable dynamics. Given a current image and candidate future actions, the model predicts future frames. Model-predictive control then searches for actions whose predicted outcome matches a goal. A user can specify that goal by clicking a source and destination pixel; the robot replans at roughly 200 ms intervals to push, rotate, and recover from contact without an explicit object model.

The resulting frames may look blurry, yet the model can still choose useful actions. Rope manipulation extends the idea: a learned dynamics model combined with image subgoals recorded by a person can reproduce sequences that shape or tie the rope.

**EDITORIAL SYNTHESIS**  A predictive model should be judged by decision sufficiency, not visual beauty alone. If uncertainty is preserved in the parts of the prediction that affect action choice, a perceptually imperfect model can be operationally excellent.

#### **4.4 Flexible goals are part of the learning problem**

Task specification is itself a bottleneck. Pixel-motion targets, human-recorded subgoals, and learned perceptual rewards all reduce dependence on instrumented state and hand-written reward functions. In the seminar, Levine discusses visual features learned from human video to define progress for a robot opening a door. The example is associated with 2017 work on unsupervised perceptual rewards and time-contrastive representation learning; the latter was posted shortly after the talk, so the seminar should be read as previewing then-current work rather than claiming a mature general solution.

### **5. Fast adaptation: where are the ImageNet features of motor control?**

**FRAMING QUESTION**  “Where are the ImageNet features of motor control?” The question shifts attention from training one excellent policy to learning parameters that are useful across a distribution of tasks.

ImageNet representations transfer because both the dataset and evaluation demand variety and held-out generalization. Many reinforcement-learning benchmarks instead ask how fast one simulated agent can run on one fixed surface. Such benchmarks reward mastery, not adaptability; they provide little incentive to discover features that transfer to ice, rocks, sand, new goals, or new bodies.

Model-Agnostic Meta-Learning (MAML) trains an initialization so that one or a few ordinary gradient updates on a new task produce maximal improvement. In the seminar examples, a neutral spider/ant policy appears to run in place; a single policy-gradient step adapts it to run forward or backward. The same algorithm also applies to few-shot image classification, suggesting that the mechanism is not tied to a specific architecture or modality.

**TALK-BACKED CAVEAT**  At the time of the seminar, the MAML results shown for locomotion were in simulation and had not yet been evaluated on real robots. The conceptual contribution is the learning objective—adaptability after a small update—not proof of real-world robustness.

### **6. Transferable insights**

*The following principles are editorial syntheses grounded in the seminar's examples. They apply beyond robotics to interactive learning systems, autonomous agents, and data-driven product systems.*

**TRANSFERABLE INSIGHT 01**

#### **Optimize downstream utility, not proxy accuracy**

**Claim.** The correct objective is the behavior the system must accomplish, not the cleanliness of an intermediate estimate.

**Why it transfers.** Whenever a pipeline optimizes a surrogate—pose error, prediction likelihood, classifier confidence—measure whether gains survive through the downstream decision. Error directions are not equally valuable.

**TRANSFERABLE INSIGHT 02**

#### **End-to-end does not mean monolithic**

**Claim.** The benefit is co-adaptation under a task loss, not the removal of planners, filters, geometry, or constraints.

**Why it transfers.** Keep structure that encodes reliable knowledge, but expose enough of the system to joint learning that interfaces can adjust. Differentiable modules permit gradients; non-differentiable modules can still be optimized through search or gradient-free updates.

**TRANSFERABLE INSIGHT 03**

#### **Choose the optimization space deliberately**

**Claim.** A hard parameter-space problem may be easy in trajectory, function, or distribution space.

**Why it transfers.** Improve behavior where local structure is available, then project the result into the global model. Distillation, imitation, and guided policy search are instances of the same broader pattern.

**TRANSFERABLE INSIGHT 04**

#### **Hybridize methods by complementary failure modes**

**Claim.** Model-based and model-free learning are not ideological alternatives; each should own the regime in which its errors are smaller.

**Why it transfers.** Use models in smooth, data-scarce regions and data-driven residual correction near contact, discontinuity, or unmodeled effects. The key question is not “which method wins?” but “where does each method fail predictably?”

**TRANSFERABLE INSIGHT 05**

#### **Treat sample efficiency as a system property**

**Claim.** Useful interactions per hour depend on more than an algorithm's learning curve.

**Why it transfers.** Parallel robots, policy-update cadence, reset automation, exploration, compute latency, safety checks, and data quality all determine wall-clock learning speed. In the Q&amp;A, Levine notes that more frequent policy updates alone could compress months of grasping collection toward days.

**TRANSFERABLE INSIGHT 06**

#### **Diversity is a coverage problem, not a row-count problem**

**Claim.** A million highly correlated examples can represent less experience than a smaller, deliberately varied dataset.

**Why it transfers.** Track effective coverage over objects, conditions, actions, failures, and embodiments. The “half erasers” anecdote is a warning against equating autonomous data collection with representative data collection.

**TRANSFERABLE INSIGHT 07**

#### **Design evaluation for the capability you want**

**Claim.** Training-task reward measures mastery; it cannot establish generalization.

**Why it transfers.** Hold out tasks, goals, domains, bodies, or dynamics, and measure adaptation cost. A robotics analogue of ImageNet needs broad training experience plus difficult tests that are structurally related but genuinely unseen.

**TRANSFERABLE INSIGHT 08**

#### **Useful motor representations are relational**

**Claim.** A reusable control feature is unlikely to live in perception or actuation alone; it captures how observations covary with effective actions.

**Why it transfers.** Representations for interactive systems should encode affordances, controllability, and contact-relevant change. This is why transferring perception-only features may be insufficient for motor control.

**TRANSFERABLE INSIGHT 09**

#### **Judge world models by decisions, not reconstruction aesthetics**

**Claim.** Prediction is an instrument for planning, not necessarily a photorealistic endpoint.

**Why it transfers.** Evaluate whether the model ranks actions correctly, supports recovery, and preserves uncertainty around decision-relevant alternatives. Pixel sharpness is only one diagnostic.

**TRANSFERABLE INSIGHT 10**

#### **Goal interfaces determine scalability**

**Claim.** If each new behavior requires a bespoke reward, the task-specification bottleneck replaces the control bottleneck.

**Why it transfers.** Use outcome examples, demonstrations, subgoals, language, or learned perceptual embeddings where appropriate—but state explicitly what human supervision and temporal meaning remain.

**TRANSFERABLE INSIGHT 11**

#### **Fast adaptation can beat one universal fixed policy**

**Claim.** A strong prior plus a small update may be more realistic than a policy that succeeds unchanged everywhere.

**Why it transfers.** Train for post-update performance, then evaluate both adaptation speed and failure before adaptation. This reframes generalization as an active process rather than static invariance.

**TRANSFERABLE INSIGHT 12**

#### **Scaling learning exposes infrastructure and safety as research problems**

**Claim.** Continuous experience collection requires uncertainty awareness, safe fallback behavior, autonomous resets, docking, recharging, and monitoring.

**Why it transfers.** These are not peripheral engineering details. They determine which data can be gathered, how often policies improve, and whether long-running autonomy is feasible.

### **7. Open problems highlighted by the Q&amp;A**

**Long-horizon hierarchy.** Complex tasks need temporal and spatial decomposition. The open question is how to learn reusable skills and subgoals while still allowing end-to-end credit assignment.

**Active data collection.** The grasping robots selected easy objects, revealing a need for uncertainty-aware sampling, failure targeting, and deliberate diversification.

**Simulation transfer.** The CAD2RL example suggests diversity can matter more than photorealism: randomized textures and lighting may place the real world inside the training distribution's coverage even when individual scenes look unrealistic.

**A robotics-scale benchmark.** There was no clear equivalent of ImageNet for motor control. Such a benchmark must define what transfers and require held-out tasks or embodiments, not simply more training reward.

**Better action abstractions.** The grasping system's roughly 20% error rate reflected a top-down grasp representation that omitted useful nonprehensile manipulation before the grasp.

**Safe, persistent autonomy.** A robot should recognize unfamiliar states, slow down or fall back safely, and manage its own charging and reset cycle if it is to collect experience continuously.

**Distribution shift during iterative learning.** Repeated local improvement and projection can reduce mismatch as step sizes converge, but the seminar does not claim a finite-iteration guarantee.

### **8. Position in Levine's 2015–2019 research trajectory**

The attached 2015 and 2019 materials clarify what is distinctive about the 2017 seminar. It is the bridge between task-level end-to-end control and the later agenda of continually improving robots.

| **Stage** | **Primary emphasis** | **What changes** |
| --- | --- | --- |
| 2015: deep visuomotor policies | End-to-end perception-to-control with guided policy search | Shows that task loss can produce robust, control-relevant visual features on real manipulation tasks. |
| 2017: deep robotic learning | Sample efficiency + generalization + fast adaptation | Diagnoses mastery as insufficient; adds hybrid optimization, multi-robot experience, predictive models, flexible goals, and MAML. |
| 2019: robots that learn by doing | Self-improving experience loops and scalable autonomy | Broadens the thesis: any fixed component that cannot improve from data can cap the system; safety and uncontrolled learning become central. |

**HISTORICAL POSITION**  2015 makes expressive policies trainable by inserting structured teachers; 2017 begins replacing teacher dependence with scalable collective experience; 2019 generalizes that move into a rule for the whole learning loop. The empirical scope also changes: 2015 shows interpolation within trained tasks; 2017 adds novel objects and bounded cross-camera/cross-robot variation; 2019 motivates post-deployment correction while its demonstrations remain controlled. The through-line is editorial synthesis, not a claim that the 2017 systems had already achieved perpetual improvement.

### **9. A practical research checklist**

For a new robot-learning project, ask:

1. What is the true downstream success criterion, and which proxy metrics could improve without improving it?

2. Which modules encode reliable prior structure, and which interfaces need co-adaptation?

3. Where is the system smooth enough for a model, and where are residual, contact, or discontinuity corrections required?

4. How many independent conditions—not just interactions—does the training set cover?

5. Can experience be self-labeled, parallelized, actively selected, or reused across tasks and embodiments?

6. How will a user specify a new goal without engineering privileged state or a bespoke dense reward?

7. What exactly is held out at evaluation time, and how much new data or computation is allowed for adaptation?

8. What safety, reset, charging, monitoring, and policy-update infrastructure determines real wall-clock learning speed?

### **10. Bottom line**

**BOTTOM LINE**  The seminar's durable contribution is a change of unit of analysis. The object to design is not merely a controller or a neural network; it is a closed learning system that connects objectives, optimization, data collection, representation, adaptation, and evaluation. Generalization emerges only when all of those parts are built to reward reuse rather than single-task mastery.

For practitioners, the immediate implication is concrete: make the end objective visible to learning, preserve structure where it is trustworthy, collect experience for coverage rather than volume, and evaluate on the new conditions the system is supposed to handle.

### **Sources and evidence note**

Primary source. The notes follow the full auto-generated caption track of the seminar, repaired only for obvious transcription errors and checked against the official event page and associated research papers. They are analytic notes, not a verbatim transcript. Timestamps refer to the linked recording.

- [CMU Robotics Institute event page — Deep Robotic Learning](<https://www.ri.cmu.edu/event/deep-robotic-learning/>) — official talk metadata

- [YouTube recording — Deep Robotic Learning](<https://www.youtube.com/watch?v=eKaYnXQUb2g>) — full seminar and Q&amp;A

#### **Associated primary papers**

- [End-to-End Training of Deep Visuomotor Policies](<https://jmlr.org/papers/v17/15-522.html>) — JMLR 2016; guided policy search and task-level visuomotor learning

- [Guided Policy Search as Approximate Mirror Descent](<https://arxiv.org/abs/1607.04614>) — MDGPS formulation

- [Combining Model-Based and Model-Free Updates for Trajectory-Centric Reinforcement Learning](<https://proceedings.mlr.press/v70/chebotar17a.html>) — PILQR; ICML 2017

- [Deep Reinforcement Learning for Robotic Manipulation with Asynchronous Off-Policy Updates](<https://arxiv.org/abs/1610.00633>) — multi-robot continuous Q-learning

- [Collective Robot Reinforcement Learning with Distributed Asynchronous Guided Policy Search](<https://arxiv.org/abs/1610.00673>) — four-robot visual door-opening study

- [Learning Hand-Eye Coordination for Robotic Grasping with Deep Learning and Large-Scale Data Collection](<https://arxiv.org/abs/1603.02199>) — roughly 800,000 grasp attempts

- [Deep Visual Foresight for Planning Robot Motion](<https://arxiv.org/abs/1610.00696>) — action-conditioned video prediction and planning

- [Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks](<https://proceedings.mlr.press/v70/finn17a.html>) — MAML; ICML 2017

- [CAD2RL: Real Single-Image Flight without a Single Real Image](<https://arxiv.org/abs/1611.04201>) — domain-randomized simulation transfer

- [Unsupervised Perceptual Rewards for Imitation Learning](<https://www.roboticsproceedings.org/rss13/p50.pdf>) — human-video-derived task progress

- [Time-Contrastive Networks: Self-Supervised Learning from Video](<https://arxiv.org/abs/1704.06888>) — related work posted shortly after the seminar

#### **Editorial scope**

The cross-year comparison uses the four user-supplied 2015 and 2019 transcript/summary documents. Claims about the 2017 talk are grounded in the recording; broader principles are explicitly labeled as editorial synthesis. No claim should be read as evidence that the methods solve long-horizon planning, safety, real-world meta-learning, or open-ended autonomous data collection.
