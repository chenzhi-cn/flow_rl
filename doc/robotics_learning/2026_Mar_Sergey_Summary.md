---
id: sergey-2026-03-summary
title: Robotic Foundation Models Research Notes
speaker: Sergey Levine
language: en
document_type: research_talk_summary
source_file: 2026_Mar_Sergey_Summary.docx
source_sha256: cf7996b1e5b7341ea5c0cd712ebd889e7dbfb65490195f3330b4991a24c8d7e5
period_from_filename: 2026-03
talk_date: '2026-03-27'
recording_date: null
video_publication_date: '2026-03-29'
video_url: https://www.youtube.com/watch?v=_b-FwaiETwM
keywords:
- robot foundation models
- vision-language-action models
- VLA
- π₀
- π₀.₅
- π*₀.₆
- RECAP
- advantage conditioning
- grounded reasoning
- cross-embodiment learning
- real-world reinforcement learning
conversion_date: '2026-09-30'
content_scope: Full source notes, with source-derived retrieval overview and normalized Markdown formatting.
verification_scope: Converted against the supplied DOCX only. Original attribution, caveats, and references are
  retained; external sources were not re-verified during conversion.
---

# Robotic Foundation Models Research Notes

## Retrieval overview

This overview is derived from the supplied summary. The full source notes follow, including their evidence qualifications and editorial interpretations.

**Research goal:** Build a general-purpose robot that follows open-ended instructions, operates in unfamiliar environments, executes dexterous motion, recovers from mistakes, and improves from experience.

**Motivation:** Describing a physical task does not provide the continuous control needed to execute it, and modest imitation success is insufficient for reliable deployment.

**Approach:** Combine broad pretraining, continuous action experts, grounded hierarchical reasoning, coherent post-training, and real-world RL that reuses successful and unsuccessful experience.

## Source notes

RESEARCH TALK NOTES

Motivation Problems Methods and Evidence

### Talk metadata

| **Field** | **Details** |
| --- | --- |
| Talk | *Robotic Foundation Models* |
| Speaker | Sergey Levine, Associate Professor, UC Berkeley |
| Series and venue | Agentic AI Frontier Seminar, online |
| Seminar date | March 27, 2026 |
| Channel | AI Agent Frontier |
| Video | [YouTube recording](<https://www.youtube.com/watch?v=_b-FwaiETwM>) |
| Video published | March 29, 2026 |
| Duration | 1 hour and 13 seconds |

### Executive takeaway

The talk asks how to turn an AI system that can explain a physical task into one that can actually perform it. Levine's target is a general-purpose robot that can accept an open-ended language instruction, operate in an unfamiliar physical environment, execute contact-rich continuous motion, recover from its own mistakes, and improve through experience.

His proposed research direction is a robotics version of the foundation-model recipe. Broad pretraining supplies visual-semantic knowledge, physical grounding, and coverage across tasks and robot embodiments. Post-training teaches the model a coherent strategy and lets humans supervise it at useful levels of abstraction. Real-world reinforcement learning then improves reliability, recovery, and throughput. The architectural progression is correspondingly from tokenized actions in early vision-language-action models, to a continuous action expert, to grounded hierarchical reasoning, and finally to an advantage-conditioned policy that can reuse both successful and unsuccessful experience.

The evidence in the talk supports this recipe in stages rather than proving that general-purpose robotics is solved. Cross-embodiment training can produce positive transfer; grounded reasoning can improve generalization; broad pretraining can make narrow adaptation more data-efficient; and real-world RL can substantially improve difficult long-horizon tasks. However, the reported systems still fail, safety remains unresolved, and several results are shown as videos or narrated plots without full statistical detail.

### Motivation

#### Practical motivation

Today's language models can describe how to clean a house, fold laundry, or prepare dinner, but they cannot carry out the required physical work. A useful physical agent must connect language and perception to dexterous behavior: manipulating deformable objects, managing contact and force, sequencing a long task, responding to changing geometry, and recovering when an action does not have its intended effect.

The central practical gap is therefore not planning alone. A symbolic plan such as "grasp the shirt, straighten it, and fold it" omits the continuous feedback and adjustments required to execute each step. Code-generation systems such as Code as Policies can compose hand-provided perception and control APIs for structured tasks, but an API boundary cannot enumerate every low-level behavior needed for arbitrary physical interaction. Levine's claim is that the mapping from perception and goals to physical action must be learned end to end across multiple levels of abstraction.

#### Scientific motivation

Everyday manipulation unifies semantic reasoning, visual and spatial understanding, temporal planning, dynamics, contact, and motor control. This makes the physical world a compact test of capabilities that are often studied separately. Levine contrasts it with a closed environment such as Go: AlphaGo could discover novel strategies within the rules of Go, but experience in that world cannot teach a system to cook.

Levine goes further and argues that physical interaction may become a route to richer machine intelligence. A robot can perform experiments in an environment governed by the physical universe, rather than interacting only with text, software, or websites authored by humans. This is a motivating research vision, not an empirical conclusion established by the manipulation experiments in the talk.

#### Scaling motivation

Hand-written controllers, fixed APIs, and one policy per task can work in restricted settings, but their engineering cost grows with the number of tasks, objects, environments, and robot platforms. The scaling alternative is to learn reusable structure from heterogeneous data. Web data contributes concepts and semantics; broad robot data contributes physical grounding and recovery states; cross-embodiment data lets experience on one platform benefit another; and a small amount of target-domain post-training selects a consistent behavior for the deployment task.

The important claim is not merely that more data is better. Different stages require different data properties. Pretraining needs coverage, including varied strategies, imperfect behavior, and off-nominal states. Post-training needs consistency, so the model learns a coherent strategy rather than averaging incompatible ones.

#### Reliability motivation

A policy that succeeds about half the time may be an interesting laboratory demonstration but is rarely useful in a home or factory. Small imitation errors compound, moving the robot into states that are absent from clean expert demonstrations. Clean data alone can therefore produce a brittle policy: it teaches what correct execution looks like but gives few examples of how to recover.

Reliability also includes speed. A robot that eventually succeeds only after repeated hesitation or retries may have low operational value. Levine therefore treats success rate and throughput as post-training objectives and argues that robots need real-world practice, not only imitation, to achieve them.

### Core research objective

The objective can be stated precisely as follows:

Learn a general-purpose policy that maps multimodal observations, robot state, and natural-language goals to continuous action trajectories, while generalizing across tasks, objects, environments, and robot embodiments; decomposing long-horizon tasks at useful abstraction levels; operating at real-time control rates; recovering from off-distribution states; and improving reliability and throughput with a limited budget of real-world interaction.

This objective contains six coupled technical problems:

1. **Representation** - combine visual-semantic knowledge with high-frequency continuous control in one trainable system.

2. **Grounding** - translate web-trained concepts and high-level reasoning into commands that a motor policy can execute.

3. **Transfer** - obtain positive transfer from data collected on different tasks, in different environments, and on different robots.

4. **Temporal abstraction** - coordinate slow deliberation with fast reactive control.

5. **Robust improvement** - learn from imperfect autonomous trajectories without simply imitating the mistakes in them.

6. **Safety** - acquire coverage of rare physical hazards without relying on unsafe exploration.

### Three challenge matrix

| **Challenge** | **Why it is difficult** | **Main method in the talk** | **Remaining gap** |
| --- | --- | --- | --- |
| Generalization across diverse scenarios | New objects, layouts, semantic states, and task formulations create distribution shift | Reuse pretrained VLM knowledge; add grounded reasoning; train steerable and hierarchical policies | Generalization remains below perfect performance and is demonstrated on bounded task and robot distributions |
| Sufficient and appropriate data | Robot data is expensive, heterogeneous, and narrow compared with web data | Pool data across tasks, laboratories, and embodiments; co-train with web data; separate broad pretraining from consistent post-training | The optimal mixture, scaling law, and transfer limits for substantially different embodiments are unknown |
| Robustness and speed | Compounding errors produce unfamiliar states; physical rollouts are costly; slow success is insufficient | Retain imperfect and recovery data; post-train on coherent demonstrations; use real-world RL and advantage-conditioned policies | Residual failures, safety, value-model error, and long-duration reliability remain open |

This mapping is a research decomposition, not a claim that each method fully solves its corresponding challenge. The challenges interact: better reasoning still needs grounded motor competence; broader data still needs task-specific strategy; and RL still depends on the quality and safety of collected experience.

### Architecture evolution

| **Stage and system** | **Interface and output** | **Contribution** | **Principal limitation** |
| --- | --- | --- | --- |
| Symbolic robot agent using Code as Policies | LLM generates a plan, program, or calls to predefined APIs | Composes known skills and handles structured spatial or logical tasks | Hand-designed primitives do not express tacit, contact-rich dexterity |
| First-generation VLA using RT-2 and RT-2-X | Image and instruction in; discretized action tokens out through a VLM-style decoder | Transfers visual-semantic knowledge directly into robot control | A language-token interface is awkward for fine, high-rate continuous motion |
| Continuous-action VLA using π₀ | VLM backbone plus a dedicated diffusion or flow-matching action expert | Produces smooth, continuous action chunks and supports dexterous control | Broad competence is still imperfect; long-horizon reasoning and reliability require additional training |
| Reasoning and steerable VLA using ECoT and Steerable Policies | Intermediate textual and spatial predictions guide a multi-level action interface | Grounds semantic reasoning in objects, locations, motions, and subgoals | Reasoning supervision, latency, and the faithfulness of generated thoughts remain concerns |
| Hierarchical generalist using π₀.₅ | High-level subtask prediction and low-level continuous action generation in one model | Supports verbal coaching, long-horizon decomposition, and open-world transfer | Performance still saturates below deployment-grade reliability |
| Experience-improving VLA using π\*₀.₆ with RECAP | State-value model plus an advantage-conditioned VLA trained on demonstrations, failures, corrections, and autonomous experience | Reuses imperfect experience and improves success and throughput without conventional on-policy batches | Physical data collection, safe exploration, value accuracy, and residual failure remain bottlenecks |

#### The Pi Zero continuous action expert

The key change from RT-2-style action tokenization to π₀ is a motor-specific continuous decoder. In the architecture described in the talk:

- the inputs include one to three images, a language instruction, and continuous robot state such as joint positions;

- a pretrained visual-language or language backbone supplies semantic context and is described as approximately 2.6 billion parameters;

- a smaller action expert has the same transformer depth and attends to the backbone's representations;

- flow matching learns to generate a continuous trajectory rather than a string of discretized action tokens; and

- the output is an action chunk, a short sequence of future controls sampled at up to 50 Hz.

The action chunk separates the frequency of motor commands from the speed of language generation. Closed-loop observation still lets the system replan between chunks. Laundry demonstrations show missed grasps, perturbations, and retries; their strongest qualitative evidence is recovery, not flawless execution.

### Pretraining and post training

| **Stage** | **Typical data** | **Intended role** | **Risk if used alone** |
| --- | --- | --- | --- |
| Vision-language pretraining | Web text and images, captioning, VQA, recognition, and spatial concepts | Acquire concepts and broad semantic and visual knowledge that are expensive to rediscover from robot trajectories | The model can describe a task without knowing how to execute it |
| Broad robot pretraining | Many tasks, environments, robots, strategies, successes, mistakes, and recoveries | Ground concepts in action, cover physical variation, and enable cross-embodiment transfer | Produces breadth but may be inconsistent, slow, or only moderately reliable |
| High-level supervision | Subtask labels, grounded locations, synthetic ECoT targets, and spoken human coaching | Teach decomposition and a grounded interface between reasoning and control | Coarse language can remain weakly grounded, and generated rationale need not be causally faithful |
| Narrow SFT or post-training | Tens of hours of consistent expert behavior for a target setting, according to the talk | Select a coherent strategy and adapt the generalist without relearning the world | Clean expert data contains few errors and recoveries, so imitation can be brittle |
| Real-world RL | Autonomous trials, task outcomes, value estimates, and expert interventions or corrections | Improve reliability, recovery, cycle time, and throughput from consequences and deployment data | Rollouts are expensive and potentially unsafe; poor reward or value estimates can misdirect learning |

The most important distinction is between **coverage** and **consistency**. Pretraining should cover the variety of states the robot may encounter, including bad states. Post-training should consistently demonstrate a strong strategy. This explains the otherwise counterintuitive claim that a broad, messy pretraining set may be more useful than a clean expert-only set for recovery, while clean data is still valuable for final specialization.

### Grounded reasoning and hierarchical control

#### Embodied Chain of Thought

Embodied Chain of Thought, or ECoT, inserts robot-relevant intermediate predictions between observation and action. The targets can include:

- task decomposition and the current subtask;

- a natural-language description of the next motion;

- task-relevant object identities and bounding boxes;

- gripper or end-effector location; and

- image-space coordinates and spatial relationships.

These targets create a bridge from web-scale visual and language supervision to embodied action. Web data rarely specifies the correct joint trajectory for a particular robot, but it contains extensive knowledge about objects, spatial relations, and task structure. ECoT makes those transferable quantities explicit before the model produces an action.

ECoT is not merely an inference-time prompt. The policy is explicitly trained to emit these grounded traces, using synthetic labels produced by pretrained perception and language systems. The reported comparison adds this supervision without adding new robot trajectories.

The speaker reports that adding ECoT to a 7B OpenVLA model substantially improves performance and that the resulting model exceeds the 55B RT-2-X comparison shown on the slide. The exact metric and bar heights are not spoken, so the defensible conclusion is the qualitative ordering, not a reconstructed numerical margin. The generated trace should also be understood as a useful intermediate control representation, not necessarily a faithful explanation of the model's internal causal computation.

#### Steerable policies

A high-level reasoner helps only if the motor policy understands its output. A steerable policy is therefore trained to accept instructions at several levels:

1. a complete task goal;

2. a semantic subtask;

3. a motion-level instruction; and

4. a grounded image-space command such as a target location or point-to-point motion.

This creates a System 1 and System 2 style interface. Familiar situations can be handled directly. Unusual or ambiguous situations can invoke more semantic reasoning, after which the result is translated into a grounded command the motor policy has learned to follow. Examples in the talk include distinguishing a banana from a yellow toy duck and inferring the steps needed to "make the blue block the only object on the plate."

In the reported human-oracle test, a person steering the policy through all available abstraction levels reaches almost 100% success, whereas access to only a subset performs worse. A learned reasoner also improves on the non-steerable base policy. The oracle result demonstrates the capability of the policy interface; it does not show that autonomous reasoning itself has reached the same success rate, and the exact autonomous gain and trial count are not stated verbally.

#### Pi Zero Point Five

π₀.₅ integrates high-level discrete prediction and low-level continuous action in one model. Given a broad command such as "clean the bedroom," it can predict the next useful subtask, such as picking up a pillow, and then generate the action trajectory to execute it.

Humans can supervise this process by speaking subtask instructions instead of teleoperating every movement. Those instructions become high-level training targets. At inference, the model can generate its own subtasks and carry out long-horizon cleaning in kitchens and bedrooms absent from its training set.

The narrated ablations support three claims. First, verbal-instruction supervision matters: removing it produces a large drop. Second, reasoning targets improve the learned representation even when explicit reasoning is disabled at test time. Third, the autonomous model can outperform the average human-guided condition, so human verbal guidance is not necessarily an upper bound. Exact values and axis definitions are slide-dependent and are therefore omitted here.

#### Slow reasoning and fast action

The Q&amp;A describes two asynchronous model paths. In the coffee example, high-level thoughts are generated at roughly 0.5 Hz, or once every two seconds, while the action path consumes the latest thought approximately every 150 ms. This is distinct from the up-to-50-Hz commands within an action chunk. The hierarchy lets deliberation run slowly without pausing reactive control. The demonstrated system runs in the cloud; Levine suggests that a safety-critical deployment could run the action generator locally while leaving slower reasoning remote.

### Cross embodiment data and RT X

The Open X-Embodiment and RT-X project asked more than 30 robotics laboratories to pool heterogeneous manipulation data. The subset discussed in the talk was curated to single-arm robots with parallel-jaw grippers, preserving substantial variation in platforms and tasks while avoiding an unconstrained embodiment gap. A single generalist model was sent back to what Levine describes as seven participating laboratories and compared with task-specific specialist systems.

Levine reports a generalist mean score around 63 and an average improvement of approximately 50% over the individual specialist models. The unit and exact specialist mean are not clear in the spoken transcript, so "63" should not be rewritten as a percentage. The result supports positive transfer on average across the evaluated tasks; it does not imply that the generalist won every individual comparison or that arbitrary embodiments can already share data without loss.

The broader data argument continues in π₀ and π₀.₅:

- π₀ is described as using roughly 10,000 hours of robot data. Levine's additional statement that newer models often use about an order of magnitude more is a broad field estimate, not a documented size for a named system in this talk.

- Only about 3% of the π₀.₅ training mixture is described as mobile-robot data; approximately 97% comes from other platforms. Its mobile manipulation performance is therefore evidence of cross-platform transfer.

- In an environment-scaling experiment, performance in unseen kitchens and bedrooms improves with the number of training environments and approaches the test-environment-trained oracle after roughly 100 environments. The talk refers to performance around 84%, leaving a meaningful reliability gap. This is specific to the evaluated distribution and should not be read as a universal "100 homes are enough" scaling law.

- Removing cross-robot data hurts more than simply reducing the number of target environments, which suggests that embodiment diversity contributes information beyond raw target-domain volume.

The core interpretation is that different robots share latent structure: object semantics, task sequences, visual relationships, contact patterns, and partial motor regularities. Cross-embodiment pretraining can learn this shared structure, while target-domain post-training resolves the remaining platform-specific details.

### Pi Star Zero Point Six and RECAP

#### Why conventional on policy RL is a poor fit

Methods such as PPO or GRPO assume that many fresh samples can be generated from a recent policy. Token generation is cheap for an LLM; each sample from a physical robot is a real attempt that consumes time, requires resets, wears hardware, and can create safety risks. Real-world RL therefore needs to reuse previously collected trajectories, including unsuccessful ones, and incorporate human corrections efficiently.

#### Mechanism

π\*₀.₆ applies **RECAP**, short for **RL with Experience and Corrections via Advantage-conditioned Policies**, to a π₀.₆-family VLA. The talk's operational description is:

1. Train the VLA on broad offline data, retaining demonstrations, imperfect trajectories, failures, and corrections.

2. Train a value function that estimates the quality or progress of the current visual state toward task completion.

3. Estimate advantage from task return and the change in value across an action or short transition segment. A segment that improves the situation receives positive advantage; one that worsens it receives negative advantage.

4. Label dataset actions with a binarized text condition, "Advantage: positive" or "Advantage: negative," and train the VLA on both kinds of experience.

5. At deployment, condition the policy on positive advantage rather than discarding all unsuccessful or corrected experience during training.

6. Collect autonomous on-robot experience and expert corrections, then update the value estimate and policy in another offline training round. Repeat as needed.

The conceptual point is that the value model evaluates a **state or situation**. The action's advantage is inferred from the change in value across the transition. Saying that the value function directly judges a candidate action would blur these two roles.

The training sequence should also remain explicit:

**π₀.₆ supervised pretraining → π\*₀.₆ offline-RL pretraining → task-specific SFT → on-robot experience and corrections → final π\*₀.₆.**

The SFT comparison in the talk is not a supervised-from-scratch baseline. It begins from a model that has already received offline-RL pretraining. The final comparison therefore attributes the additional gain more specifically to on-robot experience, corrections, and iterative RECAP updates; it does not measure the total value of all RL relative to an untrained imitation model.

#### Evidence from long horizon tasks

The value predictions shown in the talk behave qualitatively as intended: crumpling a shirt lowers predicted progress, opening a refrigerator raises it, and tipping over a water filter lowers it. These examples are diagnostic visualizations, not a calibration study.

The main task demonstrations are:

- **Espresso preparation** - grasp and insert a portafilter, tamp coffee, lock the portafilter into the machine, wait for extraction, handle liquid, and pour the drink. Levine reports an approximately 13-hour continuous evaluation. Failures still occur, including dropped portafilters, but behavior improves markedly after RL.

- **Laundry** - manipulate deformable clothing while recovering from poor grasps and unfavorable cloth states.

- **Industrial box assembly** - at Dandelion Chocolate, form a flat box, apply a label, and stack the completed box. The relevant metric is successful boxes per hour, not success alone.

Levine summarizes the aggregate slide as roughly twice the task-SFT throughput on most shown tasks. The primary RECAP report makes the narrower explicit claim that throughput more than doubles on diverse laundry and espresso, while box assembly reaches about twice the throughput after two RECAP iterations. Exact slide bars, uncertainty, and trial details are not spoken, so the talk's summary should not be generalized beyond these evaluated tasks.

### Evidence and results

| **System or study** | **Reported observation** | **Supported conclusion** | **Important qualification** |
| --- | --- | --- | --- |
| π₀ laundry | The robot separates, straightens, folds, and transports clothing and sometimes retries after misses or perturbations | A continuous-action VLA can express contact-rich, recoverable behavior | No success rate, recovery rate, or trial count is stated |
| ECoT on OpenVLA | The 7B reasoning model exceeds its non-reasoning base and the shown 55B RT-2-X comparison | Grounded intermediate prediction can improve generalization without simply increasing parameter count | Exact metric and bar values are not spoken |
| Steerable Policies | A human oracle with access to every abstraction level reaches almost 100%, and learned steering improves over the base policy | A multi-level command interface exposes useful low-level competence and can help autonomous reasoning | Oracle success is not autonomous success; exact learned gain and trial count are absent |
| π₀.₅ long-horizon cleaning | The model cleans previously unseen kitchens and bedrooms; verbal supervision and reasoning targets matter in ablations | Hierarchical prediction and heterogeneous co-training support open-world transfer | The evaluation covers a particular task, robot, and home distribution |
| RT-X | One cross-lab generalist obtains a mean near 63 and is about 50% better than specialists on average | Cross-embodiment pooling can generate positive transfer strong enough to beat specialists on average | The score's unit is not spoken; the claim is an average, not universal dominance |
| π₀.₅ data mixture | About 3% mobile-robot data and 97% other-platform data still yield useful mobile performance | Non-target embodiments can supply transferable knowledge | Dataset definitions and each source's contribution are not isolated |
| π₀.₅ environment scaling | Performance rises with more training environments, approaches an in-environment oracle around 100 locations, and remains near the mid-80s rather than 100% | Environment diversity improves generalization but does not close the robustness gap | Approximate values are task- and metric-specific |
| Narrow specialist real-world RL | Per-task policies reach 100% in their training setting and are roughly two times faster or more than supervised counterparts | Real-world practice can produce precise, fast manipulation | Each task has its own policy; this does not establish generalization |
| π\*₀.₆ espresso | Roughly 13 hours of continuous operation with occasional failures and clear improvement after RL | Offline-to-online RL can improve a complex semantic and physical workflow | It is not failure-free, and a single long run is not a complete reliability distribution |
| π\*₀.₆ aggregate | The talk reports about 2× task-SFT throughput on most shown tasks; the paper reports more than 2× on diverse laundry and espresso and about 2× box throughput after two iterations | Experience and corrections can improve deployment utility, including speed and recovery | Exact slide bars, confidence intervals, and sample counts are not spoken |

### Limitations and open questions

#### Evidence quality

- Many results are presented through selected videos and bar charts without trial counts, confidence intervals, complete axes, or detailed baseline protocols.

- Aggregate comparisons across heterogeneous tasks should not be reframed as universal superiority.

- A recovery video establishes that recovery is possible, not how often it succeeds.

- Generated intermediate text may improve control without faithfully revealing the model's internal reasoning.

- The thesis that physical interaction could produce knowledge beyond human-authored sources is a forward-looking scientific claim, not a result of the reported experiments.

#### Generalization and data

- How far can cross-embodiment transfer extend when kinematics, sensors, end effectors, or locomotion differ substantially?

- What mixture of web data, robot trajectories, human video, simulation, tactile data, failures, and autonomous experience is optimal?

- Does performance continue to scale beyond the approximately 100-environment regime shown for π₀.₅, or does the task distribution need to change qualitatively?

- How can data be selected for rare but safety-critical states that ordinary operation seldom visits?

#### Reasoning and control

- What representation should connect deliberation to action: language, image-space points, trajectories, predicted images, spatial relations, or a learned latent state?

- How should a fast controller detect that slow reasoning is stale, incorrect, or inconsistent with current sensory input?

- Can intermediate predictions be verified and grounded rather than merely plausible?

- How should reasoning and action paths arbitrate conflicts under latency or network failure?

#### Reinforcement learning and reliability

- How robust is advantage-conditioned learning when the value model is wrong, poorly calibrated, or extrapolating outside its data?

- How should the system request help, stop, or fall back to a safe controller when uncertainty is high?

- What evaluation protocol demonstrates weeks or months of reliability rather than a selected video or an hours-long run?

- Can autonomous data collection remain sample-efficient as tasks become more diverse and resets become more expensive?

#### Safety and sensing

- Levine explicitly describes safety as fundamentally difficult and unresolved. Common-sense transfer may help with rare situations, but it is not a substitute for constraints, monitoring, uncertainty estimation, hardware safety, and validated intervention procedures.

- Vision, especially wrist-mounted cameras, can partially substitute for touch under occlusion, but the marginal value of high-resolution tactile sensing is still unclear. The long-run sensor suite must balance information gain against hardware complexity, reliability, and noise.

- Cloud execution raises connectivity, privacy, and failure-mode questions. Moving the fast action path to the edge reduces some latency and dependency risks but does not by itself make the policy safe.

### Key research insights

1. **End-to-end does not mean hierarchy-free.** The scalable hierarchy is learned rather than fixed: high-level goals, subtasks, spatial commands, and continuous actions remain inside a jointly trained interface.

2. **Semantic intelligence is useful only when grounded.** A strong VLM may know what should happen, but a steerable policy and continuous action expert are what make that knowledge executable.

3. **Data quality is stage-dependent.** Pretraining benefits from breadth, mistakes, and recovery states; post-training benefits from a narrow and consistent strategy.

4. **Generalists can outperform specialists through transfer.** RT-X suggests that pooled experience can outweigh the loss of task-specific optimization, at least on average within the evaluated embodiment family.

5. **Non-target robot data can be more valuable than additional target environments.** The π₀.₅ ablation indicates that cross-robot experience supplies reusable structure rather than merely more samples.

6. **Reasoning supervision can improve the policy even when reasoning is hidden at deployment.** The π₀.₅ ablation suggests that intermediate targets shape useful representations, not only visible test-time traces.

7. **Recovery requires seeing bad states.** Expert-only imitation systematically undersamples the states produced by the policy's own errors; broad data and autonomous experience fill this gap.

8. **Real-world RL should optimize operational utility.** Throughput captures speed, hesitation, recovery, and success in a way that binary success alone does not.

9. **Robotics requires multiple timescales.** Slow semantic deliberation can guide a faster action generator, which in turn produces still faster low-level controls.

10. **The emerging recipe is breadth, interface, then practice.** Broad pretraining builds knowledge and coverage; grounded hierarchical supervision provides a usable reasoning-to-action interface; on-robot RL turns competence into reliability.

### Glossary

| **Term** | **Meaning in this talk** |
| --- | --- |
| Agentic system | A system that interprets a request and takes actions to accomplish it rather than only responding with text |
| Physical intelligence | Competence grounded in perception, motion, contact, physical dynamics, and real-world interaction |
| Robotic foundation model | A broadly pretrained model intended to transfer across robot tasks, environments, and often embodiments |
| End-to-end learning | Learning the observation-and-goal-to-action mapping without relying entirely on hand-written intermediate programs |
| Generalist and specialist | One model spanning many settings versus a model optimized for one task or environment |
| VLM | Vision-language model |
| VLA | Vision-language-action model |
| RT-2 and RT-2-X | Early VLA family using tokenized action representations; RT-2-X is a cross-embodiment generalist |
| OpenVLA | Open-source 7B VLA used as the base in the ECoT comparison |
| π₀ | Flow-matching VLA with a pretrained semantic backbone and a continuous action expert |
| π₀.₅ | Generalist VLA that combines heterogeneous co-training with high-level prediction and low-level continuous action |
| π\*₀.₆ | Experience-trained version of π₀.₆ using the RECAP method |
| Action expert | Continuous-action transformer or decoder attached to the visual-language backbone |
| Flow matching | A continuous generative objective that learns a vector field transforming noise into an action trajectory |
| Action chunk | A short sequence of future low-level controls predicted together |
| Test-time compute | Additional inference-time computation, here used for intermediate reasoning before or alongside action generation |
| ECoT | Embodied Chain of Thought, using grounded intermediate predictions for tasks, objects, locations, motions, and robot state |
| Steerable policy | A VLA trained to accept commands at several semantic and spatial abstraction levels |
| SFT | Supervised fine-tuning |
| Offline RL | Learning from previously collected trajectories rather than only fresh on-policy interaction |
| Value function | A predictor of how favorable a current state or situation is for eventual task completion |
| Advantage | The relative improvement associated with an action or transition, described here through the change in value |
| RECAP | RL with Experience and Corrections via Advantage-conditioned Policies |
| Throughput | Successful task completions per hour |
| Cross-embodiment learning | Training on experience from multiple robot bodies or platforms so that shared knowledge transfers between them |

### Selected primary references

1. Sergey Levine, [Robotic Foundation Models](<https://www.youtube.com/watch?v=_b-FwaiETwM>), AI Agent Frontier Seminar.

2. Jacky Liang et al., [Code as Policies: Language Model Programs for Embodied Control](<https://arxiv.org/abs/2209.07753>), 2022.

3. Anthony Brohan et al., [RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control](<https://arxiv.org/abs/2307.15818>), 2023.

4. Open X-Embodiment Collaboration et al., [Open X-Embodiment: Robotic Learning Datasets and RT-X Models](<https://arxiv.org/abs/2310.08864>), 2023.

5. Sudeep Dasari et al., [Robotic Control via Embodied Chain-of-Thought Reasoning](<https://arxiv.org/abs/2407.08693>), 2024.

6. Physical Intelligence et al., [π₀: A Vision-Language-Action Flow Model for General Robot Control](<https://arxiv.org/abs/2410.24164>), 2024; [official overview](<https://www.pi.website/blog/pi0>).

7. Physical Intelligence et al., [π₀.₅: A Vision-Language-Action Model with Open-World Generalization](<https://arxiv.org/abs/2504.16054>), 2025; [official overview](<https://www.pi.website/blog/pi05>).

8. William Chen et al., [Steerable Vision-Language-Action Policies for Embodied Reasoning and Hierarchical Control](<https://arxiv.org/abs/2602.13193>), 2026.

9. Jianlan Luo et al., [Precise and Dexterous Robotic Manipulation via Human-in-the-Loop Reinforcement Learning](<https://arxiv.org/abs/2410.21845>), 2024.

10. Physical Intelligence et al., [π\*₀.₆: A VLA That Learns From Experience](<https://arxiv.org/abs/2511.14759>), 2025; [official RECAP overview](<https://www.pi.website/blog/pistar06>).
