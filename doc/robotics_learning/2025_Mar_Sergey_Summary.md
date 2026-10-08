---
id: sergey-2025-03-summary
title: Robotics Foundation Models
speaker: Sergey Levine
language: en
document_type: research_talk_summary
source_file: 2025_Mar_Sergey_Summary.docx
source_sha256: c30961847d064642c8b7683eb33557b10b4d56c5ec6e61e0e1b1cd4a39aaf26b
period_from_filename: 2025-03
talk_date: null
recording_date: null
video_publication_date: '2025-03-24'
video_url: https://www.youtube.com/watch?v=EYLdC3a0NHw
keywords:
- robot foundation models
- vision-language-action models
- VLA
- RT-X
- Open X-Embodiment
- π₀
- flow matching
- ECoT
- Hi Robot
- RLPD
- HIL-SERL
conversion_date: '2026-09-30'
content_scope: Full source notes, with source-derived retrieval overview and normalized Markdown formatting.
verification_scope: Converted against the supplied DOCX only. Original attribution, caveats, and references are
  retained; external sources were not re-verified during conversion.
date_note: 24 March 2025 is identified as the video publication date.
---

# Robotics Foundation Models

## Retrieval overview

This overview is derived from the supplied summary. The full source notes follow, including their evidence qualifications and editorial interpretations.

**Research goal:** Build robot foundation models that transfer across tasks, environments, and embodiments and improve through limited adaptation data and autonomous experience.

**Motivation:** Collecting data and training a separate policy for each task and robot fragments experience and limits generalization and continual improvement.

**Approach:** Combine cross-embodiment pretraining, vision-language knowledge, continuous action modeling, hierarchical reasoning, reinforcement learning, and autonomous data collection.

## Source notes

RESEARCH SEMINAR NOTES

Research Notes on Sergey Levine's Seminar

**Speaker**  Sergey Levine  UC Berkeley

**Host**  University of Toronto Robotics Institute

**Video published**  24 March 2025

**Duration**  55 minutes 15 seconds

**Source video**  [YouTube recording](<https://www.youtube.com/watch?v=EYLdC3a0NHw>)

Reader's note  These notes reorganize the seminar around its research motivation, core problems, methods, evidence, limitations, and open questions. Quantitative results follow the comparisons stated in the talk, and technical names were checked against the relevant projects and papers.

### Core Thesis of the Talk

Sergey Levine's central claim is that robot learning should move away from collecting data and training a bespoke policy separately for every robot and task, and toward the foundation-model paradigm established by large language models and vision-language models. In the ideal system, a general-purpose vision-language-action model is first pretrained on large-scale, multitask, multi-robot data and then adapted with a small amount of high-quality data. Inference-time computation, reinforcement learning, and autonomously collected robot data provide further improvement. The goal is to enable a single model to transfer across platforms and tasks and to continue improving after deployment.

The talk does not focus on a single model; it presents a research program that progressively fills in missing capabilities. Cross-embodiment data addresses whether different robots can share experience. VLAs connect internet-scale semantic knowledge to control. π₀ improves continuous-action modeling and complex manipulation. Sequential reasoning adds semantic planning over long horizons and supports interactive instruction following. Reinforcement learning directly optimizes success rate and speed. Autonomous data collection attempts to create a closed loop of continual improvement.

**Evidence standard.** All quantitative results in these notes are based on the speaker's verbal statements or the interpretation of slides shown during the talk. The video does not report absolute success rates, trial counts, error bars, or confidence intervals for every result. Accordingly, claims such as "approximately 50% better" and "3×" should be understood as relative comparisons presented in the talk, not as percentage-point gains or universal guarantees.

### Research Objective

The long-term objective is to build robot foundation models that transfer across tasks, environments, and robot embodiments, and that continue to improve through limited adaptation data and autonomous experience. Such a model must support four capabilities:

- Learn shared representations from data spanning different robots, laboratories, and tasks, rather than treating each dataset as an isolated silo.

- Inherit knowledge of objects, spatial relations, language instructions, and commonsense concepts from internet-scale vision-language pretraining.

- Produce continuous, high-frequency actions with closed-loop correction for contact-rich, bimanual, and deformable-object manipulation.

- Continually improve long-horizon strategy, precision, speed, and robustness through reasoning, reinforcement learning, and autonomous data collection.

The central question of the talk can therefore be stated as follows: How can the scaling success of foundation models in vision and language be translated into a general-purpose robot policy that acts reliably in the physical world?

### Research Motivation

#### From Task Specific Models to a General Foundation

Historically, vision and natural-language systems often collected high-quality labeled data and trained a separate model for each task. This workflow can produce strong results, but every new domain requires another cycle of data collection, annotation, and training. Foundation models changed this pattern: they are pretrained on massive, weakly labeled or self-supervised datasets and then adapted to downstream tasks through fine-tuning or prompting, so new applications need not begin from scratch.

Robot learning still relies heavily on the older paradigm. Teaching a particular robot to flip a pancake, open a drawer, or route a cable often requires collecting data specifically for that platform and task and then training a highly specialized policy. Changes to the robot platform, cameras, action space, or environment make both data and models difficult to reuse. As a result, the cost of data and engineering has become a major obstacle to general-purpose robotics.

| **Dimension** | **Task-Specific Robot Learning** | **Goal of a Robot Foundation Model** |
| --- | --- | --- |
| Data | Collected separately for each platform and task | Aggregated across tasks, environments, and embodiments |
| Model | A specialized policy trained from scratch for each task | A shared pretrained foundation adapted through fine-tuning or prompting |
| Knowledge | Limited to data from a single robot | Inherits internet-scale knowledge from vision-language models |
| Transfer | Difficult to reuse after the platform or task changes | Adapts to new tasks and embodiments with limited data |
| Improvement | Depends on additional human demonstrations | Combines post-training, reinforcement learning, and autonomous data collection |

#### Four Testable Hypotheses

The argument in the talk supports four testable hypotheses, although Levine did not present them as a formal four-part framework.

First, heterogeneous data may yield positive transfer. Data from different robots and tasks need not be mutually interfering noise; joint training may allow a generalist to outperform a specialist even within a specialist's own domain.

Second, internet-scale semantic knowledge can transfer to action. The linguistic, object-level, and spatial knowledge learned by a VLM can be converted into robot-control capabilities through a VLA.

Third, pretraining and post-training serve distinct functions. Broad, heterogeneous pretraining provides coverage and recovery capabilities, whereas narrow, high-quality post-training shapes a stable policy. The two are complementary rather than interchangeable.

Fourth, imitation learning is not the endpoint. Behavior cloning only fits the demonstration distribution; it does not directly optimize task success, speed, or robustness. Inference-time computation, reinforcement learning, and autonomous data loops are therefore necessary.

### Problems the Research Seeks to Solve

| **Core Problem** | **Current Bottleneck** | **Approach Presented in the Talk** | **What Remains Unresolved** |
| --- | --- | --- | --- |
| Fragmented data | Data from different laboratories, platforms, and tasks are difficult to reuse | Cross-embodiment aggregation and training through RT-X and Open X-Embodiment | No systematic rules yet govern data-mixture ratios, quality, or conflicts between embodiments |
| Translating semantics into action | VLMs output text, whereas robots require continuous, high-frequency control | RT-2 discretizes actions as tokens; π₀ adds a continuous-action expert | How to jointly achieve semantic competence, precision, low latency, and responsiveness |
| Complex dexterous manipulation | Contact-rich interaction, bimanual control, and deformable objects give rise to multimodal action distributions and compounding errors | π₀ uses flow-matching action chunks and multi-robot pretraining | Most complex tasks still require task-specific fine-tuning |
| Failure recovery | High-quality demonstrations mainly show correct trajectories and provide few recovery examples after deviations | Preserve failures, recoveries, and diverse trajectories during pretraining | How to measure recovery coverage while avoiding the acquisition of poor strategies |
| Long-horizon strategy | A single-level VLA may repeat actions, undo progress, or lose track of the overall goal | Intermediate reasoning and hierarchical policies in ECoT and Hi Robot | Memory, state tracking, planning correctness, and cascading errors |
| Precision and speed | Imitation learning does not directly optimize the final outcome | Online RL, RLPD, HIL-SERL, and distillation of RL data | Compute, safety, and reward design for end-to-end RL of large models |
| Adaptation to a new embodiment | Degrees of freedom, hand morphology, or locomotion may lie outside the training distribution | A unified action space and more flexible architectures such as CrossFormer | Adaptation data range from single-digit to triple-digit hours, with no clear governing law |
| Responsiveness in dynamic environments | Action chunking improves throughput but reduces the frequency of rapid closed-loop updates | Incremental inference and faster recomputation of action chunks | Low-level actions may execute at up to 50 Hz, but this does not imply a 50 Hz perception-to-action replanning loop |
| Autonomous scaling of data | Human teleoperation is expensive, so data scale is constrained by the number of deployed robots and the available human labor | A VLM proposes tasks, image editing generates goals, and the robot attempts them autonomously | Success detection, environment maintenance or reset requirements, safety, and self-reinforcing errors |
| World models and dynamics | Pure policies lack explicit prediction, but pixel-by-pixel prediction may be too expensive | SuSIE uses future-image subgoals for mid-level prediction | Whether prediction should occur in language, image, latent, or action space remains unsettled |

### Method 1 Cross Embodiment Data and Semantic Control

#### RT X and Open X Embodiment

RT-X first tests an empirical prerequisite for robot foundation models: can training a single model on data pooled across laboratories produce positive transfer? Open X-Embodiment pools 60 datasets from 34 robotics laboratories, spans 22 robot embodiments, and is described in the paper as a collaboration among 21 institutions. It covers hundreds of skills and many environments. The RT-X policy experiments used a narrower multi-manipulator mixture; the training and evaluation subset for the comparison described here was restricted to single-arm manipulators with parallel-jaw grippers.

The team trained RT-1-X on the aggregated data and sent the model back to several contributing laboratories. Each laboratory evaluated it on its own tasks and data against its strongest specialized method and single-domain baseline. On the small-dataset domains shown, Levine reported that RT-1-X achieved a mean success rate approximately 50% higher than the domain-specific comparators. This was an early indication that, in robotics, a generalist can outperform a specialist on the specialist's own task.

The result establishes a practical benefit from joint training, but it does not isolate the individual contributions of total data volume, model architecture, regularization, and cross-embodiment sharing. The gain was not uniform across tasks, and the smaller RT-1-X model underfit on the paper's large-data domains. The task scope was also deliberately restricted, so the result cannot be assumed to extend directly to dexterous hands, legged robots, or drones.

#### RT 2 and Early VLAs

Cross-embodiment data addresses only part of the data gap; robots must also inherit semantic knowledge from internet-scale VLMs. RT-2 discretizes robot actions into numerical tokens and reframes control as a problem resembling visual question answering: the model receives an image and a language instruction, and its "answer" is an action.

RT-2-X combines RT-2 with cross-embodiment data and can respond to instructions involving spatial relationships and object manipulation. In the displayed emergent-skills and cross-embodiment-transfer evaluation on the Google Robot, the speaker reported that RT-2-X achieved approximately three times the performance of RT-2 trained only on data from that robot. This was not presented as a threefold gain across every out-of-distribution task.

RT-2 also exposed the limitations of first-generation VLAs. Representing continuous control in a text-token space is not a robot-native design. Tasks and language remained relatively simple, while complex, long-horizon, contact-rich capabilities remained unresolved.

### Method 2 π₀ as a General Purpose Robot Policy

#### Model Architecture

π₀ aims to turn a VLM awkwardly adapted for robotics into a robot-centric general-purpose policy. It uses PaliGemma as its vision-language backbone, combining the Gemma language model with the SigLIP image encoder, and thereby inherits internet-scale visual-language knowledge through its initialization. Inputs include images from a base camera and one or two wrist cameras, a language instruction, and the robot state.

The model adds a smaller set of action-expert weights. The action expert attends to the VLM's internal representations and uses a flow-matching objective to iteratively generate continuous actions from partially noised action samples. At each inference step, it outputs a chunk of approximately 50 future action steps, allowing it to represent complex and multimodal action distributions.

Different robots are mapped into a unified action space with a fixed maximum dimensionality; action vectors for robots with fewer degrees of freedom are zero-padded. This design allows one model to train across single-arm, bimanual, stationary, and mobile platforms, but it does not explicitly encode the structural relationships among embodiments.

#### Data Recipe

| **Stage** | **Data Characteristics** | **Primary Function** | **Primary Risk** |
| --- | --- | --- | --- |
| Pretraining | Approximately 10,000 hours; large-scale, mixed-quality, multi-robot, and multitask | Learn coverage, general representations, failure recovery, and corrective behavior | Poor mixture ratios, inadequate long-tail coverage, and interference between embodiments |
| Post-training | Examples in this talk use approximately 1-20 hours; high-quality, narrow in scope, and behaviorally consistent | Shape the model into a stable policy for the target task | Data may be too narrow and omit failure and recovery states |
| Low-data adaptation | Some simple tasks require as little as 1-2 hours | Enter a new task or platform quickly | Required data vary substantially with the task and target performance |

The speaker emphasizes that the data and training recipe are at least as important as the architecture. High-quality demonstrations contain few mistakes, so a model trained only on post-training data does not know how to recover once it leaves the demonstration distribution. Errors, corrections, and environmental diversity in pretraining provide recovery capabilities; post-training then supplies stable, unambiguous, and consistent target behavior.

**Interpreting the duration figures.** Both the supplied transcript and the public video captions say "up to 20 hours" at this point. In the questions and answers, however, the speaker also notes that adaptation to a complex new embodiment may require a number of hours in the triple digits. The π₀ paper separately reports a range from approximately 5 hours to more than 100 hours under a broader framing of task complexity. These figures refer to different tasks and adaptation objectives, so 20 hours should not be treated as a universal upper bound on post-training.

#### Complex Tasks and Ablations

The talk presents tasks including cardboard-box assembly, placing eggs into a carton, scooping food, clearing a table, folding clothes, and unloading a dryer. Box assembly requires both arms to brace the box against the table, repeatedly press tabs into place, and correct mistakes using visual feedback. In the clothing-folding task, the robot begins with garments in arbitrary crumpled configurations and must retrieve, flatten, arrange, fold, and retry after failures.

The complete laundry demonstration is one continuous execution of a single policy, without policy switching or manually hard-coded task modules. At real-time speed, the sequence lasts approximately 13 minutes. The robot fails and retries several grasps along the way, but ultimately processes all of the garments.

The ablation study compares three conditions: the full pretraining-plus-post-training recipe, a model trained from scratch using only post-training data, and direct prompting of the pretrained model without task-specific post-training. The full recipe performs substantially better on box assembly, egg packing, and scooping, whereas the direct-prompting condition rarely succeeds on these complex tasks.

#### Current Capability Boundary

π₀ demonstrates a prototype general-purpose model trained across multiple platforms. It also allows different research teams to download the weights, adapt the model with relatively small datasets, and share a common pretrained baseline. Most complex tasks, however, still require dedicated fine-tuning. The model sometimes lacks high-level strategy - for example, it may unfold a garment that it has already folded. Because training relies primarily on imitation, it also does not directly optimize speed, robustness, or final task success.

### Method 3 Sequential Reasoning and Long Horizon Decision Making

#### Embodied Chain of Thought ECoT

A conventional VLA predicts actions directly from images and language, but the correct action often depends on the task plan, current subtask, object locations, and gripper position. ECoT autoregressively generates textual reasoning tokens that encode visually grounded information before producing an action, exchanging additional test-time computation for better generalization.

The method builds on OpenVLA and uses foundation models such as Gemini and SAM to synthesize training annotations. A textual reasoning trace encodes task decomposition, the current subtask, object bounding boxes, gripper coordinates, and semantic motion primitives before the model finally generates real robot actions. The reasoning structure is hand-designed, but after training the model produces both reasoning tokens and actions end to end through standard autoregressive decoding at test time.

On the benchmark displayed in the talk, the speaker reported that ECoT performed approximately 50% better than standard OpenVLA. OpenVLA has roughly 7 billion parameters, whereas RT-2-X has roughly 55 billion; nevertheless, OpenVLA with ECoT outperformed the larger RT-2-X in that evaluated setup. The comparison suggests that, on this benchmark, test-time computation and embodied intermediate representations can matter more than parameter count; it should not be generalized beyond the evaluation shown.

Intermediate reasoning also provides an interface for correction. In the talk, the robot misinterprets "pick up any non-yellow object" as an instruction to pick up a yellow object. Once a larger external VLM or a human edits the reasoning trace, the robot can complete the task. Readable reasoning helps localize errors, but it should not be treated as a complete explanation of the model's internal causal mechanism.

#### The Hi Robot Hierarchical Policy

Hi Robot divides reasoning into high- and low-level layers. A high-level policy based on PaliGemma interprets complex instructions, the scene, and new feedback from the user during execution, then converts them into atomic language commands. A low-level π₀ policy translates each atomic command into continuous actions.

In the sandwich demonstration, the high-level policy decomposes "make a roast beef and cheese sandwich" into steps for placing bread, meat, cheese, and other ingredients. When the user requests tomato midway through execution, the high-level policy updates the remaining plan. When the user says "that's it," the system completes the task with a second slice of bread. In the reported evaluation, the hierarchical approach improved both instruction accuracy and task progress across three domains: table bussing, sandwich making, and grocery shopping.

Running semantic planning and motor control at different frequencies facilitates open-ended interaction, but high-level misunderstandings propagate downstream. A hierarchical policy does not by itself provide reliable persistent memory or a complete world-state model.

### Method 4 Reinforcement Learning and Performance Optimization

#### Fast Online Reinforcement Learning on Real Robots

The motivation for reinforcement learning is to optimize task success, precision, speed, and robustness directly rather than merely imitate demonstrations. The talk presents precision tasks such as inserting components into a printed circuit board and routing a cable through a clip. Learning begins from a small number of demonstrations, after which the system collects experience and updates its policy entirely online.

In one PCB-insertion experiment, the speaker reported that the policy began to succeed after approximately 8 minutes of training, became relatively stable after approximately 10 minutes, and reached a 100% success rate in that specific setting after approximately 30 minutes. In the comparison with behavior cloning, the behavior-cloning baseline receives one additional minute of human demonstration for every minute of RL experience collected. Even under this comparison, RL achieves a higher success rate and reduces cycle time by a factor of approximately two to three.

These results rely on a small number of demonstrations, a pretrained visual encoder, and a learned binary success classifier; they are therefore not examples of learning from scratch. The 100% figure applies only to the specific task and evaluation setting, and the video does not provide a sample size or confidence interval.

#### RLPD and HIL SERL

RLPD can be understood as Soft Actor-Critic with demonstration replay: an off-policy actor-critic method initialized with prior data. The system uses a wrist camera and a pretrained visual encoder. An image classifier supplies a sparse task-success reward, without manually engineered dense reward shaping.

HIL-SERL additionally incorporates human teleoperation interventions. Each intervention both indicates that the policy's preceding behavior triggered a takeover and supplies high-quality corrective data. The method is demonstrated on bimanual alignment of six dashboard pins, assembly of a flexible timing belt, alignment of furniture parts, flipping an egg, extracting a Jenga block with a whip, and installing a PCIe SSD, RAM, and a USB cable in a computer.

The motherboard assembly uses neither explicit pose estimation nor force sensing. Apart from safety-related low-level control, the policy is trained end to end with RL using pretrained visual features. Engineers apply motor-torque clipping to prevent damage to the motherboard during contact. This illustrates that "end to end" does not mean "free of engineering priors": safety control, camera placement, a binary reward classifier, and teleoperation infrastructure remain important.

#### Distilling RL Experts into a General Purpose VLA

Running online RL directly on a very large VLA is computationally expensive. The alternative presented in the talk is to train task-specific RL experts for three connector types, use their high-precision rollouts as training data to fine-tune a pretrained general-purpose policy such as OpenVLA or Octo, and then evaluate the resulting model on previously unseen connectors.

The speaker's conclusion is that the general representations of a VLA and the high-precision specialized data produced by RL can complement one another, giving a policy both precision and generalization to new connectors. This line of work corresponds to RLDG, although the talk does not provide a numerical improvement for this segment.

### Method 5 Autonomous Data and Continual Improvement

Because human teleoperation cannot scale indefinitely, the team explores having robots generate their own tasks and data. This work corresponds to SOAR (Scaled Collection for Autonomous Improvement), developed by Zhiyuan "Paul" Zhou, Pranav Atreya, and collaborators. A VLM first observes a scene and proposes feasible, semantically specified tasks. An image-editing diffusion model then generates hypothetical goal images, and the robot attempts to reach those goals. A VLM determines whether the attempt succeeded, while a goal-conditioned policy uses hindsight goal relabeling to learn further from the autonomous trajectories.

The speaker reported that allowing a robot to "play" autonomously in a new environment substantially improves its performance in that environment, and that pooling autonomously collected data from several environments yields a larger gain. This is the part of the talk that most closely resembles a data flywheel: the foundation model not only consumes existing data, but also helps decide which experiences are worth collecting next.

Here, autonomous improvement means that collection proceeds without additional human annotations or demonstrations for each rollout; it does not mean that the system is free of prior knowledge. The reported implementation still depends on pretrained foundation models, an initial offline policy, a predefined feasible-task vocabulary, VLM-based success filtering, and hardware safeguards. Its tabletop collection continues from the current scene rather than resetting after every rollout, although broader deployments may still require environment maintenance and reset mechanisms. Errors in task proposal, goal generation, or outcome assessment can be amplified by the autonomous data loop.

### Unified Training Recipe

Taken together, the methods in the talk suggest the following training recipe.

| **Stage** | **Primary Inputs** | **Intended Capability** | **Typical Failure Mode** |
| --- | --- | --- | --- |
| Cross-embodiment pretraining | Large-scale, mixed-quality, multi-robot and multitask data, with internet-scale visual-language knowledge inherited through VLM initialization | Coverage, semantics, recovery, and shared representations | Poor data mixtures, embodiment conflicts, and inadequate long-tail coverage |
| High-quality post-training | A limited set of consistent, successful expert demonstrations | A stable policy for the target task | Overly narrow data, missing recovery states, and loss of general capabilities |
| Inference-time computation | The scene, complex instructions, intermediate plans, and localization | Long-horizon decomposition, contextual understanding, and an interface for correction | Latency, hallucinations, and cascading errors |
| Reinforcement learning | Online interaction, sparse rewards, and human interventions | Success rate, precision, speed, and robustness | Compute and sample cost, safety, and reward bias |
| Autonomous improvement | VLM task proposals, generated goals, and autonomous trajectories | Environmental adaptation and a continual data loop | Self-reinforcing errors and the cost of resets and monitoring |

This recipe resembles the pretraining-and-alignment pipeline used for LLMs and VLMs, but robots must additionally address continuous control, real-world safety, the cost of physical interaction, scene resets, and latency in a dynamic closed loop.

### Key Additions from the Questions and Answers

#### Adaptation to New Embodiments

The amount of data required to adapt π₀ to a new embodiment depends strongly on the task. Single-digit hours of fine-tuning may suffice for simple tasks, whereas complex tasks may require a number of hours in the triple digits. The speaker mentions examples of fine-tuning the model for humanoid robots, five-fingered hands, and navigation, but these are empirical observations rather than a standardized benchmark.

CrossFormer is presented as an example spanning a broader range of embodiments. It jointly models legged robots, ground robots, drones, and multiple manipulation platforms, using more flexible sequence modeling and a multi-head design to accommodate their differences.

#### The Data Bottleneck

The speaker does not reduce the bottleneck to either pretraining or post-training alone. Pretraining primarily requires quantity, diversity, and coverage; post-training primarily requires quality and behavioral consistency. Post-training data may amount to only single- or double-digit hours, but the demonstrations must show the desired behavior correctly and consistently. He also emphasizes the need for a more systematic science of data and refers to Dorsa Sadigh's team's work on data mixtures.

#### World Models and Levels of Abstraction

Levine argues that understanding the world should be an important model capability, but the key question is not simply whether to choose a model-based or model-free approach. It is instead the level of abstraction at which the system should make predictions. SuSIE does not predict the very next frame; instead, it generates a future image as a mid-level subgoal that a goal-conditioned policy then attempts to reach. A long-term system might use language or symbols for high-level planning, images, states, or latent representations for mid-level prediction, and raw actions for low-level control.

#### Mobile Robots and Dynamic Environments

The initial release of π₀ includes a small amount of mobile-manipulation data. From the policy's perspective, controlling wheels uses the same interface as controlling joints; the smaller dataset is largely a consequence of setup and safety costs. The true bottleneck in dynamic environments is the update rate of action chunks. Although the arm receives actions at up to 50 Hz, inference with the large model is slower. The system emits chunks of approximately 50 action steps and recomputes a chunk after executing about 25 steps, so it may react too slowly in a rapidly changing environment. At the time of the talk, the team was exploring incremental inference.

#### Engineering Boundaries in Precision Assembly

The motherboard component-insertion task uses neither explicit pose estimation nor force sensing, but it does use pretrained visual features and torque limits. Learning can reduce manual task decomposition; it does not eliminate safety control and hardware protection. This example shows that evaluating scalability requires examining not only the learning algorithm, but also the control, reward, reset, and intervention systems that support it.

#### Scaling Laws and the Economics of Data

The speaker sees no reason to expect the underlying learning laws for robot models to be fundamentally different from those of language models. What may differ is the economics of data. Useful robots naturally produce interaction data while performing real work, so future data scale may be determined primarily by the number of deployed robots. The breakthrough may not come from a mathematically different scaling law, but from lowering the marginal cost of data by learning while working.

### Key Results and Evidentiary Boundaries

| **Result Presented in the Talk** | **Defensible Interpretation** | **What It Should Not Be Overinterpreted As** |
| --- | --- | --- |
| RT-1-X cross-embodiment model is approximately 50% better | Mean success rate is roughly 50% higher across the evaluated small-dataset domains | A 50-percentage-point increase in success rate, or positive transfer on every task |
| Approximately 3× on the displayed emergent-skills evaluation | On the Google Robot evaluation shown, cross-embodiment data produce approximately three times the performance achieved using only that robot's data | A threefold improvement across every out-of-distribution task or reliable open-world commonsense reasoning |
| Approximately 10,000 hours of π₀ pretraining | Large-scale, mixed-quality robot data are central to the recipe | Simply increasing hours will reliably improve performance |
| Full laundry sequence lasts approximately 13 minutes | A single policy can recover repeatedly and complete a long sequence in one run | Reliability sufficient for long-term deployment |
| ECoT is approximately 50% better than OpenVLA | Intermediate reasoning and test-time computation help on the relevant tasks | Readable chains of thought are always correct or incur no real-time cost |
| Online RL reaches 100% on PCB insertion after approximately 30 minutes | It rapidly reaches the reported full success rate in the specified setting | Training from scratch, or 100% performance in new environments |
| RL reduces cycle time by a factor of approximately 2-3 | RL can optimize execution speed directly | Every task will yield the same speedup |
| New-embodiment adaptation may require single-digit to triple-digit hours | Data requirements vary widely with the task and target performance | A fixed, generally applicable fine-tuning budget exists |

### Open Questions and Research Assessment

#### Data Questions

- No predictive law yet relates pretraining scale, diversity, quality, and data-mixture ratios.

- Post-training emphasizes consistency, but systematic experiments have not established which forms of consistency matter most.

- Data requirements for new-embodiment adaptation vary widely, and no reliable estimation method exists.

- It remains unclear how autonomous data can improve a model over the long term without accumulating bias.

#### Architecture Questions

- π₀'s fixed-dimensional unified action space and action-chunking mechanism remain relatively simple cross-embodiment interfaces.

- Whether dexterous hands, legged robots, and drones should share a single action head remains an open question.

- Action chunking improves throughput at the expense of responsiveness in fast, dynamic environments.

- High-level language reasoning, mid-level world prediction, and low-level control have not yet been unified in a single architecture.

#### Post Training Questions

- End-to-end online RL for large VLAs remains expensive in both compute and samples.

- Distillation from RL experts is a viable intermediate path, but is limited by expert coverage and distillation loss.

- Reward classifiers, human interventions, low-level safety control, and reset systems may become new scaling bottlenecks.

- How to integrate reasoning, RL, world models, and autonomous data collection within a unified training framework remains the central open problem.

#### Research Assessment

Taken together, this research program provides reasonably compelling evidence that sharing data across robots and using general-purpose pretraining deliver practical value. It nevertheless remains far from "zero-shot control of any robot on any task with a single model." The decisive advances in the next stage are more likely to come from post-training, closed data loops, real-time inference, and world modeling across levels of abstraction, rather than merely from increasing parameter count.

One system-level question is worth asking repeatedly: as deployment scales, will any stage that requires human annotation, task scripts, reward engineering, safety takeovers, or scene resets become the next bottleneck? A truly scalable robot foundation model must make more of these stages data-driven and learned, while retaining the safety constraints that remain necessary.

### Glossary

| **Term** | **Meaning and Role in the Talk** |
| --- | --- |
| RT-X / RT-1-X | A family of jointly trained cross-laboratory models, and the cross-embodiment policy used in multi-laboratory comparisons |
| Open X-Embodiment | An open dataset aggregating robot trajectories from many institutions |
| RT-2 / RT-2-X | An early action-token VLA and the corresponding version trained on cross-embodiment data |
| VLM | Vision-language model; supplies visual, linguistic, and internet-scale semantic knowledge |
| VLA | Vision-language-action model; maps visual and language inputs to robot actions |
| π₀ | Physical Intelligence's general-purpose robot policy, using a continuous-action expert and flow matching |
| PaliGemma | The VLM backbone of π₀, combining Gemma and SigLIP |
| ECoT | Embodied Chain of Thought; produces embodied intermediate reasoning before actions |
| Hi Robot | A hierarchical interactive robot system combining a high-level VLM with a low-level π₀ policy |
| RLPD | Reinforcement Learning with Prior Data; off-policy RL that replays demonstration data |
| HIL-SERL | A sample-efficient real-robot RL system with real-time human interventions |
| RLDG | Reinforcement Learning Distilled Generalists; uses task-specific RL to generate high-quality data for fine-tuning generalist policies |
| SOAR | Scaled Collection for Autonomous Improvement; uses foundation models to guide task proposal, success assessment, and learning from autonomous data |
| CrossFormer | A cross-embodiment sequential policy designed for a broader range of robot morphologies |
| SuSIE | A high-level planning method that uses an image-editing diffusion model to predict future subgoals |

### Sources and Verification

- Video: U of T Robotics Institute Seminar - Sergey Levine, UC Berkeley  [https://www.youtube.com/watch?v=EYLdC3a0NHw](<https://www.youtube.com/watch?v=EYLdC3a0NHw>)

- Open X-Embodiment  [https://arxiv.org/abs/2310.08864](<https://arxiv.org/abs/2310.08864>)

- RT-2  [https://arxiv.org/abs/2307.15818](<https://arxiv.org/abs/2307.15818>)

- π₀  [https://www.physicalintelligence.company/blog/pi0](<https://www.physicalintelligence.company/blog/pi0>)

- π₀ paper  [https://arxiv.org/abs/2410.24164](<https://arxiv.org/abs/2410.24164>)

- Embodied Chain of Thought  [https://embodied-cot.github.io/](<https://embodied-cot.github.io/>)

- Hi Robot  [https://arxiv.org/abs/2502.19417](<https://arxiv.org/abs/2502.19417>)

- SERL  [https://arxiv.org/abs/2401.16013](<https://arxiv.org/abs/2401.16013>)

- HIL-SERL  [https://arxiv.org/abs/2410.21845](<https://arxiv.org/abs/2410.21845>)

- RLDG  [https://arxiv.org/abs/2412.09858](<https://arxiv.org/abs/2412.09858>)

- SOAR / Autonomous Improvement of Instruction Following Skills via Foundation Models  [https://arxiv.org/abs/2407.20635](<https://arxiv.org/abs/2407.20635>)

- CrossFormer  [https://arxiv.org/abs/2408.11812](<https://arxiv.org/abs/2408.11812>)

- SuSIE  [https://arxiv.org/abs/2310.10639](<https://arxiv.org/abs/2310.10639>)
