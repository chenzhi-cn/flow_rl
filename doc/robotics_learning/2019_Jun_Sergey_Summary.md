---
id: sergey-2019-06-summary
title: Robots That Learn By Doing
speaker: Sergey Levine
language: en
document_type: research_talk_summary
source_file: 2019_Jun_Sergey_Summary.docx
source_sha256: 034817a8f9fb7bee7d8075584af199042cc69031b21fa283d09d09fef73589cf
period_from_filename: 2019-06
talk_date: '2019-05-31'
recording_date: null
video_publication_date: null
video_url: https://www.youtube.com/watch?v=jAPJeJK18mw
keywords:
- robot learning
- continual improvement
- autonomous data collection
- reward learning
- VICE
- Skew-Fit
- QT-Opt
- maximum-entropy reinforcement learning
conversion_date: '2026-09-30'
content_scope: Full source notes, with source-derived retrieval overview and normalized Markdown formatting.
verification_scope: Converted against the supplied DOCX only. Original attribution, caveats, and references are
  retained; external sources were not re-verified during conversion.
date_note: The filename uses June 2019; the source explicitly gives 31 May 2019 as the delivery date. These are
  retained as distinct fields.
figure_handling: The source diagram is transcribed into text so this Markdown file is self-contained.
---

# Robots That Learn By Doing

## Retrieval overview

This overview is derived from the supplied summary. The full source notes follow, including their evidence qualifications and editorial interpretations.

**Research goal:** Enable robots to acquire physical competence from experience and keep improving the entire agent after deployment.

**Motivation:** Physical skills are tacit and exception-rich; manually specified rewards, practice goals, and other fixed components can limit scaling.

**Approach:** Learn task success from outcomes, generate practice goals, retain historical data through off-policy learning, and use diverse behavior to support robustness.

## Source notes

RESEARCH TALK NOTES

Motivation, the perpetual-improvement thesis, and broadly transferable research insights

**SPEAKER**  Sergey Levine · UC Berkeley

**EVENT**  L4DC: Learning for Dynamics and Control · MIT Kirsch Auditorium (32-123)

**DELIVERED**  31 May 2019 · 30:25 including Q&amp;A

**FOCUS**  Motivation · data-driven scaling · generalization · open problems

**VIDEO**  [Watch the talk on YouTube](<https://www.youtube.com/watch?v=jAPJeJK18mw>)

**CENTRAL MOTIVATION.** Physical competence is tacit and exception-rich. If humans cannot fully specify how robust skills work, robots need a way to acquire that knowledge from experience and to repair their own failures.

**ONE-SENTENCE THESIS.** A robot will generalize meaningfully only when experience can improve the entire agent—not merely the policy network—and when the loop can keep running after deployment.

### Executive takeaway

**Levine’s argument is systems-level.** Large models matter, but the decisive question is whether the complete experience-to-improvement pipeline scales. The talk attacks three obstructions: task rewards that require engineering, practice goals that require humans, and algorithms that discard expensive interaction data.

**The strongest reusable principle is a bottleneck test:** “any manual part of this loop will eventually become the bottleneck.” [\[06:56\]](<https://www.youtube.com/watch?v=jAPJeJK18mw&t=416s>) The claim is best treated as a scaling thesis, not a theorem. A hand-designed component may be an excellent bootstrap, but it must retain a path for correction from real experience.

**OPEN PROBLEM.** In Q&amp;A, Levine identifies autonomous learning in uncontrolled settings—especially safety and robustness—as the major unresolved blocker. That boundary is essential to interpreting every demonstration in the talk.

#### Talk at a glance

- **Problem.** How can robots acquire exception-rich physical competence and improve after failures?

- **Hypothesis.** High-capacity models need large, diverse, reusable datasets; autonomous agents can generate those datasets through interaction.

- **Three attacks.** Learn reward signals from outcomes; invent practice goals that expand coverage; reuse all historical experience through off-policy learning.

- **Evidence.** VICE and active queries, Skew-Fit door learning, QT-Opt grasping, and maximum-entropy recovery/locomotion.

- **Boundary.** Results are controlled and task-scoped; safe, robust continual learning in open environments remains unsolved.

### 1. Motivation: why robot learning needs a different scaling loop

#### 1.1 Tacit physical skill defeats complete hand specification

Levine begins with shoelace tying: people can execute the skill and adapt it to unusual shoes, yet often cannot provide a complete procedure. He connects this to a long-standing puzzle about expertise: competence can be real, reliable, and generalizable without being verbally decomposable into a compact algorithm.

**Speaker’s claim.** Open-world reality contains an effectively endless set of exceptions and special cases; elegant concepts such as rigid objects, inertia, or physics are useful but insufficient on their own for effortless generalization. \[00:23–02:18\]

**Research interpretation.** The target is not a better formula for one task. It is a mechanism for absorbing tacit, exception-rich knowledge that engineers cannot enumerate in advance.

#### 1.2 Generalization requires capacity and coverage

Levine draws an analogy to computer vision, speech, and language. Their success is not explained by neural-network capacity alone; it comes from high-capacity models trained on large amounts of diverse data. For acting agents, this creates two coupled problems: where the interaction dataset and labels come from, and how a deployed system repairs a failure on a previously unseen case. \[02:18–04:20\]

**GENERAL INSIGHT.** Generalization is partly a coverage problem and partly a correction problem. A model must see enough of the world, and deployment failures must create a route to better data rather than a permanent blind spot.

#### 1.3 Autonomy is a data advantage

An embodied agent can act, observe the consequences, and collect additional experience where its current model is weak. This turns mistakes from terminal defects into prompts for targeted data acquisition. Levine’s long-run objective is therefore perpetual improvement through interaction—not the elimination of every failure before deployment. \[04:20–06:11\]

1. **Tacit skill.** Useful behavior cannot always be reduced to rules humans can articulate.

2. **Exception-rich world.** Open environments continually produce cases outside compact abstractions.

3. **Data requirement.** Large-capacity policies generalize only when experience covers relevant variation.

4. **Autonomous correction.** Acting agents can seek new experience around their own failures.

### 2. The perpetual-improvement and bottleneck principle

The unit of scale is the full learning loop. The robot acts, stores experience, updates its model or policy, and returns to the world. If every component becomes better as experience accumulates, the whole system can improve. If one component is fixed, the rest of the system eventually runs into its ceiling. \[06:12–07:22\]

**Figure transcription — perpetual improvement loop**

1. **Act in the world:** Observe outcomes and failure.
2. **Store experience:** Retain diverse historical data.
3. **Update the agent:** Model, policy, and evaluator.
4. **Deploy and correct:** Target data at current weaknesses.

The arrows form a repeating cycle: after deployment and correction, the agent acts in the world again.

**Scaling test:** A fixed component that cannot improve from data becomes the eventual bottleneck.

Figure 1. Levine’s scaling thesis, reconstructed from the talk.

*Table 1. Useful engineering choices that become scaling ceilings when they lack a real-data correction path.*

| **Component** | **Initial value** | **Long-run limit** | **Required escape hatch** |
| --- | --- | --- | --- |
| Hand-designed state | Rigid-body poses simplify learning. | Deformables and unmodeled exceptions are invisible. | Learn from raw or richer sensing; permit representation updates from experience. |
| Instrumented reward | A wired door sensor gives a clean success signal. | The robot cannot assess or improve on ordinary uninstrumented doors. | Learn success from onboard observations and query humans selectively. |
| Simulation-only policy | Fast prototyping, safe trials, strong initialization. | Performance is bounded by simulator fidelity and coverage. | Use simulation as prior data, then adapt with target-domain experience. |

**IMPORTANT NUANCE.** The talk does not argue that simulation or structured representations are useless. They can be excellent initializations. The objection is to making them the only path by which performance can improve.

### 3. Three research programs that unblock the data pipeline

#### 3.1 Learn task success instead of engineering the reward

**Motivation.** Games expose a score; the physical world usually does not. Even deciding whether water was poured successfully can require a sophisticated perception system—meaning the hardest sensing problem may need to be solved before RL begins. \[10:26–11:37\]

**Naive failure mode.** A success/failure image classifier seems general, but the policy actively searches for states that fool it. In the talk’s example, the robot points its fingers toward the camera and receives a near-perfect predicted reward without completing the task. This is policy-induced proxy exploitation, not merely a weak held-out classifier. \[11:37–12:48\]

**Mechanism: VICE plus active queries.** Variational Inverse Control with Events alternates between a learned success classifier and the current policy, adding policy-generated states as hard negatives. A later robotic system asks a human to label only uncertain outcomes. Both reward model and policy operate directly on images. \[12:48–14:01\]

**Evidence.** The talk shows cup-to-coaster pushing learned in about 1 hour 15 minutes and cloth draping where visual feedback is necessary. The associated RSS 2019 paper reports image-based manipulation tasks learned with modest outcome examples, sparse active labels, and roughly 1–4 hours of real interaction.

**TRANSFERABLE INSIGHT.** When an optimizer acts against a learned evaluator, it changes the evaluator’s input distribution. Reward models need on-policy hard negatives, uncertainty-aware queries, or another mechanism that keeps the evaluator synchronized with the evolving policy.

**Limit.** The method reduces reward engineering; it does not eliminate supervision. It still needs successful outcome examples, occasional labels, and a controlled exploration envelope.

#### 3.2 Let the robot invent practice goals that expand coverage

**Motivation.** Outcome examples still leave a human in the training loop. Levine asks whether a robot can practice before anyone specifies the downstream task, then respond at test time to a desired goal image. \[16:01–17:37\]

**Collapse problem.** A latent image model can imagine goals, but if it is fitted to the robot’s current experience it will mostly resample familiar states. The agent practices what it already knows instead of expanding its repertoire.

**Mechanism: coverage-driven goal sampling.** Reweight visited states by inverse local density, fit the goal model to that skewed distribution, and preferentially sample frontier states. The later paper names this approach Skew-Fit. Under regularity/idealized conditions, it converges toward uniform coverage of valid states. \[17:37–18:42\]

**Evidence.** Placed in front of a door without a door-opening reward, the real robot begins with random motion and, after about 25 hours, discovers door opening. At test time it can use a target image to move the door to requested angles using pixels alone. \[18:42–19:48\]

**TRANSFERABLE INSIGHT.** Self-generated curricula require explicit coverage pressure. Without a preference for reachable but underrepresented states, autonomous practice collapses into rehearsal of already-mastered behavior.

**Limit.** The guarantee is assumption-dependent, and the physical demonstration is one constrained robot–door setup. It is a proof of self-supervised skill discovery, not broad open-world autonomy.

#### 3.3 Make past experience compound through off-policy learning

**Motivation.** Real interaction is expensive. Broad generalization therefore requires algorithms that can repeatedly reuse all historical experience and return to the world only where performance needs correction. \[19:59–20:56\]

**Mechanism: scaled fitted Q-learning.** QT-Opt combines historical and online replay buffers, distributed Bellman-target computation, and a visual Q-function. The talk describes roughly two months of data, about 1,000 training objects, about 600,000 grasp attempts, and a 1.2-million-parameter network. The paper reports more than 580,000 real grasps; the talk rounds this figure. \[20:56–23:22\]

**Evidence.** With an over-the-shoulder monocular RGB camera, the policy continuously corrects, repositions, and regrasps rather than committing to one geometric grasp. It reaches 96% success on unseen test objects in the controlled grasping evaluation. \[23:22–24:45\]

**TRANSFERABLE INSIGHT.** The value of data is not only its collection scale. Off-policy replay lets old interactions acquire new value as the policy, objective, and optimization procedure improve.

**Q&amp;A qualification.** Levine says he does not know whether QT-Opt outperforms a strong vision-plus-planning alternative and concedes that grasping may not be RL’s killer application. The scientific signal is scaled learned feedback and correction, not universal superiority over model-based grasping. \[27:57–29:22\]

### 4. Maximum-entropy RL: diversity as a route to local robustness

Standard control concentrates on the single highest-valued action. Maximum-entropy RL instead rewards task success while maintaining a distribution over multiple high-value—and occasionally slightly lower-value—actions. Levine argues that this diversity exposes the policy to alternative paths and recoveries, improving resilience to modest perturbations. \[24:45–26:12\]

- **Lego stacking.** A learned policy recovers after a person perturbs the robot, despite no explicit recovery demonstration for that exact disruption.

- **Real locomotion.** A Minitaur learns forward locomotion from scratch in about two hours without a model or simulation.

- **Boundary.** The robot handles a novel ramp and descending stairs but struggles to ascend; the associated paper claims robustness to moderate variations, not open-world invariance.

**GENERAL INSIGHT.** Robustness can come from preserving multiple competent behaviors rather than optimizing one brittle nominal trajectory. Entropy is useful only when it produces meaningful behavioral diversity under a trustworthy reward.

### 5. Broadly transferable insights

1. **The complete improvement loop is the scaling unit.** A better policy architecture is insufficient if reward sensing, data collection, resets, or evaluation cannot improve with experience.

2. **Generalization is coverage plus correction.** Diverse training experience addresses the long tail; targeted post-failure data prevents the remaining tail from becoming permanent.

3. **Autonomy changes data economics.** An embodied agent can collect the examples its current model most needs, rather than waiting for a static dataset to anticipate every exception.

4. **Every fixed component needs an escape hatch.** Representations, simulators, rewards, and priors can bootstrap learning, but real experience must be able to override their mistakes.

5. **Optimizers expose proxy defects.** A learned reward that looks accurate on static data can fail once a policy deliberately searches for its blind spots.

6. **Spend human supervision where information value is high.** Uncertainty-driven queries scale better than labeling every state or engineering a new sensor for each task.

7. **Autonomous curricula must expand support.** Novelty or density pressure is necessary to prevent self-generated practice from collapsing to familiar behavior.

8. **Historical experience should compound.** Off-policy replay permits many improvements per physical interaction and keeps past failures useful after the policy changes.

9. **Feedback can matter more than one-shot precision.** Closed-loop correction, regrasping, and recovery may be more scalable than requiring a perfect initial plan.

10. **Safety in uncontrolled settings is part of the learning algorithm.** If autonomous data collection cannot run safely outside a curated setup, the perpetual-improvement loop is broken at its source.

### 6. Evidence boundaries and unresolved problems

The demonstrations are best read as partial attacks on one systems thesis. They do not establish a finished route to general-purpose robots. The table below prevents the strongest headline numbers from being detached from their evaluation scope.

*Table 2. What the talk’s evidence supports—and what it does not yet establish.*

| **Claim area** | **Supported by the talk** | **Not established** |
| --- | --- | --- |
| Reward learning | Outcome examples + sparse active queries can replace manually programmed rewards on several image-based manipulation tasks. | Reward reliability under arbitrary deployment shift or no-human continual learning. |
| Skew-Fit | Coverage-driven goals can discover door opening from pixels in one controlled setup; theory holds under regularity conditions. | Semantic, multi-environment skill discovery or open-world coverage. |
| QT-Opt | Scaled off-policy learning produces closed-loop grasp correction and 96% success on held-out objects in its test setup. | General household manipulation or superiority to every model-based alternative. |
| Max-entropy RL | Diverse policies recover from moderate local perturbations and learn real locomotion in about two hours. | Universal robustness, safe exploration, or large distribution-shift tolerance. |
| Perpetual improvement | The loop is a compelling design principle and organizes several successful systems. | Safe, stable, autonomous improvement in uncontrolled settings—the problem Levine explicitly leaves open. |

#### 6.1 The candid open problem from Q&amp;A

When challenged that the videos remain controlled and more complex environments demand more data, Levine agrees. He identifies learning in uncontrolled settings as a major unresolved bottleneck and names safety and robustness as large open problems. \[29:30–30:15\]

**WHY THIS MATTERS.** Safe self-directed data acquisition is not a deployment afterthought. It is the precondition for the very data flywheel that the talk proposes.

### 7. Researcher’s synthesis for general robot learning

**This section is interpretation, not a claim that Levine states verbatim in the 2019 talk.** The most useful way to apply the argument is as a bottleneck audit for any proposed generalist robot-learning system.

- **Representation.** Can real failures update what the system notices, or is perception locked to a handcrafted state?

- **Objective.** Can success be recognized from onboard data without adding task-specific instrumentation?

- **Curriculum.** Does autonomous practice expand state and task coverage, or repeat familiar behavior?

- **Memory.** Can all historical interactions be reused after policies, rewards, or goals change?

- **Safety.** Can the agent gather corrective data in less controlled settings without unacceptable damage or intervention?

- **Evaluation.** Does the benchmark measure continued improvement and recovery, not merely one fixed-policy success rate?

#### 7.1 Connection to the Bitter Lesson

Levine does not name the Bitter Lesson in this talk. However, the logic closely parallels it: scalable learning and computation eventually outperform systems whose critical competence is encoded by hand. In a 2025 essay, Levine later states essentially the same corollary explicitly and extends it to simulators, surrogate data, hybrid systems, constraints, and architectural biases. [Read Levine’s later essay](<https://sergeylevine.substack.com/p/sporks-of-agi>). This later source is context, not evidence that the 2019 talk solved the open-world problem.

#### 7.2 Highest-value research opportunity implied by the talk

**RESEARCH OPPORTUNITY.** Build a safe, failure-driven data-acquisition loop for uncontrolled environments: detect when the current policy is weak, choose informative low-risk interactions, preserve all resulting experience, and demonstrate sustained improvement without task-specific reward instrumentation.

Why this opportunity is attractive: it directly attacks the bottleneck Levine names, combines reward learning, exploration, replay, and safety into one measurable loop, and can be evaluated by whether the system continues improving as environmental diversity grows.

### 8. Timestamped map of the talk

| **Timestamp** | **Topic** | **Why it matters** |
| --- | --- | --- |
| 00:23–02:18 | Tacit skill and exceptions | Why physical competence resists complete hand specification. |
| 02:18–04:20 | Capacity + data | Generalization needs large models and broad data; failures need a repair path. |
| 04:20–07:22 | Perpetual improvement | Autonomy enables corrective data; fixed components become bottlenecks. |
| 07:23–09:25 | Three fixed bottlenecks | State abstractions, instrumented rewards, and simulation-only control. |
| 09:56–15:55 | VICE and active queries | Learn success from images while adapting the evaluator to policy-induced data. |
| 16:01–19:48 | Skew-Fit / imagined goals | Coverage pressure creates an autonomous practice curriculum. |
| 19:59–24:45 | Off-policy reuse / QT-Opt | Historical interaction compounds; closed-loop correction emerges at scale. |
| 24:45–27:24 | Maximum-entropy RL | Behavioral diversity supports moderate recovery and robustness. |
| 27:24–27:51 | Conclusion | Remove obstructions from the experience-to-improvement pipeline. |
| 27:57–30:15 | Q&amp;A | Grasping superiority is unresolved; uncontrolled learning, safety, and robustness remain open. |

*Source: human-edited English caption track from the linked YouTube video; timestamps rounded to readable ranges.*

### 9. Source and reading list

The main notes follow the talk’s human-edited English caption track. Primary papers are used to verify method names, result scope, and qualifications.

**1.** **Sergey Levine — Robots That Learn By Doing.** Video, MIT Institute for Data, Systems, and Society, 31 May 2019 [Open source](<https://www.youtube.com/watch?v=jAPJeJK18mw>)

**2.** **L4DC 2019 agenda.** Official event date, venue, and talk listing [Open source](<https://l4dc.lids.mit.edu/agenda/>)

**3.** **Variational Inverse Control with Events.** Fu et al., NeurIPS 2018; data-driven reward definition from goal-state examples [Open source](<https://arxiv.org/abs/1805.11686>)

**4.** **End-to-End Robotic Reinforcement Learning without Reward Engineering.** Singh et al., RSS 2019; outcome examples plus active queries [Open source](<https://arxiv.org/abs/1904.07854>)

**5.** **Skew-Fit: State-Covering Self-Supervised Reinforcement Learning.** Pong et al., ICML 2020; maximum-entropy goal distributions and door learning [Open source](<https://arxiv.org/abs/1903.03698>)

**6.** **QT-Opt: Scalable Deep Reinforcement Learning for Vision-Based Robotic Manipulation.** Kalashnikov et al., CoRL 2018; 580k+ grasps and closed-loop visual control [Open source](<https://arxiv.org/abs/1806.10293>)

**7.** **Soft Actor-Critic.** Haarnoja et al., ICML 2018; off-policy maximum-entropy RL [Open source](<https://arxiv.org/abs/1801.01290>)

**8.** **Learning to Walk via Deep Reinforcement Learning.** Haarnoja et al., RSS 2019; real-world Minitaur locomotion in about two hours [Open source](<https://arxiv.org/abs/1812.11103>)

**9.** **CAD2RL: Real Single-Image Flight without a Single Real Image.** Sadeghi and Levine; simulation/domain randomization as strong initialization [Open source](<https://arxiv.org/abs/1611.04201>)

**10.** **Sporks of AGI.** Sergey Levine, 2025; later first-person statement connecting the bottleneck corollary to the Bitter Lesson [Open source](<https://sergeylevine.substack.com/p/sporks-of-agi>)

#### Editorial note

*Speaker claims and researcher interpretation are labeled separately. Numbers are scoped to the evaluations described in the talk or cited papers. No claim in this document should be read as evidence that general-purpose autonomous robot learning was solved by the 2019 systems.*
