---
id: sergey-2026-08-summary
title: Diffusion Models for Continuous Control
speaker: Sergey Levine
language: en
document_type: research_talk_summary
source_file: 2026_Aug_Sergey_Summary.docx
source_sha256: 1bc50d2014337b2f130eb979d592e755f346ba85dafb79717aa2602ef050e16c
period_from_filename: 2026-08
talk_date: null
recording_date: null
video_publication_date: null
video_url: https://www.youtube.com/watch?v=agi3xLTGyaU
keywords:
- diffusion policy
- flow matching
- continuous control
- multimodal action distributions
- action chunking
- vision-language-action models
- knowledge insulation
- diffusion steering
- latent-space reinforcement learning
- offline-to-online reinforcement learning
conversion_date: '2026-09-30'
content_scope: Full source notes, with source-derived retrieval overview and normalized Markdown formatting.
verification_scope: Converted against the supplied DOCX only. Original attribution, caveats, and references are
  retained; external sources were not re-verified during conversion.
date_note: August 2026 comes from the source filename. The summary does not state an exact talk or publication date.
---

# Diffusion Models for Continuous Control

## Retrieval overview

This overview is derived from the supplied summary. The full source notes follow, including their evidence qualifications and editorial interpretations.

**Research goal:** Use expressive action models to learn dexterous behavior from diverse prior data, connect control to pretrained vision-language models, and improve policies through interaction.

**Motivation:** Demonstrations contain multiple valid behaviors that a simple unimodal policy can average into invalid actions; reinforcement learning must also manage distribution shift and unsupported value estimates.

**Approach:** Study diffusion and flow action chunks, continuous action experts, knowledge insulation, and reinforcement-learning-based diffusion steering, with explicit boundaries on what the evidence establishes.

## Source notes

Research notes on Sergey Levine’s talk

### Research goal and central motivation

The research goal is to build robots that can learn dexterous, general-purpose behavior from diverse prior data and then improve through interaction. The talk examines how diffusion and flow-based action models help at three stages: representing demonstrated behavior, connecting robot control to pretrained vision-language models, and improving a policy with reinforcement learning. Levine presents these as empirical case studies rather than a unified theory of why diffusion works. \[T\]

The central insight is that a policy must be suitable for learning from the available data, even when an ideal optimal policy could be deterministic. Real demonstrations contain multiple valid actions, different strategies, and varying skill levels. An expressive action distribution can preserve these alternatives. This matters both when imitating the data and when restricting reinforcement learning to actions for which the data provide useful evidence.

#### Why a simple Gaussian can be inadequate

A single Gaussian is a convenient policy model because a network can predict its mean and covariance. Its mean can also act as a deterministic controller. However, the existence of a deterministic optimal policy in standard fully observed Markov decision process settings does not imply that a unimodal model is sufficient for learning from heterogeneous demonstrations.

The talk’s example is a drone flying around a tree. Going left and going right are both valid behaviors; averaging the two can direct the drone into the tree. The failure is not insufficient randomness. It is the inability of a simple distribution to represent separated valid modes without assigning probability to unsuitable actions between them. \[T\]

#### The problems the talk connects

| **Research setting** | **Problem to solve** | **Role of an expressive action model** |
| --- | --- | --- |
| Imitation learning | Preserve multiple valid behaviors and coherent motion | Generate multimodal sequences of continuous actions |
| Robot foundation models | Turn broad visual and language knowledge into dexterous control | Add a continuous action expert to a pretrained backbone |
| Offline to online RL | Improve on prior data without destabilizing the learned policy | Keep policy improvement near plausible behaviors and support later adaptation |

The broad motivation is data reuse. A general robot should benefit from other tasks, other robots, web pretraining, and its own subsequent experience. Diffusion provides a useful action representation within that program; it does not by itself supply reliable perception, rewards, exploration, or generalization.

### From trajectory generation to diffusion policies

#### What the policy predicts

The robot policy receives observations such as camera images, language instructions, and proprioceptive measurements. It predicts an action chunk, a short sequence of future control commands. In the opening example, the robot uses three camera views and joint readings to predict 50 action steps, approximately one second of motion. It executes 25 steps before recomputing the plan. These are the settings of that example, not a universal requirement. \[T\]

Diffusion and flow matching turn noise into a structured continuous action sample through learned iterative refinement. In this talk, “diffusion” often serves as an umbrella term for these related generative approaches. The output being generated is an action sequence, not an image. The policy can represent distinct ways to complete the same task without averaging them into one action sequence.

#### Diffuser and Diffusion Policy

Diffuser applies a generative model to whole trajectories, including states and actions. A trajectory is arranged as a matrix of time steps and state or action coordinates. Conditioning the initial state through inpainting anchors the generated future; guidance can express rewards, goals, or constraints. The generated trajectory then supports planning or control. \[1\]

Diffusion Policy instead learns actions conditioned on observations. It produces an action chunk directly, which makes the control interface simpler than generating full state-action trajectories. The original work includes convolutional and transformer variants. Modern systems discussed in the talk use diffusion-transformer-style blocks with attention to an observation encoder or to a vision-language backbone. \[2, T\]

| **Method** | **Generated object** | **How the task enters** | **Main use in the talk** |
| --- | --- | --- | --- |
| Diffuser | State-action trajectory | Inpainting and guidance | Planning through trajectory generation |
| Diffusion Policy | Action chunk | Observation conditioning | Direct visuomotor policy learning |
| VLA with action expert | Continuous action chunk | Images and language through a pretrained backbone | General-purpose robot control |

#### Why chunking matters

Levine reports that action chunking is especially important for the imitation-learning systems discussed, and can also help reinforcement learning. Executing only the first action and immediately replanning does not recover the same performance in his examples. The benefit therefore appears to involve execution as well as training. \[T\]

The talk does not establish a complete causal explanation. Temporal coherence and reduced switching between behavioral modes are plausible interpretations, but they should not be treated as demonstrated findings here. Chunking also creates a tradeoff: executing more actions without new feedback can improve coherence while delaying the response to an unexpected disturbance.

### Robot foundation models and continuous action experts

#### Motivation for generalist training

A general-purpose robot must understand varied instructions and operate across many tasks and environments. The foundation-model recipe combines broad robot data with a model pretrained on images and language. Robot data teach how to act; web pretraining supplies useful perceptual and semantic representations. \[T\]

The RT-X example motivates pooling robot data. The talk describes contributions from 34 research labs and 22 robot types, followed by comparisons with local specialist systems. Levine reports mean success rates of about 41 percent for the specialist baselines and 63 percent for the generalist. That is a 22 percentage-point increase, or approximately 54 percent relative improvement using the rounded figures. He explicitly cautions that averaging heterogeneous laboratory metrics is an imperfect statistical comparison. \[3, T\]

The result supports positive transfer in the evaluated settings. It does not establish that a generalist always beats a specialist or that arbitrary embodiments are interchangeable. The single-arm, parallel-gripper commonality described in the talk concerns the examples shown; the broader Open X-Embodiment collection also includes other embodiments. \[3\]

#### From RT 2 to a continuous motor interface

Early vision-language-action models extended next-token prediction to actions. RT-2 represented robot actions as tokens and trained with both robot examples and vision-language examples. This allowed visual and semantic knowledge to influence action selection, but the early demonstrations had limited motor complexity. The talk’s description of ASCII number strings refers to an early implementation anecdote; tokenized actions should not all be equated with literal text strings. \[4, T\]

The next design adds a dedicated continuous action expert. A pretrained vision-language backbone processes images and language, while a diffusion or flow-based module generates high-dimensional, high-frequency action chunks. Levine describes this informally as adding a “motor cortex.” It is an architectural analogy, not a biological claim. \[T\]

The π0 family illustrates this approach. Its action expert can attend to representations from the vision-language backbone, allowing semantic understanding and continuous motor control to cooperate. The talk shows clothing folding, mobile laundry handling, recovery after objects are moved, and a later π0.7 example with more detailed instructions. These demonstrations illustrate capability, but the videos alone do not isolate which architectural component causes the improvement. \[5, 10, T\]

Levine’s observation that diffusion policies often recover well from perturbations is explicitly empirical and partly anecdotal. He does not claim that accumulated error disappears; in the final discussion he says it remains a problem, though less severe than he expected.

### Knowledge insulation and efficient VLA training

#### The optimization problem

Attaching an action expert and training the entire model end to end is a natural starting point. The talk identifies two practical problems with this recipe: continuous diffusion or flow training can converge much more slowly than discrete-action training, and action-expert gradients can degrade the backbone’s language-following behavior. Levine reports a training-speed difference of roughly five to ten times in the comparisons discussed. \[T\]

This matters scientifically as well as economically. With a fixed compute budget, slower optimization reduces the amount of useful learning that can be achieved. A larger model or more training data are less helpful if the training objective cannot use the available compute efficiently.

#### The method

Knowledge insulation trains the model with two action-related objectives. A discrete-action objective adapts the backbone’s representations to robot control. A continuous objective trains the action expert, while a stop-gradient boundary prevents this second objective from updating the backbone. An attention mask prevents the action expert from reading the target action tokens. \[6, T\]

The backbone is not frozen. It continues to learn through the discrete-action objective and any applicable backbone objectives. What is blocked is the gradient route from the continuous action expert into the backbone. This distinction resolves an ambiguity in the spoken explanation.

| **Component** | **Training signal** | **Intended role** |
| --- | --- | --- |
| Vision-language backbone | Discrete-action prediction | Adapt perception and instruction representations efficiently |
| Continuous action expert | Diffusion or flow objective | Learn dexterous continuous action chunks |
| Stop-gradient boundary | Blocks expert-to-backbone gradients | Reduce harmful interference between objectives |
| Attention mask | Hides target action tokens from the expert | Prevent target leakage during training |

#### Evidence and the open question

In the plots described, knowledge insulation approaches the training speed of discrete-action learning on an easier task, while retaining a substantial final-performance advantage on harder tasks that benefit from continuous control. Levine cites about a fivefold training-speed improvement over direct end-to-end diffusion training in that comparison. \[T\]

His interpretation is deliberately cautious: the workaround exposes a training problem. It does not show that discrete and continuous objectives are the final architectural answer. A central open question is how to let the continuous-control objective improve the backbone without causing representation damage or sacrificing convergence speed. The reported multipliers should not be assumed to measure the same quantity across studies: training steps, wall-clock time, and cost are different metrics.

### Why offline to online reinforcement learning is difficult

#### Goal and basic mechanism

The goal is to initialize from useful prior experience and then improve through interaction. Offline reinforcement learning should extract good decisions from mixed-quality data, rather than merely reproduce the average demonstration. Online learning should retain that competence while discovering improvements. This is especially attractive in robotics, where starting with random movements is expensive and unproductive. \[T\]

A Q-function estimates expected future return after taking an action in a state and then following a policy. Policy iteration alternates value estimation and policy improvement. Q-learning uses a recursive Bellman target that combines immediate reward and the estimated value of future actions. Under the appropriate assumptions and accurate estimation, greedy improvement is well motivated. Neural function approximation on a fixed dataset makes those assumptions difficult to satisfy.

#### Distribution shift and optimistic errors

The critic is fitted to data collected by a behavior policy, while the improved actor chooses different actions. Maximizing an imperfect Q-function can select actions whose predicted value is high because the critic is wrong. Those errors also enter the bootstrap targets used to train the critic. The policy thus actively seeks the critic’s mistakes, rather than encountering prediction errors at random. \[T\]

One response is pessimism. Conservative methods lower values for unsupported or adversarially selected actions, with a compensating term for dataset actions to avoid excessive downward bias near observed behavior. This can make offline learning useful. It is not equivalent to discovering the true values of actions that were never adequately observed. \[T\]

#### The transition problem

The talk shows policies that improve during offline training but lose performance after online updates begin. Levine’s explanation is that exploration reaches regions where a pessimistic critic has large underestimation errors. Correcting those errors can produce large updates that destabilize the critic and policy. The observed dip can consume much of the practical benefit of offline pretraining. This is a failure mode in the methods and experiments discussed, not a theorem that all pessimistic algorithms must collapse. \[T\]

RLPD provides a strong comparison: start online learning without a separate offline pretraining phase, and draw part of each training batch from prior data. The talk describes a half-prior, half-online batch mixture. Its strength motivates a demanding test for a proposed offline-to-online method: does actual pretraining improve the subsequent learning curve over simply reusing prior data during online training? \[7, T\]

#### Why policy expressiveness changes the alternative

Another approach constrains the actor to remain near the behavior distribution, instead of imposing pessimism directly on the critic. With a simple Gaussian actor and multimodal data, the actor may either average incompatible modes or retain only one mode. An expressive policy can represent more of the useful action distribution, making the constraint more compatible with learning good behavior. \[T\]

The remaining challenge is policy extraction: a diffusion model is naturally trained to reproduce data, whereas an RL actor must prefer high-value actions. The talk next gives one simple way to connect these objectives.

### Diffusion steering and learning from experience

#### Diffusion steering

Diffusion steering first trains a conditional diffusion policy by imitation. The learned sampler maps noise and the current observation to a plausible action. Reinforcement learning then trains a separate policy to choose the sampler’s noise input, while the pretrained action generator remains fixed. \[8, T\]

The optimization takes place in the generator’s latent input space. The generator provides a learned map from that simpler space into structured behavior, reducing the need for the critic and actor to reason directly over arbitrary robot actions. The method discussed uses soft actor-critic machinery in the latent space. For offline learning, the DSRL paper additionally addresses the absence of noise labels in ordinary action datasets through noise aliasing and linked action-space and latent-space value learning. The talk compresses these implementation details; a frozen generator alone is not the complete offline algorithm. \[8\]

Levine reports a real-robot example that learns to place a mushroom-shaped object on a cloth in roughly 50 trials, together with strong results on offline tasks involving mazes, puzzles, and long horizons. These examples support the usefulness of the approach in the evaluated settings. They do not imply that any pretrained generator makes ordinary online RL reliable on arbitrary offline data. \[T\]

#### What the distribution argument does and does not establish

The useful intuition is that the generator biases learning toward actions represented in its training data. A learned generator does not mathematically guarantee that every possible latent input yields an in-distribution action. The standard Gaussian has unbounded support; bounding the steering input restricts a known latent region, rather than creating a proof of action-space support. Generator error, state distribution shift, and critic error can still matter.

A fixed generator also restricts exploration. It can select and recombine behavior that it represents, but may be unable to express an entirely new motor strategy. The audience raises this limitation directly. Levine mentions residual action corrections as a way to permit bounded changes beyond the base generator. His connection to Wasserstein distance is an informal interpretation, not a claim that the cited residual method directly optimizes that distance. The likely paper is EXPO. \[12, T\]

#### Combining a VLA with reinforcement learning

The closing π\*0.6 example combines a VLA with value-informed policy training. The talk describes adding information about action quality to the conditioning, then requesting better actions at inference. The associated RECAP approach uses a binarized advantage or improvement indicator derived using a learned value function. Advantage expresses action quality relative to a baseline value; the conditioning is more specific than pasting an arbitrary raw Q-value into a prompt. \[9, T\]

The system uses prior robot experience and additional experience from deployment to improve behavior. Levine shows coffee preparation over a 13-hour run and box assembly at Dandelion Chocolate. The coffee demonstration includes occasional failures and recovery. It is evidence of sustained operation, not 13 hours of error-free autonomy.

The quantitative comparison emphasizes successful task completions per hour. The online RL phase improves throughput over the best offline result in the displayed tasks, but the supplied transcript does not give exact bar values. Throughput is useful because it rewards both speed and successful completion; success rate alone can hide very slow execution. \[T\]

#### More computation at inference

In the final discussion, Levine describes sampling several actions, evaluating them with a Q-function, and executing the best candidate. This best-of-N strategy can be combined with diffusion steering. It is an analogue of test-time computation, not evidence that the robot performs explicit reasoning. Its value depends on the critic ranking candidates accurately and on whether the extra inference cost fits the control loop. \[T\]

The talk also points to CFGRL, which connects diffusion guidance with policy improvement. Levine skips its derivation; it is a separate policy-extraction approach from DSRL’s learned noise policy. \[11, T\]

### Evidence boundaries and remaining research problems

The talk’s strongest common message is that action-distribution expressiveness changes what learning algorithms can usefully do with existing data. Its case studies support practical progress in multimodal imitation, continuous action generation from foundation models, and offline-to-online improvement. They leave several mechanisms and scaling questions unresolved.

| **Question** | **What the talk establishes** | **What remains unresolved** |
| --- | --- | --- |
| Does chunking help? | Strong empirical benefit in the discussed systems | Why it helps and how to choose the execution horizon |
| Can generalist data help specialists? | Positive transfer in the RT-X comparisons | When task or embodiment diversity causes negative transfer |
| Can continuous action experts train efficiently? | Knowledge insulation improves the displayed comparisons | A direct objective that is fast and preserves useful backbone representations |
| Can offline competence survive online learning? | Expressive policies and steering improve the presented results | Reliability across datasets, critics, tasks, and exploration conditions |
| Can a fixed generator discover new skills? | It can steer toward represented useful behavior | How to expand behavior coverage without losing the benefits of prior data |
| Does diffusion eliminate error accumulation? | Recovery is encouraging in demonstrations | Quantitative long-horizon reliability and the causes of failure |

#### Implications for a robotics learning research agenda

An interpretation of these case studies is that the action model is also an interface for optimization. It determines which behaviors an imitation objective can preserve, what a critic can safely compare, and how an RL algorithm can change behavior. A research contribution should therefore distinguish better representation from better data, more compute, improved optimization, and easier evaluation conditions.

Three experiments would help resolve the specific uncertainties raised here. First, compare single-action and chunked policies while controlling the prediction horizon, execution horizon, feedback rate, and compute. Second, compare end-to-end continuous training, discrete training, joint training, and knowledge insulation under equal compute budgets, measuring both task completion and language following. Third, compare offline pretraining followed by online adaptation against RLPD-style prior-data reuse, tracking the initial performance dip, interaction cost, final return, and successful completions per hour.

These are research implications of the talk, not new results or claims of novelty. The motivation is to identify when expressive action models make learning more effective, and when apparent gains depend on a particular dataset or training recipe.

### References

\[T\] Sergey Levine. Diffusion in RL and robotics: how expressive policies changed how we use continuous actions. Video and accompanying transcript. Video: [https://www.youtube.com/watch?v=agi3xLTGyaU](<https://www.youtube.com/watch?v=agi3xLTGyaU>) . The supplied playback offset is 30:11. These notes cover the talk from the opening through the final questions.

\[1\] Janner, M., Du, Y., Tenenbaum, J. B., &amp; Levine, S. (2022). Planning with Diffusion for Flexible Behavior Synthesis. [https://arxiv.org/abs/2205.09991](<https://arxiv.org/abs/2205.09991>)

\[2\] Chi, C., et al. (2023). Diffusion Policy: Visuomotor Policy Learning via Action Diffusion. [https://arxiv.org/abs/2303.04137](<https://arxiv.org/abs/2303.04137>)

\[3\] Open X-Embodiment Collaboration. (2023). Open X-Embodiment: Robotic Learning Datasets and RT-X Models. [https://arxiv.org/abs/2310.08864](<https://arxiv.org/abs/2310.08864>)

\[4\] Brohan, A., et al. (2023). RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control. [https://arxiv.org/abs/2307.15818](<https://arxiv.org/abs/2307.15818>)

\[5\] Black, K., et al. (2024). π0: A Vision-Language-Action Flow Model for General Robot Control. [https://arxiv.org/abs/2410.24164](<https://arxiv.org/abs/2410.24164>)

\[6\] Driess, D., et al. (2025). Knowledge Insulating Vision-Language-Action Models: Train Fast, Run Fast, Generalize Better. [https://arxiv.org/abs/2505.23705](<https://arxiv.org/abs/2505.23705>)

\[7\] Ball, P. J., Smith, L., Kostrikov, I., &amp; Levine, S. (2023). Efficient Online Reinforcement Learning with Offline Data. [https://arxiv.org/abs/2302.02948](<https://arxiv.org/abs/2302.02948>)

\[8\] Wagenmaker, A., et al. (2025). Steering Your Diffusion Policy with Latent Space Reinforcement Learning. [https://arxiv.org/abs/2506.15799](<https://arxiv.org/abs/2506.15799>)

\[9\] Physical Intelligence et al. (2025). π\*0.6: a VLA That Learns From Experience. [https://arxiv.org/abs/2511.14759](<https://arxiv.org/abs/2511.14759>)

\[10\] Physical Intelligence et al. (2026). π0.7: a Steerable Generalist Robotic Foundation Model with Emergent Capabilities. [https://arxiv.org/abs/2604.15483](<https://arxiv.org/abs/2604.15483>)

\[11\] Frans, K., Park, S., Abbeel, P., &amp; Levine, S. (2025). Diffusion Guidance Is a Controllable Policy Improvement Operator. [https://arxiv.org/abs/2505.23458](<https://arxiv.org/abs/2505.23458>)

\[12\] Dong, P., Li, Q., Sadigh, D., &amp; Finn, C. (2025). EXPO: Stable Reinforcement Learning with Expressive Policies. [https://arxiv.org/abs/2507.07986](<https://arxiv.org/abs/2507.07986>) (Probable identification of the unnamed Q&amp;A reference.)
