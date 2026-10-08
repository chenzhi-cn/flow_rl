---
id: sergey-2015-03-summary
title: Deep Learning for Decision Making and Control
speaker: Sergey Levine
language: en
document_type: research_talk_summary
source_file: 2015_Mar_Sergey_Summary.docx
source_sha256: ac85a60004652f5152d7f54f9bcb7680ade545faab60df5a0e6bac6c4f5207ed
period_from_filename: 2015-03
talk_date: null
recording_date: '2015-03-10'
video_publication_date: null
video_url: https://www.youtube.com/watch?v=EtMyH_--vnU
keywords:
- robot learning
- guided policy search
- trajectory optimization
- local dynamics models
- visuomotor control
- end-to-end learning
- contact-rich manipulation
conversion_date: '2026-09-30'
content_scope: Full source notes, with source-derived retrieval overview and normalized Markdown formatting.
verification_scope: Converted against the supplied DOCX only. Original attribution, caveats, and references are
  retained; external sources were not re-verified during conversion.
date_note: The source labels the UW Video recording 10 March 2015.
---

# Deep Learning for Decision Making and Control

## Retrieval overview

This overview is derived from the supplied summary. The full source notes follow, including their evidence qualifications and editorial interpretations.

**Research goal:** Learn high-dimensional feedback policies efficiently and train perception and control jointly for physical manipulation.

**Motivation:** Human motor competence exposes the brittleness of separately engineered robotics modules, while direct search over a large neural policy is difficult and data intensive.

**Approach:** Guided policy search separates local trajectory optimization from supervised policy fitting, coordinates the two through constraints, and uses task performance to shape visual representations.

## Source notes

LECTURE RECONSTRUCTION  /  ROBOT LEARNING

Comprehensive Paraphrased Lecture Notes

**Speaker**  Sergey Levine UW Video  •  10 March 2015  •  56 minutes

**CENTRAL CLAIM**

High-dimensional feedback policies can be learned efficiently when trajectory optimization supplies structured supervision. This makes it possible to train perception and control together, so that a robot learns task-directed sensorimotor behavior rather than a brittle chain of separately engineered modules.

**Reader's note.** This is an independent, comprehensive paraphrase, not a verbatim transcript. It preserves the lecture's sequence, arguments, technical details, examples, and audience discussion in newly written language. Timestamps are approximate; they were aligned against a near-contemporaneous companion recording where needed.

**Primary recording:** [youtube.com/watch?v=EtMyH\_--vnU](<https://www.youtube.com/watch?v=EtMyH_--vnU>)

DOCUMENT GUIDE

### How to read these notes

The narrative below follows the talk from its opening robotics examples through the guided policy search derivation, physical experiments, end-to-end visuomotor results, research agenda, and final questions. Short interpretive labels clarify the function of each segment without adding claims that the lecture does not support.

| **Field** | **Details** |
| --- | --- |
| **Speaker** | Sergey Levine |
| **Recorded lecture** | UW Video, 10 March 2015 |
| **Linked runtime** | 56:03 |
| **Document type** | Comprehensive, time-ordered paraphrase |
| **Technical focus** | Guided policy search, contact-rich manipulation, end-to-end visuomotor control |

#### Lecture map

| **Time** | **Movement** | **Question driving the section** |
| --- | --- | --- |
| 00:00-07:00 | Motivation | Why do robots struggle with behaviors that people execute almost effortlessly? |
| 07:00-17:20 | Method | How can trajectory optimization become a teacher for a large, general policy? |
| 17:20-27:30 | Learning on hardware | How can the method work when dynamics and contacts are unknown? |
| 27:30-38:00 | Vision to torque | Can perception and control be trained jointly from raw images? |
| 38:00-49:20 | Scaling data | Where will the variety needed for general sensorimotor intelligence come from? |
| 49:20-55:12 | Discussion | How should initialization, robustness, deformable objects, and composition be handled? |

#### Four ideas to retain

- **Separate the hard parts during learning.** Use optimal control to handle dynamics locally and supervised learning to fit the expressive policy.

- **Train around trajectories, not only on them.** A distribution of nearby states supplies corrective examples and limits compounding error.

- **Let task performance shape perception.** Visual features become useful for control when the entire image-to-torque network is fine-tuned end to end.

- **Generalization is ultimately a data problem.** Broader skills require demonstrations, shared robot experience, and prior visual or geometric data in addition to self-generated trials.

### 1. Introduction: a learning-and-control research program  **00:00-01:20**

The host introduces Sergey Levine as a researcher working at the intersection of machine learning, control, perception, and robotics. His recent work with Pieter Abbeel at Berkeley builds on doctoral research with Vladlen Koltun and spans applications from character animation to manipulation. The framing matters: this is not a talk about a single robot trick, but about a reusable way to learn decision-making systems.

Levine opens by defining decision making broadly. The same mathematical language can describe how a robot chooses motor torques, how an educational game adapts the next concept to a student, how an autonomous vehicle reacts, or how an animated character selects a gesture. The lecture concentrates on robotic sensorimotor skills because they expose the central difficulty in its sharpest form: an agent must convert rich, noisy observations into a rapid sequence of actions while the physical world pushes back.

### 2. The gap between robotic and human motor skill  **01:20-03:45**

A door-opening sequence from the DARPA Robotics Challenge provides the first contrast. A sophisticated robot takes roughly a minute and a half to perform an action that a person can complete almost immediately, even with limited visual attention. The comparison is intentionally uncomfortable. It shows that better hardware and more computation do not automatically yield the adaptable coordination humans display.

The ball-catching example reveals a possible reason. A traditional engineering treatment could estimate the ball's three-dimensional position, infer its velocity, identify drag and wind, and integrate a flight model. Yet people can often catch a ball using a much simpler visual rule: move so that the ball's image remains at an approximately constant bearing. The rule couples perception directly to action. It avoids solving for every hidden physical variable, continuously corrects error, and remains effective when the environment differs from an assumed model.

**MOTIVATING CONTRAST**

The lecture is not arguing that models are useless. It is arguing that a successful controller may exploit a task-specific feedback relationship that is easier to execute and more robust than reconstructing the entire world first.

### 3. Three traditions, one end-to-end ambition  **03:45-07:00**

Levine locates the proposed approach among three research traditions. Optimal control is strong at finding a good trajectory when dynamics and state are available, but a single optimized motion does not automatically transfer to new situations. Reinforcement learning aims to learn a policy, yet many successful systems of the period rely on compact policy classes whose structure has been engineered for the task. Deep learning supplies highly expressive function approximators that can learn intermediate representations from data, but it needs an efficient training signal.

The standard robotics pipeline is then contrasted with an end-to-end policy. In the conventional pipeline, sensors feed a state estimator; the resulting state enters a task-specific controller; and that controller sends commands to a low-level servo. Each boundary is a design commitment and a potential source of mismatch. Levine instead proposes a rich function that consumes raw observations together with proprioceptive information and directly outputs motor commands. A neural network is the chosen implementation, but the substantive requirement is expressive capacity: the policy must be large enough to encode perception, feedback, and action selection jointly.

The goal is not merely to copy a motion. A useful policy should react to perturbations, correct itself, and generalize across nearby situations. That implies thousands or tens of thousands of parameters rather than a handful. It also creates the lecture's main algorithmic question: how can such a policy be trained with the limited interaction budget of a physical robot?

### 4. Formal objective and the failure of naive policy search  **07:00-09:10**

Let a policy πθ choose an action u from an observation o, with parameters θ. Over a rollout, the policy induces a distribution of states and actions. Training seeks parameters that minimize expected cumulative task cost - for example, distance from a target together with a penalty on large torques. In compact form, the objective is to minimize E\[Σt ℓ(xt, ut)\] under trajectories generated by πθ.

A direct, textbook policy-search strategy would perturb the policy, execute rollouts, reward good outcomes, penalize bad ones, and slowly adjust all parameters. In a large neural network this is disastrously inefficient. At the beginning, almost every rollout fails, so the learning signal says little about which of the many decisions was responsible. Levine illustrates the problem with a simulated biped: if a random controller mostly falls, waiting for an accidental run before assigning credit is not a viable plan.

**OPTIMIZATION BOTTLENECK**

Direct policy search asks one procedure to solve two difficult problems at once: reason through nonlinear dynamics and tune a high-dimensional policy. Guided policy search makes progress by assigning those jobs to different optimization tools.

### 5. Guided policy search: turn control into supervised learning  **09:10-14:00**

Imagine first that an oracle could label every observation with the action the robot ought to take. Fitting a neural network to those labeled pairs would be an ordinary supervised-learning problem. The oracle does not exist, but trajectory optimization can play a limited version of that role. From a small set of training conditions, an optimal-control method can produce local controllers that know the full state and solve the task around those conditions. Their state-action samples then become targets for the global policy, which may receive only the observations that will be available at test time.

This procedure is more than straightforward imitation. Training only on one nominal path creates compounding error: a small policy mistake moves the robot to a state that never appeared in the dataset, so the next prediction is less reliable, which moves the robot farther away. Guided policy search instead represents each local solution as a distribution - a tube of likely states and actions around the nominal motion. Samples from the tube teach the global policy how to correct nearby deviations.

The local distribution is constructed by an entropy-regularized optimal-control method. In the locally linear-quadratic setting, the result is a time-varying linear-Gaussian controller. Randomness is not just noise added after the fact; it is part of the training distribution and determines where corrective labels are available.

A second problem appears if the local controllers are optimized independently of the policy they supervise. A fully observed controller can exploit information hidden from the final policy. It may also choose different actions at perceptually identical situations. Levine gives two intuitions. A snake-like swimmer could produce jagged optimal paths that require memorizing time rather than learning a coherent stroke. For insertion, controllers that know the exact hidden hole location may move directly to it, while the vision-limited global policy needs a consistent search-and-correct behavior. The solution is to close the loop: improve trajectories while penalizing disagreement with the policy, then refit the policy to the revised trajectories.

### 6. The constrained optimization view  **14:00-17:20**

The formal derivation introduces a trajectory distribution p and a global policy πθ. The trajectory optimizer minimizes expected cost under p, while the constraint requires the conditional action distribution of p to match the policy at visited states. If the two distributions agree, optimizing the local controllers also optimizes behavior that the global policy can reproduce.

**CONSTRAINED FORM**

Minimize over p and θ:  E\_p\[Σ\_t ℓ(x\_t, u\_t)\], subject to p(u\_t | x\_t) = π\_θ(u\_t | o(x\_t)). In practice, matching is enforced through KL-based penalties and dual variables rather than an exact one-shot equality.

Levine describes an alternating procedure based on a dual-decomposition method. First, optimize each trajectory distribution under the current penalty for disagreeing with the policy. Second, perform supervised learning so the neural network matches actions sampled from the trajectory distributions. Third, update the dual variables, increasing pressure where the two sides still disagree. Repeating these steps gradually makes the local solutions both low-cost and representable by one global policy.

A simulated three-dimensional biped illustrates the effect. Local controllers are trained for a small collection of pushes, while a single network learns the shared response. When evaluated on varied perturbations, the policy balances and recovers rather than memorizing four separate sequences. The example supports the intended division of labor: local control supplies structured experience; the neural network identifies what can be reused across conditions.

### 7. Unknown dynamics: learn only the model needed locally  **17:20-21:00**

The derivation so far assumes that the trajectory optimizer knows the system dynamics. Real contact-rich manipulation violates that assumption. Friction, impacts, object motion, and actuator behavior are hard to model accurately in advance. The method therefore alternates between acting and fitting: execute the current controller, collect short trajectories, regress a time-varying local linear dynamics model, optimize a new controller under that fitted model, and gather more data.

A learned local model is trustworthy only near the data that produced it. If optimization proposes a controller far outside that region, model error can create an apparently excellent but physically useless solution. A KL trust region prevents this. Each new trajectory distribution is constrained to stay close to the distribution that generated the latest data. The step size can be adapted according to whether predicted improvement survives on the real system. This makes model learning and control improvement proceed in cautious, verifiable increments.

Fitting a separate high-dimensional dynamics model at every time step would still require too many samples. Levine uses a global Gaussian-mixture prior over transitions collected across time and then conditions that prior to regularize each local regression. The mixture is not expected to be a globally faithful simulator. Its role is to share statistical structure among nearby phases of the task, allowing local linear models to be estimated from a surprisingly small number of rollouts. A fifty-dimensional simulated octopus arm is cited as an extreme benchmark for this sample-efficient fitting strategy.

### 8. Contact-rich manipulation on a PR2  **21:00-27:30**

The method is then moved onto a PR2 robot. Each local policy begins with a small, stabilized random motion around the initial configuration; a demonstration could also provide initialization. The task state includes joint positions and velocities together with task-relevant geometric quantities such as the vector from the end effector to a target. The learned neural policy outputs joint torques. During training, only about five rollouts are collected per iteration.

| **Task** | **What the policy must discover** | **Evidence highlighted in the lecture** |
| --- | --- | --- |
| LEGO stacking | Contact sequence and placement under imperfect dynamics | A short interaction budget is enough to learn a smooth execution. |
| Rings on a peg | Alignment and correction around a constrained target | The controller closes the loop instead of replaying a fixed path. |
| Toy airplane | Coordinated insertion/assembly behavior | Complex contact is learned without a hand-coded contact model. |
| Shoe tree insertion | Forceful insertion while managing geometry | Local fitted dynamics capture the behavior needed around the practiced region. |
| Bottle cap | Push-and-turn coordination | A supplied wrist rotation is combined with a learned downward push. |

The examples are chosen because contact makes analytical modeling awkward. The optimizer is given a cost, not a scripted contact sequence. For the bottle-cap task, for instance, a wrist rotation is supplied, but the system learns that the cap must also be pushed down while turning. Across tasks, the learned motions appear fast and fluent rather than stop-and-check. Levine reports roughly twenty to thirty executions for a typical skill. The total wall-clock time is on the order of ten minutes in the shown implementation, split among robot execution, computation, and avoidable software overhead; an optimized version approaches five minutes.

#### Audience questions during the demonstration

Asked about force, safety, and the apparent aggressiveness of some motions, Levine explains that the objective trades off task error against control effort. A lower penalty on torque can encourage momentum; a larger penalty produces gentler behavior. The random initialization is kept near the starting pose by stabilizing feedback, so the robot does not explore by flailing without limits. The policy does not receive an explicit contact-state variable or a hand-written friction model. It receives encoders, derived velocities, and torque commands; the effects of external forces are captured implicitly through the fitted dynamics and closed-loop response.

A separate target-generalization example trains one network on several target positions and then evaluates an unseen position. The policy initially misses but observes the consequence and corrects. This is evidence of feedback behavior rather than a library of open-loop trajectories. At the same time, the generalization range is local: the demonstrations establish that one policy can interpolate among related conditions, not that it can solve arbitrary manipulation problems without new data.

### 9. From pixels to torque: the visuomotor policy  **27:30-31:00**

The next stage removes task-state inputs that would be unavailable in an ordinary environment. A convolutional neural network receives RGB images from the robot's camera, while fully connected motor layers also receive joint configuration. The complete network contains roughly ninety-two thousand parameters and maps these inputs directly to torques. The vision pathway is designed to convert feature maps into a compact spatial representation suitable for control.

At the final convolutional layer, every channel produces an activation map. A spatial softmax exponentiates and normalizes the map, forming a distribution over image location. Taking the expected horizontal and vertical coordinates yields one differentiable feature point per channel. With thirty-two channels, the controller receives thirty-two learned image points. Normalization suppresses weaker competing activations, giving a form of lateral inhibition that can reduce distraction by irrelevant visual patterns. More importantly, the entire operation is differentiable, so task error can move the visual features toward locations that matter for action.

**ARCHITECTURE INSIGHT**

The spatial bottleneck is deliberately structured: convolutional layers learn what to attend to, the spatial softmax says where it is, and the motor layers combine those coordinates with robot configuration to decide how to move.

### 10. Instrumented training, realistic testing  **31:00-35:00**

A visual manipulation task is partially observed because the target pose is visible in the image but hidden from the policy's robot-state vector. Guided policy search handles this asymmetry by allowing the local trajectory controllers to use full state during training. For the shape-sorting task, the robot holds the cube in its other gripper, so its pose is known while local controllers are optimized. The global policy is never given that target coordinate; it must infer the information from monocular RGB. At test time the cube can be placed freely on the table, and the instrumentation is removed.

Training the entire image-to-torque network from scratch inside every guided policy search iteration would be slow. Both sides are therefore initialized. The robot first moves the target object through random positions and records paired images and state. The vision layers are pretrained to regress the hidden target quantities. Separately, the local trajectory controllers are initialized with the unknown-dynamics procedure described earlier. End-to-end guided policy search then fine-tunes all layers according to task performance. Pretraining supplies a sensible starting representation; it does not determine the final visual features.

| **Visuomotor task** | **Variation withheld from the policy** | **Generalization test** |
| --- | --- | --- |
| Shape-sorting cube | Target position is absent from the robot-state input | Move the cube; add distractors; hold it by hand. |
| Coat hanger | Rack distance must be inferred from a monocular image | Add clothes and substitute a wooden hanger. |
| Toy hammer | The effective hammer pose changes with the grasp | Vary grasp angle and add visual clutter. |
| Bottle cap | Bottle position is not supplied | Move the bottle and place distractor objects nearby. |

### 11. Why end-to-end training changes the result  **35:00-38:00**

The lecture compares the final network with two modular baselines. The closest analogue to a conventional pipeline freezes the pretrained visual estimator, predicts the target pose, and feeds that estimate to a task-optimized controller. A second baseline exposes the pretrained spatial feature points to the controller but still prevents the vision layers from adapting to the task. Both lag behind the end-to-end network, especially on the tight-tolerance problems.

The best pose-estimation network has an average localization error of about 1.3 centimeters. That is respectable and broadly consistent with the precision of contemporary marker-based robotic perception cited in the talk, but the shape-sorting task allows only a few millimeters. A visually plausible estimate is therefore not necessarily a control-useful estimate. When task loss flows through the controller into perception, learned feature points move toward the target, the gripper, and the manipulated object. The representation spends capacity on distinctions that change the action rather than on a generic notion of pose accuracy.

Qualitative tests probe the resulting behavior. A policy trained with a plastic hanger can handle a wooden one and remain effective when clothes appear on the rack. The shape sorter works amid distractors and even when a person holds the cube. A bottle-cap policy reacts when the bottle slides away. Because the cost discourages excessive torque, the arm is compliant when pushed and tries to return, although it eventually fails because recovery from a large disturbance was not in the training distribution. These demonstrations are presented as bounded generalization, not universal robustness.

**EMPIRICAL LESSON**

Perception and control are not independent scoring problems. A representation optimized for an intermediate label can discard distinctions that are crucial to the final behavior; optimizing the whole sensorimotor loop can recover those distinctions.

### 12. General skills require a data strategy  **38:00-42:10**

The bottle policy can cap one kind of bottle, but a household robot must cope with many shapes, appearances, placements, and surroundings. Levine uses the history of supervised learning to frame what comes next. Recent progress in vision and speech rests on three mutually reinforcing ingredients: more computation, better models and algorithms, and large datasets. Robot learning shares the first ingredient and is improving the second, but the third remains the limiting resource. Expressive policies cannot generalize to variation they have never encountered.

The remainder of the lecture becomes a research agenda for acquiring that variation. Four sources are identified: a robot's own experience; demonstrations by humans or teleoperators; experience shared among multiple robots; and prior datasets such as images or three-dimensional models. Guided policy search is offered as a useful organizing framework because the policy-training side is supervised learning. Different sources can contribute labeled actions, initialize local controllers, reveal task objectives, or shape visual representations.

### 13. Demonstrations can reveal more than motion  **42:10-45:00**

Demonstrations already help initialize difficult trajectories. The simulated running controller, for example, begins from motion-capture behavior. Levine also points to prior work in which motion capture, a nonlinear motion embedding, and reinforcement learning produce a martial-arts character controller. But direct motion transfer is limited whenever the demonstrator and the learner have different dynamics or bodies.

A towel-folding comparison makes the deeper opportunity visible. Merely warping one demonstrated path to a new towel geometry does not capture the need to pull the fabric taut, so the fold becomes wrinkled. Multiple demonstrations reveal which relations are invariant: where grasp precision matters, when tension is needed, and what a neat outcome looks like. The desired output is not necessarily the demonstrator's coordinates; it may be the objective the demonstrator is pursuing.

Recovering a goal or cost function from behavior is inverse reinforcement learning. If a cost is grounded in image structure and generalizes to new scenes, it separates what should be achieved from the dynamics of the body that achieved it. A human can demonstrate the goal; a robot can then optimize its own actions against that goal. Levine imagines combining inverse reinforcement learning with deep visual representations, making instructional video on the internet a potential source of robot-learning data. The idea is aspirational in this talk, but it anticipates a move from learning motions to learning task semantics.

### 14. Shared experience across robots  **45:00-47:10**

Another route to diversity is parallel experience. Rather than make one robot repeatedly exchange objects, many robots can practice related tasks in different settings. Factory robots packing different boxes, against different backgrounds and with different items, could contribute to one policy whose shared structure transfers to a new station. Guided policy search already combines multiple local trajectories into one global policy, so the trajectories could in principle originate on different machines.

Useful fleet learning would have to be opportunistic. Buying a large private fleet solely to collect data undermines the efficiency argument; the system should learn from robots already doing work. That requirement introduces practical research problems: the contributing robots may execute heterogeneous nonlinear policies; data arrive asynchronously; learning updates must not interrupt production; and policy changes need safety and performance guarantees. Levine argues that standardized platforms make the timing favorable, but the infrastructure and reliability theory are not yet solved.

### 15. Prior datasets and broader applications  **47:10-49:20**

Physical interaction is not the only way to learn invariance. If the mechanics of capping bottles are similar while visual appearance changes, a robot might learn the physical operation on one bottle and learn appearance diversity from images of many others. In the guided-policy-search view, robot trajectories provide image-state-action tuples labeled with torques. External images lack action labels, turning policy fitting into a semi-supervised problem. The lecture does not claim a complete solution; it identifies semi-supervised learning as the bridge that could make large unlabeled visual collections useful for control.

Levine closes the prepared talk by returning to the breadth of decision making. In an educational fractions game, a learned concept-sequencing policy increases the number of concepts students complete before quitting, with a reported improvement of about thirty-two percent in a study of roughly two thousand students. In automatic character animation, motion-capture and speech data infer a gestural style, while reinforcement learning chooses a gesture that fits both that style and the preceding motion. Future directions include high-performance autonomous flight, predictive biomechanics, and goal-directed perception. The common structure is an agent selecting actions over time to optimize an outcome under uncertainty.

### 16. Audience discussion  **49:20-55:12**

#### Initialization

Asked how the trajectory optimizers get started, Levine distinguishes cases. The three-dimensional biped uses a demonstration derived from motion capture. The physical manipulation tasks shown earlier use stabilized random motion near the initial configuration. A kinesthetic demonstration - physically guiding the robot through a rough motion - could substitute when random exploration is unlikely to find a useful region. Initialization does not need to be optimal; it must place data where local improvement is possible.

#### Interpretability, robustness, and distribution shift

An audience member presses on fragility: a modular system exposes interpretable estimates, while a deep network may fail unexpectedly when an unfamiliar visual pattern appears. Levine agrees that reliability is a central unresolved problem. In this architecture, plotting the learned spatial feature points offers some diagnostic insight into what the vision system tracks, but it is not a full explanation or guarantee.

He locates much brittleness in mismatch between the training and deployment distributions. If the policy sees the relevant sensory situations during training, learning does not inherently preclude robustness. The hard problem is to identify the regimes where the model is fragile and obtain the right additional experience. Active learning could direct trials toward those uncertain regions. A follow-up comment extends the idea: a controller should detect when it is outside familiar conditions, decline unsafe action, and request or collect targeted training.

#### Deformable objects

The towel example prompts a question about non-rigid objects. Levine clarifies that the towel result did not use the optimal-control machinery from the main experiments; it combined learning with geometric demonstration transfer. For deformable objects, the immediate obstacle is often not merely high-dimensional dynamics but measurement of success. A robot needs a cost that can distinguish a taut, neat fold from a wrinkled one, likely from images. Learning such visual costs would make reinforcement learning on deformable materials far more tractable.

#### Compositionality

The final question asks whether learned skills can be composed. Levine separates two problems. First, controllers must be trained so that the terminal states of one lie inside the workable initial region of the next; a preceding skill should anticipate what its successor needs. Second, a higher-level planner must choose which skills to sequence. Ongoing work is aimed at the first problem, while hierarchical planning offers a natural framework for the second. The answer points beyond isolated motor primitives toward multi-step behavior.

SYNTHESIS

### The argument in one page

1. **Human skill motivates direct feedback.** People often exploit stable perception-action relationships rather than reconstructing every physical variable.

2. **Expressive policies are necessary but hard to search directly.** The number of parameters makes trial-and-error policy optimization too costly on robots.

3. **Trajectory optimization supplies structured supervision.** Local controllers solve the dynamics problem around selected conditions and label a global policy.

4. **Distribution matching makes the supervision executable.** Alternating trajectory updates, supervised policy fitting, and dual updates keeps local solutions compatible with one policy.

5. **Local model learning enables contact-rich hardware experiments.** Fitted dynamics, KL trust regions, and shared statistical priors extract improvement from a small rollout budget.

6. **End-to-end visual training aligns representation with action.** A spatial feature bottleneck allows task loss to shape what the camera pathway attends to.

7. **The next frontier is data diversity.** Demonstrations, robot fleets, and unlabeled visual datasets are proposed as complementary routes to broad generalization.

8. **Reliability remains an open requirement.** Deployment needs detection of unfamiliar conditions, targeted data collection, learned objectives, and composable skill boundaries.

**BOTTOM LINE**

Guided policy search is presented as a bridge: optimal control generates locally intelligent behavior, supervised learning distills that behavior into an expressive policy, and end-to-end training lets task success determine the internal representation. The bridge works impressively on narrow skills; crossing from those demonstrations to dependable general-purpose robots demands much broader data and stronger guarantees.

#### Research trail

1. **UW Video.** Archive: Deep Learning for Decision Making and Control. Public lecture recording, 10 March 2015. [View recording](<https://www.youtube.com/watch?v=EtMyH_--vnU>)

2. **Levine, S., and Abbeel, P.** Learning Neural Network Policies with Guided Policy Search under Unknown Dynamics. Advances in Neural Information Processing Systems 27, 2014. [NeurIPS abstract](<https://papers.nips.cc/paper_files/paper/2014/hash/c7c9344b5a3c0533e29fa69ce807cf08-Abstract.html>)

3. **Levine, S., Wagener, N., and Abbeel, P.** Learning Contact-Rich Manipulation Skills with Guided Policy Search. IEEE International Conference on Robotics and Automation, 2015. [arXiv record](<https://arxiv.org/abs/1501.05611>)

4. **Levine, S., Finn, C., Darrell, T., and Abbeel, P.** End-to-End Training of Deep Visuomotor Policies. Journal of Machine Learning Research 17(39), 2016. [JMLR article](<https://jmlr.org/papers/v17/15-522.html>)

5. **Levine, S., and Koltun, V.** Learning Complex Neural Network Policies with Trajectory Optimization. International Conference on Machine Learning, 2014. [ICML paper](<https://proceedings.mlr.press/v32/levine14.html>)

**Source-method note.** The linked UW recording was the primary source. A near-contemporaneous companion recording hosted by UC San Diego was used to stabilize the sequence and approximate timestamps where the linked stream did not expose a reliable transcript. The cited papers were used to check technical names, architecture details, and reported experimental context. All explanatory prose in this document is newly written.

**Companion recording:** [cseweb.ucsd.edu/csevideo/Sergey.Levine.mp4](<https://cseweb.ucsd.edu/csevideo/Sergey.Levine.mp4>)
