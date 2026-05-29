# Chapter 6 — Discussion and Limitations

Chapter 5 presented and analysed the experimental results in detail. This chapter steps back to interpret what those results collectively say about the two research questions, identifies the limitations that bound the strength of the conclusions, and considers the practical implications for deploying adaptive PID gain scheduling in real systems.

---

## 6.1 Main Findings Interpreted

### Research Question 1 — Feasibility of RL-Based Gain Scheduling

The central feasibility question — whether a domain-randomised PPO agent can learn to schedule PID gains reliably across unknown mass and friction conditions — is answered affirmatively. Both the Context-Aware Agent (Stage 5a) and the Blind Agent (Stage 5b) achieve 100% success across a 4:1 mass range (5 kg to 20 kg) and a 20:1 friction range (0.1 to 2.0), with zero overshoot in every evaluated episode. Both agents maintain this success under out-of-distribution conditions, succeeding at mass = 35 kg — 75% beyond the training ceiling — without any degradation in the success criterion. The gain-scheduling policy learned through domain randomisation and curriculum training generalises reliably across the full evaluation distribution, and meaningfully beyond it.

This reliability is achieved without prior knowledge of the current physics parameters. The environment randomises mass and friction at each episode reset; neither agent is given this information at deployment. The Context-Aware Agent observes its own mass and friction scales mid-episode; the Blind Agent never does. Both succeed. The answer to RQ1 is that domain-randomised PPO with frame stacking is a viable approach to the adaptive PID gain-scheduling problem under unknown dynamics.

### Research Question 2 — Benefit of Explicit Physics Context

The second question asks whether providing explicit physics context improves adaptation over blind inference from trajectory history. It does, in a specific and bounded way. The Context-Aware Agent settles consistently 14–15% faster than the Blind Agent across all three static evaluation scenarios, an advantage that is uniform across mass and friction conditions. The mechanism is clear: the Context-Aware Agent can condition gain selection on the current dynamics from the first timestep, while the Blind Agent must resolve uncertainty about mass and friction from the shape of its 10-frame trajectory window before committing to an appropriate gain schedule.

But the more important result for RQ2 is what the Blind Agent demonstrates: dynamics inference from trajectory history alone is sufficient for reliable control. The settling time penalty is the cost of blindness, not evidence of failure to adapt. Context improves adaptation *efficiency*; it does not determine adaptation *feasibility*. Both agents are capable of the task. The Blind Agent simply takes longer to establish which operating condition it is in before it can act on that knowledge.

### The Integral Reset Finding as a Methodological Contribution

Beyond the two research questions, the analysis in Section 5.4 produces a finding that reframes the interpretation of the full experiment. The `brake_integral_reset` mechanism — which zeros the PID integrator when the vehicle enters the braking zone — was active for all agents throughout training and evaluation, including the Fixed PID Classical Baseline. Its effect is to remove integral windup at the critical moment of deceleration, transforming what would otherwise be a difficult control problem into a tractable one for any sufficiently well-tuned controller.

This is not a flaw in the experiment. It is a methodological finding: the apparent speed advantage of the Fixed PID over the RL agents in the standard evaluation is not evidence that adaptive gain scheduling is unnecessary. It is evidence that a shared engineering aid, applied uniformly, happened to most benefit a baseline whose gains were pre-tuned for the exact operating conditions. When the aid is removed, the Fixed PID fails completely across all scenarios — zero success, approximately 10 m overshoot — while the RL agents were designed and trained to handle the full range of operating conditions that made the aid necessary in the first place.

Identifying this confound is itself a contribution: it demonstrates that careful auditing of simulation environment mechanisms is necessary when benchmarking adaptive control methods. An unexamined engineering aid can silently equalise controllers that should behave very differently, obscuring the adaptive advantage that motivates learning-based approaches in the first place.

---

## 6.2 Limitations

The conclusions above are bounded by several limitations, each of which constrains the confidence with which the findings can be generalised or acted upon.

**L1 — Single training seed**

All models reported in this thesis were trained using a single random seed (seed 7). Policy initialisation, data collection order, and advantage estimation are all seed-dependent. It is therefore unknown whether the 14–15% settling time gap between the Context-Aware Agent and the Blind Agent, or the absolute settling times reported, are representative of the method's typical behaviour or specific to this initialisation. A five-seed evaluation (seeds 7, 21, 42, 84, 123) would quantify initialisation variance and establish whether the reported results are robust across runs.

**L3 — Disturbance timing mismatch**

The dynamic evaluation applied mid-episode disturbances at a randomly chosen step in the range [120, 220], corresponding to 1.2–2.2 seconds after episode start. Both RL agents typically settle within 1.25–1.57 seconds across the evaluated scenarios. In a substantial fraction of dynamic evaluation episodes, the disturbance therefore fires after the vehicle has already reached and stabilised at the target position, with no opportunity to disrupt the approach or hold phase.

The consequence is that the dynamic evaluation results — 100% success and settling times indistinguishable from static evaluation — do not constitute evidence of robustness to mid-settling disturbances. They reflect robustness during the initial approach phase only. A meaningful robustness evaluation would require shifting the disturbance window to steps [30, 80], where both agents are still actively approaching the target, to test whether the learned policies can recover from sudden dynamics changes during the critical deceleration phase. As currently designed, the dynamic evaluation adds limited information beyond the static results.

**L4 — Friction irrelevance in the rolling contact model**

The MuJoCo simulation uses a rigid rolling contact model in which kinetic friction does not directly contribute to the vehicle's translational dynamics when the wheels roll without slipping. The friction parameter that is randomised at episode reset — and that the Context-Aware Agent observes in its state — therefore has no meaningful effect on vehicle response to control inputs under the conditions evaluated. The Heavy and Slippery and Light and Grippy scenarios differ primarily in mass; the friction labels are not reflected in distinct control dynamics. The friction dimension of the domain randomisation provides limited training signal, and the friction scale observation available to Stage 5a may carry little useful information for gain selection. Introducing velocity-dependent viscous damping would make friction a meaningful degree of freedom, allowing friction randomisation to contribute genuinely to the policy's generalisation.

**L5 — Integral reset confound limits the core comparison**

As established in Section 5.4, the `brake_integral_reset` mechanism was active for all agents during both training and evaluation. This limits the strength of the central RL-versus-Fixed-PID comparison. Without removing the aid and retraining the RL agents, it is not possible to determine how much of the learned policy's success is attributable to genuine adaptive gain scheduling versus shared reliance on the engineering aid at the moment braking authority is most critical. The RL agents learn conservative gain schedules — reducing Ki and raising Kd relative to the base gains — that are consistent with anticipatory windup management, but whether this conservatism is sufficient to generalise to control without the reset remains unverified. A full demonstration of RL advantage over non-adaptive control would require either disabling the aid and retraining both agents from scratch, or evaluating against a Fixed PID baseline that was not pre-tuned for the specific task conditions.

**L6 — No-reset retraining did not converge**

Retraining was attempted in an environment with the `brake_integral_reset` disabled, for 1.5M and 3M timesteps respectively, with and without additional overshoot penalties included in the reward signal. Neither run produced a converging policy. The most likely cause is the temporal credit assignment gap inherent in the required control strategy: to prevent integral windup during braking, the agent must suppress Ki during the early approach phase — at approximately step 50 of an episode — to avoid accumulating an i-term large enough that it overwhelms braking authority 500–1000 steps later, when the vehicle enters the deceleration zone. On-policy PPO with a multilayer perceptron policy and no recurrent hidden state has limited capacity to assign credit across horizons of this length. The policy receives a terminal overshoot penalty but cannot trace that penalty back to a Ki adjustment made hundreds of timesteps earlier.

A recurrent policy architecture — LSTM or GRU — would address this directly. By maintaining a hidden state that persists across the full episode, a recurrent policy could implicitly track cumulative integral error over the approach, making early Ki-suppression a learnable strategy without requiring an engineering mechanism to zero the integrator at a fixed position threshold. This represents the most technically well-motivated direction for extending this work.

---

## 6.3 Practical Implications

### The Blind Agent as the Deployable Result

Of the two learned agents, the Blind Agent (Stage 5b) is the more practically relevant for real-world deployment. In vehicle control applications, the mass and friction scale parameters that the Context-Aware Agent observes are not generally available as direct sensor measurements. Vehicle mass changes with payload and passengers; surface friction varies continuously with road condition, contamination, and tyre wear. An agent that requires these values as inputs would depend on a real-time parameter estimator — itself a non-trivial system to design, calibrate, and maintain alongside the controller.

The Blind Agent requires none of this infrastructure. It observes only position, velocity, position error, and the current PID gains — quantities that are straightforwardly measurable from standard encoders, odometers, and inertial sensors. Its 10-frame observation stack provides approximately 0.2 seconds of trajectory history, from which it infers operating dynamics implicitly and adjusts gain scheduling accordingly. The result is a self-contained adaptive controller that requires no explicit system identification step.

The cost is the 14–15% settling time penalty relative to the Context-Aware Agent. Whether this is acceptable depends on the application. In precision positioning systems with strict time budgets, additional settling time is a material constraint. In systems where reliable stopping within a tolerance band matters more than speed — loading operations, variable-payload positioning, docking systems — the Blind Agent's performance profile is likely sufficient. The zero overshoot result across all evaluated conditions is the more relevant metric for these use cases, and on that criterion both agents perform identically.

One important caveat applies to any deployment consideration: the evaluation presented in this thesis was conducted in simulation, under idealised sensing and actuation. Real hardware introduces sensor noise, actuation lag, and mechanical compliance that are absent from the MuJoCo model. The sim-to-real gap would need to be characterised and addressed before any deployment claim could be substantiated.

### Frame Stacking as Implicit Dynamics Inference

A broader observation follows from the Blind Agent's success. Frame stacking — concatenating a fixed window of recent observations as input to the policy, without recurrent connections — is a viable and computationally lightweight mechanism for implicit dynamics inference in short-horizon adaptation problems. Without architectural complexity, without a learned latent embedding of task identity, and without any explicit system identification module, a 10-frame observation stack gives a feedforward policy sufficient trajectory context to infer operating conditions and adapt gain scheduling accordingly.

This approach has practical appeal: it adds no training overhead beyond the larger input dimension and can be applied to any problem where dynamics vary across episodes but the relevant signal is present in a short window of recent trajectory behaviour. Its limitation is horizon length. As the no-reset retraining failure demonstrates, frame stacking cannot substitute for explicit memory when the credit assignment gap spans hundreds of timesteps — situations where a decision made early in an episode has consequences that only manifest at episode end. For problems where the relevant dynamics information appears within a short recent window, frame stacking is a practical and sufficient first choice. For problems requiring long-horizon integral tracking or memory of cumulative state, recurrent architectures remain necessary.

Chapter 7 develops the concrete future directions that follow from these limitations.
