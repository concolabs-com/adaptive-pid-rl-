# Chapter 1: Introduction

## 1.1 Background and Motivation

PID controllers are the dominant feedback control mechanism in industrial motion systems. Surveys of industrial process control consistently report that more than 90% of deployed control loops use PID or PI variants [^astrom1995]. Their widespread adoption reflects well-understood practical advantages: the three-term structure — proportional, integral, and derivative — is interpretable, tunable with minimal system knowledge, and provably stable for a wide class of linear plants. From robotic manipulators and CNC machining centres to autonomous wheeled platforms and conveyor systems, PID control governs real-time motion at enormous scale across virtually every domain of modern manufacturing and automation.

The limitation of PID control is equally well-understood: fixed gains assume a fixed plant. A set of gains tuned for a vehicle carrying a 10 kg payload will behave differently — often incorrectly — when the same vehicle carries 20 kg, or when the surface friction changes underfoot. In the best case, the controller becomes sluggish; in the worst case, it overshoots the target, oscillates, or fails to hold position altogether. This is not an edge case in practice. Real-world autonomous systems routinely operate across a range of loading conditions, surface types, and wear states that cannot be fully anticipated at design and tuning time. A warehouse robot moving loaded pallets across different floor surfaces, or a rehabilitation device adapting to the varying limb dynamics of different patients, represents exactly this challenge: the plant parameters are not constant, and the gains appropriate for one operating condition may be entirely wrong for another.

This thesis is motivated by a concrete instance of this problem: a two-wheeled simulated vehicle tasked with navigating to a target position. The vehicle's mass varies across a 4:1 range and the surface friction varies across a 20:1 range, and neither is known to the controller in advance. The question is whether a learning-based gain scheduler can handle this uncertainty reliably, and what the practical benefit of informing that scheduler with explicit physics measurements turns out to be.

## 1.2 Classical Approaches to Adaptive Control

The classical response to time-varying or uncertain plant parameters is adaptive control. The oldest approach — gain scheduling — maintains a lookup table of pre-computed gain sets indexed by measurable operating conditions, switching between them as the system state changes. Where the uncertainty can be parameterised and the operating conditions can be measured directly, gain scheduling works well. However, it requires the designer to enumerate the relevant operating modes, tune gains for each, and specify the switching logic. When the uncertainty is continuous, high-dimensional, or only partially observable from available sensors, constructing and maintaining this table becomes impractical.

Model Reference Adaptive Control (MRAC) takes a more principled approach: gains are adjusted continuously online to drive the closed-loop system behaviour toward that of a specified reference model. MRAC offers genuine online adaptation without requiring a pre-computed table, but it relies on an explicit mathematical model of the plant's parameter variation. The adaptation law is typically derived via Lyapunov stability theory from the system's differential equations. In practice, MRAC designs can become unstable when the real plant deviates significantly from the assumed model structure — a particular concern in scenarios with large parameter variation or nonlinear dynamics.

In this thesis, MRAC was implemented and evaluated on the target environment as part of the development process. It failed to converge — achieving 0% task success due to gain divergence — confirming the practical limitations of model-based adaptive approaches when applied to a plant with significant nonlinearity and parameter variation. This failure provides direct empirical motivation for the learning-based approach that forms the core of the thesis.

Both gain scheduling and MRAC share a fundamental design assumption: the engineer must know enough about the uncertainty to model it, parameterise it, or enumerate its modes. This assumption becomes increasingly untenable as deployment scenarios grow more diverse and operating envelopes expand beyond what was originally anticipated.

## 1.3 Reinforcement Learning as a Model-Free Alternative

Reinforcement learning (RL) offers a fundamentally different approach to the adaptive control problem. Rather than requiring an explicit model of parameter variation, an RL agent learns a control policy purely through interaction with an environment — observing system states, taking actions, and receiving scalar reward signals that encode the desired behaviour. The agent adjusts its policy over many training episodes to maximise cumulative reward, learning implicit representations of dynamics rather than relying on analytically derived adaptation laws.

When trained across a distribution of operating conditions — a technique called domain randomisation — the resulting policy can generalise to unseen parameter values at deployment time without any explicit model of how parameters affect dynamics. The agent learns not a single mapping from state to action, but a family of behaviours that are appropriate across the range of conditions encountered during training. This is sometimes described in the literature as a form of meta-learning: the policy has implicitly "learned how to adapt" by experiencing diverse conditions during training.

A critical design choice in this work is that the RL agent is placed in the role of a gain scheduler rather than a direct motor controller. The agent does not compute wheel torques or motor voltages; it adjusts the proportional (Kp), integral (Ki), and derivative (Kd) gains of an underlying PID controller at each time step. The PID controller, in turn, computes the actual control signal applied to the plant. This formulation — RL as a gain scheduler atop a classical PID layer — offers several practical advantages. First, the underlying PID structure provides a stable, interpretable control foundation whose properties are well-understood and can be bounded. Second, the action space of the RL agent is continuous and low-dimensional (three gain adjustments), which is substantially easier to learn than raw motor commands. Third, and most importantly, operating in "gain space" rather than "action space" makes the learned policy more abstract: a policy that learns to adjust PID gains is, in principle, applicable to any system governed by a PID controller, not only the specific plant on which it was trained. This opens the door to cross-system transfer as a future direction.

## 1.4 Research Questions

This thesis addresses two related research questions:

> **RQ1 — Feasibility:** Can a reinforcement learning agent learn to schedule PID gains in real time, achieving reliable position control across a range of unknown mass and friction conditions?

> **RQ2 — Context Benefit:** Does providing the agent with explicit physics context — measured mass and friction scales as inputs to the policy — improve adaptation compared to a blind agent that must infer dynamics from trajectory history alone?

Both questions are investigated on a simulated two-wheeled vehicle in the MuJoCo physics engine. At each episode, mass is drawn uniformly from [5, 20] kg and surface friction from [0.1, 2.0], neither revealed to the controller by the environment. Two agent variants are compared: a context-aware agent (Stage 5a) that observes its own physics parameters directly, and a blind agent (Stage 5b) that must infer the dynamics purely from a window of recent trajectory history. A fixed PID baseline with pre-tuned gains serves as the classical non-adaptive reference.

## 1.5 Contributions

The specific contributions of this thesis are:

1. **A domain-randomised PPO agent for online PID gain scheduling under unknown dynamics.** A Proximal Policy Optimisation (PPO) agent is trained via domain randomisation and curriculum learning to schedule PID gains in real time on a simulated wheeled vehicle. The agent achieves 100% task success across the full training distribution of mass and friction conditions, and generalises to out-of-distribution conditions including mass up to 35 kg — 75% above the training ceiling — maintaining 100% success with zero overshoot.

2. **An ablation demonstrating the measurable benefit of physics context.** A context-aware variant (which observes mass and friction scales) and a blind variant (which does not) are trained and evaluated under identical conditions. The context-aware agent settles consistently 14–15% faster across all evaluation scenarios, confirming that explicit physics context in the observation improves adaptation efficiency. Critically, both agents succeed — the benefit of context is speed, not reliability.

3. **Identification and analysis of a brake integral reset confound.** A mechanism embedded in the simulation environment — which zeros the PID integrator when the vehicle enters the braking zone — was identified as an engineering aid that equalises all agents, including the fixed PID baseline. Disabling this mechanism causes fixed PID to fail completely (0% success, approximately 10 m overshoot across all scenarios), revealing that the standard evaluation does not constitute a fair test of adaptive versus non-adaptive control. This finding is presented as a methodological contribution: identifying and auditing implicit simulation aids is essential when benchmarking adaptive control methods. The confound is characterised analytically and validated experimentally.

## 1.6 Summary of Key Findings

The main experimental results of this thesis, presented in full in Chapter 5, are summarised here to orient the reader:

- Both the context-aware (Stage 5a) and blind (Stage 5b) RL agents achieve 100% position control success across all mass and friction evaluation scenarios, with zero overshoot in every case. This establishes that domain-randomised PPO is a viable approach to adaptive PID gain scheduling under significant parameter uncertainty.

- The context-aware agent settles 14–15% faster than the blind agent (mean settling time 1.25–1.39 s versus 1.45–1.57 s across scenarios), confirming the benefit of physics context in the observation. The blind agent succeeds, but pays a speed penalty for the implicit dynamics inference it must perform from trajectory shape alone.

- Both RL agents generalise robustly to out-of-distribution conditions: mass of 35 kg (far outside the training range of [5, 20] kg) and friction of 0.05 (well below the training minimum of 0.1), achieving 100% success. This suggests the learned gain-scheduling policy captures generalisable structure rather than overfitting to the training distribution.

- The brake integral reset confound, when removed, causes fixed PID to fail with massive overshoot across all scenarios. This invalidates the apparent speed advantage of fixed PID over the RL agents observed in the standard evaluation. Without this aid, the environment presents a substantially harder control problem that fixed gains cannot solve, while the RL agents are expected to benefit from a recurrent policy architecture that can learn to suppress integral accumulation proactively — a direction identified for future work.

## 1.7 Thesis Outline

The remainder of this thesis is organised as follows:

**Chapter 2 — Background and Related Work** provides the technical background required to understand the approach and situate it in the literature. It covers PID control and classical tuning methods, adaptive control including MRAC, the reinforcement learning framework and Proximal Policy Optimisation algorithm, domain randomisation and its role in policy generalisation, frame stacking as implicit temporal memory, and a review of prior work on RL-based PID gain scheduling.

**Chapter 3 — System Design and Methodology** describes the MuJoCo simulation environment, the PID control layer and gain parameterisation, the domain randomisation and disturbance scheme, the policy architecture and PPO training configuration, the curriculum learning strategy, the classical fixed PID baseline, and the brake integral reset mechanism.

**Chapter 4 — Experimental Setup** defines the evaluation protocol, performance metrics, agent configurations, and the full set of evaluation scenarios including out-of-distribution conditions.

**Chapter 5 — Results and Analysis** presents experimental results across static and dynamic evaluation scenarios, analyses the context-aware versus blind agent ablation, and provides the full confound analysis including the no-reset validation experiment with exact numbers and trajectory visualisations.

**Chapter 6 — Discussion and Limitations** interprets the findings in the context of the two research questions, critically evaluates the experimental design, and identifies the limitations that constrain the conclusions that can be drawn.

**Chapter 7 — Conclusion and Future Work** summarises the thesis findings, answers the research questions directly, and proposes concrete future directions including recurrent policy architectures for integral-windup-free control, multi-seed evaluation, earlier disturbance windows for genuine robustness testing, and transfer of the gain-scheduling policy to alternative PID-controlled systems.

[^astrom1995]: Åström, K. J., & Hägglund, T. (1995). *PID Controllers: Theory, Design, and Tuning* (2nd ed.). Instrument Society of America. See Chapter 1, where they report that in process control “more than 95%” of loops are PID type, most actually PI.