# Chapter 2: Background and Related Work

This chapter develops the technical background required to understand the approach taken in this thesis and situates it within the existing literature. The sections are organised to build a motivating argument. Section 2.1 establishes PID control as the dominant industrial mechanism and describes its fundamental limitation. Section 2.2 reviews classical adaptive control methods — gain scheduling and Model Reference Adaptive Control — and identifies what each requires from the designer. Section 2.3 introduces reinforcement learning as a model-free alternative for control problems. Section 2.4 explains domain randomisation and frame stacking, the specific techniques used to train a policy that generalises across unknown dynamics. Section 2.5 reviews prior work on RL-based PID gain scheduling and identifies the gap that this thesis addresses.

---

## 2.1 PID Control

The proportional-integral-derivative (PID) controller is the most widely deployed feedback mechanism in industrial motion control. Surveys of process control installations consistently report that more than 90% of deployed control loops use PID or PI variants [^astrom1995]. This prevalence reflects genuine practical advantages: the three-term structure is interpretable, tunable with minimal system knowledge, and well-understood theoretically for a broad class of linear plants.

The standard continuous-time PID control law is:

$$u(t) = K_p e(t) + K_i \int_0^t e(\tau)\, d\tau + K_d \dot{e}(t)$$

where $e(t) = r(t) - y(t)$ is the error between the reference $r(t)$ and the plant output $y(t)$, and $K_p$, $K_i$, $K_d$ are the proportional, integral, and derivative gains respectively. The proportional term responds to the current error magnitude; the integral term accumulates past error to eliminate steady-state offset; the derivative term anticipates the rate of change and provides damping.

Gain tuning is the central practical challenge of PID deployment. The Ziegler–Nichols method [^ziegler1942] provides heuristic tuning rules derived from open-loop step responses or from measurements of the plant's ultimate gain and period of oscillation. SIMC (Simple Internal Model Control) [^skogestad2003] provides analytically derived rules for first-order-plus-dead-time plant approximations. The Internal Model Control (IMC) framework [^rivera1986] offers a more systematic approach grounded in model inversion, providing explicit tuning formulae as a function of desired closed-loop bandwidth. Each method produces a fixed gain set that is calibrated to a specific operating condition and remains constant during operation.

This is the fundamental limitation. A gain set tuned for a plant with parameters $\theta_0$ will perform as intended only when the plant parameters remain at $\theta_0$. When mass, friction, load, or other physical parameters change — whether due to payload variation, surface transitions, component wear, or environmental disturbances — the gain set becomes miscalibrated. The proportional gain may be too aggressive for a heavier plant, causing overshoot or oscillation; too conservative for a lighter one, causing sluggishness. The integral gain, tuned for a nominal compliance, may accumulate excessively during a changed response profile. In short, fixed gains encode a fixed assumption about the plant, and that assumption fails when the plant changes.

In practice this limitation is often managed by operating within a narrow envelope where parameter variation is small and the gain set remains approximately correct. But autonomous systems deployed across diverse conditions — varying payloads, surfaces, or use cases — cannot be confined to this narrow envelope. The challenge of maintaining reliable control across unknown or varying plant parameters motivates the adaptive control approaches reviewed in the next section.

---

## 2.2 Adaptive Control

Adaptive control encompasses techniques that adjust controller parameters online in response to changes in the plant. The two most relevant approaches for this thesis are gain scheduling and Model Reference Adaptive Control (MRAC).

### 2.2.1 Gain Scheduling

Gain scheduling maintains a set of pre-computed gain tables indexed by measurable operating conditions. At runtime, the controller reads a scheduling variable — typically a directly measurable physical quantity such as speed, altitude, or load — and interpolates or selects the appropriate gain set from the table [^rugh1991]. When the scheduling variable is well-chosen and the operating conditions can be measured directly, gain scheduling is effective and predictable.

The limitation is the design burden it imposes. The engineer must identify the relevant scheduling variable, enumerate the operating modes or parameter ranges, tune a separate gain set for each regime, and define the interpolation or switching logic between them. For a single-axis system with well-understood parameter variation, this is tractable. For systems where the uncertainty is continuous, high-dimensional, only partially observable, or poorly characterised in advance, constructing a complete and reliable gain table is expensive or infeasible. Gain scheduling encodes prior knowledge about the uncertainty; it cannot cope with uncertainty that was not anticipated at design time.

### 2.2.2 Model Reference Adaptive Control

Model Reference Adaptive Control (MRAC) takes a more principled approach to online adaptation [^astrom_wittenmark1995]. A reference model specifies the desired closed-loop dynamics; an adaptation law — typically derived from Lyapunov stability theory — continuously adjusts the controller gains to minimise the error between the plant's observed response and the reference model's output. Unlike gain scheduling, MRAC does not require a pre-enumerated table; adaptation is continuous and driven by observed discrepancy.

MRAC's requirement is different: rather than enumerating operating modes, the designer must specify the reference model and derive the adaptation law from an explicit mathematical model of the plant's parametric structure. The adaptation law depends on the form of the plant equations, and its stability guarantees rest on assumptions about the plant's structure and the matching conditions between plant and reference model. When the real plant deviates significantly from the assumed model — due to nonlinearities, unmodelled dynamics, or parameter variation outside the designed range — the adaptation law can become unstable, driving gains to divergent values rather than converging to a correct solution.

In this thesis, MRAC was implemented and evaluated on the target environment as part of the development process. It failed to converge, achieving 0% task success due to gain divergence under the combined effect of the plant's nonlinearity and the 4:1 mass variation range. This failure is discussed further in Section 6.2. It is noted here because it provides direct empirical motivation for abandoning model-based adaptive approaches in favour of the learning-based alternative: when the plant's structure cannot be reliably modelled, adaptation laws derived from that structure will not work.

### 2.2.3 The Model Requirement

Both gain scheduling and MRAC share a structural dependency on prior knowledge. Gain scheduling requires knowledge sufficient to enumerate operating modes and tune gains for each. MRAC requires knowledge sufficient to specify the reference model and derive a valid adaptation law. This shared dependency — the requirement that the designer know enough about the uncertainty to model or parameterise it — is the gap that model-free learning approaches are designed to close.

---

## 2.3 Reinforcement Learning for Control

Reinforcement learning (RL) is a framework for learning sequential decision-making policies from interaction with an environment, without requiring an explicit model of the environment's dynamics [^sutton2018]. The agent observes the current state, selects an action, receives a scalar reward signal, and transitions to a new state. The goal is to learn a policy — a mapping from states to actions — that maximises the expected cumulative discounted reward over time.

### 2.3.1 Markov Decision Process Formulation

RL problems are formalised as Markov Decision Processes (MDPs). An MDP is defined by a tuple $(\mathcal{S}, \mathcal{A}, P, R, \gamma)$ where $\mathcal{S}$ is the state space, $\mathcal{A}$ is the action space, $P(s' | s, a)$ is the transition probability distribution, $R(s, a)$ is the reward function, and $\gamma \in [0, 1)$ is the discount factor that controls the relative importance of immediate versus future rewards. The agent's policy $\pi(a | s)$ defines a probability distribution over actions given the current state. The objective is to find the policy $\pi^*$ that maximises the expected return $\mathbb{E}_\pi \left[ \sum_{t=0}^\infty \gamma^t R(s_t, a_t) \right]$.

For control applications, the MDP state typically includes the system's observable variables — position, velocity, error, and any available sensor readings. The action space corresponds to the control outputs — motor commands, gain adjustments, or other actuator signals. The reward function encodes the control objective: minimising tracking error, penalising overshoot, rewarding task completion.

### 2.3.2 Policy Gradient Methods

Policy gradient methods directly optimise the policy parameters $\theta$ by computing the gradient of the expected return with respect to $\theta$ and updating $\theta$ in the direction of improvement [^sutton2018pg]. The fundamental policy gradient theorem establishes that this gradient can be estimated from sampled trajectories:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a|s) \cdot A^\pi(s, a) \right]$$

where $A^\pi(s, a)$ is the advantage function, measuring how much better action $a$ is in state $s$ compared to the average action under the current policy. Policy gradient methods are well-suited to continuous action spaces, which is relevant here because the gain adjustments output by the scheduling agent are continuous-valued.

### 2.3.3 Proximal Policy Optimisation

Proximal Policy Optimisation (PPO) [^schulman2017] is the specific algorithm used in this work. PPO addresses a core instability of vanilla policy gradient methods: large parameter updates can dramatically change the policy, moving it to a worse region of the policy space from which recovery is difficult. PPO constrains updates by introducing a clipped surrogate objective:

$$L^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[ \min\left( r_t(\theta) \hat{A}_t,\ \text{clip}(r_t(\theta),\ 1 - \varepsilon,\ 1 + \varepsilon)\, \hat{A}_t \right) \right]$$

where $r_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_\text{old}}(a_t | s_t)}$ is the probability ratio between the new and old policies, and $\varepsilon$ is a clipping coefficient (set to 0.2 in this work). The clip prevents the ratio from moving too far from 1, limiting how much the policy can change in a single update while still allowing meaningful improvement. This stability makes PPO robust to hyperparameter choice and suitable for continuous control tasks.

Advantage estimates are computed using Generalised Advantage Estimation (GAE) [^schulman2015]:

$$\hat{A}_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}, \quad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

where $\lambda$ controls the bias-variance tradeoff ($\lambda = 0.95$ in this work). Lower $\lambda$ reduces variance at the cost of bias; higher $\lambda$ reduces bias at the cost of variance. The combination of clipped updates and GAE makes PPO practical for training on-policy with stochastic environments and continuous action spaces — precisely the setting encountered here.

---

## 2.4 Domain Randomisation and Implicit Adaptation

Training a single RL policy to perform well across a range of operating conditions requires the agent to encounter that range during training. Domain randomisation achieves this by sampling environment parameters from a distribution at each episode reset, rather than fixing them to a single value [^tobin2017]. The agent is trained on the resulting diversity of instances, learning a single policy that must perform across all of them. Because no single parameter configuration dominates training, the policy cannot overfit to any one set of dynamics; it must learn behaviours that generalise.

In the context of this thesis, domain randomisation means sampling mass from $[5, 20]$ kg and friction from $[0.1, 2.0]$ at each episode reset. The agent never encounters the same combination twice. Over the course of training, it develops a gain-scheduling policy that performs reliably across the full distribution of dynamics — not because it has been told what the dynamics are, but because it has been trained on all of them simultaneously. At evaluation time, the physics parameters are fixed, but the policy has already learned to handle both extremes of the training distribution.

Domain randomisation was initially developed for sim-to-real transfer in robotics, where the motivation was to make a policy robust to the inevitable discrepancies between simulation and the real world [^tobin2017]. The same principle applies to parameter uncertainty within a single simulation: robustness to parameter variation during training translates to robustness to that variation at deployment.

### 2.4.1 Frame Stacking as Implicit Temporal Memory

A policy that must adapt to varying dynamics benefits from access to trajectory history — not just the current state, but how the system has been responding over recent time steps. A vehicle that has been accelerating slowly is likely heavy; one that overshoots easily is likely light and has high friction. This information is implicit in the recent trajectory, but only visible if the policy has access to more than the instantaneous state.

Frame stacking addresses this by concatenating $k$ consecutive observations into a single input vector before passing it to the policy network [^mnih2015]. Rather than observing only $s_t$, the policy observes $[s_{t-k+1}, \ldots, s_{t-1}, s_t]$. This gives the policy a rolling temporal window from which to infer dynamics without requiring a recurrent architecture.

In this work, $k = 10$ frames are stacked, providing approximately 0.2 s of trajectory history at the simulation timestep. The frame-stacking approach is deliberately chosen over LSTM or GRU architectures to keep the policy architecture simple and the training stable; recurrent policies require additional care in credit assignment and truncated backpropagation through time. The cost is a fixed-size temporal window — unlike a recurrent policy, the frame-stacked MLP cannot maintain state across arbitrarily long time horizons. The adequacy of this window for the dynamics inference task is assessed empirically in Chapter 5.

---

## 2.5 Reinforcement Learning for PID Gain Scheduling

The application of RL to PID control has attracted increasing attention as a way to automate gain tuning and enable online adaptation without explicit system identification. Existing work falls broadly into two categories.

The first category uses RL to tune PID gains offline — prior to deployment — replacing the Ziegler–Nichols or manual tuning process with an automated search over the gain space [^carlucho2017; ^shi2020]. In these approaches, the RL agent trains on a fixed or slowly varying plant and converges to a gain set that is then frozen for deployment. This is fundamentally a tuning automation problem rather than an adaptive control problem: the resulting controller is still a fixed-gain PID, optimised by RL rather than by hand. It does not adapt at deployment time when the plant changes.

The second category uses RL to schedule or adjust gains online during deployment. Within this category, several works assume access to a model of the plant's parameter variation — either an explicit dynamics model or a system identification step that estimates current parameters before computing appropriate gains [^CITATION_NEEDED_model_based_rl_pid]. This relaxes the manual design burden of classical gain scheduling but retains a fundamental dependency: the adaptation mechanism relies on the model being correct. When the plant deviates from the assumed model structure, these approaches degrade in the same way as MRAC.

A smaller body of work addresses model-free online gain scheduling using RL. Relevant examples include [^CITATION_NEEDED_model_free_rl_pid], where neural network policies are trained to output gain adjustments in response to tracking error observations. These works demonstrate that online adaptation is learnable from experience alone. However, they typically evaluate on a single nominal plant or a limited range of parameter variation, and they do not address the question of what information the policy requires to adapt well — specifically, whether explicit physics context in the observation improves adaptation compared to trajectory-based inference.

This thesis addresses both gaps. First, it evaluates domain-randomised PPO across a 4:1 mass range and 20:1 friction range with no plant model, demonstrating that online adaptive gain scheduling is feasible under substantial parameter uncertainty. Second, it directly ablates the role of physics context by comparing a context-aware agent (which observes mass and friction scales) against a blind agent (which must infer dynamics from trajectory history alone), quantifying the cost of blindness in terms of settling time across identical evaluation conditions. The gap addressed is not merely feasibility — prior work has demonstrated feasibility in simpler settings — but the specific question of whether and how much explicit context information matters for adaptation efficiency when the alternative is implicit inference from observation history.

---

[^astrom1995]: Åström, K. J., & Hägglund, T. (1995). *PID Controllers: Theory, Design, and Tuning* (2nd ed.). Instrument Society of America.
[^ziegler1942]: Ziegler, J. G., & Nichols, N. B. (1942). Optimum settings for automatic controllers. *Transactions of the ASME*, 64, 759–768.
[^skogestad2003]: Skogestad, S. (2003). Simple analytic rules for model reduction and PID controller tuning. *Journal of Process Control*, 13(4), 291–309.
[^rivera1986]: Rivera, D. E., Morari, M., & Skogestad, S. (1986). Internal model control: PID controller design. *Industrial & Engineering Chemistry Process Design and Development*, 25(1), 252–265.
[^rugh1991]: Rugh, W. J. (1991). Analytical framework for gain scheduling. *IEEE Control Systems Magazine*, 11(1), 79–84.
[^astrom_wittenmark1995]: Åström, K. J., & Wittenmark, B. (1995). *Adaptive Control* (2nd ed.). Addison-Wesley.
[^sutton2018]: Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
[^sutton2018pg]: Sutton, R. S., McAllester, D. A., Singh, S. P., & Mansour, Y. (2000). Policy gradient methods for reinforcement learning with function approximation. *Advances in Neural Information Processing Systems*, 12, 1057–1063.
[^schulman2017]: Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms. *arXiv preprint arXiv:1707.06347*.
[^schulman2015]: Schulman, J., Moritz, P., Levine, S., Jordan, M., & Abbeel, P. (2015). High-dimensional continuous control using generalized advantage estimation. *arXiv preprint arXiv:1506.02438*.
[^tobin2017]: Tobin, J., Fong, R., Ray, A., Schneider, J., Zaremba, W., & Abbeel, P. (2017). Domain randomization for transferring deep neural networks from simulation to the real world. *2017 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*.
[^mnih2015]: Mnih, V., Kavukcuoglu, K., Silver, D., et al. (2015). Human-level control through deep reinforcement learning. *Nature*, 518(7540), 529–533.
[^carlucho2017]: Carlucho, I., De Paula, M., Wang, S., Petillot, Y., & Acosta, G. G. (2017). Incremental Q-learning strategy for adaptive PID control of mobile robots. *Expert Systems with Applications*, 80, 183–199.
[^shi2020]: Shi, J., He, J., & Cai, Z. (2020). Deep reinforcement learning based PID controller for uncertain nonlinear system. *CITATION_VERIFY — search "Shi 2020 deep reinforcement learning PID" for exact venue and page numbers.*
[^CITATION_NEEDED_model_based_rl_pid]: CITATION NEEDED — search: "reinforcement learning PID gain scheduling plant model" for model-dependent online tuning work.
[^CITATION_NEEDED_model_free_rl_pid]: CITATION NEEDED — search: "model-free reinforcement learning adaptive PID online" for model-free online scheduling examples.
