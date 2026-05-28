# Chapter 3 — System Design and Methodology

This chapter describes the simulation environment, reinforcement learning agents, and training protocol used in this thesis. Two agent variants are evaluated: a Context-Aware Agent (Stage 5a) that directly observes the current mass and friction scales, and a Blind Agent (Stage 5b) that must infer operating dynamics from trajectory history alone. Both are trained using Proximal Policy Optimisation (PPO) on a MuJoCo vehicle simulation with domain-randomised physics. The classical baseline is a fixed-gain PID controller with no adaptation. Section 3.7 describes the brake integral reset mechanism embedded in the environment; its significance as an experimental confound is analysed in Section 5.4.

---

## 3.1 Design Overview

The final system described in this chapter was reached through an iterative development process spanning four training protocol revisions. The initial protocols included a safety speed governor — an engineering layer that overrode the PID output to enforce kinematic braking constraints whenever the vehicle's speed exceeded what was safely stoppable within the remaining distance to target. Although the governor prevented overshooting, it removed the primary adaptive challenge: the RL agent never needed to learn self-braking through gain scheduling, because the governor handled deceleration directly. The agent converged to a policy that approached the target quickly and relied on the safety layer to stop it — a viable strategy within training, but one that demonstrated no gain-scheduling adaptation.

Removing the governor exposed a deeper difficulty: the credit assignment gap between suppressing the integral gain during the approach phase (preventing windup buildup) and being rewarded for successful braking 500–1000 steps later. Three design changes were required to make this learnable. First, a deceleration bonus provided an immediate reward signal for reducing speed within the braking zone, bypassing the long discount horizon. Second, the progress reward was zeroed inside the braking zone (within 2 m of target), removing the incentive to rush the final approach. Third, 20% of training episodes were initialised with the vehicle already within 0.04 m of the target, ensuring the agent experienced the hold completion bonus before learning the full approach trajectory. A cliff termination penalty ($-300$) was also trialled at an intermediate stage but was found to eliminate exploration entirely: the agent's value function saturated at $-300$ everywhere in the braking zone, producing zero policy gradient toward braking behaviour, and was subsequently removed. The final training protocol — referred to as the cliff protocol throughout — combines the three supporting changes without the cliff termination. Its precise parameterisation is described in the sections that follow.

---

## 3.2 Simulation Environment

### 3.2.1 Platform and Vehicle Model

Experiments are conducted in MuJoCo [CITATION], a physics engine designed for efficient rigid-body simulation. The vehicle model is a two-wheeled differential-drive robot with a chassis, two driven rear wheels, and a passive ball-wheel at the front. Wheel torque is applied symmetrically to both drive motors so the robot moves in a straight line along the $x$-axis. The full model geometry and actuator limits are defined in `car_model.xml`.

### 3.2.2 Task

The vehicle starts at position $x = 0$ m at episode reset with zero velocity. It must drive to a target position, then hold within a tolerance band of $\pm 0.05$ m for 25 consecutive steps to register success. During static evaluation the target is fixed at $x = 5.0$ m. During training, the target is sampled from a curriculum-dependent range (Section 3.5). Episode length is capped at 1200 steps during training and 5000 steps during evaluation ($= 24$ s and $= 100$ s respectively at $\Delta t_{\text{RL}} = 0.02$ s).

### 3.2.3 Two-Loop Control Architecture

The system operates as two nested control loops running at different frequencies (Figure 3.1).

The **inner loop** is a discrete PID controller running at the MuJoCo physics frequency of 500 Hz ($\Delta t_{\text{phys}} = 0.002$ s). At each physics step, the PID computes a motor command from the current position error:

$$u(t) = K_p\, e(t) \;+\; K_i \sum_{k=0}^{t} e(k)\,\Delta t_{\text{phys}} \;+\; K_d \frac{e(t) - e(t-1)}{\Delta t_{\text{phys}}}$$

where $e(t) = x_{\text{target}} - x(t)$. The output $u(t)$ is clipped to $[-1, 1]$ and applied as equal torque commands to both drive motors.

The **outer loop** is the RL agent, which runs at 50 Hz ($\Delta t_{\text{RL}} = 0.02$ s, corresponding to a frame skip of 10 physics steps per RL step). At each RL step, the agent receives an observation, selects a gain adjustment action, and holds that action fixed for the 10 physics sub-steps until the next decision point. The PID gains therefore change at 50 Hz while the PID controller itself runs at 500 Hz, providing smooth torque commands at the inner loop frequency.

### 3.2.4 Gain Parameterisation

The RL agent does not output absolute gain values. Instead it outputs a normalised action vector $\mathbf{a} = [a_{K_p}, a_{K_i}, a_{K_d}] \in [-1, 1]^3$, which is mapped to gains as:

$$K_p = \bar{K}_p + \Delta K_p \cdot a_{K_p}, \qquad K_i = \bar{K}_i + \Delta K_i \cdot a_{K_i}, \qquad K_d = \bar{K}_d + \Delta K_d \cdot a_{K_d}$$

where $\bar{K}_p = 1.8$, $\bar{K}_i = 0.7$, $\bar{K}_d = 0.5$ are the base gains and $\Delta K_p = 1.0$, $\Delta K_i = 0.6$, $\Delta K_d = 2.0$ are the per-gain scaling deltas. Effective gain ranges after mapping are $K_p \in [0.8, 2.8]$, $K_i \in [0.1, 1.3]$, $K_d \in [0.0, 2.5]$, with additional clipping to these bounds.

The base gains $\bar{K}_p = 1.8$, $\bar{K}_i = 0.7$, $\bar{K}_d = 0.5$ correspond to a hand-tuned stable controller for nominal conditions (mass = 10 kg, friction = 1.0). When the RL agent outputs $\mathbf{a} = [0, 0, 0]$, the effective gains equal the base values exactly; this is the action used by the classical baseline (Section 3.6).

### 3.2.5 Reward Function

The core per-step reward is:

$$r_t = 5\,\Delta x_t \;-\; 0.75\,|e_t| \;-\; 0.12\,\dot{x}_t^2 \;-\; 2.0\,\max(0,\, x_t - x_{\text{target}})$$

where $\Delta x_t = x_t - x_{t-1}$ is signed displacement progress toward the target, $|e_t|$ is the absolute position error, $\dot{x}_t^2$ penalises excessive speed, and the final term penalises any overshoot beyond the target. Inside the braking zone ($|e_t| < 2.0$ m), the progress term $5\,\Delta x_t$ and the distance cost $0.75\,|e_t|$ are both zeroed, leaving only the velocity and overshoot penalties to shape behaviour in the final approach.

Additional shaping terms supplement the core reward in specific regions. A deceleration bonus of $10\,\max(0, \dot{x}_{t-1} - \dot{x}_t)$ is awarded for each step on which the vehicle slows within the braking zone, providing an immediate credit-assignment signal for braking behaviour. A velocity-squared penalty proportional to $\dot{x}_t^2\,/\,\max(|e_t|, 0.05)$ additionally discourages residual speed in the near-target region. Small bonuses are awarded for entering and remaining within the $\pm 0.05$ m tolerance band, and for reducing velocity toward zero once inside it. On hold completion, a terminal bonus of $+80$ is awarded. If the vehicle exits the environment bounds ($x > x_{\text{target}} + 5$ m or $x < -2$ m), the episode terminates immediately with a penalty of $-100$.

---

## 3.3 Domain Randomisation and Disturbances

### 3.3.1 Episode-Level Randomisation

At every episode reset, vehicle mass is sampled uniformly from $[5, 20]$ kg and the surface friction coefficient from $[0.1, 2.0]$. This produces a 4:1 mass ratio and a 20:1 friction ratio across the training distribution, requiring the policy to generalise across substantially different inertial regimes. The nominal parameter values — mass = 10 kg, friction = 1.0 — lie at the midpoints of both ranges.

### 3.3.2 Mid-Episode Disturbance

To test robustness to sudden parameter changes during operation, a mid-episode disturbance is applied at a random step in $[120, 220]$, corresponding to 1.2–2.2 s after episode start. The current mass is multiplied by a disturbance scale drawn from $[0.9, 1.3]$ and the friction by a scale from $[0.5, 1.4]$. These scales are multiplicative; the resulting post-disturbance parameters remain within or close to the training distribution for most episodes. The disturbance timing and its interaction with agent settling behaviour is discussed as a methodological limitation in Section 6.2.

### 3.3.3 Position Friction Patch

A low-friction corridor occupies the region $x \in [1.5, 2.4]$ m, where friction is reduced to 35% of the episode's nominal friction value at every step (Figure 3.2). The vehicle traverses this corridor on every approach to the 5.0 m target, providing a consistent within-episode traction perturbation.

### 3.3.4 Note on Friction and Dynamics

In the rolling contact model used in this MuJoCo environment, kinetic friction does not affect translational dynamics when the wheels roll without slipping. The primary source of variation in vehicle response is therefore vehicle mass (inertia), not friction. The friction randomisation and friction patch are present in the environment but their dynamical effect is limited. This is discussed as a limitation in Section 6.2.

---

## 3.4 Policy Architecture

### 3.4.1 Algorithm

Both RL agents are trained with Proximal Policy Optimisation (PPO) [CITATION], an on-policy actor-critic algorithm. PPO constrains the size of each policy update via a clipped surrogate objective:

$$\mathcal{L}^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t\!\left[\min\!\left(r_t(\theta)\,\hat{A}_t,\ \mathrm{clip}(r_t(\theta),\, 1-\varepsilon,\, 1+\varepsilon)\,\hat{A}_t\right)\right]$$

where $r_t(\theta) = \pi_\theta(a_t|s_t)\,/\,\pi_{\theta_\text{old}}(a_t|s_t)$ is the probability ratio between the updated and old policies, $\hat{A}_t$ is the advantage estimate, and $\varepsilon = 0.2$ is the clip coefficient. Advantages are estimated using Generalised Advantage Estimation (GAE) with $\lambda = 0.95$ and discount factor $\gamma = 0.99$.

### 3.4.2 Observation Space

Each single-step observation consists of 6 dimensions for Stage 5b and 8 dimensions for Stage 5a, as listed in Table 3.1.

**Table 3.1 — Per-frame observation dimensions**

| Dim | Description | Stage 5a | Stage 5b |
|:---:|-------------|:--------:|:--------:|
| 0 | Position $x$ (m) | ✓ | ✓ |
| 1 | Velocity $\dot{x}$ (m/s) | ✓ | ✓ |
| 2 | Error $e = x_{\text{target}} - x$ (m) | ✓ | ✓ |
| 3 | Previous normalised gain action $a_{K_p} \in [-1, 1]$ | ✓ | ✓ |
| 4 | Previous normalised gain action $a_{K_i} \in [-1, 1]$ | ✓ | ✓ |
| 5 | Previous normalised gain action $a_{K_d} \in [-1, 1]$ | ✓ | ✓ |
| 6 | Mass scale (current mass / 10 kg) | ✓ | ✗ |
| 7 | Friction scale (current friction / 1.0) | ✓ | ✗ |

Dimensions 3–5 contain the previous normalised gain action $\mathbf{a}_{t-1} = [a_{K_p}, a_{K_i}, a_{K_d}]$, not absolute gain values. The corresponding absolute gains can be recovered via $K_p = 1.8 + 1.0 \cdot a_{K_p}$ (and analogously for $K_i$, $K_d$), but the policy network operates on the normalised coordinates directly, keeping all observation dimensions in a well-conditioned numerical range.

Stage 5a additionally observes the current mass scale and friction scale (dimensions 6–7). These are computed at episode reset from the sampled values and updated at the disturbance step. Stage 5b omits these dimensions and must infer the operating dynamics from the trajectory shape within its observation window alone.

### 3.4.3 Frame Stacking

To provide temporal context without recurrence, 10 consecutive single-step observations are concatenated to form the policy input (Figure 3.4). The resulting input vectors are 80-dimensional for Stage 5a ($8 \times 10$ frames) and 60-dimensional for Stage 5b ($6 \times 10$ frames). At $\Delta t_{\text{RL}} = 0.02$ s per step, the stack covers a rolling window of 0.2 s of trajectory history. This window allows the policy to observe the velocity profile from the onset of motion and use the resulting position-velocity trajectory to infer inertial properties — a heavier vehicle accelerates more slowly under the same PID output, producing a distinctive profile within the observation window.

### 3.4.4 Network Architecture

Both Stage 5a and 5b use a multilayer perceptron (MLP) with separate actor and critic networks; there is no shared trunk. Each network consists of two hidden layers of 64 units with Tanh activations (Figure 3.4):

- **Actor:** $d_{\text{obs}}$-dim input → Linear(64) → Tanh → Linear(64) → Tanh → Linear(3) → action mean $\boldsymbol{\mu}$
- **Critic:** $d_{\text{obs}}$-dim input → Linear(64) → Tanh → Linear(64) → Tanh → Linear(1) → value $V$

where $d_{\text{obs}} = 80$ for Stage 5a and $d_{\text{obs}} = 60$ for Stage 5b. Weights are initialised with orthogonal initialisation [CITATION].

The actor outputs the mean of a Gaussian distribution over the three-dimensional action space. The distribution's log-standard-deviation is a learnable parameter, initialised to $-1.0$ (initial standard deviation $\approx 0.37$). During evaluation, the distribution mean is used deterministically, with no sampling noise.

### 3.4.5 PPO Hyperparameters

Full training hyperparameters are listed in Table 3.2. All experiments use a single training seed (seed 7). Multi-seed variance is not quantified; this is noted as a limitation in Section 6.2.

**Table 3.2 — PPO training hyperparameters**

| Parameter | Value |
|-----------|-------|
| Learning rate | $3 \times 10^{-4}$, linearly decayed to 0 |
| Discount factor $\gamma$ | 0.99 |
| GAE $\lambda$ | 0.95 |
| Clip coefficient $\varepsilon$ | 0.2 |
| Entropy coefficient | 0.0 |
| Value function coefficient | 0.5 |
| Minibatch size | 64 |
| Update epochs per rollout | 10 |
| Rollout steps per environment | 2048 |
| Parallel training environments | 4 |
| Total environment steps | 1,000,000 |
| Frame stack size | 10 |
| Training seed | 7 |

---

## 3.5 Curriculum Learning

Training target distance increases across four phases, each spanning 25% of total training timesteps (Table 3.3, Figure 3.3).

**Table 3.3 — Curriculum learning phases**

| Phase | Target range (m) | Timesteps |
|:-----:|:---------------:|:---------:|
| 1 | 1.0 – 3.0 | 0 – 250,000 |
| 2 | 1.0 – 5.0 | 250,000 – 500,000 |
| 3 | 1.0 – 7.0 | 500,000 – 750,000 |
| 4 | 1.0 – 10.0 | 750,000 – 1,000,000 |

The motivation for gradual target expansion is the integral windup problem. At short approach distances (Phase 1, up to 3 m), a moderate integral gain does not accumulate enough error to prevent braking — the agent can learn successful settling behaviour without yet needing to solve the gain-scheduling challenge. As the curriculum progresses to Phase 4 (up to 10 m), the vehicle spends more time in transit, the integral term accumulates significantly, and the agent must have developed an appropriate Ki-reduction strategy before the final approach. Starting with short targets ensures the hold completion bonus is reachable early in training, establishing a stable reward baseline before the harder control problem is introduced.

---

## 3.6 Classical Baseline

The classical baseline is a fixed-gain PID controller with $K_p = 1.8$, $K_i = 0.7$, $K_d = 0.5$ and a constant action output of $\mathbf{a} = [0, 0, 0]$ at every step. The gains never deviate from their base values and no adaptation occurs in response to the vehicle's dynamics. The baseline operates under the same two-loop control architecture as the RL agents: it is implemented as an RL agent whose policy always outputs zero, and the PID runs at 500 Hz inside the simulation inner loop.

The base gains were hand-tuned for nominal conditions (mass = 10 kg, friction = 1.0) and constitute the best achievable non-adaptive performance with those specific gain values. The baseline serves as a reference point: the RL agents must demonstrate value — in terms of reliability across unknown dynamics — to justify the additional complexity of the learning-based approach.

The interaction between the fixed-gain baseline and the brake integral reset mechanism is analysed in detail in Section 5.4.

---

## 3.7 The Brake Integral Reset Mechanism

During development, a `brake_integral_reset` mechanism was introduced to prevent integral windup from causing overshoot at the end of long approach trajectories. When the vehicle enters the braking zone ($|e_t| < 2.0$ m) for the first time in an episode, the PID integrator is set to zero. For the remainder of the episode within the braking zone, the integrator is held at zero on every physics sub-step. This prevents the integral accumulated during the approach phase from overpowering the derivative term's braking authority at the moment of deceleration.

The mechanism is active by default for all agents — including the fixed-gain baseline — in all standard training and evaluation runs. The consequence for the experimental comparison is non-trivial: the integral reset effectively solves the hardest part of the control problem (windup-driven overshoot) at the engineering level, reducing the practical difference between an adaptive and a non-adaptive controller in this environment. The full implications are quantified in Section 5.4, where the baseline is evaluated with the reset disabled.
