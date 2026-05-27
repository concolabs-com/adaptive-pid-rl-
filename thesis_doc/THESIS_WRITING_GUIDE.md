# Thesis Writing Guide
## Meta-RL for Adaptive PID Gain Scheduling Under Unknown Vehicle Dynamics

This document contains everything needed to write the thesis — chapter structure, narrative, exact numbers, formulas, figures list, and honest framing guidance. Write from this, not from memory.

---

## Suggested Chapter Structure

1. Introduction
2. Background and Related Work
3. System Design and Methodology
4. Experimental Setup
5. Results and Analysis
6. Discussion and Limitations
7. Conclusion and Future Work

Target length: ~15,000–20,000 words for the thesis.

---

---

## Chapter 1 — Introduction (~1,500 words)

### What to argue

PID controllers are ubiquitous in industrial motion control but their performance degrades when physical parameters (mass, friction, load) are unknown or vary during operation. Classical adaptive control (gain scheduling, MRAC) requires an explicit model of parameter variation. Reinforcement learning offers a model-free alternative: an agent can learn a gain-scheduling policy purely from experience, adapting implicitly to whatever dynamics it encounters.

The key question this thesis addresses:

> *Can a reinforcement learning agent learn to schedule PID gains in real time, and does providing the agent with physics context (measured mass and friction) improve adaptation compared to an agent that must infer dynamics from trajectory history alone?*

### Paragraph flow

1. PID control is dominant in practice — cite why (simplicity, interpretability, proven track record). State the limitation: fixed gains work well only within the operating regime for which they were tuned.
2. The adaptive control problem — when mass or friction changes unexpectedly, fixed gains either overshoot (too aggressive) or under-perform (too conservative).
3. Classical solutions (gain scheduling tables, MRAC) — require a system model or explicit parameterisation of the uncertainty. Costly to design, brittle when uncertainty is out-of-model.
4. RL as an alternative — learns from interaction, no explicit model required. Meta-RL specifically: train across a distribution of environments so the policy adapts online to whichever environment it finds itself in.
5. The specific contribution — an RL-based PID gain scheduler trained on a MuJoCo vehicle simulation, evaluated across mass and friction uncertainty, with an explicit ablation of what happens when the agent cannot observe its own physics parameters.
6. Key findings summary (one sentence each — expand in Chapter 5).
7. Thesis outline paragraph.

### Key claim to stake
Both a context-aware and a blind RL agent learn to control the vehicle across a range of unknown dynamics. The context-aware agent adapts faster (~14–15% lower settling time). An important experimental confound is identified and analysed.

---

---

## Chapter 2 — Background and Related Work (~3,000 words)

### Sections to include

#### 2.1 PID Control
- Standard PID formulation: `u(t) = Kp·e(t) + Ki·∫e(τ)dτ + Kd·ė(t)`
- Tuning methods: Ziegler–Nichols, SIMC (used in Stage 1 to establish initial gains), IMC
- Limitation: fixed gains assume fixed plant parameters

#### 2.2 Adaptive Control
- Gain scheduling: lookup tables, requires prior parameterisation
- Model Reference Adaptive Control (MRAC): adapts gains online to match a reference model
  - In this work, MRAC was implemented and evaluated — **it failed to converge** (0% success, settling timeout). Cite this as motivation for the learning-based approach.
- Self-tuning regulators

#### 2.3 Reinforcement Learning for Control
- MDP formulation: state s, action a, reward r, transition P
- Policy gradient methods — briefly explain
- PPO (Proximal Policy Optimisation): the algorithm used here. Key equations:
  - Clipped surrogate objective: `L^CLIP = E[min(r_t(θ)·A_t, clip(r_t(θ), 1-ε, 1+ε)·A_t)]`
  - ε = 0.2 (used in this work)
  - GAE for advantage estimation: λ = 0.95, γ = 0.99

#### 2.4 Meta-RL and Domain Randomisation
- Meta-RL: train across a distribution of tasks so the policy adapts to new tasks at test time
- Domain randomisation: randomise environment parameters during training (mass, friction) — the policy learns to be robust across the distribution
- Frame stacking as implicit memory: concatenating k past observations gives the policy a temporal window to infer dynamics from trajectory shape
- Relevant papers to cite (search for):
  - OpenAI domain randomisation for sim-to-real (Tobin et al., 2017)
  - RL for PID tuning (Shi et al., 2020 or similar)
  - Meta-RL survey papers
  - Frame stacking for locomotion (note: your use is the same principle)

#### 2.5 RL for PID Gain Scheduling (Direct Related Work)
- Search: "reinforcement learning PID gain scheduling", "adaptive PID RL", "online PID tuning neural network"
- Key angle to argue: most prior work either (a) uses RL to tune gains offline (not adaptive), or (b) assumes the plant model is known. This work addresses online adaptive scheduling with unknown dynamics.

#### 2.6 Integral Windup (technical background for Chapter 5 confound discussion)
- Explain what integral windup is: when the integrator accumulates during a long approach, the i-term overwhelms the d-term during braking
- Formula: at steady approach, `i_term = Ki × ∫₀ᵀ e(t) dt ≈ Ki × e_avg × T`
- For a 5 m approach at average error ≈ 2.5 m over T ≈ 1.5 s: `i_term ≈ 0.7 × 2.5 × 1.5 = 2.6`
- To brake: needs `Kd × |vel| > i_term` → at vel = 0.5 m/s and i_term = 2.6: `Kd > 5.2`
- Fixed Kd = 0.5 → braking impossible without integral reset. This is why the no-reset baseline fails.

---

---

## Chapter 3 — System Design and Methodology (~3,500 words)

### 3.1 Simulation Environment

**Platform:** MuJoCo physics engine. 2-wheeled vehicle on a flat surface.

**Task:** Drive from position 0 m to target position 5.0 m. Hold within ±0.05 m for 25 consecutive steps to register success.

**Physics parameters randomised at each episode:**
- Mass: sampled from [5, 20] kg (nominal = 10 kg)
- Friction: sampled from [0.1, 2.0] (nominal = 1.0)

**PID controller** runs at every physics sub-step. Agent outputs gain adjustments each step of the outer control loop.

**Gain parameterisation:**
```
kp = base_kp + delta_kp × a_kp     where a_kp ∈ [-1, 1]
ki = base_ki + delta_ki × a_ki
kd = base_kd + delta_kd × a_kd
```
Base gains: Kp_base = 1.8, Ki_base = 0.7, Kd_base = 0.5
Deltas: ΔKp = 1.0, ΔKi = 0.6, ΔKd = 2.0
Effective ranges: Kp ∈ [0.8, 2.8], Ki ∈ [0.1, 1.3], Kd ∈ [0.0, 2.5]

**Reward function** (per step):
```
r = progress_reward - 0.75·dist_cost - 0.12·vel_cost - 2.0·overshoot
```
- `progress_reward`: positive when reducing error, negative when regressing, zero in holding zone
- `dist_cost`: |error| from target
- `vel_cost`: velocity² (penalises excessive speed)
- `overshoot`: max(0, pos - target) — direct penalty for passing the target
- Terminal bonus: +50 on successful hold completion

**Additional training pressures:**
- Brake zone deceleration reward: `+10 × |vel|²` when in approach cutoff zone (|error| < 2.0 m) — rewards slowing down
- `near_target_init_prob = 0.20`: 20% of training episodes start with vehicle already near target, giving the agent settling experience

### 3.2 Domain Randomisation and Disturbances

**Episode-level randomisation:** mass and friction resampled each reset.

**Mid-episode disturbance** (applied during training and dynamic evaluation):
- Fires at a random step in [120, 220] (≈ 1.2–2.2 s after episode start)
- Multiplies current mass by scale ∈ [0.9, 1.3]
- Multiplies current friction by scale ∈ [0.5, 1.4]

**Position friction patch:**
- Friction reduced to 35% of nominal in the region x ∈ [1.5, 2.4] m
- Applied every step — the vehicle must learn to handle this reduced-friction corridor

**Note on friction and dynamics:** In this simulation, wheels roll without slipping, so kinetic friction does not directly affect the vehicle's translational dynamics. The mass parameter (inertia) is the dominant source of uncertainty. This is discussed as a limitation in Chapter 6.

### 3.3 Policy Architecture

**Algorithm:** Proximal Policy Optimisation (PPO), on-policy

**Hyperparameters:**
| Parameter | Value |
|-----------|-------|
| Learning rate | 3 × 10⁻⁴ (linearly decayed) |
| γ (discount) | 0.99 |
| λ (GAE) | 0.95 |
| Clip coefficient (ε) | 0.2 |
| Entropy coefficient | 0.0 |
| Value function coefficient | 0.5 |
| Minibatch size | 64 |
| Update epochs | 10 |
| Rollout steps per update | 2048 (per env) |
| Parallel environments | 4 |
| Total training steps | 1,000,000 |
| Training seed | 7 |

**Network:** MLP with frame stacking. 10 consecutive observations concatenated before being fed to the policy network. This gives the policy a 10-step temporal window (~0.2 s of history at simulation dt) without recurrence.

**Observation space per frame:**

| Dimension | Description | Stage 5a | Stage 5b |
|-----------|-------------|----------|----------|
| 0 | Position (m) | ✓ | ✓ |
| 1 | Velocity (m/s) | ✓ | ✓ |
| 2 | Error (target − pos) | ✓ | ✓ |
| 3 | Current Kp | ✓ | ✓ |
| 4 | Current Ki | ✓ | ✓ |
| 5 | Current Kd | ✓ | ✓ |
| 6 | Mass scale | ✓ | ✗ |
| 7 | Friction scale | ✓ | ✗ |

Total obs size: 80-dim (Stage 5a: 8 × 10), 60-dim (Stage 5b: 6 × 10)

**Action space:** 3-dim continuous, tanh-squashed to [-1, 1]. Interpreted as [Δa_kp, Δa_ki, Δa_kd].

### 3.4 Curriculum Learning

Training target distance increases in four phases, each covering 25% of total timesteps:

| Phase | Target range | Timesteps |
|-------|-------------|-----------|
| 1 | 1.0 – 3.0 m | 0 – 250 K |
| 2 | 1.0 – 5.0 m | 250 K – 500 K |
| 3 | 1.0 – 7.0 m | 500 K – 750 K |
| 4 | 1.0 – 10.0 m | 750 K – 1,000 K |

Motivation: starting with short, easy targets lets the agent first learn the settling behaviour, then gradually encounter the more challenging long-distance approach where integral windup becomes significant.

### 3.5 Classical Baseline

**Fixed PID:** action = [0, 0, 0] at every step. Gains fixed at base values: Kp = 1.8, Ki = 0.7, Kd = 0.5. No learning, no adaptation.

This represents the best achievable performance by a non-adaptive controller with correctly pre-tuned gains. If RL cannot match or exceed this under uncertainty, it provides no practical benefit.

### 3.6 The Brake Integral Reset Mechanism

During development, a `brake_integral_reset` mechanism was included in the environment. When the vehicle enters the braking zone (|error| < 2.0 m), the PID integrator is zeroed. This removes accumulated integral windup at the critical moment of deceleration.

This mechanism is active by default for ALL agents — including the fixed PID baseline. Its effect is analysed in Section 5.4.

---

---

## Chapter 4 — Experimental Setup (~1,500 words)

### 4.1 Evaluation Protocol

- **Evaluation seeds:** 70,000 – 70,009 (10 independent episodes per scenario)
- **Max episode length:** 5,000 steps
- **Success criterion:** position within ±0.05 m of target for 25 consecutive steps

**Metrics reported:**
| Metric | Definition |
|--------|-----------|
| Success rate (%) | Episodes where hold criterion is met |
| Settling time (s) | Time from episode start to first entry into and maintenance of ±0.05 m band |
| Overshoot (m) | max(0, max_position − target) |
| IAE | ∫|error(t)| dt — integrated absolute error over episode |
| Final abs error (m) | |error| at last step |

### 4.2 Evaluation Scenarios

**Static evaluation** — physics fixed for the full episode:

| Scenario | Mass | Friction | Notes |
|----------|------|----------|-------|
| Standard | 10.0 kg | 1.0 | Nominal / training centre |
| Heavy and Slippery | 20.0 kg | 0.2 | Upper training boundary |
| Light and Grippy | 5.0 kg | 2.0 | Lower training boundary |

**Dynamic evaluation** — same scenarios plus mid-episode disturbance (step 120–220, mass ×[0.9,1.3], friction ×[0.5,1.4]) plus OOD conditions:

| Scenario | Mass | Friction | Notes |
|----------|------|----------|-------|
| Standard | 10.0 kg | 1.0 | |
| Heavy and Slippery | 20.0 kg | 0.2 | |
| Light and Grippy | 5.0 kg | 2.0 | |
| OOD Ultra Heavy | 35.0 kg | 1.0 | Outside training range |
| OOD Ultra Slippery | 20.0 kg | 0.05 | Outside training range |

### 4.3 Agents Evaluated

| Agent | Script | Training steps | Obs dims |
|-------|--------|---------------|----------|
| Fixed PID | `stage_baseline_fixed_pid.py` | 0 (no training) | — |
| Stage 5a — Context RL | `stage5a_context_cliff.py` | 1,000,000 | 8 × 10 = 80 |
| Stage 5b — Blind RL | `stage5b_blind_cliff.py` | 1,000,000 | 6 × 10 = 60 |

### 4.4 Reproducibility

All code, configuration files (`stage2_config.json`), trained model weights (`.pth`), and raw results (`eval_seed_summary.csv`, `eval_raw.csv`) are provided in the accompanying `thesis_submission/` directory. Training seed 7 is used for all reported models.

---

---

## Chapter 5 — Results and Analysis (~4,000 words)

### 5.1 Training Convergence

**What to write:** describe the training curves (file: `results/stage5a_context_rl/stage2_learning_curve.png` and `results/stage5b_blind_rl/stage2_learning_curve.png`). Both agents showed improving episode returns over 1M steps. The curriculum transitions are visible as step changes in the difficulty-adjusted reward.

Mention that training was run once (seed 7). Multi-seed variance was not quantified — noted as a limitation.

### 5.2 Static Evaluation Results

#### Exact numbers (copy directly into thesis tables):

**Table 5.1 — Static evaluation, all agents**

| Agent | Scenario | Success | Settling (s) | Overshoot (m) | IAE |
|-------|----------|---------|-------------|---------------|-----|
| Fixed PID | Standard | 100% | 1.116 | 0.000 | 2.653 |
| Fixed PID | Heavy & Slippery | 100% | 1.136 | 0.000 | 2.789 |
| Fixed PID | Light & Grippy | 100% | 1.254 | 0.000 | 2.939 |
| Stage 5a (Context) | Standard | 100% | 1.254 | 0.000 | 2.682 |
| Stage 5a (Context) | Heavy & Slippery | 100% | 1.278 | 0.000 | 2.816 |
| Stage 5a (Context) | Light & Grippy | 100% | 1.390 | 0.000 | 2.967 |
| Stage 5b (Blind) | Standard | 100% | 1.446 | 0.000 | 2.785 |
| Stage 5b (Blind) | Heavy & Slippery | 100% | 1.476 | 0.000 | 2.917 |
| Stage 5b (Blind) | Light & Grippy | 100% | 1.568 | 0.000 | 3.061 |

All results deterministic across seeds 70000–70009 (policy is deterministic at evaluation; physics parameters are fixed per scenario).

#### Analysis to write:

**Result 1 — All agents succeed.** All three agents achieve 100% success across all mass/friction scenarios with zero overshoot. This establishes that the RL agents have learned a viable gain-scheduling policy under domain randomisation.

**Result 2 — Context RL is faster than Blind RL.**

Settling time comparison, Stage 5a vs Stage 5b:
- Standard: 1.254 s vs 1.446 s → Stage 5b is **15.3% slower**
- Heavy & Slippery: 1.278 s vs 1.476 s → Stage 5b is **15.5% slower**
- Light & Grippy: 1.390 s vs 1.568 s → Stage 5b is **12.8% slower**
- **Average: ~14.5% slower for blind RL**

Interpretation: Stage 5b, lacking mass and friction observations, must infer the dynamics from the trajectory shape within its 10-frame window. It succeeds, but the additional inference cost manifests as slower settling. Stage 5a, given the dynamics directly, can select appropriate gains immediately. This confirms that physics context is beneficial.

**Result 3 — Fixed PID is fastest.**

Fixed PID settling times are 10–12% faster than Stage 5a, and 25–30% faster than Stage 5b:
- Standard: 1.116 s (fixed) vs 1.254 s (5a) vs 1.446 s (5b)
- Heavy & Slippery: 1.136 s vs 1.278 s vs 1.476 s

This result is discussed in Section 5.4 (confound analysis).

### 5.3 Dynamic Evaluation Results (with Disturbances and OOD)

#### Exact numbers:

**Table 5.2 — Dynamic evaluation, Stage 5a**

| Scenario | Success | Settling (s) mean ± range | Overshoot |
|----------|---------|--------------------------|-----------|
| Standard | 100% | 1.47 ± 0.01 | 0 m |
| Heavy & Slippery | 100% | 1.47 ± 0.01 | 0 m |
| Light & Grippy | 100% | 1.49 ± 0.04 | 0 m |
| **OOD Ultra Heavy (35 kg)** | **100%** | 1.47 ± 0.03 | 0 m |
| **OOD Ultra Slippery** | **100%** | 1.47 ± 0.01 | 0 m |

**Table 5.3 — Dynamic evaluation, Stage 5b**

| Scenario | Success | Settling (s) mean ± range | Overshoot |
|----------|---------|--------------------------|-----------|
| Standard | 100% | 1.45 ± 0.01 | 0 m |
| Heavy & Slippery | 100% | 1.48 ± 0.01 | 0 m |
| Light & Grippy | 100% | 1.54 ± 0.06 | 0 m |
| **OOD Ultra Heavy (35 kg)** | **100%** | 1.54 ± 0.02 | 0 m |
| **OOD Ultra Slippery** | **100%** | 1.48 ± 0.01 | 0 m |

#### Analysis to write:

**Result 4 — Both RL agents generalise to OOD conditions.** The OOD Ultra Heavy scenario (35 kg) is 75% above the training mass ceiling of 20 kg. Both agents achieve 100% success with settling times comparable to in-distribution scenarios. This suggests the learned gain-scheduling policy generalises beyond the training distribution.

**Result 5 — Mid-episode disturbances had no measurable impact.** Settling times in dynamic evaluation are within 0.1–0.2 s of static evaluation — effectively the same. This is because the disturbance fires at steps 120–220 (≈ 1.2–2.2 s after episode start), but both agents typically settle within 1.3–1.6 s. The disturbance fires after settling in most episodes and has no chance to disrupt the final hold. This is a **methodological limitation** — discuss in Chapter 6.

**Result 6 — Fixed PID also robustly handles disturbances.** The fixed PID dynamic evaluation shows settling times of 1.116–1.142 s across seeds, indistinguishable from its static performance. Again, this is because the disturbance fires after settling.

### 5.4 Confound Analysis — The Brake Integral Reset

This section is your most important scientific contribution. Write it carefully.

#### What to explain:
The `brake_integral_reset` mechanism in the simulation environment zeros the PID integrator when the vehicle enters the braking zone (distance to target < 2.0 m). This was included to prevent integral windup — the phenomenon where accumulated integral error during a long approach overwhelms the derivative term's braking authority.

The mechanism fires for all agents, including the fixed PID baseline.

#### Why it matters:
Without integral windup, a pre-tuned fixed PID at Kp = 1.8, Ki = 0.7, Kd = 0.5 can brake effectively in the final 2 m regardless of how much integral accumulated during the approach. The RL agents learn a more conservative strategy (they reduce Ki and raise Kd in anticipation of braking), which is the correct adaptive behaviour — but slower than the pre-tuned fixed gains which happen to be well-calibrated for this specific task.

**In effect, the integral reset makes this a solved problem for fixed PID before RL even gets a chance to demonstrate advantage.** The RL agents are competing against a baseline that has had the hardest part of the control problem engineered away.

#### Validation experiment — Fixed PID without reset:

**Table 5.4 — Fixed PID with brake_integral_reset DISABLED**

| Scenario | Success | Overshoot (m) | Final Error (m) |
|----------|---------|---------------|-----------------|
| Standard | **0%** | 9.92 | 2.995 |
| Heavy & Slippery | **0%** | 10.00 | 10.003 |
| Light & Grippy | **0%** | 9.99 | 2.968 |

Without the integral reset, fixed PID fails completely. The integral accumulates during the 8 m approach (Extended target in the no-reset experiment) to an i_term of approximately 4–5 units. At the braking point, the PID needs `Kd × vel > i_term`, i.e. `Kd > i_term / vel`. At vel = 0.5 m/s and i_term = 4: `Kd > 8.0`. Fixed Kd = 0.5 provides no meaningful braking authority. The vehicle massively overshoots and never recovers.

**This confirms that the integral reset is a critical hidden aid** that equalises agent performance in the main evaluation. The comparison between RL and fixed PID in the standard evaluation (Section 5.2) does not reflect a fair test of adaptive versus non-adaptive control.

#### How to frame this honestly:
Do not hide this finding. Present it as a research insight:

> "An implicit assumption embedded in the environment design was identified during analysis. The `brake_integral_reset` mechanism effectively removes integral windup at the moment when braking authority is most critical. While this is a common engineering practice in real PID implementations, its presence in the simulation equalises all agents and obscures the adaptive advantage that RL should theoretically provide. Disabling this mechanism reveals that fixed PID completely fails, while RL — when retrained without the aid — would face a fundamentally harder control problem. This finding highlights the importance of carefully auditing simulation aids when benchmarking adaptive control methods."

---

---

## Chapter 6 — Discussion and Limitations (~2,000 words)

### 6.1 Main Findings Interpreted

1. **RL successfully learns adaptive PID gain scheduling.** Both agents achieve 100% success across a 4:1 mass range, confirming the feasibility of the approach.

2. **Physics context improves adaptation efficiency.** The 14–15% settling time advantage of Stage 5a over Stage 5b demonstrates that informing the policy of its operating conditions has measurable benefit, even when the policy could in principle infer this from observation history.

3. **The integral reset confound limits the strength of the RL-vs-PID comparison.** The most important and honest finding: the environment aided all agents equally in a way that happened to most benefit the fixed PID. A fairer comparison requires either (a) removing the aid and retraining RL, or (b) using a PID baseline that was not pre-tuned for the exact operating conditions — acknowledging that in practice, pre-tuned gains may not be available.

### 6.2 Limitations

**L1 — Single training seed.** All models trained on seed 7 only. Performance variance across random initialisations is unknown. A 5-seed run (seeds 7, 21, 42, 84, 123) would quantify this.

**L2 — Single-axis evaluation.** Only settling time and overshoot are compared. No comparison of gain trajectories, adaptation speed to new dynamics mid-deployment, or recovery time after disturbance.

**L3 — Disturbance timing mismatch.** The mid-episode disturbance fires at steps 120–220 (1.2–2.2 s). Both RL agents settle within 1.3–1.6 s. In approximately 50–80% of episodes, the disturbance fires after the vehicle has already reached its hold position and has no opportunity to disrupt it. The dynamic evaluation therefore does not stress-test robustness to mid-settling disturbances.

**L4 — Friction irrelevance.** The MuJoCo simulation uses a rolling contact model where kinetic friction does not affect translational dynamics when the wheel rolls without slipping. The "Heavy and Slippery" and "Light and Grippy" scenarios differ only in mass, not in the way the vehicle responds to control inputs. The friction dimension of the domain randomisation provides no useful training signal.

**L5 — Integral reset confound.** Described fully in Section 5.4.

**L6 — No-reset retraining failed to converge.** Attempts to train RL agents in an environment without the integral reset were made but did not converge within 1.5–3 M training steps with PPO. The hypothesised required strategy — suppressing Ki during the approach to prevent windup buildup, then raising Kd near the target — has a temporal credit assignment gap of 500–1000 steps. On-policy PPO with an MLP policy has limited ability to assign credit across such long horizons. A recurrent policy (LSTM/GRU) could explicitly maintain integral state in its hidden representation and would be better suited to this task.

### 6.3 Practical Implications

The blind agent (Stage 5b) is the more practically relevant result: in real deployment, mass and friction scales are typically not directly measurable. The fact that Stage 5b succeeds — slightly slower, but reliably — suggests that the frame-stacking approach to implicit dynamics inference is viable.

The 14–15% settling time penalty for blindness is likely acceptable in many applications. Whether it is acceptable depends entirely on the use case.

---

---

## Chapter 7 — Conclusion and Future Work (~1,000 words)

### Conclusion paragraph structure

1. Restate the research question.
2. Answer it directly: yes, RL can learn adaptive PID gain scheduling; context-aware is faster than blind by ~14–15%; both succeed across a range of physics conditions.
3. State the key scientific finding: the integral reset confound was identified, validating that the environment design affects the apparent advantage of different methods.
4. One sentence on what this work demonstrates methodologically: careful auditing of simulation aids is necessary when benchmarking adaptive control.

### Future Work

**FW1 — Recurrent policy for no-reset control.** An LSTM or GRU policy can maintain a hidden state that tracks cumulative integral error over the full episode. This would make the Ki-suppression strategy learnable without relying on the `brake_integral_reset` aid.

**FW2 — Multi-seed evaluation.** Run seeds [7, 21, 42, 84, 123] for both Stage 5a and 5b to quantify variance and confirm that seed 7 results are representative.

**FW3 — Earlier disturbance window.** Move the disturbance firing step to [30, 80] — during the active approach — to genuinely test robustness to mid-approach physics changes.

**FW4 — Viscous damping for friction relevance.** Add velocity-dependent friction (viscous damping) to the simulation so friction changes have a measurable effect on dynamics, making the friction randomisation dimension meaningful.

**FW5 — Real vehicle transfer.** The sim-to-real gap would need to be addressed before deploying on hardware, but this is outside the scope of this thesis.

**FW6 — Comparative baselines.** Compare against MRAC (which failed in this work) more rigorously, and against other RL algorithms (SAC, TD3) to assess whether the on-policy PPO approach is the right tool for this task.

---

---

## Figures and Tables Reference

All figures live in `chapter05/figures/`. Raw CSV data in `chapter05/data/`.

| Figure | File | Use in |
|--------|------|--------|
| Stage 5a learning curve | `chapter05/figures/training/stage5a_learning_curve.png` | Ch 5.1 |
| Stage 5b learning curve | `chapter05/figures/training/stage5b_learning_curve.png` | Ch 5.1 |
| Stage 5a trajectory plots (static) | `chapter05/figures/trajectories/stage5a_static_trajectories.png` | Ch 5.2 |
| Stage 5b trajectory plots (static) | `chapter05/figures/trajectories/stage5b_static_trajectories.png` | Ch 5.2 |
| Stage 5a eval summary | `chapter05/figures/eval/stage5a_eval_summary.png` | Ch 5.2 |
| Stage 5b eval summary | `chapter05/figures/eval/stage5b_eval_summary.png` | Ch 5.2 |
| Stage 5a dynamic trajectory | `chapter05/figures/trajectories/stage5a_dynamic_trajectories.png` | Ch 5.3 |
| Stage 5b dynamic trajectory | `chapter05/figures/trajectories/stage5b_dynamic_trajectories.png` | Ch 5.3 |
| Baseline static trajectories | `chapter05/figures/trajectories/baseline_static_trajectories.png` | Ch 5.2 or 5.4 |
| Baseline no-reset trajectories | `chapter05/figures/trajectories/baseline_no_reset_trajectories.png` | Ch 5.4 — shows massive overshoot |

**Tables to build from the exact numbers above:** Tables 5.1, 5.2, 5.3, 5.4 are ready to copy directly into the thesis.

---

---

## All Exact Numbers — Quick Reference

### Static evaluation settling times (seconds)

|  | Standard | Heavy+Slippery | Light+Grippy |
|--|----------|----------------|-------------|
| Fixed PID | **1.116** | **1.136** | **1.254** |
| Stage 5a | 1.254 | 1.278 | 1.390 |
| Stage 5b | 1.446 | 1.476 | 1.568 |

### Static evaluation IAE

|  | Standard | Heavy+Slippery | Light+Grippy |
|--|----------|----------------|-------------|
| Fixed PID | 2.653 | 2.789 | 2.939 |
| Stage 5a | 2.682 | 2.816 | 2.967 |
| Stage 5b | 2.785 | 2.917 | 3.061 |

### Stage 5a vs Stage 5b — settling time penalty

| Scenario | 5a (s) | 5b (s) | 5b slower by |
|----------|--------|--------|-------------|
| Standard | 1.254 | 1.446 | +15.3% |
| Heavy & Slippery | 1.278 | 1.476 | +15.5% |
| Light & Grippy | 1.390 | 1.568 | +12.8% |
| **Average** | | | **+14.5%** |

### Fixed PID no-reset (0% success across all scenarios)

| Scenario | Success | Overshoot | Final Error |
|----------|---------|-----------|-------------|
| Standard | 0% | 9.92 m | 2.995 m |
| Heavy & Slippery | 0% | 10.00 m | 10.003 m |
| Light & Grippy | 0% | 9.99 m | 2.968 m |

### Environment parameters (exact)
- dt = 0.02 s (MuJoCo timestep)
- Target: 5.0 m (static eval), 4–10 m random (training)
- Hold: 25 steps within ±0.05 m
- Max eval steps: 5,000 (= 100 s)
- Training episode steps: 1,200
- Gain bases: Kp=1.8, Ki=0.7, Kd=0.5
- Gain deltas: ΔKp=1.0, ΔKi=0.6, ΔKd=2.0
- Gain ranges: Kp∈[0.8,2.8], Ki∈[0.1,1.3], Kd∈[0.0,2.5]
- Terminal hold bonus: 50.0
- Decel bonus coefficient: 10.0
- Per-step overshoot penalty: 2.0 × overshoot

### PPO hyperparameters (exact)
- lr = 3×10⁻⁴ (linear decay)
- γ = 0.99, λ = 0.95, ε = 0.2
- ent_coef = 0.0, vf_coef = 0.5
- minibatch = 64, update_epochs = 10
- num_envs = 4, rollout_steps = 2048
- total_timesteps = 1,000,000
- stack_size = 10

---

---

## Framing Guidance — What to Say vs What Not to Say

| Do say | Don't say |
|--------|-----------|
| "Both RL agents achieve 100% success under unknown dynamics" | "RL outperforms fixed PID" (it doesn't on speed) |
| "Context-aware RL settles 14–15% faster than blind RL" | "RL is always better with more information" (too broad) |
| "The integral reset mechanism was identified as a confound" | "Our experiment was flawed" (it's a finding, not a failure) |
| "Fixed PID fails completely without the integral reset aid" | "Fixed PID is a bad baseline" |
| "The disturbance timing limits the strength of the robustness evaluation" | (don't hide this) |
| "No-reset retraining did not converge — a recurrent architecture is likely needed" | "We couldn't get it to work" |
| "Friction does not meaningfully affect dynamics in this rolling contact model" | (don't hide this) |
