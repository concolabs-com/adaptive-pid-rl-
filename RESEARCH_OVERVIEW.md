# Meta-Reinforcement Learning for Adaptive PID Gain Scheduling

## Problem Statement

Classical PID controllers require manual tuning for each operating condition. When a physical system changes — vehicle mass increases, road friction drops — fixed PID gains that worked before now produce overshoot, instability, or sluggish response. Re-tuning by hand is not feasible in real-time.

**Core question:** Can a neural network learn a policy that observes the current system state and dynamically adjusts PID gains (Kp, Ki, Kd) in real-time to maintain good position-tracking performance across varying physical conditions?

**System:** A two-wheeled simulated car (MuJoCo physics) must drive to a target position. Mass ranges from 5–20 kg. Friction ranges from 0.1–2.0. Mid-episode disturbances (sudden friction/mass shifts) test robustness further.

---

## Approach

Two-stage research strategy:

**Stage 1 — Classical Baseline:** Identify the system using a step-response test, apply SIMC (Simplified Internal Model Control) rules to tune fixed PI controllers. Run these across three physical scenarios to establish a performance floor.

**Stage 2 — Meta-RL Gain Scheduler:** Train a neural network with Proximal Policy Optimization (PPO) or Soft Actor-Critic (SAC) to output PID gain multipliers each timestep. The agent observes position, velocity, tracking error, current gains, and physical context (mass scale, friction scale). Domain randomization during training forces the policy to generalize rather than memorize one tuning.

**Evaluation escalation:** After training, evaluate zero-shot generalization to unseen target distances, robustness to mid-episode disturbances, and head-to-head comparison of all three methods (Fixed PID, Clean-trained Meta-RL, Disturbance-trained Meta-RL).

---

## Tools & Frameworks

| Tool | Role |
|------|------|
| **MuJoCo** | Physics simulation engine; `car_model.xml` defines the two-wheeled vehicle |
| **Gymnasium** | RL environment API wrapping the MuJoCo simulation |
| **PyTorch** | Neural network implementation (Actor-Critic, GRU) |
| **Stable-Baselines3** | SAC implementation reference; callback utilities |
| **NumPy** | Numerical operations, gain clipping, reward computation |
| **Matplotlib / Seaborn** | Training curves, trajectory plots, bar charts |
| **Pandas** | CSV aggregation, summary statistics across seeds/scenarios |
| **TQDM** | Progress bars during training loops |

---

## System Architecture

### Environment — `AdaptiveSuspensionEnv` (`envs/adaptive_suspension.py`)

**Observation space (8-dim):**
| Index | Signal | Meaning |
|-------|--------|---------|
| 0 | position (x) | Current vehicle position |
| 1 | velocity (dx) | Current vehicle velocity |
| 2 | tracking error | `target_position - current_position` |
| 3–5 | previous action | Last Kp, Ki, Kd outputs from agent |
| 6 | mass_scale | Current mass context (1.0 = nominal) |
| 7 | friction_scale | Current friction context (1.0 = nominal) |

**Action space (3-dim continuous, [-1, 1]):**
Agent outputs normalized multipliers for Kp, Ki, Kd. These scale base gains inside the PID controller each timestep.

**Reward shaping:**
- Position error penalty (primary objective)
- Velocity damping penalty (penalizes high speed near target)
- Hold bonus (reward for sustained low-error periods)
- Safety penalties: overshoot, excessive speed

### PID Controller — `utils/pid.py`

`PIDController` class accepts dynamic gain updates each step. The Meta-RL agent replaces human re-tuning by calling `update_gains(kp, ki, kd)` at every timestep.

### Domain Randomization — `agents/domain_randomization.py`

Wraps the base environment. Each episode samples:
- Mass: uniform [5.0, 20.0] kg
- Friction: uniform [0.1, 2.0]

Optional mid-episode disturbances apply sudden mass/friction shifts at random timesteps. Position-based friction patches simulate multi-surface roads.

### Neural Network — `agents/model.py`, `agents/meta_ppo.py`

Two policy variants:
- **Non-recurrent:** Linear layers (64 units) → separate Actor/Critic heads
- **Recurrent (GRU):** GRU (128 hidden units) provides temporal memory for context-dependent adaptation

Both use orthogonal initialization, log-std parameters for Gaussian action distribution, and output 3 continuous values.

---

## Experiments by Stage

### Stage 0 — Environment Validation

**Goal:** Confirm physics engine is stable before any control experiment. Catch crashes, NaN values, state leakage between resets.

---

#### `check_mujoco.py`
**What it does:** Loads MuJoCo InvertedPendulum, modifies body mass and friction geom, resets, prints before/after values.

**Question:** Is MuJoCo installed and can physics parameters be modified at runtime?

**Output:** Pass/fail printout. No metrics, no plots.

**Contribution:** Prerequisite check. Nothing else can run if this fails.

---

#### `mujoco_stage0_sanity.py`
**What it does:** Runs random control rollouts (no policy, just noise), detects NaN/crashes, verifies episode reset clears all state, applies mass/friction scenario scaling.

**Metrics tracked:** `crash` (bool), `nan_detected` (bool), `final_x`, `final_speed`, `qpos_residual`, `qvel_residual` (state leakage measure).

**Question:** Does the environment produce stable trajectories? Does reset fully clear state?

**Contribution:** Validates the simulation substrate before any PID or RL experiment runs on it.

---

### Stage 1 — Classical Control Baselines

**Goal:** Establish what fixed, hand-tuned (but principled) PID controllers can achieve. This forms the performance floor that Meta-RL must beat.

---

#### Stage 1a — System Identification & Autotune: `stage1_pid_autotune.py`

**What it does:**
1. Applies a fixed step input to the vehicle, records velocity response over time.
2. Fits a First-Order Plus Dead-Time (FOPDT) model: gain `k`, time constant `tau`, dead-time `theta`.
3. Applies SIMC tuning formulas to compute PI gains:
   - Standard: `tau_c = max(theta, 0.05)` → tighter response
   - Robust: `tau_c = 3 * theta` → more conservative

**Question:** What are the principled PI gains for this system, derived from actual dynamics rather than manual guessing?

**Output:** `k`, `tau`, `theta`, `Kp_std`, `Ki_std`, `Kp_rob`, `Ki_rob` — fed into Stage 1b benchmarks.

**Contribution:** Removes arbitrary gain selection. Provides a defensible baseline grounded in control theory.

---

#### Stage 1b — Multi-Scenario PID Benchmark: `stage1_fixed_pid_benchmark.py`

**What it does:**
- Runs SIMC-tuned Standard and Robust PI controllers across 3 scenarios, 30+ episodes each:
  - **Standard:** mass_scale=1.0, friction_scale=1.0
  - **Heavy & Slippery:** mass_scale=1.5, friction_scale=0.6
  - **Light & Grippy:** mass_scale=0.7, friction_scale=1.4
- Speed-based control: cruise at `target_speed`, decelerate linearly in final 0.6 m (DECEL_ZONE)
- Output limits: (-0.35, 1.0) asymmetric

**Metrics:** `settling_time_s`, `overshoot_m`, `IAE` (Integral Absolute Error), `final_abs_error_m`, `success_rate`, `settled_rate`

**Output:** Per-episode CSV, aggregated summary CSV, bar chart PNGs.

**Contribution:** The primary Stage 1 result. Shows how fixed PID degrades under domain shift — motivates the Meta-RL approach.

---

#### Stage 1b (variant) — Navigation Benchmark: `benchmark.py`

**What it does:**
- 2D position-reach task (x,y target) using differential-drive mixing
- Three controllers: Adaptive Scheduled (lightweight gain formula), Fixed PID Standard, Fixed PID Robust
- Adaptive formula: `kp_dist = 0.55 / (0.8 + 0.4 * mass_factor)` — a hand-coded heuristic

**Metrics:** `settling_time_s`, `overshoot`, `IAE`, `final_error`

**Contribution:** Tests whether even a simple hand-coded adaptive rule beats fixed gains in 2D navigation. Intermediate complexity between pure PID and learned Meta-RL.

---

#### Stage 1b (original) — Straight-line Benchmark: `benchmark_old.py`

**What it does:**
- Single-direction drive with two PID variants: Conservative (Kp=0.9, Ki=0.18), Aggressive (Kp=1.8, Ki=0.7)
- Records time-to-target, first arrival time, overshoot

**Metrics:** `time_to_target_s`, `steps_to_target`, `final_distance_m`, `avg_forward_speed_mps`

**Contribution:** Earlier iteration establishing the performance contrast between cautious and aggressive tuning. Simpler than the multi-scenario Stage 1 benchmark.

---

#### Stage 1c — Architectural Diagnostic: `diagnostic_pid_comparison.py`

**What it does:**
- Side-by-side comparison of two different PID formulations:
  - Stage 1 style: velocity-based error (`target_speed - measured_speed`), asymmetric limits (-0.35, 1.0)
  - Stage 2 style: position-based error (`target_pos - measured_pos`), symmetric limits (-1.0, 1.0), gain scheduling enabled
- Also runs neutral policy baseline for Stage 2 setup (no learning, just base gains)

**Metrics:** Position trace, velocity trace, velocity statistics (min/max/mean/std)

**Contribution:** Documents why Stage 1 and Stage 2 are not directly comparable — different control architectures, different error signals. Prevents false apples-to-oranges conclusions.

---

### Stage 2a — Basic Meta-RL Prototype

**Goal:** Establish that PPO can train at all on the gain-scheduling task before adding complexity.

---

#### `train.py`

**What it does:** Minimal PPO training loop — 4 parallel environments, 100k timesteps, frame-stacked observations (10 frames), no recurrence, no curriculum.

**Hyperparameters:**
- Learning rate: 3e-4 (linear annealing)
- Minibatch: 64, Update epochs: 10
- GAE lambda: 0.95, Gamma: 0.99, Clip: 0.2
- Rollout: 2048 steps across 4 envs

**Output:** Saved agent checkpoint.

**Contribution:** Proof-of-concept. Validates the environment-agent interface before investing in the full pipeline.

---

#### `evaluate.py`

**What it does:** Loads a trained Stage 2a checkpoint, tests on 3 fixed scenarios, plots learned gain trajectories over time.

**Metrics:** Position trajectories, gain adaptation traces (Kp, Ki, Kd over episode steps).

**Contribution:** Qualitative check — does the agent actually modulate gains, or does it collapse to a constant output?

---

### Stage 2b — Full Meta-RL Training Pipeline

**Goal:** Train a robust Meta-RL agent with curriculum learning, domain randomization, and mid-episode disturbances. This is the main experiment of Stage 2.

---

#### `stage2_meta_rl_reproduction.py` (~1400 lines)

**What it does:**

*Training:*
- PPO with optional GRU recurrence (10 preset configs: `thesis_v1`, `v2_curriculum`, `v3_safety`, etc.)
- 4–8 parallel environments, 80k–100k total timesteps
- Domain randomization every episode: mass [5–20 kg], friction [0.1–2.0]
- Frame stacking: 10 frames for temporal context (non-recurrent variant)
- **Curriculum learning:** Target distances grow from 1 m → 10 m across training phases. Disturbance magnitude increases gradually.
- **Mid-episode disturbances:** Stochastic mass/friction shifts at random timesteps during an episode
- **Position-based friction patches:** Friction anomalies at specific x positions simulate road transitions
- **Safety mechanisms:** Speed governor, brake margin enforcement, action slew-rate limiting, overshoot penalty

*Evaluation (multi-seed):*
- 5+ seeds per config
- 3 evaluation scenarios: Standard (10 kg, 1.0 friction), Heavy & Slippery (20 kg, 0.2 friction), Light & Grippy (5 kg, 2.0 friction)
- Deterministic policy rollout, 10 episodes per scenario per seed

**Metrics:** `settling_time_s`, `overshoot`, `IAE`, `success_rate`, `mean_reward`; training: `policy_loss`, `value_loss`, `KL_divergence`, `entropy`

**Output:** Training curves, per-seed eval CSVs, aggregated summary CSVs, learning curve PNGs, config JSON per seed.

**Contribution:** Core Stage 2 result. Demonstrates whether Meta-RL with gain scheduling outperforms fixed PID baselines, and whether curriculum + disturbances improve robustness over clean training.

---

### Stage 2c — Alternative RL Algorithm Baseline

**Goal:** Determine whether the result is PPO-specific or holds for off-policy RL.

---

#### `stage2_sac_reproduction.py` (~630 lines)

**What it does:**
- Same environment setup as Stage 2b (domain randomization, same scenarios)
- Soft Actor-Critic (SAC) instead of PPO: off-policy, entropy-regularized
- Single environment (num_envs=1) vs PPO's 4 parallel
- SAC hyperparameters: buffer_size, learning_starts, batch_size, tau (soft update), gradient_steps
- Same evaluation protocol and CSV outputs

**Question:** Does off-policy SAC match or exceed on-policy PPO for this gain-scheduling task?

**Contribution:** Algorithm ablation. If SAC and PPO reach similar performance, the result is robust to algorithm choice. If they diverge, it informs future work on algorithm selection for adaptive control.

---

### Stage 3 — Generalization & Robustness Evaluation

**Goal:** Probe the limits of trained agents — do they generalize beyond training conditions?

---

#### `eval_distance_generalization.py`

**What it does:**
- Loads a trained Stage 2 agent
- Tests zero-shot on target distances not seen (or rarely seen) during training: 1.0, 2.0, 3.0, 5.0 m+
- 10 episodes per distance, deterministic rollout

**Metrics:** `success_rate`, `mean_steps`, `mean_reward`, `final_abs_error`, `overshoot` per distance.

**Question:** Does curriculum training over distance ranges produce an agent that extrapolates to novel distances?

**Contribution:** Tests spatial generalization — critical for real-world deployment where exact target distance is unknown at deploy time.

---

#### `eval_mid_episode_friction_probe.py`

**What it does:**
- Custom wrapper `MidEpisodeFrictionPatchWrapper`: triggers a friction change at a random distance mid-episode
- Optional simultaneous tiny mass shift
- Records agent's gain adaptation response (Kp, Ki, Kd traces) before and after disturbance

**Metrics:** Position error, gain schedules, settling time after disturbance onset.

**Question:** Does the Meta-RL agent detect and respond to sudden friction changes by adjusting gains, or does it fail to adapt?

**Contribution:** Tests temporal robustness — whether the agent behaves like a true adaptive controller that reacts to dynamic changes, not just a controller pre-tuned for average conditions.

---

#### `compare_three_methods_random_distance.py`

**What it does:**
- Head-to-head comparison of all three major approaches:
  1. **Stage1_FixedPID_Robust** — SIMC-tuned robust controller from Stage 1
  2. **Stage2_CleanTrained** — Meta-RL trained without mid-episode disturbances
  3. **Stage2_DisturbanceTrained** — Meta-RL trained with mid-episode disturbances
- Random target distances sampled each episode
- Hold-settling criterion: position error AND velocity both below threshold for `hold_steps` consecutive steps

**Metrics:** `hold_settling` (settling time), `success`, `overshoot`

**Question:** Does disturbance training improve over clean training? Do both outperform fixed PID?

**Contribution:** The synthesis experiment. Directly answers the thesis question by comparing all methods on the same fair evaluation protocol.

---

### Visualization & Reporting Scripts

These scripts consume CSV outputs from the experiments above and generate thesis-ready figures. They do not run new experiments.

| Script | Input | Output |
|--------|-------|--------|
| `generate_comparison_plots.py` | Stage 1 + Stage 2 summary CSVs | Master comparison table; bar charts for settling_time, final_error, overshoot, success_rate |
| `generate_three_method_comparison.py` | Three-method CSVs | Three-way bar/line comparison plots |
| `plot_fair_pid_comparison.py` | Stage 1 CSVs | Fair side-by-side PID variant comparison |
| `plot_stage2_diagnostics.py` | Stage 2 training logs | Debug/diagnostic training curves |
| `plot_simulated_run_behaviors.py` | Episode trajectory logs | Position, velocity, gain traces per run |
| `visualize_stage2_disturbances.py` | Disturbance episode logs | Visualize mass/friction shift effects on agent behavior |

---

## Experiment Complexity Summary

| Stage | Script(s) | Complexity | What It Adds |
|-------|-----------|------------|--------------|
| 0 | `check_mujoco.py` | Trivial | MuJoCo installation check |
| 0 | `mujoco_stage0_sanity.py` | Minimal | Simulation stability + reset validation |
| 1a | `stage1_pid_autotune.py` | Low | FOPDT system ID → principled PI gains |
| 1b | `benchmark_old.py` | Low | Conservative vs aggressive gain comparison |
| 1b | `benchmark.py` | Low-Medium | Multi-controller 2D navigation benchmark |
| 1b | `stage1_fixed_pid_benchmark.py` | Medium | Multi-scenario fixed PID baseline (primary Stage 1 result) |
| 1c | `diagnostic_pid_comparison.py` | Medium | Architectural difference analysis between Stage 1/2 |
| 2a | `train.py` + `evaluate.py` | Medium | Minimal PPO proof-of-concept |
| 2b | `stage2_meta_rl_reproduction.py` | High | Full PPO pipeline: curriculum, disturbances, multi-seed |
| 2c | `stage2_sac_reproduction.py` | High | SAC alternative algorithm ablation |
| 3 | `eval_distance_generalization.py` | High | Zero-shot spatial generalization test |
| 3 | `eval_mid_episode_friction_probe.py` | High | Mid-episode disturbance robustness probe |
| 3 | `compare_three_methods_random_distance.py` | High | Final synthesis: all three methods compared |
