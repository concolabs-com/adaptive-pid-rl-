# Chapter 4 — Experimental Setup

## 4.1 Evaluation Protocol

All agents were evaluated under a shared protocol applied uniformly across every scenario. Each evaluation episode used a fixed target position of 5.0 m and a maximum episode length of 5,000 steps. At the RL control rate of 50 Hz (Section 3.2), this corresponds to a maximum episode duration of 100 seconds — far beyond the time required by any agent to settle, ensuring that failure to meet the success criterion reflects genuine inability rather than time truncation.

**Success criterion.** An episode is counted as successful if the vehicle's position remains within ±0.05 m of the target for 25 consecutive control steps. This requires sustained hold behaviour rather than a momentary crossing of the tolerance band.

**Settling time.** Settling time is defined as the elapsed simulation time at the first control step at which the vehicle enters and subsequently maintains the ±0.05 m tolerance band for at least 25 consecutive steps. Formally, if $i^*$ is the smallest step index satisfying $|e(j)| \leq 0.05$ m for all $j \in [i^*, i^* + 25)$, then settling time $t_s = i^* \cdot \Delta t$, where $\Delta t = 0.02$ s is the RL control timestep ($= \text{frame\_skip} \times \text{physics timestep} = 10 \times 0.002$ s).

**Metrics reported.** Five metrics are computed per episode:

| Metric | Definition |
|--------|-----------|
| Success rate (%) | Fraction of episodes where the 25-step hold criterion is met |
| Settling time (s) | $t_s$ as defined above; reported as mean ± range across evaluation seeds |
| Overshoot (m) | $\max(0,\, \max_t x(t) - x_\text{target})$ — maximum position exceedance past the target |
| IAE | $\sum_{k=1}^{N} |e_k| \cdot \Delta t$ — integrated absolute error over the episode |
| Final absolute error (m) | $|e(t_\text{final})|$ — tracking error at the last recorded step; reported primarily for the no-reset validation in Section 5.4 |

**Evaluation seeds.** Ten seeds (70,000–70,009) are used to call `env.reset(seed=·)` at the start of each episode. In static evaluation (Section 4.2), the physics parameters are fixed per scenario and the policy is deterministic, so no stochastic elements are seeded — all ten static seeds produce identical trajectories. Seeds are run for consistency with the dynamic protocol. In dynamic evaluation, the seed initialises the environment's random number generator, which controls both the step at which the mid-episode disturbance fires (drawn uniformly from steps 120–220) and the disturbance magnitudes. The ten dynamic seeds therefore sample ten distinct perturbation scenarios, and results are reported as mean ± range across those samples.

---

## 4.2 Evaluation Scenarios

Evaluation is structured into two protocols: static and dynamic. Static evaluation isolates agent performance under constant physics; dynamic evaluation tests robustness to mid-episode parameter changes and out-of-distribution conditions.

### Static Evaluation

In static evaluation, mass and friction are fixed for the entire episode. Three scenarios span the training distribution:

| Scenario | Mass (kg) | Friction | Notes |
|----------|-----------|----------|-------|
| Standard | 10.0 | 1.0 | Nominal operating point; centre of training distribution |
| Heavy and Slippery | 20.0 | 0.2 | Upper mass boundary, lower friction boundary |
| Light and Grippy | 5.0 | 2.0 | Lower mass boundary, upper friction boundary |

These scenarios probe whether agents have learned to generalise across the full range of training conditions. As noted in Section 3.3, translational dynamics in this rolling-contact simulation are dominated by mass; the friction dimension provides limited independent signal, a limitation discussed in Section 6.2.

### Dynamic Evaluation

Dynamic evaluation applies a mid-episode disturbance to each of the three in-distribution scenarios, and additionally tests two out-of-distribution (OOD) conditions. The disturbance mechanism is described in Section 3.3: at a random step in [120, 220], mass is multiplied by a scale factor drawn from [0.9, 1.3] and friction by a scale factor from [0.5, 1.4].

The five dynamic scenarios are:

| Scenario | Base Mass (kg) | Base Friction | Notes |
|----------|----------------|---------------|-------|
| Standard | 10.0 | 1.0 | In-distribution with disturbance |
| Heavy and Slippery | 20.0 | 0.2 | In-distribution with disturbance |
| Light and Grippy | 5.0 | 2.0 | In-distribution with disturbance |
| OOD Ultra Heavy | 35.0 | 1.0 | 75% above training mass ceiling |
| OOD Ultra Slippery | 20.0 | 0.05 | Below training friction floor |

The OOD conditions were selected to test generalisation at approximately twice the training mass ceiling (35 kg vs 20 kg maximum) and below the training friction floor (0.05 vs 0.1 minimum), without choosing values so extreme that failure would be trivially expected. These conditions were not seen during training by either RL agent.

---

## 4.3 Agents Evaluated

Three agents are evaluated. Full architectural and training details appear in Chapter 3; this section provides a concise comparative reference.

| Agent | Description | Training Steps | Observation Dims |
|-------|-------------|---------------|-----------------|
| Fixed PID | Non-adaptive baseline. Action fixed at [0, 0, 0]; gains held at base values (Kp = 1.8, Ki = 0.7, Kd = 0.5) for the entire episode. No learning. | — | — |
| Stage 5a — Context-Aware Agent | PPO agent. Observes position, velocity, error, current normalised gains, and measured mass/friction scales. Sees its own physics parameters directly. | 1,000,000 | 8 × 10 = 80 |
| Stage 5b — Blind Agent | PPO agent. Observes position, velocity, error, and current normalised gains only. Must infer dynamics from 10-frame trajectory history. No mass or friction in observation. | 1,000,000 | 6 × 10 = 60 |

Both RL agents were trained with curriculum learning (Section 3.5), domain randomisation (Section 3.3), and the `brake_integral_reset` mechanism active (Section 3.7). All evaluations use these same environment settings — the integral reset remains active for all three agents, including the Fixed PID baseline. The effect of this mechanism on the comparisons is analysed in Section 5.4.

The Fixed PID baseline serves two purposes: it establishes a performance ceiling for a correctly pre-tuned non-adaptive controller, and the no-reset ablation (Section 5.4) reveals what the baseline achieves independently of the shared engineering aid.

---

## 4.4 Reproducibility

All trained model weights (`.pth`), evaluation scripts, environment source code, and raw results (`.csv`) are provided in the accompanying `thesis_submission/` directory. The complete configuration for each agent is stored in `stage2_config.json` alongside the model checkpoint. Evaluation can be reproduced by running the scripts detailed below against the provided checkpoints:

| Artefact | Path |
|----------|------|
| Stage 5a model (seed 7) | `benchmark_results/stage5a_context_cliff/seed_7/models/meta_rl_agent.pth` |
| Stage 5b model (seed 7) | `benchmark_results/stage5b_blind_cliff/seed_7/models/meta_rl_agent.pth` |
| Fixed PID eval script | `scripts/stage_baseline_fixed_pid.py` |
| Stage 5a dynamic eval | `scripts/stage5a_eval_dynamic.py` |
| Stage 5b dynamic eval | `scripts/stage5b_eval_dynamic.py` |
| Raw results | `benchmark_results/*/eval_seed_summary.csv` |

All reported RL results use training seed 7. A single training seed was used throughout; variance across random initialisations was not quantified, which is identified as a limitation in Section 6.2.

---

## 4.5 Hardware and Compute

All training and evaluation were conducted on a laptop system running Windows 11 with Python 3.10. The hardware configuration was: Intel Core i7-11800H processor (2.30 GHz, 8 cores), 32 GB DDR4 RAM, and an NVIDIA GeForce RTX 3070 Laptop GPU (8 GB VRAM). MuJoCo physics simulation ran on CPU; PPO gradient updates used the GPU via PyTorch.

Each RL agent required approximately 45 minutes of wall-clock time to complete 1,000,000 training steps across four parallel environments. Evaluation of a single agent across all scenarios and seeds (static and dynamic) completed in under five minutes. The Fixed PID baseline required no training; full evaluation across all scenarios completed in under two minutes.
