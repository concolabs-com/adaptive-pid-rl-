# Summary of Findings
## Meta-RL for Adaptive PID Gain Scheduling Under Unknown Vehicle Dynamics

---

## Objective

Design and evaluate a reinforcement learning agent that learns to schedule PID gains in real time for a simulated 2-wheeled vehicle, without knowing the vehicle's mass or friction in advance. The central question: does learning-based adaptation add value over a fixed, hand-tuned PID controller when physics parameters are unknown?

---

## System Built

A PPO-based Meta-RL agent trained on a MuJoCo vehicle environment with:

- **Adaptive PID loop**: the RL agent outputs gain adjustments each step; gains are clipped to physically meaningful ranges (Kp ∈ [0.8, 2.8], Ki ∈ [0.1, 1.3], Kd ∈ [0.0, 2.5])
- **Temporal memory**: 10-frame observation stack, giving the policy a rolling window of recent trajectory history
- **Domain randomisation**: mass drawn from [5, 20] kg and friction from [0.1, 2.0] at each episode reset, forcing generalisation
- **Curriculum**: target distance increased from 3 m → 5 m → 7 m → 10 m over 1.5 M training steps
- **Mid-episode disturbances**: mass and friction shifted mid-episode (step 120–220) and a low-friction floor patch applied at x = 1.5–2.4 m, testing robustness to sudden physics changes

Two agent variants were trained and evaluated:

| Agent | Observation | Description |
|-------|-------------|-------------|
| **Stage 5a — Context RL** | pos, vel, error, kp, ki, kd, **mass\_scale, friction\_scale** | Can see its own physics parameters |
| **Stage 5b — Blind RL** | pos, vel, error, kp, ki, kd | Must infer dynamics from trajectory alone |

Classical baseline: fixed PID at Kp = 1.8, Ki = 0.7, Kd = 0.5, action = [0, 0, 0] (no adaptation).

---

## Results

### Static Evaluation — Fixed Physics Per Episode (10 seeds, 70000–70009)

| Agent | Scenario | Success | Settling (s) | Overshoot | IAE |
|-------|----------|---------|-------------|-----------|-----|
| Fixed PID | Standard | 100% | 1.116 | 0 m | 2.65 |
| Fixed PID | Heavy & Slippery | 100% | 1.136 | 0 m | 2.79 |
| Fixed PID | Light & Grippy | 100% | 1.254 | 0 m | 2.94 |
| **Stage 5a** | Standard | **100%** | 1.254 | 0 m | 2.68 |
| **Stage 5a** | Heavy & Slippery | **100%** | 1.278 | 0 m | 2.82 |
| **Stage 5a** | Light & Grippy | **100%** | 1.390 | 0 m | 2.97 |
| **Stage 5b** | Standard | **100%** | 1.446 | 0 m | 2.78 |
| **Stage 5b** | Heavy & Slippery | **100%** | 1.476 | 0 m | 2.92 |
| **Stage 5b** | Light & Grippy | **100%** | 1.568 | 0 m | 3.06 |

All three agents achieve 100% success with zero overshoot across all mass/friction scenarios. Stage 5b (blind) settles approximately **14–15% slower** than Stage 5a (context-aware). Fixed PID is fastest, as discussed in Finding 2 below.

### Dynamic Evaluation — Mid-Episode Disturbances + OOD Conditions

All five scenarios evaluated, including two out-of-distribution conditions (mass = 35 kg and friction = 0.05, neither seen during training):

| Agent | Scenario | Success | Settling (s) | Overshoot |
|-------|----------|---------|-------------|-----------|
| Fixed PID | Standard | 100% | 1.116–1.142 | 0 m |
| Fixed PID | Heavy & Slippery | 100% | 1.136 | 0 m |
| **Stage 5a** | Standard | **100%** | 1.466–1.476 | 0 m |
| **Stage 5a** | Heavy & Slippery | **100%** | 1.466–1.488 | 0 m |
| **Stage 5a** | OOD Ultra Heavy (35 kg) | **100%** | 1.452–1.506 | 0 m |
| **Stage 5a** | OOD Ultra Slippery | **100%** | 1.466–1.480 | 0 m |
| **Stage 5b** | Standard | **100%** | 1.446–1.448 | 0 m |
| **Stage 5b** | Heavy & Slippery | **100%** | 1.474–1.478 | 0 m |
| **Stage 5b** | OOD Ultra Heavy (35 kg) | **100%** | 1.524–1.558 | 0 m |
| **Stage 5b** | OOD Ultra Slippery | **100%** | 1.474–1.478 | 0 m |

Both RL agents maintain 100% success under mid-episode disturbances and on OOD conditions well outside the training mass range. The disturbances applied during evaluation (step 120–220) had minimal impact on settling metrics, which is discussed as a limitation below.

---

## Key Findings

### Finding 1 — Physics context improves adaptation speed

Stage 5a, which observes its own mass and friction scale, settles consistently **14–15% faster** than Stage 5b, which must infer dynamics purely from trajectory history. Both succeed; the difference is efficiency. This confirms that physics context in the observation is beneficial but not required for success.

### Finding 2 — A hidden engineering aid equalised all agents

A `brake_integral_reset` mechanism in the environment zeros the PID integrator when the vehicle enters the braking zone (|error| < 2.0 m). This was originally included to prevent integral windup and was active for all agents during both training and evaluation, including the fixed PID baseline.

The consequence: fixed PID at pre-tuned base gains already behaves near-optimally once integral windup is removed at the critical moment. The RL agents learn conservative, safe gain schedules (higher Kd, lower Ki) at the cost of settling speed, which puts them at a disadvantage on pure speed metrics.

To validate this, the fixed PID was evaluated with the `brake_integral_reset` **disabled**:

| Agent | Scenario | Success | Overshoot | Final Error |
|-------|----------|---------|-----------|-------------|
| Fixed PID (no reset) | Standard | **0%** | 9.9 m | 3.0 m |
| Fixed PID (no reset) | Heavy & Slippery | **0%** | 10.0 m | 10.0 m |
| Fixed PID (no reset) | Light & Grippy | **0%** | 10.0 m | 3.0 m |

Without the aid, fixed PID fails completely due to integral windup — the accumulated integral term during the 8 m approach exceeds what Kd = 0.5 can counteract at any reasonable braking velocity. This confirms that the aid, not the fixed gains themselves, was responsible for the strong baseline performance.

This is identified as a **key experimental confound**: the integral reset mechanism made the environment inadvertently easy for fixed PID, and the comparison between fixed PID and RL in the standard evaluation does not reflect a fair test of adaptive versus non-adaptive control.

### Finding 3 — Disturbance timing limits stress test validity

Mid-episode disturbances were scheduled at steps 120–220, corresponding to 1.2–2.2 s after episode start. However, both RL agents typically settle within 1.25–1.57 s. This means most disturbances fired after the vehicle had already reached and stabilised at the target, having no measurable effect on success metrics. The dynamic evaluation results therefore reflect robustness of the initial approach phase only, not the response to disturbances during final settling.

---

## Honest Assessment of Limitations

1. **Single training seed**: All results reported are from seed 7. Multi-seed variance is not quantified, so it is not known how sensitive the learned policy is to random initialisation.

2. **Integral reset confound**: The core comparison (RL vs fixed PID) is compromised by the brake_integral_reset aid. A fair demonstration of RL advantage would require disabling this aid and re-training — which was attempted but did not converge within the available compute budget (see below).

3. **No-reset training failed to converge**: Training was run for 1.5 M and 3 M steps with and without overshoot penalties. The agent did not learn a stable Ki-suppression strategy. The likely root cause is a long temporal credit assignment gap: suppressing Ki at step ~50 of an approach to avoid windup at step ~600 is a difficult horizon for on-policy PPO with an MLP policy. This is left as future work.

4. **Friction irrelevance**: Kinetic friction does not affect dynamics when wheels roll without slipping. The "Slippery" and "Grippy" scenarios therefore only differ meaningfully in mass, not friction. This limits the diversity of the evaluation.

5. **Disturbance timing mismatch**: As noted in Finding 3, the disturbance window does not overlap with the active approach phase for the faster-settling agents, reducing the practical value of the dynamic evaluation.

---

## Suggested Future Work

- **Recurrent policy (LSTM/GRU)**: Would provide explicit memory of the integral history across the full approach, making Ki-suppression learnable without engineering aids.
- **Multi-seed evaluation**: Repeat experiments with seeds [7, 21, 42, 84, 123] to quantify variance and confirm results are not seed-specific.
- **Earlier disturbance window**: Shift mid-episode disturbances to steps 30–80 (during approach) to genuinely stress-test robustness.
- **Explicit friction decoupling**: Use scenarios that vary mass and friction independently and confirm that mass is the dominant variable, or introduce viscous damping so friction has measurable effect.

---

## Summary Table

| Metric | Fixed PID | Stage 5a (Context) | Stage 5b (Blind) |
|--------|-----------|-------------------|-----------------|
| Success rate (static) | 100% | 100% | 100% |
| Mean settling, Standard (s) | **1.116** | 1.254 | 1.446 |
| Mean settling, Heavy+Slippery (s) | **1.136** | 1.278 | 1.476 |
| Overshoot | 0 m | 0 m | 0 m |
| OOD generalisation (35 kg) | not tested | **100%** | **100%** |
| Success without integral reset | **0%** | not retrained | not retrained |
| Adapts to unknown mass/friction | No | Yes | Yes |
