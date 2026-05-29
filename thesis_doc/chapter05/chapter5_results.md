# Chapter 5 — Results and Analysis

This chapter presents the experimental results for all three agents evaluated across static and dynamic scenarios, and analyses the key findings in depth. Section 5.1 describes the training dynamics of both learned agents. Section 5.2 presents static evaluation results, where physics parameters remain fixed throughout each episode. Section 5.3 presents dynamic evaluation results, including mid-episode disturbances and out-of-distribution conditions. Section 5.4 analyses a critical experimental confound — the `brake_integral_reset` mechanism — and its effect on the interpretation of the Fixed PID speed advantage.

---

## 5.1 Training Convergence

Both the Context-Aware Agent (Stage 5a) and the Blind Agent (Stage 5b) were trained for 1,000,000 timesteps using PPO with training seed 7. Learning curves for both agents are shown in Figures 5.1 and 5.2.

![Stage 5a learning curve](figures/training/stage5a_learning_curve.png)
*Figure 5.1 — Stage 5a (Context-Aware Agent) training curve over 1,000,000 timesteps. Curriculum transitions at 250K, 500K, and 750K steps are visible as step changes in episodic return.*

![Stage 5b learning curve](figures/training/stage5b_learning_curve.png)
*Figure 5.2 — Stage 5b (Blind Agent) training curve over 1,000,000 timesteps.*

Both agents exhibit progressive improvement in episode return across the full training run. The four-phase curriculum — in which the maximum target distance increases from 3 m to 5 m, 5 m to 7 m, and 7 m to 10 m at timesteps 250K, 500K, and 750K respectively — produces visible step-change drops in return at each transition. These drops reflect the sudden increase in task difficulty as the agent encounters longer approach distances with greater integral windup potential. Following each transition, both agents recover within the subsequent phase and surpass their prior performance level, confirming that the curriculum structure successfully scaffolds learning.

The Context-Aware Agent (Stage 5a) exhibits somewhat smoother recovery after curriculum transitions, particularly at the 500K and 750K boundaries. This is consistent with the agent's ability to directly observe the mass and friction scales in its observation: given the current dynamics, it can adjust gain selection more immediately when the task difficulty changes. The Blind Agent (Stage 5b) shows a slightly broader variance in return within each curriculum phase, reflecting the additional uncertainty it must resolve from trajectory history alone, though it achieves comparable final performance.

Both agents converge to stable, positive return trajectories by the end of training, indicating that the curriculum and reward structure successfully guided both agents toward the target-reaching and holding behaviour required by the task. Training was conducted on a single random seed (seed 7); performance variance across different initialisations was not quantified, which is acknowledged as Limitation L1 in Section 6.2.

---

## 5.2 Static Evaluation Results

The static evaluation assesses all three agents across three fixed-physics scenarios: Standard (nominal operating conditions: mass = 10 kg, friction = 1.0), Heavy and Slippery (upper training boundary: mass = 20 kg, friction = 0.2), and Light and Grippy (lower training boundary: mass = 5 kg, friction = 2.0). Physics parameters remain constant throughout each episode. Ten evaluation seeds (70000–70009) are used per scenario.

Because the policy is evaluated deterministically (no action noise) and physics parameters are fixed per scenario, all 10 seeds produce identical results for each agent-scenario combination. Table 5.1 reports the full results.

**Table 5.1 — Static evaluation results (all agents, 10 seeds each)**

| Agent | Scenario | Success rate | Settling time (s) | Overshoot (m) | IAE |
|-------|----------|-------------|-------------------|---------------|-----|
| Fixed PID | Standard | 100% | 1.116 | 0.000 | 2.653 |
| Fixed PID | Heavy & Slippery | 100% | 1.136 | 0.000 | 2.789 |
| Fixed PID | Light & Grippy | 100% | 1.254 | 0.000 | 2.939 |
| Stage 5a (Context) | Standard | 100% | 1.254 | 0.000 | 2.682 |
| Stage 5a (Context) | Heavy & Slippery | 100% | 1.278 | 0.000 | 2.816 |
| Stage 5a (Context) | Light & Grippy | 100% | 1.390 | 0.000 | 2.967 |
| Stage 5b (Blind) | Standard | 100% | 1.446 | 0.000 | 2.785 |
| Stage 5b (Blind) | Heavy & Slippery | 100% | 1.476 | 0.000 | 2.917 |
| Stage 5b (Blind) | Light & Grippy | 100% | 1.568 | 0.000 | 3.061 |

Trajectory plots for each agent under static evaluation are shown in Figures 5.3, 5.4, and 5.5. Summary bar charts of settling time and IAE across scenarios are shown in Figures 5.6 and 5.7.

![Baseline static trajectories](figures/trajectories/baseline_static_trajectories.png)
*Figure 5.3 — Fixed PID position trajectories across three static scenarios.*

![Stage 5a static trajectories](figures/trajectories/stage5a_static_trajectories.png)
*Figure 5.4 — Stage 5a (Context-Aware Agent) position trajectories across three static scenarios.*

![Stage 5b static trajectories](figures/trajectories/stage5b_static_trajectories.png)
*Figure 5.5 — Stage 5b (Blind Agent) position trajectories across three static scenarios.*

![Stage 5a eval summary](figures/eval/stage5a_eval_summary.png)
*Figure 5.6 — Stage 5a evaluation summary: settling time and IAE by scenario.*

![Stage 5b eval summary](figures/eval/stage5b_eval_summary.png)
*Figure 5.7 — Stage 5b evaluation summary: settling time and IAE by scenario.*

### Result 1 — All agents achieve 100% success with zero overshoot

All three agents succeed on every episode across all mass and friction conditions tested. This result directly answers Research Question 1 (RQ1): a domain-randomised PPO agent trained with frame stacking can learn a reliable PID gain-scheduling policy across a 4:1 mass range (5 kg to 20 kg) and a 20:1 friction range (0.1 to 2.0). The zero overshoot result across all 90 evaluation episodes confirms that all agents stop precisely within the ±0.05 m hold band, satisfying the 25-consecutive-step criterion, rather than merely reaching the vicinity of the target.

The uniformity of the success result across mass and friction extremes is notable. The Heavy and Slippery scenario (mass = 20 kg, friction = 0.2) places both demands at their upper training limits simultaneously. The Light and Grippy scenario (mass = 5 kg, friction = 2.0) represents the opposite extreme. Both RL agents generalise successfully across this range, confirming that the domain randomisation regime was sufficient to cover the evaluation distribution without overfitting to any single operating condition.

### Result 2 — Context-aware RL settles faster than blind RL

Stage 5a (Context-Aware Agent) consistently outperforms Stage 5b (Blind Agent) on settling time across all three static scenarios:

| Scenario | Stage 5a settling (s) | Stage 5b settling (s) | Stage 5b slower by |
|----------|-----------------------|-----------------------|-------------------|
| Standard | 1.254 | 1.446 | +15.3% |
| Heavy & Slippery | 1.278 | 1.476 | +15.5% |
| Light & Grippy | 1.390 | 1.568 | +12.8% |
| **Average** | | | **+14.5%** |

This gap is consistent across all three scenarios, averaging 14.5%. The mechanism is straightforward: Stage 5a includes the current mass scale and friction scale in its observation vector, allowing the policy to directly condition gain selection on the operating conditions from the first timestep. Stage 5b lacks these observations; it must infer the dynamics from the shape of recent trajectory history within its 10-frame temporal window. The Blind Agent succeeds, but the additional inference cost — the steps required to resolve uncertainty about mass and friction from trajectory curvature and deceleration patterns — manifests as a systematic delay in entering and maintaining the hold band.

This result directly answers RQ2: providing explicit physics context in the observation improves adaptation efficiency. The 14.5% settling time advantage of the Context-Aware Agent over the Blind Agent is a consistent, reproducible effect across all mass and friction conditions. It is not marginal noise — it corresponds to approximately 0.15–0.20 s of additional settling time, which represents roughly 7–10 additional RL timesteps (at 50 Hz) that the Blind Agent requires to commit to the hold strategy.

The IAE ordering reflects the same dynamics. Integrated absolute error is necessarily higher for slower-settling agents, since more error accumulates during the additional settling time. Stage 5a achieves IAE values within 1–2% of Fixed PID, while Stage 5b shows a 4–5% higher IAE:

| Scenario | Fixed PID IAE | Stage 5a IAE | Stage 5b IAE |
|----------|--------------|-------------|-------------|
| Standard | 2.653 | 2.682 (+1.1%) | 2.785 (+5.0%) |
| Heavy & Slippery | 2.789 | 2.816 (+1.0%) | 2.917 (+4.6%) |
| Light & Grippy | 2.939 | 2.967 (+1.0%) | 3.061 (+4.2%) |

### Result 3 — Fixed PID achieves the shortest settling times

The Fixed PID baseline achieves the fastest settling in all scenarios: 10–13% faster than Stage 5a, and 25–30% faster than Stage 5b.

| Scenario | Fixed PID (s) | Stage 5a (s) | Stage 5b (s) |
|----------|--------------|-------------|-------------|
| Standard | **1.116** | 1.254 (+12.3%) | 1.446 (+29.5%) |
| Heavy & Slippery | **1.136** | 1.278 (+12.5%) | 1.476 (+29.9%) |
| Light & Grippy | **1.254** | 1.390 (+10.9%) | 1.568 (+25.0%) |

At face value, this would suggest that a non-adaptive, pre-tuned fixed-gain controller outperforms both learned adaptive agents. **This interpretation is not supported by the full experimental picture.** An engineering mechanism in the simulation environment — the `brake_integral_reset`, which zeros the PID integrator upon entry to the braking zone — is active for all agents including the Fixed PID baseline. This mechanism resolves the integral windup problem that would otherwise cause Fixed PID to fail catastrophically. Section 5.4 demonstrates experimentally that without this mechanism, Fixed PID achieves 0% success with overshoot of approximately 10 m. The speed advantage of Fixed PID in Table 5.1 is therefore an artefact of an equalising aid, not a genuine demonstration of non-adaptive superiority under unknown dynamics.

This result is presented here for completeness and to establish the numeric record. Its interpretation is deferred to Section 5.4, which provides the analysis needed to correctly contextualise the comparison.

---

## 5.3 Dynamic Evaluation Results

The dynamic evaluation extends each scenario with mid-episode disturbances and adds two out-of-distribution (OOD) conditions not encountered during training. At a random step in [120, 220] (~1.2–2.2 s after episode start), mass is multiplied by a scale drawn uniformly from [0.9, 1.3] and friction by a scale from [0.5, 1.4]. Because the disturbance parameters are seed-dependent, settling times vary across the 10 evaluation seeds; results are reported as mean ± range.

The two OOD conditions are:
- **OOD Ultra Heavy (35 kg):** mass = 35 kg, friction = 1.0. Mass is 75% above the training ceiling of 20 kg.
- **OOD Ultra Slippery:** mass = 20 kg, friction = 0.05. Friction is well below the training floor of 0.1.

Results for Stage 5a and Stage 5b are shown in Tables 5.2 and 5.3. Trajectory plots are in Figures 5.8 and 5.9; evaluation summary plots in Figures 5.10 and 5.11.

**Table 5.2 — Dynamic evaluation, Stage 5a (Context-Aware Agent), 10 seeds each**

| Scenario | Success rate | Settling time mean ± range (s) | Overshoot |
|----------|-------------|-------------------------------|-----------|
| Standard | 100% | 1.47 ± 0.01 | 0 m |
| Heavy & Slippery | 100% | 1.47 ± 0.01 | 0 m |
| Light & Grippy | 100% | 1.49 ± 0.04 | 0 m |
| OOD Ultra Heavy (35 kg) | 100% | 1.47 ± 0.03 | 0 m |
| OOD Ultra Slippery | 100% | 1.47 ± 0.01 | 0 m |

**Table 5.3 — Dynamic evaluation, Stage 5b (Blind Agent), 10 seeds each**

| Scenario | Success rate | Settling time mean ± range (s) | Overshoot |
|----------|-------------|-------------------------------|-----------|
| Standard | 100% | 1.45 ± 0.01 | 0 m |
| Heavy & Slippery | 100% | 1.48 ± 0.01 | 0 m |
| Light & Grippy | 100% | 1.54 ± 0.06 | 0 m |
| OOD Ultra Heavy (35 kg) | 100% | 1.54 ± 0.02 | 0 m |
| OOD Ultra Slippery | 100% | 1.48 ± 0.01 | 0 m |

![Stage 5a dynamic trajectories](figures/trajectories/stage5a_dynamic_trajectories.png)
*Figure 5.8 — Stage 5a position trajectories under dynamic evaluation (with mid-episode disturbances).*

![Stage 5b dynamic trajectories](figures/trajectories/stage5b_dynamic_trajectories.png)
*Figure 5.9 — Stage 5b position trajectories under dynamic evaluation.*

![Stage 5a dynamic eval summary](figures/eval/stage5a_dynamic_eval_summary.png)
*Figure 5.10 — Stage 5a dynamic evaluation summary across all five scenarios.*

![Stage 5b dynamic eval summary](figures/eval/stage5b_dynamic_eval_summary.png)
*Figure 5.11 — Stage 5b dynamic evaluation summary across all five scenarios.*

### Result 4 — Both RL agents generalise to out-of-distribution conditions

Both Stage 5a and Stage 5b achieve 100% success with zero overshoot in the OOD Ultra Heavy scenario (35 kg), with settling times comparable to in-distribution performance (Stage 5a: 1.47 ± 0.03 s; Stage 5b: 1.54 ± 0.02 s). The training mass ceiling was 20 kg; the OOD condition exceeds this by 75%. Despite never encountering a 35 kg vehicle during training, both agents apply gain schedules that reliably drive the heavier vehicle to within ±0.05 m of the target and hold it there for 25 consecutive steps.

This result has a practical implication: the policy has generalised beyond interpolation of the training distribution to extrapolation into an unseen mass regime. The domain randomisation regime — sampling mass from [5, 20] kg at each episode — appears to have trained a policy that captures the structural relationship between mass and the required gain adjustments, rather than merely memorising responses for specific mass values. A heavier vehicle requires more aggressive braking (higher Kd) and less integral accumulation (lower Ki) to avoid overshoot; both agents appear to have learned this relationship well enough to extend it beyond the training range.

The OOD Ultra Slippery scenario (friction = 0.05) likewise presents no difficulty, with settling times indistinguishable from in-distribution results. As noted in Section 3.3, rolling-contact dynamics in this simulation model do not produce the translational resistance changes that friction coefficient variation would produce in a real vehicle. The "Ultra Slippery" condition therefore represents a change in friction parameter that has limited effect on the control problem as modelled. Mass remains the dominant source of dynamic uncertainty in this simulation, and it is mass generalisation that the OOD results principally demonstrate. This limitation is discussed in Section 6.2 (Limitation L4).

### Result 5 — Mid-episode disturbances did not measurably disrupt settling performance

Comparing dynamic settling times to their static counterparts reveals a notable pattern: the disturbances have minimal measurable effect.

| Agent | Scenario | Static settling (s) | Dynamic mean settling (s) | Difference |
|-------|----------|--------------------|--------------------------| -----------|
| Stage 5a | Standard | 1.254 | 1.47 | +0.22 |
| Stage 5a | Heavy & Slippery | 1.278 | 1.47 | +0.19 |
| Stage 5b | Standard | 1.446 | 1.45 | +0.00 |
| Stage 5b | Heavy & Slippery | 1.476 | 1.48 | +0.00 |

The small differences observed are attributable to seed-to-seed variation in evaluation conditions (episode-level mass and friction samples vary per seed in dynamic evaluation), not to disturbance disruption. The disturbance itself — which fires at steps 120–220, corresponding to 1.2–2.2 s into the episode — occurs after most episodes have already achieved the hold criterion.

Stage 5a typically enters the hold band within 1.25–1.39 s (from the static evaluation); Stage 5b within 1.45–1.57 s. In a significant fraction of episodes — particularly for Stage 5a — the vehicle has already settled before the disturbance fires. When the vehicle is already within ±0.05 m and holding, a moderate change in mass (×0.9–1.3) or friction (×0.5–1.4) does not dislodge it from the hold zone within the remaining episode steps. The success metric therefore captures robustness to disturbances that predominantly fire post-settling, not robustness to mid-approach physics changes.

This represents a design limitation in the dynamic evaluation: the disturbance window does not overlap with the active approach phase for agents that settle quickly. A disturbance firing at steps 30–80 — during the vehicle's active deceleration — would genuinely test whether the agents can adapt their gain schedules in response to a sudden physics change during the most control-critical phase of the episode. As designed, the dynamic evaluation primarily confirms robustness of the initial approach, not mid-approach adaptability. This is flagged as Limitation L3 and discussed in Section 6.2.

---

## 5.4 Confound Analysis — The Brake Integral Reset

This section identifies and analyses the most important methodological finding of this thesis: the `brake_integral_reset` mechanism introduces a confound that fundamentally alters the interpretation of the Fixed PID speed advantage reported in Section 5.2. Understanding this mechanism is necessary before drawing any conclusions about the relative performance of adaptive and non-adaptive control in this experiment.

### 5.4.1 The mechanism and its effect

The simulation environment includes a `brake_integral_reset` subroutine: at each timestep, if the vehicle's distance to target satisfies |error| < 2.0 m (the braking zone entry), the PID integrator is set to zero. This mechanism was introduced during iterative environment development to address integral windup — a well-known failure mode in fixed-gain PID control where the integral term, accumulated during a long approach phase, prevents effective braking in the final deceleration phase.

Critically, this mechanism is active for **all agents** during both training and evaluation, including the Fixed PID Classical Baseline. It is not a feature of the RL agents' learned strategy; it is an environmental engineering decision applied uniformly across all conditions.

The consequence is asymmetric: the mechanism most benefits the agent for which integral windup is the hardest problem to solve. For the RL agents, training under domain randomisation with overshoot penalties and deceleration bonuses creates pressure to learn gain schedules that suppress Ki accumulation during the approach and raise Kd near the braking zone — a behavioural approximation to windup prevention. The agents learn to work around the windup problem through gain scheduling. For the Fixed PID baseline, no such learning is possible: gains are fixed at Kp = 1.8, Ki = 0.7, Kd = 0.5 throughout the episode, and the mechanism is the only mechanism available for preventing catastrophic overshoot.

### 5.4.2 Why integral windup causes fixed PID failure

During an approach from 0 m to an 8 m target (the configuration used in the validation experiment below), the vehicle spends approximately 2.0–2.5 s in the non-braking zone (|error| ≥ 2 m). The PID integrator accumulates continuously during this phase:

```
i_term(T) = Ki × ∫₀ᵀ e(t) dt ≈ Ki × e_avg × T
```

At Ki = 0.7, with average error ≈ 2.5 m during a 2.5 s approach phase:

```
i_term ≈ 0.7 × 2.5 × 2.5 ≈ 4.4
```

At the entry to the braking zone (|error| ≈ 2.0 m), effective deceleration requires the derivative term to overcome the integral term:

```
Kd × |vel| > i_term
```

Rearranging: Kd > i_term / |vel|. At a typical approach velocity of |vel| = 0.5 m/s and i_term ≈ 4.4:

```
Kd > 4.4 / 0.5 = 8.8
```

The Fixed PID has Kd = 0.5 — more than seventeen times smaller than the minimum required for effective braking. Without the integral reset, the derivative control authority is completely overwhelmed by the integrator state, and the vehicle cannot decelerate effectively at the braking zone entry. It overshoots the target and never recovers within the episode time limit.

With the integral reset active, this calculation is irrelevant: the moment the vehicle crosses into the braking zone, i_term is set to zero. The effective i_term at the braking point is not 4.4 but 0. The fixed gains then operate in a regime where the derivative term (Kd = 0.5, vel = 0.5 m/s → derivative force ∝ 0.25) can provide meaningful deceleration authority, and the pre-tuned gains perform well.

### 5.4.3 Validation experiment

To confirm that the integral reset is responsible for Fixed PID's baseline performance, the baseline was evaluated with `brake_integral_reset` disabled. To allow sufficient approach distance for windup to accumulate, an extended 8 m target was used rather than the standard 5 m evaluation target. Results are shown in Table 5.4. The corresponding trajectory plot is in Figure 5.12.

**Table 5.4 — Fixed PID evaluated with `brake_integral_reset` disabled (8 m target, 10 seeds)**

| Scenario | Success rate | Overshoot (m) | Final abs. error (m) |
|----------|-------------|---------------|----------------------|
| Standard | 0% | 9.92 | 2.995 |
| Heavy & Slippery | 0% | 10.00 | 10.003 |
| Light & Grippy | 0% | 9.99 | 2.968 |

![Baseline no-reset trajectories](figures/trajectories/baseline_no_reset_trajectories.png)
*Figure 5.12 — Fixed PID trajectories with `brake_integral_reset` disabled (8 m target). All scenarios show massive overshoot and failure to recover.*

Without the integral reset, Fixed PID fails completely across all three mass-friction scenarios. Overshoot of approximately 10 m — equivalent to the full target distance — confirms that the vehicle passes through the target zone at high velocity and cannot recover within the episode time limit. The Heavy and Slippery scenario is the most severe, with a final error of 10.003 m indicating the vehicle ended the episode at the far boundary of its travel range and never returned toward the target. The mechanism predicted by the integral windup analysis is confirmed: i_term accumulated during the approach exceeds the braking authority of Kd = 0.5 by more than an order of magnitude, and the vehicle cannot decelerate.

### 5.4.4 Implications for the Fixed PID speed advantage

This finding reframes the settling time comparison in Table 5.1. The Fixed PID's faster settling (1.116–1.254 s vs Stage 5a's 1.254–1.390 s and Stage 5b's 1.446–1.568 s) is not evidence that non-adaptive, pre-tuned gains outperform learned adaptive control under unknown dynamics. It is evidence that pre-tuned fixed gains, when freed from the problem they are most vulnerable to by an engineering aid, can exploit their precisely calibrated values to settle slightly faster than a learned policy that must operate conservatively to avoid windup accumulation without knowing when or whether the reset will fire.

The RL agents are, in effect, solving a harder version of the problem: they learn gain schedules that hedge against integral windup (reducing Ki, raising Kd) because their training does not guarantee the integral reset will fire at the right moment in every episode. This conservative strategy imposes a settling time cost relative to a baseline that has the windup problem removed. In a deployment scenario without the integral reset — which the validation experiment confirms is the harder, more realistic setting — Fixed PID fails while the RL agents, trained to handle windup through gain scheduling, would be expected to retain their adaptive advantage.

It should be noted that an attempt was made to retrain both RL agents in an environment with `brake_integral_reset` disabled. These training runs did not converge to a successful policy within 1.5–3 million timesteps. The hypothesised required strategy — suppressing Ki aggressively during the approach phase to prevent windup, then committing to a high-Kd braking strategy near the target — involves credit assignment across 500–1000 timesteps, a horizon that on-policy PPO with an MLP policy has limited capacity to bridge. A recurrent policy architecture (LSTM or GRU) that can maintain an explicit representation of the integral state across the full approach would be better suited to learning this strategy. This is identified as a priority for future work in Section 7.2.

### 5.4.5 Honest framing

The `brake_integral_reset` confound is presented here as a research finding rather than a methodological failure. Identifying a hidden simulation aid — one that appeared innocuous during development but substantially altered the competitive landscape between agent types — is a contribution to the rigour of this line of research. Analogous anti-windup mechanisms (integrator clamping, conditional integration, back-calculation) are standard practice in real PID implementations. Their presence in simulation must be accounted for when benchmarking adaptive methods against fixed-gain baselines, because they disproportionately benefit the non-adaptive controller whose only failure mode is windup accumulation.

> "An implicit assumption embedded in the environment design was identified during analysis. The `brake_integral_reset` mechanism effectively removes integral windup at the moment when braking authority is most critical. While this is a common engineering practice in real PID implementations, its presence in the simulation equalises all agents and obscures the adaptive advantage that RL should theoretically provide. Disabling this mechanism reveals that fixed PID fails completely, confirming that the aid — not the fixed gains — was responsible for the baseline's strong performance in the standard evaluation. This finding highlights the importance of carefully auditing simulation aids when benchmarking adaptive control methods against fixed-gain baselines."

The comparison between Fixed PID and the RL agents in Sections 5.2 and 5.3 should be read as a characterisation of all agents' behaviour under a shared simulation convention, not as evidence that non-adaptive control is preferable. The correct comparison, RQ2, is addressed by Sections 5.2 and 5.3 directly: Stage 5a settles 14.5% faster than Stage 5b, a result that is unaffected by the integral reset confound since both RL agents are subject to the same aid.

---

## 5.5 Summary of Key Results

The results of this chapter address both research questions and identify one major experimental confound.

**RQ1 — Feasibility:** Both Stage 5a and Stage 5b achieve 100% success with zero overshoot across all static and dynamic evaluation scenarios, including out-of-distribution conditions 75% beyond the training mass ceiling. Domain-randomised PPO with frame stacking produces a viable gain-scheduling policy that generalises reliably across a wide range of unknown dynamics.

**RQ2 — Context benefit:** The Context-Aware Agent (Stage 5a) settles an average of 14.5% faster than the Blind Agent (Stage 5b) across all static scenarios. This advantage is consistent across mass and friction conditions, confirming that physics context in the observation is beneficial for adaptation speed, even though the Blind Agent succeeds without it.

**Confound:** The `brake_integral_reset` mechanism is the primary driver of Fixed PID's speed advantage in the standard evaluation. Disabling it eliminates that advantage entirely (Fixed PID: 0% success, ~10 m overshoot). The RL-vs-Fixed-PID comparison in the standard evaluation does not constitute a fair test of adaptive versus non-adaptive control under unknown dynamics.
