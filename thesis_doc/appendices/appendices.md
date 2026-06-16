# Appendices

## Appendix A — Full Hyperparameters

### A.1 PPO (car, both agents)

| Parameter | Value |
|-----------|-------|
| Algorithm | PPO (clipped surrogate + GAE) |
| Learning rate | 3×10⁻⁴, linear decay to 0 |
| Discount γ | 0.99 |
| GAE λ | 0.95 |
| Clip ε | 0.2 |
| Value coefficient | 0.5 |
| Entropy coefficient | 0.0 |
| Max grad norm | 0.5 |
| Minibatch size | 64 |
| Update epochs | 10 |
| Rollout steps / env | 2048 |
| Parallel envs | 4 |
| Total timesteps | 1,000,000 |
| Frame stack (MLP) | 10 |
| Recurrent hidden (GRU) | 128 |
| Actor/critic | 2×64 Tanh, orthogonal init |
| Training seeds | 7, 21, 42, 84, 123 |

### A.2 Environment (car, `thesis_v6_hipmdp`)

| Parameter | Value |
|-----------|-------|
| Physics timestep | 0.002 s |
| frame_skip | 10 → control period 0.02 s |
| Target (train) | 3–7 m, 4-phase curriculum to 1–10 m |
| Target (eval) | 5 m ± 0.5 m jitter |
| Hold criterion | ±0.05 m for 25 steps |
| Max eval steps | 5000 (100 s) |
| Mass range | 5–20 kg |
| Friction range | 0.1–2.0 (inert) |
| Actuator strength range | 0.6–1.4 |
| Mid-episode disturbance | step 120–220; mass ×[0.9,1.3], fric ×[0.5,1.4], act ×[0.85,1.15] |
| Friction patch | x∈[1.5,2.4] m, ×0.35 |
| Base gains | Kp=1.8, Ki=0.7, Kd=0.5 |
| Gain deltas | ΔKp=1.0, ΔKi=0.6, ΔKd=2.0 |
| Reward | 5·Δx − 0.75·|e| − 0.12·ẋ² − 2·overshoot; +decel bonus, +80 hold |

### A.3 Reward-term rationale

| Term | Purpose | Note |
|------|---------|------|
| 5·Δx progress | drive toward target | zeroed inside braking zone (|e|<2 m) |
| −0.75·|e| distance | reduce error | zeroed inside braking zone |
| −0.12·ẋ² velocity | discourage excess speed | active throughout |
| −2·max(0,x−tgt) overshoot | punish passing target | active throughout |
| +10·Δ(−ẋ) decel bonus | dense braking signal | braking zone only |
| +80 terminal | reward held stop | on 25-step hold |

(Governor and hard-overshoot "cliff" variants were tried and dropped — they
starved exploration; see §3.2.)

### A.4 Anti-windup PID

Back-calculation tracking constant Tt = 1 s; conditional-integration variant
suspends integration when saturated and error deepens saturation. Output limits
[−1, 1]. Base gains as above.

### A.5 MRAC (feasible)

Reference model τ_m ∈ {10, 15} s (feasible at v_max ≈ 0.5 m/s); normalized MIT
rule dK = −γ·e_m·φ/(β+φ²); velocity regressor for Kp; σ-modification leakage
{0, 0.05}; control period 0.02 s (corrected).

### A.6 Pendulum

Pole-mass scale 0.5–2.5; gear scale 0.6–1.4; frame_skip 2 (control 0.04 s);
base angle-PID Kp=3.5, Kd=0.4; cascade outer loop kx=−0.05, kẋ=−0.10,
θ_ref_max=0.15 rad; PPO 300k steps, seeds 7, 21.

### A.7 Dropped reward variants (negative design results)

Two earlier reward/protocol variants were evaluated and discarded; they are
recorded because their failure modes informed the final design.

- **Speed governor (`thesis_v3_safety`).** A hard cap on approach speed
  (braking command injected when the predicted stopping distance exceeded the
  remaining distance). It produced safe trajectories but *masked* what the
  policy had learned about braking — every agent looked equally good because
  the governor, not the learned gains, did the stopping. Removed so the
  evaluation reflects the learned controller.
- **Overshoot cliff.** A large terminating penalty (−300) for passing the
  target. Intended to teach precise stopping, it instead starved exploration:
  the episode ended on the first overshoot, the agent rarely survived to reach
  the hold bonus, and the value function collapsed toward the cliff penalty
  everywhere (zero useful gradient toward braking). Replaced by the soft
  per-step overshoot penalty (−2·overshoot) plus the dense deceleration bonus.

The lesson — that on an input-saturated plant, *hard* safety constraints
(governor, cliff) either mask the learned behaviour or starve exploration,
whereas *dense soft* shaping (progress, decel bonus) succeeds — is the reward-
design analogue of the anti-windup finding in Chapter 5.

## Appendix B — Full Per-Seed Results

Per-seed `eval_seed_summary.csv` files and cross-seed aggregates
(`aggregate_all_seeds.csv`, with bootstrap 95% CIs) are provided for:
`stage6a_context_hipmdp/`, `stage6b_blind_hipmdp/`,
`stage6c_stack{1,3,5,20}/`, `stage6d_gru_blind_s{7,21}/`. Sweep, shock,
probing, gain-analysis, anti-windup, MRAC, and pendulum CSVs are under the
correspondingly named `benchmark_results/` directories. Each cross-seed
aggregate is produced by `utils/aggregate_seeds.py`, which reports, per
scenario and metric, the mean of the five per-seed means, their standard
deviation, and a percentile bootstrap 95% confidence interval over the five
seed means (10,000 resamples). Because $n_s=5$, these intervals are wide by
construction and are reported as such; they bound, rather than precisely
estimate, the seed-level variance (§4.4).

### B.1 Reproducibility and compute log

All runs were executed on a single laptop (Intel i7-11800H, 32 GB RAM, RTX 3070
Laptop), CPU-bound for the small policies. Indicative wall-clock: car MLP
1M-step run ≈ 15 min; car GRU ≈ 75 min; pendulum 300k ≈ 8–12 min; eval-only
passes (sweeps, probes, shock) ≈ 1–10 min each. Two operational interruptions
occurred and were handled without affecting results: a laptop sleep/hibernate
suspended a training chain (resolved by disabling AC sleep and resuming only
the missing seeds, since the PPO implementation checkpoints only at seed end),
and a transient Windows DLL-initialization failure (`0xC0000142`) aborted a
batch under `set -e` (resolved with idempotent, retry-once resume scripts).
Both incidents, and the full audit trail, are documented in the accompanying
`PROJECT_LOG/` and `AUDIT_FINDINGS.md`.

## Appendix C — Code and Artifact Map

| Artifact | Path |
|----------|------|
| Training / eval | `stage2_meta_rl_reproduction.py` (`--protocol-preset thesis_v6_hipmdp`, `--eval-protocol v2`) |
| Stage 6 runners | `stage6a_context_hipmdp.py`, `stage6b_blind_hipmdp.py`, `stage6c_stack_ablation.py`, `stage6d_gru_blind.py` |
| Car environment | `envs/adaptive_suspension.py` (9-dim obs) |
| Pendulum env / exp | `envs/adaptive_pendulum.py`, `pendulum_experiment.py` |
| Domain randomization | `agents/domain_randomization.py` (mass/friction/actuator) |
| Policy | `agents/model.py` (MLP + GRU) |
| PID + anti-windup | `utils/pid.py` (`AntiWindupPIDController`) |
| Anti-windup baseline | `stage_baseline_antiwindup_pid.py` |
| MRAC (feasible) | `stage4b_mrac_feasible.py` |
| Probing (RQ4) | `probe_hidden_params.py` |
| Sweeps | `eval_mass_sweep.py` (`--axis mass|actuator`) |
| Shock probe | `stage6_shock_probe.py` |
| Gain analysis | `analyze_gain_trajectories.py` |
| Cross-seed aggregation | `utils/aggregate_seeds.py` |
| Audit log | `AUDIT_FINDINGS.md` |
| Resumable run scripts | `resume_stage6.sh`, `resume_phaseC.sh` |

## Appendix D — Audit Cross-Reference

| Defect | Effect on original claim | Correction |
|--------|--------------------------|------------|
| dt = physics timestep, not control period | settling/IAE 10× too small | use env.dt = 0.02 s |
| context obs not pushed at eval | teacher saw (1,1), ~11× OOD → fake +14.5% | protocol v2 true-context push |
| no-reset metrics vs wrong target | "naive PID 0%, never recovers" | measure vs env target (8 m) |
| brake_integral_reset (task-aware aid) | masked adaptive vs non-adaptive | fair anti-windup baseline |
| mass-only randomization (inert) | task non-discriminative | added actuator-strength axis |
| MRAC τ_m infeasible + dt bug | strawman failure | feasible τ_m, normalized MIT |
