# Chapter 4 — Experimental Setup

This chapter defines how the controllers are evaluated and compared: the
evaluation protocol and its honest treatment of variance, the scenario suite,
the agents and baselines, the metrics, the statistical methodology, and the
compute environment.

## 4.1 Evaluation Protocol (v2)

A controller is evaluated by rolling it out, with a deterministic policy, on a
scenario for a fixed budget of up to 5000 control steps (100 s) and recording
the per-step position trace. The original protocol (v1) fixed the scenario
physics exactly and repeated the identical deterministic episode ten times, so
every "seed" produced byte-identical numbers — reporting $n=1$ as $n=10$, with
zero measured variance. Protocol v2 corrects this by introducing genuine
per-episode variation:

- **Physics bands.** Each episode samples mass within $\pm10\%$, friction
  within $\pm10\%$, and actuator strength within $\pm5\%$ of the scenario
  centre.
- **Target jitter.** The target distance is jittered $\pm0.5$ m per episode.
- **True context push.** The sampled physics are written *both* to the plant
  and to the context observation. (In v1 the context observation was left at a
  constant $(1,1)$, ~11× outside the agent's training context distribution —
  the bug that manufactured the original RQ2 result.)

Ten episodes are drawn per scenario per seed, over five training seeds, giving
50 genuinely distinct episodes per scenario per agent.

**Success criterion.** Position within $\pm0.05$ m of target for 25 consecutive
control steps. **Settling time** is the earliest time after which the error
remains within band; **overshoot** is $\max(0, \max_t x_t - x_{\text{target}})$;
**IAE** is $\int|e|\,dt$ over the episode; all use $\Delta t = 0.02$ s.

## 4.2 Scenario Suite

Eight static scenarios span the three dynamics axes, including out-of-
distribution (OOD) conditions beyond the training ranges:

| Scenario | Mass (kg) | Friction | Actuator | Note |
|----------|-----------|----------|----------|------|
| Standard | 10 | 1.0 | 1.0 | nominal |
| Light Strong Motor | 6 | 1.0 | 1.3 | fast |
| Heavy Weak Motor | 18 | 1.0 | 0.7 | slow |
| Heavy and Slippery | 20 | 0.2 | 1.0 | mass edge |
| Light and Grippy | 5 | 2.0 | 1.0 | mass edge |
| OOD Ultra Heavy | 35 | 1.0 | 1.0 | mass 1.75× ceiling |
| OOD Weak Motor | 10 | 1.0 | 0.45 | actuator below floor |
| OOD Heavy Weak | 30 | 1.0 | 0.55 | combined OOD |

**Dynamic evaluation** adds the mid-episode disturbance (§3.3) to each
scenario. **Robustness probes** (Chapter 5) additionally sweep a single axis
(mass 5→50 kg; actuator 0.5→2.0) and apply large mid-approach shocks (mass
×2.5; actuator ×0.5). The pendulum uses an analogous seven-scenario suite over
pole mass and gear, including OOD corners.

## 4.3 Agents and Baselines

| Controller | Type | Observation | Action |
|------------|------|-------------|--------|
| Fixed PID | classical | — | constant $[0,0,0]$ |
| Anti-Windup PID | classical | — | constant, + back-calc / clamp |
| MRAC | classical adaptive | error/ref-model | MIT-rule gains |
| Context-Aware (Stage 6a) | learned (teacher) | 9-dim ×10 | scheduled gains |
| Blind (Stage 6b) | learned (student) | 6-dim ×10 | scheduled gains |
| GRU Blind (Stage 6d) | learned (student) | 6-dim, recurrent | scheduled gains |
| Stack-$k$ Blind (Stage 6c) | learned (student) | 6-dim ×$k$ | scheduled gains |

Learned agents are trained for 1M steps on the `thesis_v6_hipmdp` protocol;
the context/blind pair uses five seeds, the ablations one to two seeds (trends,
not per-point variance).

## 4.4 Metrics and Statistical Methodology

Three nested sources of variation are kept separate: **training seed** (5),
**evaluation episode** (10 per scenario, genuinely varied under v2), and
**scenario** (reported separately, never pooled).

Per-scenario results are reported as mean ± standard deviation over the 50
episodes and, for headline comparisons, as the **seed-level 95% bootstrap
confidence interval** (10,000 resamples of the five per-seed means, percentile
method — preferred over a normal approximation because $n_s = 5$ is far from
asymptotic). Success rates use **Wilson score intervals**, which behave
correctly near 0% and 100%.

Pairwise method comparisons (e.g. context vs blind on a scenario) use **Welch's
t-test** on the seed-level means, cross-checked with the distribution-free
**Mann–Whitney U** where settling times are right-skewed by timeouts. Effect
size is **Cliff's delta** (rank-based, robust to timeout truncation). Families
of related comparisons are corrected with **Holm–Bonferroni**, reporting raw
and adjusted $p$.

What is *not* claimed: five seeds bound but do not precisely estimate seed-level
variance, so CIs are wide and stated as such; episode-level pooling is not
treated as independent sampling of the training process; swept-parameter curves
(mass, actuator, stack depth) are descriptive trends with per-point dispersion,
not hypothesis tests at every point.

## 4.5 Reproducibility and Compute

All training and evaluation ran on a single laptop (Intel i7-11800H, 32 GB RAM,
RTX 3070 Laptop; the small MLP/GRU policies train on CPU). A 1M-step car run
takes ~15 min (MLP) to ~75 min (GRU); the full 5-seed teacher/student set,
ablations, and pendulum runs total roughly a day of wall-clock compute. Code,
configuration presets (`thesis_v6_hipmdp`), trained weights, per-seed CSVs, and
the analysis scripts are provided with the submission; the corrected
evaluation pipeline and the audit log (`AUDIT_FINDINGS.md`) document the
provenance of every reported number. Two operational interruptions (a sleep
suspend and a transient Windows DLL-init failure) were handled with idempotent,
resumable run scripts and did not affect results.
