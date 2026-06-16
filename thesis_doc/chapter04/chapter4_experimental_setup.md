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

### 4.1.1 Metric definitions

All metrics are computed from the per-control-step position trace
$\{x_t\}_{t=0}^{T}$ with error $e_t = x_{\text{target}} - x_t$ and control
period $\Delta t = 0.02$ s.

- **Success** $\in\{0,1\}$: the episode achieves the hold criterion — there
  exists a window of $H=25$ consecutive steps, all with $|e_t|\le\delta$
  ($\delta = 0.05$ m). This is the binary task outcome.
- **Settling time** $t_s$: the earliest time from which the error remains
  within the band for the rest of the run,
  $t_s = \Delta t\cdot\min\{i : |e_j|\le\delta\ \forall j\ge i\}$. If no such
  $i$ exists the episode is recorded as a timeout at $T\Delta t$ (100 s). This
  "stays-within" definition (rather than first-entry) avoids crediting an
  agent that enters the band, overshoots out, and returns.
- **Overshoot** $= \max(0,\ \max_t x_t - x_{\text{target}})$ — the furthest the
  vehicle travels past the target.
- **IAE** (integral of absolute error) $= \sum_t |e_t|\,\Delta t$ — a single
  scalar capturing the whole transient's error mass; lower is tighter tracking.
- **Final absolute error** $= |e_T|$ — residual at episode end, a check that a
  "success" is genuinely held, not a fly-through.

The unit error corrected in the audit (F1) affected $t_s$ and IAE (both scale
with $\Delta t$): using the physics timestep 0.002 s instead of the control
period 0.02 s reported them 10× too small. Overshoot, success, and final error
are dimensionless-in-time and were unaffected — which is one reason the error
went unnoticed originally (the success rates looked correct).

## 4.2 Scenario Suite

The scenario suite is designed, not arbitrary: each row isolates a specific
question about generalization. The first five sit inside the training
distribution and probe different *combinations* of the axes — a nominal anchor,
a fast plant (light + strong motor) and a slow plant (heavy + weak motor) to
span the discriminating actuator axis, and the two mass edges (heavy-slippery,
light-grippy) retained from the original design to confirm that the friction
axis is inert (these two should, and do, behave like their mass-only
equivalents). The last three are deliberately **out-of-distribution**, one per
extrapolation direction: OOD Ultra Heavy pushes mass to 1.75× the training
ceiling, OOD Weak Motor pushes the actuator below the training floor (the
single hardest in-isolation axis), and OOD Heavy Weak combines an above-range
mass with a below-range actuator to test whether failures *compound* off-
distribution. Reporting these separately (never pooled) lets a reader see
exactly where, if anywhere, the learned policies break — and they do not, which
is the substance of the RQ1 generalization claim.

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

**Worked example of the seed-level test.** Take the Standard scenario,
context vs blind. Each agent yields five per-seed settling means (one per
training seed, each itself averaged over ten episodes); the context means
cluster near 13.7 s, the blind near 14.6 s, with per-seed standard deviations
of ~0.7 s. Welch's statistic is $t = (\bar x_c - \bar x_b)/\sqrt{s_c^2/5 +
s_b^2/5}$; with a ~0.85 s difference and ~0.7 s within-group spread this gives
$t\approx 1.9$ on ~8 effective degrees of freedom (Welch–Satterthwaite),
$p\approx 0.09$ raw — and after Holm–Bonferroni over the eight scenarios, the
Standard difference does *not* clear $\alpha=0.05$, whereas the larger-gap fast
scenarios (Light Strong Motor, +12.8%) do. Cliff's delta on the same seed means
is $\delta\approx 0.4$ (medium). This is exactly why the thesis reports RQ2 as a
*regime-dependent trend that is significant on the fast scenarios and within
noise on the slow ones*, rather than a single global $p$-value: with five seeds
the test is honestly underpowered for the smaller gaps, and saying so is more
defensible than over-claiming. The 95% bootstrap CI for the context Standard
mean (10,000 resamples of the five seed means) is roughly $13.7 \pm 0.6$ s —
wide, as five points require, and reported as such.

What is *not* claimed: five seeds bound but do not precisely estimate seed-level
variance, so CIs are wide and stated as such; episode-level pooling is not
treated as independent sampling of the training process; swept-parameter curves
(mass, actuator, stack depth) are descriptive trends with per-point dispersion,
not hypothesis tests at every point. Where a single-seed ablation is reported
(stack depth, GRU eval) it is explicitly a *trend*, and no inferential claim is
attached to a difference smaller than the seed-level spread measured on the
five-seed pair.

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
