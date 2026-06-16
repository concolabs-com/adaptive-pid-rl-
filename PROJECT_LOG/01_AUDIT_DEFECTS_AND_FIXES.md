# 01 — Audit: Defects, Root Causes, and Fixes

The original results told a clean, attractive story that turned out to rest on
measurement and environment-design artifacts. Nine findings (F1–F9) emerged
from the audit; five of them (F1, F4, F5, F6, F9) materially changed
conclusions. Each is documented below with **symptom → how it was found →
root cause → fix → before/after**.

---

## F1 — Settling times were 10× too small (dt unit error)

**Symptom.** Original thesis reported settling times of ~1.1–1.6 s for a 5 m
drive.

**How found.** Sanity check against the plant's terminal speed: the car's
actuator-limited cruise is ≈ 0.5 m/s, so a 5 m drive *cannot* finish in ~1.2 s
(needs ≥10 s). The reported episode length (652 control steps) × the claimed
"settling 1.254 s" implied 0.002 s/step — exactly `model.opt.timestep`, not the
control period `env.dt`.

**Root cause.** `run_eval_episode` and the plot/baseline scripts computed
settling time and IAE with `dt = model.opt.timestep` (0.002 s) instead of the
control period `env.dt = timestep × frame_skip = 0.002 × 10 = 0.02 s`. Position
is logged once per control step, so the correct divisor is 0.02 s. Everything
was therefore 10× too fast. The progress tracker had *noted* the fix was needed
but it was never actually applied to the data; the `thesis_writing/scripts/`
copies were patched while the root scripts (the ones actually run) were not.

**Fix.** Changed `model.opt.timestep` → `env.unwrapped.dt` in every eval/plot
script: `stage2_meta_rl_reproduction.py` (×2), `stage_baseline_fixed_pid.py`,
`stage_mass_shock_eval.py`, `stage2_sac_reproduction.py`,
`compare_three_methods_random_distance.py`, `eval_mid_episode_friction_probe.py`,
`plot_trajectories.py`, `plot_simulated_run_behaviors.py`,
`plot_stage2_diagnostics.py`, `plot_fair_pid_comparison.py`. The PID sub-step
loop in `envs/adaptive_suspension.py` correctly uses `model.opt.timestep`
(physics rate) — left untouched. Then reran every evaluation.

**Before → after (static settling, s).** Fixed PID 1.116→**11.16**; Stage 5a
1.254→**12.54**; Stage 5b 1.446→**14.46**. Relative orderings unchanged; the
whole thesis's absolute numbers moved 10×.

---

## F2 — "Disturbances fire after settling" was backwards

**Symptom.** Original Result 5 / Limitation L3 claimed mid-episode disturbances
(steps 120–220) fired *after* the agents settled (~1.3 s), so the dynamic
evaluation tested nothing.

**Root cause.** A consequence of F1. With correct dt, the disturbance window is
2.4–4.4 s and settling is ~12–15 s, so disturbances fire **during the
approach**, in every episode.

**Fix / consequence.** No code fix; the *analysis* was rewritten. The dynamic
evaluation is actually valid, and disturbances do perturb (e.g. Stage 5a Light
& Grippy static 13.9 s → dynamic 19.7 ± 8.5 s). This *improved* the thesis.

---

## F3 — "10 evaluation seeds" were 10 identical episodes

**Symptom.** All 10 per-scenario eval rows were byte-identical to 15 decimals.

**Root cause.** Deterministic policy + fixed scenario physics + no per-episode
randomness ⇒ the "10 seeds" reran the identical episode. n=1 reported as n=10;
all variance was exactly 0.

**Fix.** Evaluation **protocol v2**: per-episode physics bands (mass ±10%,
friction ±10%, actuator ±5%), per-episode target jitter (±0.5 m), per-episode
RNG seeded by episode seed. Now 50 genuinely distinct episodes per scenario
(10 × 5 seeds), with real mean ± sd and bootstrap CIs.

---

## F4 — No-reset baseline measured against the wrong target

**Symptom.** Original Table 5.4: fixed PID without the integral-reset aid =
"0% success, ~10 m overshoot, never recovers."

**Root cause.** The no-reset environment uses target = 8 m (the V5 preset), but
the metric code compared positions against the module constant `TARGET_POS = 5`
m. So a car correctly approaching 8 m was scored as 3 m of error / massive
overshoot relative to the wrong 5 m line.

**Fix.** `stage_baseline_fixed_pid.py` now reads `env.unwrapped.target_pos` for
error/overshoot/settling. Reran.

**Before → after.** "0% / never recovers" → naive PID actually **recovers on
2/3 scenarios** (Standard 100% @ 73.98 s, 6.92 m overshoot transient; Light &
Grippy 100% @ 92.5 s; Heavy & Slippery genuinely fails via backward runaway).
The catastrophic-windup-transient story survives; the "total failure" claim
does not.

---

## F5 — The fair classical baseline (anti-windup PID) was missing

**Symptom.** The comparison was PID-with-task-aware-aid vs PID-with-no-windup-
protection — two unrealistic extremes. Real PID ships anti-windup.

**Root cause.** `utils/pid.py` had no anti-windup; the env's
`brake_integral_reset` (a task-aware integrator zero in the braking zone) was
the only windup protection, and it uses privileged distance-to-target info a
generic PID block does not have.

**Fix.** Added `AntiWindupPIDController` (back-calculation with tracking
constant Tt=1 s; and conditional integration / clamping) and
`stage_baseline_antiwindup_pid.py`. Evaluated with and without the env aid.

**Result.** **Anti-windup PID solves the no-reset task: 100% success, ~21 s,
0.13 m overshoot.** With the env aid active it is indistinguishable from plain
PID (aid masks windup). **Implication:** "fixed PID fails, therefore RL is
needed" is untenable — two lines of textbook engineering close the windup gap.
RL's value must rest on *adaptation*, not windup handling. This reframed the
entire confound chapter.

---

## F6 — The RQ2 "context is 14.5% faster" result was a bug artifact

**Symptom.** Original headline: context-aware agent settles 14.5% faster than
blind, uniformly across all scenarios.

**How found.** When re-running the old context agent with the corrected eval
that *pushes the true context* to the observation, its advantage collapsed to
≈ the blind agent's level. Re-running with the *broken* (1,1) context
reproduced the original fast numbers.

**Root cause.** `prepare_eval_episode` (v1) wrote the scenario mass/friction
into the MuJoCo model but **never pushed them into the context observation**;
`reset_model` then re-zeroed the context to (1,1). So in *all* static
evaluations the context agent observed (mass_scale, friction_scale) = (1.0,
1.0). Its training context (normalized by the XML nominal mass ≈ 0.43 kg)
spanned mass_scale ≈ [11.6, 46.5] — the eval context was ~11× outside the
training distribution. The agent's "advantage" was an out-of-distribution
artifact, not a real benefit of context.

**Fix.** Eval protocol v2 pushes the true sampled context (`apply_eval_physics`
sets model + context, re-applied after reset). The dynamic eval was always
correct (the DomainRandomizationWrapper pushes context), which is why only the
static RQ2 number was wrong. RQ2 was then re-answered honestly with the 5-seed
Stage 6 retrains.

**Before → after.** "+14.5% everywhere" → **+1–13%, regime-dependent** (largest
when the plant is fast, ≈0 when actuator-limited).

---

## F7 — MRAC was a strawman

**Symptom.** Original MRAC: 0% success, dismissed as classical-adaptive failure.

**Root cause.** Two compounding errors. (a) `CONTROL_DT` was hardcoded 0.1 s but
the env control period is 0.02 s, so the reference model evolved 5× faster than
designed (effective τ_m = 0.6 s instead of 3 s). (b) Even at τ_m = 3 s the
reference model was *infeasible*: a first-order reference to a 5 m target has
initial required speed = 5/3 ≈ 1.67 m/s, far above the plant's 0.5 m/s terminal
speed, so the model-following error could never converge regardless of the
adaptation law.

**Fix.** `stage4b_mrac_feasible.py`: dt taken from `env.dt`; feasible τ_m ∈
{10,15} s (initial ref speed ≤ 0.5 m/s); normalized MIT rule
`dK = -γ·e·φ/(β+φ²)`; velocity regressor for Kp; σ-modification {0, 0.05}.

**Result.** **Still 0% success** on all configs (overshoot 2.4–4.6 m, timeout),
and underperforms even naive fixed PID. Now a *defensible* negative result: the
MIT rule's sensitivity assumption breaks on this saturated nonlinear plant even
when configured favourably.

---

## F8 — The friction axis is dynamically inert

**Symptom.** "Heavy and Slippery" (mass 20, fr 0.2) and "OOD Ultra Slippery"
(mass 20, fr 0.05) produced *identical* results to 3+ decimals.

**Root cause.** The car's wheels roll without slipping at these operating
points, so the sliding-friction coefficient does not enter the translational
dynamics. The probing analysis (F-RQ4) later confirmed friction is at best
weakly/spuriously decodable.

**Fix / consequence.** Documented as a limitation (L1). Motivated adding a
*real* second dynamics axis (F9). Friction randomization retained for
continuity but acknowledged to provide no signal.

---

## F9 — The task was non-discriminative (mass-only); added actuator axis

**Symptom.** A 4× mass change moved fixed-PID settling only ~2% (1.116→1.136
"s"); every controller scored 100% / 0 overshoot — nothing separated them.

**Root cause.** Settling for a multi-metre drive is dominated by travel time
≈ distance / v_max, and v_max ∝ κ/b is **independent of mass** (mass only
affects the transient). So mass is a poor discriminator at the actuator-limited
cruise.

**Fix.** Added an **actuator-strength axis** that scales motor gain + joint
force limit together (range 0.6–1.4), so v_max ∝ κ. Observation extended to 9
dims (3rd context = actuator scale); new preset `thesis_v6_hipmdp`; mid-episode
actuator shifts added. Implemented in `agents/domain_randomization.py` and
`envs/adaptive_suspension.py`.

**Result.** Actuator sweep 0.5→2.0 moves fixed-PID settling **21.8→5.9 s
(3.7×)** — the task now genuinely separates controllers. Smoke test scenario
spread went from ~2% to 8.4–16.4 s.

---

## Cross-reference table (also in AUDIT_FINDINGS.md)

| ID | Defect | Conclusion it broke | Status |
|----|--------|---------------------|--------|
| F1 | dt = physics timestep | all settling/IAE 10× small | fixed, reran |
| F2 | disturbance-timing claim | "dynamic eval tests nothing" | analysis rewritten |
| F3 | fake 10-seed variance | n=1 as n=10 | protocol v2 |
| F4 | wrong-target no-reset | "naive PID 0%" | fixed, reran |
| F5 | missing anti-windup | "PID fails → RL" | baseline added |
| F6 | (1,1) context at eval | "context +14.5%" | protocol v2, retrained |
| F7 | infeasible MRAC + dt | strawman failure | feasible rerun (still 0%) |
| F8 | friction inert | friction as a real axis | documented L1 |
| F9 | mass-only non-discriminative | controllers indistinguishable | actuator axis added |
