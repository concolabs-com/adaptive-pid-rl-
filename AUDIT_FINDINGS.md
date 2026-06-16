# Experimental Audit Findings — June 2026

Consolidated record of every defect found in the original Stage 5 evaluation
pipeline, what was done about it, and what it means for the thesis. This file
is the ground-truth changelog feeding the thesis rewrite (Chapter 5/6 and the
methodological-contribution narrative).

## F1 — Settling-time dt error (10× too small)

`run_eval_episode` (and the fixed-PID baseline) computed settling time and IAE
with `dt = model.opt.timestep` (0.002 s) instead of the control period
`env.dt = timestep × frame_skip` (0.02 s). Positions are recorded once per
control step, so all published settling times and IAE were **10× too small**
(e.g. "1.254 s" = actually 12.54 s; sanity: terminal speed ≈ 0.5 m/s ⇒ a 5 m
drive cannot settle in 1.25 s).

Status: fixed in all root scripts; all evals rerun. Corrected static settling:
Fixed PID 11.16–12.54 s, Stage 5a 12.54–13.90 s, Stage 5b 14.46–15.68 s.
Relative orderings unchanged.

## F2 — Disturbance-timing claim was backwards

Old thesis claim (Result 5 / Limitation L3): "disturbances fire at 1.2–2.2 s,
after the agents settle at 1.3–1.6 s, so the dynamic evaluation tests
nothing." With correct units the disturbance window (steps 120–220 = 2.4–4.4 s)
fires **during the approach** (settling ≈ 12.5 s+). The dynamic evaluation is
valid, and disturbances DO have measurable impact: e.g. Stage 5a Light & Grippy
static 13.90 s → dynamic 19.7 ± 8.5 s. Both agents still 100% success.

Status: corrected numbers available in `benchmark_results/dtfix_*`. Result 5 /
L3 must be rewritten (improves the thesis — the robustness test is real).

## F3 — "10 evaluation seeds" were 10 identical episodes

Static eval: deterministic policy + fixed physics + no randomness ⇒ all 10
"seeds" produced byte-identical rows (n=1 presented as n=10).

Status: eval protocol v2 added — per-episode physics bands (±10%/±5%),
per-episode target jitter (±0.5 m), per-episode RNG seeded by episode seed.
Honest mean ± std across genuinely different episodes.

## F4 — No-reset baseline measured against the wrong target

The no-reset env uses target = 8 m (V5 preset) but metrics were computed
against the module constant 5 m. Published Table 5.4 ("0% success, ~10 m
overshoot, never recovers") is a measurement artifact. Corrected: naive PID
in the no-reset env actually **recovers and succeeds on 2/3 scenarios**
(Standard 100% @ 73.98 s, 6.92 m overshoot; Light & Grippy 100% @ 92.5 s;
Heavy & Slippery true failure — backward runaway).

Status: fixed (`stage_baseline_fixed_pid.py` now reads `env.target_pos`),
rerun. The qualitative windup story survives (catastrophic transient), the
"total failure" claim does not.

## F5 — PID + anti-windup solves the no-reset task (missing fair baseline)

Textbook anti-windup (back-calculation, Tt = 1 s; and conditional
integration) added to `utils/pid.py` and evaluated: **100% success on all
scenarios in the no-reset env**, settling 20.8–23.3 s, overshoot 0.12–0.15 m.
In the reset-on env anti-windup is masked by the env aid (identical numbers to
plain PID).

Implication: "fixed PID fails without the env aid, hence RL" is untenable. The
RL value claim must rest on adaptation across unknown/varying dynamics, not on
windup handling. The brake_integral_reset confound chapter becomes: env aid →
naive baseline catastrophic-but-recovering → 2 lines of classical engineering
close the gap → what remains for learning is the adaptation problem.

## F6 — Static eval fed the context agent garbage context (RQ2 artifact)

`prepare_eval_episode` (v1) wrote mass/friction into the MuJoCo model but
never pushed the context observation; `reset_model` re-zeroed it. So in ALL
static evaluations the context agent observed (mass_scale, friction_scale) =
(1.0, 1.0). Training context (normalized by the XML nominal mass 0.43 kg)
spanned mass_scale ≈ [11.6, 46.5] — the eval context was ~11× outside the
training distribution.

Mass-sweep experiment (`benchmark_results/mass_sweep/`): old Stage 5a with
TRUE context ≈ 14.6–14.8 s — i.e. **the same as the blind agent**; with broken
(1,1) context ≈ 12.4–13.6 s (the published numbers). The published "context
agent settles 14.5% faster than blind" RQ2 result is an artifact of the bug.
RQ2 must be re-answered with the Stage 6 retrains (protocol v2 pushes true
context; dynamic eval always had correct context via the wrapper).

## F7 — MRAC baseline was a strawman; fair rerun still fails

Original MRAC: `CONTROL_DT` hardcoded 0.1 s vs real 0.02 s (reference model
effectively 5× faster than designed: tau_m = 3 s → 0.6 s effective), and the
reference model was infeasible anyway (initial required speed = target/tau_m =
1.67 m/s > terminal speed 0.5 m/s).

`stage4b_mrac_feasible.py`: dt taken from env, feasible tau_m ∈ {10, 15} s
(initial ref speed ≤ 0.5 m/s), normalized MIT rule, velocity regressor, sigma
on/off grid. Result: **still 0% success on all configs** (overshoot 2.4–4.6 m,
timeout) — now a defensible negative result: MIT-rule MRAC fails on this
saturated nonlinear plant even under favourable configuration. Note MRAC also
underperforms naive fixed PID. (Possible refinement: MRAC + anti-windup PID.)

## F8 — Friction axis is inert (now starkly proven)

"OOD Ultra Slippery" (mass 20, fr 0.05) and "Heavy and Slippery" (mass 20,
fr 0.2) produce IDENTICAL static results to 3+ decimals. Rolling contact ⇒
slide friction does not bind at these operating points. Friction
randomization provided no signal.

## F9 — Task was non-discriminative; actuator axis fixes it

Mass sweep 5→50 kg: fixed PID settling varies only 11.06→12.30 s (~11%) over a
10× mass range; all controllers 100% everywhere. Root cause: settling is
dominated by the actuator-limited terminal speed v_max = τ_max/(b·r) ≈ 0.5 m/s
(independent of mass).

Fix: actuator-strength randomization axis (scales motor gain + joint force
range together, range 0.6–1.4; v_max scales linearly). Smoke test spread:
Light Strong Motor 8.4 s ↔ Heavy Weak Motor 16.4 s (±40%). New preset
`thesis_v6_hipmdp`; obs extended to 9 dims (actuator_scale as third context).

## Stage 6 (COMPLETE) — real results

Context (9-dim) + blind (6-dim) × 5 seeds on thesis_v6_hipmdp, eval protocol
v2. (One sleep-stall mid-run; resumed via `resume_stage6.sh`, AC sleep
disabled.) Cross-seed aggregates: `*/aggregate_all_seeds.csv`.

**RQ1 (feasibility):** both agents 100% success, all 8 scenarios, all 5 seeds —
including OOD actuator 0.45–0.55 and OOD mass 30–35 kg. Confirmed.

**RQ2 (context benefit), honest re-answer (settling s, mean over 5 seeds):**

| Scenario | Context 6a | Blind 6b | Blind penalty |
|----------|-----------|----------|---------------|
| Light Strong Motor | 10.90±0.68 | 12.29±0.52 | +12.8% |
| Light and Grippy | 14.14±0.94 | 15.77±0.85 | +11.5% |
| Standard | 13.72±0.68 | 14.57±0.68 | +6.2% |
| Heavy Weak Motor | 18.61±1.04 | 19.16±0.99 | +3.0% |
| OOD Weak Motor | 26.58±1.40 | 26.91±1.37 | +1.2% |

Context helps, but the gap shrinks as the plant becomes actuator-limited
(settling dominated by travel time, not control). Real seed variance now
present (sd 0.5–1.4 s) → statistically testable. This REPLACES the old
"+14.5% everywhere" claim (which was the F6 (1,1)-context bug artifact).

**Discriminativeness (actuator sweep, `benchmark_results/actuator_sweep_v6/`):**
Fixed PID settling 21.8 s (act 0.5) → 5.9 s (act 2.0), 3.7× spread — the
actuator axis genuinely separates controllers where mass (F9) did not. Both
RL agents 100% across 0.5–2.0 (OOD both ends). Context < blind throughout;
fixed PID fastest when strong-actuator fast settling is achievable (RL
conservatism costs there), all converge when actuator-limited.

**RQ4 (probing, `benchmark_results/probe_stage6b/`) — DOUBLE DISSOCIATION:**
Ridge probe (GroupKFold by episode) on the blind policy's penultimate
representation, R² by episode phase:
- **mass**: decodable during ACCELERATION (R² 0.37–0.39, steps 5–40), decays
  to ~0.05 in cruise — matches ẍ = F/m (mass observable only in transients).
- **actuator**: decodable during CRUISE (R² 0.58–0.65, steps 20–160), negative
  early — matches v_max ∝ κ (only revealed once cruising).
- **friction**: weak/spurious (~0.28) — caveat needed (rolling-contact
  non-identifiability; any signal is the friction-patch transient).
The blind agent's representation encodes exactly the parameters its trajectory
excites, in the phase where each becomes observable. Core implicit-sysID
result.

**Zero-shot no-reset (old stage5, `benchmark_results/zeroshot_noreset_5*`):**
old RL agents hold 100% in-distribution but ~4.2 m overshoot, 39–61 s — far
worse than classical anti-windup PID (F5). RL trained WITH the env aid did not
learn windup-free control; supports framing RL value as adaptation, not windup
handling.

## Phase C (COMPLETE) — RQ3 + pendulum transfer

Batch crashed once mid-run: stack k=5 hit 0xC0000142 (STATUS_DLL_INIT_FAILED,
transient Windows subprocess-spawn failure) and `set -e` aborted everything.
Recovered via `resume_phaseC.sh` (no set -e, retry-once, skip-if-done,
direct stage2 calls). All components then completed.

**RQ3a — stack-size ablation (blind, seed 7, k∈{1,3,5,10,20}):** settling is
FLAT across k (Standard 14.7/14.3/14.4/14.7/14.4 s), 100% success at every k
including k=1. Memory depth does not matter for this task. Reason: the
instantaneous obs already exposes the dynamics (actuator strength ∝ cruise
velocity; prev-gain feedback is in the obs), so frame stacking adds little.
Implication: the RQ2 context↔blind gap is about explicit-vs-state-readable
parameters, not memory. Caveat: k=1 still observes prev-action, so not
strictly Markov-blind.

**RQ3b — GRU recurrent blind (seeds 7, 21):** 100% success, 0 overshoot,
Standard settling ~11.3 s — FASTER than frame-stack blind (14.6 s) and even
the context MLP (13.7 s). GRU is the best blind variant. CONTRADICTS the
Stage 4 expectation: the earlier GRU failure (v3_safety, value-loss
explosions) was PROTOCOL-induced, not inherent to recurrence; under v6 it
trains fine. Deeply-negative training reward was a long-episode-penalty red
herring. Caveat: GRU 128 hidden vs MLP 2×64 — speed gain conflates recurrence
with capacity; do not claim "recurrence helps" without the capacity caveat.

**Pendulum transfer (`benchmark_results/pendulum/eval/`)** — strongest
RL-over-fixed-PID result in the thesis. Survival on hard dynamics corners:
- Heavy Pole + Weak Gear: Fixed 0.58, blind 0.95, context 1.00
- OOD Very Weak Gear: Fixed 0.46, blind 0.75, context 0.60
- all 5 other scenarios: 1.00 for every controller
Fixed gains fail the hard corners; RL gain scheduling rescues them. Advantage
larger than on the car (pendulum is unstable → adaptation matters more).
Demonstrates the framework transfers across plants. Pole mass [0.5,2.5] ×
gear [0.6,1.4] randomized; cascade outer loop + scheduled angle PID.

**SAC: deferred** — `stage2_sac_reproduction.py` lacks the v6/actuator/eval-v2
args; needs porting. Lower priority than the rest.

## Shock probe (`benchmark_results/stage6_shock_probe/`) — RL value is INITIAL adaptation, not shock rejection

Mid-approach (step 30–60) parameter step, 10 seeds:
- **mass ×2.5**: all 100% success; recovery Fixed 10.3 s < Context 11.1 s < Blind
  13.9 s. Mass shock is easy (integral-reset aid); fixed PID fastest. The
  context<blind 2.8 s gap = the blind re-inference cost (RQ2 signal).
- **actuator ×0.5**: ALL drop to 30% success, recovery ~48–50 s, identical
  across controllers. Halving motor authority mid-approach cannot be rescued
  by gain scheduling — the bottleneck is physical authority, not gain choice.

Conclusion: RL's value is per-episode INITIAL gain calibration (scenario table,
pendulum), NOT transient shock rejection. Clean scientific boundary to state.

## Gain analysis (`benchmark_results/gain_analysis/`) — robust operating point, not steep schedule

Steady-state (hold-phase) gains, mass 10:
- Context: Kp≈1.43, Ki≈0.57, Kd≈1.51. Blind: Kp≈1.86, Ki≈0.50, Kd≈1.17.
- Two DIFFERENT viable regimes (context low-Kp/high-Kd vs blind high-Kp/low-Kd);
  both 100%. Multiple gain settings solve the task.
- Hold-phase gains nearly FLAT vs actuator (Kd 1.521→1.498 over act 0.6→1.4):
  agents learn a robust operating point, not a steep parameter→gain schedule.
  CAVEAT: hold phase isn't where scheduling lives; approach-phase gain analysis
  is the refinement for the writing pass.
- Adaptation lag (blind→context-steady-state convergence) ~9.8 s at act 1.4;
  early gain-vector gap ~0.2 L2. Ill-posed since agents target different
  regimes — report the regime difference + early gap, not a single lag number.
- Ties the findings: gains weakly depend on inferred parameter → explains
  flat stack ablation (RQ3a) and un-rescued shocks. Context/memory shape
  initial calibration + approach transient, not a dramatic gain surface.

## ALL EXPERIMENTS COMPLETE. Remaining: writing overhaul
~100 pages, HiP-MDP/RMA reframe, RQ1–RQ4 chapters, every number from corrected
data, figures, citation pass. SAC still optional/deferred.
