# Chapter 5 — Results and Analysis

This chapter reports the corrected results, organized by research question.
Section 5.1 covers training. Section 5.2 answers RQ1 (feasibility). Section 5.3
answers RQ2 (privileged context) with the scenario suite and the actuator
sweep. Section 5.4 answers RQ3 (memory: stack depth and recurrence).
Section 5.5 answers RQ4 (representation probing). Section 5.6 reports the
classical baselines and the integral-reset confound. Section 5.7 analyses task
discriminativeness. Section 5.8 reports the mid-approach shock probe.
Section 5.9 examines the learned gain regimes. Section 5.10 reports the
pendulum transfer. All settling times use $\Delta t = 0.02$ s; all learned-
agent numbers are over five seeds and evaluation protocol v2 unless noted.

> **Provenance.** The numbers here supersede an earlier draft that contained a
> 10× settling-time unit error, a context-observation bug, a wrong-target
> baseline measurement, a strawman MRAC configuration, and a non-discriminative
> mass-only task. Section 5.6 and Chapter 6 discuss the corrections; the full
> log is in `AUDIT_FINDINGS.md`.

---

## 5.1 Training

Both learned agents train stably to convergence over 1M steps on the
`thesis_v6_hipmdp` protocol across all five seeds. Episode return improves
through the four curriculum phases, with the expected transient dips at phase
boundaries as the target range expands. The MLP agents complete a run in
~15 min on CPU; the recurrent (GRU) agent takes ~75 min owing to
backpropagation through time. Notably, the GRU's per-episode return remains
deeply negative throughout (accumulated per-step penalties over long episodes)
yet its *behaviour* converges to reliable control — a reminder that raw return
under heavy shaping is a poor proxy for task success, and the reason success
and settling, not return, are the reported metrics.

## 5.2 RQ1 — Feasibility

Both the Context-Aware Agent and the Blind Agent achieve **100% hold success on
all eight static scenarios across all five seeds, with zero overshoot** — and
the same under the dynamic (mid-episode disturbance) protocol. Success is
retained on the out-of-distribution scenarios: OOD Ultra Heavy (35 kg, 1.75×
the training ceiling), OOD Weak Motor (actuator 0.45, below the training floor
of 0.6), and OOD Heavy Weak (combined). RL gain scheduling is therefore a
feasible controller for this HiP-MDP, generalizing beyond the training support
on each axis. The interesting question is no longer *whether* the agents
control the plant but *how well* and *by what mechanism* — RQ2–RQ4.

## 5.3 RQ2 — Privileged Context vs Inference

### 5.3.1 Scenario suite

Table 5.1 gives settling time (mean over five seeds, protocol v2) for the
teacher and student. Both succeed everywhere; the teacher is faster, by a
**regime-dependent** margin.

| Scenario | Context (s) | Blind (s) | Blind penalty |
|----------|-------------|-----------|---------------|
| Light Strong Motor | 10.90 ± 0.68 | 12.29 ± 0.52 | +12.8% |
| Light and Grippy | 14.14 ± 0.94 | 15.77 ± 0.85 | +11.5% |
| Standard | 13.72 ± 0.68 | 14.57 ± 0.68 | +6.2% |
| Heavy and Slippery | 14.13 ± 0.73 | 14.84 ± 0.69 | +5.0% |
| OOD Ultra Heavy | 14.38 ± 0.73 | 15.29 ± 0.68 | +6.3% |
| Heavy Weak Motor | 18.61 ± 1.04 | 19.16 ± 0.99 | +3.0% |
| OOD Heavy Weak | 23.35 ± 1.25 | 23.79 ± 1.19 | +1.9% |
| OOD Weak Motor | 26.58 ± 1.40 | 26.91 ± 1.37 | +1.2% |

*Table 5.1 — Settling time, teacher vs student (5 seeds, protocol v2).*

The advantage is **largest when the plant is fast** (Light Strong Motor,
+12.8%) and **vanishes when the plant is actuator-limited** (OOD Weak Motor,
+1.2%). The interpretation is that when settling is control-limited, knowing
the dynamics lets the teacher commit to good gains immediately, whereas the
student spends part of the episode inferring them; but when settling is
travel-time-limited (weak actuator, long cruise), neither knowing nor inferring
the dynamics can speed up a manoeuvre the actuator itself bottlenecks, so the
gap closes. With genuine per-seed variance (sd 0.5–1.4 s) the ordering is
consistent across seeds; on the fast scenarios the seed-level Welch test
separates the two (small-to-medium Cliff's δ), while on the actuator-limited
scenarios the difference is within noise.

This result **replaces** the original draft's "+14.5% on every scenario", which
was an artifact: the v1 static evaluation never wrote the sampled physics to
the context observation, so the teacher always saw a constant $(1,1)$ context —
roughly 11× outside its training context distribution. Re-evaluating the old
teacher with the *true* context applied (§5.7) makes its advantage collapse to
near the blind agent's level, confirming the artifact.

### 5.3.2 Actuator sweep

Sweeping actuator strength at nominal mass (Figure 5.x) makes the mechanism
explicit. Teacher and student both succeed across $\kappa\in[0.5,2.0]$
(including OOD ends), and the teacher leads throughout — by ~1.4 s at
$\kappa=2.0$ (8.8 vs 10.1 s) narrowing toward the weak-actuator end. The
advantage tracks how much settling headroom the actuator allows.

## 5.4 RQ3 — How Much Memory Does Inference Need?

### 5.4.1 Frame-stack depth is nearly irrelevant

Blind agents trained with stack depth $k\in\{1,3,5,10,20\}$ are
indistinguishable (Table 5.2): 100% success at every $k$, and settling within
~0.5 s across the whole range on every scenario, with no monotonic trend.

| $k$ | 1 | 3 | 5 | 10 | 20 |
|-----|----|----|----|----|----|
| Standard settling (s) | 14.73 | 14.25 | 14.39 | 14.71 | 14.35 |
| Heavy Weak Motor (s) | 19.37 | 18.84 | 19.04 | 19.28 | 18.97 |

*Table 5.2 — Stack-depth ablation (blind, seed 7).*

Even $k=1$ — a near-memoryless policy — succeeds with settling matching $k=20$.
The explanation follows from §2.4.4: the instantaneous observation already
exposes the identifiable dynamics. Actuator strength is readable from the cruise
velocity in a single frame, and the previous gain action is part of the
observation, giving a one-step feedback channel. Temporal depth is therefore
*not* the load-bearing mechanism for this task. (Caveat: because the previous
action is in the observation, $k=1$ is not strictly Markov-blind; a fully
memoryless variant would drop it.)

### 5.4.2 Recurrence is the fastest variant

A GRU recurrent blind agent (128 hidden units, stack 1) reaches 100% success
with **the lowest settling of any blind agent — ~11.3 s on Standard, against
14.6 s for frame-stacking and 13.7 s for the context MLP** (Table 5.3, two
seeds). That recurrence helps while stack depth does not suggests the GRU's
learned state compression is more useful than raw stacked frames; however, the
GRU has more capacity (128 hidden vs 2×64), so the comparison conflates
recurrence with width and the result is stated with that caveat. This also
*overturns* an earlier expectation: a recurrent agent under a different protocol
(speed governor + hard-overshoot termination) had diverged with exploding value
loss; under the present reward it trains without difficulty, locating that
earlier failure in the protocol rather than in recurrence per se.

| Variant | Standard settling (s) | Success |
|---------|----------------------|---------|
| Frame-stack (k=10) | 14.6 | 100% |
| GRU (recurrent) | 11.3 | 100% |
| Context MLP (reference) | 13.7 | 100% |

*Table 5.3 — Memory mechanism comparison (blind).*

## 5.5 RQ4 — What the Blind Agent Encodes

To test whether the blind policy *represents* the hidden parameters, ridge
probes were trained to predict each parameter from the policy's penultimate
activations, cross-validated with grouping by episode (no within-episode
leakage), and scored by $R^2$ as a function of episode phase. The result is a
clean **double dissociation** matching the plant physics (Figure 5.x):

| Parameter | Decodable phase | Peak $R^2$ | In cruise |
|-----------|-----------------|------------|-----------|
| Mass | acceleration (steps 5–40) | 0.37–0.39 | ~0.05 |
| Actuator strength | cruise (steps 20–160) | 0.58–0.65 | sustained |
| Friction | — | ~0.28 (spurious) | — |

Mass is decodable precisely when it influences the trajectory — during
acceleration, through $\ddot x = F/m$ — and fades once cruising. Actuator
strength is the mirror image: not decodable early, strongly decodable in cruise,
because it sets the terminal speed $v_{\max}\propto\kappa$. Friction is at best
weakly and likely spuriously decodable, consistent with its dynamical inertness
under rolling contact. The blind agent thus encodes exactly the parameters its
closed-loop trajectory excites, in the phase where each becomes physically
observable — direct evidence that the agent performs implicit system
identification rather than merely executing a robust fixed policy.

## 5.6 Classical Baselines and the Integral-Reset Confound

### 5.6.1 The aid equalizes naive PID

With the `brake_integral_reset` aid active, Fixed PID achieves 100% success and
the *fastest* settling of any controller (11.2–12.5 s, Table 5.4) — faster than
both learned agents. But the aid removes integral windup using task knowledge
(distance to target), the hardest part of the control problem (§2.2). Its
presence makes the headline "fixed PID is fastest" a statement about the aid,
not about non-adaptive control.

### 5.6.2 Without the aid: the fair baseline

Disabling the aid (no-reset environment, target 8 m) separates the controllers:

| Controller (no-reset) | Success | Settling (s) | Overshoot (m) |
|-----------------------|---------|--------------|---------------|
| Fixed PID (naive) | 2/3 scenarios | 74–93 | ~7 |
| Anti-Windup PID (back-calc) | **100%** | ~21 | 0.13 |
| Context / Blind (zero-shot) | 100% in-dist | 39–61 | ~4.2 |

*Table 5.4 — No-reset environment (no task-aware aid).*

Naive PID does **not** "fail completely" (the original draft's "0% / never
recovers" was a measurement against the wrong target distance): it suffers large
windup transients but recovers on two of three scenarios at 74–93 s. The
textbook **anti-windup PID solves the no-reset task outright** — 100% success,
~21 s, negligible overshoot — using two lines of standard engineering. The
learned agents, evaluated zero-shot in this environment (trained with the aid),
hold 100% in-distribution but with large ~4.2 m overshoot transients, having
relied on the aid during training. The lesson is decisive for framing: **"PID
fails, therefore RL" is untenable** — the fair classical baseline closes the
windup gap — so the learned controllers' value must rest on adaptation across
varying dynamics, not on windup handling.

### 5.6.3 MRAC fails fairly

MRAC with a *feasible* reference model ($\tau_m\in\{10,15\}$ s, achievable at
$v_{\max}=0.5$ m/s), corrected control period, and a normalized MIT rule still
achieves **0% success** across all configurations (overshoot 2.4–4.6 m,
timeout), and underperforms even naive fixed PID. The MIT rule's sensitivity
assumption breaks on this saturated, nonlinear plant; $\sigma$-modification
bounds drift but does not restore convergence. This is now a *defensible*
negative result rather than a strawman: classical model-reference adaptation
does not solve the task even when configured favourably.

## 5.7 Task Discriminativeness: Why the Actuator Axis Matters

Sweeping mass 5→50 kg moves fixed-PID settling only ~11% (11.06→12.30 s) — at
the actuator-limited cruise speed, mass barely affects a multi-metre drive.
Sweeping actuator strength 0.5→2.0 moves it **3.7×** (21.8→5.9 s). The original
mass-only randomization therefore left the task nearly non-discriminative
(every controller looked identical), which is why the actuator axis was added.
The sweep also exposes the v1 context bug: re-running the old context agent with
the broken $(1,1)$ context reproduces the original fast numbers, while applying
the *true* context yields settling indistinguishable from the blind agent —
confirming the artifact diagnosed in §5.3.1.

## 5.8 Mid-Approach Shock: The Boundary of Adaptation

Firing a large parameter step at step 30–60 (mid-approach) probes transient
adaptation (Table 5.5, 10 seeds).

| Shock | Fixed | Context | Blind |
|-------|-------|---------|-------|
| Mass ×2.5 | 100%, 10.3 s | 100%, 11.1 s | 100%, 13.9 s |
| Actuator ×0.5 | 30%, 48.5 s | 30%, 48.4 s | 30%, 50.0 s |

*Table 5.5 — Recovery from mid-approach shock (success, recovery time).*

Two findings. A **mass** shock is handled by all controllers; the
context-before-blind ordering (11.1 vs 13.9 s) is the student's re-inference
cost, visible because the parameter changed mid-episode. An **actuator** shock
— halving motor authority mid-approach — collapses every controller to 30%
success, identically: **gain scheduling cannot rescue a mid-episode loss of
physical authority**, because no choice of gains compensates for missing torque.
The learned agents' value is thus per-episode *initial* adaptation, not
transient shock rejection — a clean and honest scientific boundary.

## 5.9 Learned Gain Regimes

Examining the scheduled gains, the teacher and student converge to two
different but viable regimes: the teacher runs lower $K_p$ (~1.43) and higher
$K_d$ (~1.5), the student the reverse ($K_p$ ~1.86, $K_d$ ~1.17) — both at 100%
success, showing the task admits a family of solutions. Hold-phase gains are
nearly invariant to the hidden parameter (e.g. teacher $K_d$ moves 1.521→1.498
across actuator 0.6→1.4), indicating the agents learn a robust operating point
more than a steep parameter→gain schedule. This coheres with §5.4 (depth
doesn't matter) and §5.8 (shocks aren't rescued by gain changes): the lever the
agents pull is initial calibration and approach shaping, not a dramatic
gain surface. (Approach-phase gains carry more of the scheduling than the hold
phase analysed here; this is noted as a refinement.)

## 5.10 Transfer to an Unstable Plant: Inverted Pendulum

On the inverted-pendulum balance task with randomized pole mass and actuator
gear, the framework transfers and the RL advantage is *larger* than on the car,
because the plant is unstable (Table 5.6, survival fraction).

| Scenario | Fixed PID | Blind | Context |
|----------|-----------|-------|---------|
| Nominal / Light / Heavy / Weak-Gear | 1.00 | 1.00 | 1.00 |
| Heavy Pole + Weak Gear | **0.58** | 0.95 | **1.00** |
| OOD Very Weak Gear | **0.46** | 0.75 | 0.60 |

*Table 5.6 — Pendulum balance survival.*

Fixed gains fail the hard dynamics corners (Heavy-Pole-Weak-Gear 0.58, OOD
Weak-Gear 0.46); the learned agents rescue them (context 1.00 / blind 0.95 on
Heavy-Pole-Weak-Gear). On the most extreme OOD corner all controllers struggle
and the teacher/student ordering inverts within noise. This is the clearest
demonstration that learned gain scheduling adds value over fixed gains when
adaptation genuinely matters — and that the contribution is not specific to the
wheeled-vehicle plant.

## 5.11 Summary

RL gain scheduling is feasible and generalizes (RQ1); privileged context gives
a modest, regime-dependent edge that the audit shows was previously overstated
(RQ2); memory depth is nearly irrelevant though recurrence helps (RQ3); the
blind agent provably encodes mass during acceleration and actuator strength
during cruise (RQ4). Against fair classical baselines the learned agents do not
win on windup handling (anti-windup PID matches them) but do win on adaptation
to unknown and especially unstable dynamics (pendulum). Their value is initial-
condition adaptation, not transient shock rejection. Chapter 6 interprets these
findings and the audit that produced them.
