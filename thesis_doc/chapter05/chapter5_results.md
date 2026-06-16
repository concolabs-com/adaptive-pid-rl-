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
`thesis_v6_hipmdp` protocol across all five seeds. Episode return rises
steeply during the first curriculum phase (targets 1–3 m), where the agent
learns the basic approach-and-hold behaviour on short, forgiving targets, then
steps down and recovers at each subsequent phase boundary (250k, 500k, 750k
steps) as the target range expands to 1–5, 1–7, and 1–10 m. The dips are the
signature of curriculum transfer: a policy tuned for short approaches must
re-learn the braking timing for longer ones, where more integral accumulates
and the actuator-limited cruise is longer. By the end of phase four the return
has re-stabilized, and — more importantly for a control task — the *success
rate* on held-out evaluation targets has saturated at 100%.

Three observations about the training dynamics are worth recording. First,
across the five seeds the learning curves are qualitatively identical, with the
phase-boundary dips occurring at the same steps and the final return spread
narrow; the policy is not unusually sensitive to initialization. Second, the
MLP agents complete a 1M-step run in ~15 min on CPU, whereas the recurrent
(GRU) agent takes ~75 min — a 5× cost from backpropagation through time over
the rollout, which is the practical price of recurrence even when it helps
(§5.4.2). Third, and as a methodological caution, the GRU's per-episode *return*
remains deeply negative throughout training (the long episodes accumulate many
small per-step distance and velocity penalties before the terminal hold bonus
is collected), and a naïve reading of the return curve alone would suggest it
had failed to learn. Its *behaviour*, however, converges to reliable control —
100% success, low settling. This dissociation between shaped return and task
success is the reason this thesis reports success rate and settling time, not
episode return, as the primary metrics, and it is the same trap that an earlier
recurrent run under a different reward (the Stage-4 governor/cliff protocol)
fell into when its exploding value loss was read as a fundamental failure of
recurrence rather than an artifact of that protocol's hard-termination
landscape (§5.4.2).

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

**Statistical treatment.** Each cell of Table 5.1 aggregates 50 episodes (10
per seed × 5 seeds) under protocol v2; the reported ± is the standard deviation
of the five per-seed means, and the headline penalty is the ratio of the
seed-level means. Significance is assessed at the seed level — the unit at
which the policies are independent — with Welch's t-test (unequal variances,
$n_s=5$ per group) cross-checked by Mann–Whitney U where the settling
distribution is right-skewed; effect size is Cliff's δ, and the family of eight
scenario comparisons is corrected with Holm–Bonferroni. On the fast scenarios
(Light Strong Motor, Light and Grippy) the teacher–student difference survives
correction with a small-to-medium effect ($|\delta|\approx0.3$–$0.5$); on the
actuator-limited scenarios (OOD Weak Motor, OOD Heavy Weak) the difference is
within seed noise ($|\delta|<0.15$, adjusted $p>0.1$) and no advantage is
claimed there. With only five seeds the seed-level bootstrap confidence
intervals are wide, and the chapter is careful to read the result as a
*trend* — context helps, conditionally — rather than a precise effect size.

This result **replaces** the original draft's "+14.5% on every scenario", which
was an artifact: the v1 static evaluation never wrote the sampled physics to
the context observation, so the teacher always saw a constant $(1,1)$ context —
roughly 11× outside its training context distribution (the training context,
normalized by the XML-nominal mass of ≈0.43 kg, spans mass-scale ≈ 11.6–46.5).
Re-evaluating the old teacher with the *true* context applied (§5.7) makes its
advantage collapse to near the blind agent's level, while feeding it the broken
$(1,1)$ context reproduces the original inflated numbers — a clean
demonstration that the "advantage" was an out-of-distribution observation
artifact, not a benefit of context. That the dynamic evaluation (which always
pushed context correctly through the randomization wrapper) was *not* affected
is what localized the bug to the static-evaluation path.

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

**Probe methodology.** A controller that merely *tolerates* unknown dynamics
(a single robust policy) and one that *identifies* them (conditioning on an
inferred parameter) can be behaviourally similar yet are mechanistically
distinct. To separate them, the blind agent is rolled out over 80 episodes with
the hidden parameters sampled per episode, and at every control step two feature
vectors are recorded: the flattened stacked observation (the policy *input*) and
the activations of the policy network's penultimate layer (what the policy
*encodes*). A linear ridge probe is then trained to regress each true hidden
parameter from each feature vector. Crucially, cross-validation uses
**GroupKFold grouped by episode**, so no step from a training episode appears in
the corresponding test fold — a parameter is constant within an episode, and
without episode-grouping a probe could "decode" it by memorizing episode
identity rather than reading dynamics. The probe is deliberately *linear*: a
high $R^2$ then means the parameter is encoded in a directly readable
(linearly separable) form, not merely recoverable by an arbitrarily powerful
decoder. Scores are reported as out-of-fold $R^2$ within episode-time buckets,
giving decodability as a function of phase.

To test whether the blind policy *represents* the hidden parameters, these
probes were scored by $R^2$ as a function of episode phase. The result is a
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

The mechanism behind the three rows of Table 5.4 is exactly the windup analysis
of §2.2, and tracing it makes the comparison concrete. The naive controller
accumulates an integral of order $K_i\bar e T \approx 17$ over the 8 m approach;
when it reaches the target the integral term alone saturates the actuator and
the derivative term ($K_d|\dot e|\approx 0.25$) cannot arrest the vehicle, so it
overshoots by ~7 m, the error reverses, the integrator unwinds, and the loop
finally settles tens of seconds later — when it settles at all (the Heavy &
Slippery case drives backward past the origin into the runaway-termination
region and is scored a failure). Back-calculation removes this by bleeding the
integrator toward the saturation-consistent value the instant the command
clips: the integral never reaches a value the derivative term cannot counter,
so the controller decelerates cleanly with 0.13 m overshoot. The learned agents
sit between the two — they were trained with the task-aware reset always
firing, so they never had to learn windup-free gain scheduling; evaluated
without it they still reach and hold the target (their learned conservatism
helps) but with multi-metre transients that a purpose-built anti-windup loop
avoids. The honest reading is that windup is a *solved* problem for which a
learned controller is unnecessary, and the no-reset environment is therefore
the wrong place to look for an RL advantage — the right place is varying,
unidentified, or unstable dynamics (§5.10), not a fixed-parameter windup task.

### 5.6.3 MRAC fails fairly

MRAC with a *feasible* reference model ($\tau_m\in\{10,15\}$ s, achievable at
$v_{\max}=0.5$ m/s), corrected control period, and a normalized MIT rule still
achieves **0% success** across all configurations (overshoot 2.4–4.6 m,
timeout), and underperforms even naive fixed PID. The failure is instructive.
The MIT rule adapts each gain along $\dot K \propto -\,e_m\,\partial y/\partial
K$, which presumes the sign of the plant sensitivity $\partial y/\partial K$ is
known and stable so the update consistently reduces the model-following error
$e_m$. On this plant neither holds: the actuator saturates for most of the
approach, so over a wide operating band $\partial y/\partial K \approx 0$ (more
gain produces no more torque) and the gradient signal vanishes exactly when the
controller most needs to act; and once the command comes off saturation the
sensitivity flips sign through the overshoot, so the same update that helped
during approach now destabilizes the hold. The earlier strawman configuration
(infeasible $\tau_m$, the 5× `dt` error) guaranteed a large persistent $e_m$
that drove the gains to their bounds; fixing those removes the artifact but not
the underlying sign/saturation problem, and $\sigma$-modification only bounds
the resulting drift without restoring a usable gradient. This is now a
*defensible* negative result rather than a strawman: classical
model-reference adaptation does not solve the task even when configured
favourably, which is precisely the gap a model-free learned policy — which
estimates what to do from returns rather than from a presumed sensitivity sign
— is positioned to fill.

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
Heavy-Pole-Weak-Gear).

The structure of the result is worth dwelling on, because it sharpens what
"adaptation matters" means. On the four benign scenarios — nominal, light pole,
heavy pole, weak gear taken singly — every controller, including fixed PID,
survives the full episode: a single well-chosen gain set is adequate when only
one parameter deviates moderately. The fixed baseline fails only when *two*
adverse deviations compound (a heavy pole that demands strong corrective torque
*and* a weak gear that cannot supply it, or an extreme gear deficit). These are
exactly the corners where a fixed compromise gain is wrong in both directions
at once — too weak for the heavy pole if tuned for the nominal, too aggressive
and oscillatory for the light pole if tuned for the heavy. The learned agents,
by selecting gains per episode, escape the compromise: the context agent reads
the pole mass and gear directly and commits to a matched gain set; the blind
agent infers them from the early sway of the pole. This is the same
"per-episode initial calibration" mechanism identified on the car (§5.8–5.9),
but here it is the difference between balancing and falling rather than between
settling in 11 s or 14 s — which is why the *magnitude* of the RL advantage is
so much larger on the unstable plant.

Two honest caveats. On the most extreme corner (OOD Very Weak Gear, 0.45)
*all* controllers struggle and the teacher/student ordering inverts within
noise (context 0.60 < blind 0.75) — at the edge of controllability, having the
true parameter does not help if no gain set can stabilize the plant with the
available torque, the angular analogue of the car's actuator-authority ceiling
(§5.8). And the pendulum study uses two seeds and a cascade outer loop specific
to balancing, so it is reported as a transfer *demonstration*, not a second
full quantitative study. With those caveats, it is the clearest evidence in the
thesis that learned gain scheduling adds real value over fixed gains when
adaptation genuinely matters, and that the contribution is not an artifact of
the wheeled-vehicle plant.

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
