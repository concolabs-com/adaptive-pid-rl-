# Chapter 6 — Discussion and Limitations

This chapter interprets the results collectively, frames the experimental audit
as a methodological contribution, states the limitations that bound the
conclusions, and considers practical implications.

## 6.1 Interpreting the Findings

**Feasibility is not the interesting part (RQ1).** That both agents reach 100%
success across a 4:1 mass range, a wide actuator range, and out-of-distribution
conditions establishes the approach works — but the self-stabilizing car makes
this a relatively low bar, as the universal success and the flat stack-depth
ablation both show. The substantive results are about *mechanism* and *fair
comparison*, not feasibility.

**Privileged context helps, modestly and conditionally (RQ2).** The teacher
settles 1–13% faster than the student, and the structure of that gap is
informative: it is largest when the plant is fast (settling control-limited)
and disappears when the plant is actuator-limited (settling travel-time-
limited). Knowing the dynamics only helps to the extent the actuator leaves
headroom to exploit the knowledge. This is a more honest and more interesting
claim than the original uniform "+14.5%", which the audit traced to a context-
observation bug. The practical reading: for a deployable controller that cannot
measure its dynamics, the cost of inferring them is small and shrinks exactly
in the regimes where the plant is hardest to move quickly.

**The blind agent identifies, it does not merely tolerate (RQ3, RQ4).** The
probing double dissociation — mass decodable during acceleration, actuator
strength during cruise — is direct evidence that the policy encodes the hidden
parameters its trajectory excites, in the phase each becomes observable. Yet
the stack-depth ablation shows it needs almost no explicit memory to do so,
because the instantaneous observation (velocity, error, previous gains) already
carries that information; recurrence helps only marginally and partly through
added capacity. The picture is consistent: identification here is easy and
largely instantaneous, which is *why* the blindness penalty (RQ2) is small.
The harder a parameter is to identify, the more we would expect context to
matter — a prediction the friction axis cannot test (it is inert) but a future
plant could.

**The learned controllers' real edge is adaptation, not control mechanics.**
Three results triangulate this. The fair anti-windup PID matches the learned
agents on the windup-prone task (§5.6), so the learned agents do not win on
windup. Mid-episode actuator shocks defeat every controller equally (§5.8), so
the learned agents do not win on transient authority. But on the unstable
pendulum, where fixed gains fail the hard corners outright, the learned agents
clearly win (§5.10). The distinctive value is *per-episode initial calibration
to unknown dynamics* — choosing, from the first interactions, gains suited to
the plant instance — which matters most precisely when a wrong fixed choice is
catastrophic (an unstable plant) rather than merely slow (the self-stabilizing
car).

## 6.2 The Audit as a Methodological Contribution

The first round of results told a clean, attractive story — RL settles fast,
beats a PID that fails without help, with a uniform context advantage — and
almost all of it was an artifact of measurement and environment design:

- a **unit error** (physics timestep vs control period) made all settling times
  appear 10× too fast, a figure no sanity check against the plant's terminal
  speed (~0.5 m/s, so ≥10 s for a 5 m drive) would have survived;
- a **context-plumbing bug** fed the teacher a constant, out-of-distribution
  context at evaluation, manufacturing its apparent advantage;
- a **wrong-target measurement** made naive PID look like total failure when it
  in fact recovers on most scenarios;
- a **task-aware environment aid** (`brake_integral_reset`) silently solved the
  hardest part of the control problem for *every* controller, masking the
  difference between adaptive and non-adaptive control; and
- a **non-discriminative randomization axis** (mass, inert at the actuator-
  limited cruise speed) left the task unable to separate any two controllers.

Each is mundane in isolation; together they would have supported a thesis whose
central claims were false. The remedy was not a single fix but a discipline:
re-derive every reported number from corrected code, replace the aid-dependent
comparison with a fair anti-windup baseline, add a randomization axis that
actually moves the dynamics, and design an evaluation protocol with genuine
per-episode variance instead of repeated deterministic rollouts. The
transferable lesson — that simulation aids, units, and observation plumbing
must be audited *before* adaptive controllers are benchmarked, because each can
independently invert the conclusion — is among the more useful outcomes of the
work, and is offered as a contribution rather than confined to an erratum.

## 6.3 Limitations

**L1 — Friction is dynamically inert.** Under rolling contact the sliding-
friction coefficient does not enter the translational dynamics; the probing
analysis confirms it is not identifiable. Only two of the three randomization
axes (mass, actuator) are real. A plant with viscous or slip-dependent friction
would make the third axis meaningful.

**L2 — Gain scheduling cannot rescue mid-episode authority loss.** The actuator
shock (§5.8) shows a hard boundary: when physical authority disappears
mid-manoeuvre, no gain choice recovers it. The contribution is bounded to
initial-condition adaptation.

**L3 — The learned schedule is shallow.** Hold-phase gains are nearly invariant
to the hidden parameter (§5.9); the agents learn a robust operating point more
than a steep schedule. This limits how much the "scheduling" framing should be
pressed, and an approach-phase analysis (noted as future refinement) is needed
to characterize what scheduling does occur.

**L4 — The GRU result confounds recurrence with capacity.** The recurrent agent
is fastest, but it has 128 hidden units against the MLP's 2×64; the comparison
does not isolate recurrence from width. A width-matched control is needed before
attributing the gain to memory architecture.

**L5 — Statistical power.** Five training seeds bound seed-level variance but do
not estimate it precisely; the bootstrap CIs are correspondingly wide, and the
ablations (stack depth, GRU eval) use one to two seeds and are reported as
trends, not tested hypotheses. The self-stabilizing car also produces ceiling
effects (universal 100% success) that limit the discriminating power of the
success metric, shifting weight onto settling time.

**L6 — Single plant family for the main study.** The pendulum transfer is
encouraging but uses a smaller seed count and a cascade outer loop specific to
balancing; broader cross-plant claims would need more plants and seeds.

**L7 — Simulation only.** No hardware; the sim-to-real gap is unaddressed by
design (§1.7). The integral-reset analysis is itself a reminder that simulation
conveniences can dominate conclusions.

## 6.4 Practical Implications

For a practitioner who cannot measure a system's dynamics, the deployable
result is the **blind agent**: it controls the plant across unknown conditions
at a small and shrinking cost relative to a context-aware controller, and it
provably infers the identifiable parameters online. But the honest comparison
tempers the enthusiasm. On a self-stabilizing plant where the main risk is
windup, a two-line anti-windup addition to an existing PID achieves comparable
reliability with none of the training cost, opacity, or sim-to-real risk of a
learned policy. The case for learned gain scheduling is strongest where a fixed
gain choice is not merely suboptimal but unsafe — unstable or near-unstable
plants, as the pendulum illustrates — and where the dynamics vary enough, and
are identifiable enough, that per-episode adaptation buys real margin. Matching
the method to that regime, rather than asserting a blanket advantage over PID,
is the practical takeaway.
