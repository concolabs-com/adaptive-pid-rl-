# Summary of Findings (v2 — post-audit, June 2026)
## Learning to Schedule PID Gains Under Hidden Dynamics Parameters

> This supersedes the original FINDINGS_SUMMARY. Every number here comes from
> the corrected pipeline (dt fix) and the Stage 6 / HiP-MDP experiments. The
> full defect log and provenance is in `../AUDIT_FINDINGS.md`. Old numbers
> (settling ~1.1–1.6 s, "context 14.5% faster everywhere", "fixed PID 0% no
> reset") were artifacts and must not be reused.

---

## Objective

Study whether a model-free RL agent can schedule the gains of a classical PID
controller in real time for a system whose physical parameters (mass, ground
friction, actuator strength) are hidden and vary between — and within —
episodes. The problem is framed as a **hidden-parameter MDP (HiP-MDP)**, and
the central comparison is a **teacher–student** one: a context-aware agent
that observes the hidden parameters versus a blind agent that must infer them
from interaction history.

## Research questions

- **RQ1 (feasibility):** Can RL learn online PID gain scheduling that holds
  position across a wide range of unknown, randomized dynamics?
- **RQ2 (privileged context):** Does observing the hidden parameters
  (context-aware "teacher") improve control over inferring them from history
  (blind "student")?
- **RQ3 (memory):** How much temporal context does the blind agent need —
  frame-stack depth, and stacking vs recurrence?
- **RQ4 (representation):** Does the blind agent's policy actually *encode*
  the hidden parameters, and when in the trajectory do they become decodable?

## System

Two-loop control: an RL policy sets PID gain multipliers at 50 Hz; the PID
runs at 500 Hz. MuJoCo two-wheeled car (primary plant) and MuJoCo inverted
pendulum (transfer plant). Hidden parameters randomized per episode: mass
[5,20] kg, friction [0.1,2.0], actuator strength [0.6,1.4]. Mid-episode
disturbances and a position friction patch add within-episode variation.
Five training seeds (7,21,42,84,123). Evaluation protocol v2: per-episode
physics bands, target jitter, and a true context push (10 episodes × 8
scenarios per seed).

---

## Key findings

### F-RQ1 — Feasibility confirmed
Both context and blind agents reach **100% hold success on all 8 evaluation
scenarios across all 5 seeds**, including out-of-distribution actuator (0.45)
and mass (35 kg) conditions. Zero overshoot throughout. RL gain scheduling is
a viable controller for this HiP-MDP.

### F-RQ2 — Privileged context gives a modest, condition-dependent edge
Context-aware settles faster than blind, but the gap depends on the regime
(settling time, s, mean over 5 seeds):

| Scenario | Context | Blind | Blind penalty |
|----------|---------|-------|---------------|
| Light Strong Motor | 10.90 | 12.29 | +12.8% |
| Standard | 13.72 | 14.57 | +6.2% |
| Heavy Weak Motor | 18.61 | 19.16 | +3.0% |
| OOD Weak Motor | 26.58 | 26.91 | +1.2% |

The advantage is largest when the plant is fast (settling control-limited) and
vanishes when the plant is actuator-limited (settling travel-time-limited).
This **replaces** the original "+14.5% everywhere", which was an artifact of a
context bug (the static evaluation fed the context agent a constant (1,1)
context, ~11× outside its training distribution).

### F-RQ3a — Frame-stack depth barely matters
Blind agents trained with stack depth k ∈ {1,3,5,10,20} are statistically
indistinguishable (Standard settling 14.2–14.7 s, 100% success at every k,
including the near-memoryless k=1). The instantaneous observation already
exposes the identifiable dynamics (actuator strength ∝ cruise velocity; the
previous gains are in the observation), so temporal depth is not the load-
bearing mechanism.

### F-RQ3b — Recurrence (GRU) is the fastest blind variant
A GRU recurrent blind policy reaches 100% success with the **lowest settling
of any blind agent (~11.3 s vs 14.6 s for frame-stacking)** — faster even than
the context MLP. This contradicts an earlier expectation (a Stage-4 GRU under
a different protocol diverged); that failure was protocol-induced (speed
governor + hard-overshoot termination), not inherent to recurrence. Caveat:
the GRU has 128 hidden units vs the MLP's 2×64, so the gain conflates
recurrence with capacity.

### F-RQ4 — Double dissociation in the learned representation
Ridge probes (cross-validated, grouped by episode) decode the hidden
parameters from the blind policy's penultimate activations, with a clean
phase structure matching the plant physics:
- **Mass** is decodable during **acceleration** (R² 0.37–0.39, steps 5–40),
  decaying to ~0.05 in cruise — mass enters only through ẍ = F/m.
- **Actuator strength** is decodable during **cruise** (R² 0.58–0.65, steps
  20–160), and is *not* decodable early — it is revealed by terminal speed
  v_max ∝ κ once cruising.
- **Friction** is at best weakly/spuriously decodable (~0.28) — consistent
  with rolling-contact non-identifiability.
The blind agent encodes exactly the parameters its trajectory excites, in the
phase where each becomes physically observable.

### F-classical — The fair classical baseline closes the windup gap
- Fixed PID with the environment's `brake_integral_reset` aid: 100% success,
  settling 11.2–12.5 s — the strongest, but the aid is task-aware.
- Without the aid (no-reset env), **naive PID does not totally fail** (the
  original "0% / never recovers" was a wrong-target measurement bug): it
  recovers on 2/3 scenarios at 74–93 s with ~7 m overshoot transients.
- **PID + back-calculation anti-windup (textbook) solves the no-reset task:
  100% success, ~21 s, 0.13 m overshoot.** So "PID fails, therefore RL" is
  untenable; RL's value rests on adaptation across varying dynamics, not on
  windup handling.
- **MRAC** (feasible reference model, corrected dt, normalized MIT rule):
  still **0% success** on all configs — a defensible negative result for
  classical adaptive control on this saturated nonlinear plant.

### F-discriminative — Actuator strength is the axis that separates controllers
Sweeping actuator strength 0.5→2.0 moves fixed-PID settling 21.8→5.9 s (3.7×).
Mass, by contrast, moves it only ~11% over a 10× range — at the actuator-
limited cruise speed, mass barely affects settling. This is why the original
mass-only randomization was non-discriminative, and why the actuator axis was
added.

### F-shock — RL adapts initial conditions, not mid-episode authority loss
Mid-approach parameter step (10 seeds):
- Mass ×2.5: all 100%; recovery Fixed 10.3 s < Context 11.1 s < Blind 13.9 s
  (the context<blind gap is the re-inference cost).
- Actuator ×0.5: all collapse to 30% success, ~48–50 s, identically — gain
  scheduling cannot rescue a mid-episode loss of physical authority.
RL's benefit is per-episode initial gain calibration, not transient shock
rejection.

### F-gains — Robust operating point, not a steep schedule
Context and blind converge to two different but viable gain regimes (context:
lower Kp, higher Kd; blind: the reverse). Hold-phase gains are nearly
invariant to the hidden parameter, indicating the agents learn a robust
operating point more than a strongly parameter-varying schedule. (Approach-
phase gain scheduling is analyzed separately in Ch. 5.)

### F-transfer — The framework transfers to a second, unstable plant
On the inverted pendulum (randomized pole mass × actuator gear), fixed PID
fails the hard dynamics corners (Heavy-Pole-Weak-Gear survival 0.58, OOD
Weak-Gear 0.46) while RL rescues them (context 1.00 / 0.60, blind 0.95 / 0.75).
The RL advantage is larger here than on the car because the pendulum is
unstable, so adaptation matters more.

---

## Honest limitations (carried into Ch. 6)

1. **Friction axis is inert** (rolling contact) — only mass and actuator are
   real dynamics axes.
2. **Gain scheduling does not rescue mid-episode authority loss** — RL value
   is initial-condition adaptation.
3. **Hold-phase gains are nearly flat** — the learned "schedule" is closer to
   a robust fixed operating point.
4. **GRU speed/capacity confound** — recurrence vs width not isolated.
5. **CPU-only training, 5 seeds** — variance bounded, not precisely estimated;
   bootstrap CIs over seeds are correspondingly wide.
6. **Single eval seed for some ablations** (stack depth, GRU eval) — trends,
   not tested hypotheses at every point.

## What changed from the original thesis (provenance)
The dt bug (10× settling error), the (1,1)-context bug (RQ2 artifact), the
wrong-target no-reset measurement, the strawman MRAC, and the non-
discriminative mass-only task were all identified and corrected. The honest
re-analysis is treated as a methodological contribution, not hidden.
