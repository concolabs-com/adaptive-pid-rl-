---
name: thesis-context
description: Canonical glossary for the thesis — domain terms, agent names, key mechanisms, key numbers (v2, post-audit)
metadata:
  type: project
---

# Thesis Domain Glossary (v2 — post-audit)

> Use these exact terms. Challenge any deviation. All numbers come from the
> corrected pipeline; see `FINDINGS_SUMMARY.md` and `../AUDIT_FINDINGS.md`.

## Framing

**Hidden-Parameter MDP (HiP-MDP)** — the formal object: a family of MDPs
indexed by a latent parameter vector ψ = (mass, friction, actuator strength),
drawn per episode. The blind agent faces the induced POMDP.

**Teacher–student / privileged information** — the context-aware agent
(teacher, observes ψ) vs the blind agent (student, infers ψ from history).
Trained from scratch independently (not distilled); isolates the observation
regime. RMA (Kumar et al. 2021) is the reference paradigm.

**Implicit system identification** — the blind agent inferring ψ from
closed-loop trajectory; measured directly by the probing analysis (RQ4).

## Agents

**Context-Aware Agent (Stage 6a)** — PPO, 9-dim observation including
mass/friction/actuator scales, frame stack 10. The privileged teacher.

**Blind Agent (Stage 6b)** — PPO, 6-dim observation (no context), frame stack
10. Must infer dynamics from history. The student. Primary deployable result.

**GRU Blind Agent (Stage 6d)** — recurrent (128 hidden) blind variant; fastest
blind agent (RQ3b).

**Fixed PID (Classical Baseline)** — constant gains Kp=1.8, Ki=0.7, Kd=0.5,
action [0,0,0].

**Anti-Windup PID** — fixed gains + textbook back-calculation (Tt=1 s) or
conditional integration. The *fair* classical baseline (solves the no-reset
task).

**MRAC** — MIT-rule model-reference adaptive control; feasible reference model
(τ_m ≥ 10 s). Fails (0%) — classical adaptive negative result.

## Key mechanisms

**Gain scheduling (here)** — RL outputs continuous PID gain multipliers each
step; PID computes the control signal. K = clip(K_base + ΔK · a), a ∈ [-1,1].

**Domain randomization** — mass [5,20] kg, friction [0.1,2.0], actuator
strength [0.6,1.4], sampled per episode.

**Actuator-strength axis** — scales motor gain + joint force limit together;
terminal speed v_max ∝ κ. The discriminative dynamics axis (mass is not).

**Brake integral reset** — env aid zeroing the integrator in the braking zone
(|e|<2 m). Task-aware; equivalent to a crude conditional-integration anti-
windup. Active for all agents; analyzed as a confound.

**Frame stacking** — 10 stacked observations as finite-memory belief
approximation. Depth barely matters here (RQ3a).

**Eval protocol v2** — per-episode physics bands (mass ±10%, friction ±10%,
actuator ±5%), target jitter ±0.5 m, true context push; 10 episodes × 8
scenarios × 5 seeds. Replaces the v1 deterministic-repeat protocol.

## Plants

**Car** — two-wheeled MuJoCo vehicle, drive to 5 m and hold ±0.05 m for 25
steps. Primary plant.

**Inverted pendulum** — MuJoCo cart-pole balance; randomized pole mass ×
actuator gear; cascade outer loop shapes the angle setpoint, scheduled angle
PID inner loop. Transfer plant.

## Research questions

- **RQ1** feasibility · **RQ2** privileged context vs inference ·
  **RQ3** memory (stack depth + recurrence) · **RQ4** representation/probing.

## Key numbers (corrected)

- Settling (Standard): Fixed PID 11.2 s, Context 13.7 s, Blind 14.6 s, GRU 11.3 s.
- All agents 100% success, 8 scenarios, 5 seeds, zero overshoot.
- Context vs blind: +1.2% to +12.8% (regime-dependent), not a flat 14.5%.
- Actuator sweep: fixed-PID settling 21.8 s (κ=0.5) → 5.9 s (κ=2.0), 3.7×.
- Anti-windup PID no-reset: 100%, ~21 s. Naive PID no-reset: 2/3 recover, 74–93 s.
- MRAC: 0% (all feasible configs).
- Probing R²: mass 0.37–0.39 (acceleration), actuator 0.58–0.65 (cruise),
  friction ~0.28 (spurious).
- Shock: mass ×2.5 → 100% (recovery 10–14 s); actuator ×0.5 → 30% (all agents).
- Pendulum Heavy-Pole-Weak-Gear survival: Fixed 0.58, blind 0.95, context 1.00.

## Title (working)
"Learning to Schedule PID Gains Under Hidden Dynamics Parameters: A
Teacher–Student Study of Implicit System Identification" (confirm with
supervisor; old title "RL for Adaptive PID Gain Scheduling Under Unknown
Vehicle Dynamics" still acceptable).
