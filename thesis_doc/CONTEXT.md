---
name: thesis-context
description: Canonical glossary for the thesis — domain terms, agent names, key mechanisms
metadata:
  type: project
---

# Thesis Domain Glossary

## Agents

**Stage 5a — Context RL (Context-Aware Agent)**
PPO agent that observes position, velocity, error, current gains (Kp, Ki, Kd), AND measured mass/friction scales. Can see its own physics parameters directly.

**Stage 5b — Blind RL (Blind Agent)**
PPO agent that observes position, velocity, error, and current gains only. Must infer dynamics from 10-frame trajectory history. No mass/friction in observation.

**Fixed PID (Classical Baseline)**
Non-adaptive controller with fixed gains Kp=1.8, Ki=0.7, Kd=0.5. Action always [0,0,0]. No learning, no adaptation.

## Key Mechanisms

**Gain Scheduling (in this thesis)**
The RL agent outputs continuous adjustments to PID gains each step, not raw motor commands. The agent operates in "gain space" — PID computes the actual control signal.

**Domain Randomisation**
Mass sampled from [5, 20] kg and friction from [0.1, 2.0] at each episode reset. Forces policy to generalise across dynamics rather than overfit to a single operating condition.

**Brake Integral Reset (`brake_integral_reset`)**
Engineering mechanism in the simulation environment: zeros the PID integrator when the vehicle enters the braking zone (|error| < 2.0 m). Active for ALL agents including Fixed PID. Identified as a key experimental confound — without it, Fixed PID fails completely (0% success, ~10 m overshoot).

**Frame Stacking**
10 consecutive observations concatenated before input to policy network. Gives the policy a rolling temporal window (~0.2 s) to infer dynamics from trajectory shape. Used as implicit memory in place of recurrence (LSTM/GRU).

**Curriculum Learning**
Target distance increases over training: 1–3 m → 1–5 m → 1–7 m → 1–10 m across 1M steps. Agent learns settling behaviour on easy targets before encountering long-distance approaches where integral windup becomes significant.

## Environment Terms

**Braking Zone**
Region where |error| < 2.0 m from target. Where the brake_integral_reset fires and where deceleration rewards are applied.

**Mid-Episode Disturbance**
Mass multiplied by scale ∈ [0.9, 1.3] and friction by scale ∈ [0.5, 1.4] at a random step in [120, 220] (~1.2–2.2 s after episode start). Applied during dynamic evaluation.

**Position Friction Patch**
Friction reduced to 35% of nominal in region x ∈ [1.5, 2.4] m. Active every step during training.

**OOD Conditions**
Out-of-distribution: mass = 35 kg (75% above training ceiling of 20 kg), friction = 0.05. Neither seen during training. Both RL agents achieve 100% success under these conditions.

## Research Questions

**RQ1 (Feasibility):** Can an RL agent learn to schedule PID gains in real time, achieving reliable position control across unknown mass and friction conditions?

**RQ2 (Context Benefit):** Does providing explicit physics context (mass/friction scales in observation) improve adaptation compared to a blind agent inferring dynamics from trajectory history alone?

## Thesis Title (working)
"Reinforcement Learning for Adaptive PID Gain Scheduling Under Unknown Vehicle Dynamics"

## Key Numbers

- Context-aware vs blind settling time gap: **14–15% slower for blind**
- Static evaluation success: **100% for all agents**
- Fixed PID without integral reset: **0% success, ~10 m overshoot**
- OOD generalisation (35 kg): **100% success for both RL agents**
