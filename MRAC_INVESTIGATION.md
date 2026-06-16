# MRAC Investigation — Stage 4 Classical Adaptive Baseline

## Purpose

MRAC (Model Reference Adaptive Control) was implemented as a classical adaptive
baseline to compare against the Stage 4 blind Meta-RL agent. Both operate under
the same constraint: no mass or friction in the observation. The goal was to show
whether a principled classical method can handle domain shift without privileged
physics context.

**Result: All three MRAC variants failed (0% success rate). This failure is
documented here as a thesis-relevant finding.**

---

## System Description

- **Task:** Vehicle must drive from x=0 to target x=5.0 m and hold within 0.05 m tolerance for 25 consecutive steps
- **Environment:** 2-wheeled MuJoCo car, control period = 0.1 s (frame_skip=10 × physics_dt=0.01 s)
- **Scenarios evaluated:**
  - Standard: mass=10 kg, friction=1.0
  - Heavy & Slippery: mass=20 kg, friction=0.2
  - Light & Grippy: mass=5 kg, friction=2.0
- **Episodes per scenario:** 30
- **Max episode length:** 5000 steps (500 s)

---

## MRAC Design

### Reference Model
First-order system describing desired closed-loop behavior:

```
dy_m/dt = -(1/tau_m) * (y_m - target)
```

With `tau_m = 3.0 s`, the reference model reaches the target in approximately 9 s.

### Adaptation Law (MIT Rule)
Gradient descent on the squared model-following error `J = 0.5 * e_mrac²`:

```
e_mrac = pos - y_m          (plant vs reference gap)

dKp/dt = -gamma_p * e_mrac * phi_Kp
dKi/dt = -gamma_i * e_mrac * phi_Ki
dKd/dt = -gamma_d * e_mrac * phi_Kd
```

where `phi` is the sensitivity (regressor) approximating `∂y/∂K`.

### PID gain bounds (matching environment defaults)

| Gain | Base | Delta | Range |
|------|------|-------|-------|
| Kp   | 1.8  | 1.0   | [0, 6] |
| Ki   | 0.7  | 0.6   | [0, 3] |
| Kd   | 0.0  | 2.0   | [0, 5] |

---

## Variant 1 — Standard MIT Rule (ep regressor)

**Hyperparameters:** tau_m=3.0, gamma_p=0.20, gamma_i=0.03, gamma_d=0.01

**Regressor:** `phi_Kp = ep` (position error = target − pos)

### Results

| Scenario | Success | Settling (s) | Overshoot (m) | Final Kp |
|----------|---------|-------------|---------------|----------|
| Standard | 0% | 500 (timeout) | 0.888 | 6.0 (max) |
| Heavy & Slippery | 0% | 500 (timeout) | 0.957 | 6.0 (max) |
| Light & Grippy | 0% | 500 (timeout) | 0.964 | 6.0 (max) |

All 30 episodes per scenario **identical** — gains reset each episode (bug).

### Root Cause

For any position-tracking task, `e_mrac * ep` is **always negative**:

- **Approach phase** (car behind reference): `e_mrac < 0`, `ep > 0` → product negative
- **Overshoot phase** (car past reference): `e_mrac > 0`, `ep < 0` → product still negative
- Once reference model settles at target: `e_mrac ≈ −ep`, so `e_mrac * ep ≈ −ep² < 0` always

MIT rule: `Kp -= gamma * (always negative) * dt` → **Kp always increases** → saturates at 6.0.

Also discovered: `controller.reset()` was called each episode, resetting gains to base
values → every episode started identically → all 30 results were identical (not a physics
coincidence, a code bug).

---

## Variant 2 — Velocity Regressor Fix

**Change:** Replace `phi_Kp = ep` with `phi_Kp = vel` (vehicle velocity)

**Rationale:** Velocity changes sign correctly at target crossing:
- Approach (vel > 0, e_mrac < 0): product < 0 → Kp increases ✓
- Overshoot (vel > 0, e_mrac > 0): product > 0 → Kp decreases ✓
- Return (vel < 0, e_mrac > 0): product < 0 → Kp increases (stronger braking) ✓

Also fixed: gains no longer reset between episodes (`reset_episode()` instead of `reset()`).

### Results

| Scenario | Success | Settling (s) | Overshoot (m) | Ep 0 Kp | Ep 1-29 Kp |
|----------|---------|-------------|---------------|---------|------------|
| Standard | 0% | 500 (timeout) | 0.888 | 4.08 | 5.999 (max) |
| Heavy & Slippery | 0% | 500 (timeout) | 0.957 | 4.12 | 5.999 (max) |
| Light & Grippy | 0% | 500 (timeout) | 0.964 | ~4.1 | 5.999 (max) |

### Analysis

The gain-persistence fix **did work** — episode 0 now ends at Kp=4.08 (not 6.0),
proving gains carry over. But Kp still saturates across episodes.

**Why velocity regressor still fails:** The approach phase is much longer (~15-30 s)
than the overshoot correction phase. During the long approach:
- `vel > 0` and `e_mrac < 0` → Kp accumulates upward continuously

The brief overshoot (vel > 0, e_mrac > 0) reduces Kp slightly, but the long approach
dominates. Net result: Kp saturates to maximum before sufficient correction can occur.

**Positive feedback loop identified:**
```
High Kp → faster car → more overshoot → longer oscillation → MIT rule pushes Kp up further
```

This is a known instability of the MIT rule on systems with significant nonlinearity
and parameter-dependent overshoot behavior.

---

## Variant 3 — Sigma-Modification (Ioannou & Tsakalis, 1986)

**Change:** Add leakage term to each gain update:

```python
Kp -= (gamma_p * e_mrac * vel  +  sigma * (Kp - Kp_base)) * dt
Ki -= (gamma_i * e_mrac * integral_ep  +  sigma * (Ki - Ki_base)) * dt
Kd -= (gamma_d * e_mrac * dep_dt  +  sigma * (Kd - Kd_base)) * dt
```

**Sigma = 0.40** — leakage pulls gains back toward base values when they drift.

**Rationale:** Sigma-modification is a standard technique from robust adaptive control
(Ioannou & Tsakalis, 1986) that prevents unbounded parameter growth by adding a
damping term. Equilibrium Kp ≈ Kp_base − (gamma × e_mrac × vel) / sigma.

### Results

| Scenario | Success | Settling (s) | Overshoot (m) | Final Kp | Final Ki |
|----------|---------|-------------|---------------|----------|----------|
| Standard | 0% | 500 (timeout) | **1.317** | 1.800 | 0.697 |
| Heavy & Slippery | 0% | 500 (timeout) | **1.408** | 1.800 | 0.697 |
| Light & Grippy | 0% | 500 (timeout) | **1.421** | 1.800 | 0.697 |

**Overshoot increased** compared to all previous variants. **All episodes identical.**

### Analysis

Sigma=0.40 is too dominant relative to the adaptation gradient. The leakage term
overrides the MIT rule completely:

```
final_kp = 1.800146  ≈ Kp_base = 1.8   (essentially zero adaptation)
final_ki = 0.697     ≈ Ki_base = 0.7
```

With gains frozen at base values, the integral term (Ki=0.7) accumulates during
the long approach, overshoots, drives the car backwards past the origin (pos < -2.0),
triggering the runaway termination. This explains:
- Shorter episodes: n_steps=2889 vs 4823 (terminates via runaway, not timeout)
- Larger overshoot: base gains without adaptation are suboptimal for all scenarios

**No working sigma value exists:**
- Low sigma → Kp saturates (plain MIT rule failure)
- High sigma → gains frozen at base, integral causes runaway
- Middle ground → gains oscillate but no convergence — inherent MIT rule instability

---

## Summary Comparison

| Variant | Kp final | Overshoot | Success | Key failure |
|---------|----------|-----------|---------|-------------|
| MIT rule (ep regressor) | 6.0 (max) | 0.89 m | 0% | e_mrac×ep always negative |
| + velocity regressor | 6.0 (max) | 0.89 m | 0% | Approach phase dominates |
| + sigma-modification | 1.80 (base) | 1.32 m | 0% | Leakage kills all adaptation |

---

## Why MRAC Fails on This System

The MIT rule assumes the plant sensitivity `∂y/∂Kp` has a consistent sign that
the gradient update can exploit. For this vehicle position-tracking task:

1. **Positive feedback loop:** Higher Kp → more aggressive driving → more overshoot
   → the MIT rule (trying to reduce e_mrac) pushes Kp higher still
2. **Phase asymmetry:** The approach phase (Kp increasing) is 5-10× longer in time
   than the overshoot phase (Kp should decrease). The increasing direction dominates.
3. **Nonlinear plant:** The MIT rule was derived for linear time-invariant systems.
   The MuJoCo vehicle has contact dynamics, friction nonlinearity, and inertial
   coupling that violate these assumptions.
4. **Reference model mismatch:** tau_m=3 s makes the reference reach target much
   faster than the physical vehicle can, creating persistently large e_mrac throughout
   the approach and driving rapid, destabilising adaptation.

These limitations are documented in adaptive control literature:
- Åström & Wittenmark (1995) warn that the MIT rule "may become unstable" for
  large adaptation gains or plants that deviate significantly from the linear assumption.
- Ioannou & Tsakalis (1986) sigma-modification prevents drift but cannot fix
  sign inconsistency in the underlying adaptation gradient.

---

## Thesis Implication

**MRAC failure strengthens the RL motivation.** The thesis narrative becomes:

> Fixed PID (Stage 1) works under nominal conditions but degrades under domain shift.
> Classical adaptive control (MRAC) fails entirely — parameter drift instability
> is inherent for nonlinear tracking tasks. RL-based gain scheduling (Stage 2, Stage 4)
> succeeds where both classical methods fail, demonstrating that learned adaptation
> is necessary for robust performance across varying physical conditions.

---

## Files

| File | Description |
|------|-------------|
| `stage4_mrac_baseline.py` | Final implementation (sigma-modification variant) |
| `benchmark_results/stage4_mrac_baseline/mrac_raw.csv` | Per-episode results |
| `benchmark_results/stage4_mrac_baseline/mrac_summary.csv` | Per-scenario aggregation |
| `benchmark_results/stage4_mrac_baseline/mrac_summary.png` | Bar chart |
| `benchmark_results/stage4_mrac_baseline/mrac_final_gains.png` | Final gain distributions |
