# Chapter 3 — Figures To Generate

## Figure 3.1 — Control Loop Block Diagram
Two-loop closed-loop architecture.

**Must show:**
- Outer loop (RL, 50 Hz / 0.02 s per step)
- Inner loop (PID, 500 Hz / 0.002 s per step, runs 10× per RL step)
- RL agent inputs: stacked observation vector (10 frames × 6 or 8 dims)
- RL agent outputs: normalized action [a_kp, a_ki, a_kd] ∈ [-1, 1]³
- Gain mapping box: kp = base_kp + delta_kp × a_kp (etc.)
- PID block: receives (kp, ki, kd) + position error → motor command u(t)
- MuJoCo plant block: receives u(t) → outputs position x, velocity ẋ
- Observation assembly box: [x, ẋ, error, a_kp, a_ki, a_kd, mass_scale*, friction_scale*]
  (*Stage 5a only — show as dashed/optional path)
- Frame stack buffer: 10 frames concatenated before policy input

**Tool suggestion:** draw.io, Inkscape, or TikZ. Keep it clean — two concentric loops with labeled arrows.

---

## Figure 3.2 — Simulation Environment Diagram
Top-down or side view of the vehicle task.

**Must show:**
- Vehicle starting at x = 0
- Target position at x = 5.0 m (static eval) / x ∈ [1, 10] m (training)
- Braking zone: |error| < 2.0 m shaded region near target
  - Label: "brake_integral_reset fires here"
- Friction patch: x ∈ [1.5, 2.4] m, friction = 35% of nominal
  - Label: "low-friction corridor"
- Arrows indicating vehicle motion direction

**Tool suggestion:** Simple diagram in PowerPoint/Inkscape. Not a screenshot — a schematic.

---

## Figure 3.3 — Curriculum Learning Schedule
Bar or timeline diagram showing target distance ramp.

**Must show:**
- x-axis: training timesteps (0 → 1,000,000)
- y-axis: target distance range (m)
- 4 phases:
  - Phase 1: 0–250K steps, target 1–3 m
  - Phase 2: 250K–500K, target 1–5 m
  - Phase 3: 500K–750K, target 1–7 m
  - Phase 4: 750K–1000K, target 1–10 m
- Shade or bracket each phase

**Tool suggestion:** matplotlib bar chart or simple timeline in draw.io.

---

## Figure 3.4 — Policy Network Architecture (optional but useful)
MLP with frame stack input.

**Must show:**
- Input: 80-dim vector (Stage 5a: 8 dims × 10 frames) or 60-dim (Stage 5b: 6 × 10)
- MLP layers (check agents/model.py for exact layer sizes)
- Actor head: outputs mean of 3-dim Gaussian → tanh squash → action
- Critic head: outputs scalar value estimate
- Note: shared trunk or separate actor/critic (check model.py)

**Tool suggestion:** Neural network diagram tool (NN-SVG, draw.io).

---

## Reference in chapter text
Figures are referenced as:
- Figure 3.1 in §3.1 (Simulation Environment) and §3.3 (Policy Architecture)
- Figure 3.2 in §3.1 (Simulation Environment)
- Figure 3.3 in §3.4 (Curriculum Learning)
- Figure 3.4 in §3.3 (Policy Architecture) — if included
