# Claude Code Instructions — Thesis Project

## What this project is

Master's thesis: "Learning to Schedule PID Gains Under Hidden Dynamics
Parameters: A Teacher–Student Study of Implicit System Identification"
(working title; old title "RL for Adaptive PID Gain Scheduling Under Unknown
Vehicle Dynamics" still acceptable). A model-free PPO agent schedules PID gains
in real time for systems whose mass, friction, and actuator strength are hidden
and randomized — framed as a hidden-parameter MDP, studied as a context-aware
teacher vs a history-based blind student.

## ⚠️ Post-audit (June 2026): read these IN ORDER

1. **`../AUDIT_FINDINGS.md`** — the defect log + every corrected/new number
   with provenance. The dt bug, context bug, no-reset measurement bug, MRAC
   strawman, and non-discriminative task were all fixed. READ FIRST.
2. **`progress-tracker.md`** — status, what's done/next.
3. **`FINDINGS_SUMMARY.md` (v2)** — results narrative, RQ1–RQ4. Never misstate.
4. **`CONTEXT.md` (v2)** — canonical glossary, agent names, key numbers.
5. **`THESIS_WRITING_GUIDE.md`** — SUPERSEDED. Outline/formulas useful; numbers
   are the WRONG original draft. Do not quote its numbers.

## Key rules (never break)

- Numbers come from `AUDIT_FINDINGS.md` / `FINDINGS_SUMMARY.md` (v2), never
  from `THESIS_WRITING_GUIDE.md` and never from memory.
- Framing is **HiP-MDP / teacher–student / implicit system identification**,
  not "meta-RL". Domain-randomized PPO + frame stacking (or GRU) is the method.
- Agent names: **Stage 6a = Context-Aware Agent (teacher, 9-dim obs)**,
  **Stage 6b = Blind Agent (student, 6-dim obs)**, **Fixed PID** and
  **Anti-Windup PID** = classical baselines, **MRAC** = classical adaptive.
  (Stage 5a/5b are the superseded pre-audit runs.)
- Honest framing, not spin: RL's value is per-episode INITIAL gain calibration,
  NOT mid-episode shock rejection (shock probe) and NOT windup handling
  (anti-windup PID closes that gap). Context beats blind by +1–13%
  (regime-dependent), not a flat 14.5%. Fixed PID is fastest on settling but
  the comparison is mediated by the brake-integral-reset aid.
- Control loop: RL at 50 Hz (frame_skip=10, dt=0.02 s), PID at 500 Hz
  (physics dt=0.002 s). Settling/IAE use env.dt = 0.02 s (NOT model.opt.timestep).
- Three dynamics axes: mass [5,20] kg, friction [0.1,2.0] (inert — rolling
  contact), actuator strength [0.6,1.4] (the discriminative axis).

## Source code locations (current root: D:\Fable\Car_Thesis)

- Training/eval: `stage2_meta_rl_reproduction.py` (preset `thesis_v6_hipmdp`,
  `--eval-protocol v2`)
- Stage 6 runners: `stage6a_context_hipmdp.py`, `stage6b_blind_hipmdp.py`,
  `stage6c_stack_ablation.py`, `stage6d_gru_blind.py`
- Analyses: `probe_hidden_params.py`, `eval_mass_sweep.py` (--axis),
  `stage6_shock_probe.py`, `analyze_gain_trajectories.py`,
  `stage_baseline_antiwindup_pid.py`, `stage4b_mrac_feasible.py`
- Pendulum: `pendulum_experiment.py`, `envs/adaptive_pendulum.py`
- Env: `envs/adaptive_suspension.py` (9-dim obs), `agents/domain_randomization.py`
- Drafted new sections: `new_sections/` (HiP-MDP, anti-windup, stats, RMA)

## Chapter files (REWRITE IN PROGRESS — post-audit)

| Chapter | File | Status |
|---------|------|--------|
| 1 — Introduction | `chapter01/chapter1_introduction.md` | needs reframe |
| 2 — Background | `chapter02/chapter2_background.md` | + new_sections |
| 3 — Methodology | `chapter03/chapter3_methodology.md` | + actuator/anti-windup/pendulum |
| 4 — Experimental Setup | `chapter04/chapter4_experimental_setup.md` | + protocol v2/stats |
| 5 — Results | `chapter05/chapter5_results.md` | full renumber + RQ3/RQ4/pendulum/shock |
| 6 — Discussion | `chapter06/chapter6_discussion.md` | reframe around RQ1–4 |
| 7 — Conclusion | `chapter07/chapter7_conclusion.md` | reframe |

## Open: figures (4 Ch3 diagrams) + citation pass (see new_sections/related_work_rma.md list)
