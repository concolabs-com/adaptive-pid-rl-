# Claude Code Instructions — Thesis Project

## What this project is

Master's thesis: "Reinforcement Learning for Adaptive PID Gain Scheduling Under Unknown Vehicle Dynamics"
Domain-randomised PPO agent that schedules PID gains in real time for a simulated wheeled vehicle with unknown mass and friction.

## Read at session start

1. **`progress-tracker.md`** — what is done, in progress, and next up. Read this first every session.
2. **`CONTEXT.md`** — canonical glossary. Use these exact terms. Challenge any deviation.
3. **`THESIS_WRITING_GUIDE.md`** — ground truth for all numbers, chapter structure, narrative framing, exact tables. Never use numbers from memory.
4. **`FINDINGS_SUMMARY.md`** — all results. Never misstate findings.

## Key rules (never break these)

- Do NOT say "RL outperforms fixed PID." Fixed PID is faster on settling time because `brake_integral_reset` aids it — this is a confound, not a fair comparison.
- Do NOT use "meta-RL" label. Approach is domain-randomised PPO with frame stacking.
- Agent names: Stage 5a = Context-Aware Agent, Stage 5b = Blind Agent, Fixed PID = Classical Baseline.
- All exact numbers live in `THESIS_WRITING_GUIDE.md`. Use them verbatim in thesis prose.
- Hold bonus = **+80** on success (not +50; +50 is terminal_hold_bonus at truncation only).
- Observation dims 3–5 = **normalised gain action** [a_Kp, a_Ki, a_Kd] ∈ [-1,1], not absolute gains.
- Control loop: RL at **50 Hz** (frame_skip=10, dt=0.02s), PID at **500 Hz** (physics dt=0.002s).

## Source code locations

- Training script: `D:\Projects\MuJoCo\thesis_writing\scripts\stage2_meta_rl_reproduction.py`
- Environment: `D:\Projects\MuJoCo\thesis_writing\environment\envs\adaptive_suspension.py`
- Agent model: `D:\Projects\MuJoCo\thesis_writing\environment\agents\model.py`
- Vehicle XML: `D:\Msc\Master_Thesis\envs\assets\car_model.xml`

## Chapter files

| Chapter | File | Status |
|---------|------|--------|
| 1 — Introduction | `chapter01/chapter1_introduction.md` | Done |
| 2 — Background | `chapter02/chapter2_background.md` | Done |
| 3 — Methodology | `chapter03/chapter3_methodology.md` | Done |
| 4 — Experimental Setup | `chapter04/chapter4_experimental_setup.md` | Done |
| 5 — Results | `chapter05/chapter5_results.md` | Done |
| 6 — Discussion | `chapter06/chapter6_discussion.md` | Done |
| 7 — Conclusion | `chapter07/chapter7_conclusion.md` | Done |

## Figures needed

`chapter03/FIGURES_NEEDED.md` — 4 diagrams for Chapter 3 not yet generated.

## Citation placeholders (open)

- Ch2 §2.5: two `CITATION_NEEDED` for model-based and model-free online RL-PID work
- Ch2 `[^shi2020]`: verify venue and page numbers
- Ch3: MuJoCo, PPO (Schulman et al. 2017), orthogonal initialisation
