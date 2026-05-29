# Progress Tracker

Update this file after every meaningful thesis writing change.

## Current Phase

- In Progress

## Current Goal

- Write all thesis chapters for "Reinforcement Learning for Adaptive PID Gain Scheduling Under Unknown Vehicle Dynamics"

## Completed

- **Chapter 4 — Experimental Setup** (`chapter04/chapter4_experimental_setup.md`)
  - ~1,500 words, 5 sections
  - 4.1 Evaluation Protocol: 25-step hold, ±0.05 m, 5 metrics, honest seed asymmetry (static = deterministic, dynamic = seeds vary disturbance)
  - 4.2 Evaluation Scenarios: 3 static + 5 dynamic (incl. OOD at ~2× mass ceiling), rolling-contact caveat referenced
  - 4.3 Agents Evaluated: concise comparison table, cross-refs Ch3 for architecture details
  - 4.4 Reproducibility: artefact paths, single-seed limitation flagged
  - 4.5 Hardware and Compute: RTX 3070 Laptop, i7-11800H, 32 GB, Python 3.10, ~45 min/agent
  - **Bug fixed**: `dt` in settling-time computation was `model.opt.timestep` (0.002 s) instead of `env.dt` (0.02 s) — all reported settling times were 10× too small. Fixed in `stage_baseline_fixed_pid.py` and `stage2_meta_rl_reproduction.py`. **Must rerun all evaluations before writing Ch5.**

- **Chapter 1 — Introduction** (`chapter1_introduction.md`)
  - ~1,800 words, 7 sections
  - Covers: PID motivation, classical adaptive control failure (incl. MRAC failure), RL gain-scheduling framing, two research questions (RQ1: feasibility, RQ2: context benefit), 3 numbered contributions, key findings preview, chapter outline
  - Citation placeholder `[^astrom1995]` added by user — needs full reference entry
- **CONTEXT.md** — canonical thesis glossary (agent names, mechanisms, key numbers)
- **Chapter 2 — Background and Related Work** (`chapter02/chapter2_background.md`)
  - ~3,200 words, 5 sections
  - Arc: gap-building — PID limitation → classical adaptive control requires model → RL model-free → domain randomisation for generalisation → prior RL-PID work leaves gap this thesis fills
  - 2.1 PID Control: standard formulation, Ziegler–Nichols, SIMC, IMC, fixed-gain limitation
  - 2.2 Adaptive Control: gain scheduling, MRAC (one-sentence failure note, forward-ref §6.2), model requirement as shared gap
  - 2.3 RL for Control: MDP formulation, policy gradient theorem, PPO clipped surrogate + GAE with exact hyperparameters
  - 2.4 Domain Randomisation and Implicit Adaptation: domain randomisation (Tobin et al.), frame stacking as implicit temporal memory — no "meta-RL" overclaim
  - 2.5 RL for PID Gain Scheduling: offline vs online tuning taxonomy, model-dependent vs model-free gap, gap statement targeting feasibility + context ablation
  - **Decision: Section 2.6 (Integral Windup) moved to Chapter 5 §5.4** — belongs next to confound analysis, not in background
  - Citations needing verification: `[^shi2020]` venue/pages; two `CITATION_NEEDED` placeholders for model-based and model-free online RL-PID work

- **Chapter 3 — System Design and Methodology** (`chapter03/chapter3_methodology.md`)
  - ~3,400 words, 7 sections
  - 3.1 Design Overview: iterative protocol history (governor → cliff protocol), cliff penalty tried and dropped, three supporting changes (decel bonus, progress cutoff, near-target init)
  - 3.2 Simulation Environment: MuJoCo platform, task, two-loop architecture (RL 50 Hz / PID 500 Hz, frame_skip=10), gain parameterisation formula, core reward (+80 hold bonus corrected), shaping terms in prose
  - 3.3 Domain Randomisation: episode randomisation, mid-episode disturbance, friction patch, rolling-contact friction caveat
  - 3.4 Policy Architecture: PPO with clipped surrogate + GAE equations, observation table (dims 3–5 = normalised action with mapping sentence), frame stacking (0.2 s window), separate 2×64 Tanh actor/critic, hyperparameter table
  - 3.5 Curriculum Learning: 4-phase table, integral windup motivation
  - 3.6 Classical Baseline: fixed-gain [0,0,0] action, same two-loop architecture
  - 3.7 Brake Integral Reset: mechanism description, forward-ref §5.4
  - `chapter03/FIGURES_NEEDED.md` created — 4 diagrams needed: control loop block diagram (Fig 3.1), environment diagram (Fig 3.2), curriculum schedule (Fig 3.3), network architecture (Fig 3.4)
  - Citations needed: MuJoCo, PPO (Schulman et al. 2017), orthogonal initialisation

## In Progress

- Nothing currently active

## Next Up

- Chapter 6 — Discussion and Limitations
- Chapter 7 — Conclusion and Future Work

## Open Questions

- Thesis title: working title is "Reinforcement Learning for Adaptive PID Gain Scheduling Under Unknown Vehicle Dynamics" — confirm with supervisor
- Contributions bullet list format — confirm with supervisor (currently numbered 3-item list)
- Citation style — confirm with university requirements

## Architecture Decisions

- **Drop "Meta-RL" label** — approach is domain-randomised PPO + frame stacking, not MAML/RL². Refer to as "domain-randomised RL with temporal context." Can reference meta-learning loosely in §1.3 and §2.4. Reason: committee will challenge loose usage; accuracy preferred.
- **Opening hook: industrial/robotics angle** — PID ubiquity → fixed-gain failure under unknown dynamics → RL as model-free alternative. Reason: matches actual system, gives committee concrete mental model.
- **Research question: two-part** — RQ1 (feasibility) + RQ2 (context benefit). Both answered cleanly by results. Reason: thesis actually resolves both.
- **Integral reset confound: featured explicitly in Chapter 1** — not hidden, presented as methodological contribution. Reason: scientific integrity; committees respect honest framing.
- **Contributions: numbered bullet list** — 3 contributions listed explicitly. Reason: common in CS/ML/engineering theses; makes examiner evaluation easier.

## Session Notes

- All exact numbers in `THESIS_WRITING_GUIDE.md` — use as ground truth, never from memory
- Key framing rule: Fixed PID beats RL on settling speed ONLY because `brake_integral_reset` aids it. Never say "RL outperforms fixed PID."
- MRAC failure (0% success) is referenced in Chapter 1 §1.2 as empirical motivation — needs brief elaboration in Chapter 2
- Chapter 2 deep dives: PID equations, RL taxonomy, PPO algorithm details, domain randomisation papers all live here — not in Chapter 1
