# Progress Tracker

Update this file after every meaningful thesis writing change.

## Current Phase

- In Progress

## Current Goal

- Write all thesis chapters for "Reinforcement Learning for Adaptive PID Gain Scheduling Under Unknown Vehicle Dynamics"

## Completed

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

## In Progress

- Chapter 3 — System Design and Methodology

## Next Up

- Chapter 3 — System Design and Methodology
- Chapter 4 — Experimental Setup
- Chapter 5 — Results and Analysis
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
