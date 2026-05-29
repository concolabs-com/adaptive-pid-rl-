# Progress Tracker

Update this file after every meaningful thesis writing change.

## Current Phase

- In Progress

## Current Goal

- Write all thesis chapters for "Reinforcement Learning for Adaptive PID Gain Scheduling Under Unknown Vehicle Dynamics"

## Completed

- **Chapter 6 — Discussion and Limitations** (`chapter06/chapter6_discussion.md`)
  - ~1,950 words, 3 sections
  - 6.1 Main Findings Interpreted: synthesis frame (no data repetition) — RQ1 verdict, RQ2 verdict, confound framed as methodological contribution
  - 6.2 Limitations: tiered treatment — L3/L5/L6 full paragraphs, L1/L4 brief, L2 dropped; L6 uses assertive + single qualifier for no-reset convergence cause
  - 6.3 Practical Implications: Stage 5b as deployable result, 14–15% penalty discussion, sim-to-real caveat, frame stacking reach paragraph, bridge to Ch7
  - **Design decisions:** Option A synthesis framing, dual confound framing (contribution in 6.1, limit in 6.2), tiered limitations, narrow+reach 6.3, assertive L6 tone

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

- **Chapter 7 — Conclusion and Future Work** (`chapter07/chapter7_conclusion.md`)
  - ~1,050 words, 2 sections
  - 7.1 Conclusions: verdict-first opening, RQ1/RQ2 with inline bold labels (2–3 sentences + key number each), third paragraph for confound as Contribution 3
  - 7.2 Future Work: 5 items (bold names), tiered — GRU/LSTM + cross-system transfer as Tier 1 (full justification), multi-seed/disturbance window/friction decoupling as Tier 2 (one sentence each)
  - **Design decisions:** verdict-first opening (no motivation recap), inline bold RQ labels not headers, GRU/LSTM as pure future work (partial code in repo is dev artefact not thesis experiment), cross-system transfer included from Ch1 §1.7

## Current Phase

- Complete — all chapters written. Thesis draft complete.

## TODO — Citations

| # | Chapter | Placeholder | Action needed |
|---|---------|-------------|---------------|
| C1 | Ch1 §1.1 | `[^astrom1995]` | Add full reference entry: Åström & Hägglund (1995), *PID Controllers: Theory, Design, and Tuning*, 2nd ed., ISA. Verify page/chapter for the 95% stat. |
| C2 | Ch2 §2.5 | `CITATION_NEEDED` (×2) | Find and add citations for (a) model-based online RL-PID work and (b) model-free online RL-PID work. These plug the gap statement. |
| C3 | Ch2 §2.5 | `[^shi2020]` | Verify venue (journal/conference name) and page numbers. Currently unconfirmed. |
| C4 | Ch3 §3.2 | MuJoCo citation | Add Todorov et al. (2012) MuJoCo reference. |
| C5 | Ch3 §3.4 | PPO citation | Add Schulman et al. (2017), *Proximal Policy Optimization Algorithms*, arXiv:1707.06347. |
| C6 | Ch3 §3.4 | Orthogonal initialisation | Find and add citation for orthogonal weight initialisation (Saxe et al. 2013 or similar). |

## TODO — Figures

| # | Figure | Chapter/Section | Description | Tool |
|---|--------|-----------------|-------------|------|
| F1 | Fig 3.1 | Ch3 §3.1 + §3.3 | Two-loop control architecture block diagram (RL 50 Hz outer / PID 500 Hz inner, gain mapping, frame stack buffer, optional Stage 5a dashed path) | draw.io / TikZ |
| F2 | Fig 3.2 | Ch3 §3.1 | Environment schematic: vehicle at x=0, target, braking zone (|error|<2m), friction patch (x=1.5–2.4m) | PowerPoint / Inkscape |
| F3 | Fig 3.3 | Ch3 §3.4 | Curriculum schedule: 4-phase bar/timeline, x-axis = timesteps 0–1M, y-axis = target distance, phases at 0/250K/500K/750K/1M | matplotlib / draw.io |
| F4 | Fig 3.4 | Ch3 §3.3 | Policy network architecture: 80-dim (5a) / 60-dim (5b) input → 2×64 Tanh MLP → actor head (3-dim Gaussian + tanh) + critic head (scalar). Check `model.py` for exact sizes. | NN-SVG / draw.io |

> Full specs for all figures: `chapter03/FIGURES_NEEDED.md`

## TODO — Content Gaps

| # | Chapter | Item | Action |
|---|---------|------|--------|
| G1 | Ch2 §2.2 | MRAC failure elaboration | Add brief sentence elaborating the MRAC 0% success failure (currently only a one-sentence note with forward-ref §6.2). Mentioned in Session Notes. |

## TODO — Supervisor Confirmation

| # | Item | Status |
|---|------|--------|
| S1 | Thesis title | Working title confirmed in writing, but needs supervisor sign-off |
| S2 | Contributions format | Currently numbered 3-item list in Ch1 §1.5 — confirm format is acceptable |
| S3 | Citation style | IEEE / APA / Harvard — confirm with university requirements before finalising reference list |

## TODO — Final Pass

| # | Task |
|---|------|
| P1 | Proofreading pass — full thesis read-through for consistency, flow, and typos |
| P2 | Check all cross-references (§X.Y forward/back-refs) still point to correct sections |
| P3 | Verify all numbers in text match `THESIS_WRITING_GUIDE.md` ground truth |
| P4 | Confirm citation style and reformat all references accordingly (after S3 resolved) |

## In Progress

- Nothing currently active

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
