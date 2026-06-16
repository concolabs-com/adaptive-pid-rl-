# 04 — Writing Overhaul and Design Decisions

What changed in each chapter, and the framing decisions behind the rewrite.

---

## The reframe (the single biggest decision)

The original framing — "RL for adaptive PID gain scheduling, does it beat PID"
— is a control-engineering question with a weak answer on this self-stabilizing
plant. For a **Data Science** thesis it was reframed as:

> **Implicit system identification under a hidden-parameter MDP, studied as a
> teacher–student (privileged-information) comparison.**

Theory backbone: **HiP-MDP** (Doshi-Velez & Konidaris 2016) + **RMA-style
teacher/student** (Kumar et al. 2021). The context agent = privileged teacher;
the blind agent = proprioceptive-history student (trained from scratch, not
distilled, to isolate the observation regime). This:
- gives a rigorous, math-heavy theory chapter (POMDP/HiP-MDP formalism,
  identifiability) for strict supervisors;
- turns "two trained agents" into a mechanism study (RQ3 memory, RQ4 probing);
- keeps the honest result (context advantage is small) *interesting* rather
  than disappointing.

"Meta-RL" was dropped as a label (it's domain-randomized PPO + frame stacking /
recurrence, not MAML/RL²); meta-learning is referenced loosely only.

---

## Research-question structure

Expanded from the original 2 (feasibility, context benefit) to **4**: added
RQ3 (memory: stack depth + recurrence) and RQ4 (representation probing). These
two are the Data-Science core and the source of the most novel results.

---

## Chapter-by-chapter

**Ch1 Introduction.** Rewritten around HiP-MDP/teacher-student; RQ1–4; 4
contributions (incl. the audit as a methodological contribution); §1.4 "a note
on scientific method" states the audit up front; scope = sim-only.

**Ch2 Background.** Restructured and expanded with drafted sections:
- 2.1 PID + tuning (SIMC/Ziegler-Nichols), fixed-gain limitation;
- 2.2 **integral windup + anti-windup** (back-calculation, conditional
  integration) — promoted to first-class because the fair baseline depends on it;
- 2.3 classical adaptive (gain scheduling, MRAC + MIT-rule instability);
- 2.4 **MDP → PPO/GAE → POMDP → HiP-MDP** formalism + identifiability;
- 2.5 domain randomization + **teacher–student/RMA/PEARL/RL²/UP-OSI**;
- 2.6 RL-for-PID gap statement.

**Ch3 Methodology.** Two-loop architecture (RL 50 Hz / PID 500 Hz); the
**terminal-speed derivation** v_max ∝ κ/b (explains why mass is inert and
actuator discriminates — own the discriminativeness problem as analysis); gain
parameterization; reward (with the dropped governor/cliff history); **three
randomization axes**; curriculum; the `brake_integral_reset` aid; classical
baselines in the same action space; **the pendulum transfer plant**.

**Ch4 Experimental Setup.** **Eval protocol v2** (bands, jitter, true context
push) vs the v1 artifact; 8-scenario suite incl. OOD; agents table;
**statistical methodology** (seed vs episode vs scenario; bootstrap CIs over 5
seeds; Welch + Mann-Whitney; Cliff's δ; Holm-Bonferroni; Wilson intervals);
compute + reproducibility + the two incidents noted honestly.

**Ch5 Results.** Fully renumbered (dt fix). Organized by RQ: 5.1 training, 5.2
RQ1, 5.3 RQ2 (+ actuator sweep + the (1,1)-bug exposure), 5.4 RQ3 (stack flat +
GRU fastest), 5.5 RQ4 (double dissociation), 5.6 classical baselines + confound
(+ the wrong-target correction + MRAC-fails-fairly), 5.7 discriminativeness,
5.8 shock boundary, 5.9 gain regimes, 5.10 pendulum, 5.11 summary. Provenance
note at the top.

**Ch6 Discussion.** Interprets RQ1–4; **§6.2 the audit as a methodological
contribution** (the five defects, the discipline that fixed them); honest
limitations L1–L7 (friction inert, shocks not rescued, flat schedule, GRU
capacity confound, statistical power/ceiling, single plant family, sim-only);
practical implications (match the method to unstable/varying plants; anti-windup
PID suffices for windup).

**Ch7 Conclusion.** Verdict per RQ; future work (width-matched memory ablation;
a plant where identification is *hard*; approach-phase scheduling; distillation
vs from-scratch; more plants/seeds; honest sim-to-sim before hardware); closing
remark on *when* learned scheduling matters.

**Appendix.** Full hyperparameters (PPO, env, reward rationale, anti-windup,
MRAC, pendulum); per-seed artifact map; code map; **audit cross-reference**.

---

## Ground-truth documents

- `FINDINGS_SUMMARY.md` → rewritten **v2** (RQ1–4, all corrected numbers,
  provenance, limitations).
- `CONTEXT.md` → **v2** glossary (HiP-MDP terms, Stage 6 agent names, key
  numbers, plants, title).
- `CLAUDE.md` (thesis_doc) → rewritten to point at v2 sources, new agent names,
  honest-framing rules, current source paths.
- `THESIS_WRITING_GUIDE.md` → **superseded banner** added (outline/formulas
  still useful; numbers are the wrong original draft).
- `references.md` → consolidated bibliography (control, RL, HiP-MDP/RMA,
  MuJoCo); 3 online-RL-PID entries flagged ⚠ for an author literature search.

## Figures

4 Chapter-3 conceptual diagrams generated by `thesis_doc/scripts/make_diagrams.py`
→ `thesis_doc/chapter03/figures/fig3_1..3_4.png` (control loop, environment,
curriculum, network). Result figures already produced by the experiment scripts
(actuator sweep, probe-R²-over-time, gain-vs-actuator, shock trajectories, mass
sweep).

---

## Honesty rules adopted (do not break)

- Never claim "RL outperforms fixed PID" flatly — fixed PID is fastest *because*
  of the brake-integral-reset aid; anti-windup PID matches RL on windup.
- Context vs blind is **+1–13%, regime-dependent** — never "+14.5% everywhere".
- RL's value is **initial per-episode adaptation**, not shock rejection, not
  windup handling — demonstrated by the shock probe and the pendulum.
- Friction is inert; say so. The learned schedule is shallow (robust operating
  point); say so. GRU speed confounds capacity; say so.
- The audit is a contribution, not an erratum to hide.

---

## Length status (the open item driving Task 20)

The rewrite is correct and complete in content/structure but **dense**:
~11k words ≈ 31 body pages, not the ~100-page target. Reaching ~100 pages
requires a deliberate **expansion pass** (real content, not padding): fuller
derivations (TRPO→PPO, GAE, plant ODE), per-figure walk-throughs in Ch5,
broader Ch2 literature, front matter (abstract, TOC, lists, nomenclature), and
per-scenario discussion. See `05_CURRENT_STATUS_AND_NEXT_STEPS.md`.
