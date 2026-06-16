# PROJECT LOG — Overview & Index

**Project:** Master's thesis (Data Science) — *Learning to Schedule PID Gains
Under Hidden Dynamics Parameters: A Teacher–Student Study of Implicit System
Identification* (working title; original title "RL for Adaptive PID Gain
Scheduling Under Unknown Vehicle Dynamics").

**Repo:** `D:\Fable\Car_Thesis` → GitHub `concolabs-com/adaptive-pid-rl-`
**Author:** Sahan (MSc), supervisors described as strict (math-heavy expected).
**Log written:** June 2026, after a major audit-and-rewrite session.

---

## Why this log exists

A single long working session (1) audited the original thesis results, (2)
found and fixed five measurement/design defects that had inverted several
conclusions, (3) added a new hidden-parameter-MDP experiment suite (RQ1–RQ4 +
classical baselines + a second plant), and (4) rewrote all seven chapters.
This log preserves the full reasoning so none of it is lost if conversation
context is compacted. It is the human-readable companion to `AUDIT_FINDINGS.md`
(the terse defect/result log).

## What the thesis is (one paragraph)

A model-free PPO agent sets the gains of a classical PID controller in real
time for plants whose physical parameters (mass, friction, actuator strength)
are hidden and randomized per episode — a **hidden-parameter MDP (HiP-MDP)**.
The core comparison is **teacher vs student**: a context-aware agent that
observes the hidden parameters versus a blind agent that must infer them from
trajectory history (implicit system identification). Plants: a two-wheeled
MuJoCo car (primary) and an inverted pendulum (transfer). Classical baselines:
fixed PID, anti-windup PID, MRAC.

## The four research questions

- **RQ1 Feasibility** — can RL learn online gain scheduling across unknown,
  randomized dynamics (incl. OOD)?
- **RQ2 Privileged context** — does observing the parameters beat inferring
  them, and when?
- **RQ3 Memory** — how much history does inference need (stack depth, GRU)?
- **RQ4 Representation** — does the blind policy *encode* the parameters, and
  when in the trajectory?

## Headline outcomes (all post-audit, corrected)

- RQ1: both agents 100% success, 8 scenarios × 5 seeds, incl. OOD. ✔
- RQ2: context faster by **+1–13%** (regime-dependent), NOT a uniform +14.5%
  (that was a bug artifact).
- RQ3: frame-stack depth ≈ irrelevant (even k=1 works); **GRU fastest** blind
  variant.
- RQ4: **double dissociation** — mass decodable during acceleration, actuator
  during cruise, friction not at all.
- Classical: **anti-windup PID solves** the windup task (so "PID fails → RL" is
  dead); **MRAC fails** even when fair; **pendulum** is the strongest
  RL-beats-fixed result.
- Scientific boundary: RL's value is **per-episode initial adaptation**, not
  mid-episode shock rejection, not windup handling.

## Document index

| File | Contents |
|------|----------|
| `00_OVERVIEW.md` | this file — orientation + index |
| `01_AUDIT_DEFECTS_AND_FIXES.md` | every defect F1–F9: symptom, root cause, diagnosis, fix, before/after |
| `02_EXPERIMENTS_AND_RESULTS.md` | every experiment, command, and result number by RQ |
| `03_INFRASTRUCTURE_AND_INCIDENTS.md` | venv repair, sleep stall, DLL crash, resumable scripts, ops lessons |
| `04_WRITING_OVERHAUL_AND_DECISIONS.md` | chapter-by-chapter changes + framing decisions |
| `05_CURRENT_STATUS_AND_NEXT_STEPS.md` | what's done, what's open, how to continue |
| `06_REPRODUCTION_GUIDE.md` | exact commands to reproduce everything |

Canonical companions (outside this dir):
- `AUDIT_FINDINGS.md` — terse defect/result ledger (root).
- `thesis_doc/FINDINGS_SUMMARY.md` (v2) — results narrative.
- `thesis_doc/CONTEXT.md` (v2) — glossary, agent names, key numbers.
- `thesis_doc/AUDIT…`/chapters — the thesis itself.

## Timeline (this session)

1. Explored repo + read original thesis/code → produced honest evaluation.
2. User locked scope: full upgrade (all tiers + pendulum), actuator axis,
   MRAC rerun, HiP-MDP/RMA reframe.
3. **Phase A** — fixed dt bug across all scripts; reran every eval.
4. **Phase B** — anti-windup baselines, feasible MRAC, actuator axis + eval
   protocol v2, 5-seed Stage 6 retrains, sweeps, zero-shot no-reset.
5. **RQ3/RQ4** — stack ablation, GRU, probing, gain analysis, shock probe.
6. **Pendulum** — second plant built + trained + evaluated.
7. **Writing** — all 7 chapters + appendix rewritten; ground-truth docs;
   diagrams; references.
8. **PR #37** opened (branch `thesis-audit-rewrite`); pre-commit lint debt
   cleaned.
9. **Now** — expanding prose toward ~100 pages + this documentation set.

## Two operational incidents (don't lose these)

- Training **stalled** once from laptop **sleep/hibernate** (clean freeze, no
  traceback). Fixed: `powercfg ... standby/hibernate-timeout-ac 0`; keep
  plugged in.
- Phase C batch **crashed** once on `0xC0000142` (Windows DLL-init, transient
  subprocess-spawn failure) + `set -e`. Fixed: resumable scripts (no `set -e`,
  retry-once, skip-if-done). See `03_INFRASTRUCTURE_AND_INCIDENTS.md`.
