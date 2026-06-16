# 05 — Current Status and Next Steps

_Last updated: June 2026, during the expansion pass._

---

## Done ✅

**Experiments (all complete, corrected, multi-seed where stated):**
- dt fix + full eval rerun (Phase A).
- Eval protocol v2 (bands, jitter, true context push).
- Actuator-strength axis + `thesis_v6_hipmdp` preset; 9-dim obs.
- Anti-windup PID baselines (back-calc + clamping).
- Feasible MRAC rerun (still 0%, fairly).
- Stage 6 context/blind retrains, 5 seeds, evaluated + aggregated.
- RQ3: stack ablation k∈{1,3,5,10,20}; GRU blind ×2 seeds.
- RQ4: probing (double dissociation).
- Mass + actuator sweeps; mid-approach shock probe; zero-shot no-reset; gain
  analysis.
- Pendulum second plant: built, tuned, trained (context+blind ×2 seeds),
  evaluated.

**Writing (content/structure complete):**
- All 7 chapters + appendix rewritten (HiP-MDP/teacher-student reframe, RQ1–4,
  corrected numbers, audit-as-contribution).
- Ground-truth docs updated (FINDINGS_SUMMARY v2, CONTEXT v2, CLAUDE.md);
  WRITING_GUIDE superseded; references consolidated.
- 4 Ch3 diagrams generated; result figures exist.

**Repo / PR:**
- PR #37 (`thesis-audit-rewrite` → `main`) opened with the full session:
  https://github.com/concolabs-com/adaptive-pid-rl-/pull/37
- Pre-commit lint debt cleaned (black/isort/flake8 pass).
- `AUDIT_FINDINGS.md` + this `PROJECT_LOG/` are the canonical context records.

---

## In progress 🔄

**Task 20 — expand prose to ~100 pages.** The current draft is correct but
dense (~31 body pages). Expansion plan (real content, per chapter):
- **Front matter:** abstract, acknowledgements, TOC, list of figures/tables,
  nomenclature/notation table.
- **Ch2:** full TRPO→PPO derivation, GAE derivation, the plant ODE solution and
  v_max derivation, broader literature (meta-RL, sysID, RL-for-control survey),
  worked windup numeric example.
- **Ch3:** worked terminal-speed + windup-inequality examples; full reward-term
  justification; pendulum cascade derivation.
- **Ch5:** a paragraph of discussion per figure; per-scenario walk-throughs;
  the statistical tests written out; embedded figures with captions.
- **Ch6/7:** deepen interpretation and future work.
- Target ~28–30k words body.

---

## Open items needing the author / supervisor ⚠

1. **3 online-RL-PID citations** — flagged ⚠ in `thesis_doc/references.md` and
   the Ch2 §2.6 gap statement. Needs a literature search the author should run
   and validate (do not fabricate specific papers/venues). Also verify the
   Shi et al. entry's venue/pages.
2. **Supervisor sign-offs:** final title (working: "Learning to Schedule PID
   Gains Under Hidden Dynamics Parameters…"); citation style (IEEE/APA/Harvard
   per university); contributions-list format.
3. **`[^astrom1995]` page reference** for the "~95% of loops are PID" statistic.

---

## Recommended next experiments (optional, if time allows)

- **Width-matched memory ablation** — GRU(128) vs MLP of equal parameter count,
  to isolate recurrence from capacity (resolves the RQ3b confound).
- **A plant where identification is hard** — viscous/slip-dependent friction or
  a delayed actuator fault, so the third axis is meaningful and the blindness
  penalty grows (context should matter more).
- **Approach-phase gain analysis** — characterize the gains during approach/
  braking (where scheduling actually lives), not just the flat hold phase.
- **SAC port** — `stage2_sac_reproduction.py` lacks the v6/actuator/eval-v2
  args; port them for an off-policy algorithm comparison (deferred, low
  priority).
- **More pendulum seeds + bootstrap significance** throughout.

---

## How to pick this up cold

1. Read `PROJECT_LOG/00_OVERVIEW.md`, then `01`–`04`.
2. `AUDIT_FINDINGS.md` for the terse number ledger; `FINDINGS_SUMMARY.md` (v2)
   for the narrative.
3. `06_REPRODUCTION_GUIDE.md` for exact commands.
4. Chapters are in `thesis_doc/chapterNN/`; appendix in
   `thesis_doc/appendices/`; new theory drafts in `thesis_doc/new_sections/`.
5. Do **not** trust numbers in `THESIS_WRITING_GUIDE.md` (superseded) or in any
   pre-audit memory; verify against `benchmark_results/` (regenerate via the
   scripts) or the v2 docs.
