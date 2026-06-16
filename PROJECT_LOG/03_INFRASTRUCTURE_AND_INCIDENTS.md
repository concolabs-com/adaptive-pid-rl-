# 03 — Infrastructure, Incidents, and Operational Lessons

Everything that went wrong with the *machinery* (not the science) and how it
was handled. Important for anyone re-running on this or a similar laptop.

---

## Environment

- **Machine:** laptop — Intel i7-11800H, 32 GB RAM, RTX 3070 Laptop. Training
  runs on **CPU** (the MLP/GRU policies are tiny; SyncVectorEnv with 4 envs
  uses ~4 cores).
- **Python:** 3.10.11, venv at `D:\Fable\Car_Thesis\venv`.
- **Timings:** car 1M-step MLP run ≈ 15 min; GRU ≈ 75 min (BPTT); pendulum
  300k ≈ 8–12 min. Full 5-seed pair + ablations + pendulum ≈ ~1 day wall-clock.

---

## Incident 1 — venv was missing analysis packages

**Symptom.** First eval rerun crashed: `ModuleNotFoundError: No module named
'pandas'`. The venv had torch/gymnasium/mujoco/numpy but not the analysis
stack.

**Cause.** The project was moved (`D:\Msc\Master_Thesis` → `D:\Fable\Car_Thesis`)
and the venv was partial / stale.

**Fix.** `venv/Scripts/python.exe -m pip install pandas matplotlib tqdm seaborn
scipy scikit-learn`. (scikit-learn/scipy later needed for the probing analysis;
autoflake added later for the lint pass.)

**Lesson.** Verify the venv imports the full stack before launching long runs:
`python -c "import pandas,matplotlib,tqdm,torch,scipy,sklearn"`.

---

## Incident 2 — eval-only path crashed writing CSV

**Symptom.** `OSError: Cannot save file into a non-existent directory:
'benchmark_results\dtfix_stage5a_static\seed_7'`.

**Cause.** The `--eval-only` path in `stage2_meta_rl_reproduction.py` wrote
`seed_dir/eval_raw.csv` without creating `seed_dir` (only the training path
made it).

**Fix.** Added `seed_dir.mkdir(parents=True, exist_ok=True)` before the CSV
write in `main()`.

---

## Incident 3 — training STALLED (laptop sleep/hibernate)

**Symptom.** Stage 6 training (5-seed chain) froze mid-run at seed 84, update
92/122. The train log stopped advancing; the chain's background task never
completed; no python process was burning CPU/RAM (only an 11 MB / 0-CPU stray).

**How diagnosed.** (1) Train-log mtime frozen for a multi-hour wall-clock gap.
(2) No CPU-heavy python in `Get-Process python`. (3) Clean freeze — **no
traceback, no MemoryError, no "killed"** in any log. (4) `seed_84/models/`
empty (PPO saves the checkpoint only at seed end). A clean stop + wall-clock
gap + laptop ⇒ the OS suspended the process and it never resumed.

**Fix.** Disabled sleep/hibernate on AC power:
```
powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0
```
Keep the laptop plugged in for long runs. Then resumed only the missing work
(`resume_stage6.sh`): seeds 7/21/42 had checkpoints, so re-ran only 84+123 for
6a, then all of 6b. PPO here has **no mid-training checkpoint**, so a killed
seed restarts clean — re-run only seeds whose `seed_N/models/meta_rl_agent.pth`
is missing.

**Diagnostic gotcha.** `Get-Process python` working-set/CPU can misreport
(snapshot quirk): a genuinely training process sometimes showed 11 MB / 0 CPU.
**Trust the advancing tqdm timestamps in the train log**, not the process
snapshot, to decide alive-vs-dead.

---

## Incident 4 — Phase C batch CRASHED (0xC0000142 DLL-init)

**Symptom.** The Phase C batch (stack ablation → GRU → pendulum) died after
stack k=1 and k=3 completed: `k=5 FAILED rc=3221225794`.
`3221225794 = 0xC0000142 = STATUS_DLL_INIT_FAILED` — a transient Windows
failure when a subprocess fails to initialize its DLLs at startup (handle /
desktop-heap exhaustion is the usual cause after many rapid subprocess spawns
in a long batch). Because the runner used `set -e`, the single failure aborted
the entire batch.

**Root contributor.** Deeply nested spawning: bash → runner python →
`subprocess.run` → trainer python, repeated per k over a long batch.

**Fix.** `resume_phaseC.sh`, hardened:
- **no `set -e`** — one transient failure must not abort the rest;
- **retry-once** wrapper per step (the DLL-init failure is transient);
- **skip-if-done** guards (re-running is cheap and idempotent — k=1/k=3 not
  redone);
- **direct `stage2` calls** (one fewer nested subprocess layer than the
  per-stage runner scripts).
Confirmed the resumed k=5 launched cleanly past the prior failure point; the
whole batch then completed (stack k=5,20; GRU s7,s21; pendulum context/blind
s7,s21 + eval).

**Lesson.** For long Windows batches that spawn many python subprocesses: avoid
`set -e` for the outer loop, make every step idempotent (skip-if-output-exists)
and retryable, and minimize spawn nesting.

---

## Resumable run scripts (all idempotent)

| Script | Purpose |
|--------|---------|
| `rerun_evals_dtfix.sh` / `rerun_evals_dtfix2.sh` | Phase A: rerun all evals with the dt fix |
| `resume_stage6.sh` | finish Stage 6 after the sleep stall (missing seeds only) |
| `run_phaseC_batch.sh` | original Phase C batch (crashed) |
| `resume_phaseC.sh` | hardened Phase C resume (no set -e, retry, skip-if-done) |

Pattern to copy for any new long run: check for the output file, skip if
present, run with one retry, never `set -e` across independent steps.

---

## Pre-commit / lint (PR #37)

The repo's `.pre-commit-config.yaml` runs **black** (line 120) → **isort** →
**flake8** (max 120, ignore E203/W503). The first commit was **blocked** by 94
flake8 issues across code that had never been linted (the repo had no python
committed). Resolved without `--no-verify`:
- `autoflake --remove-all-unused-imports --remove-unused-variables` for F401/F841;
- removed a dead duplicate `__init__` in `agents/model.py` (F811);
- wrapped long string literals / split a pyright comment (E501);
- lambda→def (E731); dropped f-prefix on placeholder-less f-strings (F541);
- added **E402** to the flake8 ignore (justified: `matplotlib.use("Agg")` and
  `sys.path.insert` legitimately precede imports in scripts).
Then black+isort+flake8 → 0 issues, committed clean.

`.gitignore` excludes `venv/`, `benchmark_results/`, `*.pth/*.csv/*.png/*.json`,
`*.log`, `*.rar`, `MUJOCO_LOG.TXT` — only code + text is committed; result
figures regenerate from scripts.
