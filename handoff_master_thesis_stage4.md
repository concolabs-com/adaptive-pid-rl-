# Handoff: Master Thesis — Stage 4 Blind Meta-RL

**Date:** 2026-05-24  
**Project:** `D:\Msc\Master_Thesis`  
**Context:** Meta-RL for adaptive PID gain scheduling on a simulated 2-wheeled MuJoCo car.

---

## Standing Artifacts (read these first)

| Artifact | Path |
|----------|------|
| Memory index | `C:\Users\sahan\.claude\projects\D--Msc-Master-Thesis\memory\MEMORY.md` |
| Project memory | `C:\Users\sahan\.claude\projects\D--Msc-Master-Thesis\memory\project_thesis.md` |
| User profile | `C:\Users\sahan\.claude\projects\D--Msc-Master-Thesis\memory\user_profile.md` |
| Codebase overview | `D:\Msc\Master_Thesis\RESEARCH_OVERVIEW.md` |
| MRAC investigation | `D:\Msc\Master_Thesis\MRAC_INVESTIGATION.md` |
| Previous plan | `C:\Users\sahan\.claude\plans\can-you-go-through-humming-piglet.md` |

---

## What's Done

### Stage 4 Blind Meta-RL (`stage4_blind_meta_rl.py`)
Trains a GRU recurrent PPO policy with `obs_keep_dims=6` — drops mass_scale (index 6) and friction_scale (index 7) from the 8-dim observation. Agent must infer physics from trajectory alone.

Key CLI args passed to `stage2_meta_rl_reproduction.py`:
```
--obs-keep-dims 6
--recurrent-policy
--stack-size 1
--recurrent-hidden-size 128
--total-timesteps 1000000
--curriculum-enabled
--curriculum-spec 1:3:0.25,1:5:0.25,1:7:0.25,1:10:0.25
--protocol-preset thesis_v3_safety
```

### MRAC Classical Baseline (`stage4_mrac_baseline.py`)
Three variants tried (plain MIT rule → velocity regressor → sigma-modification). **All failed: 0% success across all scenarios.** Fully documented in `MRAC_INVESTIGATION.md`. No further MRAC work needed.

---

## Current State

### Stage 4 Training Results (as of 2026-05-24)

Seeds **7 and 21 complete**. Seeds **42, 84, 123 still running** (or not yet started).

Eval results — both seeds produce **identical numbers**:

| Scenario | Success | Settling | IAE | Final Abs Error |
|----------|---------|----------|-----|-----------------|
| Light & Grippy (5 kg, fr=2.0) | **100%** | 5.66 s | 3.45 | 0.049 |
| Standard (10 kg, fr=1.0) | **0%** | 10.0 s (timeout) | 4.21 | 0.175 |
| Heavy & Slippery (20 kg, fr=0.2) | **0%** | 10.0 s (timeout) | 4.22 | 0.157 |

Eval results at: `benchmark_results/stage4_blind_meta_rl/seed_7/eval_seed_summary.csv` and `seed_21/`.

### Training Curve Issues (Both Seeds)
- `value_loss` explodes at every curriculum phase boundary:
  - Phase 1→2 (target up to 5 m): spike ~300
  - Phase 2→3 (up to 7 m): spike ~640
  - Phase 3→4 (up to 10 m): spike ~**1754**
- `recent_episode_return_mean` never converges — stays deeply negative throughout
- GRU BPTT + curriculum advancement = instability pattern

---

## Open Decision: GRU vs Frame Stacking

The user and previous assistant discussed switching from GRU recurrent policy to **frame stacking** as an alternative. No code changes were made — user said "Don't do any changes just wanted to discuss this with you."

**Frame stacking approach:**
- Remove `--recurrent-policy` flag from `stage4_blind_meta_rl.py`
- Change `--stack-size 1` to `--stack-size 10` (stack last 10 obs as flat 60-dim input)
- Result: MLP policy, same architecture as Stage 2 (cleaner ablation), 3-5× faster training, no BPTT

**Trade-off:**
- Pro: Cleaner comparison to Stage 2 (same arch, only difference = no context dims)
- Pro: No GRU value_loss instability
- Con: Fixed memory window (10 steps = 1 s), not theoretically infinite like GRU
- The 0% on Standard/Heavy may persist regardless — removing context dims is a hard constraint

**User has NOT decided yet.** Ask before implementing.

---

## Next Steps (in priority order)

1. **Check if seeds 42, 84, 123 have finished:**
   ```
   benchmark_results/stage4_blind_meta_rl/seed_42/
   benchmark_results/stage4_blind_meta_rl/seed_84/
   benchmark_results/stage4_blind_meta_rl/seed_123/
   ```

2. **If seeds complete:** Compile full comparison table across all 5 seeds. Aggregate mean ± std success rate per scenario.

3. **Frame stacking decision:** Confirm with user whether to rerun Stage 4 with frame stacking instead of GRU. If yes, edit `stage4_blind_meta_rl.py`:
   - Remove `"--recurrent-policy"` from `STAGE4_ARGS`
   - Change `"--stack-size", "1"` → `"--stack-size", "10"`
   - Remove `"--recurrent-hidden-size", "128"` (only relevant for GRU)

4. **Final thesis comparison table** (once Stage 4 complete):
   
   | Method | Standard | Heavy & Slippery | Light & Grippy |
   |--------|----------|------------------|----------------|
   | Fixed PID (Stage 1) | ? | ? | ? |
   | MRAC (classical adaptive) | 0% | 0% | 0% |
   | Meta-RL with context (Stage 2) | ? | ? | ? |
   | Meta-RL blind (Stage 4) | 0% | 0% | 100% |

   Stage 1 and Stage 2 results are in `benchmark_results/` — need to locate and cross-reference.

---

## Important Constraints

- **Sim-to-real is explicitly OUT OF SCOPE** — do not suggest or implement it
- Do not read documentation files — derive everything from code
- CAVEMAN MODE is active (terse, drop filler, fragments OK, code blocks normal)
- User is MSc student; explain tradeoffs clearly but don't over-engineer

---

## Suggested Skills

- `/plan` — if major architectural decision (GRU vs frame stacking) needs scoping before implementation
- `/handoff` — at end of next session to continue the chain

---

## Key Files Quick Reference

| File | Role |
|------|------|
| `stage4_blind_meta_rl.py` | Stage 4 runner (subprocess wrapper) |
| `stage4_mrac_baseline.py` | MRAC baseline (complete, no changes needed) |
| `stage2_meta_rl_reproduction.py` | Core training script (1400 lines, PPO + curriculum) |
| `envs/adaptive_suspension.py` | Gymnasium env, 8-dim obs, 3-dim action (gains) |
| `agents/model.py` | Actor-Critic, MLP + GRU variants |
| `benchmark_results/stage4_blind_meta_rl/` | All Stage 4 eval + training CSVs |
| `benchmark_results/stage4_mrac_baseline/` | MRAC results (complete) |
