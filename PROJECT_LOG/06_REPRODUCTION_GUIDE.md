# 06 — Reproduction Guide

Exact commands to reproduce every result. Run from repo root
`D:\Fable\Car_Thesis` with the project venv. `PY=venv/Scripts/python.exe`.

> Training is CPU-bound here. Keep the laptop plugged in (sleep disabled, see
> `03_INFRASTRUCTURE_AND_INCIDENTS.md`). All long batches are idempotent —
> re-running skips finished seeds.

## 0. Environment

```bash
venv/Scripts/python.exe -m pip install pandas matplotlib tqdm seaborn scipy scikit-learn autoflake
venv/Scripts/python.exe -c "import pandas,matplotlib,tqdm,torch,scipy,sklearn; print('ok')"
# Windows: prevent sleep-stall during long runs
powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0
```

## 1. Classical baselines

```bash
# Fixed PID, with the env aid (static / dynamic / no-reset)
$PY stage_baseline_fixed_pid.py
$PY stage_baseline_fixed_pid.py --dynamic
$PY stage_baseline_fixed_pid.py --no-reset
# Anti-windup PID (back-calc + clamp), both env variants — the FAIR baseline
$PY stage_baseline_antiwindup_pid.py
# Feasible MRAC (config grid)
$PY stage4b_mrac_feasible.py
```

## 2. Main agents — Stage 6 (5 seeds each)

```bash
$PY stage6a_context_hipmdp.py     # context teacher, 9-dim, seeds 7,21,42,84,123
$PY stage6b_blind_hipmdp.py       # blind student, 6-dim
# If interrupted, resume only missing seeds:
bash resume_stage6.sh
# Cross-seed aggregate (needed; per-invocation aggregate may be partial):
$PY utils/aggregate_seeds.py benchmark_results/stage6a_context_hipmdp
$PY utils/aggregate_seeds.py benchmark_results/stage6b_blind_hipmdp
```

## 3. RQ3 — memory ablations

```bash
$PY stage6c_stack_ablation.py     # k in {1,3,5,20} (k=10 is stage6b)
$PY stage6d_gru_blind.py          # GRU blind, seeds 7,21
# or the hardened batch (stack + GRU + pendulum), idempotent:
bash resume_phaseC.sh
```

## 4. RQ4 — probing

```bash
# new blind agent (9-dim env, actuator axis); decode mass/friction/actuator
$PY probe_hidden_params.py \
   --model benchmark_results/stage6b_blind_hipmdp/seed_7/models/meta_rl_agent.pth \
   --actuator-axis --episodes 80 --out-dir benchmark_results/probe_stage6b
```

## 5. Sweeps, shock, gains

```bash
CTX=benchmark_results/stage6a_context_hipmdp/seed_7/models/meta_rl_agent.pth
BLD=benchmark_results/stage6b_blind_hipmdp/seed_7/models/meta_rl_agent.pth
# actuator sweep (discriminative axis) with new models
$PY eval_mass_sweep.py --axis actuator --context-model $CTX --blind-model $BLD \
   --context-dims 9 --out-dir benchmark_results/actuator_sweep_v6
# mass sweep
$PY eval_mass_sweep.py --axis mass --context-model $CTX --blind-model $BLD --context-dims 9
# mid-approach shock probe (mass x2.5, actuator x0.5)
$PY stage6_shock_probe.py
# gain trajectories + adaptation lag
$PY analyze_gain_trajectories.py
```

## 6. Zero-shot no-reset (old stage5 agents, illustrative)

```bash
$PY stage2_meta_rl_reproduction.py --eval-only \
   --init-model-path benchmark_results/stage5a_context_cliff/seed_7/models/meta_rl_agent.pth \
   --protocol-preset thesis_v4_cliff --no-brake-integral-reset --stack-size 10 \
   --total-timesteps 0 --num-envs 1 --seeds 7 --output-dir benchmark_results/zeroshot_noreset_5a
# (blind: add --obs-keep-dims 6, swap model + output dir)
```

## 7. Pendulum transfer

```bash
$PY pendulum_experiment.py --mode tune                 # base-gain grid search
$PY pendulum_experiment.py --mode train --seed 7       # context
$PY pendulum_experiment.py --mode train --seed 7 --blind
$PY pendulum_experiment.py --mode train --seed 21
$PY pendulum_experiment.py --mode train --seed 21 --blind
$PY pendulum_experiment.py --mode eval --seed 7        # scenario-grid survival
```

## 8. Figures

```bash
$PY thesis_doc/scripts/make_diagrams.py   # Ch3 diagrams → thesis_doc/chapter03/figures/
# result figures are emitted by the experiment scripts above (PNGs in each out-dir)
```

## Key flags reference (`stage2_meta_rl_reproduction.py`)

| Flag | Meaning |
|------|---------|
| `--protocol-preset thesis_v6_hipmdp` | actuator axis + v4-cliff reward + eval v2 |
| `--eval-protocol v2` | bands + target jitter + true context push (set by v6 preset) |
| `--obs-keep-dims 6` | blind agent (drop the 3 context dims) |
| `--obs-keep-dims 0` | full obs (9 dims) — inferred from checkpoint at eval |
| `--actuator-strength-range "0.6,1.4"` | per-episode actuator randomization |
| `--recurrent-policy --recurrent-hidden-size 128 --stack-size 1` | GRU variant |
| `--no-brake-integral-reset` | disable the env aid (no-reset env) |
| `--eval-only --init-model-path <pth>` | evaluate a checkpoint, no training |

## Gotchas

- Settling/IAE use `env.dt` (0.02 s), never `model.opt.timestep` (0.002 s).
- A split training run (e.g. seeds 7/21/42 then 84/123) leaves the built-in
  aggregate covering only the last invocation — re-aggregate with
  `utils/aggregate_seeds.py`.
- Trust the train-log tqdm timestamps, not `Get-Process` memory, to tell if a
  run is alive.
- `benchmark_results/` is git-ignored; results regenerate from these commands.
