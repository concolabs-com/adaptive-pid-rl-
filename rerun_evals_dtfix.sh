#!/usr/bin/env bash
# Phase A: rerun all thesis evals with corrected dt (env.dt = 0.02 s, not model.opt.timestep = 0.002 s).
# Uses existing seed-7 checkpoints. Old chapter05 data preserved in thesis_doc/chapter05/data.
set -e
cd "$(dirname "$0")"
PY=venv/Scripts/python.exe

S5A=benchmark_results/stage5a_context_cliff/seed_7/models/meta_rl_agent.pth
S5B=benchmark_results/stage5b_blind_cliff/seed_7/models/meta_rl_agent.pth

echo "=== [1/7] Baseline static ==="
$PY stage_baseline_fixed_pid.py
echo "=== [2/7] Baseline dynamic ==="
$PY stage_baseline_fixed_pid.py --dynamic
echo "=== [3/7] Baseline no-reset ==="
$PY stage_baseline_fixed_pid.py --no-reset

echo "=== [4/7] Stage 5a static ==="
$PY stage2_meta_rl_reproduction.py --eval-only --init-model-path "$S5A" \
  --protocol-preset thesis_v4_cliff --stack-size 10 --total-timesteps 0 --num-envs 1 \
  --seeds 7 --output-dir benchmark_results/dtfix_stage5a_static
echo "=== [5/7] Stage 5a dynamic ==="
$PY stage2_meta_rl_reproduction.py --eval-only --init-model-path "$S5A" \
  --protocol-preset thesis_v4_cliff --stack-size 10 --total-timesteps 0 --num-envs 1 \
  --seeds 7 --disturbance-in-eval --output-dir benchmark_results/dtfix_stage5a_dynamic

echo "=== [6/7] Stage 5b static ==="
$PY stage2_meta_rl_reproduction.py --eval-only --init-model-path "$S5B" \
  --protocol-preset thesis_v4_cliff --obs-keep-dims 6 --stack-size 10 --total-timesteps 0 --num-envs 1 \
  --seeds 7 --output-dir benchmark_results/dtfix_stage5b_static
echo "=== [7/7] Stage 5b dynamic ==="
$PY stage2_meta_rl_reproduction.py --eval-only --init-model-path "$S5B" \
  --protocol-preset thesis_v4_cliff --obs-keep-dims 6 --stack-size 10 --total-timesteps 0 --num-envs 1 \
  --seeds 7 --disturbance-in-eval --output-dir benchmark_results/dtfix_stage5b_dynamic

echo "=== ALL EVAL RERUNS DONE ==="
