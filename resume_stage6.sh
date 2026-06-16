#!/usr/bin/env bash
# Resume Stage 6 after the sleep-induced stall.
# 6a seeds 7,21,42 already done (checkpoints present) -> run only 84,123.
# 6b not started -> run all 5 seeds via its own runner.
set -e
cd "$(dirname "$0")"
PY=venv/Scripts/python.exe

echo "=== [6a resume] seeds 84,123 ==="
$PY stage2_meta_rl_reproduction.py \
  --protocol-preset thesis_v6_hipmdp --stack-size 10 \
  --total-timesteps 1000000 --num-envs 4 --num-steps 2048 \
  --curriculum-enabled --curriculum-spec "1:3:0.25,1:5:0.25,1:7:0.25,1:10:0.25" \
  --seeds 84,123 --output-dir benchmark_results/stage6a_context_hipmdp

echo "=== [6b] all 5 seeds ==="
$PY stage6b_blind_hipmdp.py

echo "=== STAGE 6 RESUME DONE ==="
