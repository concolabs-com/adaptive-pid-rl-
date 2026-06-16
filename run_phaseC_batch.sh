#!/usr/bin/env bash
# Phase C training batch: RQ3 ablations + pendulum transfer.
# Sequential (CPU-bound). Each stage6c/6d run = 1M steps; pendulum = 300k.
set -e
cd "$(dirname "$0")"
PY=venv/Scripts/python.exe

echo "=== [1/3] stack-size ablation k in {1,3,5,20} ==="
$PY stage6c_stack_ablation.py

echo "=== [2/3] GRU blind (v6), seeds 7,21 ==="
$PY stage6d_gru_blind.py

echo "=== [3/3] pendulum transfer: context + blind, seeds 7,21 ==="
for seed in 7 21; do
  $PY pendulum_experiment.py --mode train --seed $seed
  $PY pendulum_experiment.py --mode train --seed $seed --blind
done
$PY pendulum_experiment.py --mode eval --seed 7

echo "=== PHASE C BATCH DONE ==="
