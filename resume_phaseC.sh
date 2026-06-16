#!/usr/bin/env bash
# Robust resume of the Phase C batch after the 0xC0000142 (DLL init) crash.
# - NO `set -e`: one transient failure must not abort the rest.
# - Skip-if-done guards: re-running is cheap and idempotent.
# - retry(): one automatic retry on the transient Windows DLL-init failure.
# - Direct stage2 calls (one fewer nested subprocess layer than the runners).
cd "$(dirname "$0")"
PY=venv/Scripts/python.exe

run () {  # run <model_path_to_check> <log> <cmd...>
  local model="$1"; shift
  local log="$1"; shift
  if [ -f "$model" ]; then echo "SKIP (done): $model"; return 0; fi
  for attempt in 1 2; do
    echo "--- attempt $attempt: $* (log: $log)"
    "$@" > "$log" 2>&1
    local rc=$?
    if [ $rc -eq 0 ] && [ -f "$model" ]; then echo "OK: $model"; return 0; fi
    echo "FAIL rc=$rc (attempt $attempt) — tail:"; tail -3 "$log" | tr '\r' '\n'
  done
  echo "GAVE UP: $model"; return 1
}

CURR="1:3:0.25,1:5:0.25,1:7:0.25,1:10:0.25"

echo "=== stack ablation k=5 ==="
run benchmark_results/stage6c_stack5/seed_7/models/meta_rl_agent.pth \
    benchmark_results/stage6c_stack5.log \
    $PY stage2_meta_rl_reproduction.py --protocol-preset thesis_v6_hipmdp \
      --obs-keep-dims 6 --stack-size 5 --total-timesteps 1000000 \
      --num-envs 4 --num-steps 2048 --curriculum-enabled --curriculum-spec "$CURR" \
      --seeds 7 --output-dir benchmark_results/stage6c_stack5

echo "=== stack ablation k=20 ==="
run benchmark_results/stage6c_stack20/seed_7/models/meta_rl_agent.pth \
    benchmark_results/stage6c_stack20.log \
    $PY stage2_meta_rl_reproduction.py --protocol-preset thesis_v6_hipmdp \
      --obs-keep-dims 6 --stack-size 20 --total-timesteps 1000000 \
      --num-envs 4 --num-steps 2048 --curriculum-enabled --curriculum-spec "$CURR" \
      --seeds 7 --output-dir benchmark_results/stage6c_stack20

echo "=== GRU blind seed 7 ==="
run benchmark_results/stage6d_gru_blind_s7/seed_7/models/meta_rl_agent.pth \
    benchmark_results/stage6d_gru_s7.log \
    $PY stage2_meta_rl_reproduction.py --protocol-preset thesis_v6_hipmdp \
      --obs-keep-dims 6 --recurrent-policy --recurrent-hidden-size 128 --stack-size 1 \
      --total-timesteps 1000000 --num-envs 4 --num-steps 2048 \
      --curriculum-enabled --curriculum-spec "$CURR" \
      --seeds 7 --output-dir benchmark_results/stage6d_gru_blind_s7

echo "=== GRU blind seed 21 ==="
run benchmark_results/stage6d_gru_blind_s21/seed_21/models/meta_rl_agent.pth \
    benchmark_results/stage6d_gru_s21.log \
    $PY stage2_meta_rl_reproduction.py --protocol-preset thesis_v6_hipmdp \
      --obs-keep-dims 6 --recurrent-policy --recurrent-hidden-size 128 --stack-size 1 \
      --total-timesteps 1000000 --num-envs 4 --num-steps 2048 \
      --curriculum-enabled --curriculum-spec "$CURR" \
      --seeds 21 --output-dir benchmark_results/stage6d_gru_blind_s21

for seed in 7 21; do
  echo "=== pendulum context seed $seed ==="
  run benchmark_results/pendulum/context_seed$seed/agent.pth \
      benchmark_results/pendulum_context_s$seed.log \
      $PY pendulum_experiment.py --mode train --seed $seed
  echo "=== pendulum blind seed $seed ==="
  run benchmark_results/pendulum/blind_seed$seed/agent.pth \
      benchmark_results/pendulum_blind_s$seed.log \
      $PY pendulum_experiment.py --mode train --seed $seed --blind
done

echo "=== pendulum eval ==="
$PY pendulum_experiment.py --mode eval --seed 7 > benchmark_results/pendulum_eval.log 2>&1
echo "pendulum eval rc=$?"

echo "=== PHASE C RESUME DONE ==="
