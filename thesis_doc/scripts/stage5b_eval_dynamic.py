#!/usr/bin/env python3
"""
Stage 5b Dynamic Eval: re-evaluate trained blind model with mid-episode disturbances.

Loads the Stage 5b trained model (seed 7) and runs eval with disturbance-in-eval=True:
  - Mid-episode physics change fires at random step 120-220
  - Position friction patch active (x=1.5-2.4m, friction*0.35)
  - Agent still cannot see mass_scale or friction_scale (obs_keep_dims=6)
No re-training — pure eval of the existing checkpoint.

Model:    benchmark_results/stage5b_blind_cliff/seed_7/models/meta_rl_agent.pth
Output:   benchmark_results/stage5b_dynamic_eval/
"""

import subprocess
import sys
from pathlib import Path

MODEL_PATH = "benchmark_results/stage5b_blind_cliff/seed_7/models/meta_rl_agent.pth"
OUTPUT_DIR = "benchmark_results/stage5b_dynamic_eval"

ARGS = [
    "--protocol-preset",   "thesis_v4_cliff",
    "--obs-keep-dims",     "6",
    "--stack-size",        "10",
    "--total-timesteps",   "0",
    "--num-envs",          "1",
    "--seeds",             "7",
    "--output-dir",        OUTPUT_DIR,
    "--init-model-path",   MODEL_PATH,
    "--disturbance-in-eval",
    "--eval-only",
]


def main() -> int:
    stage2_script = Path(__file__).parent / "stage2_meta_rl_reproduction.py"
    if not stage2_script.exists():
        print(f"ERROR: {stage2_script} not found", file=sys.stderr)
        return 1

    model_path = Path(__file__).parent / MODEL_PATH
    if not model_path.exists():
        print(f"ERROR: model not found: {model_path}", file=sys.stderr)
        print("  Run stage5b_blind_cliff.py first.", file=sys.stderr)
        return 1

    cmd = [sys.executable, str(stage2_script)] + ARGS

    print("=" * 65)
    print("STAGE 5b DYNAMIC EVAL — Blind RL with Disturbances")
    print("  model   : Stage 5b checkpoint (seed 7)")
    print("  obs     : 6-dim [pos, vel, err, kp, ki, kd] — no context")
    print("  disturbance: mid-episode physics change + position patch")
    print(f"  output  : {OUTPUT_DIR}")
    print("=" * 65)
    print()

    result = subprocess.run(cmd, check=False)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
