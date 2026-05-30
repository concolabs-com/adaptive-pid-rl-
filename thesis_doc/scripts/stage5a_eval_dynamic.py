#!/usr/bin/env python3
"""
Stage 5a Dynamic Eval: re-evaluate trained context-aware model with mid-episode disturbances.

Loads the Stage 5a trained model (seed 7) and runs eval with disturbance-in-eval=True:
  - Mid-episode physics change fires at random step 120-220
  - Position friction patch active (x=1.5-2.4m, friction*0.35)
No re-training — pure eval of the existing checkpoint.

Model:    benchmark_results/stage5a_context_cliff/seed_7/models/meta_rl_agent.pth
Output:   benchmark_results/stage5a_dynamic_eval/
"""

import subprocess
import sys
from pathlib import Path

MODEL_PATH = "benchmark_results/stage5a_context_cliff/seed_7/models/meta_rl_agent.pth"
OUTPUT_DIR = "benchmark_results/stage5a_dynamic_eval"

ARGS = [
    "--protocol-preset",
    "thesis_v4_cliff",
    "--stack-size",
    "10",
    "--total-timesteps",
    "0",
    "--num-envs",
    "1",
    "--seeds",
    "7",
    "--output-dir",
    OUTPUT_DIR,
    "--init-model-path",
    MODEL_PATH,
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
        print("  Run stage5a_context_cliff.py first.", file=sys.stderr)
        return 1

    cmd = [sys.executable, str(stage2_script)] + ARGS

    print("=" * 65)
    print("STAGE 5a DYNAMIC EVAL — Context-Aware RL with Disturbances")
    print("  model   : Stage 5a checkpoint (seed 7)")
    print("  obs     : 8-dim [pos, vel, err, kp, ki, kd, mass, friction]")
    print("  disturbance: mid-episode physics change + position patch")
    print(f"  output  : {OUTPUT_DIR}")
    print("=" * 65)
    print()

    result = subprocess.run(cmd, check=False)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
