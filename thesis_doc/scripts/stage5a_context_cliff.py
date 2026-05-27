#!/usr/bin/env python3
"""
Stage 5a: Meta-RL with context + cliff self-braking protocol.

Identical to Stage 2 (agent sees mass_scale and friction_scale) but trained
with the thesis_v4_cliff protocol instead of thesis_v3_safety:

  - Speed governor REMOVED: agent must learn to brake itself via gain scheduling
  - Cliff penalty (300) for overshooting past target + hard termination
  - Stronger velocity shaping near target (wider zones, higher coefficients)
  - Goal: agent learns approach fast → self-decelerate → slow creep → hold

Observation (8-dim): [pos, vel, err, kp, ki, kd, mass_scale, friction_scale]
Policy:              MLP, stack_size=10
Timesteps:           1,000,000

Run single seed 7 first to validate self-braking emerges, then expand.
"""

import subprocess
import sys
from pathlib import Path


OUTPUT_DIR = "benchmark_results/stage5a_context_cliff"

STAGE5A_ARGS = [
    "--protocol-preset",   "thesis_v4_cliff",
    "--stack-size",        "10",
    "--total-timesteps",   "1000000",
    "--num-envs",          "4",
    "--num-steps",         "2048",
    "--curriculum-enabled",
    "--curriculum-spec",   "1:3:0.25,1:5:0.25,1:7:0.25,1:10:0.25",
    "--output-dir",        OUTPUT_DIR,
]


def main() -> int:
    stage2_script = Path(__file__).parent / "stage2_meta_rl_reproduction.py"
    if not stage2_script.exists():
        print(f"ERROR: Required script not found: {stage2_script}", file=sys.stderr)
        return 1

    test_seed = "7"
    cmd = [sys.executable, str(stage2_script)] + STAGE5A_ARGS + ["--seeds", test_seed]

    print("=" * 65)
    print("STAGE 5a: Meta-RL with Context — Cliff Self-Braking Protocol")
    print("  obs       : 8-dim  [pos, vel, err, kp, ki, kd, mass, friction]")
    print("  policy    : MLP, stack_size=10")
    print("  protocol  : thesis_v4_cliff (no governor, cliff=300, vel shaping)")
    print("  timesteps : 1,000,000")
    print("  curriculum: 1->3->5->7->10 m  (4 phases, equal fractions)")
    print(f"  seed      : {test_seed}  (single seed)")
    print(f"  output    : {OUTPUT_DIR}")
    print("=" * 65)
    print()

    result = subprocess.run(cmd, check=False)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
