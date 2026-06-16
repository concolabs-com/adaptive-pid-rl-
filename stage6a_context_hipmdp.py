#!/usr/bin/env python3
"""
Stage 6a: Context-aware agent (teacher) on the HiP-MDP protocol — 5 seeds.

Successor to Stage 5a after the experimental audit:
  - thesis_v6_hipmdp preset: v4_cliff reward/curriculum + actuator-strength
    randomization (0.6-1.4), the discriminative second dynamics axis
  - Observation (9-dim): [pos, vel, err, a_kp, a_ki, a_kd,
                          mass_scale, friction_scale, actuator_scale]
  - Eval protocol v2: per-episode physics bands, target jitter, and TRUE
    context push (the v1 static eval fed the context agent (1,1) — out of
    distribution — making the original Stage 5a static results invalid)
  - 5 seeds for honest variance quantification

In the teacher-student (RMA-style) framing: this is the privileged teacher.
"""

import subprocess
import sys
from pathlib import Path

OUTPUT_DIR = "benchmark_results/stage6a_context_hipmdp"

ARGS = [
    "--protocol-preset",
    "thesis_v6_hipmdp",
    "--stack-size",
    "10",
    "--total-timesteps",
    "1000000",
    "--num-envs",
    "4",
    "--num-steps",
    "2048",
    "--curriculum-enabled",
    "--curriculum-spec",
    "1:3:0.25,1:5:0.25,1:7:0.25,1:10:0.25",
    "--seeds",
    "7,21,42,84,123",
    "--output-dir",
    OUTPUT_DIR,
]


def main() -> int:
    stage2_script = Path(__file__).parent / "stage2_meta_rl_reproduction.py"
    if not stage2_script.exists():
        print(f"ERROR: {stage2_script} not found", file=sys.stderr)
        return 1

    cmd = [sys.executable, str(stage2_script)] + ARGS
    print("=" * 65)
    print("STAGE 6a: Context-Aware Agent (teacher) — HiP-MDP protocol")
    print("  obs       : 9-dim incl. mass/friction/actuator context")
    print("  axes      : mass 5-20 kg, friction 0.1-2.0, actuator 0.6-1.4")
    print("  eval      : protocol v2 (bands + target jitter + true context)")
    print("  seeds     : 7, 21, 42, 84, 123")
    print(f"  output    : {OUTPUT_DIR}")
    print("=" * 65)
    return subprocess.run(cmd, check=False).returncode


if __name__ == "__main__":
    sys.exit(main())
