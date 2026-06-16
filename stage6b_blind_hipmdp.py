#!/usr/bin/env python3
"""
Stage 6b: Blind agent (student) on the HiP-MDP protocol — 5 seeds.

Same as Stage 6a but obs_keep_dims=6: the agent cannot observe mass_scale,
friction_scale, or actuator_scale. It must infer all hidden dynamics
parameters from its 10-frame trajectory history (implicit system
identification).

Observation (6-dim x 10 frames = 60-dim): [pos, vel, err, a_kp, a_ki, a_kd]
Hidden parameters: mass (5-20 kg), friction (0.1-2.0), actuator (0.6-1.4)

In the teacher-student (RMA-style) framing: this is the proprioceptive-history
student, trained from scratch (no distillation).
"""

import subprocess
import sys
from pathlib import Path

OUTPUT_DIR = "benchmark_results/stage6b_blind_hipmdp"

ARGS = [
    "--protocol-preset",
    "thesis_v6_hipmdp",
    "--obs-keep-dims",
    "6",
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
    print("STAGE 6b: Blind Agent (student) — HiP-MDP protocol")
    print("  obs       : 6-dim x 10 frames (no physics context)")
    print("  hidden    : mass, friction, actuator scales")
    print("  eval      : protocol v2 (bands + target jitter)")
    print("  seeds     : 7, 21, 42, 84, 123")
    print(f"  output    : {OUTPUT_DIR}")
    print("=" * 65)
    return subprocess.run(cmd, check=False).returncode


if __name__ == "__main__":
    sys.exit(main())
