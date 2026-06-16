#!/usr/bin/env python3
"""
Stage 5b: Blind Meta-RL + cliff self-braking protocol.

Same as Stage 5a but obs_keep_dims=6 — agent cannot see mass_scale or
friction_scale. Must infer dynamics from trajectory alone AND learn to
self-brake without knowing its own mass or friction.

This is the hardest blind variant: no governor, no context, cliff penalty
for overshoot. Shows whether trajectory history alone is enough to learn
safe self-braking across different physics.

Observation (6-dim × 10 frames = 60-dim): [pos, vel, err, kp, ki, kd] × 10
Hidden from agent: [mass_scale, friction_scale]
Policy:            MLP, stack_size=10 (same arch as Stage 5a — clean ablation)
Timesteps:         1,000,000

Run AFTER Stage 5a validates that self-braking is learnable with context.
"""

import subprocess
import sys
from pathlib import Path

OUTPUT_DIR = "benchmark_results/stage5b_blind_cliff"

STAGE5B_ARGS = [
    "--protocol-preset",
    "thesis_v4_cliff",
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
    "--output-dir",
    OUTPUT_DIR,
]


def main() -> int:
    stage2_script = Path(__file__).parent / "stage2_meta_rl_reproduction.py"
    if not stage2_script.exists():
        print(f"ERROR: Required script not found: {stage2_script}", file=sys.stderr)
        return 1

    test_seed = "7"
    cmd = [sys.executable, str(stage2_script)] + STAGE5B_ARGS + ["--seeds", test_seed]

    print("=" * 65)
    print("STAGE 5b: Blind Meta-RL — Cliff Self-Braking Protocol")
    print("  obs       : 6-dim × 10 frames = 60-dim flat (no context)")
    print("  hidden    : mass_scale, friction_scale (dropped)")
    print("  policy    : MLP (same arch as Stage 5a — clean ablation)")
    print("  protocol  : thesis_v4_cliff (no governor, no cliff, PD settling)")
    print("  timesteps : 1,000,000")
    print("  curriculum: 1->3->5->7->10 m  (4 phases, equal fractions)")
    print(f"  seed      : {test_seed}  (single seed — run after Stage 5a confirms)")
    print(f"  output    : {OUTPUT_DIR}")
    print("=" * 65)
    print()

    result = subprocess.run(cmd, check=False)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
