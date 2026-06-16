#!/usr/bin/env python3
"""
Stage 4b: Blind Meta-RL — frame stacking variant (ablation of Stage 4 GRU).

Stage 4 used a GRU recurrent policy to infer dynamics from trajectory history.
This variant replaces GRU with frame stacking (last 10 obs concatenated = 60-dim
flat input to MLP) — same architecture as Stage 2, only difference is obs_keep_dims=6
(no context dims). This gives a cleaner ablation: Stage 2 vs Stage 4b differs only
in whether mass_scale and friction_scale are visible.

Observation (6-dim × 10 frames = 60-dim): [pos, vel, err, kp, ki, kd] × last 10 steps
Hidden from agent: [mass_scale, friction_scale] ← indices 6, 7 dropped
Policy:            MLP (same arch as Stage 2), no BPTT instability
Timesteps:         1,000,000
Curriculum:        4 phases 1→3→5→7→10 m (same as Stage 4 GRU)

Comparison matrix:
  Stage 2        = MLP + obs_keep_dims=8 (context visible)
  Stage 4 GRU    = GRU + obs_keep_dims=6 (blind, recurrent memory)
  Stage 4b (this)= MLP + obs_keep_dims=6 + stack=10 (blind, fixed window)
"""

import subprocess
import sys
from pathlib import Path

OUTPUT_DIR = "benchmark_results/stage4b_blind_framestacking"

STAGE4B_ARGS = [
    "--protocol-preset",
    "thesis_v3_safety",
    "--obs-keep-dims",
    "6",  # drop mass_scale (idx 6) and friction_scale (idx 7)
    "--stack-size",
    "10",  # stack last 10 obs → 60-dim flat input to MLP
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

    # Single seed for initial validation; expand to full seed list once confirmed.
    test_seed = "7"
    cmd = [sys.executable, str(stage2_script)] + STAGE4B_ARGS + ["--seeds", test_seed]

    print("=" * 65)
    print("STAGE 4b: Blind Meta-RL — Frame Stacking")
    print("  obs       : 6-dim × 10 frames = 60-dim flat")
    print("  hidden    : mass_scale, friction_scale (dropped)")
    print("  policy    : MLP (no GRU, no BPTT)")
    print("  timesteps : 1,000,000")
    print("  curriculum: 1→3→5→7→10 m  (4 phases, equal fractions)")
    print(f"  seed      : {test_seed}  (single seed — validation run)")
    print(f"  output    : {OUTPUT_DIR}")
    print("=" * 65)
    print()

    result = subprocess.run(cmd, check=False)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
