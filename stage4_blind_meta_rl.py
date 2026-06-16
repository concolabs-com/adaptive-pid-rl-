#!/usr/bin/env python3
"""
Stage 4: Blind Meta-RL — GRU gain scheduling without explicit physics context.

Stage 2 gave the agent ground-truth mass_scale and friction_scale in indices 6-7
of the observation. That means it was doing context-conditioned interpolation,
not meta-learning. This script removes those two indices (obs_keep_dims=6) and
forces the agent to infer dynamics from trajectory history via GRU memory.

Observation (6-dim): [pos, vel, error, prev_kp, prev_ki, prev_kd]
Hidden from agent:   [mass_scale, friction_scale]  ← indices 6, 7 dropped
Policy:              GRU recurrent (hidden_size=128)
Timesteps:           1,000,000 (10x Stage 2, needed for harder blind task)
Curriculum:          4 phases 1→3→5→7→10 m (broader than Stage 2)

Built on the existing Stage 2 pipeline (stage2_meta_rl_reproduction.py).
The ObservationFeatureSelectWrapper with keep_dims=6 handles the masking.
"""

import subprocess
import sys
from pathlib import Path

OUTPUT_DIR = "benchmark_results/stage4_blind_meta_rl"

# Stage 4 command-line arguments for stage2_meta_rl_reproduction.py.
# Overrides thesis_v3_safety preset where needed.
STAGE4_ARGS = [
    "--protocol-preset",
    "thesis_v3_safety",
    "--obs-keep-dims",
    "6",  # drop mass_scale (idx 6) and friction_scale (idx 7)
    "--recurrent-policy",  # GRU: agent builds temporal context from trajectory
    "--stack-size",
    "1",  # no frame stacking — GRU handles temporal memory
    "--recurrent-hidden-size",
    "128",
    "--total-timesteps",
    "1000000",  # 1 M steps (10x Stage 2)
    "--num-envs",
    "4",
    "--num-steps",
    "2048",
    "--curriculum-enabled",  # gradually expand target distance
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

    cmd = [sys.executable, str(stage2_script)] + STAGE4_ARGS

    print("=" * 65)
    print("STAGE 4: Blind Meta-RL")
    print("  obs       : 6-dim  [pos, vel, err, kp, ki, kd]")
    print("  hidden    : mass_scale, friction_scale (dropped)")
    print("  policy    : GRU recurrent, hidden_size=128")
    print("  timesteps : 1,000,000")
    print("  curriculum: 1→3→5→7→10 m  (4 phases, equal fractions)")
    print(f"  output    : {OUTPUT_DIR}")
    print("=" * 65)
    print()

    result = subprocess.run(cmd, check=False)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
