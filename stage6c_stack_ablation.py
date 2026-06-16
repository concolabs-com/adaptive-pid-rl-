#!/usr/bin/env python3
"""
Stage 6c: RQ3 stack-size ablation — how much temporal context does implicit
system identification need?

Trains blind agents (obs_keep_dims=6) with frame-stack sizes k in {1, 3, 5, 20}
on the thesis_v6_hipmdp protocol (k=10 is Stage 6b itself). k=1 is the
memoryless control: a policy that cannot infer dynamics from history at all.

Single seed (7) per k — the ablation trend matters, not per-k variance
(Stage 6a/6b carry the 5-seed variance story).
"""

import subprocess
import sys
from pathlib import Path

STACKS = [1, 3, 5, 20]


def main() -> int:
    stage2_script = Path(__file__).parent / "stage2_meta_rl_reproduction.py"
    if not stage2_script.exists():
        print(f"ERROR: {stage2_script} not found", file=sys.stderr)
        return 1

    for k in STACKS:
        out = f"benchmark_results/stage6c_stack{k}"
        cmd = [
            sys.executable,
            str(stage2_script),
            "--protocol-preset",
            "thesis_v6_hipmdp",
            "--obs-keep-dims",
            "6",
            "--stack-size",
            str(k),
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
            "7",
            "--output-dir",
            out,
        ]
        print(f"=== stack ablation k={k} -> {out} ===")
        rc = subprocess.run(cmd, check=False).returncode
        if rc != 0:
            print(f"k={k} FAILED rc={rc}", file=sys.stderr)
            return rc
    return 0


if __name__ == "__main__":
    sys.exit(main())
