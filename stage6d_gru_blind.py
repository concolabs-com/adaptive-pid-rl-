#!/usr/bin/env python3
"""
Stage 6d: RQ3 memory-mechanism ablation — GRU recurrent blind agent.

Completes the three-way memory comparison on the thesis_v6_hipmdp protocol:
  - no memory      : stage6c k=1
  - frame stacking : stage6b (k=10)
  - recurrence     : this script (GRU, stack=1)

Stage 4 previously showed GRU + curriculum value-loss instability under the
v3_safety protocol (spikes at every curriculum phase boundary, 0% success on
2/3 scenarios). This run tests whether that failure was protocol-specific
(governor + hard overshoot termination) or inherent to recurrent training
here. Seeds 7, 21.
"""

import subprocess
import sys
from pathlib import Path


def main() -> int:
    stage2_script = Path(__file__).parent / "stage2_meta_rl_reproduction.py"
    if not stage2_script.exists():
        print(f"ERROR: {stage2_script} not found", file=sys.stderr)
        return 1

    out = "benchmark_results/stage6d_gru_blind"
    cmd = [
        sys.executable,
        str(stage2_script),
        "--protocol-preset",
        "thesis_v6_hipmdp",
        "--obs-keep-dims",
        "6",
        "--recurrent-policy",
        "--recurrent-hidden-size",
        "128",
        "--stack-size",
        "1",
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
        "7,21",
        "--output-dir",
        out,
    ]
    print(f"=== GRU blind (v6) -> {out} ===")
    return subprocess.run(cmd, check=False).returncode


if __name__ == "__main__":
    sys.exit(main())
