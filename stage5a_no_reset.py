#!/usr/bin/env python3
"""
Stage 5a-NR: Context-aware Meta-RL WITHOUT integral reset.

Same as Stage 5a but brake_integral_reset_enabled=False. The agent must learn
to prevent integral windup through gain scheduling alone:
  - Reduce Ki during approach to stop integral accumulating
  - Raise Kd near target to overcome residual integral
  - Fixed PID cannot do this — will overshoot on 8-10m targets

This is the true test of adaptive gain scheduling. With integral reset disabled,
fixed PID at base gains will fail; RL must discover the Ki-suppression strategy.

Observation (8-dim): [pos, vel, err, kp, ki, kd, mass_scale, friction_scale]
Policy:              MLP, stack_size=10
Gains:               Kp [0.0,3.6], Ki [0.0,1.0], Kd [0.0,8.0]
Target:              4-10m (curriculum)
Timesteps:           3,000,000
Output:              benchmark_results/stage5a_no_reset/
"""

import subprocess
import sys
from pathlib import Path

OUTPUT_DIR = "benchmark_results/stage5a_no_reset"

ARGS = [
    "--protocol-preset",
    "thesis_v5_no_reset",
    "--stack-size",
    "10",
    "--total-timesteps",
    "3000000",
    "--num-envs",
    "4",
    "--num-steps",
    "2048",
    "--ent-coef",
    "0.01",
    "--curriculum-enabled",
    "--curriculum-spec",
    "1:3:0.25,1:5:0.25,1:7:0.25,1:10:0.25",
    "--output-dir",
    OUTPUT_DIR,
    "--seeds",
    "7",
]


def main() -> int:
    stage2_script = Path(__file__).parent / "stage2_meta_rl_reproduction.py"
    if not stage2_script.exists():
        print(f"ERROR: {stage2_script} not found", file=sys.stderr)
        return 1

    cmd = [sys.executable, str(stage2_script)] + ARGS

    print("=" * 65)
    print("STAGE 5a-NR: Context RL — No Integral Reset")
    print("  obs     : 8-dim [pos, vel, err, kp, ki, kd, mass, friction]")
    print("  gains   : Kp[0,3.6]  Ki[0,1.0]  Kd[0,8.0]")
    print("  target  : 4-10m curriculum")
    print("  reset   : brake_integral_reset DISABLED")
    print("  steps   : 3,000,000")
    print(f"  output  : {OUTPUT_DIR}")
    print("=" * 65)
    print()

    result = subprocess.run(cmd, check=False)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
