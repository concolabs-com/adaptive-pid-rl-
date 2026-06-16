"""
Demo: Meta-RL Adaptive PID Gain Scheduling
Simulates 2-wheeled vehicle position control under 3 physics scenarios.
Compares Fixed PID vs Adaptive Gains (what the Meta-RL agent learns to output).

Physics model: m*xdd = u - b*xd  (mass-damper, friction = viscous damping)
"""

import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, ".")
from utils.pid import PIDController

# --- Config ---
DT = 0.1  # control timestep (matches CONTROL_DT in env)
T_END = 15.0  # episode length (s)
STEPS = int(T_END / DT)
T = np.linspace(0, T_END, STEPS)
TARGET = 1.0  # target position (m)

SCENARIOS = [
    {"name": "Standard", "mass": 10.0, "friction": 1.0, "color": "#2196F3"},
    {"name": "Heavy+Slippery", "mass": 20.0, "friction": 0.2, "color": "#E53935"},
    {"name": "Light+Grippy", "mass": 5.0, "friction": 2.0, "color": "#43A047"},
]

# Single set of gains tuned for Standard — what a fixed PID would use
FIXED_GAINS = {"kp": 4.0, "ki": 0.2, "kd": 3.0}

# Per-scenario optimal gains — what meta-RL learns to schedule
# Heavy+Slippery needs high Kd (dampen oscillation) + low Kp (avoid overshoot with large mass)
# Light+Grippy needs high Kp (fast response) + low Kd (already highly damped by friction)
OPTIMAL_GAINS = {
    "Standard": {"kp": 4.0, "ki": 0.20, "kd": 3.0},
    "Heavy+Slippery": {"kp": 2.0, "ki": 0.08, "kd": 6.5},
    "Light+Grippy": {"kp": 7.5, "ki": 0.40, "kd": 1.2},
}

ADAPTATION_STEPS = 25  # steps over which meta-RL "figures out" the scenario


def simulate(mass, friction, gains_fn):
    """
    Run one episode. gains_fn(step) -> (kp, ki, kd).
    Physics: m*xdd = u - friction*xd
    """
    x, xd = 0.0, 0.0
    pid = PIDController(setpoint=TARGET, output_limits=(-30.0, 30.0))
    pos, err, kp_log, ki_log, kd_log = [], [], [], [], []

    for step in range(STEPS):
        kp, ki, kd = gains_fn(step)
        u, _ = pid.update(x, DT, kp=kp, ki=ki, kd=kd)
        xdd = (u - friction * xd) / mass
        xd += xdd * DT
        x += xd * DT
        pos.append(x)
        err.append(TARGET - x)
        kp_log.append(kp)
        ki_log.append(ki)
        kd_log.append(kd)

    return np.array(pos), np.array(err), np.array(kp_log), np.array(ki_log), np.array(kd_log)


def fixed_schedule(gains):
    kp, ki, kd = gains["kp"], gains["ki"], gains["kd"]
    return lambda _step: (kp, ki, kd)


def adaptive_schedule(target_gains):
    """Linearly ramp from fixed gains to scenario-optimal over ADAPTATION_STEPS.
    Models the meta-RL agent identifying the dynamics and adjusting gains."""
    kp0, ki0, kd0 = FIXED_GAINS["kp"], FIXED_GAINS["ki"], FIXED_GAINS["kd"]
    kpT, kiT, kdT = target_gains["kp"], target_gains["ki"], target_gains["kd"]

    def gains_fn(step):
        alpha = min(step / ADAPTATION_STEPS, 1.0)
        return (
            kp0 + alpha * (kpT - kp0),
            ki0 + alpha * (kiT - ki0),
            kd0 + alpha * (kdT - kd0),
        )

    return gains_fn


# --- Run ---
fixed_res = {}
adaptive_res = {}
for sc in SCENARIOS:
    name = sc["name"]
    fixed_res[name] = simulate(sc["mass"], sc["friction"], fixed_schedule(FIXED_GAINS))
    adaptive_res[name] = simulate(sc["mass"], sc["friction"], adaptive_schedule(OPTIMAL_GAINS[name]))


# --- Plot ---
fig, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=True)
fig.patch.set_facecolor("#FAFAFA")

col_titles = ["Fixed PID (single gain set)", "Meta-RL Adaptive Gain Scheduling"]
row_labels = ["Position (m)", "|Tracking Error| (m)", "PID Gains"]

for col, (results, ctitle) in enumerate(zip([fixed_res, adaptive_res], col_titles)):
    axes[0, col].set_title(ctitle, fontsize=11, fontweight="bold", pad=8)

    for sc in SCENARIOS:
        name, color = sc["name"], sc["color"]
        pos, err, kp, ki, kd = results[name]

        # Row 0: position tracking
        axes[0, col].plot(T, pos, color=color, linewidth=2, label=name)
        axes[0, col].axhline(
            TARGET,
            color="black",
            linestyle="--",
            linewidth=1,
            alpha=0.5,
            label="Target" if (col == 0 and sc == SCENARIOS[0]) else "_",
        )

        # Row 1: absolute tracking error
        axes[1, col].plot(T, np.abs(err), color=color, linewidth=2, label=name)

        # Row 2: Kp and Kd evolution (Ki omitted for clarity)
        axes[2, col].plot(T, kp, color=color, linewidth=2, linestyle="-", label=f"{name} Kp")
        axes[2, col].plot(T, kd, color=color, linewidth=1.5, linestyle="--", label=f"{name} Kd")

    # Shade adaptation window in adaptive column
    if col == 1:
        t_adapt = ADAPTATION_STEPS * DT
        for ax in axes[:, col]:
            ax.axvspan(0, t_adapt, alpha=0.06, color="purple")
        axes[2, col].text(
            t_adapt / 2, axes[2, col].get_ylim()[1] * 0.9, "adapt", ha="center", fontsize=7, color="purple", alpha=0.7
        )

    for row in range(3):
        axes[row, col].grid(alpha=0.25, linestyle=":")
        axes[row, col].set_facecolor("white")

for row, label in enumerate(row_labels):
    axes[row, 0].set_ylabel(label, fontsize=10)

# Legends
axes[0, 0].legend(fontsize=8, loc="lower right")
axes[2, 0].legend(fontsize=7, ncol=2, loc="upper right")
axes[2, 1].legend(fontsize=7, ncol=2, loc="upper right")

# Annotation: add Kp/Kd legend hint
for col in range(2):
    axes[2, col].text(
        0.02, 0.05, "— Kp   -- Kd", transform=axes[2, col].transAxes, fontsize=8, va="bottom", color="gray"
    )

axes[2, 0].set_xlabel("Time (s)", fontsize=10)
axes[2, 1].set_xlabel("Time (s)", fontsize=10)

fig.suptitle("Meta-RL PID Gain Scheduling  |  Vehicle dynamics: m·ẍ = u − b·ẋ", fontsize=12, fontweight="bold", y=1.01)
plt.tight_layout()

out = "demo_visualization.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
plt.show()
