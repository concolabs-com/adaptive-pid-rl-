#!/usr/bin/env python3
"""
Stage 4: MRAC Baseline — MIT Rule Model Reference Adaptive Control.

Classical adaptive PID gain scheduling without any physics context.
No mass or friction observation; adapts purely from tracking error.

Theory reference:
  Åström & Wittenmark, "Adaptive Control", 2nd ed. (1995), Chapter 5.
  Method: MIT rule gradient descent on squared model-following error.

Reference model (desired closed-loop behavior):
  dy_m/dt = -(1 / tau_m) * (y_m - target)

Adaptation law (continuous-time MIT rule, discretised with control_dt):
  e_mrac = pos - y_m              (plant vs reference model gap)
  phi    = [ep, integral_ep, dep_dt]   (PID regressor vector)
  dKp/dt = -gamma_p * e_mrac * ep
  dKi/dt = -gamma_i * e_mrac * integral_ep
  dKd/dt = -gamma_d * e_mrac * dep_dt

Evaluated on the same three scenarios and metrics as Stage 1 and Stage 2
so results can be placed side-by-side in the thesis comparison table.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np
import pandas as pd

from envs.adaptive_suspension import AdaptiveSuspensionEnv

OUTPUT_DIR = Path("benchmark_results/stage4_mrac_baseline")

SCENARIOS = [
    {"name": "Standard", "mass": 10.0, "friction": 1.0},
    {"name": "Heavy and Slippery", "mass": 20.0, "friction": 0.2},
    {"name": "Light and Grippy", "mass": 5.0, "friction": 2.0},
]

# Must match AdaptiveSuspensionEnv constructor defaults exactly.
KP_BASE, KP_DELTA, KP_RANGE = 1.8, 1.0, (0.0, 6.0)
KI_BASE, KI_DELTA, KI_RANGE = 0.7, 0.6, (0.0, 3.0)
KD_BASE, KD_DELTA, KD_RANGE = 0.0, 2.0, (0.0, 5.0)

# MRAC hyperparameters.
TAU_M = 3.0  # reference model time constant (seconds)
GAMMA_P = 0.20  # Kp adaptation rate
GAMMA_I = 0.03  # Ki adaptation rate
GAMMA_D = 0.01  # Kd adaptation rate

# Sigma-modification (Ioannou & Tsakalis, 1986): leakage term added to each
# gain update to prevent unbounded parameter drift.  Pulls gains back toward
# nominal values when the MIT rule would otherwise saturate them.
# Equilibrium: Kp ≈ Kp_base + (gamma * e_mrac * vel) / SIGMA
# SIGMA=0.0 → plain MIT rule (unbounded).  SIGMA~0.5 → moderate damping.
SIGMA = 0.40

# Evaluation settings — match Stage 1 fixed-PID benchmark.
N_EPISODES = 30
TARGET_POS = 5.0
MAX_STEPS = 5000
TOLERANCE = 0.05  # metres
HOLD_STEPS = 25
CONTROL_DT = 0.1  # frame_skip=10, physics_dt=0.01 s → 0.1 s per env.step()
SEEDS = list(range(N_EPISODES))


# ---------------------------------------------------------------------------
# Gain mapping helpers
# ---------------------------------------------------------------------------


def gains_to_action(kp: float, ki: float, kd: float) -> np.ndarray:
    """Map absolute PID gains → normalised env action in [-1, 1]."""
    a0 = float(np.clip((kp - KP_BASE) / KP_DELTA, -1.0, 1.0))
    a1 = float(np.clip((ki - KI_BASE) / KI_DELTA, -1.0, 1.0))
    a2 = float(np.clip((kd - KD_BASE) / KD_DELTA, -1.0, 1.0))
    return np.array([a0, a1, a2], dtype=np.float32)


# ---------------------------------------------------------------------------
# MIT Rule MRAC controller
# ---------------------------------------------------------------------------


class MITRuleMRAC:
    """
    MIT Rule MRAC for online PID gain adaptation.

    Tracks a first-order reference model. Drives Kp/Ki/Kd via gradient
    descent on the squared model-following error without observing mass
    or friction. Mirrors the constraint placed on the Stage 4 RL agent.
    """

    def __init__(
        self,
        target_pos: float,
        tau_m: float = TAU_M,
        gamma_p: float = GAMMA_P,
        gamma_i: float = GAMMA_I,
        gamma_d: float = GAMMA_D,
    ) -> None:
        self.target_pos = float(target_pos)
        self.tau_m = float(tau_m)
        self.gamma_p = float(gamma_p)
        self.gamma_i = float(gamma_i)
        self.gamma_d = float(gamma_d)

        # PID gains initialised to the environment base values.
        self.kp = float(KP_BASE)
        self.ki = float(KI_BASE)
        self.kd = float(KD_BASE)

        # Reference model state.
        self._y_m = 0.0

        # Regressor history.
        self._integral_ep = 0.0
        self._prev_ep = 0.0
        self._prev_pos = 0.0  # for velocity estimate

    def reset_episode(self, initial_pos: float = 0.0) -> None:
        """Reset per-episode integration state. Gains are NOT reset so they
        accumulate knowledge across episodes within a scenario."""
        self._y_m = float(initial_pos)
        self._integral_ep = 0.0
        self._prev_ep = self.target_pos - initial_pos
        self._prev_pos = float(initial_pos)

    def step(self, pos: float, dt: float) -> np.ndarray:
        """
        Observe current position, update gains via MIT rule, return action.

        Sensitivity fix: use vehicle velocity (not position error) as the
        Kp regressor.  The velocity-based regressor has the correct sign in
        all three trajectory phases (approach, overshoot, return), whereas
        the naive ep regressor always pushes Kp toward its upper bound.

        Args:
            pos: current vehicle x-position (obs[0] from env).
            dt:  control period in seconds (CONTROL_DT = 0.1 s).

        Returns:
            Normalised action [kp_norm, ki_norm, kd_norm] for env.step().
        """
        ep = self.target_pos - pos
        vel = (pos - self._prev_pos) / dt if dt > 0.0 else 0.0
        dep_dt = (ep - self._prev_ep) / dt if dt > 0.0 else 0.0
        self._integral_ep += ep * dt
        self._prev_ep = ep
        self._prev_pos = pos

        # Advance first-order reference model one step.
        self._y_m += dt * (-(self._y_m - self.target_pos) / self.tau_m)

        # Model-following error: positive → plant ahead of reference.
        e_mrac = pos - self._y_m

        # Sigma-modification MIT rule (Ioannou & Tsakalis, 1986).
        # Each update = adaptation gradient  +  leakage toward nominal.
        # Leakage term -sigma*(K - K_base) prevents unbounded parameter drift
        # while still allowing adaptation when tracking error is large.
        sigma = SIGMA
        self.kp -= (self.gamma_p * e_mrac * vel + sigma * (self.kp - KP_BASE)) * dt
        self.ki -= (self.gamma_i * e_mrac * self._integral_ep + sigma * (self.ki - KI_BASE)) * dt
        self.kd -= (self.gamma_d * e_mrac * dep_dt + sigma * (self.kd - KD_BASE)) * dt

        # Hard-clip to prevent gains leaving valid bounds.
        self.kp = float(np.clip(self.kp, *KP_RANGE))
        self.ki = float(np.clip(self.ki, *KI_RANGE))
        self.kd = float(np.clip(self.kd, *KD_RANGE))

        return gains_to_action(self.kp, self.ki, self.kd)


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------


def make_env(target_pos: float, max_steps: int, tolerance: float, hold_steps: int) -> AdaptiveSuspensionEnv:
    return AdaptiveSuspensionEnv(
        target_pos=target_pos,
        hold_steps=hold_steps,
        max_episode_steps=max_steps,
        stop_tolerance=tolerance,
        gain_base_kp=KP_BASE,
        gain_base_ki=KI_BASE,
        gain_base_kd=KD_BASE,
        gain_delta_kp=KP_DELTA,
        gain_delta_ki=KI_DELTA,
        gain_delta_kd=KD_DELTA,
        gain_range_kp=KP_RANGE,
        gain_range_ki=KI_RANGE,
        gain_range_kd=KD_RANGE,
    )


def apply_scenario_physics(env: AdaptiveSuspensionEnv, mass: float, friction: float) -> None:
    """Write mass/friction directly into MuJoCo model (body[1] = car, geom[0] = floor)."""
    env.model.body_mass[1] = float(mass)
    env.model.geom_friction[0, 0] = float(friction)
    mujoco.mj_forward(env.model, env.data)


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------


def run_episode(
    env: AdaptiveSuspensionEnv,
    controller: MITRuleMRAC,
    target_pos: float,
    max_steps: int,
    tolerance: float,
    hold_steps: int,
    control_dt: float,
    seed: int | None = None,
) -> dict:
    """Run one episode; returns per-episode metrics dict."""
    reset_kwargs = {"seed": seed} if seed is not None else {}
    obs, _ = env.reset(**reset_kwargs)

    initial_pos = float(obs[0]) if hasattr(obs, "__len__") else float(obs)
    controller.reset_episode(initial_pos=initial_pos)

    positions, errors, kps, kis, kds = [], [], [], [], []
    steps_in_tol = 0
    success = False
    iae = 0.0

    for _ in range(max_steps):
        pos = float(obs[0]) if hasattr(obs, "__len__") else float(obs)
        error = target_pos - pos
        positions.append(pos)
        errors.append(error)
        kps.append(controller.kp)
        kis.append(controller.ki)
        kds.append(controller.kd)
        iae += abs(error) * control_dt

        action = controller.step(pos, control_dt)
        obs, _, terminated, truncated, _ = env.step(action)

        steps_in_tol = (steps_in_tol + 1) if abs(error) < tolerance else 0
        if steps_in_tol >= hold_steps:
            success = True
            break
        if terminated or truncated:
            break

    errors_arr = np.array(errors, dtype=np.float64)
    positions_arr = np.array(positions, dtype=np.float64)

    # Settling time: earliest index from which position stays in tolerance.
    within = np.abs(errors_arr) <= tolerance
    settling_step = int(max_steps)
    for i in range(len(within)):
        if np.all(within[i:]):
            settling_step = i
            break
    settling_time_s = settling_step * control_dt if success else float(max_steps * control_dt)

    overshoot = float(max(0.0, np.max(positions_arr) - target_pos))
    final_abs_error_m = float(abs(errors_arr[-1])) if len(errors_arr) else float("nan")

    return {
        "success": int(success),
        "settling_time_s": settling_time_s,
        "overshoot_m": overshoot,
        "iae": iae,
        "final_abs_error_m": final_abs_error_m,
        "n_steps": len(errors),
        "final_kp": controller.kp,
        "final_ki": controller.ki,
        "final_kd": controller.kd,
    }


# ---------------------------------------------------------------------------
# Scenario evaluation
# ---------------------------------------------------------------------------


def evaluate_scenario(
    scenario: dict,
    seeds: list[int],
    target_pos: float,
    max_steps: int,
    tolerance: float,
    hold_steps: int,
    control_dt: float,
) -> list[dict]:
    env = make_env(target_pos, max_steps, tolerance, hold_steps)
    controller = MITRuleMRAC(target_pos=target_pos)

    # Apply fixed physics for this scenario.  reset() won't change body_mass /
    # geom_friction, so one apply is enough; we re-apply for safety each episode.
    env.reset(seed=0)
    apply_scenario_physics(env, scenario["mass"], scenario["friction"])

    results = []
    for ep, seed in enumerate(seeds):
        apply_scenario_physics(env, scenario["mass"], scenario["friction"])
        result = run_episode(env, controller, target_pos, max_steps, tolerance, hold_steps, control_dt, seed=seed)
        result["episode"] = ep
        result["scenario"] = scenario["name"]
        result["mass"] = scenario["mass"]
        result["friction"] = scenario["friction"]
        result["seed"] = seed
        results.append(result)

        status = "Y" if result["success"] else "N"
        print(
            f"  ep={ep:2d}  settled={status}"
            f"  t={result['settling_time_s']:6.1f}s"
            f"  overshoot={result['overshoot_m']:.3f}m"
            f"  iae={result['iae']:.2f}"
            f"  kp={result['final_kp']:.2f}"
            f"  ki={result['final_ki']:.2f}"
            f"  kd={result['final_kd']:.2f}"
        )

    env.close()
    return results


# ---------------------------------------------------------------------------
# Summary & plots
# ---------------------------------------------------------------------------


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    metrics = ["settling_time_s", "overshoot_m", "iae", "final_abs_error_m", "success"]
    rows = []
    for scenario, grp in df.groupby("scenario"):
        row: dict = {"scenario": scenario, "n_episodes": len(grp)}
        for m in metrics:
            row[f"{m}_mean"] = float(grp[m].mean())
            row[f"{m}_std"] = float(grp[m].std())
        rows.append(row)
    return pd.DataFrame(rows)


def plot_summary(summary_df: pd.DataFrame, output_dir: Path) -> None:
    metrics = [
        ("settling_time_s_mean", "settling_time_s_std", "Settling Time (s)"),
        ("overshoot_m_mean", "overshoot_m_std", "Overshoot (m)"),
        ("iae_mean", "iae_std", "IAE"),
        ("success_mean", "success_std", "Success Rate"),
    ]
    scenarios = summary_df["scenario"].tolist()
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, (col_mean, col_std, label) in zip(axes.flat, metrics):
        vals = summary_df[col_mean].values
        stds = summary_df[col_std].values if col_std in summary_df.columns else np.zeros_like(vals)
        ax.bar(scenarios, vals, yerr=stds, capsize=5)
        ax.set_title(label)
        ax.set_ylabel(label)
        ax.tick_params(axis="x", rotation=12)
    fig.suptitle("Stage 4: MRAC Baseline — MIT Rule Adaptive PID\n(no physics context)", fontsize=11)
    fig.tight_layout()
    fig.savefig(output_dir / "mrac_summary.png", dpi=150)
    plt.close(fig)


def plot_gain_trajectories(df: pd.DataFrame, output_dir: Path) -> None:
    """Show final Kp/Ki/Kd distributions per scenario to verify adaptation."""
    scenarios = df["scenario"].unique()
    fig, axes = plt.subplots(1, len(scenarios), figsize=(5 * len(scenarios), 4), sharey=False)
    if len(scenarios) == 1:
        axes = [axes]
    for ax, sc in zip(axes, scenarios):
        sub = df[df["scenario"] == sc]
        ax.bar(
            ["Kp", "Ki", "Kd"],
            [sub["final_kp"].mean(), sub["final_ki"].mean(), sub["final_kd"].mean()],
            yerr=[sub["final_kp"].std(), sub["final_ki"].std(), sub["final_kd"].std()],
            capsize=5,
        )
        ax.set_title(sc)
        ax.set_ylabel("Gain value")
    fig.suptitle("MRAC: Mean final PID gains per scenario", fontsize=11)
    fig.tight_layout()
    fig.savefig(output_dir / "mrac_final_gains.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results: list[dict] = []
    for scenario in SCENARIOS:
        print(
            f"\n[MRAC] Scenario: {scenario['name']}"
            f"  mass={scenario['mass']} kg"
            f"  friction={scenario['friction']}"
        )
        results = evaluate_scenario(scenario, SEEDS, TARGET_POS, MAX_STEPS, TOLERANCE, HOLD_STEPS, CONTROL_DT)
        all_results.extend(results)

    df = pd.DataFrame(all_results)
    df.to_csv(OUTPUT_DIR / "mrac_raw.csv", index=False)
    print(f"\nRaw results → {OUTPUT_DIR / 'mrac_raw.csv'}")

    summary = summarise(df)
    summary.to_csv(OUTPUT_DIR / "mrac_summary.csv", index=False)
    print(f"Summary     → {OUTPUT_DIR / 'mrac_summary.csv'}")

    plot_summary(summary, OUTPUT_DIR)
    plot_gain_trajectories(df, OUTPUT_DIR)
    print(f"Plots       → {OUTPUT_DIR}/\n")

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
