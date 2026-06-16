import argparse
import os
import time
from dataclasses import dataclass
from typing import Any, Optional

import mujoco
import mujoco.viewer
import numpy as np


@dataclass(frozen=True)
class Stage0Config:
    """Configuration for Stage 0 smoke checks."""

    steps: int
    seed: int
    ctrl_low: float
    ctrl_high: float
    mass_scale: float
    friction_scale: float


def load_model_data(xml_path: str) -> tuple[Any, Any]:
    """Create a MuJoCo model and data pair from the XML path."""
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    return model, data


def apply_scenario_overrides(model: Any, mass_scale: float = 1.0, friction_scale: float = 1.0) -> None:
    """Scale mass and friction to emulate a shifted dynamics scenario."""
    if not np.isclose(mass_scale, 1.0):
        # body 0 is world; only scale physical bodies.
        model.body_mass[1:] *= mass_scale
    if not np.isclose(friction_scale, 1.0):
        # Scale all friction components: sliding/torsional/rolling.
        model.geom_friction[:, :] *= friction_scale


def run_random_smoke(
    model: Any,
    data: Any,
    steps: int,
    ctrl_low: float,
    ctrl_high: float,
    seed: int,
) -> dict[str, Any]:
    """Run random-control rollout and report stability and final state summary."""
    rng = np.random.default_rng(seed)
    crash = False
    nan_detected = False
    error_message: Optional[str] = None

    try:
        for _ in range(steps):
            random_ctrl = rng.uniform(ctrl_low, ctrl_high, size=model.nu)
            data.ctrl[:] = random_ctrl
            mujoco.mj_step(model, data)

            if (
                np.isnan(data.qpos).any()
                or np.isnan(data.qvel).any()
                or np.isnan(data.ctrl).any()
                or np.isnan(data.act).any()
            ):
                nan_detected = True
                break
    except Exception as exc:
        crash = True
        error_message = str(exc)

    # Keep output schema stable so logs can be parsed consistently.
    return {
        "crash": crash,
        "error_message": error_message,
        "nan_detected": nan_detected,
        "final_x": float(data.qpos[0]),
        "final_y": float(data.qpos[1]),
        "final_speed": float(np.linalg.norm(data.qvel[:2])),
    }


def check_reset_leakage(model: Any, data: Any) -> dict[str, Any]:
    """Verify reset restores clean initial state after intentional perturbation."""
    # Push state away from reset state, then verify reset clears it.
    data.ctrl[:] = 0.0
    for _ in range(20):
        data.ctrl[:] = 0.5
        mujoco.mj_step(model, data)

    pre_reset_qpos = data.qpos.copy()
    pre_reset_qvel = data.qvel.copy()

    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    post_reset_qpos = data.qpos.copy()
    post_reset_qvel = data.qvel.copy()

    qpos_residual = float(np.max(np.abs(post_reset_qpos - model.qpos0)))
    qvel_residual = float(np.max(np.abs(post_reset_qvel)))

    return {
        "qpos_residual": qpos_residual,
        "qvel_residual": qvel_residual,
        "state_was_perturbed": bool(
            np.max(np.abs(pre_reset_qpos - model.qpos0)) > 1e-10 or np.max(np.abs(pre_reset_qvel)) > 1e-10
        ),
        "reset_clean": bool(qpos_residual < 1e-9 and qvel_residual < 1e-9),
    }


def run_visual_mode(model: Any, data: Any) -> None:
    """Launch interactive viewer with a simple forward-drive motion pattern."""
    print("Two-wheel robot viewer started. Press Ctrl+C in terminal to stop.")

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                # Constant drive with a tiny oscillation so motion is visible.
                drive = 1.0 + 0.15 * np.sin(data.time * 2.0)
                data.ctrl[0] = drive
                data.ctrl[1] = drive

                mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(model.opt.timestep)
    except KeyboardInterrupt:
        print("Stopping viewer...")


def run_stage0_checks(xml_path: str, config: Stage0Config) -> None:
    """Execute the full Stage 0 checklist and print an easy-to-read report."""
    model, data = load_model_data(xml_path)

    print("Running Stage 0 sanity checks...")

    reset_result = check_reset_leakage(model, data)

    # Base scenario uses default dynamics.
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    base_result = run_random_smoke(
        model,
        data,
        steps=config.steps,
        ctrl_low=config.ctrl_low,
        ctrl_high=config.ctrl_high,
        seed=config.seed,
    )

    # Shifted scenario applies mass/friction changes with the same random sequence.
    model_shift, data_shift = load_model_data(xml_path)
    apply_scenario_overrides(
        model_shift,
        mass_scale=config.mass_scale,
        friction_scale=config.friction_scale,
    )
    mujoco.mj_resetData(model_shift, data_shift)
    mujoco.mj_forward(model_shift, data_shift)
    shift_result = run_random_smoke(
        model_shift,
        data_shift,
        steps=config.steps,
        ctrl_low=config.ctrl_low,
        ctrl_high=config.ctrl_high,
        seed=config.seed,
    )

    traj_delta = abs(base_result["final_x"] - shift_result["final_x"]) + abs(
        base_result["final_y"] - shift_result["final_y"]
    )

    print("\n=== Stage 0 Report ===")
    print(f"Reset state was perturbed before reset: {reset_result['state_was_perturbed']}")
    print(f"Reset clean (no leakage): {reset_result['reset_clean']}")
    print(f"Reset qpos residual: {reset_result['qpos_residual']:.3e}")
    print(f"Reset qvel residual: {reset_result['qvel_residual']:.3e}")
    print(
        f"Base scenario crash: {base_result['crash']}, NaN: {base_result['nan_detected']}, "
        f"error: {base_result['error_message']}"
    )
    print(
        f"Shift scenario crash: {shift_result['crash']}, NaN: {shift_result['nan_detected']}, "
        f"error: {shift_result['error_message']}"
    )
    print(f"Trajectory delta (base vs shift): {traj_delta:.6f}")
    print("Done.")


def build_parser() -> argparse.ArgumentParser:
    """Define CLI flags for both viewer and Stage 0 modes."""
    parser = argparse.ArgumentParser(description="Simple MuJoCo visualizer and Stage 0 sanity runner.")
    parser.add_argument("--mode", choices=["visual", "stage0"], default="visual")
    parser.add_argument("--steps", type=int, default=2000, help="Smoke-test step count in stage0 mode.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ctrl-low", type=float, default=-1.0)
    parser.add_argument("--ctrl-high", type=float, default=1.0)
    parser.add_argument("--mass-scale", type=float, default=1.0)
    parser.add_argument("--friction-scale", type=float, default=1.0)
    return parser


def main() -> None:
    """Entry point for ad-hoc visualization or Stage 0 sanity validation."""
    args = build_parser().parse_args()

    xml_path = os.path.join("envs", "assets", "car_model.xml")

    if args.mode == "visual":
        model, data = load_model_data(xml_path)
        run_visual_mode(model, data)
        return

    config = Stage0Config(
        steps=args.steps,
        seed=args.seed,
        ctrl_low=args.ctrl_low,
        ctrl_high=args.ctrl_high,
        mass_scale=args.mass_scale,
        friction_scale=args.friction_scale,
    )
    run_stage0_checks(xml_path, config)


if __name__ == "__main__":
    main()
