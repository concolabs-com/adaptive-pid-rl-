import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def build_master_table(stage2_summary: pd.DataFrame, stage1_robust: pd.DataFrame) -> pd.DataFrame:
    merged = stage2_summary.merge(
        stage1_robust,
        on="scenario",
        suffixes=("_stage2", "_stage1"),
    )
    merged["delta_settling_time_s"] = merged["settling_time_mean_stage2"] - merged["settling_time_mean_stage1"]
    merged["delta_final_abs_error_m"] = merged["final_abs_error_mean_stage2"] - merged["final_abs_error_mean_stage1"]
    merged["delta_overshoot_m"] = merged["overshoot_mean_stage2"] - merged["overshoot_mean_stage1"]
    merged["delta_success_rate_pct"] = merged["success_rate_stage2"] - merged["success_rate_stage1"]
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate thesis-ready Stage 1 vs Stage 2 comparison artifacts.")
    parser.add_argument(
        "--stage2-dir",
        type=Path,
        default=Path("benchmark_results/stage2_reward_fix_v3_curriculum_1to5_500k_multi"),
    )
    parser.add_argument(
        "--stage1-summary",
        type=Path,
        default=Path("benchmark_results/stage1_fixed_pid_summary.csv"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("benchmark_results/final_thesis_comparison"),
    )
    args = parser.parse_args()

    stage2_summary_path = args.stage2_dir / "stage2_eval_summary.csv"
    stage2_raw_path = args.stage2_dir / "stage2_eval_raw.csv"

    stage2_summary = pd.read_csv(stage2_summary_path)
    stage2_raw = pd.read_csv(stage2_raw_path)
    stage1_summary = pd.read_csv(args.stage1_summary)

    stage1_robust = stage1_summary[stage1_summary["controller"] == "Fixed PID (Robust)"].copy()
    common_scenarios = sorted(set(stage2_summary["scenario"]).intersection(set(stage1_robust["scenario"])))

    stage2_summary = stage2_summary[stage2_summary["scenario"].isin(common_scenarios)].copy()
    stage1_robust = stage1_robust[stage1_robust["scenario"].isin(common_scenarios)].copy()
    stage2_summary = stage2_summary.set_index("scenario").loc[common_scenarios].reset_index()
    stage1_robust = stage1_robust.set_index("scenario").loc[common_scenarios].reset_index()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    master_table = build_master_table(stage2_summary, stage1_robust)
    master_table.to_csv(out_dir / "stage1_vs_stage2_master_summary.csv", index=False)

    x = np.arange(len(common_scenarios))
    w = 0.36

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    metrics = [
        ("settling_time_mean", "Settling Time (s)", False),
        ("final_abs_error_mean", "Final Absolute Error (m)", True),
        ("overshoot_mean", "Overshoot (m)", False),
        ("success_rate", "Success Rate (%)", False),
    ]

    for ax, (col, title, logy) in zip(axes.flatten(), metrics):
        ax.bar(x - w / 2, stage2_summary[col].to_numpy(), w, label="Stage 2 Meta-RL", color="#1f77b4")
        ax.bar(x + w / 2, stage1_robust[col].to_numpy(), w, label="Stage 1 Fixed PID (Robust)", color="#ff7f0e")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(common_scenarios, rotation=20, ha="right")
        ax.grid(axis="y", alpha=0.25)
        if logy:
            ax.set_yscale("log")

    axes[0, 0].legend(loc="upper right")
    fig.suptitle("Stage 2 Meta-RL vs Stage 1 Fixed PID (Robust)", fontsize=14)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(out_dir / "stage2_vs_stage1_comparison_panel.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    ax.scatter(
        stage2_summary["settling_time_mean"],
        stage2_summary["final_abs_error_mean"],
        s=95,
        color="#1f77b4",
        label="Stage 2 Meta-RL",
    )
    ax.scatter(
        stage1_robust["settling_time_mean"],
        stage1_robust["final_abs_error_mean"],
        s=95,
        color="#ff7f0e",
        label="Stage 1 Fixed PID (Robust)",
    )

    for _, row in stage2_summary.iterrows():
        ax.annotate(
            f"Meta-RL: {row['scenario']}",
            (row["settling_time_mean"], row["final_abs_error_mean"]),
            textcoords="offset points",
            xytext=(6, 5),
            fontsize=8,
        )
    for _, row in stage1_robust.iterrows():
        ax.annotate(
            f"PID: {row['scenario']}",
            (row["settling_time_mean"], row["final_abs_error_mean"]),
            textcoords="offset points",
            xytext=(6, -11),
            fontsize=8,
        )

    ax.set_xlabel("Settling Time (s)")
    ax.set_ylabel("Final Absolute Error (m)")
    ax.set_yscale("log")
    ax.grid(alpha=0.25)
    ax.legend()
    ax.set_title("Speed vs Precision Trade-off")
    fig.tight_layout()
    fig.savefig(out_dir / "stage2_speed_precision_tradeoff.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    grouped = stage2_raw.groupby("scenario")
    overshoot_data = [grouped.get_group(s)["overshoot"].to_numpy() for s in common_scenarios]
    final_err_data = [grouped.get_group(s)["final_abs_error"].to_numpy() for s in common_scenarios]

    axes[0].boxplot(overshoot_data, tick_labels=common_scenarios, showfliers=False)
    axes[0].set_title("Meta-RL Overshoot by Scenario")
    axes[0].set_ylabel("Overshoot (m)")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].boxplot(final_err_data, tick_labels=common_scenarios, showfliers=False)
    axes[1].axhline(0.05, color="red", linestyle="--", linewidth=1.3, label="Success tolerance (0.05 m)")
    axes[1].set_title("Meta-RL Final Abs Error vs Tolerance")
    axes[1].set_ylabel("Final Abs Error (m)")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(out_dir / "stage2_overshoot_and_stop_quality.png", dpi=220)
    plt.close(fig)

    print(f"Saved thesis comparison package to: {out_dir}")


if __name__ == "__main__":
    main()
