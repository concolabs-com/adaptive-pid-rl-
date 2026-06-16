# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportUnknownParameterType=false, reportUnknownLambdaType=false

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REQUIRED_METRICS = [
    "settling_time_mean",
    "final_abs_error_mean",
    "overshoot_mean",
    "success_rate",
    "settled_rate",
    "iae_mean",
]

METHOD_ORDER = [
    "Stage1_FixedPID_Robust",
    "Stage2_CleanTrained",
    "Stage2_DisturbanceTrained",
]


def _require_columns(df: pd.DataFrame, required: list[str], source_name: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {source_name}: {missing}")


def load_stage1_summary(stage1_summary_path: Path, controller_name: str) -> pd.DataFrame:
    stage1_df = pd.read_csv(stage1_summary_path)
    _require_columns(stage1_df, ["scenario", "controller", *REQUIRED_METRICS], "stage1 summary")

    filtered = stage1_df[stage1_df["controller"] == controller_name].copy()
    if filtered.empty:
        raise ValueError(f"No rows found for controller '{controller_name}' in {stage1_summary_path}.")

    filtered = filtered[["scenario", *REQUIRED_METRICS]].copy()
    filtered["method"] = "Stage1_FixedPID_Robust"
    return filtered


def load_stage2_summary(stage2_dir: Path, method_name: str) -> pd.DataFrame:
    summary_path = stage2_dir / "stage2_eval_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Expected summary file not found: {summary_path}")

    df = pd.read_csv(summary_path)
    _require_columns(df, ["scenario", *REQUIRED_METRICS], f"{method_name} summary")

    df = df[["scenario", *REQUIRED_METRICS]].copy()
    df["method"] = method_name
    return df


def load_protocol_summary(
    summary_path: Path,
    protocol_name: str,
    method_name: str | None = None,
) -> pd.DataFrame:
    if not summary_path.exists():
        raise FileNotFoundError(f"Expected summary file not found: {summary_path}")

    df = pd.read_csv(summary_path)
    if "scenario" not in df.columns:
        df = df.copy()
        df["scenario"] = protocol_name

    normalized = df.copy()
    for metric in REQUIRED_METRICS:
        if metric not in normalized.columns:
            normalized[metric] = np.nan

    columns = ["scenario", *REQUIRED_METRICS]
    if "suite" in normalized.columns:
        columns.insert(1, "suite")
    if "method" in normalized.columns and method_name is None:
        columns.insert(2 if "suite" in normalized.columns else 1, "method")

    normalized = normalized[columns].copy()
    normalized["protocol"] = protocol_name
    if "suite" not in normalized.columns:
        normalized["suite"] = "EvalOnly"
    if "method" not in normalized.columns:
        normalized["method"] = method_name or "Stage2_DisturbanceTrained"
    elif method_name is not None:
        normalized["method"] = method_name

    ordered_columns = ["protocol", "suite", "scenario", "method", *REQUIRED_METRICS]
    return normalized[ordered_columns]


def align_and_stack(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    common_scenarios = set(dfs[0]["scenario"])
    for df in dfs[1:]:
        common_scenarios = common_scenarios.intersection(set(df["scenario"]))

    if not common_scenarios:
        raise ValueError("No overlapping scenarios across the three inputs.")

    scenario_order = sorted(common_scenarios)
    aligned = []
    for df in dfs:
        filtered = df[df["scenario"].isin(scenario_order)].copy()
        filtered = filtered.set_index("scenario").loc[scenario_order].reset_index()
        aligned.append(filtered)

    stacked = pd.concat(aligned, ignore_index=True)
    return stacked


def build_wide_table(long_df: pd.DataFrame) -> pd.DataFrame:
    wide = long_df.set_index(["scenario", "method"])[REQUIRED_METRICS].unstack("method")
    wide.columns = [f"{metric}_{method}" for metric, method in wide.columns]
    wide = wide.reset_index()

    # Add pairwise deltas that are useful in thesis interpretation.
    wide["delta_success_disturbance_vs_clean"] = (
        wide["success_rate_Stage2_DisturbanceTrained"] - wide["success_rate_Stage2_CleanTrained"]
    )
    wide["delta_success_clean_vs_pid"] = (
        wide["success_rate_Stage2_CleanTrained"] - wide["success_rate_Stage1_FixedPID_Robust"]
    )
    wide["delta_settling_disturbance_vs_clean_s"] = (
        wide["settling_time_mean_Stage2_DisturbanceTrained"] - wide["settling_time_mean_Stage2_CleanTrained"]
    )
    wide["delta_final_error_disturbance_vs_clean_m"] = (
        wide["final_abs_error_mean_Stage2_DisturbanceTrained"] - wide["final_abs_error_mean_Stage2_CleanTrained"]
    )

    return wide


def build_master_table(long_df: pd.DataFrame) -> pd.DataFrame:
    wide = long_df.set_index(["protocol", "suite", "scenario", "method"])[REQUIRED_METRICS].unstack("method")
    wide.columns = [f"{metric}_{method}" for metric, method in wide.columns]
    wide = wide.reset_index()

    def metric_col(metric: str, method: str) -> str:
        return f"{metric}_{method}"

    def safe_diff(metric: str, left: str, right: str) -> None:
        left_col = metric_col(metric, left)
        right_col = metric_col(metric, right)
        if left_col in wide.columns and right_col in wide.columns:
            wide[f"delta_{metric}_{left}_minus_{right}"] = wide[left_col] - wide[right_col]

    safe_diff("success_rate", "Stage2_DisturbanceTrained", "Stage2_CleanTrained")
    safe_diff("success_rate", "Stage2_CleanTrained", "Stage1_FixedPID_Robust")
    safe_diff("settling_time_mean", "Stage2_DisturbanceTrained", "Stage2_CleanTrained")
    safe_diff("final_abs_error_mean", "Stage2_DisturbanceTrained", "Stage2_CleanTrained")
    safe_diff("overshoot_mean", "Stage2_DisturbanceTrained", "Stage2_CleanTrained")

    return wide


def plot_stage2_horizon_comparison(long_df: pd.DataFrame, out_path: Path) -> None:
    subset = long_df[
        (long_df["method"] == "Stage2_DisturbanceTrained")
        & (long_df["protocol"].isin(["Stage2_ShortHorizon_DisturbanceEval", "Stage2_LongHorizon_DisturbanceEval"]))
    ].copy()

    if subset.empty:
        return

    protocol_order = ["Stage2_ShortHorizon_DisturbanceEval", "Stage2_LongHorizon_DisturbanceEval"]
    scenario_order = sorted(subset["scenario"].unique())
    metrics = [
        ("success_rate", "Success Rate (%)"),
        ("settling_time_mean", "Settling Time (s)"),
        ("final_abs_error_mean", "Final Abs Error (m)"),
        ("overshoot_mean", "Overshoot (m)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    x = np.arange(len(scenario_order))
    width = 0.36
    colors = {
        "Stage2_ShortHorizon_DisturbanceEval": "#d62728",
        "Stage2_LongHorizon_DisturbanceEval": "#2ca02c",
    }

    for ax, (metric, title) in zip(axes.flatten(), metrics):
        for i, protocol in enumerate(protocol_order):
            protocol_df = subset[subset["protocol"] == protocol].set_index("scenario").reindex(scenario_order)
            values = protocol_df[metric].to_numpy(dtype=float)
            ax.bar(
                x + (i - 0.5) * width,
                values,
                width=width,
                label=protocol.replace("Stage2_", "").replace("_", " "),
                color=colors[protocol],
            )
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(scenario_order, rotation=15, ha="right")
        ax.grid(axis="y", alpha=0.25)

    axes[0, 0].legend(loc="best")
    fig.suptitle("Stage 2 Disturbance Model: Short vs Long Horizon Evaluation", fontsize=14)
    fig.tight_layout(rect=(0.0, 0.02, 1.0, 0.96))
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_standardized_suite_comparison(long_df: pd.DataFrame, out_path: Path, metric: str, ylabel: str) -> None:
    subset = long_df[long_df["protocol"].isin(["FixedAnchors_CleanStress", "RandomDistance_CleanStress"])].copy()
    if subset.empty:
        return

    subset["group"] = subset["protocol"].str.replace("_CleanStress", "", regex=False) + " | " + subset["suite"]
    group_order = ["FixedAnchors | Clean", "FixedAnchors | Stress", "RandomDistance | Clean", "RandomDistance | Stress"]
    method_order = METHOD_ORDER
    colors = {
        "Stage1_FixedPID_Robust": "#ff7f0e",
        "Stage2_CleanTrained": "#1f77b4",
        "Stage2_DisturbanceTrained": "#2ca02c",
    }

    fig, ax = plt.subplots(figsize=(13, 6.8))
    x = np.arange(len(group_order))
    width = 0.23

    for i, method in enumerate(method_order):
        method_df = subset[subset["method"] == method].set_index("group").reindex(group_order)
        values = method_df[metric].to_numpy(dtype=float)
        ax.bar(x + (i - 1) * width, values, width=width, label=method.replace("_", " "), color=colors[method])

    ax.set_xticks(x)
    ax.set_xticklabels(group_order, rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="best")
    ax.set_title(f"Standardized Three-Method Comparison: {ylabel}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def generate_master_plots(long_df: pd.DataFrame, out_dir: Path) -> None:
    plot_stage2_horizon_comparison(
        long_df,
        out_dir / "thesis_stage2_horizon_comparison_panel.png",
    )
    plot_standardized_suite_comparison(
        long_df,
        out_dir / "thesis_standardized_success_rate.png",
        metric="success_rate",
        ylabel="Success Rate (%)",
    )
    plot_standardized_suite_comparison(
        long_df,
        out_dir / "thesis_standardized_settling_time.png",
        metric="settling_time_mean",
        ylabel="Settling Time (s)",
    )
    plot_standardized_suite_comparison(
        long_df,
        out_dir / "thesis_standardized_final_abs_error.png",
        metric="final_abs_error_mean",
        ylabel="Final Abs Error (m)",
    )


def generate_master_thesis_table(args: argparse.Namespace) -> None:
    short_df = load_protocol_summary(
        args.short_horizon_stage2_summary,
        protocol_name="Stage2_ShortHorizon_DisturbanceEval",
        method_name="Stage2_DisturbanceTrained",
    )
    long_df = load_protocol_summary(
        args.long_horizon_stage2_summary,
        protocol_name="Stage2_LongHorizon_DisturbanceEval",
        method_name="Stage2_DisturbanceTrained",
    )
    fixed_df = load_protocol_summary(
        args.fixed_anchor_summary,
        protocol_name="FixedAnchors_CleanStress",
    )
    random_df = load_protocol_summary(
        args.random_distance_summary,
        protocol_name="RandomDistance_CleanStress",
    )

    long_table = pd.concat([short_df, long_df, fixed_df, random_df], ignore_index=True)
    long_table = long_table.sort_values(["protocol", "suite", "scenario", "method"]).reset_index(drop=True)
    master_table = build_master_table(long_table)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    long_path = args.out_dir / "thesis_master_long_summary.csv"
    master_path = args.out_dir / "thesis_master_wide_summary.csv"

    long_table.to_csv(long_path, index=False)
    master_table.to_csv(master_path, index=False)
    generate_master_plots(long_table, args.out_dir)

    print(f"Saved thesis master long summary: {long_path}")
    print(f"Saved thesis master wide summary: {master_path}")
    print(f"Saved thesis plots in: {args.out_dir}")


def plot_three_way_panel(long_df: pd.DataFrame, out_path: Path) -> None:
    scenario_order = sorted(long_df["scenario"].unique())
    method_order = [
        "Stage1_FixedPID_Robust",
        "Stage2_CleanTrained",
        "Stage2_DisturbanceTrained",
    ]

    metrics = [
        ("settling_time_mean", "Settling Time (s)", False),
        ("final_abs_error_mean", "Final Absolute Error (m)", True),
        ("overshoot_mean", "Overshoot (m)", False),
        ("success_rate", "Success Rate (%)", False),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 10.5))
    x = np.arange(len(scenario_order))
    width = 0.24
    offsets = [-width, 0.0, width]
    colors = ["#ff7f0e", "#1f77b4", "#2ca02c"]

    for ax, (metric, title, logy) in zip(axes.flatten(), metrics):
        for method_name, offset, color in zip(method_order, offsets, colors):
            subset = long_df[long_df["method"] == method_name].set_index("scenario").loc[scenario_order]
            label = method_name.replace("_", " ")
            ax.bar(x + offset, subset[metric].to_numpy(), width, label=label, color=color)

        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(scenario_order, rotation=18, ha="right")
        ax.grid(axis="y", alpha=0.25)
        if logy:
            ax.set_yscale("log")

    axes[0, 0].legend(loc="upper right")
    fig.suptitle("Three-Method Comparison: PID vs Clean-Trained RL vs Disturbance-Trained RL", fontsize=14)
    fig.tight_layout(rect=(0.0, 0.02, 1.0, 0.96))
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate three-way thesis comparison artifacts.")
    parser.add_argument(
        "--merge-thesis-results",
        action="store_true",
        default=False,
        help=(
            "Merge short-horizon disturbed eval, long-horizon disturbed eval, "
            "and standardized suite outputs into thesis-master tables."
        ),
    )
    parser.add_argument(
        "--short-horizon-stage2-summary",
        type=Path,
        default=Path(
            "benchmark_results/stage2_disturbance_trained_distance_randomized_seed7_warm500k/stage2_eval_summary.csv"
        ),
        help="Short-horizon disturbed Stage 2 summary CSV.",
    )
    parser.add_argument(
        "--long-horizon-stage2-summary",
        type=Path,
        default=Path("benchmark_results/stage2_warm500k_disturbance_eval_longhorizon_v2/stage2_eval_summary.csv"),
        help="Long-horizon disturbed Stage 2 summary CSV.",
    )
    parser.add_argument(
        "--fixed-anchor-summary",
        type=Path,
        default=Path(
            "benchmark_results/three_method_fixed_anchors_clean_stress_"
            "standardized_v2_warm500k/three_method_distance_summary.csv"
        ),
        help="Fixed-anchor clean/stress three-method summary CSV.",
    )
    parser.add_argument(
        "--random-distance-summary",
        type=Path,
        default=Path(
            "benchmark_results/three_method_random_clean_stress_"
            "standardized_v2_warm500k/three_method_distance_summary.csv"
        ),
        help="Random-distance clean/stress three-method summary CSV.",
    )
    parser.add_argument(
        "--stage1-summary",
        type=Path,
        default=Path("benchmark_results/stage1_fixed_pid_summary.csv"),
        help="Stage 1 benchmark summary CSV.",
    )
    parser.add_argument(
        "--stage1-controller",
        type=str,
        default="Fixed PID (Robust)",
        help="Controller row to use from the Stage 1 summary.",
    )
    parser.add_argument(
        "--stage2-clean-dir",
        type=Path,
        required=False,
        help="Directory containing clean-trained stage2_eval_summary.csv.",
    )
    parser.add_argument(
        "--stage2-disturbance-dir",
        type=Path,
        required=False,
        help="Directory containing disturbance-trained stage2_eval_summary.csv.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("benchmark_results/final_thesis_comparison_three_methods"),
        help="Output directory for merged tables and figures.",
    )
    args = parser.parse_args()

    if args.merge_thesis_results:
        generate_master_thesis_table(args)
        return

    if args.stage2_clean_dir is None or args.stage2_disturbance_dir is None:
        raise ValueError(
            "--stage2-clean-dir and --stage2-disturbance-dir are required unless --merge-thesis-results is used."
        )

    stage1_df = load_stage1_summary(args.stage1_summary, args.stage1_controller)
    clean_df = load_stage2_summary(args.stage2_clean_dir, "Stage2_CleanTrained")
    disturbance_df = load_stage2_summary(args.stage2_disturbance_dir, "Stage2_DisturbanceTrained")

    long_df = align_and_stack([stage1_df, clean_df, disturbance_df])
    wide_df = build_wide_table(long_df)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    long_path = args.out_dir / "three_method_long_summary.csv"
    wide_path = args.out_dir / "three_method_master_summary.csv"
    panel_path = args.out_dir / "three_method_comparison_panel.png"

    long_df.to_csv(long_path, index=False)
    wide_df.to_csv(wide_path, index=False)
    plot_three_way_panel(long_df, panel_path)

    print(f"Saved long summary: {long_path}")
    print(f"Saved master summary: {wide_path}")
    print(f"Saved comparison panel: {panel_path}")


if __name__ == "__main__":
    main()
