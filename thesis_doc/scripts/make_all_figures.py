#!/usr/bin/env python3
"""
Generate ALL thesis result figures from the benchmark_results CSVs.

One place to (re)build every data figure in Chapter 5, so the thesis's figures
are reproducible from committed code + the (git-ignored) result CSVs. Robust to
missing inputs: each figure is wrapped in try/except and skipped with a notice
if its CSV is absent, so the script runs to completion on a partial results dir.

Run: venv/Scripts/python.exe thesis_doc/scripts/make_all_figures.py
Output: thesis_doc/chapter05/figures/*.png  (plus the Ch3 diagrams via make_diagrams.py)
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BR = ROOT / "benchmark_results"
OUT = ROOT / "thesis_doc" / "chapter05" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"figure.dpi": 150, "font.size": 10, "axes.grid": True, "grid.alpha": 0.3})
CTX_C, BLIND_C, FIX_C, AW_C = "#1E88E5", "#43A047", "#E53935", "#8E24AA"

made, skipped = [], []


def fig(name):
    def deco(fn):
        try:
            fn()
            made.append(name)
        except FileNotFoundError as e:
            skipped.append(f"{name} (missing {Path(str(e).split(chr(39))[-2]).name if chr(39) in str(e) else e})")
        except Exception as e:  # noqa: BLE001
            skipped.append(f"{name} ({type(e).__name__}: {e})")

    return deco


def save(figobj, fname):
    figobj.tight_layout()
    figobj.savefig(OUT / fname, bbox_inches="tight")
    plt.close(figobj)


# --- Fig: RQ2 settling, context vs blind, per scenario (with seed sd) ---
@fig("rq2_context_vs_blind_settling")
def _():
    a = pd.read_csv(BR / "stage6a_context_hipmdp" / "aggregate_all_seeds.csv")
    b = pd.read_csv(BR / "stage6b_blind_hipmdp" / "aggregate_all_seeds.csv")
    order = a.sort_values("settling_time_mean")["scenario"].tolist()
    a = a.set_index("scenario").loc[order]
    b = b.set_index("scenario").loc[order]
    x = np.arange(len(order))
    f, ax = plt.subplots(figsize=(11, 4.5))
    ax.bar(
        x - 0.2,
        a["settling_time_mean"],
        0.4,
        yerr=a["settling_time_mean_sd"],
        capsize=3,
        color=CTX_C,
        label="Context (teacher)",
    )
    ax.bar(
        x + 0.2,
        b["settling_time_mean"],
        0.4,
        yerr=b["settling_time_mean_sd"],
        capsize=3,
        color=BLIND_C,
        label="Blind (student)",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Settling time (s)")
    ax.set_title("RQ2 — Context vs Blind settling by scenario (5 seeds, mean ± sd)")
    ax.legend()
    save(f, "fig5_rq2_settling.png")


# --- Fig: training curves (return + value loss), context vs blind, seed 7 ---
@fig("training_curves")
def _():
    f, axes = plt.subplots(1, 2, figsize=(12, 4))
    for tag, c, lab in [("stage6a_context_hipmdp", CTX_C, "Context"), ("stage6b_blind_hipmdp", BLIND_C, "Blind")]:
        df = pd.read_csv(BR / tag / "seed_7" / "training_curve.csv")
        axes[0].plot(df["update"], df["recent_episode_return_mean"], color=c, label=lab, lw=1.3)
        if "value_loss_mean" in df:
            axes[1].plot(df["update"], df["value_loss_mean"], color=c, label=lab, lw=1.3)
    for ph in [30, 61, 91]:  # approx curriculum phase boundaries (122 updates)
        for ax in axes:
            ax.axvline(ph, color="gray", ls=":", lw=0.8)
    axes[0].set_xlabel("PPO update")
    axes[0].set_ylabel("recent episode return")
    axes[0].set_title("Training return")
    axes[0].legend()
    axes[1].set_xlabel("PPO update")
    axes[1].set_ylabel("value loss")
    axes[1].set_title("Value loss (curriculum boundaries dotted)")
    axes[1].legend()
    save(f, "fig5_training_curves.png")


# --- Fig: actuator sweep (settling/overshoot/success) ---
@fig("actuator_sweep")
def _():
    df = pd.read_csv(BR / "actuator_sweep_v6" / "actuator_sweep_raw.csv")
    f, axes = plt.subplots(1, 3, figsize=(15, 4))
    for name, g in df.groupby("controller"):
        g = g.sort_values("actuator")
        axes[0].plot(g["actuator"], g["settling_time_s"], marker="o", ms=3, label=name)
        axes[1].plot(g["actuator"], g["overshoot"], marker="o", ms=3, label=name)
        axes[2].plot(g["actuator"], g["success"] * 100, marker="o", ms=3, label=name)
    for ax, yl in zip(axes, ["Settling (s)", "Overshoot (m)", "Success (%)"]):
        ax.axvspan(0.6, 1.4, alpha=0.08, color="green")
        ax.set_xlabel("Actuator strength κ")
        ax.set_ylabel(yl)
    axes[0].legend(fontsize=8)
    f.suptitle("Actuator sweep (green = training range)")
    save(f, "fig5_actuator_sweep.png")


# --- Fig: mass sweep ---
@fig("mass_sweep")
def _():
    df = pd.read_csv(BR / "mass_sweep" / "mass_sweep_raw.csv")
    f, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, g in df.groupby("controller"):
        g = g.sort_values("mass")
        axes[0].plot(g["mass"], g["settling_time_s"], marker="o", ms=3, label=name)
        axes[1].plot(g["mass"], g["success"] * 100, marker="o", ms=3, label=name)
    for ax, yl in zip(axes, ["Settling (s)", "Success (%)"]):
        ax.axvspan(5, 20, alpha=0.08, color="green")
        ax.set_xlabel("Mass (kg)")
        ax.set_ylabel(yl)
    axes[0].legend(fontsize=8)
    f.suptitle("Mass sweep (green = training range): ~flat vs the steep actuator sweep")
    save(f, "fig5_mass_sweep.png")


# --- Fig: RQ4 probe R^2 over episode time ---
@fig("probe_r2_over_time")
def _():
    df = pd.read_csv(BR / "probe_stage6b" / "probe_r2.csv")
    sub = df[(df.features == "representation") & (df.bucket != "all")].copy()
    sub["t_center"] = (sub["t_lo"] + np.minimum(sub["t_hi"], 600)) / 2 * 0.02
    f, ax = plt.subplots(figsize=(8, 4.5))
    colors = {"mass": "#1E88E5", "actuator": "#E53935", "friction": "#9E9E9E"}
    for tgt, g in sub.groupby("target"):
        g = g.sort_values("t_lo")
        ax.plot(g["t_center"], g["r2"], marker="o", color=colors.get(tgt, "k"), label=tgt)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylim(-0.2, 1.0)
    ax.set_xlabel("Episode time (s)")
    ax.set_ylabel("probe out-of-fold R²")
    ax.set_title(
        "RQ4 — decodability of hidden parameters from the blind policy\n"
        "(mass↔acceleration, actuator↔cruise, friction↔none)"
    )
    ax.legend()
    save(f, "fig5_probe_r2.png")


# --- Fig: gain regimes vs actuator ---
@fig("gain_regimes")
def _():
    ss = pd.read_csv(BR / "gain_analysis" / "steady_state_gains.csv")
    sub = ss[ss.mass == 10.0].sort_values("actuator")
    f, axes = plt.subplots(1, 3, figsize=(14, 4))
    for i, g in enumerate(["kp", "ki", "kd"]):
        for ag, c in [("context", CTX_C), ("blind", BLIND_C)]:
            d = sub[sub.agent == ag]
            axes[i].plot(d["actuator"], d[g], marker="o", color=c, label=ag)
        axes[i].set_xlabel("Actuator κ")
        axes[i].set_ylabel(f"steady-state {g.upper()}")
    axes[0].legend(fontsize=8)
    f.suptitle("Learned gain regimes vs actuator strength (mass = 10 kg): two flat, offset operating points")
    save(f, "fig5_gain_regimes.png")


# --- Fig: anti-windup comparison (no-reset env) ---
@fig("antiwindup_comparison")
def _():
    df = pd.read_csv(BR / "baseline_antiwindup" / "antiwindup_comparison.csv")
    nr = df[df.env_variant == "noreset"]
    order = sorted(nr.scenario.unique())
    modes = ["none", "clamp", "backcalc"]
    x = np.arange(len(order))
    f, ax = plt.subplots(figsize=(9, 4.5))
    for i, m in enumerate(modes):
        d = nr[nr.aw_mode == m].set_index("scenario").reindex(order)
        ax.bar(x + (i - 1) * 0.27, d["settling"], 0.27, label=f"{m}", capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=15, fontsize=8)
    ax.set_ylabel("Settling (s)")
    ax.set_title("No-reset environment: anti-windup (backcalc/clamp) vs none — settling")
    ax.legend(title="anti-windup")
    save(f, "fig5_antiwindup.png")


# --- Fig: pendulum survival by scenario ---
@fig("pendulum_survival")
def _():
    df = pd.read_csv(BR / "pendulum" / "eval" / "pendulum_eval_summary.csv")
    order = df.groupby("scenario").survival.min().sort_values().index.tolist()
    ctrls = df.controller.unique()
    x = np.arange(len(order))
    f, ax = plt.subplots(figsize=(11, 4.5))
    w = 0.8 / len(ctrls)
    for i, c in enumerate(ctrls):
        d = df[df.controller == c].set_index("scenario").reindex(order)
        ax.bar(x + (i - (len(ctrls) - 1) / 2) * w, d["survival"], w, label=c)
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Survival fraction")
    ax.set_ylim(0, 1.05)
    ax.set_title("Pendulum transfer — balance survival by scenario")
    ax.legend(fontsize=8)
    save(f, "fig5_pendulum_survival.png")


# --- Fig: stack-depth ablation ---
@fig("stack_ablation")
def _():
    rows = []
    for k, d in {
        1: "stage6c_stack1",
        3: "stage6c_stack3",
        5: "stage6c_stack5",
        10: "stage6b_blind_hipmdp",
        20: "stage6c_stack20",
    }.items():
        p = BR / d / "seed_7" / "eval_seed_summary.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        for sc in ["Standard", "Heavy Weak Motor", "OOD Weak Motor"]:
            v = df[df.scenario == sc]["settling_time_mean"]
            if len(v):
                rows.append({"k": k, "scenario": sc, "settling": float(v.mean())})
    d = pd.DataFrame(rows)
    f, ax = plt.subplots(figsize=(8, 4.5))
    for sc, g in d.groupby("scenario"):
        g = g.sort_values("k")
        ax.plot(g["k"], g["settling"], marker="o", label=sc)
    ax.set_xlabel("Frame-stack depth k")
    ax.set_ylabel("Settling (s)")
    ax.set_title("RQ3 — stack-depth ablation: flat across k (even k=1 succeeds)")
    ax.legend(fontsize=8)
    save(f, "fig5_stack_ablation.png")


# --- Fig: MRAC config comparison (all 0% — overshoot/settling) ---
@fig("mrac_configs")
def _():
    df = pd.read_csv(BR / "stage4b_mrac_feasible" / "mrac_config_comparison.csv")
    piv = df.pivot_table(index="config", columns="scenario", values="success_rate")
    f, ax = plt.subplots(figsize=(8, 3.5))
    im = ax.imshow(piv.values, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels(piv.columns, rotation=15, fontsize=8)
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(piv.index, fontsize=8)
    for i in range(len(piv.index)):
        for j in range(len(piv.columns)):
            ax.text(j, i, f"{piv.values[i, j]:.0f}%", ha="center", va="center", fontsize=8)
    ax.set_title("MRAC (feasible reference model): success rate — 0% across all configs")
    f.colorbar(im, ax=ax, label="success %")
    save(f, "fig5_mrac.png")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        pass  # decorators already executed at import time
    print(f"\n=== figures written to {OUT} ===")
    for m in made:
        print(f"  OK   {m}")
    for s in skipped:
        print(f"  SKIP {s}")
    print(f"\n{len(made)} made, {len(skipped)} skipped")
