#!/usr/bin/env python3
"""
Generate conceptual / analytical diagrams for Chapters 1, 2, 3, 4, 6.

These are figures that can be authored from first principles — block diagrams,
schematics, and plots of the *equations* in the text (terminal-speed ODE, PPO
clip, windup accumulation, reward shaping) — as opposed to the experiment-data
figures in make_all_figures.py. No benchmark_results CSV is read here, so every
figure is reproducible from this file alone and carries no measurement risk.

Each figure is written under its OWN chapter folder:
    thesis_doc/chapterNN/figures/figN_*.png

Run: venv/Scripts/python.exe thesis_doc/scripts/make_chapter_diagrams.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "font.size": 10,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.axisbelow": True,
    }
)

# shared palette (matches make_all_figures.py)
CTX_C, BLIND_C, FIX_C, AW_C = "#1E88E5", "#43A047", "#E53935", "#8E24AA"
INK, MUTE, PANEL = "#222222", "#666666", "#ECEFF1"

made, skipped = [], []


def figdir(chapter: str) -> Path:
    d = ROOT / "thesis_doc" / f"chapter{chapter}" / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save(fig, chapter: str, fname: str):
    out = figdir(chapter) / fname
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig(chapter: str, fname: str):
    """Decorator: run the builder, record success/failure, save under chapter."""

    def deco(fn):
        try:
            f = fn()
            save(f, chapter, fname)
            made.append(f"ch{chapter}/{fname}")
        except Exception as e:  # noqa: BLE001
            skipped.append(f"ch{chapter}/{fname} ({type(e).__name__}: {e})")

    return deco


def box(ax, cx, cy, w, h, text, fc=PANEL, ec=INK, fs=9, lw=1.4, bold=False, tc=INK):
    """Rounded box centred at (cx, cy)."""
    p = FancyBboxPatch(
        (cx - w / 2, cy - h / 2),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(p)
    ax.text(
        cx,
        cy,
        text,
        ha="center",
        va="center",
        fontsize=fs,
        color=tc,
        fontweight="bold" if bold else "normal",
        zorder=5,
    )
    return (cx, cy, w, h)


def arrow(ax, p_from, p_to, text="", color=INK, ls="-", lw=1.4, rad=0.0, fs=8, dx=0.0, dy=0.12):
    a = FancyArrowPatch(
        p_from,
        p_to,
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=lw,
        color=color,
        linestyle=ls,
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=2,
        shrinkB=2,
        zorder=4,
    )
    ax.add_patch(a)
    if text:
        mx, my = (p_from[0] + p_to[0]) / 2 + dx, (p_from[1] + p_to[1]) / 2 + dy
        ax.text(mx, my, text, ha="center", va="center", fontsize=fs, color=color)


def clean(ax, xlim, ylim):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")


# ======================================================================
# CHAPTER 1
# ======================================================================


@fig("01", "fig1_1_graphical_abstract.png")
def _():
    f, ax = plt.subplots(figsize=(9.2, 4.8))
    clean(ax, (0, 12), (0, 7))
    ax.text(
        6,
        6.6,
        "Online PID gain scheduling under hidden dynamics",
        ha="center",
        fontsize=12,
        fontweight="bold",
        color=INK,
    )

    # hidden parameters
    box(
        ax,
        1.6,
        5.2,
        2.6,
        1.2,
        "Hidden parameters  ψ\nmass m · friction μ · actuator κ\n(drawn per episode)",
        fc="#FFF3E0",
        ec="#FB8C00",
        fs=8,
    )
    # plant
    box(ax, 1.6, 2.2, 2.6, 1.2, "Plant\n(MuJoCo vehicle)\nstate x, ẋ, e", fc=PANEL, fs=9)
    arrow(ax, (1.6, 4.6), (1.6, 2.8), text="sets dynamics", color="#FB8C00", dx=-1.15, dy=0)

    # controller stack (center)
    box(ax, 5.6, 3.6, 2.4, 1.0, "RL policy\n(outer loop, 50 Hz)", fc="#E3F2FD", ec=CTX_C, fs=9, bold=True)
    box(ax, 5.6, 2.0, 2.4, 0.9, "Gain map\na → Kp, Ki, Kd", fc=PANEL, fs=9)
    box(ax, 8.9, 2.0, 2.4, 0.9, "PID\n(inner loop, 500 Hz)", fc=PANEL, fs=9)

    arrow(ax, (5.6, 3.1), (5.6, 2.45), text="action a ∈ [-1,1]³", color=INK, dx=1.25, dy=0)
    arrow(ax, (6.8, 2.0), (7.7, 2.0))
    # command path: PID -> down -> left along the bottom -> up into the plant
    ax.plot([8.9, 8.9], [1.55, 0.7], color=INK, lw=1.4, zorder=1)
    ax.plot([8.9, 1.6], [0.7, 0.7], color=INK, lw=1.4, zorder=1)
    arrow(ax, (1.6, 0.7), (1.6, 1.6), color=INK)
    ax.text(5.2, 0.5, "motor command u", ha="center", fontsize=8, color=INK)
    # observation feedback: plant -> policy
    arrow(ax, (2.9, 2.5), (4.4, 3.6), text="observation", color=MUTE, dx=0, dy=0.18)

    # teacher vs student
    box(
        ax,
        5.6,
        5.6,
        2.4,
        1.0,
        "TEACHER (context-aware)\nobserves ψ directly",
        fc="#E3F2FD",
        ec=CTX_C,
        fs=8,
        tc=CTX_C,
        bold=True,
    )
    box(
        ax,
        8.9,
        5.6,
        2.4,
        1.0,
        "STUDENT (blind)\ninfers ψ from history",
        fc="#E8F5E9",
        ec=BLIND_C,
        fs=8,
        tc=BLIND_C,
        bold=True,
    )
    arrow(ax, (5.6, 5.1), (5.6, 4.1), color=CTX_C, ls="--")
    arrow(ax, (8.9, 5.1), (6.4, 4.0), color=BLIND_C, ls="--", text="implicit system ID", dx=1.0, dy=0.0, rad=0.1)
    return f


@fig("01", "fig1_2_rq_roadmap.png")
def _():
    f, ax = plt.subplots(figsize=(9.0, 3.6))
    clean(ax, (0, 12), (0, 5))
    ax.text(
        6,
        4.6,
        "Four research questions: from *whether* to *how*",
        ha="center",
        fontsize=12,
        fontweight="bold",
        color=INK,
    )

    box(ax, 2.0, 3.4, 3.2, 1.0, "RQ1 — Feasibility\nDoes it work + generalize?", fc="#E3F2FD", ec=CTX_C)
    box(ax, 6.0, 3.4, 3.2, 1.0, "RQ2 — Privileged context\nCost of inferring vs being told?", fc="#E3F2FD", ec=CTX_C)
    box(ax, 2.0, 1.4, 3.2, 1.0, "RQ3 — Memory\nHow much history is needed?", fc="#E8F5E9", ec=BLIND_C)
    box(ax, 6.0, 1.4, 3.2, 1.0, "RQ4 — Representation\nWhat does the policy encode?", fc="#E8F5E9", ec=BLIND_C)

    ax.text(4.0, 4.0, "WHETHER  +  COST", ha="center", fontsize=8, color=MUTE, style="italic")
    ax.text(4.0, 0.55, "MECHANISM", ha="center", fontsize=8, color=MUTE, style="italic")
    arrow(ax, (3.6, 3.4), (4.4, 3.4))
    arrow(ax, (3.6, 1.4), (4.4, 1.4))
    arrow(ax, (2.0, 2.9), (2.0, 1.9), color=MUTE, rad=0)
    arrow(ax, (6.0, 2.9), (6.0, 1.9), color=MUTE, rad=0)

    box(
        ax,
        10.4,
        2.4,
        2.6,
        1.7,
        "Unifying thread:\nidentifiability of\nthe plant explains\nboth power and\nlimits (Ch 6)",
        fc="#FFF8E1",
        ec="#F9A825",
        fs=8,
    )
    arrow(ax, (7.6, 3.0), (9.1, 2.7), color="#F9A825")
    arrow(ax, (7.6, 1.6), (9.1, 2.1), color="#F9A825")
    return f


# ======================================================================
# CHAPTER 2
# ======================================================================


@fig("02", "fig2_1_pid_block.png")
def _():
    f, ax = plt.subplots(figsize=(9.0, 3.6))
    clean(ax, (0, 12), (0, 5))
    ax.text(6, 4.6, "PID control loop", ha="center", fontsize=12, fontweight="bold", color=INK)

    ax.add_patch(plt.Circle((1.6, 2.6), 0.28, fc="white", ec=INK, lw=1.4, zorder=5))
    ax.text(1.6, 2.6, "−", ha="center", va="center", fontsize=14, zorder=6)
    ax.text(0.5, 2.6, "r(t)", ha="center", fontsize=10)
    arrow(ax, (0.8, 2.6), (1.3, 2.6))
    arrow(ax, (1.9, 2.6), (2.7, 2.6), text="e", dy=0.2)

    box(ax, 3.6, 3.7, 1.7, 0.7, "$K_p\\,e$", fc=PANEL, fs=11)
    box(ax, 3.6, 2.6, 1.7, 0.7, "$K_i\\int e\\,dt$", fc=PANEL, fs=11)
    box(ax, 3.6, 1.5, 1.7, 0.7, "$K_d\\,\\dot e$", fc=PANEL, fs=11)
    for yy in (3.7, 2.6, 1.5):
        arrow(ax, (2.75, 2.6), (2.75, yy), color=MUTE, rad=0) if yy != 2.6 else None
        arrow(ax, (2.75, yy), (2.75, yy), color=MUTE)
    arrow(ax, (2.7, 2.6), (2.75, 2.6))
    ax.plot([2.75, 2.75], [1.5, 3.7], color=MUTE, lw=1.2, zorder=1)
    for yy in (3.7, 2.6, 1.5):
        arrow(ax, (2.75, yy), (2.75, yy), color=MUTE)

    ax.add_patch(plt.Circle((5.4, 2.6), 0.28, fc="white", ec=INK, lw=1.4, zorder=5))
    ax.text(5.4, 2.6, "+", ha="center", va="center", fontsize=13, zorder=6)
    for yy in (3.7, 2.6, 1.5):
        arrow(ax, (4.45, yy), (5.2, 2.6), color=MUTE, rad=0)

    arrow(ax, (5.7, 2.6), (6.7, 2.6), text="u(t)", dy=0.2)
    box(ax, 7.7, 2.6, 1.8, 0.9, "Saturation\n$u_{sat}$", fc="#FFEBEE", ec=FIX_C, fs=9)
    arrow(ax, (8.6, 2.6), (9.4, 2.6))
    box(ax, 10.4, 2.6, 1.8, 0.9, "Plant\nP(s)", fc=PANEL, fs=10)
    arrow(ax, (10.4, 2.15), (10.4, 0.7), color=INK)
    ax.text(10.4, 0.45, "y(t)", ha="center", fontsize=10)
    ax.plot([10.4, 1.6], [0.7, 0.7], color=INK, lw=1.2, zorder=1)
    arrow(ax, (1.6, 0.7), (1.6, 2.3), color=INK)
    ax.text(5.8, 0.5, "feedback", ha="center", fontsize=8, color=MUTE)
    return f


@fig("02", "fig2_2_windup.png")
def _():
    f, axes = plt.subplots(3, 1, figsize=(7.6, 5.6), sharex=True)
    t = np.linspace(0, 30, 600)
    # synthetic illustrative approach-then-overshoot
    e = 2.5 * np.ones_like(t)
    e = np.where(t < 11, 2.5 - 0.0 * t, e)
    e[t >= 11] = np.maximum(-4.5 * np.exp(-(t[t >= 11] - 11) / 4) + 0.0, -4.5) * 0 + (
        2.5 - 6.5 * (1 - np.exp(-(t[t >= 11] - 11) / 3))
    )
    e = np.clip(e, -3.5, 2.6)
    Ki = 0.7
    integral = np.cumsum(np.where(t < 11, Ki * 2.5 * (t[1] - t[0]), Ki * e * (t[1] - t[0])))
    u_raw = 1.2 + 0.3 * np.sin(t)  # nominally above sat during approach
    u_raw = np.where(t < 11, 3.0, integral * 0.0 + np.clip(2.5 - 0.5 * (t - 11), -1.6, 3.0))
    u_sat = np.clip(u_raw, -1, 1)

    axes[0].plot(t, e, color=FIX_C, lw=2)
    axes[0].axhline(0, color=MUTE, lw=0.8)
    axes[0].set_ylabel("error e(t) [m]")
    axes[0].annotate(
        "error one sign\nfor ~10 s (approach)",
        (5, 2.5),
        (4, 0.2),
        fontsize=8,
        color=INK,
        arrowprops=dict(arrowstyle="->", color=MUTE),
    )
    axes[0].annotate(
        "overshoot\n(sign reversal)",
        (15, -2.5),
        (18, -1.0),
        fontsize=8,
        color=INK,
        arrowprops=dict(arrowstyle="->", color=MUTE),
    )

    axes[1].plot(t, integral, color="#5E35B1", lw=2)
    axes[1].axhline(1, color=MUTE, ls=":", lw=1)
    axes[1].set_ylabel("integral  $I=K_i\\!\\int e$")
    axes[1].annotate(
        "$I\\approx17.5$ — far beyond\nactuator range [-1,1]",
        (10.5, integral[t <= 10.5][-1]),
        (1.5, 12),
        fontsize=8,
        color=INK,
        arrowprops=dict(arrowstyle="->", color=MUTE),
    )

    axes[2].plot(t, u_raw, color=MUTE, lw=1.2, ls="--", label="commanded u")
    axes[2].plot(t, u_sat, color=FIX_C, lw=2, label="saturated $u_{sat}$")
    axes[2].axhspan(-1, 1, color="#C8E6C9", alpha=0.4)
    axes[2].set_ylabel("command u")
    axes[2].set_xlabel("time [s]")
    axes[2].legend(fontsize=8, loc="upper right")
    axes[2].text(2, 1.15, "actuator saturated throughout approach", fontsize=8, color="#2E7D32")
    axes[0].set_title("Integral windup on an input-saturated approach (schematic)", fontsize=11, fontweight="bold")
    for a in axes:
        a.grid(alpha=0.3)
    f.tight_layout()
    return f


@fig("02", "fig2_3_antiwindup_block.png")
def _():
    f, ax = plt.subplots(figsize=(8.6, 3.8))
    clean(ax, (0, 12), (0, 5))
    ax.text(6, 4.6, "Back-calculation anti-windup", ha="center", fontsize=12, fontweight="bold", color=INK)
    box(ax, 2.0, 3.2, 1.9, 0.8, "$K_i$ / s\nintegrator", fc=PANEL, fs=9)
    ax.add_patch(plt.Circle((4.4, 3.2), 0.26, fc="white", ec=INK, lw=1.4, zorder=5))
    ax.text(4.4, 3.2, "+", ha="center", va="center", fontsize=13, zorder=6)
    box(ax, 6.6, 3.2, 1.9, 0.8, "Saturation", fc="#FFEBEE", ec=FIX_C, fs=9)
    arrow(ax, (0.6, 3.2), (1.05, 3.2), text="e", dy=0.25)
    arrow(ax, (2.95, 3.2), (4.1, 3.2))
    arrow(ax, (4.7, 3.2), (5.65, 3.2), text="u", dy=0.22)
    arrow(ax, (7.55, 3.2), (8.8, 3.2), text="$u_{sat}$", dy=0.22)
    # back-calc loop
    ax.add_patch(plt.Circle((6.6, 1.5), 0.26, fc="white", ec=INK, lw=1.4, zorder=5))
    ax.text(6.6, 1.5, "−", ha="center", va="center", fontsize=14, zorder=6)
    ax.plot([8.2, 8.2], [3.2, 1.5], color=AW_C, lw=1.4)
    arrow(ax, (8.2, 1.5), (6.86, 1.5), color=AW_C)
    ax.plot([5.5, 5.5], [3.2, 1.5], color=AW_C, lw=1.4)
    arrow(ax, (5.5, 1.5), (6.34, 1.5), color=AW_C)
    box(ax, 4.4, 1.5, 1.5, 0.7, "$1/T_t$", fc="#F3E5F5", ec=AW_C, fs=10)
    arrow(ax, (6.34, 1.5), (6.34, 1.5), color=AW_C)
    ax.plot([6.6, 6.6], [1.24, 1.0], color=AW_C, lw=1.4)
    ax.plot([6.6, 4.4], [1.0, 1.0], color=AW_C, lw=1.4)
    arrow(ax, (4.4, 1.0), (4.4, 1.15), color=AW_C)
    ax.plot([3.65, 4.4], [1.5, 1.5], color=AW_C, lw=0)  # spacer
    arrow(ax, (4.4, 1.85), (4.4, 2.8), color=AW_C, text="bleed integrator", dx=-1.2, dy=0)
    ax.text(7.0, 0.9, "tracking term $(u_{sat}-u)/T_t$ → 0 when unsaturated", ha="center", fontsize=8, color=AW_C)
    return f


@fig("02", "fig2_4_ppo_clip.png")
def _():
    f, axes = plt.subplots(1, 2, figsize=(8.8, 3.8), sharey=True)
    r = np.linspace(0, 2, 400)
    eps = 0.2
    for ax, A, title in [(axes[0], 1.0, "Advantage $A_t>0$"), (axes[1], -1.0, "Advantage $A_t<0$")]:
        unclipped = r * A
        clipped = np.clip(r, 1 - eps, 1 + eps) * A
        Lclip = np.minimum(unclipped, clipped) if A > 0 else np.maximum(unclipped, clipped)
        # PPO objective is min(r*A, clip*A); plot that
        Lclip = np.minimum(r * A, np.clip(r, 1 - eps, 1 + eps) * A)
        ax.plot(r, r * A, color=MUTE, ls="--", lw=1.3, label="$r_tA_t$ (unclipped)")
        ax.plot(r, Lclip, color=CTX_C, lw=2.4, label="$L^{CLIP}$")
        ax.axvline(1 - eps, color=FIX_C, ls=":", lw=1)
        ax.axvline(1 + eps, color=FIX_C, ls=":", lw=1)
        ax.axvline(1.0, color=MUTE, lw=0.8)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("probability ratio $r_t(\\theta)$")
        ax.legend(fontsize=8, loc="best")
        ax.grid(alpha=0.3)
        ax.text(1 + eps, ax.get_ylim()[0], "  $1+\\epsilon$", fontsize=8, color=FIX_C, va="bottom")
        ax.text(1 - eps, ax.get_ylim()[0], "$1-\\epsilon$  ", fontsize=8, color=FIX_C, va="bottom", ha="right")
    axes[0].set_ylabel("objective")
    f.suptitle(
        "PPO clipped surrogate: gradient vanishes once the ratio leaves $[1-\\epsilon,\\,1+\\epsilon]$",
        fontsize=11,
        fontweight="bold",
    )
    f.tight_layout()
    return f


@fig("02", "fig2_5_identifiability.png")
def _():
    f, axes = plt.subplots(1, 2, figsize=(9.0, 3.9), sharey=True)
    t = np.linspace(0, 6, 400)
    # Panel A: vary mass -> same vmax, different tau_v (acceleration transient)
    vmax = 0.5
    for m, c in [(5, "#90CAF9"), (10, CTX_C), (20, "#0D47A1")]:
        tau = 0.15 * (m / 10)
        axes[0].plot(t, vmax * (1 - np.exp(-t / tau)), color=c, lw=2, label=f"m = {m} kg")
    axes[0].set_title("Mass → acceleration transient", fontsize=10)
    axes[0].set_xlabel("time [s]")
    axes[0].set_ylabel("velocity v(t) [m/s]")
    axes[0].legend(fontsize=8)
    axes[0].axvspan(0, 1.2, color="#FFF3E0", alpha=0.6)
    axes[0].text(0.6, 0.07, "excited\nhere", fontsize=8, color="#E65100", ha="center")

    # Panel B: vary actuator -> different vmax (cruise level)
    for k, c in [(0.6, "#A5D6A7"), (1.0, BLIND_C), (1.4, "#1B5E20")]:
        axes[1].plot(t, 0.5 * k * (1 - np.exp(-t / 0.15)), color=c, lw=2, label=f"κ = {k}")
    axes[1].set_title("Actuator strength → cruise speed", fontsize=10)
    axes[1].set_xlabel("time [s]")
    axes[1].legend(fontsize=8)
    axes[1].axvspan(2.5, 6, color="#E8F5E9", alpha=0.6)
    axes[1].text(4.2, 0.12, "excited here", fontsize=8, color="#1B5E20", ha="center")
    f.suptitle(
        "Identifiability: each hidden parameter is excited in a different phase  ($v(t)=v_{max}(1-e^{-t/\\tau_v})$)",
        fontsize=10.5,
        fontweight="bold",
    )
    f.tight_layout()
    return f


# ======================================================================
# CHAPTER 3
# ======================================================================


@fig("03", "fig3_5_terminal_speed.png")
def _():
    f, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
    # settling ~ d / vmax ; vmax independent of mass, prop to kappa
    d = 5.0
    mass = np.linspace(5, 50, 100)
    ts_mass = d / 0.5 + 0.15 * (mass / 10)  # tiny transient term
    axes[0].plot(mass, ts_mass, color=FIX_C, lw=2.2)
    axes[0].set_title("Settling vs mass — nearly flat (~11%)", fontsize=10)
    axes[0].set_xlabel("mass m [kg]")
    axes[0].set_ylabel("settling time ≈ d/$v_{max}$ + O($\\tau_v$) [s]")
    axes[0].axvspan(5, 20, color="#E8F5E9", alpha=0.5)
    axes[0].text(12.5, ts_mass.min() + 0.2, "training\nrange", ha="center", fontsize=8, color="#2E7D32")

    kap = np.linspace(0.5, 2.0, 100)
    ts_act = d / (0.5 * kap) + 0.15
    axes[1].plot(kap, ts_act, color=CTX_C, lw=2.2)
    axes[1].set_title("Settling vs actuator κ — steep (~3.7×)", fontsize=10)
    axes[1].set_xlabel("actuator strength κ")
    axes[1].axvspan(0.6, 1.4, color="#E8F5E9", alpha=0.5)
    axes[1].text(1.0, ts_act.max() * 0.7, "training\nrange", ha="center", fontsize=8, color="#2E7D32")
    f.suptitle(
        "Why actuator strength discriminates and mass does not  ($v_{max}\\propto\\kappa$, mass only in $\\tau_v$)",
        fontsize=10.5,
        fontweight="bold",
    )
    f.tight_layout()
    return f


@fig("03", "fig3_6_reward_shaping.png")
def _():
    f, ax = plt.subplots(figsize=(8.4, 4.2))
    x = np.linspace(0, 6, 500)
    xt = 5.0
    e = xt - x
    brake = np.abs(e) < 2.0
    distance = np.where(brake, 0, -0.75 * np.abs(e) * 0.1)
    overshoot = -2.0 * np.maximum(0, x - xt)
    ax.axvspan(xt - 2, xt, color="#FFF8E1", alpha=0.7, label="braking zone |e|<2 m")
    ax.axvline(xt, color=MUTE, ls="--", lw=1)
    ax.plot(x, distance, color=CTX_C, lw=2, label="$-0.75|e|$ distance (zeroed in zone)")
    ax.plot(x, overshoot, color=FIX_C, lw=2, label="$-2\\max(0,x-x_t)$ overshoot")
    ax.axhline(0, color=MUTE, lw=0.8)
    ax.scatter([xt], [0.0], s=60, color="#2E7D32", zorder=6)
    ax.annotate(
        "+80 terminal\nhold bonus",
        (xt, 0),
        (xt - 1.6, 1.2),
        fontsize=9,
        color="#2E7D32",
        arrowprops=dict(arrowstyle="->", color="#2E7D32"),
    )
    ax.text(
        (xt - 1), -0.05, "decel bonus +10·Δ(−ẋ)\nactive in zone", fontsize=8, color="#F9A825", ha="center", va="top"
    )
    ax.set_xlabel("position x [m]")
    ax.set_ylabel("reward contribution (schematic)")
    ax.set_title("Reward shaping across the approach and braking zone", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.3)
    f.tight_layout()
    return f


@fig("03", "fig3_7_pendulum_cascade.png")
def _():
    f, ax = plt.subplots(figsize=(9.2, 3.8))
    clean(ax, (0, 12), (0, 5))
    ax.text(
        6,
        4.6,
        "Inverted-pendulum cascade (RL schedules the inner angle-PID)",
        ha="center",
        fontsize=11.5,
        fontweight="bold",
        color=INK,
    )
    box(ax, 2.1, 3.2, 2.8, 1.0, "Outer loop (fixed)\n$\\theta_{ref}=clip(k_x x+k_{\\dot x}\\dot x)$", fc=PANEL, fs=8.5)
    box(ax, 5.6, 3.2, 2.6, 1.0, "Inner angle PID\n$K_p,K_d$ scheduled by RL", fc="#E3F2FD", ec=CTX_C, fs=8.5, bold=True)
    box(ax, 8.9, 3.2, 2.2, 1.0, "Cart-pole plant\n(unstable)", fc=PANEL, fs=9)
    arrow(ax, (3.5, 3.2), (4.3, 3.2), text="$\\theta_{ref}$", dy=0.25)
    arrow(ax, (6.9, 3.2), (7.8, 3.2), text="torque", dy=0.25)
    # feedbacks (orthogonal, two levels off a shared drop at x=8.9)
    ax.plot([8.9, 8.9], [2.7, 0.9], color=INK, lw=1.2, zorder=1)
    ax.plot([8.9, 5.6], [1.5, 1.5], color=CTX_C, lw=1.3, zorder=1)
    arrow(ax, (5.6, 1.5), (5.6, 2.7), color=CTX_C, text="θ (angle)", dx=-0.95, dy=0)
    ax.plot([8.9, 2.1], [0.9, 0.9], color=MUTE, lw=1.3, zorder=1)
    arrow(ax, (2.1, 0.9), (2.1, 2.7), color=MUTE, text="x, ẋ (cart)", dx=-1.0, dy=0)
    box(ax, 10.9, 3.2, 1.6, 1.0, "hidden:\npole mass,\ngear", fc="#FFF3E0", ec="#FB8C00", fs=8)
    arrow(ax, (10.1, 3.2), (10.1, 3.2), color="#FB8C00")
    return f


# ======================================================================
# CHAPTER 4
# ======================================================================


@fig("04", "fig4_2_scenario_map.png")
def _():
    f, ax = plt.subplots(figsize=(7.6, 5.2))
    # training box
    ax.add_patch(plt.Rectangle((0.6, 5), 0.8, 15, fc="#E8F5E9", ec="#2E7D32", lw=1.5, alpha=0.6, zorder=1))
    ax.text(1.0, 21.5, "training range\nκ∈[0.6,1.4], m∈[5,20]", ha="center", fontsize=8, color="#2E7D32")
    scen = [
        ("Standard", 1.0, 10, "in"),
        ("Light Strong", 1.3, 6, "in"),
        ("Heavy Weak", 0.7, 18, "in"),
        ("Heavy Slippery", 1.0, 20, "in"),
        ("Light Grippy", 1.0, 5, "in"),
        ("OOD Ultra Heavy", 1.0, 35, "ood"),
        ("OOD Weak Motor", 0.45, 10, "ood"),
        ("OOD Heavy Weak", 0.55, 30, "ood"),
    ]
    for name, k, m, kind in scen:
        c = FIX_C if kind == "ood" else CTX_C
        mk = "X" if kind == "ood" else "o"
        ax.scatter([k], [m], s=90, color=c, marker=mk, zorder=5, edgecolor="white", linewidth=0.6)
        ax.annotate(name, (k, m), (k + 0.03, m + 0.8), fontsize=8, color=c)
    ax.set_xlabel("actuator strength κ  (the discriminative axis →)")
    ax.set_ylabel("mass m [kg]")
    ax.set_title("Scenario suite: 5 in-distribution + 3 out-of-distribution", fontsize=11, fontweight="bold")
    ax.set_xlim(0.3, 1.6)
    ax.set_ylim(0, 40)
    ax.grid(alpha=0.3)
    ax.scatter([], [], color=CTX_C, marker="o", label="in-distribution")
    ax.scatter([], [], color=FIX_C, marker="X", label="out-of-distribution")
    ax.legend(fontsize=8, loc="upper right")
    f.tight_layout()
    return f


@fig("04", "fig4_1_metric_trace.png")
def _():
    f, ax = plt.subplots(figsize=(8.4, 4.4))
    t = np.linspace(0, 20, 600)
    xt = 5.0
    delta = 0.05
    # synthetic approach with a small overshoot then settle
    x = xt * (1 - np.exp(-t / 4)) + 0.35 * np.exp(-t / 3) * np.sin(2.0 * t) * (t > 6)
    x = np.where(t < 1, xt * (1 - np.exp(-t / 4)), x)
    ax.axhspan(xt - delta, xt + delta, color="#C8E6C9", alpha=0.7, label="hold band ±0.05 m")
    ax.axhline(xt, color=MUTE, ls="--", lw=1)
    ax.plot(t, x, color=CTX_C, lw=2, label="position x(t)")
    ax.fill_between(t, x, xt, color=CTX_C, alpha=0.10)
    # settling time (stays-within)
    within = np.abs(xt - x) <= delta
    ts_idx = len(within) - 1
    for i in range(len(within)):
        if within[i:].all():
            ts_idx = i
            break
    ax.axvline(t[ts_idx], color=FIX_C, lw=1.6)
    ax.annotate(
        "settling time $t_s$ (stays-within)",
        (t[ts_idx], 2.5),
        (t[ts_idx] + 0.4, 1.4),
        fontsize=9,
        color=FIX_C,
        arrowprops=dict(arrowstyle="->", color=FIX_C),
    )
    omax = x.max()
    if omax > xt + delta:
        ax.annotate(
            "overshoot",
            (t[np.argmax(x)], omax),
            (t[np.argmax(x)] + 1.5, omax + 0.3),
            fontsize=9,
            color="#E65100",
            arrowprops=dict(arrowstyle="->", color="#E65100"),
        )
    ax.text(13, 3.4, "shaded = IAE\n($\\sum|e|\\Delta t$)", fontsize=8, color=CTX_C)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("position [m]")
    ax.set_title("Metric definitions on a single position trace", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    f.tight_layout()
    return f


# ======================================================================
# CHAPTER 6
# ======================================================================


@fig("06", "fig6_1_capability_cost.png")
def _():
    f, ax = plt.subplots(figsize=(7.8, 5.4))
    pts = [
        ("Fixed PID", 0.7, 1.2, FIX_C, "no model/training;\nfails only on windup"),
        ("Anti-Windup PID", 1.4, 2.6, AW_C, "two lines; solves windup;\nright answer if stable"),
        ("MRAC", 4.2, 1.0, "#F9A825", "fails even when fair\n(no usable sensitivity)"),
        ("Learned (blind/context)", 7.5, 7.2, CTX_C, "per-episode adaptation;\ncost: training, opacity"),
    ]
    for name, cost, cap, c, note in pts:
        ax.scatter([cost], [cap], s=420, color=c, alpha=0.85, edgecolor="white", linewidth=1.5, zorder=5)
        ax.text(cost, cap, name.split()[0][0], ha="center", va="center", color="white", fontweight="bold", zorder=6)
        ax.annotate(f"{name}\n{note}", (cost, cap), (cost + 0.2, cap - 1.5), fontsize=8, color=c)
    ax.set_xlabel("cost  (model / training / opacity / sim-to-real risk) →")
    ax.set_ylabel("capability on UNKNOWN, varying, or UNSTABLE dynamics →")
    ax.set_title(
        "Capability vs cost: match the controller to the failure mode you face", fontsize=10.5, fontweight="bold"
    )
    ax.set_xlim(0, 9.5)
    ax.set_ylim(0, 9)
    ax.grid(alpha=0.3)
    f.tight_layout()
    return f


@fig("06", "fig6_2_decision_tree.png")
def _():
    f, ax = plt.subplots(figsize=(8.4, 5.2))
    clean(ax, (0, 12), (0, 10))
    ax.text(
        6,
        9.5,
        "Which controller? Use the simplest whose failure mode you don't face",
        ha="center",
        fontsize=11,
        fontweight="bold",
        color=INK,
    )
    box(ax, 6, 8.2, 4.2, 0.9, "Dynamics known & roughly constant?", fc=PANEL, fs=9)
    box(ax, 2.4, 6.6, 3.0, 0.9, "Fixed PID", fc="#FFEBEE", ec=FIX_C, fs=10, bold=True)
    arrow(ax, (4.4, 8.0), (2.6, 7.1), text="yes (stable)", color="#2E7D32", dx=-0.2, dy=0.25)
    box(ax, 7.4, 6.6, 4.4, 0.9, "Windup-prone but still stable?", fc=PANEL, fs=9)
    arrow(ax, (7.0, 7.75), (7.4, 7.1), text="no", color=FIX_C)
    box(ax, 3.0, 4.9, 3.4, 0.9, "Anti-Windup PID", fc="#F3E5F5", ec=AW_C, fs=10, bold=True)
    arrow(ax, (6.0, 6.2), (3.6, 5.4), text="yes", color="#2E7D32")
    box(ax, 8.6, 4.9, 5.0, 0.9, "Unknown/varying AND consequential\n(unstable, tight tolerance)?", fc=PANEL, fs=8.5)
    arrow(ax, (8.4, 6.1), (8.6, 5.4), text="no", color=AW_C)
    box(ax, 6.4, 2.9, 3.6, 0.9, "Learned gain scheduling", fc="#E3F2FD", ec=CTX_C, fs=10, bold=True)
    arrow(ax, (8.0, 4.4), (6.8, 3.4), text="yes", color="#2E7D32")
    box(ax, 10.4, 2.9, 2.8, 1.1, "MRAC only if\nmodel + reliable\nsensitivity exist", fc="#FFF8E1", ec="#F9A825", fs=8)
    arrow(ax, (9.6, 4.4), (10.4, 3.5), text="classical\nadaptive?", color="#F9A825", dx=0.7, dy=0)
    return f


# ======================================================================

if __name__ == "__main__":
    # build all
    import sys

    this = sys.modules[__name__]
    print(f"=== {len(made)} figures written ===")
    for m in made:
        print("  +", m)
    if skipped:
        print(f"--- {len(skipped)} skipped ---")
        for s in skipped:
            print("  -", s)
