#!/usr/bin/env python3
"""Generate the Chapter 3 conceptual diagrams (Fig 3.1-3.4) as PNGs.

Outputs to thesis_doc/chapter03/figures/.
Run: venv/Scripts/python.exe thesis_doc/scripts/make_diagrams.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parents[1] / "chapter03" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def box(ax, x, y, w, h, text, fc="#E3F2FD", ec="#1565C0"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.05", fc=fc, ec=ec, lw=1.6))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9)


def arrow(ax, x1, y1, x2, y2, text="", color="#37474F"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14, lw=1.4, color=color))
    if text:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.04, text, ha="center", va="bottom", fontsize=7.5)


# --- Fig 3.1: two-loop control architecture ---
fig, ax = plt.subplots(figsize=(10, 3.6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 3.6)
ax.axis("off")
box(ax, 0.2, 1.4, 1.7, 0.9, "RL policy\n(50 Hz)\noutputs ΔK", fc="#E8F5E9", ec="#2E7D32")
box(ax, 2.4, 1.4, 1.6, 0.9, "gain map\nK=Kb+ΔK·a")
box(ax, 4.5, 1.4, 1.7, 0.9, "PID\n(500 Hz)\ninner loop")
box(ax, 6.8, 1.4, 1.6, 0.9, "MuJoCo\nplant", fc="#FFF3E0", ec="#E65100")
arrow(ax, 1.9, 1.85, 2.4, 1.85, "a∈[-1,1]³")
arrow(ax, 4.0, 1.85, 4.5, 1.85, "Kp,Ki,Kd")
arrow(ax, 6.2, 1.85, 6.8, 1.85, "u")
# feedback
arrow(ax, 7.6, 1.4, 7.6, 0.5)
arrow(ax, 7.6, 0.5, 1.05, 0.5)
arrow(ax, 1.05, 0.5, 1.05, 1.4)
ax.text(4.3, 0.35, "state: pos, vel, error  (+ context for teacher)", ha="center", fontsize=7.5)
ax.text(
    5.0,
    3.2,
    "Figure 3.1 — Two-loop architecture: RL schedules PID gains; PID computes torque.",
    ha="center",
    fontsize=9,
)
fig.savefig(OUT / "fig3_1_control_loop.png", dpi=150, bbox_inches="tight")
plt.close(fig)


# --- Fig 3.2: environment schematic ---
fig, ax = plt.subplots(figsize=(10, 2.8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 2.8)
ax.axis("off")
ax.plot([0.5, 9.5], [0.8, 0.8], color="#455A64", lw=2)
ax.add_patch(plt.Rectangle((0.5, 0.85), 0.6, 0.4, fc="#1565C0"))
ax.text(0.8, 0.6, "start x=0", ha="center", fontsize=7.5)
ax.axvspan(2.0, 3.2, ymin=0.28, ymax=0.45, color="#FF9800", alpha=0.5)
ax.text(2.6, 1.35, "friction patch\nx∈[1.5,2.4]", ha="center", fontsize=7)
ax.axvline(8.5, color="k", ls="--")
ax.text(8.5, 1.5, "target 5 m\n±0.05 m hold", ha="center", fontsize=7.5)
ax.axvspan(6.5, 8.5, ymin=0.28, ymax=0.32, color="#90CAF9", alpha=0.6)
ax.text(7.5, 0.25, "braking zone |e|<2 m", ha="center", fontsize=7)
ax.text(
    5.0,
    2.5,
    "Figure 3.2 — Car task: drive to target through a friction patch; hidden mass/actuator per episode.",
    ha="center",
    fontsize=9,
)
fig.savefig(OUT / "fig3_2_environment.png", dpi=150, bbox_inches="tight")
plt.close(fig)


# --- Fig 3.3: curriculum schedule ---
fig, ax = plt.subplots(figsize=(8, 3.2))
phases = [(0, 250, 3), (250, 500, 5), (500, 750, 7), (750, 1000, 10)]
for a, b, top in phases:
    ax.fill_between([a, b], 1, top, alpha=0.25, color="#1565C0")
    ax.plot([a, b], [top, top], color="#0D47A1", lw=2)
    ax.text((a + b) / 2, top + 0.3, f"{top} m", ha="center", fontsize=8)
ax.set_xlabel("Training steps (×1000)")
ax.set_ylabel("Target distance range (m)")
ax.set_title("Figure 3.3 — Curriculum: target range grows over four phases", fontsize=9)
ax.set_xlim(0, 1000)
ax.set_ylim(0, 11)
ax.grid(alpha=0.3)
fig.savefig(OUT / "fig3_3_curriculum.png", dpi=150, bbox_inches="tight")
plt.close(fig)


# --- Fig 3.4: network architecture ---
fig, ax = plt.subplots(figsize=(9, 3.6))
ax.set_xlim(0, 9)
ax.set_ylim(0, 3.6)
ax.axis("off")
box(ax, 0.2, 1.3, 1.9, 1.0, "obs ×10 frames\n90-dim (teacher)\n60-dim (student)", fc="#F3E5F5", ec="#6A1B9A")
box(ax, 2.5, 1.3, 1.5, 1.0, "FC 64\nTanh")
box(ax, 4.2, 1.3, 1.5, 1.0, "FC 64\nTanh")
box(ax, 6.0, 2.0, 1.4, 0.8, "actor head\n3-dim Gauss", fc="#E8F5E9", ec="#2E7D32")
box(ax, 6.0, 0.8, 1.4, 0.8, "critic head\nscalar V", fc="#FFEBEE", ec="#C62828")
arrow(ax, 2.1, 1.8, 2.5, 1.8)
arrow(ax, 4.0, 1.8, 4.2, 1.8)
arrow(ax, 5.7, 1.8, 6.0, 2.3)
arrow(ax, 5.7, 1.8, 6.0, 1.2)
ax.text(
    4.5,
    3.25,
    "Figure 3.4 — Actor–critic MLP (GRU variant replaces the trunk with a 128-unit GRU).",
    ha="center",
    fontsize=9,
)
fig.savefig(OUT / "fig3_4_network.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print(f"Wrote 4 diagrams to {OUT}")
for p in sorted(OUT.glob("fig3_*.png")):
    print(" ", p.name)
