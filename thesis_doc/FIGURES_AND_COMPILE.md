# Figures Catalog & Compile Guide

How the thesis's figures are produced, where they live, how to compile the
markdown thesis into a figure-rich PDF, and how to add more figures.

---

## 1. Current figure set (14, all committed)

| Figure | File | Source | Embedded in |
|--------|------|--------|-------------|
| 3.1 Two-loop architecture | `chapter03/figures/fig3_1_control_loop.png` | `scripts/make_diagrams.py` | Ch3 §3.1 |
| 3.2 Car task schematic | `chapter03/figures/fig3_2_environment.png` | `make_diagrams.py` | Ch3 §3.2 |
| 3.3 Curriculum schedule | `chapter03/figures/fig3_3_curriculum.png` | `make_diagrams.py` | Ch3 §3.5 |
| 3.4 Network architecture | `chapter03/figures/fig3_4_network.png` | `make_diagrams.py` | Ch3 §3.4 |
| 5.0 Training curves | `figures/fig5_training_curves.png` | `make_all_figures.py` | Ch5 §5.1 |
| 5.1 RQ2 settling bars | `figures/fig5_rq2_settling.png` | `make_all_figures.py` | Ch5 §5.3 |
| 5.2 Actuator sweep | `figures/fig5_actuator_sweep.png` | `make_all_figures.py` | Ch5 §5.3 |
| 5.3 Stack ablation | `figures/fig5_stack_ablation.png` | `make_all_figures.py` | Ch5 §5.4 |
| 5.4 RQ4 probe R² | `figures/fig5_probe_r2.png` | `make_all_figures.py` | Ch5 §5.5 |
| 5.5 Anti-windup | `figures/fig5_antiwindup.png` | `make_all_figures.py` | Ch5 §5.6 |
| 5.6 MRAC heatmap | `figures/fig5_mrac.png` | `make_all_figures.py` | Ch5 §5.6 |
| 5.7 Mass sweep | `figures/fig5_mass_sweep.png` | `make_all_figures.py` | Ch5 §5.7 |
| 5.8 Gain regimes | `figures/fig5_gain_regimes.png` | `make_all_figures.py` | Ch5 §5.9 |
| 5.9 Pendulum survival | `figures/fig5_pendulum_survival.png` | `make_all_figures.py` | Ch5 §5.10 |

Regenerate everything:
```
venv/Scripts/python.exe thesis_doc/scripts/make_diagrams.py      # conceptual diagrams
venv/Scripts/python.exe thesis_doc/scripts/make_all_figures.py   # data figures from CSVs
```
The data figures read the (git-ignored) `benchmark_results/` CSVs; the PNGs
themselves ARE version-controlled (see `.gitignore` exceptions for
`thesis_doc/**/figures/`).

---

## 2. Compiling markdown → a figure-rich PDF

The chapters are GitHub-flavoured Markdown with `![caption](path)` image
embeds and `$…$` / `$$…$$` math. The recommended toolchain is **Pandoc + a
LaTeX engine**, which renders the images, math, tables, and auto-numbers
everything.

Install once: Pandoc (https://pandoc.org) and a TeX distribution (MiKTeX on
Windows, or TeX Live). Then, from the repo root:

```bash
pandoc \
  thesis_doc/chapter00/front_matter.md \
  thesis_doc/chapter01/chapter1_introduction.md \
  thesis_doc/chapter02/chapter2_background.md \
  thesis_doc/chapter03/chapter3_methodology.md \
  thesis_doc/chapter04/chapter4_experimental_setup.md \
  thesis_doc/chapter05/chapter5_results.md \
  thesis_doc/chapter06/chapter6_discussion.md \
  thesis_doc/chapter07/chapter7_conclusion.md \
  thesis_doc/appendices/appendices.md \
  thesis_doc/references.md \
  -o thesis.pdf \
  --pdf-engine=xelatex \
  --toc --number-sections \
  --resource-path=thesis_doc:thesis_doc/chapter03:thesis_doc/figures \
  -V geometry:margin=1in -V documentclass=report -V fontsize=11pt
```

Notes:
- `--resource-path` lets Pandoc resolve the relative image paths
  (`../figures/…` from Ch5, `figures/…` from Ch3). Adjust if you reorganize.
- `--toc --number-sections` gives the table of contents and section numbers;
  add `--listings` or a filter for a list of figures if your template wants
  one. `pandoc-crossref` enables `\ref`-style figure cross-references if you
  later switch the inline "Figure 5.x" text to `@fig:label` syntax.
- For a university template, point `--reference-doc` (DOCX) or a LaTeX
  `--template` at the required style; the markdown body is unchanged.

A one-command build script can be dropped at `thesis_doc/build_pdf.sh`.

---

## 3. How to add MORE figures (the workflow you asked about)

Adding figures is now a three-line loop: **produce the PNG → embed it →
recompile.** Three routes, easiest first.

**(a) From existing result CSVs — edit `make_all_figures.py`.** Every number in
`benchmark_results/` can become a figure. Copy any `@fig(...)` block, point it
at the CSV, and re-run. Cheap candidates not yet plotted:
- IAE and success-rate bar charts (companion to the settling bars), from the
  `aggregate_all_seeds.csv` files.
- Per-seed training-curve spread (mean ± band over the five seeds).
- Adaptation-lag bars from `gain_analysis/adaptation_lag.csv`.
- GRU-vs-stack-vs-context settling bars (RQ3b).
- Anti-windup *trajectory* overlays (already saved as PNGs under
  `benchmark_results/baseline_antiwindup/*/trajectories.png`).

**(b) Trajectory / rollout figures — use the existing plot scripts.** The repo
already has `plot_trajectories.py`, `plot_simulated_run_behaviors.py`,
`stage6_shock_probe.py`, and `analyze_gain_trajectories.py`, each of which
rolls out a checkpoint and saves position/velocity/gain-vs-time PNGs. These are
the most *illustrative* figures (they show the controller actually working) and
are excellent for the results chapter — e.g. a context-vs-blind-vs-fixed
position-trace overlay for the Standard scenario, or the shock-recovery traces
already produced by `stage6_shock_probe.py` under
`benchmark_results/stage6_shock_probe/`.

**(c) Conceptual diagrams — edit `make_diagrams.py`** (matplotlib boxes/arrows)
or draw in draw.io / Inkscape / TikZ and drop the PNG/PDF into a `figures/`
dir. Good additions: a HiP-MDP / teacher–student schematic for Ch2, an
anti-windup block diagram for §2.2, and a MuJoCo screenshot of each plant
(render with `mujoco` viewer or `env.render(mode="rgb_array")`).

**Embedding:** add `![caption](relative/path.png)` plus an italic
`*Figure N — …*` line where you want it (see Ch5 for the pattern). Keep the
PNG under a `thesis_doc/**/figures/` directory so it is version-controlled.

**Rule of thumb for a strict committee:** every results subsection should have
at least one figure; every quantitative claim in the text should be visible in
a figure or table; and every figure needs a self-contained caption that states
the takeaway, not just what is plotted. The current set satisfies the first
two; trajectory overlays (route b) are the highest-value next addition for
clarity.
