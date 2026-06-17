# Front Matter — Figures

Front matter carries **no figures of its own**. Two maintenance notes:

1. **List of Figures is stale.** `front_matter.md` §"List of Figures" still
   lists five Ch5 figures as "Fig 5.x" and omits the four Ch3 diagrams already
   embedded and the conceptual figures added under each chapter (Ch1, Ch2, Ch3,
   Ch4, Ch6). On compile, generate it automatically with LaTeX
   `\listoffigures` / Pandoc rather than maintaining it by hand. If kept manual,
   it must enumerate the full set (now 14 result figures in Ch5, 4 + 3 in Ch3,
   2 in Ch1, 5 in Ch2, 2 in Ch4, 2 in Ch6).

2. **No graphical-abstract duplication.** If the institution allows a frontis-
   piece/graphical abstract, reuse `chapter01/figures/fig1_1_graphical_abstract.png`
   rather than drawing a second one.

Status legend used across the per-chapter files:
`✅ generated` · `📐 author by hand (draw.io/Inkscape/TikZ or extend a script)` ·
`🔶 generatable from data (needs a CSV + a plot script)` · `⚠️ blocked / needs reconciliation`
