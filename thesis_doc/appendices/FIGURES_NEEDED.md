# Appendices — Figures

The appendices are **table-driven** (hyperparameters, artifact map, audit
cross-reference) and need no figures to be complete. Two optional additions,
both only worth it if a committee wants the negative design results illustrated:

| Fig | Title | Status | §/where | Why it helps |
|-----|-------|--------|---------|--------------|
| A.1 | Dropped-variant failure: overshoot-cliff value-loss collapse vs the final soft reward | 🔶 generatable | §A.7 | Visual evidence for "the hard cliff starved exploration / collapsed the value function" — if a training-log CSV for the cliff variant survives under `benchmark_results/` |
| A.2 | Governor masking: trajectories with vs without the speed governor | 🔶 generatable | §A.7 | Shows the governor, not the learned gains, did the stopping — if the `thesis_v3_safety` rollouts survive |

**Recommendation:** skip unless the survived CSVs make these cheap. The §A.7
prose already states the lesson; a figure is a nice-to-have, not a gap. All
other appendix content (A.1–A.6, B, C, D) is correctly tabular.
