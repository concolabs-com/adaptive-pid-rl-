# New Section (Ch. 4): Statistical Methodology

> Integration target: new §4.x in Experimental Setup. Replaces the implicit
> single-run reporting of the original draft. Written for a Data Science
> committee: every reported comparison states its uncertainty and test.

## 4.x.1 Sources of variation and the evaluation unit

Three nested sources of variation are distinguished and reported separately:

1. **Training seed** ($n_s = 5$: seeds 7, 21, 42, 84, 123). Captures
   optimisation stochasticity (weight initialisation, rollout sampling,
   domain-randomization draws during training). Each seed yields one policy.
2. **Evaluation episode** ($n_e = 10$ per scenario per policy). Under eval
   protocol v2 each episode samples physics within the scenario band
   (mass ±10%, friction ±10%, actuator ±5%), a target distance jitter of
   ±0.5 m, and — in dynamic evaluation — the disturbance time and magnitude.
   The policy itself is deterministic at evaluation.
3. **Scenario** — fixed condition families (Standard, Heavy Weak Motor, …)
   reported separately, never pooled.

The original evaluation protocol (v1) repeated one deterministic episode ten
times; all variance reported under v1 was therefore exactly zero, and the
original tables presented $n = 1$ as $n = 10$. This is corrected throughout
(audit finding F3).

## 4.x.2 Aggregation and intervals

Per-scenario results are reported as mean ± standard deviation over the
$n_s \times n_e = 50$ evaluation episodes, together with the **seed-level**
mean ± sd (each seed first aggregated over its 10 episodes, then aggregated
across the 5 seeds). Where a single headline interval is given it is the
seed-level 95% bootstrap confidence interval (10,000 resamples of the 5 seed
means, percentile method). Bootstrap over seeds is preferred to a normal
approximation because $n_s = 5$ is far from asymptotic.

Success rates are reported with Wilson score intervals rather than normal
approximations, which misbehave at rates near 0% and 100% — the operating
region of most scenarios here.

## 4.x.3 Hypothesis tests and effect sizes

Pairwise method comparisons (e.g. context vs blind settling time on a given
scenario) use:

- **Welch's t-test** on seed-level means (unequal variances, $n_s = 5$ per
  group), two-sided, $\alpha = 0.05$;
- **Mann–Whitney U** as a distribution-free check when normality is doubtful
  (settling times are right-skewed when timeouts occur);
- **Cliff's delta** as the effect size, with the conventional thresholds
  |δ| < 0.147 negligible, < 0.33 small, < 0.474 medium, otherwise large.
  Cliff's delta is chosen over Cohen's d because it is rank-based and robust
  to the truncation of settling times at the evaluation timeout.

Where a family of related comparisons is reported together (one method pair
across all scenarios), p-values are adjusted with the **Holm–Bonferroni**
procedure and both raw and adjusted values are tabulated.

## 4.x.4 What is *not* claimed

Five seeds bound the seed-level uncertainty but do not estimate it precisely;
reported CIs at the seed level are accordingly wide and stated as such. No
claim of statistical significance is made from episode-level pooling alone,
since episodes within a seed share a policy and are not independent samples
of the training process. Curves over swept parameters (mass sweep, stack-size
ablation) are presented as descriptive trends with per-point dispersion, not
as tested hypotheses at every point.
