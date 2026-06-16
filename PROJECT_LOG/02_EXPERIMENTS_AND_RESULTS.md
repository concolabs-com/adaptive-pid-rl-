# 02 — Experiments and Results (by RQ)

All numbers post-audit (corrected dt, protocol v2, true context). Result CSVs
live under `benchmark_results/` (git-ignored — regenerate from scripts). Cross-
seed aggregates: `*/aggregate_all_seeds.csv` via `utils/aggregate_seeds.py`.

---

## Setup recap

- **Plants:** car (`envs/adaptive_suspension.py`, 9-dim obs), inverted pendulum
  (`envs/adaptive_pendulum.py`).
- **Hidden axes:** mass [5,20] kg, friction [0.1,2.0] (inert), actuator
  strength [0.6,1.4] (discriminative).
- **Agents:** Context-Aware (Stage 6a, 9-dim), Blind (Stage 6b, 6-dim),
  GRU-Blind (Stage 6d), Stack-k Blind (Stage 6c), Fixed PID, Anti-Windup PID,
  MRAC.
- **Training:** PPO, 1M steps, 5 seeds {7,21,42,84,123} for the main pair,
  preset `thesis_v6_hipmdp`, curriculum 1→3→5→7→10 m.
- **Eval:** protocol v2, 8 scenarios, 10 episodes/seed.

---

## RQ1 — Feasibility

Both agents: **100% hold success, all 8 scenarios, all 5 seeds, zero
overshoot**, including OOD (mass 35 kg = 1.75× ceiling; actuator 0.45 = below
floor; combined heavy-weak). Conclusion: feasible and generalizes beyond
training support.

## RQ2 — Context vs blind (settling s, mean over 5 seeds)

| Scenario | Context | Blind | Blind penalty |
|----------|---------|-------|---------------|
| Light Strong Motor | 10.90 ± 0.68 | 12.29 ± 0.52 | +12.8% |
| Light and Grippy | 14.14 ± 0.94 | 15.77 ± 0.85 | +11.5% |
| OOD Ultra Heavy | 14.38 ± 0.73 | 15.29 ± 0.68 | +6.3% |
| Standard | 13.72 ± 0.68 | 14.57 ± 0.68 | +6.2% |
| Heavy and Slippery | 14.13 ± 0.73 | 14.84 ± 0.69 | +5.0% |
| Heavy Weak Motor | 18.61 ± 1.04 | 19.16 ± 0.99 | +3.0% |
| OOD Heavy Weak | 23.35 ± 1.25 | 23.79 ± 1.19 | +1.9% |
| OOD Weak Motor | 26.58 ± 1.40 | 26.91 ± 1.37 | +1.2% |

Advantage largest when fast (control-limited), ≈0 when actuator-limited
(travel-time-limited). **Actuator sweep** (context, true ctx): 29.6→8.8 s over
κ 0.4→2.0 — monotone, both agents 100% across, context leads throughout.

## RQ3 — Memory

**Stack-depth ablation (blind, seed 7), Standard settling:** k=1 14.73 · k=3
14.25 · k=5 14.39 · k=10 14.71 · k=20 14.35 s. **Flat**, 100% at every k incl.
near-memoryless k=1. Reason: instantaneous obs (velocity, prev-gains) already
carries the identifiable dynamics. Caveat: k=1 still sees prev-action, not
strictly Markov-blind.

**GRU (recurrent blind, seeds 7,21):** 100% success, Standard settling
**~11.3 s** — faster than frame-stack blind (14.6 s) and the context MLP
(13.7 s). Fastest blind variant. Caveat: 128 hidden vs MLP 2×64 → recurrence/
capacity confound. Note: contradicts a Stage-4 expectation (GRU diverged under
the v3_safety governor+cliff protocol); under v6 it trains fine → that earlier
failure was protocol-induced.

## RQ4 — Probing (representation decodability)

Ridge probe (GroupKFold by episode) on the blind policy's penultimate
activations, R² by episode phase:

| Parameter | Decodable phase | Peak R² | Cruise R² |
|-----------|-----------------|---------|-----------|
| Mass | acceleration (steps 5–40) | 0.37–0.39 | ~0.05 |
| Actuator | cruise (steps 20–160) | 0.58–0.65 | sustained |
| Friction | — | ~0.28 (spurious) | — |

**Double dissociation** matching physics: mass via ẍ=F/m (acceleration only);
actuator via v_max∝κ (cruise only); friction inert. Direct evidence of implicit
system identification.

## Classical baselines

**With env aid (`brake_integral_reset`):** Fixed PID 100%, 11.2–12.5 s (fastest
overall — but the aid solved the hard part).

**No-reset env (target 8 m, no aid):**
| Controller | Success | Settling | Overshoot |
|------------|---------|----------|-----------|
| Fixed PID (naive) | 2/3 scenarios | 74–93 s | ~7 m |
| Anti-Windup PID (back-calc) | 100% | ~21 s | 0.13 m |
| Context/Blind (zero-shot) | 100% in-dist | 39–61 s | ~4.2 m |

**MRAC (feasible):** 0% all configs (overshoot 2.4–4.6 m, timeout).

## Mass sweep (discriminativeness)

Fixed-PID settling 5→50 kg: 11.06→12.30 s (~11% over 10× mass). Actuator sweep
0.5→2.0: 21.8→5.9 s (3.7×). Confirms actuator is the discriminative axis,
mass is not.

## Shock probe (mid-approach, 10 seeds)

| Shock | Fixed | Context | Blind |
|-------|-------|---------|-------|
| Mass ×2.5 | 100%, 10.3 s | 100%, 11.1 s | 100%, 13.9 s |
| Actuator ×0.5 | 30%, 48.5 s | 30%, 48.4 s | 30%, 50.0 s |

Mass shock: all recover; context<blind (2.8 s = re-inference cost). Actuator
shock: all collapse to 30% identically — gain scheduling cannot rescue lost
physical authority. RL value = initial adaptation, not shock rejection.

## Gain analysis

Two viable regimes: context low-Kp(~1.43)/high-Kd(~1.5); blind reverse
(Kp~1.86, Kd~1.17). Hold-phase gains nearly flat vs actuator (context Kd
1.521→1.498 over κ 0.6→1.4) ⇒ robust operating point, not a steep schedule.
Caveat: hold phase isn't where scheduling lives; approach-phase analysis is a
refinement.

## Pendulum transfer (survival fraction)

| Scenario | Fixed PID | Blind | Context |
|----------|-----------|-------|---------|
| Nominal/Light/Heavy/Weak-Gear | 1.00 | 1.00 | 1.00 |
| Heavy Pole + Weak Gear | 0.58 | 0.95 | 1.00 |
| OOD Very Weak Gear | 0.46 | 0.75 | 0.60 |

Fixed gains fail hard corners; RL rescues them. Larger RL advantage than on the
car (pendulum unstable → adaptation matters more). Strongest RL-beats-fixed
result in the thesis.

## Key result-artifact locations

- `benchmark_results/stage6a_context_hipmdp/`, `stage6b_blind_hipmdp/`
- `benchmark_results/stage6c_stack{1,3,5,20}/`, `stage6d_gru_blind_s{7,21}/`
- `benchmark_results/actuator_sweep_v6/`, `mass_sweep/`
- `benchmark_results/probe_stage6b/`, `gain_analysis/`, `stage6_shock_probe/`
- `benchmark_results/baseline_antiwindup/`, `baseline_fixed_pid*/`,
  `stage4b_mrac_feasible/`, `zeroshot_noreset_5{a,b}/`
- `benchmark_results/pendulum/eval/`
