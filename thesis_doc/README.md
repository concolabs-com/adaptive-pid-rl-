# Thesis — Meta-RL for Adaptive PID Gain Scheduling Under Unknown Vehicle Dynamics

**Research question:** Can a Meta-RL agent learn to schedule PID gains in real-time to handle unknown vehicle dynamics (mass, friction), and does providing physics context improve adaptation over a blind agent?

---

## Directory Structure

```
thesis_doc/
├── README.md                          ← this file
├── THESIS_WRITING_GUIDE.md            ← detailed chapter-by-chapter writing guide
├── FINDINGS_SUMMARY.md                ← key results and findings (quick reference)
├── progress-tracker.md                ← writing progress tracker
│
├── chapter01/                         ← Introduction (~1,500 words)
│   ├── chapter1_introduction.md
│   └── figures/
│
├── chapter02/                         ← Background and Related Work (~3,000 words)
│   ├── chapter2_background.md
│   └── figures/
│
├── chapter03/                         ← System Design and Methodology (~3,500 words)
│   ├── chapter3_methodology.md
│   └── figures/
│
├── chapter04/                         ← Experimental Setup (~1,500 words)
│   ├── chapter4_experimental_setup.md
│   └── figures/
│
├── chapter05/                         ← Results and Analysis (~4,000 words)
│   ├── chapter5_results.md
│   ├── figures/
│   │   ├── training/
│   │   │   ├── stage5a_learning_curve.png
│   │   │   └── stage5b_learning_curve.png
│   │   ├── trajectories/
│   │   │   ├── stage5a_static_trajectories.png
│   │   │   ├── stage5b_static_trajectories.png
│   │   │   ├── stage5a_dynamic_trajectories.png
│   │   │   ├── stage5b_dynamic_trajectories.png
│   │   │   ├── baseline_static_trajectories.png
│   │   │   ├── baseline_dynamic_trajectories.png
│   │   │   └── baseline_no_reset_trajectories.png
│   │   └── eval/
│   │       ├── stage5a_eval_summary.png
│   │       ├── stage5b_eval_summary.png
│   │       ├── stage5a_dynamic_eval_summary.png
│   │       └── stage5b_dynamic_eval_summary.png
│   └── data/                          ← raw CSV results (for tables + reproduction)
│       ├── stage5a_context_rl/
│       ├── stage5b_blind_rl/
│       ├── stage5a_dynamic_eval/
│       ├── stage5b_dynamic_eval/
│       ├── baseline_static/
│       ├── baseline_dynamic/
│       └── baseline_no_reset_finding/
│
├── chapter06/                         ← Discussion and Limitations (~2,000 words)
│   ├── chapter6_discussion.md
│   └── figures/
│
├── chapter07/                         ← Conclusion and Future Work (~1,000 words)
│   ├── chapter7_conclusion.md
│   └── figures/
│
├── environment/                       ← simulation source code
│   ├── envs/adaptive_suspension.py    ← MuJoCo PID control environment
│   └── agents/
│       ├── model.py                   ← policy network (MLP + frame stack)
│       └── domain_randomization.py   ← mass/friction randomization wrapper
│
├── scripts/                           ← experiment launch scripts
│   ├── stage2_meta_rl_reproduction.py ← core PPO training & evaluation framework
│   ├── stage5a_context_cliff.py       ← Stage 5a: context-aware RL training
│   ├── stage5b_blind_cliff.py         ← Stage 5b: blind RL training
│   ├── stage5a_eval_dynamic.py        ← Stage 5a: eval with mid-episode disturbances
│   ├── stage5b_eval_dynamic.py        ← Stage 5b: eval with mid-episode disturbances
│   └── stage_baseline_fixed_pid.py   ← fixed PID classical baseline
│
└── results/                           ← original raw results (source of chapter05/data)
```

---

## Thesis Overview

### Agents Compared

| Agent | Observation | Description |
|-------|-------------|-------------|
| Fixed PID | — | Classical baseline, Kp=1.8 Ki=0.7 Kd=0.5, no adaptation |
| **Stage 5a** | pos, vel, err, kp, ki, kd, **mass, friction** | Context-aware RL — sees its own physics parameters |
| **Stage 5b** | pos, vel, err, kp, ki, kd | Blind RL — must infer dynamics from trajectory history |

Both RL agents use PPO with frame stacking (stack_size=10), curriculum learning (3→5→7→10 m), and domain randomisation (mass ∈ [5,20] kg, friction ∈ [0.1,2.0]).

### Key Results

| Agent | Success (static) | Mean settling — Standard | Mean settling — Heavy+Slippery |
|-------|-----------------|--------------------------|-------------------------------|
| Fixed PID | 100% | 1.116 s | 1.136 s |
| Stage 5a (Context RL) | 100% | 1.254 s | 1.278 s |
| Stage 5b (Blind RL) | 100% | 1.446 s | 1.476 s |

Both RL agents achieve 100% success with zero overshoot. Stage 5b settles ~14–15% slower than Stage 5a. Fixed PID is fastest — due to the `brake_integral_reset` confound (see Chapter 5).

---

## How to Reproduce

```bash
# Train Stage 5a (context-aware)
python scripts/stage5a_context_cliff.py

# Train Stage 5b (blind)
python scripts/stage5b_blind_cliff.py

# Fixed PID baseline
python scripts/stage_baseline_fixed_pid.py

# Evaluate with mid-episode disturbances
python scripts/stage5a_eval_dynamic.py
python scripts/stage5b_eval_dynamic.py
```

Raw results are in `results/`. Figures used in the thesis are in `chapter05/figures/`. Data tables are in `chapter05/data/`.

---

## Writing Status

See `progress-tracker.md` for current status of each chapter.
