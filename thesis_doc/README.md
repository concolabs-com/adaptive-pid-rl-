# Thesis Submission — Meta-RL for Adaptive PID Gain Scheduling

**Topic:** Reinforcement learning for adaptive PID gain scheduling on a simulated 2-wheeled vehicle (MuJoCo), with domain randomization across unknown mass and friction parameters.

---

## Research Question

Can a Meta-RL agent learn to schedule PID gains in real-time to handle unknown vehicle dynamics (mass, friction), and does providing physics context (mass, friction as observations) improve adaptation over a blind agent?

---

## Directory Structure

```
thesis_submission/
├── README.md                        ← this file
├── environment/                     ← simulation environment source code
│   ├── envs/adaptive_suspension.py  ← MuJoCo PID control environment
│   └── agents/
│       ├── meta_ppo.py              ← PPO training loop
│       ├── model.py                 ← policy network (MLP + frame stack)
│       └── domain_randomization.py ← mass/friction randomization wrapper
├── scripts/                         ← experiment launch scripts
│   ├── stage2_meta_rl_reproduction.py  ← core PPO training & evaluation framework
│   ├── stage5a_context_cliff.py        ← Stage 5a: context-aware RL training
│   ├── stage5b_blind_cliff.py          ← Stage 5b: blind RL training
│   ├── stage5a_eval_dynamic.py         ← Stage 5a: evaluation with mid-episode disturbances
│   ├── stage5b_eval_dynamic.py         ← Stage 5b: evaluation with mid-episode disturbances
│   └── stage_baseline_fixed_pid.py     ← Fixed PID classical baseline evaluation
└── results/                         ← all experimental results (CSV + plots)
    ├── stage5a_context_rl/          ← Stage 5a: context-aware RL (main result)
    ├── stage5b_blind_rl/            ← Stage 5b: blind RL (ablation)
    ├── stage5a_dynamic_eval/        ← Stage 5a under mid-episode disturbances
    ├── stage5b_dynamic_eval/        ← Stage 5b under mid-episode disturbances
    ├── baseline_static/             ← Fixed PID, static physics
    ├── baseline_dynamic/            ← Fixed PID, mid-episode disturbances
    └── baseline_no_reset_finding/   ← Fixed PID without integral reset (confound analysis)
```

---

## Experimental Design

### Environment
- MuJoCo 2-wheeled vehicle, target position 5.0 m
- PID controller with RL-scheduled gains: Kp ∈ [0.8, 2.8], Ki ∈ [0.1, 1.3], Kd ∈ [0.0, 2.5]
- Action = gain adjustment delta; observation = [pos, vel, error, kp, ki, kd, (mass_scale, friction_scale)]
- Domain randomization: mass ∈ [5, 20] kg, friction ∈ [0.1, 2.0] at each episode reset

### Agents Compared

| Agent | Observation | Gains | Description |
|-------|-------------|-------|-------------|
| Fixed PID | — | Kp=1.8, Ki=0.7, Kd=0.5 | Classical baseline, no adaptation |
| **Stage 5a** | [pos, vel, err, kp, ki, kd, **mass, friction**] | adaptive | Context-aware RL — knows its own physics |
| **Stage 5b** | [pos, vel, err, kp, ki, kd] | adaptive | Blind RL — must infer dynamics from trajectory |

Both RL agents use:
- PPO with frame stacking (stack_size=10) as temporal memory
- Curriculum learning: target distance 3→5→7→10 m over 1.5M training steps
- Mid-episode mass/friction disturbances during training
- Position friction patch (x=1.5–2.4 m, friction×0.35)

### Evaluation Scenarios

| Scenario | Mass | Friction | Notes |
|----------|------|----------|-------|
| Standard | 10 kg | 1.0 | Training distribution |
| Heavy and Slippery | 20 kg | 0.2 | Hardest in-distribution |
| Light and Grippy | 5 kg | 2.0 | Most responsive |
| OOD Ultra Heavy | 35 kg | 1.0 | Out-of-distribution |
| OOD Ultra Slippery | 20 kg | 0.05 | Out-of-distribution |

Eval seeds: 70000–70009 (10 episodes per scenario).

---

## Key Results

### Static Evaluation (fixed physics per episode)

| Agent | Standard Success | Heavy+Slippery Success | Light+Grippy Success | Mean Settling (s) |
|-------|-----------------|----------------------|---------------------|-------------------|
| Fixed PID | 100% | 100% | 100% | 1.12–1.25 |
| Stage 5a (Context RL) | 100% | 100% | 100% | 1.25–1.39 |
| Stage 5b (Blind RL) | 100% | 100% | 100% | 1.45–1.57 |

**Finding:** Both RL agents achieve 100% success across all mass/friction conditions. Context-aware RL (Stage 5a) settles ~10–15% faster than blind RL (Stage 5b), demonstrating that physics context in the observation improves adaptation speed.

### Dynamic Evaluation (mid-episode mass/friction disturbances)
See `results/stage5a_dynamic_eval/` and `results/stage5b_dynamic_eval/` for full breakdown.

### Confound Analysis — Integral Reset
Running Fixed PID **without** the `brake_integral_reset` engineering aid:
- Fixed PID: 0% success, massive overshoot (~10 m) across all scenarios
- See `results/baseline_no_reset_finding/`

**Finding:** A `brake_integral_reset` mechanism in the environment (zeroes PID integral when entering braking zone) acts as an engineering aid that equalizes all agents. Without it, fixed PID at base gains completely fails due to integral windup. This is a key identified confound in the experimental setup.

---

## How to Reproduce

### Requirements
```
python >= 3.10
mujoco >= 2.3
gymnasium
stable-baselines3 (PPO reference)
torch
numpy, pandas, matplotlib
```

### Train Stage 5a (Context RL)
```bash
python scripts/stage5a_context_cliff.py
```

### Train Stage 5b (Blind RL)
```bash
python scripts/stage5b_blind_cliff.py
```

### Run Fixed PID Baseline
```bash
python scripts/stage_baseline_fixed_pid.py           # static
python scripts/stage_baseline_fixed_pid.py --dynamic  # with disturbances
python scripts/stage_baseline_fixed_pid.py --no-reset # confound analysis
```

### Evaluate Trained Models with Disturbances
```bash
python scripts/stage5a_eval_dynamic.py
python scripts/stage5b_eval_dynamic.py
```

The core framework (`stage2_meta_rl_reproduction.py`) accepts many CLI arguments for custom configurations. Run with `--help` for full reference.
