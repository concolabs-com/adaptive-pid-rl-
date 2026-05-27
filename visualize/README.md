# Adaptive Suspension Agent

Train and visualize RL agents that learn to drive a car to a target position
by scheduling PID gains (Kp, Ki, Kd) at each timestep.

---

## Training

Train a new agent from scratch (~5–10 min on a laptop CPU):

```bash
python train.py
```

Longer run for a better policy:
```bash
python train.py --timesteps 200000 --run-name my_run
```

All outputs are saved to `training_runs/<run_name>/`:

| File | Contents |
|---|---|
| `model.pth` | Trained model weights |
| `config.json` | All hyperparameters (for exact reproduction) |
| `training_curve.csv` | Per-episode rewards and lengths |
| `training_curve.png` | Reward over training steps |
| `displacement_time.png` | Position vs time — 3 eval scenarios |
| `velocity_time.png` | Velocity vs time — 3 eval scenarios |
| `eval_results.csv` | Final error, overshoot, settling per scenario |

After training, visualize the result:
```bash
python run_agent.py --model training_runs/<run_name>/model.pth --obs-dims 6
```

### Training options

```
--timesteps        Total training steps                  [default: 100000]
--target           Target position in metres             [default: 5.0]
--run-name         Output folder name                    [default: timestamp]
--obs-dims         6=blind (faster), 8=context-aware     [default: 6]
--seed             Random seed                           [default: 42]
```

---

## Quick Start (pre-trained models)

Change directory to the visualize folder:
```bash
cd ..\visualize
```

**Stage 5a — Context-aware agent** (sees mass + friction):
```bash
python run_agent.py --model environment/agents/models/stage5a_context_cliff/meta_rl_agent.pth
```

**Stage 5b — Blind agent** (no context):
```bash
python run_agent.py --model environment/agents/models/stage5b_blind_cliff/meta_rl_agent.pth --obs-dims 6
```

---

## Physics Scenarios

Use `--scenario` for preset conditions, or set `--mass` / `--friction` manually.

| Flag | Mass (kg) | Friction | Description |
|------|-----------|----------|-------------|
| `--scenario standard` | 10.0 | 1.0 | Training nominal *(default)* |
| `--scenario heavy` | 20.0 | 0.2 | Heavy + slippery |
| `--scenario light` | 5.0 | 2.0 | Light + grippy |

```bash
# Heavy scenario with context-aware agent
python run_agent.py \
  --model environment/agents/models/stage5a_context_cliff/meta_rl_agent.pth \
  --scenario heavy

# Custom mass/friction
python run_agent.py \
  --model environment/agents/models/stage5a_context_cliff/meta_rl_agent.pth \
  --mass 15.0 --friction 0.4
```

---

## All Options

```
--model            Path to .pth checkpoint                    [required]
--target           Target position in metres                  [default: 5.0]
--seed             Random seed for episode reset              [default: 42]
--max-steps        Max episode steps before timeout           [default: 500]
--obs-dims         Obs dims per step: 8=context, 6=blind      [default: 8]
--stack-size       Frame stack size (must match training)     [default: 10]
--scenario         Preset: standard | heavy | light
--mass             Car mass in kg                             [default: 10.0]
--friction         Floor friction coefficient                 [default: 1.0]
```

---

## Model Reference

| Model file | Agent type | `--obs-dims` | Protocol |
|---|---|---|---|
| `stage5a_context_cliff/meta_rl_agent.pth` | Context-aware MLP | `8` | thesis_v4_cliff |
| `stage5b_blind_cliff/meta_rl_agent.pth`   | Blind MLP         | `6` | thesis_v4_cliff |

**Always pass `--obs-dims 6` for the blind (stage5b) model.**  
The default `--obs-dims 8` is correct for stage5a.

---

## File Layout

```
visualize/
├── run_agent.py                                  ← visualize a trained model
├── train.py                                      ← train a new model from scratch
├── environment/
│   ├── envs/
│   │   ├── adaptive_suspension.py                ← MuJoCo gym environment
│   │   └── assets/
│   │       └── car_model.xml                     ← MuJoCo XML model
│   └── agents/
│       ├── model.py                              ← Agent network (MLP / GRU)
│       ├── domain_randomization.py               ← DR wrapper
│       └── models/
│           ├── stage5a_context_cliff/
│           │   └── meta_rl_agent.pth             ← pre-trained context-aware
│           └── stage5b_blind_cliff/
│               └── meta_rl_agent.pth             ← pre-trained blind
├── training_runs/                                ← created by train.py
│   └── <run_name>/
│       ├── model.pth
│       ├── config.json
│       ├── training_curve.csv / .png
│       ├── displacement_time.png
│       ├── velocity_time.png
│       └── eval_results.csv
└── utils/
    ├── pid.py                                    ← PID controller
    └── wrappers.py                               ← Obs wrappers
```

---

## Dependencies

Can be installed via the `requriements.txt` file
```bash
pip install -r ..\requirements.txt
```

Or installing manually
```bash
pip install gymnasium[mujoco] torch numpy
```

Requires MuJoCo ≥ 3.x (passive viewer API).
