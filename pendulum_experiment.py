#!/usr/bin/env python3
"""
Pendulum transfer experiment — PID gain scheduling on a second plant.

Demonstrates that the gain-scheduling framework (RL outer loop + PID inner
loop, context vs blind ablation, domain randomization over hidden parameters)
transfers from the car (stable setpoint task) to the inverted pendulum
(unstable balance task).

Hidden parameters per episode: pole mass scale [0.5, 3.0], gear scale [0.5, 1.5].

Modes:
  --mode tune    grid-search stabilising base gains for the fixed-PID baseline
  --mode train   PPO training (use --blind for the no-context student)
  --mode eval    scenario-grid evaluation of fixed PID + trained agents

Examples:
  python pendulum_experiment.py --mode tune
  python pendulum_experiment.py --mode train --seed 7
  python pendulum_experiment.py --mode train --seed 7 --blind
  python pendulum_experiment.py --mode eval
"""

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import matplotlib
import mujoco
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

matplotlib.use("Agg")

from agents.model import Agent
from envs.adaptive_pendulum import AdaptivePendulumEnv
from stage2_meta_rl_reproduction import ObservationFeatureSelectWrapper

OUTPUT_ROOT = Path("benchmark_results/pendulum")
STACK_SIZE = 10
FULL_DIMS = 9
BLIND_DIMS = 7

POLE_MASS_RANGE = (0.5, 2.5)
GEAR_RANGE = (0.6, 1.4)

EVAL_SCENARIOS = [
    ("Nominal", 1.0, 1.0),
    ("Heavy Pole", 2.2, 1.0),
    ("Light Pole", 0.6, 1.0),
    ("Weak Gear", 1.0, 0.65),
    ("Heavy Pole Weak Gear", 2.2, 0.7),
    ("OOD Very Heavy Pole", 3.0, 1.0),
    ("OOD Very Weak Gear", 1.0, 0.5),
]


class PendulumRandomizationWrapper(gym.Wrapper):
    """Per-episode hidden-parameter randomization + context push."""

    def __init__(self, env, polemass_range=POLE_MASS_RANGE, gear_range=GEAR_RANGE, randomize: bool = True):
        super().__init__(env)
        self.polemass_range = polemass_range
        self.gear_range = gear_range
        self.randomize = randomize
        base = env.unwrapped
        model = base.model
        self.pole_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pole")
        self.nominal_pole_mass = float(model.body_mass[self.pole_body_id])
        self.nominal_pole_inertia = model.body_inertia[self.pole_body_id].copy()
        self.nominal_gainprm = model.actuator_gainprm.copy()
        self.active_polemass_scale = 1.0
        self.active_gear_scale = 1.0

    def apply(self, polemass_scale: float, gear_scale: float) -> None:
        base = self.env.unwrapped
        model = base.model
        model.body_mass[self.pole_body_id] = self.nominal_pole_mass * polemass_scale
        model.body_inertia[self.pole_body_id, :] = self.nominal_pole_inertia * polemass_scale
        model.actuator_gainprm[:, 0] = self.nominal_gainprm[:, 0] * gear_scale
        self.active_polemass_scale = float(polemass_scale)
        self.active_gear_scale = float(gear_scale)
        base.set_context(polemass_scale, gear_scale)

    def reset(self, **kwargs):
        if self.randomize:
            rng = getattr(self.env.unwrapped, "np_random", np.random)
            pm = float(rng.uniform(*self.polemass_range))
            gr = float(rng.uniform(*self.gear_range))
            self.apply(pm, gr)
        obs, info = self.env.reset(**kwargs)
        # reset_model does not clear context, but re-push for symmetry.
        self.env.unwrapped.set_context(self.active_polemass_scale, self.active_gear_scale)
        return self.env.unwrapped._get_obs(), info


def make_env(
    blind: bool, impulse: bool, randomize: bool, max_steps: int = 500, base_kp: float = 1.0, base_kd: float = 0.1
):
    def thunk():
        env = AdaptivePendulumEnv(
            max_episode_steps=max_steps,
            impulse_enabled=impulse,
            gain_base_kp=base_kp,
            gain_base_kd=base_kd,
        )
        env = PendulumRandomizationWrapper(env, randomize=randomize)
        if blind:
            env = ObservationFeatureSelectWrapper(env, keep_dims=BLIND_DIMS)
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_steps)
        env = gym.wrappers.FrameStackObservation(env, stack_size=STACK_SIZE)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


# --------------------------------------------------------------------------
# Mode: tune — find stabilising base gains for the fixed baseline
# --------------------------------------------------------------------------


def run_fixed(env, episodes: int, seed0: int) -> dict:
    survivals, rms_thetas = [], []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed0 + ep)
        thetas = []
        steps = 0
        for _ in range(500):
            obs, r, term, trunc, info = env.step(np.zeros(3, dtype=np.float32))
            thetas.append(info["state"]["theta"])
            steps += 1
            if term or trunc:
                break
        survivals.append(steps / 500.0)
        rms_thetas.append(float(np.sqrt(np.mean(np.square(thetas)))) if thetas else float("nan"))
    return {"survival": float(np.mean(survivals)), "rms_theta": float(np.nanmean(rms_thetas))}


def mode_tune(args) -> int:
    rows = []
    for kp in [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0]:
        for kd in [0.02, 0.05, 0.1, 0.2, 0.4]:
            env = make_env(blind=False, impulse=False, randomize=False, base_kp=kp, base_kd=kd)()
            wrapper = env
            while not isinstance(wrapper, PendulumRandomizationWrapper):
                wrapper = wrapper.env
            # Probe robustness over the hidden-parameter grid corners + centre.
            results = []
            for pm, gr in [(1.0, 1.0), (0.5, 0.5), (3.0, 0.5), (0.5, 1.5), (3.0, 1.5)]:
                wrapper.apply(pm, gr)
                r = run_fixed(env, episodes=3, seed0=1000)
                results.append(r["survival"])
            env.close()
            row = {"kp": kp, "kd": kd, "min_survival": float(np.min(results)), "mean_survival": float(np.mean(results))}
            rows.append(row)
            print(f"kp={kp:4.1f} kd={kd:5.2f}  min={row['min_survival']:.2f} mean={row['mean_survival']:.2f}")
    df = pd.DataFrame(rows).sort_values(["min_survival", "mean_survival"], ascending=False)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_ROOT / "base_gain_tuning.csv", index=False)
    print("\nBest base gains:\n", df.head(5).to_string(index=False))
    return 0


# --------------------------------------------------------------------------
# Mode: train — compact PPO (mirrors the car hyperparameters)
# --------------------------------------------------------------------------


@dataclass
class PPOCfg:
    total_timesteps: int = 300_000
    num_envs: int = 4
    num_steps: int = 1024
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    epochs: int = 10
    minibatch: int = 64
    vf_coef: float = 0.5
    ent_coef: float = 0.0
    max_grad_norm: float = 0.5


def mode_train(args) -> int:
    cfg = PPOCfg(total_timesteps=args.timesteps)
    variant = "blind" if args.blind else "context"
    out_dir = OUTPUT_ROOT / f"{variant}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    envs = gym.vector.SyncVectorEnv(
        [
            make_env(blind=args.blind, impulse=True, randomize=True, base_kp=args.base_kp, base_kd=args.base_kd)
            for _ in range(cfg.num_envs)
        ]
    )
    agent = Agent(envs)
    optimizer = optim.Adam(agent.parameters(), lr=cfg.lr, eps=1e-5)

    obs_shape = envs.single_observation_space.shape
    obs_buf = torch.zeros((cfg.num_steps, cfg.num_envs) + obs_shape)
    act_buf = torch.zeros((cfg.num_steps, cfg.num_envs, 3))
    logp_buf = torch.zeros(cfg.num_steps, cfg.num_envs)
    rew_buf = torch.zeros(cfg.num_steps, cfg.num_envs)
    done_buf = torch.zeros(cfg.num_steps, cfg.num_envs)
    val_buf = torch.zeros(cfg.num_steps, cfg.num_envs)

    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.as_tensor(next_obs, dtype=torch.float32)
    next_done = torch.zeros(cfg.num_envs)
    n_updates = cfg.total_timesteps // (cfg.num_steps * cfg.num_envs)
    ep_returns = []
    log_rows = []
    t0 = time.time()

    for update in range(1, n_updates + 1):
        frac = 1.0 - (update - 1.0) / n_updates
        optimizer.param_groups[0]["lr"] = frac * cfg.lr

        for step in range(cfg.num_steps):
            obs_buf[step] = next_obs
            done_buf[step] = next_done
            with torch.no_grad():
                action, logp, _, value = agent.get_action_and_value(next_obs)
            act_buf[step] = action
            logp_buf[step] = logp
            val_buf[step] = value.flatten()
            next_obs_np, reward, term, trunc, infos = envs.step(torch.clamp(action, -1, 1).numpy())
            rew_buf[step] = torch.as_tensor(reward, dtype=torch.float32)
            next_done = torch.as_tensor(np.logical_or(term, trunc), dtype=torch.float32)
            next_obs = torch.as_tensor(next_obs_np, dtype=torch.float32)
            if "episode" in infos:
                finished = infos["episode"]["_r"] if "_r" in infos["episode"] else None
                rets = infos["episode"]["r"]
                mask = finished if finished is not None else np.ones_like(rets, dtype=bool)
                for r_val, m in zip(np.atleast_1d(rets), np.atleast_1d(mask)):
                    if m:
                        ep_returns.append(float(r_val))

        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rew_buf)
            lastgaelam = 0
            for t in reversed(range(cfg.num_steps)):
                if t == cfg.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value.flatten()
                else:
                    nextnonterminal = 1.0 - done_buf[t + 1]
                    nextvalues = val_buf[t + 1]
                delta = rew_buf[t] + cfg.gamma * nextvalues * nextnonterminal - val_buf[t]
                advantages[t] = lastgaelam = delta + cfg.gamma * cfg.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + val_buf

        b_obs = obs_buf.reshape((-1,) + obs_shape)
        b_act = act_buf.reshape(-1, 3)
        b_logp = logp_buf.reshape(-1)
        b_adv = advantages.reshape(-1)
        b_ret = returns.reshape(-1)
        val_buf.reshape(-1)
        batch = b_obs.shape[0]
        idx = np.arange(batch)

        for _ in range(cfg.epochs):
            np.random.shuffle(idx)
            for start in range(0, batch, cfg.minibatch):
                mb = idx[start : start + cfg.minibatch]
                _, newlogp, entropy, newval = agent.get_action_and_value(b_obs[mb], b_act[mb])
                ratio = (newlogp - b_logp[mb]).exp()
                adv = b_adv[mb]
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                pg1 = -adv * ratio
                pg2 = -adv * torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip)
                pg_loss = torch.max(pg1, pg2).mean()
                v_loss = 0.5 * ((newval.flatten() - b_ret[mb]) ** 2).mean()
                loss = pg_loss + cfg.vf_coef * v_loss - cfg.ent_coef * entropy.mean()
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
                optimizer.step()

        recent = float(np.mean(ep_returns[-40:])) if ep_returns else float("nan")
        log_rows.append({"update": update, "recent_return": recent, "elapsed_s": time.time() - t0})
        if update % 5 == 0 or update == n_updates:
            print(f"[{variant} seed {args.seed}] update {update}/{n_updates} recent_return={recent:.1f}")

    envs.close()
    torch.save(agent.state_dict(), out_dir / "agent.pth")
    pd.DataFrame(log_rows).to_csv(out_dir / "training_curve.csv", index=False)
    print(f"Saved {out_dir / 'agent.pth'}")
    return 0


# --------------------------------------------------------------------------
# Mode: eval — fixed PID vs context vs blind across the scenario grid
# --------------------------------------------------------------------------


def eval_episode(env, action_fn, wrapper, pm, gr, seed):
    wrapper.apply(pm, gr)
    obs, _ = env.reset(seed=seed)
    wrapper.apply(pm, gr)
    thetas = []
    impulse_step = None
    steps = 0
    for step in range(500):
        a = action_fn(obs)
        obs, r, term, trunc, info = env.step(a)
        thetas.append(abs(info["state"]["theta"]))
        impulse_step = info.get("impulse_step")
        steps += 1
        if term or trunc:
            break
    surv = steps / 500.0
    rms = float(np.sqrt(np.mean(np.square(thetas)))) if thetas else float("nan")
    post = float("nan")
    if impulse_step is not None and steps > impulse_step:
        post = float(np.max(thetas[impulse_step:]))
    return {"survival": surv, "rms_theta": rms, "post_impulse_max_theta": post, "steps": steps}


def mode_eval(args) -> int:
    out_dir = OUTPUT_ROOT / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    controllers = {"Fixed PID": (False, None)}
    ctx_path = OUTPUT_ROOT / f"context_seed{args.seed}" / "agent.pth"
    blind_path = OUTPUT_ROOT / f"blind_seed{args.seed}" / "agent.pth"
    if ctx_path.exists():
        controllers["RL context"] = (False, ctx_path)
    if blind_path.exists():
        controllers["RL blind"] = (True, blind_path)

    rows = []
    for name, (blind, model_path) in controllers.items():
        env = make_env(blind=blind, impulse=True, randomize=False, base_kp=args.base_kp, base_kd=args.base_kd)()
        wrapper = env
        while not isinstance(wrapper, PendulumRandomizationWrapper):
            wrapper = wrapper.env

        if model_path is None:

            def action_fn(obs):
                return np.zeros(3, dtype=np.float32)

        else:
            agent = Agent(env)
            agent.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
            agent.eval()

            def action_fn(obs, _agent=agent):
                with torch.no_grad():
                    t = torch.as_tensor(np.asarray(obs), dtype=torch.float32).unsqueeze(0)
                    a = _agent.actor_mean(t.reshape(1, -1))
                    return torch.clamp(a, -1, 1).squeeze(0).numpy()

        for scen_name, pm, gr in EVAL_SCENARIOS:
            rng = np.random.default_rng(70_000)
            for ep in range(args.episodes):
                pm_s = pm * (1 + rng.uniform(-0.05, 0.05))
                gr_s = gr * (1 + rng.uniform(-0.05, 0.05))
                r = eval_episode(env, action_fn, wrapper, pm_s, gr_s, seed=70_000 + ep)
                rows.append(
                    {"controller": name, "scenario": scen_name, "polemass": pm_s, "gear": gr_s, "episode": ep, **r}
                )
            sub = [x for x in rows if x["controller"] == name and x["scenario"] == scen_name]
            print(
                f"{name:12s} | {scen_name:22s} survival={np.mean([x['survival'] for x in sub]):.2f} "
                f"rms_theta={np.nanmean([x['rms_theta'] for x in sub]):.3f}"
            )
        env.close()

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "pendulum_eval_raw.csv", index=False)

    pivot = (
        df.groupby(["controller", "scenario"])
        .agg(
            survival=("survival", "mean"),
            rms_theta=("rms_theta", "mean"),
            post_impulse=("post_impulse_max_theta", "mean"),
        )
        .reset_index()
    )
    pivot.to_csv(out_dir / "pendulum_eval_summary.csv", index=False)
    print("\n", pivot.to_string(index=False))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["tune", "train", "eval"], required=True)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--blind", action="store_true", default=False)
    ap.add_argument("--timesteps", type=int, default=300_000)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--base-kp", type=float, default=3.5)
    ap.add_argument("--base-kd", type=float, default=0.4)
    args = ap.parse_args()
    return {"tune": mode_tune, "train": mode_train, "eval": mode_eval}[args.mode](args)


if __name__ == "__main__":
    raise SystemExit(main())
