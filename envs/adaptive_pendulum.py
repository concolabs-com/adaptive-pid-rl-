"""
AdaptivePendulumEnv — PID gain scheduling on the MuJoCo inverted pendulum.

Second plant for the thesis: same two-loop architecture as the car
(RL gain scheduler in the outer loop, PID in the inner loop at physics rate),
different dynamics class (unstable balance task instead of setpoint reaching).

Hidden parameters (HiP-MDP axes), randomized per episode by the experiment
script:
  - pole mass scale   (default training range 0.5 - 3.0)
  - actuator gear scale (default training range 0.5 - 1.5)

Observation (9-dim full):
  [x, x_dot, theta, theta_dot, a_kp, a_ki, a_kd, polemass_scale, gear_scale]
Blind variant keeps the first 7 dims (no context).

Action (3-dim, [-1, 1]): normalized gain deltas around base PID gains,
identical parameterisation to the car environment:
  K = clip(K_base + K_delta * a, K_range)

Inner loop: u = Kp*(-theta) + Ki*int(-theta) + Kd*d(-theta)/dt, clipped to the
actuator ctrlrange, applied at every physics sub-step (PID rate = 50 Hz,
RL rate = 25 Hz with frame_skip=2).
"""

import os

import mujoco
import numpy as np
from gymnasium import spaces
from gymnasium.envs.mujoco import MujocoEnv

from utils.pid import PIDController


class AdaptivePendulumEnv(MujocoEnv):
    metadata = {"render_modes": ["human", "rgb_array", "depth_array"], "render_fps": 25}

    def __init__(
        self,
        render_mode=None,
        max_episode_steps: int = 500,
        angle_fail_rad: float = 0.4,
        x_fail_m: float = 0.95,
        gain_base_kp: float = 3.5,
        gain_base_ki: float = 0.0,
        gain_base_kd: float = 0.4,
        gain_delta_kp: float = 3.0,
        gain_delta_ki: float = 0.05,
        gain_delta_kd: float = 0.35,
        gain_range_kp: tuple[float, float] = (0.5, 8.0),
        gain_range_ki: tuple[float, float] = (0.0, 0.3),
        gain_range_kd: tuple[float, float] = (0.0, 2.0),
        init_angle_range: float = 0.05,
        impulse_enabled: bool = False,
        impulse_step_range: tuple[int, int] = (100, 300),
        impulse_magnitude: float = 1.0,
        # Cascade position feedback: pure angle PID cannot regulate the cart
        # position (the cart drifts into the rail). A weak outer loop shapes
        # the angle setpoint: theta_ref = clip(kx*x + kxd*x_dot, +-ref_max).
        # Sign: cart at +x needs theta_ref < 0 (lean back toward the centre),
        # hence NEGATIVE gains. Part of the base controller for ALL agents
        # (fixed, not scheduled by the RL action).
        pos_feedback_kx: float = -0.05,
        pos_feedback_kxd: float = -0.10,
        theta_ref_max: float = 0.15,
        **kwargs,
    ):
        xml_path = os.path.join(os.path.dirname(__file__), "assets", "pendulum_model.xml")
        frame_skip = 2  # physics dt 0.02 s -> control dt 0.04 s (25 Hz)
        observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float64)
        super().__init__(
            model_path=xml_path,
            frame_skip=frame_skip,
            observation_space=observation_space,
            render_mode=render_mode,
            **kwargs,
        )

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        self.base_gains = {"kp": float(gain_base_kp), "ki": float(gain_base_ki), "kd": float(gain_base_kd)}
        self.gain_delta = {"kp": float(gain_delta_kp), "ki": float(gain_delta_ki), "kd": float(gain_delta_kd)}
        self.gain_ranges = {
            "kp": tuple(sorted(map(float, gain_range_kp))),
            "ki": tuple(sorted(map(float, gain_range_ki))),
            "kd": tuple(sorted(map(float, gain_range_kd))),
        }

        self.max_episode_steps = int(max_episode_steps)
        self.angle_fail_rad = float(angle_fail_rad)
        self.x_fail_m = float(x_fail_m)
        self.init_angle_range = float(init_angle_range)

        self.impulse_enabled = bool(impulse_enabled)
        self.impulse_step_range = (int(impulse_step_range[0]), int(impulse_step_range[1]))
        self.impulse_magnitude = float(impulse_magnitude)
        self.pos_feedback_kx = float(pos_feedback_kx)
        self.pos_feedback_kxd = float(pos_feedback_kxd)
        self.theta_ref_max = float(theta_ref_max)

        ctrl_lo = float(self.model.actuator_ctrlrange[0, 0])
        ctrl_hi = float(self.model.actuator_ctrlrange[0, 1])
        # PID acts on pole angle, setpoint 0; output is the actuator command.
        self.pid = PIDController(setpoint=0.0, output_limits=(ctrl_lo, ctrl_hi))

        self._prev_action = np.zeros(3)
        self._context = np.array([1.0, 1.0], dtype=np.float64)  # polemass_scale, gear_scale
        self._current_step = 0
        self._impulse_step = None
        self._impulse_done = True

    # ---- hidden-parameter interface (used by the experiment script) ----

    def set_context(self, polemass_scale: float, gear_scale: float) -> None:
        self._context[0] = float(polemass_scale)
        self._context[1] = float(gear_scale)

    # ---- gym plumbing ----

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            low=-self.init_angle_range, high=self.init_angle_range, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(low=-0.01, high=0.01, size=self.model.nv)
        self.set_state(qpos, qvel)
        self.pid.reset()
        self._prev_action = np.zeros(3)
        self._current_step = 0
        if self.impulse_enabled:
            lo, hi = self.impulse_step_range
            self._impulse_step = int(self.np_random.integers(lo, hi + 1))
            self._impulse_done = False
        else:
            self._impulse_step = None
            self._impulse_done = True
        return self._get_obs()

    def _get_obs(self):
        x = float(self.data.qpos[0])
        theta = float(self.data.qpos[1])
        xd = float(self.data.qvel[0])
        thetad = float(self.data.qvel[1])
        return np.concatenate([[x, xd, theta, thetad], self._prev_action, self._context])

    def _map_gains(self, action):
        kp = float(np.clip(self.base_gains["kp"] + self.gain_delta["kp"] * float(action[0]), *self.gain_ranges["kp"]))
        ki = float(np.clip(self.base_gains["ki"] + self.gain_delta["ki"] * float(action[1]), *self.gain_ranges["ki"]))
        kd = float(np.clip(self.base_gains["kd"] + self.gain_delta["kd"] * float(action[2]), *self.gain_ranges["kd"]))
        return kp, ki, kd

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        self._prev_action = action.astype(np.float64)
        kp, ki, kd = self._map_gains(action)
        self.pid.kp, self.pid.ki, self.pid.kd = kp, ki, kd

        # Mid-episode impulse on the pole (angular velocity kick).
        if not self._impulse_done and self._current_step >= self._impulse_step:
            kick = float(self.np_random.uniform(-self.impulse_magnitude, self.impulse_magnitude))
            self.data.qvel[1] += kick
            mujoco.mj_forward(self.model, self.data)
            self._impulse_done = True

        for _ in range(self.frame_skip):
            theta = float(self.data.qpos[1])
            # Outer loop: lean the pole back toward the rail centre so the
            # inner angle loop drags the cart home (negative kx, kxd).
            x_now = float(self.data.qpos[0])
            xd_now = float(self.data.qvel[0])
            theta_ref = float(
                np.clip(
                    self.pos_feedback_kx * x_now + self.pos_feedback_kxd * xd_now,
                    -self.theta_ref_max,
                    self.theta_ref_max,
                )
            )
            self.pid.setpoint = theta_ref
            # Inner loop PID on angle: measurement = theta.
            ctrl, _ = self.pid.update(theta, self.model.opt.timestep)
            # Positive theta (pole falling toward +x) requires pushing the cart
            # toward +x; the PID computes setpoint - measurement = -theta, so
            # invert to get the stabilising direction.
            self.data.ctrl[0] = -ctrl
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        x, xd, theta, thetad = obs[0], obs[1], obs[2], obs[3]

        self._current_step += 1
        fail = abs(theta) > self.angle_fail_rad or abs(x) > self.x_fail_m
        truncated = self._current_step >= self.max_episode_steps

        reward = 1.0 - 5.0 * theta * theta - 0.1 * x * x - 0.05 * xd * xd
        if fail:
            reward -= 20.0

        info = {
            "gains": {"kp": kp, "ki": ki, "kd": kd},
            "state": {"x": float(x), "theta": float(theta), "x_dot": float(xd), "theta_dot": float(thetad)},
            "impulse_step": self._impulse_step,
        }
        if self.render_mode == "human":
            self.render()
        return obs, reward, bool(fail), bool(truncated), info
