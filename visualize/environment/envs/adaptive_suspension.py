import gymnasium as gym
import numpy as np
import mujoco
import os
import math
from gymnasium import spaces
from gymnasium.envs.mujoco import MujocoEnv
from utils.pid import PIDController

class AdaptiveSuspensionEnv(MujocoEnv):
    metadata = {"render_modes": ["human", "rgb_array", "depth_array"], "render_fps": 50}

    def __init__(
        self,
        render_mode=None,
        target_pos: float = 5.0,
        hold_steps: int = 10,
        max_episode_steps: int = 1000,
        stop_tolerance: float = 0.05,
        gain_base_kp: float = 1.8,
        gain_base_ki: float = 0.7,
        gain_base_kd: float = 0.0,
        gain_delta_kp: float = 1.0,
        gain_delta_ki: float = 0.6,
        gain_delta_kd: float = 2.0,
        gain_range_kp: tuple[float, float] = (0.0, 6.0),
        gain_range_ki: tuple[float, float] = (0.0, 3.0),
        gain_range_kd: tuple[float, float] = (0.0, 5.0),
        terminal_hold_bonus: float = 0.0,
        terminal_hold_velocity_threshold: float = 0.05,
        action_slew_limit: float = 1.0,
        action_rate_penalty_coef: float = 0.0,
        safety_speed_governor_enabled: bool = False,
        safety_brake_margin_m: float = 0.30,
        safety_max_decel_mps2: float = 2.0,
        safety_brake_k: float = 3.5,
        safety_hard_overshoot_m: float = -1.0,
        safety_overshoot_penalty: float = 200.0,
        near_approach_zone_m: float = 0.3,
        near_approach_coef: float = 0.75,
        near_target_zone_m: float = 0.2,
        near_target_coef: float = 1.1,
        near_target_excess_thresh: float = 0.15,
        near_target_excess_coef: float = 1.0,
        approach_progress_cutoff_m: float = 0.0,
        brake_zone_vel_sq_coef: float = 0.0,
        decel_bonus_coef: float = 0.0,
        near_target_init_prob: float = 0.0,
        near_target_init_range_m: float = 0.15,
        brake_integral_reset_enabled: bool = True,
        **kwargs,
    ):
        # Locate the XML file
        xml_path = os.path.join(os.path.dirname(__file__), "assets", "car_model.xml")
        
        # Frame skip: Control interval (e.g., 100Hz for RL, 1000Hz for Physics)
        frame_skip = 10 
        
        # Initialize MuJoCo Environment
        # gymnasium > 0.26 uses __init__ differently, ensuring compatibility
        observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float64)
        
        super().__init__(model_path=xml_path, frame_skip=frame_skip, observation_space=observation_space, render_mode=render_mode, **kwargs)
        
        # Gain-scheduling action in normalized coordinates.
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        # Schedule gains around a stable PI baseline instead of spanning huge absolute ranges.
        self.base_gains = {'kp': float(gain_base_kp), 'ki': float(gain_base_ki), 'kd': float(gain_base_kd)}
        self.gain_delta = {'kp': float(gain_delta_kp), 'ki': float(gain_delta_ki), 'kd': float(gain_delta_kd)}
        self.gain_ranges = {
            'kp': (float(min(gain_range_kp)), float(max(gain_range_kp))),
            'ki': (float(min(gain_range_ki)), float(max(gain_range_ki))),
            'kd': (float(min(gain_range_kd)), float(max(gain_range_kd))),
        }
        
        # Target State (Position)
        self.target_pos = float(target_pos)
        
        # Hold-phase tracking for reward shaping
        self.hold_steps = int(hold_steps)
        self.max_episode_steps = int(max_episode_steps)
        self._current_step = 0
        self._steps_in_tolerance = 0  # Track consecutive steps within tolerance
        self._strict_hold_steps = 0  # Track consecutive strict hold steps (pos + low velocity)
        self.tolerance = float(stop_tolerance)  # Target position tolerance
        self.terminal_hold_bonus = float(terminal_hold_bonus)
        self.terminal_hold_velocity_threshold = float(terminal_hold_velocity_threshold)
        self.action_slew_limit = float(action_slew_limit)
        self.action_rate_penalty_coef = float(action_rate_penalty_coef)
        self.safety_speed_governor_enabled = bool(safety_speed_governor_enabled)
        self.safety_brake_margin_m = float(safety_brake_margin_m)
        self.safety_max_decel_mps2 = float(safety_max_decel_mps2)
        self.safety_brake_k = float(safety_brake_k)
        self.safety_hard_overshoot_m = float(safety_hard_overshoot_m)
        self.safety_overshoot_penalty = float(safety_overshoot_penalty)
        self.near_approach_zone_m = float(near_approach_zone_m)
        self.near_approach_coef = float(near_approach_coef)
        self.near_target_zone_m = float(near_target_zone_m)
        self.near_target_coef = float(near_target_coef)
        self.near_target_excess_thresh = float(near_target_excess_thresh)
        self.near_target_excess_coef = float(near_target_excess_coef)
        self.approach_progress_cutoff_m = float(approach_progress_cutoff_m)
        self.brake_zone_vel_sq_coef = float(brake_zone_vel_sq_coef)
        self.decel_bonus_coef = float(decel_bonus_coef)
        self.near_target_init_prob = float(near_target_init_prob)
        self.near_target_init_range_m = float(near_target_init_range_m)
        self.brake_integral_reset_enabled = bool(brake_integral_reset_enabled)

        # Internal PID Controller
        self.pid = PIDController(setpoint=self.target_pos, output_limits=(-1.0, 1.0))
        
        # History for Observation
        self._prev_action = np.zeros(3)
        self._drive_signs = (1.0, 1.0)
        self._last_pos = 0.0
        self._prev_vel = 0.0
        self._brake_integral_reset = False
        self._disturbance_context = np.array([1.0, 1.0], dtype=np.float64)

    def set_disturbance_context(self, mass_scale: float, friction_scale: float) -> None:
        self._disturbance_context[0] = float(mass_scale)
        self._disturbance_context[1] = float(friction_scale)

    def set_target_pos(self, target_pos: float) -> None:
        """Update target position and keep PID setpoint in sync."""
        self.target_pos = float(target_pos)
        self.pid.setpoint = self.target_pos

    def _yaw_from_quat(self, quat_wxyz):
        w, x, y, z = quat_wxyz
        return float(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))

    def _get_forward_speed(self):
        return float(self.data.qvel.flat[0])

    def _detect_drive_signs(self):
        start_qpos = self.data.qpos.copy()
        start_qvel = self.data.qvel.copy()

        yaw0 = self._yaw_from_quat(self.data.qpos[3:7])
        heading = np.array([math.cos(yaw0), math.sin(yaw0)], dtype=np.float64)

        def distance_along_heading():
            x = float(self.data.qpos[0])
            y = float(self.data.qpos[1])
            return float(x * heading[0] + y * heading[1])

        def trial(left_sign, right_sign):
            self.data.qpos[:] = start_qpos
            self.data.qvel[:] = start_qvel
            mujoco.mj_forward(self.model, self.data)

            for _ in range(20):
                self.data.ctrl[0] = left_sign * 0.2
                self.data.ctrl[1] = right_sign * 0.2
                mujoco.mj_step(self.model, self.data)

            return distance_along_heading()

        candidates = [(+1.0, +1.0), (-1.0, -1.0), (+1.0, -1.0), (-1.0, +1.0)]
        best_pair = candidates[0]
        best_progress = -np.inf

        for left_sign, right_sign in candidates:
            progress = trial(left_sign, right_sign)
            if progress > best_progress:
                best_progress = progress
                best_pair = (left_sign, right_sign)

        self.data.qpos[:] = start_qpos
        self.data.qvel[:] = start_qvel
        mujoco.mj_forward(self.model, self.data)
        self._drive_signs = best_pair

    def reset_model(self):
        # Start from a stable straight-line pose so the task matches the benchmark.
        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()

        # Near-target initialization: occasionally start within tolerance of the goal so
        # the agent experiences the hold bonus immediately, before it learns to approach.
        if self.near_target_init_prob > 0.0 and np.random.random() < self.near_target_init_prob:
            offset = np.random.uniform(0.0, self.near_target_init_range_m)
            start_x = max(0.0, self.target_pos - offset)  # within range_m before target
        else:
            start_x = 0.0

        qpos[0] = start_x
        qpos[1] = 0.0
        qpos[2] = 0.03
        qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
        qvel[:] = 0.0

        self.set_state(qpos, qvel)
        self.pid.reset()
        self._prev_action = np.zeros(3)
        self._last_pos = float(qpos[0])
        self._prev_vel = 0.0
        self._brake_integral_reset = False
        self._current_step = 0
        self._steps_in_tolerance = 0
        self._strict_hold_steps = 0
        self._disturbance_context[:] = 1.0
        self._detect_drive_signs()
        return self._get_obs()

    def _get_obs(self):
        # Get ground truth state from MuJoCo
        qpos = self.data.qpos.flat[:1] # Position (x)
        qvel = self.data.qvel.flat[:1] # Velocity (dx)
        
        error = self.target_pos - qpos[0]
        
        # Observation: [Pos, Vel, Error, Prev_Kp, Prev_Ki, Prev_Kd, MassScale, FrictionScale]
        # Including previous action helps the policy know what gains it applied recently
        return np.concatenate([
            qpos, 
            qvel, 
            [error], 
            self._prev_action,
            self._disturbance_context,
        ])

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)
        prev_action = self._prev_action.copy()

        # Limit per-step gain changes in normalized action coordinates.
        if self.action_slew_limit < 1.0:
            delta = np.clip(action - prev_action, -self.action_slew_limit, self.action_slew_limit)
            action = np.clip(prev_action + delta, -1.0, 1.0)

        action_rate_cost = float(np.sum((action - prev_action) ** 2))

        # Map normalized action to gain deltas around baseline gains.
        kp = float(np.clip(self.base_gains['kp'] + self.gain_delta['kp'] * float(action[0]), *self.gain_ranges['kp']))
        ki = float(np.clip(self.base_gains['ki'] + self.gain_delta['ki'] * float(action[1]), *self.gain_ranges['ki']))
        kd = float(np.clip(self.base_gains['kd'] + self.gain_delta['kd'] * float(action[2]), *self.gain_ranges['kd']))

        # Keep normalized action in observation so inputs stay well-conditioned.
        self._prev_action = action
        
        self.do_simulation(action, self.frame_skip)
        
        # 3. Observation & Reward
        obs = self._get_obs()
        
        # Extract state and build reward around progress + tracking quality.
        pos = float(obs[0])
        vel = float(obs[1])
        error = float(self.target_pos - pos)
        progress = pos - self._last_pos
        self._last_pos = pos

        abs_error = abs(error)
        dist_cost = abs_error
        vel_cost = vel * vel
        # Inside braking zone: zero both progress and dist_cost so the only
        # rewards are velocity penalties (negative for going fast) and the hold
        # bonus. This removes every incentive to rush and lets velocity shaping
        # dominate — the agent must decelerate to avoid accumulating vel penalties.
        if self.approach_progress_cutoff_m > 0.0 and abs_error < self.approach_progress_cutoff_m:
            progress_reward = 0.0
            dist_cost = 0.0
        else:
            progress_reward = 5.0 * progress
        overshoot = max(0.0, pos - self.target_pos)

        done = False
        reward = progress_reward - 0.75 * dist_cost - 0.12 * vel_cost - 2.0 * overshoot
        reward -= self.action_rate_penalty_coef * action_rate_cost

        # Kill integral windup once on entry to braking zone (only when enabled).
        # When disabled, RL must learn to prevent windup via gain scheduling (low Ki
        # during approach, high Kd near target) — the true adaptive control challenge.
        if self.brake_integral_reset_enabled:
            if self.approach_progress_cutoff_m > 0.0 and abs_error < self.approach_progress_cutoff_m:
                if not self._brake_integral_reset:
                    self.pid._integral = 0.0
                    self._brake_integral_reset = True

        # Encourage braking behavior as the car approaches target.
        if abs_error < self.near_approach_zone_m:
            reward -= self.near_approach_coef * abs(vel)
            if self.brake_zone_vel_sq_coef > 0.0:
                reward -= self.brake_zone_vel_sq_coef * vel * vel / max(abs_error, 0.05)
            # Dense decel bonus: immediate reward for each step the car slows down.
            # Bypasses discount-horizon problem — agent gets positive signal for braking
            # even in episodes that end at the cliff, before the hold bonus is reachable.
            if self.decel_bonus_coef > 0.0:
                vel_decrease = self._prev_vel - vel
                if vel_decrease > 0.0:
                    reward += self.decel_bonus_coef * vel_decrease

        self._prev_vel = vel

        # Increase damping pressure in the near-target band so policy learns hold behavior.
        if abs_error < self.near_target_zone_m:
            reward -= self.near_target_coef * abs(vel)
            if abs(vel) > self.near_target_excess_thresh:
                reward -= self.near_target_excess_coef * (abs(vel) - self.near_target_excess_thresh)

        # Safety shaping: strongly discourage high speed close to target.
        if self.safety_speed_governor_enabled and error > 0.0 and vel > 0.0:
            remaining = max(error, 0.0)
            effective_remaining = max(remaining - self.safety_brake_margin_m, 0.0)
            safe_v = math.sqrt(max(0.0, 2.0 * self.safety_max_decel_mps2 * effective_remaining))
            if vel > safe_v:
                reward -= 8.0 * (vel - safe_v)

        if abs_error < 0.1:
            reward += 1.0 - 1.0 * abs(vel)

        if abs_error <= self.tolerance:
            reward += 1.0
            if abs(vel) < 0.05:
                reward += 1.0

        # Track consecutive steps inside the tolerance band for the full episode.
        if abs_error < self.tolerance:
            self._steps_in_tolerance += 1
        else:
            self._steps_in_tolerance = 0

        if abs_error < self.tolerance and abs(vel) <= self.terminal_hold_velocity_threshold:
            self._strict_hold_steps += 1
        else:
            self._strict_hold_steps = 0

        # HOLD-PHASE PENALTY: In final hold_steps of episode, aggressively penalize
        # not being within tolerance to force precise settling behavior
        steps_until_end = self.max_episode_steps - self._current_step
        if steps_until_end <= self.hold_steps and (
            abs_error > self.tolerance or abs(vel) > self.terminal_hold_velocity_threshold
        ):
            hold_phase_penalty = -5.0 * abs_error - 1.0 * abs(vel)
            reward += hold_phase_penalty

        # Only allow termination after holding for hold_steps consecutive frames within tolerance.
        if self._steps_in_tolerance >= self.hold_steps:
            done = True
            reward += 80.0
            
        # Truncation / Termination
        # If car flies away
        runaway_limit = max(15.0, self.target_pos + 5.0)
        if pos > runaway_limit or pos < -2.0:
            done = True
            reward -= 100.0
        
        # Auto-truncate at max_episode_steps and increment step counter
        self._current_step += 1
        if self._current_step >= self.max_episode_steps:
            if self._strict_hold_steps >= self.hold_steps and self.terminal_hold_bonus > 0.0:
                reward += self.terminal_hold_bonus
            done = True

        # Optional hard safety constraint: terminate as failure if overshoot exceeds limit.
        overshoot_violation = False
        if self.safety_hard_overshoot_m >= 0.0:
            overshoot_violation = (pos - self.target_pos) > self.safety_hard_overshoot_m
            if overshoot_violation:
                done = True
                reward -= self.safety_overshoot_penalty

        info = {
            "gains": {"kp": kp, "ki": ki, "kd": kd},
            "state": {"pos": pos, "vel": vel},
            "action_rate_cost": action_rate_cost,
            "safety": {
                "speed_governor_enabled": self.safety_speed_governor_enabled,
                "hard_overshoot_violation": overshoot_violation,
            },
        }
        
        if self.render_mode == "human":
            self.render()

        return obs, reward, done, False, info

    def do_simulation(self, action, n_frames):
        # Override to implement PID loop inside the physics stepping.
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)
        kp = float(np.clip(self.base_gains['kp'] + self.gain_delta['kp'] * float(action[0]), *self.gain_ranges['kp']))
        ki = float(np.clip(self.base_gains['ki'] + self.gain_delta['ki'] * float(action[1]), *self.gain_ranges['ki']))
        kd = float(np.clip(self.base_gains['kd'] + self.gain_delta['kd'] * float(action[2]), *self.gain_ranges['kd']))
        
        # Set gains directly to avoid corrupting _prev_error with a dummy measurement=0 call.
        self.pid.kp = kp
        self.pid.ki = ki
        self.pid.kd = kd

        for _ in range(n_frames):
            # 1. Get current state (perfect sensor for now)
            current_pos = self.data.qpos.flat[0]
            current_vel = self._get_forward_speed()

            # In braking zone: zero integral each sub-step so windup can't re-accumulate.
            if self.brake_integral_reset_enabled and self._brake_integral_reset:
                self.pid._integral = 0.0

            # 2. Compute PID Control Signal
            # dt is model.opt.timestep
            ctrl, _ = self.pid.update(current_pos, self.model.opt.timestep)

            # Safety governor: limit approach speed by braking if stopping distance is too short.
            if self.safety_speed_governor_enabled and current_vel > 0.0:
                remaining = max(self.target_pos - current_pos, 0.0)
                effective_remaining = max(remaining - self.safety_brake_margin_m, 0.0)
                safe_v = math.sqrt(max(0.0, 2.0 * self.safety_max_decel_mps2 * effective_remaining))
                if current_vel > safe_v:
                    brake_cmd = -np.clip(self.safety_brake_k * (current_vel - safe_v), 0.0, 1.0)
                    ctrl = min(ctrl, float(brake_cmd))
            
            # 3. Apply to Actuator
            self.data.ctrl[0] = self._drive_signs[0] * ctrl
            self.data.ctrl[1] = self._drive_signs[1] * ctrl
            
            # 4. Step Physics
            mujoco.mj_step(self.model, self.data)
            
            # Render if needed (usually handled by viewer, but for offscreen...)
