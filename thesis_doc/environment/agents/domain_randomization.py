import gymnasium as gym
import mujoco
import numpy as np


class DomainRandomizationWrapper(gym.Wrapper):
    def __init__(self, env, randomization_config=None):
        super().__init__(env)
        self.config = randomization_config or {
            "mass_range": (5.0, 20.0),
            "friction_range": (0.1, 2.0),
            # Backward-compatible defaults: initial randomization only.
            "initial_randomization_enabled": True,
            "mid_episode_disturbance_enabled": False,
            "disturbance_mode": "step",  # step | time
            "disturbance_step_range": (100, 300),
            "disturbance_time_range_s": (1.0, 3.0),
            "disturbance_mass_scale_range": (0.8, 1.4),
            "disturbance_friction_scale_range": (0.4, 1.6),
            "position_patch_enabled": False,
            "patch_x_range": (1.5, 2.0),
            "patch_friction_scale": 0.35,
            "patch_visual_enabled": True,
            "patch_visual_rgba": (1.0, 0.55, 0.12, 0.7),
            "patch_visual_y_half_width": 0.35,
            "patch_visual_half_height": 0.002,
            "mass_visual_enabled": True,
            "mass_visual_low_rgba": (0.18, 0.55, 1.0, 1.0),
            "mass_visual_high_rgba": (1.0, 0.24, 0.18, 1.0),
        }
        self.unwrapped_env = env.unwrapped

        model = self.unwrapped_env.model
        self.car_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "car")
        if self.car_body_id < 0:
            self.car_body_id = 1

        self.floor_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        if self.floor_geom_id < 0:
            self.floor_geom_id = 0

        self.chassis_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "chasis")
        if self.chassis_geom_id < 0:
            self.chassis_geom_id = -1

        self.patch_visual_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "friction_patch_visual")
        if self.patch_visual_geom_id < 0:
            self.patch_visual_geom_id = -1

        self.nominal_car_mass = float(model.body_mass[self.car_body_id])
        self.nominal_car_inertia = model.body_inertia[self.car_body_id].copy()
        self.nominal_floor_friction = float(model.geom_friction[self.floor_geom_id, 0])
        self.nominal_chassis_rgba = model.geom_rgba[self.chassis_geom_id].copy() if self.chassis_geom_id >= 0 else None
        self.nominal_patch_rgba = (
            model.geom_rgba[self.patch_visual_geom_id].copy() if self.patch_visual_geom_id >= 0 else None
        )

        self.episode_base_mass = self.nominal_car_mass
        self.episode_base_friction = self.nominal_floor_friction
        self.active_mass = self.nominal_car_mass
        self.active_friction = self.nominal_floor_friction

        self._episode_step = 0
        self._disturbance_fired = False
        self._disturbance_step = None
        self._disturbance_time_s = None
        self._last_event = {
            "type": "",
            "step": -1,
            "time_s": float("nan"),
            "mass": float("nan"),
            "friction": float("nan"),
        }

    def _push_context_to_env(self) -> None:
        mass_scale = float(self.active_mass / max(self.nominal_car_mass, 1e-6))
        friction_scale = float(self.active_friction / max(self.nominal_floor_friction, 1e-6))
        if hasattr(self.unwrapped_env, "set_disturbance_context"):
            self.unwrapped_env.set_disturbance_context(mass_scale=mass_scale, friction_scale=friction_scale)

    def _cfg(self, key, default):
        return self.config.get(key, default)

    def _set_car_mass(self, new_mass: float) -> None:
        model = self.unwrapped_env.model
        new_mass = float(max(new_mass, 1e-6))
        model.body_mass[self.car_body_id] = new_mass

        # Keep shape/density roughly consistent by scaling inertia with mass.
        scale = new_mass / max(self.nominal_car_mass, 1e-6)
        model.body_inertia[self.car_body_id, :] = self.nominal_car_inertia * scale

    def _set_floor_friction(self, new_friction: float) -> None:
        model = self.unwrapped_env.model
        model.geom_friction[self.floor_geom_id, 0] = float(max(new_friction, 1e-6))

    def _set_patch_visual(self) -> None:
        if (
            self.patch_visual_geom_id < 0
            or not bool(self._cfg("patch_visual_enabled", True))
            or not bool(self._cfg("position_patch_enabled", False))
        ):
            return

        model = self.unwrapped_env.model
        x0, x1 = self._cfg("patch_x_range", (1.5, 2.0))
        x_min = float(min(x0, x1))
        x_max = float(max(x0, x1))
        center_x = 0.5 * (x_min + x_max)
        half_length = max(0.5 * (x_max - x_min), 1e-3)
        y_half_width = float(self._cfg("patch_visual_y_half_width", 0.35))
        half_height = float(self._cfg("patch_visual_half_height", 0.002))

        model.geom_pos[self.patch_visual_geom_id] = np.array([center_x, 0.0, half_height], dtype=np.float64)
        model.geom_size[self.patch_visual_geom_id] = np.array(
            [half_length, y_half_width, half_height], dtype=np.float64
        )
        model.geom_rgba[self.patch_visual_geom_id] = np.array(
            self._cfg("patch_visual_rgba", (1.0, 0.55, 0.12, 0.7)), dtype=np.float64
        )

    def _set_mass_visual(self) -> None:
        if self.chassis_geom_id < 0 or not bool(self._cfg("mass_visual_enabled", True)):
            return

        model = self.unwrapped_env.model
        mass_ratio = float(self.active_mass / max(self.nominal_car_mass, 1e-6))
        mass_ratio = float(np.clip(mass_ratio, 0.7, 1.6))
        t = (mass_ratio - 0.7) / (1.6 - 0.7)

        low = np.array(self._cfg("mass_visual_low_rgba", (0.18, 0.55, 1.0, 1.0)), dtype=np.float64)
        high = np.array(self._cfg("mass_visual_high_rgba", (1.0, 0.24, 0.18, 1.0)), dtype=np.float64)
        rgba = (1.0 - t) * low + t * high
        rgba[3] = 1.0
        model.geom_rgba[self.chassis_geom_id] = rgba

    def _update_visuals(self) -> None:
        self._set_patch_visual()
        self._set_mass_visual()

    def _sample_disturbance_trigger(self) -> None:
        mode = str(self._cfg("disturbance_mode", "step")).lower()
        rng = getattr(self.unwrapped_env, "np_random", np.random)

        self._disturbance_step = None
        self._disturbance_time_s = None

        if mode == "time":
            t0, t1 = self._cfg("disturbance_time_range_s", (1.0, 3.0))
            self._disturbance_time_s = float(rng.uniform(float(t0), float(t1)))
        else:
            s0, s1 = self._cfg("disturbance_step_range", (100, 300))
            low = int(min(s0, s1))
            high = int(max(s0, s1))
            self._disturbance_step = int(rng.integers(low, high + 1))

    def _apply_mid_episode_disturbance_if_needed(self) -> None:
        if not bool(self._cfg("mid_episode_disturbance_enabled", False)):
            return
        if self._disturbance_fired:
            return

        mode = str(self._cfg("disturbance_mode", "step")).lower()
        if mode == "time":
            should_fire = float(self.unwrapped_env.data.time) >= float(self._disturbance_time_s)
        else:
            should_fire = self._episode_step >= int(self._disturbance_step)

        if not should_fire:
            return

        rng = getattr(self.unwrapped_env, "np_random", np.random)
        m0, m1 = self._cfg("disturbance_mass_scale_range", (0.8, 1.4))
        f0, f1 = self._cfg("disturbance_friction_scale_range", (0.4, 1.6))
        mass_scale = float(rng.uniform(float(m0), float(m1)))
        friction_scale = float(rng.uniform(float(f0), float(f1)))

        self.active_mass = self.episode_base_mass * mass_scale
        self.active_friction = self.episode_base_friction * friction_scale

        self._set_car_mass(self.active_mass)
        self._set_floor_friction(self.active_friction)
        mujoco.mj_forward(self.unwrapped_env.model, self.unwrapped_env.data)
        self._update_visuals()
        self._push_context_to_env()

        self._disturbance_fired = True
        self._last_event = {
            "type": "mid_episode_disturbance",
            "step": int(self._episode_step),
            "time_s": float(self.unwrapped_env.data.time),
            "mass": float(self.active_mass),
            "friction": float(self.active_friction),
        }

    def _apply_position_patch_if_needed(self) -> None:
        if not bool(self._cfg("position_patch_enabled", False)):
            self._set_floor_friction(self.active_friction)
            return

        x = float(self.unwrapped_env.data.qpos[0])
        x0, x1 = self._cfg("patch_x_range", (1.5, 2.0))
        x_min = float(min(x0, x1))
        x_max = float(max(x0, x1))
        in_patch = x_min <= x <= x_max

        patch_scale = float(self._cfg("patch_friction_scale", 0.35))
        if in_patch:
            self._set_floor_friction(self.active_friction * patch_scale)
        else:
            self._set_floor_friction(self.active_friction)
        self._push_context_to_env()

    def reset(self, **kwargs):
        self._episode_step = 0
        self._disturbance_fired = False
        self._last_event = {
            "type": "",
            "step": -1,
            "time_s": float("nan"),
            "mass": float("nan"),
            "friction": float("nan"),
        }

        if bool(self._cfg("initial_randomization_enabled", True)):
            self.randomize_parameters()
        else:
            model = self.unwrapped_env.model
            self.episode_base_mass = float(model.body_mass[self.car_body_id])
            self.episode_base_friction = float(model.geom_friction[self.floor_geom_id, 0])
            self.active_mass = self.episode_base_mass
            self.active_friction = self.episode_base_friction

        self._sample_disturbance_trigger()
        self._update_visuals()
        self._push_context_to_env()
        return self.env.reset(**kwargs)

    def step(self, action):
        # Apply parameter changes before physics integration of this env step.
        self._apply_mid_episode_disturbance_if_needed()
        self._apply_position_patch_if_needed()
        mujoco.mj_forward(self.unwrapped_env.model, self.unwrapped_env.data)

        obs, reward, terminated, truncated, info = self.env.step(action)
        self._episode_step += 1

        info = dict(info)
        info["external_factors"] = {
            "mass": float(self.unwrapped_env.model.body_mass[self.car_body_id]),
            "friction": float(self.unwrapped_env.model.geom_friction[self.floor_geom_id, 0]),
            "disturbance_fired": bool(self._disturbance_fired),
            "disturbance_step": self._disturbance_step,
            "disturbance_time_s": self._disturbance_time_s,
            "last_event": self._last_event,
        }
        return obs, reward, terminated, truncated, info

    def randomize_parameters(self):
        model = self.unwrapped_env.model
        rng = getattr(self.unwrapped_env, "np_random", np.random)

        new_mass = float(rng.uniform(*self.config["mass_range"]))
        self._set_car_mass(new_mass)

        new_friction = float(rng.uniform(*self.config["friction_range"]))
        self._set_floor_friction(new_friction)

        self.episode_base_mass = new_mass
        self.episode_base_friction = new_friction
        self.active_mass = new_mass
        self.active_friction = new_friction

        mujoco.mj_forward(model, self.unwrapped_env.data)
        self._update_visuals()
        self._push_context_to_env()
