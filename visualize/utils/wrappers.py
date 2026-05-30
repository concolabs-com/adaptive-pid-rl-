import gymnasium as gym
import numpy as np
from gymnasium import spaces


class ObservationFeatureSelectWrapper(gym.ObservationWrapper):
    """Keep only the first N observation features; used for blind-policy ablations."""

    def __init__(self, env: gym.Env, keep_dims: int):
        super().__init__(env)
        source_space = env.observation_space
        if not isinstance(source_space, spaces.Box) or len(source_space.shape) != 1:
            raise ValueError("ObservationFeatureSelectWrapper requires 1D Box observations.")
        if keep_dims <= 0 or keep_dims > int(source_space.shape[0]):
            raise ValueError(f"Invalid keep_dims={keep_dims} for source shape {source_space.shape}")
        self.keep_dims = int(keep_dims)
        self.observation_space = spaces.Box(
            low=source_space.low[: self.keep_dims],
            high=source_space.high[: self.keep_dims],
            shape=(self.keep_dims,),
            dtype=source_space.dtype,
        )

    def observation(self, observation):
        return np.asarray(observation)[: self.keep_dims]


class TargetRandomizationWrapper(gym.Wrapper):
    """Randomize target distance on reset for distance generalization training."""

    def __init__(self, env: gym.Env, target_min: float, target_max: float):
        super().__init__(env)
        self.target_min = float(target_min)
        self.target_max = float(target_max)
        if self.target_min <= 0.0 or self.target_max <= 0.0:
            raise ValueError("target_min/target_max must be positive.")
        if self.target_min > self.target_max:
            raise ValueError("target_min must be <= target_max.")

    def reset(self, **kwargs):
        rng = getattr(self.unwrapped, "np_random", np.random)
        sampled_target = float(rng.uniform(self.target_min, self.target_max))
        if hasattr(self.unwrapped, "set_target_pos"):
            self.unwrapped.set_target_pos(sampled_target)
        else:
            self.unwrapped.target_pos = sampled_target
        return self.env.reset(**kwargs)

    def set_range(self, target_min: float, target_max: float) -> None:
        self.target_min = float(target_min)
        self.target_max = float(target_max)
