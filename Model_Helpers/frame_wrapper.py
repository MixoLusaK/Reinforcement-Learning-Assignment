import numpy as np
import gymnasium as gym

class FrameStackWrapper(gym.ObservationWrapper):
    """
    Uses pre-allocated array for better memory control.
    """

    def __init__(self, env, k=4):
        super().__init__(env)
        self.k = k

        # Validate observation space
        if not isinstance(env.observation_space, gym.spaces.Box):
            raise ValueError(f"Expected Box observation space, got {type(env.observation_space)}")

        shp = env.observation_space.shape
        if len(shp) != 3:
            raise ValueError(f"Expected 3D observations (H, W, C), got shape {shp}")

        self.base_shape = shp
        print(f"[RobustFrameStack] Input shape: {shp}")

        # Pre-allocate the stacked observation array
        stacked_shape = (shp[0], shp[1], shp[2] * k)
        self._stacked_obs = np.zeros(stacked_shape, dtype=env.observation_space.dtype)

        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=stacked_shape,
            dtype=env.observation_space.dtype
        )
        print(f"[RobustFrameStack] Output shape: {self.observation_space.shape}")

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        # Validate observation shape
        assert obs.shape == self.base_shape, \
            f"Observation shape {obs.shape} doesn't match expected {self.base_shape}"

        # Fill the entire stacked observation with the initial frame
        channels_per_frame = self.base_shape[2]
        for i in range(self.k):
            start_idx = i * channels_per_frame
            end_idx = start_idx + channels_per_frame
            self._stacked_obs[..., start_idx:end_idx] = obs

        return self._stacked_obs.copy(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Shift frames: move frames [1:k] to [0:k-1], add new frame at [k-1]
        channels_per_frame = self.base_shape[2]

        # Shift existing frames left
        self._stacked_obs[..., :-channels_per_frame] = \
            self._stacked_obs[..., channels_per_frame:]

        # Add new frame at the end
        self._stacked_obs[..., -channels_per_frame:] = obs

        return self._stacked_obs.copy(), reward, terminated, truncated, info