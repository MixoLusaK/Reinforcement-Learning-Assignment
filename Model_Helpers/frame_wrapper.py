import numpy as np
from collections import deque
import gymnasium as gym


class FrameStackWrapper(gym.ObservationWrapper):
    def __init__(self, env, k=4):
        super().__init__(env)
        self.k = k
        self.frames = deque([], maxlen=k)

        # Get the original observation space
        shp = env.observation_space.shape

        # Create new observation space with stacked frames
        # Note: We use the same dtype as original observation
        self.observation_space = gym.spaces.Box(
            low=env.observation_space.low.min(),
            high=env.observation_space.high.max(),
            shape=(shp[0], shp[1], shp[2] * k),
            dtype=env.observation_space.dtype
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Initialize with the first frame repeated k times
        for _ in range(self.k):
            self.frames.append(obs)
        return self._get_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        assert len(self.frames) == self.k
        return np.concatenate(list(self.frames), axis=2)