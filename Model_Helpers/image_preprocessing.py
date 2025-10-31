#image_preprocessing.py
import gymnasium as gym
import numpy as np
from gymnasium import spaces


class GrayscaleNormalizeWrapper(gym.ObservationWrapper):
    """Combines grayscale conversion and normalization into one wrapper. """
    def __init__(self, env):
        super().__init__(env)

        # Get original shape
        obs_shape = self.observation_space.shape

        # New shape: (height, width, 1) with float32
        new_shape = (obs_shape[0], obs_shape[1], 1)

        # Update observation space
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=new_shape,
            dtype=np.float32
        )

    def observation(self, obs):
        """ Convert RGB observation to grayscale and normalize. """

        # Convert to grayscale using luminosity method
        grayscale = np.dot(obs[..., :3], [0.299, 0.587, 0.114])

        # Normalize to [0, 1] range
        normalized = grayscale.astype(np.float32) / 255.0

        # Add channel dimension back: (64, 64) -> (64, 64, 1)
        return np.expand_dims(normalized, axis=-1)
